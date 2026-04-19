"""OpenAI Chat Completions wire-format adapter.

Mirrors `anthropic_adapter.py`: consumes the engine's
`AsyncIterator[TokenEvent]` and renders either a full `ChatCompletion`
(non-streaming) or an async iterator of SSE strings. The model's thinking
phase maps to the DeepSeek-R1 `reasoning_content` field on assistant
messages / deltas so clients that care can surface it, while clients that
don't simply ignore the unknown field.
"""

import asyncio
import json
import uuid
from collections.abc import AsyncIterator

from fastapi import HTTPException

from cantollm.api.openai_types import (
    ChatCompletion,
    ChatCompletionChoice,
    ChatCompletionMessage,
    CompletionTokensDetails,
    CompletionUsage,
    FinishReason,
    OpenAIError,
    OpenAIErrorEnvelope,
)
from cantollm.api.phase import DecodeState, phase_tagged_events
from cantollm.engine.types import TokenEvent
from cantollm.stream_events import TextChunk


def _to_finish_reason(finish_reason: str | None) -> FinishReason | None:
    """Map engine FinishReason to OpenAI's finish_reason literal.

    `abort` (client disconnect) has no OpenAI equivalent — matches the spec's
    silence on that path.
    """
    if finish_reason == "end_turn":
        return "stop"
    if finish_reason == "max_tokens":
        return "length"
    return None


def _new_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _chunk_line(
    *,
    completion_id: str,
    created: int,
    model_name: str,
    delta: dict,
    finish_reason: FinishReason | None = None,
    usage: dict | None = None,
) -> str:
    """Format one OpenAI chunk as a single SSE `data:` line.

    Built from a dict rather than via Pydantic serialization so we can
    emit exactly what OpenAI SDKs expect: `finish_reason: null` and
    `logprobs: null` are always present on choices, while inside `delta`
    we omit fields we don't have content for (so a content delta doesn't
    carry a `reasoning_content: null` and vice versa).

    OpenAI SSE framing has no `event:` lines — one JSON blob on `data:`,
    blank-line terminated. Stream ends with `data: [DONE]\\n\\n`.
    """
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
            "logprobs": None,
        }],
    }
    if usage is not None:
        chunk["usage"] = usage
    return f"data: {json.dumps(chunk)}\n\n"


def _usage_only_chunk_line(
    *, completion_id: str, created: int, model_name: str, usage: dict,
) -> str:
    """Final usage-only chunk (empty choices array) when
    `stream_options.include_usage=true`."""
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [],
        "usage": usage,
    }
    return f"data: {json.dumps(chunk)}\n\n"


def _error_line(message: str, err_type: str = "server_error") -> str:
    envelope = OpenAIErrorEnvelope(error=OpenAIError(message=message, type=err_type))
    return f"data: {envelope.model_dump_json()}\n\n"


def _build_usage_dict(state: DecodeState, input_tokens: int) -> dict:
    usage = {
        "prompt_tokens": input_tokens,
        "completion_tokens": state.text,
        "total_tokens": input_tokens + state.text,
    }
    if state.thinking:
        usage["completion_tokens_details"] = {
            "reasoning_tokens": state.thinking,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0,
        }
    return usage


async def render_chat_completion(
    events: AsyncIterator[TokenEvent],
    tokenizer,
    model_name: str,
    input_tokens: int,
    completion_id: str,
    created: int,
) -> ChatCompletion:
    """Drain the event stream into a single ChatCompletion."""
    state = DecodeState()
    visible: list[str] = []
    reasoning: list[str] = []

    async for phase, dec_evt in phase_tagged_events(events, tokenizer, state):
        if not isinstance(dec_evt, TextChunk):
            # ThinkingStart/End markers are internal framing — OpenAI has no
            # wire concept for them.
            continue
        (reasoning if phase == "thinking" else visible).append(dec_evt.text)

    if state.error is not None:
        raise HTTPException(status_code=500, detail=state.error)

    content = "".join(visible) or None
    reasoning_content = "".join(reasoning) or None

    usage = CompletionUsage(
        prompt_tokens=input_tokens,
        completion_tokens=state.text,
        total_tokens=input_tokens + state.text,
        completion_tokens_details=CompletionTokensDetails(
            reasoning_tokens=state.thinking,
        ) if state.thinking else None,
    )

    return ChatCompletion(
        id=completion_id,
        created=created,
        model=model_name,
        choices=[ChatCompletionChoice(
            index=0,
            message=ChatCompletionMessage(
                content=content,
                reasoning_content=reasoning_content,
            ),
            finish_reason=_to_finish_reason(state.finish_reason),
        )],
        usage=usage,
    )


async def render_chat_completion_sse(
    events: AsyncIterator[TokenEvent],
    tokenizer,
    model_name: str,
    input_tokens: int,
    completion_id: str,
    created: int,
    include_usage: bool,
) -> AsyncIterator[str]:
    """Stream the event stream as OpenAI SSE chunks.

    Lifecycle:
    1. One opening chunk with `{role: "assistant"}` only.
    2. One chunk per text delta, routed to `content` (text phase) or
       `reasoning_content` (thinking phase).
    3. One closing chunk with `finish_reason` set and an empty delta.
    4. If `include_usage`, one usage-only chunk (empty `choices`, populated
       `usage`).
    5. `data: [DONE]\\n\\n` sentinel.

    No keepalives — OpenAI's protocol doesn't use them and standard HTTP
    proxies typically tolerate the visible-token cadence once decoding
    starts. A long leading thinking-phase gap could still hit a proxy
    timeout; that's a known tradeoff of the dialect.
    """
    state = DecodeState()
    out: asyncio.Queue[str | None] = asyncio.Queue()

    def emit(delta: dict, finish_reason: FinishReason | None = None) -> str:
        return _chunk_line(
            completion_id=completion_id, created=created, model_name=model_name,
            delta=delta, finish_reason=finish_reason,
        )

    async def produce():
        try:
            # Opening chunk: role only.
            await out.put(emit({"role": "assistant"}))

            async for phase, dec_evt in phase_tagged_events(events, tokenizer, state):
                if not isinstance(dec_evt, TextChunk):
                    continue
                key = "reasoning_content" if phase == "thinking" else "content"
                await out.put(emit({key: dec_evt.text}))

            if state.error is not None:
                # Minimum-viable mid-stream error: one error envelope as a
                # data chunk, then [DONE]. Proper parity with Anthropic's
                # SSE contract is a separate Phase 1b bullet.
                await out.put(_error_line(state.error))
                await out.put("data: [DONE]\n\n")
                return

            # Closing chunk: empty delta + finish_reason.
            await out.put(emit({}, finish_reason=_to_finish_reason(state.finish_reason)))

            if include_usage:
                await out.put(_usage_only_chunk_line(
                    completion_id=completion_id, created=created, model_name=model_name,
                    usage=_build_usage_dict(state, input_tokens),
                ))

            await out.put("data: [DONE]\n\n")
        finally:
            await out.put(None)

    producer = asyncio.create_task(produce())

    try:
        while (chunk := await out.get()) is not None:
            yield chunk
    finally:
        if not producer.done():
            producer.cancel()
        # Let the producer finalize (e.g. abort the engine via its finally block).
        try:
            await producer
        except BaseException:
            pass
