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
import logging
import uuid
from collections.abc import AsyncIterator

from fastapi import HTTPException

from cantollm.api.openai_types import (
    ChatCompletion,
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChoiceLogprobs,
    CompletionTokensDetails,
    CompletionUsage,
    FinishReason,
    OpenAIError,
    OpenAIErrorEnvelope,
    TokenLogprob,
)
from cantollm.api.phase import DecodeState, phase_tagged_events
from cantollm.decoder import StopStringWatcher
from cantollm.engine.types import TokenEvent
from cantollm.stream_events import TextChunk

logger = logging.getLogger(__name__)


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


def _final_finish_reason(state: DecodeState) -> FinishReason | None:
    # A matched stop string is a plain "stop" on this dialect (OpenAI does
    # not report which sequence matched).
    if state.stop_sequence is not None:
        return "stop"
    return _to_finish_reason(state.finish_reason)


def _watcher_for(stop: str | list[str] | None) -> StopStringWatcher | None:
    if not stop:
        return None
    return StopStringWatcher([stop] if isinstance(stop, str) else list(stop))


def _new_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _token_logprob_entry(token_text: str, logprob: float) -> dict:
    return {
        "token": token_text,
        "logprob": logprob,
        "bytes": list(token_text.encode("utf-8")) or None,
        "top_logprobs": [],
    }


def _chunk_line(
    *,
    completion_id: str,
    created: int,
    model_name: str,
    delta: dict,
    finish_reason: FinishReason | None = None,
    usage: dict | None = None,
    logprobs: dict | None = None,
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
            "logprobs": logprobs,
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
    # completion_tokens is the full generated count (text + reasoning); the
    # reasoning share is broken out under completion_tokens_details, a subset
    # of it — matching OpenAI's spec.
    completion_tokens = state.text + state.thinking
    usage = {
        "prompt_tokens": input_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": input_tokens + completion_tokens,
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
    logprobs_requested: bool = False,
    stop: str | list[str] | None = None,
) -> ChatCompletion:
    """Drain the event stream into a single ChatCompletion."""
    state = DecodeState()
    visible: list[str] = []
    reasoning: list[str] = []

    async for phase, dec_evt in phase_tagged_events(
        events, tokenizer, state, _watcher_for(stop)
    ):
        if not isinstance(dec_evt, TextChunk):
            # ThinkingStart/End markers are internal framing — OpenAI has no
            # wire concept for them.
            continue
        (reasoning if phase == "thinking" else visible).append(dec_evt.text)

    if state.error is not None:
        raise HTTPException(status_code=500, detail=state.error)

    content = "".join(visible) or None
    reasoning_content = "".join(reasoning) or None

    # completion_tokens includes reasoning tokens; reasoning_tokens is a
    # subset breakdown of it (OpenAI spec).
    completion_tokens = state.text + state.thinking
    usage = CompletionUsage(
        prompt_tokens=input_tokens,
        completion_tokens=completion_tokens,
        total_tokens=input_tokens + completion_tokens,
        completion_tokens_details=CompletionTokensDetails(
            reasoning_tokens=state.thinking,
        ) if state.thinking else None,
    )

    choice_logprobs = None
    if logprobs_requested:
        # Entries whose engine event carried no logprob are dropped —
        # OpenAI's schema requires a float per entry.
        choice_logprobs = ChoiceLogprobs(content=[
            TokenLogprob(**_token_logprob_entry(text, lp))
            for text, lp in state.content_logprobs
            if lp is not None
        ])

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
            finish_reason=_final_finish_reason(state),
            logprobs=choice_logprobs,
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
    logprobs_requested: bool = False,
    stop: str | list[str] | None = None,
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

    def emit(
        delta: dict,
        finish_reason: FinishReason | None = None,
        logprobs: dict | None = None,
    ) -> str:
        return _chunk_line(
            completion_id=completion_id, created=created, model_name=model_name,
            delta=delta, finish_reason=finish_reason, logprobs=logprobs,
        )

    async def produce():
        try:
            # Opening chunk: role only.
            await out.put(emit({"role": "assistant"}))

            async for phase, dec_evt in phase_tagged_events(
                events, tokenizer, state, _watcher_for(stop)
            ):
                if not isinstance(dec_evt, TextChunk):
                    continue
                key = "reasoning_content" if phase == "thinking" else "content"
                chunk_logprobs = None
                if logprobs_requested and phase == "text" and state.content_logprobs:
                    # Lockstep with phase_tagged_events: the entry for the
                    # token that produced this chunk was just appended.
                    text, lp = state.content_logprobs[-1]
                    if lp is not None:
                        chunk_logprobs = {
                            "content": [_token_logprob_entry(text, lp)]
                        }
                await out.put(emit({key: dec_evt.text}, logprobs=chunk_logprobs))

            if state.error is not None:
                # Minimum-viable mid-stream error: one error envelope as a
                # data chunk, then [DONE]. Proper parity with Anthropic's
                # SSE contract is a separate Phase 1b bullet.
                await out.put(_error_line(state.error))
                await out.put("data: [DONE]\n\n")
                return

            # Closing chunk: empty delta + finish_reason.
            await out.put(emit({}, finish_reason=_final_finish_reason(state)))

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
        # CancelledError is expected (we just cancelled it); anything else is a
        # real fault in generation that would otherwise vanish — log it.
        try:
            await producer
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("OpenAI SSE producer failed during finalization")
