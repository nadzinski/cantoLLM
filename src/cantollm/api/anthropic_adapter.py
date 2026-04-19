"""Anthropic wire-format adapter.

Consumes the engine's AsyncIterator[TokenEvent] and renders either a full
MessageResponse (non-streaming) or an async iterator of SSE strings. This is
the only layer that knows about Anthropic's format or thinking/text phases.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

from fastapi import HTTPException

from cantollm.api.anthropic_types import (
    ContentBlock,
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    ErrorBody,
    ErrorEvent,
    MessageDeltaBody,
    MessageDeltaEvent,
    MessageResponse,
    MessageStartEvent,
    MessageStartSnapshot,
    MessageStopEvent,
    StopReason,
    StreamUsage,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    Usage,
    sse,
)
from cantollm.decoder import StreamingDecoder
from cantollm.engine.types import TokenEvent
from cantollm.stream_events import TextChunk, ThinkingEndEvent, ThinkingStartEvent

PING_INTERVAL_SECONDS = 15.0


@dataclass
class _DecodeState:
    thinking: int = 0
    text: int = 0
    total: int = 0
    finish_reason: str | None = None
    error: str | None = None


def _to_stop_reason(finish_reason: str | None) -> StopReason | None:
    """Map engine FinishReason to Anthropic's StopReason literal.

    `abort` (client disconnect) has no Anthropic equivalent; callers reading
    the terminal events on that path are already in cleanup anyway.
    """
    if finish_reason in ("end_turn", "max_tokens"):
        return finish_reason  # type: ignore[return-value]
    return None


def _new_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def _classify(token_id: int, tokenizer, phase_is_thinking: bool) -> tuple[bool, str]:
    """Return (new_phase_is_thinking, bucket) where bucket is 'thinking' or 'text'."""
    if token_id == tokenizer.thinking_start_id:
        return True, "thinking"
    if token_id == tokenizer.thinking_end_id:
        return False, "thinking"
    return phase_is_thinking, "thinking" if phase_is_thinking else "text"


async def _decoded_events(
    events: AsyncIterator[TokenEvent],
    tokenizer,
    state: _DecodeState,
):
    """Drive StreamingDecoder from an async TokenEvent source, tracking counts.

    Terminal events (finish_reason or error) end the stream and record their
    reason on `state` so the caller can emit the right wire-level closer.
    """
    decoder = StreamingDecoder(tokenizer)
    phase_is_thinking = False

    async for evt in events:
        if evt.error is not None:
            state.error = evt.error
            break
        if evt.finish_reason is not None:
            state.finish_reason = evt.finish_reason
            break
        if evt.token_id is None:
            continue
        phase_is_thinking, bucket = _classify(evt.token_id, tokenizer, phase_is_thinking)
        setattr(state, bucket, getattr(state, bucket) + 1)
        state.total += 1
        for dec_evt in decoder.process(evt.token_id):
            yield dec_evt

    for dec_evt in decoder.flush():
        yield dec_evt


async def render_message(
    events: AsyncIterator[TokenEvent],
    tokenizer,
    model_name: str,
    input_tokens: int,
) -> MessageResponse:
    """Drain the event stream into a single MessageResponse."""
    state = _DecodeState()
    content_blocks: list[ContentBlock] = []
    current_text: list[str] = []
    current_thinking: list[str] = []
    in_thinking = False

    async for dec_evt in _decoded_events(events, tokenizer, state):
        match dec_evt:
            case ThinkingStartEvent():
                in_thinking = True
            case ThinkingEndEvent():
                thinking_text = "".join(current_thinking)
                if thinking_text:
                    content_blocks.append(ThinkingBlock(thinking=thinking_text))
                current_thinking = []
                in_thinking = False
            case TextChunk(text=t):
                (current_thinking if in_thinking else current_text).append(t)

    if state.error is not None:
        raise HTTPException(status_code=500, detail=state.error)

    text = "".join(current_text)
    if text:
        content_blocks.append(TextBlock(text=text))

    # Anthropic requires at least one content block; fall back to empty text
    # if the model produced nothing (e.g. first token was a stop token).
    if not content_blocks:
        content_blocks.append(TextBlock(text=""))

    return MessageResponse(
        id=_new_message_id(),
        content=content_blocks,
        model=model_name,
        stop_reason=_to_stop_reason(state.finish_reason),
        usage=Usage(input_tokens=input_tokens, output_tokens=state.total),
    )


async def render_sse(
    events: AsyncIterator[TokenEvent],
    tokenizer,
    model_name: str,
    input_tokens: int,
) -> AsyncIterator[str]:
    """Stream the event stream as Anthropic SSE event strings.

    Runs the decode pipeline in a background task pushing strings onto a
    merge queue; a separate ping task emits keepalives during idle gaps so
    proxies don't close the stream during long thinking phases.
    """
    msg_id = _new_message_id()
    state = _DecodeState()
    out: asyncio.Queue[str | None] = asyncio.Queue()

    async def produce():
        try:
            await out.put(sse(MessageStartEvent(
                message=MessageStartSnapshot(
                    id=msg_id,
                    model=model_name,
                    usage=Usage(input_tokens=input_tokens, output_tokens=0),
                ),
            )))

            block_index = 0
            in_thinking = False
            started_text_block = False

            async for dec_evt in _decoded_events(events, tokenizer, state):
                match dec_evt:
                    case ThinkingStartEvent():
                        in_thinking = True
                        await out.put(sse(ContentBlockStartEvent(
                            index=block_index,
                            content_block=ThinkingBlock(thinking=""),
                        )))
                    case ThinkingEndEvent():
                        in_thinking = False
                        await out.put(sse(ContentBlockStopEvent(index=block_index)))
                        block_index += 1
                    case TextChunk(text=t):
                        if in_thinking:
                            await out.put(sse(ContentBlockDeltaEvent(
                                index=block_index,
                                delta=ThinkingDelta(thinking=t),
                            )))
                        else:
                            if not started_text_block:
                                await out.put(sse(ContentBlockStartEvent(
                                    index=block_index,
                                    content_block=TextBlock(text=""),
                                )))
                                started_text_block = True
                            await out.put(sse(ContentBlockDeltaEvent(
                                index=block_index,
                                delta=TextDelta(text=t),
                            )))

            if started_text_block:
                await out.put(sse(ContentBlockStopEvent(index=block_index)))

            if state.error is not None:
                # Emit the error and stop — no message_delta/message_stop on
                # the error path, matching Anthropic's SSE contract.
                await out.put(sse(ErrorEvent(error=ErrorBody(message=state.error))))
                return

            await out.put(sse(MessageDeltaEvent(
                delta=MessageDeltaBody(stop_reason=_to_stop_reason(state.finish_reason)),
                usage=StreamUsage(
                    output_tokens=state.total,
                    thinking_tokens=state.thinking,
                    text_tokens=state.text,
                ),
            )))
            await out.put(sse(MessageStopEvent()))
        finally:
            await out.put(None)

    async def ping():
        while True:
            await asyncio.sleep(PING_INTERVAL_SECONDS)
            await out.put("event: ping\ndata: {}\n\n")

    producer = asyncio.create_task(produce())
    pinger = asyncio.create_task(ping())

    try:
        while (chunk := await out.get()) is not None:
            yield chunk
    finally:
        pinger.cancel()
        if not producer.done():
            producer.cancel()
        # Let the producer finalize (e.g. abort the engine via its finally block).
        try:
            await producer
        except BaseException:
            pass
