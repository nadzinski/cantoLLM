"""Anthropic wire-format adapter.

Consumes the engine's AsyncIterator[TokenEvent] and renders either a full
MessageResponse (non-streaming) or an async iterator of SSE strings. This is
the only layer that knows about Anthropic's format or thinking/text phases.
"""

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator

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
from cantollm.api.phase import DecodeState, phase_tagged_events
from cantollm.decoder import StopStringWatcher
from cantollm.engine.types import TokenEvent
from cantollm.stream_events import TextChunk, ThinkingEndEvent, ThinkingStartEvent

logger = logging.getLogger(__name__)

PING_INTERVAL_SECONDS = 15.0


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


def _watcher_for(stop_sequences: list[str] | None) -> StopStringWatcher | None:
    return StopStringWatcher(stop_sequences) if stop_sequences else None


def _final_stop_reason(state: DecodeState) -> StopReason | None:
    if state.stop_sequence is not None:
        return "stop_sequence"
    return _to_stop_reason(state.finish_reason)


async def _decoded_events(
    events: AsyncIterator[TokenEvent],
    tokenizer,
    state: DecodeState,
    stop_watcher: StopStringWatcher | None = None,
):
    """Adapter-local view of phase_tagged_events that drops the phase tag.

    Anthropic's renderers track `in_thinking` themselves via the
    ThinkingStartEvent/ThinkingEndEvent markers, so the phase tuple is
    discarded; the helper is here for the counter side-effects on `state`.
    """
    async for _phase, dec_evt in phase_tagged_events(
        events, tokenizer, state, stop_watcher
    ):
        yield dec_evt


async def render_message(
    events: AsyncIterator[TokenEvent],
    tokenizer,
    model_name: str,
    input_tokens: int,
    stop_sequences: list[str] | None = None,
) -> MessageResponse:
    """Drain the event stream into a single MessageResponse.

    Content is grouped into blocks by contiguous run of the same kind
    (thinking vs text), in emission order — so a text→thinking→text stream
    yields three blocks in that order, not a merged [thinking, text]. Each
    TextChunk's kind comes from `in_thinking`, which the markers flip; empty
    runs (e.g. a thinking block with no deltas) produce no block.
    """
    state = DecodeState()
    content_blocks: list[ContentBlock] = []
    open_kind: str | None = None
    buf: list[str] = []
    in_thinking = False

    def flush_segment() -> None:
        nonlocal open_kind, buf
        text = "".join(buf)
        if text:
            content_blocks.append(
                ThinkingBlock(thinking=text) if open_kind == "thinking"
                else TextBlock(text=text)
            )
        buf = []
        open_kind = None

    async for dec_evt in _decoded_events(
        events, tokenizer, state, _watcher_for(stop_sequences)
    ):
        match dec_evt:
            case ThinkingStartEvent():
                in_thinking = True
            case ThinkingEndEvent():
                in_thinking = False
            case TextChunk(text=t):
                kind = "thinking" if in_thinking else "text"
                if open_kind != kind:
                    flush_segment()
                    open_kind = kind
                buf.append(t)
    flush_segment()

    if state.error is not None:
        raise HTTPException(status_code=500, detail=state.error)

    # Anthropic requires at least one content block; fall back to empty text
    # if the model produced nothing (e.g. first token was a stop token).
    if not content_blocks:
        content_blocks.append(TextBlock(text=""))

    return MessageResponse(
        id=_new_message_id(),
        content=content_blocks,
        model=model_name,
        stop_reason=_final_stop_reason(state),
        stop_sequence=state.stop_sequence,
        usage=Usage(input_tokens=input_tokens, output_tokens=state.total),
    )


async def render_sse(
    events: AsyncIterator[TokenEvent],
    tokenizer,
    model_name: str,
    input_tokens: int,
    stop_sequences: list[str] | None = None,
) -> AsyncIterator[str]:
    """Stream the event stream as Anthropic SSE event strings.

    Runs the decode pipeline in a background task pushing strings onto a
    merge queue; a separate ping task emits keepalives during idle gaps so
    proxies don't close the stream during long thinking phases.
    """
    msg_id = _new_message_id()
    state = DecodeState()
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

            # Block framing by contiguous run: a block is opened lazily on the
            # first delta of its kind and closed when the kind changes or the
            # stream ends. This keeps indices correct for any phase order
            # (mid-text <think>, a stray </think>, or think/text/think/text),
            # unlike a fixed [thinking][text] assumption.
            open_kind: str | None = None
            block_index = -1
            in_thinking = False

            async def ensure_block(kind: str) -> None:
                nonlocal open_kind, block_index
                if open_kind == kind:
                    return
                if open_kind is not None:
                    await out.put(sse(ContentBlockStopEvent(index=block_index)))
                block_index += 1
                open_kind = kind
                block = (
                    ThinkingBlock(thinking="") if kind == "thinking"
                    else TextBlock(text="")
                )
                await out.put(sse(ContentBlockStartEvent(
                    index=block_index, content_block=block,
                )))

            async for dec_evt in _decoded_events(
                events, tokenizer, state, _watcher_for(stop_sequences)
            ):
                match dec_evt:
                    case ThinkingStartEvent():
                        in_thinking = True
                    case ThinkingEndEvent():
                        in_thinking = False
                    case TextChunk(text=t):
                        if in_thinking:
                            await ensure_block("thinking")
                            await out.put(sse(ContentBlockDeltaEvent(
                                index=block_index,
                                delta=ThinkingDelta(thinking=t),
                            )))
                        else:
                            await ensure_block("text")
                            await out.put(sse(ContentBlockDeltaEvent(
                                index=block_index,
                                delta=TextDelta(text=t),
                            )))

            if open_kind is not None:
                await out.put(sse(ContentBlockStopEvent(index=block_index)))
            elif state.error is None:
                # Zero output (e.g. first token was a stop token): emit one
                # empty text block so the stream carries >=1 content block,
                # matching the non-streaming path and Anthropic's contract.
                await out.put(sse(ContentBlockStartEvent(
                    index=0, content_block=TextBlock(text=""),
                )))
                await out.put(sse(ContentBlockStopEvent(index=0)))

            if state.error is not None:
                # Emit the error and stop — no message_delta/message_stop on
                # the error path, matching Anthropic's SSE contract.
                await out.put(sse(ErrorEvent(error=ErrorBody(message=state.error))))
                return

            await out.put(sse(MessageDeltaEvent(
                delta=MessageDeltaBody(
                    stop_reason=_final_stop_reason(state),
                    stop_sequence=state.stop_sequence,
                ),
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
        # CancelledError is expected (we just cancelled it); anything else is a
        # real fault in generation that would otherwise vanish — log it.
        try:
            await producer
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Anthropic SSE producer failed during finalization")
