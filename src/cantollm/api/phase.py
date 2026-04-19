"""Shared thinking/text phase classifier for API adapters.

Engine `TokenEvent`s carry raw token IDs plus optional terminal markers. A
`StreamingDecoder` turns tokens into `StreamEvent`s (TextChunk / thinking
markers), and each visible text chunk belongs to either the model's thinking
phase or its visible-text phase. Both the Anthropic and OpenAI adapters want
the same classification — Anthropic uses the phase to wrap content blocks,
OpenAI uses it to route deltas into `content` vs `reasoning_content`.

This module centralizes the classifier and token counting so adapters stay
focused on wire format.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

from cantollm.decoder import StreamingDecoder
from cantollm.engine.types import TokenEvent
from cantollm.stream_events import StreamEvent, TextChunk, ThinkingEndEvent, ThinkingStartEvent

Phase = Literal["thinking", "text"]


@dataclass
class DecodeState:
    thinking: int = 0
    text: int = 0
    total: int = 0
    finish_reason: str | None = None
    error: str | None = None


def _classify(token_id: int, tokenizer, phase_is_thinking: bool) -> tuple[bool, Phase]:
    """Return (new_phase_is_thinking, bucket) where bucket is 'thinking' or 'text'."""
    if token_id == tokenizer.thinking_start_id:
        return True, "thinking"
    if token_id == tokenizer.thinking_end_id:
        return False, "thinking"
    return phase_is_thinking, "thinking" if phase_is_thinking else "text"


async def phase_tagged_events(
    events: AsyncIterator[TokenEvent],
    tokenizer,
    state: DecodeState,
) -> AsyncIterator[tuple[Phase, StreamEvent]]:
    """Drive StreamingDecoder from an async TokenEvent source, tagging each
    emitted event with the phase it belongs to.

    The boundary tokens themselves (thinking_start / thinking_end) are
    counted into the `thinking` bucket, matching the adapter's historical
    accounting. Terminal events (finish_reason or error) end the stream and
    record their reason on `state` so the caller can emit the right
    wire-level closer.

    For ThinkingStartEvent the yielded phase reflects the new (thinking)
    phase so callers can branch consistently; for ThinkingEndEvent the
    yielded phase reflects the outgoing thinking phase (the next TextChunk
    will be tagged 'text').
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
            match dec_evt:
                case ThinkingStartEvent():
                    yield "thinking", dec_evt
                case ThinkingEndEvent():
                    yield "thinking", dec_evt
                case TextChunk():
                    yield ("thinking" if phase_is_thinking else "text"), dec_evt

    for dec_evt in decoder.flush():
        match dec_evt:
            case ThinkingEndEvent():
                yield "thinking", dec_evt
            case TextChunk():
                yield ("thinking" if phase_is_thinking else "text"), dec_evt
