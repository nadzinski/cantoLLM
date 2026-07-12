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
from dataclasses import dataclass, field
from typing import Literal

from cantollm.decoder import StopStringWatcher, StreamingDecoder
from cantollm.engine.types import TokenEvent
from cantollm.stream_events import StreamEvent, TextChunk, ThinkingEndEvent, ThinkingStartEvent

Phase = Literal["thinking", "text"]


def logprobs_for_emitted(
    entries: list[tuple[str, float | None]], emitted_len: int
) -> list[tuple[str, float | None]]:
    """Keep only the content-logprob entries that fall within emitted content.

    `content_logprobs` is recorded per text-phase token, before the stop
    watcher decides how much text is actually emitted — so when a stop
    sequence matches, the tokens forming it (excluded from `content`) still
    have entries. Drop every entry that begins at or past `emitted_len`; a
    token straddling the boundary contributed emitted characters and is kept.
    Alignment is approximate at multi-byte boundaries, as the docstring on
    `DecodeState.content_logprobs` notes."""
    kept: list[tuple[str, float | None]] = []
    consumed = 0
    for text, lp in entries:
        if consumed >= emitted_len:
            break
        kept.append((text, lp))
        consumed += len(text)
    return kept


@dataclass
class DecodeState:
    thinking: int = 0
    text: int = 0
    total: int = 0
    finish_reason: str | None = None
    error: str | None = None
    stop_sequence: str | None = None
    """The stop string that ended the stream, when one matched. The wire
    stop reason then comes from this, not from `finish_reason` (the engine
    reports "abort" — it was told to stop, it doesn't know why)."""
    content_logprobs: list[tuple[str, float | None]] = field(default_factory=list)
    """(token_text, logprob) per text-phase token, in emission order. The
    token text is the piece the incremental decoder released for that token
    — "" for tokens whose bytes are still held back (multi-byte UTF-8), with
    the completing token carrying the combined text. Counts stay 1:1 with
    tokens; texts are approximate at byte boundaries."""


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
    stop_watcher: StopStringWatcher | None = None,
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

    `stop_watcher` scans visible-text chunks only (thinking passes through):
    emitted text is filtered through its holdback so no partial stop string
    ever escapes; on a match the matched string lands on
    `state.stop_sequence` and the engine's stream is closed — which is the
    disconnect→abort path, so generation actually stops and the slot frees.
    With an active watcher, streaming per-chunk logprob attribution becomes
    approximate (held-back text lags its token); `state.content_logprobs`
    stays exact.
    """
    decoder = StreamingDecoder(tokenizer)
    phase_is_thinking = False

    def scan_text(chunk: TextChunk) -> TextChunk | None:
        if stop_watcher is None:
            return chunk
        released = stop_watcher.feed(chunk.text)
        return TextChunk(released) if released else None

    async for evt in events:
        if evt.error is not None:
            state.error = evt.error
            break
        if evt.finish_reason is not None:
            state.finish_reason = evt.finish_reason
            break
        if evt.token_id is None:
            continue
        # Phase *before* this token classifies it. A marker token's only
        # TextChunk is the outgoing phase's held bytes flushed at the boundary
        # (see StreamingDecoder._release_held), so those chunks belong to
        # `was_thinking`; for content tokens `was_thinking == phase_is_thinking`.
        was_thinking = phase_is_thinking
        phase_is_thinking, bucket = _classify(evt.token_id, tokenizer, phase_is_thinking)
        setattr(state, bucket, getattr(state, bucket) + 1)
        state.total += 1
        dec_events = list(decoder.process(evt.token_id))
        if bucket == "text":
            piece = "".join(
                d.text for d in dec_events if isinstance(d, TextChunk)
            )
            state.content_logprobs.append((piece, evt.logprob))
        for dec_evt in dec_events:
            match dec_evt:
                case ThinkingStartEvent():
                    yield "thinking", dec_evt
                case ThinkingEndEvent():
                    yield "thinking", dec_evt
                case TextChunk() if was_thinking:
                    yield "thinking", dec_evt
                case TextChunk():
                    scanned = scan_text(dec_evt)
                    if scanned is not None:
                        yield "text", scanned
                    if stop_watcher is not None and stop_watcher.matched is not None:
                        state.stop_sequence = stop_watcher.matched
                        # Close the engine's stream deterministically: this
                        # is submit()'s finally → abort → slot freed.
                        await events.aclose()
                        return

    for dec_evt in decoder.flush():
        match dec_evt:
            case ThinkingEndEvent():
                yield "thinking", dec_evt
            case TextChunk() if phase_is_thinking:
                yield "thinking", dec_evt
            case TextChunk():
                scanned = scan_text(dec_evt)
                if scanned is not None:
                    yield "text", scanned

    if stop_watcher is not None:
        if stop_watcher.matched is not None:
            # A match completed by the decoder's own flush (held UTF-8 bytes).
            state.stop_sequence = stop_watcher.matched
        else:
            # No match came: release the watcher's held-back tail.
            tail = stop_watcher.flush()
            if tail:
                yield "text", TextChunk(tail)
