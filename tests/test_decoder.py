"""StreamingDecoder: phase framing and the thinking→text byte boundary.

Focus: multi-byte UTF-8 bytes held back by the incremental decoder must not
leak across a thinking/text phase boundary. A marker token (<think>/</think>)
separates the phases in the real token stream, so held bytes belong to the
outgoing phase and must be released there — not carried into the next phase's
first token.
"""

from cantollm.decoder import StreamingDecoder
from cantollm.models.qwen3.tokenizer import IncrementalDecoder
from cantollm.stream_events import TextChunk, ThinkingEndEvent, ThinkingStartEvent

THINK_START = 90001
THINK_END = 90002


class _ByteTokenizer:
    """Tokenizer stub whose decode() concatenates each id's bytes and UTF-8
    decodes with replacement — so a char split across ids is only stable once
    every byte has arrived, exactly like the real BPE decoder. Drives the real
    IncrementalDecoder (the code under test's holdback logic)."""

    thinking_start_id = THINK_START
    thinking_end_id = THINK_END

    def __init__(self, id_to_bytes: dict[int, bytes]):
        self._bytes = id_to_bytes

    def decode(self, ids: list[int]) -> str:
        raw = b"".join(self._bytes[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    def incremental_decoder(self) -> IncrementalDecoder:
        return IncrementalDecoder(self)


def _drive(tokenizer, token_ids):
    """Run the token stream through a StreamingDecoder; return the flat event
    list plus, for each TextChunk, whether it came before/after ThinkingEnd."""
    dec = StreamingDecoder(tokenizer)
    events = []
    for tid in token_ids:
        events.extend(dec.process(tid))
    events.extend(dec.flush())
    return events


def test_multibyte_char_does_not_leak_across_thinking_boundary():
    # 'é' is 0xC3 0xA9. Split it across the boundary: the first byte is the
    # last thinking token, the second byte is the first text token — the
    # pathological case where the incremental decoder is mid-character exactly
    # at the phase switch.
    tok = _ByteTokenizer({
        THINK_START: b"",
        THINK_END: b"",
        1: "reason".encode(),   # thinking text
        2: b"\xc3",             # first byte of 'é', last thinking token
        3: b"\xa9",             # second byte of 'é', first *text* token
        4: b"X",                # visible text
    })
    events = _drive(tok, [THINK_START, 1, 2, THINK_END, 3, 4])

    # Partition text by phase using the ThinkingEnd marker as the divider.
    end_idx = next(i for i, e in enumerate(events) if isinstance(e, ThinkingEndEvent))
    text_after = "".join(
        e.text for e in events[end_idx + 1:] if isinstance(e, TextChunk)
    )
    thinking_text = "".join(
        e.text for e in events[:end_idx] if isinstance(e, TextChunk)
    )

    # The é's leading byte belonged to the thinking phase; it must NOT surface
    # inside the visible text. (It shows as U+FFFD in thinking — a broken char,
    # correct given the byte was split across the boundary — but never leaks.)
    assert "é" not in text_after
    assert "X" in text_after
    assert "reason" in thinking_text


def test_multibyte_within_a_phase_still_completes():
    # No boundary split: 'é' fully inside the text phase must decode normally
    # (the fix must not break ordinary multi-byte holdback).
    tok = _ByteTokenizer({
        THINK_END: b"",
        10: b"\xc3",  # first byte of 'é'
        11: b"\xa9",  # completes 'é'
        12: b"!",
    })
    events = _drive(tok, [10, 11, 12])
    text = "".join(e.text for e in events if isinstance(e, TextChunk))
    assert text == "é!"


def test_benign_thinking_then_text_phases():
    tok = _ByteTokenizer({
        THINK_START: b"", THINK_END: b"",
        1: b"hi", 2: b"there",
    })
    events = _drive(tok, [THINK_START, 1, THINK_END, 2])
    kinds = [type(e).__name__ for e in events]
    assert kinds[0] == "ThinkingStartEvent"
    assert "ThinkingEndEvent" in kinds
    thinking = "".join(
        e.text for e in events[:kinds.index("ThinkingEndEvent")]
        if isinstance(e, TextChunk)
    )
    text = "".join(
        e.text for e in events[kinds.index("ThinkingEndEvent") + 1:]
        if isinstance(e, TextChunk)
    )
    assert thinking == "hi"
    assert text == "there"
