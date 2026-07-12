"""IncrementalDecoder: windowed streaming detokenizer, vs the full-decode oracle.

The decoder now buffers only the current incomplete-character run instead of
every token (O(n) streaming, not O(n^2)). This suite proves the optimization
is behavior-preserving by replaying real token streams through both the
windowed decoder and a reference that re-decodes the whole sequence every
token (the previous algorithm), and asserting identical output — plus that
the window actually stays bounded.

Uses the local Qwen3-0.6B tokenizer for genuine byte-level BPE with multi-byte
characters that split across tokens; skips when the checkpoint is absent.
"""

import random
from pathlib import Path

import pytest

from cantollm.models.qwen3.tokenizer import IncrementalDecoder, Qwen3Tokenizer

TOKENIZER_PATH = (
    Path(__file__).resolve().parent.parent
    / "src/cantollm/models/model_data/Qwen3-0.6B/tokenizer.json"
)

pytestmark = pytest.mark.skipif(
    not TOKENIZER_PATH.exists(), reason="needs the local Qwen3-0.6B tokenizer.json"
)


class _FullDecodeReference:
    """The previous algorithm: re-decode the entire accumulated token list on
    every token. O(n^2), but obviously correct — the equivalence oracle."""

    def __init__(self, tokenizer):
        self._tok = tokenizer
        self._tokens: list[int] = []
        self._emitted = ""

    def add(self, token_id: int) -> str:
        self._tokens.append(token_id)
        decoded = self._tok.decode(self._tokens)
        end = len(decoded)
        while end > 0 and decoded[end - 1] == "�":
            end -= 1
        stable = decoded[:end]
        new = stable[len(self._emitted):]
        self._emitted = stable
        return new

    def flush(self) -> str:
        if not self._tokens:
            return ""
        decoded = self._tok.decode(self._tokens)
        rem = decoded[len(self._emitted):]
        self._emitted = decoded
        return rem


@pytest.fixture(scope="module")
def tokenizer():
    return Qwen3Tokenizer(
        tokenizer_file_path=str(TOKENIZER_PATH),
        is_instruct_model=True, apply_chat_template=False,
        add_generation_prompt=False, enable_thinking=True,
    )


_TEXTS = [
    "Hello, world!",
    "café résumé naïve — accents everywhere",
    "日本語のテキストと絵文字 🎉🔥🚀 mixed together",
    "emoji run 🎉🎊🥳🙌👍 back to ascii",
    "code: def f(x):\n    return x ** 2  # 日本\n",
    "zero-width​ and combining áé marks",
    "	tabs and	 mixed 中文 with 🌏 earth",
]


@pytest.mark.parametrize("text", _TEXTS)
def test_matches_full_decode_oracle(tokenizer, text):
    ids = tokenizer.encode(text, chat_wrapped=False)
    windowed = IncrementalDecoder(tokenizer)
    reference = _FullDecodeReference(tokenizer)

    win_stream, ref_stream = "", ""
    for tid in ids:
        win_stream += windowed.add(tid)
        ref_stream += reference.add(tid)
        # Streams must agree token-by-token, not just at the end.
        assert win_stream == ref_stream
    win_stream += windowed.flush()
    ref_stream += reference.flush()

    assert win_stream == ref_stream
    assert win_stream == tokenizer.decode(ids)


def test_matches_oracle_on_random_token_streams(tokenizer):
    """Arbitrary id sequences (not just well-formed text) stress every byte
    boundary, including runs that never fully stabilize."""
    rng = random.Random(20260711)
    for _ in range(200):
        ids = [rng.randrange(0, 151_000) for _ in range(rng.randint(1, 40))]
        windowed = IncrementalDecoder(tokenizer)
        reference = _FullDecodeReference(tokenizer)
        win, ref = "", ""
        for tid in ids:
            win += windowed.add(tid)
            ref += reference.add(tid)
        win += windowed.flush()
        ref += reference.flush()
        assert win == ref, f"diverged on ids={ids}"


def test_window_stays_bounded(tokenizer):
    """The whole point of the rewrite: the buffer must not grow with the
    generation length. A long ASCII stream should keep the window tiny."""
    ids = tokenizer.encode("word " * 500, chat_wrapped=False)
    dec = IncrementalDecoder(tokenizer)
    max_window = 0
    for tid in ids:
        dec.add(tid)
        max_window = max(max_window, len(dec._window))
    # ASCII resets every token; even allowing for the odd multi-token char,
    # the window never approaches the ~1000-token stream length.
    assert max_window <= 8, max_window
