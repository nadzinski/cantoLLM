"""Tests for the Qwen3Tokenizer and IncrementalDecoder.

Since we can't download the real tokenizer.json in CI, we mock the
underlying HuggingFace Tokenizer to test our wrapping logic in isolation.
"""

from unittest.mock import MagicMock, patch

import pytest

from cantollm.models.qwen3.tokenizer import IncrementalDecoder, Qwen3Tokenizer, _SPECIAL_TOKENS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Fake vocabulary: special tokens get IDs 0-15, regular tokens start at 100.
_FAKE_VOCAB = {tok: i for i, tok in enumerate(_SPECIAL_TOKENS)}


def _make_tokenizer(
    is_instruct_model=True,
    apply_chat_template=True,
    add_generation_prompt=True,
    enable_thinking=True,
):
    """Build a Qwen3Tokenizer with a mocked HuggingFace Tokenizer backend."""
    mock_hf = MagicMock()
    mock_hf.token_to_id.side_effect = lambda t: _FAKE_VOCAB.get(t)

    # Regular text encoding: return distinct IDs based on text content
    def fake_encode(text):
        result = MagicMock()
        # Deterministic fake: each character gets ID = 100 + ord
        result.ids = [100 + ord(c) for c in text]
        return result

    mock_hf.encode.side_effect = fake_encode

    # Decoding: join token IDs back (good enough for testing)
    def fake_decode(ids, skip_special_tokens=False):
        parts = []
        id_to_special = {v: k for k, v in _FAKE_VOCAB.items()}
        for tid in ids:
            if tid in id_to_special and not skip_special_tokens:
                parts.append(id_to_special[tid])
            elif tid >= 100:
                parts.append(chr(tid - 100))
            else:
                parts.append(f"[{tid}]")
        return "".join(parts)

    mock_hf.decode.side_effect = fake_decode

    with patch("cantollm.models.qwen3.tokenizer.Tokenizer") as MockTokenizer:
        MockTokenizer.from_file.return_value = mock_hf
        tok = Qwen3Tokenizer(
            tokenizer_file_path="fake.json",
            is_instruct_model=is_instruct_model,
            apply_chat_template=apply_chat_template,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
    return tok


# ---------------------------------------------------------------------------
# Qwen3Tokenizer: init and token IDs
# ---------------------------------------------------------------------------


class TestTokenizerInit:
    def test_special_token_ids_populated(self):
        tok = _make_tokenizer()
        assert len(tok._special_to_id) == len(_SPECIAL_TOKENS)
        for name in _SPECIAL_TOKENS:
            assert name in tok._special_to_id

    def test_pad_token_id(self):
        tok = _make_tokenizer()
        assert tok.pad_token_id == _FAKE_VOCAB["<|endoftext|>"]

    def test_eos_instruct_model(self):
        tok = _make_tokenizer(is_instruct_model=True)
        assert tok.eos_token_id == _FAKE_VOCAB["<|im_end|>"]

    def test_eos_base_model(self):
        tok = _make_tokenizer(is_instruct_model=False)
        assert tok.eos_token_id == tok.pad_token_id

    def test_stop_token_ids(self):
        tok = _make_tokenizer()
        assert tok.pad_token_id in tok.stop_token_ids
        assert tok.eos_token_id in tok.stop_token_ids


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


class TestEncode:
    def test_plain_text_no_wrapping(self):
        tok = _make_tokenizer(apply_chat_template=False)
        ids = tok.encode("hi")
        # "hi" -> [100+ord('h'), 100+ord('i')]
        assert ids == [100 + ord("h"), 100 + ord("i")]

    def test_special_token_shortcut(self):
        tok = _make_tokenizer(apply_chat_template=False)
        ids = tok.encode("<|im_start|>")
        assert ids == [_FAKE_VOCAB["<|im_start|>"]]

    def test_special_token_shortcut_with_whitespace(self):
        tok = _make_tokenizer(apply_chat_template=False)
        ids = tok.encode("  <|im_end|>  ")
        assert ids == [_FAKE_VOCAB["<|im_end|>"]]

    def test_mixed_text_and_special_tokens(self):
        tok = _make_tokenizer(apply_chat_template=False)
        ids = tok.encode("a<|im_start|>b")
        assert ids[0] == 100 + ord("a")
        assert ids[1] == _FAKE_VOCAB["<|im_start|>"]
        assert ids[2] == 100 + ord("b")

    def test_think_tokens_encoded_as_special(self):
        tok = _make_tokenizer(apply_chat_template=False)
        ids = tok.encode("x<think>y</think>z")
        assert _FAKE_VOCAB["<think>"] in ids
        assert _FAKE_VOCAB["</think>"] in ids

    def test_chat_wrapped_default(self):
        """When apply_chat_template=True, encode() wraps by default."""
        tok = _make_tokenizer(apply_chat_template=True)
        ids = tok.encode("hello")
        # Should contain <|im_start|> and <|im_end|> tokens from the template
        assert _FAKE_VOCAB["<|im_start|>"] in ids
        assert _FAKE_VOCAB["<|im_end|>"] in ids

    def test_chat_wrapped_explicit_false(self):
        """chat_wrapped=False overrides the default."""
        tok = _make_tokenizer(apply_chat_template=True)
        ids = tok.encode("hello", chat_wrapped=False)
        assert _FAKE_VOCAB["<|im_start|>"] not in ids
        assert _FAKE_VOCAB["<|im_end|>"] not in ids

    def test_special_token_shortcut_respects_chat_wrapped(self):
        """A single-special-token input must still be wrapped when chat_wrapped=True.

        Regression: the quick-path shortcut used to fire before the
        chat_wrapped check, returning the raw id instead of the wrapped form.
        """
        tok = _make_tokenizer(apply_chat_template=False)
        ids = tok.encode("<|im_start|>", chat_wrapped=True)
        # Wrapping emits multiple <|im_start|> / <|im_end|> pairs (user turn +
        # generation prompt), so the result must be longer than a single id.
        assert len(ids) > 1
        assert ids.count(_FAKE_VOCAB["<|im_start|>"]) >= 2


# ---------------------------------------------------------------------------
# Chat template
# ---------------------------------------------------------------------------


class TestChatTemplate:
    def test_wrap_chat_basic(self):
        tok = _make_tokenizer(add_generation_prompt=True, enable_thinking=True)
        result = tok._wrap_chat("hi")
        assert result == "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"

    def test_wrap_chat_no_generation_prompt(self):
        tok = _make_tokenizer(add_generation_prompt=False, enable_thinking=True)
        result = tok._wrap_chat("hi")
        assert result == "<|im_start|>user\nhi<|im_end|>\n"
        assert "assistant" not in result

    def test_wrap_chat_thinking_disabled(self):
        tok = _make_tokenizer(add_generation_prompt=True, enable_thinking=False)
        result = tok._wrap_chat("hi")
        assert "<think>\n\n</think>\n\n" in result

    def test_wrap_chat_thinking_enabled(self):
        tok = _make_tokenizer(add_generation_prompt=True, enable_thinking=True)
        result = tok._wrap_chat("hi")
        assert "<think>" not in result


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


class TestDecode:
    def test_roundtrip_plain_text(self):
        tok = _make_tokenizer(apply_chat_template=False)
        ids = tok.encode("abc")
        text = tok.decode(ids)
        assert text == "abc"

    def test_decode_includes_special_tokens(self):
        tok = _make_tokenizer()
        text = tok.decode([_FAKE_VOCAB["<|im_start|>"]])
        assert "<|im_start|>" in text


# ---------------------------------------------------------------------------
# IncrementalDecoder
# ---------------------------------------------------------------------------


class TestIncrementalDecoder:
    def test_basic_streaming(self):
        tok = _make_tokenizer(apply_chat_template=False)
        dec = tok.incremental_decoder()
        assert isinstance(dec, IncrementalDecoder)

        # Feed tokens one at a time, accumulate output
        accumulated = ""
        for tid in [100 + ord("h"), 100 + ord("i")]:
            accumulated += dec.add(tid)
        accumulated += dec.flush()
        assert accumulated == "hi"

    def test_flush_on_empty(self):
        tok = _make_tokenizer()
        dec = tok.incremental_decoder()
        assert dec.flush() == ""

    def test_reset_clears_state(self):
        tok = _make_tokenizer(apply_chat_template=False)
        dec = tok.incremental_decoder()
        dec.add(100 + ord("a"))
        dec.reset()
        # After reset, new tokens start fresh
        result = dec.add(100 + ord("b"))
        result += dec.flush()
        assert result == "b"
        assert "a" not in result

    def test_no_duplicate_text(self):
        """Each call to add() should only return *new* text, not repeat prior output."""
        tok = _make_tokenizer(apply_chat_template=False)
        dec = tok.incremental_decoder()
        chunks = []
        for c in "hello":
            chunks.append(dec.add(100 + ord(c)))
        chunks.append(dec.flush())
        assert "".join(chunks) == "hello"
