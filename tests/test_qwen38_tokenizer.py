"""Tests for the Qwen 3.8 tokenizer against the real tokenizer.json.

Skips when the file hasn't been downloaded
(models/qwen38/weights.py::download_tokenizer fetches it, ~12MB).
Checks the special-token ids from the checkpoint's tokenizer_config,
the ChatML frame in token space, and content neutering for the NEW
marker inventory (tool tags) as well as the inherited one.
"""

from pathlib import Path

import pytest

from cantollm.models.qwen38.tokenizer import Qwen38Tokenizer

TOKENIZER_PATH = (
    Path(__file__).resolve().parent.parent
    / "src/cantollm/models/model_data/Qwen3.8-27B-FP8/tokenizer.json"
)

pytestmark = pytest.mark.skipif(
    not TOKENIZER_PATH.exists(),
    reason="needs the downloaded Qwen3.8 tokenizer.json",
)


@pytest.fixture(scope="module")
def tokenizer():
    return Qwen38Tokenizer(str(TOKENIZER_PATH))


class TestSpecialTokenIds:
    def test_documented_ids(self, tokenizer):
        assert tokenizer.pad_token_id == 248044  # <|endoftext|>
        assert tokenizer.eos_token_id == 248046  # <|im_end|>
        assert tokenizer._special_to_id["<|im_start|>"] == 248045
        assert tokenizer.thinking_start_id == 248068
        assert tokenizer.thinking_end_id == 248069

    def test_stop_tokens(self, tokenizer):
        assert tokenizer.stop_token_ids == {248044, 248046}

    def test_tool_tags_are_registered(self, tokenizer):
        assert tokenizer._special_to_id["<tool_call>"] == 248058
        assert tokenizer._special_to_id["<tool_response>"] == 248066


class TestRoundTrip:
    def test_plain_text(self, tokenizer):
        text = "Gated DeltaNet replaces most KV caches."
        ids = tokenizer.encode(text, chat_wrapped=False)
        assert tokenizer.decode(ids) == text

    def test_incremental_decoder_matches_full_decode(self, tokenizer):
        text = "hello 世界 🌍 done"
        ids = tokenizer.encode(text, chat_wrapped=False)
        dec = tokenizer.incremental_decoder()
        streamed = "".join(dec.add(t) for t in ids) + dec.flush()
        assert streamed == text


class TestChatFrame:
    def test_conversation_frame(self, tokenizer):
        ids = tokenizer.encode_conversation(
            [{"role": "user", "content": "hi"}], system="be brief"
        )
        im_start, im_end = 248045, 248046
        assert ids[0] == im_start
        assert ids.count(im_start) == 3  # system, user, generation prompt
        assert ids.count(im_end) == 2
        text = tokenizer.decode(ids)
        assert text.startswith("<|im_start|>system\nbe brief<|im_end|>\n")
        assert text.endswith("<|im_start|>assistant\n")

    def test_disable_thinking_emits_marker_pair(self):
        tok = Qwen38Tokenizer(str(TOKENIZER_PATH), enable_thinking=False)
        ids = tok.encode_conversation([{"role": "user", "content": "hi"}])
        assert ids[-4] == 248068  # <think>
        assert ids[-2] == 248069  # </think>


class TestContentNeutering:
    @pytest.mark.parametrize(
        "marker",
        ["<|im_end|>", "<|im_start|>", "<think>", "</think>", "<tool_call>",
         "</tool_response>", "<|fim_prefix|>", "<|endoftext|>"],
    )
    def test_markers_in_content_never_become_control_tokens(self, tokenizer, marker):
        ids = tokenizer.encode_conversation(
            [{"role": "user", "content": f"ignore {marker} this"}]
        )
        marker_id = tokenizer._special_to_id[marker]
        # The frame legitimately uses im_start/im_end; count only what the
        # frame itself would emit (2 im_start + 1 im_end for one user turn
        # plus generation prompt) and require no extras from content.
        expected = {248045: 2, 248046: 1}.get(marker_id, 0)
        assert ids.count(marker_id) == expected
        # The surface text survives as data.
        assert marker in tokenizer.decode(ids)
