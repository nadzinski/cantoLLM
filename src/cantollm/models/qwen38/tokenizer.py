"""Tokenizer for Qwen 3.8 models.

A copy of the qwen3 tokenizer adapted to the new vocab (248,320 tokens):
ChatML survives unchanged (<|im_start|> 248045, <|im_end|> 248046 = eos,
<|endoftext|> 248044 = pad, <think>/</think> 248068/248069), but the
added-token inventory grows (tool_call/tool_response tags, FIM, repo,
audio markers), and every one of those must be neutered in untrusted
message content, so the reserved-marker regex covers the XML-ish tags
too, not just <|...|> and the think markers.
"""

import re

from tokenizers import Tokenizer

_SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|fim_pad|>",
    "<|repo_name|>",
    "<|file_sep|>",
    "<think>",
    "</think>",
    "<|audio_start|>",
    "<|audio_end|>",
]

# Any registered added token that could appear literally in content and
# be extracted by the raw tokenizer: <|...|> plus the XML-ish tags.
_SPECIAL_RE = re.compile(
    r"(<\|[^>]+?\|>|<think>|</think>|</?tool_call>|</?tool_response>)"
)


class IncrementalDecoder:
    """Copy of the qwen3 incremental decoder (see that docstring for the
    U+FFFD windowing rationale); depends only on `decode`."""

    def __init__(self, tokenizer: "Qwen38Tokenizer"):
        self._tokenizer = tokenizer
        self._window: list[int] = []
        self._emitted: str = ""

    def add(self, token_id: int) -> str:
        self._window.append(token_id)
        decoded = self._tokenizer.decode(self._window)

        stable_end = len(decoded)
        while stable_end > 0 and decoded[stable_end - 1] == "�":
            stable_end -= 1

        new_text = decoded[len(self._emitted):stable_end]
        if stable_end == len(decoded):
            self._window = []
            self._emitted = ""
        else:
            self._emitted = decoded[:stable_end]
        return new_text

    def flush(self) -> str:
        if not self._window:
            return ""
        decoded = self._tokenizer.decode(self._window)
        remaining = decoded[len(self._emitted):]
        self._window = []
        self._emitted = ""
        return remaining

    def reset(self):
        self._window = []
        self._emitted = ""


class Qwen38Tokenizer:
    """Same duck-typed surface as Qwen3Tokenizer (the API layer consumes
    encode_conversation / encode / decode / incremental_decoder /
    stop_token_ids / eos, pad, thinking ids / apply_chat_template)."""

    def __init__(
        self,
        tokenizer_file_path: str,
        is_instruct_model: bool = True,
        apply_chat_template: bool = True,
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
    ):
        self._tok = Tokenizer.from_file(tokenizer_file_path)

        self._special_to_id: dict[str, int] = {}
        for tok in _SPECIAL_TOKENS:
            tid = self._tok.token_to_id(tok)
            if tid is not None:
                self._special_to_id[tok] = tid

        self.pad_token_id: int = self._special_to_id["<|endoftext|>"]
        self.eos_token_id: int = (
            self._special_to_id.get("<|im_end|>", self.pad_token_id)
            if is_instruct_model
            else self.pad_token_id
        )
        self.stop_token_ids: set[int] = {self.eos_token_id, self.pad_token_id}
        self.thinking_start_id: int | None = self._special_to_id.get("<think>")
        self.thinking_end_id: int | None = self._special_to_id.get("</think>")

        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt = add_generation_prompt
        self.enable_thinking = enable_thinking

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def encode(self, text: str, chat_wrapped: bool = None) -> list[int]:
        if chat_wrapped is None:
            chat_wrapped = self.apply_chat_template

        stripped = text.strip()
        if not chat_wrapped and "\n" not in stripped and stripped in self._special_to_id:
            return [self._special_to_id[stripped]]

        if chat_wrapped:
            text = self._wrap_chat(text)

        parts = _SPECIAL_RE.split(text)
        ids: list[int] = []
        for part in parts:
            if not part:
                continue
            if part in self._special_to_id:
                ids.append(self._special_to_id[part])
            else:
                ids.extend(self._tok.encode(part).ids)
        return ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=False)

    def incremental_decoder(self) -> IncrementalDecoder:
        return IncrementalDecoder(self)

    # ------------------------------------------------------------------
    # Chat template helpers
    # ------------------------------------------------------------------

    def encode_conversation(self, messages: list[dict], system: str | None = None) -> list[int]:
        """ChatML frame assembled in token space; content is data, never
        control (see the qwen3 docstring: markers inside content encode
        as their surface text, so bodies can't forge turn boundaries)."""
        im_start = self._special_to_id["<|im_start|>"]
        im_end = self._special_to_id["<|im_end|>"]
        ids: list[int] = []

        def _turn(role: str, content: str) -> None:
            ids.append(im_start)
            ids.extend(self._encode_content(f"{role}\n{content}"))
            ids.append(im_end)
            ids.extend(self._encode_content("\n"))

        if system:
            _turn("system", system)

        for msg in messages:
            content = msg["content"]
            if isinstance(content, list):
                content = "\n".join(
                    block["text"] for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            _turn(msg["role"], content)

        if self.add_generation_prompt:
            ids.append(im_start)
            ids.extend(self._encode_content("assistant\n"))
            if not self.enable_thinking:
                ids.append(self.thinking_start_id)
                ids.extend(self._encode_content("\n\n"))
                ids.append(self.thinking_end_id)
                ids.extend(self._encode_content("\n\n"))

        return ids

    def _encode_content(self, text: str) -> list[int]:
        """Neuter every reserved marker in untrusted content: split the
        marker, feed its first char and remainder through the BPE model
        separately so the added-token trie can never match it whole."""
        out: list[int] = []
        for part in _SPECIAL_RE.split(text):
            if not part:
                continue
            if part in self._special_to_id:
                out.extend(self._tok.encode(part[0]).ids)
                out.extend(self._tok.encode(part[1:]).ids)
            else:
                out.extend(self._tok.encode(part).ids)
        return out

    def _wrap_chat(self, user_msg: str) -> str:
        parts = [f"<|im_start|>user\n{user_msg}<|im_end|>\n"]
        if self.add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
            if not self.enable_thinking:
                parts.append("<think>\n\n</think>\n\n")
        return "".join(parts)
