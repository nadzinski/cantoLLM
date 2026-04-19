"""Tokenizer for Qwen3 models.

Wraps the HuggingFace tokenizers Rust library and adds chat template
formatting, special token handling, and an incremental decoder for streaming.

Known quirk: the quick-path in encode() (lines ~135-137) checks if the
stripped text is a bare special token and returns its ID immediately,
*before* considering chat_wrapped. So encode("<|im_start|>",
chat_wrapped=True) returns the raw token ID rather than wrapping it in a
chat template. To fix this, the quick-path should be gated on
``not chat_wrapped``.
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
    "<think>",
    "</think>",
]

_SPECIAL_RE = re.compile(r"(<\|[^>]+?\|>|<think>|</think>)")


class IncrementalDecoder:
    """Buffers BPE tokens and emits only fully-decoded (stable) text.

    A single Unicode character (like an emoji) can span multiple BPE tokens.
    Decoding one token at a time produces U+FFFD replacement characters for
    incomplete multi-byte sequences.  This class accumulates tokens and only
    emits text once it is confident the bytes are stable.
    """

    def __init__(self, tokenizer: "Qwen3Tokenizer"):
        self._tokenizer = tokenizer
        self._tokens: list[int] = []
        self._emitted: str = ""

    def add(self, token_id: int) -> str:
        """Append a token and return any newly-stable text."""
        self._tokens.append(token_id)
        decoded = self._tokenizer.decode(self._tokens)

        # Find the stable prefix by scanning backward past any replacement chars.
        stable_end = len(decoded)
        while stable_end > 0 and decoded[stable_end - 1] == "\ufffd":
            stable_end -= 1

        stable = decoded[:stable_end]
        new_text = stable[len(self._emitted):]
        self._emitted = stable
        return new_text

    def flush(self) -> str:
        """Return any remaining text that hasn't been emitted yet."""
        if not self._tokens:
            return ""
        decoded = self._tokenizer.decode(self._tokens)
        remaining = decoded[len(self._emitted):]
        self._emitted = decoded
        return remaining

    def reset(self):
        """Clear internal state."""
        self._tokens = []
        self._emitted = ""


class Qwen3Tokenizer:
    """Tokenizer for Qwen3 base and instruct models.

    Handles special-token-aware encoding, ChatML wrapping, and provides
    an incremental decoder for streaming generation.
    """

    def __init__(
        self,
        tokenizer_file_path: str,
        is_instruct_model: bool = True,
        apply_chat_template: bool = True,
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
    ):
        self._tok = Tokenizer.from_file(tokenizer_file_path)

        # Build special token -> id mapping, skipping any the vocab doesn't have.
        self._special_to_id: dict[str, int] = {}
        for tok in _SPECIAL_TOKENS:
            tid = self._tok.token_to_id(tok)
            if tid is not None:
                self._special_to_id[tok] = tid

        # Convenience token IDs.
        self.pad_token_id: int = self._special_to_id["<|endoftext|>"]
        self.eos_token_id: int = (
            self._special_to_id.get("<|im_end|>", self.pad_token_id)
            if is_instruct_model
            else self.pad_token_id
        )
        self.stop_token_ids: set[int] = {self.eos_token_id, self.pad_token_id}
        self.thinking_start_id: int | None = self._special_to_id.get("<think>")
        self.thinking_end_id: int | None = self._special_to_id.get("</think>")

        # Public config flags.
        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt = add_generation_prompt
        self.enable_thinking = enable_thinking

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def encode(self, text: str, chat_wrapped: bool = None) -> list[int]:
        """Encode text to token IDs, optionally wrapping in a chat template.

        Args:
            text: Input text to encode.
            chat_wrapped: Whether to apply ChatML wrapping.  Defaults to
                ``self.apply_chat_template`` when *None*.

        Returns:
            List of token IDs.
        """
        if chat_wrapped is None:
            chat_wrapped = self.apply_chat_template

        # Quick path: the entire (stripped) text is a single special token.
        stripped = text.strip()
        if "\n" not in stripped and stripped in self._special_to_id:
            return [self._special_to_id[stripped]]

        if chat_wrapped:
            text = self._wrap_chat(text)

        # Split on special-token boundaries and encode piece by piece.
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
        """Decode token IDs back to a string (including special tokens)."""
        return self._tok.decode(ids, skip_special_tokens=False)

    def incremental_decoder(self) -> IncrementalDecoder:
        """Return a new :class:`IncrementalDecoder` bound to this tokenizer."""
        return IncrementalDecoder(self)

    # ------------------------------------------------------------------
    # Chat template helpers
    # ------------------------------------------------------------------

    def encode_conversation(self, messages: list[dict], system: str | None = None) -> list[int]:
        """Encode an Anthropic-style messages array into token IDs.

        Builds a full ChatML string from the message history and encodes it.

        Args:
            messages: List of {"role": "user"|"assistant", "content": ...} dicts.
                ``content`` is either a string or a list of content-block dicts
                of the form ``{"type": "text", "text": "..."}``.
            system: Optional system prompt.

        Returns:
            List of token IDs ready for model input.
        """
        parts = []

        if system:
            parts.append(f"<|im_start|>system\n{system}<|im_end|>\n")

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if isinstance(content, list):
                # Content block array -> extract text
                content = "\n".join(
                    block["text"] for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

        # Generation prompt for assistant
        if self.add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
            if not self.enable_thinking:
                parts.append("<think>\n\n</think>\n\n")

        return self.encode("".join(parts), chat_wrapped=False)

    def _wrap_chat(self, user_msg: str) -> str:
        """Wrap a user message in ChatML format.

        Produces::

            <|im_start|>user
            {user_msg}<|im_end|>
            <|im_start|>assistant
            [<think>

            </think>

            ]   # only when thinking is disabled

        The thinking-suppression block is appended only when
        ``self.enable_thinking`` is *False*.
        """
        parts = [f"<|im_start|>user\n{user_msg}<|im_end|>\n"]
        if self.add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
            if not self.enable_thinking:
                parts.append("<think>\n\n</think>\n\n")
        return "".join(parts)
