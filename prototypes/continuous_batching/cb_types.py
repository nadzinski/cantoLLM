"""Domain types for the continuous-batching prototype.

Stripped-down analogues of `src/cantollm/engine/types.py` — same shape,
fewer fields. The scheduler mutates `Sequence`; the rest are immutable
records.
"""

from dataclasses import dataclass, field
from typing import Literal


FinishReason = Literal["end_turn", "max_tokens", "abort"]


@dataclass
class Request:
    request_id: str
    prompt_token_ids: list[int]
    max_tokens: int
    stop_token_ids: set[int] = field(default_factory=set)


@dataclass
class Sequence:
    request_id: str
    prompt_token_ids: list[int]
    max_tokens: int
    stop_token_ids: set[int]
    slot_idx: int | None = None
    position: int = 0
    output_token_ids: list[int] = field(default_factory=list)
    finish_reason: FinishReason | None = None
    aborted: bool = False

    def is_prefilling(self) -> bool:
        return self.position < len(self.prompt_token_ids)

    def input_tokens_at(self, start: int, n: int) -> list[int]:
        """Tokens this sequence would feed in if a forward pass started at `start`.

        During prefill, that's the next slice of the prompt. After prefill,
        the only input is the most recently sampled output token.
        """
        if start < len(self.prompt_token_ids):
            return self.prompt_token_ids[start : start + n]
        assert n == 1, "decode rows feed back exactly one token"
        return self.output_token_ids[-1:]


@dataclass
class TokenEvent:
    request_id: str
    token_id: int | None = None
    finish_reason: FinishReason | None = None
