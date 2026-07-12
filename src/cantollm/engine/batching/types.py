"""Seam types for the continuous-batching engine.

`BatchedForwardFn` is the scheduler's entire view of the model (decision 4):
the scheduler is constructed with one of these — in production it's
`ModelRuntime.forward_batched`, in tests a toy stepper — and never imports a
model class.

The command dataclasses are everything the outside world can tell the engine
(decision 6): they cross the API→scheduler-thread boundary on one
thread-safe queue, drained at the top of each step. Deliberately
message-shaped — when Phase 2's process split lands, these serialize over
IPC and the scheduler side doesn't change.

`SchedulerLike` is the shell's entire view of the scheduler — what the
scripted test double implements, and what the real scheduler (step 8)
implements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch

from cantollm.engine.types import InferenceRequest, SamplingParams, TokenEvent
from cantollm.kv_pool import PaddedKVPool
from cantollm.models.attention.protocol import BatchMeta


@dataclass(frozen=True)
class AddRequest:
    request: InferenceRequest


@dataclass(frozen=True)
class Abort:
    request_id: str


@dataclass(frozen=True)
class Shutdown:
    pass


Command = AddRequest | Abort | Shutdown


@dataclass
class CBSequence:
    """Per-request state owned by the CB scheduler (decision 2: deliberately
    NOT unified with `engine/types.Sequence`, whose `cache: KVCache` and
    `stop_event` are sequential-path artifacts).

    Ported from the prototype's `cb_types.Sequence`, plus `sampling_params`
    (the prototype was greedy-only). `position` counts how many tokens of
    this sequence the model has consumed — prompt first, then generated
    tokens; the KV write offset and the RoPE base both derive from it.
    """

    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    max_tokens: int
    stop_token_ids: set[int]
    slot_idx: int | None = None
    position: int = 0
    output_token_ids: list[int] = field(default_factory=list)

    def is_prefilling(self) -> bool:
        return self.position < len(self.prompt_token_ids)

    def input_tokens_at(self, start: int, n: int) -> list[int]:
        """The next `n` input tokens from position `start`: prompt tokens
        while prefilling; after that, the single last generated token."""
        if start < len(self.prompt_token_ids):
            return self.prompt_token_ids[start : start + n]
        assert n == 1, "decode rows consume exactly one token"
        return self.output_token_ids[-1:]


class SchedulerLike(Protocol):
    """What the engine shell drives, on the scheduler thread only.

    Contract notes the shell relies on:
      - `is_idle()` is False whenever `step()` would produce events —
        including pending abort acknowledgements with no forward pass to
        run. The shell blocks on the command queue while idle, so an event
        that `is_idle()` doesn't announce would never be flushed.
      - `step()` is never called while idle.
      - `abort()` of an unknown/finished request id is a silent no-op.
    """

    def add_request(self, request: InferenceRequest) -> None: ...

    def abort(self, request_id: str) -> None: ...

    def step(self) -> list[TokenEvent]: ...

    def is_idle(self) -> bool: ...


class BatchedForwardFn(Protocol):
    def __call__(
        self,
        input_ids: torch.Tensor,
        meta: BatchMeta,
        pool: PaddedKVPool,
    ) -> torch.Tensor:
        """One mixed prefill/decode step.

        `input_ids` is (B, num_new_max) int64, left-aligned, 0-padded, row
        order matching `meta.rows`. Returns (B, vocab) logits taken at each
        row's last real token. Rows still mid-prefill get logits too in v1;
        the scheduler must not sample from them.
        """
        ...
