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

from dataclasses import dataclass
from typing import Protocol

import torch

from cantollm.engine.types import InferenceRequest, TokenEvent
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
