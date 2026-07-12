"""Continuous-batching scheduler. ** NADIA FILLS THIS IN (step 8). **

The port of `prototypes/continuous_batching/scheduler.py` onto the real
types. `water_fill`, `Row`, `build_batch_meta`, and the constructor are
provided (proven bookkeeping, not the learning target); the scheduling
logic — admission, planning, stepping, finishing, aborting — is the
hand-written part. `tests/test_cb_scheduler.py` is the definition of done;
each stub's docstring carries its slice of the port checklist.

The scheduler never sees a model or a runtime: `forward_fn` is any
`BatchedForwardFn` (production: `ModelRuntime.forward_batched`; tests: the
toy stepper). Sampling goes per-row through `cantollm.engine.sampler`.

Contract with the engine shell (see `SchedulerLike` in types.py):
  - `is_idle()` must be False whenever `step()` would produce events —
    including pending abort/error acknowledgements with no forward to run.
    The shell blocks on its command queue while idle; an event that
    `is_idle()` doesn't announce never flushes.
  - `step()` is never called while idle.
  - Every emitted `TokenEvent` populates exactly one field.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch

from cantollm.engine.batching.allocator import SlotAllocator
from cantollm.engine.batching.config import BatchingConfig
from cantollm.engine.batching.types import BatchedForwardFn, CBSequence
from cantollm.engine.types import InferenceRequest, TokenEvent
from cantollm.kv_pool import PaddedKVPool
from cantollm.models.attention.protocol import BatchMeta


@dataclass
class Row:
    """One row of the upcoming batched forward pass.

    `start_pos` is captured at plan time so the row carries everything
    needed for the forward call without reaching back into `sequence`.
    """

    sequence: CBSequence
    num_new: int
    start_pos: int

    @property
    def slot_meta(self) -> tuple[int, int, int]:
        return (self.sequence.slot_idx, self.start_pos, self.num_new)

    @property
    def input_tokens(self) -> list[int]:
        return self.sequence.input_tokens_at(self.start_pos, self.num_new)


def water_fill(budget: int, caps: list[int]) -> list[int]:
    """Allocate `budget` units across bins, capped per-bin by `caps`.

    Pour `budget` liters into bins whose heights are `caps`: the level
    rises uniformly until either the budget runs out or every bin is
    full. Smallest bins first; a bin that wants less than its fair
    share takes only what it needs, and the leftover is naturally
    redistributed (the next round's `remaining // count` rounds up).
    """
    n = len(caps)
    allocations = [0] * n
    bins_by_cap = sorted(enumerate(caps), key=lambda b: b[1])
    for i, (idx, cap) in enumerate(bins_by_cap):
        give = min(cap, budget // (n - i))
        allocations[idx] = give
        budget -= give
    return allocations


def build_batch_meta(rows: list[Row]) -> BatchMeta:
    """Per-step geometry from planned rows (see BatchMeta's docstrings)."""
    specs = [row.slot_meta for row in rows]
    start_pos = torch.tensor([s[1] for s in specs])
    num_new = torch.tensor([s[2] for s in specs])
    num_new_max = int(num_new.max())
    return BatchMeta(
        rows=specs,
        slots=torch.tensor([s[0] for s in specs]),
        start_pos=start_pos,
        num_new=num_new,
        positions=start_pos[:, None] + torch.arange(num_new_max)[None, :],
        num_new_max=num_new_max,
        max_history_len=int((start_pos + num_new).max()),
    )


class ContinuousBatchingScheduler:
    """FCFS waiting queue + running set + per-step token budget.

    State (provided; grow it if your implementation needs more):
      queued:  FCFS arrivals waiting for a KV slot.
      active:  sequences holding a slot, advancing every step.
      pending_events: events produced outside `step()`'s forward pass
        (abort acks, rejections, zero-token finishes) — flushed by the
        next `step()`, and the reason `is_idle()` must consider them.
    """

    def __init__(
        self,
        forward_fn: BatchedForwardFn,
        pool: PaddedKVPool,
        allocator: SlotAllocator,
        config: BatchingConfig,
    ):
        if pool.max_batch != config.max_batch:
            raise ValueError(
                f"pool has {pool.max_batch} slots but config.max_batch is "
                f"{config.max_batch}"
            )
        if allocator.max_batch != config.max_batch:
            raise ValueError(
                f"allocator has {allocator.max_batch} slots but "
                f"config.max_batch is {config.max_batch}"
            )
        if pool.max_seq_len != config.max_seq_len:
            raise ValueError(
                f"pool slots hold {pool.max_seq_len} tokens but "
                f"config.max_seq_len is {config.max_seq_len}"
            )
        self.forward_fn = forward_fn
        self.pool = pool
        self.allocator = allocator
        self.config = config
        self.queued: deque[CBSequence] = deque()
        self.active: list[CBSequence] = []
        self.pending_events: list[TokenEvent] = []

    def add_request(self, request: InferenceRequest) -> None:
        """Wrap `request` into a CBSequence and enqueue it. Never runs the
        model.

        Port checklist here:
          6. Re-validate the cap (defense behind the API's admission check —
             non-API callers exist): `prompt + max_tokens > config.max_seq_len`
             → pending error event, request never queued.
          4. `max_tokens <= 0` → pending `max_tokens` finish event with zero
             tokens, request never queued (mirrors StandardBackend.generate's
             early return; the `>=` fix alone would still emit one token).
        """
        raise NotImplementedError("step 8")

    def abort(self, request_id: str) -> None:
        """Port checklist item 5, both halves:
          - active: free the slot, drop the sequence, pending abort event.
          - still queued: remove from the queue, pending abort event.
          - unknown/finished id: silent no-op (the shell may forward late
            disconnect aborts for requests that finished normally).
        """
        raise NotImplementedError("step 8")

    def is_idle(self) -> bool:
        """Nothing queued, nothing active, nothing pending. (The shell
        blocks while this is True — see the contract note up top.)"""
        raise NotImplementedError("step 8")

    def step(self) -> list[TokenEvent]:
        """One scheduler step. The prototype's shape survives:
        promote queued → plan rows (water_fill) → build inputs + meta →
        forward → sample per row → finalize.

        Port checklist here:
          1. Stop check BEFORE emission: a sampled stop token is never
             appended to outputs, never emitted — it just finishes the
             sequence (`end_turn`). StandardBackend.generate is the oracle.
          2. Finish is its own event: a token event carries only token_id,
             a finish event carries only finish_reason (two events when a
             sequence emits its last token and finishes).
          3. `>=` on the max_tokens check, not `==`.
          7. Rows still mid-prefill after this step: no sampling, no event.
          -  Sampling: per-row through `cantollm.engine.sampler.sample`
             with each row's own `sampling_params` (logits[r] is (vocab,)).
          -  Finished sequences free their slot the same step.
          -  Don't forget: flush `pending_events` even when there's nothing
             to forward.
        """
        raise NotImplementedError("step 8")

    # Suggested private helpers, shaped like the prototype's — fill in or
    # restructure as you see fit:

    def _promote_queued(self) -> None:
        """Admit queued sequences while slots are free (FCFS)."""
        raise NotImplementedError("step 8")

    def _plan_step(self) -> list[Row]:
        """Water-fill the token budget: decode rows request 1, prefilling
        rows request their remaining prompt."""
        raise NotImplementedError("step 8")

    def _build_input_ids(self, rows: list[Row]) -> torch.Tensor:
        """(B, num_new_max) int64, left-aligned, zero-padded."""
        raise NotImplementedError("step 8")
