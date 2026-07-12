"""Continuous-batching scheduler: who advances, and by how much, each step.

Hand-written port of `prototypes/continuous_batching/scheduler.py` onto the
real engine types. Each step: promote queued sequences into free KV slots
(FCFS), water-fill the per-step token budget across active rows (decode
rows want 1 token, prefilling rows their remaining prompt), run one mixed
batched forward, then sample and finalize per row.

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
from cantollm.engine import sampler
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

    State:
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
        """Validate and enqueue `request`; never runs the model.

        Two short-circuits produce pending events instead of queueing:
          - `prompt + max_tokens > config.max_seq_len` → error event.
            Defense behind the API's admission 400 (non-API callers exist);
            an over-cap request would take a slot it must overflow.
          - `max_tokens <= 0` → immediate `max_tokens` finish with zero
            tokens, mirroring StandardBackend.generate's early return.
        """
        total = len(request.prompt_token_ids) + request.max_tokens
        if total > self.config.max_seq_len:
            # Defense behind the API's admission 400 — never take a slot the
            # request must eventually overflow.
            self.pending_events.append(TokenEvent(
                error=(
                    f"prompt ({len(request.prompt_token_ids)} tokens) + "
                    f"max_tokens ({request.max_tokens}) = {total} exceeds the "
                    f"slot capacity of {self.config.max_seq_len}"
                ),
                request_id=request.request_id,
            ))
            return
        if request.max_tokens <= 0:
            # Nothing to generate: finish immediately, never queue — the >=
            # check in step() alone would still emit one spurious token.
            self.pending_events.append(TokenEvent(
                finish_reason="max_tokens", request_id=request.request_id
            ))
            return

        sequence = CBSequence(
            request_id=request.request_id,
            prompt_token_ids=list(request.prompt_token_ids),
            sampling_params=request.sampling_params,
            max_tokens=request.max_tokens,
            stop_token_ids=set(request.stop_token_ids),
        )
        self.queued.append(sequence)

    def abort(self, request_id: str) -> None:
        """Stop a request wherever it is; both halves emit a pending
        `abort` finish event.

        Active: drop the sequence and free its slot (next step's promotion
        can hand it out). Queued: just remove. Unknown/finished ids are a
        silent no-op — the shell forwards disconnect aborts even for
        requests that already finished normally.
        """
        seq = next((s for s in self.active if s.request_id == request_id), None)
        if seq is not None:
            self.active.remove(seq)
            self.allocator.free(seq.slot_idx)
        else:
            seq = next((s for s in self.queued if s.request_id == request_id), None)
            if seq is None:
                return
            self.queued.remove(seq)
        self.pending_events.append(
            TokenEvent(finish_reason="abort", request_id=request_id)
        )

    def is_idle(self) -> bool:
        """Nothing queued, nothing active, nothing pending. (The shell
        blocks while this is True — see the contract note up top.)"""
        return not self.queued and not self.active and not self.pending_events

    def step(self) -> list[TokenEvent]:
        """One scheduler step: flush pending events, promote queued
        sequences, water-fill the budget into rows, forward once, then
        sample and finalize per row.

        Finalize semantics (StandardBackend.generate is the oracle):
          - Rows still mid-prefill: no sampling, no event.
          - A sampled stop token is never appended or emitted — the
            sequence just finishes (`end_turn`).
          - Finish is its own event: token events carry only `token_id`,
            finish events only `finish_reason`.
          - `>=` on the max_tokens check: monotone, can't be skipped past.
          - Finished sequences free their slot the same step.
        """
        events = self.pending_events
        self.pending_events = []

        self._promote_queued()

        rows = self._plan_step()
        if not rows:
            # Only pending events to flush (abort acks, rejections) — no
            # active sequences means no forward pass this step.
            return events

        input_ids = self._build_input_ids(rows)
        meta = build_batch_meta(rows)

        logits = self.forward_fn(input_ids, meta, self.pool)
        sampled = []
        for r, row in enumerate(rows):
            token_tensor, probs = sampler.sample(
                logits[r], row.sequence.sampling_params
            )
            token = int(token_tensor.item())
            sampled.append((token, probs[token].log().item()))

        still_active = []
        for row, (token, logprob) in zip(rows, sampled):
            seq = row.sequence
            seq.position += row.num_new
            if seq.is_prefilling():
                # we haven't finished prefilling yet, keep for next step
                # emit no event for this row and throw away the useless token
                still_active.append(seq)
                continue

            if token in seq.stop_token_ids:
                # we never emit stop tokens to clients
                events.append(
                    TokenEvent(finish_reason="end_turn", request_id=seq.request_id)
                )
                self.allocator.free(seq.slot_idx)
                continue

            seq.output_token_ids.append(token)
            events.append(TokenEvent(
                token_id=token, logprob=logprob, request_id=seq.request_id
            ))

            if len(seq.output_token_ids) >= seq.max_tokens:
                # finish is its own event, after the last token
                events.append(
                    TokenEvent(finish_reason="max_tokens", request_id=seq.request_id)
                )
                self.allocator.free(seq.slot_idx)
                continue

            still_active.append(seq)

        self.active = still_active
        return events

    def _promote_queued(self) -> None:
        """Admit queued sequences while slots are free (FCFS)."""
        while self.queued and self.allocator.num_free() > 0:
            seq = self.queued.popleft()
            seq.slot_idx = self.allocator.allocate()
            self.active.append(seq)

    def _plan_step(self) -> list[Row]:
        """Water-fill the token budget: decode rows request 1, prefilling
        rows request their remaining prompt."""
        requested = [
            seq.remaining_prompt if seq.is_prefilling() else 1
            for seq in self.active
        ]
        allocated = water_fill(self.config.max_tokens_per_step, requested)
        return [
            Row(sequence=seq, num_new=n, start_pos=seq.position)
            for seq, n in zip(self.active, allocated)
        ]

    def _build_input_ids(self, rows: list[Row]) -> torch.Tensor:
        """(B, num_new_max) int64, left-aligned, zero-padded."""
        width = max(row.num_new for row in rows)
        input_ids = torch.zeros((len(rows), width), dtype=torch.int64)
        for i, row in enumerate(rows):
            input_ids[i, : row.num_new] = torch.tensor(row.input_tokens, dtype=torch.int64)
        return input_ids
