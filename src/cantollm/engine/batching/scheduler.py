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

from cantollm.engine import sampler
from cantollm.engine.batching.allocator import BlockAllocator, SlotAllocator
from cantollm.engine.batching.config import BatchingConfig
from cantollm.engine.batching.paging import PagedStepState
from cantollm.engine.batching.shaping import shape_step
from cantollm.engine.batching.types import BatchedForwardFn, CBSequence
from cantollm.engine.types import InferenceRequest, TokenEvent
from cantollm.kv_pool import KVPool
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


def build_batch_meta(
    rows: list[Row], device: torch.device | None = None
) -> BatchMeta:
    """Per-step geometry from planned rows (see BatchMeta's docstrings).

    `device` is where `meta.kv_write_map`'s index tensors land — pass the
    KV pool's device so the per-step upload happens once, not per layer.
    """
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
        device=device,
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
        pool: KVPool,
        allocator: SlotAllocator,
        config: BatchingConfig,
        *,
        block_allocator: BlockAllocator | None = None,
        paged_state: PagedStepState | None = None,
    ):
        # Slot-count check is padded-layout-specific (the paged pool sizes
        # memory in blocks, not slots); its capacity checks live below.
        pool_slots = getattr(pool, "max_batch", None)
        if pool_slots is not None and pool_slots != config.max_batch:
            raise ValueError(
                f"pool has {pool_slots} slots but config.max_batch is "
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
        if config.paged_kv != (block_allocator is not None) or (
            config.paged_kv != (paged_state is not None)
        ):
            raise ValueError(
                "paged_kv configs take a block_allocator and a paged_state; "
                "padded configs take neither"
            )
        if block_allocator is not None:
            if block_allocator.num_blocks != config.resolved_kv_blocks:
                raise ValueError(
                    f"block allocator manages {block_allocator.num_blocks} "
                    f"blocks but the config resolves "
                    f"{config.resolved_kv_blocks}"
                )
            pool_block_size = getattr(pool, "block_size", None)
            if pool_block_size != config.block_size:
                raise ValueError(
                    f"pool block size {pool_block_size} does not match "
                    f"config.block_size {config.block_size}"
                )
        self.forward_fn = forward_fn
        self.pool = pool
        self.allocator = allocator
        self.block_allocator = block_allocator
        self.paged_state = paged_state
        self.config = config
        self.queued: deque[CBSequence] = deque()
        self.active: list[CBSequence] = []
        self.pending_events: list[TokenEvent] = []

    @property
    def kv_state(self) -> tuple[int, int] | None:
        """(allocated, capacity) pool tokens for the stats collector, in
        the paged pool's unit of reservation (whole blocks). None on the
        padded path, where the collector derives slot-based numbers itself."""
        if self.block_allocator is None:
            return None
        block_size = self.config.block_size
        return (
            self.block_allocator.num_allocated() * block_size,
            self.block_allocator.num_blocks * block_size,
        )

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
            self._release_kv(seq)
        else:
            seq = next((s for s in self.queued if s.request_id == request_id), None)
            if seq is None:
                return
            self.queued.remove(seq)
        self.pending_events.append(
            TokenEvent(finish_reason="abort", request_id=request_id)
        )

    def _release_kv(self, seq: CBSequence) -> None:
        """Return everything an active sequence holds: its slot, and on
        the paged path every block in its table. Queued sequences hold
        neither (promotion is where both are acquired)."""
        self.allocator.free(seq.slot_idx)
        if self.block_allocator is not None:
            for block in seq.block_table:
                self.block_allocator.free(block)
            seq.block_table.clear()

    def _preempt_sequence(self, seq: CBSequence) -> None:
        """Evict ``seq`` and queue it to rebuild its KV cache later.

        Preserve the client-visible output ledger, but reset the model-side
        state: resumption re-prefills the original prompt plus every token
        already emitted. Preemption itself emits no client event.
        """
        self.active.remove(seq)
        self._release_kv(seq)
        seq.slot_idx = None
        seq.replay_prefix_token_ids = seq.prompt_token_ids + seq.output_token_ids
        seq.position = 0
        # Resume evicted work before admitting requests that arrived later.
        self.queued.appendleft(seq)

    def _preempt_lifo(self) -> None:
        """Preempt the most recently admitted active sequence."""
        self._preempt_sequence(self.active[-1])

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
        # Cleared up front so a no-forward step (pending flush only) never
        # reports the previous step's shape or token counts.
        self.last_forward_shape = None
        self.last_step_plan = None

        self._promote_queued()

        rows = self._plan_step()
        if not rows:
            if self.active:
                self._preempt_lifo()
            # Only pending events to flush (abort acks, rejections) — no
            # forward pass this step. Preemption also emits nothing.
            return events

        input_ids = self._build_input_ids(rows)
        meta = build_batch_meta(rows, device=self.pool.device)
        # Pad the planned geometry into the bounded shape vocabulary (an
        # exact no-op without the bucket knobs) — see shaping.py for why
        # kernels care about step shapes.
        input_ids, meta = shape_step(input_ids, meta, self.config)
        if self.paged_state is not None:
            # Project the host block tables into the step's device tensors
            # and hand the meta its references (paged-kv-plan.md §2.5);
            # filler rows appended by shape_step get the scratch block.
            # The width names the mask family (chunk 6): fill returns the
            # family's cached BlockMask with the tables.
            meta.seed_paged_tables(self.paged_state.fill(
                meta.rows,
                [row.sequence.block_table for row in rows],
                self.config.block_size,
                meta.num_new_max,
            ))

        # The forward's actual problem shape (post-bucketing), for the
        # stats collector — StepStats.rows counts real sequences only.
        self.last_forward_shape = (
            len(meta.rows), meta.num_new_max, meta.max_history_len
        )
        # (rows, prefill, decode) of this forward, stated at plan time for
        # the stats collector: the snapshot-diff derivation it replaces
        # leaned on invariants ("finish implies past prefill", "finished
        # rows free slots in-step") that preemption and overlap break in
        # Phase 4. A chunk never straddles the prompt boundary, so a row
        # is all-prefill or all-decode.
        prefill = sum(
            r.num_new for r in rows
            if r.start_pos < len(r.sequence.prefill_token_ids)
        )
        self.last_step_plan = (
            len(rows), prefill, sum(r.num_new for r in rows) - prefill
        )

        logits = self.forward_fn(input_ids, meta, self.pool)

        # Optimization: we don't pull data back onto CPU until the end, as
        # item() in the loop was causing the CPU to wait for the GPU, starving
        # the GPU of future work
        toks, lps = [], []
        for r, row in enumerate(rows):
            token_tensor, probs = sampler.sample(
                logits[r], row.sequence.sampling_params
            )
            toks.append(token_tensor)
            lps.append(probs[token_tensor].log())

        tokens = torch.stack(toks).cpu().tolist()
        log_probs = torch.stack(lps).cpu().tolist()
        sampled = list(zip(tokens, log_probs))

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
                self._release_kv(seq)
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
                self._release_kv(seq)
                continue

            still_active.append(seq)

        # Drop exactly the sequences whose rows finished this step. A row
        # that never made it into `rows` (paged block starvation trimmed
        # its grant to zero) stays active, in its place, and retries next
        # step. Padded rows always run, so this is the same assignment
        # `still_active` used to be.
        finished = (
            {id(row.sequence) for row in rows}
            - {id(seq) for seq in still_active}
        )
        self.active = [s for s in self.active if id(s) not in finished]
        return events

    def _promote_queued(self) -> None:
        """Admit queued sequences while slots are free (FCFS). Paged mode
        also takes the sequence's first block at admission, or stops
        promoting when the pool has none, leaving the queue intact
        (paged-kv-plan.md §3): an admitted sequence always has somewhere
        to write, and admission is what a full pool pushes back on."""
        while self.queued and self.allocator.num_free() > 0:
            if self.block_allocator is not None:
                first = self.block_allocator.allocate()
                if first is None:
                    return
                self.queued[0].block_table.append(first)
            seq = self.queued.popleft()
            seq.slot_idx = self.allocator.allocate()
            self.active.append(seq)

    def _plan_step(self) -> list[Row]:
        """Water-fill the token budget: decode rows request 1, prefilling
        rows request their remaining prompt. With a `prefill_widths` menu,
        mid-prompt chunks are then quantized down to menu widths. Paged
        mode then reserves each row's blocks, trimming grants the pool
        cannot back; a row trimmed to nothing sits this step out but
        stays active (see step())."""
        requested = [
            seq.remaining_prompt if seq.is_prefilling() else 1
            for seq in self.active
        ]
        allocated = water_fill(self.config.max_tokens_per_step, requested)
        if self.config.prefill_widths is not None:
            allocated = [
                self._quantize_chunk(seq, n)
                for seq, n in zip(self.active, allocated)
            ]
        if self.block_allocator is not None:
            allocated = [
                self._reserve_blocks(seq, n)
                for seq, n in zip(self.active, allocated)
            ]
        return [
            Row(sequence=seq, num_new=n, start_pos=seq.position)
            for seq, n in zip(self.active, allocated)
            if n >= 1
        ]

    def _reserve_blocks(self, seq: CBSequence, num_new: int) -> int:
        """Grow `seq.block_table` to cover `position + num_new`; return the
        grant, trimmed to what reserved blocks can actually hold.

        The plan/allocate atomicity rule (paged-kv-plan.md §4): a row
        commits to the step only with every write destination reserved.
        On exhaustion the grant shrinks to the covered span: a prefill
        chunk gets shorter (possibly off the width menu; the step width
        still pads up over it), a boundary-crossing decode row shrinks to
        zero and sits the step out with its blocks kept (§9.6's
        self-victimize call). Nothing is preempted here; eviction under
        sustained shortage is chunk 9's."""
        block_size = self.config.block_size
        end = seq.position + num_new
        while len(seq.block_table) * block_size < end:
            block = self.block_allocator.allocate()
            if block is None:
                covered = len(seq.block_table) * block_size - seq.position
                return max(0, covered)
            seq.block_table.append(block)
        return num_new

    def _quantize_chunk(self, seq: CBSequence, allocated: int) -> int:
        """Snap a mid-prompt chunk down to the largest menu width that fits.

        Exemptions keep this a pure narrowing: decode rows (1 token), the
        final chunk of a prompt (takes exactly what remains — the step
        width rounds up over it), and tight-budget allocations below the
        smallest menu width (rare; the width padding still bounds the
        step's shape). Freed budget is not redistributed — a menu-width
        chunk next step picks up the slack.
        """
        if not seq.is_prefilling() or allocated <= 1:
            return allocated
        if allocated >= seq.remaining_prompt:
            return allocated  # final chunk: exact, padded by the step width
        menu = self.config.prefill_widths
        fitting = [w for w in menu if w <= allocated]
        return fitting[-1] if fitting else allocated

    def _build_input_ids(self, rows: list[Row]) -> torch.Tensor:
        """(B, num_new_max) int64, left-aligned, zero-padded."""
        width = max(row.num_new for row in rows)
        input_ids = torch.zeros((len(rows), width), dtype=torch.int64)
        for i, row in enumerate(rows):
            input_ids[i, : row.num_new] = torch.tensor(row.input_tokens, dtype=torch.int64)
        return input_ids
