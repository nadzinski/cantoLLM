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
from dataclasses import dataclass, field

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


@dataclass
class InFlightStep:
    """One launched-but-not-finalized step (overlap scheduling, §2.12).

    Holds everything reap needs one step later: the planned rows, the
    launch-time keep/discard decision per row (`emits`: the sample is a
    real token, not a mid-prefill throwaway; frozen at launch because
    positions keep moving), the device-resident sampled tokens (the next
    launch's decode inputs), the pinned host landing buffers their D2H
    targets on the side stream, and the CUDA event that says the copy
    arrived (None off-CUDA: the "copy" was synchronous). `dead` rows are
    skipped at reap: rolled back after a late-detected finish, or
    aborted while in flight. Their KV was already released, so reap
    releasing again would be the §4 double-free."""

    rows: list[Row]
    emits: list[bool]
    tokens_dev: torch.Tensor
    logprobs_dev: torch.Tensor
    tokens_host: torch.Tensor
    logprobs_host: torch.Tensor
    done: object | None
    row_index: dict[int, int]
    dead: set[int] = field(default_factory=set)


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
        # Monotonic preemption counters for the stats collector: total
        # evictions, and total tokens those victims must re-prefill on
        # resume (their replay prefixes).
        self.preemptions_total = 0
        self.preempted_tokens_total = 0
        # Overlap scheduling (§2.12): the one step whose forward is
        # enqueued but not yet finalized, and the lazily created side
        # stream its sampled-token D2H rides. None when overlap is off
        # or the machine is settled.
        self._in_flight: InFlightStep | None = None
        self._side_stream = None

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
            priority=request.priority,
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
            if self._in_flight is not None:
                # Abort-while-in-flight (§4): the pending sample must not
                # emit after this ack, and reap must not release again;
                # dead rows are skipped there entirely.
                self._in_flight.dead.add(id(seq))
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

    def _preempt_one(self) -> None:
        """Pick a victim by `config.preemption_policy` and evict it,
        counting the eviction and its recompute cost (the replay prefix
        the victim must re-prefill) for the stats collector.

        `lifo` is the hand-written chunk-9 machine; `priority` and `cost`
        are the author's chunk-10 session, red→green against
        tests/test_victim_policies.py. The selector stubs below carry
        the contracts."""
        policy = self.config.preemption_policy
        if policy == "lifo":
            victim = self.active[-1]
        elif policy == "priority":
            victim = self._select_victim_priority()
        else:
            victim = self._select_victim_cost()
        self.preemptions_total += 1
        self.preempted_tokens_total += (
            len(victim.prompt_token_ids) + len(victim.output_token_ids)
        )
        self._preempt_sequence(victim)

    def _select_victim_priority(self) -> CBSequence:
        """[HAND, chunk 10] The `priority` victim policy (§2.10): the
        lowest-priority active sequence, LIFO (newest-admitted) tiebreak
        among equal-lowest. Never picks a higher-priority row while a
        lower one is active."""
        victim = None
        for seq in self.active:
            if victim is None or victim.priority >= seq.priority:
                victim = seq
        return victim

    def _select_victim_cost(self) -> CBSequence:
        """[HAND, chunk 10] The `cost` victim policy (§2.10): the active
        sequence with the fewest KV tokens to lose, i.e. the cheapest
        recompute (the smallest consumed position)."""
        victim = None
        for seq in self.active:
            if victim is None or victim.position >= seq.position:
                victim = seq
        return victim

    def is_idle(self) -> bool:
        """Nothing queued, nothing active, nothing pending, nothing in
        flight. (The shell blocks while this is True; see the contract
        note up top. An in-flight step still owes its reap's events, so
        it must keep the loop stepping.)"""
        return (
            not self.queued and not self.active
            and not self.pending_events and self._in_flight is None
        )

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
        if self.config.overlap_scheduling:
            return self._step_overlapped()

        events = self.pending_events
        self.pending_events = []
        # Cleared up front so a no-forward step (pending flush only) never
        # reports the previous step's shape or token counts.
        self.last_forward_shape = None
        self.last_step_plan = None

        self._promote_queued()

        rows = self._plan_step()
        # Same-step retry under total starvation: evict, then REPLAN, so
        # the freed blocks go to the rows the eviction was meant to
        # unblock. Deferring the benefit to the next step livelocks: its
        # promotion runs first and hands the freed block straight back to
        # the re-admitted victim (found by chunk 10's swapped-arrival
        # policy tests; the loop is bounded by len(active), and a lone
        # survivor can always advance, so it terminates with rows or with
        # nothing active). Planning itself is the can-anyone-advance
        # oracle; eviction still emits nothing.
        while not rows and self.active:
            self._preempt_one()
            rows = self._plan_step() if self.active else []
        if not rows:
            # Only pending events to flush (abort acks, rejections) — no
            # forward pass this step.
            return events

        input_ids = self._build_input_ids(rows)
        input_ids, meta = self._stage_forward(rows, input_ids)

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

    def _stage_forward(
        self, rows: list[Row], input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, BatchMeta]:
        """Shared launch staging (extracted unchanged from the serial
        step() for the overlap split): meta, shape-vocabulary padding,
        paged table projection, plan-time stats."""
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
        return input_ids, meta

    def _step_overlapped(self) -> list[TokenEvent]:
        """The §2.12 split: LAUNCH this step's forward (planned from
        positions the previous launch already advanced, its decode inputs
        the previous step's still-GPU-resident samples), then REAP the
        previous step (wait its D2H event, finalize, emit). The CPU never
        blocks on the step it just enqueued, so planning k+1 overlaps
        forward k on the GPU; finalize runs one step late by design.

        Eviction stays a settled-state move: with a step in flight, a
        starved plan skips preemption for one step (the reap below may
        free what it needs; if not, the next call evicts with nothing in
        flight), so the §4 in-flight hazards never open."""
        events = self.pending_events
        self.pending_events = []
        self.last_forward_shape = None
        self.last_step_plan = None

        self._promote_queued()
        rows = self._plan_step()
        if not rows and self.active and self._in_flight is None:
            # Same-step evict-and-replan retry, exactly as the serial
            # loop: planning is the can-anyone-advance oracle.
            while not rows and self.active:
                self._preempt_one()
                rows = self._plan_step() if self.active else []

        launched = self._launch(rows) if rows else None
        if self._in_flight is not None:
            events.extend(self._reap(self._in_flight, launched))
        self._in_flight = launched
        return events

    def _launch(self, rows: list[Row]) -> InFlightStep:
        """Enqueue one forward + sampling + side-stream D2H; no host
        sync. Decode rows whose last token is still in flight get it
        scattered device-side into the input (§3); everyone else builds
        from host lists exactly like the serial path."""
        prev = self._in_flight
        width = max(row.num_new for row in rows)
        input_ids = torch.zeros((len(rows), width), dtype=torch.int64)
        scatter: list[tuple[int, int]] = []
        for i, row in enumerate(rows):
            seq = row.sequence
            src = prev.row_index.get(id(seq)) if prev is not None else None
            if src is not None and prev.emits[src] and row.num_new == 1:
                # Its decode input is the previous step's sample, still
                # device-resident and unread by the host.
                scatter.append((i, src))
            else:
                input_ids[i, : row.num_new] = torch.tensor(
                    row.input_tokens, dtype=torch.int64
                )

        # Advance positions at launch (the serial path advances at
        # finalize) so the next plan sees the committed state; freeze the
        # keep/discard decision per row now, while position is fresh.
        emits = []
        for row in rows:
            row.sequence.position += row.num_new
            emits.append(not row.sequence.is_prefilling())

        input_ids, meta = self._stage_forward(rows, input_ids)
        input_ids = input_ids.to(self.pool.device)
        if scatter:
            dst = torch.tensor([d for d, _ in scatter],
                               device=input_ids.device)
            src_idx = torch.tensor([s for _, s in scatter],
                                   device=input_ids.device)
            # One vectorized scatter, not a kernel per row (the Phase 2
            # launch-flood lesson).
            input_ids[dst, 0] = prev.tokens_dev[src_idx]

        logits = self.forward_fn(input_ids, meta, self.pool)
        toks, lps = [], []
        for r, row in enumerate(rows):
            token_tensor, probs = sampler.sample(
                logits[r], row.sequence.sampling_params
            )
            toks.append(token_tensor)
            lps.append(probs[token_tensor].log())
        tokens_dev = torch.stack(toks)
        logprobs_dev = torch.stack(lps)

        if tokens_dev.is_cuda:
            tokens_host = torch.empty_like(
                tokens_dev, device="cpu", pin_memory=True
            )
            logprobs_host = torch.empty_like(
                logprobs_dev, device="cpu", pin_memory=True
            )
            if self._side_stream is None:
                self._side_stream = torch.cuda.Stream()
            sampled = torch.cuda.Event()
            sampled.record(torch.cuda.current_stream())
            done = torch.cuda.Event()
            with torch.cuda.stream(self._side_stream):
                self._side_stream.wait_event(sampled)
                tokens_host.copy_(tokens_dev, non_blocking=True)
                logprobs_host.copy_(logprobs_dev, non_blocking=True)
                done.record(self._side_stream)
        else:
            # Off-CUDA the forward already ran synchronously; the reap's
            # "wait" degenerates to reading these directly.
            tokens_host, logprobs_host, done = tokens_dev, logprobs_dev, None

        return InFlightStep(
            rows=rows,
            emits=emits,
            tokens_dev=tokens_dev,
            logprobs_dev=logprobs_dev,
            tokens_host=tokens_host,
            logprobs_host=logprobs_host,
            done=done,
            row_index={id(row.sequence): i for i, row in enumerate(rows)},
        )

    def _reap(
        self, flight: InFlightStep, launched: InFlightStep | None
    ) -> list[TokenEvent]:
        """Finalize the previous launch: wait its D2H event, then the
        serial finalize semantics with positions already advanced. A row
        found finished here may have a bonus decode in `launched`; it
        rolls back (dead-marked, position decremented) and its whole
        table, bonus block included, returns via _release_kv."""
        if flight.done is not None:
            flight.done.synchronize()
        tokens = flight.tokens_host.tolist()
        logprobs = flight.logprobs_host.tolist()

        events: list[TokenEvent] = []
        for i, row in enumerate(flight.rows):
            seq = row.sequence
            if id(seq) in flight.dead or not flight.emits[i]:
                # Rolled back / aborted in flight (KV already released),
                # or a mid-prefill throwaway sample.
                continue
            token = tokens[i]
            if token in seq.stop_token_ids:
                events.append(TokenEvent(
                    finish_reason="end_turn", request_id=seq.request_id
                ))
                self._finish_in_flight(seq, launched)
                continue
            seq.output_token_ids.append(token)
            events.append(TokenEvent(
                token_id=token, logprob=logprobs[i],
                request_id=seq.request_id,
            ))
            if len(seq.output_token_ids) >= seq.max_tokens:
                events.append(TokenEvent(
                    finish_reason="max_tokens", request_id=seq.request_id
                ))
                self._finish_in_flight(seq, launched)
        return events

    def _finish_in_flight(
        self, seq: CBSequence, launched: InFlightStep | None
    ) -> None:
        """A reap discovered `seq` finished one step ago. If the current
        launch optimistically ran its bonus decode, roll it back: dead-
        mark (its sample must not emit), un-advance the position. Then
        release everything; the bonus block, if planning allocated one,
        goes back with the rest of the table."""
        if launched is not None:
            idx = launched.row_index.get(id(seq))
            if idx is not None:
                launched.dead.add(id(seq))
                seq.position -= launched.rows[idx].num_new
        self.active = [s for s in self.active if s is not seq]
        self._release_kv(seq)

    def _promote_queued(self) -> None:
        """Admit queued sequences while slots are free, highest priority
        first, FCFS within a priority (paged-kv-plan.md §2.10). Paged mode
        also takes the sequence's first block at admission, or stops
        promoting when the pool has none, leaving the queue intact
        (paged-kv-plan.md §3): an admitted sequence always has somewhere
        to write, and admission is what a full pool pushes back on."""
        if self.queued and any(s.priority for s in self.queued):
            # Stable sort by priority alone: deque order IS arrival order,
            # so equal priorities keep today's FCFS exactly, and a
            # preempted victim's appendleft stays "front of its priority
            # class". Skipped entirely at the all-default fast path.
            self.queued = deque(
                sorted(self.queued, key=lambda s: -s.priority)
            )
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
