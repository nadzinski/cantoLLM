"""Per-step engine statistics — the bench harness's engine-side view.

Observer-style by design (bench-spec.md §4): `StepStatsCollector` reads the
scheduler's public state (`queued`, `active`, `allocator`, `config`) around
`step()` calls in the engine shell; the scheduler itself is never modified
and never knows it is being measured. Schedulers that don't expose that
surface (scripted/toy test doubles) get no collector and the shell behaves
exactly as before.

`StepUpdate` is what crosses from the drive loop to the API side — the
step's token events plus (optionally) its stats, still one message per
step. `EngineStatsAccumulator` lives on the API side (`EventMultiplexer`),
keeps bounded rings of steps and engine-ITL samples, and serves the
`/debug/engine-stats` scrapes with a `since` cursor.

Derivation notes: the real scheduler states its per-step row/token counts
at plan time (`last_step_plan`, Phase 4 chunk 1) and the collector prefers
them — the older snapshot-diff reconstruction leaned on two invariants
("a row can only finish once past prefill", "finished rows free their
slot within the same step") that Phase 4's preemption and overlap break.
The reconstruction is kept as the fallback for schedulers that expose the
observed surface but not the plan-time counters:
  - a row that finished consumed either 1 decode token or, when it
    completed its final prefill chunk and stopped immediately, its
    remaining prompt;
  - post-step `active` plus this step's row-finish events reconstructs the
    forward's row count. Abort acks and zero-token rejections are *pending*
    events, flushed at the head of `step()`'s return — the collector
    snapshots `len(pending_events)` beforehand so they are never mistaken
    for rows.
KV capacity fields: padded pools derive (allocated, capacity) from slots
here in the collector; the paged scheduler will expose `kv_state`
(allocated_tokens, capacity_tokens) and the collector prefers it.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass, field, replace

from cantollm.engine.types import TokenEvent

# v2 (Phase 4 chunk 1): additive — kv_allocated_tokens / kv_capacity_tokens
# optional fields, and prefill/decode preferred from the scheduler's
# plan-time counters over the snapshot-diff reconstruction.
# v3 (Phase 4 chunk 10): additive; preemptions / preempted_tokens optional
# per-step counters, diffed from the scheduler's monotonic totals; None
# when the scheduler predates them, 0 on any step without an eviction.
STATS_SCHEMA_VERSION = 3

STEP_RING_SIZE = 4096
ITL_RING_SIZE = 65536

# Finish reasons that mean "this sequence was a row in the step's forward".
_ROW_FINISH_REASONS = ("end_turn", "max_tokens")


@dataclass(frozen=True)
class StepStats:
    """One scheduler step, as seen from the engine shell."""

    seq: int              # engine-lifetime monotonic step counter
    t_wall: float         # time.time() at step end — coarse cross-process alignment only
    t_perf: float         # perf_counter() at step end — ITL deltas; engine-clock only
    dur_s: float          # perf_counter time spent inside scheduler.step()
    rows: int             # sequences in this step's forward pass
    occupied_slots: int   # KV slots held after the step
    queue_depth: int      # sequences waiting before the step (post command drain)
    kv_tokens: int        # sum of active sequences' positions after the step
    prefill_tokens: int   # prompt tokens consumed by this step's forward
    decode_tokens: int    # decode tokens consumed by this step's forward
    fwd_rows: int | None = None    # forward's batch dim (incl. filler rows)
    fwd_width: int | None = None   # forward's num_new_max (incl. pad columns)
    fwd_kv_len: int | None = None  # forward's max_history_len (post-bucketing)
    graph_replayed: bool | None = None  # this step rode a CUDA graph; None = graphs off
    # Pool-memory view (schema v2). kv_tokens above stays "tokens in use";
    # these are what the pool has RESERVED for them and its total, so
    # utilization stops assuming slot layout. Padded: occupied_slots *
    # max_seq_len / max_batch * max_seq_len. Paged: blocks * block_size.
    kv_allocated_tokens: int | None = None
    kv_capacity_tokens: int | None = None
    # Preemption view (schema v3): evictions this step fired and the
    # tokens those victims must re-prefill on resume (recompute cost,
    # §6 prediction 4's denominator). Zero-cost when nothing evicts.
    preemptions: int | None = None
    preempted_tokens: int | None = None


@dataclass(frozen=True)
class StepUpdate:
    """Per-step message from the drive loop: events plus optional stats.

    Replaces the bare `list[TokenEvent]` on the process-split wire (see
    process-split-design.md) and on the in-process dispatch hop. Still one
    emission per step, never per token.
    """

    events: list[TokenEvent]
    stats: StepStats | None = None


class StepStatsCollector:
    """Snapshots scheduler state around `step()` in the drive loop.

    Stateful and single-threaded by construction: only the scheduler
    thread/process touches it, strictly before_step → step() → after_step.
    """

    def __init__(self) -> None:
        self._seq = 0
        self._snapshot: dict[str, tuple[int, int]] = {}
        self._pending_count = 0
        self._queue_depth = 0
        self._graph_hits: int | None = None
        self._preemptions: int | None = None
        self._preempted_tokens: int | None = None
        self._t0 = 0.0

    @classmethod
    def for_scheduler(cls, scheduler) -> "StepStatsCollector | None":
        """A collector, or None when the scheduler doesn't expose the real
        scheduler's public surface (scripted/toy doubles in tests)."""
        needed = ("queued", "active", "allocator", "config", "pending_events")
        if all(hasattr(scheduler, attr) for attr in needed):
            return cls()
        return None

    def before_step(self, scheduler) -> None:
        # Queued sequences are included: promotion happens inside step(),
        # and a promoted row's pre-step position is its queued position (0).
        self._snapshot = {
            seq.request_id: (seq.position, len(seq.prompt_token_ids))
            for seq in (*scheduler.active, *scheduler.queued)
        }
        self._pending_count = len(scheduler.pending_events)
        self._queue_depth = len(scheduler.queued)
        # The graphed forward keeps cumulative hit/miss counters (the SDPA
        # tripwire lesson: fast paths that can silently fall back need
        # proof they ran); snapshotting hits around step() turns them into
        # a per-step replayed/eager flag.
        self._graph_hits = getattr(
            getattr(scheduler, "forward_fn", None), "hits", None
        )
        # Monotonic eviction totals (chunk 10), diffed after the step the
        # same way graph hits are: preemption happens inside step().
        self._preemptions = getattr(scheduler, "preemptions_total", None)
        self._preempted_tokens = getattr(
            scheduler, "preempted_tokens_total", None
        )
        self._t0 = time.perf_counter()

    def after_step(self, scheduler, events: list[TokenEvent]) -> StepStats:
        t_perf = time.perf_counter()
        dur_s = t_perf - self._t0

        # Events past the flushed-pending prefix were produced by this
        # step's forward; only those can be row finishes.
        step_events = events[self._pending_count:]
        finished = {
            e.request_id
            for e in step_events
            if e.finish_reason in _ROW_FINISH_REASONS
        }

        kv_tokens = sum(seq.position for seq in scheduler.active)
        planned = getattr(scheduler, "last_step_plan", None)
        if planned is not None:
            # The real scheduler states its counts at plan time (Phase 4
            # chunk 1) — exact under preemption/overlap, where the
            # snapshot-diff below misattributes.
            rows, prefill, decode = planned
        else:
            rows = len(scheduler.active) + len(finished)
            prefill = 0
            decode = 0
            for seq in scheduler.active:
                pre_pos, prompt_len = self._snapshot.get(
                    seq.request_id, (0, len(seq.prompt_token_ids))
                )
                consumed_prefill = (
                    min(seq.position, prompt_len) - min(pre_pos, prompt_len)
                )
                prefill += consumed_prefill
                decode += (seq.position - pre_pos) - consumed_prefill
            for request_id in finished:
                pre_pos, prompt_len = self._snapshot[request_id]
                if pre_pos >= prompt_len:
                    decode += 1
                else:
                    # Completed its final prefill chunk, stopped immediately.
                    prefill += prompt_len - pre_pos

        max_batch = scheduler.config.max_batch
        occupied = max_batch - scheduler.allocator.num_free()
        # (allocated, capacity) pool tokens: the paged scheduler exposes
        # kv_state (blocks are its unit of reservation); padded reservation
        # is whole slots, derivable right here.
        kv_state = getattr(scheduler, "kv_state", None)
        if kv_state is not None:
            kv_allocated, kv_capacity = kv_state
        else:
            kv_allocated = occupied * scheduler.config.max_seq_len
            kv_capacity = max_batch * scheduler.config.max_seq_len
        # (B, num_new_max, max_history_len) of the forward this step ran —
        # the shape the kernel saw, which bucketing changes and `rows`
        # (real sequences) cannot reconstruct. None when no forward ran.
        fwd_shape = getattr(scheduler, "last_forward_shape", None) or (None,) * 3
        hits_now = getattr(
            getattr(scheduler, "forward_fn", None), "hits", None
        )
        graph_replayed = (
            None if hits_now is None or self._graph_hits is None
            else hits_now > self._graph_hits
        )
        preempt_now = getattr(scheduler, "preemptions_total", None)
        preemptions = (
            None if preempt_now is None or self._preemptions is None
            else preempt_now - self._preemptions
        )
        preempted_now = getattr(scheduler, "preempted_tokens_total", None)
        preempted_tokens = (
            None if preempted_now is None or self._preempted_tokens is None
            else preempted_now - self._preempted_tokens
        )
        stats = StepStats(
            seq=self._seq,
            t_wall=time.time(),
            t_perf=t_perf,
            dur_s=dur_s,
            rows=rows,
            occupied_slots=occupied,
            queue_depth=self._queue_depth,
            kv_tokens=kv_tokens,
            prefill_tokens=prefill,
            decode_tokens=decode,
            fwd_rows=fwd_shape[0],
            fwd_width=fwd_shape[1],
            fwd_kv_len=fwd_shape[2],
            graph_replayed=graph_replayed,
            kv_allocated_tokens=kv_allocated,
            kv_capacity_tokens=kv_capacity,
            preemptions=preemptions,
            preempted_tokens=preempted_tokens,
        )
        self._seq += 1
        return stats


@dataclass(frozen=True)
class ITLSample:
    """One engine-side inter-token gap for one request (bench-spec.md §4)."""

    seq: int      # step at which the gap closed
    request_id: str
    gap_s: float  # t_perf delta between this and the previous token-bearing step


@dataclass
class EngineStatsAccumulator:
    """API-side ring buffers behind /debug/engine-stats.

    Loop-confined: `record` runs only on the event loop (both engines hop
    via call_soon_threadsafe), `read` runs in route handlers on the same
    loop — no locks needed.
    """

    engine_kind: str = "unknown"
    max_batch: int | None = None
    max_seq_len: int | None = None
    load_seconds: float | None = None
    # Watchdog input: wall (monotonic) time of the last recorded update.
    # Reset by the watchdog when it arms, so a request submitted right
    # after READY never compares against a stale pre-warm stamp.
    last_update_mono: float = field(default_factory=time.monotonic)

    _steps: deque = field(default_factory=lambda: deque(maxlen=STEP_RING_SIZE))
    _itl: deque = field(default_factory=lambda: deque(maxlen=ITL_RING_SIZE))
    _last_token_t: dict[str, float] = field(default_factory=dict)
    _total_steps: int = 0
    _total_output_tokens: int = 0
    # Seq rebase across engine generations: a restarted child restarts its
    # step counter at 0, which would make every `since` cursor filter the
    # new steps out and silently blank bench scrapes. The lifecycle handle
    # calls note_generation_start() when it injects this accumulator into
    # a fresh engine; recorded seqs stay monotonic across restarts.
    _seq_base: int = 0
    _last_seen_seq: int = -1
    # Optional push hook (Phase 3.5 /metrics): called once per record with
    # the (rebased) StepStats or None and the newly derived ITL gaps. Set
    # by the observability layer; the engine stays ignorant of Prometheus.
    on_record: object = None

    def note_generation_start(self) -> None:
        self._seq_base = self._last_seen_seq + 1

    def record(self, update: StepUpdate) -> None:
        self.last_update_mono = time.monotonic()
        stats = update.stats
        new_gaps: list[float] = []
        if stats is not None:
            if self._seq_base:
                stats = replace(stats, seq=stats.seq + self._seq_base)
            self._last_seen_seq = stats.seq
            self._steps.append(stats)
            self._total_steps += 1
        for evt in update.events:
            if evt.token_id is not None:
                self._total_output_tokens += 1
                if stats is not None:
                    last = self._last_token_t.get(evt.request_id)
                    if last is not None:
                        gap = stats.t_perf - last
                        self._itl.append(ITLSample(
                            seq=stats.seq,
                            request_id=evt.request_id,
                            gap_s=gap,
                        ))
                        new_gaps.append(gap)
                    self._last_token_t[evt.request_id] = stats.t_perf
            if evt.finish_reason is not None or evt.error is not None:
                self._last_token_t.pop(evt.request_id, None)
        if self.on_record is not None:
            self.on_record(stats, new_gaps)

    def recent_decode_rate(self, window: int = 100) -> float | None:
        """Decode tokens/sec over the last `window` steps. None without at
        least two steps, or when the window straddles a generation boundary
        (child perf_counters restart, making the span meaningless)."""
        steps = list(self._steps)[-window:]
        if len(steps) < 2:
            return None
        span = steps[-1].t_perf - steps[0].t_perf
        if span <= 0:
            return None
        return sum(s.decode_tokens for s in steps[1:]) / span

    def read(self, since: int = -1) -> dict:
        steps = [asdict(s) for s in self._steps if s.seq > since]
        itl = [asdict(s) for s in self._itl if s.seq > since]
        next_since = steps[-1]["seq"] if steps else since
        return {
            "engine_kind": self.engine_kind,
            "load_seconds": self.load_seconds,
            "capacity": {
                "max_batch": self.max_batch,
                "max_seq_len": self.max_seq_len,
            },
            "totals": {
                "steps": self._total_steps,
                "output_tokens": self._total_output_tokens,
            },
            "steps": steps,
            "itl": itl,
            "next_since": next_since,
        }
