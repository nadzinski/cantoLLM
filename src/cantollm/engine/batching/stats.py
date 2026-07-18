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

Derivation notes (the two scheduler invariants this borrows — revisit for
Phase 4 paged KV):
  - a row can only finish (`end_turn`/`max_tokens`) once past prefill, so a
    finished row consumed either 1 decode token or, when it completed its
    final prefill chunk and stopped immediately, its remaining prompt;
  - finished rows free their slot within the same step, so post-step
    `active` plus this step's row-finish events reconstructs the forward's
    row count. Abort acks and zero-token rejections are *pending* events,
    flushed at the head of `step()`'s return — the collector snapshots
    `len(pending_events)` beforehand so they are never mistaken for rows.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass, field

from cantollm.engine.types import TokenEvent

STATS_SCHEMA_VERSION = 1

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

        prefill = 0
        decode = 0
        kv_tokens = 0
        for seq in scheduler.active:
            pre_pos, prompt_len = self._snapshot.get(
                seq.request_id, (0, len(seq.prompt_token_ids))
            )
            consumed_prefill = min(seq.position, prompt_len) - min(pre_pos, prompt_len)
            prefill += consumed_prefill
            decode += (seq.position - pre_pos) - consumed_prefill
            kv_tokens += seq.position
        for request_id in finished:
            pre_pos, prompt_len = self._snapshot[request_id]
            if pre_pos >= prompt_len:
                decode += 1
            else:
                # Completed its final prefill chunk and stopped immediately.
                prefill += prompt_len - pre_pos

        max_batch = scheduler.config.max_batch
        stats = StepStats(
            seq=self._seq,
            t_wall=time.time(),
            t_perf=t_perf,
            dur_s=dur_s,
            rows=len(scheduler.active) + len(finished),
            occupied_slots=max_batch - scheduler.allocator.num_free(),
            queue_depth=self._queue_depth,
            kv_tokens=kv_tokens,
            prefill_tokens=prefill,
            decode_tokens=decode,
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

    _steps: deque = field(default_factory=lambda: deque(maxlen=STEP_RING_SIZE))
    _itl: deque = field(default_factory=lambda: deque(maxlen=ITL_RING_SIZE))
    _last_token_t: dict[str, float] = field(default_factory=dict)
    _total_steps: int = 0
    _total_output_tokens: int = 0

    def record(self, update: StepUpdate) -> None:
        stats = update.stats
        if stats is not None:
            self._steps.append(stats)
            self._total_steps += 1
        for evt in update.events:
            if evt.token_id is not None:
                self._total_output_tokens += 1
                if stats is not None:
                    last = self._last_token_t.get(evt.request_id)
                    if last is not None:
                        self._itl.append(ITLSample(
                            seq=stats.seq,
                            request_id=evt.request_id,
                            gap_s=stats.t_perf - last,
                        ))
                    self._last_token_t[evt.request_id] = stats.t_perf
            if evt.finish_reason is not None or evt.error is not None:
                self._last_token_t.pop(evt.request_id, None)

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
