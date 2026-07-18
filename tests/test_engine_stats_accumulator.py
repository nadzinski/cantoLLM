"""EngineStatsAccumulator tests: rings, since-cursor, engine-ITL derivation.

Pure-Python, synthetic StepUpdates — no scheduler, no torch. The ITL rule
under test (bench-spec.md §4): a request's gap closes on each step that
emitted a token for it, measured on the engine perf clock; first tokens
open no gap; finish/error clears the request's state.
"""

from cantollm.engine.batching.stats import (
    STEP_RING_SIZE,
    EngineStatsAccumulator,
    StepStats,
    StepUpdate,
)
from cantollm.engine.types import TokenEvent


def step_stats(seq: int, t_perf: float, **overrides) -> StepStats:
    defaults = dict(
        seq=seq, t_wall=1000.0 + t_perf, t_perf=t_perf, dur_s=0.01,
        rows=1, occupied_slots=1, queue_depth=0, kv_tokens=5,
        prefill_tokens=0, decode_tokens=1,
    )
    defaults.update(overrides)
    return StepStats(**defaults)


def token(rid: str, tid: int = 7) -> TokenEvent:
    return TokenEvent(token_id=tid, request_id=rid)


def finish(rid: str, reason: str = "end_turn") -> TokenEvent:
    return TokenEvent(finish_reason=reason, request_id=rid)


def test_itl_gaps_per_request():
    acc = EngineStatsAccumulator()
    acc.record(StepUpdate(events=[token("a")], stats=step_stats(0, 1.0)))
    acc.record(StepUpdate(events=[token("a"), token("b")], stats=step_stats(1, 1.5)))
    acc.record(StepUpdate(events=[token("a"), token("b")], stats=step_stats(2, 2.5)))

    out = acc.read()
    gaps = [(s["request_id"], round(s["gap_s"], 6)) for s in out["itl"]]
    # a: first token at 1.0 opens no gap; then 0.5 and 1.0.
    # b: first token at 1.5; then 1.0.
    assert gaps == [("a", 0.5), ("a", 1.0), ("b", 1.0)]
    assert out["totals"] == {"steps": 3, "output_tokens": 5}


def test_finish_resets_request_state():
    acc = EngineStatsAccumulator()
    acc.record(StepUpdate(events=[token("a")], stats=step_stats(0, 1.0)))
    acc.record(StepUpdate(events=[token("a"), finish("a")], stats=step_stats(1, 1.4)))
    # Same rid reappears (new request reusing an id): no gap across the finish.
    acc.record(StepUpdate(events=[token("a")], stats=step_stats(2, 5.0)))
    acc.record(StepUpdate(events=[token("a")], stats=step_stats(3, 5.2)))

    gaps = [round(s["gap_s"], 6) for s in acc.read()["itl"]]
    assert gaps == [0.4, 0.2]


def test_error_resets_request_state():
    acc = EngineStatsAccumulator()
    acc.record(StepUpdate(events=[token("a")], stats=step_stats(0, 1.0)))
    acc.record(StepUpdate(
        events=[TokenEvent(error="boom", request_id="a")], stats=step_stats(1, 1.2),
    ))
    acc.record(StepUpdate(events=[token("a")], stats=step_stats(2, 9.0)))
    assert acc.read()["itl"] == []


def test_since_cursor_pages_and_next_since_advances():
    acc = EngineStatsAccumulator()
    for i in range(5):
        acc.record(StepUpdate(events=[token("a")], stats=step_stats(i, float(i))))

    first = acc.read(since=-1)
    assert [s["seq"] for s in first["steps"]] == [0, 1, 2, 3, 4]
    assert first["next_since"] == 4

    again = acc.read(since=first["next_since"])
    assert again["steps"] == [] and again["itl"] == []
    assert again["next_since"] == 4

    acc.record(StepUpdate(events=[token("a")], stats=step_stats(5, 5.0)))
    fresh = acc.read(since=4)
    assert [s["seq"] for s in fresh["steps"]] == [5]
    assert [s["seq"] for s in fresh["itl"]] == [5]


def test_ring_overflow_shows_as_seq_gap():
    acc = EngineStatsAccumulator()
    for i in range(STEP_RING_SIZE + 10):
        acc.record(StepUpdate(events=[], stats=step_stats(i, float(i))))

    steps = acc.read()["steps"]
    assert len(steps) == STEP_RING_SIZE
    assert steps[0]["seq"] == 10          # oldest 10 evicted → visible gap from -1
    assert acc.read()["totals"]["steps"] == STEP_RING_SIZE + 10


def test_events_without_stats_count_tokens_only():
    # Toy schedulers get no collector: updates carry stats=None.
    acc = EngineStatsAccumulator()
    acc.record(StepUpdate(events=[token("a"), token("a")], stats=None))
    out = acc.read()
    assert out["steps"] == [] and out["itl"] == []
    assert out["totals"] == {"steps": 0, "output_tokens": 2}


def test_metadata_fields_surface_in_read():
    acc = EngineStatsAccumulator(
        engine_kind="batched-split", max_batch=8, max_seq_len=4096,
    )
    acc.load_seconds = 3.25
    out = acc.read()
    assert out["engine_kind"] == "batched-split"
    assert out["capacity"] == {"max_batch": 8, "max_seq_len": 4096}
    assert out["load_seconds"] == 3.25
