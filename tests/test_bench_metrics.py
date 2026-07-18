"""Bench math: percentiles (vs numpy), CV, repeat summaries, medians."""

import numpy as np
import pytest

from cantollm.bench.metrics import (
    cv,
    median_across_repeats,
    percentile,
    summarize_repeat,
)
from cantollm.bench.records import RequestRecord


def rec(i: int, *, t0: float = 0.0, ttft: float = 0.5, dur: float = 2.0,
        out: int = 10, error: str | None = None, excluded: bool = False,
        finish: str = "length", scheduled: float | None = None) -> RequestRecord:
    r = RequestRecord(
        cell_id="c", repeat=0, request_index=i, prompt_id=f"p{i}",
        dialect="openai", t_scheduled=scheduled, t_send=t0,
        t_headers=t0 + 0.01,
        t_first_token=None if error else t0 + ttft,
        t_done=None if error else t0 + dur,
        output_tokens=0 if error else out,
        finish_reason=None if error else finish,
        error=error, excluded=excluded,
    )
    return r.finalize()


def test_percentile_matches_numpy_linear_interpolation():
    rng = np.random.default_rng(0)
    values = rng.uniform(0, 10, size=97).tolist()
    for q in (0.0, 0.25, 0.5, 0.9, 0.99, 1.0):
        assert percentile(values, q) == pytest.approx(
            float(np.percentile(values, q * 100, method="linear"))
        )
    assert np.isnan(percentile([], 0.5))


def test_cv():
    assert cv([10.0, 10.0, 10.0]) == 0.0
    assert cv([1.0]) != cv([1.0])  # nan
    assert cv([9.0, 10.0, 11.0]) == pytest.approx(0.1)


def test_finalize_derives_client_metrics():
    r = rec(0, t0=100.0, ttft=0.4, dur=2.4, out=11, scheduled=99.9)
    assert r.ttft_s == pytest.approx(0.4)
    assert r.completion_s == pytest.approx(2.4)
    assert r.accept_s == pytest.approx(0.01)
    assert r.client_itl_mean_s == pytest.approx(2.0 / 10)
    assert r.dispatch_lag_s == pytest.approx(0.1)


def test_summarize_repeat_client_side():
    records = [rec(i, t0=float(i), dur=2.0, out=10) for i in range(4)]
    records.append(rec(9, t0=0.0, error="boom"))
    records.append(rec(10, t0=0.0, excluded=True))   # warmup: invisible

    s = summarize_repeat(0, records)
    assert (s.n_requests, s.n_ok, s.n_errors) == (5, 4, 1)
    # Window: first send t=0 → last done t=3+2=5; 40 tokens.
    assert s.wall_s == pytest.approx(5.0)
    assert s.aggregate_tok_s == pytest.approx(8.0)
    assert s.ttft_p50 == pytest.approx(0.5)
    assert s.finish_reasons == {"length": 4}
    assert any("error rate" in w for w in s.warnings)  # 1/5 = 20% > 1%


def test_summarize_repeat_engine_side_and_fixed_length_check():
    records = [rec(i, finish="stop") for i in range(3)]
    steps = [
        {"dur_s": 0.1, "occupied_slots": 2, "kv_tokens": 40, "queue_depth": 1},
        {"dur_s": 0.3, "occupied_slots": 4, "kv_tokens": 80, "queue_depth": 5},
    ]
    itl = [{"gap_s": 0.1}, {"gap_s": 0.2}, {"gap_s": 0.3}]
    s = summarize_repeat(
        0, records, engine_steps=steps, engine_itl=itl,
        max_batch=4, max_seq_len=100, expect_fixed_length=True,
    )
    assert s.step_dur_p50 == pytest.approx(0.2)
    assert s.occupancy_mean == pytest.approx((0.5 + 1.0) / 2)
    assert s.kv_fill_mean == pytest.approx((0.1 + 0.2) / 2)
    assert s.queue_depth_max == 5
    assert s.engine_itl_p50 == pytest.approx(0.2)
    assert any("unexpected finish reasons" in w for w in s.warnings)


def test_open_loop_dispatch_lag_warning():
    records = [rec(i, t0=float(i), scheduled=float(i) - 0.5) for i in range(3)]
    s = summarize_repeat(0, records)
    assert any("load generator overloaded" in w for w in s.warnings)


def test_median_across_repeats_and_cv_flag():
    summaries = [
        summarize_repeat(k, [rec(i, dur=2.0, out=tokens) for i in range(4)])
        for k, tokens in enumerate((10, 10, 20))   # tok/s varies wildly
    ]
    out = median_across_repeats(summaries)
    assert out["n_repeats"] == 3
    assert out["aggregate_tok_s"] == pytest.approx(summaries[1].aggregate_tok_s)
    assert any("CV" in w for w in out["warnings"])
    assert out["finish_reasons"] == {"length": 12}
