"""The math over request/step records (bench-spec.md §3–§5).

Percentiles use linear interpolation on sorted values — the same method the
original clients/bench.py used, now pinned by the spec so history stays
comparable. All functions are pure and torch-free.
"""

from __future__ import annotations

import statistics

from cantollm.bench.records import RepeatSummary, RequestRecord

# Validity thresholds (bench-spec.md §5).
CV_WARN = 0.05
ERROR_RATE_WARN = 0.01
ERROR_RATE_FAIL = 0.50
DISPATCH_LAG_P99_WARN_S = 0.100


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    idx = q * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def cv(values: list[float]) -> float:
    """Coefficient of variation (sample stdev / mean); nan if undefined."""
    if len(values) < 2:
        return float("nan")
    mean = statistics.fmean(values)
    if mean == 0:
        return float("nan")
    return statistics.stdev(values) / mean


def summarize_repeat(
    repeat: int,
    records: list[RequestRecord],
    engine_steps: list[dict] | None = None,
    engine_itl: list[dict] | None = None,
    max_batch: int | None = None,
    max_seq_len: int | None = None,
    expect_fixed_length: bool = False,
) -> RepeatSummary:
    """One repeat's aggregates. `records` are the measured (non-excluded)
    requests of the repeat; engine data is the scraped window for it."""
    measured = [r for r in records if not r.excluded]
    ok = [r for r in measured if r.ok]
    errors = [r for r in measured if r.error is not None]

    s = RepeatSummary(repeat=repeat)
    s.n_requests = len(measured)
    s.n_ok = len(ok)
    s.n_errors = len(errors)

    if ok:
        # §3: untrimmed window over the repeat's measured requests.
        window_start = min(r.t_send for r in ok)
        window_end = max(r.t_done for r in ok)
        s.wall_s = window_end - window_start
        total_out = sum(r.output_tokens for r in ok)
        if s.wall_s > 0:
            s.aggregate_tok_s = total_out / s.wall_s

        ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]
        if ttfts:
            s.ttft_p50 = percentile(ttfts, 0.50)
            s.ttft_p90 = percentile(ttfts, 0.90)
            s.ttft_p99 = percentile(ttfts, 0.99)
        completions = [r.completion_s for r in ok if r.completion_s is not None]
        if completions:
            s.completion_p50 = percentile(completions, 0.50)
            s.completion_p99 = percentile(completions, 0.99)
        decode_rates = [
            r.output_tokens / (r.t_done - r.t_first_token)
            for r in ok
            if r.t_first_token is not None
            and r.t_done is not None
            and r.output_tokens >= 2
            and r.t_done > r.t_first_token
        ]
        if decode_rates:
            s.request_tok_s_p50 = percentile(decode_rates, 0.50)
        client_itls = [r.client_itl_mean_s for r in ok if r.client_itl_mean_s is not None]
        if client_itls:
            s.client_itl_mean_s = statistics.fmean(client_itls)
        lags = [r.dispatch_lag_s for r in measured if r.dispatch_lag_s is not None]
        if lags:
            s.dispatch_lag_p99 = percentile(lags, 0.99)

    for r in measured:
        if r.finish_reason:
            s.finish_reasons[r.finish_reason] = s.finish_reasons.get(r.finish_reason, 0) + 1

    if engine_steps:
        durs = [st["dur_s"] for st in engine_steps]
        s.step_dur_p50 = percentile(durs, 0.50)
        s.step_dur_p99 = percentile(durs, 0.99)
        if max_batch:
            s.occupancy_mean = statistics.fmean(
                st["occupied_slots"] / max_batch for st in engine_steps
            )
        if max_batch and max_seq_len:
            s.kv_fill_mean = statistics.fmean(
                st["kv_tokens"] / (max_batch * max_seq_len) for st in engine_steps
            )
        s.queue_depth_max = max(st["queue_depth"] for st in engine_steps)
    if engine_itl:
        gaps = [g["gap_s"] for g in engine_itl]
        s.engine_itl_p50 = percentile(gaps, 0.50)
        s.engine_itl_p99 = percentile(gaps, 0.99)

    # §5 validity rules that are visible within a single repeat.
    if s.n_requests:
        rate = s.n_errors / s.n_requests
        if rate > ERROR_RATE_FAIL:
            s.warnings.append(f"error rate {rate:.0%} exceeds {ERROR_RATE_FAIL:.0%} — cell failed")
        elif rate > ERROR_RATE_WARN:
            s.warnings.append(f"error rate {rate:.0%} exceeds {ERROR_RATE_WARN:.0%}")
    if s.dispatch_lag_p99 is not None and s.dispatch_lag_p99 > DISPATCH_LAG_P99_WARN_S:
        s.warnings.append(
            f"dispatch lag p99 {s.dispatch_lag_p99 * 1000:.0f}ms — load generator overloaded"
        )
    if expect_fixed_length:
        unexpected = {
            reason: n for reason, n in s.finish_reasons.items()
            if reason not in ("max_tokens", "length")
        }
        if unexpected:
            s.warnings.append(f"unexpected finish reasons under ignore_eos: {unexpected}")

    return s


def median_across_repeats(summaries: list[RepeatSummary]) -> dict:
    """Headline-table row: per-metric median across repeats, plus the §5
    CV validity check on aggregate tok/s and TTFT p50."""
    numeric = [
        "ttft_p50", "ttft_p90", "ttft_p99",
        "completion_p50", "completion_p99",
        "aggregate_tok_s", "request_tok_s_p50", "client_itl_mean_s",
        "engine_itl_p50", "engine_itl_p99",
        "step_dur_p50", "step_dur_p99",
        "occupancy_mean", "kv_fill_mean", "wall_s",
    ]
    out: dict = {"n_repeats": len(summaries)}
    for name in numeric:
        values = [getattr(s, name) for s in summaries if getattr(s, name) is not None]
        out[name] = statistics.median(values) if values else None

    warnings = [w for s in summaries for w in s.warnings]
    for name, label in (("aggregate_tok_s", "aggregate tok/s"), ("ttft_p50", "TTFT p50")):
        values = [getattr(s, name) for s in summaries if getattr(s, name) is not None]
        if len(values) >= 2:
            c = cv(values)
            if c > CV_WARN:
                warnings.append(f"{label} CV {c:.1%} across repeats exceeds {CV_WARN:.0%}")
    out["warnings"] = warnings

    reasons: dict[str, int] = {}
    for s in summaries:
        for reason, n in s.finish_reasons.items():
            reasons[reason] = reasons.get(reason, 0) + n
    out["finish_reasons"] = reasons
    out["n_errors"] = sum(s.n_errors for s in summaries)
    return out
