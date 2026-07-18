"""Result record shapes + persistence primitives (bench-spec.md §6).

Every persisted record carries `RESULTS_SCHEMA_VERSION`; comparisons refuse
mismatched versions. Timing fields follow the §3 taxonomy on the client's
perf clock; engine step records are the scraped /debug/engine-stats dicts,
tagged with their cell/repeat.

Gzip append: `.jsonl.gz` files grow by whole gzip members (one per flush),
which Python's gzip module reads back as a single concatenated stream.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

RESULTS_SCHEMA_VERSION = 1


@dataclass
class RequestRecord:
    """One request as the load generator saw it."""

    cell_id: str
    repeat: int              # 0-based; warmup rows carry excluded=True
    request_index: int       # per-repeat submission order
    prompt_id: str           # workload prompt id
    dialect: str             # "openai" | "anthropic"

    # §3 timestamp taxonomy, client perf clock (seconds).
    t_scheduled: float | None  # open-loop intended send; None closed-loop
    t_send: float
    t_headers: float | None = None
    t_first_token: float | None = None
    t_done: float | None = None

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    finish_reason: str | None = None
    error: str | None = None
    http_status: int | None = None
    excluded: bool = False   # warmup / deliberately unaggregated

    schema_version: int = RESULTS_SCHEMA_VERSION

    # Derived, populated by finalize().
    ttft_s: float | None = None
    completion_s: float | None = None
    accept_s: float | None = None
    client_itl_mean_s: float | None = None
    dispatch_lag_s: float | None = None

    def finalize(self) -> "RequestRecord":
        """Compute the §3 derived metrics from raw timestamps."""
        if self.t_first_token is not None:
            self.ttft_s = self.t_first_token - self.t_send
        if self.t_done is not None:
            self.completion_s = self.t_done - self.t_send
        if self.t_headers is not None:
            self.accept_s = self.t_headers - self.t_send
        if (
            self.t_done is not None
            and self.t_first_token is not None
            and self.output_tokens >= 2
        ):
            self.client_itl_mean_s = (
                (self.t_done - self.t_first_token) / (self.output_tokens - 1)
            )
        if self.t_scheduled is not None:
            self.dispatch_lag_s = self.t_send - self.t_scheduled
        return self

    @property
    def ok(self) -> bool:
        return self.error is None and self.t_done is not None


@dataclass
class RepeatSummary:
    """Aggregates over one measured repeat of one cell (§3, §4, §7)."""

    repeat: int
    n_requests: int = 0
    n_ok: int = 0
    n_errors: int = 0
    wall_s: float = 0.0

    ttft_p50: float | None = None
    ttft_p90: float | None = None
    ttft_p99: float | None = None
    completion_p50: float | None = None
    completion_p99: float | None = None
    aggregate_tok_s: float | None = None
    request_tok_s_p50: float | None = None
    client_itl_mean_s: float | None = None
    dispatch_lag_p99: float | None = None

    engine_itl_p50: float | None = None
    engine_itl_p99: float | None = None
    step_dur_p50: float | None = None
    step_dur_p99: float | None = None
    occupancy_mean: float | None = None
    kv_fill_mean: float | None = None
    queue_depth_max: int | None = None

    finish_reasons: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ── gzip-append JSONL ─────────────────────────────────────────────────


def append_jsonl_gz(path: Path, rows: list[dict]) -> None:
    """Append `rows` as one gzip member; a no-op for an empty batch."""
    if not rows:
        return
    payload = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "ab") as f:
        f.write(gzip.compress(payload.encode()))


def read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]
