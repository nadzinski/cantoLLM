"""Full-pipeline mini-bench: executor → spawned server → REAL in-process
ContinuousBatchingEngine on tiny-Qwen3 → real engine stats → history dir.

The one test where every layer is real except the model weights (random
tiny init) — a few seconds on CPU. Asserts run.json completeness, engine
steps captured through /debug/engine-stats, and finite metrics.
"""

import asyncio
import sys

from cantollm.bench.config import parse_run_config
from cantollm.bench.executor import execute_run
from cantollm.bench.records import read_jsonl_gz
from tests.test_bench_executor import write_tiny_workload

PORT = 8397


def test_tiny_model_end_to_end(tmp_path):
    config = parse_run_config({
        "schema_version": 1,
        "name": "tiny-e2e",
        "server": {"model": "0.6B", "port": PORT, "health_timeout_s": 120},
        "defaults": {"max_tokens": 6, "warmup_requests": 1, "repeats": 1,
                     "ignore_eos": True, "temperature": 0.0},
        "points": [{
            "workload": "short_chat", "mode": "closed",
            "concurrency": [2], "requests_per_level": 6,
        }],
    })

    handle = asyncio.run(execute_run(
        config,
        history_dir=tmp_path / "history",
        workloads_dir=write_tiny_workload(tmp_path),
        server_command=lambda serve_argv: [
            sys.executable, "-m", "tests.bench_server_stub",
            "--port", str(PORT), "--mode", "tiny",
        ],
    ))

    assert handle.status == "done"
    cell = handle.cells[0]
    assert cell.status == "done", cell.reason
    median = cell.median

    # Client side: finite, sane numbers.
    assert median["aggregate_tok_s"] > 0
    assert median["ttft_p50"] > 0
    assert median["completion_p50"] >= median["ttft_p50"]
    assert median["finish_reasons"] == {"length": 6}
    assert median["n_errors"] == 0

    # Engine side: real steps crossed the scrape and landed in the summary.
    assert median["engine_itl_p50"] is not None and median["engine_itl_p50"] > 0
    assert median["step_dur_p50"] > 0
    assert 0 < median["occupancy_mean"] <= 1
    assert 0 < median["kv_fill_mean"] <= 1
    assert handle.run_dir.steps_path.exists()
    steps = read_jsonl_gz(handle.run_dir.steps_path)
    assert steps and all(s["cell_id"] == cell.cell.cell_id for s in steps)
    assert sum(s["decode_tokens"] for s in steps) > 0
    assert sum(s["prefill_tokens"] for s in steps) > 0

    # History: request records complete and repeat-tagged.
    rows = read_jsonl_gz(handle.run_dir.requests_path)
    measured = [r for r in rows if not r["excluded"]]
    assert len(measured) == 6
    assert all(r["output_tokens"] == 6 for r in measured)
    assert all(r["finish_reason"] == "length" for r in measured)
