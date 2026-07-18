"""Executor state-machine tests against the runnable stub server.

Each test spawns a real uvicorn subprocess via the `server_command` seam
(tests/bench_server_stub.py) — seconds per test, like test_process_engine.
Torch never loads: the stub serves FakeEngine-style doubles.
"""

import asyncio
import json
import sys

from cantollm.bench.config import parse_run_config
from cantollm.bench.executor import RunHandle, execute_run
from cantollm.bench.records import read_jsonl_gz


def stub_command(port: int, mode: str = "fake", crash_after: int | None = None):
    def command(serve_argv):
        argv = [sys.executable, "-m", "tests.bench_server_stub",
                "--port", str(port), "--mode", mode]
        if crash_after is not None:
            argv += ["--crash-after", str(crash_after)]
        return argv
    return command


def config_for(port: int, **point_overrides) -> dict:
    point = {
        "workload": "short_chat", "mode": "closed",
        "concurrency": [2], "requests_per_level": 6,
        **point_overrides,
    }
    return {
        "schema_version": 1,
        "name": "stub-test",
        "server": {"model": "0.6B", "port": port, "health_timeout_s": 30},
        "defaults": {"max_tokens": 5, "warmup_requests": 2, "repeats": 2,
                     "ignore_eos": True, "temperature": 0.0},
        "points": [point],
    }


def write_tiny_workload(tmp_path, n=8):
    wdir = tmp_path / "workloads"
    wdir.mkdir()
    lines = [json.dumps({"schema_version": 1, "set": "short_chat",
                         "tokenizer": "test"})]
    for i in range(n):
        lines.append(json.dumps({
            "id": f"p{i}", "messages": [{"role": "user", "content": f"q {i}"}],
            "input_tokens": 5,
        }))
    (wdir / "short_chat.jsonl").write_text("\n".join(lines) + "\n")
    return wdir


def execute(config_dict, tmp_path, port, mode="fake", crash_after=None,
            handle=None):
    config = parse_run_config(config_dict)
    return asyncio.run(execute_run(
        config,
        history_dir=tmp_path / "history",
        workloads_dir=write_tiny_workload(tmp_path),
        server_command=stub_command(port, mode, crash_after),
        handle=handle,
    ))


def test_happy_path_persists_everything(tmp_path):
    handle = execute(config_for(8391), tmp_path, port=8391)

    assert handle.status == "done"
    cell = handle.cells[0]
    assert cell.status == "done"
    assert len(cell.repeats) == 2
    assert cell.median["aggregate_tok_s"] > 0
    assert cell.median["ttft_p50"] is not None
    assert cell.spawn_to_ready_s > 0
    assert cell.median["finish_reasons"] == {"length": 12}

    run_json = handle.run_dir.read_run_json()
    assert run_json["status"] == "done"
    assert run_json["env"]["python"]
    assert run_json["config"]["name"] == "stub-test"

    rows = read_jsonl_gz(handle.run_dir.requests_path)
    assert len(rows) == 2 + 2 * 6            # warmup + 2 repeats × 6
    warm = [r for r in rows if r["excluded"]]
    assert len(warm) == 2 and all(r["repeat"] == -1 for r in warm)
    measured = [r for r in rows if not r["excluded"]]
    assert all(r["output_tokens"] == 5 for r in measured)
    # No engine stats from the fake stub → no steps file, engine fields null.
    assert not handle.run_dir.steps_path.exists()
    assert cell.median["engine_itl_p50"] is None


def test_stats_scrape_populates_load_and_capacity(tmp_path):
    handle = execute(config_for(8392), tmp_path, port=8392, mode="stats")
    cell = handle.cells[0]
    assert cell.status == "done"
    assert cell.load_seconds == 0.123


def test_server_crash_fails_cell_with_log_tail(tmp_path):
    cfg = config_for(8393, requests_per_level=8)
    cfg["defaults"]["repeats"] = 1
    cfg["defaults"]["warmup_requests"] = 0
    handle = execute(cfg, tmp_path, port=8393, mode="crash", crash_after=2)

    cell = handle.cells[0]
    assert cell.status == "failed"
    assert cell.reason
    assert handle.status == "done"            # run completes, cell marked
    run_json = handle.run_dir.read_run_json()
    assert run_json["cells"][0]["status"] == "failed"


def test_abort_persists_partials(tmp_path):
    cfg = config_for(8394, requests_per_level=200)
    cfg["defaults"]["max_tokens"] = 50        # slow enough to abort mid-flight

    async def main():
        handle = RunHandle(run_id="pending", config_name="stub-test")
        task = asyncio.create_task(execute_run(
            parse_run_config(cfg),
            history_dir=tmp_path / "history",
            workloads_dir=write_tiny_workload(tmp_path),
            server_command=stub_command(8394),
            handle=handle,
        ))
        # Let it spawn + get some requests in flight, then pull the plug.
        for _ in range(200):
            await asyncio.sleep(0.1)
            if handle.requests_done >= 2:
                break
        handle.abort()
        return await asyncio.wait_for(task, timeout=30.0)

    handle = asyncio.run(main())
    assert handle.status == "aborted"
    assert handle.run_dir.read_run_json()["status"] == "aborted"
