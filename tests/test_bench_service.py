"""Control-panel service tests: config listing, run lifecycle (409 guard,
abort), history, compare join. Real executor + stub server behind POST."""

import asyncio
import json
import sys

import httpx

from cantollm.bench.service import create_bench_app
from tests.test_bench_executor import write_tiny_workload

PORT = 8398


def seed_config(tmp_path, name="svc-test", port=PORT, requests=4):
    cdir = tmp_path / "configs"
    cdir.mkdir(exist_ok=True)
    (cdir / f"{name}.toml").write_text(
        'schema_version = 1\n'
        f'name = "{name}"\n'
        f'[server]\nmodel = "0.6B"\nport = {port}\nhealth_timeout_s = 30\n'
        '[defaults]\nmax_tokens = 4\nwarmup_requests = 0\nrepeats = 1\n'
        '[[points]]\nworkload = "short_chat"\nmode = "closed"\n'
        f'concurrency = [2]\nrequests_per_level = {requests}\n'
    )
    return cdir


def seed_history_run(history_dir, run_id, tok_s):
    d = history_dir / run_id
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({
        "schema_version": 1, "run_id": run_id, "status": "done",
        "started": "t0", "finished": "t1",
        "env": {"git_sha": "abc1234", "device_name": "test"},
        "config": {"name": "seeded"},
        "cells": [{
            "cell_id": "s0-p0-c4", "server": {"model": "0.6B", "engine": "batched"},
            "workload": "short_chat", "mode": "closed", "level": 4.0,
            "status": "done", "median": {
                "aggregate_tok_s": tok_s, "ttft_p50": 1.0, "warnings": [],
            },
        }],
    }))


def make_client(tmp_path):
    app = create_bench_app(
        history_dir=tmp_path / "history",
        workloads_dir=write_tiny_workload(tmp_path),
        configs_dir=seed_config(tmp_path),
        server_command=lambda argv: [
            sys.executable, "-m", "tests.bench_server_stub",
            "--port", str(PORT), "--mode", "fake",
        ],
    )
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t"
    )


def test_configs_workloads_history_endpoints(tmp_path):
    seed_history_run(tmp_path / "history", "2026-01-01T0000_abc1234_x", 30.0)

    async def main():
        async with make_client(tmp_path) as client:
            configs = (await client.get("/api/configs")).json()
            expand = (await client.get("/api/configs/svc-test/expand")).json()
            workloads = (await client.get("/api/workloads")).json()
            runs = (await client.get("/api/runs")).json()
            index = await client.get("/")
            no_records = await client.get(
                "/api/runs/2026-01-01T0000_abc1234_x/chats",
                params={"cell": "s0-p0-c4"},
            )
            return configs, expand, workloads, runs, index, no_records

    configs, expand, workloads, runs, index, no_records = asyncio.run(main())
    assert configs[0]["name"] == "svc-test" and configs[0]["n_cells"] == 1
    assert expand["cells"][0]["workload"] == "short_chat"
    assert workloads[0]["name"] == "short_chat"
    assert runs["active"] is None
    assert runs["history"][0]["run_id"] == "2026-01-01T0000_abc1234_x"
    assert index.status_code == 200 and b"CantoLLM" in index.content
    assert index.headers["cache-control"] == "no-cache"
    assert no_records.status_code == 404   # seeded run has no requests file


def test_run_lifecycle_409_and_completion(tmp_path):
    async def main():
        async with make_client(tmp_path) as client:
            started = await client.post(
                "/api/runs", json={"config": "svc-test", "capture_text": True}
            )
            assert started.status_code == 201, started.text
            run_id = started.json()["run_id"]

            second = await client.post("/api/runs", json={"config": "svc-test"})
            assert second.status_code == 409

            for _ in range(300):
                runs = (await client.get("/api/runs")).json()
                if runs["active"] is None:
                    break
                await asyncio.sleep(0.1)
            assert runs["active"] is None, "run did not finish"

            detail = (await client.get(f"/api/runs/{run_id}")).json()
            requests = (await client.get(f"/api/runs/{run_id}/requests")).json()
            cell_id = detail["cells"][0]["cell_id"]
            chats = (await client.get(
                f"/api/runs/{run_id}/chats", params={"cell": cell_id}
            )).json()
            return run_id, detail, requests, chats

    run_id, detail, requests, chats = asyncio.run(main())
    assert detail["status"] == "done"
    assert detail["cells"][0]["status"] == "done"
    assert len(requests) == 4
    assert all(r["output_tokens"] == 4 for r in requests)

    # /chats joins records with workload prompts and captured output text.
    assert chats["captured"] is True
    assert chats["workload"] == "short_chat"
    assert len(chats["rows"]) == 4
    for row in chats["rows"]:
        assert row["prompt"]["messages"][-1]["role"] == "user"
        assert isinstance(row["output_text"], str) and row["output_text"]


def test_compare_joins_and_deltas(tmp_path):
    seed_history_run(tmp_path / "history", "runA", 30.0)
    seed_history_run(tmp_path / "history", "runB", 45.0)

    async def main():
        async with make_client(tmp_path) as client:
            r = await client.get("/api/compare", params={"runs": "runA,runB"})
            missing = await client.get("/api/compare", params={"runs": "runA,nope"})
            return r.json(), missing.status_code

    cmp, missing_status = asyncio.run(main())
    assert missing_status == 404
    row = cmp["rows"][0]
    assert row["deltas"]["aggregate_tok_s"]["abs"] == 15.0
    assert round(row["deltas"]["aggregate_tok_s"]["pct"]) == 50
