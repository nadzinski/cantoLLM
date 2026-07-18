"""The bench control panel (`canto bench ui`, port 8002; bench-spec.md §7).

A small FastAPI app over the same executor library the CLI uses: launch a
run from a committed config, watch it live (1 s polling of the atomic
run.json partials + the RunHandle snapshot), browse history, compare two
runs. One active run at a time — a second start gets 409; a bench run
saturates the machine, so concurrency would corrupt both runs' numbers.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from cantollm.bench.config import ConfigError, load_run_config
from cantollm.bench.executor import RunHandle, execute_run
from cantollm.bench.history import DEFAULT_HISTORY_DIR, list_runs, load_run
from cantollm.bench.records import read_jsonl_gz
from cantollm.bench.workloads import (
    DEFAULT_WORKLOADS_DIR,
    WorkloadError,
    list_workloads,
    load_workload,
)

DEFAULT_CONFIGS_DIR = Path("bench/configs")
STATIC_DIR = Path(__file__).parent / "static"


def create_bench_app(
    history_dir: Path | None = None,
    workloads_dir: Path | None = None,
    configs_dir: Path | None = None,
    server_command=None,   # test seam, forwarded to execute_run
) -> FastAPI:
    history_dir = history_dir or DEFAULT_HISTORY_DIR
    workloads_dir = workloads_dir or DEFAULT_WORKLOADS_DIR
    configs_dir = configs_dir or DEFAULT_CONFIGS_DIR

    app = FastAPI(title="CantoLLM bench")
    active: dict = {"handle": None, "task": None}

    def active_handle() -> RunHandle | None:
        handle = active["handle"]
        task = active["task"]
        if handle is None:
            return None
        if task is not None and task.done():
            return None
        return handle

    @app.get("/api/configs")
    async def configs():
        out = []
        for path in sorted(configs_dir.glob("*.toml")):
            try:
                run = load_run_config(path)
                out.append({
                    "name": path.stem,
                    "path": str(path),
                    "n_cells": len(run.cells),
                    "attach": run.attach,
                })
            except (ConfigError, OSError) as e:
                out.append({"name": path.stem, "path": str(path), "error": str(e)})
        return out

    @app.get("/api/configs/{name}/expand")
    async def expand(name: str):
        path = configs_dir / f"{name}.toml"
        if not path.exists():
            raise HTTPException(404, f"no such config: {name}")
        try:
            run = load_run_config(path)
        except ConfigError as e:
            raise HTTPException(400, str(e))
        return {"name": run.name, "attach": run.attach,
                "cells": [c.to_dict() for c in run.cells]}

    @app.get("/api/workloads")
    async def workloads():
        return list_workloads(workloads_dir)

    @app.get("/api/runs")
    async def runs():
        handle = active_handle()
        return {
            "active": handle.snapshot() if handle else None,
            "history": list_runs(history_dir),
        }

    @app.post("/api/runs", status_code=201)
    async def start_run(body: dict):
        if active_handle() is not None:
            raise HTTPException(
                409, "a run is already active — a bench run saturates the "
                "machine, so concurrent runs would corrupt both"
            )
        name = body.get("config")
        if not name:
            raise HTTPException(400, "body must carry {'config': <name>}")
        path = configs_dir / f"{name}.toml"
        if not path.exists():
            raise HTTPException(404, f"no such config: {name}")
        try:
            config = load_run_config(path)
        except ConfigError as e:
            raise HTTPException(400, str(e))
        attach_url = body.get("attach_url")
        if config.attach and not attach_url:
            raise HTTPException(400, "config has no [server] — pass attach_url")

        handle = RunHandle(run_id="starting", config_name=config.name)
        task = asyncio.create_task(execute_run(
            config,
            history_dir=history_dir,
            workloads_dir=workloads_dir,
            attach_url=attach_url,
            capture_text=bool(body.get("capture_text")),
            server_command=server_command,
            handle=handle,
        ))
        active["handle"], active["task"] = handle, task
        for _ in range(40):                      # run_id lands almost at once
            if handle.run_id != "starting" or task.done():
                break
            await asyncio.sleep(0.05)
        return handle.snapshot()

    @app.get("/api/runs/{run_id}")
    async def run_detail(run_id: str):
        try:
            return load_run(run_id, history_dir)
        except FileNotFoundError:
            raise HTTPException(404, f"no such run: {run_id}")

    @app.get("/api/runs/{run_id}/live")
    async def run_live(run_id: str):
        handle = active_handle()
        if handle is None or handle.run_id != run_id:
            raise HTTPException(404, "run is not active")
        return handle.snapshot()

    @app.post("/api/runs/{run_id}/abort")
    async def abort_run(run_id: str):
        handle = active_handle()
        if handle is None or handle.run_id != run_id:
            raise HTTPException(404, "run is not active")
        handle.abort()
        return {"status": "aborting"}

    @app.get("/api/runs/{run_id}/requests")
    async def run_requests(run_id: str, cell: str | None = None,
                           repeat: int | None = None):
        path = history_dir / run_id / "requests.jsonl.gz"
        if not path.exists():
            raise HTTPException(404, "no request records for this run")
        rows = read_jsonl_gz(path)
        if cell is not None:
            rows = [r for r in rows if r["cell_id"] == cell]
        if repeat is not None:
            rows = [r for r in rows if r["repeat"] == repeat]
        return rows

    @app.get("/api/runs/{run_id}/chats")
    async def run_chats(run_id: str, cell: str, repeat: int | None = None):
        """One cell's request records joined with prompt content (from the
        committed workload file) and captured output text, when the run was
        launched with capture-text — one fetch renders the transcript
        drill-down. Warmup rows ride along (repeat=-1, excluded=true)."""
        req_path = history_dir / run_id / "requests.jsonl.gz"
        if not req_path.exists():
            raise HTTPException(404, "no request records for this run")
        rows = [r for r in read_jsonl_gz(req_path) if r["cell_id"] == cell]
        if repeat is not None:
            rows = [r for r in rows if r["repeat"] == repeat]
        rows.sort(key=lambda r: (r["repeat"], r["request_index"]))

        text_path = history_dir / run_id / "output_text.jsonl.gz"
        texts: dict[str, str] = {}
        if text_path.exists():
            for entry in read_jsonl_gz(text_path):
                texts[entry["key"]] = entry["text"]

        try:
            run = load_run(run_id, history_dir)
        except FileNotFoundError:
            run = {}
        workload_name = next(
            (c.get("workload") for c in run.get("cells", [])
             if c.get("cell_id") == cell),
            None,
        )
        prompts: dict[str, dict] = {}
        if workload_name:
            try:
                w = load_workload(workload_name, workloads_dir)
                prompts = {
                    p.id: {"system": p.system, "messages": list(p.messages)}
                    for p in w.prompts
                }
            except WorkloadError:
                pass   # workload file gone since the run — records still show

        return {
            "workload": workload_name,
            "captured": text_path.exists(),
            "rows": [{
                **r,
                "prompt": prompts.get(r["prompt_id"]),
                "output_text": texts.get(
                    f"{r['cell_id']}:{r['repeat']}:{r['request_index']}:{r['prompt_id']}"
                ),
            } for r in rows],
        }

    @app.get("/api/compare")
    async def compare(runs: str):
        ids = [r for r in runs.split(",") if r]
        if len(ids) != 2:
            raise HTTPException(400, "pass exactly two run ids: ?runs=a,b")
        loaded = []
        for run_id in ids:
            try:
                loaded.append(load_run(run_id, history_dir))
            except FileNotFoundError:
                raise HTTPException(404, f"no such run: {run_id}")
        return _join_cells(loaded[0], loaded[1])

    @app.get("/")
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.middleware("http")
    async def no_stale_ui(request, call_next):
        # Force revalidation on the static UI (ETag turns it into a cheap
        # 304) — browsers otherwise apply heuristic freshness and keep
        # serving a stale app.js for hours after an edit.
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def _cell_key(cell: dict) -> tuple:
    server = cell.get("server") or {}
    server_key = tuple(sorted(
        (k, v) for k, v in server.items()
        if k not in ("port", "health_timeout_s", "respawn")
    ))
    return (cell["workload"], cell["mode"], cell["level"], server_key)


def _join_cells(run_a: dict, run_b: dict) -> dict:
    """Comparison join (bench-spec.md §7): same schema version, same
    workload hash where recorded; rows joined on (workload, mode, level,
    server config) with absolute + % deltas on the headline metrics."""
    if run_a.get("schema_version") != run_b.get("schema_version"):
        raise HTTPException(400, "runs have different schema_version — not comparable")

    b_by_key = {_cell_key(c): c for c in run_b.get("cells", [])}
    metrics = ("aggregate_tok_s", "ttft_p50", "ttft_p99", "completion_p50",
               "engine_itl_p50", "occupancy_mean", "kv_fill_mean")
    rows = []
    for cell_a in run_a.get("cells", []):
        cell_b = b_by_key.get(_cell_key(cell_a))
        row = {
            "workload": cell_a["workload"],
            "mode": cell_a["mode"],
            "level": cell_a["level"],
            "server": cell_a.get("server"),
            "a": cell_a.get("median") or {},
            "b": (cell_b.get("median") or {}) if cell_b else None,
            "deltas": {},
        }
        if cell_b:
            for m in metrics:
                va = (cell_a.get("median") or {}).get(m)
                vb = (cell_b.get("median") or {}).get(m)
                if va is not None and vb is not None:
                    row["deltas"][m] = {
                        "abs": vb - va,
                        "pct": ((vb - va) / va * 100) if va else None,
                    }
        rows.append(row)
    return {
        "run_a": {"run_id": run_a["run_id"], "env": run_a.get("env", {})},
        "run_b": {"run_id": run_b["run_id"], "env": run_b.get("env", {})},
        "rows": rows,
    }


def run_service(host: str = "127.0.0.1", port: int = 8002) -> None:
    import uvicorn

    print(f"bench control panel: http://{host}:{port}/")
    uvicorn.run(create_bench_app(), host=host, port=port, log_level="warning")
