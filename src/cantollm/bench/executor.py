"""The run state machine (bench-spec.md §5) — a library with two frontends.

`canto bench run <config>` wraps `execute_run` in asyncio.run; the control
panel runs it as a task and polls the `RunHandle`. Per server variant:
spawn → /health → per cell: warmup (excluded) → N measured repeats (each:
loadgen + engine-stats window + summary + persist) → drain barrier → next.
run.json is atomically rewritten after every repeat, so the live view and
crash recovery always see a coherent partial state.

Failure policy: a dead or unhealthy server fails the *cell* (reason + log
tail recorded), teardown, continue with the next cell on a fresh spawn —
unless stop_on_cell_failure. Abort cancels in-flight streams (the server's
disconnect→abort path frees its slots), tears down, and persists partials
with status "aborted".
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import httpx

from cantollm.bench import env as bench_env
from cantollm.bench import metrics
from cantollm.bench.config import Cell, RunConfig, ServerVariant, serve_argv
from cantollm.bench.history import DEFAULT_HISTORY_DIR, RunDir, make_run_id
from cantollm.bench.loadgen import run_closed_loop, run_open_loop
from cantollm.bench.records import RequestRecord, append_jsonl_gz
from cantollm.bench.server_ctl import (
    AttachedServer,
    ServerProcess,
    ServerStartupError,
    StatsScraper,
    default_serve_command,
)
from cantollm.bench.sse_clients import SendOptions, build_sender
from cantollm.bench.workloads import DEFAULT_WORKLOADS_DIR, load_workload

DRAIN_TIMEOUT_S = 60.0


@dataclass
class CellState:
    cell: Cell
    status: str = "pending"      # pending|running|done|failed|aborted
    reason: str | None = None
    repeats: list[dict] = field(default_factory=list)
    median: dict = field(default_factory=dict)
    spawn_to_ready_s: float | None = None
    load_seconds: float | None = None

    def to_dict(self) -> dict:
        return {
            **self.cell.to_dict(),
            "status": self.status,
            "reason": self.reason,
            "repeats": self.repeats,
            "median": self.median,
            "spawn_to_ready_s": self.spawn_to_ready_s,
            "load_seconds": self.load_seconds,
        }


@dataclass
class RunHandle:
    run_id: str
    config_name: str
    status: str = "running"      # running|done|aborted|failed
    started: str = ""
    finished: str | None = None
    cells: list[CellState] = field(default_factory=list)
    cell_index: int = 0
    repeat_index: int = 0
    requests_done: int = 0
    abort_event: asyncio.Event = field(default_factory=asyncio.Event)
    run_dir: RunDir | None = None

    def abort(self) -> None:
        self.abort_event.set()

    @property
    def aborted(self) -> bool:
        return self.abort_event.is_set()

    def snapshot(self) -> dict:
        return {
            "run_id": self.run_id,
            "config_name": self.config_name,
            "status": self.status,
            "cell_index": self.cell_index,
            "n_cells": len(self.cells),
            "repeat_index": self.repeat_index,
            "requests_done": self.requests_done,
            "current_cell": (
                self.cells[self.cell_index].to_dict()
                if self.cell_index < len(self.cells) else None
            ),
        }


async def execute_run(
    config: RunConfig,
    *,
    history_dir: Path | None = None,
    workloads_dir: Path | None = None,
    attach_url: str | None = None,
    capture_text: bool = False,
    server_command=None,          # (list[str]) -> list[str]; test seam
    handle: RunHandle | None = None,
) -> RunHandle:
    if config.attach and not attach_url:
        raise ValueError("config has no [server] section — pass --attach --url")
    server_command = server_command or default_serve_command
    workloads_dir = workloads_dir or DEFAULT_WORKLOADS_DIR

    environment = bench_env.fingerprint()
    run_id = make_run_id(config.name, environment.get("git_sha"))
    run_dir = RunDir(history_dir or DEFAULT_HISTORY_DIR, run_id)

    if handle is None:
        handle = RunHandle(run_id=run_id, config_name=config.name)
    else:
        handle.run_id = run_id
        handle.config_name = config.name
    handle.run_dir = run_dir
    handle.started = datetime.now().isoformat(timespec="seconds")
    handle.cells = [CellState(cell=c) for c in config.cells]

    def persist() -> None:
        run_dir.write_run_json({
            "run_id": run_id,
            "status": handle.status,
            "started": handle.started,
            "finished": handle.finished,
            "env": environment,
            "config": {"name": config.name, **config.raw},
            "attach_url": attach_url,
            "cells": [c.to_dict() for c in handle.cells],
        })

    persist()
    spawn_index = 0
    try:
        for variant, states in _group_by_variant(handle.cells):
            if handle.aborted:
                break
            respawn_per_cell = (
                variant is not None
                and variant.as_dict.get("respawn") == "per-cell"
            )
            if respawn_per_cell:
                groups = [[s] for s in states]
            else:
                groups = [states]
            for group in groups:
                if handle.aborted:
                    break
                spawn_index += 1
                await _run_server_group(
                    group, variant, config, handle, run_dir, workloads_dir,
                    attach_url, capture_text, server_command, spawn_index,
                    persist,
                )
                if config.stop_on_cell_failure and any(
                    s.status == "failed" for s in group
                ):
                    handle.status = "failed"
                    break
            if handle.status == "failed":
                break
    finally:
        if handle.status == "running":
            handle.status = "aborted" if handle.aborted else "done"
        handle.finished = datetime.now().isoformat(timespec="seconds")
        persist()
    return handle


async def _run_server_group(
    states: list[CellState],
    variant: ServerVariant | None,
    config: RunConfig,
    handle: RunHandle,
    run_dir: RunDir,
    workloads_dir: Path,
    attach_url: str | None,
    capture_text: bool,
    server_command,
    spawn_index: int,
    persist,
) -> None:
    if variant is None:
        server = AttachedServer(attach_url)
        base_url = attach_url
        health_timeout = 30.0
    else:
        cfg = variant.as_dict
        base_url = f"http://127.0.0.1:{cfg['port']}"
        server = ServerProcess(
            server_command(serve_argv(variant)),
            base_url,
            run_dir.server_log_path(spawn_index),
        )
        health_timeout = float(cfg.get("health_timeout_s", 900.0))

    try:
        await server.start(health_timeout)
    except ServerStartupError as e:
        for state in states:
            state.status = "failed"
            state.reason = f"server startup failed: {e}"
        persist()
        return

    async with httpx.AsyncClient(base_url=base_url, timeout=None) as client:
        scraper = StatsScraper(client)
        await scraper.start()
        try:
            for state in states:
                if handle.aborted:
                    state.status = "aborted"
                    continue
                handle.cell_index = handle.cells.index(state)
                state.spawn_to_ready_s = server.spawn_to_ready_s
                await _run_cell(
                    state, handle, run_dir, workloads_dir, client, scraper,
                    server, capture_text, persist,
                )
                persist()
        finally:
            await scraper.stop()
            await server.stop()


async def _run_cell(
    state: CellState,
    handle: RunHandle,
    run_dir: RunDir,
    workloads_dir: Path,
    client: httpx.AsyncClient,
    scraper: StatsScraper,
    server,
    capture_text: bool,
    persist,
) -> None:
    cell = state.cell
    options = cell.options
    state.status = "running"
    try:
        workload = load_workload(cell.workload, workloads_dir)
    except Exception as e:
        state.status = "failed"
        state.reason = f"workload: {e}"
        return

    model = await _resolve_model(client)
    if model is None:
        state.status = "failed"
        state.reason = "could not resolve model from /v1/models"
        return

    send = build_sender(client, SendOptions(
        model=model,
        dialect=options["dialect"],
        max_tokens=int(options["max_tokens"]),
        temperature=float(options["temperature"]),
        top_p=float(options["top_p"]),
        ignore_eos=bool(options["ignore_eos"]),
        capture_text=capture_text,
    ))
    seed = int(options["seed"])
    prompt_limit = options.get("prompt_limit")

    if not await scraper.wait_drained(DRAIN_TIMEOUT_S):
        state.median.setdefault("warnings", []).append("drain barrier timed out")

    # Warmup (excluded from aggregation, persisted with excluded=true).
    warmup_n = int(options["warmup_requests"])
    if warmup_n and not handle.aborted:
        warm = await run_closed_loop(
            send, workload.iterator(seed=seed + 1000, limit=prompt_limit),
            concurrency=min(warmup_n, max(1, int(cell.level)) if cell.mode == "closed" else 4),
            total_requests=warmup_n,
            cell_id=cell.cell_id, repeat=-1, excluded=True,
            abort=handle.abort_event,
        )
        _persist_records(run_dir, warm.records, warm.texts, capture_text)
        scraper.take()  # warmup steps are nobody's window

    summaries = []
    for repeat in range(int(options["repeats"])):
        if handle.aborted:
            break
        handle.repeat_index = repeat
        scraper.overflowed = False
        prompts = workload.iterator(seed=seed + repeat, limit=prompt_limit)

        if cell.mode == "closed":
            result = await run_closed_loop(
                send, prompts,
                concurrency=int(cell.level), total_requests=cell.requests,
                cell_id=cell.cell_id, repeat=repeat,
                abort=handle.abort_event,
                on_record=lambda r: _tick(handle),
            )
        else:
            result = await run_open_loop(
                send, prompts,
                rate_rps=cell.level, arrivals=cell.arrivals,
                total_requests=cell.requests,
                max_inflight=cell.max_inflight or 64,
                seed=seed + repeat,
                cell_id=cell.cell_id, repeat=repeat,
                abort=handle.abort_event,
                on_record=lambda r: _tick(handle),
            )

        await scraper.scrape_once()
        steps, itl = scraper.take()
        _persist_records(run_dir, result.records, result.texts, capture_text)
        if steps:
            append_jsonl_gz(run_dir.steps_path, [
                {"cell_id": cell.cell_id, "repeat": repeat, **s} for s in steps
            ])

        summary = metrics.summarize_repeat(
            repeat, result.records,
            engine_steps=steps or None,
            engine_itl=itl or None,
            max_batch=scraper.capacity.get("max_batch"),
            max_seq_len=scraper.capacity.get("max_seq_len"),
            expect_fixed_length=bool(options["ignore_eos"]),
        )
        if result.hit_inflight_cap:
            summary.warnings.append("open-loop max_inflight cap hit")
        if scraper.overflowed:
            summary.warnings.append("engine-stats ring overflowed between scrapes")
        if result.aborted:
            summary.warnings.append("repeat aborted mid-flight")
        summaries.append(summary)
        state.repeats.append(summary.to_dict())
        state.load_seconds = scraper.load_seconds
        persist()

        if not server.alive:
            state.status = "failed"
            state.reason = (
                f"server died mid-cell (exit {server.exitcode}); "
                f"log tail:\n{server.log_tail()}"
            )
            return
        error_rate = (
            summary.n_errors / summary.n_requests if summary.n_requests else 0.0
        )
        if error_rate > metrics.ERROR_RATE_FAIL:
            state.status = "failed"
            state.reason = f"error rate {error_rate:.0%} in repeat {repeat}"
            return

    if summaries:
        state.median = metrics.median_across_repeats(summaries)
    state.status = "aborted" if handle.aborted else "done"


def _tick(handle: RunHandle) -> None:
    handle.requests_done += 1


def _group_by_variant(states: list[CellState]):
    groups: list[tuple[ServerVariant | None, list[CellState]]] = []
    for state in states:
        if groups and groups[-1][0] == state.cell.server:
            groups[-1][1].append(state)
        else:
            groups.append((state.cell.server, [state]))
    return groups


def _persist_records(
    run_dir: RunDir, records: list[RequestRecord], texts: dict, capture_text: bool
) -> None:
    append_jsonl_gz(run_dir.requests_path, [asdict(r) for r in records])
    if capture_text and texts:
        append_jsonl_gz(run_dir.text_path, [
            {"key": k, "text": v} for k, v in texts.items()
        ])


async def _resolve_model(client: httpx.AsyncClient) -> str | None:
    try:
        r = await client.get("/v1/models")
        if r.status_code != 200:
            return None
        data = r.json().get("data") or []
        return data[0]["id"] if data else None
    except (httpx.HTTPError, KeyError, IndexError, ValueError):
        return None


def run_from_config_path(
    path: str | Path,
    *,
    attach_url: str | None = None,
    capture_text: bool = False,
    history_dir: Path | None = None,
    workloads_dir: Path | None = None,
) -> RunHandle:
    """The headless CLI entry: parse, execute, return the finished handle."""
    from cantollm.bench.config import load_run_config

    config = load_run_config(path)
    return asyncio.run(execute_run(
        config,
        attach_url=attach_url,
        capture_text=capture_text,
        history_dir=history_dir,
        workloads_dir=workloads_dir,
    ))
