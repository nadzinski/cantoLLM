"""Watchdog + stats continuity (3.5 chunk 2e).

A wedged child (alive, requests pending, zero step progress) is invisible
to every other liveness mechanism; the watchdog kills it and the supervisor
rebuilds. The stats half proves the persistent accumulator: one object
across generations, seq cursor monotonic through a restart, so
/debug/engine-stats `since` paging (the bench scrape) never silently
blanks.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import httpx

from cantollm.api import create_app
from cantollm.engine.batching import EngineProcessClient
from cantollm.engine.batching.stats import (
    EngineStatsAccumulator,
    StepStats,
    StepUpdate,
)
from cantollm.engine.types import TokenEvent
from cantollm.lifecycle import BuiltEngine
from cantollm.registry import EngineRegistry
from tests.fakes import FakeEngine, FakeRuntime, FakeTokenizer, parse_sse, wait_ready
from tests.test_process_engine import NeverFinishingScheduler

# ── module-level factory (pickled by reference across spawn) ──────────


class WedgingScheduler(NeverFinishingScheduler):
    """Alive but hung: accepts requests, then step() never returns."""

    def step(self) -> list[TokenEvent]:
        time.sleep(600)
        return []  # pragma: no cover - the watchdog kills us first


def wedging_factory():
    return WedgingScheduler()


# ── the watchdog catches a wedged child ──────────────────────────────


def test_watchdog_kills_wedged_child_and_supervisor_recovers():
    runtime = FakeRuntime(FakeTokenizer())

    def factory() -> BuiltEngine:
        return BuiltEngine(EngineProcessClient(wedging_factory), runtime)

    registry = EngineRegistry()
    registry.register(
        "m", factory, max_request_tokens=64, runtime=runtime,
        watchdog_timeout_s=0.5,
    )
    handle = registry.get("m").handle
    handle.watchdog_poll_s = 0.1
    handle.backoff_initial_s = 0.01

    async def main():
        app = create_app(registry)
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                await wait_ready(client, timeout_s=60)
                assert handle.generation == 1
                # This request wedges the child: it steps into a 600 s
                # sleep with the request pending. Only the watchdog can
                # notice.
                r = await client.post("/v1/messages", json={
                    "model": "m", "max_tokens": 8, "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                })
                assert r.status_code == 200
                events = parse_sse(r.text)
                assert any(e.event == "error" for e in events)
                await wait_ready(client, timeout_s=60)
                assert handle.generation == 2
                assert "exit code" in handle.last_error

    asyncio.run(main())


def test_watchdog_idle_engine_never_fires():
    # No pending work -> staleness is irrelevant and the child survives.
    runtime = FakeRuntime(FakeTokenizer())

    def factory() -> BuiltEngine:
        return BuiltEngine(EngineProcessClient(wedging_factory), runtime)

    registry = EngineRegistry()
    registry.register(
        "m", factory, max_request_tokens=64, runtime=runtime,
        watchdog_timeout_s=0.2,
    )
    handle = registry.get("m").handle
    handle.watchdog_poll_s = 0.05

    async def main():
        app = create_app(registry)
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                await wait_ready(client, timeout_s=60)
                await asyncio.sleep(1.0)  # >> timeout, idle the whole time
                assert handle.generation == 1
                assert (await client.get("/ready")).status_code == 200

    asyncio.run(main())


# ── stats continuity ─────────────────────────────────────────────────


def _step(seq: int) -> StepUpdate:
    return StepUpdate(events=[], stats=StepStats(
        seq=seq, t_wall=0.0, t_perf=float(seq), dur_s=0.01, rows=1,
        occupied_slots=1, queue_depth=0, kv_tokens=4, prefill_tokens=0,
        decode_tokens=1,
    ))


def test_accumulator_seq_rebase_across_generations():
    acc = EngineStatsAccumulator()
    for seq in range(3):
        acc.record(_step(seq))
    first = acc.read(-1)
    assert [s["seq"] for s in first["steps"]] == [0, 1, 2]
    cursor = first["next_since"]

    acc.note_generation_start()  # child restarted; its counter is 0 again
    for seq in range(3):
        acc.record(_step(seq))
    page = acc.read(cursor)
    assert [s["seq"] for s in page["steps"]] == [3, 4, 5]
    assert page["totals"]["steps"] == 6
    assert page["next_since"] == 5


@dataclass
class StatsFakeEngine(FakeEngine):
    engine_stats: EngineStatsAccumulator = field(
        default_factory=lambda: EngineStatsAccumulator(engine_kind="fake")
    )
    on_failed: object = None


def test_handle_keeps_one_accumulator_across_restart():
    engines: list[StatsFakeEngine] = []

    def factory() -> BuiltEngine:
        engine = StatsFakeEngine()
        engines.append(engine)
        return BuiltEngine(engine, FakeRuntime(FakeTokenizer()))

    registry = EngineRegistry()
    registry.register("m", factory, max_request_tokens=64)
    handle = registry.get("m").handle
    handle.backoff_initial_s = 0.01

    async def main():
        app = create_app(registry)
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                await wait_ready(client)
                # Generation 1 donated its accumulator to the handle.
                assert handle.engine_stats is engines[0].engine_stats
                handle.engine_stats.record(_step(0))
                scrape = await client.get("/debug/engine-stats")
                cursor = scrape.json()["next_since"]
                assert cursor == 0

                engines[0].on_failed("boom")  # death -> rebuild
                await wait_ready(client)
                # Generation 2 got the same object, seq-rebased.
                assert engines[1].engine_stats is handle.engine_stats
                engines[1].engine_stats.record(_step(0))
                page = (await client.get(
                    f"/debug/engine-stats?since={cursor}"
                )).json()
                assert [s["seq"] for s in page["steps"]] == [1]
                assert page["available"] is True

    asyncio.run(main())
