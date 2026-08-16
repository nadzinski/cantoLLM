"""Progress protocol (3.5 chunk 2b): report()/sink, IPC transport, /ready.

Three layers: the throttle in cantollm.progress, Progress messages riding
the engine-process event queue ahead of Ready, and /ready surfacing the
latest snapshot mid-warm through the ambient-sink path (contextvar
inherited across asyncio.to_thread).
"""

from __future__ import annotations

import asyncio
import threading

import httpx

from cantollm.api import create_app
from cantollm.engine.batching import EngineProcessClient
from cantollm.lifecycle import BuiltEngine
from cantollm.progress import Progress, bind_sink, report, unbind_sink
from cantollm.registry import EngineRegistry
from tests.fakes import FakeEngine, FakeRuntime, FakeTokenizer, wait_ready


# ── module-level factory (pickled by reference across spawn) ──────────


def staged_progress_factory():
    from cantollm.progress import report as _report

    from tests.test_process_engine import toy_scheduler_factory

    _report("load", 0, 3, "downloading weights")
    _report("load", 3, 3, "model on device")
    _report("sweep", 2, 2)
    return toy_scheduler_factory()


# ── report() throttle ────────────────────────────────────────────────


def _bound_recorder():
    seen: list[Progress] = []
    token = bind_sink(seen.append)
    return seen, token


def test_no_sink_is_a_noop():
    report("sweep", 1, 10)  # must not raise


def test_stage_change_and_completion_always_emit():
    seen, token = _bound_recorder()
    try:
        report("load", 0, 3)          # stage change -> emit
        report("load", 1, 3)          # throttled (same stage, < 0.5 s)
        report("load", 2, 3)          # throttled
        report("load", 3, 3)          # completion -> emit
        report("sweep", 1, 100)       # stage change -> emit
        for i in range(2, 100):
            report("sweep", i, 100)   # throttled
        report("sweep", 100, 100)     # completion -> emit
    finally:
        unbind_sink(token)
    stages = [(p.stage, p.done) for p in seen]
    assert stages == [("load", 0), ("load", 3), ("sweep", 1), ("sweep", 100)]
    assert all(p.total is not None for p in seen)


def test_elapsed_is_per_stage():
    seen, token = _bound_recorder()
    try:
        report("load", 0, 3)
        report("sweep", 1, 1)
    finally:
        unbind_sink(token)
    assert seen[-1].stage == "sweep"
    assert seen[-1].elapsed_s < 0.4  # measured from the sweep start, not load


# ── over the IPC boundary ────────────────────────────────────────────


def test_progress_arrives_before_ready_over_ipc():
    seen: list[Progress] = []

    async def main():
        client = EngineProcessClient(
            staged_progress_factory, on_progress=seen.append
        )
        await client.start()
        try:
            # start() returns only at Ready; everything recorded so far
            # was therefore pre-Ready by construction.
            assert [(p.stage, p.done) for p in seen] == [
                ("load", 0), ("load", 3), ("sweep", 2),
            ]
            assert client.engine_stats.load_seconds is not None
        finally:
            await client.shutdown()

    asyncio.run(main())


# ── /ready shows progress mid-warm (ambient sink through to_thread) ──


def test_ready_reports_progress_mid_warm():
    gate = threading.Event()
    reported = threading.Event()
    runtime = FakeRuntime(FakeTokenizer())
    engine = FakeEngine()

    def factory() -> BuiltEngine:
        report("sweep", 40, 477)
        reported.set()
        gate.wait(timeout=15)
        return BuiltEngine(engine, runtime)

    registry = EngineRegistry()
    registry.register("m", factory, max_request_tokens=64)

    async def main():
        app = create_app(registry)
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                await asyncio.to_thread(reported.wait, 15)
                # The sink hop is call_soon_threadsafe; give the loop a tick.
                for _ in range(100):
                    r = await client.get("/ready")
                    m = r.json()["models"]["m"]
                    if "progress" in m:
                        break
                    await asyncio.sleep(0.01)
                assert r.status_code == 503
                assert m["progress"]["stage"] == "sweep"
                assert m["progress"]["done"] == 40
                assert m["progress"]["total"] == 477

                # The 503 error detail names the stage too.
                r = await client.post("/v1/messages", json={
                    "model": "m", "max_tokens": 4,
                    "messages": [{"role": "user", "content": "hi"}],
                })
                assert "sweep 40/477" in r.json()["error"]["message"]

                gate.set()
                await wait_ready(client)
                # Progress clears once ready.
                r = await client.get("/ready")
                assert "progress" not in r.json()["models"]["m"]

    asyncio.run(main())
