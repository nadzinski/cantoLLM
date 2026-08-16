"""Supervisor loop (3.5 chunk 2d): auto-restart, backoff/give-up, admin.

Fake-engine factories exercise the loop's transitions deterministically
(policy knobs shrunk per-handle); one test drives the real IPC path with a
child whose scheduler dies post-Ready (DyingScheduler), proving the
mux._fail -> on_failed -> rebuild chain end to end.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import httpx

from cantollm.api import create_app
from cantollm.engine.batching import EngineProcessClient
from cantollm.lifecycle import BuiltEngine
from cantollm.registry import EngineRegistry
from tests.fakes import (
    FakeEngine,
    FakeRuntime,
    FakeTokenizer,
    ScriptStep,
    parse_sse,
    wait_ready,
)
from tests.test_process_engine import dying_factory


@dataclass
class SupervisedFakeEngine(FakeEngine):
    """FakeEngine plus the on_failed hook the supervisor wires."""

    on_failed: object = field(default=None)


def _registry_with(factory) -> EngineRegistry:
    registry = EngineRegistry()
    registry.register("m", factory, max_request_tokens=64)
    handle = registry.get("m").handle
    handle.backoff_initial_s = 0.01
    handle.backoff_cap_s = 0.05
    return registry


def _client(registry):
    app = create_app(registry)
    transport = httpx.ASGITransport(app=app)
    return app, httpx.AsyncClient(transport=transport, base_url="http://test")


def _messages_body(stream: bool = False, max_tokens: int = 4) -> dict:
    return {
        "model": "m",
        "max_tokens": max_tokens,
        "stream": stream,
        "messages": [{"role": "user", "content": "hi"}],
    }


async def _wait_state(client, state: str, timeout_s: float = 10.0):
    for _ in range(int(timeout_s / 0.02)):
        r = await client.get("/ready")
        m = r.json()["models"]["m"]
        if m["state"] == state:
            return m
        await asyncio.sleep(0.02)
    raise AssertionError(f"never reached state {state}; last: {m}")


# ── death -> auto-restart ────────────────────────────────────────────


def test_death_triggers_auto_restart():
    engines: list[SupervisedFakeEngine] = []

    def factory() -> BuiltEngine:
        engine = SupervisedFakeEngine(script=[ScriptStep(token_id=2000)])
        engines.append(engine)
        return BuiltEngine(engine, FakeRuntime(FakeTokenizer()))

    registry = _registry_with(factory)

    async def main():
        app, client = _client(registry)
        async with app.router.lifespan_context(app):
            async with client:
                await wait_ready(client)
                handle = registry.get("m").handle
                assert handle.generation == 1
                # Simulate the mux._fail tail: engine reports batch-wide
                # death through the wired hook (runs on the loop).
                engines[0].on_failed("child exploded")
                await wait_ready(client)
                assert handle.generation == 2
                assert handle.consecutive_failures == 1  # not yet stable-reset
                assert handle.last_error == "child exploded"
                assert engines[0].shutdown_called  # retired before rebuild
                # New generation serves.
                r = await client.post("/v1/messages", json=_messages_body())
                assert r.status_code == 200
                assert engines[1].last_request is not None

    asyncio.run(main())
    assert len(engines) == 2


# ── give-up -> CRASHED -> manual restart ─────────────────────────────


def test_give_up_then_manual_restart_recovers():
    attempts = {"n": 0}

    def factory() -> BuiltEngine:
        attempts["n"] += 1
        if attempts["n"] <= 3:
            raise RuntimeError(f"build {attempts['n']} failed")
        return BuiltEngine(
            SupervisedFakeEngine(script=[ScriptStep(token_id=2000)]),
            FakeRuntime(FakeTokenizer()),
        )

    registry = _registry_with(factory)
    registry.get("m").handle.give_up_after = 2

    async def main():
        app, client = _client(registry)
        async with app.router.lifespan_context(app):
            async with client:
                m = await _wait_state(client, "crashed")
                assert m["consecutive_failures"] == 2
                assert "build 2 failed" in m["last_error"]
                assert m["hint"] == "POST /admin/restart"
                assert attempts["n"] == 2  # gave up, no burn loop

                r = await client.post("/admin/restart", json={})
                assert r.status_code == 202
                # Attempt 3 fails (counter freshly reset -> 1 < 2, so it
                # backs off instead of crashing), attempt 4 succeeds.
                await wait_ready(client)
                handle = registry.get("m").handle
                assert handle.generation == 1
                assert attempts["n"] == 4

    asyncio.run(main())


# ── reload: drain, swap, resume ──────────────────────────────────────


def test_reload_drains_inflight_then_swaps():
    engines: list[SupervisedFakeEngine] = []

    def factory() -> BuiltEngine:
        engine = SupervisedFakeEngine(script=[
            ScriptStep(token_id=2000), ScriptStep(sleep=0.4),
            ScriptStep(token_id=2001),
        ])
        engines.append(engine)
        return BuiltEngine(engine, FakeRuntime(FakeTokenizer()))

    registry = _registry_with(factory)

    async def main():
        app, client = _client(registry)
        async with app.router.lifespan_context(app):
            async with client:
                await wait_ready(client)
                handle = registry.get("m").handle
                post = asyncio.create_task(client.post(
                    "/v1/messages", json=_messages_body(stream=True)
                ))
                for _ in range(200):
                    if handle.inflight == 1:
                        break
                    await asyncio.sleep(0.01)
                assert handle.inflight == 1

                r = await client.post("/admin/reload", json={})
                assert r.status_code == 202
                assert r.json()["status"] == "accepted"

                # The in-flight stream drains to completion: no truncation.
                resp = await post
                assert resp.status_code == 200
                events = parse_sse(resp.text)
                kinds = [e.event for e in events if e.event != "ping"]
                assert kinds[-1] == "message_stop"
                assert "error" not in kinds

                await wait_ready(client)
                assert handle.generation == 2
                assert engines[0].shutdown_called
                # Fresh generation takes traffic.
                r = await client.post("/v1/messages", json=_messages_body())
                assert r.status_code == 200
                assert engines[1].last_request is not None

    asyncio.run(main())
    assert len(engines) == 2


def test_admin_409_while_building():
    import threading

    gate = threading.Event()

    def factory() -> BuiltEngine:
        gate.wait(timeout=15)
        return BuiltEngine(
            SupervisedFakeEngine(script=[ScriptStep(token_id=2000)]),
            FakeRuntime(FakeTokenizer()),
        )

    registry = _registry_with(factory)

    async def main():
        app, client = _client(registry)
        async with app.router.lifespan_context(app):
            async with client:
                for path in ("/admin/reload", "/admin/restart"):
                    r = await client.post(path, json={})
                    assert r.status_code == 409, path
                    assert "cannot" in r.json()["detail"]
                gate.set()
                await wait_ready(client)
                # From READY, restart behaves like reload.
                r = await client.post("/admin/restart", json={})
                assert r.status_code == 202
                await wait_ready(client)
                assert registry.get("m").handle.generation == 2

    asyncio.run(main())


def test_admin_unknown_model_404():
    registry = _registry_with(lambda: BuiltEngine(
        SupervisedFakeEngine(), FakeRuntime(FakeTokenizer())
    ))

    async def main():
        app, client = _client(registry)
        async with app.router.lifespan_context(app):
            async with client:
                r = await client.post("/admin/reload", json={"model": "nope"})
                assert r.status_code == 404

    asyncio.run(main())


# ── the real IPC path: dying child -> streams fail -> rebuild ────────


def test_process_engine_death_recovers_via_supervisor():
    runtime = FakeRuntime(FakeTokenizer())

    def factory() -> BuiltEngine:
        return BuiltEngine(EngineProcessClient(dying_factory), runtime)

    registry = EngineRegistry()
    registry.register("m", factory, max_request_tokens=64, runtime=runtime)
    handle = registry.get("m").handle
    handle.backoff_initial_s = 0.01

    async def main():
        app, client = _client(registry)
        async with app.router.lifespan_context(app):
            async with client:
                await wait_ready(client, timeout_s=60)
                assert handle.generation == 1
                # First request steps the child; DyingScheduler os._exit(3)s.
                r = await client.post(
                    "/v1/messages", json=_messages_body(stream=True)
                )
                assert r.status_code == 200
                events = parse_sse(r.text)
                assert any(e.event == "error" for e in events)
                # Supervisor notices, retires, rebuilds a fresh child.
                await wait_ready(client, timeout_s=60)
                assert handle.generation == 2
                assert handle.consecutive_failures >= 1
                assert "exit code 3" in handle.last_error

    asyncio.run(main())
