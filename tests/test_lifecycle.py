"""Lifecycle core (3.5 chunk 2a): background start, /ready, 503 gating.

Real EngineRegistry + real EngineHandle supervisors; the engines behind the
factories are fakes. A threading.Event gates the factory (it runs on the
supervisor's worker thread), which lets tests observe the pre-ready window
deterministically.
"""

from __future__ import annotations

import asyncio
import threading

import httpx

from cantollm.api import create_app
from cantollm.api.common import tracked_events
from cantollm.lifecycle import BuiltEngine, EngineHandle, EngineState
from cantollm.registry import EngineRegistry
from tests.fakes import FakeEngine, FakeRuntime, FakeTokenizer, ScriptStep, wait_ready


def _registry(factory, *, runtime=None) -> EngineRegistry:
    registry = EngineRegistry()
    registry.register("m", factory, max_request_tokens=64, runtime=runtime)
    return registry


def _app_client(registry):
    app = create_app(registry)
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")
    return app, client


def _messages_body(stream: bool = False) -> dict:
    return {
        "model": "m",
        "max_tokens": 4,
        "stream": stream,
        "messages": [{"role": "user", "content": "hi"}],
    }


def _gated_factory(gate: threading.Event, engine: FakeEngine):
    runtime = FakeRuntime(FakeTokenizer())

    def factory() -> BuiltEngine:
        gate.wait(timeout=15)
        return BuiltEngine(engine, runtime)

    return factory


# ── /ready flip and 503 gating ───────────────────────────────────────


def test_ready_flips_and_gates_inference():
    gate = threading.Event()
    engine = FakeEngine(script=[ScriptStep(token_id=2000)])
    registry = _registry(_gated_factory(gate, engine))

    async def main():
        app, client = _app_client(registry)
        async with app.router.lifespan_context(app):
            async with client:
                r = await client.get("/ready")
                assert r.status_code == 503
                body = r.json()
                assert body["status"] in ("starting", "warming")
                assert body["models"]["m"]["generation"] == 0

                # Inference is gated with the Anthropic 503 envelope.
                r = await client.post("/v1/messages", json=_messages_body())
                assert r.status_code == 503
                assert r.json()["error"]["type"] == "overloaded_error"
                assert "Retry-After" in r.headers

                # OpenAI dialect gets its own envelope shape.
                r = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "m", "max_tokens": 4,
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
                assert r.status_code == 503
                assert r.json()["error"]["type"] == "server_error"
                assert "Retry-After" in r.headers

                # /health and /v1/models answer while warming.
                assert (await client.get("/health")).status_code == 200
                assert (await client.get("/v1/models")).status_code == 200

                gate.set()
                await wait_ready(client)
                r = await client.get("/ready")
                assert r.json()["models"]["m"]["generation"] == 1

                r = await client.post("/v1/messages", json=_messages_body())
                assert r.status_code == 200

    asyncio.run(main())
    assert engine.started
    assert engine.shutdown_called  # lifespan exit stopped the handle


def test_factory_failure_lands_crashed():
    def factory() -> BuiltEngine:
        raise RuntimeError("weights exploded")

    registry = _registry(factory)

    async def main():
        app, client = _app_client(registry)
        async with app.router.lifespan_context(app):
            async with client:
                for _ in range(200):
                    r = await client.get("/ready")
                    if r.json()["models"]["m"]["state"] == "crashed":
                        break
                    await asyncio.sleep(0.02)
                body = r.json()
                assert r.status_code == 503
                assert body["status"] == "crashed"
                m = body["models"]["m"]
                assert "weights exploded" in m["last_error"]
                assert m["consecutive_failures"] == 1
                assert m["hint"] == "POST /admin/restart"

                r = await client.post("/v1/messages", json=_messages_body())
                assert r.status_code == 503
                assert "weights exploded" in r.json()["error"]["message"]

    asyncio.run(main())


# ── in-flight tickets ────────────────────────────────────────────────


def test_ticket_close_is_idempotent():
    handle = EngineHandle("m", lambda: None)
    t1 = handle.begin_request()
    t2 = handle.begin_request()
    assert handle.inflight == 2
    t1.close()
    t1.close()
    assert handle.inflight == 1
    t2.close()
    assert handle.inflight == 0


def test_tracked_events_closes_on_exhaustion_and_disconnect():
    handle = EngineHandle("m", lambda: None)

    inner_finalized = []

    async def inner(n):
        try:
            for i in range(n):
                yield i
        finally:
            inner_finalized.append(True)

    async def main():
        # Full drain closes the ticket at exhaustion.
        ticket = handle.begin_request()
        assert [e async for e in tracked_events(ticket, inner(3))] == [0, 1, 2]
        assert handle.inflight == 0
        assert inner_finalized == [True]

        # aclose mid-stream (the disconnect path) closes ticket AND inner.
        ticket = handle.begin_request()
        wrapper = tracked_events(ticket, inner(100))
        assert (await anext(wrapper)) == 0
        await wrapper.aclose()
        assert handle.inflight == 0
        assert inner_finalized == [True, True]

    asyncio.run(main())


def test_inflight_counts_streaming_request_until_stream_ends():
    # The sleep keeps the engine stream open long enough to observe the
    # mid-flight claim; without it the adapter's producer drains the fake
    # instantly and the ticket is already closed by the first read.
    engine = FakeEngine(script=[
        ScriptStep(token_id=2000), ScriptStep(sleep=0.5), ScriptStep(token_id=2001),
    ])
    runtime = FakeRuntime(FakeTokenizer())
    registry = _registry(lambda: BuiltEngine(engine, runtime), runtime=runtime)

    async def main():
        app, client = _app_client(registry)
        async with app.router.lifespan_context(app):
            async with client:
                await wait_ready(client)
                handle = registry.get("m").handle
                # ASGITransport buffers responses, so observe the in-flight
                # claim by polling the handle while the request task runs
                # (the script's sleep guarantees a window).
                post = asyncio.create_task(
                    client.post("/v1/messages", json=_messages_body(stream=True))
                )
                seen_inflight = False
                for _ in range(200):
                    if handle.inflight == 1:
                        seen_inflight = True
                        break
                    await asyncio.sleep(0.01)
                resp = await post
                assert resp.status_code == 200
                assert seen_inflight, "in-flight claim never observed"
                for _ in range(100):
                    if handle.inflight == 0:
                        break
                    await asyncio.sleep(0.01)
                assert handle.inflight == 0

    asyncio.run(main())


def test_state_enum_render():
    # /ready serializes enum values as plain strings.
    assert EngineState.WARMING.value == "warming"
