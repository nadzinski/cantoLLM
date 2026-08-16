"""Load-based admission (3.5 chunk 3): per-model in-flight cap, queue with
timeout, 429 + Retry-After. (Validation admission — the 400s — lives in
tests/test_admission.py; this file is the capacity side.)
"""

from __future__ import annotations

import asyncio

import httpx

from cantollm.api import create_app
from cantollm.lifecycle import BuiltEngine
from cantollm.registry import EngineRegistry
from tests.fakes import FakeEngine, FakeRuntime, FakeTokenizer, ScriptStep, wait_ready


def _slow_engine(sleep: float = 0.5) -> FakeEngine:
    return FakeEngine(script=[
        ScriptStep(token_id=2000), ScriptStep(sleep=sleep),
        ScriptStep(token_id=2001),
    ])


def _registry(engine, *, max_inflight, admission_timeout_s) -> EngineRegistry:
    runtime = FakeRuntime(FakeTokenizer())
    registry = EngineRegistry()
    registry.register(
        "m", lambda: BuiltEngine(engine, runtime), runtime=runtime,
        max_request_tokens=64, max_inflight=max_inflight,
        admission_timeout_s=admission_timeout_s,
    )
    return registry


def _client(registry):
    app = create_app(registry)
    transport = httpx.ASGITransport(app=app)
    return app, httpx.AsyncClient(
        transport=transport, base_url="http://test", timeout=30.0
    )


def _body(stream: bool = True) -> dict:
    return {
        "model": "m",
        "max_tokens": 4,
        "stream": stream,
        "messages": [{"role": "user", "content": "hi"}],
    }


def test_over_cap_gets_429_with_retry_after():
    registry = _registry(_slow_engine(0.8), max_inflight=1,
                         admission_timeout_s=0.1)

    async def main():
        app, client = _client(registry)
        async with app.router.lifespan_context(app):
            async with client:
                await wait_ready(client)
                first = asyncio.create_task(
                    client.post("/v1/messages", json=_body())
                )
                handle = registry.get("m").handle
                for _ in range(200):
                    if handle.inflight == 1:
                        break
                    await asyncio.sleep(0.01)
                # Anthropic dialect: rate_limit_error envelope.
                r = await client.post("/v1/messages", json=_body())
                assert r.status_code == 429
                assert r.json()["error"]["type"] == "rate_limit_error"
                assert int(r.headers["Retry-After"]) >= 1
                # OpenAI dialect: same status, its own envelope.
                r = await client.post("/v1/chat/completions", json={
                    "model": "m", "max_tokens": 4,
                    "messages": [{"role": "user", "content": "hi"}],
                })
                assert r.status_code == 429
                assert r.json()["error"]["type"] == "rate_limit_error"
                assert (await first).status_code == 200

    asyncio.run(main())


def test_queued_request_admits_when_slot_frees():
    registry = _registry(_slow_engine(0.3), max_inflight=1,
                         admission_timeout_s=10.0)

    async def main():
        app, client = _client(registry)
        async with app.router.lifespan_context(app):
            async with client:
                await wait_ready(client)
                first = asyncio.create_task(
                    client.post("/v1/messages", json=_body())
                )
                handle = registry.get("m").handle
                for _ in range(200):
                    if handle.inflight == 1:
                        break
                    await asyncio.sleep(0.01)
                # This one queues behind the slot and then succeeds: a
                # burst that clears within the timeout eats no rejection.
                second = await client.post("/v1/messages", json=_body())
                assert second.status_code == 200
                assert (await first).status_code == 200
                assert handle.inflight == 0

    asyncio.run(main())


def test_capacity_recycles_across_sequential_requests():
    engine = FakeEngine(script=[ScriptStep(token_id=2000)])
    registry = _registry(engine, max_inflight=2, admission_timeout_s=0.1)

    async def main():
        app, client = _client(registry)
        async with app.router.lifespan_context(app):
            async with client:
                await wait_ready(client)
                for _ in range(6):  # 3x the cap, strictly sequential
                    r = await client.post("/v1/messages", json=_body(stream=False))
                    assert r.status_code == 200

    asyncio.run(main())


def test_drain_beats_a_queued_waiter():
    registry = _registry(_slow_engine(0.5), max_inflight=1,
                         admission_timeout_s=10.0)

    async def main():
        app, client = _client(registry)
        async with app.router.lifespan_context(app):
            async with client:
                await wait_ready(client)
                handle = registry.get("m").handle
                first = asyncio.create_task(
                    client.post("/v1/messages", json=_body())
                )
                for _ in range(200):
                    if handle.inflight == 1:
                        break
                    await asyncio.sleep(0.01)
                waiter = asyncio.create_task(
                    client.post("/v1/messages", json=_body())
                )
                await asyncio.sleep(0.05)   # waiter is queued on the sem
                handle.begin_drain()
                # First request drains to completion; the waiter acquires a
                # slot afterwards but the re-check sees DRAINING -> 503.
                assert (await first).status_code == 200
                r = await waiter
                assert r.status_code == 503
                assert "draining" in r.json()["error"]["message"]

    asyncio.run(main())


def test_validation_failures_release_the_slot():
    engine = FakeEngine(script=[ScriptStep(token_id=2000)])
    registry = _registry(engine, max_inflight=1, admission_timeout_s=0.1)

    async def main():
        app, client = _client(registry)
        async with app.router.lifespan_context(app):
            async with client:
                await wait_ready(client)
                # Over the token cap -> 400 via check_admission; the slot
                # must come back or the next request would 429.
                bad = dict(_body(stream=False), max_tokens=100)
                for _ in range(3):
                    r = await client.post("/v1/messages", json=bad)
                    assert r.status_code == 400
                r = await client.post("/v1/messages", json=_body(stream=False))
                assert r.status_code == 200
                assert registry.get("m").handle.inflight == 0

    asyncio.run(main())
