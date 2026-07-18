"""Contract tests for GET /debug/engine-stats (bench-spec.md §4).

Two registries: FakeEngine (no accumulator → available:false, the
sequential shape) and a real in-process ContinuousBatchingEngine on the
toy scheduler (available:true with real steps). The endpoint is outside
both dialects, so errors keep FastAPI's default {"detail": ...} shape.
"""

from __future__ import annotations

import asyncio

import httpx

from cantollm.api import create_app
from cantollm.engine import ContinuousBatchingEngine
from tests.fakes import FakeEngine, FakeRegistry, FakeRuntime, FakeTokenizer
from tests.test_engine_step_stats import make_scheduler


def _transport_client(registry) -> httpx.AsyncClient:
    app = create_app(registry)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


def test_engine_without_accumulator_reports_unavailable():
    registry = FakeRegistry(
        entries={"test-model": (FakeEngine(script=[]), FakeRuntime(FakeTokenizer()))}
    )

    async def main():
        async with _transport_client(registry) as client:
            r = await client.get("/debug/engine-stats")
            assert r.status_code == 200
            return r.json()

    body = asyncio.run(main())
    assert body == {
        "schema_version": 1, "model": "test-model", "available": False,
    }


def test_unknown_model_404_and_multi_model_needs_param():
    tokenizer = FakeTokenizer()
    registry = FakeRegistry(entries={
        "m1": (FakeEngine(script=[]), FakeRuntime(tokenizer)),
        "m2": (FakeEngine(script=[]), FakeRuntime(tokenizer)),
    })

    async def main():
        async with _transport_client(registry) as client:
            ambiguous = await client.get("/debug/engine-stats")
            missing = await client.get("/debug/engine-stats", params={"model": "nope"})
            explicit = await client.get("/debug/engine-stats", params={"model": "m1"})
            return ambiguous, missing, explicit

    ambiguous, missing, explicit = asyncio.run(main())
    assert ambiguous.status_code == 400
    assert missing.status_code == 404
    assert explicit.status_code == 200


def test_batched_engine_serves_real_steps_and_cursor():
    engine = ContinuousBatchingEngine(make_scheduler())
    registry = FakeRegistry(
        entries={"test-model": (engine, FakeRuntime(FakeTokenizer()))}
    )

    async def main():
        await engine.start()
        try:
            async with _transport_client(registry) as client:
                r = await client.post("/v1/chat/completions", json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 4,
                    "temperature": 0.0,
                    "ignore_eos": True,
                })
                assert r.status_code == 200
                first = (await client.get("/debug/engine-stats")).json()
                cursor = await client.get(
                    "/debug/engine-stats", params={"since": first["next_since"]}
                )
                return first, cursor.json()
        finally:
            await engine.shutdown()

    first, after = asyncio.run(main())

    assert first["available"] is True
    assert first["engine_kind"] == "batched-inprocess"
    assert first["capacity"] == {"max_batch": 2, "max_seq_len": 32}
    # ignore_eos + FakeTokenizer stop ids outside toy vocab → exactly max_tokens.
    assert first["totals"]["output_tokens"] == 4
    assert first["totals"]["steps"] >= 1
    assert first["steps"] and all(s["rows"] >= 1 for s in first["steps"])
    assert len(first["itl"]) == 3

    assert after["steps"] == [] and after["itl"] == []
    assert after["next_since"] == first["next_since"]
