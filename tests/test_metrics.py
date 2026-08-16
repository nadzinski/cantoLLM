"""/metrics (3.5 chunk 4): exposition, request metrics, step metrics,
HTTP histogram, event-loop lag, pull-time gauges.

Each app owns its own CollectorRegistry, so many apps per test process
never collide. GPU gauges are absent on machines without NVML; the tests
only assert the always-available families.
"""

from __future__ import annotations

import asyncio

import httpx

from cantollm.api import create_app
from cantollm.engine.batching.stats import StepStats, StepUpdate
from cantollm.lifecycle import BuiltEngine
from cantollm.registry import EngineRegistry
from tests.fakes import (
    FakeEngine,
    FakeRegistry,
    FakeRuntime,
    FakeTokenizer,
    ScriptStep,
    wait_ready,
)


def _real_registry(engine) -> EngineRegistry:
    runtime = FakeRuntime(FakeTokenizer())
    registry = EngineRegistry()
    registry.register(
        "m", lambda: BuiltEngine(engine, runtime),
        runtime=runtime, max_request_tokens=64,
    )
    return registry


def _client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _body() -> dict:
    return {
        "model": "m", "max_tokens": 4, "stream": False,
        "messages": [{"role": "user", "content": "hi"}],
    }


def test_metrics_endpoint_serves_with_fake_registry():
    # Fake entries have no handles: /metrics must still serve (process
    # gauges, HTTP histogram), with zero request/engine series.
    engine = FakeEngine(script=[ScriptStep(token_id=2000)])
    registry = FakeRegistry(
        entries={"test-model": (engine, FakeRuntime(FakeTokenizer()))}
    )
    app = create_app(registry)

    async def main():
        async with _client(app) as client:
            r = await client.get("/metrics")
            assert r.status_code == 200
            assert "cantollm_process_cpu_percent" in r.text
            # The scrape itself lands in the HTTP histogram next time.
            r = await client.get("/metrics")
            assert 'route="/metrics"' in r.text

    asyncio.run(main())


def test_request_metrics_observe_ttft_duration_and_reason():
    engine = FakeEngine(script=[ScriptStep(token_id=2000)])
    registry = _real_registry(engine)
    app = create_app(registry)

    async def main():
        async with app.router.lifespan_context(app):
            async with _client(app) as client:
                await wait_ready(client)
                for _ in range(3):
                    assert (await client.post(
                        "/v1/messages", json=_body()
                    )).status_code == 200
                text = (await client.get("/metrics")).text

        assert 'cantollm_request_ttft_seconds_count{model="m"} 3.0' in text
        assert 'cantollm_request_duration_seconds_count{model="m"} 3.0' in text
        assert ('cantollm_requests_finished_total'
                '{model="m",reason="end_turn"} 3.0') in text
        # In-flight gauge exists per model and reads zero at rest.
        assert 'cantollm_inflight_requests{model="m"} 0.0' in text

    asyncio.run(main())


def test_step_metrics_flow_through_the_stats_observer():
    engine = FakeEngine(script=[ScriptStep(token_id=2000)])
    registry = _real_registry(engine)
    app = create_app(registry)

    def step(seq: int, queue_depth: int) -> StepUpdate:
        return StepUpdate(events=[], stats=StepStats(
            seq=seq, t_wall=0.0, t_perf=float(seq) * 0.01, dur_s=0.008,
            rows=2, occupied_slots=2, queue_depth=queue_depth, kv_tokens=64,
            prefill_tokens=16, decode_tokens=2, graph_replayed=True,
        ))

    async def main():
        async with app.router.lifespan_context(app):
            async with _client(app) as client:
                await wait_ready(client)
                handle = registry.get("m").handle
                # gen-1 adoption created the accumulator lazily only for
                # engines that carry one; wire manually here by feeding
                # the handle a recorded accumulator.
                from cantollm.engine.batching.stats import (
                    EngineStatsAccumulator,
                )

                acc = EngineStatsAccumulator()
                acc.on_record = handle.stats_observer
                acc.record(step(0, queue_depth=3))
                acc.record(step(1, queue_depth=1))
                text = (await client.get("/metrics")).text

        assert 'cantollm_engine_steps_total{model="m"} 2.0' in text
        assert 'cantollm_engine_prefill_tokens_total{model="m"} 32.0' in text
        assert 'cantollm_engine_decode_tokens_total{model="m"} 4.0' in text
        assert 'cantollm_engine_graph_replays_total{model="m"} 2.0' in text
        assert 'cantollm_engine_queue_depth{model="m"} 1.0' in text
        assert 'cantollm_engine_step_duration_seconds_count{model="m"} 2.0' in text

    asyncio.run(main())


def test_event_loop_lag_gauge_updates_under_lifespan():
    engine = FakeEngine(script=[ScriptStep(token_id=2000)])
    registry = _real_registry(engine)
    app = create_app(registry)

    async def main():
        async with app.router.lifespan_context(app):
            async with _client(app) as client:
                await asyncio.sleep(0.3)  # a few sampler periods
                text = (await client.get("/metrics")).text
        # The gauge exists and carries a sampled (tiny) value.
        assert "cantollm_event_loop_lag_seconds" in text

    asyncio.run(main())


def test_http_metrics_label_by_route_template_and_status():
    engine = FakeEngine(script=[ScriptStep(token_id=2000)])
    registry = _real_registry(engine)
    app = create_app(registry)

    async def main():
        async with app.router.lifespan_context(app):
            async with _client(app) as client:
                await wait_ready(client)
                await client.post("/v1/messages", json=_body())
                await client.post("/v1/messages", json=dict(
                    _body(), max_tokens=100,   # over cap -> 400
                ))
                text = (await client.get("/metrics")).text

        assert ('cantollm_http_request_duration_seconds_count'
                '{method="POST",route="/v1/messages",status="200"} 1.0') in text
        assert ('cantollm_http_request_duration_seconds_count'
                '{method="POST",route="/v1/messages",status="400"} 1.0') in text

    asyncio.run(main())
