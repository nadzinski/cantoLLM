"""OTel tracing (3.5 chunk 5): off-by-default, API spans, engine spans.

Uses the in-memory exporter with a SimpleSpanProcessor for determinism;
configure/shutdown pairs keep the module-global provider from leaking
between tests (and into the rest of the suite, which runs traced-off).
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager

import httpx
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from cantollm.api import create_app
from cantollm.engine import ContinuousBatchingEngine
from cantollm.engine.types import InferenceRequest, SamplingParams
from cantollm.lifecycle import BuiltEngine
from cantollm.obs import tracing
from cantollm.registry import EngineRegistry
from tests.fakes import FakeEngine, FakeRuntime, FakeTokenizer, ScriptStep, wait_ready
from tests.test_process_engine import toy_scheduler_factory


@contextmanager
def _in_memory_tracing():
    exporter = InMemorySpanExporter()
    tracing.configure_tracing("cantollm-test", exporter=exporter)
    try:
        yield exporter
    finally:
        tracing.shutdown_tracing()


def _real_registry(engine) -> EngineRegistry:
    runtime = FakeRuntime(FakeTokenizer())
    registry = EngineRegistry()
    registry.register(
        "m", lambda: BuiltEngine(engine, runtime),
        runtime=runtime, max_request_tokens=64,
    )
    return registry


def _body() -> dict:
    return {
        "model": "m", "max_tokens": 4, "stream": False,
        "messages": [{"role": "user", "content": "hi"}],
    }


async def _one_request(registry) -> None:
    app = create_app(registry)
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            await wait_ready(client)
            r = await client.post(
                "/v1/messages", json=_body(),
                headers={"X-Request-ID": "rid-traced"},
            )
            assert r.status_code == 200


def test_tracing_off_leaves_requests_untouched():
    assert tracing.get_tracer() is None
    engine = FakeEngine(script=[ScriptStep(token_id=2000)])
    asyncio.run(_one_request(_real_registry(engine)))
    assert engine.last_request.trace_context is None


def test_api_spans_root_and_tokenize_with_carrier():
    engine = FakeEngine(script=[ScriptStep(token_id=2000)])
    registry = _real_registry(engine)
    with _in_memory_tracing() as exporter:
        asyncio.run(_one_request(registry))
        spans = {s.name: s for s in exporter.get_finished_spans()}

    assert "anthropic m" in spans and "tokenize" in spans
    root = spans["anthropic m"]
    tok = spans["tokenize"]
    assert tok.parent is not None
    assert tok.parent.span_id == root.context.span_id
    assert root.attributes["request_id"] == "rid-traced"
    assert root.attributes["output_tokens"] == 1
    assert root.attributes["finish_reason"] == "end_turn"
    assert any(e.name == "first_token" for e in root.events)
    # The engine got a W3C carrier pointing at the root's trace.
    carrier = engine.last_request.trace_context
    assert carrier is not None
    assert format(root.context.trace_id, "032x") in carrier["traceparent"]


def test_engine_spans_through_the_inprocess_drive_loop():
    async def main(exporter):
        engine = ContinuousBatchingEngine(toy_scheduler_factory())
        await engine.start()
        try:
            tracer = tracing.get_tracer()
            root = tracer.start_span("root")
            req = InferenceRequest(
                request_id="r-traced",
                prompt_token_ids=[1, 2, 3],
                sampling_params=SamplingParams.from_temperature_top_p(0.0, 1.0),
                max_tokens=4,
                stop_token_ids=set(),
                trace_context=tracing.inject_span_context(root),
            )
            events = [evt async for evt in engine.submit(req)]
            assert events[-1].finish_reason is not None
            root.end()
        finally:
            await engine.shutdown()
        return root

    with _in_memory_tracing() as exporter:
        root = asyncio.run(main(exporter))
        spans = exporter.get_finished_spans()

    by_name = {}
    for s in spans:
        by_name.setdefault(s.name, []).append(s)
    assert "queue" in by_name
    assert "prefill chunk" in by_name
    assert "decode" in by_name
    trace_id = root.context.trace_id
    for name in ("queue", "prefill chunk", "decode"):
        for s in by_name[name]:
            assert s.context.trace_id == trace_id, name
    decode = by_name["decode"][0]
    assert decode.attributes["output_tokens"] == 4
    assert any(e.name == "first_token" for e in decode.events)
    chunk = by_name["prefill chunk"][0]
    assert chunk.attributes["width"] == 3
    # Timestamps are coherent: queue ends before the chunk ends.
    assert by_name["queue"][0].end_time <= chunk.end_time


def test_untraced_request_produces_no_engine_spans():
    async def main():
        engine = ContinuousBatchingEngine(toy_scheduler_factory())
        await engine.start()
        try:
            req = InferenceRequest(
                request_id="r-plain",
                prompt_token_ids=[1, 2, 3],
                sampling_params=SamplingParams.from_temperature_top_p(0.0, 1.0),
                max_tokens=2,
                stop_token_ids=set(),
            )
            _ = [evt async for evt in engine.submit(req)]
        finally:
            await engine.shutdown()

    with _in_memory_tracing() as exporter:
        asyncio.run(main())
        assert exporter.get_finished_spans() == ()

    asyncio.run(main())  # and with tracing fully off, nothing breaks
