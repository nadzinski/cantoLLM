"""Phase 3.5 chunk 1: structured logging, X-Request-ID, /version.

JSON formatter shape, configure_logging idempotence, the request-id
middleware (inbound header honored end to end into the engine's
InferenceRequest; minted when absent), and the /version fingerprint route.
"""

from __future__ import annotations

import asyncio
import json
import logging

import httpx

from cantollm.api import create_app
from cantollm.obs.logging import JsonFormatter, configure_logging, request_id_var
from tests.fakes import FakeEngine, FakeRegistry, FakeRuntime, FakeTokenizer, ScriptStep

# ── Helpers ──────────────────────────────────────────────────────────


def _client(engine: FakeEngine) -> httpx.AsyncClient:
    runtime = FakeRuntime(tokenizer=FakeTokenizer())
    registry = FakeRegistry(entries={"test-model": (engine, runtime)})
    app = create_app(registry)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _messages_body() -> dict:
    return {
        "model": "test-model",
        "max_tokens": 4,
        "stream": False,
        "messages": [{"role": "user", "content": "hi"}],
    }


def _run(coro):
    return asyncio.run(coro)


def _format(record_msg: str, **record_kwargs) -> dict:
    record = logging.LogRecord(
        name="cantollm.test", level=logging.INFO, pathname=__file__, lineno=1,
        msg=record_msg, args=(), exc_info=record_kwargs.pop("exc_info", None),
    )
    for k, v in record_kwargs.items():
        setattr(record, k, v)
    return json.loads(JsonFormatter("api").format(record))


# ── JsonFormatter ────────────────────────────────────────────────────


def test_json_formatter_basic_shape():
    payload = _format("hello %s" % "world")
    assert payload["msg"] == "hello world"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "cantollm.test"
    assert payload["process"] == "api"
    assert payload["ts"].endswith("Z")
    assert "request_id" not in payload


def test_json_formatter_reads_request_id_contextvar():
    token = request_id_var.set("rid-ctx")
    try:
        assert _format("x")["request_id"] == "rid-ctx"
    finally:
        request_id_var.reset(token)


def test_json_formatter_record_attr_wins_over_contextvar():
    token = request_id_var.set("rid-ctx")
    try:
        assert _format("x", request_id="rid-attr")["request_id"] == "rid-attr"
    finally:
        request_id_var.reset(token)


def test_json_formatter_exception_text():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        payload = _format("failed", exc_info=sys.exc_info())
    assert "ValueError: boom" in payload["exc"]


def test_configure_logging_is_idempotent():
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        configure_logging("api")
        configure_logging("api")
        ours = [h for h in root.handlers if getattr(h, "_cantollm_json", False)]
        assert len(ours) == 1
    finally:
        for h in list(root.handlers):
            if getattr(h, "_cantollm_json", False):
                root.removeHandler(h)
        for h in before:
            if h not in root.handlers:
                root.addHandler(h)


# ── X-Request-ID middleware ──────────────────────────────────────────


def test_inbound_request_id_reaches_engine_and_response():
    engine = FakeEngine(script=[ScriptStep(token_id=2000)])

    async def run():
        async with _client(engine) as client:
            r = await client.post(
                "/v1/messages", json=_messages_body(),
                headers={"X-Request-ID": "rid-inbound-42"},
            )
            assert r.status_code == 200
            assert r.headers["x-request-id"] == "rid-inbound-42"

    _run(run())
    assert engine.last_request.request_id == "rid-inbound-42"


def test_minted_request_id_when_header_absent():
    engine = FakeEngine(script=[ScriptStep(token_id=2000)])

    async def run():
        async with _client(engine) as client:
            r = await client.post("/v1/messages", json=_messages_body())
            assert r.status_code == 200
            rid = r.headers["x-request-id"]
            assert len(rid) == 32 and all(c in "0123456789abcdef" for c in rid)
            return rid

    rid = _run(run())
    assert engine.last_request.request_id == rid


def test_request_ids_distinct_across_requests():
    engine = FakeEngine(script=[ScriptStep(token_id=2000)])

    async def run():
        async with _client(engine) as client:
            r1 = await client.post("/v1/messages", json=_messages_body())
            r2 = await client.post("/v1/messages", json=_messages_body())
            assert r1.headers["x-request-id"] != r2.headers["x-request-id"]

    _run(run())


def test_openai_dialect_carries_request_id_too():
    engine = FakeEngine(script=[ScriptStep(token_id=2000)])
    body = {
        "model": "test-model",
        "max_tokens": 4,
        "stream": False,
        "messages": [{"role": "user", "content": "hi"}],
    }

    async def run():
        async with _client(engine) as client:
            r = await client.post(
                "/v1/chat/completions", json=body,
                headers={"X-Request-ID": "rid-openai"},
            )
            assert r.status_code == 200
            assert r.headers["x-request-id"] == "rid-openai"

    _run(run())
    assert engine.last_request.request_id == "rid-openai"


# ── /version ─────────────────────────────────────────────────────────


def test_version_shape_and_engine_class():
    engine = FakeEngine()

    async def run():
        async with _client(engine) as client:
            r = await client.get("/version")
            assert r.status_code == 200
            payload = r.json()
            assert payload["name"] == "cantollm"
            for key in ("git_sha", "git_dirty", "python", "torch", "device_name"):
                assert key in payload
            assert payload["models"] == {"test-model": {"engine": "FakeEngine"}}
            # Cached: a second hit returns the identical object shape.
            assert (await client.get("/version")).json() == payload

    _run(run())
