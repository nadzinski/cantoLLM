"""HTTP/SSE contract tests for the Messages API.

Locks down the observable behavior of `create_app` — request validation,
non-streaming response shape, SSE event sequence, thinking-block framing,
mid-stream abort, ping cadence — before Phase 1a reshapes the engine seams.
A fake engine and fake tokenizer stand in for the real model stack.
"""

from __future__ import annotations

import asyncio

import httpx

from cantollm.api import create_app
from cantollm.api.anthropic_adapter import render_sse
from tests.fakes import (
    STOP_TOKEN_ID,
    THINKING_END_ID,
    THINKING_START_ID,
    FakeEngine,
    FakeRegistry,
    FakeRuntime,
    FakeTokenizer,
    ScriptStep,
    parse_sse,
)

# ── Helpers ──────────────────────────────────────────────────────────


def _char_ids(text: str) -> list[int]:
    """Map each char to a unique token id (a=2000, b=2001, ...)."""
    return [2000 + (ord(c) - ord("a")) for c in text]


def _script_from_text(text: str) -> list[ScriptStep]:
    return [ScriptStep(token_id=tid) for tid in _char_ids(text)]


def _tokenizer_for(text: str) -> FakeTokenizer:
    id_to_text = {2000 + i: chr(ord("a") + i) for i in range(26)}
    return FakeTokenizer(id_to_text=id_to_text)


def _client(engine: FakeEngine, tokenizer: FakeTokenizer) -> httpx.AsyncClient:
    runtime = FakeRuntime(tokenizer=tokenizer)
    registry = FakeRegistry(entries={"test-model": (engine, runtime)})
    app = create_app(registry)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _messages_body(max_tokens: int, stream: bool) -> dict:
    return {
        "model": "test-model",
        "max_tokens": max_tokens,
        "stream": stream,
        "messages": [{"role": "user", "content": "hi"}],
    }


def _run(coro):
    return asyncio.run(coro)


# ── Structured content passes through unflattened ───────────────────


def test_structured_content_blocks_reach_tokenizer_unflattened():
    tokenizer = _tokenizer_for("a")
    engine = FakeEngine(script=_script_from_text("a"))

    body = {
        "model": "test-model",
        "max_tokens": 1,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first"},
                    {"type": "text", "text": "second"},
                ],
            }
        ],
    }

    async def run():
        async with _client(engine, tokenizer) as client:
            r = await client.post("/v1/messages", json=body)
            assert r.status_code == 200

    _run(run())

    assert tokenizer.last_messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"},
            ],
        }
    ]


def test_string_content_still_passes_through_as_string():
    tokenizer = _tokenizer_for("a")
    engine = FakeEngine(script=_script_from_text("a"))

    async def run():
        async with _client(engine, tokenizer) as client:
            r = await client.post("/v1/messages", json=_messages_body(max_tokens=1, stream=False))
            assert r.status_code == 200

    _run(run())

    assert tokenizer.last_messages == [{"role": "user", "content": "hi"}]


# ── 1. Non-streaming happy path ──────────────────────────────────────


def test_non_streaming_happy_path():
    text = "hello"
    tokenizer = _tokenizer_for(text)
    engine = FakeEngine(script=_script_from_text(text))

    async def run():
        async with _client(engine, tokenizer) as client:
            r = await client.post("/v1/messages", json=_messages_body(max_tokens=100, stream=False))
            assert r.status_code == 200
            return r.json()

    body = _run(run())

    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["model"] == "test-model"
    assert body["content"] == [{"type": "text", "text": "hello"}]
    assert body["stop_reason"] == "end_turn"
    assert body["usage"] == {"input_tokens": 3, "output_tokens": 5}


# ── 2. Non-streaming: max_tokens stop reason ─────────────────────────


def test_non_streaming_reports_max_tokens_stop_reason():
    text = "hello"  # 5 tokens
    tokenizer = _tokenizer_for(text)
    engine = FakeEngine(script=_script_from_text(text))

    async def run():
        async with _client(engine, tokenizer) as client:
            r = await client.post("/v1/messages", json=_messages_body(max_tokens=5, stream=False))
            return r.json()

    body = _run(run())
    assert body["stop_reason"] == "max_tokens"
    assert body["usage"]["output_tokens"] == 5


# ── 3. SSE text-only event sequence ──────────────────────────────────


def test_sse_event_sequence_text_only():
    text = "hi"
    tokenizer = _tokenizer_for(text)
    engine = FakeEngine(script=_script_from_text(text))

    async def run():
        async with _client(engine, tokenizer) as client:
            async with client.stream(
                "POST", "/v1/messages", json=_messages_body(max_tokens=100, stream=True)
            ) as r:
                body = ""
                async for chunk in r.aiter_text():
                    body += chunk
                return body

    body = _run(run())
    events = [e for e in parse_sse(body) if e.event != "ping"]
    kinds = [e.event for e in events]

    assert kinds == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]

    # message_start carries model + input token count, empty output count.
    start = events[0].data
    assert start["type"] == "message_start"
    assert start["message"]["model"] == "test-model"
    assert start["message"]["usage"] == {"input_tokens": 3, "output_tokens": 0}

    # content block is text, index 0.
    block_start = events[1].data
    assert block_start["index"] == 0
    assert block_start["content_block"] == {"type": "text", "text": ""}

    # deltas concatenate to the full text.
    deltas = [e.data for e in events if e.event == "content_block_delta"]
    assert "".join(d["delta"]["text"] for d in deltas) == "hi"
    assert all(d["delta"]["type"] == "text_delta" for d in deltas)
    assert all(d["index"] == 0 for d in deltas)

    # stop_reason is end_turn, output token count is right.
    delta = events[-2].data
    assert delta["delta"]["stop_reason"] == "end_turn"
    assert delta["usage"]["output_tokens"] == 2
    assert delta["usage"]["text_tokens"] == 2
    assert delta["usage"]["thinking_tokens"] == 0


# ── 4. SSE thinking then text ────────────────────────────────────────


def test_sse_thinking_then_text_indices():
    tokenizer = _tokenizer_for("hi")
    script = [
        ScriptStep(token_id=THINKING_START_ID),
        *_script_from_text("hi"),
        ScriptStep(token_id=THINKING_END_ID),
        *_script_from_text("ok"),
    ]
    engine = FakeEngine(script=script)

    async def run():
        async with _client(engine, tokenizer) as client:
            async with client.stream(
                "POST", "/v1/messages",
                json={
                    "model": "test-model",
                    "max_tokens": 100,
                    "stream": True,
                    "messages": [{"role": "user", "content": "go"}],
                },
            ) as r:
                body = ""
                async for chunk in r.aiter_text():
                    body += chunk
                return body

    body = _run(run())
    events = [e for e in parse_sse(body) if e.event != "ping"]

    # Indices: 0 = thinking block, 1 = text block.
    block_starts = [e.data for e in events if e.event == "content_block_start"]
    assert len(block_starts) == 2
    assert block_starts[0]["index"] == 0
    assert block_starts[0]["content_block"]["type"] == "thinking"
    assert block_starts[1]["index"] == 1
    assert block_starts[1]["content_block"]["type"] == "text"

    # Thinking deltas go to index 0 with thinking_delta type.
    thinking_deltas = [
        e.data for e in events
        if e.event == "content_block_delta" and e.data["delta"]["type"] == "thinking_delta"
    ]
    assert [d["index"] for d in thinking_deltas] == [0, 0]
    assert "".join(d["delta"]["thinking"] for d in thinking_deltas) == "hi"

    # Text deltas go to index 1 with text_delta type.
    text_deltas = [
        e.data for e in events
        if e.event == "content_block_delta" and e.data["delta"]["type"] == "text_delta"
    ]
    assert [d["index"] for d in text_deltas] == [1, 1]
    assert "".join(d["delta"]["text"] for d in text_deltas) == "ok"

    # Two stops, one per block, in the right order.
    block_stops = [e.data for e in events if e.event == "content_block_stop"]
    assert [s["index"] for s in block_stops] == [0, 1]

    # message_delta reports split counts. The </think> end token is classified
    # as thinking-bucket even though it transitions the phase out of thinking,
    # so thinking_tokens counts it (4 = <think>, h, i, </think>).
    delta = [e.data for e in events if e.event == "message_delta"][0]
    assert delta["usage"]["thinking_tokens"] == 4
    assert delta["usage"]["text_tokens"] == 2
    assert delta["usage"]["output_tokens"] == 6


# ── 5. SSE thinking-only (stream ends mid-thinking) ──────────────────


def test_sse_thinking_only_still_closes_block():
    """If the stream ends while still in a thinking block, the decoder's flush
    should emit a synthetic ThinkingEnd so the block gets a proper stop event.
    """
    tokenizer = _tokenizer_for("hi")
    script = [
        ScriptStep(token_id=THINKING_START_ID),
        *_script_from_text("hi"),
    ]
    engine = FakeEngine(script=script)

    async def run():
        async with _client(engine, tokenizer) as client:
            async with client.stream(
                "POST", "/v1/messages",
                json={
                    "model": "test-model",
                    "max_tokens": 100,
                    "stream": True,
                    "messages": [{"role": "user", "content": "go"}],
                },
            ) as r:
                body = ""
                async for chunk in r.aiter_text():
                    body += chunk
                return body

    body = _run(run())
    events = [e for e in parse_sse(body) if e.event != "ping"]
    kinds = [e.event for e in events]

    # Exactly one thinking block, properly stopped, then message_delta/stop.
    assert kinds == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    block_start = events[1].data
    assert block_start["content_block"]["type"] == "thinking"


# ── 6. Ping cadence ──────────────────────────────────────────────────


def test_sse_ping_fires_during_idle_gap(monkeypatch):
    monkeypatch.setattr(
        "cantollm.api.anthropic_adapter.PING_INTERVAL_SECONDS", 0.05
    )
    tokenizer = _tokenizer_for("hi")
    # Sleep long enough between tokens that at least one ping lands.
    script = [
        ScriptStep(token_id=_char_ids("h")[0]),
        ScriptStep(sleep=0.2),
        ScriptStep(token_id=_char_ids("i")[0]),
    ]
    engine = FakeEngine(script=script)

    async def run():
        async with _client(engine, tokenizer) as client:
            async with client.stream(
                "POST", "/v1/messages", json=_messages_body(max_tokens=100, stream=True)
            ) as r:
                body = ""
                async for chunk in r.aiter_text():
                    body += chunk
                return body

    body = _run(run())
    events = parse_sse(body)
    pings = [e for e in events if e.event == "ping"]
    assert len(pings) >= 1, f"expected at least one ping, got events: {[e.event for e in events]}"


# ── 7. Mid-stream abort: adapter cleans up engine generator ──────────


def test_render_sse_aclose_propagates_to_engine():
    """Closing the adapter's async generator early must close the engine's
    submit generator too, so the engine can release its per-request state.
    The current adapter does this via producer-task cancellation inside its
    `finally` block.
    """
    tokenizer = _tokenizer_for("abcde")
    # Sleep between every token so the consumer can stop mid-stream.
    script = []
    for tid in _char_ids("abcde"):
        script.append(ScriptStep(token_id=tid))
        script.append(ScriptStep(sleep=0.05))
    engine = FakeEngine(script=script)

    from cantollm.engine.types import InferenceRequest, SamplingParams

    req = InferenceRequest(
        request_id="test-req",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(),
        max_tokens=100,
        stop_token_ids=set(),
    )

    async def run():
        gen = render_sse(engine.submit(req), tokenizer, "test-model", 3)
        # Pull the first chunk (message_start) then close.
        first = await gen.__anext__()
        assert "message_start" in first
        await gen.aclose()
        # Give the cancelled producer a tick to finalize.
        await asyncio.sleep(0.05)

    _run(run())

    assert engine.aborted is True
    assert engine.completed is False


# ── 8. Error propagation ─────────────────────────────────────────────


def test_sse_error_event_on_midstream_exception():
    """Mid-stream backend error surfaces as an Anthropic-style `error` SSE
    event, not a silent truncation. No message_delta/message_stop on the
    error path.
    """
    tokenizer = _tokenizer_for("hi")
    script = [
        ScriptStep(token_id=_char_ids("h")[0]),
        ScriptStep(raise_error=RuntimeError("boom")),
    ]
    engine = FakeEngine(script=script)

    async def run():
        async with _client(engine, tokenizer) as client:
            async with client.stream(
                "POST", "/v1/messages", json=_messages_body(max_tokens=100, stream=True)
            ) as r:
                body = ""
                async for chunk in r.aiter_text():
                    body += chunk
                return body

    body = _run(run())
    events = [e for e in parse_sse(body) if e.event != "ping"]
    kinds = [e.event for e in events]

    assert "error" in kinds
    error_evt = next(e for e in events if e.event == "error")
    assert error_evt.data["error"]["type"] == "api_error"
    assert "boom" in error_evt.data["error"]["message"]
    # No terminal message_delta/message_stop on the error path.
    assert "message_delta" not in kinds
    assert "message_stop" not in kinds


def test_non_streaming_error_returns_500():
    """Non-streaming error path surfaces as an HTTP 500, not a 200 with a
    malformed body.
    """
    tokenizer = _tokenizer_for("hi")
    script = [
        ScriptStep(token_id=_char_ids("h")[0]),
        ScriptStep(raise_error=RuntimeError("boom")),
    ]
    engine = FakeEngine(script=script)

    async def run():
        async with _client(engine, tokenizer) as client:
            r = await client.post("/v1/messages", json=_messages_body(max_tokens=100, stream=False))
            return r.status_code, r.json()

    status, body = _run(run())
    assert status == 500
    assert "boom" in body.get("detail", "")


# ── 9. Pydantic validation ───────────────────────────────────────────


def test_validation_rejects_bad_requests():
    tokenizer = _tokenizer_for("hi")
    engine = FakeEngine(script=_script_from_text("hi"))

    async def run():
        async with _client(engine, tokenizer) as client:
            missing_max = await client.post(
                "/v1/messages",
                json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
            )
            empty_messages = await client.post(
                "/v1/messages",
                json={"model": "test-model", "max_tokens": 10, "messages": []},
            )
            negative_max = await client.post(
                "/v1/messages",
                json={
                    "model": "test-model",
                    "max_tokens": -1,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            return missing_max.status_code, empty_messages.status_code, negative_max.status_code

    s1, s2, s3 = _run(run())
    assert (s1, s2, s3) == (422, 422, 422)


# Reference so ruff doesn't complain about imported-but-unused names.
_ = STOP_TOKEN_ID
