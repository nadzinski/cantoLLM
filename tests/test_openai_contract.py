"""HTTP/SSE contract tests for the OpenAI Chat Completions API.

Mirrors test_api_contract.py's shape — same FakeEngine / FakeTokenizer /
FakeRegistry doubles, same httpx ASGITransport pattern — but exercises the
OpenAI-specific wire format (data-only SSE, [DONE] terminator,
reasoning_content for thinking tokens, DeepSeek-style usage details).
"""

from __future__ import annotations

import asyncio
import json
import re

import httpx

from cantollm.api import create_app
from tests.fakes import (
    THINKING_END_ID,
    THINKING_START_ID,
    FakeEngine,
    FakeRegistry,
    FakeRuntime,
    FakeTokenizer,
    ScriptStep,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _char_ids(text: str) -> list[int]:
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


def _chat_body(*, stream: bool, max_tokens: int | None = None,
               max_completion_tokens: int | None = None,
               messages: list[dict] | None = None,
               **extra) -> dict:
    body = {
        "model": "test-model",
        "messages": messages or [{"role": "user", "content": "hi"}],
        "stream": stream,
        **extra,
    }
    if max_completion_tokens is not None:
        body["max_completion_tokens"] = max_completion_tokens
    elif max_tokens is not None:
        body["max_tokens"] = max_tokens
    return body


def _run(coro):
    return asyncio.run(coro)


_DATA_LINE = re.compile(r"^data:\s*(.*)$", re.MULTILINE)


def _parse_openai_sse(body: str) -> tuple[list[dict], bool]:
    """Parse an OpenAI-style SSE stream into (chunks, saw_done).

    OpenAI framing: every event is a single `data:` line, followed by a
    blank line. The stream terminates with the literal `data: [DONE]`
    sentinel.
    """
    chunks: list[dict] = []
    saw_done = False
    for block in body.split("\n\n"):
        block = block.strip("\n")
        if not block:
            continue
        m = _DATA_LINE.search(block)
        if not m:
            continue
        payload = m.group(1).strip()
        if payload == "[DONE]":
            saw_done = True
            continue
        chunks.append(json.loads(payload))
    return chunks, saw_done


# ── 1. Non-streaming happy path ──────────────────────────────────────


def test_non_streaming_happy_path():
    text = "hello"
    tokenizer = _tokenizer_for(text)
    engine = FakeEngine(script=_script_from_text(text))

    async def run():
        async with _client(engine, tokenizer) as client:
            r = await client.post(
                "/v1/chat/completions",
                json=_chat_body(stream=False, max_tokens=100),
            )
            assert r.status_code == 200
            return r.json()

    body = _run(run())

    assert body["object"] == "chat.completion"
    assert body["model"] == "test-model"
    assert body["id"].startswith("chatcmpl-")
    assert isinstance(body["created"], int) and body["created"] > 0

    assert len(body["choices"]) == 1
    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "hello"
    assert choice["message"].get("reasoning_content") is None

    usage = body["usage"]
    assert usage["prompt_tokens"] == 3
    assert usage["completion_tokens"] == 5
    assert usage["total_tokens"] == 8
    # No thinking tokens → no completion_tokens_details emitted.
    assert usage.get("completion_tokens_details") is None


# ── 2. Non-streaming with thinking tokens ────────────────────────────


def test_non_streaming_separates_reasoning_from_content():
    # Script: think "ab", exit thinking, then visible "cd".
    script = [
        ScriptStep(token_id=THINKING_START_ID),
        *_script_from_text("ab"),
        ScriptStep(token_id=THINKING_END_ID),
        *_script_from_text("cd"),
    ]
    tokenizer = _tokenizer_for("abcd")
    engine = FakeEngine(script=script)

    async def run():
        async with _client(engine, tokenizer) as client:
            r = await client.post(
                "/v1/chat/completions",
                json=_chat_body(stream=False, max_tokens=100),
            )
            return r.json()

    body = _run(run())
    choice = body["choices"][0]
    assert choice["message"]["content"] == "cd"
    assert choice["message"]["reasoning_content"] == "ab"
    assert choice["finish_reason"] == "stop"

    usage = body["usage"]
    assert usage["completion_tokens"] == 2            # visible text only
    assert usage["completion_tokens_details"]["reasoning_tokens"] == 4
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


# ── 3. Streaming chunk sequence ──────────────────────────────────────


def test_streaming_chunk_sequence():
    text = "hi"
    tokenizer = _tokenizer_for(text)
    engine = FakeEngine(script=_script_from_text(text))

    async def run():
        async with _client(engine, tokenizer) as client:
            async with client.stream(
                "POST", "/v1/chat/completions",
                json=_chat_body(stream=True, max_tokens=100),
            ) as r:
                body = ""
                async for chunk in r.aiter_text():
                    body += chunk
                return body

    body = _run(run())
    chunks, saw_done = _parse_openai_sse(body)
    assert saw_done

    # Opening chunk carries role only, no content/reasoning.
    opening = chunks[0]
    assert opening["object"] == "chat.completion.chunk"
    assert opening["choices"][0]["delta"] == {"role": "assistant"}
    assert opening["choices"][0]["finish_reason"] is None

    # Middle chunks each carry one content delta.
    middle = chunks[1:-1]
    text_deltas = [c["choices"][0]["delta"]["content"] for c in middle]
    assert "".join(text_deltas) == "hi"
    for c in middle:
        assert c["choices"][0]["finish_reason"] is None

    # Closing chunk has empty delta + finish_reason.
    closing = chunks[-1]
    assert closing["choices"][0]["delta"] == {}
    assert closing["choices"][0]["finish_reason"] == "stop"


def test_streaming_thinking_routed_to_reasoning_content():
    script = [
        ScriptStep(token_id=THINKING_START_ID),
        *_script_from_text("xy"),
        ScriptStep(token_id=THINKING_END_ID),
        *_script_from_text("z"),
    ]
    tokenizer = _tokenizer_for("xyz")
    engine = FakeEngine(script=script)

    async def run():
        async with _client(engine, tokenizer) as client:
            async with client.stream(
                "POST", "/v1/chat/completions",
                json=_chat_body(stream=True, max_tokens=100),
            ) as r:
                return "".join([c async for c in r.aiter_text()])

    body = _run(run())
    chunks, saw_done = _parse_openai_sse(body)
    assert saw_done

    reasoning = [c["choices"][0]["delta"].get("reasoning_content")
                 for c in chunks if c["choices"]]
    content = [c["choices"][0]["delta"].get("content")
               for c in chunks if c["choices"]]

    assert "".join(r for r in reasoning if r) == "xy"
    assert "".join(c for c in content if c) == "z"


# ── 4. Usage chunk is opt-in via stream_options.include_usage ─────────


def test_streaming_usage_chunk_only_when_requested():
    text = "hi"
    tokenizer = _tokenizer_for(text)
    engine = FakeEngine(script=_script_from_text(text))

    async def collect(body: dict) -> list[dict]:
        async with _client(engine, tokenizer) as client:
            async with client.stream("POST", "/v1/chat/completions", json=body) as r:
                text = "".join([c async for c in r.aiter_text()])
        chunks, saw_done = _parse_openai_sse(text)
        assert saw_done
        return chunks

    chunks_without = _run(collect(_chat_body(stream=True, max_tokens=100)))
    assert all(c.get("usage") is None for c in chunks_without)

    engine = FakeEngine(script=_script_from_text(text))
    chunks_with = _run(collect(_chat_body(
        stream=True, max_tokens=100,
        stream_options={"include_usage": True},
    )))
    usage_chunks = [c for c in chunks_with if c.get("usage") is not None]
    assert len(usage_chunks) == 1
    usage_chunk = usage_chunks[0]
    assert usage_chunk["choices"] == []
    assert usage_chunk["usage"]["prompt_tokens"] == 3
    assert usage_chunk["usage"]["completion_tokens"] == 2


# ── 5. Finish-reason mapping ─────────────────────────────────────────


def test_non_streaming_max_tokens_maps_to_length():
    text = "hello"  # 5 tokens
    tokenizer = _tokenizer_for(text)
    engine = FakeEngine(script=_script_from_text(text))

    async def run():
        async with _client(engine, tokenizer) as client:
            r = await client.post(
                "/v1/chat/completions",
                json=_chat_body(stream=False, max_tokens=5),
            )
            return r.json()

    body = _run(run())
    assert body["choices"][0]["finish_reason"] == "length"


# ── 6. Mid-stream error: error chunk then [DONE] ─────────────────────


def test_streaming_error_produces_error_chunk_then_done():
    script = [
        *_script_from_text("ab"),
        ScriptStep(raise_error=RuntimeError("boom")),
    ]
    tokenizer = _tokenizer_for("ab")
    engine = FakeEngine(script=script)

    async def run():
        async with _client(engine, tokenizer) as client:
            async with client.stream(
                "POST", "/v1/chat/completions",
                json=_chat_body(stream=True, max_tokens=100),
            ) as r:
                return "".join([c async for c in r.aiter_text()])

    body = _run(run())
    chunks, saw_done = _parse_openai_sse(body)
    assert saw_done

    error_chunks = [c for c in chunks if "error" in c]
    assert len(error_chunks) == 1
    assert "boom" in error_chunks[0]["error"]["message"]

    # No closing finish_reason chunk on the error path — the error envelope
    # is the terminal event (before [DONE]).
    content_chunks = [c for c in chunks if c.get("object") == "chat.completion.chunk"]
    finishes = [c["choices"][0]["finish_reason"]
                for c in content_chunks if c.get("choices")]
    assert all(f is None for f in finishes)


def test_non_streaming_error_returns_500():
    script = [ScriptStep(raise_error=RuntimeError("boom"))]
    tokenizer = _tokenizer_for("a")
    engine = FakeEngine(script=script)

    async def run():
        async with _client(engine, tokenizer) as client:
            return await client.post(
                "/v1/chat/completions",
                json=_chat_body(stream=False, max_tokens=1),
            )

    r = _run(run())
    assert r.status_code == 500


# ── 7. Unsupported fields get 400 ────────────────────────────────────


def test_unknown_top_level_field_rejected():
    tokenizer = _tokenizer_for("a")
    engine = FakeEngine(script=_script_from_text("a"))

    async def run():
        async with _client(engine, tokenizer) as client:
            return await client.post(
                "/v1/chat/completions",
                json={**_chat_body(stream=False, max_tokens=1),
                      "tools": [{"type": "function", "function": {"name": "x"}}]},
            )

    r = _run(run())
    assert r.status_code == 422  # Pydantic extra="forbid" rejects at validation


def test_unknown_content_part_type_rejected():
    tokenizer = _tokenizer_for("a")
    engine = FakeEngine(script=_script_from_text("a"))

    async def run():
        async with _client(engine, tokenizer) as client:
            return await client.post(
                "/v1/chat/completions",
                json=_chat_body(
                    stream=False, max_tokens=1,
                    messages=[{
                        "role": "user",
                        "content": [{"type": "image_url",
                                     "image_url": {"url": "http://x"}}],
                    }],
                ),
            )

    r = _run(run())
    assert r.status_code == 422


def test_non_leading_system_message_rejected():
    tokenizer = _tokenizer_for("a")
    engine = FakeEngine(script=_script_from_text("a"))

    async def run():
        async with _client(engine, tokenizer) as client:
            return await client.post(
                "/v1/chat/completions",
                json=_chat_body(
                    stream=False, max_tokens=1,
                    messages=[
                        {"role": "user", "content": "hi"},
                        {"role": "system", "content": "you are helpful"},
                    ],
                ),
            )

    r = _run(run())
    assert r.status_code == 400


# ── 8. developer role folded into system ─────────────────────────────


def test_developer_role_folded_into_system():
    tokenizer = _tokenizer_for("a")
    engine = FakeEngine(script=_script_from_text("a"))

    async def run():
        async with _client(engine, tokenizer) as client:
            r = await client.post(
                "/v1/chat/completions",
                json=_chat_body(
                    stream=False, max_tokens=1,
                    messages=[
                        {"role": "developer", "content": "be concise"},
                        {"role": "system", "content": "be helpful"},
                        {"role": "user", "content": "hi"},
                    ],
                ),
            )
            assert r.status_code == 200

    _run(run())

    # The router should have concatenated both system/developer messages and
    # passed only the user turn through as a message.
    assert tokenizer.last_system == "be concise\n\nbe helpful"
    assert tokenizer.last_messages == [{"role": "user", "content": "hi"}]


# ── 9. max_completion_tokens preferred over max_tokens ───────────────


def test_max_completion_tokens_preferred_over_max_tokens():
    # FakeEngine emits its whole script regardless of req.max_tokens, but
    # reports finish_reason="max_tokens" iff tokens_emitted >= req.max_tokens.
    # Script is 10 tokens; with max_tokens=100 the engine would say
    # "end_turn" ("stop"), but with max_completion_tokens=3 it should say
    # "max_tokens" ("length") — which only happens if the router passed 3.
    text = "abcdefghij"
    tokenizer = _tokenizer_for(text)
    engine = FakeEngine(script=_script_from_text(text))

    async def run():
        async with _client(engine, tokenizer) as client:
            r = await client.post(
                "/v1/chat/completions",
                json=_chat_body(
                    stream=False,
                    max_tokens=100,
                    max_completion_tokens=3,
                ),
            )
            return r.json()

    body = _run(run())
    assert body["choices"][0]["finish_reason"] == "length"
