"""Admission control (step 7): prompt + max_tokens > cap -> 400 at the door.

The cap rides on the registry entry (None = no cap, the sequential
default). FakeTokenizer always encodes a 3-token prompt, so the boundaries
below are exact: cap 10 admits max_tokens 7, rejects 8.
"""

import asyncio

import httpx

from cantollm.api import create_app
from tests.fakes import (
    FakeEngine,
    FakeRegistry,
    FakeRuntime,
    FakeTokenizer,
    ScriptStep,
)

PROMPT_LEN = 3  # FakeTokenizer default
CAP = 10


def _client(cap: int | None) -> tuple[httpx.AsyncClient, FakeEngine]:
    engine = FakeEngine(script=[ScriptStep(token_id=2000)])
    runtime = FakeRuntime(tokenizer=FakeTokenizer())
    caps = {"test-model": cap} if cap is not None else None
    registry = FakeRegistry(
        entries={"test-model": (engine, runtime)}, max_request_tokens=caps
    )
    app = create_app(registry)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test"), engine


def _post(client, path, body):
    return client.post(path, json=body)


def _anthropic_body(max_tokens: int) -> dict:
    return {
        "model": "test-model",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": "hi"}],
    }


def _openai_body(max_tokens: int) -> dict:
    return {
        "model": "test-model",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": "hi"}],
    }


class TestAnthropicAdmission:
    def test_at_cap_is_admitted(self):
        async def main():
            client, _ = _client(cap=CAP)
            async with client:
                return await _post(client, "/v1/messages",
                                   _anthropic_body(CAP - PROMPT_LEN))

        assert asyncio.run(main()).status_code == 200

    def test_over_cap_is_rejected_before_the_engine(self):
        async def main():
            client, engine = _client(cap=CAP)
            async with client:
                resp = await _post(client, "/v1/messages",
                                   _anthropic_body(CAP - PROMPT_LEN + 1))
            return resp, engine

        resp, engine = asyncio.run(main())
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        for expected in ("3 tokens", "8", "11", "10"):
            assert expected in detail, f"{expected!r} missing from: {detail}"
        # The engine was never touched.
        assert not engine.completed and not engine.aborted

    def test_capless_entry_admits_anything(self):
        async def main():
            client, _ = _client(cap=None)
            async with client:
                return await _post(client, "/v1/messages",
                                   _anthropic_body(100_000))

        assert asyncio.run(main()).status_code == 200


class TestOpenAIAdmission:
    def test_at_cap_is_admitted(self):
        async def main():
            client, _ = _client(cap=CAP)
            async with client:
                return await _post(client, "/v1/chat/completions",
                                   _openai_body(CAP - PROMPT_LEN))

        assert asyncio.run(main()).status_code == 200

    def test_over_cap_is_rejected_before_the_engine(self):
        async def main():
            client, engine = _client(cap=CAP)
            async with client:
                resp = await _post(client, "/v1/chat/completions",
                                   _openai_body(CAP - PROMPT_LEN + 1))
            return resp, engine

        resp, engine = asyncio.run(main())
        assert resp.status_code == 400
        assert "exceeds this model's limit" in resp.json()["detail"]
        assert not engine.completed and not engine.aborted
