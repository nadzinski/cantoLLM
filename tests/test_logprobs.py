"""Logprobs (tail step 12): sampled-token log-probabilities end to end.

The value is computed once at each sample site (post-pipeline distribution —
honest since the step-1 greedy fix), rides TokenEvent.logprob as an
annotation on token events, and surfaces on the OpenAI dialect
(`"logprobs": true`). The Anthropic Messages API has no logprobs concept and
stays untouched.
"""

import asyncio
import math
import threading

import httpx
import torch

from cantollm.api import create_app
from cantollm.engine.types import SamplingParams, Sequence
from cantollm.runtime import build_runtime
from tests.fakes import (
    FakeEngine,
    FakeRegistry,
    FakeRuntime,
    FakeTokenizer,
    ScriptStep,
)
from tests.test_cb_end_to_end import PROMPTS, build_engines, make_request
from tests.test_openai_contract import _parse_openai_sse
from tests.tiny_model import tiny_qwen3_spec

GREEDY = SamplingParams.from_temperature_top_p(temperature=0.0, top_p=1.0)


async def collect_with_logprobs(engine, req):
    tokens, logprobs, finish = [], [], None
    async for evt in engine.submit(req):
        if evt.token_id is not None:
            tokens.append(evt.token_id)
            logprobs.append(evt.logprob)
        if evt.finish_reason is not None:
            finish = evt.finish_reason
    return tokens, logprobs, finish


class TestEngineLogprobs:
    def test_sequential_events_carry_sane_logprobs(self):
        async def main():
            oracle_engine, _ = build_engines()
            return await collect_with_logprobs(
                oracle_engine, make_request("r", PROMPTS[1], max_tokens=5)
            )

        tokens, logprobs, _ = asyncio.run(main())
        assert len(logprobs) == len(tokens) == 5
        assert all(lp is not None and lp <= 0.0 and math.isfinite(lp)
                   for lp in logprobs)

    def test_batched_logprobs_match_sequential_within_tolerance(self):
        """Same tokens (strict equivalence already pinned) ⇒ same
        probabilities up to reduction-order noise."""
        async def main():
            oracle_engine, cb = build_engines()
            await cb.start()
            try:
                req_a = make_request("a", PROMPTS[2], max_tokens=6)
                req_b = make_request("b", PROMPTS[2], max_tokens=6)
                seq_out = await collect_with_logprobs(oracle_engine, req_a)
                cb_out = await collect_with_logprobs(cb, req_b)
                return seq_out, cb_out
            finally:
                await cb.shutdown()

        (seq_tokens, seq_lps, _), (cb_tokens, cb_lps, _) = asyncio.run(main())
        assert cb_tokens == seq_tokens
        for i, (a, b) in enumerate(zip(seq_lps, cb_lps)):
            assert abs(a - b) < 1e-4, f"token {i}: {a} vs {b}"

    def test_speculative_backend_appends_in_lockstep(self):
        """Speculative path: one logprob per yielded token, from main's
        distribution."""
        runtime = build_runtime(
            tiny_qwen3_spec(), torch.device("cpu"),
            speculative=tiny_qwen3_spec(),
        )
        seq = Sequence(
            request_id="s", prompt_token_ids=[11, 12, 13],
            sampling_params=GREEDY, stop_token_ids=set(),
            max_tokens=6, cache=runtime.new_cache(),
            stop_event=threading.Event(),
        )
        tokens = list(runtime.backend.generate(seq))
        assert len(seq.logprobs) == len(tokens) > 0
        assert all(lp <= 0.0 and math.isfinite(lp) for lp in seq.logprobs)


class TestOpenAILogprobs:
    SCRIPT = [
        ScriptStep(token_id=2000, logprob=-0.25),
        ScriptStep(token_id=2001, logprob=-1.5),
        ScriptStep(token_id=2002, logprob=-0.03125),
    ]

    def _client(self):
        engine = FakeEngine(script=list(self.SCRIPT))
        tokenizer = FakeTokenizer(id_to_text={2000: "a", 2001: "b", 2002: "c"})
        registry = FakeRegistry(
            entries={"m": (engine, FakeRuntime(tokenizer=tokenizer))}
        )
        app = create_app(registry)
        transport = httpx.ASGITransport(app=app)
        return httpx.AsyncClient(transport=transport, base_url="http://t")

    def _body(self, logprobs: bool, stream: bool = False) -> dict:
        return {
            "model": "m", "max_tokens": 8, "stream": stream,
            "logprobs": logprobs,
            "messages": [{"role": "user", "content": "hi"}],
        }

    def test_non_streaming_logprobs(self):
        async def main():
            async with self._client() as client:
                return await client.post(
                    "/v1/chat/completions", json=self._body(logprobs=True)
                )

        resp = asyncio.run(main())
        assert resp.status_code == 200
        content = resp.json()["choices"][0]["logprobs"]["content"]
        assert [e["logprob"] for e in content] == [-0.25, -1.5, -0.03125]
        assert [e["token"] for e in content] == ["a", "b", "c"]
        assert content[0]["bytes"] == [ord("a")]

    def test_logprobs_off_yields_null(self):
        async def main():
            async with self._client() as client:
                return await client.post(
                    "/v1/chat/completions", json=self._body(logprobs=False)
                )

        resp = asyncio.run(main())
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["logprobs"] is None

    def test_streaming_logprobs_ride_content_chunks(self):
        async def main():
            async with self._client() as client:
                return await client.post(
                    "/v1/chat/completions",
                    json=self._body(logprobs=True, stream=True),
                )

        resp = asyncio.run(main())
        chunks, saw_done = _parse_openai_sse(resp.text)
        assert saw_done
        content_chunks = [
            c for c in chunks
            if c["choices"] and c["choices"][0]["delta"].get("content")
        ]
        lps = [
            c["choices"][0]["logprobs"]["content"][0]["logprob"]
            for c in content_chunks
        ]
        assert lps == [-0.25, -1.5, -0.03125]

    def test_top_logprobs_still_rejected(self):
        async def main():
            async with self._client() as client:
                body = self._body(logprobs=True)
                body["top_logprobs"] = 5
                return await client.post("/v1/chat/completions", json=body)

        assert asyncio.run(main()).status_code == 422
