"""End-to-end: the real ContinuousBatchingEngine on tiny-Qwen3 (step 9).

The strict promise, cashed in: N concurrent requests through the composed
engine (shell + hand-written scheduler + padded attention + KV pool) produce
token-for-token identical streams and finish reasons to per-request runs
through SequentialEngine + StandardBackend on the einsum path — greedy,
float32, CPU. Everything here goes through public engine/API surfaces only.
"""

import asyncio

import httpx
import pytest
import torch

from cantollm.api import create_app
from cantollm.engine import ContinuousBatchingEngine, SequentialEngine
from cantollm.engine.batching import BatchingConfig
from cantollm.registry import EngineRegistry
from cantollm.runtime import build_runtime
from tests.cb_helpers import PROMPTS, build_engines, collect, make_request
from tests.fakes import parse_sse
from tests.tiny_model import tiny_qwen3_spec


class TestEquivalence:
    def test_concurrent_streams_match_sequential_oracle(self):
        async def main():
            oracle_engine, cb = build_engines()
            await cb.start()
            try:
                requests = [
                    make_request("r0", PROMPTS[0], max_tokens=6),
                    make_request("r1", PROMPTS[1], max_tokens=4),
                    make_request("r2", PROMPTS[2], max_tokens=8),
                ]
                batched = await asyncio.gather(
                    *(collect(cb, r) for r in requests)
                )
                oracle = [await collect(oracle_engine, r) for r in requests]
                return batched, oracle
            finally:
                await cb.shutdown()

        batched, oracle = asyncio.run(main())
        for i, (got, want) in enumerate(zip(batched, oracle)):
            assert got == want, f"request {i}: {got} != {want}"
            assert got[1] == "max_tokens"

    def test_stop_token_equivalence(self):
        async def main():
            oracle_engine, cb = build_engines()
            await cb.start()
            try:
                # Discover a token the model will emit, make it a stop token.
                free_run, _ = await collect(
                    oracle_engine, make_request("probe", PROMPTS[2], max_tokens=6)
                )
                stop = free_run[2]
                req_o = make_request("o", PROMPTS[2], max_tokens=6,
                                     stop_token_ids={stop})
                req_b = make_request("b", PROMPTS[2], max_tokens=6,
                                     stop_token_ids={stop})
                oracle = await collect(oracle_engine, req_o)
                batched = await collect(cb, req_b)
                return oracle, batched, stop
            finally:
                await cb.shutdown()

        oracle, batched, stop = asyncio.run(main())
        assert batched == oracle
        assert batched[1] == "end_turn"
        assert stop not in batched[0]

    def test_staggered_arrival_matches_oracle(self):
        async def main():
            oracle_engine, cb = build_engines()
            await cb.start()
            try:
                early = make_request("early", PROMPTS[0], max_tokens=8)
                late = make_request("late", PROMPTS[1], max_tokens=5)

                early_task = asyncio.create_task(collect(cb, early))
                await asyncio.sleep(0.05)  # early is mid-generation
                late_result = await collect(cb, late)
                early_result = await early_task

                oracle_early = await collect(oracle_engine, early)
                oracle_late = await collect(oracle_engine, late)
                return early_result, late_result, oracle_early, oracle_late
            finally:
                await cb.shutdown()

        early_result, late_result, oracle_early, oracle_late = asyncio.run(main())
        assert early_result == oracle_early
        assert late_result == oracle_late


class TestAbortUnderLoad:
    def test_abort_frees_capacity_for_queued_request(self):
        async def main():
            oracle_engine, cb = build_engines(max_batch=1)
            await cb.start()
            try:
                hog = make_request("hog", PROMPTS[0], max_tokens=50)
                waiting = make_request("waiting", PROMPTS[1], max_tokens=4)

                hog_events = []

                async def consume_hog():
                    async for evt in cb.submit(hog):
                        hog_events.append(evt)
                        if evt.token_id is not None and len(
                            [e for e in hog_events if e.token_id is not None]
                        ) >= 2:
                            cb.abort("hog")

                waiting_task = asyncio.create_task(collect(cb, waiting))
                await consume_hog()
                waiting_result = await waiting_task
                oracle_waiting = await collect(oracle_engine, waiting)
                return hog_events, waiting_result, oracle_waiting
            finally:
                await cb.shutdown()

        hog_events, waiting_result, oracle_waiting = asyncio.run(main())
        assert hog_events[-1].finish_reason == "abort"
        assert waiting_result == oracle_waiting


class TestThroughTheAPI:
    def make_app(self):
        cpu = torch.device("cpu")
        runtime = build_runtime(tiny_qwen3_spec(), cpu, attention="padded")
        config = BatchingConfig(max_batch=2, max_seq_len=32, max_tokens_per_step=8)
        engine = ContinuousBatchingEngine.from_runtime(runtime, config)
        registry = EngineRegistry()
        registry.register(
            "tiny-cb", engine, runtime, max_request_tokens=config.max_seq_len
        )
        return create_app(registry)

    def _body(self, max_tokens: int, stream: bool = False) -> dict:
        return {
            "model": "tiny-cb",
            "max_tokens": max_tokens,
            "stream": stream,
            "messages": [{"role": "user", "content": "hi"}],
        }

    def test_concurrent_requests_and_admission_over_real_entry(self):
        async def main():
            app = self.make_app()
            transport = httpx.ASGITransport(app=app)
            async with app.router.lifespan_context(app):  # start/stop engine
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    responses = await asyncio.gather(
                        *(client.post("/v1/messages", json=self._body(5))
                          for _ in range(3))
                    )
                    over_cap = await client.post(
                        "/v1/messages", json=self._body(40)  # 3 + 40 > 32
                    )
                    streamed = await client.post(
                        "/v1/messages", json=self._body(4, stream=True)
                    )
            return responses, over_cap, streamed

        responses, over_cap, streamed = asyncio.run(main())
        for resp in responses:
            assert resp.status_code == 200
            body = resp.json()
            assert body["stop_reason"] == "max_tokens"
            assert body["usage"]["output_tokens"] == 5
        assert over_cap.status_code == 400
        assert "exceeds this model's limit" in over_cap.json()["error"]["message"]
        events = parse_sse(streamed.text)
        assert events[0].event == "message_start"
        assert any(e.event == "message_stop" for e in events)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
class TestOnMPS:
    """Functional smoke on the real device — completion, not equality
    (token-for-token holds only in the fp32/CPU/greedy cell)."""

    def test_concurrent_requests_complete(self):
        async def main():
            runtime = build_runtime(
                tiny_qwen3_spec(), torch.device("mps"), attention="padded"
            )
            config = BatchingConfig(max_batch=2, max_seq_len=64,
                                    max_tokens_per_step=8)
            cb = ContinuousBatchingEngine.from_runtime(runtime, config)
            await cb.start()
            try:
                results = await asyncio.gather(
                    collect(cb, make_request("m0", PROMPTS[0], max_tokens=4)),
                    collect(cb, make_request("m1", PROMPTS[1], max_tokens=4)),
                )
            finally:
                await cb.shutdown()
            return results

        results = asyncio.run(main())
        for tokens, finish in results:
            assert len(tokens) == 4 and finish == "max_tokens"
