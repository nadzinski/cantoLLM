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
from cantollm.engine.batching import BatchingConfig, SlotAllocator
from cantollm.engine.batching.scheduler import ContinuousBatchingScheduler
from cantollm.engine.types import InferenceRequest, SamplingParams, Sequence
from cantollm.kv_cache import KVCache
from cantollm.standard import StandardBackend
from cantollm.runtime import build_runtime
from tests.fakes import (
    FakeEngine,
    FakeRegistry,
    FakeRuntime,
    FakeTokenizer,
    ScriptStep,
    parse_openai_sse,
)
from tests.cb_helpers import PROMPTS, build_engines, make_request
from tests.tiny_model import tiny_qwen3_spec
from tests.toy_stepper import make_toy_pool

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


class _FixedLogitsForward:
    """BatchedForwardFn returning the same logits row for every batch row —
    the model is out of the picture, so expected logprobs are closed-form."""

    def __init__(self, logits: torch.Tensor):
        self.logits = logits

    def __call__(self, input_ids, meta, pool):
        return self.logits.repeat(input_ids.shape[0], 1)


class _FixedLogitsModel:
    """Model stub for StandardBackend: (B, S, vocab) of one fixed logits row."""

    def __init__(self, logits: torch.Tensor):
        self.logits = logits

    def __call__(self, tokens, start_pos, kv_cache=None):
        b, sq = tokens.shape
        return self.logits.repeat(b, sq, 1)


# softmax of these logits is exactly [0.1, 0.2, 0.3, 0.4]: expected greedy
# token is 3 and its logprob is ln(0.4) — a closed-form constant, computed
# with no code shared with the samplers under test.
_FIXED_LOGITS = torch.log(torch.tensor([0.1, 0.2, 0.3, 0.4]))
_EXPECTED_GREEDY_TOKEN = 3
_EXPECTED_GREEDY_LOGPROB = math.log(0.4)


class TestLogprobGroundTruth:
    """Every other logprob test asserts 'finite and <= 0' or CB==sequential —
    both paths share the sampler code, so a systematic error (wrong base,
    pre-processor distribution, wrong index) would pass all of them. These
    pin the emitted values to hand-computed constants."""

    def test_cb_scheduler_emits_log_softmax_of_logits(self):
        config = BatchingConfig(max_batch=1, max_seq_len=8, max_tokens_per_step=1)
        scheduler = ContinuousBatchingScheduler(
            forward_fn=_FixedLogitsForward(_FIXED_LOGITS),
            pool=make_toy_pool(config),
            allocator=SlotAllocator(1),
            config=config,
        )
        scheduler.add_request(InferenceRequest(
            request_id="r", prompt_token_ids=[1],
            sampling_params=GREEDY, max_tokens=2, stop_token_ids=set(),
        ))
        events = []
        while not scheduler.is_idle():
            events.extend(scheduler.step())

        token_events = [e for e in events if e.token_id is not None]
        assert [e.token_id for e in token_events] == [_EXPECTED_GREEDY_TOKEN] * 2
        for e in token_events:
            assert abs(e.logprob - _EXPECTED_GREEDY_LOGPROB) < 1e-6, (
                f"{e.logprob} != ln(0.4) = {_EXPECTED_GREEDY_LOGPROB}"
            )

    def test_standard_backend_appends_log_softmax_of_logits(self):
        backend = StandardBackend(
            model=_FixedLogitsModel(_FIXED_LOGITS), device=torch.device("cpu")
        )
        seq = Sequence(
            request_id="r", prompt_token_ids=[1, 2],
            sampling_params=GREEDY, stop_token_ids=set(),
            max_tokens=2, cache=KVCache(1), stop_event=threading.Event(),
        )
        tokens = list(backend.generate(seq))
        assert tokens == [_EXPECTED_GREEDY_TOKEN] * 2
        for lp in seq.logprobs:
            assert abs(lp - _EXPECTED_GREEDY_LOGPROB) < 1e-6

    def test_logprob_reflects_post_processor_distribution(self):
        """With temperature, the reported logprob must come from the
        post-pipeline distribution log_softmax(logits / T) — recomputed here
        independently with torch primitives, including replaying the
        multinomial draw under the same seed to know which token to expect."""
        temperature = 0.5
        sampling = SamplingParams.from_temperature_top_p(temperature, 1.0)
        backend = StandardBackend(
            model=_FixedLogitsModel(_FIXED_LOGITS), device=torch.device("cpu")
        )
        seq = Sequence(
            request_id="r", prompt_token_ids=[1, 2],
            sampling_params=sampling, stop_token_ids=set(),
            max_tokens=1, cache=KVCache(1), stop_event=threading.Event(),
        )
        torch.manual_seed(77)
        tokens = list(backend.generate(seq))

        scaled = _FIXED_LOGITS / temperature
        torch.manual_seed(77)
        expected_token = int(torch.multinomial(
            torch.softmax(scaled, dim=-1).unsqueeze(0), num_samples=1
        ).item())
        expected_lp = float(torch.log_softmax(scaled, dim=-1)[expected_token])

        assert tokens == [expected_token]
        assert abs(seq.logprobs[0] - expected_lp) < 1e-6


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
        chunks, saw_done = parse_openai_sse(resp.text)
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

        resp = asyncio.run(main())
        assert resp.status_code == 400  # OpenAI invalid_request_error, not 422
        assert resp.json()["error"]["type"] == "invalid_request_error"

    def _client_with(self, script, id_to_text):
        engine = FakeEngine(script=script)
        tokenizer = FakeTokenizer(id_to_text=id_to_text)
        registry = FakeRegistry(
            entries={"m": (engine, FakeRuntime(tokenizer=tokenizer))}
        )
        app = create_app(registry)
        transport = httpx.ASGITransport(app=app)
        return httpx.AsyncClient(transport=transport, base_url="http://t")

    def test_non_streaming_logprobs_exclude_stop_sequence_tokens(self):
        """logprobs.content must align 1:1 with emitted content — a stop
        sequence's tokens are excluded from content, so their per-token
        entries must be dropped too."""
        script = [
            ScriptStep(token_id=2000, logprob=-0.25),
            ScriptStep(token_id=2001, logprob=-1.5),
            ScriptStep(token_id=2002, logprob=-0.03125),
        ]

        async def main():
            async with self._client_with(
                script, {2000: "a", 2001: "b", 2002: "c"}
            ) as client:
                # decoded "abc"; stop "bc" matches -> emitted content is "a".
                return await client.post("/v1/chat/completions", json={
                    "model": "m", "max_tokens": 8, "logprobs": True,
                    "stop": ["bc"],
                    "messages": [{"role": "user", "content": "hi"}],
                })

        choice = asyncio.run(main()).json()["choices"][0]
        assert choice["message"]["content"] == "a"
        assert [e["token"] for e in choice["logprobs"]["content"]] == ["a"]
        assert [e["logprob"] for e in choice["logprobs"]["content"]] == [-0.25]

    def test_streaming_logprobs_not_duplicated_by_flush(self):
        """A logprob entry must ride exactly one chunk. Token 2001 decodes to
        'bZ'; with stop 'Zz' the trailing 'Z' is held mid-token and released
        by the end-of-stream flush as a separate chunk — whose logprob must
        NOT re-attach the already-emitted entry."""
        script = [
            ScriptStep(token_id=2000, logprob=-0.25),
            ScriptStep(token_id=2001, logprob=-1.5),
        ]

        async def main():
            async with self._client_with(
                script, {2000: "a", 2001: "bZ"}
            ) as client:
                return await client.post("/v1/chat/completions", json={
                    "model": "m", "max_tokens": 8, "stream": True,
                    "logprobs": True, "stop": ["Zz"],
                    "messages": [{"role": "user", "content": "hi"}],
                })

        chunks, _ = parse_openai_sse(asyncio.run(main()).text)
        all_lps = []
        for c in chunks:
            choices = c.get("choices") or []
            if choices and choices[0].get("logprobs"):
                all_lps.extend(
                    e["logprob"] for e in choices[0]["logprobs"]["content"]
                )
        assert all_lps == [-0.25, -1.5]  # each once, not [-0.25, -1.5, -1.5]
