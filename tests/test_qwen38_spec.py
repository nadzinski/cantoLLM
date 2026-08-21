"""Spec/registry/CLI wiring tests for the qwen38 family, plus the
engine-level proof: the tiny qwen38 spec served through the REAL
runtime -> ContinuousBatchingEngine stack, token-for-token against the
SequentialEngine (the cb_helpers ritual on the hybrid model).
"""

import asyncio

import pytest
import torch

from cantollm.engine import ContinuousBatchingEngine, SequentialEngine
from cantollm.engine.batching import BatchingConfig
from cantollm.models.qwen38.model import Qwen38
from cantollm.models.qwen38.pool import HybridCache, HybridStatePool
from cantollm.models.qwen38.spec import qwen38_spec
from cantollm.runtime import build_runtime
from cantollm.spec import known_models, resolve_spec
from tests.cb_helpers import PROMPTS, collect, make_request
from tests.tiny_qwen38 import tiny_qwen38_spec


class TestResolveSpec:
    def test_qwen38_prefix_dispatches(self):
        spec = resolve_spec("qwen38-27B")
        assert spec.name == "qwen38-27B"
        assert spec.model_cls is Qwen38
        assert spec.cache_factory is not None
        assert spec.kv_pool_factory is not None
        assert spec.arch["num_transformers"] == 64
        assert spec.dtype == torch.bfloat16

    def test_qwen3_sizes_still_resolve(self):
        spec = resolve_spec("0.6B")
        assert spec.name == "qwen3-0.6B"
        assert spec.cache_factory is None
        assert spec.kv_pool_factory is None

    def test_unknown_qwen38_size_rejected(self):
        with pytest.raises(ValueError, match="Unknown Qwen3.8 size"):
            resolve_spec("qwen38-9B")

    def test_known_models_lists_both_families(self):
        models = known_models()
        assert "0.6B" in models and "32B" in models
        assert "qwen38-27B" in models

    def test_cli_choices_include_qwen38(self):
        from cantollm.main import _model_choices

        assert "qwen38-27B" in _model_choices()


class TestSpecHooks:
    def test_cache_factory_builds_hybrid_cache(self):
        cache = tiny_qwen38_spec().cache_factory()
        assert isinstance(cache, HybridCache)
        assert len(cache) == 8

    def test_runtime_new_cache_uses_hook(self):
        runtime = build_runtime(tiny_qwen38_spec(), torch.device("cpu"))
        assert isinstance(runtime.new_cache(), HybridCache)

    def test_runtime_new_kv_pool_uses_hook(self):
        runtime = build_runtime(
            tiny_qwen38_spec(), torch.device("cpu"), attention="padded"
        )
        config = BatchingConfig(max_batch=2, max_seq_len=32, max_tokens_per_step=8)
        pool = runtime.new_kv_pool(config)
        assert isinstance(pool, HybridStatePool)
        assert pool.max_batch == 2

    def test_27b_pool_factory_rope_guard(self):
        spec = qwen38_spec("27B")
        config = BatchingConfig(
            max_batch=1, max_seq_len=65536, max_tokens_per_step=128
        )
        # 65536 + 128 - 2 >= 65536: must refuse BEFORE allocating anything.
        with pytest.raises(ValueError, match="RoPE table length"):
            spec.kv_pool_factory(config, torch.device("cpu"))


class TestEngineLevelEndToEnd:
    """The cb_helpers ritual on the hybrid model: same weights, einsum
    sequential vs padded CB through the real engines, greedy, token for
    token. Chunked prefill exercised via a small max_tokens_per_step."""

    def _build_engines(self, **config_overrides):
        cpu = torch.device("cpu")
        seq_runtime = build_runtime(tiny_qwen38_spec(), cpu)
        cb_runtime = build_runtime(tiny_qwen38_spec(), cpu, attention="padded")
        cb_runtime.model.load_state_dict(seq_runtime.model.state_dict())
        config = BatchingConfig(**{
            "max_batch": 3, "max_seq_len": 64, "max_tokens_per_step": 8,
            **config_overrides,
        })
        return SequentialEngine(seq_runtime), ContinuousBatchingEngine.from_runtime(
            cb_runtime, config
        )

    def test_cb_matches_sequential_token_for_token(self):
        async def main():
            sequential, cb = self._build_engines()
            await cb.start()
            try:
                results = []
                for i, prompt in enumerate(PROMPTS):
                    seq = await collect(sequential, make_request(f"seq-{i}", prompt))
                    batched = await collect(cb, make_request(f"cb-{i}", prompt))
                    results.append((seq, batched))
                return results
            finally:
                await cb.shutdown()

        for i, (seq, batched) in enumerate(asyncio.run(main())):
            assert batched[0] == seq[0], f"prompt {i} diverged"
            assert batched[1] == seq[1]

    def test_chunked_prefill_via_tiny_step_budget(self):
        # max_tokens_per_step 4 forces the 8-token prompt through
        # multi-step prefill chunks with GDN state carry between them.
        async def main():
            sequential, cb = self._build_engines(max_tokens_per_step=4)
            await cb.start()
            try:
                seq = await collect(sequential, make_request("seq", PROMPTS[0], max_tokens=5))
                batched = await collect(cb, make_request("cb", PROMPTS[0], max_tokens=5))
                return seq, batched
            finally:
                await cb.shutdown()

        seq, batched = asyncio.run(main())
        assert batched[0] == seq[0]

    def test_concurrent_requests_share_the_batch(self):
        async def main():
            sequential, cb = self._build_engines()
            await cb.start()
            try:
                expected = []
                for i, prompt in enumerate(PROMPTS):
                    toks, _ = await collect(sequential, make_request(f"s{i}", prompt))
                    expected.append(toks)
                results = await asyncio.gather(*[
                    collect(cb, make_request(f"c{i}", prompt))
                    for i, prompt in enumerate(PROMPTS)
                ])
                return expected, results
            finally:
                await cb.shutdown()

        expected, results = asyncio.run(main())
        for i, (tokens, _) in enumerate(results):
            assert tokens == expected[i], f"concurrent request {i} diverged"
