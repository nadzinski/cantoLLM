"""Tests for the Qwen 3.8 hybrid model (models/qwen38/model.py), sequential path.

The core check is the self-oracle: one full-sequence forward must match
a prefill + token-by-token decode replay through the HybridCache, which
exercises KV growth on full layers and GDN state carry on linear layers
together. Plus cache validation error paths and HybridCache behavior.
"""

import pytest
import torch

from cantollm.models.attention import EinsumAttentionMethod
from cantollm.models.qwen38.model import FULL, LINEAR, ZeroCenteredRMSNorm, qwen38_layer_types
from cantollm.models.qwen38.pool import HybridCache
from tests.tiny_qwen38 import TINY_QWEN38_ARCH, make_tiny_qwen38


@pytest.fixture(scope="module")
def model():
    return make_tiny_qwen38(EinsumAttentionMethod())


def fresh_cache():
    return HybridCache(TINY_QWEN38_ARCH["layer_types"])


class TestLayerPattern:
    def test_layer_types_match_checkpoint_indices(self):
        types = qwen38_layer_types(8)
        assert [i for i, t in enumerate(types) if t == FULL] == [3, 7]
        assert types.count(LINEAR) == 6

    def test_27b_pattern(self):
        types = qwen38_layer_types(64)
        assert [i for i, t in enumerate(types) if t == FULL] == list(range(3, 64, 4))


class TestZeroCenteredRMSNorm:
    def test_zero_weight_is_plain_rmsnorm(self):
        norm = ZeroCenteredRMSNorm(8)
        x = torch.randn(2, 8)
        expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        assert torch.allclose(norm(x), expected, atol=1e-6)

    def test_weight_is_an_offset_from_one(self):
        norm = ZeroCenteredRMSNorm(8)
        with torch.no_grad():
            norm.weight.fill_(0.5)
        x = torch.randn(2, 8)
        base = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        assert torch.allclose(norm(x), base * 1.5, atol=1e-6)


class TestForwardShapes:
    def test_no_cache_forward(self, model):
        tokens = torch.randint(0, 2048, (2, 7))
        with torch.inference_mode():
            logits = model(tokens, start_pos=0)
        assert logits.shape == (2, 7, 2048)

    def test_deterministic(self, model):
        tokens = torch.randint(0, 2048, (1, 5))
        with torch.inference_mode():
            a = model(tokens, start_pos=0)
            b = model(tokens, start_pos=0)
        assert torch.equal(a, b)


class TestIncrementalReplay:
    def test_prefill_then_decode_matches_full_forward(self, model):
        """The KV cache + GDN state carry must make incremental decode
        indistinguishable from one full forward."""
        torch.manual_seed(1)
        tokens = torch.randint(0, 2048, (1, 12))
        prefill_len = 8

        with torch.inference_mode():
            full = model(tokens, start_pos=0)

            cache = fresh_cache()
            prefill = model(tokens[:, :prefill_len], start_pos=0, kv_cache=cache)
            assert torch.allclose(full[:, :prefill_len], prefill, atol=1e-5)
            assert cache.position == prefill_len

            for pos in range(prefill_len, 12):
                step = model(tokens[:, pos : pos + 1], start_pos=pos, kv_cache=cache)
                assert torch.allclose(full[:, pos : pos + 1], step, atol=1e-5), (
                    f"decode diverged at position {pos}"
                )
        assert cache.position == 12

    def test_chunked_prefill_matches_full_forward(self, model):
        torch.manual_seed(2)
        tokens = torch.randint(0, 2048, (1, 10))
        with torch.inference_mode():
            full = model(tokens, start_pos=0)

            cache = fresh_cache()
            outs = []
            for start, end in [(0, 4), (4, 5), (5, 10)]:
                outs.append(model(tokens[:, start:end], start_pos=start, kv_cache=cache))
        assert torch.allclose(full, torch.cat(outs, dim=1), atol=1e-5)


class TestCacheValidation:
    def test_wrong_layer_count_rejected(self, model):
        cache = HybridCache(qwen38_layer_types(4))
        with pytest.raises(ValueError, match="4 entries"):
            model(torch.randint(0, 2048, (1, 3)), start_pos=0, kv_cache=cache)

    def test_start_pos_without_cache_rejected(self, model):
        with pytest.raises(ValueError, match="no kv_cache"):
            model(torch.randint(0, 2048, (1, 1)), start_pos=3)

    def test_gdn_position_mismatch_rejected(self, model):
        cache = fresh_cache()
        with torch.inference_mode():
            model(torch.randint(0, 2048, (1, 4)), start_pos=0, kv_cache=cache)
        # Tamper with one GDN layer's position counter: replay forbidden.
        cache[0]["pos"] = 2
        with pytest.raises(ValueError, match="cannot replay or skip"):
            model(torch.randint(0, 2048, (1, 1)), start_pos=4, kv_cache=cache)

    def test_full_layer_length_mismatch_rejected(self, model):
        cache = fresh_cache()
        with torch.inference_mode():
            model(torch.randint(0, 2048, (1, 4)), start_pos=0, kv_cache=cache)
        # Shorten only the full-attention layer (index 3): GDN counters
        # still agree with start_pos, so the KV-length branch must fire.
        cache[3]["keys"] = cache[3]["keys"][:, :3]
        with pytest.raises(ValueError, match=r"kv_cache\[3\] has 3 positions"):
            model(torch.randint(0, 2048, (1, 1)), start_pos=4, kv_cache=cache)


class TestHybridCache:
    def test_position_empty_and_after_reset(self, model):
        cache = fresh_cache()
        assert cache.position == 0
        with torch.inference_mode():
            model(torch.randint(0, 2048, (1, 6)), start_pos=0, kv_cache=cache)
        assert cache.position == 6
        cache.reset()
        assert cache.position == 0
        for kind, layer in zip(cache.layer_types, cache):
            if kind == LINEAR:
                assert layer["S"] is None and layer["pos"] == 0
            else:
                assert layer["keys"] is None

    def test_truncate_refuses(self):
        with pytest.raises(NotImplementedError, match="rewind"):
            fresh_cache().truncate(3)

    def test_requires_a_full_attention_layer(self):
        with pytest.raises(AssertionError):
            HybridCache([LINEAR, LINEAR])
