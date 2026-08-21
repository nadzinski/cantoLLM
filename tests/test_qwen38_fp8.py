"""Tests for the FP8 weight-only path (models/qwen38/fp8.py)."""

import pytest
import torch
import torch.nn.functional as F

from cantollm.models.qwen38.fp8 import FP8Linear


def make_fp8(out_features, in_features, block=(16, 8), seed=0):
    torch.manual_seed(seed)
    weight = torch.randn(out_features, in_features).to(torch.float8_e4m3fn)
    rows = -(-out_features // block[0])
    cols = -(-in_features // block[1])
    scale = torch.rand(rows, cols) + 0.5
    return weight, scale


def manual_dequant(weight, scale, block):
    """Independent reference: explicit per-block loop."""
    out = torch.empty(weight.shape, dtype=torch.float32)
    for bi in range(scale.shape[0]):
        for bj in range(scale.shape[1]):
            rs = slice(bi * block[0], min((bi + 1) * block[0], weight.shape[0]))
            cs = slice(bj * block[1], min((bj + 1) * block[1], weight.shape[1]))
            out[rs, cs] = weight[rs, cs].float() * scale[bi, bj]
    return out


class TestDequant:
    @pytest.mark.parametrize("shape", [(32, 16), (100, 70), (16, 8), (33, 9)])
    def test_matches_manual_blockwise_expansion(self, shape):
        """Including shapes that are NOT multiples of the block size."""
        block = (16, 8)
        weight, scale = make_fp8(*shape, block=block)
        layer = FP8Linear(weight, scale, block_size=block)
        expected = manual_dequant(weight, scale, block)
        assert torch.equal(layer.dequantized(torch.float32), expected)

    def test_forward_equals_linear_on_dequant(self):
        block = (16, 8)
        weight, scale = make_fp8(40, 24, block=block)
        layer = FP8Linear(weight, scale, block_size=block)
        x = torch.randn(3, 5, 24)
        expected = F.linear(x, manual_dequant(weight, scale, block))
        assert torch.allclose(layer(x), expected, atol=1e-6)

    def test_output_dtype_follows_input(self):
        weight, scale = make_fp8(16, 8, block=(16, 8))
        layer = FP8Linear(weight, scale, block_size=(16, 8))
        x = torch.randn(2, 8, dtype=torch.bfloat16)
        assert layer(x).dtype == torch.bfloat16


class TestValidation:
    def test_rejects_non_fp8_weight(self):
        with pytest.raises(ValueError, match="float8_e4m3fn"):
            FP8Linear(torch.randn(16, 8), torch.ones(1, 1), block_size=(16, 8))

    def test_rejects_mismatched_scale_shape(self):
        weight, _ = make_fp8(32, 16, block=(16, 8))
        with pytest.raises(ValueError, match="does not match"):
            FP8Linear(weight, torch.ones(3, 3), block_size=(16, 8))


class TestModuleBehavior:
    def test_no_parameters_only_buffers(self):
        weight, scale = make_fp8(16, 8, block=(16, 8))
        layer = FP8Linear(weight, scale, block_size=(16, 8))
        assert list(layer.parameters()) == []
        names = {name for name, _ in layer.named_buffers()}
        assert names == {"weight", "weight_scale_inv"}

    def test_state_dict_round_trip(self):
        weight, scale = make_fp8(16, 8, block=(16, 8))
        a = FP8Linear(weight, scale, block_size=(16, 8))
        b = FP8Linear(*make_fp8(16, 8, block=(16, 8), seed=1), block_size=(16, 8))
        b.load_state_dict(a.state_dict())
        assert torch.equal(
            a.dequantized(torch.float32), b.dequantized(torch.float32)
        )
