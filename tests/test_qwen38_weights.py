"""Tests for the qwen38 weight loader: mapping round-trip, dtype-driven
FP8 wrapping, meta-device materialization, skip/loudness rules, and the
lazy ShardedWeights view. (Naming against real HF output is covered by
test_qwen38_hf_parity.py; these tests cover the loader mechanics.)"""

import json

import pytest
import torch
from safetensors.torch import save_file

from cantollm.models.attention import EinsumAttentionMethod
from cantollm.models.qwen38.fp8 import FP8Linear
from cantollm.models.qwen38.model import Qwen38
from cantollm.models.qwen38.weights import (
    ShardedWeights,
    build_param_mapping,
    load_weights_into_model,
)
from tests.tiny_qwen38 import TINY_QWEN38_ARCH, make_tiny_qwen38


def export_checkpoint(model) -> dict:
    """Fabricate a checkpoint dict in HF naming from a live tiny model."""
    return {
        name: entry.param.detach().clone()
        for name, entry in build_param_mapping(model).items()
    }


@pytest.fixture(scope="module")
def source_model():
    return make_tiny_qwen38(EinsumAttentionMethod(), seed=7)


@pytest.fixture()
def checkpoint(source_model):
    return export_checkpoint(source_model)


def fresh_model(**arch_overrides):
    arch = {**TINY_QWEN38_ARCH, **arch_overrides}
    return Qwen38(arch, attention_method=EinsumAttentionMethod())


def logits(model, tokens):
    with torch.inference_mode():
        return model(tokens, start_pos=0)


class TestPlainLoad:
    def test_round_trip_reproduces_outputs(self, source_model, checkpoint):
        target = fresh_model()
        load_weights_into_model(target, TINY_QWEN38_ARCH, checkpoint)
        target.eval()
        tokens = torch.randint(0, 2048, (1, 9))
        assert torch.equal(logits(source_model, tokens), logits(target, tokens))

    def test_missing_key_raises(self, checkpoint):
        del checkpoint["model.language_model.layers.0.linear_attn.A_log"]
        with pytest.raises(KeyError, match="A_log"):
            load_weights_into_model(fresh_model(), TINY_QWEN38_ARCH, checkpoint)

    def test_unexpected_key_raises(self, checkpoint):
        checkpoint["model.language_model.layers.0.renamed_thing.weight"] = torch.zeros(2)
        with pytest.raises(ValueError, match="unexpected checkpoint keys"):
            load_weights_into_model(fresh_model(), TINY_QWEN38_ARCH, checkpoint)

    def test_visual_and_mtp_keys_are_skipped(self, checkpoint):
        checkpoint["model.visual.blocks.0.attn.qkv.weight"] = torch.zeros(4, 4)
        checkpoint["model.language_model.mtp.layers.0.mlp.gate_proj.weight"] = torch.zeros(2, 2)
        load_weights_into_model(fresh_model(), TINY_QWEN38_ARCH, checkpoint)

    def test_shape_mismatch_raises(self, checkpoint):
        checkpoint["lm_head.weight"] = torch.zeros(3, 3)
        with pytest.raises(ValueError, match="Shape mismatch"):
            load_weights_into_model(fresh_model(), TINY_QWEN38_ARCH, checkpoint)


class TestFP8Wrap:
    def quantize(self, checkpoint, name):
        w = checkpoint[name]
        checkpoint[name] = w.to(torch.float8_e4m3fn)
        rows = -(-w.shape[0] // 128)
        cols = -(-w.shape[1] // 128)
        checkpoint[f"{name}_scale_inv"] = torch.ones(rows, cols)

    def test_fp8_weight_swaps_linear_for_fp8linear(self, checkpoint):
        name = "model.language_model.layers.0.mlp.gate_proj.weight"
        self.quantize(checkpoint, name)
        target = fresh_model()
        load_weights_into_model(target, TINY_QWEN38_ARCH, checkpoint)
        swapped = target.transformer_blocks[0].feed_forward.linear_1
        assert isinstance(swapped, FP8Linear)
        # Unquantized neighbors stay plain Linears with exact weights.
        assert isinstance(target.transformer_blocks[0].feed_forward.linear_2, torch.nn.Linear)
        # Forward still runs end to end.
        target.eval()
        logits(target, torch.randint(0, 2048, (1, 4)))

    def test_bf16_checkpoint_produces_zero_fp8_modules(self, checkpoint):
        target = fresh_model()
        load_weights_into_model(target, TINY_QWEN38_ARCH, checkpoint)
        assert not any(isinstance(m, FP8Linear) for m in target.modules())

    def test_fp8_without_scale_raises(self, checkpoint):
        name = "model.language_model.layers.0.mlp.gate_proj.weight"
        checkpoint[name] = checkpoint[name].to(torch.float8_e4m3fn)
        with pytest.raises(KeyError, match="_scale_inv"):
            load_weights_into_model(fresh_model(), TINY_QWEN38_ARCH, checkpoint)


class TestMetaMaterialization:
    def test_meta_model_materializes_and_matches(self, source_model, checkpoint):
        target = fresh_model(init_device="meta")
        assert next(target.parameters()).is_meta
        load_weights_into_model(target, TINY_QWEN38_ARCH, checkpoint)
        assert not any(p.is_meta for p in target.parameters())
        target.eval()
        tokens = torch.randint(0, 2048, (1, 6))
        assert torch.equal(logits(source_model, tokens), logits(target, tokens))

    def test_freqs_buffer_is_real_despite_meta_init(self):
        target = fresh_model(init_device="meta")
        assert not target.freqs_cis.is_meta

    def test_fp8_wrap_works_on_meta_model(self, checkpoint):
        name = "model.language_model.layers.0.mlp.gate_proj.weight"
        TestFP8Wrap().quantize(checkpoint, name)
        target = fresh_model(init_device="meta")
        load_weights_into_model(target, TINY_QWEN38_ARCH, checkpoint)
        assert isinstance(target.transformer_blocks[0].feed_forward.linear_1, FP8Linear)
        assert not any(p.is_meta for p in target.parameters())


class TestShardedWeights:
    def write_shards(self, tmp_path, with_index=True):
        a = {"model.language_model.embed_tokens.weight": torch.randn(4, 2)}
        b = {"lm_head.weight": torch.randn(4, 2)}
        save_file(a, str(tmp_path / "shard-a.safetensors"))
        save_file(b, str(tmp_path / "shard-b.safetensors"))
        if with_index:
            index = {"weight_map": {
                "model.language_model.embed_tokens.weight": "shard-a.safetensors",
                "lm_head.weight": "shard-b.safetensors",
            }}
            (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
        return a, b

    def test_indexed_access(self, tmp_path):
        a, b = self.write_shards(tmp_path)
        weights = ShardedWeights(str(tmp_path))
        assert len(weights) == 2
        assert "lm_head.weight" in weights
        assert torch.equal(weights["lm_head.weight"], b["lm_head.weight"])
        assert torch.equal(
            weights["model.language_model.embed_tokens.weight"],
            a["model.language_model.embed_tokens.weight"],
        )

    def test_no_index_falls_back_to_header_scan(self, tmp_path):
        a, _ = self.write_shards(tmp_path, with_index=False)
        weights = ShardedWeights(str(tmp_path))
        assert set(weights) == {
            "model.language_model.embed_tokens.weight", "lm_head.weight",
        }
        assert torch.equal(
            weights["model.language_model.embed_tokens.weight"],
            a["model.language_model.embed_tokens.weight"],
        )

    def test_missing_key_raises(self, tmp_path):
        self.write_shards(tmp_path)
        with pytest.raises(KeyError):
            ShardedWeights(str(tmp_path))["nope"]

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ShardedWeights(str(tmp_path))
