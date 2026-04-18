import pytest
import torch

from cantollm.models.qwen3.model import Qwen3
from cantollm.models.qwen3.weights import VALID_SIZES, load_weights_into_model


def make_config():
    return {
        "token_count": 1000,
        "token_embedding_dim": 64,
        "expanded_dim": 128,
        "num_heads": 8,
        "num_groups": 4,
        "head_dim": 8,
        "max_seq_len": 128,
        "num_transformers": 2,
        "dtype": None,
    }


def make_fake_weights(config):
    """Build a weights_dict that matches what HuggingFace would provide."""
    num_layers = config["num_transformers"]
    emb_dim = config["token_embedding_dim"]
    exp_dim = config["expanded_dim"]
    num_heads = config["num_heads"]
    num_groups = config["num_groups"]
    head_dim = config["head_dim"]
    vocab = config["token_count"]
    q_out = num_heads * head_dim
    kv_dim = num_groups * head_dim

    w = {}
    w["model.embed_tokens.weight"] = torch.randn(vocab, emb_dim)

    for i in range(num_layers):
        p = f"model.layers.{i}"
        w[f"{p}.self_attn.q_proj.weight"] = torch.randn(q_out, emb_dim)
        w[f"{p}.self_attn.k_proj.weight"] = torch.randn(kv_dim, emb_dim)
        w[f"{p}.self_attn.v_proj.weight"] = torch.randn(kv_dim, emb_dim)
        w[f"{p}.self_attn.o_proj.weight"] = torch.randn(emb_dim, q_out)
        w[f"{p}.self_attn.q_norm.weight"] = torch.randn(head_dim)
        w[f"{p}.self_attn.k_norm.weight"] = torch.randn(head_dim)
        w[f"{p}.input_layernorm.weight"] = torch.randn(emb_dim)
        w[f"{p}.post_attention_layernorm.weight"] = torch.randn(emb_dim)
        w[f"{p}.mlp.gate_proj.weight"] = torch.randn(exp_dim, emb_dim)
        w[f"{p}.mlp.up_proj.weight"] = torch.randn(exp_dim, emb_dim)
        w[f"{p}.mlp.down_proj.weight"] = torch.randn(emb_dim, exp_dim)

    w["model.norm.weight"] = torch.randn(emb_dim)
    w["lm_head.weight"] = torch.randn(vocab, emb_dim)

    return w


def test_load_weights_copies_values():
    """Weights from the dict should actually end up in the model parameters."""
    config = make_config()
    model = Qwen3(config)
    weights = make_fake_weights(config)

    load_weights_into_model(model, config, weights)

    assert torch.equal(model.initial_embedding_layer.weight.data, weights["model.embed_tokens.weight"])
    assert torch.equal(model.output_layer.weight.data, weights["lm_head.weight"])
    assert torch.equal(
        model.transformer_blocks[0].GQA.W_q.weight.data,
        weights["model.layers.0.self_attn.q_proj.weight"],
    )
    assert torch.equal(
        model.transformer_blocks[1].FF.linear_3.weight.data,
        weights["model.layers.1.mlp.down_proj.weight"],
    )


def test_load_weights_all_layers_populated():
    """Every transformer block should receive its weights, not just the first."""
    config = make_config()
    model = Qwen3(config)
    weights = make_fake_weights(config)

    load_weights_into_model(model, config, weights)

    for i in range(config["num_transformers"]):
        block = model.transformer_blocks[i]
        expected = weights[f"model.layers.{i}.self_attn.v_proj.weight"]
        assert torch.equal(block.GQA.W_v.weight.data, expected), f"Block {i} V weights not loaded"


def test_load_weights_weight_tying():
    """When lm_head.weight is absent, output layer should be the same parameter as embedding."""
    config = make_config()
    model = Qwen3(config)
    weights = make_fake_weights(config)
    del weights["lm_head.weight"]

    load_weights_into_model(model, config, weights)

    # Must be the same object, not just equal values
    assert model.output_layer.weight is model.initial_embedding_layer.weight


def test_load_weights_shape_mismatch_raises():
    """A shape mismatch between model and weights should raise ValueError."""
    config = make_config()
    model = Qwen3(config)
    weights = make_fake_weights(config)
    # Corrupt one weight to have the wrong shape
    weights["model.norm.weight"] = torch.randn(999)

    with pytest.raises(ValueError, match="Shape mismatch"):
        load_weights_into_model(model, config, weights)


def test_load_weights_missing_key_raises():
    """A missing key in weights_dict should raise KeyError."""
    config = make_config()
    model = Qwen3(config)
    weights = make_fake_weights(config)
    del weights["model.layers.0.self_attn.q_proj.weight"]

    with pytest.raises(KeyError, match="not found"):
        load_weights_into_model(model, config, weights)


def test_download_weights_rejects_invalid_size():
    """download_weights should reject sizes not in VALID_SIZES."""
    from cantollm.models.qwen3.weights import download_weights

    with pytest.raises(ValueError, match="Invalid model size"):
        download_weights(model_size="99B")


def test_valid_sizes_constant():
    """Sanity check that the exported sizes match the known Qwen3 lineup."""
    assert VALID_SIZES == ("0.6B", "1.7B", "4B", "8B", "14B")
