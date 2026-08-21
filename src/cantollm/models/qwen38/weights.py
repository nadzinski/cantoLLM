"""Weight loading for Qwen 3.8: checkpoint-name mapping onto Qwen38.

Checkpoint layout (Qwen/Qwen3.8-27B and its FP8 variant): the text
decoder lives under `model.language_model.*`, the LM head at top-level
`lm_head.weight`, and the vision tower under `model.visual.*`. The PoC
is text-only, so visual weights are skipped; the release ships no MTP
weights but `mtp.*` is skipped defensively too.

Download + sharded/lazy loading + the FP8 wrap land with the spec
wiring chunk; this module starts at the mapping because the HF-parity
test loads its artifact through here, putting the naming and the
zero-centered-norm semantics under test from the first green run.
"""

import torch

from cantollm.models.qwen38.model import FULL

PREFIX = "model.language_model."
SKIP_PREFIXES = ("model.visual.",)


def _is_skipped(name: str) -> bool:
    return name.startswith(SKIP_PREFIXES) or ".mtp." in name or name.startswith("mtp.")


def build_param_mapping(model) -> dict:
    """Checkpoint name -> model parameter, driven by the model's own
    layer_types so full/linear layers map their respective mixers."""
    mapping = {
        f"{PREFIX}embed_tokens.weight": model.initial_embedding_layer.weight,
        f"{PREFIX}norm.weight": model.output_RMSNorm.weight,
        # Untied for the whole family: the checkpoint always ships lm_head.
        "lm_head.weight": model.output_layer.weight,
    }
    for i, kind in enumerate(model.layer_types):
        block = model.transformer_blocks[i]
        hf = f"{PREFIX}layers.{i}"
        mapping[f"{hf}.input_layernorm.weight"] = block.input_norm.weight
        mapping[f"{hf}.post_attention_layernorm.weight"] = block.post_attention_norm.weight
        mapping[f"{hf}.mlp.gate_proj.weight"] = block.feed_forward.linear_1.weight
        mapping[f"{hf}.mlp.up_proj.weight"] = block.feed_forward.linear_2.weight
        mapping[f"{hf}.mlp.down_proj.weight"] = block.feed_forward.linear_3.weight
        if kind == FULL:
            attn = block.attention
            mapping[f"{hf}.self_attn.q_proj.weight"] = attn.W_q.weight
            mapping[f"{hf}.self_attn.k_proj.weight"] = attn.W_k.weight
            mapping[f"{hf}.self_attn.v_proj.weight"] = attn.W_v.weight
            mapping[f"{hf}.self_attn.o_proj.weight"] = attn.out_proj.weight
            mapping[f"{hf}.self_attn.q_norm.weight"] = attn.q_norm.weight
            mapping[f"{hf}.self_attn.k_norm.weight"] = attn.k_norm.weight
        else:
            la = block.linear_attn
            for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
                mapping[f"{hf}.linear_attn.{name}.weight"] = getattr(la, name).weight
            mapping[f"{hf}.linear_attn.conv1d.weight"] = la.conv1d.weight
            mapping[f"{hf}.linear_attn.A_log"] = la.A_log
            mapping[f"{hf}.linear_attn.dt_bias"] = la.dt_bias
            mapping[f"{hf}.linear_attn.norm.weight"] = la.norm.weight
    return mapping


def load_weights_into_model(model, config: dict, weights_dict) -> None:
    """Copy checkpoint tensors into the model, loudly.

    Loud in both directions: a mapped name missing from the checkpoint
    raises, and so does a checkpoint key that is neither mapped nor a
    known skip (visual/mtp): silently ignored keys are how a renamed
    parameter ships zeros. `weights_dict` only needs mapping-style
    access (`in`, `[]`, iteration of keys), so a lazy sharded view
    works here too.
    """
    mapping = build_param_mapping(model)

    with torch.no_grad():
        for hf_name, param in mapping.items():
            if hf_name not in weights_dict:
                raise KeyError(f"Weight '{hf_name}' not found in checkpoint")
            weight = weights_dict[hf_name]
            if param.shape != weight.shape:
                raise ValueError(
                    f"Shape mismatch for '{hf_name}': "
                    f"model expects {tuple(param.shape)}, got {tuple(weight.shape)}"
                )
            param.copy_(weight)

    unexpected = [
        k for k in weights_dict
        if k not in mapping and not _is_skipped(k) and not k.endswith("_scale_inv")
    ]
    if unexpected:
        preview = ", ".join(sorted(unexpected)[:8])
        raise ValueError(
            f"{len(unexpected)} unexpected checkpoint keys (first: {preview})"
        )
