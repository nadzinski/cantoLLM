"""Download Qwen3 model weights from HuggingFace and load them into the custom model."""

import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file

VALID_SIZES = ("0.6B", "1.7B", "4B", "8B", "14B")
MODEL_DATA_DIR = Path(__file__).parent.parent / "model_data"


def download_weights(
    model_size: str = "0.6B", use_instruct: bool = True
) -> tuple[str, dict]:
    """Download Qwen3 weights from HuggingFace and return the local dir and weight dict.

    Args:
        model_size: One of "0.6B", "1.7B", "4B", "8B", "14B".
        use_instruct: If True, download the instruct model; otherwise the base model.

    Returns:
        A tuple of (local_dir_path, weights_dict) where weights_dict maps
        HuggingFace parameter names to tensors.
    """
    if model_size not in VALID_SIZES:
        raise ValueError(
            f"Invalid model size '{model_size}'. Must be one of {VALID_SIZES}"
        )

    model_name = f"Qwen3-{model_size}" if use_instruct else f"Qwen3-{model_size}-Base"
    repo_id = f"Qwen/{model_name}"
    local_dir = MODEL_DATA_DIR / model_name

    # Download the tokenizer
    hf_hub_download(
        repo_id=repo_id,
        filename="tokenizer.json",
        local_dir=str(local_dir),
    )

    if model_size == "0.6B":
        # Small model: single safetensors file
        hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
            local_dir=str(local_dir),
        )
        weights_dict = load_file(str(local_dir / "model.safetensors"))
    else:
        # Larger models: sharded safetensors files
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
        )

        index_path = local_dir / "model.safetensors.index.json"
        with open(index_path) as f:
            index = json.load(f)

        # Collect unique shard filenames
        shard_files = sorted(set(index["weight_map"].values()))

        # Load and merge all shards into one dict
        weights_dict = {}
        for shard_file in shard_files:
            shard_path = local_dir / shard_file
            shard_weights = load_file(str(shard_path))
            weights_dict.update(shard_weights)

    return str(local_dir), weights_dict


def load_weights_into_model(model, config: dict, weights_dict: dict) -> None:
    """Map HuggingFace weight names onto the custom Qwen3 model parameters.

    Args:
        model: A Qwen3 model instance.
        config: Model config dict containing at least "num_transformers".
        weights_dict: Dict mapping HuggingFace weight names to tensors.
    """
    num_layers = config["num_transformers"]

    # Build the mapping from HuggingFace names to model parameters
    mapping = {}

    # Embedding
    mapping["model.embed_tokens.weight"] = model.initial_embedding_layer.weight

    # Transformer blocks
    for i in range(num_layers):
        block = model.transformer_blocks[i]
        hf = f"model.layers.{i}"

        # Self-attention projections
        mapping[f"{hf}.self_attn.q_proj.weight"] = block.GQA.W_q.weight
        mapping[f"{hf}.self_attn.k_proj.weight"] = block.GQA.W_k.weight
        mapping[f"{hf}.self_attn.v_proj.weight"] = block.GQA.W_v.weight
        mapping[f"{hf}.self_attn.o_proj.weight"] = block.GQA.out_proj.weight

        # QK norms
        mapping[f"{hf}.self_attn.q_norm.weight"] = block.GQA.q_norm.scaling_weight
        mapping[f"{hf}.self_attn.k_norm.weight"] = block.GQA.k_norm.scaling_weight

        # Layer norms
        mapping[f"{hf}.input_layernorm.weight"] = block.RMSNorm_1.scaling_weight
        mapping[f"{hf}.post_attention_layernorm.weight"] = block.RMSNorm_2.scaling_weight

        # Feed-forward
        mapping[f"{hf}.mlp.gate_proj.weight"] = block.FF.linear_1.weight
        mapping[f"{hf}.mlp.up_proj.weight"] = block.FF.linear_2.weight
        mapping[f"{hf}.mlp.down_proj.weight"] = block.FF.linear_3.weight

    # Output norm and head
    mapping["model.norm.weight"] = model.output_RMSNorm.scaling_weight

    if "lm_head.weight" in weights_dict:
        mapping["lm_head.weight"] = model.output_layer.weight

    # Copy weights into model parameters
    with torch.no_grad():
        for hf_name, param in mapping.items():
            if hf_name not in weights_dict:
                raise KeyError(f"Weight '{hf_name}' not found in weights_dict")
            weight = weights_dict[hf_name]

            if param.shape != weight.shape:
                raise ValueError(
                    f"Shape mismatch for '{hf_name}': "
                    f"model expects {param.shape}, got {weight.shape}"
                )

            param.copy_(weight)

    # Weight tying: output layer shares the same parameter as the embedding
    if "lm_head.weight" not in weights_dict:
        model.output_layer.weight = model.initial_embedding_layer.weight
