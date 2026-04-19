"""Declarative model specs.

A `ModelSpec` is the recipe for one model: arch params, dtype, and the
callables that turn those into an on-device model + tokenizer. It's
consumed by `ModelRuntime` (see `runtime.py`); the CLI builds a spec via
`qwen3_spec(size)` instead of reaching into a config dict directly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from cantollm.models.qwen3.model import Qwen3
from cantollm.models.qwen3.tokenizer import Qwen3Tokenizer
from cantollm.models.qwen3.weights import download_weights, load_weights_into_model


MODEL_CONFIGS: dict[str, dict] = {
    "0.6B": {
        "token_count": 151_936,
        "max_seq_len": 40_960,
        "token_embedding_dim": 1024,
        "num_heads": 16,
        "num_transformers": 28,
        "expanded_dim": 3072,
        "num_groups": 8,
        "head_dim": 128,
        "dtype": torch.bfloat16,
    },
    "1.7B": {
        "token_count": 151_936,
        "max_seq_len": 40_960,
        "token_embedding_dim": 2048,
        "num_heads": 16,
        "num_transformers": 28,
        "expanded_dim": 6144,
        "num_groups": 8,
        "head_dim": 128,
        "dtype": torch.bfloat16,
    },
    "4B": {
        "token_count": 151_936,
        "max_seq_len": 40_960,
        "token_embedding_dim": 2560,
        "num_heads": 32,
        "num_transformers": 36,
        "expanded_dim": 9728,
        "num_groups": 8,
        "head_dim": 128,
        "dtype": torch.bfloat16,
    },
    "8B": {
        "token_count": 151_936,
        "max_seq_len": 40_960,
        "token_embedding_dim": 4096,
        "num_heads": 32,
        "num_transformers": 36,
        "expanded_dim": 12288,
        "num_groups": 8,
        "head_dim": 128,
        "dtype": torch.bfloat16,
    },
    "14B": {
        "token_count": 151_936,
        "max_seq_len": 40_960,
        "token_embedding_dim": 5120,
        "num_heads": 40,
        "num_transformers": 40,
        "expanded_dim": 17408,
        "num_groups": 8,
        "head_dim": 128,
        "dtype": torch.bfloat16,
    },
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    size: str
    arch: dict
    dtype: torch.dtype
    model_cls: type
    weights_loader: Callable[[], tuple[str, dict]]
    apply_weights: Callable[[Any, dict, dict], None]
    tokenizer_factory: Callable[[str], Any]
    chat_template: str


def qwen3_spec(size: str) -> ModelSpec:
    if size not in MODEL_CONFIGS:
        raise ValueError(f"Unknown Qwen3 size '{size}'. Must be one of {list(MODEL_CONFIGS)}")
    arch = MODEL_CONFIGS[size]

    def _load_weights() -> tuple[str, dict]:
        return download_weights(model_size=size, use_instruct=True)

    def _build_tokenizer(local_dir: str) -> Qwen3Tokenizer:
        return Qwen3Tokenizer(
            tokenizer_file_path=f"{local_dir}/tokenizer.json",
            is_instruct_model=True,
            apply_chat_template=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )

    return ModelSpec(
        name=f"qwen3-{size}",
        size=size,
        arch=arch,
        dtype=arch["dtype"],
        model_cls=Qwen3,
        weights_loader=_load_weights,
        apply_weights=load_weights_into_model,
        tokenizer_factory=_build_tokenizer,
        chat_template="qwen3-chatml",
    )
