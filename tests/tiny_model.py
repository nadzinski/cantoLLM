"""Tiny Qwen3 fixture for fast end-to-end tests.

Builds a 2-layer, 64-dim Qwen3 with random init weights wrapped in a real
ModelSpec/ModelRuntime/StandardBackend stack. Use this when a test needs to
exercise SequentialEngine -> backend -> Qwen3 -> KVCache without HF downloads.

For model-internals tests, use tests/test_qwen3.py:make_config() directly.
For API/SSE contract tests, use tests/fakes.py:FakeEngine.
"""

import torch

from cantollm.models.qwen3.model import Qwen3
from cantollm.spec import ModelSpec
from tests.fakes import FakeTokenizer


TINY_ARCH = {
    "token_count": 2048,
    "token_embedding_dim": 64,
    "expanded_dim": 128,
    "num_heads": 8,
    "num_groups": 4,
    "head_dim": 8,
    "max_seq_len": 128,
    "num_transformers": 2,
    "dtype": None,
}


def tiny_qwen3_spec() -> ModelSpec:
    def _no_load() -> tuple[str, dict]:
        return "", {}

    def _no_apply(model, config, weights_dict) -> None:
        # Mirror what load_weights_into_model does for small Qwen3
        # (weights.py:137-138): tie output projection to the embedding.
        model.output_layer.weight = model.initial_embedding_layer.weight

    def _build_tokenizer(local_dir: str) -> FakeTokenizer:
        return FakeTokenizer()

    return ModelSpec(
        name="qwen3-tiny",
        size="tiny",
        arch=TINY_ARCH,
        dtype=torch.float32,
        model_cls=Qwen3,
        weights_loader=_no_load,
        apply_weights=_no_apply,
        tokenizer_factory=_build_tokenizer,
        chat_template="qwen3-chatml",
    )
