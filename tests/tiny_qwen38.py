"""Tiny Qwen 3.8 fixture: 8 layers of the real [L, L, L, F] hybrid
pattern at toy dimensions, random init, fp32. Head ratios mirror the
27B (GDN v:k heads 3:1, GQA 2 groups); rotary_dim covers a quarter of
head_dim like the real partial factor.

`make_tiny_qwen38` for model-internals tests; `tiny_qwen38_spec` wraps
the same arch in a real ModelSpec (with the hybrid cache/pool hooks) so
engine-level tests can run the full runtime -> engine stack.
"""

import torch

from cantollm.models.qwen38.model import Qwen38, qwen38_layer_types
from cantollm.models.qwen38.pool import HybridCache, HybridStatePool
from cantollm.spec import ModelSpec
from tests.fakes import FakeTokenizer

TINY_QWEN38_ARCH = {
    "token_count": 2048,
    "token_embedding_dim": 64,
    "expanded_dim": 128,
    "num_transformers": 8,
    "layer_types": qwen38_layer_types(8),
    "num_heads": 4,
    "num_groups": 2,
    "head_dim": 16,
    "rotary_dim": 4,
    "rope_theta": 10_000_000.0,
    "linear_num_k_heads": 2,
    "linear_num_v_heads": 6,
    "linear_head_k_dim": 8,
    "linear_head_v_dim": 8,
    "linear_conv_kernel": 4,
    "max_seq_len": 128,
    "dtype": None,
}


def make_tiny_qwen38(attention_method, seed: int = 0) -> Qwen38:
    torch.manual_seed(seed)
    model = Qwen38(TINY_QWEN38_ARCH, attention_method=attention_method)
    model.eval()
    return model


def tiny_qwen38_spec() -> ModelSpec:
    def _no_load() -> tuple[str, dict]:
        return "", {}

    def _no_apply(model, config, weights_dict) -> None:
        pass  # random untied init; qwen38 never ties

    def _build_tokenizer(local_dir: str) -> FakeTokenizer:
        return FakeTokenizer()

    def _no_tokenizer_files() -> str:
        return ""

    def _cache_factory() -> HybridCache:
        return HybridCache(TINY_QWEN38_ARCH["layer_types"])

    def _kv_pool_factory(config, device) -> HybridStatePool:
        return HybridStatePool.from_arch(
            TINY_QWEN38_ARCH,
            max_batch=config.max_batch,
            max_seq_len=config.max_seq_len,
            dtype=torch.float32,
            device=device,
        )

    return ModelSpec(
        name="qwen38-tiny",
        size="tiny",
        arch=TINY_QWEN38_ARCH,
        dtype=torch.float32,
        model_cls=Qwen38,
        weights_loader=_no_load,
        apply_weights=_no_apply,
        tokenizer_factory=_build_tokenizer,
        tokenizer_files_loader=_no_tokenizer_files,
        chat_template="qwen38-chatml",
        cache_factory=_cache_factory,
        kv_pool_factory=_kv_pool_factory,
    )
