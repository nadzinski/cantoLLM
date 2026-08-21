"""Tiny Qwen 3.8 fixture: 8 layers of the real [L, L, L, F] hybrid
pattern at toy dimensions, random init, fp32. Head ratios mirror the
27B (GDN v:k heads 3:1, GQA 2 groups); rotary_dim covers a quarter of
head_dim like the real partial factor.

Direct-construction helpers only for now; the ModelSpec wrapper (for
engine-level tests) arrives with the spec wiring chunk.
"""

import torch

from cantollm.models.qwen38.model import Qwen38, qwen38_layer_types

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
