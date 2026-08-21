"""Partial rotary embeddings for the Qwen 3.8 (qwen3_5) architecture.

Qwen 3.8's full-attention layers rotate only the first `rotary_dim` dims
of each head (partial_rotary_factor 0.25: 64 of the 256-dim head); the
remaining dims carry no position signal and pass through untouched.

Within the rotated slice the layout is the same half-split convention as
models/rope.py, verified against HF transformers
models/qwen3_5/modeling_qwen3_5.py::apply_rotary_pos_emb: the head is
split at rotary_dim into (q_rot, q_pass), rotate_half acts on q_rot with
cos/sin tables built as cat((freqs, freqs), -1), which is exactly the
"first half real, second half imaginary" complex-multiply layout of the
shared helpers. So this module only slices; the rotation itself is
delegated.

The checkpoint also declares interleaved mRoPE (mrope_interleaved true,
mrope_section [11, 11, 10] over t/h/w). HF's apply_interleaved_mrope
overwrites t-stream frequencies with h/w-stream values at interleaved
indices; for text-only inference all three streams carry identical
position ids, so the result equals plain 1-D RoPE and none of that
machinery is needed here
(modeling_qwen3_5.py::Qwen3_5TextRotaryEmbedding.forward).
"""

import torch

from cantollm.models.rope import (
    apply_rotary_emb,
    apply_rotary_emb_batched,
    precompute_freqs_cis,
)


def precompute_partial_freqs_cis(rotary_dim: int, max_seq_len: int, theta: float):
    """Frequency table for the rotated slice only: (max_seq_len, rotary_dim / 2).

    rotary_dim is head_dim * partial_rotary_factor (64 for Qwen 3.8-27B),
    and theta is config.json's rope_theta (1e7 for Qwen 3.8).
    """
    return precompute_freqs_cis(rotary_dim, max_seq_len, theta=theta)


def apply_partial_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, offset: int = 0):
    """Rotate x[..., :rotary_dim]; pass x[..., rotary_dim:] through unchanged.

    rotary_dim is implied by the freqs table (2 * freqs_cis.shape[-1]),
    so a table covering the whole head makes this ordinary full RoPE.
    """
    rotary_dim = 2 * freqs_cis.shape[-1]
    if rotary_dim == x.shape[-1]:
        return apply_rotary_emb(x, freqs_cis, offset=offset)
    rotated = apply_rotary_emb(x[..., :rotary_dim], freqs_cis, offset=offset)
    return torch.cat([rotated, x[..., rotary_dim:]], dim=-1)


def apply_partial_rotary_emb_batched(
    x: torch.Tensor, freqs_cis: torch.Tensor, positions: torch.Tensor
):
    """Per-row-positions variant of `apply_partial_rotary_emb` for the
    continuous-batching path; same contract as
    models/rope.py::apply_rotary_emb_batched otherwise."""
    rotary_dim = 2 * freqs_cis.shape[-1]
    if rotary_dim == x.shape[-1]:
        return apply_rotary_emb_batched(x, freqs_cis, positions)
    rotated = apply_rotary_emb_batched(x[..., :rotary_dim], freqs_cis, positions)
    return torch.cat([rotated, x[..., rotary_dim:]], dim=-1)
