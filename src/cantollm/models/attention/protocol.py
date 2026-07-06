"""Pluggable attention-compute boundary.

An `AttentionMethod` owns the attention math (scores + weighted values)
and the KV cache layout it reads/writes. `GroupedQueryAttention` delegates
to one: the module handles Q/K/V projections, head reshaping, q_norm/k_norm,
and RoPE, then hands post-RoPE tensors to the method for the compute.

The `Method` suffix disambiguates the protocol from the `GroupedQueryAttention`
module (also "an attention") and from the generation-strategy `InferenceBackend`
at `cantollm/engine/backend.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class BatchMeta:
    """Per-step batch geometry for the continuous-batching path.

    Built once per scheduler step and passed unchanged to every layer —
    per-row history lengths are the same across layers. Carries the same
    facts twice on purpose: `rows` for the per-row bookkeeping loops
    (ragged KV writes), the tensors for vectorized gathers (RoPE, slot
    reads). Row order matches the `(B, ...)` batch dim everywhere.
    """

    rows: list[tuple[int, int, int]]
    """Per row: (slot_idx, start_pos, num_new)."""

    slots: torch.Tensor
    """(B,) long — each row's slot in the KV pool."""

    start_pos: torch.Tensor
    """(B,) long — each row's first new token position."""

    num_new: torch.Tensor
    """(B,) long — real (unpadded) new tokens per row; always >= 1."""

    positions: torch.Tensor
    """(B, num_new_max) long — start_pos[b] + arange; pad columns unused."""

    num_new_max: int
    """Padded width of this step's input_ids."""

    max_history_len: int
    """max(start_pos + num_new) over rows — the KV span attention reads."""


class AttentionMethod(Protocol):
    def build_mask(
        self,
        start_pos: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Construct the causal mask this method expects.

        Shape is method-specific: the einsum path wants
        `(seq_len, start_pos + seq_len)` with `True` marking positions to
        mask out. Padded/paged methods will return per-sequence variants.
        """
        ...

    def forward_prefill(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        kv_cache: dict | None,
    ) -> torch.Tensor:
        """Full-prompt path: cache is empty (or None), compute attention
        over the whole sequence and populate the cache.

        Shapes:
          queries: (batch, seq, groups, heads_per_group, head_dim)
          keys:    (batch, seq, groups, head_dim)
          values:  (batch, seq, groups, head_dim)
          returns: (batch, seq, groups, heads_per_group, head_dim)
        """
        ...

    def forward_decode(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        kv_cache: dict,
    ) -> torch.Tensor:
        """Incremental path: cache is populated; append new K/V, compute
        attention of fresh queries against cached + fresh keys.

        Shapes match `forward_prefill`; the `seq` dim on the output matches
        the queries' `seq` (1 for single-token decode, >1 for speculative
        chunks).
        """
        ...

    def build_batched_mask(
        self,
        meta: BatchMeta,
        device: torch.device,
    ) -> torch.Tensor:
        """Mask for one mixed prefill/decode batch, built once per step.

        Returns (B, num_new_max, max_history_len) bool, True = masked.
        Pure per-row causality: mask[b, i, j] = j > start_pos[b] + i.
        That alone covers everything: future tokens, stale K/V beyond a
        row's own history (its hist_len is within the masked bound), and
        pad query rows stay finite (they attend to their own earlier
        keys) — the last-token gather never reads them. Callers broadcast
        over the group/head dims at use.
        """
        ...

    def forward_batched(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        layer_k: torch.Tensor,
        layer_v: torch.Tensor,
        meta: BatchMeta,
    ) -> torch.Tensor:
        """Mixed prefill/decode batch against a preallocated KV pool layer.

        Write each row's new K/V into `layer_k/v[slot, start:start+num_new]`
        (bounds-assert the write — one overlong row must fail loudly, not
        corrupt a neighbor slot), then attend each row's queries against its
        own slot history `[0, start_pos + num_new)`. Vectorize the math,
        loop the writes.

        Shapes:
          queries: (B, num_new_max, groups, heads_per_group, head_dim), post-RoPE
          keys:    (B, num_new_max, groups, head_dim), post-RoPE
          values:  (B, num_new_max, groups, head_dim)
          mask:    (B, num_new_max, max_history_len) bool, from build_batched_mask
          layer_k: (max_batch, max_seq_len, groups, head_dim) pool view, written in place
          layer_v: same shape as layer_k
          returns: (B, num_new_max, groups, heads_per_group, head_dim)
        """
        ...
