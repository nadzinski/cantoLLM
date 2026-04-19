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

from typing import Protocol

import torch


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
