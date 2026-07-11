"""Padded attention: the continuous-batching target (Phase 2).

Batched-only, the mirror image of `EinsumAttentionMethod`: the sequential
methods raise here, the batched ones raise there. There is deliberately no
prefill/decode split — a continuous-batching step mixes prefill-chunk rows
and decode rows in one forward, so the padded path has a single mixed-batch
entrypoint (`forward_batched`) driven by per-row `BatchMeta` geometry
against a preallocated `PaddedKVPool` layer.

Fill-in order per continuous-batching-plan.md:
  - `build_batched_mask` — step 4 (geometry/bookkeeping).
  - `forward_batched` — step 5, hand-written (the attention math).
Shape contracts live on the `AttentionMethod` protocol docstrings.
"""

from __future__ import annotations

import torch

from cantollm.models.attention.protocol import BatchMeta


class PaddedAttentionMethod:
    def build_mask(
        self,
        start_pos: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        raise NotImplementedError("PaddedAttentionMethod is batched-only")

    def forward_prefill(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        kv_cache: dict | None,
    ) -> torch.Tensor:
        raise NotImplementedError("PaddedAttentionMethod is batched-only")

    def forward_decode(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        kv_cache: dict,
    ) -> torch.Tensor:
        raise NotImplementedError("PaddedAttentionMethod is batched-only")

    def build_batched_mask(
        self,
        meta: BatchMeta,
        device: torch.device,
    ) -> torch.Tensor:
        # mask[b, i, j] = j > start_pos[b] + i  (True = masked out).
        # Pure per-row causality; see the protocol docstring for why this
        # alone also fences stale slot data and keeps pad rows finite.
        i = torch.arange(meta.num_new_max, device=device)
        j = torch.arange(meta.max_history_len, device=device)
        causal_bound = meta.start_pos.to(device)[:, None] + i[None, :]
        return j[None, None, :] > causal_bound[:, :, None]

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
        raise NotImplementedError("PaddedAttentionMethod: TODO (step 5)")
