"""Padded attention: the continuous-batching target (Phase 2).

Stubbed. Intended shape of the implementation, to be filled in:
  - Per-sequence masks: `build_mask` returns a batched mask accounting for
    each slot's own `start_pos` and sequence length.
  - Preallocated per-slot KV: a `max_batch × max_seq × ...` cache pool
    replaces the grow-via-cat dict. Writes target specific slot positions.
  - Variable `start_pos` per batch row: the scalar in the einsum protocol
    signature becomes a `(batch,)` tensor.

Both `forward_prefill` and `forward_decode` will exist so the SDPA
implementation in Phase 3 can specialize each shape.
"""

from __future__ import annotations

import torch


class PaddedAttentionMethod:
    def build_mask(
        self,
        start_pos: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        raise NotImplementedError("PaddedAttentionMethod: TODO")

    def forward_prefill(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        kv_cache: dict | None,
    ) -> torch.Tensor:
        raise NotImplementedError("PaddedAttentionMethod: TODO")

    def forward_decode(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        kv_cache: dict,
    ) -> torch.Tensor:
        raise NotImplementedError("PaddedAttentionMethod: TODO")
