"""Preallocated multi-layer KV storage for the continuous-batching path.

Memory only — the pool knows nothing about which slots are in use. Allocator
state (the free list) lives with the scheduler in
`engine/batching/allocator.py` (decision 1: runtime owns the memory,
scheduler owns the allocator, a sequence carries its slot index). Phase 4's
paged cache keeps the same seam: a block table is also just a handle into a
runtime-owned pool.
"""

from __future__ import annotations

import torch


class PaddedKVPool:
    """K/V tensors of shape (num_layers, max_batch, max_seq_len, num_groups,
    head_dim), zero-initialized. Freed slots are not zeroed: the per-row
    causal mask already hides a previous occupant's stale K/V.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        max_batch: int,
        max_seq_len: int,
        num_groups: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        shape = (num_layers, max_batch, max_seq_len, num_groups, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.num_layers = num_layers
        self.max_batch = max_batch
        self.max_seq_len = max_seq_len

    def layer(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(k, v) views for layer i, each (max_batch, max_seq_len, num_groups,
        head_dim) — no copy; attention writes through them into the pool."""
        return self.k[i], self.v[i]
