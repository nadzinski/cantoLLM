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
    """Per-layer K/V tensors, each of shape (max_batch, max_seq_len + 1,
    num_groups, head_dim), zero-initialized. Freed slots are not zeroed: the
    per-row causal mask already hides a previous occupant's stale K/V.

    The layers are separate tensors, not views of one stacked
    (num_layers, ...) tensor, and that is a load-bearing choice for
    torch.compile (torch-compile-design.md §4, learned the hard way on the
    2026-08-08 5090 round): the traced forward mutates `layer(i)` in place,
    and AOTAutograd keeps mutations of graph *inputs* in place but
    functionalizes mutations through *views* of inputs into full-base
    rebuilds — with a stacked pool that meant chained pool-sized copy
    kernels per layer (a ~23x decode slowdown and an OOM at 48 slots).
    Per-layer tensors are also the shape every production static-KV cache
    uses (vLLM, gpt-fast, HF StaticCache), for this same reason.

    The extra position column is `scratch_pos` (= max_seq_len): a parking
    spot for KV writes that must happen but must not land anywhere real.
    CUDA-graph replay needs the write map's length to be a constant of the
    step shape, so filler rows' map entries write their garbage here
    (cuda-graphs-design.md §3, decision 4), and the warm-up sweep's seeded
    maps do the same. Reads never see it: every gather spans
    `[:max_history_len]` with `max_history_len <= max_seq_len`
    (`Qwen3._validate_batched` enforces the cap). Cost: one position out of
    thousands, ~2 MB at the 0.6B serve geometry.
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
        shape = (max_batch, max_seq_len + 1, num_groups, head_dim)
        self.k_layers = [
            torch.zeros(shape, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        self.v_layers = [
            torch.zeros(shape, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        self.num_layers = num_layers
        self.max_batch = max_batch
        self.max_seq_len = max_seq_len
        self.scratch_pos = max_seq_len
        # The tensors' resolved device (e.g. cuda:0), not the constructor
        # argument (possibly bare "cuda") — device-identity comparisons
        # against tensors must not repeat the 40fbcf9 lesson. Resolved via
        # a throwaway tensor so a zero-layer pool (plumbing tests) works.
        self.device = torch.empty(0, device=device).device

    def layer(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(k, v) tensors for layer i, each (max_batch, max_seq_len + 1,
        num_groups, head_dim) — the layer's actual storage; attention
        writes through them into the pool. The last position is the
        scratch column: the KV scatter may write it, gathers never read
        it."""
        return self.k_layers[i], self.v_layers[i]

    def stacked_k(self) -> torch.Tensor:
        """All layers' K stacked into (num_layers, max_batch, ...) — a
        COPY, for tests and debugging only; writes to it do not reach the
        pool."""
        return torch.stack(self.k_layers)

    def stacked_v(self) -> torch.Tensor:
        """All layers' V stacked — a copy, same caveat as `stacked_k`."""
        return torch.stack(self.v_layers)
