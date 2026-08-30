"""Preallocated multi-layer KV storage for the continuous-batching path.

Memory only — the pool knows nothing about which slots are in use. Allocator
state (the free list) lives with the scheduler in
`engine/batching/allocator.py` (decision 1: runtime owns the memory,
scheduler owns the allocator, a sequence carries its slot index). Phase 4's
paged cache keeps the same seam: a block table is also just a handle into a
runtime-owned pool.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class KVPool(Protocol):
    """The pool surface everything outside the attention method consumes
    (Phase 4 chunk 1, paged-kv-plan.md §5).

    `BatchedForwardFn`, the scheduler, and the runtime front are typed
    against this, so the paged pool slots in beside `PaddedKVPool` without
    retyping them. `max_seq_len` here means per-request logical token
    capacity — the admission cap and (for the paged pool) the block-table
    length bound — not preallocated memory: each layout sizes its memory its
    own way (`max_batch` slots here, `num_kv_blocks` there). Layout-specific
    surface (`scratch_pos` vs a scratch block, slot vs block geometry)
    deliberately stays off this protocol; code that needs it (warm-up map
    seeding, graph marshaling) is layout-specific by nature and keeps the
    concrete type.
    """

    num_layers: int
    max_seq_len: int
    device: torch.device

    def layer(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(k, v) storage tensors for layer i; shapes are layout-defined."""
        ...


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


class PagedKVPool:
    """Block-indexed KV storage for the paged path (Phase 4,
    paged-kv-plan.md §2.3/§2.4). Memory only, like `PaddedKVPool`: block
    tables and the allocator live with the scheduler; a sequence carries a
    block table the way it carries a slot index today.

    Each layer's K and V is ONE FLAT tensor of shape
    `((num_kv_blocks + 1) * block_size, num_groups, head_dim)`: block `b`
    occupies first-dimension indices
    `[b * block_size, (b + 1) * block_size)`. A token at logical position
    `pos` of a request whose table maps `pos // block_size -> b` lives at
    flat pool index
    `b * block_size + pos % block_size`. Flat rather than
    `(blocks, block_size, ...)` on purpose: the KV scatter must mutate the
    layer tensor DIRECTLY — under torch.compile a write through a view of
    the base regresses to pool-scale copy chains (the ab4f438 lesson the
    padded pool's docstring records). Reads may view the flat tensor
    however the attention method likes; only writes through views are
    poison.

    The extra block past the allocatable range is the SCRATCH BLOCK
    (`scratch_block == num_kv_blocks`), the paged analog of `scratch_pos`:
    filler-row and warm-up writes park there, and a filler row's table
    points at it with one visible block so its softmax row stays finite
    (a fully-masked row is NaN — paged-kv-plan.md §4). The allocator is
    sized to `num_kv_blocks` and never hands it out.

    Freed blocks are not zeroed: masking must hide a previous occupant's
    stale K/V, exactly as freed padded slots rely on the causal mask (the
    flex equivalence suite pins stale-block reuse).
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_kv_blocks: int,
        block_size: int,
        max_seq_len: int,
        num_groups: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        shape = ((num_kv_blocks + 1) * block_size, num_groups, head_dim)
        self.k_layers = [
            torch.zeros(shape, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        self.v_layers = [
            torch.zeros(shape, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        self.num_layers = num_layers
        self.num_kv_blocks = num_kv_blocks
        self.block_size = block_size
        # Per-request LOGICAL capacity (admission cap, block-table length
        # bound) — not memory, which is sized by num_kv_blocks. KVPool
        # protocol semantics.
        self.max_seq_len = max_seq_len
        self.scratch_block = num_kv_blocks
        # Resolved device, same rationale as PaddedKVPool.
        self.device = torch.empty(0, device=device).device

    def layer(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(k, v) storage for layer i, each
        `((num_kv_blocks + 1) * block_size, num_groups, head_dim)` — the
        actual flat storage; attention writes through it. The last
        `block_size` first-dimension entries are the scratch block: writes
        may land there, but no mask ever makes them visible."""
        return self.k_layers[i], self.v_layers[i]
