"""FlexAttention paged attend (Phase 4, paged-kv-plan.md §2.1-§2.6).

Batched-only, like the padded method, and deliberately NOT a
`PaddedAttentionMethod` subclass (decision 2.2): the template's shared
mechanics — the 2D (slot, pos) scatter, the `[:max_history_len]` gather,
the dense bool mask — are exactly what paging replaces, so inheritance
would share nothing but bugs.

STUB: `build_batched_mask` and `forward_batched` are the author's chunk-4
session (§8), the mask-side half of the paged index translation — the
phase's educational core (flex-spike-results.md §1). What the session
inherits from the spike and the design note:

  - the mask comes from `BlockMask.from_kv_blocks` over the meta's seeded
    table tensors (`meta.paged_tables`), with `seq_lengths` passed
    EXPLICITLY — omitting it at q_len=1 raises (spike §2);
  - partial last blocks need the mask_mod physical→logical translation:
    `logical = inverse_tables[b, kv_idx // block_size] * block_size +
    kv_idx % block_size`, admitted when `logical <= start_pos[b] + q_idx`
    (unowned blocks carry a past-any-bound sentinel, so they fail this);
  - the KV scatter is one 1-D index write of `meta.paged_tables.write_map`
    into the FLAT layer tensor — never through a view (§4, the ab4f438
    hazard);
  - `flex_attention(..., enable_gqa=True)`; building the BlockMask per
    step is correctness-fine here (eager CPU) — the per-family reuse
    discipline arrives with chunk 6.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch

from cantollm.models.attention.protocol import BatchMeta


class FlexAttentionMethod:
    def __init__(self, block_size: int):
        # The registry constructs methods without arguments today; chunk 7
        # wires a config-closing factory when `--attention flex` lands.
        self.block_size = block_size

    def execution_context(self):
        """No dispatcher state to pin: the cuDNN priority pin is sdpa's
        (paged-kv-plan.md §2.8); Flex dispatches through Inductor."""
        return nullcontext()

    def build_mask(
        self,
        start_pos: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        raise NotImplementedError("FlexAttentionMethod is batched-only")

    def forward_prefill(self, queries, keys, values, mask, kv_cache):
        raise NotImplementedError("FlexAttentionMethod is batched-only")

    def forward_decode(self, queries, keys, values, mask, kv_cache):
        raise NotImplementedError("FlexAttentionMethod is batched-only")

    def build_batched_mask(self, meta: BatchMeta, device: torch.device):
        raise NotImplementedError("author's chunk-4 session")

    def forward_batched(
        self, queries, keys, values, mask, layer_k, layer_v, meta: BatchMeta
    ) -> torch.Tensor:
        raise NotImplementedError("author's chunk-4 session")
