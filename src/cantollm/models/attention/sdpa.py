"""SDPA attention: Phase 3's CUDA attend, via `F.scaled_dot_product_attention`.

Subclasses `PaddedAttentionMethod` and overrides only `_attend_batched`: the
KV-pool mechanics (bounds validation, the `kv_write_map` scatter, the slot
gather) stay literally the same code, so an equivalence failure can only mean
the attend differs. The einsum → mask → softmax → einsum block becomes one
fused call — the score tensor never materializes in HBM.

Decided design (2026-07-19, PLAN.md Phase 3): keep the explicit per-row bool
mask from `build_batched_mask`. That routes SDPA to the memory-efficient
backend — flash only applies masks it can compute from index arithmetic and
rejects mask tensors — and that is accepted: still fused, S never exists.
The flash-proper restructure (raggedness as lengths metadata) rides with
Phase 4's paged pool. `bench/probe_sdpa.py` verifies which backend the
dispatcher actually picks on real hardware; don't assume.

Wrinkles the implementation owns (recorded from the design discussion):
  - layout: SDPA wants (B, num_heads, seq, head_dim); our contract is
    (B, num_new_max, groups, heads_per_group, head_dim) in and out.
  - GQA: `enable_gqa=True` accepts K/V at `groups` heads against Q's
    `groups * heads_per_group` — no expand needed.
  - mask convention: ours is True = masked OUT; SDPA's bool mask is
    True = attend. Broadcasts over the head dim at use.
  - scale: SDPA's default is already 1/sqrt(head_dim) — nothing to pass.

The attend is hand-written by the project author (the Phase 2 author-note
rule carries over to attention-adjacent Phase 3 work).
tests/test_sdpa_equivalence.py is the definition of done.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from cantollm.models.attention.padded import PaddedAttentionMethod


class SDPAAttentionMethod(PaddedAttentionMethod):
    def _attend_batched(
        self,
        queries: torch.Tensor,
        full_keys: torch.Tensor,
        full_values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        queries_sdpa = queries.flatten(2, 3).transpose(1, 2)
        keys_sdpa = full_keys.transpose(1, 2)
        values_sdpa = full_values.transpose(1, 2)
        attn_mask = ~mask[:, None, :, :]
        output = F.scaled_dot_product_attention(
            queries_sdpa,
            keys_sdpa,
            values_sdpa,
            attn_mask=attn_mask,
            enable_gqa=True
        )
        return output.transpose(1, 2).unflatten(2, (queries.shape[2], -1))
