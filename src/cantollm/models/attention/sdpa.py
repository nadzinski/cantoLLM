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

Amended on 5090 validation (2026-07-19, torch 2.10.0+cu128, sm_120): the
memory-efficient kernel rejects dense GQA inputs (it does not honor
`enable_gqa`; flash does but rejects the mask tensor), so the fused backend
that actually takes this call is **cuDNN** — and the default dispatcher
ranks cuDNN below math on this build, silently running the unfused math
path. Hence the explicit `sdpa_kernel` pin below: cuDNN first, math kept as
the CPU/MPS fallback so the attend stays runnable (and the equivalence
suite meaningful) off-CUDA. Same design, corrected routing.

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
from torch.nn.attention import SDPBackend, sdpa_kernel

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
        # cuDNN pinned ahead of math, which stays listed as the CPU/MPS
        # fallback. set_priority is load-bearing: without it sdpa_kernel
        # only restricts the backend *set*, and this build's default
        # priority ranks math above cuDNN — the call would silently run
        # unfused (see the amendment in the module docstring).
        with sdpa_kernel(
            [SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH], set_priority=True
        ):
            output = F.scaled_dot_product_attention(
                queries_sdpa,
                keys_sdpa,
                values_sdpa,
                attn_mask=attn_mask,
                enable_gqa=True
            )
        return output.transpose(1, 2).unflatten(2, (queries.shape[2], -1))
