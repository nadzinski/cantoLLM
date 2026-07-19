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

`forward_batched` is a template: the KV-pool mechanics (validate, ragged
write, gather) are shared by every pooled method, and the attention compute
itself lives in `_attend_batched` — the single method `SDPAAttentionMethod`
overrides. Keeping the mechanics literally the same code means an
equivalence failure between the two can only be the attend.
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
        # ragged KV write — validate every row before mutating the pool, so a
        # bad row can't leave a half-written step behind
        slot_capacity = layer_k.shape[1]
        for r, (slot_idx, start_pos, num_new) in enumerate(meta.rows):
            if start_pos + num_new > slot_capacity:
                raise ValueError(
                    f"row {r} (slot {slot_idx}) would write positions "
                    f"[{start_pos}, {start_pos + num_new}) past slot capacity "
                    f"{slot_capacity}"
                )

        # Optimization: Do the ragged write in one go for all rows using
        # advanced indexing (a mapping in tensors that are created once per
        # step for all layers, and which in CUDA live on-GPU) to avoid the
        # overhead of a CPU->GPU dispatch for each row in each layer
        m = meta.kv_write_map
        layer_k[m.slot, m.pos] = keys[m.row, m.off]
        layer_v[m.slot, m.pos] = values[m.row, m.off]

        # gather (pull the KV block out for active slots only, as far as meta.max_history_len)
        full_keys = layer_k[meta.slots, :meta.max_history_len]
        full_values = layer_v[meta.slots, :meta.max_history_len]

        return self._attend_batched(queries, full_keys, full_values, mask)

    def _attend_batched(
        self,
        queries: torch.Tensor,
        full_keys: torch.Tensor,
        full_values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """The attend: each row's queries against its gathered history.

        The KV-pool mechanics (validate, ragged write, gather) live in
        `forward_batched` and are shared by every pooled method; subclasses
        swap only this compute.

        Shapes:
          queries:     (B, num_new_max, groups, heads_per_group, head_dim)
          full_keys:   (B, max_history_len, groups, head_dim)
          full_values: (B, max_history_len, groups, head_dim)
          mask:        (B, num_new_max, max_history_len) bool, True = masked out
          returns:     (B, num_new_max, groups, heads_per_group, head_dim)
        """
        # Form attn matrices for each head by dot product along the head dim
        # I used to study General Relativity so I find einstein summations
        # easier to think about than complicated transpose sequences and
        # matmuls; it's a cute way to pack in the GQA keys broadcast.
        # (batches, num_new_max, groups, heads, dim) @ (batches, max_history_len, groups, dim)
        #   -> (batches, groups, heads, num_new_max, max_history_len)
        # Thanks to KV caching, at this point the first seq dimension of
        # attn is going to be num_new_max, so we don't form the full attention
        # matrix. Nonetheless, rows narrower than the widest have padding in
        # the query rows, which produce garbage outputs that we throw away.
        # This uses compute to do useless work, but one rectangular kernel
        # beats per-row ragged ones. Decode is memory-bound and so decode-heavy
        # has little waste in wall clock time. The greatest penalty is with one long
        # prefill chunk maxing out num_new_max ~= max_tokens_per_step.
        # We accept the tradeoff and tune max_tokens_per_step accordingly.
        attn = torch.einsum("bighd,bjgd->bghij", queries, full_keys)

        head_dim = queries.shape[-1]
        broadcasted_mask = mask[:, None, None, :, :]
        masked_attn = attn.masked_fill(broadcasted_mask, -float("inf"))
        attn_weights = torch.softmax(masked_attn / head_dim**0.5, dim=-1)

        # Weighted sum of values, *and* rearrange dims to get ready to stitch heads together
        # (batches, groups, heads, num_new_max, max_history_len) @ (batches, max_history_len, groups, dim)
        #   -> (batches, num_new_max, groups, heads, dim)
        return torch.einsum("bghij,bjgd->bighd", attn_weights, full_values)
