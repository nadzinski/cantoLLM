"""Einsum attention: the current single-request implementation, factored out.

Scores via `torch.einsum("bighd,bjgd->bghij", ...)` and value aggregate via
`torch.einsum("bghij,bjgd->bighd", ...)`. KV cache is a per-layer dict
`{"keys": Tensor, "values": Tensor}` whose tensors grow via `torch.cat` along
the sequence dim on each step. This is the correctness reference and the
Mac / CPU fallback going forward.
"""

from __future__ import annotations

import torch


class EinsumAttentionMethod:
    def build_mask(
        self,
        start_pos: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        full_seq_len = start_pos + seq_len
        return torch.ones(
            seq_len, full_seq_len, dtype=torch.bool, device=device
        ).triu(diagonal=start_pos + 1)

    def forward_prefill(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        kv_cache: dict | None,
    ) -> torch.Tensor:
        return self._attend(queries, keys, values, mask, kv_cache)

    def forward_decode(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        kv_cache: dict,
    ) -> torch.Tensor:
        return self._attend(queries, keys, values, mask, kv_cache)

    def _attend(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        kv_cache: dict | None,
    ) -> torch.Tensor:
        if kv_cache and kv_cache["keys"] is not None:
            full_keys = torch.cat((kv_cache["keys"], keys), dim=1)
            full_values = torch.cat((kv_cache["values"], values), dim=1)
        else:
            full_keys = keys
            full_values = values

        if kv_cache is not None:
            kv_cache["keys"] = full_keys
            kv_cache["values"] = full_values

        # (batch, seq_i, groups, heads, dim) @ (batch, seq_j, groups, dim)
        #   -> (batch, groups, heads, seq_i, seq_j)
        attn = torch.einsum("bighd,bjgd->bghij", queries, full_keys)

        head_dim = queries.shape[-1]
        masked_attn = attn.masked_fill(mask, -float("inf"))
        attn_weights = torch.softmax(masked_attn / head_dim**0.5, dim=-1)

        # (batch, groups, heads, seq_i, seq_j) @ (batch, seq_j, groups, dim)
        #   -> (batch, seq_i, groups, heads, dim)
        return torch.einsum("bghij,bjgd->bighd", attn_weights, full_values)
