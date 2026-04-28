"""A tiny one-layer attention model for the continuous-batching prototype.

Real PyTorch tensors and real attention math — but stripped down: single
head, no GQA, no RoPE, no normalization, random-init weights. The model
exists to give the scheduler something nontrivial to feed: K/V written to
a padded cache at the right slot/position, mask shaped right, logits
sampled per row.

Per-row processing is a Python loop. Slow, but each row's history length
varies, and the explicit loop keeps the math readable.
"""

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from continuous_batching.padded_kv import PaddedKVCache


class ToyModel(nn.Module):
    def __init__(self, vocab_size: int = 32, dim: int = 16, seed: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim

        gen = torch.Generator().manual_seed(seed)
        self.embedding = nn.Embedding(vocab_size, dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator=gen) * 0.1)

        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(
        self,
        input_ids: torch.Tensor,
        slot_metas: list[tuple[int, int, int]],
        kv_cache: "PaddedKVCache",
    ) -> torch.Tensor:
        """Run one batched forward pass.

        Args:
            input_ids: (batch, num_new_max) int64 tensor. Each row carries
                that row's new tokens left-aligned, padded with 0s.
            slot_metas: per row, (slot_idx, start_pos, num_new). The model
                writes new K/V to kv_cache.{k,v}_cache[slot_idx,
                start_pos:start_pos+num_new] and reads history up to
                start_pos+num_new for attention.
            kv_cache: holds the (max_batch, max_seq_len, dim) K/V tensors.

        Returns:
            logits: (batch, vocab_size) at the last real token of each row.
        """
        batch = input_ids.shape[0]
        assert len(slot_metas) == batch
        device = input_ids.device

        out_logits = torch.zeros(batch, self.vocab_size, device=device)

        for b, (slot_idx, start_pos, num_new) in enumerate(slot_metas):
            assert num_new >= 1, "scheduler must not pass empty rows"

            # import pdb; pdb.set_trace()

            # either 1 new token (decode) or all the tokens (prefill)
            tokens = input_ids[b, :num_new]

            # project into embedding space. Now a tensor of (num_new, dim)
            embeds = self.embedding(tokens)

            # Form Q for new tokens (num_new, dim)
            q = self.q_proj(embeds)

            # Form K, V for new tokens (num_new, dim)
            k_new = self.k_proj(embeds)
            v_new = self.v_proj(embeds)

            # Add to KV cache after previous K, V
            # cache already has full shape (max_seq_len, dim), padded out with zeros
            kv_cache.k_cache[slot_idx, start_pos:start_pos + num_new] = k_new
            kv_cache.v_cache[slot_idx, start_pos:start_pos + num_new] = v_new

            # Slice cache to get full K, V without padding
            history_len = start_pos + num_new
            k_full = kv_cache.k_cache[slot_idx, :history_len]
            v_full = kv_cache.v_cache[slot_idx, :history_len]

            # Form attention scores for new tokens (num_new, history_len)
            scores = q @ k_full.T / math.sqrt(self.dim)

            # Mask out
            q_positions = torch.arange(start_pos, history_len, device=device)
            kv_positions = torch.arange(history_len, device=device)
            causal_mask = q_positions[:, None] >= kv_positions[None, :]
            scores = scores.masked_fill(~causal_mask, float("-inf"))

            # Softmax, weighted sum of V, out projection, take final logits
            weights = torch.softmax(scores, dim=-1)
            attn_out = weights @ v_full
            projected = self.out_proj(attn_out)
            out_logits[b] = self.lm_head(projected[-1])

        return out_logits
