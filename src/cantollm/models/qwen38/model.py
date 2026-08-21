"""Qwen 3.8 hybrid decoder (qwen3_5 architecture).

64 layers in a repeating [linear, linear, linear, full] pattern: 48
Gated-DeltaNet linear-attention layers (gdn.py) and 16 gated
full-attention layers, dense SwiGLU everywhere. Deliberately a separate
copy from models/qwen3/model.py (PoC decision); the pluggable attention
methods, the complex-multiply RoPE core, and the engine surface are
shared.

Differences from qwen3 worth knowing when reading side by side:
  - Layer norms (input/post/final and the per-head q/k norms) are
    ZERO-CENTERED: the checkpoint stores weights as offsets from 1 and
    forward multiplies by (1 + weight). Loading those weights into a
    plain RMSNorm silently shifts every scale by 1
    (HF modeling_qwen3_5.py::Qwen3_5RMSNorm).
  - q_proj is double width: per head, the second half is an output gate
    applied as sigmoid(gate) to the attention output before o_proj.
  - RoPE is partial (rotary_dim of head_dim, theta 1e7) via qwen38/rope.py.
  - The sequential cache is a HybridCache (pool.py): per-layer dicts,
    {"keys","values"} on full layers, {"S","conv","pos"} on linear ones.

The constructor keyword is `qwen3_config` because runtime._load_model
passes it by that name for every model family; accepted PoC wart.
"""

from dataclasses import dataclass

import torch
from torch import nn

from cantollm.models.attention import AttentionMethod
from cantollm.models.qwen38.gdn import GatedDeltaNet
from cantollm.models.qwen38.rope import (
    apply_partial_rotary_emb,
    apply_partial_rotary_emb_batched,
    precompute_partial_freqs_cis,
)

LINEAR = "linear_attention"
FULL = "full_attention"


def qwen38_layer_types(num_layers: int, full_attention_interval: int = 4) -> list[str]:
    """The checkpoint's layer pattern: every full_attention_interval-th
    layer is full attention, the rest Gated DeltaNet (full at indices
    3, 7, ..., matching config.json's layer_types)."""
    return [
        FULL if (i + 1) % full_attention_interval == 0 else LINEAR
        for i in range(num_layers)
    ]


class ZeroCenteredRMSNorm(nn.Module):
    """RMSNorm with zero-centered weights: forward scales by (1 + weight)
    and checkpoint weights are stored as offsets from 1 (init zeros).
    Precision order matches HF Qwen3_5RMSNorm: normalize AND scale in
    fp32, then cast once at the end (qwen3's norm casts before scaling)."""

    def __init__(self, dim: int, epsilon: float = 1e-6, dtype=None):
        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.zeros(dim, dtype=dtype))

    def forward(self, x):
        x_f = x.to(torch.float32)
        normed = x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.epsilon)
        return (normed * (1.0 + self.weight.float())).type_as(x)


class FeedForward(nn.Module):
    """Dense SwiGLU, same shape as qwen3's (see that docstring for the
    diagram): linear_1 = gate_proj, linear_2 = up_proj, linear_3 = down_proj."""

    def __init__(self, token_embedding_dim, expanded_dim, dtype):
        super().__init__()
        self.linear_1 = nn.Linear(token_embedding_dim, expanded_dim, bias=False, dtype=dtype)
        self.linear_2 = nn.Linear(token_embedding_dim, expanded_dim, bias=False, dtype=dtype)
        self.linear_3 = nn.Linear(expanded_dim, token_embedding_dim, bias=False, dtype=dtype)

    def forward(self, x):
        return self.linear_3(nn.functional.silu(self.linear_1(x)) * self.linear_2(x))


class GatedAttention(nn.Module):
    """GQA with per-head output gate and partial RoPE.

    W_q emits, per head, [q ; gate] (2 * head_dim); the gate half skips
    norm/RoPE/attention entirely and multiplies the attention output as
    sigmoid(gate) just before out_proj (HF Qwen3_5Attention.forward).
    Everything between projections and the gate is the qwen3 shape
    discipline: (batch, seq, groups, heads_per_group, head_dim) for Q,
    (batch, seq, groups, head_dim) for K/V, attention math delegated to
    the shared method.
    """

    def __init__(
        self,
        token_embedding_dim,
        num_heads,
        num_groups,
        head_dim,
        attention_method: AttentionMethod,
        dtype=None,
    ):
        super().__init__()
        assert num_heads % num_groups == 0
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.heads_per_group = num_heads // num_groups
        self.head_dim = head_dim
        self.kv_dim = num_groups * head_dim
        self.q_out_dim = num_heads * head_dim

        self.W_q = nn.Linear(token_embedding_dim, 2 * self.q_out_dim, bias=False, dtype=dtype)
        self.W_k = nn.Linear(token_embedding_dim, self.kv_dim, bias=False, dtype=dtype)
        self.W_v = nn.Linear(token_embedding_dim, self.kv_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(self.q_out_dim, token_embedding_dim, bias=False, dtype=dtype)

        self.q_norm = ZeroCenteredRMSNorm(head_dim, dtype=dtype)
        self.k_norm = ZeroCenteredRMSNorm(head_dim, dtype=dtype)

        self.attention_method = attention_method

    def _project(self, x):
        """Shared front half: split q/gate, q/k norm. Returns
        (queries, keys, values, gate_flat) with queries/keys NOT yet
        position-rotated (the two paths rotate differently)."""
        batches, seq_len, _ = x.shape

        q_and_gate = self.W_q(x).view(batches, seq_len, self.num_heads, 2 * self.head_dim)
        queries, gate = torch.chunk(q_and_gate, 2, dim=-1)
        gate_flat = gate.reshape(batches, seq_len, self.q_out_dim)

        queries = queries.reshape(
            batches, seq_len, self.num_groups, self.heads_per_group, self.head_dim
        )
        keys = self.W_k(x).view(batches, seq_len, self.num_groups, self.head_dim)
        values = self.W_v(x).view(batches, seq_len, self.num_groups, self.head_dim)

        return self.q_norm(queries), self.k_norm(keys), values, gate_flat

    def forward(self, x, start_pos, mask, freqs_cis, kv_cache=None):
        batches, seq_len, _ = x.shape
        queries, keys, values, gate = self._project(x)

        queries = apply_partial_rotary_emb(queries, freqs_cis, offset=start_pos)
        keys = apply_partial_rotary_emb(keys, freqs_cis, offset=start_pos)

        if kv_cache is None or kv_cache["keys"] is None:
            z_context = self.attention_method.forward_prefill(
                queries, keys, values, mask, kv_cache
            )
        else:
            z_context = self.attention_method.forward_decode(
                queries, keys, values, mask, kv_cache
            )

        z_flat = z_context.reshape(batches, seq_len, self.q_out_dim)
        return self.out_proj(z_flat * torch.sigmoid(gate))

    def forward_batched(self, x, mask, freqs_cis, layer_k, layer_v, meta):
        """Mixed prefill/decode batch: same projections/norm/gate as
        `forward`, per-row RoPE positions, attention via the method's
        pooled path (identical mechanics to qwen3's)."""
        batches, seq_len, _ = x.shape
        queries, keys, values, gate = self._project(x)

        queries = apply_partial_rotary_emb_batched(queries, freqs_cis, meta.positions)
        keys = apply_partial_rotary_emb_batched(keys, freqs_cis, meta.positions)

        z_context = self.attention_method.forward_batched(
            queries, keys, values, mask, layer_k, layer_v, meta
        )
        z_flat = z_context.reshape(batches, seq_len, self.q_out_dim)
        return self.out_proj(z_flat * torch.sigmoid(gate))


@dataclass(frozen=True)
class _BatchedStep:
    """Per-step context shared by every layer of one batched forward:
    the geometry (meta), the pool, the method-opaque mask, and the
    derived tensors the GDN layers need (built once, not per layer)."""

    meta: object
    pool: object
    mask: object
    active_mask: torch.Tensor
    """(B, num_new_max) bool: column < num_new[row]."""
    num_new_dev: torch.Tensor
    """meta.num_new on the compute device (conv-state gather indices)."""
    real_row_idx: torch.Tensor
    """(R,) long, host: rows with num_new > 0; filler rows are excluded
    from every state write-back (they alias slot 0, which may be live)."""
    real_slots: torch.Tensor
    """(R,) long, host: those rows' slots (unique by allocator contract)."""


class Qwen38Block(nn.Module):
    """One decoder layer: pre-norm residual around the mixer (full
    attention or Gated DeltaNet by `kind`), then around the SwiGLU."""

    def __init__(self, kind: str, arch: dict, attention_method: AttentionMethod):
        super().__init__()
        assert kind in (LINEAR, FULL), kind
        self.kind = kind
        dim = arch["token_embedding_dim"]
        dtype = arch["dtype"]
        self.input_norm = ZeroCenteredRMSNorm(dim, dtype=dtype)
        self.post_attention_norm = ZeroCenteredRMSNorm(dim, dtype=dtype)
        if kind == FULL:
            self.attention = GatedAttention(
                dim,
                arch["num_heads"],
                arch["num_groups"],
                arch["head_dim"],
                attention_method,
                dtype=dtype,
            )
        else:
            self.linear_attn = GatedDeltaNet(
                dim,
                arch["linear_num_k_heads"],
                arch["linear_num_v_heads"],
                arch["linear_head_k_dim"],
                arch["linear_head_v_dim"],
                arch["linear_conv_kernel"],
                dtype=dtype,
            )
        self.feed_forward = FeedForward(dim, arch["expanded_dim"], dtype)

    def forward(self, x, start_pos, mask, freqs_cis, layer_cache=None):
        bypass = x
        h = self.input_norm(x)
        if self.kind == FULL:
            h = self.attention(h, start_pos, mask, freqs_cis, kv_cache=layer_cache)
        else:
            h = self._linear_forward(h, layer_cache)
        x = bypass + h

        bypass = x
        x = bypass + self.feed_forward(self.post_attention_norm(x))
        return x

    def _linear_forward(self, h, layer_cache):
        batches = h.shape[0]
        if layer_cache is None or layer_cache["S"] is None:
            s_state, conv_state = self.linear_attn.empty_state(batches, h.device, h.dtype)
        else:
            s_state, conv_state = layer_cache["S"], layer_cache["conv"]

        out, new_s, new_conv = self.linear_attn.forward_core(h, s_state, conv_state)

        if layer_cache is not None:
            layer_cache["S"] = new_s
            layer_cache["conv"] = new_conv
            layer_cache["pos"] = layer_cache.get("pos", 0) + h.shape[1]
        return out

    def forward_batched(self, x, freqs_cis, step: _BatchedStep, layer_idx: int):
        bypass = x
        h = self.input_norm(x)
        if self.kind == FULL:
            layer_k, layer_v = step.pool.layer(layer_idx)
            h = self.attention.forward_batched(
                h, step.mask, freqs_cis, layer_k, layer_v, step.meta
            )
        else:
            s_pool, conv_pool = step.pool.gdn_state(layer_idx)
            # Gather may repeat slot 0 (filler rows); read-only, harmless.
            s_state = s_pool[step.meta.slots]
            conv_state = conv_pool[step.meta.slots]
            h, new_s, new_conv = self.linear_attn.forward_core(
                h,
                s_state,
                conv_state,
                active_mask=step.active_mask,
                num_new=step.num_new_dev,
            )
            # Write back REAL rows only: filler rows alias slot 0, and a
            # scatter with duplicate destinations against a live slot 0
            # would race its real update.
            s_pool[step.real_slots] = new_s[step.real_row_idx]
            conv_pool[step.real_slots] = new_conv[step.real_row_idx]
        x = bypass + h

        return x + self.feed_forward(self.post_attention_norm(x))


class Qwen38(nn.Module):
    def __init__(self, qwen3_config, attention_method: AttentionMethod):
        super().__init__()
        arch = qwen3_config
        self.attention_method = attention_method
        self.layer_types = list(arch["layer_types"])
        assert len(self.layer_types) == arch["num_transformers"]

        self.initial_embedding_layer = nn.Embedding(
            arch["token_count"], arch["token_embedding_dim"], dtype=arch["dtype"]
        )
        self.transformer_blocks = nn.ModuleList(
            [Qwen38Block(kind, arch, attention_method) for kind in self.layer_types]
        )
        self.output_RMSNorm = ZeroCenteredRMSNorm(
            arch["token_embedding_dim"], dtype=arch["dtype"]
        )
        # Untied for the whole family (the 27B checkpoint ships lm_head).
        self.output_layer = nn.Linear(
            arch["token_embedding_dim"],
            arch["token_count"],
            bias=False,
            dtype=arch["dtype"],
        )

        # Table over the rotated slice only; rope_theta is config.json's
        # value (1e7 for Qwen 3.8, vs 1e6 for Qwen3).
        freqs_cis = precompute_partial_freqs_cis(
            arch["rotary_dim"], arch["max_seq_len"], theta=arch["rope_theta"]
        )
        self.register_buffer("freqs_cis", freqs_cis)

    def _validate_cache(self, start_pos, kv_cache):
        if kv_cache is not None and len(kv_cache) != len(self.transformer_blocks):
            raise ValueError(
                f"kv_cache has {len(kv_cache)} entries but model has "
                f"{len(self.transformer_blocks)} blocks"
            )
        if start_pos == 0:
            return
        if kv_cache is None:
            raise ValueError(f"start_pos={start_pos} but no kv_cache provided")
        for i, (kind, cache) in enumerate(zip(self.layer_types, kv_cache)):
            if kind == FULL:
                if cache["keys"] is None:
                    raise ValueError(f"start_pos={start_pos} but kv_cache[{i}] is empty")
                cached_len = cache["keys"].shape[1]
                if cached_len != start_pos:
                    raise ValueError(
                        f"start_pos={start_pos} but kv_cache[{i}] has "
                        f"{cached_len} positions"
                    )
            else:
                if cache["S"] is None:
                    raise ValueError(
                        f"start_pos={start_pos} but kv_cache[{i}] has no GDN state"
                    )
                if cache["pos"] != start_pos:
                    raise ValueError(
                        f"start_pos={start_pos} but kv_cache[{i}] GDN state is at "
                        f"position {cache['pos']}; the scan cannot replay or skip"
                    )

    def forward_batched(self, input_ids, meta, pool):
        """Mixed prefill/decode step for the continuous-batching engine;
        same contract as Qwen3.forward_batched (see that docstring for
        the last-token-gather and validation rationale). `pool` is a
        HybridStatePool: KV tensors on full layers, GDN S/conv state on
        linear ones."""
        self._validate_batched(meta, pool)
        with self.attention_method.execution_context():
            return self.forward_batched_impl(input_ids, meta, pool)

    def forward_batched_impl(self, input_ids, meta, pool):
        x = self.initial_embedding_layer(input_ids)

        device = input_ids.device
        mask = self.attention_method.build_batched_mask(meta, device)
        num_new_dev = meta.num_new.to(device)
        active_mask = (
            torch.arange(meta.num_new_max, device=device)[None, :]
            < num_new_dev[:, None]
        )
        real_row_idx = torch.nonzero(meta.num_new > 0).squeeze(-1)
        step = _BatchedStep(
            meta=meta,
            pool=pool,
            mask=mask,
            active_mask=active_mask,
            num_new_dev=num_new_dev,
            real_row_idx=real_row_idx,
            real_slots=meta.slots[real_row_idx],
        )

        for i, block in enumerate(self.transformer_blocks):
            x = block.forward_batched(x, self.freqs_cis, step, i)

        # Same per-row last-real-token gather as qwen3: filler rows wrap
        # to -1, a pad column nobody samples.
        row_idx = torch.arange(x.shape[0], device=x.device)
        last_hidden = x[row_idx, num_new_dev - 1]
        return self.output_layer(self.output_RMSNorm(last_hidden))

    def _validate_batched(self, meta, pool):
        """Host-side per-step validation, PLUS the GDN state bookkeeping:
        `pool.begin_step` resets slots that start a sequence, enforces
        monotone chunk order for the rest, and advances the per-slot
        position counters. State advance lives here because this is the
        single once-per-step host entry the engine contract guarantees
        (the eager path calls it via forward_batched; a compiled path
        would call it directly before the traced impl)."""
        num_blocks = len(self.transformer_blocks)
        if pool.num_layers != num_blocks:
            raise ValueError(
                f"pool has {pool.num_layers} layers but model has {num_blocks} blocks"
            )
        if any(n < 0 for _, _, n in meta.rows):
            raise ValueError("num_new must be >= 0 (0 marks a filler row)")
        derived_history = max((s + n for _, s, n in meta.rows), default=0)
        if meta.max_history_len < derived_history:
            raise ValueError(
                f"meta.max_history_len={meta.max_history_len} but rows imply "
                f"{derived_history}; the mask and KV gather would silently "
                "truncate history"
            )
        if meta.max_history_len > pool.max_seq_len:
            raise ValueError(
                f"meta.max_history_len={meta.max_history_len} exceeds the "
                f"slot capacity ({pool.max_seq_len}); the KV gather would "
                "read out of bounds"
            )
        pool.begin_step(meta.rows)

    def forward(self, tokens, start_pos: int, kv_cache=None):
        self._validate_cache(start_pos, kv_cache)

        x = self.initial_embedding_layer(tokens)

        seq_len = tokens.shape[-1]
        # Built once; only the full-attention layers consume it.
        mask = self.attention_method.build_mask(start_pos, seq_len, tokens.device)

        for i, block in enumerate(self.transformer_blocks):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x = block(x, start_pos, mask, self.freqs_cis, layer_cache=layer_cache)

        return self.output_layer(self.output_RMSNorm(x))
