"""Gated DeltaNet: the linear-attention mixer of Qwen 3.8 (qwen3_5).

Ported from HF transformers models/qwen3_5/modeling_qwen3_5.py
(Qwen3_5GatedDeltaNet, torch_recurrent_gated_delta_rule, l2norm, the
causal_conv1d helpers, Qwen3_5RMSNormGated). Token-loop recurrent form
only: one small state update per token serves both decode steps and
prefill chunks; the chunked-parallel form used by production kernels is
deliberately not implemented (PoC decision).

Per token t, per value head, all in fp32 (mamba_ssm_dtype):

    S    <- S * exp(g_t)                          decay (g_t <= 0)
    mem  <- k_t . S                               what S returns for k_t
    S    <- S + k_t outer ((v_t - mem) * beta_t)  error-correcting write
    out  <- q_t . S

with q and k l2-normalized, q scaled by Dk**-0.5, and

    g_t    = -exp(A_log) * softplus(a_t + dt_bias)   per head
    beta_t = sigmoid(b_t)

State per sequence: S (num_v_heads, Dk, Dv) fp32 plus the causal-conv
window, the last (kernel - 1) input columns.
"""

import torch
import torch.nn.functional as F
from torch import nn


def l2norm(x: torch.Tensor, eps: float = 1e-6):
    """FLA-aligned: x * rsqrt(sum(x^2) + eps), eps inside the sqrt."""
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def causal_conv_step(x: torch.Tensor, conv_state: torch.Tensor, weight: torch.Tensor):
    """Depthwise causal conv over a chunk, carrying kernel-1 columns of state.

    x: (B, C, S) new inputs; conv_state: (B, C, kernel-1) the last
    kernel-1 inputs of earlier chunks (zeros before the first, which
    reproduces the left zero-padding of a from-scratch conv); weight:
    (C, 1, kernel), the nn.Conv1d weight. Returns (out, new_conv_state):
    out is (B, C, S) with silu applied, new_conv_state the last
    kernel-1 columns of cat(state, x).
    """
    combined = torch.cat([conv_state.to(x.dtype), x], dim=-1)
    out = F.conv1d(combined, weight, padding=0, groups=x.shape[1])
    return F.silu(out), combined[..., -conv_state.shape[-1] :]


def causal_conv_step_batched(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    num_new: torch.Tensor,
):
    """Per-row variant for mixed CB steps: rows are left-aligned with
    num_new[r] real columns, the rest padding. The conv output is the
    same as `causal_conv_step` (garbage only reaches columns >= num_new,
    which the scan masks out); the state gather slides each row's window
    by its own num_new: column j of cat(state, x) is state[j] for
    j < kernel-1 and x[j - (kernel-1)] after, so gathering columns
    num_new[r] .. num_new[r]+kernel-2 keeps exactly the last kernel-1
    REAL inputs. A num_new of 0 (filler row) gathers columns 0..kernel-2,
    the old state, unchanged.
    """
    combined = torch.cat([conv_state.to(x.dtype), x], dim=-1)
    out = F.conv1d(combined, weight, padding=0, groups=x.shape[1])
    window = conv_state.shape[-1]
    idx = num_new.to(x.device).view(-1, 1, 1) + torch.arange(
        window, device=x.device
    ).view(1, 1, -1)
    new_state = combined.gather(-1, idx.expand(-1, x.shape[1], -1))
    return F.silu(out), new_state


def gdn_scan(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    active_mask: torch.Tensor | None = None,
):
    """The recurrent gated delta rule, looped over tokens.

    query/key: (B, S, H, Dk), already broadcast to H = num_v_heads;
    value: (B, S, H, Dv); g, beta: (B, S, H); state: (B, H, Dk, Dv),
    kept in fp32. active_mask: optional (B, S) bool; inactive positions
    (padding in a mixed CB step) leave the state untouched and emit
    zeros, which the last-token gather never reads.

    Returns (out (B, S, H, Dv) in query's dtype, final state fp32).
    """
    initial_dtype = query.dtype
    query = l2norm(query.float())
    key = l2norm(key.float())
    value = value.float()
    g = g.float()
    beta = beta.float()
    state = state.float()

    query = query * (query.shape[-1] ** -0.5)

    outputs = []
    for t in range(query.shape[1]):
        q_t = query[:, t]  # (B, H, Dk)
        k_t = key[:, t]
        v_t = value[:, t]  # (B, H, Dv)
        decay_t = g[:, t].exp().unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
        beta_t = beta[:, t].unsqueeze(-1)  # (B, H, 1)

        decayed = state * decay_t
        mem = (decayed * k_t.unsqueeze(-1)).sum(dim=-2)  # (B, H, Dv)
        delta = (v_t - mem) * beta_t
        new_state = decayed + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out_t = (new_state * q_t.unsqueeze(-1)).sum(dim=-2)  # (B, H, Dv)

        if active_mask is not None:
            act = active_mask[:, t].view(-1, 1, 1, 1)
            state = torch.where(act, new_state, state)
            out_t = out_t * act.view(-1, 1, 1)
        else:
            state = new_state
        outputs.append(out_t)

    out = torch.stack(outputs, dim=1)
    return out.to(initial_dtype), state


class GatedRMSNorm(nn.Module):
    """RMSNorm-then-silu-gate over the value-head dim.

    Matches HF Qwen3_5RMSNormGated exactly, including precision order:
    normalize in fp32, cast back and scale by weight, then multiply by
    silu(gate) computed in fp32. The weight here is an ordinary scale
    (init ones), NOT zero-centered like the layer norms in model.py.
    """

    def __init__(self, dim: int, eps: float = 1e-6, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    def forward(self, x: torch.Tensor, gate: torch.Tensor):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = self.weight * x.to(input_dtype)
        x = x * F.silu(gate.to(torch.float32))
        return x.to(input_dtype)


class GatedDeltaNet(nn.Module):
    """The full mixer: projections + causal conv + scan + gated norm.

    Parameter names mirror the checkpoint (linear_attn.{in_proj_qkv,
    in_proj_z, in_proj_b, in_proj_a, conv1d, dt_bias, A_log, norm,
    out_proj}) so weight loading is a straight assignment.

    The module is stateless about sequences: callers own S/conv state
    and pass it through `forward_core`, which returns the new state
    (the sequential cache and the CB pool store it differently).
    """

    def __init__(
        self,
        token_embedding_dim: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_kernel_size: int,
        dtype=None,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert num_v_heads % num_k_heads == 0
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = num_k_heads * head_k_dim
        self.value_dim = num_v_heads * head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.conv_kernel_size = conv_kernel_size

        self.in_proj_qkv = nn.Linear(
            token_embedding_dim, self.conv_dim, bias=False, dtype=dtype
        )
        self.in_proj_z = nn.Linear(
            token_embedding_dim, self.value_dim, bias=False, dtype=dtype
        )
        self.in_proj_b = nn.Linear(
            token_embedding_dim, self.num_v_heads, bias=False, dtype=dtype
        )
        self.in_proj_a = nn.Linear(
            token_embedding_dim, self.num_v_heads, bias=False, dtype=dtype
        )
        # padding=0: the causal left-context comes from the carried conv
        # state, not from Conv1d padding.
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=conv_kernel_size,
            groups=self.conv_dim,
            bias=False,
            padding=0,
            dtype=dtype,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads, dtype=dtype))
        # Same init domain as HF (log of uniform(0.01, 16)); only matters
        # for random-init tiny models, real weights overwrite it.
        self.A_log = nn.Parameter(
            torch.log(torch.empty(self.num_v_heads, dtype=dtype).uniform_(0.01, 16))
        )
        self.norm = GatedRMSNorm(head_v_dim, eps=eps, dtype=dtype)
        self.out_proj = nn.Linear(
            self.value_dim, token_embedding_dim, bias=False, dtype=dtype
        )

    def forward_core(
        self,
        x: torch.Tensor,
        s_state: torch.Tensor,
        conv_state: torch.Tensor,
        active_mask: torch.Tensor | None = None,
        num_new: torch.Tensor | None = None,
    ):
        """x: (B, S, hidden). s_state: (B, H, Dk, Dv) fp32; conv_state:
        (B, conv_dim, kernel-1). num_new: per-row real-column counts for
        the batched conv-state gather; None means every row is full
        width (sequential path). Returns (out (B, S, hidden), new_s,
        new_conv); state tensors are returned, not mutated.
        """
        batches, seq_len, _ = x.shape

        mixed = self.in_proj_qkv(x).transpose(1, 2)  # (B, conv_dim, S)
        if num_new is None:
            mixed, new_conv = causal_conv_step(mixed, conv_state, self.conv1d.weight)
        else:
            mixed, new_conv = causal_conv_step_batched(
                mixed, conv_state, self.conv1d.weight, num_new
            )
        mixed = mixed.transpose(1, 2)

        query, key, value = torch.split(
            mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.view(batches, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.view(batches, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.view(batches, seq_len, self.num_v_heads, self.head_v_dim)

        beta = self.in_proj_b(x).sigmoid()
        # fp32 so exp/softplus cannot saturate in low precision (HF does
        # the same); the scan re-floats everything anyway.
        g = -self.A_log.float().exp() * F.softplus(
            self.in_proj_a(x).float() + self.dt_bias.float()
        )

        rep = self.num_v_heads // self.num_k_heads
        if rep > 1:
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)

        out, new_s = gdn_scan(query, key, value, g, beta, s_state, active_mask)

        z = self.in_proj_z(x).view(batches, seq_len, self.num_v_heads, self.head_v_dim)
        out = self.norm(out, z)
        out = out.reshape(batches, seq_len, self.value_dim)
        return self.out_proj(out), new_s, new_conv

    def empty_state(self, batches: int, device, dtype):
        """Fresh per-sequence state: zero S (fp32 regardless of model
        dtype) and a zero conv window (model dtype)."""
        s = torch.zeros(
            batches,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            dtype=torch.float32,
            device=device,
        )
        conv = torch.zeros(
            batches,
            self.conv_dim,
            self.conv_kernel_size - 1,
            dtype=dtype,
            device=device,
        )
        return s, conv
