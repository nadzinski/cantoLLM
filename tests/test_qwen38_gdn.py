"""Tests for the Qwen 3.8 Gated DeltaNet (models/qwen38/gdn.py).

The scan is checked three ways: against an independently written naive
per-head reference, against itself under chunking (state carry must
make chunk boundaries invisible), and under the active mask (padded
positions of a mixed CB step must not touch state). The conv helpers
are checked against a plain left-zero-padded conv, and the module's
forward_core against its own token-by-token replay.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from cantollm.models.qwen38.gdn import (
    GatedDeltaNet,
    GatedRMSNorm,
    causal_conv_step,
    causal_conv_step_batched,
    gdn_scan,
    l2norm,
)

def naive_scan_one(q, k, v, g, beta, s0):
    """Independent reference for ONE (batch, head): explicit matrix ops,
    mirroring the docstring formula literally. q, k: (S, Dk) raw
    (normalization happens here, as in the scan); v: (S, Dv);
    g, beta: (S,); s0: (Dk, Dv)."""
    s_mat = s0.clone().float()
    outs = []
    for t in range(q.shape[0]):
        q_t = q[t].float()
        q_t = q_t * torch.rsqrt((q_t * q_t).sum() + 1e-6)
        q_t = q_t * (q.shape[-1] ** -0.5)
        k_t = k[t].float()
        k_t = k_t * torch.rsqrt((k_t * k_t).sum() + 1e-6)

        s_mat = math.exp(g[t].item()) * s_mat
        mem = s_mat.T @ k_t
        delta = (v[t].float() - mem) * beta[t].float()
        s_mat = s_mat + torch.outer(k_t, delta)
        outs.append(s_mat.T @ q_t)
    return torch.stack(outs), s_mat


def rand_inputs(batches=2, seq_len=5, heads=3, dk=4, dv=6, seed=0):
    # Seeded per call, not via module RNG state: these tests must not
    # depend on suite execution order.
    gen = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=gen)

    q = rand(batches, seq_len, heads, dk)
    k = rand(batches, seq_len, heads, dk)
    v = rand(batches, seq_len, heads, dv)
    g = -torch.rand(batches, seq_len, heads, generator=gen)  # decay exps, <= 0
    beta = torch.rand(batches, seq_len, heads, generator=gen)
    s0 = rand(batches, heads, dk, dv)
    return q, k, v, g, beta, s0


class TestScanAgainstNaiveReference:
    def test_matches_naive_per_head(self):
        q, k, v, g, beta, s0 = rand_inputs()
        out, s_final = gdn_scan(q, k, v, g, beta, s0)
        for b in range(q.shape[0]):
            for h in range(q.shape[2]):
                ref_out, ref_s = naive_scan_one(
                    q[b, :, h], k[b, :, h], v[b, :, h], g[b, :, h], beta[b, :, h], s0[b, h]
                )
                assert torch.allclose(out[b, :, h], ref_out, atol=1e-5), (b, h)
                assert torch.allclose(s_final[b, h], ref_s, atol=1e-5), (b, h)

    def test_single_token_closed_form(self):
        """One token from zero state: out = (1/sqrt(Dk)) * beta * <q_n, k_n> * v,
        with q_n, k_n the l2-normalized vectors (decay of a zero state is
        invisible, mem is zero)."""
        dk, dv = 4, 3
        q = torch.randn(1, 1, 1, dk)
        k = torch.randn(1, 1, 1, dk)
        v = torch.randn(1, 1, 1, dv)
        g = torch.tensor([[[-0.7]]])
        beta = torch.tensor([[[0.4]]])
        s0 = torch.zeros(1, 1, dk, dv)

        out, _ = gdn_scan(q, k, v, g, beta, s0)

        q_n = l2norm(q[0, 0, 0].float()) * dk**-0.5
        k_n = l2norm(k[0, 0, 0].float())
        expected = beta.item() * torch.dot(q_n, k_n) * v[0, 0, 0].float()
        assert torch.allclose(out[0, 0, 0], expected, atol=1e-6)

    def test_query_scale_invariance(self):
        """q and k are l2-normalized inside the scan, so scaling the raw
        inputs must (approximately) not change the output. Approximately:
        the l2norm eps (1e-6) is not scale-free, so shrinking a vector
        shifts its normalization by O(eps / |x|^2); tolerance sized for
        that, not for kernel noise."""
        q, k, v, g, beta, s0 = rand_inputs()
        out_a, _ = gdn_scan(q, k, v, g, beta, s0)
        out_b, _ = gdn_scan(q * 10, k * 0.1, v, g, beta, s0)
        assert torch.allclose(out_a, out_b, atol=1e-3)


class TestScanChunking:
    @pytest.mark.parametrize("widths", [[1, 1, 1, 1, 1, 1], [2, 2, 2], [3, 2, 1], [6]])
    def test_chunked_equals_full(self, widths):
        q, k, v, g, beta, s0 = rand_inputs(seq_len=6)
        out_full, s_full = gdn_scan(q, k, v, g, beta, s0)

        outs, state, pos = [], s0, 0
        for w in widths:
            sl = slice(pos, pos + w)
            out_w, state = gdn_scan(
                q[:, sl], k[:, sl], v[:, sl], g[:, sl], beta[:, sl], state
            )
            outs.append(out_w)
            pos += w
        assert torch.allclose(out_full, torch.cat(outs, dim=1), atol=1e-6)
        assert torch.allclose(s_full, state, atol=1e-6)


class TestScanActiveMask:
    def test_masked_rows_match_per_row_prefixes(self):
        """Rows with num_new < S must produce, on their real prefix, the
        same outputs and final state as an unpadded per-row scan; padded
        positions emit zeros."""
        q, k, v, g, beta, s0 = rand_inputs(batches=3, seq_len=4)
        num_new = torch.tensor([4, 2, 0])
        mask = torch.arange(4).unsqueeze(0) < num_new.unsqueeze(1)

        out, s_final = gdn_scan(q, k, v, g, beta, s0, active_mask=mask)

        for b, n in enumerate(num_new.tolist()):
            if n > 0:
                ref_out, ref_s = gdn_scan(
                    q[b : b + 1, :n], k[b : b + 1, :n], v[b : b + 1, :n],
                    g[b : b + 1, :n], beta[b : b + 1, :n], s0[b : b + 1],
                )
                assert torch.allclose(out[b : b + 1, :n], ref_out, atol=1e-6)
                assert torch.allclose(s_final[b : b + 1], ref_s, atol=1e-6)
            else:
                # Filler row: state untouched, outputs all zero.
                assert torch.equal(s_final[b], s0[b].float())
            assert torch.equal(
                out[b, n:], torch.zeros_like(out[b, n:])
            ), f"padded positions of row {b} leaked non-zero output"


class TestConvHelpers:
    def test_chunked_conv_equals_left_padded_full_conv(self):
        channels, kernel, seq_len = 6, 4, 10
        x = torch.randn(2, channels, seq_len)
        weight = torch.randn(channels, 1, kernel)

        reference = F.silu(
            F.conv1d(x, weight, padding=kernel - 1, groups=channels)[..., :seq_len]
        )

        state = torch.zeros(2, channels, kernel - 1)
        outs = []
        for chunk in [slice(0, 3), slice(3, 4), slice(4, 10)]:
            out, state = causal_conv_step(x[..., chunk], state, weight)
            outs.append(out)
        assert torch.allclose(reference, torch.cat(outs, dim=-1), atol=1e-5)
        assert torch.equal(state, x[..., -(kernel - 1) :])

    @pytest.mark.parametrize("num_new", [0, 1, 2, 3, 5])
    def test_batched_state_gather_matches_sequential(self, num_new):
        """Each row's new conv state must equal what the sequential helper
        produces after consuming only that row's real columns."""
        channels, kernel, width = 6, 4, 5
        x = torch.randn(1, channels, width)
        weight = torch.randn(channels, 1, kernel)
        state = torch.randn(1, channels, kernel - 1)

        out_b, state_b = causal_conv_step_batched(
            x, state, weight, torch.tensor([num_new])
        )

        if num_new == 0:
            assert torch.equal(state_b, state)
        else:
            out_s, state_s = causal_conv_step(x[..., :num_new], state, weight)
            assert torch.equal(state_b, state_s)
            assert torch.allclose(out_b[..., :num_new], out_s, atol=1e-6)


class TestGatedRMSNorm:
    def test_matches_hf_formula(self):
        dim = 8
        norm = GatedRMSNorm(dim)
        with torch.no_grad():
            norm.weight.copy_(torch.randn(dim))
        x = torch.randn(2, 3, dim)
        gate = torch.randn(2, 3, dim)

        out = norm(x, gate)

        xf = x.float()
        expected = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
        expected = norm.weight * expected
        expected = expected * F.silu(gate.float())
        assert torch.allclose(out, expected, atol=1e-6)


def make_gdn():
    return GatedDeltaNet(
        token_embedding_dim=16,
        num_k_heads=2,
        num_v_heads=6,
        head_k_dim=4,
        head_v_dim=4,
        conv_kernel_size=4,
    )


class TestGatedDeltaNetModule:
    def test_parameter_shapes_match_checkpoint_layout(self):
        gdn = make_gdn()
        key_dim, value_dim = 8, 24
        conv_dim = 2 * key_dim + value_dim
        assert gdn.in_proj_qkv.weight.shape == (conv_dim, 16)
        assert gdn.in_proj_z.weight.shape == (value_dim, 16)
        assert gdn.in_proj_b.weight.shape == (6, 16)
        assert gdn.in_proj_a.weight.shape == (6, 16)
        assert gdn.conv1d.weight.shape == (conv_dim, 1, 4)
        assert gdn.dt_bias.shape == (6,)
        assert gdn.A_log.shape == (6,)
        assert gdn.norm.weight.shape == (4,)
        assert gdn.out_proj.weight.shape == (16, value_dim)

    def test_incremental_replay_matches_full_chunk(self):
        gdn = make_gdn()
        x = torch.randn(2, 5, 16)
        s0, conv0 = gdn.empty_state(2, x.device, x.dtype)

        out_full, s_full, conv_full = gdn.forward_core(x, s0, conv0)

        s, conv = gdn.empty_state(2, x.device, x.dtype)
        outs = []
        for t in range(5):
            out_t, s, conv = gdn.forward_core(x[:, t : t + 1], s, conv)
            outs.append(out_t)
        assert torch.allclose(out_full, torch.cat(outs, dim=1), atol=1e-5)
        assert torch.allclose(s_full, s, atol=1e-5)
        assert torch.equal(conv_full, conv)

    def test_batched_num_new_matches_per_row(self):
        """Mixed rows (full chunk / short chunk / filler) against per-row
        sequential runs; garbage in padded columns must not leak."""
        gdn = make_gdn()
        width = 4
        x = torch.randn(3, width, 16)
        num_new = torch.tensor([4, 2, 0])
        mask = torch.arange(width).unsqueeze(0) < num_new.unsqueeze(1)
        s0, conv0 = gdn.empty_state(3, x.device, x.dtype)
        s0 = torch.randn_like(s0)
        conv0 = torch.randn_like(conv0)

        out, s_new, conv_new = gdn.forward_core(
            x, s0, conv0, active_mask=mask, num_new=num_new
        )

        for b, n in enumerate(num_new.tolist()):
            if n == 0:
                assert torch.equal(s_new[b], s0[b])
                assert torch.equal(conv_new[b], conv0[b])
                continue
            out_ref, s_ref, conv_ref = gdn.forward_core(
                x[b : b + 1, :n], s0[b : b + 1], conv0[b : b + 1]
            )
            assert torch.allclose(out[b : b + 1, :n], out_ref, atol=1e-5), b
            assert torch.allclose(s_new[b : b + 1], s_ref, atol=1e-5), b
            assert torch.equal(conv_new[b : b + 1], conv_ref), b
