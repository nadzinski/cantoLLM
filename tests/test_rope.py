"""
Tests for Rotary Positional Embeddings (RoPE).

These tests verify the mathematical properties of RoPE without assuming
a specific implementation strategy (complex multiplication vs rotation
matrix). They test the public contract:
  - precompute_freqs_cis(dim, max_seq_len, theta) -> opaque freq tensor
  - apply_rotary_emb(x, freqs_cis, offset) -> rotated tensor
"""

import torch
import pytest

from qwen3.rope import apply_rotary_emb, precompute_freqs_cis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rotate(x, dim, max_seq_len, offset=0, theta=100000.0):
    """Shorthand: precompute + apply."""
    freqs = precompute_freqs_cis(dim, max_seq_len, theta=theta)
    return apply_rotary_emb(x, freqs, offset=offset)


# ---------------------------------------------------------------------------
# Shape and dtype preservation
# ---------------------------------------------------------------------------

class TestShapeAndDtype:
    """RoPE must preserve shape and dtype of its input."""

    def test_basic_shape(self):
        x = torch.randn(2, 16, 8, 64)  # batch, seq, heads, dim
        out = _rotate(x, dim=64, max_seq_len=128)
        assert out.shape == x.shape

    def test_single_token(self):
        x = torch.randn(1, 1, 8, 64)
        out = _rotate(x, dim=64, max_seq_len=128)
        assert out.shape == x.shape

    @pytest.mark.parametrize("head_dim", [32, 64, 128])
    def test_various_head_dims(self, head_dim):
        x = torch.randn(1, 4, 8, head_dim)
        out = _rotate(x, dim=head_dim, max_seq_len=64)
        assert out.shape == x.shape

    def test_extra_middle_dimensions(self):
        """Should work with (batch, seq, groups, heads, dim) layout."""
        x = torch.randn(2, 8, 4, 4, 64)  # groups=4, heads_per_group=4
        out = _rotate(x, dim=64, max_seq_len=64)
        assert out.shape == x.shape

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_dtype_preserved(self, dtype):
        x = torch.randn(1, 4, 8, 64, dtype=dtype)
        out = _rotate(x, dim=64, max_seq_len=64)
        assert out.dtype == dtype


# ---------------------------------------------------------------------------
# Norm preservation (rotation is an isometry)
# ---------------------------------------------------------------------------

class TestNormPreservation:
    """Rotation preserves the L2 norm of each vector."""

    def test_norm_preserved(self):
        x = torch.randn(2, 16, 8, 64)
        out = _rotate(x, dim=64, max_seq_len=128)

        # Norm along head_dim for every (batch, seq, head)
        norm_before = torch.norm(x.float(), dim=-1)
        norm_after = torch.norm(out.float(), dim=-1)
        assert torch.allclose(norm_before, norm_after, atol=1e-5)

    def test_norm_preserved_with_offset(self):
        x = torch.randn(1, 1, 8, 64)
        out = _rotate(x, dim=64, max_seq_len=256, offset=200)

        norm_before = torch.norm(x.float(), dim=-1)
        norm_after = torch.norm(out.float(), dim=-1)
        assert torch.allclose(norm_before, norm_after, atol=1e-5)

    def test_norm_preserved_large_values(self):
        """Norm preservation should hold regardless of input magnitude."""
        x = torch.randn(1, 4, 4, 64) * 1000.0
        out = _rotate(x, dim=64, max_seq_len=64)

        norm_before = torch.norm(x.float(), dim=-1)
        norm_after = torch.norm(out.float(), dim=-1)
        assert torch.allclose(norm_before, norm_after, atol=1e-1)  # larger atol for larger values


# ---------------------------------------------------------------------------
# Determinism and basic rotation properties
# ---------------------------------------------------------------------------

class TestRotationProperties:
    """Basic properties that any correct rotation must satisfy."""

    def test_deterministic(self):
        x = torch.randn(1, 4, 8, 64)
        out1 = _rotate(x, dim=64, max_seq_len=64)
        out2 = _rotate(x, dim=64, max_seq_len=64)
        assert torch.equal(out1, out2)

    def test_position_zero_is_identity(self):
        """At position 0, all rotation angles are 0, so output == input."""
        x = torch.randn(1, 1, 8, 64)
        out = _rotate(x, dim=64, max_seq_len=64, offset=0)
        assert torch.allclose(out.float(), x.float(), atol=1e-6)

    def test_different_positions_give_different_outputs(self):
        """The same vector at different positions should get different embeddings."""
        x = torch.randn(1, 1, 4, 64)
        out_pos0 = _rotate(x, dim=64, max_seq_len=128, offset=0)
        out_pos10 = _rotate(x, dim=64, max_seq_len=128, offset=10)
        out_pos50 = _rotate(x, dim=64, max_seq_len=128, offset=50)

        assert not torch.allclose(out_pos0, out_pos10, atol=1e-4)
        assert not torch.allclose(out_pos0, out_pos50, atol=1e-4)
        assert not torch.allclose(out_pos10, out_pos50, atol=1e-4)

    def test_batch_independence(self):
        """Rotating each batch element independently should give the same result."""
        x = torch.randn(4, 8, 4, 64)
        out_batched = _rotate(x, dim=64, max_seq_len=64)

        for b in range(4):
            out_single = _rotate(x[b : b + 1], dim=64, max_seq_len=64)
            assert torch.allclose(out_batched[b : b + 1], out_single, atol=1e-6)

    def test_different_theta_gives_different_results(self):
        """Different theta values produce different rotation speeds."""
        x = torch.randn(1, 8, 4, 64)
        out_default = _rotate(x, dim=64, max_seq_len=64, theta=10000.0)
        out_large = _rotate(x, dim=64, max_seq_len=64, theta=100000.0)
        assert not torch.allclose(out_default, out_large, atol=1e-4)


# ---------------------------------------------------------------------------
# Offset / incremental equivalence (critical for KV cache)
# ---------------------------------------------------------------------------

class TestOffsetEquivalence:
    """
    The offset mechanism must produce identical results to full-sequence
    application. This is the foundation of KV cache correctness.
    """

    def test_offset_matches_full_sequence(self):
        """
        Rotating token at seq position k with offset=0 in a length-k+1 sequence
        should match rotating it as a single token with offset=k.
        """
        head_dim = 64
        freqs = precompute_freqs_cis(head_dim, max_seq_len=128)
        x_full = torch.randn(1, 20, 4, head_dim)

        # Full sequence rotation
        out_full = apply_rotary_emb(x_full, freqs, offset=0)

        # Token-by-token with offsets
        for pos in range(20):
            x_single = x_full[:, pos : pos + 1, :, :]
            out_single = apply_rotary_emb(x_single, freqs, offset=pos)
            assert torch.allclose(out_full[:, pos : pos + 1], out_single, atol=1e-6), (
                f"Mismatch at position {pos}"
            )

    def test_chunked_matches_full(self):
        """Rotating in chunks with appropriate offsets should match full-sequence."""
        head_dim = 64
        freqs = precompute_freqs_cis(head_dim, max_seq_len=128)
        x = torch.randn(1, 12, 4, head_dim)

        out_full = apply_rotary_emb(x, freqs, offset=0)

        # Chunk into 3 pieces of 4
        chunks = []
        for i in range(3):
            chunk = x[:, i * 4 : (i + 1) * 4, :, :]
            chunks.append(apply_rotary_emb(chunk, freqs, offset=i * 4))

        out_chunked = torch.cat(chunks, dim=1)
        assert torch.allclose(out_full, out_chunked, atol=1e-6)

    def test_prefill_then_decode(self):
        """
        Simulate real inference: prefill a prompt, then decode one token at
        a time. Each decode token should match the equivalent position in a
        full-sequence rotation.
        """
        head_dim = 64
        prompt_len = 10
        decode_len = 5
        total_len = prompt_len + decode_len

        freqs = precompute_freqs_cis(head_dim, max_seq_len=128)
        x = torch.randn(1, total_len, 4, head_dim)

        out_full = apply_rotary_emb(x, freqs, offset=0)

        # Prefill
        prompt = x[:, :prompt_len, :, :]
        out_prefill = apply_rotary_emb(prompt, freqs, offset=0)
        assert torch.allclose(out_full[:, :prompt_len], out_prefill, atol=1e-6)

        # Decode tokens one by one
        for i in range(decode_len):
            pos = prompt_len + i
            token = x[:, pos : pos + 1, :, :]
            out_token = apply_rotary_emb(token, freqs, offset=pos)
            assert torch.allclose(out_full[:, pos : pos + 1], out_token, atol=1e-6), (
                f"Decode mismatch at position {pos}"
            )


# ---------------------------------------------------------------------------
# The key RoPE property: relative position encoding
# ---------------------------------------------------------------------------

class TestRelativePositionProperty:
    """
    THE fundamental property of RoPE: the dot product between two rotated
    vectors depends only on their relative position, not their absolute
    positions. This is what makes RoPE a *relative* position encoding.

    If q is at position m and k is at position n, then:
        <RoPE(q, m), RoPE(k, n)> depends only on (m - n)
    """

    def test_dot_product_depends_on_relative_position(self):
        """
        Shift both q and k by the same amount — the dot product should
        not change.
        """
        head_dim = 64
        freqs = precompute_freqs_cis(head_dim, max_seq_len=256)

        q = torch.randn(1, 1, 1, head_dim)
        k = torch.randn(1, 1, 1, head_dim)

        # Place q at position m, k at position n (relative distance = 5)
        shifts = [0, 10, 50, 100]
        dots = []
        for shift in shifts:
            q_rot = apply_rotary_emb(q, freqs, offset=shift)
            k_rot = apply_rotary_emb(k, freqs, offset=shift + 5)
            dot = torch.sum(q_rot.float() * k_rot.float())
            dots.append(dot)

        # All dot products should be equal
        for i in range(1, len(dots)):
            assert torch.allclose(dots[0], dots[i], atol=1e-4), (
                f"Dot product changed when shifting by {shifts[i]}: "
                f"{dots[0].item():.6f} vs {dots[i].item():.6f}"
            )

    def test_different_relative_distances_give_different_dots(self):
        """Different relative positions should (generally) produce different dots."""
        head_dim = 64
        freqs = precompute_freqs_cis(head_dim, max_seq_len=256)

        q = torch.randn(1, 1, 1, head_dim)
        k = torch.randn(1, 1, 1, head_dim)

        q0 = apply_rotary_emb(q, freqs, offset=0)
        k1 = apply_rotary_emb(k, freqs, offset=1)
        k10 = apply_rotary_emb(k, freqs, offset=10)
        k50 = apply_rotary_emb(k, freqs, offset=50)

        dot_1 = torch.sum(q0.float() * k1.float())
        dot_10 = torch.sum(q0.float() * k10.float())
        dot_50 = torch.sum(q0.float() * k50.float())

        # These should generally differ (not a guarantee for arbitrary vectors,
        # but with 64 dimensions it's astronomically unlikely they'd be equal)
        assert not torch.allclose(dot_1, dot_10, atol=1e-3)
        assert not torch.allclose(dot_1, dot_50, atol=1e-3)


# ---------------------------------------------------------------------------
# Frequency structure: dimensions rotate at different speeds
# ---------------------------------------------------------------------------

class TestFrequencyStructure:
    """
    Lower-indexed dimension pairs should rotate faster (higher frequency)
    than higher-indexed pairs. This gives the model both fine-grained and
    coarse position sensitivity.
    """

    def test_lower_dims_rotate_faster(self):
        """
        Measure how much each dimension pair changes between position 0
        and position 1. Lower pairs should change more.
        """
        head_dim = 64
        half = head_dim // 2
        freqs = precompute_freqs_cis(head_dim, max_seq_len=64)

        # Use a known vector so we can measure per-pair rotation
        x = torch.ones(1, 1, 1, head_dim)
        out_0 = apply_rotary_emb(x, freqs, offset=0).float().squeeze()
        out_1 = apply_rotary_emb(x, freqs, offset=1).float().squeeze()

        # Measure change in each pair: pair i uses elements i and i+half
        changes = []
        for i in range(half):
            d0 = torch.tensor([out_0[i], out_0[i + half]])
            d1 = torch.tensor([out_1[i], out_1[i + half]])
            change = torch.norm(d1 - d0).item()
            changes.append(change)

        # Changes should be monotonically non-increasing (within float tolerance)
        for i in range(len(changes) - 1):
            assert changes[i] >= changes[i + 1] - 1e-6, (
                f"Pair {i} changed less ({changes[i]:.6f}) than pair {i+1} "
                f"({changes[i+1]:.6f}) — lower dimensions should rotate faster"
            )

    def test_highest_freq_dimension_rotates_most_in_one_step(self):
        """The first dimension pair should have the largest angle per step."""
        head_dim = 64
        half = head_dim // 2
        freqs = precompute_freqs_cis(head_dim, max_seq_len=64)

        # Use a unit vector so we can measure angle via atan2
        x = torch.zeros(1, 1, 1, head_dim)
        x[..., :half] = 1.0  # real parts = 1, imag parts = 0

        out_1 = apply_rotary_emb(x, freqs, offset=1).float().squeeze()

        # Measure the rotation angle of each pair using atan2
        angles = []
        for i in range(half):
            angle = torch.atan2(out_1[i + half], out_1[i]).abs().item()
            angles.append(angle)

        assert angles[0] == max(angles), "First pair should rotate the most"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_max_seq_len_boundary(self):
        """Should work right up to max_seq_len."""
        head_dim = 64
        max_seq_len = 32
        x = torch.randn(1, max_seq_len, 4, head_dim)
        out = _rotate(x, dim=head_dim, max_seq_len=max_seq_len)
        assert out.shape == x.shape

    def test_single_head(self):
        x = torch.randn(1, 4, 1, 64)
        out = _rotate(x, dim=64, max_seq_len=64)
        assert out.shape == x.shape

    def test_head_dim_2(self):
        """Minimum meaningful head_dim (one pair of elements)."""
        x = torch.randn(1, 4, 4, 2)
        out = _rotate(x, dim=2, max_seq_len=64)
        assert out.shape == x.shape
        # Should still preserve norm
        assert torch.allclose(
            torch.norm(x.float(), dim=-1),
            torch.norm(out.float(), dim=-1),
            atol=1e-5,
        )

    def test_zero_input(self):
        """Zero vectors should stay zero (rotation of origin is origin)."""
        x = torch.zeros(1, 4, 4, 64)
        out = _rotate(x, dim=64, max_seq_len=64, offset=5)
        assert torch.allclose(out, x, atol=1e-7)
