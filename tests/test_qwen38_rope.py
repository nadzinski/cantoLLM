"""Tests for Qwen 3.8 partial RoPE (models/qwen38/rope.py).

The rotation math itself is the shared models/rope.py implementation,
already covered by test_rope.py. What's under test here is the partial
split: the rotated slice must behave exactly like full RoPE applied to
that slice, the pass-through slice must come back bit-identical, and
the scalar-offset and per-row-positions variants must agree (the
sequential and batched paths share one freqs table).
"""

import pytest
import torch

from cantollm.models.qwen38.rope import (
    apply_partial_rotary_emb,
    apply_partial_rotary_emb_batched,
    precompute_partial_freqs_cis,
)
from cantollm.models.rope import apply_rotary_emb, precompute_freqs_cis

HEAD_DIM = 16
ROTARY_DIM = 4  # matches the tiny-model geometry; real 27B is 64 of 256
THETA = 10_000_000.0


@pytest.fixture
def freqs():
    return precompute_partial_freqs_cis(ROTARY_DIM, max_seq_len=64, theta=THETA)


class TestPartialSplit:
    def test_pass_through_slice_bit_identical(self, freqs):
        x = torch.randn(2, 8, 4, HEAD_DIM)
        out = apply_partial_rotary_emb(x, freqs, offset=3)
        assert torch.equal(out[..., ROTARY_DIM:], x[..., ROTARY_DIM:])

    def test_rotated_slice_matches_full_rope_on_slice(self, freqs):
        x = torch.randn(2, 8, 4, HEAD_DIM)
        out = apply_partial_rotary_emb(x, freqs, offset=5)
        expected = apply_rotary_emb(x[..., :ROTARY_DIM], freqs, offset=5)
        assert torch.allclose(out[..., :ROTARY_DIM], expected, atol=1e-6)

    def test_shape_and_dtype_preserved(self, freqs):
        x = torch.randn(1, 4, 2, 3, HEAD_DIM, dtype=torch.bfloat16)
        out = apply_partial_rotary_emb(x, freqs)
        assert out.shape == x.shape
        assert out.dtype == x.dtype

    def test_position_zero_is_identity(self, freqs):
        x = torch.randn(1, 1, 4, HEAD_DIM)
        out = apply_partial_rotary_emb(x, freqs, offset=0)
        assert torch.allclose(out, x, atol=1e-6)

    def test_positions_change_only_the_rotated_slice(self, freqs):
        x = torch.randn(1, 1, 4, HEAD_DIM)
        out_a = apply_partial_rotary_emb(x, freqs, offset=0)
        out_b = apply_partial_rotary_emb(x, freqs, offset=17)
        assert not torch.allclose(out_a[..., :ROTARY_DIM], out_b[..., :ROTARY_DIM], atol=1e-4)
        assert torch.equal(out_a[..., ROTARY_DIM:], out_b[..., ROTARY_DIM:])


class TestFullCoverageReduction:
    def test_rotary_dim_equal_to_head_dim_is_ordinary_rope(self):
        freqs = precompute_partial_freqs_cis(HEAD_DIM, max_seq_len=64, theta=THETA)
        reference = precompute_freqs_cis(HEAD_DIM, max_seq_len=64, theta=THETA)
        x = torch.randn(2, 6, 4, HEAD_DIM)
        out = apply_partial_rotary_emb(x, freqs, offset=2)
        expected = apply_rotary_emb(x, reference, offset=2)
        assert torch.equal(out, expected)


class TestBatchedAgreement:
    def test_batched_matches_scalar_offset_per_row(self, freqs):
        """Rows at different sequence positions in one batched call must
        match individual scalar-offset calls row by row."""
        seq_len = 4
        offsets = [0, 7, 23]
        x = torch.randn(len(offsets), seq_len, 4, HEAD_DIM)
        positions = torch.stack(
            [torch.arange(o, o + seq_len) for o in offsets]
        )

        out_batched = apply_partial_rotary_emb_batched(x, freqs, positions)

        for r, offset in enumerate(offsets):
            out_scalar = apply_partial_rotary_emb(x[r : r + 1], freqs, offset=offset)
            assert torch.allclose(out_batched[r : r + 1], out_scalar, atol=1e-6), (
                f"row {r} (offset {offset}) diverged from the scalar path"
            )

    def test_batched_pass_through_bit_identical(self, freqs):
        x = torch.randn(2, 4, 4, HEAD_DIM)
        positions = torch.tensor([[0, 1, 2, 3], [10, 11, 12, 13]])
        out = apply_partial_rotary_emb_batched(x, freqs, positions)
        assert torch.equal(out[..., ROTARY_DIM:], x[..., ROTARY_DIM:])
