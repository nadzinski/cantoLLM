"""Step-4 pins: batched RoPE, the 3D mask, and the forward_batched plumbing.

Everything here is green in step 4 — it tests the geometry and wiring around
the attention math, not the math itself (that's step 5's equivalence suite,
tests/test_padded_equivalence.py).
"""

import pytest
import torch

from cantollm.engine.batching import BatchingConfig
from cantollm.kv_pool import PaddedKVPool
from cantollm.models.attention import (
    BatchMeta,
    EinsumAttentionMethod,
    PaddedAttentionMethod,
)
from cantollm.models.qwen3.model import Qwen3
from cantollm.models.rope import (
    apply_rotary_emb,
    apply_rotary_emb_batched,
    precompute_freqs_cis,
)
from cantollm.runtime import ModelRuntime, build_runtime
from tests.tiny_model import TINY_ARCH, tiny_qwen3_spec


def make_meta(row_specs: list[tuple[int, int, int]]) -> BatchMeta:
    """row_specs: [(slot, start_pos, num_new)] -> a consistent BatchMeta."""
    start_pos = torch.tensor([r[1] for r in row_specs])
    num_new = torch.tensor([r[2] for r in row_specs])
    num_new_max = int(num_new.max())
    return BatchMeta(
        rows=list(row_specs),
        slots=torch.tensor([r[0] for r in row_specs]),
        start_pos=start_pos,
        num_new=num_new,
        positions=start_pos[:, None] + torch.arange(num_new_max)[None, :],
        num_new_max=num_new_max,
        max_history_len=int((start_pos + num_new).max()),
    )


class TestBatchedRope:
    """The batched gather must match the scalar-offset slice row for row —
    exactly, not approximately: same complex math, same half-recombination."""

    @pytest.mark.parametrize("starts", [[0, 7], [3, 3], [11, 0]])
    def test_matches_scalar_rope_per_row(self, starts):
        torch.manual_seed(0)
        freqs_cis = precompute_freqs_cis(8, 32)
        x = torch.randn(2, 4, 3, 8)  # K/V shape: (B, S, groups, head_dim)
        positions = torch.tensor(starts)[:, None] + torch.arange(4)[None, :]

        batched = apply_rotary_emb_batched(x, freqs_cis, positions)

        for b, start in enumerate(starts):
            reference = apply_rotary_emb(x[b : b + 1], freqs_cis, offset=start)
            assert torch.equal(batched[b : b + 1], reference), f"row {b}"

    def test_query_shape_with_head_dim(self):
        torch.manual_seed(1)
        freqs_cis = precompute_freqs_cis(8, 32)
        x = torch.randn(2, 3, 4, 2, 8)  # Q shape: (B, S, groups, heads, head_dim)
        positions = torch.tensor([[5, 6, 7], [0, 1, 2]])

        batched = apply_rotary_emb_batched(x, freqs_cis, positions)

        assert batched.shape == x.shape
        assert torch.equal(
            batched[1:2], apply_rotary_emb(x[1:2], freqs_cis, offset=0)
        )

    def test_ragged_rows_ignore_pad_positions(self):
        """A row's real columns rotate the same regardless of what pad
        columns exist beside it in the batch."""
        torch.manual_seed(2)
        freqs_cis = precompute_freqs_cis(8, 32)
        x = torch.randn(2, 5, 3, 8)
        positions = torch.stack(
            [4 + torch.arange(5), torch.zeros(5, dtype=torch.int64)]
        )  # row 1 is a 1-real-token decode row; pad positions clamped to 0

        batched = apply_rotary_emb_batched(x, freqs_cis, positions)

        row1_real = apply_rotary_emb(x[1:2, :1], freqs_cis, offset=0)
        assert torch.equal(batched[1:2, :1], row1_real)


class TestBatchedMask:
    def test_matches_einsum_mask_per_row(self):
        """Each row's real region equals the 2D mask the einsum reference
        builds for the same (start_pos, seq_len)."""
        meta = make_meta([(0, 0, 3), (1, 5, 1)])
        einsum = EinsumAttentionMethod()

        mask = PaddedAttentionMethod().build_batched_mask(meta, torch.device("cpu"))

        assert mask.shape == (2, 3, 6)
        assert mask.dtype == torch.bool
        for b, (_, start, num_new) in enumerate(meta.rows):
            reference = einsum.build_mask(start, num_new, torch.device("cpu"))
            assert torch.equal(mask[b, :num_new, : start + num_new], reference), (
                f"row {b}"
            )

    def test_everything_beyond_a_rows_history_is_masked(self):
        """The stale-slot fence: columns past start_pos + num_new are masked
        for every query row, real or pad."""
        meta = make_meta([(0, 2, 2), (1, 9, 1)])

        mask = PaddedAttentionMethod().build_batched_mask(meta, torch.device("cpu"))

        for b, (_, start, num_new) in enumerate(meta.rows):
            history = start + num_new
            assert mask[b, :, history:].all(), f"row {b} leaks past its history"

    def test_pad_query_rows_can_still_attend(self):
        """Pad rows (i >= num_new[b]) keep at least one unmasked key so the
        softmax stays finite; their outputs are garbage nobody reads."""
        meta = make_meta([(0, 0, 4), (1, 3, 1)])

        mask = PaddedAttentionMethod().build_batched_mask(meta, torch.device("cpu"))

        # Row 1's pad query rows i=1..3 still have unmasked columns.
        assert (~mask[1]).any(dim=-1).all()


class TestForwardBatchedPlumbing:
    class RecordingMethod(PaddedAttentionMethod):
        """Real mask, fake math: returns values broadcast to query shape and
        records every forward_batched call."""

        def __init__(self):
            self.calls = []

        def forward_batched(self, queries, keys, values, mask, layer_k, layer_v, meta):
            self.calls.append(
                {
                    "q_shape": tuple(queries.shape),
                    "mask_shape": tuple(mask.shape),
                    "layer_k_ptr": layer_k.data_ptr(),
                    "layer_v_ptr": layer_v.data_ptr(),
                    "meta": meta,
                }
            )
            heads_per_group = queries.shape[3]
            return values[:, :, :, None, :].expand(-1, -1, -1, heads_per_group, -1)

    def make_pool(self):
        return PaddedKVPool(
            num_layers=TINY_ARCH["num_transformers"], max_batch=3, max_seq_len=32,
            num_groups=TINY_ARCH["num_groups"], head_dim=TINY_ARCH["head_dim"],
            dtype=torch.float32, device=torch.device("cpu"),
        )

    def test_one_call_per_layer_with_that_layers_views(self):
        method = self.RecordingMethod()
        model = Qwen3(qwen3_config=TINY_ARCH, attention_method=method)
        pool = self.make_pool()
        meta = make_meta([(2, 0, 3), (0, 5, 1)])

        logits = model.forward_batched(torch.zeros(2, 3, dtype=torch.int64), meta, pool)

        assert logits.shape == (2, TINY_ARCH["token_count"])
        assert len(method.calls) == TINY_ARCH["num_transformers"]
        for i, call in enumerate(method.calls):
            k_view, v_view = pool.layer(i)
            assert call["layer_k_ptr"] == k_view.data_ptr(), f"layer {i} k view"
            assert call["layer_v_ptr"] == v_view.data_ptr(), f"layer {i} v view"
            assert call["q_shape"] == (
                2, 3, TINY_ARCH["num_groups"],
                TINY_ARCH["num_heads"] // TINY_ARCH["num_groups"],
                TINY_ARCH["head_dim"],
            )
            assert call["mask_shape"] == (2, 3, 6)
            assert call["meta"] is meta  # one meta, shared by every layer

    def test_gather_picks_each_rows_last_real_column(self):
        """With zero transformer blocks the model is embed -> gather -> norm
        -> lm_head, so the gather column is directly observable."""
        arch = dict(TINY_ARCH, num_transformers=0)
        model = Qwen3(qwen3_config=arch, attention_method=PaddedAttentionMethod())
        pool = PaddedKVPool(
            num_layers=0, max_batch=2, max_seq_len=32,
            num_groups=arch["num_groups"], head_dim=arch["head_dim"],
            dtype=torch.float32, device=torch.device("cpu"),
        )
        meta = make_meta([(0, 0, 3), (1, 5, 1)])
        input_ids = torch.tensor([[11, 12, 13], [21, 0, 0]])

        logits = model.forward_batched(input_ids, meta, pool)

        # Row 0's last real token is column 2 (id 13); row 1's is column 0
        # (id 21). Same batch shape as the model's own gather path, so the
        # comparison is exact.
        hidden = model.initial_embedding_layer(torch.tensor([13, 21]))
        expected = model.output_layer(model.output_RMSNorm(hidden))
        assert torch.equal(logits, expected)

    def test_zero_width_row_rejected(self):
        model = Qwen3(qwen3_config=TINY_ARCH, attention_method=PaddedAttentionMethod())
        meta = make_meta([(0, 0, 3), (1, 5, 1)])
        broken = BatchMeta(
            rows=meta.rows, slots=meta.slots, start_pos=meta.start_pos,
            num_new=torch.tensor([3, 0]), positions=meta.positions,
            num_new_max=meta.num_new_max, max_history_len=meta.max_history_len,
        )
        with pytest.raises(ValueError, match="num_new >= 1"):
            model.forward_batched(
                torch.zeros(2, 3, dtype=torch.int64), broken, self.make_pool()
            )

    def test_pool_layer_count_mismatch_rejected(self):
        model = Qwen3(qwen3_config=TINY_ARCH, attention_method=PaddedAttentionMethod())
        wrong_pool = PaddedKVPool(
            num_layers=5, max_batch=2, max_seq_len=32,
            num_groups=TINY_ARCH["num_groups"], head_dim=TINY_ARCH["head_dim"],
            dtype=torch.float32, device=torch.device("cpu"),
        )
        with pytest.raises(ValueError, match="layers"):
            model.forward_batched(
                torch.zeros(1, 1, dtype=torch.int64), make_meta([(0, 0, 1)]), wrong_pool
            )


class TestRuntimeFront:
    def test_forward_batched_delegates_to_model(self):
        class FakeModel:
            def __init__(self):
                self.seen = None

            def forward_batched(self, input_ids, meta, pool):
                self.seen = (input_ids, meta, pool)
                return torch.zeros(input_ids.shape[0], 7)

        fake = FakeModel()
        runtime = ModelRuntime(
            spec=None, device=torch.device("cpu"),
            model=fake, tokenizer=None, backend=None,
        )
        meta = make_meta([(0, 0, 2)])
        pool = object()

        out = runtime.forward_batched(torch.zeros(1, 2, dtype=torch.int64), meta, pool)

        assert out.shape == (1, 7)
        assert fake.seen[1] is meta and fake.seen[2] is pool

    def test_build_runtime_attention_switch(self):
        runtime = build_runtime(
            tiny_qwen3_spec(), torch.device("cpu"), attention="padded"
        )
        assert isinstance(runtime.model.attention_method, PaddedAttentionMethod)

        default = build_runtime(tiny_qwen3_spec(), torch.device("cpu"))
        assert isinstance(default.model.attention_method, EinsumAttentionMethod)

    def test_speculative_padded_combination_rejected(self):
        with pytest.raises(ValueError, match="sequential-only"):
            build_runtime(
                tiny_qwen3_spec(), torch.device("cpu"),
                speculative=tiny_qwen3_spec(), attention="padded",
            )
