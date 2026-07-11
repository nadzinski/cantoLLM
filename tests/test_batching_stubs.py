"""Step-2 design-pass pins: the batched seams exist, with frozen signatures.

Nothing here tests behavior — steps 3-5 fill the bodies in. What this suite
pins is (a) everything imports and composes, (b) BatchingConfig validation,
(c) each stub raises NotImplementedError rather than silently doing the
wrong thing, and (d) EinsumAttentionMethod stays sequential-only.
"""

import pytest
import torch

from cantollm.engine.batching import BatchedForwardFn, BatchingConfig
from cantollm.models.attention import (
    BatchMeta,
    EinsumAttentionMethod,
    PaddedAttentionMethod,
)
from cantollm.runtime import ModelRuntime


def make_meta() -> BatchMeta:
    """A small, internally consistent BatchMeta: one 3-token prefill chunk
    at start_pos 0 (slot 2) and one decode row at start_pos 5 (slot 0)."""
    rows = [(2, 0, 3), (0, 5, 1)]
    start_pos = torch.tensor([0, 5])
    num_new = torch.tensor([3, 1])
    return BatchMeta(
        rows=rows,
        slots=torch.tensor([2, 0]),
        start_pos=start_pos,
        num_new=num_new,
        positions=start_pos[:, None] + torch.arange(3)[None, :],
        num_new_max=3,
        max_history_len=6,
    )


class TestBatchMeta:
    def test_constructible_and_consistent(self):
        meta = make_meta()
        assert meta.positions.shape == (2, meta.num_new_max)
        assert meta.max_history_len == int((meta.start_pos + meta.num_new).max())
        assert [r[0] for r in meta.rows] == meta.slots.tolist()


class TestBatchingConfig:
    def test_valid(self):
        cfg = BatchingConfig(max_batch=4, max_seq_len=256, max_tokens_per_step=32)
        assert cfg.max_batch == 4

    def test_budget_must_cover_batch(self):
        # The water-fill guarantee: every active row gets >= 1 token per step.
        with pytest.raises(ValueError, match="max_tokens_per_step"):
            BatchingConfig(max_batch=8, max_seq_len=256, max_tokens_per_step=7)

    def test_equal_budget_and_batch_ok(self):
        BatchingConfig(max_batch=8, max_seq_len=256, max_tokens_per_step=8)

    @pytest.mark.parametrize("field", ["max_batch", "max_seq_len"])
    def test_rejects_non_positive(self, field):
        kwargs = {"max_batch": 4, "max_seq_len": 256, "max_tokens_per_step": 32}
        kwargs[field] = 0
        with pytest.raises(ValueError, match=field):
            BatchingConfig(**kwargs)


class TestEinsumIsSequentialOnly:
    def test_build_batched_mask_raises(self):
        with pytest.raises(NotImplementedError, match="sequential-only"):
            EinsumAttentionMethod().build_batched_mask(make_meta(), torch.device("cpu"))

    def test_forward_batched_raises(self):
        with pytest.raises(NotImplementedError, match="sequential-only"):
            EinsumAttentionMethod().forward_batched(
                None, None, None, None, None, None, make_meta()
            )


class TestPaddedIsBatchedOnly:
    def test_sequential_methods_raise(self):
        method = PaddedAttentionMethod()
        with pytest.raises(NotImplementedError, match="batched-only"):
            method.build_mask(0, 4, torch.device("cpu"))
        with pytest.raises(NotImplementedError, match="batched-only"):
            method.forward_prefill(None, None, None, None, None)
        with pytest.raises(NotImplementedError, match="batched-only"):
            method.forward_decode(None, None, None, None, {})

    def test_forward_batched_is_a_stub_for_now(self):
        # The attention math is step 5 (hand-written); the mask landed in step 4.
        method = PaddedAttentionMethod()
        with pytest.raises(NotImplementedError):
            method.forward_batched(None, None, None, None, None, None, make_meta())


class TestModelAndRuntimeStubs:
    def test_runtime_forward_batched_satisfies_the_seam_protocol(self):
        # BatchedForwardFn is runtime-checkable only structurally; pin the
        # signature by reference so a drift breaks loudly here.
        fn: BatchedForwardFn = ModelRuntime(
            spec=None, device=None, model=None, tokenizer=None, backend=None
        ).forward_batched
        assert callable(fn)
