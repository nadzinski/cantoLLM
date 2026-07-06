"""PaddedKVPool + SlotAllocator + ModelRuntime.new_kv_pool (step 3).

Ports the prototype's padded_kv tests onto the multi-layer pool, plus the
two integration-specific pins: layer(i) returns real views (attention will
write through them), and slot reuse is FIFO-deterministic (fable-review's
reproducibility nit — the prototype's set.pop() gave arbitrary order).
"""

import pytest
import torch

from cantollm.engine.batching import BatchingConfig, SlotAllocator
from cantollm.kv_pool import PaddedKVPool
from cantollm.runtime import ModelRuntime
from tests.tiny_model import TINY_ARCH, tiny_qwen3_spec


def make_pool(**overrides) -> PaddedKVPool:
    kwargs = dict(
        num_layers=2, max_batch=3, max_seq_len=16, num_groups=4, head_dim=8,
        dtype=torch.float32, device=torch.device("cpu"),
    )
    kwargs.update(overrides)
    return PaddedKVPool(**kwargs)


class TestPaddedKVPool:
    def test_shapes_dtype_and_zero_init(self):
        pool = make_pool()
        assert pool.k.shape == (2, 3, 16, 4, 8)
        assert pool.v.shape == (2, 3, 16, 4, 8)
        assert pool.k.dtype == torch.float32
        assert torch.all(pool.k == 0) and torch.all(pool.v == 0)
        assert pool.max_batch == 3 and pool.max_seq_len == 16

    def test_layer_returns_writable_views(self):
        pool = make_pool()
        k1, v1 = pool.layer(1)
        assert k1.shape == (3, 16, 4, 8)
        k1[2, 5] = 7.0
        v1[0, 0] = -1.0
        # Writes through the view land in the pool storage (no copy) ...
        assert torch.all(pool.k[1, 2, 5] == 7.0)
        assert torch.all(pool.v[1, 0, 0] == -1.0)
        # ... and other layers are untouched.
        assert torch.all(pool.k[0] == 0)


class TestSlotAllocator:
    def test_allocates_distinct_ascending_slots(self):
        alloc = SlotAllocator(3)
        assert [alloc.allocate() for _ in range(3)] == [0, 1, 2]

    def test_exhaustion_returns_none(self):
        alloc = SlotAllocator(2)
        alloc.allocate(), alloc.allocate()
        assert alloc.allocate() is None

    def test_fifo_reuse_is_deterministic(self):
        alloc = SlotAllocator(4)
        for _ in range(4):
            alloc.allocate()
        # Free in scrambled order; reallocation must follow the free order.
        alloc.free(3)
        alloc.free(1)
        alloc.free(2)
        assert [alloc.allocate() for _ in range(3)] == [3, 1, 2]

    def test_bookkeeping(self):
        alloc = SlotAllocator(3)
        assert alloc.num_free() == 3 and alloc.num_active() == 0
        slot = alloc.allocate()
        assert alloc.num_free() == 2 and alloc.num_active() == 1
        alloc.free(slot)
        assert alloc.num_free() == 3 and alloc.num_active() == 0

    def test_double_free_raises(self):
        alloc = SlotAllocator(2)
        slot = alloc.allocate()
        alloc.free(slot)
        with pytest.raises(ValueError, match="double free"):
            alloc.free(slot)

    def test_out_of_range_free_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            SlotAllocator(2).free(5)


class TestRuntimeNewKVPool:
    def test_pool_from_tiny_spec(self):
        spec = tiny_qwen3_spec()
        runtime = ModelRuntime(
            spec=spec, device=torch.device("cpu"),
            model=None, tokenizer=None, backend=None,
        )
        config = BatchingConfig(max_batch=2, max_seq_len=32, max_tokens_per_step=8)

        pool = runtime.new_kv_pool(config)

        assert pool.k.shape == (
            TINY_ARCH["num_transformers"], 2, 32,
            TINY_ARCH["num_groups"], TINY_ARCH["head_dim"],
        )
        assert pool.k.dtype == spec.dtype
        assert pool.k.device.type == "cpu"
