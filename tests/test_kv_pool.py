"""PaddedKVPool + SlotAllocator + ModelRuntime.new_kv_pool (step 3).

Ports the prototype's padded_kv tests onto the multi-layer pool, plus the
two integration-specific pins: layer(i) returns real views (attention will
write through them), and slot reuse is FIFO-deterministic (fable-review's
reproducibility nit — the prototype's set.pop() gave arbitrary order).
"""

import pytest
import torch

from cantollm.engine.batching import BatchingConfig, SlotAllocator
from cantollm.kv_pool import KVPool, PaddedKVPool, PagedKVPool
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
        # 17 positions = 16 logical + the scratch column (graph-replay
        # parking spot for filler-row writes; gathers never read it).
        pool = make_pool()
        assert len(pool.k_layers) == 2 and len(pool.v_layers) == 2
        for k, v in zip(pool.k_layers, pool.v_layers):
            assert k.shape == (3, 17, 4, 8) and v.shape == (3, 17, 4, 8)
            assert k.dtype == torch.float32
            assert torch.all(k == 0) and torch.all(v == 0)
        assert pool.max_batch == 3 and pool.max_seq_len == 16
        assert pool.scratch_pos == 16
        assert pool.device == pool.k_layers[0].device

    def test_layers_are_separate_writable_tensors(self):
        # Per-layer separate tensors (not views of one stacked tensor) is
        # the compile contract: AOTAutograd keeps direct input mutations
        # in place but functionalizes view-of-input mutations into
        # pool-scale copies (torch-compile-design.md §4).
        pool = make_pool()
        k1, v1 = pool.layer(1)
        assert k1 is pool.k_layers[1] and v1 is pool.v_layers[1]
        assert k1.shape == (3, 17, 4, 8)
        k1[2, 5] = 7.0
        v1[0, 0] = -1.0
        # Writes land in the pool storage (no copy) ...
        assert torch.all(pool.k_layers[1][2, 5] == 7.0)
        assert torch.all(pool.v_layers[1][0, 0] == -1.0)
        # ... other layers are untouched, and no two layers share storage.
        assert torch.all(pool.k_layers[0] == 0)
        k0, _ = pool.layer(0)
        assert k0.untyped_storage().data_ptr() != k1.untyped_storage().data_ptr()

    def test_stacked_helpers_are_copies(self):
        pool = make_pool()
        stacked = pool.stacked_k()
        assert stacked.shape == (2, 3, 17, 4, 8)
        stacked[0, 0, 0] = 5.0
        assert torch.all(pool.k_layers[0] == 0), "stacked_k must be a copy"


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

        assert len(pool.k_layers) == TINY_ARCH["num_transformers"]
        assert pool.k_layers[0].shape == (
            2, 33, TINY_ARCH["num_groups"], TINY_ARCH["head_dim"],
        )
        assert pool.k_layers[0].dtype == spec.dtype
        assert pool.device.type == "cpu"

    def test_rejects_capacity_beyond_rope_table(self):
        # TINY_ARCH's RoPE table is max_seq_len=128. A padded decode row can
        # index freqs_cis at (max_seq_len - 1) + (max_tokens_per_step - 1), so
        # a config reaching that far must be rejected up front, not IndexError
        # mid-step.
        spec = tiny_qwen3_spec()
        runtime = ModelRuntime(
            spec=spec, device=torch.device("cpu"),
            model=None, tokenizer=None, backend=None,
        )
        # 124 + 8 - 2 = 130 >= 128
        config = BatchingConfig(max_batch=2, max_seq_len=124, max_tokens_per_step=8)
        with pytest.raises(ValueError, match="RoPE table"):
            runtime.new_kv_pool(config)

    def test_accepts_capacity_at_rope_table_boundary(self):
        spec = tiny_qwen3_spec()
        runtime = ModelRuntime(
            spec=spec, device=torch.device("cpu"),
            model=None, tokenizer=None, backend=None,
        )
        # 121 + 8 - 2 = 127 < 128 — the largest that fits.
        config = BatchingConfig(max_batch=2, max_seq_len=121, max_tokens_per_step=8)
        assert runtime.new_kv_pool(config).max_seq_len == 121

    def test_paged_branch_builds_at_parity_capacity(self):
        # num_kv_blocks unset: parity capacity (max_batch * max_seq_len /
        # block_size = 2 * 32 / 16 = 4 blocks) + the scratch block.
        spec = tiny_qwen3_spec()
        runtime = ModelRuntime(
            spec=spec, device=torch.device("cpu"),
            model=None, tokenizer=None, backend=None,
        )
        config = BatchingConfig(
            max_batch=2, max_seq_len=32, max_tokens_per_step=8, paged_kv=True,
            block_size=16,
        )
        pool = runtime.new_kv_pool(config)
        assert isinstance(pool, PagedKVPool)
        assert pool.num_kv_blocks == 4 and pool.block_size == 16
        assert pool.k_layers[0].shape == (
            5 * 16, TINY_ARCH["num_groups"], TINY_ARCH["head_dim"],
        )
        assert pool.k_layers[0].dtype == spec.dtype

    def test_paged_branch_honors_num_kv_blocks_and_rope_guard(self):
        runtime = ModelRuntime(
            spec=tiny_qwen3_spec(), device=torch.device("cpu"),
            model=None, tokenizer=None, backend=None,
        )
        pool = runtime.new_kv_pool(BatchingConfig(
            max_batch=2, max_seq_len=32, max_tokens_per_step=8,
            paged_kv=True, block_size=16, num_kv_blocks=3,
        ))
        assert pool.num_kv_blocks == 3
        # The RoPE guard predates the layout branch and covers both:
        # 128 + 8 - 2 = 134 >= 128 must be rejected paged too.
        with pytest.raises(ValueError, match="RoPE table"):
            runtime.new_kv_pool(BatchingConfig(
                max_batch=2, max_seq_len=128, max_tokens_per_step=8,
                paged_kv=True,
            ))


class TestKVPoolProtocol:
    """The structural surface everything outside the attention method is
    typed against (Phase 4 chunk 1) — the paged pool must satisfy exactly
    this to slot in beside PaddedKVPool."""

    def test_padded_pool_satisfies_the_protocol(self):
        assert isinstance(make_pool(), KVPool)

    def test_paged_pool_satisfies_the_protocol(self):
        assert isinstance(make_paged_pool(), KVPool)

    def test_layerless_object_does_not(self):
        class NotAPool:
            num_layers = 1
            max_seq_len = 8
            device = torch.device("cpu")

        assert not isinstance(NotAPool(), KVPool)


def make_paged_pool(**overrides) -> PagedKVPool:
    kwargs = dict(
        num_layers=2, num_kv_blocks=4, block_size=4, max_seq_len=16,
        num_groups=4, head_dim=8, dtype=torch.float32,
        device=torch.device("cpu"),
    )
    kwargs.update(overrides)
    return PagedKVPool(**kwargs)


class TestPagedKVPool:
    def test_flat_shapes_dtype_and_zero_init(self):
        # (4 + 1) * 4 = 20 flat rows per layer: 16 allocatable + the
        # 4-row scratch block at the tail.
        pool = make_paged_pool()
        assert len(pool.k_layers) == 2 and len(pool.v_layers) == 2
        for k, v in zip(pool.k_layers, pool.v_layers):
            assert k.shape == (20, 4, 8) and v.shape == (20, 4, 8)
            assert k.dtype == torch.float32
            assert torch.all(k == 0) and torch.all(v == 0)
        assert pool.num_kv_blocks == 4 and pool.block_size == 4
        assert pool.max_seq_len == 16
        assert pool.scratch_block == 4
        assert pool.device == pool.k_layers[0].device

    def test_layer_returns_the_real_storage(self):
        # Attention writes through layer(i); a copy would silently drop
        # every KV write (the padded pin, ported).
        pool = make_paged_pool()
        k, v = pool.layer(1)
        k[7, 2, 3] = 1.5
        v[19, 0, 0] = -2.0          # scratch-block row: writable too
        assert pool.k_layers[1][7, 2, 3] == 1.5
        assert pool.v_layers[1][19, 0, 0] == -2.0

    def test_block_row_span_arithmetic(self):
        # Block b owns flat rows [b*bs, (b+1)*bs): the layout contract the
        # write-map destination computation (chunk 4) builds on.
        pool = make_paged_pool()
        k, _ = pool.layer(0)
        block, offset = 2, 3
        k[block * pool.block_size + offset] = 7.0
        flat = pool.k_layers[0]
        assert torch.all(flat[2 * 4 + 3] == 7.0)
        assert torch.all(flat[: 2 * 4] == 0) and torch.all(flat[2 * 4 + 4:] == 0)
