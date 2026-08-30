"""BlockAllocator contract suite (P4 chunk 2).

Pre-landed red for the author's hand-written session (paged-kv-plan.md §8),
the step-5 / scheduler-port pattern from Phase 2; green since 2026-08-29.

Contract: ascending-deque determinism, FIFO reuse, refcount-1 allocation,
free-at-zero, loud misuse (double free, incref of a free block,
out-of-range ids), and the scratch block never entering circulation
because the allocator is simply sized without it.
"""

import pytest
import torch

from cantollm.engine.batching.allocator import BlockAllocator
from cantollm.kv_pool import PagedKVPool


class TestAllocationOrder:
    def test_fresh_allocator_hands_out_ascending(self):
        alloc = BlockAllocator(4)
        assert [alloc.allocate() for _ in range(4)] == [0, 1, 2, 3]

    def test_exhaustion_returns_none(self):
        alloc = BlockAllocator(2)
        alloc.allocate(), alloc.allocate()
        assert alloc.allocate() is None

    def test_freed_blocks_are_reused_fifo(self):
        # Free 2 then 0: the next allocations must come back in exactly
        # that order — reproducible failures, like SlotAllocator.
        alloc = BlockAllocator(3)
        for _ in range(3):
            alloc.allocate()
        alloc.free(2)
        alloc.free(0)
        assert alloc.allocate() == 2
        assert alloc.allocate() == 0

    def test_counts_track_allocation(self):
        alloc = BlockAllocator(3)
        assert (alloc.num_free(), alloc.num_allocated()) == (3, 0)
        block = alloc.allocate()
        assert (alloc.num_free(), alloc.num_allocated()) == (2, 1)
        alloc.free(block)
        assert (alloc.num_free(), alloc.num_allocated()) == (3, 0)


class TestRefcounts:
    def test_allocate_starts_at_one_hold_free_releases(self):
        alloc = BlockAllocator(2)
        block = alloc.allocate()
        alloc.free(block)
        assert alloc.num_free() == 2

    def test_incref_defers_release_until_last_free(self):
        # Two holders: the first free keeps the block allocated, the
        # second releases it (the Phase 5 prefix-sharing contract).
        alloc = BlockAllocator(2)
        block = alloc.allocate()
        alloc.incref(block)
        alloc.free(block)
        assert alloc.num_allocated() == 1
        alloc.free(block)
        assert alloc.num_allocated() == 0
        assert alloc.num_free() == 2

    def test_released_shared_block_rejoins_fifo_once(self):
        # A block freed by both holders must appear in circulation exactly
        # once — a double insert would hand one block to two sequences.
        alloc = BlockAllocator(2)
        block = alloc.allocate()
        other = alloc.allocate()
        alloc.incref(block)
        alloc.free(block)
        alloc.free(block)
        alloc.free(other)
        drained = [alloc.allocate() for _ in range(2)]
        assert sorted(drained) == [block, other]
        assert alloc.allocate() is None


class TestLoudMisuse:
    def test_double_free_raises(self):
        alloc = BlockAllocator(2)
        block = alloc.allocate()
        alloc.free(block)
        with pytest.raises(ValueError, match="free"):
            alloc.free(block)

    def test_incref_of_a_free_block_raises(self):
        # Sharing what nobody holds is a prefix-cache bookkeeping bug.
        alloc = BlockAllocator(2)
        with pytest.raises(ValueError):
            alloc.incref(0)

    def test_out_of_range_ids_raise(self):
        alloc = BlockAllocator(2)
        with pytest.raises(ValueError, match="range"):
            alloc.free(5)
        with pytest.raises(ValueError, match="range"):
            alloc.incref(-1)


class TestScratchBlockExclusion:
    def test_allocator_sized_without_the_scratch_block(self):
        # The scratch block is the pool's LAST block, past num_kv_blocks;
        # an allocator sized to num_kv_blocks can never hand it out, so a
        # filler row's parking spot is never a live sequence's storage.
        pool = PagedKVPool(
            num_layers=1, num_kv_blocks=4, block_size=4, max_seq_len=16,
            num_groups=1, head_dim=8, dtype=torch.float32,
            device=torch.device("cpu"),
        )
        alloc = BlockAllocator(pool.num_kv_blocks)
        drained = {alloc.allocate() for _ in range(pool.num_kv_blocks)}
        assert pool.scratch_block not in drained
        assert alloc.allocate() is None
