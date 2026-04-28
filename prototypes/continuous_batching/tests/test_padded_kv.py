"""Direct unit tests for the user's PaddedKVCache.

These cover allocation bookkeeping; the K/V tensor read/write is exercised
indirectly through the model + reference engine.
"""

import torch

from continuous_batching.padded_kv import PaddedKVCache


def test_init_shapes():
    cache = PaddedKVCache(max_batch=4, max_seq_len=8, dim=16)
    assert cache.k_cache.shape == (4, 8, 16)
    assert cache.v_cache.shape == (4, 8, 16)
    assert torch.equal(cache.k_cache, torch.zeros(4, 8, 16))
    assert torch.equal(cache.v_cache, torch.zeros(4, 8, 16))


def test_allocate_returns_distinct_slots():
    cache = PaddedKVCache(max_batch=4, max_seq_len=8, dim=16)
    slots = [cache.allocate_slot() for _ in range(4)]
    assert all(s is not None for s in slots)
    assert set(slots) == {0, 1, 2, 3}


def test_allocate_beyond_capacity_returns_none():
    cache = PaddedKVCache(max_batch=2, max_seq_len=8, dim=16)
    cache.allocate_slot()
    cache.allocate_slot()
    assert cache.allocate_slot() is None


def test_free_makes_slot_reusable():
    cache = PaddedKVCache(max_batch=2, max_seq_len=8, dim=16)
    a = cache.allocate_slot()
    cache.allocate_slot()
    assert cache.allocate_slot() is None
    cache.free_slot(a)
    reused = cache.allocate_slot()
    assert reused is not None
    assert reused == a


def test_num_free_and_num_active_track_state():
    cache = PaddedKVCache(max_batch=3, max_seq_len=8, dim=16)
    assert cache.num_free() == 3
    assert cache.num_active() == 0

    s0 = cache.allocate_slot()
    s1 = cache.allocate_slot()
    assert cache.num_free() == 1
    assert cache.num_active() == 2

    cache.free_slot(s0)
    assert cache.num_free() == 2
    assert cache.num_active() == 1

    cache.free_slot(s1)
    assert cache.num_free() == 3
    assert cache.num_active() == 0
