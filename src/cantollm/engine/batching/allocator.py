"""Allocators for the KV pools: slots (padded) and blocks (paged).

Scheduler-owned, both of them (decision 1: the runtime owns the pool
memory, the scheduler owns which parts are in use). A slot is an index into
the padded pool's `max_batch` dim; a block is an index into the paged
pool's allocatable range `[0, num_kv_blocks)` — the scratch block sits past
it and never enters a free list. Free lists are deques in ascending order,
so allocation is deterministic and freed entries are reused FIFO — failures
reproduce run-to-run, unlike the prototype's `set.pop()`.
"""

from __future__ import annotations

from collections import deque


class SlotAllocator:
    def __init__(self, max_batch: int):
        self.max_batch = max_batch
        self._free: deque[int] = deque(range(max_batch))

    def allocate(self) -> int | None:
        """Take the next free slot, or None if all are in use."""
        return self._free.popleft() if self._free else None

    def free(self, slot: int) -> None:
        """Return a slot; it goes to the back of the FIFO."""
        if not 0 <= slot < self.max_batch:
            raise ValueError(f"slot {slot} out of range [0, {self.max_batch})")
        if slot in self._free:
            raise ValueError(f"double free of slot {slot}")
        self._free.append(slot)

    def num_free(self) -> int:
        return len(self._free)

    def num_active(self) -> int:
        return self.max_batch - len(self._free)


class BlockAllocator:
    """Free-list + refcount block allocator for the paged KV pool
    (paged-kv-plan.md §2, hand-written per §8 — this stub pins the API so
    `tests/test_block_allocator.py` pre-lands red).

    Refcounts exist for Phase 5 prefix sharing (several sequences holding
    one cached block) and shape the free-path contract now: `allocate()`
    hands out a block at refcount 1; `free()` decrements and only returns
    the block to the free list when the count reaches 0; `incref()` adds a
    holder. Phase 4 traffic never increfs, so every count is 1 and free
    means free — but the contract is already the shared one.

    Determinism mirrors `SlotAllocator`: a fresh allocator hands out
    0, 1, 2, ... ascending; freed blocks are reused FIFO. Misuse fails
    loudly: freeing a free block, increfing a free block, or touching an
    out-of-range id raises ValueError.
    """

    def __init__(self, num_blocks: int):
        raise NotImplementedError("author's chunk-2 session")

    def allocate(self) -> int | None:
        """The next free block at refcount 1, or None when exhausted."""
        raise NotImplementedError("author's chunk-2 session")

    def free(self, block: int) -> None:
        """Drop one hold; at refcount 0 the block joins the FIFO's back."""
        raise NotImplementedError("author's chunk-2 session")

    def incref(self, block: int) -> None:
        """One more holder of an allocated block (Phase 5 prefix sharing)."""
        raise NotImplementedError("author's chunk-2 session")

    def num_free(self) -> int:
        raise NotImplementedError("author's chunk-2 session")

    def num_allocated(self) -> int:
        raise NotImplementedError("author's chunk-2 session")
