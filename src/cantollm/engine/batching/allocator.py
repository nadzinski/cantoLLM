"""Slot allocator for the padded KV pool.

Scheduler-owned (decision 1: the runtime owns the pool memory, the scheduler
owns which slots are in use). A slot is just an index into the pool's
`max_batch` dim. The free list is a deque in ascending order, so allocation
is deterministic and freed slots are reused FIFO — failures reproduce
run-to-run, unlike the prototype's `set.pop()`.
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
