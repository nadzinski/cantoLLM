"""Padded KV cache pool. ** YOU FILL THIS IN. **

A pool of (max_batch, max_seq_len, dim) K and V tensors with simple slot
allocation. The model writes to and reads from these tensors directly via
slicing — your job is the bookkeeping: which slot indices are free, which
are in use.

For the prototype, slots are interchangeable. Real engines do the same
thing (with paged KV in vLLM, this becomes block-level instead of slot-level
allocation, but the bookkeeping pattern is identical).
"""

import torch


class PaddedKVCache:
    k_cache: torch.Tensor
    v_cache: torch.Tensor

    def __init__(
        self,
        max_batch: int,
        max_seq_len: int,
        dim: int,
        device: str | torch.device = "cpu",
    ):
        """Preallocate K and V tensors and initialize the free-slot tracker.

        Required attributes after __init__:
            self.k_cache: (max_batch, max_seq_len, dim) zeros
            self.v_cache: (max_batch, max_seq_len, dim) zeros
            self.max_batch: int
            self.max_seq_len: int
            self.dim: int
            (plus whatever you need to track free slots)
        """
        self.k_cache = torch.zeros(max_batch, max_seq_len, dim)
        self.v_cache = torch.zeros(max_batch, max_seq_len, dim)
        self.max_batch = max_batch
        self.max_seq_len = max_seq_len
        self.dim = dim
        self.slots = [True] * max_batch  # True if free, False if full

    def allocate_slot(self) -> int | None:
        """Reserve a free slot and return its index.

        Returns:
            int in [0, max_batch) if a slot was free, otherwise None.
        """
        first_free = next((i for i, s in enumerate(self.slots) if s), None)
        if first_free is not None:
            self.slots[first_free] = False
        return first_free

    def free_slot(self, slot_idx: int) -> None:
        """Mark a slot as free for reuse.

        Zeroing the slot's K/V is allowed but not required — tests assume
        that whoever next allocates the slot will overwrite from position 0.
        """
        self.k_cache[slot_idx].zero_()  # optional, but we'll do it
        self.v_cache[slot_idx].zero_()  # optional, but we'll do it
        self.slots[slot_idx] = True

    def num_free(self) -> int:
        return sum(1 for s in self.slots if s is True)

    def num_active(self) -> int:
        return sum(1 for s in self.slots if s is False)
