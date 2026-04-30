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
        self.k_cache = torch.zeros(max_batch, max_seq_len, dim, device=device)
        self.v_cache = torch.zeros(max_batch, max_seq_len, dim, device=device)
        self.max_batch = max_batch
        self.max_seq_len = max_seq_len
        self.dim = dim
        self.free: set[int] = set(range(max_batch))

    def allocate_slot(self) -> int | None:
        """Reserve a free slot and return its index.

        Returns:
            int in [0, max_batch) if a slot was free, otherwise None.
        """
        if not self.free:
            return None
        return self.free.pop()

    def free_slot(self, slot_idx: int) -> None:
        """Mark a slot as free for reuse.

        K/V contents are left as-is; the next allocator writes from position 0.
        """
        self.free.add(slot_idx)

    def num_free(self) -> int:
        return len(self.free)

    def num_active(self) -> int:
        return self.max_batch - len(self.free)
