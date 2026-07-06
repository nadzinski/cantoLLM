"""Deployment knobs for the continuous-batching engine.

Engine config, deliberately not `ModelSpec` (decision 7): these are sized to
the machine (smaller on the Mac than on the 5090), not to the model. Note the
two coexisting "max seq lens": the model's `spec.arch["max_seq_len"]` (40 960
for Qwen3) is a RoPE-table bound and far too large to preallocate KV for;
`max_seq_len` here is the per-slot pool capacity, and doubles as the
admission cap (`prompt_len + max_tokens <= max_seq_len`).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BatchingConfig:
    max_batch: int
    """Slots in the KV pool == max concurrently active sequences."""

    max_seq_len: int
    """Per-slot token capacity; also the per-request admission cap."""

    max_tokens_per_step: int
    """Total new tokens (prefill chunks + decodes) per forward pass."""

    def __post_init__(self) -> None:
        if self.max_batch <= 0:
            raise ValueError(f"max_batch must be positive, got {self.max_batch}")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        if self.max_tokens_per_step < self.max_batch:
            # The water-fill guarantee: budget >= active rows means every row
            # (decode rows cap at 1) gets at least one token per step, so no
            # zero-width rows ever reach the forward pass.
            raise ValueError(
                f"max_tokens_per_step ({self.max_tokens_per_step}) must be >= "
                f"max_batch ({self.max_batch})"
            )
