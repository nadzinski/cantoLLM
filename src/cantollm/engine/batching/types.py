"""Seam types for the continuous-batching engine.

`BatchedForwardFn` is the scheduler's entire view of the model (decision 4):
the scheduler is constructed with one of these — in production it's
`ModelRuntime.forward_batched`, in tests a toy stepper — and never imports a
model class. Step 6 adds the engine-shell command types here.
"""

from __future__ import annotations

from typing import Protocol

import torch

from cantollm.kv_pool import PaddedKVPool
from cantollm.models.attention.protocol import BatchMeta


class BatchedForwardFn(Protocol):
    def __call__(
        self,
        input_ids: torch.Tensor,
        meta: BatchMeta,
        pool: PaddedKVPool,
    ) -> torch.Tensor:
        """One mixed prefill/decode step.

        `input_ids` is (B, num_new_max) int64, left-aligned, 0-padded, row
        order matching `meta.rows`. Returns (B, vocab) logits taken at each
        row's last real token. Rows still mid-prefill get logits too in v1;
        the scheduler must not sample from them.
        """
        ...
