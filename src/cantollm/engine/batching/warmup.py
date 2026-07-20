"""Shape-vocabulary warm-up: pay per-shape one-time costs before serving.

Kernels with shape-keyed caches (cuDNN SDPA compiles a ~200 ms execution
plan per distinct problem shape; CUDA graphs later need one capture per
shape) turn a cold shape into a live-request stall. With the shape
vocabulary bounded (`BatchingConfig.shapes_bounded`), the whole vocabulary
is enumerable — so run one throwaway forward per shape at engine build
time, behind the process split's Ready, and no request can ever hit a cold
shape.

Each dummy step is built entirely from filler rows (num_new == 0): nothing
is written to the pool (`kv_write_map` skips fillers by construction), the
gather reads slot 0's uninitialized memory under the causal mask, and the
output is dropped. Only the shape reaches the kernel.
"""

from __future__ import annotations

import logging
import time

import torch

from cantollm.engine.batching.config import BatchingConfig
from cantollm.engine.batching.types import BatchedForwardFn
from cantollm.kv_pool import PaddedKVPool
from cantollm.models.attention.protocol import BatchMeta

logger = logging.getLogger(__name__)


def warmup_meta(
    batch: int, width: int, kv_len: int, device: torch.device | None
) -> BatchMeta:
    """All-filler geometry for one vocabulary shape: `batch` rows of
    (slot 0, start 0, num_new 0), tensor width `width`, KV span `kv_len`."""
    zeros = torch.zeros(batch, dtype=torch.int64)
    return BatchMeta(
        rows=[(0, 0, 0)] * batch,
        slots=zeros.clone(),
        start_pos=zeros.clone(),
        num_new=zeros.clone(),
        positions=torch.arange(width)[None, :].expand(batch, -1).clone(),
        num_new_max=width,
        max_history_len=kv_len,
        device=device,
    )


def warmup_shape_vocabulary(
    forward_fn: BatchedForwardFn, pool: PaddedKVPool, config: BatchingConfig
) -> int:
    """One dummy forward per (batch, width, kv_len) in the vocabulary.
    Returns the number of shapes warmed. Logs progress and total time."""
    vocabulary = config.shape_vocabulary()
    device = pool.k.device
    logger.info(
        "warming %d shapes (batch buckets %s, widths {1} + %s, kv step %d)",
        len(vocabulary), config.batch_buckets, config.prefill_widths,
        config.kv_bucket,
    )
    t0 = time.perf_counter()
    for batch, width, kv_len in vocabulary:
        input_ids = torch.zeros((batch, width), dtype=torch.int64, device=device)
        meta = warmup_meta(batch, width, kv_len, device)
        forward_fn(input_ids, meta, pool)
    if device.type == "cuda":
        torch.cuda.synchronize()
    logger.info(
        "shape warm-up done: %d shapes in %.1f s",
        len(vocabulary), time.perf_counter() - t0,
    )
    return len(vocabulary)
