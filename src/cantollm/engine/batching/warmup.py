"""Shape-vocabulary warm-up: pay per-shape one-time costs before serving.

Kernels with shape-keyed caches (cuDNN SDPA compiles a ~200 ms execution
plan per distinct problem shape; CUDA graphs later need one capture per
shape) turn a cold shape into a live-request stall. With the shape
vocabulary bounded (`BatchingConfig.shapes_bounded`), the whole vocabulary
is enumerable — so run one throwaway forward per shape at engine build
time, behind the process split's Ready, and no request can ever hit a cold
shape.

Each dummy step is built from filler rows (num_new == 0), so the mask and
gather see the shape's real geometry while no real pool position is ever
written. The write map, however, is NOT left at the fillers' natural
length of zero: torch.compile artifacts guard on the map length, 0-sized
dims specialize, and a sweep of empty maps would leave the first real
request to pay a compile stall — the guard-set gap found on the 2026-08-08
5090 round. So each meta gets a seeded map with one entry per row, every
entry parked on the pool's scratch column (the same convention graph
replay uses for filler rows): compile sees the traffic map lengths
(1 specializes, >= 2 goes symbolic) and the writes land where no gather
ever reads. Only the scratch column gets dirty; the logical pool stays
untouched.
"""

from __future__ import annotations

import logging
import time

import torch

from cantollm.engine.batching.config import BatchingConfig
from cantollm.engine.batching.types import BatchedForwardFn
from cantollm.kv_pool import PaddedKVPool
from cantollm.models.attention.protocol import BatchMeta, KVWriteMap

logger = logging.getLogger(__name__)


def warmup_meta(
    batch: int, width: int, kv_len: int, device: torch.device | None
) -> BatchMeta:
    """All-filler geometry for one vocabulary shape: `batch` rows of
    (slot 0, start 0, num_new 0), tensor width `width`, KV span `kv_len`.

    Tensors are built on `device` directly: the runtime front only keeps a
    seeded `kv_write_map` when no device move is needed (a move `replace`s
    the meta, dropping the seed — the 40fbcf9 lesson), and the sweep's
    seeded scratch maps must survive into the traced region.
    """
    zeros = torch.zeros(batch, dtype=torch.int64, device=device)
    return BatchMeta(
        rows=[(0, 0, 0)] * batch,
        slots=zeros.clone(),
        start_pos=zeros.clone(),
        num_new=zeros.clone(),
        positions=torch.arange(width, device=device)[None, :]
        .expand(batch, -1).clone(),
        num_new_max=width,
        max_history_len=kv_len,
        device=device,
    )


def scratch_write_map(
    batch: int, scratch_pos: int, device: torch.device | None
) -> KVWriteMap:
    """A seeded map with one entry per row, all parked on the scratch
    column: row r writes its offset 0 into (slot r, scratch). Gives the
    warm-up sweep the map lengths real traffic produces without touching
    a single logical pool position."""
    long = dict(dtype=torch.int64, device=device)
    return KVWriteMap(
        row=torch.arange(batch, **long),
        off=torch.zeros(batch, **long),
        slot=torch.arange(batch, **long),
        pos=torch.full((batch,), scratch_pos, **long),
    )


def warmup_shape_vocabulary(
    forward_fn: BatchedForwardFn, pool: PaddedKVPool, config: BatchingConfig
) -> int:
    """One dummy forward per (batch, width, kv_len) in the vocabulary.
    Returns the number of shapes warmed. Logs progress and total time."""
    vocabulary = config.shape_vocabulary()
    device = pool.device
    logger.info(
        "warming %d shapes (batch buckets %s, widths {1} + %s, kv step %d)",
        len(vocabulary), config.batch_buckets, config.prefill_widths,
        config.kv_bucket,
    )
    t0 = time.perf_counter()
    for batch, width, kv_len in vocabulary:
        input_ids = torch.zeros((batch, width), dtype=torch.int64, device=device)
        meta = warmup_meta(batch, width, kv_len, device)
        meta.seed_kv_write_map(scratch_write_map(batch, pool.scratch_pos, device))
        forward_fn(input_ids, meta, pool)
    if device.type == "cuda":
        torch.cuda.synchronize()
    logger.info(
        "shape warm-up done: %d shapes in %.1f s",
        len(vocabulary), time.perf_counter() - t0,
    )
    return len(vocabulary)
