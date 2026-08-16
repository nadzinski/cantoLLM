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
5090 round. So each meta gets a seeded map whose entries all park on the
pool's scratch column (the same convention graph replay uses for filler
rows), with lengths chosen so compile sees both traffic populations
(1 specializes, >= 2 goes symbolic — see the loop comment). The writes
land where no gather ever reads: only the scratch column gets dirty, the
logical pool stays untouched.

Everything here is built as CPU tensors, exactly like the scheduler
builds a traffic step, so the warm-up forwards enter the runtime front
through the same device move traffic takes. That is deliberate and
guard-load-bearing: the front's move runs under `inference_mode`, and
Dynamo artifacts guard on the tensors' dispatch key set — warm-up tensors
that skip the move (e.g. pre-built on device) carry ADInplaceOrView where
traffic's moved tensors do not, and every artifact the sweep builds gets
rejected and recompiled by the first live request (the §3 recompile
tripwire caught exactly this on the 2026-08-08 A/B).
"""

from __future__ import annotations

import logging
import time

import torch

from cantollm import progress
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
    CPU tensors, like a scheduler-built step; `device` is only where a
    derived map would land (the seeded map makes that moot)."""
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


def scratch_write_map(
    length: int, batch: int, scratch_pos: int
) -> KVWriteMap:
    """A seeded map of `length` entries, all parked on the scratch column.
    Rows cycle over the batch (entries stay valid for any length); offsets
    stay 0, so entries are valid at any width. Duplicate destinations are
    fine: every write lands on scratch cells no gather ever reads. CPU
    tensors; the runtime front moves them with the rest of the meta."""
    rows = torch.arange(length, dtype=torch.int64) % batch
    return KVWriteMap(
        row=rows,
        off=torch.zeros(length, dtype=torch.int64),
        slot=rows.clone(),
        pos=torch.full((length,), scratch_pos, dtype=torch.int64),
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
    # Seeded map lengths alternate between 1 and max(2, batch) along each
    # (batch, width) family's kv sweep. Compile needs BOTH artifacts per
    # family: torch's 0/1 rule specializes a length-1 map (a step with one
    # real new token — a lone decode row, or a lone prefill row's final
    # short chunk), while any length >= 2 goes symbolic and serves every
    # other real-token count. Seeding only length = batch left the
    # batch-1 prefill families specialized at length 1, and the first
    # 1-row prefill chunk after Ready paid a compile stall (the 2026-08-08
    # A/B's last tripwire find). Alternating inside the family costs zero
    # extra forwards and gives each artifact lineage the >= 2 kv values
    # automatic dynamic needs to promote the span.
    family = None
    idx = 0
    for i, (batch, width, kv_len) in enumerate(vocabulary):
        if (batch, width) != family:
            family, idx = (batch, width), 0
        else:
            idx += 1
        length = 1 if idx % 2 == 0 else max(2, batch)
        input_ids = torch.zeros((batch, width), dtype=torch.int64)
        meta = warmup_meta(batch, width, kv_len, device)
        meta.seed_kv_write_map(
            scratch_write_map(length, batch, pool.scratch_pos)
        )
        forward_fn(input_ids, meta, pool)
        progress.report("sweep", i + 1, len(vocabulary))
    if device.type == "cuda":
        torch.cuda.synchronize()
    logger.info(
        "shape warm-up done: %d shapes in %.1f s",
        len(vocabulary), time.perf_counter() - t0,
    )
    return len(vocabulary)
