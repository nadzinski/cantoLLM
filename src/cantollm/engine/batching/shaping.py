"""Step shaping: pad each step's geometry into the bounded shape vocabulary.

Why this module exists: some kernel machinery pays a one-time cost per
distinct problem shape — cuDNN's SDPA backend compiles a ~200 ms execution
plan per shape, and CUDA graphs (Phase 3's next item) need one capture per
shape. The scheduler naturally produces a new shape almost every step (the
KV span grows token by token; batch size moves with arrivals). Left alone,
that turns per-shape costs into live-request stalls (sdpa-results.md);
bounded to a small vocabulary and pre-warmed, they vanish
(shape-buckets-results.md).

This is the single boundary where that concern lives. The scheduler plans
a step exactly as if shapes were free — the v1 read of scheduler.py is
unchanged — and `shape_step` then pads the planned batch into the
vocabulary, on three axes:

  - rows:  pad the batch to the next `batch_buckets` entry with filler
    rows (`FILLER_SPEC`, num_new == 0): they write no KV (`kv_write_map`
    skips them by construction), read slot 0's history under the causal
    mask, and their output rows are garbage nobody gathers.
  - width: round `num_new_max` up into {1} ∪ `prefill_widths` — extra pad
    columns, the same machinery narrow rows already use. (The water-fill
    separately quantizes mid-prompt chunk *allocations* to the menu — that
    is allocation policy and stays in the scheduler; this handles the
    final-chunk remainders.)
  - kv:    round `max_history_len` up to a `kv_bucket` multiple, capped at
    the slot capacity — the per-row causal mask fences the over-read.

All knobs default to None (see BatchingConfig): with none set this is an
exact no-op and the engine runs v1 geometry. The padded compute is wasted
on purpose — a bounded, warm shape beats an exact, cold one.
"""

from __future__ import annotations

import torch

from cantollm.engine.batching.config import BatchingConfig
from cantollm.models.attention.protocol import BatchMeta


def round_up_to(value: int, menu: list[int] | tuple[int, ...]) -> int:
    """Smallest menu entry >= value (menu ascending; caller guarantees one
    exists — config validation pins the menus' tops)."""
    for entry in menu:
        if entry >= value:
            return entry
    raise ValueError(f"{value} exceeds the menu top {menu[-1]}")


FILLER_SPEC = (0, 0, 0)
"""A filler row's (slot, start_pos, num_new): reads slot 0's history under
the causal mask, writes nothing, and its output row is garbage nobody
gathers. Appended after the real rows, so real row indices are stable."""


def shape_step(
    input_ids: torch.Tensor, meta: BatchMeta, config: BatchingConfig
) -> tuple[torch.Tensor, BatchMeta]:
    """Pad a planned step's (input_ids, meta) into the shape vocabulary.

    Exact no-op when no bucket knobs are set, and when the planned geometry
    already sits on a vocabulary point (steady-state decode usually does).
    """
    rows = len(meta.rows)
    pad_rows = rows
    if config.batch_buckets is not None:
        pad_rows = round_up_to(rows, config.batch_buckets)
    width = meta.num_new_max
    if config.prefill_widths is not None and width > 1:
        width = round_up_to(width, config.prefill_widths)
    kv_len = meta.max_history_len
    if config.kv_bucket is not None:
        rounded = -(-kv_len // config.kv_bucket) * config.kv_bucket
        kv_len = min(rounded, config.max_seq_len)
    if (pad_rows, width, kv_len) == (rows, meta.num_new_max, meta.max_history_len):
        return input_ids, meta

    shaped_ids = input_ids.new_zeros((pad_rows, width))
    shaped_ids[:rows, : meta.num_new_max] = input_ids

    specs = list(meta.rows) + [FILLER_SPEC] * (pad_rows - rows)
    start_pos = torch.tensor([s[1] for s in specs])
    num_new = torch.tensor([s[2] for s in specs])
    shaped_meta = BatchMeta(
        rows=specs,
        slots=torch.tensor([s[0] for s in specs]),
        start_pos=start_pos,
        num_new=num_new,
        positions=start_pos[:, None] + torch.arange(width)[None, :],
        num_new_max=width,
        max_history_len=kv_len,
        device=meta.device,
    )
    return shaped_ids, shaped_meta
