"""Paged-step bookkeeping: block tables in tensor form + the write map.

Scheduler-side, per decision 2.5 (paged-kv-plan.md): the pool stays memory
only, the allocator owns which blocks are free, a sequence carries its
block table as host truth (`CBSequence.block_table`, chunk 5), and this
module turns that host truth into the step's tensor form — the
`PagedTables` references a `BatchMeta` carries seeded.

`paged_write_map` is hand-written (§8): the write-side half of the paged
index translation, the phase's educational core. `PagedStepState` is the
persistent-tensor owner; in this chunk it is a shell that allocates and
sentinel-fills the tensors — the per-step `fill()` lands with scheduler
integration (chunk 5), the per-family BlockMask reuse with warm-up/compile
(chunk 6), and the graph-static double duty with capture (chunk 8).
"""

from __future__ import annotations

import torch

from cantollm.models.attention.protocol import PagedKVWriteMap


def paged_write_map(
    rows: list[tuple[int, int, int]],
    tables: list[list[int]],
    block_size: int,
    device: torch.device | None = None,
) -> PagedKVWriteMap:
    """[HAND-WRITTEN, P4 chunk 4] The step's ragged KV write against the
    flat paged pool: one entry per real new token.

    `rows` is `BatchMeta.rows` — (slot_idx, start_pos, num_new) per row;
    `slot_idx` is the padded arm's field and is ignored here. `tables[r]`
    is row r's block table (logical block index → physical block id),
    covering at least `ceil((start_pos + num_new) / block_size)` blocks.

    For each row r and offset `off` in `range(num_new)`, the token at
    logical position `pos = start_pos + off` lands at flat row

        dst = tables[r][pos // block_size] * block_size + pos % block_size

    Filler rows (num_new == 0) contribute no entries, exactly like the
    padded derivation. Columns are (total_new,) int64 on `device`
    (None = CPU), aligned by construction — build all three in one walk.
    """
    raise NotImplementedError("author's chunk-4 session")


class PagedStepState:
    """Owner of the persistent paged-step tensors (decision 2.5).

    One instance per paged scheduler, allocated once at maximum geometry:
    per-step values are written into these tensors IN PLACE — never
    reallocated — because compiled artifacts guard on their properties and
    graph capture bakes their addresses (decision 2.6; spike gates 3-4).

    Chunk-3 shell: allocation and the sentinel convention only. `fill()`
    (chunk 5) will write a step's tables/lengths/inverse/write-map;
    per-family BlockMask reuse (chunk 6) and capture double duty (chunk 8)
    build on the same tensors.
    """

    def __init__(
        self,
        *,
        max_rows: int,
        max_blocks_per_seq: int,
        num_kv_blocks: int,
        device: torch.device,
    ):
        self.max_rows = max_rows
        self.max_blocks_per_seq = max_blocks_per_seq
        self.num_kv_blocks = num_kv_blocks
        self.block_tables = torch.zeros(
            (max_rows, max_blocks_per_seq), dtype=torch.int32, device=device
        )
        self.kv_num_blocks = torch.zeros(
            max_rows, dtype=torch.int32, device=device
        )
        # Physical block -> logical block index, per row, scratch block
        # included (hence + 1). The sentinel is max_blocks_per_seq: past
        # any logical bound, so an unowned block's translated positions
        # fail every causal bound (never 0/-1 — those alias visible
        # positions).
        self.inverse_tables = torch.full(
            (max_rows, num_kv_blocks + 1), max_blocks_per_seq,
            dtype=torch.int32, device=device,
        )
