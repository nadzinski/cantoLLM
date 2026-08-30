"""Tensor machinery for one paged-KV attention step.

The paged KV pool contains only K/V data. It does not know which physical
blocks belong to which request. The scheduler owns that assignment and keeps
each request's block table as ordinary Python state.

Before the model can run a batch, the scheduler must describe those block
assignments with tensors. Attention needs four pieces of information:

* each batch row's logical-to-physical block table;
* the number of populated blocks in each batch row;
* the inverse, physical-to-logical mapping used by the causal mask; and
* a write map telling the model where to store the new K/V values.

`paged_write_map` builds the last item. `PagedStepState` owns the first three
as reusable buffers. The buffers are allocated once and updated in place so
their shapes and memory addresses stay stable for ``torch.compile`` and CUDA
graphs.

`PagedStepState.fill` (chunk 5) is the scheduler-facing operation that
updates them for one step and hands back the `PagedTables` a `BatchMeta`
seeds. The state also owns the per-family FlexAttention `BlockMask`
objects (chunk 6, paged-kv-plan.md §2.5/§2.6): one mask per
(batch, width) family, built once over the persistent buffers by an
injected `mask_builder` and reused for every later step of that family,
so `BlockMask.from_kv_blocks` never runs in the step loop. Everything a
cached mask reads per step (tables, inverse tables, and each row's
logical start position) therefore lives in these buffers and is
rewritten in place by `fill`. Chunk 8 reuses the same buffers when
capturing CUDA graphs.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from cantollm.models.attention.protocol import PagedKVWriteMap, PagedTables


def paged_write_map(
    batch_rows: list[tuple[int, int, int]],
    block_tables: list[list[int]],
    block_size: int,
    device: torch.device | None = None,
) -> PagedKVWriteMap:
    """Build the indices used to write one step's new K/V values.

    The model produces K/V tensors with a rectangular batch shape, but rows
    can contain different numbers of real tokens. The paged pool is also
    flat: a request's consecutive logical tokens may be stored in unrelated
    physical blocks. This function connects those two layouts.

    Args:
        batch_rows: One ``(slot, start_pos, num_new)`` tuple per batch row,
            in the same format as :attr:`BatchMeta.rows`. ``slot`` belongs
            to the padded layout and is not used here. ``start_pos`` is the
            logical position of the batch row's first new token, and
            ``num_new`` is the number of real tokens in that batch row.
        block_tables: One block table per batch row.
            ``block_tables[batch_row][logical_block]`` gives the physical
            block for that request. Each table must cover all positions
            written by its batch row.
        block_size: Number of token positions in one physical block.
        device: Device for the returned index tensors. ``None`` creates CPU
            tensors.

    Returns:
        Three aligned one-dimensional ``int64`` tensors, wrapped in
        :class:`PagedKVWriteMap`. For every index ``k``::

            layer_k[map.pool_index[k]] = keys[map.batch_row[k], map.token_offset[k]]

        and the equivalent assignment applies to values. There is exactly
        one entry per real new token, ordered by batch row and then by token
        offset. Filler rows (``num_new == 0``) add no entries.

    A token's destination is found by splitting its logical position into a
    logical block number and an offset within that block, translating the
    block number through the batch row's block table, and flattening the
    physical block and offset into one flat pool index.

    """
    entries: list[tuple[int, int, int]] = []
    for batch_row_index, row_spec in enumerate(batch_rows):
        _, start_pos, num_new = row_spec
        for token_offset in range(num_new):
            abs_token_pos = start_pos + token_offset
            logical_block = abs_token_pos // block_size
            offset_within_block = abs_token_pos % block_size
            kv_pool_block = block_tables[batch_row_index][logical_block]
            pool_index = kv_pool_block * block_size + offset_within_block
            entries.append((batch_row_index, token_offset, pool_index))

    if entries:
        batch_rows_out, token_offsets, pool_indexes = zip(*entries)
    else:
        batch_rows_out, token_offsets, pool_indexes = (), (), ()

    def as_index(values) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.long, device=device)

    return PagedKVWriteMap(
        batch_row=as_index(batch_rows_out),
        token_offset=as_index(token_offsets),
        pool_index=as_index(pool_indexes),
    )


class PagedStepState:
    """Reusable tensor storage describing the paged layout of a batch.

    Each paged scheduler owns one instance. The tensors are sized for the
    largest supported batch and block table, then reused for every step.
    Only their contents change. Keeping the same tensor objects matters for
    two later optimizations: compiled functions specialize on tensor
    properties, and CUDA graphs record the addresses of their inputs.

    Attributes:
        block_tables: ``(max_rows, max_blocks_per_seq)`` int32 tensor. Entry
            ``[batch_row, logical_block]`` gives that batch row's physical
            block.
        kv_num_blocks: ``(max_rows,)`` int32 tensor. Gives the number of
            meaningful block-table entries for each batch row.
        inverse_tables: ``(max_rows, num_kv_blocks + 1)`` int32 tensor. For
            each batch row, maps a physical block back to its logical block
            number. The extra column represents the pool's scratch block.
        start_pos: ``(max_rows,)`` int64 tensor. Each batch row's first new
            token's logical position. Exists for the cached masks: the
            Flex ``mask_mod`` needs per-row starts, and a mask reused
            across steps must read them from a persistent buffer, not from
            the step's own ``meta.start_pos`` (a fresh tensor every step).
        mask_builder: Optional callable
            ``(tables, start_positions, num_new_max) -> mask`` injected at
            engine assembly (the attention method owns mask semantics; the
            state owns only the cache). ``None`` skips mask caching and
            `fill` returns ``mask=None`` (the eager test arrangement: the
            method then builds a fresh mask per step).
    """

    def __init__(
        self,
        *,
        max_rows: int,
        max_blocks_per_seq: int,
        num_kv_blocks: int,
        device: torch.device,
        mask_builder: Callable[[PagedTables, torch.Tensor, int], object]
        | None = None,
    ):
        self.max_rows = max_rows
        self.max_blocks_per_seq = max_blocks_per_seq
        self.num_kv_blocks = num_kv_blocks
        self.mask_builder = mask_builder
        # One mask per (batch, num_new_max) family, built on first
        # occurrence (warm-up covers every family behind Ready, so traffic
        # never pays a construction) and reused forever: the mask's
        # tensors are views of the buffers below, which fill() rewrites in
        # place.
        self.masks: dict[tuple[int, int], object] = {}
        self.block_tables = torch.zeros(
            (max_rows, max_blocks_per_seq), dtype=torch.int32, device=device
        )
        self.kv_num_blocks = torch.zeros(
            max_rows, dtype=torch.int32, device=device
        )
        self.start_pos = torch.zeros(
            max_rows, dtype=torch.int64, device=device
        )
        # Most physical blocks do not belong to a given batch row. Initialize
        # every inverse-table entry to an invalid logical block number, then
        # let the per-step fill operation replace entries for blocks the batch
        # row actually owns. Valid logical block indices are in
        # [0, max_blocks_per_seq), so max_blocks_per_seq is a safe sentinel:
        # the causal mask will reject positions translated through it. Using
        # 0 or -1 would be unsafe because either can alias a visible position.
        # The final column is included for the pool's scratch block.
        self.inverse_tables = torch.full(
            (max_rows, num_kv_blocks + 1), max_blocks_per_seq,
            dtype=torch.int32, device=device,
        )

    def fill(
        self,
        step_rows: list[tuple[int, int, int]],
        block_tables: list[list[int]],
        block_size: int,
        num_new_max: int | None = None,
    ) -> PagedTables:
        """Describe one step's batch in the persistent buffers.

        Args:
            step_rows: The shaped batch's ``(slot, start_pos, num_new)``
                specs, in batch-row order: ``meta.rows`` after
                ``shape_step``, so filler rows appear at the tail.
            block_tables: One block table per REAL row, aligned with the
                head of ``step_rows``; rows past ``len(block_tables)`` are
                fillers. Each table must cover its row's
                ``start_pos + num_new`` positions (the scheduler's block
                reservation guarantees this before the row commits).
            block_size: Token positions per physical block.
            num_new_max: The step's padded width, which names the mask
                family together with the batch size. ``None`` (or a state
                with no ``mask_builder``) skips the mask and the returned
                tables carry ``mask=None``.

        Returns:
            A `PagedTables` whose table tensors are views of this state's
            buffers, sliced to the step's batch size (references, never
            copies; the seeded-buffer discipline compile and graph
            capture rely on), plus the step's freshly built write map and
            the family's cached mask.

        A filler row (``num_new == 0``) points at the scratch block, mapped
        to logical block 0 with one visible block: its softmax row stays
        finite over garbage nobody reads, where an all-sentinel inverse
        would mask the row completely and produce NaN
        (paged-kv-plan.md §4). Real rows never see the scratch block; its
        inverse entry stays the sentinel.
        """
        batch = len(step_rows)
        if batch > self.max_rows:
            raise ValueError(
                f"step has {batch} rows but the buffers hold {self.max_rows}"
            )
        if len(block_tables) > batch:
            raise ValueError(
                f"{len(block_tables)} block tables for {batch} rows"
            )
        scratch_block = self.num_kv_blocks

        # Reset only the rows this step uses: entries a previous step wrote
        # for blocks a row no longer owns must not leak into its mask.
        self.block_tables[:batch].zero_()
        self.kv_num_blocks[:batch].zero_()
        self.inverse_tables[:batch].fill_(self.max_blocks_per_seq)

        for r, (_, start_pos, num_new) in enumerate(step_rows):
            self.start_pos[r] = start_pos
            if r >= len(block_tables):
                # Filler row: one visible block, the scratch block, mapped
                # to logical 0 so position 0 is visible to every query.
                self.block_tables[r, 0] = scratch_block
                self.kv_num_blocks[r] = 1
                self.inverse_tables[r, scratch_block] = 0
                continue
            table = block_tables[r]
            history = start_pos + num_new
            visible = -(-history // block_size)
            if visible > len(table):
                raise ValueError(
                    f"row {r} reaches position {history} but its table "
                    f"holds {len(table)} blocks of {block_size}"
                )
            for logical, physical in enumerate(table):
                self.block_tables[r, logical] = physical
                self.inverse_tables[r, physical] = logical
            self.kv_num_blocks[r] = visible

        tables = PagedTables(
            block_tables=self.block_tables[:batch],
            kv_num_blocks=self.kv_num_blocks[:batch],
            inverse_tables=self.inverse_tables[:batch],
            write_map=paged_write_map(
                step_rows,
                block_tables + [[]] * (batch - len(block_tables)),
                block_size,
                device=self.block_tables.device,
            ),
        )
        if self.mask_builder is not None and num_new_max is not None:
            key = (batch, num_new_max)
            mask = self.masks.get(key)
            if mask is None:
                # First occurrence of this family (warm-up, normally). The
                # mask is built over views of the persistent buffers, so
                # this step's in-place writes above, and every later
                # step's, are what the reused mask reads.
                mask = self.mask_builder(
                    tables, self.start_pos[:batch], num_new_max
                )
                self.masks[key] = mask
            tables = tables._replace(mask=mask)
        return tables
