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
        # Decode-step write map, persistent like the tables (chunk 8):
        # graph capture bakes these addresses, and the round-1 probe
        # measured fill's former per-element device writes as the
        # dominant flex-side host cost. batch_row/token_offset are
        # constants of the decode convention (row k writes its own
        # offset 0); pool_index is rewritten per step.
        self.map_batch_row = torch.arange(
            max_rows, dtype=torch.int64, device=device
        )
        self.map_token_offset = torch.zeros(
            max_rows, dtype=torch.int64, device=device
        )
        self.map_pool_index = torch.zeros(
            max_rows, dtype=torch.int64, device=device
        )
        # CPU staging twins: fill() assembles a step host-side, then
        # lands it in the device buffers with one bulk copy_ per buffer
        # (addresses stable; the copies replace hundreds of tiny H2D
        # writes per step at long KV).
        self._stage_block_tables = torch.zeros(
            (max_rows, max_blocks_per_seq), dtype=torch.int32
        )
        self._stage_kv_num_blocks = torch.zeros(max_rows, dtype=torch.int32)
        self._stage_inverse = torch.full(
            (max_rows, num_kv_blocks + 1), max_blocks_per_seq,
            dtype=torch.int32,
        )
        self._stage_start_pos = torch.zeros(max_rows, dtype=torch.int64)
        self._stage_map_pool_index = torch.zeros(max_rows, dtype=torch.int64)
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

        Decode-shaped steps (every row ``num_new <= 1``, at least one
        real row) get their write map from the PERSISTENT map buffers,
        padded to the batch: one entry per row, fillers writing their
        garbage to the scratch block's first position (the padded
        graph path's convention, §3's map spec). Graph capture bakes
        those addresses, so replay needs no map marshal at all: this
        method IS the marshal. Prefill-shaped steps keep the exact
        one-entry-per-real-token map, built fresh.

        The step is assembled in CPU staging buffers and landed with one
        bulk ``copy_`` per device buffer: same addresses, but a handful
        of H2D copies instead of hundreds of per-element writes (the
        round-1 probe's flex-side host gap).
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
        decode_shaped = (
            all(n <= 1 for _, _, n in step_rows)
            and any(n == 1 for _, _, n in step_rows)
        )

        # Reset only the rows this step uses: entries a previous step wrote
        # for blocks a row no longer owns must not leak into its mask.
        stage_bt = self._stage_block_tables[:batch]
        stage_kn = self._stage_kv_num_blocks[:batch]
        stage_inv = self._stage_inverse[:batch]
        stage_sp = self._stage_start_pos[:batch]
        stage_bt.zero_()
        stage_kn.zero_()
        stage_inv.fill_(self.max_blocks_per_seq)

        for r, (_, start_pos, num_new) in enumerate(step_rows):
            stage_sp[r] = start_pos
            if r >= len(block_tables):
                # Filler row: one visible block, the scratch block, mapped
                # to logical 0 so position 0 is visible to every query.
                stage_bt[r, 0] = scratch_block
                stage_kn[r] = 1
                stage_inv[r, scratch_block] = 0
                if decode_shaped:
                    self._stage_map_pool_index[r] = scratch_block * block_size
                continue
            table = block_tables[r]
            history = start_pos + num_new
            visible = -(-history // block_size)
            if visible > len(table):
                raise ValueError(
                    f"row {r} reaches position {history} but its table "
                    f"holds {len(table)} blocks of {block_size}"
                )
            physical = torch.tensor(table, dtype=torch.int32)
            stage_bt[r, : len(table)] = physical
            stage_inv[r, physical.long()] = torch.arange(
                len(table), dtype=torch.int32
            )
            stage_kn[r] = visible
            if decode_shaped and num_new == 1:
                self._stage_map_pool_index[r] = (
                    table[start_pos // block_size] * block_size
                    + start_pos % block_size
                )

        self.block_tables[:batch].copy_(stage_bt)
        self.kv_num_blocks[:batch].copy_(stage_kn)
        self.inverse_tables[:batch].copy_(stage_inv)
        self.start_pos[:batch].copy_(stage_sp)
        if decode_shaped:
            self.map_pool_index[:batch].copy_(
                self._stage_map_pool_index[:batch]
            )
            write_map = PagedKVWriteMap(
                batch_row=self.map_batch_row[:batch],
                token_offset=self.map_token_offset[:batch],
                pool_index=self.map_pool_index[:batch],
            )
        else:
            write_map = paged_write_map(
                step_rows,
                block_tables + [[]] * (batch - len(block_tables)),
                block_size,
                device=self.block_tables.device,
            )

        tables = PagedTables(
            block_tables=self.block_tables[:batch],
            kv_num_blocks=self.kv_num_blocks[:batch],
            inverse_tables=self.inverse_tables[:batch],
            write_map=write_map,
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
                # Pin the mask's own tensors static: they enter the
                # compiled graph as inputs whose batch dim differs per
                # family, and automatic dynamic would promote them to
                # symbolic on the second family. A symbolic batch
                # anywhere near the flex call risks the same silent
                # flex-decoding disqualification the 2026-08-30 round-1
                # A/B measured as a 4x multi-row decode cliff
                # (runtime._mark_compile_dims covers the meta-side
                # tensors; these views are created inside the builder,
                # out of its reach).
                for t in (
                    getattr(mask, "kv_num_blocks", None),
                    getattr(mask, "kv_indices", None),
                ):
                    if isinstance(t, torch.Tensor):
                        torch._dynamo.mark_static(t)
                self.masks[key] = mask
            tables = tables._replace(mask=mask)
        return tables
