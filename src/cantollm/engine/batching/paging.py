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

At the end of Chunk 3, `PagedStepState` only allocates and initializes those
buffers. Chunk 5 will add the scheduler-facing operation that fills them for a
step. Later chunks will reuse them when caching FlexAttention masks and
capturing CUDA graphs.
"""

from __future__ import annotations

import torch

from cantollm.models.attention.protocol import PagedKVWriteMap


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

    This Chunk 3 version only allocates the buffers and initializes the
    inverse table. Chunk 5 will add the per-step fill operation.
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
