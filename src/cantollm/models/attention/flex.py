"""Paged KV-cache attention implemented with PyTorch FlexAttention.

This attention method is used only by the continuous-batching path. A single
batch can contain both prefill rows and one-token decode rows, and each batch
row's KV history may be spread across non-contiguous physical blocks.

The padded attention implementation cannot be reused here. It assumes that a
request owns one contiguous cache slot, reads a contiguous prefix from that
slot, and uses a dense Boolean mask. The paged implementation instead works
with the flat tensors in :class:`PagedKVPool` and the table data attached to
``BatchMeta.paged_tables``.

For each step, that table data has four parts:

* ``block_tables`` maps each request's logical blocks to physical blocks in
  the pool;
* ``kv_num_blocks`` says how much of each batch row's block table is
  populated;
* ``inverse_tables`` maps physical blocks back to logical blocks so causal
  masking still follows the request's token order; and
* ``write_map.pool_index`` gives the flat pool destinations for the new K/V
  values.

Why both mapping directions are needed
--------------------------------------
Writes start with a logical token position and must find a flat pool index, so
they use the ordinary block table. FlexAttention sees keys by those physical
pool indices, but causality is defined by logical token positions, so its mask
uses the inverse table. This distinction is especially important when block
tables are permuted or their final block is only partly filled.

Implementation constraints
--------------------------
The paged attention path preserves these constraints:

* Build the ``BlockMask`` from the already-seeded table tensors. Pass the
  query and KV sequence lengths explicitly; FlexAttention otherwise infers
  the wrong query length for one-token decode.
* Apply causal masking in logical token order. Physical blocks that the
  request in a batch row does not own contain a sentinel in the inverse table
  and must remain invisible.
* Write directly into the flat ``layer_k`` and ``layer_v`` tensors using the
  paged write map. Do not mutate through a reshaped or sliced view: under
  ``torch.compile``, that can turn a small update into copies of the entire
  KV pool.
* Enable grouped-query attention when calling FlexAttention.

The correctness-first implementation constructs a fresh ``BlockMask`` on each
eager CPU step. Chunk 6 will cache masks over persistent table buffers, and
Chunk 8 will reuse those buffers during CUDA graph capture.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from cantollm.models.attention.protocol import BatchMeta

# Geometry floors of the compiled CUDA kernels on the probed build (5090,
# torch 2.10.0+cu128 / sm_120, 2026-08-30). Eager Flex checks neither, so
# CPU runs cannot catch a violation; revalidate both on torch upgrades
# (paged-kv-plan.md chunk log, chunk 4). The prefill templates tile
# queries at BLOCK_M = 128 and the mask's Q BLOCK_SIZE must be a multiple
# of it; mask KV blocks below 64 prune every template choice
# (NoValidChoicesError), which is what pins the served block_size default
# (paged-kv-plan.md §2.13), enforced at engine assembly.
Q_BLOCK_MULTIPLE = 128
MIN_CUDA_KV_BLOCK = 64


class FlexAttentionMethod:
    """AttentionMethod implementation for the flat, paged KV layout.

    Only the mixed-batch API is supported. The sequential prefill and decode
    APIs use a per-request ``KVCache`` and therefore cannot supply the block
    tables required by this implementation.

    Args:
        block_size: Number of token positions stored in each KV block.
    """

    def __init__(self, block_size: int):
        # The attention registry currently assumes constructors take no
        # arguments. Chunk 7 will update that wiring so the selected batching
        # configuration can provide this block size.
        self.block_size = block_size

    def execution_context(self):
        """Return the context used while running or compiling attention.

        FlexAttention needs no temporary dispatcher configuration, so this
        is a no-op context. The analogous hook in the SDPA implementation is
        different because it must explicitly select the cuDNN backend.
        """
        return nullcontext()

    def build_mask(
        self,
        start_pos: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Reject the sequential mask API.

        Paged masking requires one block table per batch row. Those tables are
        available only through :meth:`build_batched_mask` and ``BatchMeta``.
        """
        raise NotImplementedError("FlexAttentionMethod is batched-only")

    def forward_prefill(self, queries, keys, values, mask, kv_cache):
        """Reject sequential prefill; paged KV uses the mixed-batch API."""
        raise NotImplementedError("FlexAttentionMethod is batched-only")

    def forward_decode(self, queries, keys, values, mask, kv_cache):
        """Reject sequential decode; paged KV uses the mixed-batch API."""
        raise NotImplementedError("FlexAttentionMethod is batched-only")

    def build_batched_mask(
        self, meta: BatchMeta, device: torch.device
    ) -> BlockMask:
        """Build this step's FlexAttention ``BlockMask``.

        The mask must expose only the physical blocks assigned to the request
        in each batch row and must enforce causality using logical token
        positions. All table tensors are supplied in ``meta.paged_tables``;
        ``device`` is the device on which attention will run.
        """
        # from_kv_blocks' canonical ranks: (B, heads, q_blocks) and
        # (B, heads, q_blocks, N) with broadcast heads and one q block.
        # Eager Flex broadcasts squeezed shapes too, but the CUDA
        # templates derive kernel strides from the actual ndim and emit
        # broken code for them (5090 probe, 2026-08-30).
        kv_num_blocks = meta.paged_tables.kv_num_blocks[:, None, None]
        kv_indices = meta.paged_tables.block_tables[:, None, None, :]
        inverse_tables = meta.paged_tables.inverse_tables
        start_positions = meta.start_pos

        def mask_mod(
            batch_index: torch.Tensor,
            head_index: torch.Tensor,
            query_index: torch.Tensor,
            physical_key_index: torch.Tensor,
        ) -> torch.Tensor:
            """
            batch_index:       scalar index in [0, B)
            head_index:        scalar index in [0, number_of_query_heads)
            query_index:       scalar index in [0, meta.num_new_max)
            physical_key_index: scalar index in [0, total_flat_pool_positions)

            This decides whether a specific key is visible to a query.
            Position specified by:
            1) which matrix, given by (batch_index, head_index) pair
            2) which query (numbered from 0, where 0 is the first new position we're
            computing attention for)
            3) which key (specified by physical index in KV pool)

            We don't need to check if the key is in the batch, because this
            will be taken care of by the coarse block list.

            So we just have to deal with causality violation:
            Keep if logical key position <= logical query position
            """
            logical_query_pos = start_positions[batch_index] + query_index
            physical_key_block = physical_key_index // self.block_size
            k_offset = physical_key_index % self.block_size
            logical_key_block = inverse_tables[batch_index, physical_key_block]
            logical_key_pos = logical_key_block * self.block_size + k_offset
            return logical_key_pos <= logical_query_pos

        total_flat_pool_positions = inverse_tables.shape[1] * self.block_size

        # One q block spanning the step width, rounded up to the CUDA
        # prefill template's tile (BLOCK_M = 128 on the probed build,
        # which requires the mask's q block to be a multiple of it; a raw
        # width like 150 fails lowering, and eager never checks). The
        # wider block changes no semantics: per-position causality is
        # mask_mod's job either way.
        q_block = -(-meta.num_new_max // Q_BLOCK_MULTIPLE) * Q_BLOCK_MULTIPLE

        return BlockMask.from_kv_blocks(
            kv_num_blocks=kv_num_blocks,
            kv_indices=kv_indices,
            BLOCK_SIZE=(q_block, self.block_size),
            mask_mod=mask_mod,
            seq_lengths=(meta.num_new_max, total_flat_pool_positions),
            # Query-block metadata is only needed for the backward pass;
            # inference only runs the forward pass.
            compute_q_blocks=False,
        )

    def forward_batched(
        self, queries, keys, values, mask, layer_k, layer_v, meta: BatchMeta
    ) -> torch.Tensor:
        """Run one mixed prefill/decode step against one paged KV layer.

        ``keys`` and ``values`` contain the newly projected tokens for this
        step. Real tokens must first be written into the flat layer tensors
        at the destinations in ``meta.paged_tables.write_map``. Attention
        then combines ``queries`` with the visible cached K/V selected by
        ``mask``.

        Args:
            queries: Post-RoPE query tensor with shape
                ``(B, width, groups, heads_per_group, head_dim)``.
            keys: Post-RoPE keys with shape
                ``(B, width, groups, head_dim)``.
            values: Values with the same shape as ``keys``.
            mask: The method-specific object returned by
                :meth:`build_batched_mask`.
            layer_k: Flat K storage for one transformer layer. It is mutated
                in place.
            layer_v: Flat V storage for the same layer. It is mutated in
                place.
            meta: Batch geometry and the seeded paged tables for this step.

        Returns:
            Attention output with the same shape as ``queries``.
        """
        # Write the newly-computed k and v into the appropriate KV cache pages
        write_map = meta.paged_tables.write_map
        layer_k[write_map.pool_index] = keys[write_map.batch_row, write_map.token_offset]
        layer_v[write_map.pool_index] = values[write_map.batch_row, write_map.token_offset]

        # Reshape things into the form the FlexAttention kernel expects
        # For queries, flatten groups, heads_per_group into one and move it to the right place
        # (B, width, groups, heads_per_group, head_dim) -> (B, query_heads, width, head_dim)
        flex_queries = queries.flatten(2, 3).movedim(2, 1)
        # For KV, move groups dim, and add a singleton batch
        # dimension so the shared physical pool broadcasts across batch rows.
        # (pool_length, groups, head_dim) -> (batch_or_1, groups, pool_length, head_dim)
        flex_keys = layer_k.movedim(1, 0).unsqueeze(0)
        flex_values = layer_v.movedim(1, 0).unsqueeze(0)

        flex_context = flex_attention(
            flex_queries,
            flex_keys,
            flex_values,
            block_mask=mask,
            enable_gqa=True,
        )

        # Reshape to our expected output shape:
        # (B, query_heads, width, head_dim) -> (B, width, groups, heads_per_group, head_dim)
        context = flex_context.movedim(1, 2).unflatten(2, (queries.shape[2], -1))

        return context
