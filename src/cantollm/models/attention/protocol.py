"""Pluggable attention-compute boundary.

An `AttentionMethod` owns the attention math (scores + weighted values)
and the KV cache layout it reads/writes. `GroupedQueryAttention` delegates
to one: the module handles Q/K/V projections, head reshaping, q_norm/k_norm,
and RoPE, then hands post-RoPE tensors to the method for the compute.

The `Method` suffix disambiguates the protocol from the `GroupedQueryAttention`
module (also "an attention") and from the generation-strategy `InferenceBackend`
at `cantollm/engine/backend.py`.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple, Protocol

import torch


class KVWriteMap(NamedTuple):
    """The step's ragged KV write, flattened to one entry per real new token.

    Entry k copies one token's (groups, head_dim) vector:
    `keys[row[k], off[k]] → layer_k[slot[k], pos[k]]`. All four tensors are
    (total_new,) long and aligned by position — the alignment is guaranteed
    by construction (one walk over `BatchMeta.rows` appends all four values
    per token), not by the types; don't rebuild any column independently.
    """

    row: torch.Tensor
    """(total_new,) long — source row in the padded keys/values batch."""

    off: torch.Tensor
    """(total_new,) long — source offset along that row's num_new axis."""

    slot: torch.Tensor
    """(total_new,) long — destination slot in the KV pool."""

    pos: torch.Tensor
    """(total_new,) long — destination position within that slot."""


class PagedKVWriteMap(NamedTuple):
    """Indices for copying this step's new K/V vectors into the paged pool.

    Suppose the current batch contains ``N`` real new tokens in total. Each
    field is a one-dimensional ``int64`` tensor of length ``N``. The three
    values at index ``k`` describe one complete copy operation::

        layer_k[pool_index[k]] = keys[batch_row[k], token_offset[k]]
        layer_v[pool_index[k]] = values[batch_row[k], token_offset[k]]

    ``batch_row`` and ``token_offset`` locate the source token in the
    rectangular K/V tensors produced by the model for this step.
    ``pool_index`` is the exact index in the first dimension of the flat
    paged-pool tensor where that token belongs. The block-table lookup has
    already been performed when ``pool_index`` is constructed; code that
    consumes this map does not need to inspect a block table.

    For example, if batch row 0 contributes two new tokens and batch row 1
    contributes one, a map might contain::

        batch_row =    [0, 0, 1]
        token_offset = [0, 1, 0]
        pool_index =   [22, 23, 8]

    That means: copy batch row 0's first two new K/V vectors to flat pool
    indices 22 and 23, then copy batch row 1's first new K/V vector to flat
    pool index 8.

    The fields must remain aligned: index ``k`` in one field is meaningful
    only with index ``k`` in the other two. Padding and filler rows do not
    appear in the map. The same map is reused for every transformer layer,
    because the source and destination positions are identical across
    layers even though the K/V values differ.
    """

    batch_row: torch.Tensor
    """Source batch-row index for each token.

    ``batch_row[k]`` selects the first dimension of ``keys`` and ``values``.
    It is the token's position in the current model batch, not a request ID,
    cache slot, physical block number, or index in the KV pool.
    """

    token_offset: torch.Tensor
    """Source token offset within each batch row.

    ``token_offset[k]`` selects the second dimension of ``keys`` and
    ``values``. Zero means the first token newly processed for that batch row
    in this step. It is relative to the current input chunk, not the token's
    absolute position in the request.
    """

    pool_index: torch.Tensor
    """Destination index in one layer's flat paged KV tensor.

    ``pool_index[k]`` already combines the token's physical block number with
    its offset inside that block. It directly indexes the first dimension of
    ``layer_k`` and ``layer_v``.
    """


class PagedTables(NamedTuple):
    """One paged step's table references (paged-kv-plan.md §2.5).

    Engine-owned tensors — persistent, mutated in place per step by the
    scheduler side — carried by `BatchMeta` the way a seeded
    `kv_write_map` is: references, never copies. Table tensors are int32
    per the `BlockMask.from_kv_blocks` contract; the write map's columns
    stay int64 index tensors.
    """

    block_tables: torch.Tensor
    """(B, max_blocks_per_seq) int32 — logical-to-physical mappings.

    ``block_tables[batch_row, logical_block]`` is the physical block assigned
    to that request. Entries beyond ``kv_num_blocks[batch_row]`` are unused.
    """

    kv_num_blocks: torch.Tensor
    """(B,) int32 — number of visible blocks for each batch row."""

    inverse_tables: torch.Tensor
    """(B, num_kv_blocks + 1) int32 — physical-to-logical mappings.

    ``inverse_tables[batch_row, physical_block]`` is that physical block's
    logical position for the request in the batch row. Blocks the request
    does not own contain an out-of-range sentinel, causing the causal mask to
    reject them. The final column represents the scratch block.
    """

    write_map: PagedKVWriteMap


@dataclass(frozen=True)
class BatchMeta:
    """Per-step batch geometry for the continuous-batching path.

    Built once per scheduler step and passed unchanged to every layer —
    per-row history lengths are the same across layers. Carries the same
    facts twice on purpose: `rows` for the per-row bookkeeping (bounds
    validation, deriving `kv_write_map`), the tensors for vectorized
    gathers and scatters (RoPE, slot reads, KV writes). Row order matches
    the `(B, ...)` batch dim everywhere.
    """

    rows: list[tuple[int, int, int]]
    """Per row: (slot_idx, start_pos, num_new)."""

    slots: torch.Tensor
    """(B,) long — each row's slot in the KV pool."""

    start_pos: torch.Tensor
    """(B,) long — each row's first new token position."""

    num_new: torch.Tensor
    """(B,) long — real (unpadded) new tokens per row; >= 1 for real rows.
    0 marks a filler row (batch padded up to a shape bucket): it writes no
    KV (`kv_write_map` skips it by construction), reads slot 0's history
    under the causal mask, and its output row is garbage nobody gathers."""

    positions: torch.Tensor
    """(B, num_new_max) long — start_pos[b] + arange; pad columns unused."""

    num_new_max: int
    """Padded width of this step's input_ids."""

    max_history_len: int
    """The KV span attention reads: >= max(start_pos + num_new) over rows.
    May exceed it when the span is rounded up to a shape bucket (never past
    the slot capacity) — the causal mask fences the over-read."""

    device: torch.device | None = None
    """Where `kv_write_map`'s index tensors are created (None = CPU).
    The other tensor fields stay host-side; advanced indexing accepts
    either, at the cost of an implicit transfer per use."""

    def seed_kv_write_map(self, write_map: KVWriteMap) -> None:
        """Install `write_map` as this meta's `kv_write_map`, bypassing the
        derived construction.

        Exists for CUDA-graph capture (cuda-graphs-design.md §4): a
        recording bakes the *addresses* of the map tensors it saw, so the
        meta that flows through capture must carry the caller's static
        buffers — the derived map would allocate fresh tensors at addresses
        no replay reads. The caller owns the alignment contract that
        derivation normally guarantees, and the seeded map may deliberately
        disagree with `rows` (the graph path pads it to the batch bucket,
        filler entries pointing at the pool's scratch column).

        Seeding must happen before first use: once `kv_write_map` has been
        read (or seeded), kernels and recordings may hold the existing
        tensors, so replacing the map would silently split the two — that
        case raises instead.
        """
        if "kv_write_map" in self.__dict__:
            raise ValueError(
                "kv_write_map is already set (derived or seeded); seeding "
                "after first use would leave stale tensors in flight"
            )
        lengths = {t.shape for t in write_map}
        if len(lengths) != 1 or write_map.row.dim() != 1:
            raise ValueError(
                f"write map columns must be 1-D and aligned, got shapes "
                f"{[tuple(t.shape) for t in write_map]}"
            )
        if any(t.dtype != torch.long for t in write_map):
            raise ValueError(
                f"write map columns must be int64 (index tensors), got "
                f"{[t.dtype for t in write_map]}"
            )
        # cached_property stores through the instance __dict__, which the
        # frozen dataclass's __setattr__ block does not cover — this is the
        # same slot the property itself writes.
        self.__dict__["kv_write_map"] = write_map

    def seed_paged_tables(self, tables: PagedTables) -> None:
        """Install this step's paged table references (paged-kv-plan.md
        §2.5). Same discipline as `seed_kv_write_map`: seed-once (graph
        capture bakes tensor addresses; replacing after first use would
        split what kernels hold from what the meta reports), references
        never copies, and the runtime front's device move must carry the
        seeded tables with the meta (the 40fbcf9 family). Unlike
        `kv_write_map` there is no derived fallback — the meta has no
        block tables to derive from; the scheduler side owns them.
        """
        if "paged_tables" in self.__dict__:
            raise ValueError(
                "paged_tables is already seeded; seeding after first use "
                "would leave stale tensors in flight"
            )
        bt, knb, inv, wm = tables
        if bt.dim() != 2 or knb.dim() != 1 or inv.dim() != 2:
            raise ValueError(
                f"table tensors must be (B, T)/(B,)/(B, P), got dims "
                f"{(bt.dim(), knb.dim(), inv.dim())}"
            )
        if not (bt.shape[0] == knb.shape[0] == inv.shape[0]):
            raise ValueError(
                f"table tensors disagree on B: "
                f"{(bt.shape[0], knb.shape[0], inv.shape[0])}"
            )
        if any(t.dtype != torch.int32 for t in (bt, knb, inv)):
            raise ValueError(
                f"table tensors must be int32 (the BlockMask.from_kv_blocks "
                f"contract), got {[t.dtype for t in (bt, knb, inv)]}"
            )
        lengths = {t.shape for t in wm}
        if len(lengths) != 1 or wm.batch_row.dim() != 1:
            raise ValueError(
                f"write map columns must be 1-D and aligned, got shapes "
                f"{[tuple(t.shape) for t in wm]}"
            )
        if any(t.dtype != torch.long for t in wm):
            raise ValueError(
                f"write map columns must be int64 (index tensors), got "
                f"{[t.dtype for t in wm]}"
            )
        self.__dict__["paged_tables"] = tables

    @property
    def paged_tables(self) -> PagedTables:
        """This step's seeded `PagedTables`. Raises when unseeded: only
        the paged path seeds tables, so reading them off a padded-path
        meta is a wiring bug, not a derivable state."""
        tables = self.__dict__.get("paged_tables")
        if tables is None:
            raise ValueError(
                "this BatchMeta carries no paged tables; the paged "
                "scheduler side seeds them per step (seed_paged_tables) — "
                "there is no derived construction"
            )
        return tables

    @cached_property
    def kv_write_map(self) -> KVWriteMap:
        """The ragged KV write as data instead of control flow.

        Lets `forward_batched` replace its per-row slice-assign loop
        (one dispatch per row per layer) with one gather + one scatter
        per tensor:

            m = meta.kv_write_map
            layer_k[m.slot, m.pos] = keys[m.row, m.off]

        Entries cover only real tokens (off < num_new), so pad columns
        are never read. Destinations are unique — rows own distinct
        slots, offsets are distinct within a row — so the scatter is
        race-free. Cached: built once per step, reused by every layer.
        """
        row_l: list[int] = []
        off_l: list[int] = []
        slot_l: list[int] = []
        pos_l: list[int] = []
        for r, (slot, start, num_new) in enumerate(self.rows):
            for off in range(num_new):
                row_l.append(r)
                off_l.append(off)
                slot_l.append(slot)
                pos_l.append(start + off)

        def as_index(values: list[int]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.long, device=self.device)

        return KVWriteMap(
            row=as_index(row_l), off=as_index(off_l),
            slot=as_index(slot_l), pos=as_index(pos_l),
        )


class AttentionMethod(Protocol):
    def execution_context(self) -> AbstractContextManager:
        """Context the caller holds open around a whole forward's
        *execution* — dispatcher state the method needs live when its
        kernels run, e.g. the SDPA backend-priority pin.

        This lives on the method but is entered by the forward's entry
        points (`Qwen3.forward_batched` for eager callers, the runtime
        front for the compiled path), deliberately OUTSIDE the
        torch.compile-traced region: tracing the context manager plants
        an unserializable call in the graph, which bypasses the
        AOTAutograd/FX caches and re-runs full Inductor codegen on every
        boot (found on the 2026-08-08 5090 round — the entire warm-cache
        Ready bill). Methods with no dispatcher needs return a
        nullcontext.
        """
        ...

    def build_mask(
        self,
        start_pos: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Construct the causal mask this method expects.

        Shape is method-specific: the einsum path wants
        `(seq_len, start_pos + seq_len)` with `True` marking positions to
        mask out. Padded/paged methods will return per-sequence variants.
        """
        ...

    def forward_prefill(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        kv_cache: dict | None,
    ) -> torch.Tensor:
        """Full-prompt path: cache is empty (or None), compute attention
        over the whole sequence and populate the cache.

        Shapes:
          queries: (batch, seq, groups, heads_per_group, head_dim)
          keys:    (batch, seq, groups, head_dim)
          values:  (batch, seq, groups, head_dim)
          returns: (batch, seq, groups, heads_per_group, head_dim)
        """
        ...

    def forward_decode(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        kv_cache: dict,
    ) -> torch.Tensor:
        """Incremental path: cache is populated; append new K/V, compute
        attention of fresh queries against cached + fresh keys.

        Shapes match `forward_prefill`; the `seq` dim on the output matches
        the queries' `seq` (1 for single-token decode, >1 for speculative
        chunks).
        """
        ...

    def build_batched_mask(
        self,
        meta: BatchMeta,
        device: torch.device,
    ) -> object:
        """Mask for one mixed prefill/decode batch, built once per step.

        The mask object is method-opaque: the model builds it here and
        passes it to `forward_batched` unchanged, never inspecting it.
        Padded/sdpa return a (B, num_new_max, max_history_len) bool tensor,
        True = masked, pure per-row causality: mask[b, i, j] =
        j > start_pos[b] + i. That alone covers everything: future tokens,
        stale K/V beyond a row's own history (its hist_len is within the
        masked bound), and pad query rows stay finite (they attend to
        their own earlier keys) — the last-token gather never reads them.
        Callers broadcast over the group/head dims at use. The Flex method
        (Phase 4) returns a `BlockMask` over engine-owned table tensors.
        """
        ...

    def forward_batched(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: object,
        layer_k: torch.Tensor,
        layer_v: torch.Tensor,
        meta: BatchMeta,
    ) -> torch.Tensor:
        """Mixed prefill/decode batch against a preallocated KV pool layer.

        Write each row's new K/V into `layer_k/v[slot, start:start+num_new]`
        (bounds-assert the write — one overlong row must fail loudly, not
        corrupt a neighbor slot), then attend each row's queries against its
        own slot history `[0, start_pos + num_new)`. Vectorize the math
        and the writes — the ragged write goes through `meta.kv_write_map`
        as one gather + scatter per tensor, not a per-row loop.

        Shapes:
          queries: (B, num_new_max, groups, heads_per_group, head_dim), post-RoPE
          keys:    (B, num_new_max, groups, head_dim), post-RoPE
          values:  (B, num_new_max, groups, head_dim)
          mask:    from build_batched_mask, method-opaque (padded/sdpa:
                   (B, num_new_max, max_history_len) bool)
          layer_k: layer i's pool storage, written in place; the shape is
                   pool-layout-defined (padded: (max_batch, max_seq_len + 1,
                   groups, head_dim); paged: flat block rows)
          layer_v: same shape as layer_k
          returns: (B, num_new_max, groups, heads_per_group, head_dim)
        """
        ...
