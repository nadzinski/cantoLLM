"""CUDA-graph replay for steady-state decode steps (cuda-graphs-design.md).

`GraphedBatchedForward` is a `BatchedForwardFn` that wraps another one
(production: `ModelRuntime.forward_batched`) and, for decode-shaped steps it
has previously captured, replays the whole 28-layer forward as a single
`cudaGraphLaunch` instead of ~1900 per-step CUDA API calls. Everything else
falls through to the wrapped eager path unchanged — the scheduler cannot
tell the two apart, the same altitude call as `shaping.py::shape_step`.

Why a graph can stand in for the forward at all: a captured graph bakes
*shapes and addresses*, not values. Every tensor the recording reads lives
in a static buffer owned by this wrapper; each replayed step first copies
the step's real values into those buffers (the prologue), then replays.
Device-side gathers (`freqs_cis[positions]`, `layer_k[slots]`, the KV
scatter through the write map) read whatever the buffers hold at replay
time, so slots, positions, and write destinations move freely between
replays while the kernel sequence stays frozen.

The write map is where a graph and a padded batch collide (design note §3,
decisions 3-4): eagerly the map holds one entry per *real* new token, so a
batch padded with filler rows has a shorter map — a *shape* change, which
a graph cannot absorb. On the replay path the map is therefore padded to
the bucket: exactly one entry per row, and a filler row's entry writes its
garbage to the pool's scratch position column (`pool.scratch_pos`), which
no gather ever reads. The eager path's map is untouched — padding lives
entirely in this wrapper's marshal, so the two paths stay independently
correct.

Capture cannot reuse the warm-up sweep's all-filler metas for the same
reason: an all-filler step's write map is empty, and a graph captured from
it would bake a zero-length scatter. `capture_decode_shapes` instead builds
an all-real dummy meta per shape (each row writing the last position of its
own slot). The garbage it writes into the pool during capture is harmless:
capture runs behind Ready, before any request exists, and stale pool data
behind the causal mask is already the pool's normal state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cantollm import progress
from cantollm.engine.batching.config import BatchingConfig
from cantollm.engine.batching.types import BatchedForwardFn
from cantollm.kv_pool import PaddedKVPool
from cantollm.models.attention.protocol import BatchMeta, KVWriteMap

_WARM_ITERS = 3
"""Eager runs per shape on a side stream before capture, per the
torch.cuda.graph recipe: first-ever executions do one-time work (allocator
growth, cuDNN plan compile) that must not be recorded."""


@dataclass
class _CapturedShape:
    """One (batch, 1, kv_len) decode shape's recording and static buffers.

    The buffers are the only memory the recording reads for per-step data;
    `out` is the graph-owned logits tensor the model returned during
    capture, refreshed in place by every replay. `map_row`/`map_off` are
    constants for decode steps (row k writes its own offset 0), baked at
    allocation; `map_slot`/`map_pos` change per step and are marshaled.
    `graph` stays None until capture succeeds — an entry without a graph
    never replays.
    """

    input_ids: torch.Tensor      # (B, 1) int64
    slots: torch.Tensor          # (B,) long
    start_pos: torch.Tensor      # (B,) long
    num_new: torch.Tensor        # (B,) long
    positions: torch.Tensor      # (B, 1) long
    map_row: torch.Tensor        # (B,) long, constant arange(B)
    map_off: torch.Tensor        # (B,) long, constant zeros
    map_slot: torch.Tensor       # (B,) long
    map_pos: torch.Tensor        # (B,) long
    graph: torch.cuda.CUDAGraph | None = None
    out: torch.Tensor | None = None


class GraphedBatchedForward:
    """Dispatch decode steps to captured CUDA graphs; everything else eager.

    Satisfies `BatchedForwardFn`, so it drops between the scheduler and
    `ModelRuntime.forward_batched` with zero scheduler changes. The table
    is keyed by the post-bucketing step shape `(B, num_new_max,
    max_history_len)` — the same triple `shape_step` produces — so a hit
    is only possible when the shape vocabulary is on.

    `hits`/`misses` count replayed vs eager forwards. They exist for the
    tripwire lesson from SDPA (sdpa-results.md): a fast path that can fall
    back silently needs a counter that proves it actually ran.
    """

    def __init__(self, inner: BatchedForwardFn, config: BatchingConfig):
        if not config.shapes_bounded:
            raise ValueError(
                "GraphedBatchedForward requires the bounded shape vocabulary "
                "(prefill_widths, kv_bucket, batch_buckets): one graph is "
                "captured per decode shape, and an unbounded vocabulary "
                "cannot be enumerated"
            )
        self.inner = inner
        self.config = config
        self.hits = 0
        self.misses = 0
        self._table: dict[tuple[int, int, int], _CapturedShape] = {}
        self._mem_handle = None  # shared graph memory pool, created lazily

    # ---------------- serving path ----------------

    def __call__(
        self,
        input_ids: torch.Tensor,
        meta: BatchMeta,
        pool: PaddedKVPool,
    ) -> torch.Tensor:
        key = (len(meta.rows), meta.num_new_max, meta.max_history_len)
        entry = self._table.get(key)
        if entry is None or entry.graph is None or not self._replayable(meta, pool):
            self.misses += 1
            return self.inner(input_ids, meta, pool)
        self.hits += 1
        self._marshal(entry, input_ids, meta, pool)
        entry.graph.replay()
        return entry.out

    def _replayable(self, meta: BatchMeta, pool: PaddedKVPool) -> bool:
        """Host-side guard for the replay path, from `meta.rows` only —
        touching the device tensors here would cost the sync the graph
        exists to remove.

        Decode buckets replay: real rows are single-token decodes
        (`num_new == 1`), filler rows (`num_new == 0`) are welcome — their
        padded map entries write to the scratch column. Anything else
        (prefill widths, negative widths) is not a captured shape. The
        bounds check the eager path does in `forward_batched` also lives
        here — replay skips all Python inside the model, so the "one
        overlong row must fail loudly" contract is enforced by rejecting:
        an out-of-bounds row falls through to eager, which raises the same
        error it always has.
        """
        capacity = pool.max_seq_len
        for _, start, num_new in meta.rows:
            if num_new == 0:
                continue
            if num_new != 1 or start + num_new > capacity:
                return False
        return True

    @torch.inference_mode()
    def _marshal(
        self,
        entry: _CapturedShape,
        input_ids: torch.Tensor,
        meta: BatchMeta,
        pool: PaddedKVPool,
    ) -> None:
        """The replay prologue: copy the step's values into the recording's
        static buffers (design note §3, decision 7).

        Replaces the eager path's per-step `.to(device)` moves, which
        create fresh tensors at fresh addresses a graph cannot follow.
        Under `inference_mode` because the static buffers are inference
        tensors (allocated so, to keep compile guards consistent with
        traffic) and `copy_` into an inference tensor is only legal
        inside inference mode.
        The write map needs no host loop: row k writes offset 0 of its own
        row (the baked `map_row`/`map_off` constants), real rows into
        `(slots[k], start_pos[k])`, filler rows into the scratch position
        column — the map padded to the bucket is what makes its length a
        constant of the shape. All the arithmetic runs on the scheduler's
        small CPU tensors; the device sees only `copy_`s.
        """
        real = meta.num_new != 0  # (B,) bool, CPU — fillers to scratch
        entry.input_ids.copy_(input_ids)
        entry.slots.copy_(meta.slots)
        entry.start_pos.copy_(meta.start_pos)
        entry.num_new.copy_(meta.num_new)
        entry.positions.copy_(meta.positions)
        entry.map_slot.copy_(meta.slots)
        entry.map_pos.copy_(
            torch.where(
                real, meta.start_pos,
                torch.full_like(meta.start_pos, pool.scratch_pos),
            )
        )

    # ---------------- capture (behind Ready) ----------------

    def decode_shapes(self) -> list[tuple[int, int]]:
        """The capture set: every (batch, kv_len) with width 1 in the
        vocabulary (design note §3, decision 2)."""
        return [
            (b, kv) for b, w, kv in self.config.shape_vocabulary() if w == 1
        ]

    def capture_decode_shapes(self, pool: PaddedKVPool) -> int:
        """Capture one graph per decode shape into a shared memory pool.

        Runs once at engine build, behind the process split's Ready, after
        the eager warm-up sweep (cuDNN plans must be compiled before
        capture so the recording holds the fused kernel, not the plan
        compile). Largest batches first, so the shared pool is sized by
        the biggest activation footprint and smaller shapes sublet it.
        Returns the number of graphs captured.
        """
        device = pool.device
        if device.type != "cuda":
            raise RuntimeError(
                f"CUDA graphs need a CUDA pool, got device {device}"
            )
        if self._mem_handle is None:
            self._mem_handle = torch.cuda.graph_pool_handle()
        shapes = sorted(self.decode_shapes(), reverse=True)
        for i, (batch, kv_len) in enumerate(shapes):
            self._capture_one(batch, kv_len, pool, device)
            progress.report("capture", i + 1, len(shapes))
        return len(shapes)

    @torch.inference_mode()
    def _alloc_entry(
        self, batch: int, kv_len: int, device: torch.device
    ) -> _CapturedShape:
        """Static buffers for one decode shape, filled with the capture-time
        dummy geometry: row k writes the last position of slot k.

        Allocated under `inference_mode` so the compiled forward sees the
        same tensor dispatch keys during capture warm/record as it does
        from traffic (whose meta tensors become inference tensors in the
        runtime front's move) — otherwise capture would compile a
        duplicate artifact set behind the warm-up sweep's back (same
        guard-leak family the 2026-08-08 A/B caught on warm-up metas)."""
        long = dict(dtype=torch.int64, device=device)
        last = kv_len - 1
        return _CapturedShape(
            input_ids=torch.zeros((batch, 1), **long),
            slots=torch.arange(batch, **long),
            start_pos=torch.full((batch,), last, **long),
            num_new=torch.ones((batch,), **long),
            positions=torch.full((batch, 1), last, **long),
            map_row=torch.arange(batch, **long),
            map_off=torch.zeros((batch,), **long),
            map_slot=torch.arange(batch, **long),
            map_pos=torch.full((batch,), last, **long),
        )

    def _capture_meta(
        self, entry: _CapturedShape, batch: int, kv_len: int,
        device: torch.device,
    ) -> BatchMeta:
        """A BatchMeta whose tensor fields ARE the static buffers.

        The recording will read exactly these addresses forever, so the
        meta that flows through capture must be built from them — a meta
        with fresh tensors would bake the wrong addresses. The write map
        is seeded (`BatchMeta.seed_kv_write_map`) rather than derived for
        the same reason.
        """
        meta = BatchMeta(
            rows=[(r, kv_len - 1, 1) for r in range(batch)],
            slots=entry.slots,
            start_pos=entry.start_pos,
            num_new=entry.num_new,
            positions=entry.positions,
            num_new_max=1,
            max_history_len=kv_len,
            device=device,
        )
        meta.seed_kv_write_map(KVWriteMap(
            row=entry.map_row, off=entry.map_off,
            slot=entry.map_slot, pos=entry.map_pos,
        ))
        return meta

    def _capture_one(
        self, batch: int, kv_len: int, pool: PaddedKVPool,
        device: torch.device,
    ) -> None:
        """Warm, capture, and register one decode shape.

        The dummy rows write garbage K/V into slots 0..batch-1 at position
        kv_len-1 — harmless behind Ready (no request exists yet, and the
        pool's contract already tolerates stale data behind the mask).
        The entry only enters the table after capture succeeds, so a
        failed capture leaves the shape permanently eager rather than
        half-wired.
        """
        entry = self._alloc_entry(batch, kv_len, device)
        meta = self._capture_meta(entry, batch, kv_len, device)

        # Warm on a side stream (torch.cuda.graph recipe): one-time work —
        # allocator growth, lazy kernel/plan setup for this exact shape —
        # happens here, not inside the recording.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(_WARM_ITERS):
                self.inner(entry.input_ids, meta, pool)
        torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._mem_handle):
            out = self.inner(entry.input_ids, meta, pool)
        entry.graph = graph
        entry.out = out
        self._table[(batch, 1, kv_len)] = entry
