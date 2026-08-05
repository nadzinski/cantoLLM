"""GraphedBatchedForward: dispatch, marshaling, and capture/replay.

The dispatch/marshal/guard logic is device-agnostic and tested on CPU with
a fake inner forward; real capture+replay needs CUDA and lives in
TestCaptureReplayCUDA (skipped off-CUDA, run on the 5090 per
cuda-graphs-design.md §7).
"""

import pytest
import torch

from cantollm.engine.batching.config import BatchingConfig
from cantollm.engine.batching.graphs import GraphedBatchedForward
from cantollm.kv_pool import PaddedKVPool
from cantollm.models.attention.protocol import BatchMeta

CONFIG = BatchingConfig(
    max_batch=4,
    max_seq_len=512,
    max_tokens_per_step=512,
    prefill_widths=(512,),
    kv_bucket=256,
    batch_buckets=(1, 2, 4),
)


def make_pool(device="cpu"):
    return PaddedKVPool(
        num_layers=1, max_batch=4, max_seq_len=512, num_groups=1,
        head_dim=2, dtype=torch.float32, device=torch.device(device),
    )


def make_meta(rows, kv_len, width=1):
    """BatchMeta from (slot, start, num_new) specs, mirroring
    build_batch_meta but with an explicit (bucketed) kv_len."""
    start = torch.tensor([r[1] for r in rows])
    num = torch.tensor([r[2] for r in rows])
    return BatchMeta(
        rows=rows,
        slots=torch.tensor([r[0] for r in rows]),
        start_pos=start,
        num_new=num,
        positions=start[:, None] + torch.arange(width)[None, :],
        num_new_max=width,
        max_history_len=kv_len,
    )


class FakeInner:
    """Records calls, returns logits encoding the call count."""

    def __init__(self):
        self.calls = []

    def __call__(self, input_ids, meta, pool):
        self.calls.append((input_ids, meta, pool))
        return torch.full((len(meta.rows), 7), float(len(self.calls)))


class TestDispatch:
    def test_unknown_shape_delegates_eagerly(self):
        inner = FakeInner()
        wrapper = GraphedBatchedForward(inner, CONFIG)
        pool = make_pool()
        meta = make_meta([(0, 3, 1), (1, 5, 1)], kv_len=256)
        ids = torch.tensor([[11], [12]])

        out = wrapper(ids, meta, pool)

        assert len(inner.calls) == 1
        assert inner.calls[0] == (ids, meta, pool)
        assert torch.equal(out, torch.full((2, 7), 1.0))
        assert (wrapper.hits, wrapper.misses) == (0, 1)

    def test_entry_without_graph_never_replays(self):
        """A table entry whose capture did not complete must stay eager —
        the graph=None guard, not an error."""
        inner = FakeInner()
        wrapper = GraphedBatchedForward(inner, CONFIG)
        wrapper._table[(2, 1, 256)] = wrapper._alloc_entry(
            2, 256, torch.device("cpu")
        )
        meta = make_meta([(0, 3, 1), (1, 5, 1)], kv_len=256)

        wrapper(torch.tensor([[11], [12]]), meta, make_pool())

        assert len(inner.calls) == 1
        assert (wrapper.hits, wrapper.misses) == (0, 1)

    def test_requires_bounded_shapes(self):
        with pytest.raises(ValueError, match="bounded shape vocabulary"):
            GraphedBatchedForward(
                FakeInner(),
                BatchingConfig(max_batch=4, max_seq_len=512,
                               max_tokens_per_step=512),
            )

    def test_decode_shapes_are_the_width_one_vocabulary(self):
        wrapper = GraphedBatchedForward(FakeInner(), CONFIG)
        assert sorted(wrapper.decode_shapes()) == [
            (b, kv) for b in (1, 2, 4) for kv in (256, 512)
        ]


class TestReplayGuard:
    def setup_method(self):
        self.wrapper = GraphedBatchedForward(FakeInner(), CONFIG)
        self.pool = make_pool()

    def test_all_real_decode_rows_pass(self):
        meta = make_meta([(0, 3, 1), (1, 5, 1)], kv_len=256)
        assert self.wrapper._replayable(meta, self.pool)

    def test_filler_padded_bucket_passes(self):
        """A batch padded with filler rows replays: their padded map
        entries write to the scratch column instead of anywhere real."""
        meta = make_meta([(0, 3, 1), (0, 0, 0)], kv_len=256)
        assert self.wrapper._replayable(meta, self.pool)

    def test_all_filler_step_passes(self):
        """All-filler (warm-up-shaped) steps are valid replays too: every
        entry writes scratch, the output is garbage nobody reads."""
        meta = make_meta([(0, 0, 0), (0, 0, 0)], kv_len=256)
        assert self.wrapper._replayable(meta, self.pool)

    def test_prefill_row_falls_back(self):
        meta = make_meta([(0, 0, 3), (1, 5, 1)], kv_len=256, width=3)
        assert not self.wrapper._replayable(meta, self.pool)

    def test_out_of_bounds_row_falls_back_to_the_loud_eager_error(self):
        """Replay skips the model's Python bounds check, so the wrapper
        rejects and lets eager raise, preserving the fail-loudly contract."""
        meta = make_meta([(0, 512, 1)], kv_len=512)
        assert not self.wrapper._replayable(meta, self.pool)


class TestMarshal:
    def test_static_buffers_take_the_steps_values(self):
        wrapper = GraphedBatchedForward(FakeInner(), CONFIG)
        entry = wrapper._alloc_entry(2, 256, torch.device("cpu"))
        meta = make_meta([(3, 17, 1), (1, 40, 1)], kv_len=256)
        ids = torch.tensor([[101], [202]])

        wrapper._marshal(entry, ids, meta, make_pool())

        assert torch.equal(entry.input_ids, ids)
        assert torch.equal(entry.slots, torch.tensor([3, 1]))
        assert torch.equal(entry.start_pos, torch.tensor([17, 40]))
        assert torch.equal(entry.num_new, torch.tensor([1, 1]))
        assert torch.equal(entry.positions, torch.tensor([[17], [40]]))
        # decode write map: row k writes offset 0 into (slot k, start k)
        assert torch.equal(entry.map_slot, torch.tensor([3, 1]))
        assert torch.equal(entry.map_pos, torch.tensor([17, 40]))
        assert torch.equal(entry.map_row, torch.tensor([0, 1]))
        assert torch.equal(entry.map_off, torch.tensor([0, 0]))

    def test_filler_rows_route_to_the_scratch_column(self):
        wrapper = GraphedBatchedForward(FakeInner(), CONFIG)
        entry = wrapper._alloc_entry(4, 256, torch.device("cpu"))
        pool = make_pool()
        meta = make_meta(
            [(3, 17, 1), (1, 40, 1), (0, 0, 0), (0, 0, 0)], kv_len=256
        )
        ids = torch.tensor([[101], [202], [0], [0]])

        wrapper._marshal(entry, ids, meta, pool)

        # real rows keep their destinations; fillers park in scratch
        assert torch.equal(
            entry.map_pos, torch.tensor([17, 40, 512, 512])
        )
        assert entry.map_pos[2] == pool.scratch_pos

    def test_padded_scatter_never_touches_real_positions(self):
        """The invariant the scratch column exists for: apply the padded
        map's scatter (exactly what the captured graph replays) and check
        filler writes land only in the scratch column, leaving every real
        position of every slot untouched."""
        wrapper = GraphedBatchedForward(FakeInner(), CONFIG)
        entry = wrapper._alloc_entry(4, 256, torch.device("cpu"))
        pool = make_pool()
        meta = make_meta(
            [(3, 17, 1), (1, 40, 1), (0, 0, 0), (0, 0, 0)], kv_len=256
        )
        wrapper._marshal(entry, torch.zeros((4, 1), dtype=torch.int64),
                         meta, pool)

        layer_k, _ = pool.layer(0)
        keys = (torch.arange(4, dtype=torch.float32) + 10)[
            :, None, None, None
        ].expand(4, 1, 1, 2)
        layer_k[entry.map_slot, entry.map_pos] = keys[
            entry.map_row, entry.map_off
        ]

        # real writes landed where the rows said
        assert torch.all(layer_k[3, 17] == 10.0)
        assert torch.all(layer_k[1, 40] == 11.0)
        # both fillers wrote scratch (last writer wins there, don't care)
        assert torch.all(layer_k[0, pool.scratch_pos] == 13.0)
        # and every REAL position of the fillers' slot 0 is untouched
        assert torch.all(layer_k[0, : pool.max_seq_len] == 0.0)

    def test_capture_meta_aliases_the_static_buffers(self):
        """The meta that flows through capture must read the recording's
        buffers — fresh tensors would bake the wrong addresses."""
        wrapper = GraphedBatchedForward(FakeInner(), CONFIG)
        entry = wrapper._alloc_entry(2, 256, torch.device("cpu"))
        meta = wrapper._capture_meta(entry, 2, 256, torch.device("cpu"))

        assert meta.slots is entry.slots
        assert meta.positions is entry.positions
        m = meta.kv_write_map  # pre-seeded: must not rebuild from rows
        assert m.slot is entry.map_slot
        assert m.pos is entry.map_pos
        # and the dummy geometry is a valid all-real decode step
        assert meta.num_new_max == 1
        assert meta.max_history_len == 256
        assert all(n == 1 for _, _, n in meta.rows)


class TestEngineWiring:
    def test_scheduler_from_runtime_rejects_graphs_off_cuda(self):
        """The engine build must fail loudly, not silently serve eager,
        when cuda_graphs is on for a non-CUDA device (the CLI defaults
        never produce this; an explicit config can)."""
        import cantollm.engine  # noqa: F401  (engine must import before runtime)
        from cantollm.engine.batching.engine import scheduler_from_runtime
        from cantollm.runtime import build_runtime
        from tests.tiny_model import tiny_qwen3_spec

        runtime = build_runtime(
            tiny_qwen3_spec(), torch.device("cpu"), attention="padded"
        )
        config = BatchingConfig(
            max_batch=4, max_seq_len=64, max_tokens_per_step=64,
            prefill_widths=(64,), kv_bucket=32, batch_buckets=(1, 2, 4),
            warmup_shapes=True, cuda_graphs=True,
        )
        with pytest.raises(RuntimeError, match="CUDA"):
            scheduler_from_runtime(runtime, config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
class TestCaptureReplayCUDA:
    """Real capture + replay against the tiny test model (5090 protocol,
    cuda-graphs-design.md §7). Correctness bar: replay logits equal eager
    logits for the same step values — same kernels, same order, so exact
    equality, not a tolerance."""

    def _build(self):
        import cantollm.engine  # noqa: F401  (engine must import before runtime)
        from cantollm.runtime import build_runtime
        from tests.tiny_model import tiny_qwen3_spec

        device = torch.device("cuda")
        runtime = build_runtime(
            tiny_qwen3_spec(), device, attention="padded"
        )
        config = BatchingConfig(
            max_batch=4, max_seq_len=64, max_tokens_per_step=64,
            prefill_widths=(64,), kv_bucket=32, batch_buckets=(1, 2, 4),
        )
        pool = runtime.new_kv_pool(config)
        wrapper = GraphedBatchedForward(runtime.forward_batched, config)
        return runtime, config, pool, wrapper

    def test_capture_then_replay_matches_eager(self):
        runtime, config, pool, wrapper = self._build()
        captured = wrapper.capture_decode_shapes(pool)
        assert captured == len(wrapper.decode_shapes())

        # a real decode step: 2 rows (bucketed shape (2, 1, 32))
        meta = make_meta([(0, 3, 1), (1, 7, 1)], kv_len=32)
        ids = torch.tensor([[5], [9]])
        replayed = wrapper(ids, meta, pool).clone()
        assert wrapper.hits == 1

        eager = runtime.forward_batched(ids, meta, pool)
        assert torch.equal(replayed, eager)

    def test_replay_tracks_new_values(self):
        """Two replays with different tokens/slots must differ: the graph
        reads the buffers, not the capture-time values."""
        runtime, config, pool, wrapper = self._build()
        wrapper.capture_decode_shapes(pool)

        first = wrapper(
            torch.tensor([[5], [9]]),
            make_meta([(0, 3, 1), (1, 7, 1)], kv_len=32), pool,
        ).clone()
        second = wrapper(
            torch.tensor([[6], [2]]),
            make_meta([(2, 4, 1), (3, 8, 1)], kv_len=32), pool,
        ).clone()

        assert wrapper.hits == 2
        assert not torch.equal(first, second)

    def test_scheduler_from_runtime_installs_and_captures(self):
        """The assembled engine: warm-up sweep, then capture, then a
        graphed forward_fn the scheduler drives — and real decode steps
        actually replay (the hit counter is the proof)."""
        import cantollm.engine  # noqa: F401
        from cantollm.engine.batching.engine import scheduler_from_runtime
        from cantollm.engine.batching.graphs import GraphedBatchedForward
        from cantollm.engine.types import InferenceRequest, SamplingParams
        from cantollm.runtime import build_runtime
        from tests.tiny_model import tiny_qwen3_spec

        runtime = build_runtime(
            tiny_qwen3_spec(), torch.device("cuda"), attention="padded"
        )
        config = BatchingConfig(
            max_batch=4, max_seq_len=64, max_tokens_per_step=64,
            prefill_widths=(64,), kv_bucket=32, batch_buckets=(1, 2, 4),
            warmup_shapes=True, cuda_graphs=True,
        )
        sched = scheduler_from_runtime(runtime, config)
        assert isinstance(sched.forward_fn, GraphedBatchedForward)
        assert len(sched.forward_fn._table) == len(
            sched.forward_fn.decode_shapes()
        )

        greedy = SamplingParams.from_temperature_top_p(
            temperature=0.0, top_p=1.0
        )
        sched.add_request(InferenceRequest(
            request_id="a", prompt_token_ids=[3, 5, 7],
            sampling_params=greedy, max_tokens=6, stop_token_ids=set(),
        ))
        while not sched.is_idle():
            sched.step()
        # prefill steps miss (width 64 is not captured); decode steps hit
        assert sched.forward_fn.hits > 0

    def test_filler_padded_replay_matches_eager(self):
        """A bucket-padded step (real + filler rows) replays and matches
        eager exactly on the real rows, without corrupting the pool."""
        runtime, config, pool, wrapper = self._build()
        wrapper.capture_decode_shapes(pool)

        meta = make_meta([(0, 3, 1), (0, 0, 0)], kv_len=32)
        ids = torch.tensor([[5], [0]])
        replayed = wrapper(ids, meta, pool).clone()
        assert wrapper.hits == 1

        pool_after_replay = pool.k[:, :, : pool.max_seq_len].clone()
        eager = runtime.forward_batched(ids, meta, pool)
        # real row's logits identical; filler row's are garbage on both
        # paths and not compared
        assert torch.equal(replayed[0], eager[0])
        # the replayed scatter left every real pool position exactly as
        # the eager step then wrote it (same destinations, same values)
        assert torch.equal(
            pool_after_replay, pool.k[:, :, : pool.max_seq_len]
        )
