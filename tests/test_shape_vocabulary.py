"""Bounded shape vocabulary (Phase 3): quantized prefill widths, KV-span
rounding, batch padding with filler rows, and the startup warm-up sweep.

The stakes: shape-keyed kernel caches (cuDNN SDPA plans, later CUDA graphs)
pay a one-time cost per distinct problem shape, and v1's scheduler produces
a new shape almost every step. With the bucket knobs set, every step's
(batch, width, kv_len) must come from `config.shape_vocabulary()` — and
bucketing must be output-invisible: same requests, same tokens.

Trust model: the unbucketed scheduler is the oracle (itself proven
token-for-token against the sequential engine by test_cb_end_to_end.py).
Everything here runs the real scheduler + tiny-Qwen3 forward on CPU.
"""

from collections import defaultdict

import pytest
import torch

from cantollm.engine.batching import BatchingConfig, default_shape_buckets
from cantollm.engine.batching.engine import scheduler_from_runtime
from cantollm.engine.batching.scheduler import build_batch_meta
from cantollm.engine.batching.shaping import round_up_to, shape_step
from cantollm.engine.batching.warmup import warmup_shape_vocabulary
from cantollm.engine.types import InferenceRequest, SamplingParams
from cantollm.runtime import build_runtime
from tests.tiny_model import tiny_qwen3_spec

GREEDY = SamplingParams.from_temperature_top_p(temperature=0.0, top_p=1.0)

BASE = {"max_batch": 3, "max_seq_len": 64, "max_tokens_per_step": 8}
BUCKETS = {
    "prefill_widths": (4, 8),
    "kv_bucket": 16,
    "batch_buckets": (1, 2, 3),
}

PROMPTS = [
    [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],  # needs chunking
    [31, 32],
    [41, 42, 43, 44, 45],
]


def make_request(rid: str, prompt: list[int], max_tokens: int = 6) -> InferenceRequest:
    return InferenceRequest(
        request_id=rid, prompt_token_ids=list(prompt),
        sampling_params=GREEDY, max_tokens=max_tokens, stop_token_ids=set(),
    )


def make_scheduler(**config_overrides):
    """Real scheduler over tiny-Qwen3 on CPU. Reseeded per build (the tiny
    spec random-inits), so every scheduler sees identical weights."""
    torch.manual_seed(1234)
    runtime = build_runtime(tiny_qwen3_spec(), torch.device("cpu"), attention="padded")
    config = BatchingConfig(**{**BASE, **config_overrides})
    return scheduler_from_runtime(runtime, config)


def drive(scheduler, arrivals: dict[int, list[InferenceRequest]]):
    """Run to completion with requests added before the given step index.
    Returns ({request_id: [token_ids]}, {request_id: finish_reason})."""
    tokens = defaultdict(list)
    finishes = {}
    step_i = 0
    pending = dict(arrivals)
    while pending or not scheduler.is_idle():
        for req in pending.pop(step_i, []):
            scheduler.add_request(req)
        if not scheduler.is_idle():
            for evt in scheduler.step():
                assert evt.error is None, evt.error
                if evt.token_id is not None:
                    tokens[evt.request_id].append(evt.token_id)
                if evt.finish_reason is not None:
                    finishes[evt.request_id] = evt.finish_reason
        step_i += 1
    return dict(tokens), finishes


STAGGERED = lambda: {  # noqa: E731 — arrivals rebuilt fresh per drive
    0: [make_request("a", PROMPTS[0])],
    1: [make_request("b", PROMPTS[1])],
    3: [make_request("c", PROMPTS[2])],
}


class TestConfig:
    def test_knob_validation(self):
        with pytest.raises(ValueError, match="ascending"):
            BatchingConfig(**BASE, prefill_widths=(8, 4))
        with pytest.raises(ValueError, match="max_tokens_per_step"):
            BatchingConfig(**BASE, prefill_widths=(2, 4))  # top < budget
        with pytest.raises(ValueError, match="end at max_batch"):
            BatchingConfig(**BASE, batch_buckets=(1, 2))
        with pytest.raises(ValueError, match="kv_bucket"):
            BatchingConfig(**BASE, kv_bucket=0)
        with pytest.raises(ValueError, match="warmup_shapes"):
            BatchingConfig(**BASE, warmup_shapes=True)  # unbounded

    def test_vocabulary_enumeration(self):
        config = BatchingConfig(**BASE, **BUCKETS)
        vocabulary = config.shape_vocabulary()
        # widths {1, 4, 8} x kv {16, 32, 48, 64} x batches {1, 2, 3},
        # kv >= width holds everywhere here
        assert len(vocabulary) == 3 * 3 * 4
        assert (3, 1, 16) in vocabulary and (1, 8, 64) in vocabulary
        assert all(kv >= w for _, w, kv in vocabulary)

    def test_default_shape_buckets_derivation(self):
        knobs = default_shape_buckets(max_batch=16, max_tokens_per_step=512)
        assert knobs["prefill_widths"] == (128, 256, 512)
        assert knobs["batch_buckets"] == (1, 2, 4, 8, 16)
        assert knobs["kv_bucket"] == 256
        # degenerate sizes stay valid
        tiny = default_shape_buckets(max_batch=3, max_tokens_per_step=8)
        BatchingConfig(max_batch=3, max_seq_len=64, max_tokens_per_step=8, **tiny)

    def test_round_up_to(self):
        assert round_up_to(3, (4, 8)) == 4
        assert round_up_to(4, (4, 8)) == 4
        assert round_up_to(5, (4, 8)) == 8
        with pytest.raises(ValueError):
            round_up_to(9, (4, 8))


class TestShapeStep:
    ROWS_FIXTURE = [(2, 10, 3), (0, 0, 1)]  # (slot, start, num_new)

    def _planned(self, rows_fixture=None):
        from cantollm.engine.batching.scheduler import Row
        from cantollm.engine.batching.types import CBSequence

        rows = []
        for slot, start, num_new in rows_fixture or self.ROWS_FIXTURE:
            seq = CBSequence(
                request_id=f"s{slot}", prompt_token_ids=list(range(70)),
                sampling_params=GREEDY, max_tokens=4, stop_token_ids=set(),
            )
            seq.slot_idx = slot
            seq.position = start
            rows.append(Row(sequence=seq, num_new=num_new, start_pos=start))
        meta = build_batch_meta(rows)
        input_ids = torch.ones(
            (len(rows), meta.num_new_max), dtype=torch.int64
        )
        return input_ids, meta

    def _config(self, **overrides):
        return BatchingConfig(**{**BASE, **BUCKETS, **overrides})

    def test_no_knobs_is_identity(self):
        input_ids, meta = self._planned()
        out_ids, out_meta = shape_step(input_ids, meta, BatchingConfig(**BASE))
        assert out_ids is input_ids and out_meta is meta

    def test_filler_rows_and_width(self):
        input_ids, meta = self._planned()
        out_ids, out = shape_step(
            input_ids, meta, self._config(batch_buckets=(1, 3))
        )
        assert out.rows[2] == (0, 0, 0)  # padded 2 -> batch bucket 3
        assert out.num_new_max == 4      # widest row 3 -> menu width 4
        assert out.positions.shape == (3, 4)
        assert out_ids.shape == (3, 4)
        assert (out_ids[2] == 0).all() and (out_ids[0, :3] == 1).all()
        # fillers contribute no KV writes
        m = out.kv_write_map
        assert len(m.row) == 3 + 1  # only the real tokens
        assert (m.row < 2).all()

    def test_kv_rounding_and_cap(self):
        _, meta = self._planned()
        _, out = shape_step(torch.ones(2, 3, dtype=torch.int64), meta, self._config())
        assert out.max_history_len == 16  # derived 13 -> bucket 16
        input_ids, meta = self._planned([(2, 60, 3), (0, 0, 1)])
        _, out = shape_step(input_ids, meta, self._config())
        assert out.max_history_len == 64  # derived 63 -> 64, capped at slot

    def test_on_vocabulary_geometry_is_identity(self):
        """Steady-state decode at a bucket batch size sits on a vocabulary
        point already — shaping must not rebuild anything."""
        input_ids, meta = self._planned([(0, 15, 1), (1, 31, 1), (2, 15, 1)])
        out_ids, out_meta = shape_step(input_ids, meta, self._config())
        assert out_ids is input_ids and out_meta is meta


class TestOutputEquivalence:
    """Bucketing must be invisible in the outputs: same tokens, same finish
    reasons, same KV bytes — only the step shapes differ."""

    def test_tokens_match_unbucketed(self):
        base_tokens, base_fin = drive(make_scheduler(), STAGGERED())
        bucket_tokens, bucket_fin = drive(make_scheduler(**BUCKETS), STAGGERED())
        assert bucket_tokens == base_tokens
        assert bucket_fin == base_fin
        assert set(base_tokens) == {"a", "b", "c"}  # all three actually ran

    def test_pool_contents_match_unbucketed(self):
        """Fillers write nothing and rounded spans read, never write: after
        identical traffic the pool holds the same K/V. Tolerance, not
        bit-equality: a rounded span changes the attention matmul's blocking,
        which reorders float sums — K/V written downstream of that attention
        (decode tokens' deeper layers) carry ~1e-7 noise. What must be exact
        is the zero region: no write outside the real histories."""
        sched_a = make_scheduler()
        sched_b = make_scheduler(**BUCKETS)
        drive(sched_a, {0: [make_request("x", PROMPTS[0])]})
        drive(sched_b, {0: [make_request("x", PROMPTS[0])]})
        torch.testing.assert_close(sched_b.pool.k, sched_a.pool.k, atol=1e-5, rtol=0)
        torch.testing.assert_close(sched_b.pool.v, sched_a.pool.v, atol=1e-5, rtol=0)
        written = (sched_a.pool.k != 0)
        assert torch.all(sched_b.pool.k[~written] == 0), (
            "bucketed run wrote where the unbucketed run did not"
        )

    def test_forced_fillers_are_harmless(self):
        """batch_buckets starting above 1 force fillers into every step."""
        base_tokens, _ = drive(make_scheduler(), {0: [make_request("solo", PROMPTS[2])]})
        padded_tokens, _ = drive(
            make_scheduler(prefill_widths=(4, 8), kv_bucket=16, batch_buckets=(2, 3)),
            {0: [make_request("solo", PROMPTS[2])]},
        )
        assert padded_tokens == base_tokens


class TestShapeProperty:
    def test_every_step_shape_in_vocabulary(self):
        scheduler = make_scheduler(**BUCKETS)
        vocabulary = set(scheduler.config.shape_vocabulary())
        shapes = []
        inner = scheduler.forward_fn

        def spy(input_ids, meta, pool):
            shapes.append(
                (input_ids.shape[0], input_ids.shape[1], meta.max_history_len)
            )
            return inner(input_ids, meta, pool)

        scheduler.forward_fn = spy
        drive(scheduler, STAGGERED())
        assert shapes, "no steps ran"
        offenders = [s for s in shapes if s not in vocabulary]
        assert not offenders, f"shapes outside the vocabulary: {offenders}"

    def test_unbucketed_scheduler_shape_churn_exists(self):
        """The disease the vocabulary cures: without knobs, consecutive
        decode steps produce distinct kv spans (this is what makes every
        step a fresh cuDNN plan compile)."""
        scheduler = make_scheduler()
        kv_spans = []
        inner = scheduler.forward_fn

        def spy(input_ids, meta, pool):
            kv_spans.append(meta.max_history_len)
            return inner(input_ids, meta, pool)

        scheduler.forward_fn = spy
        drive(scheduler, {0: [make_request("a", PROMPTS[1], max_tokens=6)]})
        assert len(set(kv_spans)) > 3  # grows nearly every step


class TestWarmup:
    def test_warmup_covers_vocabulary_and_writes_nothing(self):
        runtime = build_runtime(
            tiny_qwen3_spec(), torch.device("cpu"), attention="padded"
        )
        config = BatchingConfig(**BASE, **BUCKETS)
        pool = runtime.new_kv_pool(config)
        warmed = warmup_shape_vocabulary(runtime.forward_batched, pool, config)
        assert warmed == len(config.shape_vocabulary())
        assert torch.all(pool.k == 0) and torch.all(pool.v == 0), (
            "warm-up wrote into the pool — filler rows must not write"
        )

    def test_scheduler_build_runs_warmup_then_serves(self):
        scheduler = make_scheduler(**BUCKETS, warmup_shapes=True)
        tokens, finishes = drive(scheduler, {0: [make_request("r", PROMPTS[1])]})
        assert tokens["r"], "engine did not serve after warm-up"
        assert finishes["r"] == "max_tokens"
