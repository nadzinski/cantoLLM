"""torch.compile wiring: config, hoists, and trace health (chunk 1).

CPU-only on purpose: Dynamo tracing is device-independent, so fullgraph
regressions, guard leaks, and hoist bugs all surface here. The compiled
arms use backend="eager" (or a counting backend), which runs Dynamo's
tracing and guards without paying Inductor codegen, so the suite stays
fast. Inductor kernel quality, numerics, and graph-capture composition
are 5090 work (torch-compile-design.md §7).
"""

from __future__ import annotations

import pytest
import torch

from cantollm.engine.batching import BatchingConfig
from cantollm.kv_pool import PaddedKVPool
from cantollm.models.attention.padded import PaddedAttentionMethod
from cantollm.models.attention.protocol import BatchMeta
from cantollm.models.attention.sdpa import SDPAAttentionMethod
from cantollm.models.qwen3.model import Qwen3
from cantollm.runtime import ModelRuntime
from cantollm.standard import StandardBackend
from tests.tiny_model import TINY_ARCH, tiny_qwen3_spec

MAX_SEQ = 32
CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def _fresh_dynamo():
    """Dynamo caches artifacts on the CODE OBJECT, shared across every
    model instance in the process, so without a reset the tests here
    pollute each other's artifact counts (and can trip the recompile
    limit, which fullgraph=True turns into a hard error)."""
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def make_pool(max_batch: int = 4) -> PaddedKVPool:
    return PaddedKVPool(
        num_layers=TINY_ARCH["num_transformers"], max_batch=max_batch,
        max_seq_len=MAX_SEQ, num_groups=TINY_ARCH["num_groups"],
        head_dim=TINY_ARCH["head_dim"], dtype=torch.float32, device=CPU,
    )


def make_meta(row_specs: list[tuple[int, int, int]]) -> BatchMeta:
    """row_specs: [(slot, start_pos, num_new)]."""
    start_pos = torch.tensor([r[1] for r in row_specs])
    num_new = torch.tensor([r[2] for r in row_specs])
    num_new_max = int(num_new.max())
    return BatchMeta(
        rows=list(row_specs),
        slots=torch.tensor([r[0] for r in row_specs]),
        start_pos=start_pos,
        num_new=num_new,
        positions=start_pos[:, None] + torch.arange(num_new_max)[None, :],
        num_new_max=num_new_max,
        max_history_len=int((start_pos + num_new).max()),
        device=CPU,
    )


def make_model(attention=None) -> Qwen3:
    torch.manual_seed(1234)
    model = Qwen3(
        qwen3_config=TINY_ARCH,
        attention_method=attention or PaddedAttentionMethod(),
    )
    model.eval()
    return model


def make_runtime(model: Qwen3) -> ModelRuntime:
    return ModelRuntime(
        spec=tiny_qwen3_spec(), device=CPU, model=model,
        tokenizer=None, backend=StandardBackend(model=model, device=CPU),
    )


def ids_for(meta: BatchMeta) -> torch.Tensor:
    torch.manual_seed(7 + meta.num_new_max)
    return torch.randint(
        0, TINY_ARCH["token_count"], (len(meta.rows), meta.num_new_max)
    )


# A small step schedule: prefill, mixed widths, then decode.
STEPS = [
    [(0, 0, 5)],                       # single prefill
    [(0, 5, 1), (1, 0, 8)],            # decode + prefill mixed
    [(0, 6, 1), (1, 8, 1)],            # pure decode
    [(0, 7, 1), (1, 9, 1), (2, 0, 3)],  # decode + late joiner
]


class TestConfig:
    def _base(self, **kw):
        return BatchingConfig(
            max_batch=4, max_seq_len=MAX_SEQ, max_tokens_per_step=8,
            prefill_widths=(8,), kv_bucket=8, batch_buckets=(1, 2, 4), **kw,
        )

    def test_torch_compile_requires_warmup_shapes(self):
        with pytest.raises(ValueError, match="torch_compile requires warmup"):
            self._base(torch_compile=True)

    def test_torch_compile_with_warmup_accepted(self):
        config = self._base(torch_compile=True, warmup_shapes=True)
        assert config.torch_compile
        assert config.torch_compile_strategy == "dynamic"

    def test_strategy_validated(self):
        with pytest.raises(ValueError, match="torch_compile_strategy"):
            self._base(torch_compile_strategy="per-shape")

    def test_batch_bucket_strategy_accepted(self):
        config = self._base(
            torch_compile=True, warmup_shapes=True,
            torch_compile_strategy="batch-bucket",
        )
        assert config.torch_compile_strategy == "batch-bucket"


class TestHoists:
    def test_impl_matches_validated_entry(self):
        """forward_batched == validate + force + forward_batched_impl:
        the split moved code, not behavior."""
        model = make_model()
        pool_a, pool_b = make_pool(), make_pool()
        for specs in STEPS:
            meta_a, meta_b = make_meta(specs), make_meta(specs)
            input_ids = ids_for(meta_a)
            want = model.forward_batched(input_ids, meta_a, pool_a)
            model._validate_batched(meta_b, pool_b)
            _ = meta_b.kv_write_map
            got = model.forward_batched_impl(input_ids, meta_b, pool_b)
            torch.testing.assert_close(got, want, atol=0, rtol=0)
        torch.testing.assert_close(pool_a.k, pool_b.k, atol=0, rtol=0)

    def test_overlong_row_rejected_before_any_write(self):
        """The bounds guarantee moved from the padded write loop to
        _validate_batched; the validate-then-write promise must survive
        the move: fail loudly, pool untouched."""
        model = make_model()
        pool = make_pool()
        meta = make_meta([(0, 0, 4), (1, MAX_SEQ - 2, 4)])  # 30 + 4 > 32
        with pytest.raises(ValueError):
            model.forward_batched(ids_for(meta), meta, pool)
        assert torch.all(pool.k == 0) and torch.all(pool.v == 0)


class TestCompiledRuntime:
    def _twin_runtimes(self, attention_cls):
        eager = make_runtime(make_model(attention_cls()))
        compiled = make_runtime(make_model(attention_cls()))
        compiled.model.load_state_dict(eager.model.state_dict())
        compiled.enable_torch_compile(backend="eager")
        return eager, compiled

    @pytest.mark.parametrize(
        "attention_cls", [PaddedAttentionMethod, SDPAAttentionMethod]
    )
    def test_compiled_path_matches_eager(self, attention_cls):
        """The runtime front's compiled path (hoists + fullgraph=True)
        matches the eager path step for step. fullgraph makes this double
        as the graph-break tripwire: a break raises, loudly, here. The
        sdpa parametrization pins that the sdpa_kernel priority context
        traces clean (the design's §4 hazard)."""
        eager, compiled = self._twin_runtimes(attention_cls)
        pool_e, pool_c = make_pool(), make_pool()
        for specs in STEPS:
            meta_e, meta_c = make_meta(specs), make_meta(specs)
            input_ids = ids_for(meta_e)
            want = eager.forward_batched(input_ids, meta_e, pool_e)
            got = compiled.forward_batched(input_ids, meta_c, pool_c)
            # backend="eager" runs the traced graph with the same kernels,
            # so equality is exact; Inductor tolerance questions are box
            # work.
            torch.testing.assert_close(got, want, atol=0, rtol=0)

    def test_compiled_path_still_validates(self):
        """The hoisted validation runs on the compiled path: an overlong
        row fails loudly before the traced region, pool untouched."""
        _, compiled = self._twin_runtimes(PaddedAttentionMethod)
        pool = make_pool()
        meta = make_meta([(0, MAX_SEQ - 2, 4)])
        with pytest.raises(ValueError):
            compiled.forward_batched(ids_for(meta), meta, pool)
        assert torch.all(pool.k == 0)


class TestStrategies:
    """The §3.2 artifact-count contracts, per strategy. A counting
    backend measures Dynamo compiles; the exact counts are torch details,
    the bounds are the contract."""

    def _compiled_runtime(self, strategy):
        runtime = make_runtime(make_model())
        compiles = []

        def counting_backend(gm, example_inputs):
            compiles.append(1)
            return gm.forward

        runtime.enable_torch_compile(
            strategy=strategy, backend=counting_backend
        )
        return runtime, compiles

    def _decode_sweep(self, runtime):
        pool = make_pool()
        for b in (1, 2, 4):
            for kv in (16, 24, 32):
                meta = make_meta([(r, kv - 1, 1) for r in range(b)])
                runtime.forward_batched(ids_for(meta), meta, pool)

    def test_dynamic_strategy_few_artifacts(self):
        """Marked dims skip automatic dynamic's static stepping stones:
        the 9-shape decode sweep compiles at most 4 artifacts (B=1
        specializes by torch's 0/1 rule; kv promotes once), and marking
        must not error at size-1 dims."""
        runtime, compiles = self._compiled_runtime("dynamic")
        self._decode_sweep(runtime)
        assert len(compiles) <= 4, f"expected few artifacts: {len(compiles)}"

    def test_batch_bucket_strategy_pins_rows(self):
        """One artifact per batch bucket, at least: the batch dim must
        never unify across buckets (that would un-bake the row-count
        constants the strategy exists for). kv stays dynamic within a
        bucket, so the total stays bounded."""
        runtime, compiles = self._compiled_runtime("batch-bucket")
        self._decode_sweep(runtime)
        n = len(compiles)
        assert n >= 3, f"batch dim unified across buckets: {n} artifacts"
        assert n <= 7, f"artifact explosion: {n}"
        # A second sweep over the same shapes must be all cache hits.
        self._decode_sweep(runtime)
        assert len(compiles) == n, "batch-bucket artifacts not reused"

    def test_filler_rows_vary_map_length_without_storm(self):
        """Inside a fixed batch bucket the write map's length tracks the
        REAL row count (fillers skipped by construction), so it changes
        step to step. The dim marking must absorb that without one
        artifact per real-row count; the sole allowed extra is the
        length-1 map (a single real row), which specializes by torch's
        0/1 rule."""
        runtime, compiles = self._compiled_runtime("dynamic")
        pool = make_pool()

        def step(real):
            filler = [(0, 0, 0)] * (4 - real)
            meta = make_meta(
                [(r, 15, 1) for r in range(real)] + filler
            )
            runtime.forward_batched(ids_for(meta), meta, pool)

        for real in (4, 3, 2, 1):
            step(real)
        first_pass = len(compiles)
        assert first_pass <= 2, (
            f"map length recompiles per real-row count: {first_pass}"
        )
        for real in (4, 3, 2, 1):  # repeat: everything must hit
            step(real)
        assert len(compiles) == first_pass


class TestEngineWiring:
    def test_scheduler_from_runtime_enables_compile(self, monkeypatch):
        """config.torch_compile wires runtime.enable_torch_compile with
        the config's strategy, BEFORE the warm-up sweep (the sweep is
        what builds the artifacts behind Ready). Inductor itself is not
        paid here: the enable call is intercepted and re-pointed at the
        eager backend, keeping this a wiring test."""
        from cantollm.engine.batching.engine import scheduler_from_runtime

        runtime = make_runtime(make_model())
        calls = []
        original = ModelRuntime.enable_torch_compile

        def spy(self, strategy="dynamic", backend="inductor"):
            calls.append(strategy)
            original(self, strategy=strategy, backend="eager")

        monkeypatch.setattr(ModelRuntime, "enable_torch_compile", spy)
        config = BatchingConfig(
            max_batch=2, max_seq_len=MAX_SEQ, max_tokens_per_step=8,
            prefill_widths=(8,), kv_bucket=16, batch_buckets=(1, 2),
            warmup_shapes=True, torch_compile=True,
            torch_compile_strategy="batch-bucket",
        )
        scheduler = scheduler_from_runtime(runtime, config)
        assert calls == ["batch-bucket"]
        assert runtime._compiled_batched is not None
        # The warm-up sweep ran through the compiled path without a
        # graph break (fullgraph=True would have raised) and the
        # scheduler is servable.
        assert scheduler.is_idle()


class TestGuardHealth:
    def test_no_recompile_on_row_values(self):
        """The recompile-storm tripwire (design §4): with the hoists in
        place, per-step VALUES (start_pos, slots) must never appear in
        guards. Sweep shapes once, then repeat the same shapes with
        different values: zero new artifacts allowed."""
        model = make_model()
        compiles = []

        def counting_backend(gm, example_inputs):
            compiles.append(1)
            return gm.forward

        fn = torch.compile(
            model.forward_batched_impl, backend=counting_backend,
            fullgraph=True,
        )
        pool = make_pool()

        def run(specs):
            meta = make_meta(specs)
            model._validate_batched(meta, pool)
            _ = meta.kv_write_map
            fn(ids_for(meta), meta, pool)

        with torch.inference_mode():
            for b in (1, 2, 4):
                for kv in (16, 24, 32):
                    run([(r, kv - 1, 1) for r in range(b)])
            after_shapes = len(compiles)
            for b in (1, 2, 4):  # same shapes, shifted values
                for kv in (16, 24, 32):
                    run([(b - 1 - r, kv - 2, 1) for r in range(b)])
        assert len(compiles) == after_shapes, (
            "row values leaked into compile guards: the hoist regressed"
        )
        # Automatic dynamic should keep the artifact count small; the
        # exact number is a torch detail, the ceiling is the contract.
        assert after_shapes <= 5, f"artifact explosion: {after_shapes}"
