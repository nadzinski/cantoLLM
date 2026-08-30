"""P4 chunk 6 gates: paged vocabulary, paged warm-up, mask caching, and
the compile integration (paged-kv-plan.md §5.6).

The two headline gates:

  - the recompile counter: on the paged path, kv length is table VALUES
    over maximal tensors (§2.6), so once a (batch, width) family is warm,
    steps at new kv lengths, permuted tables, and shifted starts must
    compile NOTHING new;
  - mask constructions per step: `BlockMask.from_kv_blocks` allocates,
    and chunk 8's graph capture bakes addresses, so after warm-up the
    per-family cached masks must serve every step (§4's counter).

Alongside them: the paged vocabulary drops the kv axis (with the
standard-bench-geometry counts §6's prediction 2 wants recorded), the
paged warm-up writes only into the scratch block while filling the same
persistent `PagedStepState` buffers traffic uses, and the whole
`forward_batched_impl` traces with FlexAttention (the model-level compile
wiring the chunk-4 CUDA twin deliberately deferred here).

Everything CPU with backend="eager" or a counting backend, per
test_torch_compile.py's rationale: Dynamo tracing and guards are
device-independent. `TestPagedCompiledCUDA` is the chunk's CUDA-skipif
twin, validated on the 5090 in the chunk-7 round.
"""

from __future__ import annotations

import pytest
import torch

from cantollm.engine.batching import BatchingConfig, default_shape_buckets
from cantollm.engine.batching.engine import scheduler_from_runtime
from cantollm.engine.batching.paging import PagedStepState
from cantollm.engine.batching.warmup import warmup_shape_vocabulary
from cantollm.kv_pool import PaddedKVPool, PagedKVPool
from cantollm.models.attention import (
    BatchMeta,
    FlexAttentionMethod,
    PaddedAttentionMethod,
)
from cantollm.models.qwen3.model import Qwen3
from cantollm.runtime import ModelRuntime
from cantollm.standard import StandardBackend
from tests.test_shape_vocabulary import drive, make_request
from tests.tiny_model import TINY_ARCH, tiny_qwen3_spec

BLOCK = 4
MAX_SEQ = 32
NUM_BLOCKS = 16          # allocatable; scratch sits past
MAX_BLOCKS_PER_SEQ = MAX_SEQ // BLOCK
CPU = torch.device("cpu")

PROMPT = [31, 32, 33, 34, 35]


@pytest.fixture(autouse=True)
def _fresh_dynamo():
    """Dynamo caches artifacts on the code object, shared process-wide;
    without a reset the counting tests pollute each other."""
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def make_flex_model() -> Qwen3:
    torch.manual_seed(1234)
    model = Qwen3(
        qwen3_config=TINY_ARCH,
        attention_method=FlexAttentionMethod(block_size=BLOCK),
    )
    return model.eval()


def make_runtime(model: Qwen3, device: torch.device = CPU) -> ModelRuntime:
    return ModelRuntime(
        spec=tiny_qwen3_spec(), device=device, model=model,
        tokenizer=None, backend=StandardBackend(model=model, device=device),
    )


def make_paged_pool(device: torch.device = CPU) -> PagedKVPool:
    return PagedKVPool(
        num_layers=TINY_ARCH["num_transformers"], num_kv_blocks=NUM_BLOCKS,
        block_size=BLOCK, max_seq_len=MAX_SEQ,
        num_groups=TINY_ARCH["num_groups"], head_dim=TINY_ARCH["head_dim"],
        dtype=torch.float32, device=device,
    )


def make_state(
    model: Qwen3, max_rows: int = 4, device: torch.device = CPU
) -> PagedStepState:
    """A traffic-shaped step state: the model's own attention method is
    the mask builder, exactly as `scheduler_from_runtime` wires it."""
    return PagedStepState(
        max_rows=max_rows, max_blocks_per_seq=MAX_BLOCKS_PER_SEQ,
        num_kv_blocks=NUM_BLOCKS, device=device,
        mask_builder=model.attention_method.build_family_mask,
    )


def paged_meta_via_state(
    state: PagedStepState,
    row_specs: list[tuple[int, int]],
    tables: list[list[int]],
    block_size: int = BLOCK,
) -> BatchMeta:
    """A step meta seeded through `state.fill`, the production route:
    persistent buffers rewritten in place, family mask attached.
    row_specs: [(start_pos, num_new)]."""
    rows = [(0, start, num_new) for start, num_new in row_specs]
    start_pos = torch.tensor([r[1] for r in rows])
    num_new = torch.tensor([r[2] for r in rows])
    num_new_max = int(num_new.max())
    meta = BatchMeta(
        rows=rows,
        slots=torch.tensor([r[0] for r in rows]),
        start_pos=start_pos,
        num_new=num_new,
        positions=start_pos[:, None] + torch.arange(num_new_max)[None, :],
        num_new_max=num_new_max,
        max_history_len=int((start_pos + num_new).max()),
    )
    meta.seed_paged_tables(
        state.fill(meta.rows, tables, block_size, num_new_max)
    )
    return meta


def ids_for(meta: BatchMeta, fill_value: int = 7) -> torch.Tensor:
    return torch.full(
        (len(meta.rows), meta.num_new_max), fill_value, dtype=torch.int64
    )


def paged_config(**overrides) -> BatchingConfig:
    """Paged bucket config with NO kv_bucket: under paged the kv axis is
    a value, so the vocabulary is bounded without it."""
    kwargs = dict(
        max_batch=2, max_seq_len=MAX_SEQ, max_tokens_per_step=8,
        prefill_widths=(8,), batch_buckets=(1, 2),
        paged_kv=True, block_size=BLOCK,
    )
    kwargs.update(overrides)
    return BatchingConfig(**kwargs)


class TestPagedVocabulary:
    def test_kv_axis_dropped(self):
        config = paged_config(prefill_widths=(4, 8))
        vocab = config.shape_vocabulary()
        # One entry per (batch, width) family; the kv element is the
        # constant logical bound, not a swept axis.
        assert vocab == [
            (b, w, MAX_SEQ) for b in (1, 2) for w in (1, 4, 8)
        ]

    def test_bounded_without_kv_bucket_only_under_paged(self):
        assert paged_config().shapes_bounded
        paged_config(warmup_shapes=True)  # accepted
        padded_kwargs = dict(
            max_batch=2, max_seq_len=MAX_SEQ, max_tokens_per_step=8,
            prefill_widths=(8,), batch_buckets=(1, 2),
        )
        assert not BatchingConfig(**padded_kwargs).shapes_bounded
        with pytest.raises(ValueError, match="warmup_shapes"):
            BatchingConfig(**padded_kwargs, warmup_shapes=True)

    def test_standard_bench_geometry_counts(self):
        """The §6 prediction-2 structural measurement, recorded in the
        chunk log: sweep shapes 315 -> 20 and decode shapes (the future
        graph keys) 80 -> 5 at the standard 5090 geometry."""
        knobs = default_shape_buckets(max_batch=16, max_tokens_per_step=512)
        base = dict(
            max_batch=16, max_seq_len=4096, max_tokens_per_step=512, **knobs
        )
        padded_vocab = BatchingConfig(**base).shape_vocabulary()
        assert len(padded_vocab) == 315
        assert sum(1 for _, w, _ in padded_vocab if w == 1) == 80
        # block_size stays the served default 64; kv_bucket 256 stays set
        # and inert, the mixed-config shape §9.5 wanted validated.
        paged_vocab = BatchingConfig(
            **base, paged_kv=True
        ).shape_vocabulary()
        assert len(paged_vocab) == 20
        assert sum(1 for _, w, _ in paged_vocab if w == 1) == 5


class TestMaskCache:
    def test_mask_built_once_per_family_and_reused(self):
        model = make_flex_model()
        state = make_state(model)
        m1 = state.fill([(0, 4, 1)], [[0, 1]], BLOCK, 1).mask
        m2 = state.fill([(0, 9, 1)], [[2, 5, 3]], BLOCK, 1).mask
        assert m2 is m1, "same (1, 1) family must reuse its mask"
        wide = state.fill(
            [(0, 12, 1), (0, 2, 1)], [[0, 1, 2, 6], [4]], BLOCK, 1
        ).mask
        assert wide is not m1, "(2, 1) is a different family"
        assert model.attention_method.mask_constructions == 2
        # Steady state: many more steps, zero new constructions.
        for start in (13, 17, 21):
            state.fill(
                [(0, start, 1), (0, 3, 1)],
                [[0, 1, 2, 6, 7, 8], [4]], BLOCK, 1,
            )
        assert model.attention_method.mask_constructions == 2

    def test_cached_mask_reads_current_step_values(self):
        """The stale-closure trap the persistent start_pos buffer exists
        for: the SAME cached mask must be correct at different starts,
        kv lengths, and table contents, against the padded oracle."""
        torch.manual_seed(1234)
        oracle = Qwen3(
            qwen3_config=TINY_ARCH, attention_method=PaddedAttentionMethod()
        ).eval()
        model = make_flex_model()
        model.load_state_dict(oracle.state_dict())
        state = make_state(model)
        paged_pool = make_paged_pool()
        padded_pool = PaddedKVPool(
            num_layers=TINY_ARCH["num_transformers"], max_batch=2,
            max_seq_len=MAX_SEQ, num_groups=TINY_ARCH["num_groups"],
            head_dim=TINY_ARCH["head_dim"], dtype=torch.float32, device=CPU,
        )
        table = [3, 6]

        def oracle_step(start, toks):
            specs = [(0, start, len(toks))]
            start_t = torch.tensor([start])
            meta = BatchMeta(
                rows=specs, slots=torch.tensor([0]), start_pos=start_t,
                num_new=torch.tensor([len(toks)]),
                positions=start_t[:, None] + torch.arange(len(toks))[None, :],
                num_new_max=len(toks), max_history_len=start + len(toks),
            )
            ids = torch.tensor([toks], dtype=torch.int64)
            with torch.inference_mode():
                return oracle.forward_batched(ids, meta, padded_pool)

        def flex_step(start, toks):
            meta = paged_meta_via_state(state, [(start, len(toks))], [table])
            ids = torch.tensor([toks], dtype=torch.int64)
            with torch.inference_mode():
                return model.forward_batched(ids, meta, paged_pool)

        want = oracle_step(0, PROMPT)
        got = flex_step(0, PROMPT)
        torch.testing.assert_close(got, want, atol=1e-5, rtol=0)
        for step in range(3):
            start = len(PROMPT) + step
            want = oracle_step(start, [77])
            got = flex_step(start, [77])
            torch.testing.assert_close(got, want, atol=1e-5, rtol=0)
        # One (1, 5) prefill mask, one (1, 1) decode mask, reused twice.
        assert model.attention_method.mask_constructions == 2


class TestPagedWarmup:
    def test_warmup_covers_families_writes_only_scratch_caches_masks(self):
        model = make_flex_model()
        runtime = make_runtime(model)
        config = paged_config(warmup_shapes=True)
        pool = runtime.new_kv_pool(config)
        state = make_state(model, max_rows=config.max_batch)
        warmed = warmup_shape_vocabulary(
            runtime.forward_batched, pool, config, paged_state=state
        )
        vocabulary = config.shape_vocabulary()
        assert warmed == len(vocabulary) == 4  # {1,2} x {1,8}
        # Every family's mask was built behind Ready, once.
        assert len(state.masks) == len(vocabulary)
        assert model.attention_method.mask_constructions == len(vocabulary)
        # Writes parked on the scratch block only; the allocatable pool
        # stays untouched (the padded sweep's scratch-column invariant,
        # in flat-pool clothes).
        scratch_lo = NUM_BLOCKS * BLOCK
        for k, v in zip(pool.k_layers, pool.v_layers):
            assert torch.all(k[:scratch_lo] == 0)
            assert torch.all(v[:scratch_lo] == 0)
        assert any(
            torch.any(k[scratch_lo:] != 0) for k in pool.k_layers
        ), "warm-up never exercised the paged scatter"

    def test_scheduler_build_warms_paged_and_serves(self):
        """scheduler_from_runtime with a paged config: the trio is built
        before the sweep, the sweep fills the traffic buffers, and
        traffic then constructs zero masks (the §4 counter, engine
        level)."""
        model = make_flex_model()
        runtime = make_runtime(model)
        scheduler = scheduler_from_runtime(
            runtime, paged_config(warmup_shapes=True)
        )
        method = model.attention_method
        assert len(scheduler.paged_state.masks) == 4
        built_behind_ready = method.mask_constructions
        tokens, finishes = drive(
            scheduler, {0: [make_request("r", PROMPT, max_tokens=6)]}
        )
        assert tokens["r"], "engine did not serve after paged warm-up"
        assert method.mask_constructions == built_behind_ready, (
            "traffic constructed a BlockMask; the step loop must only "
            "write ints into preallocated tensors (paged-kv-plan.md §2.6)"
        )

    def test_cuda_graphs_with_paged_refused_until_chunk_8(self):
        model = make_flex_model()
        runtime = make_runtime(model)
        with pytest.raises(RuntimeError, match="chunk 8"):
            scheduler_from_runtime(
                runtime,
                paged_config(
                    warmup_shapes=True, cuda_graphs=True, torch_compile=True
                ),
            )


class TestCompiledPagedRuntime:
    def _twin_runtimes(self):
        eager_model, compiled_model = make_flex_model(), make_flex_model()
        compiled_model.load_state_dict(eager_model.state_dict())
        eager = make_runtime(eager_model)
        compiled = make_runtime(compiled_model)
        compiled.enable_torch_compile(backend="eager")
        return eager, compiled

    def test_compiled_flex_matches_eager(self):
        """forward_batched_impl traced WHOLE with FlexAttention (the
        wiring the chunk-4 twin deferred here), through the production
        arrangement: state-filled tables, cached masks, the paged hoists.
        backend="eager" runs the traced graph with the same kernels, so
        equality is exact; fullgraph=True doubles as the graph-break
        tripwire."""
        eager, compiled = self._twin_runtimes()
        arms = [
            (eager, make_paged_pool(), make_state(eager.model)),
            (compiled, make_paged_pool(), make_state(compiled.model)),
        ]
        steps = [
            # (row_specs, tables, ids): prefill, mixed widths, decode.
            ([(0, 5)], [[3, 6]], [PROMPT]),
            ([(5, 1), (0, 3)], [[3, 6], [1]], [[77], [51, 52, 53]]),
            ([(6, 1), (3, 1)], [[3, 6], [1]], [[78], [54]]),
        ]
        for row_specs, tables, ids in steps:
            outs = []
            for runtime, pool, state in arms:
                meta = paged_meta_via_state(state, row_specs, tables)
                width = meta.num_new_max
                input_ids = torch.zeros(
                    (len(row_specs), width), dtype=torch.int64
                )
                for i, toks in enumerate(ids):
                    input_ids[i, : len(toks)] = torch.tensor(toks)
                outs.append(runtime.forward_batched(input_ids, meta, pool))
            torch.testing.assert_close(outs[1], outs[0], atol=0, rtol=0)

    def test_compiled_paged_hoist_skips_padded_map(self):
        _, compiled = self._twin_runtimes()
        pool, state = make_paged_pool(), make_state(compiled.model)
        meta = paged_meta_via_state(state, [(0, 5)], [[3, 6]])
        compiled.forward_batched(ids_for(meta), meta, pool)
        assert "kv_write_map" not in meta.__dict__, (
            "the compiled hoist derived a padded write map on a paged "
            "meta; nothing reads it"
        )

    def test_kv_length_changes_recompile_nothing(self):
        """THE chunk-6 exit gate. Warm a decode family, a lone-row decode
        family, and a prefill family; then run them at new kv lengths
        (crossing block boundaries), permuted tables, and shifted starts:
        zero new artifacts, zero new masks. On the padded path each kv
        bucket was its own shape; here kv is data."""
        model = make_flex_model()
        runtime = make_runtime(model)
        compiles = []

        def counting_backend(gm, example_inputs):
            compiles.append(1)
            return gm.forward

        runtime.enable_torch_compile(backend=counting_backend)
        pool, state = make_paged_pool(), make_state(model)

        def step(row_specs, tables):
            meta = paged_meta_via_state(state, row_specs, tables)
            runtime.forward_batched(ids_for(meta), meta, pool)

        # Warm: two kv values per family, so any int automatic dynamic
        # wants promoted has its second value behind the warm line.
        step([(3, 1), (5, 1)], [[0], [1, 2]])          # (2,1) decode
        step([(7, 1), (9, 1)], [[0, 3], [1, 2, 4]])
        step([(2, 1)], [[5]])                          # (1,1) lone decode
        step([(6, 1)], [[5, 6]])
        step([(0, 8)], [[0, 1]])                       # (1,8) prefill chunk
        step([(8, 8)], [[0, 1, 2, 3]])
        warm = len(compiles)
        masks_built = model.attention_method.mask_constructions

        # Growth: same families, histories deeper by whole blocks, tables
        # permuted, starts shifted. Values only; nothing may compile.
        step([(11, 1), (19, 1)], [[6, 7, 8], [9, 10, 11, 12, 13]])
        step([(27, 1), (30, 1)],
             [[1, 0, 3, 2, 4, 5, 6, 14], [8, 7, 9, 10, 11, 12, 13, 15]])
        step([(21, 1)], [[5, 6, 7, 8, 9, 10]])
        step([(16, 8)], [[3, 2, 1, 0, 4, 5]])
        assert len(compiles) == warm, (
            "kv length leaked into compile guards: the paged path must "
            "keep kv as table values (paged-kv-plan.md §2.6)"
        )
        assert model.attention_method.mask_constructions == masks_built
        # Ceiling, not exact count (torch details move): a handful of
        # artifacts for three families incl. the specializing lone row.
        assert warm <= 6, f"artifact explosion: {warm}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestPagedCompiledCUDA:
    """The chunk-6 CUDA-skipif twin (validated on the 5090 in the chunk-7
    round): the full batched forward through real Inductor on the paged
    path, cached masks, at compiled-lowering-legal geometry (block 64,
    head_dim 16; the chunk-4 floors). After warm-up,
    `error_on_recompile` turns any kv-length guard leak into a hard
    failure on-device, where the counting backend cannot go."""

    CUDA_BLOCK = 64
    CUDA_MAX_SEQ = 512
    CUDA_NUM_BLOCKS = 12
    CUDA_ARCH = TINY_ARCH | {"head_dim": 16, "max_seq_len": CUDA_MAX_SEQ}
    PROMPT_150 = [(7 * i + 11) % 2048 for i in range(150)]

    def test_compiled_paged_stack_stable_and_matches_oracle(self):
        device = torch.device("cuda")
        torch.manual_seed(1234)
        oracle = Qwen3(
            qwen3_config=self.CUDA_ARCH,
            attention_method=PaddedAttentionMethod(),
        )
        flex = Qwen3(
            qwen3_config=self.CUDA_ARCH,
            attention_method=FlexAttentionMethod(block_size=self.CUDA_BLOCK),
        )
        flex.load_state_dict(oracle.state_dict())
        oracle = oracle.eval().to(device)
        flex = flex.eval().to(device)
        oracle_rt = make_runtime(oracle, device)
        flex_rt = make_runtime(flex, device)
        flex_rt.enable_torch_compile()  # real Inductor

        padded_pool = PaddedKVPool(
            num_layers=self.CUDA_ARCH["num_transformers"], max_batch=2,
            max_seq_len=self.CUDA_MAX_SEQ,
            num_groups=self.CUDA_ARCH["num_groups"],
            head_dim=self.CUDA_ARCH["head_dim"], dtype=torch.float32,
            device=device,
        )
        paged_pool = PagedKVPool(
            num_layers=self.CUDA_ARCH["num_transformers"],
            num_kv_blocks=self.CUDA_NUM_BLOCKS, block_size=self.CUDA_BLOCK,
            max_seq_len=self.CUDA_MAX_SEQ,
            num_groups=self.CUDA_ARCH["num_groups"],
            head_dim=self.CUDA_ARCH["head_dim"], dtype=torch.float32,
            device=device,
        )
        state = PagedStepState(
            max_rows=2,
            max_blocks_per_seq=self.CUDA_MAX_SEQ // self.CUDA_BLOCK,
            num_kv_blocks=self.CUDA_NUM_BLOCKS, device=device,
            mask_builder=flex.attention_method.build_family_mask,
        )
        table = [5, 2, 9]

        def oracle_step(start, toks):
            specs = [(0, start, len(toks))]
            start_t = torch.tensor([start])
            meta = BatchMeta(
                rows=specs, slots=torch.tensor([0]), start_pos=start_t,
                num_new=torch.tensor([len(toks)]),
                positions=start_t[:, None] + torch.arange(len(toks))[None, :],
                num_new_max=len(toks), max_history_len=start + len(toks),
            )
            ids = torch.tensor([toks], dtype=torch.int64)
            return oracle_rt.forward_batched(ids, meta, padded_pool)

        def flex_step(start, toks):
            meta = paged_meta_via_state(
                state, [(start, len(toks))], [table],
                block_size=self.CUDA_BLOCK,
            )
            ids = torch.tensor([toks], dtype=torch.int64)
            return flex_rt.forward_batched(ids, meta, paged_pool)

        want = oracle_step(0, self.PROMPT_150)
        got = flex_step(0, self.PROMPT_150)
        torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)
        # Warm the decode family (two kv values), then forbid recompiles
        # and keep growing the history: kv must stay a value on-device.
        for step_i in range(2):
            start = len(self.PROMPT_150) + step_i
            want = oracle_step(start, [77])
            got = flex_step(start, [77])
            torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)
        prev = torch._dynamo.config.error_on_recompile
        torch._dynamo.config.error_on_recompile = True
        try:
            for step_i in range(2, 5):
                start = len(self.PROMPT_150) + step_i
                want = oracle_step(start, [77])
                got = flex_step(start, [77])
                torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)
        finally:
            torch._dynamo.config.error_on_recompile = prev
        assert flex.attention_method.mask_constructions == 2
