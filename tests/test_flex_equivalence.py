"""Chunk-4 equivalence suite: FlexAttentionMethod + paged pool vs the
padded stack, on a weight-shared tiny Qwen3, float32, CPU eager, greedy.

THIS FILE IS THE DEFINITION OF DONE FOR THE HAND-WRITTEN PAGED ATTEND
(P4 chunk 4: `paged_write_map` in engine/batching/paging.py + the Flex
method in models/attention/flex.py — both index-translation sites).
The translation is complete and every test runs unmarked. Run the suite with

    pytest tests/test_flex_equivalence.py -x

The write-map tests depend only on `paged_write_map`; the model-level tests
exercise the complete paged attention path.

Trust model: the oracle is the PADDED stack (proven against the
sequential einsum path by test_padded_equivalence.py), exercising none of
the new paging machinery. atol=1e-5 for logits (a different attend sums
in a different order), atol=1e-6 for pool-write pins (identical
projection math up to the write). The CPU half is eager only — eager Flex
proves the index translation, not the flex-decoding kernel, which exists
only compiled on CUDA. `TestOnCUDA` is the compiled twin: the chunk's
5090 exit gate (paged-kv-plan.md §4, §5.4).
"""

from contextlib import contextmanager

import pytest
import torch
from torch.nn.attention.flex_attention import flex_attention

import cantollm.models.attention.flex as flex_module
from cantollm.engine.batching.paging import paged_write_map
from cantollm.kv_pool import PaddedKVPool, PagedKVPool
from cantollm.models.attention import (
    BatchMeta,
    FlexAttentionMethod,
    PaddedAttentionMethod,
    PagedTables,
    SDPAAttentionMethod,
)
from cantollm.models.qwen3.model import Qwen3
from cantollm.runtime import move_batch_to
from tests.tiny_model import TINY_ARCH

MAX_SEQ = 32
BLOCK = 4
MAX_BLOCKS_PER_SEQ = MAX_SEQ // BLOCK   # 8 — also the inverse sentinel
NUM_BLOCKS = 16                          # allocatable; scratch sits past

PROMPT_A = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
PROMPT_B = [31, 32, 33, 34, 35]
DECODE_TOKEN = 77


def build_models() -> tuple[Qwen3, Qwen3]:
    """(padded oracle, flex under-test), identical weights."""
    torch.manual_seed(1234)
    oracle = Qwen3(
        qwen3_config=TINY_ARCH, attention_method=PaddedAttentionMethod()
    )
    flex = Qwen3(
        qwen3_config=TINY_ARCH,
        attention_method=FlexAttentionMethod(block_size=BLOCK),
    )
    flex.load_state_dict(oracle.state_dict())
    oracle.eval()
    flex.eval()
    return oracle, flex


def make_padded_pool(max_batch: int = 4) -> PaddedKVPool:
    return PaddedKVPool(
        num_layers=TINY_ARCH["num_transformers"], max_batch=max_batch,
        max_seq_len=MAX_SEQ, num_groups=TINY_ARCH["num_groups"],
        head_dim=TINY_ARCH["head_dim"], dtype=torch.float32,
        device=torch.device("cpu"),
    )


def make_paged_pool() -> PagedKVPool:
    return PagedKVPool(
        num_layers=TINY_ARCH["num_transformers"], num_kv_blocks=NUM_BLOCKS,
        block_size=BLOCK, max_seq_len=MAX_SEQ,
        num_groups=TINY_ARCH["num_groups"], head_dim=TINY_ARCH["head_dim"],
        dtype=torch.float32, device=torch.device("cpu"),
    )


def base_meta(row_specs: list[tuple[int, int, int]]) -> BatchMeta:
    """row_specs: [(slot, start_pos, num_new)] — slot is the padded arm's
    field; the paged path carries 0 there."""
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
    )


def paged_meta(
    row_specs: list[tuple[int, int]], tables: list[list[int]],
    block: int = BLOCK, max_blocks_per_seq: int = MAX_BLOCKS_PER_SEQ,
    num_blocks: int = NUM_BLOCKS,
) -> BatchMeta:
    """A seeded paged meta. row_specs: [(start_pos, num_new)]; tables[r]
    is row r's block table, covering at least its visible history. The
    geometry defaults are the CPU suite's; `TestOnCUDA` passes its own
    (the compiled lowering rejects the tiny CPU geometry)."""
    rows = [(0, start, num_new) for start, num_new in row_specs]
    meta = base_meta(rows)
    batch = len(rows)
    block_tables = torch.zeros(
        (batch, max_blocks_per_seq), dtype=torch.int32
    )
    kv_num_blocks = torch.zeros(batch, dtype=torch.int32)
    inverse = torch.full(
        (batch, num_blocks + 1), max_blocks_per_seq, dtype=torch.int32
    )
    for r, ((start, num_new), table) in enumerate(zip(row_specs, tables)):
        history = start + num_new
        kv_num_blocks[r] = -(-history // block)
        for j, blk in enumerate(table):
            block_tables[r, j] = blk
            inverse[r, blk] = j
    meta.seed_paged_tables(PagedTables(
        block_tables=block_tables,
        kv_num_blocks=kv_num_blocks,
        inverse_tables=inverse,
        write_map=paged_write_map(rows, tables, block),
    ))
    return meta


@torch.inference_mode()
def padded_step(model, pool, rows: list[tuple[int, int, list[int]]]):
    """rows: [(slot, start_pos, token_ids)] → (B, vocab) logits."""
    meta = base_meta([(s, st, len(t)) for s, st, t in rows])
    input_ids = torch.zeros(len(rows), meta.num_new_max, dtype=torch.int64)
    for i, (_, _, toks) in enumerate(rows):
        input_ids[i, : len(toks)] = torch.tensor(toks)
    return model.forward_batched(input_ids, meta, pool)


@torch.inference_mode()
def flex_step(model, pool, rows: list[tuple[int, list[int], list[int]]]):
    """rows: [(start_pos, token_ids, block_table)] → (B, vocab) logits."""
    meta = paged_meta(
        [(start, len(toks)) for start, toks, _ in rows],
        [table for _, _, table in rows],
    )
    input_ids = torch.zeros(len(rows), meta.num_new_max, dtype=torch.int64)
    for i, (_, toks, _) in enumerate(rows):
        input_ids[i, : len(toks)] = torch.tensor(toks)
    return model.forward_batched(input_ids, meta, pool)


class TestWriteMap:
    """Pins for `paged_write_map` alone — no attend involved."""

    def test_matches_naive_per_token_loop(self):
        batch_rows = [(0, 0, 5), (0, 6, 3), (0, 13, 1)]
        block_tables = [[5, 2], [9, 1, 7], [0, 3, 8, 11]]
        wm = paged_write_map(batch_rows, block_tables, BLOCK)
        expected = []
        for batch_row_index, (_, start, num_new) in enumerate(batch_rows):
            for token_offset in range(num_new):
                pos = start + token_offset
                physical_block = block_tables[batch_row_index][pos // BLOCK]
                pool_index = physical_block * BLOCK + pos % BLOCK
                expected.append((batch_row_index, token_offset, pool_index))
        got = list(zip(
            wm.batch_row.tolist(),
            wm.token_offset.tolist(),
            wm.pool_index.tolist(),
        ))
        assert got == expected

    def test_skips_filler_rows(self):
        # A filler (num_new == 0) between real rows contributes nothing;
        # the real rows keep their own batch-row indices.
        rows = [(0, 0, 2), (0, 0, 0), (0, 4, 1)]
        tables = [[3], [], [6, 2]]
        wm = paged_write_map(rows, tables, BLOCK)
        assert wm.batch_row.tolist() == [0, 0, 2]
        assert wm.token_offset.tolist() == [0, 1, 0]
        assert wm.pool_index.tolist() == [12, 13, 8]  # 3*4+0, 3*4+1, 2*4+0

    def test_crosses_block_boundaries(self):
        # One chunk spanning three blocks: positions 2..8 through table
        # [5, 1, 9] land at 5*4+{2,3}, 1*4+{0..3}, 9*4+0.
        wm = paged_write_map([(0, 2, 7)], [[5, 1, 9]], BLOCK)
        assert wm.pool_index.tolist() == [22, 23, 4, 5, 6, 7, 36]

    def test_all_filler_step_yields_an_empty_map(self):
        wm = paged_write_map([(0, 0, 0), (0, 0, 0)], [[], []], BLOCK)
        assert wm.batch_row.numel() == 0 and wm.pool_index.numel() == 0


class TestSingleRow:
    def test_full_prefill_matches_padded(self):
        oracle, flex = build_models()
        expected = padded_step(
            oracle, make_padded_pool(), [(0, 0, PROMPT_A)]
        )
        table = [0, 1, 2]                      # identity: 12 tokens, 3 blocks
        logits = flex_step(flex, make_paged_pool(), [(0, PROMPT_A, table)])
        torch.testing.assert_close(logits, expected, atol=1e-5, rtol=0)

    def test_scattered_table_matches_padded(self):
        # THE paged pin: a permuted, non-contiguous table must be
        # invisible to the math.
        oracle, flex = build_models()
        expected = padded_step(
            oracle, make_padded_pool(), [(0, 0, PROMPT_A)]
        )
        table = [5, 2, 9]
        logits = flex_step(flex, make_paged_pool(), [(0, PROMPT_A, table)])
        torch.testing.assert_close(logits, expected, atol=1e-5, rtol=0)

    def test_chunked_prefill_across_block_boundaries(self):
        # Chunks of 3 then 9: neither lands on a block boundary, so the
        # second chunk both finishes block 0 and spans blocks 1-2.
        oracle, flex = build_models()
        padded_pool, paged_pool = make_padded_pool(), make_paged_pool()
        table = [5, 2, 9]
        padded_step(oracle, padded_pool, [(0, 0, PROMPT_A[:3])])
        expected = padded_step(oracle, padded_pool, [(0, 3, PROMPT_A[3:])])
        flex_step(flex, paged_pool, [(0, PROMPT_A[:3], table)])
        logits = flex_step(flex, paged_pool, [(3, PROMPT_A[3:], table)])
        torch.testing.assert_close(logits, expected, atol=1e-5, rtol=0)

    def test_prefill_then_decode_matches(self):
        oracle, flex = build_models()
        padded_pool, paged_pool = make_padded_pool(), make_paged_pool()
        table = [3, 6]                         # 5-token prompt: 2 blocks
        padded_step(oracle, padded_pool, [(0, 0, PROMPT_B)])
        flex_step(flex, paged_pool, [(0, PROMPT_B, table)])
        for step in range(2):
            start = len(PROMPT_B) + step
            expected = padded_step(
                oracle, padded_pool, [(0, start, [DECODE_TOKEN])]
            )
            logits = flex_step(
                flex, paged_pool, [(start, [DECODE_TOKEN], table)]
            )
            torch.testing.assert_close(
                logits, expected, atol=1e-5, rtol=0
            )


class TestMixedBatch:
    def test_mixed_decode_and_prefill_chunk_matches(self):
        # Row A decodes (needs a boundary-crossing 4th block at history
        # 13); row B finishes a split prefill. Distinct scattered tables.
        oracle, flex = build_models()
        padded_pool, paged_pool = make_padded_pool(), make_paged_pool()
        table_a, table_b = [1, 2, 7, 5], [4, 0, 6]
        prompt_b = [51, 52, 53, 54, 55, 56, 57, 58, 59]

        padded_step(oracle, padded_pool, [(0, 0, PROMPT_A)])
        padded_step(oracle, padded_pool, [(1, 0, prompt_b[:3])])
        flex_step(flex, paged_pool, [(0, PROMPT_A, table_a)])
        flex_step(flex, paged_pool, [(0, prompt_b[:3], table_b)])

        expected = padded_step(oracle, padded_pool, [
            (0, len(PROMPT_A), [DECODE_TOKEN]),
            (1, 3, prompt_b[3:]),
        ])
        logits = flex_step(flex, paged_pool, [
            (len(PROMPT_A), [DECODE_TOKEN], table_a),
            (3, prompt_b[3:], table_b),
        ])
        torch.testing.assert_close(logits, expected, atol=1e-5, rtol=0)


class TestPoolState:
    def test_stale_block_reuse_is_clean(self):
        # Garbage-fill the whole paged pool (a freed block is never
        # zeroed), then run a fresh full prefill through it: the causal
        # fencing must hide every unwritten position, scratch included.
        oracle, flex = build_models()
        expected = padded_step(
            oracle, make_padded_pool(), [(0, 0, PROMPT_A)]
        )
        paged_pool = make_paged_pool()
        g = torch.Generator().manual_seed(99)
        for k, v in zip(paged_pool.k_layers, paged_pool.v_layers):
            k.copy_(torch.randn(k.shape, generator=g))
            v.copy_(torch.randn(v.shape, generator=g))
        logits = flex_step(
            flex, paged_pool, [(0, PROMPT_A, [5, 2, 9])]
        )
        torch.testing.assert_close(logits, expected, atol=1e-5, rtol=0)

    def test_pool_writes_land_at_table_positions(self):
        # The write side pinned independently of the attend: after one
        # full prefill, every layer's paged entries at the translated pool
        # indices equal the corresponding positions in the padded slot.
        oracle, flex = build_models()
        padded_pool, paged_pool = make_padded_pool(), make_paged_pool()
        table = [5, 2, 9]
        padded_step(oracle, padded_pool, [(0, 0, PROMPT_A)])
        flex_step(flex, paged_pool, [(0, PROMPT_A, table)])
        for i in range(TINY_ARCH["num_transformers"]):
            pk, pv = padded_pool.layer(i)
            fk, fv = paged_pool.layer(i)
            for pos in range(len(PROMPT_A)):
                pool_index = table[pos // BLOCK] * BLOCK + pos % BLOCK
                torch.testing.assert_close(
                    fk[pool_index], pk[0, pos], atol=1e-6, rtol=0
                )
                torch.testing.assert_close(
                    fv[pool_index], pv[0, pos], atol=1e-6, rtol=0
                )


# ---------------------------------------------------------------------------
# The 5090 half: the compiled-CUDA twin (paged-kv-plan.md §4, §5.4).
#
# The compiled lowering constrains geometry in ways eager CPU never does
# (probed on the 5090, 2026-08-30, torch 2.10.0+cu128 / sm_120):
# head_dim must be >= 16 (a tl.dot constraint), and the BlockMask's KV
# BLOCK_SIZE must be >= 64 on this build — below that every Triton
# template choice is pruned away (NoValidChoicesError); Q BLOCK_SIZE and
# q_len don't matter. Revalidate the 64 floor on torch upgrades: it is
# Inductor template pruning, not a documented contract.
#
# So the twin runs the CPU scenarios at kernel-supported geometry:
# 64-token blocks, multi-block prompts for the boundary crossings, and
# head_dim >= 16. The method under test is untouched — it uses its
# `block_size` for both index translation and mask granularity, which is
# exactly why production's 16-token pages vs this 64 floor is a
# chunk-5/6 design decision (paged-kv-plan.md chunk log), not this
# gate's concern.
# ---------------------------------------------------------------------------

CUDA_BLOCK = 64        # the mask KV-block floor this build will lower
CUDA_MAX_SEQ = 512
CUDA_MAX_BLOCKS_PER_SEQ = CUDA_MAX_SEQ // CUDA_BLOCK   # 8 — the sentinel
CUDA_NUM_BLOCKS = 12   # allocatable; scratch sits past

# Multi-block prompts (3 and 2 blocks at CUDA_BLOCK), deterministic.
CUDA_PROMPT_A = [(7 * i + 11) % 2048 for i in range(150)]
CUDA_PROMPT_B = [(13 * i + 5) % 2048 for i in range(100)]

# f32 toy arch at CUDA-lowerable head_dim; RoPE table sized for the
# longer prompts.
CUDA_ARCH = TINY_ARCH | {"head_dim": 16, "max_seq_len": CUDA_MAX_SEQ}


def build_cuda_models(arch, oracle_method, device):
    """(oracle, flex under-test) on `device`, identical weights."""
    torch.manual_seed(1234)
    oracle = Qwen3(qwen3_config=arch, attention_method=oracle_method)
    flex = Qwen3(
        qwen3_config=arch,
        attention_method=FlexAttentionMethod(block_size=CUDA_BLOCK),
    )
    flex.load_state_dict(oracle.state_dict())
    return oracle.eval().to(device), flex.eval().to(device)


def cuda_pools(arch, dtype, device):
    """(padded, paged) pools at the CUDA twin geometry."""
    padded = PaddedKVPool(
        num_layers=arch["num_transformers"], max_batch=4,
        max_seq_len=CUDA_MAX_SEQ, num_groups=arch["num_groups"],
        head_dim=arch["head_dim"], dtype=dtype, device=device,
    )
    paged = PagedKVPool(
        num_layers=arch["num_transformers"], num_kv_blocks=CUDA_NUM_BLOCKS,
        block_size=CUDA_BLOCK, max_seq_len=CUDA_MAX_SEQ,
        num_groups=arch["num_groups"], head_dim=arch["head_dim"],
        dtype=dtype, device=device,
    )
    return padded, paged


@torch.inference_mode()
def padded_step_on(model, pool, rows, device):
    """`padded_step` through the production device boundary
    (`move_batch_to`), so the meta the model sees took the same path a
    served step's meta takes."""
    meta = base_meta([(s, st, len(t)) for s, st, t in rows])
    input_ids = torch.zeros(len(rows), meta.num_new_max, dtype=torch.int64)
    for i, (_, _, toks) in enumerate(rows):
        input_ids[i, : len(toks)] = torch.tensor(toks)
    input_ids, meta = move_batch_to(input_ids, meta, device)
    return model.forward_batched(input_ids, meta, pool)


@torch.inference_mode()
def flex_step_on(model, pool, rows, device):
    """`flex_step` at the CUDA twin geometry, through `move_batch_to` —
    which must deliver the seeded paged tables to the device intact (the
    40fbcf9 hazard family; the CPU suite can't see a drop because nothing
    actually moves there)."""
    meta = paged_meta(
        [(start, len(toks)) for start, toks, _ in rows],
        [table for _, _, table in rows],
        block=CUDA_BLOCK, max_blocks_per_seq=CUDA_MAX_BLOCKS_PER_SEQ,
        num_blocks=CUDA_NUM_BLOCKS,
    )
    input_ids = torch.zeros(len(rows), meta.num_new_max, dtype=torch.int64)
    for i, (_, toks, _) in enumerate(rows):
        input_ids[i, : len(toks)] = torch.tensor(toks)
    input_ids, meta = move_batch_to(input_ids, meta, device)
    return model.forward_batched(input_ids, meta, pool)


_compiled_flex = None


def _compiled_flex_attention():
    """One compiled callable for the whole session: `dynamic=False`
    compiles per shape family, and sharing the object lets later tests
    reuse families earlier tests already paid to compile."""
    global _compiled_flex
    if _compiled_flex is None:
        _compiled_flex = torch.compile(flex_attention, dynamic=False)
    return _compiled_flex


@contextmanager
def compiled_flex_kernels():
    """Route FlexAttentionMethod's attention call through torch.compile.

    Only the `flex_attention` call is compiled — the spike's exact route
    (flex-spike-results.md §3) and the narrowest change that makes the
    flex-decoding split-KV kernel exist. The model-level compile wiring
    (`forward_batched_impl` traced whole) is chunk 6's, not this gate's.
    """
    prev = flex_module.flex_attention
    flex_module.flex_attention = _compiled_flex_attention()
    try:
        yield
    finally:
        flex_module.flex_attention = prev


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestOnCUDA:
    """The chunk-4 exit gate. The CPU half proves the index translation;
    these rerun the equivalence scenarios with the attention call
    compiled, on the device, where the flex-decoding kernel is a distinct
    code path — plus the kernel-actually-ran counter (the SDPA
    silent-fallback lesson, generalized) and a bf16 run at production
    attention geometry (kernel support surfaces are dtype/shape-dependent;
    the sdpa suite's lesson)."""

    def test_scattered_chunked_prefill_then_decode_matches_padded(self):
        # The suite's core scenario at twin geometry: chunked prefill
        # through a scattered table (chunks of 70 and 80, neither split
        # on a 64-token block boundary), then two q_len=1 steps — the
        # flex-decoding path.
        device = torch.device("cuda")
        oracle, flex = build_cuda_models(
            CUDA_ARCH, PaddedAttentionMethod(), device
        )
        padded_pool, paged_pool = cuda_pools(CUDA_ARCH, torch.float32, device)
        table = [5, 2, 9]
        with compiled_flex_kernels():
            padded_step_on(
                oracle, padded_pool, [(0, 0, CUDA_PROMPT_A[:70])], device
            )
            flex_step_on(
                flex, paged_pool, [(0, CUDA_PROMPT_A[:70], table)], device
            )
            expected = padded_step_on(
                oracle, padded_pool, [(0, 70, CUDA_PROMPT_A[70:])], device
            )
            got = flex_step_on(
                flex, paged_pool, [(70, CUDA_PROMPT_A[70:], table)], device
            )
            torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)
            for step in range(2):
                start = len(CUDA_PROMPT_A) + step
                expected = padded_step_on(
                    oracle, padded_pool, [(0, start, [DECODE_TOKEN])], device
                )
                got = flex_step_on(
                    flex, paged_pool, [(start, [DECODE_TOKEN], table)], device
                )
                torch.testing.assert_close(
                    got, expected, atol=1e-4, rtol=1e-4
                )

    def test_mixed_batch_matches_padded(self):
        # One batch, two rows, two scattered tables: row A decodes at
        # position 150, row B finishes a split prefill (30 + 70).
        device = torch.device("cuda")
        oracle, flex = build_cuda_models(
            CUDA_ARCH, PaddedAttentionMethod(), device
        )
        padded_pool, paged_pool = cuda_pools(CUDA_ARCH, torch.float32, device)
        table_a, table_b = [5, 2, 9], [7, 0]
        with compiled_flex_kernels():
            padded_step_on(oracle, padded_pool, [(0, 0, CUDA_PROMPT_A)], device)
            padded_step_on(
                oracle, padded_pool, [(1, 0, CUDA_PROMPT_B[:30])], device
            )
            flex_step_on(
                flex, paged_pool, [(0, CUDA_PROMPT_A, table_a)], device
            )
            flex_step_on(
                flex, paged_pool, [(0, CUDA_PROMPT_B[:30], table_b)], device
            )

            expected = padded_step_on(oracle, padded_pool, [
                (0, len(CUDA_PROMPT_A), [DECODE_TOKEN]),
                (1, 30, CUDA_PROMPT_B[30:]),
            ], device)
            got = flex_step_on(flex, paged_pool, [
                (len(CUDA_PROMPT_A), [DECODE_TOKEN], table_a),
                (30, CUDA_PROMPT_B[30:], table_b),
            ], device)
            torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)

    def test_flex_kernel_actually_ran(self):
        # Every output-level test above stays green if compilation quietly
        # falls back to eager Flex — profile a decode step and require an
        # Inductor flex kernel on the GPU timeline (this build names them
        # triton_tem_fused_flex_attention_0 and friends). Caught twice on
        # the sdpa side during 5090 bring-up; hence a standing counter.
        from torch.profiler import ProfilerActivity, profile

        device = torch.device("cuda")
        _, flex = build_cuda_models(CUDA_ARCH, PaddedAttentionMethod(), device)
        _, paged_pool = cuda_pools(CUDA_ARCH, torch.float32, device)
        table = [5, 2, 9]
        with compiled_flex_kernels():
            flex_step_on(flex, paged_pool, [(0, CUDA_PROMPT_A, table)], device)
            # Warm the decode shape family so the profiled step replays a
            # compiled kernel instead of timing Dynamo.
            flex_step_on(
                flex, paged_pool,
                [(len(CUDA_PROMPT_A), [DECODE_TOKEN], table)], device,
            )
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                flex_step_on(
                    flex, paged_pool,
                    [(len(CUDA_PROMPT_A) + 1, [DECODE_TOKEN], table)], device,
                )
                torch.cuda.synchronize()
        kernels = [
            e.key for e in prof.key_averages() if e.self_device_time_total > 0
        ]
        assert any("flex" in k.lower() for k in kernels), (
            f"no flex kernel on the GPU timeline — the compiled route fell "
            f"back to eager/unfused attention. Kernels seen: {sorted(kernels)}"
        )

    def test_bf16_production_geometry_close_to_sdpa(self):
        # Compiled Flex vs the served sdpa stack at the geometry the
        # engine actually runs: GQA 16 query / 8 KV heads, head_dim 128,
        # bf16. Same tiny depth and embedding so it stays a fixture, not
        # a checkpoint. Tolerance is the sdpa suite's bf16 pair; the
        # first 5090 run calibrates it — if it fails, report the
        # max-diff, don't loosen silently.
        device = torch.device("cuda")
        arch = TINY_ARCH | {
            "num_heads": 16, "num_groups": 8, "head_dim": 128,
            "max_seq_len": CUDA_MAX_SEQ, "dtype": torch.bfloat16,
        }
        sdpa_model, flex_model = build_cuda_models(
            arch, SDPAAttentionMethod(), device
        )
        padded_pool, paged_pool = cuda_pools(arch, torch.bfloat16, device)
        table = [5, 2, 9]
        with compiled_flex_kernels():
            expected = padded_step_on(
                sdpa_model, padded_pool, [(0, 0, CUDA_PROMPT_A)], device
            )
            got = flex_step_on(
                flex_model, paged_pool, [(0, CUDA_PROMPT_A, table)], device
            )
            torch.testing.assert_close(got, expected, atol=3e-2, rtol=1e-2)

            expected = padded_step_on(
                sdpa_model, padded_pool,
                [(0, len(CUDA_PROMPT_A), [DECODE_TOKEN])], device,
            )
            got = flex_step_on(
                flex_model, paged_pool,
                [(len(CUDA_PROMPT_A), [DECODE_TOKEN], table)], device,
            )
            torch.testing.assert_close(got, expected, atol=3e-2, rtol=1e-2)
