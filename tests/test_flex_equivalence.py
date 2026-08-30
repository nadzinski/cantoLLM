"""Chunk-4 equivalence suite: FlexAttentionMethod + paged pool vs the
padded stack, on a weight-shared tiny Qwen3, float32, CPU eager, greedy.

THIS FILE IS THE DEFINITION OF DONE FOR THE HAND-WRITTEN PAGED ATTEND
(P4 chunk 4: `paged_write_map` in engine/batching/paging.py + the Flex
method in models/attention/flex.py — both index-translation sites).
Every test is @xfail(raises=NotImplementedError, strict=True): red until
the translation exists. Work through them with

    pytest tests/test_flex_equivalence.py -x

and DELETE each xfail marker as its test goes green — strict=True makes a
passing-but-still-marked test fail loudly (XPASS), so the suite polices
its own markers. The write-map tests depend only on `paged_write_map`;
the model-level tests need the attend too.

Trust model: the oracle is the PADDED stack (proven against the
sequential einsum path by test_padded_equivalence.py), exercising none of
the new paging machinery. atol=1e-5 for logits (a different attend sums
in a different order), atol=1e-6 for pool-write pins (identical
projection math up to the write). CPU eager only — eager Flex proves the
index translation, not the flex-decoding kernel; the compiled-CUDA twin
is the chunk's 5090 exit gate (paged-kv-plan.md §4, §5.4).
"""

import pytest
import torch

from cantollm.engine.batching.paging import paged_write_map
from cantollm.kv_pool import PaddedKVPool, PagedKVPool
from cantollm.models.attention import (
    BatchMeta,
    FlexAttentionMethod,
    PaddedAttentionMethod,
    PagedTables,
)
from cantollm.models.qwen3.model import Qwen3
from tests.tiny_model import TINY_ARCH

xfail_until_chunk4 = pytest.mark.xfail(
    raises=NotImplementedError, strict=True,
    reason="paged_write_map + the Flex attend are the author's chunk-4 "
    "session (paged-kv-plan.md §8); delete each marker as it goes green",
)

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
    row_specs: list[tuple[int, int]], tables: list[list[int]]
) -> BatchMeta:
    """A seeded paged meta. row_specs: [(start_pos, num_new)]; tables[r]
    is row r's block table, covering at least its visible history."""
    rows = [(0, start, num_new) for start, num_new in row_specs]
    meta = base_meta(rows)
    batch = len(rows)
    block_tables = torch.zeros(
        (batch, MAX_BLOCKS_PER_SEQ), dtype=torch.int32
    )
    kv_num_blocks = torch.zeros(batch, dtype=torch.int32)
    inverse = torch.full(
        (batch, NUM_BLOCKS + 1), MAX_BLOCKS_PER_SEQ, dtype=torch.int32
    )
    for r, ((start, num_new), table) in enumerate(zip(row_specs, tables)):
        history = start + num_new
        kv_num_blocks[r] = -(-history // BLOCK)
        for j, block in enumerate(table):
            block_tables[r, j] = block
            inverse[r, block] = j
    meta.seed_paged_tables(PagedTables(
        block_tables=block_tables,
        kv_num_blocks=kv_num_blocks,
        inverse_tables=inverse,
        write_map=paged_write_map(rows, tables, BLOCK),
    ))
    return meta


def padded_step(model, pool, rows: list[tuple[int, int, list[int]]]):
    """rows: [(slot, start_pos, token_ids)] → (B, vocab) logits."""
    meta = base_meta([(s, st, len(t)) for s, st, t in rows])
    input_ids = torch.zeros(len(rows), meta.num_new_max, dtype=torch.int64)
    for i, (_, _, toks) in enumerate(rows):
        input_ids[i, : len(toks)] = torch.tensor(toks)
    return model.forward_batched(input_ids, meta, pool)


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

    @xfail_until_chunk4
    def test_matches_naive_per_token_loop(self):
        rows = [(0, 0, 5), (0, 6, 3), (0, 13, 1)]
        tables = [[5, 2], [9, 1, 7], [0, 3, 8, 11]]
        wm = paged_write_map(rows, tables, BLOCK)
        expected = []
        for r, (_, start, num_new) in enumerate(rows):
            for off in range(num_new):
                pos = start + off
                dst = tables[r][pos // BLOCK] * BLOCK + pos % BLOCK
                expected.append((r, off, dst))
        got = list(zip(wm.row.tolist(), wm.off.tolist(), wm.dst.tolist()))
        assert got == expected

    @xfail_until_chunk4
    def test_skips_filler_rows(self):
        # A filler (num_new == 0) between real rows contributes nothing;
        # the real rows keep their own batch indices.
        rows = [(0, 0, 2), (0, 0, 0), (0, 4, 1)]
        tables = [[3], [], [6, 2]]
        wm = paged_write_map(rows, tables, BLOCK)
        assert wm.row.tolist() == [0, 0, 2]
        assert wm.off.tolist() == [0, 1, 0]
        assert wm.dst.tolist() == [12, 13, 8]   # 3*4+0, 3*4+1, 2*4+0

    @xfail_until_chunk4
    def test_crosses_block_boundaries(self):
        # One chunk spanning three blocks: positions 2..8 through table
        # [5, 1, 9] land at 5*4+{2,3}, 1*4+{0..3}, 9*4+0.
        wm = paged_write_map([(0, 2, 7)], [[5, 1, 9]], BLOCK)
        assert wm.dst.tolist() == [22, 23, 4, 5, 6, 7, 36]

    @xfail_until_chunk4
    def test_all_filler_step_yields_an_empty_map(self):
        wm = paged_write_map([(0, 0, 0), (0, 0, 0)], [[], []], BLOCK)
        assert wm.row.numel() == 0 and wm.dst.numel() == 0


class TestSingleRow:
    @xfail_until_chunk4
    def test_full_prefill_matches_padded(self):
        oracle, flex = build_models()
        expected = padded_step(
            oracle, make_padded_pool(), [(0, 0, PROMPT_A)]
        )
        table = [0, 1, 2]                      # identity: 12 tokens, 3 blocks
        logits = flex_step(flex, make_paged_pool(), [(0, PROMPT_A, table)])
        torch.testing.assert_close(logits, expected, atol=1e-5, rtol=0)

    @xfail_until_chunk4
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

    @xfail_until_chunk4
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

    @xfail_until_chunk4
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
    @xfail_until_chunk4
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
    @xfail_until_chunk4
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

    @xfail_until_chunk4
    def test_pool_writes_land_at_table_positions(self):
        # The write side pinned independently of the attend: after one
        # full prefill, every layer's paged rows at the table-translated
        # destinations equal the padded pool's slot rows.
        oracle, flex = build_models()
        padded_pool, paged_pool = make_padded_pool(), make_paged_pool()
        table = [5, 2, 9]
        padded_step(oracle, padded_pool, [(0, 0, PROMPT_A)])
        flex_step(flex, paged_pool, [(0, PROMPT_A, table)])
        for i in range(TINY_ARCH["num_transformers"]):
            pk, pv = padded_pool.layer(i)
            fk, fv = paged_pool.layer(i)
            for pos in range(len(PROMPT_A)):
                dst = table[pos // BLOCK] * BLOCK + pos % BLOCK
                torch.testing.assert_close(
                    fk[dst], pk[0, pos], atol=1e-6, rtol=0
                )
                torch.testing.assert_close(
                    fv[dst], pv[0, pos], atol=1e-6, rtol=0
                )
