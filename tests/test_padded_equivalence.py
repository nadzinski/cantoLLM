"""Step-5 equivalence suite: PaddedAttentionMethod.forward_batched vs the
sequential einsum stack, on a weight-shared tiny Qwen3, float32, CPU, greedy.

THIS FILE IS THE DEFINITION OF DONE FOR STEP 5 (hand-written attention math).
Every test is @xfail(raises=NotImplementedError, strict=True): red until the
math exists. Work through them with

    pytest tests/test_padded_equivalence.py -x

and DELETE each xfail marker as its test goes green — strict=True makes a
passing-but-still-marked test fail loudly (XPASS), so the suite polices its
own markers. Step 5 is finished when no markers remain and everything passes.

Trust model: the oracle is the sequential path (Qwen3.forward + KVCache +
EinsumAttentionMethod) — months-proven code exercising ZERO new machinery.
Comparisons use atol=1e-5 against the oracle (a vectorized batch sums in a
different order than the sequential kernel — bitwise equality is not the
contract) and atol=1e-6 for batched-vs-batched invariances (near-bitwise).
"""

import pytest
import torch

from cantollm.kv_cache import KVCache
from cantollm.kv_pool import PaddedKVPool
from cantollm.models.attention import (
    BatchMeta,
    EinsumAttentionMethod,
    PaddedAttentionMethod,
)
from cantollm.models.qwen3.model import Qwen3
from tests.tiny_model import TINY_ARCH

xfail_until_step5 = pytest.mark.xfail(
    raises=NotImplementedError, strict=True,
    reason="PaddedAttentionMethod.forward_batched not implemented yet (step 5)",
)

MAX_SEQ = 32

# Fixed prompts (ids well inside the tiny 2048 vocab). Decode steps feed
# fixed token ids rather than sampled ones: we're testing the forward pass,
# and both paths must see identical inputs.
PROMPT_A = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
PROMPT_B = [31, 32, 33, 34, 35]
PROMPT_C = [41, 42, 43, 44]
DECODE_TOKEN = 77


def build_models() -> tuple[Qwen3, Qwen3]:
    """(einsum oracle, padded under-test), identical weights."""
    torch.manual_seed(1234)
    oracle = Qwen3(qwen3_config=TINY_ARCH, attention_method=EinsumAttentionMethod())
    padded = Qwen3(qwen3_config=TINY_ARCH, attention_method=PaddedAttentionMethod())
    padded.load_state_dict(oracle.state_dict())
    oracle.eval()
    padded.eval()
    return oracle, padded


def make_pool(max_batch: int = 4) -> PaddedKVPool:
    return PaddedKVPool(
        num_layers=TINY_ARCH["num_transformers"], max_batch=max_batch,
        max_seq_len=MAX_SEQ, num_groups=TINY_ARCH["num_groups"],
        head_dim=TINY_ARCH["head_dim"], dtype=torch.float32,
        device=torch.device("cpu"),
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
    )


def batched_step(
    model: Qwen3, pool: PaddedKVPool, rows: list[tuple[int, int, list[int]]]
) -> torch.Tensor:
    """One forward_batched step. rows: [(slot, start_pos, token_ids)].
    Returns (B, vocab) logits."""
    specs = [(slot, start, len(toks)) for slot, start, toks in rows]
    meta = make_meta(specs)
    input_ids = torch.zeros(len(rows), meta.num_new_max, dtype=torch.int64)
    for i, (_, _, toks) in enumerate(rows):
        input_ids[i, : len(toks)] = torch.tensor(toks)
    return model.forward_batched(input_ids, meta, pool)


def oracle_last_logits(model: Qwen3, chunks: list[list[int]]) -> torch.Tensor:
    """Sequential oracle: run chunks through the einsum path on a fresh
    per-request KVCache; return the final chunk's last-token logits."""
    cache = KVCache(TINY_ARCH["num_transformers"])
    logits = None
    for chunk in chunks:
        tokens = torch.tensor([chunk])
        logits = model(tokens, start_pos=cache.position, kv_cache=cache)
    return logits[0, -1]


class TestSingleRow:
    @xfail_until_step5
    def test_full_prefill_matches_sequential(self):
        oracle, padded = build_models()
        logits = batched_step(padded, make_pool(), [(0, 0, PROMPT_B)])
        expected = oracle_last_logits(oracle, [PROMPT_B])
        torch.testing.assert_close(logits[0], expected, atol=1e-5, rtol=0)

    @xfail_until_step5
    def test_prefill_then_decode_matches_sequential(self):
        oracle, padded = build_models()
        pool = make_pool()
        batched_step(padded, pool, [(0, 0, PROMPT_B)])
        logits = batched_step(
            padded, pool, [(0, len(PROMPT_B), [DECODE_TOKEN])]
        )
        expected = oracle_last_logits(oracle, [PROMPT_B, [DECODE_TOKEN]])
        torch.testing.assert_close(logits[0], expected, atol=1e-5, rtol=0)

    @xfail_until_step5
    def test_chunked_prefill_matches_full_prefill(self):
        oracle, padded = build_models()
        pool = make_pool()
        batched_step(padded, pool, [(0, 0, PROMPT_A[:8])])
        logits = batched_step(padded, pool, [(0, 8, PROMPT_A[8:])])
        expected = oracle_last_logits(oracle, [PROMPT_A])
        torch.testing.assert_close(logits[0], expected, atol=1e-5, rtol=0)


class TestMixedBatch:
    @xfail_until_step5
    def test_mixed_batch_matches_per_row_sequential(self):
        """The real thing: one step carrying a mid-prefill chunk, a fresh
        full prefill, and a decode row — each row must match a sequential
        run that never saw the others."""
        oracle, padded = build_models()
        pool = make_pool()

        # Step 1: A starts chunked prefill (slot 0); C prefills fully (slot 2).
        batched_step(padded, pool, [(0, 0, PROMPT_A[:8]), (2, 0, PROMPT_C)])

        # Step 2: A finishes prefill, B prefills fresh, C decodes — mixed widths.
        logits = batched_step(
            padded, pool,
            [
                (0, 8, PROMPT_A[8:]),          # mid-prefill chunk, width 4
                (1, 0, PROMPT_B),              # fresh prefill, width 5
                (2, len(PROMPT_C), [DECODE_TOKEN]),  # decode, width 1
            ],
        )

        expectations = [
            oracle_last_logits(oracle, [PROMPT_A]),
            oracle_last_logits(oracle, [PROMPT_B]),
            oracle_last_logits(oracle, [PROMPT_C, [DECODE_TOKEN]]),
        ]
        for row, expected in enumerate(expectations):
            torch.testing.assert_close(
                logits[row], expected, atol=1e-5, rtol=0,
                msg=lambda m, row=row: f"row {row}: {m}",
            )

    @xfail_until_step5
    def test_row_output_invariant_to_batch_padding(self):
        """A row's logits must not depend on who else is in the batch —
        batched-vs-batched, so near-bitwise."""
        _, padded = build_models()

        alone = batched_step(padded, make_pool(), [(0, 0, PROMPT_C)])

        beside = batched_step(
            padded, make_pool(), [(0, 0, PROMPT_C), (1, 0, PROMPT_A[:8])]
        )
        torch.testing.assert_close(beside[0], alone[0], atol=1e-6, rtol=0)


class TestPoolState:
    @xfail_until_step5
    def test_pool_writes_match_sequential_cache(self):
        """After a single-row prefill, the slot's K/V must equal the
        sequential cache's post-RoPE keys/values, layer by layer."""
        oracle, padded = build_models()
        pool = make_pool()
        slot, n = 1, len(PROMPT_B)

        batched_step(padded, pool, [(slot, 0, PROMPT_B)])

        cache = KVCache(TINY_ARCH["num_transformers"])
        oracle(torch.tensor([PROMPT_B]), start_pos=0, kv_cache=cache)
        for i, layer in enumerate(cache):
            torch.testing.assert_close(
                pool.k[i, slot, :n], layer["keys"][0], atol=1e-6, rtol=0,
                msg=lambda m, i=i: f"layer {i} keys: {m}",
            )
            torch.testing.assert_close(
                pool.v[i, slot, :n], layer["values"][0], atol=1e-6, rtol=0,
                msg=lambda m, i=i: f"layer {i} values: {m}",
            )
        # And the write stayed inside the slot's occupied region.
        assert torch.all(pool.k[:, slot, n:] == 0)

    @xfail_until_step5
    def test_stale_slot_reuse_is_clean(self):
        """Garbage from a slot's previous occupant must not affect a new
        sequence — the mask, not zeroing, is the fence."""
        _, padded = build_models()

        clean_pool = make_pool()
        clean = batched_step(padded, clean_pool, [(0, 0, PROMPT_B)])

        dirty_pool = make_pool()
        torch.manual_seed(9)
        dirty_pool.k[:, 0].normal_()  # previous occupant's leftovers
        dirty_pool.v[:, 0].normal_()
        dirty = batched_step(padded, dirty_pool, [(0, 0, PROMPT_B)])

        torch.testing.assert_close(dirty[0], clean[0], atol=1e-6, rtol=0)

    @xfail_until_step5
    def test_overlong_write_is_rejected(self):
        """start_pos + num_new past the slot capacity must fail loudly at
        the write (the bounds assert), never corrupt a neighbor slot."""
        _, padded = build_models()
        pool = make_pool()
        overlong = [(0, MAX_SEQ - 2, [1, 2, 3, 4])]  # 30 + 4 > 32

        with pytest.raises((AssertionError, ValueError)):
            batched_step(padded, pool, overlong)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available"
)
class TestOnMPS:
    """The softer promise on real hardware: logit tolerance, not token
    equality (different kernels, different reduction order)."""

    @xfail_until_step5
    def test_single_row_prefill_close_to_cpu(self):
        _, padded = build_models()
        cpu_logits = batched_step(padded, make_pool(), [(0, 0, PROMPT_B)])

        device = torch.device("mps")
        padded_mps = padded.to(device)
        pool = PaddedKVPool(
            num_layers=TINY_ARCH["num_transformers"], max_batch=4,
            max_seq_len=MAX_SEQ, num_groups=TINY_ARCH["num_groups"],
            head_dim=TINY_ARCH["head_dim"], dtype=torch.float32, device=device,
        )
        specs = [(0, 0, len(PROMPT_B))]
        meta = make_meta(specs)
        meta = BatchMeta(
            rows=meta.rows, slots=meta.slots.to(device),
            start_pos=meta.start_pos.to(device), num_new=meta.num_new.to(device),
            positions=meta.positions.to(device),
            num_new_max=meta.num_new_max, max_history_len=meta.max_history_len,
        )
        input_ids = torch.tensor([PROMPT_B], device=device)
        mps_logits = padded_mps.forward_batched(input_ids, meta, pool)

        torch.testing.assert_close(
            mps_logits.cpu()[0], cpu_logits[0], atol=1e-3, rtol=1e-3
        )
