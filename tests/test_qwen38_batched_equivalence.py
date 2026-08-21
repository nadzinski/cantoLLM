"""Batched-vs-sequential equivalence for the Qwen 3.8 hybrid model.

Oracle discipline as in test_padded_equivalence.py: the sequential
einsum stack is the oracle, fp32 CPU, atol 1e-5. Steps are hand-built
BatchMeta mixes of prefill chunks, decode rows, and filler rows against
one shared HybridStatePool; every returned logits row must match a
full-sequence sequential forward at the same position.

The hybrid-specific hazards under test: GDN state carry across chunked
prefill, filler rows aliasing a LIVE slot 0, slot reuse resetting GDN
state, and the monotone-chunk contract failing loudly.
"""

import pytest
import torch

from cantollm.models.attention import EinsumAttentionMethod, PaddedAttentionMethod
from cantollm.models.attention.protocol import BatchMeta
from cantollm.kv_pool import KVPool
from cantollm.models.qwen38.model import Qwen38
from cantollm.models.qwen38.pool import HybridStatePool
from tests.tiny_qwen38 import TINY_QWEN38_ARCH, make_tiny_qwen38

MAX_SEQ = 64
ATOL = 1e-5


@pytest.fixture(scope="module")
def oracle():
    return make_tiny_qwen38(EinsumAttentionMethod())


@pytest.fixture(scope="module")
def cb_model(oracle):
    model = Qwen38(TINY_QWEN38_ARCH, attention_method=PaddedAttentionMethod())
    model.load_state_dict(oracle.state_dict())
    model.eval()
    return model


def fresh_pool(max_batch=4):
    return HybridStatePool.from_arch(
        TINY_QWEN38_ARCH,
        max_batch=max_batch,
        max_seq_len=MAX_SEQ,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def make_meta(rows, width=None):
    width = width if width is not None else max(n for _, _, n in rows)
    start = torch.tensor([r[1] for r in rows])
    return BatchMeta(
        rows=list(rows),
        slots=torch.tensor([r[0] for r in rows]),
        start_pos=start,
        num_new=torch.tensor([r[2] for r in rows]),
        positions=(start[:, None] + torch.arange(width)[None, :]).clamp(max=MAX_SEQ - 1),
        num_new_max=width,
        max_history_len=max(s + n for _, s, n in rows),
        device=None,
    )


def make_input_ids(rows, sequences, width):
    """Left-aligned, 0-padded (B, width): row r carries its sequence's
    tokens [start, start+num_new); filler rows are all padding."""
    out = torch.zeros(len(rows), width, dtype=torch.long)
    for r, (slot, start, num_new) in enumerate(rows):
        if num_new:
            out[r, :num_new] = torch.tensor(sequences[slot][start : start + num_new])
    return out


def run_step(cb_model, pool, rows, sequences, width=None):
    meta = make_meta(rows, width)
    input_ids = make_input_ids(rows, sequences, meta.num_new_max)
    with torch.inference_mode():
        return cb_model.forward_batched(input_ids, meta, pool)


@pytest.fixture(scope="module")
def oracle_logits(oracle):
    """position -> logits oracle: full teacher-forced forward per sequence."""

    def compute(tokens):
        with torch.inference_mode():
            return oracle(torch.tensor(tokens).unsqueeze(0), start_pos=0)[0]

    return compute


class TestMixedStepsMatchSequential:
    def test_chunked_prefill_decode_and_filler(self, cb_model, oracle_logits):
        torch.manual_seed(3)
        seq = {  # keyed by slot
            1: torch.randint(0, 2048, (10,)).tolist(),
            2: torch.randint(0, 2048, (8,)).tolist(),
        }
        ref = {slot: oracle_logits(tokens) for slot, tokens in seq.items()}
        pool = fresh_pool()

        # Step 1: two prefill chunks (one full, one partial).
        logits = run_step(cb_model, pool, [(1, 0, 4), (2, 0, 5)], seq)
        assert torch.allclose(logits[0], ref[1][3], atol=ATOL)
        assert torch.allclose(logits[1], ref[2][4], atol=ATOL)

        # Step 2: finish slot 1's prefill next to a decode row (padded).
        logits = run_step(cb_model, pool, [(1, 4, 3), (2, 5, 1)], seq)
        assert torch.allclose(logits[0], ref[1][6], atol=ATOL)
        assert torch.allclose(logits[1], ref[2][5], atol=ATOL)

        # Step 3: two decodes plus a filler row (slot 0 is EMPTY here).
        logits = run_step(cb_model, pool, [(1, 7, 1), (2, 6, 1), (0, 0, 0)], seq, width=1)
        assert torch.allclose(logits[0], ref[1][7], atol=ATOL)
        assert torch.allclose(logits[1], ref[2][6], atol=ATOL)
        assert pool.gdn_pos == [0, 8, 7, 0]

    def test_filler_rows_do_not_corrupt_live_slot_0(self, cb_model, oracle_logits):
        torch.manual_seed(4)
        seq = {0: torch.randint(0, 2048, (8,)).tolist()}
        ref = oracle_logits(seq[0])
        pool = fresh_pool()

        logits = run_step(cb_model, pool, [(0, 0, 4)], seq)
        assert torch.allclose(logits[0], ref[3], atol=ATOL)
        s_after_prefill = pool.s_layers[0][0].clone()

        # Decode steps with filler rows aliasing the LIVE slot 0.
        logits = run_step(cb_model, pool, [(0, 4, 1), (0, 0, 0), (0, 0, 0)], seq, width=1)
        assert torch.allclose(logits[0], ref[4], atol=ATOL)
        assert not torch.equal(pool.s_layers[0][0], s_after_prefill), (
            "slot 0's state should have advanced with its real row"
        )

        logits = run_step(cb_model, pool, [(0, 5, 1), (0, 0, 0)], seq, width=1)
        assert torch.allclose(logits[0], ref[5], atol=ATOL)
        assert pool.gdn_pos[0] == 6

    def test_slot_reuse_resets_gdn_state(self, cb_model, oracle_logits):
        torch.manual_seed(5)
        first = {2: torch.randint(0, 2048, (6,)).tolist()}
        pool = fresh_pool()
        run_step(cb_model, pool, [(2, 0, 6)], first)
        assert pool.gdn_pos[2] == 6

        # A new sequence claims slot 2 with start_pos 0: state must reset
        # and its logits must match a fresh sequential run exactly.
        second = {2: torch.randint(0, 2048, (5,)).tolist()}
        ref = oracle_logits(second[2])
        logits = run_step(cb_model, pool, [(2, 0, 5)], second)
        assert torch.allclose(logits[0], ref[4], atol=ATOL)
        assert pool.gdn_pos[2] == 5


class TestPoolContract:
    def test_satisfies_kv_pool_protocol(self):
        assert isinstance(fresh_pool(), KVPool)

    def test_layer_and_gdn_state_are_kind_checked(self):
        pool = fresh_pool()
        with pytest.raises(KeyError, match="linear_attention"):
            pool.layer(0)  # layer 0 is a GDN layer
        with pytest.raises(KeyError, match="full_attention"):
            pool.gdn_state(3)  # layer 3 is full attention
        assert pool.layer(3)[0].shape == (4, MAX_SEQ + 1, 2, 16)
        assert pool.gdn_state(0)[0].shape == (4, 6, 8, 8)
        assert pool.gdn_state(0)[0].dtype == torch.float32

    def test_only_full_layers_allocate_kv(self):
        pool = fresh_pool()
        assert sorted(pool.k_layers) == [3, 7]
        assert sorted(pool.s_layers) == [0, 1, 2, 4, 5, 6]

    def test_monotone_chunk_violation_fails_loudly(self, cb_model):
        torch.manual_seed(6)
        seq = {1: torch.randint(0, 2048, (8,)).tolist()}
        pool = fresh_pool()
        run_step(cb_model, pool, [(1, 0, 4)], seq)
        with pytest.raises(ValueError, match="cannot replay or skip"):
            run_step(cb_model, pool, [(1, 6, 1)], seq)  # skipped positions 4-5
        with pytest.raises(ValueError, match="cannot replay or skip"):
            run_step(cb_model, pool, [(1, 2, 1)], seq)  # replay
