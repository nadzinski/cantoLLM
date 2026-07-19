"""Phase-3 equivalence suite: SDPAAttentionMethod vs PaddedAttentionMethod,
on a weight-shared tiny Qwen3, float32, CPU (plus device classes at the end).

This file was the definition of done for the attend (author-hand-written,
landed 2026-07-19): it began as strict xfails that were deleted one by one
as the implementation went green, the same protocol as
test_padded_equivalence.py was for step 5.

Trust model: the oracle is `PaddedAttentionMethod.forward_batched`, itself
proven against the sequential einsum stack by test_padded_equivalence.py.
SDPA subclasses it and overrides only `_attend_batched`, so the KV-pool
mechanics are shared by construction — the TestSharedMechanics class pins
that down, and a failure anywhere else can only be the attend. Comparisons
use atol=1e-5: SDPA reduces in a different order than the einsum chain, so
bit-exactness is wrong in principle, not just in practice.
"""

import pytest
import torch

from cantollm.models.attention import PaddedAttentionMethod, SDPAAttentionMethod
from cantollm.models.qwen3.model import Qwen3
from tests.test_padded_equivalence import (
    DECODE_TOKEN,
    MAX_SEQ,
    PROMPT_A,
    PROMPT_B,
    PROMPT_C,
    batched_step,
    make_pool,
)
from tests.tiny_model import TINY_ARCH


def build_models() -> tuple[Qwen3, Qwen3]:
    """(padded oracle, sdpa under-test), identical weights."""
    torch.manual_seed(1234)
    oracle = Qwen3(qwen3_config=TINY_ARCH, attention_method=PaddedAttentionMethod())
    sdpa = Qwen3(qwen3_config=TINY_ARCH, attention_method=SDPAAttentionMethod())
    sdpa.load_state_dict(oracle.state_dict())
    oracle.eval()
    sdpa.eval()
    return oracle, sdpa


def run_both(steps: list[list[tuple[int, int, list[int]]]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Drive the identical step sequence through both models on fresh pools;
    return (oracle logits, sdpa logits) from the final step."""
    oracle, sdpa = build_models()
    out = []
    for model in (oracle, sdpa):
        pool = make_pool()
        logits = None
        for rows in steps:
            logits = batched_step(model, pool, rows)
        out.append(logits)
    return out[0], out[1]


class TestSingleRow:
    def test_full_prefill_matches_padded(self):
        expected, got = run_both([[(0, 0, PROMPT_B)]])
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=0)

    def test_prefill_then_decode_matches_padded(self):
        expected, got = run_both([
            [(0, 0, PROMPT_B)],
            [(0, len(PROMPT_B), [DECODE_TOKEN])],
        ])
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=0)

    def test_chunked_prefill_matches_padded(self):
        expected, got = run_both([
            [(0, 0, PROMPT_A[:8])],
            [(0, 8, PROMPT_A[8:])],
        ])
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=0)


class TestMixedBatch:
    def test_mixed_batch_matches_padded(self):
        """The real thing: one step carrying a mid-prefill chunk, a fresh
        full prefill, and a decode row — ragged start_pos and widths is
        exactly where the SDPA mask handling can go wrong."""
        expected, got = run_both([
            [(0, 0, PROMPT_A[:8]), (2, 0, PROMPT_C)],
            [
                (0, 8, PROMPT_A[8:]),                # mid-prefill chunk, width 4
                (1, 0, PROMPT_B),                    # fresh prefill, width 5
                (2, len(PROMPT_C), [DECODE_TOKEN]),  # decode, width 1
            ],
        ])
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=0)

    def test_row_output_invariant_to_batch_padding(self):
        """A row's logits must not depend on who else is in the batch. The
        pad query rows attend to something real (build_batched_mask keeps
        them finite — no fully-masked rows, SDPA's NaN footgun), but their
        garbage must never leak into real rows."""
        _, sdpa = build_models()

        alone = batched_step(sdpa, make_pool(), [(0, 0, PROMPT_C)])
        beside = batched_step(
            sdpa, make_pool(), [(0, 0, PROMPT_C), (1, 0, PROMPT_A[:8])]
        )
        torch.testing.assert_close(beside[0], alone[0], atol=1e-5, rtol=0)

    def test_stale_slot_reuse_is_clean(self):
        """Garbage from a slot's previous occupant must not affect a new
        sequence — the mask (as SDPA applies it), not zeroing, is the fence."""
        _, sdpa = build_models()

        clean = batched_step(sdpa, make_pool(), [(0, 0, PROMPT_B)])

        dirty_pool = make_pool()
        torch.manual_seed(9)
        dirty_pool.k[:, 0].normal_()  # previous occupant's leftovers
        dirty_pool.v[:, 0].normal_()
        dirty = batched_step(sdpa, dirty_pool, [(0, 0, PROMPT_B)])

        torch.testing.assert_close(dirty[0], clean[0], atol=1e-5, rtol=0)


class TestSharedMechanics:
    """The inherited KV-pool path — these passed from day one, before the
    attend existed, proving the factoring keeps the mechanics literally
    shared."""

    def test_kv_writes_match_padded_pool_exactly(self):
        """The KV write path is shared code, not merely equivalent code: after
        the same step, both methods' pools must match byte-for-byte (same
        weights, same projections — layer 0's K/V are computed before either
        attend runs, and deeper layers differ only through the attend's
        output, which is why this asserts the write, not atol-equivalence)."""
        oracle, sdpa = build_models()
        slot = 1

        oracle_pool = make_pool()
        batched_step(oracle, oracle_pool, [(slot, 0, PROMPT_B)])

        sdpa_pool = make_pool()
        batched_step(sdpa, sdpa_pool, [(slot, 0, PROMPT_B)])

        torch.testing.assert_close(sdpa_pool.k[0], oracle_pool.k[0], atol=0, rtol=0)
        torch.testing.assert_close(sdpa_pool.v[0], oracle_pool.v[0], atol=0, rtol=0)

    def test_overlong_write_is_rejected_before_anything(self):
        """The validate-before-write promise is inherited: an overlong row
        fails loudly (ValueError, not NotImplementedError — validation runs
        first) and writes nothing."""
        _, sdpa = build_models()
        pool = make_pool()
        rows = [
            (0, 0, PROMPT_C),                # valid row first
            (1, MAX_SEQ - 2, [1, 2, 3, 4]),  # overlong: 30 + 4 > 32
        ]

        with pytest.raises(ValueError):
            batched_step(sdpa, pool, rows)

        assert torch.all(pool.k == 0) and torch.all(pool.v == 0), (
            "a failed step wrote into the pool — validate-before-write broken"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestOnCUDA:
    """The 5090 half: equivalence on the real device, and the dispatcher
    facts the design decision rests on. `bench/probe_sdpa.py` is the
    fuller version of the dispatch story (per-backend timing included)."""

    def _cuda_step(self, model, rows):
        device = torch.device("cuda")
        model = model.to(device)
        from cantollm.kv_pool import PaddedKVPool
        from tests.test_padded_equivalence import make_meta

        from cantollm.models.attention import BatchMeta

        pool = PaddedKVPool(
            num_layers=TINY_ARCH["num_transformers"], max_batch=4,
            max_seq_len=MAX_SEQ, num_groups=TINY_ARCH["num_groups"],
            head_dim=TINY_ARCH["head_dim"], dtype=torch.float32, device=device,
        )
        specs = [(slot, start, len(toks)) for slot, start, toks in rows]
        meta = make_meta(specs)
        meta = BatchMeta(
            rows=meta.rows, slots=meta.slots.to(device),
            start_pos=meta.start_pos.to(device), num_new=meta.num_new.to(device),
            positions=meta.positions.to(device),
            num_new_max=meta.num_new_max, max_history_len=meta.max_history_len,
            device=device,
        )
        input_ids = torch.zeros(
            len(rows), meta.num_new_max, dtype=torch.int64, device=device
        )
        for i, (_, _, toks) in enumerate(rows):
            input_ids[i, : len(toks)] = torch.tensor(toks, device=device)
        return model.forward_batched(input_ids, meta, pool)

    def test_mixed_batch_close_to_padded_on_cuda(self):
        oracle, sdpa = build_models()
        rows = [(0, 0, PROMPT_A[:8]), (1, 0, PROMPT_B), (2, 0, PROMPT_C)]
        expected = self._cuda_step(oracle, rows)
        got = self._cuda_step(sdpa, rows)
        torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)

    # -- dispatcher tripwires --------------------------------------------
    #
    # These probe the RAW F.scaled_dot_product_attention call, not the
    # model: the attend now pins its own backends internally, and nested
    # sdpa_kernel contexts replace (not intersect) the outer one, so
    # driving forward_batched under an outer pin would test nothing.
    # Production geometry and dtype on purpose — cuDNN's support surface
    # is dtype/shape-dependent (the f32 tiny model is the wrong probe).

    @staticmethod
    def _raw_call(backend):
        """Our exact call shape at 0.6B geometry: bf16, GQA 16q/8kv heads,
        head_dim 128, explicit bool mask (True = attend), enable_gqa."""
        import torch.nn.functional as F
        from torch.nn.attention import sdpa_kernel

        device = torch.device("cuda")
        q = torch.randn(2, 16, 5, 128, dtype=torch.bfloat16, device=device)
        k = torch.randn(2, 8, 64, 128, dtype=torch.bfloat16, device=device)
        v = torch.randn(2, 8, 64, 128, dtype=torch.bfloat16, device=device)
        mask = torch.ones(2, 1, 5, 64, dtype=torch.bool, device=device).tril(30)
        with sdpa_kernel([backend]):
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, enable_gqa=True
            )

    def test_cudnn_backend_accepts_our_call(self):
        """The amended design (see sdpa.py's docstring) rests on cuDNN —
        the one fused backend that takes both the explicit mask and dense
        GQA on this stack, and the backend the attend now pins first. If
        this raises ('No available kernel'), the production pin is falling
        through to unfused math — this test is the tripwire."""
        from torch.nn.attention import SDPBackend

        self._raw_call(SDPBackend.CUDNN_ATTENTION)

    def test_cudnn_matches_math_at_production_geometry(self):
        """The pinned kernel's numerics vs the reference math backend, at
        the dtype/shape the engine actually runs (the f32 tiny-model
        equivalence above never exercises the cuDNN kernel). Loose bf16
        tolerance; catches wrong-mask/wrong-head-mapping classes of bug."""
        from torch.nn.attention import SDPBackend

        torch.manual_seed(7)
        got = self._raw_call(SDPBackend.CUDNN_ATTENTION)
        torch.manual_seed(7)
        expected = self._raw_call(SDPBackend.MATH)
        torch.testing.assert_close(got, expected, atol=3e-2, rtol=1e-2)

    def test_attend_runs_fused_on_cuda(self):
        """The production-path tripwire: profile the real `_attend_batched`
        at production dtype/geometry and assert a cuDNN SDPA kernel actually
        ran. This is the check the pinned-backend tests above cannot make —
        sdpa_kernel only restricts the backend set, and this build's default
        priority ranks math above cuDNN, so without `set_priority=True` in
        the attend the call silently runs unfused while every output-level
        test stays green. Caught twice during 5090 bring-up; hence a test."""
        from torch.profiler import ProfilerActivity, profile

        device = torch.device("cuda")
        method = SDPAAttentionMethod()
        q = torch.randn(4, 1, 8, 2, 128, dtype=torch.bfloat16, device=device)
        keys = torch.randn(4, 64, 8, 128, dtype=torch.bfloat16, device=device)
        values = torch.randn(4, 64, 8, 128, dtype=torch.bfloat16, device=device)
        mask = torch.zeros(4, 1, 64, dtype=torch.bool, device=device)
        method._attend_batched(q, keys, values, mask)  # warm/dispatch once
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            method._attend_batched(q, keys, values, mask)
            torch.cuda.synchronize()
        kernels = [
            e.key for e in prof.key_averages() if e.self_device_time_total > 0
        ]
        assert any("sdpa" in k.lower() or "cudnn" in k.lower() for k in kernels), (
            f"no fused SDPA kernel in the attend's profile — the pin fell "
            f"through to the math backend. Kernels seen: {kernels}"
        )

    def test_memory_efficient_backend_rejects_our_call(self):
        """Why the original decision was amended: mem-efficient does not
        honor enable_gqa for dense inputs on this stack (torch 2.10/sm_120)
        and rejects our head layout. If this ever starts passing, the
        efficient route is back on the table — not actionable while cuDNN
        is fused and fast, but worth knowing; note it in PLAN.md Phase 3."""
        from torch.nn.attention import SDPBackend

        with pytest.raises(RuntimeError):
            self._raw_call(SDPBackend.EFFICIENT_ATTENTION)

    def test_flash_backend_rejects_our_mask(self):
        """The documented reason for the original decision: flash computes
        masks from index arithmetic and takes no mask tensors. If this ever
        starts passing under flash-only dispatch, the explicit-mask
        compromise is obsolete — good news, revisit PLAN.md Phase 3/4."""
        from torch.nn.attention import SDPBackend

        with pytest.raises(RuntimeError):
            self._raw_call(SDPBackend.FLASH_ATTENTION)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available"
)
class TestOnMPS:
    """A local Mac signal before the 5090: same device, different attends,
    loose tolerance (different kernels, different reduction order)."""

    def test_single_row_prefill_close_to_padded_on_mps(self):
        from cantollm.kv_pool import PaddedKVPool
        from cantollm.models.attention import BatchMeta
        from tests.test_padded_equivalence import make_meta

        device = torch.device("mps")
        oracle, sdpa = build_models()
        results = []
        for model in (oracle, sdpa):
            model = model.to(device)
            pool = PaddedKVPool(
                num_layers=TINY_ARCH["num_transformers"], max_batch=4,
                max_seq_len=MAX_SEQ, num_groups=TINY_ARCH["num_groups"],
                head_dim=TINY_ARCH["head_dim"], dtype=torch.float32, device=device,
            )
            meta = make_meta([(0, 0, len(PROMPT_B))])
            meta = BatchMeta(
                rows=meta.rows, slots=meta.slots.to(device),
                start_pos=meta.start_pos.to(device), num_new=meta.num_new.to(device),
                positions=meta.positions.to(device),
                num_new_max=meta.num_new_max, max_history_len=meta.max_history_len,
                device=device,
            )
            input_ids = torch.tensor([PROMPT_B], device=device)
            results.append(model.forward_batched(input_ids, meta, pool))

        torch.testing.assert_close(
            results[1].cpu(), results[0].cpu(), atol=1e-3, rtol=1e-3
        )
