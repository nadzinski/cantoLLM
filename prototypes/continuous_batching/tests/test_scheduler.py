"""Scheduler equivalence-with-reference tests. These are the meaty ones.

Every test asserts that the scheduler's per-request output token list is
identical to what `SequentialReference.generate(req)` would produce for the
same request. Mismatches mean the scheduler corrupted the per-sequence
state — the reference is the oracle.
"""

import pytest

from continuous_batching.padded_kv import PaddedKVCache
from continuous_batching.runner import run_to_completion, run_with_late_arrivals
from continuous_batching.scheduler import ContinuousBatchingScheduler


MAX_SEQ_LEN = 64
DIM = 16


def _build_scheduler(model, max_batch=4, max_tokens_per_step=8):
    cache = PaddedKVCache(max_batch=max_batch, max_seq_len=MAX_SEQ_LEN, dim=DIM)
    return ContinuousBatchingScheduler(
        model=model, cache=cache, max_tokens_per_step=max_tokens_per_step,
    )


def test_single_request_matches_reference(model, reference, make_request):
    req = make_request(prompt_len=5, max_tokens=6)
    expected = reference.generate(req)

    sched = _build_scheduler(model)
    tokens, _ = run_to_completion(sched, [req])

    assert tokens[req.request_id] == expected


def test_three_concurrent_requests_match_reference(model, reference, make_request):
    reqs = [
        make_request(prompt_len=4, max_tokens=5, request_id="a"),
        make_request(prompt_len=9, max_tokens=8, request_id="b"),
        make_request(prompt_len=12, max_tokens=3, request_id="c"),
    ]
    expected = {r.request_id: reference.generate(r) for r in reqs}

    sched = _build_scheduler(model, max_batch=4, max_tokens_per_step=16)
    tokens, _ = run_to_completion(sched, reqs)

    for r in reqs:
        assert tokens[r.request_id] == expected[r.request_id], (
            f"{r.request_id}: got {tokens[r.request_id]} vs ref {expected[r.request_id]}"
        )


def test_chunked_prefill_matches_reference(model, reference, make_request):
    req = make_request(prompt_len=20, max_tokens=4)
    expected = reference.generate(req)

    # Budget of 4 tokens/step, 20-token prompt -> at least 5 steps for prefill.
    sched = _build_scheduler(model, max_batch=2, max_tokens_per_step=4)
    tokens, _ = run_to_completion(sched, [req])

    assert tokens[req.request_id] == expected


def test_more_requests_than_slots(model, reference, make_request):
    reqs = [
        make_request(prompt_len=4, max_tokens=3, request_id=f"r{i}")
        for i in range(6)
    ]
    expected = {r.request_id: reference.generate(r) for r in reqs}

    sched = _build_scheduler(model, max_batch=2, max_tokens_per_step=8)
    tokens, _ = run_to_completion(sched, reqs)

    for r in reqs:
        assert tokens[r.request_id] == expected[r.request_id]


def test_stop_token_emitted_at_correct_position(model, reference, make_request):
    probe = make_request(prompt_len=5, max_tokens=10, request_id="probe")
    full = reference.generate(probe)
    assert len(full) >= 4
    stop = full[3]
    if stop in full[:3]:
        pytest.skip("stop token also appears earlier; skip this seed")

    reqs = [
        make_request(prompt_len=5, max_tokens=10, stop_tokens=(stop,), request_id="x"),
        make_request(prompt_len=5, max_tokens=10, stop_tokens=(stop,), request_id="y"),
    ]
    expected = full[:4]

    sched = _build_scheduler(model, max_batch=2, max_tokens_per_step=8)
    tokens, finish = run_to_completion(sched, reqs)

    for r in reqs:
        assert tokens[r.request_id] == expected
        assert finish[r.request_id] == "end_turn"


class _BudgetRecorder:
    """Wraps a ToyModel and records sum(num_new) per forward call."""

    def __init__(self, inner):
        self.inner = inner
        self.per_step_totals: list[int] = []

    def __call__(self, input_ids, slot_metas, kv_cache):
        self.per_step_totals.append(sum(n for _, _, n in slot_metas))
        return self.inner(input_ids, slot_metas, kv_cache)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def test_step_respects_token_budget(model, make_request):
    recorder = _BudgetRecorder(model)
    cache = PaddedKVCache(max_batch=4, max_seq_len=MAX_SEQ_LEN, dim=DIM)
    budget = 4
    sched = ContinuousBatchingScheduler(
        model=recorder, cache=cache, max_tokens_per_step=budget,
    )
    req = make_request(prompt_len=20, max_tokens=4)
    run_to_completion(sched, [req])

    violations = [t for t in recorder.per_step_totals if t > budget]
    assert not violations, (
        f"step(s) exceeded budget={budget}: per-step totals were "
        f"{recorder.per_step_totals}"
    )


def test_step_respects_token_budget_with_concurrent_prefills(model, make_request):
    """Two seqs both in prefill, both falling into the else branch — n>=2."""
    recorder = _BudgetRecorder(model)
    cache = PaddedKVCache(max_batch=2, max_seq_len=MAX_SEQ_LEN, dim=DIM)
    budget = 4
    sched = ContinuousBatchingScheduler(
        model=recorder, cache=cache, max_tokens_per_step=budget,
    )
    reqs = [
        make_request(prompt_len=20, max_tokens=2, request_id="a"),
        make_request(prompt_len=20, max_tokens=2, request_id="b"),
    ]
    run_to_completion(sched, reqs)

    violations = [t for t in recorder.per_step_totals if t > budget]
    assert not violations, (
        f"step(s) exceeded budget={budget}: per-step totals were "
        f"{recorder.per_step_totals}"
    )


def test_rejects_budget_smaller_than_max_batch(model):
    cache = PaddedKVCache(max_batch=4, max_seq_len=MAX_SEQ_LEN, dim=DIM)
    with pytest.raises(ValueError, match="max_tokens_per_step"):
        ContinuousBatchingScheduler(
            model=model, cache=cache, max_tokens_per_step=3,
        )


def test_late_arriving_request(model, reference, make_request):
    req_a = make_request(prompt_len=6, max_tokens=8, request_id="a")
    req_b = make_request(prompt_len=4, max_tokens=5, request_id="b")
    expected_a = reference.generate(req_a)
    expected_b = reference.generate(req_b)

    sched = _build_scheduler(model, max_batch=2, max_tokens_per_step=8)
    schedule = [(0, req_a), (3, req_b)]
    tokens, _ = run_with_late_arrivals(sched, schedule)

    assert tokens[req_a.request_id] == expected_a
    assert tokens[req_b.request_id] == expected_b
