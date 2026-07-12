"""Step-8 scheduler suite: the port's definition of done.

`water_fill`, `build_batch_meta`, and constructor validation are provided
code — those tests are green from day one. Everything under an
`@xfail_until_step8` marker is the hand-written part: work through with

    pytest tests/test_cb_scheduler.py -x

and DELETE each marker as its test goes green (strict=True polices
leftovers). The oracle is `tests/toy_stepper.toy_oracle` — the same toy
forward run one request at a time under the real emission contract
(stop suppression, >=, max_tokens<=0), mirroring StandardBackend.generate.

Checklist numbers referenced in test docstrings match the stub docstrings
in engine/batching/scheduler.py.
"""

import itertools

import pytest
import torch

from cantollm.engine.batching import BatchingConfig, SlotAllocator
from cantollm.engine.batching.scheduler import (
    ContinuousBatchingScheduler,
    Row,
    build_batch_meta,
    water_fill,
)
from cantollm.engine.batching.types import CBSequence
from cantollm.engine.types import InferenceRequest, SamplingParams, TokenEvent
from tests.toy_stepper import ToyStepper, make_toy_pool, toy_oracle

xfail_until_step8 = pytest.mark.xfail(
    raises=NotImplementedError, strict=True,
    reason="scheduler port not implemented yet (step 8)",
)

GREEDY = SamplingParams.from_temperature_top_p(temperature=0.0, top_p=1.0)


class RankInverter:
    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        return -logits


def make_request(
    rid: str,
    prompt: list[int],
    max_tokens: int = 5,
    stop_token_ids: set[int] | None = None,
    sampling: SamplingParams | None = None,
) -> InferenceRequest:
    return InferenceRequest(
        request_id=rid,
        prompt_token_ids=list(prompt),
        sampling_params=sampling or GREEDY,
        max_tokens=max_tokens,
        stop_token_ids=stop_token_ids if stop_token_ids is not None else set(),
    )


class RecordingForward:
    """Wraps the toy stepper, recording each step's real-token total."""

    def __init__(self, inner: ToyStepper):
        self.inner = inner
        self.step_token_counts: list[int] = []

    def __call__(self, input_ids, meta, pool):
        self.step_token_counts.append(int(meta.num_new.sum()))
        return self.inner(input_ids, meta, pool)


def make_scheduler(
    max_batch: int = 2, max_seq_len: int = 64, max_tokens_per_step: int = 8
) -> tuple[ContinuousBatchingScheduler, RecordingForward]:
    config = BatchingConfig(
        max_batch=max_batch, max_seq_len=max_seq_len,
        max_tokens_per_step=max_tokens_per_step,
    )
    forward = RecordingForward(ToyStepper())
    scheduler = ContinuousBatchingScheduler(
        forward_fn=forward,
        pool=make_toy_pool(config),
        allocator=SlotAllocator(max_batch),
        config=config,
    )
    return scheduler, forward


def assert_one_field(evt: TokenEvent) -> None:
    populated = [
        f for f in (evt.token_id, evt.finish_reason, evt.error) if f is not None
    ]
    assert len(populated) == 1, f"event populates {len(populated)} fields: {evt}"


def drain(
    scheduler: ContinuousBatchingScheduler,
    arrivals: dict[int, list[InferenceRequest]],
    max_steps: int = 500,
) -> dict[str, dict]:
    """Drive the scheduler to completion; arrivals[i] are added before step i.

    Returns per-request {"tokens": [...], "finish": str|None, "errors": [...],
    "events": [...]}. Also asserts the one-field-per-event contract and that
    step() is never called while idle.
    """
    results: dict[str, dict] = {}
    pending_arrivals = dict(arrivals)
    for step_idx in itertools.count():
        assert step_idx < max_steps, "scheduler did not converge"
        for req in pending_arrivals.pop(step_idx, []):
            results[req.request_id] = {
                "tokens": [], "finish": None, "errors": [], "events": [],
            }
            scheduler.add_request(req)
        if scheduler.is_idle():
            if not pending_arrivals:
                return results
            continue  # idle gap before a future arrival — must not drop it
        for evt in scheduler.step():
            assert_one_field(evt)
            r = results[evt.request_id]
            r["events"].append(evt)
            if evt.token_id is not None:
                r["tokens"].append(evt.token_id)
            if evt.finish_reason is not None:
                assert r["finish"] is None, "two finish events for one request"
                r["finish"] = evt.finish_reason
            if evt.error is not None:
                r["errors"].append(evt.error)


# ── Provided code: green from day one ────────────────────────────────


class TestWaterFill:
    def test_empty(self):
        assert water_fill(10, []) == []

    def test_single_bin_under_budget(self):
        assert water_fill(10, [4]) == [4]

    def test_single_bin_over_budget(self):
        assert water_fill(3, [10]) == [3]

    def test_exact_even_split(self):
        assert water_fill(9, [3, 3, 3]) == [3, 3, 3]

    def test_small_bins_first_leftover_to_large(self):
        assert water_fill(10, [1, 1, 20]) == [1, 1, 8]

    def test_budget_smaller_than_bin_count_starves_someone(self):
        alloc = water_fill(2, [1, 1, 1])
        assert sum(alloc) == 2 and alloc.count(0) == 1

    def test_zero_budget(self):
        assert water_fill(0, [5, 5]) == [0, 0]

    def test_invariants(self):
        for budget, caps in [(7, [3, 9, 1]), (16, [8, 8, 8]), (5, [2, 2, 2, 2])]:
            alloc = water_fill(budget, caps)
            assert sum(alloc) <= budget
            assert all(a <= c for a, c in zip(alloc, caps))


class TestProvidedPlumbing:
    def test_build_batch_meta(self):
        seq_a = CBSequence("a", [2, 3, 4], GREEDY, 5, set(), slot_idx=1, position=0)
        seq_b = CBSequence("b", [5], GREEDY, 5, set(), slot_idx=0, position=7)
        rows = [Row(seq_a, num_new=3, start_pos=0), Row(seq_b, num_new=1, start_pos=7)]

        meta = build_batch_meta(rows)

        assert meta.rows == [(1, 0, 3), (0, 7, 1)]
        assert meta.num_new_max == 3
        assert meta.max_history_len == 8
        assert meta.positions.shape == (2, 3)

    def test_constructor_rejects_mismatched_capacities(self):
        config = BatchingConfig(max_batch=2, max_seq_len=64, max_tokens_per_step=8)
        with pytest.raises(ValueError, match="slots"):
            ContinuousBatchingScheduler(
                forward_fn=ToyStepper(), pool=make_toy_pool(config),
                allocator=SlotAllocator(3), config=config,
            )


class TestToyOracle:
    """The oracle must be trustworthy before anything is compared to it."""

    def test_deterministic(self):
        req = make_request("x", [2, 3, 4], max_tokens=5)
        assert toy_oracle(req) == toy_oracle(req)

    def test_generates_requested_tokens_and_reports_max_tokens(self):
        tokens, finish = toy_oracle(make_request("x", [2, 3, 4], max_tokens=5))
        assert len(tokens) == 5 and finish == "max_tokens"

    def test_stop_token_suppressed(self):
        free_run, _ = toy_oracle(make_request("x", [2, 3, 4], max_tokens=5))
        stop = free_run[1]
        tokens, finish = toy_oracle(
            make_request("x", [2, 3, 4], max_tokens=5, stop_token_ids={stop})
        )
        assert tokens == free_run[:1] and stop not in tokens
        assert finish == "end_turn"

    def test_max_tokens_zero(self):
        assert toy_oracle(make_request("x", [2, 3], max_tokens=0)) == ([], "max_tokens")


# ── The hand-written part: delete markers as these go green ─────────


class TestSingleRequest:
    @xfail_until_step8
    def test_matches_oracle(self):
        scheduler, _ = make_scheduler()
        req = make_request("r1", [2, 3, 4, 5], max_tokens=6)

        results = drain(scheduler, {0: [req]})

        tokens, finish = toy_oracle(req)
        assert results["r1"]["tokens"] == tokens
        assert results["r1"]["finish"] == finish == "max_tokens"

    @xfail_until_step8
    def test_stop_token_suppressed_and_finishes_end_turn(self):
        """Checklist 1: the stop token is never emitted; the stream just
        ends with its own end_turn finish event."""
        scheduler, _ = make_scheduler()
        free_run, _ = toy_oracle(make_request("probe", [2, 3, 4], max_tokens=6))
        stop = free_run[3]  # make the 4th generated token a stop token
        req = make_request("r1", [2, 3, 4], max_tokens=6, stop_token_ids={stop})

        results = drain(scheduler, {0: [req]})

        assert results["r1"]["tokens"] == free_run[:3]
        assert stop not in results["r1"]["tokens"]
        assert results["r1"]["finish"] == "end_turn"

    @xfail_until_step8
    def test_finish_is_its_own_event(self):
        """Checklist 2: N token events + exactly one finish event, each
        populating exactly one field (drain asserts the field contract)."""
        scheduler, _ = make_scheduler()
        req = make_request("r1", [2, 3], max_tokens=3)

        results = drain(scheduler, {0: [req]})

        events = results["r1"]["events"]
        assert len(events) == len(results["r1"]["tokens"]) + 1
        assert events[-1].finish_reason is not None

    @xfail_until_step8
    def test_max_tokens_zero_finishes_immediately(self):
        """Checklist 4: no slot taken, no forward run, immediate finish."""
        scheduler, forward = make_scheduler()
        req = make_request("r1", [2, 3, 4], max_tokens=0)

        results = drain(scheduler, {0: [req]})

        assert results["r1"]["tokens"] == []
        assert results["r1"]["finish"] == "max_tokens"
        assert forward.step_token_counts == []
        assert scheduler.allocator.num_free() == scheduler.config.max_batch

    @xfail_until_step8
    def test_chunked_prefill_matches_oracle(self):
        """A 20-token prompt through a budget of 4 — prefill spans multiple
        steps and must land on the same tokens as a one-shot run."""
        scheduler, forward = make_scheduler(
            max_batch=2, max_seq_len=64, max_tokens_per_step=4
        )
        prompt = [(i % 29) + 2 for i in range(20)]
        req = make_request("r1", prompt, max_tokens=4)

        results = drain(scheduler, {0: [req]})

        tokens, finish = toy_oracle(req)
        assert results["r1"]["tokens"] == tokens
        assert results["r1"]["finish"] == finish


class TestConcurrency:
    @xfail_until_step8
    def test_three_concurrent_match_oracles(self):
        scheduler, _ = make_scheduler(max_batch=3)
        reqs = [
            make_request("r1", [2, 3, 4], max_tokens=5),
            make_request("r2", [9, 8], max_tokens=4),
            make_request("r3", [15, 16, 17, 18], max_tokens=6),
        ]

        results = drain(scheduler, {0: list(reqs)})

        for req in reqs:
            tokens, finish = toy_oracle(req)
            assert results[req.request_id]["tokens"] == tokens, req.request_id
            assert results[req.request_id]["finish"] == finish, req.request_id

    @xfail_until_step8
    def test_more_requests_than_slots(self):
        """5 requests, 2 slots: FCFS admission as slots free up; everyone
        still matches their solo oracle run."""
        scheduler, _ = make_scheduler(max_batch=2)
        reqs = [
            make_request(f"r{i}", [2 + i, 3 + i, 4 + i], max_tokens=3 + (i % 3))
            for i in range(5)
        ]

        results = drain(scheduler, {0: list(reqs)})

        for req in reqs:
            tokens, finish = toy_oracle(req)
            assert results[req.request_id]["tokens"] == tokens, req.request_id
            assert results[req.request_id]["finish"] == finish, req.request_id

    @xfail_until_step8
    def test_late_arrival_is_served(self):
        """The idle-gap trap (fable-review #5): a request arriving after the
        scheduler went idle must still be admitted and served."""
        scheduler, _ = make_scheduler()
        early = make_request("early", [2, 3], max_tokens=2)
        late = make_request("late", [7, 8, 9], max_tokens=3)

        results = drain(scheduler, {0: [early], 10: [late]})

        tokens, finish = toy_oracle(late)
        assert results["late"]["tokens"] == tokens
        assert results["late"]["finish"] == finish

    @xfail_until_step8
    def test_token_budget_respected_every_step(self):
        scheduler, forward = make_scheduler(
            max_batch=3, max_seq_len=64, max_tokens_per_step=6
        )
        reqs = [
            make_request("r1", [(i % 29) + 2 for i in range(18)], max_tokens=4),
            make_request("r2", [3, 4], max_tokens=6),
            make_request("r3", [5, 6, 7, 8, 9, 10, 11], max_tokens=5),
        ]

        drain(scheduler, {0: list(reqs)})

        assert forward.step_token_counts, "nothing ran"
        assert all(n <= 6 for n in forward.step_token_counts), (
            forward.step_token_counts
        )

    @xfail_until_step8
    def test_per_row_sampling_params(self):
        """Two identical prompts, one with a ranking-inverting processor:
        each row must be sampled with its own params (via engine/sampler)."""
        inverted = SamplingParams(processors=[RankInverter()], greedy=True)
        scheduler, _ = make_scheduler(max_batch=2)
        plain = make_request("plain", [2, 3, 4], max_tokens=4)
        weird = make_request("weird", [2, 3, 4], max_tokens=4, sampling=inverted)

        results = drain(scheduler, {0: [plain, weird]})

        assert results["plain"]["tokens"] == toy_oracle(plain)[0]
        assert results["weird"]["tokens"] == toy_oracle(weird)[0]
        assert results["plain"]["tokens"] != results["weird"]["tokens"]


class TestAbort:
    @xfail_until_step8
    def test_abort_active_frees_slot_for_queued(self):
        """Checklist 5, active half: abort emits its event, frees the slot,
        and the queued request takes it and completes correctly."""
        scheduler, _ = make_scheduler(max_batch=1)
        r1 = make_request("r1", [2, 3], max_tokens=50)
        r2 = make_request("r2", [9, 8, 7], max_tokens=3)
        scheduler.add_request(r1)
        scheduler.add_request(r2)  # queued: no free slot

        scheduler.step()
        scheduler.step()
        scheduler.abort("r1")

        events = []
        for _ in range(200):
            if scheduler.is_idle():
                break
            events.extend(scheduler.step())
        else:
            pytest.fail("did not converge after abort")

        r1_finishes = [e for e in events if e.request_id == "r1" and e.finish_reason]
        assert [e.finish_reason for e in r1_finishes] == ["abort"]
        r2_tokens = [e.token_id for e in events
                     if e.request_id == "r2" and e.token_id is not None]
        assert r2_tokens == toy_oracle(r2)[0]

    @xfail_until_step8
    def test_abort_queued_request(self):
        """Checklist 5, queued half: no slot to free, but the abort event
        still comes and the queue entry is gone."""
        scheduler, _ = make_scheduler(max_batch=1)
        r1 = make_request("r1", [2, 3], max_tokens=3)
        r2 = make_request("r2", [9, 8], max_tokens=3)
        scheduler.add_request(r1)
        scheduler.add_request(r2)

        scheduler.abort("r2")

        results_events = []
        for _ in range(200):
            if scheduler.is_idle():
                break
            results_events.extend(scheduler.step())
        r2_events = [e for e in results_events if e.request_id == "r2"]
        assert [e.finish_reason for e in r2_events] == ["abort"]
        r1_tokens = [e.token_id for e in results_events
                     if e.request_id == "r1" and e.token_id is not None]
        assert r1_tokens == toy_oracle(r1)[0]

    @xfail_until_step8
    def test_abort_unknown_is_silent_noop(self):
        scheduler, _ = make_scheduler()
        scheduler.abort("ghost")
        assert scheduler.is_idle()

    @xfail_until_step8
    def test_is_idle_false_while_events_pend(self):
        """The shell contract: after aborting the LAST active sequence the
        scheduler holds an un-flushed abort event — is_idle() must say False
        or the shell would block forever and the event would never flush."""
        scheduler, _ = make_scheduler(max_batch=1)
        scheduler.add_request(make_request("r1", [2, 3], max_tokens=50))
        scheduler.step()
        scheduler.abort("r1")

        assert not scheduler.is_idle(), "pending abort event not announced"
        events = scheduler.step()
        assert [e.finish_reason for e in events if e.request_id == "r1"] == ["abort"]
        assert scheduler.is_idle()


class TestAdmissionDefense:
    @xfail_until_step8
    def test_over_cap_request_rejected_with_error_event(self):
        """Checklist 6: defense behind the API's admission check — an
        over-cap add_request never queues and surfaces an error event."""
        scheduler, forward = make_scheduler(max_batch=2, max_seq_len=16)
        req = make_request("big", [2] * 10, max_tokens=10)  # 20 > 16

        results = drain(scheduler, {0: [req]})

        assert results["big"]["errors"], "expected an error event"
        assert results["big"]["tokens"] == []
        assert forward.step_token_counts == []
        assert scheduler.allocator.num_free() == 2
