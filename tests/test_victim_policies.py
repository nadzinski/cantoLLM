"""Chunk-10 priority scheduling + victim-policy suite (paged-kv-plan.md
§2.10, §5.10).

Two halves, one file:

- GREEN (lands with the delegated chunk-10 commit): priority-sorted
  promotion (stable by priority, FCFS within a class, a preempted
  victim resumes at the front of its class but behind higher classes)
  and the preemption counters the stats collector reads.
- PRE-LANDED RED: TestPriorityVictimPolicy and TestCostVictimPolicy are
  THE DEFINITION OF DONE FOR THE HAND-WRITTEN VICTIM POLICIES (the
  author's chunk-10 session). Their class-level strict xfail markers
  (raises=NotImplementedError, so a partial implementation fails
  loudly instead of quietly xfailing, the chunk-2 lesson) come off as
  each policy lands; every test then runs unmarked. The `lifo` policy
  is the chunk-9 machine and needs no new selector.

Policy contracts under test (§2.10):
- `priority`: evict the lowest-priority active sequence; LIFO
  (newest-admitted) tiebreak among equal-lowest. Never a
  higher-priority row while a lower one is active.
- `cost`: evict the active sequence with the fewest KV tokens to lose,
  i.e. the cheapest recompute (the smallest consumed position).

The toy layer reuses the chunk-5/chunk-9 helpers: ConstantForward
scheduling isolation, the 4-block 16-token exhaustion geometry, and
drive_until_evicted's victim probe.
"""

import pytest

from cantollm.engine.batching.stats import StepStatsCollector
from tests.test_cb_scheduler import make_request
from tests.test_paged_scheduler import make_paged_scheduler, run_to_completion
from tests.test_preemption import (
    POOL_KW,
    PROMPT_A,
    PROMPT_B,
    drive_until_evicted,
    exhaust,
)


def prio_request(rid, prompt, max_tokens, priority):
    req = make_request(rid, prompt, max_tokens=max_tokens)
    req.priority = priority
    return req


def finish_order(scheduler, max_steps=200):
    order = []
    for _ in range(max_steps):
        if scheduler.is_idle():
            break
        for evt in scheduler.step():
            assert evt.error is None, evt.error
            if evt.finish_reason is not None:
                order.append(evt.request_id)
    assert scheduler.is_idle(), "run did not converge"
    return order


class TestPriorityPromotion:
    def test_priority_reaches_the_sequence(self):
        scheduler, _ = make_paged_scheduler()
        scheduler.add_request(prio_request("r", [1, 2], 2, priority=2))
        assert scheduler.queued[0].priority == 2

    def test_high_priority_promotes_first(self):
        scheduler, _ = make_paged_scheduler(max_batch=1)
        scheduler.add_request(prio_request("lo", [1, 2], 2, priority=0))
        scheduler.add_request(prio_request("hi", [3, 4], 2, priority=1))
        scheduler.step()
        assert [s.request_id for s in scheduler.active] == ["hi"]
        assert [s.request_id for s in scheduler.queued] == ["lo"]

    def test_equal_priorities_keep_fcfs_exactly(self):
        # §2.10's compatibility clause: all-default priorities must be
        # byte-for-byte today's FCFS admission order.
        scheduler, _ = make_paged_scheduler(max_batch=2)
        for rid in ("a", "b", "c"):
            scheduler.add_request(make_request(rid, [1, 2], max_tokens=2))
        scheduler.step()
        assert [s.request_id for s in scheduler.active] == ["a", "b"]
        assert [s.request_id for s in scheduler.queued] == ["c"]

    def test_higher_class_arrival_overtakes_a_preempted_victim(self):
        # The victim requeues at the front of ITS class (§3), not of the
        # whole queue: a higher-priority arrival waiting when eviction
        # frees capacity is served before the victim resumes.
        scheduler, _ = make_paged_scheduler(**POOL_KW)
        exhaust(scheduler)
        scheduler.step()   # a and b admitted before c exists
        scheduler.add_request(prio_request("c", [21, 22], 1, priority=1))
        order = finish_order(scheduler)
        assert order.index("c") < order.index("b")

    def test_preempted_victim_still_leads_its_own_class(self):
        # Same shape with an equal-priority arrival: the victim resumes
        # ahead of it (the chunk-9 front-of-queue pin, restated against
        # the sorted promotion path).
        scheduler, _ = make_paged_scheduler(**POOL_KW)
        exhaust(scheduler)
        scheduler.step()
        scheduler.add_request(make_request("c", [21, 22], max_tokens=1))
        drive_until_evicted(scheduler, "b")
        queued_ids = [s.request_id for s in scheduler.queued]
        assert queued_ids.index("b") < queued_ids.index("c")
        run_to_completion(scheduler, max_steps=100)


class TestPreemptionCounters:
    def test_totals_track_evictions_and_replay_cost(self):
        scheduler, _ = make_paged_scheduler(**POOL_KW)
        assert scheduler.preemptions_total == 0
        assert scheduler.preempted_tokens_total == 0
        exhaust(scheduler)
        drive_until_evicted(scheduler, "b")
        victim = next(s for s in scheduler.queued if s.request_id == "b")
        assert scheduler.preemptions_total == 1
        # The recompute cost is exactly what resume must re-prefill.
        assert scheduler.preempted_tokens_total == len(
            victim.replay_prefix_token_ids
        )

    def test_collector_reports_per_step_preemptions(self):
        scheduler, _ = make_paged_scheduler(**POOL_KW)
        collector = StepStatsCollector.for_scheduler(scheduler)
        assert collector is not None
        exhaust(scheduler)
        steps = []
        for _ in range(100):
            if scheduler.is_idle():
                break
            collector.before_step(scheduler)
            events = scheduler.step()
            steps.append(collector.after_step(scheduler, events))
        assert scheduler.is_idle()
        assert all(st.preemptions is not None for st in steps)
        assert sum(st.preemptions for st in steps) == \
            scheduler.preemptions_total
        assert sum(st.preempted_tokens for st in steps) == \
            scheduler.preempted_tokens_total
        assert scheduler.preemptions_total >= 1
        # Evictions are rare step events, not a per-step constant.
        assert any(st.preemptions == 0 for st in steps)


@pytest.mark.xfail(
    strict=True,
    raises=NotImplementedError,
    reason="P4 chunk 10: the hand-written 'priority' victim policy is "
    "not implemented",
)
class TestPriorityVictimPolicy:
    def test_victim_is_lowest_priority_not_newest(self):
        scheduler, _ = make_paged_scheduler(
            **POOL_KW, preemption_policy="priority"
        )
        # Stagger admission so the HIGH-priority row is the newest:
        # LIFO would evict "hi", the policy must evict "lo".
        scheduler.add_request(prio_request("lo", PROMPT_A, 8, priority=0))
        scheduler.step()
        scheduler.add_request(prio_request("hi", PROMPT_B, 8, priority=1))
        drive_until_evicted(scheduler, "lo")
        assert [s.request_id for s in scheduler.active] == ["hi"]
        run_to_completion(scheduler, max_steps=100)

    def test_lifo_tiebreak_among_equal_lowest(self):
        scheduler, _ = make_paged_scheduler(
            **POOL_KW, preemption_policy="priority"
        )
        exhaust(scheduler)   # both priority 0: newest ("b") is the victim
        drive_until_evicted(scheduler, "b")
        run_to_completion(scheduler, max_steps=100)


@pytest.mark.xfail(
    strict=True,
    raises=NotImplementedError,
    reason="P4 chunk 10: the hand-written 'cost' victim policy is not "
    "implemented",
)
class TestCostVictimPolicy:
    def test_victim_is_cheapest_recompute(self):
        scheduler, _ = make_paged_scheduler(
            **POOL_KW, preemption_policy="cost"
        )
        # PROMPT_A (4 tokens) stays behind PROMPT_B (8) at every step,
        # so "a" is always the cheaper re-prefill; LIFO would evict "b".
        exhaust(scheduler)
        drive_until_evicted(scheduler, "a")
        victim = next(s for s in scheduler.queued if s.request_id == "a")
        remaining = scheduler.active[0]
        assert len(victim.replay_prefix_token_ids) <= remaining.position
        run_to_completion(scheduler, max_steps=100)

    def test_cheapest_wins_regardless_of_admission_order(self):
        # Swap the arrival order: "a" is now the newest AND the cheapest;
        # a policy accidentally implementing "oldest" would evict "b".
        scheduler, _ = make_paged_scheduler(
            **POOL_KW, preemption_policy="cost"
        )
        scheduler.add_request(make_request("b", PROMPT_B, max_tokens=8))
        scheduler.add_request(make_request("a", PROMPT_A, max_tokens=8))
        drive_until_evicted(scheduler, "a")
        run_to_completion(scheduler, max_steps=100)
