"""Chunk-9 preemption suite (paged-kv-plan.md §5.9).

Pre-landed red one chunk ahead; the author's evict/resume machine
(6457e17, LIFO) turned every test green and the module-level strict
xfail came off in the same session. Run with:

    pytest tests/test_preemption.py -x

Semantics under test (§2.9, §3, and the §9.6 call):

- Trigger: eviction fires where chunk 5 raises "paged KV deadlock":
  only after grant trimming and boundary-starvation can advance no row.
  Shortage still trims before anyone is evicted (§4 atomicity).
- Victim: LIFO, the newest-admitted active sequence (victim POLICIES
  are chunk 10; this chunk hard-codes newest-first).
- Eviction: the victim's blocks AND slot are freed, it re-queues at
  the FRONT of the queue with its known prefix (prompt + tokens
  already emitted), and the eviction emits no events.
- Resume: the prefix re-prefills and decoding continues. The client
  stream is append-only: nothing already emitted is ever re-emitted,
  and under greedy the resumed stream is token-identical to a run
  that was never preempted (the model-free correctness bar).

Trust model: the toy layer (ConstantForward, borrowed from the chunk-5
suite) isolates the scheduling and accounting; the oracle layer runs
the weight-shared tiny Qwen3 pair from test_paged_engine_oracle, with
the padded arm as ground truth. The chunk-5 deadlock pin
(test_paged_scheduler) was updated deliberately alongside the machine:
it is now test_true_deadlock_preempts_and_completes.
"""

import torch

from cantollm.engine.batching import BatchingConfig
from cantollm.engine.batching.allocator import BlockAllocator, SlotAllocator
from cantollm.engine.batching.paging import PagedStepState
from cantollm.engine.batching.scheduler import ContinuousBatchingScheduler
from cantollm.kv_pool import PagedKVPool
from tests.test_cb_scheduler import drain, make_request
from tests.test_paged_engine_oracle import (
    assert_streams_match,
    build_arms,
    forward_fn,
)
from tests.test_paged_scheduler import (
    BLOCK,
    make_paged_scheduler,
    run_to_completion,
)
from tests.tiny_model import TINY_ARCH

# The exhaustion shape (from the chunk-5 deadlock pin): a 4-block pool
# at 16-token capacity, two rows that both grow across block boundaries
# with nothing finishing in time. Pre-chunk-9 this deadlocks; with
# eviction it must complete.
POOL_KW = dict(
    max_seq_len=16, num_kv_blocks=4, max_tokens_per_step=16, max_batch=2
)
PROMPT_A = [1, 2, 3, 4]
PROMPT_B = [5, 6, 7, 8, 9, 10, 11, 12]


def exhaust(scheduler):
    scheduler.add_request(make_request("a", PROMPT_A, max_tokens=8))
    scheduler.add_request(make_request("b", PROMPT_B, max_tokens=8))


def drive_until_evicted(scheduler, victim: str, max_steps: int = 50):
    """Step until `victim` has been preempted (active -> queued with an
    empty table), collecting events. Fails loudly if it never happens."""
    events = []
    for _ in range(max_steps):
        events.extend(scheduler.step())
        queued_ids = [s.request_id for s in scheduler.queued]
        if victim in queued_ids:
            seq = next(s for s in scheduler.queued if s.request_id == victim)
            assert seq.block_table == [], (
                "a preempted sequence must hold no blocks while queued"
            )
            return events
    raise AssertionError(f"{victim} was never evicted")


def held_blocks(scheduler) -> list[int]:
    return [b for s in scheduler.active for b in s.block_table]


class TestEvictResumeToy:
    def test_exhaustion_evicts_instead_of_deadlocking(self):
        scheduler, _ = make_paged_scheduler(**POOL_KW)
        exhaust(scheduler)
        finishes = {}
        for _ in range(100):
            if scheduler.is_idle():
                break
            for evt in scheduler.step():
                assert evt.error is None, evt.error
                if evt.finish_reason is not None:
                    finishes[evt.request_id] = evt.finish_reason
        assert scheduler.is_idle(), "run did not converge"
        assert finishes == {"a": "max_tokens", "b": "max_tokens"}
        assert scheduler.block_allocator.num_allocated() == 0

    def test_lifo_victim_requeues_at_the_front(self):
        # "c" waits in the queue when the pool jams; the victim must be
        # the NEWEST active sequence ("b", admitted after "a") and must
        # requeue AHEAD of "c", not behind it.
        scheduler, _ = make_paged_scheduler(**POOL_KW)
        exhaust(scheduler)
        scheduler.add_request(make_request("c", [21, 22], max_tokens=1))
        drive_until_evicted(scheduler, "b")
        assert [s.request_id for s in scheduler.active] == ["a"]
        queued_ids = [s.request_id for s in scheduler.queued]
        assert queued_ids.index("b") < queued_ids.index("c")
        run_to_completion(scheduler, max_steps=100)

    def test_eviction_emits_nothing_and_stream_stays_append_only(self):
        scheduler, _ = make_paged_scheduler(**POOL_KW)
        exhaust(scheduler)
        token_counts = {"a": 0, "b": 0}
        finish_counts = {"a": 0, "b": 0}
        for _ in range(100):
            if scheduler.is_idle():
                break
            for evt in scheduler.step():
                if evt.token_id is not None:
                    token_counts[evt.request_id] += 1
                if evt.finish_reason is not None:
                    assert evt.finish_reason == "max_tokens"
                    finish_counts[evt.request_id] += 1
        # Append-only: exactly max_tokens tokens each, once — a resume
        # that re-emitted its prefix would overshoot, an eviction that
        # emitted a finish would double-count.
        assert token_counts == {"a": 8, "b": 8}
        assert finish_counts == {"a": 1, "b": 1}

    def test_abort_while_preempted(self):
        scheduler, _ = make_paged_scheduler(**POOL_KW)
        exhaust(scheduler)
        drive_until_evicted(scheduler, "b")
        held = len(held_blocks(scheduler))
        assert scheduler.block_allocator.num_allocated() == held
        scheduler.abort("b")
        # The victim held nothing, so nothing may be freed twice: the
        # allocator's count must not move.
        assert scheduler.block_allocator.num_allocated() == held
        events = []
        for _ in range(60):
            if scheduler.is_idle():
                break
            events.extend(scheduler.step())
        assert scheduler.is_idle()
        assert any(
            e.request_id == "b" and e.finish_reason == "abort"
            for e in events
        )
        assert scheduler.block_allocator.num_allocated() == 0

    def test_accounting_and_stats_stay_coherent_through_evictions(self):
        scheduler, _ = make_paged_scheduler(**POOL_KW)
        exhaust(scheduler)
        for _ in range(100):
            if scheduler.is_idle():
                break
            scheduler.step()
            held = held_blocks(scheduler)
            assert len(held) == len(set(held)), "one block in two tables"
            assert scheduler.block_allocator.num_allocated() == len(held)
            allocated, capacity = scheduler.kv_state
            assert allocated == len(held) * BLOCK
            assert capacity == scheduler.block_allocator.num_blocks * BLOCK
            if scheduler.last_step_plan is not None:
                rows, prefill, decode = scheduler.last_step_plan
                assert rows >= 0 and prefill >= 0 and decode >= 0
        assert scheduler.is_idle()


class TestGreedyOracle:
    def _paged_scheduler(self, model, num_kv_blocks=None):
        config = BatchingConfig(
            max_batch=2, max_seq_len=16, max_tokens_per_step=16,
            paged_kv=True, block_size=BLOCK, num_kv_blocks=num_kv_blocks,
        )
        pool = PagedKVPool(
            num_layers=TINY_ARCH["num_transformers"],
            num_kv_blocks=config.resolved_kv_blocks, block_size=BLOCK,
            max_seq_len=16, num_groups=TINY_ARCH["num_groups"],
            head_dim=TINY_ARCH["head_dim"], dtype=torch.float32,
            device=torch.device("cpu"),
        )
        return ContinuousBatchingScheduler(
            forward_fn=forward_fn(model), pool=pool,
            allocator=SlotAllocator(2), config=config,
            block_allocator=BlockAllocator(config.resolved_kv_blocks),
            paged_state=PagedStepState(
                max_rows=2, max_blocks_per_seq=16 // BLOCK,
                num_kv_blocks=config.resolved_kv_blocks,
                device=torch.device("cpu"),
            ),
        )

    def test_preempted_greedy_stream_token_identical_to_unconstrained(self):
        """The §2.9 correctness bar, model-free testable: greedy through
        an eviction and resume must emit exactly the tokens a
        never-preempted run emits — the re-prefilled prefix reproduces
        the KV state bit-for-bit at this dtype, and everything after is
        determined by it."""
        _, flex_a = build_arms()
        _, flex_b = build_arms()
        arrivals = lambda: {  # noqa: E731 — fresh requests per drive
            0: [
                make_request("a", PROMPT_A, max_tokens=8),
                make_request("b", PROMPT_B, max_tokens=8),
            ],
        }
        unconstrained = drain(self._paged_scheduler(flex_a), arrivals())
        constrained_scheduler = self._paged_scheduler(flex_b, num_kv_blocks=4)
        constrained = drain(constrained_scheduler, arrivals())
        assert_streams_match(unconstrained, constrained)
        assert all(r["tokens"] for r in unconstrained.values())
        assert constrained_scheduler.block_allocator.num_allocated() == 0
