"""Chunk-12 overlap suite (paged-kv-plan.md §2.12, §5.12).

The launch/reap split must be invisible in the streams: token-for-token
with the serial loop on both layouts, finalize one step late, the
late-stop bonus decode rolled back without a trace, abort-in-flight
with no double-free, and preemption still correct with a step in
flight. ToyStepper is the padded oracle here deliberately: it is a real
(seeded, deterministic) attention layer over the KV pool, so a
position or rollback bug changes tokens; the paged arm runs the
weight-shared tiny Qwen3 pair from the engine oracle. Off-CUDA the
side-stream D2H degenerates to a synchronous copy, which is exactly
why the whole machine is CPU-testable.
"""

import torch

from cantollm.engine.batching import BatchingConfig, SlotAllocator
from cantollm.engine.batching.allocator import BlockAllocator
from cantollm.engine.batching.paging import PagedStepState
from cantollm.engine.batching.scheduler import ContinuousBatchingScheduler
from cantollm.kv_pool import PagedKVPool
from tests.test_cb_scheduler import drain, make_request, make_scheduler
from tests.test_paged_engine_oracle import (
    assert_streams_match,
    build_arms,
    forward_fn,
)
from tests.test_paged_scheduler import (
    BLOCK,
    assert_no_leaks,
    make_paged_scheduler,
)
from tests.test_preemption import POOL_KW, exhaust
from tests.tiny_model import TINY_ARCH
from tests.toy_stepper import ToyStepper, make_toy_pool


def make_overlap_scheduler(
    max_batch: int = 2, max_seq_len: int = 64, max_tokens_per_step: int = 8
) -> ContinuousBatchingScheduler:
    config = BatchingConfig(
        max_batch=max_batch, max_seq_len=max_seq_len,
        max_tokens_per_step=max_tokens_per_step,
        overlap_scheduling=True,
    )
    return ContinuousBatchingScheduler(
        forward_fn=ToyStepper(),
        pool=make_toy_pool(config),
        allocator=SlotAllocator(max_batch),
        config=config,
    )


def tiny_paged_scheduler(model, *, overlap: bool) -> ContinuousBatchingScheduler:
    config = BatchingConfig(
        max_batch=2, max_seq_len=16, max_tokens_per_step=16,
        paged_kv=True, block_size=BLOCK,
        overlap_scheduling=overlap,
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


ARRIVALS = lambda: {  # noqa: E731 — fresh requests per drive
    0: [
        make_request("a", [1, 2, 3], max_tokens=6),
        make_request("b", [4, 5, 6, 7, 8, 9, 10], max_tokens=4),
    ],
    3: [make_request("c", [11, 12], max_tokens=5)],
}


class TestSerialEquivalence:
    def test_padded_streams_token_identical_to_serial(self):
        serial, _ = make_scheduler()
        overlap = make_overlap_scheduler()
        expected = drain(serial, ARRIVALS())
        got = drain(overlap, ARRIVALS())
        assert set(expected) == set(got)
        for rid in expected:
            assert got[rid]["tokens"] == expected[rid]["tokens"], rid
            assert got[rid]["finish"] == expected[rid]["finish"], rid
            assert got[rid]["errors"] == expected[rid]["errors"] == [], rid

    def test_paged_streams_token_identical_to_serial(self):
        _, flex_a = build_arms()
        _, flex_b = build_arms()
        arrivals = lambda: {  # noqa: E731
            0: [
                make_request("a", [1, 2, 3, 4], max_tokens=8),
                make_request("b", [5, 6, 7, 8, 9, 10, 11, 12], max_tokens=8),
            ],
        }
        serial = drain(tiny_paged_scheduler(flex_a, overlap=False), arrivals())
        overlap_scheduler = tiny_paged_scheduler(flex_b, overlap=True)
        overlapped = drain(overlap_scheduler, arrivals())
        assert_streams_match(serial, overlapped)
        assert overlap_scheduler.block_allocator.num_allocated() == 0

    def test_stop_token_stream_matches_serial(self):
        # Learn the first greedy token, then stop on it in both arms:
        # the late-detected stop is the rollback path (§2.12).
        probe = drain(make_overlap_scheduler(),
                      {0: [make_request("p", [1, 2, 3], max_tokens=3)]})
        stop = probe["p"]["tokens"][0]

        def arrivals():
            return {0: [make_request(
                "s", [1, 2, 3], max_tokens=5, stop_token_ids={stop},
            )]}

        serial, _ = make_scheduler()
        expected = drain(serial, arrivals())
        got = drain(make_overlap_scheduler(), arrivals())
        assert expected["s"]["tokens"] == got["s"]["tokens"] == []
        assert expected["s"]["finish"] == got["s"]["finish"] == "end_turn"


class TestDeferredFinalize:
    def test_events_arrive_one_step_late(self):
        scheduler = make_overlap_scheduler()
        scheduler.add_request(make_request("a", [1, 2], max_tokens=3))
        first = scheduler.step()    # launch only: prefill+sample in flight
        assert first == []
        assert not scheduler.is_idle()   # the flight owes its reap
        second = scheduler.step()   # reap step 1 (and launch step 2)
        assert any(e.token_id is not None for e in second)

    def test_trailing_reap_then_idle(self):
        scheduler = make_overlap_scheduler()
        scheduler.add_request(make_request("a", [1, 2], max_tokens=1))
        seen_finish = False
        for _ in range(20):
            if scheduler.is_idle():
                break
            for evt in scheduler.step():
                if evt.finish_reason is not None:
                    seen_finish = True
        assert scheduler.is_idle() and seen_finish
        assert scheduler._in_flight is None
        assert scheduler.allocator.num_free() == 2   # slot came back


class TestRollback:
    def test_max_tokens_bonus_decode_rolls_back(self):
        # One request to its max_tokens: the finish is detected at reap,
        # one step after a bonus decode launched. The rollback must leave
        # no trace: exact token count, one finish, all KV back.
        scheduler, _ = make_paged_scheduler(
            max_seq_len=16, num_kv_blocks=4, max_tokens_per_step=16,
            max_batch=2, overlap_scheduling=True,
        )
        scheduler.add_request(make_request("a", [1, 2, 3, 4], max_tokens=8))
        tokens = 0
        finishes = 0
        for _ in range(40):
            if scheduler.is_idle():
                break
            for evt in scheduler.step():
                assert evt.error is None, evt.error
                tokens += evt.token_id is not None
                finishes += evt.finish_reason is not None
            assert_no_leaks(scheduler)
        assert scheduler.is_idle()
        assert (tokens, finishes) == (8, 1)
        assert scheduler.block_allocator.num_allocated() == 0

    def test_abort_while_in_flight(self):
        scheduler = make_overlap_scheduler()
        scheduler.add_request(make_request("a", [1, 2, 3], max_tokens=8))
        scheduler.step()             # launch; the step is now in flight
        scheduler.abort("a")
        events = []
        for _ in range(10):
            if scheduler.is_idle():
                break
            events.extend(scheduler.step())
        assert scheduler.is_idle()
        # Exactly one abort ack; the in-flight sample never emits after it.
        finishes = [e for e in events if e.finish_reason is not None]
        assert [e.finish_reason for e in finishes] == ["abort"]
        after_ack = events[events.index(finishes[0]) + 1:]
        assert all(e.token_id is None for e in after_ack)
        assert scheduler.allocator.num_free() == 2


class TestPreemptionUnderOverlap:
    def test_exhaustion_still_completes_append_only(self):
        # The chunk-9 exhaustion shape with the flag on: eviction defers
        # to a settled machine (no step in flight) and the streams stay
        # append-only by exact counts.
        scheduler, _ = make_paged_scheduler(
            **POOL_KW, overlap_scheduling=True
        )
        exhaust(scheduler)
        token_counts = {"a": 0, "b": 0}
        finish_counts = {"a": 0, "b": 0}
        for _ in range(150):
            if scheduler.is_idle():
                break
            for evt in scheduler.step():
                assert evt.error is None, evt.error
                if evt.token_id is not None:
                    token_counts[evt.request_id] += 1
                if evt.finish_reason is not None:
                    assert evt.finish_reason == "max_tokens"
                    finish_counts[evt.request_id] += 1
        assert scheduler.is_idle()
        assert token_counts == {"a": 8, "b": 8}
        assert finish_counts == {"a": 1, "b": 1}
        assert scheduler.preemptions_total >= 1
        assert scheduler.block_allocator.num_allocated() == 0
