"""StepStatsCollector tests against the REAL scheduler + toy forward.

The collector is observer-only — these tests pin that its derivations
(rows, prefill/decode split, KV occupancy, pending-event exclusion) match
what the scheduler actually did, across the tricky shapes: chunked prefill,
finish-on-final-prefill-chunk, abort-ack-only steps, and zero-token
rejections. Toy doubles without the real scheduler surface must get no
collector at all.
"""

from cantollm.engine.batching.allocator import SlotAllocator
from cantollm.engine.batching.config import BatchingConfig
from cantollm.engine.batching.scheduler import ContinuousBatchingScheduler
from cantollm.engine.batching.stats import StepStatsCollector
from cantollm.engine.types import InferenceRequest, SamplingParams
from tests.toy_stepper import ToyStepper, make_toy_pool, toy_oracle

GREEDY = SamplingParams.from_temperature_top_p(temperature=0.0, top_p=1.0)


def make_scheduler(max_batch=2, max_seq_len=32, max_tokens_per_step=8):
    config = BatchingConfig(
        max_batch=max_batch,
        max_seq_len=max_seq_len,
        max_tokens_per_step=max_tokens_per_step,
    )
    return ContinuousBatchingScheduler(
        forward_fn=ToyStepper(),
        pool=make_toy_pool(config),
        allocator=SlotAllocator(config.max_batch),
        config=config,
    )


def request(rid: str, prompt: list[int], max_tokens: int = 4,
            stop_token_ids: set[int] | None = None) -> InferenceRequest:
    return InferenceRequest(
        request_id=rid,
        prompt_token_ids=list(prompt),
        sampling_params=GREEDY,
        max_tokens=max_tokens,
        stop_token_ids=stop_token_ids or set(),
    )


def stepped(sched, collector):
    collector.before_step(sched)
    events = sched.step()
    return events, collector.after_step(sched, events)


def test_for_scheduler_rejects_bare_doubles():
    class Bare:
        def add_request(self, r): ...
        def abort(self, rid): ...
        def step(self): return []
        def is_idle(self): return True

    assert StepStatsCollector.for_scheduler(Bare()) is None
    assert StepStatsCollector.for_scheduler(make_scheduler()) is not None


def test_chunked_prefill_split_and_finish():
    # Prompt 12 wide, budget 8: prefill spans two steps, then decode.
    sched = make_scheduler(max_batch=2, max_seq_len=32, max_tokens_per_step=8)
    collector = StepStatsCollector.for_scheduler(sched)
    sched.add_request(request("r1", list(range(1, 13)), max_tokens=2))

    events, s0 = stepped(sched, collector)
    assert events == []                      # still prefilling: no events
    assert (s0.rows, s0.prefill_tokens, s0.decode_tokens) == (1, 8, 0)
    assert s0.queue_depth == 1               # waiting at step start, promoted inside
    assert s0.occupied_slots == 1
    assert s0.kv_tokens == 8
    assert s0.seq == 0 and s0.dur_s > 0

    events, s1 = stepped(sched, collector)   # final chunk: 4 prefill + first sample
    assert len(events) == 1 and events[0].token_id is not None
    assert (s1.rows, s1.prefill_tokens, s1.decode_tokens) == (1, 4, 0)
    assert s1.kv_tokens == 12
    assert s1.queue_depth == 0

    events, s2 = stepped(sched, collector)   # decode to max_tokens=2: token + finish
    assert [e.finish_reason for e in events] == [None, "max_tokens"]
    assert (s2.rows, s2.prefill_tokens, s2.decode_tokens) == (1, 0, 1)
    assert s2.occupied_slots == 0            # freed in-step
    assert s2.kv_tokens == 0
    assert s2.seq == 2


def test_two_rows_mixed_prefill_decode():
    sched = make_scheduler(max_batch=2, max_seq_len=32, max_tokens_per_step=8)
    collector = StepStatsCollector.for_scheduler(sched)
    sched.add_request(request("long", list(range(1, 11)), max_tokens=4))  # 10 wide
    stepped(sched, collector)                # long: 8 prefill
    sched.add_request(request("short", [21, 22], max_tokens=4))

    _, s = stepped(sched, collector)
    # long finishes prefill (2) + short prefills fully (2); water-fill had
    # room for both. Neither decoded an input token yet.
    assert s.rows == 2
    assert s.prefill_tokens == 4 and s.decode_tokens == 0
    assert s.occupied_slots == 2
    assert s.kv_tokens == 12

    _, s = stepped(sched, collector)         # both decode
    assert (s.prefill_tokens, s.decode_tokens) == (0, 2)


def test_pending_events_are_not_rows():
    sched = make_scheduler()
    collector = StepStatsCollector.for_scheduler(sched)
    # Zero-token rejection parks a max_tokens finish in pending_events.
    sched.add_request(request("zero", [1, 2], max_tokens=0))

    events, s = stepped(sched, collector)
    assert [e.finish_reason for e in events] == ["max_tokens"]
    assert s.rows == 0                       # a flush, not a forward
    assert (s.prefill_tokens, s.decode_tokens) == (0, 0)


def test_abort_ack_only_step():
    sched = make_scheduler()
    collector = StepStatsCollector.for_scheduler(sched)
    sched.add_request(request("r1", list(range(1, 13)), max_tokens=4))
    stepped(sched, collector)                # mid-prefill
    sched.abort("r1")

    events, s = stepped(sched, collector)
    assert [e.finish_reason for e in events] == ["abort"]
    assert s.rows == 0
    assert s.occupied_slots == 0 and s.kv_tokens == 0


def test_finish_on_final_prefill_chunk_counts_as_prefill():
    # Learn the first sampled token, then make it a stop token: the row
    # completes prefill and finishes end_turn in the same step, consuming
    # only prompt tokens.
    prompt = list(range(1, 13))
    first_token = toy_oracle(request("probe", prompt, max_tokens=1))[0][0]

    sched = make_scheduler(max_batch=2, max_seq_len=32, max_tokens_per_step=8)
    collector = StepStatsCollector.for_scheduler(sched)
    sched.add_request(
        request("r1", prompt, max_tokens=4, stop_token_ids={first_token})
    )
    stepped(sched, collector)                # 8 prefill

    events, s = stepped(sched, collector)    # 4 prefill, sample hits stop set
    assert [e.finish_reason for e in events] == ["end_turn"]
    assert s.rows == 1
    assert (s.prefill_tokens, s.decode_tokens) == (4, 0)
    assert s.occupied_slots == 0
