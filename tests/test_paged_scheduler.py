"""Scheduler paged mode (P4 chunk 5): block accounting + step-state fill.

The chunk's first exit gate: toy-stepper-style block accounting with no
leaks: promotion takes a first block, planning reserves through each
row's grant, finish/abort return everything, shortage trims grants,
and a genuinely stuck pool preempts one sequence so the batch can continue.
The forward here is a stub returning
constant logits: chunk 4's equivalence suite already proved the attend,
so these tests isolate the scheduling; the engine-level oracle
(test_paged_engine_oracle.py) is the second gate and runs the real
model.
"""

import pytest
import torch

from cantollm.engine.batching import BatchingConfig
from cantollm.engine.batching.allocator import BlockAllocator, SlotAllocator
from cantollm.engine.batching.paging import PagedStepState
from cantollm.engine.batching.scheduler import ContinuousBatchingScheduler
from cantollm.engine.batching.shaping import FILLER_SPEC
from cantollm.kv_pool import PagedKVPool
from tests.test_cb_scheduler import make_request

VOCAB = 32
BLOCK = 4
FAVORITE = 7  # the constant argmax token; keep it out of stop sets


class ConstantForward:
    """BatchedForwardFn stub: fixed logits, so greedy always emits
    FAVORITE and the tests exercise accounting, not attention. Records
    each step's meta so tests can inspect what the model would have
    seen."""

    def __init__(self):
        self.metas = []

    def __call__(self, input_ids, meta, pool):
        self.metas.append(meta)
        logits = torch.zeros(input_ids.shape[0], VOCAB)
        logits[:, FAVORITE] = 1.0
        return logits


def make_paged_scheduler(
    max_batch: int = 2,
    max_seq_len: int = 32,
    max_tokens_per_step: int = 8,
    num_kv_blocks: int | None = None,
    **config_overrides,
) -> tuple[ContinuousBatchingScheduler, ConstantForward]:
    config = BatchingConfig(
        max_batch=max_batch, max_seq_len=max_seq_len,
        max_tokens_per_step=max_tokens_per_step,
        paged_kv=True, block_size=BLOCK, num_kv_blocks=num_kv_blocks,
        **config_overrides,
    )
    pool = PagedKVPool(
        num_layers=1, num_kv_blocks=config.resolved_kv_blocks,
        block_size=BLOCK, max_seq_len=max_seq_len, num_groups=1,
        head_dim=8, dtype=torch.float32, device=torch.device("cpu"),
    )
    forward = ConstantForward()
    scheduler = ContinuousBatchingScheduler(
        forward_fn=forward,
        pool=pool,
        allocator=SlotAllocator(max_batch),
        config=config,
        block_allocator=BlockAllocator(config.resolved_kv_blocks),
        paged_state=PagedStepState(
            max_rows=max_batch,
            max_blocks_per_seq=max_seq_len // BLOCK,
            num_kv_blocks=config.resolved_kv_blocks,
            device=torch.device("cpu"),
        ),
    )
    return scheduler, forward


def assert_no_leaks(scheduler: ContinuousBatchingScheduler) -> None:
    """The accounting invariant: every allocated block is in exactly one
    active sequence's table, and tables match consumed positions."""
    allocator = scheduler.block_allocator
    held = [b for s in scheduler.active for b in s.block_table]
    assert len(held) == len(set(held)), "one block in two tables"
    assert allocator.num_allocated() == len(held)
    for seq in scheduler.active:
        # Reservation covers exactly through the consumed position once a
        # step's grant lands (position advances to the reserved end).
        assert len(seq.block_table) == -(-seq.position // BLOCK) or (
            seq.position == 0 and len(seq.block_table) == 1
        ), (
            f"{seq.request_id}: {len(seq.block_table)} blocks at "
            f"position {seq.position}"
        )


def run_to_completion(scheduler, max_steps: int = 200) -> None:
    for _ in range(max_steps):
        if scheduler.is_idle():
            return
        scheduler.step()
        assert_no_leaks(scheduler)
    raise AssertionError("scheduler did not converge")


# ── PagedStepState.fill ──────────────────────────────────────────────


class TestFill:
    def make_state(self, max_rows=2, max_blocks_per_seq=8, num_kv_blocks=16):
        return PagedStepState(
            max_rows=max_rows, max_blocks_per_seq=max_blocks_per_seq,
            num_kv_blocks=num_kv_blocks, device=torch.device("cpu"),
        )

    def test_real_rows_fill_tables_and_inverse(self):
        state = self.make_state()
        tables = state.fill(
            [(0, 3, 6), (1, 0, 1)], [[5, 2, 9], [4]], BLOCK
        )
        assert tables.block_tables[0].tolist()[:3] == [5, 2, 9]
        assert tables.block_tables[1].tolist()[:1] == [4]
        # Row 0 reaches position 9: ceil(9 / 4) = 3 visible blocks.
        assert tables.kv_num_blocks.tolist() == [3, 1]
        inv = tables.inverse_tables
        assert inv[0, 5] == 0 and inv[0, 2] == 1 and inv[0, 9] == 2
        assert inv[1, 4] == 0
        # Unowned physical blocks keep the past-any-bound sentinel.
        assert inv[0, 4] == state.max_blocks_per_seq
        # The write map covers exactly the real new tokens.
        assert tables.write_map.pool_index.tolist() == [
            5 * BLOCK + 3, 2 * BLOCK + 0, 2 * BLOCK + 1, 2 * BLOCK + 2,
            2 * BLOCK + 3, 9 * BLOCK + 0,        # row 0, positions 3..8
            4 * BLOCK + 0,                        # row 1, position 0
        ]

    def test_filler_rows_point_at_scratch(self):
        state = self.make_state()
        tables = state.fill([(0, 0, 2), FILLER_SPEC], [[3]], BLOCK)
        scratch = state.num_kv_blocks
        assert tables.block_tables[1, 0] == scratch
        assert tables.kv_num_blocks[1] == 1
        # Scratch maps to logical block 0 so the filler's softmax row has
        # at least one visible (garbage, unread) position; an
        # all-sentinel inverse would mask the row completely (NaN).
        assert tables.inverse_tables[1, scratch] == 0
        # Fillers write nothing.
        assert tables.write_map.batch_row.tolist() == [0, 0]
        # A real row never sees the scratch block.
        assert tables.inverse_tables[0, scratch] == state.max_blocks_per_seq

    def test_refill_resets_the_previous_step(self):
        state = self.make_state()
        state.fill([(0, 0, 8)], [[5, 2]], BLOCK)
        tables = state.fill([(0, 0, 4)], [[3]], BLOCK)
        inv = tables.inverse_tables
        # Blocks 5 and 2 belonged to the previous step's row only.
        assert inv[0, 5] == state.max_blocks_per_seq
        assert inv[0, 2] == state.max_blocks_per_seq
        assert inv[0, 3] == 0
        assert tables.block_tables[0].tolist()[:2] == [3, 0]
        assert tables.kv_num_blocks.tolist() == [1]

    def test_views_share_the_persistent_buffers(self):
        # The seeded-buffer discipline chunk 6 builds on: fill hands out
        # views of the same storage every step, never fresh tensors.
        state = self.make_state()
        first = state.fill([(0, 0, 2)], [[1]], BLOCK)
        second = state.fill([(0, 2, 2)], [[1]], BLOCK)
        assert first.block_tables.data_ptr() == state.block_tables.data_ptr()
        assert second.block_tables.data_ptr() == state.block_tables.data_ptr()
        assert second.inverse_tables.data_ptr() == state.inverse_tables.data_ptr()

    def test_short_table_raises(self):
        state = self.make_state()
        with pytest.raises(ValueError, match="holds 1 block"):
            state.fill([(0, 0, 6)], [[5]], BLOCK)

    def test_too_many_rows_raises(self):
        state = self.make_state(max_rows=1)
        with pytest.raises(ValueError, match="buffers hold 1"):
            state.fill([(0, 0, 1), (0, 0, 1)], [[1], [2]], BLOCK)


# ── Construction ─────────────────────────────────────────────────────


class TestConstruction:
    def test_paged_config_requires_the_paged_trio(self):
        config = BatchingConfig(
            max_batch=2, max_seq_len=32, max_tokens_per_step=8,
            paged_kv=True, block_size=BLOCK,
        )
        pool = PagedKVPool(
            num_layers=1, num_kv_blocks=config.resolved_kv_blocks,
            block_size=BLOCK, max_seq_len=32, num_groups=1, head_dim=8,
            dtype=torch.float32, device=torch.device("cpu"),
        )
        with pytest.raises(ValueError, match="block_allocator"):
            ContinuousBatchingScheduler(
                forward_fn=ConstantForward(), pool=pool,
                allocator=SlotAllocator(2), config=config,
            )

    def test_padded_config_rejects_the_paged_trio(self):
        from tests.toy_stepper import ToyStepper, make_toy_pool

        config = BatchingConfig(
            max_batch=2, max_seq_len=32, max_tokens_per_step=8
        )
        with pytest.raises(ValueError, match="padded configs take neither"):
            ContinuousBatchingScheduler(
                forward_fn=ToyStepper(), pool=make_toy_pool(config),
                allocator=SlotAllocator(2), config=config,
                block_allocator=BlockAllocator(4),
                paged_state=PagedStepState(
                    max_rows=2, max_blocks_per_seq=8, num_kv_blocks=4,
                    device=torch.device("cpu"),
                ),
            )

    def test_block_count_mismatch_raises(self):
        scheduler, _ = make_paged_scheduler()
        config, pool = scheduler.config, scheduler.pool
        with pytest.raises(ValueError, match="resolves"):
            ContinuousBatchingScheduler(
                forward_fn=ConstantForward(), pool=pool,
                allocator=SlotAllocator(2), config=config,
                block_allocator=BlockAllocator(3),
                paged_state=scheduler.paged_state,
            )


# ── Block accounting through real scheduling ─────────────────────────


class TestBlockAccounting:
    def test_lifecycle_returns_every_block(self):
        scheduler, _ = make_paged_scheduler()
        capacity = scheduler.block_allocator.num_blocks
        scheduler.add_request(make_request("a", [1, 2, 3, 4, 5], max_tokens=6))
        scheduler.add_request(make_request("b", list(range(1, 13)), max_tokens=3))
        run_to_completion(scheduler)
        assert scheduler.block_allocator.num_allocated() == 0
        assert scheduler.block_allocator.num_free() == capacity

    def test_tables_grow_with_positions(self):
        scheduler, _ = make_paged_scheduler(max_tokens_per_step=4)
        scheduler.add_request(make_request("a", list(range(1, 11)), max_tokens=4))
        seq_positions = []
        while not scheduler.is_idle():
            scheduler.step()
            assert_no_leaks(scheduler)
            for seq in scheduler.active:
                seq_positions.append((seq.position, len(seq.block_table)))
        # Chunked prefill: 10 prompt tokens at 4/step, then decode.
        assert (4, 1) in seq_positions and (8, 2) in seq_positions

    def test_abort_frees_active_blocks(self):
        scheduler, _ = make_paged_scheduler()
        scheduler.add_request(make_request("a", list(range(1, 9)), max_tokens=8))
        scheduler.step()
        assert scheduler.block_allocator.num_allocated() > 0
        scheduler.abort("a")
        assert scheduler.block_allocator.num_allocated() == 0
        assert scheduler.active == []

    def test_abort_queued_touches_no_blocks(self):
        # One-slot scheduler: the second arrival waits in the queue and
        # must hold nothing while it does.
        scheduler, _ = make_paged_scheduler(max_batch=1)
        scheduler.add_request(make_request("a", [1, 2], max_tokens=4))
        scheduler.add_request(make_request("b", [3, 4], max_tokens=4))
        scheduler.step()
        queued = scheduler.queued[0]
        assert queued.request_id == "b" and queued.block_table == []
        held = scheduler.block_allocator.num_allocated()
        scheduler.abort("b")
        assert scheduler.block_allocator.num_allocated() == held

    def test_admission_waits_for_a_free_block(self):
        # "a" grows into all four blocks; a later arrival then has a free
        # slot but no block, and promotion must leave it queued (never
        # admit a sequence with nowhere to write) until "a" finishes.
        scheduler, _ = make_paged_scheduler(
            max_seq_len=16, num_kv_blocks=4, max_tokens_per_step=16
        )
        scheduler.add_request(make_request("a", list(range(1, 13)), max_tokens=4))
        scheduler.step()               # full prefill: 3 blocks, 1st token
        scheduler.step()               # decode crosses 12 -> 13: 4th block
        assert scheduler.block_allocator.num_free() == 0
        scheduler.add_request(make_request("b", [1, 2], max_tokens=2))
        scheduler.step()
        assert [s.request_id for s in scheduler.active] == ["a"]
        assert [s.request_id for s in scheduler.queued] == ["b"]
        run_to_completion(scheduler)
        assert scheduler.is_idle()

    def test_shortage_trims_a_prefill_grant(self):
        # Pool of 4 blocks, two rows: "a" reserves three blocks for its
        # 12-token prompt, "b" gets its admission block and then finds
        # the pool empty; its 8-token chunk must shrink to the 4
        # positions that block covers, never overcommit.
        scheduler, forward = make_paged_scheduler(
            max_seq_len=16, num_kv_blocks=4, max_tokens_per_step=32,
            max_batch=2,
        )
        scheduler.add_request(make_request("a", list(range(1, 13)), max_tokens=1))
        scheduler.add_request(make_request("b", list(range(1, 9)), max_tokens=1))
        scheduler.step()
        assert_no_leaks(scheduler)
        first = forward.metas[0]
        assert first.rows[0][2] == 12          # "a": full prompt
        assert first.rows[1][2] == 4           # "b": trimmed to one block
        run_to_completion(scheduler)

    def test_starved_row_stays_active_and_finishes_later(self):
        # A 4-block pool: "a" (4-token prompt, 4 outputs) takes the last
        # free block for its first decode boundary, so "b" (8-token
        # prompt, 5 outputs) starves at ITS boundary: zero-grant steps
        # where its position does not move, while it stays active. When
        # "a" finishes and frees, "b" resumes and completes.
        scheduler, _ = make_paged_scheduler(
            max_seq_len=16, num_kv_blocks=4, max_tokens_per_step=16,
            max_batch=2,
        )
        scheduler.add_request(make_request("a", [1, 2, 3, 4], max_tokens=4))
        scheduler.add_request(make_request("b", [5, 6, 7, 8, 9, 10, 11, 12],
                                           max_tokens=5))
        starved_steps = 0
        for _ in range(50):
            if scheduler.is_idle():
                break
            before = {s.request_id: s.position for s in scheduler.active}
            scheduler.step()
            assert_no_leaks(scheduler)
            for seq in scheduler.active:
                if seq.position == before.get(seq.request_id, -1):
                    starved_steps += 1
                    assert seq.request_id == "b"
        assert scheduler.is_idle()
        assert starved_steps >= 1, "the shortage never actually starved b"

    def test_true_deadlock_preempts_and_completes(self):
        # Both rows eventually park on block boundaries with no free block.
        # The newest active row, "b", must be evicted and reset for replay,
        # allowing "a" to finish and release enough blocks for both requests
        # to complete.
        scheduler, _ = make_paged_scheduler(
            max_seq_len=16, num_kv_blocks=4, max_tokens_per_step=16,
            max_batch=2,
        )
        scheduler.add_request(make_request("a", [1, 2, 3, 4], max_tokens=8))
        scheduler.add_request(make_request("b", [5, 6, 7, 8, 9, 10, 11, 12],
                                           max_tokens=8))
        for _ in range(50):
            scheduler.step()
            assert_no_leaks(scheduler)
            if scheduler.queued and scheduler.queued[0].request_id == "b":
                break
        else:
            raise AssertionError("block-starved sequence was not preempted")

        victim = scheduler.queued[0]
        assert victim.position == 0
        assert victim.slot_idx is None
        assert victim.block_table == []
        assert victim.replay_prefix_token_ids == (
            victim.prompt_token_ids + victim.output_token_ids
        )
        run_to_completion(scheduler)


# ── The meta the forward receives ────────────────────────────────────


class TestSeededMeta:
    def test_meta_carries_the_sequences_tables(self):
        scheduler, forward = make_paged_scheduler()
        scheduler.add_request(make_request("a", list(range(1, 7)), max_tokens=2))
        scheduler.step()
        meta = forward.metas[0]
        tables = meta.paged_tables
        seq = scheduler.active[0]
        n = len(seq.block_table)
        assert tables.block_tables[0, :n].tolist() == seq.block_table
        assert tables.kv_num_blocks[0] == n
        assert tables.write_map.pool_index.numel() == 6

    def test_bucketed_step_routes_fillers_to_scratch(self):
        scheduler, forward = make_paged_scheduler(
            batch_buckets=(2,),
        )
        scheduler.add_request(make_request("a", [1, 2, 3], max_tokens=1))
        scheduler.step()
        meta = forward.metas[0]
        assert len(meta.rows) == 2 and meta.rows[1] == FILLER_SPEC
        tables = meta.paged_tables
        scratch = scheduler.paged_state.num_kv_blocks
        assert tables.block_tables[1, 0] == scratch
        assert tables.kv_num_blocks[1] == 1
        # Filler wrote nothing; the real row wrote its three tokens.
        assert tables.write_map.batch_row.unique().tolist() == [0]


# ── kv_state (the stats hook) ────────────────────────────────────────


class TestKVState:
    def test_paged_reports_block_reservation_in_tokens(self):
        scheduler, _ = make_paged_scheduler()
        capacity_tokens = scheduler.block_allocator.num_blocks * BLOCK
        assert scheduler.kv_state == (0, capacity_tokens)
        scheduler.add_request(make_request("a", list(range(1, 7)), max_tokens=2))
        scheduler.step()
        allocated, capacity = scheduler.kv_state
        assert capacity == capacity_tokens
        assert allocated == len(scheduler.active[0].block_table) * BLOCK

    def test_padded_scheduler_reports_none(self):
        from tests.test_cb_scheduler import make_scheduler

        scheduler, _ = make_scheduler()
        assert scheduler.kv_state is None
