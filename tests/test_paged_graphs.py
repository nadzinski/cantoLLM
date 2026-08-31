"""P4 chunk 8: CUDA graphs on paged decode (paged-kv-plan.md §5.8).

The design's payoff line: the static buffers ARE the step tables. The
scheduler's `PagedStepState` buffers (tables, inverse, start_pos, and,
new this chunk, the padded decode write map) are persistent and
rewritten in place by `fill()`, so a recording that baked their
addresses reads each later step's values with no table-side marshal.
Graphs key on (batch, 1): kv length is a value (§2.6), so the capture
set is the batch buckets, not |B| x |KV|.

CPU here: fill's decode-map convention and bulk-copy address stability,
the wrapper's paged dispatch/guard logic, and capture-setup buffer
identity (everything up to the recording itself). `TestOnCUDA` holds
the chunk's exit gate: bit-exact replay, including after an in-place
table permutation (the spike's gate-4 probe), validated on the 5090 in
round 2.
"""

from __future__ import annotations

import pytest
import torch

from cantollm.engine.batching.graphs import GraphedBatchedForward
from cantollm.engine.batching.paging import PagedStepState
from cantollm.kv_pool import PagedKVPool
from cantollm.models.attention import FlexAttentionMethod
from cantollm.models.qwen3.model import Qwen3
from cantollm.runtime import ModelRuntime
from cantollm.standard import StandardBackend
from tests.test_paged_compile import (
    BLOCK,
    MAX_SEQ,
    NUM_BLOCKS,
    ids_for,
    make_flex_model,
    make_paged_pool,
    make_runtime,
    make_state,
    paged_config,
    paged_meta_via_state,
)
from tests.tiny_model import TINY_ARCH, tiny_qwen3_spec

CPU = torch.device("cpu")

SCRATCH_START = NUM_BLOCKS * BLOCK


class TestFillDecodeMap:
    """The §3 map spec lands with chunk 8: decode-shaped steps take their
    write map from persistent buffers, padded to the batch with fillers
    parked on the scratch block; prefill steps keep the exact
    one-entry-per-real-token map."""

    def test_decode_map_is_padded_and_persistent(self):
        model = make_flex_model()
        state = make_state(model)
        tables = state.fill(
            [(0, 5, 1), (0, 9, 1), (0, 0, 0)],
            [[0, 3], [1, 2, 4]], BLOCK, 1,
        )
        wm = tables.write_map
        assert wm.batch_row.tolist() == [0, 1, 2]
        assert wm.token_offset.tolist() == [0, 0, 0]
        # Row 0 decodes position 5 -> block 3's offset 1; row 1 position
        # 9 -> block 4's offset 1; the filler parks on scratch.
        assert wm.pool_index.tolist() == [
            3 * BLOCK + 1, 4 * BLOCK + 1, SCRATCH_START,
        ]
        assert wm.pool_index.data_ptr() == state.map_pool_index.data_ptr()
        # Refill rewrites values at the same addresses.
        again = state.fill([(0, 6, 1)], [[0, 3]], BLOCK, 1)
        assert again.write_map.pool_index.data_ptr() == \
            state.map_pool_index.data_ptr()
        assert again.write_map.pool_index.tolist() == [3 * BLOCK + 2]

    def test_prefill_map_stays_exact(self):
        model = make_flex_model()
        state = make_state(model)
        tables = state.fill(
            [(0, 0, 5), (0, 0, 0)], [[5, 2]], BLOCK, 8,
        )
        wm = tables.write_map
        # One entry per real token, none for the filler, fresh tensors
        # (prefill steps are never replayed by a graph).
        assert wm.batch_row.tolist() == [0] * 5
        assert wm.pool_index.data_ptr() != state.map_pool_index.data_ptr()

    def test_table_buffers_move_by_bulk_copy_not_realloc(self):
        # The address-stability contract under the staging rewrite: two
        # very different fills, same storage throughout.
        model = make_flex_model()
        state = make_state(model)
        first = state.fill([(0, 3, 6), (0, 0, 1)], [[5, 2, 9], [4]], BLOCK, 8)
        ptrs = (
            state.block_tables.data_ptr(),
            state.inverse_tables.data_ptr(),
            state.kv_num_blocks.data_ptr(),
            state.start_pos.data_ptr(),
        )
        second = state.fill([(0, 8, 1)], [[7, 6, 1]], BLOCK, 1)
        assert first.block_tables.data_ptr() == ptrs[0]
        assert second.block_tables.data_ptr() == ptrs[0]
        assert second.inverse_tables.data_ptr() == ptrs[1]
        assert state.kv_num_blocks.data_ptr() == ptrs[2]
        assert state.start_pos.data_ptr() == ptrs[3]
        # And the values are the second step's.
        assert second.block_tables[0].tolist()[:3] == [7, 6, 1]
        assert state.start_pos[0].item() == 8


class TestWrapperPaged:
    def _wrapped(self):
        model = make_flex_model()
        runtime = make_runtime(model)
        config = paged_config(warmup_shapes=True)
        pool = make_paged_pool()
        state = make_state(model, max_rows=config.max_batch)
        wrapped = GraphedBatchedForward(
            runtime.forward_batched, config, paged_state=state
        )
        return wrapped, pool, state

    def test_paged_config_requires_the_state(self):
        model = make_flex_model()
        runtime = make_runtime(model)
        with pytest.raises(ValueError, match="PagedStepState"):
            GraphedBatchedForward(
                runtime.forward_batched, paged_config(warmup_shapes=True)
            )

    def test_uncaptured_steps_fall_through_eager(self):
        wrapped, pool, state = self._wrapped()
        meta = paged_meta_via_state(state, [(0, 5)], [[3, 6]])
        out = wrapped(ids_for(meta), meta, pool)
        assert out.shape[0] == 1
        assert wrapped.misses == 1 and wrapped.hits == 0

    def test_replayable_requires_the_state_buffers(self):
        # A meta whose seeded tables are NOT the state's buffers must
        # never replay: the recording reads addresses, not the meta.
        wrapped, pool, state = self._wrapped()
        good = paged_meta_via_state(state, [(4, 1)], [[0, 3]])
        assert wrapped._replayable(good, pool)

        from tests.test_flex_equivalence import paged_meta

        foreign = paged_meta([(4, 1)], [[0, 3]])
        assert not wrapped._replayable(foreign, pool)

    def test_replayable_rejects_prefill_and_overlong(self):
        wrapped, pool, state = self._wrapped()
        prefill = paged_meta_via_state(state, [(0, 3)], [[3]])
        assert not wrapped._replayable(prefill, pool)
        overlong = paged_meta_via_state(
            state, [(MAX_SEQ - 1, 1)], [[i for i in range(MAX_SEQ // BLOCK)]]
        )
        # position MAX_SEQ-1 + 1 == capacity: allowed; one past is not.
        assert wrapped._replayable(overlong, pool)
        too_far = paged_meta_via_state(state, [(4, 1), (0, 0)], [[0, 1]])
        too_far.rows[0] = (0, MAX_SEQ, 1)  # simulate a corrupt plan
        assert not wrapped._replayable(too_far, pool)

    def test_capture_setup_bakes_the_state_buffers(self):
        # Everything the recording would read on the table side must BE
        # the state's storage, and the family mask must ride in.
        wrapped, pool, state = self._wrapped()
        entry, meta = wrapped._paged_capture_setup(2, CPU)
        tables = meta.paged_tables
        assert tables.block_tables.data_ptr() == state.block_tables.data_ptr()
        assert tables.inverse_tables.data_ptr() == \
            state.inverse_tables.data_ptr()
        assert tables.write_map.pool_index.data_ptr() == \
            state.map_pool_index.data_ptr()
        assert tables.mask is state.masks[(2, 1)]
        assert meta.slots is entry.slots
        assert meta.positions is entry.positions
        assert [n for _, _, n in meta.rows] == [1, 1]
        # And the dummy step is itself replayable by its own guard.
        assert wrapped._replayable(meta, pool)

    def test_paged_key_ignores_kv(self):
        # Same family, different histories: one key. The padded triple
        # would have split these into distinct (never-captured) keys.
        wrapped, pool, state = self._wrapped()
        m1 = paged_meta_via_state(state, [(4, 1)], [[0, 3]])
        m2 = paged_meta_via_state(state, [(17, 1)], [[5, 6, 7, 8, 9]])
        k1 = (len(m1.rows), m1.num_new_max)
        k2 = (len(m2.rows), m2.num_new_max)
        assert k1 == k2 == (1, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestOnCUDA:
    """The chunk-8 exit gate (§5.8), validated on the 5090 in round 2:
    replay is bit-exact against the same compiled forward run without
    the graph, including after an IN-PLACE table permutation between
    replays (the spike's gate-4 probe: fill rewrites the persistent
    buffers, the recording reads the new values)."""

    CUDA_BLOCK = 64
    CUDA_MAX_SEQ = 512
    CUDA_NUM_BLOCKS = 12
    CUDA_ARCH = TINY_ARCH | {"head_dim": 16, "max_seq_len": CUDA_MAX_SEQ}
    PROMPT = [(7 * i + 11) % 2048 for i in range(150)]

    def test_replay_bit_exact_and_tracks_table_permutation(self):
        device = torch.device("cuda")
        torch.manual_seed(1234)
        model = Qwen3(
            qwen3_config=self.CUDA_ARCH,
            attention_method=FlexAttentionMethod(block_size=self.CUDA_BLOCK),
        ).eval().to(device)
        runtime = ModelRuntime(
            spec=tiny_qwen3_spec(), device=device, model=model,
            tokenizer=None, backend=StandardBackend(model=model, device=device),
        )
        runtime.enable_torch_compile()  # real Inductor: what capture records
        config = paged_config(
            max_batch=2, max_seq_len=self.CUDA_MAX_SEQ,
            max_tokens_per_step=256, prefill_widths=(256,),
            batch_buckets=(1, 2), block_size=self.CUDA_BLOCK,
            warmup_shapes=True, torch_compile=True,
        )
        pool = PagedKVPool(
            num_layers=self.CUDA_ARCH["num_transformers"],
            num_kv_blocks=self.CUDA_NUM_BLOCKS, block_size=self.CUDA_BLOCK,
            max_seq_len=self.CUDA_MAX_SEQ,
            num_groups=self.CUDA_ARCH["num_groups"],
            head_dim=self.CUDA_ARCH["head_dim"], dtype=torch.float32,
            device=device,
        )
        state = PagedStepState(
            max_rows=2,
            max_blocks_per_seq=self.CUDA_MAX_SEQ // self.CUDA_BLOCK,
            num_kv_blocks=self.CUDA_NUM_BLOCKS, device=device,
            mask_builder=model.attention_method.build_family_mask,
        )
        wrapped = GraphedBatchedForward(
            runtime.forward_batched, config, paged_state=state
        )

        def step(row_specs, tables, toks_per_row, use_wrapper):
            meta = paged_meta_via_state(
                state, row_specs, tables, block_size=self.CUDA_BLOCK
            )
            ids = torch.zeros(
                (len(row_specs), meta.num_new_max), dtype=torch.int64
            )
            for i, toks in enumerate(toks_per_row):
                ids[i, : len(toks)] = torch.tensor(toks)
            fn = wrapped if use_wrapper else runtime.forward_batched
            return fn(ids, meta, pool)

        # Prefill through the eager path (never graphed), then capture.
        table = [5, 2, 9]
        step([(0, 150)], [table], [self.PROMPT], use_wrapper=False)
        captured = wrapped.capture_decode_shapes(pool)
        assert captured == 2  # batch buckets (1, 2)

        # Decode replays vs the same compiled forward, bit exact.
        want = step([(150, 1)], [table], [[77]], use_wrapper=False)
        got = step([(150, 1)], [table], [[77]], use_wrapper=True)
        assert wrapped.hits == 1, "decode step did not replay"
        torch.testing.assert_close(got, want, atol=0, rtol=0)

        # THE gate probe: permute the physical layout in place. Move
        # each logical block's K/V to new physical blocks, refill the
        # tables through the same persistent buffers, and replay: the
        # recording must read the NEW tables and produce the identical
        # logits (kv order is logical either way).
        new_table = [1, 7, 4]
        for i in range(pool.num_layers):
            k, v = pool.layer(i)
            for old, new in zip(table, new_table):
                sl_old = slice(
                    old * self.CUDA_BLOCK, (old + 1) * self.CUDA_BLOCK
                )
                sl_new = slice(
                    new * self.CUDA_BLOCK, (new + 1) * self.CUDA_BLOCK
                )
                k[sl_new] = k[sl_old].clone()
                v[sl_new] = v[sl_old].clone()
        got_permuted = step([(150, 1)], [new_table], [[77]], use_wrapper=True)
        assert wrapped.hits == 2
        torch.testing.assert_close(got_permuted, want, atol=0, rtol=0)

        # And a genuinely new value flows through: the next position
        # replays with different logits (values, not a frozen output).
        step([(150, 1)], [new_table], [[78]], use_wrapper=False)
        different = step([(151, 1)], [new_table], [[78]], use_wrapper=True)
        assert not torch.equal(different, want)
