"""PagedTables seeding + device-move survival (P4 chunk 3).

`seed_paged_tables` mirrors `seed_kv_write_map`'s discipline (seed-once,
loud validation), and `move_batch_to` must carry BOTH seeded passengers
through its `replace()` — the 40fbcf9 family. The survival test is
MPS-gated: on CPU the `.to` identity gate never opens (same-device `.to`
returns the same object), which is the point of the gate.
"""

import pytest
import torch

import cantollm.engine  # noqa: F401  (engine must import before runtime)
from cantollm.models.attention import BatchMeta, PagedTables
from cantollm.models.attention.protocol import KVWriteMap, PagedKVWriteMap
from cantollm.runtime import move_batch_to

SENTINEL = 8  # past-any-bound logical block index for a 32-token/4-block geometry


def make_meta(row_specs: list[tuple[int, int, int]]) -> BatchMeta:
    start_pos = torch.tensor([r[1] for r in row_specs])
    num_new = torch.tensor([r[2] for r in row_specs])
    num_new_max = int(num_new.max())
    return BatchMeta(
        rows=list(row_specs),
        slots=torch.tensor([r[0] for r in row_specs]),
        start_pos=start_pos,
        num_new=num_new,
        positions=start_pos[:, None] + torch.arange(num_new_max)[None, :],
        num_new_max=num_new_max,
        max_history_len=int((start_pos + num_new).max()),
    )


def make_tables(
    batch: int = 2, total_new: int = 3, **overrides
) -> PagedTables:
    fields = dict(
        block_tables=torch.zeros((batch, 8), dtype=torch.int32),
        kv_num_blocks=torch.ones(batch, dtype=torch.int32),
        inverse_tables=torch.full((batch, 5), SENTINEL, dtype=torch.int32),
        write_map=PagedKVWriteMap(
            row=torch.zeros(total_new, dtype=torch.long),
            off=torch.arange(total_new, dtype=torch.long),
            dst=torch.arange(total_new, dtype=torch.long),
        ),
    )
    fields.update(overrides)
    return PagedTables(**fields)


class TestSeeding:
    def test_seed_then_read_returns_the_same_references(self):
        meta = make_meta([(0, 0, 3), (0, 4, 1)])
        tables = make_tables()
        meta.seed_paged_tables(tables)
        assert meta.paged_tables is tables
        assert meta.paged_tables.block_tables is tables.block_tables

    def test_unseeded_read_raises(self):
        meta = make_meta([(0, 0, 3)])
        with pytest.raises(ValueError, match="no paged tables"):
            _ = meta.paged_tables

    def test_seed_twice_raises(self):
        meta = make_meta([(0, 0, 3)])
        meta.seed_paged_tables(make_tables())
        with pytest.raises(ValueError, match="already seeded"):
            meta.seed_paged_tables(make_tables())

    def test_table_dtype_must_be_int32(self):
        # int64 tables would silently violate the from_kv_blocks contract.
        meta = make_meta([(0, 0, 3)])
        bad = make_tables(
            kv_num_blocks=torch.ones(2, dtype=torch.long),
        )
        with pytest.raises(ValueError, match="int32"):
            meta.seed_paged_tables(bad)

    def test_write_map_columns_must_be_int64_and_aligned(self):
        meta = make_meta([(0, 0, 3)])
        with pytest.raises(ValueError, match="int64"):
            meta.seed_paged_tables(make_tables(
                write_map=PagedKVWriteMap(
                    row=torch.zeros(3, dtype=torch.int32),
                    off=torch.zeros(3, dtype=torch.int32),
                    dst=torch.zeros(3, dtype=torch.int32),
                ),
            ))
        with pytest.raises(ValueError, match="aligned"):
            meta.seed_paged_tables(make_tables(
                write_map=PagedKVWriteMap(
                    row=torch.zeros(3, dtype=torch.long),
                    off=torch.zeros(3, dtype=torch.long),
                    dst=torch.zeros(2, dtype=torch.long),
                ),
            ))

    def test_batch_dims_must_agree(self):
        meta = make_meta([(0, 0, 3)])
        with pytest.raises(ValueError, match="disagree on B"):
            meta.seed_paged_tables(make_tables(
                kv_num_blocks=torch.ones(3, dtype=torch.int32),
            ))


class TestMoveSurvival:
    def test_cpu_to_cpu_is_identity_and_keeps_seeds(self):
        # Same device: the identity gate stays shut, the meta object (and
        # its seeded slots) pass through untouched.
        meta = make_meta([(0, 0, 3)])
        tables = make_tables()
        meta.seed_paged_tables(tables)
        input_ids = torch.zeros((1, 3), dtype=torch.int64)
        moved_ids, moved_meta = move_batch_to(
            input_ids, meta, torch.device("cpu")
        )
        assert moved_meta is meta
        assert moved_meta.paged_tables is tables

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="needs MPS"
    )
    def test_device_move_carries_both_seeded_passengers(self):
        # The 40fbcf9 family, third appearance: replace() builds a fresh
        # meta with empty seeded slots; both passengers must be re-seeded
        # on the far side, on-device.
        mps = torch.device("mps")
        meta = make_meta([(0, 0, 3), (1, 4, 1)])
        meta.seed_kv_write_map(KVWriteMap(
            row=torch.zeros(2, dtype=torch.long),
            off=torch.zeros(2, dtype=torch.long),
            slot=torch.zeros(2, dtype=torch.long),
            pos=torch.zeros(2, dtype=torch.long),
        ))
        meta.seed_paged_tables(make_tables())
        input_ids = torch.zeros((2, 3), dtype=torch.int64)

        moved_ids, moved = move_batch_to(input_ids, meta, mps)

        assert moved is not meta
        assert moved_ids.device.type == "mps"
        assert moved.positions.device.type == "mps"
        # Both seeds survived, on-device — no derivation, no error.
        wm = moved.__dict__["kv_write_map"]
        assert wm.slot.device.type == "mps"
        tables = moved.paged_tables
        assert tables.block_tables.device.type == "mps"
        assert tables.kv_num_blocks.device.type == "mps"
        assert tables.inverse_tables.device.type == "mps"
        assert tables.write_map.dst.device.type == "mps"
        # Values intact through the move.
        assert tables.inverse_tables.cpu().eq(SENTINEL).all()