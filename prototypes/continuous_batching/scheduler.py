"""Continuous-batching scheduler. ** YOU FILL THIS IN. **

See INSTRUCTIONS.md for the contract. HINTS.md if stuck.
"""

from collections import deque
from dataclasses import dataclass

import torch

from continuous_batching.cb_types import Request, Sequence, TokenEvent
from continuous_batching.padded_kv import PaddedKVCache
from continuous_batching.sampler import greedy_sample
from continuous_batching.toy_model import ToyModel


@dataclass
class Row:
    """One row of the upcoming batched forward pass.

    `start_pos` is captured at plan time so the row carries everything
    needed for the forward call without reaching back into `sequence`.
    """

    sequence: Sequence
    num_new: int
    start_pos: int

    @property
    def slot_meta(self) -> tuple[int, int, int]:
        return (self.sequence.slot_idx, self.start_pos, self.num_new)

    @property
    def input_tokens(self) -> list[int]:
        return self.sequence.input_tokens_at(self.start_pos, self.num_new)


def water_fill(budget: int, caps: list[int]) -> list[int]:
    """Allocate `budget` units across bins, capped per-bin by `caps`.

    Pour `budget` liters into bins whose heights are `caps`: the level
    rises uniformly until either the budget runs out or every bin is
    full. Smallest bins first; a bin that wants less than its fair
    share takes only what it needs, and the leftover is naturally
    redistributed (the next round's `remaining // count` rounds up).
    """
    n = len(caps)
    allocations = [0] * n
    bins_by_cap = sorted(enumerate(caps), key=lambda b: b[1])
    for i, (idx, cap) in enumerate(bins_by_cap):
        give = min(cap, budget // (n - i))
        allocations[idx] = give
        budget -= give
    return allocations


class ContinuousBatchingScheduler:
    def __init__(
        self,
        model: ToyModel,
        cache: PaddedKVCache,
        max_tokens_per_step: int,
    ):
        if max_tokens_per_step < cache.max_batch:
            raise ValueError(
                f"max_tokens_per_step ({max_tokens_per_step}) must be >= "
                f"cache.max_batch ({cache.max_batch})"
            )
        self.model = model
        self.cache = cache
        self.max_tokens_per_step = max_tokens_per_step
        self.queued_sequences: deque[Sequence] = deque()
        self.active_sequences: list[Sequence] = []

    def add_request(self, request: Request) -> None:
        seq = Sequence(
            request_id=request.request_id,
            prompt_token_ids=list(request.prompt_token_ids),
            max_tokens=request.max_tokens,
            stop_token_ids=set(request.stop_token_ids),
        )
        self.queued_sequences.append(seq)

    def is_idle(self) -> bool:
        return not self.queued_sequences and not self.active_sequences

    def step(self) -> list[TokenEvent]:
        self._promote_queued()

        rows = self._plan_step()
        input_ids = self._build_input_ids(rows)
        slot_metas = [row.slot_meta for row in rows]

        logits = self.model(input_ids, slot_metas, self.cache)
        sampled = greedy_sample(logits).tolist()

        events = []
        still_active = []
        for row, token in zip(rows, sampled):
            seq = row.sequence
            seq.position += row.num_new
            if seq.is_prefilling():
                still_active.append(seq)
                continue
            event = self._finalize_output(seq, token)
            events.append(event)
            if event.finish_reason is None:
                still_active.append(seq)
            else:
                self.cache.free_slot(seq.slot_idx)
        self.active_sequences = still_active
        return events

    def _promote_queued(self):
        while self.queued_sequences:
            if self.cache.num_free() == 0:
                return

            seq = self.queued_sequences.popleft()
            seq.slot_idx = self.cache.allocate_slot()
            self.active_sequences.append(seq)

    def _plan_step(self) -> list[Row]:
        requested = [
            len(seq.prompt_token_ids) - seq.position if seq.is_prefilling() else 1
            for seq in self.active_sequences
        ]
        allocated = water_fill(self.max_tokens_per_step, requested)
        return [
            Row(sequence=seq, num_new=n, start_pos=seq.position)
            for seq, n in zip(self.active_sequences, allocated)
        ]

    def _build_input_ids(self, rows: list[Row]) -> torch.Tensor:
        width = max(row.num_new for row in rows)
        input_ids = torch.zeros((len(rows), width), dtype=torch.int64)
        for i, row in enumerate(rows):
            input_ids[i, : row.num_new] = torch.tensor(row.input_tokens, dtype=torch.int64)
        return input_ids

    def _finalize_output(self, seq: Sequence, token: int) -> TokenEvent:
        seq.output_token_ids.append(token)
        return TokenEvent(seq.request_id, token, self._finish_reason(seq, token))

    def _finish_reason(self, seq: Sequence, token: int) -> str | None:
        if token in seq.stop_token_ids:
            return "end_turn"
        if len(seq.output_token_ids) == seq.max_tokens:
            return "max_tokens"
        return None
