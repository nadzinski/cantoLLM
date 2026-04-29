"""Continuous-batching scheduler. ** YOU FILL THIS IN. **

See INSTRUCTIONS.md for the contract. HINTS.md if stuck.
"""

"""
NADIA's RANDOM NOTES:

Questions:

* Based on internet reading I thought we were going to do this in one big block
  with masking, without using batches...

Extras:

* one way to think of the KV cache is it stores old tokens plus the state needed to produce
  them that is relevant for future tokens
"""
from collections import deque

import torch

from continuous_batching.cb_types import Request, Sequence, TokenEvent
from continuous_batching.padded_kv import PaddedKVCache
from continuous_batching.toy_model import ToyModel
from continuous_batching.sampler import greedy_sample


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
        self.queued_sequences = deque()
        self.active_sequences = {}

    def add_request(self, request: Request) -> None:
        seq = self._create_sequence_from_request(request)
        self.queued_sequences.appendleft(seq)

    def step(self) -> list[TokenEvent]:
        # import pdb; pdb.set_trace()
        self._allocate_free_slots()

        input_ids, num_news = self._get_input_ids_and_num_news()
        slot_metas = [
            (
                s.slot_idx,
                s.position,
                num_news[s.slot_idx]
            )
            for s in self.active_sequences.values()
        ]

        logits = self.model(input_ids, slot_metas, self.cache)
        tokens = greedy_sample(logits)

        finished_slots = set()
        token_events = []
        for pos, s in enumerate(self.active_sequences.values()):
            # advance position
            num_new = num_news[s.slot_idx]
            s.position += num_new

            # If we're mid-prefill we have no output token
            if s.position < len(s.prompt_token_ids):
                continue

            output_token = tokens[pos].item()
            s.output_token_ids.append(output_token)

            finish_reason = None
            if output_token in s.stop_token_ids:
                finish_reason = "end_turn"
            elif len(s.output_token_ids) == s.max_tokens:
                finish_reason = "max_tokens"


            token_events.append(
                TokenEvent(
                    s.request_id,
                    output_token,
                    finish_reason,
                )
            )

            if finish_reason:
                finished_slots.add(s.slot_idx)

        # Free up finished slots
        for idx in finished_slots:
            self.cache.free_slot(idx)
            del self.active_sequences[idx]

        return token_events

    def is_idle(self) -> bool:
        return not self.queued_sequences and not self.active_sequences

    def _create_sequence_from_request(self, request: Request) -> Sequence:
        return Sequence(
            request_id=request.request_id,
            prompt_token_ids=list(request.prompt_token_ids),
            max_tokens=request.max_tokens,
            stop_token_ids=set(request.stop_token_ids),
        )

    def _allocate_tokens(self, requested_tokens: dict[int, int]) -> dict[int, int]:
        # We allocate our max_tokens_per_step budget evenly across seqs
        allocations = {}

        reqs_and_idxs = sorted(
            ((req, idx) for idx, req in requested_tokens.items()), reverse=True
        )

        remaining = self.max_tokens_per_step

        while reqs_and_idxs:
            alloc = remaining // len(reqs_and_idxs)
            if reqs_and_idxs[-1][0] <= alloc:
                # We can do this whole seq for sure
                req, idx = reqs_and_idxs.pop()
                allocations[idx] = req
                remaining = remaining - req
            else:
                # We can't do the whole of any of the remaining seqs
                # so just allocate what we have, first the
                # evenly diving part and then the residue 
                residue = remaining % len(reqs_and_idxs)
                for i, (_, idx) in enumerate(reqs_and_idxs):
                    allocations[idx] = alloc + 1 if i < residue else alloc
                break

        return allocations


    def _get_input_ids_and_num_news(self):
        requested_tokens = {
            s.slot_idx: len(s.prompt_token_ids) - s.position if s.is_prefilling() else 1
            for s in self.active_sequences.values()
        }
        allocated_tokens = self._allocate_tokens(requested_tokens)

        # todo: could not create this every time?
        batch = len(self.active_sequences)
        input_ids = torch.zeros(batch, self.cache.max_seq_len, dtype=torch.int64)
        num_news = {}

        for b, s in enumerate(self.active_sequences.values()):
            allocated = allocated_tokens[s.slot_idx]
            if s.is_prefilling():
                input_ids_for_seq = s.prompt_token_ids[s.position : s.position + allocated]
            else:
                assert(allocated == 1)
                input_ids_for_seq = s.output_token_ids[-1:]

            num_news[s.slot_idx] = allocated

            input_ids[b][:len(input_ids_for_seq)] = torch.tensor(input_ids_for_seq, dtype=input_ids.dtype)

        return input_ids, num_news

    def _allocate_free_slots(self):
        while self.queued_sequences:
            if self.cache.num_free() == 0:
                return

            seq = self.queued_sequences.pop() 

            idx = self.cache.allocate_slot()
            seq.slot_idx = idx

            self.active_sequences[idx] = seq
