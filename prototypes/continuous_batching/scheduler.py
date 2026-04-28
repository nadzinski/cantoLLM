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

        input_ids = self._get_input_ids()
        slot_metas = self._get_slot_metas()

        logits = self.model.forward(input_ids, slot_metas, self.cache)
        tokens = greedy_sample(logits)

        finished_slots = set()
        token_events = []
        for pos, s in enumerate(self.active_sequences.values()):
            # Append output token and advance position
            output_token = tokens[pos].item()
            s.output_token_ids.append(output_token)
            num_new = len(s.prompt_token_ids) if s.position == 0 else 1
            s.position += num_new

            finish_reason = None
            if output_token in s.stop_token_ids:
                # todo: max tokens finish reason            
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
            prompt_token_ids=request.prompt_token_ids,
            max_tokens=request.max_tokens,
            stop_token_ids=request.stop_token_ids,
        )

    def _get_input_ids(self):
        # todo: need chunked prefill after I make it work without 

        # todo: could not create this every time?
        batch = len(self.active_sequences)
        input_ids = torch.zeros(batch, self.cache.max_seq_len, dtype=torch.int64)

        for b, s in enumerate(self.active_sequences.values()):
            if s.output_token_ids:
                # decode
                input_ids_for_seq = s.output_token_ids[-1:]
            else:
                # prefill
                input_ids_for_seq = s.prompt_token_ids

            input_ids[b][:len(input_ids_for_seq)] = torch.tensor(input_ids_for_seq, dtype=input_ids.dtype)

        return input_ids

    def _get_slot_metas(self):
        return [
            (
                s.slot_idx, 
                s.position, 
                len(s.prompt_token_ids) if s.position == 0 else 1
            )
            for s in self.active_sequences.values()
        ]

    def _allocate_free_slots(self):
        while self.queued_sequences:
            if self.cache.num_free() == 0:
                return

            seq = self.queued_sequences.pop() 

            idx = self.cache.allocate_slot()
            seq.slot_idx = idx

            self.active_sequences[idx] = seq
