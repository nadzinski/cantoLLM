"""Sequential reference engine: one request at a time. The correctness oracle.

Mirrors `src/cantollm/engine/sequential.py:SequentialEngine`, minus the
async/queue plumbing. Tests compare the scheduler's per-request output to
this engine's output token-for-token.

The reference uses the same `PaddedKVCache` the scheduler uses (with
max_batch=1), so a bug in the cache surfaces here first — and is easier to
debug here, where there's no batching.
"""

import torch

from continuous_batching.padded_kv import PaddedKVCache
from continuous_batching.sampler import greedy_sample
from continuous_batching.toy_model import ToyModel
from continuous_batching.cb_types import Request


class SequentialReference:
    def __init__(self, model: ToyModel, max_seq_len: int):
        self.model = model
        self.max_seq_len = max_seq_len

    def generate(self, request: Request) -> list[int]:
        cache = PaddedKVCache(
            max_batch=1, max_seq_len=self.max_seq_len, dim=self.model.dim,
        )
        slot = cache.allocate_slot()
        assert slot is not None

        prompt = list(request.prompt_token_ids)
        prompt_len = len(prompt)

        # Prefill: feed the whole prompt in one shot.
        input_ids = torch.tensor([prompt], dtype=torch.long)
        logits = self.model(input_ids, [(slot, 0, prompt_len)], cache)
        next_tok = int(greedy_sample(logits)[0].item())

        outputs: list[int] = []
        position = prompt_len

        for _ in range(request.max_tokens):
            outputs.append(next_tok)
            if next_tok in request.stop_token_ids:
                break
            input_ids = torch.tensor([[next_tok]], dtype=torch.long)
            logits = self.model(input_ids, [(slot, position, 1)], cache)
            position += 1
            next_tok = int(greedy_sample(logits)[0].item())

        cache.free_slot(slot)
        return outputs
