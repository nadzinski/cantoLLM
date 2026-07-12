"""Toy BatchedForwardFn + sequential oracle for scheduler tests.

The scheduler never sees a model — just a `BatchedForwardFn` — so its tests
run against this: the prototype's one-layer, one-head toy model
(prototypes/continuous_batching/toy_model.py) re-shaped to the real seams
(BatchMeta + PaddedKVPool with num_layers=1, num_groups=1). 10× faster than
tiny-Qwen3 and pins scheduler logic in isolation from model bugs.

`toy_oracle` is the per-request sequential reference, with the REAL
emission contract (unlike the prototype's reference): stop tokens are
suppressed, `>=` on max_tokens, `max_tokens <= 0` emits nothing — matching
StandardBackend.generate, which the step-9 equivalence tests use as the
oracle on the real model.
"""

import math

import torch

from cantollm.engine import sampler
from cantollm.engine.batching.config import BatchingConfig
from cantollm.engine.types import InferenceRequest
from cantollm.kv_pool import PaddedKVPool
from cantollm.models.attention.protocol import BatchMeta

VOCAB_SIZE = 32
DIM = 16


class ToyStepper:
    """Satisfies BatchedForwardFn against a single-layer PaddedKVPool."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, dim: int = DIM, seed: int = 0):
        self.vocab_size = vocab_size
        self.dim = dim
        gen = torch.Generator().manual_seed(seed)

        def w(*shape):
            return torch.randn(*shape, generator=gen) * 0.1

        self.embedding = w(vocab_size, dim)
        self.w_q = w(dim, dim)
        self.w_k = w(dim, dim)
        self.w_v = w(dim, dim)
        self.w_out = w(dim, dim)
        self.lm_head = w(dim, vocab_size)

    def __call__(
        self, input_ids: torch.Tensor, meta: BatchMeta, pool: PaddedKVPool
    ) -> torch.Tensor:
        k_pool, v_pool = pool.layer(0)  # (max_batch, max_seq_len, 1, dim)
        batch = input_ids.shape[0]
        logits = torch.zeros(batch, self.vocab_size)

        for b, (slot_idx, start_pos, num_new) in enumerate(meta.rows):
            assert num_new >= 1, "scheduler must not pass empty rows"
            embeds = self.embedding[input_ids[b, :num_new]]

            q = embeds @ self.w_q
            k_new = embeds @ self.w_k
            v_new = embeds @ self.w_v

            k_pool[slot_idx, start_pos : start_pos + num_new, 0] = k_new
            v_pool[slot_idx, start_pos : start_pos + num_new, 0] = v_new

            history_len = start_pos + num_new
            k_full = k_pool[slot_idx, :history_len, 0]
            v_full = v_pool[slot_idx, :history_len, 0]

            scores = q @ k_full.T / math.sqrt(self.dim)
            q_positions = torch.arange(start_pos, history_len)
            kv_positions = torch.arange(history_len)
            causal = q_positions[:, None] >= kv_positions[None, :]
            scores = scores.masked_fill(~causal, float("-inf"))

            attn_out = torch.softmax(scores, dim=-1) @ v_full
            logits[b] = (attn_out @ self.w_out)[-1] @ self.lm_head
        return logits


def make_toy_pool(config: BatchingConfig) -> PaddedKVPool:
    return PaddedKVPool(
        num_layers=1, max_batch=config.max_batch, max_seq_len=config.max_seq_len,
        num_groups=1, head_dim=DIM,
        dtype=torch.float32, device=torch.device("cpu"),
    )


def toy_oracle(request: InferenceRequest, seed: int = 0) -> tuple[list[int], str]:
    """Run `request` alone, sequentially, on a fresh stepper + 1-slot pool.

    Returns (emitted_tokens, finish_reason) under the real contract:
    stop tokens suppressed, finish after `max_tokens` emissions,
    `max_tokens <= 0` emits nothing.
    """
    if request.max_tokens <= 0:
        return [], "max_tokens"

    stepper = ToyStepper(seed=seed)
    config = BatchingConfig(
        max_batch=1,
        max_seq_len=len(request.prompt_token_ids) + request.max_tokens,
        max_tokens_per_step=max(1, len(request.prompt_token_ids)),
    )
    pool = make_toy_pool(config)

    def forward(tokens: list[int], start: int) -> torch.Tensor:
        specs = [(0, start, len(tokens))]
        start_t = torch.tensor([start])
        num_new = torch.tensor([len(tokens)])
        meta = BatchMeta(
            rows=specs, slots=torch.tensor([0]),
            start_pos=start_t, num_new=num_new,
            positions=start_t[:, None] + torch.arange(len(tokens))[None, :],
            num_new_max=len(tokens),
            max_history_len=start + len(tokens),
        )
        input_ids = torch.tensor([tokens], dtype=torch.int64)
        return stepper(input_ids, meta, pool)

    emitted: list[int] = []
    position = 0
    logits = forward(list(request.prompt_token_ids), 0)
    position = len(request.prompt_token_ids)
    while True:
        token_t, _ = sampler.sample(logits[0], request.sampling_params)
        token = int(token_t.item())
        if token in request.stop_token_ids:
            return emitted, "end_turn"
        emitted.append(token)
        if len(emitted) >= request.max_tokens:
            return emitted, "max_tokens"
        logits = forward([token], position)
        position += 1
