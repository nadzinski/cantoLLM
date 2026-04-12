"""Slow token generator for testing speculative decoding."""

import random
import time
from collections.abc import Iterator

import torch

from qwen3.generator import TokenGenerator
from qwen3.kv_cache import KVCache


class SlowTokenGenerator:
    """Wraps a TokenGenerator, adding artificial delay to simulate a slower model."""

    def __init__(
        self,
        generator: TokenGenerator,
        delay_mean: float = 0.05,
        delay_jitter: float = 0.02,
    ):
        self.generator = generator
        self.delay_mean = delay_mean
        self.delay_jitter = delay_jitter

    @property
    def temperature(self) -> float:
        return self.generator.temperature

    @property
    def top_p(self) -> float:
        return self.generator.top_p

    def _apply_top_p(self, logits: torch.Tensor) -> torch.Tensor:
        return self.generator._apply_top_p(logits)

    def get_probs(self, logits: torch.Tensor) -> torch.Tensor:
        return self.generator.get_probs(logits)

    def sample(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.generator.sample(logits)

    def _delay(self):
        delay = self.delay_mean + random.uniform(-self.delay_jitter, self.delay_jitter)
        time.sleep(max(0, delay))

    def forward(
        self,
        token_ids: list[int],
        cache: KVCache,
        start_pos: int,
    ) -> torch.Tensor:
        self._delay()
        return self.generator.forward(token_ids, cache, start_pos)

    def generate(
        self,
        input_ids: list[int],
        cache: KVCache,
        stop_token_ids: set[int],
        max_tokens: int,
    ) -> Iterator[int]:
        for token_id in self.generator.generate(input_ids, cache, stop_token_ids, max_tokens):
            self._delay()
            yield token_id
