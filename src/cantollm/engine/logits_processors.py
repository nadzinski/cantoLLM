"""Per-request logits transformations composed as a pipeline.

Each processor maps pre-softmax logits to pre-softmax logits. The sampler
runs the chain and then draws a token. Widening the signature (history,
request state) is deferred until the first caller needs it.
"""

from typing import Protocol

import torch


class LogitsProcessor(Protocol):
    def __call__(self, logits: torch.Tensor) -> torch.Tensor: ...


class TemperatureProcessor:
    def __init__(self, temperature: float):
        self.temperature = temperature

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


class TopPProcessor:
    def __init__(self, top_p: float):
        self.top_p = top_p

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        remove_mask = cumulative_probs > self.top_p

        # Shift right: keep the token that crosses the threshold too
        # (position 0 stays False, ensuring at least the top token survives)
        sorted_indices_to_remove = torch.full_like(remove_mask, False)
        sorted_indices_to_remove[..., 1:] = remove_mask[..., :-1]

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        return logits.masked_fill(indices_to_remove, float("-inf"))
