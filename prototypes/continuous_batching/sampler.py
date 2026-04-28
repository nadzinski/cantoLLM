"""Greedy sampler. Deterministic — argmax over vocab dim."""

import torch


def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    """Pick the highest-probability token from each row of logits.

    Args:
        logits: (batch, vocab_size) float tensor.

    Returns:
        (batch,) int64 tensor of token ids.
    """
    return logits.argmax(dim=-1)
