"""Shared token sampling: the logits-processor pipeline + greedy/multinomial.

Lives at module level (not on a backend) so both engine paths draw tokens
through the same code: `StandardBackend` delegates here, and the
continuous-batching scheduler — which bypasses `InferenceBackend` entirely —
calls these functions per row on its `(B, vocab)` step logits.

All functions accept `(vocab,)` or `(batch, vocab)` logits.

Greedy runs the full processor pipeline before argmax. Skipping it is only
sound for monotonic transforms (temperature, top-p); processors that move
specific token logits — repetition penalty, logit bias, guided decoding —
change the argmax, so there is deliberately no shortcut here.
"""

import torch

from cantollm.engine.types import SamplingParams


def apply_processors(logits: torch.Tensor, sampling: SamplingParams) -> torch.Tensor:
    """Run the per-request processor pipeline over pre-softmax logits."""
    for processor in sampling.processors:
        logits = processor(logits)
    return logits


def get_probs(logits: torch.Tensor, sampling: SamplingParams) -> torch.Tensor:
    """Processor pipeline + softmax: the distribution a token is drawn from."""
    return torch.softmax(apply_processors(logits, sampling), dim=-1)


def sample(
    logits: torch.Tensor, sampling: SamplingParams
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw a token from raw logits.

    Returns:
        Tensor containing the sampled token ID(s).
        Tensor containing the post-pipeline probs the draw came from.
    """
    probs = get_probs(logits, sampling)
    if sampling.greedy:
        return torch.argmax(probs, dim=-1), probs
    return torch.multinomial(probs, num_samples=1).squeeze(-1), probs
