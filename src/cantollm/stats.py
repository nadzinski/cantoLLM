"""Statistics collection for token generation."""

from dataclasses import dataclass


@dataclass
class SpeculativeStats:
    """Statistics from speculative decoding."""

    draft_tokens_proposed: int
    draft_tokens_accepted: int
    iterations: int

    @property
    def acceptance_rate(self) -> float:
        """Fraction of draft tokens that were accepted."""
        if self.draft_tokens_proposed == 0:
            return 0.0
        return self.draft_tokens_accepted / self.draft_tokens_proposed

    @property
    def tokens_per_iteration(self) -> float:
        """Average tokens yielded per speculative iteration.

        Each iteration yields: accepted_drafts + 1 main token.
        Higher is better (more tokens per forward pass on main model).
        """
        if self.iterations == 0:
            return 0.0
        total_yielded = self.draft_tokens_accepted + self.iterations  # +1 main per iter
        return total_yielded / self.iterations
