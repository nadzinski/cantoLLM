from collections.abc import Iterator

import torch

from cantollm.kv_cache import KVCache
from cantollm.stats import SpeculativeStats


class TokenGenerator:
    """Generates tokens from a model using temperature and top-p sampling."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        self.model = model
        self.device = device
        self.temperature = temperature
        self.top_p = top_p
        # Optional cooperative cancel flag. When set, generate() returns between
        # steps. Checked per token; torch forward itself can't be interrupted.
        self.stop_event = None  # type: ignore[assignment]

    def reset(self):
        """Reset generator state. No-op for standard generation."""
        pass

    def reset_stats(self):
        """Reset stats counters. No-op for standard generation."""
        pass

    def get_stats(self) -> SpeculativeStats | None:
        """Get speculative decoding stats. Returns None for standard generation."""
        return None

    def _apply_top_p(self, logits: torch.Tensor) -> torch.Tensor:
        """Zero out tokens outside the top-p probability mass."""
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

    def get_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature/top_p and return probabilities.

        Args:
            logits: Raw logits from model, shape (batch, vocab) or (vocab,)

        Returns:
            Probability tensor after temperature scaling and top-p filtering.
        """
        if self.temperature > 0:
            logits = logits / self.temperature

        if self.top_p < 1.0:
            logits = self._apply_top_p(logits)

        return torch.softmax(logits, dim=-1)

    def sample(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a token from logits.

        Applies temperature scaling and top-p filtering before sampling.

        Args:
            logits: Raw logits from model, shape (batch, vocab) or (vocab,)

        Returns:
            Tensor containing the sampled token ID(s).
            Tensor containing the probs
        """
        if self.temperature == 0:
            # Greedy: skip top_p work entirely, softmax is cheap
            probs = torch.softmax(logits, dim=-1)
            return torch.argmax(logits, dim=-1), probs
        probs = self.get_probs(logits)
        return torch.multinomial(probs, num_samples=1).squeeze(-1), probs

    @torch.inference_mode()
    def forward(
        self,
        token_ids: list[int],
        cache: KVCache,
        start_pos: int,
    ) -> torch.Tensor:
        """Run model forward pass, returning logits.

        Args:
            token_ids: Input token IDs
            cache: KV cache for all transformer layers (modified in place)
            start_pos: Starting position in the sequence

        Returns:
            Logits tensor of shape (batch, seq_len, vocab_size)
        """
        tokens = torch.tensor(token_ids, device=self.device).unsqueeze(0)
        return self.model(tokens, start_pos=start_pos, kv_cache=cache)

    def generate(
        self,
        input_ids: list[int],
        cache: KVCache,
        stop_token_ids: set[int],
        max_tokens: int,
    ) -> Iterator[int]:
        """Generate tokens, yielding each as it's produced.

        Args:
            input_ids: Input token IDs to process
            cache: KV cache for all transformer layers (modified in place)
            stop_token_ids: Token IDs that should stop generation
            max_tokens: Maximum number of new tokens to generate

        Yields:
            Token IDs one at a time
        """
        if max_tokens <= 0:
            return

        stop_event = self.stop_event

        # Process input and get first token
        logits = self.forward(input_ids, cache, cache.position)
        token_id, _ = self.sample(logits[:, -1])
        token_id = token_id.item()

        if token_id in stop_token_ids:
            return
        yield token_id

        # Generate remaining tokens
        for _ in range(max_tokens - 1):
            if stop_event is not None and stop_event.is_set():
                return
            logits = self.forward([token_id], cache, cache.position)
            token_id, _ = self.sample(logits[:, -1])
            token_id = token_id.item()

            if token_id in stop_token_ids:
                return
            yield token_id
