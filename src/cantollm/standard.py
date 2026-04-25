from collections.abc import Iterator

import torch

from cantollm.engine.types import SamplingParams, Sequence
from cantollm.kv_cache import KVCache


class StandardBackend:
    """Generates tokens from a model, applying a per-request logits processor
    pipeline before sampling."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
    ):
        self.model = model
        self.device = device

    def reset(self):
        """Reset generator state. No-op for standard generation."""
        pass

    def get_probs(self, logits: torch.Tensor, sampling: SamplingParams) -> torch.Tensor:
        """Run the processor pipeline and return the resulting distribution.

        Args:
            logits: Raw logits from model, shape (batch, vocab) or (vocab,)
            sampling: Per-request sampling parameters

        Returns:
            Probability tensor after all processors have been applied.
        """
        for processor in sampling.processors:
            logits = processor(logits)
        return torch.softmax(logits, dim=-1)

    def sample(
        self, logits: torch.Tensor, sampling: SamplingParams
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a token from logits.

        Args:
            logits: Raw logits from model, shape (batch, vocab) or (vocab,)
            sampling: Per-request sampling parameters

        Returns:
            Tensor containing the sampled token ID(s).
            Tensor containing the probs
        """
        if sampling.greedy:
            # Skip the pipeline: argmax is invariant under monotonic shifts
            # and softmax of the raw logits is cheap.
            probs = torch.softmax(logits, dim=-1)
            return torch.argmax(logits, dim=-1), probs
        probs = self.get_probs(logits, sampling)
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

    def generate(self, sequence: Sequence) -> Iterator[int]:
        """Generate tokens for `sequence`, yielding each as it's produced.

        The backend reads `prompt_token_ids`, `cache`, `sampling_params`,
        `stop_token_ids`, `max_tokens`, and `stop_event` off the sequence;
        `tokens_emitted` is bumped by the engine as tokens are consumed,
        not here. `stop_event` is a cooperative cancel flag checked between
        tokens — the torch forward itself can't be interrupted.
        """
        cache = sequence.cache
        sampling = sequence.sampling_params
        stop_token_ids = sequence.stop_token_ids
        max_tokens = sequence.max_tokens
        stop_event = sequence.stop_event

        if max_tokens <= 0:
            return

        # Process input and get first token
        logits = self.forward(sequence.prompt_token_ids, cache, cache.position)
        token_id, _ = self.sample(logits[:, -1], sampling)
        token_id = token_id.item()

        if token_id in stop_token_ids:
            return
        yield token_id

        # Generate remaining tokens
        for _ in range(max_tokens - 1):
            if stop_event.is_set():
                return
            logits = self.forward([token_id], cache, cache.position)
            token_id, _ = self.sample(logits[:, -1], sampling)
            token_id = token_id.item()

            if token_id in stop_token_ids:
                return
            yield token_id
