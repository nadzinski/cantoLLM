from collections.abc import Iterator

import torch

from cantollm.engine.types import SamplingParams, Sequence
from cantollm.standard import StandardBackend
from cantollm.kv_cache import KVCache
from cantollm.stats import SpeculativeStats


class SpeculativeBackend:
    """Speculative decoding using draft and main models.

    Owns its own draft KV cache internally.
    """

    def __init__(
        self,
        draft: StandardBackend,
        main: StandardBackend,
        num_layers: int,
        draft_num_layers: int | None = None,
        speculative_tokens: int = 10,
    ):
        self.draft = draft
        self.main = main
        self.draft_cache = KVCache(draft_num_layers or num_layers)
        self.speculative_tokens = speculative_tokens
        self.reset_stats()

    def reset(self):
        """Reset for a new conversation: clear draft cache and stats."""
        self.draft_cache.reset()
        self.reset_stats()

    def reset_stats(self):
        """Reset stats counters for a new generation run."""
        self._draft_proposed = 0
        self._draft_accepted = 0
        self._iterations = 0

    def get_stats(self) -> SpeculativeStats:
        """Get statistics from the last generation run."""
        return SpeculativeStats(
            draft_tokens_proposed=self._draft_proposed,
            draft_tokens_accepted=self._draft_accepted,
            iterations=self._iterations,
        )

    @torch.inference_mode()
    def generate_draft_tokens(
        self,
        input_tokens: list[int],
        num_tokens: int,
        sampling: SamplingParams,
        stop_token_ids: set[int],
    ):
        """Generate num_tokens draft predictions starting from input_tokens.

        Processes input_tokens through draft model to get first prediction,
        then generates remaining tokens autoregressively.

        Args:
            input_tokens: Tokens to process (prompt for first call, continuation tokens after)
            num_tokens: Number of draft tokens to generate
            sampling: Per-request sampling parameters
            stop_token_ids: Stop generation early if any of these are produced

        Returns:
            (tokens, probs) where tokens is tuple of ints, probs is tuple of tensors
        """
        tokens = []
        probs = []
        current_input = input_tokens

        for _ in range(num_tokens):
            logits = self.draft.forward(current_input, self.draft_cache, self.draft_cache.position)
            token_id, token_probs = self.draft.sample(logits[:, -1], sampling)
            token_int = token_id.item()
            tokens.append(token_int)
            probs.append(token_probs.squeeze(0))

            if token_int in stop_token_ids:
                break
            current_input = [token_int]

        return tuple(tokens), tuple(probs)

    def _verify_draft_tokens(
        self,
        draft_tokens: tuple[int, ...],
        draft_probs: tuple[torch.Tensor, ...],
        main_probs: torch.Tensor,
        sampling: SamplingParams,
    ) -> list[int]:
        """Determine which draft tokens to accept.

        Uses the speculative decoding acceptance rule: accept draft token
        with probability min(1, p_main / p_draft) to preserve main model's
        distribution.

        Returns list of accepted tokens (may be empty).
        """
        # Greedy shortcut: when main is deterministic (temp=0), accept draft
        # tokens iff they match main's argmax. The stochastic rule below still
        # gives correct behavior, but wastes draft tokens when p_draft < 1 at
        # positions where the draft happened to agree with main's argmax.
        if sampling.greedy:
            accepted = []
            main_argmax = torch.argmax(main_probs, dim=-1)
            for i, token in enumerate(draft_tokens):
                if main_argmax[i].item() != token:
                    break
                accepted.append(token)
            return accepted

        accepted = []
        for i, token in enumerate(draft_tokens):
            p_draft = draft_probs[i][token].item()
            p_main = main_probs[i][token].item()

            # Magic(!): To preserve the distribution of main, accept draft tokens
            # with probability min(1, p_main / p_draft). The p_draft > 0 guard
            # handles potential bfloat16 underflow.
            if p_draft > 0.0:
                accept_prob = p_main / p_draft
            else:
                accept_prob = 1.0 if p_main > 0.0 else 0.0

            if accept_prob < 1.0 and torch.rand(1).item() > accept_prob:
                break
            accepted.append(token)
        return accepted

    @torch.inference_mode()
    def generate(self, sequence: Sequence) -> Iterator[int]:
        """
        Here is the speculative generation algorithm:

        0) enter loop
        1) generate a run of speculative tokens using the draft model
           (first iteration also prefills draft kv cache)
        2) run main model on draft tokens in one inference operation to verify
           (first iteration also prefills main kv cache)
        3) for each draft token, compare main's probability vs draft's probability
        4) accept draft token with probability min(1, p_main / p_draft), otherwise reject
        5) the first time there is a reject, we stop, and sample from main at that point
           as that token will be a valid prediction for main.
           NB: sampling from main here ensures that if there are N accepts, we can yield
           N+1 tokens (so 0 accepts, we still have a token)
           If there are no rejects, we sample from main at the last token.
        6) yield those N+1 tokens up (unless we hit a stop token, then return)
        7) truncate both kv caches to reflect what we accepted and what we threw away
        8) continue round loop

        NB: Currently single-sequence only (batch size 1). Batched speculative
        decoding would require per-sequence position tracking, padded KV caches,
        and attention masking to handle divergent acceptance rates across sequences.
        """
        input_ids = sequence.prompt_token_ids
        cache = sequence.cache
        sampling = sequence.sampling_params
        stop_token_ids = sequence.stop_token_ids
        max_tokens = sequence.max_tokens
        stop_event = sequence.stop_event

        draft_input = input_ids
        main_prefix = input_ids
        tokens_yielded = 0

        while tokens_yielded < max_tokens:
            if stop_event.is_set():
                return
            draft_tokens, draft_probs = self.generate_draft_tokens(
                draft_input, self.speculative_tokens, sampling, stop_token_ids
            )
            self._draft_proposed += len(draft_tokens)
            self._iterations += 1

            # Verify drafts against main model
            # main_prefix is input_ids on first iteration (prefill + verify),
            # then [main_tail_token] on subsequent iterations (just verify)
            main_input = [*main_prefix, *draft_tokens]
            main_logits = self.main.forward(main_input, cache, cache.position)[0]
            # Slice to verification positions: position i verifies draft[i]
            verify_logits = main_logits[len(main_prefix) - 1:]
            main_probs = self.main.get_probs(verify_logits, sampling)

            accepted_tokens = self._verify_draft_tokens(
                draft_tokens, draft_probs, main_probs, sampling
            )
            self._draft_accepted += len(accepted_tokens)

            # Sample next token from main at first rejection point (or after all accepted)
            main_tail_token, _ = self.main.sample(verify_logits[len(accepted_tokens)], sampling)
            main_tail_token = main_tail_token.item()

            # Yield tokens, respecting max_tokens budget
            all_tokens = (*accepted_tokens, main_tail_token)
            remaining = max_tokens - tokens_yielded
            emit_tokens = all_tokens[:remaining]

            hit_stop = False
            n_emitted = 0
            for tok in emit_tokens:
                if tok in stop_token_ids:
                    hit_stop = True
                    break
                yield tok
                n_emitted += 1
            tokens_yielded += n_emitted

            # Always truncate caches: remove rejected draft tokens' KV entries
            # and any entries beyond what we actually emitted (including stop token).
            # cache.position is post-forward (includes all of main_prefix + draft_tokens),
            # so we subtract the rejected/unemitted drafts to get the correct position.
            n_accepted_emitted = min(len(accepted_tokens), n_emitted)
            new_pos = cache.position - len(draft_tokens) + n_accepted_emitted
            cache.truncate(new_pos)
            self.draft_cache.truncate(new_pos)

            if hit_stop or tokens_yielded >= max_tokens:
                return

            # Set up for next iteration
            last_token = emit_tokens[-1]
            main_prefix = [last_token]  # Only this token pending for main

            # NB: draft's KV at position N is valid even if the prediction from that
            # position was wrong. The KV represents the model's internal state after seeing the
            # correct token at N; the incorrect prediction was just a bad output logit. So we
            # can keep the valid KV and just provide the correct *next* token to continue.
            if len(accepted_tokens) == len(draft_tokens):
                # All tokens accepted - draft is structurally 1 behind;
                # need to add last accepted token
                draft_input = [accepted_tokens[-1], last_token]
            else:
                draft_input = [last_token]
