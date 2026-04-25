import threading
from unittest.mock import MagicMock, patch

import torch
import pytest

from cantollm.engine.types import SamplingParams, Sequence
from cantollm.speculative import SpeculativeBackend
from cantollm.kv_cache import KVCache


GREEDY = SamplingParams.from_temperature_top_p(temperature=0.0, top_p=1.0)
STOCHASTIC = SamplingParams.from_temperature_top_p(temperature=0.7, top_p=0.9)


def make_sequence(
    prompt_token_ids,
    cache,
    sampling=GREEDY,
    stop_token_ids=None,
    max_tokens=10,
):
    return Sequence(
        request_id="test",
        prompt_token_ids=list(prompt_token_ids),
        sampling_params=sampling,
        stop_token_ids=set(stop_token_ids) if stop_token_ids is not None else {999},
        max_tokens=max_tokens,
        cache=cache,
        stop_event=threading.Event(),
    )


def make_mock_generator():
    """Create a mock StandardBackend."""
    mock = MagicMock()
    mock._apply_top_p = MagicMock(side_effect=lambda x, top_p: x)
    mock.get_probs = MagicMock(side_effect=lambda x, sampling: torch.softmax(x, dim=-1))
    return mock


def make_cache_growing_forward(vocab_size=100):
    """Create a forward function that grows cache properly."""

    def mock_forward(tokens, cache, pos):
        for layer in cache:
            new_len = pos + len(tokens)
            layer["keys"] = torch.zeros(1, new_len, 4, 8)
            layer["values"] = torch.zeros(1, new_len, 4, 8)
        return torch.randn(1, len(tokens), vocab_size)

    return mock_forward


class TestSpeculativeBackendInit:
    def test_init_stores_draft_and_main(self):
        """Test that constructor stores draft and main generators."""
        draft = make_mock_generator()
        main = make_mock_generator()

        spec_gen = SpeculativeBackend(draft=draft, main=main, num_layers=2)

        assert spec_gen.draft is draft
        assert spec_gen.main is main

    def test_init_creates_draft_cache(self):
        """Test that constructor creates internal draft cache."""
        draft = make_mock_generator()
        main = make_mock_generator()

        spec_gen = SpeculativeBackend(draft=draft, main=main, num_layers=4)

        assert spec_gen.draft_cache is not None
        assert len(spec_gen.draft_cache) == 4

    def test_init_stores_speculative_tokens(self):
        """Test that speculative_tokens parameter is stored."""
        draft = make_mock_generator()
        main = make_mock_generator()

        spec_gen = SpeculativeBackend(draft=draft, main=main, num_layers=2, speculative_tokens=8)

        assert spec_gen.speculative_tokens == 8


class TestGenerateDraftTokens:
    def test_generates_correct_number_of_tokens(self):
        """Test that we generate the requested number of speculative tokens."""
        draft = make_mock_generator()
        main = make_mock_generator()
        spec_gen = SpeculativeBackend(draft=draft, main=main, num_layers=1, speculative_tokens=4)

        vocab_size = 100
        probs = torch.softmax(torch.randn(1, vocab_size), dim=-1)

        draft.forward = make_cache_growing_forward(vocab_size)
        draft.sample = MagicMock(return_value=(torch.tensor([42]), probs))

        tokens, probs = spec_gen.generate_draft_tokens(
            input_tokens=[1],
            num_tokens=4,
            sampling=STOCHASTIC,
            stop_token_ids={999},
        )

        assert len(tokens) == 4
        assert len(probs) == 4

    def test_stops_early_on_stop_token(self):
        """Test that generation stops when a stop token is produced."""
        draft = make_mock_generator()
        main = make_mock_generator()
        spec_gen = SpeculativeBackend(draft=draft, main=main, num_layers=1)

        vocab_size = 100
        stop_token = 50
        probs = torch.softmax(torch.randn(1, vocab_size), dim=-1)

        draft.forward = make_cache_growing_forward(vocab_size)

        call_count = [0]

        def sample_side_effect(*args):
            call_count[0] += 1
            if call_count[0] == 1:
                return (torch.tensor(42), probs)  # non-stop token
            else:
                return (torch.tensor(stop_token), probs)  # stop token

        draft.sample = MagicMock(side_effect=sample_side_effect)

        tokens, probs = spec_gen.generate_draft_tokens(
            input_tokens=[1],
            num_tokens=4,
            sampling=STOCHASTIC,
            stop_token_ids={stop_token},
        )

        assert len(tokens) == 2  # stopped early
        assert tokens[-1] == stop_token


class TestSpeculativeGenerate:
    def test_returns_early_if_draft_produces_stop_token(self):
        """Test that generate returns immediately if draft's first token is stop token."""
        draft = make_mock_generator()
        main = make_mock_generator()
        spec_gen = SpeculativeBackend(draft=draft, main=main, num_layers=1)

        vocab_size = 100
        stop_token = 50  # within vocab_size

        # Draft produces stop token immediately
        draft_probs = torch.zeros(1, vocab_size)
        draft_probs[0, stop_token] = 1.0
        draft.forward = make_cache_growing_forward(vocab_size)
        draft.sample = MagicMock(return_value=(torch.tensor([stop_token]), draft_probs))

        # Main verifies - high prob for stop token means it gets accepted
        def mock_main_forward(tokens, cache, pos):
            for layer in cache:
                new_len = pos + len(tokens)
                layer["keys"] = torch.zeros(1, new_len, 4, 8)
                layer["values"] = torch.zeros(1, new_len, 4, 8)
            logits = torch.full((1, len(tokens), vocab_size), -100.0)
            logits[0, :, stop_token] = 10.0  # high prob for stop token
            return logits

        main.forward = mock_main_forward
        main.get_probs = MagicMock(side_effect=lambda x, sampling: torch.softmax(x, dim=-1))
        main.sample = MagicMock(
            return_value=(torch.tensor([stop_token]), torch.softmax(torch.zeros(vocab_size), dim=-1))
        )

        cache = KVCache(1)

        result = list(
            spec_gen.generate(
                make_sequence([1, 2, 3], cache, stop_token_ids={stop_token}, max_tokens=10)
            )
        )

        # Stop token is accepted but not yielded (we return before yielding it)
        assert result == []

    def test_yields_tokens_from_accepted_drafts(self):
        """Test that accepted draft tokens are yielded."""
        draft = make_mock_generator()
        main = make_mock_generator()
        spec_gen = SpeculativeBackend(draft=draft, main=main, num_layers=1, speculative_tokens=1)

        vocab_size = 100
        draft_token = 10

        draft_probs = torch.zeros(1, vocab_size)
        draft_probs[0, draft_token] = 0.5

        draft.forward = make_cache_growing_forward(vocab_size)
        draft.sample = MagicMock(return_value=(torch.tensor([draft_token]), draft_probs))

        def mock_main_forward(tokens, cache, pos):
            for layer in cache:
                new_len = pos + len(tokens)
                layer["keys"] = torch.zeros(1, new_len, 4, 8)
                layer["values"] = torch.zeros(1, new_len, 4, 8)
            logits = torch.full((1, len(tokens), vocab_size), -100.0)
            logits[0, :, draft_token] = 10.0  # high logit for draft token -> accept
            return logits

        main.forward = mock_main_forward
        main.get_probs = MagicMock(side_effect=lambda x, sampling: torch.softmax(x, dim=-1))
        main.sample = MagicMock(
            return_value=(torch.tensor([draft_token]), torch.softmax(torch.zeros(vocab_size), dim=-1))
        )

        cache = KVCache(1)

        result = list(
            spec_gen.generate(make_sequence([1, 2, 3], cache, max_tokens=2))
        )

        # Should yield accepted draft token + next sampled from main
        assert len(result) == 2
        assert draft_token in result


class TestFirstTokenYielded:
    def test_first_token_from_draft_verification_is_yielded(self):
        """Test that the first token comes from draft verified against main."""
        draft = make_mock_generator()
        main = make_mock_generator()
        spec_gen = SpeculativeBackend(draft=draft, main=main, num_layers=1, speculative_tokens=1)

        vocab_size = 100
        draft_token = 10

        draft_probs = torch.zeros(1, vocab_size)
        draft_probs[0, draft_token] = 0.5

        draft.forward = make_cache_growing_forward(vocab_size)
        draft.sample = MagicMock(return_value=(torch.tensor(draft_token), draft_probs))

        def mock_main_forward(tokens, cache, pos):
            for layer in cache:
                new_len = pos + len(tokens)
                layer["keys"] = torch.zeros(1, new_len, 4, 8)
                layer["values"] = torch.zeros(1, new_len, 4, 8)
            logits = torch.full((1, len(tokens), vocab_size), -100.0)
            logits[0, :, draft_token] = 10.0  # high prob -> draft accepted
            return logits

        main.forward = mock_main_forward
        main.get_probs = MagicMock(side_effect=lambda x, sampling: torch.softmax(x, dim=-1))
        main.sample = MagicMock(
            return_value=(torch.tensor([draft_token]), torch.softmax(torch.zeros(vocab_size), dim=-1))
        )

        cache = KVCache(1)

        result = list(
            spec_gen.generate(make_sequence([1, 2, 3], cache, max_tokens=2))
        )

        # First token should be the accepted draft token
        assert result[0] == draft_token, f"First token should be draft token, got {result}"


class TestSpeculativeStats:
    def test_stats_reset_on_init(self):
        """Test that stats are zeroed on init."""
        draft = make_mock_generator()
        main = make_mock_generator()
        spec_gen = SpeculativeBackend(draft=draft, main=main, num_layers=1)

        stats = spec_gen.get_stats()
        assert stats.draft_tokens_proposed == 0
        assert stats.draft_tokens_accepted == 0
        assert stats.iterations == 0

    def test_stats_tracked_during_generation(self):
        """Test that stats are accumulated during generate()."""
        draft = make_mock_generator()
        main = make_mock_generator()
        spec_gen = SpeculativeBackend(draft=draft, main=main, num_layers=1, speculative_tokens=2)

        vocab_size = 100
        draft_token = 10

        draft_probs = torch.zeros(1, vocab_size)
        draft_probs[0, draft_token] = 0.5

        draft.forward = make_cache_growing_forward(vocab_size)
        draft.sample = MagicMock(return_value=(torch.tensor([draft_token]), draft_probs))

        def mock_main_forward(tokens, cache, pos):
            for layer in cache:
                new_len = pos + len(tokens)
                layer["keys"] = torch.zeros(1, new_len, 4, 8)
                layer["values"] = torch.zeros(1, new_len, 4, 8)
            logits = torch.full((1, len(tokens), vocab_size), -100.0)
            logits[0, :, draft_token] = 10.0  # high prob -> accept
            return logits

        main.forward = mock_main_forward
        main.sample = MagicMock(return_value=(torch.tensor([5]), torch.zeros(1, vocab_size)))
        main.get_probs = MagicMock(side_effect=lambda x, sampling: torch.softmax(x, dim=-1))

        cache = KVCache(1)

        list(
            spec_gen.generate(make_sequence([1, 2, 3], cache, max_tokens=3))
        )

        stats = spec_gen.get_stats()
        # With new flow: 1 iteration yields 2 accepted + 1 next_main = 3 tokens
        assert stats.iterations == 1
        assert stats.draft_tokens_proposed == 2
        assert stats.acceptance_rate >= 0

    def test_stats_reset(self):
        """Test that reset_stats() clears counters."""
        draft = make_mock_generator()
        main = make_mock_generator()
        spec_gen = SpeculativeBackend(draft=draft, main=main, num_layers=1)

        spec_gen._draft_proposed = 10
        spec_gen._draft_accepted = 5
        spec_gen._iterations = 2

        spec_gen.reset_stats()

        stats = spec_gen.get_stats()
        assert stats.draft_tokens_proposed == 0
        assert stats.draft_tokens_accepted == 0
        assert stats.iterations == 0


class TestVerifyDraftTokens:
    def test_always_accepts_when_main_prob_higher(self):
        """When p_main >= p_draft, token should always be accepted."""
        draft = make_mock_generator()
        main = make_mock_generator()
        spec_gen = SpeculativeBackend(draft=draft, main=main, num_layers=1)

        vocab_size = 100
        token = 42

        draft_probs = torch.zeros(vocab_size)
        draft_probs[token] = 0.3

        main_probs = torch.zeros(vocab_size)
        main_probs[token] = 0.8

        # accept_prob = 0.8/0.3 ≈ 2.67 >= 1.0, so torch.rand is never consulted
        accepted = spec_gen._verify_draft_tokens(
            draft_tokens=(token,),
            draft_probs=(draft_probs,),
            main_probs=main_probs.unsqueeze(0),
            sampling=STOCHASTIC,
        )

        assert token in accepted

    def test_probabilistic_rejection_when_main_prob_lower(self):
        """When p_main < p_draft, rejection depends on torch.rand."""
        draft = make_mock_generator()
        main = make_mock_generator()
        spec_gen = SpeculativeBackend(draft=draft, main=main, num_layers=1)

        vocab_size = 100
        token = 42

        draft_probs = torch.zeros(vocab_size)
        draft_probs[token] = 0.8

        main_probs = torch.zeros(vocab_size)
        main_probs[token] = 0.2

        # accept_prob = 0.2/0.8 = 0.25; torch.rand returns 0.9 > 0.25 → reject
        with patch("torch.rand", return_value=torch.tensor([0.9])):
            accepted = spec_gen._verify_draft_tokens(
                draft_tokens=(token,),
                draft_probs=(draft_probs,),
                main_probs=main_probs.unsqueeze(0),
                sampling=STOCHASTIC,
            )

        assert token not in accepted

        # accept_prob = 0.25; torch.rand returns 0.1 < 0.25 → accept
        with patch("torch.rand", return_value=torch.tensor([0.1])):
            accepted = spec_gen._verify_draft_tokens(
                draft_tokens=(token,),
                draft_probs=(draft_probs,),
                main_probs=main_probs.unsqueeze(0),
                sampling=STOCHASTIC,
            )

        assert token in accepted

    def test_accepts_when_p_draft_zero_and_p_main_positive(self):
        """When p_draft is 0 (bfloat16 underflow) but p_main > 0, accept."""
        draft = make_mock_generator()
        main = make_mock_generator()
        spec_gen = SpeculativeBackend(draft=draft, main=main, num_layers=1)

        vocab_size = 100
        token = 42

        draft_probs = torch.zeros(vocab_size)
        draft_probs[token] = 0.0

        main_probs = torch.zeros(vocab_size)
        main_probs[token] = 0.5

        accepted = spec_gen._verify_draft_tokens(
            draft_tokens=(token,),
            draft_probs=(draft_probs,),
            main_probs=main_probs.unsqueeze(0),
            sampling=STOCHASTIC,
        )

        assert token in accepted

    def test_rejects_when_both_probs_zero(self):
        """When both p_draft and p_main are 0, reject."""
        draft = make_mock_generator()
        main = make_mock_generator()
        spec_gen = SpeculativeBackend(draft=draft, main=main, num_layers=1)

        vocab_size = 100
        token = 42

        draft_probs = torch.zeros(vocab_size)
        main_probs = torch.zeros(vocab_size)

        accepted = spec_gen._verify_draft_tokens(
            draft_tokens=(token,),
            draft_probs=(draft_probs,),
            main_probs=main_probs.unsqueeze(0),
            sampling=STOCHASTIC,
        )

        assert token not in accepted


class TestCacheTruncation:
    """Tests for correct cache truncation after verification."""

    def _make_spec_gen_with_controlled_rejection(self, vocab_size=100, speculative_tokens=3):
        """Set up a SpeculativeBackend where we can control which drafts are rejected.

        Draft always produces token 10. Main's logits control acceptance:
        high logit for token 10 → accept, low logit → reject (via probability ratio).
        """
        draft = make_mock_generator()
        main = make_mock_generator()
        spec_gen = SpeculativeBackend(
            draft=draft, main=main, num_layers=1, speculative_tokens=speculative_tokens,
        )

        draft_token = 10
        draft_probs = torch.zeros(1, vocab_size)
        draft_probs[0, draft_token] = 0.5

        draft.forward = make_cache_growing_forward(vocab_size)
        draft.sample = MagicMock(return_value=(torch.tensor([draft_token]), draft_probs))

        main_token = 20
        main.sample = MagicMock(
            return_value=(torch.tensor([main_token]), torch.softmax(torch.zeros(vocab_size), dim=-1))
        )
        main.get_probs = MagicMock(side_effect=lambda x, sampling: torch.softmax(x, dim=-1))

        return spec_gen, draft, main, draft_token, main_token

    def test_cache_position_correct_after_full_accept(self):
        """When all drafts are accepted, cache keeps all entries."""
        vocab_size = 100
        spec_gen, draft, main, draft_token, main_token = (
            self._make_spec_gen_with_controlled_rejection(vocab_size, speculative_tokens=3)
        )

        def mock_main_forward(tokens, cache, pos):
            for layer in cache:
                new_len = pos + len(tokens)
                layer["keys"] = torch.zeros(1, new_len, 4, 8)
                layer["values"] = torch.zeros(1, new_len, 4, 8)
            logits = torch.full((1, len(tokens), vocab_size), -100.0)
            logits[0, :, draft_token] = 10.0  # high prob → accept all
            return logits

        main.forward = mock_main_forward

        cache = KVCache(1)
        input_ids = [1, 2, 3]

        result = list(spec_gen.generate(make_sequence(input_ids, cache, max_tokens=4)))

        assert len(result) == 4
        # Cache should contain: 3 input + 3 draft + 0 rejected = 6 entries
        # (main_tail_token hasn't been fed through the model yet)
        assert cache.position == len(input_ids) + 3  # 6

    def test_cache_position_correct_after_partial_accept(self):
        """When some drafts are rejected, cache is truncated to remove them."""
        vocab_size = 100
        spec_gen, draft, main, draft_token, main_token = (
            self._make_spec_gen_with_controlled_rejection(vocab_size, speculative_tokens=3)
        )

        # Force rejection of all drafts by making torch.rand always return 1.0
        # (accept_prob = p_main/p_draft, and rand > accept_prob → reject)
        def mock_main_forward(tokens, cache, pos):
            for layer in cache:
                new_len = pos + len(tokens)
                layer["keys"] = torch.zeros(1, new_len, 4, 8)
                layer["values"] = torch.zeros(1, new_len, 4, 8)
            logits = torch.full((1, len(tokens), vocab_size), 0.0)  # uniform → low accept_prob
            return logits

        main.forward = mock_main_forward

        cache = KVCache(1)
        input_ids = [1, 2, 3]

        # Force all rejections: torch.rand returns 0.999 which is > any accept_prob
        with patch("torch.rand", return_value=torch.tensor([0.999])):
            result = list(spec_gen.generate(
                make_sequence(input_ids, cache, sampling=STOCHASTIC, max_tokens=1)
            ))

        assert len(result) == 1  # only main_tail_token
        # Cache should contain: 3 input + 0 accepted = 3 entries
        assert cache.position == len(input_ids)

    def test_cache_position_correct_across_multiple_iterations(self):
        """Cache position stays correct across multiple speculative iterations."""
        vocab_size = 100
        spec_gen, draft, main, draft_token, main_token = (
            self._make_spec_gen_with_controlled_rejection(vocab_size, speculative_tokens=2)
        )

        call_count = [0]

        def mock_main_forward(tokens, cache, pos):
            call_count[0] += 1
            for layer in cache:
                new_len = pos + len(tokens)
                layer["keys"] = torch.zeros(1, new_len, 4, 8)
                layer["values"] = torch.zeros(1, new_len, 4, 8)
            logits = torch.full((1, len(tokens), vocab_size), -100.0)
            logits[0, :, draft_token] = 10.0  # accept all
            return logits

        main.forward = mock_main_forward

        cache = KVCache(1)
        input_ids = [1, 2, 3]

        # max_tokens=6 → should take 2 iterations of (2 accepted + 1 main = 3 tokens)
        result = list(spec_gen.generate(make_sequence(input_ids, cache, max_tokens=6)))

        assert len(result) == 6
        # After 2 iterations: 3 input + 2 accepted + 2 accepted = 7
        # (last main_tail_token from each iter gets fed as main_prefix next iter)
        # Iter 1: input[3] + draft[2] → cache at 5, keep all 5, yield 3
        # Iter 2: main_prefix[1] + draft[2] → cache at 8, keep all 8, yield 3
        # But the last main_tail hasn't been processed, so cache = 8
        assert cache.position == len(input_ids) + 2 + 1 + 2  # 8

    def test_draft_cache_stays_in_sync(self):
        """Draft cache and main cache stay at compatible positions."""
        vocab_size = 100
        spec_gen, draft, main, draft_token, main_token = (
            self._make_spec_gen_with_controlled_rejection(vocab_size, speculative_tokens=3)
        )

        def mock_main_forward(tokens, cache, pos):
            for layer in cache:
                new_len = pos + len(tokens)
                layer["keys"] = torch.zeros(1, new_len, 4, 8)
                layer["values"] = torch.zeros(1, new_len, 4, 8)
            logits = torch.full((1, len(tokens), vocab_size), 0.0)  # uniform
            return logits

        main.forward = mock_main_forward

        cache = KVCache(1)
        input_ids = [1, 2, 3]

        # Reject all drafts
        with patch("torch.rand", return_value=torch.tensor([0.999])):
            list(spec_gen.generate(
                make_sequence(input_ids, cache, sampling=STOCHASTIC, max_tokens=2)
            ))

        # Both caches should be truncated to the same base
        # (draft may lag by 1 since it doesn't process main_tail, but truncate
        # should bring them to the same position)
        assert spec_gen.draft_cache.position <= cache.position


class TestReset:
    def test_reset_clears_draft_cache(self):
        """reset() should zero out draft cache position."""
        draft = make_mock_generator()
        main = make_mock_generator()
        spec_gen = SpeculativeBackend(draft=draft, main=main, num_layers=2)

        # Simulate some cache state
        for layer in spec_gen.draft_cache:
            layer["keys"] = torch.zeros(1, 10, 4, 8)
            layer["values"] = torch.zeros(1, 10, 4, 8)
        assert spec_gen.draft_cache.position == 10

        spec_gen.reset()

        assert spec_gen.draft_cache.position == 0

    def test_reset_clears_stats(self):
        """reset() should also zero stats."""
        draft = make_mock_generator()
        main = make_mock_generator()
        spec_gen = SpeculativeBackend(draft=draft, main=main, num_layers=1)

        spec_gen._draft_proposed = 10
        spec_gen._draft_accepted = 5
        spec_gen._iterations = 2

        spec_gen.reset()

        stats = spec_gen.get_stats()
        assert stats.draft_tokens_proposed == 0
        assert stats.draft_tokens_accepted == 0
        assert stats.iterations == 0


class TestStopTokenCacheTruncation:
    """Tests that caches are properly truncated when a stop token ends generation."""

    def test_cache_truncated_on_stop_token(self):
        """After generate() returns due to stop token, caches have no phantom entries."""
        vocab_size = 100
        stop_token = 50
        draft_token = 10

        spec_tokens = 3
        draft = make_mock_generator()
        main = make_mock_generator()
        spec_gen = SpeculativeBackend(
            draft=draft, main=main, num_layers=1, speculative_tokens=spec_tokens,
        )

        # Draft produces: draft_token, draft_token, stop_token
        draft_call_count = [0]
        draft_probs = torch.zeros(1, vocab_size)
        draft_probs[0, draft_token] = 0.5

        stop_probs = torch.zeros(1, vocab_size)
        stop_probs[0, stop_token] = 0.5

        def draft_sample_side_effect(*args):
            draft_call_count[0] += 1
            if draft_call_count[0] % 3 == 0:
                return (torch.tensor([stop_token]), stop_probs)
            return (torch.tensor([draft_token]), draft_probs)

        draft.forward = make_cache_growing_forward(vocab_size)
        draft.sample = MagicMock(side_effect=draft_sample_side_effect)

        # Main accepts everything (high logit for both tokens)
        def mock_main_forward(tokens, cache, pos):
            for layer in cache:
                new_len = pos + len(tokens)
                layer["keys"] = torch.zeros(1, new_len, 4, 8)
                layer["values"] = torch.zeros(1, new_len, 4, 8)
            logits = torch.full((1, len(tokens), vocab_size), -100.0)
            # Verify rows start at (len(main_prefix) - 1); main_prefix is
            # whatever precedes the drafts in `tokens`. Rows after that
            # correspond to: draft_1, draft_2, drafted stop, main_tail.
            verify_start = len(tokens) - spec_tokens - 1
            logits[0, verify_start, draft_token] = 10.0
            logits[0, verify_start + 1, draft_token] = 10.0
            logits[0, verify_start + 2, stop_token] = 10.0
            logits[0, verify_start + 3, draft_token] = 10.0
            return logits

        main.forward = mock_main_forward
        main.get_probs = MagicMock(side_effect=lambda x, sampling: torch.softmax(x, dim=-1))
        main.sample = MagicMock(
            return_value=(torch.tensor([draft_token]), torch.softmax(torch.zeros(vocab_size), dim=-1))
        )

        cache = KVCache(1)
        input_ids = [1, 2, 3]

        result = list(spec_gen.generate(
            make_sequence(input_ids, cache, stop_token_ids={stop_token}, max_tokens=100)
        ))

        # Should yield draft_token, draft_token (then stop_token halts without yielding)
        assert stop_token not in result
        assert len(result) == 2

        # Cache should have: 3 input + 2 accepted tokens = 5 entries
        # NOT 3 input + 3 draft tokens = 6 (which would include stop_token's KV)
        expected_pos = len(input_ids) + len(result)
        assert cache.position == expected_pos, (
            f"Main cache has {cache.position} entries, expected {expected_pos} "
            f"(stop token KV should be truncated)"
        )

    def test_caches_in_sync_after_stop_token(self):
        """Both main and draft caches are at compatible positions after stop token."""
        vocab_size = 100
        stop_token = 50
        draft_token = 10

        draft = make_mock_generator()
        main = make_mock_generator()
        spec_gen = SpeculativeBackend(
            draft=draft, main=main, num_layers=1, speculative_tokens=3,
        )

        # Draft produces stop token as first draft token
        draft_probs = torch.zeros(1, vocab_size)
        draft_probs[0, stop_token] = 0.5
        draft.forward = make_cache_growing_forward(vocab_size)
        draft.sample = MagicMock(return_value=(torch.tensor([stop_token]), draft_probs))

        def mock_main_forward(tokens, cache, pos):
            for layer in cache:
                new_len = pos + len(tokens)
                layer["keys"] = torch.zeros(1, new_len, 4, 8)
                layer["values"] = torch.zeros(1, new_len, 4, 8)
            logits = torch.full((1, len(tokens), vocab_size), -100.0)
            logits[0, :, stop_token] = 10.0
            return logits

        main.forward = mock_main_forward
        main.get_probs = MagicMock(side_effect=lambda x, sampling: torch.softmax(x, dim=-1))
        main.sample = MagicMock(
            return_value=(torch.tensor([stop_token]), torch.softmax(torch.zeros(vocab_size), dim=-1))
        )

        cache = KVCache(1)
        input_ids = [1, 2, 3]

        list(spec_gen.generate(
            make_sequence(input_ids, cache, stop_token_ids={stop_token}, max_tokens=100)
        ))

        # After stop token on first draft, no tokens yielded.
        # Main cache had: input_ids[3] + stop_token[1] = 4 entries from forward.
        # Truncation should remove stop_token's KV: position = 3 (just input_ids).
        assert cache.position == len(input_ids)
        # Draft cache should also be truncated to match
        assert spec_gen.draft_cache.position == cache.position

    def test_multi_turn_caches_stay_in_sync(self):
        """Caches remain in sync across two sequential generate() calls (multi-turn)."""
        vocab_size = 100
        stop_token = 50
        draft_token = 10

        spec_tokens = 2
        draft = make_mock_generator()
        main = make_mock_generator()
        spec_gen = SpeculativeBackend(
            draft=draft, main=main, num_layers=1, speculative_tokens=spec_tokens,
        )

        # Draft produces: draft_token, then stop_token
        draft_call_count = [0]
        draft_probs = torch.zeros(1, vocab_size)
        draft_probs[0, draft_token] = 0.5
        stop_probs = torch.zeros(1, vocab_size)
        stop_probs[0, stop_token] = 0.5

        def draft_sample_side_effect(*args):
            draft_call_count[0] += 1
            if draft_call_count[0] % 2 == 0:
                return (torch.tensor([stop_token]), stop_probs)
            return (torch.tensor([draft_token]), draft_probs)

        draft.forward = make_cache_growing_forward(vocab_size)
        draft.sample = MagicMock(side_effect=draft_sample_side_effect)

        def mock_main_forward(tokens, cache, pos):
            for layer in cache:
                new_len = pos + len(tokens)
                layer["keys"] = torch.zeros(1, new_len, 4, 8)
                layer["values"] = torch.zeros(1, new_len, 4, 8)
            logits = torch.full((1, len(tokens), vocab_size), -100.0)
            # Verify rows start at (len(main_prefix) - 1). Accept draft_token,
            # then stop_token; the trailing row is the main_tail sample.
            verify_start = len(tokens) - spec_tokens - 1
            logits[0, verify_start, draft_token] = 10.0
            logits[0, verify_start + 1, stop_token] = 10.0
            logits[0, verify_start + 2, draft_token] = 10.0
            return logits

        main.forward = mock_main_forward
        main.get_probs = MagicMock(side_effect=lambda x, sampling: torch.softmax(x, dim=-1))
        main.sample = MagicMock(
            return_value=(torch.tensor([draft_token]), torch.softmax(torch.zeros(vocab_size), dim=-1))
        )

        cache = KVCache(1)

        # Turn 1
        input_ids_1 = [1, 2, 3]
        result_1 = list(spec_gen.generate(
            make_sequence(input_ids_1, cache, stop_token_ids={stop_token}, max_tokens=100)
        ))

        pos_after_turn_1 = cache.position
        draft_pos_after_turn_1 = spec_gen.draft_cache.position

        # Both caches should be in sync after turn 1
        assert draft_pos_after_turn_1 == pos_after_turn_1, (
            f"Cache desync after turn 1: main={pos_after_turn_1}, draft={draft_pos_after_turn_1}"
        )

        # Turn 2: new input appended to existing cache
        spec_gen.reset_stats()
        input_ids_2 = [4, 5, 6]
        result_2 = list(spec_gen.generate(
            make_sequence(input_ids_2, cache, stop_token_ids={stop_token}, max_tokens=100)
        ))

        pos_after_turn_2 = cache.position
        draft_pos_after_turn_2 = spec_gen.draft_cache.position

        # Both caches should still be in sync after turn 2
        assert draft_pos_after_turn_2 == pos_after_turn_2, (
            f"Cache desync after turn 2: main={pos_after_turn_2}, draft={draft_pos_after_turn_2}"
        )


class TestMaxTokensCap:
    def test_max_tokens_not_exceeded(self):
        """generate() should never yield more than max_tokens."""
        draft = make_mock_generator()
        main = make_mock_generator()
        spec_gen = SpeculativeBackend(
            draft=draft, main=main, num_layers=1, speculative_tokens=10,
        )

        vocab_size = 100
        draft_token = 10

        draft_probs = torch.zeros(1, vocab_size)
        draft_probs[0, draft_token] = 0.5

        draft.forward = make_cache_growing_forward(vocab_size)
        draft.sample = MagicMock(return_value=(torch.tensor([draft_token]), draft_probs))

        def mock_main_forward(tokens, cache, pos):
            for layer in cache:
                new_len = pos + len(tokens)
                layer["keys"] = torch.zeros(1, new_len, 4, 8)
                layer["values"] = torch.zeros(1, new_len, 4, 8)
            logits = torch.full((1, len(tokens), vocab_size), -100.0)
            logits[0, :, draft_token] = 10.0  # accept all
            return logits

        main.forward = mock_main_forward
        main.get_probs = MagicMock(side_effect=lambda x, sampling: torch.softmax(x, dim=-1))
        main.sample = MagicMock(
            return_value=(torch.tensor([draft_token]), torch.softmax(torch.zeros(vocab_size), dim=-1))
        )

        cache = KVCache(1)

        # speculative_tokens=10, so one iteration wants to yield 11 tokens,
        # but max_tokens=3 should cap it
        result = list(
            spec_gen.generate(make_sequence([1, 2, 3], cache, max_tokens=3))
        )

        assert len(result) == 3
