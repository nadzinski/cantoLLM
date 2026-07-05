"""Tests for the shared sampler (engine/sampler.py).

The headline pin: greedy sampling runs the processor pipeline. The old
StandardBackend shortcut argmaxed raw logits — sound for temperature/top-p,
silently wrong for any processor that moves specific token logits
(repetition penalty, logit bias, guided decoding).
"""

import torch

from cantollm.engine import sampler
from cantollm.engine.types import SamplingParams
from cantollm.standard import StandardBackend


class RankInverter:
    """Processor that inverts the ranking of every token."""

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        return -logits


class TestGreedyRunsPipeline:
    def test_greedy_applies_processors(self):
        """Greedy must argmax the *processed* logits, not the raw ones."""
        logits = torch.tensor([1.0, 2.0, 3.0])
        sampling = SamplingParams(processors=[RankInverter()], greedy=True)

        token, probs = sampler.sample(logits, sampling)

        # Raw argmax is 2; the inverter makes token 0 the winner.
        assert token.item() == 0
        assert probs.argmax().item() == 0

    def test_greedy_probs_reflect_pipeline(self):
        """The returned probs come from the post-pipeline distribution."""
        logits = torch.tensor([0.0, 0.0, 10.0])
        sampling = SamplingParams(processors=[RankInverter()], greedy=True)

        _, probs = sampler.sample(logits, sampling)

        # After inversion, token 2 has logit -10 and near-zero probability.
        assert probs[2].item() < 1e-3

    def test_greedy_without_processors_is_plain_argmax(self):
        logits = torch.tensor([1.0, 3.0, 2.0])
        sampling = SamplingParams(greedy=True)

        token, probs = sampler.sample(logits, sampling)

        assert token.item() == 1
        assert torch.allclose(probs, torch.softmax(logits, dim=-1))


class TestShapes:
    def test_greedy_batched(self):
        logits = torch.tensor([[1.0, 2.0], [4.0, 3.0]])
        sampling = SamplingParams(greedy=True)

        token, probs = sampler.sample(logits, sampling)

        assert token.shape == (2,)
        assert token.tolist() == [1, 0]
        assert probs.shape == (2, 2)

    def test_multinomial_batched(self):
        torch.manual_seed(0)
        logits = torch.randn(3, 16)
        sampling = SamplingParams.from_temperature_top_p(temperature=0.7, top_p=0.9)

        token, probs = sampler.sample(logits, sampling)

        assert token.shape == (3,)
        assert probs.shape == (3, 16)

    def test_multinomial_1d(self):
        torch.manual_seed(0)
        logits = torch.randn(16)
        sampling = SamplingParams.from_temperature_top_p(temperature=0.7, top_p=1.0)

        token, probs = sampler.sample(logits, sampling)

        assert token.dim() == 0
        assert probs.shape == (16,)


class TestMultinomialUnchanged:
    def test_matches_pre_extraction_semantics(self):
        """Pipeline → softmax → multinomial, byte-identical to the old code."""
        logits = torch.randn(1, 50)
        sampling = SamplingParams.from_temperature_top_p(temperature=0.7, top_p=0.9)

        torch.manual_seed(123)
        token, probs = sampler.sample(logits.clone(), sampling)

        ref_logits = logits.clone()
        for processor in sampling.processors:
            ref_logits = processor(ref_logits)
        ref_probs = torch.softmax(ref_logits, dim=-1)
        torch.manual_seed(123)
        ref_token = torch.multinomial(ref_probs, num_samples=1).squeeze(-1)

        assert torch.equal(token, ref_token)
        assert torch.equal(probs, ref_probs)


class TestStandardBackendDelegates:
    def test_backend_sample_matches_shared_sampler(self):
        """StandardBackend.sample/get_probs are thin delegates (model unused)."""
        backend = StandardBackend(model=None, device=torch.device("cpu"))
        logits = torch.tensor([1.0, 2.0, 3.0])
        sampling = SamplingParams(processors=[RankInverter()], greedy=True)

        token, probs = backend.sample(logits, sampling)

        assert token.item() == 0
        assert torch.equal(probs, sampler.get_probs(logits, sampling))
        assert torch.equal(backend.get_probs(logits, sampling), probs)
