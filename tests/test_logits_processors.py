import torch

from cantollm.engine.logits_processors import (
    TemperatureProcessor,
    TopPProcessor,
)
from cantollm.engine.types import SamplingParams


def test_temperature_scales_logits():
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = TemperatureProcessor(2.0)(logits)
    assert torch.allclose(out, logits / 2.0)


def test_temperature_is_monotonic_in_probs():
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    hot = torch.softmax(TemperatureProcessor(0.5)(logits), dim=-1)
    cold = torch.softmax(TemperatureProcessor(2.0)(logits), dim=-1)
    # Lower temperature → sharper distribution at the max.
    assert hot.argmax().item() == cold.argmax().item() == 3
    assert hot[3] > cold[3]


def _reference_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """The pre-refactor implementation, kept here as an oracle."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    remove_mask = cumulative_probs > top_p
    sorted_indices_to_remove = torch.full_like(remove_mask, False)
    sorted_indices_to_remove[..., 1:] = remove_mask[..., :-1]
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    return logits.masked_fill(indices_to_remove, float("-inf"))


def test_top_p_matches_reference_1d():
    torch.manual_seed(0)
    logits = torch.randn(50)
    for top_p in (0.1, 0.5, 0.9, 0.95):
        assert torch.equal(TopPProcessor(top_p)(logits), _reference_top_p(logits, top_p))


def test_top_p_matches_reference_2d():
    torch.manual_seed(1)
    logits = torch.randn(3, 50)
    for top_p in (0.2, 0.7, 0.99):
        assert torch.equal(TopPProcessor(top_p)(logits), _reference_top_p(logits, top_p))


def test_top_p_keeps_at_least_one_token():
    # Even with a vanishingly small p, top-p must leave the argmax alive.
    logits = torch.tensor([-5.0, 2.0, -3.0, 0.0])
    out = TopPProcessor(0.01)(logits)
    assert out[1].item() == 2.0
    assert out.argmax().item() == 1


class TestFromTemperatureTopP:
    def test_temperature_zero_is_greedy_with_empty_pipeline(self):
        params = SamplingParams.from_temperature_top_p(0.0, 1.0)
        assert params.greedy is True
        assert params.processors == []

    def test_stochastic_builds_temperature_and_top_p(self):
        params = SamplingParams.from_temperature_top_p(0.7, 0.9)
        assert params.greedy is False
        assert len(params.processors) == 2
        assert isinstance(params.processors[0], TemperatureProcessor)
        assert isinstance(params.processors[1], TopPProcessor)
        assert params.processors[0].temperature == 0.7
        assert params.processors[1].top_p == 0.9

    def test_top_p_one_is_omitted(self):
        params = SamplingParams.from_temperature_top_p(0.7, 1.0)
        assert params.greedy is False
        assert len(params.processors) == 1
        assert isinstance(params.processors[0], TemperatureProcessor)

    def test_default_zero_arg_has_no_processors_and_is_not_greedy(self):
        params = SamplingParams()
        assert params.greedy is False
        assert params.processors == []
