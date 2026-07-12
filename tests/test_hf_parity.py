"""Reference-parity test against HuggingFace transformers (real 0.6B weights).

PLAN.md's cross-cutting commitment: "Reference-parity tests against HF
transformers for every new backend." This is the seed — it pins the einsum
path; extend it (padded, paged, quant, TP) rather than letting it bit-rot.

The reference (tests/reference/qwen3_0_6b_hf_reference.json) stores per-position
next-token logprobs + argmax ids computed by HF transformers in float32 on CPU
— see tests/reference/generate_reference.py for regeneration. This test replays
the same token ids through CantoLLM's own model in float32 on CPU and compares.

Why this exists: the rope_theta bug (1e5 instead of Qwen3's 1e6) survived from
the first commit until 2026-07-11 because every equivalence test compared this
codebase's paths against each other on tiny random weights — nothing ever
compared against the model the weights were trained for. Positional-encoding
errors grow with distance, hence the assertion on the *late* positions.

Skips (rather than fails) when the local checkpoint or the reference JSON is
missing. Runtime is ~15s, dominated by loading 1.4 GB of weights; run
`pytest -k "not parity"` to leave it out of a quick loop.
"""

import json
from pathlib import Path

import pytest
import torch

from cantollm.models.attention import EinsumAttentionMethod
from cantollm.models.qwen3.model import Qwen3
from cantollm.models.qwen3.weights import load_weights_into_model
from cantollm.spec import qwen3_spec

MODEL_DIR = (
    Path(__file__).resolve().parent.parent
    / "src/cantollm/models/model_data/Qwen3-0.6B"
)
REFERENCE_PATH = Path(__file__).parent / "reference/qwen3_0_6b_hf_reference.json"

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "model.safetensors").exists() or not REFERENCE_PATH.exists(),
    reason="needs the local Qwen3-0.6B checkpoint and the generated HF reference",
)

# fp32 CPU on both sides: residual disagreement is kernel summation order only.
# Measured max |Δlogprob| on the fixed prompt is ~3e-5; 0.05 leaves >1000x
# headroom while catching real bugs (the theta bug measured 0.78 max, 88.5%
# argmax agreement — over threshold on every assertion here).
LOGPROB_ATOL = 0.05


@pytest.fixture(scope="module")
def reference() -> dict:
    return json.loads(REFERENCE_PATH.read_text())


@pytest.fixture(scope="module")
def logprob_pairs(reference) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(ours, hf, our_argmax) — one fp32 CPU forward over the reference ids."""
    from safetensors.torch import load_file

    arch = dict(qwen3_spec("0.6B").arch, dtype=torch.float32)
    model = Qwen3(qwen3_config=arch, attention_method=EinsumAttentionMethod())
    weights = load_file(str(MODEL_DIR / "model.safetensors"))
    load_weights_into_model(model, arch, weights)
    del weights
    model.eval()

    token_ids = reference["token_ids"]
    with torch.inference_mode():
        logits = model(torch.tensor(token_ids).unsqueeze(0), start_pos=0)[0]
    logprobs = torch.log_softmax(logits.float(), dim=-1)

    ours = torch.tensor([
        logprobs[i, token_ids[i + 1]].item() for i in range(len(token_ids) - 1)
    ])
    hf = torch.tensor(reference["next_token_logprobs"])
    return ours, hf, logits.argmax(dim=-1)


def test_next_token_logprobs_match_hf(logprob_pairs):
    ours, hf, _ = logprob_pairs
    diff = (ours - hf).abs()
    assert diff.max() < LOGPROB_ATOL, (
        f"max |Δlogprob| = {diff.max():.4f} at position {int(diff.argmax())} "
        f"of {len(diff)} (mean {diff.mean():.5f}) — model output has drifted "
        "from the HF reference"
    )


def test_late_positions_match_hf(logprob_pairs):
    """Positional-encoding bugs (wrong rope_theta, mask/offset errors) grow
    with distance; pin the last quarter separately so a small early-position
    tolerance can never mask a large late-position drift."""
    ours, hf, _ = logprob_pairs
    tail = len(ours) * 3 // 4
    diff = (ours[tail:] - hf[tail:]).abs()
    assert diff.max() < LOGPROB_ATOL, (
        f"max |Δlogprob| over positions {tail}.. = {diff.max():.4f} — "
        "distance-dependent drift (check RoPE / mask geometry)"
    )


def test_greedy_argmax_matches_hf(logprob_pairs, reference):
    _, _, our_argmax = logprob_pairs
    hf_argmax = torch.tensor(reference["argmax_ids"])
    agree = (our_argmax == hf_argmax).float().mean().item()
    assert agree >= 0.99, (
        f"greedy argmax agrees with HF at only {agree:.1%} of positions"
    )
