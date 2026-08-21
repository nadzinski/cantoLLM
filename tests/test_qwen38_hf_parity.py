"""Reference-parity test against HF transformers for the Qwen 3.8 port.

Same commitment as test_hf_parity.py, adapted for a family with no tiny
downloadable checkpoint: tests/reference/generate_qwen38_reference.py
creates a seeded random tiny Qwen3_5ForCausalLM, saves its weights in
the real checkpoint naming, and stores fp32 CPU per-position logprobs.
This test loads those weights through models/qwen38/weights.py (the
mapping itself is under test: zero-centered norms loaded into plain
norms would fail here) and replays the same token ids.

The single forward pins: name mapping, zero-centered RMSNorm, the
attention output gate, partial RoPE layout, GDN scan math, and the
GDN gated norm. The late positions matter most: positional and
state-accumulation errors grow with distance.
"""

import json
from pathlib import Path

import pytest
import torch

from cantollm.models.attention import EinsumAttentionMethod
from cantollm.models.qwen38.model import Qwen38
from cantollm.models.qwen38.weights import load_weights_into_model
from tests.tiny_qwen38 import TINY_QWEN38_ARCH

WEIGHTS_PATH = Path(__file__).parent / "reference/qwen38_tiny_weights.safetensors"
REFERENCE_PATH = Path(__file__).parent / "reference/qwen38_tiny_hf_reference.json"

pytestmark = pytest.mark.skipif(
    not WEIGHTS_PATH.exists() or not REFERENCE_PATH.exists(),
    reason="needs the generated qwen38 tiny weights + HF reference",
)

# Same rationale as test_hf_parity.py: fp32 CPU both sides, tolerance far
# above kernel-order noise, far below any real bug's signature.
LOGPROB_ATOL = 0.05


@pytest.fixture(scope="module")
def reference() -> dict:
    return json.loads(REFERENCE_PATH.read_text())


@pytest.fixture(scope="module")
def logprob_pairs(reference):
    from safetensors.torch import load_file

    model = Qwen38(TINY_QWEN38_ARCH, attention_method=EinsumAttentionMethod())
    load_weights_into_model(model, TINY_QWEN38_ARCH, load_file(str(WEIGHTS_PATH)))
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


def test_next_token_logprobs_match(logprob_pairs):
    ours, hf, _ = logprob_pairs
    diff = (ours - hf).abs()
    assert diff.max().item() < LOGPROB_ATOL, (
        f"max |Δlogprob| {diff.max().item():.4f} at position {diff.argmax().item()}"
    )


def test_late_positions_match(logprob_pairs):
    """Errors in RoPE or GDN state accumulation grow with distance; the
    tail is where they surface."""
    ours, hf, _ = logprob_pairs
    tail = slice(len(ours) // 2, None)
    diff = (ours[tail] - hf[tail]).abs()
    assert diff.max().item() < LOGPROB_ATOL


def test_argmax_ids_match(logprob_pairs, reference):
    _, _, argmax = logprob_pairs
    assert argmax.tolist() == reference["argmax_ids"]
