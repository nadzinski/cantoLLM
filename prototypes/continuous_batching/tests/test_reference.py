"""Sanity checks on the SequentialReference oracle.

If these don't pass, the scheduler-equivalence tests can't be trusted —
the reference is what they compare against.
"""

import pytest


def test_reference_runs_and_emits_max_tokens(reference, make_request):
    req = make_request(prompt_len=5, max_tokens=7)
    out = reference.generate(req)
    assert len(out) == 7


def test_reference_is_deterministic(reference, make_request):
    req1 = make_request(prompt_len=5, max_tokens=6, request_id="x")
    req2 = make_request(prompt_len=5, max_tokens=6, request_id="x")
    assert reference.generate(req1) == reference.generate(req2)


def test_reference_stops_on_stop_token(reference, make_request):
    # Run once with no stop tokens to learn what the model emits.
    probe = make_request(prompt_len=5, max_tokens=10, request_id="probe")
    full = reference.generate(probe)
    assert len(full) >= 4, "need at least 4 tokens to test mid-stream stop"

    stop = full[3]
    if stop in full[:3]:
        pytest.skip("stop token also appears earlier; skip this seed")

    req = make_request(
        prompt_len=5, max_tokens=10, stop_tokens=(stop,), request_id="stop",
    )
    out = reference.generate(req)
    assert out == full[:4]
