"""Drives a `ContinuousBatchingScheduler` to completion and collects output.

Sync analogue of the API-side consumer loop. Tests use these helpers so
the scheduler-vs-reference comparisons are one-liners.
"""

from continuous_batching.scheduler import ContinuousBatchingScheduler
from continuous_batching.cb_types import FinishReason, Request


def run_to_completion(
    scheduler: ContinuousBatchingScheduler,
    requests: list[Request],
) -> tuple[dict[str, list[int]], dict[str, FinishReason]]:
    """Submit every request up front, then step until idle.

    Returns:
        (tokens, finish_reasons): tokens[req_id] is the per-request token list,
        finish_reasons[req_id] is whatever the scheduler emitted.
    """
    for req in requests:
        scheduler.add_request(req)

    tokens: dict[str, list[int]] = {req.request_id: [] for req in requests}
    finish: dict[str, FinishReason] = {}

    while not scheduler.is_idle():
        for evt in scheduler.step():
            if evt.token_id is not None:
                tokens[evt.request_id].append(evt.token_id)
            if evt.finish_reason is not None:
                finish[evt.request_id] = evt.finish_reason

    return tokens, finish


def run_with_late_arrivals(
    scheduler: ContinuousBatchingScheduler,
    schedule: list[tuple[int, Request]],
) -> tuple[dict[str, list[int]], dict[str, FinishReason]]:
    """Add requests at scheduled step indices, run until idle.

    `schedule` is a list of (step_index, request). When the loop reaches
    that step index (counted from 0), the request is admitted before the
    step runs.
    """
    by_step: dict[int, list[Request]] = {}
    for step_idx, req in schedule:
        by_step.setdefault(step_idx, []).append(req)

    tokens: dict[str, list[int]] = {req.request_id: [] for _, req in schedule}
    finish: dict[str, FinishReason] = {}

    step_idx = 0
    while True:
        for req in by_step.get(step_idx, []):
            scheduler.add_request(req)
        if scheduler.is_idle():
            break
        for evt in scheduler.step():
            if evt.token_id is not None:
                tokens[evt.request_id].append(evt.token_id)
            if evt.finish_reason is not None:
                finish[evt.request_id] = evt.finish_reason
        step_idx += 1

    return tokens, finish
