"""Shared fixtures for the continuous-batching prototype tests."""

import itertools

import pytest
import torch

from continuous_batching.padded_kv import PaddedKVCache
from continuous_batching.reference import SequentialReference
from continuous_batching.toy_model import ToyModel
from continuous_batching.cb_types import Request


VOCAB_SIZE = 32
DIM = 16
MAX_SEQ_LEN = 64


@pytest.fixture
def model():
    return ToyModel(vocab_size=VOCAB_SIZE, dim=DIM, seed=0)


@pytest.fixture
def cache_factory():
    def make(max_batch: int = 4, max_seq_len: int = MAX_SEQ_LEN, dim: int = DIM):
        return PaddedKVCache(max_batch=max_batch, max_seq_len=max_seq_len, dim=dim)
    return make


@pytest.fixture
def reference(model):
    return SequentialReference(model=model, max_seq_len=MAX_SEQ_LEN)


@pytest.fixture
def make_request():
    counter = itertools.count()

    def _make(
        prompt_len: int,
        max_tokens: int,
        stop_tokens: tuple[int, ...] = (),
        request_id: str | None = None,
    ) -> Request:
        rid = request_id if request_id is not None else f"req-{next(counter)}"
        # Deterministic, non-zero so 0-padding in batched input_ids is
        # distinguishable from real tokens. Stays under VOCAB_SIZE.
        prompt = [(i % (VOCAB_SIZE - 2)) + 2 for i in range(prompt_len)]
        return Request(
            request_id=rid,
            prompt_token_ids=prompt,
            max_tokens=max_tokens,
            stop_token_ids=set(stop_tokens),
        )

    return _make


@pytest.fixture(autouse=True)
def _set_torch_seed():
    torch.manual_seed(0)
