"""Debug probe for ToyModel.forward.

Not part of the correctness suite — this is here to give you a place to
drop a breakpoint and inspect tensor shapes / values during a forward
pass. Three scenarios:

    1. Pure prefill: one row, all prompt tokens at once.
    2. Pure decode: one row after prefill, single token at the right position.
    3. Mixed batch: one prefill row + one decode row in the same call.

Run with:
    cd prototypes/continuous_batching
    pytest tests/test_forward_debug.py -v -s

The `-s` flag matters — without it pytest captures stdout and a
`breakpoint()` you drop in won't show its prompt.
"""

import torch

from continuous_batching.padded_kv import PaddedKVCache
from continuous_batching.toy_model import ToyModel


def test_forward_debug():
    model = ToyModel(vocab_size=32, dim=16, seed=0)
    cache = PaddedKVCache(max_batch=2, max_seq_len=64, dim=16)

    slot_a = cache.allocate_slot()
    slot_b = cache.allocate_slot()

    # --- Scenario 1: pure prefill on slot_a ---
    prompt_a = [3, 7, 11, 15, 19]
    prefill_input = torch.tensor([prompt_a], dtype=torch.long)
    prefill_metas = [(slot_a, 0, len(prompt_a))]

    print("\n[1] PREFILL")
    print(f"  input_ids: {tuple(prefill_input.shape)}  values={prefill_input.tolist()}")
    print(f"  slot_metas: {prefill_metas}")

    logits_prefill = model(prefill_input, prefill_metas, cache)

    print(f"  logits: {tuple(logits_prefill.shape)}")
    print(f"  k_cache[slot_a, :7] norms: "
          f"{cache.k_cache[slot_a, :7].norm(dim=-1).tolist()}")

    # --- Scenario 2: pure decode on slot_a (continuing from prefill) ---
    decode_input = torch.tensor([[23]], dtype=torch.long)
    decode_metas = [(slot_a, len(prompt_a), 1)]

    print("\n[2] DECODE (continuing on slot_a)")
    print(f"  input_ids: {tuple(decode_input.shape)}")
    print(f"  slot_metas: {decode_metas}  (start_pos={len(prompt_a)})")

    logits_decode = model(decode_input, decode_metas, cache)

    print(f"  logits: {tuple(logits_decode.shape)}")

    # --- Scenario 3: mixed batch — slot_a in decode + slot_b prefilling ---
    prompt_b = [5, 8, 13]
    pad = 0
    mixed_input = torch.tensor(
        [
            [25, pad, pad],   # slot_a: 1 real token (decode), padded
            prompt_b,         # slot_b: 3 real tokens (prefill)
        ],
        dtype=torch.long,
    )
    mixed_metas = [
        (slot_a, len(prompt_a) + 1, 1),   # slot_a continues from position 6
        (slot_b, 0, len(prompt_b)),       # slot_b fresh prefill
    ]

    print("\n[3] MIXED BATCH (decode + prefill in one call)")
    print(f"  input_ids: {tuple(mixed_input.shape)}")
    print(f"  slot_metas: {mixed_metas}")

    logits_mixed = model(mixed_input, mixed_metas, cache)

    print(f"  logits: {tuple(logits_mixed.shape)}")
    print(f"  per-row argmax: {logits_mixed.argmax(dim=-1).tolist()}")

    # The assertions below are loose — this test exists for inspection,
    # not correctness pinning. They just guard against shape regressions.
    assert logits_prefill.shape == (1, 32)
    assert logits_decode.shape == (1, 32)
    assert logits_mixed.shape == (2, 32)
