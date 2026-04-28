# Hints

Read only when stuck.

- The model's forward signature is the source of truth for what your
  scheduler needs to compute. If you can describe — for one row of one
  step — what `slot_idx`, `start_pos`, and `num_new` mean, the rest is
  bookkeeping.
- "Prefill" and "decode" aren't really different operations from the
  model's perspective; they're both "feed N new tokens at position P,
  get logits for the last one." The scheduler's job is to decide N
  and P per row.
- The reference engine in `reference.py` is short. Read it to see the
  single-request lifecycle, then think about how to interleave many of
  them.
- The end-of-prefill / start-of-decode transition is the trickiest
  piece. Stare at when the first **output** token is sampled vs. when
  the last **prompt** token is consumed.

## A recipe for `step()` (read last)

If you want a fully-worked breakdown — only after you've genuinely tried.

1. Promote sequences from `_waiting` to `_running` while the cache has
   free slots.
2. Build the batch for this step. Decode-priority: every running
   sequence whose prefill is already complete gets one decode token
   (cost: 1). Use the remaining budget on prefill chunks (cost: chunk
   length, may be a partial slice of a long prompt).
3. Stack `input_ids` into a `(batch, num_new_max)` tensor padded with
   0, and build the `slot_metas` list of `(slot_idx, start_pos, num_new)`.
4. Forward pass → `(batch, vocab)` logits.
5. Greedy-sample one token per row.
6. For each row, decide whether the sampled token is "real":
   - mid-prefill (more prompt tokens still to consume): drop it,
     advance position by `num_new`.
   - prefill just finished this step (the row consumed the last chunk
     of prompt): the sampled token **is** the first output token —
     append, emit, advance position.
   - already in decode: append, emit, advance position.
7. Check stop conditions per sequence: `max_tokens` reached →
   `"max_tokens"`; sampled token in `stop_token_ids` → `"end_turn"`.
   On finish, free the slot and queue a finish `TokenEvent` for
   return.
8. Return all token + finish events from this step.

Recommended scheduler state:

```
self.model = model
self.cache = cache
self.max_tokens_per_step = max_tokens_per_step
self._waiting: deque[Sequence]     # admitted, no slot yet
self._running: list[Sequence]      # has a slot
```
