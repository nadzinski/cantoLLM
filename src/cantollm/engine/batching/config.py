"""Deployment knobs for the continuous-batching engine.

Engine config, deliberately not `ModelSpec` (decision 7): these are sized to
the machine (smaller on the Mac than on the 5090), not to the model. Note the
two coexisting "max seq lens": the model's `spec.arch["max_seq_len"]` (40 960
for Qwen3) is a RoPE-table bound and far too large to preallocate KV for;
`max_seq_len` here is the per-slot pool capacity, and doubles as the
admission cap (`prompt_len + max_tokens <= max_seq_len`).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BatchingConfig:
    max_batch: int
    """Slots in the KV pool == max concurrently active sequences."""

    max_seq_len: int
    """Per-slot token capacity; also the per-request admission cap."""

    max_tokens_per_step: int
    """Total new tokens (prefill chunks + decodes) per forward pass."""

    prefill_widths: tuple[int, ...] | None = None
    """Menu of allowed prefill chunk widths, ascending (e.g. (128, 256, 512)).
    When set, the water-fill quantizes mid-prompt chunks down to menu values
    (real prompt tokens fill the width — little waste) and the step's tensor
    width (`num_new_max`) rounds up into {1} ∪ menu, so the kernel only ever
    sees widths from a fixed set. None = widths float freely (v1 behavior)."""

    kv_bucket: int | None = None
    """Round the KV gather span (`max_history_len`) up to this granularity,
    capped at `max_seq_len`. The per-row causal mask already fences reads
    past each row's real history, so the over-read is garbage-in,
    masked-out. None = the span grows token by token (v1 behavior)."""

    batch_buckets: tuple[int, ...] | None = None
    """Allowed batch sizes, ascending, last == max_batch (e.g. (1, 2, 4)).
    Steps pad to the next bucket with filler rows (num_new == 0: no KV
    write, reads slot 0 under the mask, output discarded), so a request
    joining or leaving lands on an already-seen shape. None = B is the
    exact active count (v1 behavior)."""

    warmup_shapes: bool = False
    """Run one dummy forward per shape in `shape_vocabulary()` at engine
    build time (behind the process split's Ready), so per-shape one-time
    costs (cuDNN plan compiles, later graph captures) are paid at startup,
    never on a live request. Requires all three bucket knobs."""

    def __post_init__(self) -> None:
        if self.max_batch <= 0:
            raise ValueError(f"max_batch must be positive, got {self.max_batch}")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        if self.max_tokens_per_step < self.max_batch:
            # The water-fill guarantee: budget >= active rows means every row
            # (decode rows cap at 1) gets at least one token per step, so no
            # zero-width rows ever reach the forward pass.
            raise ValueError(
                f"max_tokens_per_step ({self.max_tokens_per_step}) must be >= "
                f"max_batch ({self.max_batch})"
            )
        self._validate_shape_knobs()

    def _validate_shape_knobs(self) -> None:
        if self.prefill_widths is not None:
            widths = self.prefill_widths
            if not widths or list(widths) != sorted(set(widths)):
                raise ValueError(
                    f"prefill_widths must be ascending and unique, got {widths}"
                )
            if widths[0] < 1:
                raise ValueError(f"prefill_widths must be positive, got {widths}")
            if widths[-1] < self.max_tokens_per_step:
                # A lone prefill row can be allocated the whole step budget;
                # the step width must have a menu value to round up into.
                raise ValueError(
                    f"prefill_widths[-1] ({widths[-1]}) must be >= "
                    f"max_tokens_per_step ({self.max_tokens_per_step})"
                )
        if self.kv_bucket is not None and self.kv_bucket < 1:
            raise ValueError(f"kv_bucket must be positive, got {self.kv_bucket}")
        if self.batch_buckets is not None:
            buckets = self.batch_buckets
            if not buckets or list(buckets) != sorted(set(buckets)):
                raise ValueError(
                    f"batch_buckets must be ascending and unique, got {buckets}"
                )
            if buckets[0] < 1:
                raise ValueError(f"batch_buckets must be positive, got {buckets}")
            if buckets[-1] != self.max_batch:
                raise ValueError(
                    f"batch_buckets must end at max_batch ({self.max_batch}), "
                    f"got {buckets}"
                )
        if self.warmup_shapes and not self.shapes_bounded:
            raise ValueError(
                "warmup_shapes requires prefill_widths, kv_bucket, and "
                "batch_buckets to all be set — an unbounded vocabulary "
                "cannot be enumerated"
            )

    @property
    def shapes_bounded(self) -> bool:
        """True when every step shape comes from `shape_vocabulary()`."""
        return (
            self.prefill_widths is not None
            and self.kv_bucket is not None
            and self.batch_buckets is not None
        )

    def shape_vocabulary(self) -> list[tuple[int, int, int]]:
        """Every (batch, width, kv_len) a bounded scheduler can produce.

        Widths are {1} ∪ prefill_widths; kv spans are kv_bucket multiples
        capped at max_seq_len; a step's history always covers its own new
        tokens, so pairs with kv_len < width are unreachable and skipped.
        """
        if not self.shapes_bounded:
            raise ValueError("shape_vocabulary requires all bucket knobs set")
        widths = [1, *self.prefill_widths]
        kv_spans = list(range(self.kv_bucket, self.max_seq_len, self.kv_bucket))
        kv_spans.append(self.max_seq_len)
        return [
            (b, w, kv)
            for b in self.batch_buckets
            for w in widths
            for kv in kv_spans
            if kv >= w
        ]


def default_shape_buckets(
    max_batch: int, max_tokens_per_step: int
) -> dict[str, object]:
    """Sensible bucket knobs for `--shape-buckets`: power-of-two prefill
    widths from 128 (or the step budget, if smaller) up to the budget,
    256-token KV granularity, power-of-two batch buckets ending at
    max_batch."""
    widths = []
    w = min(128, max_tokens_per_step)
    while w < max_tokens_per_step:
        widths.append(w)
        w *= 2
    widths.append(max_tokens_per_step)
    batches = []
    b = 1
    while b < max_batch:
        batches.append(b)
        b *= 2
    batches.append(max_batch)
    return {
        "prefill_widths": tuple(dict.fromkeys(widths)),
        "kv_bucket": 256,
        "batch_buckets": tuple(dict.fromkeys(batches)),
    }
