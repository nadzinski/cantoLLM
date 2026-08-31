"""Deployment knobs for the continuous-batching engine.

Engine config, deliberately not `ModelSpec` (decision 7): these are sized to
the machine (smaller on the Mac than on the 5090), not to the model. Note the
two coexisting "max seq lens": the model's `spec.arch["max_seq_len"]` (40 960
for Qwen3) is a RoPE-table bound and far too large to preallocate KV for;
`max_seq_len` here is the per-request logical capacity, and doubles as the
admission cap (`prompt_len + max_tokens <= max_seq_len`). For the padded
pool that is also literal per-slot memory; for the paged pool (Phase 4)
memory is sized separately by `num_kv_blocks`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BatchingConfig:
    max_batch: int
    """Slots in the KV pool == max concurrently active sequences."""

    max_seq_len: int
    """Per-request logical token capacity; also the admission cap. Padded:
    literal per-slot memory. Paged: the block-table length bound."""

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

    cuda_graphs: bool = False
    """Capture one CUDA graph per decode shape (width-1 vocabulary entries)
    at engine build time and replay it for matching steps
    (cuda-graphs-design.md). Requires `warmup_shapes` (and so the bucket
    knobs): capture must follow the eager warm so recordings hold the warm
    kernel choices, and an unbounded vocabulary cannot be captured."""

    torch_compile: bool = False
    """Run the batched forward through torch.compile
    (torch-compile-design.md): Inductor fuses the between-matmul kernels,
    and graph capture then records the fused kernels. Requires
    `warmup_shapes`: the sweep is where every compiled artifact gets
    built, behind Ready, never on a live request. Default off until the
    5090 A/B (design note §3.6)."""

    torch_compile_strategy: str = "dynamic"
    """How compiled artifacts map onto the shape vocabulary (design note
    §3.2, the A/B experiment). "dynamic": batch/width dims marked dynamic
    up front, a handful of artifacts cover the whole vocabulary (the kv
    span is a Python int and rides automatic dynamic). "batch-bucket":
    the batch dim is pinned static, one artifact per batch bucket with
    the row count baked into the fused kernels as a constant."""

    paged_kv: bool = False
    """Use the block-indexed paged KV pool + FlexAttention read path
    (Phase 4, paged-kv-plan.md) instead of the padded slot pool.
    `max_seq_len` then means per-request logical capacity (block-table
    length bound), not preallocated memory. On CUDA this requires
    `torch_compile`: Flex only performs compiled (flex-spike-results.md
    §7) — enforced at engine assembly, where the device is known."""

    block_size: int = 64
    """Paged pool block size in tokens. 64 because the compiled CUDA
    Flex kernels refuse mask KV blocks below it on the probed build
    (paged-kv-plan.md §2.13; enforced at engine assembly, where the
    device is known). It stays a knob: 16 (the vLLM default) is fine on
    CPU/eager and may return if a torch upgrade lowers the floor, and
    the deferred flash-attn arm would need 256 (§2.1). Ignored without
    `paged_kv`."""

    num_kv_blocks: int | None = None
    """Paged pool capacity in blocks. None = parity capacity
    (max_batch * max_seq_len / block_size): exhaustion is impossible and
    the paged engine is a drop-in. Benches undercommit explicitly to
    make scarcity happen — preemption needs someone to preempt."""

    preemption_policy: str = "lifo"
    """Victim selection under block exhaustion (paged-kv-plan.md §2.10):
    "lifo" evicts the newest-admitted active sequence, "priority" the
    lowest-priority one (LIFO tiebreak), "cost" the one with the fewest
    KV tokens to recompute. Eviction only exists under paged_kv (it is
    how an exhaustible pool refuses to deadlock), so any other layout
    rejects a non-default value as a config lie."""

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
        self._validate_paged_knobs()

    def _validate_paged_knobs(self) -> None:
        if self.block_size < 1:
            raise ValueError(
                f"block_size must be positive, got {self.block_size}"
            )
        if self.num_kv_blocks is not None and not self.paged_kv:
            raise ValueError("num_kv_blocks is a paged_kv-only knob")
        if self.preemption_policy not in ("lifo", "priority", "cost"):
            raise ValueError(
                "preemption_policy must be 'lifo', 'priority', or "
                f"'cost', got {self.preemption_policy!r}"
            )
        if self.preemption_policy != "lifo" and not self.paged_kv:
            raise ValueError("preemption_policy is a paged_kv-only knob")
        if not self.paged_kv:
            return
        if self.max_seq_len % self.block_size != 0:
            raise ValueError(
                f"paged_kv requires max_seq_len ({self.max_seq_len}) to be "
                f"a multiple of block_size ({self.block_size}): the "
                "admission cap must land on a block boundary"
            )
        blocks_per_request = self.max_seq_len // self.block_size
        if (
            self.num_kv_blocks is not None
            and self.num_kv_blocks < blocks_per_request
        ):
            raise ValueError(
                f"num_kv_blocks ({self.num_kv_blocks}) cannot hold even one "
                f"max-length request ({blocks_per_request} blocks): an "
                "admitted max-size request could never complete, alone"
            )
        if self.kv_bucket is not None and self.kv_bucket % self.block_size:
            raise ValueError(
                f"kv_bucket ({self.kv_bucket}) must be a multiple of "
                f"block_size ({self.block_size}) under paged_kv — the "
                "bucket is inert on the paged kernel path "
                "(paged-kv-plan.md §2.6) but geometry that rounds to it "
                "must still land on block boundaries"
            )

    @property
    def resolved_kv_blocks(self) -> int:
        """Paged pool capacity in blocks: `num_kv_blocks`, or parity
        capacity when unset (see the field docstring)."""
        if not self.paged_kv:
            raise ValueError("resolved_kv_blocks is meaningful only with paged_kv")
        if self.num_kv_blocks is not None:
            return self.num_kv_blocks
        return self.max_batch * (self.max_seq_len // self.block_size)

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
                "warmup_shapes requires prefill_widths and batch_buckets "
                "(plus kv_bucket on the padded layout, where the kv span "
                "is a shape): an unbounded vocabulary cannot be enumerated"
            )
        if self.cuda_graphs and not self.warmup_shapes:
            raise ValueError(
                "cuda_graphs requires warmup_shapes (and the bucket knobs): "
                "capture must follow the eager warm-up so recordings hold "
                "warm kernel choices, and one graph is captured per decode "
                "shape of the bounded vocabulary"
            )
        if self.torch_compile and not self.warmup_shapes:
            raise ValueError(
                "torch_compile requires warmup_shapes: compiled artifacts "
                "must be built by the warm-up sweep behind Ready, never on "
                "a live request"
            )
        if self.torch_compile_strategy not in ("dynamic", "batch-bucket"):
            raise ValueError(
                "torch_compile_strategy must be 'dynamic' or "
                f"'batch-bucket', got {self.torch_compile_strategy!r}"
            )

    @property
    def shapes_bounded(self) -> bool:
        """True when every step shape comes from `shape_vocabulary()`.
        Paged needs no kv_bucket: the kv axis is a value there, not a
        shape (paged-kv-plan.md §2.6), so (batch, width) alone bounds the
        vocabulary. A kv_bucket that IS set stays validated-but-inert."""
        return (
            self.prefill_widths is not None
            and self.batch_buckets is not None
            and (self.kv_bucket is not None or self.paged_kv)
        )

    def shape_vocabulary(self) -> list[tuple[int, int, int]]:
        """Every (batch, width, kv_len) a bounded scheduler can produce.

        Widths are {1} ∪ prefill_widths; kv spans are kv_bucket multiples
        capped at max_seq_len; a step's history always covers its own new
        tokens, so pairs with kv_len < width are unreachable and skipped.

        Paged (paged-kv-plan.md §2.6): the kv axis leaves shape space
        (length is table values over maximal tensors), so the vocabulary
        collapses to one entry per (batch, width) family. The third
        element stays for callers' benefit and carries `max_seq_len`, the
        constant logical bound a warm-up meta can claim; traffic's actual
        kv values never key anything.
        """
        if not self.shapes_bounded:
            raise ValueError("shape_vocabulary requires the bucket knobs set")
        widths = [1, *self.prefill_widths]
        if self.paged_kv:
            return [
                (b, w, self.max_seq_len)
                for b in self.batch_buckets
                for w in widths
            ]
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
