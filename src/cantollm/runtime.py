"""ModelRuntime: the instantiated, on-device state for one model.

Owns weights, tokenizer, device, and an `InferenceBackend`. Hands out fresh
KV caches via `new_cache()` — the engine no longer knows how deep the model
is. `build_runtime(spec, device, speculative=...)` is the factory the CLI
calls; everything per-model-specific lives here instead of `main.py`.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Literal

import torch

from cantollm.engine.backend import InferenceBackend
from cantollm.engine.batching import BatchingConfig
from cantollm.kv_cache import KVCache
from cantollm.kv_pool import KVPool, PaddedKVPool, PagedKVPool
from cantollm.models.attention import (
    BatchMeta,
    EinsumAttentionMethod,
    PaddedAttentionMethod,
    PagedTables,
    SDPAAttentionMethod,
)
from cantollm import progress
from cantollm.spec import ModelSpec
from cantollm.speculative import SpeculativeBackend
from cantollm.standard import StandardBackend

logger = logging.getLogger(__name__)


def move_batch_to(
    input_ids: torch.Tensor, meta: BatchMeta, device: torch.device
) -> tuple[torch.Tensor, BatchMeta]:
    """The boundary where scheduler-built CPU tensors move to the model
    device. Move-gate by `.to` identity, not device equality: a bare
    "cuda" device compares unequal to a tensor's resolved "cuda:0", and
    `replace` on an already-on-device meta would drop a seeded
    kv_write_map (graph capture's static buffers) and rebuild it
    mid-recording — the H2D copy invalidates the capture (40fbcf9).

    Seeded passengers must survive the replace: `replace` builds a fresh
    instance whose seeded slots are empty. For kv_write_map (warm-up's
    scratch maps; capture metas never take this branch, their tensors are
    already on-device), re-deriving would produce an EMPTY map for
    all-filler rows — silently un-seeding the sweep. For paged_tables
    there is no derivation at all: dropping them turns the first paged
    forward into a "no paged tables" error, or worse, a stale-reference
    split. Move the columns with the meta, both passengers.
    """
    input_ids = input_ids.to(device)
    positions = meta.positions.to(device)
    if positions is not meta.positions:
        seeded = meta.__dict__.get("kv_write_map")
        paged = meta.__dict__.get("paged_tables")
        meta = replace(
            meta,
            slots=meta.slots.to(device),
            start_pos=meta.start_pos.to(device),
            num_new=meta.num_new.to(device),
            positions=positions,
        )
        if seeded is not None:
            meta.seed_kv_write_map(
                type(seeded)(*(t.to(device) for t in seeded))
            )
        if paged is not None:
            meta.seed_paged_tables(PagedTables(
                block_tables=paged.block_tables.to(device),
                kv_num_blocks=paged.kv_num_blocks.to(device),
                inverse_tables=paged.inverse_tables.to(device),
                write_map=type(paged.write_map)(
                    *(t.to(device) for t in paged.write_map)
                ),
            ))
    return input_ids, meta


class ModelRuntime:
    def __init__(
        self,
        spec: ModelSpec,
        device: torch.device,
        model: torch.nn.Module,
        tokenizer: Any,
        backend: InferenceBackend,
    ):
        self.spec = spec
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.backend = backend
        self._compiled_batched = None  # both set by enable_torch_compile()
        self._compile_strategy = "dynamic"

    def new_cache(self) -> KVCache:
        return KVCache(self.spec.arch["num_transformers"])

    def new_kv_pool(self, config: BatchingConfig) -> KVPool:
        """Preallocate the shared KV pool for a continuous-batching engine.

        Layer count / groups / head_dim come from `spec.arch` and dtype from
        `spec.dtype`; capacity comes from the engine config. This is the
        layout branch: `config.paged_kv` selects the paged pool (Phase 4;
        memory sized by `num_kv_blocks`) over the padded slot pool (memory
        sized by `max_batch` x `max_seq_len`). Memory only either way — the
        allocator lives with the scheduler (decision 1).
        """
        # A step mixes a decode row near its slot end (position up to
        # max_seq_len - 1) with a prefill row up to max_tokens_per_step wide;
        # the batched RoPE gather indexes freqs_cis at the decode row's padded
        # columns, reaching (max_seq_len - 1) + (max_tokens_per_step - 1). The
        # freqs_cis table has `arch["max_seq_len"]` rows, so guard here rather
        # than let a rare step IndexError mid-flight.
        rope_len = self.spec.arch["max_seq_len"]
        max_rope_index = config.max_seq_len + config.max_tokens_per_step - 2
        if max_rope_index >= rope_len:
            raise ValueError(
                f"max_seq_len ({config.max_seq_len}) + max_tokens_per_step "
                f"({config.max_tokens_per_step}) exceeds the RoPE table length "
                f"({rope_len}); a padded decode row could index freqs_cis out "
                f"of range. Lower either, or raise the model's max_seq_len."
            )
        if config.paged_kv:
            return PagedKVPool(
                num_layers=self.spec.arch["num_transformers"],
                num_kv_blocks=config.resolved_kv_blocks,
                block_size=config.block_size,
                max_seq_len=config.max_seq_len,
                num_groups=self.spec.arch["num_groups"],
                head_dim=self.spec.arch["head_dim"],
                dtype=self.spec.dtype,
                device=self.device,
            )
        return PaddedKVPool(
            num_layers=self.spec.arch["num_transformers"],
            max_batch=config.max_batch,
            max_seq_len=config.max_seq_len,
            num_groups=self.spec.arch["num_groups"],
            head_dim=self.spec.arch["head_dim"],
            dtype=self.spec.dtype,
            device=self.device,
        )

    @torch.inference_mode()
    def forward_batched(
        self,
        input_ids: torch.Tensor,
        meta: BatchMeta,
        pool: KVPool,
    ) -> torch.Tensor:
        """The batched-forward front the CB scheduler drives (decision 4).

        Satisfies `engine.batching.BatchedForwardFn`: (B, num_new_max)
        input_ids + BatchMeta + pool -> (B, vocab) logits at each row's
        last real token. The engine never imports a model class.

        The scheduler builds tensors on CPU; this is the boundary where
        they move to the model's device (`move_batch_to`).
        """
        input_ids, meta = move_batch_to(input_ids, meta, self.device)
        if self._compiled_batched is None:
            return self.model.forward_batched(input_ids, meta, pool)
        # The hoists (torch-compile-design.md §3.1): validation is host
        # Python over meta.rows, and forcing kv_write_map here makes every
        # traced read of the property a cache hit. Inside the traced region
        # either one is poison: rows reads guard on per-step values and
        # recompile every step; the property's miss path takes a lock
        # Dynamo cannot trace. Force AFTER the device move: replace() drops
        # the cache (the 40fbcf9 lesson).
        self.model._validate_batched(meta, pool)
        _ = meta.kv_write_map
        self._mark_compile_dims(input_ids, meta)
        # The attention method's dispatcher state (sdpa's cuDNN pin) is
        # entered here, around — never inside — the traced region: a
        # traced sdpa_kernel context bypasses the AOTAutograd/FX caches
        # and the warm-up recompiles the world every boot (2026-08-08).
        # Trace-time dispatch happens under this context too, since the
        # artifacts are built by warm-up forwards passing through here.
        with self.model.attention_method.execution_context():
            return self._compiled_batched(input_ids, meta, pool)

    def enable_torch_compile(
        self, strategy: str = "dynamic", backend="inductor"
    ) -> None:
        """Swap the batched serving path onto a torch.compile'd
        `forward_batched_impl` (torch-compile-design.md).

        fullgraph=True is the tripwire: a future graph break fails loudly
        at the first warm-up forward instead of silently fragmenting the
        step into eager pieces. `strategy` is the §3.2 artifact question:
        "dynamic" or "batch-bucket", validated by BatchingConfig and
        applied per step in `_mark_compile_dims`. `backend` exists for
        tests (a name like "eager", or a callable, exercises tracing
        without paying Inductor); serving uses the default. Call before
        the warm-up sweep so every artifact compiles behind Ready, never
        on a live request.
        """
        # Dynamo's per-function recompile limit defaults to 8, counted on
        # the code object, and with fullgraph=True hitting it is a hard
        # error at serve time rather than a fallback to eager. The
        # batch-bucket strategy alone wants one artifact per bucket plus
        # kv/width promotions, so size the cache to the vocabulary with
        # headroom.
        torch._dynamo.config.cache_size_limit = 64
        self._compile_strategy = strategy
        self._compiled_batched = torch.compile(
            self.model.forward_batched_impl, fullgraph=True, backend=backend
        )

    def _mark_compile_dims(self, input_ids, meta: BatchMeta) -> None:
        """Per-step shape hints for the §3.2 strategy.

        "dynamic": batch dims are marked dynamic so the first compile is
        already symbolic, skipping automatic dynamic's static-first
        stepping stones. mark_dynamic is a promise the dim will not
        specialize, and torch hard-errors when the promise breaks, which
        constrains the marking twice over: 0/1-sized dims always
        specialize (so mark only at sizes > 1), and the width dim is tied
        to the Python int `meta.num_new_max` (the mask arange burns it
        in), so width gets the soft `maybe_mark_dynamic` instead and goes
        symbolic when automatic dynamic promotes the int on its second
        value. The kv span (`max_history_len`) is the same story with no
        tensor dim to mark at all.

        "batch-bucket": the batch dim is pinned static instead, so each
        batch bucket compiles its own artifact with the row count baked
        in. The write-map length still varies with the real-row count
        inside a bucket (fillers are skipped by construction), so map
        columns are never pinned.
        """
        m = meta.kv_write_map
        batch_dim = (input_ids, meta.positions, meta.slots,
                     meta.start_pos, meta.num_new)
        if self._compile_strategy == "batch-bucket":
            for t in batch_dim:
                torch._dynamo.mark_static(t, 0)
            return
        for t in batch_dim + (m.row, m.off, m.slot, m.pos):
            if t.shape[0] > 1:
                torch._dynamo.mark_dynamic(t, 0)
        if input_ids.shape[1] > 1:
            torch._dynamo.maybe_mark_dynamic(input_ids, 1)
            torch._dynamo.maybe_mark_dynamic(meta.positions, 1)

    async def start(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass


class TokenizerRuntime:
    """API-process counterpart of ModelRuntime when the engine runs in its
    own process: the API layer needs `spec` metadata and a tokenizer
    (tokenization stays API-side, per Phase 1a) and must not pay for weights
    it never touches — those live in the engine process. Satisfies the
    registry's runtime surface (tokenizer/start/shutdown)."""

    def __init__(self, spec: ModelSpec, tokenizer: Any):
        self.spec = spec
        self.tokenizer = tokenizer

    async def start(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass


def build_tokenizer_runtime(spec: ModelSpec) -> TokenizerRuntime:
    """Fetch tokenizer files only (no weights) and build the API-side
    runtime for a model served from an engine process."""
    return TokenizerRuntime(spec, spec.tokenizer_factory(spec.tokenizer_files_loader()))


_ATTENTION_METHODS = {
    "einsum": EinsumAttentionMethod,
    "padded": PaddedAttentionMethod,
    "sdpa": SDPAAttentionMethod,
}


def _load_model(
    spec: ModelSpec,
    device: torch.device,
    attention: Literal["einsum", "padded", "sdpa"] = "einsum",
) -> tuple[torch.nn.Module, str]:
    logger.info("Downloading %s model weights...", spec.size)
    progress.report("load", 0, 3, "downloading weights")
    local_dir, weights_dict = spec.weights_loader()

    logger.info("Creating model...")
    progress.report("load", 1, 3, "creating model")
    attention_method = _ATTENTION_METHODS[attention]()
    model = spec.model_cls(
        qwen3_config=spec.arch,
        attention_method=attention_method,
    )

    logger.info("Loading pretrained weights...")
    progress.report("load", 2, 3, "applying weights")
    spec.apply_weights(model, spec.arch, weights_dict)
    del weights_dict

    model.to(device)
    model.eval()
    progress.report("load", 3, 3, "model on device")
    return model, local_dir


def build_runtime(
    spec: ModelSpec,
    device: torch.device,
    *,
    speculative: ModelSpec | None = None,
    attention: Literal["einsum", "padded", "sdpa"] = "einsum",
) -> ModelRuntime:
    if speculative is not None and attention != "einsum":
        # Speculative decoding stays on the sequential engine (PLAN.md:
        # batched speculation is explicitly out of scope).
        raise ValueError("speculative runtimes are sequential-only (attention='einsum')")
    if speculative is not None:
        draft_model, draft_dir = _load_model(speculative, device)
        main_model, _ = _load_model(spec, device)
        tokenizer = speculative.tokenizer_factory(draft_dir)
        draft_gen = StandardBackend(model=draft_model, device=device)
        main_gen = StandardBackend(model=main_model, device=device)
        backend: InferenceBackend = SpeculativeBackend(
            draft=draft_gen,
            main=main_gen,
            num_layers=spec.arch["num_transformers"],
            draft_num_layers=speculative.arch["num_transformers"],
        )
        return ModelRuntime(
            spec=spec, device=device, model=main_model,
            tokenizer=tokenizer, backend=backend,
        )

    model, local_dir = _load_model(spec, device, attention)
    tokenizer = spec.tokenizer_factory(local_dir)
    backend = StandardBackend(model=model, device=device)
    return ModelRuntime(
        spec=spec, device=device, model=model,
        tokenizer=tokenizer, backend=backend,
    )
