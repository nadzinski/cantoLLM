"""ModelRuntime: the instantiated, on-device state for one model.

Owns weights, tokenizer, device, and an `InferenceBackend`. Hands out fresh
KV caches via `new_cache()` — the engine no longer knows how deep the model
is. `build_runtime(spec, device, speculative=...)` is the factory the CLI
calls; everything per-model-specific lives here instead of `main.py`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

import torch

from cantollm.engine.backend import InferenceBackend
from cantollm.engine.batching import BatchingConfig
from cantollm.kv_cache import KVCache
from cantollm.kv_pool import PaddedKVPool
from cantollm.models.attention import (
    BatchMeta,
    EinsumAttentionMethod,
    PaddedAttentionMethod,
    SDPAAttentionMethod,
)
from cantollm.spec import ModelSpec
from cantollm.speculative import SpeculativeBackend
from cantollm.standard import StandardBackend


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

    def new_cache(self) -> KVCache:
        return KVCache(self.spec.arch["num_transformers"])

    def new_kv_pool(self, config: BatchingConfig) -> PaddedKVPool:
        """Preallocate the shared KV pool for a continuous-batching engine.

        Layer count / groups / head_dim come from `spec.arch` and dtype from
        `spec.dtype`; capacity (`max_batch`, `max_seq_len`) comes from the
        engine config. Memory only — the allocator lives with the scheduler
        (decision 1).
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
        pool: PaddedKVPool,
    ) -> torch.Tensor:
        """The batched-forward front the CB scheduler drives (decision 4).

        Satisfies `engine.batching.BatchedForwardFn`: (B, num_new_max)
        input_ids + BatchMeta + pool -> (B, vocab) logits at each row's
        last real token. The engine never imports a model class.

        The scheduler builds tensors on CPU; this is the boundary where
        they move to the model's device.
        """
        input_ids = input_ids.to(self.device)
        if meta.positions.device != self.device:
            meta = replace(
                meta,
                slots=meta.slots.to(self.device),
                start_pos=meta.start_pos.to(self.device),
                num_new=meta.num_new.to(self.device),
                positions=meta.positions.to(self.device),
            )
        return self.model.forward_batched(input_ids, meta, pool)

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
    print(f"Downloading {spec.size} model weights...")
    local_dir, weights_dict = spec.weights_loader()

    print("Creating model...")
    attention_method = _ATTENTION_METHODS[attention]()
    model = spec.model_cls(
        qwen3_config=spec.arch,
        attention_method=attention_method,
    )

    print("Loading pretrained weights...")
    spec.apply_weights(model, spec.arch, weights_dict)
    del weights_dict

    model.to(device)
    model.eval()
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
