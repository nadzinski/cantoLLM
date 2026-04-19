"""ModelRuntime: the instantiated, on-device state for one model.

Owns weights, tokenizer, device, and an `InferenceBackend`. Hands out fresh
KV caches via `new_cache()` — the engine no longer knows how deep the model
is. `build_runtime(spec, device, speculative=...)` is the factory the CLI
calls; everything per-model-specific lives here instead of `main.py`.
"""

from __future__ import annotations

from typing import Any

import torch

from cantollm.engine.backend import InferenceBackend
from cantollm.kv_cache import KVCache
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

    async def start(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass


def _load_model(spec: ModelSpec, device: torch.device) -> tuple[torch.nn.Module, str]:
    print(f"Downloading {spec.size} model weights...")
    local_dir, weights_dict = spec.weights_loader()

    print("Creating model...")
    model = spec.model_cls(qwen3_config=spec.arch)

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
) -> ModelRuntime:
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

    model, local_dir = _load_model(spec, device)
    tokenizer = spec.tokenizer_factory(local_dir)
    backend = StandardBackend(model=model, device=device)
    return ModelRuntime(
        spec=spec, device=device, model=model,
        tokenizer=tokenizer, backend=backend,
    )
