"""ModelSpec wiring for the Qwen 3.8 family.

Reached through cantollm.spec.resolve_spec ("qwen38-27B"); everything
model-specific (arch table, loaders, tokenizer, hybrid cache/pool
factories) is closed over here so the shared spec/runtime stay
family-agnostic.
"""

import torch

from cantollm.models.qwen38 import weights as qwen38_weights
from cantollm.models.qwen38.model import Qwen38, qwen38_layer_types
from cantollm.models.qwen38.pool import HybridCache, HybridStatePool
from cantollm.models.qwen38.tokenizer import Qwen38Tokenizer
from cantollm.spec import ModelSpec

QWEN38_CONFIGS = {
    "27B": {
        "token_count": 248_320,
        "token_embedding_dim": 5120,
        "expanded_dim": 17408,
        "num_transformers": 64,
        "layer_types": qwen38_layer_types(64),
        "num_heads": 24,
        "num_groups": 4,
        "head_dim": 256,
        "rotary_dim": 64,  # partial_rotary_factor 0.25 of head_dim
        "rope_theta": 10_000_000.0,
        "linear_num_k_heads": 16,
        "linear_num_v_heads": 48,
        "linear_head_k_dim": 128,
        "linear_head_v_dim": 128,
        "linear_conv_kernel": 4,
        # PoC cap; the checkpoint supports 262,144 natively. The freqs
        # table is (max_seq_len, 32) complex64, 16MB at this cap; raise
        # it when a serve config actually needs longer requests.
        "max_seq_len": 65536,
        "dtype": torch.bfloat16,
        # Build params as meta shapes; the loader materializes them from
        # the checkpoint one at a time (54GB of random bf16 init on the
        # host is the alternative).
        "init_device": "meta",
    },
}


def qwen38_spec(size: str) -> ModelSpec:
    if size not in QWEN38_CONFIGS:
        raise ValueError(
            f"Unknown Qwen3.8 size '{size}'. Must be one of {list(QWEN38_CONFIGS)}"
        )
    arch = QWEN38_CONFIGS[size]

    def _load_weights() -> tuple[str, object]:
        return qwen38_weights.download_weights()

    def _tokenizer_files() -> str:
        return qwen38_weights.download_tokenizer()

    def _build_tokenizer(local_dir: str) -> Qwen38Tokenizer:
        return Qwen38Tokenizer(
            tokenizer_file_path=f"{local_dir}/tokenizer.json",
            is_instruct_model=True,
            apply_chat_template=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )

    def _cache_factory() -> HybridCache:
        return HybridCache(arch["layer_types"])

    def _kv_pool_factory(config, device) -> HybridStatePool:
        # Same RoPE-table guard as the default pool path (see
        # runtime.new_kv_pool): a padded decode row's freqs gather can
        # reach max_seq_len + max_tokens_per_step - 2.
        rope_len = arch["max_seq_len"]
        max_rope_index = config.max_seq_len + config.max_tokens_per_step - 2
        if max_rope_index >= rope_len:
            raise ValueError(
                f"max_seq_len ({config.max_seq_len}) + max_tokens_per_step "
                f"({config.max_tokens_per_step}) exceeds the RoPE table length "
                f"({rope_len}); a padded decode row could index freqs_cis out "
                f"of range. Lower either, or raise the model's max_seq_len."
            )
        return HybridStatePool.from_arch(
            arch,
            max_batch=config.max_batch,
            max_seq_len=config.max_seq_len,
            dtype=arch["dtype"],
            device=device,
        )

    return ModelSpec(
        name=f"qwen38-{size}",
        size=size,
        arch=arch,
        dtype=arch["dtype"],
        model_cls=Qwen38,
        weights_loader=_load_weights,
        apply_weights=qwen38_weights.load_weights_into_model,
        tokenizer_factory=_build_tokenizer,
        tokenizer_files_loader=_tokenizer_files,
        chat_template="qwen38-chatml",
        cache_factory=_cache_factory,
        kv_pool_factory=_kv_pool_factory,
    )
