"""Generate the tiny random weights AND the HF parity reference for Qwen 3.8.

    uv run --with 'transformers>=5,<6' python tests/reference/generate_qwen38_reference.py

Unlike generate_reference.py (which loads the real 0.6B checkpoint),
there is no tiny Qwen 3.8 to download: this script CREATES one with HF
transformers (Qwen3_5ForCausalLM, seeded random init, geometry from
tests/tiny_qwen38.py), saves its weights in the real checkpoint's naming
(model.language_model.* + top-level lm_head), runs one teacher-forced
fp32 CPU forward, and stores per-position next-token logprobs + argmax.

tests/test_qwen38_hf_parity.py loads the weights through cantollm's own
mapping (models/qwen38/weights.py), so one artifact pins together: the
name mapping, the zero-centered norm semantics, the attention output
gate, partial/interleaved RoPE collapse, and the GDN scan math.
"""

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

WEIGHTS_PATH = Path(__file__).parent / "qwen38_tiny_weights.safetensors"
OUT_PATH = Path(__file__).parent / "qwen38_tiny_hf_reference.json"

SEED = 42
SEQ_LEN = 48


def main() -> None:
    import transformers
    from safetensors.torch import save_file
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    from tests.tiny_qwen38 import TINY_QWEN38_ARCH as a

    cfg = Qwen3_5TextConfig(
        vocab_size=a["token_count"],
        hidden_size=a["token_embedding_dim"],
        intermediate_size=a["expanded_dim"],
        num_hidden_layers=a["num_transformers"],
        num_attention_heads=a["num_heads"],
        num_key_value_heads=a["num_groups"],
        head_dim=a["head_dim"],
        layer_types=list(a["layer_types"]),
        full_attention_interval=4,
        linear_num_key_heads=a["linear_num_k_heads"],
        linear_num_value_heads=a["linear_num_v_heads"],
        linear_key_head_dim=a["linear_head_k_dim"],
        linear_value_head_dim=a["linear_head_v_dim"],
        linear_conv_kernel_dim=a["linear_conv_kernel"],
        max_position_embeddings=a["max_seq_len"],
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        attention_bias=False,
        hidden_act="silu",
        use_cache=False,
    )
    cfg.rope_parameters = {
        "rope_type": "default",
        "rope_theta": a["rope_theta"],
        "partial_rotary_factor": a["rotary_dim"] / a["head_dim"],
    }
    cfg._attn_implementation = "eager"

    torch.manual_seed(SEED)
    model = Qwen3_5ForCausalLM(cfg).float().eval()

    gen = torch.Generator().manual_seed(SEED + 1)
    token_ids = torch.randint(
        0, a["token_count"], (SEQ_LEN,), generator=gen
    ).tolist()

    with torch.inference_mode():
        logits = model(torch.tensor(token_ids).unsqueeze(0), use_cache=False).logits[0]
    logprobs = torch.log_softmax(logits.float(), dim=-1)

    state = model.state_dict()
    renamed = {}
    for key, value in state.items():
        if ".mtp." in key or key.startswith("mtp."):
            continue
        if key == "lm_head.weight":
            renamed[key] = value.contiguous()
        elif key.startswith("model."):
            renamed["model.language_model." + key[len("model.") :]] = value.contiguous()
        else:
            raise ValueError(f"unrecognized state-dict key: {key}")
    save_file(renamed, str(WEIGHTS_PATH))
    print(f"wrote {WEIGHTS_PATH} ({len(renamed)} tensors)")

    OUT_PATH.write_text(json.dumps({
        "model": "qwen38-tiny (seeded random init)",
        "seed": SEED,
        "dtype": "float32",
        "device": "cpu",
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "token_ids": token_ids,
        "next_token_logprobs": [
            logprobs[i, token_ids[i + 1]].item() for i in range(len(token_ids) - 1)
        ],
        "argmax_ids": logits.argmax(dim=-1).tolist(),
    }, indent=1))
    print(f"wrote {OUT_PATH} ({len(token_ids)} positions)")


if __name__ == "__main__":
    main()
