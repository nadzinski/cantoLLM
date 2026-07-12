"""Generate the HF-transformers parity reference for tests/test_hf_parity.py.

Run once (or whenever the reference needs refreshing):

    uv run --with transformers python tests/reference/generate_reference.py

Loads the local Qwen3-0.6B checkpoint through HuggingFace transformers in
float32 on CPU, runs one teacher-forced forward over a fixed chat-wrapped
prompt, and writes per-position next-token logprobs + argmax ids to
qwen3_0_6b_hf_reference.json. The parity test replays the stored token ids
through CantoLLM's own stack (also fp32/CPU) and compares.

float32 on CPU keeps the comparison deterministic and the tolerance tight —
bf16 kernel-order noise would force a tolerance loose enough to hide real
bugs. transformers is deliberately NOT a project dependency; the ephemeral
`--with` install is only needed for generation, never for running the test.
"""

import json
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = REPO_ROOT / "src/cantollm/models/model_data/Qwen3-0.6B"
OUT_PATH = Path(__file__).parent / "qwen3_0_6b_hf_reference.json"

PROMPT_TEXT = (
    "Briefly explain how a paged KV cache differs from a contiguous "
    "preallocated one in an LLM inference engine, and why block tables "
    "make preemption cheaper. Mention fragmentation, copy-on-write "
    "prefix sharing, and the cost model for recompute-versus-swap "
    "preemption, then give a one-sentence summary a new contributor "
    "could repeat from memory."
)


def main() -> None:
    import transformers
    from transformers import AutoModelForCausalLM

    from cantollm.models.qwen3.tokenizer import Qwen3Tokenizer

    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=str(MODEL_DIR / "tokenizer.json"),
        is_instruct_model=True,
        apply_chat_template=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    token_ids = tokenizer.encode_conversation(
        [{"role": "user", "content": PROMPT_TEXT}]
    )
    print(f"prompt: {len(token_ids)} tokens")

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.float32
    )
    model.eval()

    with torch.inference_mode():
        logits = model(torch.tensor(token_ids).unsqueeze(0)).logits[0]
    logprobs = torch.log_softmax(logits.float(), dim=-1)

    next_token_logprobs = [
        logprobs[i, token_ids[i + 1]].item() for i in range(len(token_ids) - 1)
    ]
    argmax_ids = logits.argmax(dim=-1).tolist()

    OUT_PATH.write_text(json.dumps({
        "model": "Qwen3-0.6B",
        "dtype": "float32",
        "device": "cpu",
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "prompt_text": PROMPT_TEXT,
        "token_ids": token_ids,
        "next_token_logprobs": next_token_logprobs,
        "argmax_ids": argmax_ids,
    }, indent=1))
    print(f"wrote {OUT_PATH} ({len(token_ids)} positions)")


if __name__ == "__main__":
    main()
