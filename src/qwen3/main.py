"""CLI entry point for Qwen3 chat."""

import argparse
import os
import sys
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwen3.conversation import Conversation
from qwen3.decoder import StreamingDecoder
from qwen3.generator import TokenGenerator
from qwen3.model import Qwen3
from qwen3.presenter import Colors, TerminalPresenter
from qwen3.speculative import SpeculativeGenerator
from qwen3.tokenizer import Qwen3Tokenizer
from qwen3.weights import download_weights, load_weights_into_model

# Reconfigure stdout to use UTF-8 encoding for emoji support
sys.stdout.reconfigure(encoding="utf-8")

# Allow MPS to use more memory (be careful - may cause system instability)
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Disable SSL verification for HuggingFace (needed for some network configs)
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"


MODEL_CONFIGS = {
    "0.6B": {
        "token_count": 151_936,
        "max_seq_len": 40_960,
        "token_embedding_dim": 1024,
        "num_heads": 16,
        "num_transformers": 28,
        "expanded_dim": 3072,
        "num_groups": 8,
        "head_dim": 128,
        "dtype": torch.bfloat16,
    },
    "1.7B": {
        "token_count": 151_936,
        "max_seq_len": 40_960,
        "token_embedding_dim": 2048,
        "num_heads": 16,
        "num_transformers": 28,
        "expanded_dim": 6144,
        "num_groups": 8,
        "head_dim": 128,
        "dtype": torch.bfloat16,
    },
    "4B": {
        "token_count": 151_936,
        "max_seq_len": 40_960,
        "token_embedding_dim": 2560,
        "num_heads": 32,
        "num_transformers": 36,
        "expanded_dim": 9728,
        "num_groups": 8,
        "head_dim": 128,
        "dtype": torch.bfloat16,
    },
    "8B": {
        "token_count": 151_936,
        "max_seq_len": 40_960,
        "token_embedding_dim": 4096,
        "num_heads": 32,
        "num_transformers": 36,
        "expanded_dim": 12288,
        "num_groups": 8,
        "head_dim": 128,
        "dtype": torch.bfloat16,
    },
    "14B": {
        "token_count": 151_936,
        "max_seq_len": 40_960,
        "token_embedding_dim": 5120,
        "num_heads": 40,
        "num_transformers": 40,
        "expanded_dim": 17408,
        "num_groups": 8,
        "head_dim": 128,
        "dtype": torch.bfloat16,
    },
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Qwen3 Chat")
    parser.add_argument(
        "--model",
        "-m",
        choices=list(MODEL_CONFIGS.keys()),
        default="0.6B",
        help="Model size to use (default: 0.6B)",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling threshold (default: 0.9)",
    )
    parser.add_argument(
        "--speculative",
        action="store_true",
        help="Enable speculative decoding with two models",
    )
    parser.add_argument(
        "--main-model",
        choices=list(MODEL_CONFIGS.keys()),
        default=None,
        help="Main (verifier) model for speculative decoding (default: same as --model)",
    )
    parser.add_argument(
        "--draft-model",
        choices=list(MODEL_CONFIGS.keys()),
        default=None,
        help="Draft model for speculative decoding (default: 0.6B)",
    )
    return parser.parse_args()


def select_device() -> torch.device:
    """Select the best available device."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: {device} (Mac Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device} (NVIDIA GPU)")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device} (CPU fallback)")
    return device



def load_model(model_size: str, config: dict, device: torch.device):
    """Download weights and initialize the model."""
    print(f"Downloading {model_size} model weights...")
    local_dir, weights_dict = download_weights(model_size=model_size, use_instruct=True)

    print("Creating model...")
    model = Qwen3(qwen3_config=config)

    print("Loading pretrained weights...")
    load_weights_into_model(model, config, weights_dict)
    del weights_dict  # Free memory

    model.to(device)
    model.eval()

    return model, local_dir


def create_tokenizer(local_dir: str) -> Qwen3Tokenizer:
    """Initialize the tokenizer."""
    tokenizer_path = f"{local_dir}/tokenizer.json"
    return Qwen3Tokenizer(
        tokenizer_file_path=tokenizer_path,
        is_instruct_model=True,
        apply_chat_template=True,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def run_repl(conversation: Conversation):
    """Run the interactive chat REPL."""
    print("\nType 'quit' or 'exit' to end, 'reset' to start fresh.\n")

    while True:
        try:
            prompt = input(f"{Colors.USER}You: {Colors.RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"{Colors.RESET}\nGoodbye!")
            break

        if not prompt:
            continue

        if prompt.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if prompt.lower() == "reset":
            conversation.reset()
            print("Conversation reset.\n")
            continue

        conversation.generate_response(prompt)


def main():
    """Main entry point."""
    args = parse_args()
    device = select_device()

    if args.speculative:
        main_size = args.main_model or args.model
        draft_size = args.draft_model or "0.6B"
        main_config = MODEL_CONFIGS[main_size]
        draft_config = MODEL_CONFIGS[draft_size]

        # Load both models (draft first — smaller, faster to load)
        draft_model, draft_dir = load_model(draft_size, draft_config, device)
        main_model, _ = load_model(main_size, main_config, device)
        tokenizer = create_tokenizer(draft_dir)

        draft_gen = TokenGenerator(
            model=draft_model,
            device=device,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        main_gen = TokenGenerator(
            model=main_model,
            device=device,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        generator = SpeculativeGenerator(
            draft=draft_gen,
            main=main_gen,
            num_layers=main_config["num_transformers"],
            draft_num_layers=draft_config["num_transformers"],
        )
        # Use main model config for conversation (it determines output quality)
        config = main_config
        label = f"Qwen3 {main_size} + {draft_size} draft (speculative)"
    else:
        model_size = args.model
        config = MODEL_CONFIGS[model_size]
        model, local_dir = load_model(model_size, config, device)
        tokenizer = create_tokenizer(local_dir)

        generator = TokenGenerator(
            model=model,
            device=device,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        label = f"Qwen3 {model_size}"

    decoder = StreamingDecoder(tokenizer)
    presenter = TerminalPresenter(tokenizer)

    # Create conversation orchestrator
    conversation = Conversation(
        generator=generator,
        decoder=decoder,
        presenter=presenter,
        tokenizer=tokenizer,
        config=config,
    )

    print(f"\n{label} ready!")
    run_repl(conversation)


if __name__ == "__main__":
    main()
