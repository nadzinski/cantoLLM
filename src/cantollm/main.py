"""CLI entry point for CantoLLM."""

import argparse
import os
import sys
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cantollm.models.qwen3.model import Qwen3
from cantollm.models.qwen3.tokenizer import Qwen3Tokenizer
from cantollm.models.qwen3.weights import download_weights, load_weights_into_model
from cantollm.speculative import SpeculativeBackend
from cantollm.standard import StandardBackend

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


# ── Subcommand: serve ───────────────────────────────────────────────

def cmd_serve(args):
    """Start the inference server."""
    import uvicorn

    from cantollm.api import create_app
    from cantollm.engine import SequentialEngine

    device = select_device()

    if args.speculative:
        main_size = args.main_model or args.model
        draft_size = args.draft_model or "0.6B"
        main_config = MODEL_CONFIGS[main_size]
        draft_config = MODEL_CONFIGS[draft_size]

        draft_model, draft_dir = load_model(draft_size, draft_config, device)
        main_model, _ = load_model(main_size, main_config, device)
        tokenizer = create_tokenizer(draft_dir)
        config = main_config
        model_name = f"qwen3-{main_size}+{draft_size}-speculative"

        draft_gen = StandardBackend(model=draft_model, device=device)
        main_gen = StandardBackend(model=main_model, device=device)
        backend = SpeculativeBackend(
            draft=draft_gen, main=main_gen,
            num_layers=main_config["num_transformers"],
            draft_num_layers=draft_config["num_transformers"],
        )
    else:
        model_size = args.model
        config = MODEL_CONFIGS[model_size]
        model, local_dir = load_model(model_size, config, device)
        tokenizer = create_tokenizer(local_dir)
        model_name = f"qwen3-{model_size}"

        backend = StandardBackend(model=model, device=device)

    engine = SequentialEngine(backend=backend, config=config)
    app = create_app(engine=engine, tokenizer=tokenizer, model_name=model_name)

    print(f"\nCantoLLM server starting on http://{args.host}:{args.port}")
    print("  POST /v1/messages  — Anthropic-compatible Messages API")
    print("  GET  /health       — Health check")
    print("  GET  /docs         — OpenAPI docs")
    print(f"\nModel: {model_name}\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


# ── Subcommand: chat ────────────────────────────────────────────────

def cmd_chat(args):
    """Start the chat client REPL."""
    from cantollm.clients.client import run_client

    run_client(
        base_url=args.url,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        show_thinking=args.show_thinking,
    )


# ── Subcommand: bench ───────────────────────────────────────────────

def cmd_bench(args):
    """Run concurrent requests against a running server."""
    from cantollm.clients.bench import run_bench

    run_bench(
        url=args.url,
        prompts_path=args.prompts,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        verbose=args.verbose,
        output_path=args.output,
    )


# ── Argument parsing ────────────────────────────────────────────────

def _add_model_args(parser):
    """Add common model/sampling arguments to a parser."""
    parser.add_argument("--model", "-m", choices=list(MODEL_CONFIGS.keys()),
                        default="0.6B", help="Model size (default: 0.6B)")
    parser.add_argument("--temperature", "-t", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p sampling threshold (default: 0.9)")


def _add_speculative_args(parser):
    """Add speculative decoding arguments to a parser."""
    parser.add_argument("--speculative", action="store_true",
                        help="Enable speculative decoding")
    parser.add_argument("--main-model", choices=list(MODEL_CONFIGS.keys()),
                        default=None, help="Main model for speculative decoding")
    parser.add_argument("--draft-model", choices=list(MODEL_CONFIGS.keys()),
                        default=None, help="Draft model for speculative decoding")


def parse_args():
    """Parse command line arguments with subcommands."""
    parser = argparse.ArgumentParser(description="CantoLLM — from-scratch Qwen3 inference")
    subparsers = parser.add_subparsers(dest="command")

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start the inference server")
    _add_model_args(serve_parser)
    _add_speculative_args(serve_parser)
    serve_parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    serve_parser.add_argument("--port", "-p", type=int, default=8000,
                              help="Port (default: 8000)")

    # chat
    chat_parser = subparsers.add_parser("chat", help="Chat client (connects to a running server)")
    chat_parser.add_argument("--url", default="http://localhost:8000",
                             help="Server URL (default: http://localhost:8000)")
    chat_parser.add_argument("--temperature", "-t", type=float, default=0.7,
                             help="Sampling temperature (default: 0.7)")
    chat_parser.add_argument("--top-p", type=float, default=0.9,
                             help="Top-p sampling threshold (default: 0.9)")
    chat_parser.add_argument("--max-tokens", type=int, default=2048,
                             help="Max tokens per response (default: 2048)")
    chat_parser.add_argument("--show-thinking", action="store_true",
                             help="Show model thinking blocks (default: hidden)")

    # bench
    bench_parser = subparsers.add_parser("bench", help="Fire concurrent requests at a running server")
    bench_parser.add_argument("--url", default="http://localhost:8000",
                              help="Server URL (default: http://localhost:8000)")
    bench_parser.add_argument("--prompts", required=True,
                              help="File of prompts, one per line (# for comments)")
    bench_parser.add_argument("--concurrency", "-c", type=int, default=4,
                              help="Number of concurrent workers (default: 4)")
    bench_parser.add_argument("--max-tokens", type=int, default=2048,
                              help="Max tokens per response (default: 2048)")
    bench_parser.add_argument("--temperature", "-t", type=float, default=0.7,
                              help="Sampling temperature (default: 0.7)")
    bench_parser.add_argument("--top-p", type=float, default=0.9,
                              help="Top-p sampling threshold (default: 0.9)")
    bench_parser.add_argument("--verbose", "-v", action="store_true",
                              help="Print per-request start/ttft/done events")
    bench_parser.add_argument("--output", "-o", default=None,
                              help="Write full results (including generated text) as JSON to this path")

    return parser.parse_args(), parser


def main():
    """Main entry point."""
    args, parser = parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    match args.command:
        case "serve":
            cmd_serve(args)
        case "chat":
            cmd_chat(args)
        case "bench":
            cmd_bench(args)


if __name__ == "__main__":
    main()
