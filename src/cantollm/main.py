"""CLI entry point for CantoLLM."""

import argparse
import os
import sys
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cantollm.spec import MODEL_CONFIGS, qwen3_spec

# Reconfigure stdout to use UTF-8 encoding for emoji support
sys.stdout.reconfigure(encoding="utf-8")

# Allow MPS to use more memory (be careful - may cause system instability).
# Mac-only knob; leave the environment alone on CUDA/CPU boxes.
if sys.platform == "darwin":
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# SSL verification for HuggingFace downloads stays on by default. Some
# corporate/proxy networks need it off; that's what the standard
# HF_HUB_DISABLE_SSL_VERIFICATION env var is for — set it in the environment
# rather than have CantoLLM weaken TLS for everyone unconditionally.


def select_device(requested: str = "auto") -> torch.device:
    """Select the compute device, honoring an explicit request.

    "auto" prefers MPS (Mac) then CUDA then CPU. An explicit torch device
    string ("cuda", "cuda:1", "mps", "cpu") is validated and used as-is —
    the debugging escape hatch for bring-up on new hardware.
    """
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            sys.exit(f"error: --device {requested} requested but CUDA is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            sys.exit(f"error: --device {requested} requested but MPS is not available")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        name = torch.cuda.get_device_name(device.index or 0)
        print(f"Using device: {device} ({name})")
    elif device.type == "mps":
        print(f"Using device: {device} (Mac Silicon GPU)")
    else:
        print(f"Using device: {device} (CPU)")
    return device


# ── Subcommand: serve ───────────────────────────────────────────────

def cmd_serve(args):
    """Start the inference server."""
    import uvicorn

    from cantollm.api import create_app
    from cantollm.engine import (
        ContinuousBatchingEngine,
        EngineProcessClient,
        SequentialEngine,
    )
    from cantollm.engine.batching import BatchingConfig, build_qwen3_batched_scheduler
    from cantollm.registry import EngineRegistry
    from cantollm.runtime import build_runtime, build_tokenizer_runtime

    device = select_device(args.device)
    registry = EngineRegistry()

    if args.engine == "batched":
        if args.speculative:
            sys.exit(
                "error: --engine batched is incompatible with --speculative "
                "(speculative decoding stays on the sequential engine; "
                "batched speculation is out of scope)"
            )
        spec = qwen3_spec(args.model)
        # CUDA defaults are the measured winner (shape-buckets-results.md):
        # sdpa attention + bounded shape vocabulary + warm-up. Everywhere
        # else: padded, exact v1 geometry. Explicit flags override.
        on_cuda = device.type == "cuda"
        attention = args.attention or ("sdpa" if on_cuda else "padded")
        shape_buckets = (
            args.shape_buckets if args.shape_buckets is not None else on_cuda
        )
        warmup_shapes = (
            args.warmup_shapes if args.warmup_shapes is not None
            else shape_buckets and on_cuda
        )
        if warmup_shapes and not shape_buckets:
            sys.exit("error: --warmup-shapes requires shape buckets "
                     "(an unbounded shape vocabulary cannot be enumerated)")
        if attention == "sdpa" and not shape_buckets:
            print("warning: sdpa without --shape-buckets recompiles a cuDNN "
                  "plan per step shape — expect stall tails "
                  "(shape-buckets-results.md)")
        bucket_kwargs = {}
        if shape_buckets:
            from cantollm.engine.batching import default_shape_buckets

            bucket_kwargs = default_shape_buckets(
                args.max_batch, args.max_tokens_per_step
            )
            bucket_kwargs["warmup_shapes"] = warmup_shapes
        config = BatchingConfig(
            max_batch=args.max_batch,
            max_seq_len=args.batch_max_seq_len,
            max_tokens_per_step=args.max_tokens_per_step,
            **bucket_kwargs,
        )
        if args.in_process:
            runtime = build_runtime(spec, device, attention=attention)
            engine = ContinuousBatchingEngine.from_runtime(runtime, config)
            api_runtime = runtime
            where = "in-process"
        else:
            # The engine process loads the weights (at engine.start(), inside
            # the app lifespan); the API process only ever holds the tokenizer.
            engine = EngineProcessClient(
                build_qwen3_batched_scheduler,
                {
                    "size": args.model,
                    "device": str(device),
                    "config": config,
                    "attention": attention,
                },
            )
            api_runtime = build_tokenizer_runtime(spec)
            where = "engine process"
        model_name = spec.name
        # The per-slot capacity doubles as the admission cap.
        registry.register(
            model_name, engine, api_runtime, max_request_tokens=config.max_seq_len
        )
        engine_desc = (
            f"continuous batching, {where} (max_batch={config.max_batch}, "
            f"slot={config.max_seq_len} tok, "
            f"budget={config.max_tokens_per_step} tok/step, "
            f"attention={attention}, shape_buckets={'on' if shape_buckets else 'off'}, "
            f"warmup={'on' if warmup_shapes else 'off'})"
        )
    else:
        if args.speculative:
            main_spec = qwen3_spec(args.main_model or args.model)
            draft_spec = qwen3_spec(args.draft_model or "0.6B")
            runtime = build_runtime(main_spec, device, speculative=draft_spec)
            model_name = f"qwen3-{main_spec.size}+{draft_spec.size}-speculative"
        else:
            spec = qwen3_spec(args.model)
            runtime = build_runtime(spec, device)
            model_name = spec.name
        engine = SequentialEngine(runtime)
        # Cap admission at the RoPE table length: the sequential forward
        # indexes freqs_cis by absolute position, so prompt + max_tokens past
        # arch max_seq_len would IndexError mid-generation. A clean 400 beats
        # that.
        registry.register(
            model_name, engine, runtime,
            max_request_tokens=runtime.spec.arch["max_seq_len"],
        )
        engine_desc = "sequential"

    app = create_app(registry)

    print(f"\nCantoLLM server starting on http://{args.host}:{args.port}")
    print("  POST /v1/messages  — Anthropic-compatible Messages API")
    print("  GET  /health       — Health check")
    print("  GET  /docs         — OpenAPI docs")
    print(f"\nModel: {model_name}  ·  Engine: {engine_desc}\n")

    # uvloop + httptools, explicitly rather than via "auto": the API now
    # serves many concurrent streams (and, post-split, the IPC bridge), and
    # the end-of-phase baseline shouldn't depend on which extras happened to
    # be importable.
    uvicorn.run(
        app, host=args.host, port=args.port, log_level="info",
        loop="uvloop", http="httptools",
    )


# ── Subcommand: chat ────────────────────────────────────────────────

def cmd_chat(args):
    """Start the chat client REPL."""
    from cantollm.clients.client import run_client

    run_client(
        base_url=args.url,
        api=args.api,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        show_thinking=args.show_thinking,
    )


# ── Subcommand: webchat ─────────────────────────────────────────────

def cmd_webchat(args):
    """Start the browser-based chat client."""
    from cantollm.clients.web.server import run_server

    run_server(host=args.host, port=args.port, upstream=args.upstream)


# ── Subcommand: bench ───────────────────────────────────────────────

def cmd_bench(args):
    """Bench harness dispatch: run | ui | verify-workloads (bench-spec.md)."""
    if args.bench_command == "run":
        from cantollm.bench.executor import run_from_config_path

        if args.attach and not args.url:
            sys.exit("error: --attach requires --url")
        handle = run_from_config_path(
            args.config,
            attach_url=args.url if args.attach else None,
            capture_text=args.capture_text,
        )
        _print_bench_summary(handle)
        sys.exit(0 if handle.status == "done" else 1)

    elif args.bench_command == "ui":
        from cantollm.bench.service import run_service

        run_service(host=args.host, port=args.port)

    elif args.bench_command == "verify-workloads":
        from cantollm.bench.verify import verify_workloads

        for report in verify_workloads(model_size=args.model):
            print(
                f"{report['file']}: {report['prompts']} prompts, "
                f"input_tokens min/p50/max = {report['input_tokens_min']}"
                f"/{report['input_tokens_p50']}/{report['input_tokens_max']}"
            )
    else:
        sys.exit("usage: canto bench {run,ui,verify-workloads} ...")


def _print_bench_summary(handle):
    """Headline lines per cell — full tables live in the run dir + UI."""
    print(f"\nrun {handle.run_id}: {handle.status}")
    for state in handle.cells:
        cell = state.cell
        median = state.median or {}
        line = f"  [{state.status:<7}] {cell.workload} {cell.mode}@{cell.level:g}"
        if median.get("aggregate_tok_s") is not None:
            ttft = median.get("ttft_p50")
            line += f"  agg={median['aggregate_tok_s']:.1f} tok/s"
            if ttft is not None:
                line += f"  ttft_p50={ttft:.2f}s"
        if median.get("warnings"):
            line += f"  [{len(median['warnings'])} warning(s)]"
        if state.reason:
            line += f"  ({state.reason.splitlines()[0][:80]})"
        print(line)
    if handle.run_dir is not None:
        print(f"  -> {handle.run_dir.path}/run.json")


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
    serve_parser.add_argument("--engine", choices=("sequential", "batched"),
                              default="sequential",
                              help="Inference engine (default: sequential; "
                                   "batched = continuous batching)")
    serve_parser.add_argument("--max-batch", type=int, default=8,
                              help="Batched engine: concurrent KV slots (default: 8)")
    serve_parser.add_argument("--batch-max-seq-len", type=int, default=4096,
                              help="Batched engine: per-slot token capacity, also the "
                                   "per-request prompt+max_tokens cap (default: 4096)")
    serve_parser.add_argument("--max-tokens-per-step", type=int, default=256,
                              help="Batched engine: total new tokens per forward pass; "
                                   "bounds the prefill chunk width (default: 256)")
    serve_parser.add_argument("--attention", choices=("padded", "sdpa"),
                              default=None,
                              help="Batched engine: attention method (default: sdpa "
                                   "on CUDA, padded einsum elsewhere; sdpa = "
                                   "F.scaled_dot_product_attention via cuDNN). The "
                                   "sequential engine always uses einsum")
    serve_parser.add_argument("--shape-buckets", default=None,
                              action=argparse.BooleanOptionalAction,
                              help="Batched engine: bound the step-shape vocabulary "
                                   "(quantized prefill chunk widths, 256-token KV "
                                   "spans, power-of-two batch padding) so shape-keyed "
                                   "kernel caches (cuDNN SDPA plans, CUDA graphs) "
                                   "never compile on a live request (default: on for "
                                   "CUDA; --no-shape-buckets for exact v1 geometry)")
    serve_parser.add_argument("--warmup-shapes", default=None,
                              action=argparse.BooleanOptionalAction,
                              help="Batched engine: with shape buckets, run one dummy "
                                   "forward per vocabulary shape at startup (behind "
                                   "readiness) so every shape is warm before traffic "
                                   "(default: on when shape buckets are on for CUDA; "
                                   "--no-warmup-shapes for faster dev starts)")
    serve_parser.add_argument("--in-process", action="store_true",
                              help="Batched engine: run the scheduler inside the API "
                                   "process (debugging aid; default is a dedicated "
                                   "engine process)")
    serve_parser.add_argument("--device", default="auto",
                              help="Compute device: auto (default; MPS > CUDA > CPU) "
                                   "or an explicit torch device string like cuda, "
                                   "cuda:1, mps, cpu")

    # chat
    chat_parser = subparsers.add_parser("chat", help="Chat client (connects to a running server)")
    chat_parser.add_argument("--url", default="http://localhost:8000",
                             help="Server URL (default: http://localhost:8000)")
    chat_parser.add_argument("--api", choices=("anthropic", "openai"), default="anthropic",
                             help="API dialect to use (default: anthropic)")
    chat_parser.add_argument("--temperature", "-t", type=float, default=0.7,
                             help="Sampling temperature (default: 0.7)")
    chat_parser.add_argument("--top-p", type=float, default=0.9,
                             help="Top-p sampling threshold (default: 0.9)")
    chat_parser.add_argument("--max-tokens", type=int, default=2048,
                             help="Max tokens per response (default: 2048)")
    chat_parser.add_argument("--show-thinking", action="store_true",
                             help="Show model thinking blocks (default: hidden)")

    # webchat
    web_parser = subparsers.add_parser("webchat", help="Browser-based chat client")
    web_parser.add_argument("--upstream", default="http://localhost:8000",
                            help="API server URL (default: http://localhost:8000)")
    web_parser.add_argument("--host", default="127.0.0.1",
                            help="Bind address for the web UI (default: 127.0.0.1)")
    web_parser.add_argument("--port", type=int, default=8001,
                            help="Port for the web UI (default: 8001)")

    # bench (the harness — see bench-spec.md)
    bench_parser = subparsers.add_parser("bench", help="Benchmark harness (bench-spec.md)")
    bench_sub = bench_parser.add_subparsers(dest="bench_command")

    bench_run = bench_sub.add_parser("run", help="Execute a run config headlessly")
    bench_run.add_argument("config", help="Run config TOML (bench/configs/*.toml)")
    bench_run.add_argument("--attach", action="store_true",
                           help="Don't spawn servers; drive --url instead "
                                "(also the vLLM-comparison path)")
    bench_run.add_argument("--url", default=None,
                           help="Base URL of the already-running server (--attach)")
    bench_run.add_argument("--capture-text", action="store_true",
                           help="Also persist generated text (gitignored file; debug)")

    bench_ui = bench_sub.add_parser("ui", help="Control panel: launch/watch/compare runs")
    bench_ui.add_argument("--host", default="127.0.0.1",
                          help="Bind address (default: 127.0.0.1)")
    bench_ui.add_argument("--port", type=int, default=8002,
                          help="Port (default: 8002)")

    bench_verify = bench_sub.add_parser(
        "verify-workloads", help="Stamp real token counts into bench/workloads/*.jsonl")
    bench_verify.add_argument("--model", choices=list(MODEL_CONFIGS.keys()),
                              default="0.6B",
                              help="Tokenizer to verify against (default: 0.6B)")

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
        case "webchat":
            cmd_webchat(args)
        case "bench":
            cmd_bench(args)


if __name__ == "__main__":
    main()
