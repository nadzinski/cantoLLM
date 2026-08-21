"""CLI entry point for CantoLLM."""

import argparse
import os
import sys
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cantollm.spec import MODEL_CONFIGS, known_models, qwen3_spec, resolve_spec

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
    # torch.compile's on-disk caches default to /tmp/torchinductor_$USER,
    # and systemd empties /tmp at every boot (tmpfiles.d `D /tmp`), so a
    # box reboot silently demotes the next start from the warm compile
    # bill (~+21 s behind Ready) to the cold one (~+147 s). Park the
    # cache somewhere that survives. Set before the engine process
    # spawns (it inherits the environment) and before any Inductor use;
    # an explicit TORCHINDUCTOR_CACHE_DIR still wins.
    os.environ.setdefault(
        "TORCHINDUCTOR_CACHE_DIR",
        os.path.expanduser("~/.cache/cantollm/inductor"),
    )
    import uvicorn

    from cantollm.api import create_app
    from cantollm.engine import (
        ContinuousBatchingEngine,
        EngineProcessClient,
        SequentialEngine,
    )
    from cantollm.engine.batching import BatchingConfig, build_qwen3_batched_scheduler
    from cantollm.lifecycle import BuiltEngine
    from cantollm.obs.logging import configure_logging
    from cantollm.registry import EngineRegistry
    from cantollm.runtime import build_runtime, build_tokenizer_runtime

    configure_logging("api")
    # Tracing: --otlp-endpoint exports the env var BEFORE the engine child
    # spawns (it inherits the environment and builds its own provider).
    if args.otlp_endpoint:
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = args.otlp_endpoint
    from cantollm.obs.tracing import configure_from_env

    configure_from_env("cantollm-api")
    device = select_device(args.device)
    registry = EngineRegistry()

    if args.engine == "batched":
        if args.speculative:
            sys.exit(
                "error: --engine batched is incompatible with --speculative "
                "(speculative decoding stays on the sequential engine; "
                "batched speculation is out of scope)"
            )
        spec = resolve_spec(args.model)
        # CUDA defaults are the measured winner (shape-buckets-results.md):
        # sdpa attention + bounded shape vocabulary + warm-up. Everywhere
        # else: padded, exact v1 geometry. Explicit flags override.
        on_cuda = device.type == "cuda"
        if args.model.startswith("qwen38"):
            # PoC serves eager, exact v1 geometry: the shape vocabulary,
            # warm-up sweep, graph capture, and compiled forward are all
            # typed against PaddedKVPool/uniform-KV assumptions the hybrid
            # pool does not honor.
            explicit = [
                flag for flag, value in [
                    ("--shape-buckets", args.shape_buckets),
                    ("--warmup-shapes", args.warmup_shapes),
                    ("--cuda-graphs", args.cuda_graphs),
                    ("--torch-compile", args.torch_compile),
                ] if value
            ]
            if explicit:
                sys.exit(
                    f"error: {', '.join(explicit)} not supported with qwen38 "
                    "models (the PoC serves eager, exact v1 geometry)"
                )
            args.shape_buckets = args.warmup_shapes = False
            args.cuda_graphs = args.torch_compile = False
        attention = args.attention or ("sdpa" if on_cuda else "padded")
        shape_buckets = (
            args.shape_buckets if args.shape_buckets is not None else on_cuda
        )
        warmup_shapes = (
            args.warmup_shapes if args.warmup_shapes is not None
            else shape_buckets and on_cuda
        )
        cuda_graphs = (
            args.cuda_graphs if args.cuda_graphs is not None
            else warmup_shapes and on_cuda
        )
        # Default-on for CUDA since the 2026-08-08/09 A/B cleared the
        # gates (+49.6% short_chat c=16, +64% longctx c=1, warm Ready
        # bill +21 s; run record in bench/history, decision: greedy
        # bf16 tie-drift vs eager accepted): compile joins sdpa +
        # buckets + warm-up + graphs as the fifth piece of the CUDA
        # serve default.
        torch_compile = (
            args.torch_compile if args.torch_compile is not None
            else warmup_shapes and on_cuda
        )
        if warmup_shapes and not shape_buckets:
            sys.exit("error: --warmup-shapes requires shape buckets "
                     "(an unbounded shape vocabulary cannot be enumerated)")
        if cuda_graphs and not warmup_shapes:
            sys.exit("error: --cuda-graphs requires shape buckets and "
                     "warm-up (capture must follow the eager warm, one "
                     "graph per decode shape of the bounded vocabulary)")
        if cuda_graphs and not on_cuda:
            sys.exit("error: --cuda-graphs needs a CUDA device")
        if torch_compile and not warmup_shapes:
            sys.exit("error: --torch-compile requires shape buckets and "
                     "warm-up (compiled artifacts are built by the sweep "
                     "behind readiness, never on a live request)")
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
            bucket_kwargs["cuda_graphs"] = cuda_graphs
            bucket_kwargs["torch_compile"] = torch_compile
            bucket_kwargs["torch_compile_strategy"] = args.torch_compile_strategy
        config = BatchingConfig(
            max_batch=args.max_batch,
            max_seq_len=args.batch_max_seq_len,
            max_tokens_per_step=args.max_tokens_per_step,
            **bucket_kwargs,
        )
        # Factories, not built engines: the registry's supervisor task runs
        # them in the background so uvicorn binds immediately and /ready
        # reports warm-up. The same closure rebuilds after a crash or reload.
        if args.in_process:
            def factory(spec=spec, device=device, attention=attention,
                        config=config) -> BuiltEngine:
                # Weights + compile + sweep + capture all run in here, on
                # the supervisor's worker thread.
                runtime = build_runtime(spec, device, attention=attention)
                engine = ContinuousBatchingEngine.from_runtime(runtime, config)
                return BuiltEngine(engine=engine, runtime=runtime)

            api_runtime = None
            where = "in-process"
        else:
            # The engine process loads the weights (at engine.start(), from
            # the supervisor task); the API process only ever holds the
            # tokenizer, built eagerly so /v1/messages can tokenize the
            # moment the engine is ready — and it is generation-independent,
            # so every rebuild reuses the same object.
            api_runtime = build_tokenizer_runtime(spec)

            def factory(api_runtime=api_runtime, config=config,
                        attention=attention, model=args.model,
                        device=device) -> BuiltEngine:
                engine = EngineProcessClient(
                    build_qwen3_batched_scheduler,
                    {
                        "size": model,
                        "device": str(device),
                        "config": config,
                        "attention": attention,
                    },
                )
                return BuiltEngine(engine=engine, runtime=api_runtime)

            where = "engine process"
        model_name = spec.name
        # The per-slot capacity doubles as the admission cap.
        registry.register(
            model_name, factory,
            max_request_tokens=config.max_seq_len, runtime=api_runtime,
            drain_timeout_s=args.drain_timeout,
            watchdog_timeout_s=args.watchdog_timeout,
            # Slots bound the batch; 4x slots bounds the queue behind them.
            max_inflight=args.max_inflight or 4 * config.max_batch,
            admission_timeout_s=args.admission_timeout,
        )
        engine_desc = (
            f"continuous batching, {where} (max_batch={config.max_batch}, "
            f"slot={config.max_seq_len} tok, "
            f"budget={config.max_tokens_per_step} tok/step, "
            f"attention={attention}, shape_buckets={'on' if shape_buckets else 'off'}, "
            f"warmup={'on' if warmup_shapes else 'off'}, "
            f"cuda_graphs={'on' if cuda_graphs else 'off'}, "
            f"torch_compile="
            f"{args.torch_compile_strategy if torch_compile else 'off'})"
        )
    else:
        if args.speculative:
            spec_models = (args.model, args.main_model or "", args.draft_model or "")
            if any(m.startswith("qwen38") for m in spec_models):
                sys.exit(
                    "error: --speculative is not supported with qwen38 models "
                    "(the hybrid cache cannot rewind rejected drafts)"
                )
            main_spec = qwen3_spec(args.main_model or args.model)
            draft_spec = qwen3_spec(args.draft_model or "0.6B")
            model_name = f"qwen3-{main_spec.size}+{draft_spec.size}-speculative"

            def factory(main_spec=main_spec, draft_spec=draft_spec,
                        device=device) -> BuiltEngine:
                runtime = build_runtime(main_spec, device, speculative=draft_spec)
                return BuiltEngine(SequentialEngine(runtime), runtime)

            cap_spec = main_spec
        else:
            spec = resolve_spec(args.model)
            model_name = spec.name

            def factory(spec=spec, device=device) -> BuiltEngine:
                runtime = build_runtime(spec, device)
                return BuiltEngine(SequentialEngine(runtime), runtime)

            cap_spec = spec
        # Cap admission at the RoPE table length: the sequential forward
        # indexes freqs_cis by absolute position, so prompt + max_tokens past
        # arch max_seq_len would IndexError mid-generation. A clean 400 beats
        # that.
        registry.register(
            model_name, factory,
            max_request_tokens=cap_spec.arch["max_seq_len"],
            drain_timeout_s=args.drain_timeout,
            watchdog_timeout_s=args.watchdog_timeout,
            max_inflight=args.max_inflight or 16,
            admission_timeout_s=args.admission_timeout,
        )
        engine_desc = "sequential"

    app = create_app(registry)

    print(f"\nCantoLLM server starting on http://{args.host}:{args.port}")
    print("  POST /v1/messages  — Anthropic-compatible Messages API")
    print("  GET  /health       — Health check (API liveness)")
    print("  GET  /ready        — Readiness + warm-up progress")
    print("  GET  /docs         — OpenAPI docs")
    print(f"\nModel: {model_name}  ·  Engine: {engine_desc}\n")

    # uvloop + httptools, explicitly rather than via "auto": the API now
    # serves many concurrent streams (and, post-split, the IPC bridge), and
    # the end-of-phase baseline shouldn't depend on which extras happened to
    # be importable. CantoServer wraps uvicorn with drain-on-signal; the
    # timeout_graceful_shutdown backstop means a wedged connection can never
    # hold shutdown open indefinitely.
    from cantollm.server import CantoServer, DrainController

    config = uvicorn.Config(
        app, host=args.host, port=args.port, log_level="info",
        loop="uvloop", http="httptools", timeout_graceful_shutdown=10,
    )
    drainer = DrainController(registry, drain_timeout_s=args.drain_timeout)
    try:
        CantoServer(config, drainer).run()
    finally:
        # Flush batched trace spans on the paths that unwind normally
        # (Ctrl-C re-raises as KeyboardInterrupt; a re-raised SIGTERM's
        # default handler skips this — the batch processor's 5 s cadence
        # bounds what a SIGTERM exit can lose).
        from cantollm.obs.tracing import shutdown_tracing

        shutdown_tracing()


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

def _model_choices() -> list[str]:
    choices = known_models()
    if os.environ.get("CANTOLLM_TEST_SPEC"):
        choices.append("tiny")  # chaos-suite hook; see spec.qwen3_spec
    return choices


def _add_model_args(parser):
    """Add common model/sampling arguments to a parser."""
    parser.add_argument("--model", "-m", choices=_model_choices(),
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


def _load_serve_config(path: str, serve_parser) -> dict:
    """A `[serve]` TOML table, validated against the serve parser's own
    options, becomes parser defaults: CLI flags override file values, file
    values override built-ins (argparse set_defaults + re-parse). Same
    conventions as the bench configs: underscore keys, real TOML booleans
    for the tri-state flags, unknown keys are hard errors."""
    import tomllib

    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except FileNotFoundError:
        sys.exit(f"error: config file not found: {path}")
    except tomllib.TOMLDecodeError as exc:
        sys.exit(f"error: invalid TOML in {path}: {exc}")
    unknown_tables = set(raw) - {"serve"}
    if unknown_tables:
        sys.exit(f"error: {path}: unknown table(s) {sorted(unknown_tables)}; "
                 "a serve config is a single [serve] table")
    actions = {
        a.dest: a for a in serve_parser._actions
        if a.dest not in ("help", "config")
    }
    values: dict = {}
    for key, value in raw.get("serve", {}).items():
        dest = key.replace("-", "_")
        action = actions.get(dest)
        if action is None:
            sys.exit(f"error: {path}: unknown serve option '{key}'. Known: "
                     f"{', '.join(sorted(actions))}")
        if isinstance(action, argparse.BooleanOptionalAction) or action.nargs == 0:
            if not isinstance(value, bool):
                sys.exit(f"error: {path}: '{key}' must be true or false")
        elif action.choices is not None and value not in action.choices:
            sys.exit(f"error: {path}: '{key}' must be one of "
                     f"{list(action.choices)}, got {value!r}")
        elif action.type is int:
            if not isinstance(value, int) or isinstance(value, bool):
                sys.exit(f"error: {path}: '{key}' must be an integer")
        elif action.type is float:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                sys.exit(f"error: {path}: '{key}' must be a number")
            value = float(value)
        values[dest] = value
    return values


def parse_args(argv=None):
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
    serve_parser.add_argument("--drain-timeout", type=float, default=30.0,
                              help="Seconds in-flight requests get to finish after "
                                   "SIGTERM/Ctrl-C before being aborted; a second "
                                   "signal forces immediate exit (default: 30)")
    serve_parser.add_argument("--watchdog-timeout", type=float, default=60.0,
                              help="Kill and rebuild the engine process if it has "
                                   "pending requests but makes no step progress for "
                                   "this many seconds (default: 60; 0 disables)")
    serve_parser.add_argument("--max-inflight", type=int, default=None,
                              help="Admission cap: concurrent in-flight requests "
                                   "per model (default: 4x max-batch for the "
                                   "batched engine, 16 sequential)")
    serve_parser.add_argument("--admission-timeout", type=float, default=30.0,
                              help="Seconds an over-cap request queues for a slot "
                                   "before 429 + Retry-After (default: 30)")
    serve_parser.add_argument("--otlp-endpoint", default=None,
                              help="OTLP/HTTP endpoint for request traces (e.g. "
                                   "http://localhost:4318); also honors "
                                   "OTEL_EXPORTER_OTLP_ENDPOINT. Default: tracing off")
    serve_parser.add_argument("--config", default=None, metavar="TOML",
                              help="Serve config file: a [serve] table whose keys "
                                   "are these options (underscored); CLI flags "
                                   "override file values (see serve.example.toml)")
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
    serve_parser.add_argument("--cuda-graphs", default=None,
                              action=argparse.BooleanOptionalAction,
                              help="Batched engine: capture one CUDA graph per decode "
                                   "shape at startup (after the warm-up sweep, behind "
                                   "readiness) and replay it for matching steps — "
                                   "collapses the per-step launch flood to one call "
                                   "(default: on when warm-up is on for CUDA; "
                                   "--no-cuda-graphs to serve eager)")
    serve_parser.add_argument("--torch-compile", default=None,
                              action=argparse.BooleanOptionalAction,
                              help="Batched engine: torch.compile the batched "
                                   "forward (Inductor kernel fusion; artifacts "
                                   "build during the warm-up sweep behind "
                                   "readiness, and CUDA-graph capture records "
                                   "the fused kernels; +48-64%% aggregate on "
                                   "the 2026-08 5090 A/B). Default: on when "
                                   "warm-up is on for CUDA; --no-torch-compile "
                                   "for eager kernels or bit-stable-vs-eager "
                                   "greedy output")
    serve_parser.add_argument("--torch-compile-strategy", default="dynamic",
                              choices=["dynamic", "batch-bucket"],
                              help="Artifact strategy for --torch-compile: "
                                   "dynamic = batch/width dims symbolic, a "
                                   "handful of artifacts cover the vocabulary; "
                                   "batch-bucket = one artifact per batch "
                                   "bucket with the row count baked in "
                                   "(default: dynamic)")
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

    args = parser.parse_args(argv)
    if args.command == "serve" and args.config is not None:
        # File values become serve defaults, then a re-parse lets any
        # explicitly passed CLI flag win over them.
        serve_parser.set_defaults(**_load_serve_config(args.config, serve_parser))
        args = parser.parse_args(argv)
    return args, parser


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
