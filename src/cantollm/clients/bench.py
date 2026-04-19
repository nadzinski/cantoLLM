"""Concurrent request bench for the CantoLLM Messages API.

Fires requests at a running server via a ThreadPoolExecutor, reusing ChatClient
in quiet mode. The server is FCFS today, so parallel client requests queue
server-side; this harness verifies connection handling and stream separation,
and will stay useful once a batching engine lands.
"""

import json
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from cantollm.clients.client import ChatClient, Colors


@dataclass
class BenchResult:
    request_id: int
    prompt: str
    submit_time: float
    start_time: float | None = None
    first_token_time: float | None = None
    end_time: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    text_tokens: int = 0
    stop_reason: str | None = None
    error: str | None = None
    output_text: str = ""


def _load_prompts(path: Path) -> list[str]:
    prompts = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        prompts.append(line)
    return prompts


def _health_check(base_url: str) -> str | None:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/health", timeout=5) as resp:
            if resp.status != 200:
                return f"server returned status {resp.status}"
    except Exception as e:
        return str(e)
    return None


def _run_one(
    request_id: int,
    prompt: str,
    submit_time: float,
    url: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    model: str,
    verbose: bool,
    print_lock: threading.Lock,
) -> BenchResult:
    result = BenchResult(request_id=request_id, prompt=prompt, submit_time=submit_time)

    client = ChatClient(
        base_url=url,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        model=model,
        quiet=True,
    )

    result.start_time = time.perf_counter()
    if verbose:
        with print_lock:
            print(f"[req-{request_id:02d}] start  (queue_wait={result.start_time - submit_time:.2f}s)")

    try:
        resp = client.send_message(prompt, stream=True)
    except Exception as e:
        result.end_time = time.perf_counter()
        result.error = f"{type(e).__name__}: {e}"
        if verbose:
            with print_lock:
                print(f"[req-{request_id:02d}] error  {result.error}")
        return result

    if resp is None:
        result.end_time = time.perf_counter()
        result.error = "request failed"
        return result

    if resp.get("error"):
        result.end_time = resp.get("end_time", time.perf_counter())
        result.error = resp["error"]
        if verbose:
            with print_lock:
                print(f"[req-{request_id:02d}] error  {result.error}")
        return result

    result.first_token_time = resp.get("first_token_time")
    result.end_time = resp.get("end_time", time.perf_counter())
    result.stop_reason = resp.get("stop_reason")
    if client.messages and client.messages[-1]["role"] == "assistant":
        result.output_text = client.messages[-1]["content"]
    usage = resp.get("usage") or {}
    result.input_tokens = usage.get("input_tokens", 0)
    result.output_tokens = usage.get("output_tokens", 0)
    result.thinking_tokens = usage.get("thinking_tokens", 0)
    result.text_tokens = usage.get("text_tokens", 0)

    if verbose:
        with print_lock:
            ttft = (result.first_token_time - result.start_time) if result.first_token_time else float("nan")
            total = result.end_time - result.start_time
            print(f"[req-{request_id:02d}] done   ttft={ttft:.2f}s total={total:.2f}s "
                  f"out={result.output_tokens}")

    return result


def _fmt_prompt(p: str, width: int = 30) -> str:
    p = p.replace("\n", " ")
    return p if len(p) <= width else p[: width - 1] + "…"


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    # Linear interpolation on sorted values
    s = sorted(values)
    idx = q * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _render_summary(results: list[BenchResult], wall_time: float, concurrency: int) -> None:
    sep = f"{Colors.SEPARATOR}{'─' * 96}{Colors.RESET}"
    print(f"\n{sep}")
    print(f"  {Colors.TITLE}Bench results{Colors.RESET}  (concurrency={concurrency}, n={len(results)}, wall={wall_time:.2f}s)")
    print(sep)

    header = f"{'id':>3}  {'prompt':<30}  {'q_wait':>7}  {'ttft':>7}  {'total':>7}  {'in':>4}  {'out':>4}  {'tok/s':>6}  {'stop':<10}"
    print(header)
    print(f"{Colors.SEPARATOR}{'-' * 96}{Colors.RESET}")

    for r in sorted(results, key=lambda x: x.request_id):
        q_wait = (r.start_time - r.submit_time) if r.start_time else float("nan")
        if r.error:
            print(f"{r.request_id:>3}  {_fmt_prompt(r.prompt):<30}  "
                  f"{q_wait:>7.2f}  {'-':>7}  {'-':>7}  {'-':>4}  {'-':>4}  {'-':>6}  "
                  f"{Colors.GRAY}ERR: {r.error[:60]}{Colors.RESET}")
            continue
        ttft = (r.first_token_time - r.start_time) if r.first_token_time and r.start_time else float("nan")
        total = (r.end_time - r.start_time) if r.end_time and r.start_time else float("nan")
        gen_time = (r.end_time - r.first_token_time) if r.first_token_time and r.end_time else None
        tps = (r.output_tokens / gen_time) if gen_time and gen_time > 0 else float("nan")
        print(f"{r.request_id:>3}  {_fmt_prompt(r.prompt):<30}  "
              f"{q_wait:>7.2f}  {ttft:>7.2f}  {total:>7.2f}  "
              f"{r.input_tokens:>4}  {r.output_tokens:>4}  {tps:>6.1f}  "
              f"{r.stop_reason or '-':<10}")

    print(sep)

    ok = [r for r in results if not r.error and r.start_time and r.end_time]
    errors = [r for r in results if r.error]
    ttfts = [r.first_token_time - r.start_time for r in ok if r.first_token_time]
    totals = [r.end_time - r.start_time for r in ok]
    total_out = sum(r.output_tokens for r in ok)

    print(f"  ok: {len(ok)}   errors: {len(errors)}   wall: {wall_time:.2f}s")
    if ok:
        print(f"  ttft   p50={_percentile(ttfts, 0.50):.2f}s  p90={_percentile(ttfts, 0.90):.2f}s  "
              f"p99={_percentile(ttfts, 0.99):.2f}s  (n={len(ttfts)})")
        print(f"  total  p50={_percentile(totals, 0.50):.2f}s  p90={_percentile(totals, 0.90):.2f}s  "
              f"p99={_percentile(totals, 0.99):.2f}s")
        agg_tps = total_out / wall_time if wall_time > 0 else 0
        print(f"  throughput: {total_out} output tokens in {wall_time:.2f}s = {agg_tps:.1f} tok/s (aggregate)")
    print(f"{sep}\n")


def _write_output(path: Path, results: list[BenchResult], wall_time: float,
                  concurrency: int, url: str, max_tokens: int,
                  temperature: float, top_p: float) -> None:
    """Serialize full results (including generated text) as JSON."""
    payload = {
        "config": {
            "url": url,
            "concurrency": concurrency,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        },
        "wall_time": wall_time,
        "results": [asdict(r) for r in sorted(results, key=lambda x: x.request_id)],
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote bench output to {path}")


def run_bench(
    url: str,
    prompts_path: str,
    concurrency: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    verbose: bool,
    output_path: str | None = None,
) -> None:
    err = _health_check(url)
    if err:
        print(f"Cannot connect to server at {url}: {err}")
        return

    try:
        probe = ChatClient(base_url=url, quiet=True)
        model = probe.fetch_model()
    except Exception as e:
        print(f"Cannot discover model from {url}: {e}")
        return

    path = Path(prompts_path)
    if not path.exists():
        print(f"Prompts file not found: {prompts_path}")
        return
    prompts = _load_prompts(path)
    if not prompts:
        print(f"No prompts found in {prompts_path}")
        return

    print(f"Loaded {len(prompts)} prompt(s). Running with concurrency={concurrency}...")

    print_lock = threading.Lock()
    results: list[BenchResult] = []
    wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = []
        for i, prompt in enumerate(prompts):
            submit_time = time.perf_counter()
            fut = pool.submit(
                _run_one, i, prompt, submit_time, url,
                temperature, top_p, max_tokens, model, verbose, print_lock,
            )
            futures.append(fut)

        for fut in as_completed(futures):
            results.append(fut.result())

    wall_time = time.perf_counter() - wall_start
    _render_summary(results, wall_time, concurrency)

    if output_path:
        _write_output(Path(output_path), results, wall_time, concurrency,
                      url, max_tokens, temperature, top_p)
