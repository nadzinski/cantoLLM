"""§4 greedy equivalence: three manually spawned servers (default /
+compile dynamic / +compile batch-bucket), same seven greedy requests,
token streams must be identical across arms.

Prompt lengths span the batch buckets and kv spans (roughly 7 to ~1000
tokens). Sequential requests, temperature 0, ignore_eos, 64 max_tokens.
Records spawn->Ready per arm as a bonus (the §3 warm-bill cross-check).
"""
import json
import subprocess
import sys
import time
import urllib.request

PORT = 8378
BASE = f"http://127.0.0.1:{PORT}"
GEOMETRY = [
    "--engine", "batched", "--device", "cuda", "-m", "0.6B",
    "--max-batch", "16", "--batch-max-seq-len", "4096",
    "--max-tokens-per-step", "512", "--port", str(PORT),
]
ARMS = {
    "default": [],
    "compile-dynamic": ["--torch-compile", "--torch-compile-strategy", "dynamic"],
    "compile-batch-bucket": ["--torch-compile", "--torch-compile-strategy", "batch-bucket"],
}

WORD = "ossify "
PROMPTS = [
    "Hi",                                   # ~7 wrapped tokens
    "Name three colors.",
    WORD * 40,                              # ~60
    WORD * 120,                             # ~140
    WORD * 300,                             # ~330
    WORD * 600,                             # ~620
    WORD * 980,                             # ~1000
]


def wait_ready(proc, timeout=900):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server died (exit {proc.returncode})")
        try:
            with urllib.request.urlopen(f"{BASE}/health", timeout=2) as r:
                if r.status == 200:
                    return time.perf_counter() - t0
        except Exception:
            pass
        time.sleep(0.25)
    raise RuntimeError("server not ready in time")


def ask(prompt):
    body = json.dumps({
        "model": "qwen3-0.6B",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64,
        "temperature": 0.0,
        "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.load(r)
    msg = out["choices"][0]["message"]
    return (msg.get("reasoning_content") or "") + "\x00" + (msg.get("content") or "")


def run_arm(name, extra, log_path):
    log = open(log_path, "wb")
    proc = subprocess.Popen(
        [sys.executable, "-m", "cantollm.main", "serve", *GEOMETRY, *extra],
        stdout=log, stderr=subprocess.STDOUT,
    )
    try:
        ready_s = wait_ready(proc)
        print(f"[{name}] spawn->Ready {ready_s:.1f} s", flush=True)
        outs = []
        for i, p in enumerate(PROMPTS):
            outs.append(ask(p))
            print(f"[{name}] prompt {i} done ({len(outs[-1])} chars)", flush=True)
        return outs
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log.close()


def main(outdir):
    results = {}
    for name, extra in ARMS.items():
        results[name] = run_arm(name, extra, f"{outdir}/greedy_server_{name}.log")
        time.sleep(3)
    with open(f"{outdir}/greedy_outputs.json", "w") as f:
        json.dump(results, f, indent=1)
    base = results["default"]
    ok = True
    for name in ("compile-dynamic", "compile-batch-bucket"):
        for i, (a, b) in enumerate(zip(base, results[name])):
            if a != b:
                ok = False
                div = next(
                    (j for j, (x, y) in enumerate(zip(a, b)) if x != y),
                    min(len(a), len(b)),
                )
                print(f"DIVERGENCE arm={name} prompt={i} at char {div}:")
                print(f"  default : ...{a[max(0, div - 40):div + 80]!r}")
                print(f"  {name}: ...{b[max(0, div - 40):div + 80]!r}")
    print("IDENTICAL across all three arms" if ok else "STREAMS DIVERGED")


if __name__ == "__main__":
    main(sys.argv[1])
