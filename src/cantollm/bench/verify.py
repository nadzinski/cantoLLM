"""`canto bench verify-workloads`: stamp real token counts into prompt sets.

Re-tokenizes every prompt with the actual Qwen3 chat-template tokenization
(the same `encode_conversation` the API uses) and rewrites `input_tokens`
in place. Committed workload files are the pinned truth (bench-spec.md §1);
this is the stamping step that makes them trustworthy — rerun it after any
tokenizer or chat-template change and the git diff shows the drift.
"""

from __future__ import annotations

import json
from pathlib import Path

from cantollm.bench.workloads import DEFAULT_WORKLOADS_DIR


def verify_workloads(
    model_size: str = "0.6B",
    workloads_dir: Path | None = None,
) -> list[dict]:
    """Stamp `input_tokens` in every bench/workloads/*.jsonl. Returns a
    per-file report. Heavy imports stay local: only tokenizer files are
    downloaded (no weights)."""
    import cantollm.engine  # noqa: F401 — runtime↔engine cycle: engine must init first
    from cantollm.runtime import build_tokenizer_runtime
    from cantollm.spec import qwen3_spec

    tokenizer = build_tokenizer_runtime(qwen3_spec(model_size)).tokenizer
    base = workloads_dir or DEFAULT_WORKLOADS_DIR
    reports = []
    for path in sorted(base.glob("*.jsonl")):
        reports.append(_stamp_file(path, tokenizer))
    return reports


def _stamp_file(path: Path, tokenizer) -> dict:
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    meta = json.loads(lines[0])
    meta["tokenizer"] = getattr(tokenizer, "name", meta.get("tokenizer", "unknown"))
    prefixes = meta.get("shared_prefixes") or {}

    out_lines = [json.dumps(meta, separators=(",", ":"), ensure_ascii=False)]
    counts = []
    for line in lines[1:]:
        rec = json.loads(line)
        messages = [dict(m) for m in rec["messages"]]
        prefix_name = rec.get("prefix")
        if prefix_name:
            messages[0]["content"] = (
                prefixes[prefix_name] + "\n\n" + messages[0]["content"]
            )
        ids = tokenizer.encode_conversation(messages, system=rec.get("system"))
        rec["input_tokens"] = len(ids)
        counts.append(len(ids))
        out_lines.append(json.dumps(rec, separators=(",", ":"), ensure_ascii=False))

    path.write_text("\n".join(out_lines) + "\n")
    return {
        "file": str(path),
        "prompts": len(counts),
        "input_tokens_min": min(counts) if counts else None,
        "input_tokens_p50": sorted(counts)[len(counts) // 2] if counts else None,
        "input_tokens_max": max(counts) if counts else None,
    }
