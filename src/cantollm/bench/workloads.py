"""Prompt-set loading (bench-spec.md §1).

A workload file is JSONL under bench/workloads/: line 1 a meta record
(schema_version, set name, tokenizer id, optional shared_prefixes), then
one record per prompt (id, messages in the OpenAI shape, optional system,
optional prefix name, verified input_tokens, tags). Committed files are the
pinned truth; `canto bench verify-workloads` stamps the token counts.

The shared-prefix knob is dormant in v1: v1 sets carry `prefix: null`
everywhere, but the loader already composes prefixes so a Phase-5 variant
is a data change, not a code change.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

WORKLOAD_SCHEMA_VERSION = 1

DEFAULT_WORKLOADS_DIR = Path("bench/workloads")


class WorkloadError(ValueError):
    pass


@dataclass(frozen=True)
class Prompt:
    id: str
    messages: tuple[dict, ...]   # OpenAI-shape, prefix already composed
    system: str | None
    input_tokens: int | None     # verified count; None until stamped
    tags: tuple[str, ...] = ()


@dataclass
class Workload:
    name: str
    tokenizer: str
    prompts: list[Prompt]
    path: Path | None = None
    sha256: str | None = None    # stamped by the loader; joins comparisons

    def iterator(self, seed: int, limit: int | None = None):
        """Endless seeded round-robin over a shuffled order — closed-loop
        workers and the open-loop scheduler both draw from this."""
        prompts = self.prompts[:limit] if limit else list(self.prompts)
        if not prompts:
            raise WorkloadError(f"workload {self.name} has no prompts")
        order = list(prompts)
        random.Random(seed).shuffle(order)
        i = 0
        while True:
            yield order[i % len(order)]
            i += 1


def load_workload(name_or_path: str | Path, base_dir: Path | None = None) -> Workload:
    path = Path(name_or_path)
    if not path.suffix:
        path = (base_dir or DEFAULT_WORKLOADS_DIR) / f"{path.name}.jsonl"
    if not path.exists():
        raise WorkloadError(f"workload file not found: {path}")

    raw = path.read_bytes()
    lines = [ln for ln in raw.decode().splitlines() if ln.strip()]
    if not lines:
        raise WorkloadError(f"workload file is empty: {path}")

    meta = json.loads(lines[0])
    if meta.get("schema_version") != WORKLOAD_SCHEMA_VERSION:
        raise WorkloadError(
            f"{path}: workload schema_version {meta.get('schema_version')!r} "
            f"!= supported {WORKLOAD_SCHEMA_VERSION}"
        )
    prefixes: dict[str, str] = meta.get("shared_prefixes") or {}

    prompts: list[Prompt] = []
    for i, line in enumerate(lines[1:], start=2):
        rec = json.loads(line)
        try:
            messages = rec["messages"]
            prompt_id = rec["id"]
        except KeyError as e:
            raise WorkloadError(f"{path}:{i} missing field {e}")
        if not messages or messages[-1].get("role") != "user":
            raise WorkloadError(f"{path}:{i} last message must be role=user")

        prefix_name = rec.get("prefix")
        if prefix_name is not None:
            if prefix_name not in prefixes:
                raise WorkloadError(f"{path}:{i} unknown prefix {prefix_name!r}")
            first = dict(messages[0])
            first["content"] = prefixes[prefix_name] + "\n\n" + first["content"]
            messages = [first, *messages[1:]]

        prompts.append(Prompt(
            id=prompt_id,
            messages=tuple(messages),
            system=rec.get("system"),
            input_tokens=rec.get("input_tokens"),
            tags=tuple(rec.get("tags", ())),
        ))

    import hashlib
    return Workload(
        name=meta.get("set", path.stem),
        tokenizer=meta.get("tokenizer", "unknown"),
        prompts=prompts,
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def list_workloads(base_dir: Path | None = None) -> list[dict]:
    """Metadata for the control panel's workload listing."""
    base = base_dir or DEFAULT_WORKLOADS_DIR
    out = []
    for path in sorted(base.glob("*.jsonl")):
        try:
            w = load_workload(path)
        except (WorkloadError, json.JSONDecodeError) as e:
            out.append({"name": path.stem, "error": str(e)})
            continue
        counted = [p.input_tokens for p in w.prompts if p.input_tokens is not None]
        out.append({
            "name": w.name,
            "path": str(path),
            "prompts": len(w.prompts),
            "tokenizer": w.tokenizer,
            "sha256": w.sha256,
            "input_tokens_min": min(counted) if counted else None,
            "input_tokens_max": max(counted) if counted else None,
            "verified": len(counted) == len(w.prompts),
        })
    return out
