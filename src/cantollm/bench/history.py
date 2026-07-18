"""bench/history/<run_id>/ persistence (bench-spec.md §6).

run.json is rewritten atomically after every repeat — it doubles as the
live/partial state the control panel polls and what crash recovery reads.
Request/step records append to gzip JSONL files beside it.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from cantollm.bench.records import RESULTS_SCHEMA_VERSION

DEFAULT_HISTORY_DIR = Path("bench/history")

REQUESTS_FILE = "requests.jsonl.gz"
STEPS_FILE = "engine_steps.jsonl.gz"
TEXT_FILE = "output_text.jsonl.gz"  # only with --capture-text; gitignored
LOGS_DIR = "logs"


def make_run_id(config_name: str, git_sha: str | None, now: datetime | None = None) -> str:
    stamp = (now or datetime.now()).strftime("%Y-%m-%dT%H%M%S")
    sha7 = (git_sha or "nogit")[:7]
    return f"{stamp}_{sha7}_{config_name}"


class RunDir:
    def __init__(self, history_dir: Path, run_id: str):
        self.run_id = run_id
        self.path = history_dir / run_id
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / LOGS_DIR).mkdir(exist_ok=True)

    @property
    def requests_path(self) -> Path:
        return self.path / REQUESTS_FILE

    @property
    def steps_path(self) -> Path:
        return self.path / STEPS_FILE

    @property
    def text_path(self) -> Path:
        return self.path / TEXT_FILE

    def server_log_path(self, spawn_index: int) -> Path:
        return self.path / LOGS_DIR / f"server-{spawn_index}.log"

    def write_run_json(self, payload: dict) -> None:
        """Atomic replace: pollers never observe a torn file."""
        payload = {"schema_version": RESULTS_SCHEMA_VERSION, **payload}
        tmp = self.path / "run.json.tmp"
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, self.path / "run.json")

    def read_run_json(self) -> dict:
        return json.loads((self.path / "run.json").read_text())


def list_runs(history_dir: Path | None = None) -> list[dict]:
    """History listing, newest first: each run.json's headline fields."""
    base = history_dir or DEFAULT_HISTORY_DIR
    if not base.exists():
        return []
    out = []
    for run_json in sorted(base.glob("*/run.json"), reverse=True):
        try:
            data = json.loads(run_json.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        out.append({
            "run_id": data.get("run_id", run_json.parent.name),
            "name": data.get("config", {}).get("name"),
            "status": data.get("status"),
            "started": data.get("started"),
            "finished": data.get("finished"),
            "env": data.get("env", {}),
            "n_cells": len(data.get("cells", [])),
            "warnings": sum(
                len(c.get("median", {}).get("warnings", []))
                for c in data.get("cells", [])
            ),
        })
    return out


def load_run(run_id: str, history_dir: Path | None = None) -> dict:
    base = history_dir or DEFAULT_HISTORY_DIR
    path = base / run_id / "run.json"
    if not path.exists():
        raise FileNotFoundError(f"no such run: {run_id}")
    return json.loads(path.read_text())
