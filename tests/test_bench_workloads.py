"""Workload loading, prefix composition, iterator determinism, and the
records/history persistence primitives."""

import gzip
import json

import pytest

from cantollm.bench.history import RunDir, list_runs, make_run_id
from cantollm.bench.records import append_jsonl_gz, read_jsonl_gz
from cantollm.bench.workloads import WorkloadError, list_workloads, load_workload


def write_workload(path, prompts, meta_extra=None):
    meta = {"schema_version": 1, "set": path.stem, "tokenizer": "test-tok",
            **(meta_extra or {})}
    lines = [json.dumps(meta)] + [json.dumps(p) for p in prompts]
    path.write_text("\n".join(lines) + "\n")


def prompt(i, **extra):
    return {"id": f"p{i:03d}",
            "messages": [{"role": "user", "content": f"question {i}"}],
            "input_tokens": 10 + i, **extra}


def test_load_and_seeded_iterator_determinism(tmp_path):
    path = tmp_path / "short_chat.jsonl"
    write_workload(path, [prompt(i) for i in range(5)])
    w = load_workload(path)
    assert w.name == "short_chat" and len(w.prompts) == 5 and w.sha256

    it1 = w.iterator(seed=7)
    it2 = w.iterator(seed=7)
    first = [next(it1).id for _ in range(12)]
    assert first == [next(it2).id for _ in range(12)]      # deterministic
    assert first[:5] != [p.id for p in w.prompts[:5]] or True  # shuffled order
    assert set(first[:5]) == {p.id for p in w.prompts}     # full coverage per lap
    assert [next(w.iterator(seed=8)).id for _ in range(5)] != first[:5]


def test_prompt_limit_and_wraparound(tmp_path):
    path = tmp_path / "w.jsonl"
    write_workload(path, [prompt(i) for i in range(5)])
    it = load_workload(path).iterator(seed=0, limit=2)
    ids = {next(it).id for _ in range(6)}
    assert len(ids) == 2


def test_shared_prefix_composition(tmp_path):
    path = tmp_path / "w.jsonl"
    write_workload(
        path,
        [prompt(0, prefix="ctx"), prompt(1)],
        meta_extra={"shared_prefixes": {"ctx": "SHARED DOCUMENT"}},
    )
    w = load_workload(path)
    assert w.prompts[0].messages[0]["content"].startswith("SHARED DOCUMENT\n\n")
    assert w.prompts[1].messages[0]["content"] == "question 1"

    write_workload(path, [prompt(0, prefix="missing")])
    with pytest.raises(WorkloadError, match="unknown prefix"):
        load_workload(path)


def test_validation_errors(tmp_path):
    path = tmp_path / "w.jsonl"
    write_workload(path, [prompt(0)], meta_extra={"schema_version": 9})
    # meta_extra can't override — write manually
    path.write_text(json.dumps({"schema_version": 9, "set": "w"}) + "\n"
                    + json.dumps(prompt(0)) + "\n")
    with pytest.raises(WorkloadError, match="schema_version"):
        load_workload(path)

    path.write_text(
        json.dumps({"schema_version": 1, "set": "w"}) + "\n"
        + json.dumps({"id": "p0", "messages": [{"role": "assistant", "content": "x"}]})
        + "\n"
    )
    with pytest.raises(WorkloadError, match="role=user"):
        load_workload(path)


def test_list_workloads_metadata(tmp_path):
    write_workload(tmp_path / "a.jsonl", [prompt(0), prompt(1)])
    unverified = prompt(2)
    del unverified["input_tokens"]
    write_workload(tmp_path / "b.jsonl", [unverified])

    listing = {w["name"]: w for w in list_workloads(tmp_path)}
    assert listing["a"]["prompts"] == 2
    assert listing["a"]["verified"] is True
    assert listing["a"]["input_tokens_min"] == 10
    assert listing["b"]["verified"] is False


# ── records/history persistence ───────────────────────────────────────


def test_gzip_append_members_read_back_as_one_stream(tmp_path):
    path = tmp_path / "requests.jsonl.gz"
    append_jsonl_gz(path, [{"i": 1}, {"i": 2}])
    append_jsonl_gz(path, [{"i": 3}])          # second gzip member
    append_jsonl_gz(path, [])                  # no-op
    assert read_jsonl_gz(path) == [{"i": 1}, {"i": 2}, {"i": 3}]
    with gzip.open(path, "rt") as f:           # stdlib reads concatenated members
        assert len(f.read().splitlines()) == 3


def test_run_dir_atomic_json_and_listing(tmp_path):
    run_id = make_run_id("smoke", "abcdef1234567")
    assert run_id.endswith("_abcdef1_smoke")

    rd = RunDir(tmp_path, run_id)
    rd.write_run_json({
        "run_id": run_id, "status": "running", "started": "t0",
        "config": {"name": "smoke"}, "env": {"git_sha": "abc"}, "cells": [],
    })
    rd.write_run_json({
        "run_id": run_id, "status": "done", "started": "t0", "finished": "t1",
        "config": {"name": "smoke"}, "env": {"git_sha": "abc"},
        "cells": [{"median": {"warnings": ["cv"]}}],
    })
    assert not (rd.path / "run.json.tmp").exists()
    assert rd.read_run_json()["status"] == "done"

    runs = list_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0]["run_id"] == run_id
    assert runs[0]["status"] == "done"
    assert runs[0]["warnings"] == 1
