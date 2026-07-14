"""Observability sidecar: provenance hashing, manifest, trace, and snapshot JSON.

Network (GCP metadata) and GPU (pynvml) collectors are monkeypatched to keep the tests
hermetic and fast; the snapshot recorder is driven with injected fake progress/serving
sources and PNG rendering off, so no matplotlib/pynvml is required.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.observability import provenance as prov
from src.observability import SnapshotRecorder, TraceRecorder, build_manifest, sha256_file, write_provenance


def test_sha256_file_stable_and_none_on_missing(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("hello", encoding="utf-8")
    h1 = sha256_file(str(f))
    assert h1 == sha256_file(str(f))                      # deterministic
    assert h1 == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    assert sha256_file(str(tmp_path / "missing.txt")) is None


def test_write_provenance_hashes_result_files(tmp_path: Path) -> None:
    (tmp_path / "b1").mkdir()
    (tmp_path / "b1" / "results.csv").write_text("example_id,f1\n1,1.0\n", encoding="utf-8")
    (tmp_path / "b1" / "metrics.json").write_text("{}", encoding="utf-8")
    (tmp_path / "b1" / "ignore.txt").write_text("nope", encoding="utf-8")
    out = tmp_path / "provenance.json"
    payload = write_provenance(str(tmp_path), str(out), created_at="2026-07-14T00:00:00+00:00")
    assert out.exists()
    assert payload["file_count"] == 2                      # results.csv + metrics.json, NOT ignore.txt
    files = payload["files"]
    assert any(k.endswith("results.csv") for k in files)
    assert all(v["sha256"] and v["size_bytes"] >= 0 for v in files.values())


def test_build_manifest_is_hermetic(tmp_path: Path, monkeypatch) -> None:
    # Avoid real network / GPU: patch the collectors build_manifest calls by name.
    monkeypatch.setattr(prov, "gcp_instance_metadata", lambda: {"on_gce": False})
    monkeypatch.setattr(prov, "gpu_info", lambda: {"name": None})
    monkeypatch.setattr(prov, "vllm_version", lambda: "0.11.0")
    m = build_manifest(
        run_id="r1", created_at="2026-07-14T00:00:00+00:00", cage_repo_dir=str(tmp_path),
        model="Qwen/Qwen3-8B", dataset="squad_v2", num_queries=500, num_trials=3, seed=42,
        kv_cache_dtype="fp8",
    )
    d = m.to_dict()
    assert d["run_id"] == "r1"
    assert d["model"] == "Qwen/Qwen3-8B" and d["kv_cache_dtype"] == "fp8"
    assert d["vllm_version"] == "0.11.0"
    assert d["seed"] == 42 and d["num_queries"] == 500


def test_trace_recorder_appends_jsonl(tmp_path: Path) -> None:
    tr = TraceRecorder(str(tmp_path / "trace.jsonl"))
    tr.event("observe_start", run_id="r1")
    tr.event("baseline_done", name="no_cache")
    lines = (tmp_path / "trace.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["kind"] == "observe_start" and rec["run_id"] == "r1" and "ts" in rec


def test_snapshot_recorder_writes_json_with_injected_sources(tmp_path: Path) -> None:
    rec = SnapshotRecorder(
        str(tmp_path),
        render_png=False,  # no matplotlib needed
        progress_fn=lambda: {"completed": 42, "baselines_done": 1, "active_baseline": "no_cache"},
        serving_fn=lambda: {"available": True, "prefix_cache_hit_rate": 0.9},
    )
    sample = rec.snapshot(label="baseline:no_cache")
    assert sample["label"] == "baseline:no_cache"
    assert sample["progress"]["completed"] == 42
    assert sample["serving"]["available"] is True
    assert set(sample["gpu"].keys()) >= {"mem_used_pct", "util_pct"}  # nulls off-GPU, but present
    assert (tmp_path / "snapshots" / "latest.json").exists()
    assert (tmp_path / "snapshots" / "snapshot_00001.json").exists()
    rec.stop(final=False)
