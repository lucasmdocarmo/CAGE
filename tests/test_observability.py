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


def test_instrument_versions_records_installed_and_missing() -> None:
    # scipy IS installed in the test venv -> exact version string; a nonsense
    # distribution -> None (fail-soft record, never a raise).
    from importlib import metadata

    got = prov.instrument_versions(["scipy", "cage-definitely-not-a-package"])
    assert got["scipy"] == metadata.version("scipy")
    assert got["cage-definitely-not-a-package"] is None


def test_instrument_versions_default_covers_scoring_stack() -> None:
    got = prov.instrument_versions()
    # Every declared scoring-stack package gets a key (value may be None off-GPU,
    # e.g. torch/transformers are absent from the slim analysis venv).
    assert set(got.keys()) == set(prov.SCORING_STACK_PACKAGES)
    for name in ("transformers", "sentence-transformers", "lettucedetect", "bert-score", "torch"):
        assert name in got                                # review §4.8 required set


def test_collect_dataset_fingerprints_hashes_and_fails_soft(tmp_path: Path) -> None:
    f = tmp_path / "query_manifest.json"
    f.write_text("hello", encoding="utf-8")
    fps = prov.collect_dataset_fingerprints(
        {"query_manifest": str(f), "missing_corpus": str(tmp_path / "nope.jsonl")}
    )
    qm = fps["query_manifest"]
    assert qm["sha256"] == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    assert qm["size_bytes"] == 5 and qm["path"] == str(f)
    miss = fps["missing_corpus"]
    assert miss["sha256"] is None and miss["size_bytes"] is None  # soft record, no raise


def test_build_manifest_scoring_provenance_additive(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(prov, "gcp_instance_metadata", lambda: {"on_gce": False})
    monkeypatch.setattr(prov, "gpu_info", lambda: {"name": None})
    monkeypatch.setattr(prov, "vllm_version", lambda: None)
    models = {
        "grounding": {"model_id": "KRLabsOrg/lettucedect-base-modernbert-en-v1", "revision": "abc123"},
        "serving_llm": {"model_id": "Qwen/Qwen3-8B", "revision": "def456"},
    }
    fps = {"query_manifest": {"path": "x", "sha256": None, "size_bytes": None}}
    m = build_manifest(
        run_id="r2", created_at="2026-08-04T00:00:00+00:00", cage_repo_dir=str(tmp_path),
        instrument_models=models, dataset_fingerprints=fps,
    )
    d = m.to_dict()
    # New keys present and faithful.
    assert d["instrument_models"] == models
    assert d["dataset_fingerprints"] == fps
    assert set(d["instrument_versions"].keys()) == set(prov.SCORING_STACK_PACKAGES)
    # Existing keys unbroken (backward compatibility of the manifest structure).
    for k in ("run_id", "created_at", "cage_git_sha", "vllm_version", "torch_version",
              "model", "dataset", "kv_cache_dtype", "gpu", "gcp_instance", "extra"):
        assert k in d


def test_build_manifest_defaults_keep_new_keys_empty(tmp_path: Path, monkeypatch) -> None:
    # Callers that predate the scoring-provenance fields get empty dicts, not errors.
    monkeypatch.setattr(prov, "gcp_instance_metadata", lambda: {"on_gce": False})
    monkeypatch.setattr(prov, "gpu_info", lambda: {"name": None})
    monkeypatch.setattr(prov, "vllm_version", lambda: None)
    d = build_manifest(
        run_id="r3", created_at="2026-08-04T00:00:00+00:00", cage_repo_dir=str(tmp_path),
    ).to_dict()
    assert d["instrument_models"] == {} and d["dataset_fingerprints"] == {}
    assert d["instrument_versions"]  # auto-collected, never empty


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
