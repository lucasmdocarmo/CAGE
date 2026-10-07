"""scripts/6_experiments/write_analysis.py (ADR-0143): the analysis.txt writer.

Design section 9 test 6: given a small fake landing with index/cells_index.csv,
a verify report, analysis/<stamp>/stats.json and summary.md, plan.json and
state.json, the script writes every section with the right numbers, the
MISSING annotation and the section 5 skeleton sentence; absent artifacts are
named absent, never guessed.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "6_experiments" / "write_analysis.py"


def _load():
    spec = importlib.util.spec_from_file_location("write_analysis_under_test", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


wa = _load()


@pytest.fixture()
def landing(tmp_path: Path) -> Path:
    land = tmp_path / "experiments" / "S1" / "2026-10-06"
    run = land / "run" / "camp1" / "a" / "20261006-100000-a-qwen3-14b"
    (land / "extras" / "verify").mkdir(parents=True)
    (land / "plots").mkdir()
    (land / "extras" / "state.json").write_text(json.dumps({
        "experiment": "S1", "date_utc": "2026-10-06", "run_id": run.name,
        "pod": {"id": "pod123", "dc": "US-NE-1", "price_per_hour_usd": "3.49", "created_utc": "2026-10-06T10:00:00Z", "deleted_utc": "2026-10-06T12:00:00Z"},
        "build": {"sha": "abc123"},
        "cost": {"balance_before_raw": '{"balance": 324.5}', "balance_after_raw": '{"balance": 317.5}', "true_zero_utc": "2026-10-06T12:01:00Z"},
        "go": [{"stage": "provision", "instant_utc": "2026-10-06T09:59:00Z", "words": "--yes provision on the command line"}],
        "stages": {"run": {"status": "passed", "notes": ["exit 2: STOP FAILED (ADR-0139); data tree complete"]}},
    }), encoding="utf-8")
    (land / "extras" / "plan.json").write_text(json.dumps({"schema": "cage-campaign-plan-v5", "steps": [
        {"kind": "relaunch", "engine": "vllm"}, {"kind": "cell", "engine": "vllm", "row_key": "k1"}, {"kind": "cell", "engine": "hf", "row_key": "k2"}],
        "blocked_row_keys": ["k9"]}), encoding="utf-8")
    (land / "extras" / "verify" / "verify_report.md").write_text("# verify\n1 FAIL, 1 WARN\nFAIL: cell k3 missing ledger entry\nWARN: a mini manifest\n", encoding="utf-8")
    (run / "index").mkdir(parents=True)
    (run / "index" / "cells_index.csv").write_text(
        "row_key,engine,model,dataset,family,window\n"
        "k1,vllm,qwen3-14b,squad_v2,F1,window_squad_v2-01\n"
        "k1,vllm,qwen3-14b,squad_v2,F1,window_squad_v2-02\n"
        "k2,hf,qwen3-14b,squad_v2,F1,window_squad_v2-01\n", encoding="utf-8")
    (run / "index" / "coverage_report.md").write_text("# coverage\nMISSING: B3 vllm\nMISSING: B5 sglang\n", encoding="utf-8")
    stamp = run / "analysis" / "20261006-120000"
    stamp.mkdir(parents=True)
    (stamp / "stats.json").write_text(json.dumps({"mode": "DESIGN-INPUT-ONLY",
        "contrasts": [{"id": 4, "metric": "ttft_ms", "n": 50, "estimate": -12.5}, {"id": 1, "metric": "ttft_ms", "n": 50}],
        "figures": [{"kind": "forest", "file": "forest_ttft_ms.png"}, {"kind": "wlt", "file": "wlt_ttft_ms.png"}]}), encoding="utf-8")
    (stamp / "summary.md").write_text("# summary\ncontrast 4: ttft_ms, estimate -12.5\n", encoding="utf-8")
    (run / "predicate" / "full-cuda-1").mkdir(parents=True)
    (run / "predicate" / "full-cuda-1" / "predicate_manifest.json").write_text(json.dumps({
        "schema_version": 1, "scoring_run_id": "full-cuda-1",
        "config": {"max_null_fraction": 1.0, "qasper_tau": 0.9955},
        "counts": {"n_rows": 100, "n_true": 60, "n_false": 10, "n_null": 30, "null_fraction": 0.3, "n_windows": 2},
        "per_window": [
            {"window": "k1/window_squad_v2-01", "n_rows": 50, "n_true": 45, "n_false": 2, "n_null": 3, "null_fraction": 0.06},
            {"window": "k2/window_squad_v2-01", "n_rows": 50, "n_true": 15, "n_false": 8, "n_null": 27, "null_fraction": 0.54},
        ],
        "skipped_windows": [],
    }), encoding="utf-8")
    (run / "observability" / "serving_configs").mkdir(parents=True)
    (run / "observability" / "serving_configs" / "x_vllm.json").write_text(json.dumps({"engine": "vllm", "gpu_memory_utilization": 0.9, "kv_pool_bytes_realized": 5713920000, "kv_pool_tokens_realized": 38736, "kv_pool_captured_utc": "2026-10-06T10:30:00Z"}), encoding="utf-8")
    return land


def _run_root(land: Path) -> Path:
    return land / "run" / "camp1" / "a" / "20261006-100000-a-qwen3-14b"


def test_every_section_with_the_right_numbers(landing: Path) -> None:
    out = landing / "plots" / "analysis.txt"
    assert wa.main(["--landing", str(landing), "--run-root", str(_run_root(landing)), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    for h in ("1. HEADER", "2. WHAT RAN", "3. TECHNICAL READING", "4. ANALYTIC READING", "5. ACADEMIC READING (SKELETON)", "6. LIMITATIONS", "7. INDEX OF ARTIFACTS"):
        assert h in text, h
    assert "experiment: S1   date: 2026-10-06   run id: 20261006-100000-a-qwen3-14b" in text
    assert "pod: id=pod123 dc=US-NE-1 price_per_hour_usd=3.49" in text
    assert "owner GO: provision at 2026-10-06T09:59:00Z (--yes provision on the command line)" in text
    assert "verify_results: 1 FAIL line(s), 1 WARN line(s)" in text   # the "1 FAIL, 1 WARN" summary line is not a verdict
    assert "organize coverage: 2 MISSING line(s)" in text and "charter F1 arm floor" in text
    assert "not failures of this run" not in text                      # no untagged interpretation (review LOW 14)
    assert "analysis: stamp 20261006-120000, mode DESIGN-INPUT-ONLY, 2 figure record(s)" in text
    assert "rows (cell x window): 3   cells: 2" in text
    assert "k1: 2 window(s)" in text and "k2: 1 window(s)" in text
    assert "by engine: hf=1, vllm=2" in text
    assert "plan: schema cage-campaign-plan-v5, 3 steps (2 cells, 1 relaunches), 1 blocked" in text
    assert "contrast rows in stats.json: 2" in text and "id=4  metric=ttft_ms  n=50  estimate=-12.5" in text
    assert "kv_pool_bytes_realized=5713920000 tokens=38736" in text
    # the per-window ungraded share (owner, 2026-10-06), thinnest window first, flagged past one half
    assert "predicate full-cuda-1: bound max_null_fraction=1.0 (1.0 = the stop never fires), rows=100 graded true=60 false=10 ungraded=30 (0.3) windows=2 skipped=0" in text
    k2 = text.index("window k2/window_squad_v2-01: ungraded 27 of 50 (54.0%); graded true=15 false=8   <- more than half of this window has no grade")
    k1 = text.index("window k1/window_squad_v2-01: ungraded 3 of 50 (6.0%); graded true=45 false=2")
    assert k2 < k1
    assert "FAIL: cell k3 missing ledger entry" in text
    assert "contrast 4: ttft_ms, estimate -12.5" in text          # summary.md verbatim
    assert "registered contrasts present in this run: 4, 1" in text
    assert text.count(wa.SKELETON_SENTENCE) >= 4
    assert "stage notes: run: exit 2: STOP FAILED" in text
    assert "n (rows) per engine: hf=1, vllm=2" in text
    assert "extras/state.json" in text and "total files:" in text
    assert chr(0x2014) not in text
    for banned in ("suggests", "shows that", "demonstrates", "proves"):
        assert banned not in text.lower(), banned


def test_absent_artifacts_are_named_absent_not_guessed(tmp_path: Path) -> None:
    land = tmp_path / "S1" / "2026-10-06"
    land.mkdir(parents=True)
    out = land / "plots" / "analysis.txt"
    assert wa.main(["--landing", str(land), "--run-root", str(land / "run" / "nope"), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "verify_results: absent" in text and "organize coverage: absent" in text
    assert "analysis: absent" in text and "cells index: absent or empty" in text
    assert "plan: absent" in text and "stats.json: absent" in text and "summary.md: absent" in text
    assert "predicate: absent" in text
    assert "registered contrasts present in this run: none" in text
    assert wa.SKELETON_SENTENCE in text


def test_missing_landing_is_refused() -> None:
    assert wa.main(["--landing", "/nonexistent/landing", "--run-root", "/x", "--out", "/tmp/never.txt"]) == 2
