"""scripts/checks/probe_scoring_batch.py (ADR-0176 Group D, owner 2026-10-22,
step 1): the stage 10 proof that the batched scoring path grades exactly as the
row-by-row path. The comparison is tested on hand-built rows; the end-to-end
runs the real script on a tiny sealed tree with the model-free evaluator
(--fast: the plumbing, never the proof) and checks the verdict line, the exit
code, the record and that the run tree is never written. Each test fails on a
missing script or on a probe that writes into the run tree.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE = REPO_ROOT / "scripts" / "checks" / "probe_scoring_batch.py"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_rescore_wiring import EVIDENCE_ROWS, _make_v2_tree  # noqa: E402


def _probe_module() -> Any:
    spec = importlib.util.spec_from_file_location("probe_scoring_batch_under_test", PROBE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        h.update(str(p.relative_to(root)).encode()); h.update(p.read_bytes())
    return h.hexdigest()


def test_compare_rows_identical_and_the_first_difference_is_named() -> None:
    probe = _probe_module()
    a = [{"example_id": "e0", "f1_score": 1.0, "faithfulness": 0.5, "abstained": False},
         {"example_id": "e1", "f1_score": 0.0, "faithfulness": None, "abstained": True}]
    same = probe.compare_rows(a, [dict(r) for r in a])
    assert same["ok"] is True and same["identical"] == 2 and same["first_difference"] is None
    assert same["max_abs_numeric_diff"] == {}  # only fields that differ are listed
    b = [dict(a[0], faithfulness=0.5000001), dict(a[1])]
    diff = probe.compare_rows(a, b)
    assert diff["ok"] is False and diff["identical"] == 1
    assert diff["first_difference"] == {"row": 0, "example_id": "e0", "field": "faithfulness",
                                        "row_by_row": 0.5, "batched": 0.5000001}
    assert diff["max_abs_numeric_diff"]["faithfulness"] == pytest.approx(1e-7)
    short = probe.compare_rows(a, a[:1])
    assert short["ok"] is False and "2 rows row by row, 1 rows batched" in short["first_difference"]["reason"]
    # a bool is never a numeric difference
    assert "abstained" not in probe.compare_rows(a, a)["max_abs_numeric_diff"]


def test_verdict_lines() -> None:
    probe = _probe_module()
    ok = {"ok": True, "rows": 50, "batch_size": 32, "row_by_row_s_per_row": 0.31, "batched_s_per_row": 0.1,
          "speedup": 3.1, "window": "cells/k/window_qasper-01"}
    assert probe.verdict_line(ok).startswith("SCORING_BATCH_OK rows=50 batch_size=32 row_by_row_s_per_row=0.3100")
    failed = {"ok": False, "rows": 50, "batch_size": 32, "identical": 49,
              "first_difference": {"row": 3, "example_id": "e3", "field": "faithfulness", "row_by_row": 0.5, "batched": 0.6}}
    line = probe.verdict_line(failed)
    assert line.startswith("SCORING_BATCH_FAILED rows=50 batch_size=32 reason=1 of 50 rows differ; first: row 3 (e3) field faithfulness")
    assert probe.verdict_line({"ok": False, "batch_size": 32, "reason": "ValueError: x"}).endswith("reason=ValueError: x")
    # review 2026-10-23 L1 and M2: the largest differences and the control verdict ride the line
    with_diffs = dict(failed, max_abs_numeric_diff={"faithfulness": 9.6e-4, "completeness_bertscore": 9.5e-7}, control_identical=True)
    assert probe.verdict_line(with_diffs).endswith(" max_diff=completeness_bertscore:9.5e-07,faithfulness:0.00096 control_identical=yes")
    assert probe.verdict_line(dict(ok, speedup=None)).split("speedup=")[1].startswith("0.00")


def test_end_to_end_fast_on_a_sealed_tree_never_writes_the_run_tree(tmp_path: Path) -> None:
    run = _make_v2_tree(tmp_path)
    before = _tree_digest(run)
    out = tmp_path / "probe.json"
    scratch = tmp_path / "scratch"
    proc = subprocess.run(
        [sys.executable, str(PROBE), "--run-root", str(run), "--batch-size", "2", "--rows", "3", "--fast",
         "--device", "cpu", "--scratch-dir", str(scratch), "--out", str(out)],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    line = [l for l in proc.stdout.splitlines() if l.startswith("SCORING_BATCH_")][-1]
    assert line.startswith("SCORING_BATCH_OK rows=3 batch_size=2 ")
    assert "window=cells/B1/window_squad_v2-01" in line
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["ok"] is True and rec["identical"] == 3 and rec["rows"] == 3 and rec["mode"] == "fast"
    assert rec["row_by_row_s_per_row"] > 0 and rec["batched_s_per_row"] > 0
    assert _tree_digest(run) == before, "the probe wrote into the run tree"
    assert not (run / "scoring").exists()
    # review M2: the row-by-row control pass is recorded beside the verdict
    assert rec["control_identical"] is True and rec["control_max_abs_numeric_diff"] == {}
    # the one sample lives under the scratch root with the window's own names (M1: no separate warm-up row)
    samples = list(scratch.rglob("qa_evidence.jsonl"))
    assert len(samples) == 1 and "B1" in str(samples[0]) and "window_squad_v2-01" in str(samples[0])
    assert len(samples[0].read_text(encoding="utf-8").splitlines()) == 3 == min(3, len(EVIDENCE_ROWS))


def test_a_tree_without_completed_rows_is_refused_with_exit_2(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "cells" / "B1" / "window_squad_v2-01").mkdir(parents=True)
    (run / "cells" / "B1" / "window_squad_v2-01" / "qa_evidence.jsonl").write_text(
        json.dumps({"example_id": "x", "ok": False, "error": "drain cut"}) + "\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(PROBE), "--run-root", str(run), "--batch-size", "2", "--fast", "--device", "cpu",
         "--scratch-dir", str(tmp_path / "s")],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=120,
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "SCORING_BATCH_FAILED" in proc.stdout and "carries a completed row" in proc.stdout


def test_a_differing_batched_row_fails_with_exit_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _probe_module()
    real = probe.load_rescorer()

    class _Altered:
        """The rescorer with a batched path that returns one altered row."""

        def __getattr__(self, name: str) -> Any:
            return getattr(real, name)

        def _score_evidence_file(self, ev_path, evaluator, batch_size=None, **kw):
            rows, n = real._score_evidence_file(ev_path, evaluator, batch_size=batch_size, **kw)
            if batch_size is not None and rows:
                rows[0] = dict(rows[0], f1_score=0.123)
            return rows, n

    monkeypatch.setattr(probe, "load_rescorer", lambda: _Altered())
    run = _make_v2_tree(tmp_path)
    out = tmp_path / "probe.json"
    code = probe.main(["--run-root", str(run), "--batch-size", "2", "--rows", "3", "--fast", "--device", "cpu",
                       "--scratch-dir", str(tmp_path / "s"), "--out", str(out)])
    assert code == 1
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["ok"] is False and rec["first_difference"]["field"] == "f1_score" and rec["first_difference"]["batched"] == 0.123
    assert "SCORING_BATCH_FAILED" in probe.verdict_line(rec) and "field f1_score" in probe.verdict_line(rec)


def test_the_full_mode_needs_the_freeze_file_and_crashes_are_a_verdict(tmp_path: Path) -> None:
    run = _make_v2_tree(tmp_path)
    proc = subprocess.run(
        [sys.executable, str(PROBE), "--run-root", str(run), "--batch-size", "2", "--device", "cpu",
         "--scratch-dir", str(tmp_path / "s")],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=120,
    )
    assert proc.returncode == 1 and "SCORING_BATCH_FAILED" in proc.stdout and "--freeze-file is required" in proc.stdout
    assert "Traceback" not in proc.stderr
