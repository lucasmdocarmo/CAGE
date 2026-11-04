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


# ---------------------------------------------------------------------------
# S2 block F row F-11 (owner 2026-11-04, option (a)): the fields the predicate
# reads and every other field stay exact; faithfulness and completeness_bertscore
# may differ per row within the profile's stated bounds (0 = exact), which the
# master passes as --tol-faithfulness and --tol-bertscore and the verdict line keeps
# ---------------------------------------------------------------------------

S2_TOL = {"faithfulness": 0.005, "completeness_bertscore": 1e-5}
PREDICATE_INPUTS = ("grounding_score", "exact_match", "predicted_no_answer", "hallucination_detected")


def _s1_shaped_rows(n: int = 50, shifted_from: int = 10):
    """Fifty rows in the shape the S1 probe measured (experiments/S1/2026-10-23_2350,
    logs/engines/scoring_batch_probe.json: 10 identical, 40 with faithfulness moved by
    up to 0.0035 and completeness_bertscore by 2.2e-6, every other field equal)."""
    seq, bat = [], []
    for i in range(n):
        row = {"example_id": f"e{i}", "grounding_score": 1.0, "exact_match": 0.0, "predicted_no_answer": 0.0,
               "hallucination_detected": 0.0, "supported_claim_ratio": 1.0, "faithfulness": 0.9,
               "completeness_bertscore": 0.7, "abstained": False, "sanitized_answer": f"answer {i}"}
        seq.append(dict(row))
        if i >= shifted_from:
            sign = 1 if i % 2 else -1
            row = dict(row, faithfulness=0.9 + sign * 0.0035, completeness_bertscore=0.7 + sign * 2.2e-6)
        bat.append(row)
    return seq, bat


def test_the_s1_shaped_sample_passes_under_the_s2_bounds_and_fails_exact_or_tighter_f11() -> None:
    probe = _probe_module()
    seq, bat = _s1_shaped_rows()
    exact = probe.compare_rows(seq, bat)
    assert exact["ok"] is False and exact["identical"] == 10 and exact["within_tolerance"] == 0 and exact["tolerances"] == {}
    assert exact["first_difference"]["row"] == 10 and exact["first_difference"]["field"] == "completeness_bertscore"
    passed = probe.compare_rows(seq, bat, tolerances=S2_TOL)
    assert passed["ok"] is True and passed["identical"] == 10 and passed["within_tolerance"] == 40
    assert passed["first_difference"] is None and passed["tolerances"] == S2_TOL
    # the tolerated differences stay visible in the record (and on the OK line)
    assert passed["max_abs_numeric_diff"]["faithfulness"] == pytest.approx(0.0035)
    assert passed["max_abs_numeric_diff"]["completeness_bertscore"] == pytest.approx(2.2e-6)
    tighter = probe.compare_rows(seq, bat, tolerances=dict(S2_TOL, faithfulness=0.003))
    assert tighter["ok"] is False and tighter["within_tolerance"] == 0 and tighter["identical"] == 10
    assert tighter["first_difference"]["row"] == 10 and tighter["first_difference"]["field"] == "faithfulness"
    # a predicate input moved by 1e-9 on one row fails, whatever the bounds say
    moved = [dict(r) for r in bat]
    moved[20] = dict(moved[20], grounding_score=1.0 - 1e-9)
    flag = probe.compare_rows(seq, moved, tolerances=S2_TOL)
    assert flag["ok"] is False and flag["identical"] == 10 and flag["within_tolerance"] == 39
    assert flag["first_difference"] == {"row": 20, "example_id": "e20", "field": "grounding_score",
                                        "row_by_row": 1.0, "batched": 1.0 - 1e-9}


def test_a_tolerated_field_passes_only_as_two_finite_numbers_within_its_bound_f11() -> None:
    probe = _probe_module()
    nan = float("nan")

    def one(a_val, b_val, tol):
        a = [{"example_id": "e0", "faithfulness": a_val, "exact_match": 0.0}]
        b = [{"example_id": "e0", "faithfulness": b_val, "exact_match": 0.0}]
        return probe.compare_rows(a, b, tolerances=None if tol is None else {"faithfulness": tol})

    assert one(0.5, 0.504, 0.005)["ok"] is True
    assert one(0.5, 0.506, 0.005)["ok"] is False
    assert one(None, None, 0.005)["identical"] == 1           # neither path scored: identical, as today
    assert one(None, 0.5, 0.005)["ok"] is False                # one path scored, the other did not
    assert one(nan, nan, 0.005)["identical"] == 1              # equal as canonical text, as today
    assert one(nan, 0.5, 0.005)["ok"] is False
    assert one(True, False, 0.005)["ok"] is False              # a bool is never numeric
    assert one("0.9", 0.9, 0.005)["ok"] is False               # text is never numeric (review gap 3)
    assert one(-0.0, 0.0, 0.0)["within_tolerance"] == 1        # differ as text, difference 0: 0 = exact on the numbers
    assert one(1, 1.0, 0.0)["within_tolerance"] == 1
    assert one(1, 1.0, None)["ok"] is False                    # no bound: the text rule, as today
    # a key present on one side only is a difference, named, bound or not
    a = [{"example_id": "e0", "faithfulness": 0.5}]
    gone = probe.compare_rows(a, [{"example_id": "e0"}], tolerances=S2_TOL)
    assert gone["ok"] is False and gone["within_tolerance"] == 0 and gone["first_difference"]["field"] == "faithfulness"
    half = probe.compare_rows([{"example_id": "e0", "faithfulness": None}], [{"example_id": "e0"}], tolerances=S2_TOL)
    assert half["ok"] is False and half["first_difference"]["field"] == "faithfulness"


def test_a_bound_must_be_a_non_negative_finite_number_on_a_field_the_predicate_never_reads_f11() -> None:
    probe = _probe_module()
    rows = [{"example_id": "e0", "faithfulness": 0.5}]
    for bad in ({"faithfulness": -1e-3}, {"faithfulness": float("nan")}, {"faithfulness": float("inf")},
                {"faithfulness": "0.005"}, {"faithfulness": True}, {"faithfulness": None},
                {"faithfulness": -0.0}):                                  # review LOW 2: a signed zero is not a stated bound
        with pytest.raises(ValueError, match="faithfulness"):
            probe.compare_rows(rows, rows, tolerances=bad)
    for field in PREDICATE_INPUTS:
        with pytest.raises(ValueError, match=field):
            probe.compare_rows(rows, rows, tolerances={field: 0.001})
    assert probe.PREDICATE_INPUTS == PREDICATE_INPUTS


def test_the_verdict_lines_carry_the_bounds_and_the_counts_and_exact_mode_prints_todays_lines_f11() -> None:
    probe = _probe_module()
    ok = {"ok": True, "rows": 50, "batch_size": 32, "row_by_row_s_per_row": 0.31, "batched_s_per_row": 0.1, "speedup": 3.1,
          "window": "cells/k/window_qasper-01", "identical": 10, "within_tolerance": 40, "tolerances": S2_TOL,
          "max_abs_numeric_diff": {"faithfulness": 0.00353, "completeness_bertscore": 2.21e-6}}
    head = ("SCORING_BATCH_OK rows=50 batch_size=32 row_by_row_s_per_row=0.3100 batched_s_per_row=0.1000 "
            "speedup=3.10 window=cells/k/window_qasper-01")
    line = probe.verdict_line(ok)
    assert line.startswith(head)
    assert line.endswith(" identical=10 within_tol=40 tol=completeness_bertscore:1e-05,faithfulness:0.005"
                         " max_diff=completeness_bertscore:2.21e-06,faithfulness:0.00353")
    # exact mode (no bound given): the OK line is today's, byte for byte
    assert probe.verdict_line(dict(ok, identical=50, within_tolerance=0, tolerances={}, max_abs_numeric_diff={})) == head
    failed = {"ok": False, "rows": 50, "batch_size": 32, "identical": 10, "within_tolerance": 39, "tolerances": S2_TOL,
              "first_difference": {"row": 20, "example_id": "e20", "field": "grounding_score", "row_by_row": 1.0, "batched": 0.999999999},
              "max_abs_numeric_diff": {"faithfulness": 0.00353, "grounding_score": 1e-9}, "control_identical": True}
    line = probe.verdict_line(failed)
    assert line.startswith("SCORING_BATCH_FAILED rows=50 batch_size=32 reason=1 of 50 rows differ; first: row 20 (e20) "
                           "field grounding_score: 1.0 row by row vs 0.999999999 batched")
    assert line.endswith(" tol=completeness_bertscore:1e-05,faithfulness:0.005"
                         " max_diff=faithfulness:0.00353,grounding_score:1e-09 control_identical=yes")
    # the FAILED line of a refusal before any comparison carries the bounds it was given
    refused = {"ok": False, "batch_size": 32, "reason": "ValueError: x", "tolerances": {"faithfulness": 0.0}}
    assert probe.verdict_line(refused).endswith("reason=ValueError: x tol=faithfulness:0")


def test_the_full_mode_needs_both_bounds_and_a_bad_bound_is_a_verdict_f11(tmp_path: Path) -> None:
    run = _make_v2_tree(tmp_path)
    # review gap 2: the freeze file is named but absent, so a bound check that came after the
    # freeze read would surface as FileNotFoundError instead of naming the flag
    freeze = tmp_path / "absent_freeze.json"
    base = [sys.executable, str(PROBE), "--run-root", str(run), "--batch-size", "2", "--device", "cpu",
            "--scratch-dir", str(tmp_path / "s"), "--freeze-file", str(freeze)]

    def run_probe(*extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(base + list(extra), capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=120)

    proc = run_probe()
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "SCORING_BATCH_FAILED" in proc.stdout and "--tol-faithfulness is required with the full instruments" in proc.stdout
    assert "FileNotFoundError" not in proc.stdout and "Traceback" not in proc.stderr
    proc = run_probe("--tol-faithfulness", "0.005")
    assert proc.returncode == 1 and "--tol-bertscore is required with the full instruments" in proc.stdout, proc.stdout
    # the freeze file is still asked for first (the existing contract), the bounds right after it
    proc = subprocess.run(base[:-2] + ["--tol-faithfulness", "0.005", "--tol-bertscore", "1e-5"],
                          capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=120)
    assert proc.returncode == 1 and "--freeze-file is required" in proc.stdout
    # a negative bound, in the fast mode too, is a verdict naming the field, never a traceback
    proc = run_probe("--fast", "--tol-faithfulness", "-0.001", "--tol-bertscore", "1e-5")
    assert proc.returncode == 1 and "SCORING_BATCH_FAILED" in proc.stdout, proc.stdout + proc.stderr
    assert "faithfulness" in proc.stdout and "non-negative" in proc.stdout and "Traceback" not in proc.stderr
    # review MEDIUM 1: a negative in exponent form is a verdict in the --flag=value form the master
    # sends; the space form is argparse's own usage error (no verdict line), for a hand-typed call only
    proc = run_probe("--fast", "--tol-faithfulness", "0.005", "--tol-bertscore=-1e-5")
    assert proc.returncode == 1 and "SCORING_BATCH_FAILED" in proc.stdout, proc.stdout + proc.stderr
    assert "completeness_bertscore" in proc.stdout and "non-negative" in proc.stdout and "Traceback" not in proc.stderr
    proc = run_probe("--fast", "--tol-faithfulness", "0.005", "--tol-bertscore", "-1e-5")
    assert proc.returncode == 2 and "SCORING_BATCH_" not in proc.stdout and "expected one argument" in proc.stderr


def test_end_to_end_fast_passes_the_bounds_through_and_the_control_stays_exact_f11(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _probe_module()
    real = probe.load_rescorer()

    class _Shifted:
        """The rescorer with faithfulness 0.9 row by row and 0.904 batched on the first row
        (the fast evaluator leaves the field None, so both paths set it)."""

        def __getattr__(self, name: str) -> Any:
            return getattr(real, name)

        def _score_evidence_file(self, ev_path, evaluator, batch_size=None, **kw):
            rows, n = real._score_evidence_file(ev_path, evaluator, batch_size=batch_size, **kw)
            if rows:
                rows[0] = dict(rows[0], faithfulness=0.9 if batch_size is None else 0.904)
            return rows, n

    monkeypatch.setattr(probe, "load_rescorer", lambda: _Shifted())
    run = _make_v2_tree(tmp_path)
    out = tmp_path / "probe.json"
    argv = ["--run-root", str(run), "--batch-size", "2", "--rows", "3", "--fast", "--device", "cpu",
            "--scratch-dir", str(tmp_path / "s"), "--out", str(out)]
    code = probe.main(argv + ["--tol-faithfulness", "0.005", "--tol-bertscore", "1e-5"])
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0 and rec["ok"] is True and rec["identical"] == 2 and rec["within_tolerance"] == 1, rec
    assert rec["tolerances"] == {"faithfulness": 0.005, "completeness_bertscore": 1e-5}
    assert rec["max_abs_numeric_diff"] == {"faithfulness": pytest.approx(0.004)}
    # the control compares two row-by-row passes exactly, as before: nothing is tolerated there
    assert rec["control_identical"] is True and rec["control_max_abs_numeric_diff"] == {}
    line = probe.verdict_line(rec)
    assert line.startswith("SCORING_BATCH_OK rows=3 batch_size=2 ")
    assert " identical=2 within_tol=1 tol=completeness_bertscore:1e-05,faithfulness:0.005 max_diff=faithfulness:0.004" in line
    code = probe.main(argv + ["--tol-faithfulness", "0.003", "--tol-bertscore", "1e-5"])
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert code == 1 and rec["ok"] is False and rec["within_tolerance"] == 0 and rec["first_difference"]["field"] == "faithfulness"
    line = probe.verdict_line(rec)
    assert "reason=1 of 3 rows differ; first: row 0 " in line and " tol=completeness_bertscore:1e-05,faithfulness:0.003 max_diff=faithfulness:0.004 control_identical=yes" in line


def test_a_row_by_row_drift_inside_the_bound_still_reads_control_identical_no_f11(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Review gap 1: the control pass is compared exactly. A device whose row-by-row
    answers drift by 0.001 per call, inside the 0.005 bound, passes the proof (the batched
    row sits within the bound of the timed row-by-row row) and still reads
    control_identical=no with the drift on control_max_abs_numeric_diff."""
    probe = _probe_module()
    real = probe.load_rescorer()
    calls = {"seq": 0}

    class _Drifting:
        def __getattr__(self, name: str) -> Any:
            return getattr(real, name)

        def _score_evidence_file(self, ev_path, evaluator, batch_size=None, **kw):
            rows, n = real._score_evidence_file(ev_path, evaluator, batch_size=batch_size, **kw)
            if batch_size is None:
                value = 0.9 + 0.001 * calls["seq"]      # warm-up 0.900, timed 0.901, control 0.902
                calls["seq"] += 1
            else:
                value = 0.905                           # within 0.005 of the timed row-by-row 0.901
            if rows:
                rows[0] = dict(rows[0], faithfulness=value)
            return rows, n

    monkeypatch.setattr(probe, "load_rescorer", lambda: _Drifting())
    run = _make_v2_tree(tmp_path)
    out = tmp_path / "probe.json"
    code = probe.main(["--run-root", str(run), "--batch-size", "2", "--rows", "3", "--fast", "--device", "cpu",
                       "--scratch-dir", str(tmp_path / "s"), "--out", str(out),
                       "--tol-faithfulness", "0.005", "--tol-bertscore", "1e-5"])
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0 and rec["ok"] is True and rec["within_tolerance"] == 1, rec
    assert rec["control_identical"] is False
    assert rec["control_max_abs_numeric_diff"] == {"faithfulness": pytest.approx(0.001)}
    assert probe.verdict_line(rec).startswith("SCORING_BATCH_OK ")
