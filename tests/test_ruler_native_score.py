"""ADR-0170 (owner 2026-10-13, "yes, do the native score fix before attempt 5"):
the RULER instrument is scored by its NATIVE containment match per task
(charter D5#5), never by the QA F1 of a 32K haystack.

Three seams, one test group each: the runner writes ``ruler_native_score``
and ``ruler_task`` on RULER rows (and nothing on other datasets' rows), the
verifier FAILs a RULER window whose served rows lack them, and the analysis
conditions RULER windows on the native metric with the task in the curve key.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "4_analysis"))

import test_run_experiment_campaign_layout as layout  # noqa: E402
import run_campaign_analysis as rca  # noqa: E402
import verify_results as vr  # noqa: E402
from src.analysis import conditioned_curves as cc  # noqa: E402
from src.data.loader import CAGExample  # noqa: E402

runner = layout.runner
TASK = "niah_multikey"
N = 4


class _WordTok:
    """Counts words: a chat render is the words of every message."""

    def apply_chat_template(self, messages, **kw):
        return [0] * sum(len(str(m.get("content", "")).split()) for m in messages)

    def encode(self, text, add_special_tokens=True):
        return [0] * len(text.split())


class _RulerLoader:
    """RULER items with the scoring contract (metadata task + gold_answers):
    even items' gold is a substring of the stub engine's answer ("stub answer
    for <id>"), odd items' gold is not, so the native scores alternate 1, 0."""

    def __init__(self, dataset: str, **_kw: Any) -> None:
        self.dataset = dataset

    def load(self, max_examples: Optional[int] = None) -> list:
        n = N if max_examples is None else min(N, max_examples)
        out = []
        for i in range(n):
            gold = "stub answer" if i % 2 == 0 else "nope-zzz"
            out.append(CAGExample(
                id=f"ruler_{TASK}_64_{i:04d}",
                question=f"What is the magic number for key{i}?",
                context=[f"haystack text {i} " + " ".join(f"w{k}" for k in range(20))],
                answer=gold,
                metadata={"task": TASK, "gold_answers": [gold], "needle_key": f"key{i}"},
            ))
        return out


def _patch_ruler_cell(monkeypatch: pytest.MonkeyPatch, root: Path, loader_cls: Any, out_dir: Path) -> None:
    """The stub seams of layout._run_cell for a RULER cell, with a get_loader
    stub that accepts the RULER call's token_counter keywords."""
    layout._campaign_env(monkeypatch, root)
    monkeypatch.setenv("CAGE_CAMPAIGN_DATASETS", "squad_v2 ruler")
    monkeypatch.setattr(runner, "CAMPAIGN_TELEMETRY_BACKENDS", frozenset())
    monkeypatch.setattr(
        runner, "setup_inference_engine",
        lambda model, cfg, *, backend, use_offline=False, strict=True: layout.StubEngine(model, "no_cache", 100.0),
    )
    monkeypatch.setattr(runner, "get_loader", lambda ds, split=None, seed=0, **kw: loader_cls(ds, **kw))
    monkeypatch.setattr(runner, "ruler_token_counter", lambda model: (lambda s: len(s.split()), "words", _WordTok()))
    monkeypatch.setattr(runner, "start_http_server", lambda *a, **k: None)
    monkeypatch.setattr(runner, "_safe_get_json", lambda url, timeout=5: None)
    monkeypatch.setattr(sys, "argv", [
        "run_experiment.py", "--baseline", "no_cache", "--baseline-label", "no_cache",
        "--model", "Qwen/Qwen3-14B", "--dataset", "ruler", "--num-queries", str(N), "--num-trials", "1",
        "--seed", "7", "--max-tokens", "16", "--api-base", "http://127.0.0.1:9", "--reranker-model", "none",
        "--output-dir", str(out_dir),
        "--ruler-task", TASK, "--ruler-context-tokens", "64", "--ruler-rendered-input-cap", "300",
    ])


def _run_ruler_cell(monkeypatch: pytest.MonkeyPatch, root: Path) -> Path:
    _patch_ruler_cell(monkeypatch, root, _RulerLoader, root / ".staging" / "test" / "no_cache" / "ruler")
    assert runner.main() is None
    (wdir,) = sorted(root.glob("cells/*/window_ruler-*"))
    return wdir


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TestRunnerWritesTheNativeScore:
    def test_ruler_rows_carry_the_native_score_and_task_and_the_window_its_mean(self, tmp_path, monkeypatch):
        wdir = _run_ruler_cell(monkeypatch, tmp_path / "camp" / "a" / layout.RUN_ID)
        rows = sorted(_rows(wdir / "requests.jsonl"), key=lambda r: r["example_id"])
        assert len(rows) == N
        for i, row in enumerate(rows):
            assert row["ruler_task"] == TASK
            assert row["ruler_native_score"] == (1.0 if i % 2 == 0 else 0.0)
            # the QA columns stay on the row; they are not RULER's quality
            assert "f1_score" in row
        evidence = sorted(_rows(wdir / "qa_evidence.jsonl"), key=lambda r: r["example_id"])
        for i, ev in enumerate(evidence):
            assert ev["ruler_task"] == TASK and ev["ruler_gold_answers"] == ["stub answer" if i % 2 == 0 else "nope-zzz"]
            assert ev["ruler_native_score"] == (1.0 if i % 2 == 0 else 0.0)
        meta = json.loads((wdir / "metrics.json").read_text(encoding="utf-8"))
        assert meta["quality"]["ruler_native_score"] == pytest.approx(0.5)
        assert meta["experiment"]["ruler_task"] == TASK
        # the verifier accepts this window (check (p) finds nothing)
        findings: list = []
        vr._check_window(wdir.parents[2], wdir, "ruler", findings, topology="single", engine="vllm")
        assert not [f for f in findings if f.check == "ruler-native"]

    def test_an_empty_generation_is_never_scored(self, tmp_path, monkeypatch):
        """Absent stays absent: a RULER row the engine answered with nothing
        carries no native score (not 0), is flagged empty_generation, and the
        window mean averages the scored rows only."""
        class _Blanks(layout.StubEngine):
            def _respond(self, request):
                response = super()._respond(request)
                if str(request.request_id).endswith(("1", "3")):  # the odd items
                    response.generated_text = ""
                return response

        root = tmp_path / "camp" / "a" / layout.RUN_ID
        _patch_ruler_cell(monkeypatch, root, _RulerLoader, root / ".staging" / "x")
        monkeypatch.setattr(
            runner, "setup_inference_engine",
            lambda model, cfg, *, backend, use_offline=False, strict=True: _Blanks(model, "no_cache", 100.0),
        )
        assert runner.main() is None
        (wdir,) = sorted(root.glob("cells/*/window_ruler-*"))
        rows = sorted(_rows(wdir / "requests.jsonl"), key=lambda r: r["example_id"])
        assert [r["empty_generation"] for r in rows] == [False, True, False, True]
        assert [r["ruler_native_score"] for r in rows] == [1.0, None, 1.0, None]
        assert all(r["ruler_task"] == TASK for r in rows)
        meta = json.loads((wdir / "metrics.json").read_text(encoding="utf-8"))
        assert meta["quality"]["ruler_native_score"] == pytest.approx(1.0)  # over the two scored rows
        findings: list = []
        vr._check_window(wdir.parents[2], wdir, "ruler", findings, topology="single", engine="vllm")
        assert not [f for f in findings if f.check == "ruler-native"]

    def test_other_datasets_rows_are_untouched(self, tmp_path, monkeypatch):
        root = tmp_path / "camp" / "a" / layout.RUN_ID
        layout._run_cell(monkeypatch, root, "no_cache", "squad_v2", ttft_base=200.0, num_trials=1)
        (wdir,) = layout._window_dirs(root, "no_cache", "squad_v2")
        for row in _rows(wdir / "requests.jsonl"):
            assert "ruler_native_score" not in row and "ruler_task" not in row
        meta = json.loads((wdir / "metrics.json").read_text(encoding="utf-8"))
        assert "ruler_native_score" not in meta["quality"] and "ruler_task" not in meta["experiment"]

    def test_an_item_without_the_contract_fails_the_cell(self, tmp_path, monkeypatch):
        class _NoContract(_RulerLoader):
            def load(self, max_examples=None):
                items = super().load(max_examples)
                for ex in items:
                    ex.metadata = {"needle_key": "k"}  # no task, no gold_answers
                return items

        root = tmp_path / "camp" / "a" / layout.RUN_ID
        _patch_ruler_cell(monkeypatch, root, _NoContract, root / ".staging" / "x")
        with pytest.raises(SystemExit) as exc:
            runner.main()
        assert exc.value.code == 1
        assert not list(root.glob("cells/*/window_ruler-*"))  # no window of unscored RULER rows


def _served(i: int, **extra: Any) -> dict[str, Any]:
    row = {"example_id": f"r{i}", "ok": True, "error": None, "empty_generation": False,
           "ruler_task": TASK, "ruler_native_score": 1.0 if i % 2 else 0.0}
    row.update(extra)
    return row


class TestVerifierCheckP:
    def test_served_rows_need_a_native_score_and_a_registered_task(self):
        findings: list = []
        vr._check_ruler_native_rows([_served(0), _served(1)], "w/requests.jsonl", findings)
        assert findings == []
        findings = []
        vr._check_ruler_native_rows(
            [_served(0, ruler_native_score=None), _served(1, ruler_native_score=1.5), _served(2, ruler_task="niah_single_typo")],
            "w/requests.jsonl", findings,
        )
        kinds = sorted((f.severity, f.check) for f in findings)
        assert kinds == [("FAIL", "ruler-native"), ("FAIL", "ruler-native")]
        texts = " | ".join(f.detail for f in findings)
        assert "2 of 3 served RULER row(s) carry no native score" in texts
        assert "1 of 3 served RULER row(s) carry no registered ruler_task" in texts
        # error and empty rows are outside the rule (absent stays absent)
        findings = []
        vr._check_ruler_native_rows(
            [_served(0, ok=False, error="boom", ruler_native_score=None),
             _served(1, empty_generation=True, ruler_native_score=None), _served(2)],
            "w/requests.jsonl", findings,
        )
        assert findings == []

    def test_the_check_runs_on_ruler_windows_only(self, tmp_path):
        from test_verify_results_v2 import _window_metrics, _write_jsonl  # noqa: E402

        for dataset in ("ruler", "squad_v2"):
            run_dir = tmp_path / dataset / "run"
            wdir = run_dir / "cells" / "gold-fresh|none|none|single|vllm|qwen3-14b|F2" / f"window_{dataset}-01"
            wdir.mkdir(parents=True)
            rows = [{"example_id": f"e{i}", "record_index": i, "ok": True, "error": None, "empty_generation": False,
                     "finish_reason": "stop", "num_tokens": 8, "tpot_ms": 12.0, "ttft_ms": 100.0} for i in range(3)]
            _write_jsonl(wdir / "requests.jsonl", rows)
            _write_jsonl(wdir / "qa_evidence.jsonl", rows)
            (wdir / "engine_metrics.json").write_text("{}", encoding="utf-8")
            _write_jsonl(wdir / "cage_stats.jsonl", [{"ts_s": 0.0}])
            (wdir / vr._WINDOW_METRICS_NAME).write_text(json.dumps(_window_metrics(n_offered=3)), encoding="utf-8")
            findings: list = []
            vr._check_window(run_dir, wdir, dataset, findings, topology="single", engine="vllm")
            native = [f for f in findings if f.check == "ruler-native"]
            if dataset == "ruler":
                assert len(native) == 2 and all(f.severity == "FAIL" for f in native)
            else:
                assert native == []


_KEY = "gold-fresh|none|none|single|vllm|qwen3-14b|F2|r0.5|lam0.85"


class TestAnalysisConditionsOnTheNativeMetric:
    def test_quality_metrics_for_and_the_task_in_the_curve_key(self):
        assert cc.quality_metrics_for("ruler", ("f1_score", "grounding_score")) == ("ruler_native_score",)
        assert cc.quality_metrics_for("qasper", ("f1_score", "grounding_score")) == ("f1_score", "grounding_score")
        frame = pd.DataFrame([
            {"row_key": _KEY, "dataset": "ruler", "window_key": "ruler-01", "rho_own": 0.3,
             "metric": "ruler_native_score", "value": 0.5, "n": 10, "task": "qa"},
            {"row_key": _KEY, "dataset": "ruler", "window_key": "ruler-04", "rho_own": 0.35,
             "metric": "ruler_native_score", "value": 0.9, "n": 10, "task": TASK},
            {"row_key": _KEY, "dataset": "qasper", "window_key": "qasper-01", "rho_own": 0.3,
             "metric": "f1_score", "value": 0.5, "n": 10, "task": None},
        ])
        curve = cc.quality_by_rho_own(frame)
        keys = sorted(curve["curve_key"])
        base = "gold-fresh|none|none|single|vllm|qwen3-14b|F2"
        # two RULER tasks in the same bin are two curves, never one mean
        assert keys == [base, f"{base}|task={TASK}", f"{base}|task=qa"]
        # without a task column nothing changes (the pre-ADR-0170 input shape)
        plain = cc.quality_by_rho_own(frame.drop(columns=["task"]).iloc[[2]])
        assert list(plain["curve_key"]) == [base]

    def test_the_pass_scores_ruler_windows_natively_per_task_and_others_as_before(self, tmp_path):
        run_dir = tmp_path / "run"
        analysis_dir = run_dir / "analysis" / "t0"
        windows = {
            "cells/c/window_ruler-01": ("ruler", [
                {"example_id": "a", "f1_score": 0.1, "ruler_native_score": 1.0, "ruler_task": "qa"},
                {"example_id": "b", "f1_score": 0.1, "ruler_native_score": 0.0, "ruler_task": "qa"},
            ]),
            "cells/c/window_ruler-04": ("ruler", [
                {"example_id": "a", "f1_score": 0.1, "ruler_native_score": 1.0, "ruler_task": "qa"},
                {"example_id": "b", "f1_score": 0.1, "ruler_native_score": 1.0, "ruler_task": TASK},  # two tasks: skip
            ]),
            "cells/c/window_qasper-01": ("qasper", [
                {"example_id": "a", "f1_score": 0.8}, {"example_id": "b", "f1_score": 0.6},
            ]),
        }
        index_rows = []
        for window, (dataset, rows) in windows.items():
            wdir = run_dir / window
            wdir.mkdir(parents=True)
            (wdir / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            own = analysis_dir / rca.OWN_ACCOUNTING_DIRNAME / window / rca.OWN_ACCOUNTING_NAME
            own.parent.mkdir(parents=True)
            own.write_text(json.dumps({"rho_own": 0.42}), encoding="utf-8")
            index_rows.append({"row_key": _KEY, "dataset": dataset, "window_key": window.rsplit("_", 1)[1],
                               "window_dir": window, "budget_r": 0.5, "rate_frac": 0.85})
        analysis_dir.mkdir(parents=True, exist_ok=True)
        doc = rca.run_conditioned_curves_pass(
            run_dir, pd.DataFrame(index_rows), analysis_dir, "DESIGN-INPUT-ONLY",
            blinding_active=False, metrics=("f1_score", "grounding_score"),
        )
        rho = doc["quality_vs_rho_own"]
        joined = {(w["window"], w["metric"]): w for w in rho["windows"]}
        # the RULER window conditions on its native score only (F1 never enters)
        assert ("cells/c/window_ruler-01", "ruler_native_score") in joined
        assert joined[("cells/c/window_ruler-01", "ruler_native_score")]["value"] == pytest.approx(0.5)
        assert ("cells/c/window_ruler-01", "f1_score") not in joined
        # the qasper window keeps the default metrics
        assert joined[("cells/c/window_qasper-01", "f1_score")]["value"] == pytest.approx(0.7)
        # the two-task RULER window is a named skip, never pooled
        reasons = " | ".join(s["reason"] for s in rho["skips"])
        assert "names 2 task(s)" in reasons and "cells/c/window_ruler-04" in {s["window"] for s in rho["skips"]}
        curve = pd.DataFrame(rho["curve"])
        base = "gold-fresh|none|none|single|vllm|qwen3-14b|F2"
        assert set(curve["curve_key"]) == {base, f"{base}|task=qa"}
        assert "per task" in rho["conditioning"]
        # review M1: the evidence-position curve carries no RULER trial (the
        # instrument has no position design and is never scored by F1 there)
        evidence = doc["quality_vs_evidence_position"]
        assert not [r for r in evidence["curve"] if r["dataset"] == "ruler"]
        assert any("RULER trial" in reason for reason in evidence["skipped_trials"])
        assert [r["dataset"] for r in evidence["curve"]] == ["qasper"] * len(evidence["curve"])
        # review L2: the document's metric list names the RULER metric it joined on
        assert "ruler_native_score" in doc["metrics"] and "f1_score" in doc["metrics"]
