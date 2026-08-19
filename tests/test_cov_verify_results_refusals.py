"""Refusal-arm coverage for scripts/4_analysis/verify_results.py (K-COV8, #142).

test_verify_results_v2.py pins the green path, reconciliation, duplicates,
windows[] coverage, sealing and --out placement; the REFUSAL arms below had
never been exercised offline:

- verify_run on a nonexistent run root (VerifyRefusal)
- layout arms: cells/ missing, stray file in cells/, unparseable cell dir,
  missing / invalid / non-object cell.json, non-window entries, window
  identity collisions (H12), unknown dataset ids, the empty-run refusal
- schema arms: malformed JSONL lines, non-object rows, missing example_id,
  missing engine_metrics.json / cage_stats.jsonl, invalid engine_metrics
- the ShareGPT qa_evidence exemption (serving streams only — no FAIL)
- windows[] entry arms: non-object entry, dataset mismatch, WARN fields
- ledger arms: corrupt ledger (LedgerError), manifest.json-not-sealed WARN
- render_markdown ledger phrasing arms; pilot-mode usage refusals

Synthetic §1 trees in tmp_path only; the pilot archive under results/ is
never touched. Offline, seconds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = REPO_ROOT / "scripts" / "4_analysis"
for _p in (str(_SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import verify_results as vr  # noqa: E402
from src.analysis.cellspec import CellSpec  # noqa: E402
from src.analysis.stats.ledger import hash_artifacts, write_ledger  # noqa: E402

RUN_ID = "20260818-1200-a-qwen3-14b"
MODEL = "qwen3-14b"
DATASET = "squad_v2"
BASELINE = "B3"


def _manifest() -> dict[str, Any]:
    return {
        "campaign": "camp1",
        "session": "a",
        "run_id": RUN_ID,
        "model": MODEL,
        "git_sha": "deadbeef",
        "git_dirty": False,
        "engine": "vllm",
        "engine_version": "0.19.1",
        "seed": 1,
        "provider": "gcp",
        "hardware": "a2-ultragpu-1g x1",
        "dataset_manifests_sha256": "0" * 64,
        "cellspec_schema_version": 1,
        "created_utc": "2026-08-18T12:00:00+00:00",
    }


def _row(i: int) -> dict[str, Any]:
    return {
        "example_id": f"e{i}",
        "repeat_index": 0,
        "record_index": i,
        "ok": True,
        "error": None,
        "empty_generation": False,
        "ttft_ms": 100.0 + i,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _fill_window(wdir: Path, n_rows: int = 2, *, evidence: bool = True) -> None:
    rows = [_row(i) for i in range(n_rows)]
    _write_jsonl(wdir / "requests.jsonl", rows)
    if evidence:
        _write_jsonl(wdir / "qa_evidence.jsonl", rows)
    (wdir / "engine_metrics.json").write_text(json.dumps({"snapshot": "x"}), encoding="utf-8")
    _write_jsonl(wdir / "cage_stats.jsonl", [{"ts_s": 0.0, "kv_cache_usage": 0.1}])


def _cell_dir(run_dir: Path) -> Path:
    spec = CellSpec.from_baseline(BASELINE, model=MODEL)  # type: ignore[arg-type]
    return run_dir / "cells" / spec.to_row_key()


def _build_tree(tmp_path: Path, *, dataset: str = DATASET, evidence: bool = True) -> Path:
    """One-cell, one-window §1 tree (UNSEALED; seal with _seal)."""
    run_dir = tmp_path / "results" / "camp1" / "a" / RUN_ID
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps(_manifest()), encoding="utf-8")
    cell_dir = _cell_dir(run_dir)
    cell_dir.mkdir(parents=True)
    key = f"{dataset}-01"
    wdir = cell_dir / f"window_{key}"
    wdir.mkdir()
    _fill_window(wdir, evidence=evidence)
    (cell_dir / "cell.json").write_text(
        json.dumps({
            "baseline": BASELINE,
            "windows": {
                key: {"dataset": dataset, "seed": 1, "rep": 1,
                      "t_start": 0.0, "t_end": 60.0},
            },
        }),
        encoding="utf-8",
    )
    return run_dir


def _seal(run_dir: Path) -> None:
    sealed = [
        p for p in sorted(run_dir.rglob("*")) if p.is_file() and p.name != "ledger.json"
    ]
    write_ledger(hash_artifacts(sealed, base_dir=run_dir), run_dir / "ledger.json")


def _findings(report: dict[str, Any], severity: str, check: str | None = None):
    return [
        f for f in report["findings"]
        if f["severity"] == severity and (check is None or f["check"] == check)
    ]


def _details(report: dict[str, Any], severity: str, check: str | None = None) -> str:
    return " | ".join(f["detail"] for f in _findings(report, severity, check))


# --------------------------------------------------------------------------- #
# Usage-level refusals
# --------------------------------------------------------------------------- #


class TestUsageRefusals:
    def test_nonexistent_run_dir_is_refused(self, tmp_path: Path):
        with pytest.raises(vr.VerifyRefusal, match="does not exist"):
            vr.verify_run(tmp_path / "nope")

    def test_out_equal_to_run_root_is_refused(self, tmp_path: Path):
        run_dir = _build_tree(tmp_path)
        with pytest.raises(vr.VerifyRefusal, match="OUTSIDE"):
            vr.resolve_out_dir(run_dir, run_dir)

    def test_out_nested_inside_run_root_is_refused(self, tmp_path: Path):
        run_dir = _build_tree(tmp_path)
        with pytest.raises(vr.VerifyRefusal, match="OUTSIDE"):
            vr.resolve_out_dir(run_dir, run_dir / "cells" / "deep" / "reports")

    def test_default_out_is_the_sibling_verification_dir(self, tmp_path: Path):
        run_dir = _build_tree(tmp_path)
        out = vr.resolve_out_dir(run_dir, None)
        assert out == run_dir.parent / f"{RUN_ID}_verification"

    def test_cli_refusal_exits_2(self, tmp_path: Path, capsys):
        rc = vr.main([str(tmp_path / "missing-run")])
        assert rc == 2
        assert "ERROR" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Layout refusal arms
# --------------------------------------------------------------------------- #


class TestLayoutArms:
    def test_missing_cells_dir_fails(self, tmp_path: Path):
        run_dir = _build_tree(tmp_path)
        import shutil

        shutil.rmtree(run_dir / "cells")
        _seal(run_dir)
        report = vr.verify_run(run_dir)
        assert not report["ok"]
        assert "cells/ directory missing" in _details(report, "FAIL", "layout")

    def test_stray_file_in_cells_fails(self, tmp_path: Path):
        run_dir = _build_tree(tmp_path)
        (run_dir / "cells" / "notes.txt").write_text("stray", encoding="utf-8")
        _seal(run_dir)
        report = vr.verify_run(run_dir)
        assert "stray file in cells/" in _details(report, "FAIL", "layout")

    def test_unparseable_cell_dir_name_fails(self, tmp_path: Path):
        run_dir = _build_tree(tmp_path)
        bogus = run_dir / "cells" / "not-a-row-key"
        bogus.mkdir()
        _seal(run_dir)
        report = vr.verify_run(run_dir)
        assert not report["ok"]
        assert any(
            "not-a-row-key" in f["where"] for f in _findings(report, "FAIL", "layout")
        )

    def test_missing_cell_json_fails(self, tmp_path: Path):
        run_dir = _build_tree(tmp_path)
        (_cell_dir(run_dir) / "cell.json").unlink()
        _seal(run_dir)
        report = vr.verify_run(run_dir)
        assert "missing cell.json" in _details(report, "FAIL", "layout")

    def test_invalid_cell_json_fails(self, tmp_path: Path):
        run_dir = _build_tree(tmp_path)
        (_cell_dir(run_dir) / "cell.json").write_text("{TRUNC", encoding="utf-8")
        _seal(run_dir)
        report = vr.verify_run(run_dir)
        assert "invalid JSON" in _details(report, "FAIL", "layout")

    def test_non_object_cell_json_fails(self, tmp_path: Path):
        run_dir = _build_tree(tmp_path)
        (_cell_dir(run_dir) / "cell.json").write_text("[1, 2]", encoding="utf-8")
        _seal(run_dir)
        report = vr.verify_run(run_dir)
        assert "root must be an object" in _details(report, "FAIL", "layout")

    def test_non_window_entry_in_cell_fails(self, tmp_path: Path):
        run_dir = _build_tree(tmp_path)
        (_cell_dir(run_dir) / "scratch").mkdir()
        _seal(run_dir)
        report = vr.verify_run(run_dir)
        assert "expected a window_<dataset>-<ordinal>" in _details(
            report, "FAIL", "layout"
        )

    def test_window_identity_collision_fails_h12(self, tmp_path: Path):
        # window_squad_v2-1 and window_squad_v2-01 parse to the SAME
        # (dataset, ordinal) identity: the §8 join key would be ambiguous.
        run_dir = _build_tree(tmp_path)
        twin = _cell_dir(run_dir) / f"window_{DATASET}-1"
        twin.mkdir()
        _fill_window(twin)
        _seal(run_dir)
        report = vr.verify_run(run_dir)
        assert "collides with sibling" in _details(report, "FAIL", "layout")

    def test_unknown_dataset_id_fails(self, tmp_path: Path):
        run_dir = _build_tree(tmp_path)
        alien = _cell_dir(run_dir) / "window_tinystories-01"
        alien.mkdir()
        _fill_window(alien)
        _seal(run_dir)
        report = vr.verify_run(run_dir)
        assert "not a §1 dataset id" in _details(report, "FAIL", "layout")

    def test_empty_run_verifies_nothing(self, tmp_path: Path):
        run_dir = _build_tree(tmp_path)
        import shutil

        shutil.rmtree(_cell_dir(run_dir) / f"window_{DATASET}-01")
        (_cell_dir(run_dir) / "cell.json").write_text(
            json.dumps({"baseline": BASELINE, "windows": {}}), encoding="utf-8"
        )
        _seal(run_dir)
        report = vr.verify_run(run_dir)
        assert "an empty run verifies nothing" in _details(report, "FAIL", "layout")


# --------------------------------------------------------------------------- #
# Per-window schema refusal arms
# --------------------------------------------------------------------------- #


def _window(run_dir: Path) -> Path:
    return _cell_dir(run_dir) / f"window_{DATASET}-01"


class TestSchemaArms:
    def test_malformed_jsonl_line_fails_with_line_number(self, tmp_path: Path):
        run_dir = _build_tree(tmp_path)
        path = _window(run_dir) / "requests.jsonl"
        path.write_text(
            json.dumps(_row(0)) + "\n{BROKEN\n" + json.dumps(_row(1)) + "\n",
            encoding="utf-8",
        )
        _seal(run_dir)
        report = vr.verify_run(run_dir)
        fails = _findings(report, "FAIL", "schema")
        assert any("invalid JSON" in f["detail"] and f["where"].endswith(":2")
                   for f in fails)

    def test_non_object_row_fails(self, tmp_path: Path):
        run_dir = _build_tree(tmp_path)
        path = _window(run_dir) / "requests.jsonl"
        path.write_text('["a", "list"]\n' + json.dumps(_row(0)) + "\n", encoding="utf-8")
        _seal(run_dir)
        report = vr.verify_run(run_dir)
        assert "must be a JSON object" in _details(report, "FAIL", "schema")

    def test_rows_without_example_id_fail_join_key(self, tmp_path: Path):
        run_dir = _build_tree(tmp_path)
        rows = [_row(0), {"repeat_index": 0, "ok": True}, {**_row(2), "example_id": ""}]
        _write_jsonl(_window(run_dir) / "requests.jsonl", rows)
        _seal(run_dir)
        report = vr.verify_run(run_dir)
        assert "2 row(s) lack a non-empty string 'example_id'" in _details(
            report, "FAIL", "schema"
        )

    def test_missing_engine_metrics_fails(self, tmp_path: Path):
        run_dir = _build_tree(tmp_path)
        (_window(run_dir) / "engine_metrics.json").unlink()
        _seal(run_dir)
        report = vr.verify_run(run_dir)
        assert any(
            f["where"].endswith("engine_metrics.json")
            for f in _findings(report, "FAIL", "schema")
        )

    def test_invalid_engine_metrics_json_fails(self, tmp_path: Path):
        run_dir = _build_tree(tmp_path)
        (_window(run_dir) / "engine_metrics.json").write_text("{oops", encoding="utf-8")
        _seal(run_dir)
        report = vr.verify_run(run_dir)
        assert any(
            f["where"].endswith("engine_metrics.json") and "invalid JSON" in f["detail"]
            for f in _findings(report, "FAIL", "schema")
        )

    def test_missing_cage_stats_fails(self, tmp_path: Path):
        run_dir = _build_tree(tmp_path)
        (_window(run_dir) / "cage_stats.jsonl").unlink()
        _seal(run_dir)
        report = vr.verify_run(run_dir)
        assert any(
            f["where"].endswith("cage_stats.jsonl")
            for f in _findings(report, "FAIL", "schema")
        )

    def test_sharegpt_window_is_exempt_from_qa_evidence(self, tmp_path: Path):
        # §1: ShareGPT windows carry serving streams only — a missing
        # qa_evidence.jsonl is NOT a finding there.
        run_dir = _build_tree(tmp_path, dataset="sharegpt", evidence=False)
        _seal(run_dir)
        report = vr.verify_run(run_dir)
        assert report["ok"], report["findings"]
        assert not any(
            "qa_evidence" in f["where"] for f in report["findings"]
        )

    def test_missing_qa_evidence_fails_for_qa_dataset(self, tmp_path: Path):
        run_dir = _build_tree(tmp_path, evidence=False)
        _seal(run_dir)
        report = vr.verify_run(run_dir)
        assert any(
            f["where"].endswith("qa_evidence.jsonl")
            and "required §1 window artifact is missing" in f["detail"]
            for f in _findings(report, "FAIL", "schema")
        )


# --------------------------------------------------------------------------- #
# windows[] entry arms
# --------------------------------------------------------------------------- #


class TestWindowsTableEntryArms:
    def _rewrite_windows(self, run_dir: Path, windows: dict[str, Any]) -> None:
        (_cell_dir(run_dir) / "cell.json").write_text(
            json.dumps({"baseline": BASELINE, "windows": windows}), encoding="utf-8"
        )

    def test_non_object_entry_fails(self, tmp_path: Path):
        run_dir = _build_tree(tmp_path)
        self._rewrite_windows(run_dir, {f"{DATASET}-01": "not-an-object"})
        _seal(run_dir)
        report = vr.verify_run(run_dir)
        assert "must be an object" in _details(report, "FAIL", "window-coverage")

    def test_dataset_mismatch_fails(self, tmp_path: Path):
        run_dir = _build_tree(tmp_path)
        self._rewrite_windows(
            run_dir,
            {f"{DATASET}-01": {"dataset": "hotpotqa", "seed": 1, "rep": 1,
                               "t_start": 0.0, "t_end": 60.0}},
        )
        _seal(run_dir)
        report = vr.verify_run(run_dir)
        assert "does not match the window name's dataset" in _details(
            report, "FAIL", "window-coverage"
        )

    def test_missing_seed_rep_bounds_is_warn(self, tmp_path: Path):
        run_dir = _build_tree(tmp_path)
        self._rewrite_windows(run_dir, {f"{DATASET}-01": {"dataset": DATASET}})
        _seal(run_dir)
        report = vr.verify_run(run_dir)
        warn = _details(report, "WARN", "window-coverage")
        for field in ("seed", "rep", "t_start", "t_end"):
            assert field in warn
        # WARN only: the gate still passes.
        assert report["ok"]


# --------------------------------------------------------------------------- #
# Ledger arms + markdown phrasing
# --------------------------------------------------------------------------- #


class TestLedgerArms:
    def test_corrupt_ledger_is_a_ledger_fail(self, tmp_path: Path):
        run_dir = _build_tree(tmp_path)
        (run_dir / "ledger.json").write_text("{not json", encoding="utf-8")
        report = vr.verify_run(run_dir)
        assert _findings(report, "FAIL", "ledger")
        assert report["ledger"]["present"] is True
        assert report["ledger"]["n_entries"] is None
        # Markdown renders the present-but-unusable arm.
        assert "present but unusable" in vr.render_markdown(report)

    def test_manifest_not_sealed_is_warn(self, tmp_path: Path):
        run_dir = _build_tree(tmp_path)
        sealed = [
            p for p in sorted((run_dir / "cells").rglob("*")) if p.is_file()
        ]
        write_ledger(hash_artifacts(sealed, base_dir=run_dir), run_dir / "ledger.json")
        report = vr.verify_run(run_dir)
        assert any(
            "manifest.json is not among the sealed entries" in f["detail"]
            for f in _findings(report, "WARN", "ledger")
        )

    def test_absent_ledger_renders_absent(self, tmp_path: Path):
        run_dir = _build_tree(tmp_path)  # never sealed
        report = vr.verify_run(run_dir)
        assert "the run is unsealed" in _details(report, "FAIL", "ledger")
        assert "ABSENT" in vr.render_markdown(report)


# --------------------------------------------------------------------------- #
# Pilot-mode refusal arms
# --------------------------------------------------------------------------- #


class TestPilotModeArms:
    def test_nonexistent_results_dir_exits_2(self, tmp_path: Path, capsys):
        rc = vr._run_pilot(tmp_path / "missing", None)
        assert rc == 2
        assert "does not exist" in capsys.readouterr().err

    def test_empty_results_dir_fails_closed(self, tmp_path: Path, capsys):
        empty = tmp_path / "empty"
        empty.mkdir()
        out = tmp_path / "reports"
        rc = vr._run_pilot(empty, out)
        capsys.readouterr()
        assert rc == 1
        report = json.loads(
            (out / "verification_report.json").read_text(encoding="utf-8")
        )
        assert report["ok"] is False
        assert report["errors"] == ["no_per_trial_metrics_found"]
