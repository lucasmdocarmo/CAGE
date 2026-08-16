"""Tests for scripts/4_analysis/verify_results.py — the v2 campaign gate (#129/H6).

Synthetic RESULTS_LAYOUT §1 fixture trees exercise every gate check:

- a green tree PASSES (exit 0) and the report lands OUTSIDE the run root
  (sibling ``<run_root>_verification/``; the run tree gains no files);
- a lost qa_evidence row fails requests-vs-evidence reconciliation (H3);
- duplicate (example_id, repeat_index, record_index) identities fail — and the
  no-record_index variant points at producer task #127;
- a file added to cells/ AFTER sealing is detected as EXTRA (H7);
- rows carrying no ok/error validity field are a WARN, not a FAIL (#119/#127
  producer fix lands in parallel);
- cell.json windows[] coverage mismatches fail in both directions (§1);
- an unsealed run fails (§5), a tampered sealed artifact fails (HASH-MISMATCH);
- ``--out`` inside the run root is refused (exit 2);
- ``--pilot`` preserves the pilot-era metrics-vs-CSV behavior verbatim.
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

RUN_ID = "20260814-0900-a-qwen3-14b"
CAMPAIGN = "camp1"
SESSION = "a"
MODEL = "qwen3-14b"
DATASET = "squad_v2"
BASELINES = ("B3", "B6")
N_ROWS = 3
N_WINDOWS = 2


def _manifest() -> dict[str, Any]:
    return {
        "campaign": CAMPAIGN,
        "session": SESSION,
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
        "created_utc": "2026-08-14T09:00:00+00:00",
    }


def _row(
    i: int, *, with_validity: bool, with_record_index: bool
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "example_id": f"e{i}",
        "repeat_index": 0,
        "ttft_ms": 100.0 + i,
    }
    if with_record_index:
        row["record_index"] = i
    if with_validity:
        row.update(ok=True, error=None, empty_generation=False)
    return row


def _build_tree(
    tmp_path: Path,
    *,
    with_validity: bool = True,
    with_record_index: bool = True,
) -> Path:
    """UNSEALED §1 tree (2 cells x 2 windows x N_ROWS); seal with _seal()."""
    run_dir = tmp_path / "results" / CAMPAIGN / SESSION / RUN_ID
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(_manifest(), indent=2), encoding="utf-8"
    )
    for baseline in BASELINES:
        spec = CellSpec.from_baseline(baseline, model=MODEL)  # type: ignore[arg-type]
        cell_dir = run_dir / "cells" / spec.to_row_key()
        cell_dir.mkdir(parents=True)
        windows: dict[str, dict[str, Any]] = {}
        for ordinal in range(1, N_WINDOWS + 1):
            key = f"{DATASET}-{ordinal:02d}"
            windows[key] = {
                "dataset": DATASET,
                "seed": 1,
                "rep": ordinal,
                "t_start": 0.0,
                "t_end": 60.0,
            }
            wdir = cell_dir / f"window_{key}"
            wdir.mkdir()
            request_rows = [
                _row(i, with_validity=with_validity, with_record_index=with_record_index)
                for i in range(N_ROWS)
            ]
            evidence_rows = [
                {
                    **_row(
                        i,
                        with_validity=with_validity,
                        with_record_index=with_record_index,
                    ),
                    "question": "What color is the sky?",
                    "generated_answer": "blue",
                    "reference_answer": "blue",
                    "used_contexts": ["The sky is blue."],
                }
                for i in range(N_ROWS)
            ]
            _write_jsonl(wdir / "requests.jsonl", request_rows)
            _write_jsonl(wdir / "qa_evidence.jsonl", evidence_rows)
            (wdir / "engine_metrics.json").write_text(
                json.dumps({"snapshot": "before/after"}), encoding="utf-8"
            )
            _write_jsonl(wdir / "cage_stats.jsonl", [{"ts_s": 0.0, "kv_cache_usage": 0.1}])
        (cell_dir / "cell.json").write_text(
            json.dumps(
                {
                    "cellspec": spec.to_flat_dict(),
                    "baseline": baseline,
                    "windows": windows,
                }
            ),
            encoding="utf-8",
        )
    return run_dir


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _seal(run_dir: Path) -> None:
    """§5 seal: every artifact under cells/ PLUS manifest.json."""
    sealed = [p for p in sorted(run_dir.rglob("*")) if p.is_file() and p.name != "ledger.json"]
    write_ledger(hash_artifacts(sealed, base_dir=run_dir), run_dir / "ledger.json")


def _mk_green(tmp_path: Path, **kwargs: Any) -> Path:
    run_dir = _build_tree(tmp_path, **kwargs)
    _seal(run_dir)
    return run_dir


def _first_window(run_dir: Path) -> Path:
    return sorted(run_dir.glob("cells/*/window_*"))[0]


def _findings(report: dict[str, Any], severity: str, check: str | None = None) -> list[dict[str, Any]]:
    return [
        f
        for f in report["findings"]
        if f["severity"] == severity and (check is None or f["check"] == check)
    ]


# ---------------------------------------------------------------------------
# Green path + report placement (g)
# ---------------------------------------------------------------------------


def test_green_tree_passes_and_report_lands_outside(tmp_path: Path) -> None:
    run_dir = _mk_green(tmp_path)
    before = {p for p in run_dir.rglob("*")}
    assert vr.main([str(run_dir)]) == 0
    assert {p for p in run_dir.rglob("*")} == before  # run tree gained NOTHING

    out_dir = run_dir.parent / f"{RUN_ID}{vr.VERIFICATION_DIR_SUFFIX}"
    assert (out_dir / vr.REPORT_JSON_NAME).is_file()
    assert (out_dir / vr.REPORT_MD_NAME).is_file()
    report = json.loads((out_dir / vr.REPORT_JSON_NAME).read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["n_fail"] == 0
    assert report["n_warn"] == 0
    totals = report["accounting"]["totals"]
    assert totals["n_windows"] == len(BASELINES) * N_WINDOWS
    assert totals["n_requests_rows"]["sum_over_known_windows"] == (
        len(BASELINES) * N_WINDOWS * N_ROWS
    )
    assert totals["n_valid_known"]["sum_over_known_windows"] == (
        len(BASELINES) * N_WINDOWS * N_ROWS
    )
    assert totals["n_error"]["sum_over_known_windows"] == 0


def test_out_inside_run_root_is_refused(tmp_path: Path) -> None:
    run_dir = _mk_green(tmp_path)
    assert vr.main([str(run_dir), "--out", str(run_dir / "verification")]) == 2
    assert vr.main([str(run_dir), "--out", str(run_dir)]) == 2
    with pytest.raises(vr.VerifyRefusal, match="OUTSIDE"):
        vr.resolve_out_dir(run_dir, run_dir / "index")
    # A legal explicit --out still works.
    elsewhere = tmp_path / "reports"
    assert vr.main([str(run_dir), "--out", str(elsewhere)]) == 0
    assert (elsewhere / vr.REPORT_JSON_NAME).is_file()


# ---------------------------------------------------------------------------
# (b) reconciliation
# ---------------------------------------------------------------------------


def test_missing_evidence_row_fails_reconciliation(tmp_path: Path) -> None:
    run_dir = _build_tree(tmp_path)
    evidence = _first_window(run_dir) / "qa_evidence.jsonl"
    rows = _read_jsonl(evidence)
    _write_jsonl(evidence, rows[:-1])  # one evidence append lost (H3)
    _seal(run_dir)
    report = vr.verify_run(run_dir)
    assert report["ok"] is False
    recon = _findings(report, "FAIL", "reconciliation")
    assert len(recon) == 1
    assert f"{N_ROWS} row(s)" in recon[0]["detail"]
    assert f"{N_ROWS - 1}" in recon[0]["detail"]
    assert "e2" in recon[0]["detail"]  # the lost identity is named


# ---------------------------------------------------------------------------
# (c) duplicate identities
# ---------------------------------------------------------------------------


def test_duplicate_identity_fails(tmp_path: Path) -> None:
    run_dir = _build_tree(tmp_path)
    for name in ("requests.jsonl", "qa_evidence.jsonl"):
        path = _first_window(run_dir) / name
        rows = _read_jsonl(path)
        _write_jsonl(path, rows + [rows[0]])  # same (id, repeat, record) twice
    _seal(run_dir)
    report = vr.verify_run(run_dir)
    assert report["ok"] is False
    dups = _findings(report, "FAIL", "duplicates")
    assert len(dups) == 2  # both files carry the duplicate
    assert "example_id='e0'" in dups[0]["detail"]


def test_duplicate_without_record_index_points_at_task_127(tmp_path: Path) -> None:
    run_dir = _build_tree(tmp_path, with_record_index=False)
    path = _first_window(run_dir) / "requests.jsonl"
    rows = _read_jsonl(path)
    _write_jsonl(path, rows + [rows[0]])  # open-loop replay: same id, no key
    _seal(run_dir)
    report = vr.verify_run(run_dir)
    assert report["ok"] is False
    dups = _findings(report, "FAIL", "duplicates")
    assert any("#127" in f["detail"] for f in dups)


# ---------------------------------------------------------------------------
# (a) validity fields WARN — absence is not a gate failure while #119/#127 land
# ---------------------------------------------------------------------------


def test_validity_fields_absent_is_warn_not_fail(tmp_path: Path) -> None:
    run_dir = _mk_green(tmp_path, with_validity=False)
    report = vr.verify_run(run_dir)
    assert report["ok"] is True  # WARNs do not flip the gate
    warns = _findings(report, "WARN", "schema")
    assert warns, "expected ok/error-absence WARNs"
    assert any("#119" in f["detail"] and "#127" in f["detail"] for f in warns)
    # §9.10 accounting reports UNKNOWN validity, never a coerced zero.
    row = report["accounting"]["per_window"][0]
    assert row["n_validity_unknown"] == N_ROWS
    assert row["n_valid_known"] == 0


# ---------------------------------------------------------------------------
# (d) windows[] coverage
# ---------------------------------------------------------------------------


def test_window_coverage_mismatch_fails_both_directions(tmp_path: Path) -> None:
    run_dir = _build_tree(tmp_path)
    cell_json = sorted(run_dir.glob("cells/*/cell.json"))[0]
    meta = json.loads(cell_json.read_text(encoding="utf-8"))
    del meta["windows"][f"{DATASET}-01"]  # directory without declaration
    meta["windows"][f"{DATASET}-09"] = {  # declaration without directory
        "dataset": DATASET,
        "seed": 1,
        "rep": 9,
        "t_start": 0.0,
        "t_end": 60.0,
    }
    cell_json.write_text(json.dumps(meta), encoding="utf-8")
    _seal(run_dir)
    report = vr.verify_run(run_dir)
    assert report["ok"] is False
    coverage = _findings(report, "FAIL", "window-coverage")
    details = "\n".join(f["detail"] for f in coverage)
    assert f"no windows['{DATASET}-01'] entry" in details
    assert f"windows['{DATASET}-09'] declared but no window_{DATASET}-09" in details


def test_windows_table_absent_fails_with_126_pointer(tmp_path: Path) -> None:
    run_dir = _build_tree(tmp_path)
    cell_json = sorted(run_dir.glob("cells/*/cell.json"))[0]
    meta = json.loads(cell_json.read_text(encoding="utf-8"))
    del meta["windows"]
    cell_json.write_text(json.dumps(meta), encoding="utf-8")
    _seal(run_dir)
    report = vr.verify_run(run_dir)
    assert report["ok"] is False
    coverage = _findings(report, "FAIL", "window-coverage")
    assert any("#126" in f["detail"] for f in coverage)


# ---------------------------------------------------------------------------
# (f) ledger: unsealed / EXTRA / tamper
# ---------------------------------------------------------------------------


def test_unsealed_run_fails(tmp_path: Path) -> None:
    run_dir = _build_tree(tmp_path)  # never sealed
    report = vr.verify_run(run_dir)
    assert report["ok"] is False
    ledger_fails = _findings(report, "FAIL", "ledger")
    assert any("unsealed" in f["detail"] for f in ledger_fails)


def test_extra_unsealed_file_detected(tmp_path: Path) -> None:
    run_dir = _mk_green(tmp_path)
    sneaky = _first_window(run_dir) / "sneaky_extra.jsonl"
    _write_jsonl(sneaky, [{"example_id": "ghost"}])  # added AFTER the seal
    report = vr.verify_run(run_dir)
    assert report["ok"] is False
    ledger_fails = _findings(report, "FAIL", "ledger")
    assert any(
        f["detail"].startswith("EXTRA ") and "sneaky_extra.jsonl" in f["detail"]
        for f in ledger_fails
    )


def test_tampered_sealed_artifact_fails(tmp_path: Path) -> None:
    run_dir = _mk_green(tmp_path)
    victim = _first_window(run_dir) / "requests.jsonl"
    rows = _read_jsonl(victim)
    rows[0]["ttft_ms"] = 1.0
    _write_jsonl(victim, rows)
    report = vr.verify_run(run_dir)
    assert report["ok"] is False
    assert any(
        f["detail"].startswith("HASH-MISMATCH ")
        for f in _findings(report, "FAIL", "ledger")
    )


# ---------------------------------------------------------------------------
# (h) exit codes + refusals
# ---------------------------------------------------------------------------


def test_gate_exit_codes(tmp_path: Path) -> None:
    run_dir = _mk_green(tmp_path)
    assert vr.main([str(run_dir)]) == 0
    _write_jsonl(_first_window(run_dir) / "sneaky.jsonl", [{"example_id": "x"}])
    assert vr.main([str(run_dir)]) == 1  # FAIL -> nonzero (gate semantics)
    assert vr.main([str(tmp_path / "no-such-run")]) == 2


# ---------------------------------------------------------------------------
# --pilot mode: the pre-v2 metrics-vs-CSV behavior, preserved
# ---------------------------------------------------------------------------


def _write_pilot_cell(trial_dir: Path, *, n_rows: int) -> None:
    trial_dir.mkdir(parents=True, exist_ok=True)
    stem = "no_cache_squad_20260101"
    metrics = {
        "experiment": {"baseline": "no_cache", "dataset": "squad", "model": "m"},
        "performance": {"total_requests": n_rows},
    }
    (trial_dir / f"{stem}_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    rows = "\n".join(str(i) for i in range(n_rows))
    (trial_dir / f"{stem}_results.csv").write_text(f"example_id\n{rows}\n", encoding="utf-8")


def test_pilot_mode_preserves_old_behavior(tmp_path: Path) -> None:
    pilot = tmp_path / "pilot_run"
    _write_pilot_cell(pilot / "baselines" / "no_cache" / "trial_1", n_rows=2)
    assert vr.main(["--pilot", "--results-dir", str(pilot)]) == 0
    # Old contract: reports land INSIDE the pilot dir (pilot trees are unsealed).
    report = json.loads((pilot / "verification_report.json").read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert len(report["checks"]) == 1
    assert (pilot / "verification_report.txt").is_file()


def test_pilot_mode_gate_exit_on_mismatch(tmp_path: Path) -> None:
    pilot = tmp_path / "pilot_bad"
    _write_pilot_cell(pilot / "baselines" / "no_cache" / "trial_1", n_rows=2)
    csv = next(pilot.rglob("*_results.csv"))
    csv.write_text("example_id\n0\n", encoding="utf-8")  # 1 row vs expected 2
    assert vr.main(["--pilot", "--results-dir", str(pilot)]) == 1
