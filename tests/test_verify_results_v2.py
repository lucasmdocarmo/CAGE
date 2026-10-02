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
- a row absent from BOTH per-query chains (the ADR-0116 W3 signature, invisible
  to the requests-vs-evidence reconciliation) fails the row-count check against
  the window's offered population, and a nonzero consort counter fails too;
- check (j), ADR-0118 (Batch 2 W5): a completed row with two or more output
  tokens must carry a finite positive tpot_ms, a completed row with at most
  one output token must carry a null tpot_ms (no decode phase; counted as
  n_no_decode), a completed row must carry an integer num_tokens, and a
  whitespace-sourced token count is a WARN naming the missing usage chunk;
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
        # The runner writes both on every result row (run_experiment
        # record_result): a decode phase of 7 tokens at 12 ms per token.
        "num_tokens": 8,
        "tpot_ms": 12.0,
    }
    if with_record_index:
        row["record_index"] = i
    if with_validity:
        row.update(ok=True, error=None, empty_generation=False)
    return row


ZERO_CONSORT: dict[str, Any] = {
    "n_dropped_prepare": 0,
    "n_dropped_record": 0,
    "n_dropped_turn": 0,
    "evidence_write_failures": 0,
    "evidence_write_first_error": None,
}


def _window_metrics(
    *,
    mode: str = "single",
    n_offered: int = N_ROWS,
    consort: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The runner summary campaign_session writes beside the §1 artifacts: the
    offered population (closed loop: experiment.num_measured_requests; open
    loop: workload.open_loop.n_scheduled) and the §9.10 consort counters."""
    doc: dict[str, Any] = {
        "experiment": {"num_measured_requests": n_offered, "stale_index_opt_in": False},
        "workload": {"mode": mode},
        "consort": {**ZERO_CONSORT, **(consort or {})},
    }
    if mode == "open_loop":
        doc["workload"]["open_loop"] = {"n_scheduled": n_offered}
    return doc


def _build_tree(
    tmp_path: Path,
    *,
    with_validity: bool = True,
    with_record_index: bool = True,
    metrics: bool = True,
    metrics_mode: str = "single",
    consort: dict[str, Any] | None = None,
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
            if metrics:
                (wdir / vr._WINDOW_METRICS_NAME).write_text(
                    json.dumps(_window_metrics(mode=metrics_mode, consort=consort)),
                    encoding="utf-8",
                )
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


# ---------------------------------------------------------------------------
# (i) row count vs the offered population + consort counters (ADR-0116, W3)
# ---------------------------------------------------------------------------


def test_window_metrics_name_and_counters_match_the_producer() -> None:
    from src.orchestration import campaign_session as cs

    assert vr._WINDOW_METRICS_NAME == cs.WINDOW_METRICS_NAME
    assert set(vr._CONSORT_COUNTERS) == set(cs.CONSORT_COUNTERS)


def test_expected_row_count_reads_the_offered_population() -> None:
    assert vr.expected_row_count(_window_metrics()) == (
        N_ROWS, "experiment.num_measured_requests",
    )
    assert vr.expected_row_count(_window_metrics(mode="open_loop", n_offered=7)) == (
        7, "workload.open_loop.n_scheduled",
    )
    # Absence and non-integers are unknown, never coerced.
    assert vr.expected_row_count({})[0] is None
    assert vr.expected_row_count({"workload": {"mode": "open_loop"}})[0] is None
    assert vr.expected_row_count({"experiment": {"num_measured_requests": "3"}})[0] is None
    assert vr.expected_row_count({"experiment": {"num_measured_requests": True}})[0] is None


def test_row_dropped_from_both_chains_fails_the_row_count_check(tmp_path: Path) -> None:
    # The W3 signature: the request is absent from requests.jsonl AND
    # qa_evidence.jsonl, so the requests-vs-evidence reconciliation passes;
    # only the offered population reveals the loss.
    run_dir = _build_tree(tmp_path)
    wdir = _first_window(run_dir)
    for name in ("requests.jsonl", "qa_evidence.jsonl"):
        rows = _read_jsonl(wdir / name)
        _write_jsonl(wdir / name, rows[:-1])
    _seal(run_dir)
    report = vr.verify_run(run_dir)
    assert report["ok"] is False
    assert not _findings(report, "FAIL", "reconciliation")  # both chains agree
    (finding,) = _findings(report, "FAIL", "row-count")
    assert f"{N_ROWS - 1} row(s)" in finding["detail"]
    assert f"offered {N_ROWS}" in finding["detail"]
    assert "num_measured_requests" in finding["detail"] and "ADR-0116" in finding["detail"]
    row = next(r for r in report["accounting"]["per_window"] if r["window"] == finding["where"])
    assert row["n_expected_rows"] == N_ROWS and row["n_requests_rows"] == N_ROWS - 1


def test_nonzero_consort_counter_fails(tmp_path: Path) -> None:
    run_dir = _mk_green(tmp_path, consort={"n_dropped_record": 1})
    report = vr.verify_run(run_dir)
    assert report["ok"] is False
    fails = _findings(report, "FAIL", "row-count")
    assert len(fails) == len(BASELINES) * N_WINDOWS
    assert all("n_dropped_record = 1" in f["detail"] for f in fails)


def test_open_loop_offered_population_is_the_schedule(tmp_path: Path) -> None:
    run_dir = _mk_green(tmp_path, metrics_mode="open_loop")
    report = vr.verify_run(run_dir)
    assert report["ok"] is True and report["n_warn"] == 0
    # One scheduled arrival without a row (no dispatch stub either) fails.
    run_dir2 = _build_tree(tmp_path / "two", metrics_mode="open_loop")
    wdir = _first_window(run_dir2)
    doc = json.loads((wdir / vr._WINDOW_METRICS_NAME).read_text(encoding="utf-8"))
    doc["workload"]["open_loop"]["n_scheduled"] = N_ROWS + 1
    (wdir / vr._WINDOW_METRICS_NAME).write_text(json.dumps(doc), encoding="utf-8")
    _seal(run_dir2)
    report2 = vr.verify_run(run_dir2)
    (finding,) = _findings(report2, "FAIL", "row-count")
    assert "n_scheduled" in finding["detail"] and f"offered {N_ROWS + 1}" in finding["detail"]


def test_absent_window_metrics_is_a_warn_and_an_incomplete_one_fails(tmp_path: Path) -> None:
    run_dir = _mk_green(tmp_path, metrics=False)
    report = vr.verify_run(run_dir)
    assert report["ok"] is True  # a WARN never flips the gate
    warns = _findings(report, "WARN", "row-count")
    assert len(warns) == len(BASELINES) * N_WINDOWS
    assert all("unknown" in f["detail"] for f in warns)
    # A summary without the consort block or the offered population refuses:
    # a present summary is the producer's own record, a gap in it is a defect.
    run_dir2 = _build_tree(tmp_path / "two")
    wdir = _first_window(run_dir2)
    (wdir / vr._WINDOW_METRICS_NAME).write_text(json.dumps({"experiment": {}}), encoding="utf-8")
    _seal(run_dir2)
    report2 = vr.verify_run(run_dir2)
    details = "\n".join(f["detail"] for f in _findings(report2, "FAIL", "row-count"))
    assert "consort" in details and "offered population unknown" in details


# ---------------------------------------------------------------------------
# (j) per-row TPOT vs the output-token count (ADR-0118, Batch 2 W5)
# ---------------------------------------------------------------------------


def _rewrite_first_window_row(run_dir: Path, **fields: Any) -> Path:
    """Overwrite the first request row of the first window with ``fields``
    (a None value writes JSON null) and seal; returns the window dir."""
    wdir = _first_window(run_dir)
    rows = _read_jsonl(wdir / "requests.jsonl")
    rows[0].update(fields)
    _write_jsonl(wdir / "requests.jsonl", rows)
    _seal(run_dir)
    return wdir


def _accounting_row(report: dict[str, Any], run_dir: Path, wdir: Path) -> dict[str, Any]:
    rel = wdir.relative_to(run_dir).as_posix()
    return next(r for r in report["accounting"]["per_window"] if r["window"] == rel)


@pytest.mark.parametrize("num_tokens", [0, 1])
def test_completion_without_a_decode_phase_passes_and_is_counted(
    tmp_path: Path, num_tokens: int
) -> None:
    run_dir = _build_tree(tmp_path)
    wdir = _rewrite_first_window_row(run_dir, num_tokens=num_tokens, tpot_ms=None)
    report = vr.verify_run(run_dir)
    assert report["ok"] is True and report["n_warn"] == 0
    assert _accounting_row(report, run_dir, wdir)["n_no_decode"] == 1
    assert all(
        r["n_no_decode"] == 0
        for r in report["accounting"]["per_window"]
        if r["window"] != wdir.relative_to(run_dir).as_posix()
    )
    totals = report["accounting"]["totals"]["n_no_decode"]
    assert totals == {"sum_over_known_windows": 1, "n_windows_known": len(BASELINES) * N_WINDOWS}


@pytest.mark.parametrize("tpot_ms", [None, 0.0, -1.0, "12"], ids=["null", "zero", "negative", "string"])
def test_decode_tokens_without_a_positive_tpot_fails(tmp_path: Path, tpot_ms: Any) -> None:
    run_dir = _build_tree(tmp_path)
    _rewrite_first_window_row(run_dir, num_tokens=5, tpot_ms=tpot_ms)
    report = vr.verify_run(run_dir)
    assert report["ok"] is False
    (finding,) = _findings(report, "FAIL", "tpot")
    assert "1 completed (ok) row(s)" in finding["detail"]
    assert "ADR-0118" in finding["detail"] and "e0" in finding["detail"]


def test_one_token_completion_with_a_tpot_value_fails(tmp_path: Path) -> None:
    run_dir = _build_tree(tmp_path)
    _rewrite_first_window_row(run_dir, num_tokens=1, tpot_ms=12.0)
    report = vr.verify_run(run_dir)
    assert report["ok"] is False
    (finding,) = _findings(report, "FAIL", "tpot")
    assert "no decode phase" in finding["detail"] and "e0" in finding["detail"]


@pytest.mark.parametrize(
    "num_tokens", ["absent", None, "8", 8.0, True, -1],
    ids=["absent", "null", "string", "float", "bool", "negative"],
)
def test_completed_row_without_an_integer_num_tokens_fails(
    tmp_path: Path, num_tokens: Any
) -> None:
    run_dir = _build_tree(tmp_path)
    wdir = _first_window(run_dir)
    rows = _read_jsonl(wdir / "requests.jsonl")
    if num_tokens == "absent":
        del rows[0]["num_tokens"]
    else:
        rows[0]["num_tokens"] = num_tokens
    _write_jsonl(wdir / "requests.jsonl", rows)
    _seal(run_dir)
    report = vr.verify_run(run_dir)
    assert report["ok"] is False
    (finding,) = _findings(report, "FAIL", "tpot")
    assert "num_tokens" in finding["detail"] and "e0" in finding["detail"]


def test_not_ok_rows_are_exempt_from_the_tpot_check(tmp_path: Path) -> None:
    run_dir = _build_tree(tmp_path)
    wdir = _rewrite_first_window_row(
        run_dir, ok=False, error="HTTP 500", num_tokens=0, tpot_ms=None
    )
    report = vr.verify_run(run_dir)
    assert not _findings(report, "FAIL", "tpot")
    # A failed request has no completion to exempt: never counted.
    assert _accounting_row(report, run_dir, wdir)["n_no_decode"] == 0


def test_validity_unknown_rows_are_exempt_from_the_tpot_check(tmp_path: Path) -> None:
    run_dir = _build_tree(tmp_path, with_validity=False)
    wdir = _rewrite_first_window_row(run_dir, num_tokens=1, tpot_ms=None)
    report = vr.verify_run(run_dir)
    assert report["ok"] is True  # the validity WARN stands alone
    assert not _findings(report, "FAIL", "tpot")
    # The file's accounting convention (review T2, kept): the count is over
    # known completions, so a validity-unknown window reads 0 beside its
    # n_validity_unknown, exactly like n_valid_known.
    row = _accounting_row(report, run_dir, wdir)
    assert row["n_no_decode"] == 0 and row["n_valid_known"] == 0
    assert row["n_validity_unknown"] == N_ROWS


def test_whitespace_token_count_is_a_warn_naming_the_missing_usage_chunk(
    tmp_path: Path,
) -> None:
    run_dir = _build_tree(tmp_path)
    _rewrite_first_window_row(run_dir, num_tokens_source="whitespace")
    report = vr.verify_run(run_dir)
    assert report["ok"] is True
    (warn,) = _findings(report, "WARN", "tpot")
    assert "1 completed (ok) row(s)" in warn["detail"]
    assert "whitespace" in warn["detail"] and "usage" in warn["detail"]
    # A usage-sourced count (or a row without the column) never warns.
    run_dir2 = _build_tree(tmp_path / "two")
    _rewrite_first_window_row(run_dir2, num_tokens_source="usage")
    assert not _findings(vr.verify_run(run_dir2), "WARN", "tpot")


def test_markdown_report_renders_the_no_decode_column(tmp_path: Path) -> None:
    run_dir = _build_tree(tmp_path)
    wdir = _rewrite_first_window_row(run_dir, num_tokens=1, tpot_ms=None)
    md = vr.render_markdown(vr.verify_run(run_dir))
    lines = md.splitlines()
    header = next(l for l in lines if l.startswith("| window |"))
    assert header.split("|")[-2].strip() == "no-decode"  # the last column
    rel = wdir.relative_to(run_dir).as_posix()
    rewritten = next(l for l in lines if l.startswith(f"| {rel} |"))
    assert rewritten.split("|")[-2].strip() == "1"  # the cell, not just the header
    other = next(l for l in lines if l.startswith("| cells/") and not l.startswith(f"| {rel} |"))
    assert other.split("|")[-2].strip() == "0"


def test_reference_engine_rows_are_exempt_from_the_tpot_clauses(tmp_path: Path) -> None:
    # The HF oracle never streams: ttft_ms == total_time_ms by contract, so the
    # runner's formula writes tpot_ms 0.0 on every multi-token completion. That
    # is the engine's documented shape, not a captured-timing defect (review
    # R1); the oracle is never scored for timeliness.
    run_dir = _build_tree(tmp_path)
    wdir = _rewrite_first_window_row(
        run_dir, num_tokens=9, tpot_ms=0.0, engine_id="hf_reference",
        reference_engine=True, num_tokens_source="token_ids",
    )
    report = vr.verify_run(run_dir)
    assert report["ok"] is True and not _findings(report, "FAIL", "tpot")
    assert _accounting_row(report, run_dir, wdir)["n_no_decode"] == 0
    # A one-token oracle completion still counts as no-decode.
    run_dir2 = _build_tree(tmp_path / "two")
    wdir2 = _rewrite_first_window_row(
        run_dir2, num_tokens=1, tpot_ms=None, engine_id="hf_reference", reference_engine=True,
    )
    report2 = vr.verify_run(run_dir2)
    assert report2["ok"] is True
    assert _accounting_row(report2, run_dir2, wdir2)["n_no_decode"] == 1
    # The same 0.0 on a serving engine IS the defect; a false flag exempts nothing.
    run_dir3 = _build_tree(tmp_path / "three")
    _rewrite_first_window_row(run_dir3, num_tokens=9, tpot_ms=0.0, engine_id="vllm", reference_engine=False)
    (finding,) = _findings(vr.verify_run(run_dir3), "FAIL", "tpot")
    assert "captured-timing defect" in finding["detail"]


# ---------------------------------------------------------------------------
# (k) per-window pd transfer proof (S0F-22 Batch 2, ADR-0134)
# ---------------------------------------------------------------------------


def _pd_transfer_record(**over: Any) -> dict[str, Any]:
    """A verified decode-side record for a window of N_ROWS served requests,
    built by the producer's own function so the fixture cannot drift from
    the record shape run_experiment writes."""
    from src.monitoring import vllm_telemetry as vt

    start = {"bytes_sum": 4096, "transfer_count": 1, "failed_transfers": 0,
             "failed_notifications": 0, "external_kv_tokens": 7, "local_compute_tokens": 1,
             "local_cache_hit_tokens": 0, "recomputed_tokens": 1, "preemptions": 0}
    end = {"bytes_sum": 4096 + N_ROWS * 1048576, "transfer_count": 1 + N_ROWS,
           "failed_transfers": 0, "failed_notifications": 0, "external_kv_tokens": 7 + 481,
           "local_compute_tokens": 1 + N_ROWS, "local_cache_hit_tokens": 32,
           "recomputed_tokens": 1 + N_ROWS, "preemptions": 0}
    record = vt.pd_transfer_record(
        start=start, end=end, prefill_start={"kv_expired_reqs": 0}, prefill_end={"kv_expired_reqs": 0},
        n_served_rows=N_ROWS, prompt_tokens_sum=513,
        decode_url="http://localhost:8200", prefill_url="http://localhost:8100",
        scrape_start_ts=0.0, scrape_end_ts=60.0,
    )
    record.update(over)
    return record


def _add_pd_cell(
    run_dir: Path, *, pd_transfer: dict[str, Any] | None | str = "verified",
    metrics: bool = True, n_error_rows: int = 0,
) -> Path:
    """Add ONE pd cell (B3, family DIST) with one window to an unsealed tree;
    returns the window dir. ``pd_transfer="verified"`` writes a healthy record,
    None writes no key, a dict is written verbatim. ``n_error_rows`` turns that
    many request rows into refused requests (``error`` set), the shape the
    runner's served-row rule excludes."""
    spec = CellSpec.from_baseline("B3", model=MODEL, family="DIST", topology="pd")  # type: ignore[arg-type]
    cell_dir = run_dir / "cells" / spec.to_row_key()
    cell_dir.mkdir(parents=True)
    key = f"{DATASET}-01"
    wdir = cell_dir / f"window_{key}"
    wdir.mkdir()
    rows = [_row(i, with_validity=True, with_record_index=True) for i in range(N_ROWS)]
    for row in rows[:n_error_rows]:
        row.update(ok=False, error="HTTP 502: prefill returned no usable KV transfer ticket",
                   num_tokens=0, tpot_ms=None)
    _write_jsonl(wdir / "requests.jsonl", rows)
    _write_jsonl(wdir / "qa_evidence.jsonl", [
        {**r, "question": "q", "generated_answer": "a", "reference_answer": "a", "used_contexts": ["c"]}
        for r in rows
    ])
    (wdir / "engine_metrics.json").write_text(json.dumps({"snapshot": "x"}), encoding="utf-8")
    _write_jsonl(wdir / "cage_stats.jsonl", [{"ts_s": 0.0, "kv_cache_usage": 0.1}])
    if metrics:
        doc = _window_metrics()
        if pd_transfer == "verified":
            doc["pd_transfer"] = _pd_transfer_record()
        elif pd_transfer is not None:
            doc["pd_transfer"] = pd_transfer
        (wdir / vr._WINDOW_METRICS_NAME).write_text(json.dumps(doc), encoding="utf-8")
    (cell_dir / "cell.json").write_text(json.dumps({
        "cellspec": spec.to_flat_dict(), "baseline": "B3", "gpu_count": 2,
        "windows": {key: {"dataset": DATASET, "seed": 1, "rep": 1, "t_start": 0.0, "t_end": 60.0}},
    }), encoding="utf-8")
    return wdir


def test_pd_window_with_a_verified_record_passes_and_rides_the_accounting(tmp_path: Path) -> None:
    run_dir = _build_tree(tmp_path)
    wdir = _add_pd_cell(run_dir)
    _seal(run_dir)
    report = vr.verify_run(run_dir)
    assert report["ok"] is True and report["n_warn"] == 0
    assert _accounting_row(report, run_dir, wdir)["pd_transfer_verified"] is True
    # single-topology windows carry no proof and no verdict: unknown, not False
    others = [r for r in report["accounting"]["per_window"] if r["window"] != wdir.relative_to(run_dir).as_posix()]
    assert len(others) == len(BASELINES) * N_WINDOWS
    assert all(r["pd_transfer_verified"] is None for r in others)
    # the markdown table is unchanged: no-decode stays the last column
    header = next(l for l in vr.render_markdown(report).splitlines() if l.startswith("| window |"))
    assert header.split("|")[-2].strip() == "no-decode"


def test_pd_window_without_the_record_fails(tmp_path: Path) -> None:
    run_dir = _build_tree(tmp_path)
    _add_pd_cell(run_dir, pd_transfer=None)
    _seal(run_dir)
    report = vr.verify_run(run_dir)
    assert report["ok"] is False
    (finding,) = _findings(report, "FAIL", "pd-transfer")
    assert "pd_transfer" in finding["detail"] and "ADR-0134" in finding["detail"]
    assert finding["where"].endswith(vr._WINDOW_METRICS_NAME)


def test_pd_window_without_metrics_json_fails_not_warns(tmp_path: Path) -> None:
    # (i) alone gives a WARN for a missing summary; a pd window without its
    # transfer proof must never verify green on that WARN.
    run_dir = _build_tree(tmp_path)
    wdir = _add_pd_cell(run_dir, metrics=False)
    _seal(run_dir)
    report = vr.verify_run(run_dir)
    assert report["ok"] is False
    (finding,) = _findings(report, "FAIL", "pd-transfer")
    assert "no runner summary" in finding["detail"]
    assert _accounting_row(report, run_dir, wdir)["pd_transfer_verified"] is None


def test_pd_window_whose_deltas_fail_a_clause_is_refused_offline(tmp_path: Path) -> None:
    run_dir = _build_tree(tmp_path)
    record = _pd_transfer_record()
    record["delta"]["failed_transfers"] = 1
    record["verified"] = False
    record["reasons"] = ["failed_transfers == 0 violated: 1 failed pull(s)"]
    wdir = _add_pd_cell(run_dir, pd_transfer=record)
    _seal(run_dir)
    report = vr.verify_run(run_dir)
    assert report["ok"] is False
    (finding,) = _findings(report, "FAIL", "pd-transfer")
    assert "failed_transfers == 0" in finding["detail"]
    assert _accounting_row(report, run_dir, wdir)["pd_transfer_verified"] is False


def test_recorded_verdict_that_disagrees_with_the_deltas_fails(tmp_path: Path) -> None:
    # the producer's flag is re-derived, never trusted
    run_dir = _build_tree(tmp_path)
    record = _pd_transfer_record()
    record["delta"]["bytes_sum"] = 0  # the flag still says verified
    _add_pd_cell(run_dir, pd_transfer=record)
    _seal(run_dir)
    report = vr.verify_run(run_dir)
    fails = _findings(report, "FAIL", "pd-transfer")
    details = "\n".join(f["detail"] for f in fails)
    assert "bytes_sum > 0" in details and "disagrees" in details


def test_recorded_served_count_must_match_the_rows(tmp_path: Path) -> None:
    # one request refused by the proxy: 2 served rows, the record claims 3
    run_dir = _build_tree(tmp_path)
    _add_pd_cell(run_dir, n_error_rows=1)
    _seal(run_dir)
    report = vr.verify_run(run_dir)
    assert report["ok"] is False
    (finding,) = _findings(report, "FAIL", "pd-transfer")
    assert "n_served_rows" in finding["detail"] and "3" in finding["detail"] and "2" in finding["detail"]


@pytest.mark.parametrize("record", [
    "not a dict", {"verified": True}, {"delta": "x", "n_served_rows": 3, "verified": True},
], ids=["string", "no-delta", "delta-not-a-dict"])
def test_malformed_record_fails(tmp_path: Path, record: Any) -> None:
    run_dir = _build_tree(tmp_path)
    _add_pd_cell(run_dir, pd_transfer=record)
    _seal(run_dir)
    report = vr.verify_run(run_dir)
    assert report["ok"] is False
    assert _findings(report, "FAIL", "pd-transfer")


def test_single_topology_window_with_a_record_fails(tmp_path: Path) -> None:
    # a non-pd cell ran against a decode endpoint: a mislabeled cell
    run_dir = _build_tree(tmp_path)
    wdir = _first_window(run_dir)
    doc = json.loads((wdir / vr._WINDOW_METRICS_NAME).read_text(encoding="utf-8"))
    doc["pd_transfer"] = _pd_transfer_record()
    (wdir / vr._WINDOW_METRICS_NAME).write_text(json.dumps(doc), encoding="utf-8")
    _seal(run_dir)
    report = vr.verify_run(run_dir)
    assert report["ok"] is False
    (finding,) = _findings(report, "FAIL", "pd-transfer")
    assert "single" in finding["detail"]


def test_prefill_expiry_inside_the_window_is_a_warn(tmp_path: Path) -> None:
    # Review LOW-4: a nonzero prefill expiry delta means a ticket was never
    # pulled before its 480 s deadline (an earlier window's request, or a
    # decode that never notified); the gate clauses do not see it, so the
    # verifier names it without flipping the gate.
    run_dir = _build_tree(tmp_path)
    record = _pd_transfer_record()
    record["prefill_delta"] = {"kv_expired_reqs": 2}
    wdir = _add_pd_cell(run_dir, pd_transfer=record)
    _seal(run_dir)
    report = vr.verify_run(run_dir)
    assert report["ok"] is True
    (warn,) = _findings(report, "WARN", "pd-transfer")
    assert "kv_expired_reqs" in warn["detail"] and "2" in warn["detail"]
    assert _accounting_row(report, run_dir, wdir)["pd_transfer_verified"] is True


def test_check_k_re_derives_with_the_producers_rule() -> None:
    from src.monitoring import vllm_telemetry as vt

    assert vr._pd_transfer_reasons is vt.pd_transfer_reasons
    assert vr._PD_TRANSFER_KEY == "pd_transfer"
