"""Tests for scripts/4_analysis/run_campaign_analysis.py (the D9 analysis driver).

Builds a synthetic v2 campaign run tree (the RESULTS_LAYOUT §1 fixture pattern
from tests/test_organize_results.py) with real per-query metric values in
requests.jsonl/qa_evidence.jsonl, organizes it with organize_results.py, then
exercises the driver:

- design-input default: stats.json with the expected keys, forest/wlt figures
  rendered, DESIGN-INPUT-ONLY stamp on every output;
- missing index refuses and names organize_results (no auto-run);
- confirmatory without --i-understand-one-look / --registered-sha /
  --calibration-report refuses;
- the §9.7 calibration precondition: a FAILING report artifact refuses, a
  passing one is recorded in stats['preconditions'];
- the §9.10 ledger precondition: a tampered artifact refuses confirmatory;
- a second confirmatory run refuses via <run>/analysis_lock.json (§9.11);
- the §9.3 gatekeeping chain (Dmitrienko serial + Holm within family) runs
  with an auditable trace, flagging the incomplete registered chain;
- window-unit baseline-pair contrasts (#15) compute batch means (§9.4);
  unconsumed F2 rows stay a labeled skip, never numbers;
- the §9.5 conditional-TOST equivalence legs compute when margin + pair +
  policy_event mask exist, and are labeled skips otherwise;
- §9.8 blinding: a sealed arm map scrambles design-input output labels and
  suppresses figures; the confirmatory look records the one-time unblinding;
- unknown contrast ids fail loud; selector contrasts skip, labeled;
- the PILOT-ERA fences stand on generate_plots.py / run_phase2_stats.sh.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = REPO_ROOT / "scripts" / "4_analysis"
for _p in (str(_SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import organize_results as org  # noqa: E402
import run_campaign_analysis as rca  # noqa: E402
from src.analysis.cellspec import CellSpec  # noqa: E402
from src.analysis.stats.blinding import scramble_labels  # noqa: E402
from src.analysis.stats.ledger import hash_artifacts, write_ledger  # noqa: E402

RUN_ID = "20260802-1400-a-qwen3-14b"
CAMPAIGN = "camp1"
SESSION = "a"
MODEL = "qwen3-14b"
DATASETS = ["squad_v2", "hotpotqa"]
CELL_BASELINES = ["B1", "B3", "B6"]  # B6 vs B3 = the default headline contrast #4
WINDOWS_PER_DATASET = 2
#: Large enough that a tie-dominated conditional population can pass the §9.5
#: dominance layer (bootstrap CI inside ±0.147) — the pilot's real data shape.
N_EXAMPLES = 64

#: Deterministic per-baseline serving/quality offsets: B6 (RAG) pays TTFT vs
#: B3 (CAG) on every example -> the paired Wilcoxon has an unambiguous sign.
TTFT_OFFSET = {"B1": 200.0, "B3": 140.0, "B6": 235.0, "B11": 180.0}
F1_OFFSET = {"B1": 0.80, "B3": 0.62, "B6": 0.71, "B11": 0.66}

WINDOW_ARTIFACTS = (
    "requests.jsonl",
    "qa_evidence.jsonl",
    "engine_metrics.json",
    "cage_stats.jsonl",
)

EvidenceExtra = Callable[[int], dict[str, Any]]


def _specs() -> list[CellSpec]:
    return [CellSpec.from_baseline(b, model=MODEL) for b in CELL_BASELINES]  # type: ignore[arg-type]


def _write_window(
    wdir: Path,
    dataset: str,
    baseline: str,
    *,
    ordinal: int,
    evidence_extra: EvidenceExtra | None = None,
) -> list[Path]:
    """§1 artifact set with REAL per-query metrics keyed by example_id."""
    wdir.mkdir(parents=True)
    written: list[Path] = []
    requests_lines = []
    evidence_lines = []
    for i in range(N_EXAMPLES):
        example_id = f"{dataset}-e{i:03d}"
        ttft = TTFT_OFFSET[baseline] + 1.7 * i + 0.3 * ordinal
        f1 = min(1.0, F1_OFFSET[baseline] + 0.01 * (i % 5))
        requests_lines.append(
            json.dumps({"example_id": example_id, "ttft_ms": ttft, "latency_ms": ttft + 50.0})
        )
        extra = evidence_extra(i) if evidence_extra is not None else {}
        evidence_lines.append(
            json.dumps(
                {"example_id": example_id, "f1_score": f1, "answer": "text", **extra}
            )
        )
    payloads = {
        "requests.jsonl": "\n".join(requests_lines) + "\n",
        "qa_evidence.jsonl": "\n".join(evidence_lines) + "\n",
        "engine_metrics.json": json.dumps({"snapshot": "before/after"}),
        "cage_stats.jsonl": json.dumps({"t": 0, "kv_bytes": 1}) + "\n",
    }
    for name in WINDOW_ARTIFACTS:
        path = wdir / name
        path.write_text(payloads[name], encoding="utf-8")
        written.append(path)
    return written


def _build_run_tree(
    tmp_path: Path,
    *,
    extra_specs: list[CellSpec] | None = None,
    special_specs: list[tuple[CellSpec, EvidenceExtra]] | None = None,
) -> Path:
    """Synthetic organized-ready run tree per RESULTS_LAYOUT §1, sealed."""
    run_dir = tmp_path / "results" / CAMPAIGN / SESSION / RUN_ID
    run_dir.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "campaign": CAMPAIGN,
        "session": SESSION,
        "run_id": RUN_ID,
        "model": MODEL,
        "git_sha": "deadbeef",
        "engine": "vllm",
        "seed": 1,
        "created_utc": "2026-08-02T14:00:00Z",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    sealed: list[Path] = []
    all_specs: list[tuple[CellSpec, EvidenceExtra | None]] = [
        (spec, None) for spec in _specs() + list(extra_specs or [])
    ]
    all_specs.extend((spec, fn) for spec, fn in (special_specs or []))
    for spec, evidence_extra in all_specs:
        baseline = org.BASELINE_OF_CELL.get((spec.arm, spec.retriever), "")
        cell_dir = run_dir / "cells" / spec.to_row_key()
        cell_dir.mkdir(parents=True)
        windows: dict[str, dict[str, Any]] = {}
        for dataset in DATASETS:
            for ordinal in range(1, WINDOWS_PER_DATASET + 1):
                k = f"{dataset}-{ordinal:02d}"
                windows[k] = {"dataset": dataset, "seed": 1, "rep": ordinal}
                sealed.extend(
                    _write_window(
                        cell_dir / f"window_{k}",
                        dataset,
                        baseline or "B1",
                        ordinal=ordinal,
                        evidence_extra=evidence_extra,
                    )
                )
        cell_json = cell_dir / "cell.json"
        cell_json.write_text(
            json.dumps(
                {
                    "cellspec": spec.to_flat_dict(),
                    "baseline": baseline,
                    "windows": windows,
                }
            ),
            encoding="utf-8",
        )
        sealed.append(cell_json)
    write_ledger(hash_artifacts(sealed, base_dir=run_dir), run_dir / "ledger.json")
    return run_dir


def _f2_spec() -> CellSpec:
    """A pressure cell (family F2, coords set) — must never be per-query paired."""
    return CellSpec(
        arm="gold-fresh",
        retriever="none",
        policy="none",
        topology="single",
        engine="vllm",
        model=MODEL,  # type: ignore[arg-type]
        family="F2",
        budget_r=0.5,
        rate_frac=0.9,
    )


def _f2_pair_specs() -> list[CellSpec]:
    """B11 vs B6 under pressure (contrast #15's registered F2 leg)."""
    return [
        CellSpec.from_baseline(
            "B11", model=MODEL, family="F2", budget_r=0.5, rate_frac=0.8  # type: ignore[arg-type]
        ),
        CellSpec.from_baseline(
            "B6", model=MODEL, family="F2", budget_r=0.5, rate_frac=0.8  # type: ignore[arg-type]
        ),
    ]


def _write_calibration_report(path: Path, *, passing: bool = True) -> Path:
    """A §9.7 CalibrationReport JSON artifact (A/A + one injection)."""
    ci = [0.01, 0.09] if passing else [0.10, 0.20]  # nominal α=0.05 in/out of CI
    payload = {
        "seed": 7,
        "n_observations": 128,
        "aa": {
            "n_splits": 200,
            "alpha": 0.05,
            "n_rejections": 8,
            "fp_rate": 0.04,
            "ci_low": ci[0],
            "ci_high": ci[1],
        },
        "injections": [
            {
                "effect_size": 5.0,
                "kind": "shift",
                "n_splits": 200,
                "alpha": 0.05,
                "n_rejections": 170,
                "power": 0.85,
                "ci_low": 0.79,
                "ci_high": 0.90,
                "target_power": 0.8,
            }
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _confirmatory_argv(run_dir: Path, calibration: Path, *extra: str) -> list[str]:
    return [
        str(run_dir),
        "--confirmatory",
        "--i-understand-one-look",
        "--registered-sha",
        "prereg-sha-abc123",
        "--calibration-report",
        str(calibration),
        *extra,
    ]


@pytest.fixture()
def organized_run(tmp_path: Path) -> Path:
    run_dir = _build_run_tree(tmp_path)
    org.organize_run(run_dir)
    return run_dir


@pytest.fixture()
def organized_run_with_f2(tmp_path: Path) -> Path:
    run_dir = _build_run_tree(tmp_path, extra_specs=[_f2_spec()])
    org.organize_run(run_dir)
    return run_dir


@pytest.fixture()
def calibration_ok(tmp_path: Path) -> Path:
    return _write_calibration_report(tmp_path / "calibration_report.json")


def _load_stats(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    dirs = sorted((run_dir / "analysis").iterdir())
    assert dirs, "no analysis/<timestamp>/ directory was created"
    analysis_dir = dirs[-1]
    stats_path = analysis_dir / "stats.json"
    assert stats_path.is_file()
    return analysis_dir, json.loads(stats_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Design-input mode (the default)
# ---------------------------------------------------------------------------


def test_design_input_default_stats_figures_and_stamp(organized_run: Path) -> None:
    rc = rca.main([str(organized_run)])
    assert rc == 0
    analysis_dir, stats = _load_stats(organized_run)

    assert stats["mode_stamp"] == "DESIGN-INPUT-ONLY"
    assert stats["schema_version"] == 2
    assert stats["run"]["run_id"] == RUN_ID
    assert stats["one_look"] == {
        "mode": "design-input",
        "registered_sha": None,
        "lock_file": None,
    }
    assert stats["requested_contrast_ids"] == [4]
    # Preconditions are confirmatory-mode gates; design-input records them
    # unchecked (never silently green).
    assert stats["preconditions"]["ledger"] == {"checked": False}
    assert stats["preconditions"]["calibration"] == {"checked": False}
    # The §9.3 family map is compiled for this run's group/datasets.
    assert stats["family_map"]["group"] == "A"
    assert set(stats["family_map"]["datasets"]) == set(DATASETS)
    assert stats["family_map"]["n_rows"] > 0

    # The headline contrast, one entry per metric (default: ttft_ms).
    assert len(stats["contrasts"]) == 1
    entry = stats["contrasts"][0]
    assert entry["contrast_id"] == 4
    assert entry["cell_baseline"] == "B6"
    assert entry["reference_baseline"] == "B3"
    assert entry["metric"] == "ttft_ms"
    assert entry["unit"] == "per_query"
    assert entry["test"] == "paired_wilcoxon"
    assert entry["higher_is_better"] is False
    # Contrast #4 is tier="primary" (§9.1 co-primary SET): NO cross-dataset
    # correction — full alpha per dataset, pooling prohibited.
    assert entry["tier"] == "primary"
    assert "primary tier" in entry["correction"]
    assert {d["dataset"] for d in entry["per_dataset"]} == set(DATASETS)
    for row in entry["per_dataset"]:
        assert row["n_pairs"] == N_EXAMPLES
        assert row["wins"] + row["losses"] + row["ties"] == row["n_pairs"]
        # B6 pays TTFT on every example -> all losses, positive median delta.
        assert row["losses"] == N_EXAMPLES
        assert row["median_delta"] > 0
        assert 0.0 <= row["p_value"] <= 1.0
        assert row["p_holm_across_datasets"] is None
        assert -1.0 <= row["cliffs_delta_paired"] <= 1.0
        assert row["in_family_map"] is True

    # The §9.3 chain ran on the computed primary, flagged INCOMPLETE loudly.
    gate = stats["gatekeeping"]
    assert gate["chain_complete"] is False
    assert gate["primary_chain_order_executed"] == ["contrast-4"]
    assert set(gate["missing_primary_endpoints"]) == {"contrast-14", "contrast-13"}
    assert "INCOMPLETE" in gate["missing_endpoints_note"]
    assert {p["endpoint"] for p in gate["primaries"]} == {"contrast-4"}
    for p in gate["primaries"]:
        assert p["status"] == "confirmatory"
        assert p["passed"] is True  # all-losses TTFT -> tiny p

    # §9.5 equivalence legs are DECLARED and labeled-skipped without a margin.
    equiv = stats["equivalence"]
    assert {leg["policy"] for leg in equiv["declared_legs"]} == {
        "recompute",
        "offload",
        "distribute",
    }
    assert equiv["results"] == []
    assert {s["policy"] for s in equiv["skipped"]} == {
        "recompute",
        "offload",
        "distribute",
    }

    # No sealed map -> blinding inactive.
    assert stats["blinding"]["sealed_map"] is None
    assert stats["blinding"]["active"] is False

    # Figures rendered and recorded.
    assert set(stats["figures"]) == {"forest_ttft_ms.png", "wlt_ttft_ms.png"}
    for name in stats["figures"]:
        path = analysis_dir / name
        assert path.is_file() and path.stat().st_size > 0

    # summary.md is stamped and readable.
    summary = (analysis_dir / "summary.md").read_text(encoding="utf-8")
    assert "DESIGN-INPUT-ONLY" in summary
    assert "Contrast #4" in summary
    assert "Gatekeeping chain" in summary
    # No lock in design-input mode; repeatable.
    assert not (organized_run / "analysis_lock.json").exists()
    assert rca.main([str(organized_run)]) == 0


def test_quality_metric_from_qa_evidence_joins(organized_run: Path) -> None:
    rc = rca.main([str(organized_run), "--metrics", "f1_score"])
    assert rc == 0
    _, stats = _load_stats(organized_run)
    entry = stats["contrasts"][0]
    assert entry["metric"] == "f1_score"
    assert entry["higher_is_better"] is True
    for row in entry["per_dataset"]:
        # B6's f1 offset beats B3's on every example.
        assert row["wins"] == N_EXAMPLES


def test_missing_index_refuses_and_names_organizer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = _build_run_tree(tmp_path)  # NOT organized: no index/
    rc = rca.main([str(run_dir)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "organize_results" in err
    assert not (run_dir / "analysis").exists()


# ---------------------------------------------------------------------------
# One-look policy (§9.11) + confirmatory preconditions (§9.7 / §9.10)
# ---------------------------------------------------------------------------


def test_confirmatory_without_flags_refuses(
    organized_run: Path, calibration_ok: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for argv in (
        [str(organized_run), "--confirmatory"],
        [str(organized_run), "--confirmatory", "--i-understand-one-look"],
        [str(organized_run), "--confirmatory", "--registered-sha", "abc123"],
        # All intent flags but NO calibration report: §9.7 refusal.
        [
            str(organized_run),
            "--confirmatory",
            "--i-understand-one-look",
            "--registered-sha",
            "abc123",
        ],
    ):
        rc = rca.main(argv)
        assert rc == 1, argv
        err = capsys.readouterr().err
        assert "REFUSED" in err
    # Nothing computed, no lock written.
    assert not (organized_run / "analysis").exists()
    assert not (organized_run / "analysis_lock.json").exists()


def test_confirmatory_flags_without_confirmatory_refuse(
    organized_run: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = rca.main([str(organized_run), "--registered-sha", "abc123"])
    assert rc == 1
    assert "confirmatory" in capsys.readouterr().err


def test_confirmatory_refuses_failing_calibration(
    organized_run: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = _write_calibration_report(tmp_path / "cal_bad.json", passing=False)
    rc = rca.main(_confirmatory_argv(organized_run, bad))
    assert rc == 1
    err = capsys.readouterr().err
    assert "BLOCKED" in err and "A/A" in err
    # The refusal happens BEFORE the lock: the one-look budget is unspent.
    assert not (organized_run / "analysis_lock.json").exists()
    assert not (organized_run / "analysis").exists()


def test_confirmatory_refuses_malformed_calibration(
    organized_run: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "cal_malformed.json"
    bad.write_text(json.dumps({"seed": 1}), encoding="utf-8")
    rc = rca.main(_confirmatory_argv(organized_run, bad))
    assert rc == 1
    assert "schema" in capsys.readouterr().err
    assert not (organized_run / "analysis_lock.json").exists()


def test_confirmatory_refuses_tampered_ledger(
    organized_run: Path, calibration_ok: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Tamper with one sealed artifact AFTER organize: §9.10 must catch it.
    victim = next(iter((organized_run / "cells").glob("*/window_*/requests.jsonl")))
    victim.write_text(victim.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    rc = rca.main(_confirmatory_argv(organized_run, calibration_ok))
    assert rc == 1
    err = capsys.readouterr().err
    assert "LEDGER PRECONDITION FAILED" in err
    assert not (organized_run / "analysis_lock.json").exists()


def test_confirmatory_runs_once_then_locks(
    organized_run: Path, calibration_ok: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _confirmatory_argv(organized_run, calibration_ok)
    assert rca.main(argv) == 0
    analysis_dir, stats = _load_stats(organized_run)
    assert stats["mode_stamp"] == "CONFIRMATORY"
    assert stats["one_look"]["mode"] == "confirmatory"
    assert stats["one_look"]["registered_sha"] == "prereg-sha-abc123"
    # Preconditions recorded: ledger verified + calibration PASS.
    assert stats["preconditions"]["ledger"]["verified"] is True
    assert stats["preconditions"]["ledger"]["mismatches"] == 0
    assert stats["preconditions"]["calibration"]["verdict"] == "PASS"
    # Gatekeeping trace present in the ONE confirmatory output.
    assert stats["gatekeeping"]["primary_chain_order_executed"] == ["contrast-4"]
    assert "CONFIRMATORY" in (analysis_dir / "summary.md").read_text(encoding="utf-8")

    lock_path = organized_run / "analysis_lock.json"
    assert lock_path.is_file()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["registered_sha"] == "prereg-sha-abc123"
    assert lock["analysis_dir"] == analysis_dir.name
    capsys.readouterr()

    # The second look REFUSES and creates nothing new.
    n_dirs_before = len(list((organized_run / "analysis").iterdir()))
    rc = rca.main(argv)
    assert rc == 1
    err = capsys.readouterr().err
    assert "ONE-LOOK" in err
    assert len(list((organized_run / "analysis").iterdir())) == n_dirs_before


# ---------------------------------------------------------------------------
# §9.3 gatekeeping: registered Holm-within-family for secondaries
# ---------------------------------------------------------------------------


def test_gatekeeping_gates_secondaries_with_holm_within_family(
    organized_run: Path,
) -> None:
    rc = rca.main([str(organized_run), "--contrasts", "4", "3"])
    assert rc == 0
    _, stats = _load_stats(organized_run)
    gate = stats["gatekeeping"]
    # #4 passed everywhere (all-losses TTFT) -> the family gates OPEN and #3
    # receives the REGISTERED Holm-within-family correction.
    secondaries = gate["secondaries"]
    assert len(secondaries) == len(DATASETS)
    for s in secondaries:
        assert s["contrast"].startswith("#3 ")
        assert s["status"] == "confirmatory"
        assert s["p_holm_within_family"] is not None
        assert s["family_id"].startswith("A|ttft_ms|")
    # The auditable trace records one gate event per family.
    family_events = [e for e in gate["events"] if not e["family_id"].startswith("primary-chain")]
    assert {e["family_id"] for e in family_events} == {s["family_id"] for s in secondaries}
    for e in family_events:
        assert e["opened"] is True
        assert e["upstream"] == "contrast-4"


# ---------------------------------------------------------------------------
# Guards: pressure rows, unknown ids, selector contrasts, batch means
# ---------------------------------------------------------------------------


def test_f2_rows_produce_labeled_skip_not_numbers(
    organized_run_with_f2: Path,
) -> None:
    rc = rca.main([str(organized_run_with_f2)])
    assert rc == 0
    _, stats = _load_stats(organized_run_with_f2)
    f2_key = _f2_spec().to_row_key()

    block = stats["skipped"]["pressure_rows"]
    assert block is not None
    assert block["label"] == "PRESSURE-ROWS-NOT-IN-A-COMPUTED-CONTRAST"
    assert "batch" in block["reason"].lower()
    assert f2_key in block["row_keys"]
    assert block["n_windows"] == len(DATASETS) * WINDOWS_PER_DATASET

    # The F2 row never enters any computed contrast.
    for entry in stats["contrasts"]:
        assert f2_key not in (entry["cell_row_key"], entry["reference_row_key"])
    # And the skip is visible in the human summary too.
    dirs = sorted((organized_run_with_f2 / "analysis").iterdir())
    summary = (dirs[-1] / "summary.md").read_text(encoding="utf-8")
    assert f2_key in summary


def test_unknown_contrast_id_fails_loud(
    organized_run: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = rca.main([str(organized_run), "--contrasts", "99"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "unknown" in err.lower() and "99" in err
    assert not (organized_run / "analysis").exists()


def test_selector_contrast_is_skipped_labeled(organized_run: Path) -> None:
    # #14 (serving-yield, window unit) has NO single baseline pair — its
    # estimand selector is not driver-computable; it must skip, labeled.
    rc = rca.main([str(organized_run), "--contrasts", "14"])
    assert rc == 0
    _, stats = _load_stats(organized_run)
    assert stats["contrasts"] == []
    assert stats["figures"] == []
    skipped = stats["skipped"]["contrasts"]
    assert len(skipped) == 1
    assert skipped[0]["contrast_id"] == 14
    assert skipped[0]["label"] == "NOT-IMPLEMENTED-YET"
    assert "baseline pair" in skipped[0]["reason"]


def test_window_contrast_computes_batch_means(tmp_path: Path) -> None:
    # #15 (B11 vs B6, F2) is a window-unit baseline pair: with loaded-window
    # rows present it computes batch means (§9.4), never per-query pairing.
    run_dir = _build_run_tree(tmp_path, extra_specs=_f2_pair_specs())
    org.organize_run(run_dir)
    rc = rca.main([str(run_dir), "--contrasts", "15"])
    assert rc == 0
    _, stats = _load_stats(run_dir)

    entries = stats["contrasts"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["contrast_id"] == 15
    assert entry["unit"] == "window"
    assert "batch_means" in entry["test"]
    assert entry["family"] == "F2"
    for row in entry["per_dataset"]:
        assert row["n_windows_cell"] == WINDOWS_PER_DATASET
        assert row["n_windows_reference"] == WINDOWS_PER_DATASET
        # B11 saves TTFT vs B6 by construction (180 vs 235 offsets).
        assert row["mean_diff"] < 0
        assert 0.0 <= row["p_value"] <= 1.0
        assert row["ci95_low"] <= row["mean_diff"] <= row["ci95_high"]

    # The consumed F2 rows are NOT in the pressure skip block…
    block = stats["skipped"]["pressure_rows"]
    assert block is None
    # …and the absent extra leg (B12 vs B3, F3) is a labeled skip.
    leg_skips = [
        s for s in stats["skipped"]["contrasts"] if s["contrast_id"] == 15
    ]
    assert len(leg_skips) == 1
    assert leg_skips[0]["label"] == "NO-WINDOW-PAIR-IN-RUN"
    assert "B12" in leg_skips[0]["reason"]


def test_multiple_f1_contrasts_share_reference_grouping(organized_run: Path) -> None:
    # #4 (B6 vs B3) + #3 (B1 vs B6): two references -> per-reference forests.
    rc = rca.main([str(organized_run), "--contrasts", "4", "3"])
    assert rc == 0
    analysis_dir, stats = _load_stats(organized_run)
    assert {e["contrast_id"] for e in stats["contrasts"]} == {3, 4}
    names = set(stats["figures"])
    assert "wlt_ttft_ms.png" in names
    forest_names = {n for n in names if n.startswith("forest_")}
    assert forest_names == {"forest_ttft_ms__vs_B3.png", "forest_ttft_ms__vs_B6.png"}
    for name in names:
        assert (analysis_dir / name).stat().st_size > 0


def test_unregistered_metric_direction_fails_closed(
    organized_run: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = rca.main([str(organized_run), "--metrics", "mystery_units"])
    assert rc == 1
    assert "direction" in capsys.readouterr().err
    assert not (organized_run / "analysis").exists()


# ---------------------------------------------------------------------------
# §9.5 conditional TOST equivalence legs
# ---------------------------------------------------------------------------


#: TOST fixture shape: 48 policy-touched examples, only 4 discordant (±0.002,
#: alternating sign), the rest exact ties — the tie-heavy conditional
#: population the §9.5 design targets (pilot: 15/289 discordant).
_TOST_N_EVENTS = 48
_TOST_N_DISCORDANT = 4


def _tost_pair_specs() -> list[tuple[CellSpec, EvidenceExtra]]:
    """policy=recompute vs policy=none under F2, with grounding + event mask.

    Equivalent under a 0.05 margin on BOTH layers: mean diff ~0 (domain) and
    a tie-dominated sign vector whose bootstrap CI sits inside ±0.147
    (dominance)."""
    base = dict(
        arm="gold-fresh", retriever="none", topology="single", engine="vllm",
        model=MODEL, family="F2", budget_r=0.5, rate_frac=0.9,
    )
    policy_cell = CellSpec(policy="recompute", **base)  # type: ignore[arg-type]
    none_cell = CellSpec(policy="none", **base)  # type: ignore[arg-type]

    def policy_extra(i: int) -> dict[str, Any]:
        delta = 0.0
        if i < _TOST_N_EVENTS and i % 12 == 0:  # i in {0, 12, 24, 36}
            delta = 0.002 if (i // 12) % 2 == 0 else -0.002
        return {
            "grounding_score": 0.8 + delta,
            "policy_event": 1 if i < _TOST_N_EVENTS else 0,
        }

    def none_extra(i: int) -> dict[str, Any]:
        return {"grounding_score": 0.8}

    return [(policy_cell, policy_extra), (none_cell, none_extra)]


def test_equivalence_tost_computed_with_margin_and_mask(tmp_path: Path) -> None:
    run_dir = _build_run_tree(tmp_path, special_specs=_tost_pair_specs())
    org.organize_run(run_dir)
    rc = rca.main(
        [
            str(run_dir),
            "--tost-margin", "0.05",
            "--equivalence-metric", "grounding_score",
        ]
    )
    assert rc == 0
    _, stats = _load_stats(run_dir)
    equiv = stats["equivalence"]

    results = equiv["results"]
    assert {r["dataset"] for r in results} == set(DATASETS)
    for r in results:
        assert r["policy"] == "recompute"
        assert r["metric"] == "grounding_score"
        assert r["margin"] == 0.05
        assert r["n_total"] == N_EXAMPLES
        assert r["n_events"] == _TOST_N_EVENTS  # the CONDITIONAL population
        assert r["n_discordant"] == _TOST_N_DISCORDANT
        assert r["domain_verdict"] == "equivalent"
        assert r["dominance_verdict"] == "equivalent"
        assert r["equivalent"] is True
        rope = r["rope_sensitivity"]
        assert rope["verdict"] == "equivalent"
        assert rope["p_rope"] >= 0.95

    # offload has no cells; distribute is a topology-slot leg: labeled skips.
    assert {s["policy"] for s in equiv["skipped"]} == {"offload", "distribute"}


def test_equivalence_skipped_without_event_mask(tmp_path: Path) -> None:
    # Same pair but WITHOUT the policy_event column: the conditional §9.5
    # population is unavailable -> a labeled skip, never an unconditional TOST.
    specs = _tost_pair_specs()
    no_mask: list[tuple[CellSpec, EvidenceExtra]] = [
        (specs[0][0], lambda i: {"grounding_score": 0.8}),
        specs[1],
    ]
    run_dir = _build_run_tree(tmp_path, special_specs=no_mask)
    org.organize_run(run_dir)
    rc = rca.main(
        [
            str(run_dir),
            "--tost-margin", "0.05",
            "--equivalence-metric", "grounding_score",
        ]
    )
    assert rc == 0
    _, stats = _load_stats(run_dir)
    equiv = stats["equivalence"]
    assert equiv["results"] == []
    recompute_skips = [s for s in equiv["skipped"] if s["policy"] == "recompute"]
    assert len(recompute_skips) == 1
    assert "policy_event" in recompute_skips[0]["reason"]


# ---------------------------------------------------------------------------
# §9.8 blinding integration
# ---------------------------------------------------------------------------


def _seal_arm_map(run_dir: Path) -> Path:
    index = pd.read_csv(run_dir / "index" / "cells_index.csv")
    df = pd.DataFrame({"arm": sorted(index["arm"].unique())})
    sealed_path = run_dir / "blinding" / "sealed_arm_map.json"
    scramble_labels(df, "arm", seed=11, sealed_map_path=sealed_path)
    return sealed_path


def test_blinded_design_input_masks_labels_and_suppresses_figures(
    organized_run: Path,
) -> None:
    _seal_arm_map(organized_run)
    rc = rca.main([str(organized_run)])
    assert rc == 0
    _, stats = _load_stats(organized_run)
    blinding = stats["blinding"]
    assert blinding["active"] is True
    assert blinding["sealed_map"] == "blinding/sealed_arm_map.json"
    assert blinding["unblind_event"] is None
    # Arm-revealing labels are masked; figures suppressed until unblinding.
    entry = stats["contrasts"][0]
    for field in ("cell_row_key", "reference_row_key", "cell_baseline", "reference_baseline"):
        assert entry[field].startswith("BLINDED:ARM-")
    assert stats["figures"] == []
    # The seal itself is untouched (no unblinding happened).
    sealed = json.loads(
        (organized_run / "blinding" / "sealed_arm_map.json").read_text(encoding="utf-8")
    )
    assert sealed["unblinded_utc"] is None


def test_confirmatory_records_one_time_unblinding(
    organized_run: Path, calibration_ok: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _seal_arm_map(organized_run)
    assert rca.main(_confirmatory_argv(organized_run, calibration_ok)) == 0
    _, stats = _load_stats(organized_run)
    event = stats["blinding"]["unblind_event"]
    assert event is not None
    assert event["log"] == "blinding/unblind_log.jsonl"
    # Real labels in the frozen confirmatory output (unblinding just happened).
    entry = stats["contrasts"][0]
    assert entry["cell_baseline"] == "B6"
    # The log carries exactly one dated UNBLIND event; the seal is stamped.
    log_lines = (
        (organized_run / "blinding" / "unblind_log.jsonl")
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    )
    assert len(log_lines) == 1
    assert json.loads(log_lines[0])["event"] == "UNBLIND"
    sealed = json.loads(
        (organized_run / "blinding" / "sealed_arm_map.json").read_text(encoding="utf-8")
    )
    assert sealed["unblinded_utc"] is not None
    capsys.readouterr()

    # After unblinding, design-input runs are NOT blinded (mapping is public).
    assert rca.main([str(organized_run)]) == 0
    _, stats2 = _load_stats(organized_run)
    assert stats2["blinding"]["active"] is False


# ---------------------------------------------------------------------------
# Legacy fences (banner-only) on the pilot-era analysis entry points
# ---------------------------------------------------------------------------


def test_pilot_era_fences_present() -> None:
    fence = "PILOT-ERA"
    plots = (_SCRIPTS_DIR / "generate_plots.py").read_text(encoding="utf-8")
    stats_sh = (_SCRIPTS_DIR / "run_phase2_stats.sh").read_text(encoding="utf-8")
    for text, name in ((plots, "generate_plots.py"), (stats_sh, "run_phase2_stats.sh")):
        assert fence in text, f"{name} lost its PILOT-ERA fence"
        assert "run_campaign_analysis.py" in text, (
            f"{name} fence must point at run_campaign_analysis.py"
        )


def test_run_phase2_stats_sh_still_parses() -> None:
    proc = subprocess.run(
        ["bash", "-n", str(_SCRIPTS_DIR / "run_phase2_stats.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"bash -n failed:\n{proc.stderr}"
