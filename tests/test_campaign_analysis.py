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
  masking covers the equivalence/fingerprint/pressure sections too (G12);
- unknown contrast ids fail loud; selector contrasts skip, labeled;
- Topic-7 registration binding (2026-08-16 batch): G1 (SHA grammar / HEAD /
  clean worktree / PRE_REGISTRATION.md / alpha / metrics / margins), G2/G3
  (map-row tier routing + separated exploratory BH-FDR), decision a
  (registered per-row sidedness EXECUTED), decision b (pratt + n_nonzero),
  decision c (window-block TOST/ROPE + MIN_UNIQUE_WINDOWS floor), decision d
  (registered upstream topology + registered-m Holm + set completeness
  naming the producer command), G4 (#13 fingerprint executor, #12
  lambda-star executor, #14 truth-tax executor — task #119; the end-to-end
  chain fixtures live in tests/test_predicate_chain.py), G6
  (policy_event exclusion counting), G11 (bool
  coercion + duplicate keys), G14 (atomic outputs + provenance), G16
  (ADR-0086 realized-n ladder);
- the PILOT-ERA fences stand on generate_plots.py / run_phase2_stats.sh.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
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
#: §9.5 pressure fixtures need >= equivalence.MIN_UNIQUE_WINDOWS (= 5, the
#: registered block-bootstrap floor — decision c 2026-08-16).
TOST_WINDOWS_PER_DATASET = 5
#: Large enough that a tie-dominated conditional population can pass the §9.5
#: dominance layer (bootstrap CI inside ±0.147) — the pilot's real data shape.
N_EXAMPLES = 64

#: A well-formed hex registration SHA for the G1 binding fixtures.
REG_SHA = "deadbeefcafef00d"

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
                {
                    "example_id": example_id,
                    "f1_score": f1,
                    # ADR-0087: the demoted per-query continuous metric — the
                    # G2/G3 exploratory-tier tests request it via --metrics.
                    "faithfulness": f1,
                    "answer": "text",
                    **extra,
                }
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
    windows_per_dataset: int = WINDOWS_PER_DATASET,
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
        "git_dirty": False,
        "engine": "vllm",
        "engine_version": "0.0-test",
        "seed": 1,
        "provider": "test",
        "hardware": "test-gpu",
        "dataset_manifests_sha256": "0" * 64,
        "cellspec_schema_version": 1,
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
            for ordinal in range(1, windows_per_dataset + 1):
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
        REG_SHA,
        "--calibration-report",
        str(calibration),
        *extra,
    ]


@pytest.fixture()
def confirmatory_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> dict[str, Path]:
    """The G1 registration-binding environment a confirmatory look demands.

    Simulates the post-#112 frozen state: executing HEAD == REG_SHA on a
    clean worktree, PRE_REGISTRATION.md embedding that SHA, a registered
    margins artifact, and the ADR-0086 ladder scaled to the fixture's
    N_EXAMPLES (dedicated G16 tests pin the REAL registered ladder).
    """
    prereg = tmp_path / "PRE_REGISTRATION.md"
    prereg.write_text(
        "# PRE_REGISTRATION — CAGE campaign (test fixture)\n\n"
        f"Machinery SHA: `{REG_SHA}` (registered rung (b)).\n",
        encoding="utf-8",
    )
    margins = tmp_path / "registered_margins.json"
    margins.write_text(json.dumps({"grounding_score": 0.05}), encoding="utf-8")
    monkeypatch.setattr(
        rca, "_git_head_state", lambda repo_dir=None: (REG_SHA, False)
    )
    monkeypatch.setattr(rca, "PREREG_PATH", prereg)
    monkeypatch.setattr(rca, "REGISTERED_MARGINS_PATH", margins)
    monkeypatch.setattr(rca, "ADR0086_REALIZED_N_LADDER", (N_EXAMPLES,))
    return {"prereg": prereg, "margins": margins}


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
        assert row["realized_n"] == N_EXAMPLES
        assert row["wins"] + row["losses"] + row["ties"] == row["n_pairs"]
        # B6 pays TTFT on every example -> all losses, positive median delta.
        assert row["losses"] == N_EXAMPLES
        assert row["median_delta"] > 0
        assert 0.0 <= row["p_value"] <= 1.0
        assert row["p_holm_across_datasets"] is None
        assert -1.0 <= row["cliffs_delta_paired"] <= 1.0
        assert row["in_family_map"] is True
        # Decision b: the registered tie handling runs on the driver path,
        # with the effective n surfaced beside the W/L/T triple.
        assert row["zero_method"] == "pratt"
        assert row["n_nonzero"] == N_EXAMPLES
    # G14: executing-code provenance + registered seeds stamped.
    prov = stats["provenance"]
    assert prov["bootstrap_seed"] == 42
    assert prov["rope_seed"] == 42
    assert "executing_git_sha" in prov and "executing_git_dirty" in prov

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

    # Figures rendered from the REGISTERED statistics (audit I1) and recorded
    # with metadata: the per-dataset W/L/T panels are the default view, the
    # pooled view is a disclosed supplementary file (audit I2).
    files = {e["file"] for e in stats["figures"] if "file" in e}
    assert files == {
        "forest_ttft_ms.png",
        "wlt_ttft_ms.png",
        "wlt_ttft_ms_pooled_supplementary.png",
    }
    for entry in stats["figures"]:
        assert "file" in entry  # every requested metric rendered — no skips
        assert "stats.json" in entry["source"]
        assert entry["n_dropped_nan_total"] == 0  # I11: counted disclosure
        assert entry["consumed"]  # the figures-agree-with-stats seam
        path = analysis_dir / entry["file"]
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
    organized_run: Path,
    tmp_path: Path,
    confirmatory_env: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
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
    organized_run: Path,
    tmp_path: Path,
    confirmatory_env: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    bad = tmp_path / "cal_malformed.json"
    bad.write_text(json.dumps({"seed": 1}), encoding="utf-8")
    rc = rca.main(_confirmatory_argv(organized_run, bad))
    assert rc == 1
    assert "schema" in capsys.readouterr().err
    assert not (organized_run / "analysis_lock.json").exists()


def test_confirmatory_refuses_tampered_ledger(
    organized_run: Path,
    calibration_ok: Path,
    confirmatory_env: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
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
    organized_run: Path,
    calibration_ok: Path,
    confirmatory_env: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = _confirmatory_argv(organized_run, calibration_ok)
    assert rca.main(argv) == 0
    analysis_dir, stats = _load_stats(organized_run)
    assert stats["mode_stamp"] == "CONFIRMATORY"
    assert stats["one_look"]["mode"] == "confirmatory"
    assert stats["one_look"]["registered_sha"] == REG_SHA
    # Preconditions recorded: G1 binding BOUND + ledger verified +
    # calibration PASS + the ADR-0086 realized-n gate.
    assert stats["preconditions"]["registration"]["verdict"] == "BOUND"
    assert stats["preconditions"]["registration"]["executing_git_sha"] == REG_SHA
    assert stats["preconditions"]["ledger"]["verified"] is True
    assert stats["preconditions"]["ledger"]["mismatches"] == 0
    assert stats["preconditions"]["calibration"]["verdict"] == "PASS"
    assert stats["preconditions"]["realized_n"]["checked"] is True
    # G1a: the confirmatory look tests the REGISTERED §9.1 metric pair —
    # never the CLI default.
    assert stats["metrics"] == ["ttft_ms", "predicate"]
    # Gatekeeping trace present in the ONE confirmatory output.
    assert stats["gatekeeping"]["primary_chain_order_executed"] == ["contrast-4"]
    assert "CONFIRMATORY" in (analysis_dir / "summary.md").read_text(encoding="utf-8")

    lock_path = organized_run / "analysis_lock.json"
    assert lock_path.is_file()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["registered_sha"] == REG_SHA
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
    # receives the REGISTERED Holm-within-family correction, at the
    # REGISTERED family size (G5b: the map's F1 per-query family holds 9
    # secondary legs for group A; only #3 is supplied here, so Holm runs at
    # the registered m=9 — the shrunken-m diagnostic was the audited defect).
    secondaries = gate["secondaries"]
    assert len(secondaries) == len(DATASETS)
    for s in secondaries:
        assert s["contrast"].startswith("#3 ")
        assert s["status"] == "confirmatory"
        assert s["p_holm_within_family"] is not None
        # Decision d (G19): the 5-axis registered family id (unit split).
        assert s["family_id"].startswith("A|ttft_ms|")
        assert s["family_id"].endswith("|F1|per_query")
        assert s["m_supplied"] == 1
        assert s["m_registered"] == 9
        assert s["p_holm_within_family"] == pytest.approx(
            min(1.0, 9 * s["p_value"])
        )
    # The auditable trace records one gate event per family.
    family_events = [e for e in gate["events"] if not e["family_id"].startswith("primary-chain")]
    assert {e["family_id"] for e in family_events} == {s["family_id"] for s in secondaries}
    for e in family_events:
        assert e["opened"] is True
        assert e["upstream"] == "contrast-4"
    # Decision a: #3's registered ONE-SIDED execution — B1 (oracle context)
    # is the registered better cell on serving, ttft lower-is-better ->
    # alternative "less"; the "conservative superset" two-sided label is gone.
    entry3 = next(e for e in stats["contrasts"] if e["contrast_id"] == 3)
    assert entry3["registered_sidedness"] == "one-sided"
    assert entry3["executed_alternative"] == "less"
    assert "conservative superset" not in json.dumps(stats)


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
    # #10 (engine slot) has NO single baseline pair — its selector is not
    # driver-computable; it must skip, labeled.
    rc = rca.main([str(organized_run), "--contrasts", "10"])
    assert rc == 0
    _, stats = _load_stats(organized_run)
    assert stats["contrasts"] == []
    assert stats["figures"] == []
    skipped = stats["skipped"]["contrasts"]
    assert len(skipped) == 1
    assert skipped[0]["contrast_id"] == 10
    assert skipped[0]["label"] == "NOT-IMPLEMENTED-YET"
    assert "baseline pair" in skipped[0]["reason"]


def test_contrast_14_pre_predicate_refusal_no_longer_names_119_unbuilt(
    organized_run: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Task #119 pin: the executor replaced the stub. On a tree WITHOUT a
    # predicate table the request still FAILS LOUD (a missing chain PRIMARY
    # input is never a silent skip) — but the refusal now names the PRODUCER
    # COMMAND, not "#119 has not landed" (the producer exists).
    rc = rca.main([str(organized_run), "--contrasts", "14"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "truth_tax" in err or "truth-tax" in err
    assert "build_predicate_table.py" in err  # the fix, named
    assert "has not landed" not in err  # the unbuilt claim is GONE
    assert not (organized_run / "analysis").exists()


def test_driver_source_no_longer_claims_119_unbuilt() -> None:
    # Task #119 pin: no refusal in the driver may still claim the §8.5
    # predicate producer "has not landed" — it landed with this build.
    source = (
        REPO_ROOT / "scripts" / "4_analysis" / "run_campaign_analysis.py"
    ).read_text(encoding="utf-8")
    for chunk in source.split('"'):
        if "#119" in chunk:
            assert "has not landed" not in chunk


def test_contrast_12_lambda_star_fails_loud_naming_missing_inputs(
    organized_run: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # G4b: #12 requested on a run without the §6.1 rate grid must name the
    # missing artifact, never emit a silent skip.
    rc = rca.main([str(organized_run), "--contrasts", "12"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "MISSING ARTIFACT" in err
    assert "rate_frac" in err


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

    # I11: a metric with no per-query contrast entry renders no figure — a
    # COUNTED skip entry in the figures list, never a silent omission.
    assert len(stats["figures"]) == 1
    fig_skip = stats["figures"][0]
    assert fig_skip["skipped_metric"] == "ttft_ms"
    assert "window-unit" in fig_skip["reason"]

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
    names = {e["file"] for e in stats["figures"] if "file" in e}
    assert "wlt_ttft_ms.png" in names
    assert "wlt_ttft_ms_pooled_supplementary.png" in names
    forest_names = {n for n in names if n.startswith("forest_")}
    assert forest_names == {"forest_ttft_ms__vs_B3.png", "forest_ttft_ms__vs_B6.png"}
    for name in names:
        assert (analysis_dir / name).stat().st_size > 0


def test_figures_agree_with_stats_bit_for_bit(organized_run: Path) -> None:
    # #131 item 6 (audit I1): every statistic a published figure consumed must
    # equal the stats.json value BIT-FOR-BIT. render_figures feeds the
    # renderers from the same dicts serialized into stats['contrasts'] and
    # records the consumed values per figure — the two are compared here on
    # the loaded JSON, so any divergence (recomputation, rounding, drift)
    # fails this test.
    rc = rca.main([str(organized_run), "--contrasts", "4", "3"])
    assert rc == 0
    _, stats = _load_stats(organized_run)

    registered: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for entry in stats["contrasts"]:
        for row in entry["per_dataset"]:
            key = (
                entry["cell_row_key"],
                entry["reference_row_key"],
                entry["metric"],
                row["dataset"],
            )
            assert key not in registered
            registered[key] = row

    rendered = [e for e in stats["figures"] if "file" in e]
    assert rendered, "no figure was rendered"
    for fig_entry in rendered:
        assert fig_entry["consumed"], fig_entry["file"]
        for consumed in fig_entry["consumed"]:
            row = registered[
                (
                    consumed["cell_row_key"],
                    consumed["reference_row_key"],
                    fig_entry["metric"],
                    consumed["dataset"],
                )
            ]
            assert consumed["p_value"] == row["p_value"]
            assert consumed["p_corrected"] == row["p_holm_across_datasets"]
            assert consumed["median_delta"] == row["median_delta"]
            assert consumed["n_pairs"] == row["n_pairs"]
            assert consumed["n_dropped_nan"] == row["n_dropped_nan"]
            assert (
                consumed["wins"], consumed["losses"], consumed["ties"]
            ) == (row["wins"], row["losses"], row["ties"])
    # The W/L/T figures consumed EVERY registered per-query row (nothing
    # silently omitted); each forest consumed its reference's rows.
    wlt = next(e for e in rendered if e["file"] == "wlt_ttft_ms.png")
    assert len(wlt["consumed"]) == len(registered)


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


#: TOST fixture shape: 48 policy-touched examples PER WINDOW, only 4
#: discordant per window (±0.002, alternating sign), the rest exact ties —
#: the tie-heavy conditional population the §9.5 design targets (pilot:
#: 15/289 discordant). Decision c (2026-08-16): pairing is per
#: (example_id, window_key), so counts scale with TOST_WINDOWS_PER_DATASET.
_TOST_N_EVENTS_PER_WINDOW = 48
_TOST_N_DISCORDANT_PER_WINDOW = 4


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
        if i < _TOST_N_EVENTS_PER_WINDOW and i % 12 == 0:  # i in {0,12,24,36}
            delta = 0.002 if (i // 12) % 2 == 0 else -0.002
        return {
            "grounding_score": 0.8 + delta,
            "policy_event": 1 if i < _TOST_N_EVENTS_PER_WINDOW else 0,
        }

    def none_extra(i: int) -> dict[str, Any]:
        return {"grounding_score": 0.8}

    return [(policy_cell, policy_extra), (none_cell, none_extra)]


def test_equivalence_tost_computed_with_margin_and_mask(tmp_path: Path) -> None:
    # Decision c (2026-08-16): the §9.5 pressure legs pair per
    # (example, window) and resample WINDOWS (block bootstrap) — this pins
    # the new registered behavior; the old test pinned the per-example
    # resampling path (the G18/G13 defect this decision closes).
    run_dir = _build_run_tree(
        tmp_path,
        special_specs=_tost_pair_specs(),
        windows_per_dataset=TOST_WINDOWS_PER_DATASET,
    )
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

    w = TOST_WINDOWS_PER_DATASET
    results = equiv["results"]
    assert {r["dataset"] for r in results} == set(DATASETS)
    for r in results:
        assert r["policy"] == "recompute"
        assert r["metric"] == "grounding_score"
        assert r["margin"] == 0.05
        assert r["n_total"] == N_EXAMPLES * w
        assert r["n_policy_event_missing"] == 0
        # The CONDITIONAL population, per (example, window).
        assert r["n_events"] == _TOST_N_EVENTS_PER_WINDOW * w
        assert r["n_discordant"] == _TOST_N_DISCORDANT_PER_WINDOW * w
        # Decision c: window-block resampling is ACTIVE on the §9.5 path.
        assert r["resampling"] == "window-block"
        assert r["n_windows"] == w
        assert r["domain_verdict"] == "equivalent"
        assert r["dominance_verdict"] == "equivalent"
        assert r["equivalent"] is True
        rope = r["rope_sensitivity"]
        assert rope["verdict"] == "equivalent"
        assert rope["p_rope"] >= 0.95
        assert rope["resampling"] == "window-block"

    # offload has no cells; distribute is a topology-slot leg: labeled skips.
    assert {s["policy"] for s in equiv["skipped"]} == {"offload", "distribute"}


def test_equivalence_window_floor_refuses_below_five_windows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The registered MIN_UNIQUE_WINDOWS=5 floor (decision c): a pressure run
    # with only 2 windows REFUSES the leg fail-loud — never a silent
    # degradation to per-example resampling.
    run_dir = _build_run_tree(
        tmp_path, special_specs=_tost_pair_specs(), windows_per_dataset=2
    )
    org.organize_run(run_dir)
    rc = rca.main(
        [
            str(run_dir),
            "--tost-margin", "0.05",
            "--equivalence-metric", "grounding_score",
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "unique windows" in err
    assert "recompute" in err


def test_equivalence_excludes_missing_policy_event_rows(tmp_path: Path) -> None:
    # G6: rows whose policy_event telemetry is ABSENT are EXCLUDED with a
    # counted, labeled reason — the old .fillna(0.0) silently declared
    # "no event" on missing telemetry.
    base = dict(
        arm="gold-fresh", retriever="none", topology="single", engine="vllm",
        model=MODEL, family="F2", budget_r=0.5, rate_frac=0.9,
    )
    policy_cell = CellSpec(policy="recompute", **base)  # type: ignore[arg-type]
    none_cell = CellSpec(policy="none", **base)  # type: ignore[arg-type]
    n_missing_examples = 8

    def policy_extra(i: int) -> dict[str, Any]:
        delta = 0.0
        if i < 40 and i % 12 == 0:
            delta = 0.002 if (i // 12) % 2 == 0 else -0.002
        extra: dict[str, Any] = {"grounding_score": 0.8 + delta}
        if 40 <= i < 40 + n_missing_examples:
            return extra  # MISSING policy_event telemetry — not zero
        extra["policy_event"] = 1 if i < 40 else 0
        return extra

    def none_extra(i: int) -> dict[str, Any]:
        return {"grounding_score": 0.8}

    run_dir = _build_run_tree(
        tmp_path,
        special_specs=[(policy_cell, policy_extra), (none_cell, none_extra)],
        windows_per_dataset=TOST_WINDOWS_PER_DATASET,
    )
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
    w = TOST_WINDOWS_PER_DATASET
    results = stats["equivalence"]["results"]
    assert {r["dataset"] for r in results} == set(DATASETS)
    for r in results:
        # 8 examples x w windows excluded AND counted — never mask=False.
        assert r["n_policy_event_missing"] == n_missing_examples * w
        assert r["n_total"] == (N_EXAMPLES - n_missing_examples) * w
        assert r["n_events"] == 40 * w
        assert r["n_discordant"] == 4 * w


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
    organized_run: Path,
    calibration_ok: Path,
    confirmatory_env: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
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
# H4 regression: NaN pressure-coordinate pairing + index schema guards
# ---------------------------------------------------------------------------


def _f2_pair_specs_unset_coords() -> list[CellSpec]:
    """Contrast #15's F2 leg with UNSET pressure coords — legal per CellSpec
    (budget_r/rate_frac are ``float | None``; only F1 forbids SETTING them)."""
    return [
        CellSpec.from_baseline("B11", model=MODEL, family="F2"),  # type: ignore[arg-type]
        CellSpec.from_baseline("B6", model=MODEL, family="F2"),  # type: ignore[arg-type]
    ]


def test_window_pair_with_unset_coords_pairs(tmp_path: Path) -> None:
    # H4 regression: these legal F2 cells index budget_r/rate_frac as NaN;
    # groupby(dropna=False) tuple keys holding NaN never matched across the
    # two group dicts (NaN != NaN), so the pair was silently skipped as
    # "no legal pair". It MUST compute.
    run_dir = _build_run_tree(tmp_path, extra_specs=_f2_pair_specs_unset_coords())
    org.organize_run(run_dir)
    index = pd.read_csv(run_dir / "index" / "cells_index.csv")
    f2 = index[index["family"] == "F2"]
    assert f2["budget_r"].isna().all() and f2["rate_frac"].isna().all()

    rc = rca.main([str(run_dir), "--contrasts", "15"])
    assert rc == 0
    _, stats = _load_stats(run_dir)
    entries = stats["contrasts"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["contrast_id"] == 15
    assert entry["unit"] == "window"
    assert entry["family"] == "F2"
    for row in entry["per_dataset"]:
        assert row["n_windows_cell"] == WINDOWS_PER_DATASET
        assert row["n_windows_reference"] == WINDOWS_PER_DATASET
        # B11 saves TTFT vs B6 by construction (180 vs 235 offsets).
        assert row["mean_diff"] < 0
    # The paired F2 rows were CONSUMED, not dumped into the pressure skip.
    assert stats["skipped"]["pressure_rows"] is None


def _window_index_frame(
    entries: list[tuple[str, float | None, float | None]]
) -> pd.DataFrame:
    """Minimal index frame for select_window_pairs: one row per dataset/cell."""
    arm_retr = {"B11": ("retr-trunc", "rerank"), "B6": ("retr-fresh", "rerank")}
    rows: list[dict[str, Any]] = []
    for baseline, budget_r, rate_frac in entries:
        arm, retriever = arm_retr[baseline]
        spec = CellSpec(
            arm=arm, retriever=retriever, policy="none", topology="single",  # type: ignore[arg-type]
            engine="vllm", model=MODEL, family="F2",  # type: ignore[arg-type]
            budget_r=budget_r, rate_frac=rate_frac,
        )
        for dataset in DATASETS:
            rows.append(
                {
                    "family": "F2",
                    "baseline": baseline,
                    "engine": "vllm",
                    "model": MODEL,
                    "topology": "single",
                    "policy": "none",
                    "budget_r": np.nan if budget_r is None else budget_r,
                    "rate_frac": np.nan if rate_frac is None else rate_frac,
                    "row_key": spec.to_row_key(),
                    "dataset": dataset,
                }
            )
    return pd.DataFrame(rows)


def test_unset_coords_never_pair_with_set_coords() -> None:
    # Absence is NOT a value: a B11 cell with no coords must never pair with
    # a B6 cell at (0.5, 0.8) — only unset-vs-unset matches.
    contrast = rca.CONTRAST_BY_ID[15]
    mixed = _window_index_frame([("B11", None, None), ("B6", 0.5, 0.8)])
    pairs, reasons = rca.select_window_pairs(mixed, contrast)
    assert pairs == []
    assert any("no legal pair" in r for r in reasons)

    both_unset = _window_index_frame([("B11", None, None), ("B6", None, None)])
    pairs, _ = rca.select_window_pairs(both_unset, contrast)
    assert len(pairs) == 1
    assert pairs[0].datasets == tuple(sorted(DATASETS))


def test_equivalence_tost_pairs_with_unset_coords(tmp_path: Path) -> None:
    # The same NaN-key hazard at the §9.5 equivalence matcher: a
    # policy-vs-none F2 pair with UNSET pressure coords must still pair.
    specs = [
        (dc_replace(spec, budget_r=None, rate_frac=None), extra)
        for spec, extra in _tost_pair_specs()
    ]
    run_dir = _build_run_tree(
        tmp_path,
        special_specs=specs,
        windows_per_dataset=TOST_WINDOWS_PER_DATASET,
    )
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
    results = stats["equivalence"]["results"]
    assert {r["dataset"] for r in results} == set(DATASETS)
    for r in results:
        assert r["policy"] == "recompute"
        assert r["equivalent"] is True


@pytest.mark.parametrize(
    "column", ["budget_r", "rate_frac", "cell_json", "artifacts"]
)
def test_index_missing_required_column_refuses(
    organized_run: Path, column: str, capsys: pytest.CaptureFixture[str]
) -> None:
    # The four columns added to _INDEX_REQUIRED_COLUMNS (task #128): the
    # matchers group on budget_r/rate_frac; cell_json/artifacts complete the
    # 20-column INDEX_COLUMNS contract. A truncated index refuses, named.
    index_path = organized_run / "index" / "cells_index.csv"
    index = pd.read_csv(index_path)
    index.drop(columns=[column]).to_csv(index_path, index=False)
    rc = rca.main([str(organized_run)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "missing required columns" in err
    assert column in err
    assert not (organized_run / "analysis").exists()


# ---------------------------------------------------------------------------
# G1: confirmatory registration binding (SHA / worktree / prereg / alpha /
# metrics / margins)
# ---------------------------------------------------------------------------


def _assert_nothing_written(run_dir: Path) -> None:
    assert not (run_dir / "analysis").exists()
    assert not (run_dir / "analysis_lock.json").exists()


def test_confirmatory_registered_sha_hex_grammar_refused(
    organized_run: Path, calibration_ok: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = rca.main(
        [
            str(organized_run), "--confirmatory", "--i-understand-one-look",
            "--registered-sha", "NOT-A-SHA!",
            "--calibration-report", str(calibration_ok),
        ]
    )
    assert rc == 1
    assert "hex" in capsys.readouterr().err
    _assert_nothing_written(organized_run)


def test_confirmatory_sha_must_name_executing_head(
    organized_run: Path,
    calibration_ok: Path,
    confirmatory_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        rca, "_git_head_state", lambda repo_dir=None: ("beef" * 10, False)
    )
    rc = rca.main(_confirmatory_argv(organized_run, calibration_ok))
    assert rc == 1
    err = capsys.readouterr().err
    assert "EXECUTING" in err and REG_SHA in err
    _assert_nothing_written(organized_run)


def test_confirmatory_dirty_worktree_refused(
    organized_run: Path,
    calibration_ok: Path,
    confirmatory_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        rca, "_git_head_state", lambda repo_dir=None: (REG_SHA, True)
    )
    rc = rca.main(_confirmatory_argv(organized_run, calibration_ok))
    assert rc == 1
    assert "DIRTY" in capsys.readouterr().err
    _assert_nothing_written(organized_run)


def test_confirmatory_missing_prereg_names_freeze_task(
    organized_run: Path,
    calibration_ok: Path,
    confirmatory_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(rca, "PREREG_PATH", tmp_path / "absent" / "PRE_REGISTRATION.md")
    rc = rca.main(_confirmatory_argv(organized_run, calibration_ok))
    assert rc == 1
    assert "#112" in capsys.readouterr().err
    _assert_nothing_written(organized_run)


def test_confirmatory_prereg_embedded_sha_must_match(
    organized_run: Path,
    calibration_ok: Path,
    confirmatory_env: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    confirmatory_env["prereg"].write_text(
        "Machinery SHA: `aaaaaaaaaaaaaaaa` (a DIFFERENT freeze).\n",
        encoding="utf-8",
    )
    rc = rca.main(_confirmatory_argv(organized_run, calibration_ok))
    assert rc == 1
    assert "embedded" in capsys.readouterr().err
    _assert_nothing_written(organized_run)


def test_confirmatory_alpha_is_registration_content(
    organized_run: Path,
    calibration_ok: Path,
    confirmatory_env: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = rca.main(
        _confirmatory_argv(organized_run, calibration_ok, "--alpha", "0.01")
    )
    assert rc == 1
    assert "registered alpha" in capsys.readouterr().err
    _assert_nothing_written(organized_run)


def test_confirmatory_metrics_override_refused_unless_registered_pair(
    organized_run: Path,
    calibration_ok: Path,
    confirmatory_env: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    # G1a: a confirmatory --metrics differing from the registered §9.1 pair
    # refuses, naming both lists; the registered pair itself is accepted.
    rc = rca.main(
        _confirmatory_argv(organized_run, calibration_ok, "--metrics", "ttft_ms")
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "ttft_ms" in err and "predicate" in err and "design-input" in err
    _assert_nothing_written(organized_run)
    rc = rca.main(
        _confirmatory_argv(
            organized_run, calibration_ok, "--metrics", "ttft_ms", "predicate"
        )
    )
    assert rc == 0


def test_confirmatory_tost_margin_bound_to_registered_artifact(
    organized_run: Path,
    calibration_ok: Path,
    confirmatory_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Mismatching CLI margin refuses (the registered value is what runs).
    rc = rca.main(
        _confirmatory_argv(
            organized_run, calibration_ok,
            "--tost-margin", "0.1",
            "--equivalence-metric", "grounding_score",
        )
    )
    assert rc == 1
    assert "REGISTERED margin" in capsys.readouterr().err
    _assert_nothing_written(organized_run)
    # Without the registered-margins artifact a CLI margin cannot be minted.
    monkeypatch.setattr(
        rca, "REGISTERED_MARGINS_PATH", tmp_path / "absent_margins.json"
    )
    rc = rca.main(
        _confirmatory_argv(
            organized_run, calibration_ok,
            "--tost-margin", "0.05",
            "--equivalence-metric", "grounding_score",
        )
    )
    assert rc == 1
    assert "registration content" in capsys.readouterr().err
    _assert_nothing_written(organized_run)


def test_confirmatory_registered_set_fails_on_missing_predicate_leg(
    organized_run: Path,
    calibration_ok: Path,
    confirmatory_env: dict[str, Path],
) -> None:
    # G5 + decision d: the fixture has no per-query `predicate` column (no
    # predicate table was built), so the registered co-primary SET of #4 must
    # FAIL on the missing legs with a reason naming the PRODUCER COMMAND
    # (task #119's build_predicate_table.py) — never shrink to the
    # supplied TTFT conjunction.
    assert rca.main(_confirmatory_argv(organized_run, calibration_ok)) == 0
    _, stats = _load_stats(organized_run)
    exclusions = stats["skipped"]["confirmatory_exclusions"]
    assert any(
        e["metric"] == "predicate" and "build_predicate_table.py" in e["reason"]
        for e in exclusions
    )
    gate = stats["gatekeeping"]
    decision = next(
        d for d in gate["set_decisions"] if d["endpoint"] == "contrast-4"
    )
    assert decision["passed"] is False
    assert decision["missing_legs"] == sorted(
        f"{ds}|predicate" for ds in DATASETS
    )
    assert any(
        "build_predicate_table.py" in r for r in decision["missing_leg_reasons"]
    )
    # Per-leg TTFT decisions keep the registered per-dataset semantics
    # (audit §2.2): the supplied legs individually passed.
    for p in gate["primaries"]:
        assert p["passed"] is True


# ---------------------------------------------------------------------------
# G2/G3: family-map ROW tier routing + the separated exploratory BH-FDR tier
# ---------------------------------------------------------------------------


def test_exploratory_faithfulness_never_enters_chain_or_holm(
    organized_run: Path,
) -> None:
    # G2 leak test: ADR-0087 demoted faithfulness to the exploratory tier at
    # MAP level; requesting `--metrics faithfulness` must route EVERY row
    # (even #4's) to the exploratory section — no primary chain, no Holm
    # family, regardless of Contrast.tier.
    rc = rca.main(
        [str(organized_run), "--contrasts", "4", "3", "--metrics", "faithfulness"]
    )
    assert rc == 0
    _, stats = _load_stats(organized_run)
    for entry in stats["contrasts"]:
        assert entry["tier"] == "exploratory"
        assert entry["tier_source"] == "family-map row (§9.3)"
        assert "bh-fdr" in entry["correction"]
        for row in entry["per_dataset"]:
            assert row["p_holm_across_datasets"] is None
    # The chain never saw a primary: the demoted #4 rows cannot gate.
    assert "skipped" in stats["gatekeeping"]
    # The separated exploratory section carries BH-FDR (G3).
    exploratory = stats["exploratory"]
    assert exploratory["n_computed"] == 2 * len(DATASETS)
    assert "NON-CONFIRMATORY" in exploratory["label"]
    from src.analysis.stats.corrections import benjamini_hochberg

    expected = benjamini_hochberg([r["p_value"] for r in exploratory["rows"]])
    for row, p_bh in zip(exploratory["rows"], expected):
        assert row["p_bh_fdr"] == pytest.approx(float(p_bh))
    # And the human summary carries the separated non-confirmatory section.
    dirs = sorted((organized_run / "analysis").iterdir())
    summary = (dirs[-1] / "summary.md").read_text(encoding="utf-8")
    assert "NON-CONFIRMATORY" in summary


# ---------------------------------------------------------------------------
# Decision a: registered per-row one-sided execution
# ---------------------------------------------------------------------------


def test_derive_alternative_direction_mapping() -> None:
    d = rca._derive_alternative
    assert d(4, "two-sided", "ttft_ms", False) == "two-sided"
    # #3 cell-better (oracle context beats retrieval): serving metric
    # (lower better) -> "less"; quality metric -> "greater".
    assert d(3, "one-sided", "ttft_ms", False) == "less"
    assert d(3, "one-sided", "f1_score", True) == "greater"
    # #1/#5 cell-worse (reuse buys / BERGEN monotone): mirrored tails.
    assert d(1, "one-sided", "ttft_ms", False) == "greater"
    assert d(5, "one-sided", "f1_score", True) == "less"
    # #15 metric-dependent ("latency saved vs truth lost"): both sides "less".
    assert d(15, "one-sided", "ttft_ms", False) == "less"
    assert d(15, "one-sided", "f1_score", True) == "less"
    # An undeclared one-sided direction fails loud.
    with pytest.raises(rca.AnalysisError, match="REGISTERED_CELL_DIRECTION"):
        d(20, "one-sided", "ttft_ms", False)


def test_one_sided_row_executes_registered_tail_exactly(
    organized_run: Path,
) -> None:
    # The executed p must be scipy's ONE-SIDED pratt p on the fixture arrays
    # — pinning that the registered sidedness (not a two-sided superset)
    # actually ran (the old driver pinned two-sided-everywhere).
    from src.analysis.stats.tests_by_unit import paired_wilcoxon

    rc = rca.main([str(organized_run), "--contrasts", "3"])
    assert rc == 0
    _, stats = _load_stats(organized_run)
    entry = stats["contrasts"][0]
    assert entry["contrast_id"] == 3
    assert entry["executed_alternative"] == "less"
    # Reconstruct the fixture arrays: value = offset + 1.7*i + 0.3*mean(ordinals).
    ordinal_mean = (
        sum(range(1, WINDOWS_PER_DATASET + 1)) / WINDOWS_PER_DATASET
    )
    a = [TTFT_OFFSET["B1"] + 1.7 * i + 0.3 * ordinal_mean for i in range(N_EXAMPLES)]
    b = [TTFT_OFFSET["B6"] + 1.7 * i + 0.3 * ordinal_mean for i in range(N_EXAMPLES)]
    expected = paired_wilcoxon(a, b, alternative="less", zero_method="pratt")
    for row in entry["per_dataset"]:
        assert row["p_value"] == pytest.approx(expected.p_value)
        assert row["zero_method"] == "pratt"
        assert row["n_nonzero"] == expected.n_nonzero


# ---------------------------------------------------------------------------
# Decision d: registered upstream topology (window secondaries never gate on #4)
# ---------------------------------------------------------------------------


def test_window_secondary_gates_on_registered_upstream_not_headline(
    tmp_path: Path,
) -> None:
    run_dir = _build_run_tree(tmp_path, extra_specs=_f2_pair_specs())
    org.organize_run(run_dir)
    rc = rca.main([str(run_dir), "--contrasts", "4", "15"])
    assert rc == 0
    _, stats = _load_stats(run_dir)
    entry15 = next(e for e in stats["contrasts"] if e["contrast_id"] == 15)
    # The map's registered upstream for the F2 window family is #14 —
    # the old driver hard-wired everything onto #4 (G10).
    assert entry15["upstream"] == "contrast-14"
    assert entry15["family_id"].endswith("|F2|window")
    gate = stats["gatekeeping"]
    ungated_15 = [u for u in gate["ungated"] if u["contrast_id"] == 15]
    assert ungated_15, "the #15 rows must be listed, never dropped"
    for u in ungated_15:
        assert u["upstream"] == "contrast-14"
        assert "contrast-14" in u["reason"]
    assert not any(s["contrast"].startswith("#15") for s in gate["secondaries"])


# ---------------------------------------------------------------------------
# G4a: #13 fingerprint superiority legs + intersection-union endpoint
# ---------------------------------------------------------------------------


def _fingerprint_specs(
    *, include_truncate: bool
) -> list[tuple[CellSpec, EvidenceExtra]]:
    """evict + compress-fp8 (+ B11/B6 truncation pair) vs policy=none under
    F2, each with a constant grounding harm vs the 0.8 reference."""
    base = dict(
        arm="gold-fresh", retriever="none", topology="single", engine="vllm",
        model=MODEL, family="F2", budget_r=0.5, rate_frac=0.9,
    )

    def harm(level: float) -> EvidenceExtra:
        return lambda i: {"grounding_score": level}

    specs: list[tuple[CellSpec, EvidenceExtra]] = [
        (CellSpec(policy="evict", **base), harm(0.70)),  # type: ignore[arg-type]
        (CellSpec(policy="compress-fp8", **base), harm(0.75)),  # type: ignore[arg-type]
        (CellSpec(policy="none", **base), harm(0.80)),  # type: ignore[arg-type]
    ]
    if include_truncate:
        specs.extend(
            [
                (
                    CellSpec.from_baseline(
                        "B11", model=MODEL, family="F2",  # type: ignore[arg-type]
                        budget_r=0.5, rate_frac=0.8,
                    ),
                    harm(0.72),
                ),
                (
                    CellSpec.from_baseline(
                        "B6", model=MODEL, family="F2",  # type: ignore[arg-type]
                        budget_r=0.5, rate_frac=0.8,
                    ),
                    harm(0.80),
                ),
            ]
        )
    return specs


def test_fingerprint_incomplete_legs_yield_iu_p_one(tmp_path: Path) -> None:
    # evict + compress computed, truncate absent: Holm runs at the REGISTERED
    # m=3 (pad p=1.0) and the intersection-union p is 1.0 by construction —
    # an incomplete fingerprint can never pass its chain step.
    run_dir = _build_run_tree(
        tmp_path, special_specs=_fingerprint_specs(include_truncate=False)
    )
    org.organize_run(run_dir)
    rc = rca.main(
        [str(run_dir), "--equivalence-metric", "grounding_score"]
    )
    assert rc == 0
    _, stats = _load_stats(run_dir)
    fp = stats["fingerprint"]
    assert fp["holm_m_registered"] == 3
    legs = fp["legs"]
    assert {leg["leg"] for leg in legs} == {"evict", "compress"}
    for leg in legs:
        assert leg["executed_alternative"] == "less"  # harm on higher-better
        assert leg["zero_method"] == "pratt"
        assert leg["p_value"] < 0.05
        assert leg["p_holm_within_fingerprint"] is not None
    for iu in fp["per_dataset_intersection"]:
        assert iu["missing_legs"] == ["truncate"]
        assert iu["p_intersection_union"] == 1.0
        assert iu["n_legs_supplied"] == 2
    # The chain saw contrast-13 with p=1.0: executed, but never passing.
    gate = stats["gatekeeping"]
    assert gate["primary_chain_order_executed"] == ["contrast-4", "contrast-13"]
    fp_primaries = [
        p for p in gate["primaries"] if p["endpoint"] == "contrast-13"
    ]
    assert {p["dataset_metric"] for p in fp_primaries} == {
        f"{ds}|fingerprint" for ds in DATASETS
    }
    for p in fp_primaries:
        assert p["p_value"] == 1.0
        assert p["passed"] is False


def test_fingerprint_complete_legs_holm_and_iu_pass(tmp_path: Path) -> None:
    run_dir = _build_run_tree(
        tmp_path, special_specs=_fingerprint_specs(include_truncate=True)
    )
    org.organize_run(run_dir)
    rc = rca.main([str(run_dir), "--equivalence-metric", "grounding_score"])
    assert rc == 0
    _, stats = _load_stats(run_dir)
    fp = stats["fingerprint"]
    assert {leg["leg"] for leg in fp["legs"]} == {"evict", "compress", "truncate"}
    from src.analysis.stats.corrections import holm as holm_ref

    for iu in fp["per_dataset_intersection"]:
        assert iu["missing_legs"] == []
        assert iu["n_legs_supplied"] == 3
        dataset_legs = [
            leg for leg in fp["legs"] if leg["dataset"] == iu["dataset"]
        ]
        expected_iu = float(
            max(holm_ref([leg["p_value"] for leg in dataset_legs]))
        )
        assert iu["p_intersection_union"] == pytest.approx(expected_iu)
        assert iu["p_intersection_union"] < 0.05
    gate = stats["gatekeeping"]
    assert gate["primary_chain_order_executed"] == ["contrast-4", "contrast-13"]
    for p in gate["primaries"]:
        if p["endpoint"] == "contrast-13":
            assert p["passed"] is True
    decision_13 = next(
        d for d in gate["set_decisions"] if d["endpoint"] == "contrast-13"
    )
    assert decision_13["passed"] is True


# ---------------------------------------------------------------------------
# G4b: #12 lambda_star_onset executor
# ---------------------------------------------------------------------------


def test_lambda_star_grid_interior_argmax_within_band() -> None:
    rates = [0.5, 0.7, 0.85, 0.95, 1.05, 1.2]
    powers = [0.30, 0.55, 0.80, 0.90, 0.70, 0.40]
    result = rca.lambda_star_onset_from_grid(rates, powers)
    assert result["interpolated"] is True
    assert 0.85 < result["onset_rate_frac"] < 1.05
    assert result["verdict"] == "WITHIN-BAND"
    assert result["band"] == [pytest.approx(1 / 1.15), 1.15]


def test_lambda_star_grid_edge_argmax_is_inconclusive_at_resolution() -> None:
    result = rca.lambda_star_onset_from_grid(
        [0.5, 0.7, 0.85], [0.1, 0.2, 0.3]  # monotone: argmax at the edge
    )
    assert result["interpolated"] is False
    assert result["verdict"] == "INCONCLUSIVE-AT-RESOLUTION"


def test_lambda_star_grid_outside_band_is_falsified_label() -> None:
    # Interior argmax far below the band -> OUTSIDE-BAND (either direction
    # is publishable; no α involved).
    result = rca.lambda_star_onset_from_grid(
        [0.4, 0.5, 0.6, 1.0, 1.2], [0.5, 0.9, 0.5, 0.2, 0.1]
    )
    assert result["interpolated"] is True
    assert result["verdict"] == "OUTSIDE-BAND"


def test_lambda_star_grid_needs_three_points() -> None:
    with pytest.raises(rca.AnalysisError, match="distinct rate_frac"):
        rca.lambda_star_onset_from_grid([0.85, 1.05], [0.5, 0.4])


def test_lambda_star_computed_from_f2_rate_grid(tmp_path: Path) -> None:
    # A real F2 rate grid with per-window goodput_frac + latency_ms: the
    # suite computes per-cell onsets and labels them against the ×/÷1.15 band.
    goodput_of_rate = {0.85: 0.90, 0.95: 0.95, 1.05: 0.60}
    specs: list[tuple[CellSpec, EvidenceExtra]] = []
    for rate, goodput in goodput_of_rate.items():
        spec = CellSpec(
            arm="gold-fresh", retriever="none", policy="none",
            topology="single", engine="vllm", model=MODEL,  # type: ignore[arg-type]
            family="F2", budget_r=0.5, rate_frac=rate,
        )
        specs.append((spec, lambda i, g=goodput: {"goodput_frac": g}))
    run_dir = _build_run_tree(tmp_path, special_specs=specs)
    org.organize_run(run_dir)
    rc = rca.main([str(run_dir), "--contrasts", "12"])
    assert rc == 0
    _, stats = _load_stats(run_dir)
    fals = stats["falsification"]
    assert fals["contrast_id"] == 12
    assert fals["tier"] == "falsification"
    assert len(fals["results"]) == len(DATASETS)  # one grid per dataset
    for result in fals["results"]:
        assert result["interpolated"] is True
        assert 0.85 < result["onset_rate_frac"] < 1.05
        assert result["verdict"] == "WITHIN-BAND"


# ---------------------------------------------------------------------------
# G16: ADR-0086 realized-n gate
# ---------------------------------------------------------------------------


def test_adr0086_ladder_is_registered_data() -> None:
    assert rca.ADR0086_REALIZED_N_LADDER == (2000, 1600, 1200)


def test_confirmatory_refuses_realized_n_below_floor(
    organized_run: Path,
    calibration_ok: Path,
    confirmatory_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Restore the REAL registered ladder: the fixture's n=64 sits below every
    # rung, so the look refuses naming ADR-0086 — and the one-look budget
    # survives (placeholder lock released).
    monkeypatch.setattr(rca, "ADR0086_REALIZED_N_LADDER", (2000, 1600, 1200))
    rc = rca.main(_confirmatory_argv(organized_run, calibration_ok))
    assert rc == 1
    err = capsys.readouterr().err
    assert "ADR-0086" in err and "2000" in err
    assert not (organized_run / "analysis_lock.json").exists()


def test_confirmatory_step_down_rung_accepted_and_recorded(
    organized_run: Path,
    calibration_ok: Path,
    confirmatory_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        rca, "ADR0086_REALIZED_N_LADDER", (2000, 1600, N_EXAMPLES)
    )
    # A non-rung value is refused.
    rc = rca.main(
        _confirmatory_argv(
            organized_run, calibration_ok, "--accept-step-down", "63"
        )
    )
    assert rc == 1
    assert "not a rung" in capsys.readouterr().err
    # The pre-declared rung is accepted AND recorded.
    rc = rca.main(
        _confirmatory_argv(
            organized_run, calibration_ok,
            "--accept-step-down", str(N_EXAMPLES),
        )
    )
    assert rc == 0
    _, stats = _load_stats(organized_run)
    realized = stats["preconditions"]["realized_n"]
    assert realized["checked"] is True
    assert realized["floor"] == N_EXAMPLES
    assert realized["step_down_accepted"] == N_EXAMPLES
    assert realized["ladder"] == [2000, 1600, N_EXAMPLES]


def test_design_input_records_ladder_unchecked(organized_run: Path) -> None:
    rc = rca.main([str(organized_run)])
    assert rc == 0
    _, stats = _load_stats(organized_run)
    realized = stats["preconditions"]["realized_n"]
    assert realized["checked"] is False
    assert realized["ladder"] == [2000, 1600, 1200]


# ---------------------------------------------------------------------------
# G11: loader hazards (boolean coercion + duplicate keys)
# ---------------------------------------------------------------------------


def test_loader_coerces_json_booleans_with_note(tmp_path: Path) -> None:
    spec = CellSpec.from_baseline("B11", model=MODEL)  # type: ignore[arg-type]

    def extra(i: int) -> dict[str, Any]:
        return {"served_ok": bool(i % 2 == 0)}

    run_dir = _build_run_tree(tmp_path, special_specs=[(spec, extra)])
    org.organize_run(run_dir)
    index = pd.read_csv(run_dir / "index" / "cells_index.csv")
    row_key = spec.to_row_key()
    per_query = rca.load_per_query(run_dir, index, {row_key})
    # true/false became 1.0/0.0 — never silently dropped (the future #119
    # predicate producer may emit JSON booleans).
    assert "served_ok" in per_query.columns
    assert set(per_query["served_ok"].unique()) == {0.0, 1.0}
    assert "served_ok" in per_query.attrs["bool_coerced_fields"]


def test_loader_refuses_duplicate_example_record_index(
    organized_run: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    index = pd.read_csv(organized_run / "index" / "cells_index.csv")
    b3 = index[index["baseline"] == "B3"].iloc[0]
    victim = organized_run / str(b3["window_dir"]) / "requests.jsonl"
    first_line = victim.read_text(encoding="utf-8").splitlines()[0]
    victim.write_text(
        victim.read_text(encoding="utf-8") + first_line + "\n", encoding="utf-8"
    )
    rc = rca.main([str(organized_run)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "duplicate (example_id, record_index)" in err
    assert str(victim) in err


def test_loader_accepts_replay_rows_with_distinct_record_index(
    organized_run: Path,
) -> None:
    index = pd.read_csv(organized_run / "index" / "cells_index.csv")
    b3 = index[index["baseline"] == "B3"].iloc[0]
    victim = organized_run / str(b3["window_dir"]) / "requests.jsonl"
    first = json.loads(victim.read_text(encoding="utf-8").splitlines()[0])
    replayed = [
        json.dumps(
            {
                "example_id": first["example_id"],
                "record_index": k,
                "ttft_ms": first["ttft_ms"] + 10.0 * k,
            }
        )
        for k in (1, 2)
    ]
    victim.write_text(
        victim.read_text(encoding="utf-8") + "\n".join(replayed) + "\n",
        encoding="utf-8",
    )
    per_query = rca.load_per_query(organized_run, index, {str(b3["row_key"])})
    sub = per_query[
        (per_query["example_id"] == first["example_id"])
        & (per_query["window_key"] == str(b3["window_key"]))
    ]
    assert len(sub) == 1
    expected = (first["ttft_ms"] * 3 + 10.0 + 20.0) / 3
    assert sub["ttft_ms"].iloc[0] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# G12: blinding masks the equivalence / fingerprint / pressure sections too
# ---------------------------------------------------------------------------


def test_blinding_masks_equivalence_fingerprint_and_pressure_sections(
    tmp_path: Path,
) -> None:
    base = dict(
        arm="gold-fresh", retriever="none", topology="single", engine="vllm",
        model=MODEL, family="F2", budget_r=0.5, rate_frac=0.9,
    )
    specs = _tost_pair_specs() + [
        (
            CellSpec(policy="evict", **base),  # type: ignore[arg-type]
            lambda i: {"grounding_score": 0.70},
        )
    ]
    run_dir = _build_run_tree(
        tmp_path,
        special_specs=specs,
        windows_per_dataset=TOST_WINDOWS_PER_DATASET,
    )
    org.organize_run(run_dir)
    _seal_arm_map(run_dir)
    rc = rca.main(
        [
            str(run_dir),
            "--tost-margin", "0.05",
            "--equivalence-metric", "grounding_score",
        ]
    )
    assert rc == 0
    _, stats = _load_stats(run_dir)
    assert stats["blinding"]["active"] is True
    results = stats["equivalence"]["results"]
    assert results, "the TOST leg must have computed"
    for r in results:
        assert r["cell_row_key"].startswith("BLINDED:ARM-")
        assert r["reference_row_key"].startswith("BLINDED:ARM-")
    legs = stats["fingerprint"]["legs"]
    assert legs, "the evict fingerprint leg must have computed"
    for leg in legs:
        assert leg["cell_row_key"].startswith("BLINDED:ARM-")
        assert leg["reference_row_key"].startswith("BLINDED:ARM-")
    block = stats["skipped"]["pressure_rows"]
    assert block is not None
    assert block["row_keys"]
    for key in block["row_keys"]:
        assert key.startswith("BLINDED:ARM-")


# ---------------------------------------------------------------------------
# G14: atomic outputs, no temp residue
# ---------------------------------------------------------------------------


def test_outputs_atomic_no_temp_residue(organized_run: Path) -> None:
    rc = rca.main([str(organized_run)])
    assert rc == 0
    analysis_dir, stats = _load_stats(organized_run)
    assert not list(analysis_dir.glob("*.tmp-*"))
    # loader notes surfaced (G11 note channel).
    assert stats["loader_notes"]["bool_coerced_fields"] == []


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
