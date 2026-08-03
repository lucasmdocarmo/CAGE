"""Tests for scripts/4_analysis/run_campaign_analysis.py (the D9 analysis driver).

Builds a synthetic v2 campaign run tree (the RESULTS_LAYOUT §1 fixture pattern
from tests/test_organize_results.py) with real per-query metric values in
requests.jsonl/qa_evidence.jsonl, organizes it with organize_results.py, then
exercises the driver:

- design-input default: stats.json with the expected keys, forest/wlt figures
  rendered, DESIGN-INPUT-ONLY stamp on every output;
- missing index refuses and names organize_results (no auto-run);
- confirmatory without --i-understand-one-look / --registered-sha refuses;
- a second confirmatory run refuses via <run>/analysis_lock.json (§9.11);
- F2 pressure rows produce the NOT-IMPLEMENTED-YET labeled skip, not numbers;
- unknown contrast ids fail loud; window-unit contrasts skip, labeled;
- the PILOT-ERA fences stand on generate_plots.py / run_phase2_stats.sh.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

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
from src.analysis.stats.ledger import hash_artifacts, write_ledger  # noqa: E402

RUN_ID = "20260802-1400-a-qwen3-14b"
CAMPAIGN = "camp1"
SESSION = "a"
MODEL = "qwen3-14b"
DATASETS = ["squad_v2", "hotpotqa"]
CELL_BASELINES = ["B1", "B3", "B6"]  # B6 vs B3 = the default headline contrast #4
WINDOWS_PER_DATASET = 2
N_EXAMPLES = 16

#: Deterministic per-baseline serving/quality offsets: B6 (RAG) pays TTFT vs
#: B3 (CAG) on every example -> the paired Wilcoxon has an unambiguous sign.
TTFT_OFFSET = {"B1": 200.0, "B3": 140.0, "B6": 235.0}
F1_OFFSET = {"B1": 0.80, "B3": 0.62, "B6": 0.71}

WINDOW_ARTIFACTS = (
    "requests.jsonl",
    "qa_evidence.jsonl",
    "engine_metrics.json",
    "cage_stats.jsonl",
)


def _specs() -> list[CellSpec]:
    return [CellSpec.from_baseline(b, model=MODEL) for b in CELL_BASELINES]  # type: ignore[arg-type]


def _write_window(
    wdir: Path, dataset: str, baseline: str, *, ordinal: int
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
        evidence_lines.append(
            json.dumps({"example_id": example_id, "f1_score": f1, "answer": "text"})
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


def _build_run_tree(tmp_path: Path, *, extra_specs: list[CellSpec] | None = None) -> Path:
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
    all_specs = _specs() + list(extra_specs or [])
    for spec in all_specs:
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
                        cell_dir / f"window_{k}", dataset, baseline or "B1", ordinal=ordinal
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
    assert stats["schema_version"] == 1
    assert stats["run"]["run_id"] == RUN_ID
    assert stats["one_look"] == {
        "mode": "design-input",
        "registered_sha": None,
        "lock_file": None,
    }
    assert stats["requested_contrast_ids"] == [4]

    # The headline contrast, one entry per metric (default: ttft_ms).
    assert len(stats["contrasts"]) == 1
    entry = stats["contrasts"][0]
    assert entry["contrast_id"] == 4
    assert entry["cell_baseline"] == "B6"
    assert entry["reference_baseline"] == "B3"
    assert entry["metric"] == "ttft_ms"
    assert entry["higher_is_better"] is False
    assert {d["dataset"] for d in entry["per_dataset"]} == set(DATASETS)
    for row in entry["per_dataset"]:
        assert row["n_pairs"] == N_EXAMPLES
        assert row["wins"] + row["losses"] + row["ties"] == row["n_pairs"]
        # B6 pays TTFT on every example -> all losses, positive median delta.
        assert row["losses"] == N_EXAMPLES
        assert row["median_delta"] > 0
        assert 0.0 <= row["p_value"] <= 1.0
        assert row["p_holm_across_datasets"] >= row["p_value"]
        assert -1.0 <= row["cliffs_delta_paired"] <= 1.0

    # Figures rendered and recorded.
    assert set(stats["figures"]) == {"forest_ttft_ms.png", "wlt_ttft_ms.png"}
    for name in stats["figures"]:
        path = analysis_dir / name
        assert path.is_file() and path.stat().st_size > 0

    # summary.md is stamped and readable.
    summary = (analysis_dir / "summary.md").read_text(encoding="utf-8")
    assert "DESIGN-INPUT-ONLY" in summary
    assert "Contrast #4" in summary
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
# One-look policy (§9.11)
# ---------------------------------------------------------------------------


def test_confirmatory_without_flags_refuses(
    organized_run: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for argv in (
        [str(organized_run), "--confirmatory"],
        [str(organized_run), "--confirmatory", "--i-understand-one-look"],
        [str(organized_run), "--confirmatory", "--registered-sha", "abc123"],
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


def test_confirmatory_runs_once_then_locks(
    organized_run: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [
        str(organized_run),
        "--confirmatory",
        "--i-understand-one-look",
        "--registered-sha",
        "prereg-sha-abc123",
    ]
    assert rca.main(argv) == 0
    analysis_dir, stats = _load_stats(organized_run)
    assert stats["mode_stamp"] == "CONFIRMATORY"
    assert stats["one_look"]["mode"] == "confirmatory"
    assert stats["one_look"]["registered_sha"] == "prereg-sha-abc123"
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
# Guards: pressure rows, unknown ids, window-unit contrasts
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
    assert block["label"] == "NOT-IMPLEMENTED-YET"
    assert "batch" in block["reason"].lower()
    assert f2_key in block["row_keys"]
    assert block["n_windows"] == len(DATASETS) * WINDOWS_PER_DATASET

    # The F2 row never enters any computed contrast.
    for entry in stats["contrasts"]:
        assert f2_key not in (entry["cell_row_key"], entry["reference_row_key"])
    # And the skip is visible in the human summary too.
    dirs = sorted((organized_run_with_f2 / "analysis").iterdir())
    summary = (dirs[-1] / "summary.md").read_text(encoding="utf-8")
    assert "NOT-IMPLEMENTED-YET" in summary
    assert f2_key in summary


def test_unknown_contrast_id_fails_loud(
    organized_run: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = rca.main([str(organized_run), "--contrasts", "99"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "unknown" in err.lower() and "99" in err
    assert not (organized_run / "analysis").exists()


def test_window_unit_contrast_is_skipped_labeled(organized_run: Path) -> None:
    # #14 (serving-yield, F2, window unit) is registered but not per-query.
    rc = rca.main([str(organized_run), "--contrasts", "14"])
    assert rc == 0
    _, stats = _load_stats(organized_run)
    assert stats["contrasts"] == []
    assert stats["figures"] == []
    skipped = stats["skipped"]["contrasts"]
    assert len(skipped) == 1
    assert skipped[0]["contrast_id"] == 14
    assert skipped[0]["label"] == "NOT-IMPLEMENTED-YET"
    assert "9.4" in skipped[0]["reason"]


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
