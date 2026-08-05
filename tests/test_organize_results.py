"""Tests for scripts/4_analysis/organize_results.py + scripts/5_observability/pull_run.sh.

Builds a synthetic campaign run tree following cloud/RESULTS_LAYOUT.md §1
EXACTLY — window dirs named ``window_<dataset>-<ordinal>``, a ``cell.json`` per
cell, the four per-window artifacts (requests.jsonl / qa_evidence.jsonl /
engine_metrics.json / cage_stats.jsonl), NO window.json, singular ``model`` in
the manifest — seals it with the §9.10 content-hash ledger, and checks:

- REGRESSION (2026-08-02 spec/code contract split): a tree built exactly per
  RESULTS_LAYOUT §1 is ACCEPTED, and the §8 dataset-scoped glob
  (``window_<Y>-*``) works on the tree the organizer accepts,
- index row count / columns / baseline mapping / artifact paths,
- coverage report lists MISSING cells vs the §7.6.1 arm-level floor,
- invalid row keys, unknown dataset ids, missing cell.json, missing required
  window artifacts, and model-mismatch cells fail loud, listed,
- the sharegpt load-donor exemption (no qa_evidence.jsonl required),
- a tampered artifact is caught by src.analysis.stats.ledger.verify_ledger,
- pull_run.sh parses (bash -n).
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
from src.analysis.cellspec import CellSpec  # noqa: E402
from src.analysis.stats.ledger import hash_artifacts, verify_ledger, write_ledger  # noqa: E402

PULL_RUN_SH = REPO_ROOT / "scripts" / "5_observability" / "pull_run.sh"

# §1 grammar: lowercase [a-z0-9-], YYYYMMDD-hhmm-<session>-<model-slug>.
RUN_ID = "20260802-1200-a-qwen3-14b"
CAMPAIGN = "camp1"
SESSION = "a"
MODEL = "qwen3-14b"  # group A (carries B1-B12 in F1); §3: model is SINGULAR
DATASETS = ["squad_v2", "hotpotqa"]
CELL_BASELINES = ["B1", "B3", "B6"]  # gold-fresh, corpus-reuse, retr-fresh · rerank
WINDOWS_PER_DATASET = 2

#: §1 per-window artifact contract (spec order).
WINDOW_ARTIFACTS = (
    "requests.jsonl",
    "qa_evidence.jsonl",
    "engine_metrics.json",
    "cage_stats.jsonl",
)


def _specs() -> list[CellSpec]:
    return [CellSpec.from_baseline(b, model=MODEL) for b in CELL_BASELINES]  # type: ignore[arg-type]


def _write_window(wdir: Path, dataset: str, *, skip: set[str] = frozenset()) -> list[Path]:
    """Write the §1 per-window artifact set; returns the files written."""
    wdir.mkdir(parents=True)
    written: list[Path] = []
    for name in WINDOW_ARTIFACTS:
        if name in skip:
            continue
        path = wdir / name
        if name.endswith(".jsonl"):
            path.write_text(
                "\n".join(
                    json.dumps({"example_id": f"{dataset}-e{i}", "ttft_ms": 100.0 + i})
                    for i in range(2)
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text(json.dumps({"snapshot": "before/after"}), encoding="utf-8")
        written.append(path)
    return written


def _build_run_tree(tmp_path: Path, manifest_extra: dict[str, Any] | None = None) -> Path:
    """Synthetic run tree built EXACTLY per RESULTS_LAYOUT §1, sealed (§5).

    1 model x 2 datasets x 3 cells x 2 windows; window dirs are
    ``window_<dataset>-<ordinal>``; each cell carries cell.json; each window
    carries the four required artifacts; there is NO window.json anywhere.
    """
    run_dir = tmp_path / "results" / CAMPAIGN / SESSION / RUN_ID
    run_dir.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "campaign": CAMPAIGN,
        "session": SESSION,
        "run_id": RUN_ID,
        "model": MODEL,
        # §3 provenance fields (opaque to the organizer, present for realism).
        "git_sha": "deadbeef",
        "git_dirty": False,
        "engine": "vllm",
        "engine_version": "0.19.1",
        "seed": 1,
        "provider": "gcp",
        "hardware": "a2-ultragpu-1g x1",
        "dataset_manifests_sha256": "0" * 64,
        "cellspec_schema_version": 1,
        "created_utc": "2026-08-02T12:00:00Z",
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    sealed: list[Path] = []
    for spec in _specs():
        cell_dir = run_dir / "cells" / spec.to_row_key()
        cell_dir.mkdir(parents=True)
        windows: dict[str, dict[str, Any]] = {}
        for dataset in DATASETS:
            for ordinal in range(1, WINDOWS_PER_DATASET + 1):
                k = f"{dataset}-{ordinal:02d}"
                windows[k] = {"dataset": dataset, "seed": 1, "rep": ordinal}
                sealed.extend(_write_window(cell_dir / f"window_{k}", dataset))
        cell_json = cell_dir / "cell.json"
        cell_json.write_text(
            json.dumps(
                {
                    "cellspec": spec.to_flat_dict(),
                    "baseline": org.BASELINE_OF_CELL.get((spec.arm, spec.retriever), ""),
                    "windows": windows,
                }
            ),
            encoding="utf-8",
        )
        sealed.append(cell_json)
    # One auxiliary telemetry artifact to prove extra *.json files are indexed.
    aux = (
        run_dir
        / "cells"
        / _specs()[0].to_row_key()
        / f"window_{DATASETS[0]}-01"
        / "telemetry.json"
    )
    aux.write_text(json.dumps({"kv_bytes": 123}), encoding="utf-8")
    sealed.append(aux)
    (run_dir / "scoring").mkdir()

    write_ledger(hash_artifacts(sealed, base_dir=run_dir), run_dir / "ledger.json")
    return run_dir


@pytest.fixture()
def run_tree(tmp_path: Path) -> Path:
    return _build_run_tree(tmp_path)


# ---------------------------------------------------------------------------
# REGRESSION: the spec's §1 tree is the accepted tree (2026-08-02 finding —
# the organizer used to REJECT window_<dataset>-<ordinal> + cell.json trees)
# ---------------------------------------------------------------------------


def test_spec_section1_tree_is_accepted(run_tree: Path) -> None:
    """A tree built exactly per RESULTS_LAYOUT §1 must organize cleanly."""
    csv_path, md_path = org.organize_run(run_tree)
    assert csv_path.is_file() and md_path.is_file()


def test_section8_dataset_scoped_glob_works(run_tree: Path) -> None:
    """§8 contract: ``window_<Y>-*`` selects one dataset's windows by NAME."""
    org.organize_run(run_tree)  # the tree the organizer accepts...
    squad = sorted(run_tree.glob("cells/*/window_squad_v2-*"))
    hotpot = sorted(run_tree.glob("cells/*/window_hotpotqa-*"))
    assert len(squad) == len(CELL_BASELINES) * WINDOWS_PER_DATASET
    assert len(hotpot) == len(CELL_BASELINES) * WINDOWS_PER_DATASET
    assert not set(squad) & set(hotpot)


def test_no_window_json_in_spec_tree(run_tree: Path) -> None:
    """The §1 tree has NO window.json — and the organizer must not demand one."""
    assert not list(run_tree.glob("cells/**/window.json"))
    org.organize_run(run_tree)  # must not raise


# ---------------------------------------------------------------------------
# Index correctness
# ---------------------------------------------------------------------------


def test_index_row_count_and_fields(run_tree: Path) -> None:
    csv_path, md_path = org.organize_run(run_tree)
    assert csv_path == run_tree / "index" / "cells_index.csv"
    assert md_path == run_tree / "index" / "coverage_report.md"
    df = pd.read_csv(csv_path)

    # 3 cells x (2 datasets x 2 windows) = 12 window rows.
    assert len(df) == 12
    assert list(df.columns) == list(org.INDEX_COLUMNS)
    assert set(df["model"]) == {MODEL}
    assert set(df["dataset"]) == set(DATASETS)
    assert set(df["baseline"]) == set(CELL_BASELINES)
    assert (df["run_id"] == RUN_ID).all()
    assert (df["session"] == SESSION).all()
    # window_key is the §8 join key <dataset>-<ordinal>; window is the ordinal.
    assert set(df["window_key"]) == {
        f"{d}-{o:02d}" for d in DATASETS for o in range(1, WINDOWS_PER_DATASET + 1)
    }
    assert set(df["window"]) == set(range(1, WINDOWS_PER_DATASET + 1))
    # Every row_key round-trips through CellSpec (charter-valid identity).
    for key in df["row_key"].unique():
        assert org.parse_row_key_dir(key).to_row_key() == key
    # Every row points at its cell.json.
    for rel in df["cell_json"].unique():
        assert (run_tree / rel).is_file() and rel.endswith("cell.json")
    # B6 is the reranked retr-fresh cell; B1/B3 carry no retriever.
    b6 = df[df["baseline"] == "B6"]
    assert set(b6["arm"]) == {"retr-fresh"} and set(b6["retriever"]) == {"rerank"}
    assert set(df[df["baseline"] == "B3"]["arm"]) == {"corpus-reuse"}
    # Sub-pressure F1 cells: coords stay empty.
    assert df["budget_r"].isna().all() and df["rate_frac"].isna().all()


def test_index_artifact_paths_exist_and_include_aux(run_tree: Path) -> None:
    csv_path, _ = org.organize_run(run_tree)
    df = pd.read_csv(csv_path)
    all_artifacts = [a for cell in df["artifacts"] for a in cell.split(";")]
    assert all_artifacts, "no artifacts indexed"
    for rel in all_artifacts:
        assert (run_tree / rel).is_file(), f"indexed artifact missing on disk: {rel}"
    assert any(a.endswith("telemetry.json") for a in all_artifacts)
    # The four §1 record artifacts are indexed for every window.
    for row_artifacts in df["artifacts"]:
        names = {Path(a).name for a in row_artifacts.split(";")}
        assert set(WINDOW_ARTIFACTS) <= names
    assert not any(a.endswith("cell.json") for a in all_artifacts), (
        "cell.json is cell metadata (cell_json column), not a window artifact"
    )


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------


def test_coverage_lists_missing_arms_vs_7_6_1(run_tree: Path) -> None:
    _, md_path = org.organize_run(run_tree)
    report = md_path.read_text(encoding="utf-8")
    # Group A expects the 11 distinct B1-B12 arms in F1; the fixture covers
    # 3 arms -> 8 MISSING per dataset -> 8 * 2 = 16 total.
    expected_arms = org.expected_arms_for_model(MODEL)
    assert len(expected_arms) == 11
    assert "TOTAL MISSING entries: 16" in report
    assert "MISSING qwen3-14b x squad_v2 x gold-reuse" in report
    assert "MISSING qwen3-14b x hotpotqa x corpus-trunc" in report
    # Present arms are counted, not listed as missing.
    assert "MISSING qwen3-14b x squad_v2 x gold-fresh" not in report


def test_coverage_lists_entirely_absent_declared_dataset(tmp_path: Path) -> None:
    # `datasets` is an OPTIONAL manifest declaration; when present it widens
    # the coverage grid to datasets with no windows at all.
    run_dir = _build_run_tree(
        tmp_path, manifest_extra={"datasets": [*DATASETS, "musique"]}
    )
    _, md_path = org.organize_run(run_dir)
    report = md_path.read_text(encoding="utf-8")
    assert "## qwen3-14b (group A) x musique" in report
    assert "No windows at all for this model x dataset." in report
    assert "MISSING qwen3-14b x musique x gold-fresh" in report
    # 11 arms missing for musique + 8 for each covered dataset: 11 + 8 + 8 = 27.
    assert "TOTAL MISSING entries: 27" in report


def test_coverage_checks_manifest_declared_expected_cells(tmp_path: Path) -> None:
    present = CellSpec.from_baseline("B1", model=MODEL).to_row_key()
    absent = CellSpec.from_baseline("B2", model=MODEL).to_row_key()
    run_dir = _build_run_tree(
        tmp_path,
        manifest_extra={
            "expected_cells": [
                {"model": MODEL, "dataset": "squad_v2", "row_key": present},
                {"model": MODEL, "dataset": "squad_v2", "row_key": absent},
            ]
        },
    )
    _, md_path = org.organize_run(run_dir)
    report = md_path.read_text(encoding="utf-8")
    assert f"MISSING qwen3-14b x squad_v2 x `{absent}`" in report
    assert f"MISSING qwen3-14b x squad_v2 x `{present}`" not in report


# ---------------------------------------------------------------------------
# Fail-loud layout validation
# ---------------------------------------------------------------------------


def test_invalid_row_key_dir_raises_listing_it(run_tree: Path) -> None:
    bad_short = run_tree / "cells" / "not-a-row-key"
    bad_short.mkdir()
    # 7 segments but charter-illegal: retriever on a non-retrieval arm (§7.2).
    bad_illegal = run_tree / "cells" / "gold-fresh|dense|none|single|vllm|qwen3-14b|F1"
    (bad_illegal / "window_squad_v2-01").mkdir(parents=True)
    with pytest.raises(org.LayoutError) as excinfo:
        org.organize_run(run_tree)
    message = str(excinfo.value)
    assert "not-a-row-key" in message
    assert "gold-fresh|dense|none|single|vllm|qwen3-14b|F1" in message
    assert len(excinfo.value.problems) >= 2  # both bad keys listed, none swallowed


def test_unknown_dataset_in_window_name_raises(run_tree: Path) -> None:
    spec = _specs()[0]
    cell_dir = run_tree / "cells" / spec.to_row_key()
    _write_window(cell_dir / "window_notadataset-01", "notadataset")
    with pytest.raises(org.LayoutError, match="notadataset"):
        org.organize_run(run_tree)


def test_numeric_only_window_dir_raises(run_tree: Path) -> None:
    # The PRE-spec code contract (window_<int>) is itself a violation now.
    spec = _specs()[0]
    _write_window(run_tree / "cells" / spec.to_row_key() / "window_0", DATASETS[0])
    with pytest.raises(org.LayoutError, match="window_0"):
        org.organize_run(run_tree)


def test_missing_cell_json_raises(run_tree: Path) -> None:
    spec = _specs()[0]
    (run_tree / "cells" / spec.to_row_key() / "cell.json").unlink()
    with pytest.raises(org.LayoutError, match="missing cell.json"):
        org.organize_run(run_tree)


def test_cell_json_cellspec_must_match_dirname(run_tree: Path) -> None:
    lying = CellSpec.from_baseline("B2", model=MODEL)  # != the B1 dir it sits in
    cell_json = run_tree / "cells" / _specs()[0].to_row_key() / "cell.json"
    cell_json.write_text(json.dumps({"cellspec": lying.to_flat_dict()}), encoding="utf-8")
    with pytest.raises(org.LayoutError, match="contradicts directory name"):
        org.organize_run(run_tree)


def test_missing_required_window_artifact_raises(run_tree: Path) -> None:
    spec = _specs()[0]
    victim = (
        run_tree / "cells" / spec.to_row_key() / "window_squad_v2-01" / "qa_evidence.jsonl"
    )
    victim.unlink()
    with pytest.raises(org.LayoutError, match="qa_evidence.jsonl"):
        org.organize_run(run_tree)


def test_sharegpt_load_donor_needs_no_qa_evidence(run_tree: Path) -> None:
    # §1: ShareGPT windows carry serving streams only.
    spec = _specs()[0]
    _write_window(
        run_tree / "cells" / spec.to_row_key() / "window_sharegpt-01",
        "sharegpt",
        skip={"qa_evidence.jsonl"},
    )
    csv_path, _ = org.organize_run(run_tree)
    df = pd.read_csv(csv_path)
    assert "sharegpt-01" in set(df["window_key"])


def test_model_mismatch_cell_raises(run_tree: Path) -> None:
    # §3: one run = one model; a cell for another model is a violation.
    other = CellSpec.from_baseline("B1", model="llama-3.3-70b")
    cell_dir = run_tree / "cells" / other.to_row_key()
    cell_dir.mkdir(parents=True)
    (cell_dir / "cell.json").write_text(json.dumps({}), encoding="utf-8")
    _write_window(cell_dir / "window_squad_v2-01", "squad_v2")
    with pytest.raises(org.LayoutError, match="one run = one model"):
        org.organize_run(run_tree)


def test_missing_manifest_or_ledger_fail_loud(run_tree: Path) -> None:
    (run_tree / "ledger.json").unlink()
    with pytest.raises(org.LayoutError, match="ledger.json missing"):
        org.organize_run(run_tree)
    (run_tree / "manifest.json").unlink()
    with pytest.raises(org.LayoutError, match="manifest.json missing"):
        org.organize_run(run_tree)


def test_run_id_dirname_mismatch_raises(tmp_path: Path) -> None:
    run_dir = _build_run_tree(tmp_path)
    renamed = run_dir.parent / "some-other-run"
    run_dir.rename(renamed)
    with pytest.raises(org.LayoutError, match="run_id"):
        org.organize_run(renamed)


def test_underscore_run_id_violates_bucket_grammar(tmp_path: Path) -> None:
    """REGRESSION (2026-08-02): run_id names gs://cage-<run_id> VERBATIM, so
    the §1 grammar is [a-z0-9-] only — an underscore run_id (the old
    terraform.tfvars.example shipped one) must fail loud, not silently point
    the sync daemon at a nonexistent bucket."""
    bad = "2026-08-02_1200_smoke"
    assert org.RUN_ID_RE.match(bad) is None
    run_dir = _build_run_tree(tmp_path)
    renamed = run_dir.parent / bad
    run_dir.rename(renamed)
    manifest = json.loads((renamed / "manifest.json").read_text(encoding="utf-8"))
    manifest["run_id"] = bad
    (renamed / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(org.LayoutError, match="grammar"):
        org.organize_run(renamed)


# ---------------------------------------------------------------------------
# Ledger integration (the pull_run.sh fail-closed gate, exercised via the library)
# ---------------------------------------------------------------------------


def test_tampered_artifact_is_reported_by_verify_ledger(run_tree: Path) -> None:
    ledger_path = run_tree / "ledger.json"
    assert verify_ledger(ledger_path, run_tree) == []  # pristine pull verifies clean
    victim = next(iter(sorted(run_tree.glob("cells/*/window_squad_v2-01/requests.jsonl"))))
    victim.write_text('{"example_id": "tampered", "ttft_ms": 1.0}\n', encoding="utf-8")
    mismatches = verify_ledger(ledger_path, run_tree)
    assert len(mismatches) == 1
    assert mismatches[0].startswith("HASH-MISMATCH ")
    assert victim.relative_to(run_tree).as_posix() in mismatches[0]
    victim.unlink()  # a MISSING sealed artifact is also a verification failure
    mismatches = verify_ledger(ledger_path, run_tree)
    assert any(m.startswith("MISSING ") for m in mismatches)


# ---------------------------------------------------------------------------
# RESULTS_LAYOUT §6 scoring tree: rescore_quality writer + organizer validator
# ---------------------------------------------------------------------------

import argparse  # noqa: E402

import rescore_quality as rq  # noqa: E402


def _fast_args(**overrides: Any) -> argparse.Namespace:
    base = dict(full=False, device="cpu", apply=False)
    base.update(overrides)
    return argparse.Namespace(**base)


def _enrich_evidence_and_reseal(run_dir: Path) -> None:
    """Replace the minimal fixture evidence with scoreable QA rows + re-seal."""
    for ev in sorted(run_dir.glob("cells/*/window_*/qa_evidence.jsonl")):
        rows = [
            json.dumps(
                {
                    "example_id": f"e{i}",
                    "question": "What color is the sky?",
                    "used_contexts": ["The sky is blue."],
                    "generated_answer": "blue",
                    "reference_answer": "blue",
                    "baseline": "B1",
                    "repeat_index": 0,
                }
            )
            for i in range(2)
        ]
        ev.write_text("\n".join(rows) + "\n", encoding="utf-8")
    (run_dir / "ledger.json").unlink()
    sealed = [
        p
        for p in sorted(run_dir.rglob("*"))
        if p.is_file() and p.name != "ledger.json" and "index" not in p.parts
    ]
    write_ledger(hash_artifacts(sealed, base_dir=run_dir), run_dir / "ledger.json")


def _raw_entries_sha(run_dir: Path) -> str:
    return json.loads((run_dir / "ledger.json").read_text(encoding="utf-8"))[
        "entries_sha256"
    ]


def _manual_scoring_pass(
    run_dir: Path,
    sid: str = "s02-manual",
    *,
    entries_sha: str | None = None,
    cells: list[tuple[str, str]] | None = None,
    seal: bool = True,
) -> Path:
    """Hand-build one scoring/<sid>/ pass (for validator failure injection)."""
    sdir = run_dir / "scoring" / sid
    sdir.mkdir(parents=True)
    files: list[Path] = []
    manifest = sdir / "scoring_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "scoring_run_id": sid,
                "mode": "fast",
                "created_utc": "2026-08-04T00:00:00+00:00",
                "raw_run_ledger_entries_sha256": entries_sha
                or _raw_entries_sha(run_dir),
            }
        ),
        encoding="utf-8",
    )
    files.append(manifest)
    default_cell = _specs()[0].to_row_key()
    for cell_name, window_name in cells or [(default_cell, "window_squad_v2-01")]:
        wdir = sdir / "cells" / cell_name / window_name
        wdir.mkdir(parents=True)
        scores = wdir / "qa_scores.jsonl"
        scores.write_text(json.dumps({"example_id": "e0", "f1_score": 1.0}) + "\n")
        quality = wdir / "quality.json"
        quality.write_text(json.dumps({"rows": 1, "abstained": 0, "means": {}}))
        files.extend([scores, quality])
    if seal:
        write_ledger(hash_artifacts(files, base_dir=sdir), sdir / "ledger.json")
    return sdir


def test_scoring_tree_roundtrip_written_then_validated(run_tree: Path) -> None:
    """rescore_quality --scoring-run-id writes the §6 tree; the organizer
    validates it and reports it — the two halves compose."""
    _enrich_evidence_and_reseal(run_tree)
    rc = rq.run_scoring_tree(run_tree, "s01-fast", _fast_args())
    assert rc == 0

    sdir = run_tree / "scoring" / "s01-fast"
    manifest = json.loads((sdir / "scoring_manifest.json").read_text(encoding="utf-8"))
    assert manifest["scoring_run_id"] == "s01-fast"
    assert manifest["mode"] == "fast"
    assert manifest["raw_run_ledger_entries_sha256"] == _raw_entries_sha(run_tree)
    assert manifest["raw_run_id"] == RUN_ID
    assert manifest["n_rows"] == manifest["n_evidence_files"] * 2

    # Per-cell outputs mirror cells/<row_key>/window_<k>/ (§6), and the raw
    # tree received NOTHING.
    raw_windows = sorted(
        p.parent.relative_to(run_tree).as_posix()
        for p in run_tree.glob("cells/*/window_*/qa_evidence.jsonl")
    )
    scored_windows = sorted(
        p.parent.relative_to(sdir).as_posix()
        for p in sdir.glob("cells/*/window_*/qa_scores.jsonl")
    )
    assert scored_windows == raw_windows
    for w in scored_windows:
        assert (sdir / w / "quality.json").is_file()
        agg = json.loads((sdir / w / "quality.json").read_text(encoding="utf-8"))
        assert agg["rows"] == 2
        assert agg["means"]["f1_score"] == 1.0
    assert not list(run_tree.glob("cells/**/qa_scores.jsonl"))
    assert not list(run_tree.glob("cells/**/results_rescored.csv"))

    # The pass carries its OWN sealed ledger and it verifies.
    assert verify_ledger(sdir / "ledger.json", sdir) == []

    # The organizer accepts + reports the pass.
    _, md_path = org.organize_run(run_tree)
    report = md_path.read_text(encoding="utf-8")
    assert "Scoring passes (RESULTS_LAYOUT §6)" in report
    assert "`s01-fast`" in report


def test_scoring_tree_refuses_existing_id_apply_and_unsealed(
    run_tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _enrich_evidence_and_reseal(run_tree)
    assert rq.run_scoring_tree(run_tree, "s01-fast", _fast_args()) == 0
    capsys.readouterr()

    # Same id again: a scoring bug is fixed by a NEW scoring_run_id (§6).
    assert rq.run_scoring_tree(run_tree, "s01-fast", _fast_args()) == 2
    assert "NEW scoring_run_id" in capsys.readouterr().err

    # --apply writes into cells/ and is forbidden in scoring-tree mode.
    assert rq.run_scoring_tree(run_tree, "s02-x", _fast_args(apply=True)) == 2
    assert "cells/" in capsys.readouterr().err

    # Bad grammar refuses.
    assert rq.run_scoring_tree(run_tree, "S01_BAD", _fast_args()) == 2
    assert "grammar" in capsys.readouterr().err

    # An UNSEALED raw tree refuses (§5/§6: scoring references the seal).
    (run_tree / "ledger.json").unlink()
    assert rq.run_scoring_tree(run_tree, "s03-x", _fast_args()) == 2
    assert "SEALED" in capsys.readouterr().err


def test_validator_catches_seal_mismatch(run_tree: Path) -> None:
    _manual_scoring_pass(run_tree, entries_sha="0" * 64)
    with pytest.raises(org.LayoutError, match="DIFFERENT seal"):
        org.organize_run(run_tree)


def test_validator_catches_invented_cell_and_window(run_tree: Path) -> None:
    ghost = "gold-reuse|none|none|single|vllm|qwen3-14b|F1"  # not in the raw tree
    _manual_scoring_pass(run_tree, cells=[(ghost, "window_squad_v2-01")])
    with pytest.raises(org.LayoutError, match="cannot invent cells"):
        org.organize_run(run_tree)


def test_validator_catches_scoring_output_inside_raw_cells(run_tree: Path) -> None:
    contaminated = (
        run_tree / "cells" / _specs()[0].to_row_key() / "window_squad_v2-01" / "qa_scores.jsonl"
    )
    contaminated.write_text("{}\n", encoding="utf-8")
    with pytest.raises(org.LayoutError, match="NEVER writes into cells/"):
        org.organize_run(run_tree)


def test_validator_requires_scoring_pass_own_ledger(run_tree: Path) -> None:
    _manual_scoring_pass(run_tree, seal=False)
    with pytest.raises(org.LayoutError, match="own ledger"):
        org.organize_run(run_tree)


def test_validator_accepts_valid_manual_pass(run_tree: Path) -> None:
    _manual_scoring_pass(run_tree)
    _, md_path = org.organize_run(run_tree)
    assert "`s02-manual`" in md_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# pull_run.sh
# ---------------------------------------------------------------------------


def test_pull_run_sh_parses() -> None:
    assert PULL_RUN_SH.is_file(), f"missing {PULL_RUN_SH}"
    proc = subprocess.run(
        ["bash", "-n", str(PULL_RUN_SH)], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, f"bash -n failed:\n{proc.stderr}"


def test_pull_run_sh_fails_closed_without_args() -> None:
    proc = subprocess.run(
        ["bash", str(PULL_RUN_SH)], capture_output=True, text=True, check=False
    )
    assert proc.returncode != 0
    assert "DO NOT TEARDOWN" in proc.stderr
    assert "SAFE TO TEARDOWN" not in proc.stdout
