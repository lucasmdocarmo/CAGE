"""Canonical results loader (scripts/4_analysis/_results_loader.py).

Covers the policies the 2026-07-15 audit found inconsistent across six ad-hoc loaders:
error/empty_generation validity, literal-"None" preservation, canonical-vs-timestamped
CSV preference, both directory layouts (run-root trees + flat symlink tree), trial
parsing, and the pooled-per-example estimand vs the legacy mean-of-trial-means.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "4_analysis"))

from _results_loader import (  # noqa: E402
    discover_cells,
    is_empty_generation,
    is_error,
    load_results_long,
    per_example,
    summarize_cells,
    valid_rows,
)

HEADER = "example_id,baseline,error,empty_generation,latency_ms,grounding_score,generated_answer"


def _write_cell(cell_dir: Path, trials: dict[int, list[str]],
                name: str = "results.csv") -> None:
    for trial, rows in trials.items():
        tdir = cell_dir / f"trial_{trial}"
        tdir.mkdir(parents=True, exist_ok=True)
        (tdir / name).write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Row-validity predicates (exact legacy semantics)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw,expected", [
    ("", False), ("None", False), ("none", False), ("False", False), ("0", False),
    ("nan", False),
    ("Timeout", True), ("HTTP 500", True), ("true", True),
])
def test_is_error(raw: str, expected: bool) -> None:
    assert is_error(raw) is expected


@pytest.mark.parametrize("raw,expected", [
    ("True", True), ("true", True),
    ("", False), ("False", False), ("None", False),
])
def test_is_empty_generation(raw: str, expected: bool) -> None:
    assert is_empty_generation(raw) is expected


# --------------------------------------------------------------------------- #
# Layouts + discovery
# --------------------------------------------------------------------------- #
def test_run_root_layout_and_trial_parsing(tmp_path: Path) -> None:
    _write_cell(tmp_path / "baselines" / "no_cache",
                {1: ["ex1,no_cache,,False,100.0,0.9,Paris"],
                 2: ["ex2,no_cache,,False,200.0,0.8,Rome"]})
    _write_cell(tmp_path / "compression" / "compressed_rag",
                {1: ["ex1,compressed_rag,,False,300.0,0.7,Paris"]})
    df = load_results_long(tmp_path)
    assert sorted(df["cell"].unique()) == ["compressed_rag", "no_cache"]
    assert sorted(df["tree"].unique()) == ["baselines", "compression"]
    assert set(df[df.cell == "no_cache"]["trial"]) == {1, 2}


def test_flat_layout_via_symlinks(tmp_path: Path) -> None:
    real = tmp_path / "real_run"
    _write_cell(real / "baselines" / "no_cache", {1: ["ex1,no_cache,,False,100.0,0.9,x"]})
    _write_cell(real / "baselines" / "rag", {1: ["ex1,rag,,False,150.0,0.5,y"]})
    flat = tmp_path / "all_results"
    flat.mkdir()
    (flat / "no_cache").symlink_to(real / "baselines" / "no_cache")
    (flat / "rag").symlink_to(real / "baselines" / "rag")
    cells = discover_cells(flat)
    assert len(cells) == 2, "iterdir+glob must traverse directory symlinks (rglob would not)"
    df = load_results_long(flat)
    assert len(df) == 2


def test_canonical_csv_preferred_over_timestamped_duplicate(tmp_path: Path) -> None:
    cell = tmp_path / "baselines" / "no_cache"
    _write_cell(cell, {1: ["ex1,no_cache,,False,100.0,0.9,x"]})
    # timestamped duplicate with a DIFFERENT value: must be ignored, never double-read
    (cell / "trial_1" / "old_20260101_results.csv").write_text(
        HEADER + "\nex1,no_cache,,False,999.0,0.1,x\n", encoding="utf-8")
    df = load_results_long(tmp_path)
    assert len(df) == 1
    assert float(df["latency_ms"].iloc[0]) == 100.0


def test_timestamped_fallback_when_canonical_absent(tmp_path: Path) -> None:
    _write_cell(tmp_path / "baselines" / "no_cache",
                {1: ["ex1,no_cache,,False,100.0,0.9,x"]},
                name="run_20260101_results.csv")
    df = load_results_long(tmp_path)
    assert len(df) == 1


# --------------------------------------------------------------------------- #
# Validity + literal-string preservation
# --------------------------------------------------------------------------- #
def test_valid_rows_excludes_error_and_empty_gen(tmp_path: Path) -> None:
    _write_cell(tmp_path / "baselines" / "no_cache", {1: [
        "ex1,no_cache,,False,100.0,0.9,ok",
        "ex2,no_cache,Timeout,False,999.0,0.1,bad",
        "ex3,no_cache,,True,50.0,,",
    ]})
    df = load_results_long(tmp_path)
    v = valid_rows(df)
    assert list(v["example_id"]) == ["ex1"]


def test_literal_none_and_na_survive_parsing(tmp_path: Path) -> None:
    # pandas default NA parsing would turn "None"/"NA" into NaN and flip semantics
    _write_cell(tmp_path / "baselines" / "no_cache", {1: [
        'ex1,no_cache,None,False,100.0,None,NA',
    ]})
    df = load_results_long(tmp_path)
    assert df["error"].iloc[0] == "None"
    assert df["generated_answer"].iloc[0] == "NA"
    assert not df["is_error_row"].iloc[0]


# --------------------------------------------------------------------------- #
# The estimand: pooled per-example vs legacy mean-of-trial-means
# --------------------------------------------------------------------------- #
def test_per_example_pools_across_trials(tmp_path: Path) -> None:
    _write_cell(tmp_path / "baselines" / "no_cache", {
        1: ["ex1,no_cache,,False,100.0,0.9,a", "ex2,no_cache,,False,300.0,0.5,b"],
        2: ["ex1,no_cache,,False,200.0,0.7,a"],
    })
    pe = per_example(load_results_long(tmp_path), "latency_ms")
    vals = dict(zip(pe["example_id"], pe["value"]))
    assert vals == {"ex1": 150.0, "ex2": 300.0}  # ex1 averaged across its 2 trials


def test_unequal_coverage_bias_is_visible(tmp_path: Path) -> None:
    # trial 1: one high row scored; trial 2: three low rows scored. Equal-weight
    # trial means inflate the aggregate (the audit's 0.396-vs-0.2505 mechanism).
    _write_cell(tmp_path / "baselines" / "no_cache", {
        1: ["ex1,no_cache,,False,100.0,1.0,a",
            "ex2,no_cache,,False,100.0,,b"],       # unscored
        2: ["ex3,no_cache,,False,100.0,0.0,c",
            "ex4,no_cache,,False,100.0,0.0,d",
            "ex5,no_cache,,False,100.0,0.2,e"],
    })
    summ = summarize_cells(load_results_long(tmp_path), ["grounding_score"],
                           bootstrap_iters=100)
    row = summ.iloc[0]
    assert row["n_examples"] == 4
    assert abs(row["mean"] - (1.0 + 0.0 + 0.0 + 0.2) / 4) < 1e-9          # pooled = 0.3
    assert abs(row["mean_of_trial_means"] - (1.0 + 0.2 / 3) / 2) < 1e-9   # legacy = 0.533..
    assert row["mean"] < row["mean_of_trial_means"]


def test_summarize_handles_metric_with_no_values(tmp_path: Path) -> None:
    _write_cell(tmp_path / "baselines" / "no_cache", {1: ["ex1,no_cache,,False,100.0,,a"]})
    summ = summarize_cells(load_results_long(tmp_path), ["grounding_score"],
                           bootstrap_iters=100)
    row = summ.iloc[0]
    assert row["n_examples"] == 0 and row["mean"] is None
