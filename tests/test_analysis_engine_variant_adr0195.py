"""ADR-0195 amendment 1: the engine serving variant in the analysis tables.

The cd-act1 index holds two F1 rows for SGLang B3, the default and the
extra_buffer probe. Every step that keys rows by the ``engine`` column must
keep them apart: the index carries the variant as a column, the driver's
pairing axes include it, the yield table groups by it, and the figure block's
Landing sets the probe aside (counted, never pooled, read by row key).
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pandas as pd
import pytest

import test_academic_plots as tap
import test_campaign_analysis as tca
import test_yield_table as tyt
from src.analysis.cellspec import CellSpec

rca = tca.rca
org = tca.org
ap = tap.ap
yt = tyt.yt
MODEL = tca.MODEL


def _probe() -> CellSpec:
    return CellSpec.from_baseline("B3", model=MODEL, engine="sglang", engine_variant="extra_buffer")  # type: ignore[arg-type]


def _sglang(baseline: str) -> CellSpec:
    return CellSpec.from_baseline(baseline, model=MODEL, engine="sglang")  # type: ignore[arg-type]


# --------------------------------------------------------------------- index


def test_the_index_carries_the_engine_variant_as_its_last_column(tmp_path: Path) -> None:
    run_dir = tca._build_run_tree(tmp_path, extra_specs=[_probe()])
    csv_path, _ = org.organize_run(run_dir)
    assert org.INDEX_COLUMNS[-1] == "engine_variant" and len(org.INDEX_COLUMNS) == 21
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    assert list(df.columns) == list(org.INDEX_COLUMNS)
    probe_key = _probe().to_row_key()
    assert set(df.loc[df["row_key"] == probe_key, "engine_variant"]) == {"extra_buffer"}
    assert set(df.loc[df["row_key"] != probe_key, "engine_variant"]) == {""}


def test_the_legacy_row_builder_writes_an_empty_variant() -> None:
    import build_legacy_index as bli

    row = bli._spec_index_row(
        _sglang("B3"), run_id="r", campaign="c", session="s", dataset="squad_v2", window=1,
        window_key="squad_v2-01", row_key="k", window_dir="cells/k/window_squad_v2-01",
        cell_json="cells/k/cell.json", artifacts=(),
    )
    assert list(row) == list(org.INDEX_COLUMNS)
    assert row["engine_variant"] == ""


# -------------------------------------------------------------------- driver


def test_load_index_normalizes_a_missing_or_empty_variant_column(tmp_path: Path) -> None:
    run_dir = tca._build_run_tree(tmp_path, extra_specs=[_probe()])
    org.organize_run(run_dir)
    index = rca.load_index(run_dir)
    probe_key = _probe().to_row_key()
    assert set(index.loc[index["row_key"] == probe_key, "engine_variant"]) == {"extra_buffer"}
    assert set(index.loc[index["row_key"] != probe_key, "engine_variant"]) == {""}
    # an index organized before ADR-0195 has no column: it loads as all-default
    path = run_dir / "index" / "cells_index.csv"
    pd.read_csv(path).drop(columns=["engine_variant"]).to_csv(path, index=False)
    index = rca.load_index(run_dir)
    assert set(index["engine_variant"]) == {""}


def test_contrast_pairs_take_the_default_reference_and_never_the_probe(tmp_path: Path) -> None:
    # Before the amendment the SGLang B3 reference group held two row keys and
    # select_contrast_pairs refused the whole analysis as ambiguous.
    run_dir = tca._build_run_tree(tmp_path, extra_specs=[_sglang("B3"), _sglang("B6"), _probe()])
    org.organize_run(run_dir)
    index = rca.load_index(run_dir)
    pairs = rca.select_contrast_pairs(index, rca.CONTRAST_BY_ID[4])
    refs = {p.reference_row_key for p in pairs}
    assert _sglang("B3").to_row_key() in refs and _probe().to_row_key() not in refs
    assert all("variant=" not in p.cell_row_key and "variant=" not in p.reference_row_key for p in pairs)
    assert "engine_variant" in rca._PAIR_MATCH_AXES and "engine_variant" in rca._WINDOW_PAIR_MATCH_AXES
    assert "engine_variant" in rca._POLICY_PAIR_MATCH_AXES and "engine_variant" in rca._TRUNCATE_PAIR_MATCH_AXES


def test_pair_selection_tolerates_a_hand_built_frame_without_the_column(tmp_path: Path) -> None:
    run_dir = tca._build_run_tree(tmp_path, extra_specs=[_sglang("B3"), _sglang("B6")])
    org.organize_run(run_dir)
    index = rca.load_index(run_dir).drop(columns=["engine_variant"])
    pairs = rca.select_contrast_pairs(index, rca.CONTRAST_BY_ID[4])
    assert _sglang("B3").to_row_key() in {p.reference_row_key for p in pairs}


# --------------------------------------------------------------- yield table

KEY_B3 = f"corpus-reuse|none|none|single|{tyt.ENGINE}|qwen3-14b|F1"
KEY_PROBE = KEY_B3 + "|variant=extra_buffer"


def test_the_yield_table_keys_the_probe_apart_from_the_default_window(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    tyt._write_calibration(tmp_path / "cal")
    rows = [tyt._row(i, ttft_ms=100.0, tpot_ms=10.0, scheduled=False, mode="closed_loop") for i in range(2)]
    for key in (KEY_B3, KEY_PROBE):
        # the cell.json of the fixture carries no variant: the row key alone must label it
        tyt._window(run_root, key, "window_squad_v2-01", rows, dataset="squad_v2", open_loop=False, predicates=[True, True])
    rc, table, out = tyt._run(tmp_path, run_root)
    assert rc == 0
    assert "engine_variant" in yt.ID_COLUMNS and "engine_variant" in yt.GROUP_KEYS
    assert table[(KEY_PROBE, "window_squad_v2-01")]["engine_variant"] == "extra_buffer"
    assert table[(KEY_B3, "window_squad_v2-01")]["engine_variant"] == ""
    md = (out / "yield_summary.md").read_text(encoding="utf-8")
    # the default lines read as before; the probe is its own line under a composed label
    assert "## Medians per (engine, family, arm, rung, dataset), records only" in md
    assert f"| {tyt.ENGINE} | F1 | corpus-reuse |  | squad_v2 | 1 | 1.000 | 1.000 |" in md
    assert f"| {tyt.ENGINE}+extra_buffer | F1 | corpus-reuse |  | squad_v2 | 1 | 1.000 | 1.000 |" in md
    assert "## F1 reference (unpressured windows) per engine" in md
    assert f"| {tyt.ENGINE} | 1 | 1.000 | 1.000 |" in md
    assert f"| {tyt.ENGINE}+extra_buffer | 1 | 1.000 | 1.000 |" in md
    assert f"| {tyt.ENGINE} | 2 |" not in md


# -------------------------------------------------------------- figure block


def _add_probe(parts: dict) -> str:
    """Append the probe's index row (and its yield row) to the synthetic landing,
    with the default B3 window's request, predicate and score files copied under
    the probe's paths, so a reader that pooled the probe would have rows to read
    (review MEDIUM 1: an absent window reads as empty and detects nothing)."""
    key = CellSpec.from_baseline("B3", model=tap.MODEL, engine="sglang", engine_variant="extra_buffer").to_row_key()  # type: ignore[arg-type]
    b3 = CellSpec.from_baseline("B3", model=tap.MODEL).to_row_key()  # type: ignore[arg-type]
    for root in (parts["run_root"], parts["run_root"] / "predicate" / tap.SCORING, parts["run_root"] / "scoring" / tap.SCORING):
        src = root / "cells" / b3
        if src.is_dir():
            shutil.copytree(src, root / "cells" / key)
    index_path = parts["run_root"] / "index" / "cells_index.csv"
    with open(index_path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    template = dict(rows[0])
    for r in rows:
        r["engine_variant"] = ""
    template.update({"engine": "sglang", "arm": "corpus-reuse", "baseline": "B3", "family": "F1", "budget_r": "",
                     "rate_frac": "", "row_key": key, "window_dir": f"cells/{key}/window_{tap.DATASET}-01",
                     "cell_json": f"cells/{key}/cell.json", "engine_variant": "extra_buffer"})
    rows.append(template)
    with open(index_path, "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    with open(parts["yield"], encoding="utf-8", newline="") as fh:
        yrows = list(csv.DictReader(fh))
    y = dict(yrows[0])
    y.update({"row_key": key, "window_dir": f"window_{tap.DATASET}-01", "engine": "sglang", "family": "F1",
              "arm": "corpus-reuse", "rung": "", "rate_frac": ""})
    yrows.append(y)
    with open(parts["yield"], "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(yrows[0]))
        wr.writeheader()
        wr.writerows(yrows)
    return key


def test_the_landing_sets_the_probe_aside_and_counts_it(tmp_path: Path) -> None:
    parts = tap._write_landing(tmp_path)
    key = _add_probe(parts)
    L = tap._landing(parts)
    assert key not in set(L.cells["row_key"]) and key in set(L.cells_all["row_key"])
    assert L.probe_windows == 1 and len(L.cells_all) == len(L.cells) + 1
    y = L.yield_frame()
    assert key not in set(y["row_key"])
    assert L.yield_probe_rows == 1 and L.yield_unmatched == 0
    # the no-load request frame (plot10 and its kin) never reads the probe's rows...
    wdir = f"cells/{key}/window_{tap.DATASET}-01"
    assert not L.rows(wdir).empty
    assert key not in set(ap._f1_rows(L)["cell_row_key"])
    # ...and the detector is live: under the pooled reading the same frame holds them
    L.cells = L.cells_all
    assert key in set(ap._f1_rows(L)["cell_row_key"])
    L.cells = L.cells_all[L.cells_all["engine_variant"] == ""].reset_index(drop=True)
    # review LOW 2: the figures that report the unmatched count report the set-aside count too
    out = tmp_path / "out"
    out.mkdir()
    ap.ps.apply_style()
    for plot in (ap.plot01_load_curve, ap.plot02_truth_tax):
        _, used = plot(L, out)
        assert used.endswith("; probe windows set aside: 1"), used


def test_a_landing_organized_before_the_amendment_has_no_probe(tmp_path: Path) -> None:
    parts = tap._write_landing(tmp_path)
    L = tap._landing(parts)
    assert "engine_variant" not in pd.read_csv(parts["run_root"] / "index" / "cells_index.csv").columns
    assert L.probe_windows == 0 and len(L.cells_all) == len(L.cells)
    assert set(L.cells["engine_variant"]) == {""}
    L.yield_frame()
    assert L.yield_probe_rows == 0
    out = tmp_path / "out"
    out.mkdir()
    ap.ps.apply_style()
    _, used = ap.plot01_load_curve(L, out)
    assert "probe windows" not in used and used.endswith("table rows unknown to the index: 0")


def test_load_index_refuses_a_variant_on_a_pressure_row(tmp_path: Path) -> None:
    # review LOW 1: the truth-tax, frontier, lambda-star and bundle paths key
    # pressure rows by the bare engine; no variant is registered under pressure,
    # and the loader refuses one by name rather than letting those paths pool it.
    run_dir = tca._build_run_tree(tmp_path, extra_specs=[_probe()])
    org.organize_run(run_dir)
    path = run_dir / "index" / "cells_index.csv"
    index = pd.read_csv(path, dtype=str, keep_default_na=False)
    probe_key = _probe().to_row_key()
    index.loc[index["row_key"] == probe_key, ["family", "budget_r", "rate_frac"]] = ["F2", "0.5", "0.8"]
    index.to_csv(path, index=False)
    with pytest.raises(rca.AnalysisError, match="engine_variant") as info:
        rca.load_index(run_dir)
    assert probe_key in str(info.value) and "F2" in str(info.value)
