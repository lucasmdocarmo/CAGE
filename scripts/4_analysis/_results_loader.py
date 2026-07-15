#!/usr/bin/env python3
"""Canonical results loader: ONE parser, ONE validity rule, ONE estimand.

Why (2026-07-15 audit): six independent CSV loaders with three different None/error
policies let plots (equal-weight mean-of-trial-means) and stats (pooled per-query
Wilcoxon) disagree IN SIGN on the same data (e.g. cag_full latency +93.0 ms plotted vs
-3.4 ms tested). Every analysis consumer now derives from this module so figures and
tables share one estimand by construction.

Canonical policy decisions (documented, deliberate):
- Valid row: NOT error AND NOT empty_generation, uniformly for ALL metrics. Quality
  fields are already nulled at record time on such rows; for serving metrics an empty
  generation is a degenerate workload whose timing is not comparable.
- Primary estimand: POOLED PER-EXAMPLE -- average each (cell, example_id) across its
  valid rows (trials x repeats), then aggregate across examples. This is exactly the
  unit statistical_tests.py feeds to Wilcoxon. The legacy equal-weight-per-trial mean
  is retained ONLY as a `mean_of_trial_means` sensitivity column (never plotted): with
  unequal per-trial metric coverage it inflated no_cache bertscore 0.2505 -> 0.3962.
- Throughput (queries_per_second / tokens_per_second) is a wall-clock TRIAL-level fact,
  not derivable per-row: loaded from trial_*/metrics.json and aggregated across trials.
- Discovery uses iterdir()+glob, NEVER rglob: Path.rglob does not traverse directory
  symlinks, which would silently see zero cells through the stats/all_results tree.
- Each trial dir holds results.csv AND a timestamped *_results.csv duplicate; read the
  canonical results.csv only (fallback to the timestamped copy when absent), never both.

CLI:
  python3 scripts/4_analysis/_results_loader.py --run-root results/<phase>/<run-id>
    -> writes <run-root>/stats/results_long.csv + <run-root>/stats/summary_by_cell.csv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

TREES = ("baselines", "compression", "speculative", "envelope", "kv_store", "reference")

# Exact semantics of statistical_tests.py's legacy row filters: these string values
# mean "no error" / "not empty". Anything else in the error column is a real error.
_FALSY = {"", "none", "false", "0", "nan"}

# Default metric set for the summary CSV (serving + primary quality + abstention).
DEFAULT_SUMMARY_METRICS = [
    "ttft_ms", "latency_ms", "tpot_ms",
    "grounding_score", "faithfulness", "context_relevance",
    "completeness_bertscore", "completeness_rouge_l",
    "f1_score", "exact_match", "f1_answerable", "exact_match_answerable",
    "no_answer_correct", "abstention_precision",
    "hallucinated_span_ratio", "cached_prompt_ratio",
]

TRIAL_SCALAR_KEYS = ("queries_per_second", "tokens_per_second")


def is_error(raw: Optional[str]) -> bool:
    """True when the row's error column records a real error."""
    return str(raw or "").strip().lower() not in _FALSY


def is_empty_generation(raw: Optional[str]) -> bool:
    """True when the row is flagged as a degenerate empty generation."""
    return str(raw or "").strip().lower() == "true"


def _trial_csv(trial_dir: Path) -> Optional[Path]:
    canonical = trial_dir / "results.csv"
    if canonical.is_file():
        return canonical
    fallback = sorted(trial_dir.glob("*_results.csv"))
    return fallback[0] if fallback else None


def _cell_has_results(cell_dir: Path) -> bool:
    if not cell_dir.is_dir():
        return False
    for trial in cell_dir.glob("trial_*"):
        if _trial_csv(trial) is not None:
            return True
    return _trial_csv(cell_dir) is not None  # bare results.csv layout


def discover_cells(root: Path) -> List[Tuple[Optional[str], str, Path]]:
    """[(tree, cell_name, cell_dir)] for both layouts.

    Run-root layout: <root>/{baselines,compression,speculative,reference}/<cell>/trial_*/
    Flat layout (stats/all_results symlink tree): <root>/<cell>/trial_*/
    """
    root = Path(root)
    out: List[Tuple[Optional[str], str, Path]] = []
    for tree in TREES:
        tree_dir = root / tree
        if not tree_dir.is_dir():
            continue
        for cell_dir in sorted(tree_dir.iterdir()):
            if _cell_has_results(cell_dir):
                out.append((tree, cell_dir.name, cell_dir))
    if not out:  # flat layout
        for cell_dir in sorted(root.iterdir()):
            if _cell_has_results(cell_dir):
                out.append((None, cell_dir.name, cell_dir))
    return out


def _read_csv_raw(path: Path) -> pd.DataFrame:
    # dtype=str + keep_default_na=False: pandas must NOT turn the literal string
    # "None" (or an answer like "NA") into NaN -- that flips error/text semantics.
    # Numeric coercion is explicit and per-column, downstream.
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _finalize(frames: List[pd.DataFrame], origin: Path) -> pd.DataFrame:
    if not frames:
        raise SystemExit(f"ERROR: no results.csv found under {origin}")
    long_df = pd.concat(frames, ignore_index=True, sort=False)
    err_col = long_df["error"] if "error" in long_df else pd.Series("", index=long_df.index)
    empty_col = (long_df["empty_generation"] if "empty_generation" in long_df
                 else pd.Series("", index=long_df.index))
    long_df["is_error_row"] = err_col.map(is_error)
    long_df["is_empty_gen"] = empty_col.map(is_empty_generation)
    return long_df


def _cell_frames(cell_dir: Path, cell: str, tree: str) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    trial_dirs = sorted(cell_dir.glob("trial_*")) or [cell_dir]
    for trial_dir in trial_dirs:
        csv_path = _trial_csv(trial_dir)
        if csv_path is None:
            continue
        df = _read_csv_raw(csv_path)
        m = re.match(r"trial_(\d+)$", trial_dir.name)
        df["trial"] = int(m.group(1)) if m else 0
        df["cell"] = cell
        df["tree"] = tree
        df["source_csv"] = str(csv_path)
        frames.append(df)
    return frames


def load_cell(cell_dir: Path, cell_name: Optional[str] = None) -> pd.DataFrame:
    """Long-format rows for ONE cell directory (trial_*/ or bare results.csv layout)."""
    cell_dir = Path(cell_dir)
    frames = _cell_frames(cell_dir, cell_name or cell_dir.name, "")
    return _finalize(frames, cell_dir)


def load_results_long(root: Path) -> pd.DataFrame:
    """One row per raw CSV row across every cell, with provenance + validity columns."""
    frames: List[pd.DataFrame] = []
    for tree, cell, cell_dir in discover_cells(Path(root)):
        frames.extend(_cell_frames(cell_dir, cell, tree or ""))
    return _finalize(frames, Path(root))


def valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """The canonical validity rule (see module docstring)."""
    return df[~df["is_error_row"] & ~df["is_empty_gen"]]


def headline_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Valid rows at repeat_index 0 (or absent): the cross-cell COMPARABLE set.

    Fairness rule (2026-07-15): the repeat cell re-asks the same N questions R times
    (ids ``x__repN``); rep>0 rows are its warm-read curve, not comparable requests --
    including them would give that cell 3x the measured rows of every other cell.
    Wilcoxon pairing was already safe (``__repN`` ids never pair across cells); this
    keeps the summaries honest too. Analyze rep>0 separately, grouped by repeat_index.
    """
    v = valid_rows(df)
    if "repeat_index" not in v.columns:
        return v
    rep = v["repeat_index"].astype(str).str.strip().str.lower()
    return v[rep.isin(["", "0", "none", "nan", "0.0"])]


def metric_values(df: pd.DataFrame, metric: str) -> pd.Series:
    """Numeric view of a metric column ('None'/''/garbage -> NaN)."""
    if metric not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[metric], errors="coerce")


def per_example(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """THE shared estimand: mean per (cell, example_id) over valid, scored rows.

    Returns columns [cell, example_id, value]. This is the exact unit the Wilcoxon
    pairing uses, so any aggregate derived from it cannot disagree with the stats.
    Uses headline_rows (rep-0 only) so the repeat cell contributes comparable rows.
    """
    v = headline_rows(df).copy()
    v["_value"] = metric_values(v, metric)
    v = v.dropna(subset=["_value"])
    if v.empty:
        return pd.DataFrame(columns=["cell", "example_id", "value"])
    grouped = (v.groupby(["cell", "example_id"], sort=True)["_value"]
                 .mean().reset_index().rename(columns={"_value": "value"}))
    return grouped


def _bootstrap_ci(values: np.ndarray, iters: int, seed: int,
                  alpha: float = 0.05) -> Tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(iters, len(values)))
    means = values[idx].mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def _mean_of_trial_means(df_cell: pd.DataFrame, metric: str) -> Optional[float]:
    """Legacy sensitivity estimand ONLY (equal weight per trial; coverage-biased)."""
    v = headline_rows(df_cell).copy()
    v["_value"] = metric_values(v, metric)
    v = v.dropna(subset=["_value"])
    if v.empty:
        return None
    trial_means = v.groupby("trial")["_value"].mean()
    return float(trial_means.mean()) if len(trial_means) else None


def summarize_cells(df: pd.DataFrame, metrics: Sequence[str] = DEFAULT_SUMMARY_METRICS,
                    bootstrap_iters: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Per (cell, metric): pooled per-example mean/median/std + bootstrap 95% CI."""
    rows = []
    for cell, df_cell in df.groupby("cell", sort=True):
        tree = df_cell["tree"].iloc[0] if "tree" in df_cell else ""
        for metric in metrics:
            pe = per_example(df_cell, metric)
            values = pe["value"].to_numpy(dtype=float)
            n_rows = int(metric_values(headline_rows(df_cell), metric).notna().sum())
            if len(values):
                lo, hi = _bootstrap_ci(values, bootstrap_iters, seed)
                rows.append({
                    "cell": cell, "tree": tree, "metric": metric,
                    "n_examples": int(len(values)), "n_rows": n_rows,
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "ci95_low": lo, "ci95_high": hi,
                    "mean_of_trial_means": _mean_of_trial_means(df_cell, metric),
                })
            else:
                rows.append({
                    "cell": cell, "tree": tree, "metric": metric,
                    "n_examples": 0, "n_rows": 0,
                    "mean": None, "median": None, "std": None,
                    "ci95_low": None, "ci95_high": None,
                    "mean_of_trial_means": None,
                })
    return pd.DataFrame(rows)


def load_trial_scalars(root: Path,
                       keys: Sequence[str] = TRIAL_SCALAR_KEYS) -> pd.DataFrame:
    """Trial-level wall-clock facts from trial_*/metrics.json (performance block)."""
    rows = []
    for tree, cell, cell_dir in discover_cells(Path(root)):
        for trial_dir in sorted(cell_dir.glob("trial_*")):
            mpath = trial_dir / "metrics.json"
            if not mpath.is_file():
                continue
            try:
                meta = json.loads(mpath.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            perf = meta.get("performance") or {}
            exp = meta.get("experiment") or {}
            m = re.match(r"trial_(\d+)$", trial_dir.name)
            row = {
                "cell": cell, "tree": tree or "",
                "trial": int(m.group(1)) if m else 0,
                "dataset": exp.get("dataset"), "model": exp.get("model"),
            }
            for k in keys:
                try:
                    row[k] = float(perf[k]) if perf.get(k) is not None else None
                except (TypeError, ValueError):
                    row[k] = None
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(description="Build the canonical per-query long table.")
    p.add_argument("--run-root", required=True)
    p.add_argument("--out-dir", default=None,
                   help="Default: <run-root>/stats")
    p.add_argument("--metrics", nargs="*", default=None)
    p.add_argument("--bootstrap-iters", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    root = Path(args.run_root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    long_df = load_results_long(root)
    long_path = out_dir / "results_long.csv"
    long_df.to_csv(long_path, index=False)

    metrics = args.metrics or DEFAULT_SUMMARY_METRICS
    summary = summarize_cells(long_df, metrics, args.bootstrap_iters, args.seed)
    summary_path = out_dir / "summary_by_cell.csv"
    summary.to_csv(summary_path, index=False)

    n_cells = long_df["cell"].nunique()
    n_valid = int(len(valid_rows(long_df)))
    print(f"RESULTS_LONG_DONE cells={n_cells} rows={len(long_df)} valid={n_valid}")
    print(f"  wrote {long_path}")
    print(f"  wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
