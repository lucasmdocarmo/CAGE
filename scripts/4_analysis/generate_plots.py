#!/usr/bin/env python3
"""Generate the run's analysis figures from the canonical results loader.

2026-07-15 overhaul: this script no longer parses aggregated_metrics.json. Every
per-query aggregate is derived from _results_loader (the SAME per-example estimand the
Wilcoxon tables use), and throughput comes from trial-level wall-clock facts. Central
tendency for per-query metrics is the POOLED PER-EXAMPLE MEDIAN: on the 5x3 fixture the
per-example MEAN still disagrees in sign with the Wilcoxon for cag_full latency
(+93 ms vs -3.4 ms) because latency is heavily right-skewed; the median (-4.9 ms) is
sign-consistent with the paired test by construction of the shared per-example unit.

Figures:
- *_by_baseline.png     horizontal bars + error bars (bootstrap CI / trial std)
- delta_vs_no_cache_forest.png  forest plot rendered straight from phase2_stats.json
- pareto/tradeoff/radar/heatmap/breakdown/bubble/speedup/ranking as before, restyled.

Deleted (semantically invalid share/cumulative-% of a rate): tokens_per_sec_share_pie,
tokens_per_sec_pareto.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _plot_style import (  # noqa: E402
    BASELINE_ORDER,
    CORE_ARMS,
    abbrev,
    annotate_points,
    apply_style,
    drop_missing,
    order_cells,
    save_fig,
    write_abbrev_legend,
)
from _results_loader import (  # noqa: E402
    load_results_long,
    load_trial_scalars,
    per_example,
    summarize_cells,
)

# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------

# Frame column -> results.csv metric name (loader vocabulary).
PER_QUERY_METRICS = {
    "ttft_ms": "ttft_ms",
    "latency_ms": "latency_ms",
    "tpot_ms": "tpot_ms",
    "grounding_score": "grounding_score",
    "faithfulness": "faithfulness",
    "relevance": "context_relevance",
    "bertscore": "completeness_bertscore",
    "rouge_l": "completeness_rouge_l",
    "f1_score": "f1_score",
}

# Frame column -> trial_*/metrics.json performance key (wall-clock trial facts).
TRIAL_METRICS = {
    "qps": "queries_per_second",
    "tokens_per_sec": "tokens_per_second",
}

# Figures deleted by the 2026-07-15 overhaul; removed from the plots dir when present
# so stale copies from earlier runs cannot linger next to the regenerated set.
DELETED_FIGURES = ("tokens_per_sec_share_pie.png", "tokens_per_sec_pareto.png")

CAVEAT_LINE = (
    "All aggregates are pooled per-example medians (same per-example estimand as the "
    "Wilcoxon tables; the mean is kept in *_mean CSV columns only); throughput is "
    "trial-level."
)


def _bootstrap_median_ci(values: np.ndarray, iters: int = 10000, seed: int = 42,
                         alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI of the MEDIAN (matches the plotted central tendency)."""
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(iters, len(values)))
    meds = np.median(values[idx], axis=1)
    return (float(np.percentile(meds, 100 * alpha / 2)),
            float(np.percentile(meds, 100 * (1 - alpha / 2))))


def build_summary(results_dir: Path, bootstrap_iters: int = 10000,
                  seed: int = 42) -> pd.DataFrame:
    """Per-baseline wide frame from the canonical loader.

    Per-query metrics: pooled per-example MEDIAN + bootstrap CI of the median
    (<col>, <col>_ci_low, <col>_ci_high, <col>_mean, <col>_n examples).
    Trial metrics (qps, tokens_per_sec): mean across trials +- std
    (<col>, <col>_std, <col>_ci_low, <col>_ci_high, <col>_n trials).

    Tolerates missing trees (mid-run preview) and missing metrics columns; raises
    SystemExit (via the loader) when the run root has zero result cells.
    """
    long_df = load_results_long(results_dir)  # SystemExit when no results.csv anywhere
    cells = order_cells(long_df["cell"].unique())
    trees = long_df.groupby("cell")["tree"].first() if "tree" in long_df else {}
    rows: dict[str, dict] = {c: {"baseline": c, "tree": trees.get(c, "")} for c in cells}

    summary = summarize_cells(long_df, list(PER_QUERY_METRICS.values()),
                              bootstrap_iters=bootstrap_iters, seed=seed)
    for col, metric in PER_QUERY_METRICS.items():
        pe = per_example(long_df, metric)
        by_cell = {c: g["value"].to_numpy(dtype=float) for c, g in pe.groupby("cell")}
        sm = summary[summary["metric"] == metric].set_index("cell")
        for cell in cells:
            vals = by_cell.get(cell)
            if vals is None or len(vals) == 0:
                continue
            lo, hi = _bootstrap_median_ci(vals, bootstrap_iters, seed)
            r = rows[cell]
            r[col] = float(np.median(vals))
            r[f"{col}_ci_low"] = lo
            r[f"{col}_ci_high"] = hi
            if cell in sm.index and pd.notna(sm.loc[cell, "mean"]):
                r[f"{col}_mean"] = float(sm.loc[cell, "mean"])
            r[f"{col}_n"] = int(len(vals))

    trial_df = load_trial_scalars(results_dir, keys=tuple(TRIAL_METRICS.values()))
    if not trial_df.empty:
        grouped = trial_df.groupby("cell")
        for col, key in TRIAL_METRICS.items():
            if key not in trial_df.columns:
                continue
            agg = grouped[key].agg(["mean", "std", "count"])
            for cell in cells:
                if cell not in agg.index or pd.isna(agg.loc[cell, "mean"]):
                    continue
                m = float(agg.loc[cell, "mean"])
                s = float(agg.loc[cell, "std"]) if pd.notna(agg.loc[cell, "std"]) else 0.0
                r = rows[cell]
                r[col] = m
                r[f"{col}_std"] = s
                r[f"{col}_ci_low"] = m - s
                r[f"{col}_ci_high"] = m + s
                r[f"{col}_n"] = int(agg.loc[cell, "count"])

    return pd.DataFrame([rows[c] for c in cells])


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _fmt(v: float) -> str:
    av = abs(float(v))
    if av >= 1000:
        return f"{v:,.0f}"
    if av >= 100:
        return f"{v:.0f}"
    if av >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def _asym_xerr(vals: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    err_lo = np.nan_to_num(np.clip(vals - lo, 0, None))
    err_hi = np.nan_to_num(np.clip(hi - vals, 0, None))
    return np.vstack([err_lo, err_hi])


# ---------------------------------------------------------------------------
# By-baseline horizontal bars (one function, six figures)
# ---------------------------------------------------------------------------

def plot_metric_by_baseline(
    df: pd.DataFrame,
    col: str,
    title: str,
    xlabel: str,
    filename: Path,
    note: str = "",
    refline_zero: bool = False,
) -> bool:
    name = Path(filename).name
    data = drop_missing(df, [col], name)
    if data.empty:
        print(f"  [{name}] no data; figure skipped")
        return False

    n = len(data)
    fig, ax = plt.subplots(figsize=(9, max(3.2, 0.42 * n + 1.4)))
    y = np.arange(n)
    vals = data[col].to_numpy(dtype=float)

    xerr = None
    hi = vals.copy()
    lo = vals.copy()
    if f"{col}_ci_low" in data.columns and f"{col}_ci_high" in data.columns:
        lo = pd.to_numeric(data[f"{col}_ci_low"], errors="coerce").to_numpy(dtype=float)
        hi = pd.to_numeric(data[f"{col}_ci_high"], errors="coerce").to_numpy(dtype=float)
        lo = np.where(np.isnan(lo), vals, lo)
        hi = np.where(np.isnan(hi), vals, hi)
        xerr = _asym_xerr(vals, lo, hi)

    ax.barh(y, vals, xerr=xerr, height=0.62, capsize=3,
            color=sns.color_palette("colorblind")[0],
            error_kw={"elinewidth": 1.0, "alpha": 0.85})
    ax.set_yticks(y)
    ax.set_yticklabels(data["baseline"])  # full names: horizontal bars have room
    ax.invert_yaxis()
    ax.yaxis.grid(False)

    # Value labels just past the bar end / CI whisker.
    ends = np.maximum(vals, hi)
    starts = np.minimum(np.minimum(vals, lo), 0.0)
    span = float(np.nanmax(ends) - np.nanmin(starts)) or 1.0
    for yi, v, e in zip(y, vals, ends):
        ax.text(e + 0.015 * span, yi, _fmt(v), va="center", ha="left", fontsize=8)
    right = float(np.nanmax(ends)) + 0.15 * span
    left = float(np.nanmin(starts))
    if left < 0:
        left -= 0.04 * span
    ax.set_xlim(left=left, right=right)

    if refline_zero:
        ax.axvline(0.0, color="0.25", linewidth=1.0, linestyle="--", zorder=0)

    ax.set_xlabel(xlabel + (f"\n{note}" if note else ""))
    ax.set_title(title)
    save_fig(fig, filename)
    return True


# ---------------------------------------------------------------------------
# Forest plot straight from phase2_stats.json (cannot disagree with the stats)
# ---------------------------------------------------------------------------

FOREST_METRICS = [
    "ttft_ms", "latency_ms", "tpot_ms",
    "f1_score", "completeness_rouge_l", "grounding_score",
]
FOREST_LABELS = {
    "ttft_ms": "TTFT (ms)",
    "latency_ms": "Latency (ms)",
    "tpot_ms": "TPOT (ms)",
    "f1_score": "F1 score",
    "completeness_rouge_l": "ROUGE-L",
    "grounding_score": "Grounding score",
}


def plot_forest_from_stats(stats_path: Path, filename: Path) -> bool:
    name = Path(filename).name
    if not stats_path.is_file():
        print(f"  [{name}] stats JSON not found at {stats_path}; forest plot skipped "
              "(run run_phase2_stats.sh first)")
        return False
    try:
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  [{name}] could not read {stats_path}: {exc}; forest plot skipped")
        return False

    comparisons = payload.get("comparisons") or []
    reference = payload.get("reference", "no_cache")
    by_metric: dict[str, list[dict]] = {}
    for comp in comparisons:
        by_metric.setdefault(comp.get("metric", ""), []).append(comp)

    available = [m for m in FOREST_METRICS if by_metric.get(m)]
    missing = [m for m in FOREST_METRICS if not by_metric.get(m)]
    if missing:
        print(f"  [{name}] metrics absent from stats JSON (panels skipped): "
              f"{', '.join(missing)}")
    if not available:
        print(f"  [{name}] none of the forest metrics present in stats JSON; skipped")
        return False

    ncols = 2
    nrows = math.ceil(len(available) / ncols)
    n_baselines = max(len(by_metric[m]) for m in available)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(11.5, nrows * max(2.6, 0.28 * n_baselines + 1.3)),
    )
    axes = np.atleast_1d(axes).ravel()
    color = sns.color_palette("colorblind")[0]

    for ax, metric in zip(axes, available):
        comps = {c["baseline"]: c for c in by_metric[metric]}
        names = order_cells(comps.keys())
        ys = np.arange(len(names))
        for yi, bl in zip(ys, names):
            c = comps[bl]
            d = c.get("median_diff")
            if d is None:
                continue
            lo = c.get("ci95_low", c.get("ci_low"))
            hi = c.get("ci95_high", c.get("ci_high"))
            xerr = None
            if lo is not None and hi is not None:
                xerr = _asym_xerr(np.array([float(d)]),
                                  np.array([float(lo)]), np.array([float(hi)]))
            holm = c.get("p_value_holm")
            significant = holm is not None and float(holm) < 0.05
            ax.errorbar(
                float(d), yi, xerr=xerr, fmt="o", markersize=5.5,
                color=color, markerfacecolor=color if significant else "white",
                markeredgecolor=color, capsize=2.5, elinewidth=1.0, zorder=3,
            )
        ax.axvline(0.0, color="0.3", linewidth=1.0, linestyle="--", zorder=1)
        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels([abbrev(b) for b in names])
        ax.invert_yaxis()
        ax.yaxis.grid(False)
        ax.set_title(FOREST_LABELS.get(metric, metric), fontsize=11)
        ax.set_xlabel("median paired difference", fontsize=9)

    for ax in axes[len(available):]:
        ax.set_visible(False)

    handles = [
        plt.Line2D([], [], linestyle="", marker="o", markersize=6, color=color,
                   markerfacecolor=color, markeredgecolor=color,
                   label="significant (Holm p < 0.05)"),
        plt.Line2D([], [], linestyle="", marker="o", markersize=6, color=color,
                   markerfacecolor="white", markeredgecolor=color,
                   label="not significant"),
    ]
    fig.suptitle(
        f"Paired median difference vs {reference} (Wilcoxon, whiskers = 95% CI)",
        fontsize=13, fontweight="bold",
    )
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.015), frameon=False)
    fig.tight_layout(rect=(0, 0.02, 1, 0.965))
    save_fig(fig, filename)
    return True


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------

def compute_pareto_frontier(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_minimize: bool = True,
    y_maximize: bool = True,
) -> pd.DataFrame:
    """Pareto-optimal subset for a two-objective tradeoff."""
    df_valid = df.dropna(subset=[c for c in (x_col, y_col) if c in df.columns]).copy()
    if df_valid.empty or x_col not in df_valid or y_col not in df_valid:
        return df_valid.iloc[0:0]

    points = df_valid[[x_col, y_col]].to_numpy(dtype=float)
    n_points = len(points)
    is_pareto = np.ones(n_points, dtype=bool)
    for i in range(n_points):
        if not is_pareto[i]:
            continue
        for j in range(n_points):
            if i == j or not is_pareto[j]:
                continue
            x_i, y_i = points[i]
            x_j, y_j = points[j]
            x_j_better = (x_j < x_i) if x_minimize else (x_j > x_i)
            x_j_eq_or_better = (x_j <= x_i) if x_minimize else (x_j >= x_i)
            y_j_better = (y_j > y_i) if y_maximize else (y_j < y_i)
            y_j_eq_or_better = (y_j >= y_i) if y_maximize else (y_j <= y_i)
            if x_j_eq_or_better and y_j_eq_or_better and (x_j_better or y_j_better):
                is_pareto[i] = False
                break
    return df_valid[is_pareto].copy()


def plot_pareto_frontier(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    filename: Path,
    x_label: str | None = None,
    y_label: str | None = None,
    x_minimize: bool = True,
    y_maximize: bool = True,
) -> bool:
    name = Path(filename).name
    df_valid = drop_missing(df, [x_col, y_col], name)
    if len(df_valid) < 2:
        return False

    pareto_df = compute_pareto_frontier(df_valid, x_col, y_col, x_minimize, y_maximize)
    pareto_df = pareto_df.sort_values(by=x_col)

    fig, ax = plt.subplots(figsize=(9.5, 6))
    non_pareto = df_valid[~df_valid.index.isin(pareto_df.index)]
    ax.scatter(non_pareto[x_col], non_pareto[y_col],
               c="lightgray", s=90, alpha=0.8, edgecolors="gray",
               label="Dominated", zorder=1)
    ax.scatter(pareto_df[x_col], pareto_df[y_col],
               c="crimson", s=150, marker="*", label="Pareto optimal", zorder=3)
    if len(pareto_df) > 1:
        ax.step(pareto_df[x_col], pareto_df[y_col], where="post",
                color="crimson", linestyle="--", alpha=0.5, linewidth=1.6, zorder=2)

    annotate_points(ax, df_valid[x_col], df_valid[y_col],
                    [abbrev(b) for b in df_valid["baseline"]])

    x_dir = "←" if x_minimize else "→"
    y_dir = "↑" if y_maximize else "↓"
    ax.annotate(f"Better {x_dir}", xy=(0.02, 0.02), xycoords="axes fraction",
                fontsize=9, color="green", ha="left")
    ax.annotate(f"Better {y_dir}", xy=(0.02, 0.065), xycoords="axes fraction",
                fontsize=9, color="green", ha="left")

    ax.set_xlabel(x_label or x_col)
    ax.set_ylabel(y_label or y_col)
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    save_fig(fig, filename)
    return True


def generate_pareto_analysis(df: pd.DataFrame, plots_dir: Path) -> list[tuple[str, str]]:
    """All Pareto tradeoff figures + the pareto_optimal_baselines.csv summary."""
    generated: list[tuple[str, str]] = []

    if plot_pareto_frontier(
        df, "latency_ms", "bertscore",
        "Pareto Frontier: Latency vs Quality",
        plots_dir / "pareto_latency_vs_quality.png",
        x_label="Median latency (ms) - lower is better",
        y_label="BERTScore - higher is better",
        x_minimize=True, y_maximize=True,
    ):
        generated.append((
            "pareto_latency_vs_quality.png",
            "Pareto frontier of latency vs quality; stars are Pareto-optimal arms "
            "(no other arm is better on both axes)."
        ))

    if plot_pareto_frontier(
        df, "ttft_ms", "bertscore",
        "Pareto Frontier: TTFT vs Quality",
        plots_dir / "pareto_ttft_vs_quality.png",
        x_label="Median TTFT (ms) - lower is better",
        y_label="BERTScore - higher is better",
        x_minimize=True, y_maximize=True,
    ):
        generated.append((
            "pareto_ttft_vs_quality.png",
            "Pareto frontier for time-to-first-token vs quality."
        ))

    if plot_pareto_frontier(
        df, "qps", "faithfulness",
        "Pareto Frontier: Throughput vs Faithfulness",
        plots_dir / "pareto_throughput_vs_faithfulness.png",
        x_label="Queries per second - higher is better",
        y_label="Faithfulness - higher is better",
        x_minimize=False, y_maximize=True,
    ):
        generated.append((
            "pareto_throughput_vs_faithfulness.png",
            "Pareto frontier for throughput vs faithfulness (higher is better on both)."
        ))

    if plot_pareto_frontier(
        df, "latency_ms", "relevance",
        "Pareto Frontier: Latency vs Relevance",
        plots_dir / "pareto_latency_vs_relevance.png",
        x_label="Median latency (ms) - lower is better",
        y_label="Context relevance - higher is better",
        x_minimize=True, y_maximize=True,
    ):
        generated.append((
            "pareto_latency_vs_relevance.png",
            "Pareto frontier for latency vs context relevance."
        ))

    pareto_summary = []
    for x_col, y_col, x_min, y_max in [
        ("latency_ms", "bertscore", True, True),
        ("ttft_ms", "bertscore", True, True),
        ("qps", "faithfulness", False, True),
        ("latency_ms", "relevance", True, True),
    ]:
        pareto_df = compute_pareto_frontier(df, x_col, y_col, x_min, y_max)
        for _, row in pareto_df.iterrows():
            pareto_summary.append({
                "tradeoff": f"{x_col}_vs_{y_col}",
                "baseline": row["baseline"],
                x_col: row[x_col],
                y_col: row[y_col],
            })
    if pareto_summary:
        pd.DataFrame(pareto_summary).to_csv(
            plots_dir / "pareto_optimal_baselines.csv", index=False)
        print(f"  wrote pareto_optimal_baselines.csv")

    return generated


# ---------------------------------------------------------------------------
# Tradeoff scatter, grouped bars, heatmap, radar
# ---------------------------------------------------------------------------

def plot_tradeoff_scatter(df: pd.DataFrame, filename: Path) -> bool:
    name = Path(filename).name
    data = drop_missing(df, ["latency_ms", "bertscore"], name)
    if len(data) < 2:
        return False
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(data["latency_ms"], data["bertscore"],
               s=70, color=sns.color_palette("colorblind")[0],
               edgecolors="black", linewidth=0.6, alpha=0.85)
    annotate_points(ax, data["latency_ms"], data["bertscore"],
                    [abbrev(b) for b in data["baseline"]])
    ax.set_xlabel("Median latency (ms)")
    ax.set_ylabel("BERTScore")
    ax.set_title("Performance vs Quality (Latency vs BERTScore)")
    save_fig(fig, filename)
    return True


def plot_grouped_bar(
    df: pd.DataFrame,
    metrics: list[str],
    title: str,
    filename: Path,
    ylabel: str = "Value",
) -> bool:
    name = Path(filename).name
    present = [m for m in metrics if m in df.columns and df[m].notna().any()]
    if not present:
        print(f"  [{name}] no data for {metrics}; figure skipped")
        return False
    data = drop_missing(df, present, name, how="all")
    if data.empty:
        return False

    melted = data.melt(id_vars=["baseline"], value_vars=present,
                       var_name="Metric", value_name="Value")
    melted["baseline"] = melted["baseline"].map(abbrev)
    order = [abbrev(b) for b in data["baseline"]]

    fig, ax = plt.subplots(figsize=(max(10, 0.85 * len(data) + 4), 5.5))
    sns.barplot(data=melted, x="baseline", y="Value", hue="Metric",
                order=order, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Baseline (see baseline_abbreviations.txt)")
    ax.set_ylabel(ylabel)
    ax.legend(title="Metric", bbox_to_anchor=(1.01, 1), loc="upper left",
              borderaxespad=0)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    save_fig(fig, filename)
    return True


def plot_heatmap(df: pd.DataFrame, metrics: list[str], title: str,
                 filename: Path) -> bool:
    name = Path(filename).name
    present = [m for m in metrics if m in df.columns and df[m].notna().any()]
    if len(present) < 2:
        print(f"  [{name}] fewer than 2 metrics available; figure skipped")
        return False
    data = drop_missing(df, present, name, how="all")
    if data.empty:
        return False

    raw = data.set_index("baseline")[present].astype(float)
    norm = raw.copy()
    for col in present:
        lo, hi = norm[col].min(), norm[col].max()
        norm[col] = (norm[col] - lo) / (hi - lo) if hi > lo else 0.5
        # Invert latency-like columns so green = better holds for EVERY column
        # (the cell text still shows the raw value).
        if any(k in col.lower() for k in ("latency", "ttft", "tpot")):
            norm[col] = 1 - norm[col]

    annot = raw.map(lambda v: _fmt(v) if pd.notna(v) else "")
    fig, ax = plt.subplots(figsize=(1.35 * len(present) + 3,
                                    0.45 * len(raw) + 2))
    sns.heatmap(norm, annot=annot.values, fmt="", cmap="RdYlGn", center=0.5,
                ax=ax, annot_kws={"fontsize": 8},
                cbar_kws={"label": "Column-normalized (0-1); cell text = raw value"})
    ax.set_yticklabels([abbrev(b) for b in raw.index], rotation=0)
    ax.set_title(title)
    ax.set_ylabel("Baseline (see baseline_abbreviations.txt)")
    save_fig(fig, filename)
    return True


def plot_radar(df: pd.DataFrame, metrics: list[str], title: str,
               filename: Path) -> bool:
    """Core arms only, lines only (overlapping fills hid identical arms)."""
    name = Path(filename).name
    core = df[df["baseline"].isin(CORE_ARMS)]
    data = drop_missing(core, metrics, name)
    if len(data) < 2:
        print(f"  [{name}] fewer than 2 core arms with complete metrics; skipped")
        return False

    df_norm = data.copy()
    for col in metrics:
        lo, hi = df_norm[col].min(), df_norm[col].max()
        df_norm[col] = (df_norm[col] - lo) / (hi - lo) if hi > lo else 0.5
    inverted = [c for c in metrics
                if "latency" in c.lower() or "ttft" in c.lower() or "tpot" in c.lower()]
    for col in inverted:
        df_norm[col] = 1 - df_norm[col]

    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw=dict(polar=True))
    palette = sns.color_palette("colorblind", len(df_norm))
    for idx, (_, row) in enumerate(df_norm.iterrows()):
        values = [row[m] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=1.8, markersize=4,
                label=abbrev(row["baseline"]), color=palette[idx])
        # deliberately NO ax.fill: overlapping fills hid identical arms

    labels = [f"{m}\n(inverted)" if m in inverted else m for m in metrics]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{title}\n(core arms only; outer = better)", size=13, y=1.09)
    ax.legend(loc="upper left", bbox_to_anchor=(1.12, 1.05), borderaxespad=0)
    save_fig(fig, filename)
    return True


# ---------------------------------------------------------------------------
# Latency breakdown, quality comparison, bubble, speedup, ranking table
# ---------------------------------------------------------------------------

def plot_latency_breakdown(df: pd.DataFrame, title: str, filename: Path) -> bool:
    name = Path(filename).name
    data = drop_missing(df, ["ttft_ms", "latency_ms"], name).copy()
    if data.empty:
        return False

    data["generation_ms"] = data["latency_ms"] - data["ttft_ms"]
    negative = data[data["generation_ms"] < 0]
    if not negative.empty:
        bad = ", ".join(f"{b} ({g:.1f} ms)" for b, g in
                        zip(negative["baseline"], negative["generation_ms"]))
        print(f"  [{name}] WARNING dropped arms with NEGATIVE generation time "
              f"(latency < TTFT, timing inconsistent): {bad}")
    data = data[data["generation_ms"] >= 0]
    if data.empty:
        print(f"  [{name}] no arms left after dropping negative generation; skipped")
        return False

    n = len(data)
    fig, ax = plt.subplots(figsize=(9, max(3.2, 0.42 * n + 1.4)))
    y = np.arange(n)
    palette = sns.color_palette("colorblind")
    ax.barh(y, data["ttft_ms"], height=0.62, label="TTFT (prefill)",
            color=palette[2])
    ax.barh(y, data["generation_ms"], height=0.62, left=data["ttft_ms"],
            label="Generation (decode)", color=palette[0])
    ax.set_yticks(y)
    ax.set_yticklabels(data["baseline"])
    ax.invert_yaxis()
    ax.yaxis.grid(False)

    span = float(data["latency_ms"].max()) or 1.0
    for yi, total in zip(y, data["latency_ms"]):
        ax.text(total + 0.015 * span, yi, f"{total:,.0f} ms",
                va="center", ha="left", fontsize=8)
    ax.set_xlim(right=span * 1.16)
    ax.set_xlabel("Time (ms) - pooled per-example median")
    ax.set_title(title)
    ax.legend(loc="lower right")
    save_fig(fig, filename)
    return True


def plot_quality_comparison(df: pd.DataFrame, title: str, filename: Path) -> bool:
    quality_metrics = ["grounding_score", "faithfulness", "relevance",
                       "bertscore", "rouge_l"]
    return plot_grouped_bar(df, quality_metrics, title, filename,
                            ylabel="Score (pooled per-example median)")


def plot_efficiency_bubble(df: pd.DataFrame, title: str, filename: Path) -> bool:
    name = Path(filename).name
    data = drop_missing(df, ["latency_ms", "bertscore", "qps"], name)
    if len(data) < 2:
        return False

    fig, ax = plt.subplots(figsize=(9.5, 7))
    qps = data["qps"].to_numpy(dtype=float)
    qmin, qmax = float(qps.min()), float(qps.max())
    if qmax > qmin:
        sizes = 100 + 900 * (qps - qmin) / (qmax - qmin)
    else:
        sizes = np.full(len(qps), 400.0)

    ax.scatter(data["latency_ms"], data["bertscore"], s=sizes, alpha=0.6,
               c=range(len(data)), cmap="viridis",
               edgecolors="black", linewidth=0.8)
    annotate_points(ax, data["latency_ms"], data["bertscore"],
                    [abbrev(b) for b in data["baseline"]])

    # Bubble-size legend: three reference QPS values.
    ref_qps = [qmin, (qmin + qmax) / 2, qmax] if qmax > qmin else [qmin]
    handles = []
    for q in ref_qps:
        s = 100 + 900 * (q - qmin) / (qmax - qmin) if qmax > qmin else 400.0
        handles.append(ax.scatter([], [], s=s, facecolors="none",
                                  edgecolors="black", label=f"{q:.2f} QPS"))
    ax.legend(handles=handles, title="Bubble size = throughput",
              loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0,
              labelspacing=1.6, borderpad=1.0)

    ax.set_xlabel("Median latency (ms) - lower is better")
    ax.set_ylabel("BERTScore - higher is better")
    ax.set_title(title)
    save_fig(fig, filename)
    return True


def plot_speedup_chart(df: pd.DataFrame, reference: str, title: str,
                       filename: Path) -> bool:
    """Speedups from the pooled per-example medians in build_summary (NOT
    mean-of-trial-means, which disagreed in sign with the paired stats)."""
    name = Path(filename).name
    if reference not in df["baseline"].values:
        print(f"  [{name}] reference '{reference}' absent; figure skipped")
        return False
    ref_row = df[df["baseline"] == reference].iloc[0]
    ref_latency = ref_row.get("latency_ms")
    ref_ttft = ref_row.get("ttft_ms")
    if pd.isna(ref_latency) or ref_latency <= 0 or pd.isna(ref_ttft) or ref_ttft <= 0:
        print(f"  [{name}] reference '{reference}' has no latency/ttft; skipped")
        return False

    data = drop_missing(df[df["baseline"] != reference],
                        ["latency_ms", "ttft_ms"], name).copy()
    if data.empty:
        return False
    data["latency_speedup"] = float(ref_latency) / data["latency_ms"]
    data["ttft_speedup"] = float(ref_ttft) / data["ttft_ms"]
    labels = [abbrev(b) for b in data["baseline"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, col, sub in ((ax1, "latency_speedup", "Latency"),
                         (ax2, "ttft_speedup", "TTFT")):
        vals = data[col].to_numpy(dtype=float)
        colors = ["#27ae60" if v > 1 else "#e74c3c" for v in vals]
        bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6)
        ax.axhline(y=1, color="black", linestyle="--", linewidth=1)
        ax.set_ylabel("Speedup (x)")
        ax.set_title(f"{sub} speedup vs {reference}")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        ax.xaxis.grid(False)
        for bar, val in zip(bars, vals):
            ax.annotate(f"{val:.2f}x",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_ylim(top=float(np.nanmax(vals)) * 1.14)
    fig.suptitle(f"{title}\n(pooled per-example medians)", fontsize=13, y=1.03)
    save_fig(fig, filename)
    return True


def plot_ranking_table(df: pd.DataFrame, filename: Path) -> bool:
    name = Path(filename).name
    metrics_config = {
        "qps": {"higher_better": True, "label": "Throughput (QPS)"},
        "ttft_ms": {"higher_better": False, "label": "TTFT (ms)"},
        "latency_ms": {"higher_better": False, "label": "Latency (ms)"},
        "tokens_per_sec": {"higher_better": True, "label": "Tokens/sec"},
        "faithfulness": {"higher_better": True, "label": "Faithfulness"},
        "relevance": {"higher_better": True, "label": "Relevance"},
        "bertscore": {"higher_better": True, "label": "BERTScore"},
        "rouge_l": {"higher_better": True, "label": "ROUGE-L"},
    }

    rankings: dict[str, list[str]] = {}
    for metric, config in metrics_config.items():
        if metric not in df.columns or df[metric].isna().all():
            print(f"  [{name}] metric '{metric}' unavailable; column omitted")
            continue
        sub = df.dropna(subset=[metric])
        ranked = sub.sort_values(metric, ascending=not config["higher_better"])
        rankings[config["label"]] = [abbrev(b) for b in ranked["baseline"]]
    if not rankings:
        return False

    metrics_list = list(rankings.keys())
    n_rows = max(len(v) for v in rankings.values())
    fig, ax = plt.subplots(figsize=(1.5 + 1.5 * len(metrics_list),
                                    1.8 + 0.42 * n_rows))
    ax.axis("off")

    cell_text = []
    for i in range(n_rows):
        cell_text.append([rankings[m][i] if i < len(rankings[m]) else ""
                          for m in metrics_list])
    row_labels = [f"#{i + 1}" for i in range(n_rows)]

    table = ax.table(cellText=cell_text, rowLabels=row_labels,
                     colLabels=metrics_list, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(len(metrics_list))))
    table.scale(1, 1.55)

    for j in range(len(metrics_list)):
        table[(1, j)].set_facecolor("#2ecc71")
        table[(1, j)].set_text_props(fontweight="bold", color="white")

    ax.set_title("Baseline rankings by metric (best -> worst; abbreviated names)",
                 fontsize=13, pad=18)
    save_fig(fig, filename)
    return True


# ---------------------------------------------------------------------------
# Explanations + main
# ---------------------------------------------------------------------------

def write_plot_explanations(plots_dir: Path, plots: list[tuple[str, str]]) -> None:
    lines = ["Plot explanations", "=================", ""]
    for fname, desc in plots:
        lines.append(f"- {fname}: {desc}")
    lines += ["", f"Caveat: {CAVEAT_LINE}", ""]
    (plots_dir / "plots_explained.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate analysis figures from the canonical results loader.")
    parser.add_argument(
        "--results-dir",
        default=os.environ.get("CAGE_RUN_ROOT", "results"),
        help="Run root scanned for per-query results (default: $CAGE_RUN_ROOT, else results).",
    )
    parser.add_argument(
        "--plots-dir",
        default=os.path.join(os.environ.get("CAGE_RUN_ROOT", "results"), "plots"),
        help="Directory to write plots and summary CSV (default: <run-root>/plots).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    apply_style()

    df = build_summary(results_dir)  # SystemExit when the run root has zero cells
    if df.empty:
        raise SystemExit("No result rows found.")
    print(f"Loaded {len(df)} baselines from {results_dir}")

    # Stale figures deleted by the overhaul must not survive a regeneration.
    for stale in DELETED_FIGURES:
        stale_path = plots_dir / stale
        if stale_path.exists():
            stale_path.unlink()
            print(f"  removed stale figure {stale} (deleted by 2026-07-15 overhaul)")

    write_abbrev_legend(plots_dir)

    def _n(col: str) -> int:
        c = f"{col}_n"
        return int(df[c].max()) if c in df.columns and df[c].notna().any() else 0

    pq_note = "pooled per-example median; whiskers = 95% bootstrap CI"
    explained: list[tuple[str, str]] = []

    by_baseline = [
        ("qps", "Throughput (QPS) by Baseline", "Queries per second",
         "qps_by_baseline.png", False,
         f"mean across trials; whiskers = +-1 std (n={_n('qps')} trials)"),
        ("ttft_ms", "TTFT by Baseline", "Time to first token (ms)",
         "ttft_by_baseline.png", False,
         f"{pq_note} (n={_n('ttft_ms')} examples)"),
        ("latency_ms", "End-to-End Latency by Baseline", "Latency (ms)",
         "latency_by_baseline.png", False,
         f"{pq_note} (n={_n('latency_ms')} examples)"),
        ("tokens_per_sec", "Throughput (tokens/sec) by Baseline", "Tokens per second",
         "tokens_per_sec_by_baseline.png", False,
         f"mean across trials; whiskers = +-1 std (n={_n('tokens_per_sec')} trials)"),
        ("bertscore", "Completeness (BERTScore) by Baseline",
         "BERTScore (baseline-rescaled, signed)",
         "bertscore_by_baseline.png", True,
         f"{pq_note} (n={_n('bertscore')} examples); dashed = 0"),
        ("rouge_l", "Completeness (ROUGE-L) by Baseline", "ROUGE-L F1",
         "rouge_l_by_baseline.png", False,
         f"{pq_note} (n={_n('rouge_l')} examples)"),
    ]
    for col, title, xlabel, fname, refline, note in by_baseline:
        if plot_metric_by_baseline(df, col, title, xlabel, plots_dir / fname,
                                   note=note, refline_zero=refline):
            explained.append((fname, f"{title}: horizontal bars, {note}."))

    if plot_tradeoff_scatter(df, plots_dir / "tradeoff_latency_vs_bertscore.png"):
        explained.append(("tradeoff_latency_vs_bertscore.png",
                          "Scatter of latency vs quality to show the tradeoff."))

    # Forest plot straight from the stats JSON (skipped, with a message, when absent).
    stats_json = results_dir / "stats" / "all_results" / "phase2_stats.json"
    if plot_forest_from_stats(stats_json, plots_dir / "delta_vs_no_cache_forest.png"):
        explained.append((
            "delta_vs_no_cache_forest.png",
            "Forest plot of Wilcoxon paired median differences vs no_cache, rendered "
            "directly from phase2_stats.json (whiskers = 95% CI; filled markers = "
            "Holm-significant). By construction it cannot disagree with the stats tables."
        ))

    print("\nGenerating Pareto frontier analysis...")
    explained.extend(generate_pareto_analysis(df, plots_dir))

    print("\nGenerating overview figures...")
    radar_metrics = ["qps", "ttft_ms", "latency_ms", "faithfulness",
                     "relevance", "bertscore"]
    radar_available = [m for m in radar_metrics
                       if m in df.columns and df[m].notna().any()]
    if len(radar_available) >= 3 and plot_radar(
            df, radar_available, "Multi-Dimensional Performance Profile",
            plots_dir / "radar_performance_profile.png"):
        explained.append((
            "radar_performance_profile.png",
            "Radar chart of the core arms only, lines only (no fills, which hid "
            "identical arms). Latency-like axes inverted so outer = better."
        ))

    heatmap_metrics = ["qps", "tokens_per_sec", "ttft_ms", "latency_ms",
                       "faithfulness", "relevance", "bertscore", "rouge_l"]
    if plot_heatmap(df, heatmap_metrics, "Performance Heatmap",
                    plots_dir / "heatmap_all_metrics.png"):
        explained.append((
            "heatmap_all_metrics.png",
            "Heatmap colored by column-normalized score (green = better within the "
            "column; latency-like columns inverted); the cell text shows the RAW "
            "metric value."
        ))

    if plot_latency_breakdown(df, "Latency Breakdown: Prefill vs Decode",
                              plots_dir / "latency_breakdown_stacked.png"):
        explained.append((
            "latency_breakdown_stacked.png",
            "Horizontal stacked bars of TTFT (prefill) vs generation (decode). Arms "
            "with negative generation time (latency < TTFT) are dropped with a "
            "printed warning, never clipped to zero."
        ))

    if plot_quality_comparison(df, "Quality Metrics Comparison",
                               plots_dir / "quality_metrics_grouped.png"):
        explained.append((
            "quality_metrics_grouped.png",
            "Grouped bars of all quality metrics (grounding, faithfulness, relevance, "
            "BERTScore, ROUGE-L) side by side."
        ))

    if plot_efficiency_bubble(df, "Efficiency: Latency vs Quality vs Throughput",
                              plots_dir / "efficiency_bubble_chart.png"):
        explained.append((
            "efficiency_bubble_chart.png",
            "Bubble chart: X = latency, Y = BERTScore, bubble size = QPS (see the "
            "size legend)."
        ))

    if plot_speedup_chart(df, "no_cache", "Performance Speedup vs No-Cache Baseline",
                          plots_dir / "speedup_vs_no_cache.png"):
        explained.append((
            "speedup_vs_no_cache.png",
            "Speedup factor vs no_cache computed from the pooled per-example medians "
            "(green = faster, red = slower)."
        ))

    if plot_ranking_table(df, plots_dir / "ranking_table.png"):
        explained.append((
            "ranking_table.png",
            "Ranking table (abbreviated names) showing which arm ranks #1, #2, ... "
            "per metric."
        ))

    if plot_grouped_bar(df, ["qps", "tokens_per_sec"],
                        "Throughput Metrics Comparison",
                        plots_dir / "throughput_metrics_grouped.png",
                        ylabel="Value (trial-level mean)"):
        explained.append((
            "throughput_metrics_grouped.png",
            "Grouped comparison of trial-level throughput metrics (QPS, tokens/sec)."
        ))

    explained.append(("baseline_abbreviations.txt",
                      "Mapping from the abbreviations used in figures to full names."))
    write_plot_explanations(plots_dir, explained)

    summary_path = plots_dir / "latest_metrics_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"\nSaved plots to {plots_dir}")
    print(f"Saved summary CSV to {summary_path}")


if __name__ == "__main__":
    main()
