#!/usr/bin/env python3
"""Generate the run's publication figure set from the canonical results loader.

2026-07-16 publication overhaul (supersedes the 2026-07-15 set):
- Every figure, legend and table uses the canonical DISPLAY_NAME mapping from
  _plot_style (Title Case, no underscores) and the Arial fallback font chain.
- No figure is wider than ~7.2 in at 300 dpi (ABNT text block); anything with all
  ~20 baselines uses horizontal bars or forest panels.
- The single overloaded forest + speedup chart is replaced by two forest figures
  (serving vs quality deltas) rendered straight from phase2_stats.json.
- Grounding is saturated at the median (1.0), so grounding is always plotted as a
  MEAN with a bootstrap CI, never as a bar of medians.
- NLI faithfulness is invalid for the cells in NLI_INVALID_CELLS (premise exceeds
  the validity envelope); faithfulness panels exclude/dagger those cells.
- hit@k / MRR are never plotted (oracle retrieval, zero variance).

Figure set (see plots_explained.txt for the reader-facing description):
  F1 main_results_table.md/.tex   F2 forest_serving/forest_quality
  F3 pareto_ttft_vs_f1answerable  F4 mechanism_reuse_vs_ttft
  F5 cag_true_paired              F6 ttft_percentiles
  F7 speculative_tpot (+ table)   F8 quality_{grounding,faithfulness,
                                     f1_answerable,completeness}
  T1-T6 tri-lemma centerpiece: trilemma_overview (grouped columns),
    trilemma_bubble, trilemma_column_line, trilemma_scores_heat,
    memory_resident_vs_marginal, trilemma_table.md/.tex
  plus improved keepers: ttft/latency_by_baseline, latency_breakdown_stacked,
  pareto_latency_vs_quality, heatmap_all_metrics (appendix).

2026-07-16 review iteration: mechanism_cache_reuse (crossing leader lines),
ttft_ecdf (confusing view) and quality_panels (2x2 too dense) are DELETED and
replaced; the Pareto figures drop the numbered key for family color+marker
encoding with selective direct labels; a tri-lemma set (serving x quality x
KV-memory) is added as the centerpiece. MEMORY is a per-query KV context
footprint PROXY: resident = median prompt tokens, marginal = median new KV
tokens materialized per query (prompt - cached, missing cached = 0); a swept
memory-pressure axis is future work.

2026-07-16 review passes: the tri-lemma set was simplified twice on reviewer
feedback. Final form: trilemma_overview = an Excel-simple grouped column chart
of the 0-100 axis scores on the 9 key arms; trilemma_column_line = a classic
column+line combo (F1 columns, TTFT line, twin axes) on the same arms; the
bubble chart is kept; trilemma_scores_heat keeps the full 20-arm score matrix.
The stacked-panel overview and trilemma_parallel were retired as too complex.

Central tendency for per-query metrics is the POOLED PER-EXAMPLE MEDIAN (the same
per-example estimand the Wilcoxon tables use); quality panels additionally show the
pooled per-example MEAN with a bootstrap CI where medians saturate. Throughput comes
from trial-level wall-clock facts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _plot_style import (  # noqa: E402
    FAMILY_COLOR,
    FAMILY_MARKER,
    FAMILY_SHORT_ORDER,
    FULL_WIDTH_IN,
    NLI_DAGGER_NOTE,
    NLI_INVALID_CELLS,
    annotate_selected,
    apply_style,
    display,
    drop_missing,
    family_short,
    order_cells,
    save_fig,
    stars,
)
from _pub_tables import (  # noqa: E402
    TRILEMMA_SCORE_COLS,
    trilemma_scores,
    write_main_results_table,
    write_speculative_table,
    write_trilemma_table,
)
from _results_loader import (  # noqa: E402
    headline_rows,
    load_results_long,
    load_trial_scalars,
    metric_values,
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
    "grounding": "grounding_score",
    "faithfulness": "faithfulness",
    "relevance": "context_relevance",
    "bertscore": "completeness_bertscore",
    "rouge_l": "completeness_rouge_l",
    "f1_score": "f1_score",
    "f1_answerable": "f1_answerable",
}

# Frame column -> trial_*/metrics.json performance key (wall-clock trial facts).
TRIAL_METRICS = {
    "qps": "queries_per_second",
    "tokens_per_sec": "tokens_per_second",
}

# Figures/artifacts deleted by the 2026-07-16 publication overhaul; removed from the
# plots dir when present so stale copies cannot linger next to the regenerated set.
DELETED_FIGURES = (
    # pre-2026-07-15 legacy
    "tokens_per_sec_share_pie.png", "tokens_per_sec_pareto.png",
    # replaced by forest_serving/forest_quality
    "delta_vs_no_cache_forest.png", "speedup_vs_no_cache.png",
    # replaced by quality_panels
    "quality_metrics_grouped.png",
    "bertscore_by_baseline.png", "rouge_l_by_baseline.png",
    # dropped as unreadable / redundant / not publication-grade
    "radar_performance_profile.png", "ranking_table.png",
    "efficiency_bubble_chart.png", "tradeoff_latency_vs_bertscore.png",
    "qps_by_baseline.png", "tokens_per_sec_by_baseline.png",
    "throughput_metrics_grouped.png",
    "pareto_ttft_vs_quality.png", "pareto_latency_vs_relevance.png",
    "pareto_throughput_vs_faithfulness.png",
    # abbreviations retired: figures now carry full display names
    "baseline_abbreviations.txt",
    # 2026-07-16 review iteration
    "mechanism_cache_reuse.png",   # crossing leader lines -> mechanism_reuse_vs_ttft
    "ttft_ecdf.png",               # confusing view -> ttft_percentiles
    "quality_panels.png",          # 2x2 too dense -> one figure per quality metric
    # 2026-07-16 third review pass
    "trilemma_parallel.png",       # too complex -> trilemma_column_line
)

CAVEAT_LINE = (
    "All per-query aggregates are pooled per-example statistics (same per-example "
    "estimand as the Wilcoxon tables); medians for serving metrics, means with "
    "bootstrap CIs for saturating quality metrics; throughput is trial-level. "
    "N = 300 valid paired queries per cell (3 trials x 100 questions, repeat-0 "
    "rows; every example_id is a distinct per-example unit)."
)

SIG_COLOR = "#0173b2"   # colorblind-safe blue for Holm-significant markers
NS_COLOR = "0.55"       # grey for non-significant

# Arms that always get a direct label on family-coded scatters, on top of the
# per-figure Pareto-optimal set (reviewer rule, 2026-07-16).
SCATTER_LABEL_ARMS = frozenset({
    "no_cache", "cag_true_off", "cag_true_on", "compressed_rag",
})

# The five arms singled out on the T2 bubble chart.
BUBBLE_LABEL_ARMS = frozenset({
    "no_cache", "compressed_rag", "cag_true_on", "cag_true_off",
    "spec_qwen8b_eagle3_cag",
})

# Key arms for the TTFT percentile figure (span the reuse spectrum).
KEY_ARMS = ["no_cache", "prefix_cache", "cag_true_off", "cag_true_on",
            "compressed_rag"]

KV_PROXY_NOTE = ("KV footprint proxy (prompt-token counts); a swept "
                 "memory-pressure axis is future work.")

TRILEMMA_AXES_NOTE = (
    "Axes -- SERVING: TTFT median (ms; lower is better). QUALITY: F1 on "
    "answerable (mean; valid for all arms; higher is better). MEMORY: per-query "
    "KV context footprint -- resident = median prompt tokens held in KV; "
    "marginal = median NEW KV tokens materialized per query "
    "(prompt - cached, missing cached = 0). " + KV_PROXY_NOTE)


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
                  seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-baseline wide frame, long frame) from the canonical loader.

    Per-query metrics: pooled per-example MEDIAN + bootstrap CI of the median
    (<col>, <col>_ci_low/high) AND pooled per-example MEAN + bootstrap CI of the
    mean (<col>_mean, <col>_mean_ci_low/high), plus <col>_n examples.
    Derived mechanism columns: cached_pct (token-weighted cached-prompt share, %),
    abstention_pct (share of predicted no-answer responses, %), and the tri-lemma
    memory proxies resident_kv (median prompt tokens per query) and marginal_kv
    (median NEW KV tokens materialized per query = prompt - cached, missing
    cached treated as 0).
    Trial metrics (qps, tokens_per_sec): mean across trials +- std.
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
                r[f"{col}_mean_ci_low"] = float(sm.loc[cell, "ci95_low"])
                r[f"{col}_mean_ci_high"] = float(sm.loc[cell, "ci95_high"])
            r[f"{col}_n"] = int(len(vals))

    # Mechanism columns from the valid rep-0 rows.
    h = headline_rows(long_df).copy()
    h["_cached"] = pd.to_numeric(h.get("cached_prompt_tokens"), errors="coerce").fillna(0.0)
    h["_prompt"] = pd.to_numeric(h.get("prompt_tokens"), errors="coerce")
    for cell, g in h.groupby("cell"):
        if cell not in rows:
            continue
        prompt_total = float(g["_prompt"].sum())
        if prompt_total > 0:
            rows[cell]["cached_pct"] = 100.0 * float(g["_cached"].sum()) / prompt_total
        prompts = g["_prompt"].dropna()
        if not prompts.empty:
            # Tri-lemma memory proxies (per-query KV context footprint).
            rows[cell]["resident_kv"] = float(prompts.median())
            marginal = (g["_prompt"] - g["_cached"]).dropna()
            rows[cell]["marginal_kv"] = float(marginal.median())
        rows[cell]["n_rows"] = int(len(g))
    pe_abst = per_example(long_df, "predicted_no_answer")
    for cell, g in pe_abst.groupby("cell"):
        if cell in rows:
            rows[cell]["abstention_pct"] = 100.0 * float(g["value"].mean())

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
                r[f"{col}_n"] = int(agg.loc[cell, "count"])

    return pd.DataFrame([rows[c] for c in cells]), long_df


def load_stats(path: Path, what: str) -> dict | None:
    if not path.is_file():
        print(f"  [{what}] stats JSON not found at {path}; dependent figures skipped")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  [{what}] could not read {path}: {exc}; dependent figures skipped")
        return None


def paired_mean_deltas(long_df: pd.DataFrame, metric: str,
                       reference: str = "no_cache", iters: int = 5000,
                       seed: int = 42) -> dict[str, tuple[float, float, float, int]]:
    """cell -> (mean paired delta vs reference, ci_lo, ci_hi, n_pairs).

    Pairs the pooled per-example values on example_id (the Wilcoxon unit) and
    bootstraps the MEAN of the paired differences. Used where the median delta
    saturates at 0 (grounding).
    """
    pe = per_example(long_df, metric)
    if pe.empty:
        return {}
    ref = pe[pe["cell"] == reference].set_index("example_id")["value"]
    out: dict[str, tuple[float, float, float, int]] = {}
    rng = np.random.default_rng(seed)
    for cell, g in pe.groupby("cell"):
        if cell == reference:
            continue
        joined = g.set_index("example_id")["value"].to_frame("b").join(
            ref.to_frame("r"), how="inner")
        d = (joined["b"] - joined["r"]).to_numpy(dtype=float)
        if len(d) < 2:
            continue
        idx = rng.integers(0, len(d), size=(iters, len(d)))
        means = d[idx].mean(axis=1)
        out[cell] = (float(d.mean()), float(np.percentile(means, 2.5)),
                     float(np.percentile(means, 97.5)), int(len(d)))
    return out


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


def _fmt_p(p: float) -> str:
    if p is None or np.isnan(p):
        return "p = n/a"
    return f"p < 1e-99" if p < 1e-99 else f"p = {p:.1e}"


def _wrap(text: str, width: int) -> str:
    """Hard-wrap figure caption text so tight bboxes never exceed FULL_WIDTH_IN.

    matplotlib's Text(wrap=True) estimates line width from mean glyph width and
    routinely overshoots the canvas by ~0.2-0.4 in; explicit wrapping is the only
    way to guarantee the <= 7.4 in page-width budget.
    """
    return "\n".join(textwrap.wrap(text, width=width))


def _asym_xerr(vals: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    err_lo = np.nan_to_num(np.clip(vals - lo, 0, None))
    err_hi = np.nan_to_num(np.clip(hi - vals, 0, None))
    return np.vstack([err_lo, err_hi])


def _with_family(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of df with a _family column (short family label per arm)."""
    out = df.copy()
    trees = out["tree"] if "tree" in out.columns else pd.Series("", index=out.index)
    out["_family"] = [family_short(str(b), str(t or ""))
                      for b, t in zip(out["baseline"], trees)]
    return out


def _family_scatter(ax, data: pd.DataFrame, x_col: str, y_col: str,
                    s: float = 52) -> None:
    """One point per arm, color+marker encoded by family (fixed encoding)."""
    for fam in FAMILY_SHORT_ORDER:
        sub = data[data["_family"] == fam]
        if sub.empty:
            continue
        ax.scatter(sub[x_col], sub[y_col], s=s, marker=FAMILY_MARKER[fam],
                   color=FAMILY_COLOR[fam], edgecolors="black", linewidth=0.5,
                   alpha=0.9, zorder=3, label=fam)


# ---------------------------------------------------------------------------
# By-baseline horizontal bars (kept: ttft, latency)
# ---------------------------------------------------------------------------

def plot_metric_by_baseline(
    df: pd.DataFrame,
    col: str,
    title: str,
    xlabel: str,
    filename: Path,
    note: str = "",
) -> bool:
    name = Path(filename).name
    data = drop_missing(df, [col], name)
    if data.empty:
        print(f"  [{name}] no data; figure skipped")
        return False

    n = len(data)
    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN, max(3.0, 0.30 * n + 1.2)))
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

    ax.barh(y, vals, xerr=xerr, height=0.64, capsize=2.5,
            color=sns.color_palette("colorblind")[0],
            error_kw={"elinewidth": 0.9, "alpha": 0.85})
    ax.set_yticks(y)
    ax.set_yticklabels([display(b) for b in data["baseline"]])
    ax.invert_yaxis()
    ax.yaxis.grid(False)

    # Value labels just past the bar end / CI whisker.
    ends = np.maximum(vals, hi)
    starts = np.minimum(np.minimum(vals, lo), 0.0)
    span = float(np.nanmax(ends) - np.nanmin(starts)) or 1.0
    for yi, v, e in zip(y, vals, ends):
        ax.text(e + 0.015 * span, yi, _fmt(v), va="center", ha="left", fontsize=7)
    right = float(np.nanmax(ends)) + 0.15 * span
    left = float(np.nanmin(starts))
    if left < 0:
        left -= 0.04 * span
    ax.set_xlim(left=left, right=right)

    ax.set_xlabel(xlabel + (f"\n{note}" if note else ""))
    ax.set_title(title)
    save_fig(fig, filename)
    return True


# ---------------------------------------------------------------------------
# F2: forest panels (serving / quality) straight from phase2_stats.json
# ---------------------------------------------------------------------------

def _stats_entries(stats_payload: dict, metric: str) -> dict[str, tuple]:
    """cell -> (median_diff, lo, hi, significant) from a stats payload."""
    out: dict[str, tuple] = {}
    for comp in stats_payload.get("comparisons") or []:
        if comp.get("metric") != metric or comp.get("median_diff") is None:
            continue
        holm = comp.get("p_value_holm")
        out[comp["baseline"]] = (
            float(comp["median_diff"]),
            comp.get("ci95_low"), comp.get("ci95_high"),
            holm is not None and float(holm) < 0.05,
        )
    return out


def plot_forest_panels(panels: list[dict], filename: Path, suptitle: str,
                       footnote: str) -> bool:
    """Horizontal forest small-multiples; one panel per metric, shared cell axis.

    panels: [{"title", "xlabel", "entries": {cell: (val, lo, hi, significant)}}]
    """
    name = Path(filename).name
    panels = [p for p in panels if p["entries"]]
    if not panels:
        print(f"  [{name}] no comparisons available; skipped")
        return False
    cells = order_cells({c for p in panels for c in p["entries"]})
    n = len(cells)
    fig, axes = plt.subplots(
        1, len(panels), figsize=(FULL_WIDTH_IN, max(3.2, 0.30 * n + 2.0)),
        sharey=True)
    axes = np.atleast_1d(axes)

    for ax, panel in zip(axes, panels):
        for yi, cell in enumerate(cells):
            e = panel["entries"].get(cell)
            if e is None:
                continue
            val, lo, hi, sig = e
            xerr = None
            if lo is not None and hi is not None:
                xerr = _asym_xerr(np.array([val]),
                                  np.array([float(lo)]), np.array([float(hi)]))
            color = SIG_COLOR if sig else NS_COLOR
            ax.errorbar(val, yi, xerr=xerr, fmt="o", markersize=4.2,
                        color=color, markerfacecolor=color if sig else "white",
                        markeredgecolor=color, capsize=2, elinewidth=0.9, zorder=3)
        ax.axvline(0.0, color="0.3", linewidth=0.9, linestyle="--", zorder=1)
        ax.set_yticks(np.arange(n))
        ax.set_yticklabels([display(c) for c in cells])
        ax.set_ylim(n - 0.5, -0.5)  # first cell on top
        ax.yaxis.grid(True, alpha=0.15)
        ax.xaxis.grid(True, alpha=0.3)
        ax.set_title(panel["title"], fontsize=9.5)
        ax.set_xlabel(panel["xlabel"], fontsize=8)
        ax.tick_params(axis="x", labelsize=7)

    handles = [
        plt.Line2D([], [], linestyle="", marker="o", markersize=5, color=SIG_COLOR,
                   markerfacecolor=SIG_COLOR, label="significant (Holm p < 0.05)"),
        plt.Line2D([], [], linestyle="", marker="o", markersize=5, color=NS_COLOR,
                   markerfacecolor="white", label="not significant"),
    ]
    fig.suptitle(suptitle, fontsize=11, fontweight="bold")
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0.035), frameon=False, fontsize=8)
    fig.text(0.5, 0.005, footnote, ha="center", va="bottom", fontsize=7,
             color="0.35")
    fig.tight_layout(rect=(0, 0.055, 1, 0.95))
    save_fig(fig, filename)
    return True


# ---------------------------------------------------------------------------
# Pareto frontier (F3 + kept latency-vs-quality variant)
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
    x_label: str,
    y_label: str,
    x_minimize: bool = True,
    y_maximize: bool = True,
    note: str = "",
    x_log: bool = False,
) -> bool:
    """Family-coded Pareto scatter (2026-07-16 review: no numbered key).

    Points are color+marker coded by arm family; only the Pareto-optimal arms
    plus the reviewer-named anchors (No Cache, CAG True Off/On, Compressed RAG)
    get short direct labels (no leader lines, so labels can never cross).
    """
    name = Path(filename).name
    df_valid = _with_family(drop_missing(df, [x_col, y_col], name))
    if len(df_valid) < 2:
        return False

    pareto_df = compute_pareto_frontier(df_valid, x_col, y_col, x_minimize, y_maximize)
    pareto_df = pareto_df.sort_values(by=x_col)
    pareto_set = set(pareto_df["baseline"])

    fig = plt.figure(figsize=(FULL_WIDTH_IN, 4.8))
    ax = fig.add_axes((0.09, 0.14, 0.60, 0.76))
    if len(pareto_df) > 1:
        ax.step(pareto_df[x_col], pareto_df[y_col], where="post",
                color="crimson", linestyle="--", alpha=0.55, linewidth=1.3,
                zorder=2, label="Pareto frontier")
    _family_scatter(ax, df_valid, x_col, y_col, s=52)
    # Ring the Pareto-optimal points so optimality reads independently of family.
    ax.scatter(pareto_df[x_col], pareto_df[y_col], s=150, facecolors="none",
               edgecolors="crimson", linewidth=1.2, zorder=4,
               label="Pareto optimal")

    if x_log:
        ax.set_xscale("log")
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0 - 0.05 * (y1 - y0), y1 + 0.09 * (y1 - y0))

    to_label = df_valid[df_valid["baseline"].map(
        lambda b: b in pareto_set or b in SCATTER_LABEL_ARMS)]
    annotate_selected(ax, to_label[x_col], to_label[y_col],
                      [display(b) for b in to_label["baseline"]], fontsize=6.8)

    x_dir = "lower" if x_minimize else "higher"
    y_dir = "higher" if y_maximize else "lower"
    ax.annotate(f"better: {x_dir} x, {y_dir} y", xy=(0.02, 0.02),
                xycoords="axes fraction", fontsize=8, color="green", ha="left")

    ax.set_xlabel(x_label + (f"\n{note}" if note else ""), fontsize=8.5)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    handles, labels_ = ax.get_legend_handles_labels()
    fig.legend(handles, labels_, loc="center left", bbox_to_anchor=(0.71, 0.5),
               frameon=False, fontsize=8)
    save_fig(fig, filename)
    return True


# ---------------------------------------------------------------------------
# F4: mechanism -- cache reuse buys TTFT (aligned two-panel, zero leader lines)
# ---------------------------------------------------------------------------

def plot_mechanism_reuse_vs_ttft(df: pd.DataFrame, filename: Path) -> bool:
    """Two ALIGNED horizontal panels sharing one arm axis, sorted by cached %.

    Left: token-weighted cached-prompt share bars. Right: TTFT median bars
    (log-x). Replaces the labelled scatter (mechanism_cache_reuse) whose
    crossing leader lines were unreadable: the mechanism claim now reads
    top-to-bottom -- more reuse (top rows) -> lower TTFT.
    """
    name = Path(filename).name
    data = drop_missing(df, ["cached_pct", "ttft_ms"], name).copy()
    if len(data) < 2:
        return False
    data = data.sort_values("cached_pct", ascending=False).reset_index(drop=True)
    n = len(data)
    y = np.arange(n)
    cached = data["cached_pct"].to_numpy(dtype=float)
    ttft = data["ttft_ms"].to_numpy(dtype=float)

    fig, (axl, axr) = plt.subplots(
        1, 2, sharey=True, figsize=(FULL_WIDTH_IN, max(3.6, 0.30 * n + 1.6)),
        gridspec_kw={"wspace": 0.04})
    palette = sns.color_palette("colorblind")

    axl.barh(y, cached, height=0.66, color=palette[0])
    for yi, v in zip(y, cached):
        axl.text(v + 2.0, yi, f"{v:.0f}", va="center", ha="left", fontsize=6.6)
    axl.set_xlim(0, 114)
    axl.set_yticks(y)
    axl.set_yticklabels([display(b) for b in data["baseline"]], fontsize=7.5)
    axl.invert_yaxis()  # shared axis: inverts both panels; row 0 (most reuse) on top
    axl.yaxis.grid(False)
    axl.set_xlabel("Cached prompt tokens (%, token-weighted)\n"
                   "= total cached / total prompt; missing cached = 0", fontsize=8)
    axl.set_title("KV-cache reuse", fontsize=9.5)

    axr.barh(y, ttft, height=0.66, color=palette[1])
    axr.set_xscale("log")
    for yi, v in zip(y, ttft):
        axr.text(v * 1.12, yi, f"{v:,.0f}", va="center", ha="left", fontsize=6.6)
    axr.set_xlim(left=max(1.0, float(np.nanmin(ttft)) / 3.0),
                 right=float(np.nanmax(ttft)) * 2.6)
    axr.yaxis.grid(False)
    axr.set_xlabel("TTFT median (ms, log scale; lower is better)", fontsize=8)
    axr.set_title("Time to first token", fontsize=9.5)

    fig.suptitle("Mechanism: more KV reuse (top rows) buys lower TTFT",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, filename)
    return True


# ---------------------------------------------------------------------------
# F5: CAG True paired (On vs Off)
# ---------------------------------------------------------------------------

def plot_cag_true_paired(payload: dict, filename: Path) -> bool:
    name = Path(filename).name
    comps = {c["metric"]: c for c in payload.get("comparisons") or []
             if c.get("baseline") == "cag_true_on"}
    panels = [("ttft_ms", "TTFT"), ("latency_ms", "End-to-end latency")]
    if not all(m in comps for m, _ in panels):
        print(f"  [{name}] ttft/latency comparisons missing from cagtrue stats; skipped")
        return False

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 3.9))
    palette = sns.color_palette("colorblind")
    for ax, (metric, label) in zip(axes, panels):
        c = comps[metric]
        off_v = float(c["median_reference"])
        on_v = float(c["median_baseline"])
        bars = ax.bar([0, 1], [off_v, on_v], width=0.62,
                      color=["0.65", palette[0]], edgecolor="black", linewidth=0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([display("cag_true_off"), display("cag_true_on")],
                           fontsize=8)
        for b, v in zip(bars, (off_v, on_v)):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
        st = stars(c.get("p_value_holm"))
        ax.set_title(
            f"{label}\n" + r"$\Delta$" + f"median = {float(c['median_diff']):+,.0f} ms "
            f"({float(c['pct_change']):+.1f}%)\nWilcoxon {_fmt_p(float(c['p_value_holm']))}"
            f" {st}", fontsize=8.5)
        ax.set_ylabel("ms (pooled per-example median)", fontsize=8)
        ax.set_ylim(top=max(off_v, on_v) * 1.18)
        ax.xaxis.grid(False)

    # Quality flatness annotation, computed from the same payload.
    qmetrics = {"grounding_score": "grounding", "f1_answerable": "F1 (answerable)",
                "completeness_bertscore": "BERTScore"}
    ns, sig_q = [], []
    for m, lbl in qmetrics.items():
        c = comps.get(m)
        if c is None:
            continue
        holm = c.get("p_value_holm")
        (sig_q if (holm is not None and float(holm) < 0.05) else ns).append(lbl)
    if ns and not sig_q:
        qtxt = ("Quality unchanged: " + ", ".join(ns)
                + " all n.s. (Holm p > 0.05). NLI faithfulness excluded: premise "
                  "outside validity envelope for these cells.")
    else:
        qtxt = ("Quality: significant differences in " + ", ".join(sig_q)
                + "; n.s.: " + ", ".join(ns)) if sig_q else ""
    n_pairs = int(comps["ttft_ms"].get("n_pairs") or 0)
    fig.suptitle(f"CAG True: global KV preload Off vs On (paired, N = {n_pairs})",
                 fontsize=11, fontweight="bold")
    fig.text(0.5, -0.02, _wrap(qtxt, 112), ha="center", va="top", fontsize=7.5,
             color="0.25")
    fig.tight_layout(rect=(0, 0.0, 1, 0.94))
    save_fig(fig, filename)
    return True


# ---------------------------------------------------------------------------
# F6: TTFT percentiles (replaces the ECDF, which reviewers found confusing)
# ---------------------------------------------------------------------------

def plot_ttft_percentiles(long_df: pd.DataFrame, filename: Path) -> bool:
    """Grouped horizontal bars of per-query TTFT p50/p95/p99 for the key arms."""
    name = Path(filename).name
    h = headline_rows(long_df)
    rows: list[tuple[str, float, float, float, int]] = []
    for arm in KEY_ARMS:
        vals = metric_values(h[h["cell"] == arm], "ttft_ms").dropna().to_numpy(float)
        vals = vals[vals > 0]
        if len(vals) < 5:
            print(f"  [{name}] arm {arm} has <5 TTFT values; omitted")
            continue
        rows.append((arm, float(np.percentile(vals, 50)),
                     float(np.percentile(vals, 95)),
                     float(np.percentile(vals, 99)), int(len(vals))))
    if len(rows) < 2:
        print(f"  [{name}] fewer than 2 arms with TTFT data; skipped")
        return False

    n = len(rows)
    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN, max(3.6, 0.72 * n + 1.3)))
    palette = sns.color_palette("colorblind")
    pct_colors = [palette[0], palette[1], palette[3]]
    pct_labels = ["p50", "p95", "p99"]
    bar_h = 0.24
    for gi, (arm, p50, p95, p99, _nv) in enumerate(rows):
        for k, v in enumerate((p50, p95, p99)):
            yy = gi + (k - 1) * bar_h
            ax.barh(yy, v, height=bar_h * 0.92, color=pct_colors[k],
                    label=pct_labels[k] if gi == 0 else None)
            ax.text(v * 1.10, yy, f"{v:,.0f}", va="center", ha="left",
                    fontsize=6.6)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([display(a) for a, *_ in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    p50_min = min(r[1] for r in rows)
    p99_max = max(r[3] for r in rows)
    ax.set_xlim(left=max(1.0, p50_min / 3.0), right=p99_max * 2.4)
    ax.yaxis.grid(False)
    ax.set_xlabel("TTFT per query (ms, log scale; lower is better)")
    n_min = min(r[4] for r in rows)
    ax.set_title("TTFT percentiles per query (p50 / p95 / p99), key arms")
    ax.legend(loc="lower right", fontsize=7.5,
              title=f"percentile (n >= {n_min} queries/arm)", title_fontsize=7.5)
    save_fig(fig, filename)
    return True


# ---------------------------------------------------------------------------
# F7: speculative decoding TPOT + acceptance
# ---------------------------------------------------------------------------

def plot_speculative_tpot(df: pd.DataFrame, acc: pd.DataFrame | None,
                          filename: Path) -> bool:
    name = Path(filename).name
    spec = df[df["baseline"].astype(str).str.startswith("spec_")]
    spec = drop_missing(spec, ["tpot_ms"], name)
    if spec.empty:
        print(f"  [{name}] no speculative cells; skipped")
        return False
    spec = spec.set_index("baseline").loc[order_cells(spec["baseline"])].reset_index()

    acc_map: dict[str, float] = {}
    if acc is not None and not acc.empty and "cell" in acc.columns:
        for _, r in acc.iterrows():
            try:
                acc_map[str(r["cell"])] = float(r["acceptance_rate_cumulative"])
            except (KeyError, TypeError, ValueError):
                pass

    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN, 4.4))
    x = np.arange(len(spec))
    vals = spec["tpot_ms"].to_numpy(dtype=float)
    yerr = None
    if "tpot_ms_ci_low" in spec.columns:
        lo = spec["tpot_ms_ci_low"].to_numpy(dtype=float)
        hi = spec["tpot_ms_ci_high"].to_numpy(dtype=float)
        yerr = _asym_xerr(vals, np.where(np.isnan(lo), vals, lo),
                          np.where(np.isnan(hi), vals, hi))
    bars = ax.bar(x, vals, yerr=yerr, width=0.62, capsize=3,
                  color=sns.color_palette("colorblind")[0],
                  edgecolor="black", linewidth=0.6,
                  error_kw={"elinewidth": 0.9})
    ax.set_xticks(x)
    ax.set_xticklabels([display(b) for b in spec["baseline"]], fontsize=8)

    ymax = float(np.nanmax(vals))
    for b, cell, v in zip(bars, spec["baseline"], vals):
        a = acc_map.get(cell)
        txt = f"{v:.1f} ms"
        if a is not None:
            txt += f"\nacc | draft = {a:.3f}"
        ax.text(b.get_x() + b.get_width() / 2, v + 0.06 * ymax, txt,
                ha="center", va="bottom", fontsize=7.5)

    ref = df[df["baseline"] == "no_cache"]
    rv = 0.0
    if not ref.empty and pd.notna(ref.iloc[0].get("tpot_ms")):
        rv = float(ref.iloc[0]["tpot_ms"])
        ax.axhline(rv, color="0.25", linewidth=1.1, linestyle="--", zorder=1)
        ax.text(len(spec) - 0.5, rv, f" {display('no_cache')} = {rv:.1f} ms",
                va="bottom", ha="right", fontsize=7.5, color="0.25")
    ax.set_ylim(top=max(ymax, rv) * 1.30)
    ax.set_ylabel("TPOT (ms, pooled per-example median;\nwhiskers = 95% bootstrap CI)")
    ax.set_title("Speculative decoding: time per output token vs acceptance")
    ax.set_xlabel("acc | draft = cumulative acceptance rate GIVEN the drafter "
                  "proposed (not unconditional)")
    ax.xaxis.grid(False)
    save_fig(fig, filename)
    return True


# ---------------------------------------------------------------------------
# F8: one full-width figure per quality metric (2026-07-16: 2x2 grid split up)
# ---------------------------------------------------------------------------

QUALITY_FIGURES = [
    ("grounding_mean", "quality_grounding.png", "Grounding (LettuceDetect)",
     "Mean grounding score (95% bootstrap CI; higher is better)", False),
    ("faithfulness_mean", "quality_faithfulness.png", "Faithfulness (NLI)",
     "Mean NLI faithfulness (95% bootstrap CI; higher is better)", True),
    ("f1_answerable_mean", "quality_f1_answerable.png",
     "F1 on answerable questions",
     "Mean F1 on answerable (95% bootstrap CI; higher is better)", False),
    ("bertscore_mean", "quality_completeness.png", "Completeness (BERTScore)",
     "Mean BERTScore, baseline-rescaled (95% bootstrap CI; higher is better)",
     False),
]


def plot_quality_metric(df: pd.DataFrame, col: str, title: str, xlabel: str,
                        filename: Path, nli_gated: bool = False) -> bool:
    """One quality metric, all arms, horizontal bars SORTED BY VALUE.

    Bars are pooled per-example MEANS with 95% bootstrap CIs (medians saturate).
    When nli_gated, the NLI-invalid cells are excluded entirely and named in a
    dagger footnote instead of being plotted.
    """
    name = Path(filename).name
    data = drop_missing(df, [col], name).copy()
    excluded_names: list[str] = []
    if nli_gated:
        mask = data["baseline"].isin(NLI_INVALID_CELLS)
        excluded_names = [display(b) for b in
                          order_cells(data.loc[mask, "baseline"])]
        data = data[~mask]
    if data.empty:
        print(f"  [{name}] no data; figure skipped")
        return False

    data = data.sort_values(col, ascending=False).reset_index(drop=True)
    n = len(data)
    y = np.arange(n)
    vals = data[col].to_numpy(dtype=float)
    lo = pd.to_numeric(data.get(f"{col}_ci_low", pd.Series(vals)),
                       errors="coerce").to_numpy(dtype=float)
    hi = pd.to_numeric(data.get(f"{col}_ci_high", pd.Series(vals)),
                       errors="coerce").to_numpy(dtype=float)
    lo = np.where(np.isnan(lo), vals, lo)
    hi = np.where(np.isnan(hi), vals, hi)

    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN, max(3.2, 0.28 * n + 1.5)))
    ax.barh(y, vals, xerr=_asym_xerr(vals, lo, hi), height=0.64, capsize=2.2,
            color=sns.color_palette("colorblind")[0],
            error_kw={"elinewidth": 0.9, "alpha": 0.85})
    ax.set_yticks(y)
    ax.set_yticklabels([display(b) for b in data["baseline"]], fontsize=7.5)
    ax.invert_yaxis()  # best arm on top
    ax.yaxis.grid(False)

    span = float(np.nanmax(hi)) or 1.0
    for yi, v, e in zip(y, vals, hi):
        ax.text(e + 0.012 * span, yi, f"{v:.3f}", va="center", ha="left",
                fontsize=6.8)
    ax.set_xlim(right=span * 1.14)
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.set_title(f"{title}{' †' if nli_gated else ''} by arm (sorted, "
                 "pooled per-example mean)")
    if excluded_names:
        fig.text(0.02, -0.015,
                 "† Excluded -- NLI premise exceeds validity envelope: "
                 + ", ".join(excluded_names) + ".",
                 ha="left", va="top", fontsize=6.8, color="0.35")
    save_fig(fig, filename)
    return True


# ---------------------------------------------------------------------------
# T1-T3: the tri-lemma centerpiece (serving x quality x KV memory)
# ---------------------------------------------------------------------------

def plot_trilemma_overview(df: pd.DataFrame, filename: Path) -> bool:
    """T1: grouped column chart of the three 0-100 axis scores, 9 key arms.

    Third 2026-07-16 review pass: reviewers found both the 3-panel scatter and
    the parallel-coordinates view too complex -- this is now an Excel-simple
    grouped column chart. One group per key arm (display names, rotated ~30
    degrees), three columns per group = the trilemma_scores() serving /
    quality / memory scores (identical to trilemma_table), one color per
    axis, score printed on top of every column, legend at the top. The full
    20-arm detail lives in trilemma_scores_heat.png and trilemma_table.
    """
    name = Path(filename).name
    scored = drop_missing(trilemma_scores(df), list(TRILEMMA_SCORE_COLS),
                          name).set_index("baseline")
    arms = [a for a in KEY_TRILEMMA_ARMS if a in scored.index]
    missing = [display(a) for a in KEY_TRILEMMA_ARMS if a not in scored.index]
    if missing:
        print(f"  [{name}] key arms missing from run: {', '.join(missing)}")
    if len(arms) < 2:
        return False

    palette = sns.color_palette("colorblind")
    series = [
        ("serving_score", "Serving (TTFT)", palette[0]),
        ("quality_score", "Quality (F1 answerable)", palette[2]),
        ("memory_score", "Memory (marginal KV)", palette[1]),
    ]
    x = np.arange(len(arms))
    width = 0.26
    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN, 4.6))
    for k, (col, label, color) in enumerate(series):
        vals = scored.loc[arms, col].to_numpy(dtype=float)
        bars = ax.bar(x + (k - 1) * width, vals, width * 0.94, label=label,
                      color=color, edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=6.8)
    ax.set_xticks(x)
    ax.set_xticklabels([display(a) for a in arms], rotation=30, ha="right",
                       fontsize=9)
    ax.set_ylim(0, 126)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Score (0-100; higher is better)", fontsize=9.5)
    ax.xaxis.grid(False)
    ax.legend(loc="upper center", ncol=3, frameon=False, fontsize=9,
              bbox_to_anchor=(0.5, 1.0))
    ax.set_title("Tri-lemma scores by strategy (0-100, higher is better)",
                 fontsize=11.5, pad=14)
    fig.text(0.5, 0.005,
             _wrap("Scores are min-max normalized within this run (100 = "
                   "best arm in the run on that axis). Full 20-arm detail: "
                   "trilemma_scores_heat.png and trilemma_table.", 116),
             ha="center", va="bottom", fontsize=7.5, color="0.35")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save_fig(fig, filename)
    return True


def plot_trilemma_bubble(df: pd.DataFrame, filename: Path) -> bool:
    """T2: all three tri-lemma axes in one chart.

    x = TTFT (log, serving), y = F1-answerable (quality), bubble AREA = resident
    KV tokens/query (memory), color = family. Read: a large bubble high on the
    left holds more context in KV yet serves fast at equal quality.
    """
    name = Path(filename).name
    data = _with_family(drop_missing(
        df, ["ttft_ms", "f1_answerable_mean", "resident_kv"], name))
    if len(data) < 3:
        return False

    scale = 0.085  # pts^2 per resident token: 2800 tokens -> ~240 pts^2 bubble
    fig = plt.figure(figsize=(FULL_WIDTH_IN, 5.0))
    ax = fig.add_axes((0.09, 0.19, 0.60, 0.70))
    for fam in FAMILY_SHORT_ORDER:
        sub = data[data["_family"] == fam]
        if sub.empty:
            continue
        ax.scatter(sub["ttft_ms"], sub["f1_answerable_mean"],
                   s=sub["resident_kv"].to_numpy(dtype=float) * scale,
                   color=FAMILY_COLOR[fam], alpha=0.75, edgecolors="black",
                   linewidth=0.6, zorder=3, label=fam)
    ax.set_xscale("log")
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0 - 0.06 * (y1 - y0), y1 + 0.10 * (y1 - y0))

    to_label = data[data["baseline"].isin(BUBBLE_LABEL_ARMS)]
    annotate_selected(ax, to_label["ttft_ms"], to_label["f1_answerable_mean"],
                      [display(b) for b in to_label["baseline"]], fontsize=6.8)
    ax.annotate("better: lower x, higher y", xy=(0.02, 0.02),
                xycoords="axes fraction", fontsize=7.5, color="green", ha="left")
    ax.set_xlabel("TTFT median (ms, log scale; lower is better)", fontsize=8.5)
    ax.set_ylabel("F1 on answerable (mean; higher is better)")
    ax.set_title("Tri-lemma bubble: serving (x) vs quality (y) vs "
                 "resident KV memory (bubble size)")

    fam_handles, fam_labels = ax.get_legend_handles_labels()
    fig.legend(fam_handles, fam_labels, loc="upper left",
               bbox_to_anchor=(0.71, 0.88), frameon=False, fontsize=7.5,
               title="Arm family", title_fontsize=8)
    size_refs = [200, 600, 2800]
    # Line2D handles: markersize (pts) = sqrt(scatter area in pts^2).
    size_handles = [plt.Line2D([], [], linestyle="", marker="o",
                               markerfacecolor="none", markeredgecolor="0.3",
                               markeredgewidth=0.8,
                               markersize=float(np.sqrt(v * scale)))
                    for v in size_refs]
    fig.legend(size_handles, [f"{v:,}" for v in size_refs], loc="upper left",
               bbox_to_anchor=(0.71, 0.52), frameon=False, fontsize=7.5,
               title="Resident KV tok/query", title_fontsize=8,
               labelspacing=1.7, borderpad=1.0)

    fig.text(0.5, 0.01,
             _wrap("Read: a large bubble high on the left holds more context in "
                   "KV yet serves fast at equal quality. " + KV_PROXY_NOTE, 118),
             ha="center", va="bottom", fontsize=7, color="0.3")
    save_fig(fig, filename)
    return True


# The 9 key arms on the simple tri-lemma charts (reviewer list, 2026-07-16
# third pass): the core serving anchors, the compression pair, the CAG True
# mechanism pair, and the best speculative / KV-store representatives.
KEY_TRILEMMA_ARMS = [
    "no_cache", "prefix_cache", "rag", "compressed_rag", "compressed_cag",
    "cag_true_off", "cag_true_on", "spec_qwen8b_eagle3_cag", "lmcache_rag",
]

TRILEMMA_SCORES_NOTE = (
    "Scores are the trilemma_table 0-100 axis scores, min-max normalized "
    "WITHIN THIS RUN (relative, not absolute; 100 = best arm in the run): "
    "serving = -log TTFT median, quality = mean F1 on answerable, memory = "
    "-marginal KV tokens/query. ")


def plot_trilemma_column_line(df: pd.DataFrame, filename: Path) -> bool:
    """T5: classic column+line combo -- F1 columns vs TTFT line, 9 key arms.

    Replaces the parallel-coordinates view (reviewed as too complex). EXACTLY
    two series: columns (left axis) = mean F1 on answerable as a percentage,
    in one neutral color; a single strong-contrast line with markers (right
    axis, log scale) = TTFT median in ms, value-labelled at every marker.
    The memory axis of the tri-lemma lives in trilemma_overview.png and
    trilemma_scores_heat.png.
    """
    name = Path(filename).name
    data = drop_missing(df, ["f1_answerable_mean", "ttft_ms"],
                        name).set_index("baseline")
    arms = [a for a in KEY_TRILEMMA_ARMS if a in data.index]
    missing = [display(a) for a in KEY_TRILEMMA_ARMS if a not in data.index]
    if missing:
        print(f"  [{name}] key arms missing from run: {', '.join(missing)}")
    if len(arms) < 2:
        return False

    f1_pct = data.loc[arms, "f1_answerable_mean"].to_numpy(dtype=float) * 100.0
    ttft = data.loc[arms, "ttft_ms"].to_numpy(dtype=float)
    x = np.arange(len(arms))
    line_color = "#d55e00"  # colorblind-safe vermillion, strong vs grey bars

    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN, 4.4))
    bars = ax.bar(x, f1_pct, width=0.58, color="0.72", edgecolor="black",
                  linewidth=0.5, zorder=2,
                  label="F1 on answerable (mean, %; left axis)")
    for b, v in zip(bars, f1_pct):  # value inside the bar top: line-safe
        ax.text(b.get_x() + b.get_width() / 2, v - 2.5, f"{v:.0f}",
                ha="center", va="top", fontsize=7.5, color="0.15")
    # Bars stay in the lower half so the line reads above them.
    ax.set_ylim(0, max(100.0, float(np.nanmax(f1_pct)) * 2.4))
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("F1 on answerable (mean, %)", fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels([display(a) for a in arms], rotation=30, ha="right",
                       fontsize=9)
    ax.xaxis.grid(False)

    ax2 = ax.twinx()
    line, = ax2.plot(x, ttft, color=line_color, lw=2.0, marker="o",
                     markersize=5.5, zorder=4,
                     label="TTFT median (ms; right axis, log)")
    ax2.set_yscale("log")
    ax2.set_ylim(max(10.0, float(np.nanmin(ttft)) / 4.0),
                 float(np.nanmax(ttft)) * 1.6)
    ax2.set_ylabel("TTFT median (ms, log scale; lower is better)",
                   fontsize=9.5, color=line_color)
    ax2.tick_params(axis="y", labelsize=8.5, colors=line_color)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(line_color)
    ax2.grid(False)
    for xi, v in zip(x, ttft):
        ax2.annotate(f"{v:,.0f}", (xi, v), xytext=(0, 7),
                     textcoords="offset points", ha="center", va="bottom",
                     fontsize=7.5, color=line_color, fontweight="bold")

    ax.legend([bars, line], [bars.get_label(), line.get_label()],
              loc="upper left", frameon=False, fontsize=8.5)
    ax.set_title("Serving vs quality on the key arms: F1 columns, TTFT line",
                 fontsize=11.5, pad=12)
    fig.text(0.5, 0.005,
             _wrap("Quality bars vs latency line -- the serving x quality "
                   "trade-off; the memory axis of the tri-lemma is in "
                   "trilemma_overview.png and trilemma_scores_heat.png.", 116),
             ha="center", va="bottom", fontsize=7.5, color="0.35")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save_fig(fig, filename)
    return True


def plot_trilemma_scores_heat(df: pd.DataFrame, filename: Path) -> bool:
    """T6: the trilemma_table axis scores as a color matrix (glance view).

    Rows = arms grouped by family (same grouping as the tables), columns =
    serving / quality / memory 0-100 scores, RdYlGn with the score printed in
    every cell -- nothing to decode. The grounding dagger note stays off this
    figure by design: none of the three scores involves an NLI metric.
    """
    name = Path(filename).name
    scored = drop_missing(_with_family(trilemma_scores(df)),
                          list(TRILEMMA_SCORE_COLS), name)
    if scored.empty:
        return False

    fam_rank = {f: i for i, f in enumerate(FAMILY_SHORT_ORDER)}
    cell_rank = {c: i for i, c in
                 enumerate(order_cells(list(scored["baseline"])))}
    scored = scored.assign(
        _fam_rank=scored["_family"].map(lambda f: fam_rank.get(f, len(fam_rank))),
        _cell_rank=scored["baseline"].map(cell_rank),
    ).sort_values(["_fam_rank", "_cell_rank"]).reset_index(drop=True)

    n = len(scored)
    mat = scored[list(TRILEMMA_SCORE_COLS)].astype(float).to_numpy()
    fig_h = 0.34 * n + 1.9
    fig = plt.figure(figsize=(FULL_WIDTH_IN, fig_h))
    ax = fig.add_axes((0.285, 0.50 / fig_h, 0.53, 1 - 1.55 / fig_h))
    sns.heatmap(mat, annot=True, fmt=".0f", cmap="RdYlGn", vmin=0, vmax=100,
                cbar=False, linewidths=0.6, linecolor="white", ax=ax,
                annot_kws={"fontsize": 8.5})
    ax.grid(False)
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(3) + 0.5)
    ax.set_xticklabels(["Serving", "Quality", "Memory"], fontsize=10)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels([display(b) for b in scored["baseline"]], rotation=0,
                       fontsize=8.5)
    ax.tick_params(length=0)

    # Family separators + right-edge family labels (short family names).
    fams = scored["_family"].tolist()
    bounds = [i for i in range(1, n) if fams[i] != fams[i - 1]]
    for b in bounds:
        ax.hlines(b, 0, 3, color="white", lw=3)
    for s, e in zip([0] + bounds, bounds + [n]):
        ax.text(3.18, (s + e) / 2.0, fams[s], rotation=270, ha="center",
                va="center", fontsize=8.5, color="0.35", style="italic",
                clip_on=False)

    ax.set_title("Tri-lemma axis scores by arm (0-100, min-max within run; "
                 "green = better)", fontsize=11, fontweight="bold", pad=30)
    fig.text(0.5, 0.006, _wrap(TRILEMMA_SCORES_NOTE + KV_PROXY_NOTE, 118),
             ha="center", va="bottom", fontsize=7.5, color="0.35")
    save_fig(fig, filename)
    return True


def plot_memory_resident_vs_marginal(df: pd.DataFrame, filename: Path) -> bool:
    """T3: resident vs marginal KV tokens/query, aligned two-panel bars.

    Shared arm axis sorted by resident footprint. The CAG story: CAG True (On)
    keeps the whole corpus resident in KV yet materializes almost no new KV per
    query -- the caption spells the numbers out from the data.
    """
    name = Path(filename).name
    data = drop_missing(df, ["resident_kv", "marginal_kv"], name).copy()
    if len(data) < 2:
        return False
    data = data.sort_values(["resident_kv", "marginal_kv"],
                            ascending=False).reset_index(drop=True)
    n = len(data)
    y = np.arange(n)
    res = data["resident_kv"].to_numpy(dtype=float)
    marg = data["marginal_kv"].to_numpy(dtype=float)

    fig, (axl, axr) = plt.subplots(
        1, 2, sharey=True, figsize=(FULL_WIDTH_IN, max(3.6, 0.30 * n + 1.6)),
        gridspec_kw={"wspace": 0.04})
    palette = sns.color_palette("colorblind")
    xmax = float(np.nanmax(res)) * 1.24  # same x-range so the two panels compare

    axl.barh(y, res, height=0.66, color=palette[0])
    for yi, v in zip(y, res):
        axl.text(v + 0.012 * xmax, yi, f"{v:,.0f}", va="center", ha="left",
                 fontsize=6.6)
    axl.set_xlim(0, xmax)
    axl.set_yticks(y)
    axl.set_yticklabels([display(b) for b in data["baseline"]], fontsize=7.5)
    axl.invert_yaxis()  # shared axis: largest resident footprint on top
    axl.yaxis.grid(False)
    axl.set_xlabel("Resident KV tokens/query\n(median prompt tokens)", fontsize=8)
    axl.set_title("Resident KV footprint", fontsize=9.5)

    axr.barh(y, marg, height=0.66, color=palette[1])
    for yi, v in zip(y, marg):
        axr.text(v + 0.012 * xmax, yi, f"{v:,.0f}", va="center", ha="left",
                 fontsize=6.6)
    axr.set_xlim(0, xmax)
    axr.yaxis.grid(False)
    axr.set_xlabel("Marginal KV tokens/query\n(median prompt - cached; "
                   "missing cached = 0)", fontsize=8)
    axr.set_title("Marginal (new) KV per query", fontsize=9.5)

    story = ""
    ct = data[data["baseline"] == "cag_true_on"]
    if not ct.empty:
        r0 = float(ct.iloc[0]["resident_kv"])
        m0 = float(ct.iloc[0]["marginal_kv"])
        story = (f"{display('cag_true_on')} holds ~{r0:,.0f} context tokens "
                 f"resident in KV but materializes only ~{m0:,.0f} new KV tokens "
                 "per query: the corpus is prefilled once, every query reuses it. ")
    fig.suptitle("KV memory: resident footprint vs marginal (new) KV per query",
                 fontsize=11, fontweight="bold")
    fig.text(0.5, -0.02, _wrap(story + KV_PROXY_NOTE, 120), ha="center",
             va="top", fontsize=7, color="0.3")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, filename)
    return True


# ---------------------------------------------------------------------------
# Kept overview figures: heatmap (appendix), latency breakdown
# ---------------------------------------------------------------------------

HEATMAP_COLS = [
    ("ttft_ms", "TTFT (ms)", True),
    ("latency_ms", "Latency (ms)", True),
    ("tpot_ms", "TPOT (ms)", True),
    ("qps", "QPS", False),
    ("tokens_per_sec", "Tokens/s", False),
    ("grounding_mean", "Grounding (mean)", False),
    ("faithfulness_mean", "Faithfulness (mean)", False),
    ("f1_answerable_mean", "F1 ans. (mean)", False),
    ("bertscore_mean", "BERTScore (mean)", False),
]


def plot_heatmap(df: pd.DataFrame, filename: Path) -> bool:
    name = Path(filename).name
    cols = [(c, lbl, inv) for c, lbl, inv in HEATMAP_COLS
            if c in df.columns and df[c].notna().any()]
    if len(cols) < 2:
        print(f"  [{name}] fewer than 2 metrics available; figure skipped")
        return False
    data = df.dropna(subset=[c for c, *_ in cols], how="all").copy()
    cells = order_cells(data["baseline"])
    data = data.set_index("baseline").loc[cells]

    raw = data[[c for c, *_ in cols]].astype(float).copy()
    # Blank NLI faithfulness where the premise exceeds the validity envelope.
    if "faithfulness_mean" in raw.columns:
        for c in cells:
            if c in NLI_INVALID_CELLS:
                raw.loc[c, "faithfulness_mean"] = np.nan
    norm = raw.copy()
    for c, _, invert in cols:
        lo, hi = norm[c].min(), norm[c].max()
        norm[c] = (norm[c] - lo) / (hi - lo) if hi > lo else 0.5
        if invert:  # latency-like: green = better must hold for EVERY column
            norm[c] = 1 - norm[c]

    annot = raw.map(lambda v: _fmt(v) if pd.notna(v) else "†")
    fig, ax = plt.subplots(figsize=(7.4, 0.34 * len(raw) + 2.4))
    sns.heatmap(norm, annot=annot.values, fmt="", cmap="RdYlGn", center=0.5,
                ax=ax, annot_kws={"fontsize": 6.2}, vmin=0, vmax=1,
                cbar_kws={"label": "Normalized per column (0-1); green = better"})
    ax.set_xticklabels([lbl for _, lbl, _ in cols], rotation=30, ha="right",
                       fontsize=7.5)
    ax.set_yticklabels([display(c) for c in raw.index], rotation=0, fontsize=7.5)
    ax.set_ylabel("")
    ax.set_title("Appendix: all metrics, normalized per column\n"
                 "(cell text = raw value; latency-like columns inverted; "
                 "† = NLI outside validity envelope)", fontsize=9.5)
    save_fig(fig, filename)
    return True


def plot_latency_breakdown(df: pd.DataFrame, title: str, filename: Path) -> bool:
    name = Path(filename).name
    data = drop_missing(df, ["ttft_ms", "latency_ms"], name).copy()
    if data.empty:
        return False

    data["generation_ms"] = data["latency_ms"] - data["ttft_ms"]
    negative = data[data["generation_ms"] < 0]
    if not negative.empty:
        bad = ", ".join(f"{display(b)} ({g:.1f} ms)" for b, g in
                        zip(negative["baseline"], negative["generation_ms"]))
        print(f"  [{name}] WARNING dropped arms with NEGATIVE generation time "
              f"(latency < TTFT, timing inconsistent): {bad}")
    data = data[data["generation_ms"] >= 0]
    if data.empty:
        print(f"  [{name}] no arms left after dropping negative generation; skipped")
        return False

    n = len(data)
    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN, max(3.0, 0.30 * n + 1.2)))
    y = np.arange(n)
    palette = sns.color_palette("colorblind")
    ax.barh(y, data["ttft_ms"], height=0.64, label="TTFT (prefill)",
            color=palette[2])
    ax.barh(y, data["generation_ms"], height=0.64, left=data["ttft_ms"],
            label="Generation (decode)", color=palette[0])
    ax.set_yticks(y)
    ax.set_yticklabels([display(b) for b in data["baseline"]])
    ax.invert_yaxis()
    ax.yaxis.grid(False)

    span = float(data["latency_ms"].max()) or 1.0
    for yi, total in zip(y, data["latency_ms"]):
        ax.text(total + 0.015 * span, yi, f"{total:,.0f}",
                va="center", ha="left", fontsize=7)
    ax.set_xlim(right=span * 1.16)
    ax.set_xlabel("Time (ms) -- pooled per-example median; bar total = latency")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=7.5)
    save_fig(fig, filename)
    return True


# ---------------------------------------------------------------------------
# Explanations + main
# ---------------------------------------------------------------------------

def write_plot_explanations(plots_dir: Path, plots: list[tuple[str, str]]) -> None:
    lines = [
        "Plot explanations (publication set, 2026-07-16 overhaul)",
        "=" * 56,
        "",
        "Every figure/table uses the canonical display names (Title Case, no",
        "underscores) and the Arial font chain. One paragraph per artifact: what it",
        "shows and how to read it.",
        "",
    ]
    for fname, desc in plots:
        lines.append(f"- {fname}:")
        lines.append(f"  {desc}")
        lines.append("")
    lines += [f"Caveat: {CAVEAT_LINE}", "", NLI_DAGGER_NOTE, ""]
    (plots_dir / "plots_explained.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the publication figure set from the canonical loader.")
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

    df, long_df = build_summary(results_dir)  # SystemExit when the root has no cells
    if df.empty:
        raise SystemExit("No result rows found.")
    print(f"Loaded {len(df)} baselines from {results_dir}")

    # Stale figures deleted by the overhaul must not survive a regeneration.
    for stale in DELETED_FIGURES:
        stale_path = plots_dir / stale
        if stale_path.exists():
            stale_path.unlink()
            print(f"  removed stale artifact {stale} (deleted by 2026-07-16 overhaul)")

    stats_payload = load_stats(
        results_dir / "stats" / "all_results" / "phase2_stats.json", "phase2_stats")
    cagtrue_payload = load_stats(
        results_dir / "stats" / "all_results_cagtrue" / "cagtrue_stats.json",
        "cagtrue_stats")
    acc_path = results_dir / "stats" / "all_results" / "spec_acceptance_summary.csv"
    acc_df = pd.read_csv(acc_path) if acc_path.is_file() else None
    if acc_df is None:
        print(f"  [spec acceptance] {acc_path} not found; acceptance omitted")

    def _n(col: str) -> int:
        c = f"{col}_n"
        return int(df[c].max()) if c in df.columns and df[c].notna().any() else 0

    pq_note = "pooled per-example median; whiskers = 95% bootstrap CI"
    explained: list[tuple[str, str]] = []

    # ---- F1: main results table (md + tex) ----
    print("\nWriting tables...")
    for fname in write_main_results_table(df, stats_payload, plots_dir):
        print(f"  wrote {fname}")
        explained.append((fname,
            "Main results table grouped by arm family: serving medians (TTFT, "
            "latency, TPOT, ms) with Holm-corrected Wilcoxon stars vs No Cache, "
            "token-weighted cached-prompt %, grounding mean, F1 on answerable and "
            "abstention %. The .tex version is booktabs, ready for the dissertation."))
    for fname in write_speculative_table(df, acc_df, stats_payload, plots_dir):
        print(f"  wrote {fname}")
        explained.append((fname,
            "Companion table for the speculative arms: TPOT and latency medians, "
            "paired Delta TPOT vs No Cache with stars, and the acceptance rate "
            "conditional on the drafter proposing (acc | draft proposed)."))
    for fname in write_trilemma_table(df, plots_dir):
        print(f"  wrote {fname}")
        explained.append((fname,
            "Tri-lemma table (T4), one row per arm grouped by family: TTFT and "
            "latency medians (ms), F1-answerable and grounding means, resident "
            "and marginal KV tokens/query, cached-prompt %, plus three "
            "normalized 0-100 axis scores (serving = min-max of -log TTFT; "
            "quality = min-max of F1-answerable; memory = min-max of -marginal "
            "KV). Scores are within-run relative, not absolute. "
            + TRILEMMA_AXES_NOTE + " The .tex version is booktabs, ready for "
            "the dissertation."))

    # ---- Kept by-baseline horizontal bars ----
    print("\nGenerating serving figures...")
    by_baseline = [
        ("ttft_ms", "TTFT by arm", "Time to first token (ms; lower is better)",
         "ttft_by_baseline.png",
         f"{pq_note} (n={_n('ttft_ms')} queries: 3 trials x 100 questions)"),
        ("latency_ms", "End-to-end latency by arm", "Latency (ms; lower is better)",
         "latency_by_baseline.png",
         f"{pq_note} (n={_n('latency_ms')} queries: 3 trials x 100 questions)"),
    ]
    for col, title, xlabel, fname, note in by_baseline:
        if plot_metric_by_baseline(df, col, title, xlabel, plots_dir / fname,
                                   note=note):
            explained.append((fname,
                f"{title}: horizontal bars of the {pq_note}. Read shorter bars as "
                "faster; whiskers give the bootstrap uncertainty of the median."))

    if plot_latency_breakdown(df, "Latency breakdown: prefill (TTFT) vs decode",
                              plots_dir / "latency_breakdown_stacked.png"):
        explained.append(("latency_breakdown_stacked.png",
            "Stacked horizontal bars decomposing median end-to-end latency into "
            "TTFT (prefill) and generation (decode) time, the same decomposition "
            "the CAG paper uses for its timing results. Shows WHERE each arm "
            "spends its time: cache arms cut the prefill segment, speculative "
            "arms cut decode."))

    # ---- F2: forest panels ----
    print("\nGenerating forest figures...")
    if stats_payload:
        serving_panels = [
            {"title": "TTFT", "xlabel": r"$\Delta$ median (ms)",
             "entries": _stats_entries(stats_payload, "ttft_ms")},
            {"title": "Latency", "xlabel": r"$\Delta$ median (ms)",
             "entries": _stats_entries(stats_payload, "latency_ms")},
            {"title": "TPOT", "xlabel": r"$\Delta$ median (ms)",
             "entries": _stats_entries(stats_payload, "tpot_ms")},
        ]
        if plot_forest_panels(
                serving_panels, plots_dir / "forest_serving.png",
                "Serving deltas vs No Cache (paired Wilcoxon; negative = faster)",
                "Median of per-example paired differences; whiskers = 95% CI; "
                "filled = Holm-significant (p < 0.05). N = 300 pairs per cell."):
            explained.append(("forest_serving.png",
                "Forest plot of the paired median serving differences (TTFT, "
                "latency, TPOT) of every arm vs No Cache, rendered straight from "
                "phase2_stats.json so it cannot disagree with the stats tables. "
                "Points left of zero are faster than No Cache; filled markers are "
                "Holm-significant, open grey markers are not."))

        ground_deltas = paired_mean_deltas(long_df, "grounding_score")
        ground_sig = {b: e[3] for b, e in _stats_entries(
            stats_payload, "grounding_score").items()}
        ground_entries = {
            cell: (v[0], v[1], v[2], bool(ground_sig.get(cell, False)))
            for cell, v in ground_deltas.items()}
        quality_panels = [
            {"title": "Grounding (mean)", "xlabel": r"$\Delta$ mean",
             "entries": ground_entries},
            {"title": "F1 answerable", "xlabel": r"$\Delta$ median",
             "entries": _stats_entries(stats_payload, "f1_answerable")},
            {"title": "BERTScore", "xlabel": r"$\Delta$ median",
             "entries": _stats_entries(stats_payload, "completeness_bertscore")},
        ]
        if plot_forest_panels(
                quality_panels, plots_dir / "forest_quality.png",
                "Quality deltas vs No Cache (positive = better)",
                "Grounding: mean paired delta + bootstrap CI (medians saturate at "
                "1.0); F1/BERTScore: median paired delta + 95% CI. Significance: "
                "Holm-corrected Wilcoxon."):
            explained.append(("forest_quality.png",
                "Forest plot of quality differences vs No Cache: grounding "
                "(LettuceDetect, mean paired delta because grounding medians "
                "saturate at 1.0), F1 on answerable, and BERTScore. Points near "
                "zero with open markers mean the arm's quality is statistically "
                "indistinguishable from No Cache."))

    # ---- F3 + kept Pareto ----
    print("\nGenerating Pareto figures...")
    pareto_specs = [
        ("ttft_ms", "f1_answerable_mean", True, True,
         "TTFT median (ms, log scale; lower is better)",
         "F1 on answerable (mean; higher is better)",
         "TTFT vs answer quality: the serving-quality tradeoff",
         "pareto_ttft_vs_f1answerable.png",
         "One point per arm, color+marker = arm family; crimson rings mark the "
         "Pareto-optimal arms.",
         "Pareto frontier of the thesis tradeoff: how much answer quality (F1 on "
         "answerable questions) each arm delivers at what time-to-first-token. "
         "Points are color+marker coded by arm family (legend beside the axes); "
         "crimson rings and the dashed staircase mark the Pareto-optimal arms -- "
         "no arm is simultaneously faster and better than them. Only the "
         "Pareto-optimal arms and the anchors No Cache, CAG True (Off/On) and "
         "Compressed RAG carry direct labels; all other points stay unlabeled."),
        ("latency_ms", "bertscore_mean", True, True,
         "Latency median (ms; lower is better)",
         "BERTScore (mean, baseline-rescaled; higher is better)",
         "End-to-end latency vs completeness (BERTScore)",
         "pareto_latency_vs_quality.png",
         "One point per arm, color+marker = arm family; crimson rings mark the "
         "Pareto-optimal arms.",
         "Secondary Pareto view: end-to-end latency against BERTScore "
         "completeness, with the same family color+marker encoding and selective "
         "labelling as the TTFT-vs-F1 frontier. Complements it by using the full "
         "request time and a soft semantic-similarity quality signal."),
    ]
    pareto_rows = []
    for (xc, yc, xmin, ymax, xl, yl, title, fname, note, para) in pareto_specs:
        if plot_pareto_frontier(df, xc, yc, title, plots_dir / fname,
                                x_label=xl, y_label=yl, x_minimize=xmin,
                                y_maximize=ymax, note=note,
                                x_log=(xc == "ttft_ms")):
            explained.append((fname, para))
            for _, row in compute_pareto_frontier(df, xc, yc, xmin, ymax).iterrows():
                pareto_rows.append({
                    "tradeoff": f"{xc}_vs_{yc}",
                    "baseline": row["baseline"],
                    "display_name": display(row["baseline"]),
                    xc: row[xc], yc: row[yc],
                })
    if pareto_rows:
        pd.DataFrame(pareto_rows).to_csv(
            plots_dir / "pareto_optimal_baselines.csv", index=False)
        print("  wrote pareto_optimal_baselines.csv")

    # ---- F4: mechanism ----
    print("\nGenerating mechanism figures...")
    if plot_mechanism_reuse_vs_ttft(df, plots_dir / "mechanism_reuse_vs_ttft.png"):
        explained.append(("mechanism_reuse_vs_ttft.png",
            "Mechanism figure, two ALIGNED horizontal panels sharing one arm "
            "axis sorted by cache reuse: left = token-weighted cached-prompt "
            "share (total cached / total prompt tokens, missing cached = 0), "
            "right = TTFT median (log scale). Read top-to-bottom: arms at the "
            "top reuse the most KV cache and post the lowest TTFT; the 0%-reuse "
            "arms at the bottom (No Cache, RAG, Redis Cold, CAG True Off) anchor "
            "the no-reuse regime. Replaces the labelled scatter whose leader "
            "lines crossed."))

    # ---- F5: CAG True paired ----
    if cagtrue_payload and plot_cag_true_paired(
            cagtrue_payload, plots_dir / "cag_true_paired.png"):
        explained.append(("cag_true_paired.png",
            "Paired mechanism figure for CAG True: median TTFT and latency with "
            "the global KV preload off vs on, annotated with the paired Wilcoxon "
            "p-values and percent changes from cagtrue_stats.json, plus the "
            "quality-unchanged annotation. This is the cleanest single-mechanism "
            "evidence in the run: same arm, same prompts, only the preload "
            "toggled."))

    # ---- F6: TTFT percentiles ----
    if plot_ttft_percentiles(long_df, plots_dir / "ttft_percentiles.png"):
        explained.append(("ttft_percentiles.png",
            "Per-query TTFT tail behaviour for five key arms as grouped "
            "horizontal bars: p50, p95 and p99 of the per-query TTFT "
            "distribution, values annotated, log x-axis. Shorter bars are "
            "faster; comparing p50 to p99 within an arm shows how heavy its "
            "tail is, and the log axis makes the order-of-magnitude gap between "
            "preloaded-KV arms and full-prefill arms visible. Replaces the ECDF "
            "figure, which reviewers found hard to read."))

    # ---- F7: speculative ----
    if plot_speculative_tpot(df, acc_df, plots_dir / "speculative_tpot.png"):
        explained.append(("speculative_tpot.png",
            "TPOT medians of the four speculative-decoding arms with the No Cache "
            "median as a dashed reference line. Bars are annotated with the "
            "acceptance rate conditional on the drafter proposing (acc | draft); "
            "EAGLE-3 proposes every step at ~0.26-0.31 acceptance while the "
            "n-gram drafter proposes rarely but accepts at ~0.66-0.68, which is "
            "why acceptance alone does not rank the arms."))

    # ---- F8: one figure per quality metric ----
    print("\nGenerating quality figures...")
    quality_desc = {
        "quality_grounding.png":
            "Grounding (LettuceDetect) by arm: full-width horizontal bars of the "
            "pooled per-example MEAN with 95% bootstrap CIs (medians saturate at "
            "1.0), arms sorted by value with the best on top. Near-identical "
            "bars across arms are the point: caching does not move grounding.",
        "quality_faithfulness.png":
            "NLI faithfulness by arm, same construction (mean + 95% bootstrap "
            "CI, sorted). CAG True (Off/On) and Prefix Cache (Multi-turn) are "
            "EXCLUDED and named in the dagger footnote: their NLI premise "
            "exceeds the validity envelope (the model conditioned on more "
            "context than the per-query premise), so their scores are not "
            "comparable.",
        "quality_f1_answerable.png":
            "F1 on answerable questions by arm (mean + 95% bootstrap CI, "
            "sorted). This is the QUALITY axis of the tri-lemma -- the one "
            "quality metric that is valid for every arm -- so this figure is "
            "the quality reference for the Pareto and tri-lemma charts.",
        "quality_completeness.png":
            "Completeness (BERTScore, baseline-rescaled) by arm (mean + 95% "
            "bootstrap CI, sorted). Soft semantic-similarity signal that "
            "complements the strict token-level F1.",
    }
    for col, fname, title, xlabel, nli_gated in QUALITY_FIGURES:
        if plot_quality_metric(df, col, title, xlabel, plots_dir / fname,
                               nli_gated=nli_gated):
            explained.append((fname, quality_desc[fname]))

    # ---- T1-T3: tri-lemma centerpiece ----
    print("\nGenerating tri-lemma figures...")
    if plot_trilemma_overview(df, plots_dir / "trilemma_overview.png"):
        explained.append(("trilemma_overview.png",
            "Tri-lemma overview as an Excel-simple grouped column chart: nine "
            "key arms on the x-axis (No Cache, Prefix Cache, RAG, Compressed "
            "RAG, Compressed CAG (FP8), CAG True (Off), CAG True (On), "
            "EAGLE-3 + CAG, LMCache + RAG), three columns per arm = the 0-100 "
            "Serving / Quality / Memory scores from the trilemma_table "
            "scoring rule (min-max within this run; 100 = best arm in the "
            "run on that axis), one color per axis, score printed on top of "
            "every column. A good strategy shows three tall columns. Full "
            "20-arm detail: trilemma_scores_heat.png and trilemma_table."))
    if plot_trilemma_bubble(df, plots_dir / "trilemma_bubble.png"):
        explained.append(("trilemma_bubble.png",
            "All three tri-lemma axes in a single chart: x = TTFT median (log, "
            "serving), y = F1-answerable mean (quality), bubble size = resident "
            "KV tokens/query (memory; reference bubbles at 200/600/2,800), "
            "color = arm family. Read: a large bubble high on the left holds "
            "more context in KV yet serves fast at equal quality -- that is CAG "
            "True (On). Only No Cache, Compressed RAG, CAG True (Off/On) and "
            "EAGLE-3 + CAG are labelled. " + KV_PROXY_NOTE))
    if plot_trilemma_column_line(df, plots_dir / "trilemma_column_line.png"):
        explained.append(("trilemma_column_line.png",
            "Classic column+line combo on the same nine key arms, exactly "
            "two series: neutral grey columns (left axis) = mean F1 on "
            "answerable as a percentage, value printed inside each column "
            "top; a single contrasting line with markers (right axis, log "
            "scale) = TTFT median in ms, value-labelled at every marker. "
            "Quality bars vs latency line -- the serving x quality trade-off "
            "at a glance; the memory axis of the tri-lemma is covered by "
            "trilemma_overview.png and trilemma_scores_heat.png."))
    if plot_trilemma_scores_heat(df, plots_dir / "trilemma_scores_heat.png"):
        explained.append(("trilemma_scores_heat.png",
            "The trilemma_table axis scores rendered as a color matrix, "
            "readable at a glance: rows = arms grouped by family, columns = "
            "Serving / Quality / Memory scores (0-100, min-max within this "
            "run), RdYlGn colormap with the score printed in every cell "
            "(green = better, 100 = best arm in the run on that axis). No "
            "points or encodings to decode; use it to rank arms per axis and "
            "to spot all-round performers as all-green rows. "
            + KV_PROXY_NOTE))
    if plot_memory_resident_vs_marginal(
            df, plots_dir / "memory_resident_vs_marginal.png"):
        explained.append(("memory_resident_vs_marginal.png",
            "The memory axis unpacked: aligned two-panel horizontal bars on one "
            "arm axis sorted by resident footprint -- left = resident KV "
            "tokens/query (median prompt tokens), right = marginal KV "
            "tokens/query (median prompt - cached tokens, missing cached = 0), "
            "both panels on the same x-range. The CAG story is the top row: "
            "CAG True (On) keeps the whole corpus resident yet materializes "
            "almost no new KV per query, while CAG True (Off) pays the full "
            "corpus prefill on every query. " + KV_PROXY_NOTE))

    # ---- Appendix heatmap ----
    if plot_heatmap(df, plots_dir / "heatmap_all_metrics.png"):
        explained.append(("heatmap_all_metrics.png",
            "Appendix overview heatmap: every arm x every headline metric, "
            "normalized per column (green = better within the column; "
            "latency-like columns inverted). Cell text shows the raw value; use "
            "it to look values up, not to compare colors across columns. NLI "
            "faithfulness is blanked (†) where invalid."))

    write_plot_explanations(plots_dir, explained)
    print("  wrote plots_explained.txt")

    summary_path = plots_dir / "latest_metrics_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"\nSaved plots to {plots_dir}")
    print(f"Saved summary CSV to {summary_path}")


if __name__ == "__main__":
    main()
