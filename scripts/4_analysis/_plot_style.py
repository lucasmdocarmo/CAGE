#!/usr/bin/env python3
"""Shared figure style, baseline ordering and abbreviations for CAGE analysis plots.

Single source of truth (2026-07-15 plot overhaul):
- apply_style():        one rcParams block -> every figure gets identical dpi/palette/grid.
- BASELINE_ORDER:       canonical arm order (core arms first, then context-scale control,
                        compression, speculative, then planned/future arms). Unknown names
                        sort after all known ones, alphabetically.
- BASELINE_ABBREV / abbrev(): short labels for space-constrained figures. The mapping is
  written next to the figures (baseline_abbreviations.txt) so no figure needs a legend
  explaining its own tick labels.
- save_fig():           uniform save (dpi via rcParams, bbox_inches="tight", close).
- annotate_points():    deterministic offset cycle so point labels never all pile on the
                        same corner (the old fixed +5,+5 offset stacked labels).
- drop_missing():       replaces silent .dropna() -- prints WHICH baselines were dropped
                        from WHICH figure so missing data is visible in the run log.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Canonical arm order: core arms first, then rag_full (context-scale control),
# compression 2, speculative 4, then planned/future arms not yet in any run.
BASELINE_ORDER: List[str] = [
    # core arms
    "no_cache",
    "prefix_cache",
    "cag_full",
    "rag",
    "redis_retrieval_cache_cold",
    "hybrid_retrieval_cache_cold",
    "hybrid_retrieval_cache_warm",
    # context-scale control
    "rag_full",
    # compression axis
    "compressed_rag",
    "compressed_cag",
    # speculative decoding matrix
    "spec_qwen8b_ngram_cag",
    "spec_qwen8b_ngram_rag",
    "spec_qwen8b_eagle3_cag",
    "spec_qwen8b_eagle3_rag",
    # planned / future arms
    "cag_true_off",
    "cag_true_on",
    "prefix_cache_grouped",
    "prefix_cache_multiturn",
    "prefix_cache_repeat",
    "lmcache_rag",
    "cag_reference",
]

# The core comparison arms (used e.g. by the radar chart, where >7 overlapping
# profiles are unreadable).
CORE_ARMS: List[str] = BASELINE_ORDER[:7]

BASELINE_ABBREV: Dict[str, str] = {
    "no_cache": "no-cache",
    "prefix_cache": "prefix",
    "cag_full": "cag-full",
    "rag": "rag",
    "rag_full": "rag-full",
    "redis_retrieval_cache_cold": "redis-cold",
    "hybrid_retrieval_cache_cold": "hyb-cold",
    "hybrid_retrieval_cache_warm": "hyb-warm",
    "compressed_rag": "comp-rag",
    "compressed_cag": "comp-cag(fp8)",
    "spec_qwen8b_ngram_cag": "spec-ng-cag",
    "spec_qwen8b_ngram_rag": "spec-ng-rag",
    "spec_qwen8b_eagle3_cag": "spec-e3-cag",
    "spec_qwen8b_eagle3_rag": "spec-e3-rag",
    "cag_true_off": "cag-off",
    "cag_true_on": "cag-on",
    "prefix_cache_grouped": "prefix-grp",
    "prefix_cache_multiturn": "prefix-mt",
    "prefix_cache_repeat": "prefix-rep",
    "lmcache_rag": "lmcache-rag",
    "cag_reference": "cag-ref",
    # MiMo within-model arms (labels tagged 'mimo' by the sweep)
    "no_cache_mimo7b": "no-cache-mimo",
    "spec_mimo7b_mtp_cag": "spec-mtp-cag",
    "spec_mimo7b_mtp_rag": "spec-mtp-rag",
}

# Deterministic label-offset cycle (points): right-up, right-down, left-up, left-down.
_OFFSET_CYCLE = [(6, 6), (6, -11), (-6, 6), (-6, -11)]


def abbrev(name: str, max_len: int = 14) -> str:
    """Short label for a baseline; unknown names are truncated with a trailing '~'."""
    if name in BASELINE_ABBREV:
        return BASELINE_ABBREV[name]
    if len(name) <= max_len:
        return name
    return name[: max_len - 1] + "~"


def order_cells(seq: Sequence[str]) -> List[str]:
    """Sort cell names by BASELINE_ORDER; unknown names after, alphabetically."""
    rank = {name: i for i, name in enumerate(BASELINE_ORDER)}
    return sorted(seq, key=lambda n: (rank.get(n, len(BASELINE_ORDER)), n))


def apply_style() -> None:
    """One rcParams block for every CAGE figure (call once, before plotting)."""
    sns.set_palette("colorblind")
    plt.rcParams.update({
        "savefig.dpi": 300,
        "figure.dpi": 100,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.axisbelow": True,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.framealpha": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_fig(fig, path: Path) -> None:
    """Uniform save: rcParams dpi (300), tight bbox, always closed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.name}")


def annotate_points(ax, xs, ys, labels, fontsize: int = 8, **text_kw) -> None:
    """Label scatter points without pile-ups, deterministically.

    Strategy: start from a fixed offset cycle, flip the offset inward for points near
    an axes edge, then greedily push later labels further out (in whole label-height
    steps) until their approximate bounding boxes no longer overlap earlier ones.
    """
    pts = [(float(x), float(y), str(label))
           for x, y, label in zip(xs, ys, labels)
           if not (pd.isna(x) or pd.isna(y))]
    if not pts:
        return

    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    xspan = (x1 - x0) or 1.0
    yspan = (y1 - y0) or 1.0

    anns = []
    for i, (x, y, label) in enumerate(pts):
        dx, dy = _OFFSET_CYCLE[i % len(_OFFSET_CYCLE)]
        # Edge heuristic: keep labels inside the axes.
        if x > x1 - 0.12 * xspan:
            dx = -abs(dx)
        elif x < x0 + 0.12 * xspan:
            dx = abs(dx)
        if y > y1 - 0.06 * yspan:
            dy = -abs(dy) - 5
        elif y < y0 + 0.06 * yspan:
            dy = abs(dy)
        anns.append([x, y, label, dx, dy])

    def _bbox(x, y, label, dx, dy):
        # Approximate label bbox in display points.
        px, py = ax.transData.transform((x, y)) * 72.0 / ax.figure.dpi
        w = 0.60 * fontsize * len(label)
        h = fontsize + 3.0
        bx0 = px + dx if dx >= 0 else px + dx - w
        by0 = py + dy if dy >= 0 else py + dy - h
        return (bx0, by0, bx0 + w, by0 + h)

    step = fontsize + 4.0
    for i in range(1, len(anns)):
        for _ in range(30):  # bounded: deterministic, always terminates
            bi = _bbox(*anns[i])
            if not any(
                bi[0] < bj[2] and bi[2] > bj[0] and bi[1] < bj[3] and bi[3] > bj[1]
                for bj in (_bbox(*anns[j]) for j in range(i))
            ):
                break
            anns[i][4] += step if anns[i][4] >= 0 else -step

    for x, y, label, dx, dy in anns:
        ax.annotate(
            label, (x, y), xytext=(dx, dy), textcoords="offset points",
            fontsize=fontsize,
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
            **text_kw,
        )


def drop_missing(df: pd.DataFrame, cols: Sequence[str], fig_name: str,
                 how: str = "any") -> pd.DataFrame:
    """dropna with a printed audit trail: WHICH baselines left WHICH figure and why."""
    present = [c for c in cols if c in df.columns]
    absent = [c for c in cols if c not in df.columns]
    if absent:
        print(f"  [{fig_name}] columns absent entirely: {', '.join(absent)}")
    if not present:
        return df.iloc[0:0]
    out = df.dropna(subset=present, how=how)
    dropped = df.loc[~df.index.isin(out.index)]
    if not dropped.empty and "baseline" in dropped.columns:
        names = ", ".join(dropped["baseline"].astype(str))
        print(f"  [{fig_name}] dropped (missing {'/'.join(present)}): {names}")
    return out


def write_abbrev_legend(plots_dir: Path) -> None:
    """Write baseline_abbreviations.txt next to the figures."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    lines = ["Baseline abbreviations used in figures", "=" * 39, ""]
    for name in BASELINE_ORDER:
        lines.append(f"{BASELINE_ABBREV.get(name, name):<16} = {name}")
    extras = sorted(set(BASELINE_ABBREV) - set(BASELINE_ORDER))
    for name in extras:
        lines.append(f"{BASELINE_ABBREV[name]:<16} = {name}")
    lines.append("")
    lines.append("Unlisted arms keep their full name (truncated with '~' when long).")
    (plots_dir / "baseline_abbreviations.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
