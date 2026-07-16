#!/usr/bin/env python3
"""Shared figure style, canonical naming, ordering and helpers for CAGE analysis plots.

Publication conventions (2026-07-16 overhaul, supersedes the 2026-07-15 abbrev scheme):
- DISPLAY_NAME / display(): THE canonical cell-id -> human-readable label mapping.
  No figure, legend or table may show a raw snake_case cell id; every consumer maps
  through display(). Unknown ids degrade to Title Case words (never underscores).
- family_of() / TREE_FAMILY / FAMILY_ORDER: arm families used to group the main
  results table and the forest panels.
- NLI_INVALID_CELLS + NLI_DAGGER_NOTE: cells whose NLI faithfulness premise exceeds
  the validity envelope (multi-turn / global-KV context). Faithfulness figures and
  tables MUST exclude these cells or dagger them with the note.
- apply_style(): one rcParams block -> Arial with a safe fallback chain (Helvetica,
  Liberation Sans, DejaVu Sans) so figures render identically on macOS and bare
  Linux VMs, 300 dpi, light grid, no top/right spines.
- stars(): significance stars from a (Holm-corrected) p-value.
- save_fig(), annotate_points() (now with leader lines for displaced labels),
  drop_missing(), order_cells() as before.
- FULL_WIDTH_IN: maximum figure width (inches) for thesis pages; no figure may be
  wider than this at 300 dpi.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Maximum width (inches) of any figure destined for the dissertation (ABNT A4 text
# block ~ 15-16 cm; 7.2 in leaves margin at 300 dpi).
FULL_WIDTH_IN: float = 7.2

# Canonical arm order: core arms first, then rag_full (context-scale control),
# compression, speculative, then envelope / kv-store / reference arms.
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
    # envelope / kv-store / reference arms
    "cag_true_off",
    "cag_true_on",
    "prefix_cache_grouped",
    "prefix_cache_multiturn",
    "prefix_cache_repeat",
    "lmcache_rag",
    "cag_reference",
]

# The core comparison arms (kept for consumers that need a readable subset).
CORE_ARMS: List[str] = BASELINE_ORDER[:7]

# ---------------------------------------------------------------------------
# Canonical display names: the ONE mapping used by every figure, legend, table.
# ---------------------------------------------------------------------------

DISPLAY_NAME: Dict[str, str] = {
    # core serving arms
    "no_cache": "No Cache",
    "prefix_cache": "Prefix Cache",
    "rag": "RAG",
    "redis_retrieval_cache_cold": "Redis Cache (Cold)",
    "hybrid_retrieval_cache_cold": "Hybrid Cache (Cold)",
    "hybrid_retrieval_cache_warm": "Hybrid Cache (Warm)",
    # compression & context scale
    "cag_full": "CAG Full",
    "rag_full": "RAG Full",
    "compressed_rag": "Compressed RAG",
    "compressed_cag": "Compressed CAG (FP8)",
    # memory envelope
    "cag_true_off": "CAG True (Off)",
    "cag_true_on": "CAG True (On)",
    "prefix_cache_grouped": "Prefix Cache (Grouped)",
    "prefix_cache_multiturn": "Prefix Cache (Multi-turn)",
    "prefix_cache_repeat": "Prefix Cache (Repeat)",
    # KV store
    "lmcache_rag": "LMCache + RAG",
    # speculative decoding
    "spec_qwen8b_eagle3_cag": "EAGLE-3 + CAG",
    "spec_qwen8b_eagle3_rag": "EAGLE-3 + RAG",
    "spec_qwen8b_ngram_cag": "N-gram + CAG",
    "spec_qwen8b_ngram_rag": "N-gram + RAG",
    # reference / future arms
    "cag_reference": "CAG Reference",
    "no_cache_mimo7b": "No Cache (MiMo-7B)",
    "spec_mimo7b_mtp_cag": "MTP + CAG (MiMo-7B)",
    "spec_mimo7b_mtp_rag": "MTP + RAG (MiMo-7B)",
}


def display(name: str) -> str:
    """Human-readable label for a cell id. NEVER returns underscores."""
    if name in DISPLAY_NAME:
        return DISPLAY_NAME[name]
    return str(name).replace("_", " ").strip().title()


# ---------------------------------------------------------------------------
# Arm families (table grouping + forest panel grouping)
# ---------------------------------------------------------------------------

TREE_FAMILY: Dict[str, str] = {
    "baselines": "Core serving",
    "compression": "Compression & context scale",
    "envelope": "Memory envelope",
    "kv_store": "KV store",
    "speculative": "Speculative decoding",
    "reference": "Reference",
}

FAMILY_ORDER: List[str] = [
    "Core serving",
    "Compression & context scale",
    "Memory envelope",
    "KV store",
    "Speculative decoding",
    "Reference",
    "Other",
]

# Fallback for flat layouts where the tree column is empty.
CELL_FAMILY: Dict[str, str] = {
    "no_cache": "Core serving",
    "prefix_cache": "Core serving",
    "rag": "Core serving",
    "redis_retrieval_cache_cold": "Core serving",
    "hybrid_retrieval_cache_cold": "Core serving",
    "hybrid_retrieval_cache_warm": "Core serving",
    "cag_full": "Compression & context scale",
    "rag_full": "Compression & context scale",
    "compressed_rag": "Compression & context scale",
    "compressed_cag": "Compression & context scale",
    "cag_true_off": "Memory envelope",
    "cag_true_on": "Memory envelope",
    "prefix_cache_grouped": "Memory envelope",
    "prefix_cache_multiturn": "Memory envelope",
    "prefix_cache_repeat": "Memory envelope",
    "lmcache_rag": "KV store",
    "spec_qwen8b_eagle3_cag": "Speculative decoding",
    "spec_qwen8b_eagle3_rag": "Speculative decoding",
    "spec_qwen8b_ngram_cag": "Speculative decoding",
    "spec_qwen8b_ngram_rag": "Speculative decoding",
    "cag_reference": "Reference",
}


def family_of(cell: str, tree: str = "") -> str:
    """Family label for a cell (tree column preferred, per-cell fallback)."""
    if tree and tree in TREE_FAMILY:
        return TREE_FAMILY[tree]
    return CELL_FAMILY.get(cell, "Other")


# Short family labels + fixed color/marker encodings for the family-coded scatter
# figures (Pareto + tri-lemma set). Colors are seaborn "colorblind" palette hexes so
# the encoding is stable even if the seaborn default palette changes.
SHORT_FAMILY: Dict[str, str] = {
    "Core serving": "Core",
    "Compression & context scale": "Compression",
    "Memory envelope": "Envelope",
    "KV store": "KV Store",
    "Speculative decoding": "Speculative",
    "Reference": "Reference",
    "Other": "Other",
}

FAMILY_SHORT_ORDER: List[str] = [
    "Core", "Compression", "Envelope", "KV Store", "Speculative",
    "Reference", "Other",
]

FAMILY_COLOR: Dict[str, str] = {
    "Core": "#0173b2",         # blue
    "Compression": "#de8f05",  # orange
    "Envelope": "#029e73",     # green
    "KV Store": "#d55e00",     # vermillion
    "Speculative": "#cc78bc",  # purple
    "Reference": "#949494",    # grey
    "Other": "#ca9161",        # tan
}

FAMILY_MARKER: Dict[str, str] = {
    "Core": "o",
    "Compression": "s",
    "Envelope": "^",
    "KV Store": "D",
    "Speculative": "v",
    "Reference": "P",
    "Other": "X",
}


def family_short(cell: str, tree: str = "") -> str:
    """Short family label (legend-sized) for a cell."""
    return SHORT_FAMILY.get(family_of(cell, tree), "Other")


# ---------------------------------------------------------------------------
# Validity envelope (2026-07-16 audit)
# ---------------------------------------------------------------------------

# NLI faithfulness is INVALID for these cells: the NLI premise (per-query context)
# does not match what the model actually conditioned on (multi-turn history /
# globally preloaded KV corpus), so entailment scores are not comparable.
NLI_INVALID_CELLS = frozenset({
    "cag_true_off",
    "cag_true_on",
    "prefix_cache_multiturn",
})

NLI_DAGGER_NOTE = (
    "† NLI premise exceeds the validity envelope for this cell (the model "
    "conditioned on more context than the per-query premise); NLI-based scores "
    "are not comparable and are excluded/daggered."
)

# Deterministic label-offset cycle (points): right-up, right-down, left-up, left-down.
_OFFSET_CYCLE = [(6, 6), (6, -11), (-6, 6), (-6, -11)]


def order_cells(seq: Sequence[str]) -> List[str]:
    """Sort cell names by BASELINE_ORDER; unknown names after, alphabetically."""
    rank = {name: i for i, name in enumerate(BASELINE_ORDER)}
    return sorted(seq, key=lambda n: (rank.get(n, len(BASELINE_ORDER)), n))


def stars(p) -> str:
    """Significance stars for a (Holm-corrected) p-value: * <.05 ** <.01 *** <.001."""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return ""
    if math.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def apply_style() -> None:
    """One rcParams block for every CAGE figure (call once, before plotting).

    Font: Arial with a fallback chain so the same script renders on macOS
    (Arial/Helvetica) and on Linux VMs without Arial (Liberation Sans metrically
    matches Arial; DejaVu Sans always ships with matplotlib) -- no crash, no
    tofu glyphs, at most a findfont info message.
    """
    sns.set_palette("colorblind")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "axes.unicode_minus": False,  # U+2212 missing from some Arial builds
        "savefig.dpi": 300,
        "figure.dpi": 100,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.axisbelow": True,
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
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


def annotate_points(ax, xs, ys, labels, fontsize: int = 7, **text_kw) -> None:
    """Label scatter points without pile-ups, deterministically.

    Strategy: start from a fixed offset cycle, flip the offset inward for points near
    an axes edge, then greedily push later labels further out (in whole label-height
    steps) until their approximate bounding boxes no longer overlap earlier ones.
    Labels displaced far from their point get a thin grey leader line so the
    point <-> label association stays unambiguous.
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
        kw = dict(text_kw)
        if abs(dy) > 18 or abs(dx) > 18:  # displaced label -> leader line
            kw["arrowprops"] = dict(arrowstyle="-", lw=0.6, color="0.55",
                                    shrinkA=0, shrinkB=2)
        ax.annotate(
            label, (x, y), xytext=(dx, dy), textcoords="offset points",
            fontsize=fontsize,
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
            **kw,
        )


def annotate_selected(ax, xs, ys, labels, fontsize: float = 7, **text_kw) -> None:
    """Directly label a SMALL selected subset of points -- no leader lines, ever.

    Rules (reviewer feedback 2026-07-16: crossing leader lines were unreadable):
    - Each label sits immediately beside its point (right by default, flipped left
      near the right axes edge, below near the top edge).
    - Overlaps are resolved by pushing later labels VERTICALLY only, so a label can
      drift up/down but never across another point-label pair; with no connector
      lines drawn, label associations cannot cross by construction.
    Use annotate_points() instead when labelling ALL points of a dense scatter.
    """
    pts = [(float(x), float(y), str(label))
           for x, y, label in zip(xs, ys, labels)
           if not (pd.isna(x) or pd.isna(y))]
    if not pts:
        return

    anns = []
    for x, y, label in pts:
        # Axes-fraction position (via the data transform) handles log scales.
        fx, fy = ax.transAxes.inverted().transform(ax.transData.transform((x, y)))
        dx = -5.0 if fx > 0.70 else 5.0
        dy = -10.0 if fy > 0.88 else 4.0
        anns.append([x, y, label, dx, dy])

    def _bbox(x, y, label, dx, dy):
        px, py = ax.transData.transform((x, y)) * 72.0 / ax.figure.dpi
        w = 0.58 * fontsize * len(label)
        h = fontsize + 2.5
        bx0 = px + dx if dx >= 0 else px + dx - w
        by0 = py + dy if dy >= 0 else py + dy - h
        return (bx0, by0, bx0 + w, by0 + h)

    step = fontsize + 3.0
    for i in range(1, len(anns)):
        for _ in range(25):  # bounded: deterministic, always terminates
            bi = _bbox(*anns[i])
            if not any(
                bi[0] < bj[2] and bi[2] > bj[0] and bi[1] < bj[3] and bi[3] > bj[1]
                for bj in (_bbox(*anns[j]) for j in range(i))
            ):
                break
            anns[i][4] += step if anns[i][4] >= 0 else -step

    for x, y, label, dx, dy in anns:
        ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=fontsize,
                    ha="left" if dx >= 0 else "right",
                    va="bottom" if dy >= 0 else "top",
                    **text_kw)


def annotate_priority(ax, xs, ys, labels, fontsize: float = 9,
                      priority: Sequence[float] | None = None,
                      points: tuple | None = None,
                      **text_kw) -> List[str]:
    """Label points with a NO-OVERLAP guarantee by DROPPING colliding labels.

    Reviewer rule (2026-07-16 iteration: bundled labels were unreadable): points
    are labelled in EXTREMITY order (distance from the axes center, in
    axes-fraction space, so log scales are handled). Each label tries four
    fixed offsets around its point (right-up, right-down, left-up, left-down,
    starting from the edge-appropriate one); a candidate is valid only when it
    stays INSIDE the axes (stacked panels: a label must never spill into the
    neighboring panel) and overlaps no already-placed label. When no candidate
    is valid, the label is DROPPED -- the more extreme point keeps its label
    and the other loses it. An optional ``priority`` sequence (one number per
    point) makes named anchor arms outrank the generic Pareto labels: labels
    are placed in (priority, extremity) order, so when an anchor and a
    non-anchor would collide the anchor keeps its label. When ``points``
    (xs, ys of ALL scatter points) is given, candidates that would strike
    through a marker are avoided as a SOFT preference: a clean spot wins, but
    in a dense cluster escape spots (centered above/below, then far side) are
    tried before falling back to a marker-crossing near spot rather than
    dropping the label (text-over-text is never accepted). No free-form
    displacement, no leader lines. Call AFTER the final axis limits are set.
    Returns the labels actually drawn (callers print the dropped ones for the
    audit trail).
    """
    prio = list(priority) if priority is not None else [0.0] * len(list(labels))
    pts = [(float(x), float(y), str(label), float(p))
           for x, y, label, p in zip(xs, ys, labels, prio)
           if not (pd.isna(x) or pd.isna(y))]
    if not pts:
        return []

    scale = 72.0 / ax.figure.dpi
    ax0, ay0 = ax.transAxes.transform((0.0, 0.0)) * scale
    ax1, ay1 = ax.transAxes.transform((1.0, 1.0)) * scale

    ranked = []
    for x, y, label, p in pts:
        fx, fy = ax.transAxes.inverted().transform(ax.transData.transform((x, y)))
        extremity = max(abs(fx - 0.5), abs(fy - 0.5))
        dx = -5.0 if fx > 0.72 else 5.0
        dy = -13.0 if fy > 0.88 else 5.0
        ranked.append((p, extremity, x, y, label, dx, dy))
    ranked.sort(key=lambda t: (-t[0], -t[1]))  # anchors, then most extreme

    def _bbox(x, y, label, dx, dy):
        px, py = ax.transData.transform((x, y)) * scale
        w = 0.58 * fontsize * len(label)
        h = fontsize + 2.5
        if dx == 0:  # centered above/below the point
            bx0 = px - w / 2.0
        else:
            bx0 = px + dx if dx > 0 else px + dx - w
        by0 = py + dy if dy >= 0 else py + dy - h
        return (bx0, by0, bx0 + w, by0 + h)

    # Marker obstacle boxes (display points): ~4.5 pt half-size per marker.
    markers: List[tuple] = []
    if points is not None:
        for mx, my in zip(*points):
            if pd.isna(mx) or pd.isna(my):
                continue
            pxm, pym = ax.transData.transform((float(mx), float(my))) * scale
            markers.append((pxm - 4.5, pym - 4.5, pxm + 4.5, pym + 4.5,
                            float(mx), float(my)))

    def _hits(bi, boxes):
        return any(bi[0] < b[2] and bi[2] > b[0]
                   and bi[1] < b[3] and bi[3] > b[1] for b in boxes)

    placed: List[tuple] = []
    kept: List[str] = []
    for _, _, x, y, label, dx, dy in ranked:
        dy2 = -13.0 if dy > 0 else 5.0
        near = [(dx, dy), (dx, dy2), (-dx, dy), (-dx, dy2)]
        # Escape candidates for dense clusters, tried only in the
        # marker-avoiding pass: centered above/below, then far side.
        escape = [(0.0, 7.0), (0.0, -16.0),
                  (dx * 3.2, dy), (dx * 3.2, dy2),
                  (-dx * 3.2, dy), (-dx * 3.2, dy2)]
        chosen = None
        for avoid_markers in (True, False):
            candidates = near + escape if avoid_markers else near
            for cdx, cdy in candidates:
                bi = _bbox(x, y, label, cdx, cdy)
                if not (bi[0] >= ax0 - 1 and bi[2] <= ax1 + 1
                        and bi[1] >= ay0 - 1 and bi[3] <= ay1 + 1):
                    continue  # outside the axes -> would bleed into a neighbor
                if _hits(bi, placed):
                    continue  # text-over-text: never accepted
                if avoid_markers and _hits(
                        bi, [m[:4] for m in markers
                             if not (m[4] == x and m[5] == y)]):
                    continue  # prefer a spot that strikes no marker
                chosen = (cdx, cdy, bi)
                break
            if chosen is not None:
                break
        if chosen is None:
            continue  # the less extreme / lower-priority label is dropped
        cdx, cdy, bi = chosen
        placed.append(bi)
        kept.append(label)
        ax.annotate(label, (x, y), xytext=(cdx, cdy),
                    textcoords="offset points", fontsize=fontsize,
                    ha="center" if cdx == 0 else ("left" if cdx > 0 else "right"),
                    va="bottom" if cdy >= 0 else "top",
                    **text_kw)
    return kept


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
        names = ", ".join(display(b) for b in dropped["baseline"].astype(str))
        print(f"  [{fig_name}] dropped (missing {'/'.join(present)}): {names}")
    return out
