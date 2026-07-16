#!/usr/bin/env python3
"""Publication tables for CAGE analysis (Markdown + booktabs LaTeX).

Consumed only by generate_plots.py. Two artifacts:

- main_results_table.md / .tex  (F1): one row per cell, grouped by arm family,
  columns = TTFT / Latency / TPOT medians (ms), token-weighted cached-prompt %,
  grounding mean, F1 on answerable questions, abstention %. Serving columns carry
  Holm-corrected Wilcoxon significance stars vs No Cache (straight from
  phase2_stats.json, so the table cannot disagree with the stats). Cells whose
  NLI premise exceeds the validity envelope are daggered.
- speculative_summary_table.md  (F7 companion): the 4 speculative cells with TPOT,
  latency, Delta TPOT vs No Cache and the acceptance | draft-proposed rate.
- trilemma_table.md / .tex  (T4): one row per arm with the three tri-lemma axes
  (serving = TTFT, quality = F1-answerable, memory = resident/marginal KV
  tokens/query) plus normalized 0-100 within-run axis scores.

Also exports trilemma_scores(): the ONE 0-100 axis-score rule, shared with the
score-based tri-lemma figures (trilemma_parallel, trilemma_scores_heat) in
generate_plots.py so figure and table scores cannot disagree.

All numbers come from the summary frame built by generate_plots.build_summary
(pooled per-example estimand shared with the Wilcoxon tables).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from _plot_style import (
    FAMILY_ORDER,
    NLI_INVALID_CELLS,
    display,
    family_of,
    order_cells,
    stars,
)

SERVING_METRICS = ("ttft_ms", "latency_ms", "tpot_ms")


def significance_lookup(stats_payload: Optional[dict]) -> Dict[Tuple[str, str], dict]:
    """(baseline, metric) -> comparison dict from a phase2_stats.json payload."""
    out: Dict[Tuple[str, str], dict] = {}
    for comp in (stats_payload or {}).get("comparisons") or []:
        b, m = comp.get("baseline"), comp.get("metric")
        if b and m:
            out[(b, m)] = comp
    return out


def _num(row: pd.Series, col: str) -> float:
    try:
        v = float(row.get(col))
    except (TypeError, ValueError):
        return float("nan")
    return v


def _fmt_ms(v: float) -> str:
    return "--" if (v is None or math.isnan(v)) else f"{v:,.0f}"


def _fmt_score(v: float, nd: int = 3) -> str:
    return "--" if (v is None or math.isnan(v)) else f"{v:.{nd}f}"


def _fmt_pct(v: float) -> str:
    return "--" if (v is None or math.isnan(v)) else f"{v:.1f}"


def _tex_escape(s: str) -> str:
    return (s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
             .replace("#", r"\#"))


def _grouped_rows(df: pd.DataFrame) -> List[Tuple[str, List[pd.Series]]]:
    """[(family, [rows...])] in FAMILY_ORDER then BASELINE_ORDER."""
    fam_cells: Dict[str, List[str]] = {}
    by_cell = {r["baseline"]: r for _, r in df.iterrows()}
    for cell, row in by_cell.items():
        fam_cells.setdefault(family_of(cell, str(row.get("tree") or "")), []).append(cell)
    grouped: List[Tuple[str, List[pd.Series]]] = []
    for fam in FAMILY_ORDER:
        if fam not in fam_cells:
            continue
        grouped.append((fam, [by_cell[c] for c in order_cells(fam_cells[fam])]))
    return grouped


def _row_values(row: pd.Series, sig: Dict[Tuple[str, str], dict],
                reference: str, tex: bool) -> List[str]:
    cell = row["baseline"]
    vals: List[str] = []
    for m in SERVING_METRICS:
        txt = _fmt_ms(_num(row, m))
        if cell != reference and txt != "--":
            st = stars((sig.get((cell, m)) or {}).get("p_value_holm"))
            if st:
                txt += f"$^{{{st}}}$" if tex else st
        vals.append(txt)
    vals.append(_fmt_pct(_num(row, "cached_pct")))
    ground = _fmt_score(_num(row, "grounding_mean"))
    if cell in NLI_INVALID_CELLS and ground != "--":
        ground += r"$^{\dagger}$" if tex else "†"
    vals.append(ground)
    vals.append(_fmt_score(_num(row, "f1_answerable_mean")))
    vals.append(_fmt_pct(_num(row, "abstention_pct")))
    return vals


HEADERS = ["Baseline", "TTFT (ms)", "Latency (ms)", "TPOT (ms)",
           "Cached (%)", "Grounding", "F1 answerable", "Abstention (%)"]

NOTES = [
    "Medians are pooled per-example medians (same estimand as the Wilcoxon tables); "
    "N = 300 valid measurements per cell (100 questions x 3 trials, repeat-0 rows).",
    "Stars: Holm-corrected Wilcoxon signed-rank vs No Cache on the serving columns "
    "(* p<0.05, ** p<0.01, *** p<0.001).",
    "Cached (%) is the token-weighted cached-prompt share: sum(cached prompt tokens) / "
    "sum(prompt tokens); Grounding is the LettuceDetect mean (medians saturate at 1.0); "
    "F1 answerable is the mean F1 on answerable questions; Abstention is the share of "
    "predicted no-answer responses.",
    "† NLI premise exceeds the validity envelope for this cell; NLI-based scores are "
    "not comparable (grounding shown for completeness only).",
]

# TeX-safe versions of NOTES (kept in sync by hand; no string surgery at runtime).
NOTES_TEX = [
    "Medians are pooled per-example medians (same estimand as the Wilcoxon tables); "
    "$N = 300$ valid measurements per cell (100 questions $\\times$ 3 trials, "
    "repeat-0 rows).",
    "Stars: Holm-corrected Wilcoxon signed-rank vs.\\ No Cache on the serving columns "
    "($^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$).",
    "Cached (\\%) is the token-weighted cached-prompt share (total cached prompt "
    "tokens over total prompt tokens); Grounding is the LettuceDetect mean (medians "
    "saturate at 1.0); F1 answerable is the mean F1 on answerable questions; "
    "Abstention is the share of predicted no-answer responses.",
    "$^{\\dagger}$NLI premise exceeds the validity envelope for this cell; NLI-based "
    "scores are not comparable (grounding shown for completeness only).",
]


def write_main_results_table(df: pd.DataFrame, stats_payload: Optional[dict],
                             out_dir: Path) -> List[str]:
    """Write main_results_table.md and .tex; returns the file names written."""
    out_dir = Path(out_dir)
    sig = significance_lookup(stats_payload)
    reference = (stats_payload or {}).get("reference", "no_cache")
    grouped = _grouped_rows(df)
    written: List[str] = []

    # ---------------- Markdown ----------------
    md: List[str] = ["# Main results (Phase 2, Qwen3-8B, SQuAD v2)", ""]
    md.append("| " + " | ".join(HEADERS) + " |")
    md.append("|" + "---|" * len(HEADERS))
    for fam, rows in grouped:
        md.append(f"| **{fam}** |" + " |" * (len(HEADERS) - 1))
        for row in rows:
            cell = row["baseline"]
            name = display(cell) + (" (ref.)" if cell == reference else "")
            md.append("| " + " | ".join([name] + _row_values(row, sig, reference,
                                                             tex=False)) + " |")
    md.append("")
    for note in NOTES:
        md.append(f"- {note}")
    md.append("")
    (out_dir / "main_results_table.md").write_text("\n".join(md), encoding="utf-8")
    written.append("main_results_table.md")

    # ---------------- LaTeX (booktabs) ----------------
    ncols = len(HEADERS)
    tex: List[str] = [
        "% Auto-generated by generate_plots.py -- do not edit by hand.",
        "% Requires: \\usepackage{booktabs}",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Serving and quality results per arm (Phase 2, Qwen3-8B, "
        "SQuAD v2). Medians are pooled per-example medians; N = 300 valid "
        "measurements per cell (100 questions $\\times$ 3 trials). Stars: "
        "Holm-corrected Wilcoxon signed-rank vs.\\ No Cache "
        "($^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$).}",
        "\\label{tab:phase2-main-results}",
        "\\small",
        "\\begin{tabular}{l" + "r" * (ncols - 1) + "}",
        "\\toprule",
        " & ".join([_tex_escape(h) for h in HEADERS]) + " \\\\",
        "\\midrule",
    ]
    for gi, (fam, rows) in enumerate(grouped):
        if gi > 0:
            tex.append("\\addlinespace")
        tex.append(f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{{_tex_escape(fam)}}}}} \\\\")
        for row in rows:
            cell = row["baseline"]
            name = _tex_escape(display(cell)) + (" (ref.)" if cell == reference else "")
            tex.append(" & ".join([name] + _row_values(row, sig, reference,
                                                       tex=True)) + " \\\\")
    tex += [
        "\\bottomrule",
        "\\end{tabular}",
        "",
        "\\vspace{2pt}",
        "\\begin{minipage}{\\linewidth}\\footnotesize",
    ]
    for note_tex in NOTES_TEX:
        tex.append(note_tex + " \\\\")
    tex += [
        "\\end{minipage}",
        "\\end{table}",
        "",
    ]
    tex_text = "\n".join(tex)
    (out_dir / "main_results_table.tex").write_text(tex_text, encoding="utf-8")
    written.append("main_results_table.tex")
    if tex_text.count("{") != tex_text.count("}"):
        print("  [main_results_table.tex] WARNING: unbalanced braces "
              f"({tex_text.count('{')} vs {tex_text.count('}')})")
    return written


# ---------------------------------------------------------------------------
# T4: tri-lemma table
# ---------------------------------------------------------------------------

TRILEMMA_HEADERS = ["Baseline", "TTFT (ms)", "Latency (ms)", "F1 ans.",
                    "Grounding", "Resident KV", "Marginal KV", "Cached (%)",
                    "Serving", "Quality", "Memory"]

TRILEMMA_NOTES = [
    "Tri-lemma axes -- SERVING: TTFT median (ms; lower is better). QUALITY: mean "
    "F1 on answerable questions (valid for all arms). MEMORY: per-query KV "
    "context footprint -- Resident KV = median prompt tokens/query held in KV; "
    "Marginal KV = median NEW KV tokens materialized per query (prompt - cached, "
    "missing cached = 0); Cached (%) is the token-weighted cached-prompt share.",
    "Axis scores are 0-100, min-max normalized WITHIN THIS RUN (relative, not "
    "absolute): serving = min-max of -log TTFT; quality = min-max of "
    "F1-answerable; memory = min-max of -marginal KV. 100 = best arm in the run "
    "on that axis.",
    "KV footprint proxy (prompt-token counts); a swept memory-pressure axis is "
    "future work.",
    "† NLI premise exceeds the validity envelope for this cell; grounding shown "
    "for completeness only.",
]

TRILEMMA_NOTES_TEX = [
    "Tri-lemma axes -- SERVING: TTFT median (ms; lower is better). QUALITY: mean "
    "F1 on answerable questions (valid for all arms). MEMORY: per-query KV "
    "context footprint -- Resident KV = median prompt tokens/query held in KV; "
    "Marginal KV = median NEW KV tokens materialized per query (prompt $-$ "
    "cached, missing cached = 0); Cached (\\%) is the token-weighted "
    "cached-prompt share.",
    "Axis scores are 0--100, min-max normalized WITHIN THIS RUN (relative, not "
    "absolute): serving = min-max of $-\\log$ TTFT; quality = min-max of "
    "F1-answerable; memory = min-max of $-$marginal KV. 100 = best arm in the "
    "run on that axis.",
    "KV footprint proxy (prompt-token counts); a swept memory-pressure axis is "
    "future work.",
    "$^{\\dagger}$NLI premise exceeds the validity envelope for this cell; "
    "grounding shown for completeness only.",
]


def _minmax_score(values: pd.Series) -> pd.Series:
    """0-100 min-max normalization; degenerate ranges collapse to 50."""
    s = pd.to_numeric(values, errors="coerce")
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or not hi > lo:
        return pd.Series(50.0, index=s.index).where(s.notna())
    return 100.0 * (s - lo) / (hi - lo)


TRILEMMA_SCORE_COLS = ("serving_score", "quality_score", "memory_score")


def trilemma_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of the summary frame with the THREE canonical 0-100 axis scores.

    The one scoring rule shared by trilemma_table (T4) and the score-based
    figures (trilemma_parallel, trilemma_scores_heat), so they cannot disagree:
    serving = min-max of -log TTFT; quality = min-max of mean F1-answerable;
    memory = min-max of -marginal KV tokens/query. Min-max is WITHIN THIS RUN
    (relative, not absolute); 100 = best arm in the run on that axis. Normalize
    on the FULL frame before any figure-level row filtering so every consumer
    sees identical scores.
    """
    d = df.copy()

    def _col(name: str) -> pd.Series:
        if name in d.columns:
            return pd.to_numeric(d[name], errors="coerce")
        return pd.Series(np.nan, index=d.index, dtype=float)

    ttft = _col("ttft_ms")
    d["serving_score"] = _minmax_score(-np.log(ttft.where(ttft > 0)))
    d["quality_score"] = _minmax_score(_col("f1_answerable_mean"))
    d["memory_score"] = _minmax_score(-_col("marginal_kv"))
    return d


def _trilemma_row(row: pd.Series, tex: bool) -> List[str]:
    cell = row["baseline"]
    vals = [_fmt_ms(_num(row, "ttft_ms")), _fmt_ms(_num(row, "latency_ms")),
            _fmt_score(_num(row, "f1_answerable_mean"))]
    ground = _fmt_score(_num(row, "grounding_mean"))
    if cell in NLI_INVALID_CELLS and ground != "--":
        ground += r"$^{\dagger}$" if tex else "†"
    vals.append(ground)
    vals.append(_fmt_ms(_num(row, "resident_kv")))
    vals.append(_fmt_ms(_num(row, "marginal_kv")))
    vals.append(_fmt_pct(_num(row, "cached_pct")))
    for score_col in TRILEMMA_SCORE_COLS:
        v = _num(row, score_col)
        vals.append("--" if math.isnan(v) else f"{v:.0f}")
    return vals


def write_trilemma_table(df: pd.DataFrame, out_dir: Path) -> List[str]:
    """Write trilemma_table.md and .tex (T4); returns the file names written."""
    out_dir = Path(out_dir)
    needed = ("ttft_ms", "f1_answerable_mean", "marginal_kv")
    missing = [c for c in needed
               if c not in df.columns or not df[c].notna().any()]
    if missing:
        print(f"  [trilemma_table] missing columns {', '.join(missing)}; skipped")
        return []

    d = trilemma_scores(df)
    grouped = _grouped_rows(d)
    written: List[str] = []

    # ---------------- Markdown ----------------
    md: List[str] = ["# Tri-lemma table (Phase 2, Qwen3-8B, SQuAD v2)", ""]
    md.append("| " + " | ".join(TRILEMMA_HEADERS) + " |")
    md.append("|" + "---|" * len(TRILEMMA_HEADERS))
    for fam, rows in grouped:
        md.append(f"| **{fam}** |" + " |" * (len(TRILEMMA_HEADERS) - 1))
        for row in rows:
            md.append("| " + " | ".join([display(row["baseline"])]
                                        + _trilemma_row(row, tex=False)) + " |")
    md.append("")
    for note in TRILEMMA_NOTES:
        md.append(f"- {note}")
    md.append("")
    (out_dir / "trilemma_table.md").write_text("\n".join(md), encoding="utf-8")
    written.append("trilemma_table.md")

    # ---------------- LaTeX (booktabs) ----------------
    ncols = len(TRILEMMA_HEADERS)
    tex: List[str] = [
        "% Auto-generated by generate_plots.py -- do not edit by hand.",
        "% Requires: \\usepackage{booktabs}",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{The serving $\\times$ quality $\\times$ KV-memory tri-lemma "
        "per arm (Phase 2, Qwen3-8B, SQuAD v2). Serving = TTFT median; quality "
        "= mean F1 on answerable; memory = per-query KV context footprint "
        "(resident = median prompt tokens, marginal = median new KV tokens "
        "materialized per query). Axis scores are 0--100 min-max normalized "
        "within this run (relative, not absolute). KV footprint proxy; a swept "
        "memory-pressure axis is future work.}",
        "\\label{tab:phase2-trilemma}",
        "\\footnotesize",
        "\\begin{tabular}{l" + "r" * (ncols - 1) + "}",
        "\\toprule",
        " & ".join([_tex_escape(h) for h in TRILEMMA_HEADERS]) + " \\\\",
        "\\midrule",
    ]
    for gi, (fam, rows) in enumerate(grouped):
        if gi > 0:
            tex.append("\\addlinespace")
        tex.append(f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{{_tex_escape(fam)}}}}} \\\\")
        for row in rows:
            tex.append(" & ".join([_tex_escape(display(row["baseline"]))]
                                  + _trilemma_row(row, tex=True)) + " \\\\")
    tex += [
        "\\bottomrule",
        "\\end{tabular}",
        "",
        "\\vspace{2pt}",
        "\\begin{minipage}{\\linewidth}\\footnotesize",
    ]
    for note_tex in TRILEMMA_NOTES_TEX:
        tex.append(note_tex + " \\\\")
    tex += [
        "\\end{minipage}",
        "\\end{table}",
        "",
    ]
    tex_text = "\n".join(tex)
    (out_dir / "trilemma_table.tex").write_text(tex_text, encoding="utf-8")
    written.append("trilemma_table.tex")
    if tex_text.count("{") != tex_text.count("}"):
        print("  [trilemma_table.tex] WARNING: unbalanced braces "
              f"({tex_text.count('{')} vs {tex_text.count('}')})")
    return written


def write_speculative_table(df: pd.DataFrame, acc: Optional[pd.DataFrame],
                            stats_payload: Optional[dict], out_dir: Path) -> List[str]:
    """speculative_summary_table.md: the 4 spec cells + acceptance | draft proposed."""
    out_dir = Path(out_dir)
    spec_cells = [c for c in order_cells(df["baseline"])
                  if str(c).startswith("spec_")]
    if not spec_cells:
        print("  [speculative_summary_table.md] no speculative cells; skipped")
        return []
    sig = significance_lookup(stats_payload)
    acc_by_cell: Dict[str, dict] = {}
    if acc is not None and not acc.empty and "cell" in acc.columns:
        acc_by_cell = {str(r["cell"]): r for _, r in acc.iterrows()}
    by_cell = {r["baseline"]: r for _, r in df.iterrows()}
    ref_row = by_cell.get("no_cache")
    ref_tpot = _num(ref_row, "tpot_ms") if ref_row is not None else float("nan")

    lines = [
        "# Speculative decoding summary (Phase 2)",
        "",
        "| Arm | TPOT (ms, median) | Latency (ms, median) | Delta TPOT vs No Cache | "
        "Acceptance \\| draft proposed |",
        "|---|---|---|---|---|",
    ]
    for cell in spec_cells:
        row = by_cell[cell]
        tpot = _num(row, "tpot_ms")
        comp = sig.get((cell, "tpot_ms")) or {}
        d = comp.get("median_diff")
        st = stars(comp.get("p_value_holm"))
        delta = "--" if d is None else f"{float(d):+,.1f} ms{st}"
        a = acc_by_cell.get(cell)
        acc_txt = "--"
        if a is not None:
            try:
                acc_txt = f"{float(a['acceptance_rate_cumulative']):.3f}"
            except (KeyError, TypeError, ValueError):
                acc_txt = "--"
        lines.append(
            f"| {display(cell)} | {_fmt_ms(tpot)} | {_fmt_ms(_num(row, 'latency_ms'))} "
            f"| {delta} | {acc_txt} |")
    lines += [
        "",
        f"- No Cache reference TPOT: {_fmt_ms(ref_tpot)} ms (median).",
        "- Acceptance is the cumulative acceptance rate GIVEN the drafter proposed "
        "(acceptance | draft proposed), not an unconditional acceptance rate; "
        "EAGLE-3 proposes on every step, the n-gram drafter only on prefix hits, "
        "so the two columns are not directly comparable.",
        "- Delta TPOT and stars: Holm-corrected Wilcoxon signed-rank vs No Cache "
        "(paired per-example median difference).",
        "",
    ]
    (out_dir / "speculative_summary_table.md").write_text("\n".join(lines),
                                                          encoding="utf-8")
    return ["speculative_summary_table.md"]
