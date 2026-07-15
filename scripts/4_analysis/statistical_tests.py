#!/usr/bin/env python3
"""
Statistical significance testing for CAGE experiments.

WHY THIS EXISTS
---------------
Reporting mean +/- std over n=3 trials (the old approach) cannot support any
significance claim: the trials are few and were not always independent in early runs.
This script instead tests at the PER-QUERY level (n ~= number of queries, e.g.
50), which is the correct unit of analysis, and pairs observations by
``example_id`` so the same questions are compared across baselines.

WHAT IT DOES
------------
For each metric and each (baseline vs reference) pair:
  * Builds a per-example value for each baseline by averaging that example's rows
    across trials (so each question contributes once).
  * Inner-joins the two baselines on ``example_id`` -> paired samples.
  * Runs the Wilcoxon signed-rank test (paired, non-parametric). Falls back to
    Mann-Whitney U if pairing is impossible.
  * Reports: median difference, rank-biserial effect size, Cliff's delta, a
    bootstrap 95% CI of the mean difference, the p-value, and a Holm-Bonferroni
    adjusted p-value across the comparisons for that metric.
  * Is direction-aware: latency/TTFT/hallucination "lower is better"; quality
    metrics "higher is better". "Improvement" is reported in the right direction.

Outputs a console table, a JSON summary (--output), and an optional LaTeX table
(--latex-out) ready for the paper.

scipy is used if available; otherwise a numpy-only fallback (normal-approx
Wilcoxon + bootstrap) is used so the script never hard-fails on a bare cluster.

USAGE
-----
  python3 scripts/4_analysis/statistical_tests.py --results-dir results/<phase>/<run-id>/stats/all_results
  python3 scripts/4_analysis/statistical_tests.py --results-dir results/<phase>/<run-id>/stats/all_results \
      --reference no_cache \
      --metrics ttft_ms latency_ms faithfulness grounding_score f1_score \
      --output results/<phase>/<run-id>/stats/stats.json --latex-out results/<phase>/<run-id>/stats/stats.tex
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:  # optional, preferred
    from scipy import stats as _scipy_stats  # type: ignore
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - cluster without scipy
    _scipy_stats = None
    _HAVE_SCIPY = False


# Metric direction: True => higher is better, False => lower is better.
METRIC_HIGHER_IS_BETTER: Dict[str, bool] = {
    # latency / cost (lower better)
    "ttft_ms": False,
    "latency_ms": False,
    "avg_tpot_ms": False,
    "tpot_ms": False,  # per-query time-per-output-token (results.csv column)
    "hallucinated_span_ratio": False,
    "hallucination_detected": False,
    # quality (higher better)
    "faithfulness": True,
    "grounding_score": True,
    "supported_claim_ratio": True,
    "context_relevance": True,
    "relevance": True,
    "completeness_bertscore": True,
    "completeness_rouge_l": True,
    "f1_score": True,
    "exact_match": True,
    "precision": True,
    "recall": True,
    # SQuAD v2 no-answer decomposition (fix #4): answerable-only F1/EM and abstention
    # accuracy are all higher-better. Reported as separate comparison rows so a serving
    # config that regresses abstention (starts hallucinating on no-answer items) is visible.
    "f1_answerable": True,
    "exact_match_answerable": True,
    "no_answer_correct": True,
    "abstention_precision": True,
    "cached_prompt_ratio": True,
}

DEFAULT_METRICS = [
    "ttft_ms",
    "latency_ms",
    "grounding_score",
    "faithfulness",
    "hallucinated_span_ratio",
    "f1_score",
    "exact_match",
    # SQuAD v2 no-answer decomposition (fix #4): None on inapplicable rows, so these
    # subset automatically (answerable-only F1/EM; abstention accuracy on no-answer items).
    "f1_answerable",
    "exact_match_answerable",
    "no_answer_correct",
    "abstention_precision",
    "completeness_rouge_l",
    # The most-plotted quality metric was never significance-tested (2026-07-15 audit).
    # Signed (baseline-rescaled) and near-zero for short QA answers -- test it anyway so
    # figures and tables come from the same set.
    "completeness_bertscore",
]


@dataclass
class ComparisonResult:
    metric: str
    baseline: str
    reference: str
    test: str               # "wilcoxon" | "mannwhitney"
    n_pairs: int
    median_reference: Optional[float]
    median_baseline: Optional[float]
    median_diff: Optional[float]        # baseline - reference
    pct_change: Optional[float]         # signed % change baseline vs reference
    improvement: Optional[bool]         # True if baseline is better than reference
    effect_size: Optional[float]        # rank-biserial correlation
    cliffs_delta: Optional[float]
    ci95_low: Optional[float]           # bootstrap CI of mean(baseline-reference)
    ci95_high: Optional[float]
    p_value: Optional[float]
    p_value_holm: Optional[float] = None
    significant: Optional[bool] = None  # at adjusted alpha
    note: str = ""


# --------------------------------------------------------------------------- #
# Data loading -- delegated to the canonical loader (2026-07-15 audit fix): one
# parser + one validity rule shared with plots/verify, so figures and this table
# derive from the same estimand. Behavior delta vs the legacy inline loader:
# empty_generation rows are now excluded from serving metrics too (documented
# canonical rule); quality metrics are unchanged (nulled at source on such rows).
# --------------------------------------------------------------------------- #
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _results_loader import discover_cells, load_cell, per_example as _per_example_table  # noqa: E402


def load_baseline_per_example(baseline_dir: Path, metrics: List[str]) -> Dict[str, Dict[str, float]]:
    """Return {example_id: {metric: mean_over_trials_value}} for one baseline."""
    df = load_cell(Path(baseline_dir))
    out: Dict[str, Dict[str, float]] = {}
    for m in metrics:
        for row in _per_example_table(df, m).itertuples(index=False):
            ex_id = str(row.example_id).strip()
            if ex_id:
                out.setdefault(ex_id, {})[m] = float(row.value)
    return out


def discover_baselines(results_dir: Path) -> List[str]:
    return sorted(cell for _tree, cell, _dir in discover_cells(Path(results_dir)))


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def _wilcoxon(diffs: np.ndarray) -> Tuple[float, float]:
    """Return (statistic, p_value) for a paired signed-rank test on diffs."""
    nonzero = diffs[diffs != 0]
    n = len(nonzero)
    if n == 0:
        return float("nan"), 1.0
    if _HAVE_SCIPY:
        try:
            res = _scipy_stats.wilcoxon(diffs, zero_method="wilcox", correction=False, mode="auto")
            return float(res.statistic), float(res.pvalue)
        except Exception:
            pass
    # Normal approximation fallback.
    ranks = _rankdata(np.abs(nonzero))
    signs = np.sign(nonzero)
    w_plus = float(np.sum(ranks[signs > 0]))
    mean_w = n * (n + 1) / 4.0
    std_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if std_w == 0:
        return w_plus, 1.0
    z = (w_plus - mean_w) / std_w
    p = 2.0 * (1.0 - _norm_cdf(abs(z)))
    return w_plus, max(0.0, min(1.0, p))


def _mannwhitney(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    if _HAVE_SCIPY:
        try:
            res = _scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
            return float(res.statistic), float(res.pvalue)
        except Exception:
            pass
    combined = np.concatenate([a, b])
    ranks = _rankdata(combined)
    ra = float(np.sum(ranks[: len(a)]))
    na, nb = len(a), len(b)
    u_a = ra - na * (na + 1) / 2.0
    mean_u = na * nb / 2.0
    std_u = math.sqrt(na * nb * (na + nb + 1) / 12.0)
    if std_u == 0:
        return u_a, 1.0
    z = (u_a - mean_u) / std_u
    p = 2.0 * (1.0 - _norm_cdf(abs(z)))
    return u_a, max(0.0, min(1.0, p))


def _rankdata(x: np.ndarray) -> np.ndarray:
    if _HAVE_SCIPY:
        return _scipy_stats.rankdata(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _rank_biserial_paired(diffs: np.ndarray) -> float:
    """Rank-biserial correlation effect size for a paired signed-rank test."""
    nonzero = diffs[diffs != 0]
    n = len(nonzero)
    if n == 0:
        return 0.0
    ranks = _rankdata(np.abs(nonzero))
    total = ranks.sum()
    signs = np.sign(nonzero)
    w_plus = ranks[signs > 0].sum()
    w_minus = ranks[signs < 0].sum()
    return float((w_plus - w_minus) / total)


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta: P(a>b) - P(a<b). Robust nonparametric effect size."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    gt = 0
    lt = 0
    # Vectorised over b for each a (n is small, ~50, so this is fine).
    for x in a:
        gt += int(np.sum(x > b))
        lt += int(np.sum(x < b))
    return float((gt - lt) / (len(a) * len(b)))


def _bootstrap_ci_mean_diff(diffs: np.ndarray, iters: int = 10000, seed: int = 42) -> Tuple[float, float]:
    if len(diffs) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(diffs)
    idx = rng.integers(0, n, size=(iters, n))
    means = diffs[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _holm_bonferroni(pvals: List[Optional[float]], alpha: float = 0.05) -> Tuple[List[Optional[float]], List[Optional[bool]]]:
    indexed = [(i, p) for i, p in enumerate(pvals) if p is not None]
    indexed.sort(key=lambda t: t[1])
    m = len(indexed)
    adj: List[Optional[float]] = [None] * len(pvals)
    sig: List[Optional[bool]] = [None] * len(pvals)
    prev = 0.0
    for rank, (i, p) in enumerate(indexed):
        a = (m - rank) * p
        a = min(1.0, max(a, prev))  # enforce monotonic non-decreasing
        prev = a
        adj[i] = a
        sig[i] = a < alpha
    return adj, sig


# --------------------------------------------------------------------------- #
# Comparison driver
# --------------------------------------------------------------------------- #
def compare_pair(
    metric: str,
    baseline_name: str,
    reference_name: str,
    baseline_pe: Dict[str, Dict[str, float]],
    reference_pe: Dict[str, Dict[str, float]],
    bootstrap_iters: int,
) -> Optional[ComparisonResult]:
    # Build paired arrays on shared example_ids.
    shared = [
        ex for ex in baseline_pe
        if ex in reference_pe and metric in baseline_pe[ex] and metric in reference_pe[ex]
    ]
    if len(shared) < 3:
        # Not enough pairs -> try unpaired Mann-Whitney on whatever exists.
        a = np.array([d[metric] for d in baseline_pe.values() if metric in d], dtype=float)
        b = np.array([d[metric] for d in reference_pe.values() if metric in d], dtype=float)
        if len(a) < 3 or len(b) < 3:
            return None
        _, p = _mannwhitney(a, b)
        higher_better = METRIC_HIGHER_IS_BETTER.get(metric, True)
        med_b, med_r = float(np.median(a)), float(np.median(b))
        diff = med_b - med_r
        improvement = (diff > 0) == higher_better
        return ComparisonResult(
            metric=metric, baseline=baseline_name, reference=reference_name,
            test="mannwhitney", n_pairs=min(len(a), len(b)),
            median_reference=med_r, median_baseline=med_b, median_diff=diff,
            pct_change=(diff / med_r * 100.0) if med_r else None,
            improvement=improvement, effect_size=None,
            cliffs_delta=_cliffs_delta(a, b),
            ci95_low=None, ci95_high=None, p_value=p,
            note="unpaired (insufficient shared example_ids)",
        )

    b_vals = np.array([baseline_pe[ex][metric] for ex in shared], dtype=float)
    r_vals = np.array([reference_pe[ex][metric] for ex in shared], dtype=float)
    diffs = b_vals - r_vals

    _, p = _wilcoxon(diffs)
    higher_better = METRIC_HIGHER_IS_BETTER.get(metric, True)
    med_b, med_r = float(np.median(b_vals)), float(np.median(r_vals))
    median_diff = float(np.median(diffs))
    improvement = (median_diff > 0) == higher_better if median_diff != 0 else None
    ci_low, ci_high = _bootstrap_ci_mean_diff(diffs, iters=bootstrap_iters)

    return ComparisonResult(
        metric=metric, baseline=baseline_name, reference=reference_name,
        test="wilcoxon", n_pairs=len(shared),
        median_reference=med_r, median_baseline=med_b, median_diff=median_diff,
        pct_change=(median_diff / med_r * 100.0) if med_r else None,
        improvement=improvement,
        effect_size=_rank_biserial_paired(diffs),
        cliffs_delta=_cliffs_delta(b_vals, r_vals),
        ci95_low=ci_low, ci95_high=ci_high, p_value=p,
    )


def run(args: argparse.Namespace) -> int:
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: results dir not found: {results_dir}")
        return 2

    baselines = discover_baselines(results_dir)
    if not baselines:
        print(f"ERROR: no baseline result dirs with results.csv under {results_dir}")
        return 2

    reference = args.reference if args.reference in baselines else baselines[0]
    if args.reference and args.reference not in baselines:
        print(f"WARNING: reference '{args.reference}' not found; using '{reference}'")
    others = [b for b in baselines if b != reference]
    metrics = args.metrics or DEFAULT_METRICS

    print(f"Reference baseline : {reference}")
    print(f"Compared baselines : {', '.join(others)}")
    print(f"Metrics            : {', '.join(metrics)}")
    print(f"scipy available    : {_HAVE_SCIPY}\n")

    # Load per-example tables once.
    per_example = {b: load_baseline_per_example(results_dir / b, metrics) for b in baselines}

    results: List[ComparisonResult] = []
    for metric in metrics:
        metric_results: List[ComparisonResult] = []
        for b in others:
            res = compare_pair(metric, b, reference, per_example[b], per_example[reference], args.bootstrap_iters)
            if res is not None:
                metric_results.append(res)
        # Holm-Bonferroni within each metric across the baseline comparisons.
        adj, sig = _holm_bonferroni([r.p_value for r in metric_results], alpha=args.alpha)
        for r, a, s in zip(metric_results, adj, sig):
            r.p_value_holm = a
            r.significant = s
        results.extend(metric_results)

    _print_table(results, metrics)

    summary = {
        "results_dir": str(results_dir),
        "reference": reference,
        "alpha": args.alpha,
        "scipy": _HAVE_SCIPY,
        "comparisons": [asdict(r) for r in results],
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nWrote JSON summary -> {args.output}")
    if args.latex_out:
        Path(args.latex_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.latex_out).write_text(
            _to_latex(results, reference, args.latex_label), encoding="utf-8")
        print(f"Wrote LaTeX table  -> {args.latex_out}")
    return 0


def _fmt(x: Optional[float], nd: int = 3) -> str:
    return "  n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _print_table(results: List[ComparisonResult], metrics: List[str]) -> None:
    for metric in metrics:
        rows = [r for r in results if r.metric == metric]
        if not rows:
            continue
        better = "higher" if METRIC_HIGHER_IS_BETTER.get(metric, True) else "lower"
        print("=" * 100)
        print(f"METRIC: {metric}   ({better} is better)")
        print("-" * 100)
        print(f"{'baseline':<32}{'n':>4}{'med Δ':>10}{'%chg':>9}{'effect':>8}"
              f"{'p':>10}{'p(holm)':>10}{'sig':>5}{'better?':>9}")
        for r in rows:
            print(f"{r.baseline:<32}{r.n_pairs:>4}{_fmt(r.median_diff):>10}"
                  f"{_fmt(r.pct_change, 1):>9}{_fmt(r.effect_size, 2):>8}"
                  f"{_fmt(r.p_value):>10}{_fmt(r.p_value_holm):>10}"
                  f"{('yes' if r.significant else 'no'):>5}"
                  f"{('yes' if r.improvement else ('no' if r.improvement is not None else '?')):>9}")
        print()


def _to_latex(results: List[ComparisonResult], reference: str,
              label: str = "tab:significance") -> str:
    lines = [
        "% Auto-generated by scripts/4_analysis/statistical_tests.py",
        "\\begin{table}[t]",
        "\\centering",
        # BOTH caption lines must be f-strings: a plain-string continuation leaves "}}" as
        # two literal braces (unbalanced -> "Too many }'s") and the reference must be
        # underscore-escaped or "no_cache" kills the compile ("Missing $ inserted").
        f"\\caption{{Per-query significance vs. \\texttt{{{_latex_escape(reference)}}} "
        f"(Wilcoxon signed-rank, Holm-corrected). $\\Delta$ is median baseline$-$reference.}}",
        # Parameterized so the Qwen and MiMo tables can both be \input without a
        # duplicate-\label clash.
        f"\\label{{{label}}}",
        "\\begin{tabular}{llrrrrc}",
        "\\toprule",
        "Metric & Baseline & $n$ & Median $\\Delta$ & \\% chg & $p_{\\text{Holm}}$ & Sig. \\\\",
        "\\midrule",
    ]
    for r in results:
        sig = "\\checkmark" if r.significant else "--"
        lines.append(
            f"{_latex_escape(r.metric)} & {_latex_escape(r.baseline)} & {r.n_pairs} & "
            f"{_fmt(r.median_diff)} & {_fmt(r.pct_change, 1)} & {_fmt(r.p_value_holm)} & {sig} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def _latex_escape(s: str) -> str:
    return s.replace("_", "\\_")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-query significance testing for CAGE baselines.")
    p.add_argument("--results-dir", required=True,
                   help="Dir containing per-baseline subdirs (each with trial_*/results.csv).")
    p.add_argument("--reference", default="no_cache",
                   help="Reference baseline to compare others against (default: no_cache).")
    p.add_argument("--metrics", nargs="*", default=None,
                   help=f"Metrics to test (default: {' '.join(DEFAULT_METRICS)}).")
    p.add_argument("--alpha", type=float, default=0.05, help="Significance level (default 0.05).")
    p.add_argument("--bootstrap-iters", type=int, default=10000, help="Bootstrap iterations for CIs.")
    p.add_argument("--output", default=None, help="Path to write JSON summary.")
    p.add_argument("--latex-out", default=None, help="Path to write a LaTeX significance table.")
    p.add_argument("--latex-label", default="tab:significance",
                   help="\\label for the LaTeX table (use a distinct label per model pass).")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
