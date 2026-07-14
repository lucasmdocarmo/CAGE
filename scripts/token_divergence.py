#!/usr/bin/env python3
"""Token-divergence metric: how often does an arm's greedy output differ from no_cache?

Greedy (T=0) decoding is NEAR-lossless, not identical, across serving configs: floating-point
non-associativity (prefix-cache reuse, eager-vs-compiled kernels, context-length changes) can
flip a near-tie argmax. This tool QUANTIFIES that -- for each baseline arm it compares the
generated answer to the reference arm's answer for the same (example_id, trial) and reports the
fraction that differ. That number is what lets the write-up say "prefix caching is near-lossless
(diverged on X% of queries)" instead of an unquantified "lossless", and it bounds how much of any
cross-config quality delta is token divergence rather than the mechanism.

Interface mirrors statistical_tests.py:
    python scripts/token_divergence.py --results-dir analysis/all_results --reference no_cache \
        --output analysis/all_results/token_divergence.json

RAW divergence = exact string mismatch after strip() (most sensitive).
NORMALIZED divergence = mismatch after lowercase + punctuation/article strip (whether the
    difference survives QA-style normalization, i.e. is a *meaningfully* different answer).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import string
import sys
from pathlib import Path
from typing import Dict, List, Tuple

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))  # generated answers can be long

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def _normalize(text: str) -> str:
    t = text.lower().translate(_PUNCT)
    t = _ARTICLES.sub(" ", t)
    return " ".join(t.split())


def _is_error(row: Dict[str, str]) -> bool:
    err = (row.get("error") or "").strip().lower()
    return err not in {"", "none", "false", "0"}


def _load_answers(baseline_dir: Path) -> Dict[Tuple[str, str], str]:
    """Map (example_id, repeat_index) -> generated_answer across all trial CSVs (skip errors)."""
    out: Dict[Tuple[str, str], str] = {}
    csv_files = sorted(baseline_dir.glob("trial_*/results.csv")) or sorted(baseline_dir.glob("results.csv"))
    for csv_path in csv_files:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                ex = (row.get("example_id") or "").strip()
                if not ex or _is_error(row):
                    continue
                rep = (row.get("repeat_index") or "0").strip() or "0"
                # First non-error occurrence wins (stable if a trial was re-run).
                out.setdefault((ex, rep), row.get("generated_answer") or "")
    return out


def compute_divergence(results_dir: str, reference: str) -> Dict[str, object]:
    root = Path(results_dir)
    ref_dir = root / reference
    if not ref_dir.is_dir():
        raise FileNotFoundError(f"reference arm '{reference}' not found under {results_dir}")
    ref = _load_answers(ref_dir)
    if not ref:
        raise ValueError(f"reference arm '{reference}' has no non-error answers")

    rows: List[Dict[str, object]] = []
    for arm_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        arm = arm_dir.name
        if arm == reference:
            continue
        ans = _load_answers(arm_dir)
        keys = set(ans) & set(ref)  # compare only matched (example_id, trial) pairs
        if not keys:
            continue
        raw_div = sum(1 for k in keys if (ans[k].strip() != ref[k].strip()))
        norm_div = sum(1 for k in keys if _normalize(ans[k]) != _normalize(ref[k]))
        n = len(keys)
        rows.append({
            "arm": arm,
            "n_compared": n,
            "raw_divergent": raw_div,
            "raw_divergence_rate": round(raw_div / n, 4),
            "normalized_divergent": norm_div,
            "normalized_divergence_rate": round(norm_div / n, 4),
        })
    return {"reference": reference, "results_dir": str(root), "arms": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description="Token-divergence vs a reference arm")
    ap.add_argument("--results-dir", required=True, help="Dir of baseline subdirs (like statistical_tests).")
    ap.add_argument("--reference", default="no_cache", help="Reference arm dir name (default: no_cache).")
    ap.add_argument("--output", default=None, help="Path to write the JSON summary.")
    args = ap.parse_args()

    try:
        summary = compute_divergence(args.results_dir, args.reference)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[divergence] SKIP: {exc}", file=sys.stderr)
        return 0  # non-fatal: absent reference is a skip, not a run failure

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"\n[divergence] greedy output vs '{args.reference}' (near-lossless quantification)")
    print(f"{'arm':<28}{'n':>7}{'raw %':>9}{'norm %':>9}")
    for r in summary["arms"]:
        print(f"{r['arm']:<28}{r['n_compared']:>7}"
              f"{100 * r['raw_divergence_rate']:>8.2f}%{100 * r['normalized_divergence_rate']:>8.2f}%")
    if args.output:
        print(f"[divergence] -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
