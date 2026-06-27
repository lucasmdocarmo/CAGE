#!/usr/bin/env python3
"""
CAGE Q/A evidence extractor.

For each baseline, samples question/answer pairs from the BEGINNING, MIDDLE, and
END of its results.csv into a dissertation-ready evidence file (markdown + JSON),
so every run is documented with concrete Q/A samples per baseline/experiment.

Usage:
  python scripts/extract_qa_evidence.py
  python scripts/extract_qa_evidence.py --results-dir analysis/phase1/results --out evidence --n 10
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _f(x):
    try:
        return round(float(x), 2)
    except (TypeError, ValueError):
        return x


def sample_positions(rows: list, n: int) -> list:
    """Return [(position, index, row)] for ~n from begin/middle/end (clamped if few rows)."""
    total = len(rows)
    if total == 0:
        return []
    if total <= 3 * n:
        # Too few rows for three distinct windows: label by thirds, show all.
        picks = []
        for i, r in enumerate(rows):
            pos = "begin" if i < total / 3 else ("middle" if i < 2 * total / 3 else "end")
            picks.append((pos, i, r))
        return picks
    mid0 = total // 2 - n // 2
    return (
        [("begin", i, rows[i]) for i in range(n)]
        + [("middle", i, rows[i]) for i in range(mid0, mid0 + n)]
        + [("end", i, rows[i]) for i in range(total - n, total)]
    )


def extract(csv_path: Path, n: int):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    samples = []
    for pos, idx, r in sample_positions(rows, n):
        samples.append({
            "position": pos,
            "index": idx,
            "example_id": r.get("example_id"),
            "question": r.get("question"),
            "reference_answer": r.get("reference_answer"),
            "generated_answer": r.get("generated_answer"),
            "ttft_ms": _f(r.get("ttft_ms")),
            "latency_ms": _f(r.get("latency_ms")),
            "cached_prompt_ratio": _f(r.get("cached_prompt_ratio")),
            "finish_reason": r.get("finish_reason"),
            "error": (r.get("error") or "").strip() or None,
        })
    return rows, samples


def md_for(baseline: str, total: int, samples: list) -> str:
    out = [f"## {baseline}", "",
           f"_{total} queries; {len(samples)} Q/A samples (begin / middle / end)._", ""]
    cur = None
    for s in samples:
        if s["position"] != cur:
            cur = s["position"]
            out += [f"### {cur.capitalize()}", ""]
        out += [
            f"**Q{s['index']}:** {s['question']}",
            f"- **Reference:** {s['reference_answer']}",
            f"- **Generated:** {s['generated_answer']}",
            (f"- _ttft={s['ttft_ms']}ms, latency={s['latency_ms']}ms, "
             f"cache_ratio={s['cached_prompt_ratio']}, finish={s['finish_reason']}_"
             + (f"  **ERROR:** {s['error']}" if s["error"] else "")),
            "",
        ]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract begin/middle/end Q/A evidence per baseline")
    ap.add_argument("--results-dir", default="analysis/phase1/results")
    ap.add_argument("--out", default="evidence")
    ap.add_argument("--n", type=int, default=10, help="samples per window (begin/middle/end)")
    args = ap.parse_args()

    rd = Path(args.results_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not rd.exists():
        print(f"results dir not found: {rd}", file=sys.stderr)
        return 1

    all_md = ["# CAGE Q/A Evidence", "",
              "Concrete question/answer samples captured per baseline, for dissertation evidence.", ""]
    index = {}
    for sub in sorted(p for p in rd.iterdir() if p.is_dir()):
        csvs = sorted(sub.glob("*results.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not csvs:
            continue
        rows, samples = extract(csvs[0], args.n)
        (out / f"{sub.name}_qa.json").write_text(
            json.dumps({"baseline": sub.name, "total": len(rows), "samples": samples}, indent=2))
        all_md.append(md_for(sub.name, len(rows), samples))
        index[sub.name] = {"total": len(rows), "samples": len(samples)}

    (out / "qa_evidence.md").write_text("\n".join(all_md))
    (out / "qa_evidence_index.json").write_text(json.dumps(index, indent=2))
    print(f"Wrote Q/A evidence for {len(index)} baselines -> {out}/qa_evidence.md (+ per-baseline JSON)")
    for b, v in index.items():
        print(f"  {b}: {v['samples']} samples of {v['total']} queries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
