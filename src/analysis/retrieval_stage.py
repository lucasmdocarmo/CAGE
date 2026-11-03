"""Order:     stage 14 (analyze) of cage_experiment.sh, after the yield table (S2 block F row F-7, 2026-11-03)
Objective: the retrieval-stage sidebar of a landing, from the sealed request rows alone
Cloud:     local
           (the Mac; reads the pulled landing; standard library only)

Purpose: measure, for every landing, whether the gold passage reached the served prompt,
per dataset, baseline and engine, over the no-load reference windows (family F1) of the
retrieval arms, and pair B5 (dense) against B6 (reranked) per question.

Math: the gold-in-prompt rate is k / n over the first pass of every ok request of the
window, hit when ``gold_position_in_prompt`` is not None and at least 0. Its 95 percent
interval is the Wilson score interval, center (p + z^2 / 2n) / (1 + z^2 / n) and half
width z * sqrt(p (1 - p) / n + z^2 / 4 n^2) / (1 + z^2 / n), z = 1.959964 (the S1 hand
script's value, experiments/S1/2026-10-23_2350/plots/retrieval_stage_20261030/
retrieval_stage.py), clamped to [0, 1]. The paired contrast is the exact McNemar test on
the discordant pairs b (B5 only) and c (B6 only): p = min(1, 2 * P[X <= min(b, c)]) with
X ~ Binomial(b + c, 1/2), and p = 1 when b + c = 0. The S1 numbers this module must
reproduce are in the S1 report section 14 (MyDocs/RunPod/S1_RESULTS_REPORT_2026-10-27.md).

Row rule (the S1 rule, stated here so a reader can recompute): ok rows only; the first
row per (example_id, repeat_index as text, "0" when absent) counts; a row whose position
is None is a miss in the numerator and stays in the denominator, and ``n_none`` reports
how many; ``retrieval_hits`` counts rows whose ``retrieval_hit`` is at least 1 (the gold
passage was among the retrieved passages) and ``lost`` those retrieved and then absent
from the prompt (the truncated arm cuts them); a row without ``example_id`` is skipped
and counted (``n_rows_without_example_id``). Windows: family F1 of a cell whose arm
starts with ``retr-``; the baseline id is the cell record's (cells/<row_key>/cell.json).

Exit codes of ``main``: 0 the two files written; 3 the named refusal (no F1 window of a
retrieval arm), nothing written; 2 a missing or unreadable requests.jsonl, or a line of
it that is not a JSON object, named.
Complexity: O(R) over the rows read, O(Q) memory for the per-question maps.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from math import comb
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

#: = src.orchestration.campaign_layout.WINDOW_DIR_RE (that module imports pandas; this one
#: stays standard library, as scripts/6_experiments/regime_gate.py does; pinned by a test).
WINDOW_DIR_RE = re.compile(r"^window_([a-z0-9_]+)-(\d+)$")
RETRIEVAL_ARM_PREFIX = "retr-"
REFERENCE_FAMILY = "F1"
ROW_KEY_PARTS = 7
Z_95 = 1.959964
PAIRED_BASELINES = ("B5", "B6")
REFUSED_RC = 3
RULE = (
    "F1 windows of the retrieval arms (row key arm 'retr-*'), ok rows, first pass per "
    "(example_id, repeat_index); hit = gold_position_in_prompt >= 0; a None position is a "
    "miss kept in the denominator (n_none); Wilson 95 percent, z = 1.959964; paired B5 vs "
    "B6 per question, exact McNemar"
)


class RetrievalStageRefusal(Exception):
    """The named refusal of the whole pass (nothing to measure)."""


class RetrievalStageInputError(Exception):
    """A requests.jsonl the pass needs is missing or unreadable (names the file)."""


def wilson(k: int, n: int, z: float = Z_95) -> Tuple[float, float]:
    """The Wilson score interval of k hits in n, clamped to [0, 1]; n must be positive."""
    if n <= 0:
        raise ValueError(f"wilson needs n >= 1, got {n}")
    p = k / n
    d = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / d
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p on the discordant counts b and c (1.0 when both are 0)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2.0 * p)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _rows(path: Path) -> Iterable[Dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RetrievalStageInputError(f"{path}: line {i} is not JSON ({exc.msg})") from exc
                if not isinstance(row, dict):
                    raise RetrievalStageInputError(f"{path}: line {i} is not a JSON object")
                yield row
    except OSError as exc:
        raise RetrievalStageInputError(f"{path}: {exc.strerror or exc}") from exc


def collect(run_root: Path) -> Dict[str, Any]:
    """Read every F1 window of every retrieval-arm cell under run_root/cells.

    Returns ``{"groups": {(dataset, baseline, engine): {qkey: (hit, retrieved, none)}},
    "skipped_cells": [{"cell", "reason"}], "n_windows": int,
    "n_rows_without_example_id": int}``. A cell directory whose row key has fewer than
    seven parts, or that carries no cell.json with a baseline, is skipped and listed; a
    missing or unreadable requests.jsonl, or a line of it that is not an object, raises;
    an ok row without an example_id is skipped and counted.
    """
    groups: Dict[Tuple[str, str, str], Dict[Tuple[str, str], Tuple[bool, bool, bool]]] = {}
    skipped: List[Dict[str, str]] = []
    n_windows = 0
    n_no_id = 0
    cells = run_root / "cells"
    for cdir in sorted(p for p in cells.iterdir() if p.is_dir()) if cells.is_dir() else []:
        parts = cdir.name.split("|")
        if not parts[0].startswith(RETRIEVAL_ARM_PREFIX):
            continue
        if len(parts) < ROW_KEY_PARTS:
            skipped.append({"cell": cdir.name, "reason": f"row key has fewer than seven parts ({len(parts)})"})
            continue
        if parts[6] != REFERENCE_FAMILY:
            continue
        engine = parts[4]
        baseline: Any = None
        try:
            record = json.loads((cdir / "cell.json").read_text(encoding="utf-8"))
            baseline = record.get("baseline") if isinstance(record, dict) else None
        except (OSError, ValueError):
            baseline = None
        if not isinstance(baseline, str) or not baseline:
            skipped.append({"cell": cdir.name, "reason": "no baseline id in cell.json"})
            continue
        for wdir in sorted(p for p in cdir.iterdir() if p.is_dir()):
            m = WINDOW_DIR_RE.match(wdir.name)
            if not m:
                continue
            n_windows += 1
            questions = groups.setdefault((m.group(1), baseline, engine), {})
            for r in _rows(wdir / "requests.jsonl"):
                if not r.get("ok"):
                    continue
                if r.get("example_id") is None:
                    n_no_id += 1
                    continue
                qkey = (str(r["example_id"]), str(r.get("repeat_index") or "0"))
                if qkey in questions:
                    continue
                g = r.get("gold_position_in_prompt")
                hit = _is_number(g) and g >= 0
                retrieved = _is_number(r.get("retrieval_hit")) and r["retrieval_hit"] >= 1
                questions[qkey] = (hit, retrieved, g is None)
    return {"groups": groups, "skipped_cells": skipped, "n_windows": n_windows,
            "n_rows_without_example_id": n_no_id}


def rates_table(collected: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One row per (dataset, baseline, engine) with a positive n, sorted by that key."""
    out: List[Dict[str, Any]] = []
    for (dataset, baseline, engine), questions in sorted(collected["groups"].items()):
        n = len(questions)
        if n == 0:
            continue
        hits = sum(1 for h, _, _ in questions.values() if h)
        lo, hi = wilson(hits, n)
        out.append({
            "dataset": dataset, "baseline": baseline, "engine": engine,
            "n": n, "n_none": sum(1 for _, _, none in questions.values() if none),
            "hits": hits, "rate": hits / n, "wilson_low": lo, "wilson_high": hi,
            "retrieval_hits": sum(1 for _, r, _ in questions.values() if r),
            "lost": sum(1 for h, r, _ in questions.values() if r and not h),
        })
    return out


def paired_table(collected: Dict[str, Any]) -> List[Dict[str, Any]]:
    """B5 against B6 per (dataset, engine) over the questions present in both."""
    groups = collected["groups"]
    out: List[Dict[str, Any]] = []
    pairs = sorted({(d, e) for (d, _, e) in groups})
    for dataset, engine in pairs:
        a = groups.get((dataset, PAIRED_BASELINES[0], engine))
        b = groups.get((dataset, PAIRED_BASELINES[1], engine))
        if not a or not b:
            continue
        common = [q for q in a if q in b]
        if not common:
            continue
        b5_only = sum(1 for q in common if a[q][0] and not b[q][0])
        b6_only = sum(1 for q in common if b[q][0] and not a[q][0])
        both = sum(1 for q in common if a[q][0] and b[q][0])
        out.append({
            "dataset": dataset, "engine": engine, "n_pairs": len(common),
            "b5_only": b5_only, "b6_only": b6_only, "both": both,
            "neither": len(common) - b5_only - b6_only - both,
            "delta": (b6_only - b5_only) / len(common),
            "mcnemar_p": mcnemar_exact(b5_only, b6_only),
        })
    return out


def render_markdown(doc: Dict[str, Any]) -> str:
    lines = [
        "# Retrieval stage (S2 block F row F-7)", "",
        f"Run root: {doc['run_root']}", f"Rule: {doc['rule']}", f"Windows read: {doc['n_windows']}",
        f"Rows without example_id (skipped): {doc['n_rows_without_example_id']}", "",
        "## Gold-in-prompt rate per dataset, baseline and engine", "",
        "| dataset | baseline | engine | n | n_none | hits | rate | Wilson 95% low | Wilson 95% high | retrieval hits | lost |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for t in doc["rates"]:
        lines.append(
            f"| {t['dataset']} | {t['baseline']} | {t['engine']} | {t['n']} | {t['n_none']} | {t['hits']} | "
            f"{t['rate']:.3f} | {t['wilson_low']:.3f} | {t['wilson_high']:.3f} | {t['retrieval_hits']} | {t['lost']} |"
        )
    lines += ["", "## Paired B5 against B6 per question, same engine (questions present in both)", "",
              "| dataset | engine | n pairs | B5 only | B6 only | both | neither | delta (B6 minus B5) | McNemar exact p |",
              "|---|---|---|---|---|---|---|---|---|"]
    for p in doc["paired"]:
        lines.append(
            f"| {p['dataset']} | {p['engine']} | {p['n_pairs']} | {p['b5_only']} | {p['b6_only']} | {p['both']} | "
            f"{p['neither']} | {p['delta']:+.3f} | {p['mcnemar_p']:.2e} |"
        )
    if not doc["paired"]:
        lines.append("| (no dataset and engine with both B5 and B6) | | | | | | | | |")
    lines += ["", "## Skipped cells", ""]
    lines += [f"- {s['cell']}: {s['reason']}" for s in doc["skipped_cells"]] or ["- none"]
    return "\n".join(lines) + "\n"


def build(run_root: Path) -> Dict[str, Any]:
    collected = collect(run_root)
    rates = rates_table(collected)
    if not rates:
        raise RetrievalStageRefusal(
            f"no F1 window of a retrieval arm with an ok row under {run_root / 'cells'} "
            f"(windows read: {collected['n_windows']}, cells skipped: {len(collected['skipped_cells'])})"
        )
    return {
        "run_root": str(run_root), "rule": RULE, "n_windows": collected["n_windows"],
        "n_rows_without_example_id": collected["n_rows_without_example_id"], "rates": rates, "paired": paired_table(collected), "skipped_cells": collected["skipped_cells"],
    }


def write_outputs(doc: Dict[str, Any], out_dir: Path) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path, json_path = out_dir / "retrieval_stage.md", out_dir / "retrieval_stage.json"
    for path, text in ((md_path, render_markdown(doc)), (json_path, json.dumps(doc, indent=2, sort_keys=True) + "\n")):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    return md_path, json_path


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="the retrieval-stage sidebar of a pulled landing (row F-7)")
    ap.add_argument("run_root", help="campaign run root (holds cells/)")
    ap.add_argument("--out", required=True, help="output directory for retrieval_stage.md and .json")
    args = ap.parse_args(list(argv) if argv is not None else None)
    try:
        doc = build(Path(args.run_root))
    except RetrievalStageInputError as exc:
        print(f"[retrieval_stage] ERROR: {exc}", file=sys.stderr)
        return 2
    except RetrievalStageRefusal as exc:
        print(f"[retrieval_stage] REFUSED: {exc}", file=sys.stderr)
        return REFUSED_RC
    md_path, json_path = write_outputs(doc, Path(args.out))
    print(f"[retrieval_stage] groups={len(doc['rates'])} pairs={len(doc['paired'])} "
          f"skipped_cells={len(doc['skipped_cells'])} -> {md_path} {json_path}")
    return 0
