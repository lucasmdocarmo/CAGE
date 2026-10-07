#!/usr/bin/env python3
"""Order:     stage 13 (analyze) of scripts/6_experiments/cage_experiment.sh, after verify_results, organize_results, run_campaign_analysis and render_window_panels have written their artifacts into the pulled run tree
Objective: Write experiments/<S>/<date>/plots/analysis.txt: the mechanical sections (header, what ran, technical and analytic readings, limitations, artifact index) from the run's own artifacts, plus the academic-reading SKELETON the main session fills in (ADR-0143)
Cloud:     local

Every number in the file names the artifact it came from. The script never
writes interpretive prose: section 5 is headings, contrast ids, artifact
paths and the sentence "to be written by the main session after reading the
run" (design section 7; the epistemic rules forbid untagged interpretation).

Usage:
  write_analysis.py --landing experiments/<S>/<date> --run-root <landing>/run/<c>/<s>/<run_id> --out <landing>/plots/analysis.txt

Inputs (each optional; an absent artifact is named as absent, never guessed):
  <landing>/extras/state.json            experiment, pod, run id, GO words, cost
  <landing>/extras/profile.env           the frozen profile
  <landing>/extras/plan.json             the plan (schema, steps, blocked cells)
  <landing>/extras/verify/*              verify_results report (FAIL/WARN lines)
  <run_root>/index/cells_index.csv       one row per (cell, window)
  <run_root>/index/coverage_report.md    MISSING list (annotated as floor-based)
  <run_root>/analysis/<stamp>/stats.json and summary.md (newest stamp)
  <run_root>/observability/serving_configs/*.json   realized KV pools (S0F-24)
Exit 0 with the file written; 2 on a usage error or an unwritable output.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

SKELETON_SENTENCE = "to be written by the main session after reading the run"


def _read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def _newest_stamp(run_root: Path) -> Optional[Path]:
    base = run_root / "analysis"
    if not base.is_dir():
        return None
    stamps = sorted(p for p in base.iterdir() if p.is_dir() and (p / "stats.json").is_file())
    return stamps[-1] if stamps else None


def _rel(path: Path, landing: Path) -> str:
    try:
        return str(path.relative_to(landing))
    except ValueError:
        return str(path)


def _raw(value: Any) -> str:
    return value[:120] if isinstance(value, str) else "?"


def _is_fail_line(line: str) -> bool:
    """A verdict line, not a summary count ("0 FAIL, 1 WARN" is a summary)."""
    return bool(re.search(r"\bFAIL\b", line)) and not re.search(r"\b\d+\s+FAIL\b", line)


def _is_warn_line(line: str) -> bool:
    return bool(re.search(r"\bWARN", line)) and not re.search(r"\b\d+\s+WARN", line)


def _verify_lines(landing: Path) -> List[str]:
    verify_dirs = sorted(landing.glob("extras/verify*"))
    lines: List[str] = []
    for d in verify_dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.suffix in (".md", ".txt", ".json", ".log"):
                lines += [l.strip() for l in (_read_text(p) or "").splitlines() if _is_fail_line(l) or _is_warn_line(l)]
    return lines


def section_header(landing: Path, run_root: Path, state: Dict[str, Any], out: List[str]) -> None:
    pod = state.get("pod", {}) or {}
    cost = state.get("cost", {}) or {}
    out.append("1. HEADER")
    out.append(f"experiment: {state.get('experiment', '?')}   date: {state.get('date_utc', '?')}   run id: {state.get('run_id', '?')}")
    out.append(f"pod: id={pod.get('id', '?')} dc={pod.get('dc', '?')} price_per_hour_usd={pod.get('price_per_hour_usd', '?')} created={pod.get('created_utc', '?')} deleted={pod.get('deleted_utc', '?')}")
    out.append(f"build sha: {(state.get('build') or {}).get('sha', '?')}   profile: extras/profile.env   state: extras/state.json")
    out.append(f"cost: balance before (raw)={_raw(cost.get('balance_before_raw'))}")
    out.append(f"      balance after (raw)={_raw(cost.get('balance_after_raw'))}   TRUE $0 at: {cost.get('true_zero_utc', 'not recorded')}")
    gos = state.get("go", []) or []
    out.append("owner GO: " + ("; ".join(f"{g.get('stage')} at {g.get('instant_utc')} ({g.get('words')})" for g in gos) if gos else "none recorded"))
    stages = state.get("stages", {}) or {}
    out.append(("stages: " + ", ".join(f"{n}={s.get('status')}" for n, s in stages.items())) if stages else "stages: none recorded")
    verify_dirs = [d for d in sorted(landing.glob("extras/verify*")) if d.is_dir()]
    if verify_dirs:
        lines = _verify_lines(landing)
        fails = [l for l in lines if _is_fail_line(l)]
        warns = [l for l in lines if _is_warn_line(l) and not _is_fail_line(l)]
        out.append(f"verify_results: {len(fails)} FAIL line(s), {len(warns)} WARN line(s) (source: {', '.join(_rel(d, landing) + '/' for d in verify_dirs)})")
    else:
        out.append("verify_results: absent (extras/verify*/ not found)")
    cov = run_root / "index" / "coverage_report.md"
    cov_text = _read_text(cov)
    if cov_text is not None:
        missing = [l for l in cov_text.splitlines() if "MISSING" in l]
        out.append(f"organize coverage: {len(missing)} MISSING line(s) in {_rel(cov, landing)} (organize_results.py lists the charter F1 arm floor plus manifest.expected_cells when declared; the plan this run executed is extras/plan.json)")
    else:
        out.append("organize coverage: absent (index/coverage_report.md not found)")
    stamp = _newest_stamp(run_root)
    if stamp is not None:
        stats = _read_json(stamp / "stats.json") or {}
        figures = stats.get("figures", []) if isinstance(stats, dict) else []
        mode = stats.get("mode") or stats.get("analysis_mode") or "see stats.json"
        out.append(f"analysis: stamp {stamp.name}, mode {mode}, {len(figures)} figure record(s) (source: {_rel(stamp / 'stats.json', landing)})")
    else:
        out.append("analysis: absent (no analysis/<stamp>/stats.json)")
    out.append("")


def section_what_ran(landing: Path, run_root: Path, plan: Optional[Dict[str, Any]], out: List[str]) -> None:
    out.append("2. WHAT RAN")
    idx = run_root / "index" / "cells_index.csv"
    rows: List[Dict[str, str]] = []
    if idx.is_file():
        with idx.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    if not rows:
        out.append(f"cells index: absent or empty ({_rel(idx, landing)})")
    else:
        cols = list(rows[0].keys())
        key_cols = [c for c in ("engine", "model", "family", "topology", "arm", "baseline", "baseline_label", "dataset") if c in cols]
        cell_col = "row_key" if "row_key" in cols else None
        windows_per_cell: Dict[str, int] = Counter()
        label_per_cell: Dict[str, str] = {}
        for r in rows:
            cell = r.get(cell_col, "?") if cell_col else "|".join(r.get(c, "?") for c in key_cols)
            windows_per_cell[cell] += 1
            label_per_cell[cell] = " ".join(f"{c}={r.get(c, '?')}" for c in key_cols)
        out.append(f"rows (cell x window): {len(rows)}   cells: {len(windows_per_cell)}   source: {_rel(idx, landing)} (columns: {', '.join(cols)})")
        for cell in sorted(windows_per_cell):
            out.append(f"  {cell}: {windows_per_cell[cell]} window(s)   {label_per_cell[cell]}")
        for c in ("engine", "dataset", "family"):
            if c in cols:
                out.append(f"  by {c}: " + ", ".join(f"{k}={v}" for k, v in sorted(Counter(r.get(c, '?') for r in rows).items())))
    if plan:
        steps = [s for s in (plan.get("steps", []) if isinstance(plan, dict) else []) if isinstance(s, dict)]
        cells = [s for s in steps if s.get("kind") != "relaunch"]
        out.append(f"plan: schema {plan.get('schema', '?')}, {len(steps)} steps ({len(cells)} cells, {len(steps) - len(cells)} relaunches), {len(plan.get('blocked_row_keys', []) or [])} blocked (source: extras/plan.json)")
    else:
        out.append("plan: absent (extras/plan.json not found)")
    out.append("")


def section_technical(landing: Path, run_root: Path, out: List[str]) -> None:
    out.append("3. TECHNICAL READING (facts, every line with its source)")
    stamp = _newest_stamp(run_root)
    if stamp is None:
        out.append("stats.json: absent")
    else:
        stats = _read_json(stamp / "stats.json") or {}
        contrasts = stats.get("contrasts", []) if isinstance(stats, dict) else []
        out.append(f"contrast rows in stats.json: {len(contrasts)} (source: {_rel(stamp / 'stats.json', landing)})")
        for c in contrasts[:60]:
            if not isinstance(c, dict):
                continue
            keys = [k for k in ("id", "contrast_id", "metric", "dataset", "reference_row_key", "cell_row_key", "n", "estimate", "effect", "p_value", "ci_low", "ci_high", "verdict") if k in c]
            out.append("  " + "  ".join(f"{k}={c[k]}" for k in keys) if keys else "  " + json.dumps(c)[:200])
        if len(contrasts) > 60:
            out.append(f"  ... {len(contrasts) - 60} more rows in stats.json")
    sc_dir = run_root / "observability" / "serving_configs"
    if sc_dir.is_dir():
        found = 0
        for p in sorted(sc_dir.glob("*.json")):
            d = _read_json(p) or {}
            if isinstance(d, dict) and ("kv_pool_bytes_realized" in d or "gpu_memory_utilization" in d):
                found += 1
                out.append(f"  serving config {p.name}: engine={d.get('engine', '?')} requested_util={d.get('gpu_memory_utilization', d.get('mem_util_requested', '?'))} kv_pool_bytes_realized={d.get('kv_pool_bytes_realized', 'absent')} tokens={d.get('kv_pool_tokens_realized', 'absent')} captured={d.get('kv_pool_captured_utc', 'absent')}")
        if found == 0:
            out.append(f"  serving configs: none with a pool record under {_rel(sc_dir, landing)}")
    else:
        out.append("  serving configs: absent (observability/serving_configs/ not found)")
    lines = _verify_lines(landing)
    out.append(f"verify_results FAIL/WARN lines ({len(lines)}):")
    out.extend("  " + l for l in lines[:40])
    # the predicate step: how many answers in each window got NO grade (owner,
    # 2026-10-06: the bound is provisional at 1.0 for the dry run, so the
    # per-window ungraded share is printed here instead of relying on the stop)
    manifests = sorted(run_root.glob("predicate/*/predicate_manifest.json"))
    if not manifests:
        out.append("predicate: absent (no predicate/<scoring_run_id>/predicate_manifest.json)")
    for mp in manifests:
        m = _read_json(mp)
        if not isinstance(m, dict):
            out.append(f"predicate {mp.parent.name}: unreadable manifest ({_rel(mp, landing)})")
            continue
        cfg = m.get("config", {}) if isinstance(m.get("config"), dict) else {}
        counts = m.get("counts", {}) if isinstance(m.get("counts"), dict) else {}
        out.append(f"predicate {mp.parent.name}: bound max_null_fraction={cfg.get('max_null_fraction', '?')} (1.0 = the stop never fires), "
                   f"rows={counts.get('n_rows', '?')} graded true={counts.get('n_true', '?')} false={counts.get('n_false', '?')} "
                   f"ungraded={counts.get('n_null', '?')} ({counts.get('null_fraction', '?')}) windows={counts.get('n_windows', '?')} "
                   f"skipped={len(m.get('skipped_windows', []) or [])} (source: {_rel(mp, landing)})")
        rows = [w for w in (m.get("per_window") or []) if isinstance(w, dict)]
        for w in sorted(rows, key=lambda w: -(w.get("null_fraction") or 0)):
            nf = w.get("null_fraction")
            share = f"{100 * nf:.1f}%" if isinstance(nf, (int, float)) else "?"
            flag = "   <- more than half of this window has no grade" if isinstance(nf, (int, float)) and nf > 0.5 else ""
            out.append(f"  window {w.get('window', '?')}: ungraded {w.get('n_null', '?')} of {w.get('n_rows', '?')} ({share}); graded true={w.get('n_true', '?')} false={w.get('n_false', '?')}{flag}")
    out.append("")


def section_analytic(landing: Path, run_root: Path, out: List[str]) -> None:
    out.append("4. ANALYTIC READING (the driver's own words; no number here is outside an artifact)")
    stamp = _newest_stamp(run_root)
    summary = _read_text(stamp / "summary.md") if stamp else None
    if summary is None:
        out.append("summary.md: absent")
    else:
        out.append(f"source: {_rel(stamp / 'summary.md', landing)}")
        out.extend("  " + l for l in summary.splitlines()[:200])
    out.append("")


def section_academic(landing: Path, run_root: Path, out: List[str]) -> None:
    out.append("5. ACADEMIC READING (SKELETON)")
    stamp = _newest_stamp(run_root)
    ids: List[str] = []
    if stamp:
        stats = _read_json(stamp / "stats.json") or {}
        for c in stats.get("contrasts", []) if isinstance(stats, dict) else []:
            if isinstance(c, dict):
                cid = c.get("id", c.get("contrast_id"))
                if cid is not None and str(cid) not in ids:
                    ids.append(str(cid))
    out.append(f"registered contrasts present in this run: {', '.join(ids) if ids else 'none'}   (charter: MyDocs/Publication/PUBLICATION.md section 7.8; hypotheses D6 section 6)")
    for cid in ids or ["(none)"]:
        out.append(f"  contrast {cid}: the run is consistent with / is not consistent with / cannot speak to: {SKELETON_SENTENCE}")
    out.append(f"  S1 yield, knee and cliff (charter section 6): {SKELETON_SENTENCE}")
    out.append(f"  artifacts to read first: {_rel(stamp / 'stats.json', landing) if stamp else 'stats.json absent'}, {_rel(stamp / 'summary.md', landing) if stamp else 'summary.md absent'}, extras/verify/, {_rel(run_root / 'index' / 'coverage_report.md', landing)}")
    out.append(f"  every sentence written here carries a [V]/[D]/[A] tag; the script wrote none: {SKELETON_SENTENCE}")
    out.append("")


def section_limitations(landing: Path, run_root: Path, state: Dict[str, Any], out: List[str]) -> None:
    out.append("6. LIMITATIONS")
    idx = run_root / "index" / "cells_index.csv"
    if idx.is_file():
        with idx.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        for c in ("family", "engine"):
            if rows and c in rows[0]:
                out.append(f"  n (rows) per {c}: " + ", ".join(f"{k}={v}" for k, v in sorted(Counter(r.get(c, '?') for r in rows).items())))
    out.append("  single pod, one date folder; no replication across pods or days inside this run.")
    out.append("  organize MISSING lines are floor-based (see section 1); mini manifests, where used, bound n per row class.")
    notes = []
    for n, s in (state.get("stages", {}) or {}).items():
        for t in s.get("notes", []) or []:
            notes.append(f"{n}: {t}")
    out.append("  stage notes: " + ("; ".join(notes) if notes else "none"))
    out.append("  the charter figures no script renders stay unrendered (design section 15 item 6).")
    out.append("  assumptions carried by the profile are tagged [A] in extras/profile.env.")
    out.append("")


def section_index(landing: Path, out: List[str]) -> None:
    out.append("7. INDEX OF ARTIFACTS (relative to the date folder; size in bytes)")
    count = 0
    for p in sorted(landing.rglob("*")):
        try:
            if not p.is_file() or p.name.startswith("."):
                continue
            size = p.stat().st_size
        except OSError:
            continue  # a broken symlink or a file that vanished mid-walk
        count += 1
        if count <= 2000:
            out.append(f"  {_rel(p, landing)}  {size}")
    if count > 2000:
        out.append(f"  ... {count - 2000} more files")
    out.append(f"  total files: {count}")


def build(landing: Path, run_root: Path) -> str:
    state = _read_json(landing / "extras" / "state.json") or {}
    plan = _read_json(landing / "extras" / "plan.json")
    out: List[str] = []
    out.append(f"CAGE experiment analysis file: {landing.name} ({landing.parent.name})")
    out.append("Written by scripts/6_experiments/write_analysis.py from the run's artifacts; sections 1 to 4, 6 and 7 are mechanical; section 5 is a skeleton.")
    out.append("")
    state_d = state if isinstance(state, dict) else {}
    section_header(landing, run_root, state_d, out)
    section_what_ran(landing, run_root, plan if isinstance(plan, dict) else None, out)
    section_technical(landing, run_root, out)
    section_analytic(landing, run_root, out)
    section_academic(landing, run_root, out)
    section_limitations(landing, run_root, state_d, out)
    section_index(landing, out)
    return "\n".join(out) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Write the experiment's analysis.txt from its artifacts (ADR-0143).")
    p.add_argument("--landing", required=True, help="experiments/<S>/<date>")
    p.add_argument("--run-root", required=True, help="<landing>/run/<campaign>/<session>/<run_id>")
    p.add_argument("--out", required=True, help="the analysis.txt to write")
    a = p.parse_args(argv)
    landing, run_root, out = Path(a.landing), Path(a.run_root), Path(a.out)
    if not landing.is_dir():
        print(f"REFUSED: landing dir not found: {landing}", file=sys.stderr)
        return 2
    text = build(landing, run_root)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, out)
    except OSError as exc:
        print(f"REFUSED: cannot write {out}: {exc}", file=sys.stderr)
        return 2
    print(f"[write_analysis] wrote {out} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
