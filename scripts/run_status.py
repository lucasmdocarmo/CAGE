#!/usr/bin/env python3
"""
CAGE per-baseline run status tracker.

Reports, for every baseline in a run, an at-a-glance status:
    NOT_STARTED | RUNNING | FINISHED | ERROR
with the explicit flags the dissertation tracking asks for
(started / isRunning / isFinished / hasErrors), plus timestamps, #results,
#errors, and headline metrics (grounding / faithfulness / TTFT).

It is the continuous live-tracking tool for Phase 2 (single GPU) and Phase 3
(distributed). It reads:
  * the on-disk results tree   -> authoritative for FINISHED + metrics
  * (optionally) the run log    -> RUNNING / started-at / finished-at / errors

Usage:
  python scripts/run_status.py                                  # one-shot table
  python scripts/run_status.py --watch 10                       # refresh every 10s
  python scripts/run_status.py --json                           # machine-readable
  python scripts/run_status.py --results-dir analysis/phase1/results --run-log run.log
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
ERR_PAT = re.compile(r"Traceback|Error:|Exception|CUDA out of memory|failed to start|OutOfMemory", re.I)


def parse_run_log(run_log: Path) -> dict:
    """Per-baseline {started_at, finished_at, errors[], running} parsed from the run log."""
    info: dict = {}
    if not run_log or not Path(run_log).exists():
        return info
    cur = None
    text = ANSI.sub("", Path(run_log).read_text(errors="replace"))
    for line in text.splitlines():
        m = re.search(r">>> Running baseline:\s*(\S+)", line)
        if m:
            cur = m.group(1)
            info.setdefault(cur, {"started_at": None, "finished_at": None, "errors": [], "running": True})
            info[cur]["running"] = True
            continue
        if not cur:
            continue
        ms = re.search(r"Started at:\s*(.+)$", line)
        if ms and not info[cur]["started_at"]:
            info[cur]["started_at"] = ms.group(1).strip()
        mf = re.search(r"Finished at:\s*(.+)$", line)
        if mf:
            info[cur]["finished_at"] = mf.group(1).strip()
            info[cur]["running"] = False
        if ERR_PAT.search(line):
            info[cur]["errors"].append(line.strip()[:200])
    return info


def scan_results(results_dir: Path) -> dict:
    """Per-baseline filesystem facts: metrics present, #results, #errors, headline metrics."""
    out: dict = {}
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return out
    for sub in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        rec = {"has_metrics": False, "num_results": 0, "num_errors": 0, "quality": {}, "perf": {}}
        mj = sub / "metrics.json"
        if mj.exists():
            rec["has_metrics"] = True
            try:
                d = json.loads(mj.read_text())
                q = d.get("quality") or {}
                p = d.get("performance") or {}
                rec["quality"] = {k: q.get(k) for k in ("grounding_score", "faithfulness") if k in q}
                rec["perf"] = {k: p.get(k) for k in ("avg_ttft_ms", "tokens_per_second", "error_count") if k in p}
                rec["num_errors"] = int(p.get("error_count") or 0)
            except Exception:
                pass
        csvs = sorted(sub.glob("*results.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
        if csvs:
            try:
                with open(csvs[0]) as f:
                    rows = list(csv.DictReader(f))
                rec["num_results"] = len(rows)
                rec["num_errors"] = max(rec["num_errors"], sum(1 for r in rows if (r.get("error") or "").strip()))
            except Exception:
                pass
        out[sub.name] = rec
    return out


def derive(li: dict, ri: dict) -> dict:
    has_errors = bool(li.get("errors")) or (ri.get("num_errors", 0) > 0)
    is_finished = bool(ri.get("has_metrics"))
    started = bool(li.get("started_at")) or is_finished or ri.get("num_results", 0) > 0
    is_running = started and not is_finished
    if is_finished:
        status = "ERROR" if has_errors else "FINISHED"
    elif is_running:
        status = "ERROR" if has_errors else "RUNNING"
    else:
        status = "NOT_STARTED"
    return {"status": status, "started": started, "isRunning": is_running,
            "isFinished": is_finished, "hasErrors": has_errors}


def build(results_dir: Path, run_log: Path) -> list:
    log_info = parse_run_log(run_log)
    res_info = scan_results(results_dir)
    rows = []
    for b in sorted(set(log_info) | set(res_info)):
        li, ri = log_info.get(b, {}), res_info.get(b, {})
        flags = derive(li, ri)
        rows.append({
            "baseline": b, **flags,
            "started_at": li.get("started_at"),
            "finished_at": li.get("finished_at"),
            "num_results": ri.get("num_results", 0),
            "num_errors": (ri.get("num_errors", 0) or 0) + len(li.get("errors", [])),
            "grounding": ri.get("quality", {}).get("grounding_score"),
            "faithfulness": ri.get("quality", {}).get("faithfulness"),
            "avg_ttft_ms": ri.get("perf", {}).get("avg_ttft_ms"),
            "error_samples": li.get("errors", [])[:2],
        })
    return rows


_ICON = {"NOT_STARTED": "○", "RUNNING": "◐", "FINISHED": "✔", "ERROR": "✗"}


def fmt_table(rows: list) -> str:
    if not rows:
        return "(no baselines found yet)"
    def g(v, nd=3):
        return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "-"
    head = f"{'BASELINE':<32} {'STATUS':<12} {'#res':>5} {'#err':>5} {'ground':>7} {'faith':>7} {'ttft_ms':>8}"
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['baseline']:<32} {_ICON.get(r['status'],' ')+' '+r['status']:<12} "
            f"{r['num_results']:>5} {r['num_errors']:>5} {g(r['grounding']):>7} "
            f"{g(r['faithfulness']):>7} {g(r['avg_ttft_ms'],1):>8}"
        )
    done = sum(r["isFinished"] for r in rows)
    run = sum(r["isRunning"] for r in rows)
    err = sum(r["hasErrors"] for r in rows)
    lines.append("-" * len(head))
    lines.append(f"{len(rows)} baselines | {done} finished | {run} running | {err} with errors")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="CAGE per-baseline run status tracker")
    ap.add_argument("--results-dir", default="analysis/phase1/results")
    ap.add_argument("--run-log", default="run.log")
    ap.add_argument("--watch", type=float, default=0, help="refresh every N seconds (0 = one-shot)")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args()

    def render():
        rows = build(Path(args.results_dir), Path(args.run_log))
        if args.json:
            return json.dumps({"baselines": rows}, indent=2)
        return fmt_table(rows)

    if args.watch and args.watch > 0:
        try:
            while True:
                os.system("clear" if os.name == "posix" else "cls")
                print(f"CAGE run status  (results={args.results_dir})\n")
                print(render())
                time.sleep(args.watch)
        except KeyboardInterrupt:
            return 0
    else:
        print(render())
    return 0


if __name__ == "__main__":
    sys.exit(main())
