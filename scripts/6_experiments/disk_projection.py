#!/usr/bin/env python3
"""Order:     stage 6 (plan) of scripts/6_experiments/cage_experiment.sh, right after the plan is fetched
Objective: Project the bytes the run will write from the plan's arrivals and refuse a run that
           would fill the pod disk holding the run root and its mirror, or the Mac disk the
           pull lands on (audit 2026-10-17, G7)
Cloud:     local
           (the Mac; reads extras/plan.json)

Usage:
  python3 scripts/6_experiments/disk_projection.py PLAN_JSON --pod-gb N --mac-free-gb N
          [--copies 2] [--bytes-per-arrival 37000] [--headroom 0.8]

Exit 0 when both projections fit under ``headroom`` x the bound, 1 when either does
not (the figures are printed either way), 2 on a usage error.

Arrivals: every executable cell step contributes ``windows x window_arrivals``
(a pressure window) or ``windows x num_queries`` (any other cell); a dry window
contributes ceil(duration_s x rate) from its argv, or DRY_WINDOW_ARRIVALS_ASSUMED
when either is absent; blocked steps and relaunches contribute nothing. Bytes per
arrival: S0 attempt 5 wrote about 37 KB per arrival over requests.jsonl,
cage_stats.jsonl, qa_evidence.jsonl and engine_metrics.json (fix list G7) [V];
``--copies`` is 2 for the run root and its backup mirror, 3 with a tar archive
(BACKUP_ARCHIVE_DIR). The Mac receives one copy. The disk gate before this one
checked a fixed 20 GB floor once, and df on the network volume shows the backend,
not the quota, so the projection is the only size check the plan gets.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

#: S0 attempt 5, the 107-window landing: about 37 KB written per arrival [V fix list G7].
BYTES_PER_ARRIVAL_S0 = 37_000
#: A dry window whose argv names no duration or rate (older plans) [A].
DRY_WINDOW_ARRIVALS_ASSUMED = 100
HEADROOM_DEFAULT = 0.8


def _argv_value(argv: List[str], flag: str) -> str | None:
    try:
        return argv[argv.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def arrivals_of(plan: Mapping[str, Any]) -> Tuple[int, List[str]]:
    """Total arrivals the plan will write, and notes on what was assumed."""
    total = 0
    notes: List[str] = []
    for step in plan.get("steps", []) or []:
        kind = step.get("kind")
        if step.get("blocked_on") is not None:
            continue
        if kind == "cell":
            windows = int(step.get("windows") or 1)
            per_window = step.get("window_arrivals")
            if per_window is None:
                per_window = step.get("num_queries")
            if per_window is None:
                notes.append(f"cell {step.get('row_key', '?')}: no window_arrivals or num_queries, counted 0")
                continue
            total += windows * int(per_window)
        elif kind == "dry_window":
            argv = step.get("argv") or []
            duration, rate = _argv_value(argv, "--duration-s"), _argv_value(argv, "--rate")
            try:
                total += int(math.ceil(float(duration) * float(rate)))
            except (TypeError, ValueError):
                total += DRY_WINDOW_ARRIVALS_ASSUMED
                notes.append(f"dry window {step.get('row_key', '?')}: duration or rate absent, {DRY_WINDOW_ARRIVALS_ASSUMED} arrivals assumed")
    return total, notes


def project(
    plan: Mapping[str, Any],
    *,
    bytes_per_arrival: int = BYTES_PER_ARRIVAL_S0,
    copies: int = 2,
) -> Dict[str, Any]:
    arrivals, notes = arrivals_of(plan)
    per_copy_gb = arrivals * bytes_per_arrival / 1e9
    return {
        "arrivals": arrivals,
        "bytes_per_arrival": bytes_per_arrival,
        "copies": copies,
        "pod_gb": round(per_copy_gb * copies, 3),
        "mac_gb": round(per_copy_gb, 3),
        "notes": notes,
    }


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(description="G7: project the run's bytes against the pod and Mac disks")
    p.add_argument("plan")
    p.add_argument("--pod-gb", type=float, required=True, help="the pod disk holding the run root and its mirror")
    p.add_argument("--mac-free-gb", type=float, required=True, help="free space on the Mac disk the pull lands on")
    p.add_argument("--copies", type=int, default=2)
    p.add_argument("--bytes-per-arrival", type=int, default=BYTES_PER_ARRIVAL_S0)
    p.add_argument("--headroom", type=float, default=HEADROOM_DEFAULT, help="the fraction of a bound the projection may use")
    args = p.parse_args(argv[1:])
    if args.copies < 1 or args.bytes_per_arrival < 1 or not 0 < args.headroom <= 1:
        print("--copies and --bytes-per-arrival must be >= 1 and --headroom within (0, 1]", file=sys.stderr)
        return 2
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    rec = project(plan, bytes_per_arrival=args.bytes_per_arrival, copies=args.copies)
    pod_limit = args.headroom * args.pod_gb
    mac_limit = args.headroom * args.mac_free_gb
    print(f"disk projection (G7): {rec['arrivals']} arrivals x {rec['bytes_per_arrival']} B x {rec['copies']} copies "
          f"= {rec['pod_gb']:.2f} GB on the pod (bound {args.pod_gb:g} GB, {args.headroom:g} headroom = {pod_limit:.1f} GB); "
          f"{rec['mac_gb']:.2f} GB for the pull (Mac free {args.mac_free_gb:g} GB, limit {mac_limit:.1f} GB)")
    for note in rec["notes"]:
        print(f"  note: {note}")
    bad = []
    if rec["pod_gb"] > pod_limit:
        bad.append(f"the pod disk would fill: {rec['pod_gb']:.2f} GB projected against {pod_limit:.1f} GB usable (raise DISK_GB or VOLUME_GB, or cut the plan)")
    if rec["mac_gb"] > mac_limit:
        bad.append(f"the Mac disk cannot take the pull: {rec['mac_gb']:.2f} GB projected against {mac_limit:.1f} GB usable")
    for b in bad:
        print(f"  [FAIL] {b}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
