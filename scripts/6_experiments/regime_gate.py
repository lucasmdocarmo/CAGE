#!/usr/bin/env python3
"""Order:     stage 13 (analyze), the first step, before anything is analyzed
Objective: Print every pressure window's regime label by (engine, family) and apply the
           registered in-regime criterion (ADR-0173 amendment E12, 2026-10-17, owner
           option (a); audit G31 folded in): per engine, class (arm, dataset) and
           rate rung >= 1.0, at least MIN_IN_REGIME_WINDOWS replicate windows labeled
           IN_REGIME whose stationarity label is not transient.
Cloud:     local
           (the Mac; reads the pulled landing)

Usage:
  python3 scripts/6_experiments/regime_gate.py RUN_ROOT [MIN_IN_REGIME_WINDOWS]

MIN_IN_REGIME_WINDOWS empty or absent: report only (exit 0). Set: every
(engine, arm, dataset, budget rung, rate rung >= 1.0) group of the pressure
families F2 and F3 needs at least that many qualifying windows, and an engine
with pressure windows but no rate rung >= 1.0 fails too (the criterion cannot
be applied to it); any shortfall exits 1 and names the groups.

Reads, per window, ``regime.json`` (``label``; ``stationarity.label`` written by
campaign_layout.stationarity_record, E19: the KV occupancy alone) and the cell
directory's row key (``arm|retriever|policy|topology|engine|model|family[|r<budget>]
[|lam<rate>][|cb<tokens>]``, src/analysis/cellspec.CellSpec.to_row_key) and the
window directory's dataset (``window_<dataset>-<k>``). Stdlib only.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

WINDOW_DIR_RE = re.compile(r"^window_([a-z0-9_]+)-(\d+)$")  # = campaign_layout.WINDOW_DIR_RE
PRESSURE_FAMILIES = ("F2", "F3")
IN_REGIME = "IN_REGIME"
TRANSIENT = "transient"
#: E12: the criterion applies to the rate rungs at or above the measured lambda*.
RATE_RUNG_FLOOR = 1.0
RULE = (
    "ADR-0173 amendment E12 (a): per engine, class (arm, dataset) and rate rung >= "
    f"{RATE_RUNG_FLOOR:g}, at least MIN_IN_REGIME_WINDOWS windows labeled IN_REGIME "
    "and not transient (G31)"
)


def _coord(parts: List[str], prefix: str) -> Optional[float]:
    for part in parts[7:]:
        if part.startswith(prefix):
            try:
                return float(part[len(prefix):])
            except ValueError:
                return None
    return None


def read_windows(root: Path) -> List[Dict[str, Any]]:
    """One record per regime.json under cells/*/window_*/ (unreadable -> label 'unreadable')."""
    out: List[Dict[str, Any]] = []
    for regime in sorted(root.glob("cells/*/window_*/regime.json")):
        parts = regime.parent.parent.name.split("|")
        if len(parts) < 7:
            continue
        m = WINDOW_DIR_RE.match(regime.parent.name)
        try:
            doc = json.loads(regime.read_text(encoding="utf-8"))
            label = str(doc.get("label"))
            stationarity = (doc.get("stationarity") or {}).get("label")
        except Exception:  # noqa: BLE001 -- a broken file is a labeled row, never a crash
            label, stationarity = "unreadable", None
        out.append(
            {
                "window": f"{regime.parent.parent.name}/{regime.parent.name}",
                "arm": parts[0],
                "engine": parts[4],
                "family": parts[6],
                "dataset": m.group(1) if m else None,
                "budget_r": _coord(parts, "r"),
                "rate_frac": _coord(parts, "lam"),
                "label": label,
                "stationarity": stationarity,
            }
        )
    return out


def evaluate(root: Path, need: Optional[int]) -> Dict[str, Any]:
    windows = read_windows(root)
    by_pair: Dict[tuple, Counter] = {}
    for w in windows:
        by_pair.setdefault((w["engine"], w["family"]), Counter())[w["label"]] += 1
    groups: Dict[tuple, Dict[str, Any]] = {}
    engines_with_pressure: set = set()
    for w in windows:
        if w["family"] not in PRESSURE_FAMILIES:
            continue
        engines_with_pressure.add(w["engine"])
        if w["rate_frac"] is None or w["rate_frac"] < RATE_RUNG_FLOOR:
            continue
        key = (w["engine"], w["arm"], w["dataset"], w["budget_r"], w["rate_frac"])
        g = groups.setdefault(
            key, {"n_windows": 0, "n_qualifying": 0, "n_in_regime_transient": 0,
                  "n_in_regime_stationarity_unknown": 0}
        )
        g["n_windows"] += 1
        if w["label"] == IN_REGIME:
            if w["stationarity"] == TRANSIENT:
                g["n_in_regime_transient"] += 1
            else:
                g["n_qualifying"] += 1
                if w["stationarity"] != "stationary":
                    g["n_in_regime_stationarity_unknown"] += 1
    short: Dict[str, int] = {}
    no_rung: List[str] = []
    if need is not None:
        for key, g in groups.items():
            if g["n_qualifying"] < need:
                short["|".join(str(k) for k in key)] = g["n_qualifying"]
        grouped_engines = {key[0] for key in groups}
        no_rung = sorted(e for e in engines_with_pressure if e not in grouped_engines)
    return {
        "rule": RULE,
        "need": need,
        "n_windows": len(windows),
        "by_pair": {f"{e} {f}": dict(sorted(c.items())) for (e, f), c in sorted(by_pair.items())},
        "groups": {"|".join(str(k) for k in key): g for key, g in sorted(groups.items(), key=str)},
        "short": short,
        "engines_without_a_rate_rung_at_or_above_floor": no_rung,
        "ok": need is None or (not short and not no_rung),
    }


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    root = Path(argv[1])
    raw = argv[2].strip() if len(argv) > 2 else ""
    need: Optional[int] = None
    if raw:
        try:
            need = int(raw)
        except ValueError:
            print(f"MIN_IN_REGIME_WINDOWS={raw!r} is not an integer", file=sys.stderr)
            return 2
        if need < 1:
            print(f"MIN_IN_REGIME_WINDOWS={need} must be >= 1 (empty = report only)", file=sys.stderr)
            return 2
    result = evaluate(root, need)
    print("regime labels per (engine, family): " + ("; ".join(
        f"{pair}: " + ", ".join(f"{l}={n}" for l, n in counts.items())
        for pair, counts in result["by_pair"].items()) or "no window"))
    print(f"E12 groups (engine|arm|dataset|budget_r|rate_frac >= {RATE_RUNG_FLOOR:g}): " + ("; ".join(
        f"{key}: qualifying={g['n_qualifying']}/{g['n_windows']}"
        + (f" (in-regime but transient={g['n_in_regime_transient']})" if g["n_in_regime_transient"] else "")
        for key, g in result["groups"].items()) or "none"))
    if need is None:
        print("MIN_IN_REGIME_WINDOWS unset: report only")
        return 0
    if result["ok"]:
        print(f"[OK] MIN_IN_REGIME_WINDOWS={need}: every group meets the E12 criterion")
        return 0
    if result["short"]:
        print(f"  [FAIL] MIN_IN_REGIME_WINDOWS={need}: group(s) below it: {result['short']} ({RULE})")
    if result["engines_without_a_rate_rung_at_or_above_floor"]:
        print("  [FAIL] engine(s) with pressure windows but no rate rung >= "
              f"{RATE_RUNG_FLOOR:g}: {result['engines_without_a_rate_rung_at_or_above_floor']} "
              "(the E12 criterion cannot be applied)")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
