#!/usr/bin/env python3
"""Order:     stage 14 (analyze), the first step, before anything is analyzed
Objective: Print every pressure window's regime label by (engine, family) and apply the
           registered in-regime criterion (ADR-0173 amendment E12, 2026-10-17, owner
           option (a); audit G31 folded in; S2 block F row F-10, 2026-11-03): per
           engine, baseline (arm, retriever, policy, topology), dataset, budget rung and
           gated load, at least the floor's windows labeled IN_REGIME whose stationarity
           label is not transient; blocking for the confirmatory roster, descriptive
           elsewhere.
Cloud:     local
           (the Mac; reads the pulled landing)

Usage:
  python3 scripts/6_experiments/regime_gate.py RUN_ROOT [MIN_IN_REGIME_WINDOWS]
          [--loads REGIME_GATE_LOADS] [--confirmatory-arms CONFIRMATORY_ARMS]

MIN_IN_REGIME_WINDOWS empty or absent: report only (exit 0). Set (one integer, or
"load:N" per gated load): every (engine, arm, retriever, policy, topology, dataset,
budget rung, gated load) group of the pressure families F2 and F3 needs at least
that many qualifying windows. A BLOCKING group below it (its cell's baseline in
CONFIRMATORY_ARMS, or every group when the roster is empty) exits 1 and is named; a
DESCRIPTIVE group below it is printed and changes nothing; an engine with pressure
windows but no gated load fails too (the criterion cannot be applied to it), and so
does a gated load no window carries (review H1: a mistyped token would otherwise
pass with its column never judged). A load, floor or roster the grammar refuses
exits 2. The grammar lives in
src/analysis/regime_gate_rules.py, shared with the plan-time projection
(run_campaign.py regime-gate-projection, stage 0), and REGIME_GATE_LOADS defaults to
"1.0+", the rule before F-10.

Reads, per window, ``regime.json`` (``label``; ``stationarity.label`` written by
campaign_layout.stationarity_record, E19: the KV occupancy alone), the cell
directory's row key (``arm|retriever|policy|topology|engine|model|family[|r<budget>]
[|lam<rate>][|cb<tokens>]``, src/analysis/cellspec.CellSpec.to_row_key), the window
directory's dataset (``window_<dataset>-<k>``) and the cell's ``cell.json``
(``baseline``, the B id; missing, unreadable or not a B id = unknown, which blocks
when a roster is set). Stdlib only: the Mac's python3, with the repo root on sys.path for the
rules module.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from src.analysis.regime_gate_rules import (  # noqa: E402
    BLOCKING,
    DEFAULT_LOADS,
    GROUP_KEY_FIELDS,
    PRESSURE_FAMILIES,
    RulesError,
    baseline_id_or_none,
    load_token_for,
    mode_for,
    parse_floor,
    parse_loads,
    parse_roster,
)

WINDOW_DIR_RE = re.compile(r"^window_([a-z0-9_]+)-(\d+)$")  # = campaign_layout.WINDOW_DIR_RE
IN_REGIME = "IN_REGIME"
TRANSIENT = "transient"
RULE = (
    "ADR-0173 amendment E12 (a), S2 block F row F-10: per engine, baseline (arm, retriever, "
    "policy, topology), dataset, budget rung and gated load (REGIME_GATE_LOADS), at least "
    "MIN_IN_REGIME_WINDOWS windows labeled IN_REGIME and not transient (G31); blocking for "
    "CONFIRMATORY_ARMS, descriptive elsewhere"
)


def _coord(parts: List[str], prefix: str) -> Optional[float]:
    for part in parts[7:]:
        if part.startswith(prefix):
            try:
                return float(part[len(prefix):])
            except ValueError:
                return None
    return None


def _baseline_of(cell_dir: Path, cache: Dict[str, Optional[str]]) -> Optional[str]:
    """The cell's recorded baseline id (cell.json ``baseline``); None when the
    record is missing, unreadable or not a B id (the gate then fails closed: BLOCKING)."""
    name = cell_dir.name
    if name not in cache:
        try:
            doc = json.loads((cell_dir / "cell.json").read_text(encoding="utf-8"))
            raw = doc.get("baseline") if isinstance(doc, dict) else None
        except Exception:  # noqa: BLE001 -- an unreadable record is an unknown baseline, never a crash
            raw = None
        cache[name] = baseline_id_or_none(raw)
    return cache[name]


def read_windows(root: Path) -> List[Dict[str, Any]]:
    """One record per regime.json under cells/*/window_*/ (unreadable -> label 'unreadable')."""
    out: List[Dict[str, Any]] = []
    baselines: Dict[str, Optional[str]] = {}
    for regime in sorted(root.glob("cells/*/window_*/regime.json")):
        cell_dir = regime.parent.parent
        parts = cell_dir.name.split("|")
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
                "window": f"{cell_dir.name}/{regime.parent.name}",
                "arm": parts[0],
                "retriever": parts[1],
                "policy": parts[2],
                "topology": parts[3],
                "engine": parts[4],
                "family": parts[6],
                "dataset": m.group(1) if m else None,
                "budget_r": _coord(parts, "r"),
                "rate_frac": _coord(parts, "lam"),
                "baseline": _baseline_of(cell_dir, baselines),
                "label": label,
                "stationarity": stationarity,
            }
        )
    return out


def evaluate(root: Path, need: Any, loads: str = DEFAULT_LOADS, confirmatory_arms: str = "") -> Dict[str, Any]:
    """``need``: None or "" (report only), one integer, or the per-load form
    "0.95:2 1.05:3"; ``loads`` and ``confirmatory_arms`` as the profile writes
    them. A string the grammar refuses raises RulesError (the CLI's exit 2)."""
    gated = parse_loads(loads)
    floors = parse_floor(need, gated)
    roster = parse_roster(confirmatory_arms)
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
        token = load_token_for(w["rate_frac"], gated)
        if token is None:
            continue
        key = tuple(w[field] for field in GROUP_KEY_FIELDS)
        g = groups.setdefault(
            key, {"n_windows": 0, "n_qualifying": 0, "n_in_regime_transient": 0,
                  "n_in_regime_stationarity_unknown": 0, "baseline": w["baseline"],
                  "mode": mode_for(w["baseline"], roster), "load": token,
                  "need": floors[token] if floors else None}
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
    short_descriptive: Dict[str, int] = {}
    no_load: List[str] = []
    no_group: List[str] = []
    if floors is not None:
        for key, g in groups.items():
            if g["n_qualifying"] < g["need"]:
                target = short if g["mode"] == BLOCKING else short_descriptive
                target["|".join(str(k) for k in key)] = g["n_qualifying"]
        grouped_engines = {key[0] for key in groups}
        no_load = sorted(e for e in engines_with_pressure if e not in grouped_engines)
        grouped_tokens = {g["load"] for g in groups.values()}
        no_group = [g.token for g in gated if g.token not in grouped_tokens]   # review H1
    return {
        "rule": RULE,
        "need": None if floors is None else str(need).strip(),
        "floors": floors,
        "loads": loads,
        "roster": sorted(roster),
        "n_windows": len(windows),
        "by_pair": {f"{e} {f}": dict(sorted(c.items())) for (e, f), c in sorted(by_pair.items())},
        "groups": {"|".join(str(k) for k in key): g for key, g in sorted(groups.items(), key=str)},
        "short": short,
        "short_descriptive": short_descriptive,
        "engines_without_a_gated_load": no_load,
        "loads_without_a_group": no_group,
        "ok": floors is None or not (short or no_load or no_group),
    }


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser(prog="regime_gate.py", add_help=False)
    parser.add_argument("run_root")
    parser.add_argument("need", nargs="?", default="")
    parser.add_argument("--loads", default=DEFAULT_LOADS)
    parser.add_argument("--confirmatory-arms", default="")
    try:
        args = parser.parse_args(argv[1:])
    except SystemExit:
        return 2
    try:
        result = evaluate(Path(args.run_root), args.need, loads=args.loads,
                          confirmatory_arms=args.confirmatory_arms)
    except RulesError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print("regime labels per (engine, family): " + ("; ".join(
        f"{pair}: " + ", ".join(f"{label}={n}" for label, n in counts.items())
        for pair, counts in result["by_pair"].items()) or "no window"))
    print(f"E12 groups ({'|'.join(GROUP_KEY_FIELDS)}; loads {args.loads}): " + ("; ".join(
        f"{key}: qualifying={g['n_qualifying']}/{g['n_windows']}"
        + (f" (in-regime but transient={g['n_in_regime_transient']})" if g["n_in_regime_transient"] else "")
        + f" [{g['mode']}" + (f", need {g['need']}" if g["need"] is not None else "") + "]"
        for key, g in result["groups"].items()) or "none"))
    if result["need"] is None:
        print("MIN_IN_REGIME_WINDOWS unset: report only")
        return 0
    raw = result["need"]
    if result["short_descriptive"]:
        print(f"  [NOTE] descriptive group(s) below the floor (no exit code): {result['short_descriptive']}")
    if result["ok"]:
        print(f"[OK] MIN_IN_REGIME_WINDOWS={raw}: every blocking group meets the E12 criterion")
        return 0
    if result["short"]:
        print(f"  [FAIL] MIN_IN_REGIME_WINDOWS={raw}: group(s) below it: {result['short']} ({RULE})")
    if result["engines_without_a_gated_load"]:
        print(f"  [FAIL] engine(s) with pressure windows but no gated load ({args.loads}): "
              f"{result['engines_without_a_gated_load']} (the E12 criterion cannot be applied)")
    if result["loads_without_a_group"]:
        print(f"  [FAIL] gated load(s) with no window ({args.loads}): {result['loads_without_a_group']} "
              "(the E12 criterion was not applied to them: check REGIME_GATE_LOADS against the landing's rates)")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
