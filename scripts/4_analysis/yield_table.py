#!/usr/bin/env python3
"""Order:     stage 13 (analyze) -- per-window serving yield table (Spec Block A1, 2026-10-16)
Objective: compute the section 6.1 currencies G (goodput) and Y (serving yield) for EVERY
           window of a campaign run root, unconditionally, from the request rows, the
           section 8.5 predicate table and the calibration floors; one record or one
           named skip per window, never a silent absence.
Cloud:     local
           (the Mac or the conductor; reads the pulled landing only)

Why this exists (S0 attempt 5, MyDocs/RunPod/S0_ATTEMPT5_REPORT_2026-10-15.md section 6):
the analysis driver evaluates G and Y only inside a registered window contrast, which
needs at least two windows per side per dataset; a run with one window per cell
(S0) therefore never evaluated a single pressure window, although Y is the quantity
in which the memory-pressure effect shows (a request that completes late counts in
attainment and fails the SLO).

Rules carried from the code it calls (read 2026-10-16, cited by line):
- src/analysis/goodput.py:evaluate_window (line 425): required columns ttft_s, tpot_s,
  ok, veridical; optional decode_tokens (ADR-0118: a completed row with 0 decode tokens
  has no TPOT clause and is counted in n_no_decode); timely iff ok AND ttft <= 10 x
  baseline AND tpot <= 5 x baseline; Y iff timely AND veridical; a NaN veridical on a
  completed row refuses; a null or zero TPOT beside a decode phase refuses. The
  decode_tokens column is passed on every row, NaN where the count is missing, as the
  driver does (run_campaign_analysis.py:_decode_tokens_of): the evaluator ignores NaN
  on failed rows and refuses it on completed rows (review 2026-10-16, HIGH 1).
- scripts/4_analysis/run_campaign_analysis.py:_slo_ttft_ms (W16, ADR-0157), applied
  PER ROW exactly as the driver does: a row carrying a finite ttft_from_scheduled_ms
  is graded on it (TTFT from the intended arrival); a COMPLETED row whose
  workload_mode is open_loop and that carries only ttft_ms is a defective record and
  refuses the window; every other row is graded on ttft_ms (the send clock).
- The completion flag is the row's ``ok`` alone, as in the driver (line 3845).
- goodput.corrected_yield_from_window (line 1238): refused when n_timely == 0; Y is
  then 0 raw (ADR-0115). This pass reports the raw Y and never calls the correction.
- The section 6.1 SLO baseline is the single-stream floor of the SAME engine: the
  anchor floor (rungs artifact ladder.floor) for every class except RULER, whose
  requests are 32k tokens and take the RULER floor (classes["32768"].floor,
  ADR-0169). No default baseline exists: an engine without a floor (the HF reference
  engine runs no ladder) makes its windows named skips; no floor file at all refuses
  the whole pass.
- duration_s is measured_window.t_end - t_start, the writer's dispatch span
  (run_experiment.py: measured_window.span = "dispatch"); the writer's own span label
  is carried in the duration_basis column. Rates (goodput_rps, yield_rps) divide by
  that span, so completions landing after t_end are counted against it; fractions
  (goodput_frac, yield_frac) use n_issued and are unaffected.
- RULER rows are outside the section 8.5 predicate universe (predicate_manifest
  skipped_windows), so a RULER window carries G and a named skip for Y.

B3 (ADR-0172, owner 2026-10-16), added 2026-10-16 night:
- Verdicts are keyed by the #127 identity triple (example_id, repeat_index,
  record_index), the driver's key (run_campaign_analysis._predicate_join_key): a
  replayed arrival (ADR-0172 item 1) reads its OWN verdict row, the one the
  scoring pass scored or inherited for it (E18 amendment 2026-10-17, recorded
  as quality_source), never another arrival's row by example_id; a completed
  arrival without its own verdict makes the window a named skip (the
  evaluator refuses a NaN veridical on a completed row). Before this the map was
  keyed by (example_id, repeat_index) and the last row won.
- Item 6 i: time-resolved goodput. The window's arrivals (arrival_s, the intended
  arrival offset over the dispatch span) fall into SLICE_S second slices, each
  evaluated by the same evaluator (goodput_frac, n_timely); the running goodput
  through slice k is the evaluator's goodput_frac over every arrival up to the
  slice's end; the time to breach is the end of the first slice whose running
  goodput is below BREACH_GOODPUT, empty when it never is. The queue growth rate is
  the least squares slope of the waiting gauge (cage_stats.jsonl ``waiting`` over
  ``ts_s``, legacy ``ts``) over the regime window [t_start, t_end) read from
  regime.json, in waiting requests per second; fewer than two samples give none.
  Per-slice rows land in yield_slices.csv beside the table.
- Item 6 ii: the stationarity label the regime writer recorded (regime.json
  ``stationarity.label``) rides the table as a column; empty on a pre-B17 tree.
- Item 6 iii: the lambda* consistency flag per (engine, arm, rung, dataset) in
  yield_summary.md: the modal regime label over the windows at 0.95 x lambda* and
  at 1.05 x lambda* (a tie reads ambiguous); IN_REGIME at 0.95 or UNPRESSURED at
  1.05 flags the class, and a class flagged on two engines is listed apart.

Exit codes: 0 table written (skips are rows, not failures); 3 the named refusal of the
whole pass (printed as "[yield_table] REFUSED: ..."); 2 is left to Python and argparse
(usage error, missing file) so that the master never mistakes a broken call for a refusal.

Usage:
    python scripts/4_analysis/yield_table.py <run_root> --calibration-dir <dir> \
        --scoring-run-id full-cuda-1 --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.analysis.goodput import (  # noqa: E402
    DECODE_TOKENS_COLUMN,
    IN_REGIME,
    GoodputError,
    SLOBaseline,
    TPOT_SLO_MULTIPLIER,
    TTFT_SLO_MULTIPLIER,
    UNPRESSURED,
    WindowMetrics,
    evaluate_window,
)

RULER_DATASET = "ruler"
RULER_CLASS_KEY = "32768"
TTFT_SCHEDULED_COLUMN = "ttft_from_scheduled_ms"
TTFT_SEND_COLUMN = "ttft_ms"
OPEN_LOOP_MODE = "open_loop"
REFUSED_RC = 3
RULER_Y_SKIP = (
    "RULER is outside the section 8.5 predicate universe (instrument dataset): "
    "G is reported, Y is not defined"
)
#: B3 (ADR-0172 item 6 i and iii): the slice length, the running-goodput
#: breach level, and the two rate fractions the lambda* consistency flag reads.
SLICE_S: float = 5.0
BREACH_GOODPUT: float = 0.5
RATE_CHECK: Tuple[str, str] = ("0.95", "1.05")
WINDOW_DESIGN_ADR = "ADR-0172"

ID_COLUMNS: Tuple[str, ...] = (
    "row_key", "window_dir", "engine", "family", "arm", "rung", "rate_frac", "dataset",
    "regime_label", "stationarity", "ttft_basis", "duration_basis", "baseline_source",
    "slo_ttft_ms", "slo_tpot_ms",
)
METRIC_COLUMNS: Tuple[str, ...] = (
    "n_issued", "n_completed", "n_timely", "n_veridical", "n_yield", "n_no_decode",
    "duration_s", "attainment", "throughput_rps", "goodput_rps", "yield_rps",
    "goodput_frac", "yield_frac", "veridical_frac", "covariance_gap", "truth_tax_frac",
)
Y_COLUMNS: Tuple[str, ...] = (
    "n_veridical", "n_yield", "yield_rps", "yield_frac", "veridical_frac",
    "covariance_gap", "truth_tax_frac",
)
#: B16: the time-resolved summary of a record (blank on a skip and on a
#: closed-loop window, whose rows carry no arrival offset).
TIME_COLUMNS: Tuple[str, ...] = ("n_slices", "time_to_breach_s", "queue_growth_rps", "time_note")
ALL_COLUMNS: Tuple[str, ...] = ID_COLUMNS + METRIC_COLUMNS + TIME_COLUMNS + ("skip_reason",)
SLICE_COLUMNS: Tuple[str, ...] = (
    "row_key", "window_dir", "slice_index", "t0_s", "t1_s", "n_issued", "n_timely",
    "goodput_frac", "running_goodput_frac", "note",
)
GROUP_KEYS: Tuple[str, ...] = ("engine", "family", "arm", "rung", "dataset")
CLASS_KEYS: Tuple[str, ...] = ("engine", "arm", "rung", "dataset")


class YieldTableError(RuntimeError):
    """A refusal of the whole pass (missing inputs that no window can do without)."""


@dataclass(frozen=True)
class Baselines:
    """Per-engine section 6.1 floors: anchor class and RULER class."""

    anchor: Dict[str, SLOBaseline]
    ruler: Dict[str, SLOBaseline]
    source: Dict[str, str]

    def for_window(self, engine: str, dataset: str) -> Tuple[SLOBaseline, str]:
        """The engine's floor for the window's class; GoodputError (a per-window
        skip) when that engine has no floor. The HF reference engine never runs
        a calibration ladder, so its windows are named skips by design; no
        default baseline is ever substituted (section 6.1: the SAME engine's floor)."""
        table = self.ruler if dataset == RULER_DATASET else self.anchor
        if engine not in table:
            raise GoodputError(
                f"no section 6.1 {'RULER' if dataset == RULER_DATASET else 'anchor'} floor for "
                f"engine {engine!r} in the calibration artifacts (the reference engine runs no "
                "ladder); no default baseline exists, so the window is a named skip"
            )
        tag = ("ruler" if dataset == RULER_DATASET else "anchor") + ":" + self.source[engine]
        return table[engine], tag


def _floor(obj: Any, where: str) -> SLOBaseline:
    if not isinstance(obj, dict):
        raise YieldTableError(f"{where}: floor record missing")
    try:
        return SLOBaseline(ttft_s=float(obj["ttft_s"]), tpot_s=float(obj["tpot_s"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise YieldTableError(f"{where}: floor lacks finite ttft_s/tpot_s ({exc})") from exc


def load_baselines(calibration_dir: Path) -> Baselines:
    """Read every ``rungs_*_<engine>.json`` under ``calibration_dir``."""
    anchor: Dict[str, SLOBaseline] = {}
    ruler: Dict[str, SLOBaseline] = {}
    source: Dict[str, str] = {}
    paths = sorted(glob.glob(str(calibration_dir / "rungs_*_*.json")))
    if not paths:
        raise YieldTableError(f"no rungs_*_<engine>.json under {calibration_dir}")
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        engine = Path(path).stem.rsplit("_", 1)[-1]
        ladder = doc.get("ladder") if isinstance(doc, dict) else None
        anchor[engine] = _floor((ladder or {}).get("floor"), f"{path} ladder.floor")
        source[engine] = os.path.basename(path)
        classes = doc.get("classes") if isinstance(doc, dict) else None
        ruler_rec = (classes or {}).get(RULER_CLASS_KEY)
        if isinstance(ruler_rec, dict) and isinstance(ruler_rec.get("floor"), dict):
            ruler[engine] = _floor(ruler_rec["floor"], f"{path} classes.{RULER_CLASS_KEY}.floor")
    return Baselines(anchor=anchor, ruler=ruler, source=source)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _identity(rec: Dict[str, Any]) -> Tuple[str, str, Any]:
    """The #127 identity triple of one row, spelled as the driver spells it
    (run_campaign_analysis._predicate_join_key)."""
    return (
        str(rec.get("example_id")),
        str(rec.get("repeat_index") or "0"),
        rec.get("record_index"),
    )


def _predicate_map(path: Path) -> Dict[Tuple[str, str, Any], bool]:
    """identity triple -> predicate verdict; a null verdict is left out. B3
    (ADR-0172 item 1): keyed by record_index too, so a replayed arrival reads
    the verdict row the scoring pass wrote for IT (scored or inherited, E18),
    never another arrival's row."""
    out: Dict[Tuple[str, str, Any], bool] = {}
    for rec in _read_jsonl(path):
        verdict = rec.get("predicate")
        if verdict is None:
            continue
        out[_identity(rec)] = bool(verdict)
    return out


def _slo_ttft(rec: Dict[str, Any], is_ok: bool) -> Tuple[float, str]:
    """(TTFT in ms for the section 6.1 gate, its clock basis), the driver's W16 rule
    per row (run_campaign_analysis.py:_slo_ttft_ms)."""
    scheduled = rec.get(TTFT_SCHEDULED_COLUMN)
    send = rec.get(TTFT_SEND_COLUMN)
    if (
        isinstance(scheduled, (int, float))
        and not isinstance(scheduled, bool)
        and math.isfinite(float(scheduled))
    ):
        return float(scheduled), "scheduled"
    if rec.get("workload_mode") == OPEN_LOOP_MODE and is_ok and send is not None:
        raise GoodputError(
            f"completed open-loop row {rec.get('example_id')!r} carries {TTFT_SEND_COLUMN}={send!r} "
            f"but no {TTFT_SCHEDULED_COLUMN}: defective record, not graded on the send clock (W16, ADR-0157)"
        )
    return (float(send) if send is not None else float("nan")), "send"


def build_records(rows: List[Dict[str, Any]], predicate: Optional[Dict[Tuple[str, str, Any], bool]]) -> Tuple[pd.DataFrame, str]:
    """Rows -> the evaluate_window frame. Returns (frame, ttft_basis).

    ttft_basis names the clock of the completed rows: "scheduled", "send", or
    "mixed:scheduled=<n>,send=<m>" when a window holds both. ``predicate`` None
    means the window has no predicate verdicts (RULER): veridical is False on
    every row so that G is still computed; the caller blanks the Y columns.
    """
    ttft_s: List[float] = []
    tpot_s: List[float] = []
    ok: List[bool] = []
    verid: List[Any] = []
    decode: List[float] = []
    basis_counts = {"scheduled": 0, "send": 0}
    for rec in rows:
        is_ok = bool(rec.get("ok"))
        ttft_ms, basis = _slo_ttft(rec, is_ok)
        if is_ok:
            basis_counts[basis] += 1
        ttft_s.append(ttft_ms / 1000.0)
        tpot = rec.get("tpot_ms")
        tpot_s.append(float(tpot) / 1000.0 if tpot is not None else float("nan"))
        ok.append(is_ok)
        ntok = rec.get("num_tokens")
        decode.append(max(float(ntok) - 1.0, 0.0) if ntok is not None else float("nan"))
        if predicate is None:
            verid.append(False)
        else:
            # a completed arrival without its own verdict stays NaN: the
            # evaluator refuses the window by name (never inherited, ADR-0172)
            verid.append(predicate.get(_identity(rec), float("nan")) if is_ok else False)
    frame = pd.DataFrame(
        {"ttft_s": ttft_s, "tpot_s": tpot_s, "ok": ok, "veridical": verid, DECODE_TOKENS_COLUMN: decode}
    )
    used = [k for k, n in basis_counts.items() if n]
    if len(used) == 1:
        ttft_basis = used[0]
    elif not used:
        ttft_basis = "none"
    else:
        ttft_basis = f"mixed:scheduled={basis_counts['scheduled']},send={basis_counts['send']}"
    return frame, ttft_basis


def _row_key_parts(row_key: str) -> Dict[str, str]:
    parts = row_key.split("|")
    rung = next((p[1:] for p in parts[7:] if p.startswith("r") and p[1:2].isdigit()), "")
    # CellSpec.to_row_key spells the rate fraction as lam<frac> after the rung
    rate = next((p[3:] for p in parts[7:] if p.startswith("lam")), "")
    return {
        "arm": parts[0] if parts else "",
        "engine": parts[4] if len(parts) > 4 else "",
        "family": parts[6] if len(parts) > 6 else "",
        "rung": rung,
        "rate_frac": rate,
    }


def _arrival_offsets(rows: List[Dict[str, Any]]) -> Optional[np.ndarray]:
    """Every row's arrival_s as an array, or None when any row lacks a finite one
    (a closed-loop window has no intended arrivals)."""
    values: List[float] = []
    for rec in rows:
        a = rec.get("arrival_s")
        if isinstance(a, bool) or not isinstance(a, (int, float)) or not math.isfinite(float(a)):
            return None
        values.append(float(a))
    return np.asarray(values, dtype=float) if values else None


def time_resolved(
    rows: List[Dict[str, Any]],
    frame: pd.DataFrame,
    baseline: SLOBaseline,
    duration: float,
    gpu_count: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """B16 (ADR-0172 item 6 i): goodput per SLICE_S slice of arrivals and the
    time to breach. Returns (per-slice records, {n_slices, time_to_breach_s,
    note}). Each slice and each running prefix is judged by evaluate_window on
    its own rows, the one threshold source; a slice the evaluator refuses
    carries the reason and no number. The breach is the end of the first
    slice whose running goodput (every arrival up to the slice's end) is
    below BREACH_GOODPUT."""
    arrivals = _arrival_offsets(rows)
    if arrivals is None:
        return [], {
            "n_slices": "", "time_to_breach_s": "",
            "note": "no arrival_s on every row (closed loop): no time-resolved slices",
        }
    n_slices = max(1, int(math.ceil(duration / SLICE_S)))
    slices: List[Dict[str, Any]] = []
    breach: Optional[float] = None
    for k in range(n_slices):
        lo = k * SLICE_S
        hi = min((k + 1) * SLICE_S, duration)
        last = k == n_slices - 1
        below_hi = (arrivals <= hi) if last else (arrivals < hi)
        in_slice = (arrivals >= lo) & below_hi
        rec: Dict[str, Any] = {
            "slice_index": k, "t0_s": lo, "t1_s": hi, "n_issued": int(in_slice.sum()),
            "n_timely": "", "goodput_frac": "", "running_goodput_frac": "", "note": "",
        }
        notes: List[str] = []
        if rec["n_issued"]:
            try:
                wm = evaluate_window(
                    frame[in_slice].reset_index(drop=True), baseline,
                    duration_s=max(hi - lo, 1e-9), gpu_count=gpu_count,
                )
                rec["n_timely"], rec["goodput_frac"] = wm.n_timely, wm.goodput_frac
            except GoodputError as exc:
                notes.append(f"slice: {exc}")
        if below_hi.any():
            try:
                running = evaluate_window(
                    frame[below_hi].reset_index(drop=True), baseline,
                    duration_s=max(hi, 1e-9), gpu_count=gpu_count,
                ).goodput_frac
                rec["running_goodput_frac"] = running
                if breach is None and running < BREACH_GOODPUT:
                    breach = hi
            except GoodputError as exc:
                notes.append(f"running: {exc}")
        rec["note"] = "; ".join(notes)
        slices.append(rec)
    return slices, {
        "n_slices": n_slices,
        "time_to_breach_s": "" if breach is None else breach,
        "note": "",
    }


def queue_growth(wdir: Path, regime: Dict[str, Any]) -> Tuple[Any, str]:
    """B16: the least squares slope of the waiting gauge over the regime window,
    in waiting requests per second; ("", reason) when it cannot be read."""
    t0, t1 = regime.get("t_start"), regime.get("t_end")
    if any(isinstance(t, bool) or not isinstance(t, (int, float)) for t in (t0, t1)):
        return "", "regime.json without t_start/t_end: no queue slope"
    path = wdir / "cage_stats.jsonl"
    if not path.is_file():
        return "", "cage_stats.jsonl absent: no queue slope"
    ts: List[float] = []
    waiting: List[float] = []
    for rec in _read_jsonl(path):
        t = rec.get("ts_s", rec.get("ts"))
        w = rec.get("waiting")
        if (
            isinstance(t, bool) or not isinstance(t, (int, float))
            or isinstance(w, bool) or not isinstance(w, (int, float))
        ):
            continue
        if float(t0) <= float(t) < float(t1):
            ts.append(float(t))
            waiting.append(float(w))
    if len(set(ts)) < 2:
        return "", f"{len(ts)} queue sample(s) in the window: no slope"
    slope = float(np.polyfit(np.asarray(ts), np.asarray(waiting), 1)[0])
    return slope, ""


def lambda_star_consistency(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """B18 (ADR-0172 item 6 iii): one verdict per (engine, arm, rung, dataset)
    from the modal regime labels of its windows at 0.95 and 1.05 x lambda*.
    IN_REGIME at 0.95 (the estimate reads too high) or UNPRESSURED at 1.05
    (the estimate reads too low) flags the class; a missing, ambiguous or
    UNKNOWN_TELEMETRY label without a flag reads unknown."""
    groups: Dict[Tuple[str, ...], Dict[str, List[str]]] = {}
    for r in rows:
        rate = str(r.get("rate_frac") or "")
        if rate not in RATE_CHECK:
            continue
        key = tuple(str(r.get(k) or "") for k in CLASS_KEYS)
        groups.setdefault(key, {rc: [] for rc in RATE_CHECK})[rate].append(str(r.get("regime_label") or ""))
    out: List[Dict[str, Any]] = []
    for key in sorted(groups):
        labels: Dict[str, str] = {}
        counts: Dict[str, int] = {}
        for rate in RATE_CHECK:
            seen = [label for label in groups[key][rate] if label]
            counts[rate] = len(seen)
            if not seen:
                labels[rate] = "missing"
                continue
            tally = {label: seen.count(label) for label in set(seen)}
            top = max(tally.values())
            modal = sorted(label for label, n in tally.items() if n == top)
            labels[rate] = modal[0] if len(modal) == 1 else "ambiguous"
        flags: List[str] = []
        if labels[RATE_CHECK[0]] == IN_REGIME:
            flags.append(f"IN_REGIME at {RATE_CHECK[0]} (lambda* reads too high)")
        if labels[RATE_CHECK[1]] == UNPRESSURED:
            flags.append(f"UNPRESSURED at {RATE_CHECK[1]} (lambda* reads too low)")
        if flags:
            verdict = "FLAGGED: " + "; ".join(flags)
        elif any(labels[r] in ("missing", "ambiguous", "UNKNOWN_TELEMETRY") for r in RATE_CHECK):
            verdict = "unknown"
        else:
            verdict = "consistent"
        out.append(
            {
                **dict(zip(CLASS_KEYS, key)),
                "label_0.95": labels[RATE_CHECK[0]],
                "label_1.05": labels[RATE_CHECK[1]],
                "n_0.95": counts[RATE_CHECK[0]],
                "n_1.05": counts[RATE_CHECK[1]],
                "verdict": verdict,
            }
        )
    return out


def evaluate_run(
    run_root: Path, *, baselines: Baselines, scoring_run_id: str
) -> List[Dict[str, Any]]:
    """One output row per window directory (``cells/*/window_*``): a record or a named skip."""
    out: List[Dict[str, Any]] = []
    cells_dir = run_root / "cells"
    if not cells_dir.is_dir():
        raise YieldTableError(f"no cells/ under {run_root}")
    for wdir in sorted(p for p in cells_dir.glob("*/window_*") if p.is_dir()):
        row_key = wdir.parent.name
        parts = _row_key_parts(row_key)
        rec: Dict[str, Any] = {c: "" for c in ALL_COLUMNS}
        rec.update({"row_key": row_key, "window_dir": wdir.name, **parts})
        try:
            metrics_path = wdir / "metrics.json"
            if not metrics_path.is_file():
                raise GoodputError("metrics.json missing: the window has no dataset, duration or request count")
            with open(metrics_path, encoding="utf-8") as fh:
                metrics = json.load(fh)
            cell_path = wdir.parent / "cell.json"
            gpu_count = 1
            if cell_path.is_file():
                with open(cell_path, encoding="utf-8") as fh:
                    cell = json.load(fh)
                spec = cell.get("cellspec") or {}
                rec["engine"] = str(spec.get("engine") or rec["engine"])
                rec["family"] = str(spec.get("family") or rec["family"])
                rec["arm"] = str(spec.get("arm") or rec["arm"])
                gpu_count = int(cell.get("gpu_count") or 1)
            dataset = str((metrics.get("experiment") or {}).get("dataset") or "")
            rec["dataset"] = dataset
            regime: Dict[str, Any] = {}
            regime_path = wdir / "regime.json"
            if regime_path.is_file():
                with open(regime_path, encoding="utf-8") as fh:
                    regime = json.load(fh)
                rec["regime_label"] = str(regime.get("label") or "")
                # B17 (ADR-0172 item 6 ii): the writer's label; empty on a pre-B17 tree
                stationarity = regime.get("stationarity")
                rec["stationarity"] = (
                    str(stationarity.get("label") or "") if isinstance(stationarity, dict) else ""
                )
            mw = metrics.get("measured_window") or {}
            t0, t1 = mw.get("t_start"), mw.get("t_end")
            if t0 is None or t1 is None:
                raise GoodputError("measured_window t_start/t_end missing: no window duration")
            duration = float(t1) - float(t0)
            rec["duration_basis"] = str(mw.get("span") or "unlabeled")
            baseline, source = baselines.for_window(rec["engine"], dataset)
            rec["baseline_source"] = source
            rec["slo_ttft_ms"] = round(TTFT_SLO_MULTIPLIER * baseline.ttft_s * 1000.0, 3)
            rec["slo_tpot_ms"] = round(TPOT_SLO_MULTIPLIER * baseline.tpot_s * 1000.0, 3)
            rows = _read_jsonl(wdir / "requests.jsonl")
            total = (metrics.get("performance") or {}).get("total_requests")
            if total is not None and int(total) != len(rows):
                raise GoodputError(
                    f"requests.jsonl holds {len(rows)} rows but performance.total_requests is {total}: "
                    "n_issued is ambiguous"
                )
            predicate: Optional[Dict[Tuple[str, str], bool]] = None
            y_skip = ""
            if dataset == RULER_DATASET:
                y_skip = RULER_Y_SKIP
            else:
                ppath = run_root / "predicate" / scoring_run_id / "cells" / row_key / wdir.name / "predicate.jsonl"
                if not ppath.is_file():
                    raise GoodputError(f"no predicate table for the window: {ppath.relative_to(run_root)}")
                predicate = _predicate_map(ppath)
            frame, basis = build_records(rows, predicate)
            rec["ttft_basis"] = basis
            wm: WindowMetrics = evaluate_window(frame, baseline, duration_s=duration, gpu_count=gpu_count)
            flat = wm.to_flat_dict()
            for col in METRIC_COLUMNS:
                val = flat.get(col)
                rec[col] = "" if val is None else val
            if y_skip:
                for col in Y_COLUMNS:
                    rec[col] = ""
                rec["skip_reason"] = y_skip
            # B16 (ADR-0172 item 6 i): the window over time, from the same rows
            slices, resolved = time_resolved(rows, frame, baseline, duration, gpu_count)
            slope, queue_note = queue_growth(wdir, regime)
            rec["n_slices"] = resolved["n_slices"]
            rec["time_to_breach_s"] = resolved["time_to_breach_s"]
            rec["queue_growth_rps"] = slope
            rec["time_note"] = "; ".join(n for n in (resolved["note"], queue_note) if n)
            rec["_slices"] = slices
        except YieldTableError:
            raise
        except (GoodputError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            for col in METRIC_COLUMNS + TIME_COLUMNS:
                rec[col] = ""
            rec["_slices"] = []
            rec["skip_reason"] = f"{type(exc).__name__}: {exc}"
        out.append(rec)
    return out


def _is_record(r: Dict[str, Any]) -> bool:
    return not r["skip_reason"] or r["skip_reason"] == RULER_Y_SKIP


def write_outputs(rows: List[Dict[str, Any]], out_dir: Path, *, run_root: Path) -> Tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "yield_table.csv"
    tmp = csv_path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ALL_COLUMNS))
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in ALL_COLUMNS})
    os.replace(tmp, csv_path)
    # B16: one row per slice, beside the table
    slices_path = out_dir / "yield_slices.csv"
    tmp_slices = slices_path.with_suffix(".csv.tmp")
    n_slice_rows = 0
    with open(tmp_slices, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(SLICE_COLUMNS))
        writer.writeheader()
        for r in rows:
            for s in r.get("_slices") or []:
                writer.writerow({"row_key": r["row_key"], "window_dir": r["window_dir"], **{c: s.get(c, "") for c in SLICE_COLUMNS[2:]}})
                n_slice_rows += 1
    os.replace(tmp_slices, slices_path)
    md_path = out_dir / "yield_summary.md"
    records = [r for r in rows if _is_record(r)]
    skips = [r for r in rows if not _is_record(r)]
    sliced = [r for r in records if r.get("n_slices") not in ("", None)]
    breached = [r for r in sliced if r.get("time_to_breach_s") not in ("", None)]
    lines = [
        "# Serving yield per window (Spec Block A1)",
        "",
        f"- run root: `{run_root}`",
        f"- windows: {len(rows)}; records: {len(records)}; named skips: {len(skips)}",
        f"- SLO: TTFT <= {TTFT_SLO_MULTIPLIER:g} x floor, TPOT <= {TPOT_SLO_MULTIPLIER:g} x floor (section 6.1); "
        "rows with a scheduled clock are graded from the intended arrival (W16), the others from the send",
        "- RULER windows carry G only (outside the section 8.5 predicate universe)",
        "- duration_s is the writer's dispatch span (duration_basis column); rates divide by it, fractions use n_issued",
        f"- verdicts joined by (example_id, repeat_index, record_index): each arrival reads its own verdict row, scored or inherited by the scoring pass ({WINDOW_DESIGN_ADR} as amended 2026-10-17, E18)",
        f"- time-resolved goodput ({WINDOW_DESIGN_ADR} 6 i): {SLICE_S:g} s slices in `{slices_path.name}` ({n_slice_rows} slice row(s) over {len(sliced)} window(s)); "
        f"time_to_breach_s = end of the first slice whose running goodput is below {BREACH_GOODPUT:g} ({len(breached)} window(s) breached); "
        "queue_growth_rps = least squares slope of the waiting gauge over the regime window",
        f"- stationarity ({WINDOW_DESIGN_ADR} 6 ii): the regime writer's label per window (second vs third third), empty on a pre-B17 tree",
        "",
        "| engine | family | arm | rung | rate | dataset | window | regime | stationarity | n | completed | timely | yield | goodput_frac | yield_frac | breach s | note |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        note = "" if not r["skip_reason"] else ("G only" if r["skip_reason"] == RULER_Y_SKIP else r["skip_reason"][:90])
        lines.append(
            f"| {r['engine']} | {r['family']} | {r['arm']} | {r['rung']} | {r['rate_frac']} | {r['dataset']} | {r['window_dir']} | "
            f"{r['regime_label']} | {r['stationarity']} | {r['n_issued']} | {r['n_completed']} | {r['n_timely']} | {r['n_yield']} | "
            f"{_fmt(r['goodput_frac'])} | {_fmt(r['yield_frac'])} | {_fmt(r.get('time_to_breach_s'))} | {note} |"
        )
    if records:
        df = pd.DataFrame(records)
        for col in ("goodput_frac", "yield_frac"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        lines += [
            "",
            "## Medians per (engine, family, arm, rung, dataset), records only",
            "",
            "| engine | family | arm | rung | dataset | windows | median goodput_frac | median yield_frac |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for key, g in df.groupby(list(GROUP_KEYS), sort=True):
            lines.append(
                f"| {' | '.join(str(k) for k in key)} | {len(g)} | "
                f"{_fmt(g['goodput_frac'].median())} | {_fmt(g['yield_frac'].median())} |"
            )
        lines += [
            "",
            "## F1 reference (unpressured windows) per engine",
            "",
            "| engine | windows | median goodput_frac | median yield_frac |",
            "|---|---|---|---|",
        ]
        f1 = df[df["family"] == "F1"]
        for eng, g in f1.groupby("engine", sort=True):
            lines.append(f"| {eng} | {len(g)} | {_fmt(g['goodput_frac'].median())} | {_fmt(g['yield_frac'].median())} |")
        if f1.empty:
            lines.append("| (no F1 record) | 0 | | |")
    # B18 (ADR-0172 item 6 iii): the lambda* consistency flag per class
    verdicts = lambda_star_consistency(rows)
    lines += [
        "",
        f"## lambda* consistency per (engine, arm, rung, dataset) (B18, {WINDOW_DESIGN_ADR} 6 iii)",
        "",
        f"Rule: the modal regime label of the windows at {RATE_CHECK[0]} x lambda* and at {RATE_CHECK[1]} x lambda* "
        f"(a tie reads ambiguous); {IN_REGIME} at {RATE_CHECK[0]} or {UNPRESSURED} at {RATE_CHECK[1]} flags the class.",
        "",
        "| engine | arm | rung | dataset | label at 0.95 (n) | label at 1.05 (n) | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for v in verdicts:
        lines.append(
            f"| {v['engine']} | {v['arm']} | {v['rung']} | {v['dataset']} | {v['label_0.95']} ({v['n_0.95']}) | "
            f"{v['label_1.05']} ({v['n_1.05']}) | {v['verdict']} |"
        )
    if not verdicts:
        lines.append(f"| (no window at {RATE_CHECK[0]} or {RATE_CHECK[1]} x lambda*) | | | | | | |")
    flagged_by_class: Dict[Tuple[str, str, str], List[str]] = {}
    for v in verdicts:
        if v["verdict"].startswith("FLAGGED"):
            flagged_by_class.setdefault((v["arm"], v["rung"], v["dataset"]), []).append(v["engine"])
    both = {k: sorted(e) for k, e in flagged_by_class.items() if len(set(e)) >= 2}
    lines.append("")
    if both:
        for (arm, rung, dataset), engines in sorted(both.items()):
            lines.append(f"- class {arm} r{rung} {dataset} flagged on {len(engines)} engines ({', '.join(engines)})")
    else:
        lines.append("- no class flagged on two engines")
    tmp_md = md_path.with_suffix(".md.tmp")
    tmp_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp_md, md_path)
    return csv_path, md_path, slices_path


def _fmt(v: Any) -> str:
    if v == "" or v is None:
        return ""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "" if math.isnan(f) else f"{f:.3f}"


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_root", help="campaign run root (holds cells/, predicate/)")
    ap.add_argument("--calibration-dir", required=True, help="dir holding rungs_*_<engine>.json")
    ap.add_argument("--scoring-run-id", required=True, help="predicate table id, for example full-cuda-1")
    ap.add_argument("--out", required=True, help="output directory for yield_table.csv and yield_summary.md")
    args = ap.parse_args(list(argv) if argv is not None else None)
    run_root = Path(args.run_root)
    try:
        baselines = load_baselines(Path(args.calibration_dir))
        rows = evaluate_run(run_root, baselines=baselines, scoring_run_id=args.scoring_run_id)
    except YieldTableError as exc:
        print(f"[yield_table] REFUSED: {exc}", file=sys.stderr)
        return REFUSED_RC
    csv_path, md_path, slices_path = write_outputs(rows, Path(args.out), run_root=run_root)
    n_rec = sum(1 for r in rows if _is_record(r))
    n_skip = len(rows) - n_rec
    print(f"[yield_table] windows={len(rows)} records={n_rec} skips={n_skip} -> {csv_path} {md_path} {slices_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
