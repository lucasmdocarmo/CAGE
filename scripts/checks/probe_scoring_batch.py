#!/usr/bin/env python3
"""Order:     stage 10 (score), once, on the pod before the full scoring pass
Objective: Prove live that the cross-row batched scoring path grades as the row-by-row path within the profile's stated per-row bounds, every predicate input exact, and measure both rates (ADR-0176 Group D; fix-list B13, G19; S2 block F row F-11)
Cloud:     both

The rescorer (scripts/4_analysis/rescore_quality.py) grades every answer one
row at a time unless --batch-size is given, which batches the entailment and
the completeness instruments across rows (charter D8 section 8.1). The two
paths are proven identical on fakes (tests/test_rescore_wiring.py) but never
on the real instruments, so G19 left the profile key empty until a component
box proved it. This probe is that box, run at the start of stage 10: it takes
the first N completed answers of the first scorable window of the sealed run,
writes them to a scratch evidence file OUTSIDE the run tree, grades them twice
with the real instruments (row by row, then batched at the profile's batch
size, after an untimed warm-up pass over the same sample so every lazy model
load sits outside the timed passes, review 2026-10-23 M1), compares every
emitted row byte for byte (S2 block F row F-11, owner 2026-11-04, option (a):
faithfulness and completeness_bertscore may differ per row within the bounds
the profile states as --tol-faithfulness and --tol-bertscore, 0 = exact, while
every field the predicate reads stays exact), and prints ONE line the master
greps: SCORING_BATCH_OK with both per-row rates, the bounds and the largest
difference per field, or SCORING_BATCH_FAILED naming the first row and field
outside the bounds and the largest difference per field. A third,
untimed row-by-row pass is compared with the first as a control (review M2):
a device that is not call-to-call deterministic shows there, so a failure can
never be mistaken for a batching effect. The master passes --batch-size to
the pass only on OK. The run tree, the scoring directory and the run's score
cache are never written; the record goes to --out and the sample under logs/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

N_ROWS = 50
OK_LINE = "SCORING_BATCH_OK"
FAIL_LINE = "SCORING_BATCH_FAILED"
EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2
DEFAULT_SCRATCH = _REPO_ROOT / "logs" / "scoring_batch_probe"
# S2 block F row F-11 (owner 2026-11-04, option (a)): the two continuous
# exploratory fields a profile may bound per row (flag -> field), and the
# predicate's inputs, which no bound may cover (the S1 probe measured them
# identical on 50 rows; MyDocs/RunPod/S2_F11_SCORING_TOLERANCE_EVIDENCE_2026-11-04.md)
TOLERATED = {"--tol-faithfulness": "faithfulness", "--tol-bertscore": "completeness_bertscore"}
PREDICATE_INPUTS = ("grounding_score", "exact_match", "predicted_no_answer", "hallucination_detected")


def load_rescorer() -> Any:
    """The rescorer script as a module (it is a script, not a package)."""
    path = _REPO_ROOT / "scripts" / "4_analysis" / "rescore_quality.py"
    spec = importlib.util.spec_from_file_location("cage_rescore_quality_probe", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _canonical(value: Any, default: Callable[[Any], Any]) -> str:
    return json.dumps(value, sort_keys=True, default=default)


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_tolerances(tolerances: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """The per-row bounds as floats (row F-11): each a non-negative finite
    number on a field the predicate never reads; anything else is refused."""
    out: Dict[str, float] = {}
    for key, value in (tolerances or {}).items():
        if key in PREDICATE_INPUTS:
            raise ValueError(f"tolerance on {key}: the predicate reads it, so it stays exact (row F-11)")
        # copysign reads the sign bit, so -0.0 is refused with the negatives (review 2026-11-04 LOW 2)
        if not _numeric(value) or not math.isfinite(float(value)) or math.copysign(1.0, float(value)) < 0:
            raise ValueError(f"tolerance for {key} must be a non-negative finite number, got {value!r}")
        out[key] = float(value)
    return out


def _field_within(a: Dict[str, Any], b: Dict[str, Any], key: str, tol: Optional[float], default: Callable[[Any], Any]) -> bool:
    """One field of one row: identical as canonical text, or, on a bounded
    field, two finite numbers at most ``tol`` apart. A key present on one side
    only, a None against a number, a NaN against a number, or a bool, is a
    difference whatever the bound."""
    if (key in a) != (key in b):
        return False
    va, vb = a.get(key), b.get(key)
    if _canonical(va, default) == _canonical(vb, default):
        return True
    if tol is None or not (_numeric(va) and _numeric(vb)):
        return False
    da, db = float(va), float(vb)
    return math.isfinite(da) and math.isfinite(db) and abs(da - db) <= tol


def compare_rows(
    sequential: Sequence[Dict[str, Any]],
    batched: Sequence[Dict[str, Any]],
    *,
    tolerances: Optional[Dict[str, Any]] = None,
    default: Callable[[Any], Any] = str,
) -> Dict[str, Any]:
    """Byte-for-byte comparison of the two emitted row lists (canonical JSON
    per row, the form qa_scores.jsonl is written in), except that a field named
    in ``tolerances`` may differ per row by at most its bound (row F-11; empty =
    exact everywhere, today's rule). The record counts the identical rows and
    the rows within the bounds, names the first row and field outside them with
    both values, and keeps the largest numeric difference per field, so a
    failure from float noise reads as such instead of as a silent pass or a
    bare fail, and a pass shows what it tolerated."""
    tol = _check_tolerances(tolerances)
    record: Dict[str, Any] = {
        "rows": len(sequential),
        "rows_batched": len(batched),
        "identical": 0,
        "within_tolerance": 0,
        "tolerances": tol,
        "first_difference": None,
        "max_abs_numeric_diff": {},
    }
    if len(sequential) != len(batched):
        record["first_difference"] = {
            "reason": f"{len(sequential)} rows row by row, {len(batched)} rows batched"
        }
        record["ok"] = False
        return record
    for i, (a, b) in enumerate(zip(sequential, batched)):
        keys = sorted(set(a) | set(b))
        for key in keys:
            va, vb = a.get(key), b.get(key)
            if _numeric(va) and _numeric(vb):
                diff = abs(float(va) - float(vb))
                if diff > record["max_abs_numeric_diff"].get(key, 0.0):
                    record["max_abs_numeric_diff"][key] = diff
        if _canonical(a, default) == _canonical(b, default):
            record["identical"] += 1
            continue
        failing = [k for k in keys if not _field_within(a, b, k, tol.get(k), default)]
        if not failing:
            record["within_tolerance"] += 1
            continue
        if record["first_difference"] is None:
            field = failing[0]
            record["first_difference"] = {
                "row": i,
                "example_id": a.get("example_id"),
                "field": field,
                "row_by_row": a.get(field),
                "batched": b.get(field),
            }
    record["ok"] = record["identical"] + record["within_tolerance"] == len(sequential)
    return record


def pick_window(
    run_root: Path,
    n_rows: int,
    *,
    skip_dataset: Callable[[Path], bool],
    row_not_ok: Callable[[Dict[str, Any]], bool],
) -> Tuple[Optional[Path], List[str]]:
    """The first scorable window, in the order the pass scores them, with at
    least ``n_rows`` completed rows (the scan stops there); else the window
    with the most completed rows; ``(None, [])`` when no window has one."""
    best: Tuple[Optional[Path], List[str]] = (None, [])
    for ev_path in sorted((run_root / "cells").glob("*/window_*/qa_evidence.jsonl")):
        if skip_dataset(ev_path):
            continue
        lines = [line for line in ev_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        ok_lines = [line for line in lines if not row_not_ok(json.loads(line))]
        if len(ok_lines) >= n_rows:
            return ev_path, ok_lines[:n_rows]
        if len(ok_lines) > len(best[1]):
            best = (ev_path, ok_lines)
    return best


def _write_sample(scratch: Path, kind: str, ev_path: Path, lines: Sequence[str]) -> Path:
    """The sample as a window file under the scratch root, keeping the cell
    and window directory names (the rescorer labels rows by them)."""
    sample = scratch / kind / "cells" / ev_path.parent.parent.name / ev_path.parent.name / "qa_evidence.jsonl"
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return sample


def run_probe(args: argparse.Namespace) -> Tuple[Dict[str, Any], int]:
    """The comparison; returns (record, exit code). Every failure is a record
    with a reason, never a traceback: the master greps one line."""
    record: Dict[str, Any] = {
        "batch_size": args.batch_size,
        "rows_requested": args.rows,
        "device": args.device,
        "mode": "fast" if args.fast else "full",
        "ok": False,
    }
    try:
        if args.batch_size < 1 or args.rows < 1:
            raise ValueError("--batch-size and --rows must be >= 1")
        # row F-11: the bounds the profile states, checked before any pass runs
        given = {"faithfulness": args.tol_faithfulness, "completeness_bertscore": args.tol_bertscore}
        record["tolerances"] = {k: v for k, v in given.items() if v is not None}
        tol = _check_tolerances(record["tolerances"])
        rq = load_rescorer()
        native = rq._native_only_set(argparse.Namespace(native_only_datasets=args.native_only_datasets))
        ev_path, ok_lines = pick_window(
            args.run_root, args.rows,
            skip_dataset=lambda p: rq._window_dataset(p) in native,
            row_not_ok=rq._row_not_ok,
        )
        if ev_path is None:
            record["reason"] = f"no window under {args.run_root}/cells carries a completed row"
            return record, EXIT_REFUSED
        record["window"] = str(ev_path.parent.relative_to(args.run_root))
        record["rows"] = len(ok_lines)
        if not args.fast:
            if args.freeze_file is None:
                raise ValueError("--freeze-file is required with the full instruments (their revision pins)")
            for flag, field in TOLERATED.items():
                if field not in tol:
                    raise ValueError(f"{flag} is required with the full instruments (row F-11: the profile states the bound; 0 = exact)")
            record["revision_pins"] = rq.apply_revision_pins(rq.revision_pins_from_freeze(args.freeze_file))
        evaluator = rq._build_evaluator(argparse.Namespace(full=not args.fast, device=args.device))
        scratch = args.scratch_dir / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        sample = _write_sample(scratch, "sample", ev_path, ok_lines)
        kw: Dict[str, Any] = dict(
            blinder=None, cache=None,
            scoring_scope=rq.SCOPE_FAST if args.fast else rq.SCOPE_FULL,
        )
        # review 2026-10-23 M1: the warm-up runs both paths over the WHOLE sample
        # (a single warm-up row can abstain and skip three of the four model
        # loads, which then land inside the timed pass); untimed
        rq._score_evidence_file(sample, evaluator, batch_size=None, **kw)
        rq._score_evidence_file(sample, evaluator, batch_size=args.batch_size, **kw)
        t0 = time.perf_counter()
        sequential, _ = rq._score_evidence_file(sample, evaluator, batch_size=None, **kw)
        t_seq = time.perf_counter() - t0
        t0 = time.perf_counter()
        batched, _ = rq._score_evidence_file(sample, evaluator, batch_size=args.batch_size, **kw)
        t_bat = time.perf_counter() - t0
        # review M2: the control, row by row again, untimed; a difference HERE is
        # the device's own call-to-call variation, not batching
        control, _ = rq._score_evidence_file(sample, evaluator, batch_size=None, **kw)
        record.update(compare_rows(sequential, batched, tolerances=tol, default=rq._json_default))
        # the control stays exact: a tolerated drift here could hide inside the same margin as a batching effect
        same_path = compare_rows(sequential, control, default=rq._json_default)
        record["control_identical"] = same_path["ok"]
        record["control_max_abs_numeric_diff"] = same_path["max_abs_numeric_diff"]
        n = len(sequential) or 1
        record.update(
            row_by_row_s=t_seq, batched_s=t_bat,
            row_by_row_s_per_row=t_seq / n, batched_s_per_row=t_bat / n,
            speedup=(t_seq / t_bat) if t_bat > 0 else None,
            instrument_models=rq._instrument_models(evaluator),
            scratch=str(scratch),
        )
        return record, (EXIT_OK if record["ok"] else EXIT_FAILED)
    except Exception as exc:  # noqa: BLE001 -- the one line the master greps must still print
        record["ok"] = False
        record["reason"] = f"{type(exc).__name__}: {exc}"
        return record, EXIT_FAILED


def _bounds_tail(record: Dict[str, Any]) -> str:
    """The bounds the proof ran under (row F-11) and, review 2026-10-23 L1, the
    largest difference per field, on the line the state keeps."""
    tol = record.get("tolerances") or {}
    tail = " tol=" + ",".join(f"{k}:{v:g}" for k, v in sorted(tol.items())) if tol else ""
    diffs = record.get("max_abs_numeric_diff") or {}
    if diffs:
        tail += " max_diff=" + ",".join(f"{k}:{v:.3g}" for k, v in sorted(diffs.items()))
    return tail


def verdict_line(record: Dict[str, Any]) -> str:
    if record.get("ok"):
        line = (
            f"{OK_LINE} rows={record['rows']} batch_size={record['batch_size']} "
            f"row_by_row_s_per_row={record['row_by_row_s_per_row']:.4f} "
            f"batched_s_per_row={record['batched_s_per_row']:.4f} "
            f"speedup={(record.get('speedup') or 0.0):.2f} window={record['window']}"
        )
        # row F-11: under stated bounds the line also says what passed exactly, what
        # passed within them, the bounds and the largest differences; exact mode prints today's line
        if record.get("tolerances"):
            line += f" identical={record.get('identical', 0)} within_tol={record.get('within_tolerance', 0)}" + _bounds_tail(record)
        return line
    reason = record.get("reason")
    if reason is None:
        first = record.get("first_difference") or {}
        if "reason" in first:
            reason = first["reason"]
        else:
            outside = record.get("rows", 0) - record.get("identical", 0) - record.get("within_tolerance", 0)
            reason = (
                f"{outside} of {record.get('rows', 0)} rows differ; "
                f"first: row {first.get('row')} ({first.get('example_id')}) field {first.get('field')}: "
                f"{first.get('row_by_row')!r} row by row vs {first.get('batched')!r} batched"
            )
    tail = _bounds_tail(record)
    control = record.get("control_identical")
    if control is not None:
        tail += f" control_identical={'yes' if control else 'no'}"
    return f"{FAIL_LINE} rows={record.get('rows', 0)} batch_size={record.get('batch_size')} reason={reason}{tail}"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1].strip())
    parser.add_argument("--run-root", type=Path, required=True, help="the sealed run root")
    parser.add_argument("--batch-size", type=int, required=True, help="the profile's SCORING_BATCH_SIZE")
    parser.add_argument("--rows", type=int, default=N_ROWS, help=f"completed answers graded both ways (default {N_ROWS})")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--freeze-file", type=Path, default=None, help="the freeze artifact whose INSTRUMENT_REVISIONS pin the instruments")
    parser.add_argument("--tol-faithfulness", type=float, default=None,
                        help="row F-11: the per-row bound on faithfulness (the profile's SCORING_TOL_FAITHFULNESS; required with the full instruments; 0 = exact)")
    parser.add_argument("--tol-bertscore", type=float, default=None,
                        help="row F-11: the per-row bound on completeness_bertscore (the profile's SCORING_TOL_BERTSCORE; required with the full instruments; 0 = exact)")
    parser.add_argument("--native-only-datasets", default=None, help="windows skipped as the pass skips them (default: the rescorer's)")
    parser.add_argument("--scratch-dir", type=Path, default=DEFAULT_SCRATCH, help="where the sample files land (never the run tree)")
    parser.add_argument("--out", type=Path, default=None, help="evidence JSON record")
    parser.add_argument("--fast", action="store_true", help="the model-free evaluator: a test of the plumbing, never the proof")
    args = parser.parse_args(argv)
    record, code = run_probe(args)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    print(verdict_line(record))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
