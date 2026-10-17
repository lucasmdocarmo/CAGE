"""Tests for scripts/4_analysis/yield_table.py (Spec Block A1, 2026-10-16).

Synthetic run roots only. Every case pins one clause of the Spec Block or one
finding of the 2026-10-16 review: the counts at the SLO boundary, the W16 clock
rule per row (the driver's rule), the RULER G-only record, the HF skip (null TPOT
beside a decode phase), the per-engine missing floor, the whole-pass refusal and
its exit code, the n_timely = 0 window (Y raw 0, no correction), the decode-tokens
column kept on failed rows, the ok flag as the only completion signal, the
window without metrics.json, the request-count check and the summary sections.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "4_analysis" / "yield_table.py"


def _load():
    spec = importlib.util.spec_from_file_location("yield_table_a1", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod  # dataclasses resolve the module through sys.modules (3.13)
    spec.loader.exec_module(mod)
    return mod


yt = _load()

ENGINE = "sglang"
FLOOR_TTFT_S = 0.1
FLOOR_TPOT_S = 0.01
RULER_FLOOR_TTFT_S = 2.0
RULER_FLOOR_TPOT_S = 0.02
SCORING = "full-cuda-1"


def _write_calibration(cal_dir: Path, *, with_ruler: bool = True, engine: str = ENGINE) -> None:
    cal_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "ladder": {"floor": {"ttft_s": FLOOR_TTFT_S, "tpot_s": FLOOR_TPOT_S, "n_requests": 30}},
        "classes": {},
    }
    if with_ruler:
        doc["classes"]["32768"] = {"role": "ruler", "floor": {"ttft_s": RULER_FLOOR_TTFT_S, "tpot_s": RULER_FLOOR_TPOT_S}}
    (cal_dir / f"rungs_S0_{engine}.json").write_text(json.dumps(doc), encoding="utf-8")


def _row(i: int, *, ttft_ms, tpot_ms, ok=True, num_tokens=20, scheduled=True, engine=ENGINE,
         mode="open_loop", scheduled_ms=None, error=None):
    """One request row. ``scheduled`` True copies ttft_ms into ttft_from_scheduled_ms;
    ``scheduled_ms`` sets that clock explicitly; ``mode`` is the row's workload_mode.
    ``record_index`` is the dispatcher's arrival ordinal on an open-loop row and None
    on a closed-loop row, as the runner stamps it (run_experiment.py evidence stamp)."""
    rec = {
        "example_id": f"ex{i}",
        "repeat_index": "0",
        "record_index": i if mode == "open_loop" else None,
        "ok": ok,
        "error": error,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "num_tokens": num_tokens,
        "engine_id": engine,
        "workload_mode": mode,
    }
    if scheduled_ms is not None:
        rec["ttft_from_scheduled_ms"] = scheduled_ms
    else:
        rec["ttft_from_scheduled_ms"] = ttft_ms if scheduled else None
    return rec


def _window(
    run_root: Path,
    row_key: str,
    window: str,
    rows,
    *,
    dataset: str,
    open_loop: bool,
    predicates=None,
    engine: str = ENGINE,
    duration: float = 10.0,
    total_requests=None,
    with_metrics: bool = True,
) -> None:
    wdir = run_root / "cells" / row_key / window
    wdir.mkdir(parents=True, exist_ok=True)
    parts = row_key.split("|")
    cell = {"cellspec": {"arm": parts[0], "engine": engine, "family": parts[6]}, "gpu_count": 1}
    (run_root / "cells" / row_key / "cell.json").write_text(json.dumps(cell), encoding="utf-8")
    metrics = {
        "experiment": {"dataset": dataset},
        "measured_window": {"t_start": 1000.0, "t_end": 1000.0 + duration, "span": "dispatch"},
        "workload": {"open_loop": {"served_rps": 1.0} if open_loop else None},
    }
    if total_requests is not None:
        metrics["performance"] = {"total_requests": total_requests}
    if with_metrics:
        (wdir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (wdir / "regime.json").write_text(json.dumps({"label": "IN_REGIME"}), encoding="utf-8")
    with open(wdir / "requests.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    if predicates is not None:
        pdir = run_root / "predicate" / SCORING / "cells" / row_key / window
        pdir.mkdir(parents=True, exist_ok=True)
        with open(pdir / "predicate.jsonl", "w", encoding="utf-8") as fh:
            for i, verdict in enumerate(predicates):
                # the #127 identity triple of the row it scores (B3: the join is keyed by it)
                row = rows[i] if i < len(rows) else {}
                fh.write(json.dumps({
                    "example_id": row.get("example_id", f"ex{i}"), "repeat_index": "0",
                    "record_index": row.get("record_index"), "predicate": verdict,
                }) + "\n")


KEY_F2 = f"gold-fresh|none|none|single|{ENGINE}|qwen3-14b|F2|r0.75|lam1.05"
KEY_F1 = f"gold-fresh|none|none|single|{ENGINE}|qwen3-14b|F1"


def _run(tmp_path: Path, run_root: Path):
    out = tmp_path / "out"
    rc = yt.main([str(run_root), "--calibration-dir", str(tmp_path / "cal"), "--scoring-run-id", SCORING, "--out", str(out)])
    rows = {}
    if rc == 0:
        with open(out / "yield_table.csv", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                rows[(r["row_key"], r["window_dir"])] = r
    return rc, rows, out


def test_counts_at_the_slo_boundary_and_the_registered_columns(tmp_path: Path) -> None:
    # SLO: ttft <= 10 x 0.1 s = 1000 ms; tpot <= 5 x 0.01 s = 50 ms. Equal counts as timely.
    run_root = tmp_path / "run"
    _write_calibration(tmp_path / "cal")
    rows = [
        _row(0, ttft_ms=1000.0, tpot_ms=50.0),      # on both boundaries: timely
        _row(1, ttft_ms=1000.1, tpot_ms=50.0),      # ttft over: late
        _row(2, ttft_ms=500.0, tpot_ms=50.1),       # tpot over: late
        _row(3, ttft_ms=500.0, tpot_ms=10.0),       # timely
        _row(4, ttft_ms=500.0, tpot_ms=None, num_tokens=1),  # no decode phase: timely on TTFT alone
        _row(5, ttft_ms=None, tpot_ms=None, ok=False),       # not completed
    ]
    predicates = [True, True, True, False, True, None]
    _window(run_root, KEY_F2, "window_qasper-01", rows, dataset="qasper", open_loop=True, predicates=predicates, total_requests=6)
    rc, table, _ = _run(tmp_path, run_root)
    assert rc == 0
    r = table[(KEY_F2, "window_qasper-01")]
    assert r["skip_reason"] == ""
    assert r["ttft_basis"] == "scheduled"
    assert r["duration_basis"] == "dispatch"
    assert int(r["n_issued"]) == 6 and int(r["n_completed"]) == 5
    assert int(r["n_timely"]) == 3            # rows 0, 3, 4
    assert int(r["n_yield"]) == 2             # rows 0 and 4 (row 3 is not veridical)
    assert int(r["n_no_decode"]) == 1
    assert float(r["goodput_frac"]) == pytest.approx(3 / 6)
    assert float(r["yield_frac"]) == pytest.approx(2 / 6)
    assert float(r["goodput_rps"]) == pytest.approx(3 / 10.0)
    assert float(r["slo_ttft_ms"]) == pytest.approx(1000.0)
    assert float(r["slo_tpot_ms"]) == pytest.approx(50.0)
    assert r["baseline_source"].startswith("anchor:")
    assert r["regime_label"] == "IN_REGIME"
    assert set(yt.ALL_COLUMNS) == set(table[(KEY_F2, "window_qasper-01")].keys())


def test_open_loop_row_without_the_scheduled_clock_is_a_named_skip(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _write_calibration(tmp_path / "cal")
    rows = [_row(0, ttft_ms=100.0, tpot_ms=10.0), _row(1, ttft_ms=100.0, tpot_ms=10.0, scheduled=False)]
    _window(run_root, KEY_F2, "window_qasper-01", rows, dataset="qasper", open_loop=True, predicates=[True, True])
    rc, table, _ = _run(tmp_path, run_root)
    assert rc == 0
    r = table[(KEY_F2, "window_qasper-01")]
    assert "ttft_from_scheduled_ms" in r["skip_reason"] and "W16" in r["skip_reason"]
    assert r["n_issued"] == "" and r["goodput_frac"] == ""


def test_closed_loop_window_uses_the_send_clock(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _write_calibration(tmp_path / "cal")
    rows = [_row(i, ttft_ms=100.0, tpot_ms=10.0, scheduled=False, mode="closed_loop") for i in range(4)]
    _window(run_root, KEY_F1, "window_squad_v2-01", rows, dataset="squad_v2", open_loop=False, predicates=[True] * 4)
    rc, table, _ = _run(tmp_path, run_root)
    assert rc == 0
    r = table[(KEY_F1, "window_squad_v2-01")]
    assert r["skip_reason"] == "" and r["ttft_basis"] == "send"
    assert int(r["n_timely"]) == 4 and int(r["n_yield"]) == 4


def test_clock_follows_the_row_not_the_window_flag(tmp_path: Path) -> None:
    # Review 2026-10-16 LOW 4: the driver grades on ttft_from_scheduled_ms whenever the
    # row carries one, whatever the window's workload flag says. A late scheduled clock
    # beside an early send clock must make the row late, and the basis must say so.
    run_root = tmp_path / "run"
    _write_calibration(tmp_path / "cal")
    rows = [
        _row(0, ttft_ms=100.0, tpot_ms=10.0, scheduled_ms=5000.0, mode="closed_loop"),
        _row(1, ttft_ms=100.0, tpot_ms=10.0, scheduled=False, mode="closed_loop"),
    ]
    _window(run_root, KEY_F1, "window_squad_v2-01", rows, dataset="squad_v2", open_loop=False, predicates=[True, True])
    rc, table, _ = _run(tmp_path, run_root)
    assert rc == 0
    r = table[(KEY_F1, "window_squad_v2-01")]
    assert r["skip_reason"] == ""
    assert int(r["n_timely"]) == 1
    assert r["ttft_basis"] == "mixed:scheduled=1,send=1"


def test_ok_flag_alone_decides_completion(tmp_path: Path) -> None:
    # Review 2026-10-16 LOW 5: the driver counts a row completed on ok alone; error text
    # beside ok=True does not demote it.
    run_root = tmp_path / "run"
    _write_calibration(tmp_path / "cal")
    rows = [_row(0, ttft_ms=100.0, tpot_ms=10.0, error="engine warning text"), _row(1, ttft_ms=100.0, tpot_ms=10.0)]
    _window(run_root, KEY_F2, "window_qasper-01", rows, dataset="qasper", open_loop=True, predicates=[True, True])
    rc, table, _ = _run(tmp_path, run_root)
    assert rc == 0
    r = table[(KEY_F2, "window_qasper-01")]
    assert r["skip_reason"] == ""
    assert int(r["n_completed"]) == 2 and int(r["n_timely"]) == 2


def test_failed_row_without_a_token_count_keeps_the_decode_contract(tmp_path: Path) -> None:
    # Review 2026-10-16 HIGH 1: a failed row with no num_tokens (the S1 drain-timeout
    # shape) must not drop the decode_tokens column; with the column gone a completed
    # one-token row with a null TPOT would refuse the whole window (pre-ADR-0118 contract).
    run_root = tmp_path / "run"
    _write_calibration(tmp_path / "cal")
    rows = [
        _row(0, ttft_ms=None, tpot_ms=None, ok=False, num_tokens=None),
        _row(1, ttft_ms=500.0, tpot_ms=None, num_tokens=1),
        _row(2, ttft_ms=500.0, tpot_ms=10.0),
    ]
    _window(run_root, KEY_F2, "window_qasper-01", rows, dataset="qasper", open_loop=True, predicates=[None, True, True])
    rc, table, _ = _run(tmp_path, run_root)
    assert rc == 0
    r = table[(KEY_F2, "window_qasper-01")]
    assert r["skip_reason"] == ""
    assert int(r["n_completed"]) == 2 and int(r["n_timely"]) == 2 and int(r["n_no_decode"]) == 1


def test_ruler_window_carries_g_only_with_the_named_skip(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _write_calibration(tmp_path / "cal")
    # RULER SLO: ttft <= 20 s, tpot <= 100 ms
    rows = [_row(0, ttft_ms=15000.0, tpot_ms=80.0), _row(1, ttft_ms=25000.0, tpot_ms=80.0)]
    _window(run_root, KEY_F2, "window_ruler-01", rows, dataset="ruler", open_loop=True, predicates=None)
    rc, table, _ = _run(tmp_path, run_root)
    assert rc == 0
    r = table[(KEY_F2, "window_ruler-01")]
    assert r["skip_reason"] == yt.RULER_Y_SKIP
    assert r["baseline_source"].startswith("ruler:")
    assert int(r["n_timely"]) == 1 and float(r["goodput_frac"]) == pytest.approx(0.5)
    assert r["n_yield"] == "" and r["yield_frac"] == ""


def test_null_tpot_beside_a_decode_phase_is_a_named_skip(tmp_path: Path) -> None:
    # The HF reference shape: tpot null on completed rows with decode tokens.
    run_root = tmp_path / "run"
    _write_calibration(tmp_path / "cal", engine="hf")
    key = "gold-fresh|none|none|single|hf|qwen3-14b|F1"
    rows = [_row(i, ttft_ms=300.0, tpot_ms=None, num_tokens=20, scheduled=False, engine="hf", mode="closed_loop") for i in range(3)]
    _window(run_root, key, "window_qasper-01", rows, dataset="qasper", open_loop=False, predicates=[True] * 3, engine="hf")
    rc, table, _ = _run(tmp_path, run_root)
    assert rc == 0
    r = table[(key, "window_qasper-01")]
    assert r["skip_reason"].startswith("GoodputError")
    assert r["goodput_frac"] == ""


def test_engine_without_a_floor_is_a_named_skip_and_others_still_record(tmp_path: Path) -> None:
    # The HF reference engine runs no calibration ladder: its windows are named skips,
    # the other engine's windows are records, and no default baseline is substituted.
    run_root = tmp_path / "run"
    _write_calibration(tmp_path / "cal")  # sglang only
    key_hf = "gold-fresh|none|none|single|hf|qwen3-14b|F1"
    _window(run_root, key_hf, "window_qasper-01",
            [_row(0, ttft_ms=300.0, tpot_ms=10.0, scheduled=False, engine="hf", mode="closed_loop")],
            dataset="qasper", open_loop=False, predicates=[True], engine="hf")
    _window(run_root, KEY_F2, "window_qasper-01", [_row(0, ttft_ms=100.0, tpot_ms=10.0)],
            dataset="qasper", open_loop=True, predicates=[True])
    rc, table, _ = _run(tmp_path, run_root)
    assert rc == 0
    hf = table[(key_hf, "window_qasper-01")]
    assert "no section 6.1 anchor floor for engine 'hf'" in hf["skip_reason"]
    assert hf["goodput_frac"] == ""
    assert table[(KEY_F2, "window_qasper-01")]["skip_reason"] == ""


def test_no_calibration_files_refuses_the_whole_pass_with_rc_3(tmp_path: Path) -> None:
    # Review 2026-10-16 MEDIUM 2: the refusal code is 3, distinct from Python's and
    # argparse's 2, so the master cannot mistake a broken call for a refusal.
    run_root = tmp_path / "run"
    (tmp_path / "cal").mkdir()
    rows = [_row(0, ttft_ms=100.0, tpot_ms=10.0)]
    _window(run_root, KEY_F2, "window_qasper-01", rows, dataset="qasper", open_loop=True, predicates=[True])
    rc, table, out = _run(tmp_path, run_root)
    assert rc == yt.REFUSED_RC == 3
    assert not (out / "yield_table.csv").exists()
    with pytest.raises(SystemExit) as usage:
        yt.main([])  # a usage error is argparse's exit 2, never the refusal code
    assert usage.value.code == 2


def test_all_late_window_reports_zero_goodput_and_zero_yield(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _write_calibration(tmp_path / "cal")
    rows = [_row(i, ttft_ms=5000.0, tpot_ms=10.0) for i in range(3)]
    _window(run_root, KEY_F2, "window_qasper-01", rows, dataset="qasper", open_loop=True, predicates=[True] * 3)
    rc, table, _ = _run(tmp_path, run_root)
    assert rc == 0
    r = table[(KEY_F2, "window_qasper-01")]
    assert r["skip_reason"] == ""
    assert int(r["n_timely"]) == 0 and float(r["goodput_frac"]) == 0.0 and float(r["yield_frac"]) == 0.0


def test_missing_predicate_table_is_a_named_skip(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _write_calibration(tmp_path / "cal")
    rows = [_row(0, ttft_ms=100.0, tpot_ms=10.0)]
    _window(run_root, KEY_F2, "window_qasper-01", rows, dataset="qasper", open_loop=True, predicates=None)
    rc, table, _ = _run(tmp_path, run_root)
    assert rc == 0
    assert "no predicate table" in table[(KEY_F2, "window_qasper-01")]["skip_reason"]


def test_window_without_metrics_is_a_named_skip_row_not_an_absence(tmp_path: Path) -> None:
    # Review 2026-10-16 LOW 7: "one record or one named skip per window, never absent".
    run_root = tmp_path / "run"
    _write_calibration(tmp_path / "cal")
    _window(run_root, KEY_F2, "window_qasper-01", [_row(0, ttft_ms=100.0, tpot_ms=10.0)],
            dataset="qasper", open_loop=True, predicates=[True], with_metrics=False)
    rc, table, _ = _run(tmp_path, run_root)
    assert rc == 0
    r = table[(KEY_F2, "window_qasper-01")]
    assert "metrics.json missing" in r["skip_reason"] and r["goodput_frac"] == ""


def test_request_count_disagreeing_with_total_requests_is_a_named_skip(tmp_path: Path) -> None:
    # Review 2026-10-16 LOW 7: n_issued must equal the window's performance.total_requests.
    run_root = tmp_path / "run"
    _write_calibration(tmp_path / "cal")
    _window(run_root, KEY_F2, "window_qasper-01", [_row(0, ttft_ms=100.0, tpot_ms=10.0)],
            dataset="qasper", open_loop=True, predicates=[True], total_requests=2)
    rc, table, _ = _run(tmp_path, run_root)
    assert rc == 0
    r = table[(KEY_F2, "window_qasper-01")]
    assert "holds 1 rows but performance.total_requests is 2" in r["skip_reason"]


def test_summary_holds_group_medians_and_the_f1_reference(tmp_path: Path) -> None:
    # Review 2026-10-16 MEDIUM 3: the Spec Block's yield_summary.md sections.
    run_root = tmp_path / "run"
    _write_calibration(tmp_path / "cal")
    _window(run_root, KEY_F1, "window_squad_v2-01",
            [_row(i, ttft_ms=100.0, tpot_ms=10.0, scheduled=False, mode="closed_loop") for i in range(2)],
            dataset="squad_v2", open_loop=False, predicates=[True, True])
    _window(run_root, KEY_F2, "window_qasper-01", [_row(0, ttft_ms=5000.0, tpot_ms=10.0), _row(1, ttft_ms=100.0, tpot_ms=10.0)],
            dataset="qasper", open_loop=True, predicates=[True, True])
    rc, _, out = _run(tmp_path, run_root)
    assert rc == 0
    md = (out / "yield_summary.md").read_text(encoding="utf-8")
    assert "## Medians per (engine, family, arm, rung, dataset), records only" in md
    assert f"| {ENGINE} | F2 | gold-fresh | 0.75 | qasper | 1 | 0.500 | 0.500 |" in md
    assert "## F1 reference (unpressured windows) per engine" in md
    assert f"| {ENGINE} | 1 | 1.000 | 1.000 |" in md
