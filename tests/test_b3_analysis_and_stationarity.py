"""B3 slices 3 and 4 (ADR-0172 items 1 and 6 i to iii, 2026-10-16).

The first-pass rule in the analysis loader and the conditioned curves (item 1),
the stationarity record the regime writer emits and the truth-tax population
reading it (item 6 ii, B17), the time-resolved goodput, queue growth rate and
time to breach in the yield table (item 6 i, B16), the lambda* consistency flag
(item 6 iii, B18), and the yield table's verdict join keyed by the identity
triple. Each test fails on the code before the change it names.
"""

from __future__ import annotations

import csv
import json
import sys
import types
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "scripts" / "4_analysis", Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_campaign_analysis as rca  # noqa: E402
from src.analysis.goodput import IN_REGIME, UNPRESSURED  # noqa: E402
from src.orchestration import campaign_layout as cl  # noqa: E402
from test_campaign_layout import RUN_ID, _add_window, _manifest_kwargs, _specs  # noqa: E402
from test_yield_table import (  # noqa: E402
    ENGINE,
    KEY_F1,
    KEY_F2,
    SCORING,
    _row,
    _run,
    _write_calibration,
    yt,
)

REPLAY_REASON = "replayed arrival: quality joins the first pass only (ADR-0172 item 1)"


# --------------------------------------------------------------------------- #
# B17: the stationarity record (campaign_layout.write_window_regime)
# --------------------------------------------------------------------------- #


def _dense_window(
    tmp_path: Path,
    *,
    kv=(0.5, 0.9, 0.9),
    ttft=(100.0, 100.0, 105.0),
    rows_per_third=(10, 10, 10),
    ok_per_third=(True, True, True),
    span: float = 90.0,
    telemetry_until: float | None = None,
    roles: tuple[str, ...] | None = None,
    window_end: float | None = None,
    drain_kv: float = 0.2,
) -> dict[str, Any]:
    """One window of ``span`` seconds of arrivals: telemetry every second with
    the KV occupancy of each third, completed arrivals spread over each third
    with that third's TTFT, a failed row at the last arrival pinning the span,
    then the regime writer. ``window_end`` past ``span`` adds a drain tail at
    ``drain_kv`` (the writer closes the window after the dispatcher returns).
    ``roles`` tags every sample per role (the PD lane)."""
    run_root = tmp_path / "results" / "camp1" / "a" / RUN_ID
    run = cl.CampaignRun.create(run_root, **_manifest_kwargs())
    cell = run.cell(_specs()[0])
    third = span / 3.0
    t_end = span if window_end is None else window_end
    telemetry: list[dict[str, Any]] = []
    until = t_end if telemetry_until is None else telemetry_until
    t = 0.5
    while t < until:
        k = min(int(t // third), 2)
        for role in roles or (None,):
            sample: dict[str, Any] = {
                "ts_s": t, "kv_cache_usage": kv[k] if t < span else drain_kv,
                "preemptions_total": 0, "waiting": 1,
            }
            if role is not None:
                sample["instance"] = role
            telemetry.append(sample)
        t += 1.0
    rows: list[dict[str, Any]] = []
    for k in range(3):
        n = rows_per_third[k]
        for j in range(n):
            arrival = k * third + (j + 0.5) * third / max(n, 1)
            rows.append(
                {
                    "example_id": f"e{k}-{j}", "ok": ok_per_third[k], "arrival_s": arrival,
                    "ttft_from_scheduled_ms": ttft[k], "ttft_ms": ttft[k] - 1.0,
                    "record_index": len(rows),
                }
            )
    # the last arrival (a failed row: it bounds the span, it has no TTFT)
    rows.append({"example_id": "pin", "ok": False, "arrival_s": span, "record_index": len(rows)})
    handle = _add_window(
        cell, "squad_v2", t_start=0.0, t_end=t_end, cage_stats=telemetry, requests=rows,
    )
    kwargs: dict[str, Any] = {}
    if roles:
        kwargs["role_budgets"] = {role: 100 for role in roles}
    path = cl.write_window_regime(
        handle.window_dir, t_start=0.0, t_end=t_end, attainment=0.95, **kwargs,
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_stationarity_constants_and_record_shape(tmp_path: Path) -> None:
    assert cl.STATIONARITY_BAND == 0.10 and cl.STATIONARITY_MIN_ROWS == 3
    doc = _dense_window(tmp_path)
    assert doc["schema_version"] == 2  # additive key, schema unchanged
    assert doc["label"] == IN_REGIME
    st = doc["stationarity"]
    assert st["adr"] == "ADR-0172" and st["band"] == 0.10 and st["min_rows"] == 3
    assert set(st) == {
        "adr", "rule", "band", "min_rows", "label", "span_s", "span_basis", "kv", "ttft_ms", "reasons",
    }
    assert st["span_s"] == 90.0 and st["span_basis"] == "window bounds"  # last arrival at the bound
    assert set(st["kv"]) == {"second", "third", "relative_difference"}
    assert set(st["ttft_ms"]) == {"second", "third", "relative_difference", "n_second", "n_third"}


def test_a_steady_window_reads_stationary(tmp_path: Path) -> None:
    st = _dense_window(tmp_path)["stationarity"]
    assert st["label"] == "stationary" and st["reasons"] == []
    assert st["kv"]["second"] == pytest.approx(0.9) and st["kv"]["third"] == pytest.approx(0.9)
    assert st["kv"]["relative_difference"] == pytest.approx(0.0)
    assert st["ttft_ms"]["second"] == 100.0 and st["ttft_ms"]["third"] == 105.0
    assert st["ttft_ms"]["relative_difference"] == pytest.approx(5.0 / 105.0)
    assert st["ttft_ms"]["n_second"] == 10 and st["ttft_ms"]["n_third"] == 10


def test_a_kv_ramp_between_the_thirds_reads_transient(tmp_path: Path) -> None:
    st = _dense_window(tmp_path, kv=(0.5, 0.6, 0.9))["stationarity"]
    assert st["label"] == "transient"
    assert st["kv"]["relative_difference"] == pytest.approx(0.3 / 0.9)
    assert st["ttft_ms"]["relative_difference"] == pytest.approx(5.0 / 105.0)


def test_a_ttft_drift_between_the_thirds_reads_transient(tmp_path: Path) -> None:
    st = _dense_window(tmp_path, ttft=(100.0, 100.0, 150.0))["stationarity"]
    assert st["label"] == "transient"
    assert st["ttft_ms"]["relative_difference"] == pytest.approx(50.0 / 150.0)
    assert st["kv"]["relative_difference"] == pytest.approx(0.0)


def test_the_band_is_inclusive(tmp_path: Path) -> None:
    # 110 vs 100 is exactly 0.10: stationary; 111 vs 100 is 0.11: transient
    assert _dense_window(tmp_path / "a", ttft=(100.0, 110.0, 100.0))["stationarity"]["label"] == "stationary"
    assert _dense_window(tmp_path / "b", ttft=(100.0, 111.0, 100.0))["stationarity"]["label"] == "transient"


def test_too_few_completed_arrivals_in_a_third_read_unknown(tmp_path: Path) -> None:
    st = _dense_window(tmp_path, rows_per_third=(10, 10, 2))["stationarity"]
    assert st["label"] == "unknown"
    assert st["ttft_ms"]["third"] is None and st["ttft_ms"]["n_third"] == 2
    assert st["ttft_ms"]["second"] == 100.0
    assert any("2 completed arrival(s) in the third third, fewer than 3" in r for r in st["reasons"])
    # the KV side still reads: absence on one signal never fabricates the label
    assert st["kv"]["relative_difference"] == pytest.approx(0.0)


def test_failed_arrivals_do_not_count_toward_a_median(tmp_path: Path) -> None:
    st = _dense_window(tmp_path, ok_per_third=(True, True, False))["stationarity"]
    assert st["label"] == "unknown" and st["ttft_ms"]["n_third"] == 0


def test_an_uncertified_third_reads_unknown_with_the_reason(tmp_path: Path) -> None:
    # telemetry stops at 50 s: the third third [60, 90) has no sample
    st = _dense_window(tmp_path, telemetry_until=50.0)["stationarity"]
    assert st["label"] == "unknown"
    assert st["kv"]["third"] is None and st["kv"]["second"] == pytest.approx(0.9)
    assert any(r.startswith("KV occupancy of the third third not certified") for r in st["reasons"])


def test_a_refused_window_carries_an_unknown_record(tmp_path: Path) -> None:
    run_root = tmp_path / "results" / "camp1" / "a" / RUN_ID
    run = cl.CampaignRun.create(run_root, **_manifest_kwargs())
    handle = _add_window(run.cell(_specs()[0]), "squad_v2", cage_stats=[])
    doc = json.loads(cl.write_window_regime(handle.window_dir, t_start=0.0, t_end=10.0).read_text())
    assert doc["telemetry_ok"] is False
    assert doc["stationarity"]["label"] == "unknown" and doc["stationarity"]["reasons"]


def test_the_span_ends_at_the_last_arrival_not_at_the_drain(tmp_path: Path) -> None:
    # fresh review 2026-10-16 (HIGH, the code part): the writer closes the
    # window after the dispatcher returns, so [t_start, t_end) holds the
    # service tail and the drain, during which the pool empties; the thirds
    # are placed over the arrivals (0 to 90 s), never over the 120 s bound
    doc = _dense_window(tmp_path, window_end=120.0, drain_kv=0.2)
    assert doc["t_end"] == 120.0
    st = doc["stationarity"]
    assert st["span_s"] == 90.0 and st["span_basis"] == "last arrival_s"
    assert st["label"] == "stationary"
    assert st["kv"]["third"] == pytest.approx(0.9)  # not diluted by the 0.2 drain
    assert st["ttft_ms"]["n_third"] == 10


def test_the_pd_lane_records_stationarity_through_the_pooled_certifier(tmp_path: Path) -> None:
    doc = _dense_window(tmp_path, roles=("prefill", "decode"))
    assert "pd" in doc and doc["pd"]["budgets_by_role"] == {"decode": 100, "prefill": 100}
    st = doc["stationarity"]
    assert st["label"] == "stationary"
    assert st["kv"]["second"] == pytest.approx(0.9) and st["kv"]["third"] == pytest.approx(0.9)


def test_stationarity_record_is_pure_on_its_certifier(tmp_path: Path) -> None:
    (tmp_path / "requests.jsonl").write_text("", encoding="utf-8")
    calls: list[tuple[float, float]] = []

    def certify(lo: float, hi: float) -> float:
        calls.append((lo, hi))
        return 0.9

    rec = cl.stationarity_record(tmp_path, t_start=30.0, t_end=120.0, certify=certify)
    assert calls == [(60.0, 90.0), (90.0, 120.0)]
    assert rec["label"] == "unknown"  # no completed arrivals: no TTFT medians
    assert rec["kv"]["second"] == 0.9 and rec["kv"]["relative_difference"] == 0.0


# --------------------------------------------------------------------------- #
# The truth-tax population reads the label (item 6 ii, the consumer)
# --------------------------------------------------------------------------- #


def _rec(window: str = "cells/k/window_qasper-01") -> Any:
    return types.SimpleNamespace(
        window_dir=window, engine="vllm", cell_json="cells/k/cell.json", window_key="qasper-01",
    )


def test_truth_tax_excludes_a_transient_window_by_name(tmp_path: Path) -> None:
    wdir = tmp_path / "cells" / "k" / "window_qasper-01"
    wdir.mkdir(parents=True)
    (wdir / "regime.json").write_text(
        json.dumps({"label": IN_REGIME, "stationarity": {"label": "transient"}}), encoding="utf-8",
    )
    metrics, reason, clock = rca._window_truth_tax(tmp_path, _rec(), tmp_path / "pred", {})
    assert metrics is None and clock is None
    assert "transient" in reason and "ADR-0172" in reason
    # a stationary (or pre-B17) window passes the gate and reaches the floor check
    for stationarity in ({"label": "stationary"}, None):
        doc: dict[str, Any] = {"label": IN_REGIME}
        if stationarity is not None:
            doc["stationarity"] = stationarity
        (wdir / "regime.json").write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(rca.AnalysisError, match="floor"):
            rca._window_truth_tax(tmp_path, _rec(), tmp_path / "pred", {})


# --------------------------------------------------------------------------- #
# Item 1: the first-pass rule in the loader and the conditioned curves
# --------------------------------------------------------------------------- #


KEY = "gold-fresh|none|none|single|vllm|qwen3-14b|F2|r1|lam1.2"


def _replayed_run(
    tmp_path: Path, *, recorded_bound: int | None = 2, passes: int = 2
) -> tuple[Path, pd.DataFrame]:
    """One window of 3 ids over ``passes`` passes: pass 1 answers well (f1 1.0,
    grounding 1.0, TTFT 100), later passes badly (f1 0.0, grounding 0.0, TTFT 300)."""
    run_dir = tmp_path / "run"
    wdir = run_dir / "cells" / KEY / "window_qasper-01"
    wdir.mkdir(parents=True)
    requests, evidence = [], []
    for ri in range(3 * passes):
        eid = f"q{ri % 3}"
        first = ri < 3
        base: dict[str, Any] = {
            "example_id": eid, "repeat_index": 0, "ok": True, "workload_mode": "open_loop",
            "record_index": ri, "arrival_s": float(ri),
        }
        requests.append(
            {**base, "ttft_ms": 100.0 if first else 300.0, "ttft_from_scheduled_ms": 100.0 if first else 300.0,
             "tpot_ms": 10.0, "num_tokens": 5, "f1_score": 1.0 if first else 0.0}
        )
        evidence.append({**base, "grounding_score": 1.0 if first else 0.0, "generated_answer": "x"})
    (wdir / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in requests), encoding="utf-8")
    (wdir / "qa_evidence.jsonl").write_text("".join(json.dumps(r) + "\n" for r in evidence), encoding="utf-8")
    if recorded_bound is not None:
        (wdir / "metrics.json").write_text(
            json.dumps({"workload": {"open_loop": {"replay": {"first_pass_record_index_max": recorded_bound}}}}),
            encoding="utf-8",
        )
    index = pd.DataFrame(
        [{"row_key": KEY, "dataset": "qasper", "window_key": "qasper-01",
          "window_dir": f"cells/{KEY}/window_qasper-01", "budget_r": 1.0, "rate_frac": 1.2}]
    )
    return run_dir, index


def test_first_pass_bound_is_derived_and_cross_checked(tmp_path: Path) -> None:
    rows = [{"example_id": f"q{i % 3}", "record_index": i} for i in range(6)]
    assert rca.first_pass_record_index_max(rows) == 2
    assert rca.first_pass_record_index_max(rows[:2]) == 1  # fewer arrivals than ids: nothing replays
    assert rca.first_pass_record_index_max([{"example_id": "a"}]) is None  # closed loop
    assert rca.first_pass_record_index_max([]) is None
    run_dir, _ = _replayed_run(tmp_path, recorded_bound=2)
    assert rca.first_pass_record_index_max(rows, run_dir / "cells" / KEY / "window_qasper-01") == 2
    run_dir, _ = _replayed_run(tmp_path / "bad", recorded_bound=1)
    with pytest.raises(rca.AnalysisError, match="recorded first_pass_record_index_max=1 disagrees"):
        rca.first_pass_record_index_max(rows, run_dir / "cells" / KEY / "window_qasper-01")
    run_dir, _ = _replayed_run(tmp_path / "none", recorded_bound=None)  # a pre-B3 window
    assert rca.first_pass_record_index_max(rows, run_dir / "cells" / KEY / "window_qasper-01") == 2
    assert {"f1_score", "grounding_score", "predicate", "exact_match", "ruler_native_score"} <= rca.QUALITY_FIRST_PASS_COLUMNS
    assert not {"ttft_ms", "tpot_ms", "latency_ms", "goodput_frac"} & rca.QUALITY_FIRST_PASS_COLUMNS


def test_loader_keeps_every_arrival_for_serving_and_the_first_pass_for_quality(tmp_path: Path) -> None:
    run_dir, index = _replayed_run(tmp_path)
    frame = rca.load_per_query(run_dir, index, {KEY})
    assert len(frame) == 3
    assert frame["ttft_ms"].tolist() == pytest.approx([200.0, 200.0, 200.0])  # both passes
    assert frame["f1_score"].tolist() == [1.0, 1.0, 1.0]  # first pass only
    assert frame["grounding_score"].tolist() == [1.0, 1.0, 1.0]
    assert frame.attrs["replay_quality_rows_dropped"] == 6  # 3 f1 + 3 grounding
    assert frame.attrs["replay_windows"] == 1
    assert frame.attrs["quality_first_pass_rule"] == rca.QUALITY_FIRST_PASS_RULE


def test_loader_refuses_a_replay_record_that_contradicts_the_rows(tmp_path: Path) -> None:
    run_dir, index = _replayed_run(tmp_path, recorded_bound=4)
    with pytest.raises(rca.AnalysisError, match="disagrees with requests.jsonl"):
        rca.load_per_query(run_dir, index, {KEY})


def test_a_single_pass_window_drops_nothing(tmp_path: Path) -> None:
    run_dir, index = _replayed_run(tmp_path, recorded_bound=2, passes=1)
    frame = rca.load_per_query(run_dir, index, {KEY})
    assert frame.attrs["replay_quality_rows_dropped"] == 0 and frame.attrs["replay_windows"] == 0
    assert frame["f1_score"].tolist() == [1.0, 1.0, 1.0] and frame["ttft_ms"].tolist() == [100.0, 100.0, 100.0]


def test_conditioned_curves_skip_replayed_trials_by_name(tmp_path: Path) -> None:
    run_dir, index = _replayed_run(tmp_path)
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    doc = rca.run_conditioned_curves_pass(run_dir, index, analysis_dir, "DESIGN-INPUT-ONLY", blinding_active=False)
    assert doc is not None
    assert doc["quality_vs_evidence_position"]["skipped_trials"][REPLAY_REASON] == 3
    assert doc["quality_vs_rho_own"]["replay_rule"] == rca.QUALITY_FIRST_PASS_RULE
    # the rho leg skips for want of own_accounting (no artifact in this run) on
    # the metric the request rows carry; grounding rides the evidence file here
    skips = {s["metric"]: s["reason"] for s in doc["quality_vs_rho_own"]["skips"]}
    assert "own_accounting" in skips["f1_score"]
    assert "absent from every requests.jsonl row" in skips["grounding_score"]


# --------------------------------------------------------------------------- #
# The yield table: the keyed join, B16 and B18
# --------------------------------------------------------------------------- #


def _yt_window(
    run_root: Path, row_key: str, window: str, rows: list[dict[str, Any]], *, dataset: str,
    engine: str = ENGINE, duration: float = 10.0, predicate_rows: list[dict[str, Any]] | None = None,
    regime: dict[str, Any] | None = None, cage_stats: list[dict[str, Any]] | None = None,
) -> None:
    wdir = run_root / "cells" / row_key / window
    wdir.mkdir(parents=True, exist_ok=True)
    parts = row_key.split("|")
    (run_root / "cells" / row_key / "cell.json").write_text(
        json.dumps({"cellspec": {"arm": parts[0], "engine": engine, "family": parts[6]}, "gpu_count": 1}), encoding="utf-8",
    )
    metrics = {
        "experiment": {"dataset": dataset},
        "measured_window": {"t_start": 1000.0, "t_end": 1000.0 + duration, "span": "dispatch"},
        "performance": {"total_requests": len(rows)},
    }
    (wdir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (wdir / "regime.json").write_text(json.dumps(regime if regime is not None else {"label": IN_REGIME}), encoding="utf-8")
    (wdir / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    if cage_stats is not None:
        (wdir / "cage_stats.jsonl").write_text("".join(json.dumps(s) + "\n" for s in cage_stats), encoding="utf-8")
    if predicate_rows is not None:
        pdir = run_root / "predicate" / SCORING / "cells" / row_key / window
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "predicate.jsonl").write_text("".join(json.dumps(p) + "\n" for p in predicate_rows), encoding="utf-8")


def _arrival_row(i: int, arrival: float, *, ttft_ms: float, example_id: str | None = None, record_index: int | None = None) -> dict[str, Any]:
    rec = _row(i, ttft_ms=ttft_ms, tpot_ms=10.0)
    rec["arrival_s"] = arrival
    rec["record_index"] = i if record_index is None else record_index
    if example_id is not None:
        rec["example_id"] = example_id
    return rec


def _verdicts(rows: list[dict[str, Any]], value: bool = True) -> list[dict[str, Any]]:
    return [{"example_id": r["example_id"], "repeat_index": "0", "record_index": r["record_index"], "predicate": value} for r in rows]


def _slices(out: Path) -> list[dict[str, str]]:
    with open(out / "yield_slices.csv", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_a_replayed_arrival_needs_its_own_verdict(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _write_calibration(tmp_path / "cal")
    rows = [
        _arrival_row(0, 1.0, ttft_ms=100.0, example_id="ex0"),
        _arrival_row(1, 2.0, ttft_ms=100.0, example_id="ex1"),
        _arrival_row(2, 3.0, ttft_ms=100.0, example_id="ex0"),  # the replay of ex0
    ]
    # verdicts for the first pass only: the pre-B3 map gave the replay ex0's
    # verdict silently; now the window is a named skip
    _yt_window(run_root, KEY_F2, "window_qasper-01", rows, dataset="qasper", predicate_rows=_verdicts(rows[:2]))
    # the same window with every arrival's own verdict: a record
    _yt_window(run_root, KEY_F2, "window_qasper-02", rows, dataset="qasper", predicate_rows=_verdicts(rows))
    rc, table, _ = _run(tmp_path, run_root)
    assert rc == 0
    skipped = table[(KEY_F2, "window_qasper-01")]
    assert "veridical" in skipped["skip_reason"] and skipped["goodput_frac"] == ""
    record = table[(KEY_F2, "window_qasper-02")]
    assert record["skip_reason"] == "" and int(record["n_issued"]) == 3 and int(record["n_yield"]) == 3


def test_time_resolved_slices_breach_and_queue_growth(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _write_calibration(tmp_path / "cal")
    # SLO ttft <= 1000 ms: arrivals at 1 s and 2 s are timely, 6, 7 and 8 s are late
    rows = [_arrival_row(0, 1.0, ttft_ms=100.0), _arrival_row(1, 2.0, ttft_ms=100.0),
            _arrival_row(2, 6.0, ttft_ms=5000.0), _arrival_row(3, 7.0, ttft_ms=5000.0), _arrival_row(4, 8.0, ttft_ms=5000.0)]
    regime = {"label": IN_REGIME, "t_start": 1000.0, "t_end": 1010.0}
    stats = [{"ts_s": 1001.0, "waiting": 0}, {"ts_s": 1005.0, "waiting": 2}, {"ts_s": 1009.0, "waiting": 4}, {"ts_s": 1011.0, "waiting": 99}]
    _yt_window(run_root, KEY_F2, "window_qasper-01", rows, dataset="qasper", predicate_rows=_verdicts(rows), regime=regime, cage_stats=stats)
    rc, table, out = _run(tmp_path, run_root)
    assert rc == 0
    r = table[(KEY_F2, "window_qasper-01")]
    assert r["skip_reason"] == ""
    assert r["rate_frac"] == "1.05" and r["stationarity"] == ""
    assert int(r["n_slices"]) == 2
    assert float(r["time_to_breach_s"]) == 10.0  # running goodput 2/5 after the second slice
    assert float(r["queue_growth_rps"]) == pytest.approx(0.5)  # the 1011 s sample is outside the window
    assert r["time_note"] == ""
    slices = _slices(out)
    assert len(slices) == 2
    first, second = slices
    assert first["slice_index"] == "0" and float(first["t0_s"]) == 0.0 and float(first["t1_s"]) == 5.0
    assert int(first["n_issued"]) == 2 and int(first["n_timely"]) == 2
    assert float(first["goodput_frac"]) == 1.0 and float(first["running_goodput_frac"]) == 1.0
    assert int(second["n_issued"]) == 3 and int(second["n_timely"]) == 0
    assert float(second["goodput_frac"]) == 0.0 and float(second["running_goodput_frac"]) == pytest.approx(0.4)
    md = (out / "yield_summary.md").read_text(encoding="utf-8")
    assert "yield_slices.csv" in md and "1 window(s) breached" in md


def test_a_window_that_never_breaches_and_a_closed_loop_window(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _write_calibration(tmp_path / "cal")
    rows = [_arrival_row(i, 1.0 + i, ttft_ms=100.0) for i in range(4)] + [_arrival_row(4, 9.0, ttft_ms=5000.0)]
    _yt_window(run_root, KEY_F2, "window_qasper-01", rows, dataset="qasper", predicate_rows=_verdicts(rows))
    closed = [_row(i, ttft_ms=100.0, tpot_ms=10.0, scheduled=False, mode="closed_loop") for i in range(3)]
    for c in closed:
        c["record_index"] = None
    _yt_window(run_root, KEY_F1, "window_squad_v2-01", closed, dataset="squad_v2",
               predicate_rows=[{"example_id": c["example_id"], "repeat_index": "0", "record_index": None, "predicate": True} for c in closed])
    rc, table, out = _run(tmp_path, run_root)
    assert rc == 0
    open_loop = table[(KEY_F2, "window_qasper-01")]
    assert open_loop["skip_reason"] == "" and open_loop["time_to_breach_s"] == "" and int(open_loop["n_slices"]) == 2
    assert "regime.json without t_start/t_end" in open_loop["time_note"]
    f1 = table[(KEY_F1, "window_squad_v2-01")]
    assert f1["skip_reason"] == "" and f1["n_slices"] == "" and "closed loop" in f1["time_note"]
    assert f1["rate_frac"] == ""
    assert {s["window_dir"] for s in _slices(out)} == {"window_qasper-01"}


def test_a_skipped_window_blanks_the_time_columns(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _write_calibration(tmp_path / "cal")
    rows = [_arrival_row(0, 1.0, ttft_ms=100.0)]
    _yt_window(run_root, KEY_F2, "window_qasper-01", rows, dataset="qasper", predicate_rows=None)
    rc, table, out = _run(tmp_path, run_root)
    assert rc == 0
    r = table[(KEY_F2, "window_qasper-01")]
    assert "no predicate table" in r["skip_reason"]
    assert all(r[c] == "" for c in yt.TIME_COLUMNS) and _slices(out) == []


def test_the_stationarity_column_reads_the_writer_label(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _write_calibration(tmp_path / "cal")
    rows = [_arrival_row(0, 1.0, ttft_ms=100.0)]
    _yt_window(run_root, KEY_F2, "window_qasper-01", rows, dataset="qasper", predicate_rows=_verdicts(rows),
               regime={"label": IN_REGIME, "stationarity": {"label": "transient"}})
    rc, table, out = _run(tmp_path, run_root)
    assert rc == 0
    assert table[(KEY_F2, "window_qasper-01")]["stationarity"] == "transient"
    assert "| IN_REGIME | transient |" in (out / "yield_summary.md").read_text(encoding="utf-8")


def test_lambda_star_consistency_flags_and_the_two_engine_line(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _write_calibration(tmp_path / "cal")
    _write_calibration(tmp_path / "cal", engine="vllm")
    key_95 = KEY_F2.replace("lam1.05", "lam0.95")
    key_95_vllm = key_95.replace(ENGINE, "vllm")
    key_105_vllm = KEY_F2.replace(ENGINE, "vllm")
    rows = [_arrival_row(0, 1.0, ttft_ms=100.0)]
    # sglang: IN_REGIME at 0.95 (two windows) -> flagged; 1.05 IN_REGIME
    _yt_window(run_root, key_95, "window_qasper-01", rows, dataset="qasper", predicate_rows=_verdicts(rows))
    _yt_window(run_root, key_95, "window_qasper-02", rows, dataset="qasper", predicate_rows=_verdicts(rows))
    _yt_window(run_root, KEY_F2, "window_qasper-01", rows, dataset="qasper", predicate_rows=_verdicts(rows))
    # vllm: UNPRESSURED at 1.05 -> flagged; 0.95 UNPRESSURED
    _yt_window(run_root, key_95_vllm, "window_qasper-01", rows, dataset="qasper", engine="vllm",
               predicate_rows=_verdicts(rows), regime={"label": UNPRESSURED})
    _yt_window(run_root, key_105_vllm, "window_qasper-01", rows, dataset="qasper", engine="vllm",
               predicate_rows=_verdicts(rows), regime={"label": UNPRESSURED})
    rc, table, out = _run(tmp_path, run_root)
    assert rc == 0
    md = (out / "yield_summary.md").read_text(encoding="utf-8")
    assert "## lambda* consistency per (engine, arm, rung, dataset) (B18, ADR-0172 6 iii)" in md
    assert f"| {ENGINE} | gold-fresh | 0.75 | qasper | IN_REGIME (2) | IN_REGIME (1) | FLAGGED: IN_REGIME at 0.95 (lambda* reads too high) |" in md
    assert "| vllm | gold-fresh | 0.75 | qasper | UNPRESSURED (1) | UNPRESSURED (1) | FLAGGED: UNPRESSURED at 1.05 (lambda* reads too low) |" in md
    assert "- class gold-fresh r0.75 qasper flagged on 2 engines (sglang, vllm)" in md


def test_lambda_star_consistency_rule_on_plain_rows() -> None:
    def row(engine: str, rate: str, label: str) -> dict[str, Any]:
        return {"engine": engine, "arm": "gold-fresh", "rung": "1", "dataset": "qasper", "rate_frac": rate, "regime_label": label}

    consistent = yt.lambda_star_consistency([row("vllm", "0.95", UNPRESSURED), row("vllm", "1.05", IN_REGIME)])
    assert consistent[0]["verdict"] == "consistent"
    tie = yt.lambda_star_consistency([row("vllm", "0.95", UNPRESSURED), row("vllm", "0.95", IN_REGIME), row("vllm", "1.05", IN_REGIME)])
    assert tie[0]["label_0.95"] == "ambiguous" and tie[0]["verdict"] == "unknown"
    missing = yt.lambda_star_consistency([row("vllm", "1.05", IN_REGIME)])
    assert missing[0]["label_0.95"] == "missing" and missing[0]["verdict"] == "unknown"
    unknown_tel = yt.lambda_star_consistency([row("vllm", "0.95", "UNKNOWN_TELEMETRY"), row("vllm", "1.05", UNPRESSURED)])
    assert unknown_tel[0]["verdict"].startswith("FLAGGED: UNPRESSURED at 1.05")  # a flag outranks an unknown leg
    assert yt.lambda_star_consistency([row("vllm", "1.2", IN_REGIME)]) == []  # other rates are not read
    assert yt.ALL_COLUMNS[-1] == "skip_reason" and "rate_frac" in yt.ID_COLUMNS and "stationarity" in yt.ID_COLUMNS
