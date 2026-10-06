"""render_window_panels.py: per-window telemetry panels + the resource index.

Fixture: a synthetic layout-v2 run with one cell and two windows. Window 01
has a wall clock (strictly increasing ``ts_s``), a GPU sub-record on every
sample and the runner's energy delta; window 02 reproduces the S0 tree
(``ts_s`` = 1.0 on every sample, S0F-15; ``gpu.available`` false; no energy
delta) so the index-mode and the "not recorded" paths are pinned on the data
shape that exists. One malformed JSONL line is planted to pin the skip-and-
count rule. Numbers are synthetic and small so every derived column is
checked by hand arithmetic in the assertions.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = REPO_ROOT / "scripts" / "4_analysis"
for _p in (str(_SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import render_window_panels as rwp  # noqa: E402

ROW_KEY = "retr-fresh|rerank|none|single|vllm|qwen3-14b|F2"
T0 = 1_790_790_000.0


def _jsonl(path: Path, rows, *, raw_lines=()) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
        for line in raw_lines:
            fh.write(line + "\n")


def _sample(ts: float, *, kv: float, running: int, gpu_ok: bool, energy_mj: float | None) -> dict:
    rec = {
        "ts": ts, "ts_s": ts, "kv_cache_usage": kv, "kv_usage": kv, "kv_used_tokens": int(kv * 1000),
        "kv_capacity_tokens": 1000, "running": running, "waiting": 1, "preempt_rate": 0.5,
        "preemptions_total": 3.0, "gen_tps": 40.0, "prompt_tps": 400.0, "req_rate": 2.0,
        "energy_mj": energy_mj,
        "gpu": ({"available": True, "source": "nvml",
                 "gpus": [{"index": 0, "name": "fake", "util_gpu": 90.0, "power_w": 300.0,
                           "mem_used": 1, "mem_total": 2}], "error": None}
                if gpu_ok else {"available": False, "source": "none", "gpus": [], "error": None}),
    }
    return rec


def _request(ok: bool, num_tokens: int, prompt_tokens: int) -> dict:
    return {"ok": ok, "error": None if ok else "timeout", "num_tokens": num_tokens if ok else 0,
            "prompt_tokens": prompt_tokens, "finish_reason": "stop" if ok else "error"}


def _write_window(wdir: Path, *, t_start: float, t_end: float, samples, requests, label: str,
                  telemetry_ok: bool, energy_delta_mj, gpu_block, raw_lines=()) -> None:
    wdir.mkdir(parents=True)
    _jsonl(wdir / "cage_stats.jsonl", samples, raw_lines=raw_lines)
    _jsonl(wdir / "requests.jsonl", requests)
    (wdir / "regime.json").write_text(json.dumps({
        "schema_version": 1, "label": label, "attainment": 1.0, "t_start": t_start, "t_end": t_end,
        "telemetry_ok": telemetry_ok, "telemetry_source": "cage_stats.jsonl", "inputs": None,
        "refusal_reason": None if telemetry_ok else "too few samples",
    }))
    metrics = {
        "measured_window": {"span": "dispatch", "t_start": t_start, "t_end": t_end,
                            "stage_t_start": t_start - 5, "stage_t_end": t_end + 1, "workload_mode": "open_loop"},
        "gpu": gpu_block,
        "vllm_telemetry": ({"energy_delta_mj": energy_delta_mj} if energy_delta_mj is not None else {}),
    }
    (wdir / "metrics.json").write_text(json.dumps(metrics))


@pytest.fixture()
def run_root(tmp_path: Path) -> Path:
    root = tmp_path / "results" / "camp1" / "a" / "20260814-0900-a-qwen3-14b"
    (root / "cells").mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({"run_id": root.name, "campaign": "camp1", "session": "a"}))
    cell = root / "cells" / ROW_KEY
    cell.mkdir()
    (cell / "cell.json").write_text(json.dumps({
        "baseline": "B6", "gpu_count": 2,
        "cellspec": {"arm": "retr-fresh", "retriever": "rerank", "policy": "none", "topology": "single",
                     "engine": "vllm", "model": "qwen3-14b", "family": "F2", "budget_r": 0.5, "rate_frac": 0.95},
        "windows": {
            "qasper-01": {"budget_r": 0.5, "rate_frac": 0.95, "rep": 1, "dataset": "qasper", "seed": 42,
                          "t_start": T0, "t_end": T0 + 100.0},
            "qasper-02": {"budget_r": 0.5, "rate_frac": 0.95, "rep": 2, "dataset": "qasper", "seed": 42,
                          "t_start": T0 + 200.0, "t_end": T0 + 250.0},
        },
    }))
    # window 01: wall clock, GPU series present, energy delta from the runner, one bad JSONL line
    _write_window(
        cell / "window_qasper-01", t_start=T0, t_end=T0 + 100.0,
        samples=[_sample(T0 - 2 + i, kv=0.1 * i, running=i, gpu_ok=True, energy_mj=1_000_000.0 + 10_000.0 * i)
                 for i in range(6)],
        requests=[_request(True, 10, 100), _request(True, 20, 150), _request(False, 0, 120)],
        label="PRESSURED", telemetry_ok=True, energy_delta_mj=50_000.0,
        gpu_block={"avg_gpu_utilization": 85.5, "avg_power_watts": 290.0, "gpu_count": 2},
        raw_lines=["{this is not json"],
    )
    # window 02: the S0 shape (ts 1.0 everywhere, gpu unavailable, no energy delta)
    _write_window(
        cell / "window_qasper-02", t_start=T0 + 200.0, t_end=T0 + 250.0,
        samples=[_sample(1.0, kv=0.2, running=1, gpu_ok=False, energy_mj=None) for _ in range(4)],
        requests=[_request(True, 5, 50)],
        label="UNKNOWN_TELEMETRY", telemetry_ok=False, energy_delta_mj=None,
        gpu_block={"avg_gpu_utilization": 3.1, "avg_power_watts": 134.4, "gpu_count": 2},
    )
    return root


def _read_index(out: Path) -> list[dict]:
    with open(out / "panels_index.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_renders_every_window_and_the_contact_sheet(run_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "panels"
    rc = rwp.main([str(run_root), "--out", str(out), "--price-per-hour", "3.49"])
    assert rc == 0
    rows = _read_index(out)
    assert [r["window"] for r in rows] == ["qasper-01", "qasper-02"]
    for r in rows:
        assert (out / r["panel"]).is_file(), r["panel"]
    sheets = list(out.glob("cell_*.png"))
    assert len(sheets) == 1


def test_resource_columns_are_hand_arithmetic(run_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "panels"
    assert rwp.main([str(run_root), "--out", str(out), "--price-per-hour", "3.49"]) == 0
    w1, w2 = _read_index(out)
    # window 01: 100 s, 2 completed of 3, 30 gen tokens, 250 prompt tokens, 2 GPUs
    assert w1["clock_mode"] == "wall"
    assert w1["n_samples"] == "6" and w1["n_malformed_lines"] == "1"
    assert w1["n_requests"] == "3" and w1["n_completed"] == "2" and w1["n_error"] == "1"
    assert float(w1["duration_s"]) == 100.0
    assert float(w1["gen_tps"]) == pytest.approx(0.30)
    assert float(w1["prompt_tps"]) == pytest.approx(2.50)
    assert float(w1["gen_tps_per_gpu"]) == pytest.approx(0.15)
    assert float(w1["prompt_tps_per_gpu"]) == pytest.approx(1.25)
    assert float(w1["energy_j"]) == pytest.approx(50.0)            # runner delta wins: 50,000 mJ
    assert float(w1["joules_per_completed"]) == pytest.approx(25.0)
    assert float(w1["usd_window"]) == pytest.approx(3.49 * 100 / 3600)
    assert float(w1["usd_per_completed"]) == pytest.approx(3.49 * 100 / 3600 / 2)
    assert float(w1["avg_gpu_util_pct"]) == 85.5 and float(w1["avg_power_w"]) == 290.0
    assert w1["gpu_series"] == "recorded"
    # window 02: the S0 shape
    assert w2["clock_mode"] == "index"
    assert w2["gpu_series"] == "not recorded"
    assert w2["energy_j"] == "" and w2["joules_per_completed"] == ""
    assert w2["regime_label"] == "UNKNOWN_TELEMETRY" and w2["telemetry_ok"] == "False"
    assert "clock" in w2["notes"] and "not recorded" in w2["notes"]


def test_price_is_optional_and_never_zero_filled(run_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "panels"
    assert rwp.main([str(run_root), "--out", str(out)]) == 0
    w1, _ = _read_index(out)
    assert w1["usd_per_hour"] == "" and w1["usd_window"] == "" and w1["usd_per_completed"] == ""
    assert float(w1["gen_tps_per_gpu"]) == pytest.approx(0.15)   # the rest is unaffected


def test_filters_select_windows(run_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "panels"
    assert rwp.main([str(run_root), "--out", str(out), "--window", "qasper-02", "--no-png"]) == 0
    rows = _read_index(out)
    assert [r["window"] for r in rows] == ["qasper-02"]
    assert rows[0]["panel"] == ""                       # --no-png: index only
    assert not list(out.glob("*.png"))


def test_refuses_a_non_run_directory(tmp_path: Path) -> None:
    bare = tmp_path / "not-a-run"
    bare.mkdir()
    assert rwp.main([str(bare), "--out", str(tmp_path / "out")]) == 2
    assert not (tmp_path / "out").exists()


def test_window_without_telemetry_is_indexed_not_rendered(run_root: Path, tmp_path: Path) -> None:
    w3 = run_root / "cells" / ROW_KEY / "window_qasper-03"
    w3.mkdir()
    _jsonl(w3 / "requests.jsonl", [_request(True, 1, 10)])
    out = tmp_path / "panels"
    assert rwp.main([str(run_root), "--out", str(out)]) == 0
    rows = {r["window"]: r for r in _read_index(out)}
    assert rows["qasper-03"]["panel"] == "" and "no telemetry" in rows["qasper-03"]["notes"]
    assert rows["qasper-03"]["n_requests"] == "1"


def test_metrics_lead_the_regime_bridge_field_names() -> None:
    # The panel reads the SAME fields the regime referee reads, fallbacks included.
    assert rwp.KV_FIELD == "kv_cache_usage" and rwp.KV_FALLBACK == "kv_usage"
    assert rwp.TS_FIELD == "ts_s" and rwp.TS_FALLBACK == "ts"


# ---------------------------------------------------------------------------
# Review findings of 2026-10-05 (fresh Fable 5.1 reviewer), one test each.
# ---------------------------------------------------------------------------

def _labels(ax) -> list[str]:
    """Data-trace labels of an axis; the dashed window-span markers carry '_child' labels."""
    return sorted(l.get_label() for l in ax.get_lines() if not l.get_label().startswith("_"))


def _captured_figure(monkeypatch, wd: "rwp.WindowData"):
    """Draw one window and hand back the live figure instead of a PNG."""
    captured = {}

    def _keep(fig, path):
        captured["fig"] = fig

    monkeypatch.setattr(rwp, "save_fig", _keep)
    rwp.apply_style()
    rwp._draw_window(wd, Path("unused.png"))
    return captured["fig"]


def _window_data(samples, *, requests=None, metrics=None, cell=None, regime=None) -> "rwp.WindowData":
    cell = cell if cell is not None else {"baseline": "B1", "gpu_count": 2, "cellspec": {"engine": "vllm", "family": "F2"},
                                           "windows": {"qasper-01": {"budget_r": 0.5, "rate_frac": 0.95, "rep": 1,
                                                                     "dataset": "qasper", "t_start": T0, "t_end": T0 + 10}}}
    return rwp.WindowData(cell_dir=Path("/x/cells/row"), window="qasper-01", cell=cell, samples=samples, n_malformed=0,
                          requests=requests, regime=regime or {"label": "PRESSURED", "telemetry_ok": True},
                          metrics=metrics or {"measured_window": {"t_start": T0, "t_end": T0 + 10}})


def test_h1_multi_gpu_draws_one_trace_per_gpu(monkeypatch) -> None:
    samples = []
    for i in range(4):
        s = _sample(T0 + i, kv=0.3, running=1, gpu_ok=True, energy_mj=1.0)
        s["gpu"]["gpus"] = [{"index": 0, "util_gpu": 50.0, "power_w": 200.0}, {"index": 1, "util_gpu": 60.0, "power_w": 210.0}]
        samples.append(s)
    fig = _captured_figure(monkeypatch, _window_data(samples))
    ax_util, ax_pw = fig.axes[4], fig.axes[5]
    assert _labels(ax_util) == ["GPU 0", "GPU 1"]
    assert _labels(ax_pw) == ["GPU 0", "GPU 1"]


def test_m1_missing_requests_file_leaves_request_columns_empty(run_root: Path, tmp_path: Path) -> None:
    (run_root / "cells" / ROW_KEY / "window_qasper-02" / "requests.jsonl").unlink()
    out = tmp_path / "panels"
    assert rwp.main([str(run_root), "--out", str(out), "--price-per-hour", "3.49", "--no-png"]) == 0
    w2 = {r["window"]: r for r in _read_index(out)}["qasper-02"]
    for col in ("n_requests", "n_completed", "n_error", "gen_tokens", "gen_tps", "gen_tps_per_gpu", "usd_per_completed"):
        assert w2[col] == "", col
    assert "no requests.jsonl" in w2["notes"]
    assert w2["usd_window"] != ""                     # the window's own cost does not need requests


def test_m2_pd_roles_are_separate_traces_and_the_clock_is_judged_per_role(monkeypatch) -> None:
    samples = []
    for i in range(5):                                   # prefill and decode ticks share the same second
        for role, kv in (("prefill", 0.2), ("decode", 0.6)):
            s = _sample(T0 + i, kv=kv, running=1, gpu_ok=True, energy_mj=1.0)
            s["instance"] = role
            samples.append(s)
    wd = _window_data(samples)
    assert wd.clock_mode() == "wall"
    assert list(wd.roles()) == ["prefill", "decode"]
    fig = _captured_figure(monkeypatch, wd)
    assert _labels(fig.axes[0]) == ["kv_cache_usage (decode)", "kv_cache_usage (prefill)"]
    assert _labels(fig.axes[4]) == ["GPU 0"]             # GPU traces drawn once, not per role


def test_l1_kv_usage_only_records_are_drawn(monkeypatch) -> None:
    samples = []
    for i in range(3):
        s = _sample(T0 + i, kv=0.4, running=1, gpu_ok=False, energy_mj=None)
        del s["kv_cache_usage"]                          # legacy record: kv_usage only
        samples.append(s)
    fig = _captured_figure(monkeypatch, _window_data(samples))
    assert _labels(fig.axes[0]) == ["kv_cache_usage"]


def test_l2_kv_above_the_fraction_contract_is_noted_not_rescaled(run_root: Path, tmp_path: Path) -> None:
    w1 = run_root / "cells" / ROW_KEY / "window_qasper-01"
    rows = [json.loads(l) for l in open(w1 / "cage_stats.jsonl") if l.startswith("{") and "not json" not in l]
    for r in rows:
        r["kv_cache_usage"] = 1.2
    _jsonl(w1 / "cage_stats.jsonl", rows)
    out = tmp_path / "panels"
    assert rwp.main([str(run_root), "--out", str(out), "--no-png"]) == 0
    row = {r["window"]: r for r in _read_index(out)}["qasper-01"]
    assert "above 1.0" in row["notes"]


def test_l3_single_sample_and_partial_timestamps_get_their_own_notes() -> None:
    one = _window_data([_sample(T0, kv=0.1, running=0, gpu_ok=False, energy_mj=None)])
    assert one.clock_mode() == "wall" and one.clock_problem() is None
    three = [_sample(T0 + i, kv=0.1, running=0, gpu_ok=False, energy_mj=None) for i in range(3)]
    del three[1]["ts_s"]; del three[1]["ts"]
    partial = _window_data(three)
    assert partial.clock_mode() == "index"
    assert partial.clock_problem() == "timestamps missing on 1 of 3 samples"


def test_l4_indexless_gpu_entries_are_keyed_by_position() -> None:
    samples = []
    for i in range(3):
        s = _sample(T0 + i, kv=0.1, running=0, gpu_ok=True, energy_mj=None)
        s["gpu"]["gpus"] = [{"util_gpu": 10.0, "power_w": 100.0}, {"util_gpu": 20.0, "power_w": 110.0}]
        samples.append(s)
    util, power, recorded = rwp.WindowData.gpu_series(samples)
    assert recorded and sorted(util) == [0, 1] and len(util[1]) == 3 and util[1] == [20.0, 20.0, 20.0]


def test_l5_zero_length_span_is_noted_and_rates_stay_empty() -> None:
    wd = _window_data([_sample(T0, kv=0.1, running=0, gpu_ok=False, energy_mj=None)],
                      requests=[_request(True, 5, 50)],
                      metrics={"measured_window": {"t_start": T0, "t_end": T0}},
                      regime={"label": "PRESSURED", "telemetry_ok": True, "t_start": T0, "t_end": T0})
    wd.cell["windows"]["qasper-01"]["t_end"] = T0
    assert wd.duration_s() is None
    row = rwp._index_row(wd, "run", 3.49, "")
    assert row["gen_tps"] is None and row["usd_window"] is None


def test_l6_refuses_a_non_empty_out_directory(run_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "panels"
    assert rwp.main([str(run_root), "--out", str(out), "--no-png"]) == 0
    assert rwp.main([str(run_root), "--out", str(out), "--no-png"]) == 2    # second render into the same dir


def test_l8_boolean_gpu_count_is_not_a_count() -> None:
    wd = _window_data([_sample(T0, kv=0.1, running=0, gpu_ok=False, energy_mj=None)], requests=[_request(True, 10, 10)])
    wd.cell["gpu_count"] = True
    row = rwp._index_row(wd, "run", None, "")
    assert row["gpu_count"] is None and row["gen_tps_per_gpu"] is None and row["gen_tps"] == pytest.approx(1.0)


def test_price_zero_is_refused(run_root: Path, tmp_path: Path) -> None:
    assert rwp.main([str(run_root), "--out", str(tmp_path / "o"), "--price-per-hour", "0"]) == 2
    assert not (tmp_path / "o").exists()


def test_duration_precedence_and_null_telemetry_block() -> None:
    samples = [_sample(T0 + i, kv=0.1, running=0, gpu_ok=False, energy_mj=1_000_000.0 + 2_000.0 * i) for i in range(3)]
    wd = _window_data(samples, requests=[_request(True, 4, 40)],
                      metrics={"measured_window": {"t_start": T0, "t_end": T0 + 8}, "vllm_telemetry": None},
                      regime={"label": "PRESSURED", "telemetry_ok": True, "t_start": T0, "t_end": T0 + 100})
    assert wd.duration_s() == 8.0                        # metrics.json wins over regime.json
    assert wd.energy_j() == pytest.approx(4.0)           # series fallback: (1,004,000 - 1,000,000) mJ
