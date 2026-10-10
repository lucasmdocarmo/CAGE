"""S0F-53 (review 2026-10-10, F1): the telemetry sampler stamps each sample
with the instant its gauges describe, never the capture start.

``capture_snapshot`` polls the engine twice, one second apart, and the
snapshot describes the SECOND poll; cage-stats 51bb9ac stamps ``Snapshot.ts``
with that poll's wall clock. The pre-fix sampler stamped the capture START,
so every record sat about one second before the state it described and the
regime bridge joined the window to shifted samples.

Rule under test (``VllmTelemetrySampler._sample_stamp``): the snapshot's own
``ts`` when it is a finite number inside [capture start, capture end]; else
the capture end. The record carries ``ts_source``. No GPU, no network: the
capture is a fake that sleeps and returns a dict.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from src.monitoring import vllm_telemetry as t


def _one_tick(
    monkeypatch: pytest.MonkeyPatch, sampler: t.VllmTelemetrySampler, make_snapshot: Any,
    sleep_s: float = 0.05,
) -> tuple[float, float, dict]:
    """Run one tick on the calling thread; return (t0, t1, record)."""
    bounds: dict = {}

    def capture(url: str, **_kw: Any) -> Any:
        bounds["t0"] = time.time()
        time.sleep(sleep_s)
        snap = make_snapshot()
        bounds["t1"] = time.time()
        sampler._stop.set()
        return snap

    monkeypatch.setitem(sys.modules, "pynvml", None)  # no NVML in the test
    monkeypatch.setattr(t, "capture_snapshot", capture)
    sampler._run()
    assert len(sampler._samples) == 1
    rec = sampler._series_records()[0]
    return bounds["t0"], bounds["t1"], rec


def test_snapshot_clock_inside_the_capture_span_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    sampler = t.VllmTelemetrySampler("http://localhost:8000")
    stamped: dict = {}

    def make() -> dict:
        stamped["ts"] = time.time()  # the second poll's clock, at the END of the capture
        return {"ts": stamped["ts"], "connected": True, "kv_usage": 0.5, "waiting": 0}

    t0, t1, rec = _one_tick(monkeypatch, sampler, make)
    assert rec["ts_source"] == "snapshot"
    assert rec["ts"] == round(stamped["ts"], 3)
    assert rec["ts_s"] == rec["ts"]
    # the stamp sits at the end of the capture, never at its start
    assert rec["ts"] >= round(t0, 3) and rec["ts"] <= round(t1, 3) + 0.001
    assert rec["ts"] - t0 >= 0.04  # the fake slept 50 ms before stamping


@pytest.mark.parametrize(
    "bad_ts", [1.0, None, "1700000000.5", float("nan"), float("inf"), True, 0.0],
)
def test_an_implausible_snapshot_clock_falls_back_to_the_capture_end(
    monkeypatch: pytest.MonkeyPatch, bad_ts: Any
) -> None:
    sampler = t.VllmTelemetrySampler("http://localhost:8000")
    snap: dict = {"connected": True, "kv_usage": 0.5}
    if bad_ts is not None:
        snap["ts"] = bad_ts
    t0, t1, rec = _one_tick(monkeypatch, sampler, lambda: dict(snap))
    assert rec["ts_source"] == "capture_end"
    assert math.isfinite(rec["ts"])
    assert rec["ts"] == round(t1, 3)
    assert rec["ts"] > round(t0, 3)


def test_a_snapshot_clock_from_the_future_or_the_past_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampler = t.VllmTelemetrySampler("http://localhost:8000")
    _t0, t1, rec = _one_tick(monkeypatch, sampler, lambda: {"ts": time.time() + 100.0})
    assert rec["ts_source"] == "capture_end" and rec["ts"] == round(t1, 3)
    sampler = t.VllmTelemetrySampler("http://localhost:8000")
    _t0, t1, rec = _one_tick(monkeypatch, sampler, lambda: {"ts": time.time() - 100.0})
    assert rec["ts_source"] == "capture_end" and rec["ts"] == round(t1, 3)


def test_sample_stamp_rule_is_pure() -> None:
    f = t.VllmTelemetrySampler._sample_stamp
    assert f({"ts": 10.5}, 10.0, 11.0) == (10.5, "snapshot")
    assert f({"ts": 10.0}, 10.0, 11.0) == (10.0, "snapshot")  # inclusive bounds
    assert f({"ts": 11.0}, 10.0, 11.0) == (11.0, "snapshot")
    assert f({"ts": 9.999}, 10.0, 11.0) == (11.0, "capture_end")
    assert f({"ts": 11.001}, 10.0, 11.0) == (11.0, "capture_end")
    assert f({}, 10.0, 11.0) == (11.0, "capture_end")
    assert f({"ts": True}, 10.0, 11.0) == (11.0, "capture_end")


def test_saved_series_and_merged_series_carry_the_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sampler = t.VllmTelemetrySampler("http://localhost:8000", role="single")
    _one_tick(monkeypatch, sampler, lambda: {"ts": time.time(), "connected": True})
    out = tmp_path / "series.jsonl"
    assert sampler.save_series(str(out)) == str(out)
    rec = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert rec["ts_source"] == "snapshot"
    assert rec["instance"] == "single"
    other = t.VllmTelemetrySampler("http://localhost:8100", role="decode")
    _one_tick(monkeypatch, other, lambda: {"connected": True})
    merged = tmp_path / "merged.jsonl"
    assert t.save_merged_series([sampler, other], str(merged)) == str(merged)
    records = [json.loads(line) for line in merged.read_text(encoding="utf-8").splitlines()]
    assert {r["ts_source"] for r in records} == {"snapshot", "capture_end"}
    assert [r["ts_s"] for r in records] == sorted(r["ts_s"] for r in records)


def test_docstring_names_the_finding_and_carries_no_dashes() -> None:
    doc = t.VllmTelemetrySampler._sample_stamp.__doc__ or ""
    assert "S0F-53" in doc
    assert chr(0x2014) not in doc and chr(0x2013) not in doc
