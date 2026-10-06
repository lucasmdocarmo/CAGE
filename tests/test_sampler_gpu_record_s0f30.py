"""S0F-30 (ADR-0141): the telemetry sampler records the per-GPU utilization and
power record itself, through the NVML handles it already opens for energy.

Cause, read on 2026-10-06 [V]: the headless cage-stats API
(``cage_stats/api.py`` ``fetch_snapshot``) returns the engine's derived
``Snapshot``, whose ``gpu`` field is the dataclass default
(``cage_stats/metrics/state.py:192``); the GPU provider is built and sampled
only inside the terminal dashboard (``cage_stats/ui/app.py:101, 215``). Every
one of the 481 S0 samples (results/s0/a/s0-20260930) carried exactly that
default, ``{"available": false, "source": "none", "gpus": [], "error": null}``;
a provider that ran and failed would have carried an error string. The S0
energy read in the same process proved NVML reachable there.

The fix lives in CAGE only (owner: "Follow recommendations", option A): the
sampler reads one ``gpu`` sub-record per tick with the ``GpuSample`` field
names of cage-stats, the shape ``scripts/4_analysis/render_window_panels.py``
reads (``gpu.gpus[].util_gpu`` and ``power_w``, line 258). Absence stays
absence: no pynvml, a failed init, or no answering device leaves the record as
received; a device whose read fails is a None slot, never a zero.

pynvml is faked through ``sys.modules`` (the pattern of
tests/test_cov_performance_trackers.py); no GPU, no network.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from src.monitoring import vllm_telemetry as t

#: The record every S0 sample carried: the serialized cage-stats
#: ``GpuSnapshot()`` default (counted over the S0 tree on 2026-10-06).
S0_GPU_RECORD: dict[str, Any] = {
    "available": False, "source": "none", "gpus": [], "error": None,
}

GIB = 2**30

#: Two devices with the S0 H100 window averages as the first device's values
#: (metrics.json["gpu"] of corpus-fresh ... window_squad_v2-01: 42 %, 333 W).
DEVICES: list[dict[str, Any]] = [
    {
        "name": b"NVIDIA H100 80GB HBM3", "util": 42, "used": 60 * GIB,
        "total": 80 * GIB, "power_mw": 333000, "limit_mw": 700000, "temp": 55,
        "energy_mj": 6_344_923_519_745.0,
    },
    {
        "name": "NVIDIA H100 80GB HBM3", "util": 7, "used": 10 * GIB,
        "total": 80 * GIB, "power_mw": 90000, "limit_mw": 700000, "temp": 40,
        "energy_mj": 1000.0,
    },
]


class _Util:
    def __init__(self, gpu: int) -> None:
        self.gpu = gpu


class _Mem:
    def __init__(self, used: int, total: int) -> None:
        self.used = used
        self.total = total


def _fake_pynvml(
    *,
    devices: list[dict[str, Any]],
    fail_util: set[int] = frozenset(),
    fail_mem: set[int] = frozenset(),
    fail_power: set[int] = frozenset(),
    init_raises: bool = False,
) -> types.ModuleType:
    """The subset of pynvml the sampler calls, over ``devices``."""
    mod = types.ModuleType("pynvml")
    mod.NVML_TEMPERATURE_GPU = 0  # type: ignore[attr-defined]

    def nvml_init() -> None:
        if init_raises:
            raise RuntimeError("NVML init failed")

    def dev(handle: tuple[str, int]) -> dict[str, Any]:
        return devices[handle[1]]

    def util(handle: tuple[str, int]) -> _Util:
        if handle[1] in fail_util:
            raise RuntimeError("utilization read failed")
        return _Util(dev(handle)["util"])

    def mem(handle: tuple[str, int]) -> _Mem:
        if handle[1] in fail_mem:
            raise RuntimeError("memory read failed")
        return _Mem(dev(handle)["used"], dev(handle)["total"])

    def power(handle: tuple[str, int]) -> int:
        if handle[1] in fail_power:
            raise RuntimeError("power read failed")
        return dev(handle)["power_mw"]

    mod.nvmlInit = nvml_init  # type: ignore[attr-defined]
    mod.nvmlDeviceGetCount = lambda: len(devices)  # type: ignore[attr-defined]
    mod.nvmlDeviceGetHandleByIndex = lambda i: ("handle", i)  # type: ignore[attr-defined]
    mod.nvmlDeviceGetTotalEnergyConsumption = lambda h: dev(h)["energy_mj"]  # type: ignore[attr-defined]
    mod.nvmlDeviceGetUtilizationRates = util  # type: ignore[attr-defined]
    mod.nvmlDeviceGetMemoryInfo = mem  # type: ignore[attr-defined]
    mod.nvmlDeviceGetName = lambda h: dev(h)["name"]  # type: ignore[attr-defined]
    mod.nvmlDeviceGetPowerUsage = power  # type: ignore[attr-defined]
    mod.nvmlDeviceGetEnforcedPowerLimit = lambda h: dev(h)["limit_mw"]  # type: ignore[attr-defined]
    mod.nvmlDeviceGetTemperature = lambda h, kind: dev(h)["temp"]  # type: ignore[attr-defined]
    return mod


def _one_tick(
    monkeypatch: pytest.MonkeyPatch, sampler: t.VllmTelemetrySampler, snapshot: Any
) -> dict[str, Any]:
    """Run exactly one sampler tick on the calling thread and return its sample."""

    def capture(url: str, **_kw: Any) -> Any:
        sampler._stop.set()  # the loop ends after this tick
        return dict(snapshot) if isinstance(snapshot, dict) else snapshot

    monkeypatch.setattr(t, "capture_snapshot", capture)
    sampler._run()
    assert len(sampler._samples) == 1
    return sampler._samples[0]


def _snapshot_with(gpu: Any) -> dict[str, Any]:
    return {"ts": 1.0, "connected": True, "kv_usage": 0.0, "gpu": gpu}


# ---------------------------------------------------------------------------
# 0. the cause, pinned against the installed cage-stats
# ---------------------------------------------------------------------------


def test_s0_record_is_the_cage_stats_dataclass_default() -> None:
    from cage_stats.metrics.state import GpuSnapshot

    assert dataclasses.asdict(GpuSnapshot()) == S0_GPU_RECORD


# ---------------------------------------------------------------------------
# 1. the record on the happy path
# ---------------------------------------------------------------------------


def test_sampler_fills_the_gpu_record_from_nvml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml(devices=DEVICES))
    sampler = t.VllmTelemetrySampler("http://localhost:8000")
    sample = _one_tick(monkeypatch, sampler, _snapshot_with(dict(S0_GPU_RECORD)))
    gpu = sample["gpu"]
    assert gpu["available"] is True
    assert gpu["source"] == "nvml"
    assert gpu["error"] is None
    assert [d["index"] for d in gpu["gpus"]] == [0, 1]
    first, second = gpu["gpus"]
    assert first["name"] == "NVIDIA H100 80GB HBM3"  # bytes decoded
    assert first["util_gpu"] == 42.0
    assert first["mem_used"] == 60 * GIB and first["mem_total"] == 80 * GIB
    assert first["power_w"] == 333.0 and first["power_limit_w"] == 700.0
    assert first["temp_c"] == 55.0
    assert second["name"] == "NVIDIA H100 80GB HBM3"
    assert second["util_gpu"] == 7.0 and second["power_w"] == 90.0
    # The energy read on the same tick is unchanged (sum over all devices).
    assert sample["energy_mj"] == 6_344_923_519_745.0 + 1000.0
    assert sample["energy_mj_per_gpu"] == [6_344_923_519_745.0, 1000.0]


def test_record_is_added_when_the_snapshot_carries_no_gpu_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The CLI fallback snapshot shape has no "gpu" key at all.
    monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml(devices=DEVICES[:1]))
    sampler = t.VllmTelemetrySampler("http://localhost:8000")
    sample = _one_tick(monkeypatch, sampler, {"ts": 1.0, "connected": True})
    assert sample["gpu"]["available"] is True
    assert len(sample["gpu"]["gpus"]) == 1


# ---------------------------------------------------------------------------
# 2. holes, never zeros
# ---------------------------------------------------------------------------


def test_one_failed_device_read_leaves_a_hole_not_a_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake_pynvml(devices=DEVICES, fail_util={1}, fail_power={0})
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    sampler = t.VllmTelemetrySampler("http://localhost:8000")
    gpu = _one_tick(monkeypatch, sampler, _snapshot_with(dict(S0_GPU_RECORD)))["gpu"]
    assert gpu["available"] is True
    first, second = gpu["gpus"]
    assert first["util_gpu"] == 42.0 and first["power_w"] is None
    assert second["util_gpu"] is None and second["power_w"] == 90.0
    assert second["mem_used"] == 10 * GIB  # the memory read still answered


def test_a_bad_handle_is_listed_with_nones(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake_pynvml(devices=DEVICES)

    def handle(i: int) -> tuple[str, int]:
        if i == 1:
            raise RuntimeError("handle 1 failed")
        return ("handle", i)

    fake.nvmlDeviceGetHandleByIndex = handle  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    sampler = t.VllmTelemetrySampler("http://localhost:8000")
    gpu = _one_tick(monkeypatch, sampler, _snapshot_with(dict(S0_GPU_RECORD)))["gpu"]
    assert gpu["available"] is True
    assert gpu["gpus"][0]["util_gpu"] == 42.0
    bad = gpu["gpus"][1]
    assert bad["index"] == 1
    assert all(bad[k] is None for k in bad if k != "index")


def test_no_answering_device_leaves_the_record_as_received(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake_pynvml(devices=DEVICES, fail_util={0, 1}, fail_mem={0, 1})
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    sampler = t.VllmTelemetrySampler("http://localhost:8000")
    sample = _one_tick(monkeypatch, sampler, _snapshot_with(dict(S0_GPU_RECORD)))
    assert sample["gpu"] == S0_GPU_RECORD


def test_pynvml_absent_leaves_the_record_as_received(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", None)  # import raises
    sampler = t.VllmTelemetrySampler("http://localhost:8000")
    sample = _one_tick(monkeypatch, sampler, _snapshot_with(dict(S0_GPU_RECORD)))
    assert sample["gpu"] == S0_GPU_RECORD
    assert sample["energy_mj"] is None
    assert sampler._nvml_failed is True


def test_nvml_init_failure_disables_the_probe_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake_pynvml(devices=DEVICES, init_raises=True)
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    sampler = t.VllmTelemetrySampler("http://localhost:8000")
    sample = _one_tick(monkeypatch, sampler, {"ts": 1.0, "connected": True})
    assert "gpu" not in sample
    assert sampler._nvml_failed is True
    # A second tick never retries the init (the existing energy rule).
    calls = {"init": 0}

    def counting_init() -> None:
        calls["init"] += 1
        raise RuntimeError("again")

    fake.nvmlInit = counting_init  # type: ignore[attr-defined]
    sampler._stop.clear()
    sampler._samples.clear()
    sampler._sample_ts.clear()
    _one_tick(monkeypatch, sampler, {"ts": 2.0, "connected": True})
    assert calls["init"] == 0


# ---------------------------------------------------------------------------
# 3. the schema contract with cage-stats and the renderer
# ---------------------------------------------------------------------------


def test_record_field_names_are_the_cage_stats_sample_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cage_stats.metrics.state import GpuSample, GpuSnapshot

    monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml(devices=DEVICES))
    sampler = t.VllmTelemetrySampler("http://localhost:8000")
    gpu = _one_tick(monkeypatch, sampler, _snapshot_with(dict(S0_GPU_RECORD)))["gpu"]
    assert set(gpu) == {f.name for f in dataclasses.fields(GpuSnapshot)}
    sample_fields = {f.name for f in dataclasses.fields(GpuSample)}
    for dev in gpu["gpus"]:
        assert set(dev) <= sample_fields, sorted(set(dev) - sample_fields)
        # The two fields the panel renderer reads (render_window_panels.py:258).
        assert "util_gpu" in dev and "power_w" in dev


def test_series_records_carry_the_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml(devices=DEVICES))
    sampler = t.VllmTelemetrySampler("http://localhost:8000", role="single")
    _one_tick(monkeypatch, sampler, _snapshot_with(dict(S0_GPU_RECORD)))
    out = tmp_path / "telemetry_series.jsonl"
    assert sampler.save_series(str(out)) == str(out)
    rec = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert rec["instance"] == "single"
    assert rec["gpu"]["available"] is True
    assert rec["gpu"]["gpus"][0]["util_gpu"] == 42.0
    assert rec["gpu"]["gpus"][0]["power_w"] == 333.0


def test_a_received_available_record_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    # Review of SF-A (2026-10-06, LOW): a cage-stats that samples its provider
    # owns the record; the sampler fills only an unavailable one.
    monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml(devices=DEVICES))
    theirs = {"available": True, "source": "nvml", "error": None,
              "gpus": [{"index": 0, "util_gpu": 11.0, "power_w": 100.0}]}
    sampler = t.VllmTelemetrySampler("http://localhost:8000")
    sample = _one_tick(monkeypatch, sampler, _snapshot_with(dict(theirs)))
    assert sample["gpu"] == theirs
    # A record that says available but is not a dict shape is replaced.
    sampler = t.VllmTelemetrySampler("http://localhost:8000")
    sample = _one_tick(monkeypatch, sampler, _snapshot_with("garbage"))
    assert sample["gpu"]["source"] == "nvml" and sample["gpu"]["gpus"][0]["util_gpu"] == 42.0


def test_aggregate_final_snapshot_and_merged_series_carry_the_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml(devices=DEVICES))
    prefill = t.VllmTelemetrySampler("http://localhost:8100", role="prefill")
    decode = t.VllmTelemetrySampler("http://localhost:8200", role="decode")
    _one_tick(monkeypatch, prefill, _snapshot_with(dict(S0_GPU_RECORD)))
    _one_tick(monkeypatch, decode, _snapshot_with(dict(S0_GPU_RECORD)))
    agg = prefill.aggregate()
    assert agg is not None
    assert agg["final_snapshot"]["gpu"]["available"] is True
    assert agg["final_snapshot"]["gpu"]["gpus"][0]["power_w"] == 333.0
    out = tmp_path / "merged.jsonl"
    assert t.save_merged_series([prefill, decode], str(out)) == str(out)
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert {r["instance"] for r in records} == {"prefill", "decode"}
    for rec in records:
        assert rec["gpu"]["available"] is True
        assert rec["gpu"]["gpus"][1]["util_gpu"] == 7.0


def test_source_constant_and_docstring_name_the_finding() -> None:
    assert t.VllmTelemetrySampler.GPU_RECORD_SOURCE == "nvml"
    doc = t.VllmTelemetrySampler._read_gpu_record.__doc__ or ""
    assert "S0F-30" in doc
    assert chr(0x2014) not in doc and chr(0x2013) not in doc  # no em or en dash
