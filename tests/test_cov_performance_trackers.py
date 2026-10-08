"""Offline coverage for src/evaluation/performance.py trackers (K-COV5, #142).

The speculative tracker compute path (pure-python acceptance-rate arithmetic)
and the NVML consumption stack were at 0% coverage. All tests run offline: the
device layer (pynvml) is a fake module injected via sys.modules (the
llmlingua-stub pattern of tests/test_compression_ops.py); no GPU, no NVML, no
network.

ADR-0148 (2026-10-07): a count of our own events (requests, tokens, errors,
samples) is 0 on an empty set; every quantity derived from readings (mean,
max, percentile, ratio, rate, sum of device readings) is None when there is no
reading, and each block carries the sample count that explains its None. The
former zero arms of these trackers are pinned to None here. The
CacheMetricsTracker left src the same day: its block duplicated the per-row
cached_prompt_tokens record and fabricated a miss for engines without a count.

Covers:
- PerformanceEvaluator: start/stop discipline, serving-time (not wall-clock)
  throughput denominator, TPOT (num_tokens-1) with the single-token and
  non-streaming (ttft == total) exclusions, error filtering, the None arms
  with their counts, to_dict round-trip, reset
- SpeculativeMetricsTracker: acceptance-rate arithmetic, rollback recording
  gate (>0 only), speedup with/without baseline, reset
- GPUMetricsTracker on the fake NVML: init/static info, sampling incl. the
  per-call NVMLError None arms (a failed read is dropped, never a fake zero),
  aggregation arithmetic, the no-sample and every-read-failed None arms, the
  failed power-limit read, unavailable/import-error arms, shutdown
"""

from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest

from src.evaluation.performance import (
    GPUMetricsTracker,
    PerformanceEvaluator,
    SpeculativeMetricsTracker,
)


# --------------------------------------------------------------------------- #
# PerformanceEvaluator
# --------------------------------------------------------------------------- #


class TestPerformanceEvaluator:
    def _evaluator(self) -> PerformanceEvaluator:
        return PerformanceEvaluator(monitor_resources=False)

    def test_compute_before_start_stop_raises(self):
        ev = self._evaluator()
        with pytest.raises(ValueError, match="start.*stop"):
            ev.compute_metrics()

    def test_hand_derivable_throughput_and_tpot(self):
        ev = self._evaluator()
        ev.start()
        # 2 requests: ttft 100ms, total 600ms, 6 tokens each
        # -> per-request TPOT = (600-100)/(6-1) = 100 ms
        # -> serving_time = (600+600)/1000 = 1.2 s (summed, NOT wall-clock)
        ev.record_request("r1", ttft_ms=100.0, total_time_ms=600.0, num_tokens=6)
        ev.record_request("r2", ttft_ms=100.0, total_time_ms=600.0, num_tokens=6)
        ev.stop()
        m = ev.compute_metrics()
        assert m.serving_time_seconds == pytest.approx(1.2)
        assert m.queries_per_second == pytest.approx(2 / 1.2)
        assert m.tokens_per_second == pytest.approx(12 / 1.2)
        assert m.avg_ttft_ms == pytest.approx(100.0)
        assert m.avg_tpot_ms == pytest.approx(100.0)
        assert m.p50_tpot_ms == pytest.approx(100.0)
        assert m.avg_latency_ms == pytest.approx(600.0)
        assert m.total_requests == 2
        assert m.total_tokens == 12
        assert m.error_count == 0
        assert m.tpot_sample_count == 2  # both rows had a positive generation interval
        # Wall-clock span is the stage window, not the serving denominator.
        assert m.total_time_seconds >= 0.0

    def test_error_rows_excluded_but_counted(self):
        ev = self._evaluator()
        ev.start()
        ev.record_request("ok", ttft_ms=50.0, total_time_ms=250.0, num_tokens=5)
        ev.record_request("boom", ttft_ms=0.0, total_time_ms=0.0, num_tokens=0,
                          error="HTTP 500")
        ev.stop()
        m = ev.compute_metrics()
        assert m.error_count == 1
        assert m.total_requests == 2  # successes + errors
        assert m.total_tokens == 5    # error rows contribute no tokens
        assert m.avg_latency_ms == pytest.approx(250.0)

    def test_all_errors_returns_none_not_zero(self):
        # ADR-0148 (2026-10-07): before it this arm returned 0.0 in every
        # latency, throughput, TPOT and resource field, so a window where every
        # request failed read as a window served at zero latency. A quantity
        # derived from readings is None when there is no reading; the counts
        # (requests, errors, tokens) and the summed serving time over the empty
        # successful set stay numbers.
        ev = self._evaluator()
        ev.start()
        ev.record_request("e1", 0.0, 0.0, 0, error="x")
        ev.record_request("e2", 0.0, 0.0, 0, error="y")
        ev.stop()
        m = ev.compute_metrics()
        assert m.queries_per_second is None
        assert m.tokens_per_second is None
        assert m.avg_ttft_ms is None and m.p99_ttft_ms is None
        assert m.avg_tpot_ms is None and m.p99_tpot_ms is None
        assert m.avg_latency_ms is None and m.p99_latency_ms is None
        assert m.avg_cpu_percent is None and m.peak_memory_mb is None
        assert m.serving_time_seconds == 0.0
        assert m.total_requests == 2
        assert m.error_count == 2
        assert m.total_tokens == 0
        assert m.tpot_sample_count == 0
        assert m.resource_sample_count == 0
        d = json.loads(json.dumps(m.to_dict()))
        measured_keys = [k for k in d if k.startswith(("avg_", "p50_", "p95_", "p99_", "peak_"))]
        measured_keys += ["queries_per_second", "tokens_per_second"]
        assert all(d[k] is None for k in measured_keys), d

    def test_tpot_excludes_single_token_and_non_streaming_rows(self):
        ev = self._evaluator()
        ev.start()
        # Single-token output: no inter-token interval.
        ev.record_request("one_tok", ttft_ms=100.0, total_time_ms=400.0, num_tokens=1)
        # Non-streaming path: ttft deliberately == total -> generation time 0,
        # unmeasurable, must be EXCLUDED (review fix), not folded in as ~0.
        ev.record_request("no_stream", ttft_ms=500.0, total_time_ms=500.0, num_tokens=8)
        ev.stop()
        m = ev.compute_metrics()
        # ADR-0148: no request had a positive generation interval, so TPOT is
        # unmeasured (None with a zero count), not 0.0 ms per token.
        assert m.avg_tpot_ms is None
        assert m.p99_tpot_ms is None
        assert m.tpot_sample_count == 0
        # Both rows still count toward latency/throughput.
        assert m.total_requests == 2
        assert m.avg_latency_ms == pytest.approx(450.0)

    def test_percentiles_match_numpy_reference(self):
        ev = self._evaluator()
        ev.start()
        latencies = [100.0, 200.0, 300.0, 400.0, 1000.0]
        for i, total in enumerate(latencies):
            ev.record_request(f"r{i}", ttft_ms=10.0, total_time_ms=total, num_tokens=2)
        ev.stop()
        m = ev.compute_metrics()
        assert m.p50_latency_ms == pytest.approx(float(np.percentile(latencies, 50)))
        assert m.p95_latency_ms == pytest.approx(float(np.percentile(latencies, 95)))
        assert m.p99_latency_ms == pytest.approx(float(np.percentile(latencies, 99)))

    def test_to_dict_round_trips_every_field(self):
        ev = self._evaluator()
        ev.start()
        ev.record_request("r", 50.0, 150.0, 3)
        ev.stop()
        d = ev.compute_metrics().to_dict()
        assert d["total_requests"] == 1
        assert d["total_tokens"] == 3
        assert d["error_count"] == 0
        assert "serving_time_seconds" in d and "total_time_seconds" in d
        assert d["tpot_sample_count"] == 1 and d["resource_sample_count"] == 0  # ADR-0148 counts

    def test_reset_clears_state(self):
        ev = self._evaluator()
        ev.start()
        ev.record_request("r", 50.0, 150.0, 3)
        ev.stop()
        ev.reset()
        assert ev.start_time is None and ev.end_time is None
        assert ev.request_metrics == []
        with pytest.raises(ValueError):
            ev.compute_metrics()

    def test_resource_monitoring_samples_local_process(self):
        # psutil against the test's own process: offline, no device layer.
        ev = PerformanceEvaluator(monitor_resources=True)
        ev.start()
        ev.record_request("r", 50.0, 150.0, 3)
        ev.stop()
        m = ev.compute_metrics()
        assert len(ev.memory_samples) >= 2  # start + stop samples
        assert m.avg_memory_mb > 0.0
        assert m.peak_memory_mb >= m.avg_memory_mb
        assert m.resource_sample_count == len(ev.memory_samples)  # ADR-0148

    def test_no_resource_samples_is_none_with_count_zero(self):
        # ADR-0148: monitor_resources=False collects nothing; the resource
        # fields are None and the count says why.
        ev = self._evaluator()
        ev.start()
        ev.record_request("r", 50.0, 150.0, 3)
        ev.stop()
        m = ev.compute_metrics()
        assert m.avg_cpu_percent is None
        assert m.avg_memory_mb is None
        assert m.peak_memory_mb is None
        assert m.resource_sample_count == 0
        assert m.avg_latency_ms == pytest.approx(150.0)  # the request itself is measured

    def test_zero_serving_time_with_successes_is_none(self):
        # ADR-0148 boundary: successful rows whose summed serving time is 0
        # have no throughput denominator; the rates are None, not 0.0.
        ev = self._evaluator()
        ev.start()
        ev.record_request("a", ttft_ms=0.0, total_time_ms=0.0, num_tokens=1)
        ev.record_request("b", ttft_ms=0.0, total_time_ms=0.0, num_tokens=1)
        ev.stop()
        m = ev.compute_metrics()
        assert m.queries_per_second is None
        assert m.tokens_per_second is None
        assert m.serving_time_seconds == 0.0
        assert m.total_requests == 2
        assert m.avg_latency_ms == pytest.approx(0.0)  # a measured zero stays a number


# --------------------------------------------------------------------------- #
# SpeculativeMetricsTracker
# --------------------------------------------------------------------------- #


class TestSpeculativeMetricsTracker:
    def test_acceptance_rate_arithmetic(self):
        t = SpeculativeMetricsTracker()
        t.record_step(draft_tokens=4, accepted_tokens=3)
        t.record_step(draft_tokens=4, accepted_tokens=1, rollback_latency_ms=2.0)
        t.record_step(draft_tokens=2, accepted_tokens=2)
        m = t.compute_metrics(actual_latency_ms=100.0)
        assert m.total_draft_tokens == 10
        assert m.total_accepted_tokens == 6
        assert m.total_rejected_tokens == 4
        assert m.acceptance_rate == pytest.approx(0.6)
        assert m.avg_draft_tokens == pytest.approx(10 / 3)
        assert m.avg_accepted_tokens == pytest.approx(2.0)
        assert m.rollback_overhead_ms == pytest.approx(2.0)

    def test_zero_rollback_latency_is_not_recorded(self):
        t = SpeculativeMetricsTracker()
        t.record_step(4, 4, rollback_latency_ms=0.0)
        assert t.rollback_latencies == []
        assert t.compute_metrics(50.0).rollback_overhead_ms == 0.0

    def test_speedup_requires_baseline(self):
        t = SpeculativeMetricsTracker()
        t.record_step(4, 2)
        assert t.compute_metrics(100.0).speedup_ratio == 1.0  # no baseline
        t.set_baseline_latency(300.0)
        assert t.compute_metrics(100.0).speedup_ratio == pytest.approx(3.0)

    def test_no_steps_zero_acceptance(self):
        m = SpeculativeMetricsTracker().compute_metrics(100.0)
        assert m.acceptance_rate == 0.0
        assert m.avg_draft_tokens == 0.0
        assert m.total_draft_tokens == 0

    def test_to_dict_and_reset(self):
        t = SpeculativeMetricsTracker()
        t.record_step(4, 3)
        t.set_baseline_latency(100.0)
        d = t.compute_metrics(50.0).to_dict()
        assert d["acceptance_rate"] == pytest.approx(0.75)
        assert d["quality_degradation"] is None
        t.reset()
        assert t.draft_tokens_per_step == []
        assert t.baseline_latency_ms is None


# --------------------------------------------------------------------------- #
# Fake NVML device layer
# --------------------------------------------------------------------------- #


class _FakeNVMLError(Exception):
    pass


class _MemInfo:
    def __init__(self, total, used):
        self.total = total
        self.used = used


class _UtilRates:
    def __init__(self, gpu, memory):
        self.gpu = gpu
        self.memory = memory


def _make_fake_pynvml(
    *,
    device_count=2,
    init_raises=False,
    power_read_raises_on=frozenset(),
    power_limit_raises_on=frozenset(),
    mem_info_raises_on=frozenset(),
):
    """Fake pynvml module: 2 GPUs, deterministic readings, opt-in failures."""
    mod = types.ModuleType("pynvml")
    mod.NVMLError = _FakeNVMLError
    mod.NVML_TEMPERATURE_GPU = 0
    mod.NVML_PCIE_UTIL_TX_BYTES = 1
    mod.NVML_PCIE_UTIL_RX_BYTES = 2
    state = {"shutdown_calls": 0}
    mod._state = state

    def nvmlInit():
        if init_raises:
            raise _FakeNVMLError("no NVML on this box")

    def nvmlShutdown():
        state["shutdown_calls"] += 1

    mod.nvmlInit = nvmlInit
    mod.nvmlShutdown = nvmlShutdown
    mod.nvmlDeviceGetCount = lambda: device_count
    mod.nvmlDeviceGetHandleByIndex = lambda i: f"handle-{i}"

    def nvmlDeviceGetMemoryInfo(h):
        if h in mem_info_raises_on:
            raise _FakeNVMLError("memory info read failed")
        return _MemInfo(
            total=16 * 1024 * 1024 * 1024,           # 16384 MB per GPU
            used=(4 if h == "handle-0" else 8) * 1024 * 1024 * 1024,
        )

    mod.nvmlDeviceGetMemoryInfo = nvmlDeviceGetMemoryInfo

    def nvmlDeviceGetPowerManagementLimit(h):
        if h in power_limit_raises_on:
            raise _FakeNVMLError("power limit read failed")
        return 300_000  # 300 W in mW

    mod.nvmlDeviceGetPowerManagementLimit = nvmlDeviceGetPowerManagementLimit
    mod.nvmlDeviceGetName = lambda h: f"Fake GPU {h[-1]}"
    mod.nvmlSystemGetDriverVersion = lambda: "555.42.02"
    mod.nvmlSystemGetCudaDriverVersion = lambda: 12040
    mod.nvmlDeviceGetUtilizationRates = lambda h: _UtilRates(
        gpu=50 if h == "handle-0" else 90, memory=40
    )

    def nvmlDeviceGetPowerUsage(h):
        if h in power_read_raises_on:
            raise _FakeNVMLError("power read failed")
        return 200_000  # 200 W in mW

    mod.nvmlDeviceGetPowerUsage = nvmlDeviceGetPowerUsage
    mod.nvmlDeviceGetTemperature = lambda h, kind: 60 if h == "handle-0" else 70
    mod.nvmlDeviceGetPcieThroughput = lambda h, kind: (
        1024 * 1024 if kind == mod.NVML_PCIE_UTIL_TX_BYTES else 2 * 1024 * 1024
    )
    return mod


# --------------------------------------------------------------------------- #
# GPUMetricsTracker on the fake device layer
# --------------------------------------------------------------------------- #


class TestGPUMetricsTracker:
    def test_init_and_static_device_info(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pynvml", _make_fake_pynvml())
        t = GPUMetricsTracker()
        assert t.is_available() is True
        assert t.get_device_count() == 2
        assert t.get_device_names() == ["Fake GPU 0", "Fake GPU 1"]
        assert t._total_memory == [16384.0, 16384.0]
        assert t._power_limits == [300.0, 300.0]

    def test_sample_once_returns_latest_per_gpu_readings(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pynvml", _make_fake_pynvml())
        t = GPUMetricsTracker()
        sample = t.sample_once()
        assert sample is not None
        assert sample["gpu_utilization"] == [50, 90]
        assert sample["memory_used_mb"] == [4096.0, 8192.0]
        assert sample["power_watts"] == [200.0, 200.0]
        assert sample["temperature_c"] == [60, 70]

    def test_compute_metrics_aggregation_arithmetic(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pynvml", _make_fake_pynvml())
        t = GPUMetricsTracker()
        t._sample_gpu_metrics()
        t._sample_gpu_metrics()
        m = t.compute_metrics()
        assert m.gpu_count == 2
        assert m.sample_count == 2  # ADR-0148: two sampling ticks
        assert m.avg_gpu_utilization == pytest.approx(70.0)  # mean(50, 90)
        assert m.max_gpu_utilization == pytest.approx(90.0)
        assert m.total_memory_mb == pytest.approx(32768.0)
        assert m.used_memory_mb == pytest.approx(6144.0)  # mean(4096, 8192)
        assert m.peak_memory_mb == pytest.approx(8192.0)
        assert m.memory_usage_percent == pytest.approx(6144.0 / 32768.0 * 100)
        assert m.avg_power_watts == pytest.approx(200.0)
        assert m.power_limit_watts == pytest.approx(600.0)
        assert m.avg_temperature_c == pytest.approx(65.0)
        assert m.max_temperature_c == pytest.approx(70.0)
        # PCIe totals: 2 samples x 2 GPUs x (1 MB tx, 2 MB rx).
        assert m.pcie_tx_mb == pytest.approx(4.0)
        assert m.pcie_rx_mb == pytest.approx(8.0)
        d = m.to_dict()
        assert d["gpu_count"] == 2 and d["pcie_rx_mb"] == pytest.approx(8.0)
        assert d["sample_count"] == 2

    def test_failed_per_call_read_is_dropped_not_zero(self, monkeypatch):
        # GPU 1's power read fails: the None must be DROPPED from the mean,
        # never folded in as a fake zero (the E2b absence-is-not-zero class).
        fake = _make_fake_pynvml(power_read_raises_on=frozenset({"handle-1"}))
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        t = GPUMetricsTracker()
        t._sample_gpu_metrics()
        assert t.power_samples == [[200.0, None]]
        m = t.compute_metrics()
        assert m.avg_power_watts == pytest.approx(200.0)  # NOT 100.0

    def test_no_samples_returns_none_with_static_info(self, monkeypatch):
        # ADR-0148 (2026-10-07; BACKLOG S0F-43): before it a tracker that never
        # sampled returned 0.0 for utilization, power, memory and temperature,
        # and the window's metrics.json gpu block read as a measured idle GPU.
        # The static device facts (count, total memory, power limit) stay.
        monkeypatch.setitem(sys.modules, "pynvml", _make_fake_pynvml())
        t = GPUMetricsTracker()
        m = t.compute_metrics()
        assert m.gpu_count == 2
        assert m.sample_count == 0
        for field in (
            "avg_gpu_utilization", "max_gpu_utilization", "avg_memory_utilization",
            "used_memory_mb", "peak_memory_mb", "memory_usage_percent",
            "avg_power_watts", "max_power_watts", "avg_temperature_c",
            "max_temperature_c", "pcie_tx_mb", "pcie_rx_mb",
        ):
            assert getattr(m, field) is None, field
        assert m.total_memory_mb == pytest.approx(32768.0)
        assert m.power_limit_watts == pytest.approx(600.0)
        assert json.loads(json.dumps(m.to_dict()))["avg_power_watts"] is None

    def test_failed_power_limit_read_is_none_not_zero(self, monkeypatch):
        # ADR-0148: a device whose power limit could not be read stored 0.0 W
        # and the summed limit understated the box; the limit is None instead.
        fake = _make_fake_pynvml(power_limit_raises_on=frozenset({"handle-1"}))
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        t = GPUMetricsTracker()
        assert t._power_limits == [300.0, None]
        t._sample_gpu_metrics()
        m = t.compute_metrics()
        assert m.power_limit_watts is None
        assert m.avg_power_watts == pytest.approx(200.0)  # the draw itself was read

    def test_failed_total_memory_read_is_none_and_keeps_every_device(self, monkeypatch):
        # ADR-0148 review LOW-2: an unguarded memory-info read on device 1 left
        # one total-memory entry and a box total that understated by half,
        # while the tracker still reported two devices. The read is guarded like
        # the power limit: None for that device, box total None, both devices kept.
        fake = _make_fake_pynvml(mem_info_raises_on=frozenset({"handle-1"}))
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        t = GPUMetricsTracker()
        assert t.is_available() and t.get_device_count() == 2
        assert t._total_memory == [16384.0, None]
        assert t._power_limits == [300.0, 300.0]       # the later reads still ran
        t._sample_gpu_metrics()
        m = t.compute_metrics()
        assert m.gpu_count == 2 and m.sample_count == 1
        assert m.total_memory_mb is None and m.memory_usage_percent is None
        assert m.power_limit_watts == pytest.approx(600.0)
        assert m.avg_gpu_utilization == pytest.approx(70.0)

    def test_field_failing_on_every_tick_is_none_while_others_measured(self, monkeypatch):
        # ADR-0148: every power read failed on every device; the power fields
        # are None while utilization, memory and temperature keep their means.
        fake = _make_fake_pynvml(power_read_raises_on=frozenset({"handle-0", "handle-1"}))
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        t = GPUMetricsTracker()
        t._sample_gpu_metrics()
        m = t.compute_metrics()
        assert m.sample_count == 1
        assert m.avg_power_watts is None and m.max_power_watts is None
        assert m.avg_gpu_utilization == pytest.approx(70.0)
        assert m.used_memory_mb == pytest.approx(6144.0)
        assert m.avg_temperature_c == pytest.approx(65.0)

    def test_nvml_init_failure_disables_tracker(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pynvml", _make_fake_pynvml(init_raises=True))
        t = GPUMetricsTracker()
        assert t.is_available() is False
        assert t.get_device_count() == 0
        assert t.get_device_names() == []
        assert t.sample_once() is None
        assert t.start_monitoring() is False
        m = t.compute_metrics()
        assert m.gpu_count == 0 and m.sample_count == 0
        assert m.total_memory_mb is None and m.power_limit_watts is None  # ADR-0148: no device read

    def test_pynvml_import_error_disables_tracker(self, monkeypatch):
        # sys.modules[name] = None makes `import pynvml` raise ImportError.
        monkeypatch.setitem(sys.modules, "pynvml", None)
        t = GPUMetricsTracker()
        assert t.is_available() is False
        assert t.sample_once() is None

    def test_reset_clears_samples(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pynvml", _make_fake_pynvml())
        t = GPUMetricsTracker()
        t._sample_gpu_metrics()
        t.reset()
        assert t.gpu_util_samples == []
        assert t.compute_metrics().avg_gpu_utilization is None  # ADR-0148

    def test_shutdown_calls_nvml_shutdown_once_and_disables(self, monkeypatch):
        fake = _make_fake_pynvml()
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        t = GPUMetricsTracker()
        t.shutdown()
        assert fake._state["shutdown_calls"] == 1
        assert t._nvml_initialized is False
        t.shutdown()  # idempotent: no second NVML call
        assert fake._state["shutdown_calls"] == 1
