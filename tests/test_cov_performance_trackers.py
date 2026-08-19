"""Offline coverage for src/evaluation/performance.py trackers (K-COV5, #142).

The speculative/cache tracker compute paths (pure-python acceptance-rate /
hit-rate arithmetic) and the NVML consumption stack were at 0% coverage. All
tests run offline: the device layer (pynvml) is a fake module injected via
sys.modules (the llmlingua-stub pattern of tests/test_compression_ops.py) —
no GPU, no NVML, no network.

Covers:
- PerformanceEvaluator: start/stop discipline, serving-time (not wall-clock)
  throughput denominator, TPOT (num_tokens-1) with the single-token and
  non-streaming (ttft == total) exclusions, error filtering, zero-metrics arm,
  to_dict round-trip, reset
- SpeculativeMetricsTracker: acceptance-rate arithmetic, rollback recording
  gate (>0 only), speedup with/without baseline, reset
- CacheMetricsTracker: hit/miss ratios, remote-fetch mean, transfer MB,
  zero-request arm, reset
- GPUMetricsTracker on the fake NVML: init/static info, sampling incl. the
  per-call NVMLError None arms (a failed read is dropped, never a fake zero),
  aggregation arithmetic, unavailable/import-error arms, shutdown
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from src.evaluation.performance import (
    CacheMetricsTracker,
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

    def test_all_errors_returns_zero_metrics_arm(self):
        ev = self._evaluator()
        ev.start()
        ev.record_request("e1", 0.0, 0.0, 0, error="x")
        ev.record_request("e2", 0.0, 0.0, 0, error="y")
        ev.stop()
        m = ev.compute_metrics()
        assert m.queries_per_second == 0.0
        assert m.tokens_per_second == 0.0
        assert m.serving_time_seconds == 0.0
        assert m.total_requests == 2
        assert m.error_count == 2

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
        assert m.avg_tpot_ms == 0.0
        assert m.p99_tpot_ms == 0.0
        # Both rows still count toward latency/throughput.
        assert m.total_requests == 2

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
# CacheMetricsTracker
# --------------------------------------------------------------------------- #


class TestCacheMetricsTracker:
    def test_zero_requests_arm(self):
        m = CacheMetricsTracker().get_metrics()
        assert m == {
            "local_hit_ratio": 0.0,
            "remote_hit_ratio": 0.0,
            "miss_ratio": 0.0,
            "total_hit_ratio": 0.0,
            "avg_remote_fetch_ms": 0.0,
            "total_transfer_mb": 0.0,
        }

    def test_hit_rate_arithmetic(self):
        t = CacheMetricsTracker()
        t.record_local_hit()
        t.record_local_hit()
        t.record_remote_hit(fetch_latency_ms=10.0, bytes_transferred=1024 * 1024)
        t.record_remote_hit(fetch_latency_ms=30.0, bytes_transferred=2 * 1024 * 1024)
        t.record_miss()
        m = t.get_metrics()
        assert m["local_hit_ratio"] == pytest.approx(2 / 5)
        assert m["remote_hit_ratio"] == pytest.approx(2 / 5)
        assert m["miss_ratio"] == pytest.approx(1 / 5)
        assert m["total_hit_ratio"] == pytest.approx(4 / 5)
        assert m["avg_remote_fetch_ms"] == pytest.approx(20.0)
        assert m["total_transfer_mb"] == pytest.approx(3.0)
        # Ratios partition the request population.
        assert m["local_hit_ratio"] + m["remote_hit_ratio"] + m["miss_ratio"] == (
            pytest.approx(1.0)
        )

    def test_reset(self):
        t = CacheMetricsTracker()
        t.record_remote_hit(5.0, 100)
        t.record_miss()
        t.reset()
        assert (t.local_hits, t.remote_hits, t.misses) == (0, 0, 0)
        assert t.remote_fetch_latencies == [] and t.transfer_bytes == []


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
    mod.nvmlDeviceGetMemoryInfo = lambda h: _MemInfo(
        total=16 * 1024 * 1024 * 1024,           # 16384 MB per GPU
        used=(4 if h == "handle-0" else 8) * 1024 * 1024 * 1024,
    )
    mod.nvmlDeviceGetPowerManagementLimit = lambda h: 300_000  # 300 W in mW
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

    def test_no_samples_returns_zero_metrics_with_static_info(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pynvml", _make_fake_pynvml())
        t = GPUMetricsTracker()
        m = t.compute_metrics()
        assert m.gpu_count == 2
        assert m.avg_gpu_utilization == 0.0
        assert m.total_memory_mb == pytest.approx(32768.0)
        assert m.power_limit_watts == pytest.approx(600.0)

    def test_nvml_init_failure_disables_tracker(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pynvml", _make_fake_pynvml(init_raises=True))
        t = GPUMetricsTracker()
        assert t.is_available() is False
        assert t.get_device_count() == 0
        assert t.get_device_names() == []
        assert t.sample_once() is None
        assert t.start_monitoring() is False
        assert t.compute_metrics().gpu_count == 0

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
        assert t.compute_metrics().avg_gpu_utilization == 0.0

    def test_shutdown_calls_nvml_shutdown_once_and_disables(self, monkeypatch):
        fake = _make_fake_pynvml()
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        t = GPUMetricsTracker()
        t.shutdown()
        assert fake._state["shutdown_calls"] == 1
        assert t._nvml_initialized is False
        t.shutdown()  # idempotent: no second NVML call
        assert fake._state["shutdown_calls"] == 1
