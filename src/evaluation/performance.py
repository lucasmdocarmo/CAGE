"""
Performance metrics for CAGE evaluation.

Metrics:
- Throughput (QPS, tokens/sec)
- Latency (TTFT, TPOT, end-to-end)
- Resource utilization (CPU, memory, GPU)
- GPU utilization and memory (via pynvml)

Absence rule (ADR-0148, 2026-10-07): a count of our own events (requests,
tokens, errors, samples) is 0 on an empty set; every quantity derived from
readings (mean, max, percentile, ratio, rate, sum of device readings) is None
when there is no reading. Each result carries the sample count that explains
its None, so a reader can tell a measured zero from a value never measured.
Before this rule the trackers wrote 0.0 where nothing was sampled, and a
window with no successful request or no NVML tick read as a measured idle
system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Sequence
import time
import numpy as np
import psutil
from collections import defaultdict


def _mean_or_none(values: Sequence[float]) -> Optional[float]:
    """Arithmetic mean of the readings; None when there is none (ADR-0148)."""
    return float(np.mean(values)) if values else None


def _max_or_none(values: Sequence[float]) -> Optional[float]:
    """Maximum of the readings; None when there is none (ADR-0148)."""
    return float(np.max(values)) if values else None


def _percentile_or_none(values: Sequence[float], q: float) -> Optional[float]:
    """The q-th percentile of the readings (numpy's linear interpolation);
    None when there is none (ADR-0148)."""
    return float(np.percentile(values, q)) if values else None


@dataclass
class PerformanceMetrics:
    """Performance evaluation results.

    Absence rule (ADR-0148): the latency, throughput, TPOT and resource fields
    are None when no reading exists (no successful request, no request with a
    positive generation interval, no resource sample); ``total_requests``,
    ``total_tokens``, ``error_count`` and the two time spans stay numbers.
    ``tpot_sample_count`` and ``resource_sample_count`` say how many readings
    stand behind the TPOT and the CPU/memory fields.
    """

    # Throughput (None when the summed serving time is 0)
    queries_per_second: Optional[float]
    tokens_per_second: Optional[float]

    # Latency (None with no successful request)
    avg_ttft_ms: Optional[float]  # Average time to first token
    p50_ttft_ms: Optional[float]  # Median TTFT
    p95_ttft_ms: Optional[float]  # 95th percentile TTFT
    p99_ttft_ms: Optional[float]  # 99th percentile TTFT

    # TPOT - Time Per Output Token (sustained generation speed); None when no
    # request had a positive generation interval (tpot_sample_count == 0)
    avg_tpot_ms: Optional[float]  # Average time per output token
    p50_tpot_ms: Optional[float]  # Median TPOT
    p95_tpot_ms: Optional[float]  # 95th percentile TPOT
    p99_tpot_ms: Optional[float]  # 99th percentile TPOT

    avg_latency_ms: Optional[float]  # Average end-to-end latency
    p50_latency_ms: Optional[float]
    p95_latency_ms: Optional[float]
    p99_latency_ms: Optional[float]

    # Resource utilization of the runner process (None with no sample)
    avg_cpu_percent: Optional[float]
    avg_memory_mb: Optional[float]
    peak_memory_mb: Optional[float]

    # Additional stats
    total_requests: int
    total_tokens: int
    total_time_seconds: float          # wall-clock span of the measured stage (incl. inline CPU scoring)
    serving_time_seconds: float        # summed per-request serving time over successful requests; denominator for throughput
    error_count: int = 0
    tpot_sample_count: int = 0         # requests with a positive generation interval (ADR-0148)
    resource_sample_count: int = 0     # CPU/memory samples taken (ADR-0148)

    def to_dict(self) -> Dict[str, Optional[float]]:
        """Convert to dictionary (None stays None: absence is not zero)."""
        return {
            "queries_per_second": self.queries_per_second,
            "tokens_per_second": self.tokens_per_second,
            "avg_ttft_ms": self.avg_ttft_ms,
            "p50_ttft_ms": self.p50_ttft_ms,
            "p95_ttft_ms": self.p95_ttft_ms,
            "p99_ttft_ms": self.p99_ttft_ms,
            "avg_tpot_ms": self.avg_tpot_ms,
            "p50_tpot_ms": self.p50_tpot_ms,
            "p95_tpot_ms": self.p95_tpot_ms,
            "p99_tpot_ms": self.p99_tpot_ms,
            "avg_latency_ms": self.avg_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "avg_cpu_percent": self.avg_cpu_percent,
            "avg_memory_mb": self.avg_memory_mb,
            "peak_memory_mb": self.peak_memory_mb,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_time_seconds": self.total_time_seconds,
            "serving_time_seconds": self.serving_time_seconds,
            "error_count": self.error_count,
            "tpot_sample_count": self.tpot_sample_count,
            "resource_sample_count": self.resource_sample_count,
        }


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    
    request_id: str
    ttft_ms: float
    total_time_ms: float
    num_tokens: int
    error: Optional[str] = None


class PerformanceEvaluator:
    """Tracks and computes performance metrics."""
    
    def __init__(self, monitor_resources: bool = True):
        self.monitor_resources = monitor_resources
        
        # Timing
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
        # Request metrics
        self.request_metrics: List[RequestMetrics] = []
        
        # Resource monitoring
        self.cpu_samples: List[float] = []
        self.memory_samples: List[float] = []
        self._monitoring = False
        
        # Process handle for resource monitoring
        self.process = psutil.Process() if monitor_resources else None
    
    def start(self) -> None:
        """Start performance tracking."""
        self.start_time = time.time()
        self._monitoring = True
        
        if self.monitor_resources:
            self._sample_resources()
    
    def stop(self) -> None:
        """Stop performance tracking."""
        self.end_time = time.time()
        self._monitoring = False
        
        if self.monitor_resources:
            self._sample_resources()
    
    def _sample_resources(self) -> None:
        """Sample CPU and memory usage."""
        if not self.process:
            return
        
        try:
            cpu_percent = self.process.cpu_percent()
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            
            self.cpu_samples.append(cpu_percent)
            self.memory_samples.append(memory_mb)
        except Exception as e:
            print(f"Warning: Failed to sample resources: {e}")
    
    def record_request(
        self,
        request_id: str,
        ttft_ms: float,
        total_time_ms: float,
        num_tokens: int,
        error: Optional[str] = None,
    ) -> None:
        """Record metrics for a single request."""
        self.request_metrics.append(
            RequestMetrics(
                request_id=request_id,
                ttft_ms=ttft_ms,
                total_time_ms=total_time_ms,
                num_tokens=num_tokens,
                error=error,
            )
        )
        
        # Sample resources periodically
        if self._monitoring and self.monitor_resources and len(self.request_metrics) % 10 == 0:
            self._sample_resources()
    
    def compute_metrics(self) -> PerformanceMetrics:
        """Compute aggregate performance metrics.

        ADR-0148: a field derived from readings is None when there is no
        reading; the counts and the two time spans are always numbers. Before
        the rule a window with no successful request returned 0.0 everywhere
        and read as a window served at zero latency.
        """
        if not self.start_time or not self.end_time:
            raise ValueError("Must call start() and stop() before computing metrics")

        total_time = self.end_time - self.start_time

        # Filter out errors
        successful_requests = [
            req for req in self.request_metrics if req.error is None
        ]
        error_count = len(self.request_metrics) - len(successful_requests)

        # Extract metrics (every list is empty on an all-error window)
        ttfts = [req.ttft_ms for req in successful_requests]
        latencies = [req.total_time_ms for req in successful_requests]
        tokens = [req.num_tokens for req in successful_requests]

        total_tokens = sum(tokens)
        total_requests = len(successful_requests)

        # Compute throughput over the SUMMED per-request serving time, not the wall-clock
        # window. The measured loop runs inline CPU quality scoring (LettuceDetect/NLI/BERTScore)
        # after each generation, so wall-clock is dominated by evaluation + GPU idle between
        # sequential requests and understates serving throughput by ~4x. Summing per-request
        # latencies yields the true single-stream (back-to-back) serving rate, matching the
        # parallel computation in run_experiment.py. Pure decode speed is 1000/avg_tpot_ms.
        # A zero serving time has no rate (None), never a 0.0 QPS.
        serving_time = sum(latencies) / 1000.0
        qps = total_requests / serving_time if serving_time > 0 else None
        tps = total_tokens / serving_time if serving_time > 0 else None

        # Compute TPOT (Time Per Output Token) = mean inter-token latency.
        # TTFT already accounts for the FIRST token, so the time after the first token
        # produced (num_tokens - 1) tokens -> divide by (num_tokens - 1), not num_tokens.
        # Single-token outputs have no inter-token interval and are excluded (dividing by
        # num_tokens=1 would fold a spurious ~0 into the distribution and understate TPOT).
        # Review fix: on non-streaming paths (e.g. VLLMOfflineAdapter's --offline debug
        # engine), ttft_ms is deliberately set equal to total_time_ms because TTFT is
        # unobservable there (reported as the full response time). That makes
        # generation_time_ms == 0, which would silently fold a spurious ~0 TPOT into the
        # distribution rather than being excluded as unmeasurable. Require a positive
        # generation interval, not just num_tokens > 1.
        tpots = []
        for req in successful_requests:
            if req.num_tokens > 1 and req.total_time_ms > req.ttft_ms:
                generation_time_ms = req.total_time_ms - req.ttft_ms
                tpot = generation_time_ms / (req.num_tokens - 1)
                tpots.append(tpot)

        return PerformanceMetrics(
            queries_per_second=qps,
            tokens_per_second=tps,
            avg_ttft_ms=_mean_or_none(ttfts),
            p50_ttft_ms=_percentile_or_none(ttfts, 50),
            p95_ttft_ms=_percentile_or_none(ttfts, 95),
            p99_ttft_ms=_percentile_or_none(ttfts, 99),
            avg_tpot_ms=_mean_or_none(tpots),
            p50_tpot_ms=_percentile_or_none(tpots, 50),
            p95_tpot_ms=_percentile_or_none(tpots, 95),
            p99_tpot_ms=_percentile_or_none(tpots, 99),
            avg_latency_ms=_mean_or_none(latencies),
            p50_latency_ms=_percentile_or_none(latencies, 50),
            p95_latency_ms=_percentile_or_none(latencies, 95),
            p99_latency_ms=_percentile_or_none(latencies, 99),
            # Resource utilization of the runner process (None with no sample)
            avg_cpu_percent=_mean_or_none(self.cpu_samples),
            avg_memory_mb=_mean_or_none(self.memory_samples),
            peak_memory_mb=_max_or_none(self.memory_samples),
            total_requests=total_requests + error_count,
            total_tokens=total_tokens,
            total_time_seconds=total_time,
            serving_time_seconds=serving_time,
            error_count=error_count,
            tpot_sample_count=len(tpots),
            resource_sample_count=len(self.memory_samples),
        )
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.start_time = None
        self.end_time = None
        self.request_metrics.clear()
        self.cpu_samples.clear()
        self.memory_samples.clear()
        self._monitoring = False


@dataclass
class SpeculativeMetrics:
    """Metrics for speculative decoding evaluation."""
    
    # Core speculative decoding metrics
    acceptance_rate: float  # Accepted draft tokens / Total draft tokens
    avg_draft_tokens: float  # Average draft tokens proposed per step
    avg_accepted_tokens: float  # Average tokens accepted per step
    total_draft_tokens: int  # Total draft tokens proposed
    total_accepted_tokens: int  # Total tokens accepted
    total_rejected_tokens: int  # Total tokens rejected (rollbacks)
    
    # Performance impact
    speedup_ratio: float  # Compared to non-speculative baseline
    rollback_overhead_ms: float  # Average rollback latency
    
    # Quality impact (optional - may be None if not measured)
    quality_degradation: Optional[float] = None  # Difference in quality vs non-speculative
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "acceptance_rate": self.acceptance_rate,
            "avg_draft_tokens": self.avg_draft_tokens,
            "avg_accepted_tokens": self.avg_accepted_tokens,
            "total_draft_tokens": self.total_draft_tokens,
            "total_accepted_tokens": self.total_accepted_tokens,
            "total_rejected_tokens": self.total_rejected_tokens,
            "speedup_ratio": self.speedup_ratio,
            "rollback_overhead_ms": self.rollback_overhead_ms,
            "quality_degradation": self.quality_degradation,
        }


class SpeculativeMetricsTracker:
    """Tracks speculative decoding metrics per request."""
    
    def __init__(self):
        self.draft_tokens_per_step: List[int] = []
        self.accepted_tokens_per_step: List[int] = []
        self.rollback_latencies: List[float] = []
        self.baseline_latency_ms: Optional[float] = None  # For speedup calculation
    
    def record_step(
        self,
        draft_tokens: int,
        accepted_tokens: int,
        rollback_latency_ms: float = 0.0,
    ) -> None:
        """Record metrics for a single speculative decoding step."""
        self.draft_tokens_per_step.append(draft_tokens)
        self.accepted_tokens_per_step.append(accepted_tokens)
        if rollback_latency_ms > 0:
            self.rollback_latencies.append(rollback_latency_ms)
    
    def set_baseline_latency(self, latency_ms: float) -> None:
        """Set baseline (non-speculative) latency for speedup calculation."""
        self.baseline_latency_ms = latency_ms
    
    def compute_metrics(self, actual_latency_ms: float) -> SpeculativeMetrics:
        """Compute aggregate speculative decoding metrics."""
        import numpy as np
        
        total_draft = sum(self.draft_tokens_per_step)
        total_accepted = sum(self.accepted_tokens_per_step)
        total_rejected = total_draft - total_accepted
        
        acceptance_rate = total_accepted / total_draft if total_draft > 0 else 0.0
        avg_draft = float(np.mean(self.draft_tokens_per_step)) if self.draft_tokens_per_step else 0.0
        avg_accepted = float(np.mean(self.accepted_tokens_per_step)) if self.accepted_tokens_per_step else 0.0
        avg_rollback = float(np.mean(self.rollback_latencies)) if self.rollback_latencies else 0.0
        
        # Compute speedup if baseline is available
        speedup = 1.0
        if self.baseline_latency_ms and self.baseline_latency_ms > 0 and actual_latency_ms > 0:
            speedup = self.baseline_latency_ms / actual_latency_ms
        
        return SpeculativeMetrics(
            acceptance_rate=acceptance_rate,
            avg_draft_tokens=avg_draft,
            avg_accepted_tokens=avg_accepted,
            total_draft_tokens=total_draft,
            total_accepted_tokens=total_accepted,
            total_rejected_tokens=total_rejected,
            speedup_ratio=speedup,
            rollback_overhead_ms=avg_rollback,
        )
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.draft_tokens_per_step.clear()
        self.accepted_tokens_per_step.clear()
        self.rollback_latencies.clear()
        self.baseline_latency_ms = None


@dataclass
class GPUMetrics:
    """GPU utilization metrics.

    Provides nvidia-smi equivalent metrics via pynvml.
    Used for Layer 2 GPU metrics as per paper requirements.

    Absence rule (ADR-0148; BACKLOG S0F-43): every field derived from NVML
    readings is None when no reading exists (no sampling tick, or every read
    of that field failed); ``gpu_count`` and ``sample_count`` are counts;
    ``total_memory_mb`` and ``power_limit_watts`` are the static device facts
    read at init, None when a device's value could not be read.
    """

    # Per-GPU metrics (aggregated across devices)
    gpu_count: int  # Number of GPUs detected
    sample_count: int  # Sampling ticks recorded (ADR-0148)

    # Utilization (0-100%)
    avg_gpu_utilization: Optional[float]  # Average GPU compute utilization
    max_gpu_utilization: Optional[float]  # Peak GPU utilization
    avg_memory_utilization: Optional[float]  # Average GPU memory bandwidth utilization

    # Memory (MB)
    total_memory_mb: Optional[float]  # Total GPU memory across all devices
    used_memory_mb: Optional[float]  # Mean GPU memory in use over the samples
    peak_memory_mb: Optional[float]  # Peak GPU memory usage
    memory_usage_percent: Optional[float]  # mean per-device used memory over the box total (1/N of the box fraction with N equal GPUs; multi-GPU arithmetic is BACKLOG S0F-48)

    # Power (Watts)
    avg_power_watts: Optional[float]  # Average power draw
    max_power_watts: Optional[float]  # Peak power draw
    power_limit_watts: Optional[float]  # Power limit, summed over devices (None when any read failed)

    # Temperature (Celsius)
    avg_temperature_c: Optional[float]  # Average GPU temperature
    max_temperature_c: Optional[float]  # Peak GPU temperature

    # PCIe (for distributed systems)
    pcie_tx_mb: Optional[float]  # PCIe TX throughput (MB)
    pcie_rx_mb: Optional[float]  # PCIe RX throughput (MB)

    def to_dict(self) -> Dict[str, Optional[float]]:
        """Convert to dictionary (None stays None: absence is not zero)."""
        return {
            "gpu_count": self.gpu_count,
            "sample_count": self.sample_count,
            "avg_gpu_utilization": self.avg_gpu_utilization,
            "max_gpu_utilization": self.max_gpu_utilization,
            "avg_memory_utilization": self.avg_memory_utilization,
            "total_memory_mb": self.total_memory_mb,
            "used_memory_mb": self.used_memory_mb,
            "peak_memory_mb": self.peak_memory_mb,
            "memory_usage_percent": self.memory_usage_percent,
            "avg_power_watts": self.avg_power_watts,
            "max_power_watts": self.max_power_watts,
            "power_limit_watts": self.power_limit_watts,
            "avg_temperature_c": self.avg_temperature_c,
            "max_temperature_c": self.max_temperature_c,
            "pcie_tx_mb": self.pcie_tx_mb,
            "pcie_rx_mb": self.pcie_rx_mb,
        }


class GPUMetricsTracker:
    """Tracks GPU metrics using pynvml (NVIDIA Management Library).

    [VERIFY-LIVE at S0] (K-COV5, task #141): the NVML acquisition side
    (init, device handles, sampling loop, PCIe counters) can only execute on
    a real NVIDIA GPU — the offline suite pins the pure aggregation
    arithmetic (tests/test_cov_performance_trackers.py, #142) but never this
    stack. S0 checklist row S0-10 (MyDocs/S0_CHECKLIST.md) forces the live
    proof.

    This provides nvidia-smi equivalent functionality programmatically.
    Gracefully handles systems without NVIDIA GPUs.
    
    Example usage:
        tracker = GPUMetricsTracker()
        if tracker.is_available():
            tracker.start_monitoring()
            # ... run workload ...
            tracker.stop_monitoring()
            metrics = tracker.compute_metrics()
    """
    
    def __init__(self, sample_interval_ms: float = 100):
        """Initialize GPU metrics tracker.
        
        Args:
            sample_interval_ms: How often to sample GPU metrics (default 100ms)
        """
        self.sample_interval_ms = sample_interval_ms
        self._nvml_initialized = False
        self._monitoring = False
        self._monitor_thread = None
        
        # Samples storage
        self.gpu_util_samples: List[List[float]] = []  # Per-GPU utilization
        self.memory_util_samples: List[List[float]] = []  # Per-GPU memory bandwidth util
        self.memory_used_samples: List[List[float]] = []  # Per-GPU memory used (MB)
        self.power_samples: List[List[float]] = []  # Per-GPU power draw (W)
        self.temp_samples: List[List[float]] = []  # Per-GPU temperature (C)
        self.pcie_tx_samples: List[List[int]] = []  # Per-GPU PCIe TX (bytes)
        self.pcie_rx_samples: List[List[int]] = []  # Per-GPU PCIe RX (bytes)
        
        # Device info (populated on init)
        self._device_count = 0
        self._device_handles: List = []
        self._total_memory: List[Optional[float]] = []  # Per-GPU total memory (MB); None = read failed
        self._power_limits: List[Optional[float]] = []  # Per-GPU power limit (W); None = read failed
        
        # Try to initialize NVML
        self._init_nvml()
    
    def _init_nvml(self) -> bool:
        """Initialize NVML library."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_initialized = True
            
            # Get device count and handles
            self._device_count = pynvml.nvmlDeviceGetCount()
            self._device_handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i)
                for i in range(self._device_count)
            ]
            
            # Get static device info. A failed read is None, never 0 and never
            # a silently shorter list (ADR-0148 review LOW-2: an unguarded read
            # on device k left k entries and a box total that understated).
            for handle in self._device_handles:
                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    self._total_memory.append(mem_info.total / 1024 / 1024)  # MB
                except pynvml.NVMLError:
                    self._total_memory.append(None)

                try:
                    power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
                    self._power_limits.append(power_limit / 1000)  # W
                except pynvml.NVMLError:
                    self._power_limits.append(None)  # ADR-0148: unread, not 0 W
            
            return True
            
        except ImportError:
            print("Warning: pynvml not installed. GPU metrics disabled.")
            print("Install with: pip install nvidia-ml-py")
            return False
        except Exception as e:
            print(f"Warning: Failed to initialize NVML: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if GPU monitoring is available."""
        return self._nvml_initialized and self._device_count > 0
    
    def get_device_count(self) -> int:
        """Get number of GPUs detected."""
        return self._device_count
    
    def get_device_names(self) -> List[str]:
        """Get names of all detected GPUs."""
        if not self.is_available():
            return []
        
        try:
            import pynvml
            return [
                pynvml.nvmlDeviceGetName(handle)
                for handle in self._device_handles
            ]
        except Exception:
            return []
    
    def _sample_gpu_metrics(self) -> None:
        """Sample current GPU metrics from all devices."""
        if not self.is_available():
            return
        
        try:
            import pynvml
            
            gpu_utils = []
            mem_utils = []
            mem_used = []
            powers = []
            temps = []
            pcie_tx = []
            pcie_rx = []
            
            for handle in self._device_handles:
                # GPU utilization
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_utils.append(util.gpu)
                    mem_utils.append(util.memory)
                except pynvml.NVMLError:
                    gpu_utils.append(None)
                    mem_utils.append(None)
                
                # Memory usage
                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem_used.append(mem_info.used / 1024 / 1024)  # MB
                except pynvml.NVMLError:
                    mem_used.append(None)
                
                # Power draw
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle)
                    powers.append(power / 1000)  # Convert mW to W
                except pynvml.NVMLError:
                    powers.append(None)
                
                # Temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                    temps.append(temp)
                except pynvml.NVMLError:
                    temps.append(None)
                
                # PCIe throughput
                try:
                    tx = pynvml.nvmlDeviceGetPcieThroughput(
                        handle, pynvml.NVML_PCIE_UTIL_TX_BYTES
                    )
                    rx = pynvml.nvmlDeviceGetPcieThroughput(
                        handle, pynvml.NVML_PCIE_UTIL_RX_BYTES
                    )
                    pcie_tx.append(tx)
                    pcie_rx.append(rx)
                except pynvml.NVMLError:
                    pcie_tx.append(None)
                    pcie_rx.append(None)
            
            # Store samples
            self.gpu_util_samples.append(gpu_utils)
            self.memory_util_samples.append(mem_utils)
            self.memory_used_samples.append(mem_used)
            self.power_samples.append(powers)
            self.temp_samples.append(temps)
            self.pcie_tx_samples.append(pcie_tx)
            self.pcie_rx_samples.append(pcie_rx)
            
        except Exception as e:
            print(f"Warning: Error sampling GPU metrics: {e}")
    
    def _monitoring_loop(self) -> None:
        """Background thread for continuous GPU monitoring."""
        import time
        
        while self._monitoring:
            self._sample_gpu_metrics()
            time.sleep(self.sample_interval_ms / 1000)
    
    def start_monitoring(self) -> bool:
        """Start continuous GPU monitoring in background thread.
        
        Returns:
            True if monitoring started successfully, False otherwise
        """
        if not self.is_available():
            return False
        
        if self._monitoring:
            return True  # Already monitoring
        
        import threading
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
        )
        self._monitor_thread.start()
        return True
    
    def stop_monitoring(self) -> None:
        """Stop GPU monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
            self._monitor_thread = None
    
    def sample_once(self) -> Optional[Dict[str, List[float]]]:
        """Take a single sample of GPU metrics.
        
        Returns:
            Dict with current GPU metrics, or None if unavailable
        """
        if not self.is_available():
            return None
        
        self._sample_gpu_metrics()
        
        if not self.gpu_util_samples:
            return None
        
        return {
            "gpu_utilization": self.gpu_util_samples[-1],
            "memory_utilization": self.memory_util_samples[-1],
            "memory_used_mb": self.memory_used_samples[-1],
            "power_watts": self.power_samples[-1],
            "temperature_c": self.temp_samples[-1],
        }
    
    def compute_metrics(self) -> GPUMetrics:
        """Compute aggregate GPU metrics from collected samples.

        ADR-0148 (BACKLOG S0F-43): with no sampling tick, or when every read
        of one field failed, that field is None; before the rule it read 0.0
        and a window without NVML data looked like a measured idle GPU. The
        static device facts (count, total memory, power limit) do not depend
        on sampling.

        Returns:
            GPUMetrics with aggregated statistics
        """
        # Flatten samples, dropping None (a failed per-call NVML read, not a real zero).
        all_gpu_utils = [u for sample in self.gpu_util_samples for u in sample if u is not None]
        all_mem_utils = [u for sample in self.memory_util_samples for u in sample if u is not None]
        all_mem_used = [m for sample in self.memory_used_samples for m in sample if m is not None]
        all_powers = [p for sample in self.power_samples for p in sample if p is not None]
        all_temps = [t for sample in self.temp_samples for t in sample if t is not None]
        all_pcie_tx = [tx for sample in self.pcie_tx_samples for tx in sample if tx is not None]
        all_pcie_rx = [rx for sample in self.pcie_rx_samples for rx in sample if rx is not None]

        # Static device facts, read at init (None when a device's read failed:
        # a partial box total would understate the divisor of memory_usage_percent).
        total_memory = (
            sum(self._total_memory)
            if self._total_memory and all(m is not None for m in self._total_memory)
            else None
        )
        power_limit = (
            sum(self._power_limits)
            if self._power_limits and all(p is not None for p in self._power_limits)
            else None
        )

        used_memory = _mean_or_none(all_mem_used)
        memory_percent = (
            used_memory / total_memory * 100
            if used_memory is not None and total_memory else None
        )

        # PCIe throughput (cumulative over the samples)
        pcie_tx_mb = sum(all_pcie_tx) / 1024 / 1024 if all_pcie_tx else None  # Convert KB to MB
        pcie_rx_mb = sum(all_pcie_rx) / 1024 / 1024 if all_pcie_rx else None

        return GPUMetrics(
            gpu_count=self._device_count,
            sample_count=len(self.gpu_util_samples),
            avg_gpu_utilization=_mean_or_none(all_gpu_utils),
            max_gpu_utilization=_max_or_none(all_gpu_utils),
            avg_memory_utilization=_mean_or_none(all_mem_utils),
            total_memory_mb=total_memory,
            used_memory_mb=used_memory,
            peak_memory_mb=_max_or_none(all_mem_used),
            memory_usage_percent=memory_percent,
            avg_power_watts=_mean_or_none(all_powers),
            max_power_watts=_max_or_none(all_powers),
            power_limit_watts=power_limit,
            avg_temperature_c=_mean_or_none(all_temps),
            max_temperature_c=_max_or_none(all_temps),
            pcie_tx_mb=pcie_tx_mb,
            pcie_rx_mb=pcie_rx_mb,
        )
    
    def reset(self) -> None:
        """Reset all collected samples."""
        self.gpu_util_samples.clear()
        self.memory_util_samples.clear()
        self.memory_used_samples.clear()
        self.power_samples.clear()
        self.temp_samples.clear()
        self.pcie_tx_samples.clear()
        self.pcie_rx_samples.clear()
    
    def shutdown(self) -> None:
        """Shutdown NVML. Call when done with GPU monitoring."""
        self.stop_monitoring()
        
        if self._nvml_initialized:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_initialized = False
