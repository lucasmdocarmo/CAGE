"""
Performance metrics for CAGE evaluation.

Metrics:
- Throughput (QPS, tokens/sec)
- Latency (TTFT, TPOT, end-to-end)
- Resource utilization (CPU, memory, GPU)
- GPU utilization and memory (via pynvml)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import time
import psutil
from collections import defaultdict


@dataclass
class PerformanceMetrics:
    """Performance evaluation results."""
    
    # Throughput
    queries_per_second: float
    tokens_per_second: float
    
    # Latency
    avg_ttft_ms: float  # Average time to first token
    p50_ttft_ms: float  # Median TTFT
    p95_ttft_ms: float  # 95th percentile TTFT
    p99_ttft_ms: float  # 99th percentile TTFT
    
    # TPOT - Time Per Output Token (sustained generation speed)
    avg_tpot_ms: float  # Average time per output token
    p50_tpot_ms: float  # Median TPOT
    p95_tpot_ms: float  # 95th percentile TPOT
    p99_tpot_ms: float  # 99th percentile TPOT
    
    avg_latency_ms: float  # Average end-to-end latency
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    
    # Resource utilization
    avg_cpu_percent: float
    avg_memory_mb: float
    peak_memory_mb: float
    
    # Additional stats
    total_requests: int
    total_tokens: int
    total_time_seconds: float          # wall-clock span of the measured stage (incl. inline CPU scoring)
    serving_time_seconds: float        # summed per-request serving time; denominator for throughput
    error_count: int = 0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
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
        """Compute aggregate performance metrics."""
        if not self.start_time or not self.end_time:
            raise ValueError("Must call start() and stop() before computing metrics")
        
        total_time = self.end_time - self.start_time
        
        # Filter out errors
        successful_requests = [
            req for req in self.request_metrics if req.error is None
        ]
        error_count = len(self.request_metrics) - len(successful_requests)
        
        if not successful_requests:
            # Return zero metrics if no successful requests
            return PerformanceMetrics(
                queries_per_second=0.0,
                tokens_per_second=0.0,
                avg_ttft_ms=0.0,
                p50_ttft_ms=0.0,
                p95_ttft_ms=0.0,
                p99_ttft_ms=0.0,
                avg_tpot_ms=0.0,
                p50_tpot_ms=0.0,
                p95_tpot_ms=0.0,
                p99_tpot_ms=0.0,
                avg_latency_ms=0.0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
                avg_cpu_percent=0.0,
                avg_memory_mb=0.0,
                peak_memory_mb=0.0,
                total_requests=len(self.request_metrics),
                total_tokens=0,
                total_time_seconds=total_time,
                serving_time_seconds=0.0,
                error_count=error_count,
            )
        
        # Extract metrics
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
        serving_time = sum(latencies) / 1000.0
        qps = total_requests / serving_time if serving_time > 0 else 0.0
        tps = total_tokens / serving_time if serving_time > 0 else 0.0
        
        # Compute latency percentiles
        import numpy as np
        
        avg_ttft = float(np.mean(ttfts))
        p50_ttft = float(np.percentile(ttfts, 50))
        p95_ttft = float(np.percentile(ttfts, 95))
        p99_ttft = float(np.percentile(ttfts, 99))
        
        # Compute TPOT (Time Per Output Token) = mean inter-token latency.
        # TTFT already accounts for the FIRST token, so the time after the first token
        # produced (num_tokens - 1) tokens -> divide by (num_tokens - 1), not num_tokens.
        # Single-token outputs have no inter-token interval and are excluded (dividing by
        # num_tokens=1 would fold a spurious ~0 into the distribution and understate TPOT).
        tpots = []
        for req in successful_requests:
            if req.num_tokens > 1:
                generation_time_ms = req.total_time_ms - req.ttft_ms
                tpot = generation_time_ms / (req.num_tokens - 1)
                tpots.append(tpot)
        
        if tpots:
            avg_tpot = float(np.mean(tpots))
            p50_tpot = float(np.percentile(tpots, 50))
            p95_tpot = float(np.percentile(tpots, 95))
            p99_tpot = float(np.percentile(tpots, 99))
        else:
            avg_tpot = p50_tpot = p95_tpot = p99_tpot = 0.0
        
        avg_latency = float(np.mean(latencies))
        p50_latency = float(np.percentile(latencies, 50))
        p95_latency = float(np.percentile(latencies, 95))
        p99_latency = float(np.percentile(latencies, 99))
        
        # Compute resource utilization
        avg_cpu = float(np.mean(self.cpu_samples)) if self.cpu_samples else 0.0
        avg_memory = float(np.mean(self.memory_samples)) if self.memory_samples else 0.0
        peak_memory = float(np.max(self.memory_samples)) if self.memory_samples else 0.0
        
        return PerformanceMetrics(
            queries_per_second=qps,
            tokens_per_second=tps,
            avg_ttft_ms=avg_ttft,
            p50_ttft_ms=p50_ttft,
            p95_ttft_ms=p95_ttft,
            p99_ttft_ms=p99_ttft,
            avg_tpot_ms=avg_tpot,
            p50_tpot_ms=p50_tpot,
            p95_tpot_ms=p95_tpot,
            p99_tpot_ms=p99_tpot,
            avg_latency_ms=avg_latency,
            p50_latency_ms=p50_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            avg_cpu_percent=avg_cpu,
            avg_memory_mb=avg_memory,
            peak_memory_mb=peak_memory,
            total_requests=total_requests + error_count,
            total_tokens=total_tokens,
            total_time_seconds=total_time,
            serving_time_seconds=serving_time,
            error_count=error_count,
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


class CacheMetricsTracker:
    """Tracks cache-specific metrics."""
    
    def __init__(self):
        self.local_hits = 0
        self.remote_hits = 0
        self.misses = 0
        self.remote_fetch_latencies: List[float] = []
        self.transfer_bytes: List[int] = []
    
    def record_local_hit(self) -> None:
        """Record a local cache hit."""
        self.local_hits += 1
    
    def record_remote_hit(self, fetch_latency_ms: float, bytes_transferred: int) -> None:
        """Record a remote cache hit."""
        self.remote_hits += 1
        self.remote_fetch_latencies.append(fetch_latency_ms)
        self.transfer_bytes.append(bytes_transferred)
    
    def record_miss(self) -> None:
        """Record a cache miss."""
        self.misses += 1
    
    def get_metrics(self) -> Dict[str, float]:
        """Compute cache metrics."""
        total_requests = self.local_hits + self.remote_hits + self.misses
        
        if total_requests == 0:
            return {
                "local_hit_ratio": 0.0,
                "remote_hit_ratio": 0.0,
                "miss_ratio": 0.0,
                "total_hit_ratio": 0.0,
                "avg_remote_fetch_ms": 0.0,
                "total_transfer_mb": 0.0,
            }
        
        import numpy as np
        
        local_hit_ratio = self.local_hits / total_requests
        remote_hit_ratio = self.remote_hits / total_requests
        miss_ratio = self.misses / total_requests
        total_hit_ratio = (self.local_hits + self.remote_hits) / total_requests
        
        avg_remote_fetch = (
            float(np.mean(self.remote_fetch_latencies))
            if self.remote_fetch_latencies
            else 0.0
        )
        total_transfer_mb = sum(self.transfer_bytes) / 1024 / 1024
        
        return {
            "local_hit_ratio": local_hit_ratio,
            "remote_hit_ratio": remote_hit_ratio,
            "miss_ratio": miss_ratio,
            "total_hit_ratio": total_hit_ratio,
            "avg_remote_fetch_ms": avg_remote_fetch,
            "total_transfer_mb": total_transfer_mb,
        }
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.local_hits = 0
        self.remote_hits = 0
        self.misses = 0
        self.remote_fetch_latencies.clear()
        self.transfer_bytes.clear()


@dataclass
class GPUMetrics:
    """GPU utilization metrics.
    
    Provides nvidia-smi equivalent metrics via pynvml.
    Used for Layer 2 GPU metrics as per paper requirements.
    """
    
    # Per-GPU metrics (aggregated across devices)
    gpu_count: int  # Number of GPUs detected
    
    # Utilization (0-100%)
    avg_gpu_utilization: float  # Average GPU compute utilization
    max_gpu_utilization: float  # Peak GPU utilization
    avg_memory_utilization: float  # Average GPU memory bandwidth utilization
    
    # Memory (MB)
    total_memory_mb: float  # Total GPU memory across all devices
    used_memory_mb: float  # Current GPU memory in use
    peak_memory_mb: float  # Peak GPU memory usage
    memory_usage_percent: float  # Percentage of GPU memory used
    
    # Power (Watts)
    avg_power_watts: float  # Average power draw
    max_power_watts: float  # Peak power draw
    power_limit_watts: float  # Power limit
    
    # Temperature (Celsius)
    avg_temperature_c: float  # Average GPU temperature
    max_temperature_c: float  # Peak GPU temperature
    
    # PCIe (for distributed systems)
    pcie_tx_mb: float  # PCIe TX throughput (MB)
    pcie_rx_mb: float  # PCIe RX throughput (MB)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "gpu_count": self.gpu_count,
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
        self._total_memory: List[float] = []  # Per-GPU total memory (MB)
        self._power_limits: List[float] = []  # Per-GPU power limit (W)
        
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
            
            # Get static device info
            for handle in self._device_handles:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self._total_memory.append(mem_info.total / 1024 / 1024)  # MB
                
                try:
                    power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
                    self._power_limits.append(power_limit / 1000)  # W
                except pynvml.NVMLError:
                    self._power_limits.append(0.0)
            
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
        
        Returns:
            GPUMetrics with aggregated statistics
        """
        import numpy as np
        
        # Return zero metrics if no data
        if not self.gpu_util_samples:
            return GPUMetrics(
                gpu_count=self._device_count,
                avg_gpu_utilization=0.0,
                max_gpu_utilization=0.0,
                avg_memory_utilization=0.0,
                total_memory_mb=sum(self._total_memory) if self._total_memory else 0.0,
                used_memory_mb=0.0,
                peak_memory_mb=0.0,
                memory_usage_percent=0.0,
                avg_power_watts=0.0,
                max_power_watts=0.0,
                power_limit_watts=sum(self._power_limits) if self._power_limits else 0.0,
                avg_temperature_c=0.0,
                max_temperature_c=0.0,
                pcie_tx_mb=0.0,
                pcie_rx_mb=0.0,
            )
        
        # Flatten samples, dropping None (a failed per-call NVML read, not a real zero).
        all_gpu_utils = [u for sample in self.gpu_util_samples for u in sample if u is not None]
        all_mem_utils = [u for sample in self.memory_util_samples for u in sample if u is not None]
        all_mem_used = [m for sample in self.memory_used_samples for m in sample if m is not None]
        all_powers = [p for sample in self.power_samples for p in sample if p is not None]
        all_temps = [t for sample in self.temp_samples for t in sample if t is not None]

        # PCIe throughput (cumulative)
        total_pcie_tx = sum(tx for sample in self.pcie_tx_samples for tx in sample if tx is not None)
        total_pcie_rx = sum(rx for sample in self.pcie_rx_samples for rx in sample if rx is not None)
        
        # Compute statistics
        avg_gpu_util = float(np.mean(all_gpu_utils)) if all_gpu_utils else 0.0
        max_gpu_util = float(np.max(all_gpu_utils)) if all_gpu_utils else 0.0
        avg_mem_util = float(np.mean(all_mem_utils)) if all_mem_utils else 0.0
        
        total_memory = sum(self._total_memory) if self._total_memory else 0.0
        used_memory = float(np.mean(all_mem_used)) if all_mem_used else 0.0
        peak_memory = float(np.max(all_mem_used)) if all_mem_used else 0.0
        memory_percent = (used_memory / total_memory * 100) if total_memory > 0 else 0.0
        
        avg_power = float(np.mean(all_powers)) if all_powers else 0.0
        max_power = float(np.max(all_powers)) if all_powers else 0.0
        power_limit = sum(self._power_limits) if self._power_limits else 0.0
        
        avg_temp = float(np.mean(all_temps)) if all_temps else 0.0
        max_temp = float(np.max(all_temps)) if all_temps else 0.0
        
        pcie_tx_mb = total_pcie_tx / 1024 / 1024  # Convert KB to MB
        pcie_rx_mb = total_pcie_rx / 1024 / 1024
        
        return GPUMetrics(
            gpu_count=self._device_count,
            avg_gpu_utilization=avg_gpu_util,
            max_gpu_utilization=max_gpu_util,
            avg_memory_utilization=avg_mem_util,
            total_memory_mb=total_memory,
            used_memory_mb=used_memory,
            peak_memory_mb=peak_memory,
            memory_usage_percent=memory_percent,
            avg_power_watts=avg_power,
            max_power_watts=max_power,
            power_limit_watts=power_limit,
            avg_temperature_c=avg_temp,
            max_temperature_c=max_temp,
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
