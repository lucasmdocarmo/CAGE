"""ADR-0148 (2026-10-07): None where nothing was measured, on the runner side.

Pinned here, on the runner (loaded by path, the tests/test_wave1 pattern) and
on the performance module:

1. ``adapter_honesty_columns`` carries the HF oracle's ``corpus_prefill_ms``
   (the adapter stamps it on every reused response, hf_oracle_adapter.py) and
   None when the response has no such attribute.
2. The ``cache_telemetry`` block and its ``CacheMetricsTracker`` are gone: the
   runner no longer coerces an absent cached-token count to 0 and records a
   miss for it (an SGLang or LMDeploy window read miss_ratio 1.0 as if
   measured); the per-row ``cached_prompt_tokens`` and the ``prompt_cache``
   summary are the reuse record.
3. A campaign window whose sampler captured no tick records NO vllm_telemetry:
   the idle one-shot ``capture(api_base)`` stays on the pilot path, inside the
   T4.1 guard the multi-instance tests pin.
4. The trial aggregator averages performance values over non-None entries and
   aggregates the ``gpu`` block.
5. The run-log summary formats None fields as ``n/a`` instead of raising.
6. An all-error window and an unsampled GPU tracker produce dicts whose
   reading-derived fields are None and whose counts are 0: no 0.0 stands in
   the place of a measurement.

No GPU, no engine, no network.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest

import src.evaluation.performance as perf
from src.evaluation.performance import GPUMetricsTracker, PerformanceEvaluator

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts" / "3_run" / "run_experiment.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("cage_run_experiment_adr0148", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()
RUNNER_SRC = RUNNER_PATH.read_text(encoding="utf-8")


class _Stub:
    pass


# 1. the oracle's prefill time reaches the row --------------------------------


def test_adapter_honesty_columns_carry_the_oracle_prefill_time():
    resp = _Stub()
    resp.corpus_prefill_ms = 812.5
    assert runner.adapter_honesty_columns(resp)["corpus_prefill_ms"] == 812.5


def test_adapter_honesty_columns_leave_the_prefill_none_when_unstamped():
    assert runner.adapter_honesty_columns(_Stub())["corpus_prefill_ms"] is None


# 2. the cache_telemetry block left -------------------------------------------


def test_cache_telemetry_block_and_tracker_are_gone():
    assert not hasattr(perf, "CacheMetricsTracker")
    assert "CacheMetricsTracker" not in RUNNER_SRC
    assert "cache_tracker" not in RUNNER_SRC
    assert '"cache_telemetry"' not in RUNNER_SRC
    # the fabricated-miss coercion (an absent count read as 0) is gone
    assert "cached_prompt_tokens or 0" not in RUNNER_SRC
    # the honest per-row record and its summary stay
    assert '"cached_prompt_tokens": response.cached_prompt_tokens' in RUNNER_SRC
    assert '"prompt_cache": cache_summary' in RUNNER_SRC


# 3. no idle capture on a campaign window -------------------------------------


def test_campaign_window_refuses_the_idle_capture_inside_the_pinned_guard():
    src_text = inspect.getsource(runner.run_experiment)
    guard = "if vllm_telemetry_snapshot is None and vllm_role_samplers is None:"
    refusal = "an idle one-shot capture would read as measured"
    call = "vllm_telemetry_snapshot, _ = capture(api_base)"
    assert src_text.count(call) == 1
    assert src_text.count(refusal) == 1
    g, r, c = src_text.index(guard), src_text.index(refusal), src_text.index(call)
    assert g < r < c
    # the refusal keys on the campaign session, the capture on its absence
    between = src_text[g:c]
    assert "if campaign_session is not None:" in between
    assert "ADR-0148" in between


# 4. the trial aggregator -----------------------------------------------------


def test_trial_aggregation_filters_none_and_aggregates_gpu():
    src_text = inspect.getsource(runner.main)
    assert 'result["performance"][key] is not None' in src_text
    assert 'aggregate_numeric_section("gpu")' in src_text
    assert 'aggregate_numeric_section("cache_telemetry")' not in src_text


# 5. the summary prints -------------------------------------------------------


def test_summary_prints_are_none_safe():
    src_text = inspect.getsource(runner.run_experiment)
    assert "perf_metrics.queries_per_second:.2f" not in src_text
    assert "perf_metrics.avg_cpu_percent:.1f" not in src_text
    assert "format_metric(perf_metrics.avg_ttft_ms)" in src_text
    assert runner.format_metric(None) == "n/a"


# 6. the dicts an all-error window writes -------------------------------------


def test_all_error_window_dict_has_no_fabricated_zero():
    ev = PerformanceEvaluator(monitor_resources=False)
    ev.start()
    ev.record_request("e1", 0.0, 0.0, 0, error="HTTP 500")
    ev.stop()
    d = json.loads(json.dumps(ev.compute_metrics().to_dict()))
    assert d["total_requests"] == 1 and d["error_count"] == 1
    assert d["tpot_sample_count"] == 0 and d["resource_sample_count"] == 0
    for key in ("queries_per_second", "tokens_per_second", "avg_ttft_ms", "p95_ttft_ms",
                "avg_tpot_ms", "avg_latency_ms", "p99_latency_ms", "avg_cpu_percent",
                "avg_memory_mb", "peak_memory_mb"):
        assert d[key] is None, key


def test_unsampled_gpu_tracker_dict_has_no_fabricated_zero(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)  # import pynvml raises ImportError
    d = json.loads(json.dumps(GPUMetricsTracker().compute_metrics().to_dict()))
    assert d["gpu_count"] == 0 and d["sample_count"] == 0
    for key, value in d.items():
        if key in ("gpu_count", "sample_count"):
            continue
        assert value is None, (key, value)
