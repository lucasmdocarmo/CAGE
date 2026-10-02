"""Per-window pd transfer proof (S0F-22 Batch 2, ADR-0134).

Batch 1 proved that every served pd request carried the engine's KV ticket:
the prefill OFFERED its blocks. Batch 2 proves the decode PULLED them. vLLM
v0.19.1 counts every pull on the decode's ``/metrics`` (histogram
``vllm:nixl_bytes_transferred`` with ``_sum``/``_count``, counters
``vllm:nixl_num_failed_transfers_total`` and
``vllm:nixl_num_failed_notifications_total``, nixl_connector.py:3030-3164)
and attributes the pulled prompt tokens to
``vllm:prompt_tokens_by_source_total{source="external_kv_transfer"}``
(v1/metrics/loggers.py:600-613). The runner scrapes the decode twice per
window, before the first measured send and after the last completion, and
records the deltas in metrics.json under ``pd_transfer``; a campaign pd
window whose deltas do not prove the pull is refused before any artifact is
written.

Facts the fixtures rest on:
- prometheus_client 0.26.0 renders a labeled histogram as ``_bucket``
  (one line per ``le`` plus ``+Inf``), ``_count``, ``_sum`` and ``_created``
  siblings, and a counter as ``_total`` plus ``_created``; labels are sorted
  alphabetically (rendered in this session with the installed package);
- ``prometheus_client.parser.text_string_to_metric_families`` names each
  sample with its full suffix and keeps the label dict (checked here);
- results rows carry ``error`` and ``prompt_tokens``, never ``ok``
  (run_experiment record_result; the served-row rule is Batch 1's).
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import math
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict

import pytest
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.monitoring import vllm_telemetry as vt  # noqa: E402

RUN_EXPERIMENT_PY = REPO_ROOT / "scripts" / "3_run" / "run_experiment.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load(RUN_EXPERIMENT_PY, "run_experiment_s0f22_b2")

MODEL = "Qwen/Qwen3-8B-FP8"
NIXL_BYTES_BUCKETS = [2 ** (10 + i) for i in range(1, 25, 2)]  # nixl_connector.py:3075


# ---------------------------------------------------------------------------
# Exposition fixtures: the decode's /metrics as prometheus_client renders it
# ---------------------------------------------------------------------------


def _counter(reg: CollectorRegistry, name: str, labels: list, values: Dict[tuple, int]) -> None:
    """Register ``name`` and create every child in ``values`` (vLLM creates
    the children at logger init, so a never-incremented child reads 0.0)."""
    c = Counter(name, "x", labels, registry=reg)
    for key, value in values.items():
        child = c.labels(*key)
        if value:
            child.inc(value)


def _decode_exposition(
    *,
    bytes_obs: list = (),
    failed_transfers: int = 0,
    failed_notifications: int = 0,
    external: int = 0,
    local_compute: int = 0,
    local_cache_hit: int = 0,
    recomputed: int = 0,
    preemptions: int = 0,
    with_nixl: bool = True,
) -> str:
    """One decode scrape. ``bytes_obs`` is the list of per-transfer byte
    observations (one per handle); the by-source counter carries the three
    sources vLLM registers at logger init (stats.py:258-260)."""
    reg = CollectorRegistry()
    if with_nixl:
        h = Histogram(
            "vllm:nixl_bytes_transferred", "bytes", ["model_name", "engine"],
            registry=reg, buckets=NIXL_BYTES_BUCKETS,
        )
        child = h.labels(MODEL, "0")
        for b in bytes_obs:
            child.observe(b)
        _counter(reg, "vllm:nixl_num_failed_transfers", ["model_name", "engine"], {(MODEL, "0"): failed_transfers})
        _counter(reg, "vllm:nixl_num_failed_notifications", ["model_name", "engine"], {(MODEL, "0"): failed_notifications})
    _counter(reg, "vllm:prompt_tokens_by_source", ["model_name", "engine", "source"], {
        (MODEL, "0", "external_kv_transfer"): external,
        (MODEL, "0", "local_compute"): local_compute,
        (MODEL, "0", "local_cache_hit"): local_cache_hit,
    })
    _counter(reg, "vllm:prompt_tokens_recomputed", ["model_name", "engine"], {(MODEL, "0"): recomputed})
    _counter(reg, "vllm:num_preemptions", ["model_name", "engine"], {(MODEL, "0"): preemptions})
    return generate_latest(reg).decode()


def _prefill_exposition(*, expired: int = 0) -> str:
    reg = CollectorRegistry()
    _counter(reg, "vllm:nixl_num_kv_expired_reqs", ["model_name", "engine"], {(MODEL, "0"): expired})
    return generate_latest(reg).decode()


# Window of three served requests, 1 MiB per transfer on a TP=1 decode: 171
# prompt tokens per request, the decode hit one block (16 tokens) locally on
# the second and third request, so external = 171 + 155 + 155 = 481 and
# local_cache_hit = 32. The S0 state holds one earlier (warm-up) transfer.
S0 = dict(bytes_obs=[4096], external=7, local_compute=1, local_cache_hit=0, recomputed=1)
S1 = dict(
    bytes_obs=[4096, 1048576, 1048576, 1048576], external=7 + 481,
    local_compute=1 + 3, local_cache_hit=32, recomputed=1 + 3,
)
SERVED_ROWS = [
    {"example_id": "q1", "error": None, "empty_generation": False, "prompt_tokens": 171},
    {"example_id": "q2", "error": None, "empty_generation": True, "prompt_tokens": 171},
    {"example_id": "q3", "error": None, "empty_generation": False, "prompt_tokens": 171},
    # refused by the proxy: no 2xx, no transfer, not a served row
    {"example_id": "q4", "error": "HTTP 502: prefill returned no usable KV transfer ticket", "prompt_tokens": None},
]


# ---------------------------------------------------------------------------
# read_counter_series: exact names, label filter, siblings excluded
# ---------------------------------------------------------------------------


def test_read_counter_series_reads_sum_count_and_labeled_sources() -> None:
    values = vt.read_counter_series(_decode_exposition(**S1), vt.PD_TRANSFER_SERIES)
    assert values["bytes_sum"] == 4096 + 3 * 1048576
    assert values["transfer_count"] == 4
    assert values["failed_transfers"] == 0 and values["failed_notifications"] == 0
    assert values["external_kv_tokens"] == 488
    assert values["local_compute_tokens"] == 4
    assert values["local_cache_hit_tokens"] == 32
    assert values["recomputed_tokens"] == 4
    assert values["preemptions"] == 0


def test_read_counter_series_ignores_bucket_and_created_siblings() -> None:
    text = _decode_exposition(**S1)
    assert "_created{" in text and 'le="+Inf"' in text  # the siblings are present
    values = vt.read_counter_series(text, vt.PD_TRANSFER_SERIES)
    # _count is 4 observations, never the bucket sum (12 buckets x cumulative counts)
    assert values["transfer_count"] == 4
    # _sum is the exact byte total, never a _created epoch (about 1.79e9 today)
    assert values["bytes_sum"] == 4096 + 3 * 1048576


def test_read_counter_series_absent_series_is_none_never_zero() -> None:
    values = vt.read_counter_series(_decode_exposition(with_nixl=False, **S0), vt.PD_TRANSFER_SERIES)
    assert values["bytes_sum"] is None and values["transfer_count"] is None
    assert values["failed_transfers"] is None
    assert values["external_kv_tokens"] == 7  # the by-source family is still there
    assert vt.read_counter_series("", vt.PD_TRANSFER_SERIES) == {k: None for k in vt.PD_TRANSFER_SERIES}


def test_read_counter_series_sums_label_sets_and_rejects_non_finite() -> None:
    text = (
        'vllm:nixl_num_failed_transfers_total{engine="0",model_name="a"} 1.0\n'
        'vllm:nixl_num_failed_transfers_total{engine="1",model_name="a"} 2.0\n'
        'vllm:nixl_bytes_transferred_sum{engine="0",model_name="a"} NaN\n'
        'vllm:nixl_bytes_transferred_count{engine="0",model_name="a"} +Inf\n'
        'vllm:prompt_tokens_by_source_total{engine="0",model_name="a}b,c",source="external_kv_transfer"} 9 1790000000\n'
    )
    values = vt.read_counter_series(text, vt.PD_TRANSFER_SERIES)
    assert values["failed_transfers"] == 3  # two engines summed (DP): the lifetime total
    assert values["bytes_sum"] is None and values["transfer_count"] is None  # non-finite is absent
    assert values["external_kv_tokens"] == 9  # label with '}' and ','; trailing timestamp dropped


def test_prefill_series_reads_the_expiry_counter() -> None:
    assert vt.read_counter_series(_prefill_exposition(expired=2), vt.PD_PREFILL_SERIES) == {"kv_expired_reqs": 2}


def test_series_names_are_the_v0191_families() -> None:
    names = {name for name, _ in vt.PD_TRANSFER_SERIES.values()}
    assert names == {
        "vllm:nixl_bytes_transferred_sum", "vllm:nixl_bytes_transferred_count",
        "vllm:nixl_num_failed_transfers_total", "vllm:nixl_num_failed_notifications_total",
        "vllm:prompt_tokens_by_source_total", "vllm:prompt_tokens_recomputed_total",
        "vllm:num_preemptions_total",
    }
    assert vt.PD_TRANSFER_SERIES["external_kv_tokens"] == (
        "vllm:prompt_tokens_by_source_total", {"source": "external_kv_transfer"})
    assert vt.PD_PREFILL_SERIES == {"kv_expired_reqs": ("vllm:nixl_num_kv_expired_reqs_total", {})}


# ---------------------------------------------------------------------------
# scrape_counter_series over HTTP (stdlib server stub)
# ---------------------------------------------------------------------------


class _MetricsStub(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _MetricsHandler)
        self.body = ""
        self.status = 200
        self.hits = 0
        self.fail_first = 0  # answer 503 to this many GETs, then self.status

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


class _MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        srv: _MetricsStub = self.server  # type: ignore[assignment]
        srv.hits += 1
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return
        payload = srv.body.encode()
        status = srv.status
        if srv.hits <= srv.fail_first:
            status = 503
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # silence
        return


@pytest.fixture
def metrics_stub():
    srv = _MetricsStub()
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()


def test_scrape_counter_series_two_scrapes_give_the_window_delta(metrics_stub) -> None:
    metrics_stub.body = _decode_exposition(**S0)
    start = vt.scrape_counter_series(metrics_stub.url, vt.PD_TRANSFER_SERIES)
    metrics_stub.body = _decode_exposition(**S1)
    end = vt.scrape_counter_series(metrics_stub.url, vt.PD_TRANSFER_SERIES)
    assert metrics_stub.hits == 2
    assert end["bytes_sum"] - start["bytes_sum"] == 3 * 1048576
    assert end["transfer_count"] - start["transfer_count"] == 3
    # a base url that already ends in /metrics is not doubled
    assert vt.scrape_counter_series(metrics_stub.url + "/metrics", vt.PD_PREFILL_SERIES) == {"kv_expired_reqs": None}


def test_scrape_counter_series_raises_on_transport_and_http_failure(metrics_stub) -> None:
    # urllib raises HTTPError (a URLError, an OSError) on 503 and URLError on
    # a refused connection: the caller decides the policy, the scraper never
    # swallows and never fabricates a zero
    metrics_stub.status = 503
    with pytest.raises(OSError):
        vt.scrape_counter_series(metrics_stub.url, vt.PD_TRANSFER_SERIES, timeout=2.0)
    with pytest.raises(OSError):
        vt.scrape_counter_series("http://127.0.0.1:9", vt.PD_TRANSFER_SERIES, timeout=0.5)


# ---------------------------------------------------------------------------
# pd_transfer_reasons: the ONE rule the runner gates on and verify re-derives
# ---------------------------------------------------------------------------


def _delta(**over: Any) -> Dict[str, Any]:
    base = {
        "bytes_sum": 3 * 1048576, "transfer_count": 3, "failed_transfers": 0,
        "failed_notifications": 0, "external_kv_tokens": 481, "local_compute_tokens": 3,
        "local_cache_hit_tokens": 32, "recomputed_tokens": 3, "preemptions": 0,
    }
    base.update(over)
    return base


def test_gate_rule_passes_the_healthy_window() -> None:
    assert vt.pd_transfer_reasons(_delta(), 3) == []
    # TP=4 decode: four observations per request, still at least one per served row
    assert vt.pd_transfer_reasons(_delta(transfer_count=12), 3) == []


@pytest.mark.parametrize("over,needle", [
    ({"failed_transfers": 1}, "failed_transfers == 0"),
    ({"failed_notifications": 2}, "failed_notifications == 0"),
    ({"bytes_sum": 0}, "bytes_sum > 0"),
    ({"transfer_count": 2}, "transfer_count >= n_served_rows"),
    ({"external_kv_tokens": 0}, "external_kv_tokens > 0"),
    ({"bytes_sum": None}, "series absent"),
    ({"transfer_count": -3}, "negative"),
    ({"bytes_sum": float("nan")}, "not a finite"),
    ({"bytes_sum": True}, "not a finite"),
], ids=["failed-xfer", "failed-notif", "no-bytes", "fewer-transfers-than-rows",
        "no-external-tokens", "absent", "counter-reset", "nan", "bool"])
def test_gate_rule_names_each_failing_clause(over: Dict[str, Any], needle: str) -> None:
    reasons = vt.pd_transfer_reasons(_delta(**over), 3)
    assert reasons and any(needle in r for r in reasons), reasons


def test_gate_rule_refuses_a_bad_count_but_not_zero_served_rows() -> None:
    assert any("n_served_rows" in r for r in vt.pd_transfer_reasons(_delta(), "3"))
    assert any("n_served_rows" in r for r in vt.pd_transfer_reasons(_delta(), True))
    assert any("n_served_rows" in r for r in vt.pd_transfer_reasons(_delta(), -1))
    # Review LOW-1: an attainment-0 window at the cliff (every request timed
    # out AFTER its pull) is a data point, not a broken pair; the gate asks
    # whether the pair transferred, and bytes_sum > 0 still refuses a window
    # where nothing moved.
    assert vt.pd_transfer_reasons(_delta(), 0) == []
    assert any("bytes_sum > 0" in r for r in vt.pd_transfer_reasons(_delta(bytes_sum=0, external_kv_tokens=0), 0))


def test_identities_are_recorded_never_gated() -> None:
    # A preemption changes local_compute (stats.py:280-297): the identities
    # break, the gate does not. Pressure windows preempt by design.
    assert vt.pd_transfer_reasons(_delta(preemptions=2, local_compute_tokens=5), 3) == []
    assert "preemptions" not in " ".join(vt.PD_TRANSFER_GATE_CLAUSES)
    assert set(vt.PD_TRANSFER_GATE_CLAUSES) == {
        "failed_transfers == 0", "failed_notifications == 0", "bytes_sum > 0",
        "transfer_count >= n_served_rows", "external_kv_tokens > 0",
    }


# ---------------------------------------------------------------------------
# pd_transfer_record: the metrics.json block
# ---------------------------------------------------------------------------


def _record(**over: Any) -> Dict[str, Any]:
    kw = dict(
        start=vt.read_counter_series(_decode_exposition(**S0), vt.PD_TRANSFER_SERIES),
        end=vt.read_counter_series(_decode_exposition(**S1), vt.PD_TRANSFER_SERIES),
        prefill_start=vt.read_counter_series(_prefill_exposition(), vt.PD_PREFILL_SERIES),
        prefill_end=vt.read_counter_series(_prefill_exposition(expired=1), vt.PD_PREFILL_SERIES),
        n_served_rows=3, prompt_tokens_sum=513,
        decode_url="http://localhost:8200", prefill_url="http://localhost:8100",
        scrape_start_ts=1000.0, scrape_end_ts=1060.5,
    )
    kw.update(over)
    return vt.pd_transfer_record(**kw)


def test_record_carries_deltas_identities_and_the_verdict() -> None:
    rec = _record()
    assert rec["adr"] == "ADR-0134" and rec["verified"] is True and rec["reasons"] == []
    assert rec["delta"]["bytes_sum"] == 3 * 1048576 and rec["delta"]["transfer_count"] == 3
    assert rec["delta"]["external_kv_tokens"] == 481 and rec["delta"]["local_cache_hit_tokens"] == 32
    assert rec["prefill_delta"] == {"kv_expired_reqs": 1}
    assert rec["n_served_rows"] == 3 and rec["prompt_tokens_sum"] == 513
    assert rec["transfers_per_served_row"] == 1.0
    # identities (recorded for RC-13, decision 4b): 481 + 32 == 513, 3 == 3, 3 == 3, 0 preemptions
    assert rec["identities"] == {
        "local_compute_equals_served": True, "recomputed_equals_served": True,
        "external_plus_cache_hit_equals_prompt_tokens": True, "no_preemptions": True,
    }
    assert rec["identities_hold"] is True
    assert rec["gate_clauses"] == list(vt.PD_TRANSFER_GATE_CLAUSES)
    assert rec["start"]["transfer_count"] == 1 and rec["end"]["transfer_count"] == 4
    assert rec["scrape_start_ts"] == 1000.0 and rec["scrape_end_ts"] == 1060.5
    assert rec["decode_url"] == "http://localhost:8200" and rec["prefill_url"] == "http://localhost:8100"


def test_record_is_native_json_with_string_keys_and_no_nan() -> None:
    rec = _record()
    text = json.dumps(rec, allow_nan=False)  # raises on NaN/Inf or a non-native type
    assert json.loads(text) == rec

    def _walk(v: Any) -> None:
        if isinstance(v, dict):
            for k, x in v.items():
                assert isinstance(k, str)
                _walk(x)
        elif isinstance(v, list):
            for x in v:
                _walk(x)
        else:
            assert v is None or isinstance(v, (bool, int, float, str))
            if isinstance(v, float):
                assert math.isfinite(v)
    _walk(rec)


def test_record_refuses_when_a_clause_fails_and_keeps_the_numbers() -> None:
    failed = vt.read_counter_series(_decode_exposition(**{**S1, "failed_transfers": 1}), vt.PD_TRANSFER_SERIES)
    rec = _record(end=failed)
    assert rec["verified"] is False
    assert any("failed_transfers == 0" in r for r in rec["reasons"])
    assert rec["delta"]["failed_transfers"] == 1 and rec["delta"]["bytes_sum"] == 3 * 1048576


def test_record_without_prefill_or_prompt_tokens_leaves_holes_not_zeros() -> None:
    rec = _record(prefill_start=None, prefill_end=None, prompt_tokens_sum=None, prefill_url=None)
    assert rec["prefill_delta"] is None and rec["prefill_url"] is None
    assert rec["prompt_tokens_sum"] is None
    assert rec["identities"]["external_plus_cache_hit_equals_prompt_tokens"] is None
    assert rec["identities_hold"] is None  # unknown, not False
    assert rec["verified"] is True  # the gate never reads the identities


def test_record_marks_a_counter_reset_between_scrapes() -> None:
    # the decode was relaunched mid-window: lifetime counters restart at zero
    rec = _record(end=vt.read_counter_series(_decode_exposition(**S0), vt.PD_TRANSFER_SERIES),
                  start=vt.read_counter_series(_decode_exposition(**S1), vt.PD_TRANSFER_SERIES))
    assert rec["verified"] is False
    assert any("negative" in r for r in rec["reasons"])


# ---------------------------------------------------------------------------
# Runner side: endpoints, scrape policy, record assembly, strict gate, wiring
# ---------------------------------------------------------------------------


def test_pd_transfer_endpoints_need_a_decode_role() -> None:
    assert runner.pd_transfer_endpoints(None) is None
    assert runner.pd_transfer_endpoints([("single", "http://localhost:8000")]) is None
    assert runner.pd_transfer_endpoints(
        [("prefill", "http://localhost:8100"), ("decode", "http://localhost:8200")]
    ) == ("http://localhost:8200", "http://localhost:8100")
    assert runner.pd_transfer_endpoints([("decode", "http://h:8200")]) == ("http://h:8200", None)


def test_scrape_pd_transfer_strict_raises_and_pilot_records_the_failure(metrics_stub) -> None:
    urls = ("http://127.0.0.1:9", None)  # nothing listens on port 9
    with pytest.raises(RuntimeError, match="CAMPAIGN PD TRANSFER.*ADR-0134"):
        runner.scrape_pd_transfer(urls, strict=True, stage="before the measured stage",
                                  timeout=0.5, retry_pause_s=0.0)
    pilot = runner.scrape_pd_transfer(urls, strict=False, stage="before the measured stage",
                                      timeout=0.5, retry_pause_s=0.0)
    assert "error" in pilot and "decode" not in pilot and isinstance(pilot["ts"], float)
    # success: the decode values plus the optional prefill values
    metrics_stub.body = _decode_exposition(**S0)
    ok = runner.scrape_pd_transfer((metrics_stub.url, None), strict=True, stage="x")
    assert ok["decode"]["transfer_count"] == 1 and ok["prefill"] is None and isinstance(ok["ts"], float)


def test_scrape_pd_transfer_retries_a_transient_decode_failure(metrics_stub) -> None:
    # Review MEDIUM-1: one late /metrics answer right after a cliff window
    # must not discard the window; the counters are monotonic and the server
    # is quiescent at S1, so a later GET reads the same values.
    metrics_stub.body = _decode_exposition(**S1)
    metrics_stub.fail_first = 2  # two 503s, then 200
    out = runner.scrape_pd_transfer((metrics_stub.url, None), strict=True, stage="after the measured stage",
                                    timeout=2.0, retry_pause_s=0.01)
    assert out["decode"]["transfer_count"] == 4 and metrics_stub.hits == 3
    # the bound holds: three attempts, then the strict raise
    metrics_stub.hits = 0
    metrics_stub.fail_first = 3
    with pytest.raises(RuntimeError, match="CAMPAIGN PD TRANSFER.*3 attempt"):
        runner.scrape_pd_transfer((metrics_stub.url, None), strict=True, stage="x",
                                  timeout=2.0, retry_pause_s=0.01)
    assert metrics_stub.hits == 3
    assert runner.PD_SCRAPE_ATTEMPTS == 3 and runner.PD_SCRAPE_RETRY_PAUSE_S == 5.0


def test_scrape_pd_transfer_prefill_failure_is_recorded_not_fatal(metrics_stub) -> None:
    # the expiry counter is informational: a dead prefill never decides the proof
    metrics_stub.body = _decode_exposition(**S0)
    out = runner.scrape_pd_transfer((metrics_stub.url, "http://127.0.0.1:9"), strict=True, stage="x",
                                    timeout=0.5, retry_pause_s=0.0)
    assert out["decode"]["transfer_count"] == 1 and out["prefill"] is None


def test_build_pd_transfer_record_counts_served_rows_by_the_error_rule() -> None:
    start = {"ts": 1.0, "decode": vt.read_counter_series(_decode_exposition(**S0), vt.PD_TRANSFER_SERIES), "prefill": None}
    end = {"ts": 2.0, "decode": vt.read_counter_series(_decode_exposition(**S1), vt.PD_TRANSFER_SERIES), "prefill": None}
    rec = runner.build_pd_transfer_record(start, end, SERVED_ROWS, ("http://d:8200", None))
    assert rec["n_served_rows"] == 3  # q4 (error) excluded, q2 (empty generation, 2xx) included
    assert rec["prompt_tokens_sum"] == 513 and rec["verified"] is True
    assert rec["scrape_start_ts"] == 1.0 and rec["scrape_end_ts"] == 2.0
    # a served row without an integer prompt_tokens makes the sum unknown, never 0
    rows = [dict(r) for r in SERVED_ROWS]
    rows[0]["prompt_tokens"] = None
    assert runner.build_pd_transfer_record(start, end, rows, ("http://d:8200", None))["prompt_tokens_sum"] is None
    # a failed scrape yields an unverified record that names the failure
    bad = runner.build_pd_transfer_record({"ts": 1.0, "error": "boom"}, end, SERVED_ROWS, ("http://d:8200", None))
    assert bad["verified"] is False and "boom" in " ".join(bad["reasons"]) and bad["n_served_rows"] == 3


def test_enforce_pd_transfer_record_refuses_with_the_reasons() -> None:
    runner.enforce_pd_transfer_record(_record())  # verified: no raise
    failed = vt.read_counter_series(_decode_exposition(**{**S1, "failed_transfers": 1}), vt.PD_TRANSFER_SERIES)
    with pytest.raises(RuntimeError, match="CAMPAIGN PD TRANSFER.*failed_transfers == 0.*ADR-0134"):
        runner.enforce_pd_transfer_record(_record(end=failed))
    with pytest.raises(RuntimeError, match="CAMPAIGN PD TRANSFER"):
        runner.enforce_pd_transfer_record({"verified": False, "reasons": ["scrape failed"]})


def test_runner_wiring_scrapes_at_the_stage_boundaries_and_gates_before_artifacts() -> None:
    src = inspect.getsource(runner.run_experiment)
    # the decode endpoint is resolved, and a campaign pd cell without it
    # refuses, before the engine is set up (no GPU minute burned)
    urls = src.index("pd_transfer_urls = pd_transfer_endpoints(telemetry_endpoints)")
    assert src.index("telemetry_endpoints = resolve_telemetry_endpoints(") < urls
    assert urls < src.index("CAMPAIGN PD TRANSFER") < src.index("setup_inference_engine(")
    # S0 after the warm-up stages (synchronous) and before the stage bracket
    s0 = src.index("pd_transfer_start = ")
    assert src.index('stage_name="Warmup"') < s0 < src.index("stage_t_start = time.time()")
    # S1 right after the stage bracket, before the best-effort telemetry block
    s1 = src.index("pd_transfer_end = ")
    assert src.index("stage_t_end = time.time()") < s1 < src.index("vllm_telemetry_snapshot = None")
    # Review MEDIUM-1: a strict S1 refusal flushes the window's rows first,
    # through the same helper the stage crash path uses (forensics for a
    # window that cost its full GPU time)
    flush = src.index("def _flush_partial_rows(")
    assert flush < src.index("stage_t_start = time.time()")
    calls = src.count("_flush_partial_rows()") - src.count("def _flush_partial_rows()")
    assert calls == 2, "the stage crash path and the strict S1 refusal"
    assert s1 < src.index("_flush_partial_rows()", s1) < src.index("vllm_telemetry_snapshot = None")
    # the strict gate after the ticket gate (same guard), before any artifact
    gate = src.index("enforce_pd_transfer_record(")
    assert src.index("enforce_pd_transfer_tickets(results)") < gate < src.index("results_file = output_path /")
    assert "pd_strict" in src[gate - 200:gate]
    # the record rides metrics.json beside cold_start, before both writes
    key = src.index('experiment_summary["pd_transfer"]')
    assert src.index('experiment_summary["cold_start"]') < key < src.index("write_json_atomic(metrics_file, experiment_summary)")
    assert runner.PD_TRANSFER_KEY == "pd_transfer"


def test_strict_predicate_is_the_batch1_guard() -> None:
    src = inspect.getsource(runner.run_experiment)
    assert 'pd_strict = campaign_session is not None and campaign_session.spec.topology == "pd"' in src
