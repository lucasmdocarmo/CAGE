"""Offline contract tests for scripts/3_run/calibrate_cell.py (K-COV6, #142).

The calibration driver's glue was 0/105: the report-writer contract to its
consumers (the campaign driver's ``build_rate_grid_schedules(lambda_star_qps)``
+ ``goodput.SLOBaseline``) had never been exercised from the REAL writer. These
tests drive the driver's pure glue offline — stub streaming adapters, no GPU,
no serving, no network:

- build_requests: manifest blocks -> InferenceRequest shape + refusal arms
- build_adapter: backend routing + unknown-backend refusal
- measure_floor: streamed TTFT/TPOT derivation + fail-closed arms (error
  response, num_tokens < 2)
- main(): the calibration JSON written by the REAL writer parses back into
  the §6.1 schema, feeds both real consumers, and the exit code is 0 only on
  a bracketed (ESTIMATED) lambda* — an unbracketed label writes the JSON but
  exits 1 (never seeds a rate grid)

The probe ladder itself (probe_rate / run_probe_ladder) drives PROBE_WINDOW_S
= 75 s real-time open-loop windows through OpenLoopDispatcher — that is the
[VERIFY-LIVE at S0] surface and stays out of the offline layer; main() is
tested with the floor/ladder stages monkeypatched at their module seams.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_DRIVER_PATH = REPO_ROOT / "scripts" / "3_run" / "calibrate_cell.py"


def _load_driver():
    spec = importlib.util.spec_from_file_location("calibrate_cell", _DRIVER_PATH)
    module = importlib.util.module_from_spec(spec)
    # Register BEFORE exec (dataclass decorators resolve cls.__module__).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cc = _load_driver()

from src.analysis.goodput import SLOBaseline  # noqa: E402
from src.inference.engine import InferenceRequest, InferenceResponse  # noqa: E402
from src.orchestration.calibration import (  # noqa: E402
    FLOOR_N_REQUESTS,
    CalibrationError,
    FloorMeasurement,
    LambdaStarEstimate,
    PROBE_ATTAINMENT_MIN,
    PROBE_LADDER_FACTOR,
    PROBE_MAX_STEPS,
    PROBE_WARMUP_S,
    PROBE_WINDOW_S,
    ProbeStep,
)
from src.orchestration.load_generator import build_rate_grid_schedules  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures: manifest + stub streaming adapter
# --------------------------------------------------------------------------- #


def _write_manifest(path: Path, blocks) -> Path:
    path.write_text(json.dumps({"blocks": blocks}), encoding="utf-8")
    return path


def _green_manifest(tmp_path: Path, n_blocks: int = 3) -> Path:
    return _write_manifest(
        tmp_path / "manifest.json",
        [{"block_id": f"b{i}", "text": f"Corpus block {i} text."} for i in range(n_blocks)],
    )


def _response(*, ttft_ms=100.0, total_time_ms=600.0, num_tokens=6, error=None):
    return InferenceResponse(
        request_id="cal", generated_text="x " * num_tokens, ttft_ms=ttft_ms,
        total_time_ms=total_time_ms, num_tokens=num_tokens,
        model_name="stub-model", finish_reason="stop", error=error,
    )


class StubAdapter:
    """Deterministic offline stand-in for the streaming engine adapters."""

    engine_id = "stub-engine"

    def __init__(self, responses=None):
        self._responses = list(responses or [])
        self.calls = 0

    async def async_stream_generate(self, request, on_first_token=None):
        response = (
            self._responses[self.calls % len(self._responses)]
            if self._responses
            else _response()
        )
        self.calls += 1
        return response


# --------------------------------------------------------------------------- #
# build_requests: manifest -> InferenceRequest contract
# --------------------------------------------------------------------------- #


class TestBuildRequests:
    def test_green_manifest_builds_calibration_requests(self, tmp_path: Path):
        manifest = _green_manifest(tmp_path)
        requests = cc.build_requests(str(manifest))
        assert len(requests) == 3
        for i, req in enumerate(requests):
            assert isinstance(req, InferenceRequest)
            # The block TEXT verbatim is the prompt (same length profile the
            # cell will serve) with the registered fixed decode cap.
            assert req.prompt == f"Corpus block {i} text."
            assert req.max_tokens == cc.CAL_MAX_TOKENS == 256
            assert req.temperature == 0.0
            assert req.request_id == f"cal-block-b{i}"

    def test_missing_manifest_refused(self, tmp_path: Path):
        with pytest.raises(CalibrationError, match="file not found"):
            cc.build_requests(str(tmp_path / "nope.json"))

    def test_manifest_without_blocks_refused(self, tmp_path: Path):
        path = tmp_path / "empty.json"
        path.write_text(json.dumps({"blocks": []}), encoding="utf-8")
        with pytest.raises(CalibrationError, match="no 'blocks'"):
            cc.build_requests(str(path))

    def test_block_without_text_refused(self, tmp_path: Path):
        manifest = _write_manifest(
            tmp_path / "bad.json",
            [{"block_id": "b0", "text": "ok"}, {"block_id": "b1", "text": "   "}],
        )
        with pytest.raises(CalibrationError, match="without verbatim 'text'"):
            cc.build_requests(str(manifest))


# --------------------------------------------------------------------------- #
# build_adapter: backend routing
# --------------------------------------------------------------------------- #


class TestBuildAdapter:
    def test_unknown_backend_refused_with_roster(self):
        with pytest.raises(CalibrationError, match="vllm.*sglang.*lmdeploy"):
            cc.build_adapter("triton", "m", "http://localhost:8000")

    def test_vllm_backend_constructs_streaming_adapter(self):
        adapter = cc.build_adapter("vllm", "some/model", "http://localhost:8000")
        assert adapter.engine_id == "vllm"
        assert hasattr(adapter, "async_stream_generate")


# --------------------------------------------------------------------------- #
# measure_floor: streamed TTFT/TPOT derivation, fail-closed
# --------------------------------------------------------------------------- #


class TestMeasureFloor:
    def test_floor_medians_from_streamed_telemetry(self, tmp_path: Path):
        # ttft 100 ms; tpot = (600-100)/(6-1) = 100 ms -> 0.1 s each.
        adapter = StubAdapter()
        requests = cc.build_requests(str(_green_manifest(tmp_path)))
        floor = asyncio.run(cc.measure_floor(adapter, requests))
        assert isinstance(floor, FloorMeasurement)
        assert adapter.calls == FLOOR_N_REQUESTS  # registered n, modulo-wrapped
        assert floor.n_requests == FLOOR_N_REQUESTS
        assert floor.ttft_s == pytest.approx(0.1)
        assert floor.tpot_s == pytest.approx(0.1)
        assert floor.statistic == "median"
        # The floor pair feeds goodput.SLOBaseline UNCHANGED.
        baseline = SLOBaseline(ttft_s=floor.ttft_s, tpot_s=floor.tpot_s)
        assert baseline.ttft_s == floor.ttft_s

    def test_error_response_fails_closed(self, tmp_path: Path):
        adapter = StubAdapter([_response(error="engine 500")])
        requests = cc.build_requests(str(_green_manifest(tmp_path)))
        with pytest.raises(CalibrationError, match="drop-nothing"):
            asyncio.run(cc.measure_floor(adapter, requests))

    def test_single_token_response_fails_closed(self, tmp_path: Path):
        adapter = StubAdapter([_response(num_tokens=1)])
        requests = cc.build_requests(str(_green_manifest(tmp_path)))
        with pytest.raises(CalibrationError, match=">= 2 generated tokens"):
            asyncio.run(cc.measure_floor(adapter, requests))


# --------------------------------------------------------------------------- #
# main(): the calibration JSON contract, written by the REAL writer
# --------------------------------------------------------------------------- #


def _floor() -> FloorMeasurement:
    return FloorMeasurement(ttft_s=0.1, tpot_s=0.02, n_requests=FLOOR_N_REQUESTS)


def _estimate(label: str) -> LambdaStarEstimate:
    steps = (
        ProbeStep(rate_qps=1.0, n_scheduled=60, n_completed=60, throughput_rps=1.0),
        ProbeStep(rate_qps=1.3, n_scheduled=78, n_completed=39, throughput_rps=0.6),
    )
    if label == "ESTIMATED":
        return LambdaStarEstimate(
            label="ESTIMATED", lambda_star_qps=1.0, sustained_rate_qps=1.0,
            first_unsustainable_qps=1.3, steps=steps,
        )
    if label == "NONE_SUSTAINABLE":
        return LambdaStarEstimate(
            label="NONE_SUSTAINABLE", lambda_star_qps=None,
            sustained_rate_qps=None, first_unsustainable_qps=1.0, steps=steps,
        )
    return LambdaStarEstimate(
        label="LADDER_EXHAUSTED", lambda_star_qps=None, sustained_rate_qps=1.3,
        first_unsustainable_qps=None, steps=steps,
    )


def _run_main(tmp_path: Path, monkeypatch, *, label: str) -> tuple[int, Path]:
    """Drive main() with the live stages stubbed at their module seams."""
    manifest = _green_manifest(tmp_path)
    out_path = tmp_path / "out" / "calibration.json"

    monkeypatch.setattr(cc, "build_adapter", lambda *a, **k: StubAdapter())

    async def fake_measure_floor(adapter, requests):
        return _floor()

    async def fake_run_probe_ladder(adapter, requests, start_qps):
        return _estimate(label)

    monkeypatch.setattr(cc, "measure_floor", fake_measure_floor)
    monkeypatch.setattr(cc, "run_probe_ladder", fake_run_probe_ladder)
    rc = cc.main([
        "--backend", "vllm", "--model", "Qwen/Qwen3-8B",
        "--api-base", "http://localhost:8000",
        "--manifest", str(manifest),
        "--output", str(out_path),
        "--start-qps", "0.5", "--budget-fraction", "0.5",
    ])
    return rc, out_path


class TestMainReportContract:
    def test_estimated_writes_schema_and_exits_zero(self, tmp_path, monkeypatch):
        rc, out_path = _run_main(tmp_path, monkeypatch, label="ESTIMATED")
        assert rc == 0
        doc = json.loads(out_path.read_text(encoding="utf-8"))
        # §6.1 schema the campaign driver + run manifest consume.
        assert doc["model"] == "Qwen/Qwen3-8B"
        assert doc["engine"] == "stub-engine"  # adapter.engine_id wins
        assert doc["budget_fraction"] == 0.5
        assert doc["confirmatory"] is False  # NEVER enters confirmatory analysis
        assert doc["procedure"] == {
            "floor_n_requests": FLOOR_N_REQUESTS,
            "floor_statistic": "median",
            "probe_ladder_factor": PROBE_LADDER_FACTOR,
            "probe_window_s": PROBE_WINDOW_S,
            "probe_warmup_s": PROBE_WARMUP_S,
            "probe_attainment_min": PROBE_ATTAINMENT_MIN,
            "probe_max_steps": PROBE_MAX_STEPS,
            "probe_bisect_steps": 2,  # ADR-0122
            "start_qps_rule": "floor-service-rate",  # ADR-0122
        }
        # ADR-0122: --start-qps was given, so the artifact records it as an
        # operator prior (the floor rule applies only when it is absent).
        assert doc["start_qps"] == 0.5
        assert doc["start_qps_source"] == "operator"
        assert doc["floor"] == {
            "ttft_s": 0.1, "tpot_s": 0.02,
            "n_requests": FLOOR_N_REQUESTS, "statistic": "median",
        }
        ls = doc["lambda_star"]
        assert ls["label"] == "ESTIMATED"
        assert ls["lambda_star_qps"] == 1.0
        assert ls["n_steps"] == 2
        assert ls["steps"][0]["attainment"] == 1.0

    def test_written_json_feeds_both_registered_consumers(self, tmp_path, monkeypatch):
        rc, out_path = _run_main(tmp_path, monkeypatch, label="ESTIMATED")
        assert rc == 0
        doc = json.loads(out_path.read_text(encoding="utf-8"))
        # Consumer 1: the D6 rate grid seeds from lambda_star_qps.
        schedules = build_rate_grid_schedules(
            doc["lambda_star"]["lambda_star_qps"], seed=1, duration_s=10.0
        )
        assert 0.5 in schedules and 1.2 in schedules
        # Consumer 2: the SLO thresholds seed from the floor pair unchanged.
        baseline = SLOBaseline(
            ttft_s=doc["floor"]["ttft_s"], tpot_s=doc["floor"]["tpot_s"]
        )
        assert baseline.tpot_s == 0.02

    @pytest.mark.parametrize("label", ["NONE_SUSTAINABLE", "LADDER_EXHAUSTED"])
    def test_unbracketed_lambda_star_fails_closed(self, tmp_path, monkeypatch, label):
        # Fail closed: the JSON is still written (provenance of the failed
        # probe) but the exit code is 1 — an unbracketed lambda* must never
        # seed a rate grid.
        rc, out_path = _run_main(tmp_path, monkeypatch, label=label)
        assert rc == 1
        doc = json.loads(out_path.read_text(encoding="utf-8"))
        assert doc["lambda_star"]["label"] == label
        assert doc["lambda_star"]["lambda_star_qps"] is None


# --------------------------------------------------------------------------- #
# ADR-0122: the climb-and-bisect probe loop and the floor-derived start rung
# --------------------------------------------------------------------------- #


def _probe_stub(knee_qps: float, calls: list):
    """A probe_rate stand-in: rungs below the knee complete fully, rungs at
    or above it complete 80% (attainment 0.8 < 0.9). Records every probed
    rate and phase in ``calls``."""
    async def fake_probe_rate(adapter, requests, rate_qps, seed, *, phase="ladder"):
        calls.append((round(rate_qps, 4), phase))
        done = 100 if rate_qps < knee_qps else 80
        return ProbeStep(
            rate_qps=float(rate_qps), n_scheduled=100, n_completed=done,
            throughput_rps=done / 65.0, phase=phase,
        )
    return fake_probe_rate


def test_probe_ladder_climbs_to_the_first_failure_then_bisects(monkeypatch):
    calls: list = []
    monkeypatch.setattr(cc, "probe_rate", _probe_stub(40.0, calls))

    est = asyncio.run(cc.run_probe_ladder(object(), [object()], 20.0))

    # Climb: 20, 26, 33.8 pass; 43.94 fails and stops the climb (no 12-rung
    # tail is probed). Bisect twice inside [33.8, 43.94]: 38.87 passes (lo
    # moves up), 41.405 fails (hi moves down).
    assert [c for c in calls if c[1] == "ladder"] == [
        (20.0, "ladder"), (26.0, "ladder"), (33.8, "ladder"), (43.94, "ladder"),
    ]
    assert [c for c in calls if c[1] == "bisect"] == [(38.87, "bisect"), (41.405, "bisect")]
    assert est.label == "ESTIMATED"
    assert est.lambda_star_qps == pytest.approx(38.87)
    assert est.first_unsustainable_qps == pytest.approx(41.405)
    assert [s.phase for s in est.steps].count("bisect") == 2


def test_probe_ladder_without_a_bracket_does_not_bisect(monkeypatch):
    calls: list = []
    # The knee is above every rung the ceiling allows: no bracket, no bisection.
    monkeypatch.setattr(cc, "probe_rate", _probe_stub(1e9, calls))

    est = asyncio.run(cc.run_probe_ladder(object(), [object()], 1.0))

    assert len(calls) == PROBE_MAX_STEPS == 30
    assert all(phase == "ladder" for _, phase in calls)
    assert est.label == "LADDER_EXHAUSTED"


def test_probe_ladder_first_rung_failure_does_not_bisect(monkeypatch):
    calls: list = []
    monkeypatch.setattr(cc, "probe_rate", _probe_stub(0.1, calls))

    est = asyncio.run(cc.run_probe_ladder(object(), [object()], 1.0))

    assert calls == [(1.0, "ladder")]
    assert est.label == "NONE_SUSTAINABLE"


def test_probe_ladder_climb_survives_a_throughput_dip_at_full_attainment(monkeypatch):
    """ADR-0121 at the driver level (review F9): the S0 draw, rung 1 completes
    42 of 42 and rung 2 completes 34 of 34 (throughput 0.646 then 0.523 rps,
    attainment 1.0 both). The HEAD driver carried its own retrograde clause
    and stopped the climb here; the climb must continue to the knee."""
    calls: list = []
    arrivals = {0.5: 42, 0.65: 34}

    async def fake_probe_rate(adapter, requests, rate_qps, seed, *, phase="ladder"):
        calls.append((round(rate_qps, 4), phase))
        n = arrivals.get(round(rate_qps, 4), 100)
        done = n if rate_qps < 1.0 else int(n * 0.8)
        return ProbeStep(
            rate_qps=float(rate_qps), n_scheduled=n, n_completed=done,
            throughput_rps=done / 65.0, phase=phase,
        )

    monkeypatch.setattr(cc, "probe_rate", fake_probe_rate)
    est = asyncio.run(cc.run_probe_ladder(object(), [object()], 0.5))

    ladder = [r for r, p in calls if p == "ladder"]
    assert ladder[:3] == [0.5, 0.65, 0.845]  # the dip at 0.65 did not stop the climb
    assert est.label == "ESTIMATED"
    assert est.lambda_star_qps > 0.65


def test_main_derives_the_start_rung_from_the_floor_when_unset(tmp_path, monkeypatch):
    manifest = _green_manifest(tmp_path)
    out_path = tmp_path / "out" / "calibration.json"
    monkeypatch.setattr(cc, "build_adapter", lambda *a, **k: StubAdapter())
    seen = {}

    async def fake_measure_floor(adapter, requests):
        return _floor()  # ttft 0.1 s, tpot 0.02 s

    async def fake_run_probe_ladder(adapter, requests, start_qps):
        seen["start"] = start_qps
        return _estimate("ESTIMATED")

    monkeypatch.setattr(cc, "measure_floor", fake_measure_floor)
    monkeypatch.setattr(cc, "run_probe_ladder", fake_run_probe_ladder)
    rc = cc.main([
        "--backend", "vllm", "--model", "Qwen/Qwen3-8B",
        "--api-base", "http://localhost:8000",
        "--manifest", str(manifest), "--output", str(out_path),
        "--budget-fraction", "0.5",  # no --start-qps
    ])
    assert rc == 0
    expected = 1.0 / (0.1 + (cc.CAL_MAX_TOKENS - 1) * 0.02)
    assert seen["start"] == pytest.approx(expected)
    doc = json.loads(out_path.read_text(encoding="utf-8"))
    assert doc["start_qps"] == pytest.approx(expected)
    assert doc["start_qps_source"] == "floor-service-rate"
