"""Tests for the registered lambda*/SLO-floor calibration procedure
(src/orchestration/calibration.py).

Charter authority: MyDocs/PUBLICATION.md D6 §6.1 — offered rates are
pre-registered fractions of a PREDICTED saturation rate lambda*; the primary
SLO pair is relative to a MEASURED single-stream floor. Pure-unit per the repo
doctrine: no network, no GPU, no server — probe steps are crafted by hand and
the decision rule is exercised label by label (mirroring
tests/test_load_generator.py and tests/test_goodput.py).
"""

from __future__ import annotations

import json
import math

import pytest

from src.orchestration.calibration import (
    FLOOR_N_REQUESTS,
    FLOOR_STATISTIC,
    PROBE_ATTAINMENT_MIN,
    PROBE_BISECT_STEPS,
    PROBE_LADDER_FACTOR,
    PROBE_MAX_STEPS,
    PROBE_WARMUP_S,
    PROBE_WINDOW_S,
    START_QPS_RULE,
    CalibrationError,
    CellCalibration,
    FloorMeasurement,
    LambdaStarEstimate,
    ProbeStep,
    decide_lambda_star,
    floor_start_qps,
    geometric_rate_ladder,
    summarize_floor,
)


def _step(rate: float, *, n: int = 100, completed: int = 100, tput: float = 1.0) -> ProbeStep:
    return ProbeStep(
        rate_qps=rate, n_scheduled=n, n_completed=completed, throughput_rps=tput
    )


# --------------------------------------------------------------------------- #
# Registered constants — the values ARE the registration
# --------------------------------------------------------------------------- #


def test_registered_constants_are_the_registered_values():
    assert FLOOR_N_REQUESTS == 30
    assert FLOOR_STATISTIC == "median"
    assert PROBE_LADDER_FACTOR == 1.3
    assert 60.0 <= PROBE_WINDOW_S <= 90.0  # charter 60-90 s band
    assert PROBE_WINDOW_S == 75.0
    assert PROBE_WARMUP_S == 10.0
    assert PROBE_ATTAINMENT_MIN == 0.9
    assert PROBE_MAX_STEPS == 30  # ADR-0122: a ceiling for the climb, not the ladder length
    assert PROBE_BISECT_STEPS == 2
    assert START_QPS_RULE == "floor-service-rate"


# --------------------------------------------------------------------------- #
# geometric_rate_ladder
# --------------------------------------------------------------------------- #


def test_ladder_is_geometric_from_start():
    ladder = geometric_rate_ladder(2.0, factor=1.5, max_steps=5)

    assert len(ladder) == 5
    assert ladder[0] == 2.0
    for lo, hi in zip(ladder, ladder[1:]):
        assert hi / lo == pytest.approx(1.5)


def test_ladder_defaults_use_registered_constants():
    ladder = geometric_rate_ladder(1.0)

    assert len(ladder) == PROBE_MAX_STEPS
    assert ladder[1] / ladder[0] == pytest.approx(PROBE_LADDER_FACTOR)


@pytest.mark.parametrize("bad_start", [0.0, -1.0, math.inf, math.nan, "fast", True])
def test_ladder_rejects_bad_start(bad_start):
    with pytest.raises(CalibrationError):
        geometric_rate_ladder(bad_start)


@pytest.mark.parametrize("bad_factor", [1.0, 0.5, 0.0, -2.0, math.inf, math.nan])
def test_ladder_rejects_non_geometric_factor(bad_factor):
    with pytest.raises(CalibrationError):
        geometric_rate_ladder(1.0, factor=bad_factor)


@pytest.mark.parametrize("bad_steps", [1, 0, -3, 2.0, True])
def test_ladder_rejects_bad_max_steps(bad_steps):
    with pytest.raises(CalibrationError):
        geometric_rate_ladder(1.0, max_steps=bad_steps)


# --------------------------------------------------------------------------- #
# ProbeStep
# --------------------------------------------------------------------------- #


def test_probe_step_attainment():
    step = _step(1.0, n=200, completed=150)

    assert step.attainment == pytest.approx(0.75)


def test_probe_step_validation():
    with pytest.raises(CalibrationError):
        _step(0.0)  # non-positive rate
    with pytest.raises(CalibrationError):
        _step(1.0, n=0)  # no scheduled arrivals
    with pytest.raises(CalibrationError):
        _step(1.0, n=10, completed=-1)  # negative completions
    with pytest.raises(CalibrationError):
        _step(1.0, n=10, completed=11)  # completions exceed arrivals
    with pytest.raises(CalibrationError):
        _step(1.0, tput=-0.1)  # negative throughput
    with pytest.raises(CalibrationError):
        _step(1.0, tput=math.nan)  # non-finite throughput


def test_probe_step_manifest_carries_attainment():
    row = _step(2.0, n=100, completed=90, tput=1.8).to_manifest()

    assert row["rate_qps"] == 2.0
    assert row["attainment"] == pytest.approx(0.9)
    assert row["throughput_rps"] == pytest.approx(1.8)


# --------------------------------------------------------------------------- #
# decide_lambda_star — the registered decision rule, label by label
# --------------------------------------------------------------------------- #


def test_clean_bracket_is_estimated_at_last_sustained_rate():
    steps = [
        _step(1.0, completed=100, tput=1.0),
        _step(1.3, completed=100, tput=1.3),
        _step(1.69, completed=98, tput=1.65),   # sustainable: attainment 0.98
        _step(2.197, completed=60, tput=1.2),   # unsustainable: attainment 0.6
    ]

    est = decide_lambda_star(steps)

    assert est.label == "ESTIMATED"
    assert est.lambda_star_qps == pytest.approx(1.69)
    assert est.sustained_rate_qps == pytest.approx(1.69)
    assert est.first_unsustainable_qps == pytest.approx(2.197)
    assert est.steps == tuple(steps)


def test_all_unsustainable_is_none_sustainable():
    steps = [
        _step(4.0, completed=50, tput=2.0),
        _step(5.2, completed=40, tput=1.9),
    ]

    est = decide_lambda_star(steps)

    assert est.label == "NONE_SUSTAINABLE"
    assert est.lambda_star_qps is None
    assert est.sustained_rate_qps is None
    assert est.first_unsustainable_qps == pytest.approx(4.0)


def test_all_sustainable_is_ladder_exhausted_never_extrapolated():
    steps = [
        _step(1.0, completed=100, tput=1.0),
        _step(1.3, completed=99, tput=1.29),
        _step(1.69, completed=97, tput=1.6),
    ]

    est = decide_lambda_star(steps)

    assert est.label == "LADDER_EXHAUSTED"
    assert est.lambda_star_qps is None  # never extrapolate past the ladder
    assert est.sustained_rate_qps == pytest.approx(1.69)
    assert est.first_unsustainable_qps is None


def test_throughput_dip_with_full_attainment_stays_sustainable():
    # ADR-0121 (S0F-11, live 2026-09-30): the pre-cal-v2 rule marked a rung
    # unsustainable when its throughput fell below the previous rung's. With
    # attainment >= 0.9, throughput_rps = n_completed / measured seconds is the
    # rung's REALIZED Poisson arrival rate times attainment, a random draw: the
    # H100 run drew 42 arrivals at 0.5 qps and 34 at 0.65 qps (both 100%
    # complete) and the rule declared 0.65 unsustainable on an idle GPU. The
    # registered currency is attainment alone (charter (c): >= 90% complete);
    # the cliff is retrograde GOODPUT in the analysis, never a ladder verdict.
    steps = [
        _step(0.5, n=42, completed=42, tput=42 / 65.0),
        _step(0.65, n=34, completed=34, tput=34 / 65.0),  # fewer arrivals, all complete
    ]

    est = decide_lambda_star(steps)

    assert est.label == "LADDER_EXHAUSTED"  # both rungs pass: no bracket, never a guess
    assert est.lambda_star_qps is None
    assert est.sustained_rate_qps == pytest.approx(0.65)


def test_attainment_is_the_only_sustainability_test():
    # A throughput fall with attainment still >= 0.9 does not bracket lambda*;
    # a fall WITH an attainment shortfall does (the completed fraction decides).
    steps = [
        _step(1.0, completed=100, tput=1.0),
        _step(1.3, completed=100, tput=1.3),
        _step(1.69, completed=100, tput=1.1),  # dip, attainment 1.0: still sustainable
        _step(2.197, completed=80, tput=1.0),  # attainment 0.8 < 0.9: unsustainable
    ]

    est = decide_lambda_star(steps)

    assert est.label == "ESTIMATED"
    assert est.lambda_star_qps == pytest.approx(1.69)
    assert est.first_unsustainable_qps == pytest.approx(2.197)


def test_single_sustainable_step_is_ladder_exhausted():
    est = decide_lambda_star([_step(1.0, completed=100, tput=1.0)])

    assert est.label == "LADDER_EXHAUSTED"


def test_lambda_star_is_highest_bracketed_sustained_rate():
    # Two sustainable->unsustainable transitions: the rule takes the HIGHEST.
    steps = [
        _step(1.0, completed=100, tput=1.0),
        _step(1.3, completed=50, tput=0.7),     # dip (unsustainable)
        _step(1.69, completed=100, tput=1.69),  # recovers (sustainable)
        _step(2.197, completed=40, tput=0.9),   # unsustainable again
    ]

    est = decide_lambda_star(steps)

    assert est.label == "ESTIMATED"
    assert est.lambda_star_qps == pytest.approx(1.69)
    assert est.first_unsustainable_qps == pytest.approx(2.197)


def test_non_increasing_rates_raise():
    with pytest.raises(CalibrationError):
        decide_lambda_star([_step(2.0), _step(1.0)])
    with pytest.raises(CalibrationError):
        decide_lambda_star([_step(1.0), _step(1.0)])


def test_empty_and_non_probestep_inputs_raise():
    with pytest.raises(CalibrationError):
        decide_lambda_star([])
    with pytest.raises(CalibrationError):
        decide_lambda_star([_step(1.0), "not a step"])  # type: ignore[list-item]


def test_estimate_label_value_consistency_is_enforced():
    # Honest-labels invariant: lambda* set iff ESTIMATED.
    with pytest.raises(CalibrationError):
        LambdaStarEstimate(
            label="LADDER_EXHAUSTED",
            lambda_star_qps=2.0,  # a guess smuggled under a non-ESTIMATED label
            sustained_rate_qps=2.0,
            first_unsustainable_qps=None,
            steps=(_step(1.0),),
        )


# --------------------------------------------------------------------------- #
# summarize_floor
# --------------------------------------------------------------------------- #


def test_floor_exact_medians_and_ms_to_s_conversion():
    n = FLOOR_N_REQUESTS  # 30: even count -> median averages the middle pair
    ttft_ms = [40.0] * 14 + [50.0, 50.0] + [60.0] * 14
    tpot_ms = [10.0] * (n - 1) + [12.0]

    floor = summarize_floor(ttft_ms, tpot_ms)

    assert floor.ttft_s == pytest.approx(0.050)  # median 50 ms -> seconds
    assert floor.tpot_s == pytest.approx(0.010)  # median 10 ms -> seconds
    assert floor.n_requests == n
    assert floor.statistic == FLOOR_STATISTIC


def test_floor_rejects_fewer_than_n_min_requests():
    short = [50.0] * (FLOOR_N_REQUESTS - 1)
    with pytest.raises(CalibrationError):
        summarize_floor(short, short)


def test_floor_n_min_override():
    floor = summarize_floor([40.0, 50.0, 60.0], [9.0, 10.0, 11.0], n_min=3)

    assert floor.ttft_s == pytest.approx(0.050)
    assert floor.n_requests == 3


def test_floor_rejects_none_nonfinite_and_nonpositive_values():
    good = [50.0] * FLOOR_N_REQUESTS
    for poison in (None, math.nan, math.inf, 0.0, -5.0):
        bad = list(good)
        bad[7] = poison
        with pytest.raises(CalibrationError):
            summarize_floor(bad, good)  # poisoned TTFT
        with pytest.raises(CalibrationError):
            summarize_floor(good, bad)  # poisoned TPOT


def test_floor_rejects_unpaired_inputs():
    with pytest.raises(CalibrationError):
        summarize_floor([50.0] * FLOOR_N_REQUESTS, [10.0] * (FLOOR_N_REQUESTS + 1))


def test_floor_measurement_validates_itself():
    with pytest.raises(CalibrationError):
        FloorMeasurement(ttft_s=0.0, tpot_s=0.01, n_requests=30)
    with pytest.raises(CalibrationError):
        FloorMeasurement(ttft_s=0.05, tpot_s=math.nan, n_requests=30)
    with pytest.raises(CalibrationError):
        FloorMeasurement(ttft_s=0.05, tpot_s=0.01, n_requests=0)


# --------------------------------------------------------------------------- #
# CellCalibration manifest round-trip
# --------------------------------------------------------------------------- #


def _cell() -> CellCalibration:
    floor = summarize_floor(
        [50.0] * FLOOR_N_REQUESTS, [10.0] * FLOOR_N_REQUESTS
    )
    estimate = decide_lambda_star(
        [
            _step(1.0, completed=100, tput=1.0),
            _step(1.3, completed=50, tput=0.9),
        ]
    )
    return CellCalibration(
        model="qwen3-8b",
        engine="vllm",
        budget_fraction=0.5,
        floor=floor,
        lambda_star=estimate,
    )


def test_cell_calibration_manifest_shape_and_json_round_trip():
    manifest = _cell().to_manifest()
    restored = json.loads(json.dumps(manifest))  # must be JSON-serializable

    assert restored["model"] == "qwen3-8b"
    assert restored["engine"] == "vllm"
    assert restored["budget_fraction"] == 0.5
    assert restored["procedure_version"] == "cal-v2 (2026-09-30)"
    assert restored["confirmatory"] is False  # never enters confirmatory analysis
    assert restored["procedure"] == {
        "floor_n_requests": 30,
        "floor_statistic": "median",
        "probe_ladder_factor": 1.3,
        "probe_window_s": 75.0,
        "probe_warmup_s": 10.0,
        "probe_attainment_min": 0.9,
        "probe_max_steps": 30,  # ADR-0122 ceiling
        "probe_bisect_steps": 2,  # ADR-0122
        "start_qps_rule": "floor-service-rate",  # ADR-0122
    }
    assert restored["floor"] == {
        "ttft_s": 0.05,
        "tpot_s": 0.01,
        "n_requests": 30,
        "statistic": "median",
    }
    lam = restored["lambda_star"]
    assert lam["label"] == "ESTIMATED"
    assert lam["lambda_star_qps"] == 1.0
    assert lam["first_unsustainable_qps"] == 1.3
    assert lam["n_steps"] == 2
    assert [s["rate_qps"] for s in lam["steps"]] == [1.0, 1.3]


def test_cell_calibration_validates_identity_fields():
    good = _cell()
    with pytest.raises(CalibrationError):
        CellCalibration(
            model="", engine="vllm", budget_fraction=0.5,
            floor=good.floor, lambda_star=good.lambda_star,
        )
    # budget_fraction is the charter's KV budget RATIO r, and §6.1 measures
    # the SLO floor AT r=1.5 — values above 1 are legitimate charter points
    # (walkthrough verification catch, 2026-08-12).
    over_one = CellCalibration(
        model="qwen3-8b", engine="vllm", budget_fraction=1.5,
        floor=good.floor, lambda_star=good.lambda_star,
    )
    assert over_one.budget_fraction == 1.5
    with pytest.raises(CalibrationError):
        CellCalibration(
            model="qwen3-8b", engine="vllm", budget_fraction=0.0,  # not positive
            floor=good.floor, lambda_star=good.lambda_star,
        )
    with pytest.raises(CalibrationError):
        CellCalibration(
            model="qwen3-8b", engine="vllm", budget_fraction=0.5,
            floor={"ttft_s": 0.05}, lambda_star=good.lambda_star,  # type: ignore[arg-type]
        )


# --------------------------------------------------------------------------- #
# ADR-0122 (S0F-12): climb-until-failure ceiling, bisection, floor-derived start
# --------------------------------------------------------------------------- #


def test_probe_step_phase_is_ladder_or_bisect():
    assert _step(1.0).phase == "ladder"
    bis = ProbeStep(rate_qps=1.5, n_scheduled=10, n_completed=10, throughput_rps=0.1, phase="bisect")
    assert bis.to_manifest()["phase"] == "bisect"
    with pytest.raises(CalibrationError):
        ProbeStep(rate_qps=1.5, n_scheduled=10, n_completed=10, throughput_rps=0.1, phase="guess")


def test_floor_start_qps_is_the_single_stream_service_rate():
    # S0 floors (vLLM, H100, 2026-09-30): ttft 0.1011 s, tpot 6.58 ms; one
    # 256-token stream takes 0.1011 + 255 x 0.00658 = 1.779 s -> 0.562 qps.
    floor = FloorMeasurement(ttft_s=0.1011, tpot_s=0.00658, n_requests=30)
    assert floor_start_qps(floor, max_tokens=256) == pytest.approx(1.0 / (0.1011 + 255 * 0.00658))
    with pytest.raises(CalibrationError):
        floor_start_qps(floor, max_tokens=1)  # TPOT needs a second token


def test_decision_over_rate_sorted_bisection_steps_refines_the_bracket():
    # Climb: 20, 26, 33.8 pass; 43.94 fails. Bisect: 38.87 passes, 41.41 fails.
    # Sorted by rate, the highest pass-then-fail transition is 38.87 -> 41.41.
    climb = [
        _step(20.0, completed=100), _step(26.0, completed=100),
        _step(33.8, completed=100), _step(43.94, completed=80),
    ]
    bisect = [
        ProbeStep(rate_qps=38.87, n_scheduled=100, n_completed=100, throughput_rps=1.0, phase="bisect"),
        ProbeStep(rate_qps=41.41, n_scheduled=100, n_completed=85, throughput_rps=1.0, phase="bisect"),
    ]
    est = decide_lambda_star(sorted(climb + bisect, key=lambda s: s.rate_qps))

    assert est.label == "ESTIMATED"
    assert est.lambda_star_qps == pytest.approx(38.87)
    assert est.first_unsustainable_qps == pytest.approx(41.41)
    assert est.to_manifest()["n_steps"] == 6
    assert [s["phase"] for s in est.to_manifest()["steps"]].count("bisect") == 2


def test_ladder_ceiling_keeps_the_refusal():
    # 30 rungs all sustainable: the ceiling is reached without a bracket and
    # the rule still refuses to guess.
    steps = [_step(r, completed=100) for r in geometric_rate_ladder(0.5)]
    assert len(steps) == PROBE_MAX_STEPS == 30
    est = decide_lambda_star(steps)
    assert est.label == "LADDER_EXHAUSTED"
    assert est.lambda_star_qps is None


def test_calibration_records_the_start_rung_provenance():
    floor = FloorMeasurement(ttft_s=0.1, tpot_s=0.005, n_requests=30)
    est = decide_lambda_star([_step(1.0, completed=100), _step(1.3, completed=10)])
    cal = CellCalibration(
        model="qwen3-8b", engine="vllm", budget_fraction=1.5, floor=floor,
        lambda_star=est, start_qps=0.72, start_qps_source=START_QPS_RULE,
    )
    doc = cal.to_manifest()
    assert doc["procedure_version"] == "cal-v2 (2026-09-30)"
    assert doc["start_qps"] == pytest.approx(0.72)
    assert doc["start_qps_source"] == "floor-service-rate"
    assert doc["procedure"]["probe_bisect_steps"] == PROBE_BISECT_STEPS
    assert doc["procedure"]["probe_max_steps"] == 30
    with pytest.raises(CalibrationError):
        CellCalibration(
            model="qwen3-8b", engine="vllm", budget_fraction=1.5, floor=floor,
            lambda_star=est, start_qps=0.72, start_qps_source="guess",
        )
