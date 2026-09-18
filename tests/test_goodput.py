"""Tests for src.analysis.goodput — Y window metrics, knee/cliff onsets,
§6.1 regime labels, and the Rogan-Gladen correction."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.analysis.goodput import (
    ATTAINMENT_MIN,
    CORRECTED_YIELD_ASSUMPTION,
    CORRECTED_YIELD_ESTIMATOR,
    CorrectedYield,
    GoldStratum,
    GoodputError,
    InstrumentAccuracy,
    RHO_KV_MIN,
    SLOBaseline,
    TPOT_SLO_MULTIPLIER,
    TTFT_SLO_MULTIPLIER,
    WindowMetrics,
    classify_regime,
    corrected_rate,
    corrected_yield,
    corrected_yield_from_flags,
    corrected_yield_from_window,
    evaluate_window,
    find_cliff,
    find_knee,
    label_regime,
    reweight_gold_sample,
)

# SLO thresholds under this baseline: ttft <= 1.0 s, tpot <= 0.1 s.
BASELINE = SLOBaseline(ttft_s=0.1, tpot_s=0.02)


def _window() -> pd.DataFrame:
    """10 issued: 6 timely (4 veridical), 2 completed-but-slow (1 veridical),
    2 failed (NaN latencies, non-veridical)."""
    return pd.DataFrame(
        {
            "ttft_s": [0.5] * 6 + [2.0, 2.0] + [np.nan, np.nan],
            "tpot_s": [0.05] * 6 + [0.05, 0.05] + [np.nan, np.nan],
            "ok": [True] * 8 + [False, False],
            "veridical": [True] * 4 + [False] * 2 + [True, False] + [False, False],
            "arrival_s": list(np.linspace(100.0, 109.0, 10)),
        }
    )


class TestEvaluateWindow:
    def test_known_window_all_currencies(self) -> None:
        m = evaluate_window(_window(), BASELINE, duration_s=10.0)
        assert m.n_issued == 10
        assert m.n_completed == 8
        assert m.n_timely == 6
        assert m.n_veridical == 5
        assert m.n_yield == 4
        assert m.duration_s == 10.0
        assert m.attainment == pytest.approx(0.8)
        assert m.throughput_rps == pytest.approx(0.8)
        assert m.goodput_rps == pytest.approx(0.6)
        assert m.yield_rps == pytest.approx(0.4)
        assert m.goodput_frac == pytest.approx(0.6)
        assert m.yield_frac == pytest.approx(0.4)
        assert m.veridical_frac == pytest.approx(0.5)
        # S1 clause b: independence null G*E[v] and the covariance gap.
        assert m.independence_null_rps == pytest.approx(0.3)
        assert m.independence_null_frac == pytest.approx(0.3)
        assert m.covariance_gap == pytest.approx(0.1)
        assert m.covariance_gap_rps == pytest.approx(0.1)
        # §9.2 truth tax G - Y.
        assert m.truth_tax_rps == pytest.approx(0.2)
        assert m.truth_tax_frac == pytest.approx(0.2)

    def test_independent_flags_have_zero_covariance_gap(self) -> None:
        records = pd.DataFrame(
            {
                "ttft_s": [0.5, 0.5, 5.0, 5.0],
                "tpot_s": [0.05] * 4,
                "ok": [True] * 4,
                "veridical": [True, False, True, False],
            }
        )
        m = evaluate_window(records, BASELINE, duration_s=4.0)
        assert m.covariance_gap == pytest.approx(0.0)
        assert m.covariance_gap_rps == pytest.approx(0.0)
        assert m.yield_rps == pytest.approx(m.independence_null_rps)

    def test_slo_boundary_is_inclusive(self) -> None:
        records = pd.DataFrame(
            {
                "ttft_s": [TTFT_SLO_MULTIPLIER * BASELINE.ttft_s],
                "tpot_s": [TPOT_SLO_MULTIPLIER * BASELINE.tpot_s],
                "ok": [True],
                "veridical": [True],
            }
        )
        m = evaluate_window(records, BASELINE, duration_s=1.0)
        assert m.n_timely == 1
        assert m.n_yield == 1

    def test_secondary_gate_multipliers_change_timeliness(self) -> None:
        records = pd.DataFrame(
            {
                "ttft_s": [0.7],
                "tpot_s": [0.05],
                "ok": [True],
                "veridical": [True],
            }
        )
        primary = evaluate_window(records, BASELINE, duration_s=1.0)
        secondary = evaluate_window(
            records, BASELINE, duration_s=1.0, ttft_multiplier=5.0
        )
        assert primary.n_timely == 1
        assert secondary.n_timely == 0

    def test_duration_derived_from_arrival_span(self) -> None:
        m = evaluate_window(_window(), BASELINE)
        assert m.duration_s == pytest.approx(9.0)
        assert m.goodput_rps == pytest.approx(6 / 9.0)

    def test_veridical_nan_on_failed_rows_counts_as_false(self) -> None:
        records = _window()
        records["veridical"] = records["veridical"].astype(float)
        records.loc[8, "veridical"] = np.nan
        m = evaluate_window(records, BASELINE, duration_s=10.0)
        assert m.n_veridical == 5

    def test_to_flat_dict_round_trips_fields(self) -> None:
        m = evaluate_window(_window(), BASELINE, duration_s=10.0)
        flat = m.to_flat_dict()
        assert flat["yield_rps"] == m.yield_rps
        assert set(flat) == {f.name for f in WindowMetrics.__dataclass_fields__.values()}

    def test_empty_window_raises(self) -> None:
        with pytest.raises(GoodputError, match="empty window"):
            evaluate_window(pd.DataFrame(columns=["ttft_s"]), BASELINE, duration_s=1.0)

    def test_missing_column_raises(self) -> None:
        records = _window().drop(columns=["veridical"])
        with pytest.raises(GoodputError, match="veridical"):
            evaluate_window(records, BASELINE, duration_s=1.0)

    def test_nan_latency_on_completed_row_raises(self) -> None:
        records = _window()
        records.loc[0, "ttft_s"] = np.nan
        with pytest.raises(GoodputError, match="ttft_s"):
            evaluate_window(records, BASELINE, duration_s=1.0)

    def test_veridical_true_on_failed_row_raises(self) -> None:
        records = _window()
        records.loc[9, "veridical"] = True
        with pytest.raises(GoodputError, match="non-completed"):
            evaluate_window(records, BASELINE, duration_s=1.0)

    def test_non_boolean_ok_raises(self) -> None:
        records = _window()
        records["ok"] = records["ok"].astype(float)
        records.loc[0, "ok"] = 0.5
        with pytest.raises(GoodputError, match="'ok'"):
            evaluate_window(records, BASELINE, duration_s=1.0)

    def test_bad_duration_raises(self) -> None:
        with pytest.raises(GoodputError, match="duration_s"):
            evaluate_window(_window(), BASELINE, duration_s=0.0)

    def test_missing_arrival_and_duration_raises(self) -> None:
        records = _window().drop(columns=["arrival_s"])
        with pytest.raises(GoodputError, match="arrival_s"):
            evaluate_window(records, BASELINE)

    def test_zero_arrival_span_raises(self) -> None:
        records = _window()
        records["arrival_s"] = 100.0
        with pytest.raises(GoodputError, match="not positive"):
            evaluate_window(records, BASELINE)

    def test_bad_multiplier_raises(self) -> None:
        with pytest.raises(GoodputError, match="ttft_multiplier"):
            evaluate_window(_window(), BASELINE, duration_s=1.0, ttft_multiplier=0.0)

    @pytest.mark.parametrize("ttft_s,tpot_s", [(0.0, 0.02), (0.1, -1.0), (math.nan, 0.02)])
    def test_invalid_baseline_raises(self, ttft_s: float, tpot_s: float) -> None:
        with pytest.raises(GoodputError):
            SLOBaseline(ttft_s=ttft_s, tpot_s=tpot_s)


def _knee_sweep(rates: list[float], center: float, scale: float = 20.0) -> pd.DataFrame:
    rate_arr = np.asarray(rates, dtype=float)
    return pd.DataFrame(
        {
            "offered_rate": rate_arr,
            "throughput": 10.0 - scale * (rate_arr - center) ** 2,
            "latency": np.ones_like(rate_arr),
        }
    )


class TestFindKnee:
    def test_exact_quadratic_recovers_vertex(self) -> None:
        est = find_knee(
            _knee_sweep([1.0, 2.0, 3.0, 4.0, 5.0], center=3.2, scale=1.0),
            resolution=None,
        )
        assert est.kind == "knee"
        assert est.label == "ESTIMATED"
        assert est.onset_rate == pytest.approx(3.2)
        assert est.grid_rate == 3.0
        assert est.bracket == (2.0, 4.0)

    def test_registered_grid_conclusive_at_default_resolution(self) -> None:
        # Peak at 0.95 -> bracket (0.85, 1.05): 1.05/0.85 < 1.15**2.
        est = find_knee(_knee_sweep([0.5, 0.7, 0.85, 0.95, 1.05, 1.2], center=0.93))
        assert est.label == "ESTIMATED"
        assert est.onset_rate == pytest.approx(0.93)
        assert est.grid_rate == 0.95
        assert est.bracket == (0.85, 1.05)

    def test_coarse_bracket_is_inconclusive_never_guessed(self) -> None:
        est = find_knee(_knee_sweep([1.0, 2.0, 3.0, 4.0, 5.0], center=3.2, scale=1.0))
        assert est.label == "INCONCLUSIVE_AT_RESOLUTION"
        assert est.onset_rate is None
        assert est.grid_rate == 3.0
        assert est.bracket == (2.0, 4.0)

    def test_exact_geometric_grid_is_conclusive(self) -> None:
        # §6.1 registered grid: exact ×1.15 spacing. A two-step bracket then has
        # hi/lo == resolution**2 exactly; float rounding lands a few ulp above,
        # which must NOT flip the label to INCONCLUSIVE (2026-08-02 P0 dry-run).
        rates = (4.0 * 1.15 ** np.arange(9)).tolist()
        est = find_knee(_knee_sweep(rates, center=rates[4], scale=0.05))
        assert est.label == "ESTIMATED"
        assert est.onset_rate == pytest.approx(rates[4])
        assert est.bracket == (rates[3], rates[5])

    def test_boundary_argmax_is_not_bracketed(self) -> None:
        sweep = pd.DataFrame(
            {
                "offered_rate": [1.0, 2.0, 3.0],
                "throughput": [1.0, 2.0, 3.0],
                "latency": [1.0, 1.0, 1.0],
            }
        )
        est = find_knee(sweep)
        assert est.label == "NOT_BRACKETED"
        assert est.onset_rate is None
        assert est.grid_rate == 3.0
        assert est.bracket is None

    def test_plateau_is_inconclusive(self) -> None:
        sweep = pd.DataFrame(
            {
                "offered_rate": [1.0, 2.0, 3.0, 4.0],
                "throughput": [1.0, 5.0, 5.0, 1.0],
                "latency": [1.0, 1.0, 1.0, 1.0],
            }
        )
        est = find_knee(sweep, resolution=None)
        assert est.label == "INCONCLUSIVE_AT_RESOLUTION"
        assert est.onset_rate is None
        assert est.grid_rate == 2.0

    def test_unsorted_input_is_sorted_internally(self) -> None:
        sweep = _knee_sweep([1.0, 2.0, 3.0, 4.0, 5.0], center=3.2, scale=1.0)
        shuffled = sweep.sample(frac=1.0, random_state=7)
        assert find_knee(shuffled, resolution=None) == find_knee(sweep, resolution=None)

    def test_too_few_points_raises(self) -> None:
        with pytest.raises(GoodputError, match="grid point"):
            find_knee(_knee_sweep([1.0, 2.0], center=1.5))

    def test_duplicate_rates_raise(self) -> None:
        with pytest.raises(GoodputError, match="duplicate"):
            find_knee(_knee_sweep([1.0, 2.0, 2.0, 3.0], center=2.0))

    def test_nonpositive_latency_raises(self) -> None:
        sweep = _knee_sweep([1.0, 2.0, 3.0], center=2.0, scale=1.0)
        sweep.loc[1, "latency"] = 0.0
        with pytest.raises(GoodputError, match="latency"):
            find_knee(sweep)

    @pytest.mark.parametrize("resolution", [1.0, 0.5, -2.0])
    def test_invalid_resolution_raises(self, resolution: float) -> None:
        with pytest.raises(GoodputError, match="resolution"):
            find_knee(_knee_sweep([1.0, 2.0, 3.0], center=2.0), resolution=resolution)

    def test_invalid_alpha_raises(self) -> None:
        with pytest.raises(GoodputError, match="alpha"):
            find_knee(_knee_sweep([1.0, 2.0, 3.0], center=2.0), alpha=0.0)


def _cliff_sweep(rates: list[float], goodput: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"offered_rate": rates, "goodput": goodput})


class TestFindCliff:
    def test_first_retrograde_point_on_registered_grid(self) -> None:
        est = find_cliff(
            _cliff_sweep(
                [0.5, 0.7, 0.85, 0.95, 1.05, 1.2], [5.0, 7.0, 8.0, 8.5, 8.2, 6.0]
            )
        )
        assert est.kind == "cliff"
        assert est.label == "ESTIMATED"
        assert est.onset_rate == 1.05
        assert est.grid_rate == 1.05
        assert est.bracket == (0.95, 1.05)

    def test_monotone_goodput_is_not_observed(self) -> None:
        est = find_cliff(_cliff_sweep([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]))
        assert est.label == "NOT_OBSERVED"
        assert est.onset_rate is None
        assert est.grid_rate is None
        assert est.bracket is None

    def test_flat_goodput_is_not_retrograde(self) -> None:
        est = find_cliff(_cliff_sweep([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]))
        assert est.label == "NOT_OBSERVED"

    def test_coarse_bracket_is_inconclusive_never_guessed(self) -> None:
        est = find_cliff(_cliff_sweep([0.5, 1.0, 2.0], [5.0, 8.0, 7.0]))
        assert est.label == "INCONCLUSIVE_AT_RESOLUTION"
        assert est.onset_rate is None
        assert est.grid_rate == 2.0
        assert est.bracket == (1.0, 2.0)

    def test_unsorted_input_is_sorted_internally(self) -> None:
        sweep = _cliff_sweep(
            [0.5, 0.7, 0.85, 0.95, 1.05, 1.2], [5.0, 7.0, 8.0, 8.5, 8.2, 6.0]
        )
        shuffled = sweep.sample(frac=1.0, random_state=11)
        assert find_cliff(shuffled) == find_cliff(sweep)

    def test_single_point_raises(self) -> None:
        with pytest.raises(GoodputError, match="grid point"):
            find_cliff(_cliff_sweep([1.0], [5.0]))

    def test_negative_goodput_raises(self) -> None:
        with pytest.raises(GoodputError, match="goodput"):
            find_cliff(_cliff_sweep([1.0, 2.0], [5.0, -1.0]))


class TestRegimeLabels:
    def test_exported_constants_are_the_classifier_outputs(self) -> None:
        # 2026-08-02 harmonization: consumers (figure_pipeline) import these
        # names; they must be exactly what the classifier emits.
        from src.analysis.goodput import IN_REGIME, PAST_CLIFF, UNPRESSURED

        assert IN_REGIME == "IN_REGIME"
        assert UNPRESSURED == "UNPRESSURED"
        assert PAST_CLIFF == "PAST_CLIFF"
        assert (
            classify_regime(rho_kv=0.95, scarcity_events=3, attainment=0.95)
            == IN_REGIME
        )

    def test_in_regime(self) -> None:
        assert (
            classify_regime(rho_kv=0.95, scarcity_events=3, attainment=0.95)
            == "IN_REGIME"
        )

    def test_thresholds_are_inclusive(self) -> None:
        assert (
            classify_regime(
                rho_kv=RHO_KV_MIN, scarcity_events=1, attainment=ATTAINMENT_MIN
            )
            == "IN_REGIME"
        )

    def test_low_occupancy_is_unpressured(self) -> None:
        assert (
            classify_regime(rho_kv=0.5, scarcity_events=3, attainment=0.95)
            == "UNPRESSURED"
        )

    def test_zero_scarcity_events_is_unpressured(self) -> None:
        assert (
            classify_regime(rho_kv=0.95, scarcity_events=0, attainment=0.95)
            == "UNPRESSURED"
        )

    def test_low_attainment_is_past_cliff(self) -> None:
        assert (
            classify_regime(rho_kv=0.95, scarcity_events=3, attainment=0.5)
            == "PAST_CLIFF"
        )

    def test_joint_failure_past_cliff_wins(self) -> None:
        assert (
            classify_regime(rho_kv=0.1, scarcity_events=0, attainment=0.5)
            == "PAST_CLIFF"
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"rho_kv": math.nan, "scarcity_events": 1, "attainment": 0.95},
            {"rho_kv": -0.1, "scarcity_events": 1, "attainment": 0.95},
            {"rho_kv": 0.95, "scarcity_events": -1, "attainment": 0.95},
            {"rho_kv": 0.95, "scarcity_events": 2.5, "attainment": 0.95},
            {"rho_kv": 0.95, "scarcity_events": 1, "attainment": 1.2},
        ],
    )
    def test_domain_guards_raise(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(GoodputError):
            classify_regime(**kwargs)

    def test_vectorized_matches_scalar(self) -> None:
        cells = pd.DataFrame(
            {
                "rho_kv": [0.95, 0.5, 0.95, 0.1],
                "scarcity_events": [3, 3, 0, 0],
                "attainment": [0.95, 0.95, 0.95, 0.5],
            },
            index=[10, 20, 30, 40],
        )
        labels = label_regime(cells)
        expected = [
            classify_regime(
                rho_kv=row.rho_kv,
                scarcity_events=int(row.scarcity_events),
                attainment=row.attainment,
            )
            for row in cells.itertuples()
        ]
        assert labels.tolist() == expected
        assert labels.name == "regime"
        assert list(labels.index) == [10, 20, 30, 40]

    def test_vectorized_missing_column_raises(self) -> None:
        with pytest.raises(GoodputError, match="scarcity_events"):
            label_regime(pd.DataFrame({"rho_kv": [0.95], "attainment": [0.95]}))

    def test_vectorized_fractional_events_raise(self) -> None:
        cells = pd.DataFrame(
            {"rho_kv": [0.95], "scarcity_events": [2.5], "attainment": [0.95]}
        )
        with pytest.raises(GoodputError, match="integer"):
            label_regime(cells)


class TestCorrectedRate:
    def test_hand_computed_value(self) -> None:
        # (0.64 + 0.95 - 1) / (0.9 + 0.95 - 1) = 0.59 / 0.85
        assert corrected_rate(0.64, 0.9, 0.95) == pytest.approx(0.59 / 0.85)

    def test_perfect_instrument_is_identity(self) -> None:
        assert corrected_rate(0.37, 1.0, 1.0) == pytest.approx(0.37)

    def test_truncated_at_zero(self) -> None:
        # Raw estimate (0.02 - 0.05) / 0.85 < 0 -> truncated.
        assert corrected_rate(0.02, 0.9, 0.95) == 0.0

    def test_truncated_at_one(self) -> None:
        # Raw estimate (0.99 - 0.05) / 0.85 > 1 -> truncated.
        assert corrected_rate(0.99, 0.9, 0.95) == 1.0

    def test_uninformative_instrument_raises(self) -> None:
        with pytest.raises(GoodputError, match="uninformative"):
            corrected_rate(0.5, 0.6, 0.4)

    @pytest.mark.parametrize(
        "apparent,sensitivity,specificity",
        [(-0.1, 0.9, 0.9), (1.1, 0.9, 0.9), (0.5, 1.2, 0.9), (0.5, math.nan, 0.9)],
    )
    def test_domain_guards_raise(
        self, apparent: float, sensitivity: float, specificity: float
    ) -> None:
        with pytest.raises(GoodputError):
            corrected_rate(apparent, sensitivity, specificity)


# --------------------------------------------------------------------------- #
# ADR-0115 (backlog A3): corrected serving yield = SLO rate x corrected
# predicate rate among SLO-met requests. The instrument misclassifies only the
# predicate half of Y; "timely" is a clock measurement and is never corrected.
# --------------------------------------------------------------------------- #

# Synthetic SLO-met population with KNOWN truth: 200 timely rows, 120 truly
# veridical (p = 0.6). Instrument Se = 0.9 (108 TP, 12 FN), Sp = 0.95 (76 TN,
# 4 FP) applied EXACTLY, so apparent = (108 + 4) / 200 = 0.56 and the
# Rogan-Gladen correction recovers p exactly: (0.56 + 0.95 - 1) / 0.85 = 0.6.
_SE, _SP = 0.9, 0.95
_TRUE_P_GIVEN_SLO = 0.6
_APPARENT_GIVEN_SLO = 0.56


def _synthetic_flags(n_untimely: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """(timely, instrument_veridical) flags: 200 timely rows carrying the
    exact confusion counts above, plus ``n_untimely`` slow rows of which half
    carry an instrument-positive verdict (never part of Y)."""
    timely = np.array([True] * 200 + [False] * n_untimely)
    verdict_timely = [True] * 108 + [False] * 12 + [False] * 76 + [True] * 4
    verdict_slow = [True, False] * (n_untimely // 2) + [True] * (n_untimely % 2)
    return timely, np.array(verdict_timely + verdict_slow)


class TestCorrectedYield:
    def test_perfect_instrument_recomposes_raw_yield(self) -> None:
        rec = corrected_yield(
            slo_rate=0.6,
            apparent_predicate_rate_given_slo=0.75,
            sensitivity=1.0,
            specificity=1.0,
        )
        assert isinstance(rec, CorrectedYield)
        assert rec.yield_raw == pytest.approx(0.45)
        assert rec.yield_corrected == pytest.approx(rec.yield_raw)
        assert rec.corrected_predicate_rate_given_slo == pytest.approx(0.75)
        assert rec.truncated is False

    def test_synthetic_known_truth_is_recovered(self) -> None:
        rec = corrected_yield(
            slo_rate=0.5,
            apparent_predicate_rate_given_slo=_APPARENT_GIVEN_SLO,
            sensitivity=_SE,
            specificity=_SP,
        )
        assert rec.corrected_predicate_rate_given_slo == pytest.approx(_TRUE_P_GIVEN_SLO)
        assert rec.yield_corrected == pytest.approx(0.5 * _TRUE_P_GIVEN_SLO)
        assert rec.yield_raw == pytest.approx(0.5 * _APPARENT_GIVEN_SLO)
        assert rec.youden_j == pytest.approx(0.85)
        assert rec.truncated is False

    def test_conditional_correction_differs_from_correcting_the_conjunction(
        self,
    ) -> None:
        # The A3 defect: correcting Y = 0.28 as if the instrument saw the
        # conjunction gives (0.28 - 0.05) / 0.85 = 0.2706, not the true 0.30.
        rec = corrected_yield(
            slo_rate=0.5,
            apparent_predicate_rate_given_slo=_APPARENT_GIVEN_SLO,
            sensitivity=_SE,
            specificity=_SP,
        )
        conjunction = corrected_rate(0.5 * _APPARENT_GIVEN_SLO, _SE, _SP)
        assert conjunction == pytest.approx(0.23 / 0.85)
        assert rec.yield_corrected == pytest.approx(0.30)
        assert rec.yield_corrected != pytest.approx(conjunction)

    def test_slo_rate_is_never_corrected(self) -> None:
        rec = corrected_yield(
            slo_rate=0.37,
            apparent_predicate_rate_given_slo=0.5,
            sensitivity=0.8,
            specificity=0.9,
        )
        assert rec.slo_rate == 0.37
        assert rec.yield_corrected == pytest.approx(0.37 * rec.corrected_predicate_rate_given_slo)

    def test_truncation_at_zero_is_recorded(self) -> None:
        rec = corrected_yield(
            slo_rate=0.9,
            apparent_predicate_rate_given_slo=0.02,
            sensitivity=_SE,
            specificity=_SP,
        )
        assert rec.corrected_predicate_rate_given_slo == 0.0
        assert rec.yield_corrected == 0.0
        assert rec.truncated is True
        assert rec.corrected_predicate_rate_given_slo_untruncated == pytest.approx(-0.03 / 0.85)

    def test_truncation_at_one_is_recorded(self) -> None:
        rec = corrected_yield(
            slo_rate=0.9,
            apparent_predicate_rate_given_slo=0.99,
            sensitivity=_SE,
            specificity=_SP,
        )
        assert rec.corrected_predicate_rate_given_slo == 1.0
        assert rec.yield_corrected == pytest.approx(0.9)
        assert rec.truncated is True
        assert rec.corrected_predicate_rate_given_slo_untruncated > 1.0

    def test_record_carries_inputs_assumption_and_estimator(self) -> None:
        rec = corrected_yield(
            slo_rate=0.5,
            apparent_predicate_rate_given_slo=_APPARENT_GIVEN_SLO,
            sensitivity=_SE,
            specificity=_SP,
        )
        assert rec.apparent_predicate_rate_given_slo == _APPARENT_GIVEN_SLO
        assert rec.sensitivity == _SE
        assert rec.specificity == _SP
        assert rec.assumption == CORRECTED_YIELD_ASSUMPTION
        assert rec.estimator == CORRECTED_YIELD_ESTIMATOR
        assert "ADR-0115" in CORRECTED_YIELD_ESTIMATOR
        assert "arm" in CORRECTED_YIELD_ASSUMPTION
        assert "timel" in CORRECTED_YIELD_ASSUMPTION
        flat = rec.to_flat_dict()
        assert flat["yield_corrected"] == rec.yield_corrected
        assert flat["assumption"] == CORRECTED_YIELD_ASSUMPTION

    def test_uninformative_instrument_raises(self) -> None:
        with pytest.raises(GoodputError, match="uninformative"):
            corrected_yield(
                slo_rate=0.5,
                apparent_predicate_rate_given_slo=0.5,
                sensitivity=0.5,
                specificity=0.5,
            )

    @pytest.mark.parametrize(
        "slo_rate,apparent,sensitivity,specificity",
        [
            (-0.1, 0.5, 0.9, 0.9),
            (1.1, 0.5, 0.9, 0.9),
            (math.nan, 0.5, 0.9, 0.9),
            (0.5, -0.1, 0.9, 0.9),
            (0.5, 1.1, 0.9, 0.9),
            (0.5, 0.5, 1.2, 0.9),
            (0.5, 0.5, 0.9, math.nan),
            (True, 0.5, 0.9, 0.9),
            ("0.5", 0.5, 0.9, 0.9),
        ],
    )
    def test_domain_guards_raise(
        self, slo_rate: object, apparent: float, sensitivity: float, specificity: float
    ) -> None:
        with pytest.raises(GoodputError):
            corrected_yield(
                slo_rate=slo_rate,  # type: ignore[arg-type]
                apparent_predicate_rate_given_slo=apparent,
                sensitivity=sensitivity,
                specificity=specificity,
            )


class TestCorrectedYieldFromFlags:
    def test_flags_reproduce_the_synthetic_truth(self) -> None:
        timely, verdict = _synthetic_flags()
        rec = corrected_yield_from_flags(
            timely=timely, veridical=verdict, sensitivity=_SE, specificity=_SP
        )
        assert rec.slo_rate == pytest.approx(0.5)
        assert rec.n_issued == 400
        assert rec.n_slo_met == 200
        assert rec.apparent_predicate_rate_given_slo == pytest.approx(_APPARENT_GIVEN_SLO)
        assert rec.corrected_predicate_rate_given_slo == pytest.approx(_TRUE_P_GIVEN_SLO)
        assert rec.yield_corrected == pytest.approx(0.30)

    def test_untimely_verdicts_never_enter_the_conditional_rate(self) -> None:
        timely, verdict = _synthetic_flags(n_untimely=0)
        a = corrected_yield_from_flags(
            timely=timely, veridical=verdict, sensitivity=_SE, specificity=_SP
        )
        timely_b, verdict_b = _synthetic_flags(n_untimely=200)
        b = corrected_yield_from_flags(
            timely=timely_b, veridical=verdict_b, sensitivity=_SE, specificity=_SP
        )
        assert a.apparent_predicate_rate_given_slo == pytest.approx(
            b.apparent_predicate_rate_given_slo
        )
        assert a.slo_rate == pytest.approx(1.0)
        assert b.slo_rate == pytest.approx(0.5)

    def test_perfect_instrument_matches_evaluate_window_yield_frac(self) -> None:
        frame = _window()
        m = evaluate_window(frame, BASELINE, duration_s=10.0)
        timely = np.array([True] * 6 + [False] * 4)
        rec = corrected_yield_from_flags(
            timely=timely,
            veridical=frame["veridical"].to_numpy(dtype=bool),
            sensitivity=1.0,
            specificity=1.0,
        )
        assert rec.yield_raw == pytest.approx(m.yield_frac)
        assert rec.yield_corrected == pytest.approx(m.yield_frac)
        assert rec.slo_rate == pytest.approx(m.goodput_frac)

    def test_accepts_series_and_0_1_ints(self) -> None:
        rec = corrected_yield_from_flags(
            timely=pd.Series([1, 1, 0, 1]),
            veridical=pd.Series([1, 0, 1, 1]),
            sensitivity=1.0,
            specificity=1.0,
        )
        assert rec.n_slo_met == 3
        assert rec.apparent_predicate_rate_given_slo == pytest.approx(2 / 3)

    def test_no_slo_met_requests_refused(self) -> None:
        with pytest.raises(GoodputError, match="no SLO-met"):
            corrected_yield_from_flags(
                timely=np.array([False, False]),
                veridical=np.array([False, False]),
                sensitivity=_SE,
                specificity=_SP,
            )

    def test_length_mismatch_refused(self) -> None:
        with pytest.raises(GoodputError, match="length"):
            corrected_yield_from_flags(
                timely=np.array([True, False]),
                veridical=np.array([True]),
                sensitivity=_SE,
                specificity=_SP,
            )

    def test_empty_refused(self) -> None:
        with pytest.raises(GoodputError, match="empty"):
            corrected_yield_from_flags(
                timely=np.array([], dtype=bool),
                veridical=np.array([], dtype=bool),
                sensitivity=_SE,
                specificity=_SP,
            )

    def test_nan_and_non_boolean_flags_refused(self) -> None:
        with pytest.raises(GoodputError, match="veridical"):
            corrected_yield_from_flags(
                timely=np.array([True, True]),
                veridical=np.array([1.0, np.nan]),
                sensitivity=_SE,
                specificity=_SP,
            )
        with pytest.raises(GoodputError, match="timely"):
            corrected_yield_from_flags(
                timely=np.array([2, 1]),
                veridical=np.array([True, True]),
                sensitivity=_SE,
                specificity=_SP,
            )


class TestCorrectedYieldFromWindow:
    def test_window_route_matches_flag_route(self) -> None:
        frame = _window()
        m = evaluate_window(frame, BASELINE, duration_s=10.0)
        rec = corrected_yield_from_window(m, sensitivity=_SE, specificity=_SP)
        timely = np.array([True] * 6 + [False] * 4)
        via_flags = corrected_yield_from_flags(
            timely=timely,
            veridical=frame["veridical"].to_numpy(dtype=bool),
            sensitivity=_SE,
            specificity=_SP,
        )
        assert rec == via_flags
        assert rec.slo_rate == pytest.approx(0.6)
        assert rec.apparent_predicate_rate_given_slo == pytest.approx(4 / 6)
        assert rec.yield_raw == pytest.approx(m.yield_frac)

    def test_window_with_no_timely_requests_refused(self) -> None:
        frame = _window()
        frame.loc[:, "ttft_s"] = 5.0
        m = evaluate_window(frame, BASELINE, duration_s=10.0)
        assert m.n_timely == 0
        with pytest.raises(GoodputError, match="no SLO-met"):
            corrected_yield_from_window(m, sensitivity=_SE, specificity=_SP)

    def test_non_window_object_refused(self) -> None:
        with pytest.raises(GoodputError, match="WindowMetrics"):
            corrected_yield_from_window(
                {"n_timely": 3},  # type: ignore[arg-type]
                sensitivity=_SE,
                specificity=_SP,
            )


# Population with known accuracy for the weighting helper: N = 1000, 600 truly
# veridical; Se = 0.9 (540 TP, 60 FN), Sp = 0.95 (380 TN, 20 FP). Verdict
# shares: positive 560 (540 true), negative 440 (60 true). A verdict-stratified
# gold sample that OVERSAMPLES the negative verdict (28 positives, 220
# negatives) keeps each stratum's gold-true fraction exact (27/28 and 30/220).
def _population_strata(dataset: str | None = None) -> list[GoldStratum]:
    return [
        GoldStratum(
            verdict=True, dataset=dataset, n_sampled=28, n_gold_true=27, population_count=560
        ),
        GoldStratum(
            verdict=False, dataset=dataset, n_sampled=220, n_gold_true=30, population_count=440
        ),
    ]


class TestReweightGoldSample:
    def test_recovers_population_sensitivity_and_specificity(self) -> None:
        acc = reweight_gold_sample(_population_strata())
        assert isinstance(acc, InstrumentAccuracy)
        assert acc.sensitivity == pytest.approx(_SE)
        assert acc.specificity == pytest.approx(_SP)
        assert acc.prevalence == pytest.approx(0.6)
        assert acc.n_sampled == 248
        assert acc.n_population == 1000
        assert acc.n_strata == 2
        assert acc.youden_j == pytest.approx(0.85)
        assert "ADR-0115" in acc.estimator

    def test_unweighted_pooling_would_be_wrong(self) -> None:
        # Pooling the oversampled negatives as if they were the population
        # gives Se = 27 / (27 + 30), a verification-bias artifact.
        acc = reweight_gold_sample(_population_strata())
        naive = 27 / 57
        assert acc.sensitivity != pytest.approx(naive)

    def test_weights_are_population_verdict_shares(self) -> None:
        acc = reweight_gold_sample(_population_strata())
        assert acc.weights == {(True, None): 0.56, (False, None): pytest.approx(0.44)}

    def test_dataset_strata_pool_and_filter(self) -> None:
        strata = _population_strata("hotpotqa") + [
            GoldStratum(
                verdict=True, dataset="qasper", n_sampled=10, n_gold_true=8, population_count=100
            ),
            GoldStratum(
                verdict=False, dataset="qasper", n_sampled=20, n_gold_true=4, population_count=100
            ),
        ]
        pooled = reweight_gold_sample(strata)
        assert pooled.n_strata == 4
        assert pooled.n_population == 1200
        hot = reweight_gold_sample(strata, dataset="hotpotqa")
        assert hot.n_strata == 2
        assert hot.sensitivity == pytest.approx(_SE)
        assert hot.specificity == pytest.approx(_SP)
        # qasper alone: prevalence = 0.5 * 0.8 + 0.5 * 0.2 = 0.5;
        # Se = 0.5 * 0.8 / 0.5 = 0.8; Sp = 0.5 * 0.8 / 0.5 = 0.8.
        qas = reweight_gold_sample(strata, dataset="qasper")
        assert qas.sensitivity == pytest.approx(0.8)
        assert qas.specificity == pytest.approx(0.8)
        assert qas.dataset == "qasper"
        assert pooled.dataset is None

    def test_feeds_corrected_yield(self) -> None:
        acc = reweight_gold_sample(_population_strata())
        rec = corrected_yield(
            slo_rate=0.5,
            apparent_predicate_rate_given_slo=_APPARENT_GIVEN_SLO,
            sensitivity=acc.sensitivity,
            specificity=acc.specificity,
        )
        assert rec.yield_corrected == pytest.approx(0.30)

    def test_missing_verdict_stratum_refused(self) -> None:
        with pytest.raises(GoodputError, match="missing"):
            reweight_gold_sample(_population_strata()[:1])

    def test_dataset_filter_without_match_refused(self) -> None:
        with pytest.raises(GoodputError, match="no strata"):
            reweight_gold_sample(_population_strata("hotpotqa"), dataset="qasper")

    def test_mixed_dataset_labeling_refused(self) -> None:
        strata = [_population_strata()[0], _population_strata("hotpotqa")[1]]
        with pytest.raises(GoodputError, match="dataset"):
            reweight_gold_sample(strata)

    def test_duplicate_stratum_refused(self) -> None:
        strata = _population_strata() + _population_strata()[:1]
        with pytest.raises(GoodputError, match="duplicate"):
            reweight_gold_sample(strata)

    def test_empty_refused(self) -> None:
        with pytest.raises(GoodputError, match="no strata"):
            reweight_gold_sample([])

    def test_degenerate_prevalence_refused(self) -> None:
        all_true = [
            GoldStratum(verdict=True, dataset=None, n_sampled=10, n_gold_true=10, population_count=50),
            GoldStratum(verdict=False, dataset=None, n_sampled=10, n_gold_true=10, population_count=50),
        ]
        with pytest.raises(GoodputError, match="specificity"):
            reweight_gold_sample(all_true)
        all_false = [
            GoldStratum(verdict=True, dataset=None, n_sampled=10, n_gold_true=0, population_count=50),
            GoldStratum(verdict=False, dataset=None, n_sampled=10, n_gold_true=0, population_count=50),
        ]
        with pytest.raises(GoodputError, match="sensitivity"):
            reweight_gold_sample(all_false)

    def test_uninformative_reweighted_instrument_refused(self) -> None:
        # Same gold-true fraction in both verdict strata: the verdict carries
        # no information (Se + Sp = 1).
        strata = [
            GoldStratum(verdict=True, dataset=None, n_sampled=10, n_gold_true=5, population_count=50),
            GoldStratum(verdict=False, dataset=None, n_sampled=10, n_gold_true=5, population_count=50),
        ]
        with pytest.raises(GoodputError, match="uninformative"):
            reweight_gold_sample(strata)

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(n_sampled=0, n_gold_true=0, population_count=10),
            dict(n_sampled=5, n_gold_true=6, population_count=10),
            dict(n_sampled=5, n_gold_true=-1, population_count=10),
            dict(n_sampled=5, n_gold_true=2, population_count=0),
            dict(n_sampled=5.0, n_gold_true=2, population_count=10),
            dict(n_sampled=5, n_gold_true=True, population_count=10),
        ],
    )
    def test_stratum_count_guards(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(GoodputError):
            GoldStratum(verdict=True, dataset=None, **kwargs)  # type: ignore[arg-type]

    def test_stratum_verdict_must_be_bool(self) -> None:
        with pytest.raises(GoodputError, match="verdict"):
            GoldStratum(
                verdict=1,  # type: ignore[arg-type]
                dataset=None,
                n_sampled=5,
                n_gold_true=2,
                population_count=10,
            )
