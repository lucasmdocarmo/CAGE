"""Tests for src.analysis.goodput — Y window metrics, knee/cliff onsets,
§6.1 regime labels, and the Rogan-Gladen correction."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.analysis.goodput import (
    ATTAINMENT_MIN,
    GoodputError,
    RHO_KV_MIN,
    SLOBaseline,
    TPOT_SLO_MULTIPLIER,
    TTFT_SLO_MULTIPLIER,
    WindowMetrics,
    classify_regime,
    corrected_rate,
    evaluate_window,
    find_cliff,
    find_knee,
    label_regime,
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
