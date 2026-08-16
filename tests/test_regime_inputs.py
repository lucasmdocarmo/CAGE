"""Tests for src.analysis.regime_inputs — the telemetry → §6.1 regime-input
bridge (E2/E2b 2026-08-12): ZOH ρ_KV time-average, cumulative-preemption
deltas, fail-closed refusals, and the UNKNOWN_TELEMETRY lane."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.analysis.goodput import (
    GoodputError,
    IN_REGIME,
    PAST_CLIFF,
    UNPRESSURED,
    label_regime,
)
from src.analysis.regime_inputs import (
    REGIME_UNKNOWN,
    RegimeInputError,
    WindowRegimeInputs,
    compute_regime_inputs,
    compute_window_regime_inputs,
    label_regime_with_refusal,
)


def _samples(ts: list, kv: list, pre: list) -> pd.DataFrame:
    return pd.DataFrame(
        {"ts_s": ts, "kv_cache_usage": kv, "preemptions_total": pre}
    )


def _canonical() -> pd.DataFrame:
    """Window [0, 10): covered time 8 (first sample at 2), ZOH integral
    0.5*4 + 1.0*2 + 0.8*2 = 5.6 -> mean 0.7, coverage 0.8; counter 5 -> 9."""
    return _samples([2.0, 6.0, 8.0], [0.5, 1.0, 0.8], [5, 5, 9])


class TestComputeWindowRegimeInputs:
    def test_pinned_zoh_math(self) -> None:
        w = compute_window_regime_inputs(_canonical(), 0.0, 10.0)
        assert w.rho_kv_time_avg == pytest.approx(0.7)
        assert w.scarcity_events == 4
        assert w.n_samples == 3
        # coverage 0.8 == min_coverage default: the boundary is inclusive.
        assert w.coverage == pytest.approx(0.8)
        assert w.window_start_s == 0.0
        assert w.window_end_s == 10.0

    def test_pre_window_samples_ignored_never_extrapolated(self) -> None:
        # A sample BEFORE window_start must not hold forward into the window:
        # the pre-first-sample span stays uncovered.
        frame = _samples(
            [-1.0, 2.0, 6.0, 8.0], [0.0, 0.5, 1.0, 0.8], [0, 5, 5, 9]
        )
        w = compute_window_regime_inputs(frame, 0.0, 10.0)
        assert w.rho_kv_time_avg == pytest.approx(0.7)
        assert w.coverage == pytest.approx(0.8)
        assert w.n_samples == 3
        assert w.scarcity_events == 4

    def test_window_end_is_exclusive(self) -> None:
        # ts == window_end_s is out-of-window; the last in-window sample
        # holds until window_end_s regardless.
        frame = _samples(
            [2.0, 6.0, 8.0, 10.0], [0.5, 1.0, 0.8, 0.0], [5, 5, 9, 100]
        )
        w = compute_window_regime_inputs(frame, 0.0, 10.0)
        assert w.rho_kv_time_avg == pytest.approx(0.7)
        assert w.n_samples == 3
        assert w.scarcity_events == 4

    def test_duplicate_timestamps_carry_zero_weight(self) -> None:
        frame = _samples(
            [2.0, 2.0, 6.0, 8.0], [0.9, 0.5, 1.0, 0.8], [5, 5, 5, 9]
        )
        w = compute_window_regime_inputs(frame, 0.0, 10.0)
        assert w.rho_kv_time_avg == pytest.approx(0.7)
        assert w.n_samples == 4

    def test_integral_float_counter_accepted(self) -> None:
        frame = _samples([2.0, 6.0, 8.0], [0.5, 1.0, 0.8], [3.0, 3.0, 7.0])
        w = compute_window_regime_inputs(frame, 0.0, 10.0)
        assert w.scarcity_events == 4
        assert isinstance(w.scarcity_events, int)

    def test_non_integer_counter_raises(self) -> None:
        frame = _samples([2.0, 6.0, 8.0], [0.5, 1.0, 0.8], [3.0, 3.5, 7.0])
        with pytest.raises(RegimeInputError, match="integer"):
            compute_window_regime_inputs(frame, 0.0, 10.0)

    @pytest.mark.parametrize(
        "pre",
        [
            [9, 5, 3],  # negative endpoint delta
            [5, 3, 9],  # mid-window dip, endpoints non-negative
        ],
    )
    def test_counter_decrease_means_restart_and_raises(self, pre: list) -> None:
        frame = _samples([2.0, 6.0, 8.0], [0.5, 1.0, 0.8], pre)
        with pytest.raises(RegimeInputError, match="restart"):
            compute_window_regime_inputs(frame, 0.0, 10.0)

    @pytest.mark.parametrize("hole", [None, math.nan])
    def test_absent_kv_gauge_raises_absence_is_not_zero(self, hole: object) -> None:
        frame = _samples([2.0, 6.0, 8.0], [0.5, hole, 0.8], [5, 5, 9])
        with pytest.raises(RegimeInputError, match="absence is not zero"):
            compute_window_regime_inputs(frame, 0.0, 10.0)

    def test_absent_preempt_counter_raises_absence_is_not_zero(self) -> None:
        frame = _samples([2.0, 6.0, 8.0], [0.5, 1.0, 0.8], [5, None, 9])
        with pytest.raises(RegimeInputError, match="absence is not zero"):
            compute_window_regime_inputs(frame, 0.0, 10.0)

    def test_kv_gauge_outside_unit_interval_raises(self) -> None:
        frame = _samples([2.0, 6.0, 8.0], [0.5, 1.5, 0.8], [5, 5, 9])
        with pytest.raises(RegimeInputError, match=r"\[0, 1\]"):
            compute_window_regime_inputs(frame, 0.0, 10.0)

    def test_too_few_samples_raises_naming_count_and_threshold(self) -> None:
        frame = _samples([2.0], [0.5], [5])
        with pytest.raises(RegimeInputError, match=r"1 in-window.*need >= 2"):
            compute_window_regime_inputs(frame, 0.0, 10.0)

    def test_low_coverage_raises(self) -> None:
        # First in-window sample at 5 -> covered 5 of 10, coverage 0.5 < 0.8.
        frame = _samples([5.0, 6.0], [0.5, 0.5], [0, 0])
        with pytest.raises(RegimeInputError, match="coverage"):
            compute_window_regime_inputs(frame, 0.0, 10.0)

    @pytest.mark.parametrize(
        "start,end",
        [(5.0, 5.0), (5.0, 1.0), (math.nan, 10.0), (0.0, math.inf)],
    )
    def test_bad_window_bounds_raise(self, start: float, end: float) -> None:
        with pytest.raises(RegimeInputError):
            compute_window_regime_inputs(_canonical(), start, end)

    def test_missing_column_raises(self) -> None:
        frame = _canonical().drop(columns=["kv_cache_usage"])
        with pytest.raises(RegimeInputError, match="kv_cache_usage"):
            compute_window_regime_inputs(frame, 0.0, 10.0)

    def test_non_monotonic_timestamps_raise(self) -> None:
        frame = _samples([2.0, 1.0, 8.0], [0.5, 1.0, 0.8], [5, 5, 9])
        with pytest.raises(RegimeInputError, match="monotonic"):
            compute_window_regime_inputs(frame, 0.0, 10.0)

    def test_non_finite_timestamps_raise(self) -> None:
        frame = _samples([2.0, math.nan, 8.0], [0.5, 1.0, 0.8], [5, 5, 9])
        with pytest.raises(RegimeInputError, match="non-finite"):
            compute_window_regime_inputs(frame, 0.0, 10.0)

    @pytest.mark.parametrize(
        "kwargs", [{"min_samples": 0}, {"min_coverage": 0.0}, {"min_coverage": 1.5}]
    )
    def test_bad_thresholds_raise(self, kwargs: dict) -> None:
        with pytest.raises(RegimeInputError):
            compute_window_regime_inputs(_canonical(), 0.0, 10.0, **kwargs)

    def test_to_flat_dict_round_trips_fields(self) -> None:
        w = compute_window_regime_inputs(_canonical(), 0.0, 10.0)
        flat = w.to_flat_dict()
        assert flat["rho_kv_time_avg"] == w.rho_kv_time_avg
        assert set(flat) == {
            f.name for f in WindowRegimeInputs.__dataclass_fields__.values()
        }


class TestComputeRegimeInputs:
    def test_one_row_per_window_all_certified(self) -> None:
        frame = compute_regime_inputs(_canonical(), [(0.0, 10.0), (1.0, 9.0)])
        assert len(frame) == 2
        assert frame["telemetry_ok"].all()
        assert frame["refusal_reason"].isna().all()
        assert frame.loc[0, "rho_kv_time_avg"] == pytest.approx(0.7)
        assert frame.loc[0, "scarcity_events"] == 4
        # Window [1, 9): covered 9-2=7, integral 0.5*4 + 1.0*2 + 0.8*1 = 4.8.
        assert frame.loc[1, "rho_kv_time_avg"] == pytest.approx(4.8 / 7.0)
        assert frame.loc[1, "coverage"] == pytest.approx(7.0 / 8.0)

    def test_allow_missing_yields_refusal_row_not_a_numeric(self) -> None:
        frame = compute_regime_inputs(
            _canonical(), [(0.0, 10.0), (100.0, 110.0)], allow_missing=True
        )
        good, bad = frame.iloc[0], frame.iloc[1]
        assert bool(good["telemetry_ok"]) is True
        assert good["refusal_reason"] is None
        assert bool(bad["telemetry_ok"]) is False
        assert "sample" in bad["refusal_reason"]
        # E2b: the refused window carries NaN, never a coercible zero.
        assert math.isnan(bad["rho_kv_time_avg"])
        assert math.isnan(bad["scarcity_events"])
        assert math.isnan(bad["coverage"])
        assert bad["window_start_s"] == 100.0

    def test_default_reraises_first_error(self) -> None:
        with pytest.raises(RegimeInputError, match="sample"):
            compute_regime_inputs(_canonical(), [(0.0, 10.0), (100.0, 110.0)])

    def test_empty_windows_raise(self) -> None:
        with pytest.raises(RegimeInputError, match="empty"):
            compute_regime_inputs(_canonical(), [])

    def test_malformed_window_raises(self) -> None:
        with pytest.raises(RegimeInputError, match="pair"):
            compute_regime_inputs(_canonical(), [(0.0, 10.0, 20.0)])


class TestLabelRegimeWithRefusal:
    def _cells(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "rho_kv_time_avg": [0.95, 0.5, 0.95, math.nan],
                "scarcity_events": [3.0, 3.0, 3.0, math.nan],
                "attainment": [0.95, 0.95, 0.5, 0.95],
                "telemetry_ok": [True, True, True, False],
            },
            index=[10, 20, 30, 40],
        )

    def test_mixes_goodput_labels_and_unknown(self) -> None:
        labels = label_regime_with_refusal(self._cells())
        assert labels.tolist() == [
            IN_REGIME,
            UNPRESSURED,
            PAST_CLIFF,
            REGIME_UNKNOWN,
        ]
        assert labels.name == "regime"
        assert list(labels.index) == [10, 20, 30, 40]

    def test_all_refused_frame_needs_no_numeric_inputs(self) -> None:
        cells = self._cells()
        cells["telemetry_ok"] = False
        labels = label_regime_with_refusal(cells)
        assert (labels == REGIME_UNKNOWN).all()

    def test_absent_ok_col_delegates_to_goodput(self) -> None:
        cells = self._cells().iloc[:3].drop(columns=["telemetry_ok"])
        labels = label_regime_with_refusal(cells)
        expected = label_regime(
            cells,
            rho_col="rho_kv_time_avg",
            events_col="scarcity_events",
            attainment_col="attainment",
        )
        assert labels.tolist() == expected.tolist()

    def test_absent_ok_col_delegates_validation_to_goodput(self) -> None:
        cells = self._cells().drop(columns=["telemetry_ok"])  # row 40 has NaN rho
        with pytest.raises(GoodputError):
            label_regime_with_refusal(cells)

    def test_nan_in_ok_col_raises_fail_closed(self) -> None:
        cells = self._cells()
        cells["telemetry_ok"] = [1.0, 1.0, 1.0, math.nan]
        with pytest.raises(RegimeInputError, match="telemetry_ok"):
            label_regime_with_refusal(cells)

    def test_unknown_is_outside_the_charter_vocabulary(self) -> None:
        # UNKNOWN_TELEMETRY is an operational refusal, never a §6.1 grid label.
        assert REGIME_UNKNOWN == "UNKNOWN_TELEMETRY"
        assert REGIME_UNKNOWN not in {IN_REGIME, UNPRESSURED, PAST_CLIFF}

    def test_end_to_end_bridge_to_labels(self) -> None:
        # High-pressure telemetry: integral 0.95*4 + 0.95*2 + 0.92*2 = 7.54,
        # covered 8 -> rho 0.9425 >= 0.9; scarcity 4 > 0.
        telemetry = _samples(
            [2.0, 6.0, 8.0], [0.95, 0.95, 0.92], [5, 5, 9]
        )
        frame = compute_regime_inputs(
            telemetry, [(0.0, 10.0), (100.0, 110.0)], allow_missing=True
        )
        frame["attainment"] = [0.95, 0.95]
        labels = label_regime_with_refusal(frame)
        assert labels.tolist() == [IN_REGIME, REGIME_UNKNOWN]
