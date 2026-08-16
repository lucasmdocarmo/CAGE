"""Telemetry → §6.1 regime-input bridge: per-window ρ_KV time-average and
scarcity-event deltas from the raw cage-stats vLLM time series.

Charter bindings (PUBLICATION.md):
- §6.1 3-layer in-regime criterion: ``goodput.classify_regime`` consumes
  (ρ_KV time-avg, scarcity counters, attainment) — this module produces the
  first two from telemetry samples; it imports the thresholds and the label
  machinery FROM ``src.analysis.goodput`` and duplicates neither.
- E2 (code-assertion walkthrough 2026-08-12): nothing upstream produced
  ``rho_kv``/``scarcity_events`` — this bridge is that producer. ρ_KV is the
  zero-order-hold time-weighted mean of the KV-occupancy gauge over the
  window; scarcity_events is the delta of the CUMULATIVE preemption counter.
- E2b, absence-is-not-zero doctrine: an engine that does not report the
  occupancy gauge (None/NaN samples), a window with too few samples, or a
  sparsely-covered window can NEVER be coerced into a numeric that would
  label it UNPRESSURED. Missing telemetry surfaces as an explicit refusal —
  ``RegimeInputError`` fail-closed, or the operational ``REGIME_UNKNOWN``
  row label under ``allow_missing=True``. ``UNKNOWN_TELEMETRY`` is
  deliberately OUTSIDE the charter's 3-label vocabulary: UNKNOWN rows never
  enter in-regime aggregates and are never one of the three grid labels.
- A negative counter delta means the server restarted mid-window; a restart
  invalidates the window (the counter reset destroys the delta's meaning).

Domain logic only: stdlib + numpy/pandas, no I/O, no plotting.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from src.analysis.goodput import label_regime as _goodput_label_regime

__all__ = [
    "REGIME_UNKNOWN",
    "RegimeInputError",
    "WindowRegimeInputs",
    "compute_regime_inputs",
    "compute_window_regime_inputs",
    "label_regime_with_refusal",
]

# Operational refusal label — NOT a §6.1 regime. Kept outside the goodput
# vocabulary on purpose: {IN_REGIME, UNPRESSURED, PAST_CLIFF} are the only
# grid labels; UNKNOWN_TELEMETRY marks rows whose telemetry could not certify
# any of the three, and such rows never enter in-regime aggregates.
REGIME_UNKNOWN: str = "UNKNOWN_TELEMETRY"


class RegimeInputError(ValueError):
    """Contract violation in telemetry samples, window bounds, or counters."""


def _numeric(values: pd.Series, name: str) -> np.ndarray:
    try:
        return pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise RegimeInputError(f"column {name!r} is not numeric: {exc}") from exc


def _check_finite_scalar(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RegimeInputError(f"{name}={value!r} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise RegimeInputError(f"{name}={value!r} must be finite")
    return value


def _check_min_samples(min_samples: int) -> int:
    if isinstance(min_samples, bool) or not isinstance(min_samples, int):
        raise RegimeInputError(f"min_samples={min_samples!r} must be an int")
    if min_samples < 1:
        raise RegimeInputError(f"min_samples={min_samples!r} must be >= 1")
    return min_samples


def _check_min_coverage(min_coverage: float) -> float:
    if isinstance(min_coverage, bool) or not isinstance(min_coverage, (int, float)):
        raise RegimeInputError(f"min_coverage={min_coverage!r} must be a number")
    min_coverage = float(min_coverage)
    if not math.isfinite(min_coverage) or not 0.0 < min_coverage <= 1.0:
        raise RegimeInputError(f"min_coverage={min_coverage!r} must be in (0, 1]")
    return min_coverage


def _bool_array(values: pd.Series, name: str) -> np.ndarray:
    arr = values.to_numpy()
    if arr.dtype == np.bool_:
        return arr
    num = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if np.isnan(num).any():
        raise RegimeInputError(
            f"column {name!r} contains NaN or non-boolean values — a row whose "
            "telemetry status is unknown cannot be treated as certified"
        )
    if not np.isin(num, (0.0, 1.0)).all():
        raise RegimeInputError(f"column {name!r} must be boolean / 0-1 valued")
    return num.astype(bool)


@dataclass(frozen=True)
class WindowRegimeInputs:
    """One window's §6.1 regime inputs, certified from telemetry.

    ``rho_kv_time_avg`` = zero-order-hold time-weighted mean of the occupancy
    gauge over the covered span; ``scarcity_events`` = cumulative-preemption
    delta (last − first in-window value); ``coverage`` = covered_time /
    window duration, where covered time runs from the first in-window sample
    to ``window_end_s`` (the pre-first-sample gap is never extrapolated).
    """

    rho_kv_time_avg: float
    scarcity_events: int
    n_samples: int
    coverage: float
    window_start_s: float
    window_end_s: float

    def to_flat_dict(self) -> dict[str, int | float]:
        """Flat mapping suitable as CSV columns (joins a CellSpec row key)."""
        return asdict(self)


def compute_window_regime_inputs(
    samples: pd.DataFrame,
    window_start_s: float,
    window_end_s: float,
    *,
    ts_col: str = "ts_s",
    kv_col: str = "kv_cache_usage",
    preempt_col: str = "preemptions_total",
    min_samples: int = 2,
    min_coverage: float = 0.8,
) -> WindowRegimeInputs:
    """Certify one window's (ρ_KV time-avg, scarcity_events) from telemetry.

    ``samples`` is the cage-stats time series: ``ts_col`` monotonic
    non-decreasing seconds on the SAME clock as the window bounds, ``kv_col``
    the KV-occupancy gauge (fraction in [0, 1]; None when the engine lacks
    the metric), ``preempt_col`` the CUMULATIVE preemption counter. Samples
    with ``window_start_s <= ts < window_end_s`` are in-window.

    ρ_KV is the zero-order-hold time-weighted mean: each in-window sample's
    value holds until the next sample's timestamp; the last holds until
    ``window_end_s``; the span before the first in-window sample is NOT
    covered (no backward extrapolation). The denominator is the covered time
    (first in-window ts .. window_end_s).

    Fail-closed (E2b): missing columns, non-monotonic or non-finite
    timestamps, bad bounds, fewer than ``min_samples`` in-window samples,
    coverage below ``min_coverage``, any None/NaN gauge or counter value,
    a non-integral or decreasing counter — each raises ``RegimeInputError``.
    Absence is not zero: none of these may fall through to a numeric that
    could label the window UNPRESSURED.
    """
    min_samples = _check_min_samples(min_samples)
    min_coverage = _check_min_coverage(min_coverage)
    window_start_s = _check_finite_scalar("window_start_s", window_start_s)
    window_end_s = _check_finite_scalar("window_end_s", window_end_s)
    if window_end_s <= window_start_s:
        raise RegimeInputError(
            f"window_end_s={window_end_s!r} must be > window_start_s="
            f"{window_start_s!r}"
        )
    missing = [c for c in (ts_col, kv_col, preempt_col) if c not in samples.columns]
    if missing:
        raise RegimeInputError(f"samples frame is missing required columns {missing}")

    ts = _numeric(samples[ts_col], ts_col)
    if not np.isfinite(ts).all():
        raise RegimeInputError(f"column {ts_col!r} contains non-finite timestamps")
    if (np.diff(ts) < 0.0).any():
        raise RegimeInputError(
            f"column {ts_col!r} must be monotonic non-decreasing (a time series)"
        )

    in_window = (ts >= window_start_s) & (ts < window_end_s)
    n_samples = int(in_window.sum())
    if n_samples < min_samples:
        raise RegimeInputError(
            f"window [{window_start_s:g}, {window_end_s:g}) has {n_samples} "
            f"in-window telemetry sample(s); need >= {min_samples} — too few "
            "samples cannot certify pressure"
        )
    ts_w = ts[in_window]

    kv = pd.to_numeric(samples[kv_col], errors="coerce").to_numpy(dtype=float)
    kv_w = kv[in_window]
    n_absent = int(np.isnan(kv_w).sum())
    if n_absent > 0:
        raise RegimeInputError(
            f"KV-occupancy gauge {kv_col!r} is None/NaN on {n_absent} in-window "
            "sample(s): absence is not zero — an engine that does not report "
            "the gauge cannot be certified in-regime (E2b)"
        )
    if ((kv_w < 0.0) | (kv_w > 1.0)).any():
        raise RegimeInputError(
            f"column {kv_col!r} is an occupancy fraction and must lie in [0, 1]"
        )

    duration = window_end_s - window_start_s
    covered = window_end_s - float(ts_w[0])
    coverage = covered / duration
    if coverage < min_coverage:
        raise RegimeInputError(
            f"telemetry coverage {coverage:.3f} of window "
            f"[{window_start_s:g}, {window_end_s:g}) is below min_coverage="
            f"{min_coverage:g} — a sparsely-sampled window cannot certify "
            "pressure"
        )
    # Zero-order hold: each sample holds until the next; the last holds to
    # window_end_s. Duplicate timestamps contribute zero-width segments.
    hold_until = np.append(ts_w[1:], window_end_s)
    rho_kv_time_avg = float(((hold_until - ts_w) * kv_w).sum() / covered)

    pre = pd.to_numeric(samples[preempt_col], errors="coerce").to_numpy(dtype=float)
    pre_w = pre[in_window]
    if np.isnan(pre_w).any():
        raise RegimeInputError(
            f"preemption counter {preempt_col!r} is None/NaN on in-window "
            "sample(s): absence is not zero — scarcity cannot be certified "
            "without the counter (E2b)"
        )
    if (pre_w != np.floor(pre_w)).any():
        raise RegimeInputError(
            f"column {preempt_col!r} is a cumulative event counter and must "
            "hold integer values"
        )
    if (pre_w < 0.0).any():
        raise RegimeInputError(
            f"column {preempt_col!r} is a cumulative counter and must be >= 0"
        )
    if (np.diff(pre_w) < 0.0).any():
        raise RegimeInputError(
            f"cumulative counter {preempt_col!r} decreased in-window (negative "
            "delta): the server restarted mid-window, which invalidates the "
            "window"
        )
    scarcity_events = int(pre_w[-1] - pre_w[0])

    return WindowRegimeInputs(
        rho_kv_time_avg=rho_kv_time_avg,
        scarcity_events=scarcity_events,
        n_samples=n_samples,
        coverage=float(coverage),
        window_start_s=window_start_s,
        window_end_s=window_end_s,
    )


def _bound_or_nan(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return math.nan
    return float(value)


def compute_regime_inputs(
    samples: pd.DataFrame,
    windows: Sequence[tuple[float, float]],
    *,
    allow_missing: bool = False,
    **kwargs: object,
) -> pd.DataFrame:
    """Per-window regime inputs: one row per (start, end) window.

    Columns = the ``WindowRegimeInputs`` fields plus ``telemetry_ok`` (bool)
    and ``refusal_reason`` (str | None). With ``allow_missing=False`` (the
    default) the first ``RegimeInputError`` re-raises — fail-closed. With
    ``allow_missing=True`` a refused window yields a row with NaN for
    ``rho_kv_time_avg``/``scarcity_events``/``n_samples``/``coverage``,
    ``telemetry_ok=False``, and the refusal message in ``refusal_reason``
    (never a numeric sentinel — absence is not zero); certified rows carry
    ``telemetry_ok=True`` and ``refusal_reason=None``. Downstream,
    ``label_regime_with_refusal`` maps refused rows to ``REGIME_UNKNOWN``.
    """
    if len(windows) == 0:
        raise RegimeInputError("windows is empty: nothing to certify")
    rows: list[dict[str, object]] = []
    for window in windows:
        if len(window) != 2:
            raise RegimeInputError(
                f"each window must be a (start, end) pair; got {window!r}"
            )
        start, end = window
        try:
            inputs = compute_window_regime_inputs(samples, start, end, **kwargs)
        except RegimeInputError as exc:
            if not allow_missing:
                raise
            rows.append(
                {
                    "rho_kv_time_avg": math.nan,
                    "scarcity_events": math.nan,
                    "n_samples": math.nan,
                    "coverage": math.nan,
                    "window_start_s": _bound_or_nan(start),
                    "window_end_s": _bound_or_nan(end),
                    "telemetry_ok": False,
                    "refusal_reason": str(exc),
                }
            )
        else:
            rows.append(
                {
                    **inputs.to_flat_dict(),
                    "telemetry_ok": True,
                    "refusal_reason": None,
                }
            )
    columns = [
        "rho_kv_time_avg",
        "scarcity_events",
        "n_samples",
        "coverage",
        "window_start_s",
        "window_end_s",
        "telemetry_ok",
        "refusal_reason",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    # The DataFrame constructor coerces None -> NaN; the contract says
    # certified rows carry refusal_reason=None (a reason exists ONLY on
    # refused rows), so rebuild the column as explicit objects.
    frame["refusal_reason"] = pd.Series(
        [row["refusal_reason"] for row in rows], dtype=object, index=frame.index
    )
    return frame


def label_regime_with_refusal(
    cells: pd.DataFrame,
    *,
    rho_col: str = "rho_kv_time_avg",
    events_col: str = "scarcity_events",
    attainment_col: str = "attainment",
    ok_col: str = "telemetry_ok",
) -> pd.Series:
    """§6.1 regime labels with the operational UNKNOWN refusal lane.

    Rows with ``telemetry_ok`` True get their label from
    ``goodput.label_regime`` (the ONE source of the §6.1 thresholds — no
    duplication here); rows with ``telemetry_ok`` False get
    ``REGIME_UNKNOWN``, which is outside the 3-label grid vocabulary and
    never enters in-regime aggregates. When ``ok_col`` is absent every row
    is treated as certified, delegating entirely to goodput's own
    validation. Returns a 'regime' Series aligned to ``cells.index``.
    """
    if ok_col not in cells.columns:
        return _goodput_label_regime(
            cells,
            rho_col=rho_col,
            events_col=events_col,
            attainment_col=attainment_col,
        )
    ok = _bool_array(cells[ok_col], ok_col)
    labels = pd.Series(REGIME_UNKNOWN, index=cells.index, name="regime", dtype=object)
    if ok.any():
        labels.loc[ok] = _goodput_label_regime(
            cells.loc[ok],
            rho_col=rho_col,
            events_col=events_col,
            attainment_col=attainment_col,
        )
    return labels
