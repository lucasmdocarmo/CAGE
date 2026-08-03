"""Serving-yield (Y) window metrics, knee/cliff onset estimators, §6.1 regime
labels, and the Rogan-Gladen misclassification correction (audit gap P0-3).

Charter bindings (PUBLICATION.md):
- §6.1 chassis: relative primary SLO pair — TTFT ≤ 10× and TPOT ≤ 5× the same
  model×engine single-stream baseline; completed-only goodput; knee =
  Chiu-Jain power-metric maximum; cliff = retrograde goodput; the 3-layer
  in-regime criterion (ρ_KV time-avg ≥ 0.9, scarcity counters > 0,
  attainment ≥ 90%).
- S1: serving yield Y = timely AND veridical per request, with the
  independence null G·E[v] and the covariance gap Cov(timely, veridical)
  printed beside every Y (clause b), and the truth tax G − Y (§9.2 estimand
  variable). Y is reported raw AND Rogan-Gladen-corrected.
- §9.2: onset misses at grid resolution get the pre-registered label
  INCONCLUSIVE_AT_RESOLUTION (multiplicative ×/÷1.15 band) — labeled, never
  guessed. Knee point estimate = interpolated Chiu-Jain argmax over the three
  nearest rate points.
- Audit F1 scale note: G and Y ship in BOTH named scales — fraction-of-issued
  (``*_frac``) and per-window rate (``*_rps``) — and one figure never mixes
  them. The per-GPU basis (§6.6b) is a downstream division by GPU count.

Non-completions are non-veridical by registration (audit §2.6): a row with
``veridical=True`` while ``ok=False`` violates the scoring contract and raises.
Domain logic only: stdlib + numpy/pandas, no I/O, no plotting.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

__all__ = [
    "ATTAINMENT_MIN",
    "GoodputError",
    "IN_REGIME",
    "OnsetEstimate",
    "OnsetKind",
    "OnsetLabel",
    "PAST_CLIFF",
    "RegimeLabel",
    "RHO_KV_MIN",
    "UNPRESSURED",
    "SLOBaseline",
    "TPOT_SLO_MULTIPLIER",
    "TTFT_SLO_MULTIPLIER",
    "WindowMetrics",
    "classify_regime",
    "corrected_rate",
    "evaluate_window",
    "find_cliff",
    "find_knee",
    "label_regime",
]

# §6.1 primary relative SLO pair (the ONLY pair inside Y; Sarathi secondaries
# per §6.3 are a separate gate — pass different multipliers/baseline for them).
TTFT_SLO_MULTIPLIER: float = 10.0
TPOT_SLO_MULTIPLIER: float = 5.0

# §6.1 in-regime thresholds.
RHO_KV_MIN: float = 0.9
ATTAINMENT_MIN: float = 0.9

# §9.2 multiplicative resolution band ×/÷1.15.
DEFAULT_RESOLUTION: float = 1.15

RegimeLabel = Literal["IN_REGIME", "UNPRESSURED", "PAST_CLIFF"]
# THE canonical §6.1 label vocabulary (2026-08-02 harmonization): every
# consumer (figure_pipeline included) imports THESE constants — the charter's
# prose spellings ("in-regime", "PAST-CLIFF") are never machine labels.
IN_REGIME: RegimeLabel = "IN_REGIME"
UNPRESSURED: RegimeLabel = "UNPRESSURED"
PAST_CLIFF: RegimeLabel = "PAST_CLIFF"
OnsetKind = Literal["knee", "cliff"]
OnsetLabel = Literal[
    "ESTIMATED", "INCONCLUSIVE_AT_RESOLUTION", "NOT_BRACKETED", "NOT_OBSERVED"
]

_WINDOW_COLUMNS: tuple[str, ...] = ("ttft_s", "tpot_s", "ok", "veridical")


class GoodputError(ValueError):
    """Contract violation in window records, sweep grids, or correction inputs."""


def _numeric(values: pd.Series, name: str) -> np.ndarray:
    try:
        return pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise GoodputError(f"column {name!r} is not numeric: {exc}") from exc


def _check_positive_scalar(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GoodputError(f"{name}={value!r} must be a number")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise GoodputError(f"{name}={value!r} must be finite and > 0")
    return value


@dataclass(frozen=True)
class SLOBaseline:
    """Single-stream latency floor per model×engine (§6.1: measured at r=1.5,
    concurrency 1). SLO fairness clause (S1 d): relative to each model's OWN
    floor, so model scale cannot bias Y."""

    ttft_s: float
    tpot_s: float

    def __post_init__(self) -> None:
        _check_positive_scalar("ttft_s", self.ttft_s)
        _check_positive_scalar("tpot_s", self.tpot_s)


@dataclass(frozen=True)
class WindowMetrics:
    """One measurement window's currencies: throughput → G → Y (the S1 ladder).

    ``*_rps`` = per-window rate (events/second); ``*_frac`` = fraction of
    issued requests. covariance_gap = Cov(timely, veridical) over issued
    requests = yield_frac − goodput_frac·veridical_frac (dimensionless);
    covariance_gap_rps is the same gap on the rate scale (Y − G·E[v]).
    """

    n_issued: int
    n_completed: int
    n_timely: int
    n_veridical: int
    n_yield: int
    duration_s: float
    attainment: float
    throughput_rps: float
    goodput_rps: float
    yield_rps: float
    goodput_frac: float
    yield_frac: float
    veridical_frac: float
    independence_null_rps: float
    independence_null_frac: float
    covariance_gap: float
    covariance_gap_rps: float
    truth_tax_rps: float
    truth_tax_frac: float

    def to_flat_dict(self) -> dict[str, int | float]:
        """Flat mapping suitable as CSV columns (joins a CellSpec row key)."""
        return asdict(self)


def _ok_array(values: pd.Series) -> np.ndarray:
    arr = values.to_numpy()
    if arr.dtype == np.bool_:
        return arr
    num = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if np.isnan(num).any():
        raise GoodputError("column 'ok' contains NaN or non-boolean values")
    if not np.isin(num, (0.0, 1.0)).all():
        raise GoodputError("column 'ok' must be boolean / 0-1 valued")
    return num.astype(bool)


def _veridical_array(values: pd.Series, ok: np.ndarray) -> np.ndarray:
    arr = values.to_numpy()
    if arr.dtype == np.bool_:
        verid = arr
    else:
        num = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        nan = np.isnan(num)
        if (nan & ok).any():
            raise GoodputError(
                "'veridical' is NaN on completed (ok) rows — the §8.5 predicate "
                "must be scored for every completion"
            )
        # Non-completions are non-veridical by registration (audit §2.6).
        num = np.where(nan, 0.0, num)
        if not np.isin(num, (0.0, 1.0)).all():
            raise GoodputError("column 'veridical' must be boolean / 0-1 valued")
        verid = num.astype(bool)
    if (verid & ~ok).any():
        raise GoodputError(
            "veridical=True on a non-completed request contradicts the Y "
            "predicate (non-completions are non-veridical, audit §2.6)"
        )
    return verid


def _latency_array(values: pd.Series, name: str, ok: np.ndarray) -> np.ndarray:
    arr = _numeric(values, name)
    bad = ok & (~np.isfinite(arr) | (arr < 0.0))
    if bad.any():
        raise GoodputError(
            f"column {name!r} must be finite and >= 0 on completed rows; "
            f"{int(bad.sum())} violation(s)"
        )
    # Failed requests may carry NaN latencies; they are never timely.
    return np.where(ok, arr, np.inf)


def _arrival_span(records: pd.DataFrame) -> float:
    if "arrival_s" not in records.columns:
        raise GoodputError(
            "duration_s not supplied and no 'arrival_s' column to derive it "
            "from (§6.1 windows are pre-costed — prefer an explicit duration; "
            "arrival_s = INTENDED open-loop arrival per §6.3)"
        )
    arr = _numeric(records["arrival_s"], "arrival_s")
    if not np.isfinite(arr).all():
        raise GoodputError("column 'arrival_s' contains non-finite timestamps")
    span = float(arr.max() - arr.min())
    if span <= 0.0:
        raise GoodputError(
            "derived window duration is not positive (need >= 2 distinct "
            "arrival timestamps, or pass duration_s explicitly)"
        )
    return span


def evaluate_window(
    records: pd.DataFrame,
    baseline: SLOBaseline,
    *,
    duration_s: float | None = None,
    ttft_multiplier: float = TTFT_SLO_MULTIPLIER,
    tpot_multiplier: float = TPOT_SLO_MULTIPLIER,
) -> WindowMetrics:
    """Compute the window currencies from per-request records (§6.1 + S1).

    Required columns: ``ttft_s``, ``tpot_s`` (seconds; may be NaN on failed
    rows), ``ok`` (request completed), ``veridical`` (§8.5 per-dataset
    predicate; NaN allowed only on non-completed rows). ``arrival_s``
    (intended open-loop arrival time, §6.3 coordinated-omission clocking) is
    required only when ``duration_s`` is None, in which case the window
    duration is the arrival span — a documented under-estimate; registered
    windows supply the pre-costed duration explicitly.

    A request is timely iff ok AND ttft_s ≤ ttft_multiplier·baseline.ttft_s
    AND tpot_s ≤ tpot_multiplier·baseline.tpot_s; it counts toward Y iff
    timely AND veridical. Default multipliers are the §6.1 primary pair — the
    only pair inside Y; pass the §6.3 Sarathi settings for the secondary gate.
    """
    if records.empty:
        raise GoodputError("empty window: no issued requests")
    missing = [name for name in _WINDOW_COLUMNS if name not in records.columns]
    if missing:
        raise GoodputError(f"window records missing required columns {missing}")
    ttft_multiplier = _check_positive_scalar("ttft_multiplier", ttft_multiplier)
    tpot_multiplier = _check_positive_scalar("tpot_multiplier", tpot_multiplier)

    ok = _ok_array(records["ok"])
    verid = _veridical_array(records["veridical"], ok)
    ttft = _latency_array(records["ttft_s"], "ttft_s", ok)
    tpot = _latency_array(records["tpot_s"], "tpot_s", ok)
    if duration_s is None:
        duration = _arrival_span(records)
    else:
        duration = _check_positive_scalar("duration_s", duration_s)

    timely = (
        ok
        & (ttft <= ttft_multiplier * baseline.ttft_s)
        & (tpot <= tpot_multiplier * baseline.tpot_s)
    )
    yielded = timely & verid

    n_issued = int(len(records))
    n_completed = int(ok.sum())
    n_timely = int(timely.sum())
    n_veridical = int(verid.sum())
    n_yield = int(yielded.sum())

    goodput_frac = n_timely / n_issued
    yield_frac = n_yield / n_issued
    veridical_frac = n_veridical / n_issued
    goodput_rps = n_timely / duration
    yield_rps = n_yield / duration
    independence_null_frac = goodput_frac * veridical_frac
    independence_null_rps = goodput_rps * veridical_frac

    return WindowMetrics(
        n_issued=n_issued,
        n_completed=n_completed,
        n_timely=n_timely,
        n_veridical=n_veridical,
        n_yield=n_yield,
        duration_s=duration,
        attainment=n_completed / n_issued,
        throughput_rps=n_completed / duration,
        goodput_rps=goodput_rps,
        yield_rps=yield_rps,
        goodput_frac=goodput_frac,
        yield_frac=yield_frac,
        veridical_frac=veridical_frac,
        independence_null_rps=independence_null_rps,
        independence_null_frac=independence_null_frac,
        covariance_gap=yield_frac - independence_null_frac,
        covariance_gap_rps=yield_rps - independence_null_rps,
        truth_tax_rps=goodput_rps - yield_rps,
        truth_tax_frac=goodput_frac - yield_frac,
    )


@dataclass(frozen=True)
class OnsetEstimate:
    """Knee/cliff onset over a rate sweep — a labeled outcome, never a guess.

    ``onset_rate`` is set only when label == "ESTIMATED" (knee: interpolated
    Chiu-Jain argmax; cliff: the first retrograde grid point). ``grid_rate``
    is the nearest measured grid point (knee discrete argmax / first
    retrograde rate). ``bracket`` is the (lo, hi) grid interval known to
    contain the onset. Labels: INCONCLUSIVE_AT_RESOLUTION — the bracket does
    not fit inside a ×/÷resolution band (§9.2); NOT_BRACKETED — knee argmax
    sits on a grid edge, so the maximum was never bracketed; NOT_OBSERVED —
    no retrograde goodput anywhere in the grid (cliff not crossed).
    """

    kind: OnsetKind
    label: OnsetLabel
    onset_rate: float | None
    grid_rate: float | None
    bracket: tuple[float, float] | None


def _check_resolution(resolution: float | None) -> None:
    if resolution is None:
        return
    if isinstance(resolution, bool) or not isinstance(resolution, (int, float)):
        raise GoodputError(f"resolution={resolution!r} must be a number or None")
    if not resolution > 1.0:
        raise GoodputError(
            f"resolution={resolution!r} must be > 1 (a multiplicative ×/÷ band, "
            f"e.g. {DEFAULT_RESOLUTION} per §9.2); pass None to disable"
        )


def _within_band(lo: float, hi: float, resolution: float | None) -> bool:
    # The onset is known only to its bracket; the bracket fits inside a
    # ×/÷resolution band iff hi/lo <= resolution**2. The relative tolerance
    # keeps the boundary case decidable: on the registered exact ×resolution
    # grid (§6.1) a two-step bracket has hi/lo == resolution**2 mathematically,
    # but float rounding lands it a few ulp above and would mislabel EVERY
    # knee INCONCLUSIVE_AT_RESOLUTION (caught in the 2026-08-02 P0 dry-run).
    if resolution is None:
        return True
    return hi / lo <= float(resolution) * float(resolution) * (1.0 + 1e-9)


def _sorted_sweep(
    sweep: pd.DataFrame, rate_col: str, value_cols: tuple[str, ...], min_rows: int
) -> pd.DataFrame:
    missing = [c for c in (rate_col, *value_cols) if c not in sweep.columns]
    if missing:
        raise GoodputError(f"sweep is missing required columns {missing}")
    if len(sweep) < min_rows:
        raise GoodputError(
            f"sweep has {len(sweep)} grid point(s); need >= {min_rows}"
        )
    frame = sweep.loc[:, [rate_col, *value_cols]].sort_values(rate_col)
    frame = frame.reset_index(drop=True)
    rates = _numeric(frame[rate_col], rate_col)
    if not np.isfinite(rates).all() or (rates <= 0.0).any():
        raise GoodputError(f"column {rate_col!r} must be finite and > 0")
    if (np.diff(rates) == 0.0).any():
        raise GoodputError(
            f"duplicate {rate_col!r} grid points — aggregate replications "
            "(§6.3 batch-means) before onset estimation"
        )
    return frame


def find_knee(
    sweep: pd.DataFrame,
    *,
    rate_col: str = "offered_rate",
    throughput_col: str = "throughput",
    latency_col: str = "latency",
    alpha: float = 1.0,
    resolution: float | None = DEFAULT_RESOLUTION,
) -> OnsetEstimate:
    """Knee = Chiu-Jain power-metric maximum (§6.1), interpolated (§9.2).

    Power = throughput**alpha / latency per grid point (alpha=1 is the classic
    Chiu-Jain metric; the "throughput/latency ratio family"). The point
    estimate is the vertex of the parabola through the three rate points
    nearest the discrete argmax — the audit's interpolated-argmax estimator —
    clamped to its bracket. Boundary argmax, plateaus, and brackets wider than
    the ×/÷resolution band return labels instead of estimates.
    """
    _check_resolution(resolution)
    alpha = _check_positive_scalar("alpha", alpha)
    frame = _sorted_sweep(sweep, rate_col, (throughput_col, latency_col), min_rows=3)
    rates = _numeric(frame[rate_col], rate_col)
    throughput = _numeric(frame[throughput_col], throughput_col)
    latency = _numeric(frame[latency_col], latency_col)
    if not np.isfinite(throughput).all() or (throughput < 0.0).any():
        raise GoodputError(f"column {throughput_col!r} must be finite and >= 0")
    if not np.isfinite(latency).all() or (latency <= 0.0).any():
        raise GoodputError(f"column {latency_col!r} must be finite and > 0")

    power = throughput**alpha / latency
    peak_idx = np.flatnonzero(power == power.max())
    if len(peak_idx) > 1:
        # Plateau: the argmax is not unique at this grid resolution.
        return OnsetEstimate(
            kind="knee",
            label="INCONCLUSIVE_AT_RESOLUTION",
            onset_rate=None,
            grid_rate=float(rates[peak_idx[0]]),
            bracket=None,
        )
    i = int(peak_idx[0])
    if i == 0 or i == len(rates) - 1:
        return OnsetEstimate(
            kind="knee",
            label="NOT_BRACKETED",
            onset_rate=None,
            grid_rate=float(rates[i]),
            bracket=None,
        )
    lo, hi = float(rates[i - 1]), float(rates[i + 1])
    if not _within_band(lo, hi, resolution):
        return OnsetEstimate(
            kind="knee",
            label="INCONCLUSIVE_AT_RESOLUTION",
            onset_rate=None,
            grid_rate=float(rates[i]),
            bracket=(lo, hi),
        )
    # Parabolic vertex through the three nearest points. With a strict
    # interior maximum (plateaus excluded above) the denominator is > 0 and
    # the vertex lies inside the bracket.
    dx1 = rates[i] - rates[i - 1]
    dx2 = rates[i] - rates[i + 1]
    dy1 = power[i] - power[i - 1]
    dy2 = power[i] - power[i + 1]
    vertex = rates[i] - 0.5 * (dx1 * dx1 * dy2 - dx2 * dx2 * dy1) / (
        dx1 * dy2 - dx2 * dy1
    )
    return OnsetEstimate(
        kind="knee",
        label="ESTIMATED",
        onset_rate=float(min(max(vertex, lo), hi)),
        grid_rate=float(rates[i]),
        bracket=(lo, hi),
    )


def find_cliff(
    sweep: pd.DataFrame,
    *,
    rate_col: str = "offered_rate",
    goodput_col: str = "goodput",
    resolution: float | None = DEFAULT_RESOLUTION,
) -> OnsetEstimate:
    """Cliff = first retrograde-goodput point: G strictly falls as offered
    rate rises (§6.1). The onset lies in the bracket (previous rate, first
    retrograde rate]; the point estimate is the first retrograde grid point
    when the bracket fits the ×/÷resolution band, else the §9.2 label.
    Rates above λ* with rising G are grid points, not failures → NOT_OBSERVED.
    Replication noise is the stats layer's job (§6.3): aggregate first.
    """
    _check_resolution(resolution)
    frame = _sorted_sweep(sweep, rate_col, (goodput_col,), min_rows=2)
    rates = _numeric(frame[rate_col], rate_col)
    goodput = _numeric(frame[goodput_col], goodput_col)
    if not np.isfinite(goodput).all() or (goodput < 0.0).any():
        raise GoodputError(f"column {goodput_col!r} must be finite and >= 0")

    drops = np.flatnonzero(np.diff(goodput) < 0.0)
    if len(drops) == 0:
        return OnsetEstimate(
            kind="cliff",
            label="NOT_OBSERVED",
            onset_rate=None,
            grid_rate=None,
            bracket=None,
        )
    i = int(drops[0]) + 1
    lo, hi = float(rates[i - 1]), float(rates[i])
    if not _within_band(lo, hi, resolution):
        return OnsetEstimate(
            kind="cliff",
            label="INCONCLUSIVE_AT_RESOLUTION",
            onset_rate=None,
            grid_rate=hi,
            bracket=(lo, hi),
        )
    return OnsetEstimate(
        kind="cliff", label="ESTIMATED", onset_rate=hi, grid_rate=hi, bracket=(lo, hi)
    )


def classify_regime(
    *, rho_kv: float, scarcity_events: int, attainment: float
) -> RegimeLabel:
    """§6.1 3-layer in-regime criterion for one cell.

    (a) ρ_KV time-avg ≥ 0.9 AND (b) scarcity counters > 0 AND (c) attainment
    ≥ 0.9 → IN_REGIME; failing (a) or (b) → UNPRESSURED; failing (c) →
    PAST_CLIFF. Pinned tie-break: (c) wins on joint failure — a completion
    collapse is diagnostic regardless of occupancy. All three labels remain
    valid grid points; only IN_REGIME enters in-regime aggregates.
    """
    for name, value in (("rho_kv", rho_kv), ("attainment", attainment)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise GoodputError(f"{name}={value!r} must be a number")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise GoodputError(f"{name}={value!r} must be finite and >= 0")
    if float(attainment) > 1.0:
        raise GoodputError(f"attainment={attainment!r} must be within [0, 1]")
    if (
        isinstance(scarcity_events, bool)
        or not isinstance(scarcity_events, (int, float))
        or not float(scarcity_events).is_integer()
        or scarcity_events < 0
    ):
        raise GoodputError(
            f"scarcity_events={scarcity_events!r} must be a non-negative integer count"
        )
    if float(attainment) < ATTAINMENT_MIN:
        return PAST_CLIFF
    if float(rho_kv) < RHO_KV_MIN or int(scarcity_events) == 0:
        return UNPRESSURED
    return IN_REGIME


def label_regime(
    cells: pd.DataFrame,
    *,
    rho_col: str = "rho_kv",
    events_col: str = "scarcity_events",
    attainment_col: str = "attainment",
) -> pd.Series:
    """Vectorized ``classify_regime`` over per-cell rows; returns a 'regime'
    Series aligned to ``cells.index``."""
    missing = [c for c in (rho_col, events_col, attainment_col) if c not in cells.columns]
    if missing:
        raise GoodputError(f"cells frame is missing required columns {missing}")
    rho = _numeric(cells[rho_col], rho_col)
    events = _numeric(cells[events_col], events_col)
    attainment = _numeric(cells[attainment_col], attainment_col)
    if not np.isfinite(rho).all() or (rho < 0.0).any():
        raise GoodputError(f"column {rho_col!r} must be finite and >= 0")
    if not np.isfinite(attainment).all() or ((attainment < 0.0) | (attainment > 1.0)).any():
        raise GoodputError(f"column {attainment_col!r} must be within [0, 1]")
    if (
        not np.isfinite(events).all()
        or (events < 0.0).any()
        or (events != np.floor(events)).any()
    ):
        raise GoodputError(
            f"column {events_col!r} must hold non-negative integer counts"
        )
    labels = np.select(
        [attainment < ATTAINMENT_MIN, (rho < RHO_KV_MIN) | (events == 0.0)],
        [PAST_CLIFF, UNPRESSURED],
        default=IN_REGIME,
    )
    return pd.Series(labels, index=cells.index, name="regime")


def corrected_rate(apparent: float, sensitivity: float, specificity: float) -> float:
    """Rogan-Gladen-corrected true rate from an apparent (instrument-measured)
    rate: (apparent + sp − 1) / (se + sp − 1), truncated to [0, 1] (the
    standard truncated estimator). se/sp come from the §8.6 gold set; Y is
    reported raw AND corrected (S1). Raises when any input leaves [0, 1] or
    the instrument is uninformative (Youden's J = se + sp − 1 ≤ 0)."""
    for name, value in (
        ("apparent", apparent),
        ("sensitivity", sensitivity),
        ("specificity", specificity),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise GoodputError(f"{name}={value!r} must be a number")
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise GoodputError(f"{name}={value!r} must be within [0, 1]")
    youden = float(sensitivity) + float(specificity) - 1.0
    if youden <= 0.0:
        raise GoodputError(
            f"uninformative instrument: sensitivity + specificity = "
            f"{float(sensitivity) + float(specificity):g} <= 1 (Youden's J <= 0)"
        )
    raw = (float(apparent) + float(specificity) - 1.0) / youden
    return min(1.0, max(0.0, raw))
