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
- ADR-0115 (backlog A3, proposed): the corrected Y is RECOMPOSED, never
  corrected as a conjunction. The instrument misclassifies only the
  predicate half of Y; "timely" is a clock measurement. Y_corrected =
  P(timely) x RG(P(predicate | timely)); ``corrected_yield`` is the
  registered estimator, ``corrected_rate`` stays the scalar primitive, and
  ``reweight_gold_sample`` turns the verdict-stratified gold sample (§8.6(c),
  ADR-0109) into population sensitivity/specificity.
- §9.2: onset misses at grid resolution get the pre-registered label
  INCONCLUSIVE_AT_RESOLUTION (multiplicative ×/÷1.15 band) — labeled, never
  guessed. Knee point estimate = interpolated Chiu-Jain argmax over the three
  nearest rate points.
- Audit F1 scale note: G and Y ship in BOTH named scales — fraction-of-issued
  (``*_frac``) and per-window rate (``*_rps``) — and one figure never mixes
  them.
- §6.6 iso-basis machinery: distributed-vs-pressured comparisons run on one of
  TWO pre-registered bases, never mixed in one figure — (a) iso-aggregate-bytes
  (mechanism question; raw aggregate G/Y), and (b) per-GPU goodput =
  completed-only goodput / GPU count (deployment question; transfer/protocol
  contrasts #18/#19 ALWAYS report basis b). ``evaluate_window(gpu_count=...)``
  computes the basis-(b) division here, ``WindowMetrics.bases`` labels which
  basis every number belongs to, and ``assert_single_basis`` is the seam
  figure code calls before pooling.

Non-completions are non-veridical by registration (audit §2.6): a row with
``veridical=True`` while ``ok=False`` violates the scoring contract and raises.
Domain logic only: stdlib + numpy/pandas, no I/O, no plotting.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

__all__ = [
    "ATTAINMENT_MIN",
    "BASIS_AGGREGATE",
    "BASIS_PER_GPU",
    "BasisLabel",
    "BasisRecord",
    "CORRECTED_YIELD_ASSUMPTION",
    "CORRECTED_YIELD_ESTIMATOR",
    "CorrectedYield",
    "GoldStratum",
    "GoodputError",
    "InstrumentAccuracy",
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
    "WINDOW_BASES",
    "WindowMetrics",
    "assert_single_basis",
    "classify_regime",
    "corrected_rate",
    "corrected_yield",
    "corrected_yield_from_flags",
    "corrected_yield_from_window",
    "evaluate_window",
    "find_cliff",
    "find_knee",
    "label_regime",
    "reweight_gold_sample",
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

BasisLabel = Literal["aggregate", "per-gpu"]
# THE §6.6 basis vocabulary (charter): two pre-registered bases for
# distributed-vs-pressured comparison, never mixed in one figure. These are
# machine labels — figure code imports THESE constants, never respells them
# (the regime-label 2026-08-02 harmonization lesson applied to bases).
BASIS_AGGREGATE: BasisLabel = "aggregate"  # §6.6a iso-aggregate-bytes: raw G/Y
BASIS_PER_GPU: BasisLabel = "per-gpu"  # §6.6b deployment: G/gpu_count, Y/gpu_count

# Which WindowMetrics fields live on which basis. The per-GPU basis is a RATE
# normalization only: *_frac numbers are fractions of issued requests, a
# dimensionless quantity with no per-GPU meaning, so they are deliberately
# absent from the per-gpu tuple (dividing a fraction by GPU count would be the
# silent-fabrication bug class this module refuses).
_AGGREGATE_FIELDS: tuple[str, ...] = (
    "goodput_rps",
    "yield_rps",
    "goodput_frac",
    "yield_frac",
)
_PER_GPU_FIELDS: tuple[str, ...] = ("goodput_per_gpu", "yield_per_gpu")


@dataclass(frozen=True)
class BasisRecord:
    """Immutable §6.6 basis declaration riding on every ``WindowMetrics``:
    names the fields on each basis so no downstream consumer has to guess
    (raw G/Y are aggregate; ``*_per_gpu`` fields are basis b). Defaults ARE
    the canonical labeling; ``assert_single_basis`` refuses any record that
    disagrees (a tampered/foreign record means the numbers can't be trusted
    to the declared basis)."""

    aggregate: tuple[str, ...] = _AGGREGATE_FIELDS
    per_gpu: tuple[str, ...] = _PER_GPU_FIELDS


#: Canonical record instance; evaluate_window stamps this on every window.
WINDOW_BASES = BasisRecord()

_BASIS_FIELDS: dict[str, tuple[str, ...]] = {
    BASIS_AGGREGATE: _AGGREGATE_FIELDS,
    BASIS_PER_GPU: _PER_GPU_FIELDS,
}

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


def _check_gpu_count(value: object) -> int:
    # Strict integer, not "integral number": 8.0 is refused because a float
    # GPU count is always an upstream bookkeeping bug, and bool is refused
    # because True==1 would silently pass as a 1-GPU cell. np.bool_ is the
    # same bug arriving via pandas (it is NOT a bool subclass, yet implements
    # __index__), so it is refused by name. operator.index then accepts int
    # and numpy integers (pandas-sourced counts).
    if isinstance(value, (bool, np.bool_)):
        raise GoodputError(f"gpu_count={value!r} must be an integer >= 1")
    try:
        count = operator.index(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise GoodputError(f"gpu_count={value!r} must be an integer >= 1") from exc
    if count < 1:
        raise GoodputError(f"gpu_count={count!r} must be an integer >= 1")
    return int(count)


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

    §6.6 iso-basis fields: ``gpu_count`` is the number of GPUs serving the
    window; ``goodput_per_gpu`` = goodput_rps / gpu_count and
    ``yield_per_gpu`` = yield_rps / gpu_count are the §6.6b deployment basis
    (rate scale only — see the basis-vocabulary note). Raw G/Y stay AGGREGATE
    across all GPUs regardless of gpu_count; ``bases`` declares this labeling
    so pooled figures can be audited (``assert_single_basis``). All fields are
    required at construction: a hand-built WindowMetrics must state its basis
    facts explicitly rather than inherit a silent 1-GPU default.
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
    gpu_count: int
    goodput_per_gpu: float
    yield_per_gpu: float
    bases: BasisRecord

    def to_flat_dict(self) -> dict[str, int | float | dict[str, tuple[str, ...]]]:
        """One key per field, for JSON serialization (joins a CellSpec row
        key). NOT flat-CSV-safe: ``bases`` is a nested mapping (tuple values;
        JSON turns them into lists — ``assert_single_basis`` accepts both), so
        a CSV writer must flatten or drop it explicitly (verifier minor)."""
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
    gpu_count: int = 1,
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

    ``gpu_count`` (strict integer >= 1) is the number of GPUs serving this
    window; it feeds ONLY the §6.6b per-GPU fields — every aggregate currency
    is computed exactly as for a single GPU, so gpu_count=1 callers see
    byte-identical values.
    """
    if records.empty:
        raise GoodputError("empty window: no issued requests")
    missing = [name for name in _WINDOW_COLUMNS if name not in records.columns]
    if missing:
        raise GoodputError(f"window records missing required columns {missing}")
    ttft_multiplier = _check_positive_scalar("ttft_multiplier", ttft_multiplier)
    tpot_multiplier = _check_positive_scalar("tpot_multiplier", tpot_multiplier)
    gpu_count = _check_gpu_count(gpu_count)

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
        gpu_count=gpu_count,
        # §6.6b: IEEE division by 1 is exact, so gpu_count=1 per-GPU values are
        # byte-identical to the aggregates (the compatibility guarantee).
        goodput_per_gpu=goodput_rps / gpu_count,
        yield_per_gpu=yield_rps / gpu_count,
        bases=WINDOW_BASES,
    )


def _basis_item_value(item: object, name: str, idx: int) -> object:
    if isinstance(item, Mapping):
        if name in item:
            return item[name]
    else:
        marker = object()
        value = getattr(item, name, marker)
        if value is not marker:
            return value
    raise GoodputError(
        f"metrics_list[{idx}] carries no {name!r} — unlabeled metrics are "
        "never pooled (§6.6: every pooled number must declare its basis; "
        "legacy records must be rebuilt, not defaulted)"
    )


def _basis_item_number(item: object, name: str, idx: int) -> float:
    value = _basis_item_value(item, name, idx)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GoodputError(
            f"metrics_list[{idx}].{name}={value!r} must be a number"
        )
    value = float(value)
    if not math.isfinite(value):
        raise GoodputError(
            f"metrics_list[{idx}].{name}={value!r} must be finite"
        )
    return value


def _basis_item_record(item: object, idx: int) -> None:
    """Refuse any declared bases record that disagrees with the canonical
    §6.6 labeling — a foreign/tampered record means the numbers cannot be
    trusted to sit on the basis their field names claim."""
    declared = _basis_item_value(item, "bases", idx)
    if isinstance(declared, BasisRecord):
        agg, per = declared.aggregate, declared.per_gpu
    elif isinstance(declared, Mapping):
        try:
            agg, per = declared["aggregate"], declared["per_gpu"]
        except KeyError as exc:
            raise GoodputError(
                f"metrics_list[{idx}].bases is missing key {exc} — expected "
                "the BasisRecord mapping form with 'aggregate' and 'per_gpu'"
            ) from exc
    else:
        raise GoodputError(
            f"metrics_list[{idx}].bases={declared!r} is not a BasisRecord or "
            "its mapping form"
        )
    try:
        # JSON round-trips tuples as lists; normalize before comparing.
        agg_t = tuple(str(f) for f in agg)
        per_t = tuple(str(f) for f in per)
    except TypeError as exc:
        raise GoodputError(
            f"metrics_list[{idx}].bases holds non-sequence field lists"
        ) from exc
    if agg_t != _AGGREGATE_FIELDS or per_t != _PER_GPU_FIELDS:
        raise GoodputError(
            f"metrics_list[{idx}] declares a different basis labeling "
            f"(aggregate={agg_t!r}, per_gpu={per_t!r}) than the canonical "
            f"§6.6 record — refusing to pool metrics with mixed declared bases"
        )


def assert_single_basis(
    metrics_list: Iterable[object], basis: str
) -> tuple[str, ...]:
    """Refuse to pool window metrics unless EVERY item sits on one §6.6 basis.

    The seam figure code calls (T6.2) before pooling/plotting: ``basis`` is
    the single basis the figure declares (``BASIS_AGGREGATE`` = §6.6a
    iso-aggregate-bytes, ``BASIS_PER_GPU`` = §6.6b deployment; transfer and
    protocol contrasts #18/#19 always declare basis b). Items may be
    ``WindowMetrics`` instances or their ``to_flat_dict``/JSON mapping form.

    Raises ``GoodputError`` when: ``basis`` is not a §6.6 label; the list is
    empty (pooling zero windows is a contract violation, not a no-op); any
    item lacks the ``bases``/``gpu_count`` declaration; any item lacks ANY
    field of the requested basis, or holds a non-number/non-finite value
    there (the ``*_frac`` fields have no arithmetic invariant but presence
    and finiteness are still enforced — the returned names must all be
    readable); any item declares a labeling different from the canonical
    record; or any item's values are internally mixed-basis — its per-GPU
    numbers do not equal aggregate/gpu_count, the signature of a record
    assembled from two bases (e.g. a forgotten division on a gpu_count>1
    window, detectable because the two bases only coincide at gpu_count=1).
    The rate-pair invariant is audited regardless of the requested basis, so
    a mixed-basis record cannot slip into an aggregate-basis pool either.

    Returns the tuple of WindowMetrics field names on the requested basis, so
    callers read exactly the audited columns instead of respelling them.
    The audit itself is plain-Python attribute/mapping access: items need not
    be live dataclasses — JSON-loaded dict rows are first-class inputs. (The
    hosting module does import numpy/pandas at top level, so importing this
    seam is not dependency-free; only the per-item handling is.)
    """
    if basis not in _BASIS_FIELDS:
        raise GoodputError(
            f"basis={basis!r} is not a §6.6 basis label; expected "
            f"{BASIS_AGGREGATE!r} (§6.6a iso-aggregate-bytes) or "
            f"{BASIS_PER_GPU!r} (§6.6b per-GPU)"
        )
    items = list(metrics_list)
    if not items:
        raise GoodputError(
            "metrics_list is empty — pooling zero windows is a contract "
            "violation, not a no-op (fail-closed)"
        )
    for idx, item in enumerate(items):
        _basis_item_record(item, idx)
        gpu_count = _check_gpu_count(_basis_item_value(item, "gpu_count", idx))
        # The returned tuple is exactly what callers will read: every field on
        # the requested basis must exist and be a finite number BEFORE its name
        # is handed out. The *_frac fields carry no arithmetic invariant
        # (dimensionless, deliberately absent from basis b) but presence and
        # finiteness still apply — a missing or NaN column reaching a figure
        # is the silent-fabrication bug class this seam exists to refuse.
        for name in _BASIS_FIELDS[basis]:
            _basis_item_number(item, name, idx)
        for agg_name, per_name in (
            ("goodput_rps", "goodput_per_gpu"),
            ("yield_rps", "yield_per_gpu"),
        ):
            agg = _basis_item_number(item, agg_name, idx)
            per = _basis_item_number(item, per_name, idx)
            # rel_tol absorbs float round-trips (CSV/JSON); at gpu_count=1 the
            # bases coincide, so only gpu_count>1 windows can actually trip.
            if not math.isclose(per * gpu_count, agg, rel_tol=1e-9, abs_tol=1e-12):
                raise GoodputError(
                    f"metrics_list[{idx}] mixes bases: {per_name}={per!r} × "
                    f"gpu_count={gpu_count} != {agg_name}={agg!r} — per-GPU "
                    "and aggregate values disagree, so this record was not "
                    "produced on a single basis (§6.6: never pooled)"
                )
    return _BASIS_FIELDS[basis]


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


def _check_unit_interval(name: str, value: object) -> float:
    """Typed [0, 1] guard shared by the correction estimators (bool refused:
    True would silently pass as a rate of 1)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GoodputError(f"{name}={value!r} must be a number")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise GoodputError(f"{name}={value!r} must be within [0, 1]")
    return value


def _youden_j(sensitivity: float, specificity: float) -> float:
    """Youden's J = se + sp - 1; refuses an uninformative instrument (J <= 0),
    for which the Rogan-Gladen denominator is zero or the correction flips
    sign."""
    youden = sensitivity + specificity - 1.0
    if youden <= 0.0:
        raise GoodputError(
            f"uninformative instrument: sensitivity + specificity = "
            f"{sensitivity + specificity:g} <= 1 (Youden's J <= 0)"
        )
    return youden


def corrected_rate(apparent: float, sensitivity: float, specificity: float) -> float:
    """Rogan-Gladen-corrected true rate from an apparent (instrument-measured)
    rate: (apparent + sp − 1) / (se + sp − 1), truncated to [0, 1] (the
    standard truncated estimator). se/sp come from the §8.6 gold set. Raises
    when any input leaves [0, 1] or the instrument is uninformative (Youden's
    J = se + sp − 1 ≤ 0).

    This is the SCALAR PRIMITIVE. It corrects whatever rate it is handed, so
    it must only ever be handed a rate the instrument actually measured: the
    predicate rate among SLO-met requests. Handing it Y (the conjunction
    timely AND predicate) corrects the clock half too and is the backlog A3
    defect; the registered Y estimator is ``corrected_yield`` (ADR-0115)."""
    apparent = _check_unit_interval("apparent", apparent)
    sensitivity = _check_unit_interval("sensitivity", sensitivity)
    specificity = _check_unit_interval("specificity", specificity)
    youden = _youden_j(sensitivity, specificity)
    raw = (apparent + specificity - 1.0) / youden
    return min(1.0, max(0.0, raw))


# --------------------------------------------------------------------------- #
# ADR-0115 (backlog A3): corrected serving yield by recomposition
# --------------------------------------------------------------------------- #

#: Estimator identity stamped on every ``CorrectedYield`` and
#: ``InstrumentAccuracy`` record (ADR-0115, proposed 2026-09-17, backlog
#: Tier A item A3). A record without this stamp was not produced by the
#: registered estimator and must not be reported as the corrected Y.
CORRECTED_YIELD_ESTIMATOR: str = (
    "ADR-0115 recomposed Rogan-Gladen: Y_corrected = P(timely) x "
    "RG(P(predicate | timely); se, sp)"
)

#: The identifying assumption of ADR-0115, carried verbatim on every record so
#: the caveat travels with the number: NO DIFFERENTIAL MISCLASSIFICATION. The
#: instrument's sensitivity and specificity are taken as the same in every
#: arm (so an arm contrast on corrected Y is not an artifact of arm-specific
#: instrument error) and independent of timeliness (so the gold-set se/sp,
#: estimated without conditioning on the clock, apply to the SLO-met subset).
#: Both halves are checkable on the gold set (arm is a stratum per ADR-0109;
#: timeliness is recorded per gold item) and the check is the pre-registered
#: diagnostic, never a silent default.
CORRECTED_YIELD_ASSUMPTION: str = (
    "no differential misclassification: instrument sensitivity and "
    "specificity are the same in every arm and independent of timeliness "
    "(ADR-0115)"
)


@dataclass(frozen=True)
class CorrectedYield:
    """The ADR-0115 corrected serving yield with every input beside it.

    ``slo_rate`` = P(timely), a clock measurement, never corrected.
    ``apparent_predicate_rate_given_slo`` = instrument-positive fraction
    AMONG SLO-met requests. ``corrected_predicate_rate_given_slo`` = its
    Rogan-Gladen correction, truncated to [0, 1] (``truncated`` says whether
    the truncation bit; the untruncated value is kept for the audit).
    ``yield_raw`` = slo_rate x apparent; ``yield_corrected`` = slo_rate x
    corrected. ``n_issued``/``n_slo_met`` are the counts when the record was
    built from flags or a window, else 0 (a scalar-rate call has no counts;
    0 is the honest "unknown count", not a fabricated sample size).
    ``assumption`` and ``estimator`` are the module constants, stamped so the
    caveat and the estimator identity ride with the number.
    """

    slo_rate: float
    apparent_predicate_rate_given_slo: float
    sensitivity: float
    specificity: float
    youden_j: float
    corrected_predicate_rate_given_slo_untruncated: float
    corrected_predicate_rate_given_slo: float
    truncated: bool
    yield_raw: float
    yield_corrected: float
    n_issued: int
    n_slo_met: int
    assumption: str
    estimator: str

    def to_flat_dict(self) -> dict[str, int | float | bool | str]:
        """One key per field, for JSON serialization beside a WindowMetrics."""
        return asdict(self)


def corrected_yield(
    *,
    slo_rate: float,
    apparent_predicate_rate_given_slo: float,
    sensitivity: float,
    specificity: float,
    n_issued: int = 0,
    n_slo_met: int = 0,
) -> CorrectedYield:
    """Registered corrected-Y estimator (ADR-0115, backlog A3).

    Y = timely AND predicate. The quality instrument (§8.5 predicate at the
    registered tau) misclassifies ONLY the predicate; "timely" is measured by
    the clock. So the correction is applied to the predicate rate CONDITIONAL
    on SLO-met, and Y is recomposed::

        p_hat   = apparent_predicate_rate_given_slo
        p_corr  = clip((p_hat + sp - 1) / (se + sp - 1), 0, 1)
        Y_corr  = slo_rate x p_corr

    Correcting the conjunction directly, RG(slo_rate x p_hat), treats the
    clock's misses as instrument errors and is wrong whenever slo_rate < 1
    (the A3 defect). Identity: with se = sp = 1, Y_corr = slo_rate x p_hat =
    raw Y.

    Assumption (``CORRECTED_YIELD_ASSUMPTION``): no differential
    misclassification, i.e. se/sp are the same across arms and independent
    of timeliness, so gold-set se/sp estimated on the whole sample apply to
    the SLO-met subset of every arm. The record carries the assumption text.

    se/sp come from the §8.6(c) gold set via ``reweight_gold_sample``
    (verdict-stratified sample reweighted to population verdict shares).
    Refuses inputs outside [0, 1], an uninformative instrument (Youden's
    J <= 0), and negative or non-integer counts.
    """
    slo = _check_unit_interval("slo_rate", slo_rate)
    apparent = _check_unit_interval(
        "apparent_predicate_rate_given_slo", apparent_predicate_rate_given_slo
    )
    se = _check_unit_interval("sensitivity", sensitivity)
    sp = _check_unit_interval("specificity", specificity)
    youden = _youden_j(se, sp)
    for name, value in (("n_issued", n_issued), ("n_slo_met", n_slo_met)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise GoodputError(f"{name}={value!r} must be a non-negative integer")
        if int(value) < 0:
            raise GoodputError(f"{name}={value!r} must be a non-negative integer")
    if int(n_slo_met) > int(n_issued):
        raise GoodputError(
            f"n_slo_met={int(n_slo_met)} exceeds n_issued={int(n_issued)}"
        )
    untruncated = (apparent + sp - 1.0) / youden
    corrected = min(1.0, max(0.0, untruncated))
    return CorrectedYield(
        slo_rate=slo,
        apparent_predicate_rate_given_slo=apparent,
        sensitivity=se,
        specificity=sp,
        youden_j=youden,
        corrected_predicate_rate_given_slo_untruncated=untruncated,
        corrected_predicate_rate_given_slo=corrected,
        truncated=corrected != untruncated,
        yield_raw=slo * apparent,
        yield_corrected=slo * corrected,
        n_issued=int(n_issued),
        n_slo_met=int(n_slo_met),
        assumption=CORRECTED_YIELD_ASSUMPTION,
        estimator=CORRECTED_YIELD_ESTIMATOR,
    )


def _flag_array(values: Sequence[bool] | np.ndarray | pd.Series, name: str) -> np.ndarray:
    """Per-request boolean flags: bool arrays pass, 0/1 numerics are accepted,
    NaN and any other value are refused (a NaN flag is an unscored request,
    never a False)."""
    series = values if isinstance(values, pd.Series) else pd.Series(list(np.asarray(values).ravel()))
    arr = series.to_numpy()
    if arr.dtype == np.bool_:
        return arr
    num = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if np.isnan(num).any():
        raise GoodputError(f"column {name!r} contains NaN or non-boolean values")
    if not np.isin(num, (0.0, 1.0)).all():
        raise GoodputError(f"column {name!r} must be boolean / 0-1 valued")
    return num.astype(bool)


def corrected_yield_from_flags(
    *,
    timely: Sequence[bool] | np.ndarray | pd.Series,
    veridical: Sequence[bool] | np.ndarray | pd.Series,
    sensitivity: float,
    specificity: float,
) -> CorrectedYield:
    """``corrected_yield`` from per-request flags: ``timely`` (SLO-met by the
    clock) and ``veridical`` (the instrument's verdict on the §8.5 predicate).
    slo_rate = mean(timely); the apparent conditional rate =
    mean(veridical[timely]). Verdicts on untimely requests never enter the
    correction. Refuses empty or length-mismatched inputs and a window with
    no SLO-met request (the conditional rate is 0/0; such a window's Y is 0
    raw and needs no correction, so the caller reports it raw)."""
    t = _flag_array(timely, "timely")
    v = _flag_array(veridical, "veridical")
    if t.size == 0:
        raise GoodputError("empty window: no issued requests")
    if t.shape != v.shape:
        raise GoodputError(
            f"timely/veridical length mismatch: {t.shape[0]} vs {v.shape[0]}"
        )
    n_issued = int(t.size)
    n_slo_met = int(t.sum())
    if n_slo_met == 0:
        raise GoodputError(
            "no SLO-met requests in the window: the predicate rate given SLO "
            "is undefined (0/0); Y is 0 raw and is reported raw (ADR-0115)"
        )
    apparent = float(v[t].sum()) / n_slo_met
    return corrected_yield(
        slo_rate=n_slo_met / n_issued,
        apparent_predicate_rate_given_slo=apparent,
        sensitivity=sensitivity,
        specificity=specificity,
        n_issued=n_issued,
        n_slo_met=n_slo_met,
    )


def corrected_yield_from_window(
    metrics: WindowMetrics, *, sensitivity: float, specificity: float
) -> CorrectedYield:
    """``corrected_yield`` from an ``evaluate_window`` record: slo_rate =
    n_timely / n_issued (= goodput_frac), apparent conditional rate =
    n_yield / n_timely. The route every Y reporter takes (S1: Y raw AND
    corrected). Refuses anything that is not a WindowMetrics and a window
    with n_timely = 0 (see ``corrected_yield_from_flags``)."""
    if not isinstance(metrics, WindowMetrics):
        raise GoodputError(
            f"corrected_yield_from_window needs a WindowMetrics, got "
            f"{type(metrics).__name__}"
        )
    if metrics.n_timely == 0:
        raise GoodputError(
            "no SLO-met requests in the window: the predicate rate given SLO "
            "is undefined (0/0); Y is 0 raw and is reported raw (ADR-0115)"
        )
    return corrected_yield(
        slo_rate=metrics.n_timely / metrics.n_issued,
        apparent_predicate_rate_given_slo=metrics.n_yield / metrics.n_timely,
        sensitivity=sensitivity,
        specificity=specificity,
        n_issued=metrics.n_issued,
        n_slo_met=metrics.n_timely,
    )


def _check_count(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise GoodputError(f"{name}={value!r} must be an integer >= {minimum}")
    if int(value) < minimum:
        raise GoodputError(f"{name}={value!r} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class GoldStratum:
    """One stratum of the verdict-stratified gold sample (§8.6(c), ADR-0109).

    ``verdict``: the instrument's verdict defining the stratum (True =
    predicate positive at the registered tau). ``dataset``: the optional
    second stratification key (None for a verdict-only design; a sample must
    be all-None or all-named). ``n_sampled``: gold-annotated items drawn from
    this stratum; ``n_gold_true``: how many the annotators judged predicate
    true. ``population_count``: how many instrument-scored requests in the
    population fall in this stratum (the reweighting target)."""

    verdict: bool
    dataset: str | None
    n_sampled: int
    n_gold_true: int
    population_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.verdict, bool):
            raise GoodputError(f"verdict={self.verdict!r} must be a bool")
        if self.dataset is not None and not isinstance(self.dataset, str):
            raise GoodputError(f"dataset={self.dataset!r} must be a str or None")
        n_sampled = _check_count("n_sampled", self.n_sampled, minimum=1)
        n_true = _check_count("n_gold_true", self.n_gold_true, minimum=0)
        _check_count("population_count", self.population_count, minimum=1)
        if n_true > n_sampled:
            raise GoodputError(
                f"n_gold_true={n_true} exceeds n_sampled={n_sampled}"
            )

    @property
    def key(self) -> tuple[bool, str | None]:
        return (self.verdict, self.dataset)

    @property
    def gold_true_fraction(self) -> float:
        return self.n_gold_true / self.n_sampled


@dataclass(frozen=True)
class InstrumentAccuracy:
    """Population sensitivity/specificity reweighted from a verdict-stratified
    gold sample (``reweight_gold_sample``), with the weights beside them.
    ``weights`` maps (verdict, dataset) to its population share; ``dataset``
    is the filter used (None = pooled over all strata)."""

    sensitivity: float
    specificity: float
    youden_j: float
    prevalence: float
    n_sampled: int
    n_population: int
    n_strata: int
    weights: dict[tuple[bool, str | None], float]
    dataset: str | None
    estimator: str


def reweight_gold_sample(
    strata: Sequence[GoldStratum], *, dataset: str | None = None
) -> InstrumentAccuracy:
    """Sensitivity/specificity for ``corrected_yield`` from a gold sample
    stratified by instrument verdict (optionally x dataset), reweighted to
    the POPULATION verdict shares (ADR-0115; the two-phase verification-bias
    correction of Begg and Greenes, 1983).

    A verdict-stratified sample oversamples one verdict, so pooling its rows
    as if they were the population biases se/sp. With strata h, population
    share w_h = population_count_h / sum population_count, and gold-true
    fraction p_h = n_gold_true_h / n_sampled_h (unbiased within a stratum
    because sampling was on the verdict, not on the truth)::

        pi  = sum_h w_h p_h                          (prevalence)
        se  = sum_{h: verdict+} w_h p_h / pi
        sp  = sum_{h: verdict-} w_h (1 - p_h) / (1 - pi)

    ``dataset`` restricts the computation to that dataset's strata (per
    dataset se/sp, the §8.5 predicate being per dataset); None pools every
    stratum given. Refuses: no strata (or none matching the filter), a
    duplicate (verdict, dataset) key, mixed None/named datasets, a missing
    verdict stratum (both verdicts must be sampled for every dataset
    present), degenerate prevalence (pi = 0 leaves se undefined; pi = 1
    leaves sp undefined), and an uninformative reweighted instrument
    (se + sp <= 1).
    """
    if dataset is not None and not isinstance(dataset, str):
        raise GoodputError(f"dataset={dataset!r} must be a str or None")
    items = list(strata)
    for idx, item in enumerate(items):
        if not isinstance(item, GoldStratum):
            raise GoodputError(
                f"strata[{idx}] is {type(item).__name__}, expected GoldStratum"
            )
    if not items:
        raise GoodputError("no strata supplied")
    named = {item.dataset is not None for item in items}
    if len(named) > 1:
        raise GoodputError(
            "strata mix dataset=None with named datasets; a gold sample is "
            "stratified by verdict only or by verdict x dataset, not both"
        )
    if dataset is not None:
        items = [item for item in items if item.dataset == dataset]
        if not items:
            raise GoodputError(f"no strata for dataset={dataset!r}")
    keys = [item.key for item in items]
    if len(set(keys)) != len(keys):
        dupes = sorted({k for k in keys if keys.count(k) > 1}, key=repr)
        raise GoodputError(f"duplicate strata {dupes}")
    for ds in sorted({item.dataset for item in items}, key=repr):
        present = {item.verdict for item in items if item.dataset == ds}
        for verdict in (True, False):
            if verdict not in present:
                raise GoodputError(
                    f"missing stratum verdict={verdict} for dataset={ds!r}: "
                    "both verdicts must be gold-sampled"
                )
    n_population = sum(item.population_count for item in items)
    weights = {item.key: item.population_count / n_population for item in items}
    prevalence = sum(weights[item.key] * item.gold_true_fraction for item in items)
    true_positive_mass = sum(
        weights[item.key] * item.gold_true_fraction for item in items if item.verdict
    )
    true_negative_mass = sum(
        weights[item.key] * (1.0 - item.gold_true_fraction)
        for item in items
        if not item.verdict
    )
    if prevalence <= 0.0:
        raise GoodputError(
            "degenerate gold sample: no gold-true item in any stratum, "
            "sensitivity is undefined"
        )
    if prevalence >= 1.0:
        raise GoodputError(
            "degenerate gold sample: no gold-false item in any stratum, "
            "specificity is undefined"
        )
    sensitivity = true_positive_mass / prevalence
    specificity = true_negative_mass / (1.0 - prevalence)
    youden = _youden_j(sensitivity, specificity)
    return InstrumentAccuracy(
        sensitivity=sensitivity,
        specificity=specificity,
        youden_j=youden,
        prevalence=prevalence,
        n_sampled=sum(item.n_sampled for item in items),
        n_population=n_population,
        n_strata=len(items),
        weights=weights,
        dataset=dataset,
        estimator=CORRECTED_YIELD_ESTIMATOR,
    )
