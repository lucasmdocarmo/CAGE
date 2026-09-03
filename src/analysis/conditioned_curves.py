"""§8.12 Layer-5 conditioned quality curves — join/bin machinery (W4.11).

PUBLICATION.md §8.12: quality conditioned on serving state — "the joins no
other framework can compute because none can see the serving side". This
module holds the pure join/bin/aggregate logic for two of the three
registered curves; the driver pass in scripts/4_analysis/run_campaign_analysis.py
does the artifact I/O and scripts/4_analysis/figure_pipeline.py renders:

- **quality | ρ_own** — per-window quality means binned on the OWN-accounting
  occupancy ``rho_own`` (src/analysis/own_accounting.py, the §8.8 PRIMARY
  pressure axis; engine self-reported gauges never feed the axis), ONE curve
  per coordinate-free cell identity (``mechanism_engine_key``): §8.12
  registers "the quality-pressure curve per mechanism × engine", so pooling
  arms/engines into one curve would conflate arm composition with pressure
  (a Simpson hazard — arms are not uniformly distributed over ρ bins).
- **quality | evidence-position × pressure** — per-trial quality crossed with
  the gold doc's position among the SERVED contexts (the producer's
  ``gold_position_in_prompt`` containment column) and the cell's registered
  pressure coordinates (lost-in-the-middle under memory pressure).
- **quality | policy-event** — the SAME owner-gated S2 stub as
  ``src.analysis.degradation.join_policy_events``: declared, refused with the
  one named reason until the S2 instrumentation decision lands.

Doctrine: bins are PINNED module constants (a tunable bin edge is an
unregistered analysis choice); every unjoinable input is a named skip carried
by the caller, never a fabricated coordinate; a NaN pressure axis value is an
upstream None-honesty bug and fails loud rather than landing in a bin.
"""

from __future__ import annotations

import math
from typing import Any, NoReturn, Sequence

import pandas as pd

from src.analysis.degradation import (
    S2_POLICY_EVENT_JOIN_UNAVAILABLE,
    join_policy_events,
)

__all__ = [
    "CONDITIONED_QUALITY_METRICS",
    "ConditionedCurveError",
    "EVIDENCE_POSITION_BINS",
    "NO_PRESSURE_COORDS_BIN",
    "RHO_OWN_BIN_EDGES",
    "S2_POLICY_EVENT_JOIN_UNAVAILABLE",
    "evidence_position_bin",
    "mechanism_engine_key",
    "policy_event_curve",
    "pressure_bin_of",
    "quality_by_evidence_position",
    "quality_by_rho_own",
    "rho_own_bin",
]

#: Default per-request quality columns the driver pass conditions on (both
#: are producer/scored columns of requests.jsonl; a metric absent from a
#: window's rows is a NAMED skip, never a zero).
CONDITIONED_QUALITY_METRICS: tuple[str, ...] = ("f1_score", "grounding_score")

#: Pinned ρ_own bin edges (left-closed). The final edge is open-ended:
#: rho_own is byte-seconds over budget×duration and MAY exceed 1.0
#: (overcommit); such windows land in the labeled ">=1.00" bin, they are
#: never clipped.
RHO_OWN_BIN_EDGES: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)

#: Pinned evidence-position vocabulary (order = render order). "absent" is
#: the producer's -1 (gold not in the served context at all) — its own bin,
#: because §8.10's answerability ceiling makes it a distinct physical state,
#: not a missing value.
EVIDENCE_POSITION_BINS: tuple[str, ...] = (
    "first",
    "early",
    "middle",
    "late",
    "absent",
)

#: Pressure-bin label for cells WITHOUT registered pressure coordinates
#: (F1 cells carry budget_r/rate_frac = null by §1 — absence stays absence,
#: labeled, never coerced to a numeric grid point).
NO_PRESSURE_COORDS_BIN = "no-pressure-coords"


class ConditionedCurveError(ValueError):
    """Invalid join input (fail closed — a bad axis value never lands in a bin)."""


# --------------------------------------------------------------------------- #
# Bin functions
# --------------------------------------------------------------------------- #


def rho_own_bin(rho: float) -> str:
    """The pinned ρ_own bin label for one finite, non-negative occupancy.

    A NaN/negative rho is an upstream honesty bug (own_accounting never emits
    one) and fails loud — it must not be quietly binned or dropped here.
    """
    if isinstance(rho, bool) or not isinstance(rho, (int, float)):
        raise ConditionedCurveError(
            f"rho_own={rho!r} must be a number — absent occupancy is the "
            "caller's named skip, never an input here"
        )
    rho = float(rho)
    if not math.isfinite(rho) or rho < 0.0:
        raise ConditionedCurveError(
            f"rho_own={rho!r} must be finite and >= 0 — own_accounting "
            "never emits such a value; fix the producer, not the bin"
        )
    edges = RHO_OWN_BIN_EDGES
    for lo, hi in zip(edges, edges[1:]):
        if lo <= rho < hi:
            return f"[{lo:.2f},{hi:.2f})"
    return f">={edges[-1]:.2f}"


def evidence_position_bin(
    position: Any, n_contexts: Any
) -> tuple[str | None, str | None]:
    """(bin, None) or (None, named reason) for one served-context position.

    ``position`` is the producer's ``gold_position_in_prompt`` (0-based
    served index, -1 = absent); ``n_contexts`` the number of SERVED context
    docs (len(used_contexts)). Deterministic: -1 → "absent"; 0 → "first"
    (needs no n); otherwise the fractional position ``position/(n-1)`` maps
    to early (<= 1/3), middle (<= 2/3), late (> 2/3). Missing/malformed
    inputs return a NAMED reason — never a guessed bin.
    """

    def _as_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return None

    pos = _as_int(position)
    if position is None:
        return None, (
            "gold_position_in_prompt absent from the stored row — the "
            "producer's served-context containment column is required"
        )
    if pos is None or pos < -1:
        return None, (
            f"gold_position_in_prompt={position!r} is not a valid served "
            "index (-1 = absent, >=0 = position) — malformed row"
        )
    if pos == -1:
        return "absent", None
    if pos == 0:
        return "first", None
    n = _as_int(n_contexts)
    if n_contexts is None:
        return None, (
            "n_served_contexts unknown (used_contexts absent from the "
            "evidence row) — a non-zero position cannot be normalized "
            "without the served count"
        )
    if n is None or n < 1:
        return None, (
            f"n_served_contexts={n_contexts!r} is not a positive integer — "
            "malformed row"
        )
    if pos >= n:
        return None, (
            f"gold_position_in_prompt={pos} >= n_served_contexts={n} — "
            "inconsistent row (position outside the served list)"
        )
    frac = pos / (n - 1)  # pos >= 1 here, so n >= 2 and the division is safe
    if frac <= 1.0 / 3.0:
        return "early", None
    if frac <= 2.0 / 3.0:
        return "middle", None
    return "late", None


def mechanism_engine_key(row_key: Any) -> str:
    """The §8.12 ρ-curve identity: the row key's 7 coordinate-free axes.

    The canonical D7 row key is ``arm|retriever|policy|topology|engine|model|
    family`` plus optional swept pressure coordinates (``r<...>``/``lam<...>``
    segments). §8.12 registers quality|ρ_KV "per mechanism × engine": the
    curve identity keeps EVERY mechanism/engine axis and strips ONLY the
    pressure coordinates — pressure is the swept variable a curve traverses,
    never part of its identity (keeping the coords would shatter the sweep
    into per-cell fragments; dropping the axes would pool distinct arms —
    the Simpson hazard). A key without the 7 canonical axes fails closed.
    """
    parts = str(row_key).split("|")
    if len(parts) < 7 or any(not part for part in parts[:7]):
        raise ConditionedCurveError(
            f"row_key {row_key!r} does not carry the 7 canonical axes "
            "(arm|retriever|policy|topology|engine|model|family) — the "
            "§8.12 per-mechanism×engine curve identity cannot be derived "
            "from a non-canonical key (re-run organize_results.py)"
        )
    return "|".join(parts[:7])


def pressure_bin_of(budget_r: Any, rate_frac: Any) -> str:
    """The cell's registered pressure-grid bin label.

    Pressure cells carry BOTH coordinates (§1 windows table); the label is
    the grid point verbatim (``r<budget>|lam<rate>``). Cells without
    coordinates (F1) get the labeled ``NO_PRESSURE_COORDS_BIN`` — explicit
    absence, never a fabricated grid point. A HALF-set coordinate pair is a
    malformed index row and fails loud.
    """

    def _num(value: Any) -> float | None:
        if isinstance(value, bool):
            raise ConditionedCurveError(
                f"pressure coordinate {value!r} is a bool — malformed index"
            )
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return None if math.isnan(float(value)) else float(value)
        raise ConditionedCurveError(
            f"pressure coordinate {value!r} is not numeric/null — malformed "
            "index"
        )

    r = _num(budget_r)
    lam = _num(rate_frac)
    if (r is None) != (lam is None):
        raise ConditionedCurveError(
            f"half-set pressure coordinates (budget_r={budget_r!r}, "
            f"rate_frac={rate_frac!r}) — a §1 window carries both or neither"
        )
    if r is None:
        return NO_PRESSURE_COORDS_BIN
    return f"r{r:g}|lam{lam:g}"


# --------------------------------------------------------------------------- #
# Aggregations (pure frames in, pure frames out)
# --------------------------------------------------------------------------- #

_RHO_INPUT_COLUMNS: tuple[str, ...] = (
    "row_key", "dataset", "window_key", "rho_own", "metric", "value", "n",
)
_EVIDENCE_INPUT_COLUMNS: tuple[str, ...] = (
    "dataset", "evidence_bin", "pressure_bin", "metric", "value",
)


def _require_columns(
    df: pd.DataFrame, columns: Sequence[str], where: str
) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ConditionedCurveError(
            f"{where}: input frame is missing required columns {missing} "
            f"(required: {list(columns)})"
        )
    if df.empty:
        raise ConditionedCurveError(
            f"{where}: input frame is empty — an empty join is the caller's "
            "named skip, not an empty curve"
        )


def quality_by_rho_own(windows: pd.DataFrame) -> pd.DataFrame:
    """quality | ρ_own: bin per-window quality means on the OWN pressure axis.

    Input: one row per (window × metric) with columns
    ``row_key, dataset, window_key, rho_own, metric, value, n`` (``value`` =
    the window's mean of the metric over its per-request rows, ``n`` the row
    count behind that mean). Output: one row per
    (dataset × metric × curve_key × rho_bin) — ``curve_key`` is derived HERE
    from ``row_key`` via ``mechanism_engine_key`` (§8.12: the curve is
    registered per mechanism × engine; distinct cells are never pooled into
    one bin) — with the bin's window count, request count,
    mean-of-window-means and the min/max rho inside the bin. Bin edges are
    the pinned ``RHO_OWN_BIN_EDGES``; every input row must carry a valid
    rho (absence is filtered by the caller WITH a named skip).
    """
    _require_columns(windows, _RHO_INPUT_COLUMNS, "quality_by_rho_own")
    frame = windows.copy()
    frame["curve_key"] = [mechanism_engine_key(k) for k in frame["row_key"]]
    frame["rho_bin"] = [rho_own_bin(v) for v in frame["rho_own"]]
    grouped = (
        frame.groupby(
            ["dataset", "metric", "curve_key", "rho_bin"], observed=True
        )
        .agg(
            n_windows=("window_key", "size"),
            n_requests=("n", "sum"),
            mean_value=("value", "mean"),
            rho_min=("rho_own", "min"),
            rho_max=("rho_own", "max"),
        )
        .reset_index()
    )
    return grouped.sort_values(
        ["dataset", "metric", "curve_key", "rho_min"], kind="stable"
    ).reset_index(drop=True)


def quality_by_evidence_position(trials: pd.DataFrame) -> pd.DataFrame:
    """quality | evidence-position × pressure: the lost-in-the-middle join.

    Input: one row per (trial × metric) with columns
    ``dataset, evidence_bin, pressure_bin, metric, value``
    (``evidence_bin`` from ``evidence_position_bin``, ``pressure_bin`` from
    ``pressure_bin_of``). Output: one row per
    (dataset × metric × pressure_bin × evidence_bin) with the trial count
    and mean quality. RECORDED DECISION: this curve pools trials across
    cells — §8.12 registers it as "evidence-position × pressure" only (the
    per-mechanism×engine conditioning is registered on the quality|ρ_KV
    curve, ``quality_by_rho_own``). Unknown evidence bins fail loud (the
    vocabulary is pinned)."""
    _require_columns(
        trials, _EVIDENCE_INPUT_COLUMNS, "quality_by_evidence_position"
    )
    unknown = set(trials["evidence_bin"]) - set(EVIDENCE_POSITION_BINS)
    if unknown:
        raise ConditionedCurveError(
            f"unknown evidence bins {sorted(unknown)} — the pinned "
            f"vocabulary is {list(EVIDENCE_POSITION_BINS)}"
        )
    grouped = (
        trials.groupby(
            ["dataset", "metric", "pressure_bin", "evidence_bin"],
            observed=True,
        )
        .agg(n_trials=("value", "size"), mean_value=("value", "mean"))
        .reset_index()
    )
    order = {b: i for i, b in enumerate(EVIDENCE_POSITION_BINS)}
    grouped["_bin_order"] = grouped["evidence_bin"].map(order)
    grouped = grouped.sort_values(
        ["dataset", "metric", "pressure_bin", "_bin_order"], kind="stable"
    ).drop(columns=["_bin_order"])
    return grouped.reset_index(drop=True)


def policy_event_curve(*_args: Any, **_kwargs: Any) -> NoReturn:
    """quality | policy-event — the SAME owner-gated S2 stub (§8.12).

    Declared so the L5 output surface names all three registered curves;
    refuses through ``degradation.join_policy_events`` so there is exactly
    ONE S2 refusal string in the codebase.
    """
    join_policy_events()
    raise AssertionError("unreachable: join_policy_events always raises")
