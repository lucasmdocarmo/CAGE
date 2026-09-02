"""§7.8 DIST contrast executors — #18, #19 — and the #14 normalized-pressure
alignment (charter T1.1 + T7.2).

WHAT: the contrast-execution layer for the three distributed-measurement
analyses the registry (``src.analysis.stats.families``) registers but the
driver previously labeled NOT-IMPLEMENTED:

- :func:`execute_contrast_18` — "Distribution's buy-back at transfer price"
  (DIST | window | secondary, gated on #13): pairs windows whose cell identity
  differs ONLY in topology (tp vs pd), matched on
  (arm, retriever, policy, engine, model, dataset, budget_r, rate_frac,
  replicate ordinal), and reports per-GPU goodput/yield deltas on the §6.6
  basis (b) — every pooled pair goes through ``goodput.assert_single_basis``
  and every output row carries the basis label. The #13 gate outcome is an
  INPUT (never recomputed here); an absent gate reports PENDING, never PASS.
- :func:`execute_contrast_19` — "Does dedup survive the wire" (DIST | window |
  exploratory, ungated): per-instance raw prefix-hit accounting from the
  role-tagged telemetry series (instance column, T4.1) using the raw
  cumulative ``prefix_cache_queries_total`` / ``prefix_cache_hits_total``
  counters (T4.2) — prefill-side vs decode-side hit rates on pd windows,
  against the matched single/tp B3 comparator windows.
- :func:`align_pressure_bundles` — the #14 cross-engine policy bundles at the
  SAME normalized pressure (T7.2): windows bundle by engine and align across
  engines on the CAGE-OWN occupancy axis (``rho_own_time_avg`` from the
  own-accounting artifacts). Charter §8.8: the pressure referee must be OUR
  accounting — engine self-reported gauges are demoted to corroboration and
  are NEVER an input to this alignment (structurally: no engine-gauge
  parameter exists). A window without an own-accounting artifact is a labeled
  skip citing §8.8, never silently bucketed.

WHY a separate module: these executors are pure domain logic (no I/O, no
plotting — the run_campaign_analysis driver assembles their inputs from the
tree and writes the artifacts), and the fail-closed pairing/skip discipline
deserves its own pinned test surface (tests/test_dist_contrasts.py).

FAIL-CLOSED doctrine: ambiguous pairings (duplicate windows on one side of a
match key) refuse loudly; absent inputs become LABELED skips naming exactly
what is missing (gpu_count, the instance column, the raw counter fields, a
zero-delta denominator, the own-accounting artifact) — never a silent drop,
never a fabricated value. CIs are never invented: #18 emits point estimates
with n and an explicit "window-block bootstrap [pending]" marker (the
existing window-block bootstrap in ``stats.equivalence`` is TOST-shaped and
does not compose cheaply onto a paired-delta effect; a fabricated interval
would be worse than a labeled absence).

Domain logic only: stdlib + the goodput basis seam; mappings in, dicts out.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from src.analysis.goodput import (
    BASIS_AGGREGATE,
    BASIS_PER_GPU,
    assert_single_basis,
)
from src.analysis.stats.families import (
    UNGATED,
    WINDOW_SECONDARY_UPSTREAM,
)

__all__ = [
    "CONTRAST_18_ID",
    "CONTRAST_19_ID",
    "DIST_EXECUTOR_IDS",
    "PENDING_CI_MARKER",
    "DistContrastError",
    "align_pressure_bundles",
    "execute_contrast_18",
    "execute_contrast_19",
]

CONTRAST_18_ID: int = 18
CONTRAST_19_ID: int = 19
#: The §7.8 ids this module executes — the driver's dispatch excludes them
#: from the baseline-pair pipeline exactly as it does for #12/#13/#14.
DIST_EXECUTOR_IDS: frozenset[int] = frozenset({CONTRAST_18_ID, CONTRAST_19_ID})

#: The registered #18 gate — families.py's gating topology, honored verbatim
#: (DIST secondaries gate on the #13 fingerprint endpoint). Imported, never
#: respelled: a driver hard-code is exactly the G10 bug class.
CONTRAST_18_UPSTREAM: str = WINDOW_SECONDARY_UPSTREAM["DIST"]

#: The honest no-CI marker (see module docstring): point estimates + n ship
#: with THIS string instead of a fabricated interval.
PENDING_CI_MARKER: str = "CI: window-block bootstrap [pending]"

#: The cell-identity axes a #18 pair must AGREE on — everything but topology.
#: ``replicate`` is the window ordinal (organize_results' ``window`` column):
#: pairing replicate k against replicate k keeps the pairing honest across
#: repeated windows of the same grid point.
MATCH_AXES: tuple[str, ...] = (
    "arm",
    "retriever",
    "policy",
    "engine",
    "model",
    "dataset",
    "budget_r",
    "rate_frac",
    "replicate",
)

#: §6.6b per-GPU fields #18 reports (basis b — transfer/protocol contrasts
#: ALWAYS report basis b per the goodput module's charter binding).
_PER_GPU_DELTA_FIELDS: tuple[str, ...] = ("goodput_per_gpu", "yield_per_gpu")

#: T4.2 raw cumulative counters (cage-stats state.py spellings — the raw
#: fields ride the derived hit-rate ones precisely so this accounting never
#: depends on a derived value).
QUERIES_FIELD: str = "prefix_cache_queries_total"
HITS_FIELD: str = "prefix_cache_hits_total"
#: T4.1 role-tagged series column.
INSTANCE_FIELD: str = "instance"

_PD_ROLES: tuple[str, ...] = ("prefill", "decode")

#: #19 rides the B3 cell (families registry: baseline_a="B3") — both the pd
#: side and the single/tp comparator.
_CONTRAST_19_BASELINE: str = "B3"

#: The Y variable per §6.6 basis for the #14 alignment rows. yield_frac is
#: the aggregate-basis fraction-of-issued Y; yield_per_gpu the §6.6b rate.
_Y_FIELD_BY_BASIS: dict[str, str] = {
    BASIS_AGGREGATE: "yield_frac",
    BASIS_PER_GPU: "yield_per_gpu",
}


class DistContrastError(ValueError):
    """Ambiguous pairing or malformed input (fail closed, message first)."""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _window_label(win: Mapping[str, Any], idx: int) -> str:
    label = win.get("window")
    return str(label) if label else f"windows[{idx}]"


def _require_identity(win: Mapping[str, Any], idx: int) -> None:
    """Identity axes are load-bearing for pairing: a window that cannot state
    its identity would poison the matcher — refuse loud, never skip."""
    missing = [
        axis
        for axis in MATCH_AXES
        if axis not in win  # None IS a legal value for the pressure coords
    ]
    if missing:
        raise DistContrastError(
            f"{_window_label(win, idx)}: identity axes {missing} are absent — "
            "a window without its cell identity cannot be paired (fail closed; "
            f"required axes: {list(MATCH_AXES)})"
        )


def _coord_token(value: Any, axis: str, ctx: str) -> str:
    """String key for an optional pressure coordinate (the driver's
    ``_coord_keyed`` lesson: NaN keys never match across groupbys, so absence
    is matched as absence — '<unset>' — never coerced to a number)."""
    if value is None or (isinstance(value, float) and math.isnan(value)) or value == "":
        return "<unset>"
    try:
        return repr(float(value))
    except (TypeError, ValueError):
        raise DistContrastError(
            f"{ctx}: axis {axis!r} holds non-numeric value {value!r} — "
            "pressure coordinates must be floats or absent"
        ) from None


def _match_key(win: Mapping[str, Any], idx: int) -> tuple[str, ...]:
    ctx = _window_label(win, idx)
    parts: list[str] = []
    for axis in MATCH_AXES:
        value = win[axis]
        if axis in ("budget_r", "rate_frac"):
            parts.append(_coord_token(value, axis, ctx))
        else:
            parts.append(str(value))
    return tuple(parts)


def _axes_dict(key: tuple[str, ...]) -> dict[str, str]:
    return dict(zip(MATCH_AXES, key))


def _metrics_field(metrics: object, name: str, ctx: str) -> float:
    """Read one WindowMetrics field from a live dataclass OR its JSON mapping
    form (both are first-class after ``assert_single_basis`` audited them)."""
    if isinstance(metrics, Mapping):
        if name in metrics:
            return float(metrics[name])
    else:
        marker = object()
        value = getattr(metrics, name, marker)
        if value is not marker:
            return float(value)  # type: ignore[arg-type]
    raise DistContrastError(f"{ctx}: window metrics carry no {name!r} field")


def _finite_number(value: Any, name: str, ctx: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DistContrastError(f"{ctx}: {name}={value!r} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise DistContrastError(f"{ctx}: {name}={value!r} must be finite")
    return value


# ---------------------------------------------------------------------------
# #18 — distribution's buy-back at transfer price (tp vs pd, basis b)
# ---------------------------------------------------------------------------


def _gate_section(gate_13: Mapping[str, Any] | None) -> dict[str, Any]:
    """The #18 gate block. The #13 outcome is an INPUT (task pin: accept it,
    never recompute it); an absent outcome reports PENDING — a pending check
    must never read as PASS (fail-closed doctrine)."""
    if gate_13 is None:
        return {
            "upstream": CONTRAST_18_UPSTREAM,
            "outcome": None,
            "status": "PENDING",
            "note": (
                "no #13 outcome supplied — the gate is PENDING, not open; "
                "rows below are descriptive until the upstream fingerprint "
                "endpoint's verdict exists (§9.3 gating topology)"
            ),
        }
    if not isinstance(gate_13, Mapping) or not isinstance(
        gate_13.get("passed"), bool
    ):
        raise DistContrastError(
            f"gate_13={gate_13!r} must be a mapping carrying a bool 'passed' "
            "(the upstream #13 endpoint verdict, e.g. one of "
            "stats['gatekeeping']['set_decisions']) — or None for PENDING"
        )
    return {
        "upstream": CONTRAST_18_UPSTREAM,
        "outcome": dict(gate_13),
        "status": "OPEN" if gate_13["passed"] else "CLOSED",
        "note": None,
    }


def execute_contrast_18(
    windows: Sequence[Mapping[str, Any]],
    *,
    gate_13: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """§7.8 #18 — per-GPU goodput/yield deltas of pd vs tp matched windows.

    Each ``windows`` item is a mapping carrying the ``MATCH_AXES`` identity
    (refused loud when absent), ``topology`` ('tp' or 'pd'; anything else is
    a labeled skip — 'single' is not a #18 side), ``window`` (label),
    ``gpu_count`` (caller-supplied; ABSENT ⇒ labeled skip naming the window,
    never silent), and ``metrics`` (a ``goodput.WindowMetrics`` or its
    flat-dict/JSON form; absent ⇒ labeled skip).

    Charter §6.6: the transfer contrast reports basis (b) — every matched
    pair is audited by ``assert_single_basis(..., BASIS_PER_GPU)`` before its
    per-GPU fields are read, and every output row carries the basis label.
    delta = pd − tp per field (positive = distribution buys back per-GPU
    goodput/yield NET of the transfer price).

    Effects are point estimates with n and the paired per-window deltas;
    ``ci`` is the explicit ``PENDING_CI_MARKER`` (see module docstring —
    never a fabricated interval). Duplicate windows on one side of a match
    key refuse loud (ambiguous pairing).
    """
    skips: list[dict[str, str]] = []
    #: match key -> {"tp": window-record, "pd": window-record}
    sides: dict[tuple[str, ...], dict[str, dict[str, Any]]] = {}

    for idx, win in enumerate(windows):
        if not isinstance(win, Mapping):
            raise DistContrastError(
                f"windows[{idx}] is not a mapping ({type(win).__name__})"
            )
        _require_identity(win, idx)
        label = _window_label(win, idx)
        topology = win.get("topology")
        if topology not in ("tp", "pd"):
            skips.append(
                {
                    "window": label,
                    "reason": (
                        f"topology={topology!r} is not a #18 side — the "
                        "contrast pairs tp against pd (the DIST overlay); "
                        "labeled out, never silently dropped"
                    ),
                }
            )
            continue
        if win.get("metrics") is None:
            skips.append(
                {
                    "window": label,
                    "reason": (
                        "no WindowMetrics supplied for this window — G/Y "
                        "were never evaluated; nothing to delta (labeled "
                        "skip, never a fabricated value)"
                    ),
                }
            )
            continue
        gpu_count = win.get("gpu_count")
        if gpu_count is None:
            skips.append(
                {
                    "window": label,
                    "reason": (
                        f"window {label!r} carries no gpu_count — the §6.6b "
                        "per-GPU basis is undefined without it (caller must "
                        "supply the GPU count per window; labeled skip, "
                        "never a silent 1-GPU default)"
                    ),
                }
            )
            continue
        key = _match_key(win, idx)
        slot = sides.setdefault(key, {})
        if topology in slot:
            raise DistContrastError(
                f"contrast #18: match key {_axes_dict(key)} holds duplicate "
                f"{topology} windows ({_window_label(slot[topology], -1)!r} "
                f"and {label!r}) — the pairing is ambiguous; refusing to guess"
            )
        slot[topology] = dict(win)

    pairs: list[dict[str, Any]] = []
    for key in sorted(sides):
        slot = sides[key]
        missing_side = [t for t in ("tp", "pd") if t not in slot]
        if missing_side:
            present = next(iter(slot.values()))
            skips.append(
                {
                    "window": _window_label(present, -1),
                    "reason": (
                        f"match key {_axes_dict(key)}: no {missing_side[0]} "
                        "partner window in this pool — unmatched, labeled out"
                    ),
                }
            )
            continue
        tp_win, pd_win = slot["tp"], slot["pd"]
        # §6.6 seam: the pooled pair must sit on ONE declared basis before
        # any per-GPU number is read. GoodputError propagates — a mixed-basis
        # record is a data violation, not an absence.
        assert_single_basis([tp_win["metrics"], pd_win["metrics"]], BASIS_PER_GPU)
        for side_win in (tp_win, pd_win):
            declared = _metrics_field(
                side_win["metrics"], "gpu_count", _window_label(side_win, -1)
            )
            if declared != float(side_win["gpu_count"]):
                raise DistContrastError(
                    f"{_window_label(side_win, -1)}: caller-supplied "
                    f"gpu_count={side_win['gpu_count']!r} contradicts the "
                    f"metrics' own gpu_count={declared!r} — the window's "
                    "basis facts disagree; refusing to guess which is real"
                )
        row: dict[str, Any] = {
            "axes": _axes_dict(key),
            "dataset": _axes_dict(key)["dataset"],
            "tp_window": _window_label(tp_win, -1),
            "pd_window": _window_label(pd_win, -1),
            "gpu_count_tp": int(tp_win["gpu_count"]),
            "gpu_count_pd": int(pd_win["gpu_count"]),
            "basis": BASIS_PER_GPU,
        }
        for field in _PER_GPU_DELTA_FIELDS:
            tp_v = _metrics_field(tp_win["metrics"], field, row["tp_window"])
            pd_v = _metrics_field(pd_win["metrics"], field, row["pd_window"])
            row[f"{field}_tp"] = tp_v
            row[f"{field}_pd"] = pd_v
            row[f"delta_{field}"] = pd_v - tp_v
        pairs.append(row)

    per_dataset: list[dict[str, Any]] = []
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in pairs:
        by_dataset.setdefault(row["dataset"], []).append(row)
    for dataset in sorted(by_dataset):
        rows = by_dataset[dataset]
        entry: dict[str, Any] = {
            "dataset": dataset,
            "n_pairs": len(rows),
            "basis": BASIS_PER_GPU,
            # Point estimate + n + the paired deltas; the CI is an explicit
            # pending marker, never a fabricated interval (module docstring).
            "ci": PENDING_CI_MARKER,
        }
        for field in _PER_GPU_DELTA_FIELDS:
            deltas = [r[f"delta_{field}"] for r in rows]
            entry[f"paired_deltas_{field}"] = deltas
            entry[f"mean_delta_{field}"] = sum(deltas) / len(deltas)
        per_dataset.append(entry)

    return {
        "contrast_id": CONTRAST_18_ID,
        "name": "Distribution's buy-back at transfer price",
        "tier": "secondary",
        "unit": "window",
        "basis": BASIS_PER_GPU,
        "basis_note": (
            "§6.6 basis (b): per-GPU goodput/yield — transfer/protocol "
            "contrasts #18/#19 always report basis b; every pooled pair "
            "audited by goodput.assert_single_basis"
        ),
        "delta_convention": "delta = pd − tp per §6.6b field",
        "gate": _gate_section(gate_13),
        "pairs": pairs,
        "per_dataset": per_dataset,
        "skips": skips,
        "status": "EXECUTED" if pairs else "SKIPPED-INPUTS-INCOMPLETE",
    }


# ---------------------------------------------------------------------------
# #19 — does dedup survive the wire (raw prefix-hit accounting, per role)
# ---------------------------------------------------------------------------


def _instance_counter_deltas(
    samples: Sequence[Mapping[str, Any]], window: str, instance: str
) -> tuple[tuple[float, float] | None, str | None]:
    """(Δqueries, Δhits) of one instance's raw cumulative counters, or a
    labeled-skip reason. First/last samples CARRYING BOTH raw fields bound
    the delta (cumulative counters; the series is time-ordered by contract).
    A counter running backwards or hits > queries is a data violation and
    refuses loud; a zero-delta denominator is a labeled skip."""
    carrying = [
        s
        for s in samples
        if s.get(QUERIES_FIELD) is not None and s.get(HITS_FIELD) is not None
    ]
    if not carrying:
        return None, (
            f"instance {instance!r}: no sample carries the raw cumulative "
            f"{QUERIES_FIELD!r}/{HITS_FIELD!r} fields (Wave-1 T4.2) — raw "
            "accounting is impossible without them"
        )
    if len(carrying) < 2:
        return None, (
            f"instance {instance!r}: only 1 sample carries the raw "
            f"cumulative fields — a single point has no delta"
        )
    ctx = f"{window} / instance {instance!r}"
    q0 = _finite_number(carrying[0][QUERIES_FIELD], QUERIES_FIELD, ctx)
    q1 = _finite_number(carrying[-1][QUERIES_FIELD], QUERIES_FIELD, ctx)
    h0 = _finite_number(carrying[0][HITS_FIELD], HITS_FIELD, ctx)
    h1 = _finite_number(carrying[-1][HITS_FIELD], HITS_FIELD, ctx)
    dq, dh = q1 - q0, h1 - h0
    if dq < 0 or dh < 0:
        raise DistContrastError(
            f"{ctx}: cumulative counter ran BACKWARDS "
            f"(Δqueries={dq}, Δhits={dh}) — an engine restart or series "
            "mis-order corrupts raw accounting; refusing to fabricate"
        )
    if dh > dq:
        raise DistContrastError(
            f"{ctx}: Δhits={dh} > Δqueries={dq} — a hit rate > 1 is "
            "unphysical; refusing to clamp"
        )
    if dq == 0:
        return None, (
            f"instance {instance!r}: zero-delta denominator "
            f"(Δ{QUERIES_FIELD} == 0 over the window) — no query traffic, "
            "no hit rate (labeled skip, never 0/0 → 0)"
        )
    return (dq, dh), None


def _role_of_instance(
    instance: str, roles: Mapping[str, str] | None
) -> tuple[str | None, str | None]:
    """(role, skip-reason). Explicit ``roles`` mapping wins; otherwise the
    role is derived from the T4.1 role-tagged instance value itself (contains
    exactly one of 'prefill'/'decode'). Underivable ⇒ labeled skip — never a
    guessed role."""
    if roles is not None and instance in roles:
        role = roles[instance]
        if role not in _PD_ROLES:
            raise DistContrastError(
                f"roles[{instance!r}]={role!r} is not a pd role "
                f"{list(_PD_ROLES)}"
            )
        return role, None
    low = instance.lower()
    has_prefill = "prefill" in low
    has_decode = "decode" in low
    if has_prefill == has_decode:
        return None, (
            f"instance {instance!r} carries no derivable role tag (T4.1 "
            "role-tagged series: the instance value must name exactly one "
            "of 'prefill'/'decode', or pass an explicit roles mapping) — "
            "labeled skip, never a guessed role"
        )
    return ("prefill" if has_prefill else "decode"), None


def _window_series_by_instance(
    win: Mapping[str, Any], label: str
) -> tuple[dict[str, list[Mapping[str, Any]]] | None, str | None]:
    series = win.get("series")
    if not series:
        source = win.get("series_source")
        return None, (
            f"no telemetry series samples for {label!r}"
            + (f" ({source})" if source else "")
            + " — the T4.1 role-tagged series is the #19 input"
        )
    by_instance: dict[str, list[Mapping[str, Any]]] = {}
    for i, sample in enumerate(series):
        if not isinstance(sample, Mapping):
            raise DistContrastError(
                f"{label}: series[{i}] is not a mapping "
                f"({type(sample).__name__})"
            )
        instance = sample.get(INSTANCE_FIELD)
        if not isinstance(instance, str) or not instance:
            return None, (
                f"series sample {i} of {label!r} carries no "
                f"{INSTANCE_FIELD!r} column (T4.1 role-tagged series) — "
                "per-instance accounting is impossible without it"
            )
        by_instance.setdefault(instance, []).append(sample)
    return by_instance, None


def execute_contrast_19(
    windows: Sequence[Mapping[str, Any]],
    *,
    roles: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """§7.8 #19 — per-instance raw prefix-hit accounting across the wire.

    Each ``windows`` item carries the ``MATCH_AXES`` identity, ``topology``
    ('pd' = the measured side; 'single'/'tp' = comparator candidates),
    ``baseline`` (the registry pins B3 — other baselines are labeled out),
    ``window`` (label) and ``series`` — the T4.1 role-tagged telemetry
    samples, each a mapping with the ``instance`` column and the raw
    cumulative T4.2 counters. Exploratory and UNGATED by registration
    ("either answer is a finding").

    Per pd window: prefill-side and decode-side hit rates
    (Σ Δhits / Σ Δqueries per role, from the RAW cumulative counters — never
    a derived rate field). Per matched comparator (same identity axes,
    topology single or tp): its overall hit rate, and the per-role deltas
    against it. Missing instance column, missing raw fields, and zero-delta
    denominators are LABELED skips naming the exact missing input; duplicate
    windows on a (key, topology) slot refuse loud.
    """
    skips: list[dict[str, str]] = []
    pd_side: dict[tuple[str, ...], dict[str, Any]] = {}
    comparators: dict[tuple[str, ...], dict[str, dict[str, Any]]] = {}

    for idx, win in enumerate(windows):
        if not isinstance(win, Mapping):
            raise DistContrastError(
                f"windows[{idx}] is not a mapping ({type(win).__name__})"
            )
        _require_identity(win, idx)
        label = _window_label(win, idx)
        baseline = win.get("baseline")
        if baseline != _CONTRAST_19_BASELINE:
            skips.append(
                {
                    "window": label,
                    "reason": (
                        f"baseline={baseline!r} — #19 rides the "
                        f"{_CONTRAST_19_BASELINE} cell by registration "
                        "(families.py: baseline_a='B3'); labeled out"
                    ),
                }
            )
            continue
        topology = win.get("topology")
        key = _match_key(win, idx)
        if topology == "pd":
            if key in pd_side:
                raise DistContrastError(
                    f"contrast #19: match key {_axes_dict(key)} holds "
                    f"duplicate pd windows "
                    f"({_window_label(pd_side[key], -1)!r} and {label!r}) — "
                    "ambiguous; refusing to guess"
                )
            pd_side[key] = dict(win)
        elif topology in ("single", "tp"):
            slot = comparators.setdefault(key, {})
            if topology in slot:
                raise DistContrastError(
                    f"contrast #19: match key {_axes_dict(key)} holds "
                    f"duplicate {topology} comparator windows — ambiguous; "
                    "refusing to guess"
                )
            slot[topology] = dict(win)
        else:
            skips.append(
                {
                    "window": label,
                    "reason": f"topology={topology!r} is neither the pd side "
                    "nor a single/tp comparator — labeled out",
                }
            )

    def overall_rate(
        win: Mapping[str, Any],
    ) -> tuple[dict[str, float] | None, str | None]:
        label = _window_label(win, -1)
        by_instance, reason = _window_series_by_instance(win, label)
        if by_instance is None:
            return None, reason
        total_q = total_h = 0.0
        for instance in sorted(by_instance):
            deltas, inst_reason = _instance_counter_deltas(
                by_instance[instance], label, instance
            )
            if deltas is None:
                return None, inst_reason
            total_q += deltas[0]
            total_h += deltas[1]
        if total_q == 0:
            return None, (
                f"{label!r}: zero-delta denominator across every instance — "
                "no query traffic, no hit rate"
            )
        return (
            {
                "hit_rate": total_h / total_q,
                "delta_queries": total_q,
                "delta_hits": total_h,
            },
            None,
        )

    def per_role_rates(
        win: Mapping[str, Any],
    ) -> tuple[dict[str, dict[str, float]] | None, str | None]:
        label = _window_label(win, -1)
        by_instance, reason = _window_series_by_instance(win, label)
        if by_instance is None:
            return None, reason
        totals: dict[str, list[float]] = {r: [0.0, 0.0] for r in _PD_ROLES}
        seen: dict[str, int] = {r: 0 for r in _PD_ROLES}
        for instance in sorted(by_instance):
            role, role_reason = _role_of_instance(instance, roles)
            if role is None:
                return None, f"{label!r}: {role_reason}"
            deltas, inst_reason = _instance_counter_deltas(
                by_instance[instance], label, instance
            )
            if deltas is None:
                return None, f"{label!r}: {inst_reason}"
            totals[role][0] += deltas[0]
            totals[role][1] += deltas[1]
            seen[role] += 1
        out: dict[str, dict[str, float]] = {}
        for role in _PD_ROLES:
            if seen[role] == 0:
                return None, (
                    f"{label!r}: no {role}-role instance in the series — a "
                    "pd window without both roles cannot answer the "
                    "before/after-the-wire question (T4.1 tagging incomplete)"
                )
            dq, dh = totals[role]
            if dq == 0:
                return None, (
                    f"{label!r}: zero-delta denominator on the {role} side "
                    f"(Δ{QUERIES_FIELD} == 0) — labeled skip, never 0/0 → 0"
                )
            out[role] = {
                "hit_rate": dh / dq,
                "delta_queries": dq,
                "delta_hits": dh,
                "n_instances": float(seen[role]),
            }
        return out, None

    rows: list[dict[str, Any]] = []
    for key in sorted(pd_side):
        pd_win = pd_side[key]
        pd_label = _window_label(pd_win, -1)
        role_rates, reason = per_role_rates(pd_win)
        if role_rates is None:
            skips.append({"window": pd_label, "reason": str(reason)})
            continue
        comp_slot = comparators.get(key, {})
        if not comp_slot:
            skips.append(
                {
                    "window": pd_label,
                    "reason": (
                        f"match key {_axes_dict(key)}: no single/tp "
                        f"{_CONTRAST_19_BASELINE} comparator window — the "
                        "dedup-baseline is unmeasured for this grid point"
                    ),
                }
            )
            continue
        for comp_topology in sorted(comp_slot):
            comp_win = comp_slot[comp_topology]
            comp_label = _window_label(comp_win, -1)
            comp_rate, comp_reason = overall_rate(comp_win)
            if comp_rate is None:
                skips.append({"window": comp_label, "reason": str(comp_reason)})
                continue
            rows.append(
                {
                    "axes": _axes_dict(key),
                    "dataset": _axes_dict(key)["dataset"],
                    "pd_window": pd_label,
                    "comparator_window": comp_label,
                    "comparator_topology": comp_topology,
                    "prefill": role_rates["prefill"],
                    "decode": role_rates["decode"],
                    "comparator": comp_rate,
                    "delta_prefill_vs_comparator": (
                        role_rates["prefill"]["hit_rate"]
                        - comp_rate["hit_rate"]
                    ),
                    "delta_decode_vs_comparator": (
                        role_rates["decode"]["hit_rate"] - comp_rate["hit_rate"]
                    ),
                }
            )

    return {
        "contrast_id": CONTRAST_19_ID,
        "name": "Does dedup survive the wire",
        "tier": "exploratory",
        "unit": "window",
        "gate": UNGATED,
        "accounting": (
            f"raw cumulative {QUERIES_FIELD}/{HITS_FIELD} deltas per "
            f"{INSTANCE_FIELD} (T4.1/T4.2) — never a derived hit-rate field"
        ),
        "note": "§3.2 measurement target — either answer is a finding",
        "rows": rows,
        "skips": skips,
        "n_rows": len(rows),
        "status": "EXECUTED" if rows else "SKIPPED-INPUTS-INCOMPLETE",
    }


# ---------------------------------------------------------------------------
# T7.2 — #14 normalized-pressure alignment on the OWN-accounting rho axis
# ---------------------------------------------------------------------------


def align_pressure_bundles(
    windows: Sequence[Mapping[str, Any]],
    *,
    r_levels: Sequence[float],
    tol: float = 0.10,
    basis: str = BASIS_AGGREGATE,
    anchor_engine: str = "vllm",
) -> dict[str, Any]:
    """T7.2 — cross-engine policy bundles (#14) at the SAME normalized
    pressure, aligned on the CAGE-OWN occupancy axis.

    Charter §8.8 (audit §2.6): the pressure referee must be OUR accounting —
    engine self-reported occupancy gauges are demoted to corroboration and
    are NEVER an alignment input. That is the whole point of this pass:
    an engine could misreport its own gauge into a flattering bucket, so the
    axis here is ``rho_own_time_avg`` from the own-accounting artifacts
    (``src.analysis.own_accounting``, T2.5) and nothing else. Structurally no
    engine-gauge parameter exists; a window without its own-accounting value
    is a LABELED skip citing §8.8, never silently bucketed.

    Each ``windows`` item: ``window`` (label), ``engine``, ``dataset``,
    ``rho_own`` (the window's rho_own_time_avg; None/absent ⇒ labeled skip),
    ``metrics`` (WindowMetrics or its mapping form; audited on the declared
    ``basis`` by ``assert_single_basis`` before any Y is read).

    Bucketing: a window joins its NEAREST registered r level ONLY when
    |rho_own − r| <= tol (absolute; the boundary is INSIDE). Windows matching
    no bucket are LABELED out with their distances; a window INSIDE the band
    but exactly equidistant between two registered levels refuses loud (the
    grid cannot place it — guessing a bucket would be a silent alignment
    error; outside the band the tie is moot and the window labels out).
    ``tol`` is recorded in the output.

    Rows: per (bucket r, dataset), each non-anchor engine vs ``anchor_engine``
    on the basis-labeled Y variable (aggregate ⇒ ``yield_frac``, per-gpu ⇒
    ``yield_per_gpu``) — mean per side, delta, n, and the per-window values.
    """
    if basis not in _Y_FIELD_BY_BASIS:
        raise DistContrastError(
            f"basis={basis!r} is not a §6.6 basis label "
            f"({sorted(_Y_FIELD_BY_BASIS)})"
        )
    if not r_levels:
        raise DistContrastError(
            "r_levels is empty — alignment needs the registered §6.1 r grid"
        )
    levels: list[float] = []
    for r in r_levels:
        r = _finite_number(r, "r_level", "align_pressure_bundles")
        if r <= 0:
            raise DistContrastError(f"r_level={r!r} must be > 0")
        levels.append(r)
    if len(set(levels)) != len(levels):
        raise DistContrastError(f"duplicate r_levels in {list(r_levels)}")
    tol = _finite_number(tol, "tol", "align_pressure_bundles")
    if tol <= 0:
        raise DistContrastError(f"tol={tol!r} must be > 0")
    y_field = _Y_FIELD_BY_BASIS[basis]

    skips: list[dict[str, str]] = []
    labeled_out: list[dict[str, Any]] = []
    #: (r_level, dataset) -> engine -> list of {window, rho_own, y}
    buckets: dict[tuple[float, str], dict[str, list[dict[str, Any]]]] = {}

    for idx, win in enumerate(windows):
        if not isinstance(win, Mapping):
            raise DistContrastError(
                f"windows[{idx}] is not a mapping ({type(win).__name__})"
            )
        label = _window_label(win, idx)
        for field in ("engine", "dataset"):
            if not win.get(field):
                raise DistContrastError(
                    f"{label}: {field!r} is absent — a window without its "
                    "engine/dataset identity cannot bundle"
                )
        rho_own = win.get("rho_own")
        if rho_own is None:
            # A caller-supplied rho_own_reason names WHY (e.g. a present-but-
            # malformed artifact) — more auditable than the generic absence
            # text (2026-09-01 verifier minor).
            supplied = win.get("rho_own_reason")
            skips.append(
                {
                    "window": label,
                    "reason": supplied or (
                        f"no own-accounting rho_own for {label!r} — §8.8: "
                        "the alignment axis is CAGE-OWN occupancy "
                        "(own_accounting.json, rho_own_time_avg); the engine "
                        "self-reported gauge is corroboration only and is "
                        "never used to align (labeled skip)"
                    ),
                }
            )
            continue
        rho_own = _finite_number(rho_own, "rho_own", label)
        if win.get("metrics") is None:
            skips.append(
                {
                    "window": label,
                    "reason": (
                        f"no WindowMetrics for {label!r} — Y was never "
                        "evaluated for this window; nothing to compare"
                    ),
                }
            )
            continue
        distances = {r: abs(rho_own - r) for r in levels}
        best = min(distances.values())
        nearest = [r for r, d in distances.items() if d == best]
        # Order matters: outside the band the tie is moot (labeled out either
        # way); INSIDE the band an equidistant window has two legal buckets
        # and guessing one would be a silent alignment error — refuse loud.
        if best > tol:
            labeled_out.append(
                {
                    "window": label,
                    "rho_own": rho_own,
                    "nearest_r": nearest[0],
                    "distance": best,
                    "reason": (
                        f"|rho_own − r| = {best:.6g} > tol={tol:g} for every "
                        "registered level — outside the alignment band, "
                        "labeled out"
                    ),
                }
            )
            continue
        if len(nearest) > 1:
            raise DistContrastError(
                f"{label}: rho_own={rho_own} is exactly equidistant between "
                f"registered r levels {sorted(nearest)} within tol={tol:g} — "
                "the grid cannot place this window; refusing to guess a bucket"
            )
        bucket = buckets.setdefault((nearest[0], str(win["dataset"])), {})
        bucket.setdefault(str(win["engine"]), []).append(
            {"window": label, "rho_own": rho_own, "metrics": win["metrics"]}
        )

    rows: list[dict[str, Any]] = []
    bucket_skips: list[dict[str, Any]] = []
    for (r_level, dataset) in sorted(buckets):
        engines = buckets[(r_level, dataset)]
        # §6.6 seam: ONE declared basis per pooled bucket before any Y read.
        assert_single_basis(
            [e["metrics"] for members in engines.values() for e in members],
            basis,
        )
        if anchor_engine not in engines:
            bucket_skips.append(
                {
                    "r_level": r_level,
                    "dataset": dataset,
                    "engines": sorted(engines),
                    "reason": (
                        f"no {anchor_engine!r} anchor bundle in this bucket "
                        "— cross-engine rows need the anchor (§7.3)"
                    ),
                }
            )
            continue
        others = sorted(e for e in engines if e != anchor_engine)
        if not others:
            bucket_skips.append(
                {
                    "r_level": r_level,
                    "dataset": dataset,
                    "engines": sorted(engines),
                    "reason": "anchor engine only — no cross-engine partner",
                }
            )
            continue
        anchor_members = engines[anchor_engine]
        anchor_y = [
            _metrics_field(m["metrics"], y_field, m["window"])
            for m in anchor_members
        ]
        for engine in others:
            members = engines[engine]
            y_values = [
                _metrics_field(m["metrics"], y_field, m["window"])
                for m in members
            ]
            rows.append(
                {
                    "r_level": r_level,
                    "dataset": dataset,
                    "engine": engine,
                    "anchor_engine": anchor_engine,
                    "basis": basis,
                    "y_field": y_field,
                    "tol": tol,
                    "n_windows_engine": len(members),
                    "n_windows_anchor": len(anchor_members),
                    "mean_y_engine": sum(y_values) / len(y_values),
                    "mean_y_anchor": sum(anchor_y) / len(anchor_y),
                    "delta_y": (
                        sum(y_values) / len(y_values)
                        - sum(anchor_y) / len(anchor_y)
                    ),
                    "windows_engine": [
                        {"window": m["window"], "rho_own": m["rho_own"], "y": y}
                        for m, y in zip(members, y_values)
                    ],
                    "windows_anchor": [
                        {"window": m["window"], "rho_own": m["rho_own"], "y": y}
                        for m, y in zip(anchor_members, anchor_y)
                    ],
                }
            )

    return {
        "contrast_id": 14,
        "name": "Cross-engine policy bundles at normalized pressure (T7.2)",
        "alignment_axis": (
            "rho_own_time_avg — CAGE-OWN accounting "
            "(src.analysis.own_accounting, §8.8: engine self-reported gauges "
            "are corroboration only and never feed the alignment)"
        ),
        "r_levels": sorted(levels),
        "tol": tol,
        "basis": basis,
        "y_field": y_field,
        "anchor_engine": anchor_engine,
        "rows": rows,
        "bucket_skips": bucket_skips,
        "labeled_out": labeled_out,
        "skips": skips,
        "status": "EXECUTED" if rows else "SKIPPED-INPUTS-INCOMPLETE",
    }
