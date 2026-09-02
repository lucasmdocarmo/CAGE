"""CAGE-OWN Layer-1 accounting: ρ_KV and ρ_reuse from OUR records, engine demoted.

Audit finding (§2.6/§8.8): the pressure referee ρ_KV read the ENGINE's own
occupancy gauge — "engine-agnostic in form, engine-sourced in substance" — and
reuse existed only as engine ``cached_tokens`` self-reports. The charter
demands the Layer-1 quantities from CAGE's OWN accounting, with engine
self-reports demoted to CORROBORATION. This module is that producer:

- :func:`compute_own_occupancy` (T2.5) reconstructs per-request in-flight
  intervals from the dispatcher-recorded requests.jsonl fields and integrates
  a bytes(t) model against the per-model KV arithmetic imported from
  ``src.orchestration.cache_budget.MODEL_KV`` (never copied);
- :func:`compute_own_reuse` (T8.2) derives the shared-prefix reuse fraction
  from the WORKLOAD MANIFEST's known shared-prefix token counts (the corpus
  blocks packed by ``src/data/manifest.py``: ``blocks[i].token_count`` keyed
  through each request row's ``group_id``), with engine
  ``cached_prompt_tokens`` carried as the corroboration column only;
- :func:`divergence_report` names the own-vs-engine gap, None-honest.

Occupancy approximation (documented, honest — stated here because the code
cannot show it): only interval ENDPOINTS are recorded per request
(``actual_send_ts``, ``first_token_ts``, ``completion_ts`` on the
dispatcher's clock — ``time.monotonic`` by default, NOT the epoch clock of
cell.json's t_start/t_end; window bounds passed here must be on the SAME
clock as the request timestamps). The model holds all ``prompt_tokens`` from
``actual_send_ts`` onward (prefill allocation approximated at send — the
engine-internal queueing delay before prefill is unobservable from our own
records and would only make our estimate an UPPER bound on true occupancy),
and grows ``num_tokens`` completion tokens LINEARLY between
``first_token_ts`` and ``completion_ts`` (linear decode growth between the
recorded endpoints). Preemption/eviction inside the engine is invisible to
this model, so ρ_own may exceed 1.0 under overload — that divergence from
the engine gauge is the SIGNAL this module exists to surface, never an error.

Fail-closed doctrine: an empty requests list refuses (absence of rows is not
zero occupancy); any required field absent/None on a request that must be
classified refuses, NAMING the field and the request (skip-and-underestimate
is the exact bug class this replaces); unknown model/kv_dtype propagates
``cache_budget.CacheBudgetError``. The ONE legal exclusion is a
``dropped_by_cap`` row: it was never sent, provably held zero engine bytes,
and is counted (``n_dropped_excluded``), never silent. Conversely a window
whose recorded intervals simply do not overlap the bounds yields an HONEST
ρ_own = 0.0 — that zero is affirmative accounting over known intervals, not
a default for missing data.

Domain logic only: stdlib, no I/O, no pandas requirement (the series is a
plain tuple of segment dicts regime tooling can frame).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from src.orchestration.cache_budget import (
    KV_DTYPE_FACTOR,
    MODEL_KV,
    CacheBudgetError,
)

__all__ = [
    "OwnAccountingError",
    "OwnOccupancyWindow",
    "OwnReuseWindow",
    "compute_own_occupancy",
    "compute_own_reuse",
    "divergence_report",
    "window_bounds_from_requests",
]

# requests.jsonl field names — pinned to the producer's row schema
# (scripts/3_run/run_experiment.py result dict + RequestRecord.to_row in
# src/orchestration/load_generator.py). Renaming there must rename here.
_SEND_FIELD = "actual_send_ts"
_FIRST_TOKEN_FIELD = "first_token_ts"
_COMPLETION_FIELD = "completion_ts"
_PROMPT_TOKENS_FIELD = "prompt_tokens"
_COMPLETION_TOKENS_FIELD = "num_tokens"
_DROPPED_FIELD = "dropped_by_cap"
_GROUP_FIELD = "group_id"
_CACHED_TOKENS_FIELD = "cached_prompt_tokens"


class OwnAccountingError(ValueError):
    """A request set that cannot be accounted honestly (fail closed)."""


def _request_label(row: Mapping[str, Any], index: int) -> str:
    example_id = row.get("example_id")
    record_index = row.get("record_index")
    parts = [f"row {index}"]
    if example_id:
        parts.append(f"example_id={example_id!r}")
    if record_index is not None:
        parts.append(f"record_index={record_index!r}")
    return " ".join(parts)


def _require_number(row: Mapping[str, Any], field: str, index: int) -> float:
    value = row.get(field)
    if value is None:
        raise OwnAccountingError(
            f"required field {field!r} is absent/None on in-window request "
            f"({_request_label(row, index)}) — own accounting never "
            "skips-and-underestimates; fix the producer row or exclude the "
            "window loudly"
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OwnAccountingError(
            f"field {field!r} = {value!r} on {_request_label(row, index)} "
            "must be a number"
        )
    value = float(value)
    if not math.isfinite(value):
        raise OwnAccountingError(
            f"field {field!r} = {value!r} on {_request_label(row, index)} "
            "must be finite"
        )
    return value


def _require_count(row: Mapping[str, Any], field: str, index: int) -> int:
    value = _require_number(row, field, index)
    if value != int(value) or value < 0:
        raise OwnAccountingError(
            f"field {field!r} = {value!r} on {_request_label(row, index)} "
            "must be a non-negative integer token count"
        )
    return int(value)


def _model_kv(model: str):
    try:
        return MODEL_KV[model]
    except KeyError:
        # Same refusal shape as cache_budget._model — the error type is the
        # contract ("the cache_budget error propagates"); the arithmetic
        # itself is imported, never copied.
        raise CacheBudgetError(
            f"unknown model {model!r} — known: {sorted(MODEL_KV)} (fail "
            "closed; add the charter-derived KV arithmetic before accounting)"
        ) from None


def _dtype_factor(kv_dtype: str) -> float:
    try:
        return KV_DTYPE_FACTOR[kv_dtype]
    except KeyError:
        raise CacheBudgetError(
            f"unknown kv_dtype {kv_dtype!r} — known: {sorted(KV_DTYPE_FACTOR)}"
        ) from None


@dataclass(frozen=True)
class _InFlight:
    """One parsed, validated in-flight request interval."""

    send: float
    first_token: float | None  # None <=> no decode segment (num == 0)
    end: float
    prompt_tokens: int
    completion_tokens: int


def _parse_in_flight(
    requests: Sequence[Mapping[str, Any]],
) -> tuple[list[_InFlight], int]:
    """Validate every row into an interval; returns (flights, n_dropped).

    ``dropped_by_cap`` rows are the ONE legal exclusion (never sent — zero
    engine bytes by construction) and are counted, not silent. Every other
    row must carry send/completion timestamps and token counts, and a
    ``first_token_ts`` whenever ``num_tokens > 0`` — absence refuses naming
    the field and the request.
    """
    if not requests:
        raise OwnAccountingError(
            "requests is empty: an empty accounting basis cannot certify "
            "occupancy — absence of rows is not zero (fail closed)"
        )
    flights: list[_InFlight] = []
    n_dropped = 0
    for i, row in enumerate(requests):
        if not isinstance(row, Mapping):
            raise OwnAccountingError(
                f"request row {i} is not a mapping ({type(row).__name__})"
            )
        if row.get(_DROPPED_FIELD):
            n_dropped += 1
            continue
        send = _require_number(row, _SEND_FIELD, i)
        end = _require_number(row, _COMPLETION_FIELD, i)
        if end < send:
            raise OwnAccountingError(
                f"{_COMPLETION_FIELD} {end!r} < {_SEND_FIELD} {send!r} on "
                f"{_request_label(row, i)} — the interval is unreconstructable"
            )
        prompt = _require_count(row, _PROMPT_TOKENS_FIELD, i)
        completion = _require_count(row, _COMPLETION_TOKENS_FIELD, i)
        first_token: float | None = None
        if completion > 0:
            first_token = _require_number(row, _FIRST_TOKEN_FIELD, i)
            if not (send <= first_token <= end):
                raise OwnAccountingError(
                    f"{_FIRST_TOKEN_FIELD} {first_token!r} outside "
                    f"[{_SEND_FIELD}={send!r}, {_COMPLETION_FIELD}={end!r}] "
                    f"on {_request_label(row, i)}"
                )
        flights.append(
            _InFlight(
                send=send,
                first_token=first_token,
                end=end,
                prompt_tokens=prompt,
                completion_tokens=completion,
            )
        )
    return flights, n_dropped


def _row_token_seconds(f: _InFlight, a: float, b: float) -> float:
    """Exact ∫ tokens(t) dt of one request over [a, b) ⊆ its clipped span.

    tokens(t) = prompt on [send, decode_start); prompt + num·(t−ft)/(end−ft)
    on [ft, end). A degenerate decode (ft == end, num > 0) materializes all
    completion tokens at the completion instant — zero measure, holds prompt
    only (documented approximation limit, never a crash).
    """
    total = 0.0
    has_decode = (
        f.completion_tokens > 0
        and f.first_token is not None
        and f.end > f.first_token
    )
    decode_start = f.first_token if has_decode else f.end
    # Constant prefill segment [send, decode_start)
    lo = max(f.send, a)
    hi = min(decode_start, b)
    if hi > lo:
        total += f.prompt_tokens * (hi - lo)
    if has_decode:
        assert f.first_token is not None
        lo = max(f.first_token, a)
        hi = min(f.end, b)
        if hi > lo:
            span = f.end - f.first_token
            total += f.prompt_tokens * (hi - lo)
            total += (
                f.completion_tokens
                * ((hi - f.first_token) ** 2 - (lo - f.first_token) ** 2)
                / (2.0 * span)
            )
    return total


def _row_seq_seconds(f: _InFlight, a: float, b: float) -> float:
    """Overlap duration of one request's [send, end) with [a, b) — the
    per-sequence fixed-state (hybrid recurrent slot) holding time."""
    lo = max(f.send, a)
    hi = min(f.end, b)
    return max(0.0, hi - lo)


def _tokens_at(flights: Sequence[_InFlight], t: float, *, side: str) -> float:
    """Sum of held tokens at instant ``t`` (piecewise-linear evaluation).

    ``side='right'`` counts requests with send <= t < end (value just after
    t); ``side='left'`` counts send < t <= end (value just before t) — the
    two one-sided limits a right-open segment series needs at its endpoints.
    """
    total = 0.0
    for f in flights:
        if side == "right":
            member = f.send <= t < f.end
        else:
            member = f.send < t <= f.end
        if not member:
            continue
        tokens = float(f.prompt_tokens)
        if f.completion_tokens > 0 and f.first_token is not None and f.end > f.first_token:
            pos = min(max(t - f.first_token, 0.0), f.end - f.first_token)
            tokens += f.completion_tokens * pos / (f.end - f.first_token)
        total += tokens
    return total


@dataclass(frozen=True)
class OwnOccupancyWindow:
    """One window's OWN-accounting occupancy (Layer-1 primary; engine gauge
    is corroboration, joined by :func:`divergence_report`).

    ``rho_own_time_avg`` = byte_seconds / (budget_bytes × window duration);
    the denominator is the FULL window duration — unlike the engine-telemetry
    bridge (regime_inputs), an uncovered span here is affirmatively empty
    (no recorded request in flight), so zero is data, not absence.
    ``series`` holds exact per-segment integrals between event breakpoints,
    each entry ``{t0_s, t1_s, bytes_t0, bytes_t1, bytes_time_avg,
    rho_time_avg}`` (bytes_t0/bytes_t1 are the one-sided endpoint values of
    the piecewise-linear bytes(t)) — small and directly consumable by regime
    tooling. May exceed rho 1.0 under overload (see module docstring).
    """

    model: str
    kv_dtype: str
    kv_bytes_per_token_effective: int
    fixed_state_bytes_per_seq: int
    budget_bytes: int
    window_start_s: float
    window_end_s: float
    byte_seconds: float
    bytes_time_avg: float
    rho_own_time_avg: float
    peak_bytes: float
    peak_rho: float
    n_requests: int
    n_in_flight: int
    n_dropped_excluded: int
    series: tuple[dict[str, float], ...]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping (series as a list)."""
        out = asdict(self)
        out["series"] = list(out["series"])
        return out


def compute_own_occupancy(
    requests: Sequence[Mapping[str, Any]],
    *,
    model: str,
    budget_bytes: int,
    window_start_s: float,
    window_end_s: float,
    kv_dtype: str = "bf16",
) -> OwnOccupancyWindow:
    """T2.5: own-accounting ρ_KV over [window_start_s, window_end_s).

    Reconstructs each request's in-flight interval [actual_send_ts,
    completion_ts) from the recorded requests.jsonl fields and integrates
    bytes(t) = Σ_in-flight (held tokens at t) × kv_bytes_per_token(model,
    kv_dtype) + fixed_state_bytes_per_seq (hybrid models) exactly
    (piecewise-linear closed form, no sampling). The token-holding model and
    its documented approximations live in the module docstring: prompt
    tokens held from send onward, completion tokens growing linearly between
    the recorded first-token and completion endpoints.

    Window bounds must be on the SAME clock as the request timestamps (the
    dispatcher's clock — see :func:`window_bounds_from_requests`).

    Fail-closed: empty ``requests``, a missing/None required field on any
    non-dropped row, bad bounds, ``budget_bytes < 1`` → OwnAccountingError;
    unknown ``model``/``kv_dtype`` → CacheBudgetError (from cache_budget's
    vocabulary, never a local copy).
    """
    mk = _model_kv(model)
    factor = _dtype_factor(kv_dtype)
    if isinstance(budget_bytes, bool) or not isinstance(budget_bytes, int):
        raise OwnAccountingError(
            f"budget_bytes={budget_bytes!r} must be an int (bytes)"
        )
    if budget_bytes < 1:
        raise OwnAccountingError(
            f"budget_bytes={budget_bytes} must be >= 1 — a non-positive "
            "budget has no occupancy ratio"
        )
    for name, value in (
        ("window_start_s", window_start_s),
        ("window_end_s", window_end_s),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise OwnAccountingError(f"{name}={value!r} must be a finite number")
    w0 = float(window_start_s)
    w1 = float(window_end_s)
    if w1 <= w0:
        raise OwnAccountingError(
            f"window_end_s={w1!r} must be > window_start_s={w0!r}"
        )

    flights, n_dropped = _parse_in_flight(requests)
    # floor() mirrors cache_budget.demand_bytes — the ONE bytes-per-token
    # arithmetic, applied identically on both sides of the referee.
    kv_per_token = math.floor(mk.kv_bytes_per_token * factor)

    in_flight = [f for f in flights if f.end > w0 and f.send < w1]

    byte_seconds = 0.0
    for f in in_flight:
        byte_seconds += _row_token_seconds(f, w0, w1) * kv_per_token
        byte_seconds += _row_seq_seconds(f, w0, w1) * mk.fixed_state_bytes_per_seq

    # Event breakpoints (clipped) — bytes(t) is linear between them.
    points = {w0, w1}
    for f in in_flight:
        for t in (f.send, f.first_token, f.end):
            if t is not None and w0 < t < w1:
                points.add(t)
    breakpoints = sorted(points)

    series: list[dict[str, float]] = []
    peak_bytes = 0.0
    for a, b in zip(breakpoints[:-1], breakpoints[1:]):
        seg_int = 0.0
        for f in in_flight:
            seg_int += _row_token_seconds(f, a, b) * kv_per_token
            seg_int += _row_seq_seconds(f, a, b) * mk.fixed_state_bytes_per_seq
        n_right = sum(1 for f in in_flight if f.send <= a < f.end)
        n_left = sum(1 for f in in_flight if f.send < b <= f.end)
        bytes_t0 = (
            _tokens_at(in_flight, a, side="right") * kv_per_token
            + n_right * mk.fixed_state_bytes_per_seq
        )
        bytes_t1 = (
            _tokens_at(in_flight, b, side="left") * kv_per_token
            + n_left * mk.fixed_state_bytes_per_seq
        )
        peak_bytes = max(peak_bytes, bytes_t0, bytes_t1)
        series.append(
            {
                "t0_s": a,
                "t1_s": b,
                "bytes_t0": bytes_t0,
                "bytes_t1": bytes_t1,
                "bytes_time_avg": seg_int / (b - a),
                "rho_time_avg": seg_int / (b - a) / budget_bytes,
            }
        )

    duration = w1 - w0
    bytes_time_avg = byte_seconds / duration
    return OwnOccupancyWindow(
        model=model,
        kv_dtype=kv_dtype,
        kv_bytes_per_token_effective=kv_per_token,
        fixed_state_bytes_per_seq=mk.fixed_state_bytes_per_seq,
        budget_bytes=budget_bytes,
        window_start_s=w0,
        window_end_s=w1,
        byte_seconds=byte_seconds,
        bytes_time_avg=bytes_time_avg,
        rho_own_time_avg=bytes_time_avg / budget_bytes,
        peak_bytes=peak_bytes,
        peak_rho=peak_bytes / budget_bytes,
        n_requests=len(requests),
        n_in_flight=len(in_flight),
        n_dropped_excluded=n_dropped,
        series=tuple(series),
    )


def window_bounds_from_requests(
    requests: Sequence[Mapping[str, Any]],
) -> tuple[float, float]:
    """[min send, max completion) over the non-dropped rows — the window on
    the DISPATCHER'S clock.

    Exists because requests.jsonl timestamps are ``time.monotonic`` while
    cell.json's t_start/t_end are epoch ``time.time()`` — mixing the two
    clocks would silently misplace every interval, so callers without an
    epoch↔monotonic bridge derive bounds from the rows themselves (the
    window's requests.jsonl holds exactly that window's trial). Fail-closed:
    same field requirements as :func:`compute_own_occupancy`; zero-width or
    all-dropped input refuses.
    """
    flights, _ = _parse_in_flight(requests)
    if not flights:
        raise OwnAccountingError(
            "every request row is dropped_by_cap — no sent request, no "
            "derivable window bounds"
        )
    start = min(f.send for f in flights)
    end = max(f.end for f in flights)
    if end <= start:
        raise OwnAccountingError(
            f"derived window [{start!r}, {end!r}) is zero-width — the "
            "recorded intervals cannot bound a window"
        )
    return start, end


def divergence_report(
    rho_own: float, rho_engine: float | None
) -> dict[str, float | None]:
    """Own-vs-engine gauge divergence: {rho_own, rho_engine, abs_gap, rel_gap}.

    ``rho_own`` is the PRIMARY (this module's accounting) and must be a
    finite number. ``rho_engine`` is the demoted corroboration gauge; when
    the engine did not report it, pass None — the gaps come back None (a gap
    against an absent gauge is unknown, NEVER 0). A NaN ``rho_engine``
    refuses: NaN reaching this seam means an upstream failed to be
    None-honest, and laundering it into None here would hide that bug.
    ``rel_gap`` = abs_gap / rho_own (relative to the primary); None when
    rho_own == 0 (undefined, never inf).
    """
    if (
        isinstance(rho_own, bool)
        or not isinstance(rho_own, (int, float))
        or not math.isfinite(float(rho_own))
    ):
        raise OwnAccountingError(
            f"rho_own={rho_own!r} must be a finite number — it is the "
            "primary accounting value, not an optional gauge"
        )
    rho_own = float(rho_own)
    if rho_engine is None:
        return {
            "rho_own": rho_own,
            "rho_engine": None,
            "abs_gap": None,
            "rel_gap": None,
        }
    if (
        isinstance(rho_engine, bool)
        or not isinstance(rho_engine, (int, float))
        or not math.isfinite(float(rho_engine))
    ):
        raise OwnAccountingError(
            f"rho_engine={rho_engine!r} must be a finite number or None "
            "(absence is None, never NaN — fix the upstream that minted this)"
        )
    rho_engine = float(rho_engine)
    abs_gap = abs(rho_own - rho_engine)
    return {
        "rho_own": rho_own,
        "rho_engine": rho_engine,
        "abs_gap": abs_gap,
        "rel_gap": (abs_gap / rho_own) if rho_own > 0 else None,
    }


@dataclass(frozen=True)
class OwnReuseWindow:
    """One window's OWN-accounting reuse (T8.2).

    ``rho_reuse_own`` is the token-weighted mean shared-prefix fraction —
    Σ shared_prefix_tokens / Σ prompt_tokens — a STRUCTURAL workload
    property from the manifest (what COULD be reused by a prefix-caching
    engine), deliberately independent of what any engine's cache actually
    hit. ``cached_tokens_corroboration`` summarizes the engine's
    ``cached_prompt_tokens`` self-reports over the SAME rows (token-weighted
    over reporting rows only; ``token_weighted_mean`` is None when no row
    reported — absence is never 0) — the corroboration column, never the
    primary.
    """

    rho_reuse_own: float
    total_prompt_tokens: int
    total_shared_prefix_tokens: int
    n_requests: int
    cached_tokens_corroboration: dict[str, Any]
    per_request: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["per_request"] = list(out["per_request"])
        return out


def compute_own_reuse(
    requests: Sequence[Mapping[str, Any]],
    manifest_prefix_tokens: Mapping[Any, int] | int,
) -> OwnReuseWindow:
    """T8.2: per-request + window shared-prefix reuse from the manifest.

    ``manifest_prefix_tokens`` is either a mapping ``group -> shared-prefix
    token count`` keyed by each row's ``group_id`` (for the corpus-block
    workloads this is ``{block_id: blocks[block_id].token_count}`` from the
    query manifest built by ``src/data/manifest.py`` — the manifest DOES
    record a per-block ``token_count``, though note it is the MANIFEST
    tokenizer's count of the block text, a documented approximation of the
    engine-tokenized prefix length), or a scalar token count applying to
    every request (single-shared-document workloads).

    [WAVE-3 wiring] The manifest records block token counts but no
    per-request resolved shared-prefix column; until the producer stamps one
    into requests.jsonl, callers build the mapping from the manifest file
    (see run_campaign_analysis's own-accounting pass) — this function stays
    input-explicit on purpose.

    Fail-closed: empty requests; a row without ``prompt_tokens`` (>= 1) or —
    in mapping mode — without a ``group_id`` present in the mapping (supply
    an explicit 0 for prefix-free groups, absence is not zero); a prefix
    count exceeding the row's prompt tokens (an unphysical reuse fraction
    > 1 signals manifest/engine tokenizer divergence and must surface, not
    round); a non-integer/negative ``cached_prompt_tokens`` (None is legal
    absence). All raise OwnAccountingError naming field and request.
    """
    if not requests:
        raise OwnAccountingError(
            "requests is empty: an empty accounting basis cannot certify "
            "reuse — absence of rows is not zero (fail closed)"
        )
    scalar_mode = isinstance(manifest_prefix_tokens, int) and not isinstance(
        manifest_prefix_tokens, bool
    )
    if scalar_mode:
        if manifest_prefix_tokens < 0:
            raise OwnAccountingError(
                f"manifest_prefix_tokens={manifest_prefix_tokens} must be >= 0"
            )
    elif not isinstance(manifest_prefix_tokens, Mapping):
        raise OwnAccountingError(
            "manifest_prefix_tokens must be a mapping (group -> tokens) or "
            f"an int scalar, got {type(manifest_prefix_tokens).__name__}"
        )

    per_request: list[dict[str, Any]] = []
    total_prompt = 0
    total_prefix = 0
    cached_prompt_reported = 0
    cached_tokens_reported = 0
    n_cached_reported = 0
    for i, row in enumerate(requests):
        if not isinstance(row, Mapping):
            raise OwnAccountingError(
                f"request row {i} is not a mapping ({type(row).__name__})"
            )
        prompt = _require_count(row, _PROMPT_TOKENS_FIELD, i)
        if prompt < 1:
            raise OwnAccountingError(
                f"{_PROMPT_TOKENS_FIELD}={prompt} on {_request_label(row, i)} "
                "must be >= 1 — a zero-token prompt has no reuse denominator"
            )
        if scalar_mode:
            prefix = int(manifest_prefix_tokens)  # type: ignore[arg-type]
            group = None
        else:
            group = row.get(_GROUP_FIELD)
            if group is None:
                raise OwnAccountingError(
                    f"required field {_GROUP_FIELD!r} is absent/None on "
                    f"{_request_label(row, i)} — mapping-mode reuse joins the "
                    "manifest through the request's group"
                )
            if group not in manifest_prefix_tokens:
                known = sorted(map(repr, list(manifest_prefix_tokens)[:5]))
                raise OwnAccountingError(
                    f"group {group!r} ({_request_label(row, i)}) has no "
                    "manifest shared-prefix entry (sample of known keys: "
                    f"{known}) — supply an explicit 0 for prefix-free groups; "
                    "skipping would underestimate reuse"
                )
            prefix_raw = manifest_prefix_tokens[group]
            if (
                isinstance(prefix_raw, bool)
                or not isinstance(prefix_raw, int)
                or prefix_raw < 0
            ):
                raise OwnAccountingError(
                    f"manifest prefix tokens for group {group!r} must be a "
                    f"non-negative int, got {prefix_raw!r}"
                )
            prefix = prefix_raw
        if prefix > prompt:
            raise OwnAccountingError(
                f"shared_prefix_tokens={prefix} > {_PROMPT_TOKENS_FIELD}="
                f"{prompt} on {_request_label(row, i)} — a reuse fraction "
                "> 1 is unphysical (manifest/engine tokenizer divergence); "
                "refusing to clamp"
            )
        cached = row.get(_CACHED_TOKENS_FIELD)
        if cached is not None:
            if (
                isinstance(cached, bool)
                or not isinstance(cached, (int, float))
                or float(cached) != int(cached)
                or int(cached) < 0
            ):
                raise OwnAccountingError(
                    f"{_CACHED_TOKENS_FIELD}={cached!r} on "
                    f"{_request_label(row, i)} must be a non-negative "
                    "integer or None (None = engine did not report)"
                )
            cached = int(cached)
            n_cached_reported += 1
            cached_tokens_reported += cached
            cached_prompt_reported += prompt
        total_prompt += prompt
        total_prefix += prefix
        per_request.append(
            {
                "example_id": row.get("example_id"),
                "record_index": row.get("record_index"),
                "group_id": group,
                "prompt_tokens": prompt,
                "shared_prefix_tokens": prefix,
                "reuse_frac": prefix / prompt,
                "cached_prompt_tokens": cached,
            }
        )

    corroboration = {
        "n_reported": n_cached_reported,
        "n_missing": len(per_request) - n_cached_reported,
        # Token-weighted over REPORTING rows only — a non-reporting row must
        # not drag the corroboration toward 0.
        "token_weighted_mean": (
            cached_tokens_reported / cached_prompt_reported
            if n_cached_reported > 0
            else None
        ),
    }
    return OwnReuseWindow(
        rho_reuse_own=total_prefix / total_prompt,
        total_prompt_tokens=total_prompt,
        total_shared_prefix_tokens=total_prefix,
        n_requests=len(per_request),
        cached_tokens_corroboration=corroboration,
        per_request=tuple(per_request),
    )
