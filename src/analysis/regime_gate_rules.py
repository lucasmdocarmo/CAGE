"""Purpose: the profile grammar the regime gate (scripts/6_experiments/regime_gate.py,
stage 14) and the plan-time projection (run_campaign.py regime-gate-projection,
stage 0) share, so both read REGIME_GATE_LOADS, MIN_IN_REGIME_WINDOWS and
CONFIRMATORY_ARMS the same way (S2 block F row F-10, 2026-11-03; ADR-0185).
Stdlib only: the gate runs under the Mac's bare python3.

Grammar:
  REGIME_GATE_LOADS      whitespace-separated tokens; "X+" gates every rate
                         fraction at or above X (the rule before F-10 is "1.0+",
                         the default), "X" gates exactly X (S2: "0.95 1.05"); a
                         window takes the first token that matches it, so a
                         token at or above an earlier "X+" (never reached) and a
                         rate named twice are refused (review L2).
  MIN_IN_REGIME_WINDOWS  "" = report only; "N" = one floor for every gated
                         load; "TOKEN:N ..." = one floor per gated token, every
                         token named once, no token outside the loads.
  CONFIRMATORY_ARMS      "" = every group blocks; "B1 B6 ..." = the baselines
                         whose groups block, the others are descriptive; a cell
                         whose baseline cannot be read, or is not a B id, blocks
                         (fail closed; review M1).

Invariants: a parsed load is finite and > 0; a parsed floor is >= 1 per gated
token; a roster id matches ``B<number>``. Every refusal is a RulesError naming
the key and the offending text (the gate exits 2 on it, the planner refuses).
Complexity: O(tokens).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple

DEFAULT_LOADS = "1.0+"
PRESSURE_FAMILIES: Tuple[str, ...] = ("F2", "F3")
#: the gate's group key, in the order the key string is joined with "|"
GROUP_KEY_FIELDS: Tuple[str, ...] = (
    "engine", "arm", "retriever", "policy", "topology", "dataset", "budget_r", "rate_frac",
)
BLOCKING = "BLOCKING"
DESCRIPTIVE = "DESCRIPTIVE"
BASELINE_ID_RE = re.compile(r"^B\d+$")


class RulesError(ValueError):
    """A profile string the gate cannot read (exit 2 at the CLI, a refusal in the planner)."""


@dataclass(frozen=True)
class GatedLoad:
    value: float
    at_or_above: bool
    token: str


def parse_loads(text: Optional[str]) -> Tuple[GatedLoad, ...]:
    """REGIME_GATE_LOADS -> the gated loads, in the order written."""
    tokens = (text or "").split()
    if not tokens:
        raise RulesError(
            "REGIME_GATE_LOADS is empty: name the gated loads ('1.0+' = at or above 1.0; "
            "'0.95 1.05' = exactly those)"
        )
    out: List[GatedLoad] = []
    for token in tokens:
        at_or_above = token.endswith("+")
        raw = token[:-1] if at_or_above else token
        try:
            value = float(raw)
        except ValueError:
            raise RulesError(
                f"REGIME_GATE_LOADS token {token!r} is not a rate fraction ('X' or 'X+')"
            ) from None
        if not math.isfinite(value) or value <= 0:
            raise RulesError(f"REGIME_GATE_LOADS token {token!r} must be a finite rate fraction > 0")
        for earlier in out:
            if earlier.value == value:
                raise RulesError(f"REGIME_GATE_LOADS names the rate {value:g} twice ({earlier.token!r}, {token!r})")
            if earlier.at_or_above and value >= earlier.value:
                raise RulesError(
                    f"REGIME_GATE_LOADS token {token!r} is shadowed by {earlier.token!r} "
                    f"(every rate at or above {earlier.value:g} takes that token first)"
                )
        out.append(GatedLoad(value, at_or_above, token))
    return tuple(out)


def load_token_for(rate_frac: Optional[float], loads: Iterable[GatedLoad]) -> Optional[str]:
    """The first gated load a window's rate fraction matches; None = not gated."""
    if rate_frac is None:
        return None
    for g in loads:
        if (g.at_or_above and rate_frac >= g.value) or (not g.at_or_above and rate_frac == g.value):
            return g.token
    return None


def _floor_int(raw: str) -> int:
    try:
        n = int(raw)
    except ValueError:
        raise RulesError(f"MIN_IN_REGIME_WINDOWS={raw!r} is not an integer") from None
    if n < 1:
        raise RulesError(f"MIN_IN_REGIME_WINDOWS={n} must be >= 1 (empty = report only)")
    return n


def parse_floor(text: Any, loads: Iterable[GatedLoad]) -> Optional[Dict[str, int]]:
    """MIN_IN_REGIME_WINDOWS -> {gated token: floor}; None = report only."""
    s = "" if text is None else str(text).strip()
    if not s:
        return None
    tokens = [g.token for g in loads]
    if ":" not in s:
        n = _floor_int(s)
        return {t: n for t in tokens}
    floors: Dict[str, int] = {}
    for item in s.split():
        token, sep, raw = item.partition(":")
        if not sep:
            raise RulesError(f"MIN_IN_REGIME_WINDOWS item {item!r} is not 'load:N' (the per-load form)")
        if token not in tokens:
            raise RulesError(
                f"MIN_IN_REGIME_WINDOWS names load {token!r}, not one of REGIME_GATE_LOADS {tokens}"
            )
        if token in floors:
            raise RulesError(f"MIN_IN_REGIME_WINDOWS names load {token!r} twice")
        floors[token] = _floor_int(raw)
    missing = [t for t in tokens if t not in floors]
    if missing:
        raise RulesError(
            f"MIN_IN_REGIME_WINDOWS has no floor for gated load(s) {missing} (REGIME_GATE_LOADS {tokens})"
        )
    return floors


def parse_roster(text: Optional[str]) -> FrozenSet[str]:
    """CONFIRMATORY_ARMS -> the baseline ids whose groups block; empty = every group."""
    ids = (text or "").split()
    bad = [i for i in ids if not BASELINE_ID_RE.match(i)]
    if bad:
        raise RulesError(f"CONFIRMATORY_ARMS names {bad}: baseline ids are B<number> (B1 ... B12)")
    return frozenset(ids)


def baseline_id_or_none(raw: Any) -> Optional[str]:
    """A cell record's ``baseline`` as a B id, or None when it is missing, empty
    or not of that form (review M1: "" read as a known id outside the roster)."""
    text = "" if raw is None else str(raw)
    return text if BASELINE_ID_RE.match(text) else None


def mode_for(baseline: Optional[str], roster: FrozenSet[str]) -> str:
    """BLOCKING when no roster is set, when the baseline is unknown (fail closed)
    or when it is in the roster; DESCRIPTIVE otherwise."""
    if not roster or baseline is None or baseline in roster:
        return BLOCKING
    return DESCRIPTIVE
