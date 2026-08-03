"""The §9.3 gatekeeping chain (PUBLICATION.md D9; audit §2.1/§2.2).

Primaries are tested at full α per dataset (co-primaries never pooled, §9.1)
**in the pre-registered Dmitrienko SERIAL fixed sequence** supplied as
``primary_order``: an endpoint is tested confirmatorily only while every
earlier endpoint in the order has passed its declared intra-set rule; once an
endpoint fails, every later primary — and every family gated on it — is
labeled ``descriptive``. The co-primary SET rule (``intra_set_rule``) is the
set's own declared intra-set multiplicity handling (audit §2.1):

- ``"all-datasets"`` (default): the endpoint passes the chain step only if
  EVERY dataset in its set rejects at full α (conjunctive co-primaries, ICH
  E9 — no α splitting needed).
- ``"holm-any"``: Holm across the set's datasets; the endpoint passes if any
  member survives (disjunctive claim, α controlled within the set).

``primary_order=None`` skips the serial gate (all primaries confirmatory) and
exists for exploratory re-analysis only — the REGISTERED campaign analysis
must pass the §9.3 order (``families.PRIMARY_CHAIN_ORDER``).

A secondary family is tested confirmatorily — Holm within the family — ONLY
if its upstream primary was itself confirmatory AND passed on the same
dataset; otherwise every member is labeled ``descriptive`` (raw p reported,
no significance claim, no α spent). The returned trace records which gate
opened or closed what, so the analysis is auditable row by row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from src.analysis.stats.corrections import holm

SecondaryStatus = Literal["confirmatory", "descriptive"]
IntraSetRule = Literal["all-datasets", "holm-any"]


class GatekeepingError(ValueError):
    """Malformed chain input (dangling upstream, duplicates, bad p)."""


@dataclass(frozen=True)
class PrimaryOutcome:
    """One primary endpoint result on one dataset (input)."""

    endpoint: str
    dataset: str
    p_value: float


@dataclass(frozen=True)
class SecondaryOutcome:
    """One secondary contrast result, member of a Holm family (input)."""

    contrast: str
    family_id: str
    upstream: str
    dataset: str
    p_value: float


@dataclass(frozen=True)
class PrimaryDecision:
    """``status="descriptive"`` marks a primary downstream of a failed serial
    gate: its p is reported but it was never tested confirmatorily, so
    ``passed`` is False by construction."""

    endpoint: str
    dataset: str
    p_value: float
    alpha: float
    passed: bool
    status: SecondaryStatus = "confirmatory"


@dataclass(frozen=True)
class SecondaryDecision:
    contrast: str
    family_id: str
    dataset: str
    p_value: float
    status: SecondaryStatus
    p_holm: float | None
    significant: bool | None


@dataclass(frozen=True)
class GateEvent:
    family_id: str
    upstream: str
    dataset: str
    upstream_p: float
    opened: bool
    reason: str


@dataclass(frozen=True)
class GatekeepingTrace:
    alpha: float
    primaries: tuple[PrimaryDecision, ...]
    secondaries: tuple[SecondaryDecision, ...]
    events: tuple[GateEvent, ...]
    primary_order: tuple[str, ...] | None = None
    intra_set_rule: IntraSetRule = "all-datasets"

    def to_frame(self) -> pd.DataFrame:
        """Secondary decisions as a reporting table (one row per contrast)."""
        return pd.DataFrame(
            {
                "contrast": [s.contrast for s in self.secondaries],
                "family_id": [s.family_id for s in self.secondaries],
                "dataset": [s.dataset for s in self.secondaries],
                "p_value": [s.p_value for s in self.secondaries],
                "status": [s.status for s in self.secondaries],
                "p_holm": [s.p_holm for s in self.secondaries],
                "significant": [s.significant for s in self.secondaries],
            }
        )


def _check_p(owner: str, p: float) -> None:
    if not np.isfinite(p) or not 0.0 <= p <= 1.0:
        raise GatekeepingError(f"{owner}: p_value={p!r} must lie in [0, 1]")


def _set_passed(
    members: list[PrimaryOutcome], rule: IntraSetRule, alpha: float
) -> tuple[bool, float]:
    """Intra-set verdict for one endpoint's per-dataset co-primary set.

    Returns (passed, binding_p) — binding_p is the p that decided the verdict
    (max raw p under "all-datasets"; min Holm-adjusted p under "holm-any").
    """
    if rule == "all-datasets":
        binding = max(m.p_value for m in members)
        return binding < alpha, binding
    adjusted = holm([m.p_value for m in members])
    binding = float(min(adjusted))
    return binding < alpha, binding


def evaluate_chain(
    primaries: Sequence[PrimaryOutcome],
    secondaries: Sequence[SecondaryOutcome],
    *,
    alpha: float = 0.05,
    primary_order: Sequence[str] | None = None,
    intra_set_rule: IntraSetRule = "all-datasets",
) -> GatekeepingTrace:
    """Evaluate the §9.3 chain and return the full auditable trace.

    ``primary_order`` is the registered Dmitrienko serial sequence of primary
    ENDPOINT names; it must name exactly the supplied endpoints (no more, no
    less) — a mismatch fails loud, because a chain order that silently drops
    or invents an endpoint is a registration error. Every secondary's
    (upstream, dataset) must name a supplied primary — a dangling reference
    fails loud. All members of one family must share one (upstream, dataset)
    gate: a family straddling gates is a map error.
    """
    if not 0.0 < alpha < 1.0:
        raise GatekeepingError(f"alpha={alpha} must be in (0, 1)")
    if not primaries:
        raise GatekeepingError("no primary outcomes supplied — nothing can gate")
    if intra_set_rule not in ("all-datasets", "holm-any"):
        raise GatekeepingError(
            f"intra_set_rule={intra_set_rule!r}; allowed: 'all-datasets', 'holm-any'"
        )

    by_endpoint: dict[str, list[PrimaryOutcome]] = {}
    seen: set[tuple[str, str]] = set()
    for p in primaries:
        _check_p(f"primary {p.endpoint}/{p.dataset}", p.p_value)
        key = (p.endpoint, p.dataset)
        if key in seen:
            raise GatekeepingError(
                f"duplicate primary outcome for endpoint={p.endpoint!r} "
                f"dataset={p.dataset!r}"
            )
        seen.add(key)
        by_endpoint.setdefault(p.endpoint, []).append(p)

    if primary_order is None:
        ordered_endpoints = list(by_endpoint)
    else:
        order = [str(e) for e in primary_order]
        if len(set(order)) != len(order):
            raise GatekeepingError(f"primary_order has duplicates: {order}")
        if set(order) != set(by_endpoint):
            raise GatekeepingError(
                f"primary_order {order} must name exactly the supplied "
                f"endpoints {sorted(by_endpoint)} (Dmitrienko serial, §9.3)"
            )
        ordered_endpoints = order

    primary_decisions: list[PrimaryDecision] = []
    gate_state: dict[tuple[str, str], PrimaryDecision] = {}
    events: list[GateEvent] = []
    chain_open = True
    previous_endpoint: str | None = None
    for endpoint in ordered_endpoints:
        members = by_endpoint[endpoint]
        status: SecondaryStatus = "confirmatory" if chain_open else "descriptive"
        if primary_order is not None and previous_endpoint is not None:
            events.append(
                GateEvent(
                    family_id=f"primary-chain:{endpoint}",
                    upstream=previous_endpoint,
                    dataset="ALL",
                    upstream_p=_set_passed(
                        by_endpoint[previous_endpoint], intra_set_rule, alpha
                    )[1],
                    opened=chain_open,
                    reason=(
                        f"serial gate: endpoint {endpoint!r} "
                        f"{'tested confirmatorily' if chain_open else 'labeled descriptive'} "
                        f"— upstream chain through {previous_endpoint!r} "
                        f"{'passed' if chain_open else 'failed'} the "
                        f"{intra_set_rule!r} intra-set rule at alpha={alpha}"
                    ),
                )
            )
        for m in members:
            decision = PrimaryDecision(
                endpoint=m.endpoint, dataset=m.dataset, p_value=m.p_value,
                alpha=alpha,
                passed=chain_open and m.p_value < alpha,
                status=status,
            )
            gate_state[(m.endpoint, m.dataset)] = decision
            primary_decisions.append(decision)
        if primary_order is not None and chain_open:
            chain_open = _set_passed(members, intra_set_rule, alpha)[0]
        previous_endpoint = endpoint

    families: dict[str, list[SecondaryOutcome]] = {}
    for s in secondaries:
        _check_p(f"secondary {s.contrast} ({s.family_id})", s.p_value)
        if (s.upstream, s.dataset) not in gate_state:
            raise GatekeepingError(
                f"secondary {s.contrast!r} names upstream primary "
                f"({s.upstream!r}, {s.dataset!r}) which was not supplied"
            )
        families.setdefault(s.family_id, []).append(s)

    secondary_decisions: list[SecondaryDecision] = []
    for family_id, members in families.items():
        gates = {(m.upstream, m.dataset) for m in members}
        if len(gates) != 1:
            raise GatekeepingError(
                f"family {family_id!r} straddles gates {sorted(gates)}; one "
                f"family = one (upstream, dataset) gate (§9.3)"
            )
        upstream, dataset = next(iter(gates))
        gate = gate_state[(upstream, dataset)]
        # A primary made descriptive by the serial chain has passed=False by
        # construction, so its families never open.
        opened = gate.passed and gate.status == "confirmatory"
        gate_desc = (
            f"{'passed' if opened else 'failed'} at full alpha={alpha}"
            if gate.status == "confirmatory"
            else "was itself descriptive (serial chain closed upstream)"
        )
        events.append(
            GateEvent(
                family_id=family_id, upstream=upstream, dataset=dataset,
                upstream_p=gate.p_value, opened=opened,
                reason=(
                    f"primary {upstream!r} on {dataset!r} {gate_desc} "
                    f"(p={gate.p_value:.6g}) -> family {family_id!r} "
                    f"{'tested confirmatorily (Holm)' if opened else 'labeled descriptive'}"
                ),
            )
        )
        if opened:
            adjusted = holm([m.p_value for m in members])
            for m, p_adj in zip(members, adjusted):
                secondary_decisions.append(
                    SecondaryDecision(
                        contrast=m.contrast, family_id=family_id,
                        dataset=m.dataset, p_value=m.p_value,
                        status="confirmatory", p_holm=float(p_adj),
                        significant=bool(p_adj < alpha),
                    )
                )
        else:
            for m in members:
                secondary_decisions.append(
                    SecondaryDecision(
                        contrast=m.contrast, family_id=family_id,
                        dataset=m.dataset, p_value=m.p_value,
                        status="descriptive", p_holm=None, significant=None,
                    )
                )

    return GatekeepingTrace(
        alpha=alpha,
        primaries=tuple(primary_decisions),
        secondaries=tuple(secondary_decisions),
        events=tuple(events),
        primary_order=tuple(ordered_endpoints) if primary_order is not None else None,
        intra_set_rule=intra_set_rule,
    )
