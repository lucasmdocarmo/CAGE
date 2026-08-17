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

Registered-completeness enforcement (2026-08-16 owner decisions, assertion
G5): the caller may bind the chain to the REGISTERED expectations —
``registered_sets`` (endpoint → its registered co-primary legs; a missing
registered leg FAILS the set with an explicit reason, it never shrinks the
conjunction), ``registered_family_sizes`` (family_id → registered Holm m; a
family missing members is corrected at the REGISTERED m, which is
conservative, and the shortfall is flagged on every decision), and
``upstream_by_family`` (family_id → the registered upstream endpoint from the
§9.3 map's ``upstream`` column; a wired upstream contradicting it fails
loud). All three default to None so the flat/exploratory re-analysis path is
unchanged; the REGISTERED campaign analysis must pass them from the map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

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
    """``m_registered``/``m_supplied`` are set only when the caller passed
    ``registered_family_sizes`` (the registered path): ``m_supplied <
    m_registered`` flags a family missing registered members — its Holm
    correction ran at the REGISTERED m (conservative; see
    ``_holm_at_registered_m``), never at the shrunken invocation size."""

    contrast: str
    family_id: str
    dataset: str
    p_value: float
    status: SecondaryStatus
    p_holm: float | None
    significant: bool | None
    m_supplied: int | None = None
    m_registered: int | None = None


@dataclass(frozen=True)
class SetDecision:
    """Set-level verdict for one primary endpoint's co-primary set (§9.1).

    When ``registered_legs`` is not None and ``missing_legs`` is non-empty,
    ``passed`` is False BY CONSTRUCTION (G5, 2026-08-16 owner decision): a
    registered leg with no outcome cannot reject, and absence of evidence
    must fail the conjunction loudly — never shrink it to the supplied legs.
    """

    endpoint: str
    rule: IntraSetRule
    passed: bool
    binding_p: float
    supplied_legs: tuple[str, ...]
    registered_legs: tuple[str, ...] | None
    missing_legs: tuple[str, ...]
    reason: str


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
    set_decisions: tuple[SetDecision, ...] = ()

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
                "m_supplied": [s.m_supplied for s in self.secondaries],
                "m_registered": [s.m_registered for s in self.secondaries],
            }
        )


def _check_p(owner: str, p: float) -> None:
    if not np.isfinite(p) or not 0.0 <= p <= 1.0:
        raise GatekeepingError(f"{owner}: p_value={p!r} must lie in [0, 1]")


def _set_passed(
    members: list[PrimaryOutcome],
    rule: IntraSetRule,
    alpha: float,
    *,
    registered_legs: Sequence[str] | None = None,
) -> tuple[bool, float]:
    """Intra-set verdict for one endpoint's per-dataset co-primary set.

    Returns (passed, binding_p) — binding_p is the p that decided the verdict
    (max raw p under "all-datasets"; min Holm-adjusted p under "holm-any").
    With ``registered_legs`` (G5, 2026-08-16): any registered leg absent from
    the supplied members FAILS the set outright — the verdict may not shrink
    to the supplied conjunction/disjunction.
    """
    if registered_legs is not None:
        missing = set(registered_legs) - {m.dataset for m in members}
        if missing:
            # binding_p over the supplied legs is still reported for the
            # audit trail, but the verdict is failed by construction.
            if rule == "all-datasets":
                binding = max(m.p_value for m in members)
            else:
                binding = float(min(holm([m.p_value for m in members])))
            return False, binding
    if rule == "all-datasets":
        binding = max(m.p_value for m in members)
        return binding < alpha, binding
    adjusted = holm([m.p_value for m in members])
    binding = float(min(adjusted))
    return binding < alpha, binding


def _decide_set(
    endpoint: str,
    members: list[PrimaryOutcome],
    rule: IntraSetRule,
    alpha: float,
    registered_legs: Sequence[str] | None,
) -> SetDecision:
    """Full auditable set-level verdict for one endpoint (§9.1/G5)."""
    supplied = tuple(sorted(m.dataset for m in members))
    if registered_legs is None:
        passed, binding = _set_passed(members, rule, alpha)
        return SetDecision(
            endpoint=endpoint, rule=rule, passed=passed, binding_p=binding,
            supplied_legs=supplied, registered_legs=None, missing_legs=(),
            reason=(
                f"no registered set expectation supplied (flat/exploratory "
                f"path): {'passed' if passed else 'failed'} the {rule!r} rule "
                f"over the supplied legs at alpha={alpha}"
            ),
        )
    registered = tuple(sorted(str(leg) for leg in registered_legs))
    if not registered:
        raise GatekeepingError(
            f"endpoint {endpoint!r}: registered co-primary set is empty — a "
            f"registered set must name its legs (§9.1)"
        )
    extra = sorted(set(supplied) - set(registered))
    if extra:
        raise GatekeepingError(
            f"endpoint {endpoint!r}: supplied legs {extra} are not in the "
            f"registered co-primary set {list(registered)} — an unregistered "
            f"leg cannot enter the confirmatory set (§9.1/G5)"
        )
    missing = tuple(sorted(set(registered) - set(supplied)))
    passed, binding = _set_passed(
        members, rule, alpha, registered_legs=registered
    )
    if missing:
        reason = (
            f"co-primary set FAILED: registered legs {list(missing)} have no "
            f"outcome — absence is not evidence, and the set may not shrink "
            f"to its supplied legs {list(supplied)} (G5, 2026-08-16)"
        )
    else:
        reason = (
            f"complete registered set {list(registered)}: "
            f"{'passed' if passed else 'failed'} the {rule!r} rule at "
            f"alpha={alpha}"
        )
    return SetDecision(
        endpoint=endpoint, rule=rule, passed=passed, binding_p=binding,
        supplied_legs=supplied, registered_legs=registered,
        missing_legs=missing, reason=reason,
    )


def _holm_at_registered_m(p_values: list[float], m_registered: int) -> np.ndarray:
    """Holm step-down at the REGISTERED family size (G5b, 2026-08-16).

    Missing members are padded with p=1.0, which places them LAST in the
    step-down order, so every supplied p receives its worst-case multiplier
    ``m_registered - rank + 1``. Any true completion of the family (the
    missing p-values, had they been computed) would sort the missing members
    at the same rank or earlier, giving supplied members equal or SMALLER
    adjusted p — the registered-m correction is therefore conservative, which
    is why missing members flag-and-correct rather than fail the family: the
    shrunken-m alternative (the G5 defect) was anti-conservative, and a
    hard failure would discard supplied evidence the registration covers.
    """
    if m_registered < len(p_values):
        raise GatekeepingError(
            f"registered family size {m_registered} < {len(p_values)} "
            f"supplied members — an unregistered member cannot join a Holm "
            f"family (§9.3)"
        )
    padded = list(p_values) + [1.0] * (m_registered - len(p_values))
    return holm(padded)[: len(p_values)]


def evaluate_chain(
    primaries: Sequence[PrimaryOutcome],
    secondaries: Sequence[SecondaryOutcome],
    *,
    alpha: float = 0.05,
    primary_order: Sequence[str] | None = None,
    intra_set_rule: IntraSetRule = "all-datasets",
    registered_sets: Mapping[str, Sequence[str]] | None = None,
    registered_family_sizes: Mapping[str, int] | None = None,
    upstream_by_family: Mapping[str, str] | None = None,
) -> GatekeepingTrace:
    """Evaluate the §9.3 chain and return the full auditable trace.

    ``primary_order`` is the registered Dmitrienko serial sequence of primary
    ENDPOINT names; it must name exactly the supplied endpoints (no more, no
    less) — a mismatch fails loud, because a chain order that silently drops
    or invents an endpoint is a registration error. Every secondary's
    (upstream, dataset) must name a supplied primary — a dangling reference
    fails loud. All members of one family must share one (upstream, dataset)
    gate: a family straddling gates is a map error.

    Registered-completeness bindings (G5, 2026-08-16 — all optional, None
    keeps the flat/exploratory behavior; the REGISTERED campaign analysis
    passes all three from the §9.3 map):

    - ``registered_sets``: endpoint → the registered co-primary legs (the
      trace's ``dataset`` keys). A registered leg with no supplied outcome
      FAILS the endpoint's set verdict with an explicit reason — the
      conjunction never shrinks to the supplied legs. A supplied leg outside
      the registered set fails loud. Keys must be supplied endpoints (the
      caller reconciles whole missing endpoints itself — chain_complete).
    - ``registered_family_sizes``: family_id → registered Holm member count.
      Every supplied family must be sized; Holm runs at the REGISTERED m
      (conservative — see ``_holm_at_registered_m``) and a shortfall is
      flagged on the decisions and the gate event. More supplied members
      than registered fails loud.
    - ``upstream_by_family``: family_id → the registered upstream endpoint
      (the map's ``upstream`` column). Every supplied family must appear; a
      member wired to a different upstream fails loud (the topology is
      registered, never the caller's choice — assertion G10).
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

    if registered_sets is not None:
        unknown_endpoints = set(registered_sets) - set(by_endpoint)
        if unknown_endpoints:
            raise GatekeepingError(
                f"registered_sets names endpoints {sorted(unknown_endpoints)} "
                f"with no supplied outcomes — reconcile whole missing "
                f"endpoints explicitly (chain_complete), never here"
            )

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
    set_decisions: dict[str, SetDecision] = {}
    events: list[GateEvent] = []
    chain_open = True
    previous_endpoint: str | None = None
    for endpoint in ordered_endpoints:
        members = by_endpoint[endpoint]
        set_decision = _decide_set(
            endpoint, members, intra_set_rule, alpha,
            registered_sets.get(endpoint) if registered_sets is not None else None,
        )
        set_decisions[endpoint] = set_decision
        status: SecondaryStatus = "confirmatory" if chain_open else "descriptive"
        if primary_order is not None and previous_endpoint is not None:
            previous_set = set_decisions[previous_endpoint]
            incomplete_note = (
                f" [upstream set incomplete: registered legs "
                f"{list(previous_set.missing_legs)} missing — failed by "
                f"construction (G5)]"
                if previous_set.missing_legs
                else ""
            )
            events.append(
                GateEvent(
                    family_id=f"primary-chain:{endpoint}",
                    upstream=previous_endpoint,
                    dataset="ALL",
                    upstream_p=previous_set.binding_p,
                    opened=chain_open,
                    reason=(
                        f"serial gate: endpoint {endpoint!r} "
                        f"{'tested confirmatorily' if chain_open else 'labeled descriptive'} "
                        f"— upstream chain through {previous_endpoint!r} "
                        f"{'passed' if chain_open else 'failed'} the "
                        f"{intra_set_rule!r} intra-set rule at alpha={alpha}"
                        f"{incomplete_note}"
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
            chain_open = set_decision.passed
        previous_endpoint = endpoint

    families: dict[str, list[SecondaryOutcome]] = {}
    for s in secondaries:
        _check_p(f"secondary {s.contrast} ({s.family_id})", s.p_value)
        if upstream_by_family is not None:
            registered_up = upstream_by_family.get(s.family_id)
            if registered_up is None:
                raise GatekeepingError(
                    f"family {s.family_id!r} has no registered upstream in "
                    f"upstream_by_family — the §9.3 map must declare the "
                    f"gating topology for every gated family (G10)"
                )
            if s.upstream != registered_up:
                raise GatekeepingError(
                    f"secondary {s.contrast!r}: wired upstream {s.upstream!r} "
                    f"contradicts the registered upstream {registered_up!r} "
                    f"for family {s.family_id!r} (§9.3 registered topology, "
                    f"2026-08-16 decision)"
                )
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
        # G5b (2026-08-16): Holm m comes from the REGISTERED family size —
        # validated for every family (open or closed) so a registration error
        # fails loud regardless of gate state.
        m_registered: int | None = None
        if registered_family_sizes is not None:
            if family_id not in registered_family_sizes:
                raise GatekeepingError(
                    f"family {family_id!r} has no registered member count in "
                    f"registered_family_sizes — the registered map must size "
                    f"every Holm family (G5)"
                )
            m_registered = int(registered_family_sizes[family_id])
            if m_registered < 1:
                raise GatekeepingError(
                    f"family {family_id!r}: registered size {m_registered} "
                    f"must be >= 1"
                )
            if m_registered < len(members):
                raise GatekeepingError(
                    f"family {family_id!r} supplies {len(members)} members "
                    f"but registers only {m_registered} — an unregistered "
                    f"member cannot join a Holm family (§9.3)"
                )
        m_supplied: int | None = (
            len(members) if m_registered is not None else None
        )
        shortfall_note = ""
        if m_registered is not None and m_registered > len(members):
            shortfall_note = (
                f" [Holm at REGISTERED m={m_registered}; "
                f"{m_registered - len(members)} registered member(s) missing "
                f"— corrected at the registered m, which is conservative "
                f"(G5b)]"
            )
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
                    f"{shortfall_note}"
                ),
            )
        )
        if opened:
            p_list = [m.p_value for m in members]
            adjusted = (
                _holm_at_registered_m(p_list, m_registered)
                if m_registered is not None
                else holm(p_list)
            )
            for m, p_adj in zip(members, adjusted):
                secondary_decisions.append(
                    SecondaryDecision(
                        contrast=m.contrast, family_id=family_id,
                        dataset=m.dataset, p_value=m.p_value,
                        status="confirmatory", p_holm=float(p_adj),
                        significant=bool(p_adj < alpha),
                        m_supplied=m_supplied, m_registered=m_registered,
                    )
                )
        else:
            for m in members:
                secondary_decisions.append(
                    SecondaryDecision(
                        contrast=m.contrast, family_id=family_id,
                        dataset=m.dataset, p_value=m.p_value,
                        status="descriptive", p_holm=None, significant=None,
                        m_supplied=m_supplied, m_registered=m_registered,
                    )
                )

    return GatekeepingTrace(
        alpha=alpha,
        primaries=tuple(primary_decisions),
        secondaries=tuple(secondary_decisions),
        events=tuple(events),
        primary_order=tuple(ordered_endpoints) if primary_order is not None else None,
        intra_set_rule=intra_set_rule,
        set_decisions=tuple(set_decisions[e] for e in ordered_endpoints),
    )
