"""Refusal-arm coverage for the §7.8/§9.3 family registry guards (K-COV9, #142).

test_stats_engine.py pins the compiled map's shape and compile_family_map's
input refusals; the REGISTRY validation guard ``_validated`` (families.py:194+)
— the fail-closed wall that stops a future registry edit from silently
corrupting the registered contrast set — had never been exercised on a bad
registry. These tests mutate copies of the REAL registry (dataclasses.replace)
so each guard fires against otherwise-valid content, plus the
``family_unit_of`` / ``registered_upstream`` refusal arms.

Pure offline: no data, no network — registry logic only.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.analysis.stats.families import (
    CHAIN_ENDPOINTS,
    CONTRASTS,
    ContrastLeg,
    FamilyMapError,
    FINGERPRINT_CONTRAST_ID,
    FLOOR_SUITE_CONTRAST_ID,
    PRIMARY_CHAIN_ORDER,
    PRIMARY_IDS,
    UNGATED,
    _validated,
    chain_endpoint,
    family_unit_of,
    registered_upstream,
)


def _mutated(contrast_id: int, **changes) -> tuple:
    """The real registry with ONE contrast replaced (id keys the slot)."""
    return tuple(
        dataclasses.replace(c, **changes) if c.id == contrast_id else c
        for c in CONTRASTS
    )


# --------------------------------------------------------------------------- #
# _validated: the registry wall
# --------------------------------------------------------------------------- #


class TestRegistryValidation:
    def test_real_registry_passes(self):
        assert _validated(CONTRASTS) is CONTRASTS

    def test_dropped_contrast_refused(self):
        with pytest.raises(FamilyMapError, match="exactly contrasts 1-20"):
            _validated(CONTRASTS[:-1])

    def test_out_of_order_ids_refused(self):
        with pytest.raises(FamilyMapError, match="exactly contrasts 1-20"):
            _validated(tuple(reversed(CONTRASTS)))

    def test_contrast_21_refused_as_future_work(self):
        extra = dataclasses.replace(CONTRASTS[-1], id=21)
        with pytest.raises(FamilyMapError, match="#21 is future"):
            _validated(CONTRASTS + (extra,))

    def test_unknown_baseline_refused(self):
        with pytest.raises(FamilyMapError, match="unknown baseline 'B99'"):
            _validated(_mutated(1, baseline_a="B99"))

    def test_empty_groups_refused(self):
        with pytest.raises(FamilyMapError, match="no carrying group"):
            _validated(_mutated(2, groups=()))

    def test_unknown_group_refused(self):
        with pytest.raises(FamilyMapError, match=r"unknown groups \['E'\]"):
            _validated(_mutated(2, groups=("A", "E")))

    def test_primary_tier_restricted_to_chain_ids(self):
        # Only 4/13/14 may carry tier='primary' (§9.1; floor suite exiled).
        with pytest.raises(FamilyMapError, match="only the three §9.1 chain primaries"):
            _validated(_mutated(1, tier="primary"))

    def test_falsification_tier_restricted_to_floor_suite(self):
        with pytest.raises(FamilyMapError, match="only the floor-±15% suite"):
            _validated(_mutated(16, tier="falsification", gatekept=False))

    @pytest.mark.parametrize("tier", ["exploratory", "falsification"])
    def test_gatekept_outside_the_chain_refused(self, tier):
        target = FLOOR_SUITE_CONTRAST_ID if tier == "falsification" else 11
        with pytest.raises(FamilyMapError, match="outside the.*confirmatory chain"):
            _validated(_mutated(target, gatekept=True))

    def test_empty_pinned_metrics_refused(self):
        with pytest.raises(FamilyMapError, match="pinned metrics must be non-empty"):
            _validated(_mutated(14, metrics=()))

    def test_extra_leg_unknown_baseline_refused(self):
        bad_leg = ContrastLeg(
            baseline_a="B99", baseline_b="B3", slot="arm (x)", family="F3",
            groups=("A",),
        )
        with pytest.raises(FamilyMapError, match="unknown baseline 'B99'"):
            _validated(_mutated(15, extra_legs=(bad_leg,)))

    def test_extra_leg_empty_groups_refused(self):
        bad_leg = ContrastLeg(
            baseline_a="B12", baseline_b="B3", slot="arm (x)", family="F3",
            groups=(),
        )
        with pytest.raises(FamilyMapError, match="no carrying group"):
            _validated(_mutated(15, extra_legs=(bad_leg,)))


# --------------------------------------------------------------------------- #
# family_unit_of: the G19 unit axis
# --------------------------------------------------------------------------- #


class TestFamilyUnitAxis:
    def test_window_stays_window(self):
        assert family_unit_of("window") == "window"

    @pytest.mark.parametrize("unit", ["per_query", "binary"])
    def test_binary_collapses_onto_per_query(self, unit):
        # McNemar pairs the same §9.4 per-query draws Wilcoxon does.
        assert family_unit_of(unit) == "per_query"

    def test_unknown_unit_refused(self):
        with pytest.raises(FamilyMapError, match="unknown test unit 'trial'"):
            family_unit_of("trial")


# --------------------------------------------------------------------------- #
# registered_upstream: the G10 gating topology
# --------------------------------------------------------------------------- #


class TestRegisteredUpstream:
    @pytest.mark.parametrize("tier", ["primary", "exploratory", "falsification"])
    def test_non_secondary_tiers_are_ungated(self, tier):
        assert registered_upstream(tier, "F1", "per_query") == UNGATED
        assert registered_upstream(tier, "F2", "window") == UNGATED

    def test_per_query_secondary_gates_on_the_headline(self):
        assert registered_upstream("secondary", "F1", "per_query") == "contrast-4"
        # Binary rides the per-query axis (G19), so it shares the gate.
        assert registered_upstream("secondary", "F1", "binary") == "contrast-4"

    def test_window_secondaries_gate_tier_naturally(self):
        assert registered_upstream("secondary", "F2", "window") == "contrast-14"
        assert registered_upstream("secondary", "F3", "window") == (
            chain_endpoint(FINGERPRINT_CONTRAST_ID)
        )
        assert registered_upstream("secondary", "DIST", "window") == "contrast-13"

    def test_window_secondary_without_registered_family_refused(self):
        # F1 has no registered window-secondary upstream: fail closed rather
        # than silently wiring the row to a gate that can never open.
        with pytest.raises(FamilyMapError, match="no registered.*upstream"):
            registered_upstream("secondary", "F1", "window")

    def test_chain_endpoints_cover_exactly_the_primary_chain(self):
        assert CHAIN_ENDPOINTS == {f"contrast-{cid}" for cid in PRIMARY_CHAIN_ORDER}
        assert set(PRIMARY_CHAIN_ORDER) == PRIMARY_IDS
