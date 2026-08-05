"""Tests for the D9 statistical test engine (src/analysis/stats, §9.3-§9.5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from src.analysis.stats.corrections import benjamini_hochberg, holm
from src.analysis.stats.equivalence import (
    ConditionalTostResult,
    conditional_tost,
    rope_sensitivity,
)
from src.analysis.stats.families import (
    CONTRASTS,
    DEFAULT_METRICS,
    FINGERPRINT_SUB_HYPOTHESES,
    FamilyMapError,
    HEADLINE_CONTRAST_ID,
    PRIMARY_CHAIN_ORDER,
    PRIMARY_IDS,
    compile_family_map,
)
from src.analysis.stats.gatekeeping import (
    GatekeepingError,
    PrimaryOutcome,
    SecondaryOutcome,
    evaluate_chain,
)
from src.analysis.stats.tests_by_unit import (
    batch_means_contrast,
    mcnemar_binary,
    paired_wilcoxon,
)
from src.analysis.stats.wlt import win_loss_tie

DATASETS = ["squad_v2", "hotpotqa", "musique", "qasper"]


# --------------------------------------------------------------------------- #
# families.py
# --------------------------------------------------------------------------- #
class TestFamilies:
    def test_registry_holds_contrasts_1_to_20_only(self) -> None:
        assert [c.id for c in CONTRASTS] == list(range(1, 21))

    def test_headline_is_b6_vs_b3_two_sided_primary(self) -> None:
        c4 = CONTRASTS[HEADLINE_CONTRAST_ID - 1]
        assert (c4.baseline_a, c4.baseline_b) == ("B6", "B3")
        assert c4.tier == "primary"
        assert c4.sidedness == "two-sided"
        assert c4.gatekept

    def test_three_chain_primaries_and_floor_exile(self) -> None:
        primaries = {c.id for c in CONTRASTS if c.tier == "primary"}
        assert primaries == PRIMARY_IDS == {4, 13, 14}
        floor = CONTRASTS[11]
        # §9.1/§9.2 exile (2026-08-02): the floor suite is NOT a primary —
        # it is a standalone falsification tier that spends no α.
        assert floor.id == 12
        assert floor.tier == "falsification"
        assert not floor.gatekept
        assert all(CONTRASTS[i - 1].gatekept for i in (4, 13, 14))

    def test_primary_chain_order_declared(self) -> None:
        # Dmitrienko serial sequence declared as data (audit §2.1).
        assert PRIMARY_CHAIN_ORDER == (4, 14, 13)
        assert set(PRIMARY_CHAIN_ORDER) == set(PRIMARY_IDS)

    def test_units_follow_family_pressure_split(self) -> None:
        for c in CONTRASTS:
            if c.family == "F1":
                assert c.unit == "per_query", c.id
            else:  # F2/F3/DIST are loaded -> window batch means (§9.4)
                assert c.unit == "window", c.id

    def test_exploratory_contrasts_are_outside_the_chain(self) -> None:
        for c in CONTRASTS:
            if c.tier == "exploratory":
                assert not c.gatekept, c.id

    def test_family_map_expansion_and_corrections(self) -> None:
        fm = compile_family_map(DATASETS)
        assert set(fm["correction"].unique()) == {"none", "holm", "bh-fdr", "tost"}
        tiers = fm.drop_duplicates("contrast_id").set_index("contrast_id")["tier"]
        # chain primaries #4/#14 at full alpha; #13 carries its own holm/tost
        # decomposition (tested separately); floor suite spends no alpha
        assert set(fm.loc[fm["contrast_id"].isin([4, 14]), "correction"]) == {"none"}
        assert set(fm.loc[fm["tier"] == "secondary", "correction"]) == {"holm"}
        assert set(fm.loc[fm["tier"] == "exploratory", "correction"]) == {"bh-fdr"}
        assert set(fm.loc[fm["tier"] == "falsification", "correction"]) == {"none"}
        assert tiers.loc[4] == "primary"
        # headline expands per dataset (co-primary SET, pooling prohibited)
        headline = fm[fm["contrast_id"] == 4]
        assert set(headline["dataset"]) == set(DATASETS)
        assert set(headline["group"]) == {"A", "B", "C", "D"}
        # family membership rule: group x metric x dataset
        row = headline.iloc[0]
        assert row["family_id"] == f"{row['group']}|{row['metric']}|{row['dataset']}"

    def test_family_map_unit_binary_for_sub_pressure_predicate(self) -> None:
        fm = compile_family_map(DATASETS)
        f1_predicate = fm[(fm["contrast_id"] == 4) & (fm["metric"] == "predicate")]
        assert set(f1_predicate["unit"]) == {"binary"}  # McNemar (§9.4)
        f1_ttft = fm[(fm["contrast_id"] == 4) & (fm["metric"] == "ttft")]
        assert set(f1_ttft["unit"]) == {"per_query"}
        loaded = fm[fm["contrast_id"] == 14]
        assert set(loaded["unit"]) == {"window"}

    def test_family_map_cross_dataset_slot(self) -> None:
        fm = compile_family_map(DATASETS)
        locality = fm[fm["contrast_id"] == 11]
        assert set(locality["dataset"]) == {"cross-dataset"}
        assert len(locality) == 2 * len(DEFAULT_METRICS)  # groups A,B x metrics

    def test_family_map_row_count(self) -> None:
        fm = compile_family_map(DATASETS)
        expected = 0
        for c in CONTRASTS:
            legs = [(c.slot, c.groups)]
            legs.extend((leg.slot, leg.groups) for leg in c.extra_legs)
            if c.id == 13:
                per_cell = len(FINGERPRINT_SUB_HYPOTHESES)
            elif c.metrics is not None:
                per_cell = len(c.metrics)
            else:
                per_cell = len(DEFAULT_METRICS)
            for slot, groups in legs:
                n_ds = 1 if slot == "dataset" else len(DATASETS)
                expected += len(groups) * per_cell * n_ds
        assert len(fm) == expected

    def test_fingerprint_decomposition_registers_tost_rows(self) -> None:
        # §9.3: 6 sub-hypotheses — Holm for the 3 superiority predictions,
        # TOST for the 3 pre-registered NONE predictions; the generic metric
        # pair never applies to #13.
        fm = compile_family_map(DATASETS)
        fp = fm[fm["contrast_id"] == 13]
        assert set(fp["metric"]) == {"fingerprint"}
        for _, cell in fp.groupby(["group", "dataset"]):
            assert len(cell) == 6
            assert (cell["correction"] == "holm").sum() == 3
            assert (cell["correction"] == "tost").sum() == 3
        assert (fm["correction"] == "tost").any()  # the 2026-08-02 regression
        assert (fp["sub_hypothesis"] != "").all()
        assert set(fp.loc[fp["correction"] == "tost", "sidedness"]) == {
            "two one-sided (TOST)"
        }
        assert set(fp.loc[fp["correction"] == "holm", "sidedness"]) == {"one-sided"}

    def test_primary_estimand_metrics_are_pinned(self) -> None:
        # §9.2: #14's registered variable is truth_tax = G − Y; #12's is the
        # λ* onset — never the generic ttft/predicate pair.
        fm = compile_family_map(DATASETS)
        assert set(fm.loc[fm["contrast_id"] == 14, "metric"]) == {"truth_tax"}
        assert set(fm.loc[fm["contrast_id"] == 12, "metric"]) == {"lambda_star_onset"}
        spurious = fm[
            fm["contrast_id"].isin([12, 14]) & fm["metric"].isin(DEFAULT_METRICS)
        ]
        assert spurious.empty

    def test_b12_vs_b3_leg_is_registered(self) -> None:
        # §7.8 #15 (2026-08-02): B12-vs-B3 moved to F3, both reuse-ON — it
        # must be registered ROWS, not a prose note.
        fm = compile_family_map(DATASETS)
        leg = fm[fm["comparison"] == "B12 vs B3"]
        assert not leg.empty
        assert set(leg["contrast_id"]) == {15}
        assert set(leg["family"]) == {"F3"}
        assert set(leg["group"]) == {"A", "B"}
        f2_leg = fm[(fm["contrast_id"] == 15) & (fm["comparison"] == "B11 vs B6")]
        assert set(f2_leg["family"]) == {"F2"}

    def test_family_map_rejects_bad_input(self) -> None:
        with pytest.raises(FamilyMapError, match="non-empty"):
            compile_family_map([])
        with pytest.raises(FamilyMapError, match="unknown datasets"):
            compile_family_map(["squad_v2", "sharegpt"])
        with pytest.raises(FamilyMapError, match="duplicate"):
            compile_family_map(["squad_v2", "squad_v2"])
        with pytest.raises(FamilyMapError, match="alpha"):
            compile_family_map(DATASETS, alpha=1.5)


# --------------------------------------------------------------------------- #
# corrections.py
# --------------------------------------------------------------------------- #
class TestCorrections:
    def test_holm_reference_values(self) -> None:
        # statsmodels multipletests(method="holm") reference
        adj = holm([0.01, 0.02, 0.03, 0.04])
        np.testing.assert_allclose(adj, [0.04, 0.06, 0.06, 0.06])

    def test_holm_unsorted_input_keeps_order(self) -> None:
        adj = holm([0.04, 0.01, 0.03, 0.02])
        np.testing.assert_allclose(adj, [0.06, 0.04, 0.06, 0.06])

    def test_bh_matches_scipy_reference(self) -> None:
        rng = np.random.default_rng(11)
        p = rng.uniform(0, 1, 40)
        np.testing.assert_allclose(
            benjamini_hochberg(p), sps.false_discovery_control(p, method="bh")
        )

    def test_bh_reference_values(self) -> None:
        np.testing.assert_allclose(
            benjamini_hochberg([0.01, 0.02, 0.03, 0.04]), [0.04, 0.04, 0.04, 0.04]
        )

    def test_clipping_and_edges(self) -> None:
        assert holm([0.9, 0.8]).max() == 1.0
        assert holm([]).size == 0
        assert benjamini_hochberg([]).size == 0
        np.testing.assert_allclose(holm([0.03]), [0.03])

    @pytest.mark.parametrize("bad", [[1.2], [-0.1], [np.nan], [np.inf]])
    def test_invalid_pvalues_raise(self, bad: list[float]) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            holm(bad)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            benjamini_hochberg(bad)

    def test_accepts_pandas_series(self) -> None:
        adj = holm(pd.Series([0.01, 0.5]))
        np.testing.assert_allclose(adj, [0.02, 0.5])


# --------------------------------------------------------------------------- #
# tests_by_unit.py — paired_wilcoxon
# --------------------------------------------------------------------------- #
class TestPairedWilcoxon:
    def test_matches_scipy(self) -> None:
        rng = np.random.default_rng(7)
        a = rng.normal(0.3, 1.0, 60)
        b = rng.normal(0.0, 1.0, 60)
        res = paired_wilcoxon(a, b)
        ref = sps.wilcoxon(a - b, zero_method="wilcox")
        assert res.p_value == pytest.approx(float(ref.pvalue))
        assert res.statistic == pytest.approx(float(ref.statistic))

    def test_one_sided_alternative(self) -> None:
        rng = np.random.default_rng(8)
        a = rng.normal(0.5, 1.0, 40)
        b = rng.normal(0.0, 1.0, 40)
        res = paired_wilcoxon(a, b, alternative="greater")
        ref = sps.wilcoxon(a - b, zero_method="wilcox", alternative="greater")
        assert res.p_value == pytest.approx(float(ref.pvalue))

    def test_paired_delta_is_within_pair_not_between_groups(self) -> None:
        # The audited defect: every pair improves by 0.5 but between-query
        # spread (~2 units) swamps it. Unpaired Cliff's delta ~ 0.02; the
        # PAIRED dominance is exactly 1.0.
        base = np.linspace(0.0, 100.0, 50)
        a = base + 0.5
        b = base.copy()
        res = paired_wilcoxon(a, b)
        assert res.cliffs_delta_paired == 1.0
        gt = sum(int(np.sum(x > b)) for x in a)
        lt = sum(int(np.sum(x < b)) for x in a)
        unpaired = (gt - lt) / (len(a) * len(b))
        assert abs(unpaired) < 0.1

    def test_delta_is_tie_aware(self) -> None:
        a = np.array([1.0, 1.0, 1.0, 0.0, 5.0, 5.0])
        b = np.array([0.0, 0.0, 0.0, 1.0, 5.0, 5.0])
        res = paired_wilcoxon(a, b)
        assert res.cliffs_delta_paired == pytest.approx((3 - 1) / 6)
        assert (res.n_pos, res.n_neg, res.n_zero) == (3, 1, 2)

    def test_all_zero_diffs_is_identity_outcome(self) -> None:
        a = np.ones(20)
        res = paired_wilcoxon(a, a)
        assert res.p_value == 1.0
        assert res.cliffs_delta_paired == 0.0
        assert res.n_zero == 20

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="differ in length"):
            paired_wilcoxon([1.0, 2.0], [1.0])
        with pytest.raises(ValueError, match="non-finite"):
            paired_wilcoxon([1.0, np.nan], [1.0, 2.0])
        with pytest.raises(ValueError, match="alternative"):
            paired_wilcoxon([1.0, 2.0], [0.0, 1.0], alternative="up")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# tests_by_unit.py — mcnemar_binary
# --------------------------------------------------------------------------- #
class TestMcNemar:
    def test_exact_binomial_on_discordant_pairs(self) -> None:
        a = np.array([1] * 8 + [0] * 2 + [1] * 20 + [0] * 10, dtype=bool)
        b = np.array([0] * 8 + [1] * 2 + [1] * 20 + [0] * 10, dtype=bool)
        res = mcnemar_binary(a, b)
        assert (res.n_10, res.n_01, res.n_11, res.n_00) == (8, 2, 20, 10)
        assert res.n_discordant == 10
        assert res.p_value == pytest.approx(
            float(sps.binomtest(8, 10, 0.5).pvalue)
        )
        assert res.proportion_diff == pytest.approx(6 / 40)

    def test_zero_discordant_pairs(self) -> None:
        a = np.array([True, False, True])
        res = mcnemar_binary(a, a)
        assert res.n_discordant == 0
        assert res.p_value == 1.0

    def test_accepts_01_ints_rejects_other_values(self) -> None:
        res = mcnemar_binary([1, 0, 1, 1], [0, 0, 1, 1])
        assert res.n_10 == 1
        with pytest.raises(ValueError, match="0/1"):
            mcnemar_binary([1, 2, 0], [0, 1, 0])
        with pytest.raises(ValueError, match="0/1"):
            mcnemar_binary([1.0, np.nan, 0.0], [0.0, 1.0, 0.0])

    def test_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="differ in length"):
            mcnemar_binary([1, 0], [1])


# --------------------------------------------------------------------------- #
# tests_by_unit.py — batch_means_contrast
# --------------------------------------------------------------------------- #
class TestBatchMeans:
    def test_matches_scipy_welch(self) -> None:
        a = np.array([10.2, 9.8, 10.5, 10.1])
        b = np.array([8.1, 8.4, 7.9])
        res = batch_means_contrast(a, b)
        ref = sps.ttest_ind(a, b, equal_var=False)
        assert res.p_value == pytest.approx(float(ref.pvalue))
        assert res.statistic == pytest.approx(float(ref.statistic))
        assert res.mean_diff == pytest.approx(a.mean() - b.mean())
        assert res.ci95_low < res.mean_diff < res.ci95_high

    def test_refuses_per_query_sized_input(self) -> None:
        per_query = np.random.default_rng(3).normal(size=500)
        windows = np.array([1.0, 1.1, 0.9])
        with pytest.raises(ValueError, match="PER-QUERY"):
            batch_means_contrast(per_query, windows)
        with pytest.raises(ValueError, match="PER-QUERY"):
            batch_means_contrast(windows, per_query)

    def test_requires_two_windows_per_side(self) -> None:
        with pytest.raises(ValueError, match="≥2 windows"):
            batch_means_contrast([1.0], [1.0, 2.0])

    def test_zero_variance_both_sides_raises(self) -> None:
        with pytest.raises(ValueError, match="zero variance"):
            batch_means_contrast([1.0, 1.0], [2.0, 2.0])

    def test_one_sided(self) -> None:
        a = np.array([10.0, 10.4, 9.6])
        b = np.array([8.0, 8.2, 7.8])
        res = batch_means_contrast(a, b, alternative="greater")
        ref = sps.ttest_ind(a, b, equal_var=False, alternative="greater")
        assert res.p_value == pytest.approx(float(ref.pvalue))


# --------------------------------------------------------------------------- #
# equivalence.py
# --------------------------------------------------------------------------- #
class TestConditionalTost:
    def test_pilot_shape_equivalent_and_discordant_count_surfaced(self) -> None:
        # 289 policy-touched queries, 15 discordant (the pilot's 15/289).
        d = np.zeros(289)
        d[:8] = 0.01
        d[8:15] = -0.01
        b = np.zeros(289)
        a = b + d
        mask = np.ones(289, dtype=bool)
        res = conditional_tost(a, b, mask, margin=0.02)
        assert res.n_events == 289
        assert res.n_discordant == 15
        assert res.domain_verdict == "equivalent"
        assert res.dominance == pytest.approx((8 - 7) / 289)
        assert res.dominance_verdict == "equivalent"
        assert res.equivalent

    def test_large_shift_fails_both_layers(self) -> None:
        rng = np.random.default_rng(5)
        b = rng.normal(0.0, 0.05, 100)
        a = b + 0.5
        mask = np.ones(100, dtype=bool)
        res = conditional_tost(a, b, mask, margin=0.1)
        assert res.domain_verdict == "not-equivalent"
        assert res.dominance == pytest.approx(1.0)
        assert res.dominance_verdict == "not-equivalent"
        assert not res.equivalent

    def test_dominance_layer_can_fail_alone(self) -> None:
        # Tiny but perfectly consistent harm: inside the domain margin,
        # dominance = 1.0 -> layer 2 must catch it.
        b = np.zeros(60)
        a = b + 0.001
        mask = np.ones(60, dtype=bool)
        res = conditional_tost(a, b, mask, margin=0.05)
        assert res.domain_verdict == "equivalent"
        assert res.dominance_verdict == "not-equivalent"
        assert not res.equivalent

    def test_mask_restricts_to_event_population(self) -> None:
        # Massive diffs OUTSIDE the event mask must not leak in.
        a = np.concatenate([np.zeros(30), np.full(30, 100.0)])
        b = np.zeros(60)
        mask = np.concatenate([np.ones(30, dtype=bool), np.zeros(30, dtype=bool)])
        res = conditional_tost(a, b, mask, margin=0.01)
        assert res.n_events == 30
        assert res.n_discordant == 0
        assert res.equivalent

    def test_all_ties_is_exact_equivalence(self) -> None:
        a = np.ones(50)
        mask = np.ones(50, dtype=bool)
        res = conditional_tost(a, a, mask, margin=0.01)
        assert res.p_tost == 0.0
        assert res.dominance == 0.0
        assert res.equivalent

    def test_insufficient_events_is_labeled_not_raised(self) -> None:
        a = np.zeros(100)
        mask = np.zeros(100, dtype=bool)
        mask[:5] = True
        res = conditional_tost(a, a, mask, margin=0.01, min_events=10)
        assert res.n_events == 5
        assert res.domain_verdict == "insufficient-n"
        assert res.dominance_verdict == "insufficient-n"
        assert not res.equivalent

    def test_seeded_bootstrap_is_deterministic(self) -> None:
        rng = np.random.default_rng(9)
        b = rng.normal(0, 1, 80)
        a = b + rng.normal(0, 0.01, 80)
        mask = np.ones(80, dtype=bool)
        r1 = conditional_tost(a, b, mask, margin=0.05, seed=42)
        r2 = conditional_tost(a, b, mask, margin=0.05, seed=42)
        assert r1 == r2
        assert isinstance(r1, ConditionalTostResult)

    def test_validation(self) -> None:
        a = np.zeros(10)
        with pytest.raises(ValueError, match="margin"):
            conditional_tost(a, a, np.ones(10, dtype=bool), margin=-1.0)
        with pytest.raises(ValueError, match="length"):
            conditional_tost(a, a, np.ones(5, dtype=bool), margin=0.1)
        with pytest.raises(ValueError, match="boolean"):
            conditional_tost(a, a, np.full(10, 2.0), margin=0.1)

    def test_dominance_ci_tracks_alpha_not_hardcoded(self) -> None:
        # Regression: the dominance-layer bootstrap CI used to be pinned to
        # the literals 2.5/97.5 (a fixed 95% CI, i.e. alpha=0.025) no matter
        # what `alpha` the caller passed. It must now be a
        # (1 - 2*alpha)*100% CI, matching the domain layer's own alpha.
        rng = np.random.default_rng(9)
        b = rng.normal(0, 1, 80)
        a = b + rng.normal(0, 0.01, 80)
        mask = np.ones(80, dtype=bool)
        wide = conditional_tost(a, b, mask, margin=0.05, alpha=0.01, seed=42)
        narrow = conditional_tost(a, b, mask, margin=0.05, alpha=0.1, seed=42)
        # A smaller alpha -> a wider (1 - 2*alpha) CI; a larger alpha -> a
        # narrower one. Before the fix both had IDENTICAL bounds.
        assert (wide.dominance_ci_high - wide.dominance_ci_low) > (
            narrow.dominance_ci_high - narrow.dominance_ci_low
        )
        assert narrow.dominance_ci_low >= wide.dominance_ci_low
        assert narrow.dominance_ci_high <= wide.dominance_ci_high

    def test_dominance_ci_matches_alpha_percentile_mapping(self) -> None:
        # Locks the alpha -> percentile mapping exactly: at alpha=0.05 the
        # dominance CI must equal the 5th/95th percentiles of the bootstrap
        # sign distribution (a 90% CI, i.e. two one-sided tests each at level
        # alpha) — NOT the previously hardcoded 2.5th/97.5th (a fixed 95% CI
        # that ignored `alpha` entirely).
        d = np.zeros(200)
        d[:40] = 0.01
        d[40:70] = -0.01
        a = d.copy()
        b = np.zeros(200)
        mask = np.ones(200, dtype=bool)
        seed = 42
        res = conditional_tost(a, b, mask, margin=0.05, alpha=0.05, seed=seed)

        signs = np.sign(d)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, 200, size=(10_000, 200))
        boot = signs[idx].mean(axis=1)
        expected_low = float(np.percentile(boot, 5))
        expected_high = float(np.percentile(boot, 95))
        assert res.dominance_ci_low == pytest.approx(expected_low)
        assert res.dominance_ci_high == pytest.approx(expected_high)

        # And NOT the stale hardcoded 2.5/97.5 mapping.
        stale_low = float(np.percentile(boot, 2.5))
        stale_high = float(np.percentile(boot, 97.5))
        assert res.dominance_ci_low != pytest.approx(stale_low)
        assert res.dominance_ci_high != pytest.approx(stale_high)

    def test_alpha_out_of_bounds_rejected(self) -> None:
        # alpha >= 0.5 would invert the (1 - 2*alpha)*100% CI (ci_low >
        # ci_high), silently corrupting the equivalence verdict.
        a = np.zeros(20)
        b = np.zeros(20)
        mask = np.ones(20, dtype=bool)
        for bad_alpha in (0.5, 0.6, 0.99):
            with pytest.raises(ValueError, match="alpha"):
                conditional_tost(a, b, mask, margin=0.1, alpha=bad_alpha)


# --------------------------------------------------------------------------- #
# equivalence.py — Bayesian ROPE sensitivity (§9.5)
# --------------------------------------------------------------------------- #
class TestRopeSensitivity:
    def test_tie_heavy_null_lands_in_rope(self) -> None:
        # Pilot-shaped: dominant exact ties, a couple of ±1 discordants inside
        # a wide domain margin -> posterior mass concentrates in the ROPE.
        a = np.zeros(100)
        a[:2] = 1.0
        b = np.zeros(100)
        b[2:4] = 1.0
        res = rope_sensitivity(a, b, np.ones(100, dtype=bool), rope=0.5, seed=7)
        assert res.p_rope > 0.95
        assert res.verdict == "equivalent"
        assert res.n_events == 100

    def test_large_shift_lands_right(self) -> None:
        a = np.ones(60)
        b = np.zeros(60)
        res = rope_sensitivity(a, b, np.ones(60, dtype=bool), rope=0.1, seed=7)
        assert res.p_right > 0.99
        assert res.verdict == "not-equivalent"
        assert res.mean_theta_right > 0.9

    def test_mask_restricts_population(self) -> None:
        # Off-mask rows carry a huge shift; masked-in rows are ties.
        a = np.concatenate([np.zeros(30), np.full(30, 5.0)])
        b = np.zeros(60)
        mask = np.concatenate([np.ones(30, dtype=bool), np.zeros(30, dtype=bool)])
        res = rope_sensitivity(a, b, mask, rope=0.5, seed=7)
        assert res.n_events == 30
        assert res.verdict == "equivalent"

    def test_deterministic_given_seed(self) -> None:
        rng = np.random.default_rng(3)
        a = rng.normal(0.0, 1.0, 40)
        b = rng.normal(0.0, 1.0, 40)
        mask = np.ones(40, dtype=bool)
        r1 = rope_sensitivity(a, b, mask, rope=0.3, seed=11)
        r2 = rope_sensitivity(a, b, mask, rope=0.3, seed=11)
        assert r1 == r2

    def test_insufficient_events_is_labeled(self) -> None:
        a = np.zeros(20)
        b = np.zeros(20)
        mask = np.zeros(20, dtype=bool)
        mask[:3] = True
        res = rope_sensitivity(a, b, mask, rope=0.5, min_events=10)
        assert res.verdict == "insufficient-n"
        assert np.isnan(res.p_rope)

    def test_validation(self) -> None:
        ones = np.ones(20)
        mask = np.ones(20, dtype=bool)
        with pytest.raises(ValueError, match="rope"):
            rope_sensitivity(ones, ones, mask, rope=0.0)
        with pytest.raises(ValueError, match="prior_pseudocount"):
            rope_sensitivity(ones, ones, mask, rope=0.1, prior_pseudocount=0.0)
        with pytest.raises(ValueError, match="n_samples"):
            rope_sensitivity(ones, ones, mask, rope=0.1, n_samples=10)
        with pytest.raises(ValueError, match="posterior_threshold"):
            rope_sensitivity(ones, ones, mask, rope=0.1, posterior_threshold=1.5)
        with pytest.raises(ValueError, match="differ in length"):
            rope_sensitivity(ones, np.ones(5), mask, rope=0.1)


# --------------------------------------------------------------------------- #
# gatekeeping.py
# --------------------------------------------------------------------------- #
class TestGatekeeping:
    @staticmethod
    def _chain() -> tuple[list[PrimaryOutcome], list[SecondaryOutcome]]:
        primaries = [
            PrimaryOutcome("B6-vs-B3", "squad_v2", 0.001),
            PrimaryOutcome("B6-vs-B3", "musique", 0.40),
        ]
        secondaries = [
            SecondaryOutcome("B5-vs-B6", "A|ttft|squad_v2", "B6-vs-B3", "squad_v2", 0.01),
            SecondaryOutcome("B9-vs-B6", "A|ttft|squad_v2", "B6-vs-B3", "squad_v2", 0.03),
            SecondaryOutcome("B5-vs-B6", "A|ttft|musique", "B6-vs-B3", "musique", 0.001),
        ]
        return primaries, secondaries

    def test_open_gate_tests_confirmatorily_with_holm(self) -> None:
        primaries, secondaries = self._chain()
        trace = evaluate_chain(primaries, secondaries)
        open_rows = [s for s in trace.secondaries if s.family_id == "A|ttft|squad_v2"]
        assert all(s.status == "confirmatory" for s in open_rows)
        by_contrast = {s.contrast: s for s in open_rows}
        np.testing.assert_allclose(
            [by_contrast["B5-vs-B6"].p_holm, by_contrast["B9-vs-B6"].p_holm],
            holm([0.01, 0.03]),
        )
        assert by_contrast["B5-vs-B6"].significant is True

    def test_closed_gate_labels_descriptive_even_tiny_p(self) -> None:
        primaries, secondaries = self._chain()
        trace = evaluate_chain(primaries, secondaries)
        closed = [s for s in trace.secondaries if s.family_id == "A|ttft|musique"]
        assert len(closed) == 1
        assert closed[0].status == "descriptive"
        assert closed[0].p_holm is None
        assert closed[0].significant is None  # p=0.001 buys nothing behind a closed gate

    def test_trace_events_are_auditable(self) -> None:
        primaries, secondaries = self._chain()
        trace = evaluate_chain(primaries, secondaries)
        events = {e.family_id: e for e in trace.events}
        assert events["A|ttft|squad_v2"].opened is True
        assert events["A|ttft|musique"].opened is False
        assert "descriptive" in events["A|ttft|musique"].reason
        assert events["A|ttft|musique"].upstream_p == 0.40
        frame = trace.to_frame()
        assert set(frame["status"]) == {"confirmatory", "descriptive"}

    def test_primaries_at_full_alpha_per_dataset(self) -> None:
        primaries, secondaries = self._chain()
        trace = evaluate_chain(primaries, secondaries)
        passed = {(p.dataset): p.passed for p in trace.primaries}
        assert passed == {"squad_v2": True, "musique": False}
        assert all(p.alpha == 0.05 for p in trace.primaries)

    def test_dangling_upstream_fails_loud(self) -> None:
        primaries, _ = self._chain()
        bad = [SecondaryOutcome("x", "f", "NOT-A-PRIMARY", "squad_v2", 0.01)]
        with pytest.raises(GatekeepingError, match="not supplied"):
            evaluate_chain(primaries, bad)

    def test_family_straddling_gates_fails_loud(self) -> None:
        primaries, _ = self._chain()
        bad = [
            SecondaryOutcome("x", "fam", "B6-vs-B3", "squad_v2", 0.01),
            SecondaryOutcome("y", "fam", "B6-vs-B3", "musique", 0.02),
        ]
        with pytest.raises(GatekeepingError, match="straddles"):
            evaluate_chain(primaries, bad)

    def test_duplicate_primary_fails_loud(self) -> None:
        dup = [
            PrimaryOutcome("B6-vs-B3", "squad_v2", 0.01),
            PrimaryOutcome("B6-vs-B3", "squad_v2", 0.02),
        ]
        with pytest.raises(GatekeepingError, match="duplicate"):
            evaluate_chain(dup, [])

    @staticmethod
    def _serial_chain(
        headline_musique_p: float,
    ) -> tuple[list[PrimaryOutcome], list[SecondaryOutcome]]:
        primaries = [
            PrimaryOutcome("headline", "squad_v2", 0.001),
            PrimaryOutcome("headline", "musique", headline_musique_p),
            PrimaryOutcome("truth_tax", "squad_v2", 0.0001),
            PrimaryOutcome("fingerprint", "squad_v2", 0.0001),
        ]
        secondaries = [
            SecondaryOutcome(
                "B17", "F2|truth_tax|squad_v2", "truth_tax", "squad_v2", 0.001
            ),
        ]
        return primaries, secondaries

    def test_serial_order_failed_gate_closes_downstream_primaries(self) -> None:
        # The 2026-08-02 regression: a failed upstream primary must make every
        # later primary (and its families) descriptive — no test at full α
        # behind a failed Dmitrienko serial gate.
        primaries, secondaries = self._serial_chain(headline_musique_p=0.40)
        trace = evaluate_chain(
            primaries,
            secondaries,
            primary_order=["headline", "truth_tax", "fingerprint"],
        )
        by_endpoint = {(p.endpoint, p.dataset): p for p in trace.primaries}
        assert by_endpoint[("headline", "squad_v2")].status == "confirmatory"
        # tiny p buys nothing behind the failed serial gate
        tax = by_endpoint[("truth_tax", "squad_v2")]
        assert tax.status == "descriptive" and tax.passed is False
        fingerprint = by_endpoint[("fingerprint", "squad_v2")]
        assert fingerprint.status == "descriptive" and fingerprint.passed is False
        # the downstream secondary family stays closed too — no alpha cascade
        assert all(s.status == "descriptive" for s in trace.secondaries)
        serial = {e.family_id: e for e in trace.events if e.family_id.startswith("primary-chain:")}
        assert serial["primary-chain:truth_tax"].opened is False
        assert trace.primary_order == ("headline", "truth_tax", "fingerprint")

    def test_serial_order_all_pass_opens_whole_chain(self) -> None:
        primaries, secondaries = self._serial_chain(headline_musique_p=0.01)
        trace = evaluate_chain(
            primaries,
            secondaries,
            primary_order=["headline", "truth_tax", "fingerprint"],
        )
        assert all(p.status == "confirmatory" for p in trace.primaries)
        assert all(s.status == "confirmatory" for s in trace.secondaries)

    def test_intra_set_rule_holm_any(self) -> None:
        # One surviving co-primary under Holm keeps the chain open under the
        # declared disjunctive rule; the conjunctive default closes it.
        primaries, secondaries = self._serial_chain(headline_musique_p=0.40)
        trace = evaluate_chain(
            primaries,
            secondaries,
            primary_order=["headline", "truth_tax", "fingerprint"],
            intra_set_rule="holm-any",
        )
        by_endpoint = {(p.endpoint, p.dataset): p for p in trace.primaries}
        assert by_endpoint[("truth_tax", "squad_v2")].status == "confirmatory"
        assert trace.intra_set_rule == "holm-any"

    def test_primary_order_validation(self) -> None:
        primaries, secondaries = self._serial_chain(headline_musique_p=0.01)
        with pytest.raises(GatekeepingError, match="exactly the supplied"):
            evaluate_chain(primaries, secondaries, primary_order=["headline"])
        with pytest.raises(GatekeepingError, match="exactly the supplied"):
            evaluate_chain(
                primaries,
                secondaries,
                primary_order=["headline", "truth_tax", "fingerprint", "ghost"],
            )
        with pytest.raises(GatekeepingError, match="duplicates"):
            evaluate_chain(
                primaries,
                secondaries,
                primary_order=["headline", "headline", "truth_tax", "fingerprint"],
            )
        with pytest.raises(GatekeepingError, match="intra_set_rule"):
            evaluate_chain(
                primaries,
                secondaries,
                primary_order=["headline", "truth_tax", "fingerprint"],
                intra_set_rule="majority",  # type: ignore[arg-type]
            )

    def test_no_order_keeps_flat_behavior(self) -> None:
        primaries, secondaries = self._chain()
        trace = evaluate_chain(primaries, secondaries)
        assert trace.primary_order is None
        assert all(p.status == "confirmatory" for p in trace.primaries)


# --------------------------------------------------------------------------- #
# wlt.py
# --------------------------------------------------------------------------- #
class TestWinLossTie:
    def test_counts(self) -> None:
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([0.0, 2.0, 4.0, 4.0, 4.0])
        res = win_loss_tie(a, b)
        assert (res.wins, res.losses, res.ties) == (2, 1, 2)
        assert res.n_pairs == 5

    def test_higher_is_better_flip(self) -> None:
        a = np.array([100.0, 200.0])  # e.g. TTFT ms: lower is better
        b = np.array([150.0, 150.0])
        res = win_loss_tie(a, b, higher_is_better=False)
        assert (res.wins, res.losses, res.ties) == (1, 1, 0)

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="differ in length"):
            win_loss_tie([1.0], [1.0, 2.0])
        with pytest.raises(ValueError, match="non-finite"):
            win_loss_tie([np.inf], [1.0])
        with pytest.raises(ValueError, match="empty"):
            win_loss_tie([], [])


# --------------------------------------------------------------------------- #
# calibration.py §9.7 gate-artifact serializer <-> analysis-driver loader
# (round trip: CalibrationReport.write == the schema load_calibration_report
# parses -- run_campaign_analysis.py's loader is the schema authority)
# --------------------------------------------------------------------------- #
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from src.analysis.stats.calibration import (  # noqa: E402
    AAResult,
    CalibrationReport,
    InjectionResult,
    build_report,
)

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "4_analysis"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import run_campaign_analysis as rca  # noqa: E402


def _mwu_p(a: np.ndarray, b: np.ndarray) -> float:
    return float(sps.mannwhitneyu(a, b, alternative="two-sided").pvalue)


def _sample_report() -> CalibrationReport:
    """Hand-built report exercising every serialized field, including a
    target_power=None injection and the 'flip' kind."""
    return CalibrationReport(
        seed=11,
        n_observations=120,
        aa=AAResult(
            n_splits=40, alpha=0.05, n_rejections=2,
            fp_rate=0.05, ci_low=0.0061, ci_high=0.1692,
        ),
        injections=(
            InjectionResult(
                effect_size=0.5, kind="shift", n_splits=40, alpha=0.05,
                n_rejections=30, power=0.75, ci_low=0.588, ci_high=0.873,
                target_power=0.7,
            ),
            InjectionResult(
                effect_size=0.25, kind="flip", n_splits=40, alpha=0.05,
                n_rejections=20, power=0.5, ci_low=0.338, ci_high=0.662,
                target_power=None,
            ),
        ),
    )


class TestCalibrationReportSerializer:
    def test_write_load_round_trip_identity(self, tmp_path: Path) -> None:
        report = _sample_report()
        out = report.write(tmp_path / "nested" / "calibration_report.json")
        assert out.is_file()  # write() creates parent dirs
        loaded = rca.load_calibration_report(out)
        # Frozen-dataclass equality covers every field, incl. the injections
        # tuple; json's repr-based float serialization round-trips exactly.
        assert loaded == report
        assert loaded.aa.approximates_nominal == report.aa.approximates_nominal

    def test_round_trip_through_the_real_machinery(self, tmp_path: Path) -> None:
        # End-to-end: build_report on the literal campaign test path, write,
        # load through the driver, and pass the confirmatory gate.
        rng = np.random.default_rng(7)
        data = rng.normal(0.0, 1.0, size=120)
        report = build_report(
            data, _mwu_p, n_splits=40, seed=11,
            effect_sizes=(1.5,), target_power=0.2,
        )
        out = report.write(tmp_path / "calibration_report.json")
        loaded = rca.load_calibration_report(out)
        assert loaded == report
        summary = rca.check_calibration(loaded, out)
        assert summary["verdict"] == "PASS"
        assert summary["seed"] == 11
        assert summary["n_observations"] == 120
        assert summary["n_injections"] == 1

    def test_artifact_shape_is_exactly_the_loader_schema(
        self, tmp_path: Path
    ) -> None:
        out = _sample_report().write(tmp_path / "calibration_report.json")
        text = out.read_text(encoding="utf-8")
        assert text.endswith("\n")
        raw = json.loads(text)
        assert set(raw) == {"seed", "n_observations", "aa", "injections"}
        assert set(raw["aa"]) == {
            "n_splits", "alpha", "n_rejections", "fp_rate", "ci_low", "ci_high",
        }
        for inj in raw["injections"]:
            assert set(inj) == {
                "effect_size", "kind", "n_splits", "alpha", "n_rejections",
                "power", "ci_low", "ci_high", "target_power",
            }
        # Derived verdicts are recomputed by the gate, never serialized (a
        # stale PASS in the artifact could contradict the code's criterion).
        assert "approximates_nominal" not in raw["aa"]
        assert all("meets_target" not in inj for inj in raw["injections"])
        assert raw["injections"][1]["target_power"] is None

    def test_numpy_scalars_are_coerced_to_native_json(self, tmp_path: Path) -> None:
        # A report hand-assembled from numpy scalars must still serialize
        # (json.dumps rejects np.int64/np.float64 without coercion).
        report = CalibrationReport(
            seed=np.int64(3),  # type: ignore[arg-type]
            n_observations=np.int64(50),  # type: ignore[arg-type]
            aa=AAResult(
                n_splits=np.int64(20), alpha=np.float64(0.05),  # type: ignore[arg-type]
                n_rejections=np.int64(1), fp_rate=np.float64(0.05),  # type: ignore[arg-type]
                ci_low=np.float64(0.0013), ci_high=np.float64(0.2487),  # type: ignore[arg-type]
            ),
        )
        out = report.write(tmp_path / "calibration_report.json")
        loaded = rca.load_calibration_report(out)
        assert loaded.n_observations == 50
        assert loaded.aa.n_splits == 20

    def test_failing_aa_report_is_written_but_gate_refuses(
        self, tmp_path: Path
    ) -> None:
        # Failure is documented (write always succeeds); refusal is the
        # consumer's job -- same doctrine as instrument_calibration.write_report.
        failing = CalibrationReport(
            seed=1,
            n_observations=80,
            aa=AAResult(
                n_splits=40, alpha=0.05, n_rejections=20,
                fp_rate=0.5, ci_low=0.338, ci_high=0.662,  # CI excludes alpha
            ),
        )
        out = failing.write(tmp_path / "calibration_report.json")
        loaded = rca.load_calibration_report(out)
        assert loaded == failing
        with pytest.raises(rca.CalibrationGateError, match="A/A FAILED"):
            rca.check_calibration(loaded, out)

    def test_missed_injection_target_round_trips_and_refuses(
        self, tmp_path: Path
    ) -> None:
        report = CalibrationReport(
            seed=1,
            n_observations=80,
            aa=AAResult(
                n_splits=40, alpha=0.05, n_rejections=2,
                fp_rate=0.05, ci_low=0.0061, ci_high=0.1692,
            ),
            injections=(
                InjectionResult(
                    effect_size=0.5, kind="shift", n_splits=40, alpha=0.05,
                    n_rejections=10, power=0.25, ci_low=0.127, ci_high=0.412,
                    target_power=0.8,  # point estimate misses -> FAIL
                ),
            ),
        )
        out = report.write(tmp_path / "calibration_report.json")
        loaded = rca.load_calibration_report(out)
        assert loaded == report
        assert loaded.injections[0].meets_target is False
        with pytest.raises(rca.CalibrationGateError, match="injection targets"):
            rca.check_calibration(loaded, out)
