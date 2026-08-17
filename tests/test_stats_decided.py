"""Tests for the 2026-08-16 owner-decided registered execution knobs
(Topic-7 fix batch; PUBLICATION.md §9.3-§9.5, audit findings G8/G18/G13):

- decision (a): per-row one-sided execution — the primitives pass
  ``alternative`` through to scipy's one-sided references and carry it on
  the result;
- decision (b): paired Wilcoxon ``zero_method='pratt'`` as the REGISTERED
  tie handling with a pinned approx+continuity-correction rule, plus
  effective-n (``n_nonzero``) reported beside the W/L/T triple;
- decision (c): §9.5 TOST/ROPE uncertainty by BLOCK bootstrap over
  batch-means windows (window = resampling unit) with a registered
  MIN_UNIQUE_WINDOWS floor, per-example estimand preserved.

The registered values are passed explicitly by the driver/map layer;
the primitive defaults stay backward-compatible (scipy-parity pins in
tests/test_stats_engine.py must keep passing unchanged).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sps

from src.analysis.stats.equivalence import (
    MIN_UNIQUE_WINDOWS,
    conditional_tost,
    rope_sensitivity,
)
from src.analysis.stats.tests_by_unit import (
    batch_means_contrast,
    mcnemar_binary,
    paired_wilcoxon,
)


# --------------------------------------------------------------------------- #
# decision (a) — directional execution vs scipy one-sided references
# --------------------------------------------------------------------------- #
class TestDirectionalWilcoxon:
    def test_greater_matches_scipy(self) -> None:
        rng = np.random.default_rng(8)
        a = rng.normal(0.5, 1.0, 40)
        b = rng.normal(0.0, 1.0, 40)
        res = paired_wilcoxon(a, b, alternative="greater")
        ref = sps.wilcoxon(a - b, zero_method="wilcox", alternative="greater")
        assert res.p_value == pytest.approx(float(ref.pvalue))
        assert res.alternative == "greater"

    def test_less_matches_scipy(self) -> None:
        rng = np.random.default_rng(8)
        a = rng.normal(0.5, 1.0, 40)
        b = rng.normal(0.0, 1.0, 40)
        res = paired_wilcoxon(a, b, alternative="less")
        ref = sps.wilcoxon(a - b, zero_method="wilcox", alternative="less")
        assert res.p_value == pytest.approx(float(ref.pvalue))
        assert res.alternative == "less"

    def test_registered_direction_finds_true_positive_shift(self) -> None:
        # a > b by construction: the registered one-sided arm must reject
        # while the opposite arm must not.
        rng = np.random.default_rng(21)
        b = rng.normal(0.0, 1.0, 60)
        a = b + 0.8
        assert paired_wilcoxon(a, b, alternative="greater").p_value < 0.01
        assert paired_wilcoxon(a, b, alternative="less").p_value > 0.99


class TestDirectionalMcNemar:
    # Fixed discordant table: n_10 = 8 (a succeeds where b fails),
    # n_01 = 2 (the reverse), so n_discordant = 10 and b := n_10 = 8.
    A = np.array([1] * 8 + [0] * 2 + [1] * 20 + [0] * 10, dtype=bool)
    B = np.array([0] * 8 + [1] * 2 + [1] * 20 + [0] * 10, dtype=bool)

    def test_greater_is_upper_tail_for_arm_a(self) -> None:
        # alternative="greater" ⇔ H1: arm a better ⇔ p = P[X ≥ 8] under
        # Binomial(10, ½) = (C(10,8)+C(10,9)+C(10,10))/2^10 = 56/1024.
        res = mcnemar_binary(self.A, self.B, alternative="greater")
        assert res.p_value == pytest.approx(56 / 1024)
        assert res.p_value == pytest.approx(float(sps.binom.sf(8 - 1, 10, 0.5)))
        assert res.p_value == pytest.approx(
            float(sps.binomtest(8, 10, 0.5, alternative="greater").pvalue)
        )
        assert res.alternative == "greater"

    def test_less_is_lower_tail(self) -> None:
        res = mcnemar_binary(self.A, self.B, alternative="less")
        assert res.p_value == pytest.approx(float(sps.binom.cdf(8, 10, 0.5)))
        assert res.p_value == pytest.approx(1013 / 1024)

    def test_swapping_arms_mirrors_the_tails(self) -> None:
        # mcnemar(b, a, "greater") tests H1: b better — its n_10 becomes 2;
        # P[X ≥ 2] = P[X ≤ 8] by Binomial(10, ½) symmetry.
        swapped = mcnemar_binary(self.B, self.A, alternative="greater")
        direct = mcnemar_binary(self.A, self.B, alternative="less")
        assert swapped.p_value == pytest.approx(direct.p_value)
        assert (swapped.n_10, swapped.n_01) == (2, 8)

    def test_zero_discordant_identity_under_every_alternative(self) -> None:
        a = np.array([True, False, True, True])
        for alt in ("two-sided", "greater", "less"):
            res = mcnemar_binary(a, a, alternative=alt)  # type: ignore[arg-type]
            assert res.p_value == 1.0
            assert res.alternative == alt


class TestDirectionalBatchMeans:
    A = np.array([10.0, 10.4, 9.6, 10.1])
    B = np.array([8.0, 8.2, 7.8])

    @pytest.mark.parametrize("alt", ["greater", "less"])
    def test_one_sided_matches_scipy_welch(self, alt: str) -> None:
        res = batch_means_contrast(self.A, self.B, alternative=alt)  # type: ignore[arg-type]
        ref = sps.ttest_ind(self.A, self.B, equal_var=False, alternative=alt)
        assert res.p_value == pytest.approx(float(ref.pvalue))
        assert res.alternative == alt

    def test_direction_semantics(self) -> None:
        # A's window means are higher: "greater" rejects, "less" cannot.
        assert batch_means_contrast(self.A, self.B, alternative="greater").p_value < 0.01
        assert batch_means_contrast(self.A, self.B, alternative="less").p_value > 0.99


# --------------------------------------------------------------------------- #
# decision (b) — pratt tie handling, pinned approx rule, effective n
# --------------------------------------------------------------------------- #
class TestPrattZeroMethod:
    # Pilot-shaped heavy-tie vector: 20 zero pairs, 6 wins (+1), 2 losses (−1)
    # — 28 pairs, only 8 of them informative.
    B_VEC = np.zeros(28)
    A_VEC = np.array([0.0] * 20 + [1.0] * 6 + [-1.0] * 2)

    def test_wilcox_branch_pinned(self) -> None:
        # Back-compat default: zeros discarded, scipy defaults (method="auto"
        # → asymptotic here since n_nonzero=8 has ties and the original
        # vector has zeros; correction=False).
        res = paired_wilcoxon(self.A_VEC, self.B_VEC, zero_method="wilcox")
        assert res.statistic == 9.0
        assert res.p_value == pytest.approx(0.15729920705028505, rel=1e-12)
        ref = sps.wilcoxon(self.A_VEC - self.B_VEC, zero_method="wilcox")
        assert res.p_value == pytest.approx(float(ref.pvalue))
        assert res.zero_method == "wilcox"

    def test_pratt_branch_pinned(self) -> None:
        # REGISTERED branch: zeros kept in the ranking; execution pinned to
        # the normal approximation WITH continuity correction.
        res = paired_wilcoxon(self.A_VEC, self.B_VEC, zero_method="pratt")
        assert res.statistic == 49.0
        assert res.p_value == pytest.approx(0.16157836645236978, rel=1e-12)
        ref = sps.wilcoxon(
            self.A_VEC - self.B_VEC, zero_method="pratt",
            method="approx", correction=True,
        )
        assert res.p_value == pytest.approx(float(ref.pvalue))
        assert res.zero_method == "pratt"

    def test_pratt_one_sided_pinned(self) -> None:
        res = paired_wilcoxon(
            self.A_VEC, self.B_VEC, zero_method="pratt", alternative="greater"
        )
        assert res.p_value == pytest.approx(0.08078918322618489, rel=1e-12)
        ref = sps.wilcoxon(
            self.A_VEC - self.B_VEC, zero_method="pratt",
            alternative="greater", method="approx", correction=True,
        )
        assert res.p_value == pytest.approx(float(ref.pvalue))

    def test_branches_differ_and_share_the_wlt_triple(self) -> None:
        rw = paired_wilcoxon(self.A_VEC, self.B_VEC, zero_method="wilcox")
        rp = paired_wilcoxon(self.A_VEC, self.B_VEC, zero_method="pratt")
        assert rw.p_value != rp.p_value
        assert rw.statistic != rp.statistic
        # §8.13 W/L/T triple and effective n are zero-method-invariant.
        for res in (rw, rp):
            assert (res.n_pos, res.n_neg, res.n_zero) == (6, 2, 20)
            assert res.n_pairs == 28
            assert res.n_nonzero == 8

    def test_n_nonzero_reported_alongside_n_pairs(self) -> None:
        rng = np.random.default_rng(3)
        a = rng.normal(0.0, 1.0, 30)
        res = paired_wilcoxon(a, a * 0.5)
        assert res.n_nonzero == res.n_pos + res.n_neg
        assert res.n_nonzero == res.n_pairs - res.n_zero

    def test_all_zero_diffs_identity_outcome_under_pratt(self) -> None:
        # scipy raises on an all-zero vector for either zero_method; the
        # T=0 identity outcome must stay a labeled result, not an error.
        a = np.ones(15)
        res = paired_wilcoxon(a, a, zero_method="pratt")
        assert res.p_value == 1.0
        assert res.n_nonzero == 0
        assert res.n_zero == 15
        assert res.zero_method == "pratt"

    def test_default_is_backcompat_wilcox(self) -> None:
        rng = np.random.default_rng(4)
        a = rng.normal(0.2, 1.0, 25)
        b = rng.normal(0.0, 1.0, 25)
        assert paired_wilcoxon(a, b) == paired_wilcoxon(a, b, zero_method="wilcox")

    def test_invalid_zero_method_rejected(self) -> None:
        with pytest.raises(ValueError, match="zero_method"):
            paired_wilcoxon([1.0, 2.0], [0.0, 1.0], zero_method="zsplit")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# decision (c) — block bootstrap over windows for §9.5 TOST/ROPE
# --------------------------------------------------------------------------- #
def _window_correlated_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """10 windows × 20 events with a strong window-level random effect
    (σ_between = 1.0) and tiny within-window noise (σ_within = 0.05):
    events inside a window are almost perfectly correlated, so treating
    them as 200 independent observations wildly overstates the evidence."""
    rng = np.random.default_rng(123)
    n_windows, per = 10, 20
    u = rng.normal(0.0, 1.0, n_windows)
    eps = rng.normal(0.0, 0.05, n_windows * per)
    d = np.repeat(u, per) + eps
    a = d
    b = np.zeros_like(d)
    window_ids = np.repeat(np.arange(n_windows), per)
    return a, b, np.ones(d.size, dtype=bool), window_ids


class TestBlockBootstrapTost:
    def test_registered_floor_is_five(self) -> None:
        # Pinned so the prereg text and the code cannot drift silently.
        assert MIN_UNIQUE_WINDOWS == 5

    def test_clustered_ci_wider_and_estimand_unchanged(self) -> None:
        a, b, mask, wid = _window_correlated_data()
        naive = conditional_tost(a, b, mask, margin=1.0, seed=42)
        clust = conditional_tost(a, b, mask, margin=1.0, seed=42, window_ids=wid)
        # (i) window-clustered uncertainty must be WIDER on within-window-
        # correlated data …
        assert (clust.dominance_ci_high - clust.dominance_ci_low) > (
            naive.dominance_ci_high - naive.dominance_ci_low
        )
        # … while the per-example point estimates are untouched.
        assert clust.dominance == naive.dominance
        assert clust.mean_diff == naive.mean_diff
        assert clust.n_events == naive.n_events == 200

    def test_clustering_blocks_the_artificially_easy_equivalence(self) -> None:
        # The exact failure the decision closes: per-example resampling on
        # 200 near-duplicated observations declares equivalence at a margin
        # the 10 truly-independent windows cannot support.
        a, b, mask, wid = _window_correlated_data()
        naive = conditional_tost(a, b, mask, margin=0.3, seed=42)
        clust = conditional_tost(a, b, mask, margin=0.3, seed=42, window_ids=wid)
        assert naive.domain_verdict == "equivalent"  # the false-easy claim
        assert clust.domain_verdict == "not-equivalent"
        assert clust.p_tost > naive.p_tost

    def test_result_metadata(self) -> None:
        a, b, mask, wid = _window_correlated_data()
        naive = conditional_tost(a, b, mask, margin=0.5, seed=42)
        clust = conditional_tost(a, b, mask, margin=0.5, seed=42, window_ids=wid)
        assert naive.resampling == "per-example"
        assert naive.n_windows is None
        assert clust.resampling == "window-block"
        assert clust.n_windows == 10

    def test_window_floor_refused(self) -> None:
        # (ii) fewer than MIN_UNIQUE_WINDOWS unique windows → refusal,
        # never a silent per-example fallback.
        a, b, mask, _ = _window_correlated_data()
        four = np.repeat(np.arange(4), 50)
        with pytest.raises(ValueError, match="unique windows"):
            conditional_tost(a, b, mask, margin=0.5, window_ids=four)

    def test_window_ids_length_mismatch_raises(self) -> None:
        a, b, mask, _ = _window_correlated_data()
        with pytest.raises(ValueError, match="length"):
            conditional_tost(a, b, mask, margin=0.5, window_ids=np.arange(7))

    def test_backcompat_no_window_ids_is_the_historical_algorithm(self) -> None:
        # (iii) without window_ids the seeded result must be bit-identical
        # to the pre-decision per-example algorithm, reproduced inline.
        a, b, mask, _ = _window_correlated_data()
        alpha, seed, margin = 0.05, 42, 0.3
        res = conditional_tost(a, b, mask, margin=margin, alpha=alpha, seed=seed)

        d = a - b
        n = d.size
        mean_d = float(d.mean())
        sd = float(d.std(ddof=1))
        se = sd / np.sqrt(n)
        p_lower = float(sps.t.sf((mean_d + margin) / se, n - 1))
        p_upper = float(sps.t.cdf((mean_d - margin) / se, n - 1))
        assert res.p_tost == pytest.approx(max(p_lower, p_upper))

        signs = np.sign(d)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(10_000, n))
        boot = signs[idx].mean(axis=1)
        assert res.dominance_ci_low == pytest.approx(float(np.percentile(boot, 5)))
        assert res.dominance_ci_high == pytest.approx(float(np.percentile(boot, 95)))
        assert res.resampling == "per-example"
        assert res.n_windows is None

    def test_clustered_deterministic_given_seed(self) -> None:
        a, b, mask, wid = _window_correlated_data()
        r1 = conditional_tost(a, b, mask, margin=0.3, seed=9, window_ids=wid)
        r2 = conditional_tost(a, b, mask, margin=0.3, seed=9, window_ids=wid)
        assert r1 == r2

    def test_insufficient_events_stays_labeled_before_the_floor(self) -> None:
        # A sparse event population is an expected data state: min_events
        # gates FIRST (labeled insufficient-n), the window floor never
        # upgrades it to an exception.
        a, b, _, wid = _window_correlated_data()
        sparse = np.zeros(a.size, dtype=bool)
        sparse[:5] = True  # 5 events, all inside window 0
        res = conditional_tost(
            a, b, sparse, margin=0.5, min_events=10, window_ids=wid
        )
        assert res.domain_verdict == "insufficient-n"
        assert res.dominance_verdict == "insufficient-n"
        assert res.resampling == "window-block"
        assert res.n_windows == 1


class TestBlockBootstrapRope:
    def test_clustered_posterior_widens_on_boundary_window_data(self) -> None:
        # 6 windows with window-constant diffs straddling the ROPE edge
        # (±rope = 0.5; values 0.45/0.55): per-example weighting is certain
        # the mass sits in the ROPE; window-clustered weighting is not.
        vals = np.array([0.45, 0.55, 0.45, 0.55, 0.45, 0.55])
        d = np.repeat(vals, 10)
        a, b = d, np.zeros_like(d)
        mask = np.ones(d.size, dtype=bool)
        wid = np.repeat(np.arange(6), 10)
        naive = rope_sensitivity(a, b, mask, rope=0.5, seed=7)
        clust = rope_sensitivity(a, b, mask, rope=0.5, seed=7, window_ids=wid)
        assert naive.verdict == "equivalent"
        assert naive.p_rope == pytest.approx(1.0)
        assert clust.p_rope < naive.p_rope
        assert clust.verdict == "not-equivalent"

    def test_estimand_preserved_on_iid_data(self) -> None:
        # No true window effect → the posterior MEAN region masses (the
        # estimand) must agree between the two weighting schemes; only the
        # posterior dispersion may differ.
        rng = np.random.default_rng(5)
        d = rng.normal(0.0, 0.3, 200)
        a, b = d, np.zeros_like(d)
        mask = np.ones(200, dtype=bool)
        wid = np.repeat(np.arange(10), 20)
        naive = rope_sensitivity(a, b, mask, rope=0.5, seed=7)
        clust = rope_sensitivity(a, b, mask, rope=0.5, seed=7, window_ids=wid)
        assert clust.mean_theta_rope == pytest.approx(naive.mean_theta_rope, abs=0.01)
        assert clust.mean_theta_left == pytest.approx(naive.mean_theta_left, abs=0.01)
        assert clust.mean_theta_right == pytest.approx(naive.mean_theta_right, abs=0.01)

    def test_result_metadata(self) -> None:
        rng = np.random.default_rng(5)
        d = rng.normal(0.0, 0.3, 60)
        a, b = d, np.zeros_like(d)
        mask = np.ones(60, dtype=bool)
        wid = np.repeat(np.arange(6), 10)
        naive = rope_sensitivity(a, b, mask, rope=0.5, seed=7)
        clust = rope_sensitivity(a, b, mask, rope=0.5, seed=7, window_ids=wid)
        assert naive.resampling == "per-example"
        assert naive.n_windows is None
        assert clust.resampling == "window-block"
        assert clust.n_windows == 6

    def test_window_floor_refused(self) -> None:
        a = np.ones(60)
        mask = np.ones(60, dtype=bool)
        with pytest.raises(ValueError, match="unique windows"):
            rope_sensitivity(
                a, np.zeros(60), mask, rope=0.5,
                window_ids=np.repeat(np.arange(3), 20),
            )

    def test_window_ids_length_mismatch_raises(self) -> None:
        a = np.ones(60)
        mask = np.ones(60, dtype=bool)
        with pytest.raises(ValueError, match="length"):
            rope_sensitivity(a, np.zeros(60), mask, rope=0.5, window_ids=np.arange(9))

    def test_clustered_deterministic_given_seed(self) -> None:
        rng = np.random.default_rng(3)
        a = rng.normal(0.0, 1.0, 50)
        b = rng.normal(0.0, 1.0, 50)
        mask = np.ones(50, dtype=bool)
        wid = np.repeat(np.arange(5), 10)
        r1 = rope_sensitivity(a, b, mask, rope=0.3, seed=11, window_ids=wid)
        r2 = rope_sensitivity(a, b, mask, rope=0.3, seed=11, window_ids=wid)
        assert r1 == r2

    def test_insufficient_events_stays_labeled_before_the_floor(self) -> None:
        a = np.zeros(40)
        mask = np.zeros(40, dtype=bool)
        mask[:4] = True  # 4 events in a single window
        res = rope_sensitivity(
            a, a, mask, rope=0.5, min_events=10,
            window_ids=np.repeat(np.arange(4), 10),
        )
        assert res.verdict == "insufficient-n"
        assert res.resampling == "window-block"
        assert res.n_windows == 1
