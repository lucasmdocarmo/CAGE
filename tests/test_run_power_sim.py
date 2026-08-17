"""Tests for the §9.6 power-simulation CLI (scripts/4_analysis/run_power_sim.py).

Everything runs on synthetic data — the pilot archive under results/ is never
touched (read-only doctrine). The registered-test-path adapters are proven
EQUAL to the §9.4 functions they wrap, not merely similar, and the 2026-08-07
verification fixes (real-pair null, tie-flip effect unit, compiled Holm-m,
window-guard fail-loud, TOST stage, dirty-tree provenance) are each pinned.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "4_analysis"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import run_power_sim as rps  # noqa: E402
from src.analysis.stats.families import KNOWN_DATASETS, compile_family_map  # noqa: E402
from src.analysis.stats.power_sim import (  # noqa: E402
    PowerSimError,
    simulate_campaign,
    tie_flip_injection,
    wilcoxon_signed_p,
)
from src.analysis.stats.tests_by_unit import (  # noqa: E402
    DEFAULT_MAX_WINDOWS,
    mcnemar_binary,
)

RNG = np.random.default_rng(20260807)


# --------------------------------------------------------------------------- #
# Synthetic paired mini-archive (two arms sharing example_ids)
# --------------------------------------------------------------------------- #

N_TRIALS = 3
EXAMPLES_PER_TRIAL = 30


def _write_pair_cells(run_root: Path, rng: np.random.Generator) -> None:
    """Two pilot-layout arms over the SAME example ids (joinable pairs)."""
    base: dict[str, dict[str, float]] = {}
    for trial in range(1, N_TRIALS + 1):
        for i in range(EXAMPLES_PER_TRIAL):
            ex = f"t{trial}_ex{i}"
            base[ex] = {
                "trial": trial,
                "ttft_ms": float(rng.normal(500.0, 100.0)),
                "faithfulness": float(rng.uniform(0.2, 0.9)),
                "exact_match": float(rng.integers(0, 2)),
            }
    for arm, ttft_shift, em_flip_p in (("rag", 40.0, 0.15), ("prefix_cache", 0.0, 0.0)):
        cell = run_root / "baselines" / arm
        for trial in range(1, N_TRIALS + 1):
            rows = []
            for ex, vals in base.items():
                if vals["trial"] != trial:
                    continue
                em = vals["exact_match"]
                if em_flip_p and rng.random() < em_flip_p:
                    em = 1.0 - em
                rows.append(
                    {
                        "example_id": ex,
                        "error": "",
                        "empty_generation": "",
                        "repeat_index": "0",
                        "ttft_ms": vals["ttft_ms"] + ttft_shift + float(rng.normal(0, 20)),
                        "faithfulness": min(1.0, vals["faithfulness"] + float(rng.normal(0, 0.05))),
                        "exact_match": em,
                        "f1_score": float(rng.uniform(0.0, 1.0)),
                    }
                )
            trial_dir = cell / f"trial_{trial}"
            trial_dir.mkdir(parents=True)
            pd.DataFrame(rows).to_csv(trial_dir / "results.csv", index=False)


@pytest.fixture(scope="module")
def pair_archive(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("mini_pair_pilot") / "fake_run"
    _write_pair_cells(root, np.random.default_rng(11))
    return root


# --------------------------------------------------------------------------- #
# Registered-path adapters
# --------------------------------------------------------------------------- #


class TestAdapters:
    def test_wilcoxon_diff_p_equals_skeleton_default(self) -> None:
        diffs = RNG.normal(0.3, 1.0, size=80)
        assert rps.wilcoxon_diff_p(diffs) == pytest.approx(
            wilcoxon_signed_p(diffs), abs=1e-12
        )

    def test_wilcoxon_diff_p_all_zero_is_one(self) -> None:
        assert rps.wilcoxon_diff_p(np.zeros(50)) == 1.0

    def test_mcnemar_diff_p_equals_registered_mcnemar(self) -> None:
        # 12 wins (+1), 5 losses (-1), 83 ties (0)
        diffs = np.concatenate([np.ones(12), -np.ones(5), np.zeros(83)])
        RNG.shuffle(diffs)
        a = (diffs > 0).astype(float)
        b = (diffs < 0).astype(float)
        expected = mcnemar_binary(a, b, alternative="two-sided").p_value
        assert rps.mcnemar_diff_p(diffs) == pytest.approx(expected, abs=1e-12)

    def test_mcnemar_diff_p_all_ties_is_one(self) -> None:
        assert rps.mcnemar_diff_p(np.zeros(40)) == 1.0

    def test_mcnemar_diff_p_refuses_non_binary_diffs(self) -> None:
        with pytest.raises(PowerSimError, match="-1, 0, 1"):
            rps.mcnemar_diff_p(np.array([0.5, 0.0, -1.0]))


# --------------------------------------------------------------------------- #
# Real-pair null models (the 2026-08-07 critical fix)
# --------------------------------------------------------------------------- #


class TestPairedNullModel:
    def test_binary_pool_preserves_real_tie_mass(self) -> None:
        # 90% ties, 6% wins, 4% losses — a REAL paired shape, unlike the
        # ~50%-discordant cross-pair fiction the verification killed.
        diffs = np.concatenate([np.zeros(90), np.ones(6), -np.ones(4)])
        model = rps.make_paired_null_model(diffs, "flip")
        draw = model(np.random.default_rng(3), 200_000)
        assert np.isin(draw, (-1.0, 0.0, 1.0)).all()
        assert float(np.mean(draw == 0.0)) == pytest.approx(0.90, abs=0.01)
        # symmetrized: P(+1) == P(-1) == observed discordant / 2
        assert float(np.mean(draw == 1.0)) == pytest.approx(0.05, abs=0.01)
        assert float(np.mean(draw == -1.0)) == pytest.approx(0.05, abs=0.01)

    def test_continuous_pool_is_centered_and_symmetric(self) -> None:
        diffs = RNG.normal(40.0, 15.0, size=200)  # true location effect 40
        model = rps.make_paired_null_model(diffs, "shift")
        draw = model(np.random.default_rng(5), 100_000)
        assert abs(float(draw.mean())) < 1.0  # location removed (H0)
        assert float(draw.std()) == pytest.approx(15.0, rel=0.15)

    def test_deterministic_under_seed(self) -> None:
        diffs = RNG.normal(0.0, 1.0, size=60)
        model = rps.make_paired_null_model(diffs, "shift")
        d1 = model(np.random.default_rng(7), 100)
        d2 = model(np.random.default_rng(7), 100)
        assert np.array_equal(d1, d2)

    def test_refuses_tiny_pool(self) -> None:
        with pytest.raises(rps.PowerSimCLIError, match="pairs"):
            rps.make_paired_null_model(np.zeros(5), "flip")

    def test_composes_with_simulate_campaign_and_tie_flip(self) -> None:
        # Tie-heavy REAL paired shape: flip signal must dominate.
        diffs = np.concatenate([np.zeros(90), np.ones(6), -np.ones(4)])
        model = rps.make_paired_null_model(diffs, "flip")
        table = simulate_campaign(
            [0.0, 0.5],
            [100],
            model,
            50,
            seed=11,
            test_fn=rps.mcnemar_diff_p,
            injection=tie_flip_injection,
        )
        null_power = float(table[table["effect"] == 0.0]["power"].iloc[0])
        big_power = float(table[table["effect"] == 0.5]["power"].iloc[0])
        assert null_power <= 0.2
        assert big_power >= 0.9


class TestPairGuard:
    def test_shift_refused_on_tie_heavy_diffs(self) -> None:
        diffs = np.concatenate([np.zeros(95), RNG.normal(0, 1, 5)])
        with pytest.raises(rps.PowerSimCLIError, match="P0 2026-08-02"):
            rps.guard_pair_kind(
                "shift", diffs, metric="faithfulness", pair=("a", "b")
            )

    def test_flip_refused_on_non_binary_diffs(self) -> None:
        diffs = RNG.normal(0, 1, size=100)
        with pytest.raises(rps.PowerSimCLIError, match="strictly binary"):
            rps.guard_pair_kind(
                "flip", diffs, metric="exact_match", pair=("a", "b")
            )

    def test_diagnostics_report_real_structure(self) -> None:
        diffs = np.concatenate([np.zeros(80), np.ones(12), -np.ones(8)])
        diag = rps.guard_pair_kind(
            "flip", diffs, metric="exact_match", pair=("a", "b")
        )
        assert diag["tie_mass"] == pytest.approx(0.80)
        assert diag["discordant_rate"] == pytest.approx(0.20)
        assert diag["n_pairs"] == 100


class TestLoadPairDiffs:
    def test_joins_on_example_id(self, pair_archive: Path) -> None:
        diffs = rps.load_pair_diffs(pair_archive, "rag", "prefix_cache", "ttft_ms")
        assert diffs.size == N_TRIALS * EXAMPLES_PER_TRIAL
        # rag carries a +40ms true shift in the fixture
        assert 20.0 < float(diffs.mean()) < 60.0

    def test_binary_pair_diffs_are_ternary(self, pair_archive: Path) -> None:
        diffs = rps.load_pair_diffs(
            pair_archive, "rag", "prefix_cache", "exact_match"
        )
        assert np.isin(diffs, (-1.0, 0.0, 1.0)).all()
        assert float(np.mean(diffs == 0.0)) > 0.5  # mostly ties by design

    def test_missing_arm_fails_closed(self, pair_archive: Path) -> None:
        with pytest.raises(rps.PowerSimCLIError, match="not found"):
            rps.load_pair_diffs(pair_archive, "rag", "no_such_arm", "ttft_ms")


# --------------------------------------------------------------------------- #
# Window stage
# --------------------------------------------------------------------------- #


class TestWindowStage:
    def test_deterministic_and_schema(self) -> None:
        pool = RNG.normal(400.0, 80.0, size=90)
        t1 = rps.simulate_window_power(
            pool, [0.0], [25], [3, 5], 30, seed=5, alpha=0.05
        )
        t2 = rps.simulate_window_power(
            pool, [0.0], [25], [3, 5], 30, seed=5, alpha=0.05
        )
        pd.testing.assert_frame_equal(t1, t2)
        assert set(t1.columns) >= {
            "effect", "queries_per_window", "n_windows", "power",
            "ci_low", "ci_high", "n_degenerate",
        }

    def test_null_power_near_alpha_and_huge_effect_near_one(self) -> None:
        pool = RNG.normal(400.0, 80.0, size=90)
        table = rps.simulate_window_power(
            pool, [0.0, 500.0], [50], [8], 100, seed=9, alpha=0.05
        )
        null_power = float(table[table["effect"] == 0.0]["power"].iloc[0])
        big_power = float(table[table["effect"] == 500.0]["power"].iloc[0])
        assert null_power <= 0.15
        assert big_power >= 0.95

    def test_constant_pool_is_degenerate_not_a_crash(self) -> None:
        pool = np.full(60, 250.0)
        table = rps.simulate_window_power(
            pool, [0.0], [25], [3], 20, seed=2, alpha=0.05
        )
        row = table.iloc[0]
        assert row["n_degenerate"] == 20
        assert row["power"] == 0.0

    def test_nw_grid_beyond_max_windows_fails_loud(self) -> None:
        # 2026-08-07 fix: the §9.4 guard refusal must NOT masquerade as
        # 'degenerate' — a misconfigured grid fails upfront.
        pool = RNG.normal(0, 1, size=60)
        with pytest.raises(rps.PowerSimCLIError, match="max_windows"):
            rps.simulate_window_power(
                pool, [0.0], [25], [DEFAULT_MAX_WINDOWS + 10], 10,
                seed=1, alpha=0.05,
            )

    def test_grid_validation(self) -> None:
        pool = RNG.normal(0, 1, size=60)
        with pytest.raises(rps.PowerSimCLIError, match=">= 2"):
            rps.simulate_window_power(pool, [0.0], [25], [1], 10, seed=1, alpha=0.05)
        with pytest.raises(rps.PowerSimCLIError, match="non-empty"):
            rps.simulate_window_power(pool, [], [25], [3], 10, seed=1, alpha=0.05)


# --------------------------------------------------------------------------- #
# Conditional-TOST stage (#13 NONE legs)
# --------------------------------------------------------------------------- #


class TestTostStage:
    def test_equivalence_power_driven_by_dominance_layer(self) -> None:
        # Tie-free continuous signs make the Cliff's-delta dominance layer the
        # binding constraint: its (1-2a) bootstrap CI half-width ~1.645/sqrt(n)
        # cannot fit inside +/-0.147 at n=100 (prob ~0) and comfortably fits
        # at n=1200 (prob ~1). The domain t-TOST passes trivially throughout.
        diffs = RNG.normal(0.0, 0.01, size=100)  # tight around 0 vs margin 0.05
        table = rps.simulate_tost_power(
            diffs, [100, 1200], [1.0], 30, seed=4, margin=0.05
        )
        p100 = float(table[table["n"] == 100]["prob_equivalent"].iloc[0])
        p1200 = float(table[table["n"] == 1200]["prob_equivalent"].iloc[0])
        assert p100 <= 0.2
        assert p1200 >= 0.9

    def test_insufficient_events_counted_not_hidden(self) -> None:
        diffs = RNG.normal(0.0, 0.01, size=100)
        # 0.05 x 100 = 5 events < DEFAULT_MIN_EVENTS=10 -> all insufficient
        table = rps.simulate_tost_power(
            diffs, [100], [0.05], 20, seed=6, margin=0.05
        )
        row = table.iloc[0]
        assert row["n_insufficient"] == 20
        assert row["prob_equivalent"] == 0.0

    def test_deterministic_under_seed(self) -> None:
        diffs = RNG.normal(0.0, 0.02, size=80)
        t1 = rps.simulate_tost_power(diffs, [100], [0.25], 10, seed=9)
        t2 = rps.simulate_tost_power(diffs, [100], [0.25], 10, seed=9)
        pd.testing.assert_frame_equal(t1, t2)

    def test_tost_required_extraction(self) -> None:
        table = pd.DataFrame(
            [
                {"family": "fingerprint_none_legs", "metric": "faithfulness",
                 "dataset": "d", "pair_regime": "cross_mechanism",
                 "alpha_role": "primary_full_alpha", "event_fraction": 0.25,
                 "n": 100, "prob_equivalent": 0.4, "margin": 0.05},
                {"family": "fingerprint_none_legs", "metric": "faithfulness",
                 "dataset": "d", "pair_regime": "cross_mechanism",
                 "alpha_role": "primary_full_alpha", "event_fraction": 0.25,
                 "n": 500, "prob_equivalent": 0.9, "margin": 0.05},
            ]
        )
        req = rps.tost_required(table)
        assert req[0]["required_n"] == 500
        assert req[0]["family"] == "fingerprint_none_legs"

    def test_binary_pool_preserves_ternary_support(self) -> None:
        diffs = np.concatenate([np.zeros(80), np.ones(12), -np.ones(8)])
        table = rps.simulate_tost_power(
            diffs, [200], [1.0], 10, seed=3, margin=0.05, binary=True
        )
        assert len(table) == 1  # runs clean on a ternary pool
        with pytest.raises(rps.PowerSimCLIError, match="-1, 0, 1"):
            rps.simulate_tost_power(
                RNG.normal(0, 1, 50), [100], [1.0], 5, seed=1, binary=True
            )

    def test_specs_cover_headline_equivalence(self) -> None:
        # Amendment A2: the headline #4 machinery is simulated on its own
        # co-primary metrics at margins = the family MDEs.
        by_family = {s[0]: s for s in rps.TOST_SPECS}
        pred = by_family["headline_predicate_equiv"]
        assert pred[1] == "exact_match" and pred[2] == 0.05 and pred[4] is True
        ttft = by_family["headline_ttft_equiv"]
        assert ttft[1] == "ttft_ms" and ttft[2] == 25.0
        assert by_family["fingerprint_none_legs"][3] == (0.05, 0.25, 1.0)


# --------------------------------------------------------------------------- #
# Required-N extraction + recommendation
# --------------------------------------------------------------------------- #


def _pq_table(rows: list[tuple[str, str, float, int, float]]) -> pd.DataFrame:
    """(dataset, regime, effect, n, power) -> minimal per-query power table."""
    return pd.DataFrame(
        [
            {
                "stage": "per_query",
                "dataset": ds,
                "pair_regime": regime,
                "family": "binary_predicate",
                "metric": "exact_match",
                "alpha_role": "primary_full_alpha",
                "effect": e,
                "n": n,
                "power": p,
            }
            for ds, regime, e, n, p in rows
        ]
    )


class TestRequiredAndRecommendation:
    def test_per_query_required_reports_not_reached(self) -> None:
        table = _pq_table(
            [
                ("squad_v2", "cross_mechanism", 0.05, 100, 0.4),
                ("squad_v2", "cross_mechanism", 0.05, 500, 0.85),
                ("squad_v2", "cross_mechanism", 0.02, 100, 0.1),
                ("squad_v2", "cross_mechanism", 0.02, 500, 0.3),
            ]
        )
        req = rps.per_query_required(table)
        by_effect = {r["effect"]: r["required_n"] for r in req}
        assert by_effect[0.05] == 500
        assert by_effect[0.02] == rps.NOT_REACHED

    def test_required_windows_labels_and_offgrid(self) -> None:
        table = pd.DataFrame(
            [
                {"effect": 25.0, "queries_per_window": 50, "n_windows": 3, "power": 0.5},
                {"effect": 25.0, "queries_per_window": 50, "n_windows": 8, "power": 0.9},
            ]
        )
        assert rps.required_windows(table, 25.0, 50) == 8
        table.loc[1, "power"] = 0.6
        assert rps.required_windows(table, 25.0, 50) == rps.NOT_REACHED
        with pytest.raises(rps.PowerSimCLIError, match="not on the simulated grid"):
            rps.required_windows(table, 99.0, 50)

    def test_binding_is_max_and_not_reached_dominates(self) -> None:
        rows = [
            {"dataset": "a", "required_n": 300},
            {"dataset": "b", "required_n": 800},
        ]
        assert rps._binding(rows, "required_n")["binding"] == 800
        rows.append({"dataset": "c", "required_n": rps.NOT_REACHED})
        assert rps._binding(rows, "required_n")["binding"] == rps.NOT_REACHED

    def test_recommendation_binds_cross_mechanism_only(self) -> None:
        table = _pq_table(
            [
                ("squad_v2", "cross_mechanism", 0.05, 300, 0.85),
                ("hotpotqa", "cross_mechanism", 0.05, 300, 0.6),
                ("hotpotqa", "cross_mechanism", 0.05, 800, 0.9),
                # same_family reaches earlier — must NOT drive the binding N
                ("squad_v2", "same_family", 0.05, 100, 0.9),
                ("hotpotqa", "same_family", 0.05, 100, 0.9),
            ]
        )
        req = rps.per_query_required(table)
        rec = rps.build_recommendation(req, [], [])
        block = rec["per_query"]["binary_predicate"]["primary_full_alpha"]
        assert block["binding"] == 800  # max over datasets, binding regime only
        sens = rec["per_query_sensitivity_same_family"]["binary_predicate"]
        assert sens["primary_full_alpha"]["binding"] == 100
        assert rec["binding_pair_regime"] == "cross_mechanism"
        assert rec["qasper_policy"].startswith("zero pilot data")

    def test_guard_refusals_surface_explicitly_in_recommendation(self) -> None:
        # 2026-08-07 residual finding: a family refused everywhere must not
        # silently vanish while declared_candidate_mdes still advertises it.
        refusals = [
            {
                "dataset": ds,
                "pair_regime": regime,
                "family": "quality_continuous",
                "metric": "faithfulness",
                "reason": "REFUSED (P0 2026-08-02 doctrine): tie-heavy diffs",
            }
            for ds in ("squad_v2", "hotpotqa")
            for regime in ("cross_mechanism", "same_family")
        ]
        rec = rps.build_recommendation([], [], [], refusals)
        block = rec["per_query"]["quality_continuous"]
        assert block["status"] == "REFUSED_BY_INJECTION_GUARD"
        assert block["mde"] == rps.DECLARED_MDE["quality_continuous"]
        assert len(block["guard_refusals"]) == 2  # binding regime datasets
        assert "UNDELIVERED" in block["resolution_pending"]
        sens = rec["per_query_sensitivity_same_family"]["quality_continuous"]
        assert sens["status"] == "REFUSED_BY_INJECTION_GUARD"


# --------------------------------------------------------------------------- #
# Report assembly (G15 fixes, 2026-08-16)
# --------------------------------------------------------------------------- #


def _minimal_config() -> dict:
    """Just the keys _markdown_report reads — no simulation required."""
    return {
        "repo_state": {"dirty": False, "head": "abc1234"},
        "generated_utc": "2026-08-16T00:00:00+00:00",
        "seed": 1,
        "n_sims": 2,
        "secondary_holm_m": rps.SECONDARY_HOLM_M,
        "secondary_holm_worst_family": "A|x|y|F1|per_query",
        "source_runs": {"squad_v2": "/tmp/run"},
        "pair_regimes": {"cross_mechanism": ["rag", "prefix_cache"]},
        "null_model": "synthetic (test)",
        "effect_unit_reconciliation": "n/a (test)",
        "loader_validity_rule": "n/a (test)",
        "alpha_roles": dict(rps.ALPHA_ROLES),
    }


class TestMarkdownReport:
    def test_translation_table_derives_effect_keys_from_grid(self) -> None:
        # G15: the old table hard-coded the 0.02/0.05/0.1 keys and raised
        # KeyError AFTER a long sim whenever the binary grid changed. Keys
        # must now come from the simulated grid itself.
        translation = [
            {
                "dataset": "squad_v2",
                "pair_regime": "cross_mechanism",
                "tie_mass": 0.9,
                "implied_pp": {"0.03": 0.027, "0.07": 0.063},
            },
        ]
        md = rps._markdown_report(
            _minimal_config(), [], [], [],
            {"qasper_policy": "zero pilot data (test)"}, [], translation,
        )
        assert "effect 0.03 -> pp" in md
        assert "effect 0.07 -> pp" in md
        assert "0.0270" in md and "0.0630" in md

    def test_translation_table_ragged_grids_render_na(self) -> None:
        translation = [
            {"dataset": "a", "pair_regime": "r", "tie_mass": 0.5,
             "implied_pp": {"0.02": 0.01}},
            {"dataset": "b", "pair_regime": "r", "tie_mass": 0.5,
             "implied_pp": {"0.05": 0.025}},
        ]
        md = rps._markdown_report(
            _minimal_config(), [], [], [],
            {"qasper_policy": "zero pilot data (test)"}, [], translation,
        )
        assert "n/a" in md  # a missing key renders, never KeyErrors

    def test_translation_table_empty_is_labeled(self) -> None:
        md = rps._markdown_report(
            _minimal_config(), [], [], [],
            {"qasper_policy": "zero pilot data (test)"}, [], [],
        )
        assert "nothing to translate" in md

    def test_per_query_stage_labels_exact_match_as_proxy(
        self, pair_archive: Path
    ) -> None:
        # G15: the pilot archive predates the §8.5 predicate column, so the
        # binary family runs on exact_match — the artifact rows themselves
        # must carry the PROXY label, not just prose.
        table, refusals = rps.run_per_query_stage(
            {"squad_v2": pair_archive},
            seed=3,
            n_grid=[24],
            n_sims=3,
            alpha_roles={"primary_full_alpha": 0.05},
            pair_regimes={"cross_mechanism": ("rag", "prefix_cache")},
        )
        binary = table[table["metric"] == "exact_match"]
        assert not binary.empty
        assert binary["metric_note"].str.contains("PROXY").all()
        assert binary["metric_note"].str.contains("predicate").all()
        others = table[table["metric"] != "exact_match"]
        assert (others["metric_note"] == "").all()
        # the provenance string also names the proxy status
        assert "PROXY" in rps.EXACT_MATCH_PROXY_NOTE


# --------------------------------------------------------------------------- #
# Config sanity (charter + 2026-08-07 verification pins)
# --------------------------------------------------------------------------- #


class TestConfig:
    def test_declared_mdes_are_on_the_simulated_grids(self) -> None:
        fam_effects = {f.name: f.effect_sizes for f in rps._per_query_families()}
        for fam, mde in rps.DECLARED_MDE.items():
            if fam in fam_effects:
                assert mde in fam_effects[fam]
        window_effects = {name: eff for name, _, eff, _ in rps.WINDOW_METRICS}
        for name, effects in window_effects.items():
            assert rps.DECLARED_MDE[name] in effects

    def test_target_power_is_charter_value(self) -> None:
        assert rps.TARGET_POWER == 0.8  # §9.6

    def test_boundary_grid_densified(self) -> None:
        # Amendment A7: 1600/2400 remove the 1200->2000 grid-quantization gap.
        assert 1600 in rps.DEFAULT_N_GRID and 2400 in rps.DEFAULT_N_GRID

    def test_truth_tax_pool_and_window_metric(self, pair_archive: Path) -> None:
        # Amendment A4: #14's registered variable simulated directly.
        # The fixture has no no_cache arm -> the loader must fail closed;
        # the WINDOW_METRICS wiring is pinned structurally.
        names = [m[0] for m in rps.WINDOW_METRICS]
        assert "window_truth_tax" in names and "window_faithfulness_mean" in names
        with pytest.raises(rps.PowerSimCLIError, match="not found"):
            rps.load_truth_tax_pool(pair_archive)

    def test_secondary_holm_m_matches_compiled_family_map(self) -> None:
        fam_map = compile_family_map(sorted(KNOWN_DATASETS))
        holm = fam_map[fam_map["correction"] == "holm"]
        m = int(holm.groupby("family_id").size().max())
        assert rps.SECONDARY_HOLM_M == m
        # Pinned at the 2026-08-16 unit-split value: the pre-split map pooled
        # 9 per-query + 3 window rows into one m=12 family (the G19 unit
        # mixing); the family_id UNIT axis splits them, so the largest
        # compiled Holm family is now the 9-member per-query group-A family.
        # (The old ">= 10 regression floor" asserted exactly that pooled
        # family and is superseded by this exact pin.)
        assert m == 9
        assert rps.ALPHA_ROLES["secondary_holm_worst"] == pytest.approx(0.05 / m)

    def test_fingerprint_alpha_derived_from_registered_holm_legs(self) -> None:
        # G15 (2026-08-16): no literal /3 — the divisor is the count of Holm
        # superiority legs in the registered #13 decomposition.
        from src.analysis.stats.families import FINGERPRINT_SUB_HYPOTHESES

        holm_legs = [
            policy
            for policy, correction, _sided, _pred in FINGERPRINT_SUB_HYPOTHESES
            if correction == "holm"
        ]
        assert rps.FINGERPRINT_HOLM_LEGS == len(holm_legs) == 3
        assert holm_legs == ["evict", "compress", "truncate"]
        assert rps.ALPHA_ROLES["fingerprint_holm3_worst"] == pytest.approx(
            0.05 / rps.FINGERPRINT_HOLM_LEGS
        )

    def test_window_stage_has_all_three_alpha_roles(self) -> None:
        # 2026-08-07 fix: window secondaries (#15/#17/#18/#20) are Holm-corrected.
        assert set(rps.WINDOW_ALPHA_ROLES) == set(rps.ALPHA_ROLES)

    def test_binary_effect_unit_is_tie_flip_semantics(self) -> None:
        fams = {f.name: f for f in rps._per_query_families()}
        unit = fams["binary_predicate"].effect_unit
        assert "tie" in unit and "NOT" in unit  # explicitly disambiguated
        assert "0-outcome" not in unit.split("NOT")[0]  # §9.7 unit not claimed

    def test_git_state_reports_dirty_flag(self) -> None:
        state = rps._git_state()
        assert set(state) >= {"head", "dirty", "changed_or_untracked", "provisional"}
        assert state["provisional"] == state["dirty"]

    def test_out_dir_refusal_inside_results(self, tmp_path: Path) -> None:
        bad = tmp_path / "results" / "power"
        with pytest.raises(Exception, match="read-only"):
            rps.cal._check_out_dir(bad)
