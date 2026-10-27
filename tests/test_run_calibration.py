"""Tests for the §9.7 calibration CLI (scripts/4_analysis/run_calibration.py).

Everything runs on synthetic mini-archives in tmp_path — the pilot archive under
results/ is never touched (read-only doctrine), and the campaign driver is only
ever imported at function level (no CLI invocation, no confirmatory mode).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "4_analysis"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import run_calibration as rc  # noqa: E402
from src.analysis.stats.calibration import AAResult  # noqa: E402

# --------------------------------------------------------------------------- #
# Synthetic mini-archive
# --------------------------------------------------------------------------- #

N_TRIALS = 3
EXAMPLES_PER_TRIAL = 30  # disjoint per trial, like the pilot -> 90 per cell


def _write_no_cache_cell(run_root: Path, rng: np.random.Generator) -> None:
    """Pilot-layout cell: <run>/baselines/no_cache/trial_N/results.csv."""
    cell = run_root / "baselines" / "no_cache"
    for trial in range(1, N_TRIALS + 1):
        rows = []
        for i in range(EXAMPLES_PER_TRIAL):
            example = f"t{trial}_ex{i}"
            rows.append(
                {
                    "example_id": example,
                    "error": "",
                    "empty_generation": "",
                    "repeat_index": "0",
                    "ttft_ms": float(rng.normal(500.0, 100.0)),
                    "faithfulness": float(rng.uniform(0.0, 1.0)),
                    "exact_match": float(rng.integers(0, 2)),
                    "f1_score": float(rng.uniform(0.0, 1.0)),
                }
            )
        trial_dir = cell / f"trial_{trial}"
        trial_dir.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(trial_dir / "results.csv", index=False)


@pytest.fixture(scope="module")
def mini_archive(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("mini_pilot")
    rng = np.random.default_rng(7)
    runs: dict[str, Path] = {}
    for ds in ("squad_v2", "hotpotqa"):
        run_root = root / f"fake_run_{ds}"
        _write_no_cache_cell(run_root, rng)
        runs[ds] = run_root
    return runs


# --------------------------------------------------------------------------- #
# Guard: the P0 2026-08-02 honest-injection doctrine as code
# --------------------------------------------------------------------------- #


class TestInjectionGuard:
    def test_collision_probability_continuous_is_tiny(self) -> None:
        values = np.random.default_rng(0).normal(size=300)
        assert rc.collision_probability(values) == pytest.approx(1 / 300)

    def test_collision_probability_constant_is_one(self) -> None:
        assert rc.collision_probability(np.ones(50)) == pytest.approx(1.0)

    def test_collision_probability_empty_fails_closed(self) -> None:
        with pytest.raises(rc.CalibrationCLIError, match="empty"):
            rc.collision_probability(np.array([]))

    def test_shift_refused_on_tie_heavy_metric(self) -> None:
        # 90% of mass at 1.0 -- the f1/grounding shape the P0 bug lived on.
        values = np.concatenate([np.ones(90), np.linspace(0, 0.9, 10)])
        with pytest.raises(rc.CalibrationCLIError, match="P0 2026-08-02"):
            rc.guard_injection_kind("shift", values, metric="f1_score")

    def test_shift_allowed_on_continuous_metric(self) -> None:
        values = np.random.default_rng(1).normal(500, 100, size=300)
        diag = rc.guard_injection_kind("shift", values, metric="ttft_ms")
        assert diag["collision_probability"] < rc.MAX_SHIFT_COLLISION
        assert diag["binary"] is False

    def test_flip_refused_on_non_binary_metric(self) -> None:
        values = np.random.default_rng(2).uniform(size=100)
        with pytest.raises(rc.CalibrationCLIError, match="strictly binary"):
            rc.guard_injection_kind("flip", values, metric="faithfulness")

    def test_flip_allowed_on_binary_metric(self) -> None:
        values = np.array([0.0, 1.0] * 50)
        diag = rc.guard_injection_kind("flip", values, metric="exact_match")
        assert diag["binary"] is True

    def test_unknown_kind_refused(self) -> None:
        with pytest.raises(rc.CalibrationCLIError, match="unknown injection kind"):
            rc.guard_injection_kind("scale", np.ones(10), metric="x")

    def test_registered_families_shape(self) -> None:
        # The three registered families: serving/quality continuous + binary
        # predicate; ONLY the binary family may use the flip model, and the
        # continuous families use shift (guard-verified at run time).
        kinds = {f.name: f.kind for f in rc.FAMILIES}
        assert kinds == {
            "serving_continuous": "shift",
            "quality_continuous": "shift",
            "binary_predicate": "flip",
        }
        assert all(len(f.effect_sizes) == 3 for f in rc.FAMILIES)


# --------------------------------------------------------------------------- #
# Loader adapter: canonical validity rule
# --------------------------------------------------------------------------- #


class TestLoadArmMetric:
    def test_validity_rule_excludes_error_empty_and_repeats(
        self, tmp_path: Path
    ) -> None:
        cell = tmp_path / "run" / "baselines" / "no_cache" / "trial_1"
        cell.mkdir(parents=True)
        rows = [
            # 30 valid rows
            *[
                {
                    "example_id": f"ex{i}",
                    "error": "",
                    "empty_generation": "",
                    "repeat_index": "0",
                    "ttft_ms": 100.0 + i,
                }
                for i in range(30)
            ],
            # excluded: real error, empty generation, repeat>0
            {"example_id": "bad1", "error": "boom", "empty_generation": "",
             "repeat_index": "0", "ttft_ms": 1.0},
            {"example_id": "bad2", "error": "", "empty_generation": "True",
             "repeat_index": "0", "ttft_ms": 2.0},
            {"example_id": "rep", "error": "", "empty_generation": "",
             "repeat_index": "1", "ttft_ms": 3.0},
        ]
        pd.DataFrame(rows).to_csv(cell / "results.csv", index=False)
        values = rc.load_arm_metric(tmp_path / "run", "ttft_ms")
        assert values.size == 30
        assert values.min() >= 100.0  # none of the excluded rows leaked in

    def test_missing_cell_fails_closed(self, tmp_path: Path) -> None:
        with pytest.raises(rc.CalibrationCLIError, match="not found"):
            rc.load_arm_metric(tmp_path / "nope", "ttft_ms")

    def test_too_few_observations_fails_closed(self, tmp_path: Path) -> None:
        cell = tmp_path / "run" / "baselines" / "no_cache" / "trial_1"
        cell.mkdir(parents=True)
        pd.DataFrame(
            [
                {"example_id": f"e{i}", "error": "", "empty_generation": "",
                 "repeat_index": "0", "ttft_ms": float(i)}
                for i in range(5)
            ]
        ).to_csv(cell / "results.csv", index=False)
        with pytest.raises(rc.CalibrationCLIError, match="observations"):
            rc.load_arm_metric(tmp_path / "run", "ttft_ms")


# --------------------------------------------------------------------------- #
# Pooling + output-location guard
# --------------------------------------------------------------------------- #


class TestPoolAA:
    def test_counts_add_and_ci_recomputed(self) -> None:
        a = AAResult(n_splits=200, alpha=0.05, n_rejections=12,
                     fp_rate=0.06, ci_low=0.031, ci_high=0.102)
        b = AAResult(n_splits=200, alpha=0.05, n_rejections=8,
                     fp_rate=0.04, ci_low=0.017, ci_high=0.077)
        pooled = rc.pool_aa([a, b], alpha=0.05)
        assert pooled.n_splits == 400
        assert pooled.n_rejections == 20
        assert pooled.fp_rate == pytest.approx(0.05)
        assert pooled.ci_low < 0.05 < pooled.ci_high
        assert pooled.approximates_nominal

    def test_mixed_alpha_refused(self) -> None:
        a = AAResult(n_splits=10, alpha=0.05, n_rejections=1,
                     fp_rate=0.1, ci_low=0.0, ci_high=0.4)
        b = AAResult(n_splits=10, alpha=0.01, n_rejections=0,
                     fp_rate=0.0, ci_low=0.0, ci_high=0.3)
        with pytest.raises(rc.CalibrationCLIError, match="mixed alphas"):
            rc.pool_aa([a, b], alpha=0.05)

    def test_empty_refused(self) -> None:
        with pytest.raises(rc.CalibrationCLIError, match="no per-dataset"):
            rc.pool_aa([], alpha=0.05)


class TestOutDirGuard:
    def test_results_tree_refused(self, tmp_path: Path) -> None:
        with pytest.raises(rc.CalibrationCLIError, match="read-only"):
            rc._check_out_dir(tmp_path / "results" / "phase2" / "out")

    def test_normal_dir_allowed(self, tmp_path: Path) -> None:
        ok = rc._check_out_dir(tmp_path / "registration" / "cal")
        assert ok.name == "cal"


# --------------------------------------------------------------------------- #
# End-to-end on the synthetic archive + FUNCTION-level consumer proof
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def e2e(
    mini_archive: dict[str, Path], tmp_path_factory: pytest.TempPathFactory
) -> dict[str, object]:
    report, provenance = rc.run_calibration(
        # C7 (2026-10-10): the consumer gate's registered floor is 400 per leg.
        mini_archive, seed=123, alpha=0.05, aa_splits=400, injection_splits=400
    )
    out_dir = tmp_path_factory.mktemp("cal_out")
    paths = rc.write_outputs(report, provenance, out_dir)
    return {"report": report, "provenance": provenance, "paths": paths}


class TestEndToEnd:
    def test_gate_aa_is_pool_of_per_dataset_runs(self, e2e: dict) -> None:
        report = e2e["report"]
        prov = e2e["provenance"]
        per_ds = prov["aa_primary_per_dataset"]
        assert set(per_ds) == {"squad_v2", "hotpotqa"}
        assert report.aa.n_splits == sum(r["n_splits"] for r in per_ds.values())
        assert report.aa.n_rejections == sum(
            r["n_rejections"] for r in per_ds.values()
        )
        # Sanity of the machinery on well-behaved synthetic data: the A/A
        # false-positive rate approximates the nominal alpha (gate criterion).
        assert report.aa.approximates_nominal

    def test_injection_grid_shape_and_kinds(self, e2e: dict) -> None:
        report = e2e["report"]
        # 2 datasets x 3 families x 3 effects
        assert len(report.injections) == 18
        kinds = [inj.kind for inj in report.injections]
        assert kinds.count("flip") == 6
        assert kinds.count("shift") == 12
        # Unregistered simulation targets => nothing may gate on targets.
        assert all(inj.target_power is None for inj in report.injections)
        assert all(inj.meets_target is None for inj in report.injections)

    def test_written_files_exist_and_are_stamped(self, e2e: dict) -> None:
        paths = e2e["paths"]
        for key in ("report", "markdown", "provenance"):
            assert paths[key].is_file()
        md = paths["markdown"].read_text(encoding="utf-8")
        assert "CALIBRATION / DESIGN-INPUT ONLY" in md
        assert "NEVER scientific findings" in md
        prov = json.loads(paths["provenance"].read_text(encoding="utf-8"))
        assert "DESIGN-INPUT" in prov["stamp"]

    def test_provenance_records_required_fields(self, e2e: dict) -> None:
        prov = e2e["provenance"]
        assert prov["seed"] == 123
        assert prov["aa_splits_per_dataset"] == 400
        assert prov["injection_splits"] == 400
        assert set(prov["source_runs"]) == {"squad_v2", "hotpotqa"}
        assert "valid row = NOT error AND NOT empty_generation" in (
            prov["loader"]["validity_rule"]
        )
        assert prov["aa_arm"]["cell"] == "no_cache"
        # every labeled injection carries its seed and tie diagnostics
        for row in prov["injections"]:
            assert isinstance(row["seed"], int)
            assert "collision_probability" in row["diagnostics"]

    def test_consumer_parses_and_gate_passes(self, e2e: dict) -> None:
        """§9.7 proof: the campaign driver's loader + gate accept the artifact.

        FUNCTION-level only: run_campaign_analysis is imported, never invoked
        as a CLI; no confirmatory mode, no analysis_lock.json anywhere.
        """
        import run_campaign_analysis as rca

        report_path = e2e["paths"]["report"]
        loaded = rca.load_calibration_report(report_path)
        assert loaded == e2e["report"]  # lossless round trip of the gate schema
        summary = rca.check_calibration(loaded, report_path)
        assert summary["verdict"] == "PASS"
        assert summary["aa_approximates_nominal"] is True
        assert summary["n_injections"] == 18

    def test_failing_aa_is_refused_by_consumer_gate(self, tmp_path: Path) -> None:
        """A report whose A/A CI excludes alpha must be REFUSED (fail closed)."""
        import run_campaign_analysis as rca
        from src.analysis.stats.calibration import CalibrationReport

        from src.analysis.stats.calibration import _rejection_ci

        lo, hi = _rejection_ci(80, 400)  # C7: the gate recomputes the CI
        bad = CalibrationReport(
            seed=1,
            n_observations=100,
            aa=AAResult(n_splits=400, alpha=0.05, n_rejections=80,
                        fp_rate=0.2, ci_low=lo, ci_high=hi),
            injections=(),
        )
        path = bad.write(tmp_path / "calibration_report.json")
        loaded = rca.load_calibration_report(path)
        with pytest.raises(rca.CalibrationGateError, match="BLOCKED"):
            rca.check_calibration(loaded, path)

    def test_write_outputs_refuses_results_tree(
        self, e2e: dict, tmp_path: Path
    ) -> None:
        with pytest.raises(rc.CalibrationCLIError, match="read-only"):
            rc.write_outputs(
                e2e["report"], e2e["provenance"], tmp_path / "results" / "x"
            )


# --------------------------------------------------------------------------- #
# ADR-0178 block 2 (D4, D5): the S1 per-arm A/A pooled into the driver's shape
# --------------------------------------------------------------------------- #


def _aa_repeats(n: int = 1000, *, arms: dict[str, tuple[int, int]] | None = None,
                injection: bool = True) -> dict:
    """A cage-aa-repeats-v1 payload built from COUNTS: arms = name -> (A/A
    rejections, injection rejections); rates and exact CIs derived from them."""
    from src.analysis.stats.calibration import _rejection_ci

    arms = arms or {"armA": (2, 940), "armB": (1, 950), "armC": (0, 930)}
    out = {"schema": "cage-aa-repeats-v1", "alpha": 0.05, "n_resamples": n, "seed": 7,
           "rule": "one-sided", "injections": "run", "arms": {}, "pass": True}
    for name, (k, kp) in arms.items():
        lo, hi = _rejection_ci(k, n)
        arm = {"n_pairs": 100, "fp_rate": k / n, "ci_low": lo, "ci_high": hi, "pass": lo <= 0.05}
        if injection:
            plo, phi = _rejection_ci(kp, n)
            arm["injection"] = {"shift": 0.05, "power": kp / n, "ci_low": plo, "ci_high": phi, "n_capped": 0}
        out["arms"][name] = arm
    out["pass"] = all(a["pass"] for a in out["arms"].values())
    return out


class TestAaRepeatsToCalibrationReport:
    def test_pools_counts_exactly_and_the_driver_gate_passes(self, tmp_path: Path) -> None:
        import run_campaign_analysis as rca

        report = rc.calibration_report_from_aa_repeats(_aa_repeats(), source="aa.json")
        assert report.aa.n_splits == 3000 and report.aa.n_rejections == 3
        assert report.aa.fp_rate == pytest.approx(3 / 3000)
        assert report.n_observations == 300 and report.seed == 7
        (inj,) = report.injections
        assert inj.kind == "flip" and inj.effect_size == 0.05
        assert inj.n_splits == 3000 and inj.n_rejections == 940 + 950 + 930
        assert inj.target_power is None
        assert report.per_arm_gate["all_pass"] is True and report.per_arm_gate["n_arms"] == 3
        path = report.write(tmp_path / "calibration_report.json")
        loaded = rca.load_calibration_report(path)
        assert loaded == report
        summary = rca.check_calibration(loaded, path)
        assert summary["verdict"] == "PASS" and summary["per_arm_gate"] == {"n_arms": 3, "all_pass": True}

    def test_one_failing_arm_blocks_the_gate_even_when_the_pool_passes(self, tmp_path: Path) -> None:
        import run_campaign_analysis as rca

        # armC rejects 120 of 1000 (CI above 0.05): the pool (123/3000 = 0.041,
        # CI below 0.05) would PASS; D5 says the per-arm verdict governs.
        payload = _aa_repeats(arms={"armA": (2, 940), "armB": (1, 950), "armC": (120, 930)})
        assert payload["arms"]["armC"]["pass"] is False and payload["pass"] is False
        report = rc.calibration_report_from_aa_repeats(payload)
        assert report.aa.approximates_nominal is True  # the pool alone would pass
        assert report.per_arm_gate["all_pass"] is False
        assert report.per_arm_gate["arms_failing"] == ["armC"]
        path = report.write(tmp_path / "calibration_report.json")
        with pytest.raises(rca.CalibrationGateError, match="every arm passed"):
            rca.check_calibration(rca.load_calibration_report(path), path)

    def test_non_integer_rate_refuses_naming_the_arm(self) -> None:
        payload = _aa_repeats()
        payload["arms"]["armB"]["fp_rate"] = 0.0015  # 1.5 rejections of 1000
        with pytest.raises(rc.CalibrationCLIError, match="armB.*not an integer count"):
            rc.calibration_report_from_aa_repeats(payload)

    def test_stored_interval_must_be_the_exact_ci(self) -> None:
        payload = _aa_repeats()
        payload["arms"]["armA"]["ci_high"] += 0.01
        with pytest.raises(rc.CalibrationCLIError, match="armA.*not the exact binomial CI"):
            rc.calibration_report_from_aa_repeats(payload)

    def test_partial_injection_refuses(self) -> None:
        payload = _aa_repeats()
        del payload["arms"]["armC"]["injection"]
        with pytest.raises(rc.CalibrationCLIError, match="partial injection"):
            rc.calibration_report_from_aa_repeats(payload)

    def test_no_injection_is_written_and_the_gate_refuses_it(self, tmp_path: Path) -> None:
        import run_campaign_analysis as rca

        report = rc.calibration_report_from_aa_repeats(_aa_repeats(injection=False))
        assert report.injections == ()
        path = report.write(tmp_path / "calibration_report.json")
        with pytest.raises(rca.CalibrationGateError, match="no effect injection"):
            rca.check_calibration(rca.load_calibration_report(path), path)

    @pytest.mark.parametrize("mutate,match", [
        (lambda d: d.update(schema="other"), "not a cage-aa-repeats-v1"),
        (lambda d: d.update(n_resamples=0), "n_resamples"),
        (lambda d: d.update(arms={}), "no arms"),
        (lambda d: d["arms"]["armA"].pop("ci_low"), "lacks 'ci_low'"),
    ])
    def test_malformed_payloads_refuse(self, mutate, match) -> None:
        payload = _aa_repeats()
        mutate(payload)
        with pytest.raises(rc.CalibrationCLIError, match=match):
            rc.calibration_report_from_aa_repeats(payload)

    def test_cli_conversion_writes_the_sibling_by_default(self, tmp_path: Path) -> None:
        aa = tmp_path / "aa_repeats_run1.json"
        aa.write_text(json.dumps(_aa_repeats()), encoding="utf-8")
        assert rc.main(["--from-aa-repeats", str(aa)]) == 0
        sibling = tmp_path / "calibration_report_run1.json"
        assert sibling.is_file()
        loaded = json.loads(sibling.read_text(encoding="utf-8"))
        assert loaded["per_arm_gate"]["source"] == "aa_repeats_run1.json"
        assert set(loaded) == {"seed", "n_observations", "aa", "injections", "per_arm_gate"}

    def test_sibling_path_rule(self) -> None:
        assert rc.sibling_calibration_report_path(Path("/x/aa_repeats_r.json")) == Path("/x/calibration_report_r.json")
        assert rc.sibling_calibration_report_path(Path("/x/other.json")) == Path("/x/calibration_report_other.json")

    def test_campaign_mode_writes_both_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # the campaign branch of main: the per-arm report and its driver-shape
        # sibling beside it, written even when an arm fails (documented, never hidden)
        payload = _aa_repeats(arms={"armA": (2, 940), "armB": (120, 950)})
        monkeypatch.setattr(rc, "campaign_aa_report", lambda *a, **k: payload)
        out = tmp_path / "aa_repeats_run9.json"
        rc_code = rc.main(["--campaign-root", str(tmp_path), "--scoring-run-id", "s", "--out", str(out)])
        assert rc_code == 1  # armB fails the pod's own rule
        assert out.is_file()
        sibling = tmp_path / "calibration_report_run9.json"
        loaded = json.loads(sibling.read_text(encoding="utf-8"))
        assert loaded["per_arm_gate"]["all_pass"] is False
        assert loaded["per_arm_gate"]["arms_failing"] == ["armB"]


class TestAaRepeatsReviewFixes:
    """The 2026-10-27 review of block 2: H1, M1, L2, L3, L4, L6."""

    def test_edited_pass_on_a_failing_interval_refuses_h1(self) -> None:
        payload = _aa_repeats(arms={"a": (0, 950), "b": (0, 950), "c": (120, 930)})
        assert payload["arms"]["c"]["pass"] is False
        payload["arms"]["c"]["pass"] = True  # the edit the pool would have trusted
        with pytest.raises(rc.CalibrationCLIError, match="'c'.*contradicts the one-sided rule"):
            rc.calibration_report_from_aa_repeats(payload)

    def test_per_arm_verdict_is_recomputed_not_read(self) -> None:
        # an arm whose stored pass is consistent but whose rule fails lands in arms_failing
        payload = _aa_repeats(arms={"a": (0, 950), "b": (120, 930)})
        report = rc.calibration_report_from_aa_repeats(payload)
        assert report.per_arm_gate["arms_failing"] == ["b"] and report.per_arm_gate["all_pass"] is False

    def test_mixed_injection_shifts_refuse_m1(self) -> None:
        payload = _aa_repeats()
        payload["arms"]["armB"]["injection"]["shift"] = 0.10
        with pytest.raises(rc.CalibrationCLIError, match="2 different injection shifts"):
            rc.calibration_report_from_aa_repeats(payload)

    def test_injection_splits_count_the_arms_that_carry_the_shift(self) -> None:
        report = rc.calibration_report_from_aa_repeats(_aa_repeats())
        (inj,) = report.injections
        assert inj.n_splits == 3 * 1000 and inj.power == pytest.approx((940 + 950 + 930) / 3000)

    def test_capped_and_pair_totals_ride_the_gate_block_l2_l6(self) -> None:
        payload = _aa_repeats()
        payload["arms"]["armA"]["injection"]["n_capped"] = 4
        report = rc.calibration_report_from_aa_repeats(payload)
        assert report.per_arm_gate["n_capped_total"] == 4
        assert report.per_arm_gate["n_pairs_total"] == 300 == report.n_observations

    @pytest.mark.parametrize("key,where", [("fp_rate", "arms"), ("ci_low", "arms"), ("power", "injection")])
    def test_non_numeric_values_refuse_naming_the_arm_l4(self, key: str, where: str) -> None:
        payload = _aa_repeats()
        target = payload["arms"]["armC"] if where == "arms" else payload["arms"]["armC"]["injection"]
        target[key] = "0.1"
        with pytest.raises(rc.CalibrationCLIError, match="'armC'.*not a finite number"):
            rc.calibration_report_from_aa_repeats(payload)

    def test_a_sibling_write_failure_keeps_the_pod_verdict_and_exit_code_l3(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(rc, "campaign_aa_report", lambda *a, **k: _aa_repeats())
        monkeypatch.setattr(rc, "calibration_report_from_aa_repeats",
                            lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
        out = tmp_path / "aa_repeats_run3.json"
        rc_code = rc.main(["--campaign-root", str(tmp_path), "--scoring-run-id", "s", "--out", str(out)])
        assert rc_code == 0  # every arm passed; the sibling failure is reported, not fatal
        assert out.is_file() and not (tmp_path / "calibration_report_run3.json").exists()
        assert "driver-shape report NOT written (OSError" in capsys.readouterr().out
