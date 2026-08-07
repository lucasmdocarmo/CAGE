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
        mini_archive, seed=123, alpha=0.05, aa_splits=100, injection_splits=60
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
        assert prov["aa_splits_per_dataset"] == 100
        assert prov["injection_splits"] == 60
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

        bad = CalibrationReport(
            seed=1,
            n_observations=100,
            aa=AAResult(n_splits=400, alpha=0.05, n_rejections=80,
                        fp_rate=0.2, ci_low=0.162, ci_high=0.243),
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
