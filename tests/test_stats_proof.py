"""Tests for the D9 proof-of-honesty modules (§9.6-§9.13).

Covers src.analysis.stats.{calibration, blinding, power_sim, ledger, prereg}:
happy paths, determinism, and the fail-loud contracts (tamper, double-unblind,
sealed overwrite, blocked registration).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import mannwhitneyu

from src.analysis.stats.blinding import (
    AlreadyUnblindedError,
    BlindingError,
    SealedMapTamperError,
    scramble_labels,
    unblind,
)
from src.analysis.stats.calibration import (
    CalibrationError,
    aa_split_half,
    build_report,
    inject_effect,
    recover_power,
)
from src.analysis.stats import ledger as ledger_mod
from src.analysis.stats.ledger import (
    LedgerError,
    hash_artifacts,
    read_ledger,
    verify_ledger,
    write_ledger,
)
from src.analysis.stats.calibration import InjectionResult
from src.analysis.stats.families import compile_family_map
from src.analysis.stats.power_sim import (
    PowerSimError,
    required_n,
    shift_injection,
    simulate_campaign,
    tie_flip_injection,
    wilcoxon_signed_p,
)
from src.analysis.stats.prereg import PreregError, assemble_preregistration


def _mwu_p(a: np.ndarray, b: np.ndarray) -> float:
    return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)


@pytest.fixture(scope="module")
def null_data() -> np.ndarray:
    return np.random.default_rng(7).normal(0.0, 1.0, size=240)


# ---------------------------------------------------------------- calibration


class TestCalibration:
    def test_aa_fp_rate_approximates_alpha(self, null_data: np.ndarray) -> None:
        result = aa_split_half(null_data, _mwu_p, n_splits=300, seed=11, alpha=0.05)
        assert result.n_splits == 300
        assert result.fp_rate == result.n_rejections / 300
        assert result.ci_low <= result.fp_rate <= result.ci_high
        # Same-arm splits of null data: the exact CI must cover nominal alpha.
        assert result.approximates_nominal

    def test_aa_is_deterministic(self, null_data: np.ndarray) -> None:
        a = aa_split_half(null_data, _mwu_p, n_splits=50, seed=3)
        b = aa_split_half(null_data, _mwu_p, n_splits=50, seed=3)
        assert a == b

    def test_recover_power_large_shift(self, null_data: np.ndarray) -> None:
        result = recover_power(
            null_data, _mwu_p, effect_size=1.0, n_splits=100, seed=5, target_power=0.8
        )
        # 1 sd shift at n≈120/half: Mann-Whitney power is essentially 1.
        assert result.power >= 0.9
        assert result.meets_target is True

    def test_recover_power_null_effect_stays_near_alpha(
        self, null_data: np.ndarray
    ) -> None:
        result = recover_power(null_data, _mwu_p, effect_size=0.0, n_splits=100, seed=5)
        assert result.power <= 0.15

    def test_meets_target_is_fail_closed(self) -> None:
        # The 2026-08-02 regression: 2/5 rejections (power 0.40, exact CI up
        # to 0.853) must FAIL a 0.8 target — ci_high >= target was fail-open
        # and passed easiest when the evidence was weakest.
        weak = InjectionResult(
            effect_size=0.5, kind="shift", n_splits=5, alpha=0.05,
            n_rejections=2, power=0.4, ci_low=0.053, ci_high=0.853,
            target_power=0.8,
        )
        assert weak.meets_target is False
        strong = InjectionResult(
            effect_size=0.5, kind="shift", n_splits=200, alpha=0.05,
            n_rejections=170, power=0.85, ci_low=0.79, ci_high=0.90,
            target_power=0.8,
        )
        assert strong.meets_target is True
        no_target = InjectionResult(
            effect_size=0.5, kind="shift", n_splits=5, alpha=0.05,
            n_rejections=2, power=0.4, ci_low=0.053, ci_high=0.853,
        )
        assert no_target.meets_target is None

    def test_inject_shift(self) -> None:
        out = inject_effect(np.zeros(4), 0.5, "shift")
        assert np.allclose(out, 0.5)

    def test_inject_flip_binary(self) -> None:
        data = np.zeros(10)
        out = inject_effect(data, 0.3, "flip", seed=1)
        assert out.sum() == 3
        assert np.isin(out, (0.0, 1.0)).all()
        # deterministic given seed
        assert np.array_equal(out, inject_effect(data, 0.3, "flip", seed=1))

    def test_inject_flip_rejects_non_binary(self) -> None:
        with pytest.raises(CalibrationError, match="binary"):
            inject_effect(np.array([0.0, 0.5, 1.0]), 0.2, "flip", seed=1)

    def test_inject_flip_requires_seed(self) -> None:
        with pytest.raises(CalibrationError, match="seed"):
            inject_effect(np.array([0.0, 1.0]), 0.5, "flip")

    def test_unknown_kind_raises(self) -> None:
        with pytest.raises(CalibrationError, match="unknown injection kind"):
            inject_effect(np.zeros(4), 0.5, "scale")  # type: ignore[arg-type]

    def test_bad_inputs_raise(self, null_data: np.ndarray) -> None:
        with pytest.raises(CalibrationError, match="n_splits"):
            aa_split_half(null_data, _mwu_p, n_splits=0, seed=1)
        with pytest.raises(CalibrationError, match="alpha"):
            aa_split_half(null_data, _mwu_p, n_splits=10, seed=1, alpha=1.5)
        with pytest.raises(CalibrationError, match="at least"):
            aa_split_half([1.0, 2.0], _mwu_p, n_splits=10, seed=1)
        with pytest.raises(CalibrationError, match="non-finite"):
            aa_split_half([1.0, np.nan, 2.0, 3.0], _mwu_p, n_splits=10, seed=1)

    def test_invalid_p_from_test_fn_raises(self, null_data: np.ndarray) -> None:
        with pytest.raises(CalibrationError, match="invalid p-value"):
            aa_split_half(null_data, lambda a, b: 2.0, n_splits=5, seed=1)

    def test_report_markdown(self, null_data: np.ndarray) -> None:
        report = build_report(
            null_data,
            _mwu_p,
            n_splits=60,
            seed=9,
            effect_sizes=(1.0,),
            target_power=0.8,
        )
        md = report.to_markdown()
        assert "A/A split-half" in md
        assert "Effect injection" in md
        assert f"{report.aa.fp_rate:.4f}" in md
        assert report.injections[0].kind == "shift"


# ------------------------------------------------------------------- blinding


class TestBlinding:
    @pytest.fixture()
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "arm": ["gold-fresh", "corpus-reuse", "retr-fresh"] * 4,
                "ttft_ms": np.arange(12, dtype=float),
            }
        )

    def test_scramble_and_unblind_round_trip(
        self, frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        sealed = tmp_path / "sealed_map.json"
        blinded, path = scramble_labels(frame, "arm", seed=42, sealed_map_path=sealed)
        assert path == sealed and sealed.exists()
        assert set(blinded["arm"]) == {"ARM-01", "ARM-02", "ARM-03"}
        assert (blinded["ttft_ms"] == frame["ttft_ms"]).all()
        # group sizes survive relabeling
        assert sorted(blinded["arm"].value_counts()) == sorted(
            frame["arm"].value_counts()
        )
        mapping = unblind(sealed, tmp_path / "unblind.log")
        restored = blinded["arm"].map(mapping)
        assert (restored == frame["arm"]).all()

    def test_sealed_map_is_sha256_keyed(self, frame: pd.DataFrame, tmp_path: Path) -> None:
        sealed = tmp_path / "sealed.json"
        scramble_labels(frame, "arm", seed=1, sealed_map_path=sealed)
        doc = json.loads(sealed.read_text())
        canonical = json.dumps(doc["mapping"], sort_keys=True, separators=(",", ":"))
        assert doc["map_sha256"] == hashlib.sha256(canonical.encode()).hexdigest()
        assert doc["unblinded_utc"] is None

    def test_second_unblind_raises_and_log_has_one_event(
        self, frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        sealed = tmp_path / "sealed.json"
        log = tmp_path / "unblind.log"
        scramble_labels(frame, "arm", seed=1, sealed_map_path=sealed)
        unblind(sealed, log)
        with pytest.raises(AlreadyUnblindedError):
            unblind(sealed, log)
        events = log.read_text().strip().splitlines()
        assert len(events) == 1
        assert json.loads(events[0])["event"] == "UNBLIND"

    def test_tampered_mapping_raises(self, frame: pd.DataFrame, tmp_path: Path) -> None:
        sealed = tmp_path / "sealed.json"
        scramble_labels(frame, "arm", seed=1, sealed_map_path=sealed)
        doc = json.loads(sealed.read_text())
        doc["mapping"]["gold-fresh"] = "ARM-99"
        sealed.write_text(json.dumps(doc))
        with pytest.raises(SealedMapTamperError, match="hash mismatch"):
            unblind(sealed, tmp_path / "unblind.log")

    def test_refuses_existing_seal(self, frame: pd.DataFrame, tmp_path: Path) -> None:
        sealed = tmp_path / "sealed.json"
        scramble_labels(frame, "arm", seed=1, sealed_map_path=sealed)
        with pytest.raises(BlindingError, match="already exists"):
            scramble_labels(frame, "arm", seed=2, sealed_map_path=sealed)

    def test_missing_column_and_single_label_raise(self, tmp_path: Path) -> None:
        with pytest.raises(BlindingError, match="arm_col"):
            scramble_labels(pd.DataFrame({"x": [1]}), "arm", 1, tmp_path / "s.json")
        one_arm = pd.DataFrame({"arm": ["gold-fresh"] * 3})
        with pytest.raises(BlindingError, match="distinct arm labels"):
            scramble_labels(one_arm, "arm", 1, tmp_path / "s.json")


# ------------------------------------------------------------------ power_sim


class TestPowerSim:
    @staticmethod
    def _normal_noise(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.normal(0.0, 1.0, size=n)

    def test_power_table_shape_and_determinism(self) -> None:
        kwargs = dict(
            effect_grid=[0.0, 1.0],
            n_grid=[20, 60],
            variance_model=self._normal_noise,
            n_sims=40,
            seed=123,
        )
        table = simulate_campaign(**kwargs)
        assert list(table.columns) == [
            "effect",
            "n",
            "n_sims",
            "alpha",
            "power",
            "ci_low",
            "ci_high",
        ]
        assert len(table) == 4
        pd.testing.assert_frame_equal(table, simulate_campaign(**kwargs))

    def test_power_monotone_in_effect_and_n(self) -> None:
        table = simulate_campaign(
            [0.0, 1.0], [10, 80], self._normal_noise, n_sims=60, seed=7
        ).set_index(["effect", "n"])
        assert table.loc[(0.0, 80), "power"] <= 0.15  # null: ≈ alpha
        assert table.loc[(1.0, 80), "power"] >= 0.9  # 1 sd shift, n=80
        assert table.loc[(1.0, 80), "power"] >= table.loc[(1.0, 10), "power"]

    def test_required_n(self) -> None:
        table = simulate_campaign(
            [1.0], [10, 40, 80], self._normal_noise, n_sims=60, seed=7
        )
        n = required_n(table, 1.0, target_power=0.8)
        assert n in (10, 40, 80)
        with pytest.raises(PowerSimError, match="not on the simulated grid"):
            required_n(table, 2.0)

    def test_required_n_unreachable_raises(self) -> None:
        table = simulate_campaign([0.0], [10], self._normal_noise, n_sims=40, seed=7)
        with pytest.raises(PowerSimError, match="extend n_grid"):
            required_n(table, 0.0, target_power=0.8)

    def test_all_zero_diffs_p_is_one(self) -> None:
        assert wilcoxon_signed_p(np.zeros(10)) == 1.0

    def test_bad_inputs_raise(self) -> None:
        with pytest.raises(PowerSimError, match="non-empty"):
            simulate_campaign([], [10], self._normal_noise, 10, 1)
        with pytest.raises(PowerSimError, match=">= 2"):
            simulate_campaign([0.5], [1], self._normal_noise, 10, 1)
        with pytest.raises(PowerSimError, match="n_sims"):
            simulate_campaign([0.5], [10], self._normal_noise, 0, 1)

    def test_bad_variance_model_shape_raises(self) -> None:
        def wrong_shape(rng: np.random.Generator, n: int) -> np.ndarray:
            return rng.normal(size=n + 1)

        with pytest.raises(PowerSimError, match="shape"):
            simulate_campaign([0.5], [10], wrong_shape, 5, 1)

    @staticmethod
    def _pilot_tie_noise(rng: np.random.Generator, n: int) -> np.ndarray:
        # The pilot's own tie structure (audit §2.5): ~95% exact-zero diffs,
        # ~5% ±1 discordant.
        diffs = np.zeros(n)
        discordant = rng.random(n) < 0.05
        diffs[discordant] = rng.choice([-1.0, 1.0], size=int(discordant.sum()))
        return diffs

    def test_tie_heavy_power_tracks_discordant_pairs_not_n(self) -> None:
        # The 2026-08-02 CRITICAL regression: additive injection converts every
        # tie into signed evidence (power ≈ 1 at effect 0.02, n=300); the
        # honest discordant-pair injection powers the flip process instead.
        kwargs = dict(
            effect_grid=[0.02],
            n_grid=[300],
            variance_model=self._pilot_tie_noise,
            n_sims=100,
            seed=42,
        )
        additive = simulate_campaign(**kwargs, injection=shift_injection)
        flip = simulate_campaign(**kwargs, injection=tie_flip_injection)
        assert additive.loc[0, "power"] >= 0.9  # the dishonest number
        assert flip.loc[0, "power"] <= 0.6  # the discordant-pair process
        assert flip.loc[0, "power"] < additive.loc[0, "power"]

    def test_tie_flip_injection_is_deterministic(self) -> None:
        kwargs = dict(
            effect_grid=[0.02],
            n_grid=[200],
            variance_model=self._pilot_tie_noise,
            n_sims=30,
            seed=9,
            injection=tie_flip_injection,
        )
        pd.testing.assert_frame_equal(
            simulate_campaign(**kwargs), simulate_campaign(**kwargs)
        )

    def test_tie_flip_requires_ties(self) -> None:
        with pytest.raises(PowerSimError, match="no ties"):
            simulate_campaign(
                [0.05], [50], self._normal_noise, 5, 1,
                injection=tie_flip_injection,
            )

    def test_tie_flip_effect_is_a_fraction(self) -> None:
        with pytest.raises(PowerSimError, match="fraction"):
            simulate_campaign(
                [1.5], [50], self._pilot_tie_noise, 5, 1,
                injection=tie_flip_injection,
            )


# --------------------------------------------------------------------- ledger


class TestLedger:
    @pytest.fixture()
    def artifacts(self, tmp_path: Path) -> tuple[Path, list[Path]]:
        base = tmp_path / "run"
        (base / "baselines").mkdir(parents=True)
        a = base / "baselines" / "results.csv"
        b = base / "qa_evidence.jsonl"
        a.write_text("cell,ttft\nx,1\n")
        b.write_text('{"q": 1}\n')
        return base, [a, b]

    def test_hash_matches_hashlib(self, artifacts: tuple[Path, list[Path]]) -> None:
        base, files = artifacts
        ledger = hash_artifacts(files, base_dir=base)
        assert ledger["baselines/results.csv"] == hashlib.sha256(
            files[0].read_bytes()
        ).hexdigest()
        assert sorted(ledger) == ["baselines/results.csv", "qa_evidence.jsonl"]

    def test_write_verify_intact(
        self, artifacts: tuple[Path, list[Path]], tmp_path: Path
    ) -> None:
        base, files = artifacts
        ledger_path = write_ledger(hash_artifacts(files, base_dir=base), tmp_path / "ledger.json")
        assert verify_ledger(ledger_path, base) == []
        assert read_ledger(ledger_path) == hash_artifacts(files, base_dir=base)

    def test_verify_reports_modification_and_missing(
        self, artifacts: tuple[Path, list[Path]], tmp_path: Path
    ) -> None:
        base, files = artifacts
        ledger_path = write_ledger(hash_artifacts(files, base_dir=base), tmp_path / "ledger.json")
        files[0].write_text("cell,ttft\nx,999\n")  # post-seal edit
        files[1].unlink()
        mismatches = verify_ledger(ledger_path, base)
        assert len(mismatches) == 2
        assert any(m.startswith("HASH-MISMATCH baselines/results.csv") for m in mismatches)
        assert "MISSING qa_evidence.jsonl" in mismatches

    def test_sealed_ledger_refuses_overwrite(
        self, artifacts: tuple[Path, list[Path]], tmp_path: Path
    ) -> None:
        base, files = artifacts
        ledger = hash_artifacts(files, base_dir=base)
        path = write_ledger(ledger, tmp_path / "ledger.json")
        with pytest.raises(LedgerError, match="sealed"):
            write_ledger(ledger, path)

    def test_tampered_ledger_raises(
        self, artifacts: tuple[Path, list[Path]], tmp_path: Path
    ) -> None:
        base, files = artifacts
        path = write_ledger(hash_artifacts(files, base_dir=base), tmp_path / "ledger.json")
        doc = json.loads(path.read_text())
        doc["entries"]["qa_evidence.jsonl"] = "0" * 64
        path.write_text(json.dumps(doc))
        with pytest.raises(LedgerError, match="self-hash mismatch"):
            verify_ledger(path, base)

    def test_missing_artifact_and_outside_base_raise(self, tmp_path: Path) -> None:
        with pytest.raises(LedgerError, match="does not exist"):
            hash_artifacts([tmp_path / "ghost.csv"], base_dir=tmp_path)
        real = tmp_path / "a.csv"
        real.write_text("x")
        with pytest.raises(LedgerError, match="outside base_dir"):
            hash_artifacts([real], base_dir=tmp_path / "elsewhere")
        with pytest.raises(LedgerError, match="empty ledger"):
            write_ledger({}, tmp_path / "ledger.json")

    # ---- task #129 / H7: extra-file detection + key-join semantics ----

    def test_extra_file_reported_only_with_extra_roots(
        self, artifacts: tuple[Path, list[Path]], tmp_path: Path
    ) -> None:
        base, files = artifacts
        ledger_path = write_ledger(
            hash_artifacts(files, base_dir=base), tmp_path / "ledger.json"
        )
        sneaky = base / "baselines" / "sneaky.jsonl"
        sneaky.write_text("{}\n")
        # Default behavior unchanged: no extra_roots -> the seal is blind to it.
        assert verify_ledger(ledger_path, base) == []
        mismatches = verify_ledger(ledger_path, base, extra_roots=[base])
        assert mismatches == ["EXTRA baselines/sneaky.jsonl"]

    def test_extra_sweep_scoped_root_ignores_legal_siblings(
        self, artifacts: tuple[Path, list[Path]], tmp_path: Path
    ) -> None:
        # The §6 layout: scoring/ (and index/, analysis/) are LEGAL post-seal
        # siblings at the run root — a sweep scoped to baselines/ must not
        # flag them, while still catching additions inside the scoped root.
        base, files = artifacts
        ledger_path = write_ledger(
            hash_artifacts(files, base_dir=base), tmp_path / "ledger.json"
        )
        scoring = base / "scoring" / "s01" / "cells"
        scoring.mkdir(parents=True)
        (scoring / "qa_scores.jsonl").write_text("{}\n")
        assert verify_ledger(ledger_path, base, extra_roots=[base / "baselines"]) == []
        (base / "baselines" / "added.csv").write_text("x\n")
        assert verify_ledger(ledger_path, base, extra_roots=[base / "baselines"]) == [
            "EXTRA baselines/added.csv"
        ]

    def test_ledger_file_itself_is_not_extra(
        self, artifacts: tuple[Path, list[Path]]
    ) -> None:
        base, files = artifacts
        ledger_path = write_ledger(
            hash_artifacts(files, base_dir=base), base / "ledger.json"
        )
        assert verify_ledger(ledger_path, base, extra_roots=[base]) == []

    def test_extra_sweep_follows_directory_symlinks(
        self, artifacts: tuple[Path, list[Path]], tmp_path: Path
    ) -> None:
        # rglob-style sweeps are blind to directory symlinks (H7); the walk
        # must follow them and report the smuggled file at its LOGICAL path.
        base, files = artifacts
        ledger_path = write_ledger(
            hash_artifacts(files, base_dir=base), tmp_path / "ledger.json"
        )
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "smuggled.jsonl").write_text("{}\n")
        (base / "baselines" / "nested").symlink_to(outside, target_is_directory=True)
        mismatches = verify_ledger(ledger_path, base, extra_roots=[base])
        assert mismatches == ["EXTRA baselines/nested/smuggled.jsonl"]

    def test_extra_sweep_symlink_cycle_terminates(
        self, artifacts: tuple[Path, list[Path]], tmp_path: Path
    ) -> None:
        base, files = artifacts
        ledger_path = write_ledger(
            hash_artifacts(files, base_dir=base), tmp_path / "ledger.json"
        )
        (base / "baselines" / "loop").symlink_to(base, target_is_directory=True)
        mismatches = verify_ledger(ledger_path, base, extra_roots=[base])
        # Terminates, and everything visible through the cycle that is not a
        # sealed entry (or the ledger file) is EXTRA — no false negatives.
        assert all(m.startswith("EXTRA ") for m in mismatches)

    def test_extra_root_must_exist_and_be_inside_base(
        self, artifacts: tuple[Path, list[Path]], tmp_path: Path
    ) -> None:
        base, files = artifacts
        ledger_path = write_ledger(
            hash_artifacts(files, base_dir=base), tmp_path / "ledger.json"
        )
        with pytest.raises(LedgerError, match="not a directory"):
            verify_ledger(ledger_path, base, extra_roots=[base / "ghost"])
        outside = tmp_path / "outside-root"
        outside.mkdir()
        with pytest.raises(LedgerError, match="outside base_dir"):
            verify_ledger(ledger_path, base, extra_roots=[outside])

    def test_absolute_and_traversal_keys_refused(self, tmp_path: Path) -> None:
        # `Path(base) / "/abs"` silently DISCARDS base — an absolute-key ledger
        # would verify against unrelated files. Refused at write AND verify.
        target = tmp_path / "abs.csv"
        target.write_text("x")
        with pytest.raises(LedgerError, match="absolute"):
            write_ledger(
                {target.as_posix(): "0" * 64}, tmp_path / "ledger_abs.json"
            )
        with pytest.raises(LedgerError, match=r"\.\."):
            write_ledger({"../escape.csv": "0" * 64}, tmp_path / "ledger_dots.json")
        # A hand-tampered ledger with such keys fails verification loudly too.
        doc_path = tmp_path / "ledger_forged.json"
        entries = {target.as_posix(): "0" * 64}
        doc = {
            "ledger_version": 1,
            "algorithm": "sha256",
            "created_utc": "2026-08-14T00:00:00+00:00",
            "entries_sha256": ledger_mod._entries_sha256(entries),
            "entries": entries,
        }
        doc_path.write_text(json.dumps(doc))
        with pytest.raises(LedgerError, match="absolute"):
            verify_ledger(doc_path, tmp_path)


# --------------------------------------------------------------------- prereg


def _passing_report() -> object:
    data = np.random.default_rng(2).normal(0.0, 1.0, size=200)
    return build_report(
        data, _mwu_p, n_splits=80, seed=4, effect_sizes=(1.0,), target_power=0.8
    )


def _family_map() -> pd.DataFrame:
    # Hand-built frame in the compile_family_map schema (§9.3) — kept minimal
    # for the validation-error tests; composition with the real compiler is
    # covered by TestPrereg.test_composes_with_compiled_family_map.
    return pd.DataFrame(
        {
            "contrast_id": [4, 4, 8],
            "name": ["B6 vs B3 (SQuAD)", "B6 vs B3 (MuSiQue)", "B10 vs B3"],
            "tier": ["primary", "primary", "secondary"],
            "family": ["F1", "F1", "F1"],
            "group": ["A", "A", "A"],
            "metric": ["ttft", "ttft", "predicate"],
            "dataset": ["squad_v2", "musique", "squad_v2"],
            "correction": ["none (co-primary set rule)", "none (co-primary set rule)", "holm"],
            "sidedness": ["two-sided", "two-sided", "one-sided (greater)"],
            "unit": ["per_query", "per_query", "binary"],
            "alpha": [0.05, 0.05, 0.05],
        }
    )


class TestPrereg:
    def test_assembles_all_sections(self) -> None:
        text = assemble_preregistration(
            _family_map(),
            {"grounding rate": "|Δ| < 2 pp"},
            _passing_report(),
            "a" * 40,
        )
        for heading in (
            "## 1. Primary endpoints",
            "## 2. Family map",
            "## 3. Equivalence margins",
            "## 4. Power",
            "## 5. Exclusions",
            "## 6. One-look policy",
            "## 7. Calibration report",
            "## 8. Amendment log",
        ):
            assert heading in text
        assert "B6 vs B3 (MuSiQue)" in text
        # pipe-escaped inside the rendered margin table (2026-08-02 fix) so
        # the companion bound no longer splits the cell
        assert r"\|Cliff's δ\| < 0.147" in text
        assert f"`{'a' * 40}`" in text
        assert "serving yield" in text  # Y naming, not veridical goodput
        # no power table given -> explicit blocking placeholder, not silence
        assert "BLOCKING" in text

    def test_power_table_rendered_when_given(self) -> None:
        power = pd.DataFrame(
            {"effect": [1.0], "n": [40], "n_sims": [60], "alpha": [0.05],
             "power": [0.9], "ci_low": [0.8], "ci_high": [0.96]}
        )
        text = assemble_preregistration(
            _family_map(),
            {"grounding rate": "|Δ| < 2 pp"},
            _passing_report(),
            "b" * 40,
            power_table=power,
        )
        assert "BLOCKING" not in text
        assert "| 1.0 | 40 |" in text

    def test_missing_columns_and_empty_map_raise(self) -> None:
        report = _passing_report()
        with pytest.raises(PreregError, match="missing required columns"):
            assemble_preregistration(
                pd.DataFrame({"contrast": ["x"]}), {"m": "y"}, report, "a" * 40
            )
        with pytest.raises(PreregError, match="empty"):
            assemble_preregistration(
                _family_map().iloc[0:0], {"m": "y"}, report, "a" * 40
            )

    def test_bad_tier_alpha_sha_margins_raise(self) -> None:
        report = _passing_report()
        bad_tier = _family_map().assign(tier=["primary", "headline", "secondary"])
        with pytest.raises(PreregError, match="unknown tiers"):
            assemble_preregistration(bad_tier, {"m": "y"}, report, "a" * 40)
        bad_alpha = _family_map().assign(alpha=[0.05, 1.5, 0.05])
        with pytest.raises(PreregError, match="alpha"):
            assemble_preregistration(bad_alpha, {"m": "y"}, report, "a" * 40)
        with pytest.raises(PreregError, match="git_sha"):
            assemble_preregistration(_family_map(), {"m": "y"}, report, "NOT-A-SHA")
        with pytest.raises(PreregError, match="margins"):
            assemble_preregistration(_family_map(), {}, report, "a" * 40)

    def test_failed_calibration_blocks_registration(self) -> None:
        # A test_fn that always rejects drives the A/A FP rate to 1.0 -> BLOCKED.
        data = np.random.default_rng(2).normal(size=100)
        broken = build_report(data, lambda a, b: 0.0, n_splits=40, seed=4)
        with pytest.raises(PreregError, match="BLOCKED"):
            assemble_preregistration(_family_map(), {"m": "y"}, broken, "a" * 40)

    def test_missed_injection_target_blocks_registration(self) -> None:
        # A/A passes (never rejects -> FP CI covers alpha) but the injected
        # effect is never recovered -> the §9.7 injection gate must block.
        data = np.random.default_rng(2).normal(size=200)
        never_reject = build_report(
            data, lambda a, b: 1.0, n_splits=40, seed=4,
            effect_sizes=(1.0,), target_power=0.8,
        )
        with pytest.raises(PreregError, match="injection targets missed"):
            assemble_preregistration(
                _family_map(), {"m": "y"}, never_reject, "a" * 40
            )

    def test_composes_with_compiled_family_map(self) -> None:
        # The 2026-08-02 regression: §9.3 compiler output feeds §9.13 assembly
        # DIRECTLY — no ad-hoc rename step, no markdown pipe corruption.
        fm = compile_family_map(["squad_v2", "hotpotqa", "musique", "qasper"])
        text = assemble_preregistration(
            fm, {"grounding rate": "|Δ| < 2 pp"}, _passing_report(), "c" * 40
        )
        # the registered content that used to be missing/prose-only
        assert "B12 vs B3" in text
        assert "truth_tax" in text
        assert "lambda_star_onset" in text
        assert "tost" in text
        assert "falsification" in text
        # family_id pipes are escaped so cells never split
        assert r"A\|ttft\|squad_v2" in text
        section = text.split("## 2. Family map")[1].split("## 3.")[0]
        body = [
            line
            for line in section.splitlines()
            if line.startswith("|") and not set(line) <= {"|", "-"}
        ]
        assert len(body) == len(fm) + 1  # header + one line per registered row
        for line in body:
            cells = re.split(r"(?<!\\)\|", line)
            assert len(cells) - 2 == len(fm.columns), line

    # ---------------------------------------------------------------- #
    # D8 §8.6 L4 instrument calibration is wired into the §9.13 chain
    # ---------------------------------------------------------------- #

    @staticmethod
    def _passing_instrument_report():
        from src.evaluation.instrument_calibration import calibrate_instrument

        anchor = pd.DataFrame(
            {
                "item_id": [f"it{i}" for i in range(8)],
                "score": [0.9, 0.8, 0.7, 0.6, 0.65, 0.3, 0.2, 0.1],
                "label": [1, 1, 1, 1, 0, 0, 0, 0],
                "context_length": [500, 500, 2000, 2000, 500, 500, 2000, 2000],
            }
        )
        return calibrate_instrument(
            anchor,
            instrument_name="lettucedetect",
            instrument_version="0.1.7",
            dataset="ragtruth",
            split="test",
            auc_floor=0.9,
            bin_edges=(0, 1024, 4096),
        )

    def test_instrument_calibration_embedded_under_section_7(self) -> None:
        inst = self._passing_instrument_report()
        text = assemble_preregistration(
            _family_map(),
            {"grounding rate": "|Δ| < 2 pp"},
            _passing_report(),
            "d" * 40,
            instrument_calibrations=[inst],
        )
        # The L4 artifact rides the section-7 calibration heading, beside the
        # §9.7 stats report (the to_markdown concatenation it was built for).
        section = text.split("## 7. Calibration report")[1].split("## 8.")[0]
        assert "Instrument calibration report (D8 §8.6)" in section
        assert "lettucedetect@0.1.7" in section

    def test_failed_instrument_calibration_blocks_registration(self) -> None:
        import dataclasses as dc

        inst = self._passing_instrument_report()
        failed_bin = dc.replace(inst.length_bin_gate.bins[0], passed=False)
        failed_gate = dc.replace(
            inst.length_bin_gate,
            bins=(failed_bin, *inst.length_bin_gate.bins[1:]),
        )
        broken = dc.replace(inst, length_bin_gate=failed_gate)
        assert not broken.passed
        with pytest.raises(PreregError, match="instrument calibration FAILED"):
            assemble_preregistration(
                _family_map(),
                {"grounding rate": "|Δ| < 2 pp"},
                _passing_report(),
                "d" * 40,
                instrument_calibrations=[broken],
            )

    def test_no_instrument_reports_keeps_existing_behavior(self) -> None:
        # Default () preserves the pre-wiring assembly exactly (no L4 section).
        text = assemble_preregistration(
            _family_map(),
            {"grounding rate": "|Δ| < 2 pp"},
            _passing_report(),
            "d" * 40,
        )
        assert "Instrument calibration report (D8 §8.6)" not in text
