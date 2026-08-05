"""Tests for the L4 instrument-validation harness (charter D8 §8.6, D9 §9.13).

All data is synthetic and constructed so the correct answers (τ, precision,
recall, AUC, drift metrics) are known analytically — no model loading, no
network, no GPU (module contract: pure numpy/pandas).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.instrument_calibration import (
    ARTIFACT_KIND,
    DEFAULT_BIN_EDGES,
    DEFAULT_PRECISION_FLOOR,
    SCHEMA_VERSION,
    AnchorIdentity,
    DriftAudit,
    InstrumentCalibrationError,
    InstrumentCalibrationReport,
    anchor_fingerprint,
    assert_registrable,
    attach_drift_audit,
    calibrate_instrument,
    drift_audit,
    length_bin_gate,
    load_anchor,
    load_report,
    roc_auc,
    select_tau,
    write_report,
)

ANCHOR = AnchorIdentity(
    dataset="ragtruth", split="test", n_items=8, fingerprint_sha256="ab" * 32
)


def _analytic_anchor_frame() -> pd.DataFrame:
    """Anchor where τ is known analytically.

    Grounded scores {0.9, 0.8, 0.7, 0.6}; ungrounded {0.65, 0.3, 0.2, 0.1}.
    Candidate precisions (predicted grounded = score >= τ):
      τ=0.6  -> 4/5 = 0.80   (0.65 ungrounded included)
      τ=0.65 -> 3/4 = 0.75
      τ=0.7  -> 3/3 = 1.00   <- smallest τ achieving >= 0.90
    Lengths put the four high scores in one bin and the rest in another, with
    both classes in each bin and perfect separation inside each bin.
    """
    return pd.DataFrame(
        {
            "item_id": [f"it{i}" for i in range(8)],
            "score": [0.9, 0.8, 0.7, 0.6, 0.65, 0.3, 0.2, 0.1],
            "label": [1, 1, 1, 1, 0, 0, 0, 0],
            "context_length": [500, 500, 2000, 2000, 500, 500, 2000, 2000],
        }
    )


# --------------------------------------------------------------------------- #
# τ selection (D8 §8.6(c))
# --------------------------------------------------------------------------- #


class TestSelectTau:
    def test_analytic_tau(self) -> None:
        frame = _analytic_anchor_frame()
        result = select_tau(frame["score"], frame["label"], ANCHOR)
        assert result.tau == pytest.approx(0.7)
        assert result.precision_at_tau == pytest.approx(1.0)
        assert result.recall_at_tau == pytest.approx(0.75)
        assert result.specificity_at_tau == pytest.approx(1.0)
        assert result.n_predicted_grounded == 3
        assert result.precision_floor == DEFAULT_PRECISION_FLOOR

    def test_smallest_threshold_is_chosen_not_just_any(self) -> None:
        # Perfectly separated: every τ in the grounded range achieves 1.0;
        # the SMALLEST achieving threshold is the lowest grounded score.
        scores = [0.9, 0.7, 0.5, 0.2, 0.1]
        labels = [1, 1, 1, 0, 0]
        result = select_tau(scores, labels, ANCHOR)
        assert result.tau == pytest.approx(0.5)
        assert result.recall_at_tau == pytest.approx(1.0)

    def test_boundary_precision_exactly_at_floor_passes(self) -> None:
        # 9 grounded and 1 ungrounded at score >= 0.5: precision = 9/10 = 0.9.
        scores = [0.5 + 0.05 * i for i in range(9)] + [0.5, 0.1, 0.2]
        labels = [1] * 9 + [0, 0, 0]
        result = select_tau(scores, labels, ANCHOR, precision_floor=0.9)
        assert result.tau == pytest.approx(0.5)
        assert result.precision_at_tau == pytest.approx(0.9)

    def test_ties_at_threshold_count_all_tied_items(self) -> None:
        # Two items tied at 0.6, one of each class: predicted set at τ=0.6
        # includes BOTH -> precision 2/3 at τ=0.6; τ must move to 0.8.
        scores = [0.8, 0.6, 0.6, 0.1]
        labels = [1, 1, 0, 0]
        result = select_tau(scores, labels, ANCHOR, precision_floor=0.9)
        assert result.tau == pytest.approx(0.8)

    def test_anchor_identity_recorded_in_output(self) -> None:
        frame = _analytic_anchor_frame()
        result = select_tau(frame["score"], frame["label"], ANCHOR)
        assert result.anchor == ANCHOR
        payload = result.to_dict()
        assert payload["anchor"]["dataset"] == "ragtruth"
        assert payload["anchor"]["split"] == "test"
        assert payload["anchor"]["fingerprint_sha256"] == "ab" * 32

    def test_unreachable_floor_fails_closed(self) -> None:
        # Highest scores are ungrounded: no threshold reaches 0.9 precision.
        with pytest.raises(InstrumentCalibrationError, match="no threshold"):
            select_tau([0.9, 0.8, 0.4, 0.3], [0, 0, 1, 1], ANCHOR)

    def test_one_class_anchor_fails_closed(self) -> None:
        with pytest.raises(InstrumentCalibrationError, match="BOTH"):
            select_tau([0.9, 0.8], [1, 1], ANCHOR)

    def test_non_binary_labels_fail_closed(self) -> None:
        with pytest.raises(InstrumentCalibrationError, match="binary"):
            select_tau([0.9, 0.8], [1, 2], ANCHOR)

    def test_nan_scores_fail_closed(self) -> None:
        with pytest.raises(InstrumentCalibrationError, match="non-finite"):
            select_tau([0.9, float("nan")], [1, 0], ANCHOR)

    def test_length_mismatch_fails_closed(self) -> None:
        with pytest.raises(InstrumentCalibrationError, match="differ in length"):
            select_tau([0.9, 0.8, 0.7], [1, 0], ANCHOR)

    def test_invalid_floor_fails_closed(self) -> None:
        with pytest.raises(InstrumentCalibrationError, match="precision_floor"):
            select_tau([0.9, 0.1], [1, 0], ANCHOR, precision_floor=1.5)


# --------------------------------------------------------------------------- #
# ROC AUC (rank-sum identity, midrank ties)
# --------------------------------------------------------------------------- #


class TestRocAuc:
    def test_perfect_separation(self) -> None:
        assert roc_auc(
            np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 1, 0, 0])
        ) == pytest.approx(1.0)

    def test_inverted_separation(self) -> None:
        assert roc_auc(
            np.array([0.1, 0.2, 0.8, 0.9]), np.array([1, 1, 0, 0])
        ) == pytest.approx(0.0)

    def test_midrank_tie_handling_analytic(self) -> None:
        # Pairs: (0.5,0.5) tie=0.5, (0.5>0.3)=1, (0.7>0.5)=1, (0.7>0.3)=1
        # AUC = 3.5/4 = 0.875 (Hanley-McNeil rank-sum identity).
        assert roc_auc(
            np.array([0.5, 0.7, 0.5, 0.3]), np.array([1, 1, 0, 0])
        ) == pytest.approx(0.875)

    def test_one_class_fails_closed(self) -> None:
        with pytest.raises(InstrumentCalibrationError, match="both classes"):
            roc_auc(np.array([0.5, 0.7]), np.array([1, 1]))


# --------------------------------------------------------------------------- #
# Per-length-bin discrimination gate (D8 §8.6(b))
# --------------------------------------------------------------------------- #


class TestLengthBinGate:
    def test_all_bins_pass_with_perfect_separation(self) -> None:
        frame = _analytic_anchor_frame()
        gate = length_bin_gate(
            frame["score"],
            frame["label"],
            frame["context_length"],
            auc_floor=0.9,
            bin_edges=(0, 1024, 4096),
        )
        assert gate.passed
        assert len(gate.bins) == 2
        assert [b.auc for b in gate.bins] == pytest.approx([1.0, 1.0])
        assert gate.failed_bins == ()

    def test_failing_bin_fails_the_gate_and_is_named(self) -> None:
        # Bin 1 separated; bin 2 inverted (grounded scores BELOW ungrounded).
        scores = [0.9, 0.1, 0.2, 0.8]
        labels = [1, 0, 1, 0]
        lengths = [100, 100, 2000, 2000]
        gate = length_bin_gate(
            scores, labels, lengths, auc_floor=0.8, bin_edges=(0, 1024, 4096)
        )
        assert not gate.passed
        assert gate.bins[0].passed and not gate.bins[1].passed
        assert gate.failed_bins == (gate.bins[1].bin_label,)

    def test_boundary_auc_exactly_at_floor_passes(self) -> None:
        # AUC = 0.875 analytic (midrank tie case) in a single bin.
        gate = length_bin_gate(
            [0.5, 0.7, 0.5, 0.3],
            [1, 1, 0, 0],
            [10, 10, 10, 10],
            auc_floor=0.875,
            bin_edges=(0, 1024),
        )
        assert gate.passed
        assert gate.bins[0].auc == pytest.approx(0.875)

    def test_default_bin_edges_match_charter_grid(self) -> None:
        assert DEFAULT_BIN_EDGES == (
            0.0,
            1024.0,
            4096.0,
            8192.0,
            16384.0,
            32768.0,
            math.inf,
        )

    def test_default_edges_cover_all_charter_lengths(self) -> None:
        # Two items (one per class, separated) in each of the 6 default bins.
        lengths_per_bin = [512, 2048, 6000, 12000, 24000, 40000]
        scores, labels, lengths = [], [], []
        for length in lengths_per_bin:
            scores += [0.9, 0.1]
            labels += [1, 0]
            lengths += [length, length]
        gate = length_bin_gate(scores, labels, lengths, auc_floor=0.9)
        assert gate.passed
        assert len(gate.bins) == 6

    def test_empty_bin_fails_closed(self) -> None:
        with pytest.raises(InstrumentCalibrationError, match="EMPTY"):
            length_bin_gate(
                [0.9, 0.1],
                [1, 0],
                [100, 100],
                auc_floor=0.8,
                bin_edges=(0, 1024, 4096),
            )

    def test_one_class_bin_fails_closed(self) -> None:
        with pytest.raises(InstrumentCalibrationError, match="one class"):
            length_bin_gate(
                [0.9, 0.8, 0.9, 0.1],
                [1, 1, 1, 0],
                [100, 100, 2000, 2000],
                auc_floor=0.8,
                bin_edges=(0, 1024, 4096),
            )

    def test_length_outside_edges_fails_closed(self) -> None:
        with pytest.raises(InstrumentCalibrationError, match="outside"):
            length_bin_gate(
                [0.9, 0.1],
                [1, 0],
                [100, 9999],
                auc_floor=0.8,
                bin_edges=(0, 1024),
            )

    def test_unsorted_edges_fail_closed(self) -> None:
        with pytest.raises(InstrumentCalibrationError, match="strictly increasing"):
            length_bin_gate(
                [0.9, 0.1], [1, 0], [100, 100], auc_floor=0.8, bin_edges=(1024, 0)
            )

    def test_missing_floor_is_a_type_error(self) -> None:
        # auc_floor is deliberately required (pre-registered content, no
        # silent default) — omitting it must not fall back to anything.
        with pytest.raises(TypeError):
            length_bin_gate([0.9, 0.1], [1, 0], [100, 100])  # type: ignore[call-arg]

    def test_invalid_floor_fails_closed(self) -> None:
        with pytest.raises(InstrumentCalibrationError, match="auc_floor"):
            length_bin_gate([0.9, 0.1], [1, 0], [100, 100], auc_floor=0.0)


# --------------------------------------------------------------------------- #
# Drift audit (D8 §8.6(e))
# --------------------------------------------------------------------------- #


class TestDriftAudit:
    CAL = {"a": 0.9, "b": 0.6, "c": 0.2}

    def test_identical_scores_pass_with_zero_drift(self) -> None:
        result = drift_audit(
            self.CAL, dict(self.CAL), tau=0.5, max_mean_abs_delta=0.05
        )
        assert result.passed
        assert result.mean_abs_delta == pytest.approx(0.0)
        assert result.max_abs_delta == pytest.approx(0.0)
        assert result.flip_rate_at_tau == pytest.approx(0.0)
        assert result.n_items == 3

    def test_drift_beyond_mean_threshold_fails(self) -> None:
        drifted = {k: v + 0.2 for k, v in self.CAL.items()}
        result = drift_audit(self.CAL, drifted, tau=0.5, max_mean_abs_delta=0.1)
        assert not result.passed
        assert result.mean_abs_delta == pytest.approx(0.2)

    def test_flip_rate_analytic(self) -> None:
        # 'b' crosses τ=0.5 downward (0.6 -> 0.4); a and c stay put.
        recheck = {"a": 0.9, "b": 0.4, "c": 0.2}
        result = drift_audit(self.CAL, recheck, tau=0.5, max_flip_rate=0.5)
        assert result.flip_rate_at_tau == pytest.approx(1 / 3)
        assert result.passed
        strict = drift_audit(self.CAL, recheck, tau=0.5, max_flip_rate=0.1)
        assert not strict.passed

    def test_boundary_drift_exactly_at_threshold_passes(self) -> None:
        drifted = {k: v + 0.1 for k, v in self.CAL.items()}
        result = drift_audit(self.CAL, drifted, tau=2.0, max_mean_abs_delta=0.1)
        assert result.passed

    def test_mismatched_item_sets_fail_closed(self) -> None:
        with pytest.raises(InstrumentCalibrationError, match="differ"):
            drift_audit(self.CAL, {"a": 0.9, "b": 0.6}, tau=0.5, max_flip_rate=0.1)
        with pytest.raises(InstrumentCalibrationError, match="differ"):
            drift_audit(
                self.CAL,
                {**self.CAL, "zzz": 0.5},
                tau=0.5,
                max_flip_rate=0.1,
            )

    def test_duplicate_ids_fail_closed(self) -> None:
        dup = pd.Series([0.9, 0.8], index=["a", "a"])
        with pytest.raises(InstrumentCalibrationError, match="duplicate"):
            drift_audit(dup, dup, tau=0.5, max_flip_rate=0.1)

    def test_no_threshold_fails_closed(self) -> None:
        with pytest.raises(InstrumentCalibrationError, match="at least one threshold"):
            drift_audit(self.CAL, dict(self.CAL), tau=0.5)

    def test_nan_scores_fail_closed(self) -> None:
        bad = {"a": 0.9, "b": float("nan"), "c": 0.2}
        with pytest.raises(InstrumentCalibrationError, match="non-finite"):
            drift_audit(self.CAL, bad, tau=0.5, max_flip_rate=0.1)


# --------------------------------------------------------------------------- #
# Anchor loading + fingerprint identity
# --------------------------------------------------------------------------- #


class TestLoadAnchor:
    def test_dataframe_and_jsonl_give_identical_identity(
        self, tmp_path: Path
    ) -> None:
        frame = _analytic_anchor_frame()
        path = tmp_path / "anchor.jsonl"
        path.write_text(
            "\n".join(json.dumps(rec) for rec in frame.to_dict(orient="records"))
        )
        _, id_df = load_anchor(frame, dataset="ragtruth", split="test")
        _, id_jsonl = load_anchor(str(path), dataset="ragtruth", split="test")
        assert id_df == id_jsonl
        assert id_df.n_items == 8
        assert len(id_df.fingerprint_sha256) == 64

    def test_fingerprint_is_row_order_invariant(self) -> None:
        frame = _analytic_anchor_frame()
        shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
        cols = ["score", "label", "context_length"]
        assert anchor_fingerprint(frame, cols) == anchor_fingerprint(shuffled, cols)

    def test_fingerprint_changes_with_content(self) -> None:
        frame = _analytic_anchor_frame()
        altered = frame.copy()
        altered.loc[0, "score"] = 0.11
        cols = ["score", "label", "context_length"]
        assert anchor_fingerprint(frame, cols) != anchor_fingerprint(altered, cols)

    def test_missing_file_fails_closed(self) -> None:
        with pytest.raises(InstrumentCalibrationError, match="not found"):
            load_anchor("/nonexistent/anchor.jsonl", dataset="d", split="s")

    def test_malformed_jsonl_line_fails_closed(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text('{"score": 0.9, "label": 1, "context_length": 5}\n{oops\n')
        with pytest.raises(InstrumentCalibrationError, match="line 2"):
            load_anchor(str(path), dataset="d", split="s")

    def test_non_object_jsonl_line_fails_closed(self, tmp_path: Path) -> None:
        path = tmp_path / "arr.jsonl"
        path.write_text("[1, 2, 3]\n")
        with pytest.raises(InstrumentCalibrationError, match="not a JSON object"):
            load_anchor(str(path), dataset="d", split="s")

    def test_empty_jsonl_file_fails_closed(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("\n\n")
        with pytest.raises(InstrumentCalibrationError, match="empty"):
            load_anchor(str(path), dataset="d", split="s")

    def test_missing_column_fails_closed(self) -> None:
        frame = _analytic_anchor_frame().drop(columns=["label"])
        with pytest.raises(InstrumentCalibrationError, match="missing required"):
            load_anchor(frame, dataset="d", split="s")

    def test_empty_frame_fails_closed(self) -> None:
        frame = _analytic_anchor_frame().iloc[0:0]
        with pytest.raises(InstrumentCalibrationError, match="empty"):
            load_anchor(frame, dataset="d", split="s")

    def test_blank_identity_fails_closed(self) -> None:
        with pytest.raises(InstrumentCalibrationError, match="identity"):
            load_anchor(_analytic_anchor_frame(), dataset="", split="s")

    def test_negative_lengths_fail_closed(self) -> None:
        frame = _analytic_anchor_frame()
        frame.loc[0, "context_length"] = -5
        with pytest.raises(InstrumentCalibrationError, match="negative"):
            load_anchor(frame, dataset="d", split="s")


# --------------------------------------------------------------------------- #
# End-to-end report, JSON artifact, and the D9 refuse-to-register hook
# --------------------------------------------------------------------------- #


def _passing_report() -> InstrumentCalibrationReport:
    return calibrate_instrument(
        _analytic_anchor_frame(),
        instrument_name="lettucedetect",
        instrument_version="0.1.7",
        dataset="ragtruth",
        split="test",
        auc_floor=0.9,
        bin_edges=(0, 1024, 4096),
    )


class TestReportAndArtifact:
    def test_calibrate_instrument_end_to_end(self) -> None:
        report = _passing_report()
        assert report.passed
        assert report.tau_selection.tau == pytest.approx(0.7)
        assert report.length_bin_gate.passed
        assert report.drift is None
        assert report.anchor.dataset == "ragtruth"
        # τ selection carries the SAME anchor identity as the report.
        assert report.tau_selection.anchor == report.anchor

    def test_json_roundtrip(self, tmp_path: Path) -> None:
        report = _passing_report()
        path = write_report(report, tmp_path / "sub" / "calibration.json")
        loaded = load_report(path)
        assert loaded == report
        payload = json.loads(path.read_text())
        assert payload["schema_version"] == SCHEMA_VERSION
        assert payload["artifact"] == ARTIFACT_KIND
        assert payload["passed"] is True
        assert payload["instrument"] == {
            "name": "lettucedetect",
            "version": "0.1.7",
        }
        assert set(payload) >= {
            "anchor",
            "tau_selection",
            "length_bin_gate",
            "drift_audit",
        }

    def test_roundtrip_with_drift_attached(self, tmp_path: Path) -> None:
        report = _passing_report()
        tau = report.tau_selection.tau
        scores = {"a": 0.9, "b": 0.2}
        audit = drift_audit(scores, dict(scores), tau=tau, max_flip_rate=0.1)
        with_drift = attach_drift_audit(report, audit)
        assert with_drift.passed
        loaded = load_report(write_report(with_drift, tmp_path / "c.json"))
        assert loaded.drift == audit

    def test_attach_drift_with_wrong_tau_fails_closed(self) -> None:
        report = _passing_report()
        audit = drift_audit(
            {"a": 0.9}, {"a": 0.9}, tau=0.123, max_flip_rate=0.1
        )
        with pytest.raises(InstrumentCalibrationError, match="frozen"):
            attach_drift_audit(report, audit)

    def test_assert_registrable_passes_on_good_report(self) -> None:
        assert_registrable(_passing_report())  # must not raise

    def test_assert_registrable_blocks_failed_gate(self) -> None:
        report = _passing_report()
        # Force a failed bin (frozen dataclasses -> rebuild via replace).
        import dataclasses as dc

        failed_bin = dc.replace(report.length_bin_gate.bins[0], passed=False)
        failed_gate = dc.replace(
            report.length_bin_gate,
            bins=(failed_bin, *report.length_bin_gate.bins[1:]),
        )
        broken = dc.replace(report, length_bin_gate=failed_gate)
        assert not broken.passed
        with pytest.raises(InstrumentCalibrationError, match="BLOCKED"):
            assert_registrable(broken)

    def test_assert_registrable_blocks_failed_drift(self) -> None:
        report = _passing_report()
        tau = report.tau_selection.tau
        audit = drift_audit(
            {"a": 0.9}, {"a": 0.3}, tau=tau, max_mean_abs_delta=0.05
        )
        broken = attach_drift_audit(report, audit)
        assert not broken.passed
        with pytest.raises(InstrumentCalibrationError, match="drift"):
            assert_registrable(broken)

    def test_load_report_missing_file_fails_closed(self) -> None:
        with pytest.raises(InstrumentCalibrationError, match="not found"):
            load_report("/nonexistent/calibration.json")

    def test_load_report_malformed_json_fails_closed(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        with pytest.raises(InstrumentCalibrationError, match="not valid JSON"):
            load_report(path)

    def test_load_report_missing_keys_fails_closed(self, tmp_path: Path) -> None:
        path = tmp_path / "partial.json"
        path.write_text(json.dumps({"schema_version": SCHEMA_VERSION}))
        with pytest.raises(InstrumentCalibrationError, match="missing required"):
            load_report(path)

    def test_load_report_wrong_kind_fails_closed(self, tmp_path: Path) -> None:
        report = _passing_report()
        payload = report.to_dict()
        payload["artifact"] = "something_else"
        path = tmp_path / "kind.json"
        path.write_text(json.dumps(payload))
        with pytest.raises(InstrumentCalibrationError, match="artifact kind"):
            load_report(path)

    def test_load_report_tampered_verdict_fails_closed(self, tmp_path: Path) -> None:
        # A hand-edited passed=true over failing sections must not load.
        report = _passing_report()
        payload = report.to_dict()
        payload["length_bin_gate"]["bins"][0]["passed"] = False
        payload["length_bin_gate"]["passed"] = False
        # leave top-level "passed": true -> inconsistent artifact
        path = tmp_path / "tampered.json"
        path.write_text(json.dumps(payload))
        with pytest.raises(InstrumentCalibrationError, match="inconsistent"):
            load_report(path)

    def test_markdown_report_names_sections_and_verdicts(self) -> None:
        report = _passing_report()
        text = report.to_markdown()
        assert "§8.6(c)" in text and "§8.6(b)" in text
        assert "lettucedetect@0.1.7" in text
        assert "PASS" in text
        assert "ragtruth/test" in text
