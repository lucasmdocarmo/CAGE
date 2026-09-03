"""Tests for src/analysis/conditioned_curves.py (W4.11, PUBLICATION.md §8.12),
the figure_pipeline renderers, and the driver's conditioned-curves pass.

Covered:

1. Pinned bins: ρ_own interval labels (open-ended >=1.00 overcommit bin,
   fail-loud on NaN/negative), the evidence-position vocabulary
   (first/early/middle/late/absent with None-honesty on missing inputs), and
   the pressure-grid bin (labeled no-coords for F1, refusal on half-set
   coordinates).
2. Join correctness on synthetic fixtures for quality|ρ_own and
   quality|evidence-position×pressure (group means, counts, ordering,
   fail-closed column contracts). The ρ curve is conditioned per
   mechanism×engine cell identity (§8.12 "per mechanism × engine"):
   ``mechanism_engine_key`` strips ONLY the pressure coordinates, distinct
   cells are never pooled into one bin (Simpson pin), and a non-canonical
   row key refuses.
3. quality|policy-event refuses through the SAME owner-gated S2 stub as
   degradation.join_policy_events (one reason string).
4. Renderer smoke (figure_pipeline): both §8.12 renderers produce a PNG from
   the aggregate frames and fail loud on column/metric contract violations.
5. The driver pass (run_conditioned_curves_pass): emits
   conditioned_curves.json + figures, names every skip (missing
   own_accounting artifact, unscored metric), counts unbinnable trials,
   records the S2 stub and both curves' registered conditioning, and is
   suppressed under §9.8 blinding. Its evidence join REFUSES duplicate
   (example_id, record_index) rows within one artifact file (the H3/#127
   replay hazard, matching load_per_query) while still joining replayed
   rows that carry distinct record_index values.

Deterministic fixtures only; no GPU, no models.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = REPO_ROOT / "scripts" / "4_analysis"
for _p in (str(_SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import figure_pipeline as fp  # noqa: E402
import run_campaign_analysis as rca  # noqa: E402
from src.analysis import conditioned_curves as cc  # noqa: E402
from src.analysis.degradation import S2_POLICY_EVENT_JOIN_UNAVAILABLE  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Pinned bins
# ---------------------------------------------------------------------------


class TestRhoOwnBin:
    @pytest.mark.parametrize(
        "rho,expected",
        [
            (0.0, "[0.00,0.25)"),
            (0.1, "[0.00,0.25)"),
            (0.25, "[0.25,0.50)"),
            (0.6, "[0.50,0.75)"),
            (0.8, "[0.75,0.90)"),
            (0.95, "[0.90,1.00)"),
            (1.0, ">=1.00"),
            (1.7, ">=1.00"),  # overcommit is a labeled state, never clipped
        ],
    )
    def test_pinned_edges(self, rho: float, expected: str) -> None:
        assert cc.rho_own_bin(rho) == expected

    @pytest.mark.parametrize("bad", [float("nan"), -0.1, None, True, "0.5"])
    def test_bad_rho_fails_loud(self, bad: Any) -> None:
        with pytest.raises(cc.ConditionedCurveError):
            cc.rho_own_bin(bad)


class TestEvidencePositionBin:
    def test_gold_absent_is_its_own_bin(self) -> None:
        assert cc.evidence_position_bin(-1, None) == ("absent", None)

    def test_first_needs_no_count(self) -> None:
        assert cc.evidence_position_bin(0, None) == ("first", None)

    @pytest.mark.parametrize(
        "pos,n,expected",
        [(1, 10, "early"), (5, 10, "middle"), (9, 10, "late"), (2.0, 4, "middle")],
    )
    def test_fractional_bins(self, pos: Any, n: int, expected: str) -> None:
        assert cc.evidence_position_bin(pos, n) == (expected, None)

    def test_missing_position_is_named_none(self) -> None:
        bin_label, reason = cc.evidence_position_bin(None, 5)
        assert bin_label is None
        assert "gold_position_in_prompt absent" in reason

    def test_missing_count_is_named_none(self) -> None:
        bin_label, reason = cc.evidence_position_bin(2, None)
        assert bin_label is None
        assert "n_served_contexts unknown" in reason

    def test_inconsistent_row_is_named_none(self) -> None:
        bin_label, reason = cc.evidence_position_bin(3, 3)
        assert bin_label is None
        assert "inconsistent row" in reason

    def test_malformed_position_is_named_none(self) -> None:
        bin_label, reason = cc.evidence_position_bin("x", 5)
        assert bin_label is None
        assert "not a valid served index" in reason


class TestPressureBin:
    def test_grid_point_label(self) -> None:
        assert cc.pressure_bin_of(0.5, 0.9) == "r0.5|lam0.9"

    def test_absent_coords_are_labeled_absence(self) -> None:
        assert cc.pressure_bin_of(None, None) == cc.NO_PRESSURE_COORDS_BIN
        assert (
            cc.pressure_bin_of(float("nan"), float("nan"))
            == cc.NO_PRESSURE_COORDS_BIN
        )

    def test_half_set_coordinates_refuse(self) -> None:
        with pytest.raises(cc.ConditionedCurveError, match="half-set"):
            cc.pressure_bin_of(0.5, None)

    def test_malformed_coordinate_refuses(self) -> None:
        with pytest.raises(cc.ConditionedCurveError):
            cc.pressure_bin_of("x", 0.9)
        with pytest.raises(cc.ConditionedCurveError):
            cc.pressure_bin_of(True, 0.9)


# ---------------------------------------------------------------------------
# 2. Join correctness
# ---------------------------------------------------------------------------


# Canonical D7 row keys (arm|retriever|policy|topology|engine|model|family
# [+ coords]) — the §8.12 curve identity is the coordinate-free prefix.
_KEY_A_LOW = "rag|bm25|none|single|vllm|qwen3-8b|F2|r0.5|lam0.9"
_KEY_A_HIGH = "rag|bm25|none|single|vllm|qwen3-8b|F2|r0.9|lam1.2"
_KEY_B = "rag|bm25|none|single|sglang|qwen3-8b|F2|r0.5|lam0.9"
_CURVE_A = "rag|bm25|none|single|vllm|qwen3-8b|F2"
_CURVE_B = "rag|bm25|none|single|sglang|qwen3-8b|F2"


def _rho_windows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # Cell A on squad_v2: two windows in [0.00,0.25) at one pressure
            # coordinate, one in [0.90,1.00) at ANOTHER coordinate — the
            # swept coords belong to ONE curve (§8.12 identity).
            {"row_key": _KEY_A_LOW, "dataset": "squad_v2",
             "window_key": "squad_v2-01", "rho_own": 0.10,
             "metric": "f1_score", "value": 0.8, "n": 10},
            {"row_key": _KEY_A_LOW, "dataset": "squad_v2",
             "window_key": "squad_v2-02", "rho_own": 0.20,
             "metric": "f1_score", "value": 0.6, "n": 10},
            {"row_key": _KEY_A_HIGH, "dataset": "squad_v2",
             "window_key": "squad_v2-03", "rho_own": 0.95,
             "metric": "f1_score", "value": 0.4, "n": 5},
            # hotpotqa rides along, separate group.
            {"row_key": _KEY_A_LOW, "dataset": "hotpotqa",
             "window_key": "hotpotqa-01", "rho_own": 0.95,
             "metric": "f1_score", "value": 0.5, "n": 4},
            # Cell B (engine differs) lands in cell A's LOW bin with a
            # different quality — pooled, it would drag A's mean to ~0.47
            # (the Simpson hazard the per-cell conditioning forbids).
            {"row_key": _KEY_B, "dataset": "squad_v2",
             "window_key": "squad_v2-04", "rho_own": 0.12,
             "metric": "f1_score", "value": 0.0, "n": 10},
        ]
    )


class TestMechanismEngineKey:
    def test_strips_only_the_pressure_coords(self) -> None:
        assert cc.mechanism_engine_key(_KEY_A_LOW) == _CURVE_A
        assert cc.mechanism_engine_key(_KEY_A_HIGH) == _CURVE_A
        assert cc.mechanism_engine_key(_KEY_B) == _CURVE_B

    def test_coordinate_free_key_is_identity(self) -> None:
        assert cc.mechanism_engine_key(_CURVE_A) == _CURVE_A

    @pytest.mark.parametrize("bad", ["c1", "a|b|c", "a||c|d|e|f|g", ""])
    def test_non_canonical_key_refuses(self, bad: str) -> None:
        with pytest.raises(cc.ConditionedCurveError, match="7 canonical axes"):
            cc.mechanism_engine_key(bad)


class TestQualityByRhoOwn:
    def test_group_means_and_counts(self) -> None:
        curve = cc.quality_by_rho_own(_rho_windows())
        squad_a = curve[
            (curve["dataset"] == "squad_v2") & (curve["curve_key"] == _CURVE_A)
        ]
        low = squad_a[squad_a["rho_bin"] == "[0.00,0.25)"].iloc[0]
        assert low["n_windows"] == 2
        assert low["n_requests"] == 20
        assert low["mean_value"] == pytest.approx(0.7)
        assert low["rho_min"] == pytest.approx(0.10)
        assert low["rho_max"] == pytest.approx(0.20)
        # The high-ρ window sits at a DIFFERENT pressure coordinate but the
        # SAME mechanism×engine identity: one curve spans the sweep.
        high = squad_a[squad_a["rho_bin"] == "[0.90,1.00)"].iloc[0]
        assert high["n_windows"] == 1
        assert high["mean_value"] == pytest.approx(0.4)
        # Bins are ordered by occupancy within each dataset×metric×curve.
        assert list(squad_a["rho_bin"]) == ["[0.00,0.25)", "[0.90,1.00)"]

    def test_cells_are_never_pooled_into_one_bin(self) -> None:
        # §8.12 "per mechanism × engine": cell B shares cell A's dataset and
        # ρ bin but stays its own curve row — A's bin mean is UNCHANGED by
        # B's windows (no arm-composition/Simpson confound).
        curve = cc.quality_by_rho_own(_rho_windows())
        low = curve[
            (curve["dataset"] == "squad_v2") & (curve["rho_bin"] == "[0.00,0.25)")
        ]
        assert sorted(low["curve_key"]) == sorted([_CURVE_A, _CURVE_B])
        a_mean = low[low["curve_key"] == _CURVE_A]["mean_value"].iloc[0]
        b_mean = low[low["curve_key"] == _CURVE_B]["mean_value"].iloc[0]
        assert a_mean == pytest.approx(0.7)  # NOT the pooled ~0.4667
        assert b_mean == pytest.approx(0.0)

    def test_non_canonical_row_key_refuses(self) -> None:
        bad = _rho_windows()
        bad.loc[0, "row_key"] = "c1"
        with pytest.raises(cc.ConditionedCurveError, match="7 canonical axes"):
            cc.quality_by_rho_own(bad)

    def test_missing_column_refuses(self) -> None:
        with pytest.raises(cc.ConditionedCurveError, match="missing required"):
            cc.quality_by_rho_own(_rho_windows().drop(columns=["rho_own"]))

    def test_empty_frame_refuses(self) -> None:
        with pytest.raises(cc.ConditionedCurveError, match="empty"):
            cc.quality_by_rho_own(_rho_windows().iloc[0:0])


def _evidence_trials() -> pd.DataFrame:
    rows = []
    for value in (0.9, 0.7):
        rows.append({"dataset": "squad_v2", "evidence_bin": "first",
                     "pressure_bin": "r0.5|lam0.9", "metric": "f1_score",
                     "value": value})
    rows.append({"dataset": "squad_v2", "evidence_bin": "middle",
                 "pressure_bin": "r0.5|lam0.9", "metric": "f1_score",
                 "value": 0.2})
    rows.append({"dataset": "squad_v2", "evidence_bin": "absent",
                 "pressure_bin": cc.NO_PRESSURE_COORDS_BIN,
                 "metric": "f1_score", "value": 0.0})
    return pd.DataFrame(rows)


class TestQualityByEvidencePosition:
    def test_group_means_and_bin_order(self) -> None:
        curve = cc.quality_by_evidence_position(_evidence_trials())
        first = curve[curve["evidence_bin"] == "first"].iloc[0]
        assert first["n_trials"] == 2
        assert first["mean_value"] == pytest.approx(0.8)
        pressured = curve[curve["pressure_bin"] == "r0.5|lam0.9"]
        # Within a pressure bin the rows follow the pinned vocabulary order.
        assert list(pressured["evidence_bin"]) == ["first", "middle"]

    def test_unknown_evidence_bin_refuses(self) -> None:
        bad = _evidence_trials()
        bad.loc[0, "evidence_bin"] = "somewhere"
        with pytest.raises(cc.ConditionedCurveError, match="unknown evidence"):
            cc.quality_by_evidence_position(bad)

    def test_missing_column_refuses(self) -> None:
        with pytest.raises(cc.ConditionedCurveError, match="missing required"):
            cc.quality_by_evidence_position(
                _evidence_trials().drop(columns=["pressure_bin"])
            )


class TestPolicyEventStub:
    def test_refuses_with_the_shared_s2_reason(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            cc.policy_event_curve()
        assert str(excinfo.value) == S2_POLICY_EVENT_JOIN_UNAVAILABLE
        assert cc.S2_POLICY_EVENT_JOIN_UNAVAILABLE == (
            S2_POLICY_EVENT_JOIN_UNAVAILABLE
        )


# ---------------------------------------------------------------------------
# 4. Renderer smoke (figure_pipeline conventions)
# ---------------------------------------------------------------------------


class TestRenderers:
    def test_rho_own_curve_renders(self, tmp_path: Path) -> None:
        curve = cc.quality_by_rho_own(_rho_windows())
        out = fp.plot_quality_vs_rho_own(
            curve,
            tmp_path / "rho.png",
            config=fp.RhoOwnCurveConfig(metric="f1_score", title="t"),
        )
        assert out.is_file() and out.stat().st_size > 0

    def test_rho_own_refuses_mixed_metrics(self, tmp_path: Path) -> None:
        frame = cc.quality_by_rho_own(_rho_windows())
        other = frame.copy()
        other["metric"] = "grounding_score"
        with pytest.raises(fp.FigureDataError, match="one metric per figure"):
            fp.plot_quality_vs_rho_own(
                pd.concat([frame, other]),
                tmp_path / "x.png",
                config=fp.RhoOwnCurveConfig(metric="f1_score"),
            )

    def test_rho_own_refuses_missing_columns(self, tmp_path: Path) -> None:
        with pytest.raises(fp.FigureDataError, match="missing required"):
            fp.plot_quality_vs_rho_own(
                pd.DataFrame({"dataset": ["d"], "metric": ["m"]}),
                tmp_path / "x.png",
                config=fp.RhoOwnCurveConfig(metric="m"),
            )

    def test_rho_own_refuses_missing_curve_key(self, tmp_path: Path) -> None:
        # The §8.12 per-cell conditioning column is load-bearing: an
        # aggregate without it (a pooled pre-fix frame) must refuse, not
        # silently render one pooled line.
        curve = cc.quality_by_rho_own(_rho_windows()).drop(
            columns=["curve_key"]
        )
        with pytest.raises(fp.FigureDataError, match="missing required"):
            fp.plot_quality_vs_rho_own(
                curve,
                tmp_path / "x.png",
                config=fp.RhoOwnCurveConfig(metric="f1_score"),
            )

    def test_rho_own_refuses_over_color_budget(self, tmp_path: Path) -> None:
        # 9 distinct cells exceed the pinned 8-color colorblind-safe budget
        # — the encoding refuses (I11: never silently truncate levels).
        frames = []
        for i in range(9):
            frame = _rho_windows().iloc[[0]].copy()
            frame["row_key"] = f"arm{i}|bm25|none|single|vllm|m|F2|r0.5|lam0.9"
            frames.append(frame)
        curve = cc.quality_by_rho_own(pd.concat(frames, ignore_index=True))
        with pytest.raises(ValueError, match="colorblind-safe budget"):
            fp.plot_quality_vs_rho_own(
                curve,
                tmp_path / "x.png",
                config=fp.RhoOwnCurveConfig(metric="f1_score"),
            )

    def test_evidence_position_heatmap_renders(self, tmp_path: Path) -> None:
        curve = cc.quality_by_evidence_position(_evidence_trials())
        out = fp.plot_quality_vs_evidence_position(
            curve,
            tmp_path / "evidence.png",
            config=fp.EvidencePositionConfig(metric="f1_score", title="t"),
        )
        assert out.is_file() and out.stat().st_size > 0

    def test_evidence_position_refuses_unknown_bin(self, tmp_path: Path) -> None:
        curve = cc.quality_by_evidence_position(_evidence_trials())
        curve.loc[0, "evidence_bin"] = "elsewhere"
        with pytest.raises(fp.FigureDataError, match="unknown evidence bins"):
            fp.plot_quality_vs_evidence_position(
                curve,
                tmp_path / "x.png",
                config=fp.EvidencePositionConfig(metric="f1_score"),
            )

    def test_empty_config_metric_refuses(self) -> None:
        with pytest.raises(fp.FigureConfigError):
            fp.RhoOwnCurveConfig(metric="")
        with pytest.raises(fp.FigureConfigError):
            fp.EvidencePositionConfig(metric="")


# ---------------------------------------------------------------------------
# 5. The driver pass
# ---------------------------------------------------------------------------


def _pass_fixture(tmp_path: Path) -> tuple[Path, pd.DataFrame, Path]:
    """Two windows on disk: one with a usable own_accounting artifact and
    binnable evidence rows, one with neither (named-skip territory)."""
    run_dir = tmp_path / "run"
    analysis_dir = tmp_path / "run" / "analysis" / "t0"
    rows = [
        {"example_id": "e001", "f1_score": 0.9, "gold_position_in_prompt": 0},
        {"example_id": "e002", "f1_score": 0.5, "gold_position_in_prompt": -1},
    ]
    evidence = [
        {"example_id": "e001", "used_contexts": ["a", "b", "c"]},
        {"example_id": "e002", "used_contexts": ["a", "b", "c"]},
    ]
    windows = {
        "cells/cellF2/window_squad_v2-01": {
            "row_key": _KEY_A_LOW, "budget_r": 0.5, "rate_frac": 0.9,
            "rho": 0.42,
        },
        "cells/cellF2/window_squad_v2-02": {
            "row_key": _KEY_A_LOW, "budget_r": 0.5, "rate_frac": 0.9,
            "rho": None,  # no own_accounting artifact at all
        },
    }
    index_rows = []
    for ordinal, (window, meta) in enumerate(sorted(windows.items()), start=1):
        window_dir = run_dir / window
        window_dir.mkdir(parents=True)
        (window_dir / "requests.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
        (window_dir / "qa_evidence.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in evidence), encoding="utf-8"
        )
        if meta["rho"] is not None:
            own_path = (
                analysis_dir / rca.OWN_ACCOUNTING_DIRNAME / window
                / rca.OWN_ACCOUNTING_NAME
            )
            own_path.parent.mkdir(parents=True)
            own_path.write_text(
                json.dumps({"rho_own": meta["rho"]}), encoding="utf-8"
            )
        index_rows.append(
            {
                "row_key": meta["row_key"],
                "dataset": "squad_v2",
                "window_key": f"squad_v2-{ordinal:02d}",
                "window_dir": window,
                "budget_r": meta["budget_r"],
                "rate_frac": meta["rate_frac"],
            }
        )
    analysis_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, pd.DataFrame(index_rows), analysis_dir


def test_pass_emits_artifact_with_named_skips(tmp_path: Path) -> None:
    run_dir, index, analysis_dir = _pass_fixture(tmp_path)
    document = rca.run_conditioned_curves_pass(
        run_dir, index, analysis_dir, "DESIGN-INPUT-ONLY",
        blinding_active=False, metrics=("f1_score", "grounding_score"),
    )
    assert document is not None
    out_path = analysis_dir / rca.CONDITIONED_CURVES_NAME
    assert out_path.is_file()
    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    assert on_disk["mode_stamp"] == "DESIGN-INPUT-ONLY"

    rho = on_disk["quality_vs_rho_own"]
    # Window 01 joined; window 02 skipped naming the absent T2.5 artifact;
    # grounding_score skipped naming the unscored column — nothing silent.
    assert len(rho["windows"]) == 1
    assert rho["windows"][0]["rho_own"] == pytest.approx(0.42)
    assert rho["windows"][0]["value"] == pytest.approx(0.7)
    reasons = " | ".join(s["reason"] for s in rho["skips"])
    assert "no own_accounting.json" in reasons
    assert "'grounding_score'" in reasons
    assert rho["curve"][0]["rho_bin"] == "[0.25,0.50)"
    # §8.12: the curve is conditioned per mechanism×engine cell identity —
    # each record carries the coordinate-free curve_key, and the recorded
    # conditioning note names the registration.
    assert rho["curve"][0]["curve_key"] == _CURVE_A
    assert "per mechanism" in rho["conditioning"]

    evidence = on_disk["quality_vs_evidence_position"]
    assert "pooled across cells" in evidence["conditioning"]
    curve = pd.DataFrame(evidence["curve"])
    assert set(curve["evidence_bin"]) == {"first", "absent"}
    assert set(curve["pressure_bin"]) == {"r0.5|lam0.9"}
    # 2 windows × (first + absent) trials joined for f1_score.
    assert evidence["n_trials_joined"] == 4
    assert any(
        "grounding_score" in reason for reason in evidence["skipped_trials"]
    )

    policy = on_disk["quality_vs_policy_event"]
    assert policy["available"] is False
    assert policy["reason"] == S2_POLICY_EVENT_JOIN_UNAVAILABLE

    figure_files = [f["file"] for f in on_disk["figures"] if "file" in f]
    assert "conditioned_rho_own_f1_score.png" in figure_files
    assert "conditioned_evidence_position_f1_score.png" in figure_files
    for name in figure_files:
        assert (analysis_dir / name).stat().st_size > 0


def test_pass_suppressed_under_blinding(tmp_path: Path) -> None:
    run_dir, index, analysis_dir = _pass_fixture(tmp_path)
    document = rca.run_conditioned_curves_pass(
        run_dir, index, analysis_dir, "DESIGN-INPUT-ONLY", blinding_active=True
    )
    assert document is None
    assert not (analysis_dir / rca.CONDITIONED_CURVES_NAME).exists()


def test_pass_half_set_pressure_coordinates_fail_loud(tmp_path: Path) -> None:
    run_dir, index, analysis_dir = _pass_fixture(tmp_path)
    index.loc[0, "rate_frac"] = math.nan  # budget_r stays set -> malformed
    with pytest.raises(cc.ConditionedCurveError, match="half-set"):
        rca.run_conditioned_curves_pass(
            run_dir, index, analysis_dir, "DESIGN-INPUT-ONLY",
            blinding_active=False,
        )


def test_pass_refuses_duplicate_rows_within_one_artifact(
    tmp_path: Path,
) -> None:
    # Two rows in ONE file sharing (example_id, record_index) — here both
    # lacking record_index — are the H3/#127 replay hazard: the evidence
    # join must REFUSE like load_per_query, never last-row-wins-merge them
    # into a phantom single trial.
    run_dir, index, analysis_dir = _pass_fixture(tmp_path)
    requests_path = (
        run_dir / "cells/cellF2/window_squad_v2-01" / "requests.jsonl"
    )
    with requests_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"example_id": "e001", "f1_score": 0.1,
                 "gold_position_in_prompt": 0}
            )
            + "\n"
        )
    with pytest.raises(rca.AnalysisError) as excinfo:
        rca.run_conditioned_curves_pass(
            run_dir, index, analysis_dir, "DESIGN-INPUT-ONLY",
            blinding_active=False,
        )
    message = str(excinfo.value)
    assert "duplicate (example_id, record_index)" in message
    assert "requests.jsonl" in message
    assert "distinct record_index" in message


def test_pass_joins_replayed_rows_with_distinct_record_index(
    tmp_path: Path,
) -> None:
    # Replay rows that DO carry the disambiguating record_index are
    # legitimate distinct trials (H3 by design) — each joins the evidence
    # curve on its own; the cross-file requests⋈qa_evidence merge on the
    # shared key stays intact (the baseline 4 joined trials).
    run_dir, index, analysis_dir = _pass_fixture(tmp_path)
    requests_path = (
        run_dir / "cells/cellF2/window_squad_v2-01" / "requests.jsonl"
    )
    with requests_path.open("a", encoding="utf-8") as handle:
        for record_index in (0, 1):
            handle.write(
                json.dumps(
                    {"example_id": "e003", "record_index": record_index,
                     "f1_score": 0.3, "gold_position_in_prompt": 0}
                )
                + "\n"
            )
    document = rca.run_conditioned_curves_pass(
        run_dir, index, analysis_dir, "DESIGN-INPUT-ONLY",
        blinding_active=False, metrics=("f1_score",),
    )
    assert document is not None
    evidence = document["quality_vs_evidence_position"]
    # 2 windows × (e001 first + e002 absent) + the 2 replayed e003 trials.
    assert evidence["n_trials_joined"] == 6
