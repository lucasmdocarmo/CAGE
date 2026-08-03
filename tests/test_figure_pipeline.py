"""Tests for scripts/4_analysis/figure_pipeline.py (tuple-keyed figure pipeline).

Smoke coverage: every figure renders to a PNG in a tmpdir from synthetic data —
and, where the local pilot archive exists, from real pilot per-query rows re-keyed
through ``from_legacy``. Failure cases assert the fail-loud contract (missing
columns, unknown reference, malformed row keys, scale violations).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = REPO_ROOT / "scripts" / "4_analysis"
for _p in (str(_SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import figure_pipeline as fp  # noqa: E402
from src.analysis.cellspec import CellSpec, from_legacy  # noqa: E402
from src.analysis.goodput import (  # noqa: E402
    IN_REGIME,
    PAST_CLIFF,
    UNPRESSURED,
    label_regime,
)

PILOT_RUN = REPO_ROOT / "results" / "phase2" / "2026-07-16_0143_qwen3-8b_100x3"

K_B1 = CellSpec.from_baseline("B1").to_row_key()  # gold-fresh (the control)
K_B2 = CellSpec.from_baseline("B2").to_row_key()  # gold-reuse
K_B3 = CellSpec.from_baseline("B3").to_row_key()  # corpus-reuse (CAG)
K_B6 = CellSpec.from_baseline("B6").to_row_key()  # retr-fresh · rerank (ranked RAG)


def _assert_rendered(path: Path, outpath: Path) -> None:
    assert path == outpath
    assert path.exists()
    assert path.stat().st_size > 0


@pytest.fixture()
def per_query_df() -> pd.DataFrame:
    """Synthetic per-query long table: 4 cells × 2 datasets × 40 shared examples."""
    rng = np.random.default_rng(7)
    rows = []
    offsets = {K_B1: 0.0, K_B2: -40.0, K_B3: -60.0, K_B6: 35.0}
    for ds in ("squad_v2", "musique"):
        for key, off in offsets.items():
            base = rng.normal(200.0 + off, 20.0, size=40)
            for i, v in enumerate(base):
                rows.append(
                    {
                        "row_key": key,
                        "example_id": f"{ds}-e{i:03d}",
                        "dataset": ds,
                        "ttft_ms": float(v),
                        "f1_score": float(np.clip(rng.normal(0.6, 0.2), 0, 1)),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture()
def grid_df() -> pd.DataFrame:
    """Synthetic F2 goodput grid: 2 engines × 5 budgets × 6 rates, flags set."""
    budgets = [1.5, 1.0, 0.75, 0.5, 0.25]
    rates = [0.5, 0.7, 0.85, 0.95, 1.05, 1.2]
    rows = []
    for engine in ("vllm", "sglang"):
        for r in budgets:
            for lam in rates:
                spec = CellSpec(
                    arm="gold-fresh", retriever="none", policy="none",
                    topology="single", engine=engine, model="qwen3-14b",
                    family="F2", budget_r=r, rate_frac=lam,
                )
                g = max(0.05, min(0.98, r * 0.5 + 0.6 - abs(lam - 0.9)))
                y = g * 0.85
                regime = (
                    UNPRESSURED if r == 1.5
                    else PAST_CLIFF if lam == 1.2 and r <= 0.5
                    else IN_REGIME
                )
                rows.append(
                    {
                        "row_key": spec.to_row_key(),
                        "goodput_frac": g,
                        "yield_frac": y,
                        "regime": regime,
                        "knee": lam == 0.95 and r < 1.5,
                        "cliff": lam == 1.2 and r <= 0.5,
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Row-key plumbing
# ---------------------------------------------------------------------------


def test_expand_row_keys_roundtrip(grid_df: pd.DataFrame) -> None:
    out = fp.expand_row_keys(grid_df)
    assert set(("arm", "engine", "model", "budget_r", "rate_frac")) <= set(out.columns)
    assert sorted(out["engine"].unique()) == ["sglang", "vllm"]
    assert len(out) == len(grid_df)
    assert out["budget_r"].notna().all()


def test_expand_row_keys_malformed_key_raises() -> None:
    df = pd.DataFrame({"row_key": ["not-a-key"], "x": [1.0]})
    with pytest.raises(fp.FigureDataError, match="segments"):
        fp.expand_row_keys(df)


def test_parse_row_key_rejects_charter_illegal_cell() -> None:
    # retriever on a non-retrieval arm is illegal under §7.2.
    bad = "gold-fresh|dense|none|single|vllm|qwen3-14b|F1"
    with pytest.raises(fp.FigureDataError, match="not a valid CellSpec"):
        fp.parse_row_key(bad)


def test_parse_row_key_rejects_unknown_coord_segment() -> None:
    with pytest.raises(fp.FigureDataError, match="coord segment"):
        fp.parse_row_key(K_B1 + "|q0.5")


def test_condense_row_key_labels_drops_shared_segments() -> None:
    labels = fp.condense_row_key_labels([K_B1, K_B3, K_B6])
    assert labels[K_B1] == "gold-fresh · none"
    assert labels[K_B6] == "retr-fresh · rerank"
    # Single-key input keeps its arm.
    assert fp.condense_row_key_labels([K_B1])[K_B1] == "gold-fresh"


# ---------------------------------------------------------------------------
# Paired-summary helpers
# ---------------------------------------------------------------------------


def test_win_loss_tie_counts_directions() -> None:
    deltas = np.array([1.0, -2.0, 0.0, 0.5])
    assert fp.win_loss_tie_counts(deltas, higher_is_better=True) == (2, 1, 1)
    assert fp.win_loss_tie_counts(deltas, higher_is_better=False) == (1, 2, 1)
    assert fp.win_loss_tie_counts(deltas, higher_is_better=True, tie_atol=0.6) == (
        1, 1, 2,
    )


def test_holm_adjust_known_values() -> None:
    adj = fp.holm_adjust(np.array([0.01, 0.04, 0.03]))
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_holm_adjust_nan_passthrough() -> None:
    adj = fp.holm_adjust(np.array([0.01, np.nan]))
    assert np.isclose(adj[0], 0.01)  # m=1: NaN entries are untested, not counted
    assert np.isnan(adj[1])


def test_sign_flip_share() -> None:
    assert fp.sign_flip_share(np.array([1.0, 2.0, 3.0, -1.0])) == pytest.approx(0.25)
    assert fp.sign_flip_share(np.array([-1.0, -2.0])) == 0.0


def test_paired_deltas_averages_trials_and_pairs() -> None:
    df = pd.DataFrame(
        {
            "row_key": [K_B1, K_B1, K_B2, K_B2, K_B2],
            "example_id": ["e0", "e1", "e0", "e0", "e1"],
            "m": [10.0, 20.0, 6.0, 8.0, 15.0],
        }
    )
    deltas = fp.paired_deltas(df, cell=K_B2, reference=K_B1, metric="m")
    # e0: mean(6,8) - 10 = -3; e1: 15 - 20 = -5.
    assert sorted(deltas.tolist()) == [-5.0, -3.0]


def test_paired_deltas_missing_cell_raises(per_query_df: pd.DataFrame) -> None:
    with pytest.raises(fp.FigureDataError, match="no rows"):
        fp.paired_deltas(
            per_query_df, cell="missing|key", reference=K_B1, metric="ttft_ms"
        )


def test_forest_summary_reference_is_configurable(per_query_df: pd.DataFrame) -> None:
    cfg = fp.ForestConfig(reference=K_B3, metric="ttft_ms", higher_is_better=False,
                          n_boot=200)
    summary = fp.forest_summary(per_query_df, config=cfg)
    assert K_B3 not in set(summary["row_key"])
    assert K_B1 in set(summary["row_key"])  # the old hardcoded reference is now a row
    assert set(summary["dataset"]) == {"squad_v2", "musique"}
    assert {"median_delta", "ci_low", "ci_high", "p_holm", "wins", "losses",
            "ties"} <= set(summary.columns)
    # W/L/T totals must equal the paired n per row (§8.13: no silent drops).
    wlt = summary[["wins", "losses", "ties"]].sum(axis=1)
    assert (wlt == summary["n_pairs"]).all()


def test_forest_summary_all_tie_pair_gets_nan_p() -> None:
    # Token-identical pair (B1 vs B2 at T=0): identical metric values → NaN p.
    df = pd.DataFrame(
        {
            "row_key": [K_B1] * 3 + [K_B2] * 3,
            "example_id": ["e0", "e1", "e2"] * 2,
            "m": [1.0, 2.0, 3.0] * 2,
        }
    )
    cfg = fp.ForestConfig(reference=K_B1, metric="m", higher_is_better=True, n_boot=50)
    summary = fp.forest_summary(df, config=cfg)
    assert np.isnan(summary.loc[0, "p"])
    assert summary.loc[0, "ties"] == 3


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_forest_config_validation() -> None:
    with pytest.raises(fp.FigureConfigError):
        fp.ForestConfig(reference="", metric="m", higher_is_better=True)
    with pytest.raises(fp.FigureConfigError):
        fp.ForestConfig(reference=K_B1, metric="m", higher_is_better=True, n_boot=0)
    with pytest.raises(fp.FigureConfigError):
        fp.ForestConfig(reference=K_B1, metric="m", higher_is_better=True,
                        cells=(K_B1,))


def test_wlt_config_validation() -> None:
    with pytest.raises(fp.FigureConfigError):
        fp.WinLossTieConfig(contrasts=(), metric="m", higher_is_better=True)
    with pytest.raises(fp.FigureConfigError):
        fp.WinLossTieConfig(contrasts=((K_B1, K_B1),), metric="m",
                            higher_is_better=True)


def test_spec_curve_config_validation() -> None:
    with pytest.raises(fp.FigureConfigError):
        fp.SpecCurveConfig(spec_columns=())
    with pytest.raises(fp.FigureConfigError):
        fp.SpecCurveConfig(spec_columns=("a",), sign_flip_threshold=1.5)


# ---------------------------------------------------------------------------
# Figure smoke tests (render to PNG in tmpdir)
# ---------------------------------------------------------------------------


def test_plot_forest_renders(per_query_df: pd.DataFrame, tmp_path: Path) -> None:
    out = tmp_path / "forest.png"
    cfg = fp.ForestConfig(reference=K_B1, metric="ttft_ms", higher_is_better=False,
                          n_boot=200, title="TTFT vs control")
    _assert_rendered(fp.plot_forest(per_query_df, out, config=cfg), out)


def test_plot_forest_alternate_reference_renders(
    per_query_df: pd.DataFrame, tmp_path: Path
) -> None:
    out = tmp_path / "forest_b3ref.png"
    cfg = fp.ForestConfig(reference=K_B3, metric="ttft_ms", higher_is_better=False,
                          n_boot=200)
    _assert_rendered(fp.plot_forest(per_query_df, out, config=cfg), out)


def test_plot_forest_unknown_reference_raises(
    per_query_df: pd.DataFrame, tmp_path: Path
) -> None:
    cfg = fp.ForestConfig(reference="nope|key", metric="ttft_ms",
                          higher_is_better=False)
    with pytest.raises(fp.FigureDataError, match="reference"):
        fp.plot_forest(per_query_df, tmp_path / "x.png", config=cfg)


def test_plot_forest_empty_df_raises(tmp_path: Path) -> None:
    df = pd.DataFrame(columns=["row_key", "example_id", "ttft_ms"])
    cfg = fp.ForestConfig(reference=K_B1, metric="ttft_ms", higher_is_better=False)
    with pytest.raises(fp.FigureDataError, match="empty"):
        fp.plot_forest(df, tmp_path / "x.png", config=cfg)


def test_plot_win_loss_tie_renders(per_query_df: pd.DataFrame, tmp_path: Path) -> None:
    out = tmp_path / "wlt.png"
    cfg = fp.WinLossTieConfig(
        contrasts=((K_B6, K_B3), (K_B2, K_B1), (K_B3, K_B1)),
        metric="ttft_ms", higher_is_better=False,
    )
    _assert_rendered(fp.plot_win_loss_tie(per_query_df, out, config=cfg), out)


def test_plot_spec_curve_renders_and_triages(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    n = 48
    df = pd.DataFrame(
        {
            "estimate": rng.normal(0.4, 0.6, size=n),  # some sign flips
            "ci_low": np.nan,
            "ci_high": np.nan,
            "slo_family": rng.choice(["primary", "sarathi"], size=n),
            "instrument": rng.choice(["A", "B"], size=n),
            "y_basis": rng.choice(["raw", "corrected"], size=n),
        }
    )
    df["ci_low"] = df["estimate"] - 0.2
    df["ci_high"] = df["estimate"] + 0.2
    out = tmp_path / "spec_curve.png"
    cfg = fp.SpecCurveConfig(
        spec_columns=("slo_family", "instrument", "y_basis"),
        ci_cols=("ci_low", "ci_high"),
        sign_flip_threshold=0.25,
        effect_label="Δ TTFT (ms)",
    )
    _assert_rendered(fp.plot_spec_curve(df, out, config=cfg), out)


def test_plot_spec_curve_missing_spec_column_raises(tmp_path: Path) -> None:
    df = pd.DataFrame({"estimate": [0.1, 0.2]})
    cfg = fp.SpecCurveConfig(spec_columns=("instrument",))
    with pytest.raises(fp.FigureDataError, match="missing required columns"):
        fp.plot_spec_curve(df, tmp_path / "x.png", config=cfg)


def test_plot_truth_tax_renders(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    rows = []
    for ds in ("squad_v2", "hotpotqa"):
        for policy in ("none", "evict", "compress-fp8"):
            for engine in ("vllm", "sglang"):
                spec = CellSpec(
                    arm="gold-fresh", retriever="none", policy=policy,
                    topology="single", engine=engine, model="qwen3-14b",
                    family="F2", budget_r=0.5, rate_frac=0.85,
                )
                g = float(rng.uniform(0.5, 0.95))
                rows.append(
                    {
                        "row_key": spec.to_row_key(),
                        "dataset": ds,
                        "goodput_frac": g,
                        "yield_frac": g * float(rng.uniform(0.6, 1.0)),
                    }
                )
    out = tmp_path / "truth_tax.png"
    _assert_rendered(
        fp.plot_truth_tax(pd.DataFrame(rows), out, config=fp.TruthTaxConfig()), out
    )


def test_plot_truth_tax_yield_above_goodput_raises(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "row_key": [K_B1],
            "dataset": ["squad_v2"],
            "goodput_frac": [0.5],
            "yield_frac": [0.6],  # Y > G is impossible by construction
        }
    )
    with pytest.raises(fp.FigureDataError, match="subset"):
        fp.plot_truth_tax(df, tmp_path / "x.png", config=fp.TruthTaxConfig())


def test_plot_truth_tax_rejects_rate_scale(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "row_key": [K_B1],
            "dataset": ["squad_v2"],
            "goodput_frac": [12.5],  # a per-GPU rate, not a fraction
            "yield_frac": [9.0],
        }
    )
    with pytest.raises(fp.FigureDataError, match="fraction-of-issued"):
        fp.plot_truth_tax(df, tmp_path / "x.png", config=fp.TruthTaxConfig())


def test_plot_goodput_grid_renders(grid_df: pd.DataFrame, tmp_path: Path) -> None:
    out = tmp_path / "goodput_grid.png"
    _assert_rendered(
        fp.plot_goodput_grid(grid_df, out, config=fp.GoodputGridConfig()), out
    )


def test_plot_goodput_grid_bad_regime_raises(
    grid_df: pd.DataFrame, tmp_path: Path
) -> None:
    df = grid_df.copy()
    df.loc[0, "regime"] = "saturated"  # not a §6.1 label
    with pytest.raises(fp.FigureDataError, match="regime"):
        fp.plot_goodput_grid(df, tmp_path / "x.png", config=fp.GoodputGridConfig())


def test_plot_goodput_grid_nan_knee_raises(
    grid_df: pd.DataFrame, tmp_path: Path
) -> None:
    df = grid_df.copy()
    df["knee"] = df["knee"].astype(object)
    df.loc[0, "knee"] = np.nan
    with pytest.raises(fp.FigureDataError, match="boolean"):
        fp.plot_goodput_grid(df, tmp_path / "x.png", config=fp.GoodputGridConfig())


def test_plot_goodput_grid_requires_pressure_coords(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "row_key": [K_B1],  # F1 sub-pressure cell: no r/lam coords
            "goodput_frac": [0.9],
            "yield_frac": [0.8],
            "regime": [IN_REGIME],
            "knee": [False],
            "cliff": [False],
        }
    )
    with pytest.raises(fp.FigureDataError, match="pressure coords"):
        fp.plot_goodput_grid(df, tmp_path / "x.png", config=fp.GoodputGridConfig())


def test_regime_vocabulary_is_the_goodput_vocabulary() -> None:
    # 2026-08-02 regression: the renderer must accept exactly the §6.1 labels
    # the producer (goodput.label_regime) emits — a second hand-spelled set
    # here is the drift bug.
    assert fp._REGIME_LABELS == {IN_REGIME, UNPRESSURED, PAST_CLIFF}


def test_plot_goodput_grid_accepts_label_regime_output(
    grid_df: pd.DataFrame, tmp_path: Path
) -> None:
    # End-to-end: classify with the charter-mandated §6.1 classifier, render
    # with the F1 skeleton — the exact pipe that raised FigureDataError before
    # the 2026-08-02 harmonization.
    df = grid_df.copy()
    coords = fp.expand_row_keys(df)
    cells = pd.DataFrame(
        {
            # r=1.5 rows unpressured (no scarcity events); high-rate starved
            # rows past the cliff (attainment collapse); the rest in-regime.
            "rho_kv": np.where(coords["budget_r"] == 1.5, 0.3, 0.97),
            "scarcity_events": np.where(coords["budget_r"] == 1.5, 0, 25),
            "attainment": np.where(
                (coords["rate_frac"] == 1.2) & (coords["budget_r"] <= 0.5),
                0.5,
                0.99,
            ),
        },
        index=df.index,
    )
    df["regime"] = label_regime(cells)
    assert set(df["regime"].unique()) == {IN_REGIME, UNPRESSURED, PAST_CLIFF}
    out = tmp_path / "goodput_grid_labeled.png"
    _assert_rendered(
        fp.plot_goodput_grid(df, out, config=fp.GoodputGridConfig()), out
    )


# ---------------------------------------------------------------------------
# Pilot-archive smoke (design-input use only; skipped when the archive is absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not PILOT_RUN.exists(), reason="pilot 100x3 archive not present")
def test_plot_forest_from_pilot_archive(tmp_path: Path) -> None:
    frames = []
    for legacy_name in ("no_cache", "prefix_cache", "rag"):
        csvs = sorted(
            (PILOT_RUN / "baselines" / legacy_name).glob(
                "trial_*/*_squad_v2_*_results.csv"
            )
        )
        if not csvs:
            pytest.skip(f"no squad_v2 per-query CSV for {legacy_name}")
        raw = pd.read_csv(csvs[0])
        key = from_legacy(legacy_name).to_row_key()
        frames.append(
            pd.DataFrame(
                {
                    "row_key": key,
                    "example_id": raw["example_id"],
                    "dataset": "squad_v2",
                    "ttft_ms": raw["ttft_ms"].astype(float),
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    out = tmp_path / "pilot_forest.png"
    cfg = fp.ForestConfig(
        reference=from_legacy("no_cache").to_row_key(),
        metric="ttft_ms", higher_is_better=False, n_boot=300,
        title="Pilot 100x3 re-keyed (design input only)",
    )
    _assert_rendered(fp.plot_forest(df, out, config=cfg), out)


# ---------------------------------------------------------------------------
# (f) CONSORT flow (§9.10)
# ---------------------------------------------------------------------------


def _consort_stages() -> list[fp.ConsortStage]:
    return [
        fp.ConsortStage("cells planned", 940, excluded=40,
                        exclusion_reason="pruned per §6.8"),
        fp.ConsortStage("cells run", 900, excluded=60,
                        exclusion_reason="UNPRESSURED / PAST-CLIFF (§6.1)"),
        fp.ConsortStage("in-regime", 840, excluded=12,
                        exclusion_reason="engine-defect protocol (§9.12)"),
        fp.ConsortStage("analyzed", 828),
    ]


class TestConsortFlow:
    def test_renders(self, tmp_path):
        out = tmp_path / "consort.png"
        _assert_rendered(fp.plot_consort_flow(_consort_stages(), out), out)

    def test_arithmetic_must_close(self, tmp_path):
        stages = _consort_stages()
        stages[1] = fp.ConsortStage("cells run", 899, excluded=60,
                                    exclusion_reason="x")
        with pytest.raises(ValueError, match="does not close"):
            fp.plot_consort_flow(stages, tmp_path / "x.png")

    def test_negative_count_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="negative"):
            fp.plot_consort_flow(
                [fp.ConsortStage("a", -1), fp.ConsortStage("b", 0)],
                tmp_path / "x.png")

    def test_exclusion_needs_reason(self, tmp_path):
        with pytest.raises(ValueError, match="no reason"):
            fp.plot_consort_flow(
                [fp.ConsortStage("a", 10, excluded=2), fp.ConsortStage("b", 8)],
                tmp_path / "x.png")

    def test_too_few_stages(self, tmp_path):
        with pytest.raises(ValueError, match="two stages"):
            fp.plot_consort_flow([fp.ConsortStage("a", 10)], tmp_path / "x.png")
