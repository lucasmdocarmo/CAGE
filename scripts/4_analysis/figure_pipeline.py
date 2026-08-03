#!/usr/bin/env python3
"""Tuple-keyed campaign figure pipeline (audit P0-4; PUBLICATION.md D9/D10 figure specs).

Every figure in this module is keyed on ``CellSpec.to_row_key()`` strings — the D7
tuple identity (``arm|retriever|policy|topology|engine|model|family[|r{g}|lam{g}]``),
never on pilot-era flat baseline names. Pilot archives are re-keyed through
``src.analysis.cellspec.from_legacy`` before they reach this module.

Implemented ($0-computable today, per PROJECT_AUDIT_2026-08-02.md §4 gap #4 / §6):

- ``plot_forest``           — F9-style paired-delta forest with a CONFIGURABLE
  reference cell (the pilot machinery hardcoded ``no_cache``; here the reference is
  a required config field), median delta + bootstrap CI + Holm markers + mandatory
  win/loss/tie counts per row (§8.13), per-dataset panels (pooling prohibited, §9.1).
- ``plot_win_loss_tie``     — 100%-stacked W/L/T bars per registered contrast.
- ``plot_spec_curve``       — specification-curve renderer (F11): ranked estimates
  over a spec-matrix panel, descriptive-with-triage annotations (§9.13 / audit §2.7).
- ``plot_truth_tax``        — F2: Y-vs-G scatter with the y=x line, per-dataset
  facets; distance below the diagonal is the truth tax (§9.2 estimand G − Y).
- ``plot_goodput_grid``     — F1 skeleton: budget-ramped rate curves, G solid /
  Y dashed same hue with the gap shaded, knee/cliff open glyphs, out-of-regime
  hollow markers.

Y-SCALE PIN (decision recorded here; audit F1 flagged the ambiguity): every figure
in this module renders G and Y (serving yield) on the FRACTION-OF-ISSUED scale —
``goodput_frac``  = SLO-passing completed requests / issued requests,
``yield_frac``    = requests passing (primary SLO ∧ §8.5 predicate) / issued
(S1's "82%/64%" scale). The per-GPU *rate* rendering of D6 §6.1 goodput is the other
named scale and is NEVER mixed into these figures (audit F1: "two named scales,
never mixed in one figure"); a rate-scaled panel would be a separate figure function.
On this common scale the shaded G−Y gap IS the §9.2 truth-tax estimand variable.

Statistics printed on figures (bootstrap CI, Wilcoxon p, Holm within a panel) are
FIGURE-GRADE descriptive companions computed here for self-contained rendering; the
confirmatory numbers come from the cage-stats contrast layer (D9 family map) and, once
available, are passed through unchanged. Binding style conventions inherited from
``_plot_style``: no dual axes, no lines across nominal categories, no plt.show().

Every figure function has the signature ``(df, outpath, *, config) -> Path``.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for _p in (str(_HERE), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import _plot_style as ps
from src.analysis.cellspec import CellSpec, CellSpecError
from src.analysis.goodput import IN_REGIME, PAST_CLIFF, UNPRESSURED

KEY_COL = "row_key"
_AXES = ("arm", "retriever", "policy", "topology", "engine", "model", "family")
# §6.1 regime vocabulary — imported from src.analysis.goodput (the producer,
# P0-3) so ``label_regime`` output feeds this renderer verbatim; a second
# hand-spelled label set here is exactly the 2026-08-02 drift bug.
_REGIME_LABELS = frozenset({IN_REGIME, UNPRESSURED, PAST_CLIFF})

# W/L/T bar colors (colorblind palette hexes, consistent with _plot_style families).
_WLT_COLORS = {"win": "#029e73", "tie": "#949494", "loss": "#d55e00"}


class FigureDataError(ValueError):
    """The input DataFrame violates a figure's declared column/value contract."""


class FigureConfigError(ValueError):
    """A figure config carries an illegal value."""


# ---------------------------------------------------------------------------
# Row-key plumbing
# ---------------------------------------------------------------------------


def parse_row_key(key: str) -> dict[str, str | float | None]:
    """Split one ``CellSpec.to_row_key()`` string back into axis/coord fields.

    Validates by round-tripping through ``CellSpec`` — malformed or
    charter-illegal keys fail loud with the underlying cellspec error chained.
    """
    parts = str(key).split("|")
    if len(parts) < 7:
        raise FigureDataError(
            f"row key {key!r} has {len(parts)} segments; expected the 7 axes "
            "(arm|retriever|policy|topology|engine|model|family) + optional coords"
        )
    budget_r: float | None = None
    rate_frac: float | None = None
    for coord in parts[7:]:
        if coord.startswith("lam"):
            rate_frac = float(coord[3:])
        elif coord.startswith("r"):
            budget_r = float(coord[1:])
        else:
            raise FigureDataError(
                f"row key {key!r}: unrecognized coord segment {coord!r} "
                "(expected 'r<float>' or 'lam<float>')"
            )
    try:
        spec = CellSpec(*parts[:7], budget_r=budget_r, rate_frac=rate_frac)  # type: ignore[arg-type]
    except (CellSpecError, ValueError) as exc:
        raise FigureDataError(f"row key {key!r} is not a valid CellSpec: {exc}") from exc
    if spec.to_row_key() != key:
        raise FigureDataError(
            f"row key {key!r} does not round-trip (canonical: {spec.to_row_key()!r})"
        )
    return spec.to_flat_dict()


def expand_row_keys(df: pd.DataFrame, *, key_col: str = KEY_COL) -> pd.DataFrame:
    """Return a copy of ``df`` with the 7 axis columns + coords derived from row keys.

    Parses each UNIQUE key once (bounded by cell count, not row count) and merges
    the fields back — no per-row Python loop.
    """
    _require_columns(df, [key_col], "expand_row_keys")
    uniq = df[key_col].dropna().unique()
    if len(uniq) == 0:
        raise FigureDataError(f"expand_row_keys: no non-null values in {key_col!r}")
    fields = pd.DataFrame([{key_col: k, **parse_row_key(k)} for k in uniq])
    clash = [c for c in fields.columns if c != key_col and c in df.columns]
    if clash:
        raise FigureDataError(
            f"expand_row_keys: df already has derived columns {clash}; drop or rename them"
        )
    return df.merge(fields, on=key_col, how="left", validate="many_to_one")


def condense_row_key_labels(keys: Sequence[str]) -> dict[str, str]:
    """Human-readable per-figure labels: drop axis segments shared by ALL keys.

    E.g. pilot F1 cells differing only in arm/retriever label as ``gold-reuse`` or
    ``retr-fresh · rerank`` instead of the full 7-segment key. A single key keeps
    its arm. Coord segments (r/lam) are always kept when present.
    """
    uniq = list(dict.fromkeys(str(k) for k in keys))
    if not uniq:
        return {}
    split = [k.split("|") for k in uniq]
    labels: dict[str, str] = {}
    n_axes = len(_AXES)
    varying = [
        i for i in range(n_axes)
        if len({p[i] for p in split if len(p) > i}) > 1
    ]
    for key, parts in zip(uniq, split):
        kept = [parts[i] for i in varying if i < len(parts)]
        kept += parts[n_axes:]  # coords always shown
        labels[key] = " · ".join(kept) if kept else parts[0]
    return labels


def _require_columns(df: pd.DataFrame, cols: Sequence[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise FigureDataError(f"{where}: input df is missing required columns {missing}")
    if df.empty:
        raise FigureDataError(f"{where}: input df is empty")


# ---------------------------------------------------------------------------
# Paired-contrast summaries (figure-grade; confirmatory stats live in cage-stats)
# ---------------------------------------------------------------------------


def paired_deltas(
    df: pd.DataFrame,
    *,
    cell: str,
    reference: str,
    metric: str,
    key_col: str = KEY_COL,
    id_col: str = "example_id",
) -> np.ndarray:
    """Per-example paired deltas (cell − reference), inner-joined on ``id_col``.

    Rows duplicated per example within a cell (multiple trials) are averaged first,
    so each example contributes one pair — the pilot stats kit's unit of analysis.
    Valid SUB-PRESSURE only (per-query pairing under load is prohibited, §9.4).
    """
    _require_columns(df, [key_col, id_col, metric], "paired_deltas")
    sub = df[df[key_col].isin([cell, reference])]
    for key in (cell, reference):
        if not (sub[key_col] == key).any():
            raise FigureDataError(f"paired_deltas: no rows for row key {key!r}")
    wide = (
        sub.groupby([id_col, key_col], observed=True)[metric]
        .mean()
        .unstack(key_col)
        .dropna(subset=[cell, reference])
    )
    if wide.empty:
        raise FigureDataError(
            f"paired_deltas: no overlapping {id_col!r} between {cell!r} and {reference!r}"
        )
    return (wide[cell] - wide[reference]).to_numpy(dtype=float)


def win_loss_tie_counts(
    deltas: np.ndarray, *, higher_is_better: bool, tie_atol: float = 0.0
) -> tuple[int, int, int]:
    """(wins, losses, ties) for paired deltas; a win improves on the reference."""
    deltas = np.asarray(deltas, dtype=float)
    ties = int(np.sum(np.abs(deltas) <= tie_atol))
    pos = int(np.sum(deltas > tie_atol))
    neg = int(np.sum(deltas < -tie_atol))
    return (pos, neg, ties) if higher_is_better else (neg, pos, ties)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values; NaN entries pass through untested."""
    p = np.asarray(p_values, dtype=float)
    adj = np.full_like(p, np.nan)
    mask = ~np.isnan(p)
    m = int(mask.sum())
    if m == 0:
        return adj
    order = np.argsort(p[mask])
    ranked = p[mask][order] * (m - np.arange(m))
    ranked = np.maximum.accumulate(ranked)  # step-down monotonicity
    out = np.empty(m)
    out[order] = np.clip(ranked, 0.0, 1.0)
    adj[mask] = out
    return adj


def _bootstrap_median_ci(
    deltas: np.ndarray, *, n_boot: int, seed: int, alpha: float
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = deltas.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    med = np.median(deltas[idx], axis=1)
    lo, hi = np.quantile(med, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def _wilcoxon_p(deltas: np.ndarray, *, tie_atol: float) -> float:
    """Two-sided Wilcoxon signed-rank p; NaN when every pair ties.

    All-tie contrasts are EXPECTED (token-identical pairs at T=0, §7.7a); NaN is
    their defined, rendered-as-blank semantics — not a fallback.
    """
    if np.all(np.abs(deltas) <= tie_atol):
        return math.nan
    from scipy import stats as scipy_stats

    return float(scipy_stats.wilcoxon(deltas).pvalue)


def _panel_mask(frame: pd.DataFrame, ds: object) -> pd.Series:
    """Row mask for one dataset panel; ``ds=None`` = the no-dataset-column case."""
    if ds is None:
        return pd.Series(True, index=frame.index)
    return frame["dataset"] == ds


def forest_summary(df: pd.DataFrame, *, config: ForestConfig) -> pd.DataFrame:
    """One row per (dataset, cell) contrast vs ``config.reference``.

    Columns: dataset, row_key, n_pairs, median_delta, ci_low, ci_high, p, p_holm,
    wins, losses, ties. Holm is applied within each dataset panel (family =
    metric × dataset, matching the §9.3 membership rule).
    """
    _require_columns(df, [KEY_COL, "example_id", config.metric], "forest_summary")
    if not (df[KEY_COL] == config.reference).any():
        raise FigureDataError(
            f"forest_summary: reference {config.reference!r} has no rows; "
            f"present keys: {sorted(df[KEY_COL].unique())}"
        )
    if "dataset" in df.columns:
        if df["dataset"].isna().any():
            raise FigureDataError(
                "forest_summary: NaN in the dataset column — rows must be labeled "
                "(pooling across unlabeled data is prohibited, §9.1)"
            )
        datasets: list[object] = list(df["dataset"].unique())
    else:
        datasets = [None]
    cells = (
        list(config.cells)
        if config.cells is not None
        else [k for k in df[KEY_COL].unique() if k != config.reference]
    )
    if not cells:
        raise FigureDataError("forest_summary: no non-reference cells to contrast")
    rows: list[dict[str, object]] = []
    for ds in datasets:
        sub = df[_panel_mask(df, ds)]
        p_list: list[float] = []
        ds_rows: list[dict[str, object]] = []
        for cell in cells:
            deltas = paired_deltas(
                sub, cell=cell, reference=config.reference, metric=config.metric
            )
            lo, hi = _bootstrap_median_ci(
                deltas, n_boot=config.n_boot, seed=config.seed, alpha=config.alpha
            )
            w, l, t = win_loss_tie_counts(
                deltas,
                higher_is_better=config.higher_is_better,
                tie_atol=config.tie_atol,
            )
            p = _wilcoxon_p(deltas, tie_atol=config.tie_atol)
            p_list.append(p)
            ds_rows.append(
                {
                    "dataset": ds,
                    KEY_COL: cell,
                    "n_pairs": deltas.shape[0],
                    "median_delta": float(np.median(deltas)),
                    "ci_low": lo,
                    "ci_high": hi,
                    "p": p,
                    "wins": w,
                    "losses": l,
                    "ties": t,
                }
            )
        adj = holm_adjust(np.array(p_list))
        for row, ph in zip(ds_rows, adj):
            row["p_holm"] = float(ph) if not math.isnan(ph) else math.nan
        rows.extend(ds_rows)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForestConfig:
    """Forest-plot config. ``reference`` is REQUIRED — never hardcoded (audit P0-4)."""

    reference: str
    metric: str
    higher_is_better: bool
    n_boot: int = 2000
    seed: int = 20260802
    alpha: float = 0.05
    tie_atol: float = 0.0
    cells: tuple[str, ...] | None = None
    title: str | None = None

    def __post_init__(self) -> None:
        if not self.reference:
            raise FigureConfigError("ForestConfig.reference must be a non-empty row key")
        if not self.metric:
            raise FigureConfigError("ForestConfig.metric must be a column name")
        if self.n_boot <= 0:
            raise FigureConfigError(f"n_boot must be > 0, got {self.n_boot}")
        if not (0.0 < self.alpha < 1.0):
            raise FigureConfigError(f"alpha must be in (0, 1), got {self.alpha}")
        if self.tie_atol < 0.0:
            raise FigureConfigError(f"tie_atol must be >= 0, got {self.tie_atol}")
        if self.cells is not None and self.reference in self.cells:
            raise FigureConfigError("ForestConfig.cells must not contain the reference")


@dataclass(frozen=True)
class WinLossTieConfig:
    """W/L/T stacked-bar config; one bar per (cell, reference) contrast pair."""

    contrasts: tuple[tuple[str, str], ...]
    metric: str
    higher_is_better: bool
    tie_atol: float = 0.0
    title: str | None = None

    def __post_init__(self) -> None:
        if not self.contrasts:
            raise FigureConfigError("WinLossTieConfig.contrasts must be non-empty")
        for pair in self.contrasts:
            if len(pair) != 2 or pair[0] == pair[1]:
                raise FigureConfigError(
                    f"each contrast must be a (cell, reference) pair of distinct keys; got {pair!r}"
                )
        if not self.metric:
            raise FigureConfigError("WinLossTieConfig.metric must be a column name")
        if self.tie_atol < 0.0:
            raise FigureConfigError(f"tie_atol must be >= 0, got {self.tie_atol}")


@dataclass(frozen=True)
class SpecCurveConfig:
    """Specification-curve config (F11; descriptive-with-triage, §9.13)."""

    spec_columns: tuple[str, ...]
    estimate_col: str = "estimate"
    ci_cols: tuple[str, str] | None = None
    sign_flip_threshold: float = 0.25
    effect_label: str = "effect estimate"
    title: str | None = None

    def __post_init__(self) -> None:
        if not self.spec_columns:
            raise FigureConfigError("SpecCurveConfig.spec_columns must be non-empty")
        if not (0.0 < self.sign_flip_threshold < 1.0):
            raise FigureConfigError(
                f"sign_flip_threshold must be in (0, 1), got {self.sign_flip_threshold}"
            )
        if self.ci_cols is not None and len(self.ci_cols) != 2:
            raise FigureConfigError("ci_cols must be (low, high) column names or None")


@dataclass(frozen=True)
class TruthTaxConfig:
    """Truth-tax scatter config (F2). G and Y on the fraction-of-issued scale."""

    g_col: str = "goodput_frac"
    y_col: str = "yield_frac"
    dataset_col: str = "dataset"
    title: str | None = None

    def __post_init__(self) -> None:
        if self.g_col == self.y_col:
            raise FigureConfigError("g_col and y_col must differ")


@dataclass(frozen=True)
class GoodputGridConfig:
    """Goodput-grid config (F1 skeleton). Facets = engine × model from row keys."""

    g_col: str = "goodput_frac"
    y_col: str = "yield_frac"
    regime_col: str = "regime"
    knee_col: str = "knee"
    cliff_col: str = "cliff"
    title: str | None = None

    def __post_init__(self) -> None:
        if self.g_col == self.y_col:
            raise FigureConfigError("g_col and y_col must differ")


# ---------------------------------------------------------------------------
# (a) Forest plot — configurable reference
# ---------------------------------------------------------------------------


def plot_forest(df: pd.DataFrame, outpath: Path, *, config: ForestConfig) -> Path:
    """F9 forest: per-dataset paired median deltas vs a CONFIGURABLE reference.

    One panel per dataset (pooling across datasets is prohibited, §9.1); rows =
    cells; point = median paired delta; whisker = bootstrap CI; stars = Holm-
    adjusted Wilcoxon within the panel; ``W/L/T`` printed per row (§8.13 mandate).
    """
    summary = forest_summary(df, config=config)
    ps.apply_style()

    datasets = list(summary["dataset"].unique())
    if len(datasets) == 1 and pd.isna(datasets[0]):
        datasets = [None]  # no-dataset-column case round-trips as NaN
    cells = list(summary[KEY_COL].unique())
    # Stable row order: by median delta on the first panel.
    first = summary[_panel_mask(summary, datasets[0])].set_index(KEY_COL)
    cells.sort(key=lambda c: first.loc[c, "median_delta"] if c in first.index else 0.0)
    labels = condense_row_key_labels(cells + [config.reference])

    n_panel = len(datasets)
    fig_w = min(ps.FULL_WIDTH_IN, 2.0 + 2.1 * n_panel)
    fig_h = max(2.2, 0.55 * len(cells) + 1.4)
    fig, axes = plt.subplots(
        1, n_panel, figsize=(fig_w, fig_h), sharey=True, squeeze=False
    )
    ypos = {cell: i for i, cell in enumerate(cells)}
    for ax, ds in zip(axes[0], datasets):
        panel = summary[_panel_mask(summary, ds)].set_index(KEY_COL)
        ys = np.array([ypos[c] for c in panel.index])
        med = panel["median_delta"].to_numpy(dtype=float)
        err = np.vstack(
            [med - panel["ci_low"].to_numpy(float), panel["ci_high"].to_numpy(float) - med]
        )
        ax.errorbar(
            med, ys, xerr=np.abs(err), fmt="o", ms=5, color="#0173b2",
            ecolor="#0173b2", elinewidth=1.4, capsize=3, zorder=3,
        )
        ax.axvline(0.0, color="0.35", lw=0.9, ls="--", zorder=1)
        span = max(
            abs(float(panel["ci_low"].min())), abs(float(panel["ci_high"].max())), 1e-12
        )
        for cell, row in panel.iterrows():
            star = ps.stars(row["p_holm"])
            if star:
                ax.annotate(
                    star, (row["median_delta"], ypos[cell]),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=8, color="0.15",
                )
            ax.annotate(
                f"{int(row['wins'])}/{int(row['losses'])}/{int(row['ties'])}",
                xy=(1.0, ypos[cell]), xycoords=("axes fraction", "data"),
                xytext=(-2, 0), textcoords="offset points",
                ha="right", va="center", fontsize=6.5, color="0.35",
            )
        ax.set_xlim(-1.25 * span, 1.25 * span)
        ax.set_title(str(ds) if ds is not None else "all data")
        ax.set_xlabel(
            f"Δ {config.metric} vs {labels.get(config.reference, config.reference)}\n"
            f"({'higher' if config.higher_is_better else 'lower'} is better)"
        )
    axes[0][0].set_yticks(list(ypos.values()))
    axes[0][0].set_yticklabels([labels[c] for c in cells])
    if config.title:
        fig.suptitle(config.title, y=1.02)
    fig.text(
        0.995, 0.005, "W/L/T per row; stars = Holm within panel",
        ha="right", va="bottom", fontsize=6, color="0.45",
    )
    fig.tight_layout()
    outpath = Path(outpath)
    ps.save_fig(fig, outpath)
    return outpath


# ---------------------------------------------------------------------------
# (b) Win/loss/tie stacked bars
# ---------------------------------------------------------------------------


def plot_win_loss_tie(
    df: pd.DataFrame, outpath: Path, *, config: WinLossTieConfig
) -> Path:
    """100%-stacked win/loss/tie bar per registered contrast (§8.13 mandate).

    Bars are horizontal, one per (cell, reference) pair; raw counts printed inside
    each segment. Stacked composition only — no lines across nominal categories.
    """
    _require_columns(df, [KEY_COL, "example_id", config.metric], "plot_win_loss_tie")
    ps.apply_style()

    rows = []
    for cell, reference in config.contrasts:
        deltas = paired_deltas(df, cell=cell, reference=reference, metric=config.metric)
        w, l, t = win_loss_tie_counts(
            deltas, higher_is_better=config.higher_is_better, tie_atol=config.tie_atol
        )
        rows.append({"cell": cell, "reference": reference, "win": w, "loss": l, "tie": t})
    counts = pd.DataFrame(rows)
    totals = counts[["win", "loss", "tie"]].sum(axis=1).to_numpy(dtype=float)

    all_keys = [k for pair in config.contrasts for k in pair]
    labels = condense_row_key_labels(all_keys)
    bar_labels = [
        f"{labels[c]} vs {labels[r]}" for c, r in zip(counts["cell"], counts["reference"])
    ]

    fig_h = max(1.8, 0.5 * len(counts) + 1.2)
    fig, ax = plt.subplots(figsize=(ps.FULL_WIDTH_IN, fig_h))
    ys = np.arange(len(counts))
    left = np.zeros(len(counts))
    for part in ("win", "tie", "loss"):
        frac = counts[part].to_numpy(dtype=float) / totals
        ax.barh(ys, frac, left=left, color=_WLT_COLORS[part], label=part.title(),
                edgecolor="white", lw=0.5)
        for y, x0, f, n in zip(ys, left, frac, counts[part]):
            if n > 0 and f > 0.04:
                ax.annotate(str(int(n)), (x0 + f / 2.0, y), ha="center", va="center",
                            fontsize=7, color="white")
        left += frac
    ax.set_yticks(ys)
    ax.set_yticklabels(bar_labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"share of paired queries — {config.metric} "
                  f"({'higher' if config.higher_is_better else 'lower'} is better)")
    ax.invert_yaxis()
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    if config.title:
        ax.set_title(config.title)
    fig.tight_layout()
    outpath = Path(outpath)
    ps.save_fig(fig, outpath)
    return outpath


# ---------------------------------------------------------------------------
# (c) Specification curve
# ---------------------------------------------------------------------------


def sign_flip_share(estimates: np.ndarray) -> float:
    """Share of specifications whose estimate sign opposes the median sign.

    Zero-median curves (no dominant direction) count every nonzero estimate as a
    flip — the maximally cautious reading for the triage rule.
    """
    est = np.asarray(estimates, dtype=float)
    if est.size == 0:
        raise FigureDataError("sign_flip_share: no estimates")
    ref = np.sign(np.median(est))
    if ref == 0:
        return float(np.mean(est != 0))
    return float(np.mean(np.sign(est) == -ref))


def plot_spec_curve(df: pd.DataFrame, outpath: Path, *, config: SpecCurveConfig) -> Path:
    """F11 specification curve: ranked estimates over a spec-matrix dot panel.

    Descriptive-with-triage (§9.13 / audit §2.7): the sign-flip share across all
    defensible specifications is computed and annotated; when it exceeds
    ``config.sign_flip_threshold`` the figure itself carries the downgrade verdict
    ("suggestive"), so the triage cannot be silently omitted at write-up.
    """
    need = [config.estimate_col, *config.spec_columns]
    if config.ci_cols is not None:
        need += list(config.ci_cols)
    _require_columns(df, need, "plot_spec_curve")
    ps.apply_style()

    data = df.sort_values(config.estimate_col).reset_index(drop=True)
    est = data[config.estimate_col].to_numpy(dtype=float)
    if np.isnan(est).any():
        raise FigureDataError("plot_spec_curve: NaN estimates are not renderable")
    ranks = np.arange(len(data))

    # Spec-matrix rows: one per (column, value) level, grouped by column.
    matrix_rows: list[tuple[str, object]] = []
    for col in config.spec_columns:
        for val in data[col].dropna().unique():
            matrix_rows.append((col, val))
    if not matrix_rows:
        raise FigureDataError("plot_spec_curve: spec columns carry no values")

    n_mat = len(matrix_rows)
    fig_h = 2.6 + 0.22 * n_mat
    fig, (ax_top, ax_mat) = plt.subplots(
        2, 1, figsize=(ps.FULL_WIDTH_IN, fig_h), sharex=True,
        gridspec_kw={"height_ratios": [2.6, 0.22 * n_mat]},
    )

    flip = sign_flip_share(est)
    med = float(np.median(est))
    downgraded = flip > config.sign_flip_threshold

    if config.ci_cols is not None:
        lo = data[config.ci_cols[0]].to_numpy(dtype=float)
        hi = data[config.ci_cols[1]].to_numpy(dtype=float)
        ax_top.vlines(ranks, lo, hi, color="0.75", lw=0.9, zorder=1)
    colors = np.where(est >= 0, "#0173b2", "#d55e00")
    ax_top.scatter(ranks, est, s=14, c=colors, zorder=3)
    ax_top.axhline(0.0, color="0.35", lw=0.9, ls="--", zorder=2)
    ax_top.axhline(med, color="0.15", lw=1.1, ls=":", zorder=2)
    ax_top.set_ylabel(config.effect_label)
    verdict = (
        f"TRIAGE: sign flips in {flip:.0%} of specs "
        f"(> {config.sign_flip_threshold:.0%}) → DOWNGRADED TO SUGGESTIVE"
        if downgraded
        else f"sign flips in {flip:.0%} of specs (≤ {config.sign_flip_threshold:.0%} triage threshold)"
    )
    ax_top.annotate(
        f"median = {med:.3g} · n = {len(data)} specs\n{verdict}",
        xy=(0.02, 0.97), xycoords="axes fraction", va="top", fontsize=7,
        color="#b2182b" if downgraded else "0.25",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9),
    )
    ax_top.set_title(config.title or "Specification curve (descriptive-with-triage)")

    for i, (col, val) in enumerate(matrix_rows):
        active = (data[col] == val).to_numpy()
        ax_mat.scatter(ranks[active], np.full(active.sum(), i), s=8,
                       color="#0173b2", marker="s")
    ax_mat.set_yticks(range(n_mat))
    ax_mat.set_yticklabels([f"{c}: {v}" for c, v in matrix_rows], fontsize=6.5)
    ax_mat.set_ylim(n_mat - 0.5, -0.5)  # top-to-bottom in declaration order
    ax_mat.set_xlabel("specifications, sorted by estimate")
    ax_mat.grid(False)
    # Thin separators between spec-column blocks.
    boundary = 0
    for col in config.spec_columns[:-1]:
        boundary += data[col].dropna().nunique()
        ax_mat.axhline(boundary - 0.5, color="0.85", lw=0.7)

    fig.tight_layout()
    outpath = Path(outpath)
    ps.save_fig(fig, outpath)
    return outpath


# ---------------------------------------------------------------------------
# (d) Truth-tax scatter
# ---------------------------------------------------------------------------


def _check_fraction_scale(df: pd.DataFrame, g_col: str, y_col: str, where: str) -> None:
    g = df[g_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    if np.isnan(g).any() or np.isnan(y).any():
        raise FigureDataError(f"{where}: NaN in {g_col!r}/{y_col!r}")
    if (g < 0).any() or (g > 1).any() or (y < 0).any() or (y > 1).any():
        raise FigureDataError(
            f"{where}: {g_col!r}/{y_col!r} must be fraction-of-issued in [0, 1] "
            "(the pinned Y scale — see module docstring); rate-scaled inputs are "
            "a different named scale and are rejected here"
        )
    if (y > g + 1e-9).any():
        raise FigureDataError(
            f"{where}: {y_col!r} > {g_col!r} on some rows — serving yield Y "
            "(SLO ∧ predicate) is a subset of SLO-passing completions by "
            "construction; upstream computation is broken"
        )


def plot_truth_tax(df: pd.DataFrame, outpath: Path, *, config: TruthTaxConfig) -> Path:
    """F2 truth-tax scatter: Y (serving yield) vs G with the y=x line, per dataset.

    Both axes are fraction-of-issued (pinned scale; module docstring). Vertical
    distance below the diagonal is the truth tax G − Y (§9.2 estimand variable).
    Color = policy, marker = engine (derived from row keys); one facet per dataset
    (pooling prohibited, §9.1). Mean tax annotated per facet.
    """
    _require_columns(
        df, [KEY_COL, config.g_col, config.y_col, config.dataset_col], "plot_truth_tax"
    )
    _check_fraction_scale(df, config.g_col, config.y_col, "plot_truth_tax")
    data = expand_row_keys(df)
    ps.apply_style()

    datasets = list(data[config.dataset_col].dropna().unique())
    if not datasets:
        raise FigureDataError(f"plot_truth_tax: {config.dataset_col!r} has no values")
    policies = sorted(data["policy"].unique())
    engines = sorted(data["engine"].unique())
    pol_color = {p: c for p, c in zip(policies, plt.cm.tab10.colors)}
    eng_marker = {e: m for e, m in zip(engines, ("o", "s", "^", "D", "v", "P"))}

    n_panel = len(datasets)
    fig_w = min(ps.FULL_WIDTH_IN, 1.2 + 2.2 * n_panel)
    fig, axes = plt.subplots(
        1, n_panel, figsize=(fig_w, 2.7), sharex=True, sharey=True, squeeze=False
    )
    for ax, ds in zip(axes[0], datasets):
        panel = data[data[config.dataset_col] == ds]
        ax.plot([0, 1], [0, 1], color="0.4", lw=0.9, ls="--", zorder=1)
        for (pol, eng), grp in panel.groupby(["policy", "engine"], observed=True):
            ax.scatter(
                grp[config.g_col], grp[config.y_col], s=22,
                color=pol_color[pol], marker=eng_marker[eng],
                edgecolors="white", lw=0.4, zorder=3,
            )
        tax = (panel[config.g_col] - panel[config.y_col]).mean()
        ax.annotate(
            f"mean tax = {tax:.3f}", xy=(0.03, 0.97), xycoords="axes fraction",
            va="top", fontsize=7, color="0.25",
        )
        ax.set_title(str(ds))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("G (SLO-passing / issued)")
    # "&" not "∧": U+2227 is missing from Arial (tofu on macOS builds).
    axes[0][0].set_ylabel("Y — serving yield\n(SLO & predicate / issued)")
    handles = [
        Line2D([], [], ls="", marker="o", color=pol_color[p], label=f"policy: {p}")
        for p in policies
    ] + [
        Line2D([], [], ls="", marker=eng_marker[e], color="0.35", label=f"engine: {e}")
        for e in engines
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02),
               ncol=min(4, len(handles)), frameon=True)
    if config.title:
        fig.suptitle(config.title, y=1.04)
    fig.tight_layout()
    outpath = Path(outpath)
    ps.save_fig(fig, outpath)
    return outpath


# ---------------------------------------------------------------------------
# (e) Goodput-grid skeleton
# ---------------------------------------------------------------------------


def plot_goodput_grid(
    df: pd.DataFrame, outpath: Path, *, config: GoodputGridConfig
) -> Path:
    """F1 skeleton: goodput-vs-rate small multiples (engine × model), budget ramp.

    Per facet: one curve pair per budget r (sequential color ramp, comfortable
    r=1.5 light → starved r=0.25 dark) — G solid, Y (serving yield) dashed in
    the SAME hue, the G−Y gap shaded (= the truth tax on the pinned
    fraction-of-issued scale; module docstring). Grid points are drawn explicitly:
    filled markers when IN_REGIME, hollow when labeled UNPRESSURED or PAST_CLIFF
    (§6.1 3-layer criterion; labels = ``src.analysis.goodput`` vocabulary, the
    ``label_regime`` output). Knee (Chiu-Jain argmax) and cliff (retrograde
    goodput) are open glyphs on the G curve. Real knee/cliff interpolation +
    replication intervals arrive with GoodputEvaluator (P0-3); this renderer
    consumes their flags.
    """
    need = [KEY_COL, config.g_col, config.y_col, config.regime_col,
            config.knee_col, config.cliff_col]
    _require_columns(df, need, "plot_goodput_grid")
    _check_fraction_scale(df, config.g_col, config.y_col, "plot_goodput_grid")
    bad = set(df[config.regime_col].unique()) - _REGIME_LABELS
    if bad:
        raise FigureDataError(
            f"plot_goodput_grid: unknown regime labels {sorted(map(str, bad))}; "
            f"allowed (§6.1): {sorted(_REGIME_LABELS)}"
        )
    for flag_col in (config.knee_col, config.cliff_col):
        vals = df[flag_col]
        # NaN would silently cast to True under astype(bool) — reject it.
        if vals.isna().any() or not set(vals.unique()) <= {True, False, 0, 1}:
            raise FigureDataError(
                f"plot_goodput_grid: {flag_col!r} must be strictly boolean "
                f"(no NaN); got values {sorted(map(str, vals.unique()))}"
            )
    data = expand_row_keys(df)
    if data["budget_r"].isna().any() or data["rate_frac"].isna().any():
        raise FigureDataError(
            "plot_goodput_grid: every grid row key must carry both pressure coords "
            "(r{g} and lam{g}); sub-pressure cells do not belong on this figure"
        )
    ps.apply_style()

    models = sorted(data["model"].unique())
    engines = sorted(data["engine"].unique())
    budgets = sorted(data["budget_r"].unique(), reverse=True)  # comfortable first
    cmap = plt.get_cmap("viridis")
    # Comfortable r=1.5 light -> starved r=0.25 dark; avoid the pale cmap end.
    shade = {r: cmap(0.85 - 0.7 * i / max(1, len(budgets) - 1))
             for i, r in enumerate(budgets)}

    n_r, n_c = len(models), len(engines)
    fig_w = min(ps.FULL_WIDTH_IN, 1.4 + 2.1 * n_c)
    fig, axes = plt.subplots(
        n_r, n_c, figsize=(fig_w, 1.1 + 1.9 * n_r),
        sharex=True, sharey=True, squeeze=False,
    )
    for i, model in enumerate(models):
        for j, engine in enumerate(engines):
            ax = axes[i][j]
            facet = data[(data["model"] == model) & (data["engine"] == engine)]
            for r in budgets:
                curve = facet[facet["budget_r"] == r].sort_values("rate_frac")
                if curve.empty:
                    continue
                x = curve["rate_frac"].to_numpy(dtype=float)
                g = curve[config.g_col].to_numpy(dtype=float)
                y = curve[config.y_col].to_numpy(dtype=float)
                color = shade[r]
                ax.plot(x, g, "-", color=color, lw=1.3, zorder=3, label=f"r={r:g}")
                ax.plot(x, y, "--", color=color, lw=1.1, zorder=3)
                ax.fill_between(x, y, g, color=color, alpha=0.13, lw=0, zorder=2)
                in_regime = (curve[config.regime_col] == IN_REGIME).to_numpy()
                for series in (g, y):
                    ax.scatter(x[in_regime], series[in_regime], s=13, color=color,
                               zorder=4)
                    ax.scatter(x[~in_regime], series[~in_regime], s=13,
                               facecolors="none", edgecolors=color, lw=1.0, zorder=4)
                knee = curve[config.knee_col].to_numpy(dtype=bool)
                cliff = curve[config.cliff_col].to_numpy(dtype=bool)
                ax.scatter(x[knee], g[knee], s=95, marker="D", facecolors="none",
                           edgecolors="0.1", lw=1.2, zorder=5)
                ax.scatter(x[cliff], g[cliff], s=95, marker="v", facecolors="none",
                           edgecolors="0.1", lw=1.2, zorder=5)
            ax.set_title(f"{engine} · {model}", fontsize=8.5)
            if i == n_r - 1:
                ax.set_xlabel("offered rate (fraction of predicted λ*)")
            if j == 0:
                ax.set_ylabel("fraction of issued")
            ax.set_xticks(sorted(data["rate_frac"].unique()))  # grid points explicit
            ax.set_ylim(-0.02, 1.05)

    budget_handles = [
        Line2D([], [], color=shade[r], lw=1.3, label=f"r = {r:g}") for r in budgets
    ]
    semantic_handles = [
        Line2D([], [], color="0.25", lw=1.3, ls="-", label="G (SLO-pass / issued)"),
        Line2D([], [], color="0.25", lw=1.1, ls="--", label="Y — serving yield"),
        Line2D([], [], ls="", marker="o", mfc="none", mec="0.25", label="out-of-regime"),
        Line2D([], [], ls="", marker="D", mfc="none", mec="0.1", ms=8, label="knee"),
        Line2D([], [], ls="", marker="v", mfc="none", mec="0.1", ms=8, label="cliff"),
    ]
    fig.legend(handles=budget_handles + semantic_handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.02), ncol=min(5, len(budget_handles) + 2),
               frameon=True, fontsize=7)
    if config.title:
        fig.suptitle(config.title, y=1.02)
    fig.tight_layout()
    outpath = Path(outpath)
    ps.save_fig(fig, outpath)
    return outpath


# ---------------------------------------------------------------------------
# (f) CONSORT-style cell flow diagram (§9.10)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsortStage:
    """One box in the §9.10 cell-flow figure.

    ``excluded`` cells leave the flow AFTER this stage; every exclusion must
    carry a reason (that accountability is the figure's entire point).
    """

    label: str
    count: int
    excluded: int = 0
    exclusion_reason: str = ""


def plot_consort_flow(
    stages: Sequence[ConsortStage],
    outpath: Path,
    *,
    title: str = "Campaign cell flow (planned → run → in-regime → analyzed)",
) -> Path:
    """Render the §9.10 CONSORT-style flow: stage boxes down the middle,
    labeled exclusion boxes to the right.

    The stage arithmetic must close exactly
    (``stages[i+1].count == stages[i].count - stages[i].excluded``) — a
    mismatch raises instead of rendering a figure that miscounts cells.
    """
    if len(stages) < 2:
        raise ValueError("CONSORT flow needs at least two stages")
    for s in stages:
        if s.count < 0 or s.excluded < 0:
            raise ValueError(f"negative count in stage {s.label!r}")
        if s.excluded > s.count:
            raise ValueError(f"stage {s.label!r} excludes more cells than it has")
        if s.excluded > 0 and not s.exclusion_reason:
            raise ValueError(f"stage {s.label!r} has exclusions but no reason")
    for prev, nxt in zip(stages, stages[1:]):
        expected = prev.count - prev.excluded
        if nxt.count != expected:
            raise ValueError(
                f"flow arithmetic does not close: {prev.label!r} leaves "
                f"{expected} cells but {nxt.label!r} claims {nxt.count}"
            )

    n = len(stages)
    fig, ax = plt.subplots(figsize=(7.0, max(3.2, 1.5 * n)))
    ax.set_axis_off()
    ys = np.linspace(0.92, 0.08, n)
    stage_box = dict(boxstyle="round,pad=0.5", facecolor="white",
                     edgecolor="black", linewidth=1.2)
    excl_box = dict(boxstyle="round,pad=0.4", facecolor="0.95",
                    edgecolor="black", linewidth=0.9)
    for i, (stage, y) in enumerate(zip(stages, ys)):
        ax.text(0.32, y, f"{stage.label}\nn = {stage.count}",
                ha="center", va="center", fontsize=10,
                bbox=stage_box, transform=ax.transAxes)
        if i < n - 1:
            y_next = ys[i + 1]
            ax.annotate("", xy=(0.32, y_next + 0.05), xytext=(0.32, y - 0.05),
                        xycoords=ax.transAxes, textcoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="-|>", linewidth=1.1,
                                        color="black"))
            if stage.excluded:
                y_mid = (y + y_next) / 2.0
                ax.text(0.78, y_mid,
                        f"excluded: {stage.excluded}\n{stage.exclusion_reason}",
                        ha="center", va="center", fontsize=8.5,
                        bbox=excl_box, transform=ax.transAxes)
                ax.annotate("", xy=(0.6, y_mid), xytext=(0.335, y_mid),
                            xycoords=ax.transAxes, textcoords=ax.transAxes,
                            arrowprops=dict(arrowstyle="-|>", linewidth=0.9,
                                            color="black"))
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    outpath = Path(outpath)
    ps.save_fig(fig, outpath)
    return outpath
