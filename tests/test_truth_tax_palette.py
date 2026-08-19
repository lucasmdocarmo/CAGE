"""K-series palette pin (task #140): the truth-tax figure styling contract.

The Topic-12 note prescribes a NEGATIVE pin on the truth-tax palette: the
audit-I11 fix replaced the pilot ``plt.cm.tab10`` zips (silent truncation,
non-colorblind-safe) with the ``_plot_style`` categorical helpers, and the
only guards were source greps. This file asserts the contract from EXECUTION
and pins it against drift:

- rendering ``plot_truth_tax`` places ONLY colorblind-safe
  ``CATEGORICAL_PALETTE`` colors on the scatter artists and ONLY
  ``CATEGORICAL_MARKERS`` glyphs in the legend (captured via the real figure
  object, pre-save);
- NEGATIVE: no rendered artist wears a tab10 color, and no tab10 *usage*
  (attribute access or "tab10" argument) exists anywhere in the plotting
  stack, checked by AST so prose mentions in docstrings stay legal;
- the registered palette itself is byte-pinned (8 seaborn-colorblind hexes,
  8 marker glyphs) — a silent palette swap is a styling-contract break, not
  a tweak.

Pure local, Agg backend; no GPU, no network.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = REPO_ROOT / "scripts" / "4_analysis"
for _p in (str(_SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _plot_style as ps  # noqa: E402
import figure_pipeline as fp  # noqa: E402
from src.analysis.cellspec import CellSpec  # noqa: E402

#: matplotlib's tab10 hexes — the banned pilot palette (negative pin).
TAB10_HEXES = frozenset({
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
})

#: The frozen styling contract: seaborn "colorblind" hexes, ordered for
#: adjacent contrast, capped at the dataviz-doctrine budget of 8.
REGISTERED_PALETTE = [
    "#0173b2", "#de8f05", "#029e73", "#d55e00",
    "#cc78bc", "#ca9161", "#949494", "#56b4e9",
]
REGISTERED_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def _truth_tax_fixture() -> pd.DataFrame:
    rng = np.random.default_rng(23)
    rows = []
    for ds in ("squad_v2", "hotpotqa"):
        for policy in ("none", "evict", "compress-fp8"):
            for engine in ("vllm", "sglang", "lmdeploy"):
                spec = CellSpec(
                    arm="gold-fresh", retriever="none", policy=policy,
                    topology="single", engine=engine, model="qwen3-14b",
                    family="F2", budget_r=0.5, rate_frac=0.85,
                )
                g = float(rng.uniform(0.5, 0.95))
                rows.append({
                    "row_key": spec.to_row_key(),
                    "dataset": ds,
                    "goodput_frac": g,
                    "yield_frac": g * float(rng.uniform(0.6, 1.0)),
                })
    return pd.DataFrame(rows)


@pytest.fixture()
def rendered_truth_tax(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Render plot_truth_tax for real, capturing the Figure pre-close."""
    import matplotlib.pyplot as plt

    captured = {}
    real_save = ps.save_fig

    def _capture(fig, path):
        captured["fig"] = fig
        fig.savefig(path)  # still write the file; keep the fig open to inspect
        print(f"  wrote {Path(path).name}")

    monkeypatch.setattr(ps, "save_fig", _capture)
    try:
        fp.plot_truth_tax(
            _truth_tax_fixture(), tmp_path / "truth_tax.png",
            config=fp.TruthTaxConfig(),
        )
        assert "fig" in captured, "ps.save_fig was never reached"
        yield captured["fig"]
    finally:
        monkeypatch.setattr(ps, "save_fig", real_save)
        if "fig" in captured:
            plt.close(captured["fig"])


# ---------------------------------------------------------------------------
# Executed contract: rendered artists wear the registered palette only
# ---------------------------------------------------------------------------


def test_truth_tax_scatter_colors_are_registered_palette_only(rendered_truth_tax) -> None:
    from matplotlib.collections import PathCollection

    seen = set()
    for ax in rendered_truth_tax.axes:
        for coll in ax.collections:
            if isinstance(coll, PathCollection):
                for rgba in coll.get_facecolor():
                    seen.add(to_hex(rgba).lower())
    assert seen, "no scatter artists found on the truth-tax figure"
    allowed = {h.lower() for h in ps.CATEGORICAL_PALETTE}
    off_palette = seen - allowed
    assert not off_palette, (
        f"truth-tax scatter wears colors outside CATEGORICAL_PALETTE: "
        f"{sorted(off_palette)} (styling contract, K-series #140)"
    )
    banned = seen & {h.lower() for h in TAB10_HEXES}
    assert not banned, f"tab10 colors are back on the truth-tax figure: {sorted(banned)}"


def test_truth_tax_legend_markers_are_registered_cycle_only(rendered_truth_tax) -> None:
    legends = rendered_truth_tax.legends
    assert legends, "truth-tax figure lost its policy/engine legend"
    markers = {
        line.get_marker()
        for legend in legends
        for line in legend.get_lines()
    }
    assert markers, "no legend line handles found"
    off_cycle = markers - set(ps.CATEGORICAL_MARKERS)
    assert not off_cycle, (
        f"truth-tax legend markers outside CATEGORICAL_MARKERS: {sorted(off_cycle)}"
    )


# ---------------------------------------------------------------------------
# Negative pin: no tab10 USAGE anywhere in the plotting stack (AST)
# ---------------------------------------------------------------------------


def _tab10_usages(path: Path) -> list:
    """tab10 attribute accesses or 'tab10' string arguments — code, not prose
    (docstrings CONTAIN 'tab10' legally; no Constant EQUALS it legally)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "tab10":
            hits.append(f"{path.name}:{node.lineno}: attribute .tab10")
        if isinstance(node, ast.Constant) and node.value == "tab10":
            hits.append(f"{path.name}:{node.lineno}: 'tab10' literal")
    return hits


@pytest.mark.parametrize("module", ["_plot_style.py", "figure_pipeline.py"])
def test_no_tab10_usage_in_plotting_stack(module: str) -> None:
    hits = _tab10_usages(_SCRIPTS_DIR / module)
    assert not hits, (
        "tab10 usage is back in the plotting stack (audit I11 / K-series "
        f"negative pin, #140): {hits}"
    )


# ---------------------------------------------------------------------------
# Byte-pin the registered palette (a silent swap is a contract break)
# ---------------------------------------------------------------------------


def test_categorical_palette_is_the_registered_colorblind_set() -> None:
    assert ps.CATEGORICAL_PALETTE == REGISTERED_PALETTE, (
        "CATEGORICAL_PALETTE drifted from the registered seaborn-colorblind "
        "8-hex contract — change BOTH deliberately or neither (#140)"
    )
    assert ps.CATEGORICAL_MARKERS == REGISTERED_MARKERS
    assert len(set(ps.CATEGORICAL_PALETTE)) == 8, "palette hexes must be unique"
    assert not (set(h.lower() for h in ps.CATEGORICAL_PALETTE) & TAB10_HEXES), (
        "a tab10 hex crept into CATEGORICAL_PALETTE"
    )
