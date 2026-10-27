#!/usr/bin/env python3
"""Order:     after stage 14 (analyze), on the pulled landing (owner plot list of 2026-10-21)
Objective: Draw the academic figure set into <landing>/plots/academic/<stamp>/, one figure or one named skip per plot
Cloud:     local

The owner's plot list of 2026-10-21 (MyDocs/RunPod/S1_ACADEMIC_PLOTS_PLAN_2026-10-21.md).
This file draws plots 1, 2, 3, 5, 10, 15, 17 and 18, and the S1 figure block of
2026-10-27 (owner approval, MyDocs/RunPod/S1_TOOLING_SPEC_BLOCKS section 9): a01
to a10. Each plot reads the landing only, refuses with a named reason when its
data is missing (written to plots_manifest.json, never a silent empty figure),
states which windows it used, and writes a caption text beside the figure that
names the rule it draws (clock, denominator, target). Every strategy is labeled
by its full name (src.analysis.cellspec.strategy_name), never by a bare B code.

Sources: cells_index.csv (engine, baseline, dataset, family, budget_r, rate_frac),
requests.jsonl, the predicate tree (predicate/<scoring id>/cells/...), regime.json,
metrics.json, and the yield table (yield_table.evaluate_run: on time = timely,
on time and correct = yield; the same definitions the analysis uses).

Usage: python3 scripts/4_analysis/academic_plots.py <landing> [--run-root R]
       [--scoring-run-id ID] [--calibration-dir D] [--yield-table CSV]
       [--plan plan_registered.json] [--stats stats.json] [--only KEY ...]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for _p in (str(_REPO_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _plot_style as ps  # noqa: E402  (sets the Agg backend)
import matplotlib.pyplot as plt  # noqa: E402

from src.analysis.cellspec import BASELINES, strategy_name, strategy_name_of_cell  # noqa: E402
import figure_pipeline as fp  # noqa: E402

ENGINES: Tuple[str, ...] = ("vllm", "sglang", "hf")
ENGINE_LABEL: Dict[str, str] = {"vllm": "vLLM", "sglang": "SGLang", "hf": "HF"}
IN_REGIME = "IN_REGIME"
READABLE: Dict[str, str] = {"IN_REGIME": "under pressure", "UNPRESSURED": "not pressured",
                            "UNKNOWN_TELEMETRY": "no telemetry", "NO_REGIME_RECORD": "no regime record"}
#: HF serves in process without streaming: its ttft_ms IS the whole generate()
#: call (hf_oracle_adapter.py: streamed_ttft False), never a first-token time
NO_STREAM_ENGINES = frozenset({"hf"})


class PlotSkip(Exception):
    """A plot whose data is missing: the reason is recorded, nothing is drawn."""


def _bnum(b: str) -> int:
    return int(b[1:]) if str(b)[1:].isdigit() else 99


def _strategy_order(labels: Any) -> List[str]:
    """Strategy labels in baseline order (B1 first, the B12 cuts ascending)."""
    rank = {name: i for i, name in enumerate(strategy_name(code) for code in sorted(BASELINES, key=_bnum))}

    def key(label: str) -> Tuple[int, str]:
        base = label.split(" (")[0].split(" with policy")[0]
        return (rank.get(base, 99), label)

    return sorted(set(labels), key=key)


def blabel(b: str, row_key: Optional[str] = None) -> str:
    """The full strategy name of a baseline code (figure block 2026-10-27);
    the B12 corpus cut is read from the row key when one is given."""
    cut = None
    if row_key:
        cut = fp.parse_row_key(row_key).get("corpus_budget_tokens")
    return strategy_name(b, corpus_budget_tokens=cut if b == "B12" else None)


def _write_caption(out_dir: Path, stem: str, text: str) -> None:
    """The rule a figure draws (clock, denominator, target), beside it."""
    (out_dir / f"{stem}.caption.txt").write_text(text.strip() + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def wilson(k: float, n: float, z: float = 1.96) -> Tuple[float, float]:
    """95 percent Wilson interval of a share k/n (n > 0)."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


class Landing:
    """The pulled landing: the run root, its cell index, its predicate tree and
    the per-window yield table."""

    def __init__(self, landing: Path, *, run_root: Optional[Path] = None, scoring_run_id: Optional[str] = None,
                 calibration_dir: Optional[Path] = None, yield_table: Optional[Path] = None,
                 plan: Optional[Path] = None, stats: Optional[Path] = None) -> None:
        self.landing = landing
        #: S0F-99: with the registered plan the yield table grades the first
        #: token on the class target; the E1 figure (a01) needs it too.
        self.plan = plan
        self._stats_path = stats
        self._stats: Optional[Dict[str, Any]] = None
        if run_root is None:
            found = sorted(
                Path(p).parents[1] for p in glob.glob(str(landing / "run" / "*" / "*" / "*" / "index" / "cells_index.csv"))
                if "-dry-" not in p and "rung-calibration" not in p
            )
            if len(found) != 1:
                raise SystemExit(f"academic_plots: {len(found)} run roots under {landing}/run (pass --run-root): {found}")
            run_root = found[0]
        self.run_root = run_root
        cells = pd.read_csv(run_root / "index" / "cells_index.csv", dtype=str, keep_default_na=False)
        for col in ("budget_r", "rate_frac"):
            cells[col] = pd.to_numeric(cells[col].replace("", np.nan), errors="coerce")
        cells["window_name"] = cells["window_dir"].map(lambda s: Path(s).name)
        self.cells = cells
        pred_dirs = sorted(p for p in (run_root / "predicate").glob("*") if p.is_dir())
        if scoring_run_id is None:
            if len(pred_dirs) != 1:
                raise SystemExit(f"academic_plots: {len(pred_dirs)} scoring passes under {run_root}/predicate (pass --scoring-run-id)")
            scoring_run_id = pred_dirs[0].name
        self.scoring_run_id = scoring_run_id
        self.predicate_root = run_root / "predicate" / scoring_run_id
        self.calibration_dir = calibration_dir or landing / "extras" / "calibration" / "pod"
        self._yield_csv = yield_table
        self._yield: Optional[pd.DataFrame] = None
        self.yield_unmatched = 0
        self.grounding_seen = False
        self._rows: Dict[str, pd.DataFrame] = {}

    def rows(self, window_dir: str) -> pd.DataFrame:
        """One window's request rows, each with its predicate verdict (None when unscored)."""
        if window_dir not in self._rows:
            reqs = _read_jsonl(self.run_root / window_dir / "requests.jsonl")
            verdict = {
                (str(r.get("example_id")), str(r.get("repeat_index") or "0"), r.get("record_index")): r.get("predicate")
                for r in _read_jsonl(self.predicate_root / window_dir / "predicate.jsonl")
            }
            for r in reqs:
                r["predicate"] = verdict.get((str(r.get("example_id")), str(r.get("repeat_index") or "0"), r.get("record_index")))
            self._rows[window_dir] = pd.DataFrame(reqs)
        return self._rows[window_dir]

    def request_frame(self, mask: pd.Series) -> pd.DataFrame:
        """Every request row of the selected windows, with the cell's axes attached."""
        parts = []
        for _, c in self.cells[mask].iterrows():
            df = self.rows(c["window_dir"])
            if df.empty:
                continue
            df = df.copy()
            for col in ("engine", "baseline", "dataset", "family", "budget_r", "rate_frac", "window_dir", "row_key"):
                df[f"cell_{col}"] = c[col]
            # the strategy label keeps the two B12 cuts apart (review MEDIUM 8)
            df["cell_strategy"] = blabel(str(c["baseline"]), str(c["row_key"]))
            parts.append(df)
        if not parts:
            raise PlotSkip("no request rows in the selected windows")
        return pd.concat(parts, ignore_index=True)

    def yield_frame(self) -> pd.DataFrame:
        """The per-window yield table joined to the cell index (engine, baseline, rate)."""
        if self._yield is None:
            if self._yield_csv is not None:
                y = pd.read_csv(self._yield_csv, dtype={"row_key": str, "window_dir": str})
            else:
                import yield_table as yt  # noqa: E402

                plan_classes = yt.load_plan_classes(self.plan) if self.plan is not None else None
                y = pd.DataFrame(yt.evaluate_run(self.run_root, baselines=yt.load_baselines(self.calibration_dir),
                                                 scoring_run_id=self.scoring_run_id, plan=plan_classes))
            # S0F-104: the table carries the row key's rung and rate as strings;
            # the cell index supplies the numeric coordinates, so the string
            # columns are dropped before the merge (pandas suffixed both and
            # plot01 raised KeyError 'rate_frac').
            y = y.drop(columns=[c for c in ("rung", "rate_frac") if c in y.columns])
            y = y.merge(self.cells[["row_key", "window_name", "baseline", "budget_r", "rate_frac"]],
                        left_on=["row_key", "window_dir"], right_on=["row_key", "window_name"], how="left")
            # review LOW 13: a table row the index does not know is counted, never
            # silently dropped by the plots' groupbys
            unmatched = y["baseline"].isna()
            self.yield_unmatched = int(unmatched.sum())
            y = y[~unmatched].copy()
            for col in ("n_issued", "n_timely", "n_yield"):
                y[col] = pd.to_numeric(y[col], errors="coerce")
            y["strategy"] = [blabel(str(b), str(k)) for b, k in zip(y["baseline"], y["row_key"])]
            self._yield = y
        return self._yield

    def yield_bar(self) -> str:
        """The first-token bar the yield table was graded on (review LOW 12),
        read from the table's own baseline_source; a mixed table refuses."""
        y = self.yield_frame()
        if "baseline_source" not in y.columns:
            raise PlotSkip("the yield table carries no baseline_source column: the grading bar is unknown")
        src = y.loc[y["skip_reason"].fillna("") == "", "baseline_source"].astype(str)
        # Owner ruling 2026-10-27 (S0F-107): the section 6.1 serving yield keeps
        # its decode-time clause, and every figure that uses it says so.
        kinds = {"the registered class target (first token only)" if v.startswith("class-target:") else
                 "section 6.1: first token within ten times the engine floor AND decode time within five times its floor"
                 for v in src}
        if len(kinds) > 1:
            raise PlotSkip(f"the yield table mixes grading bars: {sorted(kinds)}; one table, one bar")
        return next(iter(kinds)) if kinds else "no record"

    def window_json(self, window_dir: str, name: str) -> Dict[str, Any]:
        return _read_json(self.run_root / window_dir / name)

    def stats(self) -> Dict[str, Any]:
        """The driver's stats.json (the newest analysis stamp unless --stats)."""
        if self._stats is None:
            path = self._stats_path
            if path is None:
                stamps = sorted(p for p in (self.run_root / "analysis").glob("*/stats.json"))
                if not stamps:
                    raise PlotSkip(f"no analysis/<stamp>/stats.json under {self.run_root} (pass --stats)")
                path = stamps[-1]
            self._stats = _read_json(path)
            self._stats["_path"] = str(path)
        return self._stats

    def plan_doc(self) -> Dict[str, Any]:
        if self.plan is None:
            raise PlotSkip("no --plan plan_registered.json supplied (the registered class targets)")
        return _read_json(self.plan)

    def baseline_label(self, row_key: str) -> str:
        """Full strategy name of a row key, or its condensed axes for a policy cell."""
        axes = fp.parse_row_key(row_key)
        if str(axes["engine"]) in NO_STREAM_ENGINES:
            # the reference engine serves no cache and no first token: its
            # cells are named by the engine and the arm, never as a baseline
            return f"{strategy_name('hf')}, {axes['arm']}"
        name = strategy_name_of_cell(str(axes["arm"]), str(axes["retriever"]), policy=str(axes["policy"]),
                                     corpus_budget_tokens=axes.get("corpus_budget_tokens"))  # type: ignore[arg-type]
        return name or f"{axes['arm']} policy {axes['policy']}"


def _out(fig: Any, out_dir: Path, stem: str) -> str:
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    ps.save_fig(fig, out_dir / f"{stem}.png")
    return f"{stem}.png"


def _engines_in(values: Any) -> List[str]:
    return [e for e in ENGINES if e in set(values)]


# --------------------------------------------------------------------------- plots


def plot01_load_curve(L: Landing, out: Path) -> Tuple[str, str]:
    """On time, and on time and correct, against load: B1 and B6 on qasper."""
    y = L.yield_frame()
    y = y[(y["family"] != "F1") & (y["dataset"] == "qasper") & y["baseline"].isin(["B1", "B6"]) & y["n_issued"].gt(0)]
    if y.empty:
        raise PlotSkip("no pressured qasper window of B1 or B6 with a yield record")
    engines = _engines_in(y["engine"])
    fig, axes = plt.subplots(1, len(engines), figsize=(ps.FULL_WIDTH_IN, 3.0), sharey=True, squeeze=False, layout="constrained")
    colors = ps.categorical_colors(["B1", "B6"], "plot01")
    budgets = sorted(y["budget_r"].dropna().unique())
    # the style helper keys its maps by str(level)
    markers = dict(zip(budgets, ps.categorical_markers([str(b) for b in budgets], "plot01").values()))
    for ax, engine in zip(axes[0], engines):
        for (b, r), g in y[y["engine"] == engine].groupby(["baseline", "budget_r"]):
            g = g.groupby("rate_frac")[["n_issued", "n_timely", "n_yield"]].sum().sort_index()
            ax.plot(g.index, g["n_timely"] / g["n_issued"], "-", color=colors[b], marker=markers[r], label=f"{blabel(b)}, r={r:g}: on time")
            ax.plot(g.index, g["n_yield"] / g["n_issued"], "--", color=colors[b], marker=markers[r], mfc="white", label=f"{blabel(b)}, r={r:g}: on time and correct")
        ax.set_title(ENGINE_LABEL[engine])
        ax.set_ylim(0, 1.02)
    fig.supxlabel("offered load (fraction of the cell's saturation rate)", fontsize=8)
    axes[0][0].set_ylabel("share of requests")
    n_loads = y["rate_frac"].nunique()
    fig.suptitle(f"Load curve: on time vs on time and correct (qasper; {blabel('B1')} and {blabel('B6')}; on time = {L.yield_bar()})" + (
        f"\n{n_loads} load level in this run: points, not curves" if n_loads < 2 else ""), fontsize=8)
    axes[0][-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
    return _out(fig, out, "p01_load_curve"), f"{len(y)} windows, {n_loads} load level(s); table rows unknown to the index: {L.yield_unmatched}"


def plot02_truth_tax(L: Landing, out: Path) -> Tuple[str, str]:
    """Under real pressure (IN_REGIME), on time vs on time and correct, per arm."""
    y = L.yield_frame()
    y = y[(y["regime_label"] == IN_REGIME) & (y["dataset"] == "qasper") & y["n_issued"].gt(0)]
    if y.empty:
        raise PlotSkip("no IN_REGIME qasper window with a yield record")
    engines = _engines_in(y["engine"])
    fig, axes = plt.subplots(len(engines), 1, figsize=(ps.FULL_WIDTH_IN, 2.4 * len(engines)), sharey=True, squeeze=False, layout="constrained")
    for ax, engine in zip(axes[:, 0], engines):
        g = y[y["engine"] == engine].groupby("strategy")[["n_issued", "n_timely", "n_yield"]].sum()
        g = g.loc[_strategy_order(g.index)]
        x = np.arange(len(g))
        timely, both = g["n_timely"] / g["n_issued"], g["n_yield"] / g["n_issued"]
        ax.bar(x - 0.2, timely, 0.4, label="on time", color="#0072B2")
        ax.bar(x + 0.2, both, 0.4, label="on time and correct", color="#E69F00")
        for xi, t, bth in zip(x, timely, both):
            if t - bth > 0.005:
                ax.text(xi, t + 0.02, f"-{(t - bth) * 100:.0f} pts", ha="center", fontsize=7)
        ax.set_xticks(x, list(g.index), rotation=30, ha="right", fontsize=7)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel(f"{ENGINE_LABEL[engine]}\nshare of requests")
    axes[0][0].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Truth tax: answers on time but wrong, under real memory pressure (qasper, IN_REGIME windows; bar = {L.yield_bar()})", fontsize=8.5)
    return _out(fig, out, "p02_truth_tax"), f"{len(y)} IN_REGIME windows; table rows unknown to the index: {L.yield_unmatched}"


def plot03_pressure_map(L: Landing, out: Path) -> Tuple[str, str]:
    """For each budget x load, the regime label the windows reached, per engine."""
    c = L.cells[(L.cells["family"] != "F1") & L.cells["budget_r"].notna() & (L.cells["dataset"] != "ruler")].copy()
    if c.empty:
        raise PlotSkip("no pressured window in the cell index")
    c["label"] = [L.window_json(w, "regime.json").get("label") or "NO_REGIME_RECORD" for w in c["window_dir"]]
    labels = sorted(c["label"].unique())
    colors = ps.categorical_colors(labels, "plot03")
    engines = _engines_in(c["engine"])
    budgets, rates = sorted(c["budget_r"].unique()), sorted(c["rate_frac"].unique())
    fig, axes = plt.subplots(1, len(engines), figsize=(ps.FULL_WIDTH_IN, 0.9 + 0.75 * len(budgets)), squeeze=False, layout="constrained")
    for ax, engine in zip(axes[0], engines):
        e = c[c["engine"] == engine]
        for i, r in enumerate(rates):
            for j, b in enumerate(budgets):
                w = e[(e["rate_frac"] == r) & (e["budget_r"] == b)]
                if w.empty:
                    ax.add_patch(plt.Rectangle((i - 0.45, j - 0.45), 0.9, 0.9, fill=False, ls=":", ec="#999999"))
                    continue
                counts = w["label"].value_counts()
                ax.add_patch(plt.Rectangle((i - 0.45, j - 0.45), 0.9, 0.9, color=colors[counts.index[0]], alpha=0.85))
                ax.text(i, j, "\n".join(f"{READABLE.get(k, k)}: {v}" for k, v in counts.items()), ha="center", va="center", fontsize=6.5)
        ax.set_xticks(range(len(rates)), [f"{r:g}" for r in rates])
        ax.set_yticks(range(len(budgets)), [f"{b:g}" for b in budgets])
        ax.set_xlim(-0.6, len(rates) - 0.4)
        ax.set_ylim(-0.6, len(budgets) - 0.4)
        ax.grid(False)
        ax.set_title(ENGINE_LABEL[engine])
        ax.set_xlabel("offered load (fraction of saturation)")
    axes[0][0].set_ylabel("cache budget r (pool / need)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in labels]
    fig.legend(handles, [READABLE.get(k, k) for k in labels], loc="outside lower center", ncol=len(labels), fontsize=7)
    fig.suptitle("Pressure map: the regime each budget x load cell reached (window counts per label)")
    return _out(fig, out, "p03_pressure_map"), f"{len(c)} pressured windows"


def plot05_avg_vs_peak(L: Landing, out: Path) -> Tuple[str, str]:
    """Average fill (regime) against peak fill (telemetry) per pressured window."""
    c = L.cells[(L.cells["family"] != "F1") & (L.cells["dataset"] != "ruler")].copy()
    c["avg"] = [(L.window_json(w, "regime.json").get("inputs") or {}).get("rho_kv_time_avg") for w in c["window_dir"]]
    c["peak"] = [(L.window_json(w, "metrics.json").get("vllm_telemetry") or {}).get("kv_usage_peak") for w in c["window_dir"]]
    c["label"] = [L.window_json(w, "regime.json").get("label") for w in c["window_dir"]]
    c = c.dropna(subset=["avg", "peak"])
    if c.empty:
        raise PlotSkip("no pressured window carries both rho_kv_time_avg and kv_usage_peak")
    fams = [f for f in ("F2", "F3") if f in set(c["family"])]
    fig, axes = plt.subplots(1, len(fams), figsize=(ps.FULL_WIDTH_IN, 3.4), sharey=True, squeeze=False, layout="constrained")
    mk = ps.categorical_markers(_engines_in(c["engine"]), "plot05")
    for ax, fam in zip(axes[0], fams):
        f = c[c["family"] == fam]
        f = f.assign(strategy=[blabel(str(b), str(k)) for b, k in zip(f["baseline"], f["row_key"])])
        col = ps.categorical_colors(_strategy_order(f["strategy"]), f"plot05 {fam}")
        for (b, e), g in f.groupby(["strategy", "engine"]):
            ax.scatter(g["avg"] * 100, g["peak"] * 100, color=col[b], marker=mk[e], s=28, label=f"{b}, {ENGINE_LABEL[e]}",
                       edgecolor="black", linewidth=0.3)
        ax.plot([0, 100], [0, 100], ":", color="#666666", lw=0.8)
        ax.set_xlim(0, 102)
        ax.set_ylim(0, 102)
        ax.set_title({"F2": "fresh prefill (F2)", "F3": "reuse (F3)"}[fam])
        ax.set_xlabel("average cache fill over the window (%)")
        ax.legend(fontsize=6, ncol=2, loc="lower right")
    axes[0][0].set_ylabel("peak cache fill (%)")
    fig.suptitle("Touching full is not staying full: peak vs average cache fill per pressured window")
    return _out(fig, out, "p05_avg_vs_peak"), f"{len(c)} windows"


def _f1_rows(L: Landing, baselines: Optional[List[str]] = None) -> pd.DataFrame:
    mask = (L.cells["family"] == "F1") & L.cells["dataset"].isin(["qasper", "squad_v2"])
    if baselines:
        mask &= L.cells["baseline"].isin(baselines)
    return L.request_frame(mask)


def plot10_ttft_per_arm(L: Landing, out: Path) -> Tuple[str, str]:
    """Median (bar) and 95th percentile (tick) first-token time per arm, no pressure."""
    d = _f1_rows(L)
    d = d[d["ok"].astype(bool) & pd.to_numeric(d["ttft_ms"], errors="coerce").notna()]
    d["ttft_ms"] = pd.to_numeric(d["ttft_ms"])
    datasets = [x for x in ("qasper", "squad_v2") if x in set(d["cell_dataset"])]
    engines = _engines_in(d["cell_engine"])
    col = ps.categorical_colors(engines, "plot10")
    fig, axes = plt.subplots(len(datasets), 1, figsize=(ps.FULL_WIDTH_IN, 2.6 * len(datasets)), squeeze=False, layout="constrained")
    for ax, ds in zip(axes[:, 0], datasets):
        g = d[d["cell_dataset"] == ds].groupby(["cell_strategy", "cell_engine"])["ttft_ms"]
        stats = pd.DataFrame({"p50": g.median(), "p95": g.quantile(0.95)}).reset_index()
        bl = _strategy_order(stats["cell_strategy"])
        w = 0.8 / len(engines)
        for k, e in enumerate(engines):
            s = stats[stats["cell_engine"] == e].set_index("cell_strategy").reindex(bl)
            x = np.arange(len(bl)) + (k - (len(engines) - 1) / 2) * w
            nostream = e in NO_STREAM_ENGINES
            ax.bar(x, s["p50"], w, color=col[e], hatch="///" if nostream else None, edgecolor="white" if nostream else None,
                   label=f"{ENGINE_LABEL[e]} (whole call: no streaming, no first-token time)" if nostream else ENGINE_LABEL[e])
            ax.scatter(x, s["p95"], marker="_", color="black", s=60, zorder=3)
        ax.set_yscale("log")
        ax.set_xticks(range(len(bl)), bl, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel(f"{ds}\nfirst-token time (ms, log)")
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="outside lower center", ncol=len(engines), fontsize=7)
    fig.suptitle("First-token time per arm with no memory pressure (bar = median, tick = 95th percentile)")
    return _out(fig, out, "p10_ttft_per_arm"), f"{len(d)} ok requests"


def plot15_correct_rate(L: Landing, out: Path) -> Tuple[str, str]:
    """Share of correct answers (predicate) per arm, B1 as the ceiling, no pressure."""
    d = _f1_rows(L)
    d = d[d["predicate"].notna()]
    if d.empty:
        raise PlotSkip("no scored request row in the no-pressure windows")
    datasets = [x for x in ("qasper", "squad_v2") if x in set(d["cell_dataset"])]
    engines = _engines_in(d["cell_engine"])
    col = ps.categorical_colors(engines, "plot15")
    fig, axes = plt.subplots(len(datasets), 1, figsize=(ps.FULL_WIDTH_IN, 2.6 * len(datasets)), sharey=True, squeeze=False, layout="constrained")
    for ax, ds in zip(axes[:, 0], datasets):
        f = d[d["cell_dataset"] == ds]
        bl = _strategy_order(f["cell_strategy"])
        w = 0.8 / len(engines)
        for k, e in enumerate(engines):
            x0 = np.arange(len(bl)) + (k - (len(engines) - 1) / 2) * w
            for x, b in zip(x0, bl):
                s = f[(f["cell_engine"] == e) & (f["cell_strategy"] == b)]["predicate"].astype(bool)
                if s.empty:
                    continue
                lo, hi = wilson(s.sum(), len(s))
                ax.bar(x, s.mean(), w, color=col[e], label=ENGINE_LABEL[e] if b == bl[0] else None)
                ax.errorbar(x, s.mean(), yerr=[[s.mean() - lo], [hi - s.mean()]], color="black", lw=0.7, capsize=1.5)
            ceiling = f[(f["cell_engine"] == e) & (f["cell_baseline"] == "B1")]["predicate"].astype(bool)
            if not ceiling.empty:
                ax.axhline(ceiling.mean(), color=col[e], ls="--", lw=0.8)
        ax.set_xticks(range(len(bl)), bl, rotation=30, ha="right", fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f"{ds}\nshare correct")
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="outside lower center", ncol=len(engines), fontsize=7)
    fig.suptitle(f"Correct-answer rate per strategy, no pressure (dashed = {blabel('B1')} per engine; bars = 95% Wilson CI)")
    return _out(fig, out, "p15_correct_rate"), f"{len(d)} scored requests"


def plot17_cache_reuse(L: Landing, out: Path) -> Tuple[str, str]:
    """Cached share of each prompt, reuse off next to its reuse twin, vLLM, no pressure."""
    pairs = [("B1", "B2"), ("B4", "B3")]
    d = _f1_rows(L, [b for p in pairs for b in p])
    d = d[(d["cell_engine"] == "vllm") & d["ok"].astype(bool)]
    d["ratio"] = pd.to_numeric(d.get("cached_prompt_ratio"), errors="coerce")
    d = d.dropna(subset=["ratio"])
    if d.empty:
        raise PlotSkip("no vLLM no-pressure row carries cached_prompt_ratio")
    datasets = [x for x in ("qasper", "squad_v2") if x in set(d["cell_dataset"])]
    fig, axes = plt.subplots(1, len(datasets), figsize=(ps.FULL_WIDTH_IN, 2.8), sharey=True, squeeze=False, layout="constrained")
    for ax, ds in zip(axes[0], datasets):
        f = d[d["cell_dataset"] == ds]
        for i, (off, on) in enumerate(pairs):
            for dx, b, colr in ((-0.2, off, "#999999"), (0.2, on, "#009E73")):
                s = f[f["cell_baseline"] == b]["ratio"]
                if not s.empty:
                    ax.bar(i + dx, s.mean() * 100, 0.4, color=colr, label=("reuse off" if dx < 0 else "reuse on") if i == 0 else None)
                    ax.text(i + dx, s.mean() * 100 + 2, f"{s.mean() * 100:.0f}%", ha="center", fontsize=7)
        ax.set_xticks(range(len(pairs)), [f"{blabel(a)}\nvs {blabel(b)}" for a, b in pairs], fontsize=7)
        ax.set_ylim(0, 105)
        ax.set_title(ds)
    axes[0][0].set_ylabel("cached share of the prompt (%)")
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Is the cache really reused? Cached prompt share, reuse off vs on (vLLM, no pressure)")
    return _out(fig, out, "p17_cache_reuse"), f"{len(d)} vLLM requests"


def plot18_prompt_vs_answer(L: Landing, out: Path) -> Tuple[str, str]:
    """Median time reading the prompt (first token) and writing the answer (the rest)."""
    bls = ["B1", "B3", "B6"]
    d = _f1_rows(L, bls)
    d = d[d["ok"].astype(bool)]
    d["ttft"] = pd.to_numeric(d["ttft_ms"], errors="coerce")
    d["decode"] = pd.to_numeric(d["latency_ms"], errors="coerce") - d["ttft"]
    d = d.dropna(subset=["ttft", "decode"])
    if d.empty:
        raise PlotSkip("no ok no-pressure row of B1, B3 or B6 with ttft_ms and latency_ms")
    datasets = [x for x in ("qasper", "squad_v2") if x in set(d["cell_dataset"])]
    engines = _engines_in(d["cell_engine"])
    fig, axes = plt.subplots(1, len(datasets), figsize=(ps.FULL_WIDTH_IN, 3.2), sharey=True, squeeze=False, layout="constrained")
    for ax, ds in zip(axes[0], datasets):
        f = d[d["cell_dataset"] == ds]
        labels, x = [], 0
        for b in bls:
            for e in engines:
                s = f[(f["cell_baseline"] == b) & (f["cell_engine"] == e)]
                if s.empty:
                    continue
                if e in NO_STREAM_ENGINES:
                    # no first token on a non-streaming engine: one undivided bar
                    ax.bar(x, s["ttft"].median() / 1000, 0.7, color="#BBBBBB", hatch="///", edgecolor="white",
                           label="whole call, not separable (HF: no streaming)" if "HF" not in "".join(labels) else None)
                else:
                    pt, dt = s["ttft"].median() / 1000, s["decode"].median() / 1000
                    ax.bar(x, pt, 0.7, color="#0072B2", label="reading the prompt (first token)" if x == 0 else None)
                    ax.bar(x, dt, 0.7, bottom=pt, color="#CC79A7", label="writing the answer" if x == 0 else None)
                labels.append(f"{blabel(b)}\n{ENGINE_LABEL[e]}")
                x += 1
        ax.set_xticks(range(len(labels)), labels, fontsize=6.5, rotation=45, ha="right")
        ax.set_title(ds)
    axes[0][0].set_ylabel("median time per request (s)")
    handles = {lab: h for ax in axes[0] for h, lab in zip(*ax.get_legend_handles_labels())}
    fig.legend(handles.values(), handles.keys(), loc="outside lower center", ncol=3, fontsize=7)
    fig.suptitle("Where the time goes: prompt reading vs answer writing (no pressure)")
    return _out(fig, out, "p18_prompt_vs_answer"), f"{len(d)} ok requests"




# --------------------------------------------------------------------------- the S1 figure block
# Owner approval 2026-10-27 (MyDocs/RunPod/S1_TOOLING_SPEC_BLOCKS_2026-10-27.md
# section 9). Every figure below reads the landing and the driver's stats.json;
# no registered p is recomputed here, and the per-window shares of a01 and the
# clock of a02 come from the driver's own functions.

CONTROL_LABEL = "control (cache rung 1.25)"
CONSTRAINED_LABEL = "constrained"
#: the per-question fingerprint pairs (strategy, its registered partner), by
#: baseline code and B12 cut (PUBLICATION section 7.8; the E4 design input)
FINGERPRINT_PAIRS: Tuple[Tuple[Tuple[str, Optional[int]], Tuple[str, Optional[int]]], ...] = (
    (("B10", None), ("B3", None)),
    (("B9", None), ("B6", None)),
    (("B11", None), ("B6", None)),
    (("B12", 700), ("B3", None)),
    (("B12", 1400), ("B3", None)),
    (("B2", None), ("B1", None)),
    (("B4", None), ("B3", None)),
    (("B7", None), ("B6", None)),
)
OUTCOME_ORDER: Tuple[str, ...] = ("accepted", "rejected", "ungrounded", "abstained", "unscored", "failed")
OUTCOME_COLOR: Dict[str, str] = {
    "accepted": "#029e73", "rejected": "#d55e00", "ungrounded": "#de8f05",
    "abstained": "#cc78bc", "unscored": "#949494", "failed": "#333333",
}


def _rca():
    """The analysis driver, imported lazily (it pulls scipy and the stats kit)."""
    import run_campaign_analysis as rca  # noqa: E402

    return rca


def _cut_of(row_key: str) -> Optional[int]:
    return fp.parse_row_key(row_key).get("corpus_budget_tokens")  # type: ignore[return-value]


def _strategy_key(baseline: str, row_key: str) -> Tuple[str, Optional[int]]:
    return (str(baseline), _cut_of(row_key) if baseline == "B12" else None)


def _skey_label(key: Tuple[str, Optional[int]]) -> str:
    return strategy_name(key[0], corpus_budget_tokens=key[1])


def _first_pass(df: pd.DataFrame) -> pd.DataFrame:
    """One row per question: the first arrival of each example in record order."""
    if "record_index" in df.columns:
        order = pd.to_numeric(df["record_index"], errors="coerce").fillna(10**9)
        df = df.assign(_order=order).sort_values("_order", kind="stable")
    return df.drop_duplicates("example_id", keep="first")


def _frontier_legs(L: Landing) -> List[Dict[str, Any]]:
    st = L.stats()
    fr = st.get("frontier") or {}
    legs = fr.get("legs") or []
    if not legs:
        raise PlotSkip(f"{st.get('_path')}: no frontier (#12, E1) legs; run the driver with --contrasts 12 --plan")
    return legs


def plot_a01_frontier_drop(L: Landing, out: Path) -> Tuple[str, str]:
    """A1: the yield frontier drop (E1) per engine: window dots, timely and
    accepted means, the registered merged p printed beside each point."""
    legs = _frontier_legs(L)
    rca = _rca()
    st = L.stats()
    fr = st["frontier"]
    verdicts: Dict[str, Dict[str, Any]] = {}
    for d in fr.get("per_dataset_intersection", []):
        verdicts[d["dataset"]] = d
    engines = _engines_in({leg["engine"] for leg in legs})
    datasets = sorted({leg["dataset"] for leg in legs})
    panels = [(ds, engine) for ds in datasets for engine in engines]
    fig, axes = plt.subplots(len(panels), 1, figsize=(ps.FULL_WIDTH_IN, 2.9 * len(panels)),
                             sharey=True, squeeze=False, layout="constrained")
    n_windows = 0
    for ax, (ds, engine) in zip(axes[:, 0], panels):
        if True:
            x = 0
            ticks: List[float] = []
            labels: List[str] = []
            rows = sorted((l for l in legs if l["engine"] == engine and l["dataset"] == ds),
                          key=lambda l: (_bnum(l["baseline"]), -float(l["budget_r"])))
            for leg in rows:
                sides = (
                    (CONTROL_LABEL, leg["control_row_keys"], leg["yield"]["mean_control"], leg["timely"]["mean_control"], "#0173b2"),
                    (f"{CONSTRAINED_LABEL} r={float(leg['budget_r']):g}", [leg["constrained_row_key"]],
                     leg["yield"]["mean_constrained"], leg["timely"]["mean_constrained"], "#d55e00"),
                )
                for k, (side, keys, y_mean, t_mean, color) in enumerate(sides):
                    windows = L.cells[L.cells["row_key"].isin(keys) & (L.cells["dataset"] == ds)]
                    try:
                        shares = [rca._window_yield_shares(L.run_root, rec, L.predicate_root, float(leg["target_s"]))
                                  for rec in windows.itertuples(index=False)]
                    except rca.AnalysisError as exc:
                        raise PlotSkip(f"a01: {exc}") from exc
                    n_windows += len(shares)
                    xs = x + k + np.linspace(-0.18, 0.18, max(len(shares), 1))
                    ax.scatter(xs, [s["yield_frac"] for s in shares], s=9, color=color, alpha=0.45, zorder=2)
                    ax.scatter([x + k], [t_mean], marker="o", s=42, facecolors="none", edgecolors=color, zorder=3,
                               label="timely share (mean)" if x == 0 and k == 0 else None)
                    ax.scatter([x + k], [y_mean], marker="o", s=42, color=color, zorder=4,
                               label="yield share (timely and accepted, mean)" if x == 0 and k == 0 else None)
                    ticks.append(x + k)
                    labels.append("control\nr=1.25" if k == 0 else f"constrained\nr={float(leg['budget_r']):g}")
                y = leg["yield"]
                ax.annotate(
                    f"{blabel(leg['baseline'])}\nΔ yield {y['delta']:+.3f} [{y['ci95_low_bootstrap']:+.2f}, {y['ci95_high_bootstrap']:+.2f}]\n"
                    f"p merged {leg.get('p_merged_engines', y['p_value']):.2g}, Holm {leg.get('p_holm_within_dataset', float('nan')):.2g}",
                    (x + 0.5, 1.05), ha="center", va="bottom", fontsize=5.5, color="0.25",
                )
                x += 3.6
            ax.set_xticks(ticks, labels, fontsize=6)
            ax.set_ylim(0, 1.3)
            ax.set_title(f"{ENGINE_LABEL[engine]}, {ds}", fontsize=9)
            ax.set_ylabel("share of issued requests")
    arm_lines = []
    for ds, d in verdicts.items():
        parts = [f"{blabel(arm)}: {rec['verdict']} (Holm p {rec['p_holm_any']:.3g})" for arm, rec in sorted(d["per_arm"].items())]
        arm_lines.append(f"{ds}: " + "; ".join(parts))
    axes[0][0].legend(fontsize=6, loc="lower left")
    axes[0][0].set_ylim(0, 1.3)
    fig.suptitle("Yield frontier drop (E1): constrained cache rung against the control at the same offered rate\n"
                 + "\n".join(arm_lines), fontsize=8)
    stem = "a01_frontier_drop"
    _write_caption(out, stem, f"""
A1. The yield frontier drop (contrast #12, E1). Each dot is one window's yield share: requests whose first token arrived
within the registered class target on the scheduled clock (ADR-0178 D12, section 6.3 coordinated omission) AND whose
answer the section 8.5 predicate accepted, over every issued request. Hollow marker: the window-mean timely share; filled
marker: the window-mean yield share. The printed delta is constrained minus control with its 95 percent window
bootstrap interval; "p merged" is the registered leg p (engine members merged by the largest p, ruling A13), "Holm" the
Holm-adjusted value over the registered (arm, rung) points; the per-arm verdicts in the title are the registered
disjunctive rule (ADR-0178 amendment 2). Source: {st.get('_path')}; window shares recomputed with the driver's
_window_yield_shares on the same rule. Per-engine readings that the merged p rejects are not confirmed.
""")
    return _out(fig, out, stem), f"{len(legs)} legs, {n_windows} windows"


def plot_a02_ttft_cdf(L: Landing, out: Path) -> Tuple[str, str]:
    """A2: first-token time CDF on the scheduled clock, control against
    constrained, with the class-target line, per engine."""
    legs = _frontier_legs(L)
    rca = _rca()
    engines = _engines_in({leg["engine"] for leg in legs})
    fig, axes = plt.subplots(1, len(engines), figsize=(ps.FULL_WIDTH_IN, 3.2), sharey=True, squeeze=False, layout="constrained")
    baselines = sorted({leg["baseline"] for leg in legs}, key=_bnum)
    colors = ps.categorical_colors(baselines, "a02")
    n_rows = 0
    for ax, engine in zip(axes[0], engines):
        targets_drawn: set = set()
        sides_drawn: set = set()
        for leg in sorted((l for l in legs if l["engine"] == engine), key=lambda l: (_bnum(l["baseline"]), -float(l["budget_r"]))):
            for side, keys, ls in ((CONTROL_LABEL, leg["control_row_keys"], "-"),
                                   (f"{CONSTRAINED_LABEL} r={float(leg['budget_r']):g}", [leg["constrained_row_key"]], "--")):
                if (tuple(keys), leg["dataset"]) in sides_drawn:
                    continue  # a control shared by two legs is drawn and counted once
                sides_drawn.add((tuple(keys), leg["dataset"]))
                mask = L.cells["row_key"].isin(keys) & (L.cells["dataset"] == leg["dataset"])
                try:
                    d = L.request_frame(mask)
                except PlotSkip:
                    continue
                d = d[d["ok"].astype(bool)]
                values = []
                for req in d.to_dict("records"):
                    try:
                        ms, _basis = rca._slo_ttft_ms(req)
                    except rca.AnalysisError as exc:
                        raise PlotSkip(f"a02: {exc}") from exc
                    if ms is not None and math.isfinite(ms):
                        values.append(ms / 1000.0)
                if not values:
                    continue
                n_rows += len(values)
                v = np.sort(np.asarray(values))
                label = f"{blabel(leg['baseline'])}, {side}"
                drawn = {h.get_label() for h in ax.get_lines()}
                ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=colors[leg["baseline"]], ls=ls, lw=1.1,
                        label=label if label not in drawn else None)
            target = float(leg["target_s"])
            if (leg["baseline"], target) not in targets_drawn:
                ax.axvline(target, color=colors[leg["baseline"]], ls=":", lw=0.9)
                ax.annotate(f"class target {target:.2f} s", (target, 0.5), rotation=90, fontsize=6, ha="right", va="center", color=colors[leg["baseline"]])
                targets_drawn.add((leg["baseline"], target))
        ax.set_xscale("log")
        ax.set_xlabel("first-token time from the scheduled arrival (s, log)")
        ax.set_title(ENGINE_LABEL[engine])
    handles = {lab: h for ax in axes[0] for h, lab in zip(*ax.get_legend_handles_labels())}
    fig.legend(handles.values(), handles.keys(), loc="outside lower center", ncol=2, fontsize=6)
    axes[0][0].set_ylabel("share of completed requests")
    fig.suptitle("First-token time on the registered clock: control against constrained cache rung, with the class target")
    stem = "a02_ttft_cdf"
    _write_caption(out, stem, """
A2. Empirical CDF of the first-token time of every completed request of the K1 cells, measured from the scheduled
(intended) arrival, the section 6.3 coordinated-omission clock the registration grades on (driver rule _slo_ttft_ms,
W16). The dotted line is the registered class target shared by both engines (plan_registered.json
class_first_token_targets_s). The curve is over completed requests; E1's timely share is over issued requests, so the
two agree only where every request completed. Solid: the control rung 1.25; dashed: the constrained rung at the same
offered rate. A control cell shared by two legs is drawn once.
""")
    return _out(fig, out, stem), f"{n_rows} completed requests"


def plot_a03_paired_identity(L: Landing, out: Path) -> Tuple[str, str]:
    """A3: the same question at no load and under load: identical answer bytes
    and identical verdicts, per strategy and engine."""
    cells = L.cells[(L.cells["dataset"] != "ruler")]
    pressure = cells[cells["family"] != "F1"]
    if pressure.empty:
        raise PlotSkip("no pressure window in the cell index: nothing to pair against the no-load pass")
    records: Dict[Tuple[str, Tuple[str, Optional[int]]], Dict[str, int]] = {}
    for (engine, baseline, dataset), grp in pressure.groupby(["engine", "baseline", "dataset"]):
        for row_key, pgrp in grp.groupby("row_key"):
            skey = _strategy_key(str(baseline), str(row_key))
            f1 = cells[(cells["engine"] == engine) & (cells["baseline"] == baseline) & (cells["dataset"] == dataset)
                       & (cells["family"] == "F1") & (cells["row_key"].map(lambda k: _strategy_key(str(baseline), str(k))) == skey)]
            if f1.empty:
                continue
            base: Dict[str, Tuple[Any, Any]] = {}
            for wdir in f1["window_dir"]:
                d = _first_pass(L.rows(wdir))
                for r in d.itertuples(index=False):
                    base.setdefault(str(r.example_id), (getattr(r, "generated_answer", None), getattr(r, "predicate", None)))
            rec = records.setdefault((str(engine), skey), {"n": 0, "same_answer": 0, "same_verdict": 0, "windows": 0})
            for wdir in pgrp["window_dir"]:
                d = _first_pass(L.rows(wdir))
                rec["windows"] += 1
                for r in d.itertuples(index=False):
                    pair = base.get(str(r.example_id))
                    verdict = getattr(r, "predicate", None)
                    if pair is None or pair[1] is None or verdict is None or (isinstance(verdict, float) and math.isnan(verdict)):
                        continue  # unscored on either side: no verdict identity to count
                    rec["n"] += 1
                    rec["same_answer"] += int(getattr(r, "generated_answer", None) == pair[0])
                    rec["same_verdict"] += int(verdict == pair[1])
    records = {k: v for k, v in records.items() if v["n"]}
    if not records:
        raise PlotSkip("no question served both at no load (F1) and under pressure by the same strategy and engine")
    engines = _engines_in({k[0] for k in records})
    keys = sorted({k[1] for k in records}, key=lambda s: (_bnum(s[0]), s[1] or 0))
    fig, axes = plt.subplots(len(engines), 1, figsize=(ps.FULL_WIDTH_IN, 2.5 * len(engines)), sharex=True, squeeze=False, layout="constrained")
    for ax, engine in zip(axes[:, 0], engines):
        xs = np.arange(len(keys))
        same_a = [records.get((engine, k), {}).get("same_answer", 0) / max(records.get((engine, k), {}).get("n", 1), 1) if (engine, k) in records else np.nan for k in keys]
        same_v = [records.get((engine, k), {}).get("same_verdict", 0) / max(records.get((engine, k), {}).get("n", 1), 1) if (engine, k) in records else np.nan for k in keys]
        ax.bar(xs - 0.2, same_a, 0.4, color="#0173b2", label="identical answer text")
        ax.bar(xs + 0.2, same_v, 0.4, color="#029e73", label="identical verdict")
        for x, k in zip(xs, keys):
            if (engine, k) in records:
                ax.annotate(f"n={records[(engine, k)]['n']}", (x, 1.01), ha="center", fontsize=5.5, color="0.35")
        ax.set_ylim(0, 1.08)
        ax.set_ylabel(f"{ENGINE_LABEL[engine]}\nshare of paired questions")
        ax.set_xticks(xs, [_skey_label(k) for k in keys], rotation=30, ha="right", fontsize=6.5)
    axes[0][0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Does load change the answer? The same question at no load and under pressure: identical text, identical verdict")
    stem = "a03_paired_identity"
    _write_caption(out, stem, """
A3. For every strategy and engine, each question served under pressure (F2/F3 windows) is paired with the same question
served at no load (the F1 window of the same strategy, engine and dataset; the first arrival of the question in each
window). The dark bar is the share of pairs whose generated answer text is byte-identical; the light bar the share whose
section 8.5 predicate verdict is identical. n is the number of pairs pooled over the strategy's pressure windows: a
question served in k pressure windows contributes k pairs against its one no-load answer, and pairs unscored on either
side are left out. The verdict identity is the scorer's determinism on its inputs (same question, same answer text,
same context), not the within-window inheritance rule of the scoring pass, which applies to replays inside one window.
""")
    return _out(fig, out, stem), f"{sum(v['n'] for v in records.values())} pairs over {len(records)} strategy-engine groups"


def _mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p on the discordant counts (binomial, 0.5)."""
    from scipy.stats import binomtest

    if b + c == 0:
        return 1.0
    return float(binomtest(min(b, c), b + c, 0.5).pvalue)


def plot_a04_fingerprint_heatmap(L: Landing, out: Path) -> Tuple[str, str]:
    """A4: the coping-strategy fingerprint per question at matched pressure
    points (descriptive), with the registered #13 leg points printed beside."""
    pressure = L.cells[(L.cells["family"] != "F1") & (L.cells["dataset"] == "qasper")]
    if pressure.empty:
        raise PlotSkip("no pressure window on qasper (the registered pressure dataset)")
    maps: Dict[Tuple[str, float, float, str], Dict[Tuple[str, Optional[int]], Dict[str, bool]]] = {}
    for row in pressure.itertuples(index=False):
        skey = _strategy_key(str(row.baseline), str(row.row_key))
        point = (str(row.engine), float(row.budget_r), float(row.rate_frac), str(row.window_name))
        d = _first_pass(L.rows(row.window_dir))
        maps.setdefault(point, {})[skey] = {str(r.example_id): getattr(r, "predicate", None) is True for r in d.itertuples(index=False)}
    totals: Dict[Tuple[str, int], List[int]] = {}
    for point, tags in maps.items():
        for i, (a, b) in enumerate(FINGERPRINT_PAIRS):
            if a in tags and b in tags:
                fa, fb = tags[a], tags[b]
                common = set(fa) & set(fb)
                t = totals.setdefault((point[0], i), [0, 0, 0, 0])
                t[0] += sum(1 for e in common if fa[e] and fb[e])
                t[1] += sum(1 for e in common if fa[e] and not fb[e])
                t[2] += sum(1 for e in common if fb[e] and not fa[e])
                t[3] += sum(1 for e in common if not fa[e] and not fb[e])
    if not totals:
        raise PlotSkip("no registered fingerprint pair shares a pressure point on qasper")
    engines = _engines_in({k[0] for k in totals})
    pair_ids = sorted({k[1] for k in totals})
    same = np.full((len(pair_ids), len(engines)), np.nan)
    diff = np.full_like(same, np.nan)
    pvals = np.full_like(same, np.nan)
    ns = np.zeros_like(same, dtype=int)
    for r, i in enumerate(pair_ids):
        for c, engine in enumerate(engines):
            t = totals.get((engine, i))
            if t is None:
                continue
            n = sum(t)
            same[r, c] = (t[0] + t[3]) / n
            diff[r, c] = (t[1] - t[2]) / n
            pvals[r, c] = _mcnemar_exact(t[1], t[2])
            ns[r, c] = n
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(ps.FULL_WIDTH_IN, 0.9 + 0.42 * len(pair_ids)),
                                        gridspec_kw={"width_ratios": [1, 1, 1.15]}, layout="constrained")
    row_labels = [f"{_skey_label(FINGERPRINT_PAIRS[i][0])}\nvs {_skey_label(FINGERPRINT_PAIRS[i][1])}" for i in pair_ids]
    lim = float(np.nanmax(np.abs(diff))) if np.isfinite(np.nanmax(np.abs(diff))) else 0.05
    im1 = ax1.imshow(same, cmap="Blues", vmin=float(np.nanmin(same)), vmax=1.0, aspect="auto")
    im2 = ax2.imshow(diff, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    for r in range(len(pair_ids)):
        for c in range(len(engines)):
            if not np.isnan(same[r, c]):
                ax1.annotate(f"{same[r, c]:.3f}\nn={ns[r, c]}", (c, r), ha="center", va="center", fontsize=6)
                p_txt = f"p={pvals[r, c]:.1e}" if pvals[r, c] > 0 else "p<1e-300"
                ax2.annotate(f"{diff[r, c]:+.3f}\n{p_txt}", (c, r), ha="center", va="center", fontsize=6)
    for ax, im, title in ((ax1, im1, "same verdict share"), (ax2, im2, "accepted share, strategy minus partner")):
        ax.set_xticks(range(len(engines)), [ENGINE_LABEL[e] for e in engines], fontsize=7)
        ax.set_title(title, fontsize=8)
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.7)
    ax1.set_yticks(range(len(pair_ids)), row_labels, fontsize=6)
    ax2.set_yticks([])
    ax3.set_axis_off()
    st = L.stats()
    fpr = st.get("fingerprint") or {}
    lines = ["Registered #13 (stats.json):"]
    for lp in fpr.get("leg_points", []) if isinstance(fpr.get("leg_points"), list) else []:
        lines.append(f"leg {lp['leg']} x {lp['dataset']}: {lp['n_points']} points, p_leg {lp['p_leg']:.2e}")
    for iu in fpr.get("per_dataset_intersection", []):
        lines.append(f"{iu['dataset']}: endpoint IU p = {iu['p_intersection_union']:.3g} ({iu.get('n_legs_supplied')}/{iu.get('holm_m_registered')} legs)")
    for sk in fpr.get("skipped", []):
        lines.append(f"{sk['leg']}: absent ({sk['reason'][:40]})")
    if len(lines) == 1:
        lines.append("no fingerprint section (run --contrasts 13)")
    ax3.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=6, family="monospace", transform=ax3.transAxes, wrap=True)
    fig.suptitle("Fingerprint of each coping strategy: per-question agreement with its partner at matched pressure points (qasper)", fontsize=8.5)
    stem = "a04_fingerprint_heatmap"
    _write_caption(out, stem, f"""
A4. Descriptive per-question fingerprint on Qasper: at every matched pressure point (same engine, cache rung, load and
window ordinal) each strategy's first-pass verdict per question is paired with its registered partner's. Left: the share
of questions with the same verdict. Right: the accepted share of the strategy minus its partner's, with the exact
two-sided McNemar p on the discordant pairs (descriptive, not a registered test). The text box prints the registered
contrast #13 values from {st.get('_path')}: the truncate leg's point count and leg p, and the intersection-union
endpoint, which is 1.0 by construction whenever the evict and compress legs were not served.
""")
    return _out(fig, out, stem), f"{int(ns.sum())} paired verdicts over {len(totals)} pair-engine cells"


def _outcome(row: Any) -> str:
    if not bool(getattr(row, "ok", False)):
        return "failed"
    pred = getattr(row, "predicate", None)
    if pred is None or (isinstance(pred, float) and math.isnan(pred)):
        return "unscored"
    if pred is True:
        return "accepted"
    if bool(getattr(row, "predicted_no_answer", False)):
        return "abstained"
    if getattr(row, "grounded", None) is False:
        return "ungrounded"
    return "rejected"


def plot_a05_outcome_breakdown(L: Landing, out: Path) -> Tuple[str, str]:
    """A5: accepted, rejected, ungrounded, abstained, unscored and failed
    shares per strategy, at no load and under load, per engine."""
    cells = L.cells[L.cells["dataset"].isin(["qasper", "squad_v2"])]
    counts: Dict[Tuple[str, Tuple[str, Optional[int]], str], Dict[str, int]] = {}
    for row in cells.itertuples(index=False):
        if row.engine in NO_STREAM_ENGINES:
            continue
        skey = _strategy_key(str(row.baseline), str(row.row_key))
        load = "no load" if row.family == "F1" else "under load"
        d = L.rows(row.window_dir)
        if d.empty:
            continue
        rec = counts.setdefault((str(row.engine), skey, load), {k: 0 for k in OUTCOME_ORDER})
        if "grounded" in d.columns and d["grounded"].notna().any():
            L.grounding_seen = True
        for r in d.itertuples(index=False):
            rec[_outcome(r)] += 1
    if not counts:
        raise PlotSkip("no request rows on qasper or squad_v2")
    grounding_ran = any(v["ungrounded"] for v in counts.values()) or L.grounding_seen
    cats = tuple(c for c in OUTCOME_ORDER if c != "ungrounded" or grounding_ran)
    engines = _engines_in({k[0] for k in counts})
    keys = sorted({k[1] for k in counts}, key=lambda s: (_bnum(s[0]), s[1] or 0))
    fig, axes = plt.subplots(len(engines), 1, figsize=(ps.FULL_WIDTH_IN, 2.6 * len(engines)), sharex=True, squeeze=False, layout="constrained")
    for ax, engine in zip(axes[:, 0], engines):
        for j, load in enumerate(("no load", "under load")):
            xs = np.arange(len(keys)) + (-0.2 if j == 0 else 0.2)
            bottom = np.zeros(len(keys))
            for cat in cats:
                vals = np.array([
                    counts[(engine, k, load)][cat] / max(sum(counts[(engine, k, load)].values()), 1) if (engine, k, load) in counts else 0.0
                    for k in keys
                ])
                ax.bar(xs, vals, 0.38, bottom=bottom, color=OUTCOME_COLOR[cat], edgecolor="white", lw=0.3,
                       hatch="//" if load == "under load" else None, label=cat if (j == 0 and engine == engines[0]) else None)
                bottom += vals
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(f"{ENGINE_LABEL[engine]}\nshare of issued requests")
        ax.set_xticks(np.arange(len(keys)), [_skey_label(k) for k in keys], rotation=30, ha="right", fontsize=6.5)
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="outside lower center", ncol=6, fontsize=7)
    fig.suptitle("What happens to each request: outcome shares per strategy (left bar: no load; hatched: under load; qasper and squad_v2)")
    stem = "a05_outcome_breakdown"
    note = "" if grounding_ran else "\nThe grounding instrument did not run on this tree (grounded is null on every row), so no request is classed ungrounded."
    _write_caption(out, stem, """
A5. Every issued request of qasper and squad_v2 is placed in one class: accepted (section 8.5 predicate true); abstained
(predicate false and the model declined to answer, predicted_no_answer); ungrounded (predicate false and the grounding
instrument found no support, grounded false); rejected (predicate false for another reason); unscored (completed but no
verdict); failed (not completed). The left bar of each strategy pools its no-load (F1) windows, the hatched bar its
pressure (F2/F3) windows. The reference engine is left out (whole-call latency, no first token).
""" + note)
    return _out(fig, out, stem), f"{sum(sum(v.values()) for v in counts.values())} requests"


def plot_a05b_retrieval_quality(L: Landing, out: Path) -> Tuple[str, str]:
    """A5b: retrieval hit share and median gold rank per RAG strategy (no load)."""
    d = _f1_rows(L)
    d = d[d["cell_baseline"].isin(["B5", "B6", "B7", "B9", "B11"]) & (d["cell_engine"] != "hf")]
    if "retrieval_hit" not in d.columns or "retrieval_rank" not in d.columns:
        raise PlotSkip("no no-load RAG row carries retrieval_hit and retrieval_rank")
    d["hit"] = d["retrieval_hit"].map(lambda v: None if v is None or (isinstance(v, float) and math.isnan(v)) else bool(v))
    d = d[d["hit"].notna()]
    if d.empty:
        raise PlotSkip("no no-load RAG row carries retrieval_hit")
    datasets = [x for x in ("qasper", "squad_v2") if x in set(d["cell_dataset"])]
    engines = _engines_in(d["cell_engine"])
    col = ps.categorical_colors(engines, "a05b")
    fig, axes = plt.subplots(1, len(datasets), figsize=(ps.FULL_WIDTH_IN, 2.8), sharey=True, squeeze=False, layout="constrained")
    for ax, ds in zip(axes[0], datasets):
        f = d[d["cell_dataset"] == ds]
        bl = sorted(f["cell_baseline"].unique(), key=_bnum)
        w = 0.8 / len(engines)
        for k, e in enumerate(engines):
            for i, b in enumerate(bl):
                s = f[(f["cell_engine"] == e) & (f["cell_baseline"] == b)]
                if s.empty:
                    continue
                x = i + (k - (len(engines) - 1) / 2) * w
                hit = s["hit"].astype(bool).mean()
                lo, hi = wilson(s["hit"].astype(bool).sum(), len(s))
                ax.bar(x, hit, w, color=col[e], label=ENGINE_LABEL[e] if i == 0 else None)
                ax.errorbar(x, hit, yerr=[[hit - lo], [hi - hit]], color="black", lw=0.7, capsize=1.5)
                rank = pd.to_numeric(s["retrieval_rank"], errors="coerce").dropna()
                if not rank.empty:
                    ax.annotate(f"rank {rank.median():.0f}", (x, hit + 0.03), ha="center", fontsize=5.5, rotation=90, va="bottom")
        ax.set_xticks(range(len(bl)), [blabel(b) for b in bl], rotation=30, ha="right", fontsize=6.5)
        ax.set_ylim(0, 1.15)
        ax.set_title(ds)
    axes[0][0].set_ylabel("gold passage retrieved (share)")
    axes[0][0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Did the evidence reach the prompt? Retrieval hit share per RAG strategy, no load (text: median rank of the gold passage)")
    stem = "a05b_retrieval_quality"
    _write_caption(out, stem, """
A5b. Share of no-load (F1) requests whose retrieved passages contain the gold evidence (retrieval_hit), per RAG strategy
and engine, with a 95 percent Wilson interval; the text above each bar is the median rank of the gold passage among the
retrieved ones (retrieval_rank). Retrieval runs before the serving clock starts, so this share is the ceiling any RAG
strategy can reach on grounding.
""")
    return _out(fig, out, stem), f"{len(d)} no-load RAG requests"


def _k1_cells(L: Landing) -> List[Dict[str, Any]]:
    """The K1 cells of the frontier legs: (engine, baseline, side, row keys,
    dataset); a control shared by two legs is listed once."""
    legs = _frontier_legs(L)
    out: List[Dict[str, Any]] = []
    seen_controls: set = set()
    for leg in legs:
        ctl = (leg["engine"], leg["baseline"], leg["dataset"], tuple(leg["control_row_keys"]))
        if ctl not in seen_controls:
            seen_controls.add(ctl)
            out.append({"engine": leg["engine"], "baseline": leg["baseline"], "budget_r": 1.25,
                        "dataset": leg["dataset"], "side": f"{CONTROL_LABEL} at {float(leg['rate_qps']):.2f} rps",
                        "keys": list(leg["control_row_keys"])})
        out.append({"engine": leg["engine"], "baseline": leg["baseline"], "budget_r": float(leg["budget_r"]),
                    "dataset": leg["dataset"], "side": f"{CONSTRAINED_LABEL} r={float(leg['budget_r']):g}",
                    "keys": [leg["constrained_row_key"]]})
    return out


def _telemetry(L: Landing, window_dir: str) -> pd.DataFrame:
    rows = _read_jsonl(L.run_root / window_dir / "cage_stats.jsonl")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["t"] = pd.to_numeric(df["ts_s"] if "ts_s" in df.columns else df.get("ts"), errors="coerce")
    usage_col = "kv_cache_usage" if "kv_cache_usage" in df.columns else "kv_usage"
    df["kv"] = pd.to_numeric(df.get(usage_col), errors="coerce")
    df["pre"] = pd.to_numeric(df.get("preemptions_total"), errors="coerce")
    return df.dropna(subset=["t"]).sort_values("t")


def plot_a06_preemption_kv(L: Landing, out: Path) -> Tuple[str, str]:
    """A6: KV occupancy over the window and preemptions per window for the
    K1 cells, control against constrained, per engine."""
    cells = _k1_cells(L)
    engines = _engines_in({c["engine"] for c in cells})
    fig, axes = plt.subplots(2, len(engines), figsize=(ps.FULL_WIDTH_IN, 5.6), squeeze=False, layout="constrained")
    baselines = sorted({c["baseline"] for c in cells}, key=_bnum)
    colors = ps.categorical_colors(baselines, "a06")
    n_windows = 0
    for j, engine in enumerate(engines):
        ax_t, ax_p = axes[0][j], axes[1][j]
        x = 0
        ticks: List[int] = []
        labels: List[str] = []
        for c in sorted((c for c in cells if c["engine"] == engine), key=lambda c: (_bnum(c["baseline"]), c["side"] != CONTROL_LABEL, -c["budget_r"])):
            windows = L.cells[L.cells["row_key"].isin(c["keys"]) & (L.cells["dataset"] == c["dataset"])]
            pre_per_window: List[float] = []
            counter_seen = False
            drawn_trace = False
            for w in windows.itertuples(index=False):
                tel = _telemetry(L, w.window_dir)
                if tel.empty:
                    continue
                regime = L.window_json(w.window_dir, "regime.json")
                t0 = regime.get("t_start", tel["t"].iloc[0])
                t1 = regime.get("t_end", tel["t"].iloc[-1])
                inside = tel[(tel["t"] >= t0) & (tel["t"] < t1)]
                if inside.empty:
                    inside = tel
                n_windows += 1
                if inside["pre"].notna().any():
                    pre_per_window.append(float(inside["pre"].max() - inside["pre"].min()))
                    counter_seen = counter_seen or bool((inside["pre"] != 0).any())
                if not drawn_trace and inside["kv"].notna().any():
                    ls = "-" if c["side"].startswith(CONTROL_LABEL) else "--"
                    ax_t.plot(inside["t"] - float(t0), inside["kv"] * (100 if inside["kv"].max() <= 1.0 else 1), ls,
                              color=colors[c["baseline"]], lw=0.9, label=f"{blabel(c['baseline'])}, {c['side']}")
                    drawn_trace = True
            if pre_per_window and counter_seen:
                ax_p.scatter(x + np.linspace(-0.15, 0.15, len(pre_per_window)), pre_per_window, s=12, color=colors[c["baseline"]],
                             marker="o" if c["side"].startswith(CONTROL_LABEL) else "^")
                ax_p.hlines(float(np.mean(pre_per_window)), x - 0.3, x + 0.3, color=colors[c["baseline"]], lw=1.2)
            else:
                # a counter that is absent or reads zero on every sample of the
                # cell is not a measured zero (review MEDIUM 4)
                ax_p.annotate("no preemption\ncounter", (x, 0.5), ha="center", va="center", fontsize=5.5, color="0.4",
                              xycoords=("data", "axes fraction"))
            ticks.append(x)
            labels.append(f"{blabel(c['baseline'])}\n{c['side']}")
            x += 1
        ax_t.set_title(ENGINE_LABEL[engine])
        ax_t.set_xlabel("time in the window (s)")
        ax_t.set_ylim(0, 105)
        ax_p.set_xticks(ticks, labels, rotation=35, ha="right", fontsize=5.5)
        ax_p.set_xlabel("K1 cell")
    handles = {lab: h for ax in axes[0] for h, lab in zip(*ax.get_legend_handles_labels())}
    fig.legend(handles.values(), handles.keys(), loc="outside lower center", ncol=2, fontsize=6)
    axes[0][0].set_ylabel("KV cache usage (%)\none window")
    axes[1][0].set_ylabel("preemptions per window\n(dots) and mean")
    fig.suptitle("Cache occupancy within the window and preemptions per window, control against constrained (K1 cells)")
    stem = "a06_preemption_kv"
    _write_caption(out, stem, """
A6. Top: the engine's KV cache usage gauge (cage_stats.jsonl, about 0.4 s samples) over the regime window of one window
per K1 cell, solid for the control rung 1.25 and dashed for the constrained rung at the same offered rate. Bottom: the
number of preemptions the engine recorded inside each window (the increase of the cumulative preemptions_total counter
over the regime window), one dot per window, the bar at the mean; a cell whose counter is absent or zero on every sample
is marked "no preemption counter" rather than drawn as zeros. The figure shows occupancy and preemptions side by side;
it does not by itself establish that preemptions carry the timely-share drop.
""")
    return _out(fig, out, stem), f"{n_windows} windows"


def plot_a07_consort_flow(L: Landing, out: Path) -> Tuple[str, str]:
    """A7: the CONSORT flow from served pressure windows to in-regime windows
    with a yield record; the pressure map (p03) is its companion."""
    c = L.cells[(L.cells["family"] != "F1") & (L.cells["dataset"] != "ruler")].copy()
    if c.empty:
        raise PlotSkip("no pressured window in the cell index")
    c["label"] = [L.window_json(w, "regime.json").get("label") for w in c["window_dir"]]
    n_served = len(c)
    with_record = c[c["label"].notna()]
    in_regime = with_record[with_record["label"] == IN_REGIME]
    y = L.yield_frame()
    y_ok = y[(y["skip_reason"].fillna("") == "") & y["n_issued"].gt(0)]
    key = set(zip(y_ok["row_key"], y_ok["window_dir"]))
    mask = pd.Series([(rk, wn) in key for rk, wn in zip(in_regime["row_key"], in_regime["window_name"])],
                     index=in_regime.index, dtype=bool)
    analyzed = in_regime[mask]
    other_labels = with_record[with_record["label"] != IN_REGIME]["label"].map(lambda k: READABLE.get(k, k)).value_counts()
    stages = []
    if L.plan is not None:
        planned = sum(int(s.get("windows") or 0) for s in L.plan_doc().get("steps", [])
                      if s.get("kind") == "cell" and (s.get("cellspec") or {}).get("budget_r"))
        if planned >= n_served:
            stages.append(fp.ConsortStage("Pressure windows planned (plan_registered.json cell steps)", planned, planned - n_served,
                                          "not served (never reached the pod or outside the sealed tree)" if planned > n_served else ""))
    stages += [
        fp.ConsortStage("Pressure windows served (F2 and F3, cell index)", n_served, n_served - len(with_record),
                        "no regime record (regime.json absent)" if n_served - len(with_record) else ""),
        fp.ConsortStage("Windows with a section 6.1 regime label", len(with_record), len(with_record) - len(in_regime),
                        "not in regime: " + ", ".join(f"{k} {v}" for k, v in other_labels.items())),
        fp.ConsortStage("Windows in regime (the registered E2 population)", len(in_regime), len(in_regime) - len(analyzed),
                        "no section 6.1 yield record (named skip)" if len(in_regime) - len(analyzed) else ""),
        fp.ConsortStage("In-regime windows with a yield record", len(analyzed)),
    ]
    stem = "a07_consort_flow"
    path = fp.plot_consort_flow(stages, out / f"{stem}.png", title="Pressure attainment: from served windows to the registered in-regime population")
    fig_pdf = plt.figure()
    plt.close(fig_pdf)
    _write_caption(out, stem, f"""
A7. CONSORT-style flow of the pressure windows: every F2 and F3 window the cell index lists, then those with a regime
record, then those the section 6.1 three-layer referee labeled in regime (queued occupancy at the measured ceiling,
attainment, a non-empty queue), then those with a section 6.1 yield record. The exclusion boxes name every reason and
the arithmetic closes stage by stage. The companion figure p03 (pressure map) shows which cache rung and load cells
reached which label. With --plan the first box is the planned pressure window count (the plan's cell steps). The last
box counts windows with a yield record graded as {L.yield_bar()}.
""")
    return path.name, f"{n_served} served, {len(in_regime)} in regime, {len(analyzed)} with a yield record"


def plot_a08_truth_tax_ratio(L: Landing, out: Path) -> Tuple[str, str]:
    """A8: the truth tax as the accepted share among timely requests (Y/G)
    against the timely share, per window, per engine and family."""
    y_all = L.yield_frame()
    y_all = y_all[(y_all["family"] != "F1") & (y_all["skip_reason"].fillna("") == "") & (y_all["dataset"] != "ruler")].copy()
    n_all_late = int(y_all["n_timely"].eq(0).sum())
    y = y_all[y_all["n_timely"].gt(0)].copy()
    if y.empty:
        raise PlotSkip("no pressure window with a yield record and at least one timely request")
    bar = L.yield_bar()
    y["ratio"] = y["n_yield"] / y["n_timely"]
    y["g"] = y["n_timely"] / y["n_issued"]
    y["skey"] = [_strategy_key(str(b), str(k)) for b, k in zip(y["baseline"], y["row_key"])]
    engines = _engines_in(y["engine"])
    fams = [f for f in ("F2", "F3") if f in set(y["family"])]
    fig, axes = plt.subplots(len(fams), len(engines), figsize=(ps.FULL_WIDTH_IN, 2.8 * len(fams)), sharex=True, sharey=True, squeeze=False, layout="constrained")
    for i, fam in enumerate(fams):
        keys = sorted({k for k in y.loc[y["family"] == fam, "skey"]}, key=lambda s: (_bnum(s[0]), s[1] or 0))
        colors = ps.categorical_colors([_skey_label(k) for k in keys], f"a08 {fam}")
        for ax, engine in zip(axes[i], engines):
            f = y[(y["family"] == fam) & (y["engine"] == engine)]
            for k in keys:
                g = f[f["skey"] == k]
                if g.empty:
                    continue
                ax.scatter(g["g"], g["ratio"], s=16, color=colors[_skey_label(k)], label=_skey_label(k), edgecolor="black", linewidth=0.2)
            ax.set_title(f"{ENGINE_LABEL[engine]}, {'fresh prefill (F2)' if fam == 'F2' else 'reuse (F3)'}", fontsize=8.5)
            ax.set_xlim(0, 1.02)
            ax.set_ylim(0, 1.02)
            if i == len(fams) - 1:
                ax.set_xlabel("timely share G (first token within the registered bar)")
        axes[i][0].set_ylabel("accepted among timely, Y/G")
        axes[i][-1].legend(fontsize=5.5, loc="lower left", ncol=2)
    fig.suptitle(f"Truth tax as a ratio: of the requests that were on time, how many were accepted (per window; timely = {bar})", fontsize=8.5)
    stem = "a08_truth_tax_ratio"
    path = _out(fig, out, stem)
    # the F2 companion from the shared renderer: Y against G with the diagonal,
    # the all-late windows included (Y = G = 0 is a valid point)
    comp = y_all[["row_key", "goodput_frac", "yield_frac", "dataset"]].copy()
    comp["goodput_frac"] = pd.to_numeric(comp["goodput_frac"], errors="coerce")
    comp["yield_frac"] = pd.to_numeric(comp["yield_frac"], errors="coerce")
    comp = comp.dropna()
    try:
        fp.plot_truth_tax(comp, out / "a08b_truth_tax_scatter.png",
                          config=fp.TruthTaxConfig(title="Serving yield Y against goodput G per pressure window (distance below the diagonal = G minus Y)"))
    except fp.FigureDataError as exc:
        raise PlotSkip(f"a08b: {exc}") from exc
    _write_caption(out, stem, f"""
A8. Each dot is one pressure window. x: the timely share G over issued requests, timely = {bar} (the bar is read from the
yield table's own baseline_source). y: the accepted share among the timely requests, Y/G; one minus it is the truth tax
as a ratio, which, unlike G minus Y, does not shrink mechanically when few requests are timely. Windows with no timely
request have no ratio and are left out here: {n_all_late} of {len(y_all)}. The companion a08b draws Y against G with
the diagonal (the section 9.2 estimand variable G minus Y) through the shared renderer figure_pipeline.plot_truth_tax,
with those windows included at Y = G = 0.
""")
    return path, f"{len(y)} pressure windows drawn, {n_all_late} with no timely request left out of the ratio"


def plot_a09_forest_headline(L: Landing, out: Path) -> Tuple[str, str]:
    """A9: the headline contrast's registered forest per metric and engine,
    labeled with the full strategy names; nothing recomputed."""
    st = L.stats()
    entries = [e for e in st.get("contrasts", []) if int(e["contrast_id"]) == 4]
    if not entries:
        raise PlotSkip("stats.json has no contrast #4 rows (run the driver with --contrasts 4)")
    files: List[str] = []
    skipped: List[str] = []
    for metric in sorted({e["metric"] for e in entries}):
        if metric == "predicate":
            # a binary metric's paired median delta is 0 by construction; the
            # forest would draw every point on a zero axis (review LOW 15)
            skipped.append(f"{metric}: binary metric, the median paired delta is uninformative")
            continue
        by_engine: Dict[str, List[Any]] = {}
        for e in (e for e in entries if e["metric"] == metric):
            engine = str(fp.parse_row_key(e["cell_row_key"])["engine"])
            if engine in NO_STREAM_ENGINES:
                # the reference engine rides the gatekeeping sidebar, never the
                # headline (S0F-102, ADR-0178 amendment 4)
                skipped.append(f"{metric} on {engine}: reference engine sidebar, not a headline member")
                continue
            label = f"{L.baseline_label(e['cell_row_key'])} vs {L.baseline_label(e['reference_row_key'])}"
            for row in e["per_dataset"]:
                if "p_value" not in row:
                    continue
                by_engine.setdefault(engine, []).append(fp.ContrastStatRow(
                    cell_row_key=str(e["cell_row_key"]), reference_row_key=str(e["reference_row_key"]),
                    dataset=str(row["dataset"]), metric=str(metric), higher_is_better=bool(e["higher_is_better"]),
                    executed_alternative=str(e["executed_alternative"]), correction=str(e["correction"]),
                    p_value=float(row["p_value"]), n_pairs=int(row["n_pairs"]), n_dropped_nan=int(row["n_dropped_nan"]),
                    median_delta=float(row["median_delta"]), wins=int(row["wins"]), losses=int(row["losses"]), ties=int(row["ties"]),
                    p_corrected=row.get("p_holm_across_datasets"), n_nonzero=row.get("n_nonzero"),
                    ci_low=row.get("ci_low"), ci_high=row.get("ci_high"), contrast_label=label,
                ))
        for engine, rows in sorted(by_engine.items()):
            stem = f"a09_forest_{metric}_{engine}"
            fp.plot_forest_registered(rows, out / f"{stem}.png",
                                      title=f"Headline (#4) on {metric}, {ENGINE_LABEL.get(engine, engine)}: registered per-dataset contrast")
            files.append(f"{stem}.png")
    if not files:
        raise PlotSkip("no headline forest to draw: " + "; ".join(skipped))
    _write_caption(out, "a09_forest_headline", f"""
A9. The registered headline contrast (#4, RAG reranked against CAG shared corpus) as the driver computed it, one forest
per metric and server engine, read from {st.get('_path')} contrast rows (median paired delta, win/loss/tie counts, the
registered p; whiskers only where stats.json carries a registered interval). Strategies are labeled by their full names.
Not drawn: {'; '.join(skipped) if skipped else 'nothing'}.
""")
    return ";".join(files), f"{len(files)} forest(s); not drawn: {len(skipped)}"


def plot_a10_capacity_table(L: Landing, out: Path) -> Tuple[str, str]:
    """A10: the capacity table from the plan and the calibration artifacts:
    offered rate per class, rung and engine beside the estimated lambda*,
    the floors and the class target."""
    plan = L.plan_doc()
    targets = plan.get("class_first_token_targets_s") or {}
    steps = [s for s in plan.get("steps", []) if s.get("kind") == "cell" and (s.get("cellspec") or {}).get("budget_r")]
    if not steps:
        raise PlotSkip("the plan has no pressure cell steps")
    floors: Dict[str, Dict[str, float]] = {}
    for path in sorted(glob.glob(str(L.calibration_dir / "rungs_*_*.json"))):
        doc = _read_json(Path(path))
        engine = Path(path).stem.rsplit("_", 1)[-1]
        floor = (doc.get("ladder") or {}).get("floor") or {}
        floors[engine] = {"ttft_s": floor.get("ttft_s"), "tpot_s": floor.get("tpot_s")}
    rows: Dict[Tuple[str, str, int, float], Dict[str, Any]] = {}
    for s in steps:
        spec = s["cellspec"]
        cls = int((s.get("demand_class") or {}).get("seq_tokens") or 0)
        key = (str(spec["engine"]), str(s.get("baseline")), cls, float(spec["budget_r"]))
        # the plan labels every lambda* ESTIMATED; the step's own rate_basis
        # says whether it is the laddered anchor or the interpolation between
        # the anchors (review HIGH 2: the alpha key is present on anchors too)
        basis = str(s.get("rate_basis") or "unknown").split(" [")[0][:90]
        rec = rows.setdefault(key, {
            "engine": key[0], "strategy": strategy_name(key[1], corpus_budget_tokens=(spec.get("corpus_budget_tokens") if key[1] == "B12" else None)),
            "class_tokens": cls, "budget_r": key[3], "lambda_star_rps": s.get("lambda_star_rps"), "lambda_star_basis": basis,
            "class_target_s": targets.get(str(cls)), "floor_ttft_s": floors.get(key[0], {}).get("ttft_s"),
            "floor_tpot_s": floors.get(key[0], {}).get("tpot_s"), "offered": {},
        })
        rec["offered"][f"{float(spec['rate_frac']):g}"] = s.get("offered_rate_rps")
    table = pd.DataFrame([
        {**{k: v for k, v in r.items() if k != "offered"},
         "offered_rps_by_load": "; ".join(f"{f}: {_fmt3(v)}" for f, v in sorted(r["offered"].items(), key=lambda kv: float(kv[0])))}
        for r in sorted(rows.values(), key=lambda r: (r["engine"], r["class_tokens"], -r["budget_r"]))
    ])
    csv_path = out / "a10_capacity_table.csv"
    table.to_csv(csv_path, index=False)
    md = ["# Capacity table (A10): offered rate per class, rung and engine", "",
          "lambda* is the saturation rate the plan used (the plan labels every value ESTIMATED); the basis column is "
          "the plan step's own rate_basis text: the laddered anchor rung or the interpolation between the anchor "
          "classes (ADR-0156). The offered rate at each load fraction is what the "
          "window actually requested. The class target is the registered first-token bar; the floors are the engine's "
          "single-stream latencies.", "",
          "| engine | strategy | class (tokens) | cache rung r | lambda* (rps) | basis | class target (s) | floor TTFT (s) | floor TPOT (s) | offered rps by load fraction |",
          "|" + "---|" * 10]
    for r in table.to_dict("records"):
        md.append(f"| {r['engine']} | {r['strategy']} | {r['class_tokens']} | {r['budget_r']:g} | {_fmt3(r['lambda_star_rps'])} | {r['lambda_star_basis']} | "
                  f"{_fmt3(r['class_target_s'])} | {_fmt3(r['floor_ttft_s'])} | {_fmt3(r['floor_tpot_s'])} | {r['offered_rps_by_load']} |")
    md_path = out / "a10_capacity_table.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    _write_caption(out, "a10_capacity_table", """
A10. Capacity table built from plan_registered.json (one row per engine, strategy, served-token class and cache rung):
the plan's lambda* with the plan step's own basis text (the laddered anchor rung, or the log-log interpolation between
the gold and the smallest class per ADR-0156; every value is labeled ESTIMATED by the plan), the offered rate at each
load fraction, the registered class first-token target, and the engine's single-stream floors from the rungs artifact.
""")
    return md_path.name, f"{len(table)} rows"


def _fmt3(v: Any) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    return "" if math.isnan(f) else f"{f:.3f}"


PLOTS: Dict[str, Callable[[Landing, Path], Tuple[str, str]]] = {
    "01_load_curve": plot01_load_curve,
    "02_truth_tax": plot02_truth_tax,
    "03_pressure_map": plot03_pressure_map,
    "05_avg_vs_peak": plot05_avg_vs_peak,
    "10_ttft_per_arm": plot10_ttft_per_arm,
    "15_correct_rate": plot15_correct_rate,
    "17_cache_reuse": plot17_cache_reuse,
    "18_prompt_vs_answer": plot18_prompt_vs_answer,
    # the S1 figure block (2026-10-27)
    "a01_frontier_drop": plot_a01_frontier_drop,
    "a02_ttft_cdf": plot_a02_ttft_cdf,
    "a03_paired_identity": plot_a03_paired_identity,
    "a04_fingerprint_heatmap": plot_a04_fingerprint_heatmap,
    "a05_outcome_breakdown": plot_a05_outcome_breakdown,
    "a05b_retrieval_quality": plot_a05b_retrieval_quality,
    "a06_preemption_kv": plot_a06_preemption_kv,
    "a07_consort_flow": plot_a07_consort_flow,
    "a08_truth_tax_ratio": plot_a08_truth_tax_ratio,
    "a09_forest_headline": plot_a09_forest_headline,
    "a10_capacity_table": plot_a10_capacity_table,
}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1].strip())
    p.add_argument("landing", type=Path)
    p.add_argument("--run-root", type=Path, default=None)
    p.add_argument("--scoring-run-id", default=None)
    p.add_argument("--calibration-dir", type=Path, default=None)
    p.add_argument("--yield-table", type=Path, default=None, help="a yield_table.csv instead of recomputing it")
    p.add_argument("--plan", type=Path, default=None, help="plan_registered.json: the registered class targets (S0F-99; a01, a08, a10)")
    p.add_argument("--stats", type=Path, default=None, help="the driver's stats.json (default: the newest analysis stamp)")
    p.add_argument("--only", nargs="*", default=None, help="plot keys to draw (default: all)")
    a = p.parse_args(argv)
    L = Landing(a.landing, run_root=a.run_root, scoring_run_id=a.scoring_run_id,
                calibration_dir=a.calibration_dir, yield_table=a.yield_table, plan=a.plan, stats=a.stats)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = a.landing / "plots" / "academic" / stamp
    out.mkdir(parents=True, exist_ok=True)
    ps.apply_style()
    manifest: Dict[str, Any] = {"run_root": str(L.run_root), "scoring_run_id": L.scoring_run_id,
                                "yield_source": str(a.yield_table) if a.yield_table else "yield_table.evaluate_run",
                                "plan": str(a.plan) if a.plan else None, "stats": str(a.stats) if a.stats else "newest analysis stamp",
                                "plots": {}}
    for key, fn in PLOTS.items():
        if a.only and key not in a.only:
            continue
        try:
            path, used = fn(L, out)
            manifest["plots"][key] = {"file": path, "used": used}
        except PlotSkip as exc:
            manifest["plots"][key] = {"skipped": str(exc)}
            print(f"  SKIP {key}: {exc}")
    (out / "plots_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[academic_plots] {sum(1 for v in manifest['plots'].values() if 'file' in v)} figure(s) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
