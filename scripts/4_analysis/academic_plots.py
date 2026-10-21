#!/usr/bin/env python3
"""Order:     after stage 14 (analyze), on the pulled landing (owner plot list of 2026-10-21)
Objective: Draw the academic figure set into <landing>/plots/academic/<stamp>/, one figure or one named skip per plot
Cloud:     local

The owner's plot list of 2026-10-21 (MyDocs/RunPod/S1_ACADEMIC_PLOTS_PLAN_2026-10-21.md).
This file draws plots 1, 2, 3, 5, 10, 15, 17 and 18. Each plot reads the landing
only, refuses with a named reason when its data is missing (written to
plots_manifest.json, never a silent empty figure), and states which windows it used.

Sources: cells_index.csv (engine, baseline, dataset, family, budget_r, rate_frac),
requests.jsonl, the predicate tree (predicate/<scoring id>/cells/...), regime.json,
metrics.json, and the yield table (yield_table.evaluate_run: on time = timely,
on time and correct = yield; the same definitions the analysis uses).

Usage: python3 scripts/4_analysis/academic_plots.py <landing> [--run-root R]
       [--scoring-run-id ID] [--calibration-dir D] [--yield-table CSV]
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

from src.analysis.cellspec import BASELINES  # noqa: E402

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


def blabel(b: str) -> str:
    """'B3' -> 'B3 corpus-reuse' (B5 and B6 share an arm: B5 dense, B6 reranked)."""
    arm = BASELINES[b].arm if b in BASELINES else "?"
    extra = {"B5": " (dense)", "B6": " (rerank)"}.get(b, "")
    return f"{b} {arm}{extra}"


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
                 calibration_dir: Optional[Path] = None, yield_table: Optional[Path] = None) -> None:
        self.landing = landing
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
            for col in ("engine", "baseline", "dataset", "family", "budget_r", "rate_frac", "window_dir"):
                df[f"cell_{col}"] = c[col]
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

                y = pd.DataFrame(yt.evaluate_run(self.run_root, baselines=yt.load_baselines(self.calibration_dir),
                                                 scoring_run_id=self.scoring_run_id))
            y = y.merge(self.cells[["row_key", "window_name", "baseline", "budget_r", "rate_frac"]],
                        left_on=["row_key", "window_dir"], right_on=["row_key", "window_name"], how="left")
            for col in ("n_issued", "n_timely", "n_yield"):
                y[col] = pd.to_numeric(y[col], errors="coerce")
            self._yield = y
        return self._yield

    def window_json(self, window_dir: str, name: str) -> Dict[str, Any]:
        return _read_json(self.run_root / window_dir / name)


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
            ax.plot(g.index, g["n_timely"] / g["n_issued"], "-", color=colors[b], marker=markers[r], label=f"{b} r={r:g}, on time")
            ax.plot(g.index, g["n_yield"] / g["n_issued"], "--", color=colors[b], marker=markers[r], mfc="white", label=f"{b} r={r:g}, on time and correct")
        ax.set_title(ENGINE_LABEL[engine])
        ax.set_xlabel("offered load (fraction of the cell's saturation rate)")
        ax.set_ylim(0, 1.02)
    axes[0][0].set_ylabel("share of requests")
    n_loads = y["rate_frac"].nunique()
    fig.suptitle("Load curve: on time vs on time and correct (qasper, B1 and B6)" + (
        f"\n{n_loads} load level in this run: points, not curves" if n_loads < 2 else ""))
    axes[0][-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
    return _out(fig, out, "p01_load_curve"), f"{len(y)} windows, {n_loads} load level(s)"


def plot02_truth_tax(L: Landing, out: Path) -> Tuple[str, str]:
    """Under real pressure (IN_REGIME), on time vs on time and correct, per arm."""
    y = L.yield_frame()
    y = y[(y["regime_label"] == IN_REGIME) & (y["dataset"] == "qasper") & y["n_issued"].gt(0)]
    if y.empty:
        raise PlotSkip("no IN_REGIME qasper window with a yield record")
    engines = _engines_in(y["engine"])
    fig, axes = plt.subplots(len(engines), 1, figsize=(ps.FULL_WIDTH_IN, 2.4 * len(engines)), sharey=True, squeeze=False, layout="constrained")
    for ax, engine in zip(axes[:, 0], engines):
        g = y[y["engine"] == engine].groupby("baseline")[["n_issued", "n_timely", "n_yield"]].sum()
        g = g.loc[sorted(g.index, key=_bnum)]
        x = np.arange(len(g))
        timely, both = g["n_timely"] / g["n_issued"], g["n_yield"] / g["n_issued"]
        ax.bar(x - 0.2, timely, 0.4, label="on time", color="#0072B2")
        ax.bar(x + 0.2, both, 0.4, label="on time and correct", color="#E69F00")
        for xi, t, bth in zip(x, timely, both):
            if t - bth > 0.005:
                ax.text(xi, t + 0.02, f"-{(t - bth) * 100:.0f} pts", ha="center", fontsize=7)
        ax.set_xticks(x, [blabel(b) for b in g.index], rotation=30, ha="right", fontsize=7)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel(f"{ENGINE_LABEL[engine]}\nshare of requests")
    axes[0][0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Truth tax: answers on time but wrong, under real memory pressure (qasper, IN_REGIME windows)")
    return _out(fig, out, "p02_truth_tax"), f"{len(y)} IN_REGIME windows"


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
        col = ps.categorical_colors(sorted(f["baseline"].unique(), key=_bnum), f"plot05 {fam}")
        for (b, e), g in f.groupby(["baseline", "engine"]):
            ax.scatter(g["avg"] * 100, g["peak"] * 100, color=col[b], marker=mk[e], s=28, label=f"{b} {ENGINE_LABEL[e]}",
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
        g = d[d["cell_dataset"] == ds].groupby(["cell_baseline", "cell_engine"])["ttft_ms"]
        stats = pd.DataFrame({"p50": g.median(), "p95": g.quantile(0.95)}).reset_index()
        bl = sorted(stats["cell_baseline"].unique(), key=_bnum)
        w = 0.8 / len(engines)
        for k, e in enumerate(engines):
            s = stats[stats["cell_engine"] == e].set_index("cell_baseline").reindex(bl)
            x = np.arange(len(bl)) + (k - (len(engines) - 1) / 2) * w
            nostream = e in NO_STREAM_ENGINES
            ax.bar(x, s["p50"], w, color=col[e], hatch="///" if nostream else None, edgecolor="white" if nostream else None,
                   label=f"{ENGINE_LABEL[e]} (whole call: no streaming, no first-token time)" if nostream else ENGINE_LABEL[e])
            ax.scatter(x, s["p95"], marker="_", color="black", s=60, zorder=3)
        ax.set_yscale("log")
        ax.set_xticks(range(len(bl)), [blabel(b) for b in bl], rotation=30, ha="right", fontsize=7)
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
        bl = sorted(f["cell_baseline"].unique(), key=_bnum)
        w = 0.8 / len(engines)
        for k, e in enumerate(engines):
            x0 = np.arange(len(bl)) + (k - (len(engines) - 1) / 2) * w
            for x, b in zip(x0, bl):
                s = f[(f["cell_engine"] == e) & (f["cell_baseline"] == b)]["predicate"].astype(bool)
                if s.empty:
                    continue
                lo, hi = wilson(s.sum(), len(s))
                ax.bar(x, s.mean(), w, color=col[e], label=ENGINE_LABEL[e] if b == bl[0] else None)
                ax.errorbar(x, s.mean(), yerr=[[s.mean() - lo], [hi - s.mean()]], color="black", lw=0.7, capsize=1.5)
            ceiling = f[(f["cell_engine"] == e) & (f["cell_baseline"] == "B1")]["predicate"].astype(bool)
            if not ceiling.empty:
                ax.axhline(ceiling.mean(), color=col[e], ls="--", lw=0.8)
        ax.set_xticks(range(len(bl)), [blabel(b) for b in bl], rotation=30, ha="right", fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f"{ds}\nshare correct")
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="outside lower center", ncol=len(engines), fontsize=7)
    fig.suptitle("Correct-answer rate per arm, no pressure (dashed = B1 gold-fresh per engine; bars = 95% Wilson CI)")
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
                labels.append(f"{b}\n{ENGINE_LABEL[e]}")
                x += 1
        ax.set_xticks(range(len(labels)), labels, fontsize=6.5, rotation=45, ha="right")
        ax.set_title(ds)
    axes[0][0].set_ylabel("median time per request (s)")
    handles = {lab: h for ax in axes[0] for h, lab in zip(*ax.get_legend_handles_labels())}
    fig.legend(handles.values(), handles.keys(), loc="outside lower center", ncol=3, fontsize=7)
    fig.suptitle("Where the time goes: prompt reading vs answer writing (no pressure; B1 gold, B3 cache, B6 RAG)")
    return _out(fig, out, "p18_prompt_vs_answer"), f"{len(d)} ok requests"


PLOTS: Dict[str, Callable[[Landing, Path], Tuple[str, str]]] = {
    "01_load_curve": plot01_load_curve,
    "02_truth_tax": plot02_truth_tax,
    "03_pressure_map": plot03_pressure_map,
    "05_avg_vs_peak": plot05_avg_vs_peak,
    "10_ttft_per_arm": plot10_ttft_per_arm,
    "15_correct_rate": plot15_correct_rate,
    "17_cache_reuse": plot17_cache_reuse,
    "18_prompt_vs_answer": plot18_prompt_vs_answer,
}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1].strip())
    p.add_argument("landing", type=Path)
    p.add_argument("--run-root", type=Path, default=None)
    p.add_argument("--scoring-run-id", default=None)
    p.add_argument("--calibration-dir", type=Path, default=None)
    p.add_argument("--yield-table", type=Path, default=None, help="a yield_table.csv instead of recomputing it")
    p.add_argument("--only", nargs="*", default=None, help="plot keys to draw (default: all)")
    a = p.parse_args(argv)
    L = Landing(a.landing, run_root=a.run_root, scoring_run_id=a.scoring_run_id,
                calibration_dir=a.calibration_dir, yield_table=a.yield_table)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = a.landing / "plots" / "academic" / stamp
    out.mkdir(parents=True, exist_ok=True)
    ps.apply_style()
    manifest: Dict[str, Any] = {"run_root": str(L.run_root), "scoring_run_id": L.scoring_run_id,
                                "yield_source": str(a.yield_table) if a.yield_table else "yield_table.evaluate_run",
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
