#!/usr/bin/env python3
"""Order:     stage 4, after pull_run.sh (reads the pulled campaign tree); beside run_campaign_analysis.py, which it never feeds
Objective: Per-window telemetry panels (KV usage, running/waiting, preemptions, tokens/s, GPU utilization, power) plus a resource index (tokens/s per GPU, joules and dollars per completed request)
Cloud:     local

render_window_panels.py: the picture of a window under pressure.

The campaign records one telemetry sample per second inside every measured
window (``cage_stats.jsonl``, written by ``src.monitoring.vllm_telemetry``)
and the runner's window aggregates (``metrics.json``), but the analysis
driver renders only end-of-run figures. This script reads ONE pulled run
(``docs/RESULTS_LAYOUT.md`` §1) and writes, per window that has telemetry
samples, a stacked time-series figure; per cell a contact sheet of the
KV-usage traces; and one index CSV with the derived resource columns the
paper's §6.6(b) per-GPU basis and the $0 columns of the minimal plan ask for.
Windows without samples are indexed, never drawn.

Rules the figures obey: read-only on the run tree; every curve comes from
``cage_stats.jsonl`` (named in the figure title) and the title's averages
from ``metrics.json``; a series the sampler did not record leaves the panel
empty with the words "not recorded" (never a fabricated zero); one quantity
per axis (no dual axes); the pipeline's ``_plot_style`` conventions. The KV
series is read the way the regime referee reads it (``ts_s`` with the ``ts``
fallback, ``kv_cache_usage`` with the ``kv_usage`` fallback,
``src.analysis.regime_inputs`` / ``campaign_layout``), so a window the
referee could not certify shows the same gap here. A prefill/decode window
carries both roles' samples in one file, told apart by ``instance``; each
role is drawn as its own trace and the clock is judged per role.

Known data shapes (read 2026-10-05 on results/s0/a/s0-20260930): the S0
tree carries ``ts_s`` = 1.0 on every sample (S0F-15, fixed since) and an
empty GPU sub-record (``gpu.available`` false, S0F-30). Both are handled,
labeled in the index ``notes`` column, and never hidden.

Usage (on the Mac, inside the repo venv):
  python3 scripts/4_analysis/render_window_panels.py <run_root> [--out DIR]
      [--cell SUBSTR] [--window SUBSTR] [--price-per-hour USD] [--no-png]
Outputs under --out (default <run_root>/analysis/panels/<UTC stamp>/, always
fresh; an explicit --out must be absent or empty so one directory never
mixes two renders):
  window_<cell slug>__<window>.png, cell_<cell slug>.png, panels_index.csv
Exit 0 on success; 2 when <run_root> is not a layout-v2 run (no manifest.json
or no cells/), when --out exists and is not empty, or when --price-per-hour
is not positive; nothing is created before those checks.
Dollars per TRUE answer is deliberately absent: it needs the window's yield,
which only the analysis driver computes (``src.analysis.goodput``).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _plot_style import apply_style, save_fig  # noqa: E402

TS_FIELD = "ts_s"
TS_FALLBACK = "ts"
KV_FIELD = "kv_cache_usage"
KV_FALLBACK = "kv_usage"
ROLE_FIELD = "instance"
SINGLE_ROLE = "single"
INDEX_COLUMNS = (
    "run_id", "cell", "baseline", "engine", "dataset", "family", "window", "budget_r", "rate_frac", "rep",
    "regime_label", "telemetry_ok", "roles", "n_samples", "n_malformed_lines", "clock_mode", "duration_s",
    "n_requests", "n_completed", "n_error", "gen_tokens", "prompt_tokens", "gpu_count",
    "gen_tps", "prompt_tps", "gen_tps_per_gpu", "prompt_tps_per_gpu",
    "energy_j", "joules_per_completed", "avg_gpu_util_pct", "avg_power_w", "gpu_series",
    "usd_per_hour", "usd_window", "usd_per_completed", "panel", "notes",
)


def _num(value: Any) -> Optional[float]:
    """A finite float, or None. Booleans and strings are not numbers here."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _read_json(path: Path) -> dict:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return obj if isinstance(obj, dict) else {}


def _read_jsonl(path: Path) -> tuple[list[dict], int]:
    """Records of a JSONL file and the count of lines that were not JSON objects."""
    records: list[dict] = []
    malformed = 0
    if not path.is_file():
        return records, malformed
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                malformed += 1
                continue
            if isinstance(obj, dict):
                records.append(obj)
            else:
                malformed += 1
    return records, malformed


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-." else "_" for ch in text)


def _fmt(value: Any) -> str:
    """CSV cell: empty for None (absence stays absence), compact floats otherwise."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _na(value: Any) -> str:
    """Display form of a cell field: 'n/a' for None (an F1 window has no budget or rate)."""
    return "n/a" if value is None else str(value)


def _sample_ts(sample: dict) -> Optional[float]:
    return _num(sample.get(TS_FIELD, sample.get(TS_FALLBACK)))


def _sample_kv(sample: dict) -> Optional[float]:
    return _num(sample.get(KV_FIELD, sample.get(KV_FALLBACK)))


@dataclass
class WindowData:
    cell_dir: Path
    window: str
    cell: dict
    samples: list[dict]
    n_malformed: int
    requests: Optional[list[dict]]        # None when requests.jsonl is absent
    regime: dict
    metrics: dict
    notes: list[str] = field(default_factory=list)

    # ---- cell identity ---------------------------------------------------
    @property
    def spec(self) -> dict:
        spec = self.cell.get("cellspec")
        return spec if isinstance(spec, dict) else {}

    @property
    def window_record(self) -> dict:
        wins = self.cell.get("windows")
        if isinstance(wins, dict):
            rec = wins.get(self.window)
            return rec if isinstance(rec, dict) else {}
        return {}

    def dataset(self) -> str:
        rec = self.window_record
        if isinstance(rec.get("dataset"), str):
            return rec["dataset"]
        return self.window.rsplit("-", 1)[0]

    # ---- roles (prefill/decode windows interleave two samplers) ----------
    def roles(self) -> dict[str, list[dict]]:
        """Samples grouped by ``instance`` in first-seen order; absent = single."""
        out: dict[str, list[dict]] = {}
        for s in self.samples:
            role = s.get(ROLE_FIELD)
            out.setdefault(role if isinstance(role, str) and role else SINGLE_ROLE, []).append(s)
        return out

    # ---- clocks and spans --------------------------------------------------
    def clock_problem(self) -> Optional[str]:
        """None when every role's timestamps are present and strictly increasing."""
        missing = sum(1 for s in self.samples if _sample_ts(s) is None)
        if missing:
            return f"timestamps missing on {missing} of {len(self.samples)} samples"
        for role, group in self.roles().items():
            ts = [_sample_ts(s) for s in group]
            if any(b <= a for a, b in zip(ts, ts[1:])):  # type: ignore[operator]
                tag = "" if role == SINGLE_ROLE else f" ({role})"
                return f"{TS_FIELD} not strictly increasing{tag}"
        return None

    def clock_mode(self) -> str:
        return "wall" if self.samples and self.clock_problem() is None else "index"

    def span(self) -> tuple[Optional[float], Optional[float]]:
        for src in (self.metrics.get("measured_window"), self.regime, self.window_record):
            if isinstance(src, dict):
                a, b = _num(src.get("t_start")), _num(src.get("t_end"))
                if a is not None and b is not None and b > a:
                    return a, b
        return None, None

    def duration_s(self) -> Optional[float]:
        a, b = self.span()
        return None if a is None else b - a

    def x_axis(self, group: list[dict]) -> tuple[list[float], str]:
        if self.clock_mode() == "wall":
            t0, _ = self.span()
            ts = [_sample_ts(s) for s in group]
            base = t0 if t0 is not None else ts[0]
            return [t - base for t in ts], "seconds from window start"  # type: ignore[operator]
        return [float(i) for i in range(len(group))], "sample index (clock defective)"

    # ---- series ----------------------------------------------------------
    @staticmethod
    def series(group: list[dict], key: str) -> list[Optional[float]]:
        return [_num(s.get(key)) for s in group]

    @staticmethod
    def kv_percent(group: list[dict]) -> tuple[list[Optional[float]], bool]:
        """KV usage as percent of the pool; the contract is a fraction in [0, 1].
        Returns the trace and whether any value broke the contract (> 1.0)."""
        vals = [_sample_kv(s) for s in group]
        out_of_contract = any(v is not None and v > 1.0 for v in vals)
        return [None if v is None else v * 100.0 for v in vals], out_of_contract

    @staticmethod
    def gpu_series(group: list[dict]) -> tuple[dict[int, list[Optional[float]]], dict[int, list[Optional[float]]], bool]:
        """Per-GPU utilization (%) and power (W) traces from the sampler's
        ``gpu.gpus[]`` sub-records; recorded=False when no sample carried one.
        A device without ``index`` is keyed by its position in the list."""
        util: dict[int, list[Optional[float]]] = {}
        power: dict[int, list[Optional[float]]] = {}
        recorded = False
        for i, s in enumerate(group):
            gpu = s.get("gpu")
            devices = gpu.get("gpus") if isinstance(gpu, dict) and gpu.get("available") else None
            if not isinstance(devices, list):
                continue
            for pos, dev in enumerate(devices):
                if not isinstance(dev, dict):
                    continue
                idx = dev.get("index")
                idx = idx if isinstance(idx, int) and not isinstance(idx, bool) else pos
                u, p = _num(dev.get("util_gpu")), _num(dev.get("power_w"))
                if u is not None or p is not None:
                    recorded = True
                util.setdefault(idx, [None] * len(group))[i] = u
                power.setdefault(idx, [None] * len(group))[i] = p
        return util, power, recorded

    def gpu_recorded(self) -> bool:
        return any(self.gpu_series(g)[2] for g in self.roles().values())

    # ---- request accounting ------------------------------------------------
    def request_counts(self) -> tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]:
        """(n_requests, n_completed, n_error, gen_tokens, prompt_tokens); all None
        when requests.jsonl is absent. Completed = ``ok`` True and no ``error``,
        the goodput module's own rule."""
        if self.requests is None:
            return None, None, None, None, None
        n = len(self.requests)
        completed = [r for r in self.requests if r.get("ok") is True and not r.get("error")]
        gen = sum(int(_num(r.get("num_tokens")) or 0) for r in completed)
        prompt = sum(int(_num(r.get("prompt_tokens")) or 0) for r in completed)
        return n, len(completed), n - len(completed), gen, prompt

    def energy_j(self) -> Optional[float]:
        """The runner's window delta first; else last minus first of ONE role's
        cumulative ``energy_mj`` (every role's sampler reads the same NVML
        counters, so the roles are never summed)."""
        tele = self.metrics.get("vllm_telemetry")
        delta = _num(tele.get("energy_delta_mj")) if isinstance(tele, dict) else None
        if delta is not None:
            return delta / 1000.0
        roles = self.roles()
        if not roles:
            return None
        first = roles[sorted(roles)[0]]
        vals = [v for v in self.series(first, "energy_mj") if v is not None]
        if len(vals) >= 2 and vals[-1] >= vals[0]:
            return (vals[-1] - vals[0]) / 1000.0
        return None


def _load_window(wdir: Path, cell: dict) -> WindowData:
    samples, malformed = _read_jsonl(wdir / "cage_stats.jsonl")
    requests_path = wdir / "requests.jsonl"
    requests = _read_jsonl(requests_path)[0] if requests_path.is_file() else None
    wd = WindowData(
        cell_dir=wdir.parent, window=wdir.name[len("window_"):], cell=cell, samples=samples,
        n_malformed=malformed, requests=requests, regime=_read_json(wdir / "regime.json"),
        metrics=_read_json(wdir / "metrics.json"),
    )
    if not (wdir / "cage_stats.jsonl").is_file():
        wd.notes.append("no telemetry file (cage_stats.jsonl absent)")
    elif not samples:
        wd.notes.append("no telemetry samples")
    if malformed:
        wd.notes.append(f"{malformed} malformed JSONL line(s) skipped")
    if requests is None:
        wd.notes.append("no requests.jsonl: request columns left empty")
    if len(samples) == 1:
        wd.notes.append("single telemetry sample")
    problem = wd.clock_problem() if samples else None
    if problem:
        wd.notes.append(f"clock defective: {problem}, drawn by sample index")
    if samples and not wd.gpu_recorded():
        wd.notes.append("gpu series not recorded by the sampler (gpu.available false)")
    if samples and any(wd.kv_percent(g)[1] for g in wd.roles().values()):
        wd.notes.append(f"{KV_FIELD} above 1.0: out of the fraction contract, drawn as given x 100")
    if wd.duration_s() is None:
        a, b = (None, None)
        for src in (wd.metrics.get("measured_window"), wd.regime, wd.window_record):
            if isinstance(src, dict) and _num(src.get("t_start")) is not None:
                a, b = _num(src.get("t_start")), _num(src.get("t_end"))
                break
        wd.notes.append("zero-length span" if a is not None and b is not None and b <= a
                        else "duration unknown (no measured_window, regime or cell.json span)")
    return wd


def _plot(ax, xs, ys, label: str, **kw) -> bool:
    pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if not pts:
        return False
    ax.plot([p[0] for p in pts], [p[1] for p in pts], label=label, linewidth=1.2, **kw)
    return True


def _plot_roles(ax, wd: WindowData, key: str, base_label: str, color: str) -> bool:
    """One trace per role for a plain series; returns whether anything was drawn."""
    drawn = []
    for n, (role, group) in enumerate(wd.roles().items()):
        xs, _ = wd.x_axis(group)
        label = base_label if role == SINGLE_ROLE else f"{base_label} ({role})"
        drawn.append(_plot(ax, xs, wd.series(group, key), label, color=color, linestyle=["-", "--", ":"][n % 3]))
    return any(drawn)


def _not_recorded(ax, what: str) -> None:
    ax.text(0.5, 0.5, f"{what}: not recorded", ha="center", va="center", transform=ax.transAxes,
            fontsize=9, color="0.4")
    ax.set_yticks([])


def _floor_axis(ax, top_min: float, *, fixed_top: Optional[float] = None) -> None:
    """Counts, rates and percentages start at 0; a flat series keeps a readable range."""
    _, hi = ax.get_ylim()
    ax.set_ylim(0.0, fixed_top if fixed_top is not None else max(hi, top_min))


def _title(wd: WindowData) -> str:
    spec, rec = wd.spec, wd.window_record
    label = wd.regime.get("label") if isinstance(wd.regime.get("label"), str) else "no regime.json"
    gpu = wd.metrics.get("gpu") if isinstance(wd.metrics.get("gpu"), dict) else {}
    util, power = _num(gpu.get("avg_gpu_utilization")), _num(gpu.get("avg_power_watts"))
    line1 = (f"{wd.cell.get('baseline', '?')} {spec.get('engine', '?')} {wd.dataset()} {spec.get('family', '?')}"
             f"  |  window {wd.window}  |  r={_na(rec.get('budget_r', spec.get('budget_r')))}"
             f"  rate={_na(rec.get('rate_frac', spec.get('rate_frac')))} x lambda*  rep={_na(rec.get('rep'))}  |  {label}")
    avg = (f"curves: cage_stats.jsonl  |  metrics.json window averages: GPU util {util:.1f} %, power {power:.0f} W"
           if util is not None and power is not None else "curves: cage_stats.jsonl  |  metrics.json window averages: not available")
    return line1 + "\n" + avg


def _draw_window(wd: WindowData, out: Path) -> None:
    fig, axes = plt.subplots(6, 1, figsize=(7.2, 11.0), sharex=True)
    ax_kv, ax_q, ax_pre, ax_tps, ax_util, ax_pw = axes
    roles = wd.roles()
    xlabel = wd.x_axis(next(iter(roles.values())))[1]

    drawn = []
    for n, (role, group) in enumerate(roles.items()):
        xs, _ = wd.x_axis(group)
        label = KV_FIELD if role == SINGLE_ROLE else f"{KV_FIELD} ({role})"
        drawn.append(_plot(ax_kv, xs, wd.kv_percent(group)[0], label, color="C0", linestyle=["-", "--", ":"][n % 3]))
    if any(drawn):
        _floor_axis(ax_kv, 100.0, fixed_top=100.0)
        if len(roles) > 1:
            ax_kv.legend(loc="upper right")
    else:
        _not_recorded(ax_kv, "KV cache usage")
    ax_kv.set_ylabel("KV cache usage (%)")

    if _plot_roles(ax_q, wd, "running", "running", "C1") | _plot_roles(ax_q, wd, "waiting", "waiting", "C2"):
        _floor_axis(ax_q, 1.0)
        ax_q.legend(loc="upper right")
    else:
        _not_recorded(ax_q, "requests running / waiting")
    ax_q.set_ylabel("requests")

    if _plot_roles(ax_pre, wd, "preempt_rate", "preempt_rate", "C3"):
        _floor_axis(ax_pre, 1.0)
        if len(roles) > 1:
            ax_pre.legend(loc="upper right")
    else:
        _not_recorded(ax_pre, "preemption rate")
    ax_pre.set_ylabel("preemptions / s")

    if _plot_roles(ax_tps, wd, "gen_tps", "generation", "C4") | _plot_roles(ax_tps, wd, "prompt_tps", "prompt", "C5"):
        _floor_axis(ax_tps, 1.0)
        ax_tps.legend(loc="upper right")
    else:
        _not_recorded(ax_tps, "tokens per second")
    ax_tps.set_ylabel("tokens / s")

    # GPU traces: every role's sampler reads the same devices, so draw the first
    # role that recorded them and nothing twice.
    util: dict[int, list[Optional[float]]] = {}
    power: dict[int, list[Optional[float]]] = {}
    xs_gpu: list[float] = []
    for group in roles.values():
        u, p, recorded = wd.gpu_series(group)
        if recorded:
            util, power, xs_gpu = u, p, wd.x_axis(group)[0]
            break
    drawn = [_plot(ax_util, xs_gpu, trace, f"GPU {i}") for i, trace in sorted(util.items())]
    if any(drawn):
        ax_util.set_ylim(0, 105)
        if len(util) > 1:
            ax_util.legend(loc="upper right")
    else:
        _not_recorded(ax_util, "GPU utilization (sampler gpu sub-record)")
    ax_util.set_ylabel("GPU util (%)")
    drawn = [_plot(ax_pw, xs_gpu, trace, f"GPU {i}") for i, trace in sorted(power.items())]
    if any(drawn):
        _floor_axis(ax_pw, 1.0)
        if len(power) > 1:
            ax_pw.legend(loc="upper right")
    else:
        _not_recorded(ax_pw, "GPU power (sampler gpu sub-record)")
    ax_pw.set_ylabel("power (W)")
    ax_pw.set_xlabel(xlabel)

    duration = wd.duration_s()
    if wd.clock_mode() == "wall" and duration is not None:
        for ax in axes:
            ax.axvline(0.0, color="0.3", linestyle="--", linewidth=0.8)
            ax.axvline(duration, color="0.3", linestyle="--", linewidth=0.8)
    label = wd.regime.get("label")
    fig.suptitle(_title(wd), fontsize=9, color=("firebrick" if label == "UNKNOWN_TELEMETRY" else "black"))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_fig(fig, out)


def _draw_contact_sheet(windows: list[WindowData], out: Path) -> None:
    n = len(windows)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.4 * rows), squeeze=False)
    for ax, wd in zip(axes.flat, windows):
        drawn = []
        xlabel = ""
        for k, (role, group) in enumerate(wd.roles().items()):
            xs, xlabel = wd.x_axis(group)
            drawn.append(_plot(ax, xs, wd.kv_percent(group)[0], role, color="C0", linestyle=["-", "--", ":"][k % 3]))
        if any(drawn):
            _floor_axis(ax, 100.0, fixed_top=100.0)
        else:
            _not_recorded(ax, "KV")
        rec = wd.window_record
        label = wd.regime.get("label", "?")
        ax.set_title(f"{wd.window}  r={_na(rec.get('budget_r'))}  f={_na(rec.get('rate_frac'))}  {label}", fontsize=7.5,
                     color=("firebrick" if label == "UNKNOWN_TELEMETRY" else "black"))
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_ylabel("KV %", fontsize=7)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    head = windows[0]
    fig.suptitle(f"{head.cell.get('baseline', '?')} {head.spec.get('engine', '?')} {head.spec.get('family', '?')}: "
                 f"KV cache usage per window, cage_stats.jsonl ({head.cell_dir.name})", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, out)


def _index_row(wd: WindowData, run_id: str, price: Optional[float], panel: str) -> dict:
    spec, rec = wd.spec, wd.window_record
    n, n_ok, n_err, gen, prompt = wd.request_counts()
    duration = wd.duration_s()
    gpu_count = wd.cell.get("gpu_count")
    gpu_count = gpu_count if isinstance(gpu_count, int) and not isinstance(gpu_count, bool) and gpu_count >= 1 else None
    gen_tps = gen / duration if gen is not None and duration else None
    prompt_tps = prompt / duration if prompt is not None and duration else None
    energy = wd.energy_j()
    gpu_avg = wd.metrics.get("gpu") if isinstance(wd.metrics.get("gpu"), dict) else {}
    usd_window = price * duration / 3600.0 if price is not None and duration else None
    label = wd.regime.get("label")
    tele_ok = wd.regime.get("telemetry_ok")
    return {
        "run_id": run_id, "cell": wd.cell_dir.name, "baseline": wd.cell.get("baseline"),
        "engine": spec.get("engine"), "dataset": wd.dataset(), "family": spec.get("family"), "window": wd.window,
        "budget_r": rec.get("budget_r", spec.get("budget_r")), "rate_frac": rec.get("rate_frac", spec.get("rate_frac")),
        "rep": rec.get("rep"), "regime_label": label if isinstance(label, str) else None,
        "telemetry_ok": tele_ok if isinstance(tele_ok, bool) else None,
        "roles": "+".join(wd.roles()) if wd.samples else None,
        "n_samples": len(wd.samples), "n_malformed_lines": wd.n_malformed,
        "clock_mode": wd.clock_mode() if wd.samples else None, "duration_s": duration,
        "n_requests": n, "n_completed": n_ok, "n_error": n_err, "gen_tokens": gen, "prompt_tokens": prompt,
        "gpu_count": gpu_count, "gen_tps": gen_tps, "prompt_tps": prompt_tps,
        "gen_tps_per_gpu": gen_tps / gpu_count if gen_tps is not None and gpu_count else None,
        "prompt_tps_per_gpu": prompt_tps / gpu_count if prompt_tps is not None and gpu_count else None,
        "energy_j": energy, "joules_per_completed": energy / n_ok if energy is not None and n_ok else None,
        "avg_gpu_util_pct": _num(gpu_avg.get("avg_gpu_utilization")), "avg_power_w": _num(gpu_avg.get("avg_power_watts")),
        "gpu_series": ("recorded" if wd.gpu_recorded() else "not recorded") if wd.samples else None,
        "usd_per_hour": price, "usd_window": usd_window,
        "usd_per_completed": usd_window / n_ok if usd_window is not None and n_ok else None,
        "panel": panel, "notes": "; ".join(wd.notes),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_root", help="campaign run root: results/<campaign>/<session>/<run_id>")
    parser.add_argument("--out", help="output directory, absent or empty (default <run_root>/analysis/panels/<UTC stamp>/)")
    parser.add_argument("--cell", default="", help="only cells whose row key contains this text")
    parser.add_argument("--window", default="", help="only windows whose id contains this text")
    parser.add_argument("--price-per-hour", type=float, default=None,
                        help="pod price in USD per hour; without it the dollar columns stay empty")
    parser.add_argument("--no-png", action="store_true", help="write the index only")
    args = parser.parse_args(argv)

    root = Path(args.run_root).resolve()
    if not (root / "manifest.json").is_file() or not (root / "cells").is_dir():
        print(f"ERROR: {root} is not a layout-v2 campaign run (manifest.json and cells/ required)", file=sys.stderr)
        return 2
    if args.price_per_hour is not None and not args.price_per_hour > 0:
        print("ERROR: --price-per-hour must be a positive number", file=sys.stderr)
        return 2
    out = Path(args.out) if args.out else root / "analysis" / "panels" / time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    if out.exists() and any(out.iterdir()):
        print(f"ERROR: --out {out} exists and is not empty; one directory never holds two renders", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    apply_style()

    rows: list[dict] = []
    for cell_dir in sorted(p for p in (root / "cells").iterdir() if p.is_dir()):
        if args.cell and args.cell not in cell_dir.name:
            continue
        cell = _read_json(cell_dir / "cell.json")
        windows = [_load_window(w, cell) for w in sorted(cell_dir.glob("window_*")) if w.is_dir()
                   and (not args.window or args.window in w.name)]
        drawable: list[WindowData] = []
        for wd in windows:
            panel = ""
            if wd.samples and not args.no_png:
                panel = f"window_{_slug(cell_dir.name)}__{_slug(wd.window)}.png"
                _draw_window(wd, out / panel)
                drawable.append(wd)
            rows.append(_index_row(wd, root.name, args.price_per_hour, panel))
        if drawable and not args.no_png:
            _draw_contact_sheet(drawable, out / f"cell_{_slug(cell_dir.name)}.png")

    with open(out / "panels_index.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(INDEX_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k)) for k in INDEX_COLUMNS})
    print(f"[render_window_panels] {len(rows)} window(s) indexed, output {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
