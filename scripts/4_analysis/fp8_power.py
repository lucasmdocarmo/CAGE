#!/usr/bin/env python3
"""Order:     stage 13 (analyze) -- power for the fp8 KV-cache contrast (Spec Block A3, 2026-10-16)
Objective: from the per-query paired F1 differences of the fp8 KV-cache arm (B10 corpus-comp)
           against its bf16 twin (B3 corpus-reuse) on the SAME engine, compute the number of
           queries S1's fp8 cell needs to detect a mean F1 difference of 0.01, 0.02 and 0.05 at
           alpha 0.05 two-sided and power 0.80, and say whether the registered n ladder
           {2,000, 1,600, 1,200} covers it. Design input for the S1 registration only.
Cloud:     local
           (the Mac or the conductor; reads a pulled landing's scoring pass only)

Methods, each named in the output (section 9.6: power is simulation-based; the registered
per-query test is the paired Wilcoxon with Pratt zero handling, section 9.4):
1. registered_additive_shift: the §9.6 CLI's own method (run_power_sim.py: symmetrized centered
   real paired differences as the H0 pool, ``shift_injection``, ``paired_wilcoxon`` with
   zero_method="pratt"). It is run through the SAME honest-injection guard the CLI applies
   (run_calibration.MAX_SHIFT_COLLISION, the P0 2026-08-02 doctrine): an additive shift on a
   tie-heavy difference pool breaks every tie and inflates power, so a pool whose collision
   probability exceeds the threshold is REFUSED and the refusal is recorded as that method's
   answer. The S0 fp8 pools are 78 to 96 percent zeros [V] and are refused.
2. tie_preserving_shift (design input, labeled): the same H0 pool WITHOUT centering
   (concat(d, -d), mean exactly zero, the observed zero mass kept) and an injection that adds
   the target mean difference to the DISCORDANT pairs only, scaled so the pool mean equals the
   target: ties stay ties, the discordance structure is the observed one, and the test is the
   registered Pratt Wilcoxon. It answers "how many queries if fp8 changes the magnitude of the
   answers it already changes". READ WITH CARE (review 2026-10-16, HIGH 1 and 2): the shift per
   discordant pair is target / discordant_rate (0.06 to 1.25 on S0), so the requirement is a
   STEP function of the few observed discordant magnitudes (a sign flips when a magnitude falls
   under the shift), not a mean-difference power; the per-pair shift is a column; and when the
   shift would carry any observed difference outside the F1 range [-1, 1] the target is REFUSED
   for this method (status SHIFT_LEAVES_F1_RANGE) instead of simulating impossible differences.
   It does not model MORE answers changing. A size row (effect 0) is simulated beside the
   targets so the realized alpha is on record.
3. paired_t_approximation: n = ((z_{1-alpha/2} + z_{power}) * sd / delta)^2 on the observed
   standard deviation of the differences; closed form, labeled approximate, never presented as
   the Wilcoxon's exact requirement.

Pools: one per (engine, dataset, family, rung) window pair where both arms have a scored window
with identical identity triples (example_id, repeat_index, record_index); rows failing the
loader validity rule (error, empty generation, not ok) on either side are dropped and counted;
a pool under MIN_PAIRS is a named skip; mismatched triples are a named skip (never an
intersection); when no pool survives the pass refuses (exit 3). The F1 values are the scoring
pass's ``f1_score`` (scoring/<id>/cells/.../qa_scores.jsonl), the sealed quality record.

Exit codes: 0 outputs written; 3 named refusal; 2 is Python's and argparse's (usage error).

Usage:
    python scripts/4_analysis/fp8_power.py <run_root> --scoring-run-id full-cuda-1 --out <dir> \\
        [--cell-arm corpus-comp --ref-arm corpus-reuse] [--n-sims 10000] [--seed 42]
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
for _p in (str(_REPO), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _results_loader import is_empty_generation, is_error  # noqa: E402
from run_calibration import MAX_SHIFT_COLLISION, collision_probability  # noqa: E402
from src.analysis.cellspec import CellSpec, CellSpecError  # noqa: E402
from src.analysis.stats.power_sim import PowerSimError, required_n, shift_injection, simulate_campaign  # noqa: E402
from src.analysis.stats.tests_by_unit import paired_wilcoxon  # noqa: E402

DEFAULT_CELL_ARM = "corpus-comp"   # B10: corpus prefix served from an fp8 KV cache
DEFAULT_REF_ARM = "corpus-reuse"   # B3: the same corpus prefix in bf16
METRIC = "f1_score"
DEFAULT_TARGETS: Tuple[float, ...] = (0.01, 0.02, 0.05)
DEFAULT_ALPHA = 0.05
TARGET_POWER = 0.8
DEFAULT_N_SIMS = 10_000
DEFAULT_SEED = 42
#: The registered n ladder [V] stats.json:preconditions.realized_n.ladder (S0 attempt 5).
N_LADDER: Tuple[int, ...] = (2000, 1600, 1200)
DEFAULT_N_GRID: Tuple[int, ...] = (50, 100, 200, 300, 500, 800, 1200, 1600, 2000, 2400, 3200, 4800)
MIN_PAIRS = 50
REFUSED_RC = 3
NOT_REACHED = "NOT_REACHED_ON_GRID"
METHOD_REGISTERED = "registered_additive_shift"
METHOD_TIE_PRESERVING = "tie_preserving_shift"
METHOD_T = "paired_t_approximation"
STATUS_SIMULATED = "SIMULATED"
STATUS_REFUSED = "REFUSED_BY_INJECTION_GUARD"
STATUS_NO_DISCORDANT = "NO_DISCORDANT_PAIRS"
STATUS_NO_VARIANCE = "NO_VARIANCE"
STATUS_CLOSED_FORM = "CLOSED_FORM_APPROXIMATION"
STATUS_OUT_OF_RANGE = "SHIFT_LEAVES_F1_RANGE"
F1_RANGE = 1.0  # |F1 difference| cannot exceed 1
CSV_COLUMNS: Tuple[str, ...] = (
    "engine", "dataset", "family", "budget_r", "rate_frac", "window", "n_pairs", "n_dropped_invalid",
    "mean_diff", "sd_diff", "tie_mass", "discordant_rate", "collision_probability",
    "target", "method", "status", "required_n", "ladder_rung", "verdict", "per_discordant_shift", "realized_alpha_max", "note",
)


class Fp8PowerError(RuntimeError):
    """A refusal of the whole pass."""


def wilcoxon_pratt_p(diffs: np.ndarray) -> float:
    """The registered per-query path, identical to run_power_sim.wilcoxon_diff_p:
    paired_wilcoxon(diffs, 0) two-sided with zero_method="pratt" (ADR-0088)."""
    return float(paired_wilcoxon(diffs, np.zeros_like(diffs), alternative="two-sided", zero_method="pratt").p_value)


def tie_preserving_shift(rng: np.random.Generator, null_diffs: np.ndarray, effect: float) -> np.ndarray:
    """Add ``effect * n / n_discordant`` to the non-zero differences only, so the pool
    mean rises by exactly ``effect`` while every tie stays a tie. A draw without a
    discordant pair cannot carry an effect and is returned as drawn: the Pratt test on
    all zeros is the T=0 identity outcome (p = 1, no rejection), which is the honest
    power cost of a tie-heavy pool at a small n, not an error."""
    out = np.array(null_diffs, dtype=float, copy=True)
    nz = np.flatnonzero(out != 0.0)
    if effect == 0.0 or nz.size == 0:
        return out
    out[nz] += float(effect) * out.size / nz.size
    return out


def t_required_n(sd: float, delta: float, *, alpha: float, power: float) -> Optional[int]:
    """Paired t approximation; None when the differences have no variance."""
    if not math.isfinite(sd) or sd <= 0.0 or delta <= 0.0:
        return None
    z = norm.ppf(1.0 - alpha / 2.0) + norm.ppf(power)
    return int(math.ceil((z * sd / delta) ** 2))


def ladder_verdict(n: Optional[int]) -> Tuple[str, str]:
    """(smallest registered rung that covers n, verdict text)."""
    if n is None:
        return "", "no requirement (see status)"
    covering = [r for r in sorted(N_LADDER) if r >= n]
    if covering:
        return str(covering[0]), "within ladder"
    return "", f"exceeds ladder (max rung {max(N_LADDER)})"


@dataclass
class Pool:
    engine: str
    dataset: str
    family: str
    budget_r: Optional[float]
    rate_frac: Optional[float]
    window: str
    cell_window: str
    ref_window: str
    diffs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n_dropped_invalid: int = 0
    skip_reason: str = ""

    def diagnostics(self) -> Dict[str, Any]:
        d = self.diffs
        if d.size == 0:
            return {"n_pairs": 0}
        ties = float(np.mean(d == 0.0))
        return {
            "n_pairs": int(d.size),
            "mean_diff": float(d.mean()),
            "sd_diff": float(d.std(ddof=1)) if d.size > 1 else 0.0,
            "tie_mass": ties,
            "discordant_rate": 1.0 - ties,
            "collision_probability": collision_probability(d),
        }


def _row_valid(row: Dict[str, Any]) -> bool:
    """The canonical loader rule (_results_loader.is_error / is_empty_generation) plus the
    scoring pass's own not_ok flag."""
    return not is_error(row.get("error")) and not is_empty_generation(row.get("empty_generation")) and not row.get("not_ok")


def _load_scored(path: Path) -> Dict[Tuple[Any, Any, Any], Tuple[Optional[float], bool]]:
    out: Dict[Tuple[Any, Any, Any], Tuple[Optional[float], bool]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row.get("example_id"), row.get("repeat_index"), row.get("record_index"))
            if key in out:
                raise Fp8PowerError(f"{path}: duplicate identity triple {key!r}")
            val = row.get(METRIC)
            out[key] = (float(val) if isinstance(val, (int, float)) and not isinstance(val, bool) else None, _row_valid(row))
    return out


def collect_pools(run_root: Path, *, scoring_run_id: str, cell_arm: str, ref_arm: str) -> List[Pool]:
    """One Pool per scored window of the cell arm whose reference twin (same engine,
    model, grid point) has a scored window of the same name."""
    cells_dir = run_root / "cells"
    scoring_dir = run_root / "scoring" / scoring_run_id / "cells"
    if not cells_dir.is_dir():
        raise Fp8PowerError(f"no cells/ under {run_root}")
    if not scoring_dir.is_dir():
        raise Fp8PowerError(f"no scoring pass {scoring_run_id!r} under {run_root} (expected {scoring_dir})")
    pools: List[Pool] = []
    for cell_dir in sorted(p for p in cells_dir.iterdir() if p.is_dir() and (p / "cell.json").is_file()):
        with open(cell_dir / "cell.json", encoding="utf-8") as fh:
            meta = json.load(fh)
        try:
            spec = CellSpec.from_flat_dict(meta.get("cellspec") or {})
        except CellSpecError as exc:
            raise Fp8PowerError(f"{cell_dir.name}/cell.json: cellspec rejected: {exc}") from exc
        if spec.arm != cell_arm:
            continue
        ref_key = dataclasses.replace(spec, arm=ref_arm).to_row_key()  # type: ignore[arg-type]
        scored_cell = scoring_dir / cell_dir.name
        windows = sorted(p for p in scored_cell.glob("window_*") if p.is_dir()) if scored_cell.is_dir() else []
        for wdir in windows:
            dataset = wdir.name[len("window_"):].rsplit("-", 1)[0]
            pool = Pool(
                engine=str(spec.engine), dataset=dataset, family=str(spec.family), budget_r=spec.budget_r,
                rate_frac=spec.rate_frac, window=wdir.name,
                cell_window=f"{cell_dir.name}/{wdir.name}", ref_window=f"{ref_key}/{wdir.name}",
            )
            cell_scores = wdir / "qa_scores.jsonl"
            ref_scores = scoring_dir / ref_key / wdir.name / "qa_scores.jsonl"
            if not cell_scores.is_file():
                pool.skip_reason = f"cell arm window has no qa_scores.jsonl: {pool.cell_window}"
            elif not ref_scores.is_file():
                pool.skip_reason = f"reference arm {ref_arm!r} has no scored window beside it: {pool.ref_window}"
            else:
                a = _load_scored(cell_scores)
                b = _load_scored(ref_scores)
                if set(a) != set(b):
                    missing_a = sorted(set(b) - set(a), key=repr)[:3]
                    missing_b = sorted(set(a) - set(b), key=repr)[:3]
                    pool.skip_reason = (
                        f"mismatched identity triples, refusing to intersect: cell {len(a)} rows, reference {len(b)} rows; "
                        f"missing on cell e.g. {missing_a}; missing on reference e.g. {missing_b}"
                    )
                else:
                    diffs: List[float] = []
                    dropped = 0
                    for key in sorted(a, key=repr):
                        fa, va = a[key]
                        fb, vb = b[key]
                        if not (va and vb) or fa is None or fb is None:
                            dropped += 1
                            continue
                        diffs.append(fa - fb)
                    pool.diffs = np.asarray(diffs, dtype=float)
                    pool.n_dropped_invalid = dropped
                    if pool.diffs.size < MIN_PAIRS:
                        pool.skip_reason = f"only {pool.diffs.size} valid pairs (< {MIN_PAIRS}); {dropped} dropped as invalid"
            pools.append(pool)
    if not pools:
        raise Fp8PowerError(
            f"no scored window of arm {cell_arm!r} under {scoring_dir}: nothing to pair with {ref_arm!r}"
        )
    return pools


def _centered_symmetrized(diffs: np.ndarray) -> Callable[[np.random.Generator, int], np.ndarray]:
    """run_power_sim.make_paired_null_model(kind='shift'): concat(c, -c), c = d - mean(d)."""
    c = diffs - diffs.mean()
    pool = np.concatenate([c, -c])
    return lambda rng, n: rng.choice(pool, size=n, replace=True)


def _symmetrized(diffs: np.ndarray) -> Callable[[np.random.Generator, int], np.ndarray]:
    """concat(d, -d): mean exactly zero, the observed zero mass kept."""
    pool = np.concatenate([diffs, -diffs])
    return lambda rng, n: rng.choice(pool, size=n, replace=True)


def _required_from_table(table, target: float) -> Any:
    try:
        return int(required_n(table, float(target), target_power=TARGET_POWER))
    except PowerSimError:
        return NOT_REACHED


def analyze_pool(
    pool: Pool, *, targets: Sequence[float], n_grid: Sequence[int], n_sims: int, seed: int, alpha: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Rows (one per method x target) and the raw power tables for one pool."""
    diag = pool.diagnostics()
    base = {
        "engine": pool.engine, "dataset": pool.dataset, "family": pool.family, "budget_r": pool.budget_r,
        "rate_frac": pool.rate_frac, "window": pool.window, "n_pairs": diag["n_pairs"],
        "n_dropped_invalid": pool.n_dropped_invalid, "mean_diff": diag["mean_diff"], "sd_diff": diag["sd_diff"],
        "tie_mass": diag["tie_mass"], "discordant_rate": diag["discordant_rate"],
        "collision_probability": diag["collision_probability"],
    }
    rows: List[Dict[str, Any]] = []
    tables: Dict[str, Any] = {}
    d = pool.diffs
    effects = [0.0] + [float(t) for t in targets]  # effect 0 is the size (realized alpha) row

    def size_max(table) -> float:
        return float(table[table["effect"] == 0.0]["power"].max())

    # 1. the registered method through its own guard
    guarded = diag["collision_probability"] > MAX_SHIFT_COLLISION
    if guarded:
        note = (
            f"collision probability {diag['collision_probability']:.3f} > {MAX_SHIFT_COLLISION:g}: an additive shift "
            "would break every tie and inflate power (P0 2026-08-02 doctrine); the registered CLI refuses this pool"
        )
        for t in targets:
            rows.append({**base, "target": t, "method": METHOD_REGISTERED, "status": STATUS_REFUSED,
                         "required_n": "", "ladder_rung": "", "verdict": "no requirement (refused)", "note": note})
    else:
        table = simulate_campaign(effects, list(n_grid), _centered_symmetrized(d), n_sims, seed,
                                  alpha=alpha, test_fn=wilcoxon_pratt_p, injection=shift_injection)
        tables[METHOD_REGISTERED] = table.to_dict(orient="records")
        for t in targets:
            n_req = _required_from_table(table, t)
            rung, verdict = ladder_verdict(n_req if isinstance(n_req, int) else None)
            rows.append({**base, "target": t, "method": METHOD_REGISTERED, "status": STATUS_SIMULATED,
                         "required_n": n_req, "ladder_rung": rung, "verdict": verdict if isinstance(n_req, int) else "not reached on the grid",
                         "realized_alpha_max": size_max(table),
                         "note": f"run_power_sim null (centered symmetrized diffs), shift_injection, Pratt Wilcoxon, {n_sims} sims, seed {seed}"})
    # 2. the tie-preserving shift (labeled design input)
    if diag["discordant_rate"] == 0.0:
        for t in targets:
            rows.append({**base, "target": t, "method": METHOD_TIE_PRESERVING, "status": STATUS_NO_DISCORDANT,
                         "required_n": "", "ladder_rung": "", "verdict": "no requirement (all differences zero)",
                         "note": "every pair agrees: no discordant pair can carry an effect; no variance to estimate"})
    else:
        max_abs = float(np.max(np.abs(d[d != 0.0])))
        shifts = {float(t): float(t) / diag["discordant_rate"] for t in targets}
        in_range = [t for t in targets if max_abs + shifts[float(t)] <= F1_RANGE]
        table = None
        if in_range:
            table = simulate_campaign([0.0] + [float(t) for t in in_range], list(n_grid), _symmetrized(d), n_sims, seed + 1,
                                      alpha=alpha, test_fn=wilcoxon_pratt_p, injection=tie_preserving_shift)
            tables[METHOD_TIE_PRESERVING] = table.to_dict(orient="records")
        for t in targets:
            shift = shifts[float(t)]
            if t not in in_range:
                rows.append({**base, "target": t, "method": METHOD_TIE_PRESERVING, "status": STATUS_OUT_OF_RANGE,
                             "required_n": "", "ladder_rung": "", "verdict": "no requirement (refused)",
                             "per_discordant_shift": shift,
                             "note": (f"shift per discordant pair {shift:.4f} plus the largest observed |difference| {max_abs:.4f} "
                                      f"exceeds the F1 range {F1_RANGE:g}: the mechanism would need impossible differences; refused")})
                continue
            n_req = _required_from_table(table, t)
            rung, verdict = ladder_verdict(n_req if isinstance(n_req, int) else None)
            below = int(np.sum(np.abs(d[d != 0.0]) < shift))
            rows.append({**base, "target": t, "method": METHOD_TIE_PRESERVING, "status": STATUS_SIMULATED,
                         "required_n": n_req, "ladder_rung": rung, "verdict": verdict if isinstance(n_req, int) else "not reached on the grid",
                         "per_discordant_shift": shift, "realized_alpha_max": size_max(table),
                         "note": (f"symmetrized uncentered diffs (zero mass {diag['tie_mass']:.2f} kept), shift {shift:.4f} on each "
                                  f"discordant pair (= target / discordant rate; mean = target), Pratt Wilcoxon, {n_sims} sims, seed {seed + 1}; "
                                  f"{below} of {int(np.sum(d != 0.0))} observed discordant magnitudes fall under the shift and flip sign, "
                                  "so the requirement is a step function of those few values, not a mean-difference power; "
                                  "models a larger change on the answers fp8 already changes, not more answers changing")})
    # 3. the paired t approximation
    for t in targets:
        n_req = t_required_n(diag["sd_diff"], float(t), alpha=alpha, power=TARGET_POWER)
        rung, verdict = ladder_verdict(n_req)
        rows.append({**base, "target": t, "method": METHOD_T, "status": STATUS_CLOSED_FORM if n_req is not None else STATUS_NO_VARIANCE,
                     "required_n": "" if n_req is None else n_req, "ladder_rung": rung, "verdict": verdict,
                     "note": f"n = ((z_{{1-alpha/2}} + z_power) * sd / delta)^2 with sd {diag['sd_diff']:.4f}; approximate, not the Wilcoxon's exact requirement"})
    return rows, tables


def recommendation(rows: List[Dict[str, Any]], targets: Sequence[float]) -> List[str]:
    """One sentence per dataset and target: the binding (largest over pools) requirement of the
    paired t approximation, the only mean-difference statement this pass makes, beside the
    labeled tie-preserving requirement (its own mechanism), each against the ladder."""
    lines: List[str] = []
    datasets = sorted({r["dataset"] for r in rows})
    for ds in datasets:
        parts = []
        for t in targets:
            seg = []
            for method, name in ((METHOD_T, "t approximation"), (METHOD_TIE_PRESERVING, "tie-preserving mechanism")):
                cands = [r for r in rows if r["dataset"] == ds and r["target"] == t and r["method"] == method]
                nums = [r["required_n"] for r in cands if isinstance(r["required_n"], int)]
                unreached = [r for r in cands if r["required_n"] == NOT_REACHED]
                if unreached:
                    seg.append(f"{name}: not reached on the grid in {len(unreached)} of {len(cands)} pools")
                elif nums and len(nums) == len(cands):
                    n = max(nums)
                    _, verdict = ladder_verdict(n)
                    seg.append(f"{name}: binding n {n} ({verdict})")
                elif nums:
                    seg.append(f"{name}: {len(cands) - len(nums)} of {len(cands)} pools refused, binding n over the rest {max(nums)}")
                else:
                    seg.append(f"{name}: no requirement ({', '.join(sorted({r['status'] for r in cands}))})")
            parts.append(f"delta {t:g}: " + "; ".join(seg))
        lines.append(f"{ds}: " + " | ".join(parts) + ".")
    return lines


def write_outputs(rows: List[Dict[str, Any]], pools: List[Pool], tables: Dict[str, Any], config: Dict[str, Any], out_dir: Path) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "fp8_power.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c, "")) for c in CSV_COLUMNS})
    json_path = out_dir / "fp8_power.json"
    skips = [{"engine": p.engine, "dataset": p.dataset, "family": p.family, "budget_r": p.budget_r, "window": p.window,
              "reason": p.skip_reason} for p in pools if p.skip_reason]
    json_path.write_text(json.dumps({"config": config, "rows": rows, "skips": skips, "power_tables": tables},
                                    indent=2, sort_keys=True, default=str), encoding="utf-8")
    md_path = out_dir / "fp8_power.md"
    lines = [
        "# Power for the fp8 KV-cache contrast (Spec Block A3)",
        "",
        f"- run root: `{config['run_root']}`; scoring pass `{config['scoring_run_id']}`; pair {config['cell_arm']} minus {config['ref_arm']} on `{METRIC}`",
        f"- alpha {config['alpha']:g} two-sided, target power {TARGET_POWER:g}, {config['n_sims']} simulations per grid point, seed {config['seed']}, n grid {config['n_grid']}, registered ladder {list(N_LADDER)}",
        f"- pools: {len([p for p in pools if not p.skip_reason])} analyzed, {len(skips)} named skips",
        "",
        "## Pools",
        "",
        "| engine | dataset | family | rung | n pairs | dropped | mean diff | sd | zero share | collision p |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    seen = set()
    for r in rows:
        k = (r["engine"], r["dataset"], r["family"], r["budget_r"], r["window"])
        if k in seen:
            continue
        seen.add(k)
        rung = "" if r["budget_r"] is None else f"r{r['budget_r']:g}"
        lines.append(f"| {r['engine']} | {r['dataset']} | {r['family']} | {rung} | {r['n_pairs']} | {r['n_dropped_invalid']} | "
                     f"{r['mean_diff']:+.4f} | {r['sd_diff']:.4f} | {r['tie_mass']:.2f} | {r['collision_probability']:.3f} |")
    lines += [
        "",
        "## Required n per pool, method and target",
        "",
        "| engine | dataset | family | rung | target | method | status | required n | ladder rung | verdict | shift per discordant pair | realized alpha (max over n) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        rung = "" if r["budget_r"] is None else f"r{r['budget_r']:g}"
        shift = r.get("per_discordant_shift")
        ra = r.get("realized_alpha_max")
        lines.append(f"| {r['engine']} | {r['dataset']} | {r['family']} | {rung} | {r['target']:g} | {r['method']} | {r['status']} | "
                     f"{r['required_n']} | {r['ladder_rung']} | {r['verdict']} | {'' if shift in (None, '') else f'{shift:.4f}'} | "
                     f"{'' if ra in (None, '') else f'{ra:.3f}'} |")
    lines += ["", "## Recommendation for the S1 registration (binding pool per dataset; the t approximation is the only mean-difference statement)", ""]
    lines += [f"- {s}" for s in recommendation(rows, config["targets"])]
    lines += [
        "",
        "## Methods and limitations",
        "",
        f"1. {METHOD_REGISTERED}: the section 9.6 CLI's method run through its own honest-injection guard; "
        f"a pool with collision probability above {MAX_SHIFT_COLLISION:g} is refused, and the refusal is the method's answer.",
        f"2. {METHOD_TIE_PRESERVING}: keeps the observed zero mass and adds target / discordant_rate to EACH discordant pair "
        "(the column 'shift per discordant pair'), so the pool mean rises by the target. The requirement is therefore a STEP "
        "function of the few observed discordant magnitudes: a magnitude smaller than the shift flips sign deterministically, "
        "and the rank test reaches power on those flips. It is a reproducible power for that labeled mechanism and NOT a "
        "mean-difference power; it assumes fp8 changes the magnitude of the answers it already changes, not the number of "
        "changed answers. A target whose shift would push any observed difference outside the F1 range [-1, 1] is refused "
        "(SHIFT_LEAVES_F1_RANGE). The effect-0 row of each simulation is the realized alpha (column 'realized alpha').",
        f"3. {METHOD_T}: closed form on the observed standard deviation of all paired differences, zeros included; approximate; "
        "it is the one statement about a mean F1 difference this pass can make, and it carries the registration recommendation.",
        "4. Single run, one window per pool; the differences are from the sealed scoring pass; no pooling across datasets.",
    ]
    if skips:
        lines += ["", "## Named skips", ""] + [f"- {s['engine']} {s['dataset']} {s['family']} {s['window']}: {s['reason']}" for s in skips]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "md": md_path}


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_root")
    ap.add_argument("--scoring-run-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell-arm", default=DEFAULT_CELL_ARM)
    ap.add_argument("--ref-arm", default=DEFAULT_REF_ARM)
    ap.add_argument("--targets", type=float, nargs="+", default=list(DEFAULT_TARGETS))
    ap.add_argument("--n-grid", type=int, nargs="+", default=list(DEFAULT_N_GRID))
    ap.add_argument("--n-sims", type=int, default=DEFAULT_N_SIMS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    args = ap.parse_args(list(argv) if argv is not None else None)
    if args.cell_arm == args.ref_arm:
        ap.error("--cell-arm and --ref-arm must differ")
    if any(t <= 0 for t in args.targets) or args.n_sims < 1 or not 0 < args.alpha < 1:
        ap.error("targets must be > 0, --n-sims >= 1, --alpha in (0, 1)")
    out_dir = Path(args.out)
    if "results" in out_dir.resolve().parts:
        print(f"[fp8_power] REFUSED: out dir {out_dir} is inside a results/ tree (pilot archives are read-only)", file=sys.stderr)
        return REFUSED_RC
    run_root = Path(args.run_root)
    try:
        pools = collect_pools(run_root, scoring_run_id=args.scoring_run_id, cell_arm=args.cell_arm, ref_arm=args.ref_arm)
    except Fp8PowerError as exc:
        print(f"[fp8_power] REFUSED: {exc}", file=sys.stderr)
        return REFUSED_RC
    live = [p for p in pools if not p.skip_reason]
    if not live:
        print("[fp8_power] REFUSED: every pool is a named skip: " + "; ".join(f"{p.cell_window}: {p.skip_reason}" for p in pools), file=sys.stderr)
        return REFUSED_RC
    rows: List[Dict[str, Any]] = []
    tables: Dict[str, Any] = {}
    for i, pool in enumerate(pools):  # index over ALL pools so a skip never shifts another pool's seed
        if pool.skip_reason:
            continue
        pool_rows, pool_tables = analyze_pool(pool, targets=args.targets, n_grid=args.n_grid, n_sims=args.n_sims,
                                             seed=args.seed + 100 * i, alpha=args.alpha)
        rows.extend(pool_rows)
        tables[pool.cell_window] = pool_tables
        print(f"[fp8_power] {pool.engine} {pool.dataset} {pool.family} {pool.window}: n_pairs={pool.diffs.size} done", flush=True)
    config = {
        "run_root": str(run_root), "scoring_run_id": args.scoring_run_id, "cell_arm": args.cell_arm, "ref_arm": args.ref_arm,
        "metric": METRIC, "targets": list(args.targets), "n_grid": list(args.n_grid), "n_sims": args.n_sims, "seed": args.seed,
        "alpha": args.alpha, "target_power": TARGET_POWER, "ladder": list(N_LADDER), "min_pairs": MIN_PAIRS,
        "max_shift_collision": MAX_SHIFT_COLLISION,
    }
    paths = write_outputs(rows, pools, tables, config, out_dir)
    for line in recommendation(rows, args.targets):
        print(f"[fp8_power] {line}")
    for name, path in paths.items():
        print(f"[fp8_power] wrote {name}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
