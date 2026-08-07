#!/usr/bin/env python3
"""§9.7 pipeline-calibration CLI — A/A split-half + effect-injection on the pilot archive.

CALIBRATION / DESIGN-INPUT ONLY. Every number this tool produces is an operating
characteristic of the measurement machinery (PUBLICATION.md §9.7 UPGRADE 1),
measured on pilot data. No output of this tool is a scientific finding of the
study (THE-WORK framing, 2026-07-27), and this tool never runs the campaign
driver, never takes the §9.11 one-look, and never writes an ``analysis_lock.json``.

What it does (thin composition over ``src.analysis.stats.calibration``):

1. Loads per-query pooled per-example values for ONE stable arm
   (``baselines/no_cache``) from the three full pilot runs via the canonical
   loader ``scripts/4_analysis/_results_loader.py`` (READ-ONLY).
2. A/A (§9.7a): per dataset, seeded split-half ``aa_split_half`` over the exact
   campaign paired-test path (``tests_by_unit.paired_wilcoxon`` p-value) at the
   registered α. The gate artifact's single ``aa`` block is the POOL of the
   per-dataset primary-path A/As (same exact binomial CI machinery).
3. Effect injection (§9.7b): ``recover_power`` per (dataset × family × effect)
   on the three registered metric families — a continuous serving metric
   (ttft_ms), a continuous quality metric (faithfulness), and a binary
   predicate metric (exact_match → McNemar, §9.4). The binary family uses the
   HONEST tie-flip model (``kind="flip"``); additive shifts are REFUSED on any
   tie-heavy metric by a coded guard (the 2026-08-02 P0 decision: an additive
   shift breaks every tie and inflated power 1.000 vs honest tie-flip 0.215).
4. Writes the ``CalibrationReport`` gate artifact via ``CalibrationReport.write``
   plus its ``to_markdown()`` companion and a labeled provenance JSON (seed,
   split counts, source runs, loader validity rule).
5. Self-check at FUNCTION level: parses the written artifact with the campaign
   driver's ``load_calibration_report`` and gates it with ``check_calibration``
   (imports only — the driver CLI is never invoked).

Injection ``target_power`` is deliberately ``None``: the §9.6 simulation-based
power targets are not yet registered, and gating on invented targets would make
the artifact pass/fail on unregistered numbers (fail-closed doctrine). The
injection table is therefore measured power, reported not gated.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for _p in (str(_HERE), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _results_loader as loader  # noqa: E402
from src.analysis.cellspec import from_legacy  # noqa: E402
from src.analysis.stats.calibration import (  # noqa: E402
    AAResult,
    CalibrationReport,
    InjectionResult,
    _rejection_ci,  # same exact binomial CI as every AAResult (one CI implementation)
    aa_split_half,
    recover_power,
)
from src.analysis.stats.tests_by_unit import (  # noqa: E402
    mcnemar_binary,
    paired_wilcoxon,
)

STAMP = (
    "CALIBRATION / DESIGN-INPUT ONLY (§9.7): measured operating characteristics "
    "of the CAGE stats machinery on pilot data. Properties of the measurement "
    "machinery — NEVER scientific findings of the study; no number here may be "
    "cited as a result (THE-WORK framing 2026-07-27)."
)

DEFAULT_DATASETS: tuple[str, ...] = ("squad_v2", "hotpotqa", "musique")
RUN_OF_DATASET: dict[str, str] = {
    ds: f"2026-07-16_full_qwen3-8b_100x3_{ds}" for ds in DEFAULT_DATASETS
}
AA_TREE = "baselines"
AA_ARM = "no_cache"
PRIMARY_AA_METRIC = "f1_score"  # the P0 §4.3 A/A config (tie-heavy stress case)
DEFAULT_SEED = 20260805
DEFAULT_ALPHA = 0.05
# P0 dry-run 2026-08-02 used 200 splits (fp=0.060, CI 0.031-0.102); we double it.
DEFAULT_AA_SPLITS = 400
DEFAULT_INJECTION_SPLITS = 400
MIN_OBSERVATIONS = 24
# Guard threshold: probability two same-arm draws collide (sum p_k^2). ttft_ms
# ~0.003, faithfulness <=0.02 on the pilot; f1_score ~0.4+ and grounding ~0.8+
# are refused for additive shifts (the P0 additive-injection bug).
MAX_SHIFT_COLLISION = 0.05

LOADER_VALIDITY_RULE = (
    "scripts/4_analysis/_results_loader.py canonical rule: valid row = NOT error "
    "AND NOT empty_generation (uniform for all metrics); headline rows = valid "
    "rows at repeat_index 0; estimand = pooled per-example mean over headline "
    "rows per (cell, example_id) (per_example) — the exact unit the campaign's "
    "paired tests consume."
)

P0_REFERENCE: dict[str, Any] = {
    "source": "scripts/4_analysis/P0_DRYRUN_REPORT.md §4.3 (2026-08-02)",
    "config": "no_cache f1_score, squad_v2, 300 obs, 200 splits, paired_wilcoxon",
    "fp_rate": 0.060,
    "ci": [0.031, 0.102],
    "n_splits": 200,
}


class CalibrationCLIError(RuntimeError):
    """Invalid input, unusable pilot data, or a refused injection (fail closed)."""


def wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    """The campaign's continuous paired-test path (§9.4), as a TestFn."""
    return float(paired_wilcoxon(a, b, alternative="two-sided").p_value)


def mcnemar_p(a: np.ndarray, b: np.ndarray) -> float:
    """The campaign's binary-predicate test path (§9.4), as a TestFn."""
    return float(mcnemar_binary(a, b, alternative="two-sided").p_value)


@dataclass(frozen=True)
class FamilySpec:
    """One registered metric family: metric column, test path, injection model."""

    name: str
    metric: str
    kind: str  # "shift" | "flip" — validated by guard_injection_kind
    test_name: str
    test_fn: Callable[[np.ndarray, np.ndarray], float]
    effect_sizes: tuple[float, ...]
    effect_unit: str


FAMILIES: tuple[FamilySpec, ...] = (
    FamilySpec(
        name="serving_continuous",
        metric="ttft_ms",
        kind="shift",
        test_name="paired_wilcoxon",
        test_fn=wilcoxon_p,
        effect_sizes=(10.0, 25.0, 50.0),
        effect_unit="ms (additive; metric is continuous/tie-free — guard-verified)",
    ),
    FamilySpec(
        name="quality_continuous",
        metric="faithfulness",
        kind="shift",
        test_name="paired_wilcoxon",
        test_fn=wilcoxon_p,
        effect_sizes=(0.02, 0.05, 0.10),
        effect_unit="score units (additive; metric is continuous — guard-verified)",
    ),
    FamilySpec(
        name="binary_predicate",
        metric="exact_match",
        kind="flip",
        test_name="mcnemar_binary",
        test_fn=mcnemar_p,
        effect_sizes=(0.02, 0.05, 0.10),
        effect_unit="fraction of 0-outcomes flipped to 1 (honest tie-flip model)",
    ),
)


def collision_probability(values: np.ndarray) -> float:
    """P(two independent same-arm draws are exactly equal) = sum p_k^2.

    This is the tie mass an artificial A/A pairing sees; an additive shift on a
    metric with non-trivial collision mass breaks EVERY tie and inflates
    recovered power (the P0 2026-08-02 additive-injection bug).
    """
    if values.size == 0:
        raise CalibrationCLIError("collision_probability on empty data")
    _, counts = np.unique(values, return_counts=True)
    p = counts / counts.sum()
    return float(np.sum(p * p))


def guard_injection_kind(
    kind: str,
    values: np.ndarray,
    *,
    metric: str,
    max_collision: float = MAX_SHIFT_COLLISION,
) -> dict[str, Any]:
    """Enforce the honest-injection doctrine; returns tie diagnostics.

    - ``shift`` is allowed ONLY on effectively continuous metrics (collision
      probability <= ``max_collision``): the P0 2026-08-02 decision.
    - ``flip`` is allowed ONLY on strictly binary 0/1 metrics (the discordant-
      pair process; ``calibration.inject_effect`` re-checks this fail-closed).
    """
    coll = collision_probability(values)
    is_binary = bool(np.isin(values, (0.0, 1.0)).all())
    if kind == "shift":
        if coll > max_collision:
            raise CalibrationCLIError(
                f"REFUSED (P0 2026-08-02 decision): additive-shift injection on "
                f"tie-heavy metric {metric!r} (collision probability {coll:.3f} > "
                f"{max_collision:g}). A shift breaks every tie and inflates "
                "recovered power (honest tie-flip 0.215 vs inflated 1.000 in the "
                "P0 fix pass); use the tie-flip model on a binary predicate."
            )
    elif kind == "flip":
        if not is_binary:
            raise CalibrationCLIError(
                f"REFUSED: tie-flip injection requires a strictly binary 0/1 "
                f"metric; {metric!r} is not binary."
            )
    else:
        raise CalibrationCLIError(f"unknown injection kind {kind!r} (shift|flip)")
    return {"collision_probability": coll, "binary": is_binary}


def load_arm_metric(run_root: Path, metric: str) -> np.ndarray:
    """Pooled per-example values for the A/A arm via the canonical loader.

    READ-ONLY on the pilot archive. Fails closed on a missing cell, a missing
    metric, or too few observations.
    """
    cell_dir = Path(run_root) / AA_TREE / AA_ARM
    if not cell_dir.is_dir():
        raise CalibrationCLIError(f"A/A arm directory not found: {cell_dir}")
    df = loader.load_cell(cell_dir, AA_ARM)
    pe = loader.per_example(df, metric)
    values = pe["value"].to_numpy(dtype=float)
    if values.size < MIN_OBSERVATIONS:
        raise CalibrationCLIError(
            f"metric {metric!r} under {cell_dir} has only {values.size} pooled "
            f"per-example observations (< {MIN_OBSERVATIONS})"
        )
    return values


def pool_aa(results: Sequence[AAResult], *, alpha: float) -> AAResult:
    """Pool per-dataset primary-path A/As into the single gate ``aa`` block.

    Counts add; the CI is the same exact binomial CI every AAResult carries.
    (Caveat, recorded in provenance: splits within a dataset reuse one
    observation vector, so the binomial CI treats mildly dependent splits as
    independent — the P0 pass shares this property.)
    """
    if not results:
        raise CalibrationCLIError("no per-dataset A/A results to pool")
    for r in results:
        if abs(r.alpha - alpha) > 1e-12:
            raise CalibrationCLIError(
                f"cannot pool A/A results at mixed alphas ({r.alpha} vs {alpha})"
            )
    n = int(sum(r.n_splits for r in results))
    k = int(sum(r.n_rejections for r in results))
    ci_low, ci_high = _rejection_ci(k, n)
    return AAResult(
        n_splits=n,
        alpha=alpha,
        n_rejections=k,
        fp_rate=k / n,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def _aa_dict(r: AAResult) -> dict[str, Any]:
    return {
        "n_splits": r.n_splits,
        "alpha": r.alpha,
        "n_rejections": r.n_rejections,
        "fp_rate": r.fp_rate,
        "ci95": [r.ci_low, r.ci_high],
        "approximates_nominal": r.approximates_nominal,
    }


def run_calibration(
    dataset_runs: Mapping[str, Path],
    *,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
    aa_splits: int = DEFAULT_AA_SPLITS,
    injection_splits: int = DEFAULT_INJECTION_SPLITS,
) -> tuple[CalibrationReport, dict[str, Any]]:
    """Run the full §9.7 suite; returns (gate report, labeled provenance)."""
    if not dataset_runs:
        raise CalibrationCLIError("no datasets given")
    if aa_splits < 1 or injection_splits < 1:
        raise CalibrationCLIError("split counts must be >= 1")

    aa_primary: dict[str, dict[str, Any]] = {}
    aa_per_family: dict[str, dict[str, Any]] = {}
    aa_results: list[AAResult] = []
    injections: list[InjectionResult] = []
    injections_labeled: list[dict[str, Any]] = []
    n_observations_primary = 0

    for i_ds, (dataset, run_root) in enumerate(sorted(dataset_runs.items())):
        run_root = Path(run_root)
        ds_seed = seed + 1000 * i_ds

        # --- (2) A/A, primary path: f1_score over paired_wilcoxon (P0 config).
        primary_values = load_arm_metric(run_root, PRIMARY_AA_METRIC)
        n_observations_primary += int(primary_values.size)
        aa = aa_split_half(
            primary_values, wilcoxon_p, aa_splits, ds_seed, alpha=alpha
        )
        aa_results.append(aa)
        aa_primary[dataset] = {
            "metric": PRIMARY_AA_METRIC,
            "test": "paired_wilcoxon",
            "n_observations": int(primary_values.size),
            "seed": ds_seed,
            **_aa_dict(aa),
        }

        # --- (3) per-family secondary A/A + injections.
        aa_per_family[dataset] = {}
        for i_fam, fam in enumerate(FAMILIES):
            fam_values = load_arm_metric(run_root, fam.metric)
            diagnostics = guard_injection_kind(
                fam.kind, fam_values, metric=fam.metric
            )
            fam_aa_seed = ds_seed + 10 * (i_fam + 1)
            fam_aa = aa_split_half(
                fam_values, fam.test_fn, aa_splits, fam_aa_seed, alpha=alpha
            )
            aa_per_family[dataset][fam.name] = {
                "metric": fam.metric,
                "test": fam.test_name,
                "n_observations": int(fam_values.size),
                "seed": fam_aa_seed,
                **_aa_dict(fam_aa),
            }
            inj_seed = ds_seed + 100 * (i_fam + 1)
            for effect in fam.effect_sizes:
                inj = recover_power(
                    fam_values,
                    fam.test_fn,
                    effect,
                    injection_splits,
                    inj_seed,
                    kind=fam.kind,  # type: ignore[arg-type]
                    alpha=alpha,
                    target_power=None,  # §9.6 targets not yet registered
                )
                injections.append(inj)
                injections_labeled.append(
                    {
                        "dataset": dataset,
                        "family": fam.name,
                        "metric": fam.metric,
                        "test": fam.test_name,
                        "kind": fam.kind,
                        "effect_size": float(effect),
                        "effect_unit": fam.effect_unit,
                        "seed": inj_seed,
                        "n_observations": int(fam_values.size),
                        "n_splits": inj.n_splits,
                        "n_rejections": inj.n_rejections,
                        "power": inj.power,
                        "ci95": [inj.ci_low, inj.ci_high],
                        "diagnostics": diagnostics,
                    }
                )

    pooled = pool_aa(aa_results, alpha=alpha)
    report = CalibrationReport(
        seed=seed,
        n_observations=n_observations_primary,
        aa=pooled,
        injections=tuple(injections),
    )

    provenance: dict[str, Any] = {
        "stamp": STAMP,
        "artifact": "§9.7 pipeline-calibration gate artifact (A/A + injection)",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repo_head": _git_head(),
        "seed": seed,
        "alpha": alpha,
        "aa_splits_per_dataset": aa_splits,
        "injection_splits": injection_splits,
        "aa_arm": {
            "tree": AA_TREE,
            "cell": AA_ARM,
            "cellspec_row_key": from_legacy(AA_ARM).to_row_key(),
        },
        "source_runs": {ds: str(Path(p)) for ds, p in sorted(dataset_runs.items())},
        "loader": {
            "module": "scripts/4_analysis/_results_loader.py",
            "validity_rule": LOADER_VALIDITY_RULE,
        },
        "gate_aa_definition": (
            "pooled primary-path A/A across datasets (counts added; exact "
            "binomial CI); per-dataset results below. Caveat: splits within a "
            "dataset resample ONE observation vector, so the binomial CI "
            "treats mildly dependent splits as independent (same property as "
            "the P0 pass)."
        ),
        "aa_primary_per_dataset": aa_primary,
        "aa_pooled_gate": _aa_dict(pooled),
        "aa_per_family_per_dataset": aa_per_family,
        "injections": injections_labeled,
        "injection_guard": {
            "max_shift_collision_probability": MAX_SHIFT_COLLISION,
            "rule": (
                "additive shift allowed ONLY where collision probability <= "
                "threshold (continuous metrics); tie-flip required for binary/"
                "tie-heavy families (P0 2026-08-02 decision)"
            ),
        },
        "target_power_policy": (
            "target_power=None on every injection: §9.6 simulation-based "
            "targets are not yet registered; gating on invented targets is "
            "prohibited (fail-closed). Injection powers are measured "
            "operating characteristics, reported not gated."
        ),
        "p0_reference": P0_REFERENCE,
    }
    return report, provenance


def _git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except OSError:
        return "unknown"


def _check_out_dir(out_dir: Path) -> Path:
    """Refuse output locations inside any results/ tree (pilot data is read-only)."""
    out_dir = Path(out_dir).resolve()
    if "results" in out_dir.parts:
        raise CalibrationCLIError(
            f"REFUSED: out dir {out_dir} is inside a results/ tree — the pilot "
            "archive is strictly read-only; outputs belong under "
            "MyDocs/registration/ or a scratchpad."
        )
    return out_dir


def _markdown_document(report: CalibrationReport, provenance: dict[str, Any]) -> str:
    lines: list[str] = [
        "# §9.7 pipeline calibration — gate artifact companion",
        "",
        f"> **{STAMP}**",
        "",
        f"Generated {provenance['generated_utc']} at repo HEAD "
        f"`{provenance['repo_head']}`; seed `{provenance['seed']}`; "
        f"α={provenance['alpha']:g}; A/A splits/dataset="
        f"{provenance['aa_splits_per_dataset']}; injection splits="
        f"{provenance['injection_splits']}.",
        "",
        report.to_markdown(),
        "",
        "## Labeled detail (the flat gate tables above, with names)",
        "",
        "### A/A per dataset — primary path "
        f"(`{PRIMARY_AA_METRIC}`, paired Wilcoxon, arm `{AA_ARM}`)",
        "",
        "| dataset | n_obs | n_splits | rejections | FP rate | 95% CI | covers α |",
        "|---|---|---|---|---|---|---|",
    ]
    for ds, r in provenance["aa_primary_per_dataset"].items():
        lines.append(
            f"| {ds} | {r['n_observations']} | {r['n_splits']} | "
            f"{r['n_rejections']} | {r['fp_rate']:.4f} | "
            f"[{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}] | "
            f"{'PASS' if r['approximates_nominal'] else 'FAIL'} |"
        )
    lines += [
        "",
        "### A/A per dataset × registered family path (secondary, provenance-only)",
        "",
        "| dataset | family | metric | test | FP rate | 95% CI | covers α |",
        "|---|---|---|---|---|---|---|",
    ]
    for ds, fams in provenance["aa_per_family_per_dataset"].items():
        for fam_name, r in fams.items():
            lines.append(
                f"| {ds} | {fam_name} | {r['metric']} | {r['test']} | "
                f"{r['fp_rate']:.4f} | [{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}] | "
                f"{'PASS' if r['approximates_nominal'] else 'FAIL'} |"
            )
    lines += [
        "",
        "### Injections, labeled (honest models; target_power unregistered → not gated)",
        "",
        "| dataset | family | metric | test | kind | effect | power | 95% CI |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in provenance["injections"]:
        lines.append(
            f"| {r['dataset']} | {r['family']} | {r['metric']} | {r['test']} | "
            f"{r['kind']} | {r['effect_size']:g} | {r['power']:.4f} | "
            f"[{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}] |"
        )
    lines += [
        "",
        "### Provenance summary",
        "",
        f"- A/A arm: `{AA_TREE}/{AA_ARM}` "
        f"(CellSpec `{provenance['aa_arm']['cellspec_row_key']}`)",
        "- source runs: "
        + ", ".join(f"`{p}`" for p in provenance["source_runs"].values()),
        f"- loader validity rule: {provenance['loader']['validity_rule']}",
        f"- injection guard: {provenance['injection_guard']['rule']}",
        f"- P0 reference: {provenance['p0_reference']['config']} → "
        f"fp={provenance['p0_reference']['fp_rate']:g} "
        f"CI {provenance['p0_reference']['ci']} "
        f"({provenance['p0_reference']['n_splits']} splits)",
        "",
    ]
    return "\n".join(lines)


def write_outputs(
    report: CalibrationReport, provenance: dict[str, Any], out_dir: Path
) -> dict[str, Path]:
    """Write gate JSON (CalibrationReport.write), markdown companion, provenance."""
    out_dir = _check_out_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = report.write(out_dir / "calibration_report.json")
    md_path = out_dir / "calibration_report.md"
    md_path.write_text(_markdown_document(report, provenance), encoding="utf-8")
    prov_path = out_dir / "calibration_provenance.json"
    prov_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"report": report_path, "markdown": md_path, "provenance": prov_path}


def self_check(report_path: Path) -> dict[str, Any]:
    """FUNCTION-level consumer proof: the campaign driver parses + gates the file.

    Imports ``run_campaign_analysis`` as a module and calls
    ``load_calibration_report`` + ``check_calibration`` — the driver CLI is
    never invoked, no confirmatory mode, no lock.
    """
    import run_campaign_analysis as rca

    loaded = rca.load_calibration_report(Path(report_path))
    return rca.check_calibration(loaded, Path(report_path))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "§9.7 A/A + effect-injection calibration on the pilot archive "
            "(design-input only; pilot data is read-only)."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=_REPO_ROOT / "results" / "phase2",
        help="Pilot archive root holding the three full 100x3 runs (READ-ONLY).",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--aa-splits", type=int, default=DEFAULT_AA_SPLITS)
    parser.add_argument(
        "--injection-splits", type=int, default=DEFAULT_INJECTION_SPLITS
    )
    parser.add_argument(
        "--datasets", nargs="+", default=list(DEFAULT_DATASETS),
        choices=sorted(DEFAULT_DATASETS),
    )
    args = parser.parse_args(argv)

    out_dir = _check_out_dir(args.out_dir)
    dataset_runs: dict[str, Path] = {}
    for ds in args.datasets:
        run_root = Path(args.results_root) / RUN_OF_DATASET[ds]
        if not run_root.is_dir():
            raise CalibrationCLIError(f"pilot run not found: {run_root}")
        dataset_runs[ds] = run_root

    print(f"[run_calibration] {STAMP}")
    report, provenance = run_calibration(
        dataset_runs,
        seed=args.seed,
        alpha=args.alpha,
        aa_splits=args.aa_splits,
        injection_splits=args.injection_splits,
    )
    paths = write_outputs(report, provenance, out_dir)

    summary = self_check(paths["report"])
    provenance["self_check"] = {
        "consumer": "scripts/4_analysis/run_campaign_analysis.py "
        "load_calibration_report + check_calibration (function-level import; "
        "driver CLI never invoked)",
        **summary,
    }
    paths["provenance"].write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(
        f"[run_calibration] gate A/A: fp={report.aa.fp_rate:.4f} "
        f"CI [{report.aa.ci_low:.4f}, {report.aa.ci_high:.4f}] over "
        f"{report.aa.n_splits} pooled splits — "
        f"{'PASS' if report.aa.approximates_nominal else 'FAIL'}"
    )
    for ds, r in provenance["aa_primary_per_dataset"].items():
        print(
            f"[run_calibration]   A/A {ds}: fp={r['fp_rate']:.4f} "
            f"CI [{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}]"
        )
    print(f"[run_calibration] consumer self-check verdict: {summary['verdict']}")
    for name, p in paths.items():
        print(f"[run_calibration] wrote {name}: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
