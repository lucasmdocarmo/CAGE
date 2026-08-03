#!/usr/bin/env python3
"""Campaign analysis driver — the §9.11 ONE-LOOK policy in code (PUBLICATION.md D9).

Input: ONE organized run directory (results/<campaign>/<session>/<run_id>) that
already carries ``index/cells_index.csv`` from ``organize_results.py``. This
driver never organizes a tree itself — if the index is absent it tells you to
run the organizer first and stops.

Two modes, mutually exclusive:

- ``--design-input`` (DEFAULT): every output is stamped ``DESIGN-INPUT-ONLY``.
  Run it as often as you like; its numbers may inform design but are barred
  from the paper's confirmatory claims.
- ``--confirmatory``: requires BOTH ``--i-understand-one-look`` AND a
  ``--registered-sha`` string (the SHA of the frozen pre-registration this
  look executes). Outputs are stamped ``CONFIRMATORY`` and the run writes
  ``<run>/analysis_lock.json``; a second confirmatory invocation on the same
  run REFUSES — the campaign data is analyzed once (§9.11).

Pipeline per run:

1. Load ``index/cells_index.csv`` (the organize_results handoff table).
2. Resolve the requested §7.8 contrast ids against the registered
   ``src.analysis.stats.families.CONTRASTS`` registry (default: #4, the
   B6-vs-B3 headline, per dataset). Unknown ids FAIL LOUD.
3. Select the concrete cell/reference row-key pairs (family F1, one tuple
   slot differing) and join per-query metrics from each selected window's
   ``requests.jsonl`` + ``qa_evidence.jsonl`` (the v2-layout loader lives in
   this file; the pilot ``_results_loader.py`` stays pilot-only).
4. Per contrast × metric × dataset: ``tests_by_unit.paired_wilcoxon`` +
   ``wlt.win_loss_tie`` (§8.13 mandatory triple), then ``corrections.holm``
   ACROSS DATASETS within each contrast × metric.
5. Emit ``<run>/analysis/<timestamp>/{stats.json, summary.md,
   forest_<metric>.png, wlt_<metric>.png}`` (figures via
   ``figure_pipeline.plot_forest`` / ``plot_win_loss_tie``), every one
   carrying the mode stamp.

Guards (§9.4 unit-of-analysis rules):

- Rows whose family is F2/F3 (pressure) are NEVER per-query paired — they are
  batch-means-only territory. They are listed in a ``NOT-IMPLEMENTED-YET``
  labeled skip block instead of producing wrong statistics.
- Registered contrasts whose unit is window-level (or whose selector is not a
  single baseline pair, e.g. the gated #6 or the engine-slot #10) are likewise
  reported as labeled skips, never silently dropped.
- Sidedness: this driver runs every Wilcoxon TWO-SIDED (conservative for any
  correctly-directed one-sided registration); the registered sidedness is
  recorded beside each result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for _p in (str(_HERE), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import figure_pipeline as fp  # noqa: E402
from src.analysis.stats.corrections import holm  # noqa: E402
from src.analysis.stats.families import (  # noqa: E402
    CONTRASTS,
    Contrast,
    HEADLINE_CONTRAST_ID,
)
from src.analysis.stats.tests_by_unit import paired_wilcoxon  # noqa: E402
from src.analysis.stats.wlt import win_loss_tie  # noqa: E402

Mode = Literal["design-input", "confirmatory"]

DESIGN_STAMP = "DESIGN-INPUT-ONLY"
CONFIRMATORY_STAMP = "CONFIRMATORY"
NOT_IMPLEMENTED_LABEL = "NOT-IMPLEMENTED-YET"
LOCK_NAME = "analysis_lock.json"
ANALYSIS_DIRNAME = "analysis"
STATS_JSON_NAME = "stats.json"
SUMMARY_MD_NAME = "summary.md"
SCHEMA_VERSION = 1

#: §9.4: per-query pairing is sub-pressure (F1) only; F2/F3 are batch-means.
PRESSURE_FAMILIES: frozenset[str] = frozenset({"F2", "F3"})

#: One tuple-slot-differs discipline: a cell/reference pair must agree on every
#: axis the baseline ids do not determine (baseline fixes arm + retriever).
_PAIR_MATCH_AXES: tuple[str, ...] = ("engine", "model", "topology", "policy")

_INDEX_REQUIRED_COLUMNS: tuple[str, ...] = (
    "run_id", "campaign", "session", "model", "engine", "arm", "baseline",
    "retriever", "policy", "topology", "family", "dataset", "window",
    "window_key", "row_key", "window_dir",
)

#: Metric direction registry — FAIL CLOSED on unknown metrics rather than
#: guessing a direction (a flipped W/L/T is worse than a refusal).
HIGHER_IS_BETTER: dict[str, bool] = {
    # serving (lower is better)
    "ttft_ms": False,
    "latency_ms": False,
    "tpot_ms": False,
    "e2e_ms": False,
    # quality (higher is better)
    "f1_score": True,
    "exact_match": True,
    "f1_answerable": True,
    "exact_match_answerable": True,
    "grounding_score": True,
    "faithfulness": True,
    "context_relevance": True,
    "completeness_bertscore": True,
    "completeness_rouge_l": True,
    "predicate": True,
    "goodput_frac": True,
    "yield_frac": True,
}

DEFAULT_METRICS: tuple[str, ...] = ("ttft_ms",)

CONTRAST_BY_ID: dict[int, Contrast] = {c.id: c for c in CONTRASTS}

#: The two per-query artifacts the v2 loader joins (RESULTS_LAYOUT §1). The
#: dataset-scoped exemption (sharegpt has no qa_evidence.jsonl) is handled by
#: presence, not by name.
_PER_QUERY_ARTIFACTS: tuple[str, ...] = ("requests.jsonl", "qa_evidence.jsonl")


class AnalysisError(RuntimeError):
    """Any refusal or data violation in the driver (fail loud, message first)."""


class OneLookError(AnalysisError):
    """§9.11 one-look policy refusal (lock present / flags missing)."""


# ---------------------------------------------------------------------------
# Index loading
# ---------------------------------------------------------------------------


def load_index(run_dir: Path) -> pd.DataFrame:
    """Load ``index/cells_index.csv`` — refuse (with the fix) when absent."""
    index_path = run_dir / "index" / "cells_index.csv"
    if not index_path.is_file():
        raise AnalysisError(
            f"no index at {index_path} — this driver consumes an ORGANIZED run. "
            f"Run: python scripts/4_analysis/organize_results.py {run_dir}  "
            "(it validates the tree and emits index/cells_index.csv), then re-run "
            "this analysis. The organizer is not auto-run on purpose: its "
            "fail-loud layout report deserves its own look."
        )
    index = pd.read_csv(index_path)
    missing = [c for c in _INDEX_REQUIRED_COLUMNS if c not in index.columns]
    if missing:
        raise AnalysisError(
            f"{index_path} is missing required columns {missing} — re-run "
            "organize_results.py (the index schema moved under you)"
        )
    if index.empty:
        raise AnalysisError(f"{index_path} has no rows — nothing to analyze")
    return index


# ---------------------------------------------------------------------------
# Contrast resolution + cell selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkippedContrast:
    """A requested-but-not-computed contrast, always carried into the outputs."""

    contrast_id: int
    name: str
    label: str
    reason: str


@dataclass(frozen=True)
class ResolvedPair:
    """One concrete (cell, reference) row-key pair for a per-query contrast."""

    contrast: Contrast
    cell_row_key: str
    reference_row_key: str
    datasets: tuple[str, ...]


def classify_contrast(contrast: Contrast) -> str | None:
    """Return a skip reason when this driver cannot compute the contrast yet."""
    if contrast.unit != "per_query" or contrast.family in PRESSURE_FAMILIES:
        return (
            f"unit={contrast.unit!r}, family={contrast.family}: pressure/window "
            "contrasts take window-level batch means only (§9.4 — per-query "
            "pairing under load is prohibited); batch-means driver support is "
            f"{NOT_IMPLEMENTED_LABEL} in run_campaign_analysis.py"
        )
    if contrast.baseline_a is None or contrast.baseline_b is None:
        return (
            f"selector (slot={contrast.slot!r}) is not a single baseline pair; "
            f"driver support is {NOT_IMPLEMENTED_LABEL}"
        )
    return None


def resolve_contrasts(
    contrast_ids: Sequence[int],
) -> tuple[list[Contrast], list[SkippedContrast]]:
    """Map requested §7.8 ids to registry entries; unknown ids FAIL LOUD."""
    unknown = [i for i in contrast_ids if i not in CONTRAST_BY_ID]
    if unknown:
        raise AnalysisError(
            f"unknown §7.8 contrast id(s) {unknown}; the registry "
            f"(src.analysis.stats.families) holds ids {sorted(CONTRAST_BY_ID)}"
        )
    computable: list[Contrast] = []
    skipped: list[SkippedContrast] = []
    for cid in dict.fromkeys(contrast_ids):  # dedupe, keep order
        contrast = CONTRAST_BY_ID[cid]
        reason = classify_contrast(contrast)
        if reason is None:
            computable.append(contrast)
        else:
            skipped.append(
                SkippedContrast(
                    contrast_id=cid,
                    name=contrast.name,
                    label=NOT_IMPLEMENTED_LABEL,
                    reason=reason,
                )
            )
    return computable, skipped


def select_contrast_pairs(index: pd.DataFrame, contrast: Contrast) -> list[ResolvedPair]:
    """Concrete row-key pairs for one baseline-pair contrast (F1 rows only).

    Pairs cells with references agreeing on every non-baseline axis
    (engine/model/topology/policy) — the one-tuple-slot-differs discipline.
    """
    assert contrast.baseline_a is not None and contrast.baseline_b is not None
    f1 = index[index["family"] == "F1"]
    cells = f1[f1["baseline"] == contrast.baseline_a]
    refs = f1[f1["baseline"] == contrast.baseline_b]
    if cells.empty or refs.empty:
        present = sorted(f1["baseline"].dropna().unique())
        raise AnalysisError(
            f"contrast #{contrast.id} ({contrast.name}) needs F1 cells for both "
            f"{contrast.baseline_a} and {contrast.baseline_b}; this run's F1 "
            f"baselines: {present}"
        )
    pairs: list[ResolvedPair] = []
    cell_groups = cells.groupby(list(_PAIR_MATCH_AXES), dropna=False)
    ref_groups = {
        key: grp for key, grp in refs.groupby(list(_PAIR_MATCH_AXES), dropna=False)
    }
    for key, cell_grp in cell_groups:
        ref_grp = ref_groups.get(key)
        if ref_grp is None:
            continue
        cell_keys = sorted(cell_grp["row_key"].unique())
        ref_keys = sorted(ref_grp["row_key"].unique())
        if len(cell_keys) != 1 or len(ref_keys) != 1:
            raise AnalysisError(
                f"contrast #{contrast.id}: axes {dict(zip(_PAIR_MATCH_AXES, key))} "
                f"select multiple cells per side (cells={cell_keys}, "
                f"refs={ref_keys}) — the pair is ambiguous; refusing to guess"
            )
        datasets = tuple(
            sorted(set(cell_grp["dataset"].unique()) & set(ref_grp["dataset"].unique()))
        )
        if not datasets:
            raise AnalysisError(
                f"contrast #{contrast.id}: {cell_keys[0]!r} and {ref_keys[0]!r} "
                "share no dataset — nothing to pair"
            )
        pairs.append(
            ResolvedPair(
                contrast=contrast,
                cell_row_key=cell_keys[0],
                reference_row_key=ref_keys[0],
                datasets=datasets,
            )
        )
    if not pairs:
        raise AnalysisError(
            f"contrast #{contrast.id} ({contrast.name}): {contrast.baseline_a} and "
            f"{contrast.baseline_b} cells never share the same "
            f"{'/'.join(_PAIR_MATCH_AXES)} axes in this run — no legal pair"
        )
    return pairs


def pressure_row_skip(index: pd.DataFrame) -> dict[str, Any] | None:
    """The §9.4 guard block: every F2/F3 row in the run, listed, never paired."""
    pressure = index[index["family"].isin(sorted(PRESSURE_FAMILIES))]
    if pressure.empty:
        return None
    return {
        "label": NOT_IMPLEMENTED_LABEL,
        "reason": (
            "family F2/F3 rows are PRESSURE cells: per-query pairing under load "
            "is prohibited (§9.4) — they take window-level batch means "
            "(tests_by_unit.batch_means_contrast), which this driver does not "
            "wire up yet. Listed so they are never silently absent."
        ),
        "row_keys": sorted(pressure["row_key"].unique()),
        "n_windows": int(len(pressure)),
    }


# ---------------------------------------------------------------------------
# v2-layout per-query loader (this file's own; _results_loader.py is pilot-only)
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise AnalysisError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise AnalysisError(
                    f"{path}:{lineno}: record must be a JSON object, "
                    f"got {type(obj).__name__}"
                )
            records.append(obj)
    return records


def load_per_query(
    run_dir: Path, index: pd.DataFrame, row_keys: Iterable[str]
) -> pd.DataFrame:
    """Long per-query table for the given cells across ALL their index windows.

    Joins ``requests.jsonl`` + ``qa_evidence.jsonl`` per window on
    ``example_id`` (numeric fields only; duplicate example_id lines within a
    window are multiple trials and are averaged, matching the figure-pipeline
    convention). Output columns: row_key, dataset, window_key, example_id,
    plus every numeric field seen. A window with zero joinable records fails
    loud.
    """
    wanted = set(row_keys)
    selection = index[index["row_key"].isin(wanted)]
    if selection.empty:
        raise AnalysisError(f"no index rows for row keys {sorted(wanted)}")
    rows: list[dict[str, Any]] = []
    for rec in selection.itertuples(index=False):
        window_dir = run_dir / str(rec.window_dir)
        merged: dict[str, dict[str, list[float]]] = {}
        n_unjoined = 0
        for name in _PER_QUERY_ARTIFACTS:
            path = window_dir / name
            if not path.is_file():
                continue  # qa_evidence is dataset-exempt for load donors (§1)
            for obj in _read_jsonl(path):
                example_id = obj.get("example_id")
                if not isinstance(example_id, str) or not example_id:
                    n_unjoined += 1
                    continue
                bucket = merged.setdefault(example_id, {})
                for field_name, value in obj.items():
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        continue
                    bucket.setdefault(field_name, []).append(float(value))
        if not merged:
            raise AnalysisError(
                f"{window_dir}: no per-query records with an example_id in "
                f"{'/'.join(_PER_QUERY_ARTIFACTS)} — cannot pair anything "
                f"({n_unjoined} record(s) lacked example_id)"
            )
        for example_id, fields in merged.items():
            rows.append(
                {
                    "row_key": rec.row_key,
                    "dataset": rec.dataset,
                    "window_key": rec.window_key,
                    "example_id": example_id,
                    **{k: float(np.mean(v)) for k, v in fields.items()},
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistics per contrast
# ---------------------------------------------------------------------------


def _metric_direction(metric: str) -> bool:
    try:
        return HIGHER_IS_BETTER[metric]
    except KeyError:
        raise AnalysisError(
            f"metric {metric!r} has no registered direction — add it to "
            f"HIGHER_IS_BETTER in run_campaign_analysis.py (known: "
            f"{sorted(HIGHER_IS_BETTER)}). Guessing a direction flips W/L/T."
        ) from None


def compute_pair_stats(
    per_query: pd.DataFrame, pair: ResolvedPair, metric: str
) -> dict[str, Any]:
    """Per-dataset paired Wilcoxon + W/L/T for one pair, Holm across datasets."""
    higher_is_better = _metric_direction(metric)
    if metric not in per_query.columns:
        raise AnalysisError(
            f"metric {metric!r} appears in no requests.jsonl/qa_evidence.jsonl "
            f"record for contrast #{pair.contrast.id} "
            f"({pair.cell_row_key} vs {pair.reference_row_key})"
        )
    per_dataset: list[dict[str, Any]] = []
    for dataset in pair.datasets:
        sub = per_query[
            (per_query["dataset"] == dataset)
            & (per_query["row_key"].isin([pair.cell_row_key, pair.reference_row_key]))
        ]
        wide = (
            sub.groupby(["example_id", "row_key"], observed=True)[metric]
            .mean()
            .unstack("row_key")
        )
        for key in (pair.cell_row_key, pair.reference_row_key):
            if key not in wide.columns:
                raise AnalysisError(
                    f"contrast #{pair.contrast.id} × {dataset}: no {metric!r} "
                    f"values for {key!r}"
                )
        n_candidates = len(wide)
        wide = wide.dropna(subset=[pair.cell_row_key, pair.reference_row_key])
        n_dropped = n_candidates - len(wide)
        if wide.empty:
            raise AnalysisError(
                f"contrast #{pair.contrast.id} × {dataset}: no overlapping "
                f"example_id with finite {metric!r} on both sides"
            )
        a = wide[pair.cell_row_key].to_numpy(dtype=float)
        b = wide[pair.reference_row_key].to_numpy(dtype=float)
        wilcoxon = paired_wilcoxon(a, b, alternative="two-sided")
        triple = win_loss_tie(a, b, higher_is_better=higher_is_better)
        per_dataset.append(
            {
                "dataset": dataset,
                "n_pairs": wilcoxon.n_pairs,
                "n_dropped_nan": int(n_dropped),
                "median_delta": float(np.median(a - b)),
                "statistic": wilcoxon.statistic,
                "p_value": wilcoxon.p_value,
                "cliffs_delta_paired": wilcoxon.cliffs_delta_paired,
                "wins": triple.wins,
                "losses": triple.losses,
                "ties": triple.ties,
            }
        )
    adjusted = holm([row["p_value"] for row in per_dataset])
    for row, p_adj in zip(per_dataset, adjusted):
        row["p_holm_across_datasets"] = float(p_adj)
    return {
        "contrast_id": pair.contrast.id,
        "name": pair.contrast.name,
        "tier": pair.contrast.tier,
        "cell_baseline": pair.contrast.baseline_a,
        "reference_baseline": pair.contrast.baseline_b,
        "cell_row_key": pair.cell_row_key,
        "reference_row_key": pair.reference_row_key,
        "metric": metric,
        "higher_is_better": higher_is_better,
        "registered_sidedness": pair.contrast.sidedness,
        "test_sidedness": "two-sided (driver policy: conservative superset of "
        "any correctly-directed one-sided registration)",
        "correction": "holm across datasets within contrast × metric",
        "per_dataset": per_dataset,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _baseline_of_key(index: pd.DataFrame, row_key: str) -> str:
    values = index.loc[index["row_key"] == row_key, "baseline"].dropna().unique()
    return str(values[0]) if len(values) else row_key


def render_figures(
    per_query: pd.DataFrame,
    pairs: Sequence[ResolvedPair],
    metrics: Sequence[str],
    out_dir: Path,
    stamp: str,
    index: pd.DataFrame,
) -> list[str]:
    """forest_<metric>.png (per reference) + wlt_<metric>.png, all stamped."""
    figures: list[str] = []
    for metric in metrics:
        higher_is_better = _metric_direction(metric)
        contrast_pairs = tuple(
            dict.fromkeys((p.cell_row_key, p.reference_row_key) for p in pairs)
        )
        involved = sorted({k for pair in contrast_pairs for k in pair})
        wlt_df = per_query[per_query["row_key"].isin(involved)][
            ["row_key", "example_id", metric]
        ]
        wlt_path = out_dir / f"wlt_{metric}.png"
        fp.plot_win_loss_tie(
            wlt_df,
            wlt_path,
            config=fp.WinLossTieConfig(
                contrasts=contrast_pairs,
                metric=metric,
                higher_is_better=higher_is_better,
                title=f"[{stamp}] W/L/T per contrast — {metric} "
                "(pooled across datasets; per-dataset triples in stats.json)",
            ),
        )
        figures.append(wlt_path.name)

        by_reference: dict[str, list[ResolvedPair]] = {}
        for pair in pairs:
            by_reference.setdefault(pair.reference_row_key, []).append(pair)
        multi_reference = len(by_reference) > 1
        for reference_key, group in by_reference.items():
            cells = tuple(dict.fromkeys(p.cell_row_key for p in group))
            common_datasets = set.intersection(*(set(p.datasets) for p in group))
            if not common_datasets:
                raise AnalysisError(
                    f"forest vs {reference_key!r}: contrasted cells share no "
                    "dataset — cannot render per-dataset panels"
                )
            forest_df = per_query[
                per_query["row_key"].isin([*cells, reference_key])
                & per_query["dataset"].isin(common_datasets)
            ][["row_key", "example_id", "dataset", metric]]
            if multi_reference:
                ref_label = _baseline_of_key(index, reference_key)
                name = f"forest_{metric}__vs_{ref_label}.png"
            else:
                name = f"forest_{metric}.png"
            forest_path = out_dir / name
            fp.plot_forest(
                forest_df,
                forest_path,
                config=fp.ForestConfig(
                    reference=reference_key,
                    metric=metric,
                    higher_is_better=higher_is_better,
                    cells=cells,
                    title=f"[{stamp}] paired Δ{metric} vs "
                    f"{_baseline_of_key(index, reference_key)}",
                ),
            )
            figures.append(forest_path.name)
    return figures


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def build_summary_md(stats: Mapping[str, Any]) -> str:
    stamp = stats["mode_stamp"]
    run = stats["run"]
    lines: list[str] = [
        f"# Campaign analysis summary — **{stamp}**",
        "",
        f"- run: `{run['campaign']}/{run['session']}/{run['run_id']}` "
        f"(model `{run['model']}`)",
        f"- generated (UTC): {stats['generated_utc']}",
        f"- mode: {stats['one_look']['mode']}"
        + (
            f" · registered SHA `{stats['one_look']['registered_sha']}`"
            if stats["one_look"]["registered_sha"]
            else ""
        ),
        f"- driver sidedness policy: two-sided Wilcoxon (registered sidedness "
        "recorded per contrast); Holm across datasets within contrast × metric",
        "",
        f"**Every number below is {stamp}.**",
        "",
    ]
    for entry in stats["contrasts"]:
        lines.append(
            f"## Contrast #{entry['contrast_id']} — {entry['name']} "
            f"[{entry['metric']}]"
        )
        lines.append("")
        lines.append(
            f"cell `{entry['cell_baseline']}` = `{entry['cell_row_key']}`  ·  "
            f"reference `{entry['reference_baseline']}` = "
            f"`{entry['reference_row_key']}`  ·  "
            f"{'higher' if entry['higher_is_better'] else 'lower'} is better"
        )
        lines.append("")
        lines.append(
            "| dataset | n_pairs | median Δ (cell−ref) | W/L/T | Cliff's δ "
            "(paired) | p | p (Holm across datasets) |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for row in entry["per_dataset"]:
            lines.append(
                f"| {row['dataset']} | {row['n_pairs']} "
                f"| {row['median_delta']:.4g} "
                f"| {row['wins']}/{row['losses']}/{row['ties']} "
                f"| {row['cliffs_delta_paired']:.3f} "
                f"| {row['p_value']:.3g} | {row['p_holm_across_datasets']:.3g} |"
            )
        lines.append("")
    skipped = stats["skipped"]
    if skipped["pressure_rows"] or skipped["contrasts"]:
        lines.append(f"## Skipped — {NOT_IMPLEMENTED_LABEL}")
        lines.append("")
        if skipped["pressure_rows"]:
            block = skipped["pressure_rows"]
            lines.append(
                f"**Pressure rows (F2/F3), {block['n_windows']} window(s) — "
                "batch-means only (§9.4), never per-query paired:**"
            )
            for key in block["row_keys"]:
                lines.append(f"- `{key}`")
            lines.append("")
        for entry in skipped["contrasts"]:
            lines.append(
                f"- contrast #{entry['contrast_id']} ({entry['name']}): "
                f"{entry['reason']}"
            )
        lines.append("")
    if stats["figures"]:
        lines.append("## Figures")
        lines.append("")
        for name in stats["figures"]:
            lines.append(f"- `{name}` (stamped {stamp} in-figure)")
        lines.append("")
    lines.append("---")
    lines.append(f"Stamp: **{stamp}** · schema v{stats['schema_version']}")
    lines.append("")
    return "\n".join(lines)


def _make_analysis_dir(run_dir: Path) -> Path:
    base = run_dir / ANALYSIS_DIRNAME
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    candidate = base / stamp
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = base / f"{stamp}-{suffix}"
    candidate.mkdir(parents=True)
    return candidate


def read_lock(run_dir: Path) -> dict[str, Any] | None:
    lock_path = run_dir / LOCK_NAME
    if not lock_path.is_file():
        return None
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OneLookError(
            f"{lock_path} exists but is not valid JSON ({exc}) — treating the "
            "run as LOCKED; a corrupt lock is not a license for a second look"
        ) from exc
    return lock if isinstance(lock, dict) else {"raw": lock}


def write_lock(
    run_dir: Path, registered_sha: str, analysis_dir: Path, stats_path: Path
) -> Path:
    lock_path = run_dir / LOCK_NAME
    payload = {
        "policy": "PUBLICATION.md §9.11 ONE-LOOK: this run's confirmatory "
        "analysis has been executed exactly once",
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "registered_sha": registered_sha,
        "analysis_dir": analysis_dir.name,
        "stats_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
    }
    lock_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return lock_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalysisResult:
    analysis_dir: Path
    stats_path: Path
    summary_path: Path
    figures: tuple[str, ...] = field(default_factory=tuple)


def run_analysis(
    run_dir: Path,
    *,
    contrast_ids: Sequence[int],
    metrics: Sequence[str],
    mode: Mode,
    registered_sha: str | None = None,
) -> AnalysisResult:
    """Execute the pipeline; the CLI wraps this with the one-look flag checks."""
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise AnalysisError(f"run directory does not exist: {run_dir}")
    for metric in metrics:
        _metric_direction(metric)  # fail before any I/O on unknown direction

    stamp = CONFIRMATORY_STAMP if mode == "confirmatory" else DESIGN_STAMP
    if mode == "confirmatory":
        if not registered_sha:
            raise OneLookError("confirmatory mode requires a registered SHA")
        lock = read_lock(run_dir)
        if lock is not None:
            raise OneLookError(
                f"ONE-LOOK REFUSAL (§9.11): {run_dir / LOCK_NAME} already exists "
                f"— this run's confirmatory analysis ran at "
                f"{lock.get('locked_utc', '<unknown time>')} under registered "
                f"SHA {lock.get('registered_sha', '<unknown>')!r}. The campaign "
                "data is analyzed once; a second look invalidates the "
                "registration. Design-input re-renders remain available via "
                "--design-input."
            )
    else:
        if read_lock(run_dir) is not None:
            print(
                f"WARNING: {run_dir / LOCK_NAME} exists — the confirmatory look "
                "is spent. These design-input outputs must never be quoted as "
                "confirmatory.",
                file=sys.stderr,
            )

    index = load_index(run_dir)
    computable, skipped_contrasts = resolve_contrasts(contrast_ids)
    pressure_block = pressure_row_skip(index)

    pairs: list[ResolvedPair] = []
    for contrast in computable:
        pairs.extend(select_contrast_pairs(index, contrast))

    per_query = (
        load_per_query(
            run_dir,
            index,
            {k for p in pairs for k in (p.cell_row_key, p.reference_row_key)},
        )
        if pairs
        else pd.DataFrame()
    )

    contrast_stats: list[dict[str, Any]] = []
    for pair in pairs:
        for metric in metrics:
            contrast_stats.append(compute_pair_stats(per_query, pair, metric))

    analysis_dir = _make_analysis_dir(run_dir)
    figures = (
        render_figures(per_query, pairs, metrics, analysis_dir, stamp, index)
        if pairs
        else []
    )

    run_identity = {
        key: str(index[key].iloc[0]) for key in ("run_id", "campaign", "session", "model")
    }
    stats: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mode_stamp": stamp,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run": run_identity,
        "one_look": {
            "mode": mode,
            "registered_sha": registered_sha,
            "lock_file": LOCK_NAME if mode == "confirmatory" else None,
        },
        "requested_contrast_ids": list(dict.fromkeys(contrast_ids)),
        "metrics": list(metrics),
        "contrasts": contrast_stats,
        "skipped": {
            "pressure_rows": pressure_block,
            "contrasts": [
                {
                    "contrast_id": s.contrast_id,
                    "name": s.name,
                    "label": s.label,
                    "reason": s.reason,
                }
                for s in skipped_contrasts
            ],
        },
        "figures": figures,
    }

    stats_path = analysis_dir / STATS_JSON_NAME
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    summary_path = analysis_dir / SUMMARY_MD_NAME
    summary_path.write_text(build_summary_md(stats), encoding="utf-8")

    if mode == "confirmatory":
        assert registered_sha is not None
        write_lock(run_dir, registered_sha, analysis_dir, stats_path)

    return AnalysisResult(
        analysis_dir=analysis_dir,
        stats_path=stats_path,
        summary_path=summary_path,
        figures=tuple(figures),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Campaign analysis driver (D9). Consumes an ORGANIZED run "
            "(index/cells_index.csv from organize_results.py), computes the "
            "registered §7.8 per-query contrasts (default: #4, the B6-vs-B3 "
            "headline, per dataset) with paired Wilcoxon + W/L/T + Holm across "
            "datasets, and emits <run>/analysis/<timestamp>/{stats.json, "
            "summary.md, forest_<metric>.png, wlt_<metric>.png}. Every output "
            "is stamped DESIGN-INPUT-ONLY unless the one-look confirmatory "
            "flags are given (§9.11)."
        )
    )
    parser.add_argument(
        "run_dir", type=Path, help="organized run root: results/<campaign>/<session>/<run_id>"
    )
    parser.add_argument(
        "--contrasts",
        nargs="+",
        type=int,
        default=[HEADLINE_CONTRAST_ID],
        metavar="ID",
        help="§7.8 contrast ids (1-20); unknown ids fail loud; window-unit ids "
        "are reported as NOT-IMPLEMENTED-YET skips (default: 4)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        metavar="METRIC",
        help=f"per-query metric columns to test (default: {' '.join(DEFAULT_METRICS)})",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--design-input",
        action="store_true",
        help="explicit default: outputs stamped DESIGN-INPUT-ONLY, repeatable",
    )
    mode.add_argument(
        "--confirmatory",
        action="store_true",
        help="the ONE confirmatory look (§9.11); requires --i-understand-one-look "
        "AND --registered-sha; writes/checks <run>/analysis_lock.json",
    )
    parser.add_argument(
        "--i-understand-one-look",
        action="store_true",
        help="confirmatory acknowledgment: this is the run's single registered look",
    )
    parser.add_argument(
        "--registered-sha",
        type=str,
        default=None,
        metavar="SHA",
        help="SHA of the frozen pre-registration this confirmatory look executes",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.confirmatory:
        problems: list[str] = []
        if not args.i_understand_one_look:
            problems.append("--i-understand-one-look is missing")
        if not args.registered_sha:
            problems.append("--registered-sha SHA is missing")
        if problems:
            print(
                "REFUSED: --confirmatory is the run's single registered look "
                f"(§9.11) and demands explicit intent: {'; '.join(problems)}. "
                "Nothing was computed and no lock was written.",
                file=sys.stderr,
            )
            return 1
        mode: Mode = "confirmatory"
    else:
        if args.i_understand_one_look or args.registered_sha:
            print(
                "REFUSED: --i-understand-one-look/--registered-sha are "
                "confirmatory-mode flags; pass --confirmatory or drop them "
                "(design-input outputs never carry a registration).",
                file=sys.stderr,
            )
            return 1
        mode = "design-input"

    try:
        result = run_analysis(
            args.run_dir,
            contrast_ids=args.contrasts,
            metrics=args.metrics,
            mode=mode,
            registered_sha=args.registered_sha,
        )
    except AnalysisError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    stamp = CONFIRMATORY_STAMP if mode == "confirmatory" else DESIGN_STAMP
    print(f"[campaign-analysis] mode    : {mode} (outputs stamped {stamp})")
    print(f"[campaign-analysis] stats   : {result.stats_path}")
    print(f"[campaign-analysis] summary : {result.summary_path}")
    for name in result.figures:
        print(f"[campaign-analysis] figure  : {result.analysis_dir / name}")
    if mode == "confirmatory":
        print(f"[campaign-analysis] LOCKED  : {Path(args.run_dir) / LOCK_NAME} (§9.11 one-look)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
