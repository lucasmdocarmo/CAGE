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
4. Per contrast × metric × dataset: ``tests_by_unit.paired_wilcoxon`` for
   continuous metrics, ``tests_by_unit.mcnemar_binary`` for the binary §8.5 Y
   predicate (§9.4), plus ``wlt.win_loss_tie`` (§8.13 mandatory triple) either
   way; window-unit baseline-pair contrasts take
   ``tests_by_unit.batch_means_contrast`` on window-level means (§9.4 — the
   only legal test under load). Multiplicity correction is tier-conditional
   (§9.1/§9.3): primary-tier contrasts carry NO cross-dataset correction
   (full α per dataset, pooling prohibited); every other tier gets a
   DIAGNOSTIC ``corrections.holm`` ACROSS DATASETS within each contrast ×
   metric — while the REGISTERED §9.3 Holm-WITHIN-FAMILY correction (sibling
   contrasts pooled per group×metric×dataset via
   ``families.compile_family_map``) is executed by
   ``gatekeeping.evaluate_chain`` and reported in ``stats['gatekeeping']``
   with the full auditable trace (Dmitrienko serial order =
   ``families.PRIMARY_CHAIN_ORDER``).
5. §9.5: the family map's declared conditional-TOST equivalence legs
   (fingerprint NONE predictions) are computed via
   ``equivalence.conditional_tost`` (+ a ROPE sensitivity line) when the
   registered margin, the policy/none cell pair, and the S2 ``policy_event``
   mask exist — and listed as labeled skips otherwise.
6. Emit ``<run>/analysis/<timestamp>/{stats.json, summary.md,
   forest_<metric>.png, wlt_<metric>.png}`` (figures via
   ``figure_pipeline.plot_forest`` / ``plot_win_loss_tie``), every one
   carrying the mode stamp.

Confirmatory preconditions (all checked BEFORE the §9.11 lock is touched):

- ``--calibration-report``: a PASSING §9.7 calibration artifact
  (CalibrationReport JSON; refusal mirrors ``prereg.py``).
- §9.10 ledger verification: the raw tree must verify against its sealed
  content-hash ledger (``stats.ledger.verify_ledger``).
- §9.8 blinding: with a sealed arm map present, design-input outputs stay
  scrambled (labels masked, figures suppressed); the confirmatory look
  performs and records the ONE-TIME logged unblinding.

Guards (§9.4 unit-of-analysis rules):

- Rows whose family is F2/F3 (pressure) are NEVER per-query paired — they are
  batch-means-only territory. Pressure rows not consumed by a computed
  batch-means contrast are listed in a labeled skip block instead of
  producing wrong statistics.
- Registered contrasts whose selector is not a single baseline pair (e.g. the
  gated #6, the engine-slot #10, the estimand-carrying #13/#14) are reported
  as labeled skips, never silently dropped.
- Sidedness: this driver runs every test TWO-SIDED (conservative for any
  correctly-directed one-sided registration); the registered sidedness is
  recorded beside each result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field, replace as dc_replace
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
from organize_results import GROUP_OF_MODEL  # noqa: E402
from src.analysis.stats import blinding as blinding_mod  # noqa: E402
from src.analysis.stats.blinding import (  # noqa: E402
    AlreadyUnblindedError,
    BlindingError,
)
from src.analysis.stats.calibration import (  # noqa: E402
    AAResult,
    CalibrationReport,
    InjectionResult,
)
from src.analysis.stats.corrections import holm  # noqa: E402
from src.analysis.stats.equivalence import (  # noqa: E402
    conditional_tost,
    rope_sensitivity,
)
from src.analysis.stats.families import (  # noqa: E402
    CONTRASTS,
    Contrast,
    FINGERPRINT_SUB_HYPOTHESES,
    HEADLINE_CONTRAST_ID,
    KNOWN_DATASETS,
    PREDICATE_METRIC,
    PRIMARY_CHAIN_ORDER,
    FamilyMapError,
    compile_family_map,
)
from src.analysis.stats.gatekeeping import (  # noqa: E402
    GatekeepingError,
    GatekeepingTrace,
    PrimaryOutcome,
    SecondaryOutcome,
    evaluate_chain,
)
from src.analysis.stats.ledger import LedgerError, read_ledger, verify_ledger  # noqa: E402
from src.analysis.stats.tests_by_unit import (  # noqa: E402
    batch_means_contrast,
    mcnemar_binary,
    paired_wilcoxon,
)
from src.analysis.stats.wlt import win_loss_tie  # noqa: E402

Mode = Literal["design-input", "confirmatory"]

DESIGN_STAMP = "DESIGN-INPUT-ONLY"
CONFIRMATORY_STAMP = "CONFIRMATORY"
NOT_IMPLEMENTED_LABEL = "NOT-IMPLEMENTED-YET"
LOCK_NAME = "analysis_lock.json"
ANALYSIS_DIRNAME = "analysis"
STATS_JSON_NAME = "stats.json"
SUMMARY_MD_NAME = "summary.md"
SCHEMA_VERSION = 2
LEDGER_NAME = "ledger.json"

#: §9.8 blinding artifacts (run-root relative). The sealed map is produced at
#: scoring time by ``src.analysis.stats.blinding.scramble_labels``; this driver
#: only CONSUMES it.
BLINDING_DIRNAME = "blinding"
SEALED_MAP_NAME = "sealed_arm_map.json"
UNBLIND_LOG_NAME = "unblind_log.jsonl"

#: §9.3 serial-gate upstream for the F1 secondary families: the headline
#: co-primary set (#4) gates them — a secondary is confirmatory only if #4
#: passed on the SAME dataset × metric (gatekeeping.evaluate_chain enforces it).
_SECONDARY_UPSTREAM_ID: int = HEADLINE_CONTRAST_ID

#: §9.5: the three pre-registered NONE fingerprints tested by conditional TOST.
#: 'distribute' is a topology-slot leg (no policy-axis pairing exists for it in
#: a single-topology run) — it is DECLARED and reported as a labeled skip until
#: DIST-topology data exists.
_TOST_POLICY_AXIS_LEGS: frozenset[str] = frozenset({"recompute", "offload"})
#: Per-query S2 policy-event mask column (conditional population, §9.5).
POLICY_EVENT_COLUMN = "policy_event"

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
    """Return a skip reason when this driver cannot compute the contrast.

    Computable paths: (a) per-query baseline-pair contrasts on sub-pressure F1
    rows (paired Wilcoxon / McNemar); (b) window-unit baseline-pair contrasts
    on loaded rows via window-level batch means
    (``tests_by_unit.batch_means_contrast``, §9.4). Selector contrasts (no
    single baseline pair — e.g. the gated #6, the engine-slot #10, the
    estimand-carrying #13/#14) remain labeled skips.
    """
    if contrast.baseline_a is None or contrast.baseline_b is None:
        return (
            f"selector (slot={contrast.slot!r}) is not a single baseline pair; "
            f"driver support is {NOT_IMPLEMENTED_LABEL}"
        )
    if contrast.unit == "per_query" and contrast.family not in PRESSURE_FAMILIES:
        return None
    if contrast.unit == "window":
        return None  # batch-means path (§9.4)
    return (
        f"unit={contrast.unit!r}, family={contrast.family}: no registered "
        f"driver path; support is {NOT_IMPLEMENTED_LABEL}"
    )


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


def pressure_row_skip(
    index: pd.DataFrame, consumed_row_keys: frozenset[str] = frozenset()
) -> dict[str, Any] | None:
    """The §9.4 guard block: F2/F3 rows NOT consumed by a computed batch-means
    contrast, listed so they are never silently absent (never per-query paired).
    """
    pressure = index[
        index["family"].isin(sorted(PRESSURE_FAMILIES))
        & ~index["row_key"].isin(consumed_row_keys)
    ]
    if pressure.empty:
        return None
    return {
        "label": "PRESSURE-ROWS-NOT-IN-A-COMPUTED-CONTRAST",
        "reason": (
            "family F2/F3 rows are PRESSURE cells: per-query pairing under load "
            "is prohibited (§9.4) — they take window-level batch means "
            "(tests_by_unit.batch_means_contrast, wired for registered "
            "baseline-pair window contrasts). These rows matched no computed "
            "batch-means contrast in this invocation. Listed so they are never "
            "silently absent."
        ),
        "row_keys": sorted(pressure["row_key"].unique()),
        "n_windows": int(len(pressure)),
    }


@dataclass(frozen=True)
class WindowPair:
    """One concrete (cell, reference) pair for a window-unit (§9.4) contrast."""

    contrast: Contrast
    family: str
    cell_row_key: str
    reference_row_key: str
    datasets: tuple[str, ...]


#: Window pairs must additionally agree on the §6.1 pressure coordinates —
#: contrasting different (r, λ) grid points would confound the slot.
_WINDOW_PAIR_MATCH_AXES: tuple[str, ...] = (*_PAIR_MATCH_AXES, "budget_r", "rate_frac")


def select_window_pairs(
    index: pd.DataFrame, contrast: Contrast
) -> tuple[list[WindowPair], list[str]]:
    """Concrete pairs for a window-unit baseline-pair contrast (all legs).

    Unlike the per-query selector, an absent leg is NOT an error: pressure
    cells are run-scoped, so a run without them yields a labeled skip reason
    instead. Ambiguous pairs (multiple cells per axes-slot) still fail loud.
    """
    assert contrast.baseline_a is not None and contrast.baseline_b is not None
    legs: list[tuple[str, str, str]] = [
        (contrast.baseline_a, contrast.baseline_b, contrast.family)
    ]
    legs.extend(
        (leg.baseline_a, leg.baseline_b, leg.family) for leg in contrast.extra_legs
    )
    pairs: list[WindowPair] = []
    skip_reasons: list[str] = []
    for baseline_a, baseline_b, family in legs:
        fam_rows = index[index["family"] == family]
        cells = fam_rows[fam_rows["baseline"] == baseline_a]
        refs = fam_rows[fam_rows["baseline"] == baseline_b]
        if cells.empty or refs.empty:
            skip_reasons.append(
                f"contrast #{contrast.id} leg {baseline_a}-vs-{baseline_b} "
                f"(family {family}): no loaded-window cells for "
                f"{'both sides' if cells.empty and refs.empty else (baseline_a if cells.empty else baseline_b)} "
                "in this run"
            )
            continue
        ref_groups = {
            key: grp
            for key, grp in refs.groupby(list(_WINDOW_PAIR_MATCH_AXES), dropna=False)
        }
        matched = False
        for key, cell_grp in cells.groupby(list(_WINDOW_PAIR_MATCH_AXES), dropna=False):
            ref_grp = ref_groups.get(key)
            if ref_grp is None:
                continue
            cell_keys = sorted(cell_grp["row_key"].unique())
            ref_keys = sorted(ref_grp["row_key"].unique())
            if len(cell_keys) != 1 or len(ref_keys) != 1:
                raise AnalysisError(
                    f"contrast #{contrast.id} leg {baseline_a}-vs-{baseline_b}: "
                    f"axes {dict(zip(_WINDOW_PAIR_MATCH_AXES, key))} select "
                    f"multiple cells per side (cells={cell_keys}, "
                    f"refs={ref_keys}) — the pair is ambiguous; refusing to guess"
                )
            datasets = tuple(
                sorted(
                    set(cell_grp["dataset"].unique())
                    & set(ref_grp["dataset"].unique())
                )
            )
            if not datasets:
                continue
            matched = True
            pairs.append(
                WindowPair(
                    contrast=contrast,
                    family=family,
                    cell_row_key=cell_keys[0],
                    reference_row_key=ref_keys[0],
                    datasets=datasets,
                )
            )
        if not matched:
            skip_reasons.append(
                f"contrast #{contrast.id} leg {baseline_a}-vs-{baseline_b} "
                f"(family {family}): {baseline_a} and {baseline_b} cells never "
                f"share the same {'/'.join(_WINDOW_PAIR_MATCH_AXES)} axes+coords "
                "and a dataset in this run — no legal pair"
            )
    return pairs, skip_reasons


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


#: §9.1: primaries are NEVER pooled/corrected across datasets (full alpha per
#: dataset; the registered pass/fail rule is gatekeeping.evaluate_chain's
#: intra-set rule, not a Holm correction — see stats['gatekeeping'] for the
#: executed chain). The Holm shown for every other tier here is
#: across-DATASET only — a labeled diagnostic; the registered §9.3
#: Holm-WITHIN-FAMILY correction (sibling contrasts sharing
#: group×metric×dataset per families.compile_family_map) is executed by
#: gatekeeping.evaluate_chain and reported in stats['gatekeeping'].
_PRIMARY_CORRECTION_LABEL = (
    "none (primary tier, full alpha per dataset, §9.1 co-primary set — "
    "cross-dataset pooling/correction PROHIBITED; the registered pass/fail "
    "rule is gatekeeping.evaluate_chain's intra-set rule, executed in "
    "stats['gatekeeping'])"
)
_DIAGNOSTIC_HOLM_LABEL = (
    "holm across datasets within contrast × metric (DIAGNOSTIC ONLY — the "
    "registered §9.3 Holm-within-family correction pools sibling contrasts "
    "sharing group×metric×dataset via families.compile_family_map and is "
    "executed by gatekeeping.evaluate_chain in stats['gatekeeping'])"
)


def compute_pair_stats(
    per_query: pd.DataFrame, pair: ResolvedPair, metric: str
) -> dict[str, Any]:
    """Per-dataset test + W/L/T for one pair.

    The test routes on the row's registered unit (§9.4): the binary §8.5 Y
    predicate goes through ``mcnemar_binary`` (exact binomial on discordant
    pairs); every other per-query metric goes through the paired Wilcoxon.
    Multiplicity correction depends on tier (§9.1/§9.3): primaries carry NO
    cross-dataset correction; everything else carries a diagnostic
    across-dataset Holm (see ``_DIAGNOSTIC_HOLM_LABEL``).
    """
    higher_is_better = _metric_direction(metric)
    if metric not in per_query.columns:
        raise AnalysisError(
            f"metric {metric!r} appears in no requests.jsonl/qa_evidence.jsonl "
            f"record for contrast #{pair.contrast.id} "
            f"({pair.cell_row_key} vs {pair.reference_row_key})"
        )
    unit = (
        "binary"
        if pair.contrast.unit == "per_query" and metric == PREDICATE_METRIC
        else pair.contrast.unit
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
        triple = win_loss_tie(a, b, higher_is_better=higher_is_better)
        if unit == "binary":
            mcnemar = mcnemar_binary(a, b, alternative="two-sided")
            row: dict[str, Any] = {
                "dataset": dataset,
                "n_pairs": mcnemar.n_pairs,
                "n_dropped_nan": int(n_dropped),
                "median_delta": float(np.median(a - b)),
                "p_value": mcnemar.p_value,
                "n_11": mcnemar.n_11,
                "n_00": mcnemar.n_00,
                "n_10": mcnemar.n_10,
                "n_01": mcnemar.n_01,
                "n_discordant": mcnemar.n_discordant,
                "proportion_diff": mcnemar.proportion_diff,
                "wins": triple.wins,
                "losses": triple.losses,
                "ties": triple.ties,
            }
        else:
            wilcoxon = paired_wilcoxon(a, b, alternative="two-sided")
            row = {
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
        per_dataset.append(row)

    if pair.contrast.tier == "primary":
        for row in per_dataset:
            row["p_holm_across_datasets"] = None
        correction_label = _PRIMARY_CORRECTION_LABEL
    else:
        adjusted = holm([row["p_value"] for row in per_dataset])
        for row, p_adj in zip(per_dataset, adjusted):
            row["p_holm_across_datasets"] = float(p_adj)
        correction_label = _DIAGNOSTIC_HOLM_LABEL

    return {
        "contrast_id": pair.contrast.id,
        "name": pair.contrast.name,
        "tier": pair.contrast.tier,
        "cell_baseline": pair.contrast.baseline_a,
        "reference_baseline": pair.contrast.baseline_b,
        "cell_row_key": pair.cell_row_key,
        "reference_row_key": pair.reference_row_key,
        "metric": metric,
        "unit": unit,
        "test": "mcnemar_binary" if unit == "binary" else "paired_wilcoxon",
        "higher_is_better": higher_is_better,
        "registered_sidedness": pair.contrast.sidedness,
        "test_sidedness": "two-sided (driver policy: conservative superset of "
        "any correctly-directed one-sided registration)",
        "correction": correction_label,
        "per_dataset": per_dataset,
    }


def compute_window_pair_stats(
    per_query: pd.DataFrame, pair: WindowPair, metric: str
) -> dict[str, Any]:
    """Batch-means Welch contrast for one loaded-window pair (§9.4).

    Per dataset: per-window means of ``metric`` on each side feed
    ``tests_by_unit.batch_means_contrast`` (Welch t on window-level batch
    means — per-query pairing under load is PROHIBITED). Datasets with < 2
    windows on either side are labeled skips inside ``per_dataset``, never
    silently dropped. No W/L/T triple here: §8.13's triple is a per-query
    mandate and window means are unpaired across cells.
    """
    higher_is_better = _metric_direction(metric)
    if metric not in per_query.columns:
        raise AnalysisError(
            f"metric {metric!r} appears in no requests.jsonl/qa_evidence.jsonl "
            f"record for window contrast #{pair.contrast.id} "
            f"({pair.cell_row_key} vs {pair.reference_row_key})"
        )
    per_dataset: list[dict[str, Any]] = []
    for dataset in pair.datasets:
        sub = per_query[per_query["dataset"] == dataset]
        means_a = (
            sub[sub["row_key"] == pair.cell_row_key]
            .groupby("window_key", observed=True)[metric]
            .mean()
            .dropna()
        )
        means_b = (
            sub[sub["row_key"] == pair.reference_row_key]
            .groupby("window_key", observed=True)[metric]
            .mean()
            .dropna()
        )
        if len(means_a) < 2 or len(means_b) < 2:
            per_dataset.append(
                {
                    "dataset": dataset,
                    "skipped": (
                        f"needs >= 2 windows per side for a Welch variance "
                        f"estimate; got cell={len(means_a)}, "
                        f"reference={len(means_b)}"
                    ),
                }
            )
            continue
        result = batch_means_contrast(
            means_a.to_numpy(dtype=float),
            means_b.to_numpy(dtype=float),
            alternative="two-sided",
        )
        per_dataset.append(
            {
                "dataset": dataset,
                "n_windows_cell": result.n_windows_a,
                "n_windows_reference": result.n_windows_b,
                "mean_cell": result.mean_a,
                "mean_reference": result.mean_b,
                "mean_diff": result.mean_diff,
                "statistic": result.statistic,
                "df": result.df,
                "p_value": result.p_value,
                "ci95_low": result.ci95_low,
                "ci95_high": result.ci95_high,
            }
        )

    computed = [row for row in per_dataset if "p_value" in row]
    if pair.contrast.tier == "primary":
        for row in computed:
            row["p_holm_across_datasets"] = None
        correction_label = _PRIMARY_CORRECTION_LABEL
    else:
        adjusted = holm([row["p_value"] for row in computed]) if computed else []
        for row, p_adj in zip(computed, adjusted):
            row["p_holm_across_datasets"] = float(p_adj)
        correction_label = _DIAGNOSTIC_HOLM_LABEL

    return {
        "contrast_id": pair.contrast.id,
        "name": pair.contrast.name,
        "tier": pair.contrast.tier,
        "family": pair.family,
        "cell_baseline": pair.contrast.baseline_a,
        "reference_baseline": pair.contrast.baseline_b,
        "cell_row_key": pair.cell_row_key,
        "reference_row_key": pair.reference_row_key,
        "metric": metric,
        "unit": "window",
        "test": "batch_means_welch_t (tests_by_unit.batch_means_contrast, §9.4)",
        "higher_is_better": higher_is_better,
        "registered_sidedness": pair.contrast.sidedness,
        "test_sidedness": "two-sided (driver policy: conservative superset of "
        "any correctly-directed one-sided registration)",
        "correction": correction_label,
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
# Confirmatory preconditions: §9.7 calibration artifact + §9.10 ledger seal
# ---------------------------------------------------------------------------


class CalibrationGateError(AnalysisError):
    """The §9.7 calibration precondition failed (absent/invalid/failing report)."""


def load_calibration_report(path: Path) -> CalibrationReport:
    """Load a §9.7 calibration-report JSON artifact (fail closed).

    Expected schema = the ``src.analysis.stats.calibration.CalibrationReport``
    dataclass tree serialized as JSON: top-level ``seed`` / ``n_observations``
    / ``aa`` (AAResult fields) / optional ``injections`` (InjectionResult
    fields each). This is the same report object ``prereg.py`` gates
    registration on — the confirmatory look reuses its schema and refusal
    pattern (§9.13: the registered SHA must be the calibrated one).
    """
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise CalibrationGateError(
            f"calibration report not found: {path} — confirmatory mode "
            "requires the §9.7 calibration artifact (A/A + effect-injection "
            "operating characteristics of the exact registered code)"
        ) from None
    except json.JSONDecodeError as exc:
        raise CalibrationGateError(
            f"calibration report is not valid JSON: {path} ({exc})"
        ) from exc
    if not isinstance(raw, dict):
        raise CalibrationGateError(
            f"calibration report root must be an object: {path}"
        )
    try:
        aa = AAResult(
            n_splits=int(raw["aa"]["n_splits"]),
            alpha=float(raw["aa"]["alpha"]),
            n_rejections=int(raw["aa"]["n_rejections"]),
            fp_rate=float(raw["aa"]["fp_rate"]),
            ci_low=float(raw["aa"]["ci_low"]),
            ci_high=float(raw["aa"]["ci_high"]),
        )
        injections = tuple(
            InjectionResult(
                effect_size=float(inj["effect_size"]),
                kind=inj["kind"],
                n_splits=int(inj["n_splits"]),
                alpha=float(inj["alpha"]),
                n_rejections=int(inj["n_rejections"]),
                power=float(inj["power"]),
                ci_low=float(inj["ci_low"]),
                ci_high=float(inj["ci_high"]),
                target_power=(
                    float(inj["target_power"])
                    if inj.get("target_power") is not None
                    else None
                ),
            )
            for inj in raw.get("injections", ())
        )
        report = CalibrationReport(
            seed=int(raw["seed"]),
            n_observations=int(raw["n_observations"]),
            aa=aa,
            injections=injections,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationGateError(
            f"calibration report {path} does not match the "
            "CalibrationReport schema (seed / n_observations / aa{n_splits, "
            "alpha, n_rejections, fp_rate, ci_low, ci_high} / injections[...]): "
            f"{exc!r}"
        ) from exc
    return report


def check_calibration(report: CalibrationReport, source: Path) -> dict[str, Any]:
    """§9.7 gate, mirroring ``prereg.assemble_preregistration``'s refusals.

    Returns the summary recorded into stats.json when the gate passes.
    """
    if not report.aa.approximates_nominal:
        raise CalibrationGateError(
            f"calibration A/A FAILED (FP rate CI [{report.aa.ci_low:.4f}, "
            f"{report.aa.ci_high:.4f}] excludes nominal α={report.aa.alpha:g}) "
            f"in {source} — the confirmatory look is BLOCKED until the "
            "machinery passes §9.7 (same rule prereg.py applies to "
            "registration)"
        )
    failed = [i for i in report.injections if i.meets_target is False]
    if failed:
        raise CalibrationGateError(
            f"calibration injection targets missed at effects "
            f"{[i.effect_size for i in failed]} in {source} — the "
            "confirmatory look is BLOCKED (§9.7/§9.13)"
        )
    return {
        "path": str(source),
        "seed": report.seed,
        "n_observations": report.n_observations,
        "aa_fp_rate": report.aa.fp_rate,
        "aa_approximates_nominal": report.aa.approximates_nominal,
        "n_injections": len(report.injections),
        "verdict": "PASS",
    }


def verify_run_ledger(run_dir: Path) -> dict[str, Any]:
    """§9.10 hard precondition of confirmatory mode: the raw tree must verify
    against its sealed content-hash ledger — analyzed data is provably the
    sealed data. Any mismatch (or a missing/tampered ledger) refuses."""
    ledger_path = run_dir / LEDGER_NAME
    try:
        entries = read_ledger(ledger_path)
        mismatches = verify_ledger(ledger_path, run_dir)
    except LedgerError as exc:
        raise AnalysisError(
            f"LEDGER PRECONDITION FAILED (§9.10): {exc} — confirmatory "
            "analysis runs only on sealed, verifiable data"
        ) from exc
    if mismatches:
        shown = "\n".join(f"  {line}" for line in mismatches[:20])
        raise AnalysisError(
            f"LEDGER PRECONDITION FAILED (§9.10): {len(mismatches)} artifact "
            f"mismatch(es) against {ledger_path}:\n{shown}\n"
            "— the raw tree is not the sealed tree; refusing the "
            "confirmatory look"
        )
    return {
        "path": LEDGER_NAME,
        "entries": len(entries),
        "mismatches": 0,
        "verified": True,
    }


# ---------------------------------------------------------------------------
# §9.8 blinding integration (sealed map consumer)
# ---------------------------------------------------------------------------


def sealed_map_path(run_dir: Path) -> Path:
    return run_dir / BLINDING_DIRNAME / SEALED_MAP_NAME


def load_blinding_state(run_dir: Path) -> dict[str, Any] | None:
    """Load + tamper-verify the run's sealed arm map, if one exists.

    Returns None when no sealed map exists. Uses the blinding module's own
    seal loader so tamper detection stays in ONE place; a tampered seal
    raises (never analyzed around).
    """
    path = sealed_map_path(run_dir)
    if not path.is_file():
        return None
    # Same-package seal verification (map_sha256 self-hash) — the loader is
    # module-internal by convention but IS the single verification authority.
    try:
        sealed = blinding_mod._load_sealed(path)  # noqa: SLF001
    except BlindingError as exc:
        raise AnalysisError(
            f"blinding refusal (§9.8): {exc} — a tampered/unreadable seal "
            "cannot certify a blinded analysis"
        ) from exc
    return dict(sealed)


def _blind_value(mapping: Mapping[str, str], value: str) -> str:
    """Real->blind code for one label; unknown labels fail loud (a label the
    seal never covered would leak through a blinded output)."""
    try:
        return mapping[value]
    except KeyError:
        raise AnalysisError(
            f"blinding: label {value!r} is not in the sealed map — the seal "
            "does not cover this run's labels; re-seal or remove the map"
        ) from None


def apply_blinding_to_entry(
    entry: dict[str, Any], mapping: Mapping[str, str], index: pd.DataFrame
) -> dict[str, Any]:
    """Mask arm-revealing labels in one contrast entry (§9.8 blinded output).

    The pipeline resolves contrasts on real labels internally (the registered
    selectors are baseline ids); the OUTPUT is what stays scrambled until the
    logged unblinding. Row keys and baseline ids both reveal the arm, so both
    are replaced by the blind code of the cell's arm.
    """
    masked = dict(entry)
    for key_field, baseline_field in (
        ("cell_row_key", "cell_baseline"),
        ("reference_row_key", "reference_baseline"),
    ):
        row_key = entry[key_field]
        arms = index.loc[index["row_key"] == row_key, "arm"].dropna().unique()
        arm = str(arms[0]) if len(arms) else row_key.split("|", 1)[0]
        code = _blind_value(mapping, arm)
        masked[key_field] = f"BLINDED:{code}"
        masked[baseline_field] = f"BLINDED:{code}"
    return masked


# ---------------------------------------------------------------------------
# §9.3 family map + Dmitrienko serial gatekeeping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilyContext:
    """The compiled §9.3 registered test table, scoped to this run."""

    group: str
    datasets: tuple[str, ...]
    table: pd.DataFrame
    #: membership set: (contrast_id, metric, dataset) rows for this run's group.
    keys: frozenset[tuple[int, str, str]]


def build_family_context(
    index: pd.DataFrame, metrics: Sequence[str], alpha: float
) -> FamilyContext | None:
    """Compile ``families.compile_family_map`` for this run's group/datasets.

    Returns None (a labeled absence recorded in stats.json) when the run has
    no charter family-map dataset at all — confirmatory mode refuses on that.
    """
    model = str(index["model"].iloc[0])
    group = GROUP_OF_MODEL.get(model)
    if group is None:
        raise AnalysisError(
            f"model {model!r} has no §7.6.1 campaign group — cannot compile "
            f"the §9.3 family map (roster: {sorted(GROUP_OF_MODEL)})"
        )
    datasets = tuple(sorted(set(index["dataset"].unique()) & KNOWN_DATASETS))
    if not datasets:
        return None
    try:
        table = compile_family_map(datasets, metrics=tuple(metrics), alpha=alpha)
    except FamilyMapError as exc:
        raise AnalysisError(f"family map compilation failed (§9.3): {exc}") from exc
    scoped = table[table["group"] == group]
    keys = frozenset(
        (int(r.contrast_id), str(r.metric), str(r.dataset))
        for r in scoped.itertuples(index=False)
    )
    return FamilyContext(group=group, datasets=datasets, table=scoped, keys=keys)


def _primary_endpoint(contrast_id: int) -> str:
    return f"contrast-{contrast_id}"


def run_gatekeeping(
    contrast_stats: Sequence[Mapping[str, Any]],
    family_ctx: FamilyContext | None,
    *,
    alpha: float = 0.05,
    intra_set_rule: str = "all-datasets",
) -> dict[str, Any]:
    """Execute the §9.3 Dmitrienko serial chain + Holm-within-family gating.

    Primaries: every computed primary-tier per-dataset p becomes a
    ``PrimaryOutcome`` under endpoint ``contrast-<id>``; the per-dataset
    co-primary SET spans dataset × metric (§9.1's metric pair are co-primary),
    so the outcome's dataset key is ``<dataset>|<metric>``. The chain runs in
    the registered ``families.PRIMARY_CHAIN_ORDER`` restricted to the
    endpoints this run computed; ``chain_complete`` is False (and loudly
    listed) whenever any registered chain endpoint is missing — an incomplete
    chain cannot license the registered confirmatory claims.

    Secondaries: computed secondary-tier rows whose (dataset, metric) has a
    matching headline (#4) outcome join their §9.3 family
    ``<group>|<metric>|<dataset>`` and receive the REGISTERED
    Holm-within-family correction from ``gatekeeping.evaluate_chain`` (sibling
    contrasts pooled per family). Rows with no computable upstream are listed
    under ``ungated`` — never silently dropped.
    """
    primaries: list[PrimaryOutcome] = []
    secondaries: list[SecondaryOutcome] = []
    ungated: list[dict[str, Any]] = []
    computed_primary_ids: set[int] = set()
    headline_keys: set[str] = set()

    for entry in contrast_stats:
        if entry["tier"] != "primary":
            continue
        cid = int(entry["contrast_id"])
        computed_primary_ids.add(cid)
        for row in entry["per_dataset"]:
            if "p_value" not in row:
                continue
            key = f"{row['dataset']}|{entry['metric']}"
            primaries.append(
                PrimaryOutcome(
                    endpoint=_primary_endpoint(cid),
                    dataset=key,
                    p_value=float(row["p_value"]),
                )
            )
            if cid == _SECONDARY_UPSTREAM_ID:
                headline_keys.add(key)

    if not primaries:
        return {
            "skipped": (
                "no primary-tier contrast was computed in this invocation — "
                "the §9.3 serial chain needs at least the headline (#4) "
                "outcomes to gate anything"
            )
        }

    group = family_ctx.group if family_ctx is not None else "?"
    for entry in contrast_stats:
        if entry["tier"] != "secondary":
            continue
        for row in entry["per_dataset"]:
            if "p_value" not in row:
                continue
            key = f"{row['dataset']}|{entry['metric']}"
            if key not in headline_keys:
                ungated.append(
                    {
                        "contrast_id": entry["contrast_id"],
                        "metric": entry["metric"],
                        "dataset": row["dataset"],
                        "reason": (
                            f"upstream primary #{_SECONDARY_UPSTREAM_ID} has no "
                            f"computed outcome on ({row['dataset']}, "
                            f"{entry['metric']}) — the gate cannot open or "
                            "close; reported raw, unregistered"
                        ),
                    }
                )
                continue
            secondaries.append(
                SecondaryOutcome(
                    contrast=f"#{entry['contrast_id']} {entry['name']}",
                    family_id=f"{group}|{entry['metric']}|{row['dataset']}",
                    upstream=_primary_endpoint(_SECONDARY_UPSTREAM_ID),
                    dataset=key,
                    p_value=float(row["p_value"]),
                )
            )

    registered_order = [
        _primary_endpoint(cid)
        for cid in PRIMARY_CHAIN_ORDER
        if cid in computed_primary_ids
    ]
    missing_endpoints = [
        _primary_endpoint(cid)
        for cid in PRIMARY_CHAIN_ORDER
        if cid not in computed_primary_ids
    ]
    try:
        trace: GatekeepingTrace = evaluate_chain(
            primaries,
            secondaries,
            alpha=alpha,
            primary_order=registered_order,
            intra_set_rule=intra_set_rule,  # type: ignore[arg-type]
        )
    except GatekeepingError as exc:
        raise AnalysisError(f"gatekeeping chain refused (§9.3): {exc}") from exc

    return {
        "alpha": trace.alpha,
        "intra_set_rule": trace.intra_set_rule,
        "primary_chain_order_registered": [
            _primary_endpoint(cid) for cid in PRIMARY_CHAIN_ORDER
        ],
        "primary_chain_order_executed": list(trace.primary_order or ()),
        "chain_complete": not missing_endpoints,
        "missing_primary_endpoints": missing_endpoints,
        "missing_endpoints_note": (
            None
            if not missing_endpoints
            else (
                "INCOMPLETE CHAIN: the registered Dmitrienko serial sequence "
                f"(families.PRIMARY_CHAIN_ORDER) includes {missing_endpoints} "
                "which this invocation could not compute — downstream "
                "endpoints of a missing gate cannot be claimed confirmatory"
            )
        ),
        "primaries": [
            {
                "endpoint": p.endpoint,
                "dataset_metric": p.dataset,
                "p_value": p.p_value,
                "alpha": p.alpha,
                "passed": p.passed,
                "status": p.status,
            }
            for p in trace.primaries
        ],
        "secondaries": [
            {
                "contrast": s.contrast,
                "family_id": s.family_id,
                "dataset_metric": s.dataset,
                "p_value": s.p_value,
                "status": s.status,
                "p_holm_within_family": s.p_holm,
                "significant": s.significant,
            }
            for s in trace.secondaries
        ],
        "events": [
            {
                "family_id": e.family_id,
                "upstream": e.upstream,
                "dataset_metric": e.dataset,
                "upstream_p": e.upstream_p,
                "opened": e.opened,
                "reason": e.reason,
            }
            for e in trace.events
        ],
        "ungated": ungated,
    }


# ---------------------------------------------------------------------------
# §9.5 conditional TOST for the family map's declared equivalence legs
# ---------------------------------------------------------------------------


def compute_equivalence(
    per_query_loader: Any,
    index: pd.DataFrame,
    family_ctx: FamilyContext | None,
    *,
    metric: str | None,
    margin: float | None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Conditional two-layer TOST (+ ROPE line) for the declared NONE legs.

    The §9.3 family map declares TOST rows for the three pre-registered NONE
    fingerprints (#13: recompute / offload / distribute). Each is either
    COMPUTED (policy-vs-none cell pair found, ``policy_event`` mask present,
    registered margin supplied) or listed as a labeled skip with the exact
    missing ingredient — declared legs are never silently absent.

    ``per_query_loader(row_keys) -> pd.DataFrame`` defers I/O so no window is
    read unless a leg is actually computable.
    """
    declared = [
        {"policy": policy, "correction": corr, "sidedness": sided, "predicted": pred}
        for policy, corr, sided, pred in FINGERPRINT_SUB_HYPOTHESES
        if corr == "tost"
    ]
    section: dict[str, Any] = {
        "declared_legs": declared,
        "source": "families.FINGERPRINT_SUB_HYPOTHESES (§9.3 TOST rows)",
        "results": [],
        "skipped": [],
    }

    def skip(policy: str, reason: str) -> None:
        section["skipped"].append({"policy": policy, "reason": reason})

    if family_ctx is None:
        for leg in declared:
            skip(leg["policy"], "no §9.3 family-map dataset in this run")
        return section
    if margin is None or metric is None:
        for leg in declared:
            skip(
                leg["policy"],
                "no registered §9.5 margin/metric supplied "
                "(--tost-margin + --equivalence-metric)",
            )
        return section

    pressure = index[index["family"].isin(sorted(PRESSURE_FAMILIES))]
    for leg in declared:
        policy = leg["policy"]
        if policy not in _TOST_POLICY_AXIS_LEGS:
            skip(
                policy,
                "topology-slot leg (distribute): policy-axis pairing does not "
                "apply; DIST-topology pairing is not implemented in this driver",
            )
            continue
        cells = pressure[pressure["policy"] == policy]
        refs = pressure[pressure["policy"] == "none"]
        if cells.empty:
            skip(policy, f"no policy={policy!r} pressure cells in this run")
            continue
        match_axes = [
            "arm", "retriever", "topology", "engine", "model", "family",
            "budget_r", "rate_frac",
        ]
        ref_groups = {
            key: grp for key, grp in refs.groupby(match_axes, dropna=False)
        }
        found_pair = False
        for key, cell_grp in cells.groupby(match_axes, dropna=False):
            ref_grp = ref_groups.get(key)
            if ref_grp is None:
                continue
            cell_keys = sorted(cell_grp["row_key"].unique())
            ref_keys = sorted(ref_grp["row_key"].unique())
            if len(cell_keys) != 1 or len(ref_keys) != 1:
                raise AnalysisError(
                    f"equivalence leg {policy!r}: ambiguous cell pair "
                    f"(cells={cell_keys}, refs={ref_keys}); refusing to guess"
                )
            cell_key, ref_key = cell_keys[0], ref_keys[0]
            datasets = sorted(
                set(cell_grp["dataset"]) & set(ref_grp["dataset"])
                & set(family_ctx.datasets)
            )
            if not datasets:
                continue
            found_pair = True
            per_query = per_query_loader({cell_key, ref_key})
            if metric not in per_query.columns:
                skip(
                    policy,
                    f"metric {metric!r} absent from the pair's per-query "
                    f"records ({cell_key} vs {ref_key})",
                )
                continue
            if POLICY_EVENT_COLUMN not in per_query.columns:
                skip(
                    policy,
                    f"no {POLICY_EVENT_COLUMN!r} column in the pair's "
                    "per-query records — the §9.5 CONDITIONAL population "
                    "needs the S2 policy-event mask; an unconditional TOST "
                    "passes trivially and proves nothing",
                )
                continue
            for dataset in datasets:
                sub = per_query[per_query["dataset"] == dataset]
                wide = (
                    sub.groupby(["example_id", "row_key"], observed=True)[metric]
                    .mean()
                    .unstack("row_key")
                )
                mask_wide = (
                    sub.groupby(["example_id", "row_key"], observed=True)[
                        POLICY_EVENT_COLUMN
                    ]
                    .max()
                    .unstack("row_key")
                )
                if cell_key not in wide.columns or ref_key not in wide.columns:
                    skip(
                        policy,
                        f"{dataset}: no {metric!r} values on both sides "
                        f"({cell_key} vs {ref_key})",
                    )
                    continue
                wide = wide.dropna(subset=[cell_key, ref_key])
                if wide.empty:
                    skip(policy, f"{dataset}: no overlapping example_id pairs")
                    continue
                a = wide[cell_key].to_numpy(dtype=float)
                b = wide[ref_key].to_numpy(dtype=float)
                mask = (
                    mask_wide.reindex(wide.index)[cell_key]
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                    > 0.0
                )
                tost = conditional_tost(a, b, mask, margin=margin, alpha=alpha)
                rope = rope_sensitivity(a, b, mask, rope=margin)
                section["results"].append(
                    {
                        "policy": policy,
                        "predicted": leg["predicted"],
                        "dataset": dataset,
                        "metric": metric,
                        "cell_row_key": cell_key,
                        "reference_row_key": ref_key,
                        "margin": tost.margin,
                        "n_total": tost.n_total,
                        "n_events": tost.n_events,
                        "n_discordant": tost.n_discordant,
                        "mean_diff": tost.mean_diff,
                        "p_tost": tost.p_tost,
                        "domain_verdict": tost.domain_verdict,
                        "dominance": tost.dominance,
                        "dominance_ci": [
                            tost.dominance_ci_low,
                            tost.dominance_ci_high,
                        ],
                        "dominance_verdict": tost.dominance_verdict,
                        "equivalent": tost.equivalent,
                        "rope_sensitivity": {
                            "p_left": rope.p_left,
                            "p_rope": rope.p_rope,
                            "p_right": rope.p_right,
                            "verdict": rope.verdict,
                            "note": "sensitivity LINE beside the TOST "
                            "conclusion, never the confirmatory gate (§9.5)",
                        },
                    }
                )
        if not found_pair:
            skip(
                policy,
                f"policy={policy!r} cells never share axes+coords+dataset "
                "with a policy='none' reference in this run",
            )
    return section


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def _fmt_effect(row: Mapping[str, Any]) -> str:
    """Cliff's δ (paired) for Wilcoxon rows; proportion-diff + discordant n
    for McNemar (binary) rows — the two tests report different effect sizes."""
    if "cliffs_delta_paired" in row:
        return f"{row['cliffs_delta_paired']:.3f}"
    return f"{row['proportion_diff']:.3f} (n_disc={row['n_discordant']})"


def _fmt_p_holm(row: Mapping[str, Any]) -> str:
    """``None`` for primary-tier rows (§9.1: no cross-dataset correction)."""
    p_holm = row.get("p_holm_across_datasets")
    return f"{p_holm:.3g}" if p_holm is not None else "n/a (primary tier, full α)"


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
        f"- driver sidedness policy: two-sided (paired Wilcoxon for continuous "
        "metrics, McNemar exact-binomial for the binary §8.5 predicate, "
        "batch-means Welch t for loaded windows; registered sidedness recorded "
        "per contrast); correction: none for primary-tier endpoints (full α "
        "per dataset, §9.1), diagnostic Holm across datasets otherwise — the "
        "registered §9.3 Holm-within-family correction is executed in the "
        "gatekeeping section below",
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
            f"{'higher' if entry['higher_is_better'] else 'lower'} is better  ·  "
            f"test: `{entry['test']}`"
        )
        lines.append("")
        if entry["unit"] == "window":
            lines.append(
                "| dataset | windows (cell/ref) | mean Δ (cell−ref) | 95% CI "
                "| p | p (Holm across datasets) |"
            )
            lines.append("|---|---|---|---|---|---|")
            for row in entry["per_dataset"]:
                if "skipped" in row:
                    lines.append(
                        f"| {row['dataset']} | — | — | — | SKIPPED: "
                        f"{row['skipped']} | — |"
                    )
                    continue
                lines.append(
                    f"| {row['dataset']} "
                    f"| {row['n_windows_cell']}/{row['n_windows_reference']} "
                    f"| {row['mean_diff']:.4g} "
                    f"| [{row['ci95_low']:.4g}, {row['ci95_high']:.4g}] "
                    f"| {row['p_value']:.3g} | {_fmt_p_holm(row)} |"
                )
        else:
            lines.append(
                "| dataset | n_pairs | median Δ (cell−ref) | W/L/T | effect | p | "
                "p (Holm across datasets) |"
            )
            lines.append("|---|---|---|---|---|---|---|")
            for row in entry["per_dataset"]:
                lines.append(
                    f"| {row['dataset']} | {row['n_pairs']} "
                    f"| {row['median_delta']:.4g} "
                    f"| {row['wins']}/{row['losses']}/{row['ties']} "
                    f"| {_fmt_effect(row)} "
                    f"| {row['p_value']:.3g} | {_fmt_p_holm(row)} |"
                )
        lines.append("")

    gate = stats.get("gatekeeping", {})
    lines.append("## Gatekeeping chain (§9.3 — Dmitrienko serial + Holm within family)")
    lines.append("")
    if "skipped" in gate:
        lines.append(f"SKIPPED: {gate['skipped']}")
    else:
        lines.append(
            f"- executed order: {' → '.join(gate['primary_chain_order_executed'])} "
            f"(registered: {' → '.join(gate['primary_chain_order_registered'])}; "
            f"chain {'COMPLETE' if gate['chain_complete'] else 'INCOMPLETE'})"
        )
        if gate.get("missing_endpoints_note"):
            lines.append(f"- **{gate['missing_endpoints_note']}**")
        lines.append(
            f"- intra-set rule: `{gate['intra_set_rule']}` at α={gate['alpha']:g}"
        )
        lines.append("")
        lines.append("| endpoint | dataset×metric | p | passed | status |")
        lines.append("|---|---|---|---|---|")
        for p in gate["primaries"]:
            lines.append(
                f"| {p['endpoint']} | {p['dataset_metric']} "
                f"| {p['p_value']:.3g} | {p['passed']} | {p['status']} |"
            )
        if gate["secondaries"]:
            lines.append("")
            lines.append(
                "| secondary | family | dataset×metric | p | p (Holm within "
                "family) | status | significant |"
            )
            lines.append("|---|---|---|---|---|---|---|")
            for s in gate["secondaries"]:
                p_holm_s = (
                    f"{s['p_holm_within_family']:.3g}"
                    if s["p_holm_within_family"] is not None
                    else "n/a (descriptive)"
                )
                lines.append(
                    f"| {s['contrast']} | {s['family_id']} "
                    f"| {s['dataset_metric']} | {s['p_value']:.3g} "
                    f"| {p_holm_s} | {s['status']} | {s['significant']} |"
                )
        lines.append("")
        lines.append(f"Gate events (auditable trace): {len(gate['events'])} — see stats.json.")
    lines.append("")

    equiv = stats.get("equivalence", {})
    lines.append("## Equivalence legs (§9.5 — conditional two-layer TOST + ROPE line)")
    lines.append("")
    for result in equiv.get("results", ()):
        lines.append(
            f"- `{result['policy']}` × {result['dataset']} [{result['metric']}]: "
            f"n_events={result['n_events']} (discordant {result['n_discordant']}), "
            f"p_TOST={result['p_tost']:.3g}, domain={result['domain_verdict']}, "
            f"dominance={result['dominance_verdict']} → "
            f"{'EQUIVALENT' if result['equivalent'] else 'not equivalent'} "
            f"(ROPE: p_rope={result['rope_sensitivity']['p_rope']:.3g}, "
            f"{result['rope_sensitivity']['verdict']})"
        )
    for skipped_leg in equiv.get("skipped", ()):
        lines.append(
            f"- `{skipped_leg['policy']}`: SKIPPED — {skipped_leg['reason']}"
        )
    lines.append("")

    blinding_info = stats.get("blinding", {})
    if blinding_info.get("sealed_map"):
        lines.append("## Blinding (§9.8)")
        lines.append("")
        lines.append(
            f"- sealed map: `{blinding_info['sealed_map']}` "
            f"(sha256 `{blinding_info.get('map_sha256')}`)"
        )
        if blinding_info.get("active"):
            lines.append(
                "- **LABELS SCRAMBLED** — arm-revealing labels masked, figures "
                "suppressed until the logged unblinding"
            )
        event = blinding_info.get("unblind_event")
        if event:
            lines.append(
                f"- UNBLINDED at {event['utc']} (logged to `{event['log']}`)"
            )
        lines.append("")

    skipped = stats["skipped"]
    if (
        skipped["pressure_rows"]
        or skipped["contrasts"]
        or skipped.get("confirmatory_exclusions")
    ):
        lines.append("## Skipped (labeled — never silently absent)")
        lines.append("")
        if skipped["pressure_rows"]:
            block = skipped["pressure_rows"]
            lines.append(
                f"**Pressure rows (F2/F3), {block['n_windows']} window(s) — "
                "batch-means only (§9.4), never per-query paired; not consumed "
                "by any computed contrast:**"
            )
            for key in block["row_keys"]:
                lines.append(f"- `{key}`")
            lines.append("")
        for entry in skipped["contrasts"]:
            lines.append(
                f"- contrast #{entry['contrast_id']} ({entry['name']}) "
                f"[{entry['label']}]: {entry['reason']}"
            )
        for excl in skipped.get("confirmatory_exclusions", ()):
            lines.append(
                f"- contrast #{excl['contrast_id']} × {excl['metric']} on "
                f"{excl['datasets']}: {excl['reason']}"
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


def _one_look_refusal_message(run_dir: Path, lock: dict[str, Any]) -> str:
    if lock.get("phase") == "IN_PROGRESS":
        detail = (
            "a confirmatory analysis is CURRENTLY IN PROGRESS (started "
            f"{lock.get('started_utc', '<unknown time>')} under registered "
            f"SHA {lock.get('registered_sha', '<unknown>')!r})"
        )
    else:
        detail = (
            "this run's confirmatory analysis ran at "
            f"{lock.get('locked_utc', '<unknown time>')} under registered "
            f"SHA {lock.get('registered_sha', '<unknown>')!r}"
        )
    return (
        f"ONE-LOOK REFUSAL (§9.11): {run_dir / LOCK_NAME} already exists "
        f"— {detail}. The campaign data is analyzed once; a second look "
        "invalidates the registration. Design-input re-renders remain "
        "available via --design-input."
    )


def _acquire_confirmatory_lock(run_dir: Path, registered_sha: str) -> None:
    """Atomically claim the run's single confirmatory look (§9.11).

    Uses ``O_CREAT | O_EXCL`` so two concurrent ``--confirmatory`` invocations
    cannot both pass a read-then-write race (the old ``read_lock`` once /
    ``write_lock`` unconditionally-at-the-end pattern): exactly one process's
    ``open()`` succeeds, the other gets ``FileExistsError`` and refuses
    immediately, before running any of the expensive pipeline. The placeholder
    written here is IN_PROGRESS-only; ``run_analysis`` removes it on any
    exception raised by *this* process after acquiring it (see its
    try/except), so a crash mid-pipeline never burns the run's one registered
    look — only ``write_lock`` finalizing to ``phase: DONE`` does that.
    """
    lock_path = run_dir / LOCK_NAME
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        lock = read_lock(run_dir)
        assert lock is not None  # FileExistsError => the file is there
        raise OneLookError(_one_look_refusal_message(run_dir, lock)) from None
    placeholder = {
        "phase": "IN_PROGRESS",
        "registered_sha": registered_sha,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        os.write(fd, (json.dumps(placeholder, indent=2) + "\n").encode("utf-8"))
    except BaseException:
        os.close(fd)
        lock_path.unlink(missing_ok=True)
        raise
    else:
        os.close(fd)


def _release_placeholder_lock(run_dir: Path) -> None:
    """Undo ``_acquire_confirmatory_lock`` after a failed pipeline attempt.

    Only removes the lock if it is STILL the IN_PROGRESS placeholder — never
    a lock some other state finalized — so this can't ever delete a genuine
    completed-look record.
    """
    lock_path = run_dir / LOCK_NAME
    try:
        current = read_lock(run_dir)
    except OneLookError:
        # Corrupt JSON: read_lock's own fail-closed policy owns this state;
        # do not delete evidence we cannot parse.
        return
    if current is not None and current.get("phase") == "IN_PROGRESS":
        lock_path.unlink(missing_ok=True)


def write_lock(
    run_dir: Path, registered_sha: str, analysis_dir: Path, stats_path: Path
) -> Path:
    """Finalize the §9.11 lock over the placeholder from
    ``_acquire_confirmatory_lock``: write to a temp file, then
    ``os.replace`` it into place — an atomic overwrite, so no reader ever
    observes a partially-written lock file."""
    lock_path = run_dir / LOCK_NAME
    payload = {
        "policy": "PUBLICATION.md §9.11 ONE-LOOK: this run's confirmatory "
        "analysis has been executed exactly once",
        "phase": "DONE",
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "registered_sha": registered_sha,
        "analysis_dir": analysis_dir.name,
        "stats_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
    }
    tmp_path = lock_path.with_name(f"{lock_path.name}.tmp-{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, lock_path)
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
    calibration_report: Path | None = None,
    tost_margin: float | None = None,
    equivalence_metric: str | None = None,
    alpha: float = 0.05,
) -> AnalysisResult:
    """Execute the pipeline; the CLI wraps this with the one-look flag checks.

    Confirmatory preconditions (checked BEFORE the §9.11 lock is acquired, so
    a refusal never touches the run's one-look budget): a registered SHA, a
    PASSING §9.7 calibration-report artifact, and a clean §9.10 ledger
    verification of the raw tree.
    """
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise AnalysisError(f"run directory does not exist: {run_dir}")
    for metric in metrics:
        _metric_direction(metric)  # fail before any I/O on unknown direction

    stamp = CONFIRMATORY_STAMP if mode == "confirmatory" else DESIGN_STAMP
    preconditions: dict[str, Any] = {
        "calibration": {"checked": False},
        "ledger": {"checked": False},
    }
    lock_acquired_here = False
    if mode == "confirmatory":
        if not registered_sha:
            raise OneLookError("confirmatory mode requires a registered SHA")
        if calibration_report is None:
            raise CalibrationGateError(
                "confirmatory mode requires --calibration-report: the §9.7 "
                "calibration artifact (A/A + effect-injection operating "
                "characteristics) is a hard precondition of the one "
                "registered look (§9.13 refusal pattern, as in prereg.py)"
            )
        report = load_calibration_report(calibration_report)
        preconditions["calibration"] = {
            "checked": True,
            **check_calibration(report, calibration_report),
        }
        preconditions["ledger"] = {"checked": True, **verify_run_ledger(run_dir)}
        # Atomic acquire-before-pipeline (§9.11): closes the TOCTOU window
        # where two concurrent --confirmatory invocations could both pass a
        # read-only check and both run the full (expensive) pipeline, with
        # the second writer silently clobbering the first's lock record.
        _acquire_confirmatory_lock(run_dir, registered_sha)
        lock_acquired_here = True
    else:
        if calibration_report is not None:
            # Allowed in design-input as a dry-run of the gate; recorded only.
            report = load_calibration_report(calibration_report)
            preconditions["calibration"] = {
                "checked": True,
                **check_calibration(report, calibration_report),
            }
        if read_lock(run_dir) is not None:
            print(
                f"WARNING: {run_dir / LOCK_NAME} exists — the confirmatory look "
                "is spent. These design-input outputs must never be quoted as "
                "confirmatory.",
                file=sys.stderr,
            )

    try:
        index = load_index(run_dir)

        # §9.8: a sealed arm map means the pipeline output stays scrambled
        # until the one-time logged unblinding (which the confirmatory look
        # performs). Tampered seals raise here, before anything is computed.
        blinding_state = load_blinding_state(run_dir)
        blinding_active = (
            blinding_state is not None
            and blinding_state.get("unblinded_utc") is None
            and mode == "design-input"
        )
        blinding_section: dict[str, Any] = {
            "sealed_map": (
                f"{BLINDING_DIRNAME}/{SEALED_MAP_NAME}"
                if blinding_state is not None
                else None
            ),
            "map_sha256": (
                blinding_state.get("map_sha256") if blinding_state else None
            ),
            "active": blinding_active,
            "unblind_event": None,
            "note": (
                "labels scrambled per §9.8 — figures suppressed and "
                "arm-revealing labels masked until the logged unblinding"
                if blinding_active
                else None
            ),
        }

        computable, skipped_contrasts = resolve_contrasts(contrast_ids)
        per_query_contrasts = [
            c
            for c in computable
            if c.unit == "per_query" and c.family not in PRESSURE_FAMILIES
        ]
        window_contrasts = [c for c in computable if c.unit == "window"]

        family_ctx = build_family_context(index, metrics, alpha)
        if mode == "confirmatory" and family_ctx is None:
            raise AnalysisError(
                "confirmatory refusal (§9.3): none of this run's datasets "
                f"appear in the registered family map ({sorted(KNOWN_DATASETS)}) "
                "— no test may run that is not a row in the table"
            )

        pairs: list[ResolvedPair] = []
        for contrast in per_query_contrasts:
            pairs.extend(select_contrast_pairs(index, contrast))

        window_pairs: list[WindowPair] = []
        for contrast in window_contrasts:
            w_pairs, w_reasons = select_window_pairs(index, contrast)
            window_pairs.extend(w_pairs)
            for reason in w_reasons:
                skipped_contrasts.append(
                    SkippedContrast(
                        contrast_id=contrast.id,
                        name=contrast.name,
                        label="NO-WINDOW-PAIR-IN-RUN",
                        reason=reason,
                    )
                )

        wanted_keys = {
            k for p in pairs for k in (p.cell_row_key, p.reference_row_key)
        } | {
            k
            for p in window_pairs
            for k in (p.cell_row_key, p.reference_row_key)
        }
        per_query = (
            load_per_query(run_dir, index, wanted_keys)
            if wanted_keys
            else pd.DataFrame()
        )

        def _in_family(contrast_id: int, metric: str, dataset: str) -> bool:
            if family_ctx is None:
                return False
            return (contrast_id, metric, dataset) in family_ctx.keys

        contrast_stats: list[dict[str, Any]] = []
        confirmatory_exclusions: list[dict[str, Any]] = []

        def _family_filter(
            contrast_id: int, metric: str, datasets: tuple[str, ...]
        ) -> tuple[str, ...]:
            """Confirmatory §9.3 discipline: only family-map rows are tested."""
            if mode != "confirmatory":
                return datasets
            kept = tuple(d for d in datasets if _in_family(contrast_id, metric, d))
            dropped = [d for d in datasets if d not in kept]
            if dropped:
                confirmatory_exclusions.append(
                    {
                        "contrast_id": contrast_id,
                        "metric": metric,
                        "datasets": dropped,
                        "reason": "not a §9.3 family-map row — no test runs "
                        "that is not a row in the table",
                    }
                )
            return kept

        for pair in pairs:
            for metric in metrics:
                allowed = _family_filter(pair.contrast.id, metric, pair.datasets)
                if not allowed:
                    continue
                entry = compute_pair_stats(
                    per_query, dc_replace(pair, datasets=allowed), metric
                )
                for row in entry["per_dataset"]:
                    row["in_family_map"] = _in_family(
                        entry["contrast_id"], metric, row["dataset"]
                    )
                contrast_stats.append(entry)
        for w_pair in window_pairs:
            for metric in metrics:
                allowed = _family_filter(
                    w_pair.contrast.id, metric, w_pair.datasets
                )
                if not allowed:
                    continue
                entry = compute_window_pair_stats(
                    per_query, dc_replace(w_pair, datasets=allowed), metric
                )
                for row in entry["per_dataset"]:
                    row["in_family_map"] = _in_family(
                        entry["contrast_id"], metric, row["dataset"]
                    )
                contrast_stats.append(entry)

        consumed = frozenset(
            k
            for p in window_pairs
            for k in (p.cell_row_key, p.reference_row_key)
        )
        pressure_block = pressure_row_skip(index, consumed)

        # §9.3 wiring: the registered chain + Holm-within-family corrections.
        gatekeeping_section = run_gatekeeping(
            contrast_stats, family_ctx, alpha=alpha
        )

        # §9.5 wiring: conditional TOST for the declared equivalence legs.
        equivalence_section = compute_equivalence(
            lambda keys: load_per_query(run_dir, index, keys),
            index,
            family_ctx,
            metric=equivalence_metric,
            margin=tost_margin,
            alpha=alpha,
        )

        if blinding_active:
            assert blinding_state is not None
            mapping = {
                str(k): str(v) for k, v in dict(blinding_state["mapping"]).items()
            }
            contrast_stats = [
                apply_blinding_to_entry(e, mapping, index) for e in contrast_stats
            ]

        analysis_dir = _make_analysis_dir(run_dir)
        figures = (
            render_figures(per_query, pairs, metrics, analysis_dir, stamp, index)
            if pairs and not blinding_active
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
            "preconditions": preconditions,
            "requested_contrast_ids": list(dict.fromkeys(contrast_ids)),
            "metrics": list(metrics),
            "family_map": (
                {
                    "group": family_ctx.group,
                    "datasets": list(family_ctx.datasets),
                    "n_rows": int(len(family_ctx.table)),
                    "source": "families.compile_family_map (§9.3)",
                }
                if family_ctx is not None
                else {
                    "absent_reason": (
                        "no charter family-map dataset in this run "
                        f"(known: {sorted(KNOWN_DATASETS)})"
                    )
                }
            ),
            "contrasts": contrast_stats,
            "gatekeeping": gatekeeping_section,
            "equivalence": equivalence_section,
            "blinding": blinding_section,
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
                "confirmatory_exclusions": confirmatory_exclusions,
            },
            "figures": figures,
        }

        # §9.8: the confirmatory look IS the freeze — record the one-time
        # unblinding event AFTER the pipeline computed, BEFORE the outputs
        # are written (a pipeline failure above leaves the seal unspent).
        if mode == "confirmatory" and blinding_state is not None:
            log_path = run_dir / BLINDING_DIRNAME / UNBLIND_LOG_NAME
            try:
                blinding_mod.unblind(sealed_map_path(run_dir), log_path)
            except AlreadyUnblindedError as exc:
                raise AnalysisError(
                    f"blinding refusal (§9.8): {exc} — a confirmatory look "
                    "after a prior unblinding would not have been blind"
                ) from exc
            blinding_section["unblind_event"] = {
                "utc": datetime.now(timezone.utc).isoformat(),
                "log": f"{BLINDING_DIRNAME}/{UNBLIND_LOG_NAME}",
                "map_sha256": blinding_state.get("map_sha256"),
                "note": "one-time logged unblinding at the confirmatory "
                "freeze (§9.8); outputs below carry REAL labels",
            }

        stats_path = analysis_dir / STATS_JSON_NAME
        stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        summary_path = analysis_dir / SUMMARY_MD_NAME
        summary_path.write_text(build_summary_md(stats), encoding="utf-8")

        if mode == "confirmatory":
            assert registered_sha is not None
            write_lock(run_dir, registered_sha, analysis_dir, stats_path)
    except BaseException:
        # A crashed/failed attempt must not consume the one-look budget: only
        # a placeholder THIS process created is removed, and only while it is
        # still IN_PROGRESS (never a lock some other completed run finalized).
        if lock_acquired_here:
            _release_placeholder_lock(run_dir)
        raise

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
    parser.add_argument(
        "--calibration-report",
        type=Path,
        default=None,
        metavar="JSON",
        help="§9.7 calibration-report artifact (CalibrationReport as JSON); "
        "REQUIRED for --confirmatory (the look refuses without a PASSING "
        "report); optional in design-input as a gate dry-run",
    )
    parser.add_argument(
        "--tost-margin",
        type=float,
        default=None,
        metavar="MARGIN",
        help="registered §9.5 domain margin (metric units) for the declared "
        "conditional-TOST equivalence legs; without it the legs are listed "
        "as labeled skips",
    )
    parser.add_argument(
        "--equivalence-metric",
        type=str,
        default=None,
        metavar="METRIC",
        help="per-query metric column the §9.5 equivalence legs test "
        "(paired with --tost-margin)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        metavar="ALPHA",
        help="significance level for the gatekeeping chain and family map "
        "(default: 0.05)",
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
        if args.calibration_report is None:
            problems.append(
                "--calibration-report JSON is missing (§9.7 calibration is a "
                "hard precondition of the one registered look)"
            )
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
            calibration_report=args.calibration_report,
            tost_margin=args.tost_margin,
            equivalence_metric=args.equivalence_metric,
            alpha=args.alpha,
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
