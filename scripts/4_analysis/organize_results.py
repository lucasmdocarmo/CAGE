#!/usr/bin/env python3
"""Validate, index, and coverage-report ONE pulled campaign run.

Layout contract (cloud/RESULTS_LAYOUT.md §1 — THE authority; the campaign tree,
NOT the pilot layout — pilots stay on scripts/4_analysis/_results_loader.py):

    results/<campaign>/<session>/<run_id>/
        manifest.json                     # run contract (§3): campaign/session/run_id/model (SINGULAR)
        ledger.json                       # §9.10 sha256 seal (verified by pull_run.sh)
        cells/<row_key>/
            cell.json                     # the CellSpec + baseline id + windows[] table
            window_<dataset>-<ordinal>/   # one measurement window; k = <dataset>-<ordinal>
                requests.jsonl            # per-request records (required)
                qa_evidence.jsonl         # raw outputs + evidence (required; sharegpt exempt)
                engine_metrics.json       # engine /metrics snapshots (required)
                cage_stats.jsonl          # cage-stats telemetry stream (required)
        scoring/<scoring_run_id>/         # offline scoring passes (§6): validated —
            scoring_manifest.json         #   manifest + own ledger + cells/ mirror of
            ledger.json                   #   the raw tree (qa_scores.jsonl, quality.json);
            cells/<row_key>/window_<k>/   #   NEVER indexed, NEVER inside raw cells/

The dataset lives IN the window directory name (e.g. ``window_squad_v2-01``) —
that is what makes the §8 dataset-scoped globs (``window_<Y>-*``) possible
without opening any JSON. There is NO window.json: window metadata lives in
cell.json's windows[] table; extra ``*.jsonl``/``*.json`` files in a window are
indexed as auxiliary artifacts.

``<row_key>`` is the VERBATIM ``CellSpec.to_row_key()`` string — the D7 tuple
``arm|retriever|policy|topology|engine|model|family[|r{g}|lam{g}]``. Every cell
directory is parsed back through ``CellSpec`` (round-trip enforced); unknown or
charter-illegal keys FAIL LOUD, listed together, none skipped.

Outputs (under <run>/index/):
- ``cells_index.csv``   — one row per window: identity axes + pressure coords +
  dataset + relative artifact paths. This is THE handoff the figure pipeline
  (scripts/4_analysis/figure_pipeline.py, keyed on ``row_key``) and the
  cage-stats engine consume for per-run/model/dataset analysis.
- ``coverage_report.md`` — per model x dataset x arm window counts; cells
  MISSING versus the PUBLICATION.md §7.6.1 family x group matrix (F1 arm-level
  floor; plus manifest ``expected_cells`` exactly when declared) are LISTED,
  never silently absent.
- ``provenance.json`` — the §3 provenance snapshot (git_sha, seed, ...)
  verbatim from the manifest, index-adjacent so no analysis has to re-open
  manifest.json for run identity (task #129 / H8).

All three are written atomically (tmp + os.replace); re-organizing over an
existing index requires ``--force``. The §5 seal is FULLY verified here
(re-hash + extra-file sweep over cells/), not merely checked for presence.

Fail-loud doctrine: layout violations (missing manifest/ledger/cells, missing
cell.json, malformed window dirs, unknown dataset ids, missing required window
artifacts, bad row keys) abort with every problem enumerated. Coverage GAPS do
not abort — a partial run is organizeable; the gap list is the deliverable.
Pure stdlib + pandas; no network.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for _p in (str(_HERE), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.analysis.cellspec import BASELINES, CellSpec, CellSpecError  # noqa: E402
from src.analysis.stats.ledger import LedgerError, read_ledger, verify_ledger  # noqa: E402

#: §1: window dir k = <dataset_id>-<ordinal>, e.g. window_squad_v2-01. The
#: dataset id is validated against DATASET_IDS separately so an unknown id is
#: reported as such (not as a malformed name).
WINDOW_DIR_RE = re.compile(r"^window_([a-z0-9_]+)-(\d+)$")
CELL_META_NAME = "cell.json"
INDEX_DIRNAME = "index"
INDEX_CSV_NAME = "cells_index.csv"
COVERAGE_MD_NAME = "coverage_report.md"
PROVENANCE_JSON_NAME = "provenance.json"
_ARTIFACT_SEP = ";"

#: §1 run_id grammar: lowercase, bucket-name-safe [a-z0-9-] ONLY — it names the
#: GCS bucket cage-<run_id> VERBATIM. terraform/variables.tf enforces the SAME
#: pattern (tests/test_terraform_contract.py pins the two together) so the
#: bucket terraform creates and the gs://cage-<run_id> the RUNBOOK exports can
#: never diverge (the pilot "synced to a bucket that didn't exist" bug class).
RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,40}$")

#: §1 session vocabulary (RUNBOOK §1).
SESSIONS: frozenset[str] = frozenset({"a", "b", "cd-act1", "cd-act2"})

#: §6 scoring-tree contract (offline quality passes; validated, never indexed).
SCORING_DIRNAME = "scoring"
SCORING_MANIFEST_NAME = "scoring_manifest.json"
SCORING_RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,40}$")
REQUIRED_SCORING_WINDOW_ARTIFACTS: tuple[str, ...] = ("qa_scores.jsonl", "quality.json")
#: Task #130 decision (a): the ONE legal non-pass file at the scoring/ root —
#: the sealed per-run-root label-stripping salt written by
#: scripts/4_analysis/rescore_quality.py (shared with score_instrument_b.py).
BLINDING_SALT_NAME = "blinding_salt.json"
#: Task #130 decision (d): abandoned pass directories
#: (rescore_quality.abandon_scoring_pass): scoring/<id>.abandoned-<UTCstamp>/
#: carrying an ABANDONED.json tombstone — tolerated, never validated/consumed.
ABANDONED_DIR_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,40}\.abandoned-\d{8}T\d{6}Z$")
ABANDONED_TOMBSTONE_NAME = "ABANDONED.json"
_SCORING_MANIFEST_REQUIRED_KEYS: tuple[str, ...] = (
    "scoring_run_id",
    "created_utc",
    "raw_run_ledger_entries_sha256",
)

#: Task #119 — the §8.5 predicate-table sibling (predicate/<scoring_run_id>/,
#: produced by scripts/4_analysis/build_predicate_table.py): a post-seal
#: derived tree, validated like a scoring pass, never indexed, never inside
#: cells/.
PREDICATE_DIRNAME = "predicate"
PREDICATE_MANIFEST_NAME = "predicate_manifest.json"
PREDICATE_ROWS_NAME = "predicate.jsonl"
_PREDICATE_MANIFEST_REQUIRED_KEYS: tuple[str, ...] = (
    "schema_version",
    "scoring_run_id",
    "raw_run_ledger_entries_sha256",
    "scoring_ledger_entries_sha256",
    "config",
    "counts",
    "created_utc",
)
#: The predicate-table row contract (schema guard; JSON null is legal for
#: predicate/record_index — None-propagation is counted, never fabricated).
_PREDICATE_ROW_REQUIRED_KEYS: tuple[str, ...] = (
    "example_id",
    "repeat_index",
    "record_index",
    "ok",
    "dataset",
    "predicate",
    "predicate_rule",
    "predicate_null_reason",
)

#: §1 dataset ids — the ONLY legal window-name datasets.
DATASET_IDS: frozenset[str] = frozenset(
    {"squad_v2", "hotpotqa", "musique", "qasper", "ruler", "scbench", "sharegpt"}
)

#: §1 per-window artifact contract. ShareGPT is the load donor — its windows
#: carry serving streams only, so qa_evidence.jsonl is exempt there.
REQUIRED_WINDOW_ARTIFACTS: tuple[str, ...] = (
    "requests.jsonl",
    "qa_evidence.jsonl",
    "engine_metrics.json",
    "cage_stats.jsonl",
)
QA_EVIDENCE_EXEMPT_DATASETS: frozenset[str] = frozenset({"sharegpt"})
_QA_EVIDENCE_EXEMPT_DATASETS = QA_EVIDENCE_EXEMPT_DATASETS  # historical alias

#: §7.6.1 group letter per D4 model (campaign roster; groups A-D).
GROUP_OF_MODEL: dict[str, str] = {
    "qwen3-14b": "A",
    "llama-3.3-70b": "B",
    "qwen3-next-80b": "C",
    "deepseek-v3": "D",
}

#: §7.6.1 F1 row — the numbered-baseline carriage per group (A/B: B1-B12;
#: C/D: B1-B10). This is the ARM-LEVEL coverage floor the report checks against;
#: F2/F3 pressure-grid density is run-scoped and checked only through the
#: manifest's optional ``expected_cells`` declaration.
F1_BASELINES_BY_GROUP: dict[str, tuple[str, ...]] = {
    "A": tuple(f"B{i}" for i in range(1, 13)),
    "B": tuple(f"B{i}" for i in range(1, 13)),
    "C": tuple(f"B{i}" for i in range(1, 11)),
    "D": tuple(f"B{i}" for i in range(1, 11)),
}

#: Reverse of the §7.1 numbered layer: (arm, retriever) -> baseline id.
#: Unique by construction (B5 vs B6 differ in retriever); cells outside the
#: numbered layer (e.g. retr-fresh · bm25 adequacy gate) get baseline = "".
BASELINE_OF_CELL: dict[tuple[str, str], str] = {
    (spec.arm, spec.retriever): bid for bid, spec in BASELINES.items()
}

#: §3 manifest identity keys with dedicated semantic checks below. `model` is
#: SINGULAR per §3 (one run = one model; a re-run is a new run_id). There are
#: NO `models`/`datasets` list keys in the spec: datasets come from the window
#: directory names (§1); an OPTIONAL `datasets` list may narrow coverage.
_MANIFEST_STR_KEYS = ("campaign", "session", "run_id", "model")

#: §3 REQUIRED manifest fields (cloud/RESULTS_LAYOUT.md §3 — task #129 / H8:
#: a run without its provenance cannot be organized; fail loud, every gap
#: listed). ``engine``/``engine_version`` structure is per-engine in the spec;
#: this organizer requires engine_version and accepts str or mapping there and
#: for hardware. The full snapshot is surfaced in index/provenance.json.
MANIFEST_REQUIRED_FIELDS: tuple[str, ...] = (
    "campaign",
    "session",
    "run_id",
    "model",
    "git_sha",
    "git_dirty",
    "engine_version",
    "seed",
    "provider",
    "hardware",
    "dataset_manifests_sha256",
    "cellspec_schema_version",
    "created_utc",
)

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


class OrganizeError(RuntimeError):
    """Base error for run-organization failures."""


class LayoutError(OrganizeError):
    """The pulled tree violates the layout contract; carries EVERY problem found."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        lines = "\n".join(f"  [{i + 1}] {p}" for i, p in enumerate(self.problems))
        super().__init__(
            f"run tree violates cloud/RESULTS_LAYOUT.md — {len(self.problems)} problem(s):\n{lines}"
        )


# ---------------------------------------------------------------------------
# Row-key and manifest parsing
# ---------------------------------------------------------------------------


def parse_row_key_dir(name: str) -> CellSpec:
    """Parse one ``cells/<row_key>`` directory name back into its CellSpec.

    Mirrors figure_pipeline.parse_row_key (kept import-free of matplotlib):
    7 axis segments + optional ``r<float>`` / ``lam<float>`` coords, validated
    by CellSpec construction AND an exact round-trip back to the dirname.
    """
    parts = name.split("|")
    if len(parts) < 7:
        raise OrganizeError(
            f"row key {name!r} has {len(parts)} segment(s); expected the 7 axes "
            "(arm|retriever|policy|topology|engine|model|family) + optional coords"
        )
    budget_r: float | None = None
    rate_frac: float | None = None
    for coord in parts[7:]:
        try:
            if coord.startswith("lam"):
                rate_frac = float(coord[3:])
            elif coord.startswith("r"):
                budget_r = float(coord[1:])
            else:
                raise OrganizeError(
                    f"row key {name!r}: unrecognized coord segment {coord!r} "
                    "(expected 'r<float>' or 'lam<float>')"
                )
        except ValueError as exc:
            raise OrganizeError(
                f"row key {name!r}: malformed coord segment {coord!r}: {exc}"
            ) from exc
    try:
        spec = CellSpec(*parts[:7], budget_r=budget_r, rate_frac=rate_frac)  # type: ignore[arg-type]
    except (CellSpecError, ValueError) as exc:
        raise OrganizeError(f"row key {name!r} is not a valid CellSpec: {exc}") from exc
    if spec.to_row_key() != name:
        raise OrganizeError(
            f"row key {name!r} does not round-trip (canonical: {spec.to_row_key()!r})"
        )
    return spec


def _validate_manifest_provenance(manifest: Mapping[str, Any]) -> list[str]:
    """§3 provenance-field enforcement (task #129 / H8) — returns problem lines.

    Every MANIFEST_REQUIRED_FIELDS entry must be present with the right shape;
    absence is NOT defaulted (a run whose git_sha/seed are unknown cannot be
    reproduced, so it cannot be indexed). The four identity keys are typed by
    the caller's _MANIFEST_STR_KEYS pass; this covers the rest.
    """
    problems: list[str] = []
    missing = [k for k in MANIFEST_REQUIRED_FIELDS if k not in manifest]
    if missing:
        problems.append(
            f"manifest.json missing REQUIRED §3 field(s) {missing} "
            "(cloud/RESULTS_LAYOUT.md §3 — provenance is not optional)"
        )
    checks: tuple[tuple[str, str], ...] = (
        ("git_sha", "non-empty string"),
        ("provider", "non-empty string"),
        ("created_utc", "non-empty string (ISO-8601)"),
    )
    for key, expected in checks:
        if key in manifest and (
            not isinstance(manifest[key], str) or not manifest[key]
        ):
            problems.append(f"manifest.json key {key!r} must be a {expected}")
    if "git_dirty" in manifest and not isinstance(manifest["git_dirty"], bool):
        problems.append("manifest.json key 'git_dirty' must be a boolean")
    for key in ("engine_version", "hardware"):
        if key in manifest:
            value = manifest[key]
            ok = (isinstance(value, str) and value) or (
                isinstance(value, dict) and value
            )
            if not ok:
                problems.append(
                    f"manifest.json key {key!r} must be a non-empty string or object"
                )
    for key in ("seed", "cellspec_schema_version"):
        if key in manifest and (
            isinstance(manifest[key], bool) or not isinstance(manifest[key], int)
        ):
            problems.append(f"manifest.json key {key!r} must be an integer")
    if "dataset_manifests_sha256" in manifest and (
        not isinstance(manifest["dataset_manifests_sha256"], str)
        or not _SHA256_HEX_RE.match(manifest["dataset_manifests_sha256"])
    ):
        problems.append(
            "manifest.json key 'dataset_manifests_sha256' must be a 64-char "
            "lowercase sha256 hex string (§3: pins the exact dataset builds)"
        )
    return problems


def load_manifest(run_dir: Path) -> dict[str, Any]:
    """Load + validate manifest.json — the run's declared contract (fail loud)."""
    path = run_dir / "manifest.json"
    if not path.is_file():
        raise LayoutError([f"manifest.json missing at {path}"])
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LayoutError([f"manifest.json is not valid JSON: {exc}"]) from exc
    problems: list[str] = []
    if not isinstance(manifest, dict):
        raise LayoutError([f"manifest.json root must be an object, got {type(manifest).__name__}"])
    for key in _MANIFEST_STR_KEYS:
        if not isinstance(manifest.get(key), str) or not manifest.get(key):
            problems.append(f"manifest.json key {key!r} must be a non-empty string")
    problems.extend(_validate_manifest_provenance(manifest))
    declared_datasets = manifest.get("datasets")
    if declared_datasets is not None:
        if (
            not isinstance(declared_datasets, list)
            or not declared_datasets
            or not all(isinstance(v, str) and v for v in declared_datasets)
        ):
            problems.append(
                "manifest.json optional key 'datasets' must be a non-empty list of strings"
            )
        else:
            unknown = [d for d in declared_datasets if d not in DATASET_IDS]
            if unknown:
                problems.append(
                    f"manifest datasets {unknown} are not RESULTS_LAYOUT §1 dataset ids "
                    f"({sorted(DATASET_IDS)})"
                )
    if not problems:
        if not RUN_ID_RE.match(manifest["run_id"]):
            problems.append(
                f"manifest run_id {manifest['run_id']!r} violates the §1 grammar "
                f"{RUN_ID_RE.pattern} (lowercase bucket-name-safe — it names "
                "gs://cage-<run_id> verbatim)"
            )
        if manifest["run_id"] != run_dir.name:
            problems.append(
                f"manifest run_id {manifest['run_id']!r} != run directory name {run_dir.name!r} "
                "(the tree was moved or the manifest lies — refuse to index either way)"
            )
        if manifest["session"] not in SESSIONS:
            problems.append(
                f"manifest session {manifest['session']!r} is not a §1 session "
                f"({sorted(SESSIONS)})"
            )
        if manifest["model"] not in GROUP_OF_MODEL:
            problems.append(
                f"manifest model {manifest['model']!r} is not on the D4 roster "
                f"({sorted(GROUP_OF_MODEL)})"
            )
    if problems:
        raise LayoutError(problems)
    return manifest


# ---------------------------------------------------------------------------
# Tree walk -> window records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowRecord:
    """One indexed measurement window of one cell."""

    spec: CellSpec
    row_key: str
    window: int  # the <ordinal> part of k
    window_key: str  # the FULL §8 join key k = <dataset>-<ordinal>
    dataset: str  # parsed from the window dir NAME (§1 — no JSON opened)
    window_dir: str  # run-dir-relative posix path
    cell_json: str  # run-dir-relative posix path to the cell's cell.json
    artifacts: tuple[str, ...]  # run-dir-relative posix paths (jsonl + json)


def _read_cell_meta(cell_dir: Path, problems: list[str]) -> dict[str, Any] | None:
    """Load + sanity-check ``cells/<row_key>/cell.json`` (§1 — one per cell).

    Contents beyond structural validity are opaque here, EXCEPT: when a
    ``cellspec`` mapping is present it must round-trip to the directory name
    (the tuple IS the name — a lying cell.json is a layout violation).
    """
    meta_path = cell_dir / CELL_META_NAME
    if not meta_path.is_file():
        problems.append(f"cells/{cell_dir.name}: missing {CELL_META_NAME} (§1: one per cell)")
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        problems.append(f"{meta_path}: invalid JSON: {exc}")
        return None
    if not isinstance(meta, dict):
        problems.append(f"{meta_path}: root must be an object, got {type(meta).__name__}")
        return None
    declared_spec = meta.get("cellspec")
    if declared_spec is not None:
        try:
            declared_key = CellSpec.from_flat_dict(declared_spec).to_row_key()
        except (CellSpecError, TypeError, ValueError) as exc:
            problems.append(f"{meta_path}: 'cellspec' is not a valid CellSpec: {exc}")
            return None
        if declared_key != cell_dir.name:
            problems.append(
                f"{meta_path}: cellspec {declared_key!r} contradicts directory name "
                f"{cell_dir.name!r}"
            )
            return None
    return meta


def walk_run_tree(run_dir: Path, manifest: Mapping[str, Any]) -> list[WindowRecord]:
    """Walk cells/ into WindowRecords; collect and raise ALL violations together."""
    problems: list[str] = []
    if not (run_dir / "ledger.json").is_file():
        problems.append(
            "ledger.json missing — the run is unsealed; pull_run.sh must have failed "
            "(verification is its job; presence is a layout requirement here)"
        )
    cells_dir = run_dir / "cells"
    if not cells_dir.is_dir():
        problems.append(f"cells/ directory missing at {cells_dir}")
        raise LayoutError(problems)

    declared_model = manifest["model"]
    declared_datasets = (
        set(manifest["datasets"]) if manifest.get("datasets") is not None else DATASET_IDS
    )
    records: list[WindowRecord] = []

    for cell_dir in sorted(p for p in cells_dir.iterdir() if not p.name.startswith(".")):
        if not cell_dir.is_dir():
            problems.append(f"cells/{cell_dir.name}: not a directory (stray file in cells/)")
            continue
        try:
            spec = parse_row_key_dir(cell_dir.name)
        except OrganizeError as exc:
            problems.append(f"cells/{cell_dir.name}: {exc}")
            continue
        if spec.model != declared_model:
            problems.append(
                f"cells/{cell_dir.name}: model {spec.model!r} != manifest model "
                f"{declared_model!r} (§3: one run = one model)"
            )
            continue
        cell_meta = _read_cell_meta(cell_dir, problems)
        if cell_meta is None:
            continue
        cell_json_rel = (cell_dir / CELL_META_NAME).relative_to(run_dir).as_posix()

        window_dirs = sorted(
            p
            for p in cell_dir.iterdir()
            if not p.name.startswith(".") and p.name != CELL_META_NAME
        )
        if not window_dirs:
            problems.append(
                f"cells/{cell_dir.name}: cell has no window_<dataset>-<ordinal> directories"
            )
            continue
        #: Task #129 / H12 collision guard: two dirs must never resolve to the
        #: same (dataset, ordinal) identity (window_x-1 vs window_x-01 would
        #: silently double-count one window under one §8 join key).
        seen_identity: dict[tuple[str, int], str] = {}
        for window_dir in window_dirs:
            match = WINDOW_DIR_RE.match(window_dir.name)
            if not window_dir.is_dir() or match is None:
                problems.append(
                    f"cells/{cell_dir.name}/{window_dir.name}: expected a "
                    "window_<dataset>-<ordinal> directory (§1, e.g. window_squad_v2-01)"
                )
                continue
            dataset, ordinal = match.group(1), int(match.group(2))
            identity = (dataset, ordinal)
            if identity in seen_identity:
                problems.append(
                    f"cells/{cell_dir.name}/{window_dir.name}: window identity "
                    f"({dataset}, {ordinal}) collides with sibling "
                    f"{seen_identity[identity]!r} — two directories resolve to "
                    "the same (dataset, ordinal); the §8 join key would be ambiguous"
                )
                continue
            seen_identity[identity] = window_dir.name
            if dataset not in DATASET_IDS:
                problems.append(
                    f"cells/{cell_dir.name}/{window_dir.name}: dataset {dataset!r} is not "
                    f"a §1 dataset id ({sorted(DATASET_IDS)})"
                )
                continue
            if dataset not in declared_datasets:
                problems.append(
                    f"cells/{cell_dir.name}/{window_dir.name}: dataset {dataset!r} is not "
                    f"declared in manifest datasets {sorted(declared_datasets)}"
                )
                continue
            missing = [
                name
                for name in REQUIRED_WINDOW_ARTIFACTS
                if not (window_dir / name).is_file()
                and not (
                    name == "qa_evidence.jsonl" and dataset in _QA_EVIDENCE_EXEMPT_DATASETS
                )
            ]
            if missing:
                problems.append(
                    f"cells/{cell_dir.name}/{window_dir.name}: missing required "
                    f"artifact(s) {missing} (§1 per-window contract)"
                )
                continue
            jsonl_files = sorted(window_dir.glob("*.jsonl"))
            aux_json = sorted(window_dir.glob("*.json"))
            artifacts = tuple(
                p.relative_to(run_dir).as_posix() for p in (*jsonl_files, *aux_json)
            )
            records.append(
                WindowRecord(
                    spec=spec,
                    row_key=cell_dir.name,
                    window=ordinal,
                    window_key=f"{dataset}-{match.group(2)}",
                    dataset=dataset,
                    window_dir=window_dir.relative_to(run_dir).as_posix(),
                    cell_json=cell_json_rel,
                    artifacts=artifacts,
                )
            )

    if problems:
        raise LayoutError(problems)
    if not records:
        raise LayoutError(["cells/ contains no indexable windows — an empty run indexes nothing"])
    return records


# ---------------------------------------------------------------------------
# Index + coverage
# ---------------------------------------------------------------------------

INDEX_COLUMNS: tuple[str, ...] = (
    "run_id",
    "campaign",
    "session",
    "model",
    "engine",
    "arm",
    "baseline",
    "retriever",
    "policy",
    "topology",
    "family",
    "dataset",
    "budget_r",
    "rate_frac",
    "window",
    "window_key",
    "row_key",
    "window_dir",
    "cell_json",
    "artifacts",
)


def build_index(manifest: Mapping[str, Any], records: list[WindowRecord]) -> pd.DataFrame:
    """One row per window, deterministically sorted — the analysis handoff table."""
    rows = []
    for rec in records:
        spec = rec.spec
        rows.append(
            {
                "run_id": manifest["run_id"],
                "campaign": manifest["campaign"],
                "session": manifest["session"],
                "model": spec.model,
                "engine": spec.engine,
                "arm": spec.arm,
                "baseline": BASELINE_OF_CELL.get((spec.arm, spec.retriever), ""),
                "retriever": spec.retriever,
                "policy": spec.policy,
                "topology": spec.topology,
                "family": spec.family,
                "dataset": rec.dataset,
                "budget_r": spec.budget_r,
                "rate_frac": spec.rate_frac,
                "window": rec.window,
                "window_key": rec.window_key,
                "row_key": rec.row_key,
                "window_dir": rec.window_dir,
                "cell_json": rec.cell_json,
                "artifacts": _ARTIFACT_SEP.join(rec.artifacts),
            }
        )
    df = pd.DataFrame(rows, columns=list(INDEX_COLUMNS))
    return df.sort_values(
        ["model", "dataset", "row_key", "window"], kind="mergesort"
    ).reset_index(drop=True)


def expected_arms_for_model(model: str) -> dict[str, tuple[str, ...]]:
    """§7.6.1 F1 arm-level floor: arm -> the baseline ids that land on it."""
    group = GROUP_OF_MODEL[model]
    by_arm: dict[str, list[str]] = {}
    for bid in F1_BASELINES_BY_GROUP[group]:
        by_arm.setdefault(BASELINES[bid].arm, []).append(bid)
    return {arm: tuple(bids) for arm, bids in by_arm.items()}


def build_coverage_report(manifest: Mapping[str, Any], index: pd.DataFrame) -> str:
    """Markdown coverage: model x dataset x arm counts + explicit MISSING lists."""
    lines: list[str] = [
        "# Run coverage report",
        "",
        f"- run_id: `{manifest['run_id']}`",
        f"- campaign / session: `{manifest['campaign']}` / `{manifest['session']}`",
        # §3 provenance surfaced where the analyst actually looks (task #129/H8).
        f"- git_sha: `{manifest['git_sha']}` (dirty: {json.dumps(manifest['git_dirty'])})",
        f"- seed: {manifest['seed']}",
        f"- indexed windows: {len(index)}",
        f"- cells (distinct row keys): {index['row_key'].nunique()}",
        "",
        "Expectation source: PUBLICATION.md §7.6.1 family x group matrix, F1 row "
        "(arm-level floor: groups A/B carry B1-B12, C/D carry B1-B10); plus the "
        "manifest's `expected_cells` declaration when present. MISSING cells are "
        "listed explicitly — never silently absent.",
        "",
    ]
    total_missing = 0
    declared_datasets = manifest.get("datasets")
    datasets: list[str] = (
        list(declared_datasets)
        if declared_datasets is not None
        else sorted(index["dataset"].unique())
    )
    for model in [manifest["model"]]:
        group = GROUP_OF_MODEL[model]
        expected = expected_arms_for_model(model)
        for dataset in datasets:
            sub = index[(index["model"] == model) & (index["dataset"] == dataset)]
            lines.append(f"## {model} (group {group}) x {dataset}")
            lines.append("")
            if sub.empty:
                lines.append("**No windows at all for this model x dataset.**")
                lines.append("")
            else:
                lines.append("| arm | baselines | windows |")
                lines.append("|---|---|---|")
                counts = sub.groupby("arm", sort=True).size()
                for arm, n in counts.items():
                    bids = ", ".join(expected.get(str(arm), ())) or "-"
                    lines.append(f"| {arm} | {bids} | {n} |")
                lines.append("")
            observed_arms = set(sub["arm"].unique())
            missing = sorted(set(expected) - observed_arms)
            extra = sorted(observed_arms - set(expected))
            if missing:
                total_missing += len(missing)
                lines.append(f"MISSING arms vs §7.6.1 ({len(missing)}):")
                for arm in missing:
                    lines.append(f"- MISSING {model} x {dataset} x {arm} ({', '.join(expected[arm])})")
            else:
                lines.append("MISSING arms vs §7.6.1: none")
            if extra:
                lines.append(
                    f"EXTRA arms beyond the group-{group} F1 floor (informational): "
                    + ", ".join(extra)
                )
            lines.append("")

    declared = manifest.get("expected_cells")
    if declared is not None:
        lines.append("## Manifest-declared expected cells")
        lines.append("")
        if not isinstance(declared, list) or not all(
            isinstance(e, dict) and {"model", "dataset", "row_key"} <= set(e) for e in declared
        ):
            raise LayoutError(
                ["manifest expected_cells must be a list of {model, dataset, row_key} objects"]
            )
        observed = {
            (r.model, r.dataset, r.row_key)
            for r in index.itertuples(index=False)
        }
        declared_missing = [
            e
            for e in declared
            if (e["model"], e["dataset"], e["row_key"]) not in observed
        ]
        total_missing += len(declared_missing)
        if declared_missing:
            lines.append(f"MISSING declared cells ({len(declared_missing)} of {len(declared)}):")
            for e in declared_missing:
                lines.append(f"- MISSING {e['model']} x {e['dataset']} x `{e['row_key']}`")
        else:
            lines.append(f"All {len(declared)} declared cells present.")
        lines.append("")

    lines.append("---")
    lines.append(f"TOTAL MISSING entries: {total_missing}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# §6 scoring-tree validation (RESULTS_LAYOUT: scoring/<scoring_run_id>/)
# ---------------------------------------------------------------------------


def _sweep_contamination(run_dir: Path) -> list[str]:
    """Find §6-forbidden scoring artifacts anywhere under ``cells/``.

    Task #129 / H7: the old ``rglob`` sweep did not traverse directory
    symlinks, so a symlinked subtree carrying qa_scores.jsonl/quality.json
    into the sealed raw tree passed undetected. ``os.walk(followlinks=True)``
    traverses them; a realpath visited-set terminates symlink cycles.
    Returns run-dir-relative posix paths, sorted.
    """
    cells_root = run_dir / "cells"
    if not cells_root.is_dir():
        return []
    hits: set[str] = set()
    visited: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(cells_root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in visited:
            dirnames[:] = []  # symlink cycle: prune, do not recurse forever
            continue
        visited.add(real)
        for name in filenames:
            # Predicate tables (#119) are §6-forbidden inside cells/ exactly
            # like scoring outputs — the raw tree stays sealed.
            if name in REQUIRED_SCORING_WINDOW_ARTIFACTS or name == PREDICATE_ROWS_NAME:
                hits.add(
                    Path(os.path.relpath(Path(dirpath) / name, run_dir)).as_posix()
                )
    return sorted(hits)


def validate_scoring_tree(run_dir: Path) -> list[str]:
    """Validate every ``scoring/<scoring_run_id>/`` pass against §6 (fail loud).

    Rules enforced:
    - the raw ``cells/`` tree carries NO scoring output (scoring NEVER writes
      into cells/ — the raw tree is sealed);
    - each scoring pass id obeys the §6 grammar and carries a
      ``scoring_manifest.json`` whose ``scoring_run_id`` matches its directory
      and whose ``raw_run_ledger_entries_sha256`` matches THIS run's sealed
      ledger (a pass that scored a different seal proves nothing here);
    - its ``cells/<row_key>/window_<k>/`` mirror only cells/windows that exist
      in the raw tree, each window carrying qa_scores.jsonl + quality.json;
    - the pass carries its OWN sealed ledger and verifies against it (§6:
      sealed before stats may consume it).

    Returns human-readable summary lines for the coverage report ("none" when
    the run has no scoring passes — scoring is optional at organize time).
    """
    problems: list[str] = []
    summary: list[str] = []

    for path in _sweep_contamination(run_dir):
        problems.append(
            f"{path}: scoring output inside the sealed raw tree — §6: scoring "
            "NEVER writes into cells/"
        )

    scoring_root = run_dir / SCORING_DIRNAME
    if not scoring_root.is_dir():
        if problems:
            raise LayoutError(problems)
        return summary

    raw_entries_sha256: str | None = None
    raw_ledger_path = run_dir / "ledger.json"
    if raw_ledger_path.is_file():
        try:
            read_ledger(raw_ledger_path)  # verifies the self-hash
            raw_entries_sha256 = json.loads(
                raw_ledger_path.read_text(encoding="utf-8")
            )["entries_sha256"]
        except (LedgerError, json.JSONDecodeError, KeyError) as exc:
            problems.append(f"ledger.json unusable for scoring validation: {exc}")

    for scoring_dir in sorted(p for p in scoring_root.iterdir() if not p.name.startswith(".")):
        sid = scoring_dir.name
        prefix = f"{SCORING_DIRNAME}/{sid}"
        if sid == BLINDING_SALT_NAME and scoring_dir.is_file():
            # Task #130 (a): the sealed per-run-root blinding salt is legal
            # scoring/ metadata (charter §9.8 label stripping), not a stray.
            summary.append(
                f"- blinding salt present (`{SCORING_DIRNAME}/{BLINDING_SALT_NAME}`,"
                " §9.8 label stripping — task #130)"
            )
            continue
        if not scoring_dir.is_dir():
            problems.append(f"{prefix}: not a directory (stray file in scoring/)")
            continue
        if ABANDONED_DIR_RE.match(sid):
            # Task #130 (d): an abandoned pass is tombstoned dead weight — it
            # is neither validated nor consumed, but it must not fail the
            # tree (its id was freed for a clean retry). A directory NAMED
            # abandoned without its tombstone is still a problem.
            if (scoring_dir / ABANDONED_TOMBSTONE_NAME).is_file():
                summary.append(
                    f"- `{sid}`: ABANDONED pass (tombstoned — not validated,"
                    " not consumed; task #130)"
                )
            else:
                problems.append(
                    f"{prefix}: abandoned-named directory without its "
                    f"{ABANDONED_TOMBSTONE_NAME} tombstone"
                )
            continue
        if not SCORING_RUN_ID_RE.match(sid):
            problems.append(
                f"{prefix}: scoring run id violates the §6 grammar "
                f"{SCORING_RUN_ID_RE.pattern}"
            )
            continue

        manifest_path = scoring_dir / SCORING_MANIFEST_NAME
        if not manifest_path.is_file():
            problems.append(f"{prefix}: missing {SCORING_MANIFEST_NAME} (§6)")
            continue
        try:
            scoring_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            problems.append(f"{prefix}/{SCORING_MANIFEST_NAME}: invalid JSON: {exc}")
            continue
        if not isinstance(scoring_manifest, dict):
            problems.append(f"{prefix}/{SCORING_MANIFEST_NAME}: root must be an object")
            continue
        missing_keys = [
            k for k in _SCORING_MANIFEST_REQUIRED_KEYS if not scoring_manifest.get(k)
        ]
        if missing_keys:
            problems.append(
                f"{prefix}/{SCORING_MANIFEST_NAME}: missing required key(s) "
                f"{missing_keys}"
            )
            continue
        if scoring_manifest["scoring_run_id"] != sid:
            problems.append(
                f"{prefix}/{SCORING_MANIFEST_NAME}: scoring_run_id "
                f"{scoring_manifest['scoring_run_id']!r} != directory name {sid!r}"
            )
        if (
            raw_entries_sha256 is not None
            and scoring_manifest["raw_run_ledger_entries_sha256"] != raw_entries_sha256
        ):
            problems.append(
                f"{prefix}: raw_run_ledger_entries_sha256 does not match this "
                "run's sealed ledger — the pass scored a DIFFERENT seal"
            )

        n_windows = 0
        scoring_cells = scoring_dir / "cells"
        if scoring_cells.is_dir():
            for cell_dir in sorted(
                p for p in scoring_cells.iterdir() if not p.name.startswith(".")
            ):
                raw_cell = run_dir / "cells" / cell_dir.name
                if not raw_cell.is_dir():
                    problems.append(
                        f"{prefix}/cells/{cell_dir.name}: no such cell in the "
                        "raw tree — a scoring pass cannot invent cells (§6 mirror)"
                    )
                    continue
                for window_dir in sorted(
                    p for p in cell_dir.iterdir() if not p.name.startswith(".")
                ):
                    if not window_dir.is_dir() or not WINDOW_DIR_RE.match(window_dir.name):
                        problems.append(
                            f"{prefix}/cells/{cell_dir.name}/{window_dir.name}: "
                            "expected a window_<dataset>-<ordinal> directory"
                        )
                        continue
                    if not (raw_cell / window_dir.name).is_dir():
                        problems.append(
                            f"{prefix}/cells/{cell_dir.name}/{window_dir.name}: "
                            "no such window in the raw tree (§6 mirror)"
                        )
                        continue
                    missing = [
                        name
                        for name in REQUIRED_SCORING_WINDOW_ARTIFACTS
                        if not (window_dir / name).is_file()
                    ]
                    if missing:
                        problems.append(
                            f"{prefix}/cells/{cell_dir.name}/{window_dir.name}: "
                            f"missing scoring artifact(s) {missing} (§6)"
                        )
                        continue
                    n_windows += 1

        scoring_ledger = scoring_dir / "ledger.json"
        if not scoring_ledger.is_file():
            problems.append(
                f"{prefix}: missing its own ledger.json — §6: scoring passes "
                "get their own ledger before being used by stats"
            )
        else:
            try:
                mismatches = verify_ledger(scoring_ledger, scoring_dir)
            except LedgerError as exc:
                problems.append(f"{prefix}/ledger.json: {exc}")
            else:
                for line in mismatches:
                    problems.append(f"{prefix}: ledger mismatch — {line}")

        summary.append(
            f"- `{sid}`: {n_windows} scored window(s), mode="
            f"{scoring_manifest.get('mode', '?')}, sealed="
            f"{'yes' if scoring_ledger.is_file() else 'NO'}"
        )

    if problems:
        raise LayoutError(problems)
    return summary


def validate_predicate_trees(run_dir: Path) -> list[str]:
    """Validate every ``predicate/<scoring_run_id>/`` table (task #119).

    Rules (mirroring the §6 scoring-pass discipline — the table is a derived
    post-seal sibling, never indexed):
    - each table carries ``predicate_manifest.json`` with the required keys,
      whose ``raw_run_ledger_entries_sha256`` matches THIS run's seal and
      whose ``scoring_run_id`` names an existing scoring pass;
    - its ``cells/<row_key>/window_<k>/`` mirror only windows that exist in
      the raw tree, each carrying ``predicate.jsonl`` rows that parse and
      hold the required keys with ``predicate`` in {true, false, null};
    - the table verifies against its OWN sealed ledger.

    Returns coverage-report summary lines ("none" when no table exists —
    the predicate build is optional at organize time).
    """
    problems: list[str] = []
    summary: list[str] = []
    predicate_root = run_dir / PREDICATE_DIRNAME
    if not predicate_root.is_dir():
        return summary

    raw_entries_sha256: str | None = None
    raw_ledger_path = run_dir / "ledger.json"
    if raw_ledger_path.is_file():
        try:
            read_ledger(raw_ledger_path)
            raw_entries_sha256 = json.loads(
                raw_ledger_path.read_text(encoding="utf-8")
            )["entries_sha256"]
        except (LedgerError, json.JSONDecodeError, KeyError) as exc:
            problems.append(f"ledger.json unusable for predicate validation: {exc}")

    for pred_dir in sorted(
        p for p in predicate_root.iterdir() if not p.name.startswith(".")
    ):
        pid = pred_dir.name
        prefix = f"{PREDICATE_DIRNAME}/{pid}"
        if not pred_dir.is_dir():
            problems.append(f"{prefix}: not a directory (stray file in predicate/)")
            continue
        if not SCORING_RUN_ID_RE.match(pid):
            problems.append(
                f"{prefix}: predicate table id violates the §6 id grammar "
                f"{SCORING_RUN_ID_RE.pattern} (the table is named after the "
                "scoring pass it joins)"
            )
            continue
        manifest_path = pred_dir / PREDICATE_MANIFEST_NAME
        if not manifest_path.is_file():
            problems.append(f"{prefix}: missing {PREDICATE_MANIFEST_NAME} (#119)")
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            problems.append(f"{prefix}/{PREDICATE_MANIFEST_NAME}: invalid JSON: {exc}")
            continue
        if not isinstance(manifest, dict):
            problems.append(f"{prefix}/{PREDICATE_MANIFEST_NAME}: root must be an object")
            continue
        missing_keys = [
            k for k in _PREDICATE_MANIFEST_REQUIRED_KEYS if manifest.get(k) is None
        ]
        if missing_keys:
            problems.append(
                f"{prefix}/{PREDICATE_MANIFEST_NAME}: missing required "
                f"key(s) {missing_keys}"
            )
            continue
        if manifest["scoring_run_id"] != pid:
            problems.append(
                f"{prefix}/{PREDICATE_MANIFEST_NAME}: scoring_run_id "
                f"{manifest['scoring_run_id']!r} != directory name {pid!r}"
            )
        if not (run_dir / SCORING_DIRNAME / pid).is_dir():
            problems.append(
                f"{prefix}: names scoring pass {pid!r} but "
                f"{SCORING_DIRNAME}/{pid}/ does not exist — a predicate "
                "table joins a real sealed scoring pass"
            )
        if (
            raw_entries_sha256 is not None
            and manifest["raw_run_ledger_entries_sha256"] != raw_entries_sha256
        ):
            problems.append(
                f"{prefix}: raw_run_ledger_entries_sha256 does not match this "
                "run's sealed ledger — the table was built against a "
                "DIFFERENT seal"
            )

        n_windows = 0
        n_rows = 0
        pred_cells = pred_dir / "cells"
        if pred_cells.is_dir():
            for cell_dir in sorted(
                p for p in pred_cells.iterdir() if not p.name.startswith(".")
            ):
                raw_cell = run_dir / "cells" / cell_dir.name
                if not raw_cell.is_dir():
                    problems.append(
                        f"{prefix}/cells/{cell_dir.name}: no such cell in the "
                        "raw tree — a predicate table cannot invent cells"
                    )
                    continue
                for window_dir in sorted(
                    p for p in cell_dir.iterdir() if not p.name.startswith(".")
                ):
                    if not window_dir.is_dir() or not WINDOW_DIR_RE.match(window_dir.name):
                        problems.append(
                            f"{prefix}/cells/{cell_dir.name}/{window_dir.name}: "
                            "expected a window_<dataset>-<ordinal> directory"
                        )
                        continue
                    if not (raw_cell / window_dir.name).is_dir():
                        problems.append(
                            f"{prefix}/cells/{cell_dir.name}/{window_dir.name}: "
                            "no such window in the raw tree (mirror-only rule)"
                        )
                        continue
                    rows_path = window_dir / PREDICATE_ROWS_NAME
                    if not rows_path.is_file():
                        problems.append(
                            f"{prefix}/cells/{cell_dir.name}/{window_dir.name}: "
                            f"missing {PREDICATE_ROWS_NAME}"
                        )
                        continue
                    n_windows += 1
                    for lineno, line in enumerate(
                        rows_path.read_text(encoding="utf-8").splitlines(), 1
                    ):
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError as exc:
                            problems.append(f"{rows_path}:{lineno}: invalid JSON: {exc}")
                            continue
                        if not isinstance(row, dict):
                            problems.append(
                                f"{rows_path}:{lineno}: row must be an object"
                            )
                            continue
                        n_rows += 1
                        missing_row_keys = [
                            k for k in _PREDICATE_ROW_REQUIRED_KEYS if k not in row
                        ]
                        if missing_row_keys:
                            problems.append(
                                f"{rows_path}:{lineno}: missing key(s) "
                                f"{missing_row_keys} (predicate row contract)"
                            )
                        if row.get("predicate") not in (True, False, None):
                            problems.append(
                                f"{rows_path}:{lineno}: predicate value "
                                f"{row.get('predicate')!r} outside "
                                "{true, false, null} — None-propagation is "
                                "tri-state, never a number"
                            )

        pred_ledger = pred_dir / "ledger.json"
        if not pred_ledger.is_file():
            problems.append(
                f"{prefix}: missing its own ledger.json — a predicate table "
                "is sealed at build time (#119)"
            )
        else:
            try:
                mismatches = verify_ledger(pred_ledger, pred_dir)
            except LedgerError as exc:
                problems.append(f"{prefix}/ledger.json: {exc}")
            else:
                for line in mismatches:
                    problems.append(f"{prefix}: ledger mismatch — {line}")

        summary.append(
            f"- `{pid}`: {n_windows} predicate window(s), {n_rows} row(s), "
            f"sealed={'yes' if pred_ledger.is_file() else 'NO'}"
        )

    if problems:
        raise LayoutError(problems)
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def verify_run_ledger(run_dir: Path) -> str:
    """Fully verify the run's §5 seal at organize time (task #129 item 4).

    Presence alone proves nothing (H8: the ledger was previously verified at
    analysis load only, and only in confirmatory mode — §5 says always). Here
    the whole tree is re-hashed, INCLUDING the H7 extra-file sweep scoped to
    ``cells/`` (the append-only-then-immutable subtree; §6 siblings such as
    ``scoring/``/``index/``/``analysis/`` are legal post-seal additions and
    are never falsely flagged). Any mismatch fails loud; an absent ledger is
    recorded honestly as "ledger: absent" (walk_run_tree already treats
    absence as a layout violation upstream, so this line is reachable only if
    that presence rule is ever relaxed).
    """
    ledger_path = run_dir / "ledger.json"
    if not ledger_path.is_file():
        return "ledger: absent (§5 — the run is unsealed)"
    cells_dir = run_dir / "cells"
    extra_roots = (cells_dir,) if cells_dir.is_dir() else None
    try:
        entries = read_ledger(ledger_path)
        mismatches = verify_ledger(ledger_path, run_dir, extra_roots=extra_roots)
    except LedgerError as exc:
        raise LayoutError([f"ledger.json: {exc}"]) from exc
    if mismatches:
        raise LayoutError(
            [f"ledger.json (§5 seal violated): {line}" for line in mismatches]
        )
    return f"ledger: verified ({len(entries)} sealed artifact(s), 0 mismatches, 0 extra)"


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via tmp + os.replace (crash cannot truncate)."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _build_provenance(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """The §3 provenance snapshot surfaced beside the index (task #129 / H8)."""
    return {key: manifest[key] for key in MANIFEST_REQUIRED_FIELDS}


def organize_run(
    run_dir: Path, index_dir: Path | None = None, *, force: bool = False
) -> tuple[Path, Path]:
    """Validate + index one pulled run; returns (cells_index.csv, coverage_report.md).

    Also writes ``provenance.json`` (the §3 snapshot) beside the index. All
    three outputs are written atomically (tmp + os.replace); an existing index
    is REFUSED unless ``force=True`` — silently re-indexing over a table other
    tooling may have already consumed is the drift this guard exists to stop.
    """
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise OrganizeError(f"run directory does not exist: {run_dir}")
    manifest = load_manifest(run_dir)
    records = walk_run_tree(run_dir, manifest)
    scoring_summary = validate_scoring_tree(run_dir)
    predicate_summary = validate_predicate_trees(run_dir)
    ledger_line = verify_run_ledger(run_dir)
    index = build_index(manifest, records)
    report = build_coverage_report(manifest, index)
    report += "\n".join(
        [
            "## Ledger (RESULTS_LAYOUT §5)",
            "",
            f"- {ledger_line}",
            "",
            "## Scoring passes (RESULTS_LAYOUT §6)",
            "",
            *(scoring_summary or ["none"]),
            "",
            "## Predicate tables (§8.5 join chain, task #119)",
            "",
            *(predicate_summary or ["none"]),
            "",
        ]
    )

    out_dir = Path(index_dir) if index_dir is not None else run_dir / INDEX_DIRNAME
    csv_path = out_dir / INDEX_CSV_NAME
    md_path = out_dir / COVERAGE_MD_NAME
    prov_path = out_dir / PROVENANCE_JSON_NAME
    existing = [p for p in (csv_path, md_path, prov_path) if p.exists()]
    if existing and not force:
        raise OrganizeError(
            f"index output(s) already exist ({', '.join(str(p) for p in existing)}) "
            "— refusing to silently overwrite an index other tooling may have "
            "consumed; pass --force to re-index deliberately"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(csv_path, index.to_csv(index=False))
    _atomic_write_text(md_path, report)
    _atomic_write_text(
        prov_path,
        json.dumps(_build_provenance(manifest), indent=2, sort_keys=True) + "\n",
    )
    return csv_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a pulled campaign run tree (cloud/RESULTS_LAYOUT.md), parse every "
            "cells/<row_key> through CellSpec, and emit index/cells_index.csv + "
            "index/coverage_report.md."
        )
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="pulled run root: results/<campaign>/<session>/<run_id>",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="output directory (default: <run_dir>/index)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing index (refused by default)",
    )
    args = parser.parse_args(argv)
    try:
        csv_path, md_path = organize_run(args.run_dir, args.index_dir, force=args.force)
    except OrganizeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"[organize_results] index   : {csv_path}")
    print(f"[organize_results] coverage: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
