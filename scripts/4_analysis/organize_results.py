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

Fail-loud doctrine: layout violations (missing manifest/ledger/cells, missing
cell.json, malformed window dirs, unknown dataset ids, missing required window
artifacts, bad row keys) abort with every problem enumerated. Coverage GAPS do
not abort — a partial run is organizeable; the gap list is the deliverable.
Pure stdlib + pandas; no network.
"""

from __future__ import annotations

import argparse
import json
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
_SCORING_MANIFEST_REQUIRED_KEYS: tuple[str, ...] = (
    "scoring_run_id",
    "created_utc",
    "raw_run_ledger_entries_sha256",
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
_QA_EVIDENCE_EXEMPT_DATASETS: frozenset[str] = frozenset({"sharegpt"})

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

#: §3 manifest keys this organizer consumes. The spec's required set is larger
#: (git_sha, engine, seed, provider, hardware, ...) — those are provenance for
#: humans/stats and are passed through opaquely here. `model` is SINGULAR per
#: §3 (one run = one model; a re-run is a new run_id). There are NO
#: `models`/`datasets` list keys in the spec: datasets come from the window
#: directory names (§1); an OPTIONAL `datasets` list may narrow coverage.
_MANIFEST_STR_KEYS = ("campaign", "session", "run_id", "model")


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
        for window_dir in window_dirs:
            match = WINDOW_DIR_RE.match(window_dir.name)
            if not window_dir.is_dir() or match is None:
                problems.append(
                    f"cells/{cell_dir.name}/{window_dir.name}: expected a "
                    "window_<dataset>-<ordinal> directory (§1, e.g. window_squad_v2-01)"
                )
                continue
            dataset, ordinal = match.group(1), int(match.group(2))
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

    contamination = sorted(
        p.relative_to(run_dir).as_posix()
        for name in REQUIRED_SCORING_WINDOW_ARTIFACTS
        for p in (run_dir / "cells").rglob(name)
    )
    for path in contamination:
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
        if not scoring_dir.is_dir():
            problems.append(f"{prefix}: not a directory (stray file in scoring/)")
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def organize_run(run_dir: Path, index_dir: Path | None = None) -> tuple[Path, Path]:
    """Validate + index one pulled run; returns (cells_index.csv, coverage_report.md)."""
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise OrganizeError(f"run directory does not exist: {run_dir}")
    manifest = load_manifest(run_dir)
    records = walk_run_tree(run_dir, manifest)
    scoring_summary = validate_scoring_tree(run_dir)
    index = build_index(manifest, records)
    report = build_coverage_report(manifest, index)
    report += "\n".join(
        [
            "## Scoring passes (RESULTS_LAYOUT §6)",
            "",
            *(scoring_summary or ["none"]),
            "",
        ]
    )

    out_dir = Path(index_dir) if index_dir is not None else run_dir / INDEX_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / INDEX_CSV_NAME
    md_path = out_dir / COVERAGE_MD_NAME
    index.to_csv(csv_path, index=False)
    md_path.write_text(report, encoding="utf-8")
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
    args = parser.parse_args(argv)
    try:
        csv_path, md_path = organize_run(args.run_dir, args.index_dir)
    except OrganizeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"[organize_results] index   : {csv_path}")
    print(f"[organize_results] coverage: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
