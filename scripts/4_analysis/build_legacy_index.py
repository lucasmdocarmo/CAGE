#!/usr/bin/env python3
"""Bridge the READ-ONLY pilot 100x3 archive into a driver-consumable index.

Purpose (PUBLICATION.md §8.13 "$0 machinery dry-run" / §9.13 "dry-runs clean on
the pilot 100x3 archive"): re-key the legacy pilot tree
(``results/phase2/<run>/<tree>/<cell>/trial_<n>/results.csv``) through
``src.analysis.cellspec.from_legacy`` and emit a v2-shaped bridge run directory

    <out_dir>/
        manifest.json                # bridge provenance (NOT a §3 campaign manifest)
        index/cells_index.csv        # EXACT organize_results.INDEX_COLUMNS contract
        index/legacy_windows_map.csv # window -> (run, tree, legacy cell, trial) provenance
        index/skipped_cells.csv      # unmappable/empty cells, each with a reason
        index/bridge_report.md       # human summary, stamped
        cells/<row_key>/cell.json
        cells/<row_key>/window_<dataset>-<NN>/{requests.jsonl, qa_evidence.jsonl}

so ``run_campaign_analysis.py`` (design-input mode) consumes the pilot archive
UNCHANGED through its normal ``index/cells_index.csv`` handoff.

Hard rules honored here:
- The pilot archive is STRICTLY READ-ONLY: this tool only ever writes under
  ``out_dir`` and REFUSES an ``out_dir`` located inside the archive root.
- Every output is stamped DESIGN-INPUT-ONLY / calibration: bridged numbers are
  properties of the measurement machinery, never findings of the study.
- Fail-closed: unknown legacy names are skipped WITH a recorded reason (the
  ``from_legacy`` refusal text), never guessed; a run whose dataset cannot be
  inferred from its name fails loud; an empty archive fails loud.

Validity rule: the canonical pilot estimand (``_results_loader``) — a row
enters the bridge iff NOT error AND NOT empty_generation AND repeat_index is
0/absent (``headline_rows``). Dropped counts are recorded per window in
``legacy_windows_map.csv`` so nothing is silently absent.

Window semantics: one pilot ``trial_<n>`` = one bridge window (the §9.4
batch-means unit). Distinct legacy cells that map to the SAME CellSpec tuple
(e.g. ``prefix_cache`` ≡ ``cag_full``, §7.7a) MERGE into one cell directory;
their windows accumulate with sequential ordinals and full provenance in
``legacy_windows_map.csv`` + the cell's ``cell.json``.

Model aliasing (documented, deliberate): ``from_legacy`` re-keys the pilot's
Qwen3-8B (off the D4 roster) onto the charter anchor ``qwen3-14b``. The bridge
manifest and report both carry the actual served model so the alias can never
be mistaken for a claim about qwen3-14b.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for _p in (str(_HERE), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _results_loader as rl  # noqa: E402
from organize_results import BASELINE_OF_CELL, INDEX_COLUMNS  # noqa: E402
from src.analysis.cellspec import CellSpec, UnknownBaselineError, from_legacy  # noqa: E402

STAMP = "DESIGN-INPUT-ONLY"
BRIDGE_KIND = "pilot-legacy-bridge (§8.13 machinery dry-run; §9.7 calibration input)"
ACTUAL_PILOT_MODEL = "qwen3-8b (served: Qwen/Qwen3-8B — off the D4 roster)"
MODEL_ALIAS_NOTE = (
    "the 'model' axis carries the charter anchor alias 'qwen3-14b' assigned by "
    "cellspec.from_legacy; it is NOT the served model (see actual_pilot_model)"
)

#: §1 dataset ids a pilot run name may end with (``..._<dataset>``).
KNOWN_RUN_DATASETS: tuple[str, ...] = ("squad_v2", "hotpotqa", "musique", "qasper")

#: Per-request serving fields -> requests.jsonl (numeric only, NaN dropped).
REQUESTS_FIELDS: tuple[str, ...] = (
    "ttft_ms",
    "latency_ms",
    "tpot_ms",
    "num_tokens",
    "prompt_tokens",
    "cached_prompt_tokens",
    "cached_prompt_ratio",
    "settle_ms",
)

#: Per-request quality/evidence fields -> qa_evidence.jsonl.
QA_FIELDS: tuple[str, ...] = (
    "f1_score",
    "exact_match",
    "f1_answerable",
    "exact_match_answerable",
    "no_answer_correct",
    "abstention_precision",
    "grounding_score",
    "faithfulness",
    "context_relevance",
    "completeness_bertscore",
    "completeness_rouge_l",
    "hallucinated_span_ratio",
    "supported_claim_ratio",
    "compression_ratio",
    "compression_latency_ms",
)

#: S2 policy-event mask column (run_campaign_analysis.POLICY_EVENT_COLUMN):
#: derived from the pilot's boolean ``compression_applied`` column.
POLICY_EVENT_SOURCE = "compression_applied"
POLICY_EVENT_COLUMN = "policy_event"

_ARTIFACT_SEP = ";"


class BridgeError(RuntimeError):
    """Any refusal in the bridge (fail loud, message states the fix)."""


# ---------------------------------------------------------------------------
# Small typed records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkippedCell:
    """One legacy cell (or window) that did NOT enter the index, with a reason."""

    run: str
    tree: str
    legacy_cell: str
    reason: str


@dataclass(frozen=True)
class WindowProvenance:
    """One bridged window's audit trail back into the read-only archive."""

    run: str
    tree: str
    legacy_cell: str
    trial: int
    dataset: str
    row_key: str
    window_key: str
    n_rows_raw: int
    n_rows_valid_rep0: int
    n_error_rows: int
    n_empty_gen_rows: int
    n_rep_gt0_rows: int
    n_examples: int


@dataclass(frozen=True)
class BridgeResult:
    """What ``build_bridge`` produced (paths + honest accounting)."""

    out_dir: Path
    index_csv: Path
    report_md: Path
    n_runs: int
    n_cells_discovered: int
    n_cells_mapped: int
    n_cells_skipped: int
    n_windows: int
    n_rows_raw: int
    n_rows_bridged: int
    skipped: tuple[SkippedCell, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def infer_dataset(run_name: str) -> str:
    """Dataset id from a pilot run directory name (``..._<dataset>``); fail closed."""
    for dataset in KNOWN_RUN_DATASETS:
        if run_name.endswith(f"_{dataset}"):
            return dataset
    raise BridgeError(
        f"cannot infer the dataset from run name {run_name!r} — expected a "
        f"'_<dataset>' suffix among {list(KNOWN_RUN_DATASETS)}; the bridge "
        "refuses to guess a dataset axis"
    )


def _numeric_series(sub: pd.DataFrame, column: str) -> pd.Series | None:
    if column not in sub.columns:
        return None
    return pd.to_numeric(sub[column], errors="coerce")


def _policy_event_series(sub: pd.DataFrame) -> pd.Series | None:
    """1.0/0.0 S2 policy-event mask from the pilot boolean column, else None."""
    if POLICY_EVENT_SOURCE not in sub.columns:
        return None
    lowered = sub[POLICY_EVENT_SOURCE].astype(str).str.strip().str.lower()
    mapped = lowered.map({"true": 1.0, "false": 0.0})
    numeric = pd.to_numeric(sub[POLICY_EVENT_SOURCE], errors="coerce")
    return mapped.where(mapped.notna(), numeric)


def _records(
    sub: pd.DataFrame, fields: Sequence[str], extra: Mapping[str, pd.Series]
) -> list[dict[str, Any]]:
    """Per-row JSONL records: example_id (str) + finite numeric fields only."""
    example_ids = sub["example_id"].astype(str).str.strip()
    series: dict[str, pd.Series] = {}
    for name in fields:
        s = _numeric_series(sub, name)
        if s is not None:
            series[name] = s
    for name, s in extra.items():
        series[name] = s
    records: list[dict[str, Any]] = []
    for pos in range(len(sub)):
        example_id = example_ids.iat[pos]
        if not example_id:
            continue
        rec: dict[str, Any] = {"example_id": example_id}
        for name, s in series.items():
            value = s.iat[pos]
            if pd.notna(value) and math.isfinite(float(value)):
                rec[name] = float(value)
        if len(rec) > 1:
            records.append(rec)
    return records


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")


def _spec_index_row(
    spec: CellSpec,
    *,
    run_id: str,
    campaign: str,
    session: str,
    dataset: str,
    window: int,
    window_key: str,
    row_key: str,
    window_dir: str,
    cell_json: str,
    artifacts: Sequence[str],
) -> dict[str, Any]:
    """One cells_index.csv row — the EXACT organize_results column contract."""
    return {
        "run_id": run_id,
        "campaign": campaign,
        "session": session,
        "model": spec.model,
        "engine": spec.engine,
        "arm": spec.arm,
        "baseline": BASELINE_OF_CELL.get((spec.arm, spec.retriever), ""),
        "retriever": spec.retriever,
        "policy": spec.policy,
        "topology": spec.topology,
        "family": spec.family,
        "dataset": dataset,
        "budget_r": spec.budget_r,
        "rate_frac": spec.rate_frac,
        "window": window,
        "window_key": window_key,
        "row_key": row_key,
        "window_dir": window_dir,
        "cell_json": cell_json,
        "artifacts": _ARTIFACT_SEP.join(artifacts),
    }


# ---------------------------------------------------------------------------
# The bridge
# ---------------------------------------------------------------------------


def build_bridge(
    archive_root: Path,
    runs: Sequence[str],
    out_dir: Path,
    *,
    run_id: str = "pilot-100x3-bridge",
    campaign: str = "pilot-bridge",
    session: str = "pilot",
) -> BridgeResult:
    """Walk the read-only pilot runs into a driver-consumable bridge run dir."""
    archive_root = Path(archive_root).resolve()
    out_dir = Path(out_dir).resolve()
    if not archive_root.is_dir():
        raise BridgeError(f"archive root does not exist: {archive_root}")
    if not runs:
        raise BridgeError("no pilot runs given — nothing to bridge")
    if archive_root == out_dir or archive_root in out_dir.parents:
        raise BridgeError(
            f"out_dir {out_dir} lies inside the archive root {archive_root} — "
            "the pilot archive is READ-ONLY; write the bridge elsewhere "
            "(MyDocs/registration/... or the scratchpad)"
        )

    datasets = {run: infer_dataset(run) for run in runs}
    run_dirs: dict[str, Path] = {}
    for run in runs:
        run_dir = archive_root / run
        if not run_dir.is_dir():
            raise BridgeError(f"pilot run directory does not exist: {run_dir}")
        run_dirs[run] = run_dir

    index_rows: list[dict[str, Any]] = []
    provenance: list[WindowProvenance] = []
    skipped: list[SkippedCell] = []
    #: (row_key, dataset) -> next window ordinal (1-based).
    ordinals: dict[tuple[str, str], int] = {}
    #: row_key -> (spec, [legacy source labels]) for cell.json.
    cell_meta: dict[str, tuple[CellSpec, list[str]]] = {}
    n_cells_discovered = 0
    n_rows_raw_total = 0
    n_rows_bridged_total = 0

    out_dir.mkdir(parents=True, exist_ok=True)
    cells_root = out_dir / "cells"

    for run in runs:
        dataset = datasets[run]
        discovered = rl.discover_cells(run_dirs[run])
        if not discovered:
            raise BridgeError(
                f"run {run}: no legacy cells with results.csv found under "
                f"{run_dirs[run]} — an empty run bridges nothing"
            )
        for tree, cell_name, cell_dir in discovered:
            n_cells_discovered += 1
            tree_label = tree or ""
            try:
                spec = from_legacy(cell_name)
            except UnknownBaselineError as exc:
                skipped.append(SkippedCell(run, tree_label, cell_name, str(exc)))
                continue
            row_key = spec.to_row_key()
            df = rl.load_cell(cell_dir, cell_name)
            headline = rl.headline_rows(df)
            source_label = f"{run}/{tree_label}/{cell_name}"
            meta = cell_meta.setdefault(row_key, (spec, []))
            if source_label not in meta[1]:
                meta[1].append(source_label)

            trials = sorted(int(t) for t in df["trial"].unique())
            wrote_any_window = False
            for trial in trials:
                raw_trial = df[df["trial"] == trial]
                sub = headline[headline["trial"] == trial]
                n_rows_raw = int(len(raw_trial))
                n_rows_raw_total += n_rows_raw
                n_error = int(raw_trial["is_error_row"].sum())
                n_empty = int((~raw_trial["is_error_row"] & raw_trial["is_empty_gen"]).sum())
                n_rep = n_rows_raw - n_error - n_empty - int(len(sub))
                if sub.empty:
                    skipped.append(
                        SkippedCell(
                            run,
                            tree_label,
                            f"{cell_name}/trial_{trial}",
                            f"window dropped: 0 valid rep-0 rows of {n_rows_raw} raw "
                            f"(error={n_error}, empty_generation={n_empty}, "
                            f"rep>0={n_rep})",
                        )
                    )
                    continue

                policy_event = _policy_event_series(sub)
                extra = (
                    {POLICY_EVENT_COLUMN: policy_event}
                    if policy_event is not None
                    else {}
                )
                requests = _records(sub, REQUESTS_FIELDS, {})
                qa_evidence = _records(sub, QA_FIELDS, extra)
                if not requests:
                    skipped.append(
                        SkippedCell(
                            run,
                            tree_label,
                            f"{cell_name}/trial_{trial}",
                            "window dropped: no per-request serving record with an "
                            "example_id and at least one finite serving metric",
                        )
                    )
                    continue

                ordinal = ordinals.get((row_key, dataset), 0) + 1
                ordinals[(row_key, dataset)] = ordinal
                window_key = f"{dataset}-{ordinal:02d}"
                window_dir = cells_root / row_key / f"window_{window_key}"
                window_dir.mkdir(parents=True, exist_ok=True)
                _write_jsonl(window_dir / "requests.jsonl", requests)
                _write_jsonl(window_dir / "qa_evidence.jsonl", qa_evidence)

                window_rel = window_dir.relative_to(out_dir).as_posix()
                cell_json_rel = (
                    (cells_root / row_key / "cell.json").relative_to(out_dir).as_posix()
                )
                artifacts = (
                    f"{window_rel}/requests.jsonl",
                    f"{window_rel}/qa_evidence.jsonl",
                )
                index_rows.append(
                    _spec_index_row(
                        spec,
                        run_id=run_id,
                        campaign=campaign,
                        session=session,
                        dataset=dataset,
                        window=ordinal,
                        window_key=window_key,
                        row_key=row_key,
                        window_dir=window_rel,
                        cell_json=cell_json_rel,
                        artifacts=artifacts,
                    )
                )
                n_examples = int(sub["example_id"].astype(str).str.strip().nunique())
                provenance.append(
                    WindowProvenance(
                        run=run,
                        tree=tree_label,
                        legacy_cell=cell_name,
                        trial=trial,
                        dataset=dataset,
                        row_key=row_key,
                        window_key=window_key,
                        n_rows_raw=n_rows_raw,
                        n_rows_valid_rep0=int(len(sub)),
                        n_error_rows=n_error,
                        n_empty_gen_rows=n_empty,
                        n_rep_gt0_rows=n_rep,
                        n_examples=n_examples,
                    )
                )
                n_rows_bridged_total += int(len(sub))
                wrote_any_window = True
            if not wrote_any_window:
                skipped.append(
                    SkippedCell(
                        run,
                        tree_label,
                        cell_name,
                        "cell mapped but produced no bridgeable window",
                    )
                )

    if not index_rows:
        raise BridgeError(
            "the bridge produced ZERO index rows — every discovered cell was "
            "skipped; see the skipped list (nothing was written silently)"
        )

    # cell.json per merged cell (spec + full legacy provenance, stamped).
    for row_key, (spec, sources) in sorted(cell_meta.items()):
        cell_dir = cells_root / row_key
        if not cell_dir.is_dir():
            continue  # mapped cell whose every window was dropped
        payload = {
            "stamp": STAMP,
            "kind": BRIDGE_KIND,
            "cellspec": spec.to_flat_dict(),
            "baseline": BASELINE_OF_CELL.get((spec.arm, spec.retriever), ""),
            "legacy_sources": sorted(sources),
            "actual_pilot_model": ACTUAL_PILOT_MODEL,
            "model_alias_note": MODEL_ALIAS_NOTE,
        }
        (cell_dir / "cell.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    index_dir = out_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    index = pd.DataFrame(index_rows, columns=list(INDEX_COLUMNS)).sort_values(
        ["model", "dataset", "row_key", "window"], kind="mergesort"
    ).reset_index(drop=True)
    index_csv = index_dir / "cells_index.csv"
    index.to_csv(index_csv, index=False)

    pd.DataFrame([vars(p) for p in provenance]).to_csv(
        index_dir / "legacy_windows_map.csv", index=False
    )
    pd.DataFrame(
        [vars(s) for s in skipped],
        columns=["run", "tree", "legacy_cell", "reason"],
    ).to_csv(index_dir / "skipped_cells.csv", index=False)

    generated = datetime.now(timezone.utc).isoformat()
    manifest = {
        "stamp": STAMP,
        "kind": BRIDGE_KIND,
        "purpose": (
            "bridge of the READ-ONLY pilot 100x3 archive into the v2 "
            "cells_index.csv handoff so run_campaign_analysis.py dry-runs the "
            "full pipeline in design-input mode; numbers are measurement-"
            "machinery properties, never study findings"
        ),
        "run_id": run_id,
        "campaign": campaign,
        "session": session,
        "generated_utc": generated,
        "archive_root": str(archive_root),
        "source_runs": {run: datasets[run] for run in runs},
        "actual_pilot_model": ACTUAL_PILOT_MODEL,
        "model_alias_note": MODEL_ALIAS_NOTE,
        "validity_rule": (
            "canonical pilot estimand (_results_loader.headline_rows): NOT "
            "error AND NOT empty_generation AND repeat_index 0/absent"
        ),
        "window_semantics": "one pilot trial_<n> = one bridge window",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    n_mapped_cells = n_cells_discovered - len(
        [s for s in skipped if "/trial_" not in s.legacy_cell and "produced no" not in s.reason]
    )
    report_lines = [
        f"# Pilot legacy-bridge report — **{STAMP}**",
        "",
        f"- kind: {BRIDGE_KIND}",
        f"- generated (UTC): {generated}",
        f"- archive root (READ-ONLY): `{archive_root}`",
        f"- source runs: " + ", ".join(f"`{r}` ({datasets[r]})" for r in runs),
        f"- actual pilot model: {ACTUAL_PILOT_MODEL}",
        f"- model alias: {MODEL_ALIAS_NOTE}",
        "",
        "**Every number in this bridge is design-input / calibration on pilot "
        "data — a property of the measurement machinery, never a finding of "
        "the study.**",
        "",
        "## Accounting",
        "",
        f"- legacy cells discovered: {n_cells_discovered}",
        f"- legacy cells mapped through from_legacy: {n_mapped_cells}",
        f"- merged CellSpec cells (distinct row keys with data): "
        f"{index['row_key'].nunique()}",
        f"- windows indexed: {len(index)}",
        f"- raw archive rows walked: {n_rows_raw_total}",
        f"- rows bridged (valid, rep-0): {n_rows_bridged_total}",
        f"- skipped entries (cells + windows, each with a reason): {len(skipped)}",
        "",
        "## Skipped (honest list — never silently absent)",
        "",
    ]
    if skipped:
        for s in skipped:
            report_lines.append(
                f"- `{s.run}/{s.tree}/{s.legacy_cell}`: {s.reason}"
            )
    else:
        report_lines.append("- none")
    report_lines.append("")
    report_md = index_dir / "bridge_report.md"
    report_md.write_text("\n".join(report_lines), encoding="utf-8")

    return BridgeResult(
        out_dir=out_dir,
        index_csv=index_csv,
        report_md=report_md,
        n_runs=len(runs),
        n_cells_discovered=n_cells_discovered,
        n_cells_mapped=n_mapped_cells,
        n_cells_skipped=len({(s.run, s.tree, s.legacy_cell) for s in skipped}),
        n_windows=len(index),
        n_rows_raw=n_rows_raw_total,
        n_rows_bridged=n_rows_bridged_total,
        skipped=tuple(skipped),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_RUNS: tuple[str, ...] = (
    "2026-07-16_full_qwen3-8b_100x3_squad_v2",
    "2026-07-16_full_qwen3-8b_100x3_hotpotqa",
    "2026-07-16_full_qwen3-8b_100x3_musique",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Bridge the READ-ONLY pilot 100x3 archive into a v2-shaped run "
            "directory (index/cells_index.csv per the organize_results "
            "contract) for the §8.13 design-input machinery dry-run."
        )
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=_REPO_ROOT / "results" / "phase2",
        help="pilot archive root (READ-ONLY; default: results/phase2)",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=list(DEFAULT_RUNS),
        metavar="RUN",
        help="pilot run directory names (dataset inferred from the _<dataset> suffix)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="bridge run directory to create (MUST be outside the archive root)",
    )
    parser.add_argument("--run-id", default="pilot-100x3-bridge")
    parser.add_argument("--campaign", default="pilot-bridge")
    parser.add_argument("--session", default="pilot")
    args = parser.parse_args(argv)
    try:
        result = build_bridge(
            args.archive_root,
            args.runs,
            args.out_dir,
            run_id=args.run_id,
            campaign=args.campaign,
            session=args.session,
        )
    except BridgeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"[build_legacy_index] stamp   : {STAMP}")
    print(f"[build_legacy_index] index   : {result.index_csv}")
    print(f"[build_legacy_index] report  : {result.report_md}")
    print(
        f"[build_legacy_index] cells   : {result.n_cells_discovered} discovered, "
        f"{result.n_cells_mapped} mapped, {result.n_cells_skipped} skipped"
    )
    print(
        f"[build_legacy_index] windows : {result.n_windows} "
        f"({result.n_rows_bridged}/{result.n_rows_raw} rows bridged)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
