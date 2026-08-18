#!/usr/bin/env python3
"""verify_results v2 — the campaign-tree verification gate (task #129, H6).

Given ONE pulled campaign run root (cloud/RESULTS_LAYOUT.md §1 — THE layout
authority), this gate checks, collecting EVERY problem instead of stopping at
the first:

(a) §1 schema conformance per window — requests.jsonl / qa_evidence.jsonl
    parse line-by-line with the required core identity field (``example_id``);
    rows carrying no ok/error validity field are a WARN, not a FAIL, with a
    pointer to tasks #119/#127 (the producer fix lands in parallel);
(b) requests-vs-evidence row-count reconciliation per window (H3: the pilot
    writer could lose an evidence row while keeping the results row);
(c) duplicate (example_id, repeat_index, record_index) identity detection
    (H3: open-loop replay duplicates example_ids BY DESIGN — without a
    disambiguating record_index the rows are indistinguishable);
(d) window coverage vs cell.json's ``windows[]`` table (§1) — every window
    directory declared, every declaration backed by a directory;
(e) a §9.10 exclusion-accounting summary (error / ok=False / empty_generation
    / validity-unknown row counts per window — absence is NOT coerced to 0);
(f) §5 ledger verification INCLUDING the H7 extra-file sweep over ``cells/``
    (files added after sealing are reported as EXTRA);
(g) the report is written OUTSIDE the tree — sibling
    ``<run_root>_verification/`` by default, ``--out`` to override; writing
    into the run root is REFUSED (the old tool dropped unsealed report files
    onto a sealed root);
(h) gate semantics — exit 0 only when no FAIL-severity finding exists
    (WARNs allowed); 1 on failure; 2 on usage/refusal.

``--pilot --results-dir DIR`` preserves the pilot-era metrics-vs-CSV check
(``verify_dir``) verbatim for pilot trees; that mode keeps writing its report
into the results dir as before (pilot trees are not sealed) unless ``--out``
is given.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for _p in (str(_HERE), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import organize_results as org  # noqa: E402
from src.analysis.stats.ledger import (  # noqa: E402
    LedgerError,
    read_ledger,
    verify_ledger,
)

#: Fields whose ABSENCE from every row of a per-query file is a WARN (not a
#: FAIL) until the producer fix lands — H2/#119: without them, serving-error
#: rows are indistinguishable from valid rows and offline scoring would
#: re-introduce zero-coercion.
_VALIDITY_FIELDS: tuple[str, ...] = ("ok", "error")
_PRODUCER_POINTER = "producer fix lands with tasks #119/#127"

#: §1 windows[] per-entry fields beyond ``dataset`` whose absence is a WARN
#: (producer task #126 lands in parallel).
_WINDOWS_ENTRY_WARN_FIELDS: tuple[str, ...] = ("seed", "rep", "t_start", "t_end")

#: Per-query artifacts subject to checks (a)-(c).
_PER_QUERY_ARTIFACTS: tuple[str, ...] = ("requests.jsonl", "qa_evidence.jsonl")

VERIFICATION_DIR_SUFFIX = "_verification"
REPORT_JSON_NAME = "verification_report.json"
REPORT_MD_NAME = "verification_report.md"


@dataclass(frozen=True)
class Finding:
    """One verification finding; ``FAIL`` findings flip the gate."""

    severity: Literal["FAIL", "WARN"]
    check: str
    where: str
    detail: str


class VerifyRefusal(RuntimeError):
    """Usage-level refusal (bad run dir, report targeted INTO the tree)."""


def _atomic_write_text(path: Path, text: str) -> None:
    """Write via tmp + os.replace so a crash can never truncate a report."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Per-window checks (a)-(c) + (e)
# ---------------------------------------------------------------------------


def _read_jsonl_objects(
    path: Path, rel: str, findings: list[Finding]
) -> list[dict[str, Any]]:
    """Parse a JSONL file; malformed lines / non-object rows are FAIL findings."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                findings.append(
                    Finding("FAIL", "schema", f"{rel}:{lineno}", f"invalid JSON: {exc}")
                )
                continue
            if not isinstance(obj, dict):
                findings.append(
                    Finding(
                        "FAIL",
                        "schema",
                        f"{rel}:{lineno}",
                        f"record must be a JSON object, got {type(obj).__name__}",
                    )
                )
                continue
            rows.append(obj)
    return rows


def _identity_key(row: dict[str, Any]) -> tuple[Any, Any, Any]:
    """The (example_id, repeat_index, record_index) row identity — components
    absent from the row stay None (absence is data, never coerced)."""
    return (row.get("example_id"), row.get("repeat_index"), row.get("record_index"))


def _check_per_query_file(
    rows: list[dict[str, Any]], rel: str, findings: list[Finding]
) -> None:
    """Checks (a) core fields + validity-field WARN and (c) duplicates for one file."""
    n_missing_id = sum(
        1
        for row in rows
        if not isinstance(row.get("example_id"), str) or not row.get("example_id")
    )
    if n_missing_id:
        findings.append(
            Finding(
                "FAIL",
                "schema",
                rel,
                f"{n_missing_id} row(s) lack a non-empty string 'example_id' "
                "(the §8 join key — unjoinable rows are unaccountable rows)",
            )
        )
    if rows and not any(
        any(field in row for field in _VALIDITY_FIELDS) for row in rows
    ):
        findings.append(
            Finding(
                "WARN",
                "schema",
                rel,
                "no row carries an ok/error validity field — serving-error rows "
                f"are indistinguishable from valid rows here ({_PRODUCER_POINTER})",
            )
        )

    seen: dict[tuple[Any, Any, Any], int] = {}
    for row in rows:
        key = _identity_key(row)
        seen[key] = seen.get(key, 0) + 1
    duplicates = {k: n for k, n in seen.items() if n > 1 and k[0] is not None}
    if duplicates:
        examples = "; ".join(
            f"(example_id={k[0]!r}, repeat_index={k[1]!r}, record_index={k[2]!r}) x{n}"
            for k, n in sorted(
                duplicates.items(), key=lambda item: (str(item[0][0]),)
            )[:3]
        )
        no_record_index = any(k[2] is None for k in duplicates)
        detail = (
            f"{len(duplicates)} duplicate (example_id, repeat_index, record_index) "
            f"identit(ies): {examples}"
        )
        if no_record_index:
            detail += (
                " — rows carry no disambiguating record_index (open-loop replay "
                "duplicates example_ids BY DESIGN; producer task #127)"
            )
        findings.append(Finding("FAIL", "duplicates", rel, detail))


def _check_window(
    run_dir: Path,
    window_dir: Path,
    dataset: str,
    findings: list[Finding],
) -> dict[str, Any]:
    """Run checks (a)-(c) + accounting (e) for one window; returns its summary."""
    rel_window = window_dir.relative_to(run_dir).as_posix()
    per_file_rows: dict[str, list[dict[str, Any]] | None] = {}
    for name in _PER_QUERY_ARTIFACTS:
        path = window_dir / name
        if not path.is_file():
            per_file_rows[name] = None
            if name == "qa_evidence.jsonl" and dataset in org.QA_EVIDENCE_EXEMPT_DATASETS:
                continue  # §1: ShareGPT windows carry serving streams only
            findings.append(
                Finding(
                    "FAIL",
                    "schema",
                    f"{rel_window}/{name}",
                    "required §1 window artifact is missing",
                )
            )
            continue
        rows = _read_jsonl_objects(path, f"{rel_window}/{name}", findings)
        per_file_rows[name] = rows
        _check_per_query_file(rows, f"{rel_window}/{name}", findings)

    # Non-per-query artifacts: JSON validity only.
    engine_metrics = window_dir / "engine_metrics.json"
    if engine_metrics.is_file():
        try:
            json.loads(engine_metrics.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            findings.append(
                Finding(
                    "FAIL",
                    "schema",
                    f"{rel_window}/engine_metrics.json",
                    f"invalid JSON: {exc}",
                )
            )
    else:
        findings.append(
            Finding(
                "FAIL",
                "schema",
                f"{rel_window}/engine_metrics.json",
                "required §1 window artifact is missing",
            )
        )
    cage_stats = window_dir / "cage_stats.jsonl"
    if cage_stats.is_file():
        _read_jsonl_objects(cage_stats, f"{rel_window}/cage_stats.jsonl", findings)
    else:
        findings.append(
            Finding(
                "FAIL",
                "schema",
                f"{rel_window}/cage_stats.jsonl",
                "required §1 window artifact is missing",
            )
        )

    # (b) requests-vs-evidence reconciliation.
    requests_rows = per_file_rows.get("requests.jsonl")
    evidence_rows = per_file_rows.get("qa_evidence.jsonl")
    if requests_rows is not None and evidence_rows is not None:
        if len(requests_rows) != len(evidence_rows):
            req_keys = {_identity_key(r) for r in requests_rows}
            ev_keys = {_identity_key(r) for r in evidence_rows}
            lost = sorted(
                (k for k in req_keys - ev_keys), key=lambda k: (str(k[0]),)
            )[:5]
            detail = (
                f"requests.jsonl has {len(requests_rows)} row(s) but "
                f"qa_evidence.jsonl has {len(evidence_rows)} — an evidence append "
                "was lost or extra rows were injected (H3 §9.10 accounting)"
            )
            if lost:
                detail += f"; first unmatched request identit(ies): {lost}"
            findings.append(Finding("FAIL", "reconciliation", rel_window, detail))

    # (e) §9.10 exclusion accounting — absence is NOT zero: rows lacking any
    # validity field are counted as validity-unknown, never as valid.
    accounting: dict[str, Any] = {
        "window": rel_window,
        "dataset": dataset,
        "n_requests_rows": None,
        "n_evidence_rows": None,
        "n_error": None,
        "n_ok_false": None,
        "n_empty_generation": None,
        "n_validity_unknown": None,
        "n_valid_known": None,
    }
    if requests_rows is not None:
        n_error = sum(1 for r in requests_rows if r.get("error"))
        n_ok_false = sum(1 for r in requests_rows if r.get("ok") is False)
        n_empty = sum(1 for r in requests_rows if r.get("empty_generation"))
        n_unknown = sum(
            1
            for r in requests_rows
            if not any(field in r for field in _VALIDITY_FIELDS)
        )
        n_valid_known = sum(
            1
            for r in requests_rows
            if any(field in r for field in _VALIDITY_FIELDS)
            and not r.get("error")
            and r.get("ok") is not False
            and not r.get("empty_generation")
        )
        accounting.update(
            n_requests_rows=len(requests_rows),
            n_error=n_error,
            n_ok_false=n_ok_false,
            n_empty_generation=n_empty,
            n_validity_unknown=n_unknown,
            n_valid_known=n_valid_known,
        )
    if evidence_rows is not None:
        accounting["n_evidence_rows"] = len(evidence_rows)
    return accounting


# ---------------------------------------------------------------------------
# Tree walk + (d) windows[] coverage
# ---------------------------------------------------------------------------


def _check_windows_table(
    cell_rel: str,
    meta: dict[str, Any] | None,
    dir_keys: dict[str, str],
    findings: list[Finding],
) -> None:
    """(d) cell.json ``windows[]`` vs the window directories, both directions."""
    if meta is None:
        return  # unreadable cell.json already produced a FAIL finding
    windows = meta.get("windows")
    if not isinstance(windows, dict) or not windows:
        findings.append(
            Finding(
                "FAIL",
                "window-coverage",
                f"{cell_rel}/cell.json",
                "no §1 windows[] table (k -> {dataset, seed, rep, t_start, "
                "t_end}) — window metadata is unrecoverable (producer task #126)",
            )
        )
        return
    declared = set(windows)
    present = set(dir_keys)
    for key in sorted(present - declared):
        findings.append(
            Finding(
                "FAIL",
                "window-coverage",
                f"{cell_rel}/{dir_keys[key]}",
                f"window directory has no windows[{key!r}] entry in cell.json",
            )
        )
    for key in sorted(declared - present):
        findings.append(
            Finding(
                "FAIL",
                "window-coverage",
                f"{cell_rel}/cell.json",
                f"windows[{key!r}] declared but no window_{key} directory exists",
            )
        )
    for key in sorted(declared & present):
        entry = windows[key]
        if not isinstance(entry, dict):
            findings.append(
                Finding(
                    "FAIL",
                    "window-coverage",
                    f"{cell_rel}/cell.json",
                    f"windows[{key!r}] must be an object, got {type(entry).__name__}",
                )
            )
            continue
        dataset_from_key = key.rsplit("-", 1)[0]
        if "dataset" not in entry or entry["dataset"] != dataset_from_key:
            findings.append(
                Finding(
                    "FAIL",
                    "window-coverage",
                    f"{cell_rel}/cell.json",
                    f"windows[{key!r}].dataset = {entry.get('dataset')!r} does not "
                    f"match the window name's dataset {dataset_from_key!r}",
                )
            )
        absent = [f for f in _WINDOWS_ENTRY_WARN_FIELDS if f not in entry]
        if absent:
            findings.append(
                Finding(
                    "WARN",
                    "window-coverage",
                    f"{cell_rel}/cell.json",
                    f"windows[{key!r}] lacks {absent} — per-window seed/rep/"
                    "t-bounds unrecoverable (producer task #126)",
                )
            )


def _walk_cells(
    run_dir: Path, findings: list[Finding]
) -> list[dict[str, Any]]:
    """Walk cells/ running checks (a)-(e); returns per-window accounting rows."""
    cells_dir = run_dir / "cells"
    if not cells_dir.is_dir():
        findings.append(
            Finding("FAIL", "layout", "cells", "cells/ directory missing (§1)")
        )
        return []
    accounting_rows: list[dict[str, Any]] = []
    n_windows = 0
    for cell_dir in sorted(p for p in cells_dir.iterdir() if not p.name.startswith(".")):
        cell_rel = f"cells/{cell_dir.name}"
        if not cell_dir.is_dir():
            findings.append(
                Finding("FAIL", "layout", cell_rel, "stray file in cells/ (§1)")
            )
            continue
        try:
            org.parse_row_key_dir(cell_dir.name)
        except org.OrganizeError as exc:
            findings.append(Finding("FAIL", "layout", cell_rel, str(exc)))

        meta: dict[str, Any] | None = None
        meta_path = cell_dir / org.CELL_META_NAME
        if not meta_path.is_file():
            findings.append(
                Finding(
                    "FAIL", "layout", f"{cell_rel}/cell.json", "missing cell.json (§1)"
                )
            )
        else:
            try:
                loaded = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                findings.append(
                    Finding(
                        "FAIL", "layout", f"{cell_rel}/cell.json", f"invalid JSON: {exc}"
                    )
                )
            else:
                if isinstance(loaded, dict):
                    meta = loaded
                else:
                    findings.append(
                        Finding(
                            "FAIL",
                            "layout",
                            f"{cell_rel}/cell.json",
                            f"root must be an object, got {type(loaded).__name__}",
                        )
                    )

        dir_keys: dict[str, str] = {}  # window_key -> dir name
        seen_identity: dict[tuple[str, int], str] = {}
        for window_dir in sorted(
            p
            for p in cell_dir.iterdir()
            if not p.name.startswith(".") and p.name != org.CELL_META_NAME
        ):
            match = org.WINDOW_DIR_RE.match(window_dir.name)
            if not window_dir.is_dir() or match is None:
                findings.append(
                    Finding(
                        "FAIL",
                        "layout",
                        f"{cell_rel}/{window_dir.name}",
                        "expected a window_<dataset>-<ordinal> directory (§1)",
                    )
                )
                continue
            dataset, ordinal_str = match.group(1), match.group(2)
            identity = (dataset, int(ordinal_str))
            if identity in seen_identity:
                findings.append(
                    Finding(
                        "FAIL",
                        "layout",
                        f"{cell_rel}/{window_dir.name}",
                        f"window identity {identity} collides with sibling "
                        f"{seen_identity[identity]!r} — the §8 join key would be "
                        "ambiguous (H12)",
                    )
                )
                continue
            seen_identity[identity] = window_dir.name
            if dataset not in org.DATASET_IDS:
                findings.append(
                    Finding(
                        "FAIL",
                        "layout",
                        f"{cell_rel}/{window_dir.name}",
                        f"dataset {dataset!r} is not a §1 dataset id "
                        f"({sorted(org.DATASET_IDS)})",
                    )
                )
                continue
            dir_keys[f"{dataset}-{ordinal_str}"] = window_dir.name
            accounting_rows.append(
                _check_window(run_dir, window_dir, dataset, findings)
            )
            n_windows += 1
        _check_windows_table(cell_rel, meta, dir_keys, findings)
    if n_windows == 0:
        findings.append(
            Finding(
                "FAIL",
                "layout",
                "cells",
                "no window_<dataset>-<ordinal> directories anywhere — an empty "
                "run verifies nothing",
            )
        )
    return accounting_rows


# ---------------------------------------------------------------------------
# (f) ledger verification
# ---------------------------------------------------------------------------


def _check_ledger(run_dir: Path, findings: list[Finding]) -> dict[str, Any]:
    """§5 seal verification incl. the H7 EXTRA sweep scoped to cells/."""
    ledger_path = run_dir / "ledger.json"
    summary: dict[str, Any] = {"present": ledger_path.is_file(), "n_entries": None}
    if not ledger_path.is_file():
        findings.append(
            Finding(
                "FAIL",
                "ledger",
                "ledger.json",
                "absent — the run is unsealed (§5: sealed at run end, BEFORE "
                "any analysis touches the data)",
            )
        )
        return summary
    cells_dir = run_dir / "cells"
    extra_roots = (cells_dir,) if cells_dir.is_dir() else None
    try:
        entries = read_ledger(ledger_path)
        mismatches = verify_ledger(ledger_path, run_dir, extra_roots=extra_roots)
    except LedgerError as exc:
        findings.append(Finding("FAIL", "ledger", "ledger.json", str(exc)))
        return summary
    summary["n_entries"] = len(entries)
    for line in mismatches:
        findings.append(Finding("FAIL", "ledger", "ledger.json", line))
    if "manifest.json" not in entries:
        findings.append(
            Finding(
                "WARN",
                "ledger",
                "ledger.json",
                "manifest.json is not among the sealed entries (§5 seals every "
                "artifact under cells/ PLUS manifest.json)",
            )
        )
    return summary


# ---------------------------------------------------------------------------
# The v2 gate
# ---------------------------------------------------------------------------


def verify_run(run_dir: Path) -> dict[str, Any]:
    """Run every campaign-gate check over one run root; returns the report."""
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise VerifyRefusal(f"run directory does not exist: {run_dir}")
    findings: list[Finding] = []

    try:
        org.load_manifest(run_dir)
    except org.LayoutError as exc:
        for problem in exc.problems:
            findings.append(Finding("FAIL", "manifest", "manifest.json", problem))

    accounting_rows = _walk_cells(run_dir, findings)
    ledger_summary = _check_ledger(run_dir, findings)

    # Task #119: the §8.5 predicate tables (predicate/<scoring_run_id>/) —
    # schema-guarded exactly like organize time (mirror-only rule, manifest
    # keys, seal cross-match, tri-state predicate values, own ledger).
    try:
        predicate_summary = org.validate_predicate_trees(run_dir)
    except org.LayoutError as exc:
        predicate_summary = []
        for problem in exc.problems:
            findings.append(Finding("FAIL", "predicate", "predicate/", problem))

    totals: dict[str, Any] = {"n_windows": len(accounting_rows)}
    for key in (
        "n_requests_rows",
        "n_evidence_rows",
        "n_error",
        "n_ok_false",
        "n_empty_generation",
        "n_validity_unknown",
        "n_valid_known",
    ):
        known = [row[key] for row in accounting_rows if row[key] is not None]
        # Absence-is-not-zero: a total over windows with unknown counts is
        # itself unknown; report the partial sum with its coverage.
        totals[key] = {
            "sum_over_known_windows": int(sum(known)) if known else None,
            "n_windows_known": len(known),
        }

    n_fail = sum(1 for f in findings if f.severity == "FAIL")
    n_warn = sum(1 for f in findings if f.severity == "WARN")
    return {
        "verifier": "verify_results v2 (task #129)",
        "layout_authority": "cloud/RESULTS_LAYOUT.md",
        "run_dir": str(run_dir),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ok": n_fail == 0,
        "n_fail": n_fail,
        "n_warn": n_warn,
        "findings": [asdict(f) for f in findings],
        "accounting": {"per_window": accounting_rows, "totals": totals},
        "ledger": ledger_summary,
        "predicate_tables": predicate_summary,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Human-readable companion to the JSON report."""
    lines = [
        "# Campaign run verification report",
        "",
        f"- run: `{report['run_dir']}`",
        f"- verdict: **{'PASS' if report['ok'] else 'FAIL'}** "
        f"({report['n_fail']} FAIL, {report['n_warn']} WARN)",
        f"- windows checked: {report['accounting']['totals']['n_windows']}",
        f"- ledger: "
        + (
            f"verified ({report['ledger']['n_entries']} sealed entries)"
            if report["ledger"]["present"] and report["ledger"]["n_entries"] is not None
            else ("present but unusable" if report["ledger"]["present"] else "ABSENT")
        ),
        "",
    ]
    for severity in ("FAIL", "WARN"):
        selected = [f for f in report["findings"] if f["severity"] == severity]
        lines.append(f"## {severity} findings ({len(selected)})")
        lines.append("")
        for f in selected:
            lines.append(f"- [{f['check']}] `{f['where']}`: {f['detail']}")
        if not selected:
            lines.append("none")
        lines.append("")
    lines.append("## §9.10 exclusion accounting (per window)")
    lines.append("")
    lines.append(
        "| window | requests | evidence | error | ok=False | empty | "
        "validity-unknown | valid-known |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")

    def _cell(value: Any) -> str:
        return "?" if value is None else str(value)

    for row in report["accounting"]["per_window"]:
        lines.append(
            f"| {row['window']} | {_cell(row['n_requests_rows'])} "
            f"| {_cell(row['n_evidence_rows'])} | {_cell(row['n_error'])} "
            f"| {_cell(row['n_ok_false'])} | {_cell(row['n_empty_generation'])} "
            f"| {_cell(row['n_validity_unknown'])} | {_cell(row['n_valid_known'])} |"
        )
    lines.append("")
    lines.append(
        "('?' = field absent at the producer — UNKNOWN, never coerced to 0; "
        + _PRODUCER_POINTER
        + ")"
    )
    lines.append("")
    return "\n".join(lines)


def resolve_out_dir(run_dir: Path, out: Path | None) -> Path:
    """Default: sibling ``<run_root>_verification/``; NEVER inside the run root."""
    run_dir = Path(run_dir).resolve()
    out_dir = (
        Path(out).resolve()
        if out is not None
        else run_dir.parent / f"{run_dir.name}{VERIFICATION_DIR_SUFFIX}"
    )
    if out_dir == run_dir or run_dir in out_dir.parents:
        raise VerifyRefusal(
            f"refusing to write the verification report into the run tree "
            f"({out_dir} is inside {run_dir}) — reports live OUTSIDE the "
            "sealed root; pick another --out"
        )
    return out_dir


def write_report(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / REPORT_JSON_NAME
    md_path = out_dir / REPORT_MD_NAME
    _atomic_write_text(json_path, json.dumps(report, indent=2) + "\n")
    _atomic_write_text(md_path, render_markdown(report))
    return json_path, md_path


# ---------------------------------------------------------------------------
# Pilot mode — the pre-v2 metrics-vs-CSV check, preserved verbatim for pilot
# trees (results/phase2/... ; RESULTS_LAYOUT §7 read-only historical data).
# ---------------------------------------------------------------------------


def verify_dir(results_dir: Path) -> dict:
    report = {
        "results_dir": str(results_dir),
        "checks": [],
        "ok": True,
    }

    # Descend into trial_*/ -- multi-trial runs write per-trial
    # <label>_<dataset>_<ts>_metrics.json under trial_N/, not at the cell root. Exclude the
    # cell-root aggregated_metrics.json (it has no sibling *_results.csv). Hard-fail on zero
    # matches so a misdirected --results-dir cannot silently pass (audit false-pass fix).
    #
    # Review fix: Path.rglob does NOT traverse directory symlinks, and run_phase2_stats.sh
    # builds exactly such a symlink tree (`ln -sfn ... stats/all_results`) -- rglob would
    # silently see zero files through it (same class of bug as _results_loader.py's own
    # documented iterdir()+glob-not-rglob rule). Use a followlinks=True os.walk instead, which
    # preserves rglob's arbitrary-depth semantics while traversing symlinked subtrees.
    metrics_files = [
        Path(dirpath) / fn
        for dirpath, _dirnames, filenames in os.walk(results_dir, followlinks=True)
        for fn in sorted(filenames)
        if fn.endswith("_metrics.json") and fn != "aggregated_metrics.json"
    ]
    metrics_files.sort()
    if not metrics_files:
        report["ok"] = False
        report["errors"] = ["no_per_trial_metrics_found"]

    for metrics_path in metrics_files:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        baseline = metrics.get("experiment", {}).get("baseline")
        expected_requests = metrics.get("performance", {}).get("total_requests")
        dataset = metrics.get("experiment", {}).get("dataset")
        model = metrics.get("experiment", {}).get("model")

        csv_path = metrics_path.with_name(metrics_path.name.replace("_metrics.json", "_results.csv"))
        check = {
            "baseline": baseline,
            "dataset": dataset,
            "model": model,
            "metrics_file": str(metrics_path),
            "csv_file": str(csv_path),
            "expected_requests": expected_requests,
            "actual_rows": None,
            "ok": True,
            "errors": [],
        }

        if not csv_path.exists():
            check["ok"] = False
            check["errors"].append("missing_results_csv")
        else:
            df = pd.read_csv(csv_path)
            check["actual_rows"] = int(len(df))
            if expected_requests is not None and check["actual_rows"] != expected_requests:
                check["ok"] = False
                check["errors"].append("row_count_mismatch")

        report["checks"].append(check)
        if not check["ok"]:
            report["ok"] = False

    # Metric-coverage section (2026-07-15 audit): per cell x key metric, how many valid
    # rows actually carry a value, split by trial. Makes coverage pathologies visible
    # (e.g. the fixture's bertscore 1/3/1 rows per trial, silently averaged before) so a
    # sparse metric can't masquerade as a well-estimated one. Advisory: never flips ok.
    try:
        from _results_loader import load_results_long, metric_values, valid_rows

        cov_metrics = ["grounding_score", "faithfulness", "completeness_bertscore",
                       "completeness_rouge_l", "ttft_ms", "abstention_precision"]
        long_df = load_results_long(results_dir)
        v = valid_rows(long_df)
        coverage = []
        for cell, df_cell in v.groupby("cell", sort=True):
            for metric in cov_metrics:
                scored = metric_values(df_cell, metric).notna()
                by_trial = {int(t): int(scored[df_cell["trial"] == t].sum())
                            for t in sorted(df_cell["trial"].unique())}
                coverage.append({
                    "cell": cell, "metric": metric,
                    "n_valid_rows": int(len(df_cell)),
                    "n_scored": int(scored.sum()),
                    "per_trial_scored": by_trial,
                })
        report["metric_coverage"] = coverage
    except SystemExit:
        pass  # no results.csv trees under this dir (e.g. bare metrics check) -- skip
    except Exception as exc:  # advisory section must never break verification
        report["metric_coverage_error"] = f"{type(exc).__name__}: {exc}"

    return report


def _run_pilot(results_dir: Path, out: Path | None) -> int:
    """The old CLI behavior: verify_dir + reports (into the tree, as before,
    unless --out redirects them) — plus gate-semantics exit codes."""
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        print(f"ERROR: results directory does not exist: {results_dir}", file=sys.stderr)
        return 2
    report = verify_dir(results_dir)
    out_dir = Path(out) if out is not None else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "verification_report.json"
    _atomic_write_text(report_path, json.dumps(report, indent=2))

    txt_lines = [
        f"Results dir: {results_dir}",
        f"Overall OK: {report['ok']}",
        "",
    ]
    for check in report["checks"]:
        txt_lines.append(
            f"{check['baseline']} | rows={check['actual_rows']} "
            f"expected={check['expected_requests']} | ok={check['ok']}"
        )
        if check["errors"]:
            txt_lines.append(f"  errors: {', '.join(check['errors'])}")
    for cov in report.get("metric_coverage", []):
        if cov["n_scored"] < cov["n_valid_rows"]:
            txt_lines.append(
                f"COVERAGE {cov['cell']} {cov['metric']}: "
                f"{cov['n_scored']}/{cov['n_valid_rows']} rows scored "
                f"(per-trial {cov['per_trial_scored']})"
            )
    txt_path = out_dir / "verification_report.txt"
    _atomic_write_text(txt_path, "\n".join(txt_lines) + "\n")

    print(f"Wrote {report_path}")
    print(f"Wrote {txt_path}")
    return 0 if report["ok"] else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a campaign run tree (cloud/RESULTS_LAYOUT.md) as the "
            "pre-analysis gate: schema, reconciliation, duplicates, windows[] "
            "coverage, §9.10 accounting, §5 ledger + EXTRA sweep. Exit 0 only "
            "on PASS. --pilot preserves the pilot-era metrics-vs-CSV check."
        )
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        help="campaign run root: results/<campaign>/<session>/<run_id>",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="report directory (default: sibling <run_root>_verification/; "
        "must be OUTSIDE the run tree)",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="pilot-era metrics-vs-CSV mode for results/phase2 trees (§7)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="(--pilot only) pilot results directory",
    )
    args = parser.parse_args(argv)

    if args.pilot:
        if args.results_dir is None:
            parser.error("--pilot requires --results-dir")
        if args.run_dir is not None:
            parser.error("--pilot takes --results-dir, not a positional run_dir")
        return _run_pilot(args.results_dir, args.out)

    if args.run_dir is None:
        parser.error("run_dir is required (or use --pilot --results-dir)")
    if args.results_dir is not None:
        parser.error("--results-dir is --pilot-only; pass the run root positionally")

    try:
        out_dir = resolve_out_dir(args.run_dir, args.out)
        report = verify_run(args.run_dir)
    except VerifyRefusal as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    json_path, md_path = write_report(report, out_dir)
    verdict = "PASS" if report["ok"] else "FAIL"
    print(f"[verify_results] {verdict}: {report['n_fail']} FAIL, {report['n_warn']} WARN")
    print(f"[verify_results] report : {json_path}")
    print(f"[verify_results] summary: {md_path}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
