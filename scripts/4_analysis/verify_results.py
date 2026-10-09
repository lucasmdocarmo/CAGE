#!/usr/bin/env python3
"""Order:     stage 4 — FIRST gate on a pulled campaign run, before organize_results.py (pilot mode: --pilot)
Objective: Fail-closed campaign-tree verification (schema, reconciliation, duplicates, coverage, ledger, contamination) -> report OUTSIDE the tree
Cloud:     local

verify_results v2 — the campaign-tree verification gate (task #129, H6).

Given ONE pulled campaign run root (docs/RESULTS_LAYOUT.md §1 — THE layout
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
    (WARNs allowed); 1 on failure; 2 on usage/refusal;
(i) row count vs the offered population per window (ADR-0116, Batch 2 W3):
    a request dropped by a per-query guard is absent from requests.jsonl AND
    qa_evidence.jsonl, so (b) cannot see it; the window's runner summary
    (metrics.json) carries the offered population (closed loop:
    experiment.num_measured_requests; open loop: workload.open_loop.n_scheduled)
    and the task #127 consort counters, and both must agree with the rows;
(j) per-row TPOT vs the output-token count (ADR-0118, Batch 2 W5): every
    completed (ok) row must carry an integer ``num_tokens``; with two or more
    output tokens its ``tpot_ms`` must be a finite positive number (a null or
    zero TPOT beside a decode phase is a captured-timing defect); with at
    most one output token its ``tpot_ms`` must be null (no decode phase; the
    row is timely on TTFT alone downstream and counted as ``n_no_decode`` in
    the accounting); a whitespace-sourced token count
    (``num_tokens_source == "whitespace"``) is a WARN naming the missing
    usage chunk;
(k) per-window pd transfer proof (ADR-0134, S0F-22 Batch 2): a window of a
    cell with topology ``pd`` must carry ``metrics.json["pd_transfer"]``, the
    runner's two-scrape record of the decode's NIXL counters; its deltas are
    re-derived here with the producer's own rule (bytes moved, at least one
    transfer per served row, zero failed transfers and notifications, prompt
    tokens attributed to the external KV source), the recorded ``verified``
    flag must agree with that re-derivation, and the recorded served-row
    count must equal the rows without ``error``; a pd window with no summary
    or no record FAILs (never the WARN of (i) alone); a window of any other
    topology carrying the record FAILs (a mislabeled cell);
(l) served rows ended with stop or length (ADR-0135, S0F-25): a
    requests.jsonl row without ``error`` must carry ``finish_reason`` ``stop``
    or ``length``, the adapter's served set; any other value (none, ``abort``,
    ``repetition``, or the adapter's own ``error`` beside an empty error
    text) FAILs, because that request failed and the row says it was served;
    rows that carry no ``finish_reason`` at all are a WARN (the rule cannot
    be applied to them);
(m) telemetry that certified nothing is named (ADR-0136, S0F-26): a window
    of a sampled engine (vLLM, SGLang) whose ``regime.json`` label is
    ``UNKNOWN_TELEMETRY`` is a WARN carrying the recorded refusal reason, and
    every window's label rides the accounting. A WARN, not a FAIL: the
    window's latency rows stay valid for contrasts that need no regime label
    (S0: all 40 windows read UNKNOWN_TELEMETRY and this gate said nothing);
(n) served text that is thinking scaffolding is named (ADR-0151, S0F-59): a
    requests.jsonl row without ``error`` whose ``generated_answer`` carries
    the chat template's thinking-open marker ``<think>`` answered with
    reasoning scaffolding under the stop sequence, never with an answer
    (the 2026-10-08 landing: 2,450 of 2,450 SGLang rows, 2 tokens each); a
    window where half or more of the served rows carry it FAILs, fewer is a
    WARN, and the median ``num_tokens`` over served rows rides the accounting
    (the landing's signature was a median of 2 against 4 to 29 on vLLM).

``--pilot --results-dir DIR`` preserves the pilot-era metrics-vs-CSV check
(``verify_dir``) verbatim for pilot trees; that mode keeps writing its report
into the results dir as before (pilot trees are not sealed) unless ``--out``
is given.
"""

from __future__ import annotations

import argparse
import json
import math
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
#: Check (k) re-derives the pd transfer verdict with the PRODUCER's rule
#: (one function, two callers: the runner's refusal and this gate), so the
#: offline verdict cannot drift from the one the window was emitted under.
from src.monitoring.vllm_telemetry import (  # noqa: E402
    pd_transfer_reasons as _pd_transfer_reasons,
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

#: The per-window runner summary campaign_session writes beside the §1
#: artifacts (campaign_session.WINDOW_METRICS_NAME; tests pin the two equal):
#: check (i) reads the offered population and the consort counters from it.
_WINDOW_METRICS_NAME = "metrics.json"
#: metrics.json["consort"] counters that must all be zero on a campaign window
#: (ADR-0116, Batch 2 W3; = campaign_session.CONSORT_COUNTERS, pinned equal).
_CONSORT_COUNTERS: tuple[str, ...] = (
    "n_dropped_prepare",
    "n_dropped_record",
    "n_dropped_turn",
    "evidence_write_failures",
)

#: The producer's whitespace-fallback label for ``num_tokens_source``
#: (= src.inference.engine.NUM_TOKENS_SOURCE_WHITESPACE; tests pin the two
#: equal): the engine returned no usage.completion_tokens, so the count is words.
_NUM_TOKENS_SOURCE_FALLBACK = "whitespace"

#: metrics.json key of the per-window pd transfer proof (= run_experiment
#: .PD_TRANSFER_KEY, pinned equal by tests) and the topology that must carry
#: it (src.analysis.cellspec Topology literal), check (k).
_PD_TRANSFER_KEY = "pd_transfer"
_PD_TOPOLOGY = "pd"

#: Check (l): the finish reasons of a served row (= src.inference
#: .openai_chat_adapter.SERVED_FINISH_REASONS; tests pin the two equal, the
#: adapter is not imported here so the gate needs no HTTP client stack).
_SERVED_FINISH_REASONS: tuple[str, ...] = ("stop", "length")

#: Check (m): the window's regime artifact (campaign_layout.write_window_regime
#: writes ``window_dir / "regime.json"``), the label of a window whose
#: telemetry certified nothing (= regime_inputs.REGIME_UNKNOWN) and the
#: engines whose campaign windows are sampled (= run_experiment
#: .CAMPAIGN_TELEMETRY_BACKENDS; the in-process hf oracle has no series by
#: design). All three pinned equal by tests.
_REGIME_NAME = "regime.json"
_REGIME_UNKNOWN = "UNKNOWN_TELEMETRY"
_TELEMETRY_ENGINES: tuple[str, ...] = ("vllm", "sglang")

#: Check (n): the chat template's thinking-open marker (Qwen3; the vLLM and
#: SGLang adapters pin enable_thinking=false, ADR-0151; = scripts/checks/
#: probe_thinking_pin.THINKING_MARKER, pinned equal by tests) and the share of
#: served rows carrying it at which a window FAILs rather than WARNs.
_THINKING_MARKER = "<think>"
_DEGENERATE_SHARE_FAIL = 0.5

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


def expected_row_count(metrics: dict[str, Any]) -> tuple[int | None, str]:
    """The number of requests.jsonl rows a window MUST carry, from its runner
    summary: every offered request produces exactly one row (a result row or,
    open loop, a dispatch stub). Closed loop: ``experiment.num_measured_requests``;
    open loop: ``workload.open_loop.n_scheduled`` (campaign mode refuses the
    warm-up trim, so no scheduled arrival is filtered). Returns
    ``(count, source)``, or ``(None, why)`` when the summary lacks the field
    (absence is unknown, never coerced)."""
    workload = metrics.get("workload")
    workload = workload if isinstance(workload, dict) else {}
    if workload.get("mode") == "open_loop":
        open_loop = workload.get("open_loop")
        open_loop = open_loop if isinstance(open_loop, dict) else {}
        value, source = open_loop.get("n_scheduled"), "workload.open_loop.n_scheduled"
    else:
        experiment = metrics.get("experiment")
        experiment = experiment if isinstance(experiment, dict) else {}
        value, source = experiment.get("num_measured_requests"), "experiment.num_measured_requests"
    if isinstance(value, bool) or not isinstance(value, int):
        return None, f"{source} is {value!r}"
    return value, source


def _check_row_count(
    window_dir: Path,
    rel_window: str,
    requests_rows: list[dict[str, Any]] | None,
    findings: list[Finding],
) -> int | None:
    """Check (i): the window's rows vs its offered population and its consort
    counters (ADR-0116). Returns the offered population, None when unknown."""
    path = window_dir / _WINDOW_METRICS_NAME
    where = f"{rel_window}/{_WINDOW_METRICS_NAME}"
    if not path.is_file():
        findings.append(
            Finding(
                "WARN",
                "row-count",
                where,
                "no runner summary beside the §1 artifacts: the offered "
                "population and the consort counters are unknown, so a dropped "
                "row cannot be detected here (campaign_session writes "
                f"{_WINDOW_METRICS_NAME} as the window's completeness sentinel)",
            )
        )
        return None
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        findings.append(Finding("FAIL", "schema", where, f"invalid JSON: {exc}"))
        return None
    if not isinstance(metrics, dict):
        findings.append(
            Finding(
                "FAIL",
                "schema",
                where,
                f"root must be an object, got {type(metrics).__name__}",
            )
        )
        return None
    consort = metrics.get("consort")
    if not isinstance(consort, dict):
        findings.append(
            Finding(
                "FAIL",
                "row-count",
                where,
                "no consort block (task #127): the per-query drop counters are "
                "unknown, never assumed zero (ADR-0116)",
            )
        )
    else:
        for key in _CONSORT_COUNTERS:
            value = consort.get(key)
            if isinstance(value, bool) or not isinstance(value, int):
                findings.append(
                    Finding(
                        "FAIL",
                        "row-count",
                        where,
                        f"consort.{key} is {value!r}, not an integer count",
                    )
                )
            elif value != 0:
                findings.append(
                    Finding(
                        "FAIL",
                        "row-count",
                        where,
                        f"consort.{key} = {value}: the runner dropped request(s) "
                        "this window's artifacts do not carry (ADR-0116)",
                    )
                )
    expected, source = expected_row_count(metrics)
    if expected is None:
        findings.append(
            Finding(
                "FAIL",
                "row-count",
                where,
                f"offered population unknown ({source}); the row count cannot "
                "be checked (ADR-0116)",
            )
        )
        return None
    if requests_rows is not None and len(requests_rows) != expected:
        findings.append(
            Finding(
                "FAIL",
                "row-count",
                rel_window,
                f"requests.jsonl has {len(requests_rows)} row(s) but the window "
                f"offered {expected} ({source}): a request without a row is a "
                "changed population, invisible to the requests-vs-evidence "
                "reconciliation because it is absent from both chains "
                "(ADR-0116, Batch 2 W3)",
            )
        )
    return expected


def _check_tpot_rows(
    rows: list[dict[str, Any]], rel: str, findings: list[Finding]
) -> int:
    """Check (j), ADR-0118 (Batch 2 W5): on every completed (ok) row the
    per-request TPOT must agree with the output-token count, and the count
    must be an integer. Returns the number of completed rows with no decode
    phase (num_tokens <= 1): the §9.10 exemption count the analysis applies.
    Rows that are not ok, or carry no validity field, are outside the rule
    (they are never timely). Rows of a reference engine (``reference_engine``
    true: the HF oracle, which never streams, so its ttft_ms is the whole
    call by contract and the runner's formula yields tpot_ms 0.0 on every
    multi-token completion) are exempt from the two TPOT clauses: that value
    is the engine's documented shape, not a captured-timing defect, and the
    oracle is never scored for timeliness (sub-pressure only, charter P3)."""
    no_decode = 0
    bad_count: list[str] = []
    missing_tpot: list[str] = []
    fabricated_tpot: list[str] = []
    fallback_counted: list[str] = []
    for row in rows:
        if row.get("ok") is not True:
            continue
        rid = str(row.get("example_id"))
        num_tokens = row.get("num_tokens")
        if (
            isinstance(num_tokens, bool)
            or not isinstance(num_tokens, int)
            or num_tokens < 0
        ):
            bad_count.append(rid)
            continue
        tpot = row.get("tpot_ms")
        reference = row.get("reference_engine") is True
        if num_tokens <= 1:
            no_decode += 1
            if tpot is not None and not reference:
                fabricated_tpot.append(rid)
        elif not reference and (
            isinstance(tpot, bool)
            or not isinstance(tpot, (int, float))
            or not math.isfinite(tpot)
            or tpot <= 0.0
        ):
            missing_tpot.append(rid)
        if row.get("num_tokens_source") == _NUM_TOKENS_SOURCE_FALLBACK:
            fallback_counted.append(rid)
    if bad_count:
        findings.append(
            Finding(
                "FAIL",
                "tpot",
                rel,
                f"{len(bad_count)} completed (ok) row(s) carry no non-negative "
                f"integer num_tokens (first: {bad_count[:3]}); the decode-phase "
                "rule cannot be applied without the output-token count "
                "(ADR-0118)",
            )
        )
    if missing_tpot:
        findings.append(
            Finding(
                "FAIL",
                "tpot",
                rel,
                f"{len(missing_tpot)} completed (ok) row(s) with two or more "
                "output tokens carry no finite positive tpot_ms (first: "
                f"{missing_tpot[:3]}): a null or zero TPOT beside a decode phase "
                "is a captured-timing defect (ADR-0118)",
            )
        )
    if fabricated_tpot:
        findings.append(
            Finding(
                "FAIL",
                "tpot",
                rel,
                f"{len(fabricated_tpot)} completed (ok) row(s) with at most one "
                "output token carry a tpot_ms value although they have no "
                f"decode phase (first: {fabricated_tpot[:3]}): the runner writes "
                "null there, TPOT is undefined before a second output token "
                "(ADR-0118)",
            )
        )
    if fallback_counted:
        findings.append(
            Finding(
                "WARN",
                "tpot",
                rel,
                f"{len(fallback_counted)} completed (ok) row(s) count output "
                "tokens by the whitespace fallback (num_tokens_source == "
                f"{_NUM_TOKENS_SOURCE_FALLBACK!r}): the engine returned no "
                "usage.completion_tokens, so num_tokens is a word count and the "
                f"decode-phase rule keys on it (first: {fallback_counted[:3]}; "
                "ADR-0118)",
            )
        )
    return no_decode


def _check_pd_transfer(
    window_dir: Path,
    rel_window: str,
    topology: str | None,
    requests_rows: list[dict[str, Any]] | None,
    findings: list[Finding],
) -> bool | None:
    """Check (k), ADR-0134 (S0F-22 Batch 2): the per-window pd transfer proof.
    Returns the re-derived verdict for a pd window (True/False), None when
    there is nothing to judge (not a pd window, topology unknown, or no
    readable record). The summary is parsed here on purpose: a missing or
    unparseable file is already (i)'s finding, and the pd-specific FAIL on a
    MISSING summary is this check's (a pd window must never verify green on
    (i)'s WARN alone)."""
    if topology is None:
        return None  # the cell dirname did not parse: a layout FAIL already stands
    path = window_dir / _WINDOW_METRICS_NAME
    where = f"{rel_window}/{_WINDOW_METRICS_NAME}"
    is_pd = topology == _PD_TOPOLOGY
    if not path.is_file():
        if is_pd:
            findings.append(
                Finding(
                    "FAIL",
                    "pd-transfer",
                    where,
                    "no runner summary beside the §1 artifacts: a pd window carries "
                    f"its KV transfer proof in {_WINDOW_METRICS_NAME}[{_PD_TRANSFER_KEY!r}] "
                    "and cannot be verified without it (ADR-0134)",
                )
            )
        return None
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None  # (i) reported the invalid JSON
    if not isinstance(metrics, dict):
        return None  # (i) reported the non-object root
    record = metrics.get(_PD_TRANSFER_KEY)
    if not is_pd:
        if _PD_TRANSFER_KEY in metrics:
            findings.append(
                Finding(
                    "FAIL",
                    "pd-transfer",
                    where,
                    f"a {topology}-topology window carries a {_PD_TRANSFER_KEY!r} record: "
                    "the cell ran against a decode telemetry endpoint, so its topology "
                    "label or its launch is wrong (ADR-0134)",
                )
            )
        return None
    if not isinstance(record, dict):
        findings.append(
            Finding(
                "FAIL",
                "pd-transfer",
                where,
                f"pd window lacks an object {_PD_TRANSFER_KEY!r} (got "
                f"{type(record).__name__}): the decode's KV pull was never recorded "
                "for this window (ADR-0134)",
            )
        )
        return False
    delta = record.get("delta")
    n_served = record.get("n_served_rows")
    if not isinstance(delta, dict):
        findings.append(
            Finding(
                "FAIL",
                "pd-transfer",
                where,
                f"{_PD_TRANSFER_KEY}.delta is {type(delta).__name__}, not an object of "
                "counter deltas (ADR-0134)",
            )
        )
        return False
    reasons = _pd_transfer_reasons(delta, n_served)
    if reasons:
        findings.append(
            Finding(
                "FAIL",
                "pd-transfer",
                where,
                "the decode's counter deltas do not prove this window's KV pull: "
                + "; ".join(reasons)
                + " (ADR-0134)",
            )
        )
    verified_recorded = record.get("verified")
    if verified_recorded is not (not reasons):
        findings.append(
            Finding(
                "FAIL",
                "pd-transfer",
                where,
                f"recorded verified={verified_recorded!r} disagrees with the re-derived "
                f"verdict {not reasons} (the producer's flag is re-derived here, never "
                "trusted; ADR-0134)",
            )
        )
    if requests_rows is not None:
        served = sum(1 for r in requests_rows if not r.get("error"))
        if n_served != served:
            findings.append(
                Finding(
                    "FAIL",
                    "pd-transfer",
                    where,
                    f"recorded n_served_rows={n_served!r} but requests.jsonl has {served} "
                    "row(s) without error: the proof was computed over a different "
                    "population than the window carries (ADR-0134)",
                )
            )
    # The prefill's expiry counter is recorded, never gated (it moves about
    # 480 s after a ticket and only on a prefill step, so it describes an
    # earlier window's requests): a nonzero delta still names a ticket the
    # decode never pulled before its deadline (review 2026-10-01 LOW-4).
    prefill_delta = record.get("prefill_delta")
    expired = prefill_delta.get("kv_expired_reqs") if isinstance(prefill_delta, dict) else None
    if isinstance(expired, (int, float)) and not isinstance(expired, bool) and expired > 0:
        findings.append(
            Finding(
                "WARN",
                "pd-transfer",
                where,
                f"prefill_delta.kv_expired_reqs = {expired:g}: the prefill released "
                "blocks of ticket(s) the decode never pulled within "
                "VLLM_NIXL_ABORT_REQUEST_TIMEOUT (a request of this or an earlier "
                "window, or a decode that never notified; RC-13 names the side) "
                "(ADR-0134)",
            )
        )
    return not reasons


def _check_finish_reasons(
    rows: list[dict[str, Any]], rel: str, findings: list[Finding]
) -> int | None:
    """Check (l), ADR-0135 (S0F-25): every row without an error must have
    ended with a served finish reason. Returns the number of rows that did
    not, or None when no judged row carries the field (unknown, never 0).
    Error rows are outside the rule: the adapter gives them ``error`` as the
    reason and an open-loop dispatch stub carries none."""
    judged = [row for row in rows if not row.get("error")]
    missing = [row for row in judged if "finish_reason" not in row]
    unserved = [
        (str(row.get("example_id")), row.get("finish_reason"))
        for row in judged
        if "finish_reason" in row and row.get("finish_reason") not in _SERVED_FINISH_REASONS
    ]
    if unserved:
        findings.append(
            Finding(
                "FAIL",
                "finish-reason",
                rel,
                f"{len(unserved)} row(s) without an error did not end with a served "
                f"finish_reason {list(_SERVED_FINISH_REASONS)} (first: "
                + ", ".join(f"{rid} -> {reason!r}" for rid, reason in unserved[:3])
                + "): the request failed and the row reads as served (ADR-0135)",
            )
        )
    if missing:
        findings.append(
            Finding(
                "WARN",
                "finish-reason",
                rel,
                f"{len(missing)} row(s) without an error carry no finish_reason: the "
                "served rule cannot be applied to them (ADR-0135)",
            )
        )
    if judged and len(missing) == len(judged):
        return None
    return len(unserved)


def _check_degenerate_output(
    rows: list[dict[str, Any]], rel: str, findings: list[Finding]
) -> tuple[int | None, float | None]:
    """Check (n), ADR-0151 (S0F-59): served rows whose text is thinking
    scaffolding. Returns (rows carrying the marker, median num_tokens over
    the served rows), both None when no served row exists; the median is
    None when no served row carries an integer num_tokens."""
    served = [row for row in rows if not row.get("error")]
    if not served:
        return None, None
    marked = [
        row for row in served
        if _THINKING_MARKER in str(row.get("generated_answer") or "")
    ]
    tokens = sorted(
        row["num_tokens"] for row in served
        if isinstance(row.get("num_tokens"), int) and not isinstance(row.get("num_tokens"), bool)
    )
    median: float | None = None
    if tokens:
        mid = len(tokens) // 2
        median = float(tokens[mid]) if len(tokens) % 2 else (tokens[mid - 1] + tokens[mid]) / 2.0
    if marked:
        share = len(marked) / len(served)
        findings.append(
            Finding(
                "FAIL" if share >= _DEGENERATE_SHARE_FAIL else "WARN",
                "degenerate-output",
                rel,
                f"{len(marked)} of {len(served)} served row(s) carry the thinking marker "
                f"{_THINKING_MARKER!r} in generated_answer (first: "
                f"{marked[0].get('example_id')!r}; median num_tokens {median}): the chat "
                "template opened a thinking block and the stop sequence ended the request "
                "before any answer, and the row reads as served (ADR-0151, S0F-59)",
            )
        )
    return len(marked), median


def _check_regime(
    window_dir: Path, rel_window: str, engine: str | None, findings: list[Finding]
) -> str | None:
    """Check (m), ADR-0136 (S0F-26): the window's regime label, None when the
    artifact is absent or unreadable. A sampled engine's window labeled
    UNKNOWN_TELEMETRY is a WARN with the recorded refusal reason."""
    path = window_dir / _REGIME_NAME
    if not path.is_file():
        return None
    where = f"{rel_window}/{_REGIME_NAME}"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        findings.append(Finding("FAIL", "schema", where, f"invalid JSON: {exc}"))
        return None
    if not isinstance(doc, dict):
        findings.append(
            Finding("FAIL", "schema", where, f"root must be an object, got {type(doc).__name__}")
        )
        return None
    label = doc.get("label")
    if label == _REGIME_UNKNOWN and engine in _TELEMETRY_ENGINES:
        findings.append(
            Finding(
                "WARN",
                "telemetry",
                where,
                f"window is labeled {_REGIME_UNKNOWN}: {doc.get('refusal_reason')!r}; "
                "its telemetry certified no regime, so no pressure contrast can use "
                "it (the latency rows stay valid) (ADR-0136)",
            )
        )
    return label if isinstance(label, str) else None


def _check_window(
    run_dir: Path,
    window_dir: Path,
    dataset: str,
    findings: list[Finding],
    *,
    topology: str | None = None,
    engine: str | None = None,
) -> dict[str, Any]:
    """Run checks (a)-(c), (i)-(n) + accounting (e) for one window; returns its summary.
    ``topology`` and ``engine`` are the cell's (from its §2 dirname); a None
    topology skips (k), a None engine keeps (m) to recording the label."""
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

    # (i) rows vs the offered population + consort counters (ADR-0116, W3).
    n_expected = _check_row_count(window_dir, rel_window, requests_rows, findings)

    # (j) per-row TPOT vs the output-token count (ADR-0118, W5); the returned
    # exemption count rides the accounting (None when the rows are unreadable).
    n_no_decode = (
        _check_tpot_rows(requests_rows, f"{rel_window}/requests.jsonl", findings)
        if requests_rows is not None
        else None
    )

    # (k) per-window pd transfer proof (ADR-0134, S0F-22 Batch 2); the
    # re-derived verdict rides the accounting (None = not a pd window).
    pd_transfer_verified = _check_pd_transfer(
        window_dir, rel_window, topology, requests_rows, findings
    )

    # (l) served rows ended with stop or length (ADR-0135, S0F-25); None when
    # the rows are unreadable or none carries the field.
    n_unserved_finish = (
        _check_finish_reasons(requests_rows, f"{rel_window}/requests.jsonl", findings)
        if requests_rows is not None
        else None
    )

    # (m) the window's regime label (ADR-0136, S0F-26); None = no artifact.
    regime_label = _check_regime(window_dir, rel_window, engine, findings)

    # (n) served text that is thinking scaffolding (ADR-0151, S0F-59); both
    # None when the rows are unreadable or no row was served.
    n_thinking_marker, median_num_tokens = (
        _check_degenerate_output(requests_rows, f"{rel_window}/requests.jsonl", findings)
        if requests_rows is not None
        else (None, None)
    )

    # (e) §9.10 exclusion accounting — absence is NOT zero: rows lacking any
    # validity field are counted as validity-unknown, never as valid.
    accounting: dict[str, Any] = {
        "window": rel_window,
        "dataset": dataset,
        "n_requests_rows": None,
        "n_expected_rows": n_expected,
        "n_evidence_rows": None,
        "n_error": None,
        "n_ok_false": None,
        "n_empty_generation": None,
        "n_validity_unknown": None,
        "n_valid_known": None,
        "n_no_decode": n_no_decode,
        "pd_transfer_verified": pd_transfer_verified,
        "n_unserved_finish": n_unserved_finish,
        "regime_label": regime_label,
        "n_thinking_marker": n_thinking_marker,
        "median_num_tokens": median_num_tokens,
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
        # The §2 dirname IS the cell identity; its topology drives check (k)
        # and its engine check (m). An unparseable dirname is a layout FAIL
        # and leaves (k) skipped (topology None) and (m) to recording the
        # label (engine None) for that cell's windows.
        topology: str | None = None
        engine: str | None = None
        try:
            cell_spec = org.parse_row_key_dir(cell_dir.name)
            topology, engine = cell_spec.topology, cell_spec.engine
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
                _check_window(
                    run_dir, window_dir, dataset, findings, topology=topology, engine=engine
                )
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
        "n_no_decode",
        "n_unserved_finish",
        "n_thinking_marker",
    ):
        known = [row[key] for row in accounting_rows if row[key] is not None]
        # Absence-is-not-zero: a total over windows with unknown counts is
        # itself unknown; report the partial sum with its coverage.
        totals[key] = {
            "sum_over_known_windows": int(sum(known)) if known else None,
            "n_windows_known": len(known),
        }

    # (m): how many windows carry each regime label (windows with no
    # regime.json are not counted: absence is unknown, never a label).
    regime_labels: dict[str, int] = {}
    for row in accounting_rows:
        if row["regime_label"] is not None:
            regime_labels[row["regime_label"]] = regime_labels.get(row["regime_label"], 0) + 1
    totals["regime_labels"] = dict(sorted(regime_labels.items()))

    n_fail = sum(1 for f in findings if f.severity == "FAIL")
    n_warn = sum(1 for f in findings if f.severity == "WARN")
    return {
        "verifier": "verify_results v2 (task #129)",
        "layout_authority": "docs/RESULTS_LAYOUT.md",
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
        "validity-unknown | valid-known | no-decode |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")

    def _cell(value: Any) -> str:
        return "?" if value is None else str(value)

    for row in report["accounting"]["per_window"]:
        lines.append(
            f"| {row['window']} | {_cell(row['n_requests_rows'])} "
            f"| {_cell(row['n_evidence_rows'])} | {_cell(row['n_error'])} "
            f"| {_cell(row['n_ok_false'])} | {_cell(row['n_empty_generation'])} "
            f"| {_cell(row['n_validity_unknown'])} | {_cell(row['n_valid_known'])} "
            f"| {_cell(row['n_no_decode'])} |"
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
            "Verify a campaign run tree (docs/RESULTS_LAYOUT.md) as the "
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
