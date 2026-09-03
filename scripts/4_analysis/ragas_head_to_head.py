#!/usr/bin/env python3
"""Order:     stage 4 — offline, over a quality-SCORED run's rows (never in the serving loop)
Objective: §8.6(d) RAGAS head-to-head runner — the three-way faithfulness table (LettuceDetect vs AlignScore vs pinned LLM judge)
Cloud:     local

("local" here: the judge endpoint is an owner-pinned OpenAI-compatible
server reached from the operator machine; NOTHING network is touched in
tests or --dry-run.)

Charter binding (PUBLICATION.md §8.6(d), "benchmark the benchmarks"): RAGAS
faithfulness is run OFFLINE on archived generations with a PINNED judge from a
different model family than the model under test, T=0; the deliverable is the
three-way comparison — token-level detector (LettuceDetect) vs claim-level
checker (AlignScore) vs LLM-judge framework — on IDENTICAL outputs. "Judge
model + token budget decided at D11": therefore this runner has NO judge
default and NO token-budget default — ``--judge-model``, ``--judge-revision``
and ``--completion-max-tokens`` are all required, and their absence is a NAMED
refusal citing the D11 gate. The script exists so the comparison is executable
the day the owner pins the judge, not a promise coded later.

Scope vs the full §8.6(d) sentence — RECORDED deviations, never silent ones
(both are carried into the artifact's ``scope`` block):

- **Faithfulness only.** §8.6(d) reads "RAGAS faithfulness/answer-relevance";
  this runner implements the faithfulness head-to-head ONLY. Answer-relevance
  is DEFERRED: it needs the question field (this runner never reads one) and
  its own protocol (RAGAS-style question re-generation + an embedding
  similarity model) — a second D11-costed pass that must not be built before
  the owner has priced it.
- **No stratifier inside.** §8.6(d) says "stratified subsample"; ``--limit``
  is a budget-cap head-slice over the input order, NOT a stratified sampler.
  The stratification duty sits UPSTREAM: the ``--rows`` file passed in must
  already BE the stratified subsample (the artifact records the rows file and
  row counts so the strata provenance stays auditable on the caller's side).

Design decisions (house doctrine):

- **Inputs are a SCORED run's rows** (JSONL): every row must CARRY the two
  instrument columns quality.py produces — ``grounding_score`` (LettuceDetect,
  primary) and ``faithfulness`` (AlignScore claim checker). A file whose rows
  lack the columns is not a scored run and REFUSES (run the quality pass /
  rescore_quality.py first). A present-but-null score is honest absence and is
  carried as None — the row simply cannot enter that instrument's pairs.
- **Judge seam**: production judge = ``OpenAICompatJudge`` — a thin
  ``POST /v1/chat/completions`` client on the same OpenAI-compatible protocol
  every serving adapter in src/inference speaks; endpoint from
  ``CAGE_JUDGE_API_BASE`` (+ optional ``CAGE_JUDGE_API_KEY``), temperature
  pinned 0 per the §8.6(d) T=0 protocol. It is NEVER constructed in tests or
  --dry-run: tests inject a deterministic test-only judge object through the
  explicit ``judge=`` parameter of :func:`main`.
- **--dry-run = the owner's cost preview**: validates ALL inputs exactly as a
  real pass would (D11 pins, thresholds, rows), then prints the EXACT planned
  judge-call count and a token estimate — and calls nothing, writes nothing.
  The token numbers are labeled an ESTIMATE with the method disclosed
  (~4 chars/token heuristic + the flat completion allowance); they are a
  budgeting aid, never recorded as a measurement.
- **Verdict thresholds**: agreement rates compare BINARY verdicts
  (score >= τ). τ_B defaults to the REGISTERED Instrument-B value
  (instrument_b_runner.TAU_REGISTERED, owner-decided 2026-08-05); an explicit
  --tau-b is recorded as an override. No registered constant exists in-repo
  for τ_A (calibrate_instrument_a_tau.py produces it) or for a judge
  threshold, so ``--tau-a`` and ``--tau-judge`` are explicit-only — a default
  would fabricate an owner decision.
- **Artifacts**: ``ragas_head_to_head.json`` + ``.md`` under --out-dir;
  existing artifacts REFUSE without --force (checked BEFORE any judge call,
  so a refusal never burns paid budget); judge identity + revision + prompt
  template hash + τ provenance recorded; per-row scores archived so the gold
  set can adjudicate disagreements (§8.6(d): "adjudicated by the gold set").
- **Judge revision**: the OpenAI-compatible protocol carries no revision
  field, so the pin is RECORDED provenance — the artifact names the
  deployment-verification duty explicitly rather than pretending the wire
  enforced it.

Usage:
  python3 scripts/4_analysis/ragas_head_to_head.py \\
      --rows results/<run>/qa_scored.jsonl --out-dir results/<run>/ragas \\
      --judge-model <D11 pin> --judge-revision <D11 pin> \\
      --completion-max-tokens <D11 budget> \\
      --tau-a <registered> --tau-judge <owner> --limit 200 --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Registered Instrument-B τ (owner decision 2026-08-05, PUBLICATION.md §8.6(c));
# stdlib-only import — the isolated AlignScore env is never touched here.
from src.evaluation.instrument_b_runner import (  # noqa: E402
    TAU_ANCHOR_SCOPE,
    TAU_REGISTERED,
)

__all__ = [
    "ARTIFACT_JSON_NAME",
    "ARTIFACT_MD_NAME",
    "DISAGREEMENT_EXAMPLES_CAP",
    "ENV_JUDGE_API_BASE",
    "ENV_JUDGE_API_KEY",
    "MAX_CONSECUTIVE_JUDGE_FAILURES",
    "MODE_STAMP",
    "PROMPT_TEMPLATE",
    "SCHEMA_VERSION",
    "JudgeReplyError",
    "JudgeSeam",
    "OpenAICompatJudge",
    "RagasRunnerError",
    "build_table",
    "estimate_prompt_tokens",
    "load_rows",
    "main",
    "parse_judge_reply",
    "prompt_template_sha256",
    "render_prompt",
]

ARTIFACT_JSON_NAME = "ragas_head_to_head.json"
ARTIFACT_MD_NAME = "ragas_head_to_head.md"
SCHEMA_VERSION = 1

#: §8.6 is the Layer-4 meta-layer (instrument validation): its outputs are
#: never campaign findings, so the stamp is fixed — no CLI escalation path
#: to CONFIRMATORY exists here (that vocabulary belongs to the §9.11 gate in
#: run_campaign_analysis.py).
MODE_STAMP = "INSTRUMENT-VALIDATION (§8.6d) — NON-CONFIRMATORY"

ENV_JUDGE_API_BASE = "CAGE_JUDGE_API_BASE"
ENV_JUDGE_API_KEY = "CAGE_JUDGE_API_KEY"

#: Budget guard: this many CONSECUTIVE transport failures abort the pass
#: (exit 3, no artifact) instead of burning the remaining judge budget on a
#: dead endpoint. Isolated failures are recorded per-row and the pass
#: continues (the calls already made are sunk cost worth keeping).
MAX_CONSECUTIVE_JUDGE_FAILURES = 5

#: Disagreement example ids listed per pair in the artifact (the TOTAL count
#: is always exact; the id list is capped so the table stays readable — the
#: per_row block archives every score for gold-set adjudication).
DISAGREEMENT_EXAMPLES_CAP = 10

#: The single RAGAS-style judge prompt (ONE call per row: claim decomposition
#: + support counting in strict JSON). ``<<CONTEXT>>``/``<<ANSWER>>`` are
#: literal markers substituted by render_prompt; the sha256 of THIS exact
#: string is the recorded template hash, so any wording change is visible in
#: provenance.
PROMPT_TEMPLATE = """\
You are a strict factual-consistency judge (RAGAS-style faithfulness).
First decompose the ANSWER into its atomic factual claims. Then count how
many of those claims are FULLY supported by the CONTEXT alone — use no
outside knowledge.

Respond with ONLY this JSON object and nothing else:
{"claims_total": <number of atomic claims>, "claims_supported": <number fully supported by the CONTEXT>}

CONTEXT:
<<CONTEXT>>

ANSWER:
<<ANSWER>>
"""

#: Instrument display order + the scored-rows column each one reads.
_INSTRUMENT_COLUMNS = {
    "lettucedetect": "grounding_score",
    "alignscore": "faithfulness",
    "judge": "judge_faithfulness",
}
_PAIRS = (
    ("lettucedetect", "alignscore"),
    ("lettucedetect", "judge"),
    ("alignscore", "judge"),
)

#: Generated-answer field preference: quality.py scores ``sanitized_answer``
#: (its own doctrine: ALL quality scoring uses the sanitized text), so the
#: judge must see the same text the instruments saw when it is present.
_ANSWER_FIELDS = ("sanitized_answer", "generated_answer", "answer")


class RagasRunnerError(ValueError):
    """§8.6(d) runner contract violation (pins, thresholds, row shape)."""


class JudgeReplyError(ValueError):
    """A judge reply that cannot be turned into a faithfulness score."""


class JudgeSeam(Protocol):
    """The injection seam: anything with ``complete(prompt) -> str``.

    Tests inject deterministic test-only objects here; production uses
    :class:`OpenAICompatJudge`. ``describe()`` is optional provenance.
    """

    def complete(self, prompt: str) -> str: ...


# ---------------------------------------------------------------------------
# Judge provider seam (production only — NEVER constructed in tests/--dry-run)
# ---------------------------------------------------------------------------


class OpenAICompatJudge:
    """Thin OpenAI-compatible chat client for the pinned §8.6(d) judge.

    Same wire protocol as every serving adapter in src/inference
    (``POST {api_base}/v1/chat/completions``), non-streaming, temperature
    pinned to 0 per the charter's T=0 protocol. The ``requests`` import is
    deferred into :meth:`complete` so importing this module (as the tests do)
    touches no network stack at all.
    """

    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: Optional[str],
        max_tokens: int,
        timeout_s: float,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self._api_key = api_key
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    @classmethod
    def from_env(cls, model: str, max_tokens: int, timeout_s: float) -> "OpenAICompatJudge":
        api_base = os.environ.get(ENV_JUDGE_API_BASE, "").strip()
        if not api_base:
            raise RagasRunnerError(
                f"judge endpoint not configured: set {ENV_JUDGE_API_BASE} to the "
                "OpenAI-compatible base URL of the pinned judge deployment "
                f"(optional {ENV_JUDGE_API_KEY} for auth) — §8.6(d) runs against "
                "an owner-provisioned endpoint, never a baked-in default"
            )
        return cls(
            model=model,
            api_base=api_base,
            api_key=os.environ.get(ENV_JUDGE_API_KEY) or None,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )

    def complete(self, prompt: str) -> str:
        import requests  # deferred: the only network dependency in this module

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        resp = requests.post(
            f"{self.api_base}/v1/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                # §8.6(d): T=0 — pinned, not configurable.
                "temperature": 0.0,
                "max_tokens": self.max_tokens,
                "stream": False,
            },
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        body = resp.json()
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise JudgeReplyError(
                f"judge response body lacks choices[0].message.content: {exc}"
            ) from exc
        if not isinstance(content, str):
            raise JudgeReplyError(
                f"judge message content is {type(content).__name__}, not str"
            )
        return content

    def describe(self) -> dict[str, Any]:
        # api_base is provenance; the key never is.
        return {
            "kind": "openai-compatible",
            "api_base": self.api_base,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "timeout_s": self.timeout_s,
        }


# ---------------------------------------------------------------------------
# Prompt + reply
# ---------------------------------------------------------------------------


def render_prompt(context: str, answer: str) -> str:
    return PROMPT_TEMPLATE.replace("<<CONTEXT>>", context).replace("<<ANSWER>>", answer)


def prompt_template_sha256() -> str:
    return hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()


def estimate_prompt_tokens(prompt: str) -> int:
    """~4 chars/token English heuristic, ceil — an ESTIMATE for the --dry-run
    budget preview, disclosed as such wherever it is printed/recorded."""
    return math.ceil(len(prompt) / 4)


def _claims_int(obj: dict[str, Any], name: str) -> int:
    value = obj.get(name)
    # bool is refused before the int check (True would count as 1 claim).
    if isinstance(value, bool) or not isinstance(value, int):
        raise JudgeReplyError(
            f"judge reply field {name!r}={value!r} must be a plain integer"
        )
    return value


def parse_judge_reply(text: str) -> tuple[int, int]:
    """Strict parse of one judge reply -> (claims_total, claims_supported).

    Tolerates a single markdown code fence around the JSON (a common chat
    habit) and NOTHING else; any other deviation raises — an unparseable
    verdict is recorded as a named judge error, never coerced into a score.
    """
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError as exc:
        raise JudgeReplyError(f"judge reply is not JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise JudgeReplyError(
            f"judge reply is JSON {type(obj).__name__}, not an object"
        )
    total = _claims_int(obj, "claims_total")
    supported = _claims_int(obj, "claims_supported")
    if total < 0 or supported < 0 or supported > total:
        raise JudgeReplyError(
            f"judge verdict out of range: claims_total={total} "
            f"claims_supported={supported} (need 0 <= supported <= total)"
        )
    return total, supported


# ---------------------------------------------------------------------------
# Scored-rows loading
# ---------------------------------------------------------------------------


@dataclass
class ScoredRow:
    """One input row after validation (judgeable = context+answer non-empty)."""

    row_id: str
    context: str
    answer: str
    answer_field: Optional[str]
    lettucedetect: Optional[float]  # grounding_score
    alignscore: Optional[float]  # faithfulness
    judgeable: bool


def _instrument_score(rec: dict[str, Any], column: str, line_no: int) -> Optional[float]:
    if column not in rec:
        raise RagasRunnerError(
            f"row at line {line_no} lacks the {column!r} column — this file is "
            "not a scored run's rows (§8.6(d) compares the three instruments "
            "on IDENTICAL outputs); produce the LettuceDetect+AlignScore "
            "columns first (quality scoring pass / rescore_quality.py), then "
            "re-run"
        )
    value = rec[column]
    if value is None:
        return None  # honest absence: the row sits out that instrument's pairs
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RagasRunnerError(
            f"row at line {line_no}: {column}={value!r} must be a number or "
            "null — a non-numeric instrument score is corrupt input, not "
            "absence (fail-closed)"
        )
    value = float(value)
    if not math.isfinite(value) or not (0.0 <= value <= 1.0):
        raise RagasRunnerError(
            f"row at line {line_no}: {column}={value!r} must be a finite "
            "score in [0, 1]"
        )
    return value


def _row_context(rec: dict[str, Any]) -> str:
    """Served context, tolerant of stringified lists from older runs (same
    parsing as score_instrument_b._build_items)."""
    contexts = rec.get("used_contexts")
    if contexts is None:
        raw = rec.get("context")
        return str(raw).strip() if raw is not None else ""
    if isinstance(contexts, str):
        try:
            contexts = json.loads(contexts)
        except json.JSONDecodeError:
            contexts = [contexts]
    if not isinstance(contexts, list):
        contexts = [contexts]
    return "\n\n".join(str(c).strip() for c in contexts if c and str(c).strip())


def _row_answer(rec: dict[str, Any]) -> tuple[str, Optional[str]]:
    """(answer_text, field_name) — first preference field with non-empty text."""
    for field in _ANSWER_FIELDS:
        value = rec.get(field)
        if isinstance(value, str) and value.strip():
            return value, field
    return "", None


def _row_id(rec: dict[str, Any], line_no: int) -> str:
    if isinstance(rec.get("id"), str) and rec["id"].strip():
        return rec["id"]
    example_id = rec.get("example_id")
    if example_id is None or not str(example_id).strip():
        raise RagasRunnerError(
            f"row at line {line_no} has neither 'id' nor 'example_id' — an "
            "unidentifiable row cannot be adjudicated against the gold set; "
            "fix the rows export"
        )
    return f"{example_id}::{str(rec.get('repeat_index') or '0')}"


def load_rows(rows_path: Path) -> list[ScoredRow]:
    """Parse + validate the scored-rows JSONL (fail-closed; duplicate ids and
    missing instrument columns refuse; empty context/answer rows load as
    judgeable=False and are counted, never silently dropped)."""
    if not rows_path.is_file():
        raise RagasRunnerError(
            f"--rows {rows_path} is not a file — pass the scored run's rows "
            "JSONL (one JSON object per line carrying grounding_score + "
            "faithfulness + the served context and generated answer)"
        )
    rows: list[ScoredRow] = []
    seen: set[str] = set()
    for line_no, line in enumerate(
        rows_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RagasRunnerError(
                f"{rows_path}:{line_no} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(rec, dict):
            raise RagasRunnerError(
                f"{rows_path}:{line_no} is a JSON {type(rec).__name__}, not an "
                "object row"
            )
        row_id = _row_id(rec, line_no)
        if row_id in seen:
            raise RagasRunnerError(
                f"duplicate row id {row_id!r} at line {line_no} — two rows "
                "claiming one id would silently shadow each other in the "
                "three-way join; refusing"
            )
        seen.add(row_id)
        context = _row_context(rec)
        answer, answer_field = _row_answer(rec)
        rows.append(
            ScoredRow(
                row_id=row_id,
                context=context,
                answer=answer,
                answer_field=answer_field,
                lettucedetect=_instrument_score(rec, "grounding_score", line_no),
                alignscore=_instrument_score(rec, "faithfulness", line_no),
                judgeable=bool(context.strip() and answer.strip()),
            )
        )
    if not rows:
        raise RagasRunnerError(f"{rows_path} contains no rows")
    return rows


# ---------------------------------------------------------------------------
# Three-way table math
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def build_table(
    per_row: list[dict[str, Any]],
    taus: dict[str, float],
) -> dict[str, Any]:
    """The §8.6(d) three-way table from per-row scores.

    ``per_row``: dicts carrying the three ``_INSTRUMENT_COLUMNS`` columns
    (floats or None) plus ``id``. ``taus``: verdict threshold per instrument
    name. Pairs are computed over rows where BOTH members are present; empty
    pools render None (absence), never 0.0.
    """
    instruments: dict[str, Any] = {}
    for name, column in _INSTRUMENT_COLUMNS.items():
        present = [r[column] for r in per_row if r[column] is not None]
        verdicts = [v >= taus[name] for v in present]
        instruments[name] = {
            "column": column,
            "tau": taus[name],
            "n_present": len(present),
            "mean_score": _mean(present),
            "verdict_positive_rate": (
                sum(verdicts) / len(verdicts) if verdicts else None
            ),
        }

    pairs: dict[str, Any] = {}
    for first, second in _PAIRS:
        col_f, col_s = _INSTRUMENT_COLUMNS[first], _INSTRUMENT_COLUMNS[second]
        both = [r for r in per_row if r[col_f] is not None and r[col_s] is not None]
        deltas = [r[col_f] - r[col_s] for r in both]
        disagreements = [
            r["id"]
            for r in both
            if (r[col_f] >= taus[first]) != (r[col_s] >= taus[second])
        ]
        n_agree = len(both) - len(disagreements)
        pairs[f"{first}__vs__{second}"] = {
            # delta sign convention: first minus second.
            "n_pairs": len(both),
            "mean_delta": _mean(deltas),
            "mean_abs_delta": _mean([abs(d) for d in deltas]),
            "n_agree": n_agree,
            "n_disagree": len(disagreements),
            "agreement_rate": n_agree / len(both) if both else None,
            "disagreement_example_ids": disagreements[:DISAGREEMENT_EXAMPLES_CAP],
            "disagreement_examples_cap": DISAGREEMENT_EXAMPLES_CAP,
        }

    all_three = [
        r
        for r in per_row
        if all(r[c] is not None for c in _INSTRUMENT_COLUMNS.values())
    ]
    unanimous = [
        r
        for r in all_three
        if len(
            {
                r[column] >= taus[name]
                for name, column in _INSTRUMENT_COLUMNS.items()
            }
        )
        == 1
    ]
    return {
        "instruments": instruments,
        "pairs": pairs,
        "three_way": {
            "n_all_three_present": len(all_three),
            "n_all_three_agree": len(unanimous),
            "unanimous_rate": (
                len(unanimous) / len(all_three) if all_three else None
            ),
        },
    }


# ---------------------------------------------------------------------------
# Artifact rendering
# ---------------------------------------------------------------------------


def _atomic_write_text(path: Path, text: str) -> None:
    # G14 discipline (as in run_campaign_analysis): a crash mid-write must
    # never leave a torn artifact that a later look could half-read.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def _git_provenance() -> dict[str, Any]:
    """Best-effort repo SHA (same doctrine as score_instrument_b.py: a tarball
    checkout records an explicit null, never a fabricated SHA)."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=REPO_ROOT,
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        )
        return {"code_git_sha": sha, "git_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"code_git_sha": None, "git_dirty": None,
                "git_note": "git unavailable at scoring time"}


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return format(value, ".6g")
    return str(value)


def _render_markdown(document: dict[str, Any]) -> str:
    judge = document["judge"]
    table = document
    lines = [
        f"# §8.6(d) RAGAS head-to-head — {document['mode_stamp']}",
        "",
        "Three-way faithfulness comparison on IDENTICAL archived outputs: "
        "token-level detector (LettuceDetect) vs claim-level checker "
        "(AlignScore) vs LLM-judge framework (RAGAS-style). Disagreements "
        "are adjudicated by the gold set (PUBLICATION.md §8.6(d)).",
        "",
        "Scope (recorded §8.6(d) deviations): FAITHFULNESS only — the "
        "answer-relevance leg is deferred (needs the question field + its "
        "own judge/embedding protocol, a separate D11-costed pass); "
        "stratification happens UPSTREAM (`--limit` is a budget head-slice, "
        "not a sampler — the rows file must be the pre-stratified "
        "subsample).",
        "",
        f"- judge: `{judge['model']}` @ revision `{judge['revision']}` "
        f"(T=0, completion cap {judge['completion_max_tokens']} tokens)",
        f"- prompt template sha256: `{judge['prompt_template_sha256']}`",
        f"- rows: {document['inputs']['n_selected']} judged of "
        f"{document['inputs']['n_rows_total']} loaded "
        f"({document['inputs']['n_unjudgeable']} unjudgeable, "
        f"limit={document['inputs']['limit']})",
        "",
        "## Instruments",
        "",
        "| instrument | column | τ (source) | n | mean score | verdict+ rate |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    thresholds = document["thresholds"]
    for name, info in table["instruments"].items():
        tau_info = thresholds[name]
        lines.append(
            f"| {name} | {info['column']} | "
            f"{_fmt(info['tau'])} ({tau_info['source']}) | {info['n_present']} "
            f"| {_fmt(info['mean_score'])} | "
            f"{_fmt(info['verdict_positive_rate'])} |"
        )
    lines += [
        "",
        "## Pairwise (delta = first − second)",
        "",
        "| pair | n | agreement | disagreements | mean Δ | mean \\|Δ\\| |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for pair_name, info in table["pairs"].items():
        lines.append(
            f"| {pair_name.replace('__vs__', ' vs ')} | {info['n_pairs']} | "
            f"{_fmt(info['agreement_rate'])} | {info['n_disagree']} | "
            f"{_fmt(info['mean_delta'])} | {_fmt(info['mean_abs_delta'])} |"
        )
    for pair_name, info in table["pairs"].items():
        if info["disagreement_example_ids"]:
            lines.append("")
            lines.append(
                f"- {pair_name.replace('__vs__', ' vs ')} disagreement "
                f"examples (first {info['disagreement_examples_cap']} of "
                f"{info['n_disagree']}): "
                + ", ".join(f"`{i}`" for i in info["disagreement_example_ids"])
            )
    three = table["three_way"]
    lines += [
        "",
        "## Three-way",
        "",
        f"- all three present: {three['n_all_three_present']}",
        f"- unanimous verdicts: {three['n_all_three_agree']} "
        f"(rate {_fmt(three['unanimous_rate'])})",
        "",
        f"- judge calls: {document['judge_calls']['n_calls']} "
        f"({document['judge_calls']['n_scored']} scored, "
        f"{document['judge_calls']['n_no_claims']} no-claims, "
        f"{document['judge_calls']['n_errors']} errors)",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _positive_int(value: str) -> int:
    iv = int(value)
    if iv < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value!r}")
    return iv


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="§8.6(d) RAGAS head-to-head: three-way faithfulness table "
                    "(LettuceDetect vs AlignScore vs pinned LLM judge) on a "
                    "scored run's rows."
    )
    p.add_argument("--rows", required=True,
                   help="Scored run's rows (JSONL): grounding_score + "
                        "faithfulness columns plus the served context "
                        "(used_contexts/context) and answer "
                        "(sanitized_answer/generated_answer/answer). Must "
                        "already BE the §8.6(d) stratified subsample — this "
                        "runner ships no stratifier.")
    p.add_argument("--out-dir", required=True,
                   help=f"Artifact directory ({ARTIFACT_JSON_NAME} + "
                        f"{ARTIFACT_MD_NAME}); existing artifacts refuse "
                        "without --force.")
    # D11-gated pins: deliberately NOT argparse-required so absence produces
    # the NAMED charter refusal below instead of a generic argparse error.
    p.add_argument("--judge-model", default=None,
                   help="REQUIRED, no default: the D11-pinned judge model "
                        "(different family than the model under test).")
    p.add_argument("--judge-revision", default=None,
                   help="REQUIRED, no default: the D11-pinned judge revision "
                        "(recorded provenance; the OpenAI-compatible wire has "
                        "no revision field).")
    p.add_argument("--completion-max-tokens", type=_positive_int, default=None,
                   help="REQUIRED, no default: per-call completion cap = the "
                        "D11-decided token budget; also the completion term "
                        "of the --dry-run estimate.")
    p.add_argument("--tau-a", type=float, default=None,
                   help="REQUIRED: Instrument-A (LettuceDetect) verdict "
                        "threshold — the registered value comes from "
                        "calibrate_instrument_a_tau.py / PRE_REGISTRATION.md; "
                        "no in-repo constant exists to default to.")
    p.add_argument("--tau-b", type=float, default=None,
                   help="Instrument-B (AlignScore) verdict threshold. DEFAULT "
                        f"= the REGISTERED tau {TAU_REGISTERED} (anchor scope "
                        f"'{TAU_ANCHOR_SCOPE}', owner-decided 2026-08-05); an "
                        "explicit value is recorded as an override.")
    p.add_argument("--tau-judge", type=float, default=None,
                   help="REQUIRED: judge verdict threshold on RAGAS-style "
                        "faithfulness (supported/total). No registered value "
                        "exists — an owner choice, recorded as source=cli.")
    p.add_argument("--limit", type=_positive_int, default=None,
                   help="Budget-bounded slice: judge only the first N "
                        "judgeable rows in input order (default: all). NOT a "
                        "stratified sampler — pre-stratify the --rows file "
                        "upstream.")
    p.add_argument("--judge-timeout-s", type=float, default=120.0,
                   help="Per-call HTTP timeout for the real judge "
                        "(default 120).")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate ALL inputs, print the exact planned "
                        "judge-call count + token ESTIMATE, call nothing, "
                        "write nothing (the owner's D11 cost preview).")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing artifacts in --out-dir (default: "
                        "refuse).")
    return p.parse_args(argv)


def _check_pins(args: argparse.Namespace) -> None:
    """The D11 gate: judge identity + token budget are OWNER decisions
    (PUBLICATION.md §8.6(d) 'Judge model + token budget decided at D11');
    absence refuses with the charter citation, never a silent default."""
    missing = [
        flag
        for flag, value in (
            ("--judge-model", args.judge_model),
            ("--judge-revision", args.judge_revision),
            ("--completion-max-tokens", args.completion_max_tokens),
        )
        if value is None or (isinstance(value, str) and not value.strip())
    ]
    if missing:
        raise RagasRunnerError(
            f"{', '.join(missing)} missing — the judge model, its revision "
            "and the token budget are D11 OWNER decisions (PUBLICATION.md "
            "§8.6(d): 'Judge model + token budget decided at D11'); pass all "
            "three explicitly once the owner pins them. This runner ships no "
            "defaults for them by design."
        )


def _check_tau(name: str, value: Optional[float], why: str) -> float:
    if value is None:
        raise RagasRunnerError(f"{name} missing — {why}")
    if not math.isfinite(value) or not (0.0 <= value <= 1.0):
        raise RagasRunnerError(
            f"{name}={value!r} must be a finite threshold in [0, 1]"
        )
    return float(value)


def _resolve_thresholds(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    tau_a = _check_tau(
        "--tau-a", args.tau_a,
        "the Instrument-A verdict threshold is the registered τ_A from "
        "calibrate_instrument_a_tau.py / PRE_REGISTRATION.md; no in-repo "
        "constant exists, so it must be passed explicitly (a default would "
        "fabricate an owner decision)",
    )
    # Same None-sentinel resolution as score_instrument_b.py: registered by
    # default, an explicit value is recorded as CHOSEN even when equal.
    if args.tau_b is None:
        tau_b, tau_b_source = TAU_REGISTERED, "registered"
    else:
        # why-text unused on the explicit path (value is non-None by here).
        tau_b = _check_tau("--tau-b", args.tau_b, "explicit --tau-b given")
        tau_b_source = "override"
    tau_judge = _check_tau(
        "--tau-judge", args.tau_judge,
        "no registered judge threshold exists — the verdict cut on RAGAS-"
        "style faithfulness is an owner choice; pass it explicitly (recorded "
        "as source=cli)",
    )
    return {
        "lettucedetect": {"value": tau_a, "source": "cli"},
        "alignscore": {
            "value": tau_b,
            "source": tau_b_source,
            "tau_registered": TAU_REGISTERED,
            "tau_anchor_scope": TAU_ANCHOR_SCOPE,
        },
        "judge": {"value": tau_judge, "source": "cli"},
    }


def _judge_selected_rows(
    selected: list[ScoredRow], judge: JudgeSeam
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the judge over the slice -> (per_row records, call accounting).

    Isolated failures become named per-row notes;
    MAX_CONSECUTIVE_JUDGE_FAILURES consecutive transport failures raise (the
    budget guard — a dead endpoint must not eat the remaining budget)."""
    per_row: list[dict[str, Any]] = []
    n_calls = n_scored = n_no_claims = n_errors = 0
    consecutive_failures = 0
    for row in selected:
        prompt = render_prompt(row.context, row.answer)
        record: dict[str, Any] = {
            "id": row.row_id,
            "answer_field": row.answer_field,
            "grounding_score": row.lettucedetect,
            "faithfulness": row.alignscore,
            "judge_faithfulness": None,
            "judge_claims_total": None,
            "judge_claims_supported": None,
            "judge_note": None,
        }
        n_calls += 1
        try:
            reply = judge.complete(prompt)
        except Exception as exc:  # transport seam: any failure is a named row
            consecutive_failures += 1
            n_errors += 1
            record["judge_note"] = f"judge-call-failed: {exc}"
            per_row.append(record)
            if consecutive_failures >= MAX_CONSECUTIVE_JUDGE_FAILURES:
                raise RagasRunnerError(
                    f"{consecutive_failures} consecutive judge-call failures "
                    f"(last: {exc}) — aborting so a dead endpoint cannot eat "
                    "the remaining token budget; no artifact is written "
                    f"(the {n_calls} calls already attempted are sunk cost)"
                ) from exc
            continue
        consecutive_failures = 0
        try:
            total, supported = parse_judge_reply(reply)
        except JudgeReplyError as exc:
            n_errors += 1
            record["judge_note"] = f"judge-reply-unparseable: {exc}"
            per_row.append(record)
            continue
        record["judge_claims_total"] = total
        record["judge_claims_supported"] = supported
        if total == 0:
            # An answer the judge found claim-free is honest absence (there
            # is nothing to be faithful OR unfaithful about), never a 1.0.
            n_no_claims += 1
            record["judge_note"] = "judge-no-claims"
        else:
            record["judge_faithfulness"] = supported / total
            n_scored += 1
        per_row.append(record)
    accounting = {
        "n_calls": n_calls,
        "n_scored": n_scored,
        "n_no_claims": n_no_claims,
        "n_errors": n_errors,
    }
    return per_row, accounting


def run(args: argparse.Namespace, judge: Optional[JudgeSeam]) -> int:
    _check_pins(args)
    thresholds = _resolve_thresholds(args)

    rows = load_rows(Path(args.rows))
    judgeable = [r for r in rows if r.judgeable]
    n_unjudgeable = len(rows) - len(judgeable)
    if not judgeable:
        raise RagasRunnerError(
            f"{args.rows}: no judgeable rows (every row has an empty served "
            "context or empty answer) — nothing for the three-way comparison"
        )
    selected = judgeable[: args.limit] if args.limit is not None else judgeable

    # Token ESTIMATE (both modes record it; --dry-run exists to print it):
    # ceil(chars/4) per prompt + the flat D11 completion cap per call. A
    # budgeting heuristic, disclosed — never a measurement.
    prompt_tokens_est = sum(
        estimate_prompt_tokens(render_prompt(r.context, r.answer)) for r in selected
    )
    completion_tokens_est = args.completion_max_tokens * len(selected)
    estimate = {
        "planned_judge_calls": len(selected),
        "prompt_tokens_est": prompt_tokens_est,
        "completion_tokens_est": completion_tokens_est,
        "total_tokens_est": prompt_tokens_est + completion_tokens_est,
        "method": "ESTIMATE: ceil(prompt_chars/4) + completion cap per call "
                  "(~4 chars/token heuristic; budgeting aid, not a "
                  "measurement)",
    }

    if args.dry_run:
        # The owner's cost preview: validated everything above, calls nothing,
        # writes nothing (the injected judge — if any — is never touched).
        print(
            f"RAGAS_DRY_RUN  planned_judge_calls={estimate['planned_judge_calls']}  "
            f"rows_total={len(rows)}  judgeable={len(judgeable)}  "
            f"unjudgeable={n_unjudgeable}  limit={args.limit}"
        )
        print(
            f"  prompt_tokens_est={prompt_tokens_est}  "
            f"completion_tokens_est={completion_tokens_est}  "
            f"total_tokens_est={estimate['total_tokens_est']}"
        )
        print(f"  {estimate['method']}")
        print(
            f"  judge={args.judge_model}@{args.judge_revision}  "
            f"template_sha256={prompt_template_sha256()}"
        )
        return 0

    out_dir = Path(args.out_dir)
    json_path = out_dir / ARTIFACT_JSON_NAME
    md_path = out_dir / ARTIFACT_MD_NAME
    existing = [p for p in (json_path, md_path) if p.exists()]
    if existing and not args.force:
        # Checked BEFORE any judge call: an overwrite refusal must never
        # follow a paid pass.
        raise RagasRunnerError(
            f"{', '.join(str(p) for p in existing)} already exist(s) — "
            "refusing to overwrite a §8.6(d) artifact; re-run with --force "
            "to replace it deliberately"
        )

    if judge is None:
        judge = OpenAICompatJudge.from_env(
            model=args.judge_model,
            max_tokens=args.completion_max_tokens,
            timeout_s=args.judge_timeout_s,
        )

    per_row, accounting = _judge_selected_rows(selected, judge)
    if accounting["n_scored"] == 0:
        raise RagasRunnerError(
            f"judge produced 0 usable faithfulness scores over "
            f"{accounting['n_calls']} calls ({accounting['n_errors']} errors, "
            f"{accounting['n_no_claims']} no-claims) — nothing to compare; "
            "no artifact is written"
        )

    taus = {name: info["value"] for name, info in thresholds.items()}
    table = build_table(per_row, taus)

    describe = getattr(judge, "describe", None)
    provider = (
        describe() if callable(describe)
        else {"kind": "injected-judge (no describe() provided)"}
    )
    answer_field_counts: dict[str, int] = {}
    for r in selected:
        key = r.answer_field or "none"
        answer_field_counts[key] = answer_field_counts.get(key, 0) + 1

    document = {
        "schema_version": SCHEMA_VERSION,
        "mode_stamp": MODE_STAMP,
        "charter": (
            "PUBLICATION.md §8.6(d) RAGAS head-to-head — three-way "
            "comparison on identical outputs, disagreements adjudicated by "
            "the gold set; judge model + token budget = D11 owner pins"
        ),
        # Recorded §8.6(d) deviations — deviations are DECLARED, never silent
        # (module docstring carries the full rationale).
        "scope": {
            "implemented": "faithfulness three-way only",
            "deferred_answer_relevance": (
                "§8.6(d) names 'faithfulness/answer-relevance'; the answer-"
                "relevance leg is NOT in this artifact — it needs the "
                "question field (never read here) and its own judge/embedding "
                "protocol, a separate D11-costed pass"
            ),
            "stratification": (
                "§8.6(d) says 'stratified subsample'; --limit is a budget-cap "
                "head-slice, not a sampler — the --rows file itself must be "
                "the pre-stratified subsample, built upstream"
            ),
        },
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "judge": {
            "model": args.judge_model,
            "revision": args.judge_revision,
            "revision_enforcement": (
                "recorded pin — the OpenAI-compatible protocol carries no "
                "revision field; the serving deployment must be verified "
                "against this pin at D11 execution"
            ),
            "temperature": 0.0,
            "completion_max_tokens": args.completion_max_tokens,
            "prompt_template_sha256": prompt_template_sha256(),
            "prompt_template": PROMPT_TEMPLATE,
            "provider": provider,
        },
        "thresholds": thresholds,
        "inputs": {
            "rows_file": str(args.rows),
            "n_rows_total": len(rows),
            "n_judgeable": len(judgeable),
            "n_unjudgeable": n_unjudgeable,
            "limit": args.limit,
            "n_selected": len(selected),
            "n_missing_lettucedetect": sum(
                1 for r in selected if r.lettucedetect is None
            ),
            "n_missing_alignscore": sum(
                1 for r in selected if r.alignscore is None
            ),
            "answer_field_counts": answer_field_counts,
        },
        "token_estimate": estimate,
        "judge_calls": accounting,
        **table,
        "per_row": per_row,
        **_git_provenance(),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(json_path, json.dumps(document, indent=2) + "\n")
    _atomic_write_text(md_path, _render_markdown(document))
    print(
        f"RAGAS_HEAD_TO_HEAD_DONE  rows={len(selected)}  "
        f"judge_calls={accounting['n_calls']}  scored={accounting['n_scored']}  "
        f"errors={accounting['n_errors']}  "
        f"judge={args.judge_model}@{args.judge_revision}"
    )
    print(f"  json : {json_path}")
    print(f"  md   : {md_path}")
    return 0


def main(
    argv: Optional[list[str]] = None, *, judge: Optional[JudgeSeam] = None
) -> int:
    """CLI entry. ``judge`` is the explicit test seam: tests pass a
    deterministic test-only object; production leaves it None and the runner
    builds :class:`OpenAICompatJudge` from the environment (never in
    --dry-run, which touches no judge at all)."""
    args = parse_args(argv)
    try:
        return run(args, judge)
    except RagasRunnerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        # Refusals = 2; the consecutive-failure budget-guard abort = 3.
        return 3 if "consecutive judge-call failures" in str(exc) else 2


if __name__ == "__main__":
    raise SystemExit(main())
