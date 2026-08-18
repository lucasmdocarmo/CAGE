"""§8.5 veridicality-predicate producer + the decoupled-scoring join chain.

Task #119 (Topic-6 F1-F4, MyDocs/registration/CODE_ASSERTION_2026-08.md): the
campaign's headline Y rides the per-query §8.5 veridicality predicate, and
until this module existed the predicate had NO producer — goodput
``evaluate_window``, ``families.PREDICATE_METRIC`` and run_campaign_analysis
were consumers of a column nothing wrote (F1), and the decoupled-scoring
sidecar rows (scoring/<id>/cells/.../qa_scores.jsonl) never rejoined the
serving evidence rows (F2).

Charter §8.5, THE definition site (PUBLICATION.md — pre-registered per
dataset; the branch DEFINITIONS below are registration text at the #112
freeze):

- **span-QA sets (SQuAD v2, HotpotQA, MuSiQue) — correctness-based**:
  normalized F1/EM + correct abstention on unanswerables. Executed here as the
  ABSTENTION-AWARE official normalized exact-match column
  (``quality.py`` fix #4: on unanswerable items EM is 1.0 iff the model
  correctly abstained; on answerable items EM is the official normalized
  match and an abstention scores 0) binarized at 1.0 — one column already
  carrying BOTH registered clauses. This is also the §9.6 power-sim surrogate
  (run_power_sim G15: surrogate = ``exact_match``), so the registered power
  analysis and the executed predicate agree. Continuous F1 stays the
  ADR-0087 exploratory companion, never the predicate.
- **Qasper — groundedness-based**: Instrument A (LettuceDetect,
  ``grounding_score``) at calibrated τ. The τ VALUE is an OPEN owner decision
  (task #120 / Topic-6 F7: charter names "Instrument A at calibrated τ" but
  the only registered τ, 0.817024, is Instrument B's) — so τ is an EXPLICIT
  config field with NO default: qasper rows without a configured τ REFUSE,
  never silently borrow a threshold.
- **Contradiction and neutral reported separately** (§8.5 claim-pipeline
  protocol: misread evidence vs invented claim are different bugs): the
  3-class NLI columns (``faithfulness_contradiction`` /
  ``faithfulness_neutral``) ride the output rows and the summary as SEPARATE
  reported measurements; they never fold into the predicate.

None-propagation (charter absence-is-not-zero): a query whose serving row is
not ok (#127 integrity stamp: error or empty generation) or whose required
verdict input is None yields ``predicate = None`` — counted, never
fabricated. The producer REFUSES the whole window loudly when the null
fraction breaches the configured sanity bound or when a required verdict
column is absent from the sidecar entirely.

Join chain (F2-F4, #127 keys): scoring sidecar rows join serving evidence
rows on the #127 identity triple ``(example_id, repeat_index, record_index)``
within one ``cells/<row_key>/window_<k>`` window (path-mirrored between the
raw tree and the scoring pass). Duplicates on either side refuse (the #127
duplicate policy — ``rescore_quality`` refuses them by default, so a
duplicate here means the chain upstream was run permissively); unmatched rows
refuse with counts in BOTH directions (serving rows without scores, score
rows without serving parents). Not-ok evidence rows are NULLED on this
consumer side (H2: the offline scorer scores a serving error's empty answer
as a hard zero — those verdicts are garbage and must never reach Y) and the
nulling is counted.

Domain logic only: stdlib, no pandas, no I/O — the artifact producer CLI is
``scripts/4_analysis/build_predicate_table.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

__all__ = [
    "GROUNDEDNESS_DATASETS",
    "JOIN_KEY_FIELDS",
    "PREDICATE_DATASETS",
    "PredicateConfig",
    "PredicateError",
    "SPAN_QA_DATASETS",
    "SPAN_QA_RULE",
    "QASPER_RULE",
    "compute_window_predicate",
    "join_window_rows",
    "required_verdict_column",
    "summarize_predicate_rows",
]

#: §8.5 registered branch membership (charter: "pre-registered per dataset").
SPAN_QA_DATASETS: frozenset[str] = frozenset({"squad_v2", "hotpotqa", "musique"})
GROUNDEDNESS_DATASETS: frozenset[str] = frozenset({"qasper"})
#: The predicate universe — SCBench/RULER/ShareGPT are instruments or load
#: donors and NEVER feed the D8 Y predicate (src/data/scbench.py doctrine).
PREDICATE_DATASETS: frozenset[str] = SPAN_QA_DATASETS | GROUNDEDNESS_DATASETS

#: Registered rule labels stamped on every produced row (audit trail).
SPAN_QA_RULE = "span_qa_em_abstention_aware"
QASPER_RULE = "qasper_grounding_at_tau"

#: The #127 identity triple (run_experiment.evidence_integrity_fields +
#: rescore_quality._evidence_row_key): example_id, repeat_index (string,
#: absent -> "0"), record_index (open-loop replay disambiguator; None on
#: closed-loop rows).
JOIN_KEY_FIELDS: tuple[str, ...] = ("example_id", "repeat_index", "record_index")

#: §8.5 3-class NLI columns reported SEPARATELY (never inside the predicate).
_SEPARATE_REPORT_COLUMNS: tuple[str, ...] = (
    "faithfulness_contradiction",
    "faithfulness_neutral",
)

_MAX_LISTED_KEYS = 5


class PredicateError(RuntimeError):
    """Any §8.5 predicate/join contract violation (fail loud, message first)."""


@dataclass(frozen=True)
class PredicateConfig:
    """Explicit predicate configuration — no silent defaults on owned knobs.

    ``max_null_fraction`` is the per-window sanity bound on the fraction of
    ``predicate = None`` rows: breaching it refuses the whole build (a run
    whose predicate is mostly unscoreable must not quietly feed Y). It has NO
    default — the caller states the bound.

    ``qasper_tau`` is the Instrument-A groundedness threshold for the Qasper
    branch. Task #120 owns the value (Topic-6 F7: no Instrument-A τ is
    registered anywhere), so there is NO default: ``None`` is legal only for
    trees without qasper rows, and a qasper window under ``None`` refuses.
    """

    max_null_fraction: float
    qasper_tau: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.max_null_fraction, bool) or not isinstance(
            self.max_null_fraction, (int, float)
        ):
            raise PredicateError(
                f"max_null_fraction={self.max_null_fraction!r} must be a number"
            )
        if not 0.0 <= float(self.max_null_fraction) <= 1.0:
            raise PredicateError(
                f"max_null_fraction={self.max_null_fraction!r} must be within "
                "[0, 1] (a fraction of rows)"
            )
        if self.qasper_tau is not None:
            if isinstance(self.qasper_tau, bool) or not isinstance(
                self.qasper_tau, (int, float)
            ):
                raise PredicateError(
                    f"qasper_tau={self.qasper_tau!r} must be a number or None"
                )
            if not 0.0 <= float(self.qasper_tau) <= 1.0:
                raise PredicateError(
                    f"qasper_tau={self.qasper_tau!r} must be within [0, 1] "
                    "(an Instrument-A score threshold)"
                )


def required_verdict_column(dataset: str) -> str:
    """The sidecar verdict column the §8.5 branch for ``dataset`` consumes."""
    if dataset in SPAN_QA_DATASETS:
        return "exact_match"
    if dataset in GROUNDEDNESS_DATASETS:
        return "grounding_score"
    raise PredicateError(
        f"dataset {dataset!r} is outside the §8.5 predicate universe "
        f"{sorted(PREDICATE_DATASETS)} — SCBench/RULER/ShareGPT are "
        "instruments/load donors and never feed Y"
    )


def _row_key(row: Mapping[str, Any], *, source: str, index: int) -> tuple[Any, str, Any]:
    """The #127 identity triple of one row (mirrors rescore_quality)."""
    example_id = row.get("example_id")
    if not isinstance(example_id, str) or not example_id:
        raise PredicateError(
            f"{source} row {index}: missing/empty example_id — an unjoinable "
            "row cannot enter the predicate chain (#127 identity contract)"
        )
    return (example_id, str(row.get("repeat_index") or "0"), row.get("record_index"))


def _keyed(
    rows: Sequence[Mapping[str, Any]], *, source: str, window: str
) -> dict[tuple[Any, str, Any], Mapping[str, Any]]:
    keyed: dict[tuple[Any, str, Any], Mapping[str, Any]] = {}
    dups: list[tuple[Any, str, Any]] = []
    for i, row in enumerate(rows):
        key = _row_key(row, source=f"{window}/{source}", index=i)
        if key in keyed:
            dups.append(key)
            continue
        keyed[key] = row
    if dups:
        shown = ", ".join(repr(k) for k in dups[:_MAX_LISTED_KEYS])
        more = "" if len(dups) <= _MAX_LISTED_KEYS else f" (+{len(dups) - _MAX_LISTED_KEYS} more)"
        raise PredicateError(
            f"{window}/{source}: {len(dups)} duplicate (example_id, "
            f"repeat_index, record_index) key(s): {shown}{more} — the #127 "
            "chain refuses indistinguishable rows (rescore_quality refuses "
            "them by default; a permissive upstream pass cannot feed the "
            "predicate)"
        )
    return keyed


def join_window_rows(
    evidence_rows: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
    *,
    window: str,
) -> list[dict[str, Any]]:
    """Deterministic fail-loud join of one window's sidecar onto its evidence.

    Join key = the #127 identity triple. Refusals: duplicate keys on either
    side; unmatched rows in EITHER direction (both counts named); an evidence
    row without the #127 ``ok`` integrity stamp (pre-#127 trees cannot feed
    the predicate — the stamp is what lets the consumer null serving
    failures instead of scoring them, H2).

    Output rows carry the join key, the evidence integrity fields
    (``ok``/``error``/``empty_generation``) and every score-row field, in
    evidence-row order (deterministic).
    """
    ev_keyed = _keyed(evidence_rows, source="qa_evidence.jsonl", window=window)
    sc_keyed = _keyed(score_rows, source="qa_scores.jsonl", window=window)

    ev_only = [k for k in ev_keyed if k not in sc_keyed]
    sc_only = [k for k in sc_keyed if k not in ev_keyed]
    if ev_only or sc_only:
        def _shown(keys: list[tuple[Any, str, Any]]) -> str:
            head = ", ".join(repr(k) for k in keys[:_MAX_LISTED_KEYS])
            return head + (
                "" if len(keys) <= _MAX_LISTED_KEYS
                else f" (+{len(keys) - _MAX_LISTED_KEYS} more)"
            )
        raise PredicateError(
            f"{window}: scoring sidecar does not reconcile with the serving "
            f"evidence — {len(ev_only)} evidence row(s) WITHOUT a score row "
            f"({_shown(ev_only) if ev_only else 'none'}) and {len(sc_only)} "
            f"score row(s) WITHOUT a serving parent "
            f"({_shown(sc_only) if sc_only else 'none'}). Every exclusion "
            "must be countable (§9.10); re-run the scoring pass against this "
            "sealed tree"
        )

    joined: list[dict[str, Any]] = []
    for key, ev in ev_keyed.items():
        if "ok" not in ev:
            raise PredicateError(
                f"{window}: evidence row {key!r} carries no 'ok' integrity "
                "stamp — pre-#127 evidence cannot feed the §8.5 predicate "
                "(the stamp is what separates a serving failure from a "
                "scored answer, H2)"
            )
        score = sc_keyed[key]
        row: dict[str, Any] = dict(score)
        row["example_id"], row["repeat_index"], row["record_index"] = key
        row["ok"] = bool(ev["ok"])
        row["error"] = ev.get("error")
        row["empty_generation"] = ev.get("empty_generation")
        joined.append(row)
    return joined


def _verdict_value(row: Mapping[str, Any], column: str, window: str) -> float | None:
    value = row.get(column)
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise PredicateError(
            f"{window}: verdict column {column!r} holds non-numeric value "
            f"{value!r} — the predicate never coerces garbage"
        )
    return float(value)


def compute_window_predicate(
    joined_rows: Sequence[Mapping[str, Any]],
    dataset: str,
    config: PredicateConfig,
    *,
    window: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The §8.5 per-query predicate over one window's joined rows.

    Returns ``(predicate_rows, summary)``. Refusals (PredicateError): dataset
    outside the predicate universe; qasper rows without a configured τ
    (#120); a required verdict column absent from EVERY joined row; a null
    fraction above ``config.max_null_fraction``; EM values outside {0, 1}.
    """
    column = required_verdict_column(dataset)
    span_qa = dataset in SPAN_QA_DATASETS
    if not span_qa and config.qasper_tau is None:
        raise PredicateError(
            f"{window}: dataset {dataset!r} takes the §8.5 groundedness "
            "branch (Instrument A at calibrated τ) but no qasper_tau is "
            "configured — the τ pairing is an OPEN owner decision (task "
            "#120 / Topic-6 F7); refusing to default"
        )
    if not any(column in row for row in joined_rows):
        raise PredicateError(
            f"{window}: required verdict column {column!r} is absent from "
            f"every scoring-sidecar row for dataset {dataset!r} — the §8.5 "
            f"{'correctness' if span_qa else 'groundedness'} branch cannot "
            "be computed; re-run the scoring pass with the instrument that "
            "produces it"
        )
    rule = SPAN_QA_RULE if span_qa else QASPER_RULE

    rows_out: list[dict[str, Any]] = []
    n_not_ok = 0
    n_missing_verdict = 0
    for row in joined_rows:
        out: dict[str, Any] = {
            "example_id": row["example_id"],
            "repeat_index": row["repeat_index"],
            "record_index": row["record_index"],
            "ok": bool(row["ok"]),
            "dataset": dataset,
            "predicate_rule": rule,
            "verdict_column": column,
        }
        # §8.5 protocol: contradiction/neutral reported SEPARATELY — carried
        # through (nulled with the rest on not-ok rows), never in the predicate.
        for sep in _SEPARATE_REPORT_COLUMNS:
            out[sep] = None if not out["ok"] else row.get(sep)
        if not out["ok"]:
            # H2 consumer-side nulling: verdicts computed on a serving
            # failure's empty answer are garbage — nulled and counted.
            n_not_ok += 1
            out["verdict"] = None
            out["predicate"] = None
            out["predicate_null_reason"] = "not_ok"
            rows_out.append(out)
            continue
        verdict = _verdict_value(row, column, window)
        out["verdict"] = verdict
        if verdict is None:
            n_missing_verdict += 1
            out["predicate"] = None
            out["predicate_null_reason"] = "missing_verdict"
            rows_out.append(out)
            continue
        if span_qa:
            if verdict not in (0.0, 1.0):
                raise PredicateError(
                    f"{window}: exact_match value {verdict!r} outside "
                    "{0, 1} — the abstention-aware EM column is binary by "
                    "contract (quality.py fix #4)"
                )
            out["predicate"] = verdict == 1.0
        else:
            assert config.qasper_tau is not None
            out["predicate"] = verdict >= float(config.qasper_tau)
        out["predicate_null_reason"] = None
        rows_out.append(out)

    summary = summarize_predicate_rows(rows_out)
    summary["n_not_ok_nulled"] = n_not_ok
    summary["n_missing_verdict"] = n_missing_verdict
    summary["dataset"] = dataset
    summary["predicate_rule"] = rule
    summary["verdict_column"] = column
    if summary["n_rows"] and summary["null_fraction"] > float(config.max_null_fraction):
        raise PredicateError(
            f"{window}: predicate null fraction "
            f"{summary['null_fraction']:.3f} ({summary['n_null']}/"
            f"{summary['n_rows']}: {n_not_ok} not-ok nulled, "
            f"{n_missing_verdict} missing verdict) breaches the configured "
            f"sanity bound max_null_fraction={float(config.max_null_fraction)} "
            "— a run whose predicate is mostly unscoreable must not feed Y; "
            "fix the chain or re-state the bound deliberately"
        )
    return rows_out, summary


def summarize_predicate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Counted accounting for one window's predicate rows (§9.10)."""
    n_rows = len(rows)
    n_true = sum(1 for r in rows if r.get("predicate") is True)
    n_false = sum(1 for r in rows if r.get("predicate") is False)
    n_null = sum(1 for r in rows if r.get("predicate") is None)
    separate: dict[str, dict[str, Any]] = {}
    for sep in _SEPARATE_REPORT_COLUMNS:
        values = [
            float(r[sep]) for r in rows
            if isinstance(r.get(sep), (int, float)) and not isinstance(r.get(sep), bool)
        ]
        separate[sep] = {
            "n": len(values),
            "mean": (sum(values) / len(values)) if values else None,
        }
    return {
        "n_rows": n_rows,
        "n_true": n_true,
        "n_false": n_false,
        "n_null": n_null,
        "null_fraction": (n_null / n_rows) if n_rows else 0.0,
        # §8.5: contradiction/neutral are SEPARATE reported measurements.
        "reported_separately": separate,
    }
