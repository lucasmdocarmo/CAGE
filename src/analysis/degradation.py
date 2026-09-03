"""§8.11 degradation taxonomy — per-request deterministic classifiers (W4.10).

PUBLICATION.md §8.11 registers a degradation taxonomy per failed request —
truncated-generation · repetition/loop · abstention-shift · missing-evidence
fabrication · wrong-context answer — and one falsifiable fingerprint per §2.4
coping policy. This module is the LABEL side: five deterministic, per-request
classifiers over the STORED run rows (the requests.jsonl ⋈ qa_evidence.jsonl
trial record — the producer schema of scripts/3_run/run_experiment.py), so the
§9.3 decomposition legs of contrast #13 (3 Holm superiority + 3 conditional
TOST, ``src.analysis.stats.families.FINGERPRINT_SUB_HYPOTHESES``) can consume
label rates as their fingerprint instrument.

Doctrine (fail-closed, absence stays absence):

- A label that cannot be judged from the row is ``value=None`` with a NAMED
  ``reason`` saying exactly which column is missing/unusable — never a guessed
  label. Downstream threading (run_campaign_analysis.load_per_query) keeps
  None labels ABSENT from the per-query frame and surfaces the counted
  reasons; a 0.0 here would fabricate "no failure" out of missing telemetry.
- No models, no randomness: every classifier is a pure function of the stored
  row, reusing the run's OWN scored columns where they exist. Abstention
  detection is quality.py's (``is_no_answer_prediction`` on the sanitized
  text) — never a second detector; grounding reuses the stored
  ``grounded``/``grounding_score`` columns at the SAME τ the producer used.
- The S2 policy-event JOIN (§8.11: "the S2 ledger join is the test") is
  OWNER-GATED: ``join_policy_events`` exists but refuses with a named
  not-available-until-S2-decision reason. Nothing here may fabricate a
  per-request policy-event attribution before that decision lands.

Repetition detector parameters are PINNED module constants (a tunable
detector would be an unregistered instrument).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, NoReturn

from src.evaluation.quality import is_no_answer_prediction, sanitize_answer

__all__ = [
    "DEGRADATION_LABELS",
    "DegradationError",
    "GROUNDED_TAU",
    "LABEL_COLUMNS",
    "LABEL_COLUMN_PREFIX",
    "LabelResult",
    "REPETITION_MAX_DISTINCT_RATIO",
    "REPETITION_MIN_TOKENS",
    "REPETITION_NGRAM_N",
    "S2_POLICY_EVENT_JOIN_UNAVAILABLE",
    "classify_abstention_shift",
    "classify_fabrication",
    "classify_repetition",
    "classify_request",
    "classify_truncated_generation",
    "classify_wrong_context",
    "column_of",
    "join_policy_events",
]

#: The §8.11 taxonomy, in charter order.
DEGRADATION_LABELS: tuple[str, ...] = (
    "truncated_generation",
    "repetition",
    "abstention_shift",
    "fabrication",
    "wrong_context",
)

#: Column-name prefix for the threaded per-query label columns. Prefixed so a
#: label column can never shadow a producer field (the loader refuses a raw
#: row that already carries one of these names).
LABEL_COLUMN_PREFIX = "deg_"


def column_of(label: str) -> str:
    """Per-query column name of one §8.11 label (fail loud on a non-label)."""
    if label not in DEGRADATION_LABELS:
        raise DegradationError(
            f"unknown degradation label {label!r} — the §8.11 taxonomy is "
            f"{list(DEGRADATION_LABELS)}"
        )
    return LABEL_COLUMN_PREFIX + label


LABEL_COLUMNS: tuple[str, ...] = tuple(column_of(l) for l in DEGRADATION_LABELS)

# --------------------------------------------------------------------------- #
# Pinned classifier parameters
# --------------------------------------------------------------------------- #

#: n-gram size of the repetition/loop detector.
REPETITION_NGRAM_N: int = 3
#: Below this many whitespace tokens a generation is too short to certify a
#: loop — classified False (a judgment on present text, not absence).
REPETITION_MIN_TOKENS: int = 20
#: distinct-n-gram ratio (unique / total) at or below which the generation is
#: labeled a repetition loop. 0.5 = every n-gram appears twice on average.
REPETITION_MAX_DISTINCT_RATIO: float = 0.5

#: Grounding-fail threshold on ``grounding_score`` — mirrors the producer's
#: ``grounded`` derivation (scripts/3_run/run_experiment.py: score >= 0.5),
#: used ONLY when the stored ``grounded`` flag itself is absent.
GROUNDED_TAU: float = 0.5

#: The OWNER-GATED S2 refusal — the ONE reason string for every policy-event
#: join surface (§8.11 fingerprints AND the §8.12 quality|policy-event curve).
S2_POLICY_EVENT_JOIN_UNAVAILABLE: str = (
    "S2 policy-event join not available: per-request attribution of "
    "preemption/eviction/offload events to the requests they touched is "
    "OWNER-GATED on the S2 instrumentation decision (PUBLICATION.md §8.11: "
    "'the S2 ledger join is the test'; §8.12 quality|policy-event). Until "
    "that decision lands there is no registered event ledger to join — "
    "absence stays absence, no per-request policy_event label is emitted. "
    "Fix: land the S2 decision, then implement the ledger join here."
)


class DegradationError(ValueError):
    """Invalid classifier input or a refused (owner-gated) join."""


@dataclass(frozen=True)
class LabelResult:
    """One label's verdict on one trial row.

    ``value`` True/False is a judged label; ``value=None`` means the row
    cannot be judged and ``reason`` NAMES exactly what is missing (the reason
    is None if and only if value is not None).
    """

    label: str
    value: bool | None
    reason: str | None

    def __post_init__(self) -> None:
        if (self.value is None) == (self.reason is None):
            raise DegradationError(
                f"label {self.label!r}: exactly one of value/reason must be "
                f"set (got value={self.value!r}, reason={self.reason!r}) — "
                "an unjudged label without a named reason is silent absence"
            )


# --------------------------------------------------------------------------- #
# Field readers (absence-honest)
# --------------------------------------------------------------------------- #


def _text_field(row: Mapping[str, Any], name: str) -> str | None:
    """A string field, or None when absent/None (empty string IS a value)."""
    value = row.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    return value


def _flag_field(row: Mapping[str, Any], name: str) -> bool | None:
    """A stored boolean-ish flag (JSON bool or 0/1 numeric); None = absent."""
    value = row.get(name)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 0.0, 1, 1.0):
        return bool(value)
    return None


def _is_error_row(row: Mapping[str, Any]) -> bool:
    """The producer's error semantics: a non-empty ``error`` string."""
    return bool(row.get("error"))


# --------------------------------------------------------------------------- #
# The five §8.11 classifiers
# --------------------------------------------------------------------------- #


def classify_truncated_generation(row: Mapping[str, Any]) -> LabelResult:
    """truncated-generation: the engine stopped at the token cap.

    ``finish_reason`` is the signal (producer column, every adapter stamps
    it): ``"length"`` → True, ``"stop"`` → False. An error row, a missing
    finish_reason, or an unrecognized value is None-with-reason — guessing a
    truncation label from an unknown stop cause would fabricate physics.
    """
    label = "truncated_generation"
    if _is_error_row(row):
        return LabelResult(label, None, (
            "error row (transport/serving failure) — there is no completed "
            "generation to judge for truncation"
        ))
    reason_value = row.get("finish_reason")
    if reason_value is None:
        return LabelResult(label, None, (
            "finish_reason absent from the stored row — the producer "
            "(run_experiment.py results row) stamps it; without it "
            "truncation cannot be judged"
        ))
    if not isinstance(reason_value, str):
        return LabelResult(label, None, (
            f"finish_reason is {reason_value!r} (not a string) — malformed "
            "row, refusing to guess"
        ))
    if reason_value == "length":
        return LabelResult(label, True, None)
    if reason_value == "stop":
        return LabelResult(label, False, None)
    return LabelResult(label, None, (
        f"finish_reason {reason_value!r} is outside the judged vocabulary "
        "('length'/'stop') — an unknown stop cause is not classifiable as "
        "truncated-or-not"
    ))


def classify_repetition(row: Mapping[str, Any]) -> LabelResult:
    """repetition/loop: distinct-n-gram collapse in the RAW generation.

    Runs on the raw ``generated_answer`` (NOT the sanitized text — a runaway
    template loop is exactly what sanitize_answer truncates away). Pinned
    parameters (module constants): ``REPETITION_NGRAM_N``-grams over
    whitespace tokens; fewer than ``REPETITION_MIN_TOKENS`` tokens → False
    (too short to loop); distinct-to-total n-gram ratio at or below
    ``REPETITION_MAX_DISTINCT_RATIO`` → True.
    """
    label = "repetition"
    if _is_error_row(row):
        return LabelResult(label, None, (
            "error row (transport/serving failure) — no generation to scan "
            "for repetition"
        ))
    text = _text_field(row, "generated_answer")
    if text is None:
        return LabelResult(label, None, (
            "generated_answer absent from the stored row — the repetition "
            "detector needs the raw generation text"
        ))
    tokens = text.split()
    if len(tokens) < REPETITION_MIN_TOKENS:
        return LabelResult(label, False, None)
    n = REPETITION_NGRAM_N
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    ratio = len(set(ngrams)) / len(ngrams)
    return LabelResult(label, ratio <= REPETITION_MAX_DISTINCT_RATIO, None)


def _abstained(row: Mapping[str, Any]) -> tuple[bool | None, str | None]:
    """Did the model abstain? Scored column first, quality.py detector second.

    Preference order (reuse, never duplicate): the run's own scored
    ``predicted_no_answer`` column (quality.py wrote it from
    ``is_no_answer_prediction``), else the SAME detector applied to the
    sanitized stored generation. None ⇒ named reason.
    """
    scored = _flag_field(row, "predicted_no_answer")
    if scored is not None:
        return scored, None
    if _is_error_row(row):
        return None, (
            "error row (transport/serving failure) — no generation to run "
            "abstention detection on"
        )
    text = _text_field(row, "generated_answer")
    if text is None:
        return None, (
            "neither the scored predicted_no_answer column nor a "
            "generated_answer text is present — abstention undecidable"
        )
    # quality.py's detector on quality.py's sanitized text — the exact
    # scoring-path semantics, recomputed only because the scored column is
    # absent from this row.
    return is_no_answer_prediction(sanitize_answer(text)), None


def _gold_answerable(row: Mapping[str, Any]) -> tuple[bool | None, str | None]:
    """Is the gold item answerable? Scored column first, gold fields second.

    Official SQuAD-v2 semantics (quality.py): no gold answers == unanswerable.
    Preference: scored ``is_answerable``; else ``all_answers`` (the full gold
    list, qa_evidence contract); else ``reference_answer``.
    """
    scored = _flag_field(row, "is_answerable")
    if scored is not None:
        return scored, None
    if "all_answers" in row and isinstance(row["all_answers"], list):
        golds = [a for a in row["all_answers"] if isinstance(a, str) and a.strip()]
        return bool(golds), None
    reference = _text_field(row, "reference_answer")
    if reference is not None:
        return bool(reference.strip()), None
    return None, (
        "no answerability signal in the stored row (is_answerable, "
        "all_answers and reference_answer all absent) — cannot tell an "
        "abstention-shift from a correct abstention"
    )


def classify_abstention_shift(row: Mapping[str, Any]) -> LabelResult:
    """abstention-shift: the model abstained where the gold IS answerable."""
    label = "abstention_shift"
    abstained, why_a = _abstained(row)
    if abstained is None:
        return LabelResult(label, None, why_a)
    if not abstained:
        return LabelResult(label, False, None)
    answerable, why_g = _gold_answerable(row)
    if answerable is None:
        return LabelResult(label, None, why_g)
    return LabelResult(label, bool(answerable), None)


def _grounding_fail(row: Mapping[str, Any]) -> tuple[bool | None, str | None]:
    """Did grounding FAIL? Stored ``grounded`` flag first, score at τ second.

    The stored flag is the producer's own ``grounding_score >= 0.5``
    derivation; the score fallback applies the SAME pinned τ
    (``GROUNDED_TAU``), never a new threshold.
    """
    grounded = _flag_field(row, "grounded")
    if grounded is not None:
        return not grounded, None
    score = row.get("grounding_score")
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        return float(score) < GROUNDED_TAU, None
    return None, (
        "grounding columns absent/unscored (grounded is "
        f"{row.get('grounded')!r}, grounding_score is "
        f"{row.get('grounding_score')!r}) — grounding-dependent labels "
        "need the LettuceDetect columns; unscored is missing data, never "
        "'ungrounded'"
    )


def classify_fabrication(row: Mapping[str, Any]) -> LabelResult:
    """missing-evidence fabrication: a confident answer that grounding rejects.

    Confident = NOT an abstention (quality.py detection, as above); grounding
    fail per the existing grounding columns. An abstention is never a
    fabrication (False); unknown abstention or unscored grounding ⇒
    None-with-reason.
    """
    label = "fabrication"
    abstained, why_a = _abstained(row)
    if abstained is None:
        return LabelResult(label, None, why_a)
    if abstained:
        return LabelResult(label, False, None)
    failed, why_g = _grounding_fail(row)
    if failed is None:
        return LabelResult(label, None, why_g)
    return LabelResult(label, bool(failed), None)


def classify_wrong_context(row: Mapping[str, Any]) -> LabelResult:
    """wrong-context answer: grounded in served text that is NOT the gold set.

    Reuses the existing containment machinery: ``gold_position_in_prompt``
    (producer column — 0-based index of the gold doc among the SERVED
    contexts by exact-or-containment match, -1 when absent) crossed with the
    stored grounding columns. Grounded answer + gold absent from the served
    context ⇒ the answer is supported by NON-gold context (True). An
    ungrounded answer, or gold present, is False; missing containment or
    grounding columns ⇒ None-with-reason.
    """
    label = "wrong_context"
    failed, why_g = _grounding_fail(row)
    if failed is None:
        return LabelResult(label, None, why_g)
    if failed:
        return LabelResult(label, False, None)
    position = row.get("gold_position_in_prompt")
    if position is None:
        return LabelResult(label, None, (
            "gold_position_in_prompt absent from the stored row — the "
            "producer's served-context containment column is required to "
            "tell gold-grounded from wrong-context-grounded"
        ))
    if isinstance(position, bool) or not isinstance(position, (int, float)) or (
        isinstance(position, float) and not position.is_integer()
    ):
        return LabelResult(label, None, (
            f"gold_position_in_prompt is {position!r} (not an integer) — "
            "malformed row, refusing to guess containment"
        ))
    position = int(position)
    if position < -1:
        return LabelResult(label, None, (
            f"gold_position_in_prompt={position} is outside the producer "
            "vocabulary (-1 = absent, >=0 = served index) — malformed row"
        ))
    return LabelResult(label, position == -1, None)


_CLASSIFIERS: tuple[Any, ...] = (
    classify_truncated_generation,
    classify_repetition,
    classify_abstention_shift,
    classify_fabrication,
    classify_wrong_context,
)


def classify_request(row: Mapping[str, Any]) -> dict[str, LabelResult]:
    """All five §8.11 labels for one merged trial row (label → LabelResult).

    ``row`` is the requests.jsonl ⋈ qa_evidence.jsonl record for ONE trial
    (same ``(example_id, record_index)`` identity). Purely deterministic;
    every unjudgeable label carries its named reason.
    """
    if not isinstance(row, Mapping):
        raise DegradationError(
            f"row must be a mapping, got {type(row).__name__}"
        )
    results: dict[str, LabelResult] = {}
    for fn in _CLASSIFIERS:
        result = fn(row)
        results[result.label] = result
    assert tuple(results) == DEGRADATION_LABELS, "classifier/taxonomy drift"
    return results


# --------------------------------------------------------------------------- #
# OWNER-GATED S2 stub
# --------------------------------------------------------------------------- #


def join_policy_events(*_args: Any, **_kwargs: Any) -> NoReturn:
    """The S2 policy-event JOIN — declared, refused until the S2 decision.

    §8.11's fingerprint test ("the S2 ledger join is the test") needs a
    per-request attribution of preemption/eviction/offload events. That
    instrumentation is an OPEN owner decision (S2); this stub is the named
    seam so callers get one honest refusal instead of five ad-hoc absences.
    """
    raise DegradationError(S2_POLICY_EVENT_JOIN_UNAVAILABLE)
