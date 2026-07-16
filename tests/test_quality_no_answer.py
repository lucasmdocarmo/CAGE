"""SQuAD v2 no-answer scoring + abstention detection (fix #4, options A + B).

evaluate_f1_score references no instance state, so we call it unbound with ``None`` as
self -- this exercises the real scoring logic without loading LettuceDetect / NLI models.
"""
from __future__ import annotations

import pytest

from src.evaluation.quality import (
    QualityEvaluator,
    is_no_answer_prediction,
    sanitize_answer,
)


def _f1(generated: str, reference: str, all_answers: list | None = None) -> dict:
    # Unbound call: the method uses no ``self`` attributes, so this avoids model init.
    return QualityEvaluator.evaluate_f1_score(None, generated, reference, all_answers)


# --------------------------------------------------------------------------- #
# Abstention detector
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "No answer.",
        "The context does not mention this.",
        "This question is unanswerable.",
        "I don't know.",
        "Not stated in the passage.",
        "There is insufficient information to answer.",
        # 2026-07-15 audit: the leading "I" must be OPTIONAL. The system prompt says
        # "say you don't know" and models emit the bare form -- 12/12 real abstentions
        # in the smoke run were exactly "Don't know." and every one was missed.
        "Don't know.",
        "don't know",
        "Do not know.",
        "I do not know.",
        # families the audit found missing entirely
        "Unknown",
        "unknown.",
        "N/A",
        "Not sure.",
        "No idea.",
        "Not enough information.",
        "Cannot be determined.",
        "This can't be determined from the passage.",
        "The answer cannot be found in the context.",
    ],
)
def test_abstention_detector_positive(text: str) -> None:
    assert is_no_answer_prediction(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "Paris",
        "No",   # a valid yes/no answer, NOT an abstention
        "Yes",
        "The Eiffel Tower is located in Paris, France.",
        # long answer that merely contains "no" must not trip the detector
        "There is no doubt the answer is Barack Obama who served two terms as president",
        # whole-answer-only tokens must NOT fire when part of a real answer
        "Unknown Pleasures",
        "The Great Unknown",
        # bare "none"/"Na" are deliberately NOT abstentions: both occur as real gold spans
        "None",
        "Na",
    ],
)
def test_abstention_detector_negative(text: str) -> None:
    assert is_no_answer_prediction(text) is False


# --------------------------------------------------------------------------- #
# SQuAD v2 scoring matrix
# --------------------------------------------------------------------------- #
def test_no_answer_item_correct_abstention_scores_one() -> None:
    r = _f1("The context does not provide an answer.", "")
    assert r["exact_match"] == 1.0 and r["f1"] == 1.0        # (A) credited
    assert r["is_answerable"] == 0.0
    assert r["no_answer_correct"] == 1.0
    assert r["f1_answerable"] is None                        # (B) excluded from answerable subset


def test_no_answer_item_hallucination_scores_zero() -> None:
    r = _f1("Paris", "")
    assert r["exact_match"] == 0.0 and r["f1"] == 0.0
    assert r["no_answer_correct"] == 0.0                     # wrong abstention decision
    assert r["predicted_no_answer"] == 0.0
    assert r["f1_answerable"] is None


def test_answerable_item_exact_match() -> None:
    r = _f1("Paris", "Paris")
    assert r["exact_match"] == 1.0 and r["f1"] == 1.0
    assert r["is_answerable"] == 1.0
    assert r["f1_answerable"] == 1.0 and r["exact_match_answerable"] == 1.0
    assert r["no_answer_correct"] is None                    # not a no-answer item


def test_answerable_item_wrong_abstention_scores_zero() -> None:
    r = _f1("I don't know.", "Paris")
    assert r["exact_match"] == 0.0
    assert r["predicted_no_answer"] == 1.0
    assert r["f1_answerable"] == 0.0                         # counts as a miss on the answerable subset


def test_answerable_item_partial_overlap() -> None:
    r = _f1("Paris France", "Paris")
    assert 0.0 < r["f1"] <= 1.0
    assert r["is_answerable"] == 1.0
    assert r["exact_match"] == 0.0                           # not an exact string match


# --------------------------------------------------------------------------- #
# Abstention precision indicator (mean over non-None rows == abstention precision)
# --------------------------------------------------------------------------- #
def test_abstention_precision_correct_abstention() -> None:
    r = _f1("Don't know.", "")
    assert r["abstention_precision"] == 1.0                  # abstained AND truly unanswerable
    assert r["no_answer_correct"] == 1.0                     # recall indicator agrees


def test_abstention_precision_wrong_abstention() -> None:
    r = _f1("Don't know.", "Paris")
    assert r["abstention_precision"] == 0.0                  # abstained on an answerable item


def test_abstention_precision_none_when_not_abstained() -> None:
    assert _f1("Paris", "Paris")["abstention_precision"] is None
    assert _f1("Paris", "")["abstention_precision"] is None  # hallucinated, no abstention predicted


# --------------------------------------------------------------------------- #
# Max over ALL gold answers (audit 2026-07-16 M5): official SQuAD v2 takes
# metric_max_over_ground_truths; only text[0] was scored before, understating
# answerable F1 ~5pp / EM ~10pp.
# --------------------------------------------------------------------------- #
def test_all_answers_max_over_golds() -> None:
    # reference (text[0]) misses, but a later gold matches exactly -> max wins.
    r = _f1("Paris", "London", all_answers=["London", "Paris"])
    assert r["f1"] == 1.0 and r["exact_match"] == 1.0
    assert r["precision"] == 1.0 and r["recall"] == 1.0  # from the F1-maximizing gold
    assert r["f1_answerable"] == 1.0 and r["exact_match_answerable"] == 1.0


def test_all_answers_em_and_f1_maximized_independently() -> None:
    # gold 1 gives partial F1 overlap and no EM; gold 2 gives nothing.
    r = _f1("x", "x y", all_answers=["x y", "z"])
    assert r["exact_match"] == 0.0
    assert r["f1"] == pytest.approx(2 * 1.0 * 0.5 / 1.5)
    assert r["f1_answerable"] == r["f1"]


def test_all_answers_none_falls_back_to_single_reference() -> None:
    # None (older evidence files / non-SQuAD datasets) must behave exactly as before.
    assert _f1("Paris", "London", all_answers=None) == _f1("Paris", "London")


def test_all_answers_empty_list_means_unanswerable() -> None:
    # Official SQuAD v2 semantics: empty gold list == no-answer item.
    r = _f1("Don't know.", "", all_answers=[])
    assert r["is_answerable"] == 0.0
    assert r["f1"] == 1.0 and r["exact_match"] == 1.0
    assert r["no_answer_correct"] == 1.0


def test_all_answers_abstention_on_answerable_stays_zero() -> None:
    r = _f1("I don't know.", "Paris", all_answers=["Paris", "the Paris"])
    assert r["f1"] == 0.0 and r["exact_match"] == 0.0
    assert r["predicted_no_answer"] == 1.0
    assert r["abstention_precision"] == 0.0


# --------------------------------------------------------------------------- #
# Answer sanitizer (B4 scoring side, 2026-07-16 pre-run package): scaffold prefix
# strip + fabricated prompt-continuation truncation. Scoring runs on the sanitized
# text; generated_answer is never overwritten.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("A: Paris", "Paris"),
        ("Answer: Paris", "Paris"),
        ("  A:   Paris", "Paris"),
        ("Answer. Paris", "Paris"),
        # fabricated continuation truncation (with and without a newline, numbered or not)
        ("Paris\nQuestion 2: Where is the Louvre?", "Paris"),
        ("Paris\nContext: The Eiffel Tower is in Paris.", "Paris"),
        ("Paris Context 3: more fabricated prompt", "Paris"),
        # combined scaffold + fabrication
        ("A: Paris\nQuestion: next one", "Paris"),
        # untouched answers
        ("Paris", "Paris"),
        ("The Eiffel Tower is located in Paris.", "The Eiffel Tower is located in Paris."),
        # 'Answers:' is NOT the scaffold token (requires A/Answer then [:.])
        ("Answers: Paris", "Answers: Paris"),
        ("", ""),
        (None, ""),
    ],
)
def test_sanitize_answer(raw, expected) -> None:
    assert sanitize_answer(raw) == expected


def _model_free_evaluator() -> QualityEvaluator:
    # Every model stack OFF: evaluate() exercises the sanitizer + model-free scoring
    # only (relevance/faithfulness/grounding/completeness come back None).
    return QualityEvaluator(
        use_nli=False, use_embeddings=False, use_bertscore=False,
        use_rouge=False, use_lettucedetect=False,
    )


def test_evaluate_scores_on_sanitized_text() -> None:
    m = _model_free_evaluator().evaluate(
        question="What is the capital of France?",
        context=["France's capital is Paris."],
        generated_text="A: Paris\nQuestion 2: What is the capital of Spain?",
        reference_answer="Paris",
    )
    # Scaffold+fabrication removed BEFORE F1/EM: exact match on the gold span.
    assert m.exact_match == 1.0 and m.f1_score == 1.0
    assert m.sanitized_answer == "Paris"


def test_evaluate_abstention_detected_on_sanitized_text() -> None:
    # 'A: I don't know.' must be credited as a correct abstention on a no-answer item.
    m = _model_free_evaluator().evaluate(
        question="q", context=["ctx"],
        generated_text="A: I don't know.", reference_answer="",
    )
    assert m.predicted_no_answer == 1.0
    assert m.exact_match == 1.0 and m.f1_score == 1.0
    assert m.no_answer_correct == 1.0


def test_sanitized_answer_and_premise_mode_in_to_dict() -> None:
    d = _model_free_evaluator().evaluate(
        question="q", context=["ctx"],
        generated_text="Answer: Paris", reference_answer="Paris",
    ).to_dict()
    assert d["sanitized_answer"] == "Paris"          # alongside, never overwriting
    assert "faithfulness_premise_mode" in d          # B2 column always present
    assert d["faithfulness_premise_mode"] is None    # NLI off -> not scored
