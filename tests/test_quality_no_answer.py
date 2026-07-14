"""SQuAD v2 no-answer scoring + abstention detection (fix #4, options A + B).

evaluate_f1_score references no instance state, so we call it unbound with ``None`` as
self -- this exercises the real scoring logic without loading LettuceDetect / NLI models.
"""
from __future__ import annotations

import pytest

from src.evaluation.quality import QualityEvaluator, is_no_answer_prediction


def _f1(generated: str, reference: str) -> dict:
    # Unbound call: the method uses no ``self`` attributes, so this avoids model init.
    return QualityEvaluator.evaluate_f1_score(None, generated, reference)


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
