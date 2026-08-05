"""D8 §8.5 windowed grounding for LettuceDetect (Instrument A) -- LAUNCH BLOCKER.

Beyond the detector's L_max the charter mandates: "windowed max-support with
``scored_windowed=true`` on every affected row -- windowing is an alert, never
silent". The limit is derived AT RUNTIME from the loaded model config
(max_position_embeddings, capped by the detector's own max_length) -- never
hardcoded. Over-long context docs split into sentence-aligned ~50%-overlap
windows (the same core algorithm as the NLI premise windows); the answer is
scored against EACH window and verdicts aggregate by MAX-SUPPORT per answer
character: a character is unsupported only if NO window supports it.

The fakes mimic the lettucedetect HallucinationDetector layout
(.detector.tokenizer / .detector.model.config.max_position_embeddings) and are
evidence-aware: an answer segment is flagged unless its evidence string is
visible in the served (per-call) context -- exactly the truncation failure mode
windowing exists to fix.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.quality import (  # noqa: E402
    InstrumentUnavailableError,
    QualityEvaluator,
)


class _FakeTokenizer:
    """Whitespace tokenizer with the HF call convention ({'input_ids': [...]})."""

    def __call__(self, text: Any, add_special_tokens: bool = False, **kw: Any) -> Dict[str, Any]:
        if isinstance(text, list):
            return {"input_ids": [[0] * len(t.split()) for t in text]}
        return {"input_ids": [0] * len(text.split())}


class _FakeConfig:
    def __init__(self, max_position_embeddings: int) -> None:
        self.max_position_embeddings = max_position_embeddings


class _FakeModel:
    def __init__(self, l_max: int) -> None:
        self.config = _FakeConfig(l_max)


class _FakeInner:
    """The wrapped TransformerDetector: tokenizer + model.config."""

    def __init__(self, l_max: int) -> None:
        self.tokenizer = _FakeTokenizer()
        self.model = _FakeModel(l_max)


class _FakeDetector:
    """Evidence-aware fake HallucinationDetector.

    ``support_map`` is a list of (answer_substring, evidence_substring): the
    answer substring is flagged as an unsupported span unless its evidence
    string appears in THIS call's context -- per-call visibility, like the
    real truncation-bounded detector.
    """

    def __init__(
        self, l_max: int, support_map: List[Tuple[str, str]]
    ) -> None:
        self.detector = _FakeInner(l_max)
        self.support_map = support_map
        self.calls: List[Dict[str, Any]] = []

    def predict(
        self, context: List[str], question: str, answer: str,
        output_format: str = "spans",
    ) -> List[Dict[str, Any]]:
        self.calls.append(
            {"context": list(context), "question": question, "answer": answer}
        )
        combined = " ".join(context).lower()
        spans: List[Dict[str, Any]] = []
        for ans_sub, evidence in self.support_map:
            start = answer.find(ans_sub)
            if start < 0:
                continue
            if evidence.lower() not in combined:
                spans.append(
                    {"start": start, "end": start + len(ans_sub), "text": ans_sub}
                )
        return spans


QUESTION = "What is the launch code?"  # 5 whitespace tokens
ANSWER = "The code is 8241."  # 4 whitespace tokens; not an abstention
EVIDENCE = "the secret code is 8241"


def _evaluator(det: Any, **overrides: Any) -> QualityEvaluator:
    kwargs: Dict[str, Any] = dict(
        use_nli=False, use_embeddings=False, use_bertscore=False,
        use_rouge=False, use_lettucedetect=True, strict=True,
    )
    kwargs.update(overrides)
    ev = QualityEvaluator(**kwargs)
    ev._lettucedetect_model = det  # bypass lazy HF loading
    return ev


def _budget(ev: QualityEvaluator, l_max: int, question: str, answer: str) -> int:
    q = len(question.split())
    a = len(answer.split())
    return l_max - q - a - ev.LETTUCE_SPECIAL_TOKENS_MARGIN


def _filler(n: int, stem: str = "Filler sentence number") -> str:
    return " ".join(f"{stem} {i} says nothing useful here." for i in range(n))


# --------------------------------------------------------------------------- #
# Native path: context within the runtime-derived L_max is scored in ONE call
# --------------------------------------------------------------------------- #
def test_short_context_stays_native_single_call() -> None:
    det = _FakeDetector(l_max=200, support_map=[(ANSWER, EVIDENCE)])
    ev = _evaluator(det)
    ctx = [f"Intro sentence. {EVIDENCE}."]
    r = ev.evaluate_hallucination(QUESTION, ctx, ANSWER)
    assert len(det.calls) == 1
    # The full multi-doc context list is passed through unchanged (old behavior).
    assert det.calls[0]["context"] == ctx
    assert r["scored_windowed"] is False
    assert r["grounding_score"] == 1.0
    assert r["hallucination_detected"] is False


def test_boundary_exact_fit_is_native_one_more_token_windows() -> None:
    l_max = 200
    det = _FakeDetector(l_max=l_max, support_map=[])
    ev = _evaluator(det)
    budget = _budget(ev, l_max, QUESTION, ANSWER)
    assert budget > 0
    # Exactly at the budget: native. Sentence terminators keep windowing
    # sentence-aligned when it does trigger.
    doc_fit = " ".join(["tok"] * budget)
    r = ev.evaluate_hallucination(QUESTION, [doc_fit], ANSWER)
    assert r["scored_windowed"] is False
    assert len(det.calls) == 1

    det2 = _FakeDetector(l_max=l_max, support_map=[])
    ev2 = _evaluator(det2)
    doc_over = _filler(40)  # ~280 tokens > budget
    assert len(doc_over.split()) > budget
    r2 = ev2.evaluate_hallucination(QUESTION, [doc_over], ANSWER)
    assert r2["scored_windowed"] is True
    assert len(det2.calls) >= 2


# --------------------------------------------------------------------------- #
# The launch-blocker scenario: evidence past L_max recovered by windowing
# --------------------------------------------------------------------------- #
def test_windowed_pass_recovers_late_evidence() -> None:
    l_max = 200
    det = _FakeDetector(l_max=l_max, support_map=[(ANSWER, EVIDENCE)])
    ev = _evaluator(det)
    # Evidence sits at the END of a doc far past the window budget: the old
    # single-call path would truncate it away and flag the whole answer.
    long_doc = f"{_filler(40)} The vault log adds: {EVIDENCE}."
    r = ev.evaluate_hallucination(QUESTION, [long_doc], ANSWER)
    assert r["scored_windowed"] is True
    assert len(det.calls) >= 2
    # Max-support: SOME window contains the evidence, so the answer is grounded.
    assert r["grounding_score"] == 1.0
    assert r["hallucination_detected"] is False
    assert r["hallucinated_span_ratio"] == 0.0
    assert r["hallucinated_spans"] == []
    # Every windowed call serves a single window within the budget.
    budget = _budget(ev, l_max, QUESTION, ANSWER)
    for call in det.calls:
        assert len(call["context"]) == 1
        assert len(call["context"][0].split()) <= budget


def test_max_support_unions_evidence_across_windows() -> None:
    # Part A's evidence lives in doc 1 (fits whole), part B's at the END of an
    # over-long doc 2. No single call sees both; the per-character max-support
    # aggregation must still ground BOTH answer parts.
    l_max = 200
    part_a = "The alpha code is 11."
    part_b = "The beta code is 22."
    answer = f"{part_a} {part_b}"
    det = _FakeDetector(
        l_max=l_max,
        support_map=[(part_a, "alpha evidence 11"), (part_b, "beta evidence 22")],
    )
    ev = _evaluator(det)
    doc1 = "Intro. Contains alpha evidence 11 in a short doc."
    doc2 = f"{_filler(40)} Late line: beta evidence 22."
    r = ev.evaluate_hallucination(QUESTION, [doc1, doc2], answer)
    assert r["scored_windowed"] is True
    # Sanity: every individual call flagged SOMETHING (no call saw both
    # evidences), so a grounded verdict can only come from the union.
    assert all(
        "alpha evidence 11" not in c["context"][0]
        or "beta evidence 22" not in c["context"][0]
        for c in det.calls
    )
    assert r["grounding_score"] == 1.0
    assert r["hallucinated_spans"] == []
    # Docs that fit are served whole as their own window.
    assert any(c["context"] == [doc1] for c in det.calls)


def test_windowed_unsupported_segment_stays_flagged_with_rebuilt_spans() -> None:
    l_max = 200
    part_a = "The alpha code is 11."
    part_b = "The gamma code is 99."  # no evidence anywhere
    answer = f"{part_a} {part_b}"
    det = _FakeDetector(
        l_max=l_max,
        support_map=[(part_a, "alpha evidence 11"), (part_b, "gamma evidence 99")],
    )
    ev = _evaluator(det)
    doc = f"{_filler(40)} Late line: alpha evidence 11."
    r = ev.evaluate_hallucination(QUESTION, [doc], answer)
    assert r["scored_windowed"] is True
    assert r["hallucination_detected"] is True
    # Exactly part B's characters are flagged; spans rebuilt from the
    # aggregated flag array carry offsets into the answer.
    start = answer.find(part_b)
    assert r["hallucinated_spans"] == [
        {"start": start, "end": start + len(part_b), "text": part_b}
    ]
    expected_ratio = len(part_b) / len(answer)
    assert abs(r["hallucinated_span_ratio"] - expected_ratio) < 1e-9
    assert abs(r["grounding_score"] - (1.0 - expected_ratio)) < 1e-9


def test_single_giant_sentence_is_hard_split_not_silently_truncated() -> None:
    """A doc that is ONE giant 'sentence' (no terminators -- key-value dump /
    minified blob) defeats the sentence-aligned windower: it used to be served
    whole and over budget, leaving the detector's internal truncation to
    silently drop the evidence past the horizon inside a scored_windowed=True
    row. The D8 §8.5 re-check must hard-split it so EVERY served window fits
    the budget and end-of-blob evidence is recovered by max-support."""
    l_max = 200
    det = _FakeDetector(l_max=l_max, support_map=[(ANSWER, EVIDENCE)])
    ev = _evaluator(det)
    budget = _budget(ev, l_max, QUESTION, ANSWER)
    blob = " ".join(f"key{i}=value{i}" for i in range(400)) + f" {EVIDENCE}"
    assert "." not in blob  # truly terminator-free: one giant "sentence"
    assert len(blob.split()) > budget
    r = ev.evaluate_hallucination(QUESTION, [blob], ANSWER)
    assert r["scored_windowed"] is True
    assert len(det.calls) >= 2
    # Every served window verifiably fits the detector budget -- nothing left
    # for internal truncation to drop silently.
    for call in det.calls:
        assert len(call["context"]) == 1
        assert len(call["context"][0].split()) <= budget
    # The evidence at the very END of the blob (past the old truncation
    # horizon) is seen whole by some window: the answer is grounded.
    assert any(EVIDENCE in c["context"][0] for c in det.calls)
    assert r["grounding_score"] == 1.0
    assert r["hallucination_detected"] is False


def test_windows_are_sentence_aligned() -> None:
    l_max = 200
    det = _FakeDetector(l_max=l_max, support_map=[])
    ev = _evaluator(det)
    doc = " ".join(
        f"Sentence number {i} contains exactly eight useful words here." for i in range(60)
    )
    ev.evaluate_hallucination(QUESTION, [doc], ANSWER)
    assert len(det.calls) >= 2
    for call in det.calls:
        w = call["context"][0]
        assert w.startswith("Sentence number")
        assert w.rstrip().endswith(("here.", "here"))
    # Nothing lost: every sentence appears in some window.
    joined = " ".join(c["context"][0] for c in det.calls)
    for i in range(60):
        assert f"Sentence number {i} " in joined + " "


# --------------------------------------------------------------------------- #
# scored_windowed rides the row (alert, never silent) + L_max derivation
# --------------------------------------------------------------------------- #
def test_scored_windowed_lands_in_quality_metrics_and_to_dict() -> None:
    det = _FakeDetector(l_max=200, support_map=[(ANSWER, EVIDENCE)])
    ev = _evaluator(det)
    long_doc = f"{_filler(40)} The vault log adds: {EVIDENCE}."
    m = ev.evaluate(QUESTION, [long_doc], ANSWER, "8241")
    assert m.scored_windowed is True
    assert m.to_dict()["scored_windowed"] == 1.0
    assert m.grounding_score == 1.0


def test_unwindowed_row_emits_false_not_missing() -> None:
    det = _FakeDetector(l_max=200, support_map=[(ANSWER, EVIDENCE)])
    ev = _evaluator(det)
    m = ev.evaluate(QUESTION, [f"Short. {EVIDENCE}."], ANSWER, "8241")
    assert m.scored_windowed is False
    assert m.to_dict()["scored_windowed"] == 0.0


def test_abstention_row_never_windows() -> None:
    det = _FakeDetector(l_max=200, support_map=[])
    ev = _evaluator(det)
    m = ev.evaluate(QUESTION, [_filler(40)], "I don't know.", "8241")
    assert m.scored_windowed is False
    assert m.grounding_instrument is None
    assert det.calls == []


def test_l_max_capped_by_detector_max_length() -> None:
    # TransformerDetector-style max_length caps the config limit.
    det = _FakeDetector(l_max=100_000, support_map=[])
    det.detector.max_length = 200  # type: ignore[attr-defined]
    ev = _evaluator(det)
    doc = _filler(40)  # ~280 tokens: over the capped budget, under the config one
    r = ev.evaluate_hallucination(QUESTION, [doc], ANSWER)
    assert r["scored_windowed"] is True


# --------------------------------------------------------------------------- #
# Fail-closed: underivable tokenizer/L_max never scores blind
# --------------------------------------------------------------------------- #
class _OpaqueDetector:
    """No tokenizer, no model config anywhere -- limits underivable."""

    def predict(self, **kw: Any) -> List[Dict[str, Any]]:  # pragma: no cover
        return []


def test_underivable_limits_strict_raises() -> None:
    ev = _evaluator(_OpaqueDetector(), strict=True)
    with pytest.raises(InstrumentUnavailableError) as ei:
        ev.evaluate_hallucination(QUESTION, ["Some context."], ANSWER)
    assert ei.value.instrument == "lettucedetect"


def test_underivable_limits_nonstrict_labels_row() -> None:
    ev = _evaluator(_OpaqueDetector(), strict=False)
    r = ev.evaluate_hallucination(QUESTION, ["Some context."], ANSWER)
    assert r["grounding_score"] is None
    assert any(
        t.startswith("lettucedetect:error:") for t in ev._row_status_tokens
    )


def test_question_plus_answer_overflow_fails_closed() -> None:
    # Windowing the context cannot fix a question+answer that alone exceed
    # L_max: typed failure, never a blind score.
    l_max = 80  # budget = 80 - len(q) - len(a) - 64 margin < 0 for a 20-token q
    det = _FakeDetector(l_max=l_max, support_map=[])
    ev = _evaluator(det, strict=True)
    long_q = " ".join(["why"] * 30)
    with pytest.raises(InstrumentUnavailableError):
        ev.evaluate_hallucination(long_q, ["ctx doc"], ANSWER)
