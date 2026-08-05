"""D8 §8.1 real batching for batch_evaluate.

The execution model is a separate GPU-batched post-serving pass ("inline
scoring was ~90% of wall-clock"); the old batch_evaluate looped evaluate() row
by row. The batched implementation accumulates (premise, claim) NLI pairs and
BERTScore texts ACROSS rows into single model calls. The contract proven here:
per-row outputs are IDENTICAL to the sequential path -- every QualityMetrics
field, including instrument_status token content and order -- while the fakes
record that the model was called once per batch, not once per pair/row.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.quality import (  # noqa: E402
    InstrumentUnavailableError,
    QualityEvaluator,
    QualityMetrics,
)


# --------------------------------------------------------------------------- #
# Deterministic fakes shared by both paths
# --------------------------------------------------------------------------- #
class _FakeTokenizer:
    def __call__(self, text: Any, add_special_tokens: bool = False, **kw: Any) -> Dict[str, Any]:
        if isinstance(text, list):
            return {"input_ids": [[0] * len(t.split()) for t in text]}
        return {"input_ids": [0] * len(text.split())}


class _FakeNLI:
    """Deterministic pair scorer accepting BOTH the sequential single-dict call
    and the batched list-of-dicts call (like the real HF pipeline)."""

    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()
        self.single_calls = 0
        self.batch_calls = 0

    @staticmethod
    def _pair_score(premise: str, claim: str) -> float:
        # Deterministic function of the pair alone: containment + length jitter.
        base = 0.9 if claim.lower().rstrip(".!? ") in premise.lower() else 0.1
        return round(base + 0.001 * (len(claim) % 7), 6)

    def _result(self, item: Dict[str, str]) -> List[Dict[str, Any]]:
        s = self._pair_score(item["text"], item["text_pair"])
        return [
            {"label": "entailment", "score": s},
            {"label": "neutral", "score": max(0.0, 1.0 - s - 0.01)},
            {"label": "contradiction", "score": 0.01},
        ]

    def __call__(self, inputs: Any, top_k: Any = None, truncation: bool = True,
                 max_length: int = 512, batch_size: Optional[int] = None) -> Any:
        if isinstance(inputs, list):
            self.batch_calls += 1
            return [self._result(x) for x in inputs]
        self.single_calls += 1
        return self._result(inputs)


class _Scalar:
    def __init__(self, v: float) -> None:
        self._v = v

    def cpu(self) -> "_Scalar":
        return self

    def numpy(self) -> Any:
        return np.float64(self._v)


class _Vec:
    def __init__(self, vals: List[float]) -> None:
        self._vals = [_Scalar(v) for v in vals]

    def __getitem__(self, i: int) -> _Scalar:
        return self._vals[i]


class _FakeBERTScorer:
    """Deterministic per-(cand, ref) scorer; batching must not change values."""

    def __init__(self) -> None:
        self.calls: List[int] = []  # batch sizes per score() call

    @staticmethod
    def _pair_f1(cand: str, ref: str) -> float:
        c, r = set(cand.lower().split()), set(ref.lower().split())
        return round(len(c & r) / max(1, len(c | r)), 6)

    def score(self, cands: List[str], refs: List[str]) -> Any:
        self.calls.append(len(cands))
        vals = [self._pair_f1(c, r) for c, r in zip(cands, refs)]
        return _Vec(vals), _Vec(vals), _Vec(vals)


class _FakeDetector:
    """Evidence-aware detector with derivable limits (windowing-capable)."""

    class _Inner:
        def __init__(self) -> None:
            self.tokenizer = _FakeTokenizer()

            class _Model:
                class _Cfg:
                    max_position_embeddings = 300

                config = _Cfg()

            self.model = _Model()

    def __init__(self) -> None:
        self.detector = self._Inner()
        self.n_calls = 0

    def predict(self, context: List[str], question: str, answer: str,
                output_format: str = "spans") -> List[Dict[str, Any]]:
        self.n_calls += 1
        combined = " ".join(context).lower()
        if answer.lower() in combined:
            return []
        return [{"start": 0, "end": len(answer), "text": answer}]


class _FakeRouge:
    """Deterministic ROUGE-L stand-in (rouge_score is absent in the CI venv)."""

    class _Score:
        def __init__(self, f: float) -> None:
            self.fmeasure = f

    def score(self, ref: str, cand: str) -> Dict[str, "_FakeRouge._Score"]:
        r, c = set(ref.lower().split()), set(cand.lower().split())
        f = round(len(r & c) / max(1, len(r | c)), 6)
        return {"rougeL": self._Score(f)}


def _make_evaluator() -> QualityEvaluator:
    ev = QualityEvaluator(
        use_nli=True, use_embeddings=False, use_bertscore=True,
        use_rouge=True, use_lettucedetect=True, strict=True,
    )
    ev._nli_model = _FakeNLI()
    ev._bertscore_model = _FakeBERTScorer()
    ev._lettucedetect_model = _FakeDetector()
    ev._rouge_scorer = _FakeRouge()
    return ev


def _filler(n: int) -> str:
    return " ".join(f"Filler sentence number {i} says nothing useful here." for i in range(n))


_ANSWER0 = "The code is 8241."
_ROWS: Dict[str, List[Any]] = {
    # normal answered row, evidence present
    "questions": [
        "What is the launch code?",
        "What is the capital?",
        "Unanswerable thing?",
        "Empty generation?",
        "Two claims?",
        "Long context?",
    ],
    "contexts": [
        [f"Intro sentence. {_ANSWER0}"],
        ["Paris is the capital of France."],
        ["Some unrelated paragraph about weather."],
        ["Context exists but the model produced nothing."],
        ["Alpha is true. Beta is unknown to this corpus."],
        [f"{_filler(50)} {_ANSWER0}"],  # ~350 tokens: windows under L_max 300
    ],
    "generated_texts": [
        _ANSWER0,
        "The capital is Paris.",
        "I don't know.",  # abstention
        "",  # empty generation == abstention semantics
        "Alpha is true. Gamma is false.",
        _ANSWER0,
    ],
    "reference_answers": [
        "8241",
        "Paris",
        "",  # unanswerable gold
        "something",
        "Alpha",
        "8241",
    ],
}
_ALL_ANSWERS: List[Optional[List[str]]] = [
    ["8241", "code 8241"], None, [], None, ["Alpha"], None,
]


def _sequential(ev: QualityEvaluator, all_answers: Optional[List[Optional[List[str]]]] = None) -> List[QualityMetrics]:
    out = []
    for i in range(len(_ROWS["questions"])):
        aa = all_answers[i] if all_answers is not None else None
        out.append(
            ev.evaluate(
                _ROWS["questions"][i], _ROWS["contexts"][i],
                _ROWS["generated_texts"][i], _ROWS["reference_answers"][i],
                all_answers=aa,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# The equivalence proof: batched == sequential, field for field
# --------------------------------------------------------------------------- #
def test_batched_outputs_identical_to_sequential() -> None:
    ev_seq = _make_evaluator()
    seq = _sequential(ev_seq)
    ev_bat = _make_evaluator()
    bat = ev_bat.batch_evaluate(
        _ROWS["questions"], _ROWS["contexts"],
        _ROWS["generated_texts"], _ROWS["reference_answers"],
    )
    assert len(bat) == len(seq) == 6
    for i, (s, b) in enumerate(zip(seq, bat)):
        assert b == s, f"row {i} diverged:\nseq={s}\nbat={b}"
        assert b.to_dict() == s.to_dict(), f"row {i} to_dict diverged"


def test_batched_outputs_identical_with_all_answers() -> None:
    ev_seq = _make_evaluator()
    seq = _sequential(ev_seq, _ALL_ANSWERS)
    ev_bat = _make_evaluator()
    bat = ev_bat.batch_evaluate(
        _ROWS["questions"], _ROWS["contexts"],
        _ROWS["generated_texts"], _ROWS["reference_answers"],
        all_answers=_ALL_ANSWERS,
    )
    for i, (s, b) in enumerate(zip(seq, bat)):
        assert b == s, f"row {i} diverged with all_answers"


def test_batched_issues_single_model_calls() -> None:
    ev = _make_evaluator()
    ev.batch_evaluate(
        _ROWS["questions"], _ROWS["contexts"],
        _ROWS["generated_texts"], _ROWS["reference_answers"],
    )
    nli: _FakeNLI = ev._nli_model  # type: ignore[assignment]
    bs: _FakeBERTScorer = ev._bertscore_model  # type: ignore[assignment]
    # ONE batched NLI call across every row's pairs, zero per-pair calls.
    assert nli.batch_calls == 1
    assert nli.single_calls == 0
    # ONE batched BERTScore call covering all scoring-eligible rows.
    assert bs.calls == [4]  # rows 0, 1, 4, 5 (2 abstains/empty-ref, 3 empty-gen)


def test_sequential_path_calls_per_pair_baseline() -> None:
    # The comparison baseline really is per-pair/per-row (guards against the
    # equivalence test passing because both paths batch).
    ev = _make_evaluator()
    _sequential(ev)
    nli: _FakeNLI = ev._nli_model  # type: ignore[assignment]
    bs: _FakeBERTScorer = ev._bertscore_model  # type: ignore[assignment]
    assert nli.batch_calls == 0
    assert nli.single_calls > 1
    assert bs.calls == [1, 1, 1, 1]


def test_batched_false_reproduces_sequential_loop() -> None:
    ev_a = _make_evaluator()
    ev_b = _make_evaluator()
    a = _sequential(ev_a)
    b = ev_b.batch_evaluate(
        _ROWS["questions"], _ROWS["contexts"],
        _ROWS["generated_texts"], _ROWS["reference_answers"],
        batched=False,
    )
    assert a == b
    nli: _FakeNLI = ev_b._nli_model  # type: ignore[assignment]
    assert nli.batch_calls == 0  # genuinely the old loop


def test_windowed_grounding_row_identical_across_paths() -> None:
    # Row 5's context exceeds the fake detector's L_max: both paths must
    # produce the same windowed max-support verdict and scored_windowed flag.
    ev_seq = _make_evaluator()
    seq = _sequential(ev_seq)
    ev_bat = _make_evaluator()
    bat = ev_bat.batch_evaluate(
        _ROWS["questions"], _ROWS["contexts"],
        _ROWS["generated_texts"], _ROWS["reference_answers"],
    )
    assert seq[5].scored_windowed is True
    assert bat[5].scored_windowed is True
    assert bat[5].grounding_score == seq[5].grounding_score == 1.0


# --------------------------------------------------------------------------- #
# Failure parity: batched failures label rows exactly like sequential ones
# --------------------------------------------------------------------------- #
class _ExplodingNLI:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()

    def __call__(self, *a: Any, **kw: Any) -> Any:
        raise RuntimeError("stub NLI scoring failure")


def _make_nonstrict_exploding() -> QualityEvaluator:
    ev = QualityEvaluator(
        use_nli=True, use_embeddings=False, use_bertscore=False,
        use_rouge=False, use_lettucedetect=False, strict=False,
    )
    ev._nli_model = _ExplodingNLI()
    return ev


def test_nonstrict_batched_call_failure_matches_sequential_labels() -> None:
    ev_seq = _make_nonstrict_exploding()
    seq = _sequential(ev_seq)
    ev_bat = _make_nonstrict_exploding()
    bat = ev_bat.batch_evaluate(
        _ROWS["questions"], _ROWS["contexts"],
        _ROWS["generated_texts"], _ROWS["reference_answers"],
    )
    for i, (s, b) in enumerate(zip(seq, bat)):
        assert b == s, f"row {i} diverged under NLI failure"
    # Non-abstention rows carry the error token; abstention rows stay clean.
    assert "nli:error:RuntimeError" in bat[0].instrument_status
    assert bat[2].instrument_status == "ok"


def test_strict_batched_call_failure_raises_typed() -> None:
    ev = QualityEvaluator(
        use_nli=True, use_embeddings=False, use_bertscore=False,
        use_rouge=False, use_lettucedetect=False, strict=True,
    )
    ev._nli_model = _ExplodingNLI()
    with pytest.raises(InstrumentUnavailableError):
        ev.batch_evaluate(
            _ROWS["questions"], _ROWS["contexts"],
            _ROWS["generated_texts"], _ROWS["reference_answers"],
        )


def test_batch_evaluate_empty_input() -> None:
    ev = _make_evaluator()
    assert ev.batch_evaluate([], [], [], []) == []
