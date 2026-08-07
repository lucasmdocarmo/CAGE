"""D8 §8.1 per-row instrument provenance + D8 §8.4 BERTScore IDF gating.

§8.1: "every score row carries instrument id+version, calibration id, and the
``scored_windowed`` flag". QualityMetrics therefore emits, UNCONDITIONALLY:
``grounding_instrument``, ``faithfulness_instrument``, ``calibration_id``
(None until the D9 calibration exists), and ``scored_windowed``.

§8.4: BERTScore "IDF weighting where references share boilerplate" (Zhang et
al. 2020 §3) -- wired behind CAGE_BERTSCORE_IDF, DEFAULT OFF so the current
scored behavior is unchanged; IDF-on without a reference corpus fails CLOSED.
"""
from __future__ import annotations

import os
import sys
import types
from typing import Any, Dict, List

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.quality import (  # noqa: E402
    InstrumentUnavailableError,
    QualityEvaluator,
    QualityMetrics,
)

QUESTION = "In what country is Normandy located?"
CONTEXT = ["Normandy is a region in France, settled by Norse raiders."]
ANSWER = "Normandy is located in France."
REFERENCE = "France"

PROVENANCE_KEYS = (
    "scored_windowed", "grounding_instrument", "faithfulness_instrument",
    "calibration_id", "bertscore_idf",
)


def _evaluator(**overrides: Any) -> QualityEvaluator:
    # claim_checker='nli' pinned: these tests exercise NLI-path provenance;
    # since the 2026-08-05 default flip to 'alignscore' (DECISION.md) the
    # NLI path must be selected explicitly.
    kwargs: Dict[str, Any] = dict(
        use_nli=False, use_embeddings=False, use_bertscore=False,
        use_rouge=False, use_lettucedetect=False, strict=True,
        claim_checker="nli",
    )
    kwargs.update(overrides)
    return QualityEvaluator(**kwargs)


class _FakeNLI:
    """Deterministic pair scorer: entailment high iff claim appears in premise."""

    def __init__(self) -> None:
        self.tokenizer = None

    def __call__(self, inputs: Any, top_k: Any = None, truncation: bool = True,
                 max_length: int = 512, **kw: Any) -> Any:
        s = 0.93 if inputs["text_pair"].lower().rstrip(".") in inputs["text"].lower() else 0.04
        return [
            {"label": "entailment", "score": s},
            {"label": "neutral", "score": 1.0 - s - 0.01},
            {"label": "contradiction", "score": 0.01},
        ]


class _FakeDetectorOK:
    """Minimal detector with derivable limits that flags nothing."""

    class _Inner:
        class _Tok:
            def __call__(self, texts: Any, add_special_tokens: bool = False, **kw: Any) -> Dict[str, Any]:
                if isinstance(texts, list):
                    return {"input_ids": [[0] * len(t.split()) for t in texts]}
                return {"input_ids": [0] * len(texts.split())}

        class _Model:
            class _Cfg:
                max_position_embeddings = 8192

            config = _Cfg()

        def __init__(self) -> None:
            self.tokenizer = self._Tok()
            self.model = self._Model()

    def __init__(self) -> None:
        self.detector = self._Inner()

    def predict(self, **kw: Any) -> List[Dict[str, Any]]:
        return []


# --------------------------------------------------------------------------- #
# Unconditional emission (stable CSV columns)
# --------------------------------------------------------------------------- #
def test_provenance_keys_emitted_unconditionally_on_bare_row() -> None:
    m = _evaluator().evaluate(QUESTION, CONTEXT, ANSWER, REFERENCE)
    d = m.to_dict()
    for key in PROVENANCE_KEYS:
        assert key in d, f"missing provenance column {key}"
    assert d["scored_windowed"] == 0.0
    assert d["grounding_instrument"] is None
    assert d["faithfulness_instrument"] is None
    assert d["calibration_id"] is None
    assert d["bertscore_idf"] is None


def test_provenance_keys_present_on_default_dataclass() -> None:
    m = QualityMetrics(
        faithfulness=None, relevance=None,
        completeness_bertscore=None, completeness_rouge_l=None,
    )
    d = m.to_dict()
    for key in PROVENANCE_KEYS:
        assert key in d


# --------------------------------------------------------------------------- #
# Instrument id+version threading
# --------------------------------------------------------------------------- #
def test_faithfulness_instrument_set_when_nli_consulted() -> None:
    ev = _evaluator(use_nli=True)
    ev._nli_model = _FakeNLI()
    m = ev.evaluate(QUESTION, CONTEXT, "Normandy is a region in France.", REFERENCE)
    assert m.faithfulness is not None
    assert m.faithfulness_instrument is not None
    assert m.faithfulness_instrument.startswith(f"{ev.nli_model_name}@transformers-")
    assert m.to_dict()["faithfulness_instrument"] == m.faithfulness_instrument
    # Grounding was config-off: its provenance stays None.
    assert m.grounding_instrument is None


def test_grounding_instrument_set_when_detector_consulted() -> None:
    ev = _evaluator(use_lettucedetect=True)
    ev._lettucedetect_model = _FakeDetectorOK()
    m = ev.evaluate(QUESTION, CONTEXT, ANSWER, REFERENCE)
    assert m.grounding_score == 1.0
    assert m.grounding_instrument is not None
    assert m.grounding_instrument.startswith(
        f"{ev.lettucedetect_model_name}@lettucedetect-"
    )
    assert m.faithfulness_instrument is None  # NLI config-off


def test_abstention_row_carries_no_instrument_provenance() -> None:
    ev = _evaluator(use_nli=True, use_lettucedetect=True)
    ev._nli_model = _FakeNLI()
    ev._lettucedetect_model = _FakeDetectorOK()
    m = ev.evaluate(QUESTION, CONTEXT, "I don't know.", REFERENCE)
    assert m.grounding_instrument is None
    assert m.faithfulness_instrument is None
    assert m.scored_windowed is False


# --------------------------------------------------------------------------- #
# calibration_id: None until D9 calibration exists; env/constructor threading
# --------------------------------------------------------------------------- #
def test_calibration_id_defaults_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CAGE_CALIBRATION_ID", raising=False)
    m = _evaluator().evaluate(QUESTION, CONTEXT, ANSWER, REFERENCE)
    assert m.calibration_id is None


def test_calibration_id_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAGE_CALIBRATION_ID", "cal-2026-08-04-a")
    m = _evaluator().evaluate(QUESTION, CONTEXT, ANSWER, REFERENCE)
    assert m.calibration_id == "cal-2026-08-04-a"
    assert m.to_dict()["calibration_id"] == "cal-2026-08-04-a"


def test_calibration_id_constructor_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAGE_CALIBRATION_ID", "cal-env")
    m = _evaluator(calibration_id="cal-ctor").evaluate(QUESTION, CONTEXT, ANSWER, REFERENCE)
    assert m.calibration_id == "cal-ctor"


# --------------------------------------------------------------------------- #
# BERTScore IDF (D8 §8.4): env-gated, default-off, fail-closed without corpus
# --------------------------------------------------------------------------- #
class _RecordingBERTScorer:
    ctor_kwargs: List[Dict[str, Any]] = []

    def __init__(self, **kw: Any) -> None:
        type(self).ctor_kwargs.append(dict(kw))

    def score(self, cands: List[str], refs: List[str]) -> Any:
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

        vals = [0.5] * len(cands)
        return _Vec(vals), _Vec(vals), _Vec(vals)


def _stub_bert_score(monkeypatch: pytest.MonkeyPatch) -> type:
    _RecordingBERTScorer.ctor_kwargs = []
    mod = types.ModuleType("bert_score")
    mod.BERTScorer = _RecordingBERTScorer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bert_score", mod)
    return _RecordingBERTScorer


def test_idf_default_off_constructs_scorer_exactly_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CAGE_BERTSCORE_IDF", raising=False)
    scorer_cls = _stub_bert_score(monkeypatch)
    ev = _evaluator(use_bertscore=True)
    m = ev.evaluate(QUESTION, CONTEXT, ANSWER, REFERENCE)
    assert ev.bertscore_idf is False
    kw = scorer_cls.ctor_kwargs[0]
    # Default behavior preserved: no idf kwargs are even passed.
    assert "idf" not in kw and "idf_sents" not in kw
    assert m.completeness_bertscore == 0.5
    assert m.bertscore_idf is False
    assert m.to_dict()["bertscore_idf"] is False


def test_idf_env_gated_on_passes_idf_and_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAGE_BERTSCORE_IDF", "1")
    scorer_cls = _stub_bert_score(monkeypatch)
    corpus = ["France", "Paris", "the capital of France"]
    ev = _evaluator(use_bertscore=True, bertscore_idf_sents=corpus)
    m = ev.evaluate(QUESTION, CONTEXT, ANSWER, REFERENCE)
    kw = scorer_cls.ctor_kwargs[0]
    assert kw["idf"] is True
    assert kw["idf_sents"] == corpus
    assert m.bertscore_idf is True
    assert m.to_dict()["bertscore_idf"] is True


def test_idf_constructor_arg_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAGE_BERTSCORE_IDF", "1")
    scorer_cls = _stub_bert_score(monkeypatch)
    ev = _evaluator(use_bertscore=True, bertscore_idf=False)
    ev.evaluate(QUESTION, CONTEXT, ANSWER, REFERENCE)
    assert "idf" not in scorer_cls.ctor_kwargs[0]


def test_idf_on_without_corpus_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAGE_BERTSCORE_IDF", "1")
    _stub_bert_score(monkeypatch)
    ev = _evaluator(use_bertscore=True, strict=True)  # no idf_sents
    with pytest.raises(InstrumentUnavailableError) as ei:
        ev.evaluate_completeness(ANSWER, REFERENCE)
    assert ei.value.instrument == "bertscore"
    assert "idf" in ei.value.cause.lower()


def test_idf_on_without_corpus_nonstrict_labels_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAGE_BERTSCORE_IDF", "1")
    _stub_bert_score(monkeypatch)
    ev = _evaluator(use_bertscore=True, strict=False)
    m = ev.evaluate(QUESTION, CONTEXT, ANSWER, REFERENCE)
    assert m.completeness_bertscore is None
    assert m.bertscore_idf is None
    assert f"bertscore:unavailable:{ev.bertscore_model_name}" in m.instrument_status
