"""F9/#147 instrument revision provenance (quality-module review 2026-08-19).

The four in-process HF checkpoints (NLI / embedding / BERTScore /
LettuceDetect) are pinned by repo NAME only, so a silent upstream repo update
changes the instrument under the same provenance id. QualityEvaluator now
best-effort captures the RESOLVED HF commit hash at each lazy load (exposed
via ``instrument_provenance()``) and, when a CAGE_*_REVISION env pin is set,
treats a mismatching -- or unresolvable -- revision as a standard fail-closed
LOAD failure (strict raises ``InstrumentUnavailableError``; non-strict rows
carry the usual 'X:unavailable:<model>' token). With the pins unset the
capture is record-only: scored behavior is byte-identical.

All model stacks are stubbed via ``sys.modules`` (patterns from
test_quality_failclosed.py / test_quality_claim_checker.py) so no downloads
occur; the stubs exercise the REAL lazy-load paths inside QualityEvaluator.
"""
from __future__ import annotations

import os
import sys
import types
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.quality import (  # noqa: E402
    InstrumentUnavailableError,
    QualityEvaluator,
)

QUESTION = "In what country is Normandy located?"
CONTEXT = ["Normandy is a region in France, settled by Norse raiders."]
ANSWER = "Normandy is located in France."
REFERENCE = "France"

SHA_A = "a" * 40
SHA_B = "b" * 40

PIN_ENVS = (
    "CAGE_NLI_REVISION",
    "CAGE_EMBEDDING_REVISION",
    "CAGE_BERTSCORE_REVISION",
    "CAGE_LETTUCEDETECT_REVISION",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pins unset by default: every test states its own pin explicitly."""
    for var in PIN_ENVS + (
        "CAGE_QUALITY_STRICT", "CAGE_CLAIM_CHECKER",
        "CAGE_DISABLE_LETTUCEDETECT", "CAGE_NLI_THREE_CLASS",
        "CAGE_BERTSCORE_IDF",
    ):
        monkeypatch.delenv(var, raising=False)


def _evaluator(**overrides: Any) -> QualityEvaluator:
    """All instruments OFF unless a test switches one on (NLI path pinned)."""
    kwargs: Dict[str, Any] = dict(
        use_nli=False, use_embeddings=False, use_bertscore=False,
        use_rouge=False, use_lettucedetect=False, claim_checker="nli",
    )
    kwargs.update(overrides)
    return QualityEvaluator(**kwargs)


# --------------------------------------------------------------------------- #
# sys.modules stubs -- WORKING loads whose configs carry a chosen commit hash
# --------------------------------------------------------------------------- #
class _Cfg:
    def __init__(self, commit: Optional[str]) -> None:
        if commit is not None:
            self._commit_hash = commit


class _HFModel:
    """Transformers-model shape: the resolved revision rides on .config."""

    def __init__(self, commit: Optional[str]) -> None:
        self.config = _Cfg(commit)


def _stub_transformers_ok(
    monkeypatch: pytest.MonkeyPatch, commit: Optional[str]
) -> Dict[str, Any]:
    """Working NLI pipeline: entailment high iff the claim appears in the doc."""
    record: Dict[str, Any] = {"requested": []}

    class _Pipe:
        def __init__(self) -> None:
            self.model = _HFModel(commit)
            self.tokenizer = None

        def __call__(self, inputs: Any, top_k: Any = None, truncation: bool = True,
                     max_length: int = 512, **kw: Any) -> Any:
            s = (
                0.93
                if inputs["text_pair"].lower().rstrip(".") in inputs["text"].lower()
                else 0.04
            )
            return [
                {"label": "entailment", "score": s},
                {"label": "neutral", "score": 1.0 - s - 0.01},
                {"label": "contradiction", "score": 0.01},
            ]

    def pipeline(task: str, model: str | None = None, **kw: Any) -> Any:
        record["requested"].append(model)
        return _Pipe()

    mod = types.ModuleType("transformers")
    mod.pipeline = pipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", mod)
    return record


def _stub_sentence_transformers(
    monkeypatch: pytest.MonkeyPatch, commit: Optional[str]
) -> None:
    class _FirstModule:
        def __init__(self) -> None:
            self.auto_model = _HFModel(commit)

    class SentenceTransformer:
        def __init__(self, name: str, device: Any = None, **kw: Any) -> None:
            self._first = _FirstModule()

        def _first_module(self) -> Any:
            return self._first

    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = SentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)


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


def _stub_bert_score_ok(
    monkeypatch: pytest.MonkeyPatch, commit: Optional[str]
) -> None:
    class BERTScorer:
        def __init__(self, model_type: str | None = None, **kw: Any) -> None:
            self._model = _HFModel(commit)

        def score(self, cands: List[str], refs: List[str]) -> Any:
            vals = [0.5] * len(cands)
            return _Vec(vals), _Vec(vals), _Vec(vals)

    mod = types.ModuleType("bert_score")
    mod.BERTScorer = BERTScorer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bert_score", mod)


def _stub_lettucedetect_ok(
    monkeypatch: pytest.MonkeyPatch, commit: Optional[str]
) -> None:
    class HallucinationDetector:
        def __init__(self, method: str, model_path: str, device: str) -> None:
            self.detector = types.SimpleNamespace(model=_HFModel(commit))

        def predict(self, **kw: Any) -> List[Dict[str, Any]]:
            return []

    root = types.ModuleType("lettucedetect")
    models = types.ModuleType("lettucedetect.models")
    inference = types.ModuleType("lettucedetect.models.inference")
    inference.HallucinationDetector = HallucinationDetector  # type: ignore[attr-defined]
    models.inference = inference  # type: ignore[attr-defined]
    root.models = models  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lettucedetect", root)
    monkeypatch.setitem(sys.modules, "lettucedetect.models", models)
    monkeypatch.setitem(sys.modules, "lettucedetect.models.inference", inference)


# --------------------------------------------------------------------------- #
# instrument_provenance(): shape + capture at each REAL lazy-load path
# --------------------------------------------------------------------------- #
def test_provenance_mapping_shape_before_any_load() -> None:
    ev = _evaluator()
    prov = ev.instrument_provenance()
    assert set(prov) == {"nli", "embedding", "bertscore", "lettucedetect"}
    assert prov["nli"] == {"model": ev.nli_model_name, "revision": None}
    assert prov["embedding"] == {"model": ev.embedding_model_name, "revision": None}
    assert prov["bertscore"] == {"model": ev.bertscore_model_name, "revision": None}
    assert prov["lettucedetect"] == {
        "model": ev.lettucedetect_model_name, "revision": None,
    }


def test_revision_captured_nli(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_transformers_ok(monkeypatch, SHA_A)
    ev = _evaluator(use_nli=True, strict=True)
    assert ev.nli_model is not None  # real lazy load through the stub
    assert ev.instrument_provenance()["nli"]["revision"] == SHA_A


def test_revision_captured_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_sentence_transformers(monkeypatch, SHA_A)
    ev = _evaluator(use_embeddings=True, strict=True)
    assert ev.embedding_model is not None
    assert ev.instrument_provenance()["embedding"]["revision"] == SHA_A


def test_revision_captured_bertscore(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_bert_score_ok(monkeypatch, SHA_A)
    ev = _evaluator(use_bertscore=True, strict=True)
    assert ev.bertscore_model is not None
    assert ev.instrument_provenance()["bertscore"]["revision"] == SHA_A


def test_revision_captured_lettucedetect(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_lettucedetect_ok(monkeypatch, SHA_A)
    ev = _evaluator(use_lettucedetect=True, strict=True)
    assert ev.lettucedetect_model is not None
    assert ev.instrument_provenance()["lettucedetect"]["revision"] == SHA_A


def test_unresolvable_revision_records_none_when_unpinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No pin set: an unresolvable hash is honest None, never a failure."""
    _stub_transformers_ok(monkeypatch, commit=None)
    ev = _evaluator(use_nli=True, strict=True)
    r = ev.evaluate_faithfulness(ANSWER, CONTEXT)
    assert r["faithfulness"] is not None  # scored normally
    assert ev.instrument_provenance()["nli"]["revision"] is None


# --------------------------------------------------------------------------- #
# Unset pin envs: record-only, scored behavior unchanged
# --------------------------------------------------------------------------- #
def test_unset_env_scored_row_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_transformers_ok(monkeypatch, SHA_A)
    ev = _evaluator(use_nli=True, strict=True)
    m = ev.evaluate(QUESTION, CONTEXT, "Normandy is a region in France.", REFERENCE)
    assert m.faithfulness is not None
    assert m.instrument_status == "ok"
    d = m.to_dict()
    # HARD CONSTRAINT: no new row columns -- revision provenance is manifest
    # metadata (instrument_provenance()), never a to_dict() key.
    assert not any("revision" in k for k in d)
    # The §8.1 instrument-id string formats are untouched.
    assert m.faithfulness_instrument.startswith(f"{ev.nli_model_name}@transformers-")


# --------------------------------------------------------------------------- #
# Pin set + match: loads and scores normally
# --------------------------------------------------------------------------- #
def test_pin_match_scores_normally(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAGE_NLI_REVISION", SHA_A)
    _stub_transformers_ok(monkeypatch, SHA_A)
    ev = _evaluator(use_nli=True, strict=True)
    r = ev.evaluate_faithfulness(ANSWER, CONTEXT)
    assert r["faithfulness"] is not None
    assert "nli" not in ev._instrument_unavailable
    assert ev.instrument_provenance()["nli"]["revision"] == SHA_A


# --------------------------------------------------------------------------- #
# Pin set + mismatch: standard fail-closed LOAD-failure machinery
# --------------------------------------------------------------------------- #
def test_pin_mismatch_strict_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAGE_NLI_REVISION", SHA_B)
    _stub_transformers_ok(monkeypatch, SHA_A)
    ev = _evaluator(use_nli=True, strict=True)
    with pytest.raises(InstrumentUnavailableError) as ei:
        ev.evaluate_faithfulness(ANSWER, CONTEXT)
    assert ei.value.instrument == "nli"
    assert ei.value.model == ev.nli_model_name
    assert "CAGE_NLI_REVISION" in ei.value.cause
    assert SHA_B in ei.value.cause and SHA_A in ei.value.cause


def test_pin_mismatch_nonstrict_labels_row_and_sticks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAGE_NLI_REVISION", SHA_B)
    record = _stub_transformers_ok(monkeypatch, SHA_A)
    ev = _evaluator(use_nli=True, strict=False)
    m = ev.evaluate(QUESTION, CONTEXT, ANSWER, REFERENCE)
    assert m.faithfulness is None
    assert m.instrument_status == f"nli:unavailable:{ev.nli_model_name}"
    # The mismatching load is DISCARDED: never score with an unpinned model.
    assert ev.nli_model is None
    m2 = ev.evaluate(QUESTION, CONTEXT, ANSWER, REFERENCE)
    assert f"nli:unavailable:{ev.nli_model_name}" in m2.instrument_status
    assert record["requested"] == [ev.nli_model_name]  # sticky: one load attempt


def test_embedding_pin_mismatch_strict_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAGE_EMBEDDING_REVISION", SHA_B)
    _stub_sentence_transformers(monkeypatch, SHA_A)
    ev = _evaluator(use_embeddings=True, strict=True)
    with pytest.raises(InstrumentUnavailableError) as ei:
        _ = ev.embedding_model
    assert ei.value.instrument == "embedding"
    assert "CAGE_EMBEDDING_REVISION" in ei.value.cause


def test_lettucedetect_pin_mismatch_strict_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAGE_LETTUCEDETECT_REVISION", SHA_B)
    _stub_lettucedetect_ok(monkeypatch, SHA_A)
    ev = _evaluator(use_lettucedetect=True, strict=True)
    with pytest.raises(InstrumentUnavailableError) as ei:
        ev.evaluate(QUESTION, CONTEXT, ANSWER, REFERENCE)
    assert ei.value.instrument == "lettucedetect"
    assert "CAGE_LETTUCEDETECT_REVISION" in ei.value.cause


def test_bertscore_pin_mismatch_nonstrict_labels_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAGE_BERTSCORE_REVISION", SHA_B)
    _stub_bert_score_ok(monkeypatch, SHA_A)
    ev = _evaluator(use_bertscore=True, strict=False)
    results = ev.evaluate_completeness(ANSWER, REFERENCE)
    assert results["bertscore_f1"] is None
    assert (
        f"bertscore:unavailable:{ev.bertscore_model_name}" in ev._row_status_tokens
    )
    assert ev.bertscore_model is None  # mismatching load discarded


# --------------------------------------------------------------------------- #
# Pin set + unresolvable revision: fails closed (never scores unpinned)
# --------------------------------------------------------------------------- #
def test_pin_unresolvable_strict_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAGE_NLI_REVISION", SHA_B)
    _stub_transformers_ok(monkeypatch, commit=None)
    ev = _evaluator(use_nli=True, strict=True)
    with pytest.raises(InstrumentUnavailableError) as ei:
        ev.evaluate_faithfulness(ANSWER, CONTEXT)
    assert ei.value.instrument == "nli"
    assert "could not be resolved" in ei.value.cause


def test_pin_unresolvable_nonstrict_labels_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAGE_BERTSCORE_REVISION", SHA_B)
    _stub_bert_score_ok(monkeypatch, commit=None)
    ev = _evaluator(use_bertscore=True, strict=False)
    m = ev.evaluate(QUESTION, CONTEXT, ANSWER, REFERENCE)
    assert m.completeness_bertscore is None
    assert f"bertscore:unavailable:{ev.bertscore_model_name}" in m.instrument_status
