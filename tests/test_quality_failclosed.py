"""P0-5 fail-closed instrument behavior (audit 2026-08-02).

A pinned instrument that cannot load or score must NEVER be silently replaced by
a fallback model scoring under the same column name -- that voids the D8/D9
predicate symmetry behind Y (serving yield). Strict mode (the D8 default) raises
``InstrumentUnavailableError``; non-strict mode (long-run harness survival)
records score=None + an ``instrument_status`` token ('X:unavailable:<model>' for
load failures, 'X:error:<ExcType>' for per-row scoring failures).

All model stacks are stubbed via ``sys.modules`` so no downloads occur; the stubs
exercise the REAL lazy-load paths inside QualityEvaluator.
"""
from __future__ import annotations

import os
import sys
import types
from typing import Any, List

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.quality import (  # noqa: E402
    InstrumentUnavailableError,
    QualityEvaluator,
    QualityMetrics,
)

QUESTION = "In what country is Normandy located?"
CONTEXT = ["Normandy is a region in France, settled by Norse raiders under Rollo."]
ANSWER = "Normandy is located in France."
REFERENCE = "France"


def _evaluator(**overrides: Any) -> QualityEvaluator:
    """All instruments OFF unless a test switches one on.

    claim_checker='nli' is pinned: these tests exercise the DeBERTa-MNLI
    NLI path, which since the 2026-08-05 default flip to 'alignscore'
    (DECISION.md) must be selected explicitly.
    """
    kwargs: dict[str, Any] = dict(
        use_nli=False, use_embeddings=False, use_bertscore=False,
        use_rouge=False, use_lettucedetect=False, claim_checker="nli",
    )
    kwargs.update(overrides)
    return QualityEvaluator(**kwargs)


# --------------------------------------------------------------------------- #
# sys.modules stubs -- each records the model names the loader asked for
# --------------------------------------------------------------------------- #
class _FailingPipelineFactory:
    def __init__(self) -> None:
        self.requested: List[str] = []

    def __call__(self, task: str, model: str | None = None, **kw: Any) -> Any:
        self.requested.append(model)
        raise RuntimeError("stub NLI load failure")


def _stub_transformers(monkeypatch: pytest.MonkeyPatch) -> _FailingPipelineFactory:
    factory = _FailingPipelineFactory()
    mod = types.ModuleType("transformers")
    mod.pipeline = factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", mod)
    return factory


def _stub_lettucedetect(monkeypatch: pytest.MonkeyPatch) -> List[str]:
    requested: List[str] = []

    class HallucinationDetector:
        def __init__(self, method: str, model_path: str, device: str) -> None:
            requested.append(model_path)
            raise RuntimeError("stub LettuceDetect load failure")

    root = types.ModuleType("lettucedetect")
    models = types.ModuleType("lettucedetect.models")
    inference = types.ModuleType("lettucedetect.models.inference")
    inference.HallucinationDetector = HallucinationDetector  # type: ignore[attr-defined]
    models.inference = inference  # type: ignore[attr-defined]
    root.models = models  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lettucedetect", root)
    monkeypatch.setitem(sys.modules, "lettucedetect.models", models)
    monkeypatch.setitem(sys.modules, "lettucedetect.models.inference", inference)
    return requested


def _stub_bert_score(monkeypatch: pytest.MonkeyPatch) -> List[str]:
    requested: List[str] = []

    class BERTScorer:
        def __init__(self, model_type: str | None = None, **kw: Any) -> None:
            requested.append(model_type)
            raise RuntimeError("stub BERTScorer load failure")

    mod = types.ModuleType("bert_score")
    mod.BERTScorer = BERTScorer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bert_score", mod)
    return requested


class _ExplodingNLI:
    """Loads fine (injected), then raises on every scoring call."""

    def __call__(self, *a: Any, **kw: Any) -> Any:
        raise RuntimeError("stub NLI scoring failure")


class _ExplodingScorer:
    def score(self, *a: Any, **kw: Any) -> Any:
        raise RuntimeError("stub BERTScore scoring failure")


# --------------------------------------------------------------------------- #
# Strict mode: load failure raises, typed and attributed
# --------------------------------------------------------------------------- #
def test_strict_nli_load_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = _stub_transformers(monkeypatch)
    ev = _evaluator(use_nli=True, strict=True)
    with pytest.raises(InstrumentUnavailableError) as ei:
        ev.evaluate_faithfulness(ANSWER, CONTEXT)
    assert ei.value.instrument == "nli"
    assert ei.value.model == ev.nli_model_name
    assert issubclass(InstrumentUnavailableError, RuntimeError)
    assert factory.requested == [ev.nli_model_name]


def test_strict_lettucedetect_load_failure_raises_via_evaluate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_lettucedetect(monkeypatch)
    ev = _evaluator(use_lettucedetect=True, strict=True)
    with pytest.raises(InstrumentUnavailableError) as ei:
        ev.evaluate(QUESTION, CONTEXT, ANSWER, REFERENCE)
    assert ei.value.instrument == "lettucedetect"


def test_strict_bertscore_load_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    requested = _stub_bert_score(monkeypatch)
    ev = _evaluator(use_bertscore=True, strict=True)
    with pytest.raises(InstrumentUnavailableError) as ei:
        ev.evaluate_completeness(ANSWER, REFERENCE)
    assert ei.value.instrument == "bertscore"
    # Pinned model only -- the old fallback chain must not be walked.
    assert requested == [ev.bertscore_model_name]


def test_strict_runtime_scoring_failure_raises() -> None:
    ev = _evaluator(use_nli=True, strict=True)
    ev._nli_model = _ExplodingNLI()  # bypass lazy load; fail at scoring time
    with pytest.raises(InstrumentUnavailableError) as ei:
        ev.evaluate_faithfulness(ANSWER, CONTEXT)
    assert ei.value.instrument == "nli"
    assert "scoring failed" in ei.value.cause


# --------------------------------------------------------------------------- #
# No silent substitution -- fallbacks are never consulted in ANY mode
# --------------------------------------------------------------------------- #
def test_fallback_env_vars_are_never_consulted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAGE_NLI_FALLBACKS", "facebook/bart-large-mnli")
    factory = _stub_transformers(monkeypatch)
    ev = _evaluator(use_nli=True, strict=False)
    result = ev.evaluate_faithfulness(ANSWER, CONTEXT)
    assert result["faithfulness"] is None
    # Only the pinned model was attempted; the configured fallback never loads.
    assert factory.requested == [ev.nli_model_name]
    assert "facebook/bart-large-mnli" not in factory.requested
    assert ev.nli_model_fallbacks == []  # attribute survives but is inert


def test_nonstrict_load_failure_is_sticky(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = _stub_transformers(monkeypatch)
    ev = _evaluator(use_nli=True, strict=False)
    ev.evaluate(QUESTION, CONTEXT, ANSWER, REFERENCE)
    ev.evaluate(QUESTION, CONTEXT, ANSWER, REFERENCE)
    # One load attempt total: the old code retried the full model load per row.
    assert len(factory.requested) == 1


# --------------------------------------------------------------------------- #
# Non-strict mode: score=None + instrument_status label, row survives
# --------------------------------------------------------------------------- #
def test_nonstrict_nli_unavailable_labels_row(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_transformers(monkeypatch)
    ev = _evaluator(use_nli=True, strict=False)
    m = ev.evaluate(QUESTION, CONTEXT, ANSWER, REFERENCE)
    assert isinstance(m, QualityMetrics)
    assert m.faithfulness is None
    assert m.supported_claim_ratio is None
    assert m.instrument_status == f"nli:unavailable:{ev.nli_model_name}"
    assert m.to_dict()["instrument_status"] == m.instrument_status
    # The row still carries the model-free metrics: the harness survived.
    assert m.f1_score > 0.0


def test_nonstrict_multiple_unavailable_instruments_all_labeled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_lettucedetect(monkeypatch)
    _stub_bert_score(monkeypatch)
    ev = _evaluator(use_lettucedetect=True, use_bertscore=True, strict=False)
    m = ev.evaluate(QUESTION, CONTEXT, ANSWER, REFERENCE)
    assert m.grounding_score is None
    assert m.completeness_bertscore is None
    assert f"lettucedetect:unavailable:{ev.lettucedetect_model_name}" in m.instrument_status
    assert f"bertscore:unavailable:{ev.bertscore_model_name}" in m.instrument_status


def test_nonstrict_runtime_scoring_failure_labels_row() -> None:
    ev = _evaluator(use_nli=True, strict=False)
    ev._nli_model = _ExplodingNLI()
    m = ev.evaluate(QUESTION, CONTEXT, ANSWER, REFERENCE)
    assert m.faithfulness is None
    assert "nli:error:RuntimeError" in m.instrument_status


def test_nonstrict_bertscore_runtime_failure_never_reloads_substitute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested = _stub_bert_score(monkeypatch)
    ev = _evaluator(use_bertscore=True, strict=False)
    ev._bertscore_model = _ExplodingScorer()  # loaded fine, breaks at scoring time
    results = ev.evaluate_completeness(ANSWER, REFERENCE)
    assert results["bertscore_f1"] is None
    # The old code reloaded a FALLBACK model here and kept scoring under the
    # same column. No loader call may happen at all.
    assert requested == []
    assert any(t.startswith("bertscore:error:") for t in ev._row_status_tokens)


# --------------------------------------------------------------------------- #
# Rows that never consult the instrument are clean
# --------------------------------------------------------------------------- #
def test_abstention_row_skips_broken_instruments(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_transformers(monkeypatch)
    _stub_lettucedetect(monkeypatch)
    # Strict + broken NLI and LettuceDetect: an abstention row never consults
    # them (abstention short-circuit), so it must neither raise nor be labeled.
    ev = _evaluator(use_nli=True, use_lettucedetect=True, strict=True)
    m = ev.evaluate(QUESTION, CONTEXT, "I don't know.", REFERENCE)
    assert m.instrument_status == "ok"
    assert m.grounding_score is None  # abstention semantics, not unavailability


def test_config_disabled_is_not_unavailable() -> None:
    # The CAGE_SKIP_QUALITY decoupled-serving config: everything off, strict on.
    # Config-off is a deliberate choice, not an instrument failure -> no raise.
    ev = _evaluator(strict=True)
    m = ev.evaluate(QUESTION, CONTEXT, ANSWER, REFERENCE)
    assert m.instrument_status == "ok"
    assert m.faithfulness is None
    assert m.f1_score > 0.0


# --------------------------------------------------------------------------- #
# strict flag resolution
# --------------------------------------------------------------------------- #
def test_strict_defaults_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CAGE_QUALITY_STRICT", raising=False)
    assert _evaluator().strict is True


def test_strict_env_opt_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAGE_QUALITY_STRICT", "0")
    assert _evaluator().strict is False


def test_strict_param_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAGE_QUALITY_STRICT", "0")
    assert _evaluator(strict=True).strict is True


# --------------------------------------------------------------------------- #
# Labeled diagnostic: cache relevance writes its method into the row
# --------------------------------------------------------------------------- #
def test_cache_relevance_lexical_substitute_is_labeled() -> None:
    ev = _evaluator(strict=True)  # embeddings config-off -> lexical fallback
    res = ev.evaluate_cache_relevance(
        ANSWER, REFERENCE, ["France is a country in Europe.", "Unrelated block."]
    )
    assert res.method == "lexical_jaccard"
    assert res.to_dict()["cache_relevance_method"] == "lexical_jaccard"


def test_cache_relevance_method_propagates_to_quality_row() -> None:
    ev = _evaluator(strict=True)
    m = ev.evaluate_with_cache_relevance(
        QUESTION, CONTEXT, ANSWER, REFERENCE, cache_blocks=["France block."]
    )
    assert m.cache_relevance is not None
    assert m.cache_relevance_method == "lexical_jaccard"
    assert m.to_dict()["cache_relevance_method"] == "lexical_jaccard"


def test_cache_relevance_empty_blocks_method_none() -> None:
    ev = _evaluator(strict=True)
    res = ev.evaluate_cache_relevance(ANSWER, REFERENCE, [])
    assert res.method == "none"
    assert res.cache_relevance == 0.0
