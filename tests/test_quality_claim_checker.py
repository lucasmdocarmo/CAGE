"""D8 §8.5 Instrument B seams (MiniCheck / AlignScore) + claim-decomposer seam.

The charter demotes generic DeBERTa-MNLI to "legacy fallback only" and
pre-registers a TRUE-selected trained claim checker (MiniCheck vs AlignScore,
decided at calibration). These tests prove the SEAMS: selection via
CAGE_CLAIM_CHECKER whose DEFAULT ('nli') leaves the scored behavior untouched,
lazy fail-closed loading (absent package / unpinned checkpoint raises
InstrumentUnavailableError -- never a silent skip), and the ClaimDecomposer
protocol whose default is the historical sentence splitter.

Packages are stubbed via sys.modules (no dependencies added here -- pins are
the rigor builder's handoff).
"""
from __future__ import annotations

import os
import sys
import types
from typing import Any, Dict, List, Tuple

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.quality import (  # noqa: E402
    AlignScoreClaimChecker,
    InstrumentUnavailableError,
    MiniCheckClaimChecker,
    QualityEvaluator,
    SentenceClaimDecomposer,
)

QUESTION = "In what country is Normandy located?"
CONTEXT = ["Normandy is a region in France. It was settled by Norse raiders."]
ANSWER = "Normandy is a region in France."
REFERENCE = "France"


def _evaluator(**overrides: Any) -> QualityEvaluator:
    kwargs: Dict[str, Any] = dict(
        use_nli=True, use_embeddings=False, use_bertscore=False,
        use_rouge=False, use_lettucedetect=False, strict=True,
    )
    kwargs.update(overrides)
    return QualityEvaluator(**kwargs)


# --------------------------------------------------------------------------- #
# sys.modules stubs
# --------------------------------------------------------------------------- #
def _stub_minicheck(monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    """Working minicheck stub: support prob 0.9 iff the claim appears in the doc."""
    record: Dict[str, Any] = {"models": [], "score_calls": []}

    class MiniCheck:
        def __init__(self, model_name: str, **kw: Any) -> None:
            record["models"].append(model_name)

        def score(self, docs: List[str], claims: List[str]) -> Tuple[Any, Any, Any, Any]:
            record["score_calls"].append(len(docs))
            probs = [
                0.9 if c.lower().rstrip(".") in d.lower() else 0.1
                for d, c in zip(docs, claims)
            ]
            labels = [1 if p >= 0.5 else 0 for p in probs]
            return labels, probs, None, None

    root = types.ModuleType("minicheck")
    inner = types.ModuleType("minicheck.minicheck")
    inner.MiniCheck = MiniCheck  # type: ignore[attr-defined]
    root.minicheck = inner  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "minicheck", root)
    monkeypatch.setitem(sys.modules, "minicheck.minicheck", inner)
    return record


def _stub_alignscore(monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    record: Dict[str, Any] = {"ctor": [], "score_calls": []}

    class AlignScore:
        def __init__(self, **kw: Any) -> None:
            record["ctor"].append(dict(kw))

        def score(self, contexts: List[str], claims: List[str]) -> List[float]:
            record["score_calls"].append(len(contexts))
            return [
                0.8 if c.lower().rstrip(".") in d.lower() else 0.2
                for d, c in zip(contexts, claims)
            ]

    mod = types.ModuleType("alignscore")
    mod.AlignScore = AlignScore  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "alignscore", mod)
    return record


def _block_import(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Make ``import name`` raise ImportError (sys.modules None sentinel)."""
    monkeypatch.setitem(sys.modules, name, None)  # type: ignore[arg-type]
    monkeypatch.setitem(sys.modules, f"{name}.{name}", None)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Default: scored behavior unchanged before calibration
# --------------------------------------------------------------------------- #
def test_default_checker_is_nli_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CAGE_CLAIM_CHECKER", raising=False)
    ev = _evaluator()
    assert ev.claim_checker_name == "nli"
    assert ev._claim_checker is None
    assert ev._faithfulness_instrument_id().startswith(f"{ev.nli_model_name}@transformers-")


def test_invalid_checker_name_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAGE_CLAIM_CHECKER", "vibes")
    with pytest.raises(ValueError):
        _evaluator()


# --------------------------------------------------------------------------- #
# MiniCheck seam
# --------------------------------------------------------------------------- #
def test_minicheck_selected_via_env_and_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAGE_CLAIM_CHECKER", "minicheck")
    monkeypatch.delenv("CAGE_MINICHECK_MODEL", raising=False)
    record = _stub_minicheck(monkeypatch)
    ev = _evaluator()
    assert isinstance(ev._claim_checker, MiniCheckClaimChecker)
    r = ev.evaluate_faithfulness(ANSWER, CONTEXT)
    assert r["faithfulness"] is not None and r["faithfulness"] >= 0.85
    assert r["supported_claim_ratio"] == 1.0
    assert r["method"] == "minicheck_claim_max"
    assert r["instrument"].startswith("minicheck:flan-t5-large@minicheck-")
    assert record["models"] == ["flan-t5-large"]
    assert len(record["score_calls"]) == 1  # one score_pairs call for the row


def test_minicheck_method_and_instrument_land_in_quality_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAGE_CLAIM_CHECKER", "minicheck")
    _stub_minicheck(monkeypatch)
    ev = _evaluator()
    m = ev.evaluate(QUESTION, CONTEXT, ANSWER, REFERENCE)
    assert m.faithfulness_method == "minicheck_claim_max"
    assert m.faithfulness_instrument is not None
    assert "minicheck" in m.faithfulness_instrument
    assert m.to_dict()["faithfulness_instrument"] == m.faithfulness_instrument


def test_minicheck_absent_package_strict_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAGE_CLAIM_CHECKER", "minicheck")
    _block_import(monkeypatch, "minicheck")
    ev = _evaluator(strict=True)
    with pytest.raises(InstrumentUnavailableError) as ei:
        ev.evaluate_faithfulness(ANSWER, CONTEXT)
    assert ei.value.instrument == "claim_checker"
    assert "minicheck" in ei.value.model


def test_minicheck_absent_package_nonstrict_labels_and_sticks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAGE_CLAIM_CHECKER", "minicheck")
    _block_import(monkeypatch, "minicheck")
    ev = _evaluator(strict=False)
    m = ev.evaluate(QUESTION, CONTEXT, ANSWER, REFERENCE)
    assert m.faithfulness is None
    assert "claim_checker:unavailable:" in m.instrument_status
    # Sticky: the second row does not retry the import and stays labeled.
    m2 = ev.evaluate(QUESTION, CONTEXT, ANSWER, REFERENCE)
    assert "claim_checker:unavailable:" in m2.instrument_status
    assert "claim_checker" in ev._instrument_unavailable


# --------------------------------------------------------------------------- #
# AlignScore seam
# --------------------------------------------------------------------------- #
def test_alignscore_requires_checkpoint_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAGE_CLAIM_CHECKER", "alignscore")
    monkeypatch.delenv("CAGE_ALIGNSCORE_CKPT", raising=False)
    _stub_alignscore(monkeypatch)  # package present; checkpoint still missing
    ev = _evaluator(strict=True)
    with pytest.raises(InstrumentUnavailableError) as ei:
        ev.evaluate_faithfulness(ANSWER, CONTEXT)
    assert "checkpoint" in ei.value.cause.lower() or "CKPT" in ei.value.cause


def test_alignscore_with_checkpoint_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAGE_CLAIM_CHECKER", "alignscore")
    monkeypatch.setenv("CAGE_ALIGNSCORE_CKPT", "/models/alignscore.ckpt")
    record = _stub_alignscore(monkeypatch)
    ev = _evaluator()
    assert isinstance(ev._claim_checker, AlignScoreClaimChecker)
    r = ev.evaluate_faithfulness(ANSWER, CONTEXT)
    assert r["faithfulness"] is not None and r["faithfulness"] >= 0.75
    assert r["method"] == "alignscore_claim_max"
    ctor = record["ctor"][0]
    assert ctor["ckpt_path"] == "/models/alignscore.ckpt"
    assert ctor["evaluation_mode"] == "nli_sp"


def test_alignscore_absent_package_strict_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAGE_CLAIM_CHECKER", "alignscore")
    monkeypatch.setenv("CAGE_ALIGNSCORE_CKPT", "/models/alignscore.ckpt")
    _block_import(monkeypatch, "alignscore")
    ev = _evaluator(strict=True)
    with pytest.raises(InstrumentUnavailableError) as ei:
        ev.evaluate_faithfulness(ANSWER, CONTEXT)
    assert ei.value.instrument == "claim_checker"


# --------------------------------------------------------------------------- #
# Checker aggregation matches the D8 §8.5 rule (max over premises, mean over claims)
# --------------------------------------------------------------------------- #
def test_checker_claim_max_aggregation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAGE_CLAIM_CHECKER", "minicheck")
    _stub_minicheck(monkeypatch)
    ev = _evaluator()
    # Claim 1 supported by doc 2 only; claim 2 supported nowhere:
    # per-claim max = [0.9, 0.1] -> mean 0.5, supported ratio 0.5.
    r = ev.evaluate_faithfulness(
        "Normandy is in France. The moon is cheese.",
        ["Unrelated paragraph about weather.", "Normandy is in France today."],
    )
    assert r["faithfulness"] == pytest.approx(0.5)
    assert r["supported_claim_ratio"] == 0.5
    assert r["premise_count"] == 2


def test_checker_batched_single_score_pairs_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAGE_CLAIM_CHECKER", "minicheck")
    record = _stub_minicheck(monkeypatch)
    ev = _evaluator()
    out = ev.batch_evaluate(
        [QUESTION, QUESTION],
        [CONTEXT, ["Normandy is in France today. Another sentence."]],
        [ANSWER, "Normandy is in France."],
        [REFERENCE, REFERENCE],
    )
    assert len(out) == 2
    assert all(m.faithfulness is not None for m in out)
    assert all(m.faithfulness_method == "minicheck_claim_max" for m in out)
    # ONE cross-row score_pairs call (real batching through the seam).
    assert len(record["score_calls"]) == 1


def test_checker_batched_equals_sequential(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAGE_CLAIM_CHECKER", "minicheck")
    _stub_minicheck(monkeypatch)
    questions = [QUESTION, "Second?"]
    contexts = [CONTEXT, ["Beta doc mentions Normandy is in France."]]
    gens = [ANSWER, "Normandy is in France. Gamma is false."]
    refs = [REFERENCE, "France"]
    ev_seq = _evaluator()
    seq = [
        ev_seq.evaluate(q, c, g, r)
        for q, c, g, r in zip(questions, contexts, gens, refs)
    ]
    ev_bat = _evaluator()
    bat = ev_bat.batch_evaluate(questions, contexts, gens, refs)
    assert bat == seq


# --------------------------------------------------------------------------- #
# ClaimDecomposer seam
# --------------------------------------------------------------------------- #
class _WordDecomposer:
    """Toy decomposer proving the seam: every word becomes a claim."""

    def __init__(self) -> None:
        self.calls: List[str] = []

    def decompose(self, text: str) -> List[str]:
        self.calls.append(text)
        return [w for w in text.split() if w.strip()]


def test_default_decomposer_matches_sentence_splitter() -> None:
    text = "First claim. Second claim! Third?\nFourth; fifth."
    assert SentenceClaimDecomposer().decompose(text) == QualityEvaluator._split_claims(text)
    assert isinstance(_evaluator().claim_decomposer, SentenceClaimDecomposer)


def test_custom_decomposer_drives_claim_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAGE_CLAIM_CHECKER", "minicheck")
    record = _stub_minicheck(monkeypatch)
    decomp = _WordDecomposer()
    ev = _evaluator(claim_decomposer=decomp)
    ev.evaluate_faithfulness("alpha beta", ["alpha beta gamma."])
    assert decomp.calls == ["alpha beta"]
    # 2 word-claims x 1 premise = 2 pairs in the single score_pairs call.
    assert record["score_calls"] == [2]


def test_custom_decomposer_with_default_nli_path() -> None:
    decomp = _WordDecomposer()
    ev = _evaluator(claim_decomposer=decomp)

    class _FakeNLI:
        tokenizer = None

        def __call__(self, inputs: Any, **kw: Any) -> Any:
            return [
                {"label": "entailment", "score": 0.9},
                {"label": "neutral", "score": 0.09},
                {"label": "contradiction", "score": 0.01},
            ]

    ev._nli_model = _FakeNLI()
    r = ev.evaluate_faithfulness("alpha beta", ["alpha beta gamma."])
    assert decomp.calls == ["alpha beta"]
    assert r["faithfulness"] == pytest.approx(0.9)
