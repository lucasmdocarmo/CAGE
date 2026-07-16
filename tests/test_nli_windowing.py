"""B2 NLI premise windowing (2026-07-16 audit).

cag_true's ~2.8k-token concatenated corpus block, passed whole as the NLI premise, is
truncated to the first ``nli_max_length`` tokens -- so evidence past the truncation
horizon can NEVER entail a claim (faithfulness collapsed to 0.107 even for in-window
evidence). The fix shortens the PREMISE: docs longer than 400 tokens are split into
sentence-aligned <=400-token windows with ~50% overlap, scored MAX over windows and
docs, and rows are tagged with faithfulness_premise_mode ('direct'|'windowed').

These tests use a FAKE NLI callable (with a whitespace fake tokenizer) that emulates
the real pipeline's truncation: it only "sees" the first ``max_length`` premise tokens.
Evidence planted past the 512-token horizon therefore scores ~0 under the old
whole-doc premise and high under windowed premises -- exactly the audit failure mode.
"""
from __future__ import annotations

import os
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.quality import QualityEvaluator  # noqa: E402

EVIDENCE = "The secret launch code is 8241."


class _FakeTokenizer:
    """Whitespace tokenizer with the HF call convention ({'input_ids': [...]})"""

    def __call__(self, text, add_special_tokens=False, **kwargs):
        if isinstance(text, list):
            return {"input_ids": [[0] * len(t.split()) for t in text]}
        return {"input_ids": [0] * len(text.split())}


class _FakeNLI:
    """Truncation-faithful fake NLI pipeline.

    Entailment is high iff the evidence sentence is VISIBLE within the first
    ``max_length`` whitespace tokens of the premise -- mimicking how the real
    pipeline truncates the premise before the model ever sees the evidence.
    """

    def __init__(self):
        self.tokenizer = _FakeTokenizer()
        self.calls: List[dict] = []

    def __call__(self, inputs, top_k=None, truncation=True, max_length=512):
        self.calls.append({"premise": inputs["text"], "max_length": max_length})
        visible = " ".join(inputs["text"].split()[:max_length])
        score = 0.97 if EVIDENCE.lower() in visible.lower() else 0.02
        return [
            {"label": "entailment", "score": score},
            {"label": "neutral", "score": 1.0 - score - 0.01},
            {"label": "contradiction", "score": 0.01},
        ]


def _evaluator_with_fake_nli() -> tuple[QualityEvaluator, _FakeNLI]:
    ev = QualityEvaluator(
        use_nli=True, use_embeddings=False, use_bertscore=False,
        use_rouge=False, use_lettucedetect=False,
    )
    fake = _FakeNLI()
    ev._nli_model = fake  # bypass lazy HF loading; property returns this directly
    return ev, fake


def _long_doc_with_late_evidence(n_filler_sentences: int = 120) -> str:
    # 120 filler sentences x ~6 words each = ~720 tokens; evidence sits past the
    # 512-token truncation horizon of the whole-doc premise.
    filler = " ".join(
        f"Filler sentence number {i} says nothing useful." for i in range(n_filler_sentences)
    )
    return f"{filler} {EVIDENCE}"


# --------------------------------------------------------------------------- #
# The audit failure mode: evidence past 512 tokens
# --------------------------------------------------------------------------- #
def test_whole_doc_premise_misses_late_evidence() -> None:
    # Sanity: the OLD behaviour (single truncated premise) cannot see the evidence.
    ev, _ = _evaluator_with_fake_nli()
    p = ev._nli_entailment_prob(_long_doc_with_late_evidence(), EVIDENCE)
    assert p is not None and p < 0.1


def test_windowed_premise_recovers_late_evidence() -> None:
    ev, _ = _evaluator_with_fake_nli()
    r = ev.evaluate_faithfulness(EVIDENCE, [_long_doc_with_late_evidence()])
    assert r["premise_mode"] == "windowed"
    assert r["faithfulness"] is not None and r["faithfulness"] >= 0.9
    assert r["supported_claim_ratio"] == 1.0


def test_short_context_stays_direct() -> None:
    ev, fake = _evaluator_with_fake_nli()
    r = ev.evaluate_faithfulness(EVIDENCE, [f"Intro sentence. {EVIDENCE}"])
    assert r["premise_mode"] == "direct"
    assert r["faithfulness"] is not None and r["faithfulness"] >= 0.9
    # A short doc must be passed whole (one premise call per claim x doc).
    assert all(len(c["premise"].split()) <= 400 for c in fake.calls)


def test_max_over_docs_preserved() -> None:
    # Evidence in the SECOND (short) doc: max-over-docs must still find it even
    # when the first doc is long and windowed.
    ev, _ = _evaluator_with_fake_nli()
    long_irrelevant = " ".join(
        f"Unrelated sentence number {i} about weather patterns." for i in range(150)
    )
    r = ev.evaluate_faithfulness(EVIDENCE, [long_irrelevant, EVIDENCE])
    assert r["faithfulness"] is not None and r["faithfulness"] >= 0.9
    assert r["premise_mode"] == "windowed"  # doc 1 was windowed


# --------------------------------------------------------------------------- #
# Window construction: sentence alignment, size cap, ~50% overlap
# --------------------------------------------------------------------------- #
def test_windows_respect_token_cap_and_overlap() -> None:
    ev, _ = _evaluator_with_fake_nli()
    doc = " ".join(
        f"Sentence number {i} contains exactly eight useful words here." for i in range(120)
    )  # ~1080 whitespace tokens
    windows = ev._split_premise_windows(doc)
    assert len(windows) >= 2
    for w in windows:
        assert len(w.split()) <= QualityEvaluator.NLI_PREMISE_WINDOW_TOKENS
    # ~50% overlap: consecutive windows share sentences.
    for a, b in zip(windows, windows[1:]):
        a_sents = set(a.split(". "))
        b_sents = set(b.split(". "))
        assert a_sents & b_sents, "consecutive windows must overlap"
    # Windowing is sentence-aligned: no window starts/ends mid-sentence.
    for w in windows:
        assert w.startswith("Sentence number")
        assert w.rstrip().endswith(("here.", "here"))
    # Nothing lost: every sentence appears in some window.
    joined = " ".join(windows)
    for i in range(120):
        assert f"Sentence number {i} " in joined + " "


def test_windowing_tokenizes_each_sentence_once() -> None:
    # The tokenizer must be called ONCE (batched) per doc, not per window.
    ev, _ = _evaluator_with_fake_nli()
    calls = {"n": 0}
    real_tok = _FakeTokenizer()

    class CountingTok:
        def __call__(self, text, add_special_tokens=False, **kw):
            calls["n"] += 1
            return real_tok(text, add_special_tokens=add_special_tokens, **kw)

    ev._nli_model.tokenizer = CountingTok()
    doc = " ".join(f"Padding sentence number {i} with several words added." for i in range(120))
    windows = ev._split_premise_windows(doc)
    assert len(windows) >= 2
    assert calls["n"] == 1


def test_single_oversized_sentence_still_forms_window() -> None:
    ev, _ = _evaluator_with_fake_nli()
    giant = "word " * 900  # one 900-token "sentence", no terminators
    windows = ev._split_premise_windows(giant.strip())
    assert windows == [giant.strip()]  # single sentence: returned whole, truncation caps it


def test_premise_mode_lands_in_quality_metrics() -> None:
    ev, _ = _evaluator_with_fake_nli()
    m = ev.evaluate(
        question="What is the launch code?",
        context=[_long_doc_with_late_evidence()],
        generated_text=EVIDENCE,
        reference_answer="8241",
    )
    assert m.faithfulness_premise_mode == "windowed"
    assert m.to_dict()["faithfulness_premise_mode"] == "windowed"
    assert m.faithfulness is not None and m.faithfulness >= 0.9
