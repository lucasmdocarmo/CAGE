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
        # NLI path under test: pinned since the 2026-08-05 default flip
        # to 'alignscore' (DECISION.md).
        claim_checker="nli",
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
    # Review fix: faithfulness_method must reflect that faithfulness WAS scored here.
    assert m.faithfulness_method == "nli_claim_max"
    assert m.to_dict()["faithfulness_method"] == "nli_claim_max"


# --------------------------------------------------------------------------- #
# Review fix: premise_count provenance (mean-of-per-claim-max is premise-count
# sensitive -- more premises raises the expected max by chance alone, independent
# of true faithfulness; premise_count lets analysis condition on it).
# --------------------------------------------------------------------------- #
def test_premise_count_reported_and_grows_with_context_docs() -> None:
    ev, _ = _evaluator_with_fake_nli()
    r_one_doc = ev.evaluate_faithfulness(EVIDENCE, [f"Intro sentence. {EVIDENCE}"])
    assert r_one_doc["premise_count"] == 1  # short doc: one direct premise

    r_two_docs = ev.evaluate_faithfulness(
        EVIDENCE, [f"Intro sentence. {EVIDENCE}", "Another short unrelated doc."]
    )
    assert r_two_docs["premise_count"] == 2

    windowed_doc = _long_doc_with_late_evidence()
    r_windowed = ev.evaluate_faithfulness(EVIDENCE, [windowed_doc])
    windows = ev._split_premise_windows(windowed_doc)
    assert r_windowed["premise_count"] == len(windows) > 1


def test_premise_count_lands_in_quality_metrics() -> None:
    ev, _ = _evaluator_with_fake_nli()
    m = ev.evaluate(
        question="What is the launch code?",
        context=[_long_doc_with_late_evidence()],
        generated_text=EVIDENCE,
        reference_answer="8241",
    )
    assert m.faithfulness_premise_count is not None and m.faithfulness_premise_count > 1
    assert m.to_dict()["faithfulness_premise_count"] == m.faithfulness_premise_count


# --------------------------------------------------------------------------- #
# D8 §8.5 3-class NLI reporting (CAGE_NLI_THREE_CLASS, default OFF).
# Charter: "contradiction and neutral reported separately (misread evidence vs
# invented claim -- different bugs)". Seam: _parse_nli_result retains the full
# MNLI 3-class distribution (Williams, Nangia & Bowman, NAACL 2018); per-claim
# aggregation is MAX over premises per class (the same rule as entailment --
# 2026-08-04 technical review, quality L3 row), then mean over claims.
# --------------------------------------------------------------------------- #
import pytest  # noqa: E402

THREE_CLASS_COLS = {"faithfulness_contradiction", "faithfulness_neutral"}


class _PairDistNLI:
    """Pair-deterministic 3-class fake accepting BOTH the sequential
    single-dict call and the batched list-of-dicts call (like the real HF
    pipeline). Distributions keyed by marker substrings in premise/claim."""

    #                       (entailment, neutral, contradiction)
    DISTS = {
        "alphadoc": (0.10, 0.15, 0.70),
        "betadoc": (0.60, 0.25, 0.20),
        "claimtwo": (0.30, 0.10, 0.40),
    }
    DEFAULT = (0.90, 0.08, 0.02)

    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()

    def _one(self, item: dict) -> List[dict]:
        # Claim (hypothesis) markers take precedence over premise markers so a
        # test can pin a distribution to one claim regardless of the premise.
        ent, neu, con = self.DEFAULT
        for text in (item["text_pair"].lower(), item["text"].lower()):
            hit = next(
                (dist for marker, dist in self.DISTS.items() if marker in text),
                None,
            )
            if hit is not None:
                ent, neu, con = hit
                break
        return [
            {"label": "entailment", "score": ent},
            {"label": "neutral", "score": neu},
            {"label": "contradiction", "score": con},
        ]

    def __call__(self, inputs, top_k=None, truncation=True, max_length=512,
                 batch_size=None):
        if isinstance(inputs, list):
            return [self._one(x) for x in inputs]
        return self._one(inputs)


def _evaluator(fake=None, **kwargs) -> QualityEvaluator:
    ev = QualityEvaluator(
        use_nli=True, use_embeddings=False, use_bertscore=False,
        use_rouge=False, use_lettucedetect=False,
        # NLI path under test: pinned since the 2026-08-05 default flip
        # to 'alignscore' (DECISION.md).
        claim_checker="nli", **kwargs,
    )
    ev._nli_model = fake if fake is not None else _FakeNLI()
    return ev


def test_three_class_default_off_to_dict_byte_identical(monkeypatch) -> None:
    # THE default-off proof: with the flag unset, to_dict() output is unchanged
    # -- the flag-on output differs by EXACTLY the two new columns and nothing
    # else (same scoring, same keys, same values).
    monkeypatch.delenv("CAGE_NLI_THREE_CLASS", raising=False)
    kwargs = dict(
        question="What is the launch code?",
        context=[f"Intro sentence. {EVIDENCE}"],
        generated_text=EVIDENCE,
        reference_answer="8241",
    )
    d_off = _evaluator().evaluate(**kwargs).to_dict()
    d_on = _evaluator(nli_three_class=True).evaluate(**kwargs).to_dict()
    assert not (THREE_CLASS_COLS & set(d_off))
    assert THREE_CLASS_COLS <= set(d_on)
    assert {k: v for k, v in d_on.items() if k not in THREE_CLASS_COLS} == d_off
    # The faithfulness result dict is equally unchanged when the flag is off.
    r_off = _evaluator().evaluate_faithfulness(EVIDENCE, [f"Intro. {EVIDENCE}"])
    assert "contradiction" not in r_off and "neutral" not in r_off


def test_three_class_env_flag_emits_columns(monkeypatch) -> None:
    monkeypatch.setenv("CAGE_NLI_THREE_CLASS", "1")
    ev = _evaluator()  # env-gated on, no constructor arg
    assert ev.nli_three_class
    m = ev.evaluate(
        question="What is the launch code?",
        context=[f"Intro sentence. {EVIDENCE}"],
        generated_text=EVIDENCE,
        reference_answer="8241",
    )
    # _FakeNLI: entailment 0.97, neutral 1-0.97-0.01=0.02, contradiction 0.01.
    assert m.faithfulness == pytest.approx(0.97)
    assert m.faithfulness_neutral == pytest.approx(0.02)
    assert m.faithfulness_contradiction == pytest.approx(0.01)
    d = m.to_dict()
    assert d["faithfulness_neutral"] == pytest.approx(0.02)
    assert d["faithfulness_contradiction"] == pytest.approx(0.01)


def test_three_class_max_over_premises_then_mean_over_claims() -> None:
    ev = _evaluator(fake=_PairDistNLI(), nli_three_class=True)
    # One claim, two premises: per-class MAX over premises (same rule as
    # entailment). alphadoc contra=0.70 > betadoc contra=0.20, etc.
    r = ev.evaluate_faithfulness(
        "The result is fine.",
        ["Something about alphadoc here.", "Something about betadoc here."],
    )
    assert r["faithfulness"] == pytest.approx(0.60)  # max entailment (betadoc)
    assert r["contradiction"] == pytest.approx(0.70)  # max contra (alphadoc)
    assert r["neutral"] == pytest.approx(0.25)  # max neutral (betadoc)
    # Two claims, one premise each scored differently: mean over claims.
    r2 = ev.evaluate_faithfulness(
        "The result is fine. This one mentions claimtwo.",
        ["Something about betadoc here."],
    )
    # claim 1 -> betadoc dist (0.60, 0.25, 0.20); claim 2 -> claimtwo dist
    # (0.30, 0.10, 0.40); means: ent 0.45, neutral 0.175, contra 0.30.
    assert r2["faithfulness"] == pytest.approx(0.45)
    assert r2["neutral"] == pytest.approx(0.175)
    assert r2["contradiction"] == pytest.approx(0.30)


def test_three_class_batched_matches_sequential() -> None:
    rows = dict(
        questions=["Q1?", "Q2?", "Q3?"],
        contexts=[
            ["Something about alphadoc here.", "Something about betadoc here."],
            ["Something about betadoc here."],
            ["Any context at all."],
        ],
        generated_texts=[
            "The result is fine.",
            "The result is fine. This one mentions claimtwo.",
            "I don't know.",  # abstention: columns present, None
        ],
        reference_answers=["fine", "fine", ""],
    )
    seq_ev = _evaluator(fake=_PairDistNLI(), nli_three_class=True)
    seq = [
        seq_ev.evaluate(q, c, g, ref).to_dict()
        for q, c, g, ref in zip(
            rows["questions"], rows["contexts"],
            rows["generated_texts"], rows["reference_answers"],
        )
    ]
    bat_ev = _evaluator(fake=_PairDistNLI(), nli_three_class=True)
    bat = [
        m.to_dict()
        for m in bat_ev.batch_evaluate(
            rows["questions"], rows["contexts"],
            rows["generated_texts"], rows["reference_answers"],
        )
    ]
    assert bat == seq  # field-for-field, incl. the two new columns
    # Both new columns are STABLE columns on every row of a flagged run;
    # the abstention row carries None (excluded from means downstream).
    for d in bat:
        assert THREE_CLASS_COLS <= set(d)
    assert bat[2]["faithfulness_contradiction"] is None
    assert bat[2]["faithfulness_neutral"] is None
    assert bat[0]["faithfulness_contradiction"] == pytest.approx(0.70)


def test_three_class_label_x_resolution_via_model_config() -> None:
    # Models emitting LABEL_x names resolve neutral/contradiction through
    # id2label, mirroring the entailment resolver (never hardcode indices).
    class _LabelXNLI:
        def __init__(self) -> None:
            self.tokenizer = _FakeTokenizer()

            class _Cfg:
                id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}

            class _Model:
                config = _Cfg()

            self.model = _Model()

        def __call__(self, inputs, top_k=None, truncation=True,
                     max_length=512, batch_size=None):
            out = [
                {"label": "LABEL_0", "score": 0.8},
                {"label": "LABEL_1", "score": 0.15},
                {"label": "LABEL_2", "score": 0.05},
            ]
            if isinstance(inputs, list):
                return [list(out) for _ in inputs]
            return list(out)

    ev = _evaluator(fake=_LabelXNLI(), nli_three_class=True)
    r = ev.evaluate_faithfulness("A claim.", ["Some context."])
    assert r["faithfulness"] == pytest.approx(0.8)
    assert r["neutral"] == pytest.approx(0.15)
    assert r["contradiction"] == pytest.approx(0.05)
