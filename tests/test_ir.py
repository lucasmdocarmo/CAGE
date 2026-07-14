"""Tests for IR (Information Retrieval) utilities.

These are unit-level tests that avoid building large FAISS indexes.
"""

import pytest

np = pytest.importorskip("numpy")

from src.orchestration.ir import build_corpus_from_contexts, retrieval_hit_rate, stable_text_id
from src.data.loader import CAGExample
from src.utils.prompting import format_qa_prompt


def test_stable_text_id_deterministic():
    a = stable_text_id("hello")
    b = stable_text_id("hello")
    c = stable_text_id("hello!")
    assert a == b
    assert a != c


def test_build_corpus_from_contexts_deduplicates():
    ex1 = CAGExample(
        id="1",
        question="q1",
        context=["doc a", "doc b"],
        answer="a",
        metadata={},
    )
    ex2 = CAGExample(
        id="2",
        question="q2",
        context=["doc a", "doc c"],
        answer="b",
        metadata={},
    )

    docs = build_corpus_from_contexts([ex1, ex2], dataset_name="unit")
    texts = sorted([d.text for d in docs])

    assert texts == ["doc a", "doc b", "doc c"]


def test_retrieval_hit_rate():
    gold = ["a", "b"]
    assert retrieval_hit_rate(gold_doc_ids=gold, retrieved_doc_ids=["x", "y"]) == 0.0
    assert retrieval_hit_rate(gold_doc_ids=gold, retrieved_doc_ids=["x", "b"]) == 1.0


def test_retrieval_rank_of_gold():
    # Graded companion (fix #5-C): returns the 1-based rank of the first gold match, or
    # None on a miss / when gold is unknown. Powers MRR = mean(1/rank) downstream.
    from src.orchestration.ir import retrieval_rank_of_gold

    gold = ["a", "b"]
    assert retrieval_rank_of_gold(gold_doc_ids=gold, retrieved_doc_ids=["a", "x"]) == 1
    assert retrieval_rank_of_gold(gold_doc_ids=gold, retrieved_doc_ids=["x", "b", "a"]) == 2
    assert retrieval_rank_of_gold(gold_doc_ids=gold, retrieved_doc_ids=["x", "y"]) is None  # miss
    assert retrieval_rank_of_gold(gold_doc_ids=[], retrieved_doc_ids=["x"]) is None  # gold unknown
    # Text fallback preserves order when ids do not match.
    assert (
        retrieval_rank_of_gold(
            gold_doc_ids=["zzz"],
            retrieved_doc_ids=["p", "q"],
            gold_texts=["the sky is blue"],
            retrieved_texts=["grass is green", "the sky is blue"],
        )
        == 2
    )


def test_format_qa_prompt_contains_context_and_question():
    prompt = format_qa_prompt("What?", ["ctx1", "ctx2"], system_prefix="SYS\n")
    assert "SYS" in prompt
    assert "Context 1:" in prompt
    assert "ctx1" in prompt
    assert "Question: What?" in prompt
    assert prompt.rstrip().endswith("Answer:")
