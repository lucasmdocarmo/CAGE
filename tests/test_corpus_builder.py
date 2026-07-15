"""Unit tests for the shared true-CAG corpus builder (src/data/corpus.py).

Torch-free by design: the builder takes an injected token counter, so these
tests use plain word count and run in the lean analysis venv.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.data.corpus import (
    DEFAULT_HEADER,
    CorpusBlock,
    build_corpus_block,
    default_token_counter,
)


@dataclass
class FakeExample:
    """Shape-compatible with src.data.loader.CAGExample (only .id/.context read)."""

    id: str
    question: str
    context: List[str]
    answer: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def wc(text: str) -> int:
    """Fake token counter: plain word count."""
    return len(text.split())


def make_example(ex_id: str, paragraph: str) -> FakeExample:
    return FakeExample(id=ex_id, question=f"q-{ex_id}", context=[paragraph], answer="a")


P1 = "alpha " * 10
P2 = "bravo " * 10
P3 = "charlie " * 10


def block_size(paragraphs: List[str]) -> int:
    """Word count of a block assembled from exactly these paragraphs."""
    return wc(build_corpus_block(
        [make_example(f"tmp{i}", p) for i, p in enumerate(paragraphs)],
        token_budget=10**9,
        count_tokens=wc,
    ).text)


class TestBudget:
    def test_budget_never_exceeded(self):
        examples = [make_example(f"e{i}", f"word{i} " * 15) for i in range(20)]
        budget = 80
        block = build_corpus_block(examples, token_budget=budget, count_tokens=wc)
        assert wc(block.text) <= budget
        assert block.token_count <= budget
        assert block.token_count == wc(block.text)
        assert 0 < len(block.paragraphs) < 20

    def test_budget_counts_header_and_formatting(self):
        # Budget below even header + one formatted document -> nothing fits.
        block = build_corpus_block([make_example("e1", P1)], token_budget=5, count_tokens=wc)
        assert block.paragraphs == []
        assert block.example_ids == []
        assert block.text == DEFAULT_HEADER

    def test_exact_fit_is_included(self):
        budget = block_size([P1, P2])
        block = build_corpus_block(
            [make_example("e1", P1), make_example("e2", P2)],
            token_budget=budget,
            count_tokens=wc,
        )
        assert block.paragraphs == [P1, P2]
        assert block.token_count == budget


class TestDedupe:
    def test_shared_paragraph_added_once_both_ids_kept(self):
        examples = [make_example("e1", P1), make_example("e2", P1)]
        block = build_corpus_block(examples, token_budget=10**9, count_tokens=wc)
        assert block.paragraphs == [P1]
        assert block.text.count("Document 1:") == 1
        assert "Document 2:" not in block.text
        assert block.example_ids == ["e1", "e2"]

    def test_five_questions_one_paragraph(self):
        examples = [make_example(f"e{i}", P1) for i in range(5)]
        block = build_corpus_block(examples, token_budget=10**9, count_tokens=wc)
        assert block.paragraphs == [P1]
        assert block.example_ids == [f"e{i}" for i in range(5)]

    def test_late_duplicate_after_budget_stop_still_counted(self):
        # e2's paragraph does not fit (budget stop), but e3 duplicates e1's
        # paragraph which IS in the block -> e3 counts.
        budget = block_size([P1])
        examples = [make_example("e1", P1), make_example("e2", P2), make_example("e3", P1)]
        block = build_corpus_block(examples, token_budget=budget, count_tokens=wc)
        assert block.paragraphs == [P1]
        assert block.example_ids == ["e1", "e3"]


class TestExclusion:
    def test_example_whose_paragraph_did_not_fit_is_excluded(self):
        budget = block_size([P1, P2])
        examples = [make_example("e1", P1), make_example("e2", P2), make_example("e3", P3)]
        block = build_corpus_block(examples, token_budget=budget, count_tokens=wc)
        assert block.example_ids == ["e1", "e2"]
        assert "e3" not in block.example_ids
        assert P3.strip() not in block.text

    def test_prefix_semantics_no_skip_and_fill(self):
        # Big paragraph stops additions; a later SMALL new paragraph must NOT
        # be back-filled (deterministic prefix).
        small = "tiny " * 3
        budget = block_size([P1]) + 2  # room for a tiny doc, not for P2
        examples = [make_example("e1", P1), make_example("e2", P2), make_example("e3", small)]
        block = build_corpus_block(examples, token_budget=budget, count_tokens=wc)
        assert block.paragraphs == [P1]
        assert block.example_ids == ["e1"]

    def test_example_with_empty_context_skipped(self):
        examples = [
            FakeExample(id="empty", question="q", context=[], answer="a"),
            make_example("e1", P1),
        ]
        block = build_corpus_block(examples, token_budget=10**9, count_tokens=wc)
        assert block.example_ids == ["e1"]


class TestDeterminism:
    def test_identical_inputs_identical_block(self):
        examples = [make_example(f"e{i}", f"word{i} " * 12) for i in range(10)]
        a = build_corpus_block(examples, token_budget=90, count_tokens=wc)
        b = build_corpus_block(examples, token_budget=90, count_tokens=wc)
        assert a == b
        assert isinstance(a, CorpusBlock)

    def test_order_follows_input_order(self):
        examples = [make_example("e1", P1), make_example("e2", P2), make_example("e3", P3)]
        block = build_corpus_block(examples, token_budget=10**9, count_tokens=wc)
        assert block.paragraphs == [P1, P2, P3]
        assert block.text.find(P1.strip()) < block.text.find(P2.strip()) < block.text.find(P3.strip())


class TestFormat:
    def test_header_and_document_numbering(self):
        examples = [make_example("e1", P1), make_example("e2", P2)]
        block = build_corpus_block(examples, token_budget=10**9, count_tokens=wc)
        assert block.text.startswith(DEFAULT_HEADER)
        assert f"Document 1:\n{P1}" in block.text
        assert f"Document 2:\n{P2}" in block.text
        expected = (
            f"{DEFAULT_HEADER}\n\nDocument 1:\n{P1}\n\nDocument 2:\n{P2}"
        )
        assert block.text == expected

    def test_custom_header(self):
        block = build_corpus_block(
            [make_example("e1", P1)],
            token_budget=10**9,
            count_tokens=wc,
            header="Reference corpus:",
        )
        assert block.text.startswith("Reference corpus:")

    def test_default_counter_heuristic(self):
        block = build_corpus_block([make_example("e1", P1)], token_budget=10**9)
        assert block.token_count == default_token_counter(block.text)
        assert default_token_counter("one two three") == 4  # 3 * 4 // 3
