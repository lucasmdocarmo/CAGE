"""A2 (2026-10-10): src.data.loader.gold_paragraphs, the one gold rule.

The runner's retrieval hit, retrieval rank and gold-position fields scored
against the WHOLE example context (HotpotQA/MuSiQue distractors, the whole
Qasper paper, or the corpus block itself). They now share the retrieval gate
table's rule; these tests pin the loader copy equal to the gate table's.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_4A = REPO_ROOT / "scripts" / "4_analysis"
for _p in (str(_SCRIPTS_4A), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_retrieval_gate_table as gate  # noqa: E402
from src.data.loader import CAGExample, gold_paragraphs  # noqa: E402

CASES = {
    "qasper_evidence_wins": CAGExample(
        id="p1_q1", question="q", context=["Intro: a", "Method: b", "Results: c"],
        answer="a", metadata={"evidence_doc_ids": [2], "supporting_titles": ["Intro"]},
    ),
    "qasper_goldless": CAGExample(
        id="p1_q2", question="q", context=["Intro: a", "Method: b"],
        answer="", metadata={"evidence_doc_ids": [], "supporting_titles": []},
    ),
    "qasper_out_of_range": CAGExample(
        id="p1_q3", question="q", context=["Intro: a"],
        answer="a", metadata={"evidence_doc_ids": [0, 5, -1]},
    ),
    "hotpot_titles": CAGExample(
        id="h1", question="q",
        context=["Noise: distractor text", "Gold A: the first fact", "Gold B: the second"],
        answer="a", metadata={"supporting_titles": ["Gold A", "Gold B"]},
    ),
    "hotpot_empty_titles": CAGExample(
        id="h2", question="q", context=["Gold A: x", "Noise: y"],
        answer="", metadata={"supporting_titles": []},
    ),
    "squad": CAGExample(
        id="s1", question="q", context=["The gold paragraph."],
        answer="gold", metadata={"dataset": "squad_v2", "is_impossible": False},
    ),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_loader_rule_equals_the_gate_table_rule(name: str) -> None:
    example = CASES[name]
    assert gold_paragraphs(example) == gate._gold_context_texts(example)


def test_gold_paragraphs_route_by_route() -> None:
    assert gold_paragraphs(CASES["qasper_evidence_wins"]) == ["Results: c"]
    assert gold_paragraphs(CASES["qasper_goldless"]) == []
    assert gold_paragraphs(CASES["qasper_out_of_range"]) == ["Intro: a"]
    # The distractor never counts as gold; both supporting paragraphs do.
    assert gold_paragraphs(CASES["hotpot_titles"]) == [
        "Gold A: the first fact", "Gold B: the second",
    ]
    assert gold_paragraphs(CASES["hotpot_empty_titles"]) == []
    assert gold_paragraphs(CASES["squad"]) == ["The gold paragraph."]


def test_a_bool_evidence_id_is_not_an_index() -> None:
    # True == 1 in Python; an id must be a real int, never a flag.
    example = CAGExample(
        id="p", question="q", context=["Intro: a", "Method: b"],
        answer="a", metadata={"evidence_doc_ids": [True]},
    )
    assert gold_paragraphs(example) == []
