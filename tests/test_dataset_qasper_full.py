"""Unit tests for the FULL QasperLoader (charter D5 item 4 — launch blocker).

Covers what tests/test_dataset_loaders.py's pinned regression tests do not:
- multi-ANNOTATOR deterministic answer resolution (all-unanswerable ->
  abstention; otherwise first non-empty annotator wins; all_answers collects
  every annotator's non-empty resolution, deduplicated, primary first)
- qrels-ready gold-evidence metadata (evidence union across annotators, exact
  texts; evidence_doc_ids mapping into the emitted context; supporting_titles
  section names driving gold_only())
- unmatched evidence (FLOAT SELECTED figure refs) kept as text, no doc id
- include_title=True prepends a "Title: ..." doc (default off, pinned layout)
- max_examples bounds PAPERS (each contributing all its questions)

Same fake-`datasets`-module technique as tests/test_dataset_loaders.py: the
HF `datasets` package is NOT required.
"""

import random
import sys
import types
from typing import Any, Dict, List, Optional

from src.data.loader import CAGExample, QasperLoader, gold_only


# ---------------------------------------------------------------------------
# Fake HuggingFace `datasets` machinery (mirrors tests/test_dataset_loaders.py)
# ---------------------------------------------------------------------------


class FakeHFDataset:
    def __init__(self, rows):
        self._rows = list(rows)

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def shuffle(self, seed=None):
        rows = list(self._rows)
        random.Random(seed).shuffle(rows)
        return FakeHFDataset(rows)

    def select(self, indices):
        return FakeHFDataset([self._rows[i] for i in indices])


def install_fake_datasets(monkeypatch, rows):
    calls = []

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeHFDataset(rows)

    fake = types.ModuleType("datasets")
    fake.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake)
    return calls


# ---------------------------------------------------------------------------
# Synthetic schema-faithful payload builders
# ---------------------------------------------------------------------------


def annot(unanswerable=False, yes_no=None, free_form="", spans=None,
          evidence=None, highlighted=None) -> Dict[str, Any]:
    """One per-annotator answer dict (real allenai/qasper answer schema)."""
    return {
        "unanswerable": unanswerable,
        "yes_no": yes_no,
        "free_form_answer": free_form,
        "extractive_spans": spans or [],
        "evidence": evidence or [],
        "highlighted_evidence": highlighted or [],
    }


def make_paper(paper_id: str, questions: List[str], answers: List[List[Dict]],
               qids: Optional[List[str]] = None) -> Dict[str, Any]:
    """One columnar allenai/qasper row (abstract TOP-LEVEL; qas dict-of-lists)."""
    qids = qids or [f"q{i}" for i in range(len(questions))]
    return {
        "id": paper_id,
        "title": f"The {paper_id} Paper",
        "abstract": f"Abstract text of {paper_id}.",
        "full_text": {
            "section_name": ["Introduction", "Method", "Results"],
            "paragraphs": [
                [f"Intro para one of {paper_id}.", f"Intro para two of {paper_id}."],
                [f"Method para of {paper_id}."],
                [f"Results para of {paper_id}."],
            ],
        },
        "qas": {
            "question": questions,
            "question_id": qids,
            "answers": [
                {"answer": ann, "annotation_id": [f"a{i}"], "worker_id": [f"w{i}"]}
                for i, ann in enumerate(answers)
            ],
        },
    }


# ---------------------------------------------------------------------------
# Multi-annotator deterministic answer resolution
# ---------------------------------------------------------------------------


def test_all_annotators_unanswerable_is_abstention(monkeypatch):
    """Every annotator unanswerable -> empty answer + is_impossible=True
    (the abstention axis is preserved, SQuAD v2 convention)."""
    rows = [make_paper("p1", ["Q?"], [[annot(unanswerable=True),
                                       annot(unanswerable=True)]])]
    install_fake_datasets(monkeypatch, rows)
    ex = QasperLoader().load()[0]

    assert ex.answer == ""
    assert ex.metadata["is_impossible"] is True
    assert ex.metadata["answer_type"] == "unanswerable"
    assert ex.metadata["all_answers"] == []
    assert ex.metadata["num_annotators"] == 2
    assert ex.metadata["num_unanswerable_annotators"] == 2


def test_first_non_empty_annotator_wins_over_unanswerable_first(monkeypatch):
    """Annotators disagree (first: unanswerable, second: yes/no) -> the
    question is ANSWERABLE and the first non-empty resolution is primary."""
    rows = [make_paper("p1", ["Q?"], [[annot(unanswerable=True),
                                       annot(yes_no=False)]])]
    install_fake_datasets(monkeypatch, rows)
    ex = QasperLoader().load()[0]

    assert ex.answer == "No"
    assert ex.metadata["is_impossible"] is False
    assert ex.metadata["answer_type"] == "yes_no"
    assert ex.metadata["all_answers"] == ["No"]
    assert ex.metadata["num_unanswerable_annotators"] == 1


def test_all_answers_collects_every_annotator_deduplicated(monkeypatch):
    """all_answers = order-preserving dedup of every annotator's non-empty
    resolution, primary first (max-over-golds consumers, SQuAD v2 key)."""
    rows = [make_paper("p1", ["Q?"], [[
        annot(spans=["span A", "span B"]),          # -> "span A; span B" (primary)
        annot(free_form="a free form answer"),
        annot(spans=["span A", "span B"]),          # duplicate -> deduped
    ]])]
    install_fake_datasets(monkeypatch, rows)
    ex = QasperLoader().load()[0]

    assert ex.answer == "span A; span B"
    assert ex.metadata["answer_type"] == "extractive"
    assert ex.metadata["all_answers"] == ["span A; span B", "a free form answer"]
    assert ex.metadata["num_annotators"] == 3


def test_within_annotator_precedence_unanswerable_before_yes_no(monkeypatch):
    """A stale/default yes_no co-occurring with unanswerable=True must still
    resolve unanswerable (fixed precedence, per docstring)."""
    rows = [make_paper("p1", ["Q?"], [[annot(unanswerable=True, yes_no=True)]])]
    install_fake_datasets(monkeypatch, rows)
    ex = QasperLoader().load()[0]

    assert ex.answer == ""
    assert ex.metadata["is_impossible"] is True


def test_unresolvable_annotator_ignored_and_empty_question_skipped(monkeypatch):
    """An annotator resolving to nothing is ignored (another annotator's
    answer carries); a question whose ONLY annotator resolves to nothing is
    skipped entirely (pinned pre-existing filter)."""
    rows = [make_paper(
        "p1",
        ["Kept?", "Dropped?"],
        [[annot(), annot(free_form="carried")],  # first annotator = nothing
         [annot()]],                             # only annotator = nothing
    )]
    install_fake_datasets(monkeypatch, rows)
    examples = QasperLoader().load()

    assert [ex.question for ex in examples] == ["Kept?"]
    assert examples[0].answer == "carried"
    assert examples[0].metadata["num_annotators"] == 1  # resolvable only


# ---------------------------------------------------------------------------
# Qrels-ready evidence metadata
# ---------------------------------------------------------------------------


def test_evidence_union_doc_ids_and_supporting_titles(monkeypatch):
    """Evidence texts are unioned across annotators (exact texts, deduped),
    mapped to context doc indices, and surfaced as section supporting_titles."""
    rows = [make_paper("p1", ["Q?"], [[
        annot(spans=["x"],
              evidence=["Method para of p1.", "FLOAT SELECTED: Figure 1"],
              highlighted=["Method para"]),
        annot(free_form="y",
              evidence=["Results para of p1.", "Method para of p1."],
              highlighted=["Results para"]),
    ]])]
    install_fake_datasets(monkeypatch, rows)
    ex = QasperLoader().load()[0]

    # context = [Abstract, Introduction, Method, Results]
    assert ex.context[2] == "Method: Method para of p1."
    assert ex.context[3] == "Results: Results para of p1."

    # Union, order-preserving, deduplicated; FLOAT kept verbatim as text.
    assert ex.metadata["evidence"] == [
        "Method para of p1.",
        "FLOAT SELECTED: Figure 1",
        "Results para of p1.",
    ]
    # Doc ids only for evidence found verbatim in a context doc.
    assert ex.metadata["evidence_doc_ids"] == [2, 3]
    assert ex.metadata["supporting_titles"] == ["Method", "Results"]
    assert ex.metadata["highlighted_evidence"] == ["Method para", "Results para"]


def test_gold_only_selects_evidence_sections(monkeypatch):
    """gold_only() (corpus/qrels machinery) returns exactly the evidence-backed
    section docs via supporting_titles — same convention as HotpotQA/MuSiQue."""
    rows = [make_paper("p1", ["Q?"], [[
        annot(spans=["x"], evidence=["Intro para two of p1."]),
    ]])]
    install_fake_datasets(monkeypatch, rows)
    ex = QasperLoader().load()[0]

    assert ex.metadata["supporting_titles"] == ["Introduction"]
    assert gold_only(ex) == [
        "Introduction: Intro para one of p1.\nIntro para two of p1."
    ]


def test_evidence_matching_abstract_maps_to_abstract_doc(monkeypatch):
    """Evidence quoting the abstract maps to the 'Abstract' pseudo-section."""
    rows = [make_paper("p1", ["Q?"], [[
        annot(yes_no=True, evidence=["Abstract text of p1."]),
    ]])]
    install_fake_datasets(monkeypatch, rows)
    ex = QasperLoader().load()[0]

    assert ex.metadata["evidence_doc_ids"] == [0]
    assert ex.metadata["supporting_titles"] == ["Abstract"]


def test_no_evidence_yields_empty_qrels_fields(monkeypatch):
    """No evidence anywhere -> empty evidence/doc-id/title lists (and
    gold_only falls back to the full context, its documented behavior)."""
    rows = [make_paper("p1", ["Q?"], [[annot(yes_no=True)]])]
    install_fake_datasets(monkeypatch, rows)
    ex = QasperLoader().load()[0]

    assert ex.metadata["evidence"] == []
    assert ex.metadata["evidence_doc_ids"] == []
    assert ex.metadata["supporting_titles"] == []
    assert gold_only(ex) == ex.context


# ---------------------------------------------------------------------------
# Context assembly options + paper-level sampling
# ---------------------------------------------------------------------------


def test_include_title_prepends_title_doc_and_shifts_doc_ids(monkeypatch):
    """include_title=True prepends 'Title: ...' (full-paper context, D5#4);
    evidence_doc_ids index the ACTUAL emitted context list."""
    rows = [make_paper("p1", ["Q?"], [[
        annot(yes_no=True, evidence=["Method para of p1."]),
    ]])]
    install_fake_datasets(monkeypatch, rows)
    ex = QasperLoader(include_title=True).load()[0]

    assert ex.context[0] == "Title: The p1 Paper"
    assert ex.context[1] == "Abstract: Abstract text of p1."
    assert ex.metadata["evidence_doc_ids"] == [3]
    assert ex.metadata["supporting_titles"] == ["Method"]


def test_default_context_layout_unchanged(monkeypatch):
    """Default (include_title=False) keeps the pinned Abstract-first layout."""
    rows = [make_paper("p1", ["Q?"], [[annot(yes_no=True)]])]
    install_fake_datasets(monkeypatch, rows)
    ex = QasperLoader().load()[0]

    assert ex.context[0] == "Abstract: Abstract text of p1."
    assert all(not c.startswith("Title: ") for c in ex.context)


def test_max_examples_bounds_papers_not_questions(monkeypatch):
    """max_examples selects PAPERS (seeded shuffle-before-select); every
    question of a selected paper is kept."""
    rows = [
        make_paper(f"p{i}", [f"p{i} qa?", f"p{i} qb?"],
                   [[annot(yes_no=True)], [annot(free_form="ans")]])
        for i in range(6)
    ]
    install_fake_datasets(monkeypatch, rows)
    examples = QasperLoader(seed=42).load(max_examples=2)

    assert len(examples) == 4  # 2 papers x 2 questions
    papers = {ex.metadata["paper_id"] for ex in examples}
    assert len(papers) == 2
    for paper in papers:
        assert sum(1 for ex in examples if ex.metadata["paper_id"] == paper) == 2

    # Reproducible under the same seed.
    install_fake_datasets(monkeypatch, rows)
    again = QasperLoader(seed=42).load(max_examples=2)
    assert [ex.id for ex in again] == [ex.id for ex in examples]


def test_examples_are_cag_examples_with_dataset_tag(monkeypatch):
    rows = [make_paper("p1", ["Q?"], [[annot(yes_no=True)]])]
    install_fake_datasets(monkeypatch, rows)
    ex = QasperLoader().load()[0]

    assert isinstance(ex, CAGExample)
    assert ex.metadata["dataset"] == "qasper"
    assert ex.metadata["paper_id"] == "p1"
    assert ex.id == "p1_q0"
