"""
Unit tests for HotpotQALoader and MuSiQueLoader using synthetic HF payloads.

The `datasets` package is NOT required (nor installed in the analysis venv):
loaders import it lazily inside load(), so these tests inject a fake `datasets`
module into sys.modules whose load_dataset() returns an in-memory stand-in that
mimics the HF Dataset API surface the loaders use (shuffle/select/len/iter).

Covers:
- context assembly (list of title-prefixed paragraph strings, distractors kept)
- supporting-titles metadata (gold-paragraph recovery for corpus/gold selection)
- seeded shuffle-BEFORE-select trial independence (different seeds -> different
  reproducible samples; same seed -> identical samples)
- answerable conventions (gold answer always non-empty, no is_impossible flag)
"""

import random
import sys
import types

import pytest

from src.data.loader import CAGExample, HotpotQALoader, MuSiQueLoader, get_loader


# ---------------------------------------------------------------------------
# Fake HuggingFace `datasets` machinery
# ---------------------------------------------------------------------------


class FakeHFDataset:
    """Minimal stand-in for datasets.Dataset (shuffle/select/len/iter only)."""

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
    """Install a fake `datasets` module whose load_dataset returns `rows`.

    Returns the list of (args, kwargs) load_dataset was called with, so tests
    can assert the HF path/config/split.
    """
    calls = []

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeHFDataset(rows)

    fake = types.ModuleType("datasets")
    fake.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake)
    return calls


# ---------------------------------------------------------------------------
# Synthetic payloads
# ---------------------------------------------------------------------------


def make_hotpotqa_rows(n=1):
    """Synthetic hotpot_qa/distractor rows (schema-faithful)."""
    rows = []
    for i in range(n):
        rows.append({
            "id": f"hp_{i}",
            "question": f"Which magazine was started first, A{i} or B{i}?",
            "answer": f"A{i}",
            "type": "comparison",
            "level": "hard",
            # One gold paragraph can contribute several supporting sentences
            # (duplicate title on purpose).
            "supporting_facts": {"title": [f"A{i}", f"B{i}", f"A{i}"], "sent_id": [0, 0, 1]},
            "context": {
                "title": [f"A{i}", f"B{i}", f"C{i}"],
                "sentences": [
                    [f"A{i} is a magazine.", " It was started in 1990."],
                    [f"B{i} is a magazine started in 1995."],
                    [f"C{i} is an unrelated distractor."],
                ],
            },
        })
    return rows


def make_musique_rows(n=1):
    """Synthetic dgslibisey/MuSiQue rows (schema-faithful)."""
    rows = []
    for i in range(n):
        rows.append({
            "id": f"2hop__{i}",
            "question": f"Who founded the company that made product P{i}?",
            "answer": f"Founder{i}",
            "answerable": True,
            "paragraphs": [
                {"idx": 0, "title": f"Product P{i}", "paragraph_text": f"P{i} was made by Corp{i}.",
                 "is_supporting": True},
                {"idx": 1, "title": f"Distractor D{i}", "paragraph_text": "Unrelated text.",
                 "is_supporting": False},
                {"idx": 2, "title": f"Corp{i}", "paragraph_text": f"Corp{i} was founded by Founder{i}.",
                 "is_supporting": True},
            ],
            "question_decomposition": [
                {"question": f"Which company made P{i}?", "answer": f"Corp{i}"},
                {"question": f"Who founded Corp{i}?", "answer": f"Founder{i}"},
            ],
        })
    return rows


# ---------------------------------------------------------------------------
# HotpotQALoader
# ---------------------------------------------------------------------------


def test_hotpotqa_context_assembly(monkeypatch):
    """Context is a list of title-prefixed paragraph strings, distractors kept."""
    calls = install_fake_datasets(monkeypatch, make_hotpotqa_rows(1))
    examples = HotpotQALoader(split="validation", seed=42).load()

    # Correct HF path/config/split.
    assert calls == [(("hotpot_qa", "distractor"), {"split": "validation"})]

    assert len(examples) == 1
    ex = examples[0]
    assert isinstance(ex, CAGExample)
    assert ex.id == "hp_0"
    # ALL paragraphs kept (gold + distractor), one string per title,
    # "<title>: " + "".join(sentences) (sentences carry own leading spaces).
    assert ex.context == [
        "A0: A0 is a magazine. It was started in 1990.",
        "B0: B0 is a magazine started in 1995.",
        "C0: C0 is an unrelated distractor.",
    ]


def test_hotpotqa_supporting_titles_metadata(monkeypatch):
    """supporting_facts titles land deduplicated in metadata['supporting_titles']."""
    install_fake_datasets(monkeypatch, make_hotpotqa_rows(1))
    ex = HotpotQALoader().load()[0]

    # Deduplicated (A0 contributed 2 supporting sentences), order-preserving,
    # and each supporting title maps back to a gold context paragraph by prefix.
    assert ex.metadata["supporting_titles"] == ["A0", "B0"]
    for title in ex.metadata["supporting_titles"]:
        assert any(c.startswith(f"{title}: ") for c in ex.context)
    # Distractor title is in context but not in supporting_titles.
    assert any(c.startswith("C0: ") for c in ex.context)
    assert "C0" not in ex.metadata["supporting_titles"]

    assert ex.metadata["dataset"] == "hotpotqa"
    assert ex.metadata["type"] == "comparison"
    assert ex.metadata["level"] == "hard"


def test_hotpotqa_answerable_conventions(monkeypatch):
    """All HotpotQA items are answerable: non-empty gold, no is_impossible flag."""
    install_fake_datasets(monkeypatch, make_hotpotqa_rows(4))
    examples = HotpotQALoader().load()

    assert len(examples) == 4
    for ex in examples:
        # Same convention as MuSiQue/NQ (SQuAD v2 signals unanswerable via
        # empty answer + metadata["is_impossible"]; never emitted here).
        assert ex.answer != ""
        assert "is_impossible" not in ex.metadata


def test_hotpotqa_trial_independence(monkeypatch):
    """Seeded shuffle BEFORE select: different seeds -> different reproducible draws."""
    rows = make_hotpotqa_rows(20)

    install_fake_datasets(monkeypatch, rows)
    ids_seed42_a = [ex.id for ex in HotpotQALoader(seed=42).load(max_examples=5)]
    install_fake_datasets(monkeypatch, rows)
    ids_seed42_b = [ex.id for ex in HotpotQALoader(seed=42).load(max_examples=5)]
    install_fake_datasets(monkeypatch, rows)
    ids_seed43 = [ex.id for ex in HotpotQALoader(seed=43).load(max_examples=5)]

    assert len(ids_seed42_a) == 5
    # Same seed reproduces the exact same sample.
    assert ids_seed42_a == ids_seed42_b
    # Different seed (per trial) draws a different sample — NOT the first-N.
    assert ids_seed42_a != ids_seed43
    assert ids_seed42_a != [f"hp_{i}" for i in range(5)]


def test_hotpotqa_registered_in_factory(monkeypatch):
    """get_loader('hotpotqa') resolves to HotpotQALoader."""
    loader = get_loader("hotpotqa", split="validation", seed=7)
    assert isinstance(loader, HotpotQALoader)
    assert loader.split == "validation"
    assert loader.seed == 7


# ---------------------------------------------------------------------------
# MuSiQueLoader
# ---------------------------------------------------------------------------


def test_musique_context_assembly(monkeypatch):
    """Context is a list of title-prefixed paragraph strings, distractors kept."""
    calls = install_fake_datasets(monkeypatch, make_musique_rows(1))
    examples = MuSiQueLoader(split="validation", seed=42).load()

    # Correct HF path/split.
    assert calls == [(("dgslibisey/MuSiQue",), {"split": "validation"})]

    assert len(examples) == 1
    ex = examples[0]
    assert isinstance(ex, CAGExample)
    assert ex.id == "2hop__0"
    assert ex.context == [
        "Product P0: P0 was made by Corp0.",
        "Distractor D0: Unrelated text.",
        "Corp0: Corp0 was founded by Founder0.",
    ]


def test_musique_supporting_titles_and_hops_metadata(monkeypatch):
    """is_supporting titles land in metadata; num_hops is the hop COUNT."""
    install_fake_datasets(monkeypatch, make_musique_rows(1))
    ex = MuSiQueLoader().load()[0]

    assert ex.metadata["supporting_titles"] == ["Product P0", "Corp0"]
    for title in ex.metadata["supporting_titles"]:
        assert any(c.startswith(f"{title}: ") for c in ex.context)
    assert "Distractor D0" not in ex.metadata["supporting_titles"]

    assert ex.metadata["dataset"] == "musique"
    # Regression: num_hops used to hold the raw question_decomposition list.
    assert ex.metadata["num_hops"] == 2


def test_musique_answerable_conventions(monkeypatch):
    """Answerable split: non-empty gold answer, no is_impossible flag."""
    install_fake_datasets(monkeypatch, make_musique_rows(4))
    examples = MuSiQueLoader().load()

    assert len(examples) == 4
    for ex in examples:
        assert ex.answer != ""
        assert "is_impossible" not in ex.metadata


def test_musique_trial_independence(monkeypatch):
    """Seeded shuffle BEFORE select: different seeds -> different reproducible draws."""
    rows = make_musique_rows(20)

    install_fake_datasets(monkeypatch, rows)
    ids_seed42_a = [ex.id for ex in MuSiQueLoader(seed=42).load(max_examples=5)]
    install_fake_datasets(monkeypatch, rows)
    ids_seed42_b = [ex.id for ex in MuSiQueLoader(seed=42).load(max_examples=5)]
    install_fake_datasets(monkeypatch, rows)
    ids_seed43 = [ex.id for ex in MuSiQueLoader(seed=43).load(max_examples=5)]

    assert len(ids_seed42_a) == 5
    assert ids_seed42_a == ids_seed42_b
    assert ids_seed42_a != ids_seed43
    assert ids_seed42_a != [f"2hop__{i}" for i in range(5)]


def test_musique_registered_in_factory(monkeypatch):
    """get_loader('musique') resolves to MuSiQueLoader."""
    loader = get_loader("musique", split="validation", seed=7)
    assert isinstance(loader, MuSiQueLoader)
    assert loader.split == "validation"
    assert loader.seed == 7
