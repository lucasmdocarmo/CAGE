"""Unit tests for the SCBench two-subset slice loader (charter D5 item 6).

Same fake-`datasets`-module technique as tests/test_dataset_loaders.py (the HF
`datasets` package is NOT required). Covers happy path (session structure
preserved per turn), boundary (session-level max_examples, list-valued
answers, env/subset configuration), and failure (typed fail-closed
DatasetUnavailableError on missing download and on schema mismatch — never a
silently empty or degraded example list).
"""

import random
import sys
import types

import pytest

from src.data.loader import DatasetLoader, DatasetUnavailableError, get_loader
from src.data.scbench import SCBENCH_SUBSETS, SCBenchLoader


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


def install_fake_datasets(monkeypatch, rows=None, raiser=None):
    """Fake `datasets` whose load_dataset returns `rows` or raises `raiser`."""
    calls = []

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        if raiser is not None:
            raise raiser
        return FakeHFDataset(rows)

    fake = types.ModuleType("datasets")
    fake.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake)
    return calls


def make_sessions(n=1, turns=2):
    """Synthetic microsoft/SCBench session rows (context + multi_turns)."""
    rows = []
    for i in range(n):
        rows.append({
            "id": i,
            "context": f"Shared long context of session {i}.",
            "multi_turns": [
                {"input": f"s{i} turn {t} question?", "answer": f"s{i}a{t}"}
                for t in range(turns)
            ],
        })
    return rows


# ---------------------------------------------------------------------------
# Session structure preserved
# ---------------------------------------------------------------------------


def test_one_example_per_turn_in_session_order(monkeypatch):
    calls = install_fake_datasets(monkeypatch, make_sessions(1, turns=3))
    examples = SCBenchLoader(split="test", subset="scbench_kv").load()

    assert calls == [(("microsoft/SCBench", "scbench_kv"), {"split": "test"})]
    assert [ex.id for ex in examples] == [
        "scbench_kv_0_turn0", "scbench_kv_0_turn1", "scbench_kv_0_turn2",
    ]
    assert [ex.metadata["turn_index"] for ex in examples] == [0, 1, 2]
    for ex in examples:
        # Every turn of a session shares the SAME verbatim context (the
        # multi-request reuse shape the harness must replay).
        assert ex.context == ["Shared long context of session 0."]
        assert ex.metadata["session_id"] == "0"
        assert ex.metadata["num_turns"] == 3
        assert ex.metadata["dataset"] == "scbench"
        assert ex.metadata["subset"] == "scbench_kv"
        assert ex.metadata["native_metrics_only"] is True
    assert examples[1].question == "s0 turn 1 question?"
    assert examples[1].answer == "s0a1"


def test_list_valued_answers_keep_all_golds(monkeypatch):
    rows = [{
        "id": 5,
        "context": "ctx",
        "multi_turns": [{"input": "q?", "answer": ["gold one", "gold two"]}],
    }]
    install_fake_datasets(monkeypatch, rows)
    ex = SCBenchLoader(split="test").load()[0]

    assert ex.answer == "gold one"
    assert ex.metadata["all_answers"] == ["gold one", "gold two"]


def test_max_examples_bounds_sessions_not_turns(monkeypatch):
    rows = make_sessions(10, turns=2)

    install_fake_datasets(monkeypatch, rows)
    a = SCBenchLoader(split="test", seed=42).load(max_examples=3)
    install_fake_datasets(monkeypatch, rows)
    b = SCBenchLoader(split="test", seed=42).load(max_examples=3)
    install_fake_datasets(monkeypatch, rows)
    c = SCBenchLoader(split="test", seed=43).load(max_examples=3)

    assert len(a) == 6  # 3 sessions x 2 turns each, turns never split
    assert len({ex.metadata["session_id"] for ex in a}) == 3
    # Seeded shuffle-before-select at the session level: reproducible per
    # seed, different across seeds, not the first-N sessions.
    assert [ex.id for ex in a] == [ex.id for ex in b]
    assert [ex.id for ex in a] != [ex.id for ex in c]


# ---------------------------------------------------------------------------
# Fail-closed behavior (typed errors, never silent degradation)
# ---------------------------------------------------------------------------


def test_missing_download_raises_typed_error(monkeypatch):
    install_fake_datasets(monkeypatch, raiser=ConnectionError("no cache, no net"))

    with pytest.raises(DatasetUnavailableError) as exc_info:
        SCBenchLoader(split="test").load()

    err = exc_info.value
    assert err.dataset == "scbench"
    assert "scbench_kv" in err.source
    assert "no cache, no net" in err.cause
    assert "download_datasets.py" in str(err)  # points at the staging script


def test_schema_mismatch_raises_typed_error(monkeypatch):
    install_fake_datasets(monkeypatch, [{"id": 0, "context": "ctx"}])  # no turns

    with pytest.raises(DatasetUnavailableError, match="session schema"):
        SCBenchLoader(split="test").load()


def test_turn_without_question_raises_typed_error(monkeypatch):
    rows = [{"id": 0, "context": "ctx",
             "multi_turns": [{"answer": "orphan gold"}]}]
    install_fake_datasets(monkeypatch, rows)

    with pytest.raises(DatasetUnavailableError, match="no\\s+input/question"):
        SCBenchLoader(split="test").load()


def test_unknown_subset_rejected_at_construction():
    with pytest.raises(ValueError, match="two-subset slice"):
        SCBenchLoader(subset="scbench_repoqa")


# ---------------------------------------------------------------------------
# Configuration + factory registration
# ---------------------------------------------------------------------------


def test_defaults_and_env_overrides(monkeypatch):
    loader = SCBenchLoader()
    assert loader.subset == "scbench_kv"
    assert loader.split == "test"  # the split microsoft/SCBench publishes
    assert loader.hf_path == "microsoft/SCBench"
    assert SCBENCH_SUBSETS == ("scbench_kv", "scbench_qa_eng")

    monkeypatch.setenv("CAGE_SCBENCH_SUBSET", "scbench_qa_eng")
    monkeypatch.setenv("CAGE_SCBENCH_SPLIT", "train")
    monkeypatch.setenv("CAGE_SCBENCH_HF_PATH", "mirror/SCBench")
    loader = SCBenchLoader()
    assert loader.subset == "scbench_qa_eng"
    assert loader.split == "train"
    assert loader.hf_path == "mirror/SCBench"


def test_registered_in_factory(monkeypatch):
    loader = get_loader("scbench", split="test", seed=9)
    assert isinstance(loader, SCBenchLoader)
    assert isinstance(loader, DatasetLoader)
    assert loader.split == "test"
    assert loader.seed == 9


def test_factory_error_message_lists_new_datasets():
    with pytest.raises(ValueError, match="ruler"):
        get_loader("not_a_dataset")
    with pytest.raises(ValueError, match="scbench"):
        get_loader("not_a_dataset")
