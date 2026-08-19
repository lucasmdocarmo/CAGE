"""Offline coverage for ShareGPTLoader (K-COV3, backlog #142).

The registered D5 load-shape donor's format-parsing heuristics were at 0%
coverage: an HF dump drift would produce degenerate load shapes in every
open-loop cell without any test noticing. These tests exercise
``ShareGPTLoader.load`` against synthetic ShareGPT payloads covering every
schema variant the parser claims to handle — no `datasets` package and no
network (fake module injected into sys.modules, mirroring
tests/test_dataset_loaders.py).

Covers:
- from/value ("human"/"gpt") AND role/content ("user"/"assistant") turn forms
- conversations / conversation / items container aliases
- first-human-turn = question, first assistant turn AFTER it = reference
- system/gpt-opening dumps fall back to the first turn's text as the question
- string (non-dict) turns
- degenerate rows are SKIPPED (empty convo, non-list convo, no text anywhere)
- serving-trace metadata contract (no_gold_answer, empty context, num_turns)
- id fallback, seeded shuffle-before-select, hf_path env/arg precedence
- get_loader("sharegpt") factory wiring
"""

from __future__ import annotations

import random
import sys
import types

import pytest

from src.data.loader import CAGExample, ShareGPTLoader, get_loader


# --------------------------------------------------------------------------- #
# Fake HuggingFace `datasets` machinery (pattern: tests/test_dataset_loaders.py)
# --------------------------------------------------------------------------- #


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
    """Install a fake `datasets` module; returns the load_dataset call log."""
    calls = []

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeHFDataset(rows)

    fake = types.ModuleType("datasets")
    fake.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake)
    return calls


def make_sharegpt_rows(n=1):
    """Schema-faithful RyokoAI/ShareGPT52K-style rows (from/value turns)."""
    rows = []
    for i in range(n):
        rows.append({
            "id": f"sg_{i}",
            "conversations": [
                {"from": "human", "value": f"How do I sort a list in Python? ({i})"},
                {"from": "gpt", "value": f"Use sorted() or list.sort(). ({i})"},
                {"from": "human", "value": "What about descending?"},
                {"from": "gpt", "value": "Pass reverse=True."},
            ],
        })
    return rows


# --------------------------------------------------------------------------- #
# Turn-form and container-alias parsing
# --------------------------------------------------------------------------- #


class TestTurnParsing:
    def test_from_value_turns_first_human_and_first_gpt(self, monkeypatch):
        install_fake_datasets(monkeypatch, make_sharegpt_rows(1))
        examples = ShareGPTLoader().load()
        assert len(examples) == 1
        ex = examples[0]
        assert isinstance(ex, CAGExample)
        assert ex.id == "sg_0"
        assert ex.question == "How do I sort a list in Python? (0)"
        # First assistant turn AFTER the question, not the last one.
        assert ex.answer == "Use sorted() or list.sort(). (0)"

    def test_role_content_turn_form(self, monkeypatch):
        install_fake_datasets(monkeypatch, [{
            "id": "rc_0",
            "conversations": [
                {"role": "user", "content": "What is CAG?"},
                {"role": "assistant", "content": "Cache-augmented generation."},
            ],
        }])
        (ex,) = ShareGPTLoader().load()
        assert ex.question == "What is CAG?"
        assert ex.answer == "Cache-augmented generation."

    @pytest.mark.parametrize("container", ["conversations", "conversation", "items"])
    def test_container_aliases(self, monkeypatch, container):
        install_fake_datasets(monkeypatch, [{
            "id": "c_0",
            container: [
                {"from": "human", "value": "Q?"},
                {"from": "gpt", "value": "A."},
            ],
        }])
        (ex,) = ShareGPTLoader().load()
        assert (ex.question, ex.answer) == ("Q?", "A.")

    def test_gpt_opening_dump_falls_back_to_first_turn_text(self, monkeypatch):
        # No human/user turn at all: question falls back to the FIRST turn's
        # text (the documented "some dumps open with a system/gpt turn" arm).
        install_fake_datasets(monkeypatch, [{
            "id": "g_0",
            "conversations": [
                {"from": "system", "value": "You are a helpful assistant."},
                {"from": "gpt", "value": "Hello! How can I help?"},
            ],
        }])
        (ex,) = ShareGPTLoader().load()
        assert ex.question == "You are a helpful assistant."
        # No question was found in-loop, so no reference was captured either.
        assert ex.answer == ""

    def test_string_turns_are_parsed_via_str_fallback(self, monkeypatch):
        # Non-dict turns: _role() is "" (never matches), _text() is str(turn),
        # so the question comes from the first-turn fallback.
        install_fake_datasets(monkeypatch, [{
            "id": "s_0",
            "conversations": ["plain first turn text", "second turn"],
        }])
        (ex,) = ShareGPTLoader().load()
        assert ex.question == "plain first turn text"
        assert ex.answer == ""

    def test_reference_requires_question_first(self, monkeypatch):
        # gpt BEFORE the first human turn is never the reference; the first
        # gpt AFTER the human turn is.
        install_fake_datasets(monkeypatch, [{
            "id": "o_0",
            "conversations": [
                {"from": "gpt", "value": "unsolicited greeting"},
                {"from": "human", "value": "Actual question?"},
                {"from": "gpt", "value": "Actual reference."},
            ],
        }])
        (ex,) = ShareGPTLoader().load()
        assert ex.question == "Actual question?"
        assert ex.answer == "Actual reference."


# --------------------------------------------------------------------------- #
# Degenerate rows are skipped, never emitted
# --------------------------------------------------------------------------- #


class TestDegenerateRows:
    def test_empty_and_missing_and_nonlist_convos_are_skipped(self, monkeypatch):
        rows = [
            {"id": "bad_empty", "conversations": []},
            {"id": "bad_missing"},
            {"id": "bad_nonlist", "conversations": "not a list"},
            make_sharegpt_rows(1)[0],
        ]
        install_fake_datasets(monkeypatch, rows)
        examples = ShareGPTLoader().load()
        assert [e.id for e in examples] == ["sg_0"]

    def test_row_with_no_text_anywhere_is_skipped(self, monkeypatch):
        # Turns exist but carry no text: fallback _text(convo[0]) is "" too.
        install_fake_datasets(monkeypatch, [{
            "id": "empty_text",
            "conversations": [{"from": "human", "value": ""}, {"from": "gpt"}],
        }])
        assert ShareGPTLoader().load() == []


# --------------------------------------------------------------------------- #
# Serving-trace metadata contract (NOT a QA benchmark)
# --------------------------------------------------------------------------- #


class TestMetadataContract:
    def test_serving_trace_metadata_and_empty_context(self, monkeypatch):
        install_fake_datasets(monkeypatch, make_sharegpt_rows(1))
        (ex,) = ShareGPTLoader().load()
        # Open conversation: no supplied gold context, reference-only answer.
        assert ex.context == []
        assert ex.metadata["dataset"] == "sharegpt"
        assert ex.metadata["dataset_type"] == "conversation_trace"
        assert ex.metadata["no_gold_answer"] is True
        assert ex.metadata["num_turns"] == 4

    def test_id_falls_back_to_running_index(self, monkeypatch):
        rows = [
            {"conversations": [{"from": "human", "value": "Q0?"}]},
            {"conversations": [{"from": "human", "value": "Q1?"}]},
        ]
        install_fake_datasets(monkeypatch, rows)
        examples = ShareGPTLoader().load()
        assert [e.id for e in examples] == ["0", "1"]


# --------------------------------------------------------------------------- #
# Sampling and configuration
# --------------------------------------------------------------------------- #


class TestSamplingAndConfig:
    def test_seeded_shuffle_before_select_is_reproducible(self, monkeypatch):
        rows = make_sharegpt_rows(30)
        install_fake_datasets(monkeypatch, rows)
        a = ShareGPTLoader(seed=7).load(max_examples=5)
        install_fake_datasets(monkeypatch, rows)
        b = ShareGPTLoader(seed=7).load(max_examples=5)
        install_fake_datasets(monkeypatch, rows)
        c = ShareGPTLoader(seed=8).load(max_examples=5)
        assert [e.id for e in a] == [e.id for e in b]
        assert len(a) == 5
        # Different seed -> different reproducible sample (30 rows, 5 picks:
        # identical ordering would be a broken shuffle, not chance).
        assert [e.id for e in a] != [e.id for e in c]

    def test_max_examples_caps_at_dataset_size(self, monkeypatch):
        install_fake_datasets(monkeypatch, make_sharegpt_rows(3))
        assert len(ShareGPTLoader().load(max_examples=100)) == 3

    def test_default_hf_path_and_split(self, monkeypatch):
        monkeypatch.delenv("CAGE_SHAREGPT_HF_PATH", raising=False)
        calls = install_fake_datasets(monkeypatch, make_sharegpt_rows(1))
        ShareGPTLoader().load()
        (args, kwargs) = calls[0]
        assert args == ("RyokoAI/ShareGPT52K",)
        assert kwargs == {"split": "train"}

    def test_env_overrides_default_and_arg_overrides_env(self, monkeypatch):
        monkeypatch.setenv("CAGE_SHAREGPT_HF_PATH", "mirror/FromEnv")
        calls = install_fake_datasets(monkeypatch, make_sharegpt_rows(1))
        ShareGPTLoader().load()
        assert calls[-1][0] == ("mirror/FromEnv",)
        ShareGPTLoader(hf_path="mirror/FromArg").load()
        assert calls[-1][0] == ("mirror/FromArg",)

    def test_get_loader_factory_returns_sharegpt_loader(self, monkeypatch):
        monkeypatch.delenv("CAGE_SHAREGPT_HF_PATH", raising=False)
        loader = get_loader("sharegpt", split="train", seed=3)
        assert isinstance(loader, ShareGPTLoader)
        assert loader.hf_path == "RyokoAI/ShareGPT52K"
        assert loader.seed == 3
