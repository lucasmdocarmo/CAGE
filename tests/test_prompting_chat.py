"""Chat-template prompt construction seams (approved pre-run package, 2026-07-16).

Covers Decision 1B (chat messages + raw escape hatch), Decision 2A (shared
abstention instruction), and Decision 3B (deterministic distractor-corpus
selection with a mocked loader pool). Lean-venv by design: no torch, no
requests, no serving deps.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.prompting import (  # noqa: E402
    ABSTAIN_INSTRUCTION,
    CHAT_SYSTEM_INSTRUCTION,
    DEFAULT_SYSTEM_PREFIX,
    format_multi_turn_messages,
    format_qa_messages,
    format_qa_prompt,
    messages_to_fallback_prompt,
    prompt_mode,
    select_distractor_texts,
)

RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "3_run" / "run_cag_reference.py"


def _load_reference_runner():
    spec = importlib.util.spec_from_file_location("run_cag_reference_chat", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class FakeExample:
    """Mocked loader output (mirrors src.data.loader.CAGExample's shape)."""

    id: str
    question: str
    context: List[str]
    answer: str = "x"
    metadata: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Decision 1B: chat message construction
# --------------------------------------------------------------------------

def test_qa_messages_structure_system_then_user():
    messages = format_qa_messages("Who?", ["CTX-A", "CTX-B"])
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == CHAT_SYSTEM_INSTRUCTION
    user = messages[1]["content"]
    # Context first, question last (prefix sharing preserved).
    assert user.startswith("Context 1: CTX-A")
    assert "Context 2: CTX-B" in user
    assert user.endswith("Question: Who?")
    assert user.index("CTX-A") < user.index("Question: Who?")


def test_qa_messages_without_context_still_has_question_last():
    messages = format_qa_messages("Who?")
    assert messages[1]["content"] == "Question: Who?"


def test_multi_turn_messages_history_alternates_and_question_last():
    history = [("q1", "a1"), ("q2", "a2")]
    messages = format_multi_turn_messages("q3", ["CTX"], history=history)
    assert [m["role"] for m in messages] == [
        "system", "user", "assistant", "user", "assistant", "user",
    ]
    assert messages[1]["content"] == "q1"
    assert messages[2]["content"] == "a1"
    final = messages[-1]["content"]
    assert final.startswith("Context 1: CTX")
    assert final.endswith("Question: q3")


def test_fallback_prompt_carries_all_message_content():
    messages = format_qa_messages("Who?", ["CTX"])
    flat = messages_to_fallback_prompt(messages)
    assert CHAT_SYSTEM_INSTRUCTION in flat
    assert "Context 1: CTX" in flat
    assert "Question: Who?" in flat


# --------------------------------------------------------------------------
# Decision 2A: abstention instruction, identical across all arms
# --------------------------------------------------------------------------

def test_abstention_instruction_exact_wording():
    assert ABSTAIN_INSTRUCTION == (
        "If the answer cannot be found in the context, reply exactly: unanswerable"
    )


def test_abstention_instruction_in_every_chat_arm_system_message():
    qa_sys = format_qa_messages("Who?", ["CTX"])[0]["content"]
    mt_sys = format_multi_turn_messages("Who?", ["CTX"], history=[("a", "b")])[0]["content"]
    assert ABSTAIN_INSTRUCTION in qa_sys
    assert ABSTAIN_INSTRUCTION in mt_sys
    # IDENTICAL across arms: same system instruction object end-to-end.
    assert qa_sys == mt_sys == CHAT_SYSTEM_INSTRUCTION


def test_abstention_reply_is_caught_by_quality_detector():
    # Decision 2A contract check: the exact instructed reply must be detected
    # as an abstention by the (untouched) quality module.
    from src.evaluation.quality import is_no_answer_prediction

    assert is_no_answer_prediction("unanswerable") is True
    assert is_no_answer_prediction("Unanswerable.") is True
    assert is_no_answer_prediction("Paris") is False


# --------------------------------------------------------------------------
# Raw-mode escape hatch (CAGE_PROMPT_MODE)
# --------------------------------------------------------------------------

def test_prompt_mode_defaults_to_chat_and_raw_escape_hatch():
    assert prompt_mode(env={}) == "chat"
    assert prompt_mode(env={"CAGE_PROMPT_MODE": "raw"}) == "raw"
    assert prompt_mode(env={"CAGE_PROMPT_MODE": " CHAT "}) == "chat"
    with pytest.raises(ValueError):
        prompt_mode(env={"CAGE_PROMPT_MODE": "completions"})


def test_raw_mode_legacy_prompt_layout_unchanged():
    # Byte-for-byte golden: the raw escape hatch must reproduce the legacy
    # /v1/completions prompt exactly.
    expected = (
        DEFAULT_SYSTEM_PREFIX.rstrip() + "\n"
        + "Context 1: CTX" + "\n\n"
        + "Question: Who?\nAnswer:"
    )
    assert format_qa_prompt("Who?", ["CTX"]) == expected


# --------------------------------------------------------------------------
# Decision 1B on the reference engine: chat prefix/suffix split seams
# --------------------------------------------------------------------------

def _fake_render_full(block_text: str):
    """Fake apply_chat_template renderer: header + system + user + gen header."""

    def render(question: str) -> str:
        user = f"Context 1: {block_text}\n\nQuestion: {question}"
        return (
            "<|im_start|>system\n" + CHAT_SYSTEM_INSTRUCTION + "<|im_end|>\n"
            "<|im_start|>user\n" + user + "<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

    return render


def test_reference_chat_prefix_ends_at_corpus_question_boundary():
    runner = _load_reference_runner()
    render = _fake_render_full("BLOCK TEXT")
    prefix = runner.compute_chat_prefix(render)
    assert prefix.endswith("Context 1: BLOCK TEXT")
    assert "Question:" not in prefix[prefix.index("BLOCK TEXT"):]

    full = render("Who?")
    suffix = runner.chat_query_suffix(full, prefix)
    assert prefix + suffix == full
    assert suffix.startswith("\n\nQuestion: Who?")
    assert suffix.endswith("<think>\n\n</think>\n\n")  # generation header in the suffix


def test_reference_chat_suffix_rejects_foreign_prefix():
    runner = _load_reference_runner()
    render = _fake_render_full("BLOCK TEXT")
    with pytest.raises(ValueError):
        runner.chat_query_suffix(render("Who?"), "NOT-A-PREFIX")


def test_reference_chat_messages_match_vllm_chat_arm_layout():
    runner = _load_reference_runner()
    assert runner.chat_messages_for_block("BLOCK", "Who?") == format_qa_messages(
        "Who?", ["BLOCK"]
    )


# --------------------------------------------------------------------------
# Decision 3B: distractor corpus -- determinism + gold exclusion (mocked loader)
# --------------------------------------------------------------------------

def _mock_loader_pool() -> List[FakeExample]:
    """Deterministic fake full-split pool: 30 examples over 10 paragraphs,
    with duplicate-content paragraphs interleaved (same paragraph shared by
    several questions, as in SQuAD)."""
    pool = []
    for p in range(10):
        para = f"paragraph-{p} " * 8
        for q in range(3):
            pool.append(FakeExample(id=f"ex_{p}_{q}", question=f"Q{p}.{q}?", context=[para]))
    return pool


def test_distractor_selection_is_deterministic():
    pool = _mock_loader_pool()
    first = select_distractor_texts(pool, [], 5)
    second = select_distractor_texts(pool, [], 5)
    assert first == second
    assert len(first) == 5
    # Load order respected: first distinct paragraphs win.
    assert first[0].startswith("paragraph-0")
    assert first[4].startswith("paragraph-4")


def test_distractor_selection_excludes_gold_paragraphs():
    pool = _mock_loader_pool()
    gold = [pool[0].context[0], pool[9].context[0]]  # paragraphs 0 and 3
    picked = select_distractor_texts(pool, gold, 100)
    assert all(g not in picked for g in gold)
    # 10 distinct paragraphs minus 2 excluded golds.
    assert len(picked) == 8


def test_distractor_selection_dedupes_by_content_and_caps_at_n():
    pool = _mock_loader_pool()
    everything = select_distractor_texts(pool, [], 1000)
    assert len(everything) == 10  # 30 examples but only 10 distinct paragraphs
    assert len(set(everything)) == 10
    assert select_distractor_texts(pool, [], 0) == []
    assert select_distractor_texts(pool, [], -3) == []


def test_distractor_selection_skips_empty_contexts():
    pool = [FakeExample(id="e0", question="q", context=["", "real para"])]
    assert select_distractor_texts(pool, [], 10) == ["real para"]
