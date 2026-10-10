"""Review 2026-10-10, F-01: the HF oracle's reuse path must serve the SAME
token ids as the plain path.

Measured on the Qwen3-14B tokenizer (2026-10-10, local cache): the whole
prompt ``...mat.\\n\\nQuestion: ...`` ends the block with ONE token ``.ĊĊ``,
while the old recipe (prefix cut before the ``\\n\\n``, suffix tokenized on its
own) produced ``.`` then ``ĊĊ``: 22 ids against 21. Cutting the prefix AFTER
the two newlines gives identical ids. These tests pin both halves of the fix:

1. the runner's ``derive_corpus_prompt_prefix`` and the CAG reference
   script's seams end the prefix after the ``\\n\\n`` (pure, no tokenizer);
2. on the real Qwen3 tokenizer (skipped when it is not in the local
   Hugging Face cache; never downloaded here) the prefix + suffix tokenized
   separately equals the whole prompt, which the old cut violates.

The adapter-side check (the full prompt's leading ids must equal the cached
ids) is exercised on the tiny real-stack model in
tests/test_hf_oracle_full_ids_and_stop_s0f60.py.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.utils.prompting import format_qa_messages, format_qa_prompt, messages_to_fallback_prompt  # noqa: E402


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, str(REPO / path))
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


runner = _load("scripts/3_run/run_experiment.py", "cage_run_experiment_f01")
reference = _load("scripts/3_run/run_cag_reference.py", "cage_cag_reference_f01")

QUESTION = "where did the cat sit?"


# ---------------------------------------------------------------------------
# 1. the prefix cut, pure
# ---------------------------------------------------------------------------


def test_runner_prefix_ends_after_the_double_newline_raw_layout() -> None:
    ctxs = ["The cat sat on the mat."]
    prefix = runner.derive_corpus_prompt_prefix(lambda q: format_qa_prompt(q, ctxs))
    full = format_qa_prompt(QUESTION, ctxs)
    assert full.startswith(prefix)
    assert prefix.endswith("\n\n")
    assert full[len(prefix):].startswith("Question:")


def test_runner_prefix_ends_after_the_double_newline_chat_fallback_layout() -> None:
    ctxs = ["The cat sat on the mat."]

    def build(q: str) -> str:
        return messages_to_fallback_prompt(format_qa_messages(q, ctxs))

    prefix = runner.derive_corpus_prompt_prefix(build)
    full = build(QUESTION)
    assert full.startswith(prefix)
    assert prefix.endswith("\n\n")
    assert full[len(prefix):].startswith("Question:")


def test_reference_script_seams_share_the_boundary() -> None:
    assert reference.PREFIX_BOUNDARY == "\n\n"
    prompt = reference.build_corpus_prompt("BLOCK")
    assert prompt.endswith("\n\n")
    suffix = reference.build_query_suffix("Who?")
    assert suffix == "Question: Who?\nAnswer:"
    # the concatenation is the format_qa_prompt layout, unchanged
    full = prompt + suffix
    assert full == format_qa_prompt("Who?", ["CTX"]).replace("Context 1: CTX", "BLOCK")

    def render(q: str) -> str:
        return "<|im_start|>user\nContext 1: BLOCK\n\nQuestion: " + q + "<|im_end|>\n"

    chat_prefix = reference.compute_chat_prefix(render)
    assert chat_prefix.endswith("Context 1: BLOCK\n\n")
    assert reference.chat_query_suffix(render("Who?"), chat_prefix).startswith("Question: Who?")


# ---------------------------------------------------------------------------
# 2. the real tokenizer (local cache only)
# ---------------------------------------------------------------------------


def _qwen3_tokenizer():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", local_files_only=True)
    except Exception as exc:  # not cached on this machine: never download in a test
        pytest.skip(f"Qwen/Qwen3-14B tokenizer not in the local cache ({type(exc).__name__})")


def test_qwen3_merges_across_the_old_cut_and_not_across_the_new_one() -> None:
    tok = _qwen3_tokenizer()
    ctxs = ["The cat sat on the mat."]
    full = format_qa_prompt(QUESTION, ctxs)
    whole = tok(full, add_special_tokens=False).input_ids
    # the old cut (before the newlines) splits a merged token
    old_cut = full.rfind("\n\nQuestion:")
    old_prefix, old_suffix = full[:old_cut], full[old_cut:]
    split_old = (
        tok(old_prefix, add_special_tokens=False).input_ids
        + tok(old_suffix, add_special_tokens=False).input_ids
    )
    assert split_old != whole, "the defect no longer reproduces on this tokenizer; re-check the cut"
    # the new cut (the runner's derivation) tokenizes like the whole prompt
    prefix = runner.derive_corpus_prompt_prefix(lambda q: format_qa_prompt(q, ctxs))
    split_new = (
        tok(prefix, add_special_tokens=False).input_ids
        + tok(full[len(prefix):], add_special_tokens=False).input_ids
    )
    assert split_new == whole
    # and the preload tokenization (default special tokens) is a prefix of the whole
    pre = tok(prefix).input_ids
    assert tok(full).input_ids[: len(pre)] == pre


def test_reference_script_boundary_tokenizes_like_the_whole_prompt() -> None:
    tok = _qwen3_tokenizer()
    for block in ("The cat sat on the mat.", "Numbers end here 42", "Quoted \"text\"", "trailing space "):
        prompt = reference.build_corpus_prompt(block)
        suffix = reference.build_query_suffix(QUESTION)
        whole = tok(prompt + suffix).input_ids
        pre = tok(prompt).input_ids
        assert whole[: len(pre)] == pre, block
        assert whole == pre + tok(suffix, add_special_tokens=False).input_ids, block
