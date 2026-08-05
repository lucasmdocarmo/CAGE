"""Regression tests for the core-area review findings (2026-08-04 fixer pass).

Covers:
- scripts/3_run/run_experiment.py: reranker init must be fail-closed by
  default (a silently-None reranker used to collapse B6/B7/B8/B9/B11's ranked
  pipeline into B5's unranked one with nothing checking it live).
- requirements.txt: huggingface-hub must carry an upper bound, since
  huggingface-hub>=1.20 breaks load_dataset() for every legacy (non-
  namespaced) HF dataset id this repo uses.

The run_experiment.py module is loaded via importlib (like
tests/test_cag_reference_runner.py) rather than a package import, since its
directory ("3_run") is not a valid Python package-name component.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts" / "3_run" / "run_experiment.py"
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_experiment", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


# --------------------------------------------------------------------------
# build_reranker: fail-closed by default (CAGE_ALLOW_NO_RERANK opt-out only)
# --------------------------------------------------------------------------


class _FakeCrossEncoderReranker:
    """Stand-in for src.orchestration.ir.CrossEncoderReranker."""

    def __init__(self, model_name, *, device="cpu"):
        self.model_name = model_name
        self.device = device


class _BrokenCrossEncoderReranker:
    """Simulates a missing dependency / OOM / bad model name / network failure."""

    def __init__(self, model_name, *, device="cpu"):
        raise ModuleNotFoundError("No module named 'sentence_transformers'")


def test_build_reranker_returns_none_when_model_not_requested():
    assert runner.build_reranker(None, "cpu") is None
    assert runner.build_reranker("", "cpu") is None


def test_build_reranker_succeeds_and_returns_instance(monkeypatch):
    monkeypatch.setattr(runner, "CrossEncoderReranker", _FakeCrossEncoderReranker)
    reranker = runner.build_reranker("BAAI/bge-reranker-large", "cpu")
    assert isinstance(reranker, _FakeCrossEncoderReranker)
    assert reranker.model_name == "BAAI/bge-reranker-large"


def test_build_reranker_is_fail_closed_by_default(monkeypatch):
    """A failed reranker init must ABORT the run, not silently return None.

    Before the fix, run_experiment.py caught the exception, printed a
    warning, and left reranker=None -- collapsing B6/B7/B8/B9/B11's ranked
    pipeline into B5's unranked one for the whole run with no live check.
    """
    monkeypatch.setattr(runner, "CrossEncoderReranker", _BrokenCrossEncoderReranker)
    monkeypatch.delenv("CAGE_ALLOW_NO_RERANK", raising=False)

    with pytest.raises(RuntimeError, match="CAGE requires the reranker"):
        runner.build_reranker("BAAI/bge-reranker-large", "cpu")


@pytest.mark.parametrize("flag_value", ["1", "true", "TRUE", "yes"])
def test_build_reranker_allows_explicit_opt_out(monkeypatch, flag_value):
    """CAGE_ALLOW_NO_RERANK=1 is the only sanctioned escape hatch."""
    monkeypatch.setattr(runner, "CrossEncoderReranker", _BrokenCrossEncoderReranker)
    monkeypatch.setenv("CAGE_ALLOW_NO_RERANK", flag_value)

    reranker = runner.build_reranker("BAAI/bge-reranker-large", "cpu")
    assert reranker is None


def test_build_reranker_opt_out_requires_truthy_value(monkeypatch):
    """An unset/empty/falsy CAGE_ALLOW_NO_RERANK must NOT silence the failure."""
    monkeypatch.setattr(runner, "CrossEncoderReranker", _BrokenCrossEncoderReranker)
    monkeypatch.setenv("CAGE_ALLOW_NO_RERANK", "0")

    with pytest.raises(RuntimeError):
        runner.build_reranker("BAAI/bge-reranker-large", "cpu")


# --------------------------------------------------------------------------
# requirements.txt: huggingface-hub must have an upper bound below 1.20
# --------------------------------------------------------------------------


def _requirements_text() -> str:
    return REQUIREMENTS_PATH.read_text()


def _huggingface_hub_line() -> str:
    text = _requirements_text()
    match = re.search(r"^huggingface-hub\S*$", text, flags=re.MULTILINE)
    assert match, "huggingface-hub pin not found in requirements.txt"
    return match.group(0)


def _version_tuple(v: str) -> tuple:
    return tuple(int(p) for p in v.split("."))


def _satisfies(line: str, version: str) -> bool:
    """Minimal >=X,<Y specifier check (no external dependency needed).

    requirements.txt only ever needs simple floor/ceiling clauses for this
    pin, so a small hand-parser keeps this test self-contained rather than
    depending on a test-only `packaging` import that isn't itself declared
    in requirements.txt.
    """
    ok = True
    for op, bound in re.findall(r"(>=|<=|<|>|==)([0-9][0-9.]*)", line):
        v, b = _version_tuple(version), _version_tuple(bound)
        if op == ">=":
            ok &= v >= b
        elif op == ">":
            ok &= v > b
        elif op == "<=":
            ok &= v <= b
        elif op == "<":
            ok &= v < b
        elif op == "==":
            ok &= v == b
    return ok


def test_requirements_huggingface_hub_has_upper_bound():
    """An unbounded huggingface-hub pin resolves to a version that breaks
    load_dataset() for every legacy (non-namespaced) HF dataset id this repo
    uses (squad_v2, hotpot_qa, trivia_qa, nq_open, mbpp, openai_humaneval)
    with HfUriError while resolving the dataset's hf:// metadata URI.
    """
    line = _huggingface_hub_line()
    assert "<" in line, (
        f"requirements.txt huggingface-hub pin {line!r} has no upper bound; "
        "a fresh install can silently resolve to a version that breaks every "
        "legacy-named HF dataset loader in this repo."
    )


def test_requirements_huggingface_hub_excludes_known_broken_versions():
    line = _huggingface_hub_line()
    # Empirically verified broken (this session): HfUriError on squad_v2/hotpot_qa.
    assert not _satisfies(line, "1.20.0"), (
        "huggingface-hub 1.20.0 is a verified-broken version for this repo's "
        "legacy dataset ids; the requirements.txt bound must exclude it."
    )
    assert not _satisfies(line, "1.26.0")


def test_requirements_huggingface_hub_allows_known_good_versions():
    line = _huggingface_hub_line()
    # Empirically verified working (this session), and required by this repo's
    # own pinned vLLM 0.19.1 (transformers>=4.56.0 -> huggingface-hub>=0.34.0).
    for good_version in ("0.23.5", "0.34.0", "1.0.0", "1.10.0"):
        assert _satisfies(line, good_version), (
            f"huggingface-hub {good_version} should satisfy the requirements.txt "
            f"bound {line!r} (verified working / required by vLLM 0.19.1)."
        )


# --------------------------------------------------------------------------
# Corpus-prefix fallback branch: gold-only packing (parity with manifest path)
# --------------------------------------------------------------------------
#
# The non-manifest CAGE_CORPUS_PREFIX_BUDGET branch used to pass raw .context
# (gold + distractors on HotpotQA/MuSiQue) into build_corpus_block, wasting
# corpus budget on distractor text, and labeled metadata["gold_context"] with
# context[0] -- an arbitrary, possibly-distractor paragraph.


def test_corpus_fallback_branch_uses_gold_only_filter():
    src = RUNNER_PATH.read_text(encoding="utf-8")
    m = re.search(r"elif _corpus_budget > 0:(.*?)CORPUS-PREFIX mode:", src, re.DOTALL)
    assert m, "non-manifest corpus-prefix branch not found in run_experiment.py"
    branch = m.group(1)
    assert "gold_only" in branch, (
        "the corpus-prefix fallback branch must pack the block from gold_only() "
        "context (parity with the manifest path in src/data/manifest.py); raw "
        ".context includes HotpotQA/MuSiQue distractors"
    )
    assert "_gold_ctx[ex.id] or [None]" in branch, (
        "metadata['gold_context'] must come from the gold-filtered context, not "
        "context[0], which can be a distractor paragraph"
    )


def test_gold_only_composed_with_corpus_block_excludes_distractors():
    from src.data.corpus import build_corpus_block
    from src.data.loader import CAGExample, gold_only

    ex = CAGExample(
        id="q1",
        question="who?",
        context=["GoldTitle: the gold paragraph text", "Noise: distractor text"],
        answer="a",
        metadata={"supporting_titles": ["GoldTitle"]},
    )
    filtered = CAGExample(
        id=ex.id, question=ex.question, context=gold_only(ex),
        answer=ex.answer, metadata=ex.metadata,
    )
    block = build_corpus_block([filtered], token_budget=10_000)
    assert "the gold paragraph text" in block.text
    assert "distractor text" not in block.text
    assert "q1" in block.example_ids
