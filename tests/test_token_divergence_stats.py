"""Tests for the charter D8 sec. 8.9 token-divergence statistics (2026-08-04 build).

Covers the three sec. 8.9 statistics added on top of the pre-existing string-divergence
outputs (which stay byte-compatible and are regression-tested in test_token_divergence.py):
  1. agreement rates (string + token stream),
  2. first-divergence token position,
  3. answer-changing vs answer-preserving classification (gold-based and pairwise), and
the per-cell reproducibility-violation rate across repetitions.

No GPU/network: the default whitespace tokenizer is pure Python and the HF-tokenizer
fail-closed test mocks the transformers import (transport-mock pattern, test_inference.py).
"""
from __future__ import annotations

import csv
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "4_analysis"))
import token_divergence as td  # noqa: E402


def _write_csv(path: Path, rows: list, fieldnames: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


_GOLD_FIELDS = ["example_id", "repeat_index", "generated_answer", "reference_answer", "prompt_tokens", "error"]


def _row(ex: str, answer: str, gold: str, rep: str = "1", pt: str = "100") -> dict:
    return {
        "example_id": ex,
        "repeat_index": rep,
        "generated_answer": answer,
        "reference_answer": gold,
        "prompt_tokens": pt,
        "error": "",
    }


# --------------------------------------------------------------------------- #
# First-divergence token position (unit level)
# --------------------------------------------------------------------------- #


def test_first_divergence_position_semantics():
    fdp = td._first_divergence_position
    assert fdp(["a", "b"], ["a", "b"]) is None  # identical -> no divergence
    assert fdp(["a", "b"], ["a", "c"]) == 1  # 0-based index of first differing token
    assert fdp(["x"], ["y"]) == 0
    assert fdp(["a", "b"], ["a", "b", "c"]) == 2  # strict prefix -> shorter length
    assert fdp([], ["a"]) == 0
    assert fdp([], []) is None


def test_make_tokenizer_whitespace_and_hf_fail_closed(monkeypatch):
    label, tok = td._make_tokenizer("whitespace")
    assert label == "whitespace"
    assert tok("The  capital is Paris") == ["The", "capital", "is", "Paris"]

    # HF tokenizer that cannot load must raise the typed error, never fall back
    # silently to whitespace (mirrors quality.py's InstrumentUnavailableError).
    stub = types.ModuleType("transformers")

    class _FailingAuto:
        @staticmethod
        def from_pretrained(name):
            raise OSError(f"no such model: {name}")

    stub.AutoTokenizer = _FailingAuto
    monkeypatch.setitem(sys.modules, "transformers", stub)
    with pytest.raises(td.TokenizerUnavailableError) as exc_info:
        td._make_tokenizer("org/nonexistent-model")
    assert "org/nonexistent-model" in str(exc_info.value)


def test_make_tokenizer_hf_uses_token_ids(monkeypatch):
    stub = types.ModuleType("transformers")

    class _StubTok:
        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return [ord(c) for c in text]  # deterministic fake token ids

    class _Auto:
        @staticmethod
        def from_pretrained(name):
            return _StubTok()

    stub.AutoTokenizer = _Auto
    monkeypatch.setitem(sys.modules, "transformers", stub)
    label, tok = td._make_tokenizer("org/stub-model")
    assert label == "hf:org/stub-model"
    assert tok("ab") == [97, 98]


# --------------------------------------------------------------------------- #
# Gold-based classification + first-divergence + agreement (end to end)
# --------------------------------------------------------------------------- #


def test_sec89_stats_gold_based(tmp_path: Path) -> None:
    ref_rows = [
        _row("e1", "Paris", "Paris"),
        _row("e2", "Paris", "Paris"),
        _row("e3", "Paris", "Paris"),
        _row("e4", "The capital is Paris", "Paris"),
        _row("e5", "unknown", ""),  # unanswerable item, reference abstains (correct)
    ]
    arm_rows = [
        _row("e1", "Paris", "Paris"),  # identical -> agree
        _row("e2", "Answer: Paris", "Paris"),  # scaffold reword -> answer-PRESERVING
        _row("e3", "London", "Paris"),  # EM flip -> answer-CHANGING
        _row("e4", "The capital is London", "Paris"),  # F1 flip at token 3 -> CHANGING
        _row("e5", "It is Paris", ""),  # abstention flip on unanswerable -> CHANGING
    ]
    _write_csv(tmp_path / "no_cache" / "trial_1" / "results.csv", ref_rows, _GOLD_FIELDS)
    _write_csv(tmp_path / "prefix_cache" / "trial_1" / "results.csv", arm_rows, _GOLD_FIELDS)

    summary = td.compute_divergence(str(tmp_path), "no_cache")
    assert summary["tokenizer"] == "whitespace"
    (arm,) = summary["arms"]
    assert arm["arm"] == "prefix_cache"
    assert arm["n_compared"] == 5

    # Backward-compatible string-divergence outputs still present and correct.
    assert arm["raw_divergent"] == 4
    assert arm["raw_divergence_rate"] == round(4 / 5, 4)

    # Stat 1: agreement rates.
    assert arm["agreement_rate"] == round(1 / 5, 4)
    assert arm["token_agreement_rate"] == round(1 / 5, 4)  # all 4 divergent pairs token-diverge

    # Stat 2: first-divergence positions: e2 -> 0, e3 -> 0, e4 -> 3, e5 -> 0.
    fd = arm["first_divergence"]
    assert fd["tokenizer"] == "whitespace"
    assert fd["n_raw_divergent"] == 4
    assert fd["n_token_divergent"] == 4
    assert fd["n_token_identical_divergent"] == 0
    assert fd["median_position"] == 0
    assert fd["max_position"] == 3
    assert fd["mean_position"] == pytest.approx(0.75)

    # Stat 3: answer-changing vs answer-preserving, gold basis.
    ad = arm["answer_divergence"]
    assert ad["n_classified"] == 4
    assert ad["answer_changing"] == 3  # e3 (EM), e4 (F1), e5 (abstention)
    assert ad["answer_preserving"] == 1  # e2: sanitize_answer strips the scaffold
    assert ad["answer_changing_rate"] == round(3 / 5, 4)
    assert ad["answer_changing_share_of_divergent"] == round(3 / 4, 4)
    assert ad["classification_basis"] == ["gold"]


def test_sec89_pairwise_fallback_without_gold_column(tmp_path: Path) -> None:
    fields = ["example_id", "repeat_index", "generated_answer", "error"]  # legacy schema
    _write_csv(tmp_path / "no_cache" / "trial_1" / "results.csv", [
        {"example_id": "e1", "repeat_index": "1", "generated_answer": "Paris.", "error": ""},
        {"example_id": "e2", "repeat_index": "1", "generated_answer": "Paris", "error": ""},
    ], fields)
    _write_csv(tmp_path / "prefix_cache" / "trial_1" / "results.csv", [
        # Punctuation-only reword: official normalization equates them -> PRESERVING.
        {"example_id": "e1", "repeat_index": "1", "generated_answer": "Paris", "error": ""},
        # Abstention flip -> CHANGING.
        {"example_id": "e2", "repeat_index": "1", "generated_answer": "I don't know", "error": ""},
    ], fields)

    summary = td.compute_divergence(str(tmp_path), "no_cache")
    (arm,) = summary["arms"]
    ad = arm["answer_divergence"]
    assert ad["classification_basis"] == ["pairwise"]
    assert ad["answer_changing"] == 1
    assert ad["answer_preserving"] == 1


def test_token_identical_divergent_pair_counts_as_token_agreement(tmp_path: Path) -> None:
    # Internal-whitespace-only difference: raw-divergent (strip() keeps inner spaces)
    # but token-identical under the whitespace tokenizer.
    _write_csv(tmp_path / "no_cache" / "trial_1" / "results.csv",
               [_row("e1", "Paris is  nice", "Paris")], _GOLD_FIELDS)
    _write_csv(tmp_path / "arm" / "trial_1" / "results.csv",
               [_row("e1", "Paris is nice", "Paris")], _GOLD_FIELDS)

    summary = td.compute_divergence(str(tmp_path), "no_cache")
    (arm,) = summary["arms"]
    assert arm["raw_divergent"] == 1
    assert arm["token_agreement_rate"] == 1.0
    fd = arm["first_divergence"]
    assert fd["n_token_divergent"] == 0
    assert fd["n_token_identical_divergent"] == 1
    assert fd["median_position"] is None


# --------------------------------------------------------------------------- #
# Per-cell reproducibility-violation rate (sec. 8.9 within-cell companion)
# --------------------------------------------------------------------------- #


def test_reproducibility_violation_rate_across_repeats(tmp_path: Path) -> None:
    ref_rows = [
        # e1: 3 repeats, all identical -> no violation.
        _row("e1", "Paris", "Paris", rep="1"),
        _row("e1", "Paris", "Paris", rep="2"),
        _row("e1", "Paris", "Paris", rep="3"),
        # e2: 3 repeats, one differs -> violation.
        _row("e2", "Paris", "Paris", rep="1"),
        _row("e2", "London", "Paris", rep="2"),
        _row("e2", "Paris", "Paris", rep="3"),
        # e3: single repeat -> excluded from the denominator.
        _row("e3", "Rome", "Rome", rep="1"),
    ]
    arm_rows = [
        # e1: 2 repeats differing only by punctuation: raw violation, NOT normalized.
        _row("e1", "Paris", "Paris", rep="1"),
        _row("e1", "Paris.", "Paris", rep="2"),
    ]
    _write_csv(tmp_path / "no_cache" / "trial_1" / "results.csv", ref_rows, _GOLD_FIELDS)
    _write_csv(tmp_path / "prefix_cache" / "trial_1" / "results.csv", arm_rows, _GOLD_FIELDS)

    summary = td.compute_divergence(str(tmp_path), "no_cache")
    repro = {r["arm"]: r for r in summary["reproducibility"]}
    assert set(repro) == {"no_cache", "prefix_cache"}

    ref_cell = repro["no_cache"]
    assert ref_cell["is_reference"] is True
    assert ref_cell["n_groups"] == 2  # e3's single repeat excluded
    assert ref_cell["n_violations"] == 1  # e2
    assert ref_cell["violation_rate"] == 0.5
    assert ref_cell["n_normalized_violations"] == 1  # Paris vs London survives normalization
    assert ref_cell["repeats_min"] == 3 and ref_cell["repeats_max"] == 3

    arm_cell = repro["prefix_cache"]
    assert arm_cell["n_groups"] == 1
    assert arm_cell["n_violations"] == 1  # raw: 'Paris' != 'Paris.'
    assert arm_cell["n_normalized_violations"] == 0  # punctuation-only, normalizes equal
    assert arm_cell["repeats_min"] == 2


def test_reproducibility_groups_are_per_trial(tmp_path: Path) -> None:
    # The same example_id in DIFFERENT trials must form separate groups (S5 keying).
    _write_csv(tmp_path / "no_cache" / "trial_1" / "results.csv",
               [_row("e1", "Paris", "Paris", rep="1"), _row("e1", "Paris", "Paris", rep="2")],
               _GOLD_FIELDS)
    _write_csv(tmp_path / "no_cache" / "trial_2" / "results.csv",
               [_row("e1", "London", "Paris", rep="1"), _row("e1", "London", "Paris", rep="2")],
               _GOLD_FIELDS)

    summary = td.compute_divergence(str(tmp_path), "no_cache")
    (ref_cell,) = summary["reproducibility"]
    # Two groups (e1@trial_1, e1@trial_2), each internally consistent -> zero violations,
    # even though the answers differ ACROSS trials.
    assert ref_cell["n_groups"] == 2
    assert ref_cell["n_violations"] == 0


# --------------------------------------------------------------------------- #
# Fail-closed wiring
# --------------------------------------------------------------------------- #


def test_unloadable_hf_tokenizer_fails_closed_end_to_end(tmp_path: Path, monkeypatch) -> None:
    _write_csv(tmp_path / "no_cache" / "trial_1" / "results.csv",
               [_row("e1", "Paris", "Paris")], _GOLD_FIELDS)

    stub = types.ModuleType("transformers")

    class _FailingAuto:
        @staticmethod
        def from_pretrained(name):
            raise OSError("offline / model missing")

    stub.AutoTokenizer = _FailingAuto
    monkeypatch.setitem(sys.modules, "transformers", stub)
    with pytest.raises(td.TokenizerUnavailableError):
        td.compute_divergence(str(tmp_path), "no_cache", tokenizer="org/missing")
