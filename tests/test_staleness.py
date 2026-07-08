"""Unit tests for the staleness/freshness baseline helpers.

Stdlib-only: `src/evaluation/staleness.py` has no model or GPU dependencies, so these run
anywhere pytest does. They pin the deterministic version-sweep contract the baseline relies on.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.staleness import (  # noqa: E402
    select_stale,
    make_stale_context,
    staleness_metrics,
)


def test_select_stale_bounds():
    assert select_stale("q1", 0.0) is False
    assert select_stale("q1", 1.0) is True


def test_select_stale_deterministic():
    a = select_stale("q-abc", 0.5, seed=42)
    b = select_stale("q-abc", 0.5, seed=42)
    assert a == b


def test_select_stale_fraction_approximate():
    n = 4000
    stale = sum(1 for i in range(n) if select_stale(f"id-{i}", 0.5, seed=7))
    frac = stale / n
    assert 0.45 <= frac <= 0.55, f"expected ~0.5, got {frac}"


def test_make_stale_context_redacts_answer():
    ctx = ["The capital of France is Paris.", "Unrelated sentence."]
    out = make_stale_context(ctx, "Paris")
    assert "[redacted]" in out[0]
    assert "Paris" not in out[0]
    assert out[1] == "Unrelated sentence."


def test_make_stale_context_drops_first_when_answer_absent():
    ctx = ["first doc", "second doc"]
    out = make_stale_context(ctx, "not-present")
    assert out == ["second doc"]


def test_make_stale_context_single_doc_unchanged_when_absent():
    ctx = ["only doc"]
    out = make_stale_context(ctx, "not-present")
    assert out == ["only doc"]


def test_staleness_metrics_known_values():
    records = [
        {"served_from_cache": True, "grounded": False, "evidence_version": "v0"},
        {"served_from_cache": True, "grounded": True, "evidence_version": "v1"},
        {"served_from_cache": True, "grounded": False, "evidence_version": "v0"},
        {"served_from_cache": False, "grounded": True, "evidence_version": None},
    ]
    m = staleness_metrics(records)
    assert m["unsafe_served_rate"] == 0.5           # 2 unsafe / 4 total
    assert m["answer_hit_rate"] == 0.75             # 3 served / 4 total
    assert abs(m["false_hit_rate"] - (2 / 3)) < 1e-9  # 2 wrong / 3 served
    assert m["stale_hit_rate"] == 1.0               # 2 ungrounded / 2 served v0


def test_staleness_metrics_empty_is_none():
    m = staleness_metrics([])
    assert m["unsafe_served_rate"] is None
    assert m["stale_hit_rate"] is None
