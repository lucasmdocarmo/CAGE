"""Token-divergence metric (fix #6): raw vs normalized divergence, error-row exclusion."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import token_divergence as td  # noqa: E402


def _write_csv(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["example_id", "repeat_index", "generated_answer", "error"])
        w.writeheader()
        w.writerows(rows)


def test_divergence_raw_vs_normalized_and_error_skip(tmp_path: Path) -> None:
    _write_csv(tmp_path / "no_cache" / "trial_1" / "results.csv", [
        {"example_id": "e1", "repeat_index": "1", "generated_answer": "Paris", "error": ""},
        {"example_id": "e2", "repeat_index": "1", "generated_answer": "London", "error": ""},
        {"example_id": "e3", "repeat_index": "1", "generated_answer": "Rome", "error": ""},
    ])
    _write_csv(tmp_path / "prefix_cache" / "trial_1" / "results.csv", [
        {"example_id": "e1", "repeat_index": "1", "generated_answer": "Paris", "error": ""},        # identical
        {"example_id": "e2", "repeat_index": "1", "generated_answer": "London.", "error": ""},      # raw-only diff (punct)
        {"example_id": "e3", "repeat_index": "1", "generated_answer": "Berlin", "error": ""},       # raw + norm diff
        {"example_id": "e9", "repeat_index": "1", "generated_answer": "ignored", "error": "Timeout"},  # error -> skipped
    ])

    summary = td.compute_divergence(str(tmp_path), "no_cache")
    assert summary["reference"] == "no_cache"
    arms = {a["arm"]: a for a in summary["arms"]}
    assert set(arms) == {"prefix_cache"}
    pc = arms["prefix_cache"]
    assert pc["n_compared"] == 3                       # e9 excluded (error), e1/e2/e3 matched
    assert pc["raw_divergent"] == 2                    # e2 (punct) + e3
    assert pc["raw_divergence_rate"] == round(2 / 3, 4)
    assert pc["normalized_divergent"] == 1             # only e3 survives normalization
    assert pc["normalized_divergence_rate"] == round(1 / 3, 4)


def test_missing_reference_raises(tmp_path: Path) -> None:
    _write_csv(tmp_path / "prefix_cache" / "trial_1" / "results.csv", [
        {"example_id": "e1", "repeat_index": "1", "generated_answer": "x", "error": ""},
    ])
    import pytest

    with pytest.raises(FileNotFoundError):
        td.compute_divergence(str(tmp_path), "no_cache")
