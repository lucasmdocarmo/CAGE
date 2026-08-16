"""Unit tests for run_experiment.py's qa_evidence.jsonl evidence writer.

Task #127 (walkthrough audit Topic 8, H2/H3/H10/H11 — sharpens #119 F2): the
evidence chain feeding decoupled scoring was previously untested. Covered here:

1. ``evidence_integrity_fields``: every evidence row is stamped with ok /
   error / empty_generation using the SAME validity semantics as the results
   row (so offline scoring can null error rows instead of scoring empty
   answers as hard zeros), plus record_index / arrival_s open-loop replay
   disambiguators (None on closed-loop rows — absence is not index 0).
2. ``append_evidence_row`` + ``count_evidence_failure``: an append failure
   never kills the run but is COUNTED (charter §9.10: exclusions countable
   from artifacts) and printed once per trial, not per row.
3. ``_json_default``: numpy scalars/arrays serialize as native JSON numbers/
   lists, never quoted strings (the old ``default=str`` drift).
4. ``write_json_atomic``: metrics.json — the completeness sentinel resume
   gates key on — is written via tmp + os.replace; a crash mid-write leaves
   the previous file intact, never a truncated sentinel (audit H10).
5. Source-level wiring pins: record_result routes its evidence block through
   the stamp + counted append, and both metrics.json writes are atomic.

The runner module is loaded via importlib (pattern of
tests/test_integration_wiring.py) because "3_run" is not a valid package name.
No GPU, no network, no engine.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts" / "3_run" / "run_experiment.py"

sys.path.insert(0, str(REPO_ROOT))

from src.orchestration.load_generator import RequestRecord  # noqa: E402


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "cage_run_experiment_evidence", RUNNER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def _failures() -> dict[str, Any]:
    return {"count": 0, "first_error": None}


def _record(index: int, offset_s: float) -> RequestRecord:
    return RequestRecord(index=index, scheduled_offset_s=offset_s, scheduled_ts=0.0)


# ---------------------------------------------------------------------------
# 1. evidence_integrity_fields — schema + semantics
# ---------------------------------------------------------------------------


class TestIntegrityFields:
    def test_ok_row(self) -> None:
        fields = runner.evidence_integrity_fields(error=None, generated_text="Paris")
        assert fields == {
            "ok": True,
            "error": None,
            "empty_generation": False,
            "record_index": None,
            "arrival_s": None,
        }

    def test_error_row(self) -> None:
        """A serving error is NOT ok and is NOT flagged empty_generation —
        exactly the results-row semantics (empty flag only on non-error rows)."""
        fields = runner.evidence_integrity_fields(
            error="HTTP 500: server died", generated_text=""
        )
        assert fields["ok"] is False
        assert fields["error"] == "HTTP 500: server died"
        assert fields["empty_generation"] is False

    def test_empty_generation_row(self) -> None:
        """Degenerate empty answer (e.g. leading newline under stop=['\\n']):
        not an error, but not ok — the loaders' shared validity predicate."""
        fields = runner.evidence_integrity_fields(error=None, generated_text="\n  ")
        assert fields["ok"] is False
        assert fields["error"] is None
        assert fields["empty_generation"] is True

    def test_none_generated_text_counts_as_empty(self) -> None:
        fields = runner.evidence_integrity_fields(error=None, generated_text=None)
        assert fields["empty_generation"] is True
        assert fields["ok"] is False

    def test_empty_string_error_normalizes_to_none(self) -> None:
        """error='' is falsy everywhere the results row is consumed; the
        evidence field normalizes it to None (the error string OR None)."""
        fields = runner.evidence_integrity_fields(error="", generated_text="fine")
        assert fields["error"] is None
        assert fields["ok"] is True

    def test_replay_disambiguation(self) -> None:
        """Open-loop replays (schedule index maps modulo over prepared
        requests) share example_id; record_index + arrival_s tell them apart."""
        first = runner.evidence_integrity_fields(
            error=None, generated_text="a", open_loop_record=_record(3, 0.5)
        )
        second = runner.evidence_integrity_fields(
            error=None, generated_text="a", open_loop_record=_record(9, 2.5)
        )
        assert first["record_index"] == 3
        assert first["arrival_s"] == 0.5
        assert second["record_index"] == 9
        assert second["arrival_s"] == 2.5
        assert (first["record_index"], first["arrival_s"]) != (
            second["record_index"],
            second["arrival_s"],
        )

    def test_record_index_zero_is_preserved(self) -> None:
        """Index 0 is a real schedule position — must never collapse to None."""
        fields = runner.evidence_integrity_fields(
            error=None, generated_text="a", open_loop_record=_record(0, 0.0)
        )
        assert fields["record_index"] == 0
        assert fields["arrival_s"] == 0.0


# ---------------------------------------------------------------------------
# 2. append_evidence_row / count_evidence_failure — countable, loud-once loss
# ---------------------------------------------------------------------------


class TestAppendEvidenceRow:
    def test_appends_and_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "trial_1" / "qa_evidence.jsonl"
        failures = _failures()
        assert runner.append_evidence_row(str(path), {"example_id": "e0"}, failures)
        assert runner.append_evidence_row(str(path), {"example_id": "e1"}, failures)
        rows = [json.loads(l) for l in path.read_text().splitlines()]
        assert [r["example_id"] for r in rows] == ["e0", "e1"]
        assert failures == {"count": 0, "first_error": None}

    def test_failure_counted_and_printed_once(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Two failed appends: count reaches 2, first_error is kept from the
        FIRST failure, and the loud message prints exactly once per trial."""
        path = tmp_path / "qa_evidence.jsonl"
        path.mkdir()  # open(path, "a") now raises IsADirectoryError
        failures = _failures()
        assert not runner.append_evidence_row(str(path), {"example_id": "e0"}, failures)
        assert not runner.append_evidence_row(str(path), {"example_id": "e1"}, failures)
        assert failures["count"] == 2
        assert failures["first_error"] is not None
        assert "IsADirectoryError" in failures["first_error"]
        out = capsys.readouterr().out
        assert out.count("qa_evidence.jsonl append FAILED") == 1
        assert "evidence_write_failures" in out

    def test_row_construction_failure_uses_same_counter(self) -> None:
        failures = _failures()
        runner.count_evidence_failure(failures, KeyError("question"))
        assert failures["count"] == 1
        assert failures["first_error"] == "KeyError: 'question'"


# ---------------------------------------------------------------------------
# 3. _json_default — numpy-native serialization (no quoted numbers)
# ---------------------------------------------------------------------------


class TestJsonDefault:
    def test_numpy_scalars_and_arrays_become_native(self, tmp_path: Path) -> None:
        row = {
            "example_id": "e0",
            "f32": np.float32(0.5),
            "f64": np.float64(0.25),
            "i64": np.int64(3),
            "flag": np.bool_(True),
            "arr": np.array([1, 2, 3]),
        }
        path = tmp_path / "qa_evidence.jsonl"
        assert runner.append_evidence_row(str(path), row, _failures())
        parsed = json.loads(path.read_text().splitlines()[0])
        assert parsed["f32"] == 0.5 and isinstance(parsed["f32"], float)
        assert parsed["f64"] == 0.25 and isinstance(parsed["f64"], float)
        assert parsed["i64"] == 3 and isinstance(parsed["i64"], int)
        assert parsed["flag"] is True
        assert parsed["arr"] == [1, 2, 3]

    def test_non_numpy_objects_still_fall_back_to_str(self) -> None:
        class Opaque:
            def __str__(self) -> str:
                return "opaque-repr"

        encoded = json.dumps({"x": Opaque()}, default=runner._json_default)
        assert json.loads(encoded) == {"x": "opaque-repr"}


# ---------------------------------------------------------------------------
# 4. write_json_atomic — the completeness sentinel is valid-or-absent
# ---------------------------------------------------------------------------


class TestWriteJsonAtomic:
    def test_writes_valid_json_and_cleans_tmp(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.json"
        runner.write_json_atomic(path, {"quality": {"f1_score": 0.5}})
        assert json.loads(path.read_text()) == {"quality": {"f1_score": 0.5}}
        assert not (tmp_path / "metrics.json.tmp").exists()

    def test_overwrites_previous_sentinel(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.json"
        runner.write_json_atomic(path, {"v": 1})
        runner.write_json_atomic(path, {"v": 2})
        assert json.loads(path.read_text()) == {"v": 2}

    def test_crash_mid_write_leaves_previous_file_intact(self, tmp_path: Path) -> None:
        """A serialization crash (stand-in for any mid-write death) must not
        truncate the existing sentinel: the old complete metrics.json survives."""
        path = tmp_path / "metrics.json"
        runner.write_json_atomic(path, {"v": 1})
        with pytest.raises(TypeError):
            runner.write_json_atomic(path, {"v": {1, 2}})  # sets are unserializable
        assert json.loads(path.read_text()) == {"v": 1}


# ---------------------------------------------------------------------------
# 5. Source-level wiring pins (record_result and the metrics writes are inside
#    run_experiment()'s closure — unreachable without a live engine, so the
#    wiring is pinned at source level; behavior is unit-tested above)
# ---------------------------------------------------------------------------


SOURCE = RUNNER_PATH.read_text(encoding="utf-8")


def test_record_result_evidence_block_is_stamped_and_counted() -> None:
    evidence_block = SOURCE[SOURCE.index("_evidence = {"):]
    head = evidence_block[:6000]
    assert "evidence_integrity_fields(" in head
    assert 'append_evidence_row(\n                os.path.join(output_dir, "qa_evidence.jsonl")' in head
    # the old fire-and-forget writer is gone: no direct dump with default=str
    assert 'json.dumps(_evidence, default=str)' not in SOURCE


def test_metrics_json_writes_are_atomic_and_loss_is_persisted() -> None:
    assert "write_json_atomic(metrics_file, experiment_summary)" in SOURCE
    assert "write_json_atomic(stable_metrics_file, experiment_summary)" in SOURCE
    assert "json.dump(experiment_summary" not in SOURCE
    # §9.10 CONSORT section persists both the guard drops and the evidence loss.
    assert '"consort": {' in SOURCE
    assert '"evidence_write_failures": evidence_failures["count"]' in SOURCE
    assert '"n_dropped_prepare"' in SOURCE
    assert '"n_dropped_record"' in SOURCE
