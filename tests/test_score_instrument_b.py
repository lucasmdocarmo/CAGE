"""CLI plumbing tests for scripts/4_analysis/score_instrument_b.py (D8 §8.5).

The manager (``src.evaluation.instrument_b_runner``) is monkeypatched: no
isolated env, no downloads, no worker subprocess. What IS exercised for real:
argument plumbing into the manager, evidence parsing (incl. the stringified
context tolerance and the empty-row skip), the REAL ``apply_tau`` (boundary
semantics), the sidecar provenance block, and the RESULTS_LAYOUT §6
scoring-tree layout with its fail-closed preconditions and own ledger.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = REPO_ROOT / "scripts" / "4_analysis"
for _p in (str(_SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import score_instrument_b as sib  # noqa: E402
from src.analysis.stats.ledger import (  # noqa: E402
    hash_artifacts,
    verify_ledger,
    write_ledger,
)

EVIDENCE_ROWS: list[dict[str, Any]] = [
    {
        "example_id": "e0",
        "repeat_index": 0,
        "used_contexts": ["The capital of France is Paris."],
        "generated_answer": "Paris.",
    },
    {
        "example_id": "e1",
        "repeat_index": 0,
        # stringified context list: older-run tolerance (mirrors rescore_quality)
        "used_contexts": json.dumps(["London is the capital of the UK."]),
        "generated_answer": "London.",
    },
    {
        "example_id": "e2",
        "repeat_index": 0,
        "used_contexts": ["Some context."],
        "generated_answer": "",  # unscoreable: skipped, never a null score
    },
    {
        "example_id": "e3",
        "repeat_index": 1,
        "used_contexts": ["Rome is the capital of Italy."],
        "generated_answer": "Rome.",
    },
]

#: id -> alignscore returned by the fake manager (keyed by example id part).
FAKE_SCORES = {"e0": 0.9, "e1": 0.5, "e3": 0.2}  # e1 sits EXACTLY at τ=0.5


def _write_evidence(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in EVIDENCE_ROWS), encoding="utf-8"
    )
    return path


@pytest.fixture()
def fake_manager(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Monkeypatch ib.score/ib.ensure_env, recording every call."""
    recorded: dict[str, Any] = {"score_calls": [], "ensure_calls": []}

    def fake_score(
        items: list[dict[str, str]], **kwargs: Any
    ) -> dict[str, float]:
        recorded["score_calls"].append({"items": list(items), **kwargs})
        scores: dict[str, float] = {}
        for item in items:
            example_part = item["id"].split("::")[1]
            scores[item["id"]] = FAKE_SCORES[example_part]
        return scores

    def fake_ensure_env(env_home: Any = None, *args: Any, **kwargs: Any) -> Path:
        recorded["ensure_calls"].append(env_home)
        return Path(env_home) if env_home is not None else Path("/fake/home")

    monkeypatch.setattr(sib.ib, "score", fake_score)
    monkeypatch.setattr(sib.ib, "ensure_env", fake_ensure_env)
    return recorded


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# --------------------------------------------------------------------------- #
# --bootstrap-only
# --------------------------------------------------------------------------- #


class TestBootstrapOnly:
    def test_bootstraps_and_scores_nothing(
        self, tmp_path: Path, fake_manager: dict[str, Any]
    ) -> None:
        rc = sib.main(["--bootstrap-only", "--env-home", str(tmp_path / "eh")])
        assert rc == 0
        assert fake_manager["ensure_calls"] == [tmp_path / "eh"]
        assert fake_manager["score_calls"] == []


# --------------------------------------------------------------------------- #
# Flat mode
# --------------------------------------------------------------------------- #


class TestFlatMode:
    def test_rows_tau_and_sidecar(
        self, tmp_path: Path, fake_manager: dict[str, Any]
    ) -> None:
        ev = _write_evidence(tmp_path / "trial_0" / "qa_evidence.jsonl")
        out = tmp_path / "ib_scores.jsonl"
        rc = sib.main([
            "--evidence", str(ev),
            "--out", str(out),
            "--tau", "0.5",
            "--env-home", str(tmp_path / "eh"),
            "--batch", "4",
            "--max-items", "7",
            "--device", "mps",
        ])
        assert rc == 0

        # -- manager plumbing: every CLI knob reaches score() -------------- #
        assert len(fake_manager["score_calls"]) == 1
        call = fake_manager["score_calls"][0]
        assert call["batch_size"] == 4
        assert call["max_items"] == 7
        assert call["device"] == "mps"
        assert call["env_home"] == tmp_path / "eh"
        # e2 (empty answer) never reached the manager
        assert [i["id"].split("::")[1] for i in call["items"]] == ["e0", "e1", "e3"]
        # stringified contexts were parsed, not scored as a JSON blob
        e1_item = call["items"][1]
        assert e1_item["context"] == "London is the capital of the UK."
        # id carries file key + example id + repeat index
        assert call["items"][2]["id"] == f"{ev.as_posix()}::e3::1"

        # -- output rows: real apply_tau, boundary INCLUSIVE at τ ---------- #
        rows = _read_jsonl(out)
        by_example = {r["id"].split("::")[1]: r for r in rows}
        assert set(by_example) == {"e0", "e1", "e3"}
        assert by_example["e0"] == {
            "id": f"{ev.as_posix()}::e0::0", "alignscore": 0.9, "grounded_b": True,
        }
        assert by_example["e1"]["grounded_b"] is True  # score exactly at τ
        assert by_example["e3"]["grounded_b"] is False

        # -- sidecar provenance -------------------------------------------- #
        sidecar = json.loads(
            Path(str(out) + ".provenance.json").read_text(encoding="utf-8")
        )
        assert sidecar["instrument"] == "alignscore_large"
        assert sidecar["citation"] == "zha2023alignscore"
        assert sidecar["tau"] == 0.5
        # An explicit --tau is recorded as an override of the registered value.
        assert sidecar["tau_source"] == "override"
        assert sidecar["tau_registered"] == sib.ib.TAU_REGISTERED
        assert sidecar["tau_anchor_scope"] == sib.ib.TAU_ANCHOR_SCOPE
        assert sidecar["spec_fingerprint"] == sib.ib.spec_fingerprint(sib.ib.SPEC)
        assert sidecar["spec"]["model_revision"] == sib.ib.SPEC.model_revision
        assert sidecar["n_items"] == 3
        assert sidecar["n_scored"] == 3
        assert sidecar["n_skipped_empty"] == 1
        assert sidecar["evidence_files"] == [str(ev)]
        # env not bootstrapped here: manifest honestly absent, note recorded
        assert sidecar["env_manifest"] is None
        assert sidecar["env_note"]

    def test_without_tau_applies_registered_default(
        self, tmp_path: Path, fake_manager: dict[str, Any]
    ) -> None:
        """No --tau: the REGISTERED τ (TAU_REGISTERED = 0.817024, ragtruth_test
        scope, owner-decided 2026-08-05) is applied and recorded as such."""
        ev = _write_evidence(tmp_path / "qa_evidence.jsonl")
        out = tmp_path / "out.jsonl"
        rc = sib.main([
            "--evidence", str(ev), "--out", str(out),
            "--env-home", str(tmp_path / "eh"),
        ])
        assert rc == 0
        rows = _read_jsonl(out)
        by_example = {r["id"].split("::")[1]: r for r in rows}
        for row in rows:
            assert set(row) == {"id", "alignscore", "grounded_b"}
        # FAKE_SCORES thresholded at the registered τ 0.817024:
        assert by_example["e0"]["grounded_b"] is True   # 0.9  >= τ
        assert by_example["e1"]["grounded_b"] is False  # 0.5  <  τ
        assert by_example["e3"]["grounded_b"] is False  # 0.2  <  τ

        sidecar = json.loads(
            Path(str(out) + ".provenance.json").read_text(encoding="utf-8")
        )
        assert sidecar["tau"] == sib.ib.TAU_REGISTERED == 0.817024
        assert sidecar["tau_source"] == "registered"
        assert sidecar["tau_anchor_scope"] == "ragtruth_test"

    def test_explicit_tau_equal_to_registered_is_still_an_override(
        self, tmp_path: Path, fake_manager: dict[str, Any]
    ) -> None:
        """tau_source records the CHOICE, not the value: an explicit --tau that
        repeats the registered number is recorded as an override."""
        ev = _write_evidence(tmp_path / "qa_evidence.jsonl")
        out = tmp_path / "out.jsonl"
        rc = sib.main([
            "--evidence", str(ev), "--out", str(out),
            "--tau", str(sib.ib.TAU_REGISTERED),
            "--env-home", str(tmp_path / "eh"),
        ])
        assert rc == 0
        sidecar = json.loads(
            Path(str(out) + ".provenance.json").read_text(encoding="utf-8")
        )
        assert sidecar["tau"] == sib.ib.TAU_REGISTERED
        assert sidecar["tau_source"] == "override"

    def test_directory_evidence_is_searched_recursively(
        self, tmp_path: Path, fake_manager: dict[str, Any]
    ) -> None:
        _write_evidence(tmp_path / "runs" / "t0" / "qa_evidence.jsonl")
        _write_evidence(tmp_path / "runs" / "t1" / "qa_evidence.jsonl")
        out = tmp_path / "out.jsonl"
        rc = sib.main([
            "--evidence", str(tmp_path / "runs"), "--out", str(out),
            "--env-home", str(tmp_path / "eh"),
        ])
        assert rc == 0
        assert len(_read_jsonl(out)) == 6  # 3 scoreable rows per file

    def test_missing_out_is_a_usage_error(
        self, tmp_path: Path, fake_manager: dict[str, Any]
    ) -> None:
        ev = _write_evidence(tmp_path / "qa_evidence.jsonl")
        assert sib.main(["--evidence", str(ev)]) == 2
        assert fake_manager["score_calls"] == []

    def test_no_evidence_and_no_bootstrap_is_a_usage_error(
        self, fake_manager: dict[str, Any]
    ) -> None:
        assert sib.main([]) == 2

    def test_no_matching_evidence_is_an_error(
        self, tmp_path: Path, fake_manager: dict[str, Any]
    ) -> None:
        rc = sib.main([
            "--evidence", str(tmp_path / "nope*"), "--out", str(tmp_path / "o"),
        ])
        assert rc == 2


# --------------------------------------------------------------------------- #
# Scoring-tree mode (RESULTS_LAYOUT §6)
# --------------------------------------------------------------------------- #


def _sealed_run_root(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "run_root"
    ev = _write_evidence(root / "cells" / "rk1" / "window_w1" / "qa_evidence.jsonl")
    (root / "manifest.json").write_text(
        json.dumps({"run_id": "20260805-raw-run"}), encoding="utf-8"
    )
    write_ledger(hash_artifacts([ev], base_dir=root), root / "ledger.json")
    return root, ev


class TestScoringTreeMode:
    def test_tree_layout_manifest_and_own_ledger(
        self, tmp_path: Path, fake_manager: dict[str, Any]
    ) -> None:
        root, ev = _sealed_run_root(tmp_path)
        rc = sib.main([
            "--evidence", str(root),
            "--scoring-run-id", "s02-instrument-b",
            "--tau", "0.5",
            "--env-home", str(tmp_path / "eh"),
        ])
        assert rc == 0

        scoring_dir = root / "scoring" / "s02-instrument-b"
        scores_path = (
            scoring_dir / "cells" / "rk1" / "window_w1" / sib.SCORES_NAME
        )
        rows = _read_jsonl(scores_path)
        assert len(rows) == 3
        # ids are ROOT-relative in tree mode (stable across machines)
        assert rows[0]["id"].startswith("cells/rk1/window_w1/qa_evidence.jsonl::")
        assert all("grounded_b" in r for r in rows)

        manifest = json.loads(
            (scoring_dir / sib.SCORING_MANIFEST_NAME).read_text(encoding="utf-8")
        )
        assert manifest["scoring_run_id"] == "s02-instrument-b"
        assert manifest["raw_run_id"] == "20260805-raw-run"
        raw_ledger = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
        assert (
            manifest["raw_run_ledger_entries_sha256"]
            == raw_ledger["entries_sha256"]
        )
        assert manifest["instrument"] == "alignscore_large"
        assert manifest["tau"] == 0.5
        assert manifest["tau_source"] == "override"

        # the pass carries its OWN intact ledger (§6)
        assert verify_ledger(scoring_dir / "ledger.json", scoring_dir) == []
        # and NEVER wrote into cells/ (raw tree still verifies)
        assert verify_ledger(root / "ledger.json", root) == []

    def test_existing_scoring_run_id_is_refused(
        self, tmp_path: Path, fake_manager: dict[str, Any]
    ) -> None:
        root, _ = _sealed_run_root(tmp_path)
        args = [
            "--evidence", str(root),
            "--scoring-run-id", "s02-instrument-b",
            "--env-home", str(tmp_path / "eh"),
        ]
        assert sib.main(args) == 0
        assert sib.main(args) == 2  # append-only: a rerun needs a NEW id

    def test_out_flag_is_forbidden(
        self, tmp_path: Path, fake_manager: dict[str, Any]
    ) -> None:
        root, _ = _sealed_run_root(tmp_path)
        rc = sib.main([
            "--evidence", str(root),
            "--scoring-run-id", "s02-instrument-b",
            "--out", str(tmp_path / "o.jsonl"),
        ])
        assert rc == 2
        assert fake_manager["score_calls"] == []

    def test_bad_grammar_is_refused(
        self, tmp_path: Path, fake_manager: dict[str, Any]
    ) -> None:
        root, _ = _sealed_run_root(tmp_path)
        rc = sib.main(["--evidence", str(root), "--scoring-run-id", "BadID"])
        assert rc == 2

    def test_unsealed_run_root_is_refused(
        self, tmp_path: Path, fake_manager: dict[str, Any]
    ) -> None:
        root, _ = _sealed_run_root(tmp_path)
        (root / "ledger.json").unlink()
        rc = sib.main([
            "--evidence", str(root), "--scoring-run-id", "s02-instrument-b",
        ])
        assert rc == 2
        assert fake_manager["score_calls"] == []

    def test_multiple_evidence_args_are_refused(
        self, tmp_path: Path, fake_manager: dict[str, Any]
    ) -> None:
        root, _ = _sealed_run_root(tmp_path)
        rc = sib.main([
            "--evidence", str(root), str(root),
            "--scoring-run-id", "s02-instrument-b",
        ])
        assert rc == 2
