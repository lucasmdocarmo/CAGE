"""T6.1 — the registered Qasper τ is CONSUMED from the freeze artifact.

WHAT is pinned and WHY: the §8.5 Qasper groundedness threshold was calibrated
and frozen (task #120 executed 2026-08-19; τ* registered in
MyDocs/registration/freeze_resolutions.json under ``QASPER_TAU``), and
build_predicate_table.py now resolves τ from that artifact instead of
trusting a hand-typed --qasper-tau — one typo in a hand-passed decimal would
silently produce an UNREGISTERED analysis. Pinned here:

- the four T6.1 resolution rules (artifact+flag must string-match AS
  WRITTEN; artifact alone supplies τ; flag alone is accepted with a LOUD
  single-line unregistered-by-hand warning; neither leaves the existing
  qasper-window refusal standing);
- the mismatch refusal names BOTH decimals, and numeric equality is NOT a
  match (the compare is on the literal as written in the artifact);
- freeze-artifact schema failures refuse (missing key, non-numeric τ,
  invalid JSON) — a broken registration artifact must never degrade to
  hand-passed mode;
- τ provenance is RECORDED: manifest ``config.qasper_tau_source``
  ("freeze-file"|"cli-flag"), ``config.freeze_file``, and the stdout summary
  line — and build_predicate_table refuses a configured τ whose source is
  unstated (bare values cannot re-enter through the API);
- resolution precedence: --freeze-file beats $CAGE_FREEZE_RESOLUTIONS.

Every test isolates $CAGE_FREEZE_RESOLUTIONS from the machine's real
artifact (present on the analysis machine, absent on pods) so the module is
hermetic on both.
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

import build_predicate_table as bpt  # noqa: E402
from src.analysis.predicate import PredicateConfig  # noqa: E402
from src.analysis.stats.ledger import hash_artifacts, write_ledger  # noqa: E402

SCORING_ID = "s01-fast"
#: The registered decimal exactly as written in the real artifact — pinned so
#: a freeze-file rewrite that changes the digits is caught by this module.
REGISTERED_LITERAL = "0.9955156950672646"


@pytest.fixture(autouse=True)
def _isolated_freeze_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Default every test to NO artifact; tests that want one point the env
    # var (or --freeze-file) at a tmp fixture. Without this, the repo-default
    # MyDocs/registration/freeze_resolutions.json would leak in on the
    # analysis machine and the same tests would pass/fail per machine.
    monkeypatch.setenv(bpt.FREEZE_ENV_VAR, str(tmp_path / "absent_freeze.json"))


def _freeze_file(tmp_path: Path, literal: str, name: str = "freeze.json") -> Path:
    # Raw text write: the T6.1 compare is against the decimal AS WRITTEN, so
    # the fixture controls the exact bytes (json.dumps could re-render them).
    path = tmp_path / name
    path.write_text(
        '{"FREEZE_SHA": "x", "QASPER_TAU": ' + literal + "}", encoding="utf-8"
    )
    return path


# ---------------------------------------------------------------------------
# Minimal sealed qasper tree (raw + scoring pass) — just enough for the CLI
# ---------------------------------------------------------------------------


def _sealed_qasper_run(
    tmp_path: Path, grounding_scores: tuple[float, ...] = (0.9, 0.5)
) -> Path:
    run_dir = tmp_path / "run"
    window = run_dir / "cells" / "cellA" / "window_qasper-01"
    window.mkdir(parents=True)
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
    evidence = window / "qa_evidence.jsonl"
    evidence.write_text(
        "".join(
            json.dumps({
                "example_id": f"e{i}", "repeat_index": 0, "record_index": None,
                "ok": True, "error": None, "empty_generation": False,
            }) + "\n"
            for i in range(len(grounding_scores))
        ),
        encoding="utf-8",
    )
    sealed = [run_dir / "manifest.json", evidence]
    write_ledger(hash_artifacts(sealed, base_dir=run_dir), run_dir / "ledger.json")
    raw_sha = json.loads((run_dir / "ledger.json").read_text())["entries_sha256"]

    scoring_dir = run_dir / "scoring" / SCORING_ID
    score_window = scoring_dir / "cells" / "cellA" / "window_qasper-01"
    score_window.mkdir(parents=True)
    scores = score_window / "qa_scores.jsonl"
    scores.write_text(
        "".join(
            json.dumps({
                "example_id": f"e{i}", "repeat_index": "0",
                "record_index": None, "grounding_score": s,
            }) + "\n"
            for i, s in enumerate(grounding_scores)
        ),
        encoding="utf-8",
    )
    manifest = scoring_dir / "scoring_manifest.json"
    manifest.write_text(
        json.dumps({
            "scoring_run_id": SCORING_ID,
            "raw_run_ledger_entries_sha256": raw_sha,
        }),
        encoding="utf-8",
    )
    write_ledger(
        hash_artifacts([scores, manifest], base_dir=scoring_dir),
        scoring_dir / "ledger.json",
    )
    return run_dir


def _main(run_dir: Path, *extra: str) -> int:
    return bpt.main([
        str(run_dir), "--scoring-run-id", SCORING_ID,
        "--max-null-fraction", "0.5", *extra,
    ])


def _manifest_config(run_dir: Path) -> dict[str, Any]:
    manifest = json.loads(
        (run_dir / "predicate" / SCORING_ID / "predicate_manifest.json").read_text()
    )
    return manifest["config"]


# ---------------------------------------------------------------------------
# Resolution rules (resolve_qasper_tau)
# ---------------------------------------------------------------------------


def test_artifact_alone_supplies_tau(tmp_path: Path) -> None:
    freeze = _freeze_file(tmp_path, "0.8")
    tau, source, warning = bpt.resolve_qasper_tau(None, freeze)
    assert tau == 0.8
    assert source == bpt.TAU_SOURCE_FREEZE
    assert warning is None


def test_matching_flag_confirms_artifact_as_source(tmp_path: Path) -> None:
    # Artifact + identical hand decimal: the artifact stays authoritative.
    freeze = _freeze_file(tmp_path, "0.8")
    tau, source, warning = bpt.resolve_qasper_tau("0.8", freeze)
    assert tau == 0.8
    assert source == bpt.TAU_SOURCE_FREEZE
    assert warning is None


def test_mismatched_flag_refuses_naming_both_values(tmp_path: Path) -> None:
    freeze = _freeze_file(tmp_path, REGISTERED_LITERAL)
    with pytest.raises(bpt.BuildPredicateError) as exc:
        bpt.resolve_qasper_tau("0.995516", freeze)
    msg = str(exc.value)
    assert "0.995516" in msg and REGISTERED_LITERAL in msg
    assert bpt.FREEZE_TAU_KEY in msg


def test_numeric_equality_is_not_a_match(tmp_path: Path) -> None:
    # "0.800" == 0.8 numerically, but the T6.1 rule compares the decimal AS
    # WRITTEN — a rewrite of the registered digits refuses.
    freeze = _freeze_file(tmp_path, "0.8")
    with pytest.raises(bpt.BuildPredicateError, match="MATCH"):
        bpt.resolve_qasper_tau("0.800", freeze)


def test_flag_without_artifact_warns_unregistered_by_hand(tmp_path: Path) -> None:
    absent = tmp_path / "nowhere.json"
    tau, source, warning = bpt.resolve_qasper_tau("0.8", absent)
    assert tau == 0.8
    assert source == bpt.TAU_SOURCE_CLI
    assert warning is not None and "UNREGISTERED-BY-HAND" in warning
    assert str(absent) in warning
    assert "\n" not in warning  # LOUD but single-line, per T6.1


def test_neither_resolves_nothing(tmp_path: Path) -> None:
    tau, source, warning = bpt.resolve_qasper_tau(None, tmp_path / "nowhere.json")
    assert tau is None and source is None and warning is None


def test_non_decimal_flag_refuses(tmp_path: Path) -> None:
    with pytest.raises(bpt.BuildPredicateError, match="not a decimal"):
        bpt.resolve_qasper_tau("0.8x", tmp_path / "nowhere.json")


# ---------------------------------------------------------------------------
# Freeze-artifact schema failures refuse (never degrade to hand-passed mode)
# ---------------------------------------------------------------------------


def test_read_frozen_tau_preserves_the_written_decimal(tmp_path: Path) -> None:
    freeze = _freeze_file(tmp_path, REGISTERED_LITERAL)
    assert bpt.read_frozen_tau(freeze) == REGISTERED_LITERAL


def test_artifact_without_tau_key_refuses(tmp_path: Path) -> None:
    path = tmp_path / "freeze.json"
    path.write_text('{"FREEZE_SHA": "x"}', encoding="utf-8")
    with pytest.raises(bpt.BuildPredicateError, match=bpt.FREEZE_TAU_KEY):
        bpt.resolve_qasper_tau(None, path)


def test_artifact_with_string_typed_tau_refuses(tmp_path: Path) -> None:
    path = tmp_path / "freeze.json"
    path.write_text('{"QASPER_TAU": "0.8"}', encoding="utf-8")
    with pytest.raises(bpt.BuildPredicateError, match="JSON number"):
        bpt.resolve_qasper_tau(None, path)


def test_artifact_with_invalid_json_refuses(tmp_path: Path) -> None:
    path = tmp_path / "freeze.json"
    path.write_text('{"QASPER_TAU": 0.8', encoding="utf-8")
    with pytest.raises(bpt.BuildPredicateError, match="not valid JSON"):
        bpt.resolve_qasper_tau(None, path)


# ---------------------------------------------------------------------------
# End-to-end CLI: τ acts on qasper rows and its source is RECORDED
# ---------------------------------------------------------------------------


def test_cli_consumes_tau_from_freeze_and_records_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = _sealed_qasper_run(tmp_path, grounding_scores=(0.9, 0.5))
    freeze = _freeze_file(tmp_path, "0.8")
    monkeypatch.setenv(bpt.FREEZE_ENV_VAR, str(freeze))
    assert _main(run_dir) == 0

    config = _manifest_config(run_dir)
    assert config["qasper_tau"] == 0.8
    assert config["qasper_tau_source"] == "freeze-file"
    assert config["freeze_file"] == str(freeze.resolve())
    # τ ACTS: 0.9 >= τ -> True, 0.5 < τ -> False (not merely recorded).
    rows_path = (
        run_dir / "predicate" / SCORING_ID
        / "cells" / "cellA" / "window_qasper-01" / "predicate.jsonl"
    )
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    assert [r["predicate"] for r in rows] == [True, False]
    out = capsys.readouterr()
    assert "freeze-file" in out.out  # stdout summary names the source
    assert "0.8" in out.out


def test_cli_hand_tau_without_artifact_warns_and_records_cli_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = _sealed_qasper_run(tmp_path)
    assert _main(run_dir, "--qasper-tau", "0.8") == 0
    config = _manifest_config(run_dir)
    assert config["qasper_tau"] == 0.8
    assert config["qasper_tau_source"] == "cli-flag"
    assert config["freeze_file"] is None
    out = capsys.readouterr()
    assert "UNREGISTERED-BY-HAND" in out.err
    assert "cli-flag" in out.out


def test_cli_mismatch_refuses_and_leaves_no_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = _sealed_qasper_run(tmp_path)
    freeze = _freeze_file(tmp_path, REGISTERED_LITERAL)
    monkeypatch.setenv(bpt.FREEZE_ENV_VAR, str(freeze))
    assert _main(run_dir, "--qasper-tau", "0.995516") == 1
    err = capsys.readouterr().err
    assert "0.995516" in err and REGISTERED_LITERAL in err
    assert not (run_dir / "predicate").exists()


def test_cli_neither_keeps_the_missing_tau_refusal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # No artifact, no flag, qasper windows present: the pre-existing #120
    # refusal stands — freeze consumption must not have invented a default.
    run_dir = _sealed_qasper_run(tmp_path)
    assert _main(run_dir) == 1
    err = capsys.readouterr().err
    assert "#120" in err
    assert not (run_dir / "predicate").exists()  # fail-closed: no half table


def test_cli_freeze_file_flag_beats_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _sealed_qasper_run(tmp_path)
    env_freeze = _freeze_file(tmp_path, "0.7", name="env_freeze.json")
    flag_freeze = _freeze_file(tmp_path, "0.8", name="flag_freeze.json")
    monkeypatch.setenv(bpt.FREEZE_ENV_VAR, str(env_freeze))
    assert _main(run_dir, "--freeze-file", str(flag_freeze)) == 0
    config = _manifest_config(run_dir)
    assert config["qasper_tau"] == 0.8
    assert config["freeze_file"] == str(flag_freeze.resolve())


# ---------------------------------------------------------------------------
# API-level provenance guard: bare τ values cannot re-enter
# ---------------------------------------------------------------------------


def test_direct_call_with_unstated_tau_provenance_refuses(tmp_path: Path) -> None:
    config = PredicateConfig(max_null_fraction=0.5, qasper_tau=0.8)
    with pytest.raises(bpt.BuildPredicateError, match="provenance"):
        bpt.build_predicate_table(tmp_path / "run", SCORING_ID, config)


def test_direct_call_with_source_but_no_tau_refuses(tmp_path: Path) -> None:
    config = PredicateConfig(max_null_fraction=0.5)
    with pytest.raises(bpt.BuildPredicateError, match="fabricated provenance"):
        bpt.build_predicate_table(
            tmp_path / "run", SCORING_ID, config,
            tau_source=bpt.TAU_SOURCE_FREEZE,
        )
