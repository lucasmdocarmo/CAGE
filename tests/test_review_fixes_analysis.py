"""Regression tests for the 2026-08-04 analysis-area code-review fixes.

Each test targets one verified finding and fails against the pre-fix driver:

- scripts/4_analysis/run_campaign_analysis.py (§9.11 one-look lock): the lock
  was a read-then-write race — ``read_lock()`` was checked once before the
  (potentially long) analysis pipeline ran, and ``write_lock()`` overwrote
  ``analysis_lock.json`` unconditionally at the end with no re-check and no
  exclusive-create. Two concurrent ``--confirmatory`` invocations could both
  pass the initial refusal check and both compute a full confirmatory
  analysis, with the second writer silently clobbering the first lock's
  record. Fixed with an atomic ``O_CREAT|O_EXCL`` acquire *before* the
  pipeline runs (``_acquire_confirmatory_lock``), an IN_PROGRESS placeholder
  removed on any failure by the acquiring process (``_release_placeholder_lock``,
  so a crash never burns the run's one registered look), and an atomic
  write-temp+``os.replace`` finalize (``write_lock``).
- scripts/4_analysis/run_campaign_analysis.py (compute_pair_stats): every
  metric was routed through ``paired_wilcoxon`` unconditionally, even the
  binary §8.5 Y predicate metric that ``families.py`` already tags
  ``unit="binary"`` for McNemar. Requesting ``--metrics predicate`` silently
  ran a Wilcoxon signed-rank test on 0/1 paired data instead of the
  registered exact-binomial McNemar test (a different reference
  distribution, not merely an approximation).

(The equivalence.py TOST-CI alpha fix and the primary-tier Holm-correction
exemption are covered in tests/test_stats_engine.py::TestConditionalTost and
tests/test_campaign_analysis.py::test_design_input_default_stats_figures_and_stamp
respectively — extended in place since those modules already exercise the
exact functions/paths involved.)
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

import organize_results as org  # noqa: E402
import run_campaign_analysis as rca  # noqa: E402
from src.analysis.cellspec import CellSpec  # noqa: E402
from src.analysis.stats.ledger import hash_artifacts, write_ledger  # noqa: E402
from src.analysis.stats.tests_by_unit import mcnemar_binary, paired_wilcoxon  # noqa: E402

RUN_ID = "20260804-0900-a-qwen3-14b"
CAMPAIGN = "camp-rfa"
SESSION = "a"
MODEL = "qwen3-14b"
DATASET = "squad_v2"
N_EXAMPLES = 18

WINDOW_ARTIFACTS = (
    "requests.jsonl",
    "qa_evidence.jsonl",
    "engine_metrics.json",
    "cage_stats.jsonl",
)

#: Per-example (B6, B3) predicate pairs — ALL discordant, no ties, but NOT
#: unanimous in one direction: 15 pairs where B6 passes and B3 fails, 3 where
#: B3 passes and B6 fails. This is deliberate: with EVERY pair discordant in
#: the SAME direction (n_01 == 0), paired_wilcoxon-on-0/1-data degenerates
#: to the exact same combinatorics as the McNemar binomial test (both reduce
#: to a plain sign test) and the two p-values coincide by mathematical
#: accident, which would make a p_value-divergence assertion vacuous. With a
#: mixed discordant pattern (n_10=15, n_01=3) scipy's Wilcoxon switches to
#: its tie-driven normal approximation while McNemar stays the exact
#: binomial — the two genuinely diverge, matching the review finding's own
#: live-executed evidence (a "materially different reference distribution,
#: not a harmless approximation").
_PREDICATE_PATTERN: tuple[tuple[int, int], ...] = tuple(
    [(1, 0)] * 15 + [(0, 1)] * 3
)
assert len(_PREDICATE_PATTERN) == N_EXAMPLES


def _write_window(wdir: Path, *, baseline: str) -> list[Path]:
    """One window's §1 artifact set carrying a binary ``predicate`` column,
    per-example values taken from ``_PREDICATE_PATTERN``."""
    wdir.mkdir(parents=True)
    written: list[Path] = []
    requests_lines = []
    evidence_lines = []
    col = 0 if baseline == "B6" else 1
    for i in range(N_EXAMPLES):
        example_id = f"{DATASET}-e{i:03d}"
        predicate = _PREDICATE_PATTERN[i][col]
        requests_lines.append(
            json.dumps(
                {"example_id": example_id, "ttft_ms": 100.0 + i, "predicate": predicate}
            )
        )
        evidence_lines.append(json.dumps({"example_id": example_id, "answer": "text"}))
    payloads = {
        "requests.jsonl": "\n".join(requests_lines) + "\n",
        "qa_evidence.jsonl": "\n".join(evidence_lines) + "\n",
        "engine_metrics.json": json.dumps({"snapshot": "before/after"}),
        "cage_stats.jsonl": json.dumps({"t": 0, "kv_bytes": 1}) + "\n",
    }
    for name in WINDOW_ARTIFACTS:
        path = wdir / name
        path.write_text(payloads[name], encoding="utf-8")
        written.append(path)
    return written


def _build_run_tree(tmp_path: Path) -> Path:
    """Minimal organized-ready run tree: just B3/B6 (contrast #4 headline),
    one dataset, one window each — WITH a binary ``predicate`` per-query
    column (the RESULTS_LAYOUT §1 fixture pattern, trimmed down)."""
    run_dir = tmp_path / "results" / CAMPAIGN / SESSION / RUN_ID
    run_dir.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "campaign": CAMPAIGN,
        "session": SESSION,
        "run_id": RUN_ID,
        "model": MODEL,
        "git_sha": "deadbeef",
        "git_dirty": False,
        "engine": "vllm",
        "engine_version": "0.0-test",
        "seed": 1,
        "provider": "test",
        "hardware": "test-gpu",
        "dataset_manifests_sha256": "0" * 64,
        "cellspec_schema_version": 1,
        "created_utc": "2026-08-04T09:00:00Z",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    sealed: list[Path] = []
    for baseline in ("B6", "B3"):
        spec = CellSpec.from_baseline(baseline, model=MODEL)  # type: ignore[arg-type]
        cell_dir = run_dir / "cells" / spec.to_row_key()
        cell_dir.mkdir(parents=True)
        window_key = f"{DATASET}-01"
        sealed.extend(
            _write_window(cell_dir / f"window_{window_key}", baseline=baseline)
        )
        cell_json = cell_dir / "cell.json"
        cell_json.write_text(
            json.dumps(
                {
                    "cellspec": spec.to_flat_dict(),
                    "baseline": baseline,
                    "windows": {window_key: {"dataset": DATASET, "seed": 1, "rep": 1}},
                }
            ),
            encoding="utf-8",
        )
        sealed.append(cell_json)
    write_ledger(hash_artifacts(sealed, base_dir=run_dir), run_dir / "ledger.json")
    return run_dir


@pytest.fixture()
def organized_run(tmp_path: Path) -> Path:
    run_dir = _build_run_tree(tmp_path)
    org.organize_run(run_dir)
    return run_dir


def _write_passing_calibration_report(path: Path) -> Path:
    """A PASSING §9.7 CalibrationReport JSON artifact (A/A + one injection).

    Confirmatory mode hard-gates on this artifact (checked BEFORE the §9.11
    lock is acquired), so the one-look tests below must supply one to reach
    the lock semantics they target. Payload mirrors
    tests/test_campaign_analysis.py::_write_calibration_report(passing=True).
    """
    payload = {
        "seed": 7,
        "n_observations": 128,
        "aa": {
            "n_splits": 200,
            "alpha": 0.05,
            "n_rejections": 8,
            "fp_rate": 0.04,
            "ci_low": 0.01,
            "ci_high": 0.09,
        },
        "injections": [
            {
                "effect_size": 5.0,
                "kind": "shift",
                "n_splits": 200,
                "alpha": 0.05,
                "n_rejections": 170,
                "power": 0.85,
                "ci_low": 0.79,
                "ci_high": 0.90,
                "target_power": 0.8,
            }
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Finding: binary predicate metric must route through McNemar, not Wilcoxon
# ---------------------------------------------------------------------------


def test_predicate_metric_routes_through_mcnemar_not_wilcoxon(organized_run: Path) -> None:
    rc = rca.main([str(organized_run), "--metrics", "predicate"])
    assert rc == 0
    dirs = sorted((organized_run / "analysis").iterdir())
    stats = json.loads((dirs[-1] / "stats.json").read_text(encoding="utf-8"))

    assert len(stats["contrasts"]) == 1
    entry = stats["contrasts"][0]
    assert entry["metric"] == "predicate"
    assert entry["unit"] == "binary"
    assert entry["test"] == "mcnemar_binary"

    row = entry["per_dataset"][0]
    assert row["dataset"] == DATASET
    # McNemar's own fields must be present ...
    assert row["n_10"] == 15  # B6 passes, B3 fails
    assert row["n_01"] == 3  # B3 passes, B6 fails
    assert row["n_discordant"] == 18
    assert row["proportion_diff"] == pytest.approx((15 - 3) / N_EXAMPLES)
    # ... and the Wilcoxon-only fields must NOT be — they would silently
    # carry the wrong reference distribution on 0/1 data, not merely an
    # approximation of the right one.
    assert "statistic" not in row
    assert "cliffs_delta_paired" not in row

    # Lock in that McNemar's exact binomial actually ran, and that it
    # disagrees with what paired_wilcoxon would have reported on the same
    # arrays (the bug this regression test targets).
    a = [pair[0] for pair in _PREDICATE_PATTERN]
    b = [pair[1] for pair in _PREDICATE_PATTERN]
    expected_mcnemar = mcnemar_binary(a, b, alternative="two-sided")
    assert row["p_value"] == pytest.approx(expected_mcnemar.p_value)
    wrong_test = paired_wilcoxon(a, b, alternative="two-sided")
    assert row["p_value"] != pytest.approx(wrong_test.p_value)


# ---------------------------------------------------------------------------
# Finding: §9.11 one-look lock TOCTOU race
# ---------------------------------------------------------------------------


def test_concurrent_confirmatory_acquire_refuses_atomically(organized_run: Path) -> None:
    # "Process A" acquires the lock first.
    rca._acquire_confirmatory_lock(organized_run, "sha-process-A")
    lock_path = organized_run / rca.LOCK_NAME
    assert lock_path.is_file()
    placeholder = json.loads(lock_path.read_text(encoding="utf-8"))
    assert placeholder["phase"] == "IN_PROGRESS"
    assert placeholder["registered_sha"] == "sha-process-A"

    # "Process B" starts a moment later, BEFORE process A's pipeline (or its
    # write_lock finalize) has run at all. Under the pre-fix code
    # (read_lock() checked once, write_lock() unconditionally at the end)
    # this would have silently succeeded — both processes would go on to run
    # the full pipeline, and whichever finished last would clobber the
    # other's lock record with no error raised anywhere.
    with pytest.raises(rca.OneLookError, match="IN PROGRESS"):
        rca._acquire_confirmatory_lock(organized_run, "sha-process-B")

    # Process A's claim is untouched by B's refused attempt.
    still = json.loads(lock_path.read_text(encoding="utf-8"))
    assert still == placeholder


def test_crashed_confirmatory_attempt_does_not_burn_the_one_look_budget(
    organized_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_load_index = rca.load_index
    calibration = _write_passing_calibration_report(
        organized_run.parent / "calibration_report.json"
    )

    def _boom(run_dir: Path) -> None:
        raise RuntimeError("simulated crash mid-pipeline (e.g. OOM)")

    monkeypatch.setattr(rca, "load_index", _boom)
    with pytest.raises(RuntimeError, match="simulated crash"):
        rca.run_analysis(
            organized_run,
            contrast_ids=[4],
            metrics=["ttft_ms"],
            mode="confirmatory",
            registered_sha="sha-attempt-1",
            calibration_report=calibration,
        )
    # The crash must NOT leave a lock behind — a retry is still legal, i.e.
    # the run's one registered look was not consumed by the failed attempt.
    assert not (organized_run / rca.LOCK_NAME).exists()

    monkeypatch.setattr(rca, "load_index", real_load_index)
    result = rca.run_analysis(
        organized_run,
        contrast_ids=[4],
        metrics=["ttft_ms"],
        mode="confirmatory",
        registered_sha="sha-attempt-2",
        calibration_report=calibration,
    )
    assert result.stats_path.is_file()
    lock = json.loads((organized_run / rca.LOCK_NAME).read_text(encoding="utf-8"))
    assert lock["phase"] == "DONE"
    assert lock["registered_sha"] == "sha-attempt-2"

    # A third confirmatory attempt now correctly refuses — the budget WAS
    # spent, by the successful attempt-2, not by the earlier crash.
    with pytest.raises(rca.OneLookError, match="ONE-LOOK"):
        rca.run_analysis(
            organized_run,
            contrast_ids=[4],
            metrics=["ttft_ms"],
            mode="confirmatory",
            registered_sha="sha-attempt-3",
            calibration_report=calibration,
        )
    # No leftover temp file from the atomic write-temp+os.replace finalize.
    assert not list(organized_run.glob(f"{rca.LOCK_NAME}.tmp-*"))
