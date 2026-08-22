"""K-PIN2 (task #140): EXECUTE every runner's cell_complete resume gate.

The Topic-12 finding: the J2 pin in tests/test_topic10_runner_hardening.py is
presence-only (greps the shell source for `metrics_json_valid`), so an
appended ``|| true`` — which makes every cell resume-proof "complete" again —
is invisible. This file extracts each runner's CURRENT ``cell_complete``
function verbatim (drift breaks loudly), sources the real
scripts/lib/_common.sh (``metrics_json_valid``), and runs it under bash
against synthetic run trees:

- complete cell (every trial has VALID parseable metrics.json)  -> complete
- missing metrics.json (trial dir present, file absent)          -> incomplete
- missing trial dir entirely                                     -> incomplete
- syntactically-invalid metrics.json (truncated JSON)            -> incomplete,
  announced LOUDLY via the metrics_json_valid warn path (J2)

Task #116: the campaign-wired runners (baselines, compression) carry a
DUAL-MODE gate — with CAGE_CAMPAIGN_ROOT set, trial t's sentinel lives in the
RESULTS_LAYOUT-v2 measurement window ``<cell>/window_<dataset>-<0t>/metrics.json``
instead of ``<cell>/trial_<t>/metrics.json``. Both modes are executed here
with the SAME four scenarios and the same J2 rigor; the pilot scenarios pin
the env OFF explicitly so an operator's exported campaign root can never flip
what these tests measure.

Pure local checks: no GPU, no server, no network.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
COMMON = SCRIPTS / "lib" / "_common.sh"

#: The five runners carrying a cell_complete resume gate (same roster as
#: tests/test_topic10_runner_hardening.py). kv_store's variant reads
#: $OUTPUT_DIR/$LABEL instead of a <cell_dir> argument.
RUNNERS = {
    "baselines": SCRIPTS / "3_run" / "run_baselines.sh",
    "compression": SCRIPTS / "3_run" / "run_compression.sh",
    "memory_sweep": SCRIPTS / "3_run" / "run_memory_sweep.sh",
    "kv_store": SCRIPTS / "3_run" / "run_kv_store.sh",
    "prefix_envelope": SCRIPTS / "3_run" / "run_prefix_envelope.sh",
}
#: Task #116: the campaign-wired subset whose gate carries the v2 window branch.
CAMPAIGN_RUNNERS = ("baselines", "compression")
NUM_TRIALS = 2
DATASET = "squad_v2"


def _extract_cell_complete(path: Path) -> str:
    """The CURRENT cell_complete function text, verbatim (re-read at test
    time so an edited runner is what executes here)."""
    text = path.read_text(encoding="utf-8")
    m = re.search(r"^cell_complete\(\) \{[^\n]*\n.*?^\}", text, re.M | re.S)
    assert m, f"{path.name}: cell_complete() not found — resume gate renamed/deleted?"
    return m.group(0)


def _run_gate(
    runner: str, cell_dir: Path, *, campaign: bool = False
) -> subprocess.CompletedProcess:
    fn = _extract_cell_complete(RUNNERS[runner])
    if runner == "kv_store":
        setup = (
            f'OUTPUT_DIR="{cell_dir.parent}"\n'
            f'LABEL="{cell_dir.name}"\n'
        )
        call = "cell_complete"
    else:
        setup = ""
        call = f'cell_complete "{cell_dir}"'
    if campaign:
        # The campaign branch keys on CAGE_CAMPAIGN_ROOT being set and reads
        # $DATASET for the window names; the cell dir itself is the argument.
        mode = (
            f'export CAGE_CAMPAIGN_ROOT="{cell_dir.parent}"\n'
            f'DATASET="{DATASET}"\n'
        )
    else:
        # Pilot semantics must hold regardless of the invoking shell's env.
        mode = "unset CAGE_CAMPAIGN_ROOT\n"
    script = f"""
set -uo pipefail
source "{COMMON}"
NUM_TRIALS={NUM_TRIALS}
{mode}{setup}{fn}
{call} && echo "CELL_COMPLETE" || echo "CELL_INCOMPLETE"
"""
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        env=dict(os.environ),
    )


def _make_cell(tmp_path: Path, *, trials: int = NUM_TRIALS) -> Path:
    cell = tmp_path / "cell_b3"
    for t in range(1, trials + 1):
        d = cell / f"trial_{t}"
        d.mkdir(parents=True)
        (d / "metrics.json").write_text('{"ttft_ms": 42.0, "ok": 1}', encoding="utf-8")
    return cell


def _make_campaign_cell(tmp_path: Path, *, windows: int = NUM_TRIALS) -> Path:
    """A v2 cells/<row_key> dir: trial t -> window_<dataset>-<0t>/metrics.json."""
    cell = tmp_path / "gold-reuse|none|none|single|vllm|qwen3-14b|F1"
    for t in range(1, windows + 1):
        d = cell / f"window_{DATASET}-{t:02d}"
        d.mkdir(parents=True)
        (d / "metrics.json").write_text('{"ttft_ms": 42.0, "ok": 1}', encoding="utf-8")
    return cell


@pytest.mark.parametrize("runner", sorted(RUNNERS))
def test_complete_cell_is_complete(runner: str, tmp_path: Path) -> None:
    cell = _make_cell(tmp_path)
    proc = _run_gate(runner, cell)
    assert "CELL_COMPLETE" in proc.stdout, (
        f"{runner}: a fully-valid cell no longer counts as complete "
        f"(resume would needlessly re-run it):\n{proc.stdout}\n{proc.stderr}"
    )


@pytest.mark.parametrize("runner", sorted(RUNNERS))
def test_missing_metrics_json_is_incomplete(runner: str, tmp_path: Path) -> None:
    cell = _make_cell(tmp_path)
    (cell / f"trial_{NUM_TRIALS}" / "metrics.json").unlink()
    proc = _run_gate(runner, cell)
    assert "CELL_INCOMPLETE" in proc.stdout, (
        f"{runner}: a trial without metrics.json counted as complete "
        f"(K-PIN2 — the resume gate went presence-blind):\n{proc.stdout}\n{proc.stderr}"
    )


@pytest.mark.parametrize("runner", sorted(RUNNERS))
def test_missing_trial_dir_is_incomplete(runner: str, tmp_path: Path) -> None:
    cell = _make_cell(tmp_path, trials=NUM_TRIALS - 1)
    proc = _run_gate(runner, cell)
    assert "CELL_INCOMPLETE" in proc.stdout, (
        f"{runner}: a cell missing trial_{NUM_TRIALS} counted as complete:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )


@pytest.mark.parametrize("runner", sorted(RUNNERS))
def test_invalid_metrics_json_is_incomplete_and_loud(runner: str, tmp_path: Path) -> None:
    """The J2 semantics under EXECUTION: existence is not validity — a
    truncated metrics.json must not freeze a corrupt cell in as complete, and
    the refusal is announced (metrics_json_valid's warn path). An `|| true`
    appended to the metrics_json_valid call fails exactly this test."""
    cell = _make_cell(tmp_path)
    (cell / f"trial_{NUM_TRIALS}" / "metrics.json").write_text(
        '{"ttft_ms": 42.0, "trunc', encoding="utf-8"
    )
    proc = _run_gate(runner, cell)
    assert "CELL_INCOMPLETE" in proc.stdout, (
        f"{runner}: a syntactically-invalid metrics.json counted as complete "
        f"(J2/K-PIN2 regression — corrupt data would be resume-proof):\n"
        f"{proc.stdout}\n{proc.stderr}"
    )
    assert "invalid metrics.json" in proc.stderr, (
        f"{runner}: the invalid-JSON refusal is no longer LOUD "
        f"(metrics_json_valid warn path):\n{proc.stderr}"
    )


# ---------------------------------------------------------------------------
# Task #116: campaign mode (v2 window paths) — same scenarios, same rigor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("runner", CAMPAIGN_RUNNERS)
def test_campaign_complete_cell_is_complete(runner: str, tmp_path: Path) -> None:
    cell = _make_campaign_cell(tmp_path)
    proc = _run_gate(runner, cell, campaign=True)
    assert "CELL_COMPLETE" in proc.stdout, (
        f"{runner}: a fully-valid v2 cell (all window_<dataset>-NN/metrics.json "
        f"parseable) no longer counts as complete:\n{proc.stdout}\n{proc.stderr}"
    )


@pytest.mark.parametrize("runner", CAMPAIGN_RUNNERS)
def test_campaign_missing_window_metrics_is_incomplete(runner: str, tmp_path: Path) -> None:
    cell = _make_campaign_cell(tmp_path)
    (cell / f"window_{DATASET}-{NUM_TRIALS:02d}" / "metrics.json").unlink()
    proc = _run_gate(runner, cell, campaign=True)
    assert "CELL_INCOMPLETE" in proc.stdout, (
        f"{runner}: a v2 window without metrics.json counted as complete:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )


@pytest.mark.parametrize("runner", CAMPAIGN_RUNNERS)
def test_campaign_missing_window_dir_is_incomplete(runner: str, tmp_path: Path) -> None:
    cell = _make_campaign_cell(tmp_path, windows=NUM_TRIALS - 1)
    proc = _run_gate(runner, cell, campaign=True)
    assert "CELL_INCOMPLETE" in proc.stdout, (
        f"{runner}: a v2 cell missing window_{DATASET}-{NUM_TRIALS:02d} counted "
        f"as complete:\n{proc.stdout}\n{proc.stderr}"
    )


@pytest.mark.parametrize("runner", CAMPAIGN_RUNNERS)
def test_campaign_invalid_window_metrics_is_incomplete_and_loud(
    runner: str, tmp_path: Path
) -> None:
    """J2 carries over to the v2 paths unchanged: a truncated window
    metrics.json is INCOMPLETE and the refusal stays loud."""
    cell = _make_campaign_cell(tmp_path)
    (cell / f"window_{DATASET}-{NUM_TRIALS:02d}" / "metrics.json").write_text(
        '{"ttft_ms": 42.0, "trunc', encoding="utf-8"
    )
    proc = _run_gate(runner, cell, campaign=True)
    assert "CELL_INCOMPLETE" in proc.stdout, (
        f"{runner}: a syntactically-invalid v2 window metrics.json counted as "
        f"complete (J2 regression at the #116 seam):\n{proc.stdout}\n{proc.stderr}"
    )
    assert "invalid metrics.json" in proc.stderr, (
        f"{runner}: the invalid-JSON refusal is no longer LOUD in campaign "
        f"mode:\n{proc.stderr}"
    )


@pytest.mark.parametrize("runner", CAMPAIGN_RUNNERS)
def test_campaign_pilot_trial_layout_is_not_accepted(runner: str, tmp_path: Path) -> None:
    """In campaign mode a PILOT-shaped cell (trial_N/) must read incomplete —
    the gate keys on the v2 window paths, never a leftover pilot tree."""
    cell = _make_cell(tmp_path)  # trial_1..N layout
    proc = _run_gate(runner, cell, campaign=True)
    assert "CELL_INCOMPLETE" in proc.stdout, (
        f"{runner}: campaign mode accepted a pilot trial_N/ tree as complete:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )
