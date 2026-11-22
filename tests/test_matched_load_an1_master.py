"""AN-1 (ADR-0210, 2026-11-20): the master hands the analysis driver the plan the run
used and the rung calibration files, when the landing holds them.

The driver's matched-load section needs both: the plan carries the class
first-token targets and the control rung, the rung files the ladder's bracket
of every class. The harness is tests/test_experiment_master.py's (fakes for
every sibling script, runpodctl, ssh and scp); its fake driver prints its argv.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import test_experiment_master as tem

world = tem.world  # the master harness's fixture


def _driver_argv(land: Path) -> str:
    logs = sorted((land / "extras" / "steps").glob("analyze_*run_campaign_analysis*.log"))
    assert logs, "the analyze stage wrote no step log for the driver"
    return next(line for line in logs[-1].read_text(encoding="utf-8").splitlines() if line.startswith("analysis done"))


def test_the_driver_gets_the_plan_and_the_rung_files_the_landing_holds(world: Dict[str, Path]) -> None:
    proc = tem._master(world, "--yes", "provision", "--yes", "teardown")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    land = world["exp_root"] / "S1" / tem.DATE
    extras = land / "extras"
    argv = _driver_argv(land)
    assert f"'--contrasts', '4', '--plan', '{extras / 'plan.json'}'" in argv
    assert argv.endswith(f"'--rung-calibration', '{extras / 'calibration' / 'rungs_S1_vllm.json'}']")


def test_a_landing_without_the_plan_gets_neither_flag(world: Dict[str, Path]) -> None:
    assert tem._master(world, "--yes", "provision", "--yes", "teardown").returncode == 0
    land = world["exp_root"] / "S1" / tem.DATE
    (land / "extras" / "plan.json").unlink()
    stamps = list((land / "run").glob("*/*/*/analysis/*/stats.json"))
    assert len(stamps) == 1
    stamps[0].unlink()   # the recorded stamp no longer holds a look, so --redo runs the driver again
    # the master stamps its per-attempt output folders to the second (panels_HHMMSS) and the harness's
    # fake renderer refuses an existing one: the redo must not start in the second the first run ended
    time.sleep(1.1)
    proc = tem._master(world, "--only", "analyze", "--redo")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    argv = _driver_argv(land)
    # the rung files are the bracket of a section that needs the plan: without it they are not passed
    assert "'--contrasts', '4']" in argv and "--plan" not in argv and "--rung-calibration" not in argv


def test_the_plan_print_shows_both_flags(world: Dict[str, Path]) -> None:
    proc = tem._master(world, "--plan")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    (line,) = [ln for ln in proc.stdout.splitlines() if "run_campaign_analysis.py" in ln]
    assert "--plan" in line and "extras/plan.json" in line
    assert "--rung-calibration" in line and "calibration/rungs_S1_vllm.json" in line


def test_two_engines_ride_one_rung_calibration_flag(world: Dict[str, Path], tmp_path: Path) -> None:
    # review 2026-11-20 (test gap): the driver's option takes several files after ONE flag; a flag
    # repeated per file would keep the last file only
    prof = tmp_path / "S1_two_engines.env"
    prof.write_text(world["profile"].read_text(encoding="utf-8")
                    .replace('ENGINES="vllm hf"', 'ENGINES="vllm sglang hf"').replace('CALIBRATE="vllm"', 'CALIBRATE="vllm sglang"'),
                    encoding="utf-8")
    world["profile"] = prof
    proc = tem._master(world, "--plan", "--only", "analyze")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    (line,) = [ln for ln in proc.stdout.splitlines() if "run_campaign_analysis.py" in ln]
    assert line.count("--rung-calibration") == 1
    tail = line.split("--rung-calibration", 1)[1].split()
    assert [Path(part).name for part in tail] == ["rungs_S1_vllm.json", "rungs_S1_sglang.json"]
