"""A stage 6 refusal on HOURS, and what it costs (second review of the eight-arm cut, M1).

Session s2a's limits (HOURS 55, the run stop at 50, the seatbelt at 88h) were kept on
2026-11-22 on the belief that raising HOURS after a stage 6 refusal would change the
calibration key and make stage 5 measure again. The second independent review read
the master and found that false: the key (pod.id:build.calib_hash) is written at
stage 0 and compared inside stage 5, the profile is read again on every invocation,
and a passed stage is skipped on a resume. This test pins the true behavior, so the
profile's comment and the records rest on a test and not on a reading.

The harness is tests/test_experiment_master.py's (fakes for runpodctl, ssh and the
pod); its world profile is S1 with HOURS=1.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import test_experiment_master as tem

world = tem.world  # the master harness's fixture


def _calibration_submits(world: Dict[str, Path]) -> int:
    # one submit = one decode line writing the remote .cmd (status probes name the file too)
    return sum(1 for c in tem._calls(world) if c.rstrip().endswith((
        "base64 -d > ~/.cage_jobs/calibrate_vllm.cmd", "base64 -d > ~/.cage_jobs/calibrate_rungs_vllm.cmd")))


def test_raising_hours_after_a_stage_6_refusal_keeps_the_calibration(world: Dict[str, Path]) -> None:
    # a projection of 2 h against HOURS=1: stage 6 refuses, after stage 5 has measured
    proc = tem._master(world, "--yes", "provision", "--to", "plan", CAGE_TEST_PROJECTION_S="7200")
    assert proc.returncode == 1, proc.stdout[-3000:] + proc.stderr[-1000:]
    assert "the projected run (2.00 h) exceeds HOURS=1" in proc.stdout
    assert "raise HOURS in the profile (the seatbelt must cover it) or cut the plan" in proc.stdout
    st = tem._state(world)
    assert st["stages"]["calibrate"]["status"] == "passed" and st["stages"]["plan"]["status"] == "failed"
    h0 = st["build"]["calib_hash"]
    key0 = f"{st['pod']['id']}:{h0}"
    assert st["calibration"]["cell_done"]["vllm"] == key0 and st["calibration"]["rungs_done"]["vllm"] == key0
    n0 = _calibration_submits(world)
    assert n0 == 2

    # the remedy the refusal names: raise HOURS in the profile, resume from the plan stage
    tem._profile_with(world, "S1_hours_raised.env", "HOURS=3")
    proc = tem._master(world, "--from", "plan", "--to", "plan", CAGE_TEST_PROJECTION_S="7200")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    st = tem._state(world)
    assert st["stages"]["plan"]["status"] == "passed"
    # stage 0 and stage 5 were not entered again: the key, the done marks and the job count stand
    assert st["build"]["calib_hash"] == h0
    assert st["calibration"]["cell_done"]["vllm"] == key0 and st["calibration"]["rungs_done"]["vllm"] == key0
    assert _calibration_submits(world) == n0
