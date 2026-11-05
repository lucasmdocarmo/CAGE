"""S2 rehearsal profile S2R.env (owner 2026-11-04, option R1, "lets do the test with
the h200 as designed"): S2B's exact shape (the 80B on four H200, both engines, every
ladder) driven through every stage of the master once, cut short at the run, before
landing B bills. Ten keys differ from S2B.env and nothing else does; the file is never
a landing of record. ADR-0190 (owner 2026-11-05, option D): RUN_CUT_BY_BUDGET=1 lets
stage 6 accept the 38 h plan projection because the budget stop cuts the run first. Spec: MyDocs/RunPod/S2_REHEARSAL_LANDING_2026-11-05.md.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILES = REPO_ROOT / "scripts" / "6_experiments" / "profiles"

#: the keys S2R.env states differently from S2B.env, and why (the Spec Block of 2026-11-04)
LEVERS: Dict[str, str] = {
    "EXP": "S2R",
    "HOURS": "3",                  # the run bound; an admitted segment can outlive the stop (review 2026-11-05 MEDIUM 1, [A])
    "RUN_BUDGET_HOURS": "1",       # exit 4 seals the tree with sentinels and passes on a registered profile
    "RUN_CUT_BY_BUDGET": "1",      # ADR-0190: stage 6 accepts the projection past HOURS; S2B keeps the refusal
    "SEATBELT": "33h",             # the live stage-bound sum is 1,860 min (31 h); 2 h of margin
    "SCORE_BOUND_MIN": "60",       # a one hour tree scores in minutes
    "PULL_MARGIN_MIN": "60",       # a one hour tree pulls in minutes
    "VOLUME_GB": "100",            # the weights live on the container disk; the owner deletes the volume by hand
    "MIN_IN_REGIME_WINDOWS": "",   # the regime counts print only; S2B's floor would fail a one hour tree
    "SKIP_BLOCKED": "1",           # cd-act1 carries one blocked row (B8 on SGLang); the loud skip path runs
}


def _shipped(name: str, *keys: str) -> Dict[str, str]:
    """The profile's values as the master reads them (sourced after _common.env)."""
    fmt = "|".join("${%s:-}" % k for k in keys)
    proc = subprocess.run(
        ["bash", "-c", f'set -a; source "{PROFILES}/_common.env"; source "{PROFILES}/{name}"; printf "%s" "{fmt}"'],
        capture_output=True, text=True, env={"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "")},
    )
    assert proc.returncode == 0, proc.stderr
    return dict(zip(keys, proc.stdout.split("|")))


def _stated(name: str) -> Dict[str, str]:
    """Every KEY=VALUE line the file itself states (comments left out, the last statement wins)."""
    out: Dict[str, str] = {}
    for line in (PROFILES / name).read_text(encoding="utf-8").splitlines():
        m = re.fullmatch(r"([A-Z_][A-Z0-9_]*)=(.*)", line)
        if m:
            out[m.group(1)] = m.group(2).strip('"')
    return out


def test_s2r_pins_the_rehearsal_levers_and_stays_a_registered_profile() -> None:
    """The nine levers hold the Spec Block's values; the two keys that would relax the
    registered-profile block stay at their S2B values (REHEARSAL_N is refused on cd-act1
    by the driver, PARTIAL_TREE_OK=1 is refused at stage 0 without it); the plan is whole."""
    v = _shipped("S2R.env", *LEVERS, "SESSION", "REHEARSAL_N", "PARTIAL_TREE_OK", "LAST_BLOCK", "MODEL", "GPU_ID", "GPU_COUNT")
    for key, want in LEVERS.items():
        assert v[key] == want, (key, v[key], want)
    assert v["SESSION"] == "cd-act1" and v["MODEL"] == "Qwen/Qwen3-Next-80B-A3B-Instruct"
    assert (v["GPU_ID"], v["GPU_COUNT"]) == ("NVIDIA H200", "4")
    assert v["REHEARSAL_N"] == "" and v["PARTIAL_TREE_OK"] == "0" and v["LAST_BLOCK"] == ""
    # the landing of record keeps the whole-plan refusal of review 2026-10-21 M2
    assert _shipped("S2B.env", "RUN_CUT_BY_BUDGET")["RUN_CUT_BY_BUDGET"] == ""


def test_s2r_states_every_s2b_key_and_differs_only_on_the_levers() -> None:
    """A later S2B edit that S2R does not mirror fails here, so the rehearsal keeps
    rehearsing the landing it stands for; a lever that drifts back to S2B's value fails too."""
    r, b = _stated("S2R.env"), _stated("S2B.env")
    assert set(b) <= set(r), {"S2B only": sorted(set(b) - set(r))}
    assert set(r) - set(b) <= set(LEVERS), {"S2R only, not a lever": sorted(set(r) - set(b) - set(LEVERS))}
    same = {k for k in b if k not in LEVERS and r[k] != b[k]}
    assert not same, {k: (r[k], b[k]) for k in sorted(same)}
    for k in LEVERS:
        assert r[k] != b.get(k, ""), (k, r[k])


def test_s2r_stage_bounds_sum_to_thirty_one_hours_under_the_seatbelt() -> None:
    """The master's live seatbelt need (cage_experiment.sh stage_bound_minutes, lines 222 to 266):
    readiness 15 + setup + validate (20, the pod suite 30, 25 per server engine, the start check 40,
    the identity gate and the PD smoke on cd-act1) + calibrate (60 + the rung bound per engine)
    + plan 15 + run + seal 10 + score + collect 30 + pull. The money gate (line 1015) charges that
    sum at the live H200 secure price per GPU times four plus the $50 floor: $705.96 at the
    $5.29 read on 2026-11-04 against a $698.73 balance (an $8 top-up or more clears it)."""
    v = _shipped("S2R.env", "SETUP_BOUND_MIN", "ENGINES", "CALIBRATE", "CALIBRATE_RUNGS_BOUND_S", "HOURS",
                 "SCORE_BOUND_MIN", "PULL_MARGIN_MIN", "SEATBELT", "SUITE_ON_POD", "START_CHECK",
                 "IDENTITY_GATE_BOUND_S", "PD_SMOKE_BOUND_S", "CLASS_CEILINGS")
    n_srv = len([e for e in v["ENGINES"].split() if e != "hf"])
    n_cal = len(v["CALIBRATE"].split())
    validate = 20 + (30 if v["SUITE_ON_POD"] == "1" else 0) + 25 * n_srv + (40 if v["START_CHECK"] == "1" else 0) \
        + int(v["IDENTITY_GATE_BOUND_S"]) // 60 + int(v["PD_SMOKE_BOUND_S"]) // 60
    calibrate = n_cal * 60 + n_cal * int(v["CALIBRATE_RUNGS_BOUND_S"]) // 60
    plan = 15 + (0 if v["CLASS_CEILINGS"] == "0" else 2 * 600 + 60)
    need = 15 + int(v["SETUP_BOUND_MIN"]) + validate + calibrate + plan + int(v["HOURS"]) * 60 + 10 \
        + int(v["SCORE_BOUND_MIN"]) + 30 + int(v["PULL_MARGIN_MIN"])
    assert (validate, calibrate, plan, need) == (240, 1200, 15, 1860)
    assert v["SEATBELT"] == "33h" and 33 * 60 >= need + 120
    assert round(need / 60 * 4 * 5.29 + 50, 2) == 705.96
