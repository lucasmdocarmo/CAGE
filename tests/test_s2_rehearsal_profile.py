"""S2 rehearsal profile S2R.env (owner 2026-11-04, option R1, "lets do the test with
the h200 as designed"): S2B's exact shape (the 80B on four H200, both engines, every
ladder) driven through every stage of the master once, cut short at the run, before
landing B bills. Nine keys differ from S2B.env and nothing else does; the file is never
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
    "RUN_BUDGET_HOURS": "1.25",    # exit 4 seals the tree with sentinels and passes on a registered profile; 1.25 (owner 2026-11-05) admits the 3,277 s first segment with 1,223 s of spare, not 323
    "RUN_CUT_BY_BUDGET": "1",      # ADR-0190: stage 6 accepts the projection past HOURS; S2B keeps the refusal
    "SEATBELT": "35h",             # the live stage-bound sum is 1,955 min (32.6 h) since ADR-0196 (pod suite 125 min); 145 min of margin
    "SCORE_BOUND_MIN": "60",       # a one hour tree scores in minutes
    "PULL_MARGIN_MIN": "60",       # a one hour tree pulls in minutes
    "VOLUME_GB": "100",            # the weights live on the container disk; the owner deletes the volume by hand
    "MIN_IN_REGIME_WINDOWS": "",   # the regime counts print only; S2B's floor would fail a one hour tree
    # SKIP_BLOCKED=1 (the loud skip of the blocked rows) is no lever since 2026-11-16: S2B.env states 1 too (owner: "ok set SKIP_BLOCKED=1 on S2B")
    # IDENTITY_FAIL_ACCEPTED=1 (owner 2026-11-10: the gate FAIL, A 187/200, accepted) is no lever since item 8 (2026-11-11): S2B.env states 1 too
    # PD_FAIL_ACCEPTED=1 (owner 2026-11-10 on the rehearsal) is no lever since 2026-11-16: S2B.env states 1 too ("Drop PD from S2B", option A)
    # FRONTIER_LOADS=0 (owner 2026-11-15) is no lever since S2B.env states 0 too ("ok A then": no cd-act1 block registers a frontier load); tests/test_s2r_followups_batch3.py pins it
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
    readiness 15 + setup + validate (20, the pod suite POD_SUITE_BOUND_S / 60, 25 per server engine, the start check 40,
    the identity gate and the PD smoke on cd-act1) + calibrate (60 + the rung bound per engine)
    + plan 15 + run + seal 10 + score + collect 30 + pull. The money gate (line 1015) charges that
    sum at the live H200 secure price per GPU times four plus the $50 floor: $739.82 at the
    $5.29 read on 2026-11-04 (ADR-0196 moved the pod suite from 30 to 125 min; $705.96 before)."""
    v = _shipped("S2R.env", "SETUP_BOUND_MIN", "ENGINES", "CALIBRATE", "CALIBRATE_RUNGS_BOUND_S", "HOURS",
                 "SCORE_BOUND_MIN", "PULL_MARGIN_MIN", "SEATBELT", "SUITE_ON_POD", "START_CHECK",
                 "IDENTITY_GATE_BOUND_S", "PD_SMOKE_BOUND_S", "CLASS_CEILINGS", "POD_SUITE_BOUND_S")
    n_srv = len([e for e in v["ENGINES"].split() if e != "hf"])
    n_cal = len(v["CALIBRATE"].split())
    validate = 20 + (int(v["POD_SUITE_BOUND_S"]) // 60 if v["SUITE_ON_POD"] == "1" else 0) + 25 * n_srv + (40 if v["START_CHECK"] == "1" else 0) \
        + int(v["IDENTITY_GATE_BOUND_S"]) // 60 + int(v["PD_SMOKE_BOUND_S"]) // 60
    calibrate = n_cal * 60 + n_cal * int(v["CALIBRATE_RUNGS_BOUND_S"]) // 60
    plan = 15 + (0 if v["CLASS_CEILINGS"] == "0" else 2 * 600 + 60)
    need = 15 + int(v["SETUP_BOUND_MIN"]) + validate + calibrate + plan + int(v["HOURS"]) * 60 + 10 \
        + int(v["SCORE_BOUND_MIN"]) + 30 + int(v["PULL_MARGIN_MIN"])
    assert (validate, calibrate, plan, need) == (335, 1200, 15, 1955)
    assert v["SEATBELT"] == "35h" and 35 * 60 >= need + 120
    # review 2026-11-08 MEDIUM 1: pod_hours_bound (line 292) rounds the hours to one decimal before pricing
    assert round(round(need / 60, 1) * 4 * 5.29 + 50, 2) == 739.82


def test_s2b_accepts_the_identity_fail_and_the_pd_smoke_fail_in_advance() -> None:
    """Item 8 (owner 2026-11-11, option (c)): the identity gate FAIL of 2026-11-10 (comparison A,
    187 of 200 prompts, on the pinned weights and engines S2B serves) is a result landing B
    carries, so the profile states IDENTITY_FAIL_ACCEPTED=1: the gate still runs once on the new
    pod, and a FAIL is recorded and excludes the vLLM prefix-on cells without stopping stage 4.
    PD_FAIL_ACCEPTED was 0 under item 8 (stage 4 stopped on a FAIL for the owner's decision);
    since 2026-11-16 (owner: "Drop PD from S2B", option A, ADR-0208, AMD-003) it is 1, stated:
    the smoke failed the same way on both S2R pods (vLLM 0.19.1 asserts "External KV connector is
    not verified yet" under --mamba-cache-mode align, scheduler.py line 305), so S2B runs it once
    on its own pod, records the FAIL, and its plan carries no pd cell."""
    assert _shipped("S2B.env", "IDENTITY_FAIL_ACCEPTED", "PD_FAIL_ACCEPTED") == {"IDENTITY_FAIL_ACCEPTED": "1", "PD_FAIL_ACCEPTED": "1"}
    stated = _stated("S2B.env")
    assert stated["IDENTITY_FAIL_ACCEPTED"] == "1" and stated["PD_FAIL_ACCEPTED"] == "1"


def test_s2r_and_s2b_keep_the_results_and_the_backup_off_the_volume_followups_item6() -> None:
    """Follow-ups item 6 (owner 2026-11-15, option B), as S0F-39 on S0: the US-CA-2 network
    volume refuses "|" in file names (EINVAL 22, landing 2026-11-14 stage 11, rsync code 23)
    and every cell directory is its row key, so both profiles put the results root and the
    backup mirror on the container disk. The profile value of BACKUP_TARGET must win over
    _common.env's hard-set /workspace one, so the check reads the sourced value too."""
    want = {"POD_RESULTS_DIR": "/root/cage-results", "POD_BACKUP_DIR": "/root/backup", "BACKUP_TARGET": "file:///root/backup"}
    for name in ("S2R.env", "S2B.env"):
        assert _shipped(name, *want) == want, name
        stated = _stated(name)
        assert {k: stated.get(k) for k in want} == want, name
