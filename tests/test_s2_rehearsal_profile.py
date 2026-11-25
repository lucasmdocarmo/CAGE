"""S2 rehearsal profile S2R.env (owner 2026-11-04, option R1, "lets do the test with
the h200 as designed"): the 80B session cd-act1 on four H200, both engines, every
ladder, driven through every stage of the master once and cut short at the run. The
file is never a landing of record. ADR-0190 (owner 2026-11-05, option D):
RUN_CUT_BY_BUDGET=1 lets stage 6 accept the 38 h plan projection because the budget
stop cuts the run first. Spec: MyDocs/RunPod/S2_REHEARSAL_LANDING_2026-11-05.md.

Until 2026-11-22 these tests held S2R.env to S2B.env on every key but nine levers,
because S2B.env was the 80B landing the rehearsals stood for. Owner 2026-11-21 ("ok
drop B7 from both, 8 arms. lets go") and 2026-11-22 ("Yes, all seven", decision B-D3
(a); MyDocs/RunPod/S2_PLAN_A_B_QWEN3_235B_2026-11-19.md section 20): experiment B is
session s2b on Qwen3-235B, S2B.env names it, and S2R.env stays the 80B rehearsal
profile as it stood at commit eadfd50. Its values are pinned here on their own
(STATED): a change to any of them fails, whatever S2B.env says.
tests/test_s2b_registration.py holds the new S2B.env. Review 2026-11-22 (L1): the
file is not what every rehearsal ran; each landing froze its values in its own
extras/profile.env (experiments/S2R/<date>/).
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILES = REPO_ROOT / "scripts" / "6_experiments" / "profiles"

#: the keys the rehearsal cut from the 80B landing it stood for, and why (the Spec Block
#: of 2026-11-04); the 80B landing's own values were HOURS 60, RUN_BUDGET_HOURS 55, no
#: RUN_CUT_BY_BUDGET, SEATBELT 98h, SCORE_BOUND_MIN 150, PULL_MARGIN_MIN 180, VOLUME_GB
#: 300 and MIN_IN_REGIME_WINDOWS "0.95:2 1.05:3" (S2B.env up to commit eadfd50)
LEVERS: Dict[str, str] = {
    "EXP": "S2R",
    "HOURS": "3",                  # the run bound; an admitted segment can outlive the stop (review 2026-11-05 MEDIUM 1, [A])
    "RUN_BUDGET_HOURS": "1.25",    # exit 4 seals the tree with sentinels and passes on a registered profile; 1.25 (owner 2026-11-05) admits the 3,277 s first segment with 1,223 s of spare, not 323
    "RUN_CUT_BY_BUDGET": "1",      # ADR-0190: stage 6 accepts the projection past HOURS; a landing of record keeps the refusal
    "SEATBELT": "35h",             # the live stage-bound sum is 1,955 min (32.6 h) since ADR-0196 (pod suite 125 min); 145 min of margin
    "SCORE_BOUND_MIN": "60",       # a one hour tree scores in minutes
    "PULL_MARGIN_MIN": "60",       # a one hour tree pulls in minutes
    "VOLUME_GB": "100",            # the weights live on the container disk; the owner deletes the volume by hand
    "MIN_IN_REGIME_WINDOWS": "",   # the regime counts print only; the landing's floor would fail a one hour tree
}

#: every KEY=VALUE line S2R.env states, read on 2026-11-22 from commit eadfd50: the 80B
#: rehearsal profile as it stood at that commit (59 keys). Five of the values were set
#: after the last rehearsal landing (2026-11-14) and have not run on a pod: FRONTIER_LOADS
#: (1 on every landing), DISK_GB (300), BACKUP_TARGET (file:///workspace/backup),
#: POD_RESULTS_DIR and POD_BACKUP_DIR (unset) [V the landings' extras/profile.env]
STATED: Dict[str, str] = {
    "EXP": "S2R",
    "SESSION": "cd-act1",
    "CALIBRATION_BUDGET_FRACTION": "1.0",
    "CLASS_FLOORS": "1",
    "FRONTIER_LOADS": "0",
    "QUERY_MANIFESTS": "musique=data/manifests/musique_2000x3_seed42_wr50_tok-Qwen3-Next-80B-A3B-Instruct.json",
    "CALIBRATION_MANIFEST": "data/manifests/musique_2000x3_seed42_wr50_tok-Qwen3-Next-80B-A3B-Instruct.json",
    "VOLUME_GB": "100",
    "DISK_GB": "400",
    "RUN_BUDGET_HOURS": "1.25",
    "RUN_CUT_BY_BUDGET": "1",
    "LAST_BLOCK": "",
    "PIN_MODEL_REVISION": "1",
    "CAMPAIGN": "camp1",
    "MODEL": "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "MODEL_SLUG": "qwen3-next-80b",
    "GPU_ID": "NVIDIA H200",
    "GPU_COUNT": "4",
    "ENGINES": "vllm sglang",
    "CALIBRATE": "vllm sglang",
    "TOPOLOGY": "tp",
    "DC_PREFS": "US-CA-2 US-CO-1 CA-MTL-3 CA-MTL-4 EU-FR-1 AP-JP-1",
    "HOURS": "3",
    "SETUP_BOUND_MIN": "50",
    "CALIBRATE_RUNGS_BOUND_S": "32400",
    "SCORE_BOUND_MIN": "60",
    "SCORING_BATCH_SIZE": "32",
    "SCORING_TOL_FAITHFULNESS": "0.005",
    "SCORING_TOL_BERTSCORE": "1e-5",
    "PULL_MARGIN_MIN": "60",
    "SEATBELT": "35h",
    "CLASS_CEILINGS_BOUND_S": "36000",
    "CLASS_CEILINGS": "0",
    "RUNG_ALL_CLASSES": "1",
    "PREFETCH_MODELS": (
        "Qwen/Qwen3-Next-80B-A3B-Instruct intfloat/e5-large-v2 BAAI/bge-reranker-large "
        "microsoft/llmlingua-2-xlm-roberta-large-meetingbank KRLabsOrg/lettucedect-base-modernbert-en-v1 "
        "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli sentence-transformers/all-MiniLM-L6-v2 roberta-base"
    ),
    "STEP_TIMEOUT_S": "7200",
    "LADDER_FAIL_FAST": "1",
    "CELL_STALL_MIN": "60",
    "BACKUP_ARCHIVE_DIR": "/workspace/backup-archives",
    "POD_RESULTS_DIR": "/root/cage-results",
    "POD_BACKUP_DIR": "/root/backup",
    "BACKUP_TARGET": "file:///root/backup",
    "REGIME_GATE_LOADS": "0.95 1.05",
    "MIN_IN_REGIME_WINDOWS": "",
    "CONFIRMATORY_ARMS": "B1 B3 B4 B6 B7",
    "MAX_NULL_FRACTION": "0.06",
    "FLOOR_CONCURRENCY": "50",
    "FLOOR_AVG_SEQ_TOKENS": "2694",
    "SUITE_ON_POD": "1",
    "POD_SUITE_BOUND_S": "7500",
    "MAC_SUITE": "1",
    "FRONT_LOAD": "0",
    "CAGE_SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH": "1",
    "INSTALL_LMCACHE": "1",
    "LMCACHE_VERSION": "0.4.4",
    "SKIP_BLOCKED": "1",
    "IDENTITY_FAIL_ACCEPTED": "1",
    "PD_FAIL_ACCEPTED": "1",
    "HF_TOKEN_FILE": "",
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
    registered-profile block stay off (REHEARSAL_N is refused on cd-act1 by the driver,
    PARTIAL_TREE_OK=1 is refused at stage 0 without it); the plan is whole."""
    v = _shipped("S2R.env", *LEVERS, "SESSION", "REHEARSAL_N", "PARTIAL_TREE_OK", "LAST_BLOCK", "MODEL", "GPU_ID", "GPU_COUNT")
    for key, want in LEVERS.items():
        assert v[key] == want, (key, v[key], want)
    assert v["SESSION"] == "cd-act1" and v["MODEL"] == "Qwen/Qwen3-Next-80B-A3B-Instruct"
    assert (v["GPU_ID"], v["GPU_COUNT"]) == ("NVIDIA H200", "4")
    assert v["REHEARSAL_N"] == "" and v["PARTIAL_TREE_OK"] == "0" and v["LAST_BLOCK"] == ""
    # the landings of record keep the whole-plan refusal of review 2026-10-21 M2
    for landing in ("S2A.env", "S2B.env"):
        assert _shipped(landing, "RUN_CUT_BY_BUDGET")["RUN_CUT_BY_BUDGET"] == "", landing


def test_s2r_keeps_every_value_the_80b_profile_stated_at_eadfd50() -> None:
    """Decision B-D3 (a) (owner 2026-11-22): the 80B rehearsal profile is pinned on its
    own. Before that day this test read the non-lever values from S2B.env, which then
    named the same session and model; S2B.env now names session s2b."""
    stated = _stated("S2R.env")
    assert stated == STATED, {
        "changed": {k: (stated.get(k), STATED.get(k)) for k in sorted(set(stated) | set(STATED)) if stated.get(k) != STATED.get(k)},
    }
    assert len(STATED) == 59 and all(STATED[k] == v for k, v in LEVERS.items())
    # the master reads what the file states (no key is overridden by a later line or by _common.env)
    assert _shipped("S2R.env", *STATED) == STATED
    # and the record no longer follows the landing profile: S2B.env names another session and model
    b = _stated("S2B.env")
    assert (b["SESSION"], b["MODEL_SLUG"]) != (STATED["SESSION"], STATED["MODEL_SLUG"])


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


def test_the_80b_profile_accepts_the_identity_fail_and_the_pd_smoke_fail_in_advance() -> None:
    """Item 8 (owner 2026-11-11, option (c)): the identity gate FAIL of 2026-11-10 (comparison A,
    187 of 200 prompts, on the pinned weights and engines cd-act1 serves) is a result an 80B
    landing carries, so its profile states IDENTITY_FAIL_ACCEPTED=1: the gate still runs once on
    the new pod, and a FAIL is recorded and excludes the vLLM prefix-on cells without stopping
    stage 4. PD_FAIL_ACCEPTED is 1 since 2026-11-16 (owner: "Drop PD from S2B", option A,
    ADR-0208, AMD-003): the smoke failed the same way on both S2R pods (vLLM 0.19.1 asserts
    "External KV connector is not verified yet" under --mamba-cache-mode align, scheduler.py
    line 305). Until 2026-11-22 this test read both keys from S2B.env, then the 80B landing;
    S2R.env states the same two values and is the 80B profile that remains. Session s2b has
    neither gate, and S2B.env states neither key (tests/test_s2b_registration.py)."""
    assert _shipped("S2R.env", "IDENTITY_FAIL_ACCEPTED", "PD_FAIL_ACCEPTED") == {"IDENTITY_FAIL_ACCEPTED": "1", "PD_FAIL_ACCEPTED": "1"}
    stated = _stated("S2R.env")
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
