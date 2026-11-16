"""S2R 2026-11-14 follow-ups, item 5 (owner 2026-11-15, "ok A then"): the frontier step.
FRONTIER_LOADS=1 makes stage 6 compute the K1 frontier file from the anchor ladders and
refuse when a rung's lowest window misses the 0.9 timely share. Only a "frontier" load
reads that file, and only S1's K1 block registers one; S2A.env and S2B.env carried the
key from S1, so on their sessions the step could change nothing in the plan and could
only refuse (landing S2R 2026-11-14: 2.9 h of pod time at stage 6).
Spec: MyDocs/RunPod/S2R_2026-11-14_FOLLOWUPS.md, "Item 5 re-read".
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import test_experiment_master as tm

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILES = REPO_ROOT / "scripts" / "6_experiments" / "profiles"


def _session_and_key(profile: Path) -> tuple[str, str]:
    """SESSION and FRONTIER_LOADS as the master reads them (the profile sourced after _common.env)."""
    proc = subprocess.run(
        ["bash", "-c", f'set -a; source "{PROFILES}/_common.env"; source "{profile}"; printf "%s|%s" "$SESSION" "${{FRONTIER_LOADS:-0}}"'],
        capture_output=True, text=True, env={"PATH": os.environ.get("PATH", "")})
    assert proc.returncode == 0, proc.stderr
    session, key = proc.stdout.split("|")
    return session, key


def _has_frontier_load(rc, session: str) -> bool:
    grid = rc.get_session_grid(session)
    return any(load.kind == "frontier" for point in (grid.blocks or ()) for load in point.loads)


def test_the_frontier_step_is_on_exactly_when_the_sessions_grid_has_a_frontier_load_item5() -> None:
    """Every shipped profile of a registered session: on without a frontier load is a
    refusal with no consumer (S2R); off with one is a plan the driver refuses."""
    rc = tm._run_campaign_module()
    judged = {}
    for profile in sorted(PROFILES.glob("S*.env")):
        if profile.read_text(encoding="utf-8").startswith("# SUPERSEDED"):
            continue
        session, key = _session_and_key(profile)
        try:
            need = _has_frontier_load(rc, session)
        except rc.PlanError:
            continue                      # a placeholder profile for a session the driver does not register yet
        judged[profile.name] = (key, need)
        assert (key == "1") == need, f"{profile.name}: FRONTIER_LOADS={key!r} but session {session!r} {'has' if need else 'has no'} frontier load"
    # the rule was judged on the landings that matter, and both directions occur
    assert {"S1.env", "S2A.env", "S2B.env", "S2R.env"} <= set(judged), sorted(judged)
    assert judged["S1.env"] == ("1", True) and judged["S2B.env"] == ("0", False), judged


def test_no_s2_block_registers_a_frontier_load_and_s1_k1_does_item5() -> None:
    """The S2 loads are 0.95 and 1.05 of the cell's own rate (plan section 8, choice 23);
    a frontier block added to an S2 session later fails here and names the profile key."""
    rc = tm._run_campaign_module()
    for session in ("s2a", "cd-act1"):
        kinds = {load.kind for point in rc.get_session_grid(session).blocks for load in point.loads}
        assert kinds == {"own"}, f"session {session}: load kinds {sorted(kinds)}; a frontier load needs FRONTIER_LOADS=1 in its profile"
    k1 = [p for p in rc.get_session_grid("s1").blocks if p.block == "K1"]
    assert k1 and all(load.kind == "frontier" for p in k1 for load in p.loads)
