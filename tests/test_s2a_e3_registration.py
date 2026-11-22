"""S2 experiment A as ruled on 2026-11-19 (variant E3): the s2a registration.

The owner's rulings (MyDocs/RunPod/S2_PLAN_A_B_QWEN3_235B_2026-11-19.md,
sections 11 and 12):
- every arm runs the same grid on both engines: cache rungs (1.0, 0.25) by
  loads 0.95 (two windows) and 1.05 (three windows); the one-point block X2
  and the arms B11 and B12 leave the session. The arms were nine on that day
  (B1, B2, B3, B4, B5, B6, B7, B9, B10) and are eight since 2026-11-21: B7
  left (owner: "ok drop B7 from both, 8 arms. lets go"; plan section 19;
  tests/test_s2a_eight_arms.py), and the pins below follow that ruling;
- A-S1: the rungs come from a session constant ``S2A_RUNGS``; ``S2_RUNGS``
  keeps (1.0, 0.5, 0.25) because cd-act1 reads it;
- A-S2: the ladders run on (1.0, 0.25) only;
- D1: N = 846 on Qasper, the trial pool of the manifest packed in
  Llama-3.3-70B tokens (the analysis realized-n floor is a confirmatory check
  and S2 is analyzed in the design-input mode, AN-5, so it is not re-keyed);
- A-S7: B5 over BM25 is a no-load reference on each engine;
- A-G1 and A-G2: only load 1.05 gates, three windows per group, and the
  confirmatory roster is the core arms (five on 2026-11-19, four without B7).

The review of 2026-11-20 (MyDocs/RunPod/S2A_E3_REVIEW_2026-11-20.md) added the
last section: the run bounds against the plan's projection (M1), the reach of
the chained 0.25 ladder (M2), the stage 14 gate, the stage 0 pool rule and the
plan on the built manifest (L1), and the n of a future reference dataset (L4).

Fixtures: tests/test_run_campaign.py (the S2 plan stack),
tests/test_batch_b_pressure.py (the ladder's fake pod),
tests/test_s2_block_f_batch3.py (the gate's window tree).
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_batch_b_pressure as bp  # noqa: E402
import test_run_campaign as trc  # noqa: E402
import test_s2_block_f_batch3 as f3  # noqa: E402
from src.analysis import regime_gate_rules as rules  # noqa: E402

rc = trc.rc

#: eight arms since 2026-11-21 (B7 left; nine before)
ARMS = ("B1", "B2", "B3", "B4", "B5", "B6", "B9", "B10")
#: the fresh arms enumerate as F2 and the reuse arms as F3 (cellspec FRESH_SET, REUSE_SET)
F2_ARMS = {"B1", "B5", "B6", "B9"}
F3_ARMS = {"B2", "B3", "B4", "B10"}
CORE_ARMS = "B1 B3 B4 B6"
MANIFEST = "data/manifests/qasper_846x12_seed42_wr50_tok-Llama-3.3-70B-Instruct.json"
#: sha256 of repr(get_session_grid(session)) before the s2a change: cd-act1 read on
#: 2026-11-19; a, b and s1 read on 2026-11-20 from commit 3fba8aa and from the working
#: tree (equal, and equal under PYTHONHASHSEED 0, 1 and 12345)
CD_ACT1_GRID_SHA256 = "cc890cf52232ab067dc64a1039eb7772fb31836a077389b3580d29d24f2ec748"
OTHER_GRID_SHA256 = {
    "a": "f0c2ac0987b6de08e7cbb67b698ad036671f8783d25a755e55831da2c2b1f147",
    "b": "42e0b61a0196daaf436f04b487d827857fc4238767bee9b658a2faab62c0a6b6",
    "s1": "9192110326c0a6f7092fe39f874b3af3c9d25b1689c87bc5718a40fbb0c8c4e7",
}


@pytest.fixture(scope="module")
def plan(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    return trc._s2_plan("s2a", tmp_path_factory.mktemp("s2a_e3"))


def _rows(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    return trc._s2_rows(plan)


# --------------------------------------------------------------------------- #
# the grid
# --------------------------------------------------------------------------- #


def test_the_session_rungs_are_their_own_constant_and_the_shared_one_is_untouched() -> None:
    """A-S1: s2a reads S2A_RUNGS; S2_RUNGS still feeds cd-act1."""
    assert rc.S2A_RUNGS == (1.0, 0.25)
    assert rc.S2_RUNGS == (1.0, 0.5, 0.25)
    assert rc.S2_LOADS == ((0.95, 2), (1.05, 3))


def test_every_arm_runs_the_same_grid_in_one_block() -> None:
    grid = rc.get_session_grid("s2a")
    assert [p.block for p in grid.blocks] == ["X1", "X1"]
    assert all(p.baselines == ARMS for p in grid.blocks)
    assert all(p.engines == ("vllm", "sglang") for p in grid.blocks)
    assert all(p.rungs == (1.0, 0.25) for p in grid.blocks)
    assert [(p.loads[0].frac, p.windows) for p in grid.blocks] == [(0.95, 2), (1.05, 3)]
    assert all(p.topology == "single" and p.dataset == "qasper" for p in grid.blocks)


def test_the_references_are_the_arms_plus_b5_over_bm25() -> None:
    """A-S7: one no-load reference per arm and engine, and B5 over BM25."""
    grid = rc.get_session_grid("s2a")
    assert grid.f1_baselines == ARMS
    assert grid.f1_retriever_cells == (("B5", "vllm", "bm25"), ("B5", "sglang", "bm25"))


def test_no_cell_of_the_session_serves_a_removed_arm() -> None:
    cells = rc.enumerate_cells(rc.get_session_grid("s2a"))
    assert {c.baseline_id for c in cells} == set(ARMS)


def test_the_ladders_run_on_the_two_ruled_rungs() -> None:
    """A-S2: no ladder on an unused 0.5 rung."""
    grid = rc.get_session_grid("s2a")
    assert grid.f2_budgets == grid.f3_budgets == (1.0, 0.25)
    assert grid.f2_rates == grid.f3_rates == (0.95, 1.05)
    assert grid.floor_budget_r == 1.0
    assert rc.budgeted_rungs(grid) == (1.0, 0.25)


def test_the_registered_n_is_the_llama_manifests_trial_pool() -> None:
    """D1: 846, the trial pool at the 2,640-token block and the 50-question reserve."""
    assert dict(rc.get_session_grid("s2a").achievable_n) == {"qasper": 846}
    # sessions a and s1 keep the 2026-10-17 manifest's 852
    assert dict(rc.get_session_grid("a").achievable_n) == {"qasper": 852}
    assert dict(rc.get_session_grid("s1").achievable_n) == {"qasper": 852}


def test_cd_act1_is_registered_exactly_as_before() -> None:
    """Regression pin (passes before and after): the s2a change must not move
    one field of the 80B session, which shares S2_RUNGS and _s2_points."""
    digest = hashlib.sha256(repr(rc.get_session_grid("cd-act1")).encode()).hexdigest()
    assert digest == CD_ACT1_GRID_SHA256


@pytest.mark.parametrize("session", sorted(OTHER_GRID_SHA256))
def test_the_other_sessions_are_registered_exactly_as_before(session: str) -> None:
    """Review 2026-11-20 L1: the same pin for the sessions that share the
    driver's constants with s2a (a and s1 read Qasper; s2a is derived from b's grid and
    took b's demand tables until register A-5 gave it its own on 2026-11-21)."""
    digest = hashlib.sha256(repr(rc.get_session_grid(session)).encode()).hexdigest()
    assert digest == OTHER_GRID_SHA256[session]


# --------------------------------------------------------------------------- #
# the plan
# --------------------------------------------------------------------------- #


def test_the_plan_counts(plan: Dict[str, Any]) -> None:
    rows = _rows(plan)
    # 8 arms x 2 engines x 2 rungs x 2 loads; 2 + 3 windows per (arm, engine, rung)
    # (72 cells in X1, 20 references, 92 cells and 200 windows with B7, before 2026-11-21)
    assert Counter(s.get("block") or ("ref-" + s["family"]) for s in rows) == {"X1": 64, "ref-F1": 18}
    assert plan["counts"]["cells"] == len(rows) == 82
    assert plan["counts"]["windows"] == 64 * 5 // 2 + 18 == 178
    assert {(s["baseline"], s["family"]) for s in rows if s.get("block")} == (
        {(b, "F2") for b in F2_ARMS} | {(b, "F3") for b in F3_ARMS}
    )
    assert plan["counts"]["by_family"] == {
        "F1": {"cells": 18, "windows": 18},
        "F2": {"cells": 4 * 8, "windows": 4 * 8 * 5 // 2},
        "F3": {"cells": 4 * 8, "windows": 4 * 8 * 5 // 2},
    }


def test_nothing_is_blocked_and_the_rungs_are_the_ruled_two(plan: Dict[str, Any]) -> None:
    assert plan["counts"]["blocked"] == 0 and plan["blocked_row_keys"] == []
    assert plan["rung_calibration"]["required_rungs"] == {"sglang": [1.0, 0.25], "vllm": [1.0, 0.25]}
    assert {s["cellspec"]["budget_r"] for s in _rows(plan) if s.get("block")} == {1.0, 0.25}


def test_the_primary_references_serve_846_questions(plan: Dict[str, Any]) -> None:
    rows = _rows(plan)
    primary = [s for s in rows if s["row_class"] == "primary"]
    assert sorted(s["row_key"] for s in primary) == [
        "corpus-reuse|none|none|single|vllm|llama-3.3-70b|F1",
        "retr-fresh|rerank|none|single|vllm|llama-3.3-70b|F1",
    ]
    assert {s["num_queries"] for s in primary} == {846}
    assert {s["num_queries"] for s in rows if s["row_class"] == "window"} == {500}
    assert sum(1 for s in rows if s["row_class"] == "window") == 64


def test_every_reference_serves_the_same_846_questions(plan: Dict[str, Any]) -> None:
    """Owner 2026-11-19 ("All references at 846"): the secondary references
    serve the full N too (800 before), so every arm and engine is read on the
    same question count; the row classes themselves are unchanged. They are 16
    of 18 references since 2026-11-21 (18 of 20 with B7)."""
    refs = [s for s in _rows(plan) if s["family"] == "F1"]
    assert len(refs) == 18
    assert Counter(s["row_class"] for s in refs) == {"secondary": 16, "primary": 2}
    assert {s["num_queries"] for s in refs} == {846}
    assert all(trc.rc._argv_flag_value(s["argv"], "--num-queries") == "846" for s in refs)
    # the other sessions keep the registered secondary n
    assert rc.get_session_grid("cd-act1").n_secondary == rc.N_SECONDARY == 800
    assert rc.get_session_grid("s1").n_secondary == 800


def test_the_relaunches_and_dry_windows(plan: Dict[str, Any]) -> None:
    """Seven budgeted serving groups (B5 shares B6's; eight with B7, before
    2026-11-21) at two rungs on two engines, plus the budget-free reference
    relaunches; one dry window per budgeted serving shape (B10's fp8 shape is
    its own). 38 relaunches and 16 dry windows with B7."""
    assert plan["counts"]["relaunches"] == 34
    assert plan["counts"]["dry_windows"] == 14
    budgeted = [s for s in plan["steps"] if s["kind"] == "relaunch" and s.get("budget_r") is not None]
    assert len(budgeted) == 7 * 2 * 2
    assert {s["budget_r"] for s in budgeted} == {1.0, 0.25}


# --------------------------------------------------------------------------- #
# the profile
# --------------------------------------------------------------------------- #


def test_the_shipped_profile_gates_load_105_only_on_the_core_arms() -> None:
    """A-G1 and A-G2: the 0.95 column is descriptive and the core arms block
    (four since 2026-11-21; five with B7)."""
    v = f3._shipped("S2A.env", "SESSION", "REGIME_GATE_LOADS", "MIN_IN_REGIME_WINDOWS", "CONFIRMATORY_ARMS")
    assert v == {
        "SESSION": "s2a", "REGIME_GATE_LOADS": "1.05",
        "MIN_IN_REGIME_WINDOWS": "1.05:3", "CONFIRMATORY_ARMS": CORE_ARMS,
    }
    doc = rc.regime_gate_projection(
        "s2a", loads=v["REGIME_GATE_LOADS"], need=v["MIN_IN_REGIME_WINDOWS"],
        confirmatory_arms=v["CONFIRMATORY_ARMS"],
    )
    assert doc["loads_without_a_group"] == [] and doc["short_descriptive"] == {} and doc["unresolved_load_cells"] == 0
    assert all(g["load"] == "1.05" and g["need"] == g["registered"] == 3 for g in doc["groups"].values())
    modes = [g["mode"] for g in doc["groups"].values()]
    # 8 arms x 2 rungs x 2 engines at load 1.05; the four core arms block
    assert (modes.count(rules.BLOCKING), modes.count(rules.DESCRIPTIVE)) == (4 * 2 * 2, 4 * 2 * 2)
    assert rules.parse_floor(v["MIN_IN_REGIME_WINDOWS"], rules.parse_loads(v["REGIME_GATE_LOADS"])) == {"1.05": 3}


def test_the_shipped_profile_names_the_built_llama_manifest() -> None:
    v = f3._shipped("S2A.env", "QUERY_MANIFESTS", "CALIBRATION_MANIFEST")
    assert v["QUERY_MANIFESTS"] == f"qasper={MANIFEST}"
    assert v["CALIBRATION_MANIFEST"] == MANIFEST
    assert (REPO_ROOT / MANIFEST).is_file()


# --------------------------------------------------------------------------- #
# AN-4: the run id of a model slug with dots
# --------------------------------------------------------------------------- #


def _mint(session: str, slug: str) -> str:
    import subprocess

    lib = REPO_ROOT / "scripts" / "lib" / "_common.sh"
    proc = subprocess.run(
        ["bash", "-c", f'source "{lib}"; mint_campaign_run_id "$1" "$2"', "mint", session, slug],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_the_s2a_run_id_is_mintable_under_the_run_id_grammar() -> None:
    """AN-4: the slug llama-3.3-70b carries dots, which the section 1 grammar
    (bucket-name safe, [a-z0-9-]) refuses; the mint writes them as dashes."""
    rid = _mint("s2a", rc.get_session_grid("s2a").model)
    assert trc._org.RUN_ID_RE.match(rid), rid
    assert rid.endswith("-s2a-llama-3-3-70b")


def test_a_slug_without_dots_mints_as_before() -> None:
    rid = _mint("cd-act1", "qwen3-next-80b")
    assert trc._org.RUN_ID_RE.match(rid), rid
    assert rid.endswith("-cd-act1-qwen3-next-80b")


# --------------------------------------------------------------------------- #
# review 2026-11-20: the bounds, the ladder reach, the gate, the pool, the manifest
# --------------------------------------------------------------------------- #


def test_the_run_bounds_hold_the_plans_projection(plan: Dict[str, Any]) -> None:
    """M1: stage 6 compares the projection with HOURS only, and the runner's
    budget stop refuses every segment that would end past RUN_BUDGET_HOURS, so
    a projection between the two leaves a partial landing (at the table's rates
    a 45 h stop cut 16 of the 20 references, 2026-11-20). Since 2026-11-21 the
    grid has eight arms and projects 42.55 h; the limits stay 55 and 50 (owner
    2026-11-22, tests/test_s2a_eight_arms.py). Limitation: the fixture's rates,
    not a pod's."""
    v = f3._shipped("S2A.env", "HOURS", "RUN_BUDGET_HOURS", "SEATBELT")
    rp = plan["run_projection"]
    assert rp["unmeasured_steps"] == []
    assert rp["total_h"] < float(v["RUN_BUDGET_HOURS"]) < float(v["HOURS"])
    assert float(v["HOURS"]) - float(v["RUN_BUDGET_HOURS"]) == 5.0   # S2A.env: the stop five hours before the bound


def test_the_chained_quarter_rung_reaches_a_fourfold_capacity_drop_and_no_further(tmp_path: Path) -> None:
    """M2: on (1.0, 0.25) the 0.25 ladder chains from the 1.0 lambda*, a 4x
    budget step where the chain was sized on 2x. The lowest rate it tries is
    the previous lambda* over LADDER_CHAIN_DIVISOR x 1.3^LADDER_DESCENT_MAX_STEPS
    (4.83), a 1.21 margin over the KV-bound ratio 4: a capacity ratio of 0.25
    brackets on the last descent window, one of 0.20 reads NONE_SUSTAINABLE
    (stage 5 fails under LADDER_FAIL_FAST=1)."""
    reach = rc.LADDER_CHAIN_DIVISOR * bp.PROBE_LADDER_FACTOR ** rc.LADDER_DESCENT_MAX_STEPS
    assert reach / (rc.S2A_RUNGS[0] / rc.S2A_RUNGS[1]) == pytest.approx(1.2067, abs=1e-4)
    grid = trc._tiny_grid(f1_baselines=("B1",), f2_baselines=("B1",), f2_budgets=rc.S2A_RUNGS, f2_rates=(0.85,))
    held = bp._calibrate(tmp_path / "held", bp.FakeWorld(cap=6.0, caps_by_relaunch=[6.0, 1.5]), grid)["rungs"]["0.25"]
    assert (held["label"], held["descent_steps"]) == ("ESTIMATED", rc.LADDER_DESCENT_MAX_STEPS)
    lost = bp._calibrate(tmp_path / "lost", bp.FakeWorld(cap=6.0, caps_by_relaunch=[6.0, 1.2]), grid)["rungs"]["0.25"]
    assert (lost["label"], lost["descent_steps"]) == ("NONE_SUSTAINABLE", rc.LADDER_DESCENT_MAX_STEPS)


def _gate_tree(root: Path, label_of) -> Path:
    """One regime.json per registered window of the 82 s2a cells (92 with B7)."""
    grid = rc.get_session_grid("s2a")
    for c in rc.enumerate_cells(grid):
        key = c.spec.to_row_key()
        f3._write_cell(root, key, c.baseline_id)
        for k in range(1, (c.windows if c.windows is not None else grid.replications) + 1):
            f3._write_window(root, key, c.dataset, k, label_of(c), "stationary")
    return root


def test_the_stage_14_gate_judges_load_105_alone_with_the_shipped_values(tmp_path: Path) -> None:
    """L1 (A-G1, A-G2 at stage 14; the stage 0 projection is tested above):
    every 0.95 window out of regime does not fail the landing, a descriptive
    arm short at 1.05 is a note, a core arm short at 1.05 fails."""
    v = f3._shipped("S2A.env", "REGIME_GATE_LOADS", "MIN_IN_REGIME_WINDOWS", "CONFIRMATORY_ARMS")
    kw = dict(loads=v["REGIME_GATE_LOADS"], confirmatory_arms=v["CONFIRMATORY_ARMS"])
    need = v["MIN_IN_REGIME_WINDOWS"]

    low_out = _gate_tree(tmp_path / "low_out", lambda c: "IN_REGIME" if c.spec.rate_frac == 1.05 else "OVERLOADED")
    res = f3.regime_gate.evaluate(low_out, need, **kw)
    modes = [g["mode"] for g in res["groups"].values()]
    assert res["ok"] is True and res["short"] == {} and res["short_descriptive"] == {}
    assert (len(modes), modes.count(rules.BLOCKING), modes.count(rules.DESCRIPTIVE)) == (32, 16, 16)   # (36, 20, 16) with B7
    assert {g["load"] for g in res["groups"].values()} == {"1.05"}

    b5_short = _gate_tree(tmp_path / "b5", lambda c: "OVERLOADED" if (c.baseline_id, c.spec.rate_frac) == ("B5", 1.05) else "IN_REGIME")
    res = f3.regime_gate.evaluate(b5_short, need, **kw)
    assert res["ok"] is True and res["short"] == {} and len(res["short_descriptive"]) == 4   # 2 engines x 2 rungs

    b6_short = _gate_tree(tmp_path / "b6", lambda c: "OVERLOADED" if (c.baseline_id, c.spec.rate_frac) == ("B6", 1.05) else "IN_REGIME")
    res = f3.regime_gate.evaluate(b6_short, need, **kw)
    assert res["ok"] is False and len(res["short"]) == 4


def test_stage_0_finds_every_pool_wide_enough_at_the_profiles_concurrency() -> None:
    """L1 (acceptance test 8 of plan section 12.3): the registration report on
    the shipped concurrency, anchor, GPU and manifest names no shortfall."""
    v = f3._shipped("S2A.env", "FLOOR_CONCURRENCY", "FLOOR_AVG_SEQ_TOKENS", "GPU_ID")
    assert (REPO_ROOT / MANIFEST).is_file()
    rep = rc.preflight_registration_check(
        "s2a", floor_avg_seq_tokens=v["FLOOR_AVG_SEQ_TOKENS"], query_manifests={"qasper": str(REPO_ROOT / MANIFEST)},
        charter_datasets=("squad_v2", "musique", "qasper"), floor_concurrency=int(v["FLOOR_CONCURRENCY"]), gpu_id=v["GPU_ID"],
    )
    assert (rep["cells"], rep["executable_cells"]) == (82, 82)   # 92 with B7
    assert rep["pool_shortfalls"] == [] and rep["pool_ceiling_overruns"] == [] and rep["problems"] == []


def test_the_plan_builds_on_the_built_manifest_and_reloads(tmp_path: Path) -> None:
    """L1: the registered n (846) meets the manifest's trial size where the plan
    checks it (_check_manifest_coverage), not only in the grid's own field."""
    assert (REPO_ROOT / MANIFEST).is_file()
    built = trc._s2_plan("s2a", tmp_path, query_manifests={"qasper": str(REPO_ROOT / MANIFEST)})
    assert (built["counts"]["cells"], built["counts"]["windows"], built["counts"]["blocked"]) == (82, 178, 0)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(built), encoding="utf-8")
    assert len(rc.load_plan(path)["steps"]) == len(built["steps"]) == 226   # 92 cells, 200 windows, 254 steps with B7


def test_every_reference_dataset_of_the_session_has_a_registered_n() -> None:
    """L4: n_secondary is N_PRIMARY (2,000) on s2a so that achievable_n alone
    sets the references' n; a reference dataset without an entry would serve
    2,000 questions on every secondary row."""
    grid = rc.get_session_grid("s2a")
    assert grid.n_secondary == rc.N_PRIMARY == 2000
    assert set(grid.f1_datasets) <= set(grid.achievable_n)
