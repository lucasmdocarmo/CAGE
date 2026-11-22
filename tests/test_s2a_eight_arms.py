"""S2 experiment A with eight arms (owner 2026-11-21: "ok drop B7 from both, 8 arms. lets go").

The ruling (MyDocs/RunPod/S2_PLAN_A_B_QWEN3_235B_2026-11-19.md, section 19):
both S2 experiments serve the same eight baselines, B1, B2, B3, B4, B5, B6, B9
and B10. B7 (RAG with prefix reuse) leaves session s2a: no registered no-load
contrast and no arm pair reads it. These are the acceptance tests of the Spec
Block (section 19.4) and of the time limits. Decision D1 (b) lowered the
limits to 50, 45 and 83h on 2026-11-21. The independent review (M1) showed
that the pod's stage 6 projection moves with its measured floors and with
B9's rate limit, and that a projection above HOURS refuses after calibration
has billed. The owner ruled on 2026-11-22: "back to 55/50/88". The limits are
HOURS 55, the run stop at 50 and the seatbelt at 88h, as on the nine-arm grid.

The other pins of the s2a registration (tests/test_s2a_e3_registration.py,
tests/test_run_campaign.py, tests/test_s2_block_f_batch3.py,
tests/test_s2_block_f_batch4.py, tests/test_s2a_a5_demand_tables.py,
tests/test_experiment_master.py) were re-pinned with this ruling.

Fixtures: tests/test_run_campaign.py (the S2 plan stack),
tests/test_s2_block_f_batch3.py (the shipped profile values).
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import pytest

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_run_campaign as trc  # noqa: E402
import test_s2_block_f_batch3 as f3  # noqa: E402
from src.analysis import regime_gate_rules as rules  # noqa: E402

rc = trc.rc

ARMS = ("B1", "B2", "B3", "B4", "B5", "B6", "B9", "B10")
GATE_ARMS = "B1 B3 B4 B6"
#: the classes the eight arms are served in, in Llama tokens (ADR-0213); 1,008,
#: the class of B7, has no arm on the session any more
CLASSES = [640, 995, 2240, 4978]
B7_CLASS = 1008


@pytest.fixture(scope="module")
def plan(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    return trc._s2_plan("s2a", tmp_path_factory.mktemp("s2a_eight"))


# --------------------------------------------------------------------------- #
# acceptance test 1: the arm list
# --------------------------------------------------------------------------- #


def test_the_session_serves_the_eight_ruled_baselines() -> None:
    grid = rc.get_session_grid("s2a")
    assert rc.S2A_ARMS == ARMS == grid.f1_baselines
    assert all(p.baselines == ARMS for p in grid.blocks)
    cells = rc.enumerate_cells(grid)
    assert {c.baseline_id for c in cells} == set(ARMS)
    assert not any(c.spec.arm == "retr-reuse" for c in cells)


def test_no_step_of_the_plan_names_b7(plan: Dict[str, Any]) -> None:
    assert {s["baseline"] for s in trc._s2_rows(plan)} == set(ARMS)
    assert not any("retr-reuse" in str(s.get("row_key")) for s in plan["steps"])
    assert not any((s.get("demand_class") or {}).get("seq_tokens") == B7_CLASS for s in plan["steps"])


# --------------------------------------------------------------------------- #
# acceptance test 2: the counts
# --------------------------------------------------------------------------- #


def test_the_plan_counts_of_the_eight_arm_grid(plan: Dict[str, Any]) -> None:
    c = plan["counts"]
    # 8 arms x 2 engines x 2 rungs x 2 loads = 64 pressure cells, 2 + 3 windows per
    # (arm, engine, rung); 8 arms x 2 engines of references plus B5 over BM25 twice
    assert (c["cells"], c["windows"], c["relaunches"], c["dry_windows"], c["blocked"]) == (82, 178, 34, 14, 0)
    assert c["by_family"] == {
        "F1": {"cells": 18, "windows": 18},
        "F2": {"cells": 32, "windows": 80},
        "F3": {"cells": 32, "windows": 80},
    }
    assert len(plan["steps"]) == 226
    rows = trc._s2_rows(plan)
    assert Counter(s["row_class"] for s in rows) == {"window": 64, "secondary": 16, "primary": 2}
    assert sum(s["windows"] for s in trc._cells(plan) if s.get("block") == "X1") == 160


def _essence(step: Dict[str, Any]) -> Dict[str, Any]:
    """What a plan step does, without the fields that carry a position, a
    temporary path or a fixture hash (index, argv, env, the rate's source)."""
    budget = step.get("budget_plan") or {}
    keep = ("kind", "engine", "baseline", "family", "row_key", "row_class", "block", "windows", "num_queries",
            "trial_index", "serving", "offered_rate_rps", "lambda_star_rps", "window", "window_arrivals",
            "demand_class", "prefix_mode", "kv_dtype", "budget_r", "budget_bytes", "max_model_len", "topology",
            "tp", "rate_frac", "duration_s")
    out = {k: step.get(k) for k in keep}
    out["budget"] = (budget.get("budget_bytes_total"), budget.get("budget_tokens_total"))
    return out


def _is_b7_step(step: Dict[str, Any]) -> bool:
    return (step.get("baseline") == "B7" or str(step.get("row_key") or "").startswith("retr-reuse|")
            or (step["kind"] == "relaunch" and (step.get("demand_class") or {}).get("seq_tokens") == B7_CLASS))


def test_the_eight_arms_keep_their_steps_when_b7_leaves(
    plan: Dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Spec Block's invariant (second review 2026-11-22, L7: no test compared
    the two plans): the eight-arm plan is the nine-arm plan without B7's steps,
    in the same order, with every remaining step's rate, window, cache size and
    request cap unchanged. The nine-arm grid is registered back in memory."""
    nine = ARMS[:6] + ("B7",) + ARMS[6:]
    grid = rc.get_session_grid("s2a")
    monkeypatch.setitem(rc.SESSION_GRIDS, "s2a", rc.replace(
        grid, f1_baselines=nine, blocks=rc._s2_points("X1", nine, rc._BOTH, "qasper", rungs=rc.S2A_RUNGS),
    ))
    plan_nine = trc._s2_plan("s2a", tmp_path)
    assert (plan_nine["counts"]["cells"], len(plan_nine["steps"])) == (92, 254)
    b7 = [s for s in plan_nine["steps"] if _is_b7_step(s)]
    # B7's steps: 22 cell steps (2 + 3 window steps on each of 2 engines and 2 rungs, and
    # 2 references), 4 relaunches and 2 dry windows (its own class at each rung and engine)
    assert Counter(s["kind"] for s in b7) == {"cell": 22, "relaunch": 4, "dry_window": 2}
    rest = [_essence(s) for s in plan_nine["steps"] if not _is_b7_step(s)]
    assert rest == [_essence(s) for s in plan["steps"]]


# --------------------------------------------------------------------------- #
# acceptance test 3: the ladders
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_the_ladders_cover_the_four_served_classes(engine: str) -> None:
    grid = rc.get_session_grid("s2a")
    assert rc.session_demand_classes(grid) == CLASSES
    selection = rc.ladder_cells(grid, engine, rc.budgeted_rungs(grid), {}, all_classes=True)
    triples = sorted((rc.demand_seq_tokens(grid, c.spec), float(c.spec.budget_r)) for _role, c in selection.roles())
    assert triples == sorted((tokens, r) for tokens in CLASSES for r in (1.0, 0.25))
    assert len(triples) == 8


def test_b7s_class_keeps_its_table_row_and_gets_no_ladder() -> None:
    """The demand table is keyed by arm name: the row of retr-reuse stays, served
    by no arm of the session, as the row of retr-trunc does. No ladder cell of
    either engine is of that class or of that arm (review 2026-11-22, L6: the
    selection itself is asserted, not only the session's class list)."""
    grid = rc.get_session_grid("s2a")
    assert grid.demand_seq_tokens["retr-reuse"] == B7_CLASS
    assert grid.demand_class_max_served_tokens[B7_CLASS] == 3814
    assert B7_CLASS not in rc.session_demand_classes(grid)
    for engine in ("vllm", "sglang"):
        selected = [c for _role, c in rc.ladder_cells(grid, engine, rc.budgeted_rungs(grid), {}, all_classes=True).roles()]
        assert {rc.demand_seq_tokens(grid, c.spec) for c in selected} == set(CLASSES)
        assert not any(c.spec.arm == "retr-reuse" for c in selected)


# --------------------------------------------------------------------------- #
# acceptance test 4: the gate
# --------------------------------------------------------------------------- #


def test_the_shipped_gate_names_four_arms_and_the_grid_serves_each() -> None:
    v = f3._shipped("S2A.env", "REGIME_GATE_LOADS", "MIN_IN_REGIME_WINDOWS", "CONFIRMATORY_ARMS")
    assert v["CONFIRMATORY_ARMS"] == GATE_ARMS
    assert set(GATE_ARMS.split()) <= set(rc.S2A_ARMS)
    doc = rc.regime_gate_projection(
        "s2a", loads=v["REGIME_GATE_LOADS"], need=v["MIN_IN_REGIME_WINDOWS"], confirmatory_arms=v["CONFIRMATORY_ARMS"],
    )
    groups = list(doc["groups"].values())
    blocking = [g for g in groups if g["mode"] == rules.BLOCKING]
    assert (len(groups), len(blocking), sum(g["mode"] == rules.DESCRIPTIVE for g in groups)) == (32, 16, 16)
    assert {g["baseline"] for g in blocking} == set(GATE_ARMS.split())
    # the windows that must all qualify: 16 blocking groups of three (60 with B7)
    assert sum(g["need"] for g in blocking) == 48


def test_a_gate_list_that_still_names_b7_is_refused_at_stage_0() -> None:
    """The failure mode of the Spec Block, as the review of 2026-11-22 corrected
    it (L3): the stage 14 gate forms its groups from the windows on disk
    (scripts/6_experiments/regime_gate.py), so a roster arm the grid does not
    serve would form no group there and the landing would pass unnoticed. The
    stage 0 projection is the only guard, and it refuses."""
    with pytest.raises(rc.PlanError, match=r"CONFIRMATORY_ARMS names \['B7'\]: not an executable pressure baseline of session 's2a'"):
        rc.regime_gate_projection("s2a", loads="1.05", need="1.05:3", confirmatory_arms="B1 B3 B4 B6 B7")


# --------------------------------------------------------------------------- #
# acceptance test 5: the other S2 session
# --------------------------------------------------------------------------- #


def test_the_other_s2_session_keeps_its_arms() -> None:
    """cd-act1 reads S2_CORE_ARMS, which the ruling does not touch (the digests of
    a, b, s1 and cd-act1 are pinned in tests/test_s2a_e3_registration.py)."""
    assert rc.S2_CORE_ARMS == ("B1", "B3", "B4", "B6", "B7")
    assert "B7" in rc.get_session_grid("cd-act1").f1_baselines


# --------------------------------------------------------------------------- #
# acceptance test 6: the limits (decision D1, then the owner's ruling on review M1)
# --------------------------------------------------------------------------- #


def test_the_limits_stay_55_50_and_88h_over_the_eight_arm_projection(plan: Dict[str, Any]) -> None:
    """Owner 2026-11-22 ("back to 55/50/88"): the limits of the nine-arm grid
    are kept on the eight-arm one. The pod's stage 6 projection uses the
    measured floors and B9's rate limit and is compared with HOURS only; a
    refusal there comes after calibration has billed. 50, 45 and 83h stood for
    one day and never reached a pod."""
    v = f3._shipped("S2A.env", "HOURS", "RUN_BUDGET_HOURS", "SEATBELT")
    assert v == {"HOURS": "55", "RUN_BUDGET_HOURS": "50", "SEATBELT": "88h"}
    rp = plan["run_projection"]
    assert rp["unmeasured_steps"] == []
    assert rp["total_h"] == pytest.approx(42.55, abs=0.01)   # 48.45 with nine arms
    # the room over the fixture projection: 12.45 h to HOURS and 7.45 h to the run
    # stop (6.55 h and 1.55 h on the nine-arm grid)
    assert float(v["HOURS"]) - rp["total_h"] == pytest.approx(12.45, abs=0.01)
    assert float(v["RUN_BUDGET_HOURS"]) - rp["total_h"] == pytest.approx(7.45, abs=0.01)


def test_the_stage_bounds_sum_to_5115_minutes_under_the_88h_seatbelt() -> None:
    """Second review 2026-11-22, L7: the sum the profile's comment states, by the
    master's rule (cage_experiment.sh stage_bound_minutes; the same formula as
    tests/test_experiment_master.py holds for every profile), and the balance
    stage 0 asks for at the 5.29 USD per GPU-hour read on 2026-11-21. The
    master rounds the hours to one decimal before pricing (pod_hours_bound)."""
    v = f3._shipped("S2A.env", "SETUP_BOUND_MIN", "ENGINES", "CALIBRATE", "CALIBRATE_RUNGS_BOUND_S", "HOURS",
                    "SCORE_BOUND_MIN", "PULL_MARGIN_MIN", "SEATBELT", "SUITE_ON_POD", "START_CHECK",
                    "POD_SUITE_BOUND_S", "GPU_COUNT")
    assert not rc.get_session_grid("s2a").identity_gated   # so stage 4 carries no identity-gate term
    n_srv = len([e for e in v["ENGINES"].split() if e != "hf"])
    n_cal = len(v["CALIBRATE"].split())
    validate = (20 + (int(v["POD_SUITE_BOUND_S"]) // 60 if v["SUITE_ON_POD"] == "1" else 0) + 25 * n_srv
                + (40 if v["START_CHECK"] == "1" else 0))
    calibrate = n_cal * 60 + n_cal * int(v["CALIBRATE_RUNGS_BOUND_S"]) // 60
    need = (15 + int(v["SETUP_BOUND_MIN"]) + validate + calibrate + 15 + int(v["HOURS"]) * 60 + 10
            + int(v["SCORE_BOUND_MIN"]) + 30 + int(v["PULL_MARGIN_MIN"]))
    assert (validate, calibrate, need) == (235, 1200, 5115)
    assert v["SEATBELT"] == "88h" and 88 * 60 == need + 165
    assert round(round(need / 60, 1) * int(v["GPU_COUNT"]) * 5.29 + 50, 2) == 1852.83
