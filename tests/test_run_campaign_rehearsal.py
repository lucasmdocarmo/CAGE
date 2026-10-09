"""run_campaign.py plan --rehearsal-n (ADR-0144): the dress rehearsal grid.

The rehearsal of a registered session is DERIVED by one rule (rehearsal_grid),
never hand-picked: every baseline, engine, HF-oracle cell, RULER task and B12
rung of the registered grid; datasets restricted to those with a registered
query manifest; F2 and F3 collapsed to one coordinate each (plus one fine-only
F2 coordinate on the anchor); one window per cell; every row class at n. The
plan header records it; the registered grids are untouched. Found on the S0
day 2026-10-07: session a registers n = 2,000 per primary cell while the
shipped manifests carry 50 ids per trial, so build_plan refuses the registered
plan (test 3 pins that refusal) and no full-chain run was possible before.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_run_campaign import (  # noqa: E402
    _rungs_for,
    _floor_table_doc,
    _tiny_grid,
    _write_calibrations,
    rc,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = {
    "squad_v2": REPO_ROOT / "data" / "manifests" / "squad_v2_50x3_seed42.json",
    "qasper": REPO_ROOT / "data" / "manifests" / "qasper_50x3_seed42.json",
}
DATASETS = frozenset(MANIFESTS)


def _coverage(grid: Any) -> set:
    return {(c.baseline_id, c.spec.engine, c.spec.family) for c in rc.enumerate_cells(grid)}


def test_rehearsal_keeps_every_arm_engine_family_hf_cell_ruler_task_and_rung() -> None:
    a = rc.SESSION_GRIDS["a"]
    g = rc.rehearsal_grid(a, n=50, datasets=DATASETS)
    # unchanged registrations
    assert (g.session, g.group, g.model) == ("a", "A", "qwen3-14b")
    assert g.f1_baselines == a.f1_baselines and g.f1_engines == a.f1_engines
    assert g.f2_baselines == a.f2_baselines and g.f3_baselines == a.f3_baselines
    assert g.f2_ruler_baselines == a.f2_ruler_baselines and g.f2_ruler_tasks == a.f2_ruler_tasks
    assert g.corpus_trunc_budgets == a.corpus_trunc_budgets and g.retr_trunc_kept_docs == a.retr_trunc_kept_docs
    assert g.primary_engine == a.primary_engine and g.primary_baselines == a.primary_baselines
    # the collapses, by the rule
    assert g.f1_datasets == ("squad_v2", "qasper")                      # QA_DATASETS order, manifests only
    assert g.hf_oracle_cells == (("B1", ("squad_v2", "qasper")),            # no B2: ADR-0152
                                 ("B3", ("squad_v2", "qasper")), ("B6", ("squad_v2", "qasper")))
    assert g.f2_budgets == (0.75,) and g.f2_rates == (0.85,)             # lower medians of the §6.1 factorial
    assert g.f2_fine_budgets == (1.25,) and g.f2_fine_rates == (0.85,)   # the first fine-only coordinate
    assert g.f3_budgets == (0.5,) and g.f3_rates == (0.95,)              # lower medians of the §6.8 reduced grid
    assert g.replications == 1
    assert (g.n_primary, g.n_secondary, g.n_identity, g.window_requests) == (50, 50, 50, 50)
    assert dict(g.achievable_n) == {}
    # coverage: every (baseline, engine, family) of session a appears in the rehearsal
    assert _coverage(a) <= _coverage(g)
    cells = rc.enumerate_cells(g)
    assert len(cells) == 110  # 6 hf (ADR-0152; old 8 with B2) + 52 sglang + 52 vllm
    assert sorted({c.ruler_task for c in cells if c.ruler_task}) == sorted(rc.RULER_F2_TASKS)
    rungs = {c.spec.corpus_budget_tokens for c in cells if c.baseline_id == "B12"}
    assert rungs == set(a.corpus_trunc_budgets)
    # the registry is untouched
    assert rc.SESSION_GRIDS["a"] is a and a.replications == rc.REPLICATIONS and a.n_primary == rc.N_PRIMARY


def test_rehearsal_plan_builds_against_the_real_manifests_through_the_cli(tmp_path: Path) -> None:
    floor = tmp_path / "floor_table.json"
    floor.write_text(json.dumps(_floor_table_doc()), encoding="utf-8")
    cal = _write_calibrations(tmp_path / "cal", ("vllm", "sglang"), model="Qwen/Qwen3-14B")
    out = tmp_path / "plan.json"
    argv = ["plan", "--session", "a", "--floor-table", str(floor), "--window-duration-s", "600",
            "--rehearsal-n", "50", "--out", str(out)]
    for ds, path in MANIFESTS.items():
        argv += ["--query-manifest", f"{ds}={path}"]
    for eng, path in cal.items():
        argv += ["--calibration", f"{eng}={path}"]
    # ADR-0154: the rehearsal's rungs are a subset of the registered session's,
    # so the session a rung artifacts cover them.
    for eng, path in _rungs_for(cal, "a").items():
        argv += ["--rung-calibration", f"{eng}={path}"]
    assert rc.main(argv) == 0
    plan = json.loads(out.read_text(encoding="utf-8"))
    reh = plan["rehearsal"]
    assert reh["of"] == "a" and reh["n"] == 50 and reh["windows"] == 1
    assert reh["datasets"] == ["qasper", "squad_v2"] and reh["adr"] == "ADR-0144"
    assert reh["f2_coordinates"] == [[0.75, 0.85]] and reh["f2_fine_coordinates"] == [[1.25, 0.85]]
    assert reh["f3_coordinates"] == [[0.5, 0.95]] and "lower-median" in reh["rule"]
    assert plan["session"] == "a" and plan["replications"] == 1
    assert plan["counts"]["cells"] == 110 and plan["counts"]["windows"] == 110  # ADR-0152 (old 112)
    # Relaunches (ADR-0155, one boundary per engine x prefix x budget x
    # demand class): per engine F2 2 budgets {0.75, 1.25} x 4 classes = 8,
    # F3 (r = 0.5) B4 corpus-fresh OFF 1, prefix ON B2/B3/B7/B10 4 + the two
    # B12 rungs (manifests registered) 2, budget-free F1 ON 2 (+1 lmcache on
    # vLLM) and OFF 1; vLLM adds the B8 lmcache F3 boundary 1
    # => vllm 8 + 1 + 4 + 2 + 1 + 3 + 1 = 20, sglang 8 + 1 + 4 + 2 + 2 + 1 = 18
    # => 38 (old pin 18 before ADR-0155). Dry windows (ADR-0153, one per
    # budgeted serving configuration minus r): per engine prefix OFF {4779,
    # 1127, 702, 348} + corpus-fresh 2336 = 5, prefix ON {4779, 2336, 2336 fp8,
    # 1126, 1205, 480} = 6, plus retr-store lmcache on vLLM => 12 + 11 = 23.
    assert plan["counts"]["relaunches"] == 38 and plan["counts"]["blocked"] == 3
    assert plan["counts"]["dry_windows"] == 23
    assert plan["per_row_n"]["n_primary"] == 50 and plan["per_row_n"]["window_requests"] == 50
    cells = [s for s in plan["steps"] if s["kind"] == "cell"]
    assert {s["num_queries"] for s in cells} == {50} and {s["windows"] for s in cells} == {1}
    # the #4 contrast arms (B3 vs B6, primary engine, F1) are present on both manifest datasets
    primary = {(s["baseline"], s["dataset"]) for s in cells if s["row_class"] == "primary"}
    assert primary == {("B3", "squad_v2"), ("B3", "qasper"), ("B6", "squad_v2"), ("B6", "qasper")}
    # the blocked cells are the registered debt, never a rehearsal artifact
    assert all("sglang" in k and "retr-store" in k for k in plan["blocked_row_keys"])
    # the plan the run side validates
    assert rc.load_plan(out)["rehearsal"]["n"] == 50


def test_the_registered_session_a_refuses_the_fifty_id_manifests(tmp_path: Path) -> None:
    # The S0-day finding (2026-10-07): without the rehearsal the registered
    # per-row N (2,000 primary) cannot be served by the shipped manifests.
    floor = tmp_path / "floor_table.json"
    floor.write_text(json.dumps(_floor_table_doc()), encoding="utf-8")
    cal = _write_calibrations(tmp_path / "cal", ("vllm", "sglang"), model="Qwen/Qwen3-14B")
    with pytest.raises(rc.PlanError, match="shortfall"):
        rc.build_plan("a", rc.load_floor_table(floor), window_duration_s=600.0,
                      query_manifests=MANIFESTS, calibrations=cal)


def test_rehearsal_refuses_bad_n_no_manifests_and_a_missing_pressure_dataset() -> None:
    a = rc.SESSION_GRIDS["a"]
    with pytest.raises(rc.PlanError, match="n=0 must be an integer >= 1"):
        rc.rehearsal_grid(a, n=0, datasets=DATASETS)
    with pytest.raises(rc.PlanError, match="at least one registered --query-manifest"):
        rc.rehearsal_grid(a, n=50, datasets=frozenset())
    with pytest.raises(rc.PlanError, match="F2 dataset 'qasper' has no registered query manifest"):
        rc.rehearsal_grid(a, n=50, datasets=frozenset({"squad_v2"}))
    with pytest.raises(rc.PlanError, match="none of its F1 datasets"):
        rc.rehearsal_grid(a, n=50, datasets=frozenset({"ruler"}))


def test_a_plan_without_the_flag_records_no_rehearsal_and_the_registered_n(tmp_path: Path) -> None:
    grid = _tiny_grid()
    floor = tmp_path / "floor_table.json"
    floor.write_text(json.dumps(_floor_table_doc()), encoding="utf-8")
    cal = _write_calibrations(tmp_path / "cal", ("vllm",), model="Qwen/Qwen3-14B")
    orig = rc.SESSION_GRIDS
    rc.SESSION_GRIDS = {"a": grid}
    try:
        plan = rc.build_plan("a", rc.load_floor_table(floor), window_duration_s=60.0, calibrations=cal)
    finally:
        rc.SESSION_GRIDS = orig
    assert plan["rehearsal"] is None
    assert plan["replications"] == rc.REPLICATIONS and plan["per_row_n"]["n_primary"] == rc.N_PRIMARY
