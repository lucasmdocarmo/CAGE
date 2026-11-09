"""ADR-0195: the SGLang extra-buffer probe cell in the campaign driver.

Owner decision 2 option 3 (2026-11-08): cd-act1 registers ONE no-load F1
reference, B3 on SGLang with the ``extra_buffer`` Mamba scheduler, as a
mechanism probe beside the default B3 cells. The grid names it in
``f1_variant_cells``; enumeration adds it as an F1 cell carrying the
``engine_variant`` coordinate; it is its own relaunch boundary whose env
carries the two launcher knobs; every default row and relaunch stays
byte-identical. Fixtures: tests/test_run_campaign.py (the cd-act1 plan).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_run_campaign as trc  # noqa: E402

rc = trc.rc

XB_KEY = "corpus-reuse|none|none|single|sglang|qwen3-next-80b|F1|variant=extra_buffer"
DEFAULT_KEY = "corpus-reuse|none|none|single|sglang|qwen3-next-80b|F1"
XB_ENV = {"CAGE_SGLANG_MAMBA_SCHEDULER_STRATEGY": "extra_buffer", "CAGE_SGLANG_DISABLE_OVERLAP_SCHEDULE": "1"}


@pytest.fixture(scope="module")
def plan(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    return trc._s2_plan("cd-act1", tmp_path_factory.mktemp("cd_act1_xb"))


def _rows(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    return trc._s2_rows(plan)


def test_the_grid_registers_the_probe_and_the_launch_env_table_names_it() -> None:
    cd = rc.get_session_grid("cd-act1")
    assert cd.f1_variant_cells == (("B3", "sglang", "extra_buffer"),)
    assert rc.get_session_grid("s2a").f1_variant_cells == ()
    assert rc.ENGINE_VARIANT_LAUNCH_ENV == {("sglang", "extra_buffer"): XB_ENV}


def test_enumeration_adds_one_f1_cell_carrying_the_variant() -> None:
    cells = rc.enumerate_cells(rc.get_session_grid("cd-act1"))
    xb = [c for c in cells if c.spec.engine_variant is not None]
    assert len(xb) == 1
    cell = xb[0]
    assert cell.spec.to_row_key() == XB_KEY
    assert (cell.baseline_id, cell.dataset, cell.spec.family) == ("B3", "musique", "F1")
    assert cell.blocked_on is None and cell.block is None
    # The default B3 reference on SGLang is still there, untouched.
    assert sum(1 for c in cells if c.spec.to_row_key() == DEFAULT_KEY) == 1


def test_the_probe_is_its_own_relaunch_boundary_with_the_knobs_in_env(plan: Dict[str, Any]) -> None:
    steps = plan["steps"]
    rows = _rows(plan)
    xb_row = next(s for s in rows if s["row_key"] == XB_KEY)
    assert xb_row["serving"]["engine_variant"] == "extra_buffer"
    assert xb_row["cellspec"]["engine_variant"] == "extra_buffer"
    assert xb_row["row_class"] == "secondary" and xb_row["blocked_on"] is None
    # The relaunch right before the probe row carries the knobs; no other does.
    idx = next(i for i, s in enumerate(steps) if s.get("row_key") == XB_KEY)
    relaunch = next(s for s in reversed(steps[:idx]) if s["kind"] == "relaunch")
    assert relaunch["engine"] == "sglang" and relaunch["engine_variant"] == "extra_buffer"
    assert all(relaunch["env"][k] == v for k, v in XB_ENV.items())
    assert relaunch["prefix_mode"] == "ON" and relaunch["budget_r"] is None
    others = [s for s in steps if s["kind"] == "relaunch" and s is not relaunch]
    # review 2026-11-08 MEDIUM 1: a default relaunch record carries NO key (absence
    # stays absence, like the cell record); only the probe's relaunch names its variant
    assert all("engine_variant" not in s for s in others)
    assert all(not (set(XB_ENV) & set(s["env"])) for s in others)
    # The runner seam carries the variant to derive_cell_spec.
    assert xb_row["env"]["CAGE_CELL_ENGINE_VARIANT"] == "extra_buffer"
    default_row = next(s for s in rows if s["row_key"] == DEFAULT_KEY)
    assert "CAGE_CELL_ENGINE_VARIANT" not in default_row["env"]
    assert "engine_variant" not in default_row["serving"] and "engine_variant" not in default_row["cellspec"]


def test_the_default_rows_and_relaunches_are_byte_identical_without_the_probe(tmp_path: Path) -> None:
    grid = rc.get_session_grid("cd-act1")
    bare = rc.SessionGrid(**{**grid.__dict__, "f1_variant_cells": ()})
    anchor = trc._S2_ANCHOR_SEQ_TOKENS["cd-act1"]
    (tmp_path / "with").mkdir()
    (tmp_path / "without").mkdir()
    with_probe = trc._s2_plan("cd-act1", tmp_path / "with", anchor=anchor)
    # The session vocabulary is fixed (campaign_layout.SESSIONS), so the bare
    # grid is planned under the same name and the registry is restored after.
    try:
        rc.SESSION_GRIDS["cd-act1"] = bare
        without = trc._s2_plan("cd-act1", tmp_path / "without", anchor=anchor)
    finally:
        rc.SESSION_GRIDS["cd-act1"] = grid
    strip = lambda s: {k: v for k, v in s.items() if k != "index"}  # noqa: E731
    probe_steps = [s for s in with_probe["steps"] if s.get("row_key") == XB_KEY or s.get("engine_variant")]
    assert len(probe_steps) == 2  # one relaunch, one cell
    rest = [strip(s) for s in with_probe["steps"] if s not in probe_steps]
    assert rest == [strip(s) for s in without["steps"]]
    assert without["counts"]["cells"] + 1 == with_probe["counts"]["cells"]
    assert without["counts"]["relaunches"] + 1 == with_probe["counts"]["relaunches"]
    # review 2026-11-08 LOW 6: the two plans above share the new code, so the record
    # SHAPE is pinned here against HEAD's: no default relaunch carries the key at all
    assert all("engine_variant" not in s for s in without["steps"] if s["kind"] == "relaunch")


def test_load_plan_ties_a_cells_variant_to_the_relaunch_it_runs_under(plan: Dict[str, Any], tmp_path: Path) -> None:
    """Review 2026-11-08 LOW 4: a hand-edited plan that places the probe cell under a
    default SGLang relaunch, or a default cell under the probe's, must refuse."""
    import json
    tampered = json.loads(json.dumps(plan))
    probe = next(s for s in tampered["steps"] if s.get("row_key") == XB_KEY)
    del probe["serving"]["engine_variant"]
    path = tmp_path / "probe_as_default.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(rc.RunError, match="engine_variant"):
        rc.load_plan(path)
    tampered = json.loads(json.dumps(plan))
    default = next(s for s in tampered["steps"] if s.get("row_key") == DEFAULT_KEY)
    default["serving"]["engine_variant"] = "extra_buffer"
    path = tmp_path / "default_as_probe.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(rc.RunError, match="engine_variant"):
        rc.load_plan(path)


def test_the_registration_report_flags_the_probe() -> None:
    report = rc.preflight_registration_check("cd-act1", floor_avg_seq_tokens=2694, gpu_id=None)
    assert report["sglang_xb_probe"] is True
    assert rc.preflight_registration_check("s2a", floor_avg_seq_tokens=trc._ANCHOR_SEQ_TOKENS, gpu_id=None)["sglang_xb_probe"] is False


def test_a_variant_on_an_engine_without_a_launch_env_refuses_at_enumeration() -> None:
    grid = rc.get_session_grid("cd-act1")
    bad = rc.SessionGrid(**{**grid.__dict__, "f1_variant_cells": (("B3", "vllm", "extra_buffer"),)})
    with pytest.raises(rc.PlanError, match="variant"):
        rc.enumerate_cells(bad)
