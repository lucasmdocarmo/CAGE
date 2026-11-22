"""AN-2 (ADR-0211, owner 2026-11-20): what the S2A look requests, and that it resolves.

The S2A profile lists the seven no-load contrasts the session serves and the two
metrics the look tests (the registered co-primary pair of contrast #4). Every listed id must resolve on the cells the session plans:
a listed id that refuses fails stage 14 of the landing. The master passes the
metric list only when a profile sets it.
"""
from __future__ import annotations

import importlib.util
import re
import sys
import time
from pathlib import Path
from typing import Dict

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT / "scripts" / "4_analysis"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import organize_results as org  # noqa: E402
import run_campaign_analysis as rca  # noqa: E402
import test_experiment_master as tem  # noqa: E402
from src.analysis.stats import families  # noqa: E402

world = tem.world  # the master harness's fixture
PROFILES = REPO_ROOT / "scripts" / "6_experiments" / "profiles"


def _load_rc():
    found = sys.modules.get("run_campaign")
    if found is not None:
        return found
    spec = importlib.util.spec_from_file_location("run_campaign", REPO_ROOT / "scripts" / "3_run" / "run_campaign.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _value(profile: str, key: str) -> str | None:
    found = re.findall(rf'^{key}=(?:"([^"]*)"|(\S*))$', (PROFILES / profile).read_text(encoding="utf-8"), flags=re.M)
    return None if not found else (found[-1][0] or found[-1][1])


def _s2a_index() -> pd.DataFrame:
    rc = _load_rc()
    rows = []
    for cell in rc.enumerate_cells(rc.get_session_grid("s2a")):
        s = cell.spec
        rows.append({"row_key": s.to_row_key(), "family": s.family, "arm": s.arm, "retriever": s.retriever,
                     "policy": s.policy, "topology": s.topology, "engine": s.engine, "model": s.model,
                     "baseline": org.BASELINE_OF_CELL.get((s.arm, s.retriever), ""), "dataset": cell.dataset,
                     "budget_r": s.budget_r, "rate_frac": s.rate_frac, "engine_variant": ""})
    return pd.DataFrame(rows)


def test_the_s2a_profile_lists_the_seven_contrasts_and_the_registered_metric_pair() -> None:
    assert _value("S2A.env", "CONTRASTS") == "1 2 3 4 5 7 8"
    assert _value("S2A.env", "ANALYSIS_METRICS") == "ttft_ms predicate"
    assert _value("_common.env", "CONTRASTS") == "4" and _value("_common.env", "ANALYSIS_METRICS") == ""
    metrics = _value("S2A.env", "ANALYSIS_METRICS").split()
    # exactly the registered co-primary pair of #4; the faithfulness score is not a column of the joined
    # rows on a campaign tree (tests/test_matched_load_an2.py pins the refusal), so it is not listed
    assert tuple(metrics) == families.DEFAULT_METRICS
    assert all(isinstance(rca._metric_direction(m), bool) for m in metrics)


def test_every_listed_contrast_resolves_on_the_cells_s2a_plans() -> None:
    index = _s2a_index()
    for cid in (int(x) for x in _value("S2A.env", "CONTRASTS").split()):
        contrast = rca.CONTRAST_BY_ID[cid]
        assert rca.classify_contrast(contrast) is None and contrast.unit == "per_query", cid
        pairs = rca.select_contrast_pairs(index, contrast)   # raises on a refusal
        assert sorted(rca.fp.parse_row_key(p.cell_row_key)["engine"] for p in pairs) == ["sglang", "vllm"], cid
        assert all(p.datasets == ("qasper",) for p in pairs)
    # the one served-arm contrast left out refuses the whole look: B8 is not on s2a
    with pytest.raises(rca.AnalysisError, match="needs F1 cells for both B8 and B6"):
        rca.select_contrast_pairs(index, rca.CONTRAST_BY_ID[9])


def test_the_registered_pairs_that_share_a_class_on_s2a() -> None:
    # the four pairs the arm_pairs block forms on S2A: both arms under pressure, one token class
    rc = _load_rc()
    grid = rc.get_session_grid("s2a")
    cls = {org.BASELINE_OF_CELL.get((c.spec.arm, c.spec.retriever), ""): rc.demand_seq_tokens(grid, c.spec)
           for c in rc.enumerate_cells(grid) if c.spec.family != "F1"}
    shared = sorted((cid, a, b) for cid, a, b in rca.registered_arm_pairs() if a in cls and b in cls and cls[a] == cls[b])
    assert shared == [(1, "B1", "B2"), (2, "B4", "B3"), (5, "B5", "B6"), (8, "B10", "B3")]


def _analysis_line(proc) -> str:
    assert proc.returncode == 0, proc.stdout + proc.stderr
    (line,) = [ln for ln in proc.stdout.splitlines() if "run_campaign_analysis.py" in ln]
    return line


def test_the_master_passes_the_metric_list_only_when_the_profile_sets_it(world: Dict[str, Path], tmp_path: Path) -> None:
    assert "--metrics" not in _analysis_line(tem._master(world, "--plan", "--only", "analyze"))
    prof = tmp_path / "S1_metrics.env"
    prof.write_text(world["profile"].read_text(encoding="utf-8") + '\nCONTRASTS="1 4"\nANALYSIS_METRICS="ttft_ms predicate faithfulness"\n',
                    encoding="utf-8")
    world["profile"] = prof
    line = _analysis_line(tem._master(world, "--plan", "--only", "analyze"))
    assert "--contrasts 1 4 --metrics ttft_ms predicate faithfulness --plan" in line
    # review 2026-11-20 LOW 8: a blank value is no list (a bare --metrics would be a usage error)
    blank = tmp_path / "S1_blank.env"
    blank.write_text(prof.read_text(encoding="utf-8") + 'ANALYSIS_METRICS=" "\n', encoding="utf-8")
    world["profile"] = blank
    assert "--metrics" not in _analysis_line(tem._master(world, "--plan", "--only", "analyze"))


#: a driver that refuses any look requesting a no-load contrast (the reference cells were cut) and
#: writes a stamp for the pressure-only look, printing its argv
FAKE_DRIVER_WITHOUT_REFERENCES = r'''
import json, sys
from pathlib import Path
args = sys.argv[1:]
start = args.index("--contrasts") + 1
ids = []
for a in args[start:]:
    if a.startswith("--"):
        break
    ids.append(a)
import os
if ids != ["13"] and os.environ.get("CAGE_TEST_FULL_LIST") != "ok":
    print("ERROR: contrast #1 (What reuse buys on classic QA) needs F1 cells for both B1 and B2; this run's F1 baselines: ['B3', 'B6']", file=sys.stderr); sys.exit(1)
if ids == ["13"] and os.environ.get("CAGE_TEST_SECOND_LOOK") == "crash":
    print("Traceback (most recent call last):\n  boom", file=sys.stderr); sys.exit(1)
stamp = Path(args[0]) / "analysis" / ("20261120-120000" if ids == ["13"] else "20261120-130000")
stamp.mkdir(parents=True, exist_ok=True)
(stamp / "stats.json").write_text(json.dumps({"mode": "DESIGN-INPUT-ONLY", "requested_contrast_ids": ids,
                                              "matched_load": {"n_legs": 6}}), encoding="utf-8")
(stamp / "summary.md").write_text("# summary\n", encoding="utf-8")
print("analysis done", args); sys.exit(0)
'''


def _partial_landing(world: Dict[str, Path], tmp_path: Path) -> Path:
    """A budget-stopped landing whose profile requests a metric list, with the driver above."""
    prof = tmp_path / "S1_an2.env"
    prof.write_text(world["profile"].read_text(encoding="utf-8") + '\nANALYSIS_METRICS="ttft_ms predicate"\n', encoding="utf-8")
    world["profile"] = prof
    tem._w(world["scripts"] / "4_analysis" / "run_campaign_analysis.py", FAKE_DRIVER_WITHOUT_REFERENCES, exe=False)
    proc = tem._master(world, "--yes", "provision", "--yes", "teardown", "--to", "teardown", CAGE_TEST_RUN_BUDGET_STOP="1")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    return world["exp_root"] / "S1" / tem.DATE


def _step_line(land: Path, name_end: str, prefix: str) -> str:
    """The line starting with ``prefix`` in the newest analyze step log whose name ends with ``name_end``."""
    logs = sorted((land / "extras" / "steps").glob(f"analyze_*{name_end}.log"))
    return next(ln for ln in logs[-1].read_text(encoding="utf-8").splitlines() if ln.startswith(prefix))


def test_a_partial_tree_without_a_no_load_cell_still_gets_the_pressure_sections(world: Dict[str, Path], tmp_path: Path) -> None:
    # AN-2l (owner 2026-11-20, "Second look, pressure only"): a budget stop cut the references of one
    # arm; the driver refuses the requested list, and the master takes one more look that needs none
    land = _partial_landing(world, tmp_path)
    proc = tem._master(world, "--only", "analyze")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    assert "needs F1 cells for both B1 and B2" in proc.stdout
    assert "the PARTIAL tree holds no cells for a registered contrast" in proc.stdout
    assert "a second look with --contrasts 13" in proc.stdout
    st = tem._state(world)
    assert st["stages"]["analyze"]["status"] == "passed"
    # review 2026-11-20 M1: the fallback has its own key; the requested list still has no stamp
    assert st["analysis"] == {"pressure_stamp": "20261120-120000"}
    notes = st["stages"]["analyze"]["notes"]
    assert any("the requested list got no look" in n for n in notes)
    assert any("pressure sections only" in n and "6 legs" in n for n in notes)
    assert (land / "plots" / "20261120-120000" / "stats.json").is_file()
    # the refused look carried the profile's metric list; the second look names no metric and no no-load contrast
    assert "--metrics ttft_ms predicate" in _step_line(land, "run_campaign_analysis__design_input_", "# cmd:")
    argv = _step_line(land, "pressure_sections_only_", "analysis done")
    assert f"'--contrasts', '13', '--plan', '{land / 'extras' / 'plan.json'}'" in argv and "--metrics" not in argv


def test_a_redo_after_the_fallback_runs_the_requested_list_again(world: Dict[str, Path], tmp_path: Path) -> None:
    # review 2026-11-20 M1: the pressure-only stamp must not stand in for the look of the requested list
    land = _partial_landing(world, tmp_path)
    assert tem._master(world, "--only", "analyze").returncode == 0
    time.sleep(1.1)   # the master stamps its per-attempt folders to the second (see test_matched_load_an1_master.py)
    proc = tem._master(world, "--only", "analyze", "--redo", CAGE_TEST_FULL_LIST="ok")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    assert "reusing analysis stamp" not in proc.stdout and "a second look" not in proc.stdout
    assert tem._state(world)["analysis"]["stamp"] == "20261120-130000"


def test_a_crash_of_the_second_look_fails_the_stage(world: Dict[str, Path], tmp_path: Path) -> None:
    # review 2026-11-20 M2: only the driver's own refusal is tolerated; a traceback is a failure
    _partial_landing(world, tmp_path)
    proc = tem._master(world, "--only", "analyze", CAGE_TEST_SECOND_LOOK="crash")
    assert proc.returncode == 1
    assert "pressure sections only): rc=1 without the driver's refusal" in proc.stdout
    assert tem._state(world)["stages"]["analyze"]["status"] == "failed"


def test_the_plan_print_names_the_second_look(world: Dict[str, Path]) -> None:
    proc = tem._master(world, "--plan", "--only", "analyze")
    assert proc.returncode == 0 and "pressure-only second look" in proc.stdout
