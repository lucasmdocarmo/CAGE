"""S2 block F row F-13 (2026-11-04): the frontier's control rung is
registration content, read from the plan, never a driver constant.

The E1 executor (contrast #12, run_campaign_analysis.compute_frontier_drop)
took the control rung from a module constant (1.25, S1's rung). S2 registers
its control at 1.0 (run_campaign.py S2_CONTROL_R), so on an S2 tree every
constrained cell read "no control cell (budget_r 1.25)" and the look refused
with "no confirmatory point". The rung now comes from the registered plan's
``calibration.budget_fraction``: run_campaign.py writes it from the grid's
floor_budget_r (the rung the floors were measured at and every pressure
point's control cells share, ruling A2) and refuses a calibration artifact
at any other rung. It is the same --plan file the class targets come from.
Absent or malformed refuses naming the key.

The fixture is tests/test_frontier_e1.py's sealed K1 tree, reshaped to S2's
rungs (1.0 control, 0.5 and 0.25 constrained) by swapping that module's
late-count table, so every number the executor reports is checked by that
module's independent recount.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "scripts" / "4_analysis", REPO_ROOT, REPO_ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_predicate_table as bpt  # noqa: E402
import run_campaign_analysis as rca  # noqa: E402
import test_frontier_e1 as e1  # noqa: E402
from src.analysis.stats import families  # noqa: E402

S2_CONTROL = 1.0
#: the E1 fixture's late pattern re-keyed on S2's rungs: the control rung is
#: never late, the half and quarter caches keep that module's counts
S2_LATE_BY_RUNG: dict[float, dict[int, int]] = {
    S2_CONTROL: dict(e1.LATE_BY_RUNG[1.25]),
    0.5: dict(e1.LATE_BY_RUNG[0.5]),
    0.25: dict(e1.LATE_BY_RUNG[0.25]),
}


@pytest.fixture(autouse=True)
def _pressure_dataset_and_no_freeze(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # as tests/test_frontier_e1.py: span-QA rows stand in for the registered pressure dataset
    monkeypatch.setattr(families, "PRESSURE_DATASET", e1.DATASET)
    monkeypatch.setenv(bpt.FREEZE_ENV_VAR, str(tmp_path / "absent_freeze.json"))


def _plan(tmp_path: Path, control: Any, *, with_calibration: bool = True) -> Path:
    """The registered plan as the driver reads it: the class targets, and the
    calibration block run_campaign.py writes (its budget_fraction is the
    control rung; registered_budget_fraction is the charter's floor fraction)."""
    doc: dict[str, Any] = {"class_first_token_targets_s": {str(e1.CLASS_TOKENS): e1.TARGET_S}}
    if with_calibration:
        doc["calibration"] = {"budget_fraction": control, "registered_budget_fraction": 1.5}
    path = tmp_path / "plan_registered.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _s2_cells() -> list[tuple[Any, float]]:
    """The E1 fixture's K1 shape on S2's rungs: vLLM two pooled controls at 1.0
    plus r0.5 and r0.25 at RATE; SGLang one control and one r0.25 at RATE."""
    return [
        (e1._spec("vllm", S2_CONTROL, 0.76), e1.RATE),
        (e1._spec("vllm", S2_CONTROL, 0.761), e1.RATE),
        (e1._spec("vllm", 0.5, 0.7), e1.RATE),
        (e1._spec("vllm", 0.25, 0.8), e1.RATE),
        (e1._spec("sglang", S2_CONTROL, 0.77), e1.RATE),
        (e1._spec("sglang", 0.25, 0.82), e1.RATE),
    ]


def _build_s2_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(e1, "LATE_BY_RUNG", S2_LATE_BY_RUNG)
    return e1._build_run(tmp_path, _s2_cells())


def test_the_control_rung_is_the_registered_plans_budget_fraction_f13(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An S2-shaped tree under a plan registering 1.0: the 1.0 cells are the
    controls, every constrained cell is a confirmatory point, and the section,
    its estimand text and the summary name the registered rung."""
    run_dir = _build_s2_run(tmp_path, monkeypatch)
    assert rca.main([str(run_dir), "--contrasts", "12", "--plan", str(_plan(tmp_path, S2_CONTROL))]) == 0
    analysis_dir, stats = e1._load_stats(run_dir)
    section = stats["frontier"]
    assert section["control_budget_r"] == S2_CONTROL
    legs = {(leg["engine"], leg["leg"]): leg for leg in section["legs"]}
    assert set(legs) == {("vllm", "B1|r0.5"), ("vllm", "B1|r0.25"), ("sglang", "B1|r0.25")}
    control = e1._expected_shares(S2_CONTROL)
    for leg in legs.values():
        expected = e1._expected_shares(leg["budget_r"])
        assert leg["yield"]["mean_control"] == pytest.approx(control["yield"])
        assert leg["yield"]["delta"] == pytest.approx(expected["yield"] - control["yield"])
        assert leg["yield"]["delta"] < 0
    # D2 pools both vLLM control cells at the registered rung
    assert len(legs[("vllm", "B1|r0.25")]["control_row_keys"]) == 2
    assert section["not_confirmatory_points"] == []
    assert "minus the control rung 1 at the matched" in section["estimand_text"]
    summary = (analysis_dir / "summary.md").read_text(encoding="utf-8")
    assert "- control rung 1; rate match within" in summary


def test_the_s1_tree_at_its_registered_rung_is_unchanged_f13(tmp_path: Path) -> None:
    """The pin: the E1 fixture's S1 shape under a plan registering 1.25 reads as
    before, the unmatched cell listed with the rung in its reason."""
    run_dir = e1._build_run(tmp_path, e1._k1_cells())
    assert rca.main([str(run_dir), "--contrasts", "12", "--plan", str(_plan(tmp_path, 1.25))]) == 0
    _, stats = e1._load_stats(run_dir)
    section = stats["frontier"]
    assert section["control_budget_r"] == 1.25
    legs = {(leg["engine"], leg["leg"]) for leg in section["legs"]}
    assert legs == {("vllm", "B1|r0.5"), ("vllm", "B1|r0.25"), ("sglang", "B1|r0.25")}
    (listed,) = section["not_confirmatory_points"]
    assert listed["reason"].startswith("no control cell (budget_r 1.25) at this offered rate")
    assert "minus the control rung 1.25 at the matched" in section["estimand_text"]


def test_the_registered_rung_drives_the_selection_not_the_index_f13(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An S1-shaped tree (1.25, 0.5, 0.25) under a plan registering 1.0: the
    1.25 cells sit above the control, no constrained cell has a control at the
    registered rung, and the look refuses naming that rung, never 1.25."""
    run_dir = e1._build_run(tmp_path, e1._k1_cells())
    assert rca.main([str(run_dir), "--contrasts", "12", "--plan", str(_plan(tmp_path, S2_CONTROL))]) == 1
    err = capsys.readouterr().err
    assert "no confirmatory point" in err and "(budget_r 1)" in err and "1.25" not in err
    assert not (run_dir / "analysis").exists()


def test_a_plan_without_the_control_rung_refuses_naming_the_key_f13(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = e1._build_run(tmp_path, e1._k1_cells())
    plan = _plan(tmp_path, None, with_calibration=False)
    assert rca.main([str(run_dir), "--contrasts", "12", "--plan", str(plan)]) == 1
    err = capsys.readouterr().err
    assert "MISSING ARTIFACT" in err and "calibration.budget_fraction" in err and str(plan) in err
    assert not (run_dir / "analysis").exists()


@pytest.mark.parametrize("bad", ["1.25", 0, -0.5, True, None])
def test_a_malformed_control_rung_refuses_naming_the_value_f13(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], bad: Any
) -> None:
    run_dir = e1._build_run(tmp_path, e1._k1_cells())
    plan = _plan(tmp_path, bad)
    assert rca.main([str(run_dir), "--contrasts", "12", "--plan", str(plan)]) == 1
    err = capsys.readouterr().err
    assert "calibration.budget_fraction" in err and repr(bad) in err
    assert not (run_dir / "analysis").exists()
