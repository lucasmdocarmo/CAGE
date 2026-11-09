"""DEC-8 (2026-11-08): the own-accounting leg of the analysis charges a
hybrid cell at the degree its pool was planned at: the vLLM geometry's tp
when the plan carries one (a single instance at serving_tp, a pd role at its
role degree), else the plan's tp, else 1."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.own_accounting import OwnAccountingError  # noqa: E402

_SPEC = importlib.util.spec_from_file_location("rca_plan_tp", REPO_ROOT / "scripts" / "4_analysis" / "run_campaign_analysis.py")
rca = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = rca
_SPEC.loader.exec_module(rca)


def test_the_plan_degree_is_the_geometry_tp_then_the_plan_tp_then_one() -> None:
    assert rca._plan_tp({"tp": 1, "geometry": {"tp": 2, "block_tokens": 544}}) == 2
    assert rca._plan_tp({"tp": 4, "geometry": None}) == 4
    assert rca._plan_tp({"tp": 4}) == 4
    assert rca._plan_tp({}) == 1
    assert rca._plan_tp({"tp": None}) == 1


@pytest.mark.parametrize("plan", [{"tp": 0}, {"tp": "4"}, {"tp": True}, {"geometry": {"tp": 0}}, {"tp": 2.0}])
def test_a_bad_plan_degree_is_refused(plan) -> None:
    with pytest.raises(OwnAccountingError, match="tp"):
        rca._plan_tp(plan)
