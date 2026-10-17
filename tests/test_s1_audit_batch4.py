"""S1 audit of 2026-10-17, batch 4 (shells; owner: "keep going lets close all open
problems"). G7: the plan-time disk projection. The master-side rows (G5, G6, G8,
G14 to G18, G30) are pinned in tests/test_experiment_master.py, which owns the
fake pod world.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "scripts" / "6_experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import disk_projection as dp  # noqa: E402


def _plan() -> dict:
    return {"steps": [
        {"kind": "relaunch", "engine": "vllm"},
        {"kind": "dry_window", "row_key": "d1", "argv": ["--duration-s", "75", "--rate", "2"]},        # 150
        {"kind": "dry_window", "row_key": "d2", "argv": []},                                          # 100 assumed
        {"kind": "cell", "row_key": "w1", "windows": 3, "window_arrivals": 1060, "num_queries": 500},  # 3180
        {"kind": "cell", "row_key": "f1", "windows": 3, "num_queries": 852},                           # 2556
        {"kind": "cell", "row_key": "b1", "windows": 3, "window_arrivals": 9999, "blocked_on": "no launcher"},
        {"kind": "cell", "row_key": "x1", "windows": 1},                                               # 0, noted
    ]}


def test_arrivals_follow_the_plan_shape() -> None:
    n, notes = dp.arrivals_of(_plan())
    assert n == 150 + 100 + 3180 + 2556
    assert len(notes) == 2 and "d2" in notes[0] and "x1" in notes[1]


def test_projection_bytes_and_copies() -> None:
    rec = dp.project(_plan(), copies=3)
    assert rec["arrivals"] == 5986 and rec["bytes_per_arrival"] == 37_000
    assert rec["pod_gb"] == pytest.approx(5986 * 37_000 * 3 / 1e9, abs=1e-3)
    assert rec["mac_gb"] == pytest.approx(5986 * 37_000 / 1e9, abs=1e-3)


def test_cli_refuses_a_disk_that_would_fill(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(_plan()), encoding="utf-8")
    # 0.443 GB on the pod (2 copies): fits 1 GB at 0.8 headroom, not 0.5 GB
    assert dp.main(["dp", str(path), "--pod-gb", "1", "--mac-free-gb", "1"]) == 0
    out = capsys.readouterr().out
    assert "disk projection (G7): 5986 arrivals x 37000 B x 2 copies" in out and "[FAIL]" not in out
    assert dp.main(["dp", str(path), "--pod-gb", "0.5", "--mac-free-gb", "1"]) == 1
    assert "the pod disk would fill" in capsys.readouterr().out
    assert dp.main(["dp", str(path), "--pod-gb", "100", "--mac-free-gb", "0.2"]) == 1
    assert "the Mac disk cannot take the pull" in capsys.readouterr().out
    assert dp.main(["dp", str(path), "--pod-gb", "1", "--mac-free-gb", "1", "--copies", "0"]) == 2
    # an empty plan (the fake worlds) projects zero and passes
    path.write_text(json.dumps({"steps": []}), encoding="utf-8")
    assert dp.main(["dp", str(path), "--pod-gb", "1", "--mac-free-gb", "0"]) == 0
