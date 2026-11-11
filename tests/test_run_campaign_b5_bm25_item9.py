"""Item 9 (owner 2026-11-11, ADR-0202): the B5-bm25 retriever references on cd-act1.

The owner's "ok add it" (2026-11-11): cd-act1 registers TWO no-load F1
references, B5's arm (retr-fresh) served over BM25 instead of the dense
retriever, one per engine, on MuSiQue. The grid names them in
``f1_retriever_cells``; enumeration re-mints the B5 spec with the retriever
swapped, so the row key carries ``bm25`` where the dense cell carries
``dense``; the cell keeps ``baseline_id`` B5 (the sort, the row class and the
plan record key on it) and its runner label is the explicit ``B5-bm25``
(ADR-0101's name); the numbered layer's maps (cellspec.BASELINE_OF_ARM,
STRATEGY_DISPLAY_NAME, campaign_layout._BASELINE_OF_CELL) are untouched; every
other row and every relaunch stays byte-identical. Fixtures:
tests/test_run_campaign.py (the cd-act1 plan).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_run_campaign as trc  # noqa: E402

rc = trc.rc

BM25_KEYS = {
    "vllm": "retr-fresh|bm25|none|single|vllm|qwen3-next-80b|F1",
    "sglang": "retr-fresh|bm25|none|single|sglang|qwen3-next-80b|F1",
}
B6_VLLM_KEY = "retr-fresh|rerank|none|single|vllm|qwen3-next-80b|F1"
BM25_ARGV = ("--retriever", "bm25", "--reranker-model", "none")


@pytest.fixture(scope="module")
def plan(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    return trc._s2_plan("cd-act1", tmp_path_factory.mktemp("cd_act1_bm25"))


def _rows(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    return trc._s2_rows(plan)


def _flag(argv: List[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_the_grid_registers_two_references_and_the_argv_table_names_bm25() -> None:
    cd = rc.get_session_grid("cd-act1")
    assert cd.f1_retriever_cells == (("B5", "vllm", "bm25"), ("B5", "sglang", "bm25"))
    assert rc.get_session_grid("s2a").f1_retriever_cells == ()
    assert rc.RETRIEVER_ARGV["bm25"] == BM25_ARGV
    # rrf stays unregistered (owner 2026-11-11: rrf out); enumerating it refuses
    assert "rrf" not in rc.RETRIEVER_ARGV and "hybrid-rrf" not in rc.RETRIEVER_ARGV


def test_enumeration_adds_exactly_two_f1_cells_keyed_by_the_retriever() -> None:
    cells = rc.enumerate_cells(rc.get_session_grid("cd-act1"))
    bm25 = [c for c in cells if c.spec.retriever == "bm25"]
    assert sorted(c.spec.to_row_key() for c in bm25) == sorted(BM25_KEYS.values())
    for cell in bm25:
        assert (cell.baseline_id, cell.dataset, cell.spec.family, cell.spec.arm) == ("B5", "musique", "F1", "retr-fresh")
        assert cell.blocked_on is None and cell.block is None
    # cd-act1 serves no dense B5 (its f1_baselines are the five core arms plus
    # B8): the bm25 references do not bring one along, and B6 is still there once
    assert not any(c.spec.arm == "retr-fresh" and c.spec.retriever == "dense" for c in cells)
    assert sum(1 for c in cells if c.spec.to_row_key() == B6_VLLM_KEY) == 1


def test_the_cell_argv_carries_the_bm25_flags_the_a5_pins_and_the_explicit_label(plan: Dict[str, Any]) -> None:
    rows = {s["row_key"]: s for s in _rows(plan)}
    for engine, key in BM25_KEYS.items():
        row = rows[key]
        argv = row["argv"]
        assert _flag(argv, "--retriever") == "bm25" and _flag(argv, "--reranker-model") == "none"
        assert "--rerank-pool" not in argv and "--reranker-device" not in argv
        # backlog A5: every retrieval cell pins the embedding model, its revision and
        # the index root (load_plan re-checks them); the runner never loads the
        # encoder under bm25 (run_experiment.py 3613), so the pins ride unused
        for pin in ("--embedding-model", "--embedding-revision", "--ir-index-dir", "--top-k"):
            assert pin in argv, (key, pin)
        assert _flag(argv, "--encoder-device") == "cuda"  # cd-act1 serves the helpers on the GPU (A14)
        assert _flag(argv, "--baseline") == "rag"   # ARM_RUNNER_BASELINE["retr-fresh"], the runner token of the arm, unchanged by the retriever
        # the explicit label (ADR-0101's name); the plan record keeps the numbered code
        assert _flag(argv, "--baseline-label") == "B5-bm25_retr-fresh"
        assert row["baseline"] == "B5" and row["family"] == "F1" and row["dataset"] == "musique"
        assert row["row_class"] == "secondary" and row["blocked_on"] is None
        assert row["cellspec"]["retriever"] == "bm25" and row["cellspec"]["engine"] == engine
        assert row["env"]["CAGE_CELL_ARM"] == "retr-fresh" and row["env"]["CAGE_CELL_RETRIEVER"] == "bm25"
        assert "CAGE_DISTRACTOR_DOCS" in row["env"]  # a retrieval cell draws the distractor corpus
        assert row["serving"]["prefix_mode"] == "OFF"
    # the dense-pair rows keep their plain label
    assert _flag(rows[B6_VLLM_KEY]["argv"], "--baseline-label") == "B6_retr-fresh"


def test_the_references_share_their_engines_prefix_off_relaunch_and_add_none(plan: Dict[str, Any]) -> None:
    steps = plan["steps"]
    for engine, key in BM25_KEYS.items():
        idx = next(i for i, s in enumerate(steps) if s.get("row_key") == key)
        relaunch = next(s for s in reversed(steps[:idx]) if s["kind"] == "relaunch")
        assert relaunch["engine"] == engine and relaunch["prefix_mode"] == "OFF" and relaunch["budget_r"] is None
        assert "engine_variant" not in relaunch


def test_the_other_rows_and_every_relaunch_are_byte_identical_without_the_references(tmp_path: Path) -> None:
    grid = rc.get_session_grid("cd-act1")
    bare = rc.SessionGrid(**{**grid.__dict__, "f1_retriever_cells": ()})
    anchor = trc._S2_ANCHOR_SEQ_TOKENS["cd-act1"]
    (tmp_path / "with").mkdir()
    (tmp_path / "without").mkdir()
    with_refs = trc._s2_plan("cd-act1", tmp_path / "with", anchor=anchor)
    # The session vocabulary is fixed (campaign_layout.SESSIONS), so the bare
    # grid is planned under the same name and the registry is restored after.
    try:
        rc.SESSION_GRIDS["cd-act1"] = bare
        without = trc._s2_plan("cd-act1", tmp_path / "without", anchor=anchor)
    finally:
        rc.SESSION_GRIDS["cd-act1"] = grid
    strip = lambda s: {k: v for k, v in s.items() if k != "index"}  # noqa: E731
    ref_steps = [s for s in with_refs["steps"] if s.get("row_key") in BM25_KEYS.values()]
    assert len(ref_steps) == 2  # one cell step each, no relaunch of their own
    rest = [strip(s) for s in with_refs["steps"] if s not in ref_steps]
    assert rest == [strip(s) for s in without["steps"]]
    assert without["counts"]["cells"] + 2 == with_refs["counts"]["cells"]
    assert without["counts"]["windows"] + 2 == with_refs["counts"]["windows"]
    assert without["counts"]["relaunches"] == with_refs["counts"]["relaunches"]
    assert without["counts"]["dry_windows"] == with_refs["counts"]["dry_windows"]
    assert without["counts"]["blocked"] == with_refs["counts"]["blocked"]
    assert without["identity_gated_row_keys"] == with_refs["identity_gated_row_keys"]


def test_run_projection_prices_the_references_on_the_dense_row_proxy(plan: Dict[str, Any]) -> None:
    # [A] item 9: no bm25 window is measured yet; the engine's dense row stands
    # in, so the master's HOURS comparison counts the cells instead of refusing
    # the plan for an unmeasured step. A later measured row is a deliberate change.
    for engine in ("vllm", "sglang"):
        assert rc.RUN_CELL_SECONDS[(engine, "bm25")] == rc.RUN_CELL_SECONDS[(engine, "dense")]
    unmeasured = plan["run_projection"]["unmeasured_steps"]
    assert not [u for u in unmeasured if "retr-fresh|bm25" in u], unmeasured


def test_load_plan_accepts_the_written_plan(plan: Dict[str, Any], tmp_path: Path) -> None:
    path = tmp_path / "plan_bm25.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    loaded = rc.load_plan(path)
    assert sorted(s["row_key"] for s in loaded["steps"] if s.get("row_key") in BM25_KEYS.values()) == sorted(BM25_KEYS.values())


def test_an_unregistered_retriever_the_baselines_own_or_an_unknown_baseline_refuses() -> None:
    grid = rc.get_session_grid("cd-act1")
    for field, match in (
        ((("B5", "vllm", "rrf"),), "retriever"),
        ((("B6", "vllm", "rerank"),), "own retriever"),
        ((("B13", "vllm", "bm25"),), "baseline"),
    ):
        bad = rc.SessionGrid(**{**grid.__dict__, "f1_retriever_cells": field})
        with pytest.raises(rc.PlanError, match=match):
            rc.enumerate_cells(bad)


def test_the_session_engines_and_the_numbered_layer_are_unchanged() -> None:
    from src.analysis.cellspec import BASELINE_OF_ARM, BASELINES, strategy_name_of_cell

    assert rc._session_engines(rc.get_session_grid("cd-act1")) == frozenset({"vllm", "sglang"})
    assert ("retr-fresh", "bm25") not in BASELINE_OF_ARM and "B5-bm25" not in BASELINES
    assert strategy_name_of_cell("retr-fresh", "bm25") is None  # the figures name it by its axes


def test_families_contrast_6_carries_group_c() -> None:
    from src.analysis.stats.families import CONTRASTS

    c6 = next(c for c in CONTRASTS if c.id == 6)
    assert (c6.baseline_a, c6.baseline_b, c6.family, c6.tier) == ("B5", None, "F1", "secondary")
    assert c6.groups == ("A", "C")
    assert "ADR-0101" in c6.notes
