"""ADR-0172 amendments of 2026-10-17 (owner: "E17 d, E18 and E19 follow
recommendations. committed. proceed").

E17 (option d): the window-class pool ceiling ``SessionGrid.window_pool_n``
lowers the pool a pressure window cycles and nothing else; session a carries
Qasper 500 and RULER 300 there, not in ``achievable_n``; the plan header
records each manifest's ``trials_disjoint``. E18 (option ii): the scoring pass
scores identical scoring inputs once per window and the other rows inherit
that score, recorded as ``quality_source``; every arrival carries a verdict.
E19 is tested beside the stationarity record in
tests/test_b3_analysis_and_stationarity.py. Each test fails on the code before
the change it names.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "scripts" / "4_analysis", Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import rescore_quality as rq  # noqa: E402
from src.analysis.stats.ledger import hash_artifacts, write_ledger  # noqa: E402
from test_b3_window_bounds import _f2_grid, _window_cells  # noqa: E402
from test_rescore_wiring import _args, _fast_evaluator, _write_evidence  # noqa: E402
from test_run_campaign import (  # noqa: E402,F401  (fixtures ride the import)
    _dump,
    _stub_plan,
    _tiny_grid,
    _write_manifest,
    calibrations_a,
    floor_table,
    rc,
    stub,
)


# --------------------------------------------------------------------------- #
# E17 (d): the window-class pool ceiling
# --------------------------------------------------------------------------- #


def _class_n(cls: str, **overrides: Any) -> int:
    kwargs: dict[str, Any] = dict(
        n_primary=2000, n_secondary=800, n_identity=300, window_requests=2000,
        achievable_n={}, window_pool_n={"qasper": 500},
    )
    kwargs.update(overrides)
    return rc.class_n(cls, "qasper", **kwargs)


def test_window_pool_n_lowers_the_window_class_only() -> None:
    assert _class_n("window") == 500
    assert _class_n("primary") == 2000 and _class_n("secondary") == 800 and _class_n("identity") == 300
    # achievable_n (A5) still lowers every class, and the two ceilings compose as a minimum
    assert _class_n("window", achievable_n={"qasper": 300}) == 300
    assert _class_n("primary", achievable_n={"qasper": 300}) == 300
    assert _class_n("window", achievable_n={"qasper": 700}) == 500
    # a ceiling never raises, and an absent mapping is the pre-amendment rule
    assert _class_n("window", window_pool_n={"qasper": 5000}) == 2000
    assert _class_n("window", window_pool_n=None) == 2000
    assert _class_n("window", window_pool_n={"ruler": 300}) == 2000  # another dataset


def test_session_a_registers_the_window_pool_apart_from_achievable_n() -> None:
    a, b = rc.SESSION_GRIDS["a"], rc.SESSION_GRIDS["b"]
    assert dict(a.window_pool_n) == {"qasper": 500, "ruler": 300}
    # ADR-0173 amendment: Qasper's own achievable n for the unpressured classes
    # (852 = the built manifest's trial pool after the G4 reserve, 2026-10-17)
    assert dict(a.achievable_n) == {"qasper": 852}
    assert dict(b.window_pool_n) == {} and dict(b.achievable_n) == {}
    rehearsal = rc.rehearsal_grid(a, n=50, datasets=frozenset(rc._grid_datasets(a)))
    assert dict(rehearsal.window_pool_n) == {} and dict(rehearsal.achievable_n) == {}


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"window_pool_n": {"nq_open": 100}}, "window_pool_n"),  # not a dataset of the grid
        ({"window_pool_n": {"qasper": 0}}, "window_pool_n"),
        ({"window_pool_n": {"qasper": 1.5}}, "window_pool_n"),
        ({"window_pool_n": {"qasper": True}}, "window_pool_n"),
        ({"window_pool_n": {"qasper": 2001}}, "exceeds window_requests"),  # a ceiling only lowers
        ({"window_pool_n": [("qasper", 5)]}, "must be a mapping"),
    ],
)
def test_window_pool_n_registration_refusals(overrides: dict[str, Any], match: str) -> None:
    with pytest.raises(rc.PlanError, match=match):
        _tiny_grid(**overrides)
    _tiny_grid(window_pool_n={"qasper": 2000})  # equal to window_requests: accepted


def test_the_plan_pools_window_cells_by_window_pool_n_and_records_it(floor_table, stub) -> None:
    # The same arithmetic as the achievable_n ceiling test of test_b3_window_bounds:
    # a 20-id pool needs 70 passes at 1.2 x 12 rps over 97 s (refused, naming
    # 94 ids), a 100-id pool needs 14 passes (built, pool 100)
    # G24 (2026-10-17): the window over the ceiling blocks its CELL (the plan stands)
    blocked_plan = _stub_plan(_f2_grid(f2_budgets=(1.5,), f2_rates=(1.2,), window_pool_n={"qasper": 20}), floor_table, stub.cmd)
    (blocked,) = _window_cells(blocked_plan)
    assert blocked["blocked_on"].startswith(rc.PASS_CEILING_BLOCK_PREFIX)
    assert "at least 94 ids per trial for qasper (SessionGrid.window_pool_n, fix list A4)" in blocked["blocked_on"]
    plan = _stub_plan(_f2_grid(f2_budgets=(1.5,), f2_rates=(1.2,), window_pool_n={"qasper": 100}), floor_table, stub.cmd)
    (window,) = _window_cells(plan)
    assert window["num_queries"] == 100 and window["window"]["pool"] == 100 and window["window"]["passes"] == 14
    head = plan["per_row_n"]
    assert head["window_pool_n"] == {"qasper": 100} and head["achievable_n"] == {}
    assert "qasper=100" in head["window_pool_caveat"] and "E17" in head["window_pool_caveat"]
    assert head["achievable_n_caveat"] is None


def test_load_plan_re_derives_the_pool_from_the_header(tmp_path: Path, floor_table, stub) -> None:
    plan = _stub_plan(_f2_grid(f2_budgets=(1.5,), f2_rates=(1.2,), window_pool_n={"qasper": 100}), floor_table, stub.cmd)
    rc.load_plan(_dump(tmp_path, plan, "ok.json"))
    # a header that lost the key would re-derive the pool as 2,000 and refuse the 100
    stale = json.loads(json.dumps(plan))
    del stale["per_row_n"]["window_pool_n"]
    with pytest.raises((rc.PlanError, rc.RunError), match="num_queries is 100 but the registered window n is 2000"):
        rc.load_plan(_dump(tmp_path, stale, "stale.json"))
    # a plan built before the amendment (no key, no cap) loads: the absent key means no cap
    plain = _stub_plan(_f2_grid(), floor_table, stub.cmd)
    assert plain["per_row_n"]["window_pool_n"] == {} and plain["per_row_n"]["window_pool_caveat"] is None
    del plain["per_row_n"]["window_pool_n"]
    del plain["per_row_n"]["window_pool_caveat"]
    rc.load_plan(_dump(tmp_path, plain, "old.json"))


def test_manifest_registration_records_trials_disjoint(tmp_path: Path) -> None:
    grid = rc.SESSION_GRIDS["a"]
    plain = _write_manifest(tmp_path, "qasper", name="plain.json")
    assert rc._register_query_manifests(grid, {"qasper": plain})["qasper"]["trials_disjoint"] is None
    for flag, name in ((False, "overlap.json"), (True, "disjoint.json")):
        path = _write_manifest(tmp_path, "qasper", name=name)
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["stats"] = {"trials_disjoint": flag, "pool_size": 1005}
        path.write_text(json.dumps(doc), encoding="utf-8")
        assert rc._register_query_manifests(grid, {"qasper": path})["qasper"]["trials_disjoint"] is flag


# --------------------------------------------------------------------------- #
# E18 (ii): one scoring call per distinct scoring input, replays inherit
# --------------------------------------------------------------------------- #


def _row(example_id: str, record_index: int | None, generated: str, *, repeat_index: int = 0, **extra: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "example_id": example_id,
        "question": f"question {example_id}",
        "used_contexts": [f"the answer to {example_id} is blue"],
        "generated_answer": generated,
        "reference_answer": "blue",
        "baseline": "B3",
        "repeat_index": repeat_index,
        "record_index": record_index,
        "ok": True,
    }
    base.update(extra)
    return base


#: e0 is replayed twice: once with the same answer (inherits), once with a
#: different answer under pressure (scored on its own); e1 arrives once.
REPLAYED_WINDOW = [
    _row("e0", 0, "blue"),
    _row("e1", 1, "green"),
    _row("e0", 2, "blue"),
    _row("e0", 4, "red"),
]


class _CountingEvaluator:
    """The fast evaluator with a record of every batch_evaluate input."""

    def __init__(self) -> None:
        self.inner = _fast_evaluator()
        self.scored: list[list[str]] = []

    def batch_evaluate(self, questions, contexts, generated, references, **kwargs):
        self.scored.append(list(generated))
        return self.inner.batch_evaluate(questions, contexts, generated, references, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def test_identical_inputs_are_scored_once_and_the_replay_inherits(tmp_path: Path) -> None:
    ev = _write_evidence(tmp_path / "w" / "qa_evidence.jsonl", REPLAYED_WINDOW)
    evaluator = _CountingEvaluator()
    rows, n_dup = rq._score_evidence_file(ev, evaluator)
    assert n_dup == 0 and len(rows) == 4  # every arrival keeps its own row, in file order
    assert evaluator.scored == [["blue", "green", "red"]]  # three distinct inputs, one call
    assert [r["record_index"] for r in rows] == [0, 1, 2, 4]
    assert [r["quality_source"] for r in rows] == ["scored", "scored", "inherited:0", "scored"]
    first, _green, replay, red = rows
    assert replay["exact_match"] == first["exact_match"] == 1.0 and replay["f1_score"] == first["f1_score"]
    assert red["exact_match"] == 0.0  # the different answer is judged on its own
    agg = rq._quality_aggregate(rows)
    assert agg["n_quality_inherited"] == 1 and agg["rows"] == 4
    assert agg["metrics"]["exact_match"]["n"] == 4  # the inherited verdict counts for its arrival


def test_a_different_repeat_index_never_inherits(tmp_path: Path) -> None:
    rows_in = [_row("e0", 0, "blue"), _row("e0", 1, "blue", repeat_index=1)]
    ev = _write_evidence(tmp_path / "w" / "qa_evidence.jsonl", rows_in)
    evaluator = _CountingEvaluator()
    rows, _ = rq._score_evidence_file(ev, evaluator)
    assert evaluator.scored == [["blue", "blue"]]
    assert [r["quality_source"] for r in rows] == ["scored", "scored"]


def test_a_changed_context_is_a_different_input(tmp_path: Path) -> None:
    rows_in = [_row("e0", 0, "blue"), _row("e0", 1, "blue", used_contexts=["another passage"])]
    ev = _write_evidence(tmp_path / "w" / "qa_evidence.jsonl", rows_in)
    evaluator = _CountingEvaluator()
    rows, _ = rq._score_evidence_file(ev, evaluator)
    assert len(evaluator.scored[0]) == 2
    assert [r["quality_source"] for r in rows] == ["scored", "scored"]


def test_a_not_ok_row_is_never_scored_and_carries_no_source(tmp_path: Path) -> None:
    rows_in = [
        _row("e0", 0, "blue"),
        _row("e0", 1, "", ok=False, error="drain timeout after 300 s"),
        _row("e0", 2, "blue"),
    ]
    ev = _write_evidence(tmp_path / "w" / "qa_evidence.jsonl", rows_in)
    evaluator = _CountingEvaluator()
    rows, _ = rq._score_evidence_file(ev, evaluator)
    assert evaluator.scored == [["blue"]]
    assert [r["quality_source"] for r in rows] == ["scored", None, "inherited:0"]
    assert rows[1]["not_ok"] is True and rows[1]["exact_match"] is None
    assert rq._quality_aggregate(rows)["n_quality_inherited"] == 1


def test_a_representative_without_an_index_is_named_none(tmp_path: Path) -> None:
    rows_in = [_row("e0", None, "blue"), _row("e0", 7, "blue")]
    ev = _write_evidence(tmp_path / "w" / "qa_evidence.jsonl", rows_in)
    rows, _ = rq._score_evidence_file(ev, _fast_evaluator())
    assert [r["quality_source"] for r in rows] == ["scored", "inherited:none"]


def test_an_evaluator_that_returns_the_wrong_count_refuses(tmp_path: Path) -> None:
    class _Short(_CountingEvaluator):
        def batch_evaluate(self, questions, contexts, generated, references, **kwargs):
            return super().batch_evaluate(questions, contexts, generated, references, **kwargs)[:-1]

    ev = _write_evidence(tmp_path / "w" / "qa_evidence.jsonl", REPLAYED_WINDOW)
    with pytest.raises(RuntimeError, match="returned 2 metric records for 3 scored inputs"):
        rq._score_evidence_file(ev, _Short())


def test_scoring_tree_records_the_rule_and_the_inherited_count(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "cells").mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"run_id": "r1"}), encoding="utf-8")
    _write_evidence(run / "cells" / "B3" / "window_qasper-01" / "qa_evidence.jsonl", REPLAYED_WINDOW)
    _write_evidence(run / "cells" / "B1" / "window_qasper-01" / "qa_evidence.jsonl", [_row("e9", 0, "blue")])
    sealed = [p for p in sorted(run.rglob("*")) if p.is_file()]
    write_ledger(hash_artifacts(sealed, base_dir=run), run / "ledger.json")
    rc_code = rq.run_scoring_tree(run, "s01", _args(scoring_run_id="s01"))
    assert rc_code == 0
    manifest = json.loads((run / "scoring" / "s01" / "scoring_manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_quality_inherited"] == 1 and manifest["n_rows"] == 5
    assert manifest["quality_replay_rule"] == rq.QUALITY_REPLAY_RULE and "E18" in rq.QUALITY_REPLAY_RULE
    window = run / "scoring" / "s01" / "cells" / "B3" / "window_qasper-01"
    assert json.loads((window / "quality.json").read_text(encoding="utf-8"))["n_quality_inherited"] == 1
    scores = [json.loads(line) for line in (window / "qa_scores.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [s["quality_source"] for s in scores] == ["scored", "scored", "inherited:0", "scored"]
    assert all(s["baseline"] == "B3" for s in scores)  # the unblind join still holds on every row


def test_dataclass_replace_keeps_the_pool_validation() -> None:
    grid = rc.SESSION_GRIDS["a"]
    with pytest.raises(rc.PlanError, match="window_pool_n"):
        dataclasses.replace(grid, window_pool_n={"qasper": -1})
