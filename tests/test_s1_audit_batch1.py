"""S1 audit of 2026-10-17, batch 1 (fix list section G; owner: "lets round all
fixes needed and open and start working").

G1: the runner's pool-covers-arrivals guard (Batch 1 V3) honors an authorized
replay (ADR-0172 item 1): a window may schedule W arrivals over a smaller pool;
a measured example that failed to prepare still refuses. Before this change
every replayed pressure window refused before its first send, while the dry
windows (duration mode) passed. G2: a drain-cut or dispatch-failed arrival's
synthesized evidence stub is scored as not ok (never refused as malformed) and
carries the ADR-0114 labels the Qasper predicate checks. G3: the confirmatory
family map compiles over the run's registered dataset roster, never the
catalog. G9: the realized-n floor is per dataset (Qasper's own achievable n).
G11: an error row without engine usage is a counted exclusion of the own
occupancy, or counts with its prepared prompt length when the runner stamped
it. Each test fails on the code before the change it names.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "scripts" / "3_run", REPO_ROOT / "scripts" / "4_analysis",
           Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import rescore_quality as rq  # noqa: E402
import run_campaign as rc  # noqa: E402
import run_campaign_analysis as rca  # noqa: E402
from src.analysis import predicate as pred  # noqa: E402
from src.analysis.own_accounting import (  # noqa: E402
    OwnAccountingError,
    compute_own_occupancy,
    compute_own_reuse,
)
from src.orchestration import load_generator as lg  # noqa: E402
from test_adr0172_amendments import _row as _scored_row  # noqa: E402
from test_own_accounting import BUDGET_42_TOK, _canonical, _row as _occ_row  # noqa: E402
from test_rescore_wiring import _fast_evaluator, _write_evidence  # noqa: E402
from test_run_experiment_campaign_layout import (  # noqa: E402
    ANSWER_TYPE_CYCLE,
    N_QUERIES,
    RUN_ID,
    _AsyncCampaignEngine,
    _TypedLoader,
    _read_jsonl,
    _run_cell,
    _window_dirs,
    runner,
)


def _async_engine(model: str) -> _AsyncCampaignEngine:
    return _AsyncCampaignEngine(model, "no_cache", 200.0)


# --------------------------------------------------------------------------- #
# G1: the guard honors an authorized replay
# --------------------------------------------------------------------------- #


def test_the_pool_guard_honors_an_authorized_replay() -> None:
    # authorized, every example prepared: W above the pool is the design, no refusal
    runner.check_prepared_covers_arrivals(
        n_prepared=5, n_arrivals=12, dropped_ids=[], replay_authorized=True
    )
    # authorized, one example failed to prepare: the pool is not the whole trial
    with pytest.raises(lg.LoadGeneratorError, match="whole trial.*ADR-0172"):
        runner.check_prepared_covers_arrivals(
            n_prepared=4, n_arrivals=12, dropped_ids=["q3"], replay_authorized=True
        )
    # the pre-amendment rule stands without the authorization
    with pytest.raises(lg.LoadGeneratorError, match="only 5 of the 12 scheduled arrivals.*Batch 1 V3"):
        runner.check_prepared_covers_arrivals(n_prepared=5, n_arrivals=12, dropped_ids=[])
    runner.check_prepared_covers_arrivals(n_prepared=12, n_arrivals=12, dropped_ids=[])
    # duration mode is untouched either way
    runner.check_prepared_covers_arrivals(n_prepared=3, n_arrivals=None, dropped_ids=["a"], replay_authorized=True)
    runner.check_prepared_covers_arrivals(n_prepared=3, n_arrivals=None, dropped_ids=["a"])


def test_a_replayed_window_runs_through_the_real_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The S1 shape: a pool of N_QUERIES ids, W = 20 arrivals authorized up to 3
    # passes. Before G1 the runner refused this window before its first send.
    root = tmp_path / "results" / "camp1" / "a" / RUN_ID
    _run_cell(
        monkeypatch, root, "no_cache", "squad_v2", ttft_base=200.0, num_trials=1,
        engine_factory=_async_engine,
        extra_argv=(
            "--workload-mode", "open_loop", "--rate", "50", "--arrival-count", "20",
            "--window-replay-max", "3", "--drain-timeout-s", "300",
        ),
    )
    (window,) = _window_dirs(root, "no_cache", "squad_v2")
    rows = _read_jsonl(window / "requests.jsonl")
    assert len(rows) == 20
    assert sorted(r["record_index"] for r in rows) == list(range(20))
    assert len({r["example_id"] for r in rows}) == N_QUERIES
    replay = json.loads((window / "metrics.json").read_text(encoding="utf-8"))["workload"]["open_loop"]["replay"]
    assert replay["first_pass_record_index_max"] == N_QUERIES - 1
    assert replay["authorized_max"] == 3


def test_an_unauthorized_short_pool_still_refuses_before_the_first_send(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "results" / "camp1" / "a" / RUN_ID
    with pytest.raises(SystemExit) as exc:
        _run_cell(
            monkeypatch, root, "no_cache", "squad_v2", ttft_base=200.0, num_trials=1,
            engine_factory=_async_engine,
            extra_argv=("--workload-mode", "open_loop", "--rate", "50", "--arrival-count", "20"),
        )
    assert exc.value.code == 1
    assert "Batch 1 V3" in capsys.readouterr().out
    assert not _window_dirs(root, "no_cache", "squad_v2")


# --------------------------------------------------------------------------- #
# G2: the drain-cut stub through scoring and the predicate's label check
# --------------------------------------------------------------------------- #


def _stub(record_index: int, **extra: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "example_id": "e0", "repeat_index": 0, "record_index": record_index, "ok": False,
        "error": "drain timeout after 300 s", "empty_generation": False, "arrival_s": 12.0,
        "synthesized": "dispatch-stub (no generation to evidence)",
    }
    base.update(extra)
    return base


def test_a_drain_cut_stub_is_scored_as_not_ok_never_refused(tmp_path: Path) -> None:
    # the stub carries no generated_answer, reference_answer or used_contexts:
    # before G2 the R1 key check raised EvidenceSchemaError on it
    ev = _write_evidence(tmp_path / "w" / "qa_evidence.jsonl", [_scored_row("e0", 0, "blue"), _stub(7)])
    rows, n_dup = rq._score_evidence_file(ev, _fast_evaluator())
    assert n_dup == 0 and [r["not_ok"] for r in rows] == [False, True]
    assert rows[1]["quality_source"] is None and rows[1]["exact_match"] is None
    assert rows[1]["error"] == "drain timeout after 300 s"
    # a COMPLETED row that lacks a scorer key still refuses (R1 stands)
    bad = _scored_row("e1", 1, "blue")
    del bad["used_contexts"]
    ev2 = _write_evidence(tmp_path / "w2" / "qa_evidence.jsonl", [bad])
    with pytest.raises(rq.EvidenceSchemaError, match="used_contexts"):
        rq._score_evidence_file(ev2, _fast_evaluator())


def test_the_qasper_label_check_holds_on_a_labeled_stub_and_refuses_an_unlabeled_one() -> None:
    # ADR-0114: every Qasper row carries its label, ok or not; the stub now
    # carries it (campaign_session copies it from the request row), so the
    # predicate builder accepts the window; an unlabeled stub still refuses
    pred._validate_qasper_labels([_stub(7, answer_type="extractive", is_impossible=False)], window="w")
    with pytest.raises(pred.PredicateError, match="missing or unknown answer_type"):
        pred._validate_qasper_labels([_stub(7)], window="w")


class _OneFailingAsyncEngine(_AsyncCampaignEngine):
    """The second request's send raises: a dispatch-error arrival, recorded as a stub."""

    async def async_stream_generate(self, request: Any, *, on_first_token: Any = None):
        if str(request.request_id).endswith("q002"):
            raise RuntimeError("engine exploded on send")
        return await super().async_stream_generate(request, on_first_token=on_first_token)


def test_the_runner_stub_row_and_the_evidence_stub_carry_the_labels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "results" / "camp1" / "a" / RUN_ID
    _run_cell(
        monkeypatch, root, "no_cache", "squad_v2", ttft_base=200.0, num_trials=1,
        loader_factory=_TypedLoader,
        engine_factory=lambda model: _OneFailingAsyncEngine(model, "no_cache", 200.0),
        extra_argv=("--workload-mode", "open_loop", "--rate", "50", "--arrival-count", str(N_QUERIES)),
    )
    (window,) = _window_dirs(root, "no_cache", "squad_v2")
    requests = _read_jsonl(window / "requests.jsonl")
    evidence = _read_jsonl(window / "qa_evidence.jsonl")
    assert len(requests) == N_QUERIES and len(evidence) == N_QUERIES
    (stub,) = [r for r in requests if r.get("ok") is False]
    assert "engine exploded" in stub["error"] and stub["example_id"].endswith("q002")
    # G2: the stub row carries the LOADER's labels by value (review 2026-10-17 HIGH 1:
    # the first cut read the prepare_example meta, which has none, so every stub
    # carried None and the predicate refused the window as before the fix)
    want = ANSWER_TYPE_CYCLE[2]
    assert stub["answer_type"] == want and stub["is_impossible"] is (want == "unanswerable")
    # G11: the prepared length key rides the stub (None here: no request cap registered)
    assert "prompt_tokens_prepared" in stub and stub["prompt_tokens_prepared"] is None
    (ev_stub,) = [r for r in evidence if r.get("ok") is False]
    assert ev_stub["synthesized"].startswith("dispatch-stub")
    assert ev_stub["answer_type"] == want and ev_stub["is_impossible"] is (want == "unanswerable")
    # the labeled stub passes the unchanged ADR-0114 contract check
    pred._validate_qasper_labels([ev_stub], window="w")


# --------------------------------------------------------------------------- #
# G3: the confirmatory family map over the registered roster
# --------------------------------------------------------------------------- #


def test_confirmatory_family_map_compiles_over_the_registered_roster() -> None:
    model = next(iter(rca.GROUP_OF_MODEL))
    index = pd.DataFrame([{"model": model, "dataset": d} for d in ("squad_v2", "qasper")])
    ctx = rca.build_family_context(
        index, rca.DEFAULT_METRICS, 0.05, confirmatory=True,
        registered_datasets=["squad_v2", "musique", "qasper"],
    )
    assert ctx is not None and ctx.datasets == ("musique", "qasper", "squad_v2")  # no hotpotqa leg
    with pytest.raises(rca.AnalysisError, match="registered dataset roster"):
        rca.build_family_context(index, rca.DEFAULT_METRICS, 0.05, confirmatory=True)
    with pytest.raises(rca.AnalysisError, match="not in the family map catalog"):
        rca.build_family_context(
            index, rca.DEFAULT_METRICS, 0.05, confirmatory=True, registered_datasets=["squad_v2", "nq_open"],
        )
    design = rca.build_family_context(index, rca.DEFAULT_METRICS, 0.05)
    assert design is not None and design.datasets == ("qasper", "squad_v2")  # design input: what ran


# --------------------------------------------------------------------------- #
# G9: the realized-n floor per dataset
# --------------------------------------------------------------------------- #


def _primary(dataset: str, realized: int) -> list[dict[str, Any]]:
    return [{
        "tier": "primary", "unit": "per_query", "contrast_id": 4, "metric": "predicate",
        "per_dataset": [{"dataset": dataset, "realized_n": realized}],
    }]


def test_the_realized_n_floor_is_qaspers_own_achievable_n() -> None:
    # 852 = the built manifest's trial pool after the G4 reserve (2026-10-17);
    # the analysis floor and the plan's registration must agree
    assert rca.ADR0086_OWN_ACHIEVABLE_N == {"qasper": 852} == dict(rc.SESSION_GRIDS["a"].achievable_n)
    record = rca.check_realized_n(_primary("qasper", 852), None)
    assert record["own_achievable_n"] == {"qasper": 852} and record["floor"] == 2000
    with pytest.raises(rca.AnalysisError, match=r"qasper: realized n=851 < own achievable n=852"):
        rca.check_realized_n(_primary("qasper", 851), None)
    with pytest.raises(rca.AnalysisError, match=r"squad_v2: realized n=1005 < floor=2000"):
        rca.check_realized_n(_primary("squad_v2", 1005), None)
    rca.check_realized_n(_primary("squad_v2", 2000), None)


# --------------------------------------------------------------------------- #
# G11: error rows in the own-accounting occupancy
# --------------------------------------------------------------------------- #


def _occupancy(rows: list[dict[str, Any]]):
    return compute_own_occupancy(rows, model="qwen3-14b", budget_bytes=BUDGET_42_TOK, window_start_s=0.0, window_end_s=4.0)


def test_an_error_row_without_usage_is_counted_not_refused() -> None:
    cut = {**_occ_row("cut", send=1.0, ft=None, end=4.0, prompt=None, num=0), "ok": False, "error": "drain timeout after 300 s"}
    occ = _occupancy(_canonical() + [cut])
    assert occ.n_error_excluded == 1 and occ.n_in_flight == 2
    assert occ.rho_own_time_avg == pytest.approx(0.5)  # the canonical 84 token-seconds, the cut excluded
    # the prepared length stands in: 10 tokens held over [1, 4) add 30 token-seconds
    occ2 = _occupancy(_canonical() + [{**cut, "prompt_tokens_prepared": 10}])
    assert occ2.n_error_excluded == 0 and occ2.n_in_flight == 3
    assert occ2.rho_own_time_avg == pytest.approx((84.0 + 30.0) / 4.0 / 42.0)
    # a COMPLETED row without prompt tokens still refuses, naming the field
    with pytest.raises(OwnAccountingError, match="prompt_tokens"):
        _occupancy(_canonical() + [_occ_row("x", send=1.0, ft=None, end=2.0, prompt=None, num=0)])


def test_a_runner_shaped_stub_counts_with_its_prepared_length(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review 2026-10-17 (batch 1, HIGH 2): the runner's stub row has NO num_tokens
    key (RequestRecord.to_row writes none) and a cut that never left the in-flight
    cap has no actual_send_ts; the first cut required both and refused the window
    it meant to save."""
    stub = {
        "example_id": "cut", "record_index": 9, "ok": False,
        "error": "drain timeout after 300 s", "dropped_by_cap": False,
        "actual_send_ts": 1.0, "first_token_ts": None, "completion_ts": 4.0,
        "prompt_tokens_prepared": 10,
    }
    occ = _occupancy(_canonical() + [stub])
    assert occ.n_error_excluded == 0 and occ.n_in_flight == 3
    assert occ.rho_own_time_avg == pytest.approx((84.0 + 30.0) / 4.0 / 42.0)
    # never sent (cut while waiting on the cap): no interval, counted exclusion
    occ2 = _occupancy(_canonical() + [{**stub, "actual_send_ts": None}])
    assert occ2.n_error_excluded == 1 and occ2.n_in_flight == 2
    assert occ2.rho_own_time_avg == pytest.approx(0.5)


def test_the_reuse_leg_excludes_the_same_error_rows_counted() -> None:
    """Review 2026-10-17 (batch 1, MEDIUM 3): the reuse leg mirrors the occupancy
    leg's counted exclusions; before, a dropped or error row refused it."""
    served = [
        _occ_row("a", send=0.0, ft=None, end=1.0, prompt=100, num=0, group=0, cached=30),
        _occ_row("c", send=0.0, ft=None, end=1.0, prompt=80, num=0, group=1, cached=0),
    ]
    stub = {"example_id": "cut", "record_index": 9, "ok": False, "error": "drain timeout after 300 s",
            "dropped_by_cap": False, "prompt_tokens_prepared": 10}
    dropped = {**_occ_row("d", send=None, ft=None, end=None, prompt=None, num=None, dropped=True), "ok": False}
    reuse = compute_own_reuse(served + [stub, dropped], {0: 50, 1: 0})
    assert reuse.n_requests == 2 and reuse.n_error_excluded == 1 and reuse.n_dropped_excluded == 1
    assert reuse.rho_reuse_own == pytest.approx(50 / 180)
    # a completed row without prompt tokens still refuses
    with pytest.raises(OwnAccountingError, match="prompt_tokens"):
        compute_own_reuse(served + [_occ_row("x", send=0.0, ft=None, end=1.0, prompt=None, num=0, group=0)], {0: 50, 1: 0})
    # nothing served: fail closed, never a reuse of zero over zero
    with pytest.raises(OwnAccountingError, match="every request row was excluded"):
        compute_own_reuse([stub, dropped], {0: 50, 1: 0})


def test_a_malformed_manifest_roster_refuses_by_name(tmp_path: Path) -> None:
    """Review 2026-10-17 (batch 1, LOW 5): a present but malformed 'datasets' value
    read as absent would blame the wrong defect in confirmatory mode."""
    path = tmp_path / "manifest.json"
    assert rca._registered_roster(tmp_path / "missing.json") is None
    path.write_text(json.dumps({"model": "m"}), encoding="utf-8")
    assert rca._registered_roster(path) is None
    path.write_text(json.dumps({"datasets": ["qasper", "squad_v2"]}), encoding="utf-8")
    assert rca._registered_roster(path) == ["qasper", "squad_v2"]
    for bad in ("qasper", [], ["qasper", 3], [""]):
        path.write_text(json.dumps({"datasets": bad}), encoding="utf-8")
        with pytest.raises(rca.AnalysisError, match="'datasets' must be a non-empty list"):
            rca._registered_roster(path)
