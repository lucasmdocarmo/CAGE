"""B3 (ADR-0172, 2026-10-16): the window bounds of a measured pressure window.

Slice 1, the runner and the dispatcher: the drain bound after the last
arrival (item 3), the plan's replay authorization and its record (item 1),
the pre-load that cycles its candidates (item 4), and the adapter timeout
taken from the drain bound.

Slice 2, the plan: the window shape (item 2: D_class and W from the class
floor), the replay ceiling an executable window refuses above and a dress
rehearsal records, the validator clauses that re-derive the record, the dry
window's flags, the projection and the header. Each test fails on the code
before the fix it names.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import math
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_run_campaign import (  # noqa: E402
    _cells,
    _dump,
    _rungs_for,
    _stub_plan,
    _tiny_grid,
    _write_calibrations,
    calibrations_a,  # noqa: F401  (fixture re-export)
    floor_table,  # noqa: F401
    rc,
    stub,  # noqa: F401
)
from src.analysis.goodput import TPOT_SLO_MULTIPLIER, TTFT_SLO_MULTIPLIER  # noqa: E402
from src.inference.engine import InferenceRequest  # noqa: E402
from src.orchestration import load_generator as lg  # noqa: E402
from src.orchestration.calibration import PROBE_WARMUP_S  # noqa: E402

SCRIPTS = REPO_ROOT / "scripts"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load("run_experiment_b3", SCRIPTS / "3_run" / "run_experiment.py")


def _run(coro):
    return asyncio.run(coro)


def _schedule(n: int, rate: float = 1000.0) -> lg.ArrivalSchedule:
    return lg.generate_arrival_schedule(rate, seed=0, n_requests=n, distribution="deterministic")


# --------------------------------------------------------------------------- #
# The dispatcher's drain bound (ADR-0172 item 3)
# --------------------------------------------------------------------------- #


def test_the_drain_cuts_requests_still_open_after_the_last_arrival() -> None:
    async def send(index: int) -> str:
        if index == 1:
            await asyncio.sleep(5.0)  # would hold the window open for 5 s
        return f"resp-{index}"

    async def scenario():
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        report = await lg.OpenLoopDispatcher().run(_schedule(3), send, drain_timeout_s=0.2)
        return report, loop.time() - t0

    report, elapsed = _run(scenario())
    assert elapsed < 2.0
    assert report.n_scheduled == 3 and report.n_sent == 3
    assert report.n_completed == 2 and report.n_errors == 1
    cut = report.records[1]
    assert cut.error == "drain timeout after 0.2 s"
    assert cut.result is None and not cut.completed
    assert cut.completion_ts is not None and cut.latency_from_scheduled_ms > 0
    assert cut.latency_from_send_ms is not None
    assert cut.to_row()["dispatch_error"] == "drain timeout after 0.2 s"
    assert report.drain["timeout_s"] == 0.2 and report.drain["n_cut"] == 1
    assert report.drain["adr"] == "ADR-0172"
    assert report.to_manifest()["drain"]["n_cut"] == 1


def test_without_a_drain_the_dispatcher_waits_for_every_completion() -> None:
    async def send(index: int) -> str:
        await asyncio.sleep(0.1)
        return f"resp-{index}"

    report = _run(lg.OpenLoopDispatcher().run(_schedule(3), send))
    assert report.n_completed == 3
    assert report.drain is None and report.replay is None
    manifest = report.to_manifest()
    assert manifest["drain"] is None and manifest["replay"] is None


@pytest.mark.parametrize("bad", [0, -1.0, math.nan, math.inf, "300", True])
def test_an_invalid_drain_bound_refuses(bad) -> None:
    async def send(index: int) -> str:
        return "x"

    with pytest.raises(lg.LoadGeneratorError, match="drain_timeout_s"):
        _run(lg.OpenLoopDispatcher().run(_schedule(1), send, drain_timeout_s=bad))


# --------------------------------------------------------------------------- #
# The replay authorization and its record (ADR-0172 item 1)
# --------------------------------------------------------------------------- #


def test_the_replay_guard_enforces_the_authorized_pass_ceiling() -> None:
    sched = lg.generate_arrival_schedule(10.0, seed=1, n_requests=31)  # 4 passes over 10
    with pytest.raises(lg.LoadGeneratorError) as exc:
        lg.ensure_no_measured_replay(sched, 10, allow_replay=True, max_passes=3)
    msg = str(exc.value)
    assert "4 times" in msg and "3 passes" in msg and "ADR-0172" in msg
    assert lg.ensure_no_measured_replay(sched, 10, allow_replay=True, max_passes=4) is True
    assert lg.ensure_no_measured_replay(sched, 10, allow_replay=True) is True
    # a schedule inside the pool never consults the ceiling
    assert lg.ensure_no_measured_replay(_schedule(5), 10, max_passes=1) is False
    with pytest.raises(lg.LoadGeneratorError, match="max_passes"):
        lg.ensure_no_measured_replay(sched, 10, allow_replay=True, max_passes=0)
    # the pin itself is untouched: no authorization, no replay
    with pytest.raises(lg.LoadGeneratorError, match="CAGE_ALLOW_REPLAY"):
        lg.ensure_no_measured_replay(sched, 10)


def test_the_replay_record_arithmetic() -> None:
    gold = lg.replay_record(453, 500)
    assert gold["passes"] == 1 and gold["replayed_arrivals"] == 0
    assert gold["first_pass_record_index_max"] == 452 and gold["authorized_max"] is None
    fast = lg.replay_record(6233, 500, authorized_max=15)
    assert fast["passes"] == 13 and fast["replayed_arrivals"] == 5733
    assert fast["first_pass_record_index_max"] == 499 and fast["authorized_max"] == 15
    assert fast["adr"] == "ADR-0172" and "first pass" in fast["rule"]
    empty = lg.replay_record(0, 5)
    assert empty["passes"] == 1 and empty["replayed_arrivals"] == 0
    for bad in ((10, 0), (-1, 5), (1.5, 5)):
        with pytest.raises(lg.LoadGeneratorError):
            lg.replay_record(*bad)
    with pytest.raises(lg.LoadGeneratorError, match="authorized_max"):
        lg.replay_record(10, 5, authorized_max=0)


# --------------------------------------------------------------------------- #
# The runner: authorization, record, pre-load replay, adapter timeout
# --------------------------------------------------------------------------- #


class _AsyncEngine:
    """Answers every streamed request at once; a hanging prompt sleeps 5 s."""

    def __init__(self, hang_prompts: tuple = ()) -> None:
        self.sent: List[str] = []
        self.hang = set(hang_prompts)

    async def async_stream_generate(self, request, on_first_token=None):
        self.sent.append(request.prompt)
        if request.prompt in self.hang:
            await asyncio.sleep(5.0)
        if on_first_token is not None:
            on_first_token()
        return types.SimpleNamespace(error=None, num_tokens=1)


def _reqs(tag: str, n: int) -> List[InferenceRequest]:
    return [
        InferenceRequest(prompt=f"{tag}{i}", max_tokens=4, temperature=0.0, top_p=1.0, request_id=f"{tag}{i}")
        for i in range(n)
    ]


def test_a_plan_authorized_replay_is_recorded_and_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CAGE_ALLOW_REPLAY", raising=False)
    engine = _AsyncEngine()
    kept, report = runner.dispatch_open_loop(
        engine, _reqs("m", 5), rate_qps=400.0, seed=7, n_arrivals=12, replay_max=3,
    )
    assert len(kept) == 12 and report.n_completed == 12
    assert engine.sent[5] == "m0" and engine.sent[11] == "m1"  # the pool cycles modulo
    rec = report.replay
    assert rec["n_unique"] == 5 and rec["n_arrivals"] == 12 and rec["passes"] == 3
    assert rec["authorized_max"] == 3 and rec["first_pass_record_index_max"] == 4
    assert report.to_manifest()["replay"]["passes"] == 3
    with pytest.raises(lg.LoadGeneratorError, match="authorized maximum of 2"):
        runner.dispatch_open_loop(_AsyncEngine(), _reqs("m", 5), rate_qps=400.0, seed=7, n_arrivals=12, replay_max=2)
    # without the authorization the E4 pin refuses as before
    with pytest.raises(lg.LoadGeneratorError, match="CAGE_ALLOW_REPLAY"):
        runner.dispatch_open_loop(_AsyncEngine(), _reqs("m", 5), rate_qps=400.0, seed=7, n_arrivals=12)
    # a schedule inside the pool records one pass
    _, plain = runner.dispatch_open_loop(_AsyncEngine(), _reqs("m", 5), rate_qps=400.0, seed=7, n_arrivals=5)
    assert plain.replay["passes"] == 1 and plain.replay["authorized_max"] is None


def test_the_drain_bound_reaches_the_dispatcher_through_the_runner() -> None:
    engine = _AsyncEngine(hang_prompts=("m2",))
    kept, report = runner.dispatch_open_loop(
        engine, _reqs("m", 3), rate_qps=400.0, seed=1, n_arrivals=3, drain_timeout_s=0.3,
    )
    assert report.drain["n_cut"] == 1 and report.drain["timeout_s"] == 0.3
    cut = [r for r in kept if r.error is not None]
    assert len(cut) == 1 and cut[0].error == "drain timeout after 0.3 s" and cut[0].result is None
    assert report.n_completed == 2


def test_a_pre_load_cut_is_counted_apart_from_the_measured_cuts() -> None:
    engine = _AsyncEngine(hang_prompts=("p1",))
    kept, report = runner.dispatch_open_loop(
        engine, _reqs("m", 3), rate_qps=400.0, seed=1, n_arrivals=3,
        preload_requests=_reqs("p", 4), drain_timeout_s=0.3,
    )
    assert report.preload["n_cut"] == 1 and report.preload["n_unique"] == 4 and report.preload["passes"] == 1
    assert report.drain["n_cut"] == 0 and all(r.error is None for r in kept)


def test_the_pre_load_replays_its_candidates_when_authorized() -> None:
    from src.data.loader import CAGExample

    cands = [CAGExample(id=f"c{i}", question=f"q{i}?", context=[f"paper {i}"], answer="a", metadata={})
             for i in range(40)]
    measured = {"c0", "c1"}
    keys = runner.warmup_context_keys([c for c in cands if c.id in measured])
    kw = dict(measured_ids=measured, measured_context_keys=keys, seed=42, trial=1)
    short = runner.draw_preload_pool(cands, n_max=100, **kw)
    assert len(short) == 38  # the pre-B3 behavior: every disjoint candidate once
    full = runner.draw_preload_pool(cands, n_max=100, allow_replay=True, **kw)
    ids = [c.id for c in full]
    assert len(ids) == 100 and len(set(ids)) == 38
    assert ids[:38] == [c.id for c in short] and ids[38] == ids[0] and ids[99] == ids[99 % 38]
    # a pool that already fills n_max is unchanged by the authorization
    exact = runner.draw_preload_pool(cands, n_max=20, allow_replay=True, **kw)
    assert len({c.id for c in exact}) == 20


def test_the_adapter_timeout_comes_from_the_drain_bound() -> None:
    assert runner.adapter_timeout_kwargs(None) == {}
    assert runner.adapter_timeout_kwargs(300) == {"timeout": 300}
    assert runner.adapter_timeout_kwargs(299.5) == {"timeout": 300}


def test_the_window_bounds_ride_open_loop_windows_only() -> None:
    ok = dict(workload_mode="open_loop", drain_timeout_s=300.0, replay_max=15)
    assert runner.validate_open_loop_window_bounds(**ok) is None
    assert runner.validate_open_loop_window_bounds(workload_mode="single", drain_timeout_s=None, replay_max=None) is None
    with pytest.raises(lg.LoadGeneratorError, match="open-loop windows only"):
        runner.validate_open_loop_window_bounds(workload_mode="single", drain_timeout_s=300.0, replay_max=None)
    with pytest.raises(lg.LoadGeneratorError, match="open-loop windows only"):
        runner.validate_open_loop_window_bounds(workload_mode="batched", drain_timeout_s=None, replay_max=15)
    for bad in (0, -5, math.nan, True):
        with pytest.raises(lg.LoadGeneratorError, match="open_loop_drain_timeout_s"):
            runner.validate_open_loop_window_bounds(workload_mode="open_loop", drain_timeout_s=bad, replay_max=None)
    for bad in (0, -1, 1.5, True):
        with pytest.raises(lg.LoadGeneratorError, match="open_loop_replay_max"):
            runner.validate_open_loop_window_bounds(workload_mode="open_loop", drain_timeout_s=None, replay_max=bad)


# --------------------------------------------------------------------------- #
# The plan: the window shape (ADR-0172 item 2)
# --------------------------------------------------------------------------- #

#: The S0 floors the ADR sized its durations on [V MyDocs/RunPod/B3_WINDOW_
#: DECISION_2026-10-16.md, Table 2; experiments/S0/2026-10-14_0016 manifest
#: slo_floors and the RULER ladder floors]: (ttft_s, tpot_s).
_S0_FLOORS = {
    ("qasper", "vllm"): (0.1318, 0.01099),
    ("qasper", "sglang"): (0.1302, 0.01115),
    ("ruler", "vllm"): (2.176, 0.01064),
    ("ruler", "sglang"): (2.154, 0.01398),
}


def test_registered_window_design_constants() -> None:
    assert rc.WINDOW_REGIME_S == 60.0
    assert rc.DRAIN_TIMEOUT_S == 300.0
    assert rc.REPLAY_FACTOR_MAX == 15
    assert rc.WINDOW_MAX_TOKENS == 256 and rc.RULER_OUTPUT_TOKENS == 256
    assert rc.WINDOW_DESIGN_ADR == "ADR-0172"
    assert rc.WINDOW_REQUESTS == 2000  # the pool is the class n, no longer 200
    # the latency bound reads goodput's two multipliers, the one threshold
    # source (ADR-0172 item 2 names them: 10 x TTFT + 5 x TPOT x max_tokens)
    assert (TTFT_SLO_MULTIPLIER, TPOT_SLO_MULTIPLIER) == (10.0, 5.0)
    assert rc.class_floor_latency_bound_s(0.1, 0.01, 256) == pytest.approx(1.0 + 12.8)


def test_window_shape_reproduces_the_adr_durations_on_the_s0_floors() -> None:
    # ADR-0172 item 2: 86 s for the Qasper classes on both engines, 106 s
    # (vLLM) and 110 s (SGLang) for RULER. Independent arithmetic per row:
    #   qasper vllm   L = 1.318 + 5 x 0.01099 x 256 = 15.385 -> ceil(85.385) = 86
    #   qasper sglang L = 1.302 + 5 x 0.01115 x 256 = 15.574 -> ceil(85.574) = 86
    #   ruler vllm    L = 21.76 + 5 x 0.01064 x 256 = 35.379 -> ceil(105.379) = 106
    #   ruler sglang  L = 21.54 + 5 x 0.01398 x 256 = 39.434 -> ceil(109.434) = 110
    want = {("qasper", "vllm"): 86, ("qasper", "sglang"): 86, ("ruler", "vllm"): 106, ("ruler", "sglang"): 110}
    for key, (ttft, tpot) in _S0_FLOORS.items():
        rec = rc.window_shape(
            rate_frac=1.0, lambda_star=1.0, floor_ttft_s=ttft, floor_tpot_s=tpot,
            max_tokens=256, pool=500, floor_source="test",
        )
        assert rec["duration_s"] == want[key], key
        assert rec["latency_bound_s"] == pytest.approx(10 * ttft + 5 * tpot * 256)
        assert rec["duration_s"] == math.ceil(PROBE_WARMUP_S + 60.0 + rec["latency_bound_s"])
        assert rec["warmup_s"] == 10.0 and rec["regime_s"] == 60.0
        assert rec["replay_max"] == 15 and rec["drain_timeout_s"] == 300.0 and rec["adr"] == "ADR-0172"
    # the S0 fastest class (348 tokens, vllm, lambda* 60 qps at its loosest
    # rung) at 1.2 x lambda* over the Qasper pool of 500 (item 5):
    # W = ceil(1.2 x 60 x 86) = 6192, 13 passes, inside the ceiling of 15
    fast = rc.window_shape(
        rate_frac=1.2, lambda_star=60.0, floor_ttft_s=0.1318, floor_tpot_s=0.01099,
        max_tokens=256, pool=500, floor_source="test",
    )
    assert fast["arrivals"] == 6192 and fast["passes"] == 13
    # RULER at its S0 lambda* (0.46 qps) and 1.2 x over the 300-prompt pool:
    # W = ceil(1.2 x 0.46 x 106) = 59, one pass
    ruler = rc.window_shape(
        rate_frac=1.2, lambda_star=0.46, floor_ttft_s=2.176, floor_tpot_s=0.01064,
        max_tokens=256, pool=300, floor_source="test",
    )
    assert ruler["arrivals"] == 59 and ruler["passes"] == 1 and ruler["pool"] == 300


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"rate_frac": 0}, "rate_frac"),
        ({"rate_frac": True}, "rate_frac"),
        ({"lambda_star": math.nan}, "lambda_star"),
        ({"lambda_star": -1.0}, "lambda_star"),
        ({"floor_ttft_s": 0.0}, "floor_ttft_s"),
        ({"floor_tpot_s": math.inf}, "floor_tpot_s"),
        ({"floor_tpot_s": None}, "floor_tpot_s"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"max_tokens": 256.0}, "max_tokens"),
        ({"pool": 0}, "pool"),
        ({"pool": 2.0}, "pool"),
    ],
)
def test_window_shape_refuses_malformed_inputs(overrides, match) -> None:
    kwargs: Dict[str, Any] = dict(
        rate_frac=0.85, lambda_star=8.0, floor_ttft_s=0.1, floor_tpot_s=0.01,
        max_tokens=256, pool=2000, floor_source="test",
    )
    kwargs.update(overrides)
    with pytest.raises(rc.PlanError, match=match):
        rc.window_shape(**kwargs)


def _ruler_grid(**overrides: Any):
    kwargs: Dict[str, Any] = dict(
        f2_baselines=("B1",), f2_budgets=(1.0,), f2_rates=(0.85,),
        f2_ruler_baselines=("B1",), f2_ruler_tasks=("qa",),
    )
    kwargs.update(overrides)
    return _tiny_grid(**kwargs)


def test_the_cell_shape_picks_the_class_floor_and_falls_back_for_ruler() -> None:
    grid = _ruler_grid()
    cells = {c.dataset: c for c in rc.enumerate_cells(grid) if c.spec.family == "F2"}
    qasper, ruler = cells["qasper"], cells["ruler"]
    floors = json.dumps({"vllm": {"ttft_s": 0.1, "tpot_s": 0.01}})
    # a Qasper class sizes on the manifest's engine floor
    rec = rc._window_shape_for(qasper.spec, qasper, 8.0, 2000, floors, {}, blocked=False)
    assert rec["duration_s"] == 84 and rec["arrivals"] == 572 and rec["max_tokens"] == 256
    assert rec["floor_source"] == "manifest slo_floors[vllm] (ADR-0117)"
    # RULER sizes on its own ladder floor when the rung artifact carries one
    rec = rc._window_shape_for(
        ruler.spec, ruler, 0.46, 300, floors,
        {"ruler_floor": {"ttft_s": 2.176, "tpot_s": 0.01064}}, blocked=False,
    )
    assert rec["duration_s"] == 106 and rec["floor_ttft_s"] == 2.176
    assert rec["floor_source"].startswith("RULER ladder floor")
    # ... and on the engine floor, saying so, when the ladder was operator-started
    rec = rc._window_shape_for(ruler.spec, ruler, 0.46, 300, floors, {}, blocked=False)
    assert rec["duration_s"] == 84 and "operator start" in rec["floor_source"]
    # no floor at all: a blocked cell records None, an executable one refuses
    assert rc._window_shape_for(qasper.spec, qasper, 8.0, 2000, None, {}, blocked=True) is None
    with pytest.raises(rc.PlanError, match="no manifest slo_floors\\[vllm\\].*to size its window"):
        rc._window_shape_for(qasper.spec, qasper, 8.0, 2000, None, {}, blocked=False)
    with pytest.raises(rc.PlanError, match="RULER ladder floor is absent"):
        rc._window_shape_for(ruler.spec, ruler, 0.46, 300, json.dumps({"sglang": {"ttft_s": 0.1, "tpot_s": 0.01}}), {}, blocked=False)


# --------------------------------------------------------------------------- #
# The plan: the replay ceiling (ADR-0172 items 1 and 5)
# --------------------------------------------------------------------------- #


def _f2_grid(**overrides: Any):
    kwargs: Dict[str, Any] = dict(f2_baselines=("B1",), f2_budgets=(1.0,), f2_rates=(0.85,))
    kwargs.update(overrides)
    return _tiny_grid(**kwargs)


def _window_cells(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [s for s in _cells(plan) if s["family"] in ("F2", "F3")]


def test_an_executable_window_above_the_ceiling_blocks_the_cell_naming_the_manifest_size(floor_table, stub, tmp_path) -> None:
    # Fixture: lambda* 8.0 x r = 12 at r = 1.5; vllm floors (0.12, 0.02):
    # L = 26.8 s, D = 97 s; W = ceil(1.2 x 12 x 97) = 1397. A 20-id pool needs
    # 70 passes (above 15): the CELL is blocked (audit 2026-10-17, G24: before,
    # the whole plan refused after the ladders had billed), naming the pool
    # that would fit (ceil(1397 / 15) = 94 ids per trial). A 100-id pool needs
    # 14 passes and runs.
    blocked_plan = _stub_plan(_f2_grid(f2_budgets=(1.5,), f2_rates=(1.2,), achievable_n={"qasper": 20}), floor_table, stub.cmd)
    (blocked,) = _window_cells(blocked_plan)
    msg = blocked["blocked_on"]
    assert msg.startswith(rc.PASS_CEILING_BLOCK_PREFIX)
    assert "needs 1397 arrivals over a pool of 20 (70 passes) for its 97 s window" in msg
    assert "above the registered ceiling of 15 passes (ADR-0172)" in msg
    assert "at least 94 ids per trial for qasper" in msg
    assert blocked["window"]["passes"] == 70 and blocked_plan["counts"]["blocked"] >= 1
    # the blocked plan loads (the stale-plan ceiling check skips a blocked cell)
    out = tmp_path / "blocked.json"
    out.write_text(json.dumps(blocked_plan), encoding="utf-8")
    rc.load_plan(out)
    plan = _stub_plan(_f2_grid(f2_budgets=(1.5,), f2_rates=(1.2,), achievable_n={"qasper": 100}), floor_table, stub.cmd)
    (window,) = _window_cells(plan)
    assert window["window"]["arrivals"] == 1397 and window["window"]["passes"] == 14
    assert window["num_queries"] == 100 and window["window_arrivals"] == 1397
    assert plan["window_design"]["replay_ceiling_enforced"] is True


def test_a_dress_rehearsal_records_passes_above_the_ceiling_without_refusing(tmp_path, floor_table, calibrations_a) -> None:
    # ADR-0144: a rehearsal is never data, so the ceiling is recorded, not
    # enforced; the 100-id Qasper manifest under the fastest class (4 x 8 x
    # 1.5 x 1.2 = 57.6 rps over 97 s = 5588 arrivals) needs 56 passes.
    manifests = {
        "squad_v2": REPO_ROOT / "data/manifests/squad_v2_50x3_seed42.json",
        "qasper": REPO_ROOT / "data/manifests/qasper_100x3_seed42.json",
    }
    plan = rc.build_plan(
        "a", rc.load_floor_table(floor_table), window_duration_s=300.0, calibrations=calibrations_a,
        rung_calibrations=_rungs_for(calibrations_a, "a"), query_manifests=manifests, rehearsal_n=50,
    )
    executable = [s for s in _window_cells(plan) if s["blocked_on"] is None]
    passes = [s["window"]["passes"] for s in executable]
    assert max(passes) > rc.REPLAY_FACTOR_MAX
    assert plan["window_design"]["replay_ceiling_enforced"] is False
    assert plan["window_spans"]["passes_max"] == max(passes)
    # every window still carries the runner's authorization at the registered
    # ceiling: the runner refuses above it, so the rehearsal's fast windows
    # are the operator's visible debt (fix list A4), never silent
    for s in executable:
        argv = s["argv"]
        assert argv[argv.index("--window-replay-max") + 1] == "15"
    # and the plan loads: the validator knows a rehearsal
    assert rc.load_plan(_dump(tmp_path, plan, "rehearsal.json"))["rehearsal"]["n"] == 50


# --------------------------------------------------------------------------- #
# The plan: the record, the validator, the dry window, the projection, the header
# --------------------------------------------------------------------------- #


def test_every_window_cell_carries_a_record_the_validator_re_derives(tmp_path, floor_table, stub) -> None:
    # Fixture: r = 1.0 so lambda* 8.0, 0.85 x 8 x 97 = 659.6 -> 660 arrivals.
    plan = _stub_plan(_f2_grid(), floor_table, stub.cmd)
    (window,) = _window_cells(plan)
    rec = window["window"]
    assert rec["duration_s"] == 97 and rec["arrivals"] == 660 == window["window_arrivals"]
    assert rec["pool"] == 2000 and rec["passes"] == 1
    assert window["window_span_s_expected"] == pytest.approx(660 / (0.85 * 8.0))
    assert window["window_span_below_floor"] is False
    for s in _cells(plan):
        if s["family"] not in ("F2", "F3"):
            assert s["window"] is None and s["window_arrivals"] is None
    assert rc.load_plan(_dump(tmp_path, plan, "fresh.json"))["counts"]["cells"] == plan["counts"]["cells"]

    def _broken(name: str, spoil) -> Path:
        copy = json.loads(json.dumps(plan))
        spoil(next(s for s in _cells(copy) if s["family"] == "F2"), copy)
        return _dump(tmp_path, copy, name)

    def _set_flag(step, flag, value):
        step["argv"][step["argv"].index(flag) + 1] = value

    def _drop_flag(step, flag):
        at = step["argv"].index(flag)
        del step["argv"][at:at + 2]

    # the record is missing on an executable window
    with pytest.raises(rc.RunError, match="carries no window record"):
        rc.load_plan(_broken("no_record.json", lambda s, p: s.__setitem__("window", None)))
    # the record does not re-derive from its own inputs (a hand-edited count)
    with pytest.raises(rc.RunError, match="does not re-derive.*recorded duration 97 s, 661 arrivals"):
        rc.load_plan(_broken("drift.json", lambda s, p: s["window"].__setitem__("arrivals", 661)))
    # a malformed record (its floor gone)
    with pytest.raises(rc.RunError, match="window record is malformed"):
        rc.load_plan(_broken("malformed.json", lambda s, p: s["window"].__setitem__("floor_tpot_s", None)))
    # the step's arrivals disagree with the record
    with pytest.raises(rc.RunError, match="window_arrivals=659 != the window's 660"):
        rc.load_plan(_broken("arrivals.json", lambda s, p: s.__setitem__("window_arrivals", 659)))
    # the argv disagrees with the step (the runner would issue another W)
    with pytest.raises(rc.RunError, match="carries --arrival-count '661' but its window needs 660 arrivals"):
        rc.load_plan(_broken("argv.json", lambda s, p: _set_flag(s, "--arrival-count", "661")))
    # the replay authorization and the drain drifted or vanished
    with pytest.raises(rc.RunError, match="carries --window-replay-max '16', the registered value is '15'"):
        rc.load_plan(_broken("replay16.json", lambda s, p: _set_flag(s, "--window-replay-max", "16")))
    with pytest.raises(rc.RunError, match="carries --drain-timeout-s None, the registered value is '300'"):
        rc.load_plan(_broken("no_drain.json", lambda s, p: _drop_flag(s, "--drain-timeout-s")))
    # a non-window cell carrying a window bound
    copy = json.loads(json.dumps(plan))
    f1 = next(s for s in _cells(copy) if s["family"] == "F1")
    f1["argv"] += ["--drain-timeout-s", "300"]
    with pytest.raises(rc.RunError, match="carries --drain-timeout-s .*window bounds ride window cells only"):
        rc.load_plan(_dump(tmp_path, copy, "f1_drain.json"))
    # a consistent record above the ceiling on a plan that is not a rehearsal:
    # pool 20 -> 33 passes (the per-row n clause fires too; this clause names
    # the passes)
    copy = json.loads(json.dumps(plan))
    cell = next(s for s in _cells(copy) if s["family"] == "F2")
    cell["num_queries"] = 20
    _set_flag(cell, "--num-queries", "20")
    cell["window"] = rc.window_shape(
        rate_frac=0.85, lambda_star=8.0, floor_ttft_s=0.12, floor_tpot_s=0.02,
        max_tokens=256, pool=20, floor_source=cell["window"]["floor_source"],
    )
    assert cell["window"]["passes"] == 33
    with pytest.raises(rc.RunError, match="needs 33 passes"):
        rc.load_plan(_dump(tmp_path, copy, "ceiling.json"))
    # fresh review 2026-10-16 (MEDIUM): a record consistent with ITS OWN floors
    # but not with the plan's floor pin (a plan built on older floors)
    copy = json.loads(json.dumps(plan))
    cell = next(s for s in _cells(copy) if s["family"] == "F2")
    cell["window"] = rc.window_shape(
        rate_frac=0.85, lambda_star=8.0, floor_ttft_s=0.5, floor_tpot_s=0.02,
        max_tokens=256, pool=2000, floor_source=cell["window"]["floor_source"],
    )
    cell["window_arrivals"] = cell["window"]["arrivals"]
    _set_flag(cell, "--arrival-count", str(cell["window_arrivals"]))
    cell["window_span_s_expected"] = cell["window_arrivals"] / (0.85 * 8.0)
    with pytest.raises(rc.RunError, match=r"window record floors \(0\.5, 0\.02\) differ from the plan's CAGE_SLO_FLOORS_JSON pin for vllm \(0\.12, 0\.02\)"):
        rc.load_plan(_dump(tmp_path, copy, "stale_floor.json"))


def test_a_stale_dry_window_without_the_drain_refuses(tmp_path, floor_table, stub) -> None:
    # fresh review 2026-10-16 (LOW): the dry window clause checks the drain
    # too, and refuses the pass ceiling on a gate
    plan = _stub_plan(_f2_grid(), floor_table, stub.cmd)
    assert rc.load_plan(_dump(tmp_path, plan, "fresh_dry.json"))
    copy = json.loads(json.dumps(plan))
    (dry,) = [s for s in copy["steps"] if s["kind"] == "dry_window"]
    at = dry["argv"].index("--drain-timeout-s")
    del dry["argv"][at:at + 2]
    with pytest.raises(rc.RunError, match="dry window carries --drain-timeout-s None"):
        rc.load_plan(_dump(tmp_path, copy, "dry_no_drain.json"))
    copy = json.loads(json.dumps(plan))
    (dry,) = [s for s in copy["steps"] if s["kind"] == "dry_window"]
    dry["argv"] += ["--window-replay-max", "15"]
    with pytest.raises(rc.RunError, match="pass ceiling, which rides window cells only"):
        rc.load_plan(_dump(tmp_path, copy, "dry_ceiling.json"))


def test_the_dry_window_keeps_the_drain_and_sheds_the_window_cell_bounds(floor_table, stub) -> None:
    plan = _stub_plan(_f2_grid(), floor_table, stub.cmd)
    (dry,) = [s for s in plan["steps"] if s["kind"] == "dry_window"]
    argv = dry["argv"]
    for flag in ("--arrival-count", "--window-replay-max", "--open-loop-preload-s"):
        assert flag not in argv, flag
    assert argv[argv.index("--duration-s") + 1] == f"{rc.DRY_WINDOW_DURATION_S:g}"
    assert argv[argv.index("--drain-timeout-s") + 1] == "300"
    assert dry["env"][rc.LADDER_REPLAY_ENV] == "1"  # the gate replays as before (ADR-0153)


def test_the_projection_charges_each_window_its_duration_and_tail(floor_table, stub) -> None:
    plan = _stub_plan(_f2_grid(), floor_table, stub.cmd)
    (window,) = _window_cells(plan)
    cost = rc.RUN_CELL_SECONDS[("vllm", "none")]
    windows = window["windows"]
    rec = window["window"]
    # preparation scales with the pool (2000 / 50 = 40 x) and is paid once per
    # window (G26, 2026-10-17: the runner prepares per trial); each window costs
    # its 30 s pre-load, its 97 s duration and a 26.8 s tail
    want = windows * (cost[0] * (2000 / rc.RUN_STEP_MEASURED_QUERIES) + 30.0 + rec["duration_s"] + rec["latency_bound_s"])
    assert windows == 3
    assert rc.run_projection([window])["by_kind_s"]["cell"] == pytest.approx(want, abs=0.06)
    # the pre-B3 arithmetic charged the pool's scale on the per-window cost too
    old = (cost[0] + cost[1] * windows) * (2000 / rc.RUN_STEP_MEASURED_QUERIES) + 30.0 * windows
    assert want != pytest.approx(old)


def test_the_header_registers_the_window_design(floor_table, stub) -> None:
    plan = _stub_plan(_f2_grid(), floor_table, stub.cmd)
    assert plan["window_design"] == {
        "adr": "ADR-0172",
        "warmup_s": 10.0,
        "regime_s": 60.0,
        "latency_bound_rule": (
            "TTFT_SLO_MULTIPLIER x floor TTFT + TPOT_SLO_MULTIPLIER x floor TPOT x max_tokens "
            "(goodput.py), the engine floor of the manifest, the RULER ladder floor for RULER"
        ),
        "max_tokens": 256,
        "ruler_max_tokens": 256,
        "drain_timeout_s": 300.0,
        "replay_factor_max": 15,
        "replay_ceiling_enforced": True,
        "quality_rows": (
            "every arrival (ADR-0172 amendment 2026-10-17, E18: the scoring pass scores "
            "identical inputs once per window and the replays inherit, recorded as "
            "quality_source; the first pass record_index <= first_pass_record_index_max "
            "stays an integrity record)"
        ),
        "serving_rows": "every arrival",
    }
    ws = plan["window_spans"]
    assert ws["durations_s"] == [97] and ws["passes_max"] == 1 and ws["below_floor"] == 0
    assert "ADR-0172" in ws["rule"]
