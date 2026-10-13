"""S0F-88 audit batch (2026-10-12): the owner's "review if EVERY FIX ... is
completed" found fixes without a behavioral test and gaps on the production
path. Each test here fails on the code before the audit fix it names.

B-2 (HIGH): the cal-v3 served-rate gate had no test; every fake completed a
request at its arrival, so a revert to the attainment-only rule passed.
B-13: the floor serves distinct prompts and proves each one cold.
Audit: the probe step records the window's peak in flight; the ladder step
records both attainments; a laddered class never falls back to the
interpolation; with every class laddered the smallest class ladders only its
own rungs.
"""

from __future__ import annotations

import asyncio
import math
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(REPO_ROOT / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.inference.engine import InferenceResponse  # noqa: E402
from src.orchestration import load_generator as lg  # noqa: E402
from src.orchestration.calibration import (  # noqa: E402
    FLOOR_N_REQUESTS,
    PROBE_WARMUP_S,
    PROBE_WINDOW_S,
    CalibrationError,
    ProbeStep,
    decide_lambda_star,
)
from test_batch_b_pressure import FakeWorld, _calibrate, _pressure_grid, _prefix_on_grid  # noqa: E402
from test_run_campaign import (  # noqa: E402
    _cells,
    _run_root,
    _stub_plan,
    floor_table,  # noqa: F401  (fixture re-export)
    stub,  # noqa: F401
    _dump,
    _floor_table_doc,
    _rungs_for,
    _tiny_grid,
    _write_calibrations,
    rc,
)

SCRIPTS = REPO_ROOT / "scripts"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cc = _load("calibrate_cell_s0f88_b3", SCRIPTS / "3_run" / "calibrate_cell.py")


# --------------------------------------------------------------------------- #
# B-2: the served-rate gate decides lambda*, not attainment alone
# --------------------------------------------------------------------------- #


def test_a_step_that_completes_everything_late_is_not_sustainable_b2() -> None:
    """S0 attempt 4: attainment read 1.000 at 7.1 qps while the engine served
    4.4 rps (requests finished long after the window)."""
    late = ProbeStep(rate_qps=7.1, n_scheduled=461, n_completed=461, throughput_rps=7.09,
                     n_served_in_window=286)
    assert late.attainment == 1.0 and late.served_ratio == pytest.approx(286 / 461)
    assert late.sustainable is False
    ok = ProbeStep(rate_qps=4.2, n_scheduled=273, n_completed=273, throughput_rps=4.2,
                   n_served_in_window=270)
    est = decide_lambda_star([ok, late])
    assert est.label == "ESTIMATED" and est.lambda_star_qps == 4.2


class _LateWorld(FakeWorld):
    """Every request completes (attainment 1.0) but the engine serves only
    ``cap`` requests per second, first come first served, so above ``cap``
    the completions spill past the window end."""

    def __call__(self, argv, env) -> int:
        rc_ = super().__call__(argv, env)
        if "--campaign-root" not in argv or rc_ != 0:
            return rc_
        root = Path(argv[argv.index("--campaign-root") + 1])
        (path,) = list(root.glob("cells/*/window_*-01/requests.jsonl"))
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        cap = self.cap
        out = []
        for k, row in enumerate(rows):
            arrival = row["arrival_s"]
            done = max(arrival, PROBE_WARMUP_S + (k + 1) / cap)
            out.append({**row, "ok": True, "completion_ts": row["scheduled_ts"] - arrival + done})
        path.write_text("".join(json.dumps(r) + "\n" for r in out), encoding="utf-8")
        return 0


def test_the_rung_ladder_stops_at_the_served_rate_not_at_attainment_b2(tmp_path: Path) -> None:
    grid = _tiny_grid(f1_baselines=("B1",), f2_baselines=("B1",), f2_budgets=(1.0,), f2_rates=(0.85,))
    doc = _calibrate(tmp_path, _LateWorld(cap=5.0), grid, anchor_only=True)
    rung = doc["rungs"]["1"]
    # attainment is 1.0 on every window, so the attainment-only rule of
    # cal-v2 would climb to LADDER_EXHAUSTED; the served gate stops near cap
    assert all(s["attainment"] == 1.0 for s in rung["steps"])
    assert rung["label"] == "ESTIMATED"
    assert rung["lambda_star_qps"] <= 5.0 / 0.9 * 1.05
    unsustainable = [s for s in rung["steps"] if not s["sustainable"]]
    assert unsustainable and all(s["served_ratio"] < 0.9 for s in unsustainable)


def test_the_probe_counts_completions_inside_the_window_b2(monkeypatch: pytest.MonkeyPatch) -> None:
    """calibrate_cell.probe_rate: every request completes, half after 75 s."""
    def fake_run(self, schedule, send):
        records = []
        for i, offset in enumerate(schedule.offsets_s):
            rec = lg.RequestRecord(index=i, scheduled_offset_s=offset, scheduled_ts=offset)
            rec.actual_send_ts = offset
            rec.completion_ts = offset + (1.0 if i % 2 == 0 else PROBE_WINDOW_S + 5.0)
            rec.result = types.SimpleNamespace(error=None)
            records.append(rec)
        report = lg.DispatchReport(schedule=schedule, run_start_ts=0.0, records=records)
        report.max_in_flight_observed = 37

        async def _done():
            return report
        return _done()

    monkeypatch.setattr(lg.OpenLoopDispatcher, "run", fake_run)
    step = asyncio.run(cc.probe_rate(object(), [object()], rate_qps=4.0, seed=1))
    assert step.attainment == 1.0
    assert step.served_ratio is not None and step.served_ratio < 0.6
    assert step.sustainable is False
    assert step.max_in_flight == 37  # the window's peak in flight is recorded


# --------------------------------------------------------------------------- #
# B-13: distinct, provably cold floor prompts
# --------------------------------------------------------------------------- #


def _resp(cached: Any = None) -> InferenceResponse:
    r = InferenceResponse(request_id="f", generated_text="a b c d e f", ttft_ms=100.0,
                          total_time_ms=600.0, num_tokens=6, model_name="m", finish_reason="stop")
    r.cached_prompt_tokens = cached
    return r


class _FloorAdapter:
    def __init__(self, cached: Any = 0) -> None:
        self.cached = cached
        self.prompts: List[str] = []

    def flush_cache(self) -> None:
        pass

    async def async_stream_generate(self, request, on_first_token=None):
        self.prompts.append(request.prompt)
        return _resp(self.cached)


def _blocks(tmp_path: Path, n: int) -> List[Any]:
    path = tmp_path / f"m{n}.json"
    path.write_text(json.dumps({"blocks": [{"block_id": f"b{i}", "text": f"Paper {i}."} for i in range(n)]}),
                    encoding="utf-8")
    return cc.build_requests(str(path))


def test_the_floor_serves_distinct_cold_prompts_and_records_it_b13(tmp_path: Path) -> None:
    adapter = _FloorAdapter(cached=0)
    floor = asyncio.run(cc.measure_floor(adapter, _blocks(tmp_path, 35)))
    assert len(set(adapter.prompts)) == FLOOR_N_REQUESTS == len(adapter.prompts)
    assert floor.n_distinct_prompts == 30 and floor.cached_tokens_max == 0
    assert floor.to_manifest()["cached_tokens_max"] == 0
    # an engine without cache telemetry records None, never a fabricated 0
    floor = asyncio.run(cc.measure_floor(_FloorAdapter(cached=None), _blocks(tmp_path, 30)))
    assert floor.cached_tokens_max is None


def test_the_floor_refuses_repeats_and_cached_prompts_b13(tmp_path: Path) -> None:
    """S0 attempt 4: one squad_v2 block (the calibration manifest's only
    block) served 30 times."""
    with pytest.raises(CalibrationError, match="distinct prompts"):
        asyncio.run(cc.measure_floor(_FloorAdapter(), _blocks(tmp_path, 1)))
    with pytest.raises(CalibrationError, match="cached prompt tokens"):
        asyncio.run(cc.measure_floor(_FloorAdapter(cached=812), _blocks(tmp_path, 30)))


def test_the_calibration_manifest_carries_enough_distinct_blocks_b13() -> None:
    common = (SCRIPTS / "6_experiments" / "profiles" / "_common.env").read_text(encoding="utf-8")
    line = next(l for l in common.splitlines() if l.startswith("CALIBRATION_MANIFEST="))
    manifest = json.loads((REPO_ROOT / line.split("=", 1)[1].strip().strip('"')).read_text(encoding="utf-8"))
    assert len({b["text"] for b in manifest["blocks"]}) >= FLOOR_N_REQUESTS


# --------------------------------------------------------------------------- #
# Audit: ladder records; laddered classes fail closed; trimmed smallest rungs
# --------------------------------------------------------------------------- #


def test_every_ladder_step_records_both_attainments_and_its_peak(tmp_path: Path) -> None:
    grid = _tiny_grid(f1_baselines=("B1",), f2_baselines=("B1",), f2_budgets=(1.0,), f2_rates=(0.85,))
    doc = _calibrate(tmp_path, FakeWorld(cap=6.0), grid, anchor_only=True)
    for step in doc["rungs"]["1"]["steps"]:
        assert "regime_attainment" in step and "max_in_flight" in step
        assert "served_in_window_rps" in step and "served_rps" not in step


def test_a_laddered_class_with_a_failed_rung_refuses_the_plan(tmp_path: Path) -> None:
    floor = tmp_path / "ft.json"
    floor.write_text(json.dumps(_floor_table_doc()), encoding="utf-8")
    cal = _write_calibrations(tmp_path / "cal", ("vllm", "sglang"))
    rungs = _rungs_for(cal, "a")
    grid = rc.get_session_grid("a")
    for engine, path in rungs.items():
        doc = json.loads(path.read_text(encoding="utf-8"))
        for template, class_rungs in rc.measured_pressure_classes(grid, engine, None):
            seq = rc.demand_seq_tokens(grid, template.spec)
            table = {f"{r:g}": {"r": r, "label": "ESTIMATED", "lambda_star_qps": 20.0 * r} for r in class_rungs}
            first = f"{class_rungs[0]:g}"
            table[first] = {"r": class_rungs[0], "label": "LADDER_EXHAUSTED", "lambda_star_qps": None}
            doc["classes"][str(seq)] = {"role": "measured", "seq_tokens": seq, "arm": template.spec.arm,
                                        "rungs": table}
        path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(rc.PlanError, match="never falls back to the interpolation"):
        rc.build_plan("a", rc.load_floor_table(floor), window_duration_s=300.0,
                      calibrations=cal, rung_calibrations=rungs)


def test_with_every_class_laddered_the_smallest_class_ladders_only_its_rungs(tmp_path: Path) -> None:
    # B7 (retr-reuse, the only class below the anchor, so the smallest)
    # carries cells at r=0.5 only; the anchor carries r=1.0 and r=0.5
    grid = _tiny_grid(
        f1_baselines=("B1",), f2_baselines=("B1",), f2_budgets=(1.0, 0.5), f2_rates=(0.85,),
        f3_baselines=("B7",), f3_budgets=(0.5,), f3_rates=(0.95,),
    )
    small = rc.smallest_pressure_class(grid, "vllm", None)
    assert small is not None and small.baseline_id == "B7"
    small_seq = rc.demand_seq_tokens(grid, small.spec)
    own = {float(c.spec.budget_r) for c in rc.enumerate_cells(grid)
           if c.spec.engine == "vllm" and c.blocked_on is None and c.spec.family in ("F2", "F3")
           and rc.demand_seq_tokens(grid, c.spec) == small_seq}
    doc = _calibrate(tmp_path, FakeWorld(cap=6.0, cap_by_arm={small.spec.arm: 24.0}), grid, all_classes=True)
    rec = doc["classes"][str(small_seq)]
    assert {float(k) for k in rec["rungs"]} == own
    plain = _calibrate(tmp_path / "p", FakeWorld(cap=6.0, cap_by_arm={small.spec.arm: 24.0}), grid)
    assert {float(k) for k in plain["classes"][str(small_seq)]["rungs"]} == {1.0, 0.5}


# --------------------------------------------------------------------------- #
# U3 (owner decision 2026-10-12): steady-state pre-load, then 100 measured
# arrivals per rehearsal pressure window
# --------------------------------------------------------------------------- #


class _AsyncEngine:
    """Answers every streamed request at once; records what it was sent."""

    def __init__(self, fail_prompts: tuple = ()) -> None:
        self.sent: List[str] = []
        self.fail_prompts = set(fail_prompts)

    async def async_stream_generate(self, request, on_first_token=None):
        self.sent.append(request.prompt)
        if on_first_token is not None:
            on_first_token()
        err = "Cannot connect to host localhost:8000 ssl:default [Too many open files]" if request.prompt in self.fail_prompts else None
        return types.SimpleNamespace(error=err)


def _reqs(tag: str, n: int) -> List[Any]:
    from src.inference.engine import InferenceRequest

    return [InferenceRequest(prompt=f"{tag}{i}", max_tokens=4, temperature=0.0, top_p=1.0,
                             request_id=f"{tag}{i}") for i in range(n)]


runner = _load("run_experiment_s0f88_b3", SCRIPTS / "3_run" / "run_experiment.py")


def test_the_preload_runs_first_and_only_the_measured_arrivals_are_kept_u3() -> None:
    engine = _AsyncEngine()
    stamps: List[int] = []
    kept, report = runner.dispatch_open_loop(
        engine, _reqs("m", 5), rate_qps=400.0, seed=7, n_arrivals=5,
        on_first_send=lambda: stamps.append(len(engine.sent)), preload_requests=_reqs("p", 12),
    )
    assert engine.sent[:12] == [f"p{i}" for i in range(12)]   # every pre-load request once, first
    assert sorted(engine.sent[12:]) == [f"m{i}" for i in range(5)]
    assert stamps == [12]                                       # the window opens at the first MEASURED send
    assert [r.index for r in kept] == [0, 1, 2, 3, 4] and len(report.records) == 5
    assert report.preload["n_requests"] == 12 and report.preload["n_completed"] == 12
    assert report.to_manifest()["preload"]["seed"] == 8
    # the measured schedule is the plan's own (same seed, no pre-load)
    _, plain = runner.dispatch_open_loop(_AsyncEngine(), _reqs("m", 5), rate_qps=400.0, seed=7, n_arrivals=5)
    assert report.schedule.offsets_s == plain.schedule.offsets_s
    assert [round(r.scheduled_offset_s, 9) for r in kept] == [round(o, 9) for o in plain.schedule.offsets_s]


def test_a_preload_connect_error_refuses_the_window_u3() -> None:
    with pytest.raises(lg.LoadGeneratorError, match="pre-load"):
        runner.dispatch_open_loop(_AsyncEngine(fail_prompts=("p3",)), _reqs("m", 3), rate_qps=400.0,
                                  seed=1, n_arrivals=3, preload_requests=_reqs("p", 6))


def test_the_preload_pool_is_distinct_disjoint_and_capped_u3() -> None:
    from src.data.loader import CAGExample

    cands = [CAGExample(id=f"c{i}", question=f"q{i}?", context=[f"paper {i}"], answer="a", metadata={})
             for i in range(40)]
    measured = {"c0", "c1"}
    keys = runner.warmup_context_keys([c for c in cands if c.id in measured])
    pool = runner.draw_preload_pool(cands, measured_ids=measured, measured_context_keys=keys,
                                    n_max=25, seed=42, trial=1)
    ids = [c.id for c in pool]
    assert len(ids) == 25 == len(set(ids)) and not (set(ids) & measured)
    short = runner.draw_preload_pool(cands, measured_ids=measured, measured_context_keys=keys,
                                     n_max=500, seed=42, trial=1)
    assert len(short) == 38  # every disjoint candidate once, never replayed to reach 500
    again = runner.draw_preload_pool(cands, measured_ids=measured, measured_context_keys=keys,
                                     n_max=25, seed=42, trial=1)
    assert [c.id for c in again] == ids  # deterministic


def test_rehearsal_pressure_cells_carry_the_preload_and_100_arrivals_u3(tmp_path: Path) -> None:
    floor = tmp_path / "ft.json"
    floor.write_text(json.dumps(_floor_table_doc()), encoding="utf-8")
    cal = _write_calibrations(tmp_path / "cal", ("vllm", "sglang"))
    manifests = {
        "squad_v2": REPO_ROOT / "data/manifests/squad_v2_50x3_seed42.json",
        "qasper": REPO_ROOT / "data/manifests/qasper_100x3_seed42.json",
    }
    plan = rc.build_plan("a", rc.load_floor_table(floor), window_duration_s=300.0, calibrations=cal,
                         rung_calibrations=_rungs_for(cal, "a"), query_manifests=manifests, rehearsal_n=50)
    for step in plan["steps"]:
        if step["kind"] == "cell":
            pressure = step["family"] in ("F2", "F3")
            argv = step["argv"]
            assert ("--open-loop-preload-s" in argv) is pressure
            if pressure:
                assert argv[argv.index("--open-loop-preload-s") + 1] == "30"
                assert step["num_queries"] == 100 and argv[argv.index("--arrival-count") + 1] == "100"
            else:
                assert step["num_queries"] == 50
        if step["kind"] == "dry_window":
            assert "--open-loop-preload-s" not in step["argv"]
    rc.load_plan(_dump(tmp_path, plan, "plan.json"))
    cell = next(s for s in plan["steps"] if s["kind"] == "cell" and s["family"] == "F2")
    at = cell["argv"].index("--open-loop-preload-s")
    del cell["argv"][at:at + 2]
    with pytest.raises(rc.RunError, match="open-loop-preload-s"):
        rc.load_plan(_dump(tmp_path, plan, "no_preload.json"))


# --------------------------------------------------------------------------- #
# Audit: U4 relaunch skip; KV ceiling scope; relief verdict; dry-window pin
# --------------------------------------------------------------------------- #


def test_a_relaunch_whose_cells_are_all_skipped_is_not_started_u4(tmp_path, floor_table, stub, monkeypatch, capsys):
    """S0 attempt 4: four engines started (about 731 s) for configurations
    whose every cell a failed dry window had already skipped."""
    plan = _stub_plan(_pressure_grid(), floor_table, stub.cmd)
    relaunches = [s for s in plan["steps"] if s["kind"] == "relaunch"]
    monkeypatch.setenv("STUB_REGIME_LABEL", "UNPRESSURED")  # prefix OFF: the dry window fails
    assert rc.run_plan(plan, _run_root(tmp_path)) == rc.EXIT_DRY_WINDOW_FAILED
    started = [c for c in stub.calls() if "restart" in c["argv"]]
    assert len(started) == len(relaunches) - 1  # the class's second rung is never launched
    assert "relaunch skipped" in capsys.readouterr().out


def _plan_with_ceilings(tmp_path: Path) -> Dict[str, Any]:
    floor = tmp_path / "ft.json"
    floor.write_text(json.dumps(_floor_table_doc()), encoding="utf-8")
    cal = _write_calibrations(tmp_path / "cal", ("vllm", "sglang"))
    rungs = _rungs_for(cal, "a")
    for path in rungs.values():
        doc = json.loads(path.read_text(encoding="utf-8"))
        for table in [doc["rungs"]] + [c["rungs"] for c in doc["classes"].values()]:
            for rec in table.values():
                rec["kv_ceiling"] = 0.86
        path.write_text(json.dumps(doc), encoding="utf-8")
    return rc.build_plan("a", rc.load_floor_table(floor), window_duration_s=300.0,
                         calibrations=cal, rung_calibrations=rungs)


def test_the_ceiling_is_never_pinned_on_fp8_or_connector_configurations(tmp_path: Path) -> None:
    """The class ceiling is measured on a bf16 pool without a connector; an
    fp8 pool holds about twice the tokens per byte, LMCache offloads."""
    plan = _plan_with_ceilings(tmp_path)
    levered = [s for s in _cells(plan)
               if s["family"] in ("F2", "F3") and s["blocked_on"] is None
               and (rc.ARM_KV_DTYPE.get(s["cellspec"]["arm"]) or rc.ARM_CONNECTOR.get(s["cellspec"]["arm"]))]
    assert levered
    for s in levered:
        assert rc.KV_CEILING_ENV not in s["env"] and s["lambda_star_source"]["kv_ceiling"] is None
    plain = [s for s in _cells(plan) if s["family"] in ("F2", "F3") and s["blocked_on"] is None
             and s["cellspec"]["budget_r"] <= 1.0 and s["rate_basis"] == rc.LAMBDA_BASIS_RUNG]
    assert plain and all(s["env"].get(rc.KV_CEILING_ENV) == "0.86" for s in plain)
    rc.load_plan(_dump(tmp_path, plan, "plan.json"))
    dry = next(s for s in plan["steps"] if s["kind"] == "dry_window" and s["env"].get(rc.KV_CEILING_ENV))
    dry["env"][rc.KV_CEILING_ENV] = "0.5"  # review L3 on dry windows
    with pytest.raises(rc.RunError, match="dry window carries CAGE_KV_CEILING"):
        rc.load_plan(_dump(tmp_path, plan, "tampered.json"))


def test_a_prefix_relief_verdict_is_persisted_in_the_dry_sidecar(tmp_path, floor_table, stub, monkeypatch):
    plan = _stub_plan(_prefix_on_grid(), floor_table, stub.cmd)
    monkeypatch.setenv("STUB_REGIME_LABEL", "UNPRESSURED")
    root = _run_root(tmp_path)
    assert rc.run_plan(plan, root) == 0
    (dry,) = [s for s in plan["steps"] if s["kind"] == "dry_window"]
    side = json.loads(rc._dry_sidecar_path(rc.dry_window_root(root, plan), dry, 1).read_text(encoding="utf-8"))
    assert side["label"] == "UNPRESSURED" and side["verdict"] == "prefix-relief"


# --------------------------------------------------------------------------- #
# U6 (owner decision 2026-10-12, "Prepare before the clock"): a closed-loop
# window's preparation runs before its first timed send
# --------------------------------------------------------------------------- #


def test_closed_loop_preparation_runs_before_the_first_measured_send_u6(tmp_path, monkeypatch):
    """S0 attempt 4: a closed-loop rerank F1 window spent 6 percent of its
    span serving; retrieval ran between the timed sends."""
    import test_run_experiment_campaign_layout as layout

    events: List[tuple] = []
    real_format = layout.runner.format_qa_messages

    def recording_format(question, contexts, *a, **k):
        events.append(("prep", question))
        return real_format(question, contexts, *a, **k)

    class _Recording(layout.StubEngine):
        def generate(self, request, stream=False):
            events.append(("send", str(request.request_id)))
            return super().generate(request, stream=stream)

    monkeypatch.setattr(layout.runner, "format_qa_messages", recording_format)
    layout._run_cell(monkeypatch, tmp_path / "camp" / "a" / layout.RUN_ID, "no_cache", "squad_v2",
                     ttft_base=100.0, num_trials=1,
                     engine_factory=lambda model: _Recording(model, "no_cache", 100.0))
    measured_questions = {f"What is fact {i} of squad_v2?" for i in range(layout.N_QUERIES)}
    first_measured_send = next(i for i, e in enumerate(events) if e[0] == "send" and "warmup" not in e[1])
    measured_preps = [i for i, e in enumerate(events) if e[0] == "prep" and e[1] in measured_questions]
    assert measured_preps and max(measured_preps) < first_measured_send


# --------------------------------------------------------------------------- #
# U1: the window tail loads neither vllm nor transformers
# --------------------------------------------------------------------------- #


def test_the_window_tail_imports_neither_vllm_nor_transformers_u1() -> None:
    """S0 attempt 4: every vLLM window idled 25 to 32 s after its last
    completion importing vllm (for a version string) and transformers (for
    a config read)."""
    import subprocess

    probe = (
        "import sys, importlib.util; sys.path.insert(0, %r)\n"
        "spec = importlib.util.spec_from_file_location('rexp', %r)\n"
        "m = importlib.util.module_from_spec(spec); sys.modules['rexp'] = m; spec.loader.exec_module(m)\n"
        "meta = m.capture_backend_metadata(api_base='http://127.0.0.1:9', backend='vllm', model_name='Qwen/Qwen3-14B', use_offline=True)\n"
        "from src.evaluation.compression import analytical_kv_footprint\n"
        "fp = analytical_kv_footprint('Qwen/Qwen3-14B', 1000)\n"
        "print('vllm' in sys.modules, 'transformers' in sys.modules, fp and fp['kv_bytes_per_token'])\n"
    ) % (str(REPO_ROOT), str(SCRIPTS / "3_run" / "run_experiment.py"))
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True,
                         env={**__import__("os").environ, "HF_HUB_OFFLINE": "1"}).stdout.split()
    assert out[:2] == ["False", "False"]
    if out[2] == "None":
        pytest.skip("Qwen/Qwen3-14B config.json is not in the local HF cache")
    from src.orchestration.cache_budget import MODEL_KV

    assert float(out[2]) == MODEL_KV["qwen3-14b"].kv_bytes_per_token


# --------------------------------------------------------------------------- #
# Q-3: served rows with no answer are named per window and counted
# --------------------------------------------------------------------------- #


def _empty_rows_tree(tmp_path: Path, n_empty: int) -> tuple:
    import test_verify_results_v2 as tv

    run_dir = tv._build_tree(tmp_path)
    wdir = tv._first_window(run_dir)
    rows = tv._read_jsonl(wdir / "requests.jsonl")
    for row in rows[:n_empty]:
        row.update(empty_generation=True, ok=False)
    tv._write_jsonl(wdir / "requests.jsonl", rows)
    tv._seal(run_dir)
    return tv, tv.vr.verify_run(run_dir)


def test_a_window_whose_served_rows_are_blank_fails_and_one_blank_row_warns_q3(tmp_path: Path) -> None:
    """S0 attempt 4: 52 served HF oracle rows answered one space and the
    gate said nothing."""
    import test_verify_results_v2 as tv

    _, report = _empty_rows_tree(tmp_path / "all", tv.N_ROWS)
    assert report["ok"] is False
    (fail,) = tv._findings(report, "FAIL", "empty-output")
    assert f"{tv.N_ROWS} of {tv.N_ROWS} served row(s)" in fail["detail"] and "Q-3" in fail["detail"]
    tv, report = _empty_rows_tree(tmp_path / "one", 1)
    assert report["ok"] is True
    (warn,) = tv._findings(report, "WARN", "empty-output")
    assert f"1 of {tv.N_ROWS} served row(s)" in warn["detail"]
    tv, report = _empty_rows_tree(tmp_path / "none", 0)
    assert not [f for f in report["findings"] if f["check"] == "empty-output"]


def test_error_rows_are_outside_the_blank_answer_check_q3(tmp_path: Path) -> None:
    import test_verify_results_v2 as tv

    run_dir = tv._build_tree(tmp_path)
    wdir = tv._first_window(run_dir)
    rows = tv._read_jsonl(wdir / "requests.jsonl")
    rows[0].update(ok=False, error="engine_error: 500", empty_generation=True,
                   finish_reason="error", num_tokens=0, tpot_ms=None)
    tv._write_jsonl(wdir / "requests.jsonl", rows)
    tv._seal(run_dir)
    report = tv.vr.verify_run(run_dir)
    assert not [f for f in report["findings"] if f["check"] == "empty-output"]


def test_the_window_summary_counts_blank_answers_q3(tmp_path, monkeypatch):
    import test_run_experiment_campaign_layout as layout

    class _Blank(layout.StubEngine):
        def _respond(self, request):
            response = super()._respond(request)
            rid = str(request.request_id or "")
            if "warmup" not in rid and (rid.endswith("q001") or rid.endswith("q002")):
                response.generated_text = " " if rid.endswith("q001") else "  "
            return response

    root = tmp_path / "camp" / "a" / layout.RUN_ID
    layout._run_cell(monkeypatch, root, "no_cache", "squad_v2", ttft_base=100.0, num_trials=1,
                     engine_factory=lambda model: _Blank(model, "no_cache", 100.0))
    (wdir,) = layout._window_dirs(root, "no_cache", "squad_v2")
    meta = json.loads((wdir / "metrics.json").read_text(encoding="utf-8"))
    rows = layout._read_jsonl(wdir / "requests.jsonl")
    assert meta["empty_generations"] == 2 == sum(1 for r in rows if r.get("empty_generation"))


# --------------------------------------------------------------------------- #
# B-4 audit: the closed-loop (requests/urllib3) connect failures are named
# --------------------------------------------------------------------------- #


def test_the_closed_loop_connect_failure_texts_are_connect_errors_b4() -> None:
    import requests
    import urllib3

    # a real refused connection: nothing listens on the discard port
    try:
        requests.post("http://127.0.0.1:9/v1/completions", json={}, timeout=2)
    except requests.exceptions.ConnectionError as exc:
        refused = f"{type(exc).__name__}: {exc}"
    else:  # pragma: no cover - something answers on port 9
        pytest.skip("a server answers on 127.0.0.1:9")
    assert "Failed to establish a new connection" in refused and lg.is_connect_error(refused)
    # the library's own text for a connect timeout and for a read timeout
    pool = urllib3.HTTPConnectionPool("10.255.255.1", 80)
    connect_timeout = urllib3.exceptions.MaxRetryError(
        pool, "/v1/completions", urllib3.exceptions.ConnectTimeoutError(None, "timed out"))
    read_timeout = urllib3.exceptions.MaxRetryError(
        pool, "/v1/completions", urllib3.exceptions.ReadTimeoutError(pool, "/v1/completions", "timed out"))
    assert lg.is_connect_error(f"ConnectTimeout: {connect_timeout}")
    # a read timeout reached the engine: it is an engine result, never a connect failure
    assert not lg.is_connect_error(f"ReadTimeout: {read_timeout}")
    assert not lg.is_connect_error("engine_error: HTTP 500 Internal Server Error")
    # review L8: a served 502 whose body quotes an upstream's connect failure
    # reached the serving system (a PD router): a served result, not a client fault
    assert not lg.is_connect_error(
        "502 Server Error: Bad Gateway for url: http://localhost:8000/v1/completions: "
        "prefill upstream: Failed to establish a new connection: [Errno 111] Connection refused")
    assert not lg.is_connect_error("engine_error: Caused by ConnectTimeoutError on the decode instance")


# --------------------------------------------------------------------------- #
# Q-5: an open-loop window reports its served rate, never a summed-latency QPS
# --------------------------------------------------------------------------- #


def test_served_tokens_per_second_spans_first_send_to_last_completion_q5() -> None:
    sched = lg.ArrivalSchedule(rate_qps=2.0, seed=1, distribution="poisson", offsets_s=(0.0, 0.5))
    recs = []
    for i, (send, done, tokens) in enumerate([(0.0, 1.0, 10), (0.5, 2.0, 30)]):
        rec = lg.RequestRecord(index=i, scheduled_offset_s=send, scheduled_ts=send)
        rec.actual_send_ts, rec.completion_ts = send, done
        rec.result = types.SimpleNamespace(error=None, num_tokens=tokens)
        recs.append(rec)
    report = lg.DispatchReport(schedule=sched, run_start_ts=0.0, records=recs)
    assert report.served_rps == pytest.approx(1.0) and report.served_tokens_per_second == pytest.approx(20.0)
    assert report.to_manifest()["served_tokens_per_second"] == pytest.approx(20.0)
    recs[1].result = types.SimpleNamespace(error=None, num_tokens=None)
    assert report.served_tokens_per_second is None  # a missing count is unknown, never 0


def test_an_open_loop_window_records_no_summed_latency_qps_q5(tmp_path, monkeypatch, capsys):
    import test_run_experiment_campaign_layout as layout

    root = tmp_path / "camp" / "a" / layout.RUN_ID
    layout._run_cell(monkeypatch, root, "no_cache", "squad_v2", ttft_base=200.0, num_trials=1,
                     engine_factory=layout._async_engine, extra_argv=layout.OPEN_LOOP_ARGV)
    (wdir,) = layout._window_dirs(root, "no_cache", "squad_v2")
    meta = json.loads((wdir / "metrics.json").read_text(encoding="utf-8"))
    assert meta["performance"]["queries_per_second"] is None
    assert meta["performance"]["tokens_per_second"] is None
    open_loop = meta["workload"]["open_loop"]
    assert open_loop["served_rps"] > 0 and open_loop["served_tokens_per_second"] > 0
    out = capsys.readouterr().out
    assert "Throughput (served, open loop):" in out and " QPS," not in out


# --------------------------------------------------------------------------- #
# L5: a retrieval arm's warm-ups are judged on the prompt it sends
# --------------------------------------------------------------------------- #


class _WordTok:
    """Counts words: a chat render is the words of every message."""

    def apply_chat_template(self, messages, **kw):
        return [0] * sum(len(str(m.get("content", "")).split()) for m in messages)

    def encode(self, text, add_special_tokens=True):
        return [0] * len(text.split())


def _l5_loader(measured_words: int):
    from src.data.loader import CAGExample

    class _Loader:
        def __init__(self, dataset):
            self.dataset = dataset

        def load(self, max_examples=None):
            n = 40 if max_examples is None else min(40, max_examples)
            out = []
            for i in range(n):
                # the first two are measured (they form the BM25 corpus); every
                # candidate after them carries a 600-word paper
                words = measured_words if i < 2 else 600
                text = f"topic{i} fact{i} " + " ".join(f"w{i}x{k}" for k in range(words))
                # every question names topic0, so BM25 retrieves a measured paper
                out.append(CAGExample(id=f"{self.dataset}-q{i:03d}", question=f"What is topic0 and fact{i}?",
                                      context=[text], answer=f"fact{i}", metadata={}))
            return out

    return _Loader


def _l5_cell(tmp_path, monkeypatch, sent: List[str], *, measured_words: int, extra: tuple = ()):
    import test_run_experiment_campaign_layout as layout

    class _Recording(layout._AsyncCampaignEngine):
        def generate(self, request, stream=False):
            sent.append(str(request.request_id))
            return super().generate(request, stream=stream)

        async def async_stream_generate(self, request, *, on_first_token=None):
            sent.append(str(request.request_id))
            return await super().async_stream_generate(request, on_first_token=on_first_token)

    monkeypatch.setattr(layout.runner, "ruler_token_counter", lambda model: (None, "words", _WordTok()))
    root = tmp_path / "camp" / "a" / layout.RUN_ID
    layout._run_cell(
        monkeypatch, root, "rag", "squad_v2", ttft_base=100.0, num_trials=1,
        loader_factory=_l5_loader(measured_words),
        engine_factory=lambda model: _Recording(model, "rag", 100.0),
        extra_argv=("--retriever", "bm25", "--top-k", "1", "--ir-index-dir", str(tmp_path / "ir"),
                    "--num-queries", "2", "--warmup-pool-queries", "2", "--request-cap-tokens", "300", *extra),
    )
    (wdir,) = sorted(root.glob("cells/*/window_squad_v2-*"))
    return json.loads((wdir / "metrics.json").read_text(encoding="utf-8"))


def test_a_retrieval_arm_draws_warmups_whose_raw_paper_exceeds_the_cap_l5(tmp_path, monkeypatch):
    """Every candidate's raw paper (600 words) is over the 300-token cap, but
    the arm sends one short retrieved passage: the raw-context filter refused
    the whole pool; now the prepared prompts are judged and they fit."""
    sent: List[str] = []
    meta = _l5_cell(tmp_path, monkeypatch, sent, measured_words=5)
    cap = meta["warmup_pool"]["length_cap"]
    assert cap["retrieval_arm"] is True and cap["excluded_by_length"] == 0
    assert 0 < cap["prepared_tokens_max"] < 300
    assert sum("warmup_pool" in rid for rid in sent) == 2


def test_a_retrieval_arm_served_gold_context_keeps_the_raw_context_filter_l5(tmp_path, monkeypatch, capsys):
    """Review L5: under --context-source gold the rag arm sends the raw paper
    (600 words, over the 300-token cap), so the draw judges the raw context
    and refuses before any send, as on every gold-context arm."""
    sent: List[str] = []
    with pytest.raises(SystemExit) as exit_info:
        _l5_cell(tmp_path, monkeypatch, sent, measured_words=5, extra=("--context-source", "gold"))
    assert exit_info.value.code == 1 and sent == []
    assert "warm-up pool needs 2 examples whose served request fits" in capsys.readouterr().out


def test_a_prepared_warmup_over_the_cap_is_excluded_and_the_cell_refuses_only_when_no_pool_fits_l5(
    tmp_path, monkeypatch, capsys
):
    """Every retrieved passage is over the cap (the one measured paper is 400
    words): each drawn warm-up is excluded, the refill finds nothing, and
    only then does the cell refuse, before any request reaches the server
    (review 2026-10-13 item 25: exclusion first, refusal last)."""
    sent: List[str] = []
    with pytest.raises(SystemExit) as exit_info:
        _l5_cell(tmp_path, monkeypatch, sent, measured_words=400)
    assert exit_info.value.code == 1 and sent == []
    out = capsys.readouterr().out
    assert "excluded and redrawn, never a refusal" in out and "> --request-cap-tokens 300" in out
    assert "whose served request fits the cell's request cap but only 0 of" in out


def test_a_drawn_warmup_over_the_cap_is_replaced_and_the_cell_runs_l5(tmp_path, monkeypatch):
    """Review 2026-10-13 item 25: measured paper 1 is long and most candidates
    retrieve it; a drawn warm-up over the cap is replaced from the candidates
    that retrieve the short paper 0, the pool stays at its registered size, and
    the cell runs (it refused before, and under decision 3 that exit 1 would
    have blocked the run)."""
    import test_run_experiment_campaign_layout as layout
    from src.data.loader import CAGExample

    class _Loader:
        def __init__(self, dataset):
            self.dataset = dataset

        def load(self, max_examples=None):
            n = 40 if max_examples is None else min(40, max_examples)
            out = []
            for i in range(n):
                words = 400 if i == 1 else 5  # measured paper 1 is over the cap, paper 0 fits
                text = f"topic{i} fact{i} " + " ".join(f"w{i}x{k}" for k in range(words))
                topic = "topic0" if i % 4 == 2 else "topic1"  # 2 of the 8 slate candidates retrieve the short paper
                if i < 2:
                    topic = f"topic{i}"
                out.append(CAGExample(id=f"{self.dataset}-q{i:03d}", question=f"What is {topic} and fact{i}?",
                                      context=[text], answer=f"fact{i}", metadata={}))
            return out

    sent: List[str] = []

    class _Recording(layout._AsyncCampaignEngine):
        def generate(self, request, stream=False):
            sent.append(str(request.request_id))
            return super().generate(request, stream=stream)

        async def async_stream_generate(self, request, *, on_first_token=None):
            sent.append(str(request.request_id))
            return await super().async_stream_generate(request, on_first_token=on_first_token)

    monkeypatch.setattr(layout.runner, "ruler_token_counter", lambda model: (None, "words", _WordTok()))
    root = tmp_path / "camp" / "a" / layout.RUN_ID
    layout._run_cell(
        monkeypatch, root, "rag", "squad_v2", ttft_base=100.0, num_trials=1, loader_factory=_Loader,
        engine_factory=lambda model: _Recording(model, "rag", 100.0),
        extra_argv=("--retriever", "bm25", "--top-k", "1", "--ir-index-dir", str(tmp_path / "ir"),
                    "--num-queries", "2", "--warmup-pool-queries", "2", "--request-cap-tokens", "300"),
    )
    (wdir,) = sorted(root.glob("cells/*/window_squad_v2-*"))
    cap = json.loads((wdir / "metrics.json").read_text(encoding="utf-8"))["warmup_pool"]["length_cap"]
    served_pool = [rid for rid in sent if "warmup_pool" in rid]
    assert len(served_pool) == 2  # the registered pool size, after the refill
    assert cap["excluded_by_prepared_length"] >= 1  # the seeded draw hit at least one long passage
    assert cap["replaced_by_prepared_length"] >= 1 and cap["prepared_tokens_max"] < 300
    fitting = {f"squad_v2-q{i:03d}" for i in range(2, 40) if i % 4 == 2}
    assert all(rid.split("__warmup_pool")[0] in fitting for rid in served_pool)


def test_preload_candidates_over_the_cap_are_excluded_and_the_window_runs_shorter_l5(tmp_path, monkeypatch, capsys):
    """The pre-load site (review 2026-10-13 item 25): the two warm-up questions
    retrieve the short paper and fit; every other candidate retrieves the long
    one and is excluded, so the pre-load keeps the two that fit and the window
    runs (it refused on the first over-cap candidate before)."""
    import test_run_experiment_campaign_layout as layout
    from src.data.loader import CAGExample

    def examples(short_ids):
        out = []
        for i in range(40):
            words = 400 if i == 0 else 5  # measured q000 is the long paper, q001 the short one
            text = f"topic{i} fact{i} " + " ".join(f"w{i}x{k}" for k in range(words))
            topic = "topic1" if (i == 1 or f"squad_v2-q{i:03d}" in short_ids) else "topic0"
            out.append(CAGExample(id=f"squad_v2-q{i:03d}", question=f"What is {topic} and fact{i}?",
                                  context=[text], answer=f"fact{i}", metadata={}))
        return out

    # the warm-up draw ignores question text on a retrieval arm (fits=None);
    # without a manifest the candidates are the slate past the measured head,
    # num_queries + pool x WARMUP_POOL_CANDIDATE_MULTIPLIER examples
    base = examples(set())
    slate = 2 + 2 * runner.WARMUP_POOL_CANDIDATE_MULTIPLIER
    warm = runner.draw_warmup_pool(base[2:slate], measured_ids={e.id for e in base[:2]},
                                   measured_context_keys=runner.warmup_context_keys(base[:2]),
                                   n=2, seed=7, trial=1)
    final = examples({e.id for e in warm})

    class _Loader:
        def __init__(self, dataset):
            pass

        def load(self, max_examples=None):
            return final[: max_examples or 40]

    sent: List[str] = []

    class _Recording(layout._AsyncCampaignEngine):
        def generate(self, request, stream=False):
            sent.append(str(request.request_id))
            return super().generate(request, stream=stream)

        async def async_stream_generate(self, request, *, on_first_token=None):
            sent.append(str(request.request_id))
            return await super().async_stream_generate(request, on_first_token=on_first_token)

    monkeypatch.setattr(layout.runner, "ruler_token_counter", lambda model: (None, "words", _WordTok()))
    root = tmp_path / "camp" / "a" / layout.RUN_ID
    layout._run_cell(
        monkeypatch, root, "rag", "squad_v2", ttft_base=100.0, num_trials=1, loader_factory=_Loader,
        engine_factory=lambda model: _Recording(model, "rag", 100.0),
        extra_argv=(*layout.OPEN_LOOP_ARGV, "--arrival-count", "2", "--open-loop-preload-s", "30",
                    "--retriever", "bm25", "--top-k", "1", "--ir-index-dir", str(tmp_path / "ir"),
                    "--num-queries", "2", "--warmup-pool-queries", "2", "--request-cap-tokens", "300"),
    )
    out = capsys.readouterr().out
    # the six long-paper candidates are excluded (printed once each); the two
    # that fit are the warm-up questions, which both pools draw from
    assert out.count("[length-cap] excluded squad_v2-q0") == 6 and "excluded and redrawn" in out
    assert "pre-load 2 distinct requests" in out and "no pre-load request could be drawn" not in out
    (wdir,) = sorted(root.glob("cells/*/window_squad_v2-*"))
    cap = json.loads((wdir / "metrics.json").read_text(encoding="utf-8"))["warmup_pool"]["length_cap"]
    assert cap["excluded_by_prepared_length"] == 6 and cap["replaced_by_prepared_length"] == 0
    assert all(f"{e.id}__warmup_pool{k}" in sent for k, e in enumerate(warm))  # the warm-ups were served
    assert any(rid.startswith("squad_v2-q00") and "__warmup" not in rid for rid in sent)  # and the window ran


# --------------------------------------------------------------------------- #
# Q-8: a silent sampler no longer reads as covered
# --------------------------------------------------------------------------- #


def _ticks(start: float, end: float, step: float, *, skip: tuple = ()) -> Any:
    import pandas as pd

    ts, t = [], start
    while t < end:
        if not any(a <= t < b for a, b in skip):
            ts.append(round(t, 6))
        t += step
    return pd.DataFrame({"ts_s": ts, "kv_cache_usage": [0.5] * len(ts),
                         "preemptions_total": [0] * len(ts), "waiting": [1] * len(ts)})


def test_a_two_second_window_at_the_quarter_second_tick_certifies_q8() -> None:
    """S0 attempt 4's small-class windows spanned 2 to 3 s; the 0.25 s
    sampler (R14) covers them under the hold rule."""
    from src.analysis.regime_inputs import MAX_SAMPLE_HOLD_S, compute_window_regime_inputs

    win = compute_window_regime_inputs(_ticks(100.1, 102.1, 0.25), 100.0, 102.0, max_hold_s=MAX_SAMPLE_HOLD_S)
    assert win.n_samples == 8 and win.coverage == pytest.approx(0.95)
    assert MAX_SAMPLE_HOLD_S == 2.0


def test_a_mid_window_blackout_refuses_and_leaves_the_average_q8() -> None:
    """S0 attempt 4: the sampler went silent for 7.6 s of a 14.3 s window
    (its scrapes failed, no sample recorded) and the window certified."""
    from src.analysis.regime_inputs import MAX_SAMPLE_HOLD_S, RegimeInputError, compute_window_regime_inputs

    stalled = _ticks(0.0, 100.0, 0.25, skip=((40.0, 70.0),))
    old = compute_window_regime_inputs(stalled, 0.0, 100.0)  # the pre-Q-8 rule
    assert old.coverage == pytest.approx(1.0)
    # last tick before the stall at 39.75 s: a 30.25 s hold, 28.25 s of it unobserved
    with pytest.raises(RegimeInputError, match=r"coverage 0\.718 .*longest hold 30\.25 s"):
        compute_window_regime_inputs(stalled, 0.0, 100.0, max_hold_s=MAX_SAMPLE_HOLD_S)
    # under the threshold the window certifies and the stall's excess leaves
    # the time average: 0.9 before the stall, 0.1 after, the stall held 0.9
    stalled["kv_cache_usage"] = [0.9 if t < 50 else 0.1 for t in stalled["ts_s"]]
    win = compute_window_regime_inputs(stalled, 0.0, 100.0, max_hold_s=MAX_SAMPLE_HOLD_S, min_coverage=0.7)
    observed_high = 39.75 + 2.0  # 0.25 s ticks to 39.75 s, then the hold capped at 2 s
    expected = (0.9 * observed_high + 0.1 * 30.0) / (observed_high + 30.0)
    assert win.coverage == pytest.approx(0.7175) and win.rho_kv_time_avg == pytest.approx(expected)
    assert old.rho_kv_time_avg == pytest.approx(0.5)  # the old fixture value


def test_the_regime_inputs_record_the_longest_hold_and_the_clipped_seconds_l9() -> None:
    """Review L9: a window that certifies under the hold rule still shows how
    much the sampler missed: the 30.25 s stall and its 28.25 s excess."""
    from src.analysis.regime_inputs import (
        MAX_SAMPLE_HOLD_S,
        compute_pd_window_regime_inputs,
        compute_window_regime_inputs,
    )

    stalled = _ticks(0.0, 100.0, 0.25, skip=((40.0, 70.0),))
    win = compute_window_regime_inputs(stalled, 0.0, 100.0, max_hold_s=MAX_SAMPLE_HOLD_S, min_coverage=0.7)
    assert win.longest_hold_s == pytest.approx(30.25) and win.hold_excess_s == pytest.approx(28.25)
    assert win.coverage == pytest.approx(1.0 - win.hold_excess_s / 100.0)  # no pre-first-sample gap here
    flat = win.to_flat_dict()
    assert flat["longest_hold_s"] == win.longest_hold_s and flat["hold_excess_s"] == win.hold_excess_s
    clean = compute_window_regime_inputs(_ticks(0.0, 100.0, 0.25), 0.0, 100.0, max_hold_s=MAX_SAMPLE_HOLD_S)
    assert clean.longest_hold_s == pytest.approx(0.25) and clean.hold_excess_s == 0.0
    no_rule = compute_window_regime_inputs(stalled, 0.0, 100.0)
    assert no_rule.longest_hold_s == pytest.approx(30.25) and no_rule.hold_excess_s == 0.0
    # pd: the worst role's gaps, as the pooled coverage is the worst role's
    pd_win = compute_pd_window_regime_inputs(
        {"prefill": _ticks(0.0, 100.0, 0.25), "decode": stalled}, 0.0, 100.0,
        budgets_by_role={"prefill": 1, "decode": 1}, max_hold_s=MAX_SAMPLE_HOLD_S, min_coverage=0.7)
    assert pd_win.longest_hold_s == pytest.approx(30.25) and pd_win.hold_excess_s == pytest.approx(28.25)
    assert pd_win.to_flat_dict()["hold_excess_s"] == pd_win.hold_excess_s


def test_the_campaign_session_applies_the_hold_rule_and_a_refusal_carries_the_clause_keys_q8(tmp_path, monkeypatch):
    import test_run_experiment_campaign_layout as layout

    root = tmp_path / "camp" / "a" / layout.RUN_ID
    layout._run_cell(monkeypatch, root, "no_cache", "squad_v2", ttft_base=100.0, num_trials=1)
    (wdir,) = sorted(root.glob("cells/*/window_squad_v2-*"))
    doc = json.loads((wdir / "regime.json").read_text(encoding="utf-8"))
    assert doc["max_hold_s"] == 2.0
    # the stub engine serves no /metrics: the empty series is refused, and the
    # refused document carries the certified one's clause keys
    assert doc["label"] == "UNKNOWN_TELEMETRY"
    assert doc["kv_ceiling"] is None and doc["clause_a_rule"] is None


# --------------------------------------------------------------------------- #
# Q-10 (the NLI part): no premise window relies on the pipeline's truncation
# --------------------------------------------------------------------------- #


class _NliWordTok:
    """Counts words, the way the NLI windower calls a tokenizer."""

    def __call__(self, texts, add_special_tokens=False, verbose=True):
        return {"input_ids": [[0] * len(t.split()) for t in texts]}


def _premise_evaluator():
    from src.evaluation.quality import QualityEvaluator

    ev = QualityEvaluator.__new__(QualityEvaluator)
    ev.use_nli = True
    ev._instrument_unavailable = {}
    ev._nli_model = types.SimpleNamespace(tokenizer=_NliWordTok())
    return ev


def test_an_over_budget_sentence_is_split_before_the_nli_sees_it_q10() -> None:
    """A terminator-free 900-word sentence came back whole and the pipeline
    cut it at 512 tokens (quality.py B2 NOTE); S1's key-value contexts have
    that shape."""
    ev = _premise_evaluator()
    budget = ev.NLI_PREMISE_WINDOW_TOKENS
    blob = " ".join(f"k{i}=v{i}" for i in range(900))
    doc = blob + ". The needle is 42."
    windows = ev._split_premise_windows(doc)
    assert len(windows) > 2 and all(len(w.split()) <= budget for w in windows)
    assert any("k0=v0" in w for w in windows) and any("k899=v899" in w for w in windows)
    assert any("The needle is 42." in w for w in windows)


def test_a_doc_without_an_over_budget_sentence_windows_as_before_q10() -> None:
    from src.evaluation.quality import _split_sentences

    ev = _premise_evaluator()
    doc = " ".join(f"Sentence {i} " + "word " * 60 + "ends." for i in range(20))
    sentences = _split_sentences(doc)
    counts = [len(s.split()) for s in sentences]
    expected = ev._sentence_window_split(doc, sentences, counts, ev.NLI_PREMISE_WINDOW_TOKENS)
    assert ev._split_premise_windows(doc) == expected and len(expected) > 1


# --------------------------------------------------------------------------- #
# U2 (owner decision 2026-10-12, "big enough for one window"): LMCache's CPU
# store is sized per relaunch for its largest window
# --------------------------------------------------------------------------- #


def _s0_rehearsal_plan(tmp_path: Path) -> Dict[str, Any]:
    floor = tmp_path / "ft.json"
    floor.write_text(json.dumps(_floor_table_doc()), encoding="utf-8")
    cal = _write_calibrations(tmp_path / "cal", ("vllm", "sglang"))
    manifests = {
        "squad_v2": REPO_ROOT / "data/manifests/squad_v2_50x3_seed42.json",
        "qasper": REPO_ROOT / "data/manifests/qasper_100x3_seed42.json",
    }
    return rc.build_plan("a", rc.load_floor_table(floor), window_duration_s=300.0, calibrations=cal,
                         rung_calibrations=_rungs_for(cal, "a"), query_manifests=manifests, rehearsal_n=50)


def test_every_lmcache_relaunch_carries_the_store_of_its_largest_window_u2(tmp_path: Path) -> None:
    """S0 attempt 4: LMCache ran at its 5.0 GiB default and logged 8,883
    allocation warnings, each an eviction."""
    from src.orchestration.cache_budget import MODEL_KV

    plan = _s0_rehearsal_plan(tmp_path)
    steps = plan["steps"]
    stores = [s for s in steps if s["kind"] == "relaunch" and s["connector"] == "lmcache"]
    assert stores
    for relaunch in stores:
        rec = relaunch["lmcache_store"]
        under = []
        for s in steps[relaunch["index"] + 1:]:
            if s["kind"] == "relaunch":
                break
            if s["kind"] in ("cell", "dry_window") and not s.get("blocked_on"):
                under.append(s)
        assert rec["requests"] == max(s["num_queries"] for s in under)
        assert rec["kv_bytes_per_token"] == MODEL_KV["qwen3-14b"].kv_bytes_per_token
        expect = -(-rec["requests"] * rec["seq_tokens"] * rec["kv_bytes_per_token"] * 5 // (4 * 2**30))
        assert rec["max_local_cpu_size_gib"] == expect > 5
        assert relaunch["env"][rc.LMCACHE_STORE_ENV] == str(expect)
    assert all(rc.LMCACHE_STORE_ENV not in s["env"] for s in steps
               if s["kind"] == "relaunch" and s["connector"] != "lmcache")
    rc.load_plan(_dump(tmp_path, plan, "plan.json"))
    stores[0]["env"][rc.LMCACHE_STORE_ENV] = "5"
    with pytest.raises(rc.RunError, match="LMCACHE_MAX_LOCAL_CPU_SIZE"):
        rc.load_plan(_dump(tmp_path, plan, "drifted.json"))


def test_stage_0_reports_the_largest_store_and_stage_4_checks_the_pod_ram_u2() -> None:
    manifests = {
        "squad_v2": str(REPO_ROOT / "data/manifests/squad_v2_50x3_seed42.json"),
        "qasper": str(REPO_ROOT / "data/manifests/qasper_100x3_seed42.json"),
    }
    report = rc.preflight_registration_check("a", floor_avg_seq_tokens="4779", rehearsal_n=50,
                                             query_manifests=manifests, floor_concurrency=50)
    assert report["needs_lmcache"] is True
    # 100 arrivals x 1127 tokens x 163,840 B x 1.25 = 21.5 GiB
    assert report["lmcache_store_gib"] == 22
    master = (SCRIPTS / "6_experiments" / "cage_experiment.sh").read_text(encoding="utf-8")
    assert "LMCACHE_STORE_GIB={report.get('lmcache_store_gib') or 0}" in master
    assert "state put registration.lmcache_store_gib" in master
    # review H1: the container's own memory (cgroup), never the host's /proc/meminfo
    assert "probe_pod_memory.py --need-gib ${store_gib:-0}" in master and 'grep -q "POD_MEMORY_OK"' in master


# --------------------------------------------------------------------------- #
# U10: vLLM starts without the launcher's own VLLM_* inputs
# --------------------------------------------------------------------------- #


def test_vllm_serve_never_sees_the_launcher_only_variables_u10(tmp_path: Path) -> None:
    """S0 attempt 4: every vLLM start warned about six unknown VLLM_*
    environment variables, the launcher's own inputs."""
    import subprocess
    import time

    import test_tp_flags as tf

    dump = tmp_path / "vllm_env.txt"
    stub = tmp_path / "bin"
    stub.mkdir()
    for name, body in {
        "pgrep": "#!/bin/sh\nexit 1\n", "curl": "#!/bin/sh\nexit 1\n",
        "nvidia-smi": "#!/bin/sh\nexit 1\n", "pkill": "#!/bin/sh\nexit 0\n",
        "vllm": f"#!/bin/sh\nenv > '{dump}'\nexit 0\n",
    }.items():
        (stub / name).write_text(body, encoding="utf-8")
        (stub / name).chmod(0o755)
    own = {"VLLM_GPU_MEMORY_UTILIZATION": "0.9", "VLLM_ENFORCE_EAGER": "1", "VLLM_MAX_MODEL_LEN": "8192",
           "VLLM_KV_CACHE_DTYPE": "fp8", "VLLM_KV_TRANSFER_CONFIG": '{"kv_connector":"x"}'}
    script = SCRIPTS / "2_serving" / "manage_vllm_server.sh"
    tf._run_launcher(script, stub, "start", "Qwen/Qwen3-14B", **own, VLLM_PORT="8000")
    for _ in range(100):
        if dump.is_file() and dump.stat().st_size:
            break
        time.sleep(0.05)
    seen = dict(line.split("=", 1) for line in dump.read_text(encoding="utf-8").splitlines() if "=" in line)
    assert not set(own) & set(seen) and "VLLM_START_TIMEOUT" not in seen
    assert seen.get("VLLM_PORT") == "8000" and seen.get("VLLM_SERVER_DEV_MODE") == "1"


# --------------------------------------------------------------------------- #
# U7: SGLang rows carry a cached-token count; U8: both engines prove the fp8
# format they serve
# --------------------------------------------------------------------------- #


def test_a_null_cached_block_reads_as_zero_only_once_the_report_is_verified_u7(monkeypatch) -> None:
    """SGLang 0.5.10.post1 sends prompt_tokens_details null for a request with
    no cached token, whether or not the report is on (usage_processor.py:13-15)."""
    from src.inference import sglang_adapter as sa

    adapter = sa.SGLangAdapter("Qwen/Qwen3-14B", api_base="http://127.0.0.1:30000")
    cold = {"prompt_tokens": 812, "completion_tokens": 4, "prompt_tokens_details": None}
    warm = {"prompt_tokens": 812, "completion_tokens": 4, "prompt_tokens_details": {"cached_tokens": 640}}
    assert adapter._extract_usage(cold) == (812, None, 4)  # report not verified: unknown

    class _Resp:
        def __init__(self, doc):
            self.doc = doc

        def raise_for_status(self):
            pass

        def json(self):
            return self.doc

    seen = []
    monkeypatch.setattr(sa.requests, "get", lambda url, timeout: seen.append(url) or _Resp({"enable_cache_report": True}))
    assert adapter.probe_cache_report() is True and seen == ["http://127.0.0.1:30000/server_info"]
    assert adapter._extract_usage(cold) == (812, 0, 4) and adapter._extract_usage(warm) == (812, 640, 4)
    assert adapter._extract_usage({"completion_tokens": 4}) == (None, None, 4)  # no usage: still unknown
    assert adapter.capabilities()["cached_token_absent_means_zero"] is True
    monkeypatch.setattr(sa.requests, "get", lambda url, timeout: _Resp({"enable_cache_report": False}))
    assert adapter.probe_cache_report() is False and adapter._extract_usage(cold) == (812, None, 4)
    monkeypatch.setattr(sa.requests, "get", lambda url, timeout: _Resp({"version": "x"}))
    assert adapter.probe_cache_report() is None and adapter.cache_report is False


def test_the_sglang_launcher_reports_cached_tokens_and_stage_4_checks_it_u7() -> None:
    launcher = (SCRIPTS / "2_serving" / "manage_sglang_server.sh").read_text(encoding="utf-8")
    assert "sglang_args+=( --enable-cache-report )" in launcher
    assert '[[ " $live_cmd " == *" --enable-cache-report "* ]] || dials_match=false' in launcher
    master = (SCRIPTS / "6_experiments" / "cage_experiment.sh").read_text(encoding="utf-8")
    assert "curl -fsS $api/server_info" in master and 'grep -q "CACHE_REPORT_ON" "$JOB_LOG"' in master
    assert rc.KV_DTYPE_LAUNCH_ENV == {"vllm": ("VLLM_KV_CACHE_DTYPE", "fp8"),
                                      "sglang": ("SGLANG_KV_CACHE_DTYPE", "fp8_e4m3")}


from test_kv_pool_capture_s0f24 import stub_bin  # noqa: E402,F401  (fixture re-export)


def test_a_vllm_fp8_start_must_print_the_dtype_it_serves_u8(stub_bin: Path, tmp_path: Path) -> None:
    import test_kv_pool_capture_s0f24 as kpt

    config = "(EngineCore pid=1) INFO [core.py:105] Initializing a V1 LLM engine (v0.19.1) with config: model='m', kv_cache_dtype=fp8, enable_prefix_caching=True\n"
    ok = kpt._run_launcher("manage_vllm_server.sh", stub_bin, tmp_path / "a", "start", "fake/test-model",
                           VLLM_START_TIMEOUT="10", VLLM_KV_CACHE_DTYPE="fp8",
                           CAGE_TEST_POOL_LINES=config + kpt.VLLM_019_BUDGETED_LOG,
                           CAGE_TEST_READY_WHEN_LOGGED="GPU KV cache size")
    assert ok.returncode == 0 and "KV cache dtype realized: fp8" in ok.stdout, ok.stdout + ok.stderr
    bad = kpt._run_launcher("manage_vllm_server.sh", stub_bin, tmp_path / "b", "start", "fake/test-model",
                            VLLM_START_TIMEOUT="10", VLLM_KV_CACHE_DTYPE="fp8",
                            CAGE_TEST_POOL_LINES=config.replace("fp8", "auto") + kpt.VLLM_019_BUDGETED_LOG,
                            CAGE_TEST_READY_WHEN_LOGGED="GPU KV cache size")
    assert bad.returncode == 1 and "never printed 'kv_cache_dtype=fp8, '" in bad.stdout, bad.stdout


def test_an_sglang_fp8_start_must_log_e4m3_u8(stub_bin: Path, tmp_path: Path) -> None:
    import test_kv_pool_capture_s0f24 as kpt

    python_stub = kpt._engine_stub(stub_bin, "sglang-python")
    lines = kpt.SGLANG_LOG + "[2026-10-12 10:00:00] Using KV cache dtype: torch.float8_e4m3fn\n" + kpt.SGLANG_READY_LINE
    ok = kpt._run_launcher("manage_sglang_server.sh", stub_bin, tmp_path / "a", "start", "fake/test-model",
                           SGLANG_START_TIMEOUT="10", CAGE_SGLANG_PYTHON=str(python_stub),
                           SGLANG_KV_CACHE_DTYPE="fp8_e4m3", CAGE_TEST_POOL_LINES=lines,
                           CAGE_TEST_READY_WHEN_LOGGED="KV Cache is allocated")
    assert ok.returncode == 0 and "KV cache dtype realized: torch.float8_e4m3fn" in ok.stdout, ok.stdout + ok.stderr
    e5m2 = lines.replace("float8_e4m3fn", "float8_e5m2")
    bad = kpt._run_launcher("manage_sglang_server.sh", stub_bin, tmp_path / "b", "start", "fake/test-model",
                            SGLANG_START_TIMEOUT="10", CAGE_SGLANG_PYTHON=str(python_stub),
                            SGLANG_KV_CACHE_DTYPE="fp8_e4m3", CAGE_TEST_POOL_LINES=e5m2,
                            CAGE_TEST_READY_WHEN_LOGGED="KV Cache is allocated")
    assert bad.returncode == 1 and "never said 'Using KV cache dtype: torch.float8_e4m3fn'" in bad.stdout, bad.stdout


def test_the_compressor_counts_tokens_without_the_length_warning_u10() -> None:
    from src.orchestration.compression import ContextCompressor

    seen: Dict[str, Any] = {}

    class _Tok:
        def __call__(self, text, **kw):
            seen.update(kw)
            return {"input_ids": text.split()}

    count = ContextCompressor()._actual_token_count(types.SimpleNamespace(tokenizer=_Tok()), ["a b", "c"])
    assert count == 3 and seen == {"add_special_tokens": False, "verbose": False}


# --------------------------------------------------------------------------- #
# U5: a pull never carries files outside the seal into the run tree, and never
# deletes them
# --------------------------------------------------------------------------- #


def test_the_pull_moves_files_outside_the_seal_aside_and_the_verifier_names_them_u5(tmp_path: Path) -> None:
    """A window the run reset after a backup pass stays in the backup mirror
    (the transfer never deletes); it must not land in the verified run tree."""
    import subprocess

    import test_verify_results_v2 as tv

    source = tv._mk_green(tmp_path / "pod")
    stale_window = tv._first_window(source).parent / "window_squad_v2-09"
    stale_window.mkdir()
    (stale_window / "requests.jsonl").write_text('{"example_id": "old"}\n', encoding="utf-8")
    (tv._first_window(source) / "partial.tmp").write_text("x", encoding="utf-8")
    sentinel = tv._first_window(source).parent / ".STATUS_keep"
    sentinel.write_text("reason=dry-window-x\n", encoding="utf-8")
    dest = tmp_path / "mac" / source.name
    proc = subprocess.run(["bash", str(SCRIPTS / "5_observability" / "pull_run.sh"), f"file://{source}", str(dest)],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0 and "SAFE TO TEARDOWN" in proc.stdout, proc.stdout + proc.stderr
    assert "files outside the seal    : 2" in proc.stdout
    stale_root = dest.parent / f"{dest.name}.stale"
    moved = (stale_root / "STALE_FILES.txt").read_text(encoding="utf-8").split()
    assert len(moved) == 2 and all((stale_root / m).is_file() for m in moved)  # moved, never deleted
    assert not any(dest.rglob("window_squad_v2-09")) and not any(dest.rglob("partial.tmp"))
    assert (dest / sentinel.relative_to(source)).is_file()  # dot sentinels stay (unsealed by design)
    report = tv.vr.verify_run(dest)
    assert report["ok"] is True
    (warn,) = [f for f in report["findings"] if f["check"] == "ledger" and "S0F-88 U5" in f["detail"]]
    assert warn["severity"] == "WARN" and "2 file(s)" in warn["detail"]


def test_a_second_pull_lists_each_stale_file_once_and_keeps_unrelated_empty_folders_l3(tmp_path: Path) -> None:
    """Review L3: the backup mirror brings the same leftovers back on every
    pull, so STALE_FILES.txt stays one line per path; and only the folders the
    move emptied are removed, not an empty folder the run tree already had."""
    import subprocess

    import test_verify_results_v2 as tv

    source = tv._mk_green(tmp_path / "pod")
    stale_window = tv._first_window(source).parent / "window_squad_v2-09"
    stale_window.mkdir()
    (stale_window / "requests.jsonl").write_text('{"example_id": "old"}\n', encoding="utf-8")
    dest = tmp_path / "mac" / source.name
    cmd = ["bash", str(SCRIPTS / "5_observability" / "pull_run.sh"), f"file://{source}", str(dest)]
    first = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert first.returncode == 0, first.stdout + first.stderr
    kept = dest / tv._first_window(source).parent.relative_to(source) / "window_already_empty"
    kept.mkdir()
    second = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert second.returncode == 0 and "files outside the seal    : 1" in second.stdout, second.stdout + second.stderr
    listing = (dest.parent / f"{dest.name}.stale" / "STALE_FILES.txt").read_text(encoding="utf-8").splitlines()
    assert listing == [stale_window.relative_to(source).joinpath("requests.jsonl").as_posix()]
    assert kept.is_dir() and not any(dest.rglob("window_squad_v2-09"))


# --------------------------------------------------------------------------- #
# U9: the master never seals a tree after a failed run stage
# --------------------------------------------------------------------------- #

from test_experiment_master import world  # noqa: E402,F401  (fixture re-export)


def test_the_seal_stage_refuses_after_a_failed_run_stage_u9(world: Dict[str, Path], tmp_path: Path) -> None:
    """run_campaign.py --seal declines a tree with failed cells; stage 9 sealed
    it anyway on a resume with --from seal."""
    import test_experiment_master as tm

    reh = tmp_path / "S1_rehearsal.env"
    reh.write_text(world["profile"].read_text(encoding="utf-8") + "\nREHEARSAL_N=50\n", encoding="utf-8")
    world["profile"] = reh
    proc = tm._master(world, "--yes", "provision", "--to", "run", CAGE_TEST_BLOCKED="1")
    assert proc.returncode == 1 and tm._state(world)["stages"]["run"]["status"] == "failed"
    proc = tm._master(world, "--from", "seal", "--to", "seal")
    assert proc.returncode == 1, proc.stdout[-2000:]
    assert "the run stage is failed, not passed" in proc.stdout
    assert not any(".cage_jobs/seal.cmd" in c for c in tm._calls(world))  # nothing was submitted


# --------------------------------------------------------------------------- #
# B-6 (R8): the GPU reranker serves what the CPU reranker serves
# --------------------------------------------------------------------------- #


def test_the_device_probe_fails_on_any_query_whose_served_list_differs_b6() -> None:
    probe = _load("probe_reranker_device_b6", SCRIPTS / "checks" / "probe_reranker_device.py")
    items = [(f"q{i}", [f"p{j}" for j in range(10)]) for i in range(5)]

    def cpu(pairs):
        return [10.0 - j for j, _ in enumerate(pairs)]

    same = probe.compare_devices(items, cpu, lambda pairs: [s + 1e-6 for s in cpu(pairs)], top_k=3)
    assert same["ok"] is True and same["identical"] == 5 and same["max_abs_score_diff"] < 1e-5

    def gpu(pairs):  # query q2 swaps its 3rd and 4th candidates
        scores = cpu(pairs)
        if pairs[0][0] == "q2":
            scores[2], scores[3] = scores[3], scores[2]
        return scores

    diff = probe.compare_devices(items, cpu, gpu, top_k=3)
    assert diff["ok"] is False and diff["identical"] == 4 and diff["order_swaps"] == 0
    (d,) = diff["differences"]
    assert d["query"] == 2 and d["served_cpu"] == [0, 1, 2] and d["served_gpu"] == [0, 1, 3]
    # a swap below the served top k changes nothing served
    below = probe.compare_devices(items, cpu, lambda p: cpu(p)[:5] + [cpu(p)[6], cpu(p)[5]] + cpu(p)[7:], top_k=3)
    assert below["ok"] is True and below["order_swaps"] == 0

    # review 2026-10-13 item 7 (decision 4): the same SET of texts in another
    # order is a recorded swap, not a failure
    def swapped(pairs):
        scores = cpu(pairs)
        if pairs[0][0] == "q1":
            scores[0], scores[1] = scores[1], scores[0]
        return scores

    swap = probe.compare_devices(items, cpu, swapped, top_k=3)
    assert swap["ok"] is True and swap["order_swaps"] == 1 and swap["identical"] == 5
    # the pools repeat texts (16 distinct paragraphs over 50 questions): two
    # identical passages changing places serve the same texts
    dup_items = [("q0", ["same", "p1", "p2", "same", "p4", "p5", "p6", "p7", "p8", "p9"])]
    dup = probe.compare_devices(dup_items, cpu, lambda p: cpu(p)[:5] + [cpu(p)[6], cpu(p)[5]] + cpu(p)[7:], top_k=3)
    assert dup["ok"] is True
    moved = probe.compare_devices(dup_items, cpu, lambda p: [s if i not in (0, 3) else {0: 7.0, 3: 10.0}[i] for i, s in enumerate(cpu(p))], top_k=3)
    assert moved["ok"] is True and moved["order_swaps"] == 0  # texts {same, p1, p2} served either way


def test_the_device_probe_main_builds_pools_from_the_manifest_and_prints_one_line_l4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Review L4: main() itself, with the runner's real manifest selection, a
    stub dataset load and a stub cross-encoder per device: the pools are the
    question's own context then the next ones in manifest order, both devices
    score the same pairs, and the master's one line and the record follow."""
    from src.data.loader import CAGExample
    from src.orchestration import ir

    probe = _load("probe_reranker_device_l4", SCRIPTS / "checks" / "probe_reranker_device.py")
    rx = _load("run_experiment_l4", SCRIPTS / "3_run" / "run_experiment.py")
    examples = [CAGExample(id=f"e{i}", question=f"q{i}", context=[f"c{i}a", f"c{i}b"], answer="", metadata={})
                for i in range(60)]
    order = [f"e{i}" for i in reversed(range(60))]  # manifest order differs from the load order
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"dataset": "squad_v2", "split": "validation", "seed": 42,
                                    "trials": {"1": order}}), encoding="utf-8")
    loads: List[Any] = []
    fake_rx = types.SimpleNamespace(
        get_loader=lambda name, *, split, seed: loads.append((name, split, seed))
        or types.SimpleNamespace(load=lambda max_examples: examples),
        select_manifest_prefix=rx.select_manifest_prefix,
    )
    monkeypatch.setitem(sys.modules, "run_experiment", fake_rx)
    scored: Dict[str, List[Any]] = {"cpu": [], "cuda": []}
    flip = {"on": False}

    class _Reranker:
        def __init__(self, model_name: str, *, device: str) -> None:
            def predict(pairs):
                scored[device].append(list(pairs))
                scores = [float(len(pairs) - j) for j in range(len(pairs))]
                if flip["on"] and device == "cuda" and pairs[0][0] == "q20":  # q0 to q9 fall outside the 50
                    scores[2], scores[3] = scores[3], scores[2]
                return scores
            self._model = types.SimpleNamespace(predict=predict)

    monkeypatch.setattr(ir, "CrossEncoderReranker", _Reranker)
    out = tmp_path / "rec.json"
    assert probe.main(["--manifest", str(manifest), "--out", str(out)]) == 0
    line = capsys.readouterr().out.strip()
    assert line.startswith("RERANK_DEVICE_OK queries=50 identical=50"), line
    assert loads == [("squad_v2", "validation", 42)]
    first = scored["cpu"][0]
    assert first[0] == ("q59", "c59a c59b") and first[1] == ("q59", "c58a c58b") and len(first) == 10
    assert scored["cpu"] == scored["cuda"] and len(scored["cpu"]) == 50
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["ok"] is True and record["pool"] == 10 and record["model"] == "BAAI/bge-reranker-large"
    # the last of the 50 wraps to the head of the 50 selected, never past them
    assert scored["cpu"][-1][1] == ("q10", "c59a c59b")

    flip["on"] = True
    assert probe.main(["--manifest", str(manifest), "--out", str(out)]) == 1
    assert capsys.readouterr().out.startswith("RERANK_DEVICE_FAILED reason=1 of 50 queries")

    monkeypatch.setattr(ir, "CrossEncoderReranker", lambda *a, **k: (_ for _ in ()).throw(OSError("no cuda")))
    assert probe.main(["--manifest", str(manifest)]) == 1
    assert "RERANK_DEVICE_FAILED reason=OSError: no cuda" in capsys.readouterr().out


def test_the_plan_header_records_the_measured_basis_and_class_tables_b9(tmp_path: Path) -> None:
    import test_s0f88_batch2 as b2

    plan = b2._plan_with_measured_classes(tmp_path)
    header = plan["rung_calibration"]
    assert header["basis_measured"] == rc.LAMBDA_BASIS_MEASURED
    for art in header["artifacts"].values():
        tables = art["measured_classes"]
        assert tables and all(set(t) == {"rungs", "labels"} for t in tables.values())
        assert all(label == "ESTIMATED" for t in tables.values() for label in t["labels"].values())


def test_a_local_model_path_still_reaches_autoconfig_u1(monkeypatch) -> None:
    """The HF cache lookup refuses a local path (HFValidationError); the U1
    change returned None there instead of reading the config as before."""
    import test_review_fixes_evaluation as tre
    from src.evaluation.compression import analytical_kv_footprint

    tre._install_fake_transformers(monkeypatch, {"num_hidden_layers": 40, "num_attention_heads": 40,
                                                 "num_key_value_heads": 8, "hidden_size": 5120, "head_dim": 128})
    fp = analytical_kv_footprint("/models/local/qwen3-14b", 1000, dtype="bf16", baseline_dtype="bf16")
    assert fp is not None and fp["kv_bytes_per_token"] == 2 * 40 * 8 * 128 * 2


def _fake_proc(root: Path, rss_gib: tuple) -> Path:
    proc = root / "proc"
    for pid, gib in enumerate(rss_gib, start=100):
        (proc / str(pid)).mkdir(parents=True)
        (proc / str(pid) / "status").write_text(f"Name:\tx\nVmRSS:\t{gib * (1 << 30) // 1024} kB\n", encoding="utf-8")
    (proc / "self").mkdir()  # a non-numeric entry is skipped
    return proc


def test_the_pod_memory_probe_bounds_on_the_pod_allocation_and_fails_closed_u2_h1_item2(tmp_path: Path) -> None:
    """Review H1 and review 2026-10-13 item 2: on S0 attempt 4 /proc/meminfo
    showed the host's 1.4 TiB on a pod allocated 172 GB. The bound is the
    pod's allocation (the RunPod API's memoryInGb) minus the container's own
    processes' resident memory; a cgroup limit is a second bound; page cache
    is never counted; with no readable bound the probe refuses (the first
    version fell back to the host's numbers and passed any store)."""
    probe = _load("probe_pod_memory_item2", SCRIPTS / "checks" / "probe_pod_memory.py")
    GIB = 1 << 30
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(f"MemTotal: {1400 * GIB // 1024} kB\nMemAvailable: {1300 * GIB // 1024} kB\n", encoding="utf-8")
    proc = _fake_proc(tmp_path, (12, 8))  # the engine and the runner: 20 GiB resident
    nolimit = tmp_path / "nolimit"; nolimit.mkdir()
    (nolimit / "memory.max").write_text("max", encoding="utf-8")
    rec = probe.judge(172, 172, meminfo, nolimit, proc)
    assert rec["bound"] == "pod-allocation" and rec["rss_bytes"] == 20 * GIB
    assert rec["available_bytes"] == 172 * 10**9 - 20 * GIB and rec["ok"] is False  # 172 GiB does not fit 172 GB
    assert probe.judge(22, 172, meminfo, nolimit, proc)["ok"] is True
    assert rec["meminfo_available_bytes"] == 1300 * GIB  # recorded, never a bound
    # a readable cgroup limit below the allocation is the tighter bound
    v2 = tmp_path / "v2"; v2.mkdir()
    (v2 / "memory.max").write_text(str(100 * GIB), encoding="utf-8")
    rec = probe.judge(90, 172, meminfo, v2, proc)
    assert rec["bound"] == "cgroup-v2" and rec["available_bytes"] == 80 * GIB and rec["ok"] is False
    assert probe.judge(80, 172, meminfo, v2, proc)["ok"] is True
    v1 = tmp_path / "v1" / "memory"; v1.mkdir(parents=True)
    (v1 / "memory.limit_in_bytes").write_text("9223372036854771712", encoding="utf-8")  # a v1 "no limit"
    assert probe.judge(22, 172, meminfo, tmp_path / "v1", proc)["bound"] == "pod-allocation"
    # fail closed: no allocation and no limit refuses even with 1.3 TiB in meminfo
    rec = probe.judge(22, 0, meminfo, nolimit, proc)
    assert rec["available_bytes"] is None and rec["ok"] is False and rec["bound"] is None
    assert probe.judge(0, 0, meminfo, nolimit, proc)["ok"] is True  # no store, nothing to hold
    args = ["--meminfo", str(meminfo), "--proc-root", str(proc)]
    assert probe.main(["--need-gib", "22", "--pod-gib", "0", "--cgroup-root", str(nolimit), *args]) == 1
    assert probe.main(["--need-gib", "22", "--pod-gib", "172", "--cgroup-root", str(nolimit), *args]) == 0
    assert probe.main(["--need-gib", "172", "--pod-gib", "172", "--cgroup-root", str(nolimit), *args]) == 1


def test_the_pod_memory_probe_prints_unknown_without_a_bound_item2(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    probe = _load("probe_pod_memory_item2b", SCRIPTS / "checks" / "probe_pod_memory.py")
    proc = _fake_proc(tmp_path, (1,))
    nolimit = tmp_path / "nolimit"; nolimit.mkdir()
    assert probe.main(["--need-gib", "22", "--pod-gib", "0", "--cgroup-root", str(nolimit), "--proc-root", str(proc),
                       "--meminfo", str(tmp_path / "absent")]) == 1
    out = capsys.readouterr().out
    assert out.startswith("POD_MEMORY_UNKNOWN") and "pass --pod-gib" in out


def test_a_campaign_window_refuses_an_sglang_server_without_the_cache_report_l1(tmp_path, monkeypatch, capsys):
    """Review L1: the probe only warned, so a server without the report would
    serve rows whose zero-cached counts read None."""
    import test_run_experiment_campaign_layout as layout

    class _NoReport(layout.StubEngine):
        def probe_cache_report(self):
            return False

    root = tmp_path / "camp" / "a" / layout.RUN_ID
    with pytest.raises(SystemExit) as exit_info:
        layout._run_cell(monkeypatch, root, "no_cache", "squad_v2", ttft_base=100.0, num_trials=1,
                         engine_factory=lambda model: _NoReport(model, "no_cache", 100.0))
    assert exit_info.value.code == 1
    assert "enable_cache_report=False" in capsys.readouterr().out

    class _Report(layout.StubEngine):
        def probe_cache_report(self):
            return True

    ok_root = tmp_path / "camp2" / "a" / layout.RUN_ID
    layout._run_cell(monkeypatch, ok_root, "no_cache", "squad_v2", ttft_base=100.0, num_trials=1,
                     engine_factory=lambda model: _Report(model, "no_cache", 100.0))
    (wdir,) = sorted(ok_root.glob("cells/*/window_squad_v2-*"))
    assert json.loads((wdir / "metrics.json").read_text(encoding="utf-8"))["cache_report_probe"] is True


def test_the_cache_report_probe_retries_a_blip_and_refuses_a_verified_off_at_once_l1(tmp_path, monkeypatch):
    """Review 2026-10-13 item 26: one unanswered GET /server_info (None) is a
    blip; the probe is retried before the window is refused."""
    import test_run_experiment_campaign_layout as layout

    monkeypatch.setattr(layout.runner, "CACHE_REPORT_PROBE_WAIT_S", 0.0)
    calls: List[Any] = []

    def _engine(answers):
        class _Probe(layout.StubEngine):
            def probe_cache_report(self):
                calls.append(answers[min(len(calls), len(answers) - 1)])
                return calls[-1]
        return _Probe

    root = tmp_path / "blip" / "a" / layout.RUN_ID
    layout._run_cell(monkeypatch, root, "no_cache", "squad_v2", ttft_base=100.0, num_trials=1,
                     engine_factory=lambda model: _engine([None, None, True])(model, "no_cache", 100.0))
    assert calls == [None, None, True]
    (wdir,) = sorted(root.glob("cells/*/window_squad_v2-*"))
    assert json.loads((wdir / "metrics.json").read_text(encoding="utf-8"))["cache_report_probe"] is True

    calls.clear()
    with pytest.raises(SystemExit):
        layout._run_cell(monkeypatch, tmp_path / "dead" / "a" / layout.RUN_ID, "no_cache", "squad_v2",
                         ttft_base=100.0, num_trials=1,
                         engine_factory=lambda model: _engine([None])(model, "no_cache", 100.0))
    assert calls == [None, None, None]  # three tries, then the refusal names the unanswered probe

    calls.clear()
    with pytest.raises(SystemExit):
        layout._run_cell(monkeypatch, tmp_path / "off" / "a" / layout.RUN_ID, "no_cache", "squad_v2",
                         ttft_base=100.0, num_trials=1,
                         engine_factory=lambda model: _engine([False])(model, "no_cache", 100.0))
    assert calls == [False]  # a verified False refuses at once


def test_the_regime_frame_carries_every_window_field_including_the_hold_statistics_item28() -> None:
    from src.analysis.regime_inputs import MAX_SAMPLE_HOLD_S, compute_regime_inputs

    frame = compute_regime_inputs(_ticks(0.0, 100.0, 0.25), [(0.0, 50.0), (200.0, 210.0)],
                                  allow_missing=True, max_hold_s=MAX_SAMPLE_HOLD_S)
    for col in ("rho_kv_queued_median", "longest_hold_s", "hold_excess_s"):
        assert col in frame.columns
    assert frame.loc[0, "longest_hold_s"] == pytest.approx(0.25) and frame.loc[0, "hold_excess_s"] == 0.0
    assert bool(frame.loc[1, "telemetry_ok"]) is False and math.isnan(frame.loc[1, "hold_excess_s"])


def test_the_floor_counts_distinct_prompts_over_the_served_head_item31(tmp_path: Path) -> None:
    """Review 2026-10-13 item 31: 35 distinct blocks, but block 3 repeats block
    0 inside the first 30 served; the floor must refuse (it counted the whole
    list before and would have served a repeat)."""
    path = tmp_path / "head_repeat.json"
    blocks = [{"block_id": f"b{i}", "text": f"Paper {i}."} for i in range(35)]
    blocks[3]["text"] = blocks[0]["text"]
    path.write_text(json.dumps({"blocks": blocks}), encoding="utf-8")
    requests = cc.build_requests(str(path))
    assert len({getattr(r, "prompt", None) for r in requests}) == 34  # distinct over the whole list
    with pytest.raises(CalibrationError, match="first 30 requests carry 29"):
        asyncio.run(cc.measure_floor(_FloorAdapter(), requests))


def test_a_warmup_the_server_refuses_for_its_length_fails_the_campaign_window_q4_item8(tmp_path, monkeypatch, capsys):
    """Review 2026-10-13 item 8: the Q-4 HTTP 400 refusal was pinned as source
    text only. A stub engine that answers a warm-up pool request with the
    server's context-length error fails the window before any measured send."""
    import test_run_experiment_campaign_layout as layout

    sent: List[str] = []

    class _Refusing(layout.StubEngine):
        def _respond(self, request):
            response = super()._respond(request)
            rid = str(request.request_id or "")
            sent.append(rid)
            if "warmup_pool" in rid:
                response.error = ("400 Client Error: This model's maximum context length is 4096 tokens. "
                                  "However, you requested 4200 tokens")
                response.generated_text = ""
            return response

    root = tmp_path / "camp" / "a" / layout.RUN_ID
    with pytest.raises(SystemExit) as exit_info:
        layout._run_cell(monkeypatch, root, "no_cache", "squad_v2", ttft_base=100.0, num_trials=1,
                         engine_factory=lambda model: _Refusing(model, "no_cache", 100.0),
                         extra_argv=("--num-queries", "2", "--warmup-pool-queries", "2"))
    assert exit_info.value.code == 1
    assert "was refused for its length" in capsys.readouterr().out
    assert all("warmup" in rid for rid in sent)  # nothing measured was sent


def test_the_step_runner_keeps_its_exit_code_and_only_times_out_under_the_env(monkeypatch) -> None:
    """Review 2026-10-13 item 21: without CAGE_STEP_TIMEOUT_S the step runner
    is unchanged; with it, a step that ends in time returns its own code."""
    import test_run_campaign as trc

    monkeypatch.delenv(trc.rc.STEP_TIMEOUT_ENV, raising=False)
    assert trc.rc._exec([sys.executable, "-c", "raise SystemExit(7)"], {}) == 7
    monkeypatch.setenv(trc.rc.STEP_TIMEOUT_ENV, "30")
    assert trc.rc._exec([sys.executable, "-c", "raise SystemExit(5)"], {}) == 5
    assert trc.rc.STEP_TIMEOUT_EXIT == 124


@pytest.mark.skipif(
    not (Path("/.dockerenv").exists() or __import__("os").environ.get("CAGE_CONTAINER") == "1"),
    reason="a process-lifecycle test: runs in a container, never on the macOS host (CLAUDE.md)",
)
def test_the_step_runner_stops_a_hung_step_through_its_own_process_group(monkeypatch) -> None:
    import test_run_campaign as trc

    monkeypatch.setenv(trc.rc.STEP_TIMEOUT_ENV, "1")
    monkeypatch.setattr(trc.rc, "STEP_KILL_GRACE_S", 2.0)
    assert trc.rc._exec([sys.executable, "-c", "import time; time.sleep(60)"], {}) == trc.rc.STEP_TIMEOUT_EXIT


def test_the_reranker_honors_the_frozen_revision_pin_item9(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review 2026-10-13 item 9: CAGE_RERANKER_REVISION (the freeze file's
    INSTRUMENT_REVISIONS.reranker, set on the run job) pins the CrossEncoder
    load; unpinned loads pass revision=None."""
    import sentence_transformers

    from src.orchestration import ir

    seen: List[Dict[str, Any]] = []

    class _Fake:
        def __init__(self, name, **kwargs):
            seen.append({"name": name, **kwargs})
            self.tokenizer = None
            self.max_length = 512

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", _Fake)
    monkeypatch.delenv("CAGE_RERANKER_REVISION", raising=False)
    r = ir.CrossEncoderReranker("BAAI/bge-reranker-large", device="cuda")
    assert r.revision is None and "revision" not in seen[-1] and seen[-1]["device"] == "cuda"
    monkeypatch.setenv("CAGE_RERANKER_REVISION", "55611d7bca2a7133960a6d3b71e083071bbfc312")
    r = ir.CrossEncoderReranker("BAAI/bge-reranker-large", device="cpu")
    assert r.revision == "55611d7bca2a7133960a6d3b71e083071bbfc312" == seen[-1]["revision"]


def test_the_backup_daemon_archives_the_run_tree_with_pipe_names_inside_item17(tmp_path: Path) -> None:
    """Review 2026-10-13 item 17: the network volume refuses "|" in names, so
    the mirror lives on the container disk; a tar archive of the run tree has
    no "|" in its own name and keeps the cell names inside it."""
    import subprocess

    tree = tmp_path / "results" / "smoke" / "a" / "run-001"
    cell = tree / "cells" / "gold-fresh|none|none|single|vllm|qwen3-14b|F1"
    cell.mkdir(parents=True)
    (cell / "cell.json").write_text("{}", encoding="utf-8")
    env = {**__import__("os").environ, "CAGE_BACKUP_ARCHIVE_DIR": str(tmp_path / "arch"), "CAGE_BACKUP_TARGET": "file:///dev/null"}
    proc = subprocess.run(["bash", str(SCRIPTS / "5_observability" / "gcs_backup_daemon.sh"), "archive", str(tree)],
                          capture_output=True, text=True, env=env, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    archive = tmp_path / "arch" / "run-001.tar.gz"
    assert archive.is_file() and "archived" in proc.stdout
    listing = subprocess.run(["tar", "tzf", str(archive)], capture_output=True, text=True, check=True).stdout
    assert "run-001/cells/gold-fresh|none|none|single|vllm|qwen3-14b|F1/cell.json" in listing
    assert not list((tmp_path / "arch").glob(".*.tmp"))  # written to a temporary name, then moved
    env.pop("CAGE_BACKUP_ARCHIVE_DIR")
    proc = subprocess.run(["bash", str(SCRIPTS / "5_observability" / "gcs_backup_daemon.sh"), "archive", str(tree)],
                          capture_output=True, text=True, env=env, timeout=120)
    assert proc.returncode != 0 and "CAGE_BACKUP_ARCHIVE_DIR is unset" in proc.stdout + proc.stderr


def test_the_premise_split_is_bounded_and_never_aborts_scoring_l2() -> None:
    """Review L2: a 130,000-character terminator-free context bisected over its
    whole length (13.5M characters tokenized); a split error escaped the
    batched scoring pass."""
    ev = _premise_evaluator()
    seen = {"chars": 0}

    class _CountingTok(_NliWordTok):
        def __call__(self, texts, add_special_tokens=False, verbose=True):
            seen["chars"] += sum(len(t) for t in texts)
            return super().__call__(texts, add_special_tokens=add_special_tokens, verbose=verbose)

    ev._nli_model = types.SimpleNamespace(tokenizer=_CountingTok())
    blob = " ".join(f"k{i}=v{i}" for i in range(13000))  # about 130,000 characters, one sentence
    windows = ev._split_premise_windows(blob)
    assert all(len(w.split()) <= ev.NLI_PREMISE_WINDOW_TOKENS for w in windows)
    assert seen["chars"] < 6_000_000, seen["chars"]

    class _BrokenStrict(_NliWordTok):
        def __call__(self, texts, add_special_tokens=False, verbose=True):
            if verbose is False:
                raise RuntimeError("tokenizer refused")
            return super().__call__(texts, add_special_tokens=add_special_tokens)

    ev._nli_model = types.SimpleNamespace(tokenizer=_BrokenStrict())
    assert ev._split_premise_windows(blob) == [blob]  # the pre-split behavior, no exception
