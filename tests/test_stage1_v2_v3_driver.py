"""Stage 1 Batch S1-A: findings V2 and V3 of the Batch 1 code validation
(MyDocs/CODE_VALIDATION_BATCH1_2026-09-17.md, items A and E), owner GO
2026-10-05.

V2, engine handoff: the plan orders cells hf, sglang, vllm, and every
relaunch step calls its OWN launcher's ``restart`` (single) or self-cleaning
``start`` (pd), which stop only that launcher's engine family. When the
family changes, the previous engine stays resident on the GPU and the next
start fails after its readiness budget (the session a dry trace skipped 430
of 870 cells this way), and nothing stops the last engine when the run ends.
Now every relaunch step records its ``launcher_key`` and ``stop_argv`` (the
launcher's ``stop`` verb, which every launcher of the fleet accepts with no
model argument and exits 0 with nothing running), ``stop_boundaries`` is the
pure function naming where a family must be stopped, and ``run_plan`` stops
every launcher the plan uses once before the first step (clean room), the
previous family before a relaunch of another launcher, and the last family
when the run ends. Same-launcher relaunches (prefix ON then OFF) stop nothing
in between: the launcher's own verb is self-cleaning.

V3, the window bound: window cells carried ``--num-queries 200`` and
``--duration-s <window_duration_s>``; the runner's replay guard refuses once
rate x duration exceeds the 200 prepared requests unless CAGE_ALLOW_REPLAY=1,
so every pressure window at a real rate refused. The charter registers W =
200 requests per window and S0 proved the request-count form live
(``--arrival-count 50``, 6.5 s spans, 50 of 50 served). Now every window cell
carries ``--arrival-count`` equal to its own ``--num-queries`` (arrivals ==
the prepared pool, so the guard never fires) and no ``--duration-s``;
``window_duration_s`` stays in the plan header as the pre-costed estimate.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.orchestration.calibration import (  # noqa: E402
    FLOOR_N_REQUESTS,
    FLOOR_STATISTIC,
    PROCEDURE_VERSION,
)
from src.orchestration.load_generator import (  # noqa: E402
    LoadGeneratorError,
    ensure_no_measured_replay,
    generate_arrival_schedule,
)

RUN_CAMPAIGN_PY = REPO_ROOT / "scripts" / "3_run" / "run_campaign.py"
RUN_EXPERIMENT_PY = REPO_ROOT / "scripts" / "3_run" / "run_experiment.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # register BEFORE exec (dataclass-safe)
    spec.loader.exec_module(module)
    return module


rc = _load("run_campaign_stage1_v2_v3", RUN_CAMPAIGN_PY)


# ---------------------------------------------------------------------------
# helpers: a hermetic plan (floor table, cal-v2 floors, a recording stub)
# ---------------------------------------------------------------------------

_DEMAND = 10_000_000_000


def _floor_table(tmp_path: Path) -> Path:
    rows = []
    for r in (1.5, 1.25, 1.0, 0.75, 0.5, 0.375, 0.25):
        rows.append(
            {
                "r": r,
                "demand_bytes": _DEMAND,
                "budget_bytes": int(r * _DEMAND),
                "lambda_kv_rps": 2.0 * r,
                "lambda_compute_rps": None,
                "lambda_star_pred_rps": 2.0 * r,
                "lambda_star_basis": "test",
            }
        )
    doc = {
        "schema": "floor-table-v1",
        "generated_inputs": {
            "model": "qwen3-14b", "engine": "vllm", "kv_dtype": "bf16", "grid": "test",
        },
        "rows": rows,
    }
    path = tmp_path / "floor_table.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _calibrations_for(grid, directory: Path) -> Dict[str, Path]:
    engines = set(grid.f1_engines) | set(grid.f2_engines) | set(grid.f3_engines)
    engines |= {engine for _bid, engine, _topology in grid.dist_cells}
    directory.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}
    for engine in sorted(engines - {"hf"}):
        doc = {
            "procedure_version": PROCEDURE_VERSION,
            "model": rc.HF_ID_OF_SLUG[grid.model],
            "engine": engine,
            "budget_fraction": 1.5,
            "procedure": {},
            "confirmatory": False,
            "floor": {
                "ttft_s": 0.1, "tpot_s": 0.01,
                "n_requests": FLOOR_N_REQUESTS, "statistic": FLOOR_STATISTIC,
            },
            "lambda_star": {"label": "ESTIMATED", "lambda_star_qps": 2.0},
        }
        path = directory / f"calibration_{engine}.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        out[engine] = path
    return out


def _grid(**overrides: Any):
    kwargs: Dict[str, Any] = dict(
        session="a",
        group="A",
        model="qwen3-14b",
        f1_baselines=("B1",),
        f1_engines=("vllm",),
        f1_datasets=("squad_v2",),
        hf_oracle_cells=(),
        f2_baselines=(),
        f2_engines=("vllm",),
        f2_budgets=(),
        f2_rates=(),
        f2_dataset="qasper",
        f3_baselines=(),
        f3_engines=("vllm",),
        f3_budgets=(),
        f3_rates=(),
        f3_dataset="qasper",
        dist_cells=(),
    )
    kwargs.update(overrides)
    return rc.SessionGrid(**kwargs)


def _plan_for(grid, floor_path: Path, stub_cmd, *, launcher_cmds=None) -> Dict[str, Any]:
    floor = rc.load_floor_table(floor_path)
    calibrations = _calibrations_for(grid, floor_path.parent / "cal")
    orig = rc.SESSION_GRIDS
    rc.SESSION_GRIDS = {grid.session: grid}
    try:
        return rc.build_plan(
            grid.session,
            floor,
            window_duration_s=60.0,
            runner_cmd=stub_cmd,
            launcher_cmds=launcher_cmds or {
                "vllm": stub_cmd, "sglang": stub_cmd, rc.PD_LAUNCHER_KEY: stub_cmd,
            },
            calibrations=calibrations,
        )
    finally:
        rc.SESSION_GRIDS = orig


_STUB_SOURCE = """\
import json, os, sys
with open(os.environ["STUB_CALLS"], "a", encoding="utf-8") as fh:
    fh.write(json.dumps({
        "argv": sys.argv[1:],
        "env": {k: v for k, v in os.environ.items() if k.startswith(("CAGE_", "VLLM_", "SGLANG_"))},
    }) + "\\n")
exact = os.environ.get("STUB_FAIL_EXACT", "")
if exact and " ".join(sys.argv[1:]) == exact:
    sys.exit(1)
sys.exit(0)
"""


@pytest.fixture()
def stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A recording subprocess stub standing in for the runner AND every
    launcher: each invocation appends one JSON line (argv after the script,
    the CAGE_/VLLM_/SGLANG_ env). STUB_FAIL_EXACT names the ONE joined argv
    that exits 1 (exact, not a substring: a cell's --campaign-root carries
    the pytest tmp path, which carries the test's name)."""
    stub_path = tmp_path / "stub.py"
    stub_path.write_text(_STUB_SOURCE, encoding="utf-8")
    calls_path = tmp_path / "stub_calls.jsonl"
    monkeypatch.setenv("STUB_CALLS", str(calls_path))
    monkeypatch.delenv("STUB_FAIL_EXACT", raising=False)
    for name in (*rc.API_BASE_OVERRIDE_ENVS, *rc.SHELL_PORT_ENVS, *rc.CELL_PIN_ENVS):
        monkeypatch.delenv(name, raising=False)

    class Stub:
        cmd = (sys.executable, str(stub_path))

        @staticmethod
        def calls() -> List[Dict[str, Any]]:
            if not calls_path.exists():
                return []
            return [
                json.loads(line)
                for line in calls_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

    return Stub


def _run_root(tmp_path: Path) -> Path:
    root = tmp_path / "results" / "camp" / "a" / "run-001"
    root.parent.mkdir(parents=True, exist_ok=True)
    return root


def _verbs(calls: List[Dict[str, Any]]) -> List[str]:
    """One token per stub call: ``stop:<engine>`` / ``restart:<engine>`` for
    launcher calls (the engine read from the port env the relaunch exports),
    ``cell`` for runner calls."""
    out: List[str] = []
    for call in calls:
        argv = call["argv"]
        env = call["env"]
        if argv and argv[0] in ("stop", "restart", "start"):
            engine = "vllm" if "VLLM_PORT" in env else "sglang" if "SGLANG_PORT" in env else "?"
            out.append(f"{argv[0]}:{engine}")
        else:
            out.append("cell")
    return out


def _cells(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [s for s in plan["steps"] if s["kind"] == "cell"]


def _relaunches(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [s for s in plan["steps"] if s["kind"] == "relaunch"]


def _argv_value(argv: List[str], flag: str):
    return argv[argv.index(flag) + 1] if flag in argv else None


def _write_plan(tmp_path: Path, plan: Dict[str, Any], name: str = "plan.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


# ===========================================================================
# V2: engine handoff
# ===========================================================================


def test_v2_constants() -> None:
    assert rc.LAUNCHER_STOP_VERB == "stop"
    assert rc.ENGINE_STOP_FINDING == "Batch 1 V2"


def test_every_launcher_of_the_fleet_accepts_a_bare_stop_verb() -> None:
    # The stop argv the plan records is ``<launcher> stop`` with no model
    # argument; every launcher dispatches that verb without one.
    for rel in rc.DEFAULT_LAUNCHER_CMDS.values():
        src = (REPO_ROOT / rel[-1]).read_text(encoding="utf-8")
        assert "    stop)\n" in src, rel
        body = src[src.index("    stop)\n"):]
        body = body[: body.index(";;")]
        assert "stop_server" in body or "stop_stack" in body, rel
        assert '"$2"' not in body, f"{rel}: the stop verb must not need a model"


def test_relaunch_steps_record_their_launcher_and_stop_argv(tmp_path, stub) -> None:
    plan = _plan_for(_grid(f1_engines=("sglang", "vllm")), _floor_table(tmp_path), stub.cmd)
    relaunches = _relaunches(plan)
    assert [r["engine"] for r in relaunches] == ["sglang", "vllm"]
    for step in relaunches:
        assert step["launcher_key"] == step["engine"]
        verb_at = step["argv"].index("restart")
        assert step["stop_argv"] == step["argv"][:verb_at] + [rc.LAUNCHER_STOP_VERB]
        assert step["stop_argv"][:2] == list(stub.cmd)


def test_a_pd_relaunch_records_the_pd_launcher_key(tmp_path, stub) -> None:
    grid = _grid(dist_cells=(("B1", "vllm", "pd"),))
    plan = _plan_for(grid, _floor_table(tmp_path), stub.cmd)
    pd = [r for r in _relaunches(plan) if r["topology"] == "pd"]
    assert len(pd) == 1
    assert pd[0]["launcher_key"] == rc.PD_LAUNCHER_KEY
    verb_at = pd[0]["argv"].index("start")
    assert pd[0]["stop_argv"] == pd[0]["argv"][:verb_at] + ["stop"]


def _relaunch(key: str, tag: str) -> Dict[str, Any]:
    return {
        "kind": "relaunch", "launcher_key": key,
        "stop_argv": ["launch", key, "stop"], "env": {"TAG": tag},
    }


def _cell() -> Dict[str, Any]:
    return {"kind": "cell"}


class TestStopBoundaries:
    def test_no_relaunch_means_no_stop(self) -> None:
        assert rc.stop_boundaries([_cell(), _cell()]) == []
        assert rc.plan_launchers([_cell()]) == []

    def test_one_relaunch_stops_once_at_the_end(self) -> None:
        steps = [_relaunch("vllm", "a"), _cell()]
        assert rc.stop_boundaries(steps) == [
            {"before_index": None, "launcher_key": "vllm",
             "stop_argv": ["launch", "vllm", "stop"], "env": {"TAG": "a"}},
        ]

    def test_same_launcher_relaunches_stop_nothing_between(self) -> None:
        # prefix ON then prefix OFF on vLLM: the launcher's restart is
        # self-cleaning, so the only stop is the end-of-run one, carrying the
        # LAST relaunch's env.
        steps = [_relaunch("vllm", "on"), _cell(), _relaunch("vllm", "off"), _cell()]
        out = rc.stop_boundaries(steps)
        assert [b["before_index"] for b in out] == [None]
        assert out[0]["env"] == {"TAG": "off"}

    def test_a_family_change_stops_the_previous_family_first(self) -> None:
        steps = [_relaunch("sglang", "s"), _cell(), _relaunch("vllm", "v"), _cell()]
        out = rc.stop_boundaries(steps)
        assert out == [
            {"before_index": 2, "launcher_key": "sglang",
             "stop_argv": ["launch", "sglang", "stop"], "env": {"TAG": "s"}},
            {"before_index": None, "launcher_key": "vllm",
             "stop_argv": ["launch", "vllm", "stop"], "env": {"TAG": "v"}},
        ]

    def test_pd_and_single_vllm_are_different_launchers(self) -> None:
        steps = [_relaunch(rc.PD_LAUNCHER_KEY, "pd"), _cell(), _relaunch("vllm", "v"), _cell()]
        out = rc.stop_boundaries(steps)
        assert [b["launcher_key"] for b in out] == [rc.PD_LAUNCHER_KEY, "vllm"]
        assert out[0]["before_index"] == 2

    def test_three_families_in_a_row(self) -> None:
        steps = [
            _relaunch("sglang", "s"), _cell(),
            _relaunch("vllm", "v1"), _cell(), _relaunch("vllm", "v2"), _cell(),
            _relaunch(rc.PD_LAUNCHER_KEY, "pd"), _cell(),
        ]
        out = rc.stop_boundaries(steps)
        assert [(b["before_index"], b["launcher_key"]) for b in out] == [
            (2, "sglang"), (6, "vllm"), (None, rc.PD_LAUNCHER_KEY),
        ]
        assert out[1]["env"] == {"TAG": "v2"}, "the stop carries the env of the LAST relaunch of that family"

    def test_plan_launchers_lists_each_launcher_once_in_first_appearance_order(self) -> None:
        steps = [
            _relaunch("sglang", "s"), _cell(),
            _relaunch("vllm", "v1"), _cell(), _relaunch("vllm", "v2"), _cell(),
            _relaunch("sglang", "s2"), _cell(),
        ]
        out = rc.plan_launchers(steps)
        assert [l["launcher_key"] for l in out] == ["sglang", "vllm"]
        assert out[0]["env"] == {"TAG": "s"} and out[1]["env"] == {"TAG": "v1"}

    def test_the_function_never_mutates_its_input(self) -> None:
        steps = [_relaunch("vllm", "a"), _cell()]
        before = json.dumps(steps, sort_keys=True)
        rc.stop_boundaries(steps)
        rc.plan_launchers(steps)
        assert json.dumps(steps, sort_keys=True) == before


class TestRunPlanStops:
    def test_the_previous_family_is_stopped_before_the_next_relaunch(self, tmp_path, stub) -> None:
        plan = _plan_for(_grid(f1_engines=("sglang", "vllm")), _floor_table(tmp_path), stub.cmd)
        assert rc.run_plan(plan, _run_root(tmp_path)) == 0
        assert _verbs(stub.calls()) == [
            # clean room: every launcher the plan uses, before the first step
            "stop:sglang", "stop:vllm",
            "restart:sglang", "cell",
            # the family changes: stop sglang BEFORE vllm starts
            "stop:sglang",
            "restart:vllm", "cell",
            # end of run: nothing outlives it
            "stop:vllm",
        ]

    def test_same_launcher_relaunches_are_not_separated_by_a_stop(self, tmp_path, stub) -> None:
        # B2 gold-reuse (prefix ON) and B4 corpus-fresh (prefix OFF, ADR-0103):
        # two vLLM relaunches, one family. (B1 gold-fresh serves prefix OFF
        # too since ADR-0150, so it would share B4's boundary.)
        plan = _plan_for(_grid(f1_baselines=("B2", "B4")), _floor_table(tmp_path), stub.cmd)
        assert len(_relaunches(plan)) == 2
        assert rc.run_plan(plan, _run_root(tmp_path)) == 0
        assert _verbs(stub.calls()) == [
            "stop:vllm", "restart:vllm", "cell", "restart:vllm", "cell", "stop:vllm",
        ]

    def test_an_hf_only_plan_stops_nothing(self, tmp_path, stub) -> None:
        grid = _grid(f1_baselines=(), hf_oracle_cells=(("B3", ("squad_v2",)),))
        plan = _plan_for(grid, _floor_table(tmp_path), stub.cmd)
        assert _relaunches(plan) == []
        assert rc.run_plan(plan, _run_root(tmp_path)) == 0
        assert _verbs(stub.calls()) == ["cell"]

    def test_the_stop_carries_the_env_of_its_relaunch(self, tmp_path, stub) -> None:
        plan = _plan_for(_grid(f1_engines=("sglang", "vllm")), _floor_table(tmp_path), stub.cmd)
        assert rc.run_plan(plan, _run_root(tmp_path)) == 0
        sglang_relaunch = next(r for r in _relaunches(plan) if r["engine"] == "sglang")
        stops = [c for c in stub.calls() if c["argv"] == ["stop"] and "SGLANG_PORT" in c["env"]]
        assert len(stops) == 2  # clean room + the boundary
        for call in stops:
            for key, value in sglang_relaunch["env"].items():
                assert call["env"].get(key) == value

    def test_a_failed_stop_is_reported_and_the_run_continues(self, tmp_path, stub, monkeypatch, capsys) -> None:
        plan = _plan_for(_grid(f1_engines=("sglang", "vllm")), _floor_table(tmp_path), stub.cmd)
        monkeypatch.setenv("STUB_FAIL_EXACT", "stop")  # every stop exits 1
        root = _run_root(tmp_path)
        assert rc.run_plan(plan, root) == rc.EXIT_STOP_FAILED, "a failed stop makes the exit code nonzero"
        verbs = _verbs(stub.calls())
        assert verbs.count("cell") == 2, "cells still run after a failed stop"
        assert verbs[-1] == "stop:vllm"
        out = capsys.readouterr().out
        assert "STOP FAILED" in out and rc.ENGINE_STOP_FINDING in out
        # no cell carries a failure sentinel: the cells themselves passed
        for step in _cells(plan):
            assert not (root / "cells" / step["row_key"] / f".STATUS-{step['dataset']}").exists()

    def test_a_failed_stop_does_not_gate_the_seal(self, tmp_path, stub, monkeypatch) -> None:
        plan = _plan_for(_grid(), _floor_table(tmp_path), stub.cmd)
        monkeypatch.setenv("STUB_FAIL_EXACT", "stop")
        root = _run_root(tmp_path)
        assert rc.run_plan(plan, root, seal=True, seal_cmd=stub.cmd) == rc.EXIT_STOP_FAILED
        assert [str(root)] in [c["argv"] for c in stub.calls()], "the sealer ran: the data tree is complete"

    def test_a_failed_relaunch_still_stops_its_family_at_the_end(self, tmp_path, stub, monkeypatch) -> None:
        plan = _plan_for(_grid(), _floor_table(tmp_path), stub.cmd)
        # the default grid's B1 (gold-fresh) relaunches prefix OFF (ADR-0150)
        monkeypatch.setenv(
            "STUB_FAIL_EXACT", f"restart {rc.HF_ID_OF_SLUG['qwen3-14b']} --no-prefix-cache"
        )
        assert rc.run_plan(plan, _run_root(tmp_path)) == 1
        assert _verbs(stub.calls()) == ["stop:vllm", "restart:vllm", "stop:vllm"]


class TestLoadPlanV2:
    def test_a_relaunch_without_the_stop_record_refuses(self, tmp_path, stub) -> None:
        plan = _plan_for(_grid(), _floor_table(tmp_path), stub.cmd)
        for key in ("stop_argv", "launcher_key"):
            broken = json.loads(json.dumps(plan))
            del _relaunches(broken)[0][key]
            with pytest.raises(rc.RunError, match=f"missing key.*{key}"):
                rc.load_plan(_write_plan(tmp_path, broken, f"{key}.json"))

    def test_a_stop_argv_aimed_at_another_launcher_refuses(self, tmp_path, stub) -> None:
        plan = _plan_for(_grid(), _floor_table(tmp_path), stub.cmd)
        broken = json.loads(json.dumps(plan))
        _relaunches(broken)[0]["stop_argv"] = ["bash", "some/other/launcher.sh", "stop"]
        with pytest.raises(rc.RunError, match="stop_argv.*Batch 1 V2"):
            rc.load_plan(_write_plan(tmp_path, broken))

    def test_a_launcher_key_that_contradicts_the_relaunch_refuses(self, tmp_path, stub) -> None:
        plan = _plan_for(_grid(), _floor_table(tmp_path), stub.cmd)
        broken = json.loads(json.dumps(plan))
        _relaunches(broken)[0]["launcher_key"] = "sglang"
        with pytest.raises(rc.RunError, match="launcher_key.*Batch 1 V2"):
            rc.load_plan(_write_plan(tmp_path, broken))

    def test_the_fresh_plan_loads(self, tmp_path, stub) -> None:
        plan = _plan_for(_grid(f1_engines=("sglang", "vllm")), _floor_table(tmp_path), stub.cmd)
        loaded = rc.load_plan(_write_plan(tmp_path, plan))
        assert loaded["counts"]["engine_stops"] == 2  # one boundary + the end of run


def test_plan_counts_name_the_engine_stops(tmp_path, stub) -> None:
    plan = _plan_for(_grid(f1_engines=("sglang", "vllm")), _floor_table(tmp_path), stub.cmd)
    assert plan["counts"]["engine_stops"] == len(rc.stop_boundaries(plan["steps"]))
    assert plan["counts"]["engine_stops"] == 2
    assert plan["counts"]["relaunches"] == 2


# ===========================================================================
# V3: the window bound
# ===========================================================================


def _f2_grid(**overrides: Any):
    kwargs: Dict[str, Any] = dict(
        f2_baselines=("B1",), f2_engines=("vllm",), f2_budgets=(1.0,), f2_rates=(0.85,),
    )
    kwargs.update(overrides)
    return _grid(**kwargs)


def test_v3_constant() -> None:
    assert rc.WINDOW_BOUND_FINDING == "Batch 1 V3"


def test_window_cells_are_bounded_by_arrival_count_equal_to_their_pool(tmp_path, stub) -> None:
    plan = _plan_for(_f2_grid(), _floor_table(tmp_path), stub.cmd)
    windows = [s for s in _cells(plan) if s["row_class"] == "window"]
    others = [s for s in _cells(plan) if s["row_class"] != "window"]
    assert len(windows) == 1 and len(others) == 1
    argv = windows[0]["argv"]
    assert _argv_value(argv, "--workload-mode") == "open_loop"
    assert _argv_value(argv, "--rate") is not None
    assert _argv_value(argv, "--arrival-count") == _argv_value(argv, "--num-queries") == str(rc.WINDOW_REQUESTS)
    assert "--duration-s" not in argv
    for step in others:
        assert "--arrival-count" not in step["argv"] and "--duration-s" not in step["argv"]


def test_achievable_n_lowers_the_arrival_count_with_the_pool(tmp_path, stub) -> None:
    plan = _plan_for(_f2_grid(achievable_n={"qasper": 50}), _floor_table(tmp_path), stub.cmd)
    window = next(s for s in _cells(plan) if s["row_class"] == "window")
    assert _argv_value(window["argv"], "--num-queries") == "50"
    assert _argv_value(window["argv"], "--arrival-count") == "50"


def test_ruler_window_steps_carry_the_same_bound(tmp_path, stub) -> None:
    grid = _f2_grid(f2_ruler_baselines=("B1",), f2_ruler_tasks=("qa",))
    plan = _plan_for(grid, _floor_table(tmp_path), stub.cmd)
    ruler = [s for s in _cells(plan) if s["dataset"] == "ruler"]
    assert len(ruler) == 1
    assert _argv_value(ruler[0]["argv"], "--arrival-count") == str(rc.WINDOW_REQUESTS)
    assert "--duration-s" not in ruler[0]["argv"]


def test_the_header_records_the_window_bound_and_keeps_the_duration_estimate(tmp_path, stub) -> None:
    plan = _plan_for(_f2_grid(), _floor_table(tmp_path), stub.cmd)
    knobs = plan["behavior_knobs"]
    assert knobs["window_bound"] == "arrival-count"
    assert knobs["window_bound_finding"] == rc.WINDOW_BOUND_FINDING
    assert plan["window_duration_s"] == 60.0  # the pre-costed estimate stays recorded


def test_the_arrivals_equal_the_pool_so_the_replay_guard_never_fires() -> None:
    # The runner's own guard (src/orchestration/load_generator.py): W arrivals
    # against W prepared requests is no replay, with no env override.
    schedule = generate_arrival_schedule(1.7, seed=42, n_requests=rc.WINDOW_REQUESTS)
    assert schedule.n_arrivals == rc.WINDOW_REQUESTS
    assert ensure_no_measured_replay(schedule, rc.WINDOW_REQUESTS) is False
    # ... while the old duration form at the same rate refuses on the first
    # window longer than the pool (the finding).
    long = generate_arrival_schedule(1.7, seed=42, duration_s=300.0)
    assert long.n_arrivals > rc.WINDOW_REQUESTS
    with pytest.raises(LoadGeneratorError, match="replay"):
        ensure_no_measured_replay(long, rc.WINDOW_REQUESTS)


def test_the_runner_takes_exactly_one_of_duration_and_arrival_count() -> None:
    src = RUN_EXPERIMENT_PY.read_text(encoding="utf-8")
    assert '"--arrival-count"' in src and "type=int" in src[src.index('"--arrival-count"'):][:200]
    assert "requires exactly one of --duration-s" in src


class TestLoadPlanV3:
    def test_a_window_cell_bounded_by_duration_refuses(self, tmp_path, stub) -> None:
        plan = _plan_for(_f2_grid(), _floor_table(tmp_path), stub.cmd)
        broken = json.loads(json.dumps(plan))
        window = next(s for s in _cells(broken) if s["row_class"] == "window")
        at = window["argv"].index("--arrival-count")
        window["argv"][at:at + 2] = ["--duration-s", "60"]
        with pytest.raises(rc.RunError, match="duration-s.*Batch 1 V3"):
            rc.load_plan(_write_plan(tmp_path, broken))

    def test_an_arrival_count_that_differs_from_the_pool_refuses(self, tmp_path, stub) -> None:
        plan = _plan_for(_f2_grid(), _floor_table(tmp_path), stub.cmd)
        broken = json.loads(json.dumps(plan))
        window = next(s for s in _cells(broken) if s["row_class"] == "window")
        window["argv"][window["argv"].index("--arrival-count") + 1] = "199"
        with pytest.raises(rc.RunError, match="arrival-count.*199.*Batch 1 V3"):
            rc.load_plan(_write_plan(tmp_path, broken))

    def test_a_window_cell_without_a_bound_refuses(self, tmp_path, stub) -> None:
        plan = _plan_for(_f2_grid(), _floor_table(tmp_path), stub.cmd)
        broken = json.loads(json.dumps(plan))
        window = next(s for s in _cells(broken) if s["row_class"] == "window")
        at = window["argv"].index("--arrival-count")
        del window["argv"][at:at + 2]
        with pytest.raises(rc.RunError, match="lacks --arrival-count.*Batch 1 V3"):
            rc.load_plan(_write_plan(tmp_path, broken))

    def test_a_per_query_cell_carrying_a_bound_refuses(self, tmp_path, stub) -> None:
        plan = _plan_for(_f2_grid(), _floor_table(tmp_path), stub.cmd)
        broken = json.loads(json.dumps(plan))
        f1 = next(s for s in _cells(broken) if s["row_class"] != "window")
        f1["argv"] += ["--arrival-count", "200"]
        with pytest.raises(rc.RunError, match="arrival-count.*Batch 1 V3"):
            rc.load_plan(_write_plan(tmp_path, broken))

    def test_the_fresh_plan_loads(self, tmp_path, stub) -> None:
        plan = _plan_for(_f2_grid(achievable_n={"qasper": 50}), _floor_table(tmp_path), stub.cmd)
        rc.load_plan(_write_plan(tmp_path, plan))


# ===========================================================================
# Review of S1-A (fresh Fable 5.1 subagent, 2026-10-05): the code findings
# ===========================================================================


class TestReviewFindings:
    def test_f1_an_abort_mid_run_stops_the_resident_family(self, tmp_path, stub, monkeypatch) -> None:
        # A Ctrl-C (or any exception) during a cell must not leave the engine
        # resident: the end-of-run stop runs from a finally block and targets
        # the family running NOW (V1 review F-2: plan a runs sglang before
        # vllm, so an abort in the SGLang half must stop SGLang, not the
        # plan's last family, vllm).
        plan = _plan_for(_grid(f1_engines=("sglang", "vllm")), _floor_table(tmp_path), stub.cmd)
        real_exec = rc._exec
        seen: List[str] = []

        def _exec(argv, env):
            if "--baseline" in argv:
                seen.append("cell")
                if seen.count("cell") == 1:
                    raise KeyboardInterrupt
                return real_exec(argv, env)
            engine = "vllm" if "VLLM_PORT" in env else "sglang" if "SGLANG_PORT" in env else "?"
            seen.append(f"{argv[-1] if argv[-1] == 'stop' else 'restart'}:{engine}")
            return real_exec(argv, env)

        monkeypatch.setattr(rc, "_exec", _exec)
        with pytest.raises(KeyboardInterrupt):
            rc.run_plan(plan, _run_root(tmp_path))
        assert seen == ["stop:sglang", "stop:vllm", "restart:sglang", "cell", "stop:sglang"], seen

    def test_f1_an_abort_before_any_relaunch_stops_nothing_more(self, tmp_path, stub, monkeypatch) -> None:
        # hf cells run before the first relaunch; an abort there has no
        # resident family to stop (the clean room already ran).
        grid = _grid(f1_baselines=("B1",), hf_oracle_cells=(("B3", ("squad_v2",)),))
        plan = _plan_for(grid, _floor_table(tmp_path), stub.cmd)
        real_exec = rc._exec
        seen: List[str] = []

        def _exec(argv, env):
            seen.append("cell" if "--baseline" in argv else argv[-1] if argv[-1] == "stop" else "restart")
            if seen[-1] == "cell":
                raise KeyboardInterrupt
            return real_exec(argv, env)

        monkeypatch.setattr(rc, "_exec", _exec)
        with pytest.raises(KeyboardInterrupt):
            rc.run_plan(plan, _run_root(tmp_path))
        assert seen == ["stop", "cell"], seen

    def test_f1_the_final_stop_runs_once_on_the_happy_path(self, tmp_path, stub) -> None:
        plan = _plan_for(_grid(), _floor_table(tmp_path), stub.cmd)
        assert rc.run_plan(plan, _run_root(tmp_path)) == 0
        assert _verbs(stub.calls()).count("stop:vllm") == 2  # clean room + end of run, no double

    def test_f4_a_stop_only_failure_returns_2_and_names_it_beside_the_seal(self, tmp_path, stub, monkeypatch, capsys) -> None:
        plan = _plan_for(_grid(), _floor_table(tmp_path), stub.cmd)
        monkeypatch.setenv("STUB_FAIL_EXACT", "stop")
        root = _run_root(tmp_path)
        assert rc.run_plan(plan, root, seal=True, seal_cmd=stub.cmd) == rc.EXIT_STOP_FAILED
        assert rc.EXIT_STOP_FAILED == 2
        out = capsys.readouterr().out
        assert "sealed" in out and "engine stop(s) failed" in out

    def test_f4_a_failed_cell_still_returns_1_even_with_a_failed_stop(self, tmp_path, stub, monkeypatch) -> None:
        plan = _plan_for(_grid(), _floor_table(tmp_path), stub.cmd)
        first = _cells(plan)[0]
        cell_argv = " ".join(first["argv"][2:] + ["--campaign-root", str(_run_root(tmp_path))])
        monkeypatch.setenv("STUB_FAIL_EXACT", cell_argv)
        assert rc.run_plan(plan, _run_root(tmp_path)) == 1

    def test_f5_a_relaunch_argv_typed_as_a_string_refuses_not_crashes(self, tmp_path, stub) -> None:
        plan = _plan_for(_grid(), _floor_table(tmp_path), stub.cmd)
        broken = json.loads(json.dumps(plan))
        step = _relaunches(broken)[0]
        step["argv"] = " ".join(step["argv"])
        with pytest.raises(rc.RunError, match="argv must be a list"):
            rc.load_plan(_write_plan(tmp_path, broken))

    def test_f6_a_launcher_prefix_token_equal_to_a_verb_is_not_the_verb(self, tmp_path, stub) -> None:
        # ``--launcher-cmd "bash start"`` style prefix: the verb is the token
        # right after the prefix, never the first occurrence of the word.
        plan = _plan_for(_grid(), _floor_table(tmp_path), stub.cmd,
                         launcher_cmds={"vllm": ("bash", "start"), "sglang": ("bash", "start"),
                                        rc.PD_LAUNCHER_KEY: ("bash", "start")})
        step = _relaunches(plan)[0]
        assert step["argv"][:3] == ["bash", "start", "restart"]
        assert step["stop_argv"] == ["bash", "start", "stop"]
        rc.load_plan(_write_plan(tmp_path, plan))

    def test_f7_cell_step_no_longer_takes_the_window_duration(self) -> None:
        import inspect

        assert "window_duration_s" not in inspect.signature(rc._cell_step).parameters

    def test_f8_the_run_summary_separates_clean_room_and_boundary_stops(self, tmp_path, stub, capsys) -> None:
        plan = _plan_for(_grid(f1_engines=("sglang", "vllm")), _floor_table(tmp_path), stub.cmd)
        assert rc.run_plan(plan, _run_root(tmp_path)) == 0
        out = capsys.readouterr().out
        assert "clean room 2" in out and "boundaries 2" in out and "0 failed" in out
        assert plan["counts"]["engine_stops"] == 2
