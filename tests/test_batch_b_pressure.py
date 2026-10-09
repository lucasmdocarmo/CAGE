"""Batch B (2026-10-09): the driver side of the pressure regime repairs.

ADR-0153 (S0F-61): the queue clause of the §6.1 in-regime rule lives in
``src/analysis`` (tested there); HERE the dry window the driver runs per
(engine, demand class) before that pair's pressure cells, its sibling root,
its gate on the regime label and the exit code 3.
ADR-0154 (S0F-62): the rung-calibration artifact, its registration on the
plan (every executable pressure rung ESTIMATED), the offered rate as
rate_frac x the measured lambda*, and ``calibrate-rungs`` (the cal-v2
ladder run through the runner under the plan's own relaunch per rung).
ADR-0155 (S0F-63): one byte budget per demand class at each r.

Pure of RunPod: the runner and the launchers are the recording stub of
``test_run_campaign`` or an in-process fake (``exec_fn`` seam).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_run_campaign import (  # noqa: E402
    _ANCHOR_DEMAND,
    _ANCHOR_SEQ_TOKENS,
    _RUNG_LAMBDA,
    _cell_calls,
    _cells,
    _dry_calls,
    _dump,
    _floor_table_doc,
    _relaunches,
    _rung_calibration_doc,
    _rungs_for,
    _run_root,
    _stub_plan,
    _tiny_grid,
    _write_calibrations,
    _write_rung_calibrations,
    calibrations_a,  # noqa: F401  (fixture re-export)
    floor_table,  # noqa: F401
    rc,
    stub,  # noqa: F401
)
from src.orchestration.calibration import (  # noqa: E402
    PROBE_ATTAINMENT_MIN,
    PROBE_BISECT_STEPS,
    PROBE_LADDER_FACTOR,
    PROBE_MAX_STEPS,
    PROBE_WARMUP_S,
    PROBE_WINDOW_S,
    FloorMeasurement,
    floor_start_qps,
)
from src.orchestration.campaign_session import derive_cell_spec  # noqa: E402

PRESSURE = ("F2", "F3")


@pytest.fixture(scope="module")
def plan_a(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    """Session a, built once for the read-only plan assertions below."""
    tmp = tmp_path_factory.mktemp("plan_a")
    floor = tmp / "ft.json"
    floor.write_text(json.dumps(_floor_table_doc()), encoding="utf-8")
    cal = _write_calibrations(tmp / "cal", ("vllm", "sglang"))
    return rc.build_plan(
        "a", rc.load_floor_table(floor), window_duration_s=300.0,
        calibrations=cal, rung_calibrations=_rungs_for(cal, "a"),
    )


def _pressure_cells(plan: Dict[str, Any], *, executable: bool = True) -> List[Dict[str, Any]]:
    return [
        s for s in _cells(plan)
        if s["family"] in PRESSURE and (s["blocked_on"] is None) == executable
    ]


def _dry_steps(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [s for s in plan["steps"] if s["kind"] == "dry_window"]


def _write_doc(tmp_path: Path, doc: Dict[str, Any], name: str = "rungs.json") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Budgeted rungs and the --rungs argument
# ---------------------------------------------------------------------------


class TestBudgetedRungs:
    def test_union_of_f2_fine_and_f3_descending(self):
        assert rc.budgeted_rungs(rc.SESSION_GRIDS["a"]) == (1.5, 1.25, 1.0, 0.75, 0.5, 0.375, 0.25)
        assert rc.budgeted_rungs(rc.SESSION_GRIDS["b"]) == (1.0, 0.5, 0.25)
        assert rc.budgeted_rungs(_tiny_grid()) == ()
        grid = _tiny_grid(
            f2_baselines=("B1",), f2_budgets=(0.5, 1.0), f2_rates=(0.85,),
            f3_baselines=("B2",), f3_budgets=(0.5, 0.25), f3_rates=(0.85,),
        )
        assert rc.budgeted_rungs(grid) == (1.0, 0.5, 0.25)

    def test_rungs_cli_prints_them(self, capsys):
        assert rc.main(["rungs", "--session", "a"]) == 0
        assert capsys.readouterr().out == "1.5 1.25 1 0.75 0.5 0.375 0.25\n"
        assert rc.main(["rungs", "--session", "zz"]) == 2

    def test_parse_rungs_arg(self):
        assert rc.parse_rungs_arg(None) is None
        assert rc.parse_rungs_arg("  ") is None
        assert rc.parse_rungs_arg("1.5, 0.25,1") == (1.5, 0.25, 1.0)
        for bad in ("x", "0", "-1", "nan", "inf", "1,1", ","):
            with pytest.raises(rc.PlanError, match="--rungs"):
                rc.parse_rungs_arg(bad)


# ---------------------------------------------------------------------------
# The rung-calibration artifact (ADR-0154)
# ---------------------------------------------------------------------------


class TestRungArtifact:
    def test_loads_and_normalizes(self, tmp_path):
        doc = _rung_calibration_doc("lmdeploy-turbomind", "a", (1.0, 0.5), labels={0.5: "LADDER_EXHAUSTED"})
        cal = rc.load_rung_calibration(_write_doc(tmp_path, doc))
        assert cal.engine == "lmdeploy" and cal.session == "a" and cal.model == "Qwen/Qwen3-14B"
        assert cal.lambdas == {1.0: _RUNG_LAMBDA}
        assert cal.labels == {1.0: "ESTIMATED", 0.5: "LADDER_EXHAUSTED"}
        assert cal.lambda_at(1.0) == _RUNG_LAMBDA and cal.lambda_at(0.5) is None
        assert cal.label_at(0.5) == "LADDER_EXHAUSTED" and cal.label_at(0.75) is None
        assert len(cal.sha256) == 64

    @pytest.mark.parametrize(
        "spoil, match",
        [
            (lambda d: d.update(schema="cage-rung-calibration-v0"), "schema"),
            (lambda d: d.update(procedure_version="cal-v1"), "procedure_version"),
            (lambda d: d.update(confirmatory=True), "confirmatory"),
            (lambda d: d.update(engine="hf-oracle"), "oracle"),
            (lambda d: d.update(engine="nope"), "not a runner backend"),
            (lambda d: d.update(model=""), "model"),
            (lambda d: d.update(session="zz"), "session"),
            (lambda d: d.update(rungs={}), "non-empty"),
            (lambda d: d["rungs"].update({"abc": {"r": 1, "label": "ESTIMATED", "lambda_star_qps": 1.0}}), "does not parse"),
            (lambda d: d["rungs"]["1"].update(r=1.25), "disagrees"),
            (lambda d: d["rungs"]["1"].update(lambda_star_qps=None), "finite number"),
            (lambda d: d["rungs"]["1"].update(lambda_star_qps=float("nan")), "finite number"),
            (lambda d: d["rungs"]["0.5"].update(label="NONE_SUSTAINABLE", lambda_star_qps=3.0), "labels are honest"),
            (lambda d: d["rungs"]["0.5"].update(label=""), "non-empty string"),
        ],
    )
    def test_refusals(self, tmp_path, spoil, match):
        doc = _rung_calibration_doc("vllm", "a", (1.0, 0.5))
        spoil(doc)
        with pytest.raises(rc.PlanError, match=match):
            rc.load_rung_calibration(_write_doc(tmp_path, doc))

    def test_missing_and_malformed_files(self, tmp_path):
        with pytest.raises(rc.PlanError, match="not found"):
            rc.load_rung_calibration(tmp_path / "absent.json")
        bad = tmp_path / "bad.json"
        bad.write_text("{", encoding="utf-8")
        with pytest.raises(rc.PlanError, match="not valid JSON"):
            rc.load_rung_calibration(bad)


class TestRungRegistration:
    def _build(self, floor_table: Path, cal: Dict[str, Path], rungs: Optional[Dict[str, Path]]):
        return rc.build_plan(
            "a", rc.load_floor_table(floor_table), window_duration_s=300.0,
            calibrations=cal, rung_calibrations=rungs,
        )

    def test_every_engine_with_executable_pressure_cells_needs_an_artifact(
        self, floor_table, calibrations_a, tmp_path
    ):
        with pytest.raises(rc.PlanError, match=r"missing for engine\(s\) \['sglang', 'vllm'\]"):
            self._build(floor_table, calibrations_a, None)
        only_vllm = _write_rung_calibrations(
            tmp_path / "v", ("vllm",), "a", rc.budgeted_rungs(rc.SESSION_GRIDS["a"])
        )
        with pytest.raises(rc.PlanError, match=r"missing for engine\(s\) \['sglang'\]"):
            self._build(floor_table, calibrations_a, only_vllm)

    def test_artifact_identity_refusals(self, floor_table, calibrations_a, tmp_path):
        rungs = _rungs_for(calibrations_a, "a")
        with pytest.raises(rc.PlanError, match="not a server engine"):
            self._build(floor_table, calibrations_a, dict(rungs, lmdeploy=rungs["vllm"]))
        with pytest.raises(rc.PlanError, match="describes engine"):
            self._build(floor_table, calibrations_a, {"vllm": rungs["sglang"], "sglang": rungs["vllm"]})
        llama = _rungs_for(calibrations_a, "a", model="meta-llama/Llama-3.3-70B-Instruct")
        with pytest.raises(rc.PlanError, match="is for model"):
            self._build(floor_table, calibrations_a, llama)
        other = _write_rung_calibrations(
            tmp_path / "other", ("vllm", "sglang"), "b", rc.budgeted_rungs(rc.SESSION_GRIDS["a"])
        )
        with pytest.raises(rc.PlanError, match="measured for session 'b'"):
            self._build(floor_table, calibrations_a, other)

    def test_every_executable_rung_must_be_estimated(self, floor_table, calibrations_a, tmp_path):
        all_rungs = rc.budgeted_rungs(rc.SESSION_GRIDS["a"])
        short = _write_rung_calibrations(tmp_path / "short", ("vllm", "sglang"), "a", all_rungs[:-1])
        with pytest.raises(rc.PlanError, match=r"r=0\.25 \(absent\)"):
            self._build(floor_table, calibrations_a, short)
        exhausted = _write_rung_calibrations(
            tmp_path / "exh", ("vllm", "sglang"), "a", all_rungs,
            labels={0.25: "LADDER_EXHAUSTED", 1.5: "NONE_SUSTAINABLE"},
        )
        with pytest.raises(rc.PlanError, match=r"r=1\.5 \(NONE_SUSTAINABLE\), r=0\.25 \(LADDER_EXHAUSTED\)"):
            self._build(floor_table, calibrations_a, exhausted)

    def test_header_records_the_registration(self, plan_a):
        header = plan_a["rung_calibration"]
        assert header["schema"] == rc.RUNG_CALIBRATION_SCHEMA
        assert header["adr"] == "ADR-0154" and header["finding"] == "S0F-62"
        assert header["anchor_arm"] == "gold-fresh" and header["anchor_seq_tokens"] == _ANCHOR_SEQ_TOKENS
        assert header["required_rungs"] == {
            "sglang": [1.5, 1.25, 1.0, 0.75, 0.5, 0.375, 0.25],
            "vllm": [1.5, 1.25, 1.0, 0.75, 0.5, 0.375, 0.25],
        }
        for engine in ("vllm", "sglang"):
            art = header["artifacts"][engine]
            assert len(art["sha256"]) == 64 and art["session"] == "a"
            assert art["rungs"] == {f"{r:g}": _RUNG_LAMBDA * r for r in (1.5, 1.25, 1.0, 0.75, 0.5, 0.375, 0.25)}
            assert set(art["labels"].values()) == {"ESTIMATED"}

    def test_cli_registers_and_refuses_malformed(self, tmp_path, floor_table, calibrations_a):
        rungs = _rungs_for(calibrations_a, "a")
        out = tmp_path / "plan.json"
        base = ["plan", "--session", "a", "--floor-table", str(floor_table), "--window-duration-s", "300",
                "--out", str(out)]
        for engine, path in calibrations_a.items():
            base += ["--calibration", f"{engine}={path}"]
        assert rc.main(base) == 2  # no --rung-calibration: refused
        assert not out.exists()
        good: List[str] = []
        for engine, path in rungs.items():
            good += ["--rung-calibration", f"{engine}={path}"]
        assert rc.main(base + good) == 0
        assert rc.load_plan(out)["rung_calibration"]["artifacts"].keys() == {"vllm", "sglang"}
        assert rc.main(base + good + ["--rung-calibration", str(rungs["vllm"])]) == 2
        assert rc.main(base + good + ["--rung-calibration", f"vllm={rungs['vllm']}"]) == 2


# ---------------------------------------------------------------------------
# Demand classes (ADR-0155)
# ---------------------------------------------------------------------------


class TestDemandClasses:
    def test_header_records_the_classes(self, plan_a):
        header = plan_a["demand_classes"]
        assert header["adr"] == "ADR-0155" and header["finding"] == "S0F-63"
        assert header["anchor_arm"] == "gold-fresh"
        assert header["anchor_seq_tokens"] == header["floor_avg_seq_tokens"] == _ANCHOR_SEQ_TOKENS
        assert header["seq_tokens"] == dict(sorted(rc.DEMAND_SEQ_TOKENS_2026_10_08.items()))
        assert header["corpus_trunc_seq_tokens"] == {"1400": 1205, "700": 480}

    def test_every_budgeted_relaunch_sizes_its_class(self, plan_a):
        seen = set()
        for s in _relaunches(plan_a):
            if s["budget_r"] is None and s["topology"] == "single":
                assert s["demand_class"] is None
                continue
            dc = s["demand_class"]
            seq = dc["seq_tokens"]
            assert dc["anchor_seq_tokens"] == _ANCHOR_SEQ_TOKENS and dc["adr"] == "ADR-0155"
            assert dc["demand_bytes"] == (_ANCHOR_DEMAND * seq) // _ANCHOR_SEQ_TOKENS
            assert s["budget_bytes"] == int(s["budget_r"] * dc["demand_bytes"])
            assert s["budget_plan"]["avg_seq_tokens"] == seq
            assert s["budget_plan"]["demand_bytes"] == dc["demand_bytes"]
            seen.add(seq)
        assert seen == {4779, 2336, 1127, 1126, 702, 348}

    def test_every_budgeted_cell_carries_its_relaunch_class(self, plan_a):
        current = None
        for s in plan_a["steps"]:
            if s["kind"] == "relaunch":
                current = s
            elif s["kind"] == "cell" and s["demand_class"] is not None and s["blocked_on"] is None:
                assert s["demand_class"] == current["demand_class"], s["row_key"]
                assert s["demand_class"]["seq_tokens"] == rc.demand_seq_tokens(
                    rc.SESSION_GRIDS["a"], rc.CellSpec.from_flat_dict(s["cellspec"])
                )

    def test_classes_sort_largest_first_within_a_budget(self, plan_a):
        # the anchor leads each budget; the B12 rungs keep their descending ladder
        order = [
            (s["engine"], s["prefix_mode"], s["budget_r"], s["demand_class"]["seq_tokens"])
            for s in _relaunches(plan_a) if s["budget_r"] is not None
        ]
        for engine in ("vllm", "sglang"):
            off = [o for o in order if o[0] == engine and o[1] == "OFF"]
            assert off[0][2:] == (0.25, 4779), off[:3]  # tightest budget first, gold first
            per_budget: Dict[float, List[int]] = {}
            for _e, _p, r, seq in off:
                per_budget.setdefault(r, []).append(seq)
            for r, seqs in per_budget.items():
                assert seqs == sorted(seqs, reverse=True), (r, seqs)

    def test_floor_shape_must_match_the_registration(self, calibrations_a, tmp_path):
        rungs = _rungs_for(calibrations_a, "a")
        doc = _floor_table_doc()
        doc["generated_inputs"]["avg_seq_tokens"] = 3100
        with pytest.raises(rc.PlanError, match="sized on avg_seq_tokens=3100"):
            rc.build_plan("a", rc.load_floor_table(_write_doc(tmp_path, doc, "ft3100.json")),
                          window_duration_s=300.0, calibrations=calibrations_a, rung_calibrations=rungs)
        del doc["generated_inputs"]["avg_seq_tokens"]
        with pytest.raises(rc.PlanError, match="carries no generated_inputs.avg_seq_tokens"):
            rc.build_plan("a", rc.load_floor_table(_write_doc(tmp_path, doc, "ftnone.json")),
                          window_duration_s=300.0, calibrations=calibrations_a, rung_calibrations=rungs)

    def test_grid_registration_refusals(self):
        with pytest.raises(rc.PlanError, match="demand_seq_tokens"):
            _tiny_grid(demand_seq_tokens={"corpus-fresh": 2336})  # no anchor
        with pytest.raises(rc.PlanError, match="demand_seq_tokens"):
            _tiny_grid(demand_seq_tokens=dict(rc.DEMAND_SEQ_TOKENS_2026_10_08, **{"gold-fresh": 0}))
        with pytest.raises(rc.PlanError, match="corpus_trunc_demand_seq_tokens"):
            _tiny_grid(corpus_trunc_demand_seq_tokens={1400: 1205})  # rung 700 has no shape

    def test_pressure_free_grid_has_no_classes_and_no_dry_windows(self, floor_table, stub):
        plan = _stub_plan(_tiny_grid(), floor_table, stub.cmd)
        assert plan["counts"]["dry_windows"] == 0 and _dry_steps(plan) == []
        assert all(s["demand_class"] is None for s in plan["steps"])
        assert plan["rung_calibration"]["required_rungs"] == {}
        assert plan["rung_calibration"]["artifacts"] == {}


# ---------------------------------------------------------------------------
# The plan's dry windows (ADR-0153)
# ---------------------------------------------------------------------------


class TestDryWindowPlan:
    def test_one_per_serving_configuration_before_its_first_pressure_cell(self, plan_a):
        dry = _dry_steps(plan_a)
        keys = [rc.dry_window_key(s["serving"], s["demand_class"]) for s in dry]
        assert len(keys) == len(set(keys)) == 19
        assert plan_a["counts"]["dry_windows"] == 19
        assert plan_a["dry_window"]["key"] == [
            "engine", "prefix_mode", "seq_tokens", "kv_dtype", "connector", "topology", "budget_r",
        ]
        assert plan_a["dry_window"]["pairs"] == [[*k, 0.25] for k in keys]
        # the executable pressure configurations ARE the dry-window keys
        configs = {
            rc.dry_window_key(s["serving"], s["demand_class"])
            for s in plan_a["steps"]
            if s["kind"] == "cell" and s["family"] in PRESSURE and s["blocked_on"] is None
        }
        assert configs == set(keys)
        # prefix ON and OFF at one class are separate gates; retr-store (lmcache)
        # is its own gate on vLLM and absent on sglang (blocked there)
        assert ("vllm", "OFF", 1127, None, None, "single") in configs
        assert ("vllm", "ON", 1127, None, "lmcache", "single") in configs
        assert ("vllm", "ON", 1126, None, None, "single") in configs
        assert not any(k[0] == "sglang" and k[4] == "lmcache" for k in configs)
        first_pressure_index: Dict[Tuple[Any, ...], int] = {}
        for s in plan_a["steps"]:
            if s["kind"] == "cell" and s["family"] in PRESSURE and s["blocked_on"] is None:
                first_pressure_index.setdefault(rc.dry_window_key(s["serving"], s["demand_class"]), s["index"])
        for s, key in zip(dry, keys):
            assert s["index"] < first_pressure_index[key]
            assert s["budget_r"] == 0.25  # the tightest r the configuration is carried at

    def test_dry_step_is_the_pressure_cell_at_rate_frac_095(self, plan_a):
        steps = plan_a["steps"]
        for s in _dry_steps(plan_a):
            following = next(
                t for t in steps[s["index"] + 1:]
                if t["kind"] == "cell" and t["family"] in PRESSURE and t["blocked_on"] is None
            )
            assert following["cellspec"]["engine"] == s["engine"]
            assert following["demand_class"] == s["demand_class"]
            assert rc.dry_window_key(following["serving"], following["demand_class"]) == rc.dry_window_key(
                s["serving"], s["demand_class"]
            )
            assert following["lambda_star_rps"] == s["lambda_star_rps"]
            assert following["serving"] == s["serving"] and following["budget_plan"] == s["budget_plan"]
            assert s["rate_frac"] == 0.95 and s["expected_label"] == "IN_REGIME"
            assert s["offered_rate_rps"] == pytest.approx(0.95 * s["lambda_star_rps"])
            spec = rc.CellSpec.from_flat_dict(s["cellspec"])
            assert spec.rate_frac == 0.95 and spec.to_row_key() == s["row_key"]
            assert s["row_key"] != following["row_key"]
            argv = s["argv"]
            assert float(argv[argv.index("--rate") + 1]) == pytest.approx(s["offered_rate_rps"], rel=1e-5)
            assert argv[argv.index("--num-trials") + 1] == "1"
            # a gate, not data: duration mode for the probe window length with
            # the pool replayed (review MEDIUM 6), never the V3 arrival bound
            assert argv[argv.index("--duration-s") + 1] == f"{rc.DRY_WINDOW_DURATION_S:g}"
            assert "--arrival-count" not in argv and "--open-loop-warmup-s" not in argv
            assert s["duration_s"] == rc.DRY_WINDOW_DURATION_S == PROBE_WINDOW_S and s["replay"] is True
            assert s["env"][rc.LADDER_REPLAY_ENV] == "1"
            assert s["env"]["CAGE_CELL_RATE_FRAC"] == "0.95"
            assert "CAGE_WINDOW_ORDINAL_BASE" not in s["env"]
            assert s["windows"] == 1 and s["gate"] and s["adr"] == "ADR-0153"
            # the preceding relaunch is the class's budgeted one
            before = next(t for t in reversed(steps[: s["index"]]) if t["kind"] == "relaunch")
            assert before["engine"] == s["engine"] and before["demand_class"] == s["demand_class"]
            assert before["budget_r"] == 0.25
            for k in ("prefix_mode", "kv_dtype", "connector", "topology"):
                assert before[k] == s["serving"][k]

    def test_header(self, plan_a):
        header = plan_a["dry_window"]
        assert header["adr"] == "ADR-0153" and header["finding"] == "S0F-61"
        assert header["rate_frac"] == 0.95 and header["expected_label"] == "IN_REGIME"
        assert header["root_suffix"] == "-dry" and header["exit_code"] == 3 and header["count"] == 19

    def test_dry_root_is_a_sibling_keyed_on_the_plan(self, plan_a):
        root = rc.dry_window_root(Path("results/c/a/run-001"), plan_a)
        assert root.parent == Path("results/c/a") and root.name.startswith("run-001-dry-")
        assert len(root.name) == len("run-001-dry-") + 6 and rc.RUN_ID_RE.match(root.name)
        # the same plan maps to the same root; other rung artifacts to another
        assert rc.dry_window_root(Path("results/c/a/run-001"), plan_a) == root
        other = json.loads(json.dumps(plan_a))
        other["rung_calibration"]["artifacts"]["vllm"]["sha256"] = "f" * 64
        assert rc.dry_window_root(Path("results/c/a/run-001"), other) != root
        # a cd-act run id (38 chars) plus '-dry' alone would break the §1 grammar;
        # the name truncates the run id to 30 and stays inside it
        long_id = "20261008-132328-cd-act1-qwen3-next-80b"
        assert len(long_id) == 38
        name = rc.dry_window_root(Path(f"results/c/cd-act1/{long_id}"), plan_a).name
        assert rc.RUN_ID_RE.match(name) and name.startswith(long_id[:30] + "-dry-")
        # no plan: a root with no rung artifacts, still a valid id
        assert rc.RUN_ID_RE.match(rc.dry_window_root(Path("results/c/a/run-001")).name)


# ---------------------------------------------------------------------------
# load_plan: the ADR-0153/0154/0155 stale clauses
# ---------------------------------------------------------------------------


class TestLoadPlanBatchB:
    @staticmethod
    def _copy(plan):
        return json.loads(json.dumps(plan))

    def test_fresh_plan_loads(self, plan_a, tmp_path):
        loaded = rc.load_plan(_dump(tmp_path, plan_a, "fresh.json"))
        assert loaded["counts"]["dry_windows"] == 19

    def test_header_clauses(self, plan_a, tmp_path):
        plan = self._copy(plan_a)
        del plan["rung_calibration"]
        with pytest.raises(rc.RunError, match="no rung_calibration header"):
            rc.load_plan(_dump(tmp_path, plan, "no_rung.json"))
        plan = self._copy(plan_a)
        del plan["demand_classes"]
        with pytest.raises(rc.RunError, match="no demand_classes header"):
            rc.load_plan(_dump(tmp_path, plan, "no_classes.json"))

    @pytest.mark.parametrize(
        "spoil, match",
        [
            (lambda s: s.update(rate_basis="kv-bound-only [pending calibration]"), "rate_basis is"),
            (lambda s: s.update(lambda_star_source=None), "no lambda_star_source"),
            (lambda s: s.update(lambda_star_rps=None), "finite number > 0"),
            (lambda s: s.update(offered_rate_rps=s["offered_rate_rps"] * 1.01), "!= rate_frac x lambda_star_rps"),
            (lambda s: s["argv"].__setitem__(s["argv"].index("--rate") + 1, "99"), "carries --rate"),
            (lambda s: s.update(demand_class=None), "must name its served-token class"),
        ],
    )
    def test_pressure_cell_clauses(self, plan_a, tmp_path, spoil, match):
        plan = self._copy(plan_a)
        cell = next(s for s in _cells(plan) if s["family"] == "F2" and s["blocked_on"] is None)
        spoil(cell)
        with pytest.raises(rc.RunError, match=match):
            rc.load_plan(_dump(tmp_path, plan, "cell.json"))

    def test_non_pressure_cell_carries_no_rate_basis(self, plan_a, tmp_path):
        plan = self._copy(plan_a)
        cell = next(s for s in _cells(plan) if s["family"] == "F1" and s["serving"] is not None)
        cell["lambda_star_rps"] = 1.0
        with pytest.raises(rc.RunError, match="only pressure cells offer a rate"):
            rc.load_plan(_dump(tmp_path, plan, "f1_rate.json"))
        plan = self._copy(plan_a)
        cell = next(s for s in _cells(plan) if s["family"] == "F1" and s["serving"] is not None)
        cell["demand_class"] = {"seq_tokens": 4779}
        with pytest.raises(rc.RunError, match="absence stays absence"):
            rc.load_plan(_dump(tmp_path, plan, "f1_class.json"))

    def test_relaunch_clauses(self, plan_a, tmp_path):
        plan = self._copy(plan_a)
        step = next(s for s in _relaunches(plan) if s["budget_r"] is not None)
        step["demand_class"] = None
        with pytest.raises(rc.RunError, match="must name its served-token class"):
            rc.load_plan(_dump(tmp_path, plan, "rl_none.json"))
        plan = self._copy(plan_a)
        step = next(s for s in _relaunches(plan) if s["budget_r"] is not None)
        step["budget_plan"]["avg_seq_tokens"] = step["demand_class"]["seq_tokens"] + 1
        with pytest.raises(rc.RunError, match="!= the relaunch demand class"):
            rc.load_plan(_dump(tmp_path, plan, "rl_drift.json"))
        plan = self._copy(plan_a)
        step = next(s for s in _relaunches(plan) if s["budget_r"] is None)
        step["demand_class"] = {"seq_tokens": 4779}
        with pytest.raises(rc.RunError, match="absence stays absence"):
            rc.load_plan(_dump(tmp_path, plan, "rl_free.json"))

    @pytest.mark.parametrize(
        "spoil, match",
        [
            (lambda s: s.update(expected_label="UNPRESSURED"), "registered gate"),
            (lambda s: s.update(rate_frac=0.85), "registered gate"),
            (lambda s: s["argv"].__setitem__(s["argv"].index("--rate") + 1, "99"), r"is not 0\.95 x"),
            (lambda s: s["argv"].__setitem__(s["argv"].index("--num-trials") + 1, "3"), "exactly one window"),
            (lambda s: s.update(row_key="tampered"), "not minted from its cellspec"),
            (lambda s: s["demand_class"].update(seq_tokens=1), "differs from its relaunch"),
            (lambda s: s.update(engine="sglang"), "no budgeted relaunch of its engine"),
            (lambda s: s["serving"].update(prefix_mode="ON" if s["serving"]["prefix_mode"] == "OFF" else "OFF"), "not the serving configuration of the relaunch"),
            (lambda s: s["argv"].__setitem__(s["argv"].index("--duration-s") + 1, "30"), "duration mode"),
            (lambda s: s["argv"].extend(["--arrival-count", "50"]), "duration mode"),
            (lambda s: s["argv"].extend(["--open-loop-warmup-s", "10"]), "duration mode"),
            (lambda s: s["env"].pop(rc.LADDER_REPLAY_ENV), "carries no CAGE_ALLOW_REPLAY=1"),
        ],
    )
    def test_dry_step_clauses(self, plan_a, tmp_path, spoil, match):
        plan = self._copy(plan_a)
        step = next(s for s in _dry_steps(plan) if s["engine"] == "vllm")
        spoil(step)
        with pytest.raises(rc.RunError, match=match):
            rc.load_plan(_dump(tmp_path, plan, "dry.json"))

    def test_dry_step_needs_every_key_and_a_known_kind(self, plan_a, tmp_path):
        plan = self._copy(plan_a)
        del _dry_steps(plan)[0]["lambda_star_rps"]
        with pytest.raises(rc.RunError, match="missing key"):
            rc.load_plan(_dump(tmp_path, plan, "dry_missing.json"))
        plan = self._copy(plan_a)
        _dry_steps(plan)[0]["kind"] = "wet_window"
        with pytest.raises(rc.RunError, match="kind must be one of"):
            rc.load_plan(_dump(tmp_path, plan, "dry_kind.json"))

    def test_v5_plan_refuses(self, plan_a, tmp_path):
        plan = self._copy(plan_a)
        plan["schema"] = "cage-campaign-plan-v5"
        with pytest.raises(rc.RunError):
            rc.load_plan(_dump(tmp_path, plan, "v5.json"))


# ---------------------------------------------------------------------------
# run: the dry window gates the class's pressure cells (ADR-0153)
# ---------------------------------------------------------------------------


def _pressure_grid():
    return _tiny_grid(
        f1_baselines=("B1",), f2_baselines=("B1",), f2_budgets=(1.0, 0.5), f2_rates=(0.85,),
    )


def _sentinel(root: Path, step: Dict[str, Any]) -> str:
    path = root / "cells" / step["row_key"] / f".STATUS-{step['dataset']}"
    return path.read_text(encoding="utf-8") if path.exists() else ""


class TestDryWindowRun:
    def test_in_regime_dry_window_lets_the_class_run(self, tmp_path, floor_table, stub):
        plan = _stub_plan(_pressure_grid(), floor_table, stub.cmd)
        (dry,) = _dry_steps(plan)
        root = _run_root(tmp_path)
        assert rc.run_plan(plan, root) == 0
        dry_root = rc.dry_window_root(root, plan)
        assert (dry_root / "cells" / dry["row_key"] / "window_qasper-01" / "regime.json").is_file()
        assert not (root / "cells" / dry["row_key"]).exists()  # never in the campaign tree
        # the attempt sidecar records what the window ran at and its verdict
        side = json.loads((dry_root / "dry_attempts" / f"{dry['row_key']}.01.json").read_text(encoding="utf-8"))
        assert side["label"] == "IN_REGIME" and side["ordinal"] == 1 and side["runner_exit"] == 0
        assert side["identity"] == rc.dry_window_identity(dry)
        assert side["identity"]["offered_rate_rps"] == pytest.approx(0.95 * _RUNG_LAMBDA * 0.5)
        (dry_call,) = _dry_calls(stub)
        assert dry_call["argv"][dry_call["argv"].index("--campaign-root") + 1] == str(dry_root)
        assert float(dry_call["argv"][dry_call["argv"].index("--rate") + 1]) == pytest.approx(
            0.95 * _RUNG_LAMBDA * 0.5
        )
        assert dry_call["env"]["CAGE_CELL_RATE_FRAC"] == "0.95"
        cell_calls = _cell_calls(stub)
        assert len(cell_calls) == 3  # F1 + the two F2 cells
        # order: the dry window ran before either pressure cell
        argvs = [c["argv"] for c in stub.calls()]
        dry_at = argvs.index(dry_call["argv"])
        pressure_at = [argvs.index(c["argv"]) for c in cell_calls if c["env"]["CAGE_CELL_FAMILY"] == "F2"]
        assert pressure_at and all(dry_at < i for i in pressure_at)
        assert all(_sentinel(root, s) == "" for s in _cells(plan))

    @pytest.mark.parametrize(
        "label, reason",
        [("UNPRESSURED", "dry-window-UNPRESSURED"), ("none", "dry-window-no-regime")],
    )
    def test_failed_dry_window_skips_the_class_and_exits_3(
        self, tmp_path, floor_table, stub, monkeypatch, label, reason
    ):
        plan = _stub_plan(_pressure_grid(), floor_table, stub.cmd)
        monkeypatch.setenv("STUB_REGIME_LABEL", label)
        root = _run_root(tmp_path)
        assert rc.run_plan(plan, root) == rc.EXIT_DRY_WINDOW_FAILED == 3
        assert len(_dry_calls(stub)) == 1
        cell_calls = _cell_calls(stub)
        assert [c["env"]["CAGE_CELL_FAMILY"] for c in cell_calls] == ["F1"]  # the budget-free cell still ran
        for s in _cells(plan):
            if s["family"] == "F2":
                assert f"reason={reason}" in _sentinel(root, s), s["row_key"]
            else:
                assert _sentinel(root, s) == ""

    def test_failed_dry_window_gates_the_seal(self, tmp_path, floor_table, stub, monkeypatch):
        plan = _stub_plan(_pressure_grid(), floor_table, stub.cmd)
        monkeypatch.setenv("STUB_REGIME_LABEL", "PAST_CLIFF")
        root = _run_root(tmp_path)
        assert rc.run_plan(plan, root, seal=True, seal_cmd=stub.cmd) == 3
        assert [str(root)] not in [c["argv"] for c in stub.calls()], "a plain seal is gated"
        root2 = root.parent / "run-002"
        assert rc.run_plan(plan, root2, seal=True, seal_partial=True, seal_cmd=stub.cmd) == 3
        assert stub.calls()[-1]["argv"] == [str(root2)], "--seal-partial seals"

    def test_a_failed_cell_outranks_a_failed_dry_window(self, tmp_path, floor_table, stub, monkeypatch):
        plan = _stub_plan(_pressure_grid(), floor_table, stub.cmd)
        monkeypatch.setenv("STUB_REGIME_LABEL", "UNPRESSURED")
        monkeypatch.setenv("STUB_FAIL_MARKER", "--dataset squad_v2")  # the F1 cell
        assert rc.run_plan(plan, _run_root(tmp_path)) == 1

    def test_failed_relaunch_fails_the_dry_window_and_its_class(self, tmp_path, floor_table, stub, monkeypatch):
        plan = _stub_plan(_pressure_grid(), floor_table, stub.cmd)
        monkeypatch.setenv("STUB_FAIL_MARKER", "--no-prefix-cache")  # every prefix-OFF relaunch
        root = _run_root(tmp_path)
        assert rc.run_plan(plan, root) == 1  # launch-failed cells
        assert _dry_calls(stub) == [] and _cell_calls(stub) == []
        for s in _cells(plan):
            assert "reason=relaunch-failed" in _sentinel(root, s)

    def test_an_in_regime_attempt_is_reused_only_for_the_same_plan(self, tmp_path, floor_table, stub):
        plan = _stub_plan(_pressure_grid(), floor_table, stub.cmd)
        root = _run_root(tmp_path)
        assert rc.run_plan(plan, root) == 0
        assert len(_dry_calls(stub)) == 1
        assert rc.run_plan(plan, root) == 0  # same plan, IN_REGIME: reused
        assert len(_dry_calls(stub)) == 1
        assert rc.run_plan(plan, root, force_rerun=True) == 0  # forced: a fresh attempt at ordinal 2
        assert len(_dry_calls(stub)) == 2
        (dry,) = _dry_steps(plan)
        dry_root = rc.dry_window_root(root, plan)
        assert (dry_root / "cells" / dry["row_key"] / "window_qasper-02" / "regime.json").is_file()
        assert _dry_calls(stub)[-1]["env"]["CAGE_WINDOW_ORDINAL_BASE"] == "1"
        # a re-plan with other rung artifacts gets its own dry root: nothing is inherited
        grid = _pressure_grid()
        floor = rc.load_floor_table(floor_table)
        cal = _write_calibrations(tmp_path / "cal2", ("vllm",))
        rungs = _write_rung_calibrations(tmp_path / "cal2", ("vllm",), "a", (1.0, 0.5), lam=9.0)
        orig = rc.SESSION_GRIDS
        rc.SESSION_GRIDS = {"a": grid}
        try:
            replan = rc.build_plan(
                "a", floor, window_duration_s=60.0, runner_cmd=stub.cmd,
                launcher_cmds={"vllm": stub.cmd, "sglang": stub.cmd},
                calibrations=cal, rung_calibrations=rungs,
            )
        finally:
            rc.SESSION_GRIDS = orig
        assert rc.dry_window_root(root, replan) != dry_root
        assert rc.run_plan(replan, root) == 0
        assert len(_dry_calls(stub)) == 3

    def test_a_failed_verdict_is_never_reused(self, tmp_path, floor_table, stub, monkeypatch):
        # review HIGH 2: a stale UNPRESSURED must not skip the class again on
        # the next run; the next run makes a fresh attempt at the next ordinal
        plan = _stub_plan(_pressure_grid(), floor_table, stub.cmd)
        root = _run_root(tmp_path)
        monkeypatch.setenv("STUB_REGIME_LABEL", "UNPRESSURED")
        assert rc.run_plan(plan, root) == 3
        assert len(_dry_calls(stub)) == 1
        monkeypatch.setenv("STUB_REGIME_LABEL", "IN_REGIME")
        assert rc.run_plan(plan, root) == 0
        assert len(_dry_calls(stub)) == 2
        (dry,) = _dry_steps(plan)
        dry_root = rc.dry_window_root(root, plan)
        sides = sorted((dry_root / "dry_attempts").glob(f"{dry['row_key']}.*.json"))
        labels = [json.loads(p.read_text(encoding="utf-8"))["label"] for p in sides]
        assert labels == ["UNPRESSURED", "IN_REGIME"]
        assert (dry_root / "cells" / dry["row_key"] / "window_qasper-02" / "regime.json").is_file()
        # and the pressure cells ran this time (their sentinels are gone)
        for s in _cells(plan):
            assert _sentinel(root, s) == "", s["row_key"]

    def test_run_cli_exit_code(self, tmp_path, floor_table, stub, monkeypatch):
        plan = _stub_plan(_pressure_grid(), floor_table, stub.cmd)
        path = _dump(tmp_path, plan, "plan.json")
        monkeypatch.setenv("STUB_REGIME_LABEL", "UNKNOWN_TELEMETRY")
        assert rc.main(["run", "--plan", str(path), "--campaign-root", str(_run_root(tmp_path))]) == 3


# ---------------------------------------------------------------------------
# calibrate-rungs (ADR-0154): the cal-v2 ladder through the runner
# ---------------------------------------------------------------------------


class FakeWorld:
    """An in-process ``exec_fn``: launcher verbs return ``relaunch_rc``; a
    runner invocation writes a window whose attainment is min(1, cap / rate)
    (a server with capacity ``cap`` requests per second), unless
    ``write_rows`` is off (a runner that died before the window)."""

    def __init__(self, cap: float, *, relaunch_rc: int = 0, write_rows: bool = True, regime: str = "IN_REGIME"):
        self.cap = cap
        self.relaunch_rc = relaunch_rc
        self.write_rows = write_rows
        self.regime = regime
        self.calls: List[Tuple[List[str], Dict[str, str]]] = []

    def __call__(self, argv, env) -> int:
        argv = list(argv)
        self.calls.append((argv, dict(env)))
        if "--campaign-root" not in argv:
            return self.relaunch_rc if "restart" in argv else 0
        if "--open-loop-warmup-s" in argv and argv[argv.index("--open-loop-warmup-s") + 1] != "0":
            # the real runner refuses the Jain trim in campaign mode (ADR-0055
            # amendment 2026-09-19); review CRITICAL 1 of 2026-10-09
            return 2
        if not self.write_rows:
            return 1
        rate = float(argv[argv.index("--rate") + 1])
        root = Path(argv[argv.index("--campaign-root") + 1])
        spec = derive_cell_spec(
            baseline=argv[argv.index("--baseline") + 1],
            baseline_label=argv[argv.index("--baseline-label") + 1],
            backend=argv[argv.index("--backend") + 1],
            model=argv[argv.index("--model") + 1],
            env=env,
        )
        wdir = root / "cells" / spec.to_row_key() / f"window_{argv[argv.index('--dataset') + 1]}-01"
        wdir.mkdir(parents=True)
        n = max(1, round(rate * PROBE_WINDOW_S))
        ok_share = min(1.0, self.cap / rate)
        # the runner's Jain trim drops the warmup rows; attainment over the
        # KEPT rows is exactly min(1, cap / rate) up to one row
        kept = [i * PROBE_WINDOW_S / n for i in range(n) if i * PROBE_WINDOW_S / n >= PROBE_WARMUP_S]
        n_ok = round(len(kept) * ok_share)
        rows = [
            {"arrival_s": arrival, "ok": k < n_ok, "example_id": f"q{k}"}
            for k, arrival in enumerate(kept)
        ]
        (wdir / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        (wdir / "regime.json").write_text(json.dumps({"label": self.regime}), encoding="utf-8")
        return 0

    def runner_calls(self):
        return [(a, e) for a, e in self.calls if "--campaign-root" in a]

    def verbs(self):
        return ["runner" if "--campaign-root" in a else a[1] for a, _ in self.calls]


def _calibrate(tmp_path: Path, world: FakeWorld, grid, **kw) -> Dict[str, Any]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    floor = tmp_path / "ft.json"
    if not floor.exists():
        floor.write_text(json.dumps(_floor_table_doc()), encoding="utf-8")
    cal = _write_calibrations(tmp_path / "cal", ("vllm",))["vllm"]
    orig = rc.SESSION_GRIDS
    rc.SESSION_GRIDS = {grid.session: grid}
    try:
        return rc.calibrate_rungs(
            "a", "vllm", rc.load_floor_table(floor), calibration=cal, out_root=tmp_path / "out",
            runner_cmd=("runner",), launcher_cmds={"vllm": ("launch",)},
            exec_fn=world, log=lambda _m: None, **kw,
        )
    finally:
        rc.SESSION_GRIDS = orig


class TestCalibrateRungs:
    def test_ladder_brackets_the_capacity_per_rung(self, tmp_path):
        world = FakeWorld(cap=6.0)
        doc = _calibrate(tmp_path, world, _pressure_grid())
        assert doc["schema"] == rc.RUNG_CALIBRATION_SCHEMA and doc["confirmatory"] is False
        assert doc["engine"] == "vllm" and doc["model"] == "Qwen/Qwen3-14B" and doc["session"] == "a"
        assert list(doc["rungs"]) == ["1", "0.5"]  # loosest first
        floors = doc["ladder"]["floor"]
        start = floor_start_qps(
            FloorMeasurement(
                ttft_s=floors["ttft_s"], tpot_s=floors["tpot_s"],
                n_requests=floors["n_requests"], statistic=floors["statistic"],
            ),
            max_tokens=rc.LADDER_START_DECODE_TOKENS,
        )
        assert doc["ladder"]["floor_start_qps"] == doc["ladder"]["first_start_qps"] == start
        assert doc["ladder"]["window_s"] == PROBE_WINDOW_S and doc["ladder"]["factor"] == PROBE_LADDER_FACTOR
        unsustainable_above = world.cap / PROBE_ATTAINMENT_MIN
        for key, rec in doc["rungs"].items():
            assert rec["label"] == "ESTIMATED"
            assert rec["lambda_star_qps"] == rec["sustained_rate_qps"] < rec["first_unsustainable_qps"]
            # attainment min(1, cap/rate): sustainable iff rate <= cap/0.9
            # exact up to the one-row rounding of the fake's attainment (1 percent)
            assert rec["sustained_rate_qps"] <= unsustainable_above * 1.01
            assert rec["first_unsustainable_qps"] >= unsustainable_above * 0.99
            steps = rec["steps"]
            ladder = [s for s in steps if s["phase"] == "ladder"]
            bisect = [s for s in steps if s["phase"] == "bisect"]
            assert len(bisect) == PROBE_BISECT_STEPS
            assert ladder[-1]["attainment"] < PROBE_ATTAINMENT_MIN <= ladder[-2]["attainment"]
            # the bracket after bisection is a quarter of the ladder gap
            gap = ladder[-1]["rate_qps"] - ladder[-2]["rate_qps"]
            assert rec["first_unsustainable_qps"] - rec["sustained_rate_qps"] == pytest.approx(gap / 4, rel=1e-6)
            for i, s in enumerate(steps):
                assert s["regime_label"] == "IN_REGIME" and s["runner_exit"] == 0
                assert s["trim_warmup_s"] == PROBE_WARMUP_S and s["replay_env"] == f"{rc.LADDER_REPLAY_ENV}=1"
                assert Path(s["root"]).name == f"cal-vllm-r{key.replace('.', 'p')}-s{i:02d}"
                assert Path(s["root"]).parent.name == "a"
        # the second rung starts two ladder steps below the first rung's lambda*
        first = doc["rungs"]["1"]["lambda_star_qps"]
        assert doc["rungs"]["0.5"]["start_qps"] == pytest.approx(first / PROBE_LADDER_FACTOR ** 2)
        assert doc["rungs"]["1"]["start_qps"] == pytest.approx(start)
        assert doc["rungs"]["1"]["start_basis"].startswith("floor-service-rate")
        assert "previous rung lambda*" in doc["rungs"]["0.5"]["start_basis"]
        assert doc["workload"]["arm"] == "gold-fresh" and doc["workload"]["replay"] is True
        assert doc["workload"]["seq_tokens"] == _ANCHOR_SEQ_TOKENS and doc["workload"]["window_mode"] == "duration"

    def test_every_window_is_the_gold_cell_under_the_rung_relaunch(self, tmp_path):
        world = FakeWorld(cap=6.0)
        doc = _calibrate(tmp_path, world, _pressure_grid())
        verbs = world.verbs()
        assert verbs[0] == "stop" and verbs[1] == "restart" and verbs[-1] == "stop"
        assert verbs.count("restart") == 2 and verbs.count("stop") == 2  # clean room + end of calibration
        restarts = [(a, e) for a, e in world.calls if "restart" in a]
        assert all("--no-prefix-cache" in a for a, _e in restarts)
        assert [e["CAGE_KV_BUDGET_BYTES"] for _a, e in restarts] == [str(_ANCHOR_DEMAND), str(_ANCHOR_DEMAND // 2)]
        for rec in doc["rungs"].values():
            assert rec["relaunch"]["exit"] == 0
            assert rec["relaunch"]["demand_class"]["seq_tokens"] == _ANCHOR_SEQ_TOKENS
        roots = set()
        for argv, env in world.runner_calls():
            assert env[rc.LADDER_REPLAY_ENV] == "1"
            assert env["CAGE_CELL_ARM"] == "gold-fresh" and env["CAGE_CELL_FAMILY"] == "F2"
            assert "CAGE_SLO_FLOORS_JSON" in env and "CAGE_BUDGET_PLAN_JSON" in env
            assert argv[argv.index("--workload-mode") + 1] == "open_loop"
            assert argv[argv.index("--duration-s") + 1] == f"{PROBE_WINDOW_S:g}"
            # the runner refuses the warm-up flag in campaign mode: the reader trims instead
            assert "--open-loop-warmup-s" not in argv
            assert argv[argv.index("--num-trials") + 1] == "1"
            assert "--arrival-count" not in argv
            assert float(argv[argv.index("--rate") + 1]) > 0
            roots.add(argv[argv.index("--campaign-root") + 1])
        assert len(roots) == len(world.runner_calls())  # one root per window, never reused

    def test_artifact_roundtrips_into_a_plan(self, tmp_path):
        doc = _calibrate(tmp_path, FakeWorld(cap=6.0), _pressure_grid())
        path = _write_doc(tmp_path, doc, "rungs_vllm.json")
        cal = rc.load_rung_calibration(path)
        assert cal.lambdas == {
            1.0: doc["rungs"]["1"]["lambda_star_qps"], 0.5: doc["rungs"]["0.5"]["lambda_star_qps"],
        }
        grid = _pressure_grid()
        floor = rc.load_floor_table(tmp_path / "ft.json")
        orig = rc.SESSION_GRIDS
        rc.SESSION_GRIDS = {"a": grid}
        try:
            plan = rc.build_plan(
                "a", floor, window_duration_s=60.0,
                calibrations={"vllm": tmp_path / "cal" / "calibration_vllm.json"},
                rung_calibrations={"vllm": path},
            )
        finally:
            rc.SESSION_GRIDS = orig
        for s in _pressure_cells(plan):
            r = s["cellspec"]["budget_r"]
            assert s["lambda_star_rps"] == pytest.approx(cal.lambda_at(r))
            assert s["lambda_star_source"]["artifact_sha256"] == cal.sha256
            assert s["offered_rate_rps"] == pytest.approx(0.85 * cal.lambda_at(r))

    def test_relaunch_failure_labels_the_rung(self, tmp_path):
        world = FakeWorld(cap=6.0, relaunch_rc=1)
        doc = _calibrate(tmp_path, world, _pressure_grid())
        assert {rec["label"] for rec in doc["rungs"].values()} == {rc.RUNG_LABEL_RELAUNCH_FAILED}
        assert world.runner_calls() == []
        assert all(rec["lambda_star_qps"] is None and rec["steps"] == [] for rec in doc["rungs"].values())
        assert world.verbs()[-1] == "stop"
        # the artifact loads (labels are honest) and the plan refuses the rungs
        path = _write_doc(tmp_path, doc, "rungs_vllm.json")
        cal = rc.load_rung_calibration(path)
        assert cal.lambdas == {} and set(cal.labels.values()) == {rc.RUNG_LABEL_RELAUNCH_FAILED}
        orig = rc.SESSION_GRIDS
        rc.SESSION_GRIDS = {"a": _pressure_grid()}
        try:
            with pytest.raises(rc.PlanError, match=r"r=1 \(RELAUNCH_FAILED\), r=0\.5 \(RELAUNCH_FAILED\)"):
                rc.build_plan(
                    "a", rc.load_floor_table(tmp_path / "ft.json"), window_duration_s=60.0,
                    calibrations={"vllm": tmp_path / "cal" / "calibration_vllm.json"},
                    rung_calibrations={"vllm": path},
                )
        finally:
            rc.SESSION_GRIDS = orig

    def test_a_dead_runner_labels_the_rung_and_the_ladder_moves_on(self, tmp_path):
        world = FakeWorld(cap=6.0, write_rows=False)
        doc = _calibrate(tmp_path, world, _pressure_grid())
        assert {rec["label"] for rec in doc["rungs"].values()} == {rc.RUNG_LABEL_PROBE_FAILED}
        for rec in doc["rungs"].values():
            assert "wrote no" in rec["error"] and rec["steps"] == [] and rec["lambda_star_qps"] is None
        assert len(world.runner_calls()) == 2  # one window attempt per rung
        # the second rung restarts from the floor start (no previous lambda*)
        assert doc["rungs"]["0.5"]["start_qps"] == doc["rungs"]["1"]["start_qps"]

    def test_none_sustainable_and_ladder_exhausted(self, tmp_path):
        tiny = _calibrate(tmp_path / "tiny", FakeWorld(cap=0.01), _pressure_grid(), rungs=(1.0,))
        assert tiny["rungs"]["1"]["label"] == "NONE_SUSTAINABLE" and len(tiny["rungs"]["1"]["steps"]) == 1
        huge = _calibrate(tmp_path / "huge", FakeWorld(cap=1e9), _pressure_grid(), rungs=(1.0,))
        assert huge["rungs"]["1"]["label"] == "LADDER_EXHAUSTED"
        assert len(huge["rungs"]["1"]["steps"]) == PROBE_MAX_STEPS
        assert huge["rungs"]["1"]["sustained_rate_qps"] == pytest.approx(
            huge["rungs"]["1"]["steps"][-1]["rate_qps"]
        )

    def test_rungs_filter_and_refusals(self, tmp_path):
        one = _calibrate(tmp_path / "one", FakeWorld(cap=6.0), _pressure_grid(), rungs=(0.5,))
        assert list(one["rungs"]) == ["0.5"]
        with pytest.raises(rc.PlanError, match="not a budgeted rung"):
            _calibrate(tmp_path / "bad", FakeWorld(cap=6.0), _pressure_grid(), rungs=(0.75,))
        with pytest.raises(rc.PlanError, match="--start-qps"):
            _calibrate(tmp_path / "start", FakeWorld(cap=6.0), _pressure_grid(), start_qps=0.0)
        forced = _calibrate(tmp_path / "forced", FakeWorld(cap=6.0), _pressure_grid(), rungs=(1.0,), start_qps=5.0)
        assert forced["rungs"]["1"]["start_qps"] == 5.0 and forced["rungs"]["1"]["start_basis"] == "--start-qps 5"
        orig = rc.SESSION_GRIDS
        rc.SESSION_GRIDS = {"a": _pressure_grid()}
        try:
            with pytest.raises(rc.PlanError, match="not a server engine"):
                rc.calibrate_rungs(
                    "a", "sglang", rc.load_floor_table(tmp_path / "forced" / "ft.json"),
                    calibration=tmp_path / "forced" / "cal" / "calibration_vllm.json", out_root=tmp_path / "x",
                    exec_fn=FakeWorld(cap=6.0), log=lambda _m: None,
                )
        finally:
            rc.SESSION_GRIDS = orig
        with pytest.raises(rc.PlanError, match="no F2 baseline on the gold-fresh arm"):
            _calibrate(
                tmp_path / "nogold", FakeWorld(cap=6.0),
                _tiny_grid(f2_baselines=("B5",), f2_budgets=(1.0,), f2_rates=(0.85,)),
            )

    def test_a_reused_root_refuses(self, tmp_path):
        _calibrate(tmp_path, FakeWorld(cap=6.0), _pressure_grid(), rungs=(1.0,))
        again = _calibrate(tmp_path, FakeWorld(cap=6.0), _pressure_grid(), rungs=(1.0,))
        assert again["rungs"]["1"]["label"] == rc.RUNG_LABEL_PROBE_FAILED
        assert "already exists" in again["rungs"]["1"]["error"]


class TestReadLadderWindow:
    def _window(self, tmp_path: Path, rows) -> Path:
        wdir = tmp_path / "cells" / "rk" / "window_qasper-01"
        wdir.mkdir(parents=True)
        (wdir / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return tmp_path

    def test_counts_after_the_warmup_trim(self, tmp_path):
        rows = [{"arrival_s": 5.0, "ok": True}, {"arrival_s": 12.0, "ok": True}, {"arrival_s": 30.0, "ok": False},
                {"arrival_s": 70.0, "ok": True}]
        step, label = rc._read_ladder_window(self._window(tmp_path, rows), "rk", "qasper", 2.0, "ladder")
        assert (step.n_scheduled, step.n_completed, step.phase) == (3, 2, "ladder")
        assert step.throughput_rps == pytest.approx(2 / (PROBE_WINDOW_S - PROBE_WARMUP_S))
        assert label is None

    def test_refusals(self, tmp_path):
        with pytest.raises(rc.RunError, match="wrote no"):
            rc._read_ladder_window(tmp_path / "absent", "rk", "qasper", 2.0, "ladder")
        with pytest.raises(rc.RunError, match="intended arrival"):
            rc._read_ladder_window(self._window(tmp_path / "a", [{"ok": True}]), "rk", "qasper", 2.0, "ladder")
        with pytest.raises(rc.RunError, match="no arrivals in the post-warmup window"):
            rc._read_ladder_window(
                self._window(tmp_path / "b", [{"arrival_s": 1.0, "ok": True}]), "rk", "qasper", 2.0, "ladder"
            )


_LADDER_STUB_SOURCE = """\
import json, os, sys
sys.path.insert(0, __REPO_ROOT__)
argv = sys.argv[1:]
with open(os.environ["STUB_CALLS"], "a", encoding="utf-8") as fh:
    fh.write(json.dumps({"argv": argv}) + "\\n")
if "--campaign-root" not in argv:
    sys.exit(0)
if "--open-loop-warmup-s" in argv and argv[argv.index("--open-loop-warmup-s") + 1] != "0":
    print("REFUSED: campaign mode refuses --open-loop-warmup-s > 0", file=sys.stderr); sys.exit(2)
from src.orchestration.campaign_session import derive_cell_spec
def flag(name):
    return argv[argv.index(name) + 1]
rate = float(flag("--rate")); cap = float(os.environ["STUB_CAPACITY_QPS"])
spec = derive_cell_spec(baseline=flag("--baseline"), baseline_label=flag("--baseline-label"),
                        backend=flag("--backend"), model=flag("--model"), env=os.environ)
wdir = os.path.join(flag("--campaign-root"), "cells", spec.to_row_key(), "window_%s-01" % flag("--dataset"))
os.makedirs(wdir)
n = max(1, round(rate * 75.0)); share = min(1.0, cap / rate)
kept = [i * 75.0 / n for i in range(n) if i * 75.0 / n >= 10.0]
n_ok = round(len(kept) * share)
with open(os.path.join(wdir, "requests.jsonl"), "w", encoding="utf-8") as fh:
    for k, t in enumerate(kept):
        fh.write(json.dumps({"arrival_s": t, "ok": k < n_ok}) + "\\n")
sys.exit(0)
""".replace("__REPO_ROOT__", repr(str(rc.REPO_ROOT)))


def _cli_calibrate(tmp_path: Path, monkeypatch, capacity: str, extra: List[str]) -> Tuple[int, Path]:
    stub_path = tmp_path / "ladder_stub.py"
    stub_path.write_text(_LADDER_STUB_SOURCE, encoding="utf-8")
    monkeypatch.setenv("STUB_CALLS", str(tmp_path / "calls.jsonl"))
    monkeypatch.setenv("STUB_CAPACITY_QPS", capacity)
    floor = tmp_path / "ft.json"
    floor.write_text(json.dumps(_floor_table_doc()), encoding="utf-8")
    cal = _write_calibrations(tmp_path / "cal", ("vllm",))["vllm"]
    out = tmp_path / "rungs_vllm.json"
    cmd = f"{sys.executable} {stub_path}"
    orig = rc.SESSION_GRIDS
    rc.SESSION_GRIDS = {"a": _pressure_grid()}
    try:
        code = rc.main([
            "calibrate-rungs", "--session", "a", "--engine", "vllm", "--floor-table", str(floor),
            "--calibration", str(cal), "--out", str(out), "--out-root", str(tmp_path / "roots"),
            "--runner-cmd", cmd, "--launcher-cmd", cmd, *extra,
        ])
    finally:
        rc.SESSION_GRIDS = orig
    return code, out


class TestCalibrateRungsCli:
    def test_end_to_end_with_a_subprocess_runner(self, tmp_path, monkeypatch):
        code, out = _cli_calibrate(tmp_path, monkeypatch, "4.0", ["--rungs", "0.5", "--start-qps", "2"])
        assert code == 0
        doc = json.loads(out.read_text(encoding="utf-8"))
        rec = doc["rungs"]["0.5"]
        assert rec["label"] == "ESTIMATED"
        assert rec["lambda_star_qps"] <= 4.0 / 0.9 * 1.01
        assert rec["first_unsustainable_qps"] >= 4.0 / 0.9 * 0.99
        assert all(Path(s["root"]).parent == tmp_path / "roots" / "a" for s in rec["steps"])
        assert rc.load_rung_calibration(out).lambda_at(0.5) == rec["lambda_star_qps"]

    def test_exit_1_when_a_rung_is_not_estimated(self, tmp_path, monkeypatch, capsys):
        code, out = _cli_calibrate(tmp_path, monkeypatch, "0.01", ["--rungs", "1"])
        assert code == 1
        assert json.loads(out.read_text(encoding="utf-8"))["rungs"]["1"]["label"] == "NONE_SUSTAINABLE"
        assert "NOT ESTIMATED" in capsys.readouterr().err
