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
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_run_campaign import (  # noqa: E402
    _ANCHOR_DEMAND,
    _ANCHOR_SEQ_TOKENS,
    _RUNG_LAMBDA,
    _RUNG_RULER_LAMBDA,
    _RUNG_SMALL_FACTOR,
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

    def test_classes_block_loads_the_smallest_ladder(self, tmp_path):
        # ADR-0156: the second anchor rides the artifact's classes block
        doc = _rung_calibration_doc("vllm", "a", (1.0, 0.5), small=(348, "retr-trunc"))
        cal = rc.load_rung_calibration(_write_doc(tmp_path, doc))
        assert cal.anchor_seq_tokens == _ANCHOR_SEQ_TOKENS
        assert (cal.small_seq_tokens, cal.small_arm, cal.small_prefix_mode) == (348, "retr-trunc", "OFF")
        assert cal.small_lambdas == {1.0: _RUNG_LAMBDA * _RUNG_SMALL_FACTOR, 0.5: _RUNG_LAMBDA * _RUNG_SMALL_FACTOR * 0.5}
        assert cal.small_label_at(0.5) == "ESTIMATED" and cal.small_lambda_at(0.25) is None
        # a one-class artifact loads with the small side empty
        one = rc.load_rung_calibration(_write_doc(tmp_path, _rung_calibration_doc("vllm", "a", (1.0,)), "one.json"))
        assert one.small_seq_tokens is None and one.small_lambdas == {} and one.anchor_seq_tokens == _ANCHOR_SEQ_TOKENS

    @pytest.mark.parametrize(
        "spoil, match",
        [
            (lambda d: d.update(schema="cage-rung-calibration-v1"), "predates the second anchor"),
            (lambda d: d.update(classes=[]), "classes must be a mapping"),
            (lambda d: d["classes"].update({"349": dict(d["classes"]["348"], seq_tokens=349)}), "two 'smallest' entries"),
            (lambda d: d["classes"]["348"].update(role="middle"), "must be 'anchor', 'smallest', 'measured' or 'ruler'"),
            (lambda d: d["classes"]["348"].update(seq_tokens=347), "the integer >= 1 the key names"),
            (lambda d: d["classes"]["348"].update(arm=""), "must name the smallest class's arm"),
            (lambda d: d["classes"]["348"].update(prefix_mode="on"), "must be 'ON' or 'OFF'"),
            (lambda d: d["classes"][str(_ANCHOR_SEQ_TOKENS)].update(rungs={"1": {"r": 1.0, "label": "ESTIMATED", "lambda_star_qps": 1.0}}), "differs from the top-level rungs table"),
            (lambda d: d.update(anchor_seq_tokens=4000), "disagrees with anchor_seq_tokens"),
            (lambda d: d["classes"]["348"]["rungs"]["1"].update(lambda_star_qps=None), r"classes\['348'\].rungs\['1'\] is ESTIMATED but"),
        ],
    )
    def test_classes_refusals(self, tmp_path, spoil, match):
        doc = _rung_calibration_doc("vllm", "a", (1.0, 0.5), small=(348, "retr-trunc"))
        spoil(doc)
        with pytest.raises(rc.PlanError, match=match):
            rc.load_rung_calibration(_write_doc(tmp_path, doc))

    def test_smallest_class_must_be_below_the_anchor(self, tmp_path):
        doc = _rung_calibration_doc("vllm", "a", (1.0,), small=(_ANCHOR_SEQ_TOKENS + 1, "x"))
        with pytest.raises(rc.PlanError, match="is not below the anchor"):
            rc.load_rung_calibration(_write_doc(tmp_path, doc))

    def test_interpolation_arithmetic(self):
        alpha = rc.interpolation_alpha(8.0, 4779, 32.0, 348)
        assert alpha == pytest.approx(math.log(4.0) / math.log(4779 / 348))
        assert rc.interpolation_alpha(8.0, 4779, 8.0 * 4779 / 348, 348) == pytest.approx(1.0)  # KV-bound limit
        assert rc.interpolation_alpha(8.0, 4779, 8.0, 348) == 0.0  # pure request cap
        with pytest.raises(rc.PlanError, match="positive finite rates"):
            rc.interpolation_alpha(0.0, 4779, 8.0, 348)
        with pytest.raises(rc.PlanError, match="s_small"):
            rc.interpolation_alpha(8.0, 348, 8.0, 4779)


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
        assert header["schema"] == rc.RUNG_CALIBRATION_SCHEMA == "cage-rung-calibration-v2"
        assert header["adr"] == "ADR-0154" and header["finding"] == "S0F-62"
        assert header["two_anchor_adr"] == "ADR-0156"
        assert header["anchor_arm"] == "gold-fresh" and header["anchor_seq_tokens"] == _ANCHOR_SEQ_TOKENS
        assert header["required_rungs"] == {
            "sglang": [1.5, 1.25, 1.0, 0.75, 0.5, 0.375, 0.25],
            "vllm": [1.5, 1.25, 1.0, 0.75, 0.5, 0.375, 0.25],
        }
        # the classes below the anchor per engine and the rungs each is carried at
        below = header["classes_below_anchor"]
        assert set(below) == {"vllm", "sglang"}
        assert set(below["vllm"]) == {"2336", "1127", "1126", "702", "348"}
        assert below["vllm"]["348"] == [1.5, 1.25, 1.0, 0.75, 0.5, 0.375, 0.25]  # retr-trunc, F2
        assert below["vllm"]["1126"] == [1.0, 0.5, 0.25]  # retr-reuse, F3 only
        alpha = math.log(_RUNG_SMALL_FACTOR) / math.log(_ANCHOR_SEQ_TOKENS / 348)
        for engine in ("vllm", "sglang"):
            art = header["artifacts"][engine]
            assert len(art["sha256"]) == 64 and art["session"] == "a"
            assert art["rungs"] == {f"{r:g}": _RUNG_LAMBDA * r for r in (1.5, 1.25, 1.0, 0.75, 0.5, 0.375, 0.25)}
            assert set(art["labels"].values()) == {"ESTIMATED"}
            small = art["small_class"]
            assert (small["seq_tokens"], small["arm"], small["prefix_mode"]) == (348, "retr-trunc", "OFF")
            assert small["rungs"]["1"] == pytest.approx(_RUNG_LAMBDA * _RUNG_SMALL_FACTOR)
            assert set(art["alpha"]) == {"1.5", "1.25", "1", "0.75", "0.5", "0.375", "0.25"}
            assert all(a == pytest.approx(alpha) for a in art["alpha"].values())

    def test_an_engine_serving_classes_below_the_anchor_needs_the_smallest_ladder(
        self, floor_table, calibrations_a, tmp_path
    ):
        # ADR-0156: session a serves five classes below the anchor on both engines
        rungs = rc.budgeted_rungs(rc.SESSION_GRIDS["a"])
        anchor_only = _write_rung_calibrations(tmp_path / "ao", ("vllm", "sglang"), "a", rungs, small=None)
        with pytest.raises(rc.PlanError, match="carries no smallest-class ladder .* 348 tokens"):
            self._build(floor_table, calibrations_a, anchor_only)
        wrong = _write_rung_calibrations(tmp_path / "wrong", ("vllm", "sglang"), "a", rungs, small=(702, "retr-comp"))
        with pytest.raises(rc.PlanError, match="smallest class is 702 tokens .* is 348 tokens"):
            self._build(floor_table, calibrations_a, wrong)
        short = _write_rung_calibrations(
            tmp_path / "short", ("vllm", "sglang"), "a", rungs,
            small=(348, "retr-trunc"), small_labels={0.25: "LADDER_EXHAUSTED"},
        )
        with pytest.raises(rc.PlanError, match=r"smallest class \(348 tokens, retr-trunc\) has no ESTIMATED lambda\* for r=0\.25 \(LADDER_EXHAUSTED\)"):
            self._build(floor_table, calibrations_a, short)
        # a smaller sequence that sustained LESS than the anchor is a defective ladder
        slower = _write_rung_calibrations(tmp_path / "slow", ("vllm", "sglang"), "a", rungs, small=(348, "retr-trunc"), small_factor=0.5)
        with pytest.raises(rc.PlanError, match=r"alpha -\d\.\d+ < 0"):
            self._build(floor_table, calibrations_a, slower)
        # a pure request cap (equal rates, alpha 0) is legal and interpolates flat
        # (ADR-0169: session a's RULER cells need the RULER ladder too for the plan to build)
        flat = _write_rung_calibrations(
            tmp_path / "flat", ("vllm", "sglang"), "a", rungs, small=(348, "retr-trunc"), small_factor=1.0,
            ruler=_RUNG_RULER_LAMBDA,
        )
        plan = self._build(floor_table, calibrations_a, flat)
        for s in _pressure_cells(plan):
            if s["dataset"] == "ruler":  # ADR-0169: the instrument offers its own ladder's rate
                assert s["lambda_star_rps"] == pytest.approx(_RUNG_RULER_LAMBDA * s["cellspec"]["budget_r"])
                continue
            assert s["lambda_star_rps"] == pytest.approx(_RUNG_LAMBDA * s["cellspec"]["budget_r"])

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
            # S0F-88 (owner 2026-10-12): the gate runs at 1.2 x lambda*
            assert s["rate_frac"] == rc.DRY_WINDOW_RATE_FRAC == 1.2 and s["expected_label"] == "IN_REGIME"
            assert s["offered_rate_rps"] == pytest.approx(1.2 * s["lambda_star_rps"])
            spec = rc.CellSpec.from_flat_dict(s["cellspec"])
            assert spec.rate_frac == 1.2 and spec.to_row_key() == s["row_key"]
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
            assert s["env"]["CAGE_CELL_RATE_FRAC"] == "1.2"
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
        assert header["rate_frac"] == 1.2 and header["expected_label"] == "IN_REGIME"
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

    def test_window_span_clauses(self, plan_a, tmp_path):
        # ADR-0156 / S0F-68: the recorded span and its floor verdict re-derive
        plan = self._copy(plan_a)
        cell = next(s for s in _cells(plan) if s["family"] == "F2" and s["blocked_on"] is None)
        cell["window_span_s_expected"] = cell["window_span_s_expected"] * 2
        with pytest.raises(rc.RunError, match="window_span_s_expected=.* != num_queries / offered rate"):
            rc.load_plan(_dump(tmp_path, plan, "span.json"))
        plan = self._copy(plan_a)
        cell = next(s for s in _cells(plan) if s["family"] == "F2" and s["blocked_on"] is None)
        cell["window_span_below_floor"] = not cell["window_span_below_floor"]
        with pytest.raises(rc.RunError, match="window_span_below_floor=.* disagrees with the registered floor"):
            rc.load_plan(_dump(tmp_path, plan, "floor.json"))
        plan = self._copy(plan_a)
        cell = next(s for s in _cells(plan) if s["family"] == "F1" and s["serving"] is not None)
        cell["window_span_s_expected"] = 12.0
        with pytest.raises(rc.RunError, match="carries a window span"):
            rc.load_plan(_dump(tmp_path, plan, "f1_span.json"))
        plan = self._copy(plan_a)
        del _cells(plan)[0]["window_span_s_expected"]
        with pytest.raises(rc.RunError, match="missing key"):
            rc.load_plan(_dump(tmp_path, plan, "span_key.json"))

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
            (lambda s: s["argv"].__setitem__(s["argv"].index("--rate") + 1, "99"), r"is not 1\.2 x"),
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
# The window span audit (ADR-0156 / S0F-68)
# ---------------------------------------------------------------------------


class TestWindowSpans:
    def test_every_pressure_cell_records_w_over_rate_and_the_floor_verdict(self, plan_a):
        executable = _pressure_cells(plan_a)
        for s in executable:
            assert s["window_span_s_expected"] == pytest.approx(s["num_queries"] / s["offered_rate_rps"])
            assert s["window_span_below_floor"] is (s["window_span_s_expected"] < rc.WINDOW_SPAN_FLOOR_S)
        for s in _pressure_cells(plan_a, executable=False):
            # a blocked cell records the same arithmetic on its labeled rate
            assert s["window_span_s_expected"] == pytest.approx(s["num_queries"] / s["offered_rate_rps"])
        for s in _cells(plan_a):
            if s["family"] not in PRESSURE:
                assert s["window_span_s_expected"] is None and s["window_span_below_floor"] is None
        # Q-8 (S0F-88): the sampler ticks at 0.25 s (was 1.0 s)
        assert rc.WINDOW_SPAN_FLOOR_S == PROBE_WARMUP_S == 10.0 and rc.TELEMETRY_SAMPLE_INTERVAL_S == 0.25

    def test_header_summarizes_the_executable_spans_per_class(self, plan_a):
        ws = plan_a["window_spans"]
        executable = _pressure_cells(plan_a)
        spans = sorted(s["window_span_s_expected"] for s in executable)
        assert ws["finding"] == "S0F-68" and ws["adr"] == "ADR-0156" and ws["floor_s"] == 10.0
        assert ws["executable_pressure_cells"] == len(executable) == len(spans)
        assert ws["min_s"] == pytest.approx(spans[0]) and ws["max_s"] == pytest.approx(spans[-1])
        assert ws["median_s"] == pytest.approx(statistics.median(spans))  # Fable LOW 8: the true median
        assert ws["below_floor"] == sum(1 for v in spans if v < 10.0)
        # W = 200 at rate_frac x (8 x r): the gold class at r = 1.5, 1.2 x 12 = 14.4 rps
        # spans 13.9 s; the smallest class (4 x the rate) spans 3.5 s and is below
        assert set(ws["by_class"]) == {"4779", "2336", "1127", "1126", "702", "348"}
        assert list(ws["by_class"]) == ["4779", "2336", "1127", "1126", "702", "348"]  # largest first
        # the small class offers 4 x the anchor's rate, so its windows are the shortest;
        # per class the count below the floor is exactly the cells whose span is under it
        for seq, rec in ws["by_class"].items():
            mine = [s for s in executable if s["demand_class"]["seq_tokens"] == int(seq)]
            assert rec["cells"] == len(mine)
            assert rec["below_floor"] == sum(1 for s in mine if s["window_span_s_expected"] < 10.0)
            assert rec["min_s"] == pytest.approx(min(s["window_span_s_expected"] for s in mine))
        assert ws["by_class"]["348"]["below_floor"] > 0 and ws["by_class"]["348"]["min_s"] < ws["by_class"]["4779"]["min_s"]
        assert ws["by_class"]["4779"]["min_s"] == pytest.approx(200 / (1.2 * _RUNG_LAMBDA * 1.5))
        assert sum(c["cells"] for c in ws["by_class"].values()) == len(executable)
        assert "never a refusal" in ws["note"]

    def test_summary_of_no_pressure_cells(self):
        ws = rc.window_span_summary([])
        assert ws["executable_pressure_cells"] == 0 and ws["below_floor"] == 0
        assert ws["min_s"] is None and ws["median_s"] is None and ws["max_s"] is None and ws["by_class"] == {}

    def test_plan_cli_prints_the_span_line(self, tmp_path, floor_table, calibrations_a, capsys):
        out = tmp_path / "plan.json"
        argv = ["plan", "--session", "a", "--floor-table", str(floor_table), "--window-duration-s", "300", "--out", str(out)]
        for engine, path in calibrations_a.items():
            argv += ["--calibration", f"{engine}={path}"]
        for engine, path in _rungs_for(calibrations_a, "a").items():
            argv += ["--rung-calibration", f"{engine}={path}"]
        assert rc.main(argv) == 0
        text = capsys.readouterr().out
        assert "window spans (S0F-68):" in text and "below the 10 s floor" in text
        assert "WARNING: those windows are shorter than the registered warm-up transient" in text


# ---------------------------------------------------------------------------
# run: the dry window gates the class's pressure cells (ADR-0153)
# ---------------------------------------------------------------------------


def _pressure_grid():
    return _tiny_grid(
        f1_baselines=("B1",), f2_baselines=("B1",), f2_budgets=(1.0, 0.5), f2_rates=(0.85,),
    )


def _two_class_grid():
    """ADR-0156: the anchor (B1 gold-fresh, 4,779 tokens) and the smallest
    class (B11 retr-trunc, 348 tokens) on F2, vLLM only."""
    return _tiny_grid(
        f1_baselines=("B1",), f2_baselines=("B1", "B11"), f2_budgets=(1.0, 0.5), f2_rates=(0.85,),
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
        assert side["identity"]["offered_rate_rps"] == pytest.approx(1.2 * _RUNG_LAMBDA * 0.5)
        (dry_call,) = _dry_calls(stub)
        assert dry_call["argv"][dry_call["argv"].index("--campaign-root") + 1] == str(dry_root)
        assert float(dry_call["argv"][dry_call["argv"].index("--rate") + 1]) == pytest.approx(
            1.2 * _RUNG_LAMBDA * 0.5
        )
        assert dry_call["env"]["CAGE_CELL_RATE_FRAC"] == "1.2"
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
            # S0F-88 U4: a later relaunch of a class whose dry window already
            # failed is skipped; its cells still name the relaunch failure
            # (Q-9, R15: a relaunch failure is a FAIL, never a dry-window skip)
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

    def test_fail_fast_infra_stops_at_the_first_failed_relaunch(self, tmp_path, floor_table, stub, monkeypatch, capsys):
        # Review 2026-10-13 B1 (S0 key RUN_FAIL_FAST=infra): the first failed
        # relaunch ends the run; without it every relaunch is tried in turn.
        plan = _stub_plan(_pressure_grid(), floor_table, stub.cmd)
        monkeypatch.setenv("STUB_FAIL_MARKER", "restart")  # every relaunch fails
        assert rc.run_plan(plan, _run_root(tmp_path / "ff"), fail_fast="infra") == 1
        restarts = [c for c in stub.calls() if c["argv"][0] == "restart"]
        assert len(restarts) == 1 and _cell_calls(stub) == []
        out = capsys.readouterr().out
        assert "FAIL FAST (infra): relaunch failed (exit 1) at step" in out and "step(s) not run after" in out
        assert "[run_campaign] totals:" in out  # decision 3's PARTIAL route reads the totals
        assert rc.run_plan(plan, _run_root(tmp_path / "walk"), fail_fast="off") == 1
        # without the stop every relaunch not already skipped by U4 is tried: more than one
        assert len([c for c in stub.calls() if c["argv"][0] == "restart"]) >= 1 + 2
        with pytest.raises(rc.RunError, match="fail_fast must be"):
            rc.run_plan(plan, _run_root(tmp_path / "bad"), fail_fast="sometimes")

    def test_fail_fast_all_stops_at_the_first_failed_cell_but_infra_does_not(
        self, tmp_path, floor_table, stub, monkeypatch, capsys
    ):
        plan = _stub_plan(_pressure_grid(), floor_table, stub.cmd)
        # the first cell the run executes is the first F2 cell; its rate is its marker
        first = next(s for s in plan["steps"] if s["kind"] == "cell" and s["family"] == "F2")
        rate = first["argv"][first["argv"].index("--rate") + 1]
        monkeypatch.setenv("STUB_FAIL_MARKER", f"--rate {rate}")
        assert rc.run_plan(plan, _run_root(tmp_path / "all"), fail_fast="all") == 1
        assert len(_cell_calls(stub)) == 1  # the failed cell, then the stop
        assert "FAIL FAST (all): cell" in capsys.readouterr().out
        # infra tier: a failed cell is a measured outcome, the run goes on
        assert rc.run_plan(plan, _run_root(tmp_path / "infra"), fail_fast="infra") == 1
        assert len(_cell_calls(stub)) == 1 + len(_cells(plan))

    def test_a_resume_skips_a_relaunch_whose_segment_is_complete(self, tmp_path, floor_table, stub, capsys):
        # Review 2026-10-13 C3: the second run of a complete plan starts no
        # engine (18 SGLang relaunches were 1.3 h on attempt 4); a forced
        # rerun starts them again.
        plan = _stub_plan(_pressure_grid(), floor_table, stub.cmd)
        root = _run_root(tmp_path)
        assert rc.run_plan(plan, root) == 0
        n_first = len([c for c in stub.calls() if c["argv"][0] == "restart"])
        assert n_first == sum(1 for s in plan["steps"] if s["kind"] == "relaunch") > 0
        capsys.readouterr()
        # the stub runner writes no cell windows; a completed run leaves each
        # window's metrics.json (count_complete_windows' sentinel): mark them
        for s in _cells(plan):
            for k in range(1, int(s["windows"]) + 1):
                wdir = root / "cells" / s["row_key"] / f"window_{s['dataset']}-{k:02d}"
                wdir.mkdir(parents=True, exist_ok=True)
                (wdir / "metrics.json").write_text("{}", encoding="utf-8")
        dry_root = rc.dry_window_root(root, plan)
        incomplete = [(s["kind"], s.get("row_key")) for s in plan["steps"]
                      if s["kind"] != "relaunch" and not rc._segment_step_complete(root, dry_root, s)]
        assert incomplete == []
        assert rc.run_plan(plan, root) == 0
        assert len([c for c in stub.calls() if c["argv"][0] == "restart"]) == n_first  # no new restart
        out = capsys.readouterr().out
        assert out.count("(review 2026-10-13 C3)") == n_first and "skipped-complete" in out
        assert rc.run_plan(plan, root, force_rerun=True) == 0
        assert len([c for c in stub.calls() if c["argv"][0] == "restart"]) == 2 * n_first

    def test_an_unknown_telemetry_dry_window_is_rerun_once_under_the_flag(
        self, tmp_path, floor_table, stub, monkeypatch
    ):
        # Decision 10 (2026-10-13): UNKNOWN_TELEMETRY is a refused measurement,
        # not a rate verdict; with --dry-window-telemetry-rerun the window runs
        # again at the next ordinal, and the rerun's label counts.
        plan = _stub_plan(_pressure_grid(), floor_table, stub.cmd)
        monkeypatch.setenv("STUB_REGIME_LABEL_B0", "UNKNOWN_TELEMETRY")
        monkeypatch.setenv("STUB_REGIME_LABEL_B1", "IN_REGIME")
        root = _run_root(tmp_path)
        assert rc.run_plan(plan, root, dry_telemetry_rerun=True) == 0
        assert len(_dry_calls(stub)) == 2
        dry_root = rc.dry_window_root(root, plan)
        (dry,) = [s for s in plan["steps"] if s["kind"] == "dry_window"]
        for ordinal, label in ((1, "UNKNOWN_TELEMETRY"), (2, "IN_REGIME")):
            side = json.loads((dry_root / "dry_attempts" / f"{dry['row_key']}.{ordinal:02d}.json").read_text(encoding="utf-8"))
            assert side["label"] == label and side["ordinal"] == ordinal
        assert [c["env"]["CAGE_CELL_FAMILY"] for c in _cell_calls(stub)].count("F2") == 2  # the pressure cells ran
        # a second UNKNOWN_TELEMETRY (one rerun only) still fails the dry window
        monkeypatch.setenv("STUB_REGIME_LABEL_B1", "UNKNOWN_TELEMETRY")
        monkeypatch.setenv("STUB_REGIME_LABEL_B2", "IN_REGIME")
        root2 = _run_root(tmp_path / "second")
        assert rc.run_plan(plan, root2, dry_telemetry_rerun=True) == rc.EXIT_DRY_WINDOW_FAILED
        # without the flag (a registered run) the first label counts, as before
        monkeypatch.delenv("STUB_REGIME_LABEL_B1"); monkeypatch.delenv("STUB_REGIME_LABEL_B2")
        root3 = _run_root(tmp_path / "third")
        assert rc.run_plan(plan, root3) == rc.EXIT_DRY_WINDOW_FAILED


# ---------------------------------------------------------------------------
# calibrate-rungs (ADR-0154): the cal-v2 ladder through the runner
# ---------------------------------------------------------------------------


def _seg(engine, **relaunch):
    step = {"kind": "relaunch", "engine": engine, "prefix_mode": "OFF", "budget_r": 0.5}
    step.update(relaunch)
    return step


def _cell(engine, dataset="qasper", family="F2", arm="gold-fresh", blocked=None):
    return {"kind": "cell", "engine": engine, "dataset": dataset, "family": family,
            "cellspec": {"arm": arm, "engine": engine}, "blocked_on": blocked}


def test_front_load_puts_one_segment_of_each_first_contact_kind_first_a2():
    # Review 2026-10-13 A2: whole relaunch segments move; the hf pre-block stays
    # first; every other segment keeps its order; the relaunch count is unchanged.
    hf = {"kind": "cell", "engine": "hf", "dataset": "qasper", "family": "F1", "cellspec": {"arm": "gold-fresh"}}
    steps = [
        hf,
        _seg("sglang", budget_r=None), _cell("sglang", family="F1"),                               # 1 bf16 F1
        _seg("sglang", kv_dtype="fp8"), {"kind": "dry_window", "engine": "sglang"}, _cell("sglang"),  # 2 fp8 pressure
        _seg("sglang"), {"kind": "dry_window", "engine": "sglang"}, _cell("sglang", dataset="ruler"),  # 3 ruler pressure
        _seg("sglang", budget_r=None), _cell("sglang", family="F1", arm="retr-comp"),             # 4 retr-comp F1
        _seg("vllm", budget_r=None), _cell("vllm", family="F1"),                                   # 5
        _seg("vllm", kv_dtype="fp8"), {"kind": "dry_window", "engine": "vllm"}, _cell("vllm"),       # 6 fp8 pressure
        _seg("vllm", connector="lmcache"), {"kind": "dry_window", "engine": "vllm"}, _cell("vllm"),  # 7 lmcache
        _seg("vllm"), _cell("vllm", dataset="ruler", blocked="x"),                                 # 8 blocked ruler: not a contact
    ]
    ordered, moved = rc.front_load_first_contacts(list(steps))
    assert len(ordered) == len(steps) and ordered[0] is hf
    assert [m["kind"] for m in moved] == [
        "fp8 start", "fp8 start", "lmcache start", "ruler pressure", "retr-comp closed loop"]
    assert [m["engine"] for m in moved] == ["sglang", "vllm", "vllm", "sglang", "sglang"]
    # "first pressure" of each engine was already picked by its fp8 segment (no double move);
    # the vLLM ruler cell is blocked, so no vLLM ruler segment moves
    relaunches = [s for s in ordered if s["kind"] == "relaunch"]
    assert len(relaunches) == 8
    assert [(r["engine"], r.get("kv_dtype"), r.get("connector")) for r in relaunches[:5]] == [
        ("sglang", "fp8", None), ("vllm", "fp8", None), ("vllm", None, "lmcache"), ("sglang", None, None), ("sglang", None, None)]
    # the rest keep their original order
    tail = [(r["engine"], r.get("budget_r")) for r in relaunches[5:]]
    assert tail == [("sglang", None), ("vllm", None), ("vllm", 0.5)]
    # every segment travels whole: each dry window still follows its relaunch
    for i, s in enumerate(ordered):
        if s["kind"] == "dry_window":
            assert ordered[i - 1]["kind"] == "relaunch" and ordered[i - 1]["engine"] == s["engine"]
    # a plan with no first contact moves nothing
    plain, moved2 = rc.front_load_first_contacts([hf, _seg("vllm", budget_r=None), _cell("vllm", family="F1")])
    assert moved2 == [] and [s["kind"] for s in plain] == ["cell", "relaunch", "cell"]


class FakeWorld:
    """An in-process ``exec_fn``: launcher verbs return ``relaunch_rc``; a
    runner invocation writes a window whose attainment is min(1, cap / rate)
    (a server with capacity ``cap`` requests per second), unless
    ``write_rows`` is off (a runner that died before the window)."""

    def __init__(
        self, cap: float, *, relaunch_rc: int = 0, write_rows: bool = True, regime: str = "IN_REGIME",
        cap_by_arm: Optional[Dict[str, float]] = None, ruler_cap: Optional[float] = None,
        ruler_floor: Tuple[float, float] = (2.3, 0.018), ruler_floor_loaded: bool = False,
        ruler_floor_unloaded_rows: Optional[int] = None, ruler_floor_refusals: int = 0,
    ):
        # review 2026-10-13 (ADR-0169 LOW 4): exactly this many unloaded rows in a
        # floor window (the rest loaded), and this many floor windows that the
        # runner "refuses" (exit 1, no rows) before one succeeds
        self.ruler_floor_unloaded_rows = ruler_floor_unloaded_rows
        self.ruler_floor_refusals = ruler_floor_refusals
        self.cap = cap
        self.cap_by_arm = dict(cap_by_arm or {})  # ADR-0156: a class's own capacity
        # ADR-0169: the RULER instrument's capacity (dataset ruler windows) and the
        # floor its floor window reports (ttft_s, tpot_s); ``ruler_floor_loaded``
        # sends every floor row with other requests in flight (no unloaded row)
        self.ruler_cap = ruler_cap
        self.ruler_floor = ruler_floor
        self.ruler_floor_loaded = ruler_floor_loaded
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
        dataset = argv[argv.index("--dataset") + 1]
        wdir = root / "cells" / spec.to_row_key() / f"window_{dataset}-01"
        wdir.mkdir(parents=True)
        if "--arrival-count" in argv:
            # ADR-0169: the RULER floor window (W arrivals, no duration); every
            # row carries the dispatcher's in-flight count and its clocks
            count = int(argv[argv.index("--arrival-count") + 1])
            if self.ruler_floor_refusals > 0:
                self.ruler_floor_refusals -= 1
                return 1  # the runner refused the floor window before any row
            ttft_s, tpot_s = self.ruler_floor
            unloaded = (
                self.ruler_floor_unloaded_rows if self.ruler_floor_unloaded_rows is not None
                else (0 if self.ruler_floor_loaded else count)
            )
            rows = [
                {"arrival_s": k / rate, "ok": True, "example_id": f"f{k}", "num_tokens": 10,
                 "in_flight_at_send": 1 if k < unloaded else 3,
                 "ttft_from_send_ms": ttft_s * 1000.0, "tpot_ms": tpot_s * 1000.0,
                 "scheduled_ts": 1000.0 + k / rate, "completion_ts": 1000.0 + k / rate + ttft_s}
                for k in range(count)
            ]
            (wdir / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            return 0
        window_s = float(argv[argv.index("--duration-s") + 1])  # ADR-0169: RULER probes 300 s
        n = max(1, round(rate * window_s))
        cap = self.cap_by_arm.get(env.get("CAGE_CELL_ARM", ""), self.cap)
        if dataset == "ruler" and self.ruler_cap is not None:
            cap = self.ruler_cap  # ADR-0169: the instrument's own capacity
        ok_share = min(1.0, cap / rate)
        # the runner's Jain trim drops the warmup rows; attainment over the
        # KEPT rows is exactly min(1, cap / rate) up to one row
        kept = [i * window_s / n for i in range(n) if i * window_s / n >= PROBE_WARMUP_S]
        n_ok = round(len(kept) * ok_share)
        # S0F-88 B-2 (cal-v3): every real row carries the dispatcher clocks
        # (RequestRecord.to_row); an ok request completes at its arrival
        # here, so the in-window served ratio equals the modeled attainment.
        rows = [
            {"arrival_s": arrival, "ok": k < n_ok, "example_id": f"q{k}",
             "scheduled_ts": 1000.0 + arrival, "completion_ts": 1000.0 + arrival}
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

    def test_partial_artifact_lands_after_every_rung_and_the_loader_refuses_it(self, tmp_path):
        # ADR-0162 (S0 attempt 2): a rung job killed at its bound after hours left
        # no artifact. Every finished rung is written at once, marked incomplete;
        # the loader refuses the partial file, so it is forensics only.
        class DiesOnTheSmallestClass(FakeWorld):
            def __call__(self, argv, env):
                if "--campaign-root" in list(argv) and env.get("CAGE_CELL_ARM") == "retr-trunc":
                    raise RuntimeError("pod job killed at its bound")
                return super().__call__(argv, env)

        world = DiesOnTheSmallestClass(cap=6.0, cap_by_arm={"retr-trunc": 12.0})
        partial = tmp_path / "rungs_partial.json"
        with pytest.raises(RuntimeError, match="killed"):
            _calibrate(tmp_path, world, _two_class_grid(), partial_out=partial)
        doc = json.loads(partial.read_text(encoding="utf-8"))
        assert doc["complete"] is False and doc["schema"] == rc.RUNG_CALIBRATION_SCHEMA
        assert doc["rungs"] and all(rec["label"] == "ESTIMATED" for rec in doc["rungs"].values())
        assert doc["classes"]["348"]["rungs"] == {}
        with pytest.raises(rc.PlanError, match="INCOMPLETE"):
            rc.load_rung_calibration(partial)
        # a run that finishes writes complete: true and loads
        full_dir = tmp_path / "full"
        world2 = FakeWorld(cap=6.0, cap_by_arm={"retr-trunc": 12.0})
        full = _calibrate(full_dir, world2, _two_class_grid(), partial_out=full_dir / "rungs.json")
        assert full["complete"] is True
        (full_dir / "rungs_final.json").write_text(json.dumps(full), encoding="utf-8")
        assert rc.load_rung_calibration(full_dir / "rungs_final.json").small_lambda_at(0.5) is not None

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
        # one window attempt per rung plus its retry (review 2026-10-13 item 3)
        assert len(world.runner_calls()) == 2 * (1 + rc.LADDER_WINDOW_RETRIES)
        # the second rung restarts from the floor start (no previous lambda*)
        assert doc["rungs"]["0.5"]["start_qps"] == doc["rungs"]["1"]["start_qps"]

    def test_fail_fast_stops_the_ladder_at_the_first_rung_that_is_not_estimated(self, tmp_path):
        # Review 2026-10-13 item 3 (S0F-77): a dead runner fails the first rung;
        # with fail_fast the second rung is never walked, the document is
        # incomplete and names the stop; without it every rung is walked.
        world = FakeWorld(cap=6.0, write_rows=False)
        doc = _calibrate(tmp_path / "ff", world, _pressure_grid(), fail_fast=True)
        assert list(doc["rungs"]) == ["1"] and doc["rungs"]["1"]["label"] == rc.RUNG_LABEL_PROBE_FAILED
        assert doc["complete"] is False and doc["fail_fast_stop"] == "anchor r=1: PROBE_FAILED"
        # one window plus its one retry (item 3), then the stop
        assert len(world.runner_calls()) == 1 + rc.LADDER_WINDOW_RETRIES
        assert world.verbs()[-1] == "stop"
        walked = _calibrate(tmp_path / "walk", FakeWorld(cap=6.0, write_rows=False), _pressure_grid())
        assert list(walked["rungs"]) == ["1", "0.5"] and walked["fail_fast_stop"] is None and walked["complete"] is True

    def test_a_window_that_dies_once_is_retried_at_a_fresh_root_and_the_rung_estimates(self, tmp_path):
        # Review 2026-10-13 item 3: one dead window used to label the rung
        # PROBE_FAILED; it is retried once, and the rung measures normally.
        class _DiesOnce(FakeWorld):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.died = 0

            def __call__(self, argv, env):
                if "--campaign-root" in argv and self.died == 0:
                    self.died += 1
                    self.calls.append((list(argv), dict(env)))
                    return 1  # the runner died before writing rows
                return super().__call__(argv, env)

        world = _DiesOnce(cap=6.0)
        doc = _calibrate(tmp_path, world, _pressure_grid())
        assert {rec["label"] for rec in doc["rungs"].values()} == {"ESTIMATED"}
        steps = doc["rungs"]["1"]["steps"]
        assert steps[0]["phase"] == "ladder" and steps[0]["attempt"] == 2  # the retry's record
        assert all(s["attempt"] == 1 for s in steps[1:]) and len(world.runner_calls()) == 1 + sum(
            len(r["steps"]) for r in doc["rungs"].values())
        roots = {s["root"] for s in doc["rungs"]["1"]["steps"]}
        assert len(roots) == len(doc["rungs"]["1"]["steps"])  # the retry took a fresh root

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

    def test_one_class_grid_writes_the_anchor_class_only(self, tmp_path):
        doc = _calibrate(tmp_path, FakeWorld(cap=6.0), _pressure_grid())
        assert doc["schema"] == "cage-rung-calibration-v2" and doc["two_anchor_adr"] == "ADR-0156"
        assert doc["anchor_seq_tokens"] == _ANCHOR_SEQ_TOKENS
        assert list(doc["classes"]) == [str(_ANCHOR_SEQ_TOKENS)]
        assert doc["classes"][str(_ANCHOR_SEQ_TOKENS)]["role"] == "anchor"
        assert doc["classes"][str(_ANCHOR_SEQ_TOKENS)]["rungs"] == doc["rungs"]
        assert doc["interpolation"]["anchor_only"] is False

    def test_smallest_class_ladder_starts_one_chain_step_below_the_anchor_rate_and_feeds_the_interpolation(self, tmp_path):
        # ADR-0156: B1 (4,779 tokens) and B11 retr-trunc (348 tokens) on F2; the
        # fake server sustains 6 rps on the anchor and 24 rps on the small class
        grid = _two_class_grid()
        world = FakeWorld(cap=6.0, cap_by_arm={"retr-trunc": 24.0})
        doc = _calibrate(tmp_path, world, grid)
        assert set(doc["classes"]) == {str(_ANCHOR_SEQ_TOKENS), "348"}
        small = doc["classes"]["348"]
        assert (small["role"], small["arm"], small["baseline_id"], small["family"], small["prefix_mode"]) == (
            "smallest", "retr-trunc", "B11", "F2", "OFF",
        )
        assert small["note"] is None  # prefix OFF: the replayed ladder is clean
        anchor = doc["rungs"]
        for key in ("1", "0.5"):
            a, m = anchor[key], small["rungs"][key]
            assert a["label"] == m["label"] == "ESTIMATED"
            assert a["lambda_star_qps"] <= 6.0 / 0.9 * 1.01 and a["first_unsustainable_qps"] >= 6.0 / 0.9 * 0.99
            assert m["lambda_star_qps"] <= 24.0 / 0.9 * 1.01 and m["first_unsustainable_qps"] >= 24.0 / 0.9 * 0.99
            assert m["class"]["seq_tokens"] == 348 and a["class"]["seq_tokens"] == _ANCHOR_SEQ_TOKENS
            # the small class's budget is its own class budget at that r
            assert m["relaunch"]["demand_class"]["seq_tokens"] == 348
            assert int(m["relaunch"]["env"]["CAGE_KV_BUDGET_BYTES"]) == int(float(key) * ((_ANCHOR_DEMAND * 348) // _ANCHOR_SEQ_TOKENS))
            assert "--no-prefix-cache" in m["relaunch"]["argv"]
            for s in m["steps"]:
                assert Path(s["root"]).name.startswith(f"cal-vllm-smallest-r{key.replace('.', 'p')}-s")
        # the first small rung starts one chain step BELOW the anchor's lambda*
        # (Fable review 2026-10-09 HIGH 1: the anchor's ESTIMATED value is the
        # last sustainable step of its own ladder; a same-capacity class
        # started exactly there reads unsustainable by Poisson variance)
        assert small["rungs"]["1"]["start_qps"] == pytest.approx(anchor["1"]["lambda_star_qps"] / rc.LADDER_CHAIN_DIVISOR)
        assert "anchor lambda*" in small["rungs"]["1"]["start_basis"] and "one chain step below" in small["rungs"]["1"]["start_basis"]
        # the second small rung chains from the higher of the two candidates
        chained = small["rungs"]["1"]["lambda_star_qps"] / PROBE_LADDER_FACTOR ** 2
        assert small["rungs"]["0.5"]["start_qps"] == pytest.approx(max(chained, anchor["0.5"]["lambda_star_qps"] / rc.LADDER_CHAIN_DIVISOR))
        # order: the anchor's two rungs, then the small class's two (four relaunches)
        restarts = [e["CAGE_KV_BUDGET_BYTES"] for a, e in world.calls if "restart" in a]
        assert len(restarts) == 4
        small_windows = [(a, e) for a, e in world.runner_calls() if e["CAGE_CELL_ARM"] == "retr-trunc"]
        assert small_windows and all(e["CAGE_CELL_BUDGET_R"] in ("1", "0.5") for _a, e in small_windows)
        assert all("--max-context-docs" in a for a, _e in small_windows)  # the arm's own behavior argv
        # the artifact feeds the plan: B11 offers its own lambda*, B1 the anchor's
        path = _write_doc(tmp_path, doc, "rungs_vllm.json")
        cal = rc.load_rung_calibration(path)
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
            if s["cellspec"]["arm"] == "retr-trunc":
                assert s["rate_basis"] == rc.LAMBDA_BASIS_SMALL
                assert s["lambda_star_rps"] == pytest.approx(cal.small_lambda_at(r))
            else:
                assert s["rate_basis"] == rc.LAMBDA_BASIS_RUNG
                assert s["lambda_star_rps"] == pytest.approx(cal.lambda_at(r))
        alpha = plan["rung_calibration"]["artifacts"]["vllm"]["alpha"]
        for key in ("1", "0.5"):
            want = math.log(cal.small_lambda_at(float(key)) / cal.lambda_at(float(key))) / math.log(_ANCHOR_SEQ_TOKENS / 348)
            assert alpha[key] == pytest.approx(want)

    def test_anchor_only_skips_the_second_ladder_and_the_plan_refuses_it(self, tmp_path):
        grid = _two_class_grid()
        world = FakeWorld(cap=6.0, cap_by_arm={"retr-trunc": 24.0})
        doc = _calibrate(tmp_path, world, grid, anchor_only=True)
        assert list(doc["classes"]) == [str(_ANCHOR_SEQ_TOKENS)] and doc["interpolation"]["anchor_only"] is True
        assert all(e.get("CAGE_CELL_ARM") == "gold-fresh" for _a, e in world.runner_calls())
        path = _write_doc(tmp_path, doc, "rungs_vllm.json")
        orig = rc.SESSION_GRIDS
        rc.SESSION_GRIDS = {"a": grid}
        try:
            with pytest.raises(rc.PlanError, match="carries no smallest-class ladder"):
                rc.build_plan(
                    "a", rc.load_floor_table(tmp_path / "ft.json"), window_duration_s=60.0,
                    calibrations={"vllm": tmp_path / "cal" / "calibration_vllm.json"},
                    rung_calibrations={"vllm": path},
                )
        finally:
            rc.SESSION_GRIDS = orig

    def test_a_defective_small_ladder_reads_none_sustainable(self, tmp_path):
        # the small class starts at the anchor's rate; a server that sustains LESS
        # on the small class labels its rung NONE_SUSTAINABLE (loud), never a value
        grid = _two_class_grid()
        doc = _calibrate(tmp_path, FakeWorld(cap=6.0, cap_by_arm={"retr-trunc": 1.0}), grid, rungs=(1.0,))
        assert doc["rungs"]["1"]["label"] == "ESTIMATED"
        assert doc["classes"]["348"]["rungs"]["1"]["label"] == "NONE_SUSTAINABLE"


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
rate = float(flag("--rate"))
cap = float(os.environ.get("STUB_CAPACITY_QPS_" + os.environ.get("CAGE_CELL_ARM", "").replace("-", "_").upper(),
                           os.environ["STUB_CAPACITY_QPS"]))
spec = derive_cell_spec(baseline=flag("--baseline"), baseline_label=flag("--baseline-label"),
                        backend=flag("--backend"), model=flag("--model"), env=os.environ)
wdir = os.path.join(flag("--campaign-root"), "cells", spec.to_row_key(), "window_%s-01" % flag("--dataset"))
os.makedirs(wdir)
n = max(1, round(rate * 75.0)); share = min(1.0, cap / rate)
kept = [i * 75.0 / n for i in range(n) if i * 75.0 / n >= 10.0]
n_ok = round(len(kept) * share)
with open(os.path.join(wdir, "requests.jsonl"), "w", encoding="utf-8") as fh:
    for k, t in enumerate(kept):
        fh.write(json.dumps({"arrival_s": t, "ok": k < n_ok, "scheduled_ts": 1000.0 + t, "completion_ts": 1000.0 + t}) + "\\n")
sys.exit(0)
""".replace("__REPO_ROOT__", repr(str(rc.REPO_ROOT)))


def _cli_calibrate(tmp_path: Path, monkeypatch, capacity: str, extra: List[str], grid=None) -> Tuple[int, Path]:
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
    rc.SESSION_GRIDS = {"a": grid or _pressure_grid()}
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

    def test_two_anchors_end_to_end_with_a_subprocess_runner(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("STUB_CAPACITY_QPS_RETR_TRUNC", "12.0")
        code, out = _cli_calibrate(tmp_path, monkeypatch, "4.0", ["--rungs", "0.5", "--start-qps", "2"], grid=_two_class_grid())
        assert code == 0
        doc = json.loads(out.read_text(encoding="utf-8"))
        small = doc["classes"]["348"]["rungs"]["0.5"]
        assert small["label"] == "ESTIMATED" and small["lambda_star_qps"] <= 12.0 / 0.9 * 1.01
        assert small["first_unsustainable_qps"] >= 12.0 / 0.9 * 0.99
        assert all(Path(s["root"]).name.startswith("cal-vllm-smallest-r0p5-s") for s in small["steps"])
        text = capsys.readouterr().out
        assert "anchor class gold-fresh (4779 tokens)" in text and "smallest class retr-trunc (348 tokens)" in text
        cal = rc.load_rung_calibration(out)
        assert cal.small_lambda_at(0.5) == small["lambda_star_qps"]

    def test_exit_1_when_the_small_class_is_not_estimated(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("STUB_CAPACITY_QPS_RETR_TRUNC", "0.01")
        code, out = _cli_calibrate(tmp_path, monkeypatch, "4.0", ["--rungs", "0.5", "--start-qps", "2"], grid=_two_class_grid())
        assert code == 1
        doc = json.loads(out.read_text(encoding="utf-8"))
        assert doc["rungs"]["0.5"]["label"] == "ESTIMATED"
        assert doc["classes"]["348"]["rungs"]["0.5"]["label"] == "NONE_SUSTAINABLE"
        assert "smallest 348 tokens r=0.5" in capsys.readouterr().err


def test_all_classes_ladders_each_mid_class_at_its_pressure_rungs_b9(tmp_path):
    """S0F-88 B-9 (owner ruling R5): with --all-classes a class between the
    smallest and the anchor gets its own ladder at its pressure rungs, and the
    plan offers it a fraction of THAT lambda*, never the interpolation."""
    grid = _tiny_grid(
        f1_baselines=("B1",), f2_baselines=("B1", "B6", "B11"), f2_budgets=(1.0, 0.5), f2_rates=(0.85,),
    )
    mid_cells = [c for c in rc.enumerate_cells(grid) if c.baseline_id == "B6" and c.spec.family == "F2"]
    assert mid_cells
    mid_arm = mid_cells[0].spec.arm
    mid_seq = rc.demand_seq_tokens(grid, mid_cells[0].spec)
    assert 348 < mid_seq < _ANCHOR_SEQ_TOKENS
    world = FakeWorld(cap=6.0, cap_by_arm={"retr-trunc": 24.0, mid_arm: 12.0})
    doc = _calibrate(tmp_path, world, grid, all_classes=True)
    mid = doc["classes"][str(mid_seq)]
    assert (mid["role"], mid["arm"], mid["baseline_id"]) == ("measured", mid_arm, "B6")
    for key in ("1", "0.5"):
        rec = mid["rungs"][key]
        assert rec["label"] == "ESTIMATED"
        assert rec["lambda_star_qps"] <= 12.0 / 0.9 * 1.01 and rec["first_unsustainable_qps"] >= 12.0 / 0.9 * 0.99
        for s in rec["steps"]:
            assert Path(s["root"]).name.startswith(f"cal-vllm-c{mid_seq}-r{key.replace('.', 'p')}-s")
    cal = rc.load_rung_calibration(_write_doc(tmp_path, doc, "rungs_vllm.json"))
    lam, basis, source = rc.rung_lambda_for_class(cal, 1.0, mid_seq, _ANCHOR_SEQ_TOKENS)
    assert basis == rc.LAMBDA_BASIS_MEASURED and lam == mid["rungs"]["1"]["lambda_star_qps"]
    assert source["alpha"] is None
    # without --all-classes the mid class is interpolated, as before
    plain = _calibrate(tmp_path / "plain", FakeWorld(cap=6.0, cap_by_arm={"retr-trunc": 24.0, mid_arm: 12.0}), grid)
    assert all(c["role"] != "measured" for c in plain["classes"].values())
    cal_plain = rc.load_rung_calibration(_write_doc(tmp_path / "plain", plain, "rungs_vllm.json"))
    assert rc.rung_lambda_for_class(cal_plain, 1.0, mid_seq, _ANCHOR_SEQ_TOKENS)[1] == rc.LAMBDA_BASIS_INTERPOLATED


def _prefix_on_grid():
    """S0F-88 B-8: one prefix-ON pressure configuration (B2, F3 at r=0.5)."""
    return _tiny_grid(
        f1_baselines=("B1",), f3_baselines=("B2",), f3_budgets=(0.5,), f3_rates=(0.95,),
    )


def test_a_prefix_on_relief_runs_its_cells_and_any_other_label_still_fails_b8(
    tmp_path, floor_table, stub, monkeypatch, capsys
):
    """S0F-88 B-8 (owner ruling R4): a prefix-ON configuration offers the OFF
    twin's rate and budget; reading UNPRESSURED there is the measured relief,
    so its pressure cells run. PAST_CLIFF on it still fails the class."""
    plan = _stub_plan(_prefix_on_grid(), floor_table, stub.cmd)
    (dry,) = _dry_steps(plan)
    assert dry["serving"]["prefix_mode"] == "ON"
    monkeypatch.setenv("STUB_REGIME_LABEL", "UNPRESSURED")
    root = _run_root(tmp_path)
    assert rc.run_plan(plan, root) == 0
    assert "DRY WINDOW PREFIX RELIEF" in capsys.readouterr().out
    assert sorted(c["env"]["CAGE_CELL_FAMILY"] for c in _cell_calls(stub)) == ["F1", "F3"]
    assert all(_sentinel(root, s) == "" for s in _cells(plan))
    monkeypatch.setenv("STUB_REGIME_LABEL", "PAST_CLIFF")
    root2 = root.parent / "run-002"
    assert rc.run_plan(plan, root2) == rc.EXIT_DRY_WINDOW_FAILED
    assert any("reason=dry-window-PAST_CLIFF" in _sentinel(root2, s) for s in _cells(plan) if s["family"] == "F3")


def test_a_prefix_off_unpressured_dry_window_still_fails_b8(tmp_path, floor_table, stub, monkeypatch):
    """The relief rule is prefix-ON only: the anchor (prefix OFF) reading
    UNPRESSURED still skips its class (ADR-0153 unchanged there)."""
    plan = _stub_plan(_pressure_grid(), floor_table, stub.cmd)
    monkeypatch.setenv("STUB_REGIME_LABEL", "UNPRESSURED")
    assert rc.run_plan(plan, _run_root(tmp_path)) == rc.EXIT_DRY_WINDOW_FAILED


# --------------------------------------------------------------------------- #
# ADR-0169 (decision 5, 2026-10-13): the RULER instrument's own rate ladder
# --------------------------------------------------------------------------- #


def _ruler_grid():
    """The anchor (B1 gold-fresh) on F2 with the RULER instrument paired on it
    (ADR-0096), vLLM only; no class below the anchor."""
    return _tiny_grid(
        f1_baselines=("B1",), f2_baselines=("B1",), f2_budgets=(1.0, 0.5), f2_rates=(0.85,),
        f2_ruler_baselines=("B1",), f2_ruler_tasks=("niah_multikey", "qa"),
    )


def _ruler_plan(tmp_path: Path, doc: Dict[str, Any], grid) -> Dict[str, Any]:
    path = _write_doc(tmp_path, doc, "rungs_vllm.json")
    floor = rc.load_floor_table(tmp_path / "ft.json")
    orig = rc.SESSION_GRIDS
    rc.SESSION_GRIDS = {"a": grid}
    try:
        return rc.build_plan(
            "a", floor, window_duration_s=60.0,
            calibrations={"vllm": tmp_path / "cal" / "calibration_vllm.json"},
            rung_calibrations={"vllm": path},
        )
    finally:
        rc.SESSION_GRIDS = orig


class TestRulerLadder:
    def test_ruler_ladder_starts_from_its_floor_and_the_plan_offers_its_own_lambda(self, tmp_path):
        """The RULER ladder runs after the anchor's under the same relaunch, starts
        from the floor window's unloaded rows, chains its own rungs, and the plan
        offers RULER cells a fraction of THAT lambda* while the Qasper twin keeps
        the anchor's (S0 attempt 4 offered the twin's 15.9 rps against 0.46 rps)."""
        grid = _ruler_grid()
        world = FakeWorld(cap=6.0, ruler_cap=0.4, ruler_floor=(2.3, 0.018))
        doc = _calibrate(tmp_path, world, grid)
        ruler = doc["classes"][str(rc.RULER_RATE_CLASS_TOKENS)]
        assert (ruler["role"], ruler["arm"], ruler["baseline_id"], ruler["dataset"]) == (
            rc.RULER_ROLE, "gold-fresh", "B1", "ruler",
        )
        assert ruler["ruler_task"] == "niah_multikey" and ruler["demand_seq_tokens"] == _ANCHOR_SEQ_TOKENS
        assert ruler["seq_tokens"] == rc.RULER_RATE_CLASS_TOKENS == 32768
        floor = ruler["floor"]
        assert floor["ttft_s"] == pytest.approx(2.3) and floor["tpot_s"] == pytest.approx(0.018)
        assert floor["n_requests"] == rc.RULER_FLOOR_ARRIVALS and floor["rate_qps"] == rc.RULER_FLOOR_RATE_QPS
        expected_start = floor_start_qps(
            FloorMeasurement(ttft_s=2.3, tpot_s=0.018, n_requests=12, statistic="median"),
            max_tokens=rc.LADDER_START_DECODE_TOKENS,
        )
        assert ruler["rungs"]["1"]["start_qps"] == pytest.approx(expected_start / rc.LADDER_CHAIN_DIVISOR)
        assert "floor_start_qps of the RULER floor window" in ruler["rungs"]["1"]["start_basis"]
        assert ruler["ladder_window_s"] == rc.RULER_PROBE_WINDOW_S == 300.0
        for rec in ruler["rungs"].values():
            assert all(step["window_s"] == rc.RULER_PROBE_WINDOW_S for step in rec["steps"])
        for rec in doc["rungs"].values():
            assert all(step["window_s"] == PROBE_WINDOW_S for step in rec["steps"])
        for key in ("1", "0.5"):
            rec = ruler["rungs"][key]
            assert rec["label"] == "ESTIMATED"
            assert rec["lambda_star_qps"] <= 0.4 / 0.9 * 1.01 and rec["first_unsustainable_qps"] >= 0.4 / 0.9 * 0.99
            assert rec["class"]["ruler_task"] == "niah_multikey" and rec["class"]["dataset"] == "ruler"
            for s in rec["steps"]:
                assert Path(s["root"]).name.startswith(f"cal-vllm-ruler-r{key.replace('.', 'p')}-s")
        # the tighter rung chains from the RULER rung above it, never from the anchor's 6 rps
        assert ruler["rungs"]["0.5"]["start_qps"] == pytest.approx(
            ruler["rungs"]["1"]["lambda_star_qps"] / rc.LADDER_CHAIN_DIVISOR
        )
        # the anchor's ladder is untouched
        for rec in doc["rungs"].values():
            assert rec["label"] == "ESTIMATED" and rec["lambda_star_qps"] >= 6.0 / 0.9 * 0.9
        # the floor window: W arrivals at the floor rate, the RULER flags, no duration
        floors = [(a, e) for a, e in world.runner_calls() if "--arrival-count" in a]
        assert len(floors) == 1
        argv, env = floors[0]
        assert argv[argv.index("--arrival-count") + 1] == str(rc.RULER_FLOOR_ARRIVALS)
        assert float(argv[argv.index("--rate") + 1]) == pytest.approx(rc.RULER_FLOOR_RATE_QPS)
        assert "--duration-s" not in argv and argv[argv.index("--dataset") + 1] == "ruler"
        assert argv[argv.index("--ruler-task") + 1] == "niah_multikey"
        assert Path(argv[argv.index("--campaign-root") + 1]).name == "cal-vllm-ruler-r1-s00"
        assert env[rc.LADDER_REPLAY_ENV] == "1" and env["CAGE_CELL_ARM"] == "gold-fresh"
        # every RULER ladder window is the RULER cell at the 75 s duration
        ruler_windows = [a for a, _e in world.runner_calls() if a[a.index("--dataset") + 1] == "ruler" and "--duration-s" in a]
        assert ruler_windows and all("--ruler-task" in a and a[a.index("--duration-s") + 1] == f"{rc.RULER_PROBE_WINDOW_S:g}" for a in ruler_windows)
        anchor_windows = [a for a, _e in world.runner_calls() if a[a.index("--dataset") + 1] == "qasper"]
        assert anchor_windows and all(a[a.index("--duration-s") + 1] == f"{PROBE_WINDOW_S:g}" for a in anchor_windows)
        # order: the anchor's two rungs, then the RULER ladder's two rungs (four relaunches)
        verbs = world.verbs()
        assert verbs.count("restart") == 4
        datasets = [a[a.index("--dataset") + 1] for a, _e in world.runner_calls()]
        assert datasets.index("ruler") > max(i for i, d in enumerate(datasets) if d == "qasper")
        # the plan: RULER cells offer a fraction of the RULER lambda*, the twin the anchor's
        plan = _ruler_plan(tmp_path, doc, grid)
        cal = rc.load_rung_calibration(_write_doc(tmp_path, doc, "rungs_vllm.json"))
        ruler_steps = [s for s in _pressure_cells(plan) if s["dataset"] == "ruler"]
        twins = {s["row_key"]: s for s in _pressure_cells(plan) if s["dataset"] == "qasper"}
        assert len(ruler_steps) == 4  # 2 rungs x 2 tasks
        for s in ruler_steps:
            r = s["cellspec"]["budget_r"]
            assert s["rate_basis"] == rc.LAMBDA_BASIS_RULER
            assert s["lambda_star_rps"] == pytest.approx(cal.ruler_lambda_at(r))
            assert s["offered_rate_rps"] == pytest.approx(0.85 * cal.ruler_lambda_at(r))
            source = s["lambda_star_source"]
            assert source["rate_class_seq_tokens"] == 32768 and source["seq_tokens_class"] == _ANCHOR_SEQ_TOKENS
            assert source["adr"] == rc.RULER_RATE_ADR and source["gold_lambda_star_qps"] == pytest.approx(cal.lambda_at(r))
            twin = twins[s["row_key"]]  # the pairing by row key holds (ADR-0096)
            assert twin["lambda_star_rps"] == pytest.approx(cal.lambda_at(r)) and twin["rate_basis"] == rc.LAMBDA_BASIS_RUNG
            assert twin["lambda_star_rps"] > 10 * s["lambda_star_rps"]
        header = plan["rung_calibration"]["artifacts"]["vllm"]["ruler_class"]
        assert header["seq_tokens"] == 32768 and header["floor"]["ttft_s"] == pytest.approx(2.3)

    def test_a_loaded_floor_window_labels_the_rung_and_names_the_operator_start(self, tmp_path):
        world = FakeWorld(cap=6.0, ruler_cap=0.4, ruler_floor_loaded=True)
        doc = _calibrate(tmp_path, world, _ruler_grid(), fail_fast=True)
        ruler = doc["classes"][str(rc.RULER_RATE_CLASS_TOKENS)]
        rec = ruler["rungs"]["1"]
        assert rec["label"] == rc.RUNG_LABEL_PROBE_FAILED and rec["lambda_star_qps"] is None
        assert "unloaded ok rows" in rec["error"] and "--ruler-start-qps" in rec["error"]
        assert ruler["floor"] is None and "0.5" not in ruler["rungs"]  # fail-fast stopped the ladder
        assert doc["fail_fast_stop"].startswith(f"{rc.RULER_ROLE} r=1:") and doc["complete"] is False
        # the anchor's rungs were measured before it
        assert all(rec["label"] == "ESTIMATED" for rec in doc["rungs"].values())

    def test_an_operator_start_skips_the_floor_window(self, tmp_path):
        world = FakeWorld(cap=6.0, ruler_cap=0.4)
        doc = _calibrate(tmp_path, world, _ruler_grid(), ruler_start_qps=0.3)
        ruler = doc["classes"][str(rc.RULER_RATE_CLASS_TOKENS)]
        assert not [a for a, _e in world.runner_calls() if "--arrival-count" in a]
        assert ruler["floor"] is None and ruler["start_basis"] == "--ruler-start-qps 0.3 (operator)"
        assert ruler["rungs"]["1"]["start_qps"] == pytest.approx(0.3)
        assert all(rec["label"] == "ESTIMATED" for rec in ruler["rungs"].values())
        with pytest.raises(rc.PlanError, match="--ruler-start-qps"):
            _calibrate(tmp_path / "bad", FakeWorld(cap=6.0), _ruler_grid(), ruler_start_qps=0.0)

    def test_the_plan_refuses_an_artifact_without_the_ruler_ladder(self, tmp_path):
        """--anchor-only skips the RULER ladder; a plan with executable RULER
        cells then refuses the artifact instead of offering the twin's rate."""
        grid = _ruler_grid()
        doc = _calibrate(tmp_path, FakeWorld(cap=6.0, ruler_cap=0.4), grid, anchor_only=True)
        assert str(rc.RULER_RATE_CLASS_TOKENS) not in doc["classes"]
        with pytest.raises(rc.PlanError, match="no ESTIMATED RULER lambda"):
            _ruler_plan(tmp_path, doc, grid)
        # a RULER rung that failed refuses the same way (never the twin's rate)
        doc2 = _calibrate(tmp_path / "two", FakeWorld(cap=6.0, ruler_cap=0.4, ruler_floor_loaded=True), grid)
        with pytest.raises(rc.PlanError, match=r"r=1 \(PROBE_FAILED\)"):
            _ruler_plan(tmp_path / "two", doc2, grid)

    def test_a_grid_without_ruler_cells_ladders_nothing_extra(self, tmp_path):
        world = FakeWorld(cap=6.0, ruler_cap=0.4)
        doc = _calibrate(tmp_path, world, _pressure_grid())
        assert set(doc["classes"]) == {str(_ANCHOR_SEQ_TOKENS)}
        assert not [a for a, _e in world.runner_calls() if "--arrival-count" in a]
        assert world.verbs().count("restart") == 2

    def test_the_loader_refuses_a_malformed_ruler_entry(self, tmp_path):
        base = _calibrate(tmp_path, FakeWorld(cap=6.0, ruler_cap=0.4), _ruler_grid())
        key = str(rc.RULER_RATE_CLASS_TOKENS)
        bad = json.loads(json.dumps(base))
        del bad["classes"][key]["demand_seq_tokens"]
        with pytest.raises(rc.PlanError, match="demand_seq_tokens"):
            rc.load_rung_calibration(_write_doc(tmp_path, bad, "bad1.json"))
        two = json.loads(json.dumps(base))
        two["classes"]["99999"] = {**two["classes"][key], "seq_tokens": 99999}
        with pytest.raises(rc.PlanError, match="two 'ruler' entries"):
            rc.load_rung_calibration(_write_doc(tmp_path, two, "bad2.json"))
        floor = json.loads(json.dumps(base))
        floor["classes"][key]["floor"] = {"ttft_s": -1.0, "tpot_s": 0.01}
        with pytest.raises(rc.PlanError, match="floor must be null"):
            rc.load_rung_calibration(_write_doc(tmp_path, floor, "bad3.json"))
        cal = rc.load_rung_calibration(_write_doc(tmp_path, base, "ok.json"))
        assert cal.ruler_seq_tokens == 32768 and cal.ruler_task == "niah_multikey"
        assert cal.ruler_floor["ttft_s"] == pytest.approx(2.3) and cal.ruler_lambda_at(0.5) is not None

    def test_a_refused_floor_window_is_retried_once_at_a_fresh_root(self, tmp_path):
        """Review 2026-10-13 (ADR-0169 MEDIUM 1): a floor window with too few
        unloaded rows (Poisson overlap, runner exit 0) or one the runner refused
        is retried once at a fresh root; the second refusal labels the rung."""
        # exit 1 then success: the floor lands at s01 and the rung estimates
        world = FakeWorld(cap=6.0, ruler_cap=0.4, ruler_floor_refusals=1)
        doc = _calibrate(tmp_path, world, _ruler_grid())
        ruler = doc["classes"][str(rc.RULER_RATE_CLASS_TOKENS)]
        floors = [a for a, _e in world.runner_calls() if "--arrival-count" in a]
        assert len(floors) == 2
        assert [Path(a[a.index("--campaign-root") + 1]).name for a in floors] == ["cal-vllm-ruler-r1-s00", "cal-vllm-ruler-r1-s01"]
        assert ruler["floor"]["root"].endswith("cal-vllm-ruler-r1-s01") and ruler["rungs"]["1"]["label"] == "ESTIMATED"
        # exit 0 with too few unloaded rows is retried the same way, and two refusals label the rung
        # (fail-fast: without it a RULER rung that failed before any lambda* measures
        # the floor again at the next rung, two more windows, as the docstring says)
        few = FakeWorld(cap=6.0, ruler_cap=0.4, ruler_floor_unloaded_rows=rc.RULER_FLOOR_MIN_ROWS - 1)
        doc2 = _calibrate(tmp_path / "few", few, _ruler_grid(), fail_fast=True)
        assert len([a for a, _e in few.runner_calls() if "--arrival-count" in a]) == 2
        assert doc2["classes"][str(rc.RULER_RATE_CLASS_TOKENS)]["rungs"]["1"]["label"] == rc.RUNG_LABEL_PROBE_FAILED
        assert doc2["fail_fast_stop"].startswith(f"{rc.RULER_ROLE} r=1:")
        # exactly the minimum of unloaded rows passes (LOW 4: the threshold)
        enough = FakeWorld(cap=6.0, ruler_cap=0.4, ruler_floor_unloaded_rows=rc.RULER_FLOOR_MIN_ROWS)
        doc3 = _calibrate(tmp_path / "enough", enough, _ruler_grid())
        ruler3 = doc3["classes"][str(rc.RULER_RATE_CLASS_TOKENS)]
        assert ruler3["floor"]["n_requests"] == rc.RULER_FLOOR_MIN_ROWS and ruler3["rungs"]["1"]["label"] == "ESTIMATED"

    def test_the_plan_refuses_a_ruler_ladder_of_another_request_shape(self, tmp_path):
        """Review 2026-10-13 (ADR-0169 LOW 3): the RULER entry must be SHAPE-32K."""
        grid = _ruler_grid()
        doc = _calibrate(tmp_path, FakeWorld(cap=6.0, ruler_cap=0.4), grid)
        key = str(rc.RULER_RATE_CLASS_TOKENS)
        odd = json.loads(json.dumps(doc))
        odd["classes"]["16384"] = {**odd["classes"].pop(key), "seq_tokens": 16384}
        with pytest.raises(rc.PlanError, match="request shape is 16384 tokens"):
            _ruler_plan(tmp_path, odd, grid)

    def test_ruler_ladder_windows_pin_the_dataset_manifests_hash(self, tmp_path):
        """S0 attempt 5 (2026-10-14): the SGLang RULER floor window was refused in
        campaign mode ("dataset_manifests_sha256 unresolvable"): a ladder window
        is the first emitter of its root and RULER has no query manifest. The
        RULER ladder windows now carry the sha256 of the F2 dataset's manifest,
        the value the run's RULER cells inherit from the first Qasper cell."""
        import hashlib

        manifest = Path(__file__).resolve().parents[1] / "data" / "manifests" / "qasper_100x3_seed42.json"
        grid = _tiny_grid(
            f1_baselines=("B1",), f2_baselines=("B1",), f2_budgets=(1.0, 0.5), f2_rates=(0.85,),
            f2_ruler_baselines=("B1",), f2_ruler_tasks=("niah_multikey",), window_requests=50,
        )
        world = FakeWorld(cap=6.0, ruler_cap=0.4)
        doc = _calibrate(tmp_path, world, grid, query_manifests={"qasper": manifest})
        assert doc["classes"][str(rc.RULER_RATE_CLASS_TOKENS)]["rungs"]["1"]["label"] == "ESTIMATED"
        want = hashlib.sha256(manifest.read_bytes()).hexdigest()
        ruler_envs = [e for a, e in world.runner_calls() if a[a.index("--dataset") + 1] == "ruler"]
        anchor_calls = [(a, e) for a, e in world.runner_calls() if a[a.index("--dataset") + 1] == "qasper"]
        assert ruler_envs and all(e["CAGE_DATASET_MANIFESTS_SHA256"] == want for e in ruler_envs)
        # the anchor's windows resolve the hash from their --query-manifest, as before
        assert anchor_calls and all("--query-manifest" in a and "CAGE_DATASET_MANIFESTS_SHA256" not in e for a, e in anchor_calls)
        # without a registered F2 manifest nothing is pinned (the runner refuses loudly, as before)
        plain = FakeWorld(cap=6.0, ruler_cap=0.4)
        _calibrate(tmp_path / "plain", plain, _ruler_grid())
        assert all("CAGE_DATASET_MANIFESTS_SHA256" not in e for _a, e in plain.runner_calls())
