"""Batch C fixes of 2026-10-09 (gap triage of DECISIONS.md + the Fable review
of the Batch C diff), each pinned by the behavior it changes:

- W16 (charter section 6.3): the section 6.1 timely gate grades an open-loop
  row on ``ttft_from_scheduled_ms`` (coordinated omission), a closed-loop row
  on ``ttft_ms``, and refuses a completed open-loop row that carries no
  scheduled clock; the send-clock G and Y ride beside as a sensitivity.
- ADR-0063: a non-completion carries no timing (the adapter's 0.0 stamp is
  dropped by the campaign loader, counted).
- ADR-0089 / section 9.5: a per-query metric resolves to its registered margin
  FAMILY.
- Stage 0 registration preflight on the Mac (manifest coverage, staged
  datasets, anchor shape, LMCache need) and the pool-vs-max_model_len rule
  (gap triage C1) in its three seats.
- Fable HIGH 1: the smallest-class ladder starts one chain step below the
  anchor's lambda*, and the two-anchor slope has a measurement tolerance.
- Fable LOW 4 / LOW 8: a floor row's prediction is a positive finite rate; the
  span summary reports the true median.
- ADR-0042 (H5): vLLM logprobs are opt-in, also on an instance built without
  ``__init__``.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT / "scripts" / "4_analysis"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_campaign_analysis as rca  # noqa: E402
from src.inference.vllm_adapter import VLLMAdapter  # noqa: E402


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rc = _load("run_campaign_batch_c", REPO_ROOT / "scripts" / "3_run" / "run_campaign.py")
dist_tests = _load("test_dist_contrasts_for_batch_c", REPO_ROOT / "tests" / "test_dist_contrasts.py")
analysis_tests = _load("test_campaign_analysis_for_batch_c", REPO_ROOT / "tests" / "test_campaign_analysis.py")
planner_tests = _load("test_run_campaign_for_batch_c", REPO_ROOT / "tests" / "test_run_campaign.py")

MANIFESTS = {
    "squad_v2": REPO_ROOT / "data" / "manifests" / "squad_v2_50x3_seed42.json",
    "qasper": REPO_ROOT / "data" / "manifests" / "qasper_50x3_seed42.json",
}
CHARTER_DATASETS = ("squad_v2", "musique", "qasper")


# ---------------------------------------------------------------------------
# W16: the coordinated-omission clock
# ---------------------------------------------------------------------------


class TestW16Clock:
    def test_scheduled_clock_wins_on_an_open_loop_row(self) -> None:
        row = {"workload_mode": "open_loop", "ok": True, "ttft_ms": 500.0, "ttft_from_scheduled_ms": 4500.0}
        assert rca._slo_ttft_ms(row) == (4500.0, rca.TTFT_BASIS_SCHEDULED)

    def test_a_closed_loop_row_keeps_the_send_clock(self) -> None:
        assert rca._slo_ttft_ms({"ok": True, "ttft_ms": 500.0}) == (500.0, rca.TTFT_BASIS_SEND)
        assert rca._slo_ttft_ms({"ok": False, "ttft_ms": None}) == (None, rca.TTFT_BASIS_SEND)

    def test_a_completed_open_loop_row_without_the_scheduled_clock_refuses(self) -> None:
        row = {"workload_mode": "open_loop", "ok": True, "ttft_ms": 500.0, "example_id": "e7"}
        with pytest.raises(rca.AnalysisError, match="coordinated-omission clock is missing"):
            rca._slo_ttft_ms(row)
        with pytest.raises(rca.AnalysisError):  # a NaN scheduled value is no clock either
            rca._slo_ttft_ms({**row, "ttft_from_scheduled_ms": float("nan")})
        # a FAILED open-loop row has no completion to grade: no refusal
        assert rca._slo_ttft_ms({"workload_mode": "open_loop", "ok": False, "ttft_ms": 0.0}) == (0.0, rca.TTFT_BASIS_SEND)

    def test_the_dist_window_is_graded_on_the_scheduled_clock_with_the_send_clock_sensitivity(self, tmp_path: Path) -> None:
        # every row answered 50 ms after the SEND but 4.5 s after the INTENDED
        # arrival (the dispatcher fell behind): the send clock passes rows the
        # scheduled clock fails, so the registered G is the lower one
        def reqs(_t: str) -> list:
            return [dist_tests._req(i, workload_mode="open_loop", ttft_ms=50.0, ttft_from_scheduled_ms=4500.0) for i in range(4)]
        run_dir, predicate_root, rows = dist_tests._dist_18_tree(tmp_path, reqs)
        metrics, _gpu, reason, clock = dist_tests._dist_metrics(run_dir, predicate_root, rows[0])
        assert reason is None and metrics is not None
        assert clock["ttft_basis_counts"] == {rca.TTFT_BASIS_SCHEDULED: 4, rca.TTFT_BASIS_SEND: 0}
        assert clock["registered_basis"] == rca.TTFT_BASIS_SCHEDULED
        sens = clock["send_clock_sensitivity"]
        assert sens["goodput_frac_registered"] == metrics.goodput_frac
        assert sens["goodput_frac_send_clock"] > sens["goodput_frac_registered"]
        assert sens["yield_frac_registered"] == metrics.yield_frac

    def test_a_defective_open_loop_row_is_a_labeled_skip_on_the_dist_pass(self, tmp_path: Path) -> None:
        def reqs(_t: str) -> list:
            return [dist_tests._req(0, workload_mode="open_loop")] + [dist_tests._req(i) for i in range(1, 4)]
        run_dir, predicate_root, rows = dist_tests._dist_18_tree(tmp_path, reqs)
        metrics, _gpu, reason, clock = dist_tests._dist_metrics(run_dir, predicate_root, rows[0])
        assert metrics is None and clock is None
        assert "coordinated-omission clock is missing" in reason and "W16" in reason

    def test_closed_loop_rows_carry_no_sensitivity_record(self, tmp_path: Path) -> None:
        run_dir, predicate_root, rows = dist_tests._dist_18_tree(tmp_path, lambda _t: [dist_tests._req(i) for i in range(4)])
        _metrics, _gpu, reason, clock = dist_tests._dist_metrics(run_dir, predicate_root, rows[0])
        assert reason is None
        assert clock["registered_basis"] == rca.TTFT_BASIS_SEND and "send_clock_sensitivity" not in clock
        assert clock["ttft_basis_counts"] == {rca.TTFT_BASIS_SCHEDULED: 0, rca.TTFT_BASIS_SEND: 4}


# ---------------------------------------------------------------------------
# ADR-0063: error rows carry no timing
# ---------------------------------------------------------------------------


def test_the_loader_drops_the_timing_of_error_rows_and_counts_them(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = analysis_tests
    spec = at.CellSpec.from_baseline("B3", model=at.MODEL)
    # stamp half of every B3 window's rows as failed requests with the adapter's
    # 0.0 placeholder (openai_chat_adapter.py:369) BEFORE the tree is sealed:
    # the pre-fix loader averaged them into the arm's TTFT
    errors: list[int] = []
    original_write_window = at._write_window

    def write_window_with_errors(wdir: Path, dataset: str, baseline: str, *, ordinal: int, evidence_extra: Any = None) -> list:
        written = original_write_window(wdir, dataset, baseline, ordinal=ordinal, evidence_extra=evidence_extra)
        if baseline == "B3":
            req_path = wdir / "requests.jsonl"
            rows = [json.loads(line) for line in req_path.read_text(encoding="utf-8").splitlines() if line]
            for i, row in enumerate(rows):
                if i % 2 == 0:
                    row.update({"ok": False, "ttft_ms": 0.0, "latency_ms": 0.0, "error": "boom"})
                    errors.append(i)
                else:
                    row["ok"] = True
            req_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        return written

    monkeypatch.setattr(at, "_write_window", write_window_with_errors)
    run_dir = at._build_run_tree(tmp_path)
    n_err = len(errors)
    assert n_err > 0
    at.org.organize_run(run_dir)
    index = at.pd.read_csv(run_dir / "index" / "cells_index.csv")
    per_query = rca.load_per_query(run_dir, index, {spec.to_row_key()})
    assert per_query.attrs["error_row_timing_dropped"] == 2 * n_err  # ttft_ms and latency_ms per error row
    # no 0.0 placeholder reached a mean: every remaining TTFT is a real reading
    assert per_query["ttft_ms"].dropna().min() >= 100.0
    assert "ttft_ms" in rca.ERROR_ROW_TIMING_COLUMNS and "ttft_from_scheduled_ms" in rca.ERROR_ROW_TIMING_COLUMNS


# ---------------------------------------------------------------------------
# ADR-0089: margins by family
# ---------------------------------------------------------------------------


class TestMarginFamilies:
    @pytest.fixture()
    def artifact(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        path = tmp_path / "registered_margins.json"
        path.write_text(json.dumps({"binary_predicate": 0.05, "quality_continuous": 0.05, "serving_continuous": 25.0, "window_ttft_ms": 25.0}), encoding="utf-8")
        monkeypatch.setattr(rca, "REGISTERED_MARGINS_PATH", path)
        return path

    def test_a_serving_metric_resolves_to_the_serving_family(self, artifact: Path) -> None:
        margin, record = rca.resolve_registered_margin(25.0, "ttft_ms")
        assert margin == 25.0 and record["margin_key"] == "serving_continuous"
        margin, record = rca.resolve_registered_margin(None, "grounding_score")
        assert margin == 0.05 and record["margin_key"] == "quality_continuous"

    def test_an_artifact_key_wins_over_the_family_map(self, artifact: Path) -> None:
        margin, record = rca.resolve_registered_margin(None, "window_ttft_ms")
        assert margin == 25.0 and record["margin_key"] == "window_ttft_ms"

    def test_an_unmapped_metric_with_a_cli_margin_still_refuses(self, artifact: Path) -> None:
        with pytest.raises(rca.AnalysisError, match="no registered"):
            rca.resolve_registered_margin(0.1, "bleu")
        margin, record = rca.resolve_registered_margin(None, "bleu")
        assert margin is None and record["margin_key"] is None


# ---------------------------------------------------------------------------
# Stage 0 preflight and the pool rule
# ---------------------------------------------------------------------------


class TestPreflight:
    def test_the_s0_rehearsal_passes_and_needs_lmcache(self) -> None:
        r = rc.preflight_registration_check("a", floor_avg_seq_tokens="4779", rehearsal_n=50,
                                            query_manifests=MANIFESTS, charter_datasets=CHARTER_DATASETS)
        assert r["problems"] == [] and r["needs_lmcache"] is True
        assert (r["cells"], r["executable_cells"]) == (110, 107) and r["pool_shortfalls"] == []

    def test_the_registered_session_a_refuses_on_the_two_s1_blockers(self) -> None:
        r = rc.preflight_registration_check("a", floor_avg_seq_tokens="4779",
                                            query_manifests=MANIFESTS, charter_datasets=CHARTER_DATASETS)
        text = "\n".join(r["problems"])
        assert "hotpotqa" in text and "shortfall" in text and len(r["problems"]) == 2

    def test_the_anchor_shape_and_an_unknown_session_are_problems(self) -> None:
        r = rc.preflight_registration_check("a", floor_avg_seq_tokens="4000", rehearsal_n=50, query_manifests=MANIFESTS)
        assert any("FLOOR_AVG_SEQ_TOKENS=4000" in p and "4779" in p for p in r["problems"])
        r = rc.preflight_registration_check("zz", floor_avg_seq_tokens="4779")
        assert r["cells"] == 0 and r["problems"] and r["anchor_seq_tokens"] is None

    def test_the_pool_rule_lists_every_short_configuration_at_the_common_concurrency(self) -> None:
        # the common c = 8 cannot hold one gold request at r <= 0.75 (ADR-0158 keeps the
        # anchor class at 32,768): the registered profiles must carry c = 50 (S0F-72)
        r = rc.preflight_registration_check("a", floor_avg_seq_tokens="4779",
                                            query_manifests={}, charter_datasets=(),
                                            floor_concurrency=8)
        short = r["pool_shortfalls"]
        assert short and all(s["pool_tokens"] < s["max_model_len"] == s["pool_tokens"] + s["shortfall_tokens"] for s in short)
        gold = [s for s in short if s["seq_tokens"] == 4779]
        assert gold and {s["max_model_len"] for s in gold} == {32768}
        assert max(s["budget_r"] for s in gold) == 0.75   # 8 x 4779 x 0.75 = 28,674 < 32,768
        assert any("cannot hold one request of their class's request cap" in p for p in r["problems"])

    def test_the_class_caps_clear_the_pool_rule_at_the_rehearsal_concurrency(self) -> None:
        # ADR-0158: with the per-class caps the S0 (c = 50) plan holds one request on
        # every class at every rung (the smallest pool, retr-trunc at r = 0.25 on the
        # registered grid, is 4,350 tokens against a 4,096 cap)
        for kw in (dict(rehearsal_n=50), {}):
            r = rc.preflight_registration_check("a", floor_avg_seq_tokens="4779",
                                                query_manifests=MANIFESTS, charter_datasets=CHARTER_DATASETS,
                                                floor_concurrency=50, **kw)
            assert r["pool_shortfalls"] == []
        r = rc.preflight_registration_check("b", floor_avg_seq_tokens="4779", floor_concurrency=50)
        assert r["pool_shortfalls"] == []


class TestRequestCapADR0158:
    def test_the_cap_table_follows_the_rule(self) -> None:
        grid = rc.get_session_grid("a")
        caps = {seq: rc.request_cap_tokens(grid, seq) for seq in rc.DEMAND_CLASS_MAX_SERVED_TOKENS_2026_10_08}
        assert caps == {4779: 32768, 2336: 8192, 1205: 4096, 1127: 8192, 1126: 8192, 702: 4096, 480: 4096, 348: 4096}
        for seq, cap in caps.items():
            if seq != 4779:
                served = rc.DEMAND_CLASS_MAX_SERVED_TOKENS_2026_10_08[seq]
                assert cap >= 2 * served > cap // 2 and cap & (cap - 1) == 0
        assert rc.request_cap_tokens(grid, None) == 32768
        with pytest.raises(rc.PlanError, match="no served maximum registered"):
            rc.request_cap_tokens(grid, 999)
        with pytest.raises(rc.PlanError, match="demand_class_max_served_tokens lacks"):
            rc.replace(grid, corpus_trunc_demand_seq_tokens={1400: 1205, 700: 999})

    def test_every_relaunch_of_the_rehearsal_carries_its_class_cap(self, tmp_path: Path) -> None:
        pt = planner_tests
        path = tmp_path / "floor.json"
        path.write_text(json.dumps(pt._floor_table_doc()), encoding="utf-8")
        cals = pt._write_calibrations(tmp_path / "cal", ("vllm", "sglang"))
        plan = rc.build_plan("a", rc.load_floor_table(path), window_duration_s=300.0, rehearsal_n=50,
                             query_manifests={k: Path(v) for k, v in MANIFESTS.items()},
                             calibrations=cals, rung_calibrations=pt._rungs_for(cals, "a"))
        grid = rc.rehearsal_grid(rc.get_session_grid("a"), n=50, datasets=frozenset(MANIFESTS))
        relaunches = [s for s in plan["steps"] if s["kind"] == "relaunch"]
        seen = set()
        for s in relaunches:
            seq = (s.get("demand_class") or {}).get("seq_tokens")
            assert s["max_model_len"] == rc.request_cap_tokens(grid, seq) == int(s["env"]["VLLM_MAX_MODEL_LEN"])
            seen.add(s["max_model_len"])
        assert seen == {32768, 8192, 4096}
        assert plan["serving_shapes"]["request_caps"]["348"] == 4096
        # load_plan accepts the plan and refuses a drifted cap
        plan_path = tmp_path / "plan.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        rc.load_plan(plan_path)
        drifted = json.loads(plan_path.read_text(encoding="utf-8"))
        victim = next(s for s in drifted["steps"] if s["kind"] == "relaunch" and (s.get("demand_class") or {}).get("seq_tokens") == 348)
        victim["max_model_len"] = 32768
        victim["env"]["VLLM_MAX_MODEL_LEN"] = "32768"
        plan_path.write_text(json.dumps(drifted), encoding="utf-8")
        with pytest.raises(rc.RunError, match="differs from the header's cap 4096"):
            rc.load_plan(plan_path)


class TestPoolShortfalls:
    def test_single_and_pd_pools_are_measured_in_tokens(self) -> None:
        steps = [
            {"kind": "relaunch", "engine": "vllm", "budget_r": 0.5, "topology": "single",
             "demand_class": {"seq_tokens": 348}, "budget_plan": {"budget_tokens_total": 8700, "model": "qwen3-14b", "kv_dtype": "bf16"}},
            {"kind": "relaunch", "engine": "vllm", "budget_r": None, "topology": "pd",
             "demand_class": {"seq_tokens": 4779},
             "budget_plan": {"r": 1.0, "budget_tokens_total": 60000, "model": "qwen3-14b", "kv_dtype": "bf16",
                             "pools_bytes": [163840 * 20000, 163840 * 40000]}},
            {"kind": "relaunch", "engine": "vllm", "budget_r": None, "topology": "single", "budget_plan": None},
        ]
        out = rc.pool_shortfalls(steps, 32768)
        assert [(s["engine"], s["seq_tokens"], s["budget_r"], s["pool"], s["pool_tokens"]) for s in out] == [
            ("vllm", 348, 0.5, "pool", 8700),
            ("vllm", 4779, 1.0, "prefill", 20000),  # the pd leg's r is the BudgetPlan's own
        ]
        assert out[0]["shortfall_tokens"] == 32768 - 8700
        assert rc.pool_shortfalls(steps, 8000) == []
        # ADR-0158: a relaunch carrying its own cap is measured against it
        steps[0]["max_model_len"] = 4096
        assert [s["seq_tokens"] for s in rc.pool_shortfalls(steps, 32768)] == [4779]

    def test_the_plan_refuses_a_short_pool_with_every_offender_listed(self, tmp_path: Path) -> None:
        # a floor table at the fixture demand / 40: the 348 class at r = 0.25
        # holds 400 x 348 x 0.25 / 40 tokens, far below one request
        doc = planner_tests._floor_table_doc()
        for row in doc["rows"]:
            row["demand_bytes"] = row["demand_bytes"] // 40
        path = tmp_path / "floor.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        cals = planner_tests._write_calibrations(tmp_path / "cal", ("vllm", "sglang"))
        with pytest.raises(rc.PlanError) as exc:
            rc.build_plan("a", rc.load_floor_table(path), window_duration_s=300.0,
                          calibrations=cals, rung_calibrations=planner_tests._rungs_for(cals, "a"))
        msg = str(exc.value)
        assert "cannot hold one request of their class's request cap" in msg and "class 348 tokens" in msg
        assert "vs cap 4096" in msg and "Remedies (owner, registration)" in msg


# ---------------------------------------------------------------------------
# Fable review: ladder start, alpha tolerance, floor row, median
# ---------------------------------------------------------------------------


class TestAlphaTolerance:
    def test_the_tolerance_is_one_bisected_ladder_step_in_log_log(self) -> None:
        step = (rc.PROBE_LADDER_FACTOR - 1.0) / 2 ** rc.PROBE_BISECT_STEPS
        assert rc.alpha_tolerance(4779, 348) == pytest.approx(math.log(1 + step) / math.log(4779 / 348))
        assert rc.alpha_tolerance(4779, 348) == pytest.approx(0.0276, abs=5e-4)

    def test_a_slope_inside_the_tolerance_clamps_to_zero_and_is_recorded(self) -> None:
        used, raw, clamped = rc.resolve_alpha(1.0, 4779, 0.97, 348)
        assert used == 0.0 and raw < 0 and clamped is True
        assert rc.resolve_alpha(1.0, 4779, 1.0, 348) == (0.0, 0.0, False)
        used, raw, clamped = rc.resolve_alpha(1.0, 4779, 4.0, 348)
        assert used == raw == pytest.approx(math.log(4.0) / math.log(4779 / 348)) and clamped is False

    def test_a_slope_beyond_the_tolerance_refuses_as_a_defective_ladder(self) -> None:
        with pytest.raises(rc.PlanError, match=r"alpha -0\.265 < 0 beyond the ladders' resolution"):
            rc.resolve_alpha(1.0, 4779, 0.5, 348, where="engine 'vllm' at r=1: ")


def test_a_floor_row_with_a_zero_or_infinite_prediction_refuses(tmp_path: Path) -> None:
    for bad in (0.0, float("inf")):
        doc = planner_tests._floor_table_doc()
        doc["rows"][0]["lambda_star_pred_rps"] = bad
        path = tmp_path / f"floor_{bad}.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(rc.PlanError, match="positive finite rate"):
            rc.load_floor_table(path)


def test_the_span_summary_reports_the_true_median() -> None:
    steps = [
        {"family": "F2", "blocked_on": None, "window_span_s_expected": v, "demand_class": {"seq_tokens": 348}}
        for v in (1.0, 2.0, 10.0, 100.0)
    ]
    summary = rc.window_span_summary(steps)
    assert summary["median_s"] == 6.0 and summary["executable_pressure_cells"] == 4
    assert summary["below_floor"] == 2 and summary["by_class"]["348"]["below_floor"] == 2


# ---------------------------------------------------------------------------
# ADR-0042: vLLM logprobs opt-in on an instance built without __init__
# ---------------------------------------------------------------------------


def test_vllm_logprobs_default_holds_without_init() -> None:
    payload: Dict[str, Any] = {}
    VLLMAdapter._apply_engine_chat_extras(VLLMAdapter.__new__(VLLMAdapter), payload)
    assert payload == {"chat_template_kwargs": {"enable_thinking": False}}
    assert VLLMAdapter.request_logprobs is False
