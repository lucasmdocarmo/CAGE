"""DEC-8 (2026-11-08, MyDocs/RunPod/S2_KV_POOL_AUDIT_2026-11-07.md): the
driver realizes the hybrid budget as equal zero-load attention-token room on
both engines. These tests pin the driver side: the stage 0 records at the
serving degree, the block rule of the request cap, the per-role rank bytes,
the geometry verification of a started vLLM pool, the pool-check command and
the registration report's serving degree."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts" / "3_run")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_campaign as rc  # noqa: E402
from src.orchestration.cache_budget import (  # noqa: E402
    demand_bytes,
    fixed_state_bytes_per_seq,
    plan_budget,
    vllm_blocks_for_request,
    vllm_hybrid_geometry,
)

_MODEL = "qwen3-next-80b"
_D_TP4 = 8_626_176_000  # demand_bytes at C = 50, L = 2,694, tp 4
_GEO4 = vllm_hybrid_geometry(_MODEL, 4)
_GEO2 = vllm_hybrid_geometry(_MODEL, 2)


def _hybrid_single_plan(r: float = 1.0):
    return plan_budget(model=_MODEL, engine="vllm", r=r, demand=_D_TP4, tp=4, topology="tp")


def _record(plan) -> dict:
    from dataclasses import asdict
    return json.loads(json.dumps(asdict(plan)))


# -- pool tokens and the block rule -----------------------------------------


def test_pool_tokens_read_the_role_token_room_when_recorded() -> None:
    pd = plan_budget(model=_MODEL, engine="vllm", r=1.0, demand=_D_TP4, tp=1, topology="pd", pd_split=0.5, role_tp=2)
    assert rc._pool_tokens_of(_record(pd)) == [("prefill", 174_624), ("decode", 174_624)]
    # a dense pd record without the new field keeps the byte conversion
    dense = {"model": "qwen3-14b", "kv_dtype": "bf16", "pools_bytes": [163_840 * 10, 163_840 * 12], "budget_tokens_total": 22}
    assert rc._pool_tokens_of(dense) == [("prefill", 10), ("decode", 12)]
    assert rc._pool_tokens_of({"budget_tokens_total": 175_168}) == [("pool", 175_168)]


def test_pool_shortfalls_apply_the_block_rule_on_a_hybrid_record() -> None:
    # 64 blocks of 272 hold 17,136 tokens of room, above the 16,384 cap, yet
    # one request needs 61 attention blocks plus two per state group: 67
    def step(n: int, cap: int = 16_384) -> dict:
        return {
            "engine": "vllm", "budget_r": 1.0, "topology": "single", "max_model_len": cap,
            "demand_class": {"seq_tokens": 2694},
            "budget_plan": {
                "model": _MODEL, "kv_dtype": "bf16", "budget_tokens_total": (n - 1) * 272,
                "geometry": {"tp": 4, "block_tokens": 272, "n_groups": 4, "group_size": 12, "blocks_per_rank": n},
            },
        }
    short = rc.pool_shortfalls([step(64)], 16_384)
    assert len(short) == 1
    assert (short[0]["pool_tokens"], short[0]["blocks"], short[0]["blocks_needed"], short[0]["shortfall_tokens"]) == (17_136, 64, 67, 3 * 272)
    assert rc.pool_shortfalls([step(67)], 16_384) == []
    assert vllm_blocks_for_request(_MODEL, 4, 16_384) == 67
    # a pd record carries one block count per role
    pd = step(64)
    pd["topology"] = "pd"
    pd["budget_plan"]["pools_bytes"] = [1, 1]
    pd["budget_plan"]["pools_tokens"] = [63 * 272, 66 * 272]
    pd["budget_plan"]["geometry"]["blocks_per_rank"] = [64, 67]
    short = rc.pool_shortfalls([pd], 16_384)
    assert [(s["pool"], s["blocks"]) for s in short] == [("prefill", 64)]
    # a dense record keeps the token rule
    dense = {"engine": "vllm", "budget_r": 1.0, "topology": "single", "max_model_len": 32_768,
             "demand_class": {"seq_tokens": 4779}, "budget_plan": {"budget_tokens_total": 32_000}}
    assert rc.pool_shortfalls([dense], 32_768)[0]["shortfall_tokens"] == 768


# -- the fixed state at the serving degree ----------------------------------


def test_fixed_state_bytes_take_the_serving_degree() -> None:
    assert rc._fixed_state_bytes(_MODEL, 50, 4) == 50 * 40_108_032 == 50 * fixed_state_bytes_per_seq(_MODEL, 4)
    assert rc._fixed_state_bytes(_MODEL, 50, 1) == rc._fixed_state_bytes(_MODEL, 50, 4)
    assert rc._fixed_state_bytes("qwen3-14b", 50, 4) == 0


def _floor_doc(model: str, tp, concurrency=50) -> dict:
    inputs = {"model": model, "engine": "vllm", "kv_dtype": "bf16", "grid": "reduced", "avg_seq_tokens": 2694,
              "concurrency_target": concurrency}
    if tp is not None:
        inputs["tp"] = tp
    return {"schema": "floor-table-v1", "generated_inputs": inputs,
            "rows": [{"r": 1.0, "demand_bytes": _D_TP4, "budget_bytes": _D_TP4, "lambda_star_pred_rps": 2.0}]}


def test_a_hybrid_floor_table_needs_its_serving_degree(tmp_path: Path) -> None:
    path = tmp_path / "floor.json"
    path.write_text(json.dumps(_floor_doc(_MODEL, 4)), encoding="utf-8")
    floor = rc.load_floor_table(path)
    assert floor.tp == 4
    assert rc._floor_fixed_bytes(floor) == 50 * 40_108_032
    path.write_text(json.dumps(_floor_doc(_MODEL, None)), encoding="utf-8")
    floor = rc.load_floor_table(path)
    assert floor.tp is None
    with pytest.raises(rc.PlanError, match="generated_inputs.tp"):
        rc._floor_fixed_bytes(floor)
    # the plan refuses a hybrid table at another degree than the session serves
    grid = rc.get_session_grid("cd-act1")
    path.write_text(json.dumps(_floor_doc(_MODEL, 2)), encoding="utf-8")
    with pytest.raises(rc.PlanError, match="serving_tp"):
        rc._check_floor_shape(grid, rc.load_floor_table(path))
    path.write_text(json.dumps(_floor_doc(_MODEL, 4)), encoding="utf-8")
    assert rc._check_floor_shape(grid, rc.load_floor_table(path)) == 2694
    # a dense table needs no degree and is not checked against one
    path.write_text(json.dumps(_floor_doc("qwen3-14b", None)), encoding="utf-8")
    dense = rc.load_floor_table(path)
    assert dense.tp is None and rc._floor_fixed_bytes(dense) == 0
    for bad in (0, -1, 1.5, True, "4"):
        path.write_text(json.dumps(_floor_doc(_MODEL, bad)), encoding="utf-8")
        with pytest.raises(rc.PlanError, match="generated_inputs.tp"):
            rc.load_floor_table(path)


# -- the stage 0 records ----------------------------------------------------


def test_mac_relaunch_records_size_the_hybrid_at_the_serving_degree() -> None:
    grid = rc.get_session_grid("cd-act1")
    records = rc._mac_relaunch_records(grid, rc.enumerate_cells(grid), 50)
    anchor_single = [s for s in records if s["topology"] == "single" and s["budget_r"] == 1.0
                     and s["demand_class"]["seq_tokens"] == 2694 and s["budget_plan"]["kv_dtype"] == "bf16"]
    by_engine = {s["engine"]: s["budget_plan"] for s in anchor_single}
    assert set(by_engine) == {"vllm", "sglang"}
    for plan in by_engine.values():
        assert plan["demand_bytes"] == plan["budget_bytes_total"] == _D_TP4
        assert plan["budget_tokens_total"] == 175_168
        assert plan["geometry"]["tp"] == 4 and plan["geometry"]["blocks_per_rank"] == 645
        assert plan["state_per_request_bytes"] == {"vllm_steady": 40_108_032, "vllm_worst": 80_216_064, "sglang_slot": 77_266_944}
    primary = {e: [a["args"] for a in p["engine_args"] if a["kind"] == "primary"] for e, p in by_engine.items()}
    assert primary["vllm"] == [["--kv-cache-memory-bytes", "2155806720"]]
    assert primary["sglang"] == [["--max-total-tokens", "175168"]]
    # one pd record per prefix mode (the serving configuration carries it); both plan the same pools
    pd = [s for s in records if s["topology"] == "pd" and s["budget_r"] == 1.0 and s["demand_class"]["seq_tokens"] == 2694]
    assert len(pd) == 2
    for plan in (s["budget_plan"] for s in pd):
        assert plan["pools_bytes"] == [4_313_088_000, 4_313_088_000]
        assert plan["pools_rank_bytes"] == [2_152_464_384, 2_152_464_384]
        assert plan["pools_tokens"] == [174_624, 174_624] and plan["budget_tokens_total"] == 349_248
        assert plan["geometry"]["tp"] == 2 and plan["geometry"]["blocks_per_rank"] == [322, 322]
        assert plan["gate_j"]["expected_bytes_total"] == 2 * 2 * 2_152_464_384
    # no record of the session falls short of its class cap or of two blocks
    assert rc.mac_pool_shortfalls(grid, rc.enumerate_cells(grid), 50) == []


def test_the_dense_sessions_records_are_unchanged() -> None:
    grid = rc.get_session_grid("s1")
    records = rc._mac_relaunch_records(grid, rc.enumerate_cells(grid), 50)
    for s in records:
        plan = s["budget_plan"]
        assert plan["geometry"] is None and plan["state_per_request_bytes"] is None
        if s["topology"] == "pd":
            assert plan["pools_tokens"] == [b // 163_840 for b in plan["pools_bytes"]]
            assert plan["pools_rank_bytes"] == plan["pools_bytes"]  # role tp 1 on s1


def test_pd_smoke_env_carries_the_recorded_role_rank_bytes() -> None:
    grid = rc.get_session_grid("cd-act1")
    pd = plan_budget(model=_MODEL, engine="vllm", r=1.0, demand=_D_TP4, tp=1, topology="pd", pd_split=0.5, role_tp=2)
    step = {"engine": "vllm", "topology": "pd", "budget_r": 1.0, "max_model_len": 16_384,
            "demand_class": {"seq_tokens": 2694}, "budget_plan": _record(pd)}
    env = rc.pd_smoke_env(grid, [step])["env"]
    assert (env["CAGE_KV_BUDGET_BYTES_PREFILL"], env["CAGE_KV_BUDGET_BYTES_DECODE"]) == ("2152464384", "2152464384")
    # a record without the field (a dense plan written before DEC-8) keeps the slice rule
    old = dict(step, budget_plan={"pools_bytes": [4_000, 4_000], "budget_bytes_total": 8_000})
    env = rc.pd_smoke_env(grid, [old])["env"]
    assert (env["CAGE_KV_BUDGET_BYTES_PREFILL"], env["CAGE_KV_BUDGET_BYTES_DECODE"]) == ("2000", "2000")


# -- the realized pool against the geometry ----------------------------------


def _vllm_record(tokens: int, *, conc, req=16_384, blocks=None, nbytes=1_323_565_056) -> dict:
    extra = {}
    if conc is not None:
        extra["max_concurrency"] = conc
        extra["max_concurrency_request_tokens"] = req
    if blocks is not None:
        extra["blocks"] = blocks
    return {"tokens": tokens, "bytes": nbytes, "extra": extra}


def _step(plan, engine="vllm", topology="single", tp=4, cap=16_384) -> dict:
    return {"engine": engine, "topology": topology, "tp": tp, "budget_r": plan.r, "max_model_len": cap,
            "budget_plan": _record(plan)}


def test_the_realized_pool_check_verifies_the_vllm_geometry(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CAGE_LOG_ROOT", str(tmp_path))
    (tmp_path / "vllm").mkdir()
    record = tmp_path / "vllm" / rc.KV_POOL_RECORD_NAME
    # the S2R start: 396 blocks, token line 26,928, concurrency 5.91x for 16,384
    old = plan_budget(model=_MODEL, engine="vllm", r=1.0, demand=5_302_681_600, tp=4, topology="tp")
    assert old.geometry["blocks_per_rank"] == 396
    record.write_text(json.dumps(_vllm_record(26_928, conc=5.91, blocks=396)), encoding="utf-8")
    out = rc.realized_pool_check(_step(old))
    assert out["ok"] is True, out
    assert (out["realized_tokens"], out["realized_blocks"], out["planned_tokens"], out["gap"]) == (107_440, 396, 107_440, 0.0)
    assert out["geometry"]["block_tokens"] == 272 and out["log_tokens"] == 26_928 and out["max_concurrency"] == 5.91
    assert "396 blocks" in out["basis"] and "null" in out["basis"]
    # a token line of another block count fails even inside 0.5 percent
    record.write_text(json.dumps(_vllm_record(26_928 + 272, conc=5.97)), encoding="utf-8")
    bad = rc.realized_pool_check(_step(old))
    assert bad["ok"] is False and bad["realized_blocks"] == 400 and "token line" in bad["reason"]
    # the concurrency line must agree with the block count
    record.write_text(json.dumps(_vllm_record(26_928, conc=5.80)), encoding="utf-8")
    assert "concurrency" in rc.realized_pool_check(_step(old))["reason"]
    # the concurrency line must be for the step's request cap
    record.write_text(json.dumps(_vllm_record(26_928, conc=5.91, req=4096)), encoding="utf-8")
    assert "16384" in rc.realized_pool_check(_step(old))["reason"]
    # without the concurrency line the block count is unknown: never a pass
    record.write_text(json.dumps(_vllm_record(26_928, conc=None)), encoding="utf-8")
    assert rc.realized_pool_check(_step(old))["ok"] is False
    # the blocks line, when printed, must agree too
    record.write_text(json.dumps(_vllm_record(26_928, conc=5.91, blocks=395)), encoding="utf-8")
    assert "blocks" in rc.realized_pool_check(_step(old))["reason"]
    # the registered r = 1.0 rung: 645 blocks, no blocks line (vLLM prints it under 512 only)
    new = _hybrid_single_plan()
    record.write_text(json.dumps(_vllm_record((645 // 4) * 272, conc=round(645 / 67, 2))), encoding="utf-8")
    out = rc.realized_pool_check(_step(new))
    assert out["ok"] is True and out["realized_tokens"] == 175_168 and out["realized_blocks"] == 645
    # a dense plan keeps the 0.5 percent token rule and reads no geometry
    dense = plan_budget(model="qwen3-14b", engine="vllm", r=1.0, demand=163_840 * 1000)
    record.write_text(json.dumps({"tokens": 1002, "bytes": 1}), encoding="utf-8")
    out = rc.realized_pool_check(_step(dense, tp=1, cap=32_768))
    assert out["ok"] is True and out.get("geometry") is None


def test_the_realized_pool_check_requires_the_exact_tokens_on_sglang_hybrid(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CAGE_LOG_ROOT", str(tmp_path))
    (tmp_path / "sglang").mkdir()
    record = tmp_path / "sglang" / rc.KV_POOL_RECORD_NAME
    plan = plan_budget(model=_MODEL, engine="sglang", r=1.0, demand=_D_TP4, tp=4, topology="tp")
    assert plan.budget_tokens_total == 175_168
    record.write_text(json.dumps({"tokens": 175_168, "bytes": 2_152_464_384,
                                  "extra": {"max_mamba_cache_size": 2286, "max_running_requests": 762}}), encoding="utf-8")
    out = rc.realized_pool_check(_step(plan, engine="sglang"))
    assert out["ok"] is True and out["realized_tokens"] == 175_168
    assert (out["max_mamba_cache_size"], out["max_running_requests"]) == (2286, 762)
    record.write_text(json.dumps({"tokens": 175_169, "bytes": 1}), encoding="utf-8")
    bad = rc.realized_pool_check(_step(plan, engine="sglang"))
    assert bad["ok"] is False and "exact" in bad["reason"]


def test_the_realized_pool_check_sums_the_pd_roles_at_the_role_geometry(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CAGE_LOG_ROOT", str(tmp_path))
    (tmp_path / "vllm").mkdir()
    record = tmp_path / "vllm" / rc.KV_POOL_PD_RECORD_NAME
    pd = plan_budget(model=_MODEL, engine="vllm", r=1.0, demand=_D_TP4, tp=1, topology="pd", pd_split=0.5, role_tp=2)
    role = _vllm_record((322 // 4) * 544, conc=round(322 / 37, 2), nbytes=2_152_464_384)
    record.write_text(json.dumps({"roles": {"prefill": role, "decode": role}}), encoding="utf-8")
    out = rc.realized_pool_check(_step(pd, topology="pd", tp=2))
    assert out["ok"] is True, out
    assert (out["realized_tokens"], out["realized_blocks"], out["planned_tokens"]) == (349_248, [322, 322], 349_248)
    short = _vllm_record((321 // 4) * 544, conc=round(321 / 37, 2), nbytes=1)
    record.write_text(json.dumps({"roles": {"prefill": role, "decode": short}}), encoding="utf-8")
    bad = rc.realized_pool_check(_step(pd, topology="pd", tp=2))
    assert bad["ok"] is False and "decode" in bad["reason"]


def test_the_pool_check_command_takes_the_model_and_the_request_cap(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("CAGE_LOG_ROOT", str(tmp_path))
    record = tmp_path / "vllm" / rc.KV_POOL_RECORD_NAME
    record.parent.mkdir(parents=True)
    out = tmp_path / "check.json"
    base = ["pool-check", "--engine", "vllm", "--planned-tokens", "175168", "--planned-bytes", "8626176000",
            "--tp", "4", "--out", str(out), "--model", _MODEL]
    record.write_text(json.dumps(_vllm_record((645 // 4) * 272, conc=round(645 / 67, 2))), encoding="utf-8")
    assert rc.main(base + ["--max-model-len", "16384"]) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert (doc["ok"], doc["realized_blocks"], doc["geometry"]["blocks_per_rank"]) == (True, 645, 645)
    assert "LARGEST_POOL_OK engine=vllm realized_tokens=175168" in capsys.readouterr().out
    # a hybrid model without the request cap cannot be checked: refused, never a pass
    assert rc.main(base) == 1
    assert "max-model-len" in capsys.readouterr().out
    # the dense call is unchanged
    record.write_text(json.dumps({"tokens": 1000, "bytes": 4000}), encoding="utf-8")
    dense = ["pool-check", "--engine", "vllm", "--planned-tokens", "1000", "--planned-bytes", "4000", "--out", str(out),
             "--model", "qwen3-14b"]
    assert rc.main(dense) == 0 and json.loads(out.read_text(encoding="utf-8"))["ok"] is True


# -- the registration report --------------------------------------------------


def test_the_registration_report_names_the_serving_degree_and_the_new_pools() -> None:
    manifest = REPO_ROOT / "data" / "manifests" / "musique_2000x3_seed42_wr50_tok-Qwen3-Next-80B-A3B-Instruct.json"
    cd = rc.preflight_registration_check(
        "cd-act1", floor_avg_seq_tokens="2694", query_manifests={"musique": str(manifest)},
        charter_datasets=("musique",), floor_concurrency=50, gpu_id="NVIDIA H200")
    assert cd["serving_tp"] == 4
    assert [p for p in cd["problems"] if "pool" in p] == []
    assert cd["largest_pools"]["vllm"]["planned_bytes"] == _D_TP4
    assert cd["largest_pools"]["sglang"]["env"]["CAGE_SGLANG_MAX_TOTAL_TOKENS"] == "175168"
    env = cd["pd_env"]["env"]
    assert cd["pd_env"]["planned_bytes"] == _D_TP4
    assert (env["CAGE_KV_BUDGET_BYTES_PREFILL"], env["CAGE_KV_BUDGET_BYTES_DECODE"]) == ("2152464384", "2152464384")
    s1 = rc.preflight_registration_check("s1", floor_avg_seq_tokens="4779", floor_concurrency=50, gpu_id="NVIDIA H100 80GB HBM3")
    assert s1["serving_tp"] == 1 and s1["problems"] == []


# -- review 2026-11-08: the estimator charge follows the start's prefix mode --


def test_a_prefix_off_vllm_start_is_judged_at_the_none_mode_charge(tmp_path: Path, monkeypatch) -> None:
    # HIGH 1: a prefix-OFF relaunch (gold-fresh, corpus-fresh, retr-fresh,
    # retr-comp, retr-trunc, every F2 cell) starts vLLM with prefix caching
    # off, which forces mamba cache mode "none": the concurrency line then
    # charges one state page per group, 645 / (61 + 3) = 10.08x, not 9.63x
    monkeypatch.setenv("CAGE_LOG_ROOT", str(tmp_path))
    (tmp_path / "vllm").mkdir()
    record = tmp_path / "vllm" / rc.KV_POOL_RECORD_NAME
    plan = _hybrid_single_plan()
    record.write_text(json.dumps(_vllm_record((645 // 4) * 272, conc=round(645 / 64, 2))), encoding="utf-8")
    off = dict(_step(plan), prefix_mode="OFF")
    out = rc.realized_pool_check(off)
    assert out["ok"] is True, out
    assert (out["realized_tokens"], out["realized_blocks"]) == (175_168, 645)
    assert (out["mamba_cache_mode"], out["blocks_needed"]) == ("none", 64)
    # the same line under a prefix-ON start is a mismatch (the align charge)
    on = rc.realized_pool_check(dict(_step(plan), prefix_mode="ON"))
    assert on["ok"] is False and "concurrency" in on["reason"] and on["mamba_cache_mode"] == "align"
    # and a step without the key is judged as prefix-ON (the validate start)
    assert rc.realized_pool_check(_step(plan))["ok"] is False
    # the pd twin: two roles of 322 blocks of 544, 322 / (31 + 3) = 9.47x per role
    record_pd = tmp_path / "vllm" / rc.KV_POOL_PD_RECORD_NAME
    pd = plan_budget(model=_MODEL, engine="vllm", r=1.0, demand=_D_TP4, tp=1, topology="pd", pd_split=0.5, role_tp=2)
    role = _vllm_record((322 // 4) * 544, conc=round(322 / 34, 2), nbytes=2_152_464_384)
    record_pd.write_text(json.dumps({"roles": {"prefill": role, "decode": role}}), encoding="utf-8")
    out = rc.realized_pool_check(dict(_step(pd, topology="pd", tp=2), prefix_mode="OFF"))
    assert out["ok"] is True, out
    assert out["realized_blocks"] == [322, 322] and out["prefill"]["mamba_cache_mode"] == "none"
    assert rc.realized_pool_check(_step(pd, topology="pd", tp=2))["ok"] is False


def test_pool_shortfalls_charge_the_state_by_the_prefix_mode() -> None:
    # 65 blocks of 272 at the 16,384 cap: short under prefix ON (needs 67),
    # enough under prefix OFF (needs 64)
    plan = plan_budget(model=_MODEL, engine="vllm", r=1.0, demand=65 * 272 * 49_152, tp=4, topology="tp")
    assert plan.geometry["blocks_per_rank"] == 65
    on = rc.pool_shortfalls([dict(_step(plan), prefix_mode="ON")], 16_384)
    assert len(on) == 1 and (on[0]["blocks_needed"], on[0]["mamba_cache_mode"]) == (67, "align")
    assert rc.pool_shortfalls([_step(plan)], 16_384)[0]["blocks_needed"] == 67
    assert rc.pool_shortfalls([dict(_step(plan), prefix_mode="OFF")], 16_384) == []


def test_mac_relaunch_records_carry_the_prefix_mode() -> None:
    grid = rc.get_session_grid("cd-act1")
    records = rc._mac_relaunch_records(grid, rc.enumerate_cells(grid), 50)
    # one arm per class, so one mode per class: the anchor arm gold-fresh
    # (2694), corpus-fresh (2386) and retr-fresh (506) start prefix-OFF; the
    # prefix-ON arms are corpus-reuse (2387) and retr-reuse (517)
    want = {2694: "OFF", 2387: "ON", 2386: "OFF", 517: "ON", 506: "OFF"}
    seen = {}
    for s in records:
        if s["budget_plan"]["kv_dtype"] != "bf16":
            continue
        seen.setdefault((s["engine"], s["topology"]), {}).setdefault(s["demand_class"]["seq_tokens"], set()).add(s["prefix_mode"])
    assert set(seen) == {("vllm", "single"), ("vllm", "pd"), ("sglang", "single")}
    for key, by_class in seen.items():
        assert by_class == {c: {m} for c, m in want.items()}, key
    assert rc.mac_pool_shortfalls(grid, rc.enumerate_cells(grid), 50) == []


def test_the_concurrency_line_is_compared_at_vllm_s_two_decimals(tmp_path: Path, monkeypatch) -> None:
    # LOW 2: vLLM prints "%.2f"; a pool one block off the plan at 134 blocks
    # prints 2.01x against the plan's 2.00x and a 0.01 tolerance let it pass
    monkeypatch.setenv("CAGE_LOG_ROOT", str(tmp_path))
    (tmp_path / "vllm").mkdir()
    record = tmp_path / "vllm" / rc.KV_POOL_RECORD_NAME
    plan = plan_budget(model=_MODEL, engine="vllm", r=1.0, demand=134 * 272 * 49_152, tp=4, topology="tp")
    assert plan.geometry["blocks_per_rank"] == 134
    record.write_text(json.dumps(_vllm_record((134 // 4) * 272, conc=2.01)), encoding="utf-8")
    bad = rc.realized_pool_check(_step(plan))
    assert bad["ok"] is False and "concurrency" in bad["reason"]
    record.write_text(json.dumps(_vllm_record((134 // 4) * 272, conc=2.00)), encoding="utf-8")
    assert rc.realized_pool_check(_step(plan))["ok"] is True
