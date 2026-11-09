"""Pins for the hybrid pool identity (DEC-8, 2026-11-08): Qwen3-Next on vLLM
and SGLang at tensor parallel 1, 2 and 4.

What is pinned and why:

- The attention cost per token per rank REPLICATES when the KV heads do not
  divide the ranks (2 heads on 4 ranks: 12,288 B per rank, 49,152 B pod
  wide); the old planner charged 24,576 pod wide at every degree, half the
  real figure at TP 4 (S2R pod 2026-11-07, MyDocs/RunPod/S2_KV_POOL_AUDIT_2026-11-07.md).
- vLLM's hybrid block geometry from the pinned formula (vllm 0.19.1
  models/config.py:266-285, kv_cache_utils.py:1043-1077, 1140-1151): block
  272 tokens at TP 4, 544 at TP 2 and TP 1; one shared block of 12 pages.
- The S2R start reproduced from the registered budget: 396 blocks, the
  token line 26,928 and the concurrency line 5.91x [V S2R vllm log].
- The state cost per running request on each engine (vLLM 3 shared blocks
  steady, 6 worst; SGLang one float32 SSM slot), engine tagged, never dtype
  scaled.
- The identity: at one budget both engines get the same zero-load
  attention-token room T; vLLM's bytes are an exact block multiple.
- Dense models: every figure unchanged at every degree.
"""

from __future__ import annotations

import json

import pytest

from src.orchestration.cache_budget import (
    MODEL_KV,
    CacheBudgetError,
    attention_bytes_per_token_per_rank,
    demand_bytes,
    fixed_state_bytes_per_seq,
    plan_budget,
    pod_attention_bytes_per_token,
    state_bytes_per_request,
    vllm_blocks_for_request,
    vllm_hybrid_geometry,
)

_H = "qwen3-next-80b"
_MIB = 1024 ** 2

# vLLM at TP 4: block 272 tokens, attention page 278,528 B, 12 pages per block
_SHARED_TP4 = 12 * 278_528  # 3,342,336
_SHARED_TP2 = 12 * 557_056  # 6,684,672
_SHARED_TP1 = 12 * 1_114_112  # 13,369,344

# demand at the S2B floor shape (FLOOR_CONCURRENCY 50, anchor 2,694 tokens)
_D_TP4 = 50 * 2694 * 49_152 + 50 * 40_108_032  # 8,626,176,000


# ---------------------------------------------------------------------------
# Attention cost per token: replication when the heads do not divide the ranks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tp, per_rank, pod", [(1, 24_576, 24_576), (2, 12_288, 24_576), (4, 12_288, 49_152)])
def test_hybrid_attention_cost_replicates_at_tp4(tp, per_rank, pod):
    # 2 (K,V) x 12 attention layers x max(1, 2 heads // tp) x 256 x 2 B
    assert attention_bytes_per_token_per_rank(_H, tp) == per_rank
    assert pod_attention_bytes_per_token(_H, tp) == pod
    assert MODEL_KV[_H].kv_bytes_per_token == 24_576 == pod_attention_bytes_per_token(_H, 1)


def test_hybrid_attention_cost_fp8_halves_the_per_rank_figure():
    assert attention_bytes_per_token_per_rank(_H, 4, kv_dtype="fp8") == 6_144
    assert pod_attention_bytes_per_token(_H, 4, kv_dtype="fp8") == 24_576


def test_dense_models_shard_without_replication_and_keep_their_figures():
    # 8 KV heads on 4 ranks: two per rank, the pod-wide cost is the table's
    assert attention_bytes_per_token_per_rank("llama-3.3-70b", 4) == 81_920
    assert pod_attention_bytes_per_token("llama-3.3-70b", 4) == 327_680
    assert attention_bytes_per_token_per_rank("qwen3-14b", 4) == 40_960
    assert pod_attention_bytes_per_token("qwen3-14b", 4) == 163_840
    assert pod_attention_bytes_per_token("qwen3-14b", 1, kv_dtype="fp8") == 81_920
    # MLA replicates the latent: per rank == the table figure, pod figure unchanged
    assert attention_bytes_per_token_per_rank("deepseek-v3", 8) == 70_272
    assert pod_attention_bytes_per_token("deepseek-v3", 8) == 70_272


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_attention_cost_refuses_a_bad_tp(bad):
    with pytest.raises(CacheBudgetError, match="tp"):
        attention_bytes_per_token_per_rank(_H, bad)


# ---------------------------------------------------------------------------
# vLLM hybrid geometry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tp, block, page, shared", [
    (4, 272, 278_528, _SHARED_TP4),
    (2, 544, 557_056, _SHARED_TP2),
    (1, 544, 1_114_112, _SHARED_TP1),
])
def test_vllm_geometry_at_the_three_degrees(tp, block, page, shared):
    g = vllm_hybrid_geometry(_H, tp)
    assert (g.block_tokens, g.page_bytes, g.group_size, g.n_groups, g.shared_block_bytes) == (
        block, page, 12, 4, shared)
    # the GDN page per layer per rank: conv 3 x 8192 / tp x 2 B + SSM 32 / tp x 128 x 128 x 2 B
    assert g.state_page_bytes == (3 * 8192 // tp) * 2 + (32 // tp) * 128 * 128 * 2
    assert g.state_page_bytes <= g.page_bytes


def test_vllm_geometry_fp8_enters_the_block_size_not_only_the_token_count():
    # a 512 B attention page per token per layer per rank doubles the block
    g = vllm_hybrid_geometry(_H, 4, kv_dtype="fp8")
    assert (g.block_tokens, g.page_bytes) == (544, 278_528)


def test_vllm_geometry_refuses_a_dense_model_and_a_bad_tp():
    with pytest.raises(CacheBudgetError, match="hybrid"):
        vllm_hybrid_geometry("qwen3-14b", 4)
    with pytest.raises(CacheBudgetError, match="tp"):
        vllm_hybrid_geometry(_H, 3)  # 8192 conv elements and 32 heads do not divide 3


def test_the_s2r_start_is_reproduced_from_the_registered_budget():
    # [V experiments/S2R/2026-11-07_1536 vllm log]: --kv-cache-memory-bytes
    # 1,325,670,400 per rank gave "GPU KV cache size: 26,928 tokens",
    # "Maximum concurrency for 16,384 tokens per request: 5.91x" and
    # "(396 blocks available)"
    g = vllm_hybrid_geometry(_H, 4)
    n = 1_325_670_400 // g.shared_block_bytes
    assert n == 396
    assert (n // g.n_groups) * g.block_tokens == 26_928
    per_request = vllm_blocks_for_request(_H, 4, 16_384)
    assert per_request == 61 + 6 == 67
    assert abs(n / per_request - 5.9104) < 0.001
    assert vllm_blocks_for_request(_H, 2, 16_384) == 31 + 6
    assert vllm_blocks_for_request(_H, 4, 4_096) == 16 + 6


# ---------------------------------------------------------------------------
# State per running request, per engine
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tp", [1, 2, 4])
def test_state_per_request_is_engine_tagged_and_degree_invariant_pod_wide(tp):
    assert state_bytes_per_request(_H, "vllm", tp, "steady") == 40_108_032  # 38.25 MiB
    assert state_bytes_per_request(_H, "vllm", tp, "worst") == 80_216_064
    assert state_bytes_per_request(_H, "sglang", tp, "slot") == 77_266_944  # 73.69 MiB
    assert fixed_state_bytes_per_seq(_H, tp) == 40_108_032


def test_state_per_request_per_rank_at_tp4_and_dense_zero():
    assert state_bytes_per_request(_H, "vllm", 4, "steady") // 4 == 3 * _SHARED_TP4 == 10_027_008
    assert state_bytes_per_request(_H, "vllm", 4, "worst") // 4 == 20_054_016
    # SGLang: 36 layers x (conv 12,288 B + SSM 524,288 B float32) per rank
    assert state_bytes_per_request(_H, "sglang", 4, "slot") // 4 == 36 * (12_288 + 524_288) == 19_316_736
    assert state_bytes_per_request("qwen3-14b", "vllm", 4, "steady") == 0
    assert fixed_state_bytes_per_seq("llama-3.3-70b", 4) == 0


@pytest.mark.parametrize("engine, mode", [("vllm", "slot"), ("sglang", "steady"), ("tgi", "steady"), ("vllm", "mean")])
def test_state_per_request_refuses_an_unknown_engine_or_mode(engine, mode):
    with pytest.raises(CacheBudgetError):
        state_bytes_per_request(_H, engine, 4, mode)


# ---------------------------------------------------------------------------
# Demand D at the serving degree
# ---------------------------------------------------------------------------

def test_hybrid_demand_at_tp4_uses_the_pod_cost_and_the_vllm_steady_state():
    assert demand_bytes(_H, concurrency=50, avg_seq_tokens=2694, tp=4) == _D_TP4 == 8_626_176_000
    assert demand_bytes(_H, concurrency=50, avg_seq_tokens=2694, tp=1) == 5_315_788_800
    assert demand_bytes(_H, concurrency=50, avg_seq_tokens=2694) == demand_bytes(_H, concurrency=50, avg_seq_tokens=2694, tp=1)
    # fp8 scales the attention part only
    assert demand_bytes(_H, concurrency=50, avg_seq_tokens=2694, tp=4, kv_dtype="fp8") == 50 * 2694 * 24_576 + 50 * 40_108_032


def test_dense_demand_is_the_same_at_every_degree():
    for tp in (1, 2, 4, 8):
        assert demand_bytes("qwen3-14b", concurrency=9, avg_seq_tokens=32768, tp=tp) == 48_318_382_080
        assert demand_bytes("llama-3.3-70b", concurrency=50, avg_seq_tokens=4779, tp=tp) == 50 * 4779 * 327_680


# ---------------------------------------------------------------------------
# The plan: equal zero-load attention-token room T on both engines
# ---------------------------------------------------------------------------

_RUNGS = {
    # r: (blocks per rank, T, vLLM bytes per rank)
    1.0: (645, 175_168, 2_155_806_720),
    0.5: (322, 87_312, 1_076_232_192),
    0.25: (161, 43_520, 538_116_096),
}


@pytest.mark.parametrize("r", sorted(_RUNGS, reverse=True))
def test_the_three_rungs_give_equal_room_and_exact_block_bytes(r):
    n, tokens, rank_bytes = _RUNGS[r]
    vl = plan_budget(model=_H, engine="vllm", r=r, demand=_D_TP4, tp=4, topology="tp")
    sg = plan_budget(model=_H, engine="sglang", r=r, demand=_D_TP4, tp=4, topology="tp")
    assert vl.budget_bytes_total == sg.budget_bytes_total == int(r * _D_TP4)
    assert vl.budget_tokens_total == sg.budget_tokens_total == tokens == (n - 1) * 272
    assert vl.per_rank_bytes == rank_bytes == n * _SHARED_TP4
    assert vl.engine_args[0].args == ("--kv-cache-memory-bytes", str(rank_bytes))
    assert next(a for a in vl.engine_args if a.kind == "fallback").args == ("--num-gpu-blocks-override", str(n))
    assert sg.engine_args[0].args == ("--max-total-tokens", str(tokens))
    assert vl.gate_j["expected_bytes_total"] == rank_bytes * 4
    assert vl.gate_j["expected_tokens_total"] == sg.gate_j["expected_tokens_total"] == tokens
    for plan in (vl, sg):
        assert plan.geometry == {
            "tp": 4, "block_tokens": 272, "page_bytes": 278_528, "group_size": 12, "n_groups": 4,
            "shared_block_bytes": _SHARED_TP4, "blocks_per_rank": n, "null_blocks": 1,
        }
        assert plan.state_per_request_bytes == {
            "vllm_steady": 40_108_032, "vllm_worst": 80_216_064, "sglang_slot": 77_266_944,
        }
    # every rung holds one 16,384-token request by vLLM's own start rule
    assert n >= vllm_blocks_for_request(_H, 4, 16_384)


def test_the_old_registration_budget_plans_the_s2r_pool():
    # the S2R registration (5,302,681,600 B total, 1,325,670,400 per rank) at
    # the corrected rule: 396 blocks, T 107,440, 1,323,565,056 B per rank
    vl = plan_budget(model=_H, engine="vllm", r=1.0, demand=5_302_681_600, tp=4, topology="tp")
    assert (vl.geometry["blocks_per_rank"], vl.budget_tokens_total, vl.per_rank_bytes) == (396, 107_440, 1_323_565_056)
    sg = plan_budget(model=_H, engine="sglang", r=1.0, demand=5_302_681_600, tp=4, topology="tp")
    assert sg.engine_args[0].args == ("--max-total-tokens", "107440")


def test_the_helper_class_pool_follows_the_same_rule():
    # class 517 tokens at r = 1.0 (the max_model_len 4,096 relaunch):
    # D_class = (D - F) x 517 // 2694 + F
    f = 50 * 40_108_032
    d_class = (_D_TP4 - f) * 517 // 2694 + f
    assert d_class == 3_275_980_800
    vl = plan_budget(model=_H, engine="vllm", r=1.0, demand=d_class, tp=4, topology="tp")
    assert (vl.geometry["blocks_per_rank"], vl.budget_tokens_total, vl.per_rank_bytes) == (245, 66_368, 818_872_320)


def test_a_single_rank_hybrid_plan_uses_the_tp1_geometry():
    vl = plan_budget(model=_H, engine="vllm", r=1.0, demand=5_315_788_800)
    # 5,315,788,800 // 24,576 = 216,300 tokens -> 397 blocks of 544
    assert (vl.geometry["block_tokens"], vl.geometry["blocks_per_rank"]) == (544, 397)
    assert vl.budget_tokens_total == 396 * 544
    assert vl.per_rank_bytes == 397 * _SHARED_TP1


@pytest.mark.parametrize("delta, n", [(0, 645), (-1, 644), (49_152 * 272 - 1, 645), (49_152 * 272, 646)])
def test_block_rounding_at_a_block_boundary(delta, n):
    budget = 645 * 272 * 49_152 + delta
    vl = plan_budget(model=_H, engine="vllm", r=1.0, demand=budget, tp=4, topology="tp")
    assert vl.geometry["blocks_per_rank"] == n
    assert vl.per_rank_bytes == n * _SHARED_TP4


def test_a_budget_under_two_blocks_is_refused():
    with pytest.raises(CacheBudgetError, match="two blocks"):
        plan_budget(model=_H, engine="vllm", r=1.0, demand=272 * 49_152 * 2 - 1, tp=4, topology="tp")
    # exactly two blocks: one usable, T = 272
    assert plan_budget(model=_H, engine="vllm", r=1.0, demand=272 * 49_152 * 2, tp=4,
                       topology="tp").budget_tokens_total == 272


def test_lmdeploy_has_no_hybrid_geometry_and_is_refused():
    with pytest.raises(CacheBudgetError, match="hybrid"):
        plan_budget(model=_H, engine="lmdeploy", r=1.0, demand=_D_TP4, free_bytes_after_weights=10 ** 12)


# ---------------------------------------------------------------------------
# P/D roles at their own degree
# ---------------------------------------------------------------------------

def test_pd_roles_at_tp2_use_the_tp2_geometry():
    plan = plan_budget(model=_H, engine="vllm", r=1.0, demand=_D_TP4, topology="pd", pd_split=0.5, role_tp=2)
    assert plan.pools_bytes == (4_313_088_000, 4_313_088_000)
    # 4,313,088,000 // 24,576 = 175,500 tokens -> 322 blocks of 544 per role
    assert plan.pools_rank_bytes == (322 * _SHARED_TP2, 322 * _SHARED_TP2) == (2_152_464_384, 2_152_464_384)
    assert plan.pools_tokens == (174_624, 174_624)
    assert plan.budget_tokens_total == 349_248
    assert plan.gate_j["expected_tokens_total"] == 349_248
    assert plan.gate_j["expected_bytes_total"] == 2 * 2 * 2_152_464_384
    assert plan.geometry["tp"] == 2 and plan.geometry["block_tokens"] == 544
    assert plan.geometry["blocks_per_rank"] == [322, 322]
    # the role knobs stay the pool totals (the launcher env carries the per-rank slice)
    assert [a.args for a in plan.engine_args] == [
        ("--kv-cache-memory-bytes", "4313088000"), ("--kv-cache-memory-bytes", "4313088000")]


def test_a_hybrid_pd_plan_without_the_role_degree_is_refused():
    with pytest.raises(CacheBudgetError, match="role_tp"):
        plan_budget(model=_H, engine="vllm", r=1.0, demand=_D_TP4, topology="pd", pd_split=0.5)


def test_dense_pd_plans_carry_the_role_slices_and_are_otherwise_unchanged():
    d = demand_bytes("qwen3-14b", concurrency=9, avg_seq_tokens=32768)
    before = plan_budget(model="qwen3-14b", engine="vllm", r=0.5, demand=d, topology="pd", pd_split=0.5)
    after = plan_budget(model="qwen3-14b", engine="vllm", r=0.5, demand=d, topology="pd", pd_split=0.5, role_tp=2)
    assert before.pools_rank_bytes is None and before.geometry is None and before.state_per_request_bytes is None
    assert before.pools_tokens == tuple(p // 163_840 for p in before.pools_bytes)
    assert after.pools_rank_bytes == tuple(p // 2 for p in after.pools_bytes)
    assert after.pools_tokens == before.pools_tokens
    assert after.budget_tokens_total == before.budget_tokens_total == 147_456
    assert after.engine_args == before.engine_args
    mla = plan_budget(model="deepseek-v3", engine="vllm", r=1.0, demand=8 * 8192 * 70_272,
                      topology="pd", pd_split=0.5, role_tp=4)
    assert mla.pools_rank_bytes == mla.pools_bytes  # replicated latent: no slice


def test_cli_plans_the_hybrid_at_the_serving_degree(capsys) -> None:
    # DEC-8: --tp sizes the demand and, above 1, plans the sharded single
    # instance (topology tp), the rule the driver applies to single relaunches
    from src.orchestration.cache_budget import _main
    assert _main(["--model", "qwen3-next-80b", "--engine", "vllm", "--r", "1.0", "--concurrency", "50",
                  "--avg-seq-tokens", "2694", "--tp", "4"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert (doc["demand_bytes"], doc["budget_tokens_total"], doc["tp"], doc["topology"]) == (8_626_176_000, 175_168, 4, "tp")
    assert doc["geometry"]["blocks_per_rank"] == 645
    # the default degree keeps the single plan
    assert _main(["--model", "qwen3-14b", "--engine", "vllm", "--r", "1.0", "--concurrency", "50", "--avg-seq-tokens", "4779"]) == 0
    assert json.loads(capsys.readouterr().out)["topology"] == "single"


def test_the_request_block_count_follows_the_mamba_cache_mode() -> None:
    # review 2026-11-08 HIGH 1: vLLM's start estimator charges two state pages
    # per group in mode "align" (prefix caching on) and one in mode "none"
    # (prefix caching off forces it) [V kv_cache_interface.py MambaSpec
    # .max_memory_usage_bytes; models/config.py, the prefix-off branch]
    from src.orchestration.cache_budget import (
        VLLM_MAMBA_CACHE_MODE_STATE_BLOCKS, vllm_blocks_for_request, vllm_mamba_cache_mode,
    )
    assert VLLM_MAMBA_CACHE_MODE_STATE_BLOCKS == {"align": 2, "none": 1}
    assert vllm_blocks_for_request("qwen3-next-80b", 4, 16_384) == 67
    assert vllm_blocks_for_request("qwen3-next-80b", 4, 16_384, mamba_cache_mode="align") == 67
    assert vllm_blocks_for_request("qwen3-next-80b", 4, 16_384, mamba_cache_mode="none") == 64
    assert vllm_blocks_for_request("qwen3-next-80b", 2, 16_384, mamba_cache_mode="none") == 34
    assert (vllm_mamba_cache_mode(True), vllm_mamba_cache_mode(False)) == ("align", "none")
    with pytest.raises(CacheBudgetError, match="mamba_cache_mode"):
        vllm_blocks_for_request("qwen3-next-80b", 4, 16_384, mamba_cache_mode="all")
