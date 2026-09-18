"""Pins for CacheBudgetPlanner (gap G-P1, 2026-08-26).

What is pinned and why:

- The per-model KV/token numbers ARE the charter's D4 arithmetic (audit-
  corrected); a drifted constant here silently mis-budgets every pressure
  cell, so the exact integers and formulas are asserted.
- The worked r=0.5 anchor cell (9 x 32k on qwen3-14b) reproduces the
  charter's own "~45 GB KV ≈ 9 concurrent at r=1.0" arithmetic and lands on
  EXACT integers (147,456 tokens) — any change to the arithmetic breaks it.
- Fail-closed refusals: HF (P2), P/D without an explicit split (§6.5),
  LMDeploy without its free-memory input, unknown model/engine/dtype.
- §6.5 exact-sum: P/D pools sum to the total budget BYTE-EXACTLY, including
  non-dividing splits.
- TP semantics: GQA shards (per-rank = total // tp); MLA REPLICATES
  (per-rank == total — registered contrast #20).
- The planner never claims live knob semantics: verify_live is non-empty for
  every plannable engine (gate (j) is the closer).
"""

from __future__ import annotations

import json

import pytest

from src.orchestration.cache_budget import (
    KV_DTYPE_FACTOR,
    MODEL_KV,
    CacheBudgetError,
    _main,
    demand_bytes,
    plan_budget,
)

# The worked anchor cell used throughout: charter D4.1's own numbers.
_D_14B = demand_bytes("qwen3-14b", concurrency=9, avg_seq_tokens=32768)


# ---------------------------------------------------------------------------
# Geometry pins (charter D4 mechanism coverage)
# ---------------------------------------------------------------------------

def test_kv_per_token_constants_are_the_charter_arithmetic() -> None:
    assert MODEL_KV["qwen3-14b"].kv_bytes_per_token == 2 * 40 * 8 * 128 * 2 == 163_840
    assert MODEL_KV["llama-3.3-70b"].kv_bytes_per_token == 2 * 80 * 8 * 128 * 2 == 327_680
    assert MODEL_KV["deepseek-v3"].kv_bytes_per_token == 61 * (512 + 64) * 2 == 70_272
    assert MODEL_KV["qwen3-next-80b"].kv_bytes_per_token == 24_576
    assert MODEL_KV["qwen3-next-80b"].fixed_state_bytes_per_seq == 38 * 1024 * 1024
    assert MODEL_KV["deepseek-v3"].mla_tp_replicated is True
    assert MODEL_KV["qwen3-14b"].mla_tp_replicated is False


def test_s0_shakedown_model_kv_arithmetic_is_registered() -> None:
    # S0 serves a small model on the 1x L40S pod (S0_CHECKLIST model guidance),
    # off the D4 roster. FP8 weights, BF16 KV (kv_cache_dtype=auto):
    # 2 (K,V) x 36 layers x 8 KV heads x 128 head_dim x 2 B = 144 KiB/tok.
    s0 = MODEL_KV["qwen3-8b-fp8"]
    assert s0.kv_bytes_per_token == 2 * 36 * 8 * 128 * 2 == 147_456
    assert s0.fixed_state_bytes_per_seq == 0
    assert s0.mla_tp_replicated is False


def test_demand_reproduces_the_charter_45gb_anchor_number() -> None:
    # 9 x 32768 x 160 KiB = 48,318,382,080 B ≈ 45 GiB — the D4.1 re-rung basis.
    assert _D_14B == 9 * 32768 * 163_840 == 48_318_382_080


def test_demand_fp8_halves_kv_exactly() -> None:
    full = demand_bytes("qwen3-14b", concurrency=4, avg_seq_tokens=1024)
    half = demand_bytes("qwen3-14b", concurrency=4, avg_seq_tokens=1024, kv_dtype="fp8")
    assert half * 2 == full
    assert KV_DTYPE_FACTOR == {"bf16": 1.0, "fp8": 0.5}


def test_demand_hybrid_adds_fixed_state_slots_undtyped() -> None:
    kv_part = 9 * 32768 * 24_576
    d = demand_bytes("qwen3-next-80b", concurrency=9, avg_seq_tokens=32768)
    assert d == kv_part + 9 * 38 * 1024 * 1024
    # fp8 scales ONLY the KV part; the recurrent slots are model state, not KV
    d8 = demand_bytes("qwen3-next-80b", concurrency=9, avg_seq_tokens=32768, kv_dtype="fp8")
    assert d8 == kv_part // 2 + 9 * 38 * 1024 * 1024


# ---------------------------------------------------------------------------
# The worked r=0.5 anchor plan (exact integers)
# ---------------------------------------------------------------------------

def test_vllm_anchor_plan_r05_exact_bytes_and_fallback_blocks() -> None:
    plan = plan_budget(model="qwen3-14b", engine="vllm", r=0.5, demand=_D_14B)
    assert plan.budget_bytes_total == 24_159_191_040
    assert plan.budget_tokens_total == 147_456  # divides exactly
    primary = plan.engine_args[0]
    assert primary.kind == "primary"
    assert primary.args == ("--kv-cache-memory-bytes", "24159191040")
    fallback = plan.engine_args[1]
    assert fallback.kind == "fallback"
    assert fallback.args == ("--num-gpu-blocks-override", str(147_456 // 16))
    assert plan.verify_live, "vllm knob semantics are live-only facts; gate (j) closes"
    assert plan.gate_j["expected_bytes_total"] == 24_159_191_040


def test_sglang_anchor_plan_r05_token_knob() -> None:
    plan = plan_budget(model="qwen3-14b", engine="sglang", r=0.5, demand=_D_14B)
    assert plan.engine_args[0].args == ("--max-total-tokens", "147456")
    assert plan.verify_live


# ---------------------------------------------------------------------------
# §6.5 P/D: explicit split, byte-exact sum
# ---------------------------------------------------------------------------

def test_pd_requires_explicit_split() -> None:
    with pytest.raises(CacheBudgetError, match="EXPLICIT pd_split"):
        plan_budget(model="qwen3-14b", engine="vllm", r=0.5, demand=_D_14B,
                    topology="pd")


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.2, 1.5])
def test_pd_split_must_be_strictly_interior(bad: float) -> None:
    with pytest.raises(CacheBudgetError, match="strictly inside"):
        plan_budget(model="qwen3-14b", engine="vllm", r=0.5, demand=_D_14B,
                    topology="pd", pd_split=bad)


@pytest.mark.parametrize("split", [0.5, 1 / 3, 0.7])
def test_pd_pools_sum_byte_exactly(split: float) -> None:
    plan = plan_budget(model="qwen3-14b", engine="vllm", r=0.5, demand=_D_14B,
                       topology="pd", pd_split=split)
    assert plan.pools_bytes is not None
    prefill, decode = plan.pools_bytes
    assert prefill + decode == plan.budget_bytes_total, "§6.5: pools SUM to B exactly"
    roles = [a.role for a in plan.engine_args]
    assert roles == ["prefill", "decode"]
    assert any("pool-sum" in v for v in plan.verify_live)


def test_pd_sglang_emits_two_token_capped_instances() -> None:
    plan = plan_budget(model="qwen3-14b", engine="sglang", r=0.5, demand=_D_14B,
                       topology="pd", pd_split=0.5)
    pre, dec = plan.engine_args
    assert pre.role == "prefill" and dec.role == "decode"
    assert pre.args[0] == dec.args[0] == "--max-total-tokens"


# ---------------------------------------------------------------------------
# TP semantics: GQA shards, MLA replicates (#20)
# ---------------------------------------------------------------------------

def test_tp_gqa_shards_per_rank() -> None:
    plan = plan_budget(model="qwen3-14b", engine="vllm", r=1.0, demand=_D_14B,
                       tp=2, topology="tp")
    assert plan.per_rank_bytes == plan.budget_bytes_total // 2
    assert "shard" in plan.per_rank_note


def test_tp_mla_replicates_per_rank_equals_total() -> None:
    d = demand_bytes("deepseek-v3", concurrency=8, avg_seq_tokens=8192)
    plan = plan_budget(model="deepseek-v3", engine="sglang", r=1.0, demand=d,
                       tp=8, topology="tp")
    assert plan.per_rank_bytes == plan.budget_bytes_total
    assert "REPLICATES" in plan.per_rank_note  # contrast #20 in the plan record


# ---------------------------------------------------------------------------
# Fail-closed refusals
# ---------------------------------------------------------------------------

def test_hf_is_refused_as_the_oracle() -> None:
    with pytest.raises(CacheBudgetError, match="oracle"):
        plan_budget(model="qwen3-14b", engine="hf", r=0.5, demand=_D_14B)


def test_lmdeploy_requires_free_memory_input() -> None:
    with pytest.raises(CacheBudgetError, match="free_bytes_after_weights"):
        plan_budget(model="qwen3-14b", engine="lmdeploy", r=0.5, demand=_D_14B)


def test_lmdeploy_fraction_derivation_and_overflow_refusal() -> None:
    plan = plan_budget(model="qwen3-14b", engine="lmdeploy", r=0.5, demand=_D_14B,
                       free_bytes_after_weights=2 * 24_159_191_040)
    assert plan.engine_args[0].args == ("cache_max_entry_count", "0.500000")
    with pytest.raises(CacheBudgetError, match="exceeds post-weight free memory"):
        plan_budget(model="qwen3-14b", engine="lmdeploy", r=0.5, demand=_D_14B,
                    free_bytes_after_weights=1024)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (dict(model="nope-7b", engine="vllm", r=0.5, demand=1), "unknown model"),
        (dict(model="qwen3-14b", engine="tgi", r=0.5, demand=1), "unknown engine"),
        (dict(model="qwen3-14b", engine="vllm", r=0.0, demand=1), "finite and > 0"),
        (dict(model="qwen3-14b", engine="vllm", r=float("inf"), demand=1), "finite and > 0"),
        (dict(model="qwen3-14b", engine="vllm", r=0.5, demand=0), "demand bytes"),
        (dict(model="qwen3-14b", engine="vllm", r=0.5, demand=1, tp=2), "requires tp=1"),
        (dict(model="qwen3-14b", engine="vllm", r=0.5, demand=1, tp=1,
              topology="tp"), "requires tp >= 2"),
        (dict(model="qwen3-14b", engine="vllm", r=0.5, demand=1,
              topology="ring"), "unknown topology"),
    ],
)
def test_fail_closed_matrix(kwargs: dict, match: str) -> None:
    with pytest.raises(CacheBudgetError, match=match):
        plan_budget(**kwargs)


def test_unknown_dtype_refused() -> None:
    with pytest.raises(CacheBudgetError, match="unknown kv_dtype"):
        demand_bytes("qwen3-14b", concurrency=1, avg_seq_tokens=1, kv_dtype="int4")


# ---------------------------------------------------------------------------
# Hybrid honesty note + CLI
# ---------------------------------------------------------------------------

def test_hybrid_plans_carry_the_concurrency_note() -> None:
    d = demand_bytes("qwen3-next-80b", concurrency=4, avg_seq_tokens=8192)
    plan = plan_budget(model="qwen3-next-80b", engine="sglang", r=0.5, demand=d)
    assert "concurrency" in plan.engine_args[0].note


def test_cli_smoke_emits_parseable_plan(capsys: pytest.CaptureFixture[str]) -> None:
    rc = _main([
        "--model", "qwen3-14b", "--engine", "vllm", "--r", "0.5",
        "--concurrency", "9", "--avg-seq-tokens", "32768",
    ])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["budget_bytes_total"] == 24_159_191_040
    assert payload["gate_j"]["expected_bytes_total"] == 24_159_191_040


def test_cli_refusal_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    rc = _main([
        "--model", "qwen3-14b", "--engine", "vllm", "--r", "0.5",
        "--concurrency", "9", "--avg-seq-tokens", "32768",
        "--topology", "pd",  # no --pd-split: must refuse
    ])
    assert rc == 2
    assert "REFUSED" in capsys.readouterr().err
