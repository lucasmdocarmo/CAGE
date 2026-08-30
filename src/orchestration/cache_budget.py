"""CacheBudgetPlanner — bytes-denominated budget → per-engine launch knobs.

Closes gap G-P1 (2026-08-26; the audit's open P1 item "CacheBudgetPlanner
bytes knob"). Charter bindings:

- P2: "Pressure = bytes of cache budget, the SAME physical quantity on every
  engine." The D6 grid's budget levels are ratios r = B/D, so running the
  cell "r=0.5" requires setting each engine's sequence-state pool to a
  specific byte count — which the engines accept only in their own dialects
  (vLLM: ``--kv-cache-memory-bytes`` / block override; SGLang: token-capped
  ``--max-total-tokens``; LMDeploy: post-weight fraction).
- §6.1: D = predicted steady-state cache demand at target concurrency from
  OUR per-model KV/token arithmetic (the P6 floor table's inputs).
- §6.5: budget B = TOTAL bytes of sequence-state cache, pools SUMMED; under
  P/D disaggregation the prefill/decode split is an EXPLICIT recorded input,
  never a silent default.
- P5: BF16 KV first; quantized KV is a separate factor level.

Division of labor: this module PLANS (emits knobs + the expected realized
bytes); preflight **gate (j)** VERIFIES (re-reads the engine startup logs and
fails on >±CAGE_ISO_BYTES_TOL disagreement). Knob semantics that only a live
engine can confirm are carried on the plan as ``verify_live`` entries — the
planner never claims a knob behaves; it claims what was requested and what
gate (j) must observe.

Fail-closed doctrine: unknown model/engine/topology, HF (the oracle is
excluded from pressure sweeps, P2), a P/D plan without an explicit split, or
an LMDeploy plan without its free-memory input all raise
:class:`CacheBudgetError` — no silent defaults anywhere in the budget path.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field

__all__ = [
    "CacheBudgetError",
    "ModelKV",
    "MODEL_KV",
    "KV_DTYPE_FACTOR",
    "EngineArgs",
    "BudgetPlan",
    "demand_bytes",
    "plan_budget",
]


class CacheBudgetError(ValueError):
    """A budget request that cannot be planned honestly (fail closed)."""


@dataclass(frozen=True)
class ModelKV:
    """Per-model KV arithmetic (charter D4 'Mechanism coverage', audit-corrected)."""

    kv_bytes_per_token: int
    formula: str
    fixed_state_bytes_per_seq: int = 0  # hybrid recurrent-state slot (Qwen3-Next)
    mla_tp_replicated: bool = False  # MLA latent replicates across TP ranks (#20)


#: BF16-first (P5); fp8 KV is a separate factor level, never a default.
KV_DTYPE_FACTOR: dict[str, float] = {"bf16": 1.0, "fp8": 0.5}

MODEL_KV: dict[str, ModelKV] = {
    # 2 (K,V) x 40 layers x 8 KV heads x 128 head_dim x 2 B (BF16) = 160 KiB/tok
    "qwen3-14b": ModelKV(163_840, "2*40*8*128*2"),
    # 2 x 80 x 8 x 128 x 2 = 320 KiB/tok
    "llama-3.3-70b": ModelKV(327_680, "2*80*8*128*2"),
    # MLA: 61 layers x (512 latent + 64 rope) x 2 B — ONE latent, no K/V pair,
    # and the latent REPLICATES across TP ranks (registered contrast #20).
    "deepseek-v3": ModelKV(70_272, "61*(512+64)*2", mla_tp_replicated=True),
    # Linear hybrid: 12 of 48 layers full-attention -> ~24 KiB/tok KV, PLUS a
    # fixed ~38 MiB recurrent-state slot per sequence (§5.1.4). Pressure
    # arrives via concurrency (§6.5); the KV-bytes floor is near-vacuous.
    "qwen3-next-80b": ModelKV(
        24_576, "2*12*4*128*2 (12/48 full-attn layers)",
        fixed_state_bytes_per_seq=38 * 1024 * 1024,
    ),
}

_ENGINES = ("vllm", "sglang", "lmdeploy", "hf")
_TOPOLOGIES = ("single", "tp", "pd")


def _model(model: str) -> ModelKV:
    try:
        return MODEL_KV[model]
    except KeyError:
        raise CacheBudgetError(
            f"unknown model {model!r} — known: {sorted(MODEL_KV)} (fail closed; "
            "add the charter-derived KV arithmetic before planning)"
        ) from None


def _dtype_factor(kv_dtype: str) -> float:
    try:
        return KV_DTYPE_FACTOR[kv_dtype]
    except KeyError:
        raise CacheBudgetError(
            f"unknown kv_dtype {kv_dtype!r} — known: {sorted(KV_DTYPE_FACTOR)}"
        ) from None


def demand_bytes(
    model: str,
    *,
    concurrency: int,
    avg_seq_tokens: int,
    kv_dtype: str = "bf16",
) -> int:
    """§6.1 demand D: steady-state sequence-state bytes at target concurrency.

    ``concurrency * avg_seq_tokens * kv/token * dtype_factor`` plus, for
    hybrid models, ``concurrency`` fixed recurrent-state slots (NOT dtype
    scaled — the state is model-internal, not KV).
    """
    if concurrency < 1 or avg_seq_tokens < 1:
        raise CacheBudgetError(
            f"concurrency and avg_seq_tokens must be >= 1 "
            f"(got {concurrency}, {avg_seq_tokens})"
        )
    mk = _model(model)
    kv = math.floor(
        concurrency * avg_seq_tokens * mk.kv_bytes_per_token * _dtype_factor(kv_dtype)
    )
    return kv + concurrency * mk.fixed_state_bytes_per_seq


@dataclass(frozen=True)
class EngineArgs:
    """One knob emission: how ONE engine process realizes (part of) the budget."""

    engine: str
    kind: str  # "primary" | "fallback"
    role: str  # "single" | "rank" | "prefill" | "decode"
    args: tuple[str, ...]
    note: str


@dataclass(frozen=True)
class BudgetPlan:
    """The planned budget: what was requested, what gate (j) must observe."""

    model: str
    engine: str
    r: float
    kv_dtype: str
    demand_bytes: int
    budget_bytes_total: int
    budget_tokens_total: int
    tp: int
    topology: str
    per_rank_bytes: int
    per_rank_note: str
    pd_split: float | None
    pools_bytes: tuple[int, int] | None  # (prefill, decode); sums EXACTLY to total
    engine_args: tuple[EngineArgs, ...]
    verify_live: tuple[str, ...]
    gate_j: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def plan_budget(
    *,
    model: str,
    engine: str,
    r: float,
    demand: int,
    kv_dtype: str = "bf16",
    tp: int = 1,
    topology: str = "single",
    pd_split: float | None = None,
    block_size: int = 16,
    free_bytes_after_weights: int | None = None,
) -> BudgetPlan:
    """Plan the launch knobs that realize budget B = r x demand on one engine.

    ``demand`` comes from :func:`demand_bytes` (or the preflight P6 floor
    table). Every knob whose live semantics are unproven is emitted WITH a
    ``verify_live`` entry; gate (j) is the closer.
    """
    mk = _model(model)
    if engine not in _ENGINES:
        raise CacheBudgetError(f"unknown engine {engine!r} — known: {_ENGINES}")
    if engine == "hf":
        raise CacheBudgetError(
            "HF Transformers is the correctness oracle, excluded from pressure "
            "sweeps by protocol P2 ('HF none') — there is no budget to plan"
        )
    if not (math.isfinite(r) and r > 0):
        raise CacheBudgetError(f"budget ratio r must be finite and > 0 (got {r!r})")
    if demand < 1:
        raise CacheBudgetError(f"demand bytes must be >= 1 (got {demand})")
    if topology not in _TOPOLOGIES:
        raise CacheBudgetError(f"unknown topology {topology!r} — known: {_TOPOLOGIES}")
    if not isinstance(tp, int) or tp < 1:
        raise CacheBudgetError(f"tp must be an integer >= 1 (got {tp!r})")
    if topology == "single" and tp != 1:
        raise CacheBudgetError("topology 'single' requires tp=1 (got tp=%d)" % tp)
    if topology == "tp" and tp < 2:
        raise CacheBudgetError("topology 'tp' requires tp >= 2")

    dtype_factor = _dtype_factor(kv_dtype)
    eff_kv_per_token = math.floor(mk.kv_bytes_per_token * dtype_factor)
    budget = math.floor(r * demand)
    tokens_total = budget // eff_kv_per_token

    # --- P/D split (§6.5): explicit, recorded, exact-sum -------------------
    pools: tuple[int, int] | None = None
    if topology == "pd":
        if pd_split is None:
            raise CacheBudgetError(
                "topology 'pd' requires an EXPLICIT pd_split in (0, 1) — §6.5 "
                "records the split as engine dialect; a silent default would "
                "unregister it"
            )
        if not (0.0 < pd_split < 1.0):
            raise CacheBudgetError(f"pd_split must be strictly inside (0, 1), got {pd_split}")
        prefill = math.floor(budget * pd_split)
        pools = (prefill, budget - prefill)  # decode takes the remainder: EXACT sum

    # --- TP semantics: GQA shards; MLA replicates (#20) --------------------
    if tp == 1:
        per_rank = budget
        per_rank_note = "single rank: pool == total budget"
    elif mk.mla_tp_replicated:
        per_rank = budget
        per_rank_note = (
            "MLA latent cache REPLICATES across TP ranks — per-rank bytes == "
            "total budget; the replication cost IS registered contrast #20"
        )
    else:
        per_rank = budget // tp
        per_rank_note = (
            f"GQA KV heads shard across ranks: per-rank = total // tp "
            f"(realized total = {per_rank * tp}; gate (j) checks the realized sum)"
        )

    verify_live: list[str] = []
    args: list[EngineArgs] = []
    hybrid_note = ""
    if mk.fixed_state_bytes_per_seq:
        hybrid_note = (
            " HYBRID: the recurrent-state pool is NOT byte-capped by this knob "
            "(§6.5/§5.1.4) — pressure arrives via the concurrency dial."
        )

    if engine == "vllm":
        verify_live.append(
            "vllm --kv-cache-memory-bytes: per-rank vs whole-pool semantics at the "
            "pinned version [VERIFY-LIVE at S0]; gate (j) closes on the startup log"
        )
        if pools is None:
            args.append(EngineArgs(
                "vllm", "primary", "single" if tp == 1 else "rank",
                ("--kv-cache-memory-bytes", str(per_rank)),
                "direct bytes knob." + hybrid_note,
            ))
            args.append(EngineArgs(
                "vllm", "fallback", "single" if tp == 1 else "rank",
                ("--num-gpu-blocks-override", str((per_rank // eff_kv_per_token) // block_size)),
                f"block-count fallback (block_size={block_size} tokens) for builds "
                "without the bytes flag." + hybrid_note,
            ))
        else:
            for role, pool in zip(("prefill", "decode"), pools):
                args.append(EngineArgs(
                    "vllm", "primary", role,
                    ("--kv-cache-memory-bytes", str(pool)),
                    f"P/D {role} pool; pools sum EXACTLY to the total budget (§6.5)."
                    + hybrid_note,
                ))
            verify_live.append(
                "P/D pool-sum realization (§6.5): two instances' realized pools must "
                "sum to B within gate (j) tolerance [VERIFY-LIVE at S0]"
            )
    elif engine == "sglang":
        if pools is None:
            args.append(EngineArgs(
                "sglang", "primary", "single" if tp == 1 else "rank",
                ("--max-total-tokens", str(tokens_total if tp == 1 else per_rank // eff_kv_per_token)),
                "token-denominated pool cap: tokens = bytes // (kv/token x dtype)."
                + hybrid_note,
            ))
        else:
            for role, pool in zip(("prefill", "decode"), pools):
                args.append(EngineArgs(
                    "sglang", "primary", role,
                    ("--max-total-tokens", str(pool // eff_kv_per_token)),
                    f"P/D {role} pool in tokens; token floors may under-fill by "
                    "< 1 token/pool — gate (j) checks realized bytes (§6.5)."
                    + hybrid_note,
                ))
        verify_live.append(
            "sglang --max-total-tokens -> realized pool bytes at the pinned version "
            "[VERIFY-LIVE at S0]; gate (j) closes on the startup log"
        )
    elif engine == "lmdeploy":
        if free_bytes_after_weights is None:
            raise CacheBudgetError(
                "lmdeploy's cache_max_entry_count is a FRACTION of post-weight free "
                "memory — pass free_bytes_after_weights (from the target GPU) or "
                "plan a different engine"
            )
        if budget > free_bytes_after_weights:
            raise CacheBudgetError(
                f"budget {budget} B exceeds post-weight free memory "
                f"{free_bytes_after_weights} B — the cell is unrealizable on this GPU"
            )
        frac = budget / free_bytes_after_weights
        args.append(EngineArgs(
            "lmdeploy", "primary", "single" if tp == 1 else "rank",
            ("cache_max_entry_count", f"{frac:.6f}"),
            "TurboMind config key (not a CLI flag): fraction of post-weight free "
            "memory; derived from the caller-supplied free bytes." + hybrid_note,
        ))
        verify_live.append(
            "lmdeploy cache_max_entry_count realized bytes + TurboMind actually "
            "selected (not the silent PyTorch fallback) [VERIFY-LIVE at S0]"
        )

    realized_total = per_rank * tp if (tp > 1 and not mk.mla_tp_replicated) else budget
    return BudgetPlan(
        model=model,
        engine=engine,
        r=r,
        kv_dtype=kv_dtype,
        demand_bytes=demand,
        budget_bytes_total=budget,
        budget_tokens_total=tokens_total,
        tp=tp,
        topology=topology,
        per_rank_bytes=per_rank,
        per_rank_note=per_rank_note,
        pd_split=pd_split,
        pools_bytes=pools,
        engine_args=tuple(args),
        verify_live=tuple(verify_live),
        gate_j={
            "expected_bytes_total": realized_total,
            "tolerance_env": "CAGE_ISO_BYTES_TOL (default 0.05)",
            "evidence": "engine startup logs pinned via CAGE_ISO_BYTES_LOGS",
        },
    )


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="cache_budget",
        description="Plan per-engine launch knobs for a bytes-denominated KV budget "
        "(B = r x D). Plans only — gate (j) verifies realized bytes.",
    )
    p.add_argument("--model", required=True, choices=sorted(MODEL_KV))
    p.add_argument("--engine", required=True, choices=[e for e in _ENGINES if e != "hf"])
    p.add_argument("--r", required=True, type=float, help="budget ratio r = B/D")
    p.add_argument("--concurrency", required=True, type=int)
    p.add_argument("--avg-seq-tokens", required=True, type=int)
    p.add_argument("--kv-dtype", default="bf16", choices=sorted(KV_DTYPE_FACTOR))
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--topology", default="single", choices=_TOPOLOGIES)
    p.add_argument("--pd-split", type=float, default=None)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--free-bytes-after-weights", type=int, default=None)
    a = p.parse_args(argv)
    try:
        d = demand_bytes(
            a.model, concurrency=a.concurrency,
            avg_seq_tokens=a.avg_seq_tokens, kv_dtype=a.kv_dtype,
        )
        result = plan_budget(
            model=a.model, engine=a.engine, r=a.r, demand=d, kv_dtype=a.kv_dtype,
            tp=a.tp, topology=a.topology, pd_split=a.pd_split,
            block_size=a.block_size,
            free_bytes_after_weights=a.free_bytes_after_weights,
        )
    except CacheBudgetError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(result.to_json())
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI shim
    raise SystemExit(_main())
