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
- DEC-8 (2026-11-08, MyDocs/RunPod/S2_KV_POOL_AUDIT_2026-11-07.md): on a
  hybrid model (Qwen3-Next) the budget is realized as EQUAL zero-load
  attention-token room T on both engines. The per-token cost is the pod-wide
  figure at the serving tensor-parallel degree (the two KV heads replicate on
  four ranks: 49,152 B per token, not 24,576), vLLM's bytes are an exact
  multiple of its shared block (block geometry from the pinned 0.19.1 rule),
  and the recurrent state is charged per running request from each engine's
  own layout (vLLM: shared blocks inside this pool; SGLang: a separate slot
  pool outside the token cap). Dense models keep every figure at every degree.

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
    "VllmGeometry",
    "attention_bytes_per_token_per_rank",
    "pod_attention_bytes_per_token",
    "vllm_hybrid_geometry",
    "vllm_blocks_for_request",
    "vllm_mamba_cache_mode",
    "VLLM_MAMBA_CACHE_MODE_STATE_BLOCKS",
    "state_bytes_per_request",
    "fixed_state_bytes_per_seq",
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
    mla_tp_replicated: bool = False  # MLA latent replicates across TP ranks (#20)
    # Hybrid (linear-attention) entries only, DEC-8: the facts the per-rank
    # attention cost, vLLM's block geometry and the per-request state are
    # derived from. None / 0 on a dense entry.
    attn_layers: int | None = None
    kv_heads_total: int | None = None
    head_dim: int | None = None
    state_layers: int = 0
    #: recurrent state per layer at TP 1, in elements (sharded by tp)
    state_conv_elements: int = 0  # (conv_kernel - 1) x conv_dim
    state_ssm_elements: int = 0  # v_heads x head_k_dim x head_v_dim

    @property
    def hybrid(self) -> bool:
        return self.attn_layers is not None


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
    # Linear hybrid (DEC-8): 12 of 48 layers full attention with 2 KV heads of
    # 256 -> 24,576 B/tok at TP 1; 12,288 B per rank at TP 2 and TP 4 (the heads
    # replicate at TP 4: 49,152 B pod wide). The 36 Gated DeltaNet layers hold
    # a conv state of (4 - 1) x 8,192 and an SSM state of 32 x 128 x 128
    # elements per layer [V config.json at revision 9c7f2fbe: linear_num_key
    # _heads 16, linear_num_value_heads 32, linear_key_head_dim 128,
    # linear_value_head_dim 128, linear_conv_kernel_dim 4; conv_dim = 16 x 128
    # x 2 + 32 x 128 = 8,192]. The per-request state cost is derived per
    # engine (state_bytes_per_request), never a constant.
    "qwen3-next-80b": ModelKV(
        24_576, "2*12*2*256*2 (12/48 full-attn layers, TP 1)",
        attn_layers=12, kv_heads_total=2, head_dim=256, state_layers=36,
        state_conv_elements=3 * 8192, state_ssm_elements=32 * 128 * 128,
    ),
    # S0 shakedown model only (1x L40S pod; off the D4 roster, never a campaign
    # cell). FP8 weights, BF16 KV under kv_cache_dtype=auto:
    # 2 x 36 x 8 x 128 x 2 = 144 KiB/tok. VERIFY-LIVE at S0-19 (gate (j)).
    "qwen3-8b-fp8": ModelKV(147_456, "2*36*8*128*2"),
}

#: A11 (owner 2026-10-20): the two-sided tolerance of the realized pool, in
#: tokens, against the plan; the campaign driver checks it after every
#: relaunch and a miss stops the run. 0.5 percent sits above the largest S0
#: gap, 0.077 percent (vLLM rounds to 16-token blocks) [V S0 attempt 5
#: serving configs]. Bytes are not compared: vLLM prints its pool in GiB to
#: two decimals, about 0.6 percent of a 4,350-token pool [D].
POOL_TOKEN_TOLERANCE: float = 0.005

_ENGINES = ("vllm", "sglang", "lmdeploy", "hf")
_TOPOLOGIES = ("single", "tp", "pd")

#: vLLM 0.19.1 hybrid geometry [V src]: a block is a multiple of 16 tokens
#: sized so one attention block holds one state page (models/config.py:266-285);
#: the state is bf16 on both parts (mamba_utils.py:55-67, model dtype); a group
#: is the smallest layer count when the largest is at least 1.5 x it
#: (kv_cache_utils.py:1043-1055); under --mamba-cache-mode align a running
#: request holds one block per state group in steady state and two at the
#: step boundary (single_type_kv_cache_manager.py:843-860; the start estimator
#: charges two, kv_cache_interface.py:297-298).
VLLM_BLOCK_ALIGN_TOKENS = 16
VLLM_STATE_DTYPE_BYTES = 2
VLLM_GROUP_RATIO = 1.5
VLLM_STATE_BLOCKS_PER_GROUP = {"steady": 1, "worst": 2}
#: State pages per recurrent group that vLLM's start estimator charges one
#: request of max_model_len tokens, by mamba cache mode: 2 in "align" (the
#: mode a prefix-caching start of Qwen3-Next runs in), 1 in "none" (forced
#: whenever prefix caching is disabled) [V vLLM v0.19.1 kv_cache_interface.py
#: MambaSpec.max_memory_usage_bytes; model_executor/models/config.py, the
#: prefix-off branch]. The block geometry itself is the same in both modes.
VLLM_MAMBA_CACHE_MODE_STATE_BLOCKS = {"align": 2, "none": 1}
#: SGLang 0.5.10.post1 [V src memory_pool.py:253-272, configs/mamba_utils.py:
#: 71-103]: one slot per running request, conv state bf16, SSM state float32
#: unless the config or SGLANG_MAMBA_SSM_DTYPE says otherwise (neither does).
SGLANG_STATE_DTYPE_BYTES = {"conv": 2, "ssm": 4}


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


def _tp(tp: object) -> int:
    if isinstance(tp, bool) or not isinstance(tp, int) or tp < 1:
        raise CacheBudgetError(f"tp must be an integer >= 1 (got {tp!r})")
    return tp


def attention_bytes_per_token_per_rank(model: str, tp: int, kv_dtype: str = "bf16") -> int:
    """Attention KV bytes one rank holds per token at tensor parallel ``tp``.

    Hybrid entry: 2 (K, V) x attention layers x max(1, kv_heads // tp) heads
    x head_dim x 2 B x dtype factor; with fewer heads than ranks every rank
    keeps a whole head, so the pod stores each token more than once (DEC-8).
    Dense entries carry no head facts: GQA shards the table figure by tp [A:
    the registered assumption since C13; every registered dense model has
    at least as many KV heads as ranks], MLA replicates it (#20).
    """
    mk = _model(model)
    tp = _tp(tp)
    factor = _dtype_factor(kv_dtype)
    if mk.hybrid:
        heads = max(1, mk.kv_heads_total // tp)
        return math.floor(2 * mk.attn_layers * heads * mk.head_dim * 2 * factor)
    full = math.floor(mk.kv_bytes_per_token * factor)
    return full if mk.mla_tp_replicated else full // tp


def pod_attention_bytes_per_token(model: str, tp: int, kv_dtype: str = "bf16") -> int:
    """The attention bytes the whole pod holds per token: per rank x tp on a
    hybrid entry (replication included); the table figure on a dense entry,
    where sharding and the registered MLA contrast leave it unchanged."""
    mk = _model(model)
    if mk.hybrid:
        return attention_bytes_per_token_per_rank(model, tp, kv_dtype) * tp
    _tp(tp)
    return math.floor(mk.kv_bytes_per_token * _dtype_factor(kv_dtype))


@dataclass(frozen=True)
class VllmGeometry:
    """vLLM's hybrid KV layout on one rank (DEC-8): ``block_tokens`` tokens of
    attention per block, ``page_bytes`` per block per layer, ``group_size``
    layers per group sharing one page, ``n_groups`` groups (one attention,
    the rest state) drawing blocks from ONE pool, ``shared_block_bytes`` =
    group_size x page_bytes per block, and the unpadded state page."""

    tp: int
    block_tokens: int
    page_bytes: int
    group_size: int
    n_groups: int
    shared_block_bytes: int
    state_page_bytes: int

    def record(self, blocks_per_rank) -> dict:
        return {
            "tp": self.tp,
            "block_tokens": self.block_tokens,
            "page_bytes": self.page_bytes,
            "group_size": self.group_size,
            "n_groups": self.n_groups,
            "shared_block_bytes": self.shared_block_bytes,
            "blocks_per_rank": blocks_per_rank,
            "null_blocks": 1,
        }


def vllm_hybrid_geometry(model: str, tp: int, kv_dtype: str = "bf16") -> VllmGeometry:
    """The pinned vLLM 0.19.1 rule for a hybrid entry at ``tp``:
    block = 16 x ceil(state page / (16 x attention bytes per token per layer
    per rank)); page = block x that attention figure; groups of the smallest
    layer count. Refuses a dense entry, a tp that does not divide the state
    shapes, and a layer ratio under 1.5 (that branch of the grouping rule is
    not pinned)."""
    mk = _model(model)
    tp = _tp(tp)
    if not mk.hybrid:
        raise CacheBudgetError(
            f"{model!r} has no hybrid geometry (dense model: tokens = bytes // kv per token)"
        )
    if mk.state_conv_elements % tp or mk.state_ssm_elements % tp:
        raise CacheBudgetError(
            f"tp={tp} does not divide the recurrent-state shapes of {model!r} "
            f"({mk.state_conv_elements} conv, {mk.state_ssm_elements} ssm elements per layer)"
        )
    heads = max(1, mk.kv_heads_total // tp)
    attn_per_layer = math.floor(2 * heads * mk.head_dim * 2 * _dtype_factor(kv_dtype))
    state_page = (mk.state_conv_elements // tp + mk.state_ssm_elements // tp) * VLLM_STATE_DTYPE_BYTES
    align = VLLM_BLOCK_ALIGN_TOKENS
    block = align * (-(-state_page // (align * attn_per_layer)))
    lo, hi = sorted((mk.attn_layers, mk.state_layers))
    if hi < VLLM_GROUP_RATIO * lo or (mk.attn_layers + mk.state_layers) % lo:
        raise CacheBudgetError(
            f"{model!r}: layer counts {mk.attn_layers} attention / {mk.state_layers} state "
            f"fall outside the pinned grouping rule (largest >= {VLLM_GROUP_RATIO} x smallest, "
            "smallest divides the total)"
        )
    page = block * attn_per_layer
    return VllmGeometry(
        tp=tp, block_tokens=block, page_bytes=page, group_size=lo,
        n_groups=(mk.attn_layers + mk.state_layers) // lo,
        shared_block_bytes=lo * page, state_page_bytes=state_page,
    )


def vllm_mamba_cache_mode(prefix_caching: bool) -> str:
    """The mamba cache mode a vLLM start of the hybrid runs in: "align" with
    prefix caching on, "none" with it off (vLLM forces it) [V models/config.py]."""
    return "align" if prefix_caching else "none"


def vllm_blocks_for_request(
    model: str, tp: int, max_model_len: int, kv_dtype: str = "bf16", mamba_cache_mode: str = "align"
) -> int:
    """Blocks one request of ``max_model_len`` tokens needs on vLLM: ceil(L /
    block) attention blocks plus VLLM_MAMBA_CACHE_MODE_STATE_BLOCKS[mode] per
    state group, the figure vLLM's start check and its concurrency line use
    [V kv_cache_utils.py:615-660, 802-820, 1332-1353; kv_cache_interface.py
    MambaSpec.max_memory_usage_bytes]."""
    if isinstance(max_model_len, bool) or not isinstance(max_model_len, int) or max_model_len < 1:
        raise CacheBudgetError(f"max_model_len must be an integer >= 1 (got {max_model_len!r})")
    if mamba_cache_mode not in VLLM_MAMBA_CACHE_MODE_STATE_BLOCKS:
        raise CacheBudgetError(
            f"mamba_cache_mode {mamba_cache_mode!r} is not one of {sorted(VLLM_MAMBA_CACHE_MODE_STATE_BLOCKS)}"
        )
    g = vllm_hybrid_geometry(model, tp, kv_dtype)
    return -(-max_model_len // g.block_tokens) + VLLM_MAMBA_CACHE_MODE_STATE_BLOCKS[mamba_cache_mode] * (g.n_groups - 1)


def state_bytes_per_request(model: str, engine: str, tp: int, mode: str = "steady") -> int:
    """Recurrent-state bytes one running request costs the whole pod on
    ``engine`` at ``tp``; 0 on a dense entry. vLLM: one shared block per state
    group (``steady``) or two (``worst``, the step boundary and the start
    estimator). SGLang: one slot (``mode='slot'``), conv bf16 and SSM float32
    over the state layers. The engine and mode are validated first."""
    mk = _model(model)
    tp = _tp(tp)
    if engine == "vllm":
        if mode not in VLLM_STATE_BLOCKS_PER_GROUP:
            raise CacheBudgetError(f"vllm state mode {mode!r} is not one of {sorted(VLLM_STATE_BLOCKS_PER_GROUP)}")
        if not mk.hybrid:
            return 0
        g = vllm_hybrid_geometry(model, tp)
        return VLLM_STATE_BLOCKS_PER_GROUP[mode] * (g.n_groups - 1) * g.shared_block_bytes * tp
    if engine == "sglang":
        if mode != "slot":
            raise CacheBudgetError(f"sglang state mode {mode!r} must be 'slot'")
        if not mk.hybrid:
            return 0
        if mk.state_conv_elements % tp or mk.state_ssm_elements % tp:
            raise CacheBudgetError(f"tp={tp} does not divide the recurrent-state shapes of {model!r}")
        per_rank = mk.state_layers * (
            (mk.state_conv_elements // tp) * SGLANG_STATE_DTYPE_BYTES["conv"]
            + (mk.state_ssm_elements // tp) * SGLANG_STATE_DTYPE_BYTES["ssm"]
        )
        return per_rank * tp
    raise CacheBudgetError(f"no recurrent-state layout pinned for engine {engine!r} (vllm, sglang)")


def fixed_state_bytes_per_seq(model: str, tp: int = 1) -> int:
    """The per-sequence state term of demand D: vLLM's steady cost (the
    engine whose pool the state shares), 0 on a dense model."""
    return state_bytes_per_request(model, "vllm", tp, "steady")


def demand_bytes(
    model: str,
    *,
    concurrency: int,
    avg_seq_tokens: int,
    kv_dtype: str = "bf16",
    tp: int = 1,
) -> int:
    """§6.1 demand D: steady-state sequence-state bytes at target concurrency.

    ``concurrency * avg_seq_tokens * pod attention bytes per token`` (the
    serving degree ``tp`` matters on a hybrid entry only) plus, for hybrid
    models, ``concurrency`` recurrent-state slots at vLLM's steady cost (NOT
    dtype scaled; the state is model-internal, not KV).
    """
    if concurrency < 1 or avg_seq_tokens < 1:
        raise CacheBudgetError(
            f"concurrency and avg_seq_tokens must be >= 1 "
            f"(got {concurrency}, {avg_seq_tokens})"
        )
    kv = concurrency * avg_seq_tokens * pod_attention_bytes_per_token(model, tp, kv_dtype)
    return kv + concurrency * fixed_state_bytes_per_seq(model, tp)


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
    """The planned budget: what was requested, what the realized pool is checked against."""

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
    # DEC-8: per-role per-rank bytes and tokens on a pd plan (None without a
    # role degree), vLLM's block geometry and the per-request state costs on a
    # hybrid plan (None on a dense model).
    pools_rank_bytes: tuple[int, int] | None = None
    pools_tokens: tuple[int, int] | None = None
    geometry: dict | None = None
    state_per_request_bytes: dict | None = None

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
    role_tp: int | None = None,
) -> BudgetPlan:
    """Plan the launch knobs that realize budget B = r x demand on one engine.

    ``demand`` comes from :func:`demand_bytes` (or the preflight P6 floor
    table). Every knob whose live semantics are unproven is emitted WITH a
    ``verify_live`` entry; gate (j) is the closer. ``role_tp`` is the degree
    each pd role instance launches with: it sizes the per-role per-rank
    slices (``pools_rank_bytes``) and, on a hybrid model, the role geometry.
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
    if role_tp is not None:
        if topology != "pd":
            raise CacheBudgetError("role_tp applies to topology 'pd' only")
        _tp(role_tp)

    dtype_factor = _dtype_factor(kv_dtype)
    eff_kv_per_token = math.floor(mk.kv_bytes_per_token * dtype_factor)
    budget = math.floor(r * demand)
    tokens_total = budget // eff_kv_per_token
    if mk.hybrid and engine == "lmdeploy":
        raise CacheBudgetError(
            f"lmdeploy has no hybrid geometry pinned for {model!r}; plan vllm or sglang"
        )
    if mk.hybrid and topology == "pd" and role_tp is None:
        raise CacheBudgetError(
            f"a pd plan of the hybrid {model!r} requires role_tp (the role geometry depends on it)"
        )

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
    pools_rank: tuple[int, int] | None = None
    pools_tokens: tuple[int, int] | None = None
    geometry: dict | None = None
    state_record: dict | None = None
    fallback_blocks = tokens_total // block_size
    if mk.hybrid:
        # DEC-8: equal zero-load attention-token room T on both engines. The
        # pod cost per token at the instance's degree buys whole vLLM blocks;
        # one block is the null block, so T = (n - 1) x block; vLLM gets
        # exactly n shared blocks per rank, SGLang gets T.
        geo_tp = role_tp if topology == "pd" else tp
        geo = vllm_hybrid_geometry(model, geo_tp, kv_dtype)
        pod_cost = pod_attention_bytes_per_token(model, geo_tp, kv_dtype)

        def _blocks(pool_bytes: int) -> int:
            n = (pool_bytes // pod_cost) // geo.block_tokens
            if n < 2:
                raise CacheBudgetError(
                    f"budget {pool_bytes} B holds {n} vLLM block(s) of {geo.block_tokens} tokens "
                    f"at tp={geo_tp} ({pod_cost} B per token pod wide); fewer than two blocks "
                    "(one is the null block) leaves no token room: the cell is unrealizable"
                )
            return n

        if pools is not None:
            ns = [_blocks(pool) for pool in pools]
            pools_rank = tuple(n * geo.shared_block_bytes for n in ns)
            pools_tokens = tuple((n - 1) * geo.block_tokens for n in ns)
            tokens_total = sum(pools_tokens)
            per_rank = budget
            per_rank_note = (
                f"pd: each role holds its pool's whole vLLM blocks at role tp={geo_tp} "
                "(pools_rank_bytes per rank, pools_tokens of attention room)"
            )
            realized_total = sum(pools_rank) * geo_tp
            geometry = geo.record(list(ns))
        else:
            n = _blocks(budget)
            tokens_total = (n - 1) * geo.block_tokens
            per_rank = n * geo.shared_block_bytes
            fallback_blocks = n
            per_rank_note = (
                f"hybrid: {n} shared blocks of {geo.block_tokens} tokens per rank, zero slack "
                f"({pod_cost} B per token pod wide at tp={tp}; realized total = {per_rank * tp})"
            )
            realized_total = per_rank * tp
            geometry = geo.record(n)
        state_record = {
            "vllm_steady": state_bytes_per_request(model, "vllm", geo_tp, "steady"),
            "vllm_worst": state_bytes_per_request(model, "vllm", geo_tp, "worst"),
            "sglang_slot": state_bytes_per_request(model, "sglang", geo_tp, "slot"),
        }
    else:
        if tp == 1:
            per_rank = budget
            per_rank_note = "single rank: pool == total budget"
        elif mk.mla_tp_replicated:
            per_rank = budget
            per_rank_note = (
                "MLA latent cache REPLICATES across TP ranks: per-rank bytes == "
                "total budget; the replication cost IS registered contrast #20"
            )
        else:
            per_rank = budget // tp
            per_rank_note = (
                f"GQA KV heads shard across ranks: per-rank = total // tp "
                f"(realized total = {per_rank * tp}; gate (j) checks the realized sum)"
            )
        realized_total = per_rank * tp if (tp > 1 and not mk.mla_tp_replicated) else budget
        if pools is not None:
            pools_tokens = tuple(pool // eff_kv_per_token for pool in pools)
            if role_tp is not None:
                sharded = role_tp >= 2 and not mk.mla_tp_replicated
                pools_rank = tuple(pool // role_tp if sharded else pool for pool in pools)

    verify_live: list[str] = []
    args: list[EngineArgs] = []
    hybrid_note = ""
    if mk.hybrid:
        hybrid_note = (
            " HYBRID (DEC-8): the knob buys the zero-load attention-token room T, equal "
            "on both engines; vLLM holds the recurrent state inside this pool (shared "
            "blocks per running request), SGLang in a separate slot pool outside the "
            "token cap; pressure on the state arrives via the concurrency dial."
        )

    if engine == "vllm":
        verify_live.append(
            "vllm --kv-cache-memory-bytes: per-rank vs whole-pool semantics at the "
            "pinned version [VERIFY-LIVE at S0]; the campaign driver compares the "
            "realized pool tokens with the plan after every relaunch (A11)"
        )
        if pools is None:
            args.append(EngineArgs(
                "vllm", "primary", "single" if tp == 1 else "rank",
                ("--kv-cache-memory-bytes", str(per_rank)),
                "direct bytes knob." + hybrid_note,
            ))
            # C13 (2026-10-10): a block holds block_size token slots on every
            # rank (each rank stores its head shard of those tokens), so the
            # count is tokens_total // block_size at any tp. per_rank divided
            # by the FULL-model bytes per token gave 1/tp of it under GQA.
            args.append(EngineArgs(
                "vllm", "fallback", "single" if tp == 1 else "rank",
                ("--num-gpu-blocks-override", str(fallback_blocks)),
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
            # C13 (2026-10-10): the token cap is the pool's token-slot count,
            # tokens_total at any tp [D: per-rank bytes budget/tp over
            # per-rank bytes per token kv/tp]. per_rank // eff_kv_per_token was
            # tokens_total / tp under GQA, a quarter of vLLM's realized pool
            # at session b's tp=4 [A: slot semantics; gate (j) closes live].
            args.append(EngineArgs(
                "sglang", "primary", "single" if tp == 1 else "rank",
                ("--max-total-tokens", str(tokens_total)),
                "token-denominated pool cap: tokens = bytes // (kv/token x dtype)."
                + hybrid_note,
            ))
        else:
            for role, pool_tokens in zip(("prefill", "decode"), pools_tokens):
                args.append(EngineArgs(
                    "sglang", "primary", role,
                    ("--max-total-tokens", str(pool_tokens)),
                    f"P/D {role} pool in tokens; token floors may under-fill by "
                    "< 1 token/pool — gate (j) checks realized bytes (§6.5)."
                    + hybrid_note,
                ))
        verify_live.append(
            "sglang --max-total-tokens -> realized pool bytes at the pinned version "
            "[VERIFY-LIVE at S0]; the campaign driver compares the realized pool "
            "tokens with the plan after every relaunch (A11)"
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
        # A11 item 2 (owner 2026-10-20): what is checked, never a claim that
        # it was; the per-relaunch result rides each window's metrics.json
        gate_j={
            "expected_bytes_total": realized_total,
            "expected_tokens_total": tokens_total,
            "tolerance_tokens": POOL_TOKEN_TOLERANCE,
            "check": "run_campaign realized_pool_check after every relaunch (single topology)",
        },
        pools_rank_bytes=pools_rank,
        pools_tokens=pools_tokens,
        geometry=geometry,
        state_per_request_bytes=state_record,
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
    p.add_argument("--tp", type=int, default=1,
                   help="the serving degree; sizes the demand on a hybrid model (DEC-8)")
    p.add_argument("--topology", default="single", choices=_TOPOLOGIES)
    p.add_argument("--pd-split", type=float, default=None)
    p.add_argument("--role-tp", type=int, default=None, help="pd: the degree of each role instance")
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--free-bytes-after-weights", type=int, default=None)
    a = p.parse_args(argv)
    try:
        d = demand_bytes(
            a.model, concurrency=a.concurrency,
            avg_seq_tokens=a.avg_seq_tokens, kv_dtype=a.kv_dtype, tp=a.tp,
        )
        # DEC-8: a degree above 1 with the default topology is the sharded
        # single instance (the rule run_campaign applies to single relaunches)
        topology = "tp" if a.topology == "single" and a.tp > 1 else a.topology
        result = plan_budget(
            model=a.model, engine=a.engine, r=a.r, demand=d, kv_dtype=a.kv_dtype,
            tp=a.tp, topology=topology, pd_split=a.pd_split,
            block_size=a.block_size,
            free_bytes_after_weights=a.free_bytes_after_weights, role_tp=a.role_tp,
        )
    except CacheBudgetError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(result.to_json())
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI shim
    raise SystemExit(_main())
