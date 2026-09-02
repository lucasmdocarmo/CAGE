#!/bin/bash
# shellcheck shell=bash
# Order:     sourced library, never executed
# Objective: Single source of truth for the uniform serving regime (non-eager, max_len, mem-util + per-engine iso-bytes dial mappings)
# Cloud:     both
# =============================================================================
# CAGE uniform serving configuration  (single source of truth; SOURCEABLE --
# deliberately no set -e/-u here: this file must never change the caller's options)
# =============================================================================
# Sourced by EVERY single-node baseline-tree driver -- run_baselines.sh (core baselines),
# run_compression.sh (compression 2x2), and historically the retired
# scripts/deprecated/run_speculative_matrix.sh (speculative 2x2, charter §7.5) --
# so all trees serve under IDENTICAL conditions and cross-mechanism comparisons are
# FAIR. Consumed by scripts/2_serving/manage_vllm_server.sh, which reads these env vars when it launches
# vLLM. The phase3 cluster path (manage_vllm_cluster.py) consumes the SAME VLLM_* env via
# build_serve_args() with fallbacks mirroring this file -- gap closed 2026-07-15 (task #63);
# source this file before cluster bring-up so overrides propagate.
#
# WHY THIS EXISTS (Option A, 2026-07-14): previously the trees diverged --
#   core:        non-eager, max_len 8192, gpu-mem-util 0.92
#   compression: --enforce-eager, max_len 4096, 0.92
#   speculative: --enforce-eager, max_len 4096, 0.90
# so a cross-tree serving delta (esp. TPOT) mixed the MECHANISM with an eager-vs-compiled +
# context-length + memory-util artifact, and cross-tree numbers were only comparable within a
# tree. Holding these three variables identical removes that confound: a cross-mechanism
# serving/quality delta is now attributable to the mechanism, not the serving regime.
#
# CONFOUND-CONTROLLED VARIABLES (held IDENTICAL across all trees):
#   VLLM_ENFORCE_EAGER=0        non-eager (CUDA graphs ON) -- production-realistic decode; the
#                               eager penalty that inflated lever-tree TPOT is removed.
#   VLLM_MAX_MODEL_LEN=4096     ample for SQuAD (contexts are short paragraphs); uniform so the
#                               KV-planning/chunked-prefill regime is identical across trees.
#
# THE SWEPT AXIS (the memory-pressure trade-off distribution objective):
#   VLLM_GPU_MEMORY_UTILIZATION default 0.90 is the uniform OPERATING POINT for a like-for-like
#   baseline comparison. The memory-pressure study OVERRIDES this one variable to trace the
#   trade-off, e.g.:  for p in 0.80 0.85 0.90 0.95; do
#                        VLLM_GPU_MEMORY_UTILIZATION=$p bash scripts/3_run/cloud_run.sh ...; done
#   Holding eager + max_len fixed means the pressure sweep varies ONLY memory, cleanly.
#
# Every value is overridable (:-default), so the pre-flight can fall back to eager for a single
# tree if a cell OOMs non-eager on the 24GB L4:
#   VLLM_ENFORCE_EAGER=1 bash scripts/3_run/run_compression.sh <model>
# Such a fallback is a DELIBERATE, RECORDED deviation -- the run manifest captures the actual
# enforce_eager/max_model_len used, so any non-uniform cell is visible in provenance.
# =============================================================================

export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"

printf '[cage] serving config: enforce_eager=%s max_model_len=%s gpu_mem_util=%s (uniform across trees; mem-util is the swept axis)\n' \
  "$VLLM_ENFORCE_EAGER" "$VLLM_MAX_MODEL_LEN" "$VLLM_GPU_MEMORY_UTILIZATION"

# =============================================================================
# §6.5 iso-BYTES cross-engine budget mapping  (charter D2; walkthrough D1 fix)
# =============================================================================
# The uniform operating point above is expressed in vLLM's dial semantics: a
# fraction F of TOTAL GPU memory the engine may occupy (weights + workspace +
# KV pool). The other charter engines meter their KV budget with dials that
# have DIFFERENT denominators, so passing F through verbatim would hand each
# engine a different byte budget and break §6.5 fairness. These helpers derive
# each engine's native dial value so all engines TARGET the same byte budget.
# The mapping is first-order: preflight gate (j) — the CAGE-ISO-BYTES-GATE in
# scripts/checks/preflight_check.sh (built by task #138; fixture-tested in
# tests/test_preflight_gates.py) — reads each engine's REALIZED KV-pool size
# from its startup log and asserts pairwise parity within CAGE_ISO_BYTES_TOL —
# the helper sets the dial, the gate proves the bytes. Consumed by
# scripts/2_serving/manage_sglang_server.sh and manage_lmdeploy_server.sh.

# Total VRAM of GPU 0 in MiB (single-GPU groups A/B + S0 scope; the multi-GPU
# phase-3 path has its own cluster tooling). Empty output = no nvidia-smi.
cage_gpu_total_mib() {
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1
}

# SGLang --mem-fraction-static: first-order one-to-one with vLLM's dial, NOT
# identical [VERIFY-LIVE at S0]. Two documented divergences (adversarial
# review 2026-08-12): (a) numerator contents differ — vLLM's F also covers its
# profiled activation workspace while SGLang budgets activations from 1-F, so
# the same F OVERSHOOTS SGLang's KV bytes by roughly that workspace; (b) SGLang
# sizes against memory available at init, which equals device total only on an
# EMPTY GPU — co-resident metric models shrink it. Kept as an identity dial
# deliberately: the workspace is not knowable statically, and the preflight
# realized-bytes gate (run WITH the co-resident stack loaded) is the actual
# equalizer and recorder. This helper exists so any future correction has
# exactly one place to land.
cage_sglang_mem_fraction() {
    printf '%s\n' "${VLLM_GPU_MEMORY_UTILIZATION}"
}

# LMDeploy (TurboMind) cache_max_entry_count is a fraction of the FREE memory
# remaining AFTER the weights are loaded (LMDeploy >= 0.4 semantics). Matching
# vLLM's byte budget therefore needs the weight footprint W:
#     frac_free = (F*T - W) / (T - W)      [T = total VRAM, all in GiB]
# Args: $1 = model weight footprint in GiB (fp16/bf16 ~ 2 x params-in-B).
# KNOWN BIASES, both documented so the S0 gate outcome is interpretable
# (adversarial review 2026-08-12): the numerator F*T-W is an UPPER bound of
# vLLM's realized KV pool (vLLM further subtracts its profiled activation
# workspace, ~1-3 GiB at 4k ctx), and the denominator T-W idealizes LMDeploy's
# measured-at-runtime free memory (real free also excludes the CUDA context and
# any co-resident GPU processes). Net: the dial targets slightly MORE KV than
# vLLM realizes. The preflight realized-bytes gate (run with the co-resident
# stack loaded) is the equalizer; expected-vs-realized lands in provenance.
# Fails (non-zero) on malformed inputs, no KV headroom, or missing nvidia-smi
# — callers die rather than launch an unmapped dial.
cage_lmdeploy_cache_fraction() {
    local weights_gib="$1" total_mib
    # Fail-closed input validation: awk would silently coerce a malformed W or
    # F to 0 and emit a confidently WRONG dial with exit 0 — the exact
    # "unmapped dial" this helper's fail-closed contract forbids.
    case "$weights_gib" in
        ''|*[!0-9.]*|.|*.*.*) return 1 ;;
    esac
    case "${VLLM_GPU_MEMORY_UTILIZATION:-}" in
        ''|*[!0-9.]*|.|*.*.*) return 1 ;;
    esac
    total_mib="$(cage_gpu_total_mib)"
    if [ -z "$total_mib" ]; then
        return 1
    fi
    # LC_ALL=C: a comma-radix locale prints "0,8500", which the server's float
    # parser rejects only after burning the full readiness timeout.
    LC_ALL=C awk -v F="$VLLM_GPU_MEMORY_UTILIZATION" -v T_mib="$total_mib" -v W="$weights_gib" 'BEGIN {
        T = T_mib / 1024.0
        kv = F * T - W          # KV-budget proxy (GiB); upper bound of vLLM realized pool
        free = T - W            # LMDeploy dial denominator (GiB), idealized
        if (kv <= 0 || free <= 0) exit 1
        f = kv / free
        if (f > 0.95) f = 0.95  # never hand the engine ~all free memory
        printf "%.4f\n", f
    }'
}

# =============================================================================
# Bytes-denominated budget knobs  (T2.1 — CacheBudgetPlanner -> launcher wiring)
# =============================================================================
# src/orchestration/cache_budget.py PLANS a byte budget (charter P2: pressure =
# bytes, the SAME physical quantity on every engine) and emits per-engine launch
# knobs. The launchers consume them via env — the planner never launches:
#   CAGE_KV_BUDGET_BYTES           vllm   --kv-cache-memory-bytes    (primary)
#   CAGE_VLLM_GPU_BLOCKS_OVERRIDE  vllm   --num-gpu-blocks-override  (fallback)
#   CAGE_SGLANG_MAX_TOTAL_TOKENS   sglang --max-total-tokens
# Validators live HERE (one source of truth; launchers call `... || die`) so a
# malformed budget is refused BEFORE any server is stopped or launched. Fail
# closed: a bad value must never degrade to fraction-only serving, because the
# run's data would then be LABELED with a budget the engine never had. Gate (j)
# (CAGE-ISO-BYTES-GATE) verifies the REALIZED bytes from the startup logs.

# cage_require_positive_int <ENV_NAME> <value> — 0 iff <value> is a positive
# decimal integer. Pure string logic (no shell arithmetic): a byte budget can
# exceed int64 territory where `[ .. -gt 0 ]` errors (exit 2, swallowed by an
# if) instead of refusing. Empty/sign/decimal/whitespace all refuse — awk or
# printf would silently coerce them into a confidently wrong flag value.
cage_require_positive_int() {
    case "$2" in
        ''|*[!0-9]*)
            printf '[cage] REFUSING launch: %s=%s is not a positive integer\n' \
                "$1" "${2:-<empty>}" >&2
            return 1
            ;;
    esac
    case "$2" in
        0?*)
            # '01' parses as 1 engine-side but defeats the space-anchored
            # textual dial-parity check (live '--flag 01' vs requested '1'
            # never matches -> needless restarts, or worse a stale reuse) —
            # refuse the ambiguous spelling outright (2026-09-01 verifier minor).
            printf '[cage] REFUSING launch: %s=%s has a leading zero — write the plain decimal\n' \
                "$1" "$2" >&2
            return 1
            ;;
    esac
    case "$2" in
        *[!0]*) return 0 ;;  # digits-only AND at least one nonzero digit
    esac
    printf '[cage] REFUSING launch: %s=%s must be > 0 (a zero budget is a refusal, not a config)\n' \
        "$1" "$2" >&2
    return 1
}

# vLLM budget knobs: bytes (primary) XOR blocks-override (fallback). BOTH set
# is two different caps for the same KV pool — a refusal, never a precedence
# rule (fail-closed doctrine: no silent pick).
cage_validate_vllm_budget_env() {
    if [ -n "${CAGE_KV_BUDGET_BYTES:-}" ] && [ -n "${CAGE_VLLM_GPU_BLOCKS_OVERRIDE:-}" ]; then
        printf '[cage] REFUSING launch: CAGE_KV_BUDGET_BYTES and CAGE_VLLM_GPU_BLOCKS_OVERRIDE are BOTH set -- they cap the SAME KV pool in different units; unset one\n' >&2
        return 1
    fi
    if [ -n "${CAGE_KV_BUDGET_BYTES:-}" ]; then
        cage_require_positive_int CAGE_KV_BUDGET_BYTES "${CAGE_KV_BUDGET_BYTES}" || return 1
    fi
    if [ -n "${CAGE_VLLM_GPU_BLOCKS_OVERRIDE:-}" ]; then
        cage_require_positive_int CAGE_VLLM_GPU_BLOCKS_OVERRIDE "${CAGE_VLLM_GPU_BLOCKS_OVERRIDE}" || return 1
    fi
    return 0
}

cage_validate_sglang_budget_env() {
    if [ -n "${CAGE_SGLANG_MAX_TOTAL_TOKENS:-}" ]; then
        cage_require_positive_int CAGE_SGLANG_MAX_TOTAL_TOKENS "${CAGE_SGLANG_MAX_TOTAL_TOKENS}" || return 1
    fi
    return 0
}

# =============================================================================
# Tensor-parallel launch knobs  (T3.1 — Wave-3 distributed serving stack)
# =============================================================================
# The 2026-08-27 audit verified NO launcher passed any TP flag, so every
# multi-GPU matrix cell (VLLM_COMPATIBILITY.md §7: Llama-3.3-70B TP=4,
# DeepSeek-V3 TP=8, SGLang pure-TP V3 cells) would silently have served TP=1.
# Env contract (validated here, one source of truth; launchers `|| die`):
#   CAGE_VLLM_TENSOR_PARALLEL  positive int -> vllm `--tensor-parallel-size N`
#   CAGE_SGLANG_TP             positive int -> sglang `--tp-size N`
#                              [VERIFY-LIVE at Run-C-prime preflight: exact
#                              SGLang flag spelling — docs say --tp-size with
#                              --tp as alias, but no SGLang pin exists yet]
# Value 1 is VALID and means the flag is OMITTED ENTIRELY: the single-GPU
# argv must stay byte-identical to the pre-T3.1 launch (pinned differentially
# in tests/test_tp_flags.py), so engine-default TP handling is untouched.
# Fail closed: a malformed degree must refuse BEFORE any server is touched —
# a launch that silently dropped to the engine default would label a TP=4
# sweep cell with parallelism the server never had.

cage_validate_vllm_tp_env() {
    if [ -n "${CAGE_VLLM_TENSOR_PARALLEL:-}" ]; then
        cage_require_positive_int CAGE_VLLM_TENSOR_PARALLEL "${CAGE_VLLM_TENSOR_PARALLEL}" || return 1
    fi
    return 0
}

cage_validate_sglang_tp_env() {
    if [ -n "${CAGE_SGLANG_TP:-}" ]; then
        cage_require_positive_int CAGE_SGLANG_TP "${CAGE_SGLANG_TP}" || return 1
    fi
    return 0
}
