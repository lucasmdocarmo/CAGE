#!/bin/bash
# =============================================================================
# CAGE memory-pressure sweep: gpu_memory_utilization is THE swept axis.
#
# MECHANISM UNDER TEST (single-stream L4): memory pressure manifests as
# PREFIX-CACHE EVICTION and KV-capacity shrinkage, not preemption storms. The
# sweep varies VLLM_GPU_MEMORY_UTILIZATION so KV capacity BRACKETS the cag_true
# corpus block (~CORPUS_BUDGET=2800 tokens):
#   0.90   -> capacity ~22.7k tokens: corpus + per-query contexts all fit
#   0.84   -> capacity approaches the corpus block: eviction begins
#   0.815  -> capacity ~undershoots the block: cag_true_on collapses toward its
#             OFF behavior (corpus KV cannot survive between queries), while
#             lmcache_rag's CPU offload should retain hits.
# Per-trial readout comes from the telemetry counters this sweep was built with:
# prefill/decode/inference/queue time SUM+COUNT deltas, preemptions_total,
# cached/recomputed tokens, kv_capacity_tokens, plus telemetry_series.jsonl.
#
# ARMS per utilization (4 x NUM_QUERIES x NUM_TRIALS):
#   no_cache       server --no-prefix-cache          (floor: recompute everything)
#   prefix_cache   server with prefix caching        (per-query contexts, ~32-tok prefix)
#   cag_true_on    prefix caching + corpus-as-prefix (the arm the bracket squeezes)
#   lmcache_rag    prefix caching + LMCache connector (CPU-offload retention arm;
#                  SKIPPED with a STATUS sentinel unless CAGE_ENABLE_LMCACHE=1)
#
# CALIBRATION GATE: after every server boot the actual kv_capacity_tokens is read
# from cage-stats/vLLM, echoed, and written to each cell dir (kv_capacity.json) so
# the capacity-vs-corpus bracket is provenance, not an assumption. A server that
# refuses to boot fails that util's pending arms LOUDLY (STATUS=failed reason=server).
#
# THE OVERRIDE: scripts/lib/_serving_config.sh owns VLLM_GPU_MEMORY_UTILIZATION
# (default 0.90) and scripts/2_serving/manage_vllm_server.sh passes it as
# --gpu-memory-utilization. This sweep re-exports that ONE variable per util;
# eager mode + max_model_len stay at the uniform Option-A values, so the swept
# delta is memory alone.
#
# Outputs: $CAGE_RUN_ROOT/memory_sweep/util_<val>_<arm>/trial_*/
# Resume: complete cells skipped; CAGE_FORCE_RERUN=1 wipes and re-runs.
# Usage: [MEM_UTILS="0.90 0.84 0.815" NUM_QUERIES=100 NUM_TRIALS=3] \
#            bash scripts/3_run/run_memory_sweep.sh [model]
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
source scripts/lib/_serving_config.sh
source scripts/lib/_log_guard.sh

MODEL=${1:-"Qwen/Qwen3-8B"}
DATASET="${DATASET:-squad_v2}"
NUM_QUERIES=${NUM_QUERIES:-100}
NUM_TRIALS=${NUM_TRIALS:-3}
SEED=${SEED:-42}
# DECOUPLED SCORING (default ON, like run_full_sweep.sh): this sweep's axis is the
# memory-pressure SERVING/TELEMETRY readout (TTFT, preemptions, kv_capacity, phase-time
# counters), NOT model-based quality. Running the CPU quality models (LettuceDetect/NLI/
# BERTScore) INLINE per query made the long-context arms (cag_true_on, lmcache_rag) take
# ~90 min/trial vs ~4 min for short-context arms -- a ~15-18x slowdown for metrics this
# sweep does not need (squad_v2 quality is already scored in the main run). F1/EM/abstention
# stay inline (model-free). Set CAGE_SKIP_QUALITY=0 to restore inline model scoring; quality
# under pressure can be rescored offline from qa_evidence.jsonl if ever wanted.
export CAGE_SKIP_QUALITY="${CAGE_SKIP_QUALITY:-1}"
CORPUS_BUDGET=${CORPUS_BUDGET:-2800}
MEM_UTILS=${MEM_UTILS:-"0.90 0.84 0.815"}
PORT=${VLLM_PORT:-8000}
OUTPUT_DIR="${CAGE_RUN_ROOT:-results/phase2/local}/memory_sweep"
mkdir -p "$OUTPUT_DIR"

# Redundant cloud backup of the whole results/<phase>/ tree for the memory sweep's
# duration too (mirrors the full-sweep behavior). LOUD no-op if CAGE_RESULTS_BUCKET unset.
_MS_PHASE="${CAGE_PHASE:-phase2}"
bash scripts/5_observability/gcs_backup_daemon.sh start "results/${_MS_PHASE}" || true
trap 'bash scripts/5_observability/gcs_backup_daemon.sh stop "results/'"${_MS_PHASE}"'" >/dev/null 2>&1 || true' EXIT

# Telemetry defaults ON here (unlike the baseline trees): the phase-time counters,
# preemptions_total, and telemetry_series.jsonl ARE this sweep's readout.
TELEMETRY_FLAG=""
if [ "${VLLM_TELEMETRY:-1}" != "0" ]; then TELEMETRY_FLAG="--vllm-telemetry"; fi

echo "=============================================="
echo "MEMORY-PRESSURE SWEEP  model=$MODEL  Q=$NUM_QUERIES  trials=$NUM_TRIALS"
echo "mem-utils: $MEM_UTILS   corpus budget: $CORPUS_BUDGET tokens"
echo "output:    $OUTPUT_DIR"
# Manifest env passthrough: run_experiment.py reads CAGE_QUERY_MANIFEST from the
# environment (uniform-yardstick contract) -- surface it so provenance is visible.
if [ -n "${CAGE_QUERY_MANIFEST:-}" ]; then
    export CAGE_QUERY_MANIFEST
    echo "manifest:  $CAGE_QUERY_MANIFEST (uniform yardstick: every cell measures its query set)"
else
    echo "manifest:  NONE -- per-script seeded sampling (build one with scripts/1_setup/build_query_manifest.py)"
fi
echo "=============================================="

FAILED=()

cleanup() {
    ./scripts/2_serving/manage_vllm_server.sh stop >/dev/null 2>&1 || true
    # Chain the log-guard's EXIT handler (our trap replaced it): final results sync.
    type __lg_cleanup >/dev/null 2>&1 && __lg_cleanup
}
trap cleanup EXIT

cell_complete() {  # <cell_dir>
    local dir="$1" t
    for ((t = 1; t <= NUM_TRIALS; t++)); do
        [ -f "$dir/trial_${t}/metrics.json" ] || return 1
    done
    return 0
}

prepare_cell() {  # <cell> -> 0 run, 1 skip
    local cell="$1" dir="$OUTPUT_DIR/$1"
    if [ "${CAGE_FORCE_RERUN:-0}" = "1" ]; then
        [ -d "$dir" ] && echo "    FORCE RERUN: wiping $cell"
        rm -rf "$dir"; return 0
    fi
    if cell_complete "$dir"; then echo "SKIP (complete): $cell"; return 1; fi
    [ -d "$dir" ] && { echo "    PARTIAL: wiping incomplete $cell"; rm -rf "$dir"; }
    return 0
}

mark_failed() {  # <reason> <cell...>
    local reason="$1" c; shift
    for c in "$@"; do
        cell_complete "$OUTPUT_DIR/$c" && continue
        mkdir -p "$OUTPUT_DIR/$c"
        echo "STATUS=failed reason=$reason model=$MODEL $(date)" > "$OUTPUT_DIR/$c/STATUS"
        FAILED+=("$c($reason)")
    done
}

server_or_fail() {  # <desc> [restart args...]
    local desc="$1"; shift
    echo ""
    echo "=== $desc ==="
    if ! ./scripts/2_serving/manage_vllm_server.sh restart "$MODEL" "$@"; then
        return 1
    fi
    sleep 5
    return 0
}

# CALIBRATION GATE: read the ACTUAL kv_capacity_tokens of the just-booted server via
# cage-stats (src/monitoring/vllm_telemetry bridge), echo it, and persist it into every
# cell dir this boot will serve. An unreadable capacity is announced LOUDLY and recorded
# as null (never fabricated) -- the boot itself already passed server_or_fail.
calibrate() {  # <util> <cell...>
    local util="$1" cap c; shift
    cap=$(CAL_URL="http://localhost:${PORT}" python3 - <<'PY'
import contextlib
import io
import os
import sys

from src.monitoring.vllm_telemetry import capture_snapshot

# capture_snapshot prints its fallback diagnostics to STDOUT; quarantine them so the
# shell substitution receives ONLY the capacity value.
noise = io.StringIO()
with contextlib.redirect_stdout(noise):
    snap = capture_snapshot(os.environ["CAL_URL"]) or {}
if noise.getvalue().strip():
    print(noise.getvalue().strip(), file=sys.stderr)
cap = snap.get("kv_capacity_tokens")
if cap is None and isinstance(snap.get("kv"), dict):
    cap = snap["kv"].get("capacity_tokens")
print(cap if cap is not None else "null")
PY
    ) || cap="null"
    if [ -z "$cap" ] || [ "$cap" = "null" ]; then
        echo "!!! CALIBRATION: kv_capacity_tokens UNREADABLE at util=$util (cage-stats missing" \
             "or /metrics lacks cache_config_info) -- recording null, cells stay interpretable" \
             "only via telemetry_series.jsonl"
        cap="null"
    fi
    echo ">>> CALIBRATION util=$util kv_capacity_tokens=$cap corpus_budget=$CORPUS_BUDGET" \
         "(bracket: corpus block must fit for CAG to retain hits)"
    for c in "$@"; do
        mkdir -p "$OUTPUT_DIR/$c"
        cat > "$OUTPUT_DIR/$c/kv_capacity.json" <<EOF
{
  "gpu_memory_utilization": $util,
  "kv_capacity_tokens": $cap,
  "corpus_budget_tokens": $CORPUS_BUDGET,
  "model": "$MODEL",
  "utc_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    done
}

run_cell() {  # <cell> <baseline_type> [extra run_experiment args...]
    local cell="$1" baseline="$2"; shift 2
    echo ""
    echo ">>> [memory_sweep] $cell  ($(date))"
    if ! python3 scripts/3_run/run_experiment.py \
        --baseline "$baseline" \
        --baseline-label "$cell" \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --num-queries "$NUM_QUERIES" \
        --num-trials "$NUM_TRIALS" \
        --seed "$SEED" \
        --reset-cache-between-trials \
        --output-dir "$OUTPUT_DIR/$cell" \
        $TELEMETRY_FLAG \
        "$@"; then
        mkdir -p "$OUTPUT_DIR/$cell"
        echo "STATUS=failed reason=run_experiment model=$MODEL $(date)" > "$OUTPUT_DIR/$cell/STATUS"
        FAILED+=("$cell(run)")
    fi
}

for UTIL in $MEM_UTILS; do
    # THE swept variable: _serving_config.sh's VLLM_GPU_MEMORY_UTILIZATION, consumed by
    # manage_vllm_server.sh as --gpu-memory-utilization on the next restart. Everything
    # else in the serving regime (eager, max_model_len) stays at the uniform values.
    export VLLM_GPU_MEMORY_UTILIZATION="$UTIL"
    echo ""
    echo "##############################################"
    echo "## MEMORY UTIL $UTIL  ($(date))"
    echo "##############################################"

    C_NOCACHE="util_${UTIL}_no_cache"
    C_PREFIX="util_${UTIL}_prefix_cache"
    C_CAG="util_${UTIL}_cag_true_on"
    C_LMCACHE="util_${UTIL}_lmcache_rag"

    # --- [1/3] prefix caching OFF: the recompute floor --------------------------------
    # Order per cell is prepare (may wipe a partial dir) -> calibrate (writes
    # kv_capacity.json into the fresh dir) -> run; calibrating first would be wiped.
    if cell_complete "$OUTPUT_DIR/$C_NOCACHE" && [ "${CAGE_FORCE_RERUN:-0}" != "1" ]; then
        echo "SKIP (complete): $C_NOCACHE"
    elif server_or_fail "[util $UTIL 1/3] server WITHOUT prefix caching" --no-prefix-cache; then
        if prepare_cell "$C_NOCACHE"; then
            calibrate "$UTIL" "$C_NOCACHE"
            run_cell "$C_NOCACHE" no_cache
        fi
    else
        mark_failed server "$C_NOCACHE"
    fi

    # --- [2/3] prefix caching ON: eviction-vs-retention arms --------------------------
    on_complete=1
    for c in "$C_PREFIX" "$C_CAG"; do cell_complete "$OUTPUT_DIR/$c" || on_complete=0; done
    if [ "$on_complete" = "1" ] && [ "${CAGE_FORCE_RERUN:-0}" != "1" ]; then
        echo "SKIP (complete): $C_PREFIX"; echo "SKIP (complete): $C_CAG"
    elif server_or_fail "[util $UTIL 2/3] server WITH prefix caching"; then
        if prepare_cell "$C_PREFIX"; then
            calibrate "$UTIL" "$C_PREFIX"
            run_cell "$C_PREFIX" prefix_cache
        fi
        # Corpus-as-prefix flags copied from run_prefix_envelope.sh (cag_true_on): the
        # corpus block (~CORPUS_BUDGET tokens) is what the capacity bracket squeezes.
        if prepare_cell "$C_CAG"; then
            calibrate "$UTIL" "$C_CAG"
            run_cell "$C_CAG" prefix_cache --corpus-prefix-budget "$CORPUS_BUDGET"
        fi
    else
        mark_failed server "$C_PREFIX" "$C_CAG"
    fi

    # --- [3/3] LMCache connector: CPU-offload retention under the same pressure -------
    if [ "${CAGE_ENABLE_LMCACHE:-0}" != "1" ]; then
        if cell_complete "$OUTPUT_DIR/$C_LMCACHE"; then
            echo "SKIP (complete): $C_LMCACHE"
        else
            mkdir -p "$OUTPUT_DIR/$C_LMCACHE"
            echo "STATUS=skipped reason=lmcache_disabled model=$MODEL $(date)" > "$OUTPUT_DIR/$C_LMCACHE/STATUS"
            echo "SKIP (CAGE_ENABLE_LMCACHE!=1): $C_LMCACHE"
        fi
    elif cell_complete "$OUTPUT_DIR/$C_LMCACHE" && [ "${CAGE_FORCE_RERUN:-0}" != "1" ]; then
        echo "SKIP (complete): $C_LMCACHE"
    elif ! python3 -c "import lmcache" >/dev/null 2>&1; then
        mark_failed lmcache_missing "$C_LMCACHE"
        echo "!!! lmcache not importable -- pip install lmcache \"transformers>=4.36,<5\" (see run_kv_store.sh)"
    else
        # Connector flags copied from run_kv_store.sh: LAUNCH-time --kv-transfer-config
        # JSON. Explicit export/unset (not an env-prefix on the shell function) so the
        # connector can NEVER leak into the next util's connectorless restarts.
        export VLLM_KV_TRANSFER_CONFIG='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
        if server_or_fail "[util $UTIL 3/3] server WITH LMCache connector"; then
            if prepare_cell "$C_LMCACHE"; then
                calibrate "$UTIL" "$C_LMCACHE"
                run_cell "$C_LMCACHE" rag
            fi
        else
            mark_failed server_with_connector "$C_LMCACHE"
        fi
        unset VLLM_KV_TRANSFER_CONFIG
    fi
done

echo ""
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "MEMORY_SWEEP_DONE_WITH_FAILURES: ${FAILED[*]}"
    exit 1
fi
echo "MEMORY_SWEEP_DONE (all cells complete or skipped)"
