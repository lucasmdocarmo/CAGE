#!/bin/bash
# =============================================================================
# CAGE core baseline suite (single-node GPU = phase2). Runs the core baselines for ONE
# model (no_cache, rag, redis, prefix_cache, hybrid cold/warm; +distributed when
# ENABLE_DISTRIBUTED=1). Phase-neutral driver: cloud_run.sh runs it as phase2 by default.
# Outputs go under the run root: results/<phase>/<run-id>/baselines/.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Uniform serving config across ALL trees (Option A): non-eager / max_len 4096 / mem-util 0.90,
# so the core baselines are served under the SAME regime as the compression + speculative
# trees and cross-mechanism comparisons are fair. (Previously the core suite ran max_len 8192 /
# mem-util 0.92, which confounded cross-tree serving deltas.) mem-util is the swept axis.
source "$SCRIPT_DIR/../lib/_serving_config.sh"

MODEL=${1:-"Qwen/Qwen3-8B"}
DATASET="squad_v2"
# Model tag appended to baseline labels (see run_baseline) so MiMo core+compression never
# collides with Qwen's result dirs. Qwen (primary) stays bare.
case "$MODEL" in *MiMo*|*mimo*) MTAG="_mimo7b" ;; *) MTAG="" ;; esac
# Plan-of-record default is 500x3 (was 50 -- a legacy CPU value that silently under-ran a
# DIRECT invocation; cloud_run.sh always overrides both).
NUM_QUERIES=${NUM_QUERIES:-500}
NUM_TRIALS=${NUM_TRIALS:-3}

# Outputs land under the run root minted by cloud_run.sh (CAGE_RUN_ROOT/baselines). A DIRECT
# invocation with no run root self-mints the SAME date_HHMM_model_NxT run-id so results still
# land in the standardized results/<phase>/<run-id>/ tree and never pollute the legacy analysis/.
if [ -n "${CAGE_RUN_ROOT:-}" ]; then
    OUTPUT_DIR="$CAGE_RUN_ROOT/baselines"
else
    _phase="${CAGE_PHASE:-phase2}"
    _slug="$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]' | sed -E 's|.*/||; s|[^a-z0-9]+|-|g; s|^-+||; s|-+$||')"
    _rid="${CAGE_RUN_ID:-$(date +%Y-%m-%d_%H%M)_${_slug}_${NUM_QUERIES}x${NUM_TRIALS}}"
    OUTPUT_DIR="$PROJECT_DIR/results/$_phase/$_rid/baselines"
fi
SEED=${SEED:-42}
VLLM_PORT=${VLLM_PORT:-8000}
CLUSTER_BASE_PORT=${CLUSTER_BASE_PORT:-8001}
ROUTER_REPLICAS_COUNT=${ROUTER_REPLICAS_COUNT:-3}
ROUTER_PORT=${ROUTER_PORT:-9000}
# Default OFF on a single L4: the distributed (3-replica router) family is a Phase-3
# baseline that OOMs a 24GB L4 (~3x VRAM). cloud_run.sh sets this explicitly; a DIRECT
# `bash scripts/3_run/run_baselines.sh ...` must not silently launch it. Opt in with ENABLE_DISTRIBUTED=1.
ENABLE_DISTRIBUTED=${ENABLE_DISTRIBUTED:-0}
# vLLM telemetry via cage-stats: VLLM_TELEMETRY=1 captures a /metrics snapshot
# (spec-decode acceptance, KV-compression, token-source, GPU) into each baseline's
# results + prints a dashboard. Auto-enabled by cloud_run.sh.
TELEMETRY_FLAG=""
if [ "${VLLM_TELEMETRY:-0}" != "0" ]; then TELEMETRY_FLAG="--vllm-telemetry"; fi
_cleanup_ran=0
echo "=============================================="
echo "CAGE core baseline suite: $MODEL on $DATASET"
echo "=============================================="

cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/logs" "$OUTPUT_DIR"

# Activate the project venv regardless of its name (.venv locally, cage-env on the GPU VM).
# Without this a fresh shell / automation falls back to system python -> ImportError on
# vllm/llmlingua/cage-stats for every baseline.
for _v in .venv cage-env ../cage-env; do
    if [ -f "$_v/bin/activate" ]; then
        echo "Activating virtual environment: $_v"
        # shellcheck disable=SC1090
        source "$_v/bin/activate"
        break
    fi
done

# Per-baseline cleanup is model-scoped inside run_baseline (removes only THIS model's own
# baseline dirs), so a second model (MiMo) run through the same suite coexists under the run
# root's baselines/ and never wipes the other model's already-collected core arms.
# Do NOT blanket-wipe the shared parent here.
echo "Core suite results dir: $OUTPUT_DIR (per-baseline dirs cleaned model-scoped in run_baseline)"

cleanup() {
    if [ "$_cleanup_ran" -eq 1 ]; then
        return
    fi
    _cleanup_ran=1
    python3 scripts/2_serving/manage_vllm_cluster.py stop >/dev/null 2>&1 || true
    ./scripts/2_serving/manage_vllm_server.sh stop >/dev/null 2>&1 || true
}

trap cleanup EXIT

redis_prefix_for() {
    printf "cage:%s:%s" "$DATASET" "$1"
}

start_server_without_prefix_cache() {
    echo "[1/4] Starting Server WITHOUT Prefix Caching..."
    ./scripts/2_serving/manage_vllm_server.sh restart "$MODEL" --no-prefix-cache
    echo "Waiting 10 seconds for stability..."
    sleep 10
}

start_server_with_prefix_cache() {
    echo "$1"
    ./scripts/2_serving/manage_vllm_server.sh restart "$MODEL"
    echo "Waiting 10 seconds for stability..."
    sleep 10
}

run_baseline() {
    local baseline=$1
    # Append the model tag so a second model (e.g. MiMo) run through this same suite lands in
    # its own dirs and never overwrites Qwen's. Qwen (primary) stays bare (MTAG="").
    local baseline_label="${2}${MTAG:-}"
    shift 2
    echo ""
    echo ">>> Running baseline: $baseline_label"
    echo "    Started at: $(date)"

    # Model-scoped clean: remove ONLY this baseline's own dir so re-running a model refreshes
    # its arms without wiping the OTHER model's already-collected core results.
    rm -rf "$OUTPUT_DIR/$baseline_label"

    python3 scripts/3_run/run_experiment.py \
        --baseline "$baseline" \
        --baseline-label "$baseline_label" \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --num-queries "$NUM_QUERIES" \
        --num-trials "$NUM_TRIALS" \
        --seed "$SEED" \
        --output-dir "$OUTPUT_DIR/$baseline_label" \
        $TELEMETRY_FLAG \
        "$@"
    echo "    Finished at: $(date)"
    echo "    Results saved to: $OUTPUT_DIR/$baseline_label"
}

run_distributed_variant() {
    local baseline_label=$1
    local policy=$2

    echo ""
    echo ">>> Running distributed variant: $baseline_label"
    echo "    Started at: $(date)"

    CAGE_REQUIRE_DISTINCT_REPLICAS=1 python3 scripts/3_run/run_experiment.py \
        --baseline distributed \
        --baseline-label "$baseline_label" \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --num-queries "$NUM_QUERIES" \
        --num-trials "$NUM_TRIALS" \
        --seed "$SEED" \
        --api-base "http://localhost:${ROUTER_PORT}" \
        --sharding-policy "$policy" \
        --output-dir "$OUTPUT_DIR/$baseline_label" \
        $TELEMETRY_FLAG

    echo "    Finished at: $(date)"
    echo "    Results saved to: $OUTPUT_DIR/$baseline_label"
}

# 1. No Cache, RAG, and Redis retrieval-cache cold baseline
start_server_without_prefix_cache

run_baseline "no_cache" "no_cache"
run_baseline "rag" "rag"
run_baseline "redis" "redis_retrieval_cache_cold" \
    --flush-redis-namespace \
    --redis-key-prefix "$(redis_prefix_for redis_retrieval_cache_cold)"

# 2. Native prefix-cache baseline
start_server_with_prefix_cache "[2/4] Starting Server WITH Prefix Caching..."

run_baseline "prefix_cache" "prefix_cache" --reset-cache-between-trials

# 3. Hybrid cold baseline: empty retrieval cache + empty prefix cache
start_server_with_prefix_cache "[3/4] Restarting Server WITH Prefix Caching for hybrid cold..."
run_baseline "hybrid" "hybrid_retrieval_cache_cold" \
    --reset-cache-between-trials \
    --flush-redis-namespace \
    --redis-key-prefix "$(redis_prefix_for hybrid_retrieval_cache_cold)"

# 4. Hybrid warm baseline: explicit warmup excluded from measured metrics
start_server_with_prefix_cache "[4/4] Restarting Server WITH Prefix Caching for hybrid warm..."
run_baseline "hybrid" "hybrid_retrieval_cache_warm" \
    --reset-cache-between-trials \
    --flush-redis-namespace \
    --redis-key-prefix "$(redis_prefix_for hybrid_retrieval_cache_warm)" \
    --warmup-queries "$NUM_QUERIES"

if [ "$ENABLE_DISTRIBUTED" != "0" ]; then
    # 5. Distributed replicated router baseline (no simulated sharded core variant)
    echo "[5/5] Starting isolated distributed cluster..."
    ./scripts/2_serving/manage_vllm_server.sh stop
    python3 scripts/2_serving/manage_vllm_cluster.py restart \
        --model "$MODEL" \
        --replicas "$ROUTER_REPLICAS_COUNT" \
        --base-port "$CLUSTER_BASE_PORT" \
        --router-port "$ROUTER_PORT"
    run_distributed_variant "distributed_router_replicated" "replicated"
fi

# Cleanup
echo "Shutting down infrastructure..."
cleanup
trap - EXIT

echo ""
echo "=============================================="
echo "Core baseline suite complete."
echo "Results in: $OUTPUT_DIR"
echo "=============================================="
