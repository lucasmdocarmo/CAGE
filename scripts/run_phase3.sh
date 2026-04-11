#!/bin/bash
# =============================================================================
# Phase 3: Qwen2.5-7B-Instruct on SQuAD v2 - All Baselines (Strict)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PHASE_DIR="$PROJECT_DIR/analysis/phase3"
OUTPUT_DIR="$PHASE_DIR/results"

MODEL=${1:-"Qwen/Qwen2.5-7B-Instruct"}
DATASET="squad_v2"
NUM_QUERIES=${NUM_QUERIES:-500}
NUM_TRIALS=${NUM_TRIALS:-3}
SEED=${SEED:-42}
VLLM_PORT=${VLLM_PORT:-8000}
CLUSTER_BASE_PORT=${CLUSTER_BASE_PORT:-8001}
ROUTER_REPLICAS_COUNT=${ROUTER_REPLICAS_COUNT:-3}
ROUTER_PORT=${ROUTER_PORT:-9000}
ENABLE_DISTRIBUTED=${ENABLE_DISTRIBUTED:-1}
_cleanup_ran=0

echo "=============================================="
echo "CAGE Phase 3 Strict Benchmarking: $MODEL on $DATASET"
echo "=============================================="

cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/logs" "$OUTPUT_DIR"

if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

echo "Cleaning previous Phase 3 result artifacts to prevent contamination..."
rm -rf "$OUTPUT_DIR"/* || true

cleanup() {
    if [ "$_cleanup_ran" -eq 1 ]; then
        return
    fi
    _cleanup_ran=1
    python3 scripts/manage_vllm_cluster.py stop >/dev/null 2>&1 || true
    ./scripts/manage_vllm_server.sh stop >/dev/null 2>&1 || true
}

trap cleanup EXIT

redis_prefix_for() {
    printf "phase3:%s:%s" "$DATASET" "$1"
}

start_server_without_prefix_cache() {
    echo "[1/4] Starting Server WITHOUT Prefix Caching..."
    ./scripts/manage_vllm_server.sh restart "$MODEL" --no-prefix-cache
    echo "Waiting 10 seconds for stability..."
    sleep 10
}

start_server_with_prefix_cache() {
    echo "$1"
    ./scripts/manage_vllm_server.sh restart "$MODEL"
    echo "Waiting 10 seconds for stability..."
    sleep 10
}

run_baseline() {
    local baseline=$1
    local baseline_label=$2
    shift 2

    echo ""
    echo ">>> Running baseline: $baseline_label"
    echo "    Started at: $(date)"

    python3 scripts/run_experiment.py \
        --baseline "$baseline" \
        --baseline-label "$baseline_label" \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --num-queries "$NUM_QUERIES" \
        --num-trials "$NUM_TRIALS" \
        --seed "$SEED" \
        --output-dir "$OUTPUT_DIR/$baseline_label" \
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

    CAGE_REQUIRE_DISTINCT_REPLICAS=1 python3 scripts/run_experiment.py \
        --baseline distributed \
        --baseline-label "$baseline_label" \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --num-queries "$NUM_QUERIES" \
        --num-trials "$NUM_TRIALS" \
        --seed "$SEED" \
        --api-base "http://localhost:${ROUTER_PORT}" \
        --sharding-policy "$policy" \
        --output-dir "$OUTPUT_DIR/$baseline_label"

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

run_baseline "prefix_cache" "prefix_cache"

# 3. Hybrid cold baseline: empty retrieval cache + empty prefix cache
start_server_with_prefix_cache "[3/4] Restarting Server WITH Prefix Caching for hybrid cold..."
run_baseline "hybrid" "hybrid_retrieval_cache_cold" \
    --flush-redis-namespace \
    --redis-key-prefix "$(redis_prefix_for hybrid_retrieval_cache_cold)"

# 4. Hybrid warm baseline: explicit warmup excluded from measured metrics
start_server_with_prefix_cache "[4/4] Restarting Server WITH Prefix Caching for hybrid warm..."
run_baseline "hybrid" "hybrid_retrieval_cache_warm" \
    --flush-redis-namespace \
    --redis-key-prefix "$(redis_prefix_for hybrid_retrieval_cache_warm)" \
    --warmup-queries "$NUM_QUERIES"

if [ "$ENABLE_DISTRIBUTED" != "0" ]; then
    # 5. Distributed replicated router baseline (no simulated sharded core variant)
    echo "[5/5] Starting isolated distributed cluster..."
    ./scripts/manage_vllm_server.sh stop
    python3 scripts/manage_vllm_cluster.py restart \
        --model "$MODEL" \
        --replicas "$ROUTER_REPLICAS_COUNT" \
        --base-port "$CLUSTER_BASE_PORT" \
        --router-port "$ROUTER_PORT"
    run_distributed_variant "distributed_router_replicated" "replicated"
fi

echo "Shutting down infrastructure..."
cleanup
trap - EXIT

echo ""
echo "=============================================="
echo "Phase 3 Complete! Strict Execution Successful."
echo "Results in: $OUTPUT_DIR"
echo "=============================================="
