#!/bin/bash
# =============================================================================
# Phase 4: Cross-Dataset Evaluation (TriviaQA, HotpotQA)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PHASE_DIR="$PROJECT_DIR/analysis/phase4"
OUTPUT_DIR="$PHASE_DIR/results"

MODEL=${1:-"Qwen/Qwen3-4B"}
NUM_QUERIES=${NUM_QUERIES:-500}
NUM_TRIALS=${NUM_TRIALS:-3}
SEED=${SEED:-42}
VLLM_PORT=${VLLM_PORT:-8000}
CLUSTER_BASE_PORT=${CLUSTER_BASE_PORT:-8001}
ROUTER_REPLICAS_COUNT=${ROUTER_REPLICAS_COUNT:-3}
ROUTER_PORT=${ROUTER_PORT:-9000}
ENABLE_DISTRIBUTED_PHASE4=${ENABLE_DISTRIBUTED_PHASE4:-0}
_cleanup_ran=0

DATASETS=(
    "trivia_qa"
    "hotpotqa"
)

echo "=============================================="
echo "CAGE Phase 4: Cross-Dataset Evaluation"
echo "=============================================="
echo "Model: $MODEL"
echo "Output directory: $OUTPUT_DIR"
echo "Datasets: ${DATASETS[*]}"
echo "Queries per baseline: $NUM_QUERIES"
echo "Trials per baseline: $NUM_TRIALS"
echo "Optional distributed suite: $ENABLE_DISTRIBUTED_PHASE4"
echo "=============================================="

cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/logs" "$OUTPUT_DIR"

if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

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
    local dataset=$1
    local baseline_label=$2
    printf "phase4:%s:%s" "$dataset" "$baseline_label"
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
    local dataset=$1
    local baseline=$2
    local baseline_label=$3
    shift 3

    echo ""
    echo ">>> Running baseline: $baseline_label on $dataset"
    echo "    Started at: $(date)"

    python3 scripts/run_experiment.py \
        --baseline "$baseline" \
        --baseline-label "$baseline_label" \
        --model "$MODEL" \
        --dataset "$dataset" \
        --num-queries "$NUM_QUERIES" \
        --num-trials "$NUM_TRIALS" \
        --seed "$SEED" \
        --output-dir "$OUTPUT_DIR/${dataset}/$baseline_label" \
        "$@"

    echo "    Finished at: $(date)"
    echo "    Results saved to: $OUTPUT_DIR/${dataset}/$baseline_label"
}

run_distributed_variant() {
    local dataset=$1
    local baseline_label=$2
    local policy=$3

    echo ""
    echo ">>> Running distributed variant: $baseline_label on $dataset"
    echo "    Started at: $(date)"

    CAGE_REQUIRE_DISTINCT_REPLICAS=1 python3 scripts/run_experiment.py \
        --baseline distributed \
        --baseline-label "$baseline_label" \
        --model "$MODEL" \
        --dataset "$dataset" \
        --num-queries "$NUM_QUERIES" \
        --num-trials "$NUM_TRIALS" \
        --seed "$SEED" \
        --api-base "http://localhost:${ROUTER_PORT}" \
        --sharding-policy "$policy" \
        --output-dir "$OUTPUT_DIR/${dataset}/$baseline_label"

    echo "    Finished at: $(date)"
    echo "    Results saved to: $OUTPUT_DIR/${dataset}/$baseline_label"
}

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Dataset: $dataset"
    echo "=========================================="

    rm -rf "$OUTPUT_DIR/${dataset}"/* || true

    start_server_without_prefix_cache

    run_baseline "$dataset" "no_cache" "no_cache"
    run_baseline "$dataset" "rag" "rag"
    run_baseline "$dataset" "redis" "redis_retrieval_cache_cold" \
        --flush-redis-namespace \
        --redis-key-prefix "$(redis_prefix_for "$dataset" redis_retrieval_cache_cold)"

    start_server_with_prefix_cache "[2/4] Starting Server WITH Prefix Caching..."

    run_baseline "$dataset" "prefix_cache" "prefix_cache"

    start_server_with_prefix_cache "[3/4] Restarting Server WITH Prefix Caching for hybrid cold..."
    run_baseline "$dataset" "hybrid" "hybrid_retrieval_cache_cold" \
        --flush-redis-namespace \
        --redis-key-prefix "$(redis_prefix_for "$dataset" hybrid_retrieval_cache_cold)"

    start_server_with_prefix_cache "[4/4] Restarting Server WITH Prefix Caching for hybrid warm..."
    run_baseline "$dataset" "hybrid" "hybrid_retrieval_cache_warm" \
        --flush-redis-namespace \
        --redis-key-prefix "$(redis_prefix_for "$dataset" hybrid_retrieval_cache_warm)" \
        --warmup-queries "$NUM_QUERIES"

    if [ "$ENABLE_DISTRIBUTED_PHASE4" != "0" ]; then
        echo "[5/5] Starting isolated distributed cluster..."
        ./scripts/manage_vllm_server.sh stop
        python3 scripts/manage_vllm_cluster.py restart \
            --model "$MODEL" \
            --replicas "$ROUTER_REPLICAS_COUNT" \
            --base-port "$CLUSTER_BASE_PORT" \
            --router-port "$ROUTER_PORT"
        run_distributed_variant "$dataset" "distributed_router_replicated" "replicated"
        python3 scripts/manage_vllm_cluster.py stop
    fi

    ./scripts/manage_vllm_server.sh stop
done

cleanup
trap - EXIT

echo ""
echo "=============================================="
echo "Phase 4 Complete!"
echo "Results in: $OUTPUT_DIR"
echo "=============================================="
