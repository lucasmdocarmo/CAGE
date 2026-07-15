#!/bin/bash
# =============================================================================
# CAGE core baseline suite (single-node GPU = phase2). Runs the core baselines for ONE
# model (no_cache, rag, redis, prefix_cache, hybrid cold/warm; +distributed when
# ENABLE_DISTRIBUTED=1). Phase-neutral driver: cloud_run.sh runs it as phase2 by default.
# Outputs go under the run root: results/<phase>/<run-id>/baselines/.
# =============================================================================

# No -e: one failed cell must NOT abort the whole multi-hour suite. Each cell is wrapped in
# fault-tolerant helpers below (STATUS sentinel + FAILED summary + skip-completed resume),
# mirroring run_speculative_matrix.sh's sentinel pattern.
set -uo pipefail

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

# ---------------------------------------------------------------------------
# Fault tolerance + resume (mirrors run_speculative_matrix.sh's sentinel pattern):
#   - a cell is COMPLETE when trial_1..NUM_TRIALS/metrics.json all exist; complete cells
#     are SKIPPED on re-run (resume) unless CAGE_FORCE_RERUN=1;
#   - a failed cell writes a STATUS sentinel and is recorded in FAILED instead of aborting;
#   - a failed server restart marks its DEPENDENT cells failed and the suite continues to
#     the next server config;
#   - the suite exits nonzero at the END if any cell failed (after attempting all).
# ---------------------------------------------------------------------------
FAILED=()

cell_complete() {  # <cell_dir> -> 0 iff trial_1..NUM_TRIALS all have metrics.json
    local dir="$1" t
    for ((t = 1; t <= NUM_TRIALS; t++)); do
        [ -f "$dir/trial_${t}/metrics.json" ] || return 1
    done
    return 0
}

prepare_cell() {  # <full label> -> 0 = run it (stale dir wiped), 1 = skip (already complete)
    local label="$1" dir="$OUTPUT_DIR/$1"
    if [ "${CAGE_FORCE_RERUN:-0}" = "1" ]; then
        [ -d "$dir" ] && echo "    FORCE RERUN (CAGE_FORCE_RERUN=1): wiping $label"
        rm -rf "$dir"
        return 0
    fi
    if cell_complete "$dir"; then
        echo "SKIP (complete): $label"
        return 1
    fi
    if [ -d "$dir" ]; then
        echo "    PARTIAL: wiping incomplete $label and re-running"
        rm -rf "$dir"
    fi
    return 0
}

group_complete() {  # <bare-label...> -> 0 iff every cell is complete (server start unnecessary)
    [ "${CAGE_FORCE_RERUN:-0}" = "1" ] && return 1
    local lbl
    for lbl in "$@"; do
        cell_complete "$OUTPUT_DIR/${lbl}${MTAG:-}" || return 1
    done
    return 0
}

skip_group() {  # <bare-label...> announce each already-complete cell
    local lbl
    for lbl in "$@"; do echo "SKIP (complete): ${lbl}${MTAG:-}"; done
}

mark_cells_failed() {  # <reason> <bare-label...> sentinel the cells a dead server orphaned
    local reason="$1" lbl full; shift
    for lbl in "$@"; do
        full="${lbl}${MTAG:-}"
        # never clobber a cell already complete from a previous (resumed) run
        cell_complete "$OUTPUT_DIR/$full" && continue
        mkdir -p "$OUTPUT_DIR/$full"
        echo "STATUS=failed reason=$reason model=$MODEL $(date)" > "$OUTPUT_DIR/$full/STATUS"
        FAILED+=("$full($reason)")
    done
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

    # Skip-completed resume, else model-scoped clean: remove ONLY this baseline's own dir so
    # re-running a model refreshes its arms without wiping the OTHER model's core results.
    prepare_cell "$baseline_label" || return 0

    if ! python3 scripts/3_run/run_experiment.py \
        --baseline "$baseline" \
        --baseline-label "$baseline_label" \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --num-queries "$NUM_QUERIES" \
        --num-trials "$NUM_TRIALS" \
        --seed "$SEED" \
        --output-dir "$OUTPUT_DIR/$baseline_label" \
        $TELEMETRY_FLAG \
        "$@"; then
        echo "    CELL $baseline_label RUN-FAIL"
        mkdir -p "$OUTPUT_DIR/$baseline_label"
        echo "STATUS=failed reason=run model=$MODEL baseline=$baseline $(date)" > "$OUTPUT_DIR/$baseline_label/STATUS"
        FAILED+=("$baseline_label(run)")
        return 0
    fi
    echo "    Finished at: $(date)"
    echo "    Results saved to: $OUTPUT_DIR/$baseline_label"
}

run_distributed_variant() {
    local baseline_label=$1
    local policy=$2

    echo ""
    echo ">>> Running distributed variant: $baseline_label"
    echo "    Started at: $(date)"

    # NOTE: distributed labels are used verbatim (no MTAG), matching prior behavior.
    prepare_cell "$baseline_label" || return 0

    if ! CAGE_REQUIRE_DISTINCT_REPLICAS=1 python3 scripts/3_run/run_experiment.py \
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
        $TELEMETRY_FLAG; then
        echo "    CELL $baseline_label RUN-FAIL"
        mkdir -p "$OUTPUT_DIR/$baseline_label"
        echo "STATUS=failed reason=run model=$MODEL baseline=distributed $(date)" > "$OUTPUT_DIR/$baseline_label/STATUS"
        FAILED+=("$baseline_label(run)")
        return 0
    fi

    echo "    Finished at: $(date)"
    echo "    Results saved to: $OUTPUT_DIR/$baseline_label"
}

# 1. No Cache, RAG, and Redis retrieval-cache cold baseline
if group_complete no_cache rag redis_retrieval_cache_cold; then
    echo "[1/4] all cells complete -- skipping server start"
    skip_group no_cache rag redis_retrieval_cache_cold
elif start_server_without_prefix_cache; then
    run_baseline "no_cache" "no_cache"
    run_baseline "rag" "rag"
    run_baseline "redis" "redis_retrieval_cache_cold" \
        --flush-redis-namespace \
        --redis-key-prefix "$(redis_prefix_for redis_retrieval_cache_cold)"
else
    echo "[1/4] SERVER-FAIL -> marking dependent cells failed and continuing"
    mark_cells_failed server no_cache rag redis_retrieval_cache_cold
fi

# 2. Native prefix-cache baseline
if group_complete prefix_cache; then
    echo "[2/4] all cells complete -- skipping server start"
    skip_group prefix_cache
elif start_server_with_prefix_cache "[2/4] Starting Server WITH Prefix Caching..."; then
    run_baseline "prefix_cache" "prefix_cache" --reset-cache-between-trials
else
    echo "[2/4] SERVER-FAIL -> marking dependent cells failed and continuing"
    mark_cells_failed server prefix_cache
fi

# 3. Hybrid cold baseline: empty retrieval cache + empty prefix cache
if group_complete hybrid_retrieval_cache_cold; then
    echo "[3/4] all cells complete -- skipping server start"
    skip_group hybrid_retrieval_cache_cold
elif start_server_with_prefix_cache "[3/4] Restarting Server WITH Prefix Caching for hybrid cold..."; then
    run_baseline "hybrid" "hybrid_retrieval_cache_cold" \
        --reset-cache-between-trials \
        --flush-redis-namespace \
        --redis-key-prefix "$(redis_prefix_for hybrid_retrieval_cache_cold)"
else
    echo "[3/4] SERVER-FAIL -> marking dependent cells failed and continuing"
    mark_cells_failed server hybrid_retrieval_cache_cold
fi

# 4. Hybrid warm baseline: explicit warmup excluded from measured metrics
if group_complete hybrid_retrieval_cache_warm; then
    echo "[4/4] all cells complete -- skipping server start"
    skip_group hybrid_retrieval_cache_warm
elif start_server_with_prefix_cache "[4/4] Restarting Server WITH Prefix Caching for hybrid warm..."; then
    run_baseline "hybrid" "hybrid_retrieval_cache_warm" \
        --reset-cache-between-trials \
        --flush-redis-namespace \
        --redis-key-prefix "$(redis_prefix_for hybrid_retrieval_cache_warm)" \
        --warmup-queries "$NUM_QUERIES"
else
    echo "[4/4] SERVER-FAIL -> marking dependent cells failed and continuing"
    mark_cells_failed server hybrid_retrieval_cache_warm
fi

if [ "$ENABLE_DISTRIBUTED" != "0" ]; then
    # 5. Distributed replicated router baseline (no simulated sharded core variant)
    # (label used verbatim -- no MTAG -- so the sentinel path below matches run_distributed_variant)
    if [ "${CAGE_FORCE_RERUN:-0}" != "1" ] && cell_complete "$OUTPUT_DIR/distributed_router_replicated"; then
        echo "[5/5] SKIP (complete): distributed_router_replicated"
    else
        echo "[5/5] Starting isolated distributed cluster..."
        ./scripts/2_serving/manage_vllm_server.sh stop || true
        if python3 scripts/2_serving/manage_vllm_cluster.py restart \
            --model "$MODEL" \
            --replicas "$ROUTER_REPLICAS_COUNT" \
            --base-port "$CLUSTER_BASE_PORT" \
            --router-port "$ROUTER_PORT"; then
            run_distributed_variant "distributed_router_replicated" "replicated"
        else
            echo "[5/5] CLUSTER-FAIL -> marking distributed cell failed and continuing"
            mkdir -p "$OUTPUT_DIR/distributed_router_replicated"
            echo "STATUS=failed reason=server model=$MODEL $(date)" > "$OUTPUT_DIR/distributed_router_replicated/STATUS"
            FAILED+=("distributed_router_replicated(server)")
        fi
    fi
fi

# Cleanup
echo "Shutting down infrastructure..."
cleanup
trap - EXIT

echo ""
echo "=============================================="
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "Core baseline suite INCOMPLETE: ${#FAILED[@]} cell(s) failed: ${FAILED[*]}"
    echo "Results in: $OUTPUT_DIR (failed cells carry a STATUS sentinel;"
    echo "re-run this script with the same CAGE_RUN_ID to resume -- complete cells are skipped)"
    echo "=============================================="
    exit 1
fi
echo "Core baseline suite complete."
echo "Results in: $OUTPUT_DIR"
echo "=============================================="
