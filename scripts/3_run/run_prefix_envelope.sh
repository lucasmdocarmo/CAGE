#!/bin/bash
# Prefix-cache WORKLOAD-ENVELOPE cells + the true-CAG baseline (2026-07-15, tasks #71/#82).
#
# Why: with workload_mode=single on SQuAD every query carries a private paragraph, so the
# prefix cache can only reuse the ~32-token system prefix (measured: 10.9% cached tokens,
# TTFT -3.3%). These cells characterize the envelope instead of one point:
#
#   cag_true_off             corpus-as-prefix prompts, prefix caching OFF  (re-prefill bound)
#   cag_true_on              corpus-as-prefix prompts, prefix caching ON   (true CAG,
#                            Chan et al. arXiv 2412.15605: corpus KV computed once, reused)
#   prefix_cache_grouped     same-paragraph questions consecutive (shared-document workload)
#   prefix_cache_multiturn   growing conversation history (realistic shared prefix)
#   prefix_cache_repeat      repeated identical queries (ORACLE upper bound -- label it so)
#
# cag_true cells answer only the IN-CORPUS subset: build_corpus_block packs gold paragraphs
# up to CORPUS_BUDGET tokens (~10-15 SQuAD paragraphs at 2800), and questions whose
# paragraph did not fit are dropped (announced by run_experiment). Size NUM_QUERIES with
# that in mind. Paired mechanism read: cag_true_on vs cag_true_off (identical prompts,
# cache is the ONLY delta). Do NOT use workload_mode=batched for TTFT (non-streaming path
# reports TTFT==latency -- vllm_adapter.py batch path).
#
# Outputs: $CAGE_RUN_ROOT/envelope/<cell>/trial_*/   (tree "envelope" in _results_loader)
# Resume: complete cells are skipped; CAGE_FORCE_RERUN=1 wipes and re-runs.
# Usage: [NUM_QUERIES=100 NUM_TRIALS=3 CORPUS_BUDGET=2800] bash scripts/3_run/run_prefix_envelope.sh [model]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
source scripts/lib/_serving_config.sh

MODEL=${1:-"Qwen/Qwen3-8B"}
DATASET="${DATASET:-squad_v2}"
NUM_QUERIES=${NUM_QUERIES:-500}
NUM_TRIALS=${NUM_TRIALS:-3}
SEED=${SEED:-42}
CORPUS_BUDGET=${CORPUS_BUDGET:-2800}
REPEAT_QUERIES=${REPEAT_QUERIES:-3}
OUTPUT_DIR="${CAGE_RUN_ROOT:-results/phase2/local}/envelope"
mkdir -p "$OUTPUT_DIR"

TELEMETRY_FLAG=""
if [ "${VLLM_TELEMETRY:-0}" != "0" ]; then TELEMETRY_FLAG="--vllm-telemetry"; fi

FAILED=()

cleanup() { ./scripts/2_serving/manage_vllm_server.sh stop >/dev/null 2>&1 || true; }
trap cleanup EXIT

cell_complete() {  # <cell_dir>
    local dir="$1" t
    for ((t = 1; t <= NUM_TRIALS; t++)); do
        [ -f "$dir/trial_${t}/metrics.json" ] || return 1
    done
    return 0
}

prepare_cell() {  # <label> -> 0 run, 1 skip
    local label="$1" dir="$OUTPUT_DIR/$1"
    if [ "${CAGE_FORCE_RERUN:-0}" = "1" ]; then
        [ -d "$dir" ] && echo "    FORCE RERUN: wiping $label"
        rm -rf "$dir"; return 0
    fi
    if cell_complete "$dir"; then echo "SKIP (complete): $label"; return 1; fi
    [ -d "$dir" ] && { echo "    PARTIAL: wiping incomplete $label"; rm -rf "$dir"; }
    return 0
}

run_cell() {  # <label> <baseline_type> [extra run_experiment args...]
    local label="$1" baseline="$2"; shift 2
    echo ""
    echo ">>> [envelope] $label  ($(date))"
    prepare_cell "$label" || return 0
    if ! python3 scripts/3_run/run_experiment.py \
        --baseline "$baseline" \
        --baseline-label "$label" \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --num-queries "$NUM_QUERIES" \
        --num-trials "$NUM_TRIALS" \
        --seed "$SEED" \
        --output-dir "$OUTPUT_DIR/$label" \
        $TELEMETRY_FLAG \
        "$@"; then
        mkdir -p "$OUTPUT_DIR/$label"
        echo "STATUS=failed reason=run_experiment model=$MODEL $(date)" > "$OUTPUT_DIR/$label/STATUS"
        FAILED+=("$label(run)")
    fi
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

mark_failed() {  # <reason> <cell...>
    local reason="$1" c; shift
    for c in "$@"; do
        cell_complete "$OUTPUT_DIR/$c" && continue
        mkdir -p "$OUTPUT_DIR/$c"
        echo "STATUS=failed reason=$reason model=$MODEL $(date)" > "$OUTPUT_DIR/$c/STATUS"
        FAILED+=("$c($reason)")
    done
}

# --- [1/2] prefix caching OFF: the re-prefill counterfactual for true CAG -------------
if cell_complete "$OUTPUT_DIR/cag_true_off" && [ "${CAGE_FORCE_RERUN:-0}" != "1" ]; then
    echo "SKIP (complete): cag_true_off"
elif server_or_fail "[1/2] server WITHOUT prefix caching" --no-prefix-cache; then
    run_cell cag_true_off no_cache --corpus-prefix-budget "$CORPUS_BUDGET"
else
    mark_failed server cag_true_off
fi

# --- [2/2] prefix caching ON: true CAG + the workload envelope ------------------------
ON_CELLS=(cag_true_on prefix_cache_grouped prefix_cache_multiturn prefix_cache_repeat)
all_on_complete=1
for c in "${ON_CELLS[@]}"; do cell_complete "$OUTPUT_DIR/$c" || all_on_complete=0; done
if [ "$all_on_complete" = "1" ] && [ "${CAGE_FORCE_RERUN:-0}" != "1" ]; then
    for c in "${ON_CELLS[@]}"; do echo "SKIP (complete): $c"; done
elif server_or_fail "[2/2] server WITH prefix caching"; then
    # True CAG: corpus KV computed once (first request), reused by every later query.
    # --reset-cache-between-trials: audit 2026-07-16 M1 -- without it, trial 1 measured a
    # cold corpus build inside the measured window while trials 2/3 started warm (a
    # different condition than every comparator, which reset). With the reset, all three
    # trials measure the SAME cold-build->reuse trajectory; within-trial persistence (the
    # CAG mechanism) is untouched.
    # TODO(audit-2026-07-16 M1): additionally issue a DISCARDED corpus-preload request
    # after each reset so all trials measure the warm-corpus condition instead. The
    # runner has no such hook today: the corpus block is composed inside
    # run_experiment.py's run_experiment() (src/data/corpus.build_corpus_block /
    # manifest blocks), not at the reset site in _run_trials, and --warmup-queries
    # replays the FULL measured set (cache priming, distorts the cell). Needs a small
    # runner feature (e.g. --corpus-preload sending one max_tokens=1 request with the
    # corpus prompt after each reset) -- not half-implemented here.
    run_cell cag_true_on prefix_cache --corpus-prefix-budget "$CORPUS_BUDGET" --reset-cache-between-trials
    # Shared-document workload: reuse within same-paragraph groups. Reset between trials
    # so each trial measures the same cold->grouped-warm trajectory.
    run_cell prefix_cache_grouped prefix_cache --order-by-context --reset-cache-between-trials
    # Realistic shared-history workload (growing conversational prefix).
    run_cell prefix_cache_multiturn prefix_cache --workload-mode multi_turn --reset-cache-between-trials
    # ORACLE upper bound: identical prompts repeated -- the workload the pre-fix Phase-1
    # accidentally measured. Report as a bound, never as the general figure.
    run_cell prefix_cache_repeat prefix_cache --repeat-queries "$REPEAT_QUERIES" --reset-cache-between-trials
else
    mark_failed server "${ON_CELLS[@]}"
fi

echo ""
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "ENVELOPE_DONE_WITH_FAILURES: ${FAILED[*]}"
    exit 1
fi
echo "ENVELOPE_DONE (all cells complete or skipped)"
