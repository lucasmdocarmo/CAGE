#!/bin/bash
# =============================================================================
# Phase 5: Speculative Decoding Evaluation
# =============================================================================
# Speculative decoding is a vLLM LAUNCH-TIME setting (--speculative-config), so this
# script (re)starts the server WITH speculation, then runs the baseline against it.
# This is the same "launch-lever" pattern compressed_cag uses for --kv-cache-dtype fp8.
#
# It is GPU-meaningful: on a CPU backend the draft+target overhead usually cancels the
# benefit. Speculative decoding is output-distribution-preserving, so it does NOT change
# answer quality -- treat it as a serving-throughput (TPOT) baseline.
#
# Draft Model: Qwen/Qwen3-0.6B   Target: Qwen/Qwen3-4B   Dataset: squad_v2
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_DIR/analysis/phase5/results"

DRAFT_MODEL="${DRAFT_MODEL:-Qwen/Qwen3-0.6B}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-4B}"
DATASET="${DATASET:-squad_v2}"
NUM_QUERIES="${NUM_QUERIES:-500}"
NUM_TRIALS="${NUM_TRIALS:-3}"
SEED="${SEED:-42}"

# Launch-time speculative config (CURRENT vLLM API; the old --speculative-model is deprecated).
# draft_model method points at a small draft model. For a model-free run instead use:
#   export VLLM_SPECULATIVE_CONFIG='{"method":"ngram","num_speculative_tokens":5}'
export VLLM_SPECULATIVE_CONFIG="${VLLM_SPECULATIVE_CONFIG:-{\"model\":\"${DRAFT_MODEL}\",\"num_speculative_tokens\":5}}"

echo "=============================================="
echo "CAGE Phase 5: Speculative Decoding"
echo "=============================================="
echo "Target model:  $TARGET_MODEL"
echo "Spec config:   $VLLM_SPECULATIVE_CONFIG"
echo "Dataset:       $DATASET   Queries: $NUM_QUERIES   Trials: $NUM_TRIALS"
echo "Output:        $OUTPUT_DIR/speculative"
echo "=============================================="

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR"

# 1) Bring the server up WITH speculative decoding enabled (the launch-lever).
echo ">>> Restarting vLLM with speculative decoding enabled"
if ! VLLM_SPECULATIVE_CONFIG="$VLLM_SPECULATIVE_CONFIG" \
        "$SCRIPT_DIR/manage_vllm_server.sh" restart "$TARGET_MODEL"; then
    echo "ERROR: vLLM failed to start with speculative config: $VLLM_SPECULATIVE_CONFIG" >&2
    echo "       (likely an OOM on the draft head, or an unsupported method on this vLLM)." >&2
    echo "       Skipping the speculative arm; the log-guard EXIT trap will collect logs." >&2
    exit 1
fi

# 2) Run the speculative baseline, capturing the /metrics acceptance rate via telemetry.
echo ">>> Running speculative decoding baseline   ($(date))"
python3 scripts/run_experiment.py \
    --baseline "speculative" \
    --model "$TARGET_MODEL" \
    --dataset "$DATASET" \
    --num-queries "$NUM_QUERIES" \
    --num-trials "$NUM_TRIALS" \
    --seed "$SEED" \
    --vllm-telemetry \
    --output-dir "$OUTPUT_DIR/speculative"

echo ">>> Finished   ($(date))"
echo "    Results:   $OUTPUT_DIR/speculative"
echo "    Acceptance rate is in the vLLM telemetry snapshot:"
echo "      vllm:spec_decode_num_accepted_tokens_total / vllm:spec_decode_num_draft_tokens_total"
echo "=============================================="
