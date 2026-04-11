#!/bin/bash
# =============================================================================
# Phase 5: Speculative Decoding Evaluation
# =============================================================================
# Estimated time: ~1-2 hours
# Draft Model: Qwen/Qwen3-0.6B (small, fast)
# Target Model: Qwen/Qwen3-4B
# Dataset: squad_v2
# Queries: 500
# Trials: 3
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PHASE_DIR="$PROJECT_DIR/analysis/phase5"
OUTPUT_DIR="$PHASE_DIR/results"

# Speculative decoding uses draft + target model
DRAFT_MODEL="Qwen/Qwen3-0.6B"
TARGET_MODEL="Qwen/Qwen3-4B"
DATASET="squad_v2"
NUM_QUERIES=500
NUM_TRIALS=3
SEED=42

echo "=============================================="
echo "CAGE Phase 5: Speculative Decoding"
echo "=============================================="
echo "Draft model: $DRAFT_MODEL"
echo "Target model: $TARGET_MODEL"
echo "Dataset: $DATASET"
echo "Output directory: $OUTPUT_DIR"
echo "Queries: $NUM_QUERIES"
echo "Trials: $NUM_TRIALS"
echo "=============================================="

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR"

echo ""
echo ">>> Running speculative decoding baseline"
echo "    Started at: $(date)"

python3 scripts/run_experiment.py \
    --baseline "speculative" \
    --model "$TARGET_MODEL" \
    --speculative-model "$DRAFT_MODEL" \
    --dataset "$DATASET" \
    --num-queries "$NUM_QUERIES" \
    --num-trials "$NUM_TRIALS" \
    --seed "$SEED" \
    --output-dir "$OUTPUT_DIR/speculative"

echo "    Finished at: $(date)"
echo "    Results saved to: $OUTPUT_DIR/speculative"

echo ""
echo "=============================================="
echo "Phase 5 Complete!"
echo "Results in: $OUTPUT_DIR"
echo "=============================================="
