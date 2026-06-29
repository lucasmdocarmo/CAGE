#!/bin/bash
# Speculative-method x context-strategy 2x2 for CAGE Phase 2 (single L4).
#   {ngram, <model-native draft>} x {CAG=gold context, RAG=retrieved context}
# The model-native draft method is chosen by model:
#   Qwen3-8B      -> eagle3  (AngelSlim/Qwen3-8B_eagle3 draft head)
#   MiMo-7B-RL    -> mtp     (native multi-token-prediction head; method "mimo_mtp")
# Speculative decoding is OUTPUT-LOSSLESS, so quality is fixed (= the underlying CAG/RAG
# baseline); this 2x2 isolates the SERVING effect (acceptance / TTFT / throughput) of each
# speculative method under each context strategy -- an angle other serving frameworks don't
# cross with retrieval.
#
# Usage: bash scripts/run_speculative_matrix.sh [MODEL]
#   MODEL defaults to Qwen/Qwen3-8B. Run ONCE PER MODEL to cover both speculative families:
#     bash scripts/run_speculative_matrix.sh Qwen/Qwen3-8B
#     bash scripts/run_speculative_matrix.sh XiaomiMiMo/MiMo-7B-RL
#   Override counts for the full design:  NUM_QUERIES=300 NUM_TRIALS=3 bash ... <MODEL>
#
# IMPORTANT: the MiMo MTP method string ("mimo_mtp") MUST be live-validated on vLLM 0.11.0
# during the smoke run -- if vLLM rejects it, the server won't start and the cell is marked
# SERVER-FAIL (with a STATUS sentinel) instead of silently missing. Override the exact JSON
# via MIMO_MTP_CONFIG if the installed vLLM expects a different method name.
#
# max_model_len capped to 4096 and mem-util 0.90 so a draft head fits alongside the target
# on a 24GB L4.
set -uo pipefail
cd "$HOME/CAGE"
source cage-env/bin/activate
# Continuous log+results mirror to GCS + full collect on exit (no built-in sync loop).
source scripts/_log_guard.sh
OUT="$HOME/CAGE/analysis/speculative_matrix"
mkdir -p "$OUT"

MODEL="${1:-Qwen/Qwen3-8B}"
DATASET="${DATASET:-squad_v2}"
NUM_QUERIES="${NUM_QUERIES:-100}"
NUM_TRIALS="${NUM_TRIALS:-1}"
SEED="${SEED:-42}"

export VLLM_ENFORCE_EAGER=1
export VLLM_MAX_MODEL_LEN=4096
export VLLM_GPU_MEMORY_UTILIZATION=0.90

# The universal draft-free method + the model-native draft method (+ a short model tag so
# Qwen and MiMo cells never collide in $OUT or in the consolidated stats).
NGRAM='{"method":"ngram","num_speculative_tokens":5}'
case "$MODEL" in
  *MiMo*|*mimo*)
    MTAG="mimo7b"
    DRAFT_LABEL="mtp"
    if [ -n "${MIMO_MTP_CONFIG:-}" ]; then
      DRAFT="$MIMO_MTP_CONFIG"
    else
      DRAFT='{"method":"mimo_mtp","num_speculative_tokens":5}'
    fi
    ;;
  *)
    MTAG="qwen8b"
    DRAFT_LABEL="eagle3"
    DRAFT='{"method":"eagle3","model":"AngelSlim/Qwen3-8B_eagle3","num_speculative_tokens":5}'
    ;;
esac

echo "[matrix] model=$MODEL  tag=$MTAG  draft-method=$DRAFT_LABEL  Q=$NUM_QUERIES  trials=$NUM_TRIALS"

# Kill any prior speculative run + its server so the matrix runs clean.
echo "[matrix] clearing prior speculative run..."
pkill -f run_phase5.sh 2>/dev/null || true
pkill -f "baseline speculative" 2>/dev/null || true
bash scripts/manage_vllm_server.sh stop >/dev/null 2>&1 || true
sleep 5

FAILED=()

run_cell() {  # <spec_json> <label> <context_source>
  local spec="$1" label="$2" ctx="$3"
  echo "=== CELL $label  (ctx=$ctx)  $(date) ==="
  # A failed launch / run writes a STATUS sentinel so a missing cell is LOUD in the stats
  # consolidation rather than silently absent (and the 2x2 reported as complete with a hole).
  if ! VLLM_SPECULATIVE_CONFIG="$spec" bash scripts/manage_vllm_server.sh restart "$MODEL"; then
    echo "CELL $label SERVER-FAIL"
    mkdir -p "$OUT/$label"
    echo "STATUS=failed reason=server model=$MODEL spec=$spec $(date)" > "$OUT/$label/STATUS"
    FAILED+=("$label(server)")
    return
  fi
  sleep 10
  if ! python3 scripts/run_experiment.py --baseline speculative --baseline-label "$label" \
      --model "$MODEL" --dataset "$DATASET" --num-queries "$NUM_QUERIES" --num-trials "$NUM_TRIALS" --seed "$SEED" \
      --context-source "$ctx" --vllm-telemetry --output-dir "$OUT/$label"; then
    echo "CELL $label RUN-FAIL"
    mkdir -p "$OUT/$label"
    echo "STATUS=failed reason=run model=$MODEL spec=$spec $(date)" > "$OUT/$label/STATUS"
    FAILED+=("$label(run)")
  fi
  bash scripts/sync_results_to_gcs.sh analysis || true
  echo "=== CELL $label DONE  $(date) ==="
}

run_cell "$NGRAM" "spec_${MTAG}_ngram_cag"          gold
run_cell "$NGRAM" "spec_${MTAG}_ngram_rag"          retrieved
run_cell "$DRAFT" "spec_${MTAG}_${DRAFT_LABEL}_cag" gold
run_cell "$DRAFT" "spec_${MTAG}_${DRAFT_LABEL}_rag" retrieved

bash scripts/sync_results_to_gcs.sh analysis || true
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "MATRIX INCOMPLETE: ${#FAILED[@]} cell(s) failed: ${FAILED[*]}"
fi
echo "SPECULATIVE_MATRIX_DONE (model=$MODEL)"
