#!/bin/bash
# Speculative-method x context-strategy 2x2 for CAGE Phase 2 (single L4).
#   {ngram, eagle3} x {CAG=gold context, RAG=retrieved context}
# Speculative decoding is OUTPUT-LOSSLESS, so quality is fixed (= the underlying
# CAG/RAG baseline); this 2x2 isolates the SERVING effect (acceptance / TTFT /
# throughput) of each speculative method under each context strategy -- an angle
# other serving frameworks don't cross with retrieval. Qwen3-8B, 100q / 1 trial.
#
# max_model_len capped to 4096 and mem-util 0.90 so the EAGLE-3 draft head fits
# alongside the 8B target on a 24GB L4.
set -uo pipefail
cd "$HOME/CAGE"
source cage-env/bin/activate
# Continuous log+results mirror to GCS + full collect on exit (no built-in sync loop).
source scripts/_log_guard.sh
OUT="$HOME/CAGE/analysis/speculative_matrix"
mkdir -p "$OUT"

export VLLM_ENFORCE_EAGER=1
export VLLM_MAX_MODEL_LEN=4096
export VLLM_GPU_MEMORY_UTILIZATION=0.90

# Kill the redundant single-ngram run + its server so the matrix runs clean.
echo "[matrix] clearing prior speculative run..."
pkill -f run_phase5.sh 2>/dev/null || true
pkill -f "baseline speculative" 2>/dev/null || true
bash scripts/manage_vllm_server.sh stop >/dev/null 2>&1 || true
sleep 5

run_cell() {  # <spec_json> <label> <context_source>
  local spec="$1" label="$2" ctx="$3"
  echo "=== CELL $label  (ctx=$ctx)  $(date) ==="
  VLLM_SPECULATIVE_CONFIG="$spec" bash scripts/manage_vllm_server.sh restart Qwen/Qwen3-8B || { echo "CELL $label SERVER-FAIL"; return; }
  sleep 10
  python3 scripts/run_experiment.py --baseline speculative --baseline-label "$label" \
      --model Qwen/Qwen3-8B --dataset squad_v2 --num-queries 100 --num-trials 1 --seed 42 \
      --context-source "$ctx" --vllm-telemetry --output-dir "$OUT/$label" \
      || echo "CELL $label RUN-FAIL"
  bash scripts/sync_results_to_gcs.sh analysis || true
  echo "=== CELL $label DONE  $(date) ==="
}

NGRAM='{"method":"ngram","num_speculative_tokens":5}'
EAGLE='{"method":"eagle3","model":"AngelSlim/Qwen3-8B_eagle3","num_speculative_tokens":5}'

run_cell "$NGRAM" spec_ngram_cag  gold
run_cell "$NGRAM" spec_ngram_rag  retrieved
run_cell "$EAGLE" spec_eagle3_cag gold
run_cell "$EAGLE" spec_eagle3_rag retrieved

bash scripts/sync_results_to_gcs.sh analysis || true
echo "SPECULATIVE_MATRIX_DONE"
