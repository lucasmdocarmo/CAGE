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
# Usage: bash scripts/3_run/run_speculative_matrix.sh [MODEL]
#   MODEL defaults to Qwen/Qwen3-8B. Run ONCE PER MODEL to cover both speculative families:
#     bash scripts/3_run/run_speculative_matrix.sh Qwen/Qwen3-8B
#     bash scripts/3_run/run_speculative_matrix.sh XiaomiMiMo/MiMo-7B-RL
#   Defaults to the full design (500 queries x 3 trials); override for a smoke, e.g.
#     NUM_QUERIES=5 NUM_TRIALS=1 bash scripts/3_run/run_speculative_matrix.sh <MODEL>
#
# IMPORTANT: the MiMo MTP method string ("mimo_mtp") MUST be live-validated on vLLM 0.11.0
# during the smoke run -- if vLLM rejects it, the server won't start and the cell is marked
# SERVER-FAIL (with a STATUS sentinel) instead of silently missing. Override the exact JSON
# via MIMO_MTP_CONFIG if the installed vLLM expects a different method name.
#
# max_model_len capped to 4096 and mem-util 0.90 so a draft head fits alongside the target
# on a 24GB L4.
set -uo pipefail
# Resolve the repo from THIS script's location (not a hardcoded $HOME/CAGE) so it works in any
# checkout / with a local .venv.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"
# Activate the project venv regardless of name (cage-env on the VM, .venv locally).
if [ -z "${VIRTUAL_ENV:-}" ]; then
  for _v in cage-env .venv ../cage-env; do
    [ -f "$_v/bin/activate" ] && { echo "Activating venv: $_v"; source "$_v/bin/activate"; break; }
  done
fi

MODEL="${1:-Qwen/Qwen3-8B}"
DATASET="${DATASET:-squad_v2}"
NUM_QUERIES="${NUM_QUERIES:-500}"
NUM_TRIALS="${NUM_TRIALS:-3}"
SEED="${SEED:-42}"

# Outputs land under the shared run root minted by cloud_run.sh (CAGE_RUN_ROOT/speculative) so the
# core + compression + speculative trees for ONE sweep cell aggregate together; a standalone run
# self-mints the SAME date_HHMM_model_NxT run-id under results/<phase>/<run-id>/.
if [ -n "${CAGE_RUN_ROOT:-}" ]; then
    RUN_ROOT="$CAGE_RUN_ROOT"
else
    _phase="${CAGE_PHASE:-phase2}"
    _slug="$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]' | sed -E 's|.*/||; s|[^a-z0-9]+|-|g; s|^-+||; s|-+$||')"
    _rid="${CAGE_RUN_ID:-$(date +%Y-%m-%d_%H%M)_${_slug}_${NUM_QUERIES}x${NUM_TRIALS}}"
    RUN_ROOT="$PROJECT_DIR/results/$_phase/$_rid"
fi
OUT="$RUN_ROOT/speculative"
mkdir -p "$OUT"
# Point the sourced log guard's GCS mirror at this run root (relative to PROJECT_DIR), not analysis/.
export CAGE_SYNC_DIR="${CAGE_SYNC_DIR:-${RUN_ROOT#"$PROJECT_DIR/"}}"
# Continuous log+results mirror to GCS + full collect on exit (no built-in sync loop).
source scripts/lib/_log_guard.sh

# Stop the vLLM server on ANY exit so it does not linger holding ~all VRAM until teardown
# (and so an early hard error does not orphan it). Matches run_baselines.sh's cleanup discipline.
trap 'bash scripts/2_serving/manage_vllm_server.sh stop >/dev/null 2>&1 || true' EXIT

# Uniform serving config across ALL trees (Option A): non-eager / max_len 4096 / mem-util 0.90
# (single source of truth), so the speculative 2x2 matches the core + compression trees and
# cross-mechanism comparisons are fair. The speculative tree is the TIGHTEST on VRAM under
# non-eager (draft head + CUDA-graph capture over variable draft lengths); validate it in the
# pre-flight, and if a cell OOMs, fall back for this tree only via
# VLLM_ENFORCE_EAGER=1 bash scripts/3_run/run_speculative_matrix.sh <model> -- a recorded deviation
# the run manifest captures. mem-util (0.90) is the swept memory-pressure axis.
source scripts/lib/_serving_config.sh

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
bash scripts/2_serving/manage_vllm_server.sh stop >/dev/null 2>&1 || true
sleep 5

FAILED=()

cell_complete() {  # <cell_dir> -> 0 iff trial_1..NUM_TRIALS all have metrics.json
  local dir="$1" t
  for ((t = 1; t <= NUM_TRIALS; t++)); do
    [ -f "$dir/trial_${t}/metrics.json" ] || return 1
  done
  return 0
}

run_cell() {  # <spec_json> <label> <context_source>
  local spec="$1" label="$2" ctx="$3"
  # Skip-completed (resume): a cell with all trial metrics.json present is done -- don't
  # burn a server restart + N queries re-measuring it. CAGE_FORCE_RERUN=1 overrides.
  if [ "${CAGE_FORCE_RERUN:-0}" != "1" ] && cell_complete "$OUT/$label"; then
    echo "SKIP (complete): $label"
    return
  fi
  # Wipe a stale cell (partial data or an old STATUS sentinel, or CAGE_FORCE_RERUN=1)
  # so the re-run starts clean and a later success cannot coexist with STATUS=failed.
  if [ -d "$OUT/$label" ]; then
    echo "[matrix] wiping stale cell $label (partial or CAGE_FORCE_RERUN=1) before re-run"
    rm -rf "$OUT/$label"
  fi
  # Record the TRUE mechanism in the manifest: parse method/tokens/draft-model from the spec
  # JSON and pass them as flags, so baseline_config.speculative_method matches what the server
  # actually launched (otherwise it defaults to 'draft_model' and the provenance is lost).
  local method tokens draft_model
  method=$(SPEC="$spec" python3 -c "import json,os;print(json.loads(os.environ['SPEC']).get('method','draft_model'))" 2>/dev/null || echo draft_model)
  tokens=$(SPEC="$spec" python3 -c "import json,os;print(json.loads(os.environ['SPEC']).get('num_speculative_tokens',5))" 2>/dev/null || echo 5)
  draft_model=$(SPEC="$spec" python3 -c "import json,os;print(json.loads(os.environ['SPEC']).get('model',''))" 2>/dev/null || echo "")
  local spec_model_flag=()
  [ -n "$draft_model" ] && spec_model_flag+=(--speculative-model "$draft_model")
  echo "=== CELL $label  (ctx=$ctx)  method=$method tokens=$tokens  $(date) ==="
  # A failed launch / run writes a STATUS sentinel so a missing cell is LOUD in the stats
  # consolidation rather than silently absent (and the 2x2 reported as complete with a hole).
  if ! VLLM_SPECULATIVE_CONFIG="$spec" bash scripts/2_serving/manage_vllm_server.sh restart "$MODEL"; then
    echo "CELL $label SERVER-FAIL"
    mkdir -p "$OUT/$label"
    echo "STATUS=failed reason=server model=$MODEL spec=$spec $(date)" > "$OUT/$label/STATUS"
    FAILED+=("$label(server)")
    return
  fi
  sleep 10
  # 1-query warmup: the FIRST request after each spec-server restart pays a 5-6x TTFT spike
  # (505-619ms vs ~110ms steady-state) from lazy CUDA-graph/kernel warm paths. One throwaway
  # generation absorbs it OUTSIDE the measured window. Deliberately NOT run_experiment.py's
  # --warmup-queries: that flag treats any N>0 as "replay the FULL measured set" (a
  # cache-priming flag, not a count), which would pre-warm the prefix cache and add ~NUM_QUERIES
  # extra generations per cell -- distorting the very cold-start serving read this 2x2 measures.
  curl -s -m 90 "http://localhost:${VLLM_PORT:-8000}/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"warmup\",\"max_tokens\":8}" >/dev/null 2>&1 || true
  if ! python3 scripts/3_run/run_experiment.py --baseline speculative --baseline-label "$label" \
      --model "$MODEL" --dataset "$DATASET" --num-queries "$NUM_QUERIES" --num-trials "$NUM_TRIALS" --seed "$SEED" \
      --speculative-method "$method" --num-speculative-tokens "$tokens" ${spec_model_flag[@]+"${spec_model_flag[@]}"} \
      --context-source "$ctx" --vllm-telemetry --output-dir "$OUT/$label"; then
    echo "CELL $label RUN-FAIL"
    mkdir -p "$OUT/$label"
    echo "STATUS=failed reason=run model=$MODEL spec=$spec $(date)" > "$OUT/$label/STATUS"
    FAILED+=("$label(run)")
  fi
  bash scripts/5_observability/sync_results_to_gcs.sh "$CAGE_SYNC_DIR" || true
  echo "=== CELL $label DONE  $(date) ==="
}

run_cell "$NGRAM" "spec_${MTAG}_ngram_cag"          gold
run_cell "$NGRAM" "spec_${MTAG}_ngram_rag"          retrieved

# Pre-flight the model-native draft method BEFORE spending the sweep on it: vLLM can
# SOFT-accept the config (server healthy) yet never speculate, yielding a "complete" cell
# with null acceptance. This gate asserts vllm:spec_decode_num_draft_tokens_total > 0. The
# ngram cells above are unaffected. SKIP_SPEC_GATE=1 overrides (accepts an unvalidated arm).
SKIP_SPEC_GATE="${SKIP_SPEC_GATE:-0}"
if [ "$SKIP_SPEC_GATE" = "1" ]; then
  DRAFT_OK=1
elif [ "${CAGE_FORCE_RERUN:-0}" != "1" ] \
    && cell_complete "$OUT/spec_${MTAG}_${DRAFT_LABEL}_cag" \
    && cell_complete "$OUT/spec_${MTAG}_${DRAFT_LABEL}_rag"; then
  # Resume: both draft cells already complete -- the gate's server restart would be wasted,
  # and run_cell will skip both cells anyway.
  echo "[matrix] both native-draft cells complete -- skipping the spec-decode gate"
  DRAFT_OK=1
else
  # exit 0 = PASS, 1 = real FAIL, 2 = INCONCLUSIVE (transient probe/timeout). Retry ONCE on 2
  # so a network hiccup during the ~3 probe generations does not permanently drop a valid arm.
  bash scripts/checks/check_mtp_spec_decode.sh "$MODEL" "$DRAFT"; _grc=$?
  if [ "$_grc" = "2" ]; then
    echo "[matrix] native-draft gate INCONCLUSIVE (exit 2); retrying once..."
    bash scripts/checks/check_mtp_spec_decode.sh "$MODEL" "$DRAFT"; _grc=$?
  fi
  if [ "$_grc" = "0" ]; then
    DRAFT_OK=1
  else
    DRAFT_OK=0
    echo "[matrix] GATE FAILED (exit $_grc) -> native draft '$DRAFT_LABEL' does not engage on this vLLM."
    echo "[matrix] skipping the 2 draft cells and marking them failed (ngram cells kept)."
    for _lbl in "spec_${MTAG}_${DRAFT_LABEL}_cag" "spec_${MTAG}_${DRAFT_LABEL}_rag"; do
      # Resume-safe: never stamp STATUS=failed over a cell already complete from a prior run.
      if [ "${CAGE_FORCE_RERUN:-0}" != "1" ] && cell_complete "$OUT/$_lbl"; then
        echo "[matrix] $_lbl already complete from a previous run -- keeping it despite gate failure"
        continue
      fi
      mkdir -p "$OUT/$_lbl"
      echo "STATUS=failed reason=spec_gate model=$MODEL spec=$DRAFT $(date)" > "$OUT/$_lbl/STATUS"
      FAILED+=("$_lbl(spec_gate)")
    done
  fi
fi

if [ "$DRAFT_OK" = "1" ]; then
  run_cell "$DRAFT" "spec_${MTAG}_${DRAFT_LABEL}_cag" gold
  run_cell "$DRAFT" "spec_${MTAG}_${DRAFT_LABEL}_rag" retrieved
fi

bash scripts/5_observability/sync_results_to_gcs.sh "$CAGE_SYNC_DIR" || true
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "MATRIX INCOMPLETE: ${#FAILED[@]} cell(s) failed: ${FAILED[*]}"
  echo "SPECULATIVE_MATRIX_DONE (model=$MODEL)"
  # Nonzero so an orchestrator (run_full_sweep.sh) records this tree as FAILED; all cells
  # were still attempted, and a re-run with the same CAGE_RUN_ID resumes only the failures.
  exit 1
fi
echo "SPECULATIVE_MATRIX_DONE (model=$MODEL)"
