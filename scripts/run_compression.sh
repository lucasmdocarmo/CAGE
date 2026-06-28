#!/bin/bash
# =============================================================================
# Compression axis (the 2x2) — ratio-matched at ~2x
# =============================================================================
#   context source {CAG, RAG}  x  compression {full, compressed}
#     cag_full        = prefix_cache              (CAG, full precision)
#     rag_full        = rag                       (RAG, full text)
#     compressed_rag  = rag + LLMLingua-2         (~2x fewer prompt tokens; CLIENT-side)
#     compressed_cag  = prefix_cache + FP8 KV     (~2x smaller KV; server LAUNCH-time lever)
#
# Read the 2x2 DOWN (CAG vs RAG) or ACROSS (full vs compressed), never on the diagonal.
# FP8 KV is GPU-meaningful. A pre-flight gate verifies FP8 does NOT disable prefix caching
# (else compressed_cag is confounded — see cloud_docs/VLLM_COMPATIBILITY.md sec 4).
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_DIR/analysis/compression/results"
# Continuous log+results mirror to GCS + full collect on exit (this script has no sync loop).
source "$SCRIPT_DIR/_log_guard.sh"

MODEL="${1:-Qwen/Qwen3-4B}"
DATASET="${DATASET:-squad_v2}"
NUM_QUERIES="${NUM_QUERIES:-100}"
NUM_TRIALS="${NUM_TRIALS:-3}"
SEED="${SEED:-42}"
SKIP_GATE="${SKIP_GATE:-0}"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR"

run_baseline() {  # <baseline> <label> [extra args...]
    local baseline=$1 label=$2; shift 2
    echo ""; echo ">>> $label ($baseline)  $(date)"
    python3 scripts/run_experiment.py \
        --baseline "$baseline" --baseline-label "$label" \
        --model "$MODEL" --dataset "$DATASET" \
        --num-queries "$NUM_QUERIES" --num-trials "$NUM_TRIALS" --seed "$SEED" \
        --vllm-telemetry --output-dir "$OUTPUT_DIR/$label" "$@"
}

echo "=============================================="
echo "CAGE Compression Axis (2x2, ratio-matched ~2x)"
echo "Model: $MODEL  Dataset: $DATASET  Q:$NUM_QUERIES  Trials:$NUM_TRIALS"
echo "=============================================="

# Pre-flight: FP8 must NOT disable prefix caching, or compressed_cag is confounded (RQ5/H4).
if [ "$SKIP_GATE" != "1" ]; then
    echo ">>> Pre-flight: FP8 x prefix-caching gate"
    if ! bash "$SCRIPT_DIR/check_fp8_prefix_cache.sh" "$MODEL"; then
        echo "GATE FAILED -> compressed_cag would be 'no-reuse + compression'."
        echo "Pin a compatible vLLM (cloud_docs/VLLM_COMPATIBILITY.md sec 4) or SKIP_GATE=1 to override."
        exit 1
    fi
fi

# Pre-flight: compressed_rag needs LLMLingua-2 or it silently no-ops (Phase-2 bug:
# compression_applied=False, ratio=1.0 for all rows -> the arm measured plain RAG).
if ! python3 -c "import llmlingua" 2>/dev/null; then
    echo "GATE FAILED -> 'llmlingua' not importable; compressed_rag would NO-OP (ratio 1.0)."
    echo "Run: pip install llmlingua   (or SKIP_GATE=1 to override and accept an invalid arm)."
    [ "$SKIP_GATE" = "1" ] || exit 1
fi

# --- Full row + compressed_rag (full-precision server, prefix caching ON) ---
echo ">>> Server: full precision, prefix caching ON"
./scripts/manage_vllm_server.sh restart "$MODEL"; sleep 10
run_baseline prefix_cache   cag_full
run_baseline rag            rag_full
export CAGE_REQUIRE_COMPRESSION=1   # raise (not silent no-op) if LLMLingua can't compress
run_baseline compressed_rag compressed_rag      # LLMLingua-2, client-side text compression
unset CAGE_REQUIRE_COMPRESSION

# --- compressed_cag (FP8 KV — the same launch-lever speculative uses) ---
echo ">>> Server: FP8 KV cache ON (compressed_cag)"
VLLM_KV_CACHE_DTYPE=fp8 ./scripts/manage_vllm_server.sh restart "$MODEL"; sleep 10
run_baseline compressed_cag compressed_cag

./scripts/manage_vllm_server.sh stop || true
echo ""; echo "=============================================="
echo "Compression 2x2 complete -> $OUTPUT_DIR"
echo "Read DOWN (CAG vs RAG) or ACROSS (full vs compressed), not diagonally."
echo "=============================================="
