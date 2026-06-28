#!/bin/bash
# Rerun the compressed_rag arm with FORCED retrieval to fix the gold-vs-retrieved
# confound (it had used gold context via --context-source auto, making it
# "CAG+compression" instead of "RAG+compression"). This makes the 2x2 RAG row valid.
#
# Self-sequencing: waits for any in-flight compression run to finish first (avoids a
# vLLM port / GPU conflict), restarts vLLM full-precision (no FP8), then reruns just
# compressed_rag with --context-source retrieved and syncs to GCS.
set -uo pipefail
cd "$HOME/CAGE"
source cage-env/bin/activate
# Continuous log+results mirror to GCS + full collect on exit (no built-in sync loop).
source scripts/_log_guard.sh

echo "[rerun] waiting for any running compression to finish..."
while pgrep -f run_compression.sh >/dev/null 2>&1; do sleep 20; done
while pgrep -f "baseline compressed_cag" >/dev/null 2>&1; do sleep 20; done
echo "[rerun] clear. restarting vLLM full-precision (prefix caching on, eager)..."

VLLM_ENFORCE_EAGER=1 bash scripts/manage_vllm_server.sh restart Qwen/Qwen3-8B
sleep 10

# Pre-flight: without LLMLingua-2 this arm silently no-ops (Phase-2 bug: prompt_tokens
# stayed at rag_full's 633, compression_applied=False). Verify the package + go strict.
if ! python3 -c "import llmlingua" 2>/dev/null; then
    echo "ABORT: 'llmlingua' not importable -> compressed_rag would NO-OP (ratio 1.0)."
    echo "Run: pip install llmlingua   then re-run this script."
    exit 1
fi

echo "[rerun] running compressed_rag with --context-source retrieved (strict compression)..."
CAGE_REQUIRE_COMPRESSION=1 python3 scripts/run_experiment.py \
    --baseline compressed_rag --baseline-label compressed_rag \
    --model Qwen/Qwen3-8B --dataset squad_v2 \
    --num-queries 100 --num-trials 1 --seed 42 \
    --context-source retrieved --vllm-telemetry \
    --output-dir "$HOME/CAGE/analysis/compression/results/compressed_rag"

echo "[rerun] syncing to GCS..."
bash scripts/sync_results_to_gcs.sh analysis
echo "RERUN_COMPRESSED_RAG_DONE"
