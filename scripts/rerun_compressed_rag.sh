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

echo "[rerun] running compressed_rag with --context-source retrieved..."
python3 scripts/run_experiment.py \
    --baseline compressed_rag --baseline-label compressed_rag \
    --model Qwen/Qwen3-8B --dataset squad_v2 \
    --num-queries 100 --num-trials 1 --seed 42 \
    --context-source retrieved --vllm-telemetry \
    --output-dir "$HOME/CAGE/analysis/compression/results/compressed_rag"

echo "[rerun] syncing to GCS..."
bash scripts/sync_results_to_gcs.sh analysis
echo "RERUN_COMPRESSED_RAG_DONE"
