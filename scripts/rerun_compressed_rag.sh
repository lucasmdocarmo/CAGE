#!/bin/bash
# Rerun the compressed_rag arm with FORCED retrieval to fix the gold-vs-retrieved
# confound (it had used gold context via --context-source auto, making it
# "CAG+compression" instead of "RAG+compression"). This makes the 2x2 RAG row valid.
#
# NOTE: run_compression.sh now passes --context-source retrieved for compressed_rag
# directly, so a fresh 2x2 no longer needs this rerun. Keep it only to repair an OLD
# confounded compressed_rag tree in place. It MUST purge the stale trial_*/ dirs first
# (see below) or statistical_tests.py keeps reading them with precedence.
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

# Purge stale confounded trial_*/ dirs FIRST: statistical_tests.load_baseline_per_example
# reads trial_*/results.csv with PRECEDENCE and only falls back to the top-level
# results.csv when NO trial_*/ exist. The old run_compression.sh wrote trial_1/2/3
# (gold/confounded); without this purge the corrected single-trial top-level CSV below is
# shadowed and the rerun silently does nothing.
echo "[rerun] purging stale trial_*/ dirs so the corrected results.csv is not shadowed..."
rm -rf "$HOME/CAGE/analysis/compression/results/compressed_rag/trial_"* 2>/dev/null || true

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
