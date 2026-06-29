#!/bin/bash
# Speculative-decoding baseline for Phase 2 on a single L4.
# Uses NGRAM speculation (no draft-model GPU cost) so the target model stays
# consistent at Qwen3-8B (a draft model would OOM alongside 8B on a 24GB L4).
# Speculative decoding is output-lossless, so quality should match no_cache; the
# payoff is in TTFT/throughput + the /metrics acceptance rate (via cage-stats).
# 100 queries / 1 trial to match the rest of the Phase-2 run.
set -uo pipefail
cd "$HOME/CAGE"
source cage-env/bin/activate
# Continuous log+results mirror to GCS + full collect on exit (this script + run_phase5.sh
# have no built-in sync loop). NOTE: run_speculative_matrix.sh is the comprehensive
# speculative path (both models, ngram + native draft, with per-cell sentinels); this
# script is the ngram-only quick path.
source scripts/_log_guard.sh

echo "[spec] waiting for any in-flight run to finish..."
while pgrep -f run_compression.sh >/dev/null 2>&1; do sleep 20; done
while pgrep -f "baseline compressed" >/dev/null 2>&1; do sleep 20; done

export NUM_QUERIES=100
export NUM_TRIALS=1
export SEED=42
export TARGET_MODEL=Qwen/Qwen3-8B
export DRAFT_MODEL=Qwen/Qwen3-8B   # unused with ngram; keep consistent
export VLLM_SPECULATIVE_CONFIG='{"method":"ngram","num_speculative_tokens":5}'
export VLLM_ENFORCE_EAGER=1

echo "[spec] launching run_phase5.sh (Qwen3-8B + ngram speculative)..."
bash scripts/run_phase5.sh

echo "[spec] syncing to GCS..."
bash scripts/sync_results_to_gcs.sh analysis
echo "SPECULATIVE_DONE"
