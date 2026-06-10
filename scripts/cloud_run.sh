#!/bin/bash
# Run the CAGE baseline suite on a SINGLE GPU machine with continuous result persistence.
#
# IMPORTANT: run this ON A GPU VM (it starts a local vLLM server via run_phase1.sh).
# Do NOT run it on the CPU router of the multi-VM Terraform cluster. For the *distributed*
# baseline against that cluster, use run_experiment.py + sync_results_to_gcs.sh instead
# (see docs/RUNBOOK.md §9, Path B).
#
# Results are mirrored to the durable GCS bucket every SYNC_INTERVAL seconds (and at exit),
# so an SSH drop, VM preemption, or VM delete cannot lose a finished baseline. Pair with
# `nohup ... &` so it survives disconnects.
#
# Usage:
#   nohup bash scripts/cloud_run.sh [MODEL] [NUM_QUERIES] [NUM_TRIALS] > run.log 2>&1 &
#     MODEL        HF model (default: Qwen/Qwen3-8B)
#     NUM_QUERIES  queries per trial (default: 100)
#     NUM_TRIALS   trials per baseline (default: 10)
#   env:
#     ENABLE_DISTRIBUTED  0 = skip the local 3-replica distributed baseline (default; it
#                         needs ~3x the VRAM and OOMs a single 24GB L4). Set 1 only on a
#                         big-VRAM box. Run the distributed baseline on the cluster instead.
#     SYNC_DIR            local dir to mirror (default: analysis)
#     SYNC_INTERVAL       seconds between background syncs (default: 120)
#     CAGE_RESULTS_BUCKET override bucket (default: gs://<project>-cage-results)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

MODEL="${1:-Qwen/Qwen3-8B}"
NUM_QUERIES="${2:-100}"
NUM_TRIALS="${3:-10}"
SYNC_DIR="${SYNC_DIR:-analysis}"
SYNC_INTERVAL="${SYNC_INTERVAL:-120}"
# Single-GPU-safe default: skip the VRAM-hungry local distributed baseline.
export ENABLE_DISTRIBUTED="${ENABLE_DISTRIBUTED:-0}"

echo "[cage] cloud_run: model=$MODEL queries=$NUM_QUERIES trials=$NUM_TRIALS distributed=$ENABLE_DISTRIBUTED"
echo "[cage] mirroring $SYNC_DIR/ -> GCS every ${SYNC_INTERVAL}s (and at exit)"

# Ensure Redis is up for the redis/hybrid baselines (best-effort; Docker is on the DLVM image).
if ! curl -s localhost:6379 >/dev/null 2>&1 && ! (exec 3<>/dev/tcp/localhost/6379) 2>/dev/null; then
  if command -v docker >/dev/null 2>&1; then
    echo "[cage] starting Redis (docker)..."
    docker run -d -p 6379:6379 --name cage-redis --restart unless-stopped redis:7-alpine >/dev/null 2>&1 || true
  else
    echo "[cage] WARNING: Redis not reachable and docker unavailable; redis/hybrid baselines may fail."
  fi
fi

# Background periodic sync.
(
  while true; do
    bash "$SCRIPT_DIR/sync_results_to_gcs.sh" "$SYNC_DIR" >/dev/null 2>&1 || true
    sleep "$SYNC_INTERVAL"
  done
) &
SYNC_PID=$!

cleanup() {
  kill "$SYNC_PID" 2>/dev/null || true
  echo "[cage] final sync..."
  bash "$SCRIPT_DIR/sync_results_to_gcs.sh" "$SYNC_DIR" || true
}
trap cleanup EXIT

# Run the validated suite (handles prefix-cache on/off + warmup; distributed gated above).
NUM_QUERIES="$NUM_QUERIES" NUM_TRIALS="$NUM_TRIALS" \
  bash "$SCRIPT_DIR/run_phase1.sh" "$MODEL"

echo "[cage] suite complete; results are in $SYNC_DIR/ and mirrored to GCS."
