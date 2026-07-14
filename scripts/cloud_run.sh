#!/bin/bash
# Run the CAGE baseline suite on a SINGLE GPU machine with continuous result persistence.
#
# IMPORTANT: run this ON A GPU VM (it starts a local vLLM server via run_phase1.sh).
# Do NOT run it on the CPU router of the multi-VM Terraform cluster. For the *distributed*
# baseline against that cluster, use run_experiment.py + sync_results_to_gcs.sh instead
# (see Cloud/RUNBOOK.md §9, Path B).
#
# Results are mirrored to the durable GCS bucket every SYNC_INTERVAL seconds (and at exit),
# so an SSH drop, VM preemption, or VM delete cannot lose a finished baseline. Pair with
# `nohup ... &` so it survives disconnects.
#
# Usage:
#   nohup bash scripts/cloud_run.sh [MODEL] [NUM_QUERIES] [NUM_TRIALS] > run.log 2>&1 &
#     MODEL        HF model (default: Qwen/Qwen3-8B)
#     NUM_QUERIES  queries per trial (default: 500)
#     NUM_TRIALS   trials per baseline (default: 3)
#   env:
#     ENABLE_DISTRIBUTED  0 = skip the local 3-replica distributed baseline (default; it
#                         needs ~3x the VRAM and OOMs a single 24GB L4). Set 1 only on a
#                         big-VRAM box. Run the distributed baseline on the cluster instead.
#     SYNC_DIR            local dir to mirror (default: analysis)
#     SYNC_INTERVAL       seconds between background syncs (default: 120)
#     CAGE_RESULTS_BUCKET override bucket (default: gs://<project>-cage-results)
#
# Launch-time levers (compressed_cag FP8 / speculative) need a server relaunch with an env var,
# so run those via their own scripts instead of this suite:
#     compression 2x2:  bash scripts/run_compression.sh $MODEL   (gates FP8 x prefix-caching)
#     speculative:      bash scripts/run_speculative_matrix.sh $MODEL   (per model; gates native draft)
# The vLLM image is pinned to v0.11.0 — see Cloud/VLLM_COMPATIBILITY.md.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

MODEL="${1:-Qwen/Qwen3-8B}"
NUM_QUERIES="${2:-500}"
NUM_TRIALS="${3:-3}"
SYNC_DIR="${SYNC_DIR:-analysis}"
SYNC_INTERVAL="${SYNC_INTERVAL:-120}"
# Single-GPU-safe default: skip the VRAM-hungry local distributed baseline.
export ENABLE_DISTRIBUTED="${ENABLE_DISTRIBUTED:-0}"

# vLLM telemetry via cage-stats: auto-capture on cloud (set VLLM_TELEMETRY=0 to disable).
export VLLM_TELEMETRY="${VLLM_TELEMETRY:-1}"
# Resolve cage-stats for the in-process telemetry path if it isn't pip-installed, and put it on
# PYTHONPATH so both the importability check below AND the run_experiment.py subprocess find it.
if [ -z "${CAGE_STATS_HOME:-}" ] && [ -d "$PROJECT_DIR/../cage-stats/cage_stats" ]; then
  export CAGE_STATS_HOME="$(cd "$PROJECT_DIR/../cage-stats" && pwd)"
fi
[ -n "${CAGE_STATS_HOME:-}" ] && export PYTHONPATH="${CAGE_STATS_HOME}:${PYTHONPATH:-}"
if [ "$VLLM_TELEMETRY" != "0" ]; then
  # Fail loud rather than run the whole suite producing spec-decode-only telemetry we cannot
  # use to build the cache/KV figures (rich fields come ONLY from an importable cage-stats).
  if ! python3 -c "import cage_stats.api" 2>/dev/null; then
    echo "[cage] FATAL: --vllm-telemetry is ON but 'cage_stats.api' is not importable." >&2
    echo "[cage]   Rich vLLM telemetry (cached_tokens / prefix-hit / KV usage) would degrade to" >&2
    echo "[cage]   spec-decode-only. Fix: pip install -e ../cage-stats (or set CAGE_STATS_HOME)," >&2
    echo "[cage]   or rerun with VLLM_TELEMETRY=0." >&2
    exit 1
  fi
  echo "[cage] vLLM telemetry ON (cage-stats${CAGE_STATS_HOME:+ @ $CAGE_STATS_HOME}) -> per-baseline vllm_telemetry.json"
fi

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

# Background periodic sync (results + logs, so an SSH drop or preemption loses neither).
(
  while true; do
    bash "$SCRIPT_DIR/sync_results_to_gcs.sh" "$SYNC_DIR" >/dev/null 2>&1 || true
    bash "$SCRIPT_DIR/collect_logs.sh" --light >/dev/null 2>&1 || true
    sleep "$SYNC_INTERVAL"
  done
) &
SYNC_PID=$!

# Load the uniform serving config (Option A) into THIS shell before the sidecar launches, so the
# run manifest records the actual enforce_eager / max_model_len / gpu_memory_utilization. It is
# idempotent (run_phase1.sh re-sources it) and only sets values not already in the env, so a
# memory-pressure sweep that exports VLLM_GPU_MEMORY_UTILIZATION beforehand is preserved.
source "$SCRIPT_DIR/_serving_config.sh"

# Observability sidecar (provenance + snapshots): writes run_manifest.json, periodic GPU/
# serving/progress JSON+PNG snapshots, and provenance hashes under $SYNC_DIR/observability/ --
# which the periodic sync above already mirrors to GCS, so a laptop can watch live via
# scripts/watch_run.sh. It observes from OUTSIDE the run (reads STATUS/results.csv), so it can
# never perturb serving timings. Set OBSERVE=0 to disable.
OBSERVE="${OBSERVE:-1}"
OBSERVE_PID=""
if [ "$OBSERVE" != "0" ]; then
  mkdir -p logs
  nohup python3 "$SCRIPT_DIR/observe_run.py" \
    --run-dir "$SYNC_DIR" --model "$MODEL" \
    --num-queries "$NUM_QUERIES" --num-trials "$NUM_TRIALS" \
    --interval "${OBSERVE_INTERVAL:-30}" > logs/observe.log 2>&1 &
  OBSERVE_PID=$!
  echo "[cage] observability sidecar started (pid $OBSERVE_PID) -> $SYNC_DIR/observability/ (log: logs/observe.log)"
fi

cleanup() {
  # Stop the observability sidecar FIRST and wait: SIGTERM makes it write a final snapshot +
  # provenance.json, which must exist before the final GCS sync below carries them off-box.
  if [ -n "$OBSERVE_PID" ]; then
    kill "$OBSERVE_PID" 2>/dev/null || true
    wait "$OBSERVE_PID" 2>/dev/null || true
  fi
  # Stop the periodic syncer and WAIT for its in-flight rsync to finish, so it does not
  # race this final sync to the same destination.
  kill "$SYNC_PID" 2>/dev/null || true
  wait "$SYNC_PID" 2>/dev/null || true
  echo "[cage] final sync (results + full logs + forensics)..."
  bash "$SCRIPT_DIR/sync_results_to_gcs.sh" "$SYNC_DIR" || true
  bash "$SCRIPT_DIR/collect_logs.sh" || true
}
# EXIT covers normal/error exits; INT/TERM cover Ctrl-C and (best-effort) the SIGTERM a
# GCP spot preemption raises. The on_signal handler just exits, which fires the EXIT trap
# once (so cleanup runs exactly once and collects the full forensic snapshot before death).
on_signal() { echo "[cage] signal received -> collecting logs before exit"; exit 1; }
trap on_signal INT TERM
trap cleanup EXIT

# Run the validated suite (handles prefix-cache on/off + warmup; distributed gated above).
NUM_QUERIES="$NUM_QUERIES" NUM_TRIALS="$NUM_TRIALS" \
  bash "$SCRIPT_DIR/run_phase1.sh" "$MODEL"

echo "[cage] suite complete; results are in $SYNC_DIR/ and mirrored to GCS."
