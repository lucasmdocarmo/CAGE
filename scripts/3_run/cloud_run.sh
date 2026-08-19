#!/bin/bash
# =============================================================================
# PILOT HARNESS — drives the retired 9-name taxonomy via the alias map; the
# campaign harness (CellSpec-native, D6 open-loop) lands at tranche P1; use for
# pilot re-scoring only.
# =============================================================================
# Run the CAGE baseline suite on a SINGLE GPU machine with continuous result persistence.
#
# IMPORTANT: run this ON A GPU box (it starts a local vLLM server via run_baselines.sh).
# Do NOT run it on the CPU router of a multi-VM cluster. For the *distributed* baseline
# against a cluster, use run_experiment.py + sync_results.sh directly (a pilot-era path;
# the cluster recipe it pointed at was retired with the pilot runbook — see git history
# of cloud/RUNBOOK.md).
#
# Results are mirrored to the durable backup target every SYNC_INTERVAL seconds (and at exit),
# so an SSH drop, VM preemption, or VM delete cannot lose a finished baseline. Pair with
# `nohup ... &` so it survives disconnects.
#
# Usage:
#   nohup bash scripts/3_run/cloud_run.sh [MODEL] [NUM_QUERIES] [NUM_TRIALS] > run.log 2>&1 &
#     MODEL        HF model (default: Qwen/Qwen3-8B)
#     NUM_QUERIES  queries per trial (default: 500)
#     NUM_TRIALS   trials per baseline (default: 3)
#   env:
#     ENABLE_DISTRIBUTED  0 = skip the local 3-replica distributed baseline (default; it
#                         needs ~3x the VRAM and OOMs a single 24GB L4). Set 1 only on a
#                         big-VRAM box. Run the distributed baseline on the cluster instead.
#     SYNC_DIR            local dir to mirror (default: results/<phase>/<run-id>, minted below)
#     PHASE               phase slug for the run root (default: phase2)
#     CAGE_RUN_ID         override the auto-minted run-id (default: date_HHMM_model_NxT)
#     SYNC_INTERVAL       seconds between background syncs (default: 120)
#     CAGE_BACKUP_TARGET  provider-neutral backup target (gs://|s3://|ssh://|file://;
#                         RunPod is the PRIMARY provider — task #137)
#     CAGE_RESULTS_BUCKET legacy GCS override (bare name or gs://); on a GCP box the
#                         metadata-derived gs://<project>-cage-results still applies
#     CAGE_ALLOW_NO_BACKUP=1  ONLY way to start with no backup target (J4 refusal
#                         gate); the override is recorded to <run-root>/NO_BACKUP_OVERRIDE
#                         and echoed into the run manifest
#
# Launch-time levers (compressed_cag FP8) need a server relaunch with an env var,
# so run those via their own scripts instead of this suite:
#     compression 2x2:  bash scripts/3_run/run_compression.sh $MODEL   (gates FP8 x prefix-caching)
#     (speculative 2x2 RETIRED per charter §7.5 -> scripts/deprecated/run_speculative_matrix.sh)
# The vLLM pin is v0.19.1 (Phase-3; Phase-2 ran v0.11.0) — see cloud/VLLM_COMPATIBILITY.md.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"
# shellcheck source=scripts/lib/_common.sh
source "$PROJECT_DIR/scripts/lib/_common.sh"

MODEL="${1:-Qwen/Qwen3-8B}"
NUM_QUERIES="${2:-500}"
NUM_TRIALS="${3:-3}"

# Mint a unique, self-describing run-id so runs NEVER mix (identically local + on GCS).
# run-id = <YYYY-MM-DD_HHMMSS>_<model-slug>_<Q>x<T>_<rand4>_<dataset> (mint_run_id in
# _common.sh; seconds + random suffix per finding J3 -- the old minute-granular ids let
# two launches in the same minute CONVERGE on one root and a resume a minute later
# FRAGMENT onto a new one; the dataset suffix carries the run's dataset identity).
PHASE="${PHASE:-phase2}"
_model_slug="$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]' | sed -E 's|.*/||; s|[^a-z0-9]+|-|g; s|^-+||; s|-+$||')"
RUN_ID="${CAGE_RUN_ID:-$(mint_run_id "$_model_slug" "$NUM_QUERIES" "$NUM_TRIALS" "${DATASET:-squad_v2}")}"
RUN_ROOT="results/${PHASE}/${RUN_ID}"
# Exported for run_baselines.sh (+ the whole run tree) to inherit the SAME root.
export CAGE_PHASE="$PHASE" CAGE_RUN_ID="$RUN_ID" CAGE_RUN_ROOT="$PROJECT_DIR/$RUN_ROOT"
mkdir -p "$CAGE_RUN_ROOT"

# One runner per run root (finding J3): a second resume instance on the same root could
# rm -rf a cell a live run is writing. Re-entrant: run_baselines.sh (child) sees
# CAGE_RUN_LOCK_HELD and skips re-acquisition; under run_full_sweep.sh the parent
# already holds it and THIS call is the no-op. Detached helpers launched below close
# the lock fd (200>&-) so a surviving daemon can never hold the lock.
acquire_run_lock "$CAGE_RUN_ROOT"

# J4 refusal gate (task #137, build item (c)): a run with NO off-box persistence
# REFUSES to start. require_backup_target resolves the provider-neutral target
# (gs://|s3://|ssh://|file://; RunPod-primary) or dies LOUD; the only override is
# CAGE_ALLOW_NO_BACKUP=1, which it records durably to $CAGE_RUN_ROOT/NO_BACKUP_OVERRIDE
# (observe_run.py echoes the override into run_manifest.json).
# shellcheck source=scripts/lib/transport.sh
source "$PROJECT_DIR/scripts/lib/transport.sh"
BACKUP_TARGET="$(require_backup_target "$CAGE_RUN_ROOT")"
export CAGE_BACKUP_TARGET="$BACKUP_TARGET"
if [ -z "$BACKUP_TARGET" ]; then
  echo "[cage] WARNING: running with NO off-box backup (CAGE_ALLOW_NO_BACKUP=1; marker: $CAGE_RUN_ROOT/NO_BACKUP_OVERRIDE)"
else
  echo "[cage] backup target: $BACKUP_TARGET"
fi

# Mirror the whole run root (baselines + observability + logs) off-box verbatim.
SYNC_DIR="${SYNC_DIR:-$RUN_ROOT}"
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
echo "[cage] run-id=$RUN_ID  phase=$PHASE  run-root=$RUN_ROOT"
echo "[cage] mirroring $SYNC_DIR/ -> ${BACKUP_TARGET:-<NO BACKUP (override active)>} every ${SYNC_INTERVAL}s (and at exit)"

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
# Launched via setsid into its OWN process group (finding J8): killing only the loop pid
# left the in-flight grandchild (the bash sync script + its gcloud/gsutil) running and
# racing the final sync. With a private pgid, cleanup kills the WHOLE tree. Args are
# passed positionally (never interpolated) and the run-lock fd is closed (200>&-) so the
# detached loop can never hold the J3 run lock after this script dies. macOS (no setsid,
# local testing only) falls back LOUDLY to a plain background job + pkill in cleanup.
# Failures inside the loop are ANNOUNCED to stderr (visible in run.log), never
# swallowed with `|| true` (finding J4: silent sync failure = empty bucket at
# run end and nobody noticed); the loop itself keeps retrying every interval.
# shellcheck disable=SC2016  # single-quoted on purpose: expands inside the child
_SYNC_LOOP='
  sync_sh="$1" collect_sh="$2" sync_dir="$3" interval="$4"
  while true; do
    bash "$sync_sh" "$sync_dir" >/dev/null 2>&1 \
      || echo "[cage] WARNING: periodic results sync FAILED (rc=$?; see .agent/last_sync_fail_*) — retrying in ${interval}s" >&2
    bash "$collect_sh" --light >/dev/null 2>&1 \
      || echo "[cage] WARNING: periodic log collection FAILED (rc=$?) — retrying in ${interval}s" >&2
    sleep "$interval"
  done
'
SYNC_IS_PGRP=0
if command -v setsid >/dev/null 2>&1; then
  setsid bash -c "$_SYNC_LOOP" _ "$SCRIPT_DIR/../5_observability/sync_results.sh" \
    "$SCRIPT_DIR/../5_observability/collect_logs.sh" "$SYNC_DIR" "$SYNC_INTERVAL" 200>&- &
  SYNC_PID=$!
  SYNC_IS_PGRP=1
else
  echo "[cage] WARNING: setsid not found (macOS?) -> sync-loop grandchildren are killed via pkill fallback"
  bash -c "$_SYNC_LOOP" _ "$SCRIPT_DIR/../5_observability/sync_results.sh" \
    "$SCRIPT_DIR/../5_observability/collect_logs.sh" "$SYNC_DIR" "$SYNC_INTERVAL" 200>&- &
  SYNC_PID=$!
fi

# Load the uniform serving config (Option A) into THIS shell before the sidecar launches, so the
# run manifest records the actual enforce_eager / max_model_len / gpu_memory_utilization. It is
# idempotent (run_baselines.sh re-sources it) and only sets values not already in the env, so a
# memory-pressure sweep that exports VLLM_GPU_MEMORY_UTILIZATION beforehand is preserved.
source "$SCRIPT_DIR/../lib/_serving_config.sh"

# Observability sidecar (provenance + snapshots): writes run_manifest.json, periodic GPU/
# serving/progress JSON+PNG snapshots, and provenance hashes under $SYNC_DIR/observability/ --
# which the periodic sync above already mirrors to GCS, so a laptop can watch live via
# scripts/5_observability/watch_run.sh. It observes from OUTSIDE the run (reads STATUS/results.csv), so it can
# never perturb serving timings. Set OBSERVE=0 to disable.
OBSERVE="${OBSERVE:-1}"
OBSERVE_PID=""
if [ "$OBSERVE" != "0" ]; then
  mkdir -p logs
  # --seed/--dataset mirror run_experiment.py's defaults so the manifest records the real
  # values instead of null (2026-07-15 audit: seed/dataset were null for the whole run).
  # kv_cache_dtype/max_model_len/gpu_mem_util are read from the VLLM_* env by observe_run.
  # 200>&-: the sidecar must not inherit the J3 run-lock fd (it can outlive a SIGKILLed
  # runner and would then hold the lock, blocking every resume until it was hunted down).
  nohup python3 "$SCRIPT_DIR/../5_observability/observe_run.py" \
    --run-dir "$SYNC_DIR" --run-id "$RUN_ID" --model "$MODEL" \
    --num-queries "$NUM_QUERIES" --num-trials "$NUM_TRIALS" \
    --seed "${SEED:-42}" --dataset "${DATASET:-squad_v2}" \
    --interval "${OBSERVE_INTERVAL:-30}" > logs/observe.log 2>&1 200>&- &
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
  # Stop the periodic syncer -- the WHOLE process group, not just the loop pid (finding
  # J8: the in-flight grandchild gsutil/gcloud survived a pid-only kill) -- and WAIT so
  # nothing races this final sync to the same destination.
  if [ "${SYNC_IS_PGRP:-0}" = "1" ]; then
    kill -TERM -- "-$SYNC_PID" 2>/dev/null || true
  else
    pkill -TERM -P "$SYNC_PID" 2>/dev/null || true
    kill -TERM "$SYNC_PID" 2>/dev/null || true
  fi
  wait "$SYNC_PID" 2>/dev/null || true
  echo "[cage] final sync (results + full logs + forensics)..."
  # LOUD on failure (J4): a failed FINAL sync means the run data may exist only
  # on this box — announce it unmissably instead of swallowing with `|| true`.
  bash "$SCRIPT_DIR/../5_observability/sync_results.sh" "$SYNC_DIR" \
    || echo "[cage] ERROR: FINAL RESULTS SYNC FAILED — the run data may exist ONLY on this box; do NOT tear down before a successful manual sync/pull (J4)" >&2
  bash "$SCRIPT_DIR/../5_observability/collect_logs.sh" \
    || echo "[cage] ERROR: final log collection FAILED — off-box forensics may be incomplete (J4)" >&2
}
# EXIT covers normal/error exits; INT/TERM cover Ctrl-C and (best-effort) the SIGTERM a
# GCP spot preemption raises. The on_signal handler just exits, which fires the EXIT trap
# once (so cleanup runs exactly once and collects the full forensic snapshot before death).
on_signal() { echo "[cage] signal received -> collecting logs before exit"; exit 1; }
trap on_signal INT TERM
trap cleanup EXIT

# Run the validated suite (handles prefix-cache on/off + warmup; distributed gated above).
NUM_QUERIES="$NUM_QUERIES" NUM_TRIALS="$NUM_TRIALS" \
  bash "$SCRIPT_DIR/../3_run/run_baselines.sh" "$MODEL"

# Auto-generate figures for the finished run (best-effort; a plot error never fails the run).
# The EXIT trap's final sync below then mirrors plots/ to GCS with the rest of the run root.
if [ "${CAGE_AUTO_PLOTS:-1}" != "0" ]; then
  echo "[cage] generating plots -> $RUN_ROOT/plots/"
  python3 "$SCRIPT_DIR/../4_analysis/generate_plots.py" --results-dir "$CAGE_RUN_ROOT" --plots-dir "$CAGE_RUN_ROOT/plots" \
    > logs/generate_plots.log 2>&1 || echo "[cage] WARNING: plot generation failed (see logs/generate_plots.log)"
fi

echo "[cage] suite complete; results are in $SYNC_DIR/ and mirrored to GCS."
