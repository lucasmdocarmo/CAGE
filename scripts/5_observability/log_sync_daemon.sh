#!/bin/bash
# Continuously mirror CAGE logs (and, by default, results) to GCS so an unexpected VM
# death (spot preemption, kernel panic, SSH loss) never loses logs. Use this with the
# run scripts that do NOT have their own sync loop (run_compression.sh, run_speculative_*,
# run_phase2_stats.sh, rerun_compressed_rag.sh). cloud_run.sh already syncs on its own.
#
# Usage (ON the VM, launch once before/alongside a run):
#   nohup bash scripts/5_observability/log_sync_daemon.sh [INTERVAL_SECONDS] [SYNC_RESULTS] >/dev/null 2>&1 &
# Defaults: interval=120; SYNC_RESULTS=1 (also mirror results/; set 0 for logs only).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERVAL="${1:-120}"
SYNC_RESULTS="${2:-1}"

echo "[log_sync_daemon] every ${INTERVAL}s: collect_logs --light$([ "$SYNC_RESULTS" = 1 ] && echo ' + results sync')"
while true; do
  if [ "$SYNC_RESULTS" = "1" ]; then
    bash "$SCRIPT_DIR/../5_observability/sync_results_to_gcs.sh" "${CAGE_SYNC_DIR:-results}" >/dev/null 2>&1 || true
  fi
  bash "$SCRIPT_DIR/../5_observability/collect_logs.sh" --light >/dev/null 2>&1 || true
  sleep "$INTERVAL"
done
