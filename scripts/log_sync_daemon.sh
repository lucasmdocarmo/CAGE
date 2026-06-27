#!/bin/bash
# Continuously mirror CAGE logs (and, by default, results) to GCS so an unexpected VM
# death (spot preemption, kernel panic, SSH loss) never loses logs. Use this with the
# run scripts that do NOT have their own sync loop (run_compression.sh, run_speculative_*,
# run_phase2_stats.sh, rerun_compressed_rag.sh). cloud_run.sh already syncs on its own.
#
# Usage (ON the VM, launch once before/alongside a run):
#   nohup bash scripts/log_sync_daemon.sh [INTERVAL_SECONDS] [SYNC_RESULTS] >/dev/null 2>&1 &
# Defaults: interval=120; SYNC_RESULTS=1 (also mirror analysis/; set 0 for logs only).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERVAL="${1:-120}"
SYNC_RESULTS="${2:-1}"

echo "[log_sync_daemon] every ${INTERVAL}s: collect_logs --light$([ "$SYNC_RESULTS" = 1 ] && echo ' + analysis sync')"
while true; do
  if [ "$SYNC_RESULTS" = "1" ]; then
    bash "$SCRIPT_DIR/sync_results_to_gcs.sh" analysis >/dev/null 2>&1 || true
  fi
  bash "$SCRIPT_DIR/collect_logs.sh" --light >/dev/null 2>&1 || true
  sleep "$INTERVAL"
done
