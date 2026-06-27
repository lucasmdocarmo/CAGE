# shellcheck shell=bash
# Sourceable log guard for standalone run scripts that have NO sync loop of their own
# (run_compression.sh, run_speculative_matrix.sh, run_phase2_stats.sh, rerun_compressed_rag.sh).
# Source it near the top of such a script:
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_log_guard.sh"
#
# It starts log_sync_daemon.sh in the background (continuous results+logs mirror) and
# registers an EXIT trap that stops the daemon and does a final full collect, so a run
# launched outside cloud_run.sh is still protected against teardown/preemption/crash.
# No-op if CAGE_LOG_GUARD=0.
if [ "${CAGE_LOG_GUARD:-1}" != "0" ]; then
  __LG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  nohup bash "$__LG_DIR/log_sync_daemon.sh" "${CAGE_LOG_GUARD_INTERVAL:-120}" 1 >/dev/null 2>&1 &
  __LG_DAEMON=$!
  __lg_cleanup() {
    kill "$__LG_DAEMON" 2>/dev/null || true
    wait "$__LG_DAEMON" 2>/dev/null || true
    bash "$__LG_DIR/sync_results_to_gcs.sh" analysis >/dev/null 2>&1 || true
    bash "$__LG_DIR/collect_logs.sh" >/dev/null 2>&1 || true
  }
  trap __lg_cleanup EXIT
  echo "[log_guard] continuous log+results mirror active (daemon pid $__LG_DAEMON); full collect on exit"
fi
