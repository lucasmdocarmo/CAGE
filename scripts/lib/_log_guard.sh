# shellcheck shell=bash
# Sourceable log guard for standalone run scripts that have NO sync loop of their own
# (run_compression.sh, run_memory_sweep.sh; historically the now-retired
# scripts/deprecated/run_speculative_matrix.sh, and run_phase2_stats.sh).
# Source it near the top of such a script:
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_log_guard.sh"
#
# It starts log_sync_daemon.sh in the background (continuous results+logs mirror) and
# registers an EXIT trap that stops the daemon and does a final full collect, so a run
# launched outside cloud_run.sh is still protected against teardown/preemption/crash.
# No-op if CAGE_LOG_GUARD=0. Idempotent: a second source in the same shell is a no-op
# (two daemons would double-mirror and the second trap would orphan the first's pid).
# NOTE for callers that set their own EXIT trap: a later `trap ... EXIT` REPLACES this
# one -- chain it explicitly (see run_memory_sweep.sh: `type __lg_cleanup && __lg_cleanup`).
if [ "${CAGE_LOG_GUARD:-1}" != "0" ] && [ -z "${__LG_DAEMON:-}" ]; then
  __LG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  # 200>&-: never let the detached daemon inherit a caller's J3 run-lock fd
  # (acquire_run_lock in _common.sh) -- it would hold the lock past the runner's death.
  nohup bash "$__LG_DIR/../5_observability/log_sync_daemon.sh" "${CAGE_LOG_GUARD_INTERVAL:-120}" 1 >/dev/null 2>&1 200>&- &
  __LG_DAEMON=$!
  __lg_cleanup() {
    kill "$__LG_DAEMON" 2>/dev/null || true
    wait "$__LG_DAEMON" 2>/dev/null || true
    # Exit-time sync failures are ANNOUNCED, never `|| true`-swallowed (J4).
    bash "$__LG_DIR/../5_observability/sync_results.sh" "${CAGE_SYNC_DIR:-results}" >/dev/null 2>&1 \
      || printf '[log_guard] WARNING: exit-time results sync FAILED (rc=%s; see .agent/last_sync_fail_*)\n' "$?" >&2
    bash "$__LG_DIR/../5_observability/collect_logs.sh" >/dev/null 2>&1 \
      || printf '[log_guard] WARNING: exit-time log collection FAILED (rc=%s)\n' "$?" >&2
  }
  trap __lg_cleanup EXIT
  printf '[log_guard] continuous log+results mirror active (daemon pid %s); full collect on exit\n' "$__LG_DAEMON"
fi
