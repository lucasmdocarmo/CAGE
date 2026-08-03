#!/bin/bash
# Continuously mirror CAGE logs (and, by default, results) to GCS so an unexpected VM
# death (spot preemption, kernel panic, SSH loss) never loses logs. Use this with the
# run scripts that do NOT have their own sync loop (via scripts/lib/_log_guard.sh).
# cloud_run.sh already syncs on its own.
#
# Usage (ON the VM, launch once before/alongside a run):
#   nohup bash scripts/5_observability/log_sync_daemon.sh [INTERVAL_SECONDS] [SYNC_RESULTS] >/dev/null 2>&1 &
#   bash scripts/5_observability/log_sync_daemon.sh status   # running check; exit 0=running,1=not
#   bash scripts/5_observability/log_sync_daemon.sh stop     # stop the loop for this run scope
# Defaults: interval=120; SYNC_RESULTS=1 (also mirror results/; set 0 for logs only).
#
# The loop runs in the FOREGROUND (callers background it and may kill it by PID --
# scripts/lib/_log_guard.sh does exactly that; that contract is unchanged). It also
# writes a run-scoped pidfile .agent/daemons/<run-scope>/log_sync.pid (scope =
# $CAGE_RUN_ID, else .agent/cage_run_id, else "default") so `status`/`stop` and
# watch_campaign.sh can find it. Start is idempotent PER SCOPE: a second launch for
# the same run exits 0 without doubling the mirror. TERM/INT take effect immediately
# (`sleep & wait`), and the pidfile is removed on every exit path (EXIT trap).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --- run-scoped state ---------------------------------------------------------
RUN_SCOPE="${CAGE_RUN_ID:-}"
if [ -z "$RUN_SCOPE" ] && [ -f "$PROJECT_DIR/.agent/cage_run_id" ]; then
  RUN_SCOPE="$(cat "$PROJECT_DIR/.agent/cage_run_id" 2>/dev/null || true)"
fi
RUN_SCOPE="$(printf '%s' "${RUN_SCOPE:-default}" | tr -c 'A-Za-z0-9_.-' '_')"
STATE_DIR="$PROJECT_DIR/.agent/daemons/$RUN_SCOPE"
PIDF="$STATE_DIR/log_sync.pid"
mkdir -p "$STATE_DIR"

pid_alive() {
  local pid
  [ -f "$PIDF" ] || return 1
  pid="$(cat "$PIDF" 2>/dev/null || true)"
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

# --- subcommands (non-numeric first arg; numeric first arg = legacy interval) --
case "${1:-}" in
  status)
    if pid_alive; then
      echo "[log_sync_daemon] RUNNING (pid $(cat "$PIDF"), scope=$RUN_SCOPE)"
      exit 0
    fi
    echo "[log_sync_daemon] NOT RUNNING (scope=$RUN_SCOPE, pidfile $PIDF)"
    exit 1
    ;;
  stop)
    if pid_alive; then
      _pid="$(cat "$PIDF")"
      kill "$_pid" 2>/dev/null || true
      for _i in 1 2 3 4 5; do kill -0 "$_pid" 2>/dev/null || break; sleep 1; done
      kill -0 "$_pid" 2>/dev/null && kill -9 "$_pid" 2>/dev/null || true
      echo "[log_sync_daemon] stopped (pid $_pid, scope=$RUN_SCOPE)"
    else
      echo "[log_sync_daemon] not running (scope=$RUN_SCOPE); nothing to stop"
    fi
    rm -f "$PIDF"
    exit 0
    ;;
esac

INTERVAL="${1:-120}"
SYNC_RESULTS="${2:-1}"
case "$INTERVAL" in
  ''|*[!0-9]*) echo "[log_sync_daemon] ERROR: INTERVAL '$INTERVAL' is not a positive integer (or use: status|stop)" >&2; exit 2 ;;
esac

# Idempotent start (per run scope): never double-mirror the same run.
if pid_alive; then
  echo "[log_sync_daemon] already running (pid $(cat "$PIDF"), scope=$RUN_SCOPE); not starting a second loop"
  exit 0
fi
rm -f "$PIDF"   # stale pidfile from a dead loop
echo "$$" > "$PIDF"

_SLEEP_PID=""
cleanup() {
  if [ -n "$_SLEEP_PID" ]; then kill "$_SLEEP_PID" 2>/dev/null || true; fi
  rm -f "$PIDF"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

echo "[log_sync_daemon] every ${INTERVAL}s: collect_logs --light$([ "$SYNC_RESULTS" = 1 ] && echo ' + results sync') (pid $$, scope=$RUN_SCOPE)"
while true; do
  if [ "$SYNC_RESULTS" = "1" ]; then
    bash "$SCRIPT_DIR/sync_results_to_gcs.sh" "${CAGE_SYNC_DIR:-results}" >/dev/null 2>&1 || true
  fi
  bash "$SCRIPT_DIR/collect_logs.sh" --light >/dev/null 2>&1 || true
  sleep "$INTERVAL" & _SLEEP_PID=$!
  wait "$_SLEEP_PID" || true
  _SLEEP_PID=""
done
