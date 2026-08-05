#!/bin/bash
# Continuous status logger for CAGE runs (Phase 2 / Phase 3 live tracking).
# Appends a compact one-line status snapshot every INTERVAL seconds to a timeline
# log, so progress is recorded at fine granularity without a held SSH connection.
#
# Usage:
#   nohup bash scripts/5_observability/run_status_logger.sh [RESULTS_DIR] [RUN_LOG] [OUT] [INTERVAL] &
#   bash scripts/5_observability/run_status_logger.sh status   # exit 0=running,1=not
#   bash scripts/5_observability/run_status_logger.sh stop
# Superseded by the observe_run.py sidecar (auto-launched by cloud_run.sh); kept as a lightweight
# manual logger. Defaults: results=$CAGE_RUN_ROOT/baselines, run.log=~/run.log, out=~/status_timeline.log, interval=20
#
# Runs the loop in the FOREGROUND (callers nohup+background it); writes a run-scoped
# pidfile .agent/daemons/<run-scope>/run_status_logger.pid (scope = $CAGE_RUN_ID, else
# .agent/cage_run_id, else "default"). Start is idempotent per scope; TERM/INT stop it
# immediately; the pidfile is removed on every exit path.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=scripts/lib/_common.sh
source "$PROJECT_DIR/scripts/lib/_common.sh"

RUN_SCOPE="${CAGE_RUN_ID:-}"
if [ -z "$RUN_SCOPE" ] && [ -f "$PROJECT_DIR/.agent/cage_run_id" ]; then
  RUN_SCOPE="$(cat "$PROJECT_DIR/.agent/cage_run_id" 2>/dev/null || true)"
fi
RUN_SCOPE="$(printf '%s' "${RUN_SCOPE:-default}" | tr -c 'A-Za-z0-9_.-' '_')"
STATE_DIR="$PROJECT_DIR/.agent/daemons/$RUN_SCOPE"
PIDF="$STATE_DIR/run_status_logger.pid"
mkdir -p "$STATE_DIR"

pid_alive() {
  local pid
  [ -f "$PIDF" ] || return 1
  pid="$(cat "$PIDF" 2>/dev/null || true)"
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

case "${1:-}" in
  status)
    if pid_alive; then
      echo "[run_status_logger] RUNNING (pid $(cat "$PIDF"), scope=$RUN_SCOPE)"
      exit 0
    fi
    echo "[run_status_logger] NOT RUNNING (scope=$RUN_SCOPE, pidfile $PIDF)"
    exit 1
    ;;
  stop)
    if pid_alive; then
      _pid="$(cat "$PIDF")"
      kill "$_pid" 2>/dev/null || true
      for _i in 1 2 3 4 5; do kill -0 "$_pid" 2>/dev/null || break; sleep 1; done
      kill -0 "$_pid" 2>/dev/null && kill -9 "$_pid" 2>/dev/null || true
      echo "[run_status_logger] stopped (pid $_pid, scope=$RUN_SCOPE)"
    else
      echo "[run_status_logger] not running (scope=$RUN_SCOPE); nothing to stop"
    fi
    rm -f "$PIDF"
    exit 0
    ;;
esac

RESULTS="${1:-${CAGE_RUN_ROOT:-$HOME/CAGE/results}/baselines}"
RUNLOG="${2:-$HOME/run.log}"
OUT="${3:-$HOME/status_timeline.log}"
INTERVAL="${4:-20}"
case "$INTERVAL" in
  ''|*[!0-9]*) echo "[run_status_logger] ERROR: INTERVAL '$INTERVAL' is not a positive integer" >&2; exit 2 ;;
esac

if pid_alive; then
  echo "[run_status_logger] already running (pid $(cat "$PIDF"), scope=$RUN_SCOPE); not starting a second loop"
  exit 0
fi
rm -f "$PIDF"
echo "$$" > "$PIDF"

_SLEEP_PID=""
cleanup() {
  if [ -n "$_SLEEP_PID" ]; then kill "$_SLEEP_PID" 2>/dev/null || true; fi
  rm -f "$PIDF"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

strip_ansi() { sed -E 's/\x1b\[[0-9;]*m//g'; }

echo "[run_status_logger] every ${INTERVAL}s: $RESULTS + $RUNLOG -> $OUT (pid $$, scope=$RUN_SCOPE)"
while true; do
  finished=$(ls "$RESULTS"/*/aggregated_metrics.json 2>/dev/null | wc -l | tr -d ' ') || finished=0
  current=$(grep ">>> Running baseline" "$RUNLOG" 2>/dev/null | tail -1 | sed -E 's/.*baseline:[[:space:]]*//' | strip_ansi) || current=""
  errors=$(grep -ciE "Traceback|Error running experiment|CUDA out of memory" "$RUNLOG" 2>/dev/null) || errors=0
  suite_done=$(grep -c "suite complete" "$RUNLOG" 2>/dev/null) || suite_done=0
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ') || gpu=""
  echo "$(date +%H:%M:%S) finished=${finished} running=${current:-none} errors=${errors} suite_done=${suite_done} gpu=[${gpu}]" >> "$OUT" || true
  sleep "$INTERVAL" & _SLEEP_PID=$!
  wait "$_SLEEP_PID" || true
  _SLEEP_PID=""
done
