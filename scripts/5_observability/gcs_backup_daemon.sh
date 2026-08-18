#!/bin/bash
# =============================================================================
# Full-run GCS backup daemon — a redundant cloud copy of EVERY cell we grab.
# =============================================================================
# WHY THIS EXISTS
#   cloud_run.sh already mirrors its run root to GCS, but only during the CORE tree:
#   the lever trees (compression/speculative/envelope/kv_store), the scoring + stats
#   passes, and the whole memory sweep run OUTSIDE that syncer, so they were never
#   backed up. Worse, when CAGE_RESULTS_BUCKET was unset the sync targeted a default
#   bucket that did not exist and the `|| true` swallowed the failure SILENTLY -- a
#   full multi-dataset run finished with an EMPTY bucket and nobody noticed.
#
#   This daemon mirrors the ENTIRE results/<phase>/ tree (every run-id: squad, musique,
#   hotpotqa, memsweep, ...) to CAGE_RESULTS_BUCKET on a fixed interval for the whole
#   duration of the sweep, and fails LOUDLY (never silently) if the bucket is unset or
#   unreachable. `stop` kills the loop and does one final authoritative sync.
#
# USAGE
#   gcs_backup_daemon.sh start  [phase_dir]  # default phase_dir: results/<CAGE_PHASE|phase2>
#   gcs_backup_daemon.sh stop   [phase_dir]  # stop loop (any scope if arg omitted) + final sync
#   gcs_backup_daemon.sh status [phase_dir]  # running check + last sync line; exit 0=running,1=not
#
# ENV
#   CAGE_BACKUP_TARGET    provider-neutral target: gs://|s3://|ssh://|file://
#                         (task #137; RunPod-primary). Takes precedence.
#   CAGE_RESULTS_BUCKET   legacy GCS spelling: gs://bucket (or bare name).
#                         With NEITHER set, `start` DIES (J4) unless
#                         CAGE_ALLOW_NO_BACKUP=1 explicitly accepts no backup.
#   CAGE_BACKUP_INTERVAL  seconds between syncs (default 300)
#   CAGE_PHASE            phase segment (default phase2)
#
# STATE  .agent/daemons/<scope>/gcs_backup.{pid,log}, scope = sanitized phase_dir, so
# each watched tree gets its own daemon and `start`/`stop` with the SAME phase_dir
# address the same instance. `start` is idempotent (live-pid check). The loop runs in
# its own session (setsid) so it survives the parent shell / `ssh --command` exiting,
# removes its pidfile on TERM/INT via an EXIT trap, and sleeps interruptibly
# (`sleep & wait`) so `stop` takes effect immediately, not after up to INTERVAL s.
#
# The remote layout is $BUCKET/results/<phase>/... so teardown_vm.sh (which pulls
# $BUCKET/results -> results) reconstructs the exact local tree. No --delete is used
# anywhere, so this is safe to run concurrently with cloud_run.sh's own syncer.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=scripts/lib/_common.sh
source "$PROJECT_DIR/scripts/lib/_common.sh"
# shellcheck source=scripts/lib/transport.sh
source "$PROJECT_DIR/scripts/lib/transport.sh"
cd "$PROJECT_DIR" || die "cannot cd to $PROJECT_DIR"

ACTION="${1:-}"
PHASE_DIR="${2:-results/${CAGE_PHASE:-phase2}}"
INTERVAL="${CAGE_BACKUP_INTERVAL:-300}"
# Provider-neutral (task #137): CAGE_BACKUP_TARGET (gs://|s3://|ssh://|file://)
# is primary; CAGE_RESULTS_BUCKET is the legacy GCS spelling.
BUCKET="${CAGE_BACKUP_TARGET:-${CAGE_RESULTS_BUCKET:-}}"
SYNC_SH="$SCRIPT_DIR/sync_results.sh"

# Legacy bare bucket name -> gs:// (scheme'd targets pass through untouched).
normalize_target() {
  case "$1" in
    "" | gs://* | s3://* | ssh://* | file://* | /*) printf '%s\n' "$1" ;;
    *) printf 'gs://%s\n' "$1" ;;
  esac
}

# Run-scoped state dir: one daemon per watched tree.
SCOPE="$(printf '%s' "$PHASE_DIR" | tr -c 'A-Za-z0-9_.-' '_')"
STATE_DIR="$PROJECT_DIR/.agent/daemons/$SCOPE"
PIDF="$STATE_DIR/gcs_backup.pid"
LOGF="$STATE_DIR/gcs_backup.log"
mkdir -p "$STATE_DIR"

# Local prefix overrides of _common.sh's log/warn (sourced above): the daemon's
# output lines are grepped by operators as "[gcs-backup]", so the prefix stays.
log()  { printf '[gcs-backup] %s\n' "$*"; }
warn() { printf '[gcs-backup] WARNING: %s\n' "$*" >&2; }

# pid_alive <pidfile> -> 0 iff pidfile holds a live pid THAT IS OUR LOOP.
# Identity = the pidfile path itself: the loop is launched with $PIDF as a positional
# arg, so it appears verbatim on the loop's command line. A recycled pid (VM reboot /
# PID wraparound, finding J8) fails the ps-command match and is treated as dead.
pid_alive() {
  pidfile_alive "$1" "$1"
}

stop_pidfile() {  # stop_pidfile <pidfile> -> stop the daemon it names (idempotent)
  local pf="$1" pid
  [ -f "$pf" ] || return 0
  pid="$(cat "$pf" 2>/dev/null || true)"
  # pid_alive (identity-checked) guards the kill: NEVER kill a recycled pid that
  # merely appears in a stale pidfile (finding J8).
  if [ -n "$pid" ] && pid_alive "$pf"; then
    kill "$pid" 2>/dev/null || true
    # Bounded wait (<=5s) for the loop's EXIT trap to remove its pidfile.
    local i
    for i in 1 2 3 4 5; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    log "stopped loop pid $pid ($pf)"
  fi
  rm -f "$pf"
}

# One sync of the whole phase tree; REMOTE_SUBPATH == local path so the bucket layout
# mirrors the local layout ($BUCKET/results/<phase>/...), matching teardown's pull.
sync_once() { bash "$SYNC_SH" "$PHASE_DIR" "$BUCKET" "$PHASE_DIR"; }

# The detached loop body (runs under setsid in its own session). Args are passed
# POSITIONALLY -- never interpolated into the -c string -- so paths with spaces or
# quotes cannot inject. The pidfile is removed by the EXIT trap on ANY exit path;
# `sleep & wait` makes TERM/INT take effect immediately instead of after INTERVAL.
# shellcheck disable=SC2016  # single-quoted on purpose: expands inside the child
LOOP_BODY='
  pidf="$1" sync_sh="$2" phase_dir="$3" bucket="$4" logf="$5" interval="$6"
  echo "$$" > "$pidf"
  _sleep_pid=""
  cleanup() { [ -n "$_sleep_pid" ] && kill "$_sleep_pid" 2>/dev/null; rm -f "$pidf"; }
  trap cleanup EXIT
  trap "exit 143" TERM
  trap "exit 130" INT
  while true; do
    bash "$sync_sh" "$phase_dir" "$bucket" "$phase_dir" >>"$logf" 2>&1 \
      || echo "[gcs-backup] $(date -u +%Y-%m-%dT%H:%M:%SZ) sync FAILED (will retry in ${interval}s)" >>"$logf"
    sleep "$interval" & _sleep_pid=$!
    wait "$_sleep_pid" || true
    _sleep_pid=""
  done
'

case "$ACTION" in
  start)
    if [ -z "$BUCKET" ]; then
      # J4 LOUD degradation: an unset target used to warn + exit 0, so a whole
      # sweep could run with ZERO off-box persistence and nothing said so.
      if [ "${CAGE_ALLOW_NO_BACKUP:-0}" = "1" ]; then
        warn "no backup target and CAGE_ALLOW_NO_BACKUP=1 -> NO off-box backup this run"
        warn "         (data will exist ONLY on the box disk until the local pull)."
        exit 0
      fi
      die "no backup target (J4): set CAGE_BACKUP_TARGET (gs://|s3://|ssh://|file://) or CAGE_RESULTS_BUCKET — or export CAGE_ALLOW_NO_BACKUP=1 to explicitly accept a run with NO off-box backup"
    fi
    BUCKET="$(normalize_target "$BUCKET")"
    case "$INTERVAL" in ''|*[!0-9]*) warn "CAGE_BACKUP_INTERVAL='$INTERVAL' not a positive integer"; exit 2 ;; esac
    if ! transport_ensure "$BUCKET"; then
      # A CONFIGURED-but-unreachable target is a misconfiguration, not a choice:
      # fail LOUD (never exit 0) so the launcher aborts instead of running bare.
      die "backup target $BUCKET is not reachable/usable (create the bucket / fix credentials / check CAGE_S3_ENDPOINT or CAGE_SSH_OPTS, then restart)"
    fi
    # Idempotent start: a live daemon for this scope is left alone.
    if pid_alive "$PIDF"; then
      log "already running (pid $(cat "$PIDF")): $PHASE_DIR -> $BUCKET/$PHASE_DIR"
      exit 0
    fi
    rm -f "$PIDF"   # stale pidfile from a dead loop
    # setsid detaches the loop into its OWN session so it survives the parent shell
    # exiting (a plain `( ) &` is SIGHUP-killed when an `ssh --command` session closes).
    # setsid is Linux-only (util-linux; the VMs have it): on macOS fall back LOUDLY to
    # nohup+disown, which also survives the parent shell (no ssh-session concern locally).
    # 200>&-: never let the detached loop inherit a caller's J3 run-lock fd
    # (acquire_run_lock in _common.sh) -- a daemon outliving its runner would
    # otherwise hold the run lock and block every resume.
    if command -v setsid >/dev/null 2>&1; then
      setsid bash -c "$LOOP_BODY" _ "$PIDF" "$SYNC_SH" "$PHASE_DIR" "$BUCKET" "$LOGF" "$INTERVAL" \
        >/dev/null 2>&1 200>&- &
    else
      warn "setsid not found (macOS?) -> falling back to nohup+disown for the loop"
      nohup bash -c "$LOOP_BODY" _ "$PIDF" "$SYNC_SH" "$PHASE_DIR" "$BUCKET" "$LOGF" "$INTERVAL" \
        >/dev/null 2>&1 200>&- &
      disown %+ 2>/dev/null || true
    fi
    sleep 1
    if pid_alive "$PIDF"; then
      log "daemon up (pid $(cat "$PIDF")): $PHASE_DIR -> $BUCKET/$PHASE_DIR every ${INTERVAL}s (log: $LOGF)"
    else
      warn "daemon FAILED to start (no live pid in $PIDF) -- check $LOGF"
      exit 1
    fi
    ;;
  stop)
    if [ -f "$PIDF" ]; then
      stop_pidfile "$PIDF"
    else
      # Back-compat: `stop` with no/other phase_dir must still find daemons started
      # under any scope (the pre-scoping behavior); stop every live one, loudly.
      found=0
      for pf in "$PROJECT_DIR"/.agent/daemons/*/gcs_backup.pid "$PROJECT_DIR"/.agent/gcs_backup.pid; do
        [ -f "$pf" ] || continue
        found=1
        stop_pidfile "$pf"
      done
      if [ "$found" -eq 0 ]; then log "no daemon pidfile found (nothing to stop)"; fi
    fi
    if [ -n "$BUCKET" ]; then
      BUCKET="$(normalize_target "$BUCKET")"
      log "final authoritative sync..."
      sync_once || warn "final sync returned nonzero"
    fi
    log "daemon stopped"
    ;;
  status)
    if pid_alive "$PIDF"; then
      log "RUNNING (pid $(cat "$PIDF")) scope=$SCOPE tree=$PHASE_DIR interval=${INTERVAL}s"
      if [ -f "$LOGF" ]; then printf '[gcs-backup] last log line: '; tail -n 1 "$LOGF"; fi
      exit 0
    fi
    log "NOT RUNNING (scope=$SCOPE, pidfile $PIDF)"
    exit 1
    ;;
  *)
    echo "usage: $0 start|stop|status [phase_dir]" >&2
    exit 2
    ;;
esac
