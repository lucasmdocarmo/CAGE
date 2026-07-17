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
#   gcs_backup_daemon.sh start [phase_dir]   # default phase_dir: results/<CAGE_PHASE|phase2>
#   gcs_backup_daemon.sh stop  [phase_dir]
#
# ENV
#   CAGE_RESULTS_BUCKET   REQUIRED. gs://bucket (or bare name). LOUD no-op if unset.
#   CAGE_BACKUP_INTERVAL  seconds between syncs (default 300)
#   CAGE_PHASE            phase segment (default phase2)
#
# The remote layout is $BUCKET/results/<phase>/... so teardown_vm.sh (which pulls
# $BUCKET/results -> results) reconstructs the exact local tree. No --delete is used
# anywhere, so this is safe to run concurrently with cloud_run.sh's own syncer.
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR" || exit 1

ACTION="${1:-}"
PHASE_DIR="${2:-results/${CAGE_PHASE:-phase2}}"
INTERVAL="${CAGE_BACKUP_INTERVAL:-300}"
BUCKET="${CAGE_RESULTS_BUCKET:-}"
PIDF=".agent/gcs_backup.pid"
LOGF=".agent/gcs_backup.log"
mkdir -p .agent

# One sync of the whole phase tree; REMOTE_SUBPATH == local path so the bucket layout
# mirrors the local layout ($BUCKET/results/<phase>/...), matching teardown's pull.
sync_once() { bash "$SCRIPT_DIR/sync_results_to_gcs.sh" "$PHASE_DIR" "$BUCKET" "$PHASE_DIR"; }

case "$ACTION" in
  start)
    if [ -z "$BUCKET" ]; then
      echo "[gcs-backup] WARNING: CAGE_RESULTS_BUCKET is unset -> NO cloud backup this run" >&2
      echo "[gcs-backup]          (data will exist ONLY on the VM disk until the local pull)." >&2
      exit 0
    fi
    case "$BUCKET" in gs://*) ;; *) BUCKET="gs://$BUCKET" ;; esac
    if ! gcloud storage ls "$BUCKET" >/dev/null 2>&1; then
      echo "[gcs-backup] WARNING: bucket $BUCKET is not reachable -> NO cloud backup this run." >&2
      echo "[gcs-backup]          Create it / fix the VM SA's storage.objectAdmin, then restart." >&2
      exit 0
    fi
    # Idempotent: replace any prior daemon.
    if [ -f "$PIDF" ]; then kill "$(cat "$PIDF")" 2>/dev/null || true; rm -f "$PIDF"; fi
    ( while true; do sync_once >>"$LOGF" 2>&1 || true; sleep "$INTERVAL"; done ) &
    echo $! > "$PIDF"
    echo "[gcs-backup] daemon up (pid $(cat "$PIDF")): $PHASE_DIR -> $BUCKET/$PHASE_DIR every ${INTERVAL}s"
    ;;
  stop)
    if [ -f "$PIDF" ]; then kill "$(cat "$PIDF")" 2>/dev/null || true; rm -f "$PIDF"; fi
    if [ -n "$BUCKET" ]; then
      case "$BUCKET" in gs://*) ;; *) BUCKET="gs://$BUCKET" ;; esac
      echo "[gcs-backup] final authoritative sync..."
      sync_once || echo "[gcs-backup] WARNING: final sync returned nonzero" >&2
    fi
    echo "[gcs-backup] daemon stopped"
    ;;
  *)
    echo "usage: $0 start|stop [phase_dir]" >&2
    exit 2
    ;;
esac
