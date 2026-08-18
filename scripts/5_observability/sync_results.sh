#!/bin/bash
# Mirror a local directory to the durable off-box backup target — provider-
# neutral (task #137, finding J4). Canonical name since 2026-08-18; the old
# sync_results_to_gcs.sh remains as a deprecated forwarding shim.
#
# Usage:
#   scripts/5_observability/sync_results.sh [LOCAL_DIR] [TARGET] [REMOTE_SUBPATH]
#     LOCAL_DIR       directory to sync (default: results — the standardized run
#                     tree results/<phase>/<run-id>/...; callers usually pass a
#                     specific run root)
#     TARGET          gs://bucket | s3://bucket | ssh://[user@]host/path |
#                     file:///path | /path  (bare names = legacy gs:// buckets).
#                     Default: $CAGE_BACKUP_TARGET, else $CAGE_RESULTS_BUCKET,
#                     else — ON A GCP BOX ONLY — the metadata-derived
#                     gs://<project>-cage-results (the GCS-port convenience).
#     REMOTE_SUBPATH  path under TARGET to mirror into (default: LOCAL_DIR).
#                     Lets collect_logs.sh sync logs/ to vm_logs/<hostname>/ so
#                     multiple boxes do not collide on log filenames.
#
# LOUD DEGRADATION (J4): with NO resolvable target this script DIES — the old
# behavior (metadata-hard derivation + swallowed failures) let a whole run
# finish with an EMPTY bucket silently. CAGE_ALLOW_NO_BACKUP=1 is the only
# skip, and it is announced (cloud_run.sh records the override in the run root).
#
# MARKERS (J4 shared-marker masking fix): success/failure markers are
# PER-BACKEND — .agent/last_sync_ok_<backend> / .agent/last_sync_fail_<backend>
# (epoch + src + dest inside; file mtime is the machine-readable timestamp) —
# so a succeeding log sync can no longer mask a failing results sync on a
# different transport. The legacy .agent/last_gcs_sync_ok is still written on
# gcs success (watch_campaign.sh back-compat). A FAILED transfer writes the
# failure marker AND exits nonzero — there is NO '|| true' on this path.
# Marker writes themselves warn (never die): a read-only checkout must not
# break a sync. CAGE_AGENT_DIR overrides the marker directory (tests).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=scripts/lib/_common.sh
source "$PROJECT_DIR/scripts/lib/_common.sh"
# shellcheck source=scripts/lib/transport.sh
source "$PROJECT_DIR/scripts/lib/transport.sh"

LOCAL_DIR="${1:-results}"
TARGET="${2:-${CAGE_BACKUP_TARGET:-${CAGE_RESULTS_BUCKET:-}}}"
REMOTE_SUBPATH="${3:-$LOCAL_DIR}"
AGENT_DIR="${CAGE_AGENT_DIR:-$PROJECT_DIR/.agent}"

# Legacy bare bucket name (CAGE_RESULTS_BUCKET convention) -> gs://
case "$TARGET" in
  "" | gs://* | s3://* | ssh://* | file://* | /*) ;;
  *) TARGET="gs://${TARGET}" ;;
esac
if [ -z "$TARGET" ]; then
  TARGET="$(transport_default_target)"
fi
if [ -z "$TARGET" ]; then
  if [ "${CAGE_ALLOW_NO_BACKUP:-0}" = "1" ]; then
    warn "no backup target and CAGE_ALLOW_NO_BACKUP=1 -> SKIPPING off-box sync (data exists ONLY on this box until pulled)"
    exit 0
  fi
  die "no backup target (J4): set CAGE_BACKUP_TARGET (gs://|s3://|ssh://[user@]host/path|file:///path) or CAGE_RESULTS_BUCKET — refusing to silently skip off-box persistence (export CAGE_ALLOW_NO_BACKUP=1 to explicitly accept none)"
fi

BACKEND="$(transport_resolve "$TARGET")"
DEST="$(transport_join "$TARGET" "$REMOTE_SUBPATH")"

if [ ! -d "$LOCAL_DIR" ]; then
  echo "[cage] nothing to sync yet (no $LOCAL_DIR/)"
  exit 0
fi

echo "[cage] syncing $LOCAL_DIR -> $DEST (backend: $BACKEND)"
mkdir -p "$AGENT_DIR" 2>/dev/null || warn "cannot create marker dir $AGENT_DIR (read-only checkout?) — sync proceeds, markers skipped"

_rc=0
transport_push "$LOCAL_DIR" "$DEST" || _rc=$?
if [ "$_rc" -ne 0 ]; then
  # Failure marker FIRST (distinct per backend), then die: the failure must be
  # both machine-readable and loud — never masked (finding J4).
  {
    printf 'epoch=%s\nutc=%s\nsrc=%s\ndest=%s\nbackend=%s\nrc=%s\n' \
      "$(date +%s)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "$LOCAL_DIR" "$DEST" "$BACKEND" "$_rc" > "$AGENT_DIR/last_sync_fail_${BACKEND}"
  } 2>/dev/null || warn "cannot write failure marker $AGENT_DIR/last_sync_fail_${BACKEND}"
  die "sync FAILED (backend=$BACKEND rc=$_rc): $LOCAL_DIR -> $DEST"
fi

# Success marker (per backend; mtime == last time THIS transport completed cleanly).
{
  printf 'epoch=%s\nutc=%s\nsrc=%s\ndest=%s\nbackend=%s\n' \
    "$(date +%s)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$LOCAL_DIR" "$DEST" "$BACKEND" > "$AGENT_DIR/last_sync_ok_${BACKEND}"
} 2>/dev/null || warn "cannot write success marker $AGENT_DIR/last_sync_ok_${BACKEND} (read-only checkout?)"
if [ "$BACKEND" = "gcs" ]; then
  # Legacy marker name kept for watch_campaign.sh's lag computation.
  cp "$AGENT_DIR/last_sync_ok_gcs" "$AGENT_DIR/last_gcs_sync_ok" 2>/dev/null \
    || warn "cannot refresh legacy marker $AGENT_DIR/last_gcs_sync_ok"
fi
