#!/bin/bash
# Mirror a local results directory to the durable CAGE GCS bucket.
#
# Usage:
#   scripts/5_observability/sync_results_to_gcs.sh [LOCAL_DIR] [BUCKET] [REMOTE_SUBPATH]
#     LOCAL_DIR       directory to sync (default: results -- the standardized run tree
#                     results/<phase>/<run-id>/...; callers usually pass a specific run root)
#     BUCKET          gs://bucket or bucket name (default: $CAGE_RESULTS_BUCKET, else
#                     gs://<project>-cage-results derived from the GCP project)
#     REMOTE_SUBPATH  path under the bucket to mirror into (default: LOCAL_DIR). Lets
#                     collect_logs.sh sync logs/ to vm_logs/<hostname>/ so multiple VMs
#                     do not collide on log filenames.
#
# On SUCCESS it touches .agent/last_gcs_sync_ok (epoch + src + dest inside; the file
# mtime is the machine-readable "last good sync" marker). watch_campaign.sh reads that
# marker to compute GCS sync lag without ever calling gcloud. Marker write is
# best-effort (a read-only checkout must never break a sync).
#
# The bucket is created by terraform/gcp (versioned, force_destroy=false) and the
# VM's default service account is granted roles/storage.objectAdmin on it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=scripts/lib/_common.sh
source "$PROJECT_DIR/scripts/lib/_common.sh"

LOCAL_DIR="${1:-results}"
BUCKET="${2:-${CAGE_RESULTS_BUCKET:-}}"
REMOTE_SUBPATH="${3:-$LOCAL_DIR}"

require_cmd gsutil "install the Google Cloud SDK; NOTHING was synced"

if [ -z "$BUCKET" ]; then
  # Derive the project id: env var, then GCE metadata server, then gcloud config.
  PROJECT="${GOOGLE_CLOUD_PROJECT:-}"
  if [ -z "$PROJECT" ]; then
    PROJECT="$(curl -s --max-time 5 -H 'Metadata-Flavor: Google' \
      http://metadata.google.internal/computeMetadata/v1/project/project-id 2>/dev/null || true)"
  fi
  if [ -z "$PROJECT" ]; then
    PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
  fi
  if [ -z "$PROJECT" ] || [ "$PROJECT" = "(unset)" ]; then
    die "cannot determine GCP project. Pass the bucket explicitly or set CAGE_RESULTS_BUCKET."
  fi
  BUCKET="gs://${PROJECT}-cage-results"
fi
case "$BUCKET" in gs://*) ;; *) BUCKET="gs://${BUCKET}" ;; esac

if [ ! -d "$LOCAL_DIR" ]; then
  echo "[cage] nothing to sync yet (no $LOCAL_DIR/)"; exit 0
fi

echo "[cage] syncing $LOCAL_DIR -> $BUCKET/$REMOTE_SUBPATH"
# -c: compare by checksum, not just size+mtime, so a file that was truncated mid-write on
# a prior pass gets re-uploaded once complete (avoids a partial upload becoming permanent).
gsutil -m rsync -c -r "$LOCAL_DIR" "$BUCKET/$REMOTE_SUBPATH"

# Success marker (best-effort): mtime == last time ANY sync path completed cleanly.
{
  mkdir -p "$PROJECT_DIR/.agent" && \
  printf 'epoch=%s\nutc=%s\nsrc=%s\ndest=%s\n' \
    "$(date +%s)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$LOCAL_DIR" "$BUCKET/$REMOTE_SUBPATH" > "$PROJECT_DIR/.agent/last_gcs_sync_ok"
} 2>/dev/null || true
