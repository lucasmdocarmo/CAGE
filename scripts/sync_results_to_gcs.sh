#!/bin/bash
# Mirror a local results directory to the durable CAGE GCS bucket.
#
# Usage:
#   scripts/sync_results_to_gcs.sh [LOCAL_DIR] [BUCKET]
#     LOCAL_DIR  directory to sync (default: analysis)
#     BUCKET     gs://bucket or bucket name (default: $CAGE_RESULTS_BUCKET, else
#                gs://<project>-cage-results derived from the GCP project)
#
# The bucket is created by terraform/gcp (versioned, force_destroy=false) and the
# VM's default service account is granted roles/storage.objectAdmin on it.
set -euo pipefail

LOCAL_DIR="${1:-analysis}"
BUCKET="${2:-${CAGE_RESULTS_BUCKET:-}}"

if [ -z "$BUCKET" ]; then
  # Derive the project id: env var, then GCE metadata server, then gcloud config.
  PROJECT="${GOOGLE_CLOUD_PROJECT:-}"
  if [ -z "$PROJECT" ]; then
    PROJECT="$(curl -s -H 'Metadata-Flavor: Google' \
      http://metadata.google.internal/computeMetadata/v1/project/project-id 2>/dev/null || true)"
  fi
  if [ -z "$PROJECT" ]; then
    PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
  fi
  if [ -z "$PROJECT" ]; then
    echo "ERROR: cannot determine GCP project. Pass the bucket explicitly or set CAGE_RESULTS_BUCKET." >&2
    exit 1
  fi
  BUCKET="gs://${PROJECT}-cage-results"
fi
case "$BUCKET" in gs://*) ;; *) BUCKET="gs://${BUCKET}" ;; esac

if [ ! -d "$LOCAL_DIR" ]; then
  echo "[cage] nothing to sync yet (no $LOCAL_DIR/)"; exit 0
fi

echo "[cage] syncing $LOCAL_DIR -> $BUCKET/$LOCAL_DIR"
gsutil -m rsync -r "$LOCAL_DIR" "$BUCKET/$LOCAL_DIR"
