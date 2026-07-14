#!/bin/bash
# Apply the CAGE GCS lifecycle policy: auto-delete OBSERVABILITY artifacts older than N days
# so repeated sweeps don't accumulate storage (pairs with the project budget alert).
#
# It is PRECISELY scoped to the observability prefix (analysis/observability/). Real results
# -- analysis/phase1/, analysis/compression/, analysis/speculative_matrix/, analysis/all_results/
# -- and vm_logs/ do NOT match that prefix and are NEVER touched. A safety guard below refuses
# any prefix that doesn't contain "observability", so a typo cannot schedule deletion of results.
#
# Usage:
#   scripts/set_gcs_lifecycle.sh [BUCKET] [DAYS] [PREFIX]
#     BUCKET  gs://... or bare name (default: $CAGE_RESULTS_BUCKET, else gs://<project>-cage-results)
#     DAYS    age in days before deletion (default: 14)
#     PREFIX  object prefix to expire (default: analysis/observability/)
#
# Run this ONCE after the results bucket is (re)created. Idempotent -- safe to re-run. Requires
# roles/storage.admin on the bucket (bucket metadata write); the run-time VM SA only needs
# objectAdmin, so apply this from an admin identity, not from the VM.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

BUCKET="${1:-${CAGE_RESULTS_BUCKET:-}}"
DAYS="${2:-14}"
PREFIX="${3:-analysis/observability/}"

# --- Resolve the bucket the same way sync_results_to_gcs.sh does ---
if [ -z "$BUCKET" ]; then
  PROJECT="${GOOGLE_CLOUD_PROJECT:-}"
  [ -z "$PROJECT" ] && PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
  if [ -z "$PROJECT" ] || [ "$PROJECT" = "(unset)" ]; then
    echo "ERROR: no bucket given and no GCP project set. Pass the bucket or set CAGE_RESULTS_BUCKET." >&2
    exit 1
  fi
  BUCKET="gs://${PROJECT}-cage-results"
fi
case "$BUCKET" in gs://*) ;; *) BUCKET="gs://${BUCKET}" ;; esac

# --- SAFETY GUARD: never expire anything that isn't an observability prefix ---
case "$PREFIX" in
  *observability*) ;;
  *) echo "REFUSING: prefix '$PREFIX' does not contain 'observability'; this guard prevents a" >&2
     echo "          typo from scheduling deletion of real results. Aborting." >&2
     exit 1 ;;
esac

if ! command -v gcloud >/dev/null 2>&1; then
  echo "ERROR: gcloud not found. Install the Google Cloud SDK." >&2
  exit 1
fi
if ! gcloud storage buckets describe "$BUCKET" >/dev/null 2>&1; then
  echo "ERROR: bucket $BUCKET not found (create it first, then re-run this)." >&2
  exit 1
fi

# --- Choose the lifecycle config: committed default, or synthesise for non-default args ---
DEFAULT_CFG="$PROJECT_DIR/configs/gcs_lifecycle.json"
if [ "$DAYS" = "14" ] && [ "$PREFIX" = "analysis/observability/" ] && [ -f "$DEFAULT_CFG" ]; then
  CFG="$DEFAULT_CFG"
  CLEANUP_CFG=0
else
  CFG="$(mktemp -t cage_lifecycle.XXXXXX.json)"
  CLEANUP_CFG=1
  cat > "$CFG" <<JSON
{
  "rule": [
    {
      "action": { "type": "Delete" },
      "condition": {
        "age": ${DAYS},
        "matchesPrefix": ["${PREFIX}"]
      }
    }
  ]
}
JSON
fi
trap '[ "${CLEANUP_CFG:-0}" = "1" ] && rm -f "$CFG"' EXIT

echo "[cage] applying lifecycle to $BUCKET: delete objects under '${PREFIX}' older than ${DAYS} days"
gcloud storage buckets update "$BUCKET" --lifecycle-file="$CFG"

echo "[cage] verifying applied lifecycle:"
gcloud storage buckets describe "$BUCKET" --format="json(lifecycle_config)"
echo "[cage] done. (gsutil equivalent: gsutil lifecycle set '$CFG' '$BUCKET')"
