#!/bin/bash
# CAGE live-run watcher (laptop side of the observability GCS bus).
#
# Pulls the run's mirrored artifacts (results, logs, and the observability snapshots the
# sidecar writes) FROM the durable GCS bucket every INTERVAL seconds, and prints a one-line
# progress status parsed from observability/snapshots/latest.json. Pure pull -- no port is
# opened on the VM, and `gsutil rsync` transfers only deltas, so egress ~= artifact size once.
#
# Usage:
#   bash scripts/watch_run.sh [BUCKET] [LOCAL_DIR] [INTERVAL]
#     BUCKET     gs://... (default: $CAGE_RESULTS_BUCKET, else gs://<gcloud-project>-cage-results)
#     LOCAL_DIR  local mirror dir (default: ./phase2_archive)
#     INTERVAL   seconds between pulls (default: 30)
#   Ctrl-C to stop. The snapshot PNG is at <LOCAL_DIR>/observability/snapshots/latest.png.
set -uo pipefail

BUCKET="${1:-${CAGE_RESULTS_BUCKET:-}}"
LOCAL_DIR="${2:-./phase2_archive}"
INTERVAL="${3:-30}"

if [ -z "$BUCKET" ]; then
  _proj="$(gcloud config get-value project 2>/dev/null)"
  if [ -z "$_proj" ] || [ "$_proj" = "(unset)" ]; then
    echo "ERROR: no bucket given and no gcloud project set." >&2
    echo "  usage: bash scripts/watch_run.sh gs://YOUR-BUCKET [local_dir] [interval]" >&2
    exit 1
  fi
  BUCKET="gs://${_proj}-cage-results"
fi

if ! command -v gsutil >/dev/null 2>&1; then
  echo "ERROR: gsutil not found. Install the Google Cloud SDK." >&2
  exit 1
fi

mkdir -p "$LOCAL_DIR"
echo "[watch] pulling $BUCKET  ->  $LOCAL_DIR  every ${INTERVAL}s (Ctrl-C to stop)"
echo "[watch] snapshot PNG will appear at: $LOCAL_DIR/observability/snapshots/latest.png"

_status_line() {
  # Print progress from the newest snapshot, if present. Prefer python3; degrade to a note.
  local latest="$LOCAL_DIR/observability/snapshots/latest.json"
  [ -f "$latest" ] || { echo "[watch]   (no snapshot yet)"; return; }
  if command -v python3 >/dev/null 2>&1; then
    python3 - "$latest" <<'PY' 2>/dev/null || echo "[watch]   (snapshot present; parse skipped)"
import json, sys
s = json.load(open(sys.argv[1]))
g = s.get("gpu") or {}; p = s.get("progress") or {}
print(f"[watch]   t+{s.get('elapsed_s','?')}s | GPU mem {g.get('mem_used_pct','?')}% "
      f"util {g.get('util_pct','?')}% {g.get('temp_c','?')}C | "
      f"done {p.get('completed','?')} (baselines {p.get('baselines_done','?')}"
      f"{', ' + str(p.get('pct')) + '%' if p.get('pct') is not None else ''}) "
      f"active={p.get('active_baseline','?')}")
PY
  else
    echo "[watch]   snapshot present (install python3 for a parsed status line)"
  fi
}

trap 'echo; echo "[watch] stopped."; exit 0' INT TERM
while true; do
  gsutil -m rsync -r "$BUCKET" "$LOCAL_DIR" >/dev/null 2>&1 || echo "[watch]   (rsync retry next tick)"
  _status_line
  sleep "$INTERVAL"
done
