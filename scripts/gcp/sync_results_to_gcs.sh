#!/bin/bash
# Order:     alongside stages 3-5 — DEPRECATED forwarding shim; canonical = 5_observability/sync_results.sh
# Objective: Forward legacy GCS-era callers verbatim (same args/env/exit code) to the provider-neutral sync_results.sh
# Cloud:     gcp
# DEPRECATED NAME — forwarding shim (task #137, 2026-08-18).
#
# The canonical, provider-neutral sync is scripts/5_observability/sync_results.sh
# (RunPod is the PRIMARY provider per FINAL SCOPE v2; GCS is one backend of
# scripts/lib/transport.sh — gs:// | s3:// | ssh:// | file:// all work). This
# shim keeps GCS-era runbooks and older GCP automation working: same arguments,
# same environment contract, same exit code — it just forwards. It lives under
# scripts/gcp/ because the legacy name is a GCS-port artifact.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=scripts/lib/_common.sh
source "$PROJECT_DIR/scripts/lib/_common.sh"

warn "sync_results_to_gcs.sh is a DEPRECATED name -> forwarding to sync_results.sh (provider-neutral, task #137)"
exec bash "$PROJECT_DIR/scripts/5_observability/sync_results.sh" "$@"
