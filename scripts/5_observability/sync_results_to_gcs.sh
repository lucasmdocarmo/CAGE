#!/bin/bash
# DEPRECATED NAME — forwarding shim (task #137, 2026-08-18).
#
# The canonical, provider-neutral sync is scripts/5_observability/sync_results.sh
# (RunPod is the PRIMARY provider per FINAL SCOPE v2; GCS is one backend of
# scripts/lib/transport.sh — gs:// | s3:// | ssh:// | file:// all work). This
# shim keeps runbooks and older automation working: same arguments, same
# environment contract, same exit code — it just forwards.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=scripts/lib/_common.sh
source "$PROJECT_DIR/scripts/lib/_common.sh"

warn "sync_results_to_gcs.sh is a DEPRECATED name -> forwarding to sync_results.sh (provider-neutral, task #137)"
exec bash "$SCRIPT_DIR/sync_results.sh" "$@"
