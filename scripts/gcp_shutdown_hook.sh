#!/bin/bash
# GCP shutdown-script for CAGE GPU VMs. GCP runs the instance's `shutdown-script` on ACPI
# soft-off, which fires on a SPOT PREEMPTION (~30s budget) and on a normal `instances
# delete`/`stop`. This guarantees results + logs are mirrored to GCS even when no operator
# is watching and the bash EXIT trap in a run script never gets to run.
#
# Install at VM creation:
#   gcloud compute instances create ... \
#     --metadata-from-file shutdown-script=scripts/gcp_shutdown_hook.sh
# Or attach to a running VM:
#   gcloud compute instances add-metadata <vm> --zone <zone> \
#     --metadata-from-file shutdown-script=scripts/gcp_shutdown_hook.sh
#
# It runs as root with a minimal environment, so it resolves the repo and the run user.
set -u
LOG=/var/log/cage_shutdown_hook.log
echo "=== cage shutdown hook fired $(date -u) ===" >> "$LOG" 2>&1

# Find the CAGE checkout (the run user's home, not root's).
for d in /home/*/CAGE /home/*/cage /root/CAGE /root/cage /opt/cage /opt/CAGE; do
  [ -d "$d" ] || continue
  USER_NAME="$(stat -c '%U' "$d" 2>/dev/null || echo root)"
  echo "[hook] using repo $d as $USER_NAME" >> "$LOG" 2>&1
  # Run as the owning user so gcloud/gsutil pick up its ADC + config.
  su - "$USER_NAME" -c "cd '$d' && bash scripts/sync_results_to_gcs.sh analysis && CAGE_COLLECT_TOKEN=shutdown_\$(date -u +%Y%m%d_%H%M%S) bash scripts/collect_logs.sh" >> "$LOG" 2>&1 || \
    ( cd "$d" && bash scripts/sync_results_to_gcs.sh analysis && bash scripts/collect_logs.sh ) >> "$LOG" 2>&1 || true
  break
done
echo "=== cage shutdown hook done $(date -u) ===" >> "$LOG" 2>&1
