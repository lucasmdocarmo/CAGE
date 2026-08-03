#!/bin/bash
# GCP shutdown-script for CAGE GPU VMs. GCP runs the instance's `shutdown-script` on ACPI
# soft-off, which fires on a SPOT PREEMPTION (~30s budget) and on a normal `instances
# delete`/`stop`. This guarantees results + logs are mirrored to GCS even when no operator
# is watching and the bash EXIT trap in a run script never gets to run.
#
# Install at VM creation:
#   gcloud compute instances create ... \
#     --metadata-from-file shutdown-script=scripts/5_observability/gcp_shutdown_hook.sh
# Or attach to a running VM:
#   gcloud compute instances add-metadata <vm> --zone <zone> \
#     --metadata-from-file shutdown-script=scripts/5_observability/gcp_shutdown_hook.sh
#
# It runs as ROOT with a MINIMAL environment (no profile, near-empty PATH), so it fixes
# PATH itself, resolves the repo and the run user, and runs the sync as that user (whose
# login shell carries the gcloud ADC + config). Every step is best-effort: with a ~30s
# preemption budget, a partial sync beats an aborted one, so nothing here may exit early.
set -u
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/snap/bin${PATH:+:$PATH}"
LOG=/var/log/cage_shutdown_hook.log
echo "=== cage shutdown hook fired $(date -u) ===" >> "$LOG" 2>&1

# Find the CAGE checkout (the run user's home, not root's).
for d in /home/*/CAGE /home/*/cage /root/CAGE /root/cage /opt/cage /opt/CAGE; do
  [ -d "$d" ] || continue
  USER_NAME="$(stat -c '%U' "$d" 2>/dev/null || echo root)"
  echo "[hook] using repo $d as $USER_NAME" >> "$LOG" 2>&1
  # Run as the owning user so gcloud/gsutil pick up its ADC + config. The paths matched by
  # the glob above contain no quotes/spaces, so embedding $d in the -c string is safe.
  su - "$USER_NAME" -c "cd '$d' && bash scripts/5_observability/sync_results_to_gcs.sh results && CAGE_COLLECT_TOKEN=shutdown_\$(date -u +%Y%m%d_%H%M%S) bash scripts/5_observability/collect_logs.sh" >> "$LOG" 2>&1 || \
    ( cd "$d" && bash scripts/5_observability/sync_results_to_gcs.sh results && bash scripts/5_observability/collect_logs.sh ) >> "$LOG" 2>&1 || true
  break
done
echo "=== cage shutdown hook done $(date -u) ===" >> "$LOG" 2>&1
