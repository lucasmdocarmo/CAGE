#!/bin/bash
# SAFE teardown. Collect ALL logs + a final results sync to GCS, then VERIFY via a
# UNIQUE per-run sentinel that THIS collection actually completed, and only THEN delete
# the VM. Fails CLOSED (refuses to delete) if the sentinel is absent; --force overrides.
# Run from your WORKSTATION (uses gcloud). Works per-node in Phase 3.
#
# Why a sentinel and not a file count: SSH to these VMs drops AFTER the command runs, so
# the collect step's exit code is unreliable, and a bucket-wide file count can be
# satisfied by a PRIOR run's or ANOTHER node's logs. collect_logs.sh writes
# vm_logs/<host>/COLLECT_OK_<token> as its very last upload; we generate the token here,
# so finding that exact object proves this teardown's collection finished.
#
# Usage (flags accepted in any position):
#   scripts/teardown_vm.sh <instance> <zone> [--force]
# Env: CAGE_RESULTS_BUCKET (default gs://<project>-cage-results)
set -uo pipefail

VM=""; ZONE=""; FORCE=0
for a in "$@"; do
  case "$a" in
    --force) FORCE=1 ;;
    -*) echo "unknown flag: $a" >&2; exit 2 ;;
    *)
      if   [ -z "$VM" ];   then VM="$a"
      elif [ -z "$ZONE" ]; then ZONE="$a"
      else echo "unexpected extra arg: $a" >&2; exit 2; fi ;;
  esac
done
if [ -z "$VM" ] || [ -z "$ZONE" ]; then
  echo "usage: scripts/teardown_vm.sh <instance> <zone> [--force]" >&2
  exit 2
fi

PROJECT="$(gcloud config get-value project 2>/dev/null)"
BUCKET="${CAGE_RESULTS_BUCKET:-gs://${PROJECT}-cage-results}"
case "$BUCKET" in gs://*) ;; *) BUCKET="gs://$BUCKET" ;; esac
TOKEN="teardown_$(date -u +%Y%m%d_%H%M%S)_$$"   # unique to THIS teardown invocation

ssh_vm() {  # SSH stdout is unreliable on these VMs; we verify via GCS, not this output.
  gcloud compute ssh "$VM" --zone="$ZONE" --quiet --command="$1" \
    -- -o StrictHostKeyChecking=no -o ConnectTimeout=60 2>&1
}
num() { local v="${1//[^0-9]/}"; echo "${v:-0}"; }   # force a clean integer (fail-closed)

echo "=== SAFE TEARDOWN: $VM ($ZONE) | project=$PROJECT | bucket=$BUCKET ==="
echo "    success token: $TOKEN"

echo "[1/5] final results sync (analysis/) ..."
ssh_vm 'cd ~/CAGE && bash scripts/sync_results_to_gcs.sh analysis' | tail -3 || true

echo "[2/5] collecting ALL logs + forensics -> GCS (writes success sentinel) ..."
ssh_vm "cd ~/CAGE && CAGE_COLLECT_TOKEN='$TOKEN' bash scripts/collect_logs.sh" | tail -6 || true

echo "[3/5] verifying THIS run's sentinel in GCS (SSH stdout is not trusted) ..."
HITS=$(num "$(gcloud storage ls -r "$BUCKET/vm_logs/**" 2>/dev/null | grep -c "COLLECT_OK_${TOKEN}\$")")
RESN=$(num "$(gcloud storage ls -r "$BUCKET/analysis/**" 2>/dev/null | grep -c '[^/]$')")
echo "    sentinel COLLECT_OK_${TOKEN} present: $HITS    analysis objects: $RESN"

if [ "$HITS" -lt 1 ]; then
  if [ "$FORCE" -eq 1 ]; then
    echo "    !!! sentinel MISSING but --force given ($(date -u)). Deleting anyway; logs may be incomplete."
  else
    echo "[teardown] ABORT (fail-closed): this run's log sentinel is NOT in $BUCKET/vm_logs/." >&2
    echo "           Collection did not verifiably complete. Re-run, or pass --force to delete anyway." >&2
    exit 1
  fi
fi

echo "[4/5] deleting instance $VM ($ZONE) ... (cost-stopping action)"
gcloud compute instances delete "$VM" --zone="$ZONE" --quiet

echo "[5/5] confirming \$0 (no instances should remain) ..."
gcloud compute instances list 2>&1 | head -3
echo "TEARDOWN_COMPLETE  logs=$BUCKET/vm_logs/  results=$BUCKET/analysis/  sentinel=COLLECT_OK_${TOKEN}"
