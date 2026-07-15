#!/bin/bash
# SAFE teardown. Collect ALL logs + a final results sync to GCS, VERIFY via a UNIQUE per-run
# sentinel that THIS collection actually completed, PULL every result down to the local
# results/ folder, and only THEN delete the VM. Fails CLOSED (refuses to delete) if the
# sentinel is absent OR the local pull is incomplete; --force overrides both.
# Run from your WORKSTATION (uses gcloud). Works per-node in Phase 3.
#
# The local pull is step [4/6] and is deliberately BEFORE the delete: teardown is
# irreversible, so the run must exist in THREE places (VM + GCS + local) before anything
# is destroyed. Set CAGE_SKIP_LOCAL_PULL=1 to skip it (you then have no local copy).
#
# Why a sentinel and not a file count: SSH to these VMs drops AFTER the command runs, so
# the collect step's exit code is unreliable, and a bucket-wide file count can be
# satisfied by a PRIOR run's or ANOTHER node's logs. collect_logs.sh writes
# vm_logs/<host>/COLLECT_OK_<token> as its very last upload; we generate the token here,
# so finding that exact object proves this teardown's collection finished.
#
# Usage (flags accepted in any position):
#   scripts/6_teardown/teardown_vm.sh <instance> <zone> [--force]
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
  echo "usage: scripts/6_teardown/teardown_vm.sh <instance> <zone> [--force]" >&2
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

echo "[1/6] final results sync (results/) ..."
ssh_vm 'cd ~/CAGE && bash scripts/5_observability/sync_results_to_gcs.sh results' | tail -3 || true

echo "[2/6] collecting ALL logs + forensics -> GCS (writes success sentinel) ..."
ssh_vm "cd ~/CAGE && CAGE_COLLECT_TOKEN='$TOKEN' bash scripts/5_observability/collect_logs.sh" | tail -6 || true

echo "[3/6] verifying THIS run's sentinel in GCS (SSH stdout is not trusted) ..."
HITS=$(num "$(gcloud storage ls -r "$BUCKET/vm_logs/**" 2>/dev/null | grep -c "COLLECT_OK_${TOKEN}\$")")
RESN=$(num "$(gcloud storage ls -r "$BUCKET/results/**" 2>/dev/null | grep -c '[^/]$')")
echo "    sentinel COLLECT_OK_${TOKEN} present: $HITS    results objects: $RESN"

if [ "$HITS" -lt 1 ]; then
  if [ "$FORCE" -eq 1 ]; then
    echo "    !!! sentinel MISSING but --force given ($(date -u)). Deleting anyway; logs may be incomplete."
  else
    echo "[teardown] ABORT (fail-closed): this run's log sentinel is NOT in $BUCKET/vm_logs/." >&2
    echo "           Collection did not verifiably complete. Re-run, or pass --force to delete anyway." >&2
    exit 1
  fi
fi

# Pull EVERYTHING down BEFORE the irreversible delete, so the run exists in THREE places
# (VM + GCS + local) rather than two. Pulling *after* teardown only works if the GCS mirror
# happened to be complete; if it was not, the delete already destroyed the only other copy and
# there is nothing left to diagnose. Local is also free, needs no GPU, and is what every offline
# re-score/fix works against. Mirrors the bucket layout verbatim: results/<phase>/<run-id>/...
# (/results/ is gitignored, so a local copy never pollutes the repo.)
echo "[4/6] pulling ALL results GCS -> local results/ (3rd copy, BEFORE the delete) ..."
if [ "${CAGE_SKIP_LOCAL_PULL:-0}" = "1" ]; then
  echo "    SKIPPED (CAGE_SKIP_LOCAL_PULL=1) -- no local copy will exist after this delete"
else
  mkdir -p results
  PULL_OUT="$(gcloud storage rsync -r "$BUCKET/results" results 2>&1)"; PULL_RC=$?
  printf '%s\n' "$PULL_OUT" | tail -2
  LOCN=$(num "$(find results -type f 2>/dev/null | wc -l)")
  echo "    rsync exit: $PULL_RC    local results files: $LOCN    GCS results objects: $RESN"
  # Fail closed on EITHER signal: a non-zero rsync, or fewer local files than GCS objects.
  # (LOCN can legitimately EXCEED RESN when older runs live locally, hence -lt not -ne.)
  if [ "$PULL_RC" -ne 0 ] || [ "$LOCN" -lt "$RESN" ]; then
    if [ "$FORCE" -eq 1 ]; then
      echo "    !!! local pull INCOMPLETE (rc=$PULL_RC, $LOCN < $RESN) but --force given. Deleting anyway; DATA MAY BE LOST."
    else
      echo "[teardown] ABORT (fail-closed): the local results pull did not verifiably complete." >&2
      echo "           rsync exit=$PULL_RC, local files=$LOCN, GCS objects=$RESN." >&2
      echo "           Deleting the VM now risks losing data. Re-run, or --force to delete anyway." >&2
      exit 1
    fi
  else
    echo "    local copy verified -- safe to delete"
  fi
fi

echo "[5/6] deleting instance $VM ($ZONE) ... (cost-stopping action)"
gcloud compute instances delete "$VM" --zone="$ZONE" --quiet

echo "[6/6] confirming \$0 (no instances should remain) ..."
gcloud compute instances list 2>&1 | head -3
echo "TEARDOWN_COMPLETE  logs=$BUCKET/vm_logs/  results=$BUCKET/results/  sentinel=COLLECT_OK_${TOKEN}"
