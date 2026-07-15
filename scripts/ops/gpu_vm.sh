#!/usr/bin/env bash
# gpu_vm.sh — create / locate / sweep the CAGE L4 GPU VM.
#
# WHY THIS EXISTS
#   L4 capacity is genuinely scarce: on 2026-07-15 `g2-standard-8` returned
#   ZONE_RESOURCE_POOL_EXHAUSTED in EVERY zone tried (us-central1 a/b/c, us-east1 b/c/d,
#   us-east4-c, us-west1 a/b/c, us-west4-a, northamerica-northeast1-b) and only the smaller
#   `g2-standard-4` shape landed. Hunting zones by hand each time is slow and easy to get wrong.
#
#   It also LABELS every VM with agent-run=<id>. The label is the REAPING KEY: an unlabeled GPU VM
#   that outlives a session is the most expensive mistake possible here (~$0.55-0.75/hr, silently).
#
# USAGE
#   gpu_vm.sh create [name] [machine_type]   # zone-hunt + shape fallback + label; writes .agent/cage_zone
#   gpu_vm.sh ip     [name]
#   gpu_vm.sh zone   [name]
#   gpu_vm.sh sweep                          # PROVE $0: any labeled/unlabeled instances, disks, buckets
#
#   Deliberately NO delete: teardown goes through scripts/6_teardown/teardown_vm.sh, which
#   collects logs + syncs results and FAILS CLOSED on a missing sentinel before deleting.
#   Duplicating a delete path here would be a footgun.
#
# ENV
#   CAGE_RUN_LABEL   agent-run label value (default: cage-<UTC date>)
#   CAGE_IMAGE_FAMILY / CAGE_IMAGE_PROJECT  (default: common-cu129-ubuntu-2204-nvidia-580 /
#                    deeplearning-platform-release -- the pinned cu121-debian-11 image was RETIRED)
#   CAGE_DISK_GB     (default 200)
#   CAGE_ZONES       override the zone hunt list
#
# NOTE ON SHAPES: g2-standard-4 = 4 vCPU/16GB, g2-standard-8 = 8 vCPU/32GB; BOTH have 1x L4 (24GB).
#   CAGE's quality scoring (LettuceDetect/NLI/BERTScore/e5) runs on CPU while vLLM holds the GPU,
#   so vCPU count directly sets sweep wall-clock: the -4 shape scores ~2x slower than the -8.
#   Prefer -8 for a real sweep; -4 is fine for validation.

set -euo pipefail
export CLOUDSDK_CORE_DISABLE_PROMPTS=1

NAME_DEFAULT="cage-gpu"
IMAGE_FAMILY="${CAGE_IMAGE_FAMILY:-common-cu129-ubuntu-2204-nvidia-580}"
IMAGE_PROJECT="${CAGE_IMAGE_PROJECT:-deeplearning-platform-release}"
DISK_GB="${CAGE_DISK_GB:-200}"
RUN_LABEL="${CAGE_RUN_LABEL:-cage-$(date -u +%Y%m%d)}"
HOOK="scripts/5_observability/gcp_shutdown_hook.sh"

# Zones where G2/L4 exists. Ordered: bucket-local (us-central1) first, then wider US.
ZONES_DEFAULT="us-central1-a us-central1-b us-central1-c us-east1-b us-east1-c us-east1-d us-east4-c us-west1-a us-west1-b us-west1-c us-west4-a northamerica-northeast1-b"

die() { printf 'gpu_vm: %s\n' "$*" >&2; exit 1; }

_try() {  # $1=name $2=zone $3=machine -> 0 on success, else prints reason
  local out
  if out=$(gcloud compute instances create "$1" \
      --zone="$2" --machine-type="$3" \
      --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
      --boot-disk-size="${DISK_GB}GB" --boot-disk-type=pd-balanced \
      --maintenance-policy=TERMINATE \
      --scopes=https://www.googleapis.com/auth/cloud-platform \
      --labels=agent-run="$RUN_LABEL" \
      --metadata=install-nvidia-driver=True \
      --metadata-from-file=shutdown-script="$HOOK" 2>&1); then
    return 0
  fi
  printf '%s' "$out" | grep -oiE 'ZONE_RESOURCE_POOL_EXHAUSTED|Quota [A-Z_]+ exceeded|does not exist|not available' | head -1 || echo "other"
  return 1
}

cmd_create() {
  local name="${1:-$NAME_DEFAULT}" machine="${2:-}"
  [ -f "$HOOK" ] || die "shutdown hook missing at $HOOK (run from the repo root)"
  local shapes
  if [ -n "$machine" ]; then shapes="$machine"; else shapes="g2-standard-8 g2-standard-4"; fi

  # NOTE: literal/expanded lists in `for` -- do NOT rely on `for Z in $ZONES` under zsh, where an
  # unquoted var does NOT word-split: the loop silently runs ONCE with the whole string as one
  # bogus zone. (Bit us on 2026-07-15.) This script is bash (`#!/usr/bin/env bash`), so splitting
  # is correct here -- keep the shebang.
  local m z reason
  for m in $shapes; do
    echo "=== shape $m ==="
    for z in ${CAGE_ZONES:-$ZONES_DEFAULT}; do
      printf '  %-26s ... ' "$z"
      if reason=$(_try "$name" "$z" "$m"); then
        echo "CREATED"
        mkdir -p .agent; echo "$z" > .agent/cage_zone
        echo "=== up: $name ($m) in $z, label agent-run=$RUN_LABEL -- BILLING STARTED ==="
        gcloud compute instances list --filter="name=$name" \
          --format='table(name,zone,machineType.basename(),status,labels.agent-run,networkInterfaces[0].accessConfigs[0].natIP)'
        return 0
      fi
      echo "$reason"
    done
  done
  echo ">>> no zone/shape had L4 capacity -- nothing created, still \$0"
  return 1
}

cmd_zone() { local n="${1:-$NAME_DEFAULT}"; gcloud compute instances list --filter="name=$n" --format='value(zone.basename())'; }
cmd_ip()   { local n="${1:-$NAME_DEFAULT}"; gcloud compute instances list --filter="name=$n" --format='value(networkInterfaces[0].accessConfigs[0].natIP)'; }

# The reaping sweep. Run before declaring a session done -- PROVE it is empty, do not assume.
cmd_sweep() {
  echo "=== labeled agent-run instances (the reaping key) ==="
  gcloud compute instances list --filter='labels.agent-run:*' --format='table(name,zone,status,labels.agent-run)' 2>&1
  echo "=== ALL instances (catches unlabeled orphans) ==="
  gcloud compute instances list --format='table(name,zone,machineType.basename(),status)' 2>&1
  echo "=== disks (survive instance delete if --keep-disks was used) ==="
  gcloud compute disks list --format='table(name,zone,sizeGb,status)' 2>&1
  echo "=== buckets (storage bills too) ==="
  gcloud storage ls 2>&1 | head -10
  echo "=== VERDICT ==="
  local n d
  n=$(gcloud compute instances list --format='value(name)' 2>/dev/null | wc -l | tr -d ' ')
  d=$(gcloud compute disks list --format='value(name)' 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n" = "0" ] && [ "$d" = "0" ]; then
    echo "  COMPUTE AT \$0 (0 instances, 0 disks). Buckets above bill separately (pennies)."
  else
    echo "  !! STILL BILLING: $n instance(s), $d disk(s) -- teardown via scripts/6_teardown/teardown_vm.sh"
  fi
}

case "${1:-}" in
  create) shift; cmd_create "$@" ;;
  ip)     shift; cmd_ip   "$@" ;;
  zone)   shift; cmd_zone "$@" ;;
  sweep)  shift; cmd_sweep "$@" ;;
  *) sed -n '2,30p' "$0"; exit 2 ;;
esac
