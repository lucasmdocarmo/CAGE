#!/usr/bin/env bash
# Order:     provisioning bracket — before 1_setup; PLAN by default, creates only with --yes (the owner's recorded GO)
# Objective: Approval-gated RunPod pod provisioning with a server-side --terminate-after cost seatbelt + pod-ledger create event
# Cloud:     runpod
# provision_pod.sh — approval-gate-aware RunPod provisioning wrapper (CLI v2).
#
# DEFAULT = PLAN MODE: resolves and prints the full provisioning plan (shape,
# image, seatbelt, $/h when derivable, estimated total) then exits 0 WITHOUT
# creating anything. Creation happens ONLY with the explicit --yes flag — the
# owner's recorded GO (standing run-approval gate: no pod without approval).
#
# COST SEATBELT (owner-mandated; MyDocs/runpod-cli-reference.md §3): pods are
# created with a server-side --terminate-after deadline (default 12h) that
# survives a dead laptop. Disabling it requires the explicit
# --no-terminate-after flag and is announced LOUDLY.
#
# LEDGER: every real create appends one JSON line to
#   results/ops/pod_ledger.jsonl        (override: CAGE_POD_LEDGER)
# schema: {"ts_utc","pod_id","name","gpu_id","gpu_count",
#          "price_per_hour_usd"(number|null),"terminate_after"(string|null),
#          "purpose","event":"create"}
# teardown_pod.sh appends the matching {"event":"delete"} line;
# pod_status.sh / cost_report.sh consume the pairs.
#
# Plan mode touches the network AT MOST via one optional read-only
# `runpodctl gpu list` (price derivation; skipped/degraded to price=null with a
# note when the CLI or network is absent — or pass --price-per-hour).
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/_common.sh
source "$SCRIPT_DIR/../lib/_common.sh"
command -v runpodctl >/dev/null 2>&1 || PATH="$PATH:/opt/homebrew/bin:/usr/local/bin"

cleanup() { :; }  # no temp state; ledger appends are atomic-enough single lines
trap 'rc=$?; cleanup; exit $rc' EXIT
trap 'exit 130' INT TERM

usage() {
  printf 'usage: %s --gpu-id "<id from runpodctl gpu list>" [options] [--yes]\n' "$0" >&2
  printf 'Options: --name <s> --image <s> --gpu-count <n> --disk-gb <n> --volume-gb <n>\n' >&2
  printf '         --ports <s> --cloud-type SECURE|COMMUNITY --terminate-after <dur>\n' >&2
  printf '         --no-terminate-after --price-per-hour <f> --hours <f> --purpose <s>\n' >&2
  printf 'Default is PLAN mode: prints the plan and creates NOTHING. --yes = the owner GO.\n' >&2
  exit 2
}

NAME="cage-$(date -u +%Y%m%d-%H%M%S)"
IMAGE="${CAGE_POD_IMAGE:-runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu24.04}"
GPU_ID=""; GPU_COUNT="1"; DISK_GB="60"; VOL_GB="100"; PORTS="22/tcp"
CLOUD_TYPE="SECURE"; TERMINATE_AFTER="12h"; NO_SEATBELT=0
PRICE=""; HOURS=""; PURPOSE="unspecified"; YES=0
need_val() { [ "$#" -ge 2 ] || die "flag $1 requires a value"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu-id)             need_val "$@"; GPU_ID="$2";          shift 2 ;;
    --name)               need_val "$@"; NAME="$2";            shift 2 ;;
    --image)              need_val "$@"; IMAGE="$2";           shift 2 ;;
    --gpu-count)          need_val "$@"; GPU_COUNT="$2";       shift 2 ;;
    --disk-gb)            need_val "$@"; DISK_GB="$2";         shift 2 ;;
    --volume-gb)          need_val "$@"; VOL_GB="$2";          shift 2 ;;
    --ports)              need_val "$@"; PORTS="$2";           shift 2 ;;
    --cloud-type)         need_val "$@"; CLOUD_TYPE="$2";      shift 2 ;;
    --terminate-after)    need_val "$@"; TERMINATE_AFTER="$2"; shift 2 ;;
    --no-terminate-after) NO_SEATBELT=1;                       shift ;;
    --price-per-hour)     need_val "$@"; PRICE="$2";           shift 2 ;;
    --hours)              need_val "$@"; HOURS="$2";           shift 2 ;;
    --purpose)            need_val "$@"; PURPOSE="$2";         shift 2 ;;
    --yes)                YES=1;                               shift ;;
    -h|--help)            usage ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; usage ;;
  esac
done
[ -n "$GPU_ID" ] || { printf 'missing required --gpu-id\n' >&2; usage; }
printf '%s' "$GPU_COUNT" | grep -qE '^[1-9][0-9]*$' || die "--gpu-count must be a positive integer: $GPU_COUNT"
[ -z "$PRICE" ] || printf '%s' "$PRICE" | grep -qE '^[0-9]+(\.[0-9]+)?$' || die "--price-per-hour must be numeric: $PRICE"
[ -z "$HOURS" ] || printf '%s' "$HOURS" | grep -qE '^[0-9]+(\.[0-9]+)?$' || die "--hours must be numeric: $HOURS"

# --- seatbelt resolution: operator duration -> absolute RFC3339 UTC ---------
# runpodctl v2 `--terminate-after` takes an ABSOLUTE datetime, not a duration
# (its --help: "auto-terminate datetime (e.g., 2026-04-15T00:00:00Z)"). Until
# 2026-08-25 this script passed the raw "12h" default straight through, so the
# seatbelt was never a valid deadline: the create either fails or the pod comes
# up with NO server-side auto-delete and bills until a manual teardown. Keep the
# duration form for the operator, resolve it to the wall-clock instant here, and
# print that instant so the plan states exactly when RunPod will kill the pod.
abs_from_secs() {  # BSD date (macOS workstation) first, GNU date (Linux) second
  date -u -v+"$1"S +%Y-%m-%dT%H:%M:%SZ 2>/dev/null \
    || date -u -d "@$(( $(date -u +%s) + $1 ))" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null \
    || die "cannot resolve the seatbelt deadline: neither BSD nor GNU date available"
}
TERMINATE_AT=""
if [ "$NO_SEATBELT" -eq 0 ]; then
  if printf '%s' "$TERMINATE_AFTER" | grep -qE '^[0-9]+[hm]$'; then
    _n="${TERMINATE_AFTER%[hm]}"
    case "$TERMINATE_AFTER" in *h) _secs=$(( _n * 3600 )) ;; *) _secs=$(( _n * 60 )) ;; esac
    [ "$_secs" -gt 0 ] || die "--terminate-after must be greater than zero: $TERMINATE_AFTER"
    TERMINATE_AT="$(abs_from_secs "$_secs")"
  elif printf '%s' "$TERMINATE_AFTER" | grep -qE '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$'; then
    TERMINATE_AT="$TERMINATE_AFTER"   # already the absolute form the CLI wants
  else
    die "--terminate-after must be a duration (24h, 90m) or an RFC3339 UTC datetime (2026-08-26T01:40:00Z); got: $TERMINATE_AFTER"
  fi
fi

LEDGER="${CAGE_POD_LEDGER:-$CAGE_ROOT/results/ops/pod_ledger.jsonl}"

# --- price resolution (read-only; the ONLY optional network touch in plan mode)
PRICE_NOTE="(from --price-per-hour)"
if [ -z "$PRICE" ]; then
  if ! command -v runpodctl >/dev/null 2>&1; then
    PRICE_NOTE="(unknown — runpodctl not on PATH; pass --price-per-hour)"
  elif ! command -v python3 >/dev/null 2>&1; then
    PRICE_NOTE="(unknown — python3 absent, cannot parse the CLI's JSON; pass --price-per-hour)"
  else
    # CLI v2 emits JSON (global -o default), so the price for a gpuId lives on a
    # DIFFERENT line than the id itself — the pre-2026-08-25 line-oriented grep
    # matched '"gpuId": "NVIDIA L40S",' and found no number there, silently
    # degrading every plan to price=unknown. Parse the document instead, and
    # pick the field matching the cloud type actually being provisioned.
    PRICE="$(runpodctl gpu list 2>/dev/null | CAGE_GPU_ID="$GPU_ID" CAGE_CLOUD="$CLOUD_TYPE" python3 -c '
import json, os, sys
want = os.environ["CAGE_GPU_ID"]
field = "communityPricePerHr" if os.environ["CAGE_CLOUD"].upper() == "COMMUNITY" else "securePricePerHr"
try:
    gpus = json.load(sys.stdin)
except Exception:
    sys.exit(0)
for g in gpus:
    if g.get("gpuId") == want:
        p = g.get(field)
        if p:
            print(p)
        break
' 2>/dev/null || true)"
    if [ -n "$PRICE" ]; then PRICE_NOTE="(list price for $CLOUD_TYPE from 'runpodctl gpu list'; --price-per-hour overrides)"
    else PRICE_NOTE="(unknown — no $CLOUD_TYPE price for '$GPU_ID' in 'runpodctl gpu list'; pass --price-per-hour)"; fi
  fi
fi
EST_TOTAL="unknown (need both a price and --hours <est>)"
if [ -n "$PRICE" ] && [ -n "$HOURS" ]; then
  EST_TOTAL="\$$(awk -v p="$PRICE" -v h="$HOURS" -v g="$GPU_COUNT" 'BEGIN { printf "%.2f", p*h*g }') (${HOURS}h x \$${PRICE}/h x ${GPU_COUNT} GPU)"
fi
SEATBELT="$TERMINATE_AFTER (deletes at $TERMINATE_AT)"
[ "$NO_SEATBELT" -eq 0 ] || SEATBELT="DISABLED (--no-terminate-after)"

log "=================== RunPod provisioning PLAN ==================="
printf '  name              : %s\n' "$NAME"
printf '  --gpu-id          : %s   (x%s, cloud-type %s)\n' "$GPU_ID" "$GPU_COUNT" "$CLOUD_TYPE"
printf '  image             : %s\n' "$IMAGE"
printf '  container disk    : %s GB   /workspace volume: %s GB\n' "$DISK_GB" "$VOL_GB"
printf '  ports             : %s\n' "$PORTS"
if [ "$NO_SEATBELT" -eq 1 ]; then
  printf '  seatbelt          : DISABLED (--no-terminate-after) — NO server-side auto-delete\n'
else
  printf '  seatbelt          : --terminate-after %s -> server-side auto-delete at %s\n' "$TERMINATE_AFTER" "$TERMINATE_AT"
fi
PRICE_DISPLAY="unknown"; [ -z "$PRICE" ] || PRICE_DISPLAY="\$$PRICE/h"
printf '  price             : %s %s\n' "$PRICE_DISPLAY" "$PRICE_NOTE"
printf '  estimated total   : %s\n' "$EST_TOTAL"
printf '  ledger on create  : %s\n' "$LEDGER"

if [ "$YES" -ne 1 ]; then
  log "PLAN ONLY — nothing was created and nothing is billing (run-approval gate)."
  log "Record the owner GO, then re-run the SAME command with --yes to create."
  exit 0
fi

# --- create (--yes = the recorded owner GO) ---------------------------------
require_cmd runpodctl "brew install runpod/runpodctl/runpodctl"
require_cmd python3 "needed to write the schema-valid ledger line"
if [ "$NO_SEATBELT" -eq 1 ]; then
  warn "COST SEATBELT DISABLED (--no-terminate-after): this pod will BILL UNTIL torn down"
  warn "manually — a dead laptop no longer stops it; teardown_pod.sh becomes the ONLY stop."
fi
CREATE_ARGS=( pod create --name "$NAME" --image "$IMAGE" --gpu-id "$GPU_ID"
  --gpu-count "$GPU_COUNT" --container-disk-in-gb "$DISK_GB" --volume-in-gb "$VOL_GB"
  --volume-mount-path /workspace --ports "$PORTS" --cloud-type "$CLOUD_TYPE" )
[ "$NO_SEATBELT" -eq 1 ] || CREATE_ARGS+=( --terminate-after "$TERMINATE_AT" )
log "creating pod (cost-STARTING action): runpodctl$(printf ' %s' "${CREATE_ARGS[@]}")"
OUT="$(runpodctl "${CREATE_ARGS[@]}" 2>&1)" || die "pod create FAILED (nothing should be billing; verify with 'runpodctl pod list --all'): $OUT"
printf '%s\n' "$OUT"
POD_ID="$(printf '%s\n' "$OUT" | sed -nE 's/.*"id"[[:space:]]*:[[:space:]]*"([A-Za-z0-9-]+)".*/\1/p' | head -1)"
[ -n "$POD_ID" ] || POD_ID="$(printf '%s\n' "$OUT" | tr -c 'a-z0-9' '\n' | awk 'length($0) >= 13 && length($0) <= 20' | head -1 || true)"
[ -n "$POD_ID" ] || die "pod CREATED (it IS billing) but its id could not be parsed from the output above — find it with 'runpodctl pod list' and append the create event to $LEDGER manually"

mkdir -p "$(dirname "$LEDGER")"
LINE="$(CAGE_LJ_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)" CAGE_LJ_ID="$POD_ID" CAGE_LJ_NAME="$NAME" \
  CAGE_LJ_GPU="$GPU_ID" CAGE_LJ_COUNT="$GPU_COUNT" CAGE_LJ_PRICE="$PRICE" \
  CAGE_LJ_TA="$([ "$NO_SEATBELT" -eq 1 ] || printf '%s' "$TERMINATE_AT")" \
  CAGE_LJ_PURPOSE="$PURPOSE" python3 -c '
import json, os
e = os.environ
price = e.get("CAGE_LJ_PRICE", "")
ta = e.get("CAGE_LJ_TA", "")
print(json.dumps({
    "ts_utc": e["CAGE_LJ_TS"], "pod_id": e["CAGE_LJ_ID"], "name": e["CAGE_LJ_NAME"],
    "gpu_id": e["CAGE_LJ_GPU"], "gpu_count": int(e["CAGE_LJ_COUNT"]),
    "price_per_hour_usd": float(price) if price else None,
    "terminate_after": ta if ta else None,
    "purpose": e["CAGE_LJ_PURPOSE"], "event": "create",
}))')" || die "pod $POD_ID CREATED and BILLING but the ledger line could not be built — append the create event to $LEDGER manually NOW"
printf '%s\n' "$LINE" >> "$LEDGER"
log "ledger: create event appended -> $LEDGER"

log "pod CREATED: $POD_ID   (seatbelt: $SEATBELT)"
log "NEXT STEPS:"
log "  1) ship the repo tarball (scripts/ops/package_repo.sh), then ON the pod: bash scripts/runpod/setup_runpod.sh"
log "  2) monitor age/spend from the workstation: bash scripts/runpod/pod_status.sh   (cost table: cost_report.sh)"
log "  3) after the run + verified pull: bash scripts/runpod/teardown_pod.sh $POD_ID <backup_target> <local_run_dir>"
[ "$NO_SEATBELT" -eq 1 ] || log "  seatbelt: RunPod auto-deletes this pod at $TERMINATE_AT (server-side, in $TERMINATE_AFTER)"
