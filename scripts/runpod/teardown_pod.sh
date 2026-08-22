#!/bin/bash
# Order:     teardown bracket — after 5_observability's final sync; LAST step of a RunPod session
# Objective: Fail-closed pod teardown: ledger-gated pull_run.sh FIRST, confirm ceremony, pod delete strictly last, read-only $0 listing
# Cloud:     runpod
# SAFE RunPod teardown (task #137; RunPod = PRIMARY provider per FINAL SCOPE v2,
# MyDocs/COST_NEBIUS_RUNPOD_2026-08-16.md). Same fail-closed ordering as the GCP
# port (teardown_vm.sh): VERIFIED PULL FIRST — the ledger-gated pull_run.sh must
# print SAFE TO TEARDOWN before anything is destroyed — then the confirm()
# ceremony, then the pod delete (strictly LAST destructive step), then a
# READ-ONLY $0 listing that deletes nothing.
#
# STANDING DISCIPLINE (binding, MEMORY "Pull results local BEFORE teardown"):
# teardown is irreversible, so the run must exist LOCALLY (ledger-verified)
# before the pod is deleted. There is deliberately NO skip-the-pull env bypass
# here (finding J10 on the GCP port): --force is the single, loud override.
#
# Run from your WORKSTATION.
#
# Usage:
#   scripts/runpod/teardown_pod.sh <pod_id> <backup_target> <local_run_dir> [--force]
#     <pod_id>        RunPod pod id (see `runpodctl pod list --all`)
#     <backup_target> where the pod's sync daemons mirrored the run tree:
#                     s3://bucket[/prefix] (RunPod network-volume S3 API; set
#                     CAGE_S3_ENDPOINT) | ssh://[user@]host/path | gs://... |
#                     file:///path
#     <local_run_dir> local destination for the verified pull, mirroring the
#                     remote structure — normally results/<campaign>/<session>/<run_id>
#   --force  proceed even when the pull gate fails (DATA MAY BE LOST; loud)
#
# Env:
#   RUNPOD_API_KEY   auth for the REST fallback (and the listing) when
#                    runpodctl is absent; runpodctl itself is pre-authed via
#                    `runpodctl config --apiKey ...`
#   CAGE_POD_SSH     optional "user@host" of the pod: when set, a FINAL on-pod
#                    sync (sync_results.sh + collect_logs.sh) runs before the
#                    pull gate; when unset that step is SKIPPED loudly and the
#                    daemons' last sync is what the pull gate verifies
#   CAGE_SSH_OPTS    extra ssh options (e.g. "-p 2222" — RunPod pods expose SSH
#                    on non-standard ports); also used by the ssh:// transport
#   CAGE_ASSUME_YES  =1 answers the confirm() ceremony (non-interactive use)
#   CAGE_RUNPOD_REST REST base override for the CLI-absent fallback
#                    (default https://api.runpod.io/v2 — the v2 API)
#   CAGE_POD_LEDGER  pod-ops ledger path override (default
#                    results/ops/pod_ledger.jsonl under the repo root); a
#                    successful delete appends the {"event":"delete"} line that
#                    closes provision_pod.sh's create event (cost_report.sh /
#                    pod_status.sh consume the pairs)
set -uo pipefail
# (Deliberately no -e, matching teardown_vm.sh: every step's exit status is
# handled explicitly and the gates decide — never an unhandled abort mid-ceremony.)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/_common.sh
source "$SCRIPT_DIR/../lib/_common.sh"
# shellcheck source=scripts/lib/transport.sh
source "$SCRIPT_DIR/../lib/transport.sh"

# Non-interactive macOS shells may lack Homebrew dirs on PATH (runpodctl lives
# in /opt/homebrew/bin) — APPEND, never prepend, so a test shim still wins.
command -v runpodctl >/dev/null 2>&1 || PATH="$PATH:/opt/homebrew/bin:/usr/local/bin"

# v2 REST fallback base (v1 https://rest.runpod.io/v1 is DEPRECATED, retires
# 2026-11-15). Routes confirmed against https://api.runpod.io/v2/openapi.json
# (fetched 2026-08-21): DELETE {base}/pods/{podId} -> 204 · GET {base}/pods
# · GET {base}/network-volumes. Auth: Authorization: Bearer <RUNPOD_API_KEY>.
RUNPOD_REST="${CAGE_RUNPOD_REST:-https://api.runpod.io/v2}"

POD_ID=""; TARGET=""; LOCAL_DIR=""; FORCE=0
for a in "$@"; do
  case "$a" in
    --force) FORCE=1 ;;
    -*) echo "unknown flag: $a" >&2; exit 2 ;;
    *)
      if   [ -z "$POD_ID" ];    then POD_ID="$a"
      elif [ -z "$TARGET" ];    then TARGET="$a"
      elif [ -z "$LOCAL_DIR" ]; then LOCAL_DIR="$a"
      else echo "unexpected extra arg: $a" >&2; exit 2; fi ;;
  esac
done
if [ -z "$POD_ID" ] || [ -z "$TARGET" ] || [ -z "$LOCAL_DIR" ]; then
  echo "usage: scripts/runpod/teardown_pod.sh <pod_id> <backup_target> <local_run_dir> [--force]" >&2
  exit 2
fi

echo "=== SAFE TEARDOWN (RunPod): pod=$POD_ID | target=$TARGET | local=$LOCAL_DIR ==="

# --- [1/5] final on-pod sync (results + full logs/forensics) -----------------
echo "[1/5] final on-pod sync (results + logs + forensics) ..."
if [ -n "${CAGE_POD_SSH:-}" ]; then
  # shellcheck disable=SC2086
  ssh ${CAGE_SSH_OPTS:-} -o StrictHostKeyChecking=no -o ConnectTimeout=60 "$CAGE_POD_SSH" \
    "cd ~/CAGE && CAGE_BACKUP_TARGET='$TARGET' bash scripts/5_observability/sync_results.sh results && CAGE_BACKUP_TARGET='$TARGET' CAGE_COLLECT_TOKEN='teardown_$(date -u +%Y%m%d_%H%M%S)_$$' bash scripts/5_observability/collect_logs.sh" \
    | tail -5 \
    || echo "    WARNING: final on-pod sync FAILED or pod unreachable — relying on the daemons' last sync; the pull gate below is still authoritative" >&2
else
  echo "    SKIPPED (CAGE_POD_SSH unset) — relying on the pod's sync daemons; the pull gate below is authoritative"
fi

# --- [2/5] VERIFIED PULL — the fail-closed gate ------------------------------
# pull_run.sh mirrors the backup target locally and re-hashes the sha256 ledger;
# ONLY its literal "SAFE TO TEARDOWN" line authorizes destruction. Both the exit
# code AND the literal line are checked (belt + braces).
echo "[2/5] verified pull (ledger gate) -> $LOCAL_DIR ..."
PULL_OUT="$(bash "$SCRIPT_DIR/../5_observability/pull_run.sh" "$TARGET" "$LOCAL_DIR" 2>&1)"; PULL_RC=$?
printf '%s\n' "$PULL_OUT" | tail -6
if [ "$PULL_RC" -ne 0 ] || ! printf '%s' "$PULL_OUT" | grep -q "SAFE TO TEARDOWN"; then
  if [ "$FORCE" -eq 1 ]; then
    echo "    !!! pull gate FAILED (rc=$PULL_RC) but --force given. Deleting anyway; DATA MAY BE LOST."
  else
    echo "[teardown_pod] ABORT (fail-closed): pull_run.sh did not print SAFE TO TEARDOWN (rc=$PULL_RC)." >&2
    echo "               The pod/backup copy may be the ONLY intact copy of this run's data." >&2
    echo "               Fix the pull or the ledger mismatch, re-run, and proceed only after" >&2
    echo "               SAFE TO TEARDOWN — or pass --force to accept possible DATA LOSS." >&2
    exit 1
  fi
fi

# --- [3/5] confirm ceremony --------------------------------------------------
echo "[3/5] confirm ceremony ..."
confirm "Delete RunPod pod $POD_ID? (irreversible, cost-stopping)" \
  || { echo "[teardown_pod] aborted by operator — pod $POD_ID NOT deleted (still billing)."; exit 1; }

# --- [4/5] delete the pod (the ONLY destructive step, strictly last) ---------
echo "[4/5] deleting pod $POD_ID ... (cost-stopping action)"
if command -v runpodctl >/dev/null 2>&1; then
  runpodctl pod delete "$POD_ID" \
    || { echo "[teardown_pod] ERROR: 'runpodctl pod delete $POD_ID' FAILED — the pod may STILL BE BILLING; retry or delete via the console." >&2; exit 1; }
elif [ -n "${RUNPOD_API_KEY:-}" ]; then
  echo "    runpodctl absent -> REST fallback: DELETE $RUNPOD_REST/pods/$POD_ID"
  HTTP="$(curl -s -o /dev/null -w '%{http_code}' -X DELETE \
    -H "Authorization: Bearer $RUNPOD_API_KEY" "$RUNPOD_REST/pods/$POD_ID")"
  case "$HTTP" in
    2*) echo "    DELETE accepted (HTTP $HTTP)" ;;
    *)  echo "[teardown_pod] ERROR: DELETE returned HTTP $HTTP — the pod may STILL BE BILLING; retry or delete via the console." >&2; exit 1 ;;
  esac
else
  die "neither runpodctl nor RUNPOD_API_KEY available — cannot delete pod $POD_ID (it is STILL BILLING)"
fi

# Ledger delete-event hook: close provision_pod.sh's create event so
# cost_report.sh stops the clock. Append-only and NEVER fatal — the pod IS
# deleted by now; a ledger hiccup must not repaint a successful cost-stop red.
LEDGER="${CAGE_POD_LEDGER:-$CAGE_ROOT/results/ops/pod_ledger.jsonl}"
if mkdir -p "$(dirname "$LEDGER")" 2>/dev/null \
   && printf '{"ts_utc":"%s","pod_id":"%s","event":"delete"}\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$POD_ID" >> "$LEDGER" 2>/dev/null; then
  echo "    ledger: delete event appended -> $LEDGER"
else
  echo "    WARNING: could not append the delete event to $LEDGER — the pod IS deleted; record it manually or cost_report.sh keeps it LIVE/BILLING." >&2
fi

# --- [5/5] prove $0 (READ-ONLY listings; this step deletes NOTHING) ----------
# Clean-room rule (MyDocs/runpod-cli-reference.md §6): a surviving network
# volume still bills storage — the $0 verdict needs the pod listing pod-free
# AND the network-volume listing empty. Both listings are report-only.
echo "[5/5] confirming \$0 (read-only pod + network-volume listings) ..."
if command -v runpodctl >/dev/null 2>&1; then
  LISTING="$(runpodctl pod list --all 2>&1)"
  NV_LISTING="$(runpodctl network-volume list 2>&1)"
else
  LISTING="$(curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" "$RUNPOD_REST/pods" 2>&1)"
  NV_LISTING="$(curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" "$RUNPOD_REST/network-volumes" 2>&1)"
fi
printf '%s\n' "$LISTING" | head -10
if printf '%s' "$LISTING" | grep -q "$POD_ID"; then
  echo "[teardown_pod] WARNING: pod $POD_ID still appears in the listing — it may be mid-termination OR STILL BILLING. Re-run this listing until it is gone." >&2
  exit 1
fi
NV_TRIM="$(printf '%s' "$NV_LISTING" | tr -d '[:space:]')"
if [ -n "$NV_TRIM" ] && [ "$NV_TRIM" != "[]" ] && [ "$NV_TRIM" != "null" ]; then
  printf '%s\n' "$NV_LISTING" | head -10
  echo "[teardown_pod] WARNING: the network-volume listing above is NOT empty — surviving volumes bill storage (STILL BILLING; clean-room violation). After verifying the local pull, delete them ('runpodctl network-volume delete <id>' or the console), then re-run the listings. (If those lines are an empty table header, verify in the console and re-check.)" >&2
  exit 1
fi
echo "  -- network-volume listing empty — no surviving volumes"
echo "  -- NOTE (read-only reminder): saved templates bill separately and are NOT"
echo "     touched here — list/delete them via the RunPod console after the local"
echo "     pull is verified, per the clean-room-per-run discipline."
echo "TEARDOWN_COMPLETE pod=$POD_ID backup=$TARGET local=$LOCAL_DIR"
