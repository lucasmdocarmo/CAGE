#!/bin/bash
# CAGE log collector. Gather EVERY run + system log into logs/ and mirror to the
# off-box backup target (provider-neutral via sync_results.sh: gs:// | s3:// |
# ssh:// | file://), so nothing is lost when a box is torn down or preempted.
# Host-namespaced (vm_logs/<host>/) for multi-node runs.
#
# On success in full mode it writes a per-run SENTINEL object as the LAST upload
# (vm_logs/<host>/COLLECT_OK_<token>), AFTER the content sync, so the teardown
# scripts (teardown_pod.sh / teardown_vm.sh) can verify robustly that THIS
# collection actually completed before deleting the box.
#
# Captures what sync_results.sh (results/ only) does NOT:
#   - vLLM server logs   (logs/vllm/*.log)
#   - run stdout / stats (HOME, repo root, and extra dirs; depth<=2; *.log/*.out/nohup.out)
#   - status timeline    (status_timeline*.log)
#   - docker/redis logs  (docker logs for every container)
#   - system forensics   (nvidia-smi, dmesg + kernel OOM/Xid, journalctl 6h, pip freeze, env)
#
# Usage (ON the VM):
#   bash scripts/5_observability/collect_logs.sh           # full: logs + forensics + sync (+ success sentinel)
#   bash scripts/5_observability/collect_logs.sh --light   # logs + sync only (cheap; periodic use; no sentinel)
# Env:
#   CAGE_BACKUP_TARGET    provider-neutral target (gs://|s3://|ssh://|file://)
#   CAGE_RESULTS_BUCKET   legacy GCS override (bare name or gs://); on a GCP box
#                         the metadata-derived default still applies (GCS port)
#   CAGE_COLLECT_TOKEN    unique token for the success sentinel (teardown scripts set this)
#   CAGE_EXTRA_LOG_DIRS   extra ':'-separated dirs to also scan for *.log/*.out
#   CAGE_LOG_NO_SYNC=1    gather + manifest only, skip the upload (for local testing)
#
# Deliberately NOT `set -e`: collection is best-effort by design -- one unreadable
# forensic source must never abort the gather; each step is individually guarded and
# only the (provider-neutral) backup sync result decides the exit code (fail-closed for teardown).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=scripts/lib/_common.sh
source "$PROJECT_DIR/scripts/lib/_common.sh"
cd "$PROJECT_DIR" || die "cannot cd to $PROJECT_DIR"

MODE="${1:-full}"
HOST="$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo vm)"
HOST="$(printf '%s' "$HOST" | tr -c 'A-Za-z0-9_.-' '_')"   # sanitize for a safe GCS path
TS="$(date +%Y%m%d_%H%M%S 2>/dev/null || echo run)"
LOGROOT="logs"
mkdir -p "$LOGROOT/vllm" "$LOGROOT/runs" "$LOGROOT/system" \
  || die "cannot create $LOGROOT/ subdirs"

is_light() { [ "$MODE" = "--light" ] || [ "$MODE" = "light" ]; }

# --- 1. Gather stray stdout/stats/timeline logs (depth<=2) from HOME, repo, extras ---
scan_dirs=("$HOME" "$PROJECT_DIR")
IFS=':' read -r -a _extra <<< "${CAGE_EXTRA_LOG_DIRS:-}" || true
for d in "${_extra[@]:-}"; do [ -n "$d" ] && scan_dirs+=("$d"); done
for base in "${scan_dirs[@]}"; do
  [ -d "$base" ] || continue
  # Tag destination by source dir so same-basename logs (~/run.log vs ~/CAGE/run.log)
  # do not overwrite each other in the flat runs/ folder.
  tag="$(printf '%s' "$base" | tr -c 'A-Za-z0-9' '_' | tail -c 16)"
  while IFS= read -r -d '' f; do
    case "$f" in "$PROJECT_DIR/$LOGROOT/"*) continue ;; esac   # skip our own tree
    cp -p "$f" "$LOGROOT/runs/${tag}__$(basename "$f")" 2>/dev/null || true
  done < <(find "$base" -maxdepth 2 -type f \( -name '*.log' -o -name '*.out' -o -name 'nohup.out' \) -print0 2>/dev/null)
done

# --- 2. System forensics (full mode only) ------------------------------------------
if ! is_light; then
  SYS="$LOGROOT/system/${HOST}_${TS}"
  mkdir -p "$SYS" || echo "[collect_logs] WARNING: cannot create $SYS (forensics skipped)" >&2
  if [ -d "$SYS" ]; then
    { hostname; uname -a; date -u; echo; uptime; }        > "$SYS/host.txt"            2>&1 || true
    nvidia-smi                                            > "$SYS/nvidia-smi.txt"      2>&1 || true
    nvidia-smi -q                                         > "$SYS/nvidia-smi-full.txt" 2>&1 || true
    free -h                                               > "$SYS/mem.txt"             2>&1 || true
    df -h                                                 > "$SYS/disk.txt"            2>&1 || true
    ps aux                                                > "$SYS/ps.txt"              2>&1 || true
    ( dmesg -T 2>/dev/null || sudo -n dmesg -T 2>/dev/null ) > "$SYS/dmesg.txt"        2>&1 || true
    grep -iE "out of memory|oom-kill|killed process|xid|nvrm" "$SYS/dmesg.txt" \
                                                          > "$SYS/dmesg_oom_gpu.txt"   2>/dev/null || true
    ( journalctl --since "-6h" --no-pager 2>/dev/null \
        || sudo -n journalctl --since "-6h" --no-pager 2>/dev/null ) > "$SYS/journal.txt" 2>&1 || true
    ( journalctl -k --no-pager 2>/dev/null \
        || sudo -n journalctl -k --no-pager 2>/dev/null ) > "$SYS/journal_kernel.txt"  2>&1 || true
    ( pip freeze 2>/dev/null )                            > "$SYS/pip_freeze.txt"      2>&1 || true
    ( python3 -c "import vllm,torch;print('vllm',vllm.__version__);print('torch',torch.__version__)" 2>/dev/null ) \
                                                          > "$SYS/versions.txt"        2>&1 || true
    # Env forensics with SECRET REDACTION (finding J9): the raw dump captured
    # HUGGING_FACE_HUB_TOKEN / VLLM_API_KEY values VERBATIM and uploaded them to the
    # bucket on every collect. Keep the variable NAMES (presence is the forensic
    # signal) but replace the VALUE of any name matching the secret pattern.
    ( env | grep -iE "VLLM|CAGE|CUDA|REDIS|HF_HOME|HUGGING" | sort \
        | awk -F= 'BEGIN{OFS="="} toupper($1) ~ /TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL/ {print $1, "***REDACTED***"; next} {print}' ) \
                                                          > "$SYS/cage_env.txt"        2>&1 || true
    ( redis-cli ping 2>/dev/null )                        > "$SYS/redis.txt"           2>&1 || true
    # docker / redis container logs (cloud_run starts a cage-redis container) - best effort
    if command -v docker >/dev/null 2>&1; then
      docker ps -a > "$SYS/docker_ps.txt" 2>&1 || true
      for c in $(docker ps -aq 2>/dev/null); do
        docker logs "$c" > "$SYS/docker_${c}.log" 2>&1 || true
      done
    fi
  fi
fi

# --- 3. Manifest -------------------------------------------------------------------
find "$LOGROOT" -type f 2>/dev/null | sort > "$LOGROOT/COLLECTED_MANIFEST_${HOST}.txt" || true
N=$(find "$LOGROOT" -type f 2>/dev/null | wc -l | tr -d ' ')
echo "[collect_logs] gathered $N files under $LOGROOT/ (host=$HOST mode=$MODE)"

# --- 4. Mirror to the backup target, host-namespaced (unless CAGE_LOG_NO_SYNC=1) ---
if [ "${CAGE_LOG_NO_SYNC:-0}" = "1" ]; then
  echo "[collect_logs] CAGE_LOG_NO_SYNC=1 -> skipping upload (local gather only)"
  echo "COLLECT_LOGS_DONE host=$HOST (no-sync)"
  exit 0
fi
# Provider-neutral sync (task #137): target resolution + per-backend markers live
# in sync_results.sh; a failure exits nonzero here — fail-closed for teardown.
if ! bash "$SCRIPT_DIR/sync_results.sh" "$LOGROOT" "${CAGE_BACKUP_TARGET:-${CAGE_RESULTS_BUCKET:-}}" "vm_logs/$HOST"; then
  echo "COLLECT_LOGS_SYNC_FAILED host=$HOST" >&2
  exit 1
fi

# Success sentinel (full mode only): written + uploaded in a SECOND sync, i.e. AFTER the
# content sync above succeeded, so its presence off-box proves the content is there too.
# The teardown scripts check for this exact token before deleting the box (fail-closed).
if is_light; then
  echo "COLLECT_LOGS_DONE host=$HOST -> vm_logs/$HOST/ (light, no sentinel)"
  exit 0
fi
TOKEN="${CAGE_COLLECT_TOKEN:-$TS}"
TOKEN="$(printf '%s' "$TOKEN" | tr -c 'A-Za-z0-9_.-' '_')"   # sentinel name must be a safe object name
SENT="$LOGROOT/COLLECT_OK_${TOKEN}"
{ echo "host=$HOST"; echo "token=$TOKEN"; echo "files=$N"; date -u; } > "$SENT" \
  || { echo "COLLECT_LOGS_SENTINEL_WRITE_FAILED host=$HOST" >&2; exit 1; }
if bash "$SCRIPT_DIR/sync_results.sh" "$LOGROOT" "${CAGE_BACKUP_TARGET:-${CAGE_RESULTS_BUCKET:-}}" "vm_logs/$HOST" >/dev/null 2>&1; then
  echo "COLLECT_LOGS_DONE host=$HOST sentinel=COLLECT_OK_${TOKEN}"
else
  echo "COLLECT_LOGS_SENTINEL_SYNC_FAILED host=$HOST" >&2
  exit 1
fi
