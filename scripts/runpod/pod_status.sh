#!/usr/bin/env bash
# Order:     alongside stages 1-5 — read-only monitoring loop over live pods + the pod ledger
# Objective: Join `runpodctl pod list` with the pod ledger for per-pod uptime/spend + runaway-cost alarm (exit 1 past --max-age-hours)
# Cloud:     runpod
# pod_status.sh — READ-ONLY RunPod monitoring + runaway-cost alarm (CLI v2).
#
# Joins `runpodctl pod list --all` (+ per-pod `runpodctl pod get`) with the
# pod-ops ledger (results/ops/pod_ledger.jsonl — provision_pod.sh writes create
# events, teardown_pod.sh delete events) and prints, per live pod: uptime
# (now − create ts) and estimated spend (uptime × price_per_hour_usd ×
# gpu_count when the ledger knows the price, else
# "unknown — pass --price-per-hour at provision").
#
# ALARM: exits 1 when any pod's KNOWN age exceeds --max-age-hours (default 24)
# — run it in a watch loop as the runaway-cost alarm. A live pod with NO
# ledger create event is flagged LOUDLY (age unverifiable) but does not trip
# the alarm exit; keep the gap closed by provisioning through provision_pod.sh.
#
# Degrades gracefully: no runpodctl / listing failure -> ledger-only view over
# the OPEN create events (the alarm still applies to them — the offline
# watchdog); no ledger -> live listing with unknown age/spend. Never mutates.
#
# Usage: scripts/runpod/pod_status.sh [--max-age-hours <n>]
# Env:   CAGE_POD_LEDGER  ledger path override (default results/ops/pod_ledger.jsonl)
# Exit:  0 ok · 1 ALARM (age > --max-age-hours) · nonzero on usage/internal error
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/_common.sh
source "$SCRIPT_DIR/../lib/_common.sh"
command -v runpodctl >/dev/null 2>&1 || PATH="$PATH:/opt/homebrew/bin:/usr/local/bin"

TMP="$(mktemp -d)"
cleanup() { rm -rf "${TMP:?}"; }
trap 'rc=$?; cleanup; exit $rc' EXIT
trap 'exit 130' INT TERM

MAX_AGE_HOURS="24"
while [ $# -gt 0 ]; do
  case "$1" in
    --max-age-hours) [ $# -ge 2 ] || die "--max-age-hours requires a value"; MAX_AGE_HOURS="$2"; shift 2 ;;
    -h|--help) printf 'usage: %s [--max-age-hours <n>]\n' "$0" >&2; exit 2 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
printf '%s' "$MAX_AGE_HOURS" | grep -qE '^[0-9]+(\.[0-9]+)?$' || die "--max-age-hours must be numeric: $MAX_AGE_HOURS"
LEDGER="${CAGE_POD_LEDGER:-$CAGE_ROOT/results/ops/pod_ledger.jsonl}"
require_cmd python3

# --- live listing (read-only; degrade LOUDLY, never silently) ----------------
LIVE_OK=0
: > "$TMP/pods.txt"
if command -v runpodctl >/dev/null 2>&1; then
  if runpodctl pod list --all > "$TMP/pods.txt" 2>"$TMP/pods.err"; then
    LIVE_OK=1
  else
    warn "'runpodctl pod list --all' FAILED — ledger-only view (first error line: $(head -1 "$TMP/pods.err" 2>/dev/null || true))"
  fi
else
  warn "runpodctl not on PATH — ledger-only view (live listing unavailable)"
fi

# Extract pod ids from the listing (JSON list preferred; table fallback).
python3 - "$TMP/pods.txt" > "$TMP/ids.txt" <<'PY'
import json, re, sys
text = open(sys.argv[1], encoding="utf-8").read().strip()
ids = []
if text:
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("pods") or data.get("data") or []
        ids = [d["id"] for d in data if isinstance(d, dict) and d.get("id")]
    except ValueError:
        for line in text.splitlines():
            tok = line.split()
            if tok and re.fullmatch(r"[a-z0-9]{12,20}", tok[0]):
                ids.append(tok[0])
for i in ids:
    print(i)
PY

# Per-pod detail (read-only `pod get`; a failed get is announced, not fatal).
if [ "$LIVE_OK" -eq 1 ] && [ -s "$TMP/ids.txt" ]; then
  while IFS= read -r pid; do
    printf -- '-- runpodctl pod get %s --\n' "$pid"
    runpodctl pod get "$pid" 2>&1 | sed -n '1,6p' || warn "'runpodctl pod get $pid' failed"
  done < "$TMP/ids.txt"
fi

# --- join with the ledger + the alarm verdict --------------------------------
rc=0
python3 - "$LEDGER" "$TMP/ids.txt" "$MAX_AGE_HOURS" "$LIVE_OK" <<'PY' || rc=$?
import datetime, json, os, sys, time
ledger_path, ids_path, max_age, live_ok = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4] == "1"
now = time.time()
def ts(s):
    return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=datetime.timezone.utc).timestamp()
creates, deletes = {}, set()
if os.path.isfile(ledger_path):
    for n, raw in enumerate(open(ledger_path, encoding="utf-8"), 1):
        if not raw.strip():
            continue
        try:
            e = json.loads(raw)
        except ValueError:
            print(f"[cage] WARNING: malformed ledger line {n} ignored for STATUS only "
                  f"(cost_report.sh refuses it loudly): {raw.strip()[:80]}", file=sys.stderr)
            continue
        if isinstance(e, dict) and e.get("event") == "create" and e.get("pod_id"):
            creates[e["pod_id"]] = e
        elif isinstance(e, dict) and e.get("event") == "delete" and e.get("pod_id"):
            deletes.add(e["pod_id"])
else:
    print(f"[cage] WARNING: no ledger at {ledger_path} — uptime/spend unknown for every pod", file=sys.stderr)
live = [l.strip() for l in open(ids_path, encoding="utf-8") if l.strip()]
rows, alarms = [], []
UNKNOWN_SPEND = "unknown — pass --price-per-hour at provision"
def describe(pid, c):
    age_h = (now - ts(c["ts_utc"])) / 3600.0
    price, gc = c.get("price_per_hour_usd"), c.get("gpu_count") or 1
    spend = f"${age_h * price * gc:.2f}" if isinstance(price, (int, float)) else UNKNOWN_SPEND
    if age_h > max_age:
        alarms.append(f"{pid} age {age_h:.2f}h exceeds --max-age-hours {max_age:g}")
    return f"{age_h:.2f}h", spend
for pid in live:
    c = creates.get(pid)
    if c and c.get("ts_utc"):
        up, spend = describe(pid, c)
        rows.append((pid, c.get("name", "?"), up, spend, "LIVE"))
    else:
        rows.append((pid, "?", "UNKNOWN", UNKNOWN_SPEND, "LIVE (not in ledger)"))
        print(f"[cage] WARNING: live pod {pid} has no ledger create event — age unverifiable "
              f"(provisioned outside provision_pod.sh?)", file=sys.stderr)
for pid, c in creates.items():
    if pid in deletes or pid in live:
        continue
    if live_ok:
        rows.append((pid, c.get("name", "?"), "-", "-", "ledger-OPEN but not listed — verify, then close via teardown's delete event"))
    else:  # offline watchdog: open ledger pods are assumed live and alarmed
        up, spend = describe(pid, c)
        rows.append((pid, c.get("name", "?"), up, spend, "OPEN in ledger (live listing unavailable)"))
if not rows:
    print("[pod_status] no live pods and no open ledger pods — nothing billing ($0 view).")
else:
    print(f"{'POD_ID':<20} {'NAME':<24} {'UPTIME':>9}  {'EST_SPEND':<44} STATE")
    for r in rows:
        print(f"{r[0]:<20} {r[1]:<24} {r[2]:>9}  {r[3]:<44} {r[4]}")
if alarms:
    for a in alarms:
        print(f"[pod_status] ALARM: {a}", file=sys.stderr)
    print(f"[pod_status] ALARM: {len(alarms)} pod(s) exceed --max-age-hours {max_age:g} — "
          f"runaway-cost check FAILED (teardown_pod.sh or justify + re-arm).", file=sys.stderr)
    sys.exit(1)
PY
exit "$rc"
