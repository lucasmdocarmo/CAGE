#!/usr/bin/env bash
# Order:     alongside/after any stage — offline ledger analysis; --billing adds the authoritative account view
# Objective: Pair create/delete pod-ledger events into a per-pod runtime+cost table (offline-first; malformed lines refused loudly)
# Cloud:     runpod
# cost_report.sh — RunPod cost analysis from the pod-ops ledger (OFFLINE-first).
#
# PRIMARY source = results/ops/pod_ledger.jsonl (override: CAGE_POD_LEDGER),
# written by provision_pod.sh (create events) and teardown_pod.sh (delete
# events). Fully OFFLINE by default — no network, no CLI: pairs create/delete
# per pod_id, computes runtime and cost (runtime × price_per_hour_usd ×
# gpu_count), prints a per-pod table + total. Open pods (create without a
# delete) use NOW as the end and are flagged LIVE/BILLING.
#
# --billing additionally shells `runpodctl billing` for the ACCOUNT view —
# that output is the AUTHORITY on real spend; the ledger table is the local
# estimate (it cannot see storage, volumes, or pods created outside the
# wrapper).
#
# Malformed ledger lines are REFUSED loudly with their line number (exit 2),
# never skipped silently — a wrong cost table is worse than no cost table.
#
# Usage: scripts/runpod/cost_report.sh [--billing]
# Env:   CAGE_POD_LEDGER  ledger path override (default results/ops/pod_ledger.jsonl)
# Exit:  0 report printed · 2 usage or malformed ledger
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/_common.sh
source "$SCRIPT_DIR/../lib/_common.sh"
command -v runpodctl >/dev/null 2>&1 || PATH="$PATH:/opt/homebrew/bin:/usr/local/bin"

cleanup() { :; }  # read-only report: no temp state to reap
trap 'rc=$?; cleanup; exit $rc' EXIT
trap 'exit 130' INT TERM

BILLING=0
while [ $# -gt 0 ]; do
  case "$1" in
    --billing) BILLING=1; shift ;;
    -h|--help) printf 'usage: %s [--billing]\n' "$0" >&2; exit 2 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
LEDGER="${CAGE_POD_LEDGER:-$CAGE_ROOT/results/ops/pod_ledger.jsonl}"
require_cmd python3

if [ ! -f "$LEDGER" ]; then
  log "no ledger at $LEDGER — nothing has been provisioned through provision_pod.sh yet (local estimate: \$0.00)."
else
  rc=0
  python3 - "$LEDGER" <<'PY' || rc=$?
import datetime, json, sys, time
path = sys.argv[1]
def refuse(n, msg, raw):
    print(f"[cost_report] MALFORMED LEDGER LINE {n}: {msg}: {raw.strip()[:120]}", file=sys.stderr)
    print(f"[cost_report] REFUSING the whole report (malformed lines are never skipped "
          f"silently) — fix {path} line {n} and re-run.", file=sys.stderr)
    sys.exit(2)
def ts(s):
    return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=datetime.timezone.utc).timestamp()
events = []
for n, raw in enumerate(open(path, encoding="utf-8"), 1):
    if not raw.strip():
        continue
    try:
        e = json.loads(raw)
    except ValueError as exc:
        refuse(n, f"not JSON ({exc})", raw)
    if not isinstance(e, dict):
        refuse(n, "not a JSON object", raw)
    for k in ("ts_utc", "pod_id", "event"):
        if not isinstance(e.get(k), str) or not e[k]:
            refuse(n, f"missing/mistyped required key {k!r}", raw)
    if e["event"] not in ("create", "delete"):
        refuse(n, f"unknown event {e['event']!r}", raw)
    if e["event"] == "create":
        if not isinstance(e.get("name"), str) or not isinstance(e.get("gpu_id"), str) \
           or not isinstance(e.get("gpu_count"), int) or not isinstance(e.get("purpose"), str):
            refuse(n, "create event missing/mistyped name/gpu_id/gpu_count/purpose", raw)
        if e.get("price_per_hour_usd") is not None and not isinstance(e["price_per_hour_usd"], (int, float)):
            refuse(n, "price_per_hour_usd must be number|null", raw)
        if e.get("terminate_after") is not None and not isinstance(e["terminate_after"], str):
            refuse(n, "terminate_after must be string|null", raw)
    try:
        e["_ts"] = ts(e["ts_utc"])
    except ValueError:
        refuse(n, "ts_utc is not ISO-8601 Z (YYYY-MM-DDTHH:MM:SSZ)", raw)
    e["_n"] = n
    events.append(e)
pods = {}
for e in events:
    slot = pods.setdefault(e["pod_id"], {})
    if e["event"] in slot:
        print(f"[cost_report] NOTE: duplicate {e['event']} event for {e['pod_id']} at line "
              f"{e['_n']} — keeping the first (line {slot[e['event']]['_n']}).", file=sys.stderr)
    else:
        slot[e["event"]] = e
now = time.time()
total, unknown, live = 0.0, 0, 0
print(f"{'POD_ID':<20} {'NAME':<20} {'GPU':<26} {'CREATED_UTC':<20} {'END_UTC':<20} "
      f"{'HOURS':>7} {'$/H':>7} {'COST_USD':>9}  FLAGS")
for pid, slot in sorted(pods.items(), key=lambda kv: kv[1].get("create", kv[1].get("delete"))["_ts"]):
    c, d = slot.get("create"), slot.get("delete")
    if c is None:
        print(f"{pid:<20} {'?':<20} {'?':<26} {'?':<20} {d['ts_utc']:<20} {'?':>7} {'?':>7} "
              f"{'unknown':>9}  ORPHAN-DELETE (delete at line {d['_n']} without a create event)")
        unknown += 1
        continue
    end = d["_ts"] if d else now
    hours = (end - c["_ts"]) / 3600.0
    price, gc = c.get("price_per_hour_usd"), c.get("gpu_count") or 1
    flags = "" if d else "LIVE/BILLING"
    if not d:
        live += 1
    if isinstance(price, (int, float)):
        cost = hours * price * gc
        total += cost
        price_s, cost_s = f"{price:7.2f}", f"{cost:9.2f}"
    else:
        unknown += 1
        price_s, cost_s = f"{'?':>7}", f"{'unknown':>9}"
    end_s = d["ts_utc"] if d else "OPEN(now)"
    print(f"{pid:<20} {c.get('name', '?'):<20} {c.get('gpu_id', '?'):<26} {c['ts_utc']:<20} "
          f"{end_s:<20} {hours:7.2f} {price_s} {cost_s}  {flags}")
print(f"TOTAL (known prices): ${total:.2f} across {len(pods)} pod(s); "
      f"{unknown} with unknown cost; {live} LIVE/BILLING")
if live:
    print("[cost_report] NOTE: LIVE/BILLING pods accrue until teardown_pod.sh runs — "
          "this total grows in real time.")
PY
  [ "$rc" -eq 0 ] || exit "$rc"
fi

if [ "$BILLING" -eq 1 ]; then
  printf '=== RunPod ACCOUNT billing (runpodctl billing) — the AUTHORITY on real spend; ===\n'
  printf '=== the ledger table above is only the local estimate.                       ===\n'
  if command -v runpodctl >/dev/null 2>&1; then
    runpodctl billing || warn "'runpodctl billing' failed (auth/network?) — no account view; the ledger estimate above stands alone"
  else
    warn "runpodctl not on PATH — cannot fetch the account billing view"
  fi
fi
