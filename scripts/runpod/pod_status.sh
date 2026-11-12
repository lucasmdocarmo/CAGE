#!/usr/bin/env bash
# Order:     alongside stages 1-5 — read-only monitoring loop over live pods + the pod ledger
# Objective: Join `runpodctl pod list` with the pod ledger for per-pod uptime/spend + runaway-cost alarm (exit 1 past --max-age-hours)
# Cloud:     runpod
# pod_status.sh — READ-ONLY RunPod monitoring + runaway-cost alarm (CLI v2).
#
# Joins `runpodctl pod list --all` (+ per-pod `runpodctl pod get`) with the
# pod-ops ledger (results/ops/pod_ledger.jsonl: provision_pod.sh writes create
# events, teardown_pod.sh delete events, pod_watchdog.sh delete events with
# "by":"watchdog" when the seatbelt fired) and prints, per live pod: uptime
# (now minus the create ts), estimated spend (uptime x price_per_hour_usd x
# gpu_count when the ledger knows the price, else
# "unknown, pass --price-per-hour at provision"), and the seatbelt state
# (watchdog: ALIVE, DEAD or NONE, from the pidfile beside the ledger).
#
# ALARM: exits 1 when any pod's KNOWN age exceeds --max-age-hours (default 24)
# — run it in a watch loop as the runaway-cost alarm. A live pod with NO
# ledger create event is flagged LOUDLY (age unverifiable) but does not trip
# the alarm exit; keep the gap closed by provisioning through provision_pod.sh.
#
# Degrades gracefully: no runpodctl / listing failure -> ledger-only view over
# the OPEN create events (the alarm still applies to them — the offline
# alarm); no ledger -> live listing with unknown age/spend. Never mutates.
#
# Usage: scripts/runpod/pod_status.sh [--max-age-hours <n>]
#        scripts/runpod/pod_status.sh close <pod_id> [--ended-at <RFC3339Z>] --reason <text>
#   close (item 7, 2026-11-11): the ONE write this script makes. It closes a ledger
#   create row whose pod is gone without a delete event (a console delete, a pod
#   that vanished): appends {"event":"delete","by":"owner","closed_late":true,...}
#   with ts_utc = --ended-at (the instant from the console's billing page) or the
#   record time (then cost_report.sh calls the hours an upper bound). Fail-closed:
#   refuses unless `pod_watchdog.sh listed` says NOT LISTED (exit 1; an UNKNOWN
#   listing judges nothing), refuses a row that already has a delete event, a pod
#   with no create row, an empty reason, and an --ended-at that is malformed, not
#   calendar-valid, before the create or in the future. Append-only: no line is
#   edited. A pod that is still listed is torn down (teardown_pod.sh), not closed.
# Env:   CAGE_POD_LEDGER  ledger path override (default results/ops/pod_ledger.jsonl)
# Exit:  0 ok · 1 ALARM (age > --max-age-hours) or a refused close · nonzero on usage/internal error
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

LEDGER="${CAGE_POD_LEDGER:-$CAGE_ROOT/results/ops/pod_ledger.jsonl}"
require_cmd python3
usage() {
  printf 'usage: %s [--max-age-hours <n>]\n       %s close <pod_id> [--ended-at <RFC3339Z>] --reason <text>\n' "$0" "$0" >&2
  exit 2
}

# cmd_close <pod_id> [--ended-at <RFC3339Z>] --reason <text>  (item 7, 2026-11-11)
# Order: input checks, then the ledger (create row present, no delete row), then
# ONE listing through pod_watchdog.sh listed (the fail-closed parser, reused and
# not copied), then one appended line. Every refusal happens before the append.
cmd_close() {
  local pod="${1:-}" ended="" reason="" rc=0 now
  [ -n "$pod" ] || usage
  shift
  while [ $# -gt 0 ]; do
    case "$1" in
      --ended-at) [ $# -ge 2 ] || die "close: --ended-at requires a value"; ended="$2"; shift 2 ;;
      --reason)   [ $# -ge 2 ] || die "close: --reason requires a value"; reason="$2"; shift 2 ;;
      *) printf 'close: unknown argument: %s\n' "$1" >&2; usage ;;
    esac
  done
  printf '%s' "$pod" | grep -qE '^[A-Za-z0-9-]{6,64}$' || die "close: pod_id must match [A-Za-z0-9-]{6,64}: '$pod'"
  [ -n "$(printf '%s' "$reason" | tr -d '[:space:]')" ] || die "close: --reason <text> is required and may not be blank (it becomes the row's note)"
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  # Ledger checks and the --ended-at checks in one pass (the create's instant is needed for both).
  CAGE_CLOSE_POD="$pod" CAGE_CLOSE_ENDED="$ended" CAGE_CLOSE_NOW="$now" python3 - "$LEDGER" <<'PY' || rc=$?
import datetime, json, os, sys
path = sys.argv[1]; pod = os.environ["CAGE_CLOSE_POD"]; ended = os.environ["CAGE_CLOSE_ENDED"]
def ts(s):
    return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc).timestamp()
def refuse(msg):
    print(f"[cage] FATAL: close: {msg}", file=sys.stderr); sys.exit(1)
if ended:
    try:
        e_ts = ts(ended)
    except ValueError:
        refuse(f"--ended-at must be a calendar-valid RFC3339 UTC instant (YYYY-MM-DDTHH:MM:SSZ): '{ended}'")
    if e_ts > ts(os.environ["CAGE_CLOSE_NOW"]):
        refuse(f"--ended-at {ended} is in the future (now {os.environ['CAGE_CLOSE_NOW']})")
create, delete = None, None
if os.path.isfile(path):
    for n, raw in enumerate(open(path, encoding="utf-8"), 1):
        if not raw.strip():
            continue
        try:
            e = json.loads(raw)
        except ValueError:
            continue   # cost_report.sh refuses malformed lines loudly; the close judges the well-formed rows
        if not isinstance(e, dict) or e.get("pod_id") != pod:
            continue
        if e.get("event") == "create" and create is None:
            create = (n, e)
        elif e.get("event") == "delete" and delete is None:
            delete = (n, e)
if create is None:
    refuse(f"no create event for {pod} in {path}: nothing to close (a delete without a create would be an ORPHAN-DELETE row)")
if delete is not None:
    refuse(f"{pod} is already closed by the delete event at line {delete[0]} ({delete[1].get('ts_utc')}, by {delete[1].get('by') or 'teardown'})")
if ended and e_ts < ts(create[1]["ts_utc"]):
    refuse(f"--ended-at {ended} is before the create at line {create[0]} ({create[1]['ts_utc']})")
PY
  [ "$rc" -eq 0 ] || exit 1
  # ONE live read, through the watchdog's parser: 0 listed, 1 not listed, 2 unknown.
  # Review 2026-11-11 M3: only the pair (exit 1, verdict NOT_LISTED) proceeds; any
  # other exit 1 (a `die` on a precondition, set -e) carries no verdict and refuses.
  local verdict
  verdict="$(bash "$SCRIPT_DIR/pod_watchdog.sh" listed "$pod")" && rc=0 || rc=$?
  case "$rc:$verdict" in
    1:NOT_LISTED) ;;
    0:*) die "close: $pod is still listed by 'runpodctl pod list --all': a live pod is torn down (teardown_pod.sh), never closed on paper" ;;
    *) die "close: the listing is UNKNOWN (pod_watchdog.sh listed rc=$rc, verdict '${verdict:-none}': CLI failed, empty, not a JSON list of pods, or a precondition failed): nothing is judged; retry when the listing answers" ;;
  esac
  mkdir -p "$(dirname "$LEDGER")"
  CAGE_CLOSE_POD="$pod" CAGE_CLOSE_ENDED="$ended" CAGE_CLOSE_NOW="$now" CAGE_CLOSE_REASON="$reason" python3 - >> "$LEDGER" <<'PY'
import json, os
ended, now = os.environ["CAGE_CLOSE_ENDED"], os.environ["CAGE_CLOSE_NOW"]
print(json.dumps({"ts_utc": ended or now, "pod_id": os.environ["CAGE_CLOSE_POD"], "event": "delete",
                  "by": "owner", "note": os.environ["CAGE_CLOSE_REASON"], "closed_late": True,
                  "ended_at_given": bool(ended), "recorded_utc": now}))
PY
  if [ -n "$ended" ]; then
    log "ledger: $pod closed late (delete event by owner, ts_utc $ended from --ended-at) -> $LEDGER"
  else
    log "ledger: $pod closed late (delete event by owner, ts_utc $now, the record time: cost_report.sh calls its hours an upper bound) -> $LEDGER"
  fi
  exit 0
}

[ "${1:-}" != close ] || { shift; cmd_close "$@"; }

MAX_AGE_HOURS="24"
while [ $# -gt 0 ]; do
  case "$1" in
    --max-age-hours) [ $# -ge 2 ] || die "--max-age-hours requires a value"; MAX_AGE_HOURS="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; usage ;;
  esac
done
printf '%s' "$MAX_AGE_HOURS" | grep -qE '^[0-9]+(\.[0-9]+)?$' || die "--max-age-hours must be numeric: $MAX_AGE_HOURS"

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
# Seatbelt state per live pod: the client-side watchdog (pod_watchdog.sh) is the
# ONLY automatic delete that exists (RunPod has no server-side one), so a DEAD or
# absent watchdog means the pod bills until teardown_pod.sh or the console.
WD_DIR="$(dirname "$LEDGER")"
if [ "$LIVE_OK" -eq 1 ] && [ -s "$TMP/ids.txt" ]; then
  while IFS= read -r pid; do
    printf -- '-- runpodctl pod get %s --\n' "$pid"
    runpodctl pod get "$pid" 2>&1 | sed -n '1,6p' || warn "'runpodctl pod get $pid' failed"
    wd_pf="$WD_DIR/watchdog_${pid}.pid"
    if pidfile_alive "$wd_pf" "pod_watchdog.sh run $pid"; then
      printf '   watchdog: ALIVE (pid %s, deletes the pod at %s)\n' "$(cat "$wd_pf")" "$(cat "$WD_DIR/watchdog_${pid}.deadline" 2>/dev/null || printf '?')"
    elif [ -f "$wd_pf" ]; then
      printf '   watchdog: DEAD (stale pidfile; the pod is UNGUARDED: re-arm with pod_watchdog.sh arm %s <deadline> or tear down)\n' "$pid"
    else
      printf '   watchdog: NONE on this machine (no client-side seatbelt armed; RunPod has no server-side one)\n'
    fi
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
        rows.append((pid, c.get("name", "?"), "-", "-", f"ledger-OPEN but not listed: verify in the console, then `pod_status.sh close {pid} --ended-at <billing-page instant> --reason <why>` (item 7)"))
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
