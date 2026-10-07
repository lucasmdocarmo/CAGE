#!/bin/bash
# Order:     alongside stages 7 to 11 of scripts/6_experiments/cage_experiment.sh (started by stage 7 in the background, stopped by stage 11 by its recorded pid)
# Objective: Mac-side pod monitor loop: one runpodctl read plus one bounded read-only ssh bundle per tick -> status.json, monitor.log, errors.jsonl (de-duplicated) and alerts.log with the PORTAL ACTION block on every alert (ADR-0143)
# Cloud:     runpod
# =============================================================================
# monitor_pod.sh --pod ID --ssh user@host --port P --run-root ROOT --out DIR
#                [--key KEY] [--pod-repo /workspace/CAGE] [--interval 120]
#                [--price USD/h] [--created UTC] [--balance-floor USD]
#                [--stall-min N] [--job NAME] [--backup-backend local|s3]
#                [--scripts DIR] [--parent PID] [--seatbelt-floor-min 60]
#                [--once] [--stop-on-alert]
#
# What the S0 day could not see (design section 6.1): every window labeled
# UNKNOWN_TELEMETRY (seen only when a contrast refused after the cells), a cell
# that ran 26 min with the GPU idle and a 0-byte log, an expected refusal that
# exited 0, a launcher that printed a start failure while the engine kept
# loading. Each has a cheap signal, read here every tick:
#   - the pod itself: runtimeStatus (never desiredStatus), the watchdog
#     (ALIVE or not) and its deadline, the cost clock, the balance every 10th tick
#   - the run tree: watch_campaign.sh verdict and exit code, .STATUS-* failure
#     sentinels, the write-time journal's age, the newest regime.json label,
#     GPU utilization and memory, compute apps, the run job's status, the sync
#     markers (the daemon log is NOT a marker: failed passes append to it too)
#   - the job log's last 200 lines filtered for Traceback, REFUS, FAIL, ERROR,
#     CAMPAIGN TELEMETRY, UNKNOWN_TELEMETRY, RELAUNCH FAILED, STOP FAILED
#   - every 15th tick: nvidia-smi -q (Xid, ECC, throttle counters) to a file
#
# Alerts (HARD stops the master only with --stop-on-alert; the master's stage 8
# fails on any HARD line): runtimeStatus not running; watchdog not ALIVE;
# seatbelt under --seatbelt-floor-min (60) whatever the job state, because the
# seatbelt deletes the pod mid-score or mid-pull just the same (review
# 2026-10-06, HIGH 4); balance under the floor; the job CRASHED; a NEW failure
# sentinel (the baseline moves only on a tick whose ssh answered); the journal
# older than --stall-min with GPU utilization under 5 percent (the idle-cell
# shape). SOFT: a verdict that is not healthy; a new Traceback; a sync failure
# marker newer than the last success. With --parent PID the loop ends on its
# own once that pid is gone (no orphan after a killed master).
#
# Every tick writes status.json (the whole picture), appends one line to
# monitor.log, appends NEW error lines to errors.jsonl (de-duplicated by text),
# and appends to alerts.log with the PORTAL ACTION block on every alert.
# All reads are bounded; nothing here changes the pod.
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=scripts/lib/_common.sh
source "$PROJECT_DIR/scripts/lib/_common.sh"

POD=""; HOST=""; PORT="22"; KEY=""; ROOT=""; OUT=""; POD_REPO="/workspace/CAGE"
INTERVAL=120; PRICE=""; CREATED=""; FLOOR="50"; STALL_MIN=30; JOB="run"; BACKEND="local"
SCRIPTS="$PROJECT_DIR/scripts"; ONCE=0; STOP_ON_ALERT=0; PARENT=""; SEATBELT_FLOOR_MIN=60
while [ $# -gt 0 ]; do
  case "$1" in
    --parent) PARENT="$2"; shift 2 ;;  --seatbelt-floor-min) SEATBELT_FLOOR_MIN="$2"; shift 2 ;;
    --pod) POD="$2"; shift 2 ;;        --ssh) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;      --key) KEY="$2"; shift 2 ;;
    --run-root) ROOT="$2"; shift 2 ;;  --out) OUT="$2"; shift 2 ;;
    --pod-repo) POD_REPO="$2"; shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --price) PRICE="$2"; shift 2 ;;    --created) CREATED="$2"; shift 2 ;;
    --balance-floor) FLOOR="$2"; shift 2 ;;
    --stall-min) STALL_MIN="$2"; shift 2 ;;
    --job) JOB="$2"; shift 2 ;;        --backup-backend) BACKEND="$2"; shift 2 ;;
    --scripts) SCRIPTS="$2"; shift 2 ;;
    --once) ONCE=1; shift ;;           --stop-on-alert) STOP_ON_ALERT=1; shift ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; sed -n '6,11p' "$0" >&2; exit 2 ;;
  esac
done
[ -n "$POD" ] && [ -n "$HOST" ] && [ -n "$ROOT" ] && [ -n "$OUT" ] || { sed -n '6,11p' "$0" >&2; exit 2; }
mkdir -p "$OUT" || die "cannot create $OUT"
PY3="$(command -v python3)" || die "python3 required"

utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
rssh() {
  if [ -n "$KEY" ]; then
    ssh -p "$PORT" -i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=25 "$HOST" "$1"
  else
    ssh -p "$PORT" -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=25 "$HOST" "$1"
  fi
}

# The read-only bundle: one ssh per tick, every value a labeled line.
BUNDLE="cd $POD_REPO 2>/dev/null || true
w=\$(bash scripts/5_observability/watch_campaign.sh '$ROOT' 2>&1); wrc=\$?
echo \"WATCH_RC=\$wrc\"; echo \"WATCH_VERDICT=\$(printf '%s\n' \"\$w\" | tail -n 1 | cut -c1-200)\"
echo \"SENTINELS=\$(find '$ROOT/cells' -name '.STATUS-*' 2>/dev/null | wc -l | tr -d ' ')\"
echo \"RUN_DIR=\$([ -d '$ROOT/cells' ] && echo 1 || echo 0)\"
echo \"SENTINEL_LIST=\$(find '$ROOT/cells' -name '.STATUS-*' 2>/dev/null | head -5 | tr '\n' ' ')\"
j=\$(stat -c %Y '$ROOT/write_time_hashes.jsonl' 2>/dev/null || echo 0); n=\$(date +%s)
if [ \"\$j\" -gt 0 ]; then echo \"JOURNAL_AGE_S=\$((n - j))\"; else echo JOURNAL_AGE_S=-1; fi
echo \"WINDOWS=\$(find '$ROOT/cells' -type d -name 'window_*' 2>/dev/null | wc -l | tr -d ' ')\"
r=\$(find '$ROOT/cells' -name regime.json 2>/dev/null | head -200 | xargs ls -t 2>/dev/null | head -1)
case \"\$r\" in */regime.json) echo \"REGIME=\$(tr -d '\n' < \"\$r\" | cut -c1-300)\" ;; *) echo REGIME= ;; esac
echo \"GPU=\$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)\"
echo \"COMPUTE_APPS=\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)\"
if [ -f ~/.cage_jobs/$JOB.status ]; then echo \"JOB_STATUS=\$(cat ~/.cage_jobs/$JOB.status)\"; elif [ -f ~/.cage_jobs/$JOB.pid ] && kill -0 \$(cat ~/.cage_jobs/$JOB.pid) 2>/dev/null; then echo JOB_STATUS=RUNNING; elif [ -f ~/.cage_jobs/$JOB.pid ]; then echo JOB_STATUS=CRASHED; else echo JOB_STATUS=NONE; fi
echo \"SYNC_OK_MTIME=\$(stat -c %Y .agent/last_sync_ok_$BACKEND 2>/dev/null || echo 0)\"
echo \"SYNC_FAIL_MTIME=\$(stat -c %Y .agent/last_sync_fail_$BACKEND 2>/dev/null || echo 0)\"
echo ERRLINES_BEGIN
tail -n 200 ~/.cage_jobs/$JOB.log 2>/dev/null | grep -nE 'Traceback|REFUS|FAIL|ERROR|exit 2|CAMPAIGN TELEMETRY|UNKNOWN_TELEMETRY|RELAUNCH FAILED|STOP FAILED' | tail -n 40
echo ERRLINES_END"

TICK=0
while :; do
  if [ -n "$PARENT" ] && ! kill -0 "$PARENT" 2>/dev/null; then
    printf '[monitor] parent %s is gone: exiting (no orphan loop)\n' "$PARENT"; exit 0
  fi
  TICK=$((TICK + 1))
  NOW="$(utc)"
  TICKF="$OUT/.tick.txt"
  : > "$TICKF"
  { printf 'TICK=%s\nNOW=%s\n' "$TICK" "$NOW"
    printf 'POD_GET_BEGIN\n'; runpodctl pod get "$POD" 2>&1 | head -c 20000; printf '\nPOD_GET_END\n'
    if [ $((TICK % 10)) -eq 1 ]; then printf 'USER_BEGIN\n'; runpodctl user 2>&1 | head -c 4000; printf '\nUSER_END\n'; fi
    printf 'WATCHDOG=%s\n' "$(bash "$SCRIPTS/runpod/pod_watchdog.sh" status "$POD" 2>&1 | head -n 1)"
    printf 'SSH_BEGIN\n'; rssh "$BUNDLE" 2>&1 | head -c 40000; printf '\nSSH_RC=%s\nSSH_END\n' "${PIPESTATUS[0]}"
  } >> "$TICKF"
  if [ $((TICK % 15)) -eq 1 ]; then rssh "nvidia-smi -q -d ECC,PERFORMANCE,CLOCK 2>&1 | head -n 200" > "$OUT/nvidia_q_$(printf '%04d' "$TICK").txt" 2>&1 || true; fi

  ALERT_RC=0
  MON_OUT="$OUT" MON_TICK="$TICKF" MON_POD="$POD" MON_PRICE="$PRICE" MON_CREATED="$CREATED" \
  MON_FLOOR="$FLOOR" MON_STALL="$STALL_MIN" MON_HOST="$HOST" MON_SCRIPTS="$SCRIPTS" MON_JOB="$JOB" MON_SEATBELT_FLOOR="$SEATBELT_FLOOR_MIN" "$PY3" - <<'PY' || ALERT_RC=$?
import json, os, re, datetime, sys
out = os.environ["MON_OUT"]; raw = open(os.environ["MON_TICK"], encoding="utf-8", errors="replace").read()
now = datetime.datetime.now(datetime.timezone.utc); now_s = now.strftime("%Y-%m-%dT%H:%M:%SZ")

def block(name):
    m = re.search(rf"{name}_BEGIN\n(.*?)\n{name}_END", raw, re.S)
    return m.group(1) if m else ""
def kv(text, key, default=""):
    m = re.search(rf"^{key}=(.*)$", text, re.M)
    return m.group(1).strip() if m else default
def jsonish(text):
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try: return json.loads(m.group(0))
            except Exception: pass
    return None
def find_key(doc, names):
    if isinstance(doc, dict):
        for k, v in doc.items():
            if k in names: return v
        for v in doc.values():
            r = find_key(v, names)
            if r is not None: return r
    if isinstance(doc, list):
        for v in doc:
            r = find_key(v, names)
            if r is not None: return r
    return None

tick = int(kv(raw, "TICK", "0"))
pod = jsonish(block("POD_GET")) or {}
runtime = str(find_key(pod, {"runtimeStatus"}) or "unknown")
desired = str(find_key(pod, {"desiredStatus"}) or "")
user = jsonish(block("USER")) if block("USER") else None
balance = find_key(user, {"balance", "currentBalance", "clientBalance"}) if user else None
watchdog = kv(raw, "WATCHDOG")
wd_alive = "watchdog=ALIVE" in watchdog
m = re.search(r"deadline=(\S+)", watchdog)
seatbelt_min_left = None
if m and re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", m.group(1)):
    dl = datetime.datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)
    seatbelt_min_left = round((dl - now).total_seconds() / 60, 1)
cost = None; hours = None
if os.environ.get("MON_PRICE") and os.environ.get("MON_CREATED"):
    try:
        c = datetime.datetime.strptime(os.environ["MON_CREATED"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)
        hours = round((now - c).total_seconds() / 3600, 3); cost = round(hours * float(os.environ["MON_PRICE"]), 2)
    except Exception:
        pass
ssh = block("SSH"); ssh_rc = kv(raw, "SSH_RC", "?")
def toint(s, d=-1):
    try: return int(s)
    except Exception: return d
watch_rc = toint(kv(ssh, "WATCH_RC"), -1)
sentinels = toint(kv(ssh, "SENTINELS"), -1)
journal_age = toint(kv(ssh, "JOURNAL_AGE_S"), -1)
windows = toint(kv(ssh, "WINDOWS"), -1)
gpu_raw = kv(ssh, "GPU")
gpu_util = None
try: gpu_util = float(gpu_raw.split(",")[0])
except Exception: pass
job_status = kv(ssh, "JOB_STATUS", "NONE")
sync_ok = toint(kv(ssh, "SYNC_OK_MTIME"), 0); sync_fail = toint(kv(ssh, "SYNC_FAIL_MTIME"), 0)
errlines = [l for l in block("ERRLINES").splitlines() if l.strip()] if "ERRLINES_BEGIN" in ssh else []

prev = {}
try: prev = json.load(open(os.path.join(out, "status.json"), encoding="utf-8"))
except Exception: pass
# the sentinel baseline is the last count a tick with a working ssh saw; a
# failed bundle (sentinels -1) never lowers it (review 2026-10-06, MEDIUM 5)
prev_base = (prev.get("run") or {}).get("sentinels_baseline")
if prev_base is None:
    prev_base = (prev.get("run") or {}).get("sentinels", 0) or 0
ssh_ok = ssh_rc == "0" and sentinels >= 0
baseline = sentinels if ssh_ok else prev_base
# The balance is read every 10th tick; the other ticks carry the last read
# forward with its instant instead of printing None (S0F-38, 2026-10-07).
balance_utc = now_s if balance is not None else None
if balance is None and (prev.get("pod") or {}).get("balance") is not None:
    balance = prev["pod"]["balance"]; balance_utc = prev["pod"].get("balance_utc")
# Before the first cell writes, the run root has no cells/ directory and
# watch_campaign reports an error; say what it is instead.
run_dir = kv(ssh, "RUN_DIR", "1") == "1"
verdict = kv(ssh, "WATCH_VERDICT")
if ssh_ok and not run_dir:
    verdict = "no cell written yet (the run root has no cells/ directory)"

alerts = []
if ssh_rc not in ("0",): alerts.append(("HARD", f"ssh bundle failed (rc={ssh_rc}); the pod did not answer"))
if runtime != "running": alerts.append(("HARD", f"runtimeStatus={runtime} (desiredStatus={desired or '?'})"))
if not wd_alive: alerts.append(("HARD", f"watchdog not ALIVE: {watchdog or 'no status line'}"))
seatbelt_floor = float(os.environ.get("MON_SEATBELT_FLOOR", "60"))
if seatbelt_min_left is not None and seatbelt_min_left < seatbelt_floor:
    alerts.append(("HARD", f"seatbelt fires in {seatbelt_min_left} min (floor {seatbelt_floor:g}); the pod and its unpulled data go with it (job {job_status})"))
if balance is not None:
    try:
        if float(balance) < float(os.environ["MON_FLOOR"]): alerts.append(("HARD", f"balance {balance} under the floor {os.environ['MON_FLOOR']}"))
    except Exception: pass
if job_status == "CRASHED": alerts.append(("HARD", "the run job CRASHED (pid gone, no status file)"))
if ssh_ok and sentinels > prev_base: alerts.append(("HARD", f"new failure sentinel(s): {sentinels} ({kv(ssh, 'SENTINEL_LIST')})"))
stall_s = int(os.environ["MON_STALL"]) * 60
if journal_age > stall_s and gpu_util is not None and gpu_util < 5 and job_status == "RUNNING":
    alerts.append(("HARD", f"journal {journal_age}s old with GPU util {gpu_util}% and the job RUNNING (the idle-cell shape, S0F-16)"))
if watch_rc in (3, 4): alerts.append(("SOFT", f"watch_campaign verdict: {kv(ssh, 'WATCH_VERDICT')} (rc {watch_rc})"))
if sync_fail > sync_ok and sync_fail > 0: alerts.append(("SOFT", "a sync FAILURE marker is newer than the last success marker"))

# errors.jsonl: de-duplicated by line text over the file's whole history
err_path = os.path.join(out, "errors.jsonl")
seen = set()
if os.path.exists(err_path):
    for line in open(err_path, encoding="utf-8"):
        try: seen.add(json.loads(line)["line"])
        except Exception: pass
new_err = []
with open(err_path, "a", encoding="utf-8") as fh:
    for l in errlines:
        text = re.sub(r"^\d+:", "", l)
        if text in seen: continue
        seen.add(text); new_err.append(text)
        fh.write(json.dumps({"ts_utc": now_s, "source": "job log", "line": text, "first_seen_utc": now_s}) + "\n")
if any("Traceback" in l for l in new_err): alerts.append(("SOFT", f"new Traceback in the job log ({sum('Traceback' in l for l in new_err)} line(s))"))

status = {
    "schema": "cage-monitor-status-v1", "tick": tick, "tick_utc": now_s,
    "pod": {"id": os.environ["MON_POD"], "runtime_status": runtime, "desired_status": desired or None,
            "watchdog": watchdog, "watchdog_alive": wd_alive, "seatbelt_min_left": seatbelt_min_left,
            "hours": hours, "cost_usd": cost, "balance": balance, "balance_utc": balance_utc},
    "run": {"watch_rc": watch_rc, "watch_verdict": verdict, "run_dir": run_dir, "sentinels": sentinels,
            "sentinels_baseline": baseline, "ssh_ok": ssh_ok,
            "journal_age_s": journal_age, "windows": windows, "regime": kv(ssh, "REGIME")[:300],
            "gpu": gpu_raw, "gpu_util": gpu_util, "compute_apps": toint(kv(ssh, "COMPUTE_APPS"), -1),
            "job_status": job_status, "sync_ok_mtime": sync_ok, "sync_fail_mtime": sync_fail, "ssh_rc": ssh_rc},
    "alerts": [{"level": lv, "text": tx} for lv, tx in alerts],
    "new_error_lines": len(new_err),
}
tmp = os.path.join(out, "status.json.tmp")
json.dump(status, open(tmp, "w", encoding="utf-8"), indent=2); os.replace(tmp, os.path.join(out, "status.json"))
line = (f"{now_s} tick={tick} pod={runtime} wd={'ALIVE' if wd_alive else 'NOT-ALIVE'} seatbelt_min={seatbelt_min_left} "
        f"cost=${cost} job={job_status} verdict={verdict[:60]!r} sentinels={sentinels} "
        f"journal_age={journal_age}s windows={windows} gpu={gpu_raw!r} new_err={len(new_err)} alerts={len(alerts)}")
open(os.path.join(out, "monitor.log"), "a", encoding="utf-8").write(line + "\n")
print(line)
hard = [a for a in alerts if a[0] == "HARD"]
if alerts:
    with open(os.path.join(out, "alerts.log"), "a", encoding="utf-8") as fh:
        for lv, tx in alerts: fh.write(f"{now_s} {lv} {tx}\n")
    print("PORTAL ACTION")
    print(f"  resource: pod   id: {os.environ['MON_POD']}   cost clock: ${cost} ({hours} h)   runtimeStatus: {runtime}")
    print("  verdict:  NOT YET (a run is in flight; results are not pulled)")
    for lv, tx in alerts: print(f"  {lv}: {tx}")
    print(f"  stop the bleeding: bash {os.environ['MON_SCRIPTS']}/runpod/teardown_pod.sh {os.environ['MON_POD']} <backup> <local>   (only after the pull)")
sys.exit(3 if hard else 0)
PY
  if [ "$ALERT_RC" -eq 3 ] && [ "$STOP_ON_ALERT" -eq 1 ]; then printf '[monitor] HARD alert with --stop-on-alert: exiting 3\n'; exit 3; fi
  [ "$ONCE" -eq 1 ] && exit 0
  sleep "$INTERVAL" &
  wait $!
done
