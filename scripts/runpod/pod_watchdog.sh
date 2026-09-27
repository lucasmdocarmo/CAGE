#!/usr/bin/env bash
# Order:     provisioning bracket: armed by provision_pod.sh right after the pod create (before the ledger create event), disarmed by teardown_pod.sh right after the delete
# Objective: Client-side cost seatbelt: a detached workstation loop that deletes ONE pod at an absolute UTC deadline (RunPod has no server-side auto-terminate; verified live 2026-09-26)
# Cloud:     runpod
# pod_watchdog.sh: the cost seatbelt that actually exists.
#
# WHY THIS FILE EXISTS (W27, live smoke test 2026-09-26): `runpodctl pod create`
# has NO --terminate-after and NO --stop-after (2.11.0 and 2.14.0 both answer
# {"error":"unknown flag: --terminate-after","code":"usage_error"}); the v2 API
# CreatePodRequest has no auto-terminate field; docs.runpod.io/pods/manage-pods
# documents only a client-side `sleep 2h; runpodctl pod stop`. So the deadline
# is enforced from HERE, the workstation: a detached loop compares the wall
# clock to the deadline every tick and runs `runpodctl pod delete` once it
# passes. It dies with this machine: a laptop that is off or asleep at the
# deadline fires at the next wake, and the pod bills until then. Backstops that
# survive this machine: pod_status.sh --max-age-hours (the alarm) and the RunPod
# console. The API key never leaves this workstation (~/.runpod/config.toml).
#
# Usage:
#   pod_watchdog.sh arm    <pod_id> <deadline>   deadline = RFC3339 UTC, e.g. 2026-09-27T00:11:49Z
#   pod_watchdog.sh status [<pod_id>]
#   pod_watchdog.sh disarm <pod_id>
#   pod_watchdog.sh run    <pod_id> <deadline>   the loop body; `arm` spawns it detached
#   pod_watchdog.sh check  <deadline>            validate a deadline (format, calendar, future) without arming; exit 0 or 1
#
# State (beside the pod ledger, default results/ops/; override CAGE_POD_LEDGER):
#   watchdog_<pod_id>.pid       the loop's PID (identity-checked on read: J8 PID-reuse guard); removing it disarms
#   watchdog_<pod_id>.deadline  the RFC3339 instant
#   watchdog_<pod_id>.log       the loop's log
#   pod_ledger.jsonl            on a successful fire, one appended line:
#     {"ts_utc":"...Z","pod_id":"...","event":"delete","by":"watchdog"}
#
# Env:  CAGE_POD_LEDGER             ledger path (the state dir is its directory)
#       CAGE_WATCHDOG_TICK          seconds between clock checks and between delete attempts (default 30)
#       CAGE_WATCHDOG_RETRIES       delete attempts before giving up LOUDLY (default 20)
#       CAGE_WATCHDOG_CLI_TIMEOUT   hard bound on each runpodctl call at fire time (default 120 s;
#                                   macOS has no timeout(1), so the bound is a killer subshell)
# Disarm paths: `disarm`, or removing the pidfile (the loop exits at its next tick
# without firing once it has seen its pidfile vanish or name another pid).
# Exit: 0 ok · 1 the pod outlived every attempt (run), a live watchdog refused a
#       second arm, or the spawned loop died at start · 2 usage
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/_common.sh
source "$SCRIPT_DIR/../lib/_common.sh"
command -v runpodctl >/dev/null 2>&1 || PATH="$PATH:/opt/homebrew/bin:/usr/local/bin"

WD_KILLER_PID=""   # the CLI-timeout killer subshell cli_timed has in flight, if any
cleanup() {
  # On ANY exit (a TERM from disarm included): stop the killer subshell this
  # script started, by its recorded pid, so no signal it would send outlives us.
  # State files are owned by arm/run/disarm explicitly.
  if [ -n "$WD_KILLER_PID" ]; then
    if kill "$WD_KILLER_PID" 2>/dev/null; then wlog "exit: stopped the CLI killer (pid $WD_KILLER_PID)"; fi
    WD_KILLER_PID=""
  fi
}
trap 'rc=$?; cleanup; exit $rc' EXIT
trap 'exit 130' INT TERM

usage() {
  printf 'usage: %s arm <pod_id> <deadline-RFC3339Z> | status [<pod_id>] | disarm <pod_id> | run <pod_id> <deadline-RFC3339Z> | check <deadline-RFC3339Z>\n' "$0" >&2
  exit 2
}

LEDGER="${CAGE_POD_LEDGER:-$CAGE_ROOT/results/ops/pod_ledger.jsonl}"
STATE_DIR="$(dirname "$LEDGER")"
TICK="${CAGE_WATCHDOG_TICK:-30}"
RETRIES="${CAGE_WATCHDOG_RETRIES:-20}"
CLI_TIMEOUT="${CAGE_WATCHDOG_CLI_TIMEOUT:-120}"
# The loop knobs are validated by the subcommands that use them (arm, run), not
# by check/status/disarm: a deadline check must not fail on an unrelated knob.
validate_loop_env() {
  printf '%s' "$TICK" | grep -qE '^[1-9][0-9]*$' || die "CAGE_WATCHDOG_TICK must be a positive integer: $TICK"
  printf '%s' "$RETRIES" | grep -qE '^[1-9][0-9]*$' || die "CAGE_WATCHDOG_RETRIES must be a positive integer: $RETRIES"
  printf '%s' "$CLI_TIMEOUT" | grep -qE '^[1-9][0-9]*$' || die "CAGE_WATCHDOG_CLI_TIMEOUT must be a positive integer: $CLI_TIMEOUT"
}

# The pod-id grammar matches what provision_pod.sh parses out of the create
# response ([A-Za-z0-9-]+), so no id the wrapper accepts is refused here.
valid_pod_id()   { printf '%s' "$1" | grep -qE '^[A-Za-z0-9-]{6,64}$'; }
valid_deadline() { printf '%s' "$1" | grep -qE '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$'; }
# date flavor, detected once: BSD date (macOS) has -j and -r <epoch>; GNU date has
# -d <string> and -d @<epoch> (and its -r reads a FILE's mtime, so it is never used there).
DATE_FLAVOR="gnu"; date -j -u +%s >/dev/null 2>&1 && DATE_FLAVOR="bsd"
epoch_of() {  # RFC3339 UTC -> epoch seconds.
  # Round-trips the instant: BSD date silently normalizes a calendar-invalid
  # deadline (2099-02-30 becomes March 2), which would make the recorded and the
  # firing instants differ; such a deadline is refused instead.
  local s="$1" e back
  if [ "$DATE_FLAVOR" = bsd ]; then
    e="$(date -j -u -f '%Y-%m-%dT%H:%M:%SZ' "$s" '+%s' 2>/dev/null)" || die "cannot parse deadline: $s"
    back="$(date -u -r "$e" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || true)"
  else
    e="$(date -u -d "$s" '+%s' 2>/dev/null)" || die "cannot parse deadline: $s"
    back="$(date -u -d "@$e" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || true)"
  fi
  [ "$back" = "$s" ] || die "deadline $s is not a calendar-valid instant (it normalizes to ${back:-?})"
  printf '%s' "$e"
}
# cli_timed <cmd...>: one runpodctl call under a hard timeout. A hung CLI at the
# deadline must count as a failed attempt, never stall the seatbelt. The bound
# covers the direct child only (runpodctl is one process and spawns none [A]).
cli_timed() {
  local p k rc
  "$@" & p=$!
  # The killer subshell must NOT inherit stdout/stderr: inside a $(...) capture
  # its `sleep` would hold the pipe open and block the caller for CLI_TIMEOUT.
  # Before it signals, it re-checks that $p is still alive AND still a runpodctl
  # process, so a recycled pid is never hit; its pid is recorded in WD_KILLER_PID
  # so cleanup() stops it if this script is terminated while the call is in flight.
  ( sleep "$CLI_TIMEOUT"
    if kill -0 "$p" 2>/dev/null; then
      case "$(ps -ww -p "$p" -o command= 2>/dev/null)" in *runpodctl*) kill "$p" 2>/dev/null ;; esac
    fi ) >/dev/null 2>&1 & k=$!
  WD_KILLER_PID="$k"
  wait "$p" && rc=0 || rc=$?
  # Only the recorded pid of the subshell we started is signaled (process safety
  # rule: never by parent pid or name pattern); its orphaned `sleep` exits on its
  # own within CLI_TIMEOUT and holds no pipe (redirected above).
  kill "$k" 2>/dev/null || true
  wait "$k" 2>/dev/null || true
  WD_KILLER_PID=""
  return "$rc"
}
now_utc()  { date -u +%Y-%m-%dT%H:%M:%SZ; }
pidfile()  { printf '%s/watchdog_%s.pid' "$STATE_DIR" "$1"; }
deadfile() { printf '%s/watchdog_%s.deadline' "$STATE_DIR" "$1"; }
logfile()  { printf '%s/watchdog_%s.log' "$STATE_DIR" "$1"; }
ident()    { printf 'pod_watchdog.sh run %s' "$1"; }   # the pidfile_alive identity token
wlog()     { printf '[watchdog %s] %s\n' "$(now_utc)" "$*"; }

# pod_listed <pod_id> -> 0 listed · 1 not listed · 2 unknown (the listing failed,
# was empty, or was not the CLI's JSON list contract). FAIL-CLOSED: only a parsed
# JSON list (bare, or under "pods" / "data") can say "not listed"; an error object
# or any other shape is UNKNOWN, because a false "gone" would close the ledger on
# a pod that still bills.
pod_listed() {
  local out
  out="$(cli_timed runpodctl pod list --all 2>/dev/null)" || return 2
  [ -n "$out" ] || return 2
  printf '%s' "$out" | CAGE_WD_POD="$1" python3 -c '
import json, os, sys
want = os.environ["CAGE_WD_POD"]; text = sys.stdin.read()
try:
    data = json.loads(text)
except Exception:
    sys.exit(2)
if isinstance(data, dict):
    data = data.get("pods") if isinstance(data.get("pods"), list) else data.get("data")
if not isinstance(data, list):
    sys.exit(2)
ids = [d.get("id") for d in data if isinstance(d, dict) and "id" in d]
if data and not ids:
    sys.exit(2)   # entries without an id field: not the contract we know, so UNKNOWN
sys.exit(0 if want in ids else 1)'
}

# ledger_has_delete <pod_id> -> 0 iff the ledger already closes the pod (teardown got there first)
ledger_has_delete() {
  [ -f "$LEDGER" ] || return 1
  CAGE_WD_POD="$1" python3 -c '
import json, os, sys
want = os.environ["CAGE_WD_POD"]
for raw in open(sys.argv[1], encoding="utf-8"):
    raw = raw.strip()
    if not raw:
        continue
    try:
        e = json.loads(raw)
    except ValueError:
        continue
    if isinstance(e, dict) and e.get("event") == "delete" and e.get("pod_id") == want:
        sys.exit(0)
sys.exit(1)' "$LEDGER"
}

cmd_arm() {
  local pod="$1" dl="$2" pf lf pid dl_epoch
  validate_loop_env
  valid_pod_id "$pod" || die "arm: pod_id must match [A-Za-z0-9-]{6,64}: '$pod'"
  valid_deadline "$dl" || die "arm: deadline must be RFC3339 UTC (YYYY-MM-DDTHH:MM:SSZ): '$dl'"
  dl_epoch="$(epoch_of "$dl")" || exit 1
  [ "$dl_epoch" -gt "$(date -u +%s)" ] || die "arm: deadline $dl is not in the future (now $(now_utc))"
  require_cmd runpodctl "brew install runpod/runpodctl/runpodctl"
  require_cmd python3
  require_cmd nohup
  pf="$(pidfile "$pod")"; lf="$(logfile "$pod")"
  if pidfile_alive "$pf" "$(ident "$pod")"; then
    die "arm: a live watchdog already guards $pod (pid $(cat "$pf"), deadline $(cat "$(deadfile "$pod")" 2>/dev/null || printf '?')); disarm it first"
  fi
  mkdir -p "$STATE_DIR" || die "cannot create the watchdog state dir $STATE_DIR"
  # pidfile_alive just proved any existing pidfile DEAD: remove it before the
  # spawn, or the new loop's first tick would read a foreign pid and exit as
  # "superseded" (its grace covers an ABSENT pidfile only).
  rm -f "$pf"
  printf '%s\n' "$dl" > "$(deadfile "$pod")"
  # Detached: nohup + background + disown (setsid is util-linux, absent on macOS).
  nohup bash "$SCRIPT_DIR/pod_watchdog.sh" run "$pod" "$dl" >> "$lf" 2>&1 &
  pid=$!
  disown "$pid" 2>/dev/null || true
  printf '%s\n' "$pid" > "$pf.tmp" && mv -f "$pf.tmp" "$pf"   # atomic: the loop never reads a truncated pidfile
  # Keep an idle Mac awake while the loop lives (a closed lid still sleeps it).
  if command -v caffeinate >/dev/null 2>&1; then
    nohup caffeinate -i -w "$pid" >/dev/null 2>&1 &
    disown $! 2>/dev/null || true
  fi
  sleep 1
  pidfile_alive "$pf" "$(ident "$pod")" || die "arm: the watchdog for $pod died at start; see $lf"
  log "watchdog ARMED for pod $pod: deletes it at $dl (client-side: dies with this machine; backstops = pod_status.sh --max-age-hours + the RunPod console) pid=$pid log=$lf"
  printf 'WATCHDOG_PID=%s\n' "$pid"
}

cmd_run() {
  local pod="$1" dl="$2" dl_epoch n rc pf seen=0
  validate_loop_env
  valid_pod_id "$pod" || die "run: bad pod_id '$pod'"
  valid_deadline "$dl" || die "run: bad deadline '$dl'"
  dl_epoch="$(epoch_of "$dl")" || exit 1
  pf="$(pidfile "$pod")"
  wlog "guarding pod $pod until $dl (tick ${TICK}s, pid $$)"
  # `sleep` in the background plus `wait` keeps the loop signal-responsive: bash
  # runs a pending trap only after the foreground command returns, so a plain
  # `sleep 30` would delay `disarm` by up to one tick.
  while [ "$(date -u +%s)" -lt "$dl_epoch" ]; do
    # The pidfile is the arming record. Once this loop has seen it, its removal
    # (an operator rm, a test's tmp dir pruned) or another pid in it (re-armed)
    # means DISARMED: exit without firing. `arm` writes the pidfile right after
    # the spawn, so the first tick may legitimately not see it yet.
    if [ -f "$pf" ]; then
      if [ "$(cat "$pf" 2>/dev/null)" = "$$" ]; then seen=1
      else wlog "pidfile $pf names another pid: superseded, exiting without firing"; return 0; fi
    elif [ "$seen" -eq 1 ]; then
      wlog "pidfile $pf removed: disarmed, exiting without firing"; return 0
    fi
    sleep "$TICK" & wait $! || true
  done
  wlog "deadline reached: deleting pod $pod (cost-stopping action)"
  n=0
  while [ "$n" -lt "$RETRIES" ]; do
    n=$((n + 1))
    if cli_timed runpodctl pod delete "$pod" >/dev/null 2>&1; then
      wlog "attempt $n: pod delete accepted"
    else
      wlog "attempt $n: pod delete returned nonzero (already gone, or auth/network); checking the listing"
    fi
    pod_listed "$pod" && rc=0 || rc=$?
    if [ "$rc" -eq 1 ]; then
      if ledger_has_delete "$pod"; then
        wlog "ledger already closes $pod (teardown got there first); nothing to append"
      else
        mkdir -p "$(dirname "$LEDGER")" 2>/dev/null || true
        if printf '{"ts_utc":"%s","pod_id":"%s","event":"delete","by":"watchdog"}\n' "$(now_utc)" "$pod" >> "$LEDGER" 2>/dev/null; then
          wlog "ledger: delete event appended (by watchdog) -> $LEDGER"
        else
          wlog "WARNING: could not append the delete event to $LEDGER; the pod IS gone, record it by hand"
        fi
      fi
      rm -f "$(pidfile "$pod")" "$(deadfile "$pod")"
      wlog "DONE: pod $pod is no longer listed; watchdog exiting"
      return 0
    fi
    if [ "$rc" -eq 0 ]; then
      wlog "attempt $n/$RETRIES: pod $pod still listed (listing rc=0); retrying in ${TICK}s"
    else
      wlog "attempt $n/$RETRIES: listing UNKNOWN (listing rc=$rc: CLI failed, empty, or not a JSON list of pods); the pod may still bill; retrying in ${TICK}s"
    fi
    sleep "$TICK" & wait $! || true
  done
  wlog "FAILED: pod $pod STILL LISTED after $RETRIES attempts; it may STILL BE BILLING. Delete it by hand: runpodctl pod delete $pod (or the RunPod console)."
  return 1
}

cmd_check() {  # the one validator provision_pod.sh calls BEFORE the plan print and the create
  local dl="$1" dl_epoch
  valid_deadline "$dl" || die "check: deadline must be RFC3339 UTC (YYYY-MM-DDTHH:MM:SSZ): '$dl'"
  dl_epoch="$(epoch_of "$dl")" || exit 1
  [ "$dl_epoch" -gt "$(date -u +%s)" ] || die "check: deadline $dl is not in the future (now $(now_utc))"
  printf 'OK %s\n' "$dl_epoch"
}

cmd_status() {
  local want="${1:-}" pf pod pid state dl found=0
  for pf in "$STATE_DIR"/watchdog_*.pid; do
    [ -f "$pf" ] || continue
    pod="$(basename "$pf" .pid)"; pod="${pod#watchdog_}"
    [ -z "$want" ] || [ "$pod" = "$want" ] || continue
    found=1
    pid="$(cat "$pf" 2>/dev/null || printf '?')"
    dl="$(cat "$(deadfile "$pod")" 2>/dev/null || printf '?')"
    if pidfile_alive "$pf" "$(ident "$pod")"; then
      state="ALIVE"
    else
      state="DEAD (stale pidfile: the pod is UNGUARDED on this machine; re-arm or tear down)"
    fi
    printf 'pod=%s watchdog=%s pid=%s deadline=%s log=%s\n' "$pod" "$state" "$pid" "$dl" "$(logfile "$pod")"
  done
  [ "$found" -eq 1 ] || printf 'no watchdog state under %s%s\n' "$STATE_DIR" "${want:+ for pod $want}"
}

cmd_disarm() {
  local pod="$1" pf pid
  valid_pod_id "$pod" || die "disarm: bad pod_id '$pod'"
  pf="$(pidfile "$pod")"
  if pidfile_alive "$pf" "$(ident "$pod")"; then
    pid="$(cat "$pf")"
    # Only the recorded pid, proven OURS by pidfile_alive above, is signaled
    # (process safety rule: never by parent pid, name pattern or broadcast). The
    # loop exits at once (its tick sleep runs in the background under `wait`);
    # that orphaned `sleep` expires on its own within one tick.
    kill "$pid" 2>/dev/null || true
    log "watchdog DISARMED for pod $pod (pid $pid killed)"
  elif [ -f "$pf" ]; then
    warn "no live watchdog for $pod (stale pidfile removed)"
  else
    log "no watchdog state for pod $pod (nothing to disarm)"
  fi
  rm -f "$pf" "$(deadfile "$pod")"
}

[ $# -ge 1 ] || usage
case "$1" in
  arm)       [ $# -eq 3 ] || usage; cmd_arm "$2" "$3" ;;
  run)       [ $# -eq 3 ] || usage; cmd_run "$2" "$3" ;;
  status)    [ $# -le 2 ] || usage; cmd_status "${2:-}" ;;
  disarm)    [ $# -eq 2 ] || usage; cmd_disarm "$2" ;;
  check)     [ $# -eq 2 ] || usage; cmd_check "$2" ;;
  -h|--help) usage ;;
  *) printf 'unknown subcommand: %s\n' "$1" >&2; usage ;;
esac
