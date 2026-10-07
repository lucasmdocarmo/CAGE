#!/bin/bash
# Order:     alongside stages 3 to 11 of scripts/6_experiments/cage_experiment.sh: every pod-side stage runs through it
# Objective: Submit, poll, tail, wait on, kill and fetch detached jobs on a RunPod pod over plain ssh, with a local JSON handle per job (the RunPod port of gcp/remote_job.sh)
# Cloud:     runpod
# =============================================================================
# pod_job.sh: submit / poll / stream / reap a LONG-RUNNING command on a RunPod
# pod over ssh (ADR-0143, design of 2026-10-03 section 5.5).
#
# WHY. An agent's shell is not a terminal somebody watches. A blocking
# `ssh pod 'bash setup.sh'` of 25 minutes hits a tool timeout, the caller sees
# a truncated failure and retries, and two bootstraps race on one GPU. An SSH
# drop silently orphans work that keeps billing. So every long command gets a
# HANDLE (the remote pid), a LOG, a STATUS file (its exit code) and a LOCAL
# state JSON that stores the poll and cancel commands verbatim, which is what
# lets a later turn, a compacted context or another operator resume it knowing
# only the job name. The Mac has no setsid, flock or timeout (drift sweep
# 2026-10-06), so the detachment happens on the pod, where they exist.
#
# CONTRACT  SUBMIT -> POLL -> STREAM (bounded) -> FINISH (exit code) -> REAP
#
# USAGE
#   pod_job.sh submit <name> '<command>' [deadline_s]   detached; prints the remote pid
#   pod_job.sh status <name>            RUNNING | DONE(0) | FAILED(n) | KILLED | CRASHED | LOST | UNKNOWN
#   pod_job.sh tail   <name> [lines]    bounded log read (default 40), never a firehose
#   pod_job.sh grep   <name> [pattern]  error triage over the remote log
#   pod_job.sh wait   <name> [seconds]  poll with backoff up to a HARD deadline (default 1800); 124 at the deadline
#   pod_job.sh kill   <name>            TERM then KILL the remote process GROUP of the recorded pid only
#   pod_job.sh fetch  <name> [dir]      copy the remote log into <dir> (default the handle dir)
#   pod_job.sh list
#
# ENV
#   CAGE_POD_SSH        user@host of the pod (required for every verb but list)
#   CAGE_POD_SSH_PORT   ssh port (default 22)
#   CAGE_SSH_KEY        private key path (optional; ssh's default identity otherwise)
#   CAGE_JOBS_DIR       local handle dir (default <repo>/.agent/pod_jobs); the master
#                       points it at experiments/<S>/<date>/extras/jobs
#
# LOCAL STATE   $CAGE_JOBS_DIR/<name>.json
#   { id, mode:"pod-ssh", host, port, handle:"pid:<remote_pid>", remote_log,
#     remote_status, remote_pid_file, poll_cmd, cancel_cmd, submitted_at,
#     deadline_at, deadline_s, billable:true }        instants are UTC with Z
# REMOTE STATE  ~/.cage_jobs/<name>.{cmd,pid,log,status}  status = the bare exit code
#
# NOTES
#   - The command ships BASE64-encoded: local shell -> ssh -> remote shell quoting
#     is a minefield; base64 removes it. The command runs under `bash`, from the
#     pod user's HOME; callers prefix `cd ~/CAGE && ...` themselves.
#   - Kills go to the RECORDED pid's process group (setsid made it a group
#     leader), never to a name pattern (CLAUDE.md process safety).
#   - Status is read from the status file first; a live pid whose command line
#     names this job's .cmd is RUNNING; a live pid that is something else is
#     LOST (pid reuse, the J8 guard); a dead pid with no status is CRASHED (the
#     pod rebooted or the shell was killed before the exit code landed).
#   - Every submit starts a FRESH remote log: a marker from an earlier attempt
#     can never satisfy a later attempt's check (review 2026-10-06).
# =============================================================================
set -euo pipefail

# shellcheck source=scripts/lib/_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/_common.sh"

HOST="${CAGE_POD_SSH:-}"
PORT="${CAGE_POD_SSH_PORT:-22}"
KEY="${CAGE_SSH_KEY:-}"
DIR="${CAGE_JOBS_DIR:-$CAGE_ROOT/.agent/pod_jobs}"
RDIR='~/.cage_jobs'
mkdir -p "$DIR"

iso()  { date -u +%Y-%m-%dT%H:%M:%SZ; }
now()  { date +%s; }
b64()  { if base64 --help 2>&1 | grep -q -- '-w'; then base64 -w0; else base64 | tr -d '\n'; fi; }
# deadline instant: BSD date (-v) on the Mac, GNU date (-d) elsewhere.
plus_s() {
  date -u -v+"$1"S +%Y-%m-%dT%H:%M:%SZ 2>/dev/null \
    || date -u -d "@$(( $(now) + $1 ))" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null \
    || echo unknown
}

need_host() { [ -n "$HOST" ] || die "CAGE_POD_SSH is unset (user@host of the pod)"; }

# One bounded ssh round trip. Never streams; the caller decides what to read.
rssh() {
  need_host
  if [ -n "$KEY" ]; then
    ssh -p "$PORT" -i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=25 "$HOST" "$1"
  else
    ssh -p "$PORT" -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=25 "$HOST" "$1"
  fi
}

# Job names are spliced into remote shell strings and remote paths: constrain
# them so a hostile or mistyped name cannot inject or escape ~/.cage_jobs.
check_name() {
  case "${1:?job name required}" in
    *[!A-Za-z0-9._-]*|'') die "invalid job name '$1' (allowed: A-Za-z0-9._-)" ;;
  esac
}

_rpid() { [ -f "$DIR/$1.json" ] && sed -n 's/.*"handle": "pid:\([0-9]*\)".*/\1/p' "$DIR/$1.json" || true; }

cmd_submit() {
  local name="${1:?name required}" command="${2:?command required}" deadline="${3:-1800}"
  check_name "$name"
  printf '%s' "$deadline" | grep -qE '^[0-9]+$' || die "deadline_s must be an integer: $deadline"
  local enc; enc="$(printf '%s' "$command" | b64)"

  # Refuse to double-submit a live job: a retry must not spawn a second run.
  local s; s="$(cmd_status "$name" 2>/dev/null || true)"
  [ "$s" = "RUNNING" ] && die "job '$name' is already RUNNING on $HOST (kill it or use another name)"

  # setsid: own process group, so the job survives the ssh channel closing.
  # ( ... ); echo $? > status : the exit code is the ONLY durable record.
  local pid
  pid="$(rssh "
    mkdir -p $RDIR
    rm -f $RDIR/$name.status
    : > $RDIR/$name.log
    printf '%s' '$enc' | base64 -d > $RDIR/$name.cmd
    nohup setsid bash -c '( bash $RDIR/$name.cmd ) >> $RDIR/$name.log 2>&1; echo \$? > $RDIR/$name.status' >/dev/null 2>&1 &
    echo \$! > $RDIR/$name.pid
    echo \$!
  " | tr -d '\r\n ')"
  [ -n "$pid" ] || die "no remote pid for '$name' (ssh problem?)"
  printf '%s' "$pid" | grep -qE '^[0-9]+$' || die "remote pid for '$name' is not a number: '$pid'"

  local poll_cmd="ssh -p $PORT $HOST 'cat $RDIR/$name.status 2>/dev/null || (kill -0 $pid 2>/dev/null && echo RUNNING)'"
  local cancel_cmd="CAGE_POD_SSH=$HOST CAGE_POD_SSH_PORT=$PORT scripts/6_experiments/pod_job.sh kill $name"
  cat > "$DIR/$name.json" <<EOF
{
  "id": "$name",
  "mode": "pod-ssh",
  "host": "$HOST",
  "port": "$PORT",
  "handle": "pid:$pid",
  "remote_log": "$RDIR/$name.log",
  "remote_status": "$RDIR/$name.status",
  "remote_pid_file": "$RDIR/$name.pid",
  "poll_cmd": "$poll_cmd",
  "cancel_cmd": "$cancel_cmd",
  "submitted_at": "$(iso)",
  "deadline_at": "$(plus_s "$deadline")",
  "deadline_s": $deadline,
  "billable": true
}
EOF
  printf 'submitted %s -> %s (remote pid %s)\n  log:   %s:%s\n  state: %s\n' \
    "$name" "$HOST" "$pid" "$HOST" "$RDIR/$name.log" "$DIR/$name.json"
}

cmd_status() {
  local name="${1:?name required}" pid out
  check_name "$name"
  pid="$(_rpid "$name")"
  [ -n "$pid" ] || { echo "UNKNOWN"; return 1; }
  # The checks are not atomic: a job that ends between the status read and
  # the ps read must report its exit code, never LOST (S0F-32: the suite's
  # fake jobs, 2026-10-07), so the status file is re-read after any miss.
  out="$(rssh "if [ -f $RDIR/$name.status ]; then cat $RDIR/$name.status; elif kill -0 $pid 2>/dev/null && ps -o args= -p $pid 2>/dev/null | grep -q '$name.cmd'; then echo RUNNING; elif [ -f $RDIR/$name.status ]; then cat $RDIR/$name.status; elif kill -0 $pid 2>/dev/null; then echo LOST; else echo CRASHED; fi" | tr -d '\r\n ')"
  case "$out" in
    RUNNING) echo "RUNNING"; return 0 ;;
    LOST)    echo "LOST";    return 1 ;;   # a live pid that is not our job (pid reuse)
    CRASHED) echo "CRASHED"; return 1 ;;
    0)       echo "DONE(0)"; return 0 ;;
    143|137) echo "KILLED";  return 1 ;;
    "")      echo "UNKNOWN"; return 1 ;;
    *)       echo "FAILED($out)"; return 1 ;;
  esac
}

cmd_tail() {
  local name="${1:?}" n="${2:-40}"
  check_name "$name"
  printf '%s' "$n" | grep -qE '^[0-9]+$' || die "lines must be an integer: $n"
  rssh "tail -n $n $RDIR/$name.log 2>/dev/null || echo '(no log yet)'"
}

cmd_grep() {
  local name="${1:?}" pat="${2:-Traceback|REFUS|FAIL|ERROR|exit 2|CAMPAIGN TELEMETRY|UNKNOWN_TELEMETRY|RELAUNCH FAILED|STOP FAILED|out of memory}"
  check_name "$name"
  rssh "grep -nE '$pat' $RDIR/$name.log 2>/dev/null | tail -n 40 || echo '(no matches)'"
}

# Poll with backoff to a HARD deadline. A job past its deadline is reported
# (124) and left to the caller's judgment, never silently killed.
cmd_wait() {
  local name="${1:?}" limit="${2:-1800}"
  check_name "$name"
  printf '%s' "$limit" | grep -qE '^[0-9]+$' || die "seconds must be an integer: $limit"
  local deadline=$(( $(now) + limit )) delay="${CAGE_POD_JOB_POLL_S:-10}" s
  while :; do
    s="$(cmd_status "$name" || true)"
    case "$s" in
      DONE*)                               echo "$s"; return 0 ;;
      FAILED*|KILLED|CRASHED|LOST|UNKNOWN) echo "$s"; return 1 ;;
    esac
    if [ "$(now)" -ge "$deadline" ]; then
      echo "DEADLINE_EXCEEDED after ${limit}s: '$name' still RUNNING on $HOST (still billing)."
      echo "  kill it:  CAGE_POD_SSH=$HOST CAGE_POD_SSH_PORT=$PORT scripts/6_experiments/pod_job.sh kill $name"
      return 124
    fi
    sleep "$delay"
    [ "$delay" -lt 60 ] && delay=$(( delay * 2 )) || delay=60
  done
}

cmd_kill() {
  local name="${1:?}" pid
  check_name "$name"
  pid="$(_rpid "$name")"
  [ -n "$pid" ] || die "no remote pid recorded for '$name'"
  # The recorded pid's process group first (setsid made it the leader), the
  # bare pid as the fallback; 143 lands in the status file either way.
  rssh "kill -- -$pid 2>/dev/null || kill $pid 2>/dev/null || true; sleep 2; kill -9 -- -$pid 2>/dev/null || kill -9 $pid 2>/dev/null || true; echo 143 > $RDIR/$name.status" >/dev/null || true
  echo "killed $name (remote pid $pid on $HOST)"
}

cmd_fetch() {
  local name="${1:?}" dest="${2:-$DIR}"
  check_name "$name"
  need_host
  mkdir -p "$dest"
  if [ -n "$KEY" ]; then
    scp -P "$PORT" -i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=no "$HOST:.cage_jobs/$name.log" "$dest/$name.log" \
      && echo "fetched -> $dest/$name.log" || die "fetch failed for '$name'"
  else
    scp -P "$PORT" -o BatchMode=yes -o StrictHostKeyChecking=no "$HOST:.cage_jobs/$name.log" "$dest/$name.log" \
      && echo "fetched -> $dest/$name.log" || die "fetch failed for '$name'"
  fi
}

cmd_list() {
  printf '%-26s %-12s %-10s %s\n' NAME STATE PID HANDLE
  local f name
  for f in "$DIR"/*.json; do
    [ -e "$f" ] || continue
    name="$(basename "$f" .json)"
    printf '%-26s %-12s %-10s %s\n' "$name" "$(cmd_status "$name" 2>/dev/null || echo UNKNOWN)" "$(_rpid "$name")" "$f"
  done
}

case "${1:-}" in
  submit) shift; cmd_submit "$@" ;;
  status) shift; cmd_status "$@" ;;
  tail)   shift; cmd_tail   "$@" ;;
  grep)   shift; cmd_grep   "$@" ;;
  wait)   shift; cmd_wait   "$@" ;;
  kill)   shift; cmd_kill   "$@" ;;
  fetch)  shift; cmd_fetch  "$@" ;;
  list)   shift; cmd_list   "$@" ;;
  *) sed -n '5,46p' "$0" >&2; exit 2 ;;
esac
