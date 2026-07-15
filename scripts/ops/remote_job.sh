#!/usr/bin/env bash
# remote_job.sh — submit / poll / reap a LONG-RUNNING command on a GCP VM over SSH.
#
# WHY THIS EXISTS
#   An agent's shell is not a terminal someone is watching. A blocking
#   `gcloud compute ssh ... 'bash setup.sh'` that takes 30 min hits the tool timeout, the agent
#   sees a truncated failure, retries -- and now two setups race on one GPU box. Worse, an SSH
#   drop silently orphans work that keeps billing.
#
#   So: every long remote command gets a HANDLE (remote PID), a LOG, a STATUS file (exit code),
#   and a LOCAL state JSON that stores poll/cancel commands VERBATIM -- which is what lets a
#   FUTURE turn, a different agent, or a compacted context resume it knowing only the job name.
#
# CONTRACT:  SUBMIT -> POLL -> STREAM (bounded) -> FINISH (exit code) -> REAP
#
# USAGE
#   remote_job.sh submit <name> '<command>' [deadline_s]   # detached; prints remote pid
#   remote_job.sh status <name>            # RUNNING | DONE(0) | FAILED(n) | KILLED | CRASHED | UNKNOWN
#   remote_job.sh tail   <name> [lines]    # bounded log read (default 40) -- never firehose
#   remote_job.sh grep   <name> [pattern]  # error triage
#   remote_job.sh wait   <name> [seconds]  # poll w/ backoff up to a HARD deadline (default 1800)
#   remote_job.sh kill   <name>            # TERM then KILL the remote process GROUP
#   remote_job.sh fetch  <name>            # copy the remote log down to .agent/tasks/
#   remote_job.sh list
#
# ENV
#   CAGE_VM    (default: cage-gpu)
#   CAGE_ZONE  (default: contents of .agent/cage_zone, else us-central1-a)
#
# LOCAL STATE   .agent/tasks/<name>.remote.json
#   { id, mode:"remote-ssh", vm, zone, handle:"pid:<remote_pid>", remote_log, remote_status,
#     poll_cmd, cancel_cmd, started_at, deadline_at, billable:true }   # ISO-8601 UTC
# REMOTE STATE  ~/.cage_jobs/<name>.{cmd,pid,log,status}               # status = bare exit code
#
# NOTES
#   - The command is shipped BASE64-ENCODED. Passing shell text through
#     local shell -> gcloud -> ssh -> remote shell is a quoting minefield; base64 removes it.
#   - CLOUDSDK_CORE_DISABLE_PROMPTS=1: a confirmation prompt in a TTY-less background shell hangs
#     forever and emits nothing. This is the #1 cause of a "stuck" task.
#   - Killing a remote job does NOT stop anything it submitted to a cloud API. Cancel those with
#     the owning service's cancel command.
#   - This kills by the RECORDED PID, never `pkill -f <script>` -- that pattern matches the SSH
#     command's own shell and kills the session (exit 255). Learned the hard way.

set -euo pipefail

VM="${CAGE_VM:-cage-gpu}"
ZONE="${CAGE_ZONE:-$( [ -f .agent/cage_zone ] && cat .agent/cage_zone || echo us-central1-a )}"
DIR="${BGTASK_DIR:-.agent/tasks}"
RDIR='~/.cage_jobs'
mkdir -p "$DIR"

export CLOUDSDK_CORE_DISABLE_PROMPTS=1

die()  { printf 'remote_job: %s\n' "$*" >&2; exit 1; }
iso()  { date -u +%Y-%m-%dT%H:%M:%SZ; }
now()  { date +%s; }
b64()  { if base64 --help 2>&1 | grep -q -- '-w'; then base64 -w0; else base64 | tr -d '\n'; fi; }

# One bounded SSH round-trip. Never streams; caller decides what to read.
rssh() {
  gcloud compute ssh "$VM" --zone="$ZONE" --quiet --command="$1" \
    -- -o StrictHostKeyChecking=no -o ConnectTimeout=25 -o BatchMode=yes 2>/dev/null
}

cmd_submit() {
  local name="${1:?name required}" command="${2:?command required}" deadline="${3:-1800}"
  local enc; enc="$(printf '%s' "$command" | b64)"

  # Refuse to double-submit a live job (idempotency: a retry must not spawn a second run).
  local s; s="$(cmd_status "$name" 2>/dev/null || true)"
  [ "$s" = "RUNNING" ] && die "job '$name' is already RUNNING on $VM -- kill it or use another name"

  # setsid: own process group so it survives the SSH channel closing.
  # ( ... ); echo $? > status : the exit code is the ONLY durable record of how it ended.
  local pid
  pid="$(rssh "
    mkdir -p $RDIR
    rm -f $RDIR/$name.status
    printf '%s' '$enc' | base64 -d > $RDIR/$name.cmd
    nohup setsid bash -c '( bash $RDIR/$name.cmd ) >> $RDIR/$name.log 2>&1; echo \$? > $RDIR/$name.status' >/dev/null 2>&1 &
    echo \$!
  " | tr -d '\r\n ')"
  [ -n "$pid" ] || die "failed to obtain a remote pid for '$name' (ssh problem?)"

  local poll_cmd="gcloud compute ssh $VM --zone=$ZONE --quiet --command='cat $RDIR/$name.status 2>/dev/null || (kill -0 $pid 2>/dev/null && echo RUNNING)'"
  local cancel_cmd="scripts/ops/remote_job.sh kill $name"
  cat > "$DIR/$name.remote.json" <<EOF
{
  "id": "$name",
  "mode": "remote-ssh",
  "vm": "$VM",
  "zone": "$ZONE",
  "handle": "pid:$pid",
  "remote_log": "$RDIR/$name.log",
  "remote_status": "$RDIR/$name.status",
  "poll_cmd": "$poll_cmd",
  "cancel_cmd": "$cancel_cmd",
  "started_at": "$(iso)",
  "deadline_at": "$(date -u -v+"${deadline}"S +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -d "@$(( $(now) + deadline ))" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)",
  "billable": true
}
EOF
  printf 'submitted %s -> %s (remote pid %s)\n  log:   %s:%s\n  state: %s\n' \
    "$name" "$VM" "$pid" "$VM" "$RDIR/$name.log" "$DIR/$name.remote.json"
}

_rpid() { [ -f "$DIR/$1.remote.json" ] && sed -n 's/.*"handle": "pid:\([0-9]*\)".*/\1/p' "$DIR/$1.remote.json" || true; }

cmd_status() {
  local name="${1:?name required}" pid; pid="$(_rpid "$name")"
  [ -n "$pid" ] || { echo "UNKNOWN"; return 1; }
  local out
  out="$(rssh "if [ -f $RDIR/$name.status ]; then cat $RDIR/$name.status; elif kill -0 $pid 2>/dev/null; then echo RUNNING; else echo CRASHED; fi" | tr -d '\r\n ')"
  case "$out" in
    RUNNING) echo "RUNNING"; return 0 ;;
    CRASHED) echo "CRASHED"; return 1 ;;          # pid gone and no status written
    0)       echo "DONE(0)"; return 0 ;;
    143|137) echo "KILLED";  return 1 ;;
    "")      echo "UNKNOWN"; return 1 ;;
    *)       echo "FAILED($out)"; return 1 ;;
  esac
}

cmd_tail() { local name="${1:?}" n="${2:-40}"; rssh "tail -n $n $RDIR/$name.log 2>/dev/null || echo '(no log yet)'"; }
cmd_grep() {
  local name="${1:?}" pat="${2:-error|fail|denied|exception|traceback|out of memory|quota}"
  rssh "grep -inE '$pat' $RDIR/$name.log 2>/dev/null | tail -n 25 || echo '(no matches)'"
}

# Poll with backoff to a HARD deadline. A task past its deadline is cancelled, not "probably nearly done".
cmd_wait() {
  local name="${1:?}" limit="${2:-1800}"
  local deadline=$(( $(now) + limit )) delay=10 s
  while :; do
    s="$(cmd_status "$name" || true)"
    case "$s" in
      DONE*|FAILED*|KILLED|CRASHED|UNKNOWN) echo "$s"; [ "${s#DONE}" != "$s" ] && return 0 || return 1 ;;
    esac
    if [ "$(now)" -ge "$deadline" ]; then
      echo "DEADLINE_EXCEEDED after ${limit}s -- '$name' still RUNNING on $VM (still billing)."
      echo "  kill it:  scripts/ops/remote_job.sh kill $name"
      return 124
    fi
    sleep "$delay"
    [ "$delay" -lt 60 ] && delay=$(( delay * 2 )) || delay=60
  done
}

cmd_kill() {
  local name="${1:?}" pid; pid="$(_rpid "$name")"
  [ -n "$pid" ] || die "no remote pid recorded for '$name'"
  rssh "kill -- -$pid 2>/dev/null || kill $pid 2>/dev/null || true; sleep 2; kill -9 -- -$pid 2>/dev/null || kill -9 $pid 2>/dev/null || true; echo 143 > $RDIR/$name.status" >/dev/null || true
  echo "killed $name (remote pid $pid on $VM)"
  echo "NOTE: this stops the REMOTE PROCESS only. Anything it submitted to a cloud API keeps running."
}

cmd_fetch() {
  local name="${1:?}"
  gcloud compute scp "$VM:$RDIR/$name.log" "$DIR/$name.remote.log" --zone="$ZONE" --quiet 2>/dev/null \
    && echo "fetched -> $DIR/$name.remote.log" || die "fetch failed for '$name'"
}

cmd_list() {
  printf '%-26s %-12s %-10s %s\n' NAME STATE PID STATE_FILE
  local f name
  for f in "$DIR"/*.remote.json; do
    [ -e "$f" ] || continue
    name="$(basename "$f" .remote.json)"
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
  *) sed -n '2,40p' "$0"; exit 2 ;;
esac
