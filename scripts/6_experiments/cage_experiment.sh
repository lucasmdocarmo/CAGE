#!/bin/bash
# Order:     THE master: stages 0 to 14 of one experiment on the Mac (preflight-mac, provision, ship, setup, validate, calibrate, plan, run, monitor, seal, score, collect, pull, analyze, teardown); it calls the existing stage scripts in order and never reimplements their checks
# Objective: Run one experiment profile end to end as numbered, resumable stages with a state file, detached pod jobs (pod_job.sh), a Mac-side pod monitor (monitor_pod.sh) and the experiments/<S>/<date>/ landing (ADR-0143)
# Cloud:     runpod
# =============================================================================
# cage_experiment.sh <S> [--from STAGE] [--to STAGE] [--only STAGE] [--redo] [--yes STAGE]...
#                        [--plan] [--date YYYY-MM-DD] [--profile FILE] [--list-stages]
#   --to STAGE stops after that stage passes (a smoke that bootstraps, validates,
#   calibrates and plans without running the grid); the pod keeps billing and the
#   PORTAL ACTION block says so.
#
# Design: MyDocs/RunPod/EXPERIMENT_PROCESS_DESIGN_2026-10-03.md (rev. 2) with the
# drift sweep of 2026-10-06 folded in. Principles (binding):
#   1. Money gate inherited: stage 1 (provision) and stage 14 (teardown) run only
#      with `--yes <stage>` on THIS invocation; the GO is recorded verbatim with
#      its instant in extras/state.json. `--plan` prints every stage's commands
#      and runs nothing.
#   2. No new mechanics: every step calls an existing script.
#   3. The sealed run tree is never modified here.
#   4. Fail closed, resumable: each stage records rc, instants and evidence;
#      a non-zero rc stops the master, prints the PORTAL ACTION block and the
#      exact resume command. A stage that passed is a no-op unless --redo.
#      Every step carries its EXPECTED exit code; a mismatch fails the stage.
#   5. Process safety: engines stop on the pod through the launchers' own
#      `stop` verbs inside pod jobs; the Mac never signals a process by name.
#      The monitor is stopped by its recorded pid only.
#   6. One clock: every instant this script writes is UTC with a Z suffix.
#
# Test seams (env): CAGE_SCRIPTS_DIR (sibling scripts root, default <repo>/scripts),
#   CAGE_EXP_ROOT (landing root, default <repo>/experiments), CAGE_EXP_DATE
#   (folder date), CAGE_MAC_PYTHON (analysis python), CAGE_POD_READY_TIMEOUT_S,
#   CAGE_POD_READY_POLL_S. runpodctl, ssh and scp come from PATH.
#
# Bash 3.2 (the Mac's /bin/bash): no associative arrays, no array-reading builtins.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=scripts/lib/_common.sh
source "$PROJECT_DIR/scripts/lib/_common.sh"

SCRIPTS="${CAGE_SCRIPTS_DIR:-$PROJECT_DIR/scripts}"
EXP_ROOT="${CAGE_EXP_ROOT:-$PROJECT_DIR/experiments}"
PY="${CAGE_MAC_PYTHON:-$PROJECT_DIR/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
PY3="$(command -v python3 || true)"
[ -n "$PY3" ] || die "python3 is required on the Mac (state file, JSON parsing)"

STAGES="preflight-mac provision ship setup validate calibrate plan run monitor seal score collect pull analyze teardown"
BILLABLE_STAGES="provision teardown"

usage() {
  sed -n '6,8p' "$0" >&2
  printf 'stages (in order): %s\n' "$STAGES" >&2
  exit 2
}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
EXP_ARG=""; FROM=""; TO=""; ONLY=""; REDO=0; PLAN=0; DATE_ARG=""; PROFILE_ARG=""; YES_STAGES=" "
while [ $# -gt 0 ]; do
  case "$1" in
    --from)    [ $# -ge 2 ] || usage; FROM="$2"; shift 2 ;;
    --to)      [ $# -ge 2 ] || usage; TO="$2"; shift 2 ;;
    --only)    [ $# -ge 2 ] || usage; ONLY="$2"; shift 2 ;;
    --redo)    REDO=1; shift ;;
    --yes)     [ $# -ge 2 ] || usage; YES_STAGES="$YES_STAGES$2 "; shift 2 ;;
    --plan)    PLAN=1; shift ;;
    --date)    [ $# -ge 2 ] || usage; DATE_ARG="$2"; shift 2 ;;
    --profile) [ $# -ge 2 ] || usage; PROFILE_ARG="$2"; shift 2 ;;
    --list-stages) printf '%s\n' $STAGES; exit 0 ;;
    -h|--help) usage ;;
    -*) printf 'unknown flag: %s\n' "$1" >&2; usage ;;
    *) [ -z "$EXP_ARG" ] || usage; EXP_ARG="$1"; shift ;;
  esac
done
[ -n "$EXP_ARG" ] || usage

stage_known() { case " $STAGES " in *" $1 "*) return 0 ;; esac; return 1; }
stage_index() { local i=0 s; for s in $STAGES; do [ "$s" = "$1" ] && { echo "$i"; return 0; }; i=$((i + 1)); done; return 1; }
stage_prev() { local prev="" s; for s in $STAGES; do [ "$s" = "$1" ] && { echo "$prev"; return 0; }; prev="$s"; done; return 1; }
is_billable() { case " $BILLABLE_STAGES " in *" $1 "*) return 0 ;; esac; return 1; }
has_yes() { case "$YES_STAGES" in *" $1 "*) return 0 ;; esac; return 1; }
[ -z "$FROM" ] || stage_known "$FROM" || die "--from: unknown stage '$FROM' (stages: $STAGES)"
[ -z "$TO" ] || stage_known "$TO" || die "--to: unknown stage '$TO' (stages: $STAGES)"
[ -z "$ONLY" ] || stage_known "$ONLY" || die "--only: unknown stage '$ONLY' (stages: $STAGES)"
for s in $YES_STAGES; do stage_known "$s" || die "--yes: unknown stage '$s'"; done
# A bare --redo would repeat every passed stage from 0, billable ones included
# (review 2026-10-06, LOW 16): it must name where to start or which one stage.
if [ "$REDO" -eq 1 ] && [ -z "$FROM" ] && [ -z "$ONLY" ] && [ "$PLAN" -eq 0 ]; then
  die "--redo needs --from <stage> or --only <stage> (a bare --redo would repeat every stage, provision included)"
fi

# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------
PROFILE="${PROFILE_ARG:-$SCRIPT_DIR/profiles/$EXP_ARG.env}"
[ -f "$PROFILE" ] || die "no profile for '$EXP_ARG': $PROFILE"
set -a
# shellcheck source=/dev/null
source "$SCRIPT_DIR/profiles/_common.env"
# shellcheck source=/dev/null
source "$PROFILE"
set +a
for v in EXP SESSION CAMPAIGN MODEL MODEL_SLUG GPU_ID ENGINES; do
  [ -n "${!v:-}" ] || die "profile $PROFILE leaves $v empty"
done
[ "$EXP" = "$EXP_ARG" ] || die "profile $PROFILE declares EXP=$EXP, not '$EXP_ARG'"
printf '%s' "$CAMPAIGN" | grep -qE '^[a-z0-9][a-z0-9-]{0,40}$' \
  || die "CAMPAIGN='$CAMPAIGN' violates the campaign slug grammar ^[a-z0-9][a-z0-9-]{0,40}$ (docs/RESULTS_LAYOUT.md)"
POD_REPO="${POD_REPO:-/workspace/CAGE}"
POD_BACKUP_DIR="${POD_BACKUP_DIR:-/workspace/backup}"
POD_TARBALL="${POD_TARBALL:-/root/cage_repo.tar.gz}"
FREEZE_FILE="${FREEZE_FILE:-$PROJECT_DIR/MyDocs/registration/freeze_resolutions.json}"
POD_FREEZE_FILE="MyDocs/registration/freeze_resolutions.json"
SCORE_BOUND_MIN="${SCORE_BOUND_MIN:-180}"
POD_READY_TIMEOUT_S="${CAGE_POD_READY_TIMEOUT_S:-900}"
POD_READY_POLL_S="${CAGE_POD_READY_POLL_S:-15}"
# Per-stage job bounds (seconds); the seatbelt must cover their sum plus the run.
VALIDATE_BOUND_S=1500; CALIBRATE_BOUND_S=3600; PLAN_BOUND_S=900; COLLECT_BOUND_S=1800; PULL_MARGIN_MIN=60

seatbelt_minutes() { # "24h" | "90m" | "2d" -> minutes, or empty when malformed
  case "$1" in
    *h) printf '%s' "$(( ${1%h} * 60 ))" ;;
    *m) printf '%s' "${1%m}" ;;
    *d) printf '%s' "$(( ${1%d} * 1440 ))" ;;
    *) return 1 ;;
  esac
}
seatbelt_need_minutes() { # the stage bounds the pod must survive, in minutes
  local n_srv=0 n_cal=0 e
  for e in $ENGINES; do [ "$e" = "hf" ] || n_srv=$((n_srv + 1)); done
  for e in $CALIBRATE; do n_cal=$((n_cal + 1)); done
  printf '%s' "$(( SETUP_BOUND_MIN + n_srv * VALIDATE_BOUND_S / 60 + n_cal * CALIBRATE_BOUND_S / 60 + PLAN_BOUND_S / 60 + HOURS * 60 + SCORE_BOUND_MIN + COLLECT_BOUND_S / 60 + PULL_MARGIN_MIN ))"
}

DATE="${DATE_ARG:-${CAGE_EXP_DATE:-$(date -u +%Y-%m-%d)}}"
printf '%s' "$DATE" | grep -qE '^[0-9]{4}-[0-9]{2}-[0-9]{2}(_[0-9]{4})?$' || die "--date must be YYYY-MM-DD[_hhmm]: $DATE"
LAND="$EXP_ROOT/$EXP/$DATE"
EXTRAS="$LAND/extras"
STATE="$EXTRAS/state.json"
JOBS="$EXTRAS/jobs"
PODJOB="$SCRIPT_DIR/pod_job.sh"

utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { printf '[%s] %s\n' "$EXP" "$*"; }
hr()  { printf '%s\n' "----------------------------------------------------------------"; }

# ---------------------------------------------------------------------------
# state file (extras/state.json): one JSON object, every write atomic
# ---------------------------------------------------------------------------
state() {
  # state <op> [args]: init EXP DATE | begin STAGE EXPECTED | end STAGE RC |
  #   status STAGE | put KEY VALUE | put-json KEY JSON | get KEY | go STAGE WORDS |
  #   evidence STAGE PATH | note STAGE TEXT | running | summary
  STATE_FILE="$STATE" "$PY3" - "$@" <<'PY'
import json, os, sys, datetime
path = os.environ["STATE_FILE"]
op, args = sys.argv[1], sys.argv[2:]
st = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def walk(key, create=False):
    cur = st
    parts = key.split(".")
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            if not create:
                return None, None
            cur[p] = {}
        cur = cur[p]
    return cur, parts[-1]

changed = True
if op == "init":
    st.setdefault("schema", "cage-experiment-state-v1")
    st["experiment"], st["date_utc"] = args[0], args[1]
    st.setdefault("stages", {}); st.setdefault("go", []); st.setdefault("jobs", {})
    st.setdefault("cost", {}); st.setdefault("pod", {})
    st.setdefault("created_utc", now)
elif op == "begin":
    st.setdefault("stages", {})[args[0]] = {
        "status": "running", "rc": None, "expected_rc": int(args[1]),
        "started_utc": now, "ended_utc": None, "evidence": [], "notes": []}
elif op == "end":
    s = st.setdefault("stages", {}).setdefault(args[0], {})
    rc = int(args[1])
    s["rc"] = rc; s["ended_utc"] = now
    s["status"] = "passed" if rc == s.get("expected_rc", 0) else "failed"
elif op == "status":
    changed = False
    print(st.get("stages", {}).get(args[0], {}).get("status", "none"))
elif op == "put":
    parent, leaf = walk(args[0], create=True); parent[leaf] = args[1]
elif op == "put-json":
    parent, leaf = walk(args[0], create=True); parent[leaf] = json.loads(args[1])
elif op == "get":
    changed = False
    parent, leaf = walk(args[0])
    v = None if parent is None else parent.get(leaf)
    if v is not None:
        print(v if isinstance(v, str) else json.dumps(v))
elif op == "go":
    st.setdefault("go", []).append({"stage": args[0], "instant_utc": now, "words": args[1]})
elif op == "evidence":
    st.setdefault("stages", {}).setdefault(args[0], {}).setdefault("evidence", []).append(args[1])
elif op == "note":
    st.setdefault("stages", {}).setdefault(args[0], {}).setdefault("notes", []).append(args[1])
elif op == "running":
    changed = False
    for name, s in st.get("stages", {}).items():
        if s.get("status") == "running":
            print(name)
elif op == "summary":
    changed = False
    for name, s in st.get("stages", {}).items():
        print(f"{name:14s} {s.get('status','none'):8s} rc={s.get('rc')} started={s.get('started_utc')} ended={s.get('ended_utc')}")
else:
    sys.exit(f"state: unknown op {op}")
if changed:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(st, fh, indent=2, sort_keys=True); fh.write("\n")
    os.replace(tmp, path)
PY
}

# ---------------------------------------------------------------------------
# steps: every command runs through here, with its EXPECTED exit code
# ---------------------------------------------------------------------------
CUR_STAGE=""
STEP_N=0
STEP_LOG=""

rc_expected() { # rc_expected <rc> <expected: "0" or "0|1">
  case "|$2|" in *"|$1|"*) return 0 ;; esac; return 1
}

run_step() {
  # run_step <expected_rc> <label> -- <cmd...>; stdout+stderr tee'd to a step log.
  # expected_rc is one code or several joined by "|" (teardown_pod.sh: 0 or 1).
  local expected="$1" label="$2"; shift 2; [ "${1:-}" = "--" ] && shift
  STEP_N=$((STEP_N + 1))
  if [ "$PLAN" -eq 1 ]; then
    printf '  [plan] %-28s expect rc=%s :' "$label" "$expected"; printf ' %q' "$@"; printf '\n'
    return 0
  fi
  STEP_LOG="$EXTRAS/steps/${CUR_STAGE}_$(printf '%02d' "$STEP_N")_$(printf '%s' "$label" | tr -c 'A-Za-z0-9' '_').log"
  mkdir -p "$EXTRAS/steps"
  printf '  [step] %s\n' "$label"
  { printf '# %s\n# stage=%s label=%s expected_rc=%s\n# cmd:' "$(utc)" "$CUR_STAGE" "$label" "$expected"; printf ' %q' "$@"; printf '\n'; } > "$STEP_LOG"
  # pipefail off for this pipeline so tee's 0 is the list status and
  # PIPESTATUS[0] (the command's own rc) is read before anything else runs.
  local rc=0
  set +o pipefail
  "$@" 2>&1 | tee -a "$STEP_LOG"
  rc="${PIPESTATUS[0]}"
  set -o pipefail
  printf '# rc=%s at %s\n' "$rc" "$(utc)" >> "$STEP_LOG"
  state evidence "$CUR_STAGE" "$STEP_LOG"
  if ! rc_expected "$rc" "$expected"; then
    printf '  [FAIL] %s: rc=%s, expected %s (log: %s)\n' "$label" "$rc" "$expected" "$STEP_LOG"
    return 1
  fi
  return 0
}

step_out() { cat "$STEP_LOG"; }            # the last step's captured output
step_has() { grep -qF -- "$1" "$STEP_LOG"; } # literal substring in the last step's output (never a regex)

plan_only() { [ "$PLAN" -eq 1 ]; }

# the ssh front door to the pod (pod_job.sh reads the same env)
pod_env() {
  POD_ID="$(state get pod.id 2>/dev/null || true)"; POD_ID="${POD_ID:-<pod_id>}"
  # The state file is the source; CAGE_POD_SSH_OVERRIDE / CAGE_POD_SSH_PORT_OVERRIDE
  # are the operator's way in when `runpodctl ssh info` could not be parsed.
  # CAGE_POD_SSH itself is what this function EXPORTS for pod_job.sh and is
  # never read back (a placeholder exported early would otherwise win).
  POD_SSH="$(state get pod.ssh_host 2>/dev/null || true)"; POD_SSH="${CAGE_POD_SSH_OVERRIDE:-${POD_SSH:-<user@host>}}"
  POD_PORT="$(state get pod.ssh_port 2>/dev/null || true)"; POD_PORT="${CAGE_POD_SSH_PORT_OVERRIDE:-${POD_PORT:-22}}"
  VOL_ID="$(state get pod.volume_id 2>/dev/null || true)"; VOL_ID="${VOL_ID:-<volume_id>}"
  RUN_ID="$(state get run_id 2>/dev/null || true)"; RUN_ID="${RUN_ID:-<run_id>}"
  PRICE="$(state get pod.price_per_hour_usd 2>/dev/null || true)"
  POD_RUN_ROOT="$POD_REPO/results/$CAMPAIGN/$SESSION/$RUN_ID"
  RUN_LOCAL="$LAND/run/$CAMPAIGN/$SESSION/$RUN_ID"
  BACKUP_RUN_REL="results/$CAMPAIGN/$SESSION/$RUN_ID"
  export CAGE_POD_SSH="$POD_SSH" CAGE_POD_SSH_PORT="$POD_PORT" CAGE_JOBS_DIR="$JOBS"
  [ -z "${SSH_KEY:-}" ] || export CAGE_SSH_KEY="$SSH_KEY"
  SSH_OPTS="-p $POD_PORT -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=25"
  [ -z "${SSH_KEY:-}" ] || SSH_OPTS="$SSH_OPTS -i $SSH_KEY"
}
pssh() { # pssh '<remote command>'
  # shellcheck disable=SC2086
  ssh $SSH_OPTS "$POD_SSH" "$1"
}
pscp_to()   { # pscp_to <local> <remote path>
  # shellcheck disable=SC2086
  scp -P "$POD_PORT" ${SSH_KEY:+-i "$SSH_KEY"} -o BatchMode=yes -o StrictHostKeyChecking=no "$1" "$POD_SSH:$2"; }
pscp_from() { # pscp_from <remote path> <local>; a remote DIRECTORY is given as "<dir>/."
  # so its contents land in <local> instead of nesting as <local>/<dir> (scp -r
  # into an existing directory nests; review 2026-10-06, LOW 13) [A on OpenSSH].
  # shellcheck disable=SC2086
  scp -r -P "$POD_PORT" ${SSH_KEY:+-i "$SSH_KEY"} -o BatchMode=yes -o StrictHostKeyChecking=no "$POD_SSH:$1" "$2"; }

job_run() {
  # job_run <name> <deadline_s> <remote command> [logdir]: submit, wait, fetch
  # the log into logdir; the job's DONE(0) is the step's expected outcome.
  # A failed attempt's log is fetched too, under its own name, and the next
  # submit starts a fresh remote log (review 2026-10-06, MEDIUM 8), so a
  # marker from an earlier attempt can never satisfy this one's check.
  local name="$1" deadline="$2" cmd="$3" logdir="${4:-$LAND/logs/runner}"
  run_step 0 "job $name: submit" -- bash "$PODJOB" submit "$name" "$cmd" "$deadline" || return 1
  if ! run_step 0 "job $name: wait" -- bash "$PODJOB" wait "$name" "$deadline"; then
    bash "$PODJOB" tail "$name" 60 || true
    mkdir -p "$logdir/failed"
    bash "$PODJOB" fetch "$name" "$logdir/failed" >/dev/null 2>&1 && mv "$logdir/failed/$name.log" "$logdir/failed/${name}_$(date -u +%Y%m%dT%H%M%SZ).log" || true
    return 1
  fi
  run_step 0 "job $name: fetch log" -- bash "$PODJOB" fetch "$name" "$logdir" || return 1
  JOB_LOG="$logdir/$name.log"
  return 0
}
JOB_LOG=""

json_empty() {
  # 0 when the file holds an empty JSON listing ([] / {} / null / a dict whose lists are all empty)
  "$PY3" - "$1" <<'PY'
import json, sys
try:
    doc = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    sys.exit(2)
def empty(d):
    if d in (None, [], {}): return True
    if isinstance(d, dict): return all(empty(v) for v in d.values() if isinstance(v, (list, dict, type(None))))
    return False
sys.exit(0 if empty(doc) else 1)
PY
}

engine_launcher() { case "$1" in vllm) echo manage_vllm_server.sh ;; sglang) echo manage_sglang_server.sh ;; lmdeploy) echo manage_lmdeploy_server.sh ;; *) return 1 ;; esac; }
engine_api()      { case "$1" in vllm) echo http://localhost:8000 ;; sglang) echo http://localhost:30000 ;; lmdeploy) echo http://localhost:23333 ;; *) return 1 ;; esac; }
server_engines()  { local e; for e in $ENGINES; do [ "$e" = "hf" ] || echo "$e"; done; }

# env pins run_campaign.py refuses on presence or on a mismatch (drift sweep 2026-10-06, 8e)
RUN_ENV_UNSET="-u CAGE_SLO_FLOORS_JSON -u CAGE_BUDGET_PLAN_JSON -u CAGE_TELEMETRY_ENDPOINTS -u CAGE_ALLOW_STALE_INDEX -u CAGE_SGLANG_API_BASE -u CAGE_LMDEPLOY_API_BASE -u VLLM_PORT -u SGLANG_PORT -u CAGE_PD_PROXY_PORT -u CAGE_PD_PREFILL_PORT -u CAGE_PD_DECODE_PORT"

portal_block() {
  pod_env
  local verdict="NOT YET (results not pulled and verified; stage 12 has not passed)"
  [ "$(state status pull 2>/dev/null || echo none)" = "passed" ] && verdict="SAFE NOW (pull verified at stage 12: $RUN_LOCAL)"
  local clock="unknown" created hours
  created="$(state get pod.created_utc 2>/dev/null || true)"
  if [ -n "$PRICE" ] && [ -n "$created" ]; then
    hours="$("$PY3" -c 'import sys,datetime; c=datetime.datetime.strptime(sys.argv[1],"%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc); print(f"{(datetime.datetime.now(datetime.timezone.utc)-c).total_seconds()/3600:.2f}")' "$created" 2>/dev/null || echo "?")"
    clock="$("$PY3" -c 'import sys; print(f"${float(sys.argv[1])*float(sys.argv[2]):.2f} ({sys.argv[2]} h at ${sys.argv[1]}/h)")' "$PRICE" "$hours" 2>/dev/null || echo "?")"
  fi
  hr
  printf 'PORTAL ACTION\n'
  printf '  resource: pod              id: %s   cost clock: %s\n' "$POD_ID" "$clock"
  printf '  resource: network volume   id: %s\n' "$VOL_ID"
  printf '  verdict:  %s\n' "$verdict"
  printf '  watchdog: bash %s/runpod/pod_watchdog.sh status %s\n' "$SCRIPTS" "$POD_ID"
  # stage 8 (monitor) only reads alerts.log, which is append-only: its resume is the next stage
  local from="${CUR_STAGE:-preflight-mac}"; [ "$from" = "monitor" ] && from="seal"
  printf '  resume:   bash %s %s --date %s --from %s%s\n' "$0" "$EXP" "$DATE" "$from" "$( is_billable "$from" && printf ' --yes %s' "$from" || true)"
  if [ "$from" = "provision" ] && [ -n "$(state get pod.id 2>/dev/null || true)" ]; then
    printf '  note:     pod %s is RECORDED; the resume continues with it and creates NO second pod\n' "$POD_ID"
  fi
  hr
}

MONITOR_PID=""
stop_monitor() {
  local pid; pid="$(state get monitor.pid 2>/dev/null || true)"
  [ -n "$pid" ] || return 0
  # only our own recorded pid, proven ours by its command line (J8 guard)
  if ps -ww -p "$pid" -o command= 2>/dev/null | grep -q "monitor_pod.sh"; then
    kill "$pid" 2>/dev/null || true
    say "monitor stopped (pid $pid)"
  fi
  state put monitor.pid ""
}

# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
stage_preflight_mac() {
  local tarball="$EXTRAS/ops/repo.tar.gz"
  run_step 0 "runpodctl version" -- runpodctl version || return 1
  run_step 0 "account balance (before)" -- runpodctl user || return 1
  plan_only || { cp "$STEP_LOG" "$EXTRAS/ops/user_before.txt"; state put cost.balance_before_raw "$(grep -v '^#' "$STEP_LOG" | tr -d '\n' | cut -c1-400)"; }
  run_step 0 "clean room: pod list --all" -- runpodctl pod list --all || return 1
  plan_only || { grep -v '^#' "$STEP_LOG" > "$EXTRAS/ops/pods_before.json"; json_empty "$EXTRAS/ops/pods_before.json" || { printf '  [FAIL] clean room: pods exist before this experiment (see %s)\n' "$EXTRAS/ops/pods_before.json"; return 1; }; }
  run_step 0 "clean room: network-volume list" -- runpodctl network-volume list || return 1
  plan_only || { grep -v '^#' "$STEP_LOG" > "$EXTRAS/ops/volumes_before.json"; json_empty "$EXTRAS/ops/volumes_before.json" || { printf '  [FAIL] clean room: network volumes exist before this experiment (see %s)\n' "$EXTRAS/ops/volumes_before.json"; return 1; }; }
  run_step 0 "Mac test suite" -- bash "$SCRIPTS/checks/run_tests.sh" -q -p no:cacheprovider || return 1
  plan_only || cp "$STEP_LOG" "$LAND/logs/setup/suite_mac.log"
  run_step 0 "package the repo" -- bash "$SCRIPTS/ops/package_repo.sh" "$tarball" || return 1
  if ! plan_only; then
    step_has "PACKAGED" || { printf '  [FAIL] package_repo.sh printed no PACKAGED line\n'; return 1; }
    [ -f "$tarball" ] || { printf '  [FAIL] tarball missing: %s\n' "$tarball"; return 1; }
    if tar tzf "$tarball" | grep -q '/\._\|^\._'; then printf '  [FAIL] tarball carries AppleDouble ._* members (S0F-7)\n'; return 1; fi
    local build; build="$(tar xzf "$tarball" -O BUILD_INFO 2>/dev/null || true)"
    printf '%s\n' "$build" > "$EXTRAS/ops/BUILD_INFO"
    printf '%s' "$build" | grep -q '^dirty=0$' || { printf '  [FAIL] BUILD_INFO is not dirty=0 (commit first):\n%s\n' "$build"; return 1; }
    state put build.sha "$(printf '%s' "$build" | sed -n 's/^sha=//p')"
    state put build.tarball "$tarball"
  fi
  run_step 0 "freeze file present" -- test -f "$FREEZE_FILE" || return 1
  # Every manifest the plan and the calibration will name must exist in the
  # repo (it ships in the tarball); a missing one refuses stage 6 after
  # hours of billing otherwise (found 2026-10-06: musique has no manifest).
  local item mpath missing=""
  for item in $QUERY_MANIFESTS; do mpath="${item#*=}"; [ -f "$PROJECT_DIR/$mpath" ] || missing="$missing $mpath"; done
  [ -f "$PROJECT_DIR/$CALIBRATION_MANIFEST" ] || missing="$missing $CALIBRATION_MANIFEST"
  run_step 0 "query and calibration manifests exist" -- test -z "$missing" || { printf '  [FAIL] manifest(s) named by the profile do not exist in the repo:%s (build them with scripts/1_setup/build_query_manifest.py or drop them from QUERY_MANIFESTS)\n' "$missing"; return 1; }
  run_step 0 "session '$SESSION' registered in the driver" -- "$PY" - "$SCRIPTS/3_run/run_campaign.py" "$SESSION" <<'PY' || return 1
import importlib.util, sys
spec = importlib.util.spec_from_file_location("cage_run_campaign_probe", sys.argv[1])
m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m
spec.loader.exec_module(m)
try:
    m.get_session_grid(sys.argv[2])
except Exception as exc:
    print(f"REFUSED: session {sys.argv[2]!r}: {exc}"); sys.exit(1)
print(f"session {sys.argv[2]!r} is registered")
PY
  run_step 0 "stock read (gpu list --include-unavailable)" -- runpodctl gpu list --include-unavailable || return 1
  if ! plan_only; then
    grep -v '^#' "$STEP_LOG" > "$EXTRAS/ops/gpu_list.json"
    local dcs; dcs="$("$PY3" - "$EXTRAS/ops/gpu_list.json" "$GPU_ID" <<'PY'
import json, sys
gpus = json.load(open(sys.argv[1], encoding="utf-8"))
want = sys.argv[2]
# Live shape (read 2026-10-06): dataCenterAvailability[].{dataCenterId, stockStatus}
# with stockStatus in High | Medium | Low | none; "none" is NO stock.
IN_STOCK = {"high", "medium", "low"}
for g in gpus if isinstance(gpus, list) else gpus.get("gpus", []):
    if g.get("gpuId") == want or g.get("id") == want:
        avail = []
        for d in g.get("dataCenterAvailability", []) or []:
            dc = d.get("dataCenterId") or d.get("id")
            if dc and (d.get("available") is True or str(d.get("stockStatus", "")).lower() in IN_STOCK):
                avail.append(str(dc))
        print(" ".join(avail)); sys.exit(0)
sys.exit(3)
PY
)" || { printf '  [FAIL] GPU_ID %s not found in runpodctl gpu list (check the profile spelling)\n' "$GPU_ID"; return 1; }
    state put siting.available_dcs "$dcs"
    say "datacenters with stock for $GPU_ID: ${dcs:-none listed}"
  fi
  # The seatbelt deletes the pod at its deadline whatever stage is running
  # (review 2026-10-06, HIGH 4): it must cover every pod-side bound, not the run alone.
  local seat_min need_min
  seat_min="$(seatbelt_minutes "$SEATBELT")" || { printf '  [FAIL] SEATBELT=%s is not <n>h, <n>m or <n>d\n' "$SEATBELT"; return 1; }
  need_min="$(seatbelt_need_minutes)"
  say "seatbelt $SEATBELT = $seat_min min; the stage bounds need $need_min min (setup $SETUP_BOUND_MIN, validate, calibrate, plan, run $((HOURS * 60)), score $SCORE_BOUND_MIN, collect, pull margin $PULL_MARGIN_MIN)"
  run_step 0 "seatbelt covers the stage bounds" -- test "$seat_min" -ge "$need_min" || { printf '  [FAIL] SEATBELT=%s (%s min) is shorter than the %s min the stages need; raise SEATBELT in the profile\n' "$SEATBELT" "$seat_min" "$need_min"; return 1; }
  run_step 0 "provision plan print" -- bash "$SCRIPTS/runpod/provision_pod.sh" --gpu-id "$GPU_ID" --gpu-count "$GPU_COUNT" --disk-gb "$DISK_GB" --volume-gb "$VOLUME_GB" --terminate-after "$SEATBELT" --hours "$HOURS" --purpose "$EXP $DATE" || return 1
  plan_only || step_has "PLAN ONLY" || { printf '  [FAIL] provision plan print lacks the PLAN ONLY line\n'; return 1; }
  if ! plan_only; then
    local rid; rid="$(mint_campaign_run_id "$SESSION" "$MODEL_SLUG")"
    printf '%s' "$rid" | grep -qE '^[a-z0-9][a-z0-9-]{2,40}$' || { printf '  [FAIL] minted run id %s violates the run_id grammar\n' "$rid"; return 1; }
    state put run_id "$rid"
    say "run id minted: $rid (campaign root on the pod: $POD_REPO/results/$CAMPAIGN/$SESSION/$rid)"
  fi
  return 0
}

stage_provision() {
  local dcs dc created=0 vol="" out
  dcs="$(state get siting.available_dcs 2>/dev/null || true)"
  local prefs="" p
  for p in $DC_PREFS; do
    if [ -z "$dcs" ] || [ "$PLAN" -eq 1 ]; then prefs="$prefs $p"; else case " $dcs " in *" $p "*) prefs="$prefs $p" ;; esac; fi
  done
  [ -n "$prefs" ] || { printf '  [FAIL] no datacenter in DC_PREFS (%s) has stock for %s (stock: %s)\n' "$DC_PREFS" "$GPU_ID" "${dcs:-none}"; return 1; }
  # A resume after a partial provision (the pod was CREATED, a later step
  # failed) continues with the recorded pod and creates NOTHING (review
  # 2026-10-06, CRITICAL 1). CAGE_POD_ID_OVERRIDE records a pod found in the console.
  local existing; existing="$(state get pod.id 2>/dev/null || true)"
  if ! plan_only && [ -n "${CAGE_POD_ID_OVERRIDE:-}" ]; then
    state put pod.id "$CAGE_POD_ID_OVERRIDE"; existing="$CAGE_POD_ID_OVERRIDE"
    [ -n "$(state get pod.created_utc 2>/dev/null || true)" ] || state put pod.created_utc "$(utc)"
  fi
  if ! plan_only && [ "$existing" = "unknown-see-portal" ]; then
    printf '  [FAIL] a pod was created earlier but its id was never parsed: find it in the console, then resume with CAGE_POD_ID_OVERRIDE=<id> (or tear it down by hand); nothing is created here\n'
    return 1
  fi
  if ! plan_only && [ -n "$existing" ]; then
    say "pod $existing is already recorded (resume after a partial provision): no volume, no pod is created"
    created=1
  fi
  plan_only || [ "$created" -eq 1 ] || state go provision "--yes provision on the command line ($(utc))"
  [ "$created" -eq 1 ] || for dc in $prefs; do
    run_step 0 "network volume create in $dc" -- runpodctl network-volume create --name "cage-$EXP-$DATE" --size "$VOLUME_GB" --data-center-id "$dc" || continue
    if plan_only; then vol="<volume_id>"; else
      vol="$(grep -v '^#' "$STEP_LOG" | sed -nE 's/.*"id"[[:space:]]*:[[:space:]]*"([A-Za-z0-9-]+)".*/\1/p' | head -1)"
      [ -n "$vol" ] || { printf '  [FAIL] volume created in %s but its id was not parsed; find it with runpodctl network-volume list\n' "$dc"; return 1; }
      state put pod.volume_id "$vol"; state put pod.dc "$dc"
    fi
    run_step 0 "provision pod in $dc (--yes)" -- bash "$SCRIPTS/runpod/provision_pod.sh" --gpu-id "$GPU_ID" --gpu-count "$GPU_COUNT" --disk-gb "$DISK_GB" --data-center-ids "$dc" --network-volume-id "$vol" --terminate-after "$SEATBELT" --hours "$HOURS" --purpose "$EXP $DATE" --yes
    local rc=$?
    if plan_only; then created=1; break; fi
    out="$(step_out)"
    local pod; pod="$(printf '%s\n' "$out" | sed -nE 's/.*pod CREATED: ([A-Za-z0-9-]+).*/\1/p' | head -1)"
    if [ -n "$pod" ]; then
      state put pod.id "$pod"; state put pod.created_utc "$(utc)"
      if printf '%s' "$out" | grep -q "SEATBELT NOT ARMED"; then
        printf '  [FAIL] pod %s is BILLING with NO seatbelt (watchdog arm failed); arm it by hand or tear down\n' "$pod"; return 1
      fi
      created=1; break
    fi
    if printf '%s' "$out" | grep -qi "CREATED" && printf '%s' "$out" | grep -qi "BILLING"; then
      state put pod.id "unknown-see-portal"
      printf '  [FAIL] a pod was CREATED and is BILLING but its id was not parsed (rc=%s); find it in the console NOW\n' "$rc"; return 1
    fi
    say "no pod in $dc (rc=$rc); deleting its volume and trying the next datacenter"
    run_step 0 "network volume delete $vol (no pod)" -- runpodctl network-volume delete "$vol" || true
    state put pod.volume_id ""
  done
  [ "$created" -eq 1 ] || { printf '  [FAIL] no datacenter accepted the pod\n'; return 1; }
  pod_env
  if ! plan_only && [ -z "$PRICE" ]; then
    local price; price="$(tail -n 50 "${CAGE_POD_LEDGER:-$PROJECT_DIR/results/ops/pod_ledger.jsonl}" 2>/dev/null | grep "\"$POD_ID\"" | tail -1 | sed -nE 's/.*"price_per_hour_usd": ?([0-9.]+).*/\1/p' || true)"
    [ -z "$price" ] || state put pod.price_per_hour_usd "$price"
  fi
  run_step 0 "pod readiness (runtimeStatus running)" -- bash -c '
    deadline=$(( $(date +%s) + '"$POD_READY_TIMEOUT_S"' ))
    while :; do
      out="$(runpodctl pod get "'"$POD_ID"'" 2>&1)" || true
      status="$(printf "%s" "$out" | sed -nE "s/.*\"runtimeStatus\"[[:space:]]*:[[:space:]]*\"([a-z]+)\".*/\1/p" | head -1)"
      echo "runtimeStatus=${status:-unknown}"
      [ "$status" = "running" ] && exit 0
      [ "$(date +%s)" -ge "$deadline" ] && { echo "pod not running after '"$POD_READY_TIMEOUT_S"'s"; exit 1; }
      sleep '"$POD_READY_POLL_S"'
    done' || return 1
  run_step 0 "ssh info" -- runpodctl ssh info "$POD_ID" || return 1
  if ! plan_only; then
    if [ -n "${CAGE_POD_SSH_OVERRIDE:-}" ]; then
      state put pod.ssh_host "$CAGE_POD_SSH_OVERRIDE"; state put pod.ssh_port "${CAGE_POD_SSH_PORT_OVERRIDE:-22}"
    else
      local info; info="$(grep -v '^#' "$STEP_LOG" | "$PY3" -c '
import json, sys
raw = sys.stdin.read()
try:
    doc = json.loads(raw)
except Exception:
    sys.exit(3)
cands = doc if isinstance(doc, list) else [doc]
for d in cands:
    if not isinstance(d, dict): continue
    host = d.get("ip") or d.get("publicIp") or d.get("host") or d.get("sshHost")
    port = d.get("port") or d.get("sshPort") or d.get("publicPort") or 22
    user = d.get("user") or d.get("username") or "root"
    if host:
        print(f"{user}@{host} {port}"); sys.exit(0)
sys.exit(3)' 2>/dev/null || true)"
      [ -n "$info" ] || { printf '  [FAIL] could not parse host/port from runpodctl ssh info (see %s); export CAGE_POD_SSH_OVERRIDE=user@host CAGE_POD_SSH_PORT_OVERRIDE=port and resume\n' "$STEP_LOG"; return 1; }
      state put pod.ssh_host "${info%% *}"; state put pod.ssh_port "${info##* }"
    fi
    pod_env
  fi
  run_step 0 "ssh probe" -- bash -c "$(printf 'ssh %s %q echo CAGE_SSH_OK' "$SSH_OPTS" "$POD_SSH")" || return 1
  plan_only || step_has "CAGE_SSH_OK" || return 1
  run_step 0 "watchdog ALIVE" -- bash "$SCRIPTS/runpod/pod_watchdog.sh" status "$POD_ID" || return 1
  plan_only || step_has "watchdog=ALIVE" || { printf '  [FAIL] the seatbelt is not ALIVE for %s\n' "$POD_ID"; return 1; }
  return 0
}

stage_ship() {
  pod_env
  local tarball; tarball="$(state get build.tarball 2>/dev/null || true)"; tarball="${tarball:-$EXTRAS/ops/repo.tar.gz}"
  run_step 0 "scp tarball" -- pscp_to "$tarball" "$POD_TARBALL" || return 1
  run_step 0 "unpack on the pod" -- pssh "mkdir -p $POD_REPO && ln -sfn $POD_REPO ~/CAGE && tar xzf $POD_TARBALL --no-same-owner -C $POD_REPO && cat $POD_REPO/BUILD_INFO && mkdir -p $POD_BACKUP_DIR $POD_REPO/MyDocs/registration $POD_REPO/results/calibration" || return 1
  if ! plan_only; then
    local sha; sha="$(state get build.sha)"
    step_has "sha=$sha" || { printf '  [FAIL] BUILD_INFO on the pod does not carry the Mac HEAD %s\n' "$sha"; return 1; }
    step_has "dirty=0" || { printf '  [FAIL] BUILD_INFO on the pod is not dirty=0\n'; return 1; }
  fi
  run_step 0 "scp freeze file" -- pscp_to "$FREEZE_FILE" "$POD_REPO/$POD_FREEZE_FILE" || return 1
  return 0
}

stage_setup() {
  pod_env
  local bound=$(( SETUP_BOUND_MIN * 60 ))
  job_run setup "$bound" "cd $POD_REPO && CHARTER_DATASETS='$CHARTER_DATASETS' PREFETCH_MODELS='$PREFETCH_MODELS' bash scripts/runpod/setup_runpod.sh" "$LAND/logs/setup" || return 1
  plan_only && return 0
  local log="$JOB_LOG" m bad=0
  grep -q "RunPod bootstrap complete" "$log" || { printf '  [FAIL] setup: no "RunPod bootstrap complete" line\n'; bad=1; }
  # WARNING, FATAL and ERROR always; NOTE only for the telemetry import note
  # (setup_runpod.sh:424): the closing "harness trees carry no ledger.json"
  # NOTE (line 450) prints on every bootstrap (review 2026-10-06, HIGH 2).
  if grep -nE '\[cage\] +(WARNING|FATAL|ERROR):|\[cage\] +NOTE:.*not importable' "$log"; then printf '  [FAIL] setup: warn-only steps left the lines above (exit 0 proves nothing)\n'; bad=1; fi
  grep -q "all charter datasets staged" "$log" || { printf '  [FAIL] setup: datasets not fully staged\n'; bad=1; }
  for m in $PREFETCH_MODELS; do grep -q "$m: cached" "$log" || { printf '  [FAIL] setup: %s not prefetched\n' "$m"; bad=1; }; done
  grep -q "pynvml OK" "$log" || { printf '  [FAIL] setup: pynvml not OK\n'; bad=1; }
  grep -q "cage_stats.api import OK" "$log" || { printf '  [FAIL] setup: cage_stats.api not importable\n'; bad=1; }
  [ "$bad" -eq 0 ] || return 1
  run_step 0 "pod shape evidence" -- pssh "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader && python3 --version && $POD_REPO/$POD_PYTHON -c 'import torch, vllm; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda, \"vllm\", vllm.__version__)' && df -h $POD_REPO | tail -1" || return 1
  cp "$STEP_LOG" "$LAND/logs/system/pod_shape.txt"
  return 0
}

stage_validate() {
  pod_env
  [ "${NEGATIVE_CONTROLS:-0}" = "1" ] && { printf '  [FAIL] NEGATIVE_CONTROLS=1: the S0 refusal controls are not built into this stage yet (profile default 6, 2026-10-06); set 0 or build them\n'; return 1; }
  [ "${PD_PROBES:-0}" = "1" ] && { printf '  [FAIL] PD_PROBES=1: the S0 pd probes are not built into this stage yet; set 0 or build them\n'; return 1; }
  [ "$TOPOLOGY" = "single" ] || { printf '  [FAIL] TOPOLOGY=%s: only single-instance validation is built; tp/pd launch flags are a separate Spec\n' "$TOPOLOGY"; return 1; }
  if [ "${SUITE_ON_POD:-0}" = "1" ]; then
    job_run suite_pod 1800 "cd $POD_REPO && env $RUN_ENV_UNSET $POD_PYTHON -m pytest -q -p no:cacheprovider" "$LAND/logs/setup" || return 1
  fi
  local e launcher api
  for e in $(server_engines); do
    launcher="$(engine_launcher "$e")" || { printf '  [FAIL] unknown engine %s\n' "$e"; return 1; }
    api="$(engine_api "$e")"
    # after `stop` the GPU process takes a few seconds to leave the compute-apps list: wait up to 60 s before reading it
    job_run "validate_$e" 1500 "cd $POD_REPO && bash scripts/2_serving/$launcher start '$MODEL' && curl -sf $api/v1/models >/dev/null && echo VALIDATE_API_OK; CAGE_PREFLIGHT_BACKENDS=$e bash scripts/checks/preflight_check.sh '$MODEL' $api; rc=\$?; bash scripts/2_serving/$launcher stop; n=1; for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do n=\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true); [ \"\$n\" -eq 0 ] && break; sleep 2; done; echo COMPUTE_APPS=\$n; exit \$rc" "$LAND/logs/setup" || return 1
    plan_only && continue
    grep -q "VALIDATE_API_OK" "$JOB_LOG" || { printf '  [FAIL] %s: /v1/models never answered\n' "$e"; return 1; }
    grep -q "PREFLIGHT PASS" "$JOB_LOG" || { printf '  [FAIL] %s: preflight did not PASS\n' "$e"; return 1; }
    grep -q "COMPUTE_APPS=0" "$JOB_LOG" || { printf '  [FAIL] %s: compute apps remain after stop\n' "$e"; return 1; }
  done
  return 0
}

stage_calibrate() {
  pod_env
  local e launcher api out stamp
  for e in $CALIBRATE; do
    launcher="$(engine_launcher "$e")" || { printf '  [FAIL] unknown engine %s\n' "$e"; return 1; }
    api="$(engine_api "$e")"
    out="results/calibration/${EXP}_${e}.json"
    job_run "calibrate_$e" 3600 "cd $POD_REPO && bash scripts/2_serving/$launcher start '$MODEL' && $POD_PYTHON scripts/3_run/calibrate_cell.py --backend $e --model '$MODEL' --api-base $api --manifest $CALIBRATION_MANIFEST --output $out --budget-fraction $CALIBRATION_BUDGET_FRACTION; rc=\$?; bash scripts/2_serving/$launcher stop; exit \$rc" "$LAND/logs/setup" || return 1
    run_step 0 "fetch calibration $e" -- pscp_from "$POD_REPO/$out" "$EXTRAS/calibration/${EXP}_${e}.json" || return 1
    plan_only && continue
    run_step 0 "calibration $e: cal-v2 and floor-derived start" -- "$PY3" - "$EXTRAS/calibration/${EXP}_${e}.json" <<'PY' || return 1
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
pv = str(d.get("procedure_version", ""))
src = d.get("start_qps_source")
print(f"procedure_version={pv!r} start_qps_source={src!r} lambda_star={d.get('lambda_star')}")
sys.exit(0 if pv.startswith("cal-v2") and src == "floor-service-rate" else 1)
PY
  done
  stamp="$(date -u +%H%M%S)"
  plan_only || mkdir -p "$EXTRAS/calibration"
  run_step 0 "floor table (P6 prediction)" -- "$PY" "$SCRIPTS/4_analysis/build_floor_table.py" --model "$MODEL_SLUG" --engine "$FLOOR_ENGINE" --concurrency-target "$FLOOR_CONCURRENCY" --avg-seq-tokens "$FLOOR_AVG_SEQ_TOKENS" --grid "$FLOOR_GRID" --service-time-s "$FLOOR_SERVICE_TIME_S" --out "$EXTRAS/calibration/floor_table_$stamp.json" || return 1
  plan_only || state put calibration.floor_table "$EXTRAS/calibration/floor_table_$stamp.json"
  local r
  for e in $CALIBRATE; do for r in $BUDGET_RATIOS; do
    run_step 0 "byte plan $e r=$r" -- bash -c "cd '$PROJECT_DIR' && '$PY' -m src.orchestration.cache_budget --model '$MODEL_SLUG' --engine '$e' --r '$r' --concurrency '$FLOOR_CONCURRENCY' --avg-seq-tokens '$FLOOR_AVG_SEQ_TOKENS' > '$EXTRAS/calibration/budget_${e}_r${r}.json'" || return 1
  done; done
  run_step 0 "scp floor table to the pod" -- pscp_to "$EXTRAS/calibration/floor_table_$stamp.json" "$POD_REPO/results/calibration/floor_table.json" || return 1
  return 0
}

stage_plan() {
  pod_env
  local qm cal e item
  qm=""; for item in $QUERY_MANIFESTS; do qm="$qm --query-manifest $item"; done
  cal=""; for e in $CALIBRATE; do cal="$cal --calibration $e=results/calibration/${EXP}_${e}.json"; done
  job_run plan 900 "cd $POD_REPO && $POD_PYTHON scripts/3_run/run_campaign.py plan --session $SESSION --floor-table results/calibration/floor_table.json --window-duration-s $WINDOW_DURATION_S --seed $SEED$qm$cal --calibration-budget-fraction $CALIBRATION_BUDGET_FRACTION --freeze-file $POD_FREEZE_FILE --out results/calibration/plan_$RUN_ID.json" "$LAND/logs/setup" || return 1
  run_step 0 "fetch plan.json" -- pscp_from "$POD_REPO/results/calibration/plan_$RUN_ID.json" "$EXTRAS/plan.json" || return 1
  plan_only && return 0
  run_step 0 "plan argv audit (S0F-26)" -- "$PY3" - "$EXTRAS/plan.json" <<'PY' || return 1
import json, sys
plan = json.load(open(sys.argv[1], encoding="utf-8"))
steps = plan.get("steps", [])
cells = [s for s in steps if s.get("kind") != "relaunch"]
bad = []
for s in cells:
    argv = s.get("argv", [])
    n = sum(1 for a in argv if a == "--vllm-telemetry")
    eng = s.get("engine")
    if eng == "hf" and n != 0:
        bad.append(f"hf cell carries --vllm-telemetry: {s.get('row_key', '?')}")
    if eng not in (None, "hf") and n != 1:
        bad.append(f"{eng} cell carries --vllm-telemetry {n} times: {s.get('row_key', '?')}")
    if s.get("topology") == "pd" and not any(k.startswith("CAGE_PD_") for k in (s.get("env") or {})):
        bad.append(f"pd cell without CAGE_PD_* env: {s.get('row_key', '?')}")
print(f"plan schema={plan.get('schema')} steps={len(steps)} cells={len(cells)} relaunches={len(steps)-len(cells)} blocked={len(plan.get('blocked_row_keys', []))}")
for b in bad: print("  [FAIL] " + b)
sys.exit(1 if bad else 0)
PY
  return 0
}

stage_run() {
  pod_env
  local bound=$(( HOURS * 3600 ))
  if ! plan_only; then
    mkdir -p "$EXTRAS/monitor"
    nohup bash "$SCRIPT_DIR/monitor_pod.sh" --pod "$POD_ID" --ssh "$POD_SSH" --port "$POD_PORT" ${SSH_KEY:+--key "$SSH_KEY"} --run-root "$POD_RUN_ROOT" --pod-repo "$POD_REPO" --out "$EXTRAS/monitor" --interval "$MONITOR_INTERVAL_S" ${PRICE:+--price "$PRICE"} --created "$(state get pod.created_utc 2>/dev/null || echo "")" --balance-floor "$BALANCE_FLOOR_USD" --stall-min "$CELL_STALL_MIN" --job run --backup-backend "$( [ "$BACKUP_SCHEME" = "file" ] && echo local || echo "$BACKUP_SCHEME")" --scripts "$SCRIPTS" --parent $$ > "$EXTRAS/monitor/monitor.out" 2>&1 &
    MONITOR_PID=$!
    state put monitor.pid "$MONITOR_PID"
    say "monitor started (pid $MONITOR_PID, every ${MONITOR_INTERVAL_S}s) -> $EXTRAS/monitor/"
  else
    printf '  [plan] monitor_pod.sh in the background: --pod %s --run-root %s --interval %s\n' "$POD_ID" "$POD_RUN_ROOT" "$MONITOR_INTERVAL_S"
  fi
  run_step 0 "backup daemon start" -- pssh "cd $POD_REPO && CAGE_BACKUP_TARGET=$BACKUP_TARGET CAGE_BACKUP_INTERVAL=$BACKUP_INTERVAL_S bash scripts/5_observability/gcs_backup_daemon.sh start $BACKUP_RUN_REL && bash scripts/5_observability/gcs_backup_daemon.sh status $BACKUP_RUN_REL" || return 1
  plan_only || step_has "[gcs-backup] RUNNING (" || { printf '  [FAIL] backup daemon not RUNNING\n'; return 1; }
  local name="run"
  run_step 0 "job $name: submit" -- bash "$PODJOB" submit "$name" "cd $POD_REPO && env $RUN_ENV_UNSET CAGE_RUN_ROOT=$POD_RUN_ROOT VLLM_START_TIMEOUT=$VLLM_START_TIMEOUT $POD_PYTHON scripts/3_run/run_campaign.py run --plan results/calibration/plan_$RUN_ID.json --campaign-root $POD_RUN_ROOT --seal" "$bound" || return 1
  if plan_only; then
    printf '  [plan] %-28s expect DONE(0) or STOP FAILED(2): bash %s wait run %s\n' "job run: wait" "$PODJOB" "$bound"
    return 0
  fi
  local wrc=0
  bash "$PODJOB" wait "$name" "$bound" | tee "$EXTRAS/steps/run_wait.log" || wrc="${PIPESTATUS[0]}"
  bash "$PODJOB" fetch "$name" "$LAND/logs/runner" || true
  JOB_LOG="$LAND/logs/runner/run.log"
  if [ "$wrc" -eq 124 ]; then printf '  [FAIL] the run exceeded its %s h bound and is STILL RUNNING on the pod (see PORTAL ACTION)\n' "$HOURS"; return 1; fi
  if [ "$wrc" -ne 0 ]; then
    if grep -q "REFUSED:" "$JOB_LOG" 2>/dev/null; then printf '  [FAIL] run REFUSED:\n'; grep "REFUSED:" "$JOB_LOG" | head -3; return 1; fi
    if grep -q "STOP FAILED" "$JOB_LOG" 2>/dev/null && grep -q "FAILED(2)" "$EXTRAS/steps/run_wait.log"; then
      say "run finished with exit 2 = STOP FAILED only (data tree complete; the pod may still hold an engine)"
      state note run "exit 2: STOP FAILED (ADR-0139); data tree complete"
    else
      printf '  [FAIL] run ended %s (see %s)\n' "$(tail -n 1 "$EXTRAS/steps/run_wait.log")" "$JOB_LOG"; return 1
    fi
  fi
  run_step 0 "no failure sentinel, every window has regime.json" -- pssh "s=\$(find $POD_RUN_ROOT/cells -name '.STATUS-*' 2>/dev/null | wc -l | tr -d ' '); w=\$(find $POD_RUN_ROOT/cells -type d -name 'window_*' 2>/dev/null | wc -l | tr -d ' '); r=\$(find $POD_RUN_ROOT/cells -name regime.json 2>/dev/null | wc -l | tr -d ' '); echo sentinels=\$s windows=\$w regimes=\$r; [ \"\$s\" -eq 0 ] && [ \"\$w\" -gt 0 ] && [ \"\$w\" -eq \"\$r\" ]" || return 1
  return 0
}

stage_monitor() {
  plan_only && { printf '  [plan] %-28s extras/monitor/status.json present, alerts.log without a HARD line\n' "monitor evidence"; return 0; }
  # The first tick needs one runpodctl read, one ssh bundle and one nvidia-smi
  # sample; wait a bounded time for it instead of judging a monitor that has
  # not ticked yet (a run of seconds against fakes, hours against a pod).
  local waited=0 limit=$(( MONITOR_INTERVAL_S * 2 + 60 )) mpid
  mpid="$(state get monitor.pid 2>/dev/null || true)"
  while [ ! -f "$EXTRAS/monitor/status.json" ] && [ "$waited" -lt "$limit" ]; do
    if [ -n "$mpid" ] && ! kill -0 "$mpid" 2>/dev/null; then break; fi
    sleep 2; waited=$((waited + 2))
  done
  if [ ! -f "$EXTRAS/monitor/status.json" ]; then
    printf '  [FAIL] no monitor status.json after %ss (monitor pid %s %s); monitor.out tail:\n' "$waited" "${mpid:-none}" "$( [ -n "$mpid" ] && kill -0 "$mpid" 2>/dev/null && echo alive || echo dead)"
    tail -n 20 "$EXTRAS/monitor/monitor.out" 2>/dev/null | sed 's/^/    /'
    return 1
  fi
  if [ -f "$EXTRAS/monitor/alerts.log" ] && grep -q "HARD" "$EXTRAS/monitor/alerts.log"; then
    printf '  [FAIL] unacknowledged HARD alerts:\n'; grep "HARD" "$EXTRAS/monitor/alerts.log" | tail -n 10
    printf '  review them, then resume with --from seal\n'; return 1
  fi
  state evidence monitor "$EXTRAS/monitor/status.json"
  return 0
}

stage_seal() {
  pod_env
  if ! plan_only && pssh "test -f $POD_RUN_ROOT/ledger.json" 2>/dev/null; then say "already sealed (ledger.json present)"; return 0; fi
  job_run seal 600 "cd $POD_REPO && $POD_PYTHON scripts/3_run/seal_campaign_run.py $POD_RUN_ROOT" "$LAND/logs/runner" || return 1
  return 0
}

stage_score() {
  pod_env
  if [ -z "${MAX_NULL_FRACTION:-}" ]; then
    if plan_only; then
      printf '  [plan] NOTE: MAX_NULL_FRACTION is empty in the profile; the live stage REFUSES until the registered bound is stated (build_predicate_table.py --max-null-fraction)\n'
    else
      printf '  [FAIL] MAX_NULL_FRACTION is empty in the profile: build_predicate_table.py needs the registered bound stated (no silent default)\n'; return 1
    fi
  fi
  local bound=$(( SCORE_BOUND_MIN * 60 )) force=""
  # Scoring passes are append-only (a bug means a NEW SCORING_RUN_ID), so an
  # existing scoring/<id>/ is reused, and the derived predicate table is
  # rebuilt with --force on --redo (review 2026-10-06, MEDIUM 7).
  [ "$REDO" -eq 1 ] && force=" --force"
  job_run score "$bound" "cd $POD_REPO && { [ -d $POD_RUN_ROOT/scoring/$SCORING_RUN_ID ] && echo 'scoring/$SCORING_RUN_ID exists: reused (append-only)' || $POD_PYTHON scripts/4_analysis/rescore_quality.py --run-root $POD_RUN_ROOT --full --device cuda --scoring-run-id $SCORING_RUN_ID; } && $POD_PYTHON scripts/4_analysis/build_predicate_table.py $POD_RUN_ROOT --scoring-run-id $SCORING_RUN_ID --max-null-fraction $MAX_NULL_FRACTION --freeze-file $POD_FREEZE_FILE$force" "$LAND/logs/runner" || return 1
  return 0
}

stage_collect() {
  pod_env
  local backend; backend="$( [ "$BACKUP_SCHEME" = "file" ] && echo local || echo "$BACKUP_SCHEME")"
  run_step 0 "backup daemon stop + marker check" -- pssh "cd $POD_REPO && t0=\$(date -u +%s); CAGE_BACKUP_TARGET=$BACKUP_TARGET bash scripts/5_observability/gcs_backup_daemon.sh stop $BACKUP_RUN_REL; m=\$(stat -c %Y .agent/last_sync_ok_$backend 2>/dev/null || echo 0); echo marker_mtime=\$m stop_started=\$t0; [ \"\$m\" -ge \"\$t0\" ]" || return 1
  local token="collect_${EXP}_$(date -u +%Y%m%d_%H%M%S)"
  job_run collect 1800 "cd $POD_REPO && CAGE_BACKUP_TARGET=$BACKUP_TARGET CAGE_COLLECT_TOKEN=$token bash scripts/5_observability/collect_logs.sh" "$LAND/logs/system" || return 1
  plan_only || grep -q "sentinel=COLLECT_OK_$token" "$JOB_LOG" || { printf '  [FAIL] collect_logs.sh wrote no COLLECT_OK_%s sentinel\n' "$token"; return 1; }
  plan_only || state put collect.token "$token"
  plan_only || stop_monitor
  return 0
}

stage_pull() {
  pod_env
  local host="${POD_SSH}"
  run_step 0 "verified pull (ledger gate)" -- env CAGE_SSH_OPTS="$SSH_OPTS" bash "$SCRIPTS/5_observability/pull_run.sh" "ssh://$host$POD_BACKUP_DIR/$BACKUP_RUN_REL" "$RUN_LOCAL" || return 1
  plan_only || step_has "SAFE TO TEARDOWN" || { printf '  [FAIL] pull_run.sh did not print SAFE TO TEARDOWN\n'; return 1; }
  plan_only || state put pull.verdict "SAFE TO TEARDOWN ($(utc))"
  plan_only || mkdir -p "$LAND/logs/engines" "$LAND/logs/runner/pod_jobs" "$LAND/logs/system/vm_logs" "$EXTRAS/calibration/pod"
  run_step 0 "pull engine logs" -- pscp_from "$POD_REPO/logs/." "$LAND/logs/engines/" || return 1
  run_step 0 "pull job logs" -- pscp_from ".cage_jobs/." "$LAND/logs/runner/pod_jobs/" || return 1
  run_step 0 "pull vm_logs forensics" -- pscp_from "$POD_BACKUP_DIR/vm_logs/." "$LAND/logs/system/vm_logs/" || return 1
  run_step 0 "pull pod calibration dir" -- pscp_from "$POD_REPO/results/calibration/." "$EXTRAS/calibration/pod/" || return 1
  return 0
}

stage_analyze() {
  pod_env
  local force=""
  [ -d "$RUN_LOCAL/index" ] && force="--force"
  run_step 0 "verify_results" -- "$PY" "$SCRIPTS/4_analysis/verify_results.py" "$RUN_LOCAL" --out "$EXTRAS/verify_$(date -u +%H%M%S)" || return 1
  # shellcheck disable=SC2086
  run_step 0 "organize_results" -- "$PY" "$SCRIPTS/4_analysis/organize_results.py" "$RUN_LOCAL" $force || return 1
  # The driver's one-look lock refuses a second run on the same root (review
  # 2026-10-06, MEDIUM 7): on --redo the stamp this master recorded is reused.
  local prev_stamp; prev_stamp="$(state get analysis.stamp 2>/dev/null || true)"
  if [ "$REDO" -eq 1 ] && [ -n "$prev_stamp" ] && [ -f "$RUN_LOCAL/analysis/$prev_stamp/stats.json" ]; then
    say "reusing analysis stamp $prev_stamp (the driver's one-look lock; a new look needs a new run root)"
  else
    # shellcheck disable=SC2086
    run_step 0 "run_campaign_analysis (design input)" -- "$PY" "$SCRIPTS/4_analysis/run_campaign_analysis.py" "$RUN_LOCAL" --contrasts $CONTRASTS || return 1
  fi
  if ! plan_only; then
    local stamp; stamp="$(ls -1 "$RUN_LOCAL/analysis" 2>/dev/null | sort | tail -n 1)"
    [ -n "$stamp" ] || { printf '  [FAIL] no analysis/<stamp>/ under %s\n' "$RUN_LOCAL"; return 1; }
    mkdir -p "$LAND/plots/$stamp"
    cp "$RUN_LOCAL/analysis/$stamp"/*.png "$LAND/plots/$stamp/" 2>/dev/null || true
    cp "$RUN_LOCAL/analysis/$stamp/stats.json" "$RUN_LOCAL/analysis/$stamp/summary.md" "$LAND/plots/$stamp/" 2>/dev/null || true
    state put analysis.stamp "$stamp"
  fi
  run_step 0 "window panels" -- "$PY" "$SCRIPTS/4_analysis/render_window_panels.py" "$RUN_LOCAL" --out "$LAND/plots/panels_$(date -u +%H%M%S)" ${PRICE:+--price-per-hour "$PRICE"} || return 1
  run_step 0 "analysis.txt" -- "$PY" "$SCRIPT_DIR/write_analysis.py" --landing "$LAND" --run-root "$RUN_LOCAL" --out "$LAND/plots/analysis.txt" || return 1
  return 0
}

stage_teardown() {
  pod_env
  plan_only || state go teardown "--yes teardown on the command line ($(utc))"
  local host="${POD_SSH}"
  # teardown_pod.sh exits 1 at [5/5] while the volume is still listed (expected here),
  # 0 only when nothing survives; the pod delete is [4/5]. Markers, not the code.
  run_step "0|1" "teardown_pod.sh (0, or 1 with the volume still listed)" -- env CAGE_ASSUME_YES=1 CAGE_POD_SSH="$host" CAGE_SSH_OPTS="$SSH_OPTS" bash "$SCRIPTS/runpod/teardown_pod.sh" "$POD_ID" "ssh://$host$POD_BACKUP_DIR/$BACKUP_RUN_REL" "$RUN_LOCAL" || return 1
  local trc=0
  if ! plan_only; then
    if ! step_has "[4/5] deleting pod" || step_has "ERROR:"; then printf '  [FAIL] the pod delete did not run cleanly (rc=%s); see the step log\n' "$trc"; return 1; fi
    step_has "TEARDOWN_COMPLETE" && say "teardown_pod.sh reports TEARDOWN_COMPLETE"
  fi
  run_step 0 "network volume delete" -- runpodctl network-volume delete "$VOL_ID" || return 1
  run_step 0 "listing: pod list --all" -- runpodctl pod list --all || return 1
  plan_only || { grep -v '^#' "$STEP_LOG" > "$EXTRAS/ops/pods_after.json"; json_empty "$EXTRAS/ops/pods_after.json" || { printf '  [FAIL] pods STILL LISTED (STILL BILLING): %s\n' "$EXTRAS/ops/pods_after.json"; return 1; }; }
  run_step 0 "listing: network-volume list" -- runpodctl network-volume list || return 1
  plan_only || { grep -v '^#' "$STEP_LOG" > "$EXTRAS/ops/volumes_after.json"; json_empty "$EXTRAS/ops/volumes_after.json" || { printf '  [FAIL] volumes STILL LISTED (STILL BILLING): %s\n' "$EXTRAS/ops/volumes_after.json"; return 1; }; }
  run_step 0 "cost report (--billing)" -- bash "$SCRIPTS/runpod/cost_report.sh" --billing || return 1
  plan_only || cp "$STEP_LOG" "$EXTRAS/ops/cost_report.txt"
  run_step 0 "account balance (after)" -- runpodctl user || return 1
  if ! plan_only; then
    cp "$STEP_LOG" "$EXTRAS/ops/user_after.txt"
    state put cost.balance_after_raw "$(grep -v '^#' "$STEP_LOG" | tr -d '\n' | cut -c1-400)"
    state put cost.true_zero_utc "$(utc)"
    state put pod.deleted_utc "$(utc)"
  fi
  return 0
}

# ---------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------
run_stage() {
  local name="$1" fn rc=0
  fn="stage_$(printf '%s' "$name" | tr '-' '_')"
  CUR_STAGE="$name"; STEP_N=0
  hr; say "stage $(stage_index "$name"): $name $( plan_only && printf '(plan)' )"
  plan_only || state begin "$name" 0
  set +e; "$fn"; rc=$?; set -e
  plan_only || state end "$name" "$rc"
  if [ "$rc" -ne 0 ]; then
    say "stage $name FAILED (rc=$rc)"
    plan_only || portal_block
    return 1
  fi
  say "stage $name passed"
  return 0
}

LOCK=""
on_exit() {
  local rc=$?
  [ "$PLAN" -eq 1 ] || stop_monitor 2>/dev/null || true
  [ -z "$LOCK" ] || rm -f "$LOCK"
  exit "$rc"
}

take_lock() {
  # One master per landing folder (review 2026-10-06, MEDIUM 9): the lock holds
  # the pid; a live pid whose command line is this script refuses; anything
  # else is a stale lock from a killed master and is taken over loudly.
  LOCK="$EXTRAS/.master.lock"
  local pid
  if [ -f "$LOCK" ]; then
    pid="$(cat "$LOCK" 2>/dev/null || true)"
    if [ -n "$pid" ] && ps -ww -p "$pid" -o command= 2>/dev/null | grep -q "cage_experiment.sh"; then
      LOCK=""; die "another master (pid $pid) is running on $LAND; wait for it or stop it by that pid"
    fi
    [ -z "$pid" ] || say "stale master lock (pid $pid is gone): taken over"
  fi
  printf '%s\n' "$$" > "$LOCK"
}

if [ "$PLAN" -eq 1 ]; then
  say "PLAN mode: printing every stage's commands for $EXP ($DATE); nothing runs, nothing is created"
  say "landing: $LAND"
  for s in $STAGES; do
    if [ -n "$FROM" ] && [ "$(stage_index "$s")" -lt "$(stage_index "$FROM")" ]; then continue; fi
    if [ -n "$ONLY" ] && [ "$s" != "$ONLY" ]; then continue; fi
    run_stage "$s" || exit 1
  done
  hr; say "plan complete"
  exit 0
fi

mkdir -p "$LAND/run" "$LAND/logs/engines" "$LAND/logs/runner" "$LAND/logs/setup" "$LAND/logs/system" \
         "$LAND/plots" "$EXTRAS/calibration" "$JOBS" "$EXTRAS/ops" "$EXTRAS/monitor" "$EXTRAS/verify" "$EXTRAS/probes" "$EXTRAS/steps"
if [ -f "$STATE" ]; then
  "$PY3" -c 'import json,sys; json.load(open(sys.argv[1]))' "$STATE" 2>/dev/null \
    || die "state file is not valid JSON: $STATE (fix or move it aside; nothing is guessed)"
  running="$(state running)"
  if [ -n "$running" ] && [ "$REDO" -ne 1 ]; then
    die "stage '$running' is recorded as RUNNING in $STATE (another master, or a crash); inspect, then pass --from $running --redo to take it over"
  fi
else
  state init "$EXP" "$DATE"
fi
[ -f "$EXTRAS/profile.env" ] || {
  { printf '# resolved profile for %s %s (frozen at %s)\n' "$EXP" "$DATE" "$(utc)"; cat "$SCRIPT_DIR/profiles/_common.env" "$PROFILE"; } > "$EXTRAS/profile.env"
}
trap on_exit EXIT
take_lock

say "experiment $EXP ($DATE) -> $LAND"
for s in $STAGES; do
  if [ -n "$FROM" ] && [ "$(stage_index "$s")" -lt "$(stage_index "$FROM")" ]; then continue; fi
  if [ -n "$TO" ] && [ "$(stage_index "$s")" -gt "$(stage_index "$TO")" ]; then
    hr; say "stopped after $TO (--to). The pod keeps billing until teardown; next: --from $s, or teardown by hand (teardown_pod.sh ... --force for a tree with no sealed run)"
    CUR_STAGE="$s"; portal_block; exit 0
  fi
  if [ -n "$ONLY" ]; then
    [ "$s" = "$ONLY" ] || continue
    prev="$(stage_prev "$s")"
    # --redo repeats the named stage; it never waives the predecessor rule
    if [ -n "$prev" ] && [ "$(state status "$prev")" != "passed" ]; then
      die "--only $s: its predecessor '$prev' has not passed (status: $(state status "$prev")); run the stages in order (--from $prev)"
    fi
  fi
  st="$(state status "$s")"
  if [ "$st" = "passed" ] && [ "$REDO" -ne 1 ]; then say "stage $s already passed (use --redo to repeat)"; continue; fi
  if is_billable "$s" && ! has_yes "$s"; then
    hr; say "stage $s is BILLABLE or IRREVERSIBLE: this is what it would run (plan print). Re-run with --yes $s to execute it."
    PLAN=1; CUR_STAGE="$s"; STEP_N=0; set +e; "stage_$(printf '%s' "$s" | tr '-' '_')"; set -e; PLAN=0
    say "stopped before $s (no --yes $s). Resume: bash $0 $EXP --date $DATE --from $s --yes $s"
    exit 10
  fi
  run_stage "$s" || exit 1
done
hr; say "all requested stages passed"; state summary
