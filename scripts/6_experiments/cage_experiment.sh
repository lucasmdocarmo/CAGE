#!/bin/bash
# Order:     THE master: stages 0 to 14 of one experiment on the Mac (preflight-mac, provision, ship, setup, validate, calibrate, plan, run, monitor, seal, score, collect, pull, teardown, analyze); it calls the existing stage scripts in order and never reimplements their checks
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

# Audit 2026-10-17 (G18): teardown comes right after the verified pull; analyze
# reads Mac files only and runs after it, so no pod hour is spent on Mac work
# and an unexpected exit in the analysis never leaves a pod billing.
STAGES="preflight-mac provision ship setup validate calibrate plan run monitor seal score collect pull teardown analyze"
BILLABLE_STAGES="provision teardown"
# Audit 2026-10-17 (G16): the stages whose bounds the seatbelt must cover and
# whose entry checks the watchdog (ship through pull; teardown deletes the pod).
POD_STAGES="ship setup validate calibrate plan run monitor seal score collect pull"
pod_stage() { case " $POD_STAGES " in *" $1 "*) return 0 ;; esac; return 1; }

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
# S0F-39 (live CA-MTL-1, 2026-10-07): the network volume (NFS4, vast backend)
# refuses "|" in names, and every cell directory is named by its row key, so
# the results root can live on the container disk instead: when the profile
# sets POD_RESULTS_DIR, stage 2 moves <repo>/results there and leaves a
# symlink at <repo>/results, so every relative path stays valid. Empty = the
# repo's own results directory. The backup target must then move too
# (profile keys BACKUP_TARGET and POD_BACKUP_DIR); that data lives on the
# container disk until stage 12 pulls it.
POD_RESULTS_DIR="${POD_RESULTS_DIR:-}"
POD_TARBALL="${POD_TARBALL:-/root/cage_repo.tar.gz}"
# The vLLM launcher runs a bare `vllm serve` from PATH (manage_vllm_server.sh)
# and the driver calls that launcher, so every pod job that starts an engine
# puts the driver venv's bin first and carries the start timeout (live
# 2026-10-07: the validate job's start died with "failed to run command
# 'vllm'" and the preflight ran under the system python3).
POD_VENV_BIN="$POD_REPO/${POD_PYTHON%/*}"
# Engine levers the profile sets reach the pod only through the job command:
# the ones the launchers read are forwarded (S0F-35: the SGLang piecewise
# CUDA-graph lever; the string is empty when the profile leaves it unset).
POD_ENGINE_ENV=""
[ "${CAGE_SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH:-0}" = "1" ] && POD_ENGINE_ENV=" CAGE_SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH=1"
# ADR-0162 (S0 attempt 2): SGLang starts took 200 to 208 s of the launcher's
# 300 s default [V]; the profile's SGLANG_START_TIMEOUT rides every engine job
# beside VLLM_START_TIMEOUT (600 unless the profile says otherwise).
SGLANG_START_TIMEOUT="${SGLANG_START_TIMEOUT:-600}"
POD_ENGINE_ENV=" SGLANG_START_TIMEOUT=$SGLANG_START_TIMEOUT$POD_ENGINE_ENV"
FREEZE_FILE="${FREEZE_FILE:-$PROJECT_DIR/MyDocs/registration/freeze_resolutions.json}"
# A10 (owner 2026-10-20): the served model's pinned revision (freeze slot
# served_model); every engine job then loads it offline at that revision, and
# setup prefetches every pinned revision and checks the Hub (a newer served
# revision stops setup, never an automatic switch).
MODEL_REVISION=""; PREFETCH_REVISIONS=""
if [ "${PIN_MODEL_REVISION:-0}" = "1" ]; then
  # review 2026-10-21 L2: only a pin for THIS model counts (stage 0 refuses an empty one)
  MODEL_REVISION="$("${PY3:-python3}" -c 'import json,sys; s = ((json.load(open(sys.argv[1], encoding="utf-8")).get("INSTRUMENT_REVISIONS") or {}).get("served_model") or {}); print(s.get("revision") or "" if s.get("model") == sys.argv[2] else "")' "$FREEZE_FILE" "${MODEL:-}" 2>/dev/null || true)"
  PREFETCH_REVISIONS="$("${PY3:-python3}" -c 'import json,sys
d = json.load(open(sys.argv[1], encoding="utf-8")).get("INSTRUMENT_REVISIONS") or {}
print(" ".join("%s=%s" % (v["model"], v["revision"]) for v in d.values() if isinstance(v, dict) and v.get("model") and v.get("revision")))' "$FREEZE_FILE" 2>/dev/null || true)"
  [ -n "$MODEL_REVISION" ] && POD_ENGINE_ENV=" HF_HUB_OFFLINE=1 CAGE_MODEL_REVISION=$MODEL_REVISION$POD_ENGINE_ENV"
fi
POD_FREEZE_FILE="MyDocs/registration/freeze_resolutions.json"
SCORE_BOUND_MIN="${SCORE_BOUND_MIN:-180}"
FAIL_PROBE_NO_VERDICT="SCORING_BATCH_FAILED reason=no verdict line (the probe crashed before its verdict; see logs/runner/score_probe.log)"
POD_READY_TIMEOUT_S="${CAGE_POD_READY_TIMEOUT_S:-900}"
POD_READY_POLL_S="${CAGE_POD_READY_POLL_S:-15}"
# Per-stage job bounds (seconds); the seatbelt must cover their sum plus the run.
# Audit 2026-10-17 (G16): the pull margin was 60 min against S0 attempt 5's
# measured pull of 68 min; the seal (600 s), the stage 4 start check (2,400 s)
# and the readiness wait (POD_READY_TIMEOUT_S) were missing from the sum.
VALIDATE_BOUND_S=1500; CALIBRATE_BOUND_S=3600; PLAN_BOUND_S=900; COLLECT_BOUND_S=1800
# ADR-0176 Group C (owner 2026-10-22): the pull margin is a profile key (default
# in _common.env, 90 min from S0 attempt 5's 68 min pull); S1 moves about three
# times S0's bytes and sets its own.
: "${PULL_MARGIN_MIN:=90}"
SEAL_BOUND_S=600; START_CHECK_BOUND_S=2400
# ADR-0154 / ADR-0156: the per-engine rung calibration job bound (profile key,
# default in _common.env): up to PROBE_MAX_STEPS windows of 75 s plus a relaunch
# per rung, for TWO classes (the anchor and the engine's smallest class).
: "${CALIBRATE_RUNGS_BOUND_S:=18000}"
# ADR-0177 (review 2026-10-25 H1): the ceiling-only ladder job bound per engine. Each
# interpolated class climbs from its own class-floor start (A16), so S1 needs about
# 94 to 98 windows per engine at the measured 222 s per window (stage 5: 46 windows
# in 10,195 s), 21,600 to 24,100 s [D]; the default doubles the larger figure's order.
: "${CLASS_CEILINGS_BOUND_S:=36000}"

seatbelt_minutes() { # "24h" | "90m" -> minutes, or empty when malformed
  # Audit 2026-10-17 (G30): "<n>d" is refused here; provision_pod.sh accepts h,
  # m or a timestamp only, so a day form passed stage 0 and failed at stage 1.
  case "$1" in
    *h) printf '%s' "$(( ${1%h} * 60 ))" ;;
    *m) printf '%s' "${1%m}" ;;
    *) return 1 ;;
  esac
}
stage_bound_minutes() { # G16: the pod-side bound of ONE stage, in minutes (one table for the seatbelt sum and the per-stage guard)
  local n_srv=0 n_cal=0 e suite_min=0 start_min=0
  for e in $ENGINES; do [ "$e" = "hf" ] || n_srv=$((n_srv + 1)); done
  for e in $CALIBRATE; do n_cal=$((n_cal + 1)); done
  case "$1" in
    setup)     printf '%s' "$SETUP_BOUND_MIN" ;;
    validate)
      # S0F-88 review L4: stage 4's load gate (600 s) and pod suite (1,800 s when
      # SUITE_ON_POD=1) bill too; review 2026-10-13 A1: the start check (START_CHECK=1)
      [ "${SUITE_ON_POD:-0}" = "1" ] && suite_min=30
      [ "${START_CHECK:-1}" = "1" ] && start_min=$(( START_CHECK_BOUND_S / 60 ))
      # S1 audit 2026-10-23: the load gate (600 s) and the offline dataset probe (600 s)
      printf '%s' "$(( 20 + suite_min + n_srv * VALIDATE_BOUND_S / 60 + start_min ))" ;;
    calibrate) printf '%s' "$(( n_cal * CALIBRATE_BOUND_S / 60 + n_cal * CALIBRATE_RUNGS_BOUND_S / 60 ))" ;;
    plan)
      # ADR-0177: CLASS_CEILINGS=1 adds the ceiling-only ladders (one bound per
      # engine) and 60 min for the re-label and the drift window
      local plan_min=$(( PLAN_BOUND_S / 60 ))
      [ "${CLASS_CEILINGS:-0}" = "1" ] && plan_min=$(( plan_min + n_cal * CLASS_CEILINGS_BOUND_S / 60 + 60 ))
      printf '%s' "$plan_min" ;;
    run)       printf '%s' "$(( HOURS * 60 ))" ;;
    seal)      printf '%s' "$(( SEAL_BOUND_S / 60 ))" ;;
    score)     printf '%s' "$SCORE_BOUND_MIN" ;;
    collect)   printf '%s' "$(( COLLECT_BOUND_S / 60 ))" ;;
    pull)      printf '%s' "$PULL_MARGIN_MIN" ;;
    *)         printf '%s' 0 ;;   # ship and monitor: minutes inside the neighbors' slack
  esac
}
remaining_bound_minutes() { # G16: the pod-side bounds from stage $1 (inclusive) through pull
  local total=0 s seen=0
  for s in $POD_STAGES; do
    [ "$s" = "$1" ] && seen=1
    [ "$seen" -eq 1 ] || continue
    total=$(( total + $(stage_bound_minutes "$s") ))
  done
  printf '%s' "$total"
}
seatbelt_need_minutes() { # the readiness wait plus every pod-side stage bound, in minutes
  printf '%s' "$(( POD_READY_TIMEOUT_S / 60 + $(remaining_bound_minutes ship) ))"
}
watchdog_guard() { # G16: before a pod-side stage, the seatbelt is ALIVE and its deadline covers the remaining bounds
  local out dl left need
  out="$(bash "$SCRIPTS/runpod/pod_watchdog.sh" status "$POD_ID" 2>&1)" || true
  if ! printf '%s\n' "$out" | grep -q "watchdog=ALIVE"; then
    printf '  [FAIL] the seatbelt is not ALIVE for %s before stage %s (a Mac reboot or a disarm leaves the pod unguarded): re-arm it (bash %s/runpod/pod_watchdog.sh arm %s <deadline>) and resume\n' "$POD_ID" "$1" "$SCRIPTS" "$POD_ID"
    printf '%s\n' "$out" | sed 's/^/    /'
    return 1
  fi
  dl="$(printf '%s\n' "$out" | sed -nE 's/.*deadline=([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z).*/\1/p' | head -1)"
  [ -n "$dl" ] || { say "watchdog ALIVE for $POD_ID (no deadline field in its status: the time left is not judged)"; state note "$1" "watchdog ALIVE, deadline not judged (no deadline field)"; return 0; }
  left="$("$PY3" -c 'import sys, datetime; d = datetime.datetime.strptime(sys.argv[1], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc); print(int((d - datetime.datetime.now(datetime.timezone.utc)).total_seconds() // 60))' "$dl" 2>/dev/null || echo "")"
  need="$(remaining_bound_minutes "$1")"
  [ -n "$left" ] || state note "$1" "watchdog ALIVE, deadline $dl unparsed: the time left was not judged (review batch 4, L2)"
  if [ -n "$left" ] && [ "$left" -lt "$need" ]; then
    printf '  [FAIL] the seatbelt fires in %s min (deadline %s) but the stages from %s through pull need %s min (G16): re-arm the watchdog with a later deadline, then resume\n' "$left" "$dl" "$1" "$need"
    return 1
  fi
  say "watchdog ALIVE for $POD_ID: ${left:-?} min left, the stages from $1 through pull need $need min"
  return 0
}
pod_hours_bound() { # S0F-88 P-4: the pod bills through every stage bound, not the run (HOURS) alone
  awk -v m="$(seatbelt_need_minutes)" 'BEGIN { printf "%.1f", m / 60 }'
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
  # expected_rc is one code or several joined by "|" (the yield table: 0 or 3).
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
# Decision 3 (2026-10-13, S0 only): PARTIAL_TREE_OK=1 lets a run stage that
# ended with failed cells, failed relaunches or failed dry windows pass as
# PARTIAL (recorded in state run.partial), so the tree is still sealed, scored
# and analyzed; stage 0 refuses the key on a registered profile.
partial_ok() { [ "${PARTIAL_TREE_OK:-0}" = "1" ]; }
run_partial() { [ -n "$(state get run.partial 2>/dev/null || true)" ]; }

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
# ADR-0161 (S0 attempt 2, 2026-10-10): one timed-out channel must not end a
# 16 h run. Up to 3 attempts when the previous one exited 255 (ssh and scp both
# return 255 for a refused or timed-out connection, measured 2026-10-10); any
# other code returns at once, so a failing REMOTE command is reported after one
# attempt. Only idempotent round trips go through here: every pssh command of
# this script repeats safely (tar over an existing tree, probes, counts, the
# backup daemon's "already running" path) and every copy is a copy; the backup
# stop's time-anchored marker check uses pssh_once. Job submits never retry: a repeat could start a second engine
# (pod_job.sh refuses a RUNNING name, but a submit whose reply was lost has no
# local handle to check against).
SSH_RETRY_S="${CAGE_SSH_RETRY_S:-20}"
ssh_retry() {
  local n=1 rc
  while :; do
    rc=0; "$@" || rc=$?
    [ "$rc" -eq 255 ] || return "$rc"
    [ "$n" -lt 3 ] || return 255
    # stderr: a caller that captures stdout as data ($(pssh ...)) never sees it;
    # run_step merges 2>&1 so the step log keeps it (review 2026-10-10, MEDIUM 1)
    printf '  [retry] ssh channel to the pod failed (exit 255) at %s: attempt %s of 3 in %ss (ADR-0161)\n' "$(utc)" "$((n + 1))" "$SSH_RETRY_S" >&2
    sleep "$SSH_RETRY_S"
    n=$((n + 1))
  done
}
pssh() { # pssh '<remote command>'
  # shellcheck disable=SC2086
  ssh_retry ssh $SSH_OPTS "$POD_SSH" "$1"
}
pssh_once() { # pssh_once '<remote command>': ONE attempt, for a command whose repeat
  # is not safe (review 2026-10-10, LOW 5: the backup stop samples t0 remotely and
  # compares the sync marker with it; a retry after a mid-command drop re-samples
  # t0 after the marker and fails a stop that succeeded). A blip there fails the
  # stage, and the resume runs it again from a clean state.
  # shellcheck disable=SC2086
  ssh $SSH_OPTS "$POD_SSH" "$1"
}
pscp_to()   { # pscp_to <local> <remote path>
  # shellcheck disable=SC2086
  ssh_retry scp -P "$POD_PORT" ${SSH_KEY:+-i "$SSH_KEY"} -o BatchMode=yes -o StrictHostKeyChecking=no "$1" "$POD_SSH:$2"; }
pscp_from() { # pscp_from <remote path> <local>; a remote DIRECTORY is given as "<dir>/."
  # so its contents land in <local> instead of nesting as <local>/<dir> (scp -r
  # into an existing directory nests; review 2026-10-06, LOW 13) [A on OpenSSH].
  # shellcheck disable=SC2086
  ssh_retry scp -r -P "$POD_PORT" ${SSH_KEY:+-i "$SSH_KEY"} -o BatchMode=yes -o StrictHostKeyChecking=no "$POD_SSH:$1" "$2"; }

TAKEOVER_STAGE=""   # G5: the stage recorded RUNNING when this master took over (--redo)
TAKEOVER_STARTED="" # review 2026-10-23 M5: that stage's started_utc, so only a job handle submitted during it is reused
job_attach_or_submit() {
  # job_attach_or_submit <name> <remote command> <deadline_s>: audit 2026-10-17
  # (G5). After a Mac crash or an ssh outage past the grace, the resume used to
  # submit the job again: pod_job.sh refused a RUNNING one (the stage died) or,
  # on a finished one, truncated its remote log and overwrote the fetched copy.
  # With a local handle: RUNNING re-attaches (the wait below continues it);
  # DONE(0) is reused when THIS stage is the one the master took over (its job
  # finished while nobody watched); anything else, and a deliberate --redo of a
  # passed stage, submits a fresh job as before.
  local name="$1" cmd="$2" deadline="$3" prior="" dl_at left
  JOB_WAIT_S="$deadline"
  if ! plan_only && [ -f "$JOBS/$name.json" ]; then prior="$(bash "$PODJOB" status "$name" 2>/dev/null || true)"; fi
  case "$prior" in
    RUNNING)
      # review 2026-10-17 batch 4 (M3): the wait that follows covers the time the
      # handle's own deadline leaves, never the whole bound again
      dl_at="$(sed -n 's/.*"deadline_at": "\([0-9TZ:-]*\)".*/\1/p' "$JOBS/$name.json" | head -1)"
      left="$("$PY3" -c 'import sys, datetime; d = datetime.datetime.strptime(sys.argv[1], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc); print(max(60, int((d - datetime.datetime.now(datetime.timezone.utc)).total_seconds())))' "$dl_at" 2>/dev/null || echo "")"
      [ -z "$left" ] || JOB_WAIT_S="$left"
      say "job $name is RUNNING on the pod from the interrupted attempt: re-attaching, no second submit; waiting ${JOB_WAIT_S}s$( [ -n "$left" ] && printf ' (the handle deadline %s)' "$dl_at" || printf ' (the full bound: handle deadline unreadable)') (G5)"
      state note "$CUR_STAGE" "job $name re-attached while RUNNING ($(utc), G5); wait ${JOB_WAIT_S}s"
      return 0 ;;
    UNREACHABLE)
      # review 2026-10-17 batch 4 (M4): a blinking ssh endpoint must not get a
      # second submit beside a job that may still be running
      printf '  [FAIL] job %s has a local handle but the pod is UNREACHABLE: its state cannot be read, so nothing is submitted; retry when ssh answers (G5)\n' "$name"
      return 1 ;;
    DONE*|FAILED*|KILLED|CRASHED)
      # S1 audit 2026-10-23 (HIGH 2): a run that ended with exit 4 (budget stop)
      # or 2 (stop failed) while the master was gone used to be resubmitted here,
      # which truncated its log on the pod and refused the reseal; any finished
      # job of the taken-over stage is reused, and the wait below judges its exit
      # review 2026-10-23 M5: a handle older than the interrupted attempt (a stage
      # that failed earlier and was redone) is not that attempt's job: resubmitted
      local sub_at; sub_at="$(sed -n 's/.*"submitted_at": "\([0-9TZ:-]*\)".*/\1/p' "$JOBS/$name.json" | head -1)"
      if [ "$CUR_STAGE" = "$TAKEOVER_STAGE" ] && { [ -z "$TAKEOVER_STARTED" ] || [ -z "$sub_at" ] || [ "$sub_at" \> "$TAKEOVER_STARTED" ] || [ "$sub_at" = "$TAKEOVER_STARTED" ]; }; then
        say "job $name finished $prior during the interrupted attempt: its log is fetched, nothing is resubmitted (G5); the wait judges that exit"
        state note "$CUR_STAGE" "job $name $prior from the interrupted attempt reused ($(utc), G5)"
        return 0
      fi
      [ "$CUR_STAGE" != "$TAKEOVER_STAGE" ] || say "job $name's handle (submitted $sub_at) predates the interrupted attempt ($TAKEOVER_STARTED): a fresh job is submitted (review 2026-10-23 M5)" ;;
  esac
  run_step 0 "job $name: submit" -- bash "$PODJOB" submit "$name" "$cmd" "$deadline"
}
job_run() {
  # job_run <name> <deadline_s> <remote command> [logdir]: submit, wait, fetch
  # the log into logdir; the job's DONE(0) is the step's expected outcome.
  # A failed attempt's log is fetched too, under its own name, and the next
  # submit starts a fresh remote log (review 2026-10-06, MEDIUM 8), so a
  # marker from an earlier attempt can never satisfy this one's check.
  local name="$1" deadline="$2" cmd="$3" logdir="${4:-$LAND/logs/runner}"
  job_attach_or_submit "$name" "$cmd" "$deadline" || return 1
  if ! run_step 0 "job $name: wait" -- bash "$PODJOB" wait "$name" "$JOB_WAIT_S"; then
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

json_only_ours() {
  # json_only_ours <file> [id]...: 0 when every resource the listing holds is
  # one of the given ids (an empty listing passes; no ids = json_empty).
  # S0F-36: a stage 0 redo mid-run (a code fix that must re-ship) meets this
  # landing's own pod and volume; anything else is still a clean-room refusal.
  "$PY3" - "$@" <<'PY'
import json, sys
try:
    doc = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    sys.exit(2)
ours = set(sys.argv[2:])
items = doc if isinstance(doc, list) else []
if isinstance(doc, dict):
    for v in doc.values():
        if isinstance(v, list):
            items = items + v
ids = [str(i.get("id", "")) for i in items if isinstance(i, dict)]
sys.exit(0 if all(i in ours for i in ids) and len(ids) == len(items) else 1)
PY
}

engine_launcher() { case "$1" in vllm) echo manage_vllm_server.sh ;; sglang) echo manage_sglang_server.sh ;; lmdeploy) echo manage_lmdeploy_server.sh ;; *) return 1 ;; esac; }
engine_api()      { case "$1" in vllm) echo http://localhost:8000 ;; sglang) echo http://localhost:30000 ;; lmdeploy) echo http://localhost:23333 ;; *) return 1 ;; esac; }
server_engines()  { local e; for e in $ENGINES; do [ "$e" = "hf" ] || echo "$e"; done; }

# env pins run_campaign.py refuses on presence or on a mismatch (drift sweep 2026-10-06, 8e)
RUN_ENV_UNSET="-u CAGE_SLO_FLOORS_JSON -u CAGE_BUDGET_PLAN_JSON -u CAGE_KV_CEILING -u CAGE_TELEMETRY_ENDPOINTS -u CAGE_ALLOW_STALE_INDEX -u CAGE_SGLANG_API_BASE -u CAGE_LMDEPLOY_API_BASE -u VLLM_PORT -u SGLANG_PORT -u CAGE_PD_PROXY_PORT -u CAGE_PD_PREFILL_PORT -u CAGE_PD_DECODE_PORT"

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
  printf '  resource: network volume   id: %s   (KEPT: never deleted by a script, owner rule 2026-10-22)\n' "$VOL_ID"
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
  # Audit 2026-10-17 (G8): the A2 conductor pod, once recorded as state
  # conductor.id, is this landing's own; the API key stays on the Mac and only
  # the stage loop moves to the conductor (owner, "follow recommendations").
  plan_only || { grep -v '^#' "$STEP_LOG" > "$EXTRAS/ops/pods_before.json"; json_only_ours "$EXTRAS/ops/pods_before.json" $(state get pod.id 2>/dev/null || true) $(state get conductor.id 2>/dev/null || true) || { printf '  [FAIL] clean room: pods exist before this experiment (see %s; this landing tolerates only its own pod and its recorded conductor, G8)\n' "$EXTRAS/ops/pods_before.json"; return 1; }; }
  run_step 0 "clean room: network-volume list" -- runpodctl network-volume list || return 1
  # Owner rule 2026-10-22 (ADR-0176 Group B): no script deletes a network volume,
  # so a volume kept from an earlier run is expected here: it is named and its
  # storage bill noted, never a refusal. Pods stay strict.
  plan_only || { grep -v '^#' "$STEP_LOG" > "$EXTRAS/ops/volumes_before.json"; json_only_ours "$EXTRAS/ops/volumes_before.json" $(state get pod.volume_id 2>/dev/null || true) || { say "NOTE: network volume(s) exist before this experiment (see $EXTRAS/ops/volumes_before.json): kept by the owner rule of 2026-10-22, billing storage until deleted by hand"; state note preflight-mac "network volume(s) listed before this experiment: kept, billing storage (owner rule 2026-10-22)"; }; }
  if [ "${PARTIAL_TREE_OK:-0}" = "1" ] && [ -z "${REHEARSAL_N:-}" ]; then
    printf '  [FAIL] PARTIAL_TREE_OK=1 on a registered run (REHEARSAL_N empty): a partial tree is never sealed on a registered run (decision 3 of 2026-10-13, S0 only); unset it or set REHEARSAL_N\n'
    return 1
  fi
  # Audit 2026-10-17 (G14, G15): a registered profile (REHEARSAL_N empty) must
  # carry the hang guards and the keys A5 names, and none of the S0-only keys;
  # stage 0 refused PARTIAL_TREE_OK alone and let the rest pass silently.
  if [ -z "${REHEARSAL_N:-}" ]; then
    local reg_bad="" m
    [ -n "${STEP_TIMEOUT_S:-}" ] || reg_bad="$reg_bad STEP_TIMEOUT_S=empty(G14: a hung step bills to the HOURS bound)"
    [ "${LADDER_FAIL_FAST:-0}" = "1" ] || reg_bad="$reg_bad LADDER_FAIL_FAST=${LADDER_FAIL_FAST:-0}(G14: must be 1)"
    [ "${FRONT_LOAD:-0}" = "0" ] || reg_bad="$reg_bad FRONT_LOAD=${FRONT_LOAD}(S0 key)"
    [ -z "${RUN_FAIL_FAST:-}" ] || reg_bad="$reg_bad RUN_FAIL_FAST=${RUN_FAIL_FAST}(S0 key)"
    [ "${DRY_TELEMETRY_RERUN:-0}" = "0" ] || reg_bad="$reg_bad DRY_TELEMETRY_RERUN=${DRY_TELEMETRY_RERUN}(S0 key)"
    [ "${MAC_SUITE:-1}" = "1" ] || reg_bad="$reg_bad MAC_SUITE=${MAC_SUITE}(ADR-0145: never 0 on a registered run)"
    case "${MAX_NULL_FRACTION:-}" in 1|1.0|1.00) reg_bad="$reg_bad MAX_NULL_FRACTION=${MAX_NULL_FRACTION}(the dry-run value; the registered bound must be stated)" ;; esac
    [ -n "${BACKUP_ARCHIVE_DIR:-}" ] || reg_bad="$reg_bad BACKUP_ARCHIVE_DIR=empty(no tar archive of the run tree)"
    [ "${PIN_MODEL_REVISION:-0}" != "1" ] || [ -n "$MODEL_REVISION" ] || reg_bad="$reg_bad PIN_MODEL_REVISION=1-but-no-served_model-revision-for-$MODEL(A10)"
    [ "${CELL_STALL_MIN:-30}" -ge 60 ] || reg_bad="$reg_bad CELL_STALL_MIN=${CELL_STALL_MIN:-30}(S0 needed 60)"
    for m in $MODEL $PREFETCH_MODELS_REQUIRED; do
      case " $PREFETCH_MODELS " in *" $m "*) ;; *) reg_bad="$reg_bad PREFETCH_MODELS-lacks:$m" ;; esac
    done
    if [ -n "$reg_bad" ]; then
      printf '  [FAIL] registered profile (REHEARSAL_N empty) carries S0-only or unset keys (G14, G15, 2026-10-17):%s\n' "$reg_bad"
      return 1
    fi
  fi
  # The Mac suite is the free gate. MAC_SUITE=0 skips that one step loudly when
  # the profile vouches for HEAD and says why (SUITE_ON_POD=1 keeps the suite
  # on the pod); every other stage 0 check stays (owner, 2026-10-07, ADR-0145).
  if [ "${MAC_SUITE:-1}" = "0" ]; then
    say "Mac test suite SKIPPED (MAC_SUITE=0 in the profile; SUITE_ON_POD=${SUITE_ON_POD:-0} on the pod is the gate)"
    plan_only || state note preflight-mac "Mac test suite skipped (MAC_SUITE=0 in the profile)"
  else
    run_step 0 "Mac test suite" -- bash "$SCRIPTS/checks/run_tests.sh" -q -p no:cacheprovider || return 1
    plan_only || cp "$STEP_LOG" "$LAND/logs/setup/suite_mac.log"
  fi
  # P-1 (S0F-88, owner ruling R17): the live-load gate on the Mac, always (it is
  # one test file; MAC_SUITE=0 does not skip it). The real client must serve
  # 2,000+ requests in flight after raising a forced 1,024 open-file limit; at
  # 1,024 it fails, as S0 attempt 4's client did (B-1).
  run_step 0 "live-load gate (P-1)" -- env CAGE_LOAD_GATE=1 bash "$SCRIPTS/checks/run_tests.sh" -q -p no:cacheprovider tests/test_load_gate_p1.py || return 1
  if ! plan_only && { ! step_has "passed" || step_has "skipped"; }; then
    printf '  [FAIL] the live-load gate did not run and pass (P-1); see %s\n' "$STEP_LOG"; return 1
  fi
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
  # Gap triage 2026-10-09: every registration check stage 6 would refuse on
  # (anchor shape ADR-0155, datasets staged, manifest coverage per row class,
  # the rehearsal derivation) runs HERE, before anything bills; the probe
  # also says whether an executable vLLM cell needs the LMCache module, which
  # stage 4 then proves importable on the pod (S0F-57).
  run_step 0 "session '$SESSION' registration preflight (anchor shape, datasets staged, manifest coverage, pool vs max_model_len, pool vs GPU memory after weights)" -- "$PY" - "$SCRIPTS/3_run/run_campaign.py" "$SESSION" "$FLOOR_AVG_SEQ_TOKENS" "${REHEARSAL_N:-}" "$QUERY_MANIFESTS" "$CHARTER_DATASETS" "$PROJECT_DIR" "$FLOOR_CONCURRENCY" "$GPU_ID" <<'PY' || return 1
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("cage_run_campaign_probe", sys.argv[1])
m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m
spec.loader.exec_module(m)
session, floor_tokens, rehearsal, manifests_raw, datasets_raw, project, concurrency, gpu_id = sys.argv[2:10]
manifests = {}
for item in manifests_raw.split():
    ds, _sep, rel = item.partition("=")
    manifests[ds] = str(Path(project) / rel)
try:
    report = m.preflight_registration_check(
        session, floor_avg_seq_tokens=floor_tokens,
        rehearsal_n=int(rehearsal) if rehearsal.strip() else None,
        query_manifests=manifests, charter_datasets=tuple(datasets_raw.split()),
        floor_concurrency=int(concurrency), gpu_id=gpu_id)
except Exception as exc:
    print(f"REFUSED: session {session!r}: {exc}"); sys.exit(1)
print(f"session {session!r}: anchor shape {report['anchor_seq_tokens']} served tokens; {report['cells']} cells, "
      f"{report['executable_cells']} executable; NEEDS_LMCACHE={1 if report['needs_lmcache'] else 0}; "
      f"LMCACHE_STORE_GIB={report.get('lmcache_store_gib') or 0}")
# A11 item 4: the largest planned pool per engine, which stage 4 starts and checks
for key, tag in (("largest_pools", "LARGEST_POOL"), ("largest_helper_pools", "LARGEST_HELPER_POOL")):
    for e, p in sorted((report.get(key) or {}).items()):
        env = ",".join(f"{k}={v}" for k, v in sorted(p["env"].items()))
        print(f"{tag} engine={e} planned_tokens={p['planned_tokens']} planned_bytes={p['planned_bytes']} env={env}")
for p in report["problems"]:
    print(f"REFUSED: {p}")
sys.exit(1 if report["problems"] else 0)
PY
  plan_only || { if step_has "NEEDS_LMCACHE=1"; then state put registration.needs_lmcache 1; else state put registration.needs_lmcache 0; fi; }
  # S0F-88 U2: the largest LMCache CPU store the plan will size (GiB, 0 = none)
  plan_only || state put registration.lmcache_store_gib "$(grep -o 'LMCACHE_STORE_GIB=[0-9]*' "$STEP_LOG" | tail -1 | cut -d= -f2)"
  # A11 item 4 (owner 2026-10-21): stage 4 starts each engine at its largest planned pool
  if ! plan_only; then
    local lp_eng lp_line lp_tag lp_key
    # (review 2026-10-21 H1: the largest pool a GPU-helper cell plans rides the same record)
    for lp_tag in LARGEST_POOL:largest_pool LARGEST_HELPER_POOL:largest_helper_pool; do
      lp_key="${lp_tag#*:}"
      for lp_eng in vllm sglang; do
        lp_line="$(grep "^${lp_tag%%:*} engine=$lp_eng " "$STEP_LOG" | tail -1 || true)"
        state put "registration.$lp_key.$lp_eng.planned_tokens" "$(printf '%s' "$lp_line" | sed -n 's/.* planned_tokens=\([0-9]*\).*/\1/p')"
        state put "registration.$lp_key.$lp_eng.planned_bytes" "$(printf '%s' "$lp_line" | sed -n 's/.* planned_bytes=\([0-9]*\).*/\1/p')"
        state put "registration.$lp_key.$lp_eng.env" "$(printf '%s' "$lp_line" | sed -n 's/.* env=\([^ ]*\).*/\1/p')"
      done
    done
  fi
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
    # Audit 2026-10-17 (G17): the balance must cover the whole pod bound at the
    # list price plus the floor; the account is PREPAID and RunPod stops the pod
    # at zero (its behavior at that instant is [?]), so the shortfall is judged
    # here, before anything bills. The price is the CLI's for CLOUD_TYPE;
    # PRICE_PER_HOUR_USD in the profile overrides it when the CLI lists none.
    local price need_usd balance
    price="${PRICE_PER_HOUR_USD:-}"
    [ -n "$price" ] || price="$("$PY3" - "$EXTRAS/ops/gpu_list.json" "$GPU_ID" "$CLOUD_TYPE" <<'PY'
import json, sys
gpus = json.load(open(sys.argv[1], encoding="utf-8"))
want, cloud = sys.argv[2], sys.argv[3].upper()
key = "securePricePerHr" if cloud == "SECURE" else "communityPricePerHr"
for g in gpus if isinstance(gpus, list) else gpus.get("gpus", []):
    if g.get("gpuId") == want or g.get("id") == want:
        p = g.get(key)
        if isinstance(p, (int, float)) and p > 0:
            print(p)
        sys.exit(0)
PY
)"
    balance="$("$PY3" - "$EXTRAS/ops/user_before.txt" <<'PY'
import json, sys
try:
    # the file is the step log copy: the master's "# ..." header and footer lines wrap the CLI's JSON
    body = "\n".join(l for l in open(sys.argv[1], encoding="utf-8").read().splitlines() if not l.startswith("#"))
    doc = json.loads(body)
except Exception:
    sys.exit(0)
for k in ("clientBalance", "balance", "currentBalance"):
    if isinstance(doc, dict) and isinstance(doc.get(k), (int, float)):
        print(doc[k]); break
PY
)"
    [ -n "$price" ] || { printf '  [FAIL] no %s price for %s in runpodctl gpu list and PRICE_PER_HOUR_USD is unset: the balance cannot be judged against the pod bound (G17)\n' "$CLOUD_TYPE" "$GPU_ID"; return 1; }
    [ -n "$balance" ] || { printf '  [FAIL] runpodctl user printed no balance field (clientBalance): the balance cannot be judged (G17); see %s\n' "$EXTRAS/ops/user_before.txt"; return 1; }
    need_usd="$(awk -v h="$(pod_hours_bound)" -v p="$price" -v g="$GPU_COUNT" -v f="$BALANCE_FLOOR_USD" 'BEGIN { printf "%.2f", h * p * g + f }')"
    state put cost.price_per_hour_list_usd "$price"; state put cost.balance_before_usd "$balance"; state put cost.pod_bound_need_usd "$need_usd"
    say "balance $balance USD against $need_usd USD needed ($(pod_hours_bound) h x $price USD/h x $GPU_COUNT GPU + the $BALANCE_FLOOR_USD USD floor)"
    run_step 0 "balance covers the pod bound plus the floor (G17)" -- awk -v b="$balance" -v n="$need_usd" 'BEGIN { exit (b + 0 >= n + 0) ? 0 : 1 }' || { printf '  [FAIL] balance %s USD is below the %s USD the pod bound needs (%s h x %s USD/h x %s GPU + %s USD floor): top up or cut HOURS (G17)\n' "$balance" "$need_usd" "$(pod_hours_bound)" "$price" "$GPU_COUNT" "$BALANCE_FLOOR_USD"; return 1; }
  fi
  # The seatbelt deletes the pod at its deadline whatever stage is running
  # (review 2026-10-06, HIGH 4): it must cover every pod-side bound, not the run alone.
  local seat_min need_min
  seat_min="$(seatbelt_minutes "$SEATBELT")" || { printf '  [FAIL] SEATBELT=%s is not <n>h or <n>m (a day form is refused: provision_pod.sh accepts none, G30)\n' "$SEATBELT"; return 1; }
  need_min="$(seatbelt_need_minutes)"
  say "seatbelt $SEATBELT = $seat_min min; the stage bounds need $need_min min (readiness $((POD_READY_TIMEOUT_S / 60)), setup $SETUP_BOUND_MIN, validate $(stage_bound_minutes validate), calibrate $(stage_bound_minutes calibrate) with calibrate-rungs $((CALIBRATE_RUNGS_BOUND_S / 60)) per engine, plan, run $((HOURS * 60)), seal, score $SCORE_BOUND_MIN, collect, pull margin $PULL_MARGIN_MIN)"
  run_step 0 "seatbelt covers the stage bounds" -- test "$seat_min" -ge "$need_min" || { printf '  [FAIL] SEATBELT=%s (%s min) is shorter than the %s min the stages need; raise SEATBELT in the profile\n' "$SEATBELT" "$seat_min" "$need_min"; return 1; }
  run_step 0 "provision plan print" -- bash "$SCRIPTS/runpod/provision_pod.sh" --gpu-id "$GPU_ID" --gpu-count "$GPU_COUNT" --disk-gb "$DISK_GB" --volume-gb "$VOLUME_GB" --cloud-type "$CLOUD_TYPE" --ports "$PORTS" --terminate-after "$SEATBELT" --hours "$(pod_hours_bound)" --purpose "$EXP $DATE" || return 1
  plan_only || step_has "PLAN ONLY" || { printf '  [FAIL] provision plan print lacks the PLAN ONLY line\n'; return 1; }
  if ! plan_only; then
    # S0F-36: a stage 0 redo (a code fix re-shipped mid-run) keeps the run id;
    # the campaign root on the pod never moves under a running landing.
    local rid; rid="$(state get run_id 2>/dev/null || true)"
    if [ -n "$rid" ]; then
      say "run id kept: $rid (a stage 0 redo never re-mints the run)"
    else
      rid="$(mint_campaign_run_id "$SESSION" "$MODEL_SLUG")"
      printf '%s' "$rid" | grep -qE '^[a-z0-9][a-z0-9-]{2,40}$' || { printf '  [FAIL] minted run id %s violates the run_id grammar\n' "$rid"; return 1; }
      state put run_id "$rid"
      say "run id minted: $rid (campaign root on the pod: $POD_REPO/results/$CAMPAIGN/$SESSION/$rid)"
    fi
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
  # Owner rule 2026-10-22 (ADR-0176 Group B): a volume recorded by an earlier
  # attempt whose pod was refused is reused in its datacenter; no second one.
  local kept_vol="" kept_dc=""
  if ! plan_only && [ "$created" -eq 0 ]; then
    kept_vol="$(state get pod.volume_id 2>/dev/null || true)"; kept_dc="$(state get pod.dc 2>/dev/null || true)"
    if [ -n "$kept_vol" ] && [ -n "$kept_dc" ]; then
      # review 2026-10-22 M1: a recorded volume the owner deleted by hand is not reused
      run_step 0 "recorded volume $kept_vol listed" -- runpodctl network-volume list || return 1
      if grep -v '^#' "$STEP_LOG" | grep -q "\"$kept_vol\""; then
        say "volume $kept_vol in $kept_dc is already recorded (kept after a refused pod): reused, no second volume is created"
        prefs=" $kept_dc"
      else
        say "recorded volume $kept_vol is no longer listed (deleted by hand): the record is cleared and a new volume is created in DC_PREFS order"
        state put pod.volume_id ""; state put pod.dc ""; kept_vol=""
      fi
    else
      kept_vol=""
    fi
  fi
  plan_only || [ "$created" -eq 1 ] || state go provision "--yes provision on the command line ($(utc))"
  [ "$created" -eq 1 ] || for dc in $prefs; do
    if [ -n "$kept_vol" ]; then vol="$kept_vol"; else
      run_step 0 "network volume create in $dc" -- runpodctl network-volume create --name "cage-$EXP-$DATE" --size "$VOLUME_GB" --data-center-id "$dc" || continue
      if plan_only; then vol="<volume_id>"; else
        vol="$(grep -v '^#' "$STEP_LOG" | sed -nE 's/.*"id"[[:space:]]*:[[:space:]]*"([A-Za-z0-9-]+)".*/\1/p' | head -1)"
        [ -n "$vol" ] || { printf '  [FAIL] volume created in %s but its id was not parsed; find it with runpodctl network-volume list\n' "$dc"; return 1; }
        state put pod.volume_id "$vol"; state put pod.dc "$dc"
      fi
    fi
    # review 2026-10-13 item 24: CLOUD_TYPE and PORTS are the profile's, not the provisioner's defaults
    run_step 0 "provision pod in $dc (--yes)" -- bash "$SCRIPTS/runpod/provision_pod.sh" --gpu-id "$GPU_ID" --gpu-count "$GPU_COUNT" --disk-gb "$DISK_GB" --data-center-ids "$dc" --network-volume-id "$vol" --cloud-type "$CLOUD_TYPE" --ports "$PORTS" --terminate-after "$SEATBELT" --hours "$(pod_hours_bound)" --purpose "$EXP $DATE" --yes
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
    # Owner rule 2026-10-22: the volume is never deleted; the loop stops here and
    # the resume reuses the recorded volume in this datacenter.
    printf '  [FAIL] no pod in %s: volume %s is KEPT there (never deleted by a script; it bills storage); resume with --from provision --yes provision to try %s again, or delete the volume by hand (the resume then creates a new one in DC_PREFS order)\n' "$dc" "$vol" "$dc"
    return 1
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
      mem="$(printf "%s" "$out" | sed -nE "s/.*\"memoryInGb\"[[:space:]]*:[[:space:]]*([0-9]+).*/\1/p" | head -1)"
      echo "runtimeStatus=${status:-unknown} memoryInGb=${mem:-unknown}"
      [ "$status" = "running" ] && exit 0
      [ "$(date +%s)" -ge "$deadline" ] && { echo "pod not running after '"$POD_READY_TIMEOUT_S"'s"; exit 1; }
      sleep '"$POD_READY_POLL_S"'
    done' || return 1
  # review 2026-10-13 item 2: the pod's allocated memory is the stage 4 memory
  # probe's bound (inside the container /proc/meminfo shows the host)
  plan_only || state put pod.memory_gb "$(step_out | grep -o 'memoryInGb=[0-9]*' | tail -1 | cut -d= -f2)"
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
  # P-2 (S0F-88): ship sends the tarball stage 0 built. On 2026-10-11 a fix was
  # committed after stage 0 and `--only ship --redo` shipped the old code
  # (4df4da9 instead of cef4cde); only the sha line in the log showed it.
  if ! plan_only; then
    local built head; built="$(state get build.sha 2>/dev/null || true)"
    head="$(git -C "$PROJECT_DIR" rev-parse HEAD 2>/dev/null || true)"
    if [ -z "$head" ]; then
      say "no git HEAD readable at $PROJECT_DIR: the tarball-vs-HEAD check is skipped"
    elif [ "$head" != "$built" ]; then
      printf '  [FAIL] the tarball stage 0 built carries %s but the Mac HEAD is %s (P-2); repackage, then ship:\n    bash %s %s --date %s --only preflight-mac --redo\n' "${built:-none}" "$head" "$0" "$EXP" "$DATE"
      return 1
    fi
  fi
  run_step 0 "scp tarball" -- pscp_to "$tarball" "$POD_TARBALL" || return 1
  local relink=""
  if [ -n "$POD_RESULTS_DIR" ]; then
    relink=" && mkdir -p $POD_RESULTS_DIR && { [ -L $POD_REPO/results ] || { mkdir -p $POD_REPO/results && cp -R $POD_REPO/results/. $POD_RESULTS_DIR/ && rm -rf $POD_REPO/results && ln -s $POD_RESULTS_DIR $POD_REPO/results; }; } && echo results_root=\$(readlink -f $POD_REPO/results)"
    say "results root on the pod: $POD_RESULTS_DIR, reached as $POD_REPO/results (S0F-39)"
  fi
  run_step 0 "unpack on the pod" -- pssh "mkdir -p $POD_REPO && ln -sfn $POD_REPO ~/CAGE && tar xzf $POD_TARBALL --no-same-owner -C $POD_REPO && cat $POD_REPO/BUILD_INFO$relink && mkdir -p $POD_BACKUP_DIR $POD_REPO/MyDocs/registration $POD_REPO/results/calibration" || return 1
  if ! plan_only && [ -n "$POD_RESULTS_DIR" ]; then
    step_has "results_root=$POD_RESULTS_DIR" || { printf '  [FAIL] %s/results does not resolve to %s\n' "$POD_REPO" "$POD_RESULTS_DIR"; return 1; }
  fi
  if ! plan_only; then
    local sha; sha="$(state get build.sha)"
    step_has "sha=$sha" || { printf '  [FAIL] BUILD_INFO on the pod does not carry the Mac HEAD %s\n' "$sha"; return 1; }
    step_has "dirty=0" || { printf '  [FAIL] BUILD_INFO on the pod is not dirty=0\n'; return 1; }
  fi
  # S0F-39: cell directories are row keys with "|"; a results root that refuses
  # the character (the CA-MTL-1 volume, NFS4 on a vast backend) fails here, at
  # stage 2, not at the first cell of stage 7. The probe runs on the EFFECTIVE
  # results root (the container disk when POD_RESULTS_DIR is set).
  run_step 0 "results root accepts | in names" -- pssh "d=$POD_REPO/results/.cage_name_probe_\$\$; mkdir -p \"\$d\" && if mkdir \"\$d/a|b\" 2>/dev/null; then rm -rf \"\$d\"; echo NAME_PROBE_OK; else rm -rf \"\$d\"; echo NAME_PROBE_REFUSED; fi" || return 1
  plan_only || step_has "NAME_PROBE_OK" || { printf '  [FAIL] the results root refuses the character | in file names (cell directories are row keys, S0F-39): set POD_RESULTS_DIR to a container-disk path in the profile, or choose a datacenter whose volume accepts it\n'; return 1; }
  run_step 0 "scp freeze file" -- pscp_to "$FREEZE_FILE" "$POD_REPO/$POD_FREEZE_FILE" || return 1
  return 0
}

stage_setup() {
  pod_env
  local bound=$(( SETUP_BOUND_MIN * 60 ))
  # S0F-57 / ADR-0158: INSTALL_LMCACHE=1 (profile) installs the B8 connector package in
  # the bootstrap, FATAL there when it fails; the line below proves it landed.
  job_run setup "$bound" "cd $POD_REPO && CHARTER_DATASETS='$CHARTER_DATASETS' PREFETCH_MODELS='$PREFETCH_MODELS' PREFETCH_REVISIONS='$PREFETCH_REVISIONS' CAGE_HUB_CHECK_MODEL='${MODEL_REVISION:+$MODEL=$MODEL_REVISION}' INSTALL_LMCACHE='${INSTALL_LMCACHE:-0}' LMCACHE_VERSION='${LMCACHE_VERSION:-}' bash scripts/runpod/setup_runpod.sh" "$LAND/logs/setup" || return 1
  plan_only && return 0
  local log="$JOB_LOG" m bad=0
  grep -q "RunPod bootstrap complete" "$log" || { printf '  [FAIL] setup: no "RunPod bootstrap complete" line\n'; bad=1; }
  if [ "${INSTALL_LMCACHE:-0}" = "1" ]; then
    grep -q "installed beside vllm" "$log" || { printf '  [FAIL] setup: INSTALL_LMCACHE=1 but the bootstrap printed no "lmcache: ... installed beside vllm" line (S0F-57)\n'; bad=1; }
  fi
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
    # A stage 4 retry on the same shipped code does not rerun the pod suite
    # (2026-10-07: six retries reran the 7-minute suite every time); the sha it
    # passed on is recorded, and a re-ship (a new build.sha) runs it again.
    local suite_sha build_sha
    suite_sha="$(state get suite_pod.sha 2>/dev/null || true)"; build_sha="$(state get build.sha 2>/dev/null || true)"
    if ! plan_only && [ -n "$build_sha" ] && [ "$suite_sha" = "$build_sha" ]; then
      say "pod suite already passed on sha $build_sha: skipped on this retry"
    else
      job_run suite_pod 1800 "cd $POD_REPO && env $RUN_ENV_UNSET $POD_PYTHON -m pytest -q -p no:cacheprovider" "$LAND/logs/setup" || return 1
      plan_only || state put suite_pod.sha "$build_sha"
    fi
  fi
  # P-1 (S0F-88, R17): the live-load gate once more on the pod, where the
  # Linux limits and the pod's own Python run the client (the suite above
  # skips it: CAGE_LOAD_GATE is unset there).
  job_run load_gate 600 "cd $POD_REPO && env $RUN_ENV_UNSET CAGE_LOAD_GATE=1 $POD_PYTHON -m pytest -q -s -p no:cacheprovider tests/test_load_gate_p1.py" "$LAND/logs/setup" || return 1
  if ! plan_only && { ! grep -q " passed" "$JOB_LOG" || grep -q "skipped" "$JOB_LOG"; }; then
    printf '  [FAIL] the pod live-load gate did not run and pass (P-1); see %s\n' "$JOB_LOG"; return 1
  fi
  # Preflight gate (p) judges the datasets named by CAGE_DATASETS (comma list)
  # and the FULL charter roster when it is unset (live 2026-10-07: it refused
  # hotpotqa, scbench and sharegpt, which this profile never stages); the
  # experiment's own CHARTER_DATASETS, the set stage 3 staged, is the request.
  local ds; ds="$(printf '%s' "$CHARTER_DATASETS" | tr ' ' ',')"
  # S1 audit 2026-10-23 (MEDIUM 4): the run job loads every dataset offline
  # (HF_HUB_OFFLINE=1 rides POD_ENGINE_ENV under A10) and S0 ran online, so the
  # loader opens each dataset once here, under the run job's own env, before any
  # engine bills; the probe refuses an online pass when the pin is on.
  local req_off=""; [ -n "$MODEL_REVISION" ] && req_off=" --require-offline"
  job_run datasets_offline 600 "cd $POD_REPO && export PATH=$POD_VENV_BIN:\$PATH$POD_ENGINE_ENV && env $RUN_ENV_UNSET $POD_PYTHON scripts/checks/probe_datasets_offline.py --datasets $ds --max-examples 3$req_off --out logs/datasets_offline_probe.json" "$LAND/logs/setup" || return 1
  if ! plan_only; then
    grep -q "DATASETS_OFFLINE_OK" "$JOB_LOG" || { printf '  [FAIL] a charter dataset does not load through the runner loader under the run job env (S1 audit 2026-10-23):\n'; grep "DATASETS_OFFLINE_FAILED" "$JOB_LOG" | sed 's/^/    /'; return 1; }
    grep "DATASETS_OFFLINE_OK" "$JOB_LOG" | tail -1 | sed 's/^/  /'
  fi
  local e launcher api
  for e in $(server_engines); do
    launcher="$(engine_launcher "$e")" || { printf '  [FAIL] unknown engine %s\n' "$e"; return 1; }
    api="$(engine_api "$e")"
    # S0F-37 (live 2026-10-07): preflight_check.sh is vLLM's gate set (gate (a)
    # probes /health and /reset_prefix_cache, gate (o) vLLM's metric names), so
    # it runs against vLLM only; its engine-independent gates then stand for the
    # pod. Every other engine gets the engine-specific probes the campaign relies
    # on: the cold-start endpoint run_experiment.py calls (SGLang POST
    # /flush_cache), the live /metrics names the regime bridge and the cold-start
    # check read (the live scrape of 2026-10-07: sglang:token_usage,
    # sglang:num_retracted_reqs, sglang:num_running_reqs), and the launcher's
    # realized-pool record (ADR-0142). An engine with no registered set refuses.
    local probe="" names="" gauge="" n_names=0 thinking="" needs_lmcache store_gib
    needs_lmcache="$(state get registration.needs_lmcache 2>/dev/null || echo 0)"
    store_gib="$(state get registration.lmcache_store_gib 2>/dev/null || echo 0)"
    case "$e" in
      vllm) probe="CAGE_PREFLIGHT_BACKENDS=$e bash scripts/checks/preflight_check.sh '$MODEL' $api; rc=\$?"
            # S0F-57 (gap triage 2026-10-09, C1): the plan's vLLM retr-store (B8)
            # cells relaunch with the LMCache connector; on the 2026-10-08 landing
            # that relaunch died with ModuleNotFoundError after hours of billing.
            # When stage 0 found such a cell executable, the import is proven here.
            # (the marker alone: the check below fails the stage with the fix text)
            [ "$needs_lmcache" = "1" ] && probe="$probe; $POD_PYTHON -c 'import lmcache' && echo LMCACHE_IMPORT_OK || echo LMCACHE_IMPORT_MISSING"
            # S0F-88 U2 (review H1): the pod's OWN memory (its cgroup; /proc/meminfo
            # shows the host's) must hold the largest LMCache store, read while vLLM is up
            [ "$needs_lmcache" = "1" ] && probe="$probe; $POD_PYTHON scripts/checks/probe_pod_memory.py --need-gib ${store_gib:-0} --pod-gib $(state get pod.memory_gb 2>/dev/null || echo 0) --out logs/pod_memory_probe.json" ;;
      # sglang:num_queue_reqs is the queue gauge the section 6.1 clause (b)
      # regime label reads (ADR-0153; gap triage 2026-10-09 M3).
      # A7 (owner 2026-10-20): the cumulative retraction counter the dialect maps is
      # checked by its TYPE declaration below. S1 (live 2026-10-23): SGLang
      # 0.5.10.post1 creates the counter, declaration included, only at its first
      # retraction (multiprocess metrics; scheduler.py 2574-2575), so a fresh
      # server shows the gauge alone, which the cage-stats dialect reads as 0
      sglang) names="sglang:token_usage sglang:num_retracted_reqs sglang:num_running_reqs sglang:num_queue_reqs"; gauge="sglang:num_running_reqs" ;;
      *) printf '  [FAIL] %s: no validation probe set is registered for this engine (S0F-37)\n' "$e"; return 1 ;;
    esac
    if [ -z "$probe" ]; then
      n_names="$(printf '%s\n' $names | grep -c .)"
      # The flush refuses while a request runs (the startup warm-up still did at
      # the fifth attempt: 400 "pending requests ... #running-req: 1"); the
      # driver reads the running-requests gauge before every flush, so does this.
      # Live 2026-10-08 (S0F-54): the gauge read 0 while the warm-up's prefill
      # was still in flight (the refusal's own counts were 0/0: the forward sits
      # in the overlap result queue), so the POST is the deferred form SGLang
      # 0.5.10.post1 offers, ?timeout=20: performed once the scheduler is fully
      # idle, 400 past the deadline (fail-closed stays), the same query the
      # adapter sends; the launcher's start now also waits for SGLang's own
      # ready line. Quoted so the remote shell never globs the "?".
      probe="rc=0; for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do r=\$(curl -fsS $api/metrics 2>/dev/null | grep '^$gauge' | awk '{print \$NF}' | head -1); r=\${r%.*}; [ \"\${r:-1}\" = 0 ] && break; sleep 2; done; echo \"RUNNING_REQS=\${r:-unread}\"; curl -fsS -X POST \"$api/flush_cache?timeout=20\" >/dev/null && echo ENGINE_FLUSH_OK || { echo ENGINE_FLUSH_FAILED; rc=1; }; m=\$(curl -fsS $api/metrics 2>/dev/null); for n in $names; do printf '%s\\n' \"\$m\" | grep -q \"^\$n\" && echo \"METRIC_OK \$n\" || { echo \"METRIC_MISSING \$n\"; rc=1; }; done; [ -s logs/$e/CURRENT.kvpool.json ] && echo POOL_RECORD_OK || { echo POOL_RECORD_MISSING; rc=1; }"
      # S0F-88 U7: the server's own record that every row carries a cached-
      # token count (/server_info lists every launch argument at 0.5.10.post1)
      probe="$probe; curl -fsS $api/server_info 2>/dev/null | grep -Eq '\"enable_cache_report\": *true' && echo CACHE_REPORT_ON || { echo CACHE_REPORT_OFF; rc=1; }"
      probe="$probe; printf '%s\\n' \"\$m\" | grep -Eq '^# TYPE sglang:num_retracted_requests(_total)? counter' && echo RETRACTION_COUNTER_DECLARED || { printf '%s\\n' \"\$m\" | grep -Eq '^sglang:num_retracted_reqs[{ ]' && echo RETRACTION_COUNTER_LAZY || { echo RETRACTION_COUNTER_MISSING; rc=1; }; }"
    fi
    # S0F-59 (ADR-0151): one chat request built the way the campaign runner
    # builds every request (its own adapter construction, the chat messages,
    # stop "\n", T=0); fails when the served text carries the thinking marker
    # or the request fails. The 2026-10-08 landing served 2,450 of 2,450
    # SGLang rows as "<think>" in 2 tokens and no gate said a word.
    thinking="$POD_PYTHON scripts/checks/probe_thinking_pin.py --backend $e --api-base $api --model '$MODEL' --out logs/$e/CURRENT.thinking_probe.json || rc=1"
    # S0F-88 B-6 (R8): the reranker serves on the GPU; once, after the vLLM
    # engine stops (the GPU is free), the probe compares its served top k with
    # the CPU ranking on 50 registered questions.
    local rerank=""
    # (the separator rides the variable: an empty one must leave no bare ";")
    [ "$e" = "vllm" ] && rerank="; $POD_PYTHON scripts/checks/probe_reranker_device.py --out logs/rerank_device_probe.json || rc=1"
    # A14 (owner 2026-10-20): the compressor and the dense encoder on the GPU serve what the CPU served
    [ "$e" = "vllm" ] && rerank="$rerank; env$(instrument_rev_env) $POD_PYTHON scripts/checks/probe_compressor_device.py --freeze-file $POD_FREEZE_FILE --out logs/compress_device_probe.json || rc=1; $POD_PYTHON scripts/checks/probe_encoder_device.py --freeze-file $POD_FREEZE_FILE --out logs/encoder_device_probe.json || rc=1"
    # A11 item 4 (owner 2026-10-21): the start runs at the engine's largest planned
    # pool (stage 0), and the relaunch rule checks it, before calibration bills
    local lp_tokens lp_bytes lp_env="" lp_check=""
    lp_tokens="$(state get "registration.largest_pool.$e.planned_tokens" 2>/dev/null || true)"
    lp_bytes="$(state get "registration.largest_pool.$e.planned_bytes" 2>/dev/null || true)"
    if [ -n "$lp_tokens" ] && [ -n "$lp_bytes" ]; then
      lp_env=" $(state get "registration.largest_pool.$e.env" 2>/dev/null | tr ',' ' ')"
      lp_check="; $POD_PYTHON scripts/3_run/run_campaign.py pool-check --engine $e --planned-tokens $lp_tokens --planned-bytes $lp_bytes --out logs/$e/largest_pool_check.json || rc=1"
    fi
    # Review 2026-10-21 H1 (owner: "Add a check"): SGLang starts a second time at the
    # largest pool a GPU-helper cell plans, and the three helper probes run beside it
    local hp_tokens="" hp_bytes="" hp_env="" helper="" vbound=1500
    if [ "$e" = "sglang" ]; then
      hp_tokens="$(state get "registration.largest_helper_pool.$e.planned_tokens" 2>/dev/null || true)"
      hp_bytes="$(state get "registration.largest_helper_pool.$e.planned_bytes" 2>/dev/null || true)"
    fi
    if [ -n "$hp_tokens" ] && [ -n "$hp_bytes" ]; then
      hp_env=" $(state get "registration.largest_helper_pool.$e.env" 2>/dev/null | tr ',' ' ')"
      helper="; env$hp_env bash scripts/2_serving/$launcher start '$MODEL' && { $POD_PYTHON scripts/3_run/run_campaign.py pool-check --engine $e --planned-tokens $hp_tokens --planned-bytes $hp_bytes --out logs/$e/helper_pool_check.json && env$(instrument_rev_env) $POD_PYTHON scripts/checks/probe_reranker_device.py --out logs/rerank_device_probe_beside_sglang.json && env$(instrument_rev_env) $POD_PYTHON scripts/checks/probe_compressor_device.py --freeze-file $POD_FREEZE_FILE --out logs/compress_device_probe_beside_sglang.json && $POD_PYTHON scripts/checks/probe_encoder_device.py --freeze-file $POD_FREEZE_FILE --out logs/encoder_device_probe_beside_sglang.json && echo HELPERS_FIT_OK; } || { echo HELPERS_FIT_FAILED; rc=1; }; bash scripts/2_serving/$launcher stop"
      vbound=3000
    fi
    # after `stop` the GPU process takes a few seconds to leave the compute-apps list: wait up to 60 s before reading it
    job_run "validate_$e" "$vbound" "cd $POD_REPO && export PATH=$POD_VENV_BIN:\$PATH VLLM_START_TIMEOUT=$VLLM_START_TIMEOUT CAGE_DATASETS=$ds$POD_ENGINE_ENV && env$lp_env bash scripts/2_serving/$launcher start '$MODEL' && curl -sf $api/v1/models >/dev/null && echo VALIDATE_API_OK; $probe$lp_check; $thinking; bash scripts/2_serving/$launcher stop$rerank$helper; n=1; for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do n=\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true); [ \"\$n\" -eq 0 ] && break; sleep 2; done; echo COMPUTE_APPS=\$n; exit \$rc" "$LAND/logs/setup" || return 1
    plan_only && continue
    if [ -n "$lp_tokens" ]; then
      grep -q "refusing to serve a smaller pool than planned" "$JOB_LOG" && { printf '  [FAIL] %s: the launcher could not hold the largest planned pool (%s tokens) on this pod (A11 item 4): no calibration bills on a pod that cannot run the plan\n' "$e" "$lp_tokens"; return 1; }
      grep -q "LARGEST_POOL_OK" "$JOB_LOG" || { printf '  [FAIL] %s: the realized pool at the largest planned pool (%s tokens) is outside the 0.5%% rule, or unread (A11 item 4):\n' "$e" "$lp_tokens"; grep 'LARGEST_POOL_FAILED' "$JOB_LOG" | sed 's/^/    /'; return 1; }
      grep 'LARGEST_POOL_OK' "$JOB_LOG" | tail -1 | sed 's/^/  /'
    else
      printf '  %s: no budgeted single-GPU pool in the plan; the largest-pool start is skipped (A11 item 4)\n' "$e"
    fi
    grep -q "VALIDATE_API_OK" "$JOB_LOG" || { printf '  [FAIL] %s: /v1/models never answered\n' "$e"; return 1; }
    if [ "$e" = "vllm" ]; then
      grep -q "PREFLIGHT PASS" "$JOB_LOG" || { printf '  [FAIL] %s: preflight did not PASS\n' "$e"; return 1; }
      grep -q "RERANK_DEVICE_OK" "$JOB_LOG" || { printf '  [FAIL] the reranker does not serve on the GPU what it serves on the CPU, or the probe failed (S0F-88 B-6):\n'; grep 'RERANK_DEVICE_FAILED' "$JOB_LOG" | sed 's/^/    /'; return 1; }
      if [ "$needs_lmcache" = "1" ]; then
        grep -q "LMCACHE_IMPORT_OK" "$JOB_LOG" || { printf '  [FAIL] %s: the lmcache module does not import in the pod venv, but the plan has executable vLLM retr-store (B8) cells whose relaunch sets the LMCache connector (S0F-57): install lmcache on the pod (setup_runpod.sh) or deregister B8 for this session, before the calibration stages bill\n' "$e"; return 1; }
        # S0F-88 U2 (review H1): the pod must hold the largest LMCache CPU store
        # the plan sizes (LMCACHE_STORE_RULE); the probe line is printed for the record.
        { grep 'POD_MEMORY_' "$JOB_LOG" || true; } | tail -1 | sed 's/^/  /'
        grep -q "POD_MEMORY_OK" "$JOB_LOG" || { printf '  [FAIL] %s: the memory left to the pod container (its cgroup limit minus its usage, with vLLM up) is below the %s GiB LMCache CPU store the plan sizes for its largest retr-store window, or the probe failed (S0F-88 U2)\n' "$e" "${store_gib:-0}"; return 1; }
      fi
    else
      grep -q "ENGINE_FLUSH_OK" "$JOB_LOG" || { printf '  [FAIL] %s: the cold-start endpoint POST /flush_cache?timeout=20 did not answer 200 (the scheduler never went fully idle within 20 s, or the endpoint is gone)\n' "$e"; return 1; }
      [ "$(grep -c '^METRIC_OK ' "$JOB_LOG")" -eq "$n_names" ] || { printf '  [FAIL] %s: live /metrics lacks a name the regime bridge or the cold-start check reads:\n' "$e"; grep 'METRIC_MISSING' "$JOB_LOG" | sed 's/^/    /'; return 1; }
      grep -q "POOL_RECORD_OK" "$JOB_LOG" || { printf '  [FAIL] %s: the launcher wrote no CURRENT.kvpool.json (ADR-0142 realized pool)\n' "$e"; return 1; }
      grep -q "CACHE_REPORT_ON" "$JOB_LOG" || { printf '  [FAIL] %s: /server_info does not report enable_cache_report=true, so no row would carry a cached-token count (S0F-88 U7)\n' "$e"; return 1; }
      grep -Eq "RETRACTION_COUNTER_(DECLARED|LAZY)" "$JOB_LOG" || { printf '  [FAIL] %s: live /metrics shows neither the cumulative retraction counter sglang:num_retracted_requests_total nor the retraction gauge sglang:num_retracted_reqs that stands for it before the first retraction (no preemption telemetry: every window records scarcity as absent)\n' "$e"; return 1; }
      if grep -q "RETRACTION_COUNTER_LAZY" "$JOB_LOG"; then
        printf '  %s: no retraction yet, so the retraction counter is not created; the dialect reads the gauge as 0 until the first retraction (S1 2026-10-23)\n' "$e"
      fi
      if [ -n "$hp_tokens" ]; then
        grep -q "HELPERS_FIT_OK" "$JOB_LOG" || { printf '  [FAIL] %s: the GPU helpers (reranker, compressor, encoder) did not run beside SGLang at the largest helper-cell pool (%s tokens), or that pool did not hold (review 2026-10-21 H1):\n' "$e" "$hp_tokens"; grep -E 'LARGEST_POOL_FAILED|_DEVICE_FAILED|HELPERS_FIT_FAILED' "$JOB_LOG" | sed 's/^/    /'; return 1; }
        printf '  %s: the GPU helpers fit beside SGLang at the largest helper-cell pool (%s tokens)\n' "$e" "$hp_tokens"
      fi
    fi
    grep -q "THINKING_PIN_OK" "$JOB_LOG" || { printf '  [FAIL] %s: the thinking pin probe did not pass (S0F-59, ADR-0151): the served text carried the thinking marker or the probe request failed\n' "$e"; grep 'THINKING_PIN_FAILED' "$JOB_LOG" | sed 's/^/    /'; return 1; }
    grep -q "COMPUTE_APPS=0" "$JOB_LOG" || { printf '  [FAIL] %s: compute apps remain after stop\n' "$e"; return 1; }
  done
  # Review 2026-10-13 (Part 8 option A1, items 29 and 32): the serving
  # configurations the run stage would start for the first time 15 to 27
  # pod-hours in are started ONCE here, each with one served request: SGLang
  # at KV fp8_e4m3 (the launcher's realized-format gate runs), vLLM at KV fp8,
  # vLLM with the LMCache connector at the plan's store size (the engine log
  # must name the store), and the GPU reranker beside a running vLLM (item
  # 29: nothing checked the two shared the GPU). Default pools: the format,
  # the connector and the sharing are what is checked, not the budget.
  if [ "${START_CHECK:-1}" = "1" ]; then
    local sc="rc=0; stopall() { bash scripts/2_serving/manage_vllm_server.sh stop >/dev/null 2>&1 || true; bash scripts/2_serving/manage_sglang_server.sh stop >/dev/null 2>&1 || true; }"
    sc="$sc; one() { name=\$1; backend=\$2; api=\$3; launcher=\$4; shift 4; env \"\$@\" bash scripts/2_serving/\$launcher restart '$MODEL' && $POD_PYTHON scripts/checks/probe_thinking_pin.py --backend \$backend --api-base \$api --model '$MODEL' --out logs/start_check_\$name.json && echo START_OK:\$name || { echo START_FAILED:\$name; rc=1; }; }"
    local sglang_on=0 vllm_on=0 sgl_api vllm_api
    for e in $ENGINES; do case "$e" in sglang) sglang_on=1 ;; vllm) vllm_on=1 ;; esac; done
    sgl_api="$(engine_api sglang)"; vllm_api="$(engine_api vllm)"
    if [ "$sglang_on" = "1" ]; then
      sc="$sc; one sglang_fp8_e4m3 sglang $sgl_api manage_sglang_server.sh SGLANG_KV_CACHE_DTYPE=fp8_e4m3; stopall"
    fi
    if [ "$vllm_on" = "1" ]; then
      sc="$sc; one vllm_fp8 vllm $vllm_api manage_vllm_server.sh VLLM_KV_CACHE_DTYPE=fp8; stopall"
      if [ "$needs_lmcache" = "1" ]; then
        sc="$sc; one vllm_lmcache vllm $vllm_api manage_vllm_server.sh VLLM_KV_TRANSFER_CONFIG='{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}' LMCACHE_MAX_LOCAL_CPU_SIZE=${store_gib:-0}; { grep -lE \"'max_local_cpu_size': ${store_gib:-0}(\\.0)?,\" logs/vllm/vllm_*.log >/dev/null 2>&1 && echo START_OK:lmcache_store_${store_gib:-0}; } || { echo START_FAILED:lmcache_store_${store_gib:-0}; rc=1; }; stopall"
      fi
      # item 29: the reranker on the GPU while vLLM holds its default pool
      sc="$sc; bash scripts/2_serving/manage_vllm_server.sh restart '$MODEL' && { $POD_PYTHON scripts/checks/probe_reranker_device.py --out logs/rerank_device_probe_beside_vllm.json && echo START_OK:rerank_beside_vllm || { echo START_FAILED:rerank_beside_vllm; rc=1; }; }; stopall"
    fi
    # the job exits 0 on purpose: the START_* markers carry each verdict and the
    # check below names the failed configuration (a crashed job still fails here)
    sc="$sc; n=1; for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do n=\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true); [ \"\$n\" -eq 0 ] && break; sleep 2; done; echo COMPUTE_APPS=\$n; echo START_CHECK_RC=\$rc; exit 0"
    job_run start_check 2400 "cd $POD_REPO && export PATH=$POD_VENV_BIN:\$PATH VLLM_START_TIMEOUT=$VLLM_START_TIMEOUT$POD_ENGINE_ENV && $sc" "$LAND/logs/setup" || return 1
    plan_only || { grep -E "START_(OK|FAILED):" "$JOB_LOG" | sed 's/^/  /'; }
    if ! plan_only && grep -q "START_FAILED:" "$JOB_LOG"; then
      printf '  [FAIL] a serving configuration the run stage would start for the first time did not start or serve (review 2026-10-13 A1); the launcher gate or the probe named it above\n'; return 1
    fi
    plan_only || grep -q "COMPUTE_APPS=0" "$JOB_LOG" || { printf '  [FAIL] start check: compute apps remain after stop\n'; return 1; }
  else
    say "stage 4 start check SKIPPED (START_CHECK=0 in the profile): the fp8 and LMCache starts and the GPU reranker beside vLLM first run in the run stage"
  fi
  return 0
}

stage_calibrate() {
  pod_env
  local e launcher api out stamp
  # Review 2026-10-13 item 3: per-engine resume. An engine whose calibration
  # or ladder passed on THIS pod and build is not rerun when the stage is
  # repeated after the other engine failed (a 7 to 9 h saving per resume).
  local done_key; done_key="$(state get pod.id 2>/dev/null || true):$(state get build.sha 2>/dev/null || true)"
  for e in $CALIBRATE; do
    launcher="$(engine_launcher "$e")" || { printf '  [FAIL] unknown engine %s\n' "$e"; return 1; }
    api="$(engine_api "$e")"
    out="results/calibration/${EXP}_${e}.json"
    if ! plan_only && [ "$(state get "calibration.cell_done.$e" 2>/dev/null || true)" = "$done_key" ] && [ -f "$EXTRAS/calibration/${EXP}_${e}.json" ]; then
      say "calibrate_cell $e already passed on this pod and build ($done_key): skipped on this resume (review 2026-10-13 item 3)"
      continue
    fi
    # B4 (owner 2026-10-20, "One start, cut prompts"): every demand class's floor on this start
    local cls=""; [ "${CLASS_FLOORS:-0}" = "1" ] && cls=" --class-floors $("$PY" "$SCRIPTS/3_run/run_campaign.py" classes --session "$SESSION")"
    # S1 audit 2026-10-23 (CRITICAL 1): the launcher's default context is 4,096
    # tokens and the anchor class floor prompt is 4,779, so this start runs at the
    # registered largest pool (stage 0's record, the stage 4 start that passed
    # live: VLLM_MAX_MODEL_LEN at the session cap plus the anchor pool). With
    # class floors on and no record, the job would fail 15 min in: refused here.
    local cal_env=""; cal_env="$(state get "registration.largest_pool.$e.env" 2>/dev/null | tr ',' ' ' || true)"
    [ -z "$cal_env" ] || cal_env=" $cal_env"
    if ! plan_only && [ -n "$cls" ] && [ -z "$cal_env" ]; then
      printf '  [FAIL] %s: CLASS_FLOORS=1 but stage 0 recorded no largest pool for this engine (registration.largest_pool.%s.env): the class floor prompts (up to the anchor class) exceed the launcher default context (S1 audit 2026-10-23)\n' "$e" "$e"
      return 1
    fi
    job_run "calibrate_$e" 3600 "cd $POD_REPO && export PATH=$POD_VENV_BIN:\$PATH VLLM_START_TIMEOUT=$VLLM_START_TIMEOUT$POD_ENGINE_ENV && env$cal_env bash scripts/2_serving/$launcher start '$MODEL' && $POD_PYTHON scripts/3_run/calibrate_cell.py --backend $e --model '$MODEL' --api-base $api --manifest $CALIBRATION_MANIFEST --output $out --budget-fraction $CALIBRATION_BUDGET_FRACTION$cls; rc=\$?; bash scripts/2_serving/$launcher stop; exit \$rc" "$LAND/logs/setup" || return 1
    run_step 0 "fetch calibration $e" -- pscp_from "$POD_REPO/$out" "$EXTRAS/calibration/${EXP}_${e}.json" || return 1
    plan_only && continue
    run_step 0 "calibration $e: cal-v3 and floor-derived start" -- "$PY3" - "$EXTRAS/calibration/${EXP}_${e}.json" <<'PY' || return 1
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
pv = str(d.get("procedure_version", ""))
src = d.get("start_qps_source")
print(f"procedure_version={pv!r} start_qps_source={src!r} lambda_star={d.get('lambda_star')}")
sys.exit(0 if pv.startswith("cal-v3") and src == "floor-service-rate" else 1)
PY
    state put "calibration.cell_done.$e" "$done_key"
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
  # ADR-0154 (S0F-62): lambda* per budget rung, measured on the gold-fresh F2
  # workload under the plan's OWN relaunch for that rung (prefix OFF, the gold
  # class budget), by the registered cal-v3 ladder through the runner. One job
  # per engine; the rungs are the session's budgeted rungs (the rehearsal's
  # collapsed ones under REHEARSAL_N) unless the profile names CALIBRATE_RUNGS.
  # The job starts and stops the engines itself (its relaunches ARE the plan's).
  local rungs qm2 reh2 item
  qm2=""; for item in $QUERY_MANIFESTS; do qm2="$qm2 --query-manifest $item"; done
  reh2=""; [ -z "${REHEARSAL_N:-}" ] || reh2=" --rehearsal-n $REHEARSAL_N"
  rungs="${CALIBRATE_RUNGS:-}"
  if [ -z "$rungs" ]; then
    rungs="$("$PY" "$SCRIPTS/3_run/run_campaign.py" rungs --session "$SESSION"$reh2$qm2 | tr ' ' ',')" || { printf '  [FAIL] run_campaign.py rungs failed for session %s\n' "$SESSION"; return 1; }
  fi
  [ -n "$rungs" ] || { printf '  [FAIL] no budgeted rung for session %s (CALIBRATE_RUNGS empty and the driver lists none)\n' "$SESSION"; return 1; }
  # The ladder roots are never reused (a reused root is a PROBE_FAILED rung),
  # so each attempt of this stage gets its own out-root under the run id
  # (review MEDIUM 4); stage 12 pulls the whole rung-calibration tree.
  local rungs_root="results/rung-calibration/${RUN_ID}-$(date -u +%H%M%S)"
  plan_only || state put calibration.rungs_out_root "$rungs_root"
  # S0F-88 B-9 (owner ruling R5): RUNG_ALL_CLASSES=1 ladders every class
  # between the two anchors at its own pressure rungs (no interpolation).
  local allc=""; [ "${RUNG_ALL_CLASSES:-0}" = "1" ] && allc=" --all-classes"
  # Review 2026-10-13 item 3 (S0F-77): LADDER_FAIL_FAST=1 stops a ladder at its
  # first non-ESTIMATED rung; item 21: STEP_TIMEOUT_S bounds every window.
  local ffast=""; [ "${LADDER_FAIL_FAST:-0}" = "1" ] && ffast=" --fail-fast"
  local step_to=""; [ -z "${STEP_TIMEOUT_S:-}" ] || step_to=" CAGE_STEP_TIMEOUT_S=$STEP_TIMEOUT_S"
  # Decision 5 (ADR-0169): the RULER instrument gets its own ladder on the
  # anchor's budget whenever the session registers RULER cells on the engine
  # (automatic in run_campaign.py); RULER_START_QPS, when set, replaces its
  # floor window with an operator start (the remedy when the floor window
  # reads too few unloaded rows).
  local rstart=""; [ -z "${RULER_START_QPS:-}" ] || rstart=" --ruler-start-qps $RULER_START_QPS"
  say "rung calibration (ADR-0154, ADR-0156, ADR-0169${allc:+, S0F-88 B-9 every class}): rungs $rungs per engine on the anchor, the RULER instrument (where registered) and the smallest class${allc:+ and every class between}, bound ${CALIBRATE_RUNGS_BOUND_S}s each, roots under $rungs_root"
  for e in $CALIBRATE; do
    out="results/calibration/rungs_${EXP}_${e}.json"
    local jrc=0
    if ! plan_only && [ "$(state get "calibration.rungs_done.$e" 2>/dev/null || true)" = "$done_key" ] && [ -f "$EXTRAS/calibration/rungs_${EXP}_${e}.json" ]; then
      say "rung calibration $e already passed on this pod and build ($done_key): skipped on this resume (review 2026-10-13 item 3)"
      continue
    fi
    # S0 attempt 2 (2026-10-10 01:00 UTC, ADR-0160): the ladder windows run the
    # runner in campaign mode, so the job carries the same seam env as the run
    # job (CAGE_PROVIDER, CAGE_HARDWARE, the poison strip); without them every
    # probe refused before its first request and all six rungs read PROBE_FAILED.
    job_run "calibrate_rungs_$e" "$CALIBRATE_RUNGS_BOUND_S" "cd $POD_REPO && export PATH=$POD_VENV_BIN:\$PATH VLLM_START_TIMEOUT=$VLLM_START_TIMEOUT$POD_ENGINE_ENV && env $RUN_ENV_UNSET CAGE_PROVIDER=runpod CAGE_HARDWARE='$GPU_ID x $GPU_COUNT'$step_to$(instrument_rev_env) $POD_PYTHON scripts/3_run/run_campaign.py calibrate-rungs --session $SESSION --engine $e --floor-table results/calibration/floor_table.json --calibration results/calibration/${EXP}_${e}.json --rungs $rungs$qm2$reh2$allc$ffast$rstart --seed $SEED --freeze-file $POD_FREEZE_FILE --out $out --out-root $rungs_root" "$LAND/logs/setup" || jrc=1
    if [ "$jrc" -ne 0 ]; then
      # calibrate-rungs exits 1 when a rung is not ESTIMATED and still writes
      # the artifact: fetch it so the labels are readable on the Mac (review LOW 10).
      plan_only || pscp_from "$POD_REPO/$out" "$EXTRAS/calibration/rungs_${EXP}_${e}.json" 2>/dev/null || true
      printf '  [FAIL] rung calibration %s failed; the artifact, if written, is at %s\n' "$e" "$EXTRAS/calibration/rungs_${EXP}_${e}.json"
      return 1
    fi
    run_step 0 "fetch rung calibration $e" -- pscp_from "$POD_REPO/$out" "$EXTRAS/calibration/rungs_${EXP}_${e}.json" || return 1
    plan_only && continue
    run_step 0 "rung calibration $e: every rung of every class ESTIMATED" -- "$PY3" - "$EXTRAS/calibration/rungs_${EXP}_${e}.json" <<'PY' || return 1
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
rungs = d.get("rungs") or {}
classes = d.get("classes") or {}
# ADR-0156: the v2 artifact carries the anchor and the smallest class under classes;
# every rung of every class must be ESTIMATED or the plan refuses the artifact.
tables = {"%s %s" % (c.get("role"), seq): (c.get("rungs") or {}) for seq, c in classes.items()} or {"anchor": rungs}
print("schema=%s classes=%s" % (d.get("schema"), {name: {k: (v.get("label"), v.get("lambda_star_qps")) for k, v in t.items()} for name, t in tables.items()}))
bad = {"%s r=%s" % (name, k): v.get("label") for name, t in tables.items() for k, v in t.items() if v.get("label") != "ESTIMATED"}
if d.get("schema") != "cage-rung-calibration-v2" or not rungs or not classes or bad:
    print("  [FAIL] rung(s) not ESTIMATED: %s (ADR-0154, ADR-0156: the plan refuses them; LADDER_EXHAUSTED: raise the start rate or the rung is beyond the ladder ceiling; NONE_SUSTAINABLE: lower it, or on the smallest class the ladder is defective; RELAUNCH_FAILED / PROBE_FAILED: read logs/setup/calibrate_rungs_<engine>.log)" % (bad or d.get("schema")))
    sys.exit(1)
PY
    state put "calibration.rungs_done.$e" "$done_key"
  done
  return 0
}

# A14 / review 2026-10-13 item 9: the frozen reranker and compressor revisions
# (INSTRUMENT_REVISIONS) as env assignments for a job string, each only when set
instrument_rev_env() {
  "$PY3" -c 'import json,sys
d = (json.load(open(sys.argv[1], encoding="utf-8")).get("INSTRUMENT_REVISIONS") or {})
for slot, env in (("reranker", "CAGE_RERANKER_REVISION"), ("compressor", "CAGE_COMPRESSOR_REVISION")):
    rev = (d.get(slot) or {}).get("revision")
    if rev: print(" %s=%s" % (env, rev), end="")' "$FREEZE_FILE" 2>/dev/null || true
}

class_ceiling_ladders() {
  # ADR-0177 (owner 2026-10-25, option A2): one ceiling-only ladder job per
  # engine (calibrate-rungs --ceilings-only: the classes between the two
  # anchors at their pressure rungs <= 1, the stage 5 ladder rule, their own
  # artifact), resumable per engine on this pod and build like stage 5. No
  # --fail-fast: a lambda* label never gates a ceiling, and a rung with no
  # queued window leaves its class on the absolute bar, recorded.
  local done_key e out item qm2="" reh2="" rungs root step_to=""
  done_key="$(state get pod.id 2>/dev/null || true):$(state get build.sha 2>/dev/null || true)"
  # review 2026-10-25 M4: these jobs start and stop engines; never beside a live run job
  if ! plan_only && [ -f "$JOBS/run.json" ] && [ "$(bash "$PODJOB" status run 2>/dev/null || true)" = "RUNNING" ]; then
    printf '  [FAIL] the run job is RUNNING on the pod: the ceiling ladders, the re-label and the drift window would run beside it (ADR-0177); wait for it to end\n'
    return 1
  fi
  for item in $QUERY_MANIFESTS; do qm2="$qm2 --query-manifest $item"; done
  [ -z "${REHEARSAL_N:-}" ] || reh2=" --rehearsal-n $REHEARSAL_N"
  [ -z "${STEP_TIMEOUT_S:-}" ] || step_to=" CAGE_STEP_TIMEOUT_S=$STEP_TIMEOUT_S"
  rungs="$("$PY" "$SCRIPTS/3_run/run_campaign.py" rungs --session "$SESSION"$reh2$qm2 | tr ' ' ',')" || { printf '  [FAIL] run_campaign.py rungs failed for session %s\n' "$SESSION"; return 1; }
  # ladder roots are never reused: a fresh out-root per attempt (stage 5's rule)
  # S1 resume 2026-10-26: the root is the runner's campaign slug (at most 41
  # characters); <run_id>-ceilings-HHMMSS was 44 for S1 and every window refused
  root="results/rung-calibration/${RUN_ID:0:30}-ceil$(date -u +%H%M%S)"
  for e in $CALIBRATE; do
    out="results/calibration/ceilings_${EXP}_${e}.json"
    if ! plan_only && [ "$(state get "calibration.ceilings_done.$e" 2>/dev/null || true)" = "$done_key" ] && [ -f "$EXTRAS/calibration/ceilings_${EXP}_${e}.json" ]; then
      say "class ceilings $e already measured on this pod and build ($done_key): skipped on this resume"
      continue
    fi
    job_run "class_ceilings_$e" "$CLASS_CEILINGS_BOUND_S" "cd $POD_REPO && export PATH=$POD_VENV_BIN:\$PATH VLLM_START_TIMEOUT=$VLLM_START_TIMEOUT$POD_ENGINE_ENV && env $RUN_ENV_UNSET CAGE_PROVIDER=runpod CAGE_HARDWARE='$GPU_ID x $GPU_COUNT'$step_to$(instrument_rev_env) $POD_PYTHON scripts/3_run/run_campaign.py calibrate-rungs --ceilings-only --session $SESSION --engine $e --floor-table results/calibration/floor_table.json --calibration results/calibration/${EXP}_${e}.json --rungs $rungs$qm2$reh2 --seed $SEED --freeze-file $POD_FREEZE_FILE --out $out --out-root $root" "$LAND/logs/setup" || return 1
    run_step 0 "fetch class ceilings $e" -- pscp_from "$POD_REPO/$out" "$EXTRAS/calibration/ceilings_${EXP}_${e}.json" || return 1
    plan_only && continue
    grep -q "CLASS_CEILINGS_DONE engine=$e .* complete=True" "$JOB_LOG" || { printf '  [FAIL] class ceilings %s: no complete artifact (see %s)\n' "$e" "$JOB_LOG"; return 1; }
    grep "class ceilings written" "$JOB_LOG" | sed 's/^/  /'
    state put "calibration.ceilings_done.$e" "$done_key"
  done
  return 0
}

ceiling_replan_followups() {
  # ADR-0177: after the promoted re-plan (its diff gate passed above), the
  # windows already written are re-judged under
  # the pinned ceilings (journaled, the writer's label kept under "registered")
  # and one completed window is rerun into the drift root (record only)
  job_run relabel 1800 "cd $POD_REPO && $POD_PYTHON scripts/3_run/run_campaign.py relabel-regime --plan results/calibration/plan_$RUN_ID.json --campaign-root $POD_RUN_ROOT --out logs/relabel_$RUN_ID.json" "$LAND/logs/setup" || return 1
  if ! plan_only; then
    grep -q "RELABEL_OK" "$JOB_LOG" || { printf '  [FAIL] relabel printed no RELABEL_OK line (see %s)\n' "$JOB_LOG"; return 1; }
    grep -E "RELABEL_(OK|FLIP)" "$JOB_LOG" | sed 's/^/  /'
  fi
  local roster; roster="$("$PY3" -c 'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); print(" ".join(sorted({str(s["dataset"]) for s in p.get("steps", []) if s.get("kind") == "cell" and s.get("dataset")})))' "$EXTRAS/plan.json" 2>/dev/null || true)"
  [ -n "$roster" ] || roster="$CHARTER_DATASETS"
  # review 2026-10-25 L7: the drift relaunch records its serving config under the
  # drift root (run_campaign.drift_root: the run id's first 30 characters), never the campaign tree
  local droot="${POD_RUN_ROOT%/*}/${RUN_ID:0:30}-dry-drift"
  job_run drift 1800 "cd $POD_REPO && export PATH=$POD_VENV_BIN:\$PATH$POD_ENGINE_ENV && env $RUN_ENV_UNSET CAGE_RUN_ROOT=$droot VLLM_START_TIMEOUT=$VLLM_START_TIMEOUT CAGE_PROVIDER=runpod CAGE_HARDWARE='$GPU_ID x $GPU_COUNT' CAGE_CAMPAIGN_DATASETS='$roster'${STEP_TIMEOUT_S:+ CAGE_STEP_TIMEOUT_S=$STEP_TIMEOUT_S}$(instrument_rev_env) $POD_PYTHON scripts/3_run/run_campaign.py drift-window --plan results/calibration/plan_$RUN_ID.json --campaign-root $POD_RUN_ROOT" "$LAND/logs/setup" || return 1
  plan_only || { grep -E "DRIFT_WINDOW_(DONE|SKIPPED)" "$JOB_LOG" | sed 's/^/  /'; state note plan "$(grep -E "DRIFT_WINDOW_(DONE|SKIPPED)" "$JOB_LOG" | tail -1)"; }
  return 0
}

stage_plan() {
  pod_env
  local qm cal e item
  qm=""; for item in $QUERY_MANIFESTS; do qm="$qm --query-manifest $item"; done
  cal=""; for e in $CALIBRATE; do cal="$cal --calibration $e=results/calibration/${EXP}_${e}.json --rung-calibration $e=results/calibration/rungs_${EXP}_${e}.json"; done
  # ADR-0144: a dress rehearsal derives its grid from the registered session
  # (every arm, engine, hf cell, RULER task and rung; the axes collapsed; one
  # window per cell; every row class at REHEARSAL_N); empty = the registered plan.
  local reh=""; [ -z "${REHEARSAL_N:-}" ] || reh=" --rehearsal-n $REHEARSAL_N"
  # review 2026-10-13 A2: FRONT_LOAD=1 (S0) runs one segment of each first-contact kind first
  local front=""; [ "${FRONT_LOAD:-0}" = "1" ] && front=" --front-load-first-contacts"
  # ADR-0175 amendment: the K1 loads, placed by the registered frontier rule on the anchor ladders
  if [ "${FRONTIER_LOADS:-0}" = "1" ]; then
    job_run frontier 600 "cd $POD_REPO && $POD_PYTHON scripts/3_run/run_campaign.py frontier --session $SESSION$cal --out results/calibration/frontier_$RUN_ID.json" "$LAND/logs/setup" || return 1
    front="$front --frontier-loads results/calibration/frontier_$RUN_ID.json"
  fi
  # ADR-0177 (owner 2026-10-25, option A2): with CLASS_CEILINGS=1 the plan pins
  # each interpolated class its measured fill ceiling. On a landing that already
  # has a plan, that plan is kept once as plan_registered.json, the baseline the
  # diff gate, the re-label and the drift window below are judged against.
  local ceil="" pout="results/calibration/plan_$RUN_ID.json" replan=0
  if [ "${CLASS_CEILINGS:-0}" = "1" ]; then
    class_ceiling_ladders || return 1
    for e in $CALIBRATE; do ceil="$ceil --class-ceilings $e=results/calibration/ceilings_${EXP}_${e}.json"; done
    if ! plan_only && [ -f "$EXTRAS/plan.json" ] && [ ! -f "$EXTRAS/plan_registered.json" ]; then
      cp "$EXTRAS/plan.json" "$EXTRAS/plan_registered.json"
      say "the registered plan is kept as $EXTRAS/plan_registered.json (ADR-0177 baseline)"
    fi
    # review 2026-10-25 M2: a re-plan is written beside the plan the run uses and
    # promoted only after the diff gate passes, so a refused re-plan never runs
    if plan_only || [ -f "$EXTRAS/plan_registered.json" ]; then replan=1; pout="results/calibration/plan_${RUN_ID}_candidate.json"; fi
  fi
  job_run plan 900 "cd $POD_REPO && $POD_PYTHON scripts/3_run/run_campaign.py plan --session $SESSION --floor-table results/calibration/floor_table.json --window-duration-s $WINDOW_DURATION_S --seed $SEED$qm$cal$reh --calibration-budget-fraction $CALIBRATION_BUDGET_FRACTION --freeze-file $POD_FREEZE_FILE$front${LAST_BLOCK:+ --last-block $LAST_BLOCK}$ceil --out $pout" "$LAND/logs/setup" || return 1
  if [ "$replan" -eq 1 ]; then
    run_step 0 "fetch the re-plan candidate" -- pscp_from "$POD_REPO/$pout" "$EXTRAS/plan_candidate.json" || return 1
    run_step 0 "plan diff: only the ceiling fields moved (ADR-0177)" -- "$PY" "$SCRIPTS/3_run/run_campaign.py" plan-diff --old "$EXTRAS/plan_registered.json" --new "$EXTRAS/plan_candidate.json" || return 1
    plan_only || step_has "PLAN_DIFF_OK" || { printf '  [FAIL] plan diff printed no PLAN_DIFF_OK line\n'; return 1; }
    run_step 0 "promote the re-plan" -- pssh "cp $POD_REPO/$pout $POD_REPO/results/calibration/plan_$RUN_ID.json" || return 1
    plan_only || cp "$EXTRAS/plan_candidate.json" "$EXTRAS/plan.json"
  else
    run_step 0 "fetch plan.json" -- pscp_from "$POD_REPO/results/calibration/plan_$RUN_ID.json" "$EXTRAS/plan.json" || return 1
  fi
  if plan_only; then
    # ADR-0177: the plan print shows the follow-ups a re-plan runs after the audits
    [ "${CLASS_CEILINGS:-0}" != "1" ] || ceiling_replan_followups || return 1
    return 0
  fi
  run_step 0 "plan argv audit (S0F-26) and run projection (S0F-88 B-5)" -- "$PY3" - "$EXTRAS/plan.json" "$HOURS" "${PRICE:-}" <<'PY' || return 1
import json, sys
plan = json.load(open(sys.argv[1], encoding="utf-8"))
hours, price = float(sys.argv[2]), sys.argv[3].strip()
steps = plan.get("steps", [])
cells = [s for s in steps if s.get("kind") != "relaunch"]
bad = []
for s in cells:
    argv = s.get("argv", [])
    n = sum(1 for a in argv if a == "--vllm-telemetry")
    # cell steps carry the engine and topology inside their cellspec record
    # (gap triage 2026-10-09 L2: the top-level keys never existed, so the
    # audit checked no cell)
    eng = (s.get("cellspec") or {}).get("engine")
    if eng == "hf" and n != 0:
        bad.append(f"hf cell carries --vllm-telemetry: {s.get('row_key', '?')}")
    if eng not in (None, "hf") and n != 1:
        bad.append(f"{eng} cell carries --vllm-telemetry {n} times: {s.get('row_key', '?')}")
    if (s.get("cellspec") or {}).get("topology") == "pd" and not any(k.startswith("CAGE_PD_") for k in (s.get("env") or {})):
        bad.append(f"pd cell without CAGE_PD_* env: {s.get('row_key', '?')}")
kinds = {}
for s in steps: kinds[s.get("kind")] = kinds.get(s.get("kind"), 0) + 1
print(f"plan schema={plan.get('schema')} steps={len(steps)} by kind={kinds} blocked={len(plan.get('blocked_row_keys', []))}")
# ADR-0156 / S0F-68: the expected pressure-window spans, counted before the GO
ws = plan.get("window_spans") or {}
if ws.get("executable_pressure_cells"):
    print(f"window spans (S0F-68): {ws['executable_pressure_cells']} executable pressure cells, "
          f"min {ws['min_s']:.3g} s, median {ws['median_s']:.3g} s, max {ws['max_s']:.3g} s; "
          f"{ws['below_floor']} below the {ws['floor_s']:g} s floor")
    for seq, rec in (ws.get("by_class") or {}).items():
        print(f"  class {seq} tokens: {rec['cells']} cells, {rec['below_floor']} below floor, min {rec['min_s']:.3g} s")
    if ws["below_floor"]:
        print(f"  [WARN] {ws['below_floor']} pressure window(s) span less than the {ws['floor_s']:g} s warm-up transient at W = per_row_n.window_requests (S0F-68): their regime labels rest on a few sampler ticks; W is registered, raising it is the owner's decision")
# S0F-88 B-5 (owner ruling R9): the measured projection against the run bound
rp = plan.get("run_projection")
if not isinstance(rp, dict):
    bad.append("the plan carries no run_projection (S0F-88 B-5): re-plan with the current code")
else:
    cost = f", about {rp['total_h'] * float(price):.0f} USD at {price} USD/h" if price else ""
    print(f"run projection (S0F-88 B-5): {rp['total_h']:.2f} h against the {hours:g} h bound{cost} "
          f"(relaunch {rp['by_kind_s']['relaunch']:.0f} s, dry windows {rp['by_kind_s']['dry_window']:.0f} s, "
          f"cells {rp['by_kind_s']['cell']:.0f} s; {rp['source']}; {rp['note']})")
    if rp["unmeasured_steps"]:
        bad.append(f"{len(rp['unmeasured_steps'])} plan step(s) have no measured duration (e.g. {rp['unmeasured_steps'][0]}): "
                   "the run time cannot be projected; measure them or extend RUN_*_SECONDS with an ADR")
    # review 2026-10-21 M2: the whole plan counts; the owner cuts the tail with
    # LAST_BLOCK (the plan then holds no later block), never by an exemption here
    if rp["total_s"] > hours * 3600:
        bad.append(f"the projected run ({rp['total_h']:.2f} h) exceeds HOURS={hours:g}: the job would be killed at the "
                   "bound while still billing; raise HOURS in the profile (the seatbelt must cover it) or cut the plan")
for b in bad: print("  [FAIL] " + b)
sys.exit(1 if bad else 0)
PY
  # Audit 2026-10-17 (G7): the bytes the run writes, projected from the plan's
  # arrivals, against the pod disk that holds the run root and its mirror (the
  # container disk when POD_RESULTS_DIR is set, else the network volume), the
  # archive when BACKUP_ARCHIVE_DIR is set, and the Mac's free space for the pull.
  local copies=2 pod_gb="$VOLUME_GB" mac_free
  [ -z "${BACKUP_ARCHIVE_DIR:-}" ] || copies=3
  [ -z "$POD_RESULTS_DIR" ] || pod_gb="$DISK_GB"
  mac_free="$(df -k "$LAND" 2>/dev/null | awk 'NR == 2 { print int($4 / 1048576) }')"
  run_step 0 "disk projection (G7)" -- "$PY3" "$SCRIPT_DIR/disk_projection.py" "$EXTRAS/plan.json" --pod-gb "$pod_gb" --mac-free-gb "${mac_free:-0}" --copies "$copies" || return 1
  if [ "${CLASS_CEILINGS:-0}" = "1" ] && [ -f "$EXTRAS/plan_registered.json" ]; then
    ceiling_replan_followups || return 1
  fi
  return 0
}

custody_copy() {
  # Audit 2026-10-17 (G6): a registered run that fails at stage 7 used to leave
  # its only copy on a billing pod (PARTIAL_TREE_OK is refused there, so seal,
  # pull and teardown all refused). The pod's backup mirror of the UNSEALED
  # tree is copied to the landing first, as non-confirmatory evidence; the
  # owner may then tear the pod down by hand (teardown_pod.sh --force).
  local dest="$LAND/run_unsealed/$RUN_ID"
  mkdir -p "$dest"
  say "custody (G6): copying the pod's mirror of the UNSEALED tree to $dest (non-confirmatory evidence; nothing here is sealed)"
  if pscp_from "$POD_BACKUP_DIR/$BACKUP_RUN_REL/." "$dest/"; then
    state note run "G6 custody: the unsealed tree was copied to $dest at $(utc); the pod may be torn down by hand (teardown_pod.sh --force) once the owner has read it"
    say "custody copy landed: $dest"
  else
    state note run "G6 custody: the mirror copy FAILED at $(utc); the only copy is on the pod ($POD_BACKUP_DIR/$BACKUP_RUN_REL)"
    say "custody copy FAILED: the only copy is still on the pod at $POD_BACKUP_DIR/$BACKUP_RUN_REL"
  fi
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
  local arch=""; [ -z "${BACKUP_ARCHIVE_DIR:-}" ] || arch=" CAGE_BACKUP_ARCHIVE_DIR=$BACKUP_ARCHIVE_DIR CAGE_BACKUP_ARCHIVE_EVERY=${BACKUP_ARCHIVE_EVERY:-6}"
  run_step 0 "backup daemon start" -- pssh "cd $POD_REPO && CAGE_BACKUP_TARGET=$BACKUP_TARGET CAGE_BACKUP_INTERVAL=$BACKUP_INTERVAL_S$arch bash scripts/5_observability/gcs_backup_daemon.sh start $BACKUP_RUN_REL && bash scripts/5_observability/gcs_backup_daemon.sh status $BACKUP_RUN_REL" || return 1
  plan_only || step_has "[gcs-backup] RUNNING (" || { printf '  [FAIL] backup daemon not RUNNING\n'; return 1; }
  # Blocked cells (a registered arm an engine's frozen launcher cannot serve,
  # e.g. retr-store on sglang) make run_campaign.py run REFUSE the whole plan
  # unless --skip-blocked is passed; the consent is the profile's SKIP_BLOCKED=1,
  # applied only when the fetched plan carries blocked_row_keys and said loudly
  # (2026-10-07, ADR-0144). The seal at stage 9 is unchanged.
  local blocked=0 partial=""
  plan_only || blocked="$("$PY3" -c 'import json,sys; print(len(json.load(open(sys.argv[1], encoding="utf-8")).get("blocked_row_keys", [])))' "$EXTRAS/plan.json" 2>/dev/null || echo 0)"
  if plan_only; then
    [ "${SKIP_BLOCKED:-0}" = "1" ] && printf '  [plan] NOTE: SKIP_BLOCKED=1: blocked cells in the plan, if any, are skipped loudly (--skip-blocked)\n'
  elif [ "$blocked" -gt 0 ]; then
    if [ "${SKIP_BLOCKED:-0}" = "1" ]; then
      partial=" --skip-blocked"
      say "the plan carries $blocked blocked cell(s) (see blocked_row_keys in $EXTRAS/plan.json): running the executable subset loudly (SKIP_BLOCKED=1)"
      state note run "$blocked blocked cell(s) skipped loudly (SKIP_BLOCKED=1, --skip-blocked)"
    else
      printf '  [FAIL] the plan carries %s blocked cell(s) (blocked_row_keys in %s) and SKIP_BLOCKED is not 1: run_campaign.py run refuses a plan with blocked cells unless --skip-blocked is passed; set SKIP_BLOCKED=1 in the profile to run the executable subset loudly, or deregister the blocked cells\n' "$blocked" "$EXTRAS/plan.json"
      return 1
    fi
  fi
  local name="run"
  # Review 2026-10-13: item 21's step timeout and decision 10's dry-window
  # telemetry rerun ride the run job when the profile sets them (S0 keys).
  local run_to="" dry_rerun="" rr_rev=""
  [ -z "${STEP_TIMEOUT_S:-}" ] || run_to=" CAGE_STEP_TIMEOUT_S=$STEP_TIMEOUT_S"
  # review 2026-10-13 item 9: the frozen reranker revision (INSTRUMENT_REVISIONS.reranker)
  # pins the GPU reranker at load, the way CAGE_*_REVISION pins the scoring instruments
  run_to="$run_to$(instrument_rev_env)"
  # ADR-0175 item 1: the registered budget stop, RUN_BUDGET_HOURS from the submit (the money cap at the hourly price)
  # review 2026-10-21 M1: the deadline is fixed at the first submit and kept in
  # state, so a resubmit after a failed or killed run job never re-arms the cap
  if [ -n "${RUN_BUDGET_HOURS:-}" ]; then
    local stop_epoch
    stop_epoch="$(state get run.stop_at_epoch 2>/dev/null || true)"
    if [ -z "$stop_epoch" ]; then
      stop_epoch="$("$PY3" -c 'import sys,time; print(int(time.time() + float(sys.argv[1]) * 3600))' "$RUN_BUDGET_HOURS")"
      plan_only || state put run.stop_at_epoch "$stop_epoch"
    fi
    dry_rerun="$dry_rerun --stop-at-epoch $stop_epoch"
  fi
  [ "${DRY_TELEMETRY_RERUN:-0}" = "1" ] && dry_rerun="$dry_rerun --dry-window-telemetry-rerun"
  # review 2026-10-13 B1: RUN_FAIL_FAST=infra|all (S0) stops the run at its first infrastructure failure
  [ -z "${RUN_FAIL_FAST:-}" ] || dry_rerun="$dry_rerun --fail-fast $RUN_FAIL_FAST"
  # S1 audit 2026-10-23 (MEDIUM 2) and review M3: the dataset roster the sealed
  # manifest records (G3: the confirmatory look refuses without it) is the plan's
  # own cell datasets (a session a plan carries RULER cells the charter roster
  # would refuse); the charter roster stands in when the plan names none
  local roster; roster="$("$PY3" -c 'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); print(" ".join(sorted({str(s["dataset"]) for s in p.get("steps", []) if s.get("kind") == "cell" and s.get("dataset")})))' "$EXTRAS/plan.json" 2>/dev/null || true)"
  [ -n "$roster" ] || roster="$CHARTER_DATASETS"
  job_attach_or_submit "$name" "cd $POD_REPO && export PATH=$POD_VENV_BIN:\$PATH$POD_ENGINE_ENV && env $RUN_ENV_UNSET CAGE_RUN_ROOT=$POD_RUN_ROOT VLLM_START_TIMEOUT=$VLLM_START_TIMEOUT CAGE_PROVIDER=runpod CAGE_HARDWARE='$GPU_ID x $GPU_COUNT' CAGE_CAMPAIGN_DATASETS='$roster'$run_to $POD_PYTHON scripts/3_run/run_campaign.py run --plan results/calibration/plan_$RUN_ID.json --campaign-root $POD_RUN_ROOT --seal$partial$dry_rerun" "$bound" || return 1
  if plan_only; then
    printf '  [plan] %-28s expect DONE(0), STOP FAILED(2) or BUDGET STOP(4); DRY WINDOW FAILED(3) fails the stage: bash %s wait run %s\n' "job run: wait" "$PODJOB" "$bound"
    return 0
  fi
  local wrc=0
  bash "$PODJOB" wait "$name" "$JOB_WAIT_S" | tee "$EXTRAS/steps/run_wait.log" || wrc="${PIPESTATUS[0]}"
  bash "$PODJOB" fetch "$name" "$LAND/logs/runner" || true
  JOB_LOG="$LAND/logs/runner/run.log"
  if [ "$wrc" -eq 124 ]; then printf '  [FAIL] the run exceeded its %s h bound and is STILL RUNNING on the pod (see PORTAL ACTION)\n' "$HOURS"; return 1; fi
  if [ "$wrc" -ne 0 ]; then
    if grep -q "REFUSED:" "$JOB_LOG" 2>/dev/null; then printf '  [FAIL] run REFUSED:\n'; grep "REFUSED:" "$JOB_LOG" | head -3; return 1; fi
    # G6: a run that FAILS on a registered profile gets its unsealed tree copied
    # off the pod before the stage fails (the two failing branches below; a
    # PARTIAL acceptance seals and pulls through the normal stages; exit 2,
    # STOP FAILED only, passes with a complete tree and copies nothing: review
    # 2026-10-17 batch 4, H1)
    if grep -q "STOP FAILED" "$JOB_LOG" 2>/dev/null && grep -q "FAILED(2)" "$EXTRAS/steps/run_wait.log"; then
      say "run finished with exit 2 = STOP FAILED only (data tree complete; the pod may still hold an engine)"
      state note run "exit 2: STOP FAILED (ADR-0139); data tree complete"
    elif grep -q "BUDGET STOP (ADR-0175)" "$JOB_LOG" 2>/dev/null && grep -q "FAILED(4)" "$EXTRAS/steps/run_wait.log"; then
      # ADR-0175 item 1: the deadline fixed before the run ended it; the run sealed
      # its tree and every unrun cell carries a budget-stop sentinel (a SKIP)
      say "run ended with exit 4 = BUDGET STOP (ADR-0175): the registered deadline cut the remaining steps"
      grep "BUDGET STOP" "$JOB_LOG" | head -2
      # S1 audit 2026-10-23 (MEDIUM 3): exit 4 rides over a failed dry window (the
      # runner says so and skips its own seal); on a registered profile that is
      # the exit 3 outcome, never a PARTIAL acceptance that stage 9 would seal
      if grep -q "(budget stop) also carries" "$JOB_LOG" 2>/dev/null && grep -q "failed dry windows" "$JOB_LOG" 2>/dev/null; then
        printf '  [%s] the budget stop also carries failed dry windows (ADR-0153): the pressure cells of those serving configurations never ran and the runner skipped its seal:\n' "$(partial_ok && echo PARTIAL || echo FAIL)"
        grep "DRY WINDOW FAILED\|also carries" "$JOB_LOG" | head -12
        if ! partial_ok; then
          state note run "exit 4: BUDGET STOP with failed dry windows (ADR-0153); pressure cells skipped, see logs/runner/run.log"
          custody_copy
          return 1
        fi
        state note run "exit 4: BUDGET STOP with failed dry windows; PARTIAL tree accepted (PARTIAL_TREE_OK=1, S0 only)"
      fi
      grep -q "failed engine stop" "$JOB_LOG" 2>/dev/null && printf '  [FAIL] the same run also reports a failed engine stop (an engine may still hold the GPU)\n'
      state note run "exit 4: BUDGET STOP (ADR-0175); the unrun cells carry budget-stop sentinels"
      state put run.partial "exit 4: budget stop (registered deadline)"
    elif grep -q "DRY WINDOW FAILED" "$JOB_LOG" 2>/dev/null && grep -q "FAILED(3)" "$EXTRAS/steps/run_wait.log"; then
      # ADR-0153: a dry window did not read IN_REGIME, so every pressure cell
      # of its (engine, demand class) was skipped with a sentinel: the tree is
      # PARTIAL. Decision 3 (2026-10-13): under PARTIAL_TREE_OK=1 (S0 only) the
      # stage passes as PARTIAL and the tree is sealed, scored and analyzed;
      # otherwise it fails loudly and the owner decides what runs next.
      printf '  [%s] run ended with exit 3 = DRY WINDOW FAILED (ADR-0153): the pressure cells of these serving configurations never ran; the data tree is partial:\n' "$(partial_ok && echo PARTIAL || echo FAIL)"
      grep "DRY WINDOW FAILED" "$JOB_LOG" | head -12
      grep -q "STOP FAILED" "$JOB_LOG" 2>/dev/null && printf '  [FAIL] the same run also reports STOP FAILED (an engine may still hold the GPU)\n'
      if partial_ok; then
        state note run "exit 3: DRY WINDOW FAILED (ADR-0153); PARTIAL tree accepted (PARTIAL_TREE_OK=1, decision 3 of 2026-10-13, S0 only)"
        state put run.partial "exit 3: dry window(s) failed; pressure cells skipped"
        say "PARTIAL tree accepted on the profile's PARTIAL_TREE_OK=1: the failed dry windows stay recorded; seal, score and analysis run on what executed"
      else
        state note run "exit 3: DRY WINDOW FAILED (ADR-0153); pressure cells skipped, see logs/runner/run.log"
        custody_copy
        return 1
      fi
    elif partial_ok && grep -q "^\[run_campaign\] totals:" "$JOB_LOG" 2>/dev/null && grep -q "FAILED(1)" "$EXTRAS/steps/run_wait.log"; then
      # Decision 3: exit 1 = the run finished (its totals line printed) with
      # failed cells or failed relaunches; a crash prints no totals line and
      # still fails below.
      printf '  [PARTIAL] run ended with exit 1 = failed cell(s) or relaunch(es); the data tree is partial:\n'
      grep "^\[run_campaign\] totals:" "$JOB_LOG" | tail -1
      grep -E "^\s+\[ *(failed|skipped-launch-failed)\]" "$JOB_LOG" | head -12
      state note run "exit 1: failed cell(s) or relaunch(es); PARTIAL tree accepted (PARTIAL_TREE_OK=1, decision 3 of 2026-10-13, S0 only)"
      state put run.partial "exit 1: failed cell(s) or relaunch(es)"
      say "PARTIAL tree accepted on the profile's PARTIAL_TREE_OK=1: the failed cells stay recorded (verify_results names them FAIL); seal, score and analysis run on what executed"
    else
      printf '  [FAIL] run ended %s (see %s)\n' "$(tail -n 1 "$EXTRAS/steps/run_wait.log")" "$JOB_LOG"
      custody_copy
      return 1
    fi
  fi
  if run_partial; then
    # a PARTIAL tree carries sentinels by definition; every executed window still needs its regime.json
    run_step 0 "every executed window has regime.json (PARTIAL tree: sentinels expected)" -- pssh "s=\$(find $POD_RUN_ROOT/cells -name '.STATUS-*' 2>/dev/null | wc -l | tr -d ' '); w=\$(find $POD_RUN_ROOT/cells -type d -name 'window_*' 2>/dev/null | wc -l | tr -d ' '); r=\$(find $POD_RUN_ROOT/cells -name regime.json 2>/dev/null | wc -l | tr -d ' '); echo sentinels=\$s windows=\$w regimes=\$r; [ \"\$w\" -gt 0 ] && [ \"\$w\" -eq \"\$r\" ]" || return 1
    return 0
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
  # A PARTIAL tree (decision 3) carries failure sentinels by definition, so the
  # monitor's "new failure sentinel" HARD lines are the recorded failures, not
  # an unacknowledged alert; every other HARD line still fails the stage.
  local hard_filter="cat"; run_partial && hard_filter="grep -v 'new failure sentinel'"
  if [ -f "$EXTRAS/monitor/alerts.log" ] && grep "HARD" "$EXTRAS/monitor/alerts.log" | eval "$hard_filter" | grep -q "HARD"; then
    printf '  [FAIL] unacknowledged HARD alerts:\n'; grep "HARD" "$EXTRAS/monitor/alerts.log" | eval "$hard_filter" | tail -n 10
    printf '  review them, then resume with --from seal\n'; return 1
  fi
  run_partial && say "PARTIAL tree: $(grep -c 'new failure sentinel' "$EXTRAS/monitor/alerts.log" 2>/dev/null || echo 0) sentinel alert line(s) are the recorded failures (PARTIAL_TREE_OK=1)"
  state evidence monitor "$EXTRAS/monitor/status.json"
  return 0
}

stage_seal() {
  pod_env
  if ! plan_only && pssh "test -f $POD_RUN_ROOT/ledger.json" 2>/dev/null; then say "already sealed (ledger.json present)"; return 0; fi
  # S0F-88 audit (U9): run_campaign.py declines to seal a tree with failed or
  # dry-failed cells (its --seal gate; --seal-partial is the explicit
  # override), so this stage never seals after a failed run stage (a resume
  # with --from seal). Blocked skips under SKIP_BLOCKED=1 stay sealable here
  # (ADR-0144): the run stage passes with them.
  local run_status; run_status="$(state status run)"
  if ! plan_only && [ "$run_status" != "passed" ]; then
    printf '  [FAIL] the run stage is %s, not passed: a tree with failed or dry-failed cells is never sealed by this stage (the run_campaign.py --seal gate); rerun the run stage, or seal by hand only on an explicit partial consent of the owner\n' "$run_status"
    return 1
  fi
  run_partial && say "sealing a PARTIAL tree ($(state get run.partial)): the profile's PARTIAL_TREE_OK=1 is the recorded consent (decision 3 of 2026-10-13, S0 only); verify_results names every failed and skipped cell"
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
  local bound=$(( SCORE_BOUND_MIN * 60 )) force="" bsize=""
  # Scoring passes are append-only (a bug means a NEW SCORING_RUN_ID), so an
  # existing scoring/<id>/ is reused, and the derived predicate table is
  # rebuilt with --force on --redo (review 2026-10-06, MEDIUM 7).
  [ "$REDO" -eq 1 ] && force=" --force"
  # G19 (2026-10-17): the profile's batch size reaches the pass (B13); the pass
  # itself resumes a partial directory window by window after a kill.
  # ADR-0176 Group D (owner 2026-10-22): the batched path is used only after the
  # probe grades the first 50 completed answers of this run both ways on the pod
  # and finds every row identical; any other outcome (a difference, a crash)
  # runs the pass row by row and says why. The job exits 0 on purpose: the
  # verdict line decides, and a crashed probe is a failed proof, never a
  # failed stage.
  if [ -n "${SCORING_BATCH_SIZE:-}" ]; then
    job_run score_probe 900 "cd $POD_REPO && env$(instrument_rev_env) $POD_PYTHON scripts/checks/probe_scoring_batch.py --run-root $POD_RUN_ROOT --batch-size $SCORING_BATCH_SIZE --rows 50 --device cuda --freeze-file $POD_FREEZE_FILE --out logs/scoring_batch_probe.json; exit 0" "$LAND/logs/runner" || return 1
    if ! plan_only; then
      local verdict; verdict="$(grep -E '^SCORING_BATCH_(OK|FAILED)' "$JOB_LOG" | tail -1 || true)"
      case "$verdict" in
        SCORING_BATCH_OK*)
          say "$verdict"; say "scoring runs batched at $SCORING_BATCH_SIZE (the proof passed)"
          state put scoring.batch_proof "$verdict"
          bsize=" --batch-size $SCORING_BATCH_SIZE" ;;
        *)
          [ -n "$verdict" ] || verdict="$FAIL_PROBE_NO_VERDICT"
          say "scoring batch proof did not pass ($verdict): the pass runs row by row"
          state put scoring.batch_proof "$verdict"
          state note score "batched scoring NOT used: $verdict" ;;
      esac
    fi
  fi
  # Review 2026-10-10 (R2): the rescorer reads the freeze file too, so the four
  # instrument revisions are enforced at load (CAGE_*_REVISION) on the pod; (R3)
  # the rescorer builds its pass under a partial name and renames it only once
  # sealed, so the "exists: reused" branch here only ever meets a complete pass.
  job_run score "$bound" "cd $POD_REPO && { [ -d $POD_RUN_ROOT/scoring/$SCORING_RUN_ID ] && echo 'scoring/$SCORING_RUN_ID exists: reused (append-only)' || $POD_PYTHON scripts/4_analysis/rescore_quality.py --run-root $POD_RUN_ROOT --full --device cuda --scoring-run-id $SCORING_RUN_ID --freeze-file $POD_FREEZE_FILE --score-cache-dir results/score_cache/$RUN_ID$bsize; } && $POD_PYTHON scripts/4_analysis/build_predicate_table.py $POD_RUN_ROOT --scoring-run-id $SCORING_RUN_ID --max-null-fraction $MAX_NULL_FRACTION --freeze-file $POD_FREEZE_FILE$force && { $POD_PYTHON scripts/4_analysis/run_calibration.py --campaign-root $POD_RUN_ROOT --scoring-run-id $SCORING_RUN_ID --out $POD_RUN_ROOT/../aa_repeats_$RUN_ID.json --freeze-file $POD_FREEZE_FILE || echo 'A6 A/A FAILED: the confirmatory look is blocked (the report says why)'; }" "$LAND/logs/runner" || return 1
  return 0
}

stage_collect() {
  pod_env
  local backend; backend="$( [ "$BACKUP_SCHEME" = "file" ] && echo local || echo "$BACKUP_SCHEME")"
  local arch2=""; [ -z "${BACKUP_ARCHIVE_DIR:-}" ] || arch2=" CAGE_BACKUP_ARCHIVE_DIR=$BACKUP_ARCHIVE_DIR"
  run_step 0 "backup daemon stop + marker check" -- pssh_once "cd $POD_REPO && t0=\$(date -u +%s); CAGE_BACKUP_TARGET=$BACKUP_TARGET$arch2 bash scripts/5_observability/gcs_backup_daemon.sh stop $BACKUP_RUN_REL; m=\$(stat -c %Y .agent/last_sync_ok_$backend 2>/dev/null || echo 0); echo marker_mtime=\$m stop_started=\$t0; [ \"\$m\" -ge \"\$t0\" ]" || return 1
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
  # ADR-0176 Group C: the A/A report (A6) the score job writes beside the run
  # root; absent when that non-blocking step failed (its line is in score.log)
  local aa="$POD_RUN_ROOT/../aa_repeats_$RUN_ID.json"
  if plan_only; then
    printf '  [plan] %-28s pscp_from %s when present\n' "pull A/A report (A6)" "$aa"
  else
    # review 2026-10-22 M1: only the pod's own "no such file" (rc 1) means absent;
    # a dead channel (ssh_retry's 255) fails the stage, never a quiet "absent"
    local aa_rc=0; pssh "test -f $aa" >/dev/null 2>&1 || aa_rc=$?
    case "$aa_rc" in
      0) run_step 0 "pull A/A report (A6)" -- pscp_from "$aa" "$EXTRAS/aa_repeats_$RUN_ID.json" || return 1
         state put pull.aa_report "$EXTRAS/aa_repeats_$RUN_ID.json" ;;
      1) say "no A/A report beside the run root (the A6 step failed or did not run; see logs/runner/score.log): nothing pulled"
         state put pull.aa_report "absent" ;;
      *) printf '  [FAIL] the A/A report probe did not reach the pod (ssh rc=%s): nothing is recorded as absent on a dead channel\n' "$aa_rc"; return 1 ;;
    esac
  fi
  # ADR-0154 / ADR-0153 (review MEDIUM 5): the ladder windows behind every
  # lambda* and the dry roots behind every regime gate are evidence; they
  # live outside the campaign root, so they are pulled here, best effort
  # (a run without pressure cells has neither).
  if plan_only; then
    printf '  [plan] %-28s pscp_from %s/results/rung-calibration/. and every %s-dry-* sibling\n' "pull rung + dry evidence" "$POD_REPO" "$POD_RUN_ROOT"
  else
    # review 2026-10-13 item 20: a dropped copy used to WARN and teardown could
    # then delete the only copy; the pull fails the stage, and the regime.json
    # count on the Mac must equal the pod's for every tree pulled
    mkdir -p "$LAND/run/rung-calibration"
    local n_pod n_mac
    if pssh "test -d $POD_REPO/results/rung-calibration" >/dev/null 2>&1; then
      pscp_from "$POD_REPO/results/rung-calibration/." "$LAND/run/rung-calibration/" || { printf '  [FAIL] rung-calibration windows not pulled (the ladder evidence behind every lambda*)\n'; return 1; }
      n_pod="$(pssh "find $POD_REPO/results/rung-calibration -name regime.json 2>/dev/null | wc -l | tr -d ' '" 2>/dev/null || echo "?")"
      n_mac="$(find "$LAND/run/rung-calibration" -name regime.json 2>/dev/null | wc -l | tr -d ' ')"
      [ "$n_pod" = "$n_mac" ] || { printf '  [FAIL] rung-calibration pull incomplete: %s regime.json on the pod, %s on the Mac\n' "$n_pod" "$n_mac"; return 1; }
      say "rung-calibration evidence pulled: $n_mac window(s) (count matches the pod)"
    else
      say "no rung-calibration tree on the pod (no ladder ran)"
    fi
    local dry_root local_dry
    for dry_root in $(pssh "ls -d ${POD_RUN_ROOT}-dry-* 2>/dev/null" 2>/dev/null || true); do
      local_dry="$(dirname "$RUN_LOCAL")/$(basename "$dry_root")"
      mkdir -p "$local_dry"
      pscp_from "$dry_root/." "$local_dry/" || { printf '  [FAIL] dry root %s not pulled (the regime gate evidence)\n' "$dry_root"; return 1; }
      n_pod="$(pssh "find $dry_root -name regime.json 2>/dev/null | wc -l | tr -d ' '" 2>/dev/null || echo "?")"
      n_mac="$(find "$local_dry" -name regime.json 2>/dev/null | wc -l | tr -d ' ')"
      [ "$n_pod" = "$n_mac" ] || { printf '  [FAIL] dry root %s pull incomplete: %s regime.json on the pod, %s on the Mac\n' "$dry_root" "$n_pod" "$n_mac"; return 1; }
    done
  fi
  return 0
}

stage_analyze() {
  pod_env
  local force=""
  [ -d "$RUN_LOCAL/index" ] && force="--force"
  # Review 2026-10-13 item 18 (decision 6): the regime labels of every pressure
  # window, by engine and family, printed before anything is analyzed. ADR-0173
  # amendment E12 (2026-10-17, owner option (a); audit G31): with
  # MIN_IN_REGIME_WINDOWS set, every (engine, arm, dataset, budget rung, rate
  # rung >= 1.0) group needs that many windows labeled IN_REGIME and not
  # transient, or the stage fails (attempt 4: 0 of 23 measured windows were
  # IN_REGIME and every stage passed). Empty = report only. The rule lives in
  # scripts/6_experiments/regime_gate.py (tested), no longer in a heredoc.
  run_step 0 "regime label table (item 18, E12 rule)" -- "$PY3" "$SCRIPT_DIR/regime_gate.py" "$RUN_LOCAL" "${MIN_IN_REGIME_WINDOWS:-}" || return 1
  # A PARTIAL tree (decision 3) verifies with FAIL findings on its failed cells
  # (R15); the report is the deliverable, so rc 1 is expected there and noted.
  local vexp=0; run_partial && vexp="0|1"
  run_step "$vexp" "verify_results" -- "$PY" "$SCRIPTS/4_analysis/verify_results.py" "$RUN_LOCAL" --out "$EXTRAS/verify_$(date -u +%H%M%S)" || return 1
  if run_partial && ! plan_only && ! step_has "[verify_results] OK"; then
    say "verify_results reports FAIL findings on the PARTIAL tree (expected: the failed cells, R15); see the step log"
    state note analyze "verify_results reported FAIL findings on the PARTIAL tree (PARTIAL_TREE_OK=1)"
  fi
  # shellcheck disable=SC2086
  # Spec Block A1 (2026-10-16): the section 6.1 currencies G and Y per window,
  # unconditionally (the driver computes them only inside a registered window
  # contrast, which needs two windows per side). Design input: the script's named
  # refusal (rc 3, no calibration floors) is printed and does not fail the stage;
  # rc 2 (a Python usage error or a missing script) fails it (review 2026-10-16).
  run_step "0|3" "yield table (A1)" -- "$PY" "$SCRIPTS/4_analysis/yield_table.py" "$RUN_LOCAL" --calibration-dir "$EXTRAS/calibration" --scoring-run-id "$SCORING_RUN_ID" --out "$LAND/plots/yield_$(date -u +%H%M%S)" || return 1
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
  # Owner rule 2026-10-22 (ADR-0176 Group B): teardown deletes the POD only. The
  # network volume keeps the run tree until the Mac-side S3 pull and is deleted
  # by the owner by hand, if ever. teardown_pod.sh exits 0 with the volume
  # listed; the pod delete is [4/5]. Markers, not the code.
  run_step 0 "teardown_pod.sh (pod only; the volume is kept)" -- env CAGE_ASSUME_YES=1 CAGE_POD_SSH="$host" CAGE_SSH_OPTS="$SSH_OPTS" bash "$SCRIPTS/runpod/teardown_pod.sh" "$POD_ID" "ssh://$host$POD_BACKUP_DIR/$BACKUP_RUN_REL" "$RUN_LOCAL" || return 1
  if ! plan_only; then
    if ! step_has "[4/5] deleting pod" || step_has "ERROR:"; then printf '  [FAIL] the pod delete did not run cleanly; see the step log\n'; return 1; fi
    step_has "TEARDOWN_COMPLETE" && say "teardown_pod.sh reports TEARDOWN_COMPLETE"
  fi
  run_step 0 "listing: pod list --all" -- runpodctl pod list --all || return 1
  # G8: the recorded A2 conductor (if any) is the one pod this proof tolerates
  plan_only || { grep -v '^#' "$STEP_LOG" > "$EXTRAS/ops/pods_after.json"; json_only_ours "$EXTRAS/ops/pods_after.json" $(state get conductor.id 2>/dev/null || true) || { printf '  [FAIL] pods STILL LISTED (STILL BILLING; only a recorded conductor.id is tolerated, G8): %s\n' "$EXTRAS/ops/pods_after.json"; return 1; }; }
  run_step 0 "listing: network-volume list" -- runpodctl network-volume list || return 1
  # the zero proof is ZERO PODS; a listed volume is the kept run tree (storage billing, by design)
  if ! plan_only; then
    grep -v '^#' "$STEP_LOG" > "$EXTRAS/ops/volumes_after.json"
    if json_empty "$EXTRAS/ops/volumes_after.json"; then say "no network volume listed"; else
      # review 2026-10-22 L4: the ids the listing holds, not the recorded one (which may be a placeholder)
      local kept_ids; kept_ids="$("$PY3" -c 'import json,sys
d = json.load(open(sys.argv[1], encoding="utf-8")); items = d if isinstance(d, list) else [v for vs in d.values() if isinstance(vs, list) for v in vs]
print(" ".join(str(i.get("id", "")) for i in items if isinstance(i, dict)))' "$EXTRAS/ops/volumes_after.json" 2>/dev/null || echo "$VOL_ID")"
      say "network volume(s) $kept_ids KEPT with the run tree (owner rule 2026-10-22): storage bills until deleted by hand after the S3 pull; listing in $EXTRAS/ops/volumes_after.json"
      state put pod.volume_kept "$kept_ids"
    fi
  fi
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
  # G16: every pod-side stage starts by proving the seatbelt is alive and covers
  # the bounds still ahead (stage 1 alone proved it before; a Mac reboot
  # between stages left the pod unguarded for days on a long run)
  if ! plan_only && pod_stage "$name"; then
    pod_env
    set +e; watchdog_guard "$name"; rc=$?; set -e
    if [ "$rc" -ne 0 ]; then
      state end "$name" "$rc"; say "stage $name FAILED (rc=$rc): the seatbelt check refused it"; portal_block; return 1
    fi
  fi
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
  # G5: the stage this master takes over may have a pod job that ran on without
  # it; job_attach_or_submit re-attaches to it instead of submitting a second one
  TAKEOVER_STAGE="$running"
  [ -z "$running" ] || TAKEOVER_STARTED="$(state get "stages.$running.started_utc" 2>/dev/null || true)"
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
