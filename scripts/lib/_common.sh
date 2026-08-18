# shellcheck shell=bash
# =============================================================================
# scripts/lib/_common.sh — shared shell helpers for every CAGE script
# =============================================================================
# Source it near the top of a script (after set -euo pipefail / set -uo pipefail):
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/_common.sh"
# or, from a script that already resolved its repo root:
#   source "$PROJECT_DIR/scripts/lib/_common.sh"
#
# Contract:
#   - sourceable, idempotent (double-source is a no-op), NO side effects beyond
#     defining functions and resolving CAGE_ROOT;
#   - never sets shell options (each script owns its own strict-mode line);
#   - die() exits the CALLING script (that is the point).
# =============================================================================
if [ -n "${_CAGE_COMMON_SOURCED:-}" ]; then
  return 0
fi
_CAGE_COMMON_SOURCED=1

# Repo root, resolved from THIS file's location (<root>/scripts/lib/_common.sh).
# An already-exported CAGE_ROOT wins (e.g. a tarball deploy that sets it explicitly).
if [ -z "${CAGE_ROOT:-}" ]; then
  CAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
export CAGE_ROOT

# Canonical CPython for BOTH the local analysis venv and the GPU cage-env
# (code assertion 2026-08-07, finding B1). The Tier-1 exact pins in
# requirements.txt were frozen and TESTED on CPython 3.13 — running any other
# interpreter turns them into an untested claim. NOT overridable: canonical
# means canonical. Consumers: setup_gpu_cloud.sh (venv creation, fail-closed),
# preflight_check.sh gate (i), tests/test_scripts_doctrine.py (pins the
# requirements.txt header and the setup script to this value).
CAGE_CANONICAL_PYTHON="3.13"
export CAGE_CANONICAL_PYTHON

# --- logging (printf, never echo -e; stderr for warn/die) --------------------
log()  { printf '[cage] %s\n' "$*"; }
warn() { printf '[cage] WARNING: %s\n' "$*" >&2; }
die()  { printf '[cage] FATAL: %s\n' "$*" >&2; exit 1; }

# require_cmd <cmd> [hint] — fail loud when a dependency is missing (no silent fallback).
require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1${2:+ ($2)}"
}

# require_env <VAR>... — fail loud when a required environment variable is unset/empty.
require_env() {
  local _v
  for _v in "$@"; do
    [ -n "${!_v:-}" ] || die "required env var unset or empty: $_v"
  done
}

# --- runner-hardening helpers (Topic-10 walkthrough J-series, task #136) ------

# metrics_json_valid <file> — 0 iff <file> exists AND parses as JSON.
# The shell resume gates (cell_complete in every runner) must never treat a
# corrupt/truncated/foreign metrics.json as "complete" (finding J2): existence
# is not validity. An unparseable file is announced LOUDLY and the cell is
# treated as incomplete, so it re-runs instead of freezing corrupt data in.
metrics_json_valid() {
  local f="$1"
  [ -f "$f" ] || return 1
  if ! python3 -c 'import json,sys; json.load(open(sys.argv[1], encoding="utf-8"))' "$f" >/dev/null 2>&1; then
    warn "invalid metrics.json (unparseable JSON) -> treating cell as INCOMPLETE: $f"
    return 1
  fi
  return 0
}

# mint_run_id <model-slug> <num-queries> <num-trials> <dataset> — unique run-id:
#   <YYYY-MM-DD_HHMMSS>_<slug>_<Q>x<T>_<4-hex-random>_<dataset>
# Seconds + a random suffix close the J3 fragment/converge hazard of the old
# minute-granular ids (two runners minting in the same minute CONVERGED on one
# root; a resumed runner one minute later FRAGMENTED onto a new one). The
# dataset lands LAST because the pilot bridge (build_legacy_index.infer_dataset)
# reads the dataset from a run name's `_<dataset>` suffix, fail-closed.
mint_run_id() {
  printf '%s_%s_%sx%s_%04x_%s' "$(date +%Y-%m-%d_%H%M%S)" "$1" "$2" "$3" "$((RANDOM % 65536))" "$4"
}

# acquire_run_lock <run_root> — exclusive NON-BLOCKING flock on
# <run_root>/.cage_run.lock (finding J3: two resume instances on one root can
# rm -rf a live cell). Fail-LOUD if another runner holds it. Re-entrant across
# the orchestrator chain (run_full_sweep -> cloud_run -> run_baselines): the
# first acquirer exports CAGE_RUN_LOCK_HELD=<lockfile>, children on the SAME
# root skip re-acquisition (the parent's fd keeps the lock alive). The lock fd
# (200) is deliberately inherited by FOREGROUND children (a live run_experiment
# keeps blocking a second instance even if the runner shell dies) but must be
# closed (200>&-) when launching DETACHED daemons, or a surviving daemon would
# hold the lock forever. flock(1) is util-linux: absent on macOS, where we warn
# LOUDLY and continue (GPU runs happen on Linux VMs; flock exists there).
acquire_run_lock() {
  local root="$1" lockf="$1/.cage_run.lock"
  if [ "${CAGE_RUN_LOCK_HELD:-}" = "$lockf" ]; then
    log "run lock already held by a parent process: $lockf"
    return 0
  fi
  mkdir -p "$root" || die "cannot create run root for lock: $root"
  if ! command -v flock >/dev/null 2>&1; then
    warn "flock(1) not found (macOS?) -- concurrent-runner exclusion NOT enforced on $lockf"
    return 0
  fi
  exec 200>>"$lockf" || die "cannot open run lock file: $lockf"
  if ! flock -n 200; then
    die "another runner already holds the run lock $lockf -- a second resume instance can rm -rf a live cell (J3). Wait for it to finish or stop it first."
  fi
  export CAGE_RUN_LOCK_HELD="$lockf"
  log "acquired exclusive run lock: $lockf"
}

# pidfile_alive <pidfile> <identity-substring> — 0 iff <pidfile> holds a live
# pid whose `ps` command line contains <identity-substring>. The identity check
# is the J8 PID-reuse guard: a recycled pid from a rebooted/busy VM must never
# be treated as (or KILLED as) our daemon just because a stale pidfile names it.
pidfile_alive() {
  local pf="$1" ident="$2" pid cmd
  [ -f "$pf" ] || return 1
  pid="$(cat "$pf" 2>/dev/null || true)"
  [ -n "$pid" ] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  # -ww: NEVER truncate the command line (BSD ps clips otherwise; the gcs loop's
  # identity token -- its pidfile path -- sits AFTER the ~700-char loop body).
  cmd="$(ps -ww -p "$pid" -o command= 2>/dev/null || true)"
  case "$cmd" in *"$ident"*) return 0 ;; esac
  return 1
}

# confirm "<prompt>" -> 0 = yes, 1 = no.  FAIL-CLOSED in non-interactive shells:
# with no TTY the answer is NO unless CAGE_ASSUME_YES=1 was exported deliberately.
# Use before every destructive / cost-starting action (terraform apply/destroy,
# instance delete, bucket rm).
confirm() {
  local prompt="${1:-Proceed?}" reply
  if [ "${CAGE_ASSUME_YES:-0}" = "1" ]; then
    log "confirm: auto-yes via CAGE_ASSUME_YES=1: $prompt"
    return 0
  fi
  if [ ! -t 0 ]; then
    warn "confirm: no TTY and CAGE_ASSUME_YES!=1 — answering NO (fail-closed): $prompt"
    return 1
  fi
  printf '%s [y/N] ' "$prompt" >&2
  IFS= read -r reply || return 1
  case "$reply" in
    y|Y|yes|YES) return 0 ;;
    *)           return 1 ;;
  esac
}
