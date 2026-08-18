#!/usr/bin/env bash
# =============================================================================
# watch_campaign.sh — bounded, single-shot campaign-run status (layout v2).
# =============================================================================
# Reads ONE campaign run directory (cloud/RESULTS_LAYOUT.md §1:
#   results/<campaign>/<session>/<run_id>/{manifest.json, cells/<row_key>/window_<k>/...})
# entirely from LOCAL disk and prints:
#   - cells present / windows written vs expected (<run>/index/cells_index.csv when
#     organize_results.py has built it; else directory counts only)
#   - latest cage-stats heartbeat age (newest cells/*/window_*/cage_stats.jsonl mtime)
#   - GCS sync lag: last-good-sync marker (.agent/last_gcs_sync_ok, written by
#     sync_results.sh on every successful pass; daemon logs as fallback) vs the
#     newest local artifact mtime. NO gcloud/gsutil call unless CAGE_WATCH_REMOTE=1.
#   - elapsed wall clock (manifest.json mtime = run start; §3: written once at start)
#     + estimated cost when CAGE_HOURLY_USD is set
#   - ONE verdict line: RUNNING-HEALTHY | STALLED>10min | SYNC-LAGGING
#
# Usage:
#   watch_campaign.sh <run_dir>                 # single shot; never blocks > 5s
#   watch_campaign.sh <run_dir> --loop 30       # refresh every 30s; clean Ctrl-C
#
# Env:
#   CAGE_HOURLY_USD       VM $/hour for the cost clock (unset -> reminder printed)
#   CAGE_WATCH_REMOTE=1   ALSO probe the run bucket with gcloud (bounded by `timeout`
#                         when available; default bucket gs://cage-<run_id>)
#   CAGE_RESULTS_BUCKET   bucket override for the remote probe
#   CAGE_SYNC_LAG_MAX_S   sync-lag verdict threshold, seconds (default 900)
#
# Exit codes (single-shot): 0 RUNNING-HEALTHY · 3 STALLED>10min · 4 SYNC-LAGGING
#                           1 error · 2 usage. Loop mode exits 0 on Ctrl-C.
# =============================================================================
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=scripts/lib/_common.sh
source "$PROJECT_DIR/scripts/lib/_common.sh"

STALL_MAX_S=600                                   # the verdict literal is STALLED>10min
SYNC_LAG_MAX_S="${CAGE_SYNC_LAG_MAX_S:-900}"
case "$SYNC_LAG_MAX_S" in ''|*[!0-9]*)
  printf '[watch_campaign] WARNING: CAGE_SYNC_LAG_MAX_S=%s not an integer; using 900\n' "$SYNC_LAG_MAX_S" >&2
  SYNC_LAG_MAX_S=900 ;;
esac

# Local overrides of _common.sh's log/die (sourced above): status output IS the
# data (bare stdout, no "[cage]" prefix) and errors carry this script's tag.
log() { printf '%s\n' "$*"; }                     # status IS the data -> stdout
err() { printf '[watch_campaign] ERROR: %s\n' "$*" >&2; }
die() { err "$*"; exit 1; }

usage() {
  cat <<EOF
usage: $0 <run_dir> [--loop SECONDS]
  <run_dir>   a layout-v2 campaign run root: results/<campaign>/<session>/<run_id>
  --loop N    re-print every N seconds until Ctrl-C (single-shot otherwise)
env: CAGE_HOURLY_USD CAGE_WATCH_REMOTE=1 CAGE_RESULTS_BUCKET CAGE_SYNC_LAG_MAX_S
exit: 0 RUNNING-HEALTHY | 3 STALLED>10min | 4 SYNC-LAGGING | 1 error | 2 usage
EOF
}

# ── portable mtime helpers (BSD stat on macOS laptops, GNU stat on VMs) ───────
if stat -f %m -- "$0" >/dev/null 2>&1; then _STAT=bsd; else _STAT=gnu; fi
mtime_of() {  # mtime_of <path> -> epoch (empty if unreadable)
  if [ "$_STAT" = bsd ]; then stat -f %m -- "$1" 2>/dev/null || true
  else stat -c %Y -- "$1" 2>/dev/null || true; fi
}
minmax_mtime_under() {  # minmax_mtime_under <dir> [find-preds...] -> "min max" | ""
  local dir="$1"; shift || true
  [ -d "$dir" ] || return 0
  {
    if [ "$_STAT" = bsd ]; then
      find "$dir" "$@" -type f -print0 2>/dev/null | xargs -0 stat -f %m -- 2>/dev/null
    else
      find "$dir" "$@" -type f -print0 2>/dev/null | xargs -0 stat -c %Y -- 2>/dev/null
    fi
  } | awk 'NR==1{min=$1;max=$1} {if($1<min)min=$1; if($1>max)max=$1} END{if(NR)print min" "max}' || true
}

fmt_dur() {  # fmt_dur <seconds> -> compact human duration
  local s="$1" d h m
  if [ "$s" -lt 0 ] 2>/dev/null; then s=0; fi
  d=$(( s / 86400 )); h=$(( (s % 86400) / 3600 )); m=$(( (s % 3600) / 60 ))
  if   [ "$d" -gt 0 ]; then printf '%dd%02dh%02dm' "$d" "$h" "$m"
  elif [ "$h" -gt 0 ]; then printf '%dh%02dm%02ds' "$h" "$m" "$(( s % 60 ))"
  elif [ "$m" -gt 0 ]; then printf '%dm%02ds' "$m" "$(( s % 60 ))"
  else printf '%ds' "$s"; fi
}

# ── args ──────────────────────────────────────────────────────────────────────
RUN_DIR=""
LOOP=""
while [ $# -gt 0 ]; do
  case "$1" in
    --loop)    LOOP="${2:-}"; [ -n "$LOOP" ] || { usage >&2; die "--loop needs a seconds value"; }; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    -*)        usage >&2; err "unknown option: $1"; exit 2 ;;
    *)         if [ -z "$RUN_DIR" ]; then RUN_DIR="$1"; else usage >&2; err "unexpected extra arg: $1"; exit 2; fi; shift ;;
  esac
done
[ -n "$RUN_DIR" ] || { usage >&2; exit 2; }
[ -d "$RUN_DIR" ] || die "run dir not found: $RUN_DIR"
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
if [ -n "$LOOP" ]; then
  case "$LOOP" in ''|*[!0-9]*) usage >&2; err "--loop '$LOOP' is not a positive integer"; exit 2 ;; esac
fi

print_status() {  # sets VERDICT (and VERDICT_RC) for the caller
  local now cells_n windows_n expected expected_note mm oldest newest
  local hb_mm hb_m hb_age hb_note act_m activity_age
  local marker_path marker_m c m lag_note lag_s
  local start_m start_note elapsed_s cost_note
  now="$(date +%s)"

  log "== CAGE campaign watch: $RUN_DIR  ($(date -u +%Y-%m-%dT%H:%M:%SZ)) =="

  # 1) cells / windows vs expected -------------------------------------------
  cells_n="$(find "$RUN_DIR/cells" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ' || true)"
  windows_n="$(find "$RUN_DIR/cells" -mindepth 2 -maxdepth 2 -type d -name 'window_*' 2>/dev/null | wc -l | tr -d ' ' || true)"
  expected="" expected_note="no index yet (organize_results.py not run) -> dir counts only"
  if [ -f "$RUN_DIR/index/cells_index.csv" ]; then
    expected="$(wc -l < "$RUN_DIR/index/cells_index.csv" | tr -d ' ' || true)"
    if [ -n "$expected" ] && [ "$expected" -gt 0 ] 2>/dev/null; then
      expected=$(( expected - 1 ))               # minus CSV header; one row per window
      expected_note="index/cells_index.csv"
    else
      expected="" expected_note="index/cells_index.csv unreadable -> dir counts only"
    fi
  fi
  if [ ! -d "$RUN_DIR/cells" ]; then
    log "cells    : NONE — $RUN_DIR/cells/ does not exist yet (pre-first-cell or wrong dir?)"
  elif [ -n "$expected" ]; then
    log "cells    : $cells_n present"
    log "windows  : $windows_n written / $expected expected ($expected_note)"
  else
    log "cells    : $cells_n present"
    log "windows  : $windows_n written ($expected_note)"
  fi

  # 2) cage-stats heartbeat ---------------------------------------------------
  hb_mm="$(minmax_mtime_under "$RUN_DIR/cells" -mindepth 3 -maxdepth 3 -name cage_stats.jsonl)"
  hb_m="${hb_mm##* }"
  if [ -n "$hb_m" ]; then
    hb_age=$(( now - hb_m )); if [ "$hb_age" -lt 0 ]; then hb_age=0; fi
    log "cage-stats heartbeat : $(fmt_dur "$hb_age") ago (newest cage_stats.jsonl)"
  else
    hb_age=""
    log "cage-stats heartbeat : none yet (no cells/*/window_*/cage_stats.jsonl)"
  fi

  # newest/oldest artifact under the whole run tree (activity + sync lag + start fallback)
  mm="$(minmax_mtime_under "$RUN_DIR")"
  oldest="${mm%% *}"; newest="${mm##* }"

  # 3) GCS sync lag — markers only, no gcloud (unless CAGE_WATCH_REMOTE=1) ----
  marker_path="" marker_m=""
  for c in "$PROJECT_DIR/.agent/last_gcs_sync_ok" \
           "$PROJECT_DIR"/.agent/last_sync_ok_* \
           "$PROJECT_DIR"/.agent/daemons/*/gcs_backup.log \
           "$PROJECT_DIR/.agent/gcs_backup.log"; do
    [ -f "$c" ] || continue
    m="$(mtime_of "$c")"
    [ -n "$m" ] || continue
    if [ -z "$marker_m" ] || [ "$m" -gt "$marker_m" ]; then marker_m="$m"; marker_path="$c"; fi
  done
  lag_s="" lag_note=""
  if [ -n "$marker_m" ]; then
    if [ -n "$newest" ] && [ "$newest" -gt "$marker_m" ]; then lag_s=$(( newest - marker_m )); else lag_s=0; fi
    log "gcs sync : last good sync $(fmt_dur $(( now - marker_m ))) ago -> lag $(fmt_dur "$lag_s") behind newest local artifact (marker: ${marker_path#"$PROJECT_DIR/"})"
  else
    lag_note="no-marker"
    log "gcs sync : NO sync marker found (.agent/last_sync_ok_<backend> or legacy .agent/last_gcs_sync_ok) — daemons not started, or never synced"
  fi
  if [ "${CAGE_WATCH_REMOTE:-0}" = "1" ]; then
    local bucket tmo rc
    bucket="${CAGE_RESULTS_BUCKET:-gs://cage-$(basename "$RUN_DIR")}"
    case "$bucket" in gs://*) ;; *) bucket="gs://$bucket" ;; esac
    tmo="$(command -v timeout || command -v gtimeout || true)"
    rc=0
    if [ -n "$tmo" ]; then "$tmo" 5 gcloud storage ls "$bucket" >/dev/null 2>&1 || rc=$?
    else gcloud storage ls "$bucket" >/dev/null 2>&1 || rc=$?; fi
    if [ "$rc" -eq 0 ]; then log "gcs remote: $bucket reachable (CAGE_WATCH_REMOTE=1)"
    elif [ "$rc" -eq 124 ]; then log "gcs remote: $bucket probe TIMED OUT after 5s (CAGE_WATCH_REMOTE=1)"
    else log "gcs remote: $bucket NOT reachable (rc=$rc, CAGE_WATCH_REMOTE=1)"; fi
  fi

  # 4) elapsed + cost ---------------------------------------------------------
  start_m="" start_note=""
  if [ -f "$RUN_DIR/manifest.json" ]; then
    start_m="$(mtime_of "$RUN_DIR/manifest.json")"; start_note="manifest.json mtime"
  elif [ -n "$oldest" ]; then
    start_m="$oldest"; start_note="oldest artifact mtime — no manifest.json yet"
  fi
  if [ -n "$start_m" ]; then
    elapsed_s=$(( now - start_m )); if [ "$elapsed_s" -lt 0 ]; then elapsed_s=0; fi
    if [ -n "${CAGE_HOURLY_USD:-}" ]; then
      cost_note="cost ~\$$(awk -v s="$elapsed_s" -v r="$CAGE_HOURLY_USD" 'BEGIN{printf "%.2f", s/3600.0*r}') (@ \$${CAGE_HOURLY_USD}/h)"
    else
      cost_note="set CAGE_HOURLY_USD for cost clock"
    fi
    log "elapsed  : $(fmt_dur "$elapsed_s") ($start_note) | $cost_note"
  else
    elapsed_s=0
    log "elapsed  : unknown (empty run dir — no manifest.json, no artifacts)"
  fi

  # 5) verdict ----------------------------------------------------------------
  # Activity = the freshest write signal we have: cage-stats heartbeat OR any artifact.
  act_m="$hb_m"
  if [ -n "$newest" ] && { [ -z "$act_m" ] || [ "$newest" -gt "$act_m" ]; }; then act_m="$newest"; fi
  if [ -n "$act_m" ]; then activity_age=$(( now - act_m )); else activity_age="$elapsed_s"; fi
  if [ "$activity_age" -lt 0 ]; then activity_age=0; fi

  if [ "$activity_age" -gt "$STALL_MAX_S" ]; then
    VERDICT="STALLED>10min"; VERDICT_RC=3
    log "VERDICT: STALLED>10min (no artifact written for $(fmt_dur "$activity_age"))"
  elif { [ -n "$lag_s" ] && [ "$lag_s" -gt "$SYNC_LAG_MAX_S" ]; } || { [ "$lag_note" = "no-marker" ] && [ "${windows_n:-0}" -gt 0 ]; }; then
    VERDICT="SYNC-LAGGING"; VERDICT_RC=4
    if [ "$lag_note" = "no-marker" ]; then
      log "VERDICT: SYNC-LAGGING (windows on disk but NO successful GCS sync recorded)"
    else
      log "VERDICT: SYNC-LAGGING (local data $(fmt_dur "$lag_s") ahead of last good sync; threshold $(fmt_dur "$SYNC_LAG_MAX_S"))"
    fi
  else
    VERDICT="RUNNING-HEALTHY"; VERDICT_RC=0
    log "VERDICT: RUNNING-HEALTHY"
  fi
}

VERDICT=""; VERDICT_RC=0
if [ -z "$LOOP" ]; then
  print_status
  exit "$VERDICT_RC"
fi

trap 'printf "\n[watch_campaign] stopped.\n"; exit 0' INT TERM
while true; do
  print_status
  log ""
  sleep "$LOOP"
done
