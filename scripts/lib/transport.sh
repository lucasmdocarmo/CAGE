# shellcheck shell=bash
# =============================================================================
# scripts/lib/transport.sh — provider-neutral off-box transport (task #137, J4)
# =============================================================================
# RunPod is the PRIMARY campaign provider (owner decision 2026-08-16, FINAL
# SCOPE v2 in MyDocs/COST_NEBIUS_RUNPOD_2026-08-16.md); GCP/GCS is RETAINED as
# a portability backend behind this abstraction. Every off-box byte (results
# mirror, log collection, pre-teardown pull) moves through these functions so
# an off-GCP box degrades LOUDLY, never silently (finding J4: every transport
# was gcloud/gsutil/metadata-hard and a RunPod pod had ZERO off-box
# persistence, no pull path, no crash forensics — and nothing said so).
#
# Target syntax (CAGE_BACKUP_TARGET, or an explicit argument):
#   gs://bucket[/prefix]           -> gcs   (gcloud storage; loud gsutil fallback)
#   s3://bucket[/prefix]           -> s3    (aws CLI; CAGE_S3_ENDPOINT points it
#                                            at the RunPod network-volume S3 API)
#   ssh://[user@]host/abs/path     -> ssh   (rsync over ssh; CAGE_SSH_OPTS for
#                                            e.g. "-p 2222" — RunPod pods expose
#                                            SSH on non-standard ports)
#   file:///abs/path or /abs/path  -> local (rsync onto an attached volume)
# Anything else dies LOUD. Bare bucket names are NOT resolved here — legacy
# callers normalize CAGE_RESULTS_BUCKET to gs:// themselves before calling in.
#
# API (all fail-loud; none swallow the transfer exit code):
#   transport_resolve <target>           -> echoes gcs|s3|ssh|local
#   transport_join <target> <subpath>    -> target/subpath (slash-safe)
#   transport_push <local_dir> <target>  -> recursive mirror local -> target
#   transport_pull <target> <local_dir>  -> recursive mirror target -> local
#   transport_ls <target>                -> recursive file listing
#   transport_exists <target>            -> 0 iff the remote path exists
#   transport_ensure <target>            -> make the target usable (mkdir -p for
#                                           local/ssh; existence probe for the
#                                           provisioned gcs/s3 buckets)
#   transport_default_target             -> GCS-port convenience: on a GCP box,
#                                           derive gs://<project>-cage-results
#                                           from the metadata server; empty (rc 0)
#                                           anywhere else — NEVER a hard dependency
#   require_backup_target <run_root>     -> resolve-or-REFUSE (J4 loud
#                                           degradation; CAGE_ALLOW_NO_BACKUP=1
#                                           is the recorded override)
#
# CAGE_TRANSPORT_DRYRUN=1 echoes `DRYRUN: <exact command>` instead of executing
# — that is how the gcs/s3/ssh argument construction is unit-tested offline
# (tests/test_topic10_transport_runpod.py); the local (file://) backend is
# tested for REAL against tmpdirs. CAGE_TRANSPORT_GCS_TOOL=gcloud|gsutil forces
# the GCS tool choice (tests; air-gapped images).
#
# Contract (matches _common.sh): sourceable, idempotent, no side effects beyond
# function definitions; never sets shell options; die() exits the caller.
# =============================================================================
if [ -n "${_CAGE_TRANSPORT_SOURCED:-}" ]; then
  return 0
fi
_CAGE_TRANSPORT_SOURCED=1

# log/warn/die come from _common.sh; source it if the caller has not already.
if [ -z "${_CAGE_COMMON_SOURCED:-}" ]; then
  # shellcheck source=scripts/lib/_common.sh
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
fi

_transport_dryrun() { [ "${CAGE_TRANSPORT_DRYRUN:-0}" = "1" ]; }

# _transport_run <cmd...> — execute, or echo the exact command under dry-run.
_transport_run() {
  if _transport_dryrun; then
    printf 'DRYRUN: %s\n' "$*"
    return 0
  fi
  "$@"
}

# _transport_run_quiet <cmd...> — like _transport_run but silences the real
# command's output (existence probes); the DRYRUN echo is never silenced.
_transport_run_quiet() {
  if _transport_dryrun; then
    printf 'DRYRUN: %s\n' "$*"
    return 0
  fi
  "$@" >/dev/null 2>&1
}

# _transport_require <cmd> [hint] — dependency gate, skipped under dry-run so
# argument construction is testable on boxes without the cloud CLIs.
_transport_require() {
  _transport_dryrun && return 0
  require_cmd "$@"
}

transport_resolve() {
  local t="${1:-}"
  [ -n "$t" ] || die "transport_resolve: empty backup target"
  case "$t" in
    gs://*)     echo gcs ;;
    s3://*)     echo s3 ;;
    ssh://*)    echo ssh ;;
    file://*)   echo local ;;
    /*|./*)     echo local ;;
    *) die "transport_resolve: unrecognized backup target '$t' (expected gs://bucket, s3://bucket, ssh://[user@]host/abs/path, file:///abs/path, or an absolute path)" ;;
  esac
}

transport_join() {
  local t="${1%/}" sub="${2:-}"
  sub="${sub#/}"
  if [ -z "$sub" ]; then
    printf '%s\n' "$t"
  else
    printf '%s/%s\n' "$t" "$sub"
  fi
}

# --- per-scheme helpers ------------------------------------------------------

_transport_local_path() {   # strip the file:// scheme (plain paths pass through)
  case "$1" in
    file://*) printf '%s\n' "${1#file://}" ;;
    *)        printf '%s\n' "$1" ;;
  esac
}

_transport_ssh_host() {     # ssh://user@host/path -> user@host
  local rest="${1#ssh://}"
  printf '%s\n' "${rest%%/*}"
}

_transport_ssh_path() {     # ssh://user@host/path -> /path (must be non-empty)
  local rest="${1#ssh://}" path
  path="${rest#*/}"
  [ "$path" != "$rest" ] || die "ssh target '$1' has no path component (need ssh://[user@]host/abs/path)"
  printf '/%s\n' "$path"
}

_transport_bucket_root() {  # gs://b/p | s3://b/p -> gs://b | s3://b
  local t="$1" scheme rest
  scheme="${t%%://*}"
  rest="${t#*://}"
  printf '%s://%s\n' "$scheme" "${rest%%/*}"
}

# GCS tool selection: gcloud storage preferred, LOUD gsutil fallback (mirrors
# pull_run.sh's historical selection); CAGE_TRANSPORT_GCS_TOOL pins it.
_transport_gcs_tool() {
  if [ -n "${CAGE_TRANSPORT_GCS_TOOL:-}" ]; then
    case "$CAGE_TRANSPORT_GCS_TOOL" in
      gcloud|gsutil) printf '%s\n' "$CAGE_TRANSPORT_GCS_TOOL"; return 0 ;;
      *) die "CAGE_TRANSPORT_GCS_TOOL must be 'gcloud' or 'gsutil', got '$CAGE_TRANSPORT_GCS_TOOL'" ;;
    esac
  fi
  if command -v gcloud >/dev/null 2>&1 && gcloud storage --help >/dev/null 2>&1; then
    echo gcloud
    return 0
  fi
  if command -v gsutil >/dev/null 2>&1; then
    warn "'gcloud storage' surface unavailable — falling back to 'gsutil -m rsync' (loud, never silent)"
    echo gsutil
    return 0
  fi
  if _transport_dryrun; then
    echo gcloud
    return 0
  fi
  die "gcs backend: neither 'gcloud storage' nor 'gsutil' is available on this box"
}

# --- the four verbs ----------------------------------------------------------

transport_push() {   # transport_push <local_dir> <target>
  local src="${1:?transport_push: local_dir required}" target="${2:?transport_push: target required}"
  local backend tool host path dst
  backend="$(transport_resolve "$target")" || return 1
  case "$backend" in
    gcs)
      tool="$(_transport_gcs_tool)" || return 1
      if [ "$tool" = "gcloud" ]; then
        _transport_run gcloud storage rsync -r "$src" "$target"
      else
        # -c: checksum compare so a file truncated mid-write on a prior pass is
        # re-uploaded once complete (a partial upload must never become permanent).
        _transport_run gsutil -m rsync -c -r "$src" "$target"
      fi
      ;;
    s3)
      _transport_require aws "install awscli; CAGE_S3_ENDPOINT points it at the RunPod network-volume S3 API"
      # shellcheck disable=SC2086
      _transport_run aws s3 sync "$src" "$target" ${CAGE_S3_ENDPOINT:+--endpoint-url "$CAGE_S3_ENDPOINT"}
      ;;
    ssh)
      _transport_require rsync
      host="$(_transport_ssh_host "$target")" || return 1
      path="$(_transport_ssh_path "$target")" || return 1
      # --rsync-path creates the remote directory in the same connection; -e
      # carries CAGE_SSH_OPTS (RunPod pods listen on non-standard ports).
      _transport_run rsync -az --rsync-path="mkdir -p '$path' && rsync" \
        -e "ssh ${CAGE_SSH_OPTS:-}" "$src/" "$host:$path/"
      ;;
    local)
      dst="$(_transport_local_path "$target")"
      _transport_run mkdir -p "$dst" || return 1
      _transport_run rsync -a "$src/" "$dst/"
      ;;
  esac
}

transport_pull() {   # transport_pull <target> <local_dir>
  local target="${1:?transport_pull: target required}" dst="${2:?transport_pull: local_dir required}"
  local backend tool host path src
  backend="$(transport_resolve "$target")" || return 1
  case "$backend" in
    gcs)
      tool="$(_transport_gcs_tool)" || return 1
      if [ "$tool" = "gcloud" ]; then
        _transport_run gcloud storage rsync -r "$target" "$dst"
      else
        _transport_run gsutil -m rsync -r "$target" "$dst"
      fi
      ;;
    s3)
      _transport_require aws "install awscli; CAGE_S3_ENDPOINT points it at the RunPod network-volume S3 API"
      # shellcheck disable=SC2086
      _transport_run aws s3 sync "$target" "$dst" ${CAGE_S3_ENDPOINT:+--endpoint-url "$CAGE_S3_ENDPOINT"}
      ;;
    ssh)
      _transport_require rsync
      host="$(_transport_ssh_host "$target")" || return 1
      path="$(_transport_ssh_path "$target")" || return 1
      _transport_run mkdir -p "$dst" || return 1
      _transport_run rsync -az -e "ssh ${CAGE_SSH_OPTS:-}" "$host:$path/" "$dst/"
      ;;
    local)
      src="$(_transport_local_path "$target")"
      _transport_run mkdir -p "$dst" || return 1
      _transport_run rsync -a "$src/" "$dst/"
      ;;
  esac
}

transport_ls() {     # transport_ls <target> — recursive file listing
  local target="${1:?transport_ls: target required}"
  local backend tool host path p
  backend="$(transport_resolve "$target")" || return 1
  case "$backend" in
    gcs)
      tool="$(_transport_gcs_tool)" || return 1
      if [ "$tool" = "gcloud" ]; then
        _transport_run gcloud storage ls -r "$target"
      else
        _transport_run gsutil ls -r "$target"
      fi
      ;;
    s3)
      _transport_require aws
      # shellcheck disable=SC2086
      _transport_run aws s3 ls "$target" --recursive ${CAGE_S3_ENDPOINT:+--endpoint-url "$CAGE_S3_ENDPOINT"}
      ;;
    ssh)
      _transport_require ssh
      host="$(_transport_ssh_host "$target")" || return 1
      path="$(_transport_ssh_path "$target")" || return 1
      # shellcheck disable=SC2086
      _transport_run ssh ${CAGE_SSH_OPTS:-} "$host" "find '$path' -type f"
      ;;
    local)
      p="$(_transport_local_path "$target")"
      _transport_run find "$p" -type f
      ;;
  esac
}

transport_exists() { # transport_exists <target> — 0 iff the remote path exists
  local target="${1:?transport_exists: target required}"
  local backend tool host path p
  backend="$(transport_resolve "$target")" || return 1
  case "$backend" in
    gcs)
      tool="$(_transport_gcs_tool)" || return 1
      if [ "$tool" = "gcloud" ]; then
        _transport_run_quiet gcloud storage ls "$target"
      else
        _transport_run_quiet gsutil ls "$target"
      fi
      ;;
    s3)
      _transport_require aws
      # aws s3 ls exits nonzero when the prefix matches nothing.
      # shellcheck disable=SC2086
      _transport_run_quiet aws s3 ls "$target" ${CAGE_S3_ENDPOINT:+--endpoint-url "$CAGE_S3_ENDPOINT"}
      ;;
    ssh)
      _transport_require ssh
      host="$(_transport_ssh_host "$target")" || return 1
      path="$(_transport_ssh_path "$target")" || return 1
      # shellcheck disable=SC2086
      _transport_run_quiet ssh ${CAGE_SSH_OPTS:-} "$host" "test -e '$path'"
      ;;
    local)
      p="$(_transport_local_path "$target")"
      _transport_run test -e "$p"
      ;;
  esac
}

transport_ensure() { # transport_ensure <target> — make the target usable, fail loud
  local target="${1:?transport_ensure: target required}"
  local backend host path p root
  backend="$(transport_resolve "$target")" || return 1
  case "$backend" in
    gcs|s3)
      # Buckets are PROVISIONED (terraform / RunPod console), never auto-created
      # here: probe the bucket root so a typo'd bucket fails NOW, not mid-run.
      root="$(_transport_bucket_root "$target")"
      transport_exists "$root" || return 1
      ;;
    ssh)
      _transport_require ssh
      host="$(_transport_ssh_host "$target")" || return 1
      path="$(_transport_ssh_path "$target")" || return 1
      # shellcheck disable=SC2086
      _transport_run ssh ${CAGE_SSH_OPTS:-} "$host" "mkdir -p '$path'"
      ;;
    local)
      p="$(_transport_local_path "$target")"
      _transport_run mkdir -p "$p"
      ;;
  esac
}

# --- target resolution -------------------------------------------------------

# GCS-port convenience ONLY: on a GCP box the metadata server names the project
# and the terraform-provisioned bucket is gs://<project>-cage-results. Off GCP
# the probe fails fast and this echoes nothing (rc 0). The old gcloud-config
# fallback is deliberately GONE (J4): workstation callers pass the target
# explicitly; implicit workstation defaults are how silent mis-syncs happen.
transport_default_target() {
  local project
  project="$(curl -s --max-time 5 -H 'Metadata-Flavor: Google' \
    http://metadata.google.internal/computeMetadata/v1/project/project-id 2>/dev/null || true)"
  case "$project" in
    *"<"* | *" "* | "") return 0 ;;   # HTML error page / garbage is not a project id
  esac
  printf 'gs://%s-cage-results\n' "$project"
}

# require_backup_target <run_root> — the J4 refusal gate (build item (c)).
# Echoes the resolved target on stdout. With NO resolvable target it DIES,
# unless CAGE_ALLOW_NO_BACKUP=1, in which case the override is durably recorded
# to <run_root>/NO_BACKUP_OVERRIDE (echoed into the run manifest by
# observe_run.py) and the empty target is returned.
require_backup_target() {
  local run_root="${1:?require_backup_target: run_root argument required}"
  local target="${CAGE_BACKUP_TARGET:-${CAGE_RESULTS_BUCKET:-}}" marker
  case "$target" in
    "" | gs://* | s3://* | ssh://* | file://* | /*) ;;
    *) target="gs://${target}" ;;   # legacy bare bucket name (CAGE_RESULTS_BUCKET)
  esac
  if [ -z "$target" ]; then
    target="$(transport_default_target)"
  fi
  if [ -n "$target" ]; then
    transport_resolve "$target" >/dev/null   # dies loud on garbage
    printf '%s\n' "$target"
    return 0
  fi
  if [ "${CAGE_ALLOW_NO_BACKUP:-0}" = "1" ]; then
    mkdir -p "$run_root" || die "require_backup_target: cannot create run root $run_root"
    marker="$run_root/NO_BACKUP_OVERRIDE"
    {
      echo "override=CAGE_ALLOW_NO_BACKUP=1"
      echo "epoch=$(date +%s)"
      echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      echo "host=$(hostname 2>/dev/null || echo unknown)"
      echo "run_root=$run_root"
    } > "$marker" || die "require_backup_target: cannot record the no-backup override marker: $marker"
    warn "NO backup target configured and CAGE_ALLOW_NO_BACKUP=1 — this run has NO off-box persistence until a manual pull (override recorded: $marker)"
    return 0
  fi
  die "no backup target configured (J4): set CAGE_BACKUP_TARGET (gs://|s3://|ssh://[user@]host/path|file:///path) or CAGE_RESULTS_BUCKET — or export CAGE_ALLOW_NO_BACKUP=1 to explicitly accept a run with NO off-box persistence"
}
