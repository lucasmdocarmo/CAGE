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
