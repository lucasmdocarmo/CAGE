#!/usr/bin/env bash
# Order:     after stage 3, before teardown — THE fail-closed pre-teardown gate (called by runpod/teardown_pod.sh and gcp/teardown_vm.sh)
# Objective: Pull ONE campaign run from its backup target and ledger-verify it locally; only its literal SAFE-TO-TEARDOWN line authorizes destruction
# Cloud:     both
# pull_run.sh — pull ONE campaign run from its clean-room backup target and
# ledger-verify it locally. THE fail-closed pre-teardown gate.
#
# Usage:
#   scripts/5_observability/pull_run.sh <target> <local_run_dir>
#     <target>         gs://bucket[/prefix] | s3://bucket[/prefix] |
#                      ssh://[user@]host/path | file:///path | /path
#                      (bare names = legacy gs:// buckets). The tree under it
#                      must BE the run tree per docs/RESULTS_LAYOUT.md:
#                      manifest.json, ledger.json, cells/<row_key>/window_<k>/...,
#                      scoring/...
#     <local_run_dir>  local destination, mirroring the remote structure — normally
#                      results/<campaign>/<session>/<run_id>
#
# STANDING DISCIPLINE (binding, see MEMORY "Pull results local BEFORE teardown"):
#   results are pulled LOCAL and ledger-verified BEFORE any teardown. This script
#   is that gate. ANY failure — transfer error, missing/tampered ledger, hash
#   mismatch — exits nonzero and prints DO-NOT-TEARDOWN. Only the literal
#   "SAFE TO TEARDOWN" line authorizes proceeding to teardown_pod.sh /
#   teardown_vm.sh / backup-destination (volume/bucket) delete.
#
# Transfer: scripts/lib/transport.sh (task #137, finding J4 — the pull path was
# gsutil/gcloud-only, so a RunPod campaign had NO pull path at all). The gcs
# backend keeps the historical tool choice (gcloud storage rsync preferred,
# LOUD gsutil fallback); s3/ssh/file are first-class. No silent fallbacks.
# Integrity is NOT delegated to transfer checksums — the sha256 content-hash
# ledger (src.analysis.stats.ledger, §9.10 UPGRADE 5) re-hashes every sealed
# artifact after the pull.
#
# Cost note (report cost on every cloud action): provider egress is
# ~\$0.12/GB order of magnitude for GCS (region-dependent; RunPod network
# volumes bill storage, not egress); the pulled byte count and that estimate
# are printed on success. GET/list op costs are noise at run-tree object counts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=scripts/lib/_common.sh
source "${REPO_ROOT}/scripts/lib/_common.sh"
# shellcheck source=scripts/lib/transport.sh
source "${REPO_ROOT}/scripts/lib/transport.sh"
# The repo venv: .venv on the workstation, cage-env on a RunPod pod (the S0
# attempt 5 pod suite failed two pull tests on the hard-coded .venv). Still
# fail-closed below when neither exists.
PY="${REPO_ROOT}/.venv/bin/python"
[ -x "${PY}" ] || PY="${REPO_ROOT}/cage-env/bin/python"

_verified=0
on_exit() {
  local rc=$?
  if [ "${_verified}" -ne 1 ]; then
    echo "" >&2
    echo "########################################################################" >&2
    echo "# PULL/VERIFY FAILED (exit ${rc}) — DO NOT TEARDOWN.                    " >&2
    echo "# The backup-target/box copy may be the ONLY intact copy of this run.   " >&2
    echo "# Fix the pull or the mismatch, re-run pull_run.sh, and proceed only    " >&2
    echo "# after it prints SAFE TO TEARDOWN.                                     " >&2
    echo "########################################################################" >&2
  fi
}
trap on_exit EXIT

usage() {
  echo "usage: $0 <target (gs://|s3://|ssh://[user@]host/path|file:///path, or a bare gs bucket name)> <local_run_dir>" >&2
  exit 64
}

[ $# -eq 2 ] || usage
TARGET="$1"
DEST="$2"
[ -n "${TARGET}" ] && [ -n "${DEST}" ] || usage
case "${TARGET}" in
  gs://* | s3://* | ssh://* | file://* | /*) ;;
  *) TARGET="gs://${TARGET}" ;;   # legacy bare bucket name
esac
# rsync semantics need a prefix with no trailing slash
TARGET="${TARGET%/}"

if [ ! -x "${PY}" ]; then
  echo "ERROR: ${PY} not found/executable — the repo venv is required for ledger verification." >&2
  exit 3
fi

# --- [1/3] transfer (provider-neutral, loud, never silent) ------------------
BACKEND="$(transport_resolve "${TARGET}")"
mkdir -p "${DEST}"
echo "[pull_run] [1/3] mirroring ${TARGET} -> ${DEST} (backend: ${BACKEND})"
transport_pull "${TARGET}" "${DEST}"

# --- [2/3] ledger verification (fail-closed) --------------------------------
# verify_ledger re-hashes the pulled tree against the sealed sha256 ledger;
# a tampered LEDGER raises (LedgerError), a tampered/missing ARTIFACT returns
# mismatch lines. Any of those -> nonzero exit -> the DO-NOT-TEARDOWN trap.
echo "[pull_run] [2/3] verifying content-hash ledger against the pulled tree"
VERIFY_SRC='
import os
import sys
from pathlib import Path

run_dir = Path(sys.argv[1]).resolve()
repo_root = sys.argv[2]
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.analysis.stats.ledger import LedgerError, read_ledger, verify_ledger

ledger_path = run_dir / "ledger.json"
if not ledger_path.is_file():
    print(f"LEDGER-MISSING: {ledger_path} — the run tree has no seal to verify against.", file=sys.stderr)
    sys.exit(2)
try:
    entries = read_ledger(ledger_path)
    mismatches = verify_ledger(ledger_path, run_dir)
except LedgerError as exc:
    print(f"LEDGER-ERROR: {exc}", file=sys.stderr)
    sys.exit(2)
if mismatches:
    for line in mismatches:
        print(line, file=sys.stderr)
    print(f"LEDGER-VERIFY-FAILED: {len(mismatches)} mismatch(es) of {len(entries)} sealed artifacts.", file=sys.stderr)
    sys.exit(1)

# S0F-88 audit (U5): the transfer mirrors without deleting (a deleting mirror
# would carry a lost source into the backup), so a file the run tree removed
# after an earlier backup pass (a reset window) can arrive here. Every file
# under cells/ outside the seal (dot entries excepted: the seal skips them by
# design) is MOVED, never deleted, to the sibling <run_dir>.stale/ under its
# relative path and listed in STALE_FILES.txt there; verify_results names it.
stale = []
cells = run_dir / "cells"
if cells.is_dir():
    for path in sorted(cells.rglob("*")):
        rel = path.relative_to(run_dir)
        if path.is_file() and not any(part.startswith(".") for part in rel.parts) and rel.as_posix() not in entries:
            stale.append(rel)
if stale:
    stale_root = run_dir.parent / (run_dir.name + ".stale")
    for rel in stale:
        target = stale_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(run_dir / rel, target)
        print(f"STALE (outside the seal, moved to {stale_root.name}/): {rel.as_posix()}", file=sys.stderr)
    # review L3: a re-pull brings the same leftovers back; the list stays one
    # line per path, and only directories the move emptied are removed
    listing = stale_root / "STALE_FILES.txt"
    known = set(filter(None, listing.read_text(encoding="utf-8").splitlines())) if listing.is_file() else set()
    listing.write_text("".join(n + "\n" for n in sorted(known | {r.as_posix() for r in stale})), encoding="utf-8")
    emptied = {parent for rel in stale for parent in (run_dir / rel).parents if cells in parent.parents}
    for path in sorted(emptied, key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()  # emptied by the move above; holds no data

files = sorted(p for p in run_dir.rglob("*") if p.is_file())
total_bytes = sum(p.stat().st_size for p in files)
gib = total_bytes / (1024 ** 3)
print(f"ledger artifacts verified : {len(entries)}")
print(f"files outside the seal    : {len(stale)} (moved to the .stale sibling)")
print(f"files pulled              : {len(files)}")
print(f"bytes pulled              : {total_bytes} ({gib:.3f} GiB)")
print(f"egress cost estimate      : ~${gib * 0.12:.2f} (at $0.12/GiB order of magnitude)")
'
"${PY}" -c "${VERIFY_SRC}" "${DEST}" "${REPO_ROOT}"

# --- [3/3] verdict ----------------------------------------------------------
_verified=1
echo "[pull_run] [3/3] ledger intact: every sealed artifact of the local copy is byte-identical to the seal (files outside the seal, if any, were moved to the .stale sibling and listed above)."
echo "SAFE TO TEARDOWN"
