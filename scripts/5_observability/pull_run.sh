#!/usr/bin/env bash
# pull_run.sh — pull ONE campaign run from its clean-room GCS bucket and
# ledger-verify it locally. THE fail-closed pre-teardown gate.
#
# Usage:
#   scripts/5_observability/pull_run.sh <bucket> <local_run_dir>
#     <bucket>         gs://bucket[/prefix] (bare bucket name accepted). The object
#                      tree under it must BE the run tree per cloud/RESULTS_LAYOUT.md:
#                      manifest.json, ledger.json, cells/<row_key>/window_<k>/...,
#                      scoring/...
#     <local_run_dir>  local destination, mirroring the bucket structure — normally
#                      results/<campaign>/<session>/<run_id>
#
# STANDING DISCIPLINE (binding, see MEMORY "Pull results local BEFORE teardown"):
#   results are pulled LOCAL and ledger-verified BEFORE any teardown. This script
#   is that gate. ANY failure — rsync error, missing/tampered ledger, hash
#   mismatch — exits nonzero and prints DO-NOT-TEARDOWN. Only the literal
#   "SAFE TO TEARDOWN" line authorizes proceeding to teardown_vm.sh / bucket delete.
#
# Transfer tool: gcloud storage rsync -r (modern surface, parallel by default);
# loud fallback to gsutil -m rsync -r only when the gcloud storage surface is
# absent. No silent fallbacks. Integrity is NOT delegated to rsync checksums —
# the sha256 content-hash ledger (src.analysis.stats.ledger, §9.10 UPGRADE 5)
# re-hashes every sealed artifact after the pull.
#
# Cost note (report cost on every cloud action): GCS -> local download is network
# egress (~\$0.12/GB order of magnitude, region-dependent); the actual pulled byte
# count and an egress estimate are printed on success. GET/list op costs are noise
# at run-tree object counts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY="${REPO_ROOT}/.venv/bin/python"

_verified=0
on_exit() {
  local rc=$?
  if [ "${_verified}" -ne 1 ]; then
    echo "" >&2
    echo "########################################################################" >&2
    echo "# PULL/VERIFY FAILED (exit ${rc}) — DO NOT TEARDOWN.                    " >&2
    echo "# The bucket/VM copy may be the ONLY intact copy of this run's data.    " >&2
    echo "# Fix the pull or the mismatch, re-run pull_run.sh, and proceed only    " >&2
    echo "# after it prints SAFE TO TEARDOWN.                                     " >&2
    echo "########################################################################" >&2
  fi
}
trap on_exit EXIT

usage() {
  echo "usage: $0 <bucket (gs://bucket[/prefix] or bare name)> <local_run_dir>" >&2
  exit 64
}

[ $# -eq 2 ] || usage
BUCKET="$1"
DEST="$2"
[ -n "${BUCKET}" ] && [ -n "${DEST}" ] || usage
case "${BUCKET}" in
  gs://*) ;;
  *) BUCKET="gs://${BUCKET}" ;;
esac
# rsync semantics need a prefix with no trailing slash
BUCKET="${BUCKET%/}"

if [ ! -x "${PY}" ]; then
  echo "ERROR: ${PY} not found/executable — the repo venv is required for ledger verification." >&2
  exit 3
fi

# --- [1/3] transfer tool selection (loud, never silent) ---------------------
TOOL=""
if command -v gcloud >/dev/null 2>&1 && gcloud storage --help >/dev/null 2>&1; then
  TOOL="gcloud"
elif command -v gsutil >/dev/null 2>&1; then
  TOOL="gsutil"
  echo "[pull_run] WARN: 'gcloud storage' surface unavailable — falling back to 'gsutil -m rsync -r'." >&2
else
  echo "ERROR: neither 'gcloud storage' nor 'gsutil' is available; cannot pull ${BUCKET}." >&2
  exit 3
fi

mkdir -p "${DEST}"
echo "[pull_run] [1/3] mirroring ${BUCKET} -> ${DEST} (tool: ${TOOL})"
if [ "${TOOL}" = "gcloud" ]; then
  gcloud storage rsync -r "${BUCKET}" "${DEST}"
else
  gsutil -m rsync -r "${BUCKET}" "${DEST}"
fi

# --- [2/3] ledger verification (fail-closed) --------------------------------
# verify_ledger re-hashes the pulled tree against the sealed sha256 ledger;
# a tampered LEDGER raises (LedgerError), a tampered/missing ARTIFACT returns
# mismatch lines. Any of those -> nonzero exit -> the DO-NOT-TEARDOWN trap.
echo "[pull_run] [2/3] verifying content-hash ledger against the pulled tree"
VERIFY_SRC='
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

files = sorted(p for p in run_dir.rglob("*") if p.is_file())
total_bytes = sum(p.stat().st_size for p in files)
gib = total_bytes / (1024 ** 3)
print(f"ledger artifacts verified : {len(entries)}")
print(f"files pulled              : {len(files)}")
print(f"bytes pulled              : {total_bytes} ({gib:.3f} GiB)")
print(f"egress cost estimate      : ~${gib * 0.12:.2f} (at $0.12/GiB order of magnitude)")
'
"${PY}" -c "${VERIFY_SRC}" "${DEST}" "${REPO_ROOT}"

# --- [3/3] verdict ----------------------------------------------------------
_verified=1
echo "[pull_run] [3/3] ledger intact — local copy is provably identical to the sealed run data."
echo "SAFE TO TEARDOWN"
