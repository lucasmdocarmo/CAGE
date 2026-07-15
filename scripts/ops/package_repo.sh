#!/bin/bash
# Package the repo for GPU-VM deploy WITH provenance. The VM tree is a tarball, not a
# git clone, so `git rev-parse` fails there and run_manifest.json recorded sha=null for
# the whole 2026-07-15 smoke run. This script stamps BUILD_INFO into the archive;
# src/observability/provenance.py falls back to it when git is unavailable.
#
# Usage: scripts/ops/package_repo.sh [out.tar.gz]     (default /tmp/cage_<sha8>.tar.gz)
# Then:  scp the tarball; on the VM: mkdir -p ~/CAGE && tar xzf cage_*.tar.gz -C ~/CAGE
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SHA="$(git rev-parse HEAD)"
DIRTY=0
[ -n "$(git status --porcelain)" ] && DIRTY=1
OUT="${1:-/tmp/cage_${SHA:0:8}.tar.gz}"

if [ "$DIRTY" -eq 1 ]; then
  echo "WARNING: working tree is DIRTY -- the archive is HEAD ($SHA) but your tree has" >&2
  echo "         uncommitted changes that will NOT be in the tarball. BUILD_INFO says dirty=1." >&2
fi

# git archive = exactly HEAD, reproducible, no venvs/results/junk. BUILD_INFO is appended
# as a plain tar member so the name inside the archive is exactly BUILD_INFO.
TMPD="$(mktemp -d)"
trap 'rm -rf "$TMPD"' EXIT
{
  echo "sha=$SHA"
  echo "dirty=$DIRTY"
  echo "packaged_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$TMPD/BUILD_INFO"

git archive --format=tar -o "$TMPD/repo.tar" HEAD
tar -rf "$TMPD/repo.tar" -C "$TMPD" BUILD_INFO
gzip -f "$TMPD/repo.tar"
mv "$TMPD/repo.tar.gz" "$OUT"

echo "PACKAGED  $OUT"
echo "  sha=$SHA dirty=$DIRTY  ($(du -h "$OUT" | cut -f1))"
echo "  verify on VM after extract: head -3 ~/CAGE/BUILD_INFO"
