#!/bin/bash
# Order:     provisioning bracket — before 1_setup (ship step; workstation side)
# Objective: Package the repo as a provenance-stamped tarball (BUILD_INFO) for deploy onto a box without git
# Cloud:     local
# Package the repo for GPU-VM deploy WITH provenance. The VM tree is a tarball, not a
# git clone, so `git rev-parse` fails there and run_manifest.json recorded sha=null for
# the whole 2026-07-15 smoke run. This script stamps BUILD_INFO into the archive;
# src/observability/provenance.py falls back to it when git is unavailable.
#
# Usage: scripts/ops/package_repo.sh [out.tar.gz]     (default /tmp/cage_<sha8>.tar.gz)
# Then:  scp the tarball; on the pod (network volume at /workspace, fresh pod so ~/CAGE
#        is not yet a real directory): mkdir -p /workspace/CAGE && ln -sfn /workspace/CAGE ~/CAGE
#        && tar xzf cage_*.tar.gz --no-same-owner -C ~/CAGE
#        (--no-same-owner: root extracting onto the network volume must not restore
#        recorded owners, the volume refuses chown; S0F-7, 2026-09-30)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=scripts/lib/_common.sh
source "$REPO_ROOT/scripts/lib/_common.sh"
require_cmd git "packaging stamps BUILD_INFO from the git HEAD"

SHA="$(git rev-parse HEAD)"
DIRTY=0
[ -n "$(git status --porcelain)" ] && DIRTY=1
OUT="${1:-/tmp/cage_${SHA:0:8}.tar.gz}"

if [ "$DIRTY" -eq 1 ]; then
  echo "WARNING: working tree is DIRTY -- the archive is HEAD ($SHA) but your tree has" >&2
  echo "         uncommitted changes that will NOT be in the tarball. BUILD_INFO says dirty=1." >&2
fi

# git archive = exactly HEAD, reproducible, no venvs/results/junk. BUILD_INFO enters
# through --add-file (git 2.30+), so git writes it like every tracked member: top-level
# name BUILD_INFO, owner root/root. The earlier `tar -rf` append ran macOS bsdtar, which
# stamped the workstation uid (501) and added the AppleDouble side file ._BUILD_INFO;
# root extracting that onto the network volume failed on the chown (S0F-7, S0 2026-09-30).
TMPD="$(mktemp -d)"
trap 'rm -rf "$TMPD"' EXIT
{
  echo "sha=$SHA"
  echo "dirty=$DIRTY"
  echo "packaged_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$TMPD/BUILD_INFO"

git archive --format=tar --add-file="$TMPD/BUILD_INFO" -o "$TMPD/repo.tar" HEAD
gzip -f "$TMPD/repo.tar"
mv "$TMPD/repo.tar.gz" "$OUT"

echo "PACKAGED  $OUT"
echo "  sha=$SHA dirty=$DIRTY  ($(du -h "$OUT" | cut -f1))"
echo "  verify on VM after extract: head -3 ~/CAGE/BUILD_INFO"
