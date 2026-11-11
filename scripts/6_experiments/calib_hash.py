#!/usr/bin/env python3
"""The calibration content hash (item 5, S2R pre-restart specs 2026-11-11, option (a)).

Purpose:   one sha256 over what calibration measures, so the three skip keys in
           cage_experiment.sh (calibration.cell_done, rungs_done, ceilings_done)
           compare pod.id:calib_hash instead of pod.id:build.sha and a doc-only
           or test-only commit keeps 7 to 9 h of anchors and ladders.
Inputs:    --tarball  the packaged repo (scripts/ops/package_repo.sh: a gzip tar
                      of git HEAD plus BUILD_INFO, bare member names; bsdtar's
                      "./x" names hash the same).
           --profile  the resolved profile (extras/profile.env, frozen per landing).
Hashed:    every regular member under scripts/, src/, configs/ or data/manifests/,
           sorted by name, as "<name>\\0<size>\\0<bytes>", then "profile\\0" and
           each profile line with surrounding whitespace stripped, blank lines
           and lines starting with "#" dropped, "\\n" after each.
Outside:   MyDocs/, tests/, docs/, BUILD_INFO, every other member, comment-only
           and blank-line profile edits.
Returns:   the 64-hex digest on stdout, exit 0.
Exit 2:    a missing or unreadable input (message on stderr, nothing on stdout);
           the master then fails stage preflight-mac before anything bills.
Invariant: a byte change in a hashed member or a kept profile line changes the
           digest; a change outside does not.
Complexity: O(total hashed bytes) time, O(number of members) memory.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile

HASHED_PREFIXES = ("scripts/", "src/", "configs/", "data/manifests/")


def _member_name(name: str) -> str:
    return name[2:] if name.startswith("./") else name


def calib_hash(tarball: str, profile: str) -> str:
    h = hashlib.sha256()
    with tarfile.open(tarball, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.isfile() and _member_name(m.name).startswith(HASHED_PREFIXES)]
        members.sort(key=lambda m: _member_name(m.name))
        for m in members:
            h.update(f"{_member_name(m.name)}\0{m.size}\0".encode())
            f = tf.extractfile(m)
            assert f is not None  # isfile() held above
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    h.update(b"profile\0")
    with open(profile, "rb") as fh:
        for raw in fh:
            line = raw.strip()
            if line and not line.startswith(b"#"):
                h.update(line + b"\n")
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="sha256 over the measurement-relevant shipped files and the resolved profile")
    ap.add_argument("--tarball", required=True)
    ap.add_argument("--profile", required=True)
    try:
        args = ap.parse_args(argv)
    except SystemExit:
        print("calib_hash: --tarball and --profile are required", file=sys.stderr)
        return 2
    try:
        digest = calib_hash(args.tarball, args.profile)
    except (OSError, tarfile.TarError) as exc:
        print(f"calib_hash: {exc}", file=sys.stderr)
        return 2
    print(digest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
