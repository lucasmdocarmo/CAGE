#!/usr/bin/env python3
"""Order:     stage 0 (preflight-mac) of scripts/6_experiments/cage_experiment.sh, right after the repo tarball is packaged
Objective: One sha256 over what calibration measures (the tarball's scripts/, src/, configs/ and
           data/manifests/ members, the pinned requirements at the tarball root, the resolved
           profile and the freeze file) so the calibration skip keys compare pod.id:calib_hash
           instead of pod.id:build.sha (item 5, S2R pre-restart specs 2026-11-11, option (a))
Cloud:     local

The calibration content hash (item 5, S2R pre-restart specs 2026-11-11, option (a)).

Purpose:   one sha256 over what calibration measures, so the three skip keys in
           cage_experiment.sh (calibration.cell_done, rungs_done, ceilings_done)
           compare pod.id:calib_hash instead of pod.id:build.sha and a doc-only
           or test-only commit keeps 7 to 9 h of anchors and ladders.
Inputs:    --tarball  the packaged repo (scripts/ops/package_repo.sh: a gzip tar
                      of git HEAD plus BUILD_INFO, bare member names; bsdtar's
                      "./x" names hash the same).
           --profile  the live profile the master sources (review 2026-11-11 L1:
                      the frozen extras/profile.env is written once per landing,
                      so a later edit of an out-of-repo --profile escaped it;
                      _common.env ships in the tarball and is hashed there).
           --extra-file  a file outside the tarball that a measurement depends on,
                      repeatable, hashed in the order given (review 2026-11-11 M1:
                      the freeze file pins the served weights' revision and is
                      scp'd beside the tarball, not inside it).
Hashed:    every regular member under scripts/, src/, configs/ or data/manifests/,
           plus the top-level dependency pins named in HASHED_NAMES (review
           2026-11-11 M2: setup builds the venvs from them, so a torch or vLLM
           pin bump plus a setup redo must remeasure), sorted by name, as
           "<name>\\0<size>\\0<bytes>", then "profile\\0" and each profile line
           with surrounding whitespace stripped, blank lines and lines starting
           with "#" dropped, "\\n" after each, then for each extra file
           "extra\\0<basename>\\0<size>\\0<bytes>" (the basename, not the path:
           the Mac path of the freeze file is not a measurement input).
Outside:   MyDocs/, tests/, docs/, BUILD_INFO, every other member, comment-only
           and blank-line profile edits.
Returns:   the 64-hex digest on stdout, exit 0.
Exit 2:    a missing or unreadable input, an extra file included (message on
           stderr, nothing on stdout);
           the master then fails stage preflight-mac before anything bills.
Invariant: a byte change in a hashed member or a kept profile line changes the
           digest; a change outside does not.
Complexity: O(total hashed bytes) time, O(number of members) memory.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tarfile

HASHED_PREFIXES = ("scripts/", "src/", "configs/", "data/manifests/")
# review 2026-11-11 M2: the files scripts/runpod/setup_runpod.sh installs the
# engine venvs from (exact names at the tarball root; a new lock file is added
# here by hand, never by prefix, so an unrelated top-level file stays outside).
HASHED_NAMES = (
    "requirements.txt",
    "requirements.lock.gpu.txt",
    "requirements.lock.gpu.vllm-step.txt",
    "requirements.lock.lmdeploy-env.txt",
    "requirements.lock.sglang-env.txt",
)


def _member_name(name: str) -> str:
    return name[2:] if name.startswith("./") else name


def _hashed(name: str) -> bool:
    return name.startswith(HASHED_PREFIXES) or name in HASHED_NAMES


def calib_hash(tarball: str, profile: str, extra_files: list[str] | None = None) -> str:
    h = hashlib.sha256()
    with tarfile.open(tarball, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.isfile() and _hashed(_member_name(m.name))]
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
    for path in extra_files or []:
        with open(path, "rb") as fh:
            data = fh.read()
        h.update(f"extra\0{os.path.basename(path)}\0{len(data)}\0".encode())
        h.update(data)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="sha256 over the measurement-relevant shipped files and the resolved profile")
    ap.add_argument("--tarball", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--extra-file", action="append", default=[], help="a measurement input outside the tarball (repeatable; the freeze file)")
    try:
        args = ap.parse_args(argv)
    except SystemExit:
        print("calib_hash: --tarball and --profile are required", file=sys.stderr)
        return 2
    try:
        digest = calib_hash(args.tarball, args.profile, args.extra_file)
    except (OSError, tarfile.TarError) as exc:
        print(f"calib_hash: {exc}", file=sys.stderr)
        return 2
    print(digest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
