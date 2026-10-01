"""The repo tarball carries uid 0 on every member and no AppleDouble file (S0F-7, ADR-0132).

S0 (2026-09-30): `tar xzf` of the packaged repo onto the network volume failed as
root with ``Cannot change ownership to uid 501``. ``package_repo.sh`` built the
archive in two steps: ``git archive`` (every member root/root) and then a macOS
``bsdtar -rf`` append of ``BUILD_INFO``, which stamped the workstation's uid 501
and added the AppleDouble side file ``._BUILD_INFO``. Extracting as root restores
recorded owners, and the volume refuses the chown. S0 worked around it with
``tar --no-same-owner --exclude='._*'``.

Fix: git writes ``BUILD_INFO`` itself (``git archive --add-file``, one tool, one
step, uid 0 like every tracked member, no macOS attributes), and the ship lines
carry ``--no-same-owner`` anyway. Pinned here by building a real archive from
HEAD into a tmp path and reading its members back.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "ops" / "package_repo.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not on PATH")


def _require_git_checkout() -> None:
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout (tarball deploy)")
    if shutil.which("git") is None:
        pytest.skip("git unavailable")


def test_build_info_is_added_by_git_archive_not_appended_by_tar() -> None:
    code = "\n".join(l for l in SCRIPT.read_text(encoding="utf-8").splitlines()
                     if not l.lstrip().startswith("#"))
    assert re.search(r'git archive --format=tar --add-file="\$TMPD/BUILD_INFO"', code), (
        "BUILD_INFO must enter the archive through git archive --add-file (uid 0, no "
        "AppleDouble side file)")
    assert re.search(r"^\s*tar -rf", code, re.M) is None, (
        "no bsdtar append step may remain: it stamps the workstation uid and adds ._BUILD_INFO")


def test_every_member_is_uid_0_and_no_appledouble_file(tmp_path: Path) -> None:
    _require_git_checkout()
    out = tmp_path / "cage_test.tar.gz"
    proc = subprocess.run(["bash", str(SCRIPT), str(out)], capture_output=True, text=True,
                          cwd=str(REPO_ROOT), timeout=300, env={**os.environ})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PACKAGED" in proc.stdout
    with tarfile.open(out, "r:gz") as tf:
        members = tf.getmembers()
        names = [m.name for m in members]
        assert "BUILD_INFO" in names, names[:5]
        bad_owner = [(m.name, m.uid, m.gid) for m in members if m.uid != 0 or m.gid != 0]
        assert bad_owner == [], f"members with a non-root owner: {bad_owner[:5]}"
        apple = [n for n in names if Path(n).name.startswith("._")]
        assert apple == [], f"AppleDouble side files in the archive: {apple}"
        info = tf.extractfile("BUILD_INFO").read().decode("utf-8")
    assert re.search(r"^sha=[0-9a-f]{40}$", info, re.M), info
    assert re.search(r"^dirty=[01]$", info, re.M) and "packaged_at=" in info


def test_ship_lines_carry_no_same_owner() -> None:
    # The pod extracts as root onto a volume that refuses chown: every documented
    # unpack line says so, and the script's own usage comment leads by example.
    for path in (SCRIPT, REPO_ROOT / "docs" / "RUNBOOK.md"):
        text = path.read_text(encoding="utf-8")
        lines = [l for l in text.splitlines() if "tar xzf cage_" in l]
        assert lines, f"{path.name}: the ship line is gone"
        for line in lines:
            assert "--no-same-owner" in line, f"{path.name}: {line.strip()!r}"
