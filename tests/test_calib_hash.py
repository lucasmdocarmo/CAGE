"""Item 5 (2026-11-11, S2R pre-restart specs L4, option (a)): scripts/6_experiments/calib_hash.py,
the content hash the three calibration skip keys compare instead of build.sha. The hash follows what
calibration measures: the shipped tarball's members under scripts/, src/, configs/ and data/manifests/
(data/manifests/ joins the approved set: the manifests ship and were re-cut in place this month) and the
resolved profile without comment or blank lines. A doc or test commit keeps 7 to 9 h of anchors and
ladders; any code, config, manifest or profile change remeasures.
"""
from __future__ import annotations

import io
import re
import subprocess
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "6_experiments" / "calib_hash.py"

BASE = {
    "BUILD_INFO": "sha=0123\ndirty=0\n",
    "scripts/2_serving/manage_vllm_server.sh": "#!/bin/bash\necho vllm\n",
    "src/orchestration/calibration.py": "X = 1\n",
    "configs/experiment/s2.yaml": "rungs: [1, 2]\n",
    "data/manifests/musique.json": "{\"n\": 2000}\n",
    "tests/test_x.py": "def test_x(): pass\n",
    "MyDocs/RunPod/NOTE.md": "# note\n",
    "docs/README.md": "readme\n",
}
PROFILE = "# resolved profile\nMODEL=qwen\n\nENGINES=\"vllm sglang\"   # both\n"


def _tarball(path: Path, files: dict, prefix: str = "") -> Path:
    with tarfile.open(path, "w:gz") as tf:
        for name, text in sorted(files.items()):
            data = text.encode()
            info = tarfile.TarInfo(prefix + name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


def _hash(tmp_path: Path, files: dict = BASE, profile: str = PROFILE, tag: str = "a", prefix: str = "") -> str:
    tb = _tarball(tmp_path / f"{tag}.tar.gz", files, prefix)
    pf = tmp_path / f"{tag}.env"
    pf.write_text(profile, encoding="utf-8")
    proc = subprocess.run([sys.executable, str(SCRIPT), "--tarball", str(tb), "--profile", str(pf)],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout.strip()
    assert re.fullmatch(r"[0-9a-f]{64}", out), out
    assert proc.stderr == ""
    return out


def test_doc_test_and_build_info_edits_keep_the_hash_item5(tmp_path: Path) -> None:
    h0 = _hash(tmp_path)
    changed = dict(BASE, **{
        "BUILD_INFO": "sha=ffff\ndirty=0\npackaged_at=later\n",
        "tests/test_x.py": "def test_y(): pass\n",
        "MyDocs/RunPod/NOTE.md": "# a longer note\n",
        "docs/README.md": "rewritten\n",
        "MyDocs/NEW.md": "new file\n",
    })
    assert _hash(tmp_path, changed, tag="b") == h0


def test_a_byte_under_each_hashed_prefix_changes_the_hash_item5(tmp_path: Path) -> None:
    h0 = _hash(tmp_path)
    seen = {h0}
    for i, name in enumerate(["scripts/2_serving/manage_vllm_server.sh", "src/orchestration/calibration.py",
                              "configs/experiment/s2.yaml", "data/manifests/musique.json"]):
        changed = dict(BASE); changed[name] = BASE[name] + "# one more byte\n"
        h = _hash(tmp_path, changed, tag=f"c{i}")
        assert h not in seen, name
        seen.add(h)
    # a new file under a hashed prefix counts too, and a renamed one
    assert _hash(tmp_path, dict(BASE, **{"scripts/new.sh": "x\n"}), tag="d") not in seen
    renamed = dict(BASE); renamed["src/orchestration/calib.py"] = renamed.pop("src/orchestration/calibration.py")
    assert _hash(tmp_path, renamed, tag="e") not in seen


def test_profile_comments_and_blank_lines_are_outside_the_hash_but_values_are_inside_item5(tmp_path: Path) -> None:
    h0 = _hash(tmp_path)
    same = "# another header, frozen at another time\n\n\nMODEL=qwen\nENGINES=\"vllm sglang\"   # both\n   \n"
    assert _hash(tmp_path, profile=same, tag="f") == h0
    assert _hash(tmp_path, profile=PROFILE.replace("MODEL=qwen", "MODEL=llama"), tag="g") != h0
    assert _hash(tmp_path, profile=PROFILE + "NEW_KEY=1\n", tag="h") != h0
    # a trailing comment on a value line is part of that line (never parsed; a changed note remeasures)
    assert _hash(tmp_path, profile=PROFILE.replace("# both", "# item 5"), tag="i") != h0


def test_dot_relative_member_names_hash_like_bare_names_item5(tmp_path: Path) -> None:
    """bsdtar `-C dir .` writes ./scripts/x; git archive writes scripts/x; the same bytes agree."""
    assert _hash(tmp_path, prefix="./", tag="j") == _hash(tmp_path, tag="k")


def test_empty_hashed_set_still_hashes_the_profile_item5(tmp_path: Path) -> None:
    """The master's test world ships a BUILD_INFO-only tarball: the hash is the profile's alone."""
    h0 = _hash(tmp_path, {"BUILD_INFO": "sha=1\n"}, tag="l")
    assert _hash(tmp_path, {"BUILD_INFO": "sha=2\n"}, tag="m") == h0
    assert _hash(tmp_path, {"BUILD_INFO": "sha=1\n"}, profile=PROFILE + "X=1\n", tag="n") != h0


def test_missing_inputs_exit_2_with_nothing_on_stdout_item5(tmp_path: Path) -> None:
    tb = _tarball(tmp_path / "ok.tar.gz", BASE)
    pf = tmp_path / "ok.env"; pf.write_text(PROFILE, encoding="utf-8")
    for argv in (["--tarball", str(tmp_path / "none.tar.gz"), "--profile", str(pf)],
                 ["--tarball", str(tb), "--profile", str(tmp_path / "none.env")],
                 ["--tarball", str(pf), "--profile", str(pf)]):   # not a tarball
        proc = subprocess.run([sys.executable, str(SCRIPT), *argv], capture_output=True, text=True, timeout=60)
        assert proc.returncode == 2 and proc.stdout == "" and "calib_hash" in proc.stderr, (argv, proc.stdout, proc.stderr)
    assert subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True).returncode == 2
