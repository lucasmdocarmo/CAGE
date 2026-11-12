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


def _hash(tmp_path: Path, files: dict = BASE, profile: str = PROFILE, tag: str = "a", prefix: str = "",
          extra: list[Path] | None = None) -> str:
    tb = _tarball(tmp_path / f"{tag}.tar.gz", files, prefix)
    pf = tmp_path / f"{tag}.env"
    pf.write_text(profile, encoding="utf-8")
    argv = [sys.executable, str(SCRIPT), "--tarball", str(tb), "--profile", str(pf)]
    for p in extra or []:
        argv += ["--extra-file", str(p)]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=60)
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


def test_the_dependency_pins_at_the_tarball_root_are_inside_the_hash_review_m2(tmp_path: Path) -> None:
    """Review 2026-11-11 M2: setup_runpod.sh builds the engine venvs from requirements.txt and
    the four requirements.lock.*.txt files at the repo root, outside every hashed prefix, so a
    torch or vLLM pin bump plus a setup redo kept the calibration. The five names are hashed
    exactly; another top-level file is not (no prefix rule at the root)."""
    pins = {**BASE, "requirements.lock.gpu.txt": "torch==2.8.0\n", "README.md": "r\n"}
    h0 = _hash(tmp_path, pins, tag="p0")
    assert _hash(tmp_path, {**pins, "requirements.lock.gpu.txt": "torch==2.9.0\n"}, tag="p1") != h0
    assert _hash(tmp_path, {**pins, "README.md": "changed\n"}, tag="p2") == h0
    assert _hash(tmp_path, {**pins, "requirements-dev.txt": "x\n"}, tag="p3") == h0, "an unlisted root file stays outside"
    for name in ("requirements.txt", "requirements.lock.gpu.vllm-step.txt", "requirements.lock.lmdeploy-env.txt", "requirements.lock.sglang-env.txt"):
        assert _hash(tmp_path, {**pins, name: "pin==1\n"}, tag="q" + name) != _hash(tmp_path, {**pins, name: "pin==2\n"}, tag="r" + name), name


def test_extra_files_are_inside_the_hash_by_bytes_and_basename_review_m1(tmp_path: Path) -> None:
    """Review 2026-11-11 M1: the freeze file (MyDocs/registration/freeze_resolutions.json) pins the
    served weights' revision, ships by scp beside the tarball and was outside the hash, so moving
    served_model to a new revision kept the anchors measured on the old weights. --extra-file hashes
    its bytes under its basename (the Mac path is not a measurement input); a missing one exits 2."""
    f1 = tmp_path / "a" / "freeze_resolutions.json"; f1.parent.mkdir(); f1.write_text('{"r": "1"}', encoding="utf-8")
    f2 = tmp_path / "b" / "freeze_resolutions.json"; f2.parent.mkdir(); f2.write_text('{"r": "1"}', encoding="utf-8")
    f3 = tmp_path / "c" / "freeze_resolutions.json"; f3.parent.mkdir(); f3.write_text('{"r": "2"}', encoding="utf-8")
    plain = _hash(tmp_path, tag="e0")
    h1 = _hash(tmp_path, tag="e1", extra=[f1])
    assert h1 != plain
    assert _hash(tmp_path, tag="e2", extra=[f2]) == h1, "same bytes, same basename, another directory"
    assert _hash(tmp_path, tag="e3", extra=[f3]) != h1, "a revision change remeasures"
    assert _hash(tmp_path, tag="e4", extra=[f1, f3]) != _hash(tmp_path, tag="e5", extra=[f3, f1]), "order is part of the digest"
    tb = _tarball(tmp_path / "x.tar.gz", BASE); pf = tmp_path / "x.env"; pf.write_text(PROFILE, encoding="utf-8")
    proc = subprocess.run([sys.executable, str(SCRIPT), "--tarball", str(tb), "--profile", str(pf), "--extra-file", str(tmp_path / "none.json")],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 2 and proc.stdout == "" and "calib_hash" in proc.stderr, (proc.stdout, proc.stderr)


def test_missing_inputs_exit_2_with_nothing_on_stdout_item5(tmp_path: Path) -> None:
    tb = _tarball(tmp_path / "ok.tar.gz", BASE)
    pf = tmp_path / "ok.env"; pf.write_text(PROFILE, encoding="utf-8")
    for argv in (["--tarball", str(tmp_path / "none.tar.gz"), "--profile", str(pf)],
                 ["--tarball", str(tb), "--profile", str(tmp_path / "none.env")],
                 ["--tarball", str(pf), "--profile", str(pf)]):   # not a tarball
        proc = subprocess.run([sys.executable, str(SCRIPT), *argv], capture_output=True, text=True, timeout=60)
        assert proc.returncode == 2 and proc.stdout == "" and "calib_hash" in proc.stderr, (argv, proc.stdout, proc.stderr)
    assert subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True).returncode == 2
