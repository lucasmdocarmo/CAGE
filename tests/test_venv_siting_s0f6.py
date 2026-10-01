"""ADR-0125 (S0F-6, live H100 2026-09-30): the three venvs live on the container
disk, linked at the repo root; the repo, logs/ and results/ stay on the volume.

WHAT and WHY: setup_runpod.sh created cage-env, sglang-env and lmdeploy-env
relative to the repo root, which on a volume-backed pod is the MooseFS network
volume. Every Python start then paid for FUSE small-file reads: bootstrap 79 min
(local boxes 7 to 13 min), vLLM launch to API up 246 to 446 s (local 62 to 86 s),
`import vllm` 21.9 s. Measured loss: 1.8 to 2.4 h of pod time per S0-sized day.

The rule pinned here:
1. the venvs are created at REAL paths under CAGE_VENV_ROOT (default
   /root/cage-venvs, the container disk) and the three repo-root names are
   symlinks to them. CPython refuses to create a venv through a link and fails
   on a dangling one (venv/__init__.py), so the real path comes first, the link
   after; a link left dangling by a pod restart (the container disk is wiped)
   is re-pointed once the venv is rebuilt; a REAL directory at the repo root
   (a pre-ADR-0125 venv on the volume) is refused, never deleted.
2. logs/ and results/ do not move (collect_logs.sh runs `find` on logs/, which
   does not follow a link; logs must outlive a seatbelt kill).
3. provision_pod.sh gives the pod a 120 GB container disk (venvs about 40 GB
   plus the model prefetch; gate (f) ran at zero margin on the 60 GB disk).

No pod, no network: static pins plus bash-level checks of the extracted
functions against fake interpreters, and one real venv through a link.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SETUP = REPO_ROOT / "scripts" / "runpod" / "setup_runpod.sh"
PROVISION = REPO_ROOT / "scripts" / "runpod" / "provision_pod.sh"
RUNBOOK = REPO_ROOT / "docs" / "RUNBOOK.md"


def _text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _code(text: str) -> str:
    return "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))


def _extract_function(text: str, name: str) -> str:
    m = re.search(rf"^{name}\(\) \{{\n.*?^\}}$", text, re.M | re.S)
    assert m, f"{name}() not found in setup_runpod.sh"
    return m.group(0)


# ---------------------------------------------------------------------------
# 1. static pins
# ---------------------------------------------------------------------------


def test_setup_sites_the_venvs_under_cage_venv_root_and_links_them() -> None:
    text = _text(SETUP)
    code = _code(text)
    assert re.search(r'^CAGE_VENV_ROOT="\$\{CAGE_VENV_ROOT:-/root/cage-venvs\}"$', text, re.M), (
        "setup_runpod.sh must declare an overridable CAGE_VENV_ROOT defaulting to the container disk"
    )
    assert re.search(r'^\s*"\$PYBIN" -m venv "\$CAGE_VENV_ROOT/cage-env"', code, re.M), (
        "cage-env must be created at its REAL container-disk path, from $PYBIN"
    )
    assert re.search(r'^\s*"\$PYBIN" -m venv cage-env', code, re.M) is None, (
        "no venv may be created at a repo-root path (the network volume on a volume-backed pod)"
    )
    assert "link_venv cage-env" in code, "cage-env must be linked at the repo root"
    # the engine venvs are linked by install_engine_venv itself, by name
    engine_fn = _extract_function(text, "install_engine_venv")
    assert 'link_venv "$name"' in engine_fn, "install_engine_venv must link the engine venv by name"
    for name in ("sglang-env", "lmdeploy-env"):
        assert f"install_engine_venv {name} " in code, f"{name} must be installed by NAME (so it is linked)"
    assert "source cage-env/bin/activate" in code, "activation goes through the repo-root link"
    assert "ADR-0125" in text and "S0F-6" in text
    header = text.split("set -euo pipefail")[0]
    assert "CAGE_VENV_ROOT" in header, "the knob is documented in the header beside the others"
    # logs/ and results/ stay where they are: no link for them
    assert "link_venv logs" not in code and "link_venv results" not in code


def test_provision_default_container_disk_is_120_gb() -> None:
    text = _text(PROVISION)
    assert re.search(r'DISK_GB="120"', text), "the container disk default must be 120 GB (ADR-0125)"
    assert "ADR-0125" in text
    # plan mode with an explicit price needs no runpodctl at all
    env = {k: v for k, v in os.environ.items() if k not in ("CAGE_POD_LEDGER", "CAGE_POD_IMAGE")}
    proc = subprocess.run(
        ["bash", str(PROVISION), "--gpu-id", "NVIDIA GeForce RTX 4090", "--price-per-hour", "1", "--hours", "1"],
        capture_output=True, text=True, timeout=120, env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "container disk    : 120 GB" in proc.stdout
    assert "PLAN ONLY" in proc.stdout + proc.stderr


def test_runbook_records_the_siting_rule() -> None:
    text = _text(RUNBOOK)
    assert "CAGE_VENV_ROOT" in text and "ADR-0125" in text


# ---------------------------------------------------------------------------
# 2. link_venv under real bash: link, re-point a dangling link, refuse a real dir
# ---------------------------------------------------------------------------


def _harness(tmp_path: Path, body: str, *, pybin: Path | None = None) -> subprocess.CompletedProcess:
    text = _text(SETUP)
    fns = _extract_function(text, "link_venv") + "\n" + _extract_function(text, "install_engine_venv")
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    root = tmp_path / "venvs"
    harness = tmp_path / "harness.sh"
    harness.write_text(
        "set -uo pipefail\n"
        'warn() { echo "[cage] WARNING: $*" >&2; }\n'
        'die() { echo "[cage] FATAL: $*" >&2; exit 1; }\n'
        f'PROJECT_DIR="{repo}"\n'
        f'CAGE_VENV_ROOT="{root}"\n'
        f'PYBIN="{pybin or "/usr/bin/false"}"\n'
        f"{fns}\n"
        f"{body}\n",
        encoding="utf-8",
    )
    return subprocess.run(["bash", str(harness)], capture_output=True, text=True, timeout=120)


def test_link_venv_creates_the_repo_root_link_and_replaces_a_dangling_one(tmp_path: Path) -> None:
    root = tmp_path / "venvs"
    (root / "cage-env").mkdir(parents=True)
    repo = tmp_path / "repo"
    repo.mkdir()
    # a dangling link from an earlier pod whose container disk was wiped
    (repo / "cage-env").symlink_to(tmp_path / "gone" / "cage-env")
    proc = _harness(tmp_path, 'link_venv cage-env; echo "rc=$?"')
    assert "rc=0" in proc.stdout, proc.stdout + proc.stderr
    link = repo / "cage-env"
    assert link.is_symlink() and link.resolve() == (root / "cage-env").resolve()


def test_link_venv_refuses_a_real_directory_at_the_repo_root_without_deleting(tmp_path: Path) -> None:
    root = tmp_path / "venvs"
    (root / "sglang-env").mkdir(parents=True)
    repo = tmp_path / "repo"
    (repo / "sglang-env").mkdir(parents=True)
    (repo / "sglang-env" / "pyvenv.cfg").write_text("home = /old\n", encoding="utf-8")
    proc = _harness(tmp_path, 'link_venv sglang-env; echo "rc=$?"')
    # review 2026-09-30 LOW 4: link_venv warns and returns 1 (the caller decides
    # whether that is fatal), so the engine step stays loud and non-fatal
    assert "rc=1" in proc.stdout, proc.stdout + proc.stderr
    assert "real directory" in proc.stderr and "by hand" in proc.stderr
    assert (repo / "sglang-env" / "pyvenv.cfg").read_text(encoding="utf-8") == "home = /old\n"
    assert not (repo / "sglang-env").is_symlink()


def test_setup_names_every_pre_adr_real_venv_directory_up_front() -> None:
    """Review 2026-09-30 LOW 4: with three real pre-ADR-0125 venvs on a reused
    volume the operator learns all three names at once, before any work."""
    code = _code(_text(SETUP))
    assert "for _n in cage-env sglang-env lmdeploy-env; do" in code
    assert "move each aside by hand" in code
    assert code.index("move each aside by hand") < code.index('"$PYBIN" -m venv "$CAGE_VENV_ROOT/cage-env"')
    assert 'link_venv cage-env || die' in code, "cage-env's link failure is fatal; the engine links are not"


def test_install_engine_venv_builds_under_the_root_and_links_the_name(tmp_path: Path) -> None:
    """A fake interpreter that materializes a fake venv tree on `-m venv <path>`:
    the venv lands under CAGE_VENV_ROOT, the repo-root name becomes a link, and
    the install runs through the venv's own pip (a stub)."""
    fake = tmp_path / "pybin" / "python-fake"
    fake.parent.mkdir(parents=True)
    fake.write_text(
        '#!/bin/sh\n'
        'case "$1" in\n'
        '  -c) echo 3.13 ;;\n'
        '  -m) d="$3"; mkdir -p "$d/bin"; echo "home = /fake" > "$d/pyvenv.cfg";\n'
        '      printf "#!/bin/sh\\necho 3.13\\n" > "$d/bin/python3"; chmod +x "$d/bin/python3";\n'
        '      printf "#!/bin/sh\\nexit 0\\n" > "$d/bin/pip"; chmod +x "$d/bin/pip" ;;\n'
        'esac\nexit 0\n',
        encoding="utf-8",
    )
    fake.chmod(0o755)
    proc = _harness(
        tmp_path, 'install_engine_venv sglang-env "SGLang test" "pkg==1"; echo "rc=$?"', pybin=fake
    )
    assert "rc=0" in proc.stdout, proc.stdout + proc.stderr
    real = tmp_path / "venvs" / "sglang-env"
    assert (real / "pyvenv.cfg").is_file(), "the venv must live under CAGE_VENV_ROOT"
    link = tmp_path / "repo" / "sglang-env"
    assert link.is_symlink() and link.resolve() == real.resolve()


def test_install_engine_venv_still_takes_an_absolute_path_for_the_bash_level_tests(tmp_path: Path) -> None:
    """tests/test_w29_w30_live_fixes.py drives the function with an absolute venv
    path; that form keeps working and creates no link."""
    fake = tmp_path / "pybin" / "python-fake"
    fake.parent.mkdir(parents=True)
    fake.write_text(
        '#!/bin/sh\ncase "$1" in -c) echo 3.13 ;; -m) mkdir -p "$3/bin"; echo "home = /fake" > "$3/pyvenv.cfg";\n'
        'printf "#!/bin/sh\\nexit 0\\n" > "$3/bin/pip"; chmod +x "$3/bin/pip" ;; esac\nexit 0\n',
        encoding="utf-8",
    )
    fake.chmod(0o755)
    target = tmp_path / "elsewhere" / "sglang-env"
    proc = _harness(tmp_path, f'install_engine_venv "{target}" "SGLang test" "pkg==1"; echo "rc=$?"', pybin=fake)
    assert "rc=0" in proc.stdout, proc.stdout + proc.stderr
    assert (target / "pyvenv.cfg").is_file()
    assert not (tmp_path / "repo" / "sglang-env").exists()


# ---------------------------------------------------------------------------
# 3. a real venv works through the repo-root link (activation, prefix, interpreter)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="posix venv layout")
def test_a_real_venv_works_through_the_repo_root_link(tmp_path: Path) -> None:
    root = tmp_path / "venvs"
    root.mkdir()
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(root / "cage-env")], check=True)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "cage-env").symlink_to(root / "cage-env")
    probe = (
        f'cd "{repo}" && source cage-env/bin/activate && '
        'python -c "import sys; print(sys.prefix); print(sys.prefix != sys.base_prefix)"'
    )
    proc = subprocess.run(["bash", "-c", probe], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    prefix, is_venv = proc.stdout.strip().splitlines()[-2:]
    assert is_venv == "True", "the interpreter must see the venv (pyvenv.cfg found through the link)"
    assert prefix.endswith("cage-env")
