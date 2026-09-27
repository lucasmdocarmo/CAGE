"""Pre-GO item 10 (2026-09-26): SGLang and LMDeploy reach the pod, each in its
own venv, and their launchers resolve the interpreter themselves.

Facts the pins rest on (PyPI metadata read 2026-09-26): vLLM 0.19.1 pins
torch 2.10.0 on the CUDA 12.8 runtime; every SGLang release from 0.5.11 pins a
CUDA 13 torch and transformers 5.x (the repo pins transformers<5), so the
engines cannot share cage-env; SGLang 0.5.10.post1 is the last release on the
CUDA 12 line; LMDeploy 0.17.0 accepts torch 2.0 to 2.12.1 and ships cp313
wheels, so its venv pins torch explicitly. TurboMind serves no FP8 weights.

What is pinned:
1. setup_runpod.sh declares the three overridable pins, equal to the section 7
   table of docs/VLLM_COMPATIBILITY.md, and creates both venvs through each
   venv's own pip, deleting nothing; lmdeploy-env comes from the canonical
   interpreter, sglang-env from SGLANG_PYTHON_VERSION (ADR-0120, 2026-09-27:
   outlines_core 0.1.26 has no cp313 wheel; see tests/test_w29_w30_live_fixes.py).
2. .gitignore covers both venvs.
3. Both launchers resolve CAGE_SGLANG_PYTHON / CAGE_LMDEPLOY_BIN (absolute
   path or a bare name through command -v), default to the venvs, fall back
   to PATH, and gate BEFORE any server action: an executable FILE, and for
   SGLang one that imports the package (review 2026-09-26 MEDIUM 1: the
   fallback used to reach the launch line and burn the readiness wait).
4. Behavior under real bash with a stubbed PATH: a bogus or directory
   override refuses with the variable named and nothing torn down; a missing
   venv with no importable sglang refuses the same way; a real override, or a
   bare name on PATH, reaches the composed argv and the launch line.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SETUP = REPO_ROOT / "scripts" / "runpod" / "setup_runpod.sh"
SGLANG_SH = REPO_ROOT / "scripts" / "2_serving" / "manage_sglang_server.sh"
LMDEPLOY_SH = REPO_ROOT / "scripts" / "2_serving" / "manage_lmdeploy_server.sh"
COMPAT = REPO_ROOT / "docs" / "VLLM_COMPATIBILITY.md"
GITIGNORE = REPO_ROOT / ".gitignore"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not on PATH")


def _pin(text: str, var: str) -> str:
    m = re.search(rf'^{var}="\$\{{{var}:-([^}}]+)\}}"$', text, re.M)
    assert m, f"setup_runpod.sh must declare an overridable {var}"
    return m.group(1)


def test_setup_declares_engine_pins_equal_to_section_7() -> None:
    text = SETUP.read_text(encoding="utf-8")
    sglang = _pin(text, "SGLANG_VERSION")
    lmdeploy = _pin(text, "LMDEPLOY_VERSION")
    torch = _pin(text, "LMDEPLOY_TORCH_VERSION")
    assert (sglang, lmdeploy, torch) == ("0.5.10.post1", "0.17.0", "2.10.0")
    doc = COMPAT.read_text(encoding="utf-8")
    m = re.search(r"^\| SGLang \| \*\*([^*]+)\*\*", doc, re.M)
    assert m and m.group(1) == sglang, "section 7 SGLang pin must equal the bootstrap default"
    m = re.search(r"^\| LMDeploy \| \*\*([^*]+)\*\*", doc, re.M)
    assert m and m.group(1) == lmdeploy, "section 7 LMDeploy pin must equal the bootstrap default"
    assert f"torch pinned to {torch}" in doc
    assert f"**SGLang {sglang}**" in doc and f"**LMDeploy-TurboMind {lmdeploy}**" in doc
    assert "(pin TBD)" not in doc.split("## 7.")[1].split("## 8.")[0]


def test_setup_creates_engine_venvs_from_the_canonical_interpreter_deleting_nothing() -> None:
    text = SETUP.read_text(encoding="utf-8")
    code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    # ADR-0120 (2026-09-27): the venv interpreter is the canonical PYBIN unless the
    # caller overrides it for ONE call (sglang-env); never a bare python3.
    assert 'local pybin="${ENGINE_PYBIN:-$PYBIN}"' in code
    assert '"$pybin" -m venv "$venv"' in code, "engine venvs come from PYBIN or the explicit per-call override"
    assert re.search(r"^\s*python3 -m venv", code, re.M) is None
    assert '"$venv/bin/pip" install --no-cache-dir "$@"' in code, (
        "install through the venv's own pip and without a pip cache; cage-env stays active"
    )
    assert 'install_engine_venv sglang-env "SGLang ${SGLANG_VERSION}" "sglang==${SGLANG_VERSION}"' in code
    assert '"torch==${LMDEPLOY_TORCH_VERSION}" "lmdeploy==${LMDEPLOY_VERSION}"' in code
    assert "rm -rf" not in code.split("install_engine_venv()")[1].split("# 4.")[0], (
        "the engine step deletes nothing: reruns reuse the venv and pip completes the install"
    )
    assert 'SKIP_ENGINE_INSTALL' in code
    for forbidden in ("sudo ", "systemctl", "add-apt-repository"):
        assert forbidden not in code


def test_gitignore_covers_both_engine_venvs() -> None:
    lines = GITIGNORE.read_text(encoding="utf-8").splitlines()
    assert "sglang-env/" in lines and "lmdeploy-env/" in lines and "cage-env/" in lines


def _gate_precedes_start_server(text: str, needle: str) -> None:
    assert text.index(needle) < text.index("start_server()"), (
        f"{needle!r} must be gated at top level, before start_server is defined"
    )


def test_sglang_launcher_resolves_its_interpreter() -> None:
    text = SGLANG_SH.read_text(encoding="utf-8")
    assert 'SGLANG_PYTHON="${CAGE_SGLANG_PYTHON:-}"' in text
    assert '"$PROJECT_DIR/sglang-env/bin/python3"' in text
    assert 'nohup "$SGLANG_PYTHON" -m sglang.launch_server "${sglang_args[@]}"' in text
    assert 'SC_ARGS="$SGLANG_PYTHON -m sglang.launch_server ${sglang_args[*]}"' in text
    assert re.search(r"^\s*nohup python3 -m sglang", text, re.M) is None
    _gate_precedes_start_server(text, '{ [ -f "$SGLANG_PYTHON" ] && [ -x "$SGLANG_PYTHON" ]; }')
    _gate_precedes_start_server(text, "\"$SGLANG_PYTHON\" -c 'import sglang'")
    assert 'command -v "$SGLANG_PYTHON"' in text, "a bare name must resolve through PATH"
    header = text.split("set -euo pipefail")[0]
    assert "CAGE_SGLANG_PYTHON" in header and "sglang-env" in header


def test_lmdeploy_launcher_resolves_its_entry_point() -> None:
    text = LMDEPLOY_SH.read_text(encoding="utf-8")
    assert 'LMDEPLOY_BIN="${CAGE_LMDEPLOY_BIN:-}"' in text
    assert '"$PROJECT_DIR/lmdeploy-env/bin/lmdeploy"' in text
    assert 'nohup "$LMDEPLOY_BIN" "${lmdeploy_args[@]}"' in text
    assert 'SC_ARGS="$LMDEPLOY_BIN ${lmdeploy_args[*]}"' in text
    assert re.search(r"^\s*nohup lmdeploy ", text, re.M) is None
    _gate_precedes_start_server(text, '{ [ -f "$LMDEPLOY_BIN" ] && [ -x "$LMDEPLOY_BIN" ]; }')
    assert 'command -v "$LMDEPLOY_BIN"' in text, "a bare name must resolve through PATH"
    # the gate is scoped to start|restart: stop and status are never blocked
    assert 'case "${1:-}" in\n    start|restart)\n        { [ -f "$LMDEPLOY_BIN" ]' in text
    header = text.split("set -euo pipefail")[0]
    assert "CAGE_LMDEPLOY_BIN" in header and "lmdeploy-env" in header and "FP8" in header


# ---------------------------------------------------------------------------
# behavior under real bash (stubbed PATH: no server, no GPU, no network)
# ---------------------------------------------------------------------------

def _clean_env(**extra: str) -> dict:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("CAGE_", "VLLM_", "SGLANG_", "LMDEPLOY_"))}
    env.update(extra)
    return env


@pytest.fixture(scope="module")
def stub_bin(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("stub_bin")
    for name, body in {
        "pgrep": "#!/bin/sh\nexit 1\n",
        "curl": "#!/bin/sh\nexit 1\n",
        "nvidia-smi": "#!/bin/sh\nexit 1\n",
        "pkill": "#!/bin/sh\nexit 0\n",
    }.items():
        p = d / name
        p.write_text(body, encoding="utf-8")
        p.chmod(0o755)
    # bare command names for the command -v resolution tests
    for name in ("sglang-py", "lmdeploy-x"):
        p = d / name
        p.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        p.chmod(0o755)
    return d


def _run(script: Path, stub_bin: Path, *args: str, **env_extra: str) -> subprocess.CompletedProcess:
    env = _clean_env(**env_extra)
    env["PATH"] = f"{stub_bin}:{env.get('PATH', '/usr/bin:/bin')}"
    env.setdefault("SGLANG_START_TIMEOUT", "0")
    env.setdefault("LMDEPLOY_START_TIMEOUT", "0")
    return subprocess.run(["bash", str(script), *args], capture_output=True, text=True, env=env, timeout=120)


def _fake_exe(tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    p.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    p.chmod(0o755)
    return p


def test_sglang_bogus_interpreter_refuses_before_any_server_action(
    stub_bin: Path, tmp_path: Path
) -> None:
    # a nonexistent path, and a DIRECTORY (passes -x alone; review LOW 2)
    for bogus in ("/nonexistent/python3", str(tmp_path)):
        for verb in ("start", "restart"):
            proc = _run(SGLANG_SH, stub_bin, verb, "fake/test-model", CAGE_SGLANG_PYTHON=bogus)
            assert proc.returncode != 0, bogus
            assert "CAGE_SGLANG_PYTHON" in proc.stderr and "FATAL" in proc.stderr
            assert "Stopping" not in proc.stdout and "Server args:" not in proc.stdout


def _path_python3_imports_sglang() -> bool:
    proc = subprocess.run(["python3", "-c", "import sglang"], capture_output=True)
    return proc.returncode == 0


@pytest.mark.skipif(
    (REPO_ROOT / "sglang-env" / "bin" / "python3").exists() or _path_python3_imports_sglang(),
    reason="an SGLang install is present (sglang-env or PATH python3): the fail-closed path is not reachable here",
)
def test_sglang_missing_venv_fails_closed_before_any_server_action(stub_bin: Path) -> None:
    """Review 2026-09-26 MEDIUM 1: with no sglang-env and no importable sglang,
    the launcher must refuse in the gate (naming setup step 3c), not fall back
    to the PATH python3, tear down a healthy server and burn the readiness wait."""
    for verb in ("start", "restart"):
        proc = _run(SGLANG_SH, stub_bin, verb, "fake/test-model")
        assert proc.returncode != 0
        assert "not importable" in proc.stderr and "step 3c" in proc.stderr
        assert "Stopping" not in proc.stdout and "Server args:" not in proc.stdout


def test_lmdeploy_bogus_entry_point_refuses_before_any_server_action(
    stub_bin: Path, tmp_path: Path
) -> None:
    # a nonexistent path, a DIRECTORY (review LOW 2), and a bare name absent from PATH
    for bogus in ("/nonexistent/lmdeploy", str(tmp_path), "lmdeploy-absent-cmd"):
        for verb in ("start", "restart"):
            proc = _run(LMDEPLOY_SH, stub_bin, verb, "fake/test-model",
                        CAGE_LMDEPLOY_BIN=bogus, LMDEPLOY_CACHE_MAX_ENTRY_COUNT="0.5")
            assert proc.returncode != 0, bogus
            assert "CAGE_LMDEPLOY_BIN" in proc.stderr and "FATAL" in proc.stderr
            assert "Stopping" not in proc.stdout and "Server args:" not in proc.stdout
    # `stop` is never gated: pinned on the source (the case block names
    # start|restart only) rather than run here, because a real stop removes
    # the repo's logs/lmdeploy pidfile (review LOW 5).


def test_sglang_explicit_interpreter_reaches_the_launch(stub_bin: Path, tmp_path: Path) -> None:
    fake = _fake_exe(tmp_path, "python3-sglang")
    proc = _run(SGLANG_SH, stub_bin, "start", "fake/test-model", CAGE_SGLANG_PYTHON=str(fake))
    assert f"Interpreter: {fake}" in proc.stdout
    lines = [l for l in proc.stdout.splitlines() if l.startswith("Server args:")]
    assert len(lines) == 1 and lines[0].startswith(f"Server args: {fake} -m sglang.launch_server ")


def test_bare_command_names_resolve_through_path(stub_bin: Path) -> None:
    """Review LOW 3: a bare name (e.g. python3.13) is resolved with command -v
    instead of being refused; the printed line shows the resolved path."""
    proc = _run(SGLANG_SH, stub_bin, "start", "fake/test-model", CAGE_SGLANG_PYTHON="sglang-py")
    assert f"Interpreter: {stub_bin / 'sglang-py'}" in proc.stdout
    lines = [l for l in proc.stdout.splitlines() if l.startswith("Server args:")]
    assert len(lines) == 1 and lines[0].startswith(f"Server args: {stub_bin / 'sglang-py'} -m sglang.launch_server ")
    proc = _run(LMDEPLOY_SH, stub_bin, "start", "fake/test-model",
                CAGE_LMDEPLOY_BIN="lmdeploy-x", LMDEPLOY_CACHE_MAX_ENTRY_COUNT="0.5")
    assert f"Entry point: {stub_bin / 'lmdeploy-x'}" in proc.stdout
    lines = [l for l in proc.stdout.splitlines() if l.startswith("Server args:")]
    assert len(lines) == 1 and lines[0].startswith(f"Server args: {stub_bin / 'lmdeploy-x'} serve api_server ")


def test_lmdeploy_explicit_entry_point_reaches_the_launch(stub_bin: Path, tmp_path: Path) -> None:
    fake = _fake_exe(tmp_path, "lmdeploy")
    proc = _run(LMDEPLOY_SH, stub_bin, "start", "fake/test-model",
                CAGE_LMDEPLOY_BIN=str(fake), LMDEPLOY_CACHE_MAX_ENTRY_COUNT="0.5")
    assert f"Entry point: {fake}" in proc.stdout
    lines = [l for l in proc.stdout.splitlines() if l.startswith("Server args:")]
    assert len(lines) == 1 and lines[0].startswith(f"Server args: {fake} serve api_server fake/test-model ")
