"""W29 and W30: the two S0 blockers the live D box found on an L40S (2026-09-27).

W29. vLLM 0.19.1 knows only ``--enable-log-requests`` (vllm/entrypoints/openai/
cli_args.py at tag v0.19.1, read 2026-09-27) and rejects ``--disable-log-requests``
with "unrecognized arguments". The server exited at once and the launcher waited
its whole readiness timeout (300 s) before failing. Pinned here: none of the three
vLLM launch paths (single launcher, pd launcher, pilot cluster router) ever passes
the removed flag; the default argv carries NO log flag; VLLM_DISABLE_LOG_REQUESTS=0
(keep per-request logs for debugging) adds ``--enable-log-requests`` and changes
nothing else; empty and "1" behave like unset.

W30. SGLang 0.5.10.post1 pins outlines 0.1.11, which pins outlines_core 0.1.26, and
that release ships no cp313 wheel (PyPI read 2026-09-27; cp312 exists), so the
sglang-env install on the canonical CPython 3.13 fell to a source build with no Rust
in the image. ADR-0120: sglang-env is created on CPython SGLANG_PYTHON_VERSION
(default 3.12) through a per-call interpreter override; cage-env and lmdeploy-env
stay on the canonical interpreter; a miss skips SGLang loudly, never silently.

No GPU, engine or network: real bash with a stubbed PATH (test_tp_flags.py style),
readiness timeouts at 0 so the launchers echo the composed argv and fail fast.
"""
from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VLLM_SH = REPO_ROOT / "scripts" / "2_serving" / "manage_vllm_server.sh"
PD_SH = REPO_ROOT / "scripts" / "2_serving" / "manage_vllm_pd.sh"
CLUSTER_PY = REPO_ROOT / "scripts" / "2_serving" / "manage_vllm_cluster.py"
SETUP = REPO_ROOT / "scripts" / "runpod" / "setup_runpod.sh"
COMPAT = REPO_ROOT / "docs" / "VLLM_COMPATIBILITY.md"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not on PATH")

REMOVED = "--disable-log-requests"
ENABLE = "--enable-log-requests"


def _code_lines(text: str) -> str:
    return "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))


# ---------------------------------------------------------------------------
# W29: static pins on the three launch paths
# ---------------------------------------------------------------------------

def test_no_vllm_launch_path_passes_the_removed_flag() -> None:
    for path in (VLLM_SH, PD_SH, CLUSTER_PY):
        code = _code_lines(path.read_text(encoding="utf-8"))
        assert REMOVED not in code, f"{path.name} still passes {REMOVED} (rejected by vLLM 0.19.1)"
        assert ENABLE in code, f"{path.name} must offer {ENABLE} for VLLM_DISABLE_LOG_REQUESTS=0"
    # the gate is the value "0" and nothing else, on every path
    assert 'if [ "${VLLM_DISABLE_LOG_REQUESTS:-1}" = "0" ]; then' in VLLM_SH.read_text(encoding="utf-8")
    assert 'if [ "${VLLM_DISABLE_LOG_REQUESTS:-1}" = "0" ]; then' in PD_SH.read_text(encoding="utf-8")
    assert 'os.environ.get("VLLM_DISABLE_LOG_REQUESTS", "1") == "0"' in CLUSTER_PY.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# W29: behavior under real bash (stubbed PATH, TIMEOUT=0)
# ---------------------------------------------------------------------------

def _clean_env(**extra: str) -> dict:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("CAGE_", "VLLM_", "SGLANG_", "LMDEPLOY_"))}
    # S0F-23: the suite's tmp log root (tests/conftest.py) rides through the
    # CAGE_ strip so a launcher start never writes under <repo>/logs/.
    if "CAGE_LOG_ROOT" in os.environ:
        env["CAGE_LOG_ROOT"] = os.environ["CAGE_LOG_ROOT"]
    env.update(extra)
    return env


@pytest.fixture(scope="module")
def stub_bin(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """No live server (pgrep/curl fail), no GPU (nvidia-smi fails), an inert
    `vllm` so the backgrounded launch exits, and an inert `pkill` so the pd
    launcher's self-cleaning stop signals nothing on this host."""
    d = tmp_path_factory.mktemp("stub_bin")
    for name, body in {
        "pgrep": "#!/bin/sh\nexit 1\n",
        "curl": "#!/bin/sh\nexit 1\n",
        "nvidia-smi": "#!/bin/sh\nexit 1\n",
        "pkill": "#!/bin/sh\nexit 0\n",
        "vllm": "#!/bin/sh\nexit 0\n",
        # ADR-0128: the pd launcher's nixl import gate probes CAGE_PD_PYTHON
        # before composing anything; this host has no nixl, so the stub answers ok.
        "pd_python_stub": "#!/bin/sh\ncat >/dev/null\necho ok\nexit 0\n",
    }.items():
        p = d / name
        p.write_text(body, encoding="utf-8")
        p.chmod(0o755)
    return d


def _run(script: Path, stub_bin: Path, *args: str, **env_extra: str) -> subprocess.CompletedProcess:
    env = _clean_env(**env_extra)
    env["PATH"] = f"{stub_bin}:{env.get('PATH', '/usr/bin:/bin')}"
    env.setdefault("VLLM_START_TIMEOUT", "0")
    env.setdefault("CAGE_PD_PYTHON", str(stub_bin / "pd_python_stub"))
    return subprocess.run(["bash", str(script), *args], capture_output=True, text=True, env=env, timeout=120)


def _args_line(stdout: str, prefix: str) -> str:
    lines = [ln for ln in stdout.splitlines() if ln.startswith(prefix)]
    assert len(lines) == 1, f"expected exactly one {prefix!r} line, got:\n{stdout}"
    return lines[0]


def _without(tokens: list, token: str) -> list:
    return [t for t in tokens if t != token]


def test_single_launcher_default_has_no_log_flag_and_zero_adds_exactly_one_token(stub_bin: Path) -> None:
    base = _args_line(_run(VLLM_SH, stub_bin, "start", "fake/test-model").stdout, "Server args:")
    assert REMOVED not in base and ENABLE not in base
    on = _args_line(
        _run(VLLM_SH, stub_bin, "start", "fake/test-model", VLLM_DISABLE_LOG_REQUESTS="0").stdout,
        "Server args:",
    )
    assert ENABLE in on and REMOVED not in on
    assert _without(on.split(), ENABLE) == base.split(), "the env must add ONE token and change nothing else"
    # position, not just presence: right after --trust-remote-code, where the removed flag sat
    assert on.split().index(ENABLE) == base.split().index("--trust-remote-code") + 1
    for like_unset in ("", "1"):
        same = _args_line(
            _run(VLLM_SH, stub_bin, "start", "fake/test-model", VLLM_DISABLE_LOG_REQUESTS=like_unset).stdout,
            "Server args:",
        )
        assert same == base, f"VLLM_DISABLE_LOG_REQUESTS={like_unset!r} must behave like unset"


def test_pd_launcher_both_roles_follow_the_same_contract(stub_bin: Path) -> None:
    pd_env = dict(CAGE_KV_BUDGET_BYTES_PREFILL="3000000000", CAGE_KV_BUDGET_BYTES_DECODE="7000000000")
    base = _run(PD_SH, stub_bin, "start", "fake/test-model", **pd_env).stdout
    on = _run(PD_SH, stub_bin, "start", "fake/test-model", VLLM_DISABLE_LOG_REQUESTS="0", **pd_env).stdout
    for role in ("prefill", "decode"):
        prefix = f"Server args [{role}]:"
        base_line = _args_line(base, prefix)
        on_line = _args_line(on, prefix)
        assert REMOVED not in base_line and ENABLE not in base_line, role
        assert ENABLE in on_line and REMOVED not in on_line, role
        assert _without(on_line.split(), ENABLE) == base_line.split(), role
        assert on_line.split().index(ENABLE) == base_line.split().index("--trust-remote-code") + 1, role


def _load_cluster_module():
    # Registered in sys.modules BEFORE exec: the module's frozen dataclass looks
    # its own module up there while the class is being built (Python 3.13).
    name = "manage_vllm_cluster_w29"
    spec = importlib.util.spec_from_file_location(name, CLUSTER_PY)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_cluster_router_serve_args_follow_the_same_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_cluster_module()
    monkeypatch.delenv("VLLM_DISABLE_LOG_REQUESTS", raising=False)
    base = mod.build_serve_args("fake/test-model", 8001, gpu_memory_utilization="0.90")
    assert REMOVED not in base and ENABLE not in base
    monkeypatch.setenv("VLLM_DISABLE_LOG_REQUESTS", "0")
    on = mod.build_serve_args("fake/test-model", 8001, gpu_memory_utilization="0.90")
    assert ENABLE in on and REMOVED not in on
    assert _without(on, ENABLE) == base
    assert on[-1] == ENABLE, "the cluster builder appends the flag last, where the removed one sat"
    for like_unset in ("1", ""):
        monkeypatch.setenv("VLLM_DISABLE_LOG_REQUESTS", like_unset)
        assert mod.build_serve_args("fake/test-model", 8001, gpu_memory_utilization="0.90") == base, like_unset


# ---------------------------------------------------------------------------
# W30: sglang-env on its own interpreter, everything else canonical (ADR-0120)
# ---------------------------------------------------------------------------

def test_setup_builds_sglang_env_on_its_own_interpreter_and_the_rest_canonical() -> None:
    text = SETUP.read_text(encoding="utf-8")
    code = _code_lines(text)
    assert re.search(r'^SGLANG_PYTHON_VERSION="\$\{SGLANG_PYTHON_VERSION:-3\.12\}"$', text, re.M), (
        "setup_runpod.sh must declare an overridable SGLANG_PYTHON_VERSION defaulting to 3.12"
    )
    assert 'local pybin="${ENGINE_PYBIN:-$PYBIN}"' in code
    assert '"$pybin" -m venv "$venv"' in code
    assert re.search(r"^\s*python3 -m venv", code, re.M) is None, "never a bare python3 venv (finding B1)"
    sglang_calls = [l for l in code.splitlines() if "install_engine_venv sglang-env" in l]
    assert len(sglang_calls) == 1 and 'ENGINE_PYBIN="$SGLANG_PYBIN"' in sglang_calls[0], (
        "the SGLang venv is the ONE call with an interpreter override"
    )
    lmdeploy_calls = [l for l in code.splitlines() if "install_engine_venv lmdeploy-env" in l]
    assert len(lmdeploy_calls) == 1 and "ENGINE_PYBIN" not in lmdeploy_calls[0], (
        "lmdeploy-env stays on the canonical interpreter"
    )
    assert 'SGLANG_PYBIN="python${SGLANG_PYTHON_VERSION}"' in code
    assert "ADR-0120" in text
    # the miss is loud and names the fix, never a silent skip
    assert "SKIPPED: no CPython ${SGLANG_PYTHON_VERSION}" in text
    # an ambient ENGINE_PYBIN never reaches the calls (review LOW 3)
    assert code.index("unset ENGINE_PYBIN") < code.index("install_engine_venv sglang-env")
    # the uv fallback never installs into the activated cage-env (review LOW 2)
    assert "( deactivate >/dev/null 2>&1; python3 -m pip install --quiet uv )" in code
    # header documents the knob beside the pins it belongs to
    header = text.split("set -euo pipefail")[0]
    assert "SGLANG_PYTHON_VERSION" in header and "SGLANG_VERSION" in header


def _extract_function(text: str, name: str) -> str:
    m = re.search(rf"^{name}\(\) \{{\n.*?^\}}$", text, re.M | re.S)
    assert m, f"{name}() not found in setup_runpod.sh"
    return m.group(0)


def _stub(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


def _run_install_engine_venv(tmp_path: Path, venv_python_version: str, pybin_version: str) -> subprocess.CompletedProcess:
    """Drive install_engine_venv() under real bash against a FAKE existing venv and a
    FAKE interpreter: no real venv is created, no pip runs, nothing is deleted."""
    fn = _extract_function(SETUP.read_text(encoding="utf-8"), "install_engine_venv")
    venv = tmp_path / "sglang-env"
    (venv / "pyvenv.cfg").parent.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /fake\n", encoding="utf-8")
    _stub(venv / "bin" / "python3", f"#!/bin/sh\necho {venv_python_version}\n")
    _stub(venv / "bin" / "pip", "#!/bin/sh\nexit 0\n")
    pybin = _stub(tmp_path / "pybin" / "python-fake", f'#!/bin/sh\ncase "$1" in -c) echo {pybin_version};; esac\nexit 0\n')
    harness = tmp_path / "harness.sh"
    harness.write_text(
        # no -e: the refusal path RETURNS 1 and the harness must still print rc
        "set -uo pipefail\n"
        'warn() { echo "[cage] WARNING: $*" >&2; }\n'
        f'PYBIN="{pybin}"\n'
        f"{fn}\n"
        f'install_engine_venv "{venv}" "SGLang test" "pkg==1"\n'
        'echo "rc=$?"\n',
        encoding="utf-8",
    )
    return subprocess.run(["bash", str(harness)], capture_output=True, text=True, timeout=60)


def test_install_engine_venv_refuses_to_recreate_a_venv_on_another_cpython(tmp_path: Path) -> None:
    """Review 2026-09-27 MEDIUM 1: `venv` without --clear keeps the old bin/python links
    and only rewrites pyvenv.cfg, so re-creating over a 3.13 sglang-env with 3.12
    would leave a mixed tree. The function must refuse loudly, name the versions and
    the operator action, delete nothing, and never reach the venv/pip steps."""
    proc = _run_install_engine_venv(tmp_path, venv_python_version="3.13", pybin_version="3.12")
    assert "rc=1" in proc.stdout, proc.stdout + proc.stderr
    assert "refusing to re-create it in place" in proc.stderr
    assert "CPython 3.13" in proc.stderr and "CPython 3.12" in proc.stderr
    assert "Remove" in proc.stderr and "by hand" in proc.stderr
    assert "installing SGLang test" not in proc.stdout
    # nothing deleted, nothing rewritten
    assert (tmp_path / "sglang-env" / "pyvenv.cfg").read_text(encoding="utf-8") == "home = /fake\n"


def test_install_engine_venv_reuses_a_venv_on_the_same_cpython(tmp_path: Path) -> None:
    """The idempotent rerun path stays: same CPython, the function proceeds (fake venv
    and fake pip both exit 0) and reports the install."""
    proc = _run_install_engine_venv(tmp_path, venv_python_version="3.12", pybin_version="3.12")
    assert "rc=0" in proc.stdout, proc.stdout + proc.stderr
    assert "installing SGLang test into" in proc.stdout
    assert "refusing" not in proc.stderr


def test_compatibility_doc_records_both_live_facts() -> None:
    doc = COMPAT.read_text(encoding="utf-8")
    assert ENABLE in doc and "unrecognized arguments" in doc and REMOVED in doc, (
        "section 2 must record that 0.19.1 rejects the removed flag and what replaces it"
    )
    m = re.search(r"^\| SGLang \| \*\*0\.5\.10\.post1\*\*.*$", doc, re.M)
    assert m, "section 7 SGLang row missing"
    row = m.group(0)
    assert "CPython 3.12" in row and "ADR-0120" in row and "outlines_core" in row
