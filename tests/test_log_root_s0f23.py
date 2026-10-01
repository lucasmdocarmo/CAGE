"""The engine launchers honor CAGE_LOG_ROOT, and the suite points it at a tmp dir (S0F-23).

Finding (2026-10-01, the S0F-9 verifier): every launcher start writes its log
under ``<repo>/logs/<engine>/`` with no override, so the five test modules that
start a launcher with a fake engine leave one 0-byte log per start in the repo
(3,207 under logs/vllm, 79 under logs/sglang, 58 under logs/lmdeploy on the dev
Mac; one on the pod after S0-11). Gate (j)'s unpinned discovery picks the newest
file by mtime, so an unpinned preflight after the suite read an empty log and
failed ``NO recognizable KV-pool line``.

Pinned here:
1. the four shell launchers derive ``LOG_DIR`` from
   ``${CAGE_LOG_ROOT:-$PROJECT_DIR/logs}/<engine>`` (default unchanged);
2. the session fixture in tests/conftest.py exports ``CAGE_LOG_ROOT`` to a tmp
   dir outside the repo, and every launcher test's ``_clean_env`` helper carries
   it through its ``CAGE_`` strip (a conftest export alone would be stripped);
3. a real launcher start under the override writes its log there and nothing
   new under the repo's logs/.
Gate (j)'s side (skip 0-byte files, fall back to CAGE_LOG_ROOT) is pinned in
tests/test_preflight_gates.py beside the other gate (j) tests.
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
SERVING = REPO_ROOT / "scripts" / "2_serving"
LAUNCHERS = {
    "manage_vllm_server.sh": "vllm",
    "manage_sglang_server.sh": "sglang",
    "manage_lmdeploy_server.sh": "lmdeploy",
    "manage_vllm_pd.sh": "vllm",
}
LAUNCHER_TEST_MODULES = (
    "test_tp_flags.py",
    "test_serving_budget_knobs.py",
    "test_pd_launcher.py",
    "test_engine_venvs_item10.py",
    "test_w29_w30_live_fixes.py",
)

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not on PATH")


@pytest.mark.parametrize("script,engine", sorted(LAUNCHERS.items()))
def test_every_launcher_derives_log_dir_from_cage_log_root(script: str, engine: str) -> None:
    text = (SERVING / script).read_text(encoding="utf-8")
    pattern = r'^LOG_DIR="\$\{CAGE_LOG_ROOT:-\$PROJECT_DIR/logs\}/' + engine + r'"$'
    assert re.search(pattern, text, re.M), (
        f"{script}: LOG_DIR must be ${{CAGE_LOG_ROOT:-$PROJECT_DIR/logs}}/{engine} (S0F-23)")
    assert re.search(r'^LOG_DIR="\$PROJECT_DIR/logs/', text, re.M) is None, (
        f"{script}: the unconditional repo log dir must be gone")


def test_session_exports_cage_log_root_outside_the_repo() -> None:
    root = os.environ.get("CAGE_LOG_ROOT")
    assert root, "tests/conftest.py must export CAGE_LOG_ROOT for the whole session"
    path = Path(root)
    assert path.is_absolute() and path.is_dir()
    assert REPO_ROOT not in path.parents and path != REPO_ROOT, (
        "the suite's log root must live outside the repository")


@pytest.mark.parametrize("module_name", LAUNCHER_TEST_MODULES)
def test_launcher_test_env_helpers_carry_the_log_root(module_name: str, monkeypatch) -> None:
    spec = importlib.util.spec_from_file_location(
        f"s0f23_{module_name[:-3]}", REPO_ROOT / "tests" / module_name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        monkeypatch.setenv("CAGE_LOG_ROOT", "/tmp/s0f23-probe-root")
        monkeypatch.setenv("CAGE_KV_BUDGET_BYTES", "1")  # a stripped CAGE_ var, for contrast
        env = module._clean_env()
    finally:
        sys.modules.pop(spec.name, None)
    assert env.get("CAGE_LOG_ROOT") == "/tmp/s0f23-probe-root", (
        f"{module_name}: _clean_env must carry CAGE_LOG_ROOT through its CAGE_ strip")
    assert "CAGE_KV_BUDGET_BYTES" not in env, f"{module_name}: the strip itself must stay"


@pytest.fixture()
def stub_bin(tmp_path: Path) -> Path:
    d = tmp_path / "stub_bin"
    d.mkdir()
    for name, body in {
        "pgrep": "#!/bin/sh\nexit 1\n",
        "curl": "#!/bin/sh\nexit 1\n",
        "nvidia-smi": "#!/bin/sh\nexit 1\n",
        "pkill": "#!/bin/sh\nexit 0\n",
        "vllm": "#!/bin/sh\nexit 0\n",
    }.items():
        p = d / name
        p.write_text(body, encoding="utf-8")
        p.chmod(0o755)
    return d


def _repo_log_listing() -> set:
    d = REPO_ROOT / "logs" / "vllm"
    return {p.name for p in d.iterdir()} if d.is_dir() else set()


def test_vllm_launcher_writes_under_the_override_and_nothing_in_the_repo(
        stub_bin: Path, tmp_path: Path) -> None:
    log_root = tmp_path / "logroot"
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("CAGE_", "VLLM_", "SGLANG_"))}
    env["PATH"] = f"{stub_bin}:{env.get('PATH', '/usr/bin:/bin')}"
    env["CAGE_LOG_ROOT"] = str(log_root)
    env["VLLM_START_TIMEOUT"] = "0"
    before = _repo_log_listing()
    proc = subprocess.run(
        ["bash", str(SERVING / "manage_vllm_server.sh"), "start", "fake/test-model"],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert "Server args:" in proc.stdout, proc.stdout + proc.stderr
    written = sorted((log_root / "vllm").glob("vllm_fake_test-model_*.log"))
    assert len(written) == 1, f"expected one start log under {log_root}/vllm, got {written}"
    assert (log_root / "vllm" / "vllm_server.pid").is_file()
    assert f"logging to {written[0]}" in proc.stdout
    assert _repo_log_listing() == before, "a start under CAGE_LOG_ROOT must leave the repo's logs/ alone"
