"""T2.1 pins: CacheBudgetPlanner budget knobs wired into the serving launchers.

WHAT is pinned and WHY (charter P2: pressure = BYTES of cache budget, the same
physical quantity on every engine; src/orchestration/cache_budget.py plans the
budget and emits per-engine knobs, but before T2.1 NOTHING consumed a plan --
the launchers were fraction-only):

1. Env contract (static): manage_vllm_server.sh conditionally passes
   ``--kv-cache-memory-bytes`` (CAGE_KV_BUDGET_BYTES, primary) or
   ``--num-gpu-blocks-override`` (CAGE_VLLM_GPU_BLOCKS_OVERRIDE, fallback);
   manage_sglang_server.sh passes ``--max-total-tokens``
   (CAGE_SGLANG_MAX_TOTAL_TOKENS). Each launcher validates via the shared
   helpers in scripts/lib/_serving_config.sh BEFORE touching any server.

2. Refusal paths (behavioral, real bash): both-vllm-knobs-set and every
   non-positive-integer value refuse LOUDLY, with a nonzero exit, before any
   server is stopped or launched. Fail-closed doctrine: a malformed budget
   must never degrade to fraction-only serving, because the run's data would
   then be labeled with a budget the engine never had.

3. Default path (behavioral): with none of the knob vars set, the composed
   server command line carries NONE of the new flags -- byte-identical default
   behavior is the contract (pre-T2.1 launches must not change).

4. Observability: the per-(re)start serving-config capture records the knobs,
   both structurally (kv_budget_bytes / num_gpu_blocks_override /
   max_total_tokens fields) and via the full captured command line (SC_ARGS
   embeds the launch argv, so the flags ride along automatically).

No GPU / engine / network: engine binaries and process probes are stubbed on
PATH, and the launchers run with *_START_TIMEOUT=0 so they fail their
readiness wait AFTER echoing the composed argv -- which is what we assert.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VLLM_SH = REPO_ROOT / "scripts" / "2_serving" / "manage_vllm_server.sh"
SGLANG_SH = REPO_ROOT / "scripts" / "2_serving" / "manage_sglang_server.sh"
SERVING_CONFIG = REPO_ROOT / "scripts" / "lib" / "_serving_config.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not on PATH")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _clean_env(**extra: str) -> dict:
    """Subprocess env with every CAGE_/VLLM_/SGLANG_ var stripped so a dev
    shell's exports can never leak into the pinned default path."""
    env = {
        k: v for k, v in os.environ.items()
        if not k.startswith(("CAGE_", "VLLM_", "SGLANG_"))
    }
    env.update(extra)
    return env


@pytest.fixture(scope="module")
def stub_bin(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Hermetic PATH stubs: no live server can be found (pgrep/curl fail), no
    GPU is touched (nvidia-smi fails), and `vllm` exists-but-exits so the
    backgrounded launch is inert. python3 stays REAL (JSON capture needs it)."""
    d = tmp_path_factory.mktemp("stub_bin")
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


def _run_launcher(script: Path, stub_bin: Path, *args: str, **env_extra: str):
    env = _clean_env(**env_extra)
    env["PATH"] = f"{stub_bin}:{env.get('PATH', '/usr/bin:/bin')}"
    # TIMEOUT=0 skips the readiness wait entirely: the launcher echoes the
    # composed argv, backgrounds the inert stub, then fails fast (exit 1).
    env.setdefault("VLLM_START_TIMEOUT", "0")
    env.setdefault("SGLANG_START_TIMEOUT", "0")
    return subprocess.run(
        ["bash", str(script), *args],
        capture_output=True, text=True, env=env, timeout=120,
    )


def _server_args_line(stdout: str) -> str:
    lines = [ln for ln in stdout.splitlines() if ln.startswith("Server args:")]
    assert len(lines) == 1, (
        f"expected exactly one 'Server args:' line in launcher stdout, got:\n{stdout}"
    )
    return lines[0]


def _validate(func_call: str, **env_extra: str):
    """Run one _serving_config.sh validator under real bash with a fake env."""
    return subprocess.run(
        ["bash", "-c", f'source "$1" >/dev/null 2>&1 && {func_call}',
         "bash", str(SERVING_CONFIG)],
        capture_output=True, text=True, env=_clean_env(**env_extra), timeout=30,
    )


# ---------------------------------------------------------------------------
# 1. static: env-conditional flag blocks + refusal gates + parse
# ---------------------------------------------------------------------------

def test_launchers_and_lib_parse_with_bash_n() -> None:
    for script in (VLLM_SH, SGLANG_SH, SERVING_CONFIG):
        proc = subprocess.run(
            ["bash", "-n", str(script)], capture_output=True, text=True, timeout=30
        )
        assert proc.returncode == 0, f"{script.name}: {proc.stderr}"


def test_vllm_launcher_has_env_conditional_flag_blocks() -> None:
    text = VLLM_SH.read_text(encoding="utf-8")
    assert re.search(
        r'if \[ -n "\$\{CAGE_KV_BUDGET_BYTES:-\}" \]; then\s*\n'
        r'\s*vllm_args\+=\( --kv-cache-memory-bytes "\$\{CAGE_KV_BUDGET_BYTES\}" \)',
        text,
    ), "primary bytes knob must be env-conditional on CAGE_KV_BUDGET_BYTES"
    assert re.search(
        r'if \[ -n "\$\{CAGE_VLLM_GPU_BLOCKS_OVERRIDE:-\}" \]; then\s*\n'
        r'\s*vllm_args\+=\( --num-gpu-blocks-override "\$\{CAGE_VLLM_GPU_BLOCKS_OVERRIDE\}" \)',
        text,
    ), "fallback blocks knob must be env-conditional on CAGE_VLLM_GPU_BLOCKS_OVERRIDE"
    # The fraction dial must STILL ship alongside the bytes knob (the bytes
    # knob is the binding cap, not a replacement for the uniform regime).
    assert "--gpu-memory-utilization" in text


def test_sglang_launcher_has_env_conditional_flag_block() -> None:
    text = SGLANG_SH.read_text(encoding="utf-8")
    assert re.search(
        r'if \[ -n "\$\{CAGE_SGLANG_MAX_TOTAL_TOKENS:-\}" \]; then\s*\n'
        r'\s*sglang_args\+=\( --max-total-tokens "\$\{CAGE_SGLANG_MAX_TOTAL_TOKENS\}" \)',
        text,
    ), "token-cap knob must be env-conditional on CAGE_SGLANG_MAX_TOTAL_TOKENS"
    assert "--mem-fraction-static" in text


def test_refusal_gates_fire_before_any_server_action() -> None:
    """The validator call must sit at top level (before start_server is even
    defined) and be wired to die: on `restart` a malformed budget must not
    tear down the healthy server it would fail to replace."""
    for script, validator in (
        (VLLM_SH, "cage_validate_vllm_budget_env"),
        (SGLANG_SH, "cage_validate_sglang_budget_env"),
    ):
        text = script.read_text(encoding="utf-8")
        m = re.search(rf"{validator}\s*\\\n\s*\|\| die ", text)
        assert m, f"{script.name}: {validator} must be wired to `|| die`"
        assert m.start() < text.index("start_server()"), (
            f"{script.name}: the refusal gate must run BEFORE any server-"
            "touching code, not inside start_server"
        )


def test_launcher_headers_document_the_env_contract() -> None:
    """The env contract lives in each script's header so an operator reading
    the launcher (the documented entry point) sees the knobs without spelunking."""
    vllm_header = VLLM_SH.read_text(encoding="utf-8").split("set -euo pipefail")[0]
    assert "CAGE_KV_BUDGET_BYTES" in vllm_header
    assert "CAGE_VLLM_GPU_BLOCKS_OVERRIDE" in vllm_header
    assert "VERIFY-LIVE at S0-19" in vllm_header, (
        "the bytes knob is unproven against the pinned vLLM until S0-19; the "
        "header must carry the VERIFY-LIVE marker"
    )
    sglang_header = SGLANG_SH.read_text(encoding="utf-8").split("set -euo pipefail")[0]
    assert "CAGE_SGLANG_MAX_TOTAL_TOKENS" in sglang_header


def test_budget_knobs_land_in_serving_config_capture() -> None:
    """Per-restart observability capture must record the knobs: structurally
    (typed fields) AND via the full command line (SC_ARGS embeds the argv
    array, so the conditional flags ride along automatically)."""
    vllm = VLLM_SH.read_text(encoding="utf-8")
    assert 'SC_KV_BUDGET_BYTES="${CAGE_KV_BUDGET_BYTES:-}"' in vllm
    assert 'SC_BLOCKS_OVERRIDE="${CAGE_VLLM_GPU_BLOCKS_OVERRIDE:-}"' in vllm
    assert '"kv_budget_bytes"' in vllm
    assert '"num_gpu_blocks_override"' in vllm
    assert 'SC_ARGS="vllm serve $model ${vllm_args[*]}"' in vllm

    sglang = SGLANG_SH.read_text(encoding="utf-8")
    assert 'SC_MAX_TOTAL_TOKENS="${CAGE_SGLANG_MAX_TOTAL_TOKENS:-}"' in sglang
    assert '"max_total_tokens"' in sglang
    assert 'SC_ARGS="python3 -m sglang.launch_server ${sglang_args[*]}"' in sglang


def test_reuse_dial_parity_covers_the_budget_knobs() -> None:
    """A pressure-sweep iteration invoked via `start` must never reuse a
    server launched under a DIFFERENT budget (or none): the budget knobs are
    part of the dials_match cmdline comparison in both launchers."""
    vllm = VLLM_SH.read_text(encoding="utf-8")
    reuse = vllm[vllm.index("dials_match=true"):vllm.index("Start server")]
    assert "--kv-cache-memory-bytes" in reuse
    assert "--num-gpu-blocks-override" in reuse
    sglang = SGLANG_SH.read_text(encoding="utf-8")
    reuse = sglang[sglang.index("dials_match=true"):sglang.index("local timestamp log_file")]
    assert "--max-total-tokens" in reuse


# ---------------------------------------------------------------------------
# 2. behavioral: validator refusal + acceptance under real bash
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["abc", "12.5", "-5", "0", "+7", "1e9", " 42", "0x10", "00"])
def test_validators_refuse_non_positive_integers(bad: str) -> None:
    for func, var in (
        ("cage_validate_vllm_budget_env", "CAGE_KV_BUDGET_BYTES"),
        ("cage_validate_vllm_budget_env", "CAGE_VLLM_GPU_BLOCKS_OVERRIDE"),
        ("cage_validate_sglang_budget_env", "CAGE_SGLANG_MAX_TOTAL_TOKENS"),
    ):
        proc = _validate(func, **{var: bad})
        assert proc.returncode != 0, f"{func} accepted {var}={bad!r}"
        assert "REFUSING" in proc.stderr, (
            f"refusal must be LOUD (stderr), got: {proc.stderr!r}"
        )
        assert var in proc.stderr, "the refusal must name the offending variable"


@pytest.mark.parametrize("good", ["1", "147456", "23456789012"])
def test_validators_accept_positive_integers(good: str) -> None:
    # 23456789012 > 2^31: byte budgets routinely exceed 32-bit range and the
    # validator's pure-string logic must not care.
    for func, var in (
        ("cage_validate_vllm_budget_env", "CAGE_KV_BUDGET_BYTES"),
        ("cage_validate_vllm_budget_env", "CAGE_VLLM_GPU_BLOCKS_OVERRIDE"),
        ("cage_validate_sglang_budget_env", "CAGE_SGLANG_MAX_TOTAL_TOKENS"),
    ):
        proc = _validate(func, **{var: good})
        assert proc.returncode == 0, f"{func} refused valid {var}={good}: {proc.stderr}"


def test_validator_refuses_both_vllm_knobs_set() -> None:
    proc = _validate(
        "cage_validate_vllm_budget_env",
        CAGE_KV_BUDGET_BYTES="1000",
        CAGE_VLLM_GPU_BLOCKS_OVERRIDE="10",
    )
    assert proc.returncode != 0
    assert "BOTH set" in proc.stderr, (
        "both-knobs-set must refuse with an explanation, never pick a precedence"
    )


def test_validators_pass_when_no_knob_is_set() -> None:
    for func in ("cage_validate_vllm_budget_env", "cage_validate_sglang_budget_env"):
        proc = _validate(func)
        assert proc.returncode == 0, f"{func} must be a no-op with no knobs set"


# ---------------------------------------------------------------------------
# 3. behavioral: full launcher runs (stubbed PATH, TIMEOUT=0)
# ---------------------------------------------------------------------------

def test_vllm_launcher_refuses_both_knobs_before_touching_anything(stub_bin: Path) -> None:
    proc = _run_launcher(
        VLLM_SH, stub_bin, "start", "fake/test-model",
        CAGE_KV_BUDGET_BYTES="1000", CAGE_VLLM_GPU_BLOCKS_OVERRIDE="10",
    )
    assert proc.returncode != 0
    assert "FATAL" in proc.stderr and "REFUSING" in proc.stderr
    assert "Server args:" not in proc.stdout, (
        "refusal must happen BEFORE any launch attempt is composed"
    )


def test_sglang_launcher_refuses_non_integer_token_cap(stub_bin: Path) -> None:
    proc = _run_launcher(
        SGLANG_SH, stub_bin, "start", "fake/test-model",
        CAGE_SGLANG_MAX_TOTAL_TOKENS="12.5",
    )
    assert proc.returncode != 0
    assert "FATAL" in proc.stderr and "REFUSING" in proc.stderr
    assert "Server args:" not in proc.stdout


def test_vllm_default_path_composes_no_budget_flags(stub_bin: Path) -> None:
    """Byte-identical default behavior is the contract: with no knob env set,
    the composed argv must not mention any budget flag."""
    proc = _run_launcher(VLLM_SH, stub_bin, "start", "fake/test-model")
    line = _server_args_line(proc.stdout)
    assert "--kv-cache-memory-bytes" not in line
    assert "--num-gpu-blocks-override" not in line
    assert "--gpu-memory-utilization 0.90" in line, (
        "the uniform fraction dial must still be present on the default path"
    )


def test_sglang_default_path_composes_no_budget_flags(stub_bin: Path) -> None:
    proc = _run_launcher(SGLANG_SH, stub_bin, "start", "fake/test-model")
    line = _server_args_line(proc.stdout)
    assert "--max-total-tokens" not in line
    assert "--mem-fraction-static 0.90" in line


def test_vllm_bytes_knob_lands_on_cmdline_and_capture(
    stub_bin: Path, tmp_path: Path
) -> None:
    proc = _run_launcher(
        VLLM_SH, stub_bin, "start", "fake/test-model",
        CAGE_KV_BUDGET_BYTES="23456789012", CAGE_RUN_ROOT=str(tmp_path),
    )
    line = _server_args_line(proc.stdout)
    assert "--kv-cache-memory-bytes 23456789012" in line
    assert "--gpu-memory-utilization 0.90" in line, (
        "the fraction dial still ships; the bytes knob is the binding cap"
    )
    # Observability: the per-start capture records the knob, typed AND in argv.
    captures = list((tmp_path / "observability" / "serving_configs").glob("*.json"))
    assert len(captures) == 1, "expected exactly one serving-config capture"
    cfg = json.loads(captures[0].read_text(encoding="utf-8"))
    assert cfg["kv_budget_bytes"] == 23456789012
    assert cfg["num_gpu_blocks_override"] is None
    assert "--kv-cache-memory-bytes 23456789012" in cfg["args"]


def test_vllm_blocks_override_lands_on_cmdline(stub_bin: Path) -> None:
    proc = _run_launcher(
        VLLM_SH, stub_bin, "start", "fake/test-model",
        CAGE_VLLM_GPU_BLOCKS_OVERRIDE="9216",
    )
    line = _server_args_line(proc.stdout)
    assert "--num-gpu-blocks-override 9216" in line
    assert "--kv-cache-memory-bytes" not in line, (
        "blocks override is the INSTEAD-OF fallback, never additive"
    )


def test_sglang_token_cap_lands_on_cmdline_and_capture(
    stub_bin: Path, tmp_path: Path
) -> None:
    proc = _run_launcher(
        SGLANG_SH, stub_bin, "start", "fake/test-model",
        CAGE_SGLANG_MAX_TOTAL_TOKENS="147456", CAGE_RUN_ROOT=str(tmp_path),
    )
    line = _server_args_line(proc.stdout)
    assert "--max-total-tokens 147456" in line
    assert "--mem-fraction-static 0.90" in line
    captures = list((tmp_path / "observability" / "serving_configs").glob("*_sglang_*.json"))
    assert len(captures) == 1, "expected exactly one sglang serving-config capture"
    cfg = json.loads(captures[0].read_text(encoding="utf-8"))
    assert cfg["max_total_tokens"] == 147456
    assert "--max-total-tokens 147456" in cfg["args"]
