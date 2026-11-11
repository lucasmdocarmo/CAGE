"""T3.1 pins: tensor-parallel flag pass-through in the serving launchers.

WHAT is pinned and WHY (the 2026-08-27 audit verified NO launcher passed any
TP flag, so every multi-GPU matrix cell -- VLLM_COMPATIBILITY.md section 7:
Llama-3.3-70B TP=4, DeepSeek-V3 TP=8, SGLang pure-TP V3 cells -- would have
silently served TP=1 while the run was labeled with the sweep's TP degree):

1. Env contract (static): manage_vllm_server.sh conditionally passes
   ``--tensor-parallel-size N`` (CAGE_VLLM_TENSOR_PARALLEL; flag per
   docs/VLLM_COMPATIBILITY.md sections 2/3); manage_sglang_server.sh passes
   ``--tp-size N`` (CAGE_SGLANG_TP; SGLang's documented long-form TP flag,
   --tp is its alias). NEITHER pass-through has run on a multi-GPU node, and
   no SGLang pin exists yet, so both exact flag spellings are
   [VERIFY-LIVE at Run-C-prime preflight]. Each launcher validates via the
   shared helpers in scripts/lib/_serving_config.sh BEFORE touching any
   server.

2. Omission contract (behavioral, differential): value 1 means the flag is
   OMITTED ENTIRELY -- the composed single-GPU argv must be BYTE-IDENTICAL
   between "env unset" and "env=1", so engine-default TP handling is
   untouched and pre-T3.1 launches do not change.

3. Refusal paths (behavioral, real bash): every non-positive-integer degree
   refuses LOUDLY, with a nonzero exit, before any server is stopped or
   launched (on `restart` a bad value must not tear down the healthy server
   it would fail to replace). Fail-closed doctrine: a malformed degree must
   never degrade to an engine-default launch, because the run's data would
   then be labeled with a parallelism the server never had.

4. Dial parity (static): the TP flags join the space-anchored dials_match
   cmdline comparison in both launchers -- a TP=2 server must never be
   reused for a TP=4 sweep point (and 2 must not prefix-match 24).

5. Observability: the per-(re)start serving-config capture records the knob,
   both structurally (typed ``tensor_parallel: int|null`` field; null =
   knob not requested) and via the full captured command line (SC_ARGS
   embeds the launch argv, so the flag rides along automatically).

No GPU / engine / network: engine binaries and process probes are stubbed on
PATH (Wave-1 test_serving_budget_knobs.py style), and the launchers run with
*_START_TIMEOUT=0 so they fail their readiness wait AFTER echoing the
composed argv -- which is what we assert.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VLLM_SH = REPO_ROOT / "scripts" / "2_serving" / "manage_vllm_server.sh"
SGLANG_SH = REPO_ROOT / "scripts" / "2_serving" / "manage_sglang_server.sh"
SERVING_CONFIG = REPO_ROOT / "scripts" / "lib" / "_serving_config.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not on PATH")


# ---------------------------------------------------------------------------
# helpers  (Wave-1 style: hermetic stub PATH, real bash)
# ---------------------------------------------------------------------------

def _clean_env(**extra: str) -> dict:
    """Subprocess env with every CAGE_/VLLM_/SGLANG_ var stripped so a dev
    shell's exports can never leak into the pinned default path."""
    env = {
        k: v for k, v in os.environ.items()
        if not k.startswith(("CAGE_", "VLLM_", "SGLANG_"))
    }
    # S0F-23: the suite's tmp log root (tests/conftest.py) rides through the
    # CAGE_ strip so a launcher start never writes under <repo>/logs/.
    if "CAGE_LOG_ROOT" in os.environ:
        env["CAGE_LOG_ROOT"] = os.environ["CAGE_LOG_ROOT"]
    env.update(extra)
    return env


@pytest.fixture(scope="module")
def stub_bin(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Hermetic PATH stubs: no live server can be found (pgrep/curl fail), no
    GPU is touched (nvidia-smi fails), and `vllm` exists-but-exits so the
    backgrounded launch is inert. python3 stays REAL (JSON capture needs it);
    the SGLang interpreter is the sglang-python stub (item 10 import gate)."""
    d = tmp_path_factory.mktemp("stub_bin")
    for name, body in {
        "pgrep": "#!/bin/sh\nexit 1\n",
        "curl": "#!/bin/sh\nexit 1\n",
        "nvidia-smi": "#!/bin/sh\nexit 1\n",
        "pkill": "#!/bin/sh\nexit 0\n",
        "vllm": "#!/bin/sh\nexit 0\n",
        # item 10 (2026-09-26): the SGLang launcher's gate imports sglang
        # through its resolved interpreter; this stub exits 0 for the probe
        # and for the inert backgrounded launch (see _run_launcher).
        "sglang-python": "#!/bin/sh\nexit 0\n",
    }.items():
        p = d / name
        p.write_text(body, encoding="utf-8")
        p.chmod(0o755)
    return d


def _run_launcher(script: Path, stub_bin: Path, *args: str, _extra_bin: Path | None = None, **env_extra: str):
    env = _clean_env(**env_extra)
    env["PATH"] = f"{stub_bin}:{env.get('PATH', '/usr/bin:/bin')}"
    # item 4 (2026-11-11): a per-test stub dir ahead of the module stubs (an
    # engine that writes a fatal line, a curl that answers ready).
    if _extra_bin is not None:
        env["PATH"] = f"{_extra_bin}:{env['PATH']}"
    # item 10 (2026-09-26): the SGLang launcher fails closed unless its
    # interpreter imports sglang; the stub interpreter satisfies the probe so
    # the composed argv is reached (the vLLM launcher ignores the variable).
    env.setdefault("CAGE_SGLANG_PYTHON", str(stub_bin / "sglang-python"))
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
# 1. static: env-conditional flag blocks + refusal gates + parity + capture
# ---------------------------------------------------------------------------

def test_launchers_and_lib_parse_with_bash_n() -> None:
    for script in (VLLM_SH, SGLANG_SH, SERVING_CONFIG):
        proc = subprocess.run(
            ["bash", "-n", str(script)], capture_output=True, text=True, timeout=30
        )
        assert proc.returncode == 0, f"{script.name}: {proc.stderr}"


def test_vllm_launcher_has_env_conditional_tp_block_with_eq1_omission() -> None:
    """The flag block must be conditional on the env var being set AND != 1
    (value 1 = flag omitted entirely; engine default untouched)."""
    text = VLLM_SH.read_text(encoding="utf-8")
    assert re.search(
        r'if \[ -n "\$\{CAGE_VLLM_TENSOR_PARALLEL:-\}" \] '
        r'&& \[ "\$\{CAGE_VLLM_TENSOR_PARALLEL\}" != "1" \]; then\s*\n'
        r'\s*vllm_args\+=\( --tensor-parallel-size "\$\{CAGE_VLLM_TENSOR_PARALLEL\}" \)',
        text,
    ), "TP knob must be env-conditional on CAGE_VLLM_TENSOR_PARALLEL with =1 omission"


def test_sglang_launcher_has_env_conditional_tp_block_with_eq1_omission() -> None:
    text = SGLANG_SH.read_text(encoding="utf-8")
    assert re.search(
        r'if \[ -n "\$\{CAGE_SGLANG_TP:-\}" \] '
        r'&& \[ "\$\{CAGE_SGLANG_TP\}" != "1" \]; then\s*\n'
        r'\s*sglang_args\+=\( --tp-size "\$\{CAGE_SGLANG_TP\}" \)',
        text,
    ), "TP knob must be env-conditional on CAGE_SGLANG_TP with =1 omission"


def test_tp_refusal_gates_fire_before_any_server_action() -> None:
    """The TP validator call must sit at top level (before start_server is
    even defined) and be wired to die: on `restart` a malformed degree must
    not tear down the healthy server it would fail to replace."""
    for script, validator in (
        (VLLM_SH, "cage_validate_vllm_tp_env"),
        (SGLANG_SH, "cage_validate_sglang_tp_env"),
    ):
        text = script.read_text(encoding="utf-8")
        m = re.search(rf"{validator}\s*\\\n\s*\|\| die ", text)
        assert m, f"{script.name}: {validator} must be wired to `|| die`"
        assert m.start() < text.index("start_server()"), (
            f"{script.name}: the refusal gate must run BEFORE any server-"
            "touching code, not inside start_server"
        )


def test_launcher_headers_document_the_tp_contract_and_verify_live() -> None:
    """The env contract lives in each script's header, and the exact flag
    spelling is unproven on real multi-GPU hardware until the session
    preflight -- the VERIFY-LIVE stamp must be surfaced, not just assumed."""
    vllm_header = VLLM_SH.read_text(encoding="utf-8").split("set -euo pipefail")[0]
    assert "CAGE_VLLM_TENSOR_PARALLEL" in vllm_header
    assert "--tensor-parallel-size" in vllm_header
    assert "VERIFY-LIVE at Run-C-prime preflight" in vllm_header, (
        "the TP pass-through is unproven on a multi-GPU node; the header "
        "must carry the VERIFY-LIVE marker"
    )
    sglang_header = SGLANG_SH.read_text(encoding="utf-8").split("set -euo pipefail")[0]
    assert "CAGE_SGLANG_TP" in sglang_header
    assert "--tp-size" in sglang_header
    assert "VERIFY-LIVE at Run-C-prime preflight" in sglang_header, (
        "no SGLang pin exists yet; the exact TP flag spelling must carry "
        "the VERIFY-LIVE marker"
    )


def test_tp_lands_in_serving_config_capture() -> None:
    """Per-restart observability capture must record the knob: structurally
    (typed tensor_parallel field) AND via the full command line (SC_ARGS
    embeds the argv array, so the conditional flag rides along)."""
    vllm = VLLM_SH.read_text(encoding="utf-8")
    assert 'SC_TENSOR_PARALLEL="${CAGE_VLLM_TENSOR_PARALLEL:-}"' in vllm
    assert '"tensor_parallel"' in vllm
    assert 'SC_ARGS="vllm serve $model ${vllm_args[*]}"' in vllm

    sglang = SGLANG_SH.read_text(encoding="utf-8")
    assert 'SC_TENSOR_PARALLEL="${CAGE_SGLANG_TP:-}"' in sglang
    assert '"tensor_parallel"' in sglang
    # item 10 (2026-09-26): the argv string records the RESOLVED interpreter
    # (CAGE_SGLANG_PYTHON / sglang-env / PATH python3), never a bare python3.
    assert 'SC_ARGS="$SGLANG_PYTHON -m sglang.launch_server ${sglang_args[*]}"' in sglang


def test_reuse_dial_parity_covers_tp_space_anchored() -> None:
    """A sweep iteration invoked via `start` must never reuse a server
    launched under a DIFFERENT TP degree: the TP flags are part of the
    dials_match cmdline comparison in both launchers, space-anchored on both
    sides so a requested 2 cannot prefix-match a live 24."""
    vllm = VLLM_SH.read_text(encoding="utf-8")
    reuse = vllm[vllm.index("dials_match=true"):vllm.index("Start server")]
    assert '" --tensor-parallel-size ${CAGE_VLLM_TENSOR_PARALLEL} "' in reuse
    # absent-side parity: unset/=1 must REJECT a live TP'd server
    assert '"$live_cmd" != *"--tensor-parallel-size"*' in reuse

    sglang = SGLANG_SH.read_text(encoding="utf-8")
    reuse = sglang[sglang.index("dials_match=true"):sglang.index("local timestamp log_file")]
    assert '" --tp-size ${CAGE_SGLANG_TP} "' in reuse
    assert '"$live_cmd" != *"--tp-size"*' in reuse


# ---------------------------------------------------------------------------
# 2. behavioral: validator refusal + acceptance under real bash
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["abc", "2.5", "-2", "0", "+4", "1e1", " 2", "0x4", "00"])
def test_tp_validators_refuse_non_positive_integers(bad: str) -> None:
    for func, var in (
        ("cage_validate_vllm_tp_env", "CAGE_VLLM_TENSOR_PARALLEL"),
        ("cage_validate_sglang_tp_env", "CAGE_SGLANG_TP"),
    ):
        proc = _validate(func, **{var: bad})
        assert proc.returncode != 0, f"{func} accepted {var}={bad!r}"
        assert "REFUSING" in proc.stderr, (
            f"refusal must be LOUD (stderr), got: {proc.stderr!r}"
        )
        assert var in proc.stderr, "the refusal must name the offending variable"


@pytest.mark.parametrize("good", ["1", "2", "4", "8"])
def test_tp_validators_accept_positive_integers(good: str) -> None:
    # 1 is VALID at the validator (it means "omit the flag"), never a refusal.
    for func, var in (
        ("cage_validate_vllm_tp_env", "CAGE_VLLM_TENSOR_PARALLEL"),
        ("cage_validate_sglang_tp_env", "CAGE_SGLANG_TP"),
    ):
        proc = _validate(func, **{var: good})
        assert proc.returncode == 0, f"{func} refused valid {var}={good}: {proc.stderr}"


def test_tp_validators_pass_when_unset() -> None:
    for func in ("cage_validate_vllm_tp_env", "cage_validate_sglang_tp_env"):
        proc = _validate(func)
        assert proc.returncode == 0, f"{func} must be a no-op with no TP env set"


# ---------------------------------------------------------------------------
# 3. behavioral: full launcher runs (stubbed PATH, TIMEOUT=0)
# ---------------------------------------------------------------------------

def test_vllm_tp_flag_lands_on_cmdline_and_capture(
    stub_bin: Path, tmp_path: Path
) -> None:
    proc = _run_launcher(
        VLLM_SH, stub_bin, "start", "fake/test-model",
        CAGE_VLLM_TENSOR_PARALLEL="4", CAGE_RUN_ROOT=str(tmp_path),
    )
    line = _server_args_line(proc.stdout)
    assert "--tensor-parallel-size 4" in line
    assert "--gpu-memory-utilization 0.90" in line, (
        "the uniform fraction dial must still ship alongside the TP flag"
    )
    captures = list((tmp_path / "observability" / "serving_configs").glob("*.json"))
    assert len(captures) == 1, "expected exactly one serving-config capture"
    cfg = json.loads(captures[0].read_text(encoding="utf-8"))
    assert cfg["tensor_parallel"] == 4
    assert "--tensor-parallel-size 4" in cfg["args"]


def test_sglang_tp_flag_lands_on_cmdline_and_capture(
    stub_bin: Path, tmp_path: Path
) -> None:
    proc = _run_launcher(
        SGLANG_SH, stub_bin, "start", "fake/test-model",
        CAGE_SGLANG_TP="2", CAGE_RUN_ROOT=str(tmp_path),
    )
    line = _server_args_line(proc.stdout)
    assert "--tp-size 2" in line
    assert "--mem-fraction-static 0.90" in line
    captures = list((tmp_path / "observability" / "serving_configs").glob("*_sglang_*.json"))
    assert len(captures) == 1, "expected exactly one sglang serving-config capture"
    cfg = json.loads(captures[0].read_text(encoding="utf-8"))
    assert cfg["tensor_parallel"] == 2
    assert "--tp-size 2" in cfg["args"]


def test_vllm_tp_1_and_unset_compose_byte_identical_argv(stub_bin: Path) -> None:
    """Differential pin of the omission contract: TP=1 must OMIT the flag and
    the composed argv must be byte-identical to the no-env launch (the
    single-GPU default must not change under an explicit =1)."""
    unset_line = _server_args_line(
        _run_launcher(VLLM_SH, stub_bin, "start", "fake/test-model").stdout
    )
    eq1_line = _server_args_line(
        _run_launcher(
            VLLM_SH, stub_bin, "start", "fake/test-model",
            CAGE_VLLM_TENSOR_PARALLEL="1",
        ).stdout
    )
    assert "--tensor-parallel-size" not in unset_line
    assert "--tensor-parallel-size" not in eq1_line
    assert unset_line == eq1_line, (
        "TP=1 must compose a byte-identical argv to the unset default"
    )


def test_sglang_tp_1_and_unset_compose_byte_identical_argv(stub_bin: Path) -> None:
    unset_line = _server_args_line(
        _run_launcher(SGLANG_SH, stub_bin, "start", "fake/test-model").stdout
    )
    eq1_line = _server_args_line(
        _run_launcher(
            SGLANG_SH, stub_bin, "start", "fake/test-model",
            CAGE_SGLANG_TP="1",
        ).stdout
    )
    assert "--tp-size" not in unset_line
    assert "--tp-size" not in eq1_line
    assert unset_line == eq1_line, (
        "TP=1 must compose a byte-identical argv to the unset default"
    )


def test_vllm_launcher_refuses_bad_tp_before_touching_anything(stub_bin: Path) -> None:
    proc = _run_launcher(
        VLLM_SH, stub_bin, "start", "fake/test-model",
        CAGE_VLLM_TENSOR_PARALLEL="abc",
    )
    assert proc.returncode != 0
    assert "FATAL" in proc.stderr and "REFUSING" in proc.stderr
    assert "Server args:" not in proc.stdout, (
        "refusal must happen BEFORE any launch attempt is composed"
    )


def test_sglang_launcher_refuses_zero_tp(stub_bin: Path) -> None:
    proc = _run_launcher(
        SGLANG_SH, stub_bin, "start", "fake/test-model",
        CAGE_SGLANG_TP="0",
    )
    assert proc.returncode != 0
    assert "FATAL" in proc.stderr and "REFUSING" in proc.stderr
    assert "Server args:" not in proc.stdout


def test_restart_with_bad_tp_never_reaches_teardown(stub_bin: Path) -> None:
    """`restart` with a malformed degree must refuse BEFORE stop_server: a
    healthy server must not be torn down for a launch that cannot happen."""
    for script, var in (
        (VLLM_SH, "CAGE_VLLM_TENSOR_PARALLEL"),
        (SGLANG_SH, "CAGE_SGLANG_TP"),
    ):
        proc = _run_launcher(script, stub_bin, "restart", "fake/test-model", **{var: "-2"})
        assert proc.returncode != 0
        assert "REFUSING" in proc.stderr
        assert "Stopping" not in proc.stdout, (
            f"{script.name}: refusal fired AFTER teardown began"
        )


# ---------------------------------------------------------------------------
# S2 block C-2 (2026-11-01, charter 5.2): the vLLM launcher pins the hybrid
# cache mode on every prefix-on start of a Qwen3-Next model. vLLM 0.19.1 serves
# Qwen3-Next with prefix caching only in --mamba-cache-mode align (the "all"
# mode raises NotImplementedError); the flag is emitted, never left to a
# fallback, and the reuse check reads it back as a dial.
# ---------------------------------------------------------------------------

QWEN3_NEXT = "Qwen/Qwen3-Next-80B-A3B-Instruct"


def test_vllm_qwen3_next_prefix_on_start_carries_the_align_cache_mode(stub_bin: Path) -> None:
    line = _server_args_line(_run_launcher(VLLM_SH, stub_bin, "start", QWEN3_NEXT).stdout)
    assert "--enable-prefix-caching" in line
    assert "--mamba-cache-mode align" in line


def test_vllm_qwen3_next_prefix_off_start_omits_the_cache_mode(stub_bin: Path) -> None:
    line = _server_args_line(
        _run_launcher(VLLM_SH, stub_bin, "start", QWEN3_NEXT, "--no-prefix-cache").stdout
    )
    assert "--no-enable-prefix-caching" in line
    assert "--mamba-cache-mode" not in line


def test_vllm_non_hybrid_models_never_carry_the_cache_mode(stub_bin: Path) -> None:
    for args in (
        ("start", "Qwen/Qwen3-14B"),
        ("start", "Qwen/Qwen3-14B", "--no-prefix-cache"),
        ("start", "meta-llama/Llama-3.3-70B-Instruct"),
    ):
        line = _server_args_line(_run_launcher(VLLM_SH, stub_bin, *args).stdout)
        assert "--mamba-cache-mode" not in line, args


def test_vllm_reuse_check_reads_the_cache_mode_as_a_dial() -> None:
    """Static pin: the running-server reuse block compares the live cmdline's
    --mamba-cache-mode with the requested one, space-anchored like the TP
    dial, so a prefix-on 80B server started without the mode is never reused
    and a prefix-off start never reuses an aligned server."""
    vllm = VLLM_SH.read_text(encoding="utf-8")
    reuse = vllm[vllm.index("dials_match=true"):vllm.index("Start server")]
    assert '*" --mamba-cache-mode align "*' in reuse
    assert '"$live_cmd" != *"--mamba-cache-mode"*' in reuse


# ---------------------------------------------------------------------------
# S2R 2026-11-11 (ADR-0198): the launcher's shell variable SGLANG_PORT never
# reaches the server process. sglang 0.5.10.post1 reads an environment variable
# of that same name in get_open_port() (sglang/srt/utils/network.py lines 17 to
# 26) as the base for its internal sockets; the TP0 scheduler's shm_broadcast
# message queue bound 127.0.0.1:30000 with it, and the HTTP server, which binds
# last, failed with "address already in use" on both ladder relaunches of the
# rehearsal (every relaunch exports the variable, run_campaign.py
# PORT_LAUNCH_ENV; the anchor calibration never does and started 13 of 13
# times on the same pod). The port itself keeps riding argv as --port.
# ---------------------------------------------------------------------------

def test_sglang_launch_strips_the_shell_port_variable_from_the_server_env(
    stub_bin: Path, tmp_path: Path
) -> None:
    dump = tmp_path / "launch_env.txt"
    stub = tmp_path / "sglang-python"
    # Answers the import gate and the version probe like the fixture's stub;
    # on the launch call (-m sglang.launch_server) it records its environment.
    # review 2026-11-11 LOW 1: write to a temp name and rename, so the file's
    # existence below implies a complete dump (the stub is read as soon as it exists)
    stub.write_text(
        '#!/bin/sh\nif [ "$1" = "-m" ]; then env > "$CAGE_TEST_ENV_DUMP.tmp"'
        ' && mv "$CAGE_TEST_ENV_DUMP.tmp" "$CAGE_TEST_ENV_DUMP"; fi\nexit 0\n',
        encoding="utf-8",
    )
    stub.chmod(0o755)
    proc = _run_launcher(
        SGLANG_SH, stub_bin, "start", "fake/test-model",
        SGLANG_PORT="30000", CAGE_SGLANG_PYTHON=str(stub), CAGE_TEST_ENV_DUMP=str(dump),
    )
    assert "--port 30000" in _server_args_line(proc.stdout)
    deadline = time.monotonic() + 5.0
    while not dump.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert dump.exists(), "the backgrounded launch never ran the stub interpreter"
    env = dict(ln.split("=", 1) for ln in dump.read_text(encoding="utf-8").splitlines() if "=" in ln)
    assert "SGLANG_PORT" not in env, (
        "SGLANG_PORT reached sglang.launch_server; sglang's get_open_port() would "
        "hand the API port to an internal socket (ADR-0198)"
    )
    # only that one variable is withheld: the rest of the environment still arrives
    assert env.get("SGLANG_START_TIMEOUT") == "0"
    assert env.get("CAGE_TEST_ENV_DUMP") == str(dump)


# ---------------------------------------------------------------------------
# item 4 (2026-11-11, S2R pre-restart specs L2, option (a)): the readiness
# loops of both launchers end in the tick that shows a fatal start line or a
# dead launch pid, with the last 40 start-log lines, the socket table of the
# port and a stop_server, instead of waiting out *_START_TIMEOUT (600 s on the
# pod: the ADR-0198 port collision cost two of them). Every pattern below was
# read in a live log, never taken from memory:
#   vLLM   experiments/S0/2026-10-08_1322/logs/engines/vllm/vllm_Qwen_Qwen3-14B_20261008_212751.log
#          and _215604.log (both lines); the second is also the HTTP 500 body
#          of the PD smoke of 2026-11-10 (experiments/S2R/2026-11-09_2233/extras/pd_smoke/pd_smoke.json).
#   SGLang ADR-0198 (uvicorn's bind failure, lowercase) and the collision log
#          of 2026-11-11 (the ZMQ XPUB bind, S2R_PRE_RESTART_SPECS item 4).
# The ready path is accepted exactly as before (the ready check runs first, so
# a pid that dies after the ready line still returns 0).
# ---------------------------------------------------------------------------

VLLM_FATAL_LIVE = [
    "Engine core initialization failed. See root cause above. Failed core proc(s): {}",
    "EngineCore encountered an issue. See stack trace (above) for the root cause.",
]
SGLANG_FATAL_LIVE = [
    "[Errno 98] error while attempting to bind on address ('127.0.0.1', 30000): address already in use",
    "zmq.error.ZMQError: Address already in use",
]
SGLANG_READY = "The server is fired up and ready to roll!"


def _code_lines(text: str) -> str:
    """The script without its comment lines (a curl named in a comment is not a call)."""
    return "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))


def _stub(d: Path, name: str, body: str) -> Path:
    p = d / name
    p.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    p.chmod(0o755)
    return p


def _sglang_stub_body(on_launch: str) -> str:
    # the import gate (-c) and the version probe exit 0 silently; the launch (-m) runs on_launch
    return 'if [ "$1" = "-m" ]; then\n' + on_launch + '\nfi\nexit 0\n'


def _run_fail_fast(script: Path, stub_bin: Path, extra_bin: Path, timeout_s: int, **env: str):
    env = dict(env, CAGE_KV_POOL_CAPTURE_RETRIES="0", CAGE_KV_POOL_CAPTURE_INTERVAL="0")
    if script is VLLM_SH:
        env["VLLM_START_TIMEOUT"] = str(timeout_s)
    else:
        env["SGLANG_START_TIMEOUT"] = str(timeout_s)
        env["CAGE_SGLANG_PYTHON"] = str(extra_bin / "sglang-python")
    t0 = time.monotonic()
    proc = _run_launcher(script, stub_bin, "start", "fake/test-model", _extra_bin=extra_bin, **env)
    return proc, time.monotonic() - t0


def _assert_failed_fast(proc, elapsed: float, engine: str, port: int) -> None:
    assert proc.returncode == 1, f"rc={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert elapsed < 10, f"the loop must end within a tick, took {elapsed:.1f}s"
    assert "last 40 lines of" in proc.stdout and f"sockets on port {port}" in proc.stdout, proc.stdout
    assert f"Stopping {engine} server" in proc.stdout, "the failure path must stop the half-started server"
    assert "Server ready" not in proc.stdout


@pytest.mark.parametrize("line", VLLM_FATAL_LIVE, ids=["init-failed", "engine-core-issue"])
def test_vllm_start_fails_within_a_tick_on_a_fatal_line_item4(stub_bin: Path, tmp_path: Path, line: str) -> None:
    d = tmp_path / "bin"; d.mkdir()
    _stub(d, "vllm", "printf '%s\\n' " + shlex.quote(line) + "\nsleep 5\nexit 0\n")
    proc, elapsed = _run_fail_fast(VLLM_SH, stub_bin, d, 60)
    _assert_failed_fast(proc, elapsed, "vLLM", 8000)
    assert "fatal start line" in proc.stdout and line in proc.stdout, proc.stdout


@pytest.mark.parametrize("line", SGLANG_FATAL_LIVE, ids=["uvicorn-bind", "zmq-bind"])
def test_sglang_start_fails_within_a_tick_on_a_fatal_line_item4(stub_bin: Path, tmp_path: Path, line: str) -> None:
    d = tmp_path / "bin"; d.mkdir()
    _stub(d, "sglang-python", _sglang_stub_body("printf '%s\\n' " + shlex.quote(line) + "\nsleep 5"))
    proc, elapsed = _run_fail_fast(SGLANG_SH, stub_bin, d, 60)
    _assert_failed_fast(proc, elapsed, "SGLang", 30000)
    assert "fatal start line" in proc.stdout and line in proc.stdout, proc.stdout


def test_both_launchers_fail_within_a_tick_when_the_launch_pid_exits_item4(stub_bin: Path, tmp_path: Path) -> None:
    d = tmp_path / "bin"; d.mkdir()
    _stub(d, "vllm", "exit 3\n")
    _stub(d, "sglang-python", _sglang_stub_body("exit 3"))
    for script, engine, port in ((VLLM_SH, "vLLM", 8000), (SGLANG_SH, "SGLang", 30000)):
        proc, elapsed = _run_fail_fast(script, stub_bin, d, 60)
        _assert_failed_fast(proc, elapsed, engine, port)
        assert "exited" in proc.stdout and "fatal start line" not in proc.stdout, proc.stdout


def test_both_launchers_still_return_0_on_the_ready_path_item4(stub_bin: Path, tmp_path: Path) -> None:
    """A ready server is accepted exactly as before: /v1/models names the model
    (vLLM also answers /health; SGLang's log also carries its ready line). The
    engine stubs outlive the check by a few seconds and exit on their own."""
    d = tmp_path / "bin"; d.mkdir()
    _stub(d, "curl", 'case "$*" in *"/v1/models"*) printf \'{"data":[{"id":"fake/test-model"}]}\\n\' ;; esac\nexit 0\n')
    _stub(d, "vllm", "sleep 5\nexit 0\n")
    _stub(d, "sglang-python", _sglang_stub_body("printf '%s\\n' " + shlex.quote(SGLANG_READY) + "\nsleep 5"))
    for script in (VLLM_SH, SGLANG_SH):
        proc, elapsed = _run_fail_fast(script, stub_bin, d, 60)
        assert proc.returncode == 0, f"{script.name}: rc={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        assert "Server ready with model: fake/test-model" in proc.stdout
        assert "fatal start line" not in proc.stdout and "exited" not in proc.stdout
        assert elapsed < 15, f"{script.name}: ready path took {elapsed:.1f}s"


def test_readiness_loops_check_ready_first_bound_by_a_wall_clock_with_bounded_curl_item4() -> None:
    """Static pins: in each loop the ready check precedes the fatal-line check,
    which precedes the pid check (ready wins when both hold); the bound is the
    SECONDS wall clock, not a count of sleeps; every curl carries --max-time 5;
    the live pattern lists are the ones the behavioral tests above exercise."""
    for script, ready_marker, patterns in (
        (VLLM_SH, 'if [ "$loaded" = "$model" ]; then', ("Engine core initialization failed", "EngineCore encountered an issue")),
        (SGLANG_SH, 'grep -qF -- "$SGLANG_READY_LINE"', ("address already in use", "zmq.error.ZMQError")),
    ):
        text = script.read_text(encoding="utf-8")
        loop = text[text.index("Waiting for server to start"):text.index("stop_server()")]
        assert loop.index(ready_marker) < loop.index("cage_start_fatal_line") < loop.index('kill -0 "$server_pid"'), script.name
        assert "SECONDS" in loop and "waited=$((waited + 2))" not in loop, script.name
        code = _code_lines(text)
        curls = re.findall(r"curl [^\n|]*", code)
        assert curls and all("--max-time 5" in c for c in curls), (script.name, curls)
        for p in patterns:
            assert f'"{p}"' in code, (script.name, p)
