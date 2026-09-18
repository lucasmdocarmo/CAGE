"""T3.2 pins: the prefill/decode disaggregation launcher stack.

WHAT is pinned and WHY (the 2026-08-27 audit verified TopologyLauncher was
ZERO code and the pilot "cluster" path is router-over-replicas, NOT PD — so
every DIST/pd matrix cell was unrunnable and, worse, easy to fake with a
replicated single-instance stack mislabeled as disaggregation):

1. manage_vllm_pd.sh start composition (behavioral, stubbed PATH): TWO vllm
   instances on DISTINCT ports whose ONLY role difference is the per-role
   --kv-transfer-config (NixlConnector, kv_producer on prefill / kv_consumer
   on decode — the kv_role tokens are [VERIFY-LIVE at Run-C-prime preflight]
   and the start banner must SAY so), per-role byte budgets from the two
   REQUIRED env vars (charter §6.5: the split is explicit, never defaulted),
   the T3.1 TP knob applied per instance, distinct log files echoed in
   gate-(j) pin form (vllm:prefill=<log>,vllm:decode=<log>), and a
   serving-config capture per role.

2. Refusal matrix (fail-closed doctrine): a missing/malformed role budget,
   an ambient single-instance budget knob, an ambient VLLM_KV_TRANSFER_CONFIG
   (the launcher OWNS the connector JSON), or a malformed TP degree refuses
   LOUDLY before ANY process is stopped or started ("Stopping" must not
   appear — start is self-cleaning, so a refusal firing after the teardown
   would kill a healthy stack for a launch that cannot happen). `stop` is
   NEVER gated (teardown discipline).

3. pd_proxy.py 1P1D semantics (in-process stub upstreams, http.client): the
   prefill request goes FIRST with max_tokens clamped to 1, the decode
   request carries the ORIGINAL generation budget, engine-provided
   kv_transfer_params pass through UNTOUCHED, and the proxy NEVER stamps a
   "source" field — the T3.3 campaign provenance gate accepts only
   engine-real stamps, so until the live NIXL path is verified the correct
   end-to-end outcome is that gate REFUSING (a proxy-fabricated stamp would
   be fake provenance). /health is 200 only when BOTH upstreams answer and
   reports the data path as PENDING, never PASS.

4. run_campaign.py pd emission + gating (stub runner, no GPU/network): a
   vllm DIST/pd cell is enumerated EXECUTABLE (blocked_on=null) with
   gate:"--allow-pd + PD preflight smoke"; its relaunch step invokes the pd
   launcher with env carrying BOTH §6.5 role budgets (exact-sum split of
   floor(r×D), pinned by independent arithmetic) and
   CAGE_TELEMETRY_ENDPOINTS="prefill=<url>,decode=<url>"; 'run' refuses pd
   cells without the explicit --allow-pd (message names the flag) and
   executes them with it; pd on sglang and the tp overlay stay BLOCKED.

No GPU / engine / network: launcher runs use a hermetic stub PATH +
VLLM_START_TIMEOUT=0 (test_tp_flags.py idiom); proxy upstreams are
in-process http.server stubs on localhost ephemeral ports.
"""
from __future__ import annotations

import http.client
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PD_SH = REPO_ROOT / "scripts" / "2_serving" / "manage_vllm_pd.sh"
PD_PROXY_PY = REPO_ROOT / "scripts" / "2_serving" / "pd_proxy.py"
RUN_CAMPAIGN_PY = REPO_ROOT / "scripts" / "3_run" / "run_campaign.py"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not on PATH")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # register BEFORE exec (dataclass-safe)
    spec.loader.exec_module(module)
    return module


rc = _load("run_campaign_pd_tests", RUN_CAMPAIGN_PY)
pd_proxy = _load("pd_proxy_tests", PD_PROXY_PY)


# ---------------------------------------------------------------------------
# helpers — launcher runs (hermetic stub PATH, real bash; test_tp_flags idiom)
# ---------------------------------------------------------------------------

import os  # noqa: E402  (used by the env helpers below)


def _clean_env(**extra: str) -> dict:
    env = {
        k: v for k, v in os.environ.items()
        if not k.startswith(("CAGE_", "VLLM_", "SGLANG_", "PD_"))
    }
    env.update(extra)
    return env


GOOD_BUDGETS = {
    "CAGE_KV_BUDGET_BYTES_PREFILL": "3000000000",
    "CAGE_KV_BUDGET_BYTES_DECODE": "7000000000",
}


@pytest.fixture(scope="module")
def stub_bin(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """No live server findable (pgrep/curl/nvidia-smi fail), pkill inert,
    `vllm` exists-but-exits so backgrounded launches are inert. python3 stays
    REAL (the serving-config capture needs it)."""
    d = tmp_path_factory.mktemp("stub_bin")
    # The vllm stub journals the env it was launched with (A1 / S0-20: the
    # per-role GPU pin must ride CUDA_VISIBLE_DEVICES in the child env and
    # nowhere else) when CAGE_TEST_VLLM_JOURNAL names a file.
    vllm_stub = (
        "#!/bin/sh\n"
        'if [ -n "${CAGE_TEST_VLLM_JOURNAL:-}" ]; then\n'
        "  printf 'CUDA_VISIBLE_DEVICES=%s args=%s\\n' "
        '"${CUDA_VISIBLE_DEVICES-unset}" "$*" >> "$CAGE_TEST_VLLM_JOURNAL"\n'
        "fi\n"
        "exit 0\n"
    )
    for name, body in {
        "pgrep": "#!/bin/sh\nexit 1\n",
        "curl": "#!/bin/sh\nexit 1\n",
        "nvidia-smi": "#!/bin/sh\nexit 1\n",
        "pkill": "#!/bin/sh\nexit 0\n",
        "vllm": vllm_stub,
    }.items():
        p = d / name
        p.write_text(body, encoding="utf-8")
        p.chmod(0o755)
    return d


def _run_pd(stub_bin: Path, *args: str, **env_extra: str):
    env = _clean_env(**env_extra)
    env["PATH"] = f"{stub_bin}:{env.get('PATH', '/usr/bin:/bin')}"
    # TIMEOUT=0: the launcher composes + echoes BOTH instances' argv, then
    # fails the readiness wait fast — the echo is what we assert.
    env.setdefault("VLLM_START_TIMEOUT", "0")
    return subprocess.run(
        ["bash", str(PD_SH), *args],
        capture_output=True, text=True, env=env, timeout=120,
    )


def _args_line(stdout: str, role: str) -> str:
    lines = [ln for ln in stdout.splitlines() if ln.startswith(f"Server args [{role}]:")]
    assert len(lines) == 1, (
        f"expected exactly one 'Server args [{role}]:' line, got:\n{stdout}"
    )
    return lines[0]


# ---------------------------------------------------------------------------
# 0. static: parseability + header VERIFY-LIVE stamps
# ---------------------------------------------------------------------------


def test_pd_launcher_parses_with_bash_n() -> None:
    proc = subprocess.run(
        ["bash", "-n", str(PD_SH)], capture_output=True, text=True, timeout=30
    )
    assert proc.returncode == 0, proc.stderr


def test_pd_proxy_compiles() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "py_compile", str(PD_PROXY_PY)],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr


def test_headers_carry_verify_live_and_provenance_interplay() -> None:
    """Unconfirmable engine-facing names (kv_role tokens, per-role pool
    semantics, the NIXL data path) must be stamped VERIFY-LIVE in the
    launcher header, and the proxy docstring must document that the T3.3
    provenance gate REFUSING is the correct pre-verification outcome."""
    launcher_header = PD_SH.read_text(encoding="utf-8").split("set -euo pipefail")[0]
    assert "VERIFY-LIVE at Run-C-prime preflight" in launcher_header
    assert "kv_producer" in launcher_header and "kv_consumer" in launcher_header
    proxy_doc = pd_proxy.__doc__ or ""
    assert "VERIFY-LIVE at Run-C-prime preflight" in proxy_doc
    assert "never" in proxy_doc.lower() and "source" in proxy_doc
    assert "REFUSING" in proxy_doc  # the gate-refuses-is-correct interplay


# ---------------------------------------------------------------------------
# 1. behavioral: start composition (both instances, roles, budgets, TP, logs)
# ---------------------------------------------------------------------------


def test_start_composes_both_role_instances(stub_bin: Path, tmp_path: Path) -> None:
    proc = _run_pd(
        stub_bin, "start", "fake/test-model",
        CAGE_VLLM_TENSOR_PARALLEL="4", CAGE_RUN_ROOT=str(tmp_path),
        **GOOD_BUDGETS,
    )
    out = proc.stdout
    prefill = _args_line(out, "prefill")
    decode = _args_line(out, "decode")

    # role difference = the per-role connector config, nothing else hidden
    assert '"kv_connector":"NixlConnector"' in prefill
    assert '"kv_role":"kv_producer"' in prefill
    assert '"kv_role":"kv_consumer"' in decode
    assert "kv_consumer" not in prefill and "kv_producer" not in decode

    # distinct ports (launcher defaults, mirrored by run_campaign.py)
    assert "--port 8100" in prefill
    assert "--port 8200" in decode

    # §6.5 per-role byte budgets land verbatim on each instance's argv
    assert "--kv-cache-memory-bytes 3000000000" in prefill
    assert "--kv-cache-memory-bytes 7000000000" in decode

    # T3.1 TP knob applied PER INSTANCE
    assert "--tensor-parallel-size 4" in prefill
    assert "--tensor-parallel-size 4" in decode

    # uniform regime still ships on both. Mem-util pin derivation (backlog
    # A1, S0-9/S0-20): no CAGE_PD_PREFILL_GPUS / CAGE_PD_DECODE_GPUS in this
    # env, so both roles share ONE GPU and the per-instance default is
    # SHARED_GPU_MEM_UTIL=0.45 (was 0.90, which vLLM's startup check refuses
    # for the second instance on a shared device). The 0.90 operating point
    # is pinned on the distinct-pins path in
    # test_start_distinct_role_pins_keep_0_90_and_ride_cuda_visible_devices.
    for line in (prefill, decode):
        assert "--max-model-len 4096" in line
        assert "--gpu-memory-utilization 0.45" in line
        assert "--enable-prompt-tokens-details" in line
    # the decision is printed so the S0 evidence shows which regime served
    assert "[cage] gpu-share decision: shared" in out
    assert "0.45" in out.split("[cage] gpu-share decision:")[1].splitlines()[0]

    # start banner surfaces the VERIFY-LIVE state (not only comments)
    assert "VERIFY-LIVE at Run-C-prime preflight" in out


def test_start_echoes_gate_j_log_pin_with_distinct_logs(stub_bin: Path) -> None:
    proc = _run_pd(stub_bin, "start", "fake/test-model", **GOOD_BUDGETS)
    m = re.search(
        r'CAGE_ISO_BYTES_LOGS="vllm:prefill=([^,"]+),vllm:decode=([^"]+)"',
        proc.stdout,
    )
    assert m, f"gate-(j) pin line missing/malformed:\n{proc.stdout}"
    prefill_log, decode_log = m.group(1), m.group(2)
    assert prefill_log != decode_log, "roles must log to DISTINCT files"
    assert "prefill" in prefill_log and "decode" in decode_log


def test_start_captures_serving_config_per_role(stub_bin: Path, tmp_path: Path) -> None:
    _run_pd(
        stub_bin, "start", "fake/test-model",
        CAGE_VLLM_TENSOR_PARALLEL="2", CAGE_RUN_ROOT=str(tmp_path),
        **GOOD_BUDGETS,
    )
    cfg_dir = tmp_path / "observability" / "serving_configs"
    prefill_caps = list(cfg_dir.glob("*_pd-prefill.json"))
    decode_caps = list(cfg_dir.glob("*_pd-decode.json"))
    assert len(prefill_caps) == 1 and len(decode_caps) == 1, (
        f"expected one capture PER ROLE, got: {sorted(cfg_dir.glob('*'))}"
    )
    prefill = json.loads(prefill_caps[0].read_text(encoding="utf-8"))
    decode = json.loads(decode_caps[0].read_text(encoding="utf-8"))
    assert prefill["topology"] == decode["topology"] == "pd"
    assert prefill["role"] == "prefill" and decode["role"] == "decode"
    assert prefill["kv_budget_bytes"] == 3_000_000_000
    assert decode["kv_budget_bytes"] == 7_000_000_000
    assert prefill["tensor_parallel"] == decode["tensor_parallel"] == 2
    assert prefill["kv_transfer_config"]["kv_role"] == "kv_producer"
    assert decode["kv_transfer_config"]["kv_role"] == "kv_consumer"
    assert "--kv-cache-memory-bytes 3000000000" in prefill["args"]
    assert "--kv-cache-memory-bytes 7000000000" in decode["args"]
    # A1 / S0-20 provenance: the capture records the gpu-share decision and
    # the realized per-instance dial (shared default here, see the pin
    # derivation in test_start_composes_both_role_instances).
    for cap in (prefill, decode):
        assert cap["gpu_memory_utilization"] == 0.45
        assert cap["gpu_share"] == "shared"
        assert cap["cuda_visible_devices"] is None


def test_start_tp_unset_omits_flag_on_both(stub_bin: Path) -> None:
    proc = _run_pd(stub_bin, "start", "fake/test-model", **GOOD_BUDGETS)
    assert "--tensor-parallel-size" not in _args_line(proc.stdout, "prefill")
    assert "--tensor-parallel-size" not in _args_line(proc.stdout, "decode")


# ---------------------------------------------------------------------------
# 1b. behavioral: shared-GPU memory-utilization rule (backlog A1, S0-9/S0-20)
# ---------------------------------------------------------------------------
# vLLM's startup check requests --gpu-memory-utilization of the device
# unconditionally, so two role instances on ONE GPU at 0.90 each cannot both
# start. Rule: no distinct per-role CUDA_VISIBLE_DEVICES pins (CAGE_PD_PREFILL_GPUS
# / CAGE_PD_DECODE_GPUS) = shared -> default SHARED_GPU_MEM_UTIL=0.45, explicit
# VLLM_GPU_MEMORY_UTILIZATION above 0.50 REFUSED; distinct pins = 0.90 stands.

DISTINCT_PINS = {"CAGE_PD_PREFILL_GPUS": "0", "CAGE_PD_DECODE_GPUS": "1"}


def _decision_line(stdout: str) -> str:
    lines = [ln for ln in stdout.splitlines() if ln.startswith("[cage] gpu-share decision:")]
    assert len(lines) == 1, f"expected exactly one gpu-share decision line, got:\n{stdout}"
    return lines[0]


def _read_journal(path: Path, expected_lines: int) -> List[str]:
    # the stubbed vllm is backgrounded (nohup ... &); give it a moment to land
    import time
    deadline = time.time() + 10
    while time.time() < deadline:
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
            if len(lines) >= expected_lines:
                return lines
        time.sleep(0.05)
    raise AssertionError(f"vllm stub journal never reached {expected_lines} lines: {path}")


def test_start_distinct_role_pins_keep_0_90_and_ride_cuda_visible_devices(
    stub_bin: Path, tmp_path: Path
) -> None:
    journal = tmp_path / "vllm_journal.txt"
    proc = _run_pd(
        stub_bin, "start", "fake/test-model",
        CAGE_RUN_ROOT=str(tmp_path), CAGE_TEST_VLLM_JOURNAL=str(journal),
        **GOOD_BUDGETS, **DISTINCT_PINS,
    )
    out = proc.stdout
    assert "--gpu-memory-utilization 0.90" in _args_line(out, "prefill")
    assert "--gpu-memory-utilization 0.90" in _args_line(out, "decode")
    line = _decision_line(out)
    assert "distinct" in line and "0.90" in line
    assert "prefill=0" in line and "decode=1" in line
    # the pin rides the child env (CUDA_VISIBLE_DEVICES), one value per role
    lines = _read_journal(journal, 2)
    prefill = [ln for ln in lines if "--port 8100" in ln]
    decode = [ln for ln in lines if "--port 8200" in ln]
    assert len(prefill) == 1 and len(decode) == 1, lines
    assert prefill[0].startswith("CUDA_VISIBLE_DEVICES=0 ")
    assert decode[0].startswith("CUDA_VISIBLE_DEVICES=1 ")
    # and the per-role capture records it
    cfg_dir = tmp_path / "observability" / "serving_configs"
    pcap = json.loads(next(cfg_dir.glob("*_pd-prefill.json")).read_text(encoding="utf-8"))
    dcap = json.loads(next(cfg_dir.glob("*_pd-decode.json")).read_text(encoding="utf-8"))
    assert pcap["gpu_share"] == dcap["gpu_share"] == "distinct"
    assert pcap["cuda_visible_devices"] == "0" and dcap["cuda_visible_devices"] == "1"
    assert pcap["gpu_memory_utilization"] == dcap["gpu_memory_utilization"] == 0.90


def test_start_shared_does_not_inject_cuda_visible_devices(
    stub_bin: Path, tmp_path: Path
) -> None:
    journal = tmp_path / "vllm_journal.txt"
    _run_pd(
        stub_bin, "start", "fake/test-model",
        CAGE_TEST_VLLM_JOURNAL=str(journal), **GOOD_BUDGETS,
    )
    for ln in _read_journal(journal, 2):
        assert ln.startswith("CUDA_VISIBLE_DEVICES=unset "), ln


def test_start_shared_explicit_above_ceiling_is_refused_before_anything(stub_bin: Path) -> None:
    proc = _run_pd(
        stub_bin, "start", "fake/test-model",
        VLLM_GPU_MEMORY_UTILIZATION="0.90", **GOOD_BUDGETS,
    )
    _assert_refused_before_anything(proc)
    err = proc.stderr
    assert "vLLM startup check" in err
    assert "--gpu-memory-utilization" in err and "0.90" in err and "0.50" in err
    # both fixes named: lower the value or pin distinct GPUs per role
    assert "VLLM_GPU_MEMORY_UTILIZATION" in err
    assert "CAGE_PD_PREFILL_GPUS" in err and "CAGE_PD_DECODE_GPUS" in err


def test_start_shared_explicit_at_or_below_ceiling_is_honored(stub_bin: Path) -> None:
    proc = _run_pd(
        stub_bin, "start", "fake/test-model",
        VLLM_GPU_MEMORY_UTILIZATION="0.40", **GOOD_BUDGETS,
    )
    assert "--gpu-memory-utilization 0.40" in _args_line(proc.stdout, "prefill")
    assert "--gpu-memory-utilization 0.40" in _args_line(proc.stdout, "decode")
    line = _decision_line(proc.stdout)
    assert "shared" in line and "0.40" in line and "explicit" in line
    proc = _run_pd(
        stub_bin, "start", "fake/test-model",
        VLLM_GPU_MEMORY_UTILIZATION="0.50", **GOOD_BUDGETS,
    )
    assert "--gpu-memory-utilization 0.50" in _args_line(proc.stdout, "prefill")


def test_start_distinct_explicit_override_is_honored(stub_bin: Path) -> None:
    proc = _run_pd(
        stub_bin, "start", "fake/test-model",
        VLLM_GPU_MEMORY_UTILIZATION="0.95", **GOOD_BUDGETS, **DISTINCT_PINS,
    )
    assert "--gpu-memory-utilization 0.95" in _args_line(proc.stdout, "prefill")
    assert "--gpu-memory-utilization 0.95" in _args_line(proc.stdout, "decode")
    assert "explicit" in _decision_line(proc.stdout)


@pytest.mark.parametrize(
    "pins",
    [
        {"CAGE_PD_PREFILL_GPUS": "0"},                                 # decode unpinned
        {"CAGE_PD_DECODE_GPUS": "1"},                                  # prefill unpinned
        {"CAGE_PD_PREFILL_GPUS": "0", "CAGE_PD_DECODE_GPUS": "0"},     # same device
        {"CAGE_PD_PREFILL_GPUS": "0,1", "CAGE_PD_DECODE_GPUS": "1,2"}, # overlapping TP sets
        {"CAGE_PD_PREFILL_GPUS": "a", "CAGE_PD_DECODE_GPUS": "1"},     # non-numeric
        {"CAGE_PD_PREFILL_GPUS": "0,", "CAGE_PD_DECODE_GPUS": "1"},    # empty device
    ],
    ids=["decode-unpinned", "prefill-unpinned", "same-device", "overlap", "non-numeric", "empty-device"],
)
def test_start_refuses_partial_overlapping_or_malformed_pins(
    stub_bin: Path, pins: Dict[str, str]
) -> None:
    proc = _run_pd(stub_bin, "start", "fake/test-model", **GOOD_BUDGETS, **pins)
    _assert_refused_before_anything(proc)
    assert "CAGE_PD_PREFILL_GPUS" in proc.stderr and "CAGE_PD_DECODE_GPUS" in proc.stderr


@pytest.mark.parametrize("bad", ["abc", "0", "1.5", "0,9", "-0.4", ""])
def test_start_refuses_malformed_mem_util(stub_bin: Path, bad: str) -> None:
    proc = _run_pd(
        stub_bin, "start", "fake/test-model",
        VLLM_GPU_MEMORY_UTILIZATION=bad, **GOOD_BUDGETS, **DISTINCT_PINS,
    )
    if bad == "":
        # empty = unset for the sourced serving config: distinct default 0.90
        assert "--gpu-memory-utilization 0.90" in _args_line(proc.stdout, "prefill")
        return
    _assert_refused_before_anything(proc)
    assert "VLLM_GPU_MEMORY_UTILIZATION" in proc.stderr


def test_stop_is_never_gated_by_gpu_share_rule(stub_bin: Path) -> None:
    proc = _run_pd(
        stub_bin, "stop",
        VLLM_GPU_MEMORY_UTILIZATION="0.90",  # would refuse a shared start
    )
    assert proc.returncode == 0, proc.stderr
    assert "Stopping" in proc.stdout


# --- bash-level unit test of the resolver (sourced, no dispatch) ------------


def _resolve_in_bash(stub_bin: Path, **env_extra: str):
    """Source the launcher (the `BASH_SOURCE != $0` guard skips the dispatch),
    call cage_resolve_pd_gpu_share, and print its outputs."""
    env = _clean_env(**env_extra)
    env["PATH"] = f"{stub_bin}:{env.get('PATH', '/usr/bin:/bin')}"
    script = (
        f'source "{PD_SH}" && cage_resolve_pd_gpu_share '
        '&& printf "RESOLVED %s %s %s %s %s\\n" '
        '"$PD_GPU_SHARE" "$PD_MEM_UTIL" "$PD_MEM_UTIL_SOURCE" '
        '"${PD_PREFILL_CUDA:-unset}" "${PD_DECODE_CUDA:-unset}"'
    )
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, env=env, timeout=60,
    )


def _resolved(proc) -> List[str]:
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESOLVED ")]
    assert len(lines) == 1, f"{proc.stdout}\n{proc.stderr}"
    return lines[0].split()[1:]


def test_bash_resolver_shared_default(stub_bin: Path) -> None:
    proc = _resolve_in_bash(stub_bin)
    assert proc.returncode == 0, proc.stderr
    assert _resolved(proc) == ["shared", "0.45", "default", "unset", "unset"]


def test_bash_resolver_distinct_default(stub_bin: Path) -> None:
    proc = _resolve_in_bash(stub_bin, CAGE_PD_PREFILL_GPUS="0,1", CAGE_PD_DECODE_GPUS="2,3")
    assert proc.returncode == 0, proc.stderr
    assert _resolved(proc) == ["distinct", "0.90", "default", "0,1", "2,3"]


def test_bash_resolver_shared_explicit_paths(stub_bin: Path) -> None:
    proc = _resolve_in_bash(stub_bin, VLLM_GPU_MEMORY_UTILIZATION="0.30")
    assert _resolved(proc) == ["shared", "0.30", "explicit", "unset", "unset"]
    proc = _resolve_in_bash(stub_bin, VLLM_GPU_MEMORY_UTILIZATION="0.51")
    assert proc.returncode != 0
    assert "REFUSING" in proc.stderr and "vLLM startup check" in proc.stderr
    assert "RESOLVED" not in proc.stdout


def test_bash_resolver_sees_the_pre_source_value_not_the_lib_default(stub_bin: Path) -> None:
    # _serving_config.sh exports VLLM_GPU_MEMORY_UTILIZATION=0.90 whenever it
    # is sourced; the resolver must treat that as a DEFAULT (shared -> 0.45),
    # not as an explicit 0.90 that would refuse every unpinned pd launch.
    proc = _resolve_in_bash(stub_bin)
    assert proc.returncode == 0, proc.stderr
    assert _resolved(proc)[:3] == ["shared", "0.45", "default"]


def test_sourcing_the_launcher_does_not_dispatch(stub_bin: Path) -> None:
    env = _clean_env()
    env["PATH"] = f"{stub_bin}:{env.get('PATH', '/usr/bin:/bin')}"
    proc = subprocess.run(
        ["bash", "-c", f'source "{PD_SH}"; echo SOURCED_OK'],
        capture_output=True, text=True, env=env, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "SOURCED_OK" in proc.stdout
    assert "Usage:" not in proc.stdout and "Stopping" not in proc.stdout


# ---------------------------------------------------------------------------
# 2. behavioral: refusal matrix (before ANY process action) + ungated stop
# ---------------------------------------------------------------------------


def _assert_refused_before_anything(proc) -> None:
    assert proc.returncode != 0
    assert "REFUSING" in proc.stderr and "FATAL" in proc.stderr
    assert "Server args" not in proc.stdout, "refusal must precede composition"
    # start is self-cleaning (stop first) — the refusal must fire BEFORE that
    # teardown, or a healthy stack dies for a launch that cannot happen.
    assert "Stopping" not in proc.stdout


@pytest.mark.parametrize(
    "env",
    [
        {},  # both budgets missing
        {"CAGE_KV_BUDGET_BYTES_PREFILL": "3000000000"},  # decode missing
        {"CAGE_KV_BUDGET_BYTES_DECODE": "7000000000"},  # prefill missing
    ],
    ids=["both-missing", "decode-missing", "prefill-missing"],
)
def test_start_refuses_missing_role_budgets(stub_bin: Path, env: Dict[str, str]) -> None:
    proc = _run_pd(stub_bin, "start", "fake/test-model", **env)
    _assert_refused_before_anything(proc)
    missing = {
        "CAGE_KV_BUDGET_BYTES_PREFILL",
        "CAGE_KV_BUDGET_BYTES_DECODE",
    } - set(env)
    for var in missing:
        assert var in proc.stderr, f"refusal must name the missing {var}"


@pytest.mark.parametrize("bad", ["abc", "0", "-2", "2.5", "00"])
def test_start_refuses_malformed_role_budgets(stub_bin: Path, bad: str) -> None:
    for var in ("CAGE_KV_BUDGET_BYTES_PREFILL", "CAGE_KV_BUDGET_BYTES_DECODE"):
        env = dict(GOOD_BUDGETS)
        env[var] = bad
        proc = _run_pd(stub_bin, "start", "fake/test-model", **env)
        _assert_refused_before_anything(proc)
        assert var in proc.stderr


def test_start_refuses_ambient_single_instance_budget(stub_bin: Path) -> None:
    # Two caps for the same pools in different scopes = refusal, never a
    # precedence rule.
    proc = _run_pd(
        stub_bin, "start", "fake/test-model",
        CAGE_KV_BUDGET_BYTES="1000000000", **GOOD_BUDGETS,
    )
    _assert_refused_before_anything(proc)
    assert "CAGE_KV_BUDGET_BYTES" in proc.stderr
    proc = _run_pd(
        stub_bin, "start", "fake/test-model",
        CAGE_VLLM_GPU_BLOCKS_OVERRIDE="512", **GOOD_BUDGETS,
    )
    _assert_refused_before_anything(proc)


def test_start_refuses_ambient_kv_transfer_config(stub_bin: Path) -> None:
    # The pd launcher OWNS the per-role connector JSON.
    proc = _run_pd(
        stub_bin, "start", "fake/test-model",
        VLLM_KV_TRANSFER_CONFIG='{"kv_connector":"X","kv_role":"kv_both"}',
        **GOOD_BUDGETS,
    )
    _assert_refused_before_anything(proc)
    assert "VLLM_KV_TRANSFER_CONFIG" in proc.stderr


def test_start_refuses_bad_tp(stub_bin: Path) -> None:
    proc = _run_pd(
        stub_bin, "start", "fake/test-model",
        CAGE_VLLM_TENSOR_PARALLEL="abc", **GOOD_BUDGETS,
    )
    _assert_refused_before_anything(proc)
    assert "CAGE_VLLM_TENSOR_PARALLEL" in proc.stderr


def test_stop_is_never_gated_by_validation(stub_bin: Path) -> None:
    # Teardown discipline: `stop` succeeds even under a poisonous env.
    proc = _run_pd(
        stub_bin, "stop",
        CAGE_KV_BUDGET_BYTES_PREFILL="abc",  # would refuse a start
    )
    assert proc.returncode == 0, proc.stderr
    assert "Stopping" in proc.stdout


def test_status_reports_pending_data_path_never_pass(stub_bin: Path) -> None:
    proc = _run_pd(stub_bin, "status")
    # nothing running under the stub PATH -> nonzero, but the pending stamp
    # must be there regardless (a pending check reports PENDING, never PASS).
    assert proc.returncode != 0
    assert "PENDING [VERIFY-LIVE at Run-C-prime preflight]" in proc.stdout


# ---------------------------------------------------------------------------
# 3. pd_proxy unit tests — in-process stub upstreams, http.client
# ---------------------------------------------------------------------------

#: The engine-shaped kv_transfer_params the prefill stub returns. NO "source"
#: field — exactly the pre-NIXL-verification reality the campaign gate must
#: refuse downstream; the proxy must forward it verbatim and add nothing.
STUB_KV_TRANSFER_PARAMS = {"remote_engine_id": "stub", "remote_block_ids": [1, 2, 3]}

DECODE_PAYLOAD = json.dumps(
    {"id": "decode-1", "choices": [{"text": "answer"}]}
).encode("utf-8")


class _StubUpstream:
    """A recording OpenAI-shaped upstream (prefill or decode role)."""

    def __init__(self, role: str, journal: List[Tuple[str, Dict[str, Any]]]):
        self.role = role
        self.journal = journal
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a: Any) -> None:  # quiet
                pass

            def do_GET(self) -> None:  # noqa: N802
                body = b'{"status":"ok"}'
                self.send_response(200 if self.path == "/health" else 404)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                outer.journal.append((outer.role, payload))
                if outer.role == "prefill":
                    body = json.dumps(
                        {
                            "id": "prefill-1",
                            "choices": [{"text": "x"}],
                            "kv_transfer_params": STUB_KV_TRANSFER_PARAMS,
                        }
                    ).encode("utf-8")
                else:
                    body = DECODE_PAYLOAD
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture()
def pd_stack():
    """(journal, prefill stub, decode stub, live proxy port) with teardown."""
    journal: List[Tuple[str, Dict[str, Any]]] = []
    prefill = _StubUpstream("prefill", journal)
    decode = _StubUpstream("decode", journal)
    proxy = pd_proxy.build_server(0, prefill.url, decode.url)
    proxy_port = proxy.server_address[1]
    thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    thread.start()
    try:
        yield journal, prefill, decode, proxy_port
    finally:
        proxy.shutdown()
        proxy.server_close()
        prefill.close()
        decode.close()


def _post(port: int, path: str, body: Dict[str, Any]):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        conn.request(
            "POST", path, body=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


def _get(port: int, path: str):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


def test_proxy_prefill_then_decode_ordering_and_clamp(pd_stack) -> None:
    journal, _, _, port = pd_stack
    status, body = _post(
        port, "/v1/completions",
        {"model": "m", "prompt": "p", "max_tokens": 64, "stream": False},
    )
    assert status == 200
    assert [role for role, _ in journal] == ["prefill", "decode"], (
        "the 1P1D pattern is prefill FIRST, decode second"
    )
    prefill_body = journal[0][1]
    decode_body = journal[1][1]
    # prefill: generation clamped to 1 token, stream forced off, same prompt
    assert prefill_body["max_tokens"] == 1
    assert prefill_body["stream"] is False
    assert prefill_body["prompt"] == "p"
    # decode: the ORIGINAL request's generation budget
    assert decode_body["max_tokens"] == 64
    assert decode_body["prompt"] == "p"
    # client sees the decode response verbatim (streaming passthrough)
    assert body == DECODE_PAYLOAD


def test_proxy_passes_kv_transfer_params_untouched_and_never_stamps_source(
    pd_stack,
) -> None:
    journal, _, _, port = pd_stack
    _post(port, "/v1/completions", {"model": "m", "prompt": "p", "max_tokens": 8})
    decode_body = journal[1][1]
    # verbatim passthrough: byte-equal structure, nothing added or renamed
    assert decode_body["kv_transfer_params"] == STUB_KV_TRANSFER_PARAMS
    # THE T3.3 interplay: no proxy-invented "source" anywhere in the
    # forwarded body — provenance belongs to the engine alone, and the
    # campaign gate refusing a source-less stamp is the correct outcome.
    assert "source" not in decode_body["kv_transfer_params"]
    assert "source" not in decode_body
    assert "source" not in json.dumps(decode_body)


def test_proxy_health_requires_both_upstreams_and_reports_pending(pd_stack) -> None:
    _, _, decode, port = pd_stack
    status, body = _get(port, "/health")
    assert status == 200
    doc = json.loads(body)
    assert doc["prefill"] == "ok" and doc["decode"] == "ok"
    # a pending check reports PENDING, never PASS
    assert doc["pd_data_path"].startswith("PENDING")
    assert "VERIFY-LIVE at Run-C-prime preflight" in doc["pd_data_path"]
    # half-up pair: health must fail closed (503), not report ready
    decode.close()
    status, body = _get(port, "/health")
    assert status == 503
    doc = json.loads(body)
    assert doc["decode"] != "ok"


def test_proxy_refuses_unparseable_body_and_unknown_paths(pd_stack) -> None:
    journal, _, _, port = pd_stack
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        conn.request("POST", "/v1/completions", body=b"{not json")
        resp = conn.getresponse()
        assert resp.status == 400
        resp.read()
    finally:
        conn.close()
    status, _ = _post(port, "/not-v1", {"x": 1})
    assert status == 404
    assert journal == [], "no refused request may reach an upstream"


def test_proxy_build_server_refuses_portless_upstreams() -> None:
    with pytest.raises(ValueError, match="http://host:port"):
        pd_proxy.build_server(0, "http://localhost", "http://localhost:8200")


# ---------------------------------------------------------------------------
# 4. run_campaign.py — pd emission + --allow-pd gating (stub runner)
# ---------------------------------------------------------------------------

_ANCHOR_DEMAND = 10_000_000_000


def _floor_table(tmp_path: Path) -> Path:
    rows = [
        {
            "r": r,
            "demand_bytes": _ANCHOR_DEMAND,
            "budget_bytes": int(r * _ANCHOR_DEMAND),
            "lambda_kv_rps": 2.0 * r,
            "lambda_compute_rps": None,
            "lambda_star_pred_rps": 2.0 * r,
            "lambda_star_basis": "test",
        }
        for r in (1.5, 1.0, 0.75, 0.5, 0.25)
    ]
    doc = {
        "schema": "floor-table-v1",
        "generated_inputs": {
            "model": "qwen3-14b", "engine": "vllm",
            "kv_dtype": "bf16", "grid": "full",
        },
        "rows": rows,
    }
    path = tmp_path / "floor_table.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _pd_grid(**overrides: Any):
    """A minimal grid: 1 executable vllm DIST/pd cell (dist_pd_split=0.25,
    an EXACT binary fraction so the independent budget arithmetic below has
    no float ambiguity)."""
    kwargs: Dict[str, Any] = dict(
        session="a", group="A", model="qwen3-14b",
        f1_baselines=(), f1_engines=("vllm",), f1_datasets=("squad_v2",),
        hf_oracle_cells=(),
        f2_baselines=(), f2_engines=("vllm",), f2_budgets=(), f2_rates=(),
        f2_dataset="qasper",
        f3_baselines=(), f3_engines=("vllm",), f3_budgets=(), f3_rates=(),
        f3_dataset="qasper",
        dist_cells=(("B3", "vllm", "pd"),),
        dist_pd_split=0.25,
        dist_budget_r=1.0,
    )
    kwargs.update(overrides)
    return rc.SessionGrid(**kwargs)


def _plan_for(grid, floor_path: Path, launcher_cmds=None, runner_cmd=("stub",)):
    floor = rc.load_floor_table(floor_path)
    orig = rc.SESSION_GRIDS
    rc.SESSION_GRIDS = {grid.session: grid}
    try:
        return rc.build_plan(
            grid.session, floor, window_duration_s=60.0,
            runner_cmd=runner_cmd, launcher_cmds=launcher_cmds,
        )
    finally:
        rc.SESSION_GRIDS = orig


def _cells(plan):
    return [s for s in plan["steps"] if s["kind"] == "cell"]


def _relaunches(plan):
    return [s for s in plan["steps"] if s["kind"] == "relaunch"]


#: independent §6.5 arithmetic: budget = floor(r × D); prefill = floor(split
#: × budget); decode = the exact remainder. Restated here, never re-calling
#: the planner it checks.
_EXPECT_TOTAL = _ANCHOR_DEMAND  # r = 1.0
_EXPECT_PREFILL = 2_500_000_000  # floor(0.25 × 1e10), exact binary fraction
_EXPECT_DECODE = _EXPECT_TOTAL - _EXPECT_PREFILL


class TestPdPlanEmission:
    def test_pd_cell_unblocked_with_gate_label(self, tmp_path):
        plan = _plan_for(_pd_grid(), _floor_table(tmp_path))
        pd = [s for s in _cells(plan) if s["cellspec"]["topology"] == "pd"]
        assert len(pd) == 1
        assert pd[0]["blocked_on"] is None
        assert pd[0]["gate"] == "--allow-pd + PD preflight smoke"
        assert pd[0]["serving"]["topology"] == "pd"
        assert plan["counts"]["blocked"] == 0

    def test_pd_relaunch_invokes_pd_launcher_with_role_budgets(self, tmp_path):
        # DEFAULT launcher cmds: the real pd script, verb `start` (the pd
        # launcher has no restart — start is self-cleaning by design).
        plan = _plan_for(_pd_grid(), _floor_table(tmp_path))
        pd_relaunches = [s for s in _relaunches(plan) if s["topology"] == "pd"]
        assert len(pd_relaunches) == 1
        step = pd_relaunches[0]
        assert step["argv"] == [
            "bash", "scripts/2_serving/manage_vllm_pd.sh", "start", "Qwen/Qwen3-14B",
        ]
        env = step["env"]
        assert env["CAGE_KV_BUDGET_BYTES_PREFILL"] == str(_EXPECT_PREFILL)
        assert env["CAGE_KV_BUDGET_BYTES_DECODE"] == str(_EXPECT_DECODE)
        # §6.5 exact-sum invariant, independent arithmetic
        assert (
            int(env["CAGE_KV_BUDGET_BYTES_PREFILL"])
            + int(env["CAGE_KV_BUDGET_BYTES_DECODE"])
            == _EXPECT_TOTAL
        )
        assert env["CAGE_TELEMETRY_ENDPOINTS"] == (
            "prefill=http://localhost:8100,decode=http://localhost:8200"
        )
        assert step["pd"] == {
            "split": 0.25,
            "budget_r": 1.0,
            "budget_bytes_total": _EXPECT_TOTAL,
            "prefill_bytes": _EXPECT_PREFILL,
            "decode_bytes": _EXPECT_DECODE,
        }

    def test_pd_cell_runs_under_its_pd_relaunch_boundary(self, tmp_path):
        plan = _plan_for(_pd_grid(), _floor_table(tmp_path))
        current_topology = None
        for s in plan["steps"]:
            if s["kind"] == "relaunch":
                current_topology = s["topology"]
            elif s["cellspec"]["topology"] == "pd":
                assert current_topology == "pd", (
                    "a pd cell must never run against a single-instance server"
                )

    def test_pd_on_sglang_and_tp_overlay_stay_blocked(self, tmp_path):
        grid = _pd_grid(
            dist_cells=(
                ("B3", "vllm", "pd"),
                ("B3", "sglang", "pd"),
                ("B3", "vllm", "tp"),
            ),
        )
        plan = _plan_for(grid, _floor_table(tmp_path))
        by = {
            (s["cellspec"]["engine"], s["cellspec"]["topology"]): s
            for s in _cells(plan)
        }
        assert by[("vllm", "pd")]["blocked_on"] is None
        assert "PD launcher" in by[("sglang", "pd")]["blocked_on"]
        assert by[("vllm", "tp")]["blocked_on"] == rc.TP_DIST_BLOCKED_ON
        # blocked cells emit NO pd relaunch beyond the one executable config
        assert len([s for s in _relaunches(plan) if s["topology"] == "pd"]) == 1

    def test_pd_plan_roundtrips_through_load_plan(self, tmp_path):
        plan = _plan_for(_pd_grid(), _floor_table(tmp_path))
        out = tmp_path / "plan_pd.json"
        out.write_text(json.dumps(plan), encoding="utf-8")
        loaded = rc.load_plan(out)
        assert loaded["counts"]["cells"] == 1

    def test_pd_split_registration_refuses_illegal_values(self):
        for bad in (0.0, 1.0, -0.5, 2.0):
            with pytest.raises(rc.PlanError, match="dist_pd_split"):
                _pd_grid(dist_pd_split=bad)


# ---------------------------------------------------------------------------
# 'run' gating on stubs
# ---------------------------------------------------------------------------

_STUB_SOURCE = """\
import json, os, sys
with open(os.environ["STUB_CALLS"], "a", encoding="utf-8") as fh:
    fh.write(json.dumps({
        "argv": sys.argv[1:],
        "env": {k: v for k, v in os.environ.items() if k.startswith("CAGE_")},
    }) + "\\n")
sys.exit(0)
"""


@pytest.fixture()
def stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    stub_path = tmp_path / "stub.py"
    stub_path.write_text(_STUB_SOURCE, encoding="utf-8")
    calls_path = tmp_path / "stub_calls.jsonl"
    monkeypatch.setenv("STUB_CALLS", str(calls_path))

    class Stub:
        cmd = (sys.executable, str(stub_path))

        @staticmethod
        def calls() -> List[Dict[str, Any]]:
            if not calls_path.exists():
                return []
            return [
                json.loads(line)
                for line in calls_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

    return Stub


def _stub_pd_plan(tmp_path: Path, stub) -> Dict[str, Any]:
    return _plan_for(
        _pd_grid(), _floor_table(tmp_path),
        launcher_cmds={
            "vllm": stub.cmd, "sglang": stub.cmd, "vllm:pd": stub.cmd,
        },
        runner_cmd=stub.cmd,
    )


def _run_root(tmp_path: Path) -> Path:
    root = tmp_path / "results" / "camp" / "a" / "run-001"
    root.parent.mkdir(parents=True, exist_ok=True)
    return root


class TestAllowPdGating:
    def test_run_refuses_pd_cells_without_allow_pd(self, tmp_path, stub):
        plan = _stub_pd_plan(tmp_path, stub)
        with pytest.raises(rc.RunError, match=r"--allow-pd"):
            rc.run_plan(plan, _run_root(tmp_path))
        assert stub.calls() == [], (
            "NOTHING may execute when pd cells lack the --allow-pd consent"
        )

    def test_refusal_names_the_preflight_gate(self, tmp_path, stub):
        plan = _stub_pd_plan(tmp_path, stub)
        with pytest.raises(rc.RunError, match="Run-C-prime preflight"):
            rc.run_plan(plan, _run_root(tmp_path))

    def test_allow_pd_executes_pd_relaunch_and_cell(self, tmp_path, stub):
        plan = _stub_pd_plan(tmp_path, stub)
        root = _run_root(tmp_path)
        assert rc.run_plan(plan, root, allow_pd=True) == 0
        calls = stub.calls()
        assert len(calls) == 2  # pd relaunch + the cell
        relaunch, cell = calls
        # the pd launcher verb is `start` (self-cleaning; no restart verb)
        assert relaunch["argv"][0] == "start"
        assert relaunch["env"]["CAGE_KV_BUDGET_BYTES_PREFILL"] == str(_EXPECT_PREFILL)
        assert relaunch["env"]["CAGE_KV_BUDGET_BYTES_DECODE"] == str(_EXPECT_DECODE)
        assert relaunch["env"]["CAGE_TELEMETRY_ENDPOINTS"].startswith("prefill=")
        # identity seam: the cell subprocess carries the pd topology axis
        assert cell["env"]["CAGE_CELL_TOPOLOGY"] == "pd"
        assert cell["env"]["CAGE_CELL_FAMILY"] == "DIST"
        assert cell["argv"][-2:] == ["--campaign-root", str(root)]

    def test_cli_help_names_the_preflight_gate(self, capsys):
        with pytest.raises(SystemExit):
            rc.main(["run", "--help"])
        # argparse wraps long help at hyphens ("Run-C-\nprime") — unwrap the
        # hyphen line-breaks before asserting the gate name survives intact.
        out = re.sub(r"-\n\s*", "-", capsys.readouterr().out)
        assert "--allow-pd" in out
        assert "Run-C-prime" in out
