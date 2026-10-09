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
   reports the data path as PENDING, never PASS. Cold start per window on
   pd (ADR-0102 amendment 2026-09-19, Batch 2 W2-R1): GET /metrics relays
   ONLY the runner's in-flight gauge, one relabeled sample per role (503
   unless both roles are readable), and POST /reset_prefix_cache fans out
   to both roles and is 200 only when both flushed (else 502 naming the
   role), so the runner's strict reset works against the proxy.

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
    # UCX_ stripped too (review 2026-10-01 LOW 4): the capture records the
    # operator's UCX_* values, so a developer shell must not leak into the
    # "nothing set" assertions.
    env = {
        k: v for k, v in os.environ.items()
        if not k.startswith(("CAGE_", "VLLM_", "SGLANG_", "PD_", "UCX_"))
    }
    # S0F-23: the suite's tmp log root (tests/conftest.py) rides through the
    # CAGE_ strip so a launcher start never writes under <repo>/logs/.
    if "CAGE_LOG_ROOT" in os.environ:
        env["CAGE_LOG_ROOT"] = os.environ["CAGE_LOG_ROOT"]
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
    # nowhere else; S0F-13: the per-role NIXL side-channel port likewise) when
    # CAGE_TEST_VLLM_JOURNAL names a file.
    # ucx_tls is journaled so "the launcher exports no UCX_TLS" is a real
    # assertion (review 2026-10-01 LOW 4), kept before nixl_port so the
    # endswith(" nixl_port=...") pins hold.
    vllm_stub = (
        "#!/bin/sh\n"
        'if [ -n "${CAGE_TEST_VLLM_JOURNAL:-}" ]; then\n'
        "  printf 'CUDA_VISIBLE_DEVICES=%s args=%s ucx_tls=%s nixl_port=%s\\n' "
        '"${CUDA_VISIBLE_DEVICES-unset}" "$*" "${UCX_TLS-unset}" '
        '"${VLLM_NIXL_SIDE_CHANNEL_PORT-unset}" >> "$CAGE_TEST_VLLM_JOURNAL"\n'
        "fi\n"
        "exit 0\n"
    )
    # The interpreter the nixl import gate probes (CAGE_PD_PYTHON): drains the
    # probe script, then answers with CAGE_TEST_PD_PY_MSG / CAGE_TEST_PD_PY_RC
    # (default: "ok", 0 -- the gate passes, as on a pod with nixl 0.9.0).
    pd_python_stub = (
        "#!/bin/sh\n"
        "cat >/dev/null\n"
        'echo "${CAGE_TEST_PD_PY_MSG:-ok}"\n'
        'exit "${CAGE_TEST_PD_PY_RC:-0}"\n'
    )
    for name, body in {
        "pgrep": "#!/bin/sh\nexit 1\n",
        "curl": "#!/bin/sh\nexit 1\n",
        "nvidia-smi": "#!/bin/sh\nexit 1\n",
        "pkill": "#!/bin/sh\nexit 0\n",
        "vllm": vllm_stub,
        "pd_python_stub": pd_python_stub,
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
    # S0F-13: the nixl import gate probes CAGE_PD_PYTHON; the stub passes by
    # default (a pod with nixl 0.9.0), refusal tests flip CAGE_TEST_PD_PY_RC.
    env.setdefault("CAGE_PD_PYTHON", str(stub_bin / "pd_python_stub"))
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
    # S0F-22 Batch 1 replaced the "gate refusing is the correct outcome"
    # doctrine: the proxy itself asks for the ticket, refuses a missing one
    # (502) before the decode, and relays it verbatim in the header.
    assert "502" in proxy_doc and "TICKET_HEADER" in proxy_doc
    assert "do_remote_decode" in proxy_doc and "ignore_eos" in proxy_doc


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
# 2b. S0F-13 (ADR-0128): nixl install pins, per-role side-channel ports, the
# import gate before the self-cleaning stop, and the joint liveness wait.
# S0 facts (MyDocs/RunPod/S0_RUN_2026-09-30.md, backlog S0F-13): nothing
# installed nixl (layer 1); the nixl 1.x wheels ship nixl_ep built for torch
# 2.11+ and vLLM imports it on sight (layer 2); both roles defaulted
# VLLM_NIXL_SIDE_CHANNEL_PORT to 5600 on one host and the prefill's handshake
# listener died EADDRINUSE (layer 3); attempts 1 and 2 waited 11 min each on
# dead role processes.
# ---------------------------------------------------------------------------

SETUP_RUNPOD_SH = REPO_ROOT / "scripts" / "runpod" / "setup_runpod.sh"
VLLM_COMPAT_MD = REPO_ROOT / "docs" / "VLLM_COMPATIBILITY.md"
RUNBOOK_MD = REPO_ROOT / "docs" / "RUNBOOK.md"


def test_setup_installs_nixl_0_9_0_in_the_vllm_pip_call() -> None:
    text = SETUP_RUNPOD_SH.read_text(encoding="utf-8")
    assert 'NIXL_VERSION="${NIXL_VERSION:-0.9.0}"' in text
    # one pip call: vLLM's resolver sees the connector wheels at install time
    assert ('pip install "vllm==${VLLM_VERSION}" "nixl==${NIXL_VERSION}" '
            '"nixl-cu12==${NIXL_VERSION}"') in text, (
        "step 2 must install vllm, nixl and nixl-cu12 in ONE pip call")
    # the pod runs CUDA 12.8 torch; the 0.9.0 dispatcher tries cu13 first when present
    assert '"nixl-cu13==' not in text


def test_compatibility_doc_pins_the_same_nixl_version() -> None:
    doc = VLLM_COMPAT_MD.read_text(encoding="utf-8")
    script = SETUP_RUNPOD_SH.read_text(encoding="utf-8")
    pinned = re.search(r'NIXL_VERSION="\$\{NIXL_VERSION:-([^}]+)\}"', script).group(1)
    row = [ln for ln in doc.splitlines() if ln.startswith("| NIXL")]
    assert len(row) == 1, "section 7 engine-pin table needs exactly one NIXL row"
    assert f"**{pinned}**" in row[0]
    assert "nixl-cu12" in row[0] and "VLLM_NIXL_SIDE_CHANNEL_PORT" in doc


def test_runbook_carries_the_no_overlapping_stop_rule() -> None:
    text = RUNBOOK_MD.read_text(encoding="utf-8")
    assert "never run a pd `stop` while another chain's `start` is live" in text
    assert "UCX_LOG_LEVEL=info" in text


def test_start_gives_each_role_its_own_side_channel_port(stub_bin: Path, tmp_path: Path) -> None:
    journal = tmp_path / "vllm_journal.txt"
    proc = _run_pd(
        stub_bin, "start", "fake/test-model",
        CAGE_RUN_ROOT=str(tmp_path), CAGE_TEST_VLLM_JOURNAL=str(journal), **GOOD_BUDGETS,
    )
    lines = _read_journal(journal, 2)
    prefill = [ln for ln in lines if "--port 8100" in ln]
    decode = [ln for ln in lines if "--port 8200" in ln]
    assert len(prefill) == 1 and len(decode) == 1, lines
    assert prefill[0].endswith(" nixl_port=5600")
    assert decode[0].endswith(" nixl_port=5601")
    # the banner says so, and the per-role capture records it
    assert "nixl side channel: prefill=tcp://localhost:5600 decode=tcp://localhost:5601" in proc.stdout
    cfg_dir = tmp_path / "observability" / "serving_configs"
    pcap = json.loads(next(cfg_dir.glob("*_pd-prefill.json")).read_text(encoding="utf-8"))
    dcap = json.loads(next(cfg_dir.glob("*_pd-decode.json")).read_text(encoding="utf-8"))
    assert pcap["nixl_side_channel_port"] == 5600
    assert dcap["nixl_side_channel_port"] == 5601
    # part 5 (record only): no UCX_* set -> an empty record, and the child env
    # carries no UCX_TLS the launcher could have exported (the stub journals it)
    assert pcap["ucx_env"] == {} and dcap["ucx_env"] == {}
    assert " ucx_tls=unset " in prefill[0] and " ucx_tls=unset " in decode[0]


def test_side_channel_port_overrides_and_ucx_record(stub_bin: Path, tmp_path: Path) -> None:
    journal = tmp_path / "vllm_journal.txt"
    _run_pd(
        stub_bin, "start", "fake/test-model",
        CAGE_RUN_ROOT=str(tmp_path), CAGE_TEST_VLLM_JOURNAL=str(journal),
        CAGE_PD_NIXL_PORT_PREFILL="5700", CAGE_PD_NIXL_PORT_DECODE="5701",
        UCX_LOG_LEVEL="info", **GOOD_BUDGETS,
    )
    lines = _read_journal(journal, 2)
    assert any(ln.endswith(" nixl_port=5700") and "--port 8100" in ln for ln in lines), lines
    assert any(ln.endswith(" nixl_port=5701") and "--port 8200" in ln for ln in lines), lines
    cfg_dir = tmp_path / "observability" / "serving_configs"
    pcap = json.loads(next(cfg_dir.glob("*_pd-prefill.json")).read_text(encoding="utf-8"))
    assert pcap["nixl_side_channel_port"] == 5700
    # the operator's UCX_* values are RECORDED (ADR-0128 part 5), nothing added
    assert pcap["ucx_env"] == {"UCX_LOG_LEVEL": "info"}


@pytest.mark.parametrize(
    "env, needle",
    [
        ({"CAGE_PD_NIXL_PORT_PREFILL": "5700", "CAGE_PD_NIXL_PORT_DECODE": "5700"}, "EADDRINUSE"),
        ({"CAGE_PD_NIXL_PORT_PREFILL": "8200"}, "8200"),            # equals the decode HTTP port
        ({"CAGE_PD_NIXL_PORT_DECODE": "abc"}, "CAGE_PD_NIXL_PORT_DECODE"),
        ({"CAGE_PD_NIXL_PORT_PREFILL": "0"}, "CAGE_PD_NIXL_PORT_PREFILL"),
        ({"VLLM_NIXL_SIDE_CHANNEL_PORT": "5600"}, "VLLM_NIXL_SIDE_CHANNEL_PORT"),  # the launcher owns it
    ],
    ids=["equal-roles", "collides-with-http", "non-numeric", "zero", "ambient-env"],
)
def test_start_refuses_side_channel_port_misconfiguration(
    stub_bin: Path, env: Dict[str, str], needle: str
) -> None:
    proc = _run_pd(stub_bin, "start", "fake/test-model", **GOOD_BUDGETS, **env)
    _assert_refused_before_anything(proc)
    assert needle in proc.stderr, proc.stderr


def test_import_gate_passes_and_is_printed(stub_bin: Path) -> None:
    proc = _run_pd(stub_bin, "start", "fake/test-model", **GOOD_BUDGETS)
    assert "[cage] nixl import gate: ok" in proc.stdout
    # it ran BEFORE the self-cleaning teardown
    assert proc.stdout.index("nixl import gate") < proc.stdout.index("Stopping")


def test_import_gate_refuses_missing_nixl_before_anything(stub_bin: Path) -> None:
    proc = _run_pd(
        stub_bin, "start", "fake/test-model",
        CAGE_TEST_PD_PY_RC="2", CAGE_TEST_PD_PY_MSG="nixl: ModuleNotFoundError: No module named 'nixl'",
        **GOOD_BUDGETS,
    )
    _assert_refused_before_anything(proc)
    assert "nixl._api" in proc.stderr and "nixl==0.9.0" in proc.stderr
    assert "No module named 'nixl'" in proc.stderr  # the probe's own words are shown


def test_import_gate_refuses_broken_nixl_ep_before_anything(stub_bin: Path) -> None:
    proc = _run_pd(
        stub_bin, "start", "fake/test-model",
        CAGE_TEST_PD_PY_RC="3", CAGE_TEST_PD_PY_MSG="nixl_ep: ImportError: undefined symbol: torch",
        **GOOD_BUDGETS,
    )
    _assert_refused_before_anything(proc)
    assert "nixl_ep" in proc.stderr and "undefined symbol" in proc.stderr
    assert "0.9.0" in proc.stderr  # the fix is named


def test_import_gate_probe_text_matches_what_the_connector_imports() -> None:
    text = PD_SH.read_text(encoding="utf-8")
    assert "import nixl._api, nixl._bindings" in text
    assert 'find_spec("nixl_ep")' in text
    # S0F-22 Batch 1: the connector records transfer stats (the Batch 2 proof)
    # only when nixl_agent_config imports (nixl_connector.py:131-138 at
    # v0.19.1: "NIXL agent config is not available" means no telemetry), so
    # the probe imports it too, in the same refusal lane.
    assert "from nixl._api import nixl_agent_config" in text


def test_launcher_readiness_probes_fail_on_http_errors() -> None:
    # S0F-22 Batch 1 (integration audit distributed-9): `curl -s` without
    # `-f` exits 0 on a 503, so the proxy's fail-closed /health (503 while a
    # role is down) read as READY; the two role probes get the same flag.
    code = "\n".join(
        l for l in PD_SH.read_text(encoding="utf-8").splitlines()
        if not l.lstrip().startswith("#")
    )
    probes = re.findall(r'curl (-\w+) "http://localhost:\$\{[A-Za-z_]+\}/health"', code)
    assert len(probes) == 3, probes
    assert all(p == "-sf" for p in probes), probes
    assert re.search(r'curl -s "http://localhost', code) is None


def test_side_channel_comment_names_the_upstream_ports() -> None:
    # Review addition A14 (2026-10-01): vLLM's own harness uses 5559+i for
    # the prefill side channel and 5659+i*TP for the decode
    # (run_accuracy_test.sh:152, 205 at v0.19.1), not 5600 and 5601.
    text = PD_SH.read_text(encoding="utf-8")
    assert "harness uses 5600 and 5601" not in text
    assert "5559" in text and "5659" in text


def test_stop_is_never_gated_by_the_import_gate(stub_bin: Path) -> None:
    proc = _run_pd(stub_bin, "stop", CAGE_TEST_PD_PY_RC="2")
    assert proc.returncode == 0, proc.stderr
    assert "Stopping" in proc.stdout and "nixl import gate" not in proc.stdout


def test_dead_role_fails_the_start_at_once_with_its_log_tail(stub_bin: Path, tmp_path: Path) -> None:
    # The stubbed vllm exits at once; with a 20 s budget the old per-role wait
    # would have burned all of it (S0: 11 min per attempt).
    import time
    t0 = time.monotonic()
    proc = _run_pd(
        stub_bin, "start", "fake/test-model",
        VLLM_START_TIMEOUT="20", CAGE_TEST_VLLM_JOURNAL=str(tmp_path / "j.txt"), **GOOD_BUDGETS,
    )
    elapsed = time.monotonic() - t0
    assert proc.returncode != 0
    assert "exited before both roles were ready" in proc.stdout
    assert "last 20 log lines" in proc.stdout
    assert "not ready within" not in proc.stdout
    assert elapsed < 12, f"liveness check did not short-circuit the wait ({elapsed:.1f}s)"


def test_both_roles_are_launched_before_one_joint_wait(stub_bin: Path, tmp_path: Path) -> None:
    journal = tmp_path / "vllm_journal.txt"
    proc = _run_pd(
        stub_bin, "start", "fake/test-model",
        CAGE_TEST_VLLM_JOURNAL=str(journal), **GOOD_BUDGETS,
    )
    out = proc.stdout
    assert len(_read_journal(journal, 2)) == 2
    wait_line = "Waiting for prefill (port 8100) and decode (port 8200)"
    assert wait_line in out
    # both argv echoes precede the single wait line
    assert out.index("Server args [prefill]") < out.index(wait_line)
    assert out.index("Server args [decode]") < out.index(wait_line)
    assert out.count("Waiting for") == 1  # one joint loop, no per-role serial waits


# --- review 2026-10-01 (Fable 5.1, batch 2): LOW 2, LOW 3, LOW 5 -------------


@pytest.mark.parametrize("env", [
    {"CAGE_PD_NIXL_PORT_PREFILL": "70000"},
    {"CAGE_PD_NIXL_PORT_DECODE": "65536"},
], ids=["prefill-70000", "decode-65536"])
def test_start_refuses_side_channel_port_above_65535(stub_bin: Path, env: Dict[str, str]) -> None:
    # LOW 3: a positive int is not yet a TCP port; without the bound the start
    # composed both argv lines and ran the self-cleaning stop on a healthy stack.
    proc = _run_pd(stub_bin, "start", "fake/test-model", **GOOD_BUDGETS, **env)
    _assert_refused_before_anything(proc)
    assert "65535" in proc.stderr
    assert "65535" not in _run_pd(
        stub_bin, "start", "fake/test-model", CAGE_PD_NIXL_PORT_DECODE="65535", **GOOD_BUDGETS
    ).stderr  # the bound itself is a legal port


def test_import_gate_refuses_an_interpreter_without_vllm(stub_bin: Path) -> None:
    # LOW 2: the probe runs under CAGE_PD_PYTHON (default python3); an
    # interpreter that cannot even find vllm is not the serving interpreter,
    # and the refusal names that cause instead of a misleading nixl message.
    proc = _run_pd(
        stub_bin, "start", "fake/test-model",
        CAGE_TEST_PD_PY_RC="4", CAGE_TEST_PD_PY_MSG="vllm: not findable", **GOOD_BUDGETS,
    )
    _assert_refused_before_anything(proc)
    assert "not the interpreter that runs" in proc.stderr
    assert "cage-env" in proc.stderr and "CAGE_PD_PYTHON" in proc.stderr
    assert "nixl==" not in proc.stderr.split("REFUSING", 1)[1].splitlines()[0]
    text = PD_SH.read_text(encoding="utf-8")
    assert 'find_spec("vllm")' in text


def test_a_role_that_dies_after_reporting_ready_fails_the_start_at_once(
    stub_bin: Path, tmp_path: Path
) -> None:
    # LOW 5: liveness must hold until BOTH roles are ready, not only until a
    # role first answers /health. Here the prefill answers at once and dies
    # about 1 s later while the decode is still starting (alive, not ready);
    # the start must fail naming the prefill well inside the budget.
    import time
    local_bin = tmp_path / "bin"
    local_bin.mkdir()
    for p in stub_bin.iterdir():
        (local_bin / p.name).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
        (local_bin / p.name).chmod(0o755)
    (local_bin / "curl").write_text(
        '#!/bin/sh\ncase "$*" in *:8100/health*) exit 0 ;; *) exit 1 ;; esac\n', encoding="utf-8")
    (local_bin / "vllm").write_text(
        '#!/bin/sh\ncase "$*" in *"--port 8100"*) sleep 1 ;; *) sleep 6 ;; esac\nexit 0\n',
        encoding="utf-8")
    for name in ("curl", "vllm"):
        (local_bin / name).chmod(0o755)
    t0 = time.monotonic()
    proc = _run_pd(local_bin, "start", "fake/test-model", VLLM_START_TIMEOUT="20", **GOOD_BUDGETS)
    elapsed = time.monotonic() - t0
    assert proc.returncode != 0
    assert "prefill instance ready" in proc.stdout
    assert "prefill instance exited" in proc.stdout, proc.stdout
    assert "not ready within" not in proc.stdout
    assert elapsed < 12, f"the dead ready role was not caught before the budget ({elapsed:.1f}s)"


# ---------------------------------------------------------------------------
# 3. pd_proxy unit tests — in-process stub upstreams, http.client
# ---------------------------------------------------------------------------

#: The ticket vLLM 0.19.1's prefill returns when the request asked for a
#: remote decode: the eight keys of ``NixlConnector.request_finished``
#: (nixl_connector.py:989-998), ``remote_block_ids`` NESTED per KV group, and
#: NO "source" field (the engine never writes one; S0F-22). The proxy must
#: forward it verbatim, relay it to the client verbatim, and add nothing.
STUB_KV_TRANSFER_PARAMS = {
    "do_remote_prefill": True,
    "do_remote_decode": False,
    "remote_block_ids": [[1, 2, 3]],
    "remote_engine_id": "stub-prefill-engine",
    "remote_request_id": "cmpl-stub-1",
    "remote_host": "localhost",
    "remote_port": 5600,
    "tp_size": 1,
}

#: Sentinel: the prefill stub answers with NO kv_transfer_params key at all
#: (the no-ticket case the S0 proxy produced on every request).
_NO_TICKET_KEY = object()

DECODE_PAYLOAD = json.dumps(
    {"id": "decode-1", "choices": [{"text": "answer"}]}
).encode("utf-8")


class _StubUpstream:
    """A recording OpenAI-shaped upstream (prefill or decode role).

    ``running`` is the in-flight count its /metrics reports (beside a decoy
    occupancy family the proxy must NOT relay); ``metrics_text`` overrides
    the whole exposition; ``reset_status`` is what POST /reset_prefix_cache
    answers (the reset is journaled as (role, {"reset": path}));
    ``prefill_ticket`` is the ``kv_transfer_params`` value the prefill role
    returns (``_NO_TICKET_KEY`` omits the key). GET /v1/models and GET
    /version answer a role-tagged document so a relay test can tell which
    role served it.
    """

    def __init__(self, role: str, journal: List[Tuple[str, Dict[str, Any]]]):
        self.role = role
        self.journal = journal
        self.running = 0
        self.metrics_text: Optional[str] = None
        self.reset_status = 200
        self.prefill_ticket: Any = STUB_KV_TRANSFER_PARAMS
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a: Any) -> None:  # quiet
                pass

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/metrics":
                    text = outer.metrics_text
                    if text is None:
                        text = (
                            "# TYPE vllm:num_requests_running gauge\n"
                            f'vllm:num_requests_running{{model_name="m"}} {outer.running}\n'
                            "vllm:gpu_cache_usage_perc 0.5\n"
                        )
                    body = text.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; version=0.0.4")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path == "/v1/models":
                    body = json.dumps(
                        {"object": "list", "data": [{"id": "m", "served_by": outer.role}]}
                    ).encode("utf-8")
                elif self.path == "/version":
                    body = json.dumps({"version": "0.19.1", "served_by": outer.role}).encode("utf-8")
                else:
                    body = b'{"status":"ok"}'
                self.send_response(200 if self.path in ("/health", "/v1/models", "/version") else 404)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                # S0F-27: the proxy posts the reset WITH its query; the path
                # (query included) is what the journal records.
                if self.path.split("?", 1)[0] == "/reset_prefix_cache":
                    outer.journal.append((outer.role, {"reset": self.path}))
                    body = b"{}"
                    self.send_response(outer.reset_status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                outer.journal.append((outer.role, payload))
                if outer.role == "prefill":
                    doc: Dict[str, Any] = {"id": "prefill-1", "choices": [{"text": "x"}]}
                    if outer.prefill_ticket is not _NO_TICKET_KEY:
                        doc["kv_transfer_params"] = outer.prefill_ticket
                    body = json.dumps(doc).encode("utf-8")
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


def _post_full(port: int, path: str, body: Dict[str, Any]):
    """(status, response headers, body) of one POST through the proxy."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        conn.request(
            "POST", path, body=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        return resp.status, dict(resp.getheaders()), resp.read()
    finally:
        conn.close()


def _post(port: int, path: str, body: Dict[str, Any]):
    status, _headers, data = _post_full(port, path, body)
    return status, data


def _get(port: int, path: str):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


#: What the prefill leg must carry so vLLM 0.19.1 writes the ticket: the
#: request-side kv_transfer_params of vLLM's own NIXL proxy
#: (tests/v1/kv_connector/nixl_integration/toy_proxy_server.py:162-169 at
#: v0.19.1) and ignore_eos (a prefill whose single sampled token is EOS
#: finishes STOPPED and writes no ticket, nixl_connector.py:960-966).
PREFILL_REQUEST_TICKET = {
    "do_remote_decode": True,
    "do_remote_prefill": False,
    "remote_engine_id": None,
    "remote_block_ids": None,
    "remote_host": None,
    "remote_port": None,
}


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
    # prefill: generation clamped to 1 token, stream forced off, same prompt,
    # and (S0F-22) the request that makes the engine write a ticket
    assert prefill_body["max_tokens"] == 1
    assert prefill_body["stream"] is False
    assert prefill_body["prompt"] == "p"
    assert prefill_body["kv_transfer_params"] == PREFILL_REQUEST_TICKET
    assert prefill_body["ignore_eos"] is True
    # decode: the ORIGINAL request's generation budget, no prefill-only knobs
    assert decode_body["max_tokens"] == 64
    assert decode_body["prompt"] == "p"
    assert "ignore_eos" not in decode_body
    # client sees the decode response verbatim (streaming passthrough)
    assert body == DECODE_PAYLOAD


def test_proxy_prefill_leg_strips_stream_options_and_keeps_the_client_stream_for_decode(
    pd_stack,
) -> None:
    # S0F-22 F1: the runner streams every vLLM request with
    # stream_options.include_usage; vLLM 0.19.1 refuses stream_options when
    # stream is false (completion/protocol.py:409-414), so a prefill leg that
    # forced stream=false but kept stream_options got a 400 and the proxy
    # turned it into a 502 on EVERY streamed request.
    journal, _, _, port = pd_stack
    status, _ = _post(
        port, "/v1/chat/completions",
        {
            "model": "m", "messages": [{"role": "user", "content": "p"}],
            "max_completion_tokens": 32, "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    assert status == 200
    prefill_body, decode_body = journal[0][1], journal[1][1]
    assert prefill_body["stream"] is False
    assert "stream_options" not in prefill_body
    assert prefill_body["max_completion_tokens"] == 1
    # the decode leg is the client's request: streamed, with its usage chunk
    assert decode_body["stream"] is True
    assert decode_body["stream_options"] == {"include_usage": True}
    assert decode_body["max_completion_tokens"] == 32


def test_proxy_passes_kv_transfer_params_untouched_and_never_stamps_source(
    pd_stack,
) -> None:
    journal, _, _, port = pd_stack
    status, headers, _ = _post_full(
        port, "/v1/completions", {"model": "m", "prompt": "p", "max_tokens": 8}
    )
    assert status == 200
    decode_body = journal[1][1]
    # verbatim passthrough: byte-equal structure, nothing added or renamed
    assert decode_body["kv_transfer_params"] == STUB_KV_TRANSFER_PARAMS
    # S0F-22: the client sees the SAME engine ticket, verbatim, in the response
    # header the adapter already parses (openai_chat_adapter.py
    # _extract_header_kv_transfer_params); the decode response body carries
    # none (the decode's request_finished returns no params).
    relayed = json.loads(headers["x-kv-transfer-params"])
    assert relayed == STUB_KV_TRANSFER_PARAMS
    # no proxy-invented "source" anywhere: provenance belongs to the engine
    # alone; the runner's gate checks the engine's ticket SHAPE, never a stamp.
    assert "source" not in decode_body["kv_transfer_params"]
    assert "source" not in decode_body
    assert "source" not in json.dumps(decode_body)
    assert "source" not in headers["x-kv-transfer-params"]


@pytest.mark.parametrize("ticket,reason", [
    (_NO_TICKET_KEY, "absent"),
    (None, "absent"),
    ({}, "absent"),
    ({**STUB_KV_TRANSFER_PARAMS, "remote_block_ids": []}, "remote_block_ids"),
    ({**STUB_KV_TRANSFER_PARAMS, "remote_block_ids": [[]]}, "remote_block_ids"),
    ({**STUB_KV_TRANSFER_PARAMS, "remote_block_ids": None}, "remote_block_ids"),
    ({k: v for k, v in STUB_KV_TRANSFER_PARAMS.items() if k != "remote_host"}, "remote_host"),
    ({k: v for k, v in STUB_KV_TRANSFER_PARAMS.items() if k != "remote_request_id"}, "remote_request_id"),
    ({**STUB_KV_TRANSFER_PARAMS, "do_remote_prefill": False}, "do_remote_prefill"),
    ("not-a-dict", "object"),
], ids=[
    "no-key", "null", "empty", "ids-empty", "ids-empty-group", "ids-null",
    "no-host", "no-request-id", "prefill-flag-false", "not-a-dict",
])
def test_proxy_refuses_a_missing_or_malformed_ticket_before_the_decode(
    pd_stack, ticket: Any, reason: str,
) -> None:
    # S0F-22: a prefill that returns no usable ticket means the decode would
    # recompute the prompt under a pd label (the silent path S0 produced), and
    # a ticket with do_remote_prefill true but empty block ids kills the
    # decode engine (nixl_connector.py:855-856 asserts). Refuse, loudly,
    # before the decode is ever called.
    journal, prefill, _, port = pd_stack
    prefill.prefill_ticket = ticket
    status, body = _post(port, "/v1/completions", {"model": "m", "prompt": "p", "max_tokens": 8})
    assert status == 502
    doc = json.loads(body)
    assert "ticket" in doc["error"]
    assert reason in doc["reason"]
    assert [role for role, _ in journal] == ["prefill"], "no decode call after a refused ticket"


def test_proxy_relays_models_and_version_to_the_decode_role_only(pd_stack) -> None:
    # S0F-22 (integration audit distributed-1): the runner's readiness check
    # dials GET /v1/models and its engine-version capture GET /version; the
    # proxy answered 404 to both, so every campaign pd cell refused at engine
    # setup. Both are relayed from the DECODE role, the instance that answers
    # the client's generation requests.
    _, _, _, port = pd_stack
    status, body = _get(port, "/v1/models")
    assert status == 200
    doc = json.loads(body)
    assert doc["data"][0]["id"] == "m" and doc["data"][0]["served_by"] == "decode"
    status, body = _get(port, "/version")
    assert status == 200 and json.loads(body)["served_by"] == "decode"
    status, _ = _get(port, "/v1/other")
    assert status == 404


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


# --- ADR-0102 on pd (Batch 2 W2-R1): the strict reset against the proxy ---

RUN_EXPERIMENT_PY = REPO_ROOT / "scripts" / "3_run" / "run_experiment.py"
VLLM_ADAPTER_PY = REPO_ROOT / "src" / "inference" / "vllm_adapter.py"


def test_proxy_relayed_paths_mirror_the_runner_and_the_adapter() -> None:
    # The gauge the runner's quiescence probe sums, and the flush path the
    # vLLM adapter POSTs, restated from their SOURCE (no heavy imports): a
    # drift on either side fails here.
    runner = RUN_EXPERIMENT_PY.read_text(encoding="utf-8")
    m = re.search(r'COLD_START_RUNNING_GAUGE[^}]*"vllm":\s*"([^"]+)"', runner, re.S)
    assert m and m.group(1) == pd_proxy.RUNNING_GAUGE == "vllm:num_requests_running"
    adapter = VLLM_ADAPTER_PY.read_text(encoding="utf-8")
    m = re.search(r'_flush_endpoint:\s*Optional\[str\]\s*=\s*"([^"]+)"', adapter)
    assert m and m.group(1) == pd_proxy.RESET_PATH == "/reset_prefix_cache"
    assert pd_proxy.METRICS_PATH == "/metrics"
    # The summing rule is the runner's (labeled, bare, timestamp, comments).
    text = (
        "# HELP x\n"
        'vllm:num_requests_running{a="1"} 2\n'
        "vllm:num_requests_running 1 1700000000\n"
        "vllm:num_requests_running_total 99\n"
        "other 5\n"
    )
    assert pd_proxy.sum_gauge(text, "vllm:num_requests_running") == 3
    assert pd_proxy.sum_gauge("other 5\n", "vllm:num_requests_running") is None
    # A non-finite sample reads as ABSENT (503 lane), never as a count or a crash.
    for bad in ("NaN", "+Inf", "-Inf"):
        assert pd_proxy.sum_gauge(f"vllm:num_requests_running {bad}\n", "vllm:num_requests_running") is None
    # The relay's per-leg timeouts stay inside the runner's budgets even at
    # http.client's per-operation semantics (about twice the value per leg),
    # because the two roles are visited concurrently.
    assert 2 * pd_proxy._METRICS_TIMEOUT < 10.0
    assert 2 * pd_proxy._RESET_TIMEOUT < 30.0


def test_proxy_metrics_relays_only_the_running_gauge_per_role(pd_stack) -> None:
    _, prefill, decode, port = pd_stack
    prefill.running, decode.running = 3, 1
    status, body = _get(port, "/metrics")
    assert status == 200
    text = body.decode("utf-8")
    # The runner's own parser reads the STACK total (the sum over both labels).
    assert pd_proxy.sum_gauge(text, "vllm:num_requests_running") == 4
    assert 'vllm:num_requests_running{pd_role="prefill"} 3' in text
    assert 'vllm:num_requests_running{pd_role="decode"} 1' in text
    # Nothing else is relayed: the decoy occupancy family of each role never
    # appears (a sampler on the proxy finds absence, never a doubled value).
    assert "gpu_cache_usage_perc" not in text
    assert "model_name" not in text
    prefill.running = decode.running = 0
    _, body = _get(port, "/metrics")
    assert pd_proxy.sum_gauge(body.decode("utf-8"), "vllm:num_requests_running") == 0


def test_proxy_metrics_fails_closed_when_a_role_is_unreadable(pd_stack) -> None:
    _, _, decode, port = pd_stack
    # gauge absent on one role: a partial count must never pass as the whole
    decode.metrics_text = "vllm:gpu_cache_usage_perc 0.5\n"
    status, body = _get(port, "/metrics")
    assert status == 503
    doc = json.loads(body)
    assert "decode" in doc["roles"] and "absent" in doc["roles"]["decode"]
    assert "prefill" not in doc["roles"]
    # a non-finite sample on one role: absent, never a count, never a dropped
    # connection (the runner would otherwise read a socket error)
    decode.metrics_text = "vllm:num_requests_running NaN\n"
    status, body = _get(port, "/metrics")
    assert status == 503
    assert "absent" in json.loads(body)["roles"]["decode"]
    # role down
    decode.close()
    status, body = _get(port, "/metrics")
    assert status == 503
    assert "unreachable" in json.loads(body)["roles"]["decode"]


def test_proxy_reset_fans_out_to_both_roles_and_requires_both(pd_stack, monkeypatch) -> None:
    # S0F-27 (ADR-0137) changed what one reset is: a plain wake completion
    # sent straight to each role, then the reset with RESET_QUERY (the vLLM
    # mode in which a declined reset is a 5xx), and one repeat of that cycle
    # when a role declines. Before it the relay was one bare POST per role.
    # The wake order, the retry and the budget are pinned in
    # tests/test_pd_proxy_s0f27_s0f28.py.
    monkeypatch.setattr(pd_proxy, "_RESET_RETRY_PAUSE_S", 0.01)
    journal, _, decode, port = pd_stack
    honest = {"reset": f"{pd_proxy.RESET_PATH}?{pd_proxy.RESET_QUERY}"}
    status, body = _post(port, "/reset_prefix_cache", {})
    assert status == 200
    doc = json.loads(body)
    assert doc["roles"] == {"prefill": 200, "decode": 200}
    # Both roles, concurrently (arrival order in the journal is not defined).
    assert sorted(r for r, p in journal if p == honest) == ["decode", "prefill"], (
        "the reset must reach BOTH roles"
    )
    assert sorted(r for r, p in journal if p == pd_proxy.WAKE_BODY) == ["decode", "prefill"]
    assert len(journal) == 4
    # one role declines: never a partial success under a cold-start label
    decode.reset_status = 500
    status, body = _post(port, "/reset_prefix_cache", {})
    assert status == 502
    doc = json.loads(body)
    assert doc["roles"]["prefill"] == 200 and doc["roles"]["decode"] == 500
    assert "decode" in doc["failed"] and "prefill" not in doc["failed"]
    # one role down
    decode.close()
    status, body = _post(port, "/reset_prefix_cache", {})
    assert status == 502
    assert "unreachable" in json.loads(body)["failed"]["decode"]
    # the reset never touches the ticket path: every upstream call is the
    # honest reset or the plain wake body (no kv_transfer_params anywhere)
    assert all(p == honest or p == pd_proxy.WAKE_BODY for _, p in journal)


# ---------------------------------------------------------------------------
# 4. run_campaign.py — pd emission + --allow-pd gating (stub runner)
# ---------------------------------------------------------------------------

#: ADR-0155: the floor table is sized on the registered anchor shape and the
#: planner scales D per demand class; the one pd cell here is B3 (corpus-reuse).
#: ADR-0157 pool rule: each role pool must hold one request of the class cap
#: (8,192 tokens for the corpus class, ADR-0158), so the demand is the anchor
#: demand at the registered floor concurrency (c = 50: the prefill pool holds
#: 50 x 2336 / 4 = 29,200 tokens), never a round stub number.
_ANCHOR_TOKENS = rc.DEMAND_SEQ_TOKENS_2026_10_08[rc.DEMAND_ANCHOR_ARM]
_B3_TOKENS = rc.DEMAND_SEQ_TOKENS_2026_10_08["corpus-reuse"]
_ANCHOR_DEMAND = rc.demand_bytes("qwen3-14b", concurrency=50, avg_seq_tokens=_ANCHOR_TOKENS)


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
            # ADR-0155: the shape D was sized on; the plan refuses a table without it
            "avg_seq_tokens": _ANCHOR_TOKENS, "concurrency_target": 50,
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


def _calibrations_for(grid, directory: Path) -> Dict[str, Path]:
    """Batch 2 W4: one cal-v1 floor artifact (calibrate_cell.py shape) per
    server engine of the grid; the plan refuses without one per executable
    engine. Floors are provenance here, never asserted."""
    engines = set(grid.f1_engines) | set(grid.f2_engines) | set(grid.f3_engines)
    engines |= {engine for _bid, engine, _topology in grid.dist_cells}
    directory.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}
    for engine in sorted(engines - {"hf"}):
        doc = {
            "procedure_version": "cal-v2 (2026-09-30)",
            "model": rc.HF_ID_OF_SLUG[grid.model],
            "engine": engine,
            "budget_fraction": 1.5,
            "procedure": {},
            "confirmatory": False,
            "floor": {"ttft_s": 0.1, "tpot_s": 0.01, "n_requests": 30, "statistic": "median"},
            "lambda_star": {"label": "ESTIMATED", "lambda_star_qps": 2.0},
        }
        path = directory / f"calibration_{engine}.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        out[engine] = path
    return out


def _plan_for(grid, floor_path: Path, launcher_cmds=None, runner_cmd=("stub",)):
    floor = rc.load_floor_table(floor_path)
    calibrations = _calibrations_for(grid, Path(floor_path).parent / "cal")
    orig = rc.SESSION_GRIDS
    rc.SESSION_GRIDS = {grid.session: grid}
    try:
        return rc.build_plan(
            grid.session, floor, window_duration_s=60.0,
            runner_cmd=runner_cmd, launcher_cmds=launcher_cmds,
            calibrations=calibrations,
        )
    finally:
        rc.SESSION_GRIDS = orig


def _cells(plan):
    return [s for s in plan["steps"] if s["kind"] == "cell"]


def _relaunches(plan):
    return [s for s in plan["steps"] if s["kind"] == "relaunch"]


#: independent §6.5 arithmetic: D_class = floor(D_anchor x s_class / s_anchor)
#: (ADR-0155); budget = floor(r × D_class); prefill = floor(split × budget);
#: decode = the exact remainder. Restated here, never re-calling the planner
#: it checks (the registered class tokens are inputs, not planner output).
_EXPECT_TOTAL = _ANCHOR_DEMAND * _B3_TOKENS // _ANCHOR_TOKENS  # r = 1.0
_EXPECT_PREFILL = _EXPECT_TOTAL // 4  # floor(0.25 × total), exact binary fraction
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

    # -- S0F-22 Batch 1 (integration audit distributed-4): the pd CELL carries
    # the per-role telemetry pair the pd RELAUNCH already carried, so its
    # windows sample both roles (the regime pd lane) and the runner can reach
    # the decode role's /metrics for the Batch 2 transfer proof.

    def test_pd_cell_carries_the_role_telemetry_pair(self, tmp_path):
        plan = _plan_for(
            _pd_grid(f1_baselines=("B1",)), _floor_table(tmp_path)
        )
        pd = [s for s in _cells(plan) if s["cellspec"]["topology"] == "pd"]
        single = [s for s in _cells(plan) if s["cellspec"]["topology"] != "pd"]
        assert len(pd) == 1 and len(single) == 1
        assert "--vllm-telemetry" in pd[0]["argv"]
        assert pd[0]["env"]["CAGE_TELEMETRY_ENDPOINTS"] == rc.PD_TELEMETRY_ENDPOINTS
        # the same value its relaunch carries (one table, never two spellings)
        relaunch = [s for s in _relaunches(plan) if s["topology"] == "pd"][0]
        assert pd[0]["env"]["CAGE_TELEMETRY_ENDPOINTS"] == relaunch["env"]["CAGE_TELEMETRY_ENDPOINTS"]
        # a single-topology cell never carries the role pair (the role grammar
        # is the pd stack's; its sampler dials --api-base). Since S0F-26
        # (ADR-0136) it carries the flag: before, only pd cells did and every
        # other campaign window would have read UNKNOWN_TELEMETRY.
        assert "CAGE_TELEMETRY_ENDPOINTS" not in single[0]["env"]
        assert single[0]["argv"].count("--vllm-telemetry") == 1
        assert pd[0]["argv"].count("--vllm-telemetry") == 1  # once, not twice

    # -- S0F-26 (ADR-0136): the telemetry flag on every server-engine cell ----

    def _mixed_plan(self, tmp_path):
        # vllm + sglang F1 cells, one hf oracle cell, one executable pd cell
        # and one BLOCKED pd cell (sglang has no pd launcher)
        grid = _pd_grid(
            f1_baselines=("B1",), f1_engines=("vllm", "sglang"),
            hf_oracle_cells=(("B1", ("squad_v2",)),),
            dist_cells=(("B3", "vllm", "pd"), ("B3", "sglang", "pd")),
        )
        return _plan_for(grid, _floor_table(tmp_path))

    def test_every_server_engine_cell_carries_the_flag_once_and_hf_never(self, tmp_path):
        plan = self._mixed_plan(tmp_path)
        cells = _cells(plan)
        assert {s["cellspec"]["engine"] for s in cells} == {"vllm", "sglang", "hf"}
        assert any(s["blocked_on"] for s in cells), "the grid must carry a blocked cell"
        for s in cells:
            n = s["argv"].count(rc.TELEMETRY_FLAG)
            if s["cellspec"]["engine"] == "hf":
                assert n == 0, "the in-process oracle serves no /metrics"
            else:
                assert n == 1, (s["row_key"], s["blocked_on"])
        assert rc.PD_TELEMETRY_FLAG == rc.TELEMETRY_FLAG == "--vllm-telemetry"
        # the flag exists on the runner CLI the cells invoke
        assert '"--vllm-telemetry"' in RUN_EXPERIMENT_PY.read_text(encoding="utf-8")
        # and the mixed plan is what load_plan accepts
        out = tmp_path / "plan_mixed.json"
        out.write_text(json.dumps(plan), encoding="utf-8")
        assert rc.load_plan(out)["counts"]["cells"] == len(cells)

    def test_load_plan_refuses_a_server_cell_without_the_flag(self, tmp_path):
        plan = self._mixed_plan(tmp_path)
        out = tmp_path / "plan_stale.json"
        for engine in ("vllm", "sglang"):
            stale = json.loads(json.dumps(plan))
            cell = [
                s for s in stale["steps"]
                if s["kind"] == "cell" and s["cellspec"]["engine"] == engine
                and s["cellspec"]["topology"] == "single"
            ][0]
            cell["argv"] = [a for a in cell["argv"] if a != "--vllm-telemetry"]
            out.write_text(json.dumps(stale), encoding="utf-8")
            with pytest.raises(rc.RunError, match="vllm-telemetry") as exc:
                rc.load_plan(out)
            assert "S0F-26" in str(exc.value) and "UNKNOWN_TELEMETRY" in str(exc.value)

    def test_load_plan_refuses_the_flag_twice_and_on_an_hf_cell(self, tmp_path):
        plan = self._mixed_plan(tmp_path)
        out = tmp_path / "plan_stale.json"
        # twice on a server cell
        stale = json.loads(json.dumps(plan))
        cell = [s for s in stale["steps"] if s["kind"] == "cell"
                and s["cellspec"]["engine"] == "vllm"][0]
        cell["argv"].append("--vllm-telemetry")
        out.write_text(json.dumps(stale), encoding="utf-8")
        with pytest.raises(rc.RunError, match="exactly once"):
            rc.load_plan(out)
        # on the hf oracle cell: its sampler would dial the runner's default port
        stale = json.loads(json.dumps(plan))
        cell = [s for s in stale["steps"] if s["kind"] == "cell"
                and s["cellspec"]["engine"] == "hf"][0]
        cell["argv"].append("--vllm-telemetry")
        out.write_text(json.dumps(stale), encoding="utf-8")
        with pytest.raises(rc.RunError, match="hf") as exc:
            rc.load_plan(out)
        assert "vllm-telemetry" in str(exc.value) and "S0F-26" in str(exc.value)

    def test_load_plan_refuses_a_pd_cell_without_the_telemetry_pair(self, tmp_path):
        plan = _plan_for(_pd_grid(), _floor_table(tmp_path))
        pd = [s for s in plan["steps"] if s["kind"] == "cell"][0]
        out = tmp_path / "plan_pd.json"
        # (a) the flag dropped
        stale = json.loads(json.dumps(plan))
        cell = [s for s in stale["steps"] if s["kind"] == "cell"][0]
        cell["argv"] = [a for a in cell["argv"] if a != "--vllm-telemetry"]
        out.write_text(json.dumps(stale), encoding="utf-8")
        with pytest.raises(rc.RunError, match="vllm-telemetry"):
            rc.load_plan(out)
        # (b) the env dropped
        stale = json.loads(json.dumps(plan))
        cell = [s for s in stale["steps"] if s["kind"] == "cell"][0]
        del cell["env"]["CAGE_TELEMETRY_ENDPOINTS"]
        out.write_text(json.dumps(stale), encoding="utf-8")
        with pytest.raises(rc.RunError, match="CAGE_TELEMETRY_ENDPOINTS"):
            rc.load_plan(out)
        # (c) the env hand-pointed elsewhere
        stale = json.loads(json.dumps(plan))
        cell = [s for s in stale["steps"] if s["kind"] == "cell"][0]
        cell["env"]["CAGE_TELEMETRY_ENDPOINTS"] = "decode=http://localhost:9999"
        out.write_text(json.dumps(stale), encoding="utf-8")
        with pytest.raises(rc.RunError, match="CAGE_TELEMETRY_ENDPOINTS"):
            rc.load_plan(out)
        assert pd["env"]["CAGE_TELEMETRY_ENDPOINTS"] == rc.PD_TELEMETRY_ENDPOINTS

    def test_load_plan_refuses_the_role_pair_on_a_single_topology_cell(self, tmp_path):
        plan = _plan_for(_pd_grid(f1_baselines=("B1",)), _floor_table(tmp_path))
        stale = json.loads(json.dumps(plan))
        single = [
            s for s in stale["steps"]
            if s["kind"] == "cell" and s["cellspec"]["topology"] != "pd"
        ][0]
        single["env"]["CAGE_TELEMETRY_ENDPOINTS"] = rc.PD_TELEMETRY_ENDPOINTS
        out = tmp_path / "plan_single.json"
        out.write_text(json.dumps(stale), encoding="utf-8")
        with pytest.raises(rc.RunError, match="CAGE_TELEMETRY_ENDPOINTS"):
            rc.load_plan(out)


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
        # V2 (2026-10-05): the pd stack is stopped once before the first step
        # (clean room) and once when the run ends; between them the pd
        # relaunch and the cell (tests/test_stage1_v2_v3_driver.py pins the
        # stop rule).
        assert [c["argv"][0] for c in calls] == ["stop", "start", "--baseline", "stop"]
        relaunch, cell = calls[1], calls[2]
        # the pd launcher verb is `start` (self-cleaning; no restart verb)
        assert relaunch["argv"][0] == "start"
        assert relaunch["env"]["CAGE_KV_BUDGET_BYTES_PREFILL"] == str(_EXPECT_PREFILL)
        assert relaunch["env"]["CAGE_KV_BUDGET_BYTES_DECODE"] == str(_EXPECT_DECODE)
        assert relaunch["env"]["CAGE_TELEMETRY_ENDPOINTS"].startswith("prefill=")
        # identity seam: the cell subprocess carries the pd topology axis
        assert cell["env"]["CAGE_CELL_TOPOLOGY"] == "pd"
        assert cell["env"]["CAGE_CELL_FAMILY"] == "DIST"
        assert cell["argv"][-2:] == ["--campaign-root", str(root)]
        # S0F-22 Batch 1: the cell subprocess carries the role telemetry pair
        assert cell["env"]["CAGE_TELEMETRY_ENDPOINTS"] == rc.PD_TELEMETRY_ENDPOINTS
        assert "--vllm-telemetry" in cell["argv"]

    def test_run_refuses_a_shell_telemetry_endpoints_export(self, tmp_path, stub, monkeypatch):
        # The runner refuses CAGE_TELEMETRY_ENDPOINTS without --vllm-telemetry
        # and would otherwise sample whatever the shell named: a plan fact the
        # step env owns, refused on presence like the other cell pins.
        plan = _stub_pd_plan(tmp_path, stub)
        monkeypatch.setenv("CAGE_TELEMETRY_ENDPOINTS", "decode=http://localhost:9999")
        with pytest.raises(rc.RunError, match="CAGE_TELEMETRY_ENDPOINTS"):
            rc.run_plan(plan, _run_root(tmp_path), allow_pd=True)
        assert stub.calls() == []
        assert "CAGE_TELEMETRY_ENDPOINTS" in rc.CELL_PIN_ENVS

    def test_cli_help_names_the_preflight_gate(self, capsys):
        with pytest.raises(SystemExit):
            rc.main(["run", "--help"])
        # argparse wraps long help at hyphens ("Run-C-\nprime") — unwrap the
        # hyphen line-breaks before asserting the gate name survives intact.
        out = re.sub(r"-\n\s*", "-", capsys.readouterr().out)
        assert "--allow-pd" in out
        assert "Run-C-prime" in out
