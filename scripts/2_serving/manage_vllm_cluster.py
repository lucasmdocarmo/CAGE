#!/usr/bin/env python3
"""
Order:     stage 2 — after 1_setup, before the distributed 3_run arm
Objective: Start/validate/stop a local multi-replica vLLM cluster plus the CAGE router as one unit
Cloud:     both

Manage a local multi-replica vLLM cluster plus the CAGE router.

This script is intentionally separate from manage_vllm_server.sh because the
distributed baseline needs to treat the replica set as one unit: start N
isolated vLLM backends, validate them, then start the router pointed at those
distinct endpoints.

[VERIFY-LIVE at S0] (K-COV6, task #141): 0% offline coverage and no prior
marker — the start/validate/stop lifecycle, replica health-gating, and the
state-file contract execute nowhere in the offline suite. S0 checklist row
S0-9 (MyDocs/S0_CHECKLIST.md) forces the live proof (start -> status ->
routed traffic -> stop with no orphan replica) before any DIST-topology cell.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import requests


PROJECT_DIR = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_DIR / "logs" / "cluster"
STATE_FILE = LOG_DIR / "cluster_state.json"

# S0F-18 (live 2026-09-30): the RunPod image runpod/pytorch:1.0.2-cu1281-torch280-
# ubuntu2404 runs its own nginx on these ports (8001 forwards to 8000 and its 502
# page answers HTTP 200), so the old default --base-port 8001 could never bind.
IMAGE_NGINX_PORTS: Tuple[int, ...] = (3001, 7270, 7861, 8001, 8081, 9091)
DEFAULT_BASE_PORT: int = 8101   # proven live at S0 run 6; replicas take base..base+N-1
DEFAULT_ROUTER_PORT: int = 9000

# S0F-18: the readiness budgets read the same env knobs the engine launchers read
# (manage_vllm_server.sh VLLM_START_TIMEOUT), so one exported value covers every
# launcher on a slow box; an explicit flag still wins. Empty means unset.
REPLICA_TIMEOUT_ENV = "VLLM_START_TIMEOUT"
REPLICA_TIMEOUT_DEFAULT = 300
ROUTER_TIMEOUT_ENV = "ROUTER_START_TIMEOUT"
ROUTER_TIMEOUT_DEFAULT = 60


def env_timeout_default(name: str, fallback: int, env: Mapping[str, str]) -> int:
    """The readiness budget for start/restart: ``env[name]`` when set and a
    non-negative integer (0 = check once, then fail, the bash launchers' and the
    test suite's idiom), ``fallback`` when unset or blank; anything else raises
    ValueError (main turns it into exit 2 before any work). Resolved only on
    start/restart (review 2026-09-30, MEDIUM 2): stop and status, which the
    cleanup traps call, never read this knob."""
    raw = (env.get(name) or "").strip()
    if not raw:
        return fallback
    if not raw.isdigit():
        raise ValueError(f"{name}={raw!r} must be a non-negative integer number of seconds")
    return int(raw)


# =============================================================================
# Shared-GPU memory-utilization rule  (backlog Tier A item A1; S0-9 / S0-20)
# =============================================================================

SHARED_GPU_MEM_UTIL: float = 0.45
"""Per-instance --gpu-memory-utilization DEFAULT when two or more vLLM
instances share ONE GPU (backlog Tier A item A1; S0 checklist rows S0-9 and
S0-20).

vLLM 0.19.1's startup check requests the fraction of the device
unconditionally, so a second instance launched at the 0.90 uniform operating
point on a shared GPU is refused at startup and the S0-9 cluster proof (start,
status, routed traffic, stop) cannot begin. Two instances at 0.45 each fit
under the device with headroom for the CUDA contexts. This value is the
DEFAULT only: VLLM_GPU_MEMORY_UTILIZATION still overrides it, subject to
SHARED_GPU_MEM_UTIL_CEILING.
"""

SHARED_GPU_MEM_UTIL_CEILING: float = 0.50
"""Largest explicit VLLM_GPU_MEMORY_UTILIZATION accepted per instance on a
shared GPU (backlog A1). Above it the second instance cannot pass vLLM's
startup check, so the launcher REFUSES rather than launching a cluster whose
second replica dies after the first one has already claimed the device.
"""

DISTINCT_GPU_MEM_UTIL: float = 0.90
"""Per-instance default when every instance is pinned to its own GPU set (or
there is a single instance): the Option-A uniform operating point, mirroring
scripts/lib/_serving_config.sh (VLLM_GPU_MEMORY_UTILIZATION default).
"""

MEM_UTIL_DEFAULTED_ENV = "CAGE_VLLM_MEM_UTIL_DEFAULTED"
"""S0F-21 (ADR-0141): scripts/lib/_serving_config.sh exports this marker as "1"
exactly when it set VLLM_GPU_MEMORY_UTILIZATION to its 0.90 default itself.
The pilot entry points source that library before this manager runs
(cloud_run.sh, then run_baselines.sh), so without the marker the shared-GPU
rule read the library's default as an explicit 0.90 and refused the pilot's
three-replica family on one GPU (found 2026-09-30 by the S0F-19 verifier).
resolve_gpu_share treats the pair marker "1" plus the value spelled exactly
as the library spells it ("0.90") as unset; any other value is the operator's.
"""


class GpuShareError(ValueError):
    """Typed refusal for the shared-GPU rule: malformed or overlapping replica
    GPU pins, or a per-instance memory fraction the shared device cannot
    honor. Raised BEFORE any process is stopped or started."""


@dataclass(frozen=True)
class GpuShareDecision:
    """The resolved shared-vs-distinct regime for one cluster launch.

    mode:        "shared" (some instance has no distinct pin) or "distinct".
    mem_util:    the --gpu-memory-utilization string handed to EVERY instance.
    source:      "default" (SHARED_GPU_MEM_UTIL / DISTINCT_GPU_MEM_UTIL),
                 "default (...; S0F-21)" when the env carried the serving
                 library's own marked default, which counts as unset, or
                 "explicit" (VLLM_GPU_MEMORY_UTILIZATION honored).
    replica_gpus: per-replica CUDA_VISIBLE_DEVICES value, None = unpinned.
    """

    mode: str
    mem_util: str
    source: str
    replica_gpus: Tuple[Optional[str], ...]

    def banner(self) -> str:
        pins = ",".join("unpinned" if g is None else g.replace(",", "+") for g in self.replica_gpus)
        rule = "SHARED_GPU_MEM_UTIL" if self.mode == "shared" else "DISTINCT_GPU_MEM_UTIL"
        return (
            f"[cage] gpu-share decision: {self.mode} (replica pins: {pins}) -> "
            f"--gpu-memory-utilization {self.mem_util} per instance "
            f"[{self.source}; rule {rule}, backlog A1 / S0-9 / S0-20]"
        )

    def as_state(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "mem_util": self.mem_util,
            "source": self.source,
            "replica_gpus": list(self.replica_gpus),
        }


def parse_replica_gpus(spec: Optional[str], replica_count: int) -> Tuple[Optional[str], ...]:
    """Parse --replica-gpus into one CUDA_VISIBLE_DEVICES value per replica.

    Grammar: comma-separated entries, one per replica; an entry that spans
    several GPUs (tensor parallel) joins them with '+' ("0+1,2+3"). Entries
    must be numeric and pairwise disjoint. None or blank means no pins at all
    (every replica unpinned, ambient visibility). Anything else is a
    GpuShareError: a partially or ambiguously pinned cluster must never be
    labeled "distinct".
    """
    if spec is None or not spec.strip():
        return tuple(None for _ in range(replica_count))
    entries = spec.split(",")
    if len(entries) != replica_count:
        raise GpuShareError(
            f"--replica-gpus {spec!r} names {len(entries)} replica pin(s) but "
            f"--replicas is {replica_count}; give exactly one entry per replica "
            f"(join a replica's several GPUs with '+', e.g. 0+1,2+3)"
        )
    seen: set = set()
    pins: List[str] = []
    for entry in entries:
        devices = entry.split("+")
        if any(not d.isdigit() for d in devices):
            raise GpuShareError(
                f"--replica-gpus {spec!r}: entry {entry!r} is not a '+'-joined "
                f"list of GPU indices"
            )
        for d in devices:
            if len(d) > 1 and d.startswith("0"):
                raise GpuShareError(
                    f"--replica-gpus {spec!r}: GPU index {d!r} has a leading zero; "
                    f"write the plain index so disjointness is unambiguous"
                )
            if d in seen:
                raise GpuShareError(
                    f"--replica-gpus {spec!r}: GPU {d} appears in more than one "
                    f"replica pin; pins must be pairwise disjoint or the "
                    f"replicas share a device"
                )
            seen.add(d)
        pins.append(",".join(devices))
    return tuple(pins)


def _parse_mem_util(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise GpuShareError(
            f"VLLM_GPU_MEMORY_UTILIZATION={raw!r} is not a decimal fraction"
        ) from exc
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise GpuShareError(
            f"VLLM_GPU_MEMORY_UTILIZATION={raw!r} must be a fraction in (0, 1]"
        )
    return value


def resolve_gpu_share(
    *,
    replica_count: int,
    replica_gpus: Tuple[Optional[str], ...],
    env: Mapping[str, str],
) -> GpuShareDecision:
    """Apply the shared-GPU rule (backlog A1; S0-9 / S0-20).

    shared   = replica_count > 1 and at least one replica has no pin.
    distinct = one replica, or every replica pinned (parse_replica_gpus has
               already proven the pins pairwise disjoint).
    Default: SHARED_GPU_MEM_UTIL when shared, DISTINCT_GPU_MEM_UTIL otherwise.
    Override: VLLM_GPU_MEMORY_UTILIZATION (non-empty) is honored, except that
    a shared value above SHARED_GPU_MEM_UTIL_CEILING is refused with the fix.
    S0F-21 (ADR-0141): the serving library's own marked default (the marker
    MEM_UTIL_DEFAULTED_ENV == "1" beside the value "0.90" as the library
    spells it) counts as unset on both paths, and the source says so.
    """
    shared = replica_count > 1 and any(g is None for g in replica_gpus)
    mode = "shared" if shared else "distinct"
    # Empty is unset (the bash launcher cannot tell the two apart either).
    raw = (env.get("VLLM_GPU_MEMORY_UTILIZATION") or "").strip()
    lib_defaulted = (
        (env.get(MEM_UTIL_DEFAULTED_ENV) or "").strip() == "1"
        and raw == f"{DISTINCT_GPU_MEM_UTIL:.2f}"
    )
    if not raw or lib_defaulted:
        default = SHARED_GPU_MEM_UTIL if shared else DISTINCT_GPU_MEM_UTIL
        # The marker survives the shell: an explicit `export ...=0.90` typed
        # AFTER the library was sourced in the same shell reads as the
        # library default too (review 2026-10-06, MEDIUM); the source string
        # names the way out, and the state file records it.
        source = (
            f"default (the serving library's marked {raw} default counts as "
            f"unset; unset {MEM_UTIL_DEFAULTED_ENV} to force an explicit "
            f"{raw}; S0F-21)"
            if lib_defaulted else "default"
        )
        return GpuShareDecision(mode, f"{default:.2f}", source, tuple(replica_gpus))
    value = _parse_mem_util(raw)
    if shared and value > SHARED_GPU_MEM_UTIL_CEILING:
        raise GpuShareError(
            f"REFUSING cluster launch: {replica_count} replicas share one GPU and "
            f"VLLM_GPU_MEMORY_UTILIZATION={raw} asks each instance for "
            f"--gpu-memory-utilization {raw} of the device; the vLLM startup check "
            f"requests that fraction unconditionally, so the second replica cannot "
            f"start. Fix: lower VLLM_GPU_MEMORY_UTILIZATION to at most "
            f"{SHARED_GPU_MEM_UTIL_CEILING:.2f} (default {SHARED_GPU_MEM_UTIL:.2f}), "
            f"or pin the replicas to distinct GPUs with --replica-gpus (e.g. 0,1,2)."
        )
    return GpuShareDecision(mode, raw, "explicit", tuple(replica_gpus))


# =============================================================================
# Per-replica KV pool pin  (ADR-0124; S0F-19, live H100 2026-09-30)
# =============================================================================

KV_BUDGET_ENV = "CAGE_KV_BUDGET_BYTES_REPLICA"
"""Byte budget of EVERY replica's KV pool, passed as --kv-cache-memory-bytes.

Why a pin and not the fraction: vLLM 0.19.1's memory profiler measures the whole
device, so two replicas profiling at the same time on one GPU each book the
other's allocations as their own overhead and the pool collapses (S0: 1.31 GiB
instead of about 18 GiB at 0.45, both replicas dead at the KV check). With the
byte pin vLLM skips the profiler (gpu_worker.determine_available_memory); the S0
pd pair proved it live: two instances at 0.45, 5,713,920,000 B each, started in
the same second, identical 38,736-token pools. The pd launcher's per-role
contract (CAGE_KV_BUDGET_BYTES_PREFILL / _DECODE) is the model; the planner has
no replica topology, so the operator supplies one value for every replica and
the cluster state records it. REQUIRED in shared mode; optional in distinct mode.
"""

SINGLE_INSTANCE_BUDGET_ENVS: Tuple[str, ...] = ("CAGE_KV_BUDGET_BYTES", "CAGE_VLLM_GPU_BLOCKS_OVERRIDE")
"""The single-launcher knobs; set beside the replica pin they are a refusal (which
pool would they cap?), never a precedence rule, like manage_vllm_pd.sh."""


class KvBudgetError(ValueError):
    """Typed refusal for the replica KV pin: missing in shared mode, malformed, or
    set beside a single-instance budget knob. Raised BEFORE any process is
    stopped or started."""


def resolve_kv_budget(*, mode: str, env: Mapping[str, str]) -> Optional[int]:
    """Apply the ADR-0124 pin rule for one cluster launch.

    Returns the byte budget (int) or None (distinct mode, no pin: the legacy
    argv). Shared mode without the pin refuses, naming the knob and the fix.
    """
    clash = [name for name in SINGLE_INSTANCE_BUDGET_ENVS if (env.get(name) or "").strip()]
    if clash:
        raise KvBudgetError(
            f"REFUSING cluster launch: single-instance budget env {', '.join(clash)} is "
            f"set beside the cluster's per-replica pin {KV_BUDGET_ENV}; which pool "
            f"would it cap? Unset it for cluster launches (the same rule as the pd "
            f"launcher's per-role budgets)."
        )
    raw = (env.get(KV_BUDGET_ENV) or "").strip()
    if not raw:
        if mode == "shared":
            raise KvBudgetError(
                f"REFUSING cluster launch: the replicas share one GPU and "
                f"{KV_BUDGET_ENV} is unset. vLLM profiles the whole device, so "
                f"concurrent replicas see each other's memory as their own overhead "
                f"and their KV pools collapse (S0F-19, live 2026-09-30). Set "
                f"{KV_BUDGET_ENV}=<bytes per replica>; it becomes "
                f"--kv-cache-memory-bytes on every replica and vLLM skips the profiler, "
                f"the contract the pd launcher already uses per role (ADR-0124)."
            )
        return None
    if not raw.isdigit() or int(raw) <= 0:
        raise KvBudgetError(
            f"{KV_BUDGET_ENV}={raw!r} must be a positive integer number of bytes"
        )
    return int(raw)


def build_serve_args(
    model: str, port: int, *, gpu_memory_utilization: str, kv_budget_bytes: Optional[int] = None
) -> List[str]:
    """vLLM serve argv honoring the Option-A serving contract (lib/_serving_config.sh).

    The cluster path previously hardcoded --max-model-len 2048 with no gpu-mem-util and
    no --trust-remote-code (blocks MiMo), silently diverging from the single-node path --
    a serving-uniformity confound for any distributed-vs-single comparison (2026-07-15
    audit). Values come from the VLLM_* env exported by scripts/lib/_serving_config.sh;
    the fallbacks here mirror that file so an unsourced shell still gets Option A.

    gpu_memory_utilization is REQUIRED and comes from resolve_gpu_share (backlog
    A1): there is no inline per-instance default, because the value depends on
    whether the replicas share a GPU. kv_budget_bytes (ADR-0124) adds the per-replica
    --kv-cache-memory-bytes pin right after it; None keeps the legacy argv.
    """
    args = [
        "vllm", "serve", model,
        "--port", str(port),
        "--enable-prefix-caching",
        "--enable-prompt-tokens-details",
        "--trust-remote-code",
        "--max-model-len", os.environ.get("VLLM_MAX_MODEL_LEN", "4096"),
        "--gpu-memory-utilization", gpu_memory_utilization,
    ]
    if kv_budget_bytes is not None:
        args += ["--kv-cache-memory-bytes", str(int(kv_budget_bytes))]
    if os.environ.get("VLLM_ENFORCE_EAGER", "0") == "1":
        args.append("--enforce-eager")
    kv_dtype = (os.environ.get("VLLM_KV_CACHE_DTYPE") or "").strip()
    if kv_dtype:
        args += ["--kv-cache-dtype", kv_dtype]
    # W29 (live L40S 2026-09-27): vLLM 0.19.1 rejects --disable-log-requests; per-request
    # logging is off unless --enable-log-requests (mirrors manage_vllm_server.sh:
    # VLLM_DISABLE_LOG_REQUESTS=0 keeps the logs, anything else passes no flag).
    if os.environ.get("VLLM_DISABLE_LOG_REQUESTS", "1") == "0":
        args.append("--enable-log-requests")
    return args


def build_replica_configs(replica_count: int, base_port: int) -> List[Dict[str, Any]]:
    return [
        {
            "replica_id": f"replica-{idx}",
            "port": base_port + idx - 1,
            "api_base": f"http://localhost:{base_port + idx - 1}",
        }
        for idx in range(1, replica_count + 1)
    ]


def sanitize_name(value: str) -> str:
    return value.replace("/", "_").replace(":", "_")


def load_state() -> Optional[Dict[str, Any]]:
    if not STATE_FILE.exists():
        return None
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_state(state: Dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def remove_state() -> None:
    if STATE_FILE.exists():
        STATE_FILE.unlink()


# --- S0F-18: bound-port probe, before any process is stopped or started --------

PORT_PROBE_HOST = "0.0.0.0"   # the address vLLM and uvicorn bind


class PortInUseError(RuntimeError):
    """Typed refusal: a replica or router port already has a listener."""


def port_is_free(port: int) -> Tuple[bool, str]:
    """Bind a throwaway socket the way vLLM does NOT: SO_REUSEADDR on (a closed
    listener's TIME_WAIT never false-refuses) and SO_REUSEPORT OFF (vLLM sets it,
    and Linux lets two SO_REUSEPORT sockets of one user share a port, so a probe
    that copied vLLM's options would walk past a stale replica). Returns
    (True, "") or (False, the OSError text)."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((PORT_PROBE_HOST, port))
        return True, ""
    except OSError as exc:
        return False, str(exc)
    finally:
        probe.close()


def _psutil_sockets_on(port: int) -> List[Tuple[str, Optional[int]]]:
    """[(status, pid)] of every TCP socket whose local port is ``port``, from
    psutil; [] when psutil is absent or refuses (macOS without root raises
    AccessDenied). Never raises."""
    try:
        import psutil  # a vLLM dependency and in requirements.txt; absent locally is fine

        return [
            (str(conn.status), conn.pid)
            for conn in psutil.net_connections(kind="tcp")
            if conn.laddr and conn.laddr.port == port
        ]
    except Exception:
        return []


def port_has_live_socket(port: int) -> Optional[str]:
    """Review 2026-09-30 (MEDIUM 3): on Linux a SO_REUSEADDR bind succeeds over a
    socket that is bound but not yet listening, and vLLM binds its port before
    the engine is up and listens only afterwards. The bind probe alone would
    call such a port free and a second replica would then share it (both carry
    SO_REUSEPORT). So any socket on the port in a state other than TIME_WAIT
    (the one state a dead server legitimately leaves behind) refuses too.
    Returns a description of the first such socket, or None."""
    for status, pid in _psutil_sockets_on(port):
        if status.upper() == "TIME_WAIT":
            continue
        return f"socket in state {status} (pid {pid if pid else 'unknown'})"
    return None


def describe_port_owner(port: int) -> str:
    """Best effort: the listening process on ``port`` (psutil first, then the
    platform tool), or "unknown". Never raises."""
    try:
        import psutil  # a vLLM dependency and in requirements.txt; absent locally is fine

        for conn in psutil.net_connections(kind="tcp"):
            if conn.status == psutil.CONN_LISTEN and conn.laddr and conn.laddr.port == port:
                name = "?"
                if conn.pid:
                    try:
                        name = psutil.Process(conn.pid).name()
                    except Exception:
                        pass
                return f"pid {conn.pid} ({name})"
    except Exception:
        pass
    for tool, argv in (
        ("ss", ["ss", "-ltnp", f"sport = :{port}"]),
        ("lsof", ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"]),
    ):
        if shutil.which(tool) is None:
            continue
        try:
            out = subprocess.run(argv, capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:
            continue
        lines = [l for l in out.splitlines()[1:] if l.strip()]
        if lines:
            return f"{tool}: {lines[0].strip()}"
    return "unknown"


def refuse_bound_ports(ports: Mapping[str, int]) -> None:
    """Refuse the launch when any planned port already has a listener (bind
    probe) or any non-TIME_WAIT socket (psutil; a vLLM still loading weights is
    bound but not yet listening), or when two planned ports coincide, naming the
    port, its holder and the fix (S0F-18: the image's nginx on 8001 cost a full
    readiness budget per attempt at S0)."""
    values = [int(p) for p in ports.values()]
    if len(set(values)) != len(values):
        raise PortInUseError(
            f"REFUSING cluster launch: planned ports collide ({dict(ports)}); the replica "
            f"range base..base+N-1 and --router-port must be distinct."
        )
    for label, port in ports.items():
        free, reason = port_is_free(int(port))
        if free:
            live = port_has_live_socket(int(port))
            if live is None:
                continue
            reason = live
        raise PortInUseError(
            f"REFUSING cluster launch: {label} port {port} is already bound ({reason}); "
            f"holder: {describe_port_owner(int(port))}. Pick another --base-port / "
            f"--router-port or stop the holder. The RunPod image's nginx listens on "
            f"{', '.join(str(p) for p in IMAGE_NGINX_PORTS)} (S0F-18)."
        )


# --- S0F-18: the manager's own children, so a dead replica is noticed ---------

_CHILDREN: Dict[int, Any] = {}
"""pid -> the Popen handle launch_process created. CPython reaps a dropped child
only when the next Popen is built, so os.kill(pid, 0) reports a zombie as alive;
at S0 the manager waited 11 min 54 s on two replicas dead since 18:51:13."""


class ChildExitedError(RuntimeError):
    """A launched child exited while the manager was waiting for it."""


def child_exit_code(pid: Optional[int]) -> Optional[int]:
    proc = _CHILDREN.get(pid) if pid else None
    return None if proc is None else proc.poll()


def is_pid_running(pid: Optional[int]) -> bool:
    if not pid:
        return False
    if child_exit_code(pid) is not None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _log_tail(log_path: Optional[Path], n: int = 20) -> str:
    try:
        lines = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""
    return "\n".join(lines[-n:])


def wait_for(
    predicate,
    timeout_seconds: int,
    label: str,
    *,
    pid: Optional[int] = None,
    log_path: Optional[Path] = None,
) -> float:
    """Poll ``predicate`` every 2 s until true (returns the elapsed seconds) or the
    budget ends (RuntimeError). With ``pid`` the wait also stops on the first poll
    after that child exits (ChildExitedError with the exit code and the log tail),
    instead of idling the GPU for the whole budget."""
    t0 = time.time()
    deadline = t0 + timeout_seconds
    while True:
        if predicate():
            return time.time() - t0
        rc = child_exit_code(pid)
        if rc is not None:
            tail = _log_tail(log_path)
            raise ChildExitedError(
                f"{label} exited with exit code {rc} after {time.time() - t0:.0f}s "
                f"(pid {pid}); log: {log_path}" + (f"\n--- last lines ---\n{tail}" if tail else "")
            )
        if time.time() >= deadline:
            break
        time.sleep(2)
    raise RuntimeError(f"Timed out waiting for {label} after {timeout_seconds}s")


def get_loaded_model(api_base: str) -> Optional[str]:
    try:
        resp = requests.get(f"{api_base.rstrip('/')}/v1/models", timeout=5)
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data") or []
        if data and isinstance(data[0], dict):
            return data[0].get("id")
    except Exception:
        return None
    return None


def health_ready(api_base: str) -> bool:
    try:
        resp = requests.get(f"{api_base.rstrip('/')}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def replica_ready(api_base: str, model: str) -> bool:
    return health_ready(api_base) and get_loaded_model(api_base) == model


def fetch_router_stats(router_url: str) -> Optional[Dict[str, Any]]:
    try:
        resp = requests.get(f"{router_url.rstrip('/')}/stats", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def router_ready(router_url: str, expected_replicas: int) -> bool:
    if not health_ready(router_url):
        return False
    stats = fetch_router_stats(router_url)
    if not isinstance(stats, dict):
        return False
    if int(stats.get("num_replicas") or 0) != expected_replicas:
        return False
    return int(stats.get("distinct_api_bases") or 0) == expected_replicas


def build_router_replicas_env(replicas: List[Dict[str, Any]]) -> str:
    return ",".join(f"{cfg['replica_id']}={cfg['api_base']}" for cfg in replicas)


def launch_process(cmd: List[str], log_path: Path, env: Optional[Dict[str, str]] = None) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_DIR),
            env=env or os.environ.copy(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    _CHILDREN[proc.pid] = proc   # S0F-18: keep the handle so an exit is visible
    return proc.pid


def _group_alive(pgid: int) -> bool:
    """True while any process of the group can be signaled (killpg with signal 0)."""
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def terminate_process_group(pid: Optional[int], name: str, *, silent: bool = False) -> None:
    """SIGTERM, then SIGKILL after 15 s, the whole process group of ``pid``.

    Review 2026-09-30 (MEDIUM 1): the group is signaled even when its leader
    has already exited. A replica's `vllm serve` parent can die while its
    EngineCore child keeps the GPU (manage_vllm_server.sh documents that case),
    and an exited leader of OUR child is a zombie until reaped, so the liveness
    test here is "does the group still have a member", never the leader's pid.
    Under start_new_session=True the child is its own session leader, so its
    group id equals its pid; the group is reached through that even after the
    leader is reaped.
    """
    if not pid:
        return
    proc = _CHILDREN.get(pid)
    if proc is not None:
        proc.poll()   # reap an exited leader so a zombie does not read as alive
    try:
        pgid = os.getpgid(pid)
    except (ProcessLookupError, PermissionError):
        # launch_process starts every child as a session leader, so the group
        # id equals the pid; a reaped leader still leaves its group reachable.
        pgid = pid
    if not _group_alive(pgid):
        return
    if not silent:
        print(f"Stopping {name} (pid={pid})...")
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return

    deadline = time.time() + 15
    while time.time() < deadline:
        if proc is not None:
            proc.poll()
        if not _group_alive(pgid):
            return
        time.sleep(1)

    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        return


def validate_cluster_state(
    state: Dict[str, Any],
    *,
    model: str,
    require_distinct_replicas: bool = True,
) -> Tuple[bool, str]:
    replicas = state.get("replicas") or []
    router = state.get("router") or {}
    if not replicas or not router:
        return False, "cluster state is incomplete"

    for replica in replicas:
        pid = replica.get("pid")
        api_base = replica.get("api_base")
        replica_id = replica.get("replica_id")
        if not is_pid_running(pid):
            return False, f"{replica_id} is not running"
        if not replica_ready(api_base, model):
            return False, f"{replica_id} is not healthy with model {model}"

    router_pid = router.get("pid")
    router_url = router.get("api_base")
    if not is_pid_running(router_pid):
        return False, "router is not running"

    stats = fetch_router_stats(router_url)
    if not isinstance(stats, dict):
        return False, "router stats endpoint is unavailable"
    if int(stats.get("num_replicas") or 0) != len(replicas):
        return False, "router replica count does not match expected cluster size"

    distinct_api_bases = int(stats.get("distinct_api_bases") or 0)
    if require_distinct_replicas and distinct_api_bases != len(replicas):
        return False, "router is not configured with isolated replica endpoints"

    return True, "cluster is healthy"


def stop_cluster(*, silent: bool = False) -> int:
    state = load_state()
    if not state:
        if not silent:
            print("No managed cluster state found.")
        return 0

    terminate_process_group((state.get("router") or {}).get("pid"), "router", silent=silent)
    for replica in reversed(state.get("replicas") or []):
        terminate_process_group(replica.get("pid"), replica.get("replica_id", "replica"), silent=silent)

    remove_state()
    if not silent:
        print("Cluster stopped.")
    return 0


def state_matches_requested_config(
    state: Dict[str, Any],
    *,
    model: str,
    replica_count: int,
    base_port: int,
    router_port: int,
    router_strategy: str,
    replica_gpus: Optional[str],
    kv_budget_bytes: Optional[int] = None,
) -> bool:
    # The pin spec is a dial: a running cluster under other pins (or none) is
    # never reused for a launch that asked for these (backlog A1). The KV byte
    # pin is a dial too (ADR-0124; review 2026-09-30, LOW 5).
    return (
        state.get("model") == model
        and int(state.get("replica_count") or 0) == replica_count
        and int(state.get("base_port") or 0) == base_port
        and int(state.get("router_port") or 0) == router_port
        and state.get("router_strategy") == router_strategy
        and (state.get("replica_gpus") or None) == (replica_gpus or None)
        and (state.get("kv_budget_bytes") or None) == (kv_budget_bytes or None)
    )


def print_cluster_status(state: Dict[str, Any]) -> int:
    healthy, detail = validate_cluster_state(state, model=state.get("model", ""))
    print(f"Cluster status: {'healthy' if healthy else 'degraded'}")
    print(f"  model: {state.get('model')}")
    print(f"  replicas: {state.get('replica_count')}")
    print(f"  base_port: {state.get('base_port')}")
    print(f"  router_port: {state.get('router_port')}")
    print(f"  strategy: {state.get('router_strategy')}")
    gpu_share = state.get("gpu_share") or {}
    print(
        f"  gpu_share: {gpu_share.get('mode')} "
        f"(--gpu-memory-utilization {gpu_share.get('mem_util')} per instance, "
        f"{gpu_share.get('source')}; pins: {state.get('replica_gpus') or 'none'})"
    )
    for replica in state.get("replicas") or []:
        pid = replica.get("pid")
        print(
            f"  - {replica.get('replica_id')}: {replica.get('api_base')} "
            f"(pid={pid}, running={is_pid_running(pid)}, "
            f"CUDA_VISIBLE_DEVICES={replica.get('cuda_visible_devices') or 'unpinned'})"
        )
    router = state.get("router") or {}
    router_pid = router.get("pid")
    print(
        f"  - router: {router.get('api_base')} "
        f"(pid={router_pid}, running={is_pid_running(router_pid)})"
    )
    print(f"  validation: {detail}")
    return 0 if healthy else 1


def start_cluster(
    *,
    model: str,
    replica_count: int,
    base_port: int,
    router_port: int,
    router_strategy: str,
    replica_timeout: int,
    router_timeout: int,
    replica_gpus: Optional[str],
) -> int:
    # Backlog A1 (S0-9 / S0-20): the shared-vs-distinct decision is resolved
    # FIRST so a refusal fires before any process is stopped or started.
    pins = parse_replica_gpus(replica_gpus, replica_count)
    decision = resolve_gpu_share(
        replica_count=replica_count, replica_gpus=pins, env=os.environ
    )
    print(decision.banner())
    # ADR-0124 (S0F-19): the per-replica KV pin, resolved in the same
    # before-anything slot; shared mode without it is a refusal.
    kv_budget = resolve_kv_budget(mode=decision.mode, env=os.environ)
    # Spec 9 option C: on a shared GPU the replicas start one at a time, so each
    # one's init-time fraction check meets a settled device and a start failure
    # names one replica; distinct GPUs keep the concurrent launch.
    start_order = "sequential" if decision.mode == "shared" else "concurrent"
    print(
        f"[cage] kv pool pin: "
        + (f"--kv-cache-memory-bytes {kv_budget} per replica [{KV_BUDGET_ENV}; ADR-0124]"
           if kv_budget is not None else "none (distinct GPUs, vLLM profiles each device)")
        + f"; start order: {start_order}"
    )

    replicas = build_replica_configs(replica_count, base_port)
    for replica, pin in zip(replicas, pins):
        replica["cuda_visible_devices"] = pin
    router_url = f"http://localhost:{router_port}"

    existing_state = load_state()
    if existing_state and state_matches_requested_config(
        existing_state,
        model=model,
        replica_count=replica_count,
        base_port=base_port,
        router_port=router_port,
        router_strategy=router_strategy,
        replica_gpus=replica_gpus,
        kv_budget_bytes=kv_budget,
    ):
        healthy, detail = validate_cluster_state(existing_state, model=model)
        if healthy:
            print("Cluster already running with the requested configuration.")
            return print_cluster_status(existing_state)
        print(f"Existing cluster is unhealthy: {detail}. Restarting it...")
        stop_cluster(silent=True)
    elif existing_state:
        stop_cluster(silent=True)

    # S0F-18: every planned port must be free NOW (after the managed stale
    # cluster, if any, was stopped and before anything is launched).
    refuse_bound_ports(
        {**{r["replica_id"]: int(r["port"]) for r in replicas}, "router": int(router_port)}
    )

    model_slug = sanitize_name(model)
    state: Dict[str, Any] = {
        "model": model,
        "replica_count": replica_count,
        "base_port": base_port,
        "router_port": router_port,
        "router_strategy": router_strategy,
        "replica_gpus": replica_gpus or None,
        "gpu_share": decision.as_state(),
        "kv_budget_bytes": kv_budget,
        "start_order": start_order,
        "replicas": replicas,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Pre-flight: verify router dependencies are importable in the active interpreter.
    for dep in ("fastapi", "uvicorn", "aiohttp", "prometheus_client"):
        if importlib.util.find_spec(dep) is None:
            raise RuntimeError(
                f"Router dependency '{dep}' is not installed in the active Python "
                f"interpreter ({sys.executable}). Install it and retry."
            )

    def _launch_replica(replica: Dict[str, Any]) -> None:
        log_path = LOG_DIR / f"vllm_{model_slug}_{replica['replica_id']}_{replica['port']}.log"
        replica_env = os.environ.copy()
        pin = replica.get("cuda_visible_devices")
        if pin is not None:
            # The pin rides CUDA_VISIBLE_DEVICES in the child env only.
            replica_env["CUDA_VISIBLE_DEVICES"] = pin
        pid = launch_process(
            build_serve_args(
                model, replica["port"],
                gpu_memory_utilization=decision.mem_util,
                kv_budget_bytes=kv_budget,
            ),
            log_path,
            env=replica_env,
        )
        replica["pid"] = pid
        replica["log_file"] = str(log_path)
        save_state(state)   # the cleanup path reads the pids from the state file

    def _await_replica(replica: Dict[str, Any]) -> None:
        elapsed = wait_for(
            lambda api_base=replica["api_base"]: replica_ready(api_base, model),
            replica_timeout,
            f"{replica['replica_id']} on {replica['api_base']}",
            pid=replica.get("pid"),
            log_path=Path(replica["log_file"]) if replica.get("log_file") else None,
        )
        print(f"{replica['replica_id']} ready after {elapsed:.0f} s")

    try:
        if start_order == "sequential":
            for replica in replicas:
                _launch_replica(replica)
                _await_replica(replica)
        else:
            for replica in replicas:
                _launch_replica(replica)
            for replica in replicas:
                _await_replica(replica)

        router_env = os.environ.copy()
        router_env["ROUTER_REPLICAS"] = build_router_replicas_env(replicas)
        router_env["ROUTER_STRATEGY"] = router_strategy
        router_env["ROUTER_PORT"] = str(router_port)
        router_log = LOG_DIR / f"router_{router_port}.log"
        router_pid = launch_process(
            [sys.executable, "-m", "src.orchestration.router"],
            router_log,
            env=router_env,
        )
        state["router"] = {
            "pid": router_pid,
            "port": router_port,
            "api_base": router_url,
            "log_file": str(router_log),
        }
        save_state(state)

        elapsed = wait_for(
            lambda: router_ready(router_url, replica_count),
            router_timeout,
            f"router on {router_url}",
            pid=router_pid,
            log_path=router_log,
        )
        print(f"router ready after {elapsed:.0f} s")

        stats = fetch_router_stats(router_url)
        if not isinstance(stats, dict):
            raise RuntimeError("router stats did not return valid JSON")
        if int(stats.get("distinct_api_bases") or 0) != replica_count:
            raise RuntimeError("router did not expose isolated replica endpoints")

        state["last_validated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        state["last_router_stats"] = stats
        save_state(state)

        print("Cluster started successfully.")
        return print_cluster_status(state)
    except Exception:
        save_state(state)
        try:
            stop_cluster(silent=True)
        except Exception as cleanup_exc:
            print(
                f"Warning: cluster cleanup failed after startup error: {cleanup_exc}",
                file=sys.stderr,
            )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage a local multi-replica vLLM cluster for distributed CAGE baselines."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("start", "restart"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--model", required=True, help="Model name to serve on every replica.")
        sub.add_argument("--replicas", type=int, default=3, help="Number of vLLM replicas to launch.")
        sub.add_argument(
            "--base-port",
            type=int,
            default=DEFAULT_BASE_PORT,
            help=(
                f"First vLLM replica port (replicas take base..base+N-1). Default "
                f"{DEFAULT_BASE_PORT}: the RunPod image's nginx holds "
                f"{', '.join(str(p) for p in IMAGE_NGINX_PORTS)} (S0F-18); a bound port "
                f"is refused before launch, naming its holder."
            ),
        )
        sub.add_argument(
            "--router-port", type=int, default=DEFAULT_ROUTER_PORT, help="Port for the CAGE router."
        )
        sub.add_argument(
            "--router-strategy",
            default="hash",
            choices=["hash", "round_robin"],
            help="Router strategy to use for the distributed baseline.",
        )
        sub.add_argument(
            "--replica-timeout",
            type=int,
            default=None,
            help=(
                f"Maximum seconds to wait for each replica (default {REPLICA_TIMEOUT_ENV} "
                f"when set, else {REPLICA_TIMEOUT_DEFAULT}; a dead replica fails the wait at once)."
            ),
        )
        sub.add_argument(
            "--router-timeout",
            type=int,
            default=None,
            help=(
                f"Maximum seconds to wait for the router (default {ROUTER_TIMEOUT_ENV} "
                f"when set, else {ROUTER_TIMEOUT_DEFAULT})."
            ),
        )
        sub.add_argument(
            "--replica-gpus",
            default=None,
            help=(
                "Per-replica CUDA_VISIBLE_DEVICES pins, one comma-separated entry "
                "per replica, '+' joining a replica's several GPUs (e.g. 0,1,2 or "
                "0+1,2+3). Unset = every replica shares the visible GPU, so the "
                "per-instance --gpu-memory-utilization default drops to "
                f"{SHARED_GPU_MEM_UTIL:.2f} and an explicit value above "
                f"{SHARED_GPU_MEM_UTIL_CEILING:.2f} is refused (backlog A1)."
            ),
        )

    subparsers.add_parser("stop")
    subparsers.add_parser("status")
    return parser


def resolve_timeouts(args: argparse.Namespace, env: Mapping[str, str]) -> Tuple[int, int]:
    """(replica_timeout, router_timeout) for start/restart: the explicit flag,
    else the env knob, else the default. ValueError on a malformed knob."""
    replica = args.replica_timeout
    if replica is None:
        replica = env_timeout_default(REPLICA_TIMEOUT_ENV, REPLICA_TIMEOUT_DEFAULT, env)
    router = args.router_timeout
    if router is None:
        router = env_timeout_default(ROUTER_TIMEOUT_ENV, ROUTER_TIMEOUT_DEFAULT, env)
    return int(replica), int(router)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command in ("start", "restart"):
        # S0F-18: the env-backed budgets are read HERE, for start and restart
        # only; stop and status (the cleanup traps' commands) never see them.
        try:
            replica_timeout, router_timeout = resolve_timeouts(args, os.environ)
        except ValueError as exc:
            parser.error(str(exc))

    try:
        if args.command == "start":
            return start_cluster(
                model=args.model,
                replica_count=args.replicas,
                base_port=args.base_port,
                router_port=args.router_port,
                router_strategy=args.router_strategy,
                replica_timeout=replica_timeout,
                router_timeout=router_timeout,
                replica_gpus=args.replica_gpus,
            )
        if args.command == "restart":
            stop_cluster(silent=True)
            return start_cluster(
                model=args.model,
                replica_count=args.replicas,
                base_port=args.base_port,
                router_port=args.router_port,
                router_strategy=args.router_strategy,
                replica_timeout=replica_timeout,
                router_timeout=router_timeout,
                replica_gpus=args.replica_gpus,
            )
        if args.command == "stop":
            return stop_cluster()
        if args.command == "status":
            state = load_state()
            if not state:
                print("Cluster is not running.")
                return 1
            return print_cluster_status(state)
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
