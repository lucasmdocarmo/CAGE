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
import signal
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


class GpuShareError(ValueError):
    """Typed refusal for the shared-GPU rule: malformed or overlapping replica
    GPU pins, or a per-instance memory fraction the shared device cannot
    honor. Raised BEFORE any process is stopped or started."""


@dataclass(frozen=True)
class GpuShareDecision:
    """The resolved shared-vs-distinct regime for one cluster launch.

    mode:        "shared" (some instance has no distinct pin) or "distinct".
    mem_util:    the --gpu-memory-utilization string handed to EVERY instance.
    source:      "default" (SHARED_GPU_MEM_UTIL / DISTINCT_GPU_MEM_UTIL) or
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
    """
    shared = replica_count > 1 and any(g is None for g in replica_gpus)
    mode = "shared" if shared else "distinct"
    # Empty is unset (the bash launcher cannot tell the two apart either).
    raw = (env.get("VLLM_GPU_MEMORY_UTILIZATION") or "").strip()
    if not raw:
        default = SHARED_GPU_MEM_UTIL if shared else DISTINCT_GPU_MEM_UTIL
        return GpuShareDecision(mode, f"{default:.2f}", "default", tuple(replica_gpus))
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


def build_serve_args(model: str, port: int, *, gpu_memory_utilization: str) -> List[str]:
    """vLLM serve argv honoring the Option-A serving contract (lib/_serving_config.sh).

    The cluster path previously hardcoded --max-model-len 2048 with no gpu-mem-util and
    no --trust-remote-code (blocks MiMo), silently diverging from the single-node path --
    a serving-uniformity confound for any distributed-vs-single comparison (2026-07-15
    audit). Values come from the VLLM_* env exported by scripts/lib/_serving_config.sh;
    the fallbacks here mirror that file so an unsourced shell still gets Option A.

    gpu_memory_utilization is REQUIRED and comes from resolve_gpu_share (backlog
    A1): there is no inline per-instance default, because the value depends on
    whether the replicas share a GPU.
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
    if os.environ.get("VLLM_ENFORCE_EAGER", "0") == "1":
        args.append("--enforce-eager")
    kv_dtype = (os.environ.get("VLLM_KV_CACHE_DTYPE") or "").strip()
    if kv_dtype:
        args += ["--kv-cache-dtype", kv_dtype]
    if os.environ.get("VLLM_DISABLE_LOG_REQUESTS", "1") == "1":
        args.append("--disable-log-requests")
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


def is_pid_running(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def wait_for(predicate, timeout_seconds: int, label: str) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(2)
    raise RuntimeError(f"Timed out waiting for {label}")


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
    return proc.pid


def terminate_process_group(pid: Optional[int], name: str, *, silent: bool = False) -> None:
    if not pid or not is_pid_running(pid):
        return
    try:
        pgid = os.getpgid(pid)
    except (ProcessLookupError, PermissionError):
        return
    if not silent:
        print(f"Stopping {name} (pid={pid})...")
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return

    deadline = time.time() + 15
    while time.time() < deadline:
        if not is_pid_running(pid):
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
) -> bool:
    # The pin spec is a dial: a running cluster under other pins (or none) is
    # never reused for a launch that asked for these (backlog A1).
    return (
        state.get("model") == model
        and int(state.get("replica_count") or 0) == replica_count
        and int(state.get("base_port") or 0) == base_port
        and int(state.get("router_port") or 0) == router_port
        and state.get("router_strategy") == router_strategy
        and (state.get("replica_gpus") or None) == (replica_gpus or None)
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
    ):
        healthy, detail = validate_cluster_state(existing_state, model=model)
        if healthy:
            print("Cluster already running with the requested configuration.")
            return print_cluster_status(existing_state)
        print(f"Existing cluster is unhealthy: {detail}. Restarting it...")
        stop_cluster(silent=True)
    elif existing_state:
        stop_cluster(silent=True)

    model_slug = sanitize_name(model)
    state: Dict[str, Any] = {
        "model": model,
        "replica_count": replica_count,
        "base_port": base_port,
        "router_port": router_port,
        "router_strategy": router_strategy,
        "replica_gpus": replica_gpus or None,
        "gpu_share": decision.as_state(),
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

    try:
        for replica in replicas:
            log_path = LOG_DIR / f"vllm_{model_slug}_{replica['replica_id']}_{replica['port']}.log"
            replica_env = os.environ.copy()
            pin = replica.get("cuda_visible_devices")
            if pin is not None:
                # The pin rides CUDA_VISIBLE_DEVICES in the child env only.
                replica_env["CUDA_VISIBLE_DEVICES"] = pin
            pid = launch_process(
                build_serve_args(
                    model, replica["port"], gpu_memory_utilization=decision.mem_util
                ),
                log_path,
                env=replica_env,
            )
            replica["pid"] = pid
            replica["log_file"] = str(log_path)

        save_state(state)

        for replica in replicas:
            wait_for(
                lambda api_base=replica["api_base"]: replica_ready(api_base, model),
                replica_timeout,
                f"{replica['replica_id']} on {replica['api_base']}",
            )

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

        wait_for(
            lambda: router_ready(router_url, replica_count),
            router_timeout,
            f"router on {router_url}",
        )

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
        sub.add_argument("--base-port", type=int, default=8001, help="First vLLM replica port.")
        sub.add_argument("--router-port", type=int, default=9000, help="Port for the CAGE router.")
        sub.add_argument(
            "--router-strategy",
            default="hash",
            choices=["hash", "round_robin"],
            help="Router strategy to use for the distributed baseline.",
        )
        sub.add_argument(
            "--replica-timeout",
            type=int,
            default=300,
            help="Maximum time to wait for each replica to become ready.",
        )
        sub.add_argument(
            "--router-timeout",
            type=int,
            default=60,
            help="Maximum time to wait for the router to become ready.",
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


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "start":
            return start_cluster(
                model=args.model,
                replica_count=args.replicas,
                base_port=args.base_port,
                router_port=args.router_port,
                router_strategy=args.router_strategy,
                replica_timeout=args.replica_timeout,
                router_timeout=args.router_timeout,
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
                replica_timeout=args.replica_timeout,
                router_timeout=args.router_timeout,
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
