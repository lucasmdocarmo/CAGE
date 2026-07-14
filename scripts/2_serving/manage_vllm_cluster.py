#!/usr/bin/env python3
"""
Manage a local multi-replica vLLM cluster plus the CAGE router.

This script is intentionally separate from manage_vllm_server.sh because the
distributed baseline needs to treat the replica set as one unit: start N
isolated vLLM backends, validate them, then start the router pointed at those
distinct endpoints.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


PROJECT_DIR = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_DIR / "logs" / "cluster"
STATE_FILE = LOG_DIR / "cluster_state.json"


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
) -> bool:
    return (
        state.get("model") == model
        and int(state.get("replica_count") or 0) == replica_count
        and int(state.get("base_port") or 0) == base_port
        and int(state.get("router_port") or 0) == router_port
        and state.get("router_strategy") == router_strategy
    )


def print_cluster_status(state: Dict[str, Any]) -> int:
    healthy, detail = validate_cluster_state(state, model=state.get("model", ""))
    print(f"Cluster status: {'healthy' if healthy else 'degraded'}")
    print(f"  model: {state.get('model')}")
    print(f"  replicas: {state.get('replica_count')}")
    print(f"  base_port: {state.get('base_port')}")
    print(f"  router_port: {state.get('router_port')}")
    print(f"  strategy: {state.get('router_strategy')}")
    for replica in state.get("replicas") or []:
        pid = replica.get("pid")
        print(
            f"  - {replica.get('replica_id')}: {replica.get('api_base')} "
            f"(pid={pid}, running={is_pid_running(pid)})"
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
) -> int:
    replicas = build_replica_configs(replica_count, base_port)
    router_url = f"http://localhost:{router_port}"

    existing_state = load_state()
    if existing_state and state_matches_requested_config(
        existing_state,
        model=model,
        replica_count=replica_count,
        base_port=base_port,
        router_port=router_port,
        router_strategy=router_strategy,
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
            pid = launch_process(
                [
                    "vllm",
                    "serve",
                    model,
                    "--port",
                    str(replica["port"]),
                    "--enable-prefix-caching",
                    "--enable-prompt-tokens-details",
                    "--max-model-len",
                    "2048",
                ],
                log_path,
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
