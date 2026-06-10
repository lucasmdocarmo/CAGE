#!/usr/bin/env python3
"""
CAGE Experiment Runner

Runs baseline experiments with specified model, dataset, and configuration.
"""

import argparse
import sys
import json
import csv
import hashlib
import time
import platform
import subprocess
import shlex
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import get_loader, CAGExample
from src.inference.engine import InferenceEngine, InferenceRequest
from src.inference.vllm_adapter import VLLMAdapter, VLLMOfflineAdapter
from src.inference.ollama_adapter import OllamaAdapter
from src.inference.gemini_adapter import GeminiAdapter
from src.evaluation.quality import QualityEvaluator
from src.evaluation.performance import PerformanceEvaluator, CacheMetricsTracker
from src.evaluation.code_evaluator import CodeQualityEvaluator
from src.orchestration.baselines import get_baseline_config, check_baseline_requirements
from src.orchestration.ir import (
    IRHit,
    build_corpus_from_contexts,
    ensure_ir_index,
    default_index_dir,
    retrieval_hit_rate,
    stable_text_id,
    CrossEncoderReranker,
)
from src.orchestration.redis_cache import RedisConfig, RedisClient, RetrievalCache
from src.utils.prompting import format_qa_prompt, format_multi_turn_prompt
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, start_http_server
import os
import psutil
_PROM_REGISTRY = CollectorRegistry()
_PROM_STARTED = False
_RUNNER_METRICS: Dict[str, Any] = {}


def get_runner_metrics() -> Dict[str, Any]:
    global _RUNNER_METRICS
    if _RUNNER_METRICS:
        return _RUNNER_METRICS

    _RUNNER_METRICS = {
        "REQ_COUNTER": Counter(
            "cage_runner_requests_total",
            "Total requests",
            ["baseline", "backend", "dataset"],
            registry=_PROM_REGISTRY,
        ),
        "ERR_COUNTER": Counter(
            "cage_runner_errors_total",
            "Total errors",
            ["baseline", "backend", "dataset"],
            registry=_PROM_REGISTRY,
        ),
        "TTFT_HIST": Histogram(
            "cage_runner_ttft_seconds",
            "TTFT in seconds",
            buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5),
            labelnames=["baseline", "backend", "dataset"],
            registry=_PROM_REGISTRY,
        ),
        "LAT_HIST": Histogram(
            "cage_runner_latency_seconds",
            "End-to-end latency in seconds",
            buckets=(0.1, 0.2, 0.5, 1, 2, 5, 10),
            labelnames=["baseline", "backend", "dataset"],
            registry=_PROM_REGISTRY,
        ),
        "TOK_HIST": Histogram(
            "cage_runner_tokens",
            "Tokens per response",
            buckets=(5, 10, 20, 50, 100, 200, 500, 1000),
            labelnames=["baseline", "backend", "dataset"],
            registry=_PROM_REGISTRY,
        ),
        "PROMPT_TOK_HIST": Histogram(
            "cage_runner_prompt_tokens",
            "Prompt tokens per request (from backend usage)",
            buckets=(1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096),
            labelnames=["baseline", "backend", "dataset"],
            registry=_PROM_REGISTRY,
        ),
        "CACHED_PROMPT_TOK_HIST": Histogram(
            "cage_runner_cached_prompt_tokens",
            "Cached prompt tokens per request (from backend usage)",
            buckets=(0, 1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048),
            labelnames=["baseline", "backend", "dataset"],
            registry=_PROM_REGISTRY,
        ),
        "CACHED_PROMPT_RATIO_HIST": Histogram(
            "cage_runner_cached_prompt_ratio",
            "cached_prompt_tokens / prompt_tokens (from backend usage)",
            buckets=(0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0),
            labelnames=["baseline", "backend", "dataset"],
            registry=_PROM_REGISTRY,
        ),
        "CACHED_PROMPT_REQ_COUNTER": Counter(
            "cage_runner_cached_prompt_requests_total",
            "Requests with cached_prompt_tokens > 0",
            ["baseline", "backend", "dataset"],
            registry=_PROM_REGISTRY,
        ),
        "CPU_GAUGE": Gauge(
            "cage_runner_cpu_percent",
            "Process CPU percent",
            registry=_PROM_REGISTRY,
        ),
        "RSS_GAUGE": Gauge(
            "cage_runner_rss_mb",
            "Process RSS (MB)",
            registry=_PROM_REGISTRY,
        ),
    }
    return _RUNNER_METRICS

def format_metric(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"

def _safe_run(cmd: List[str], *, timeout: int = 2, cwd: Optional[Path] = None) -> Optional[str]:
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            cwd=str(cwd) if cwd else None,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        return None
    return None


def get_git_commit() -> Optional[str]:
    repo_root = Path(__file__).resolve().parent.parent
    return _safe_run(["git", "rev-parse", "HEAD"], cwd=repo_root)


def capture_git_metadata() -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parent.parent
    try:
        repo_probe = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2,
            cwd=str(repo_root),
        )
    except Exception as e:
        repo_probe = None
        probe_error = str(e)
    else:
        probe_error = None

    if repo_probe is None or repo_probe.returncode != 0:
        error_text = probe_error
        if repo_probe is not None:
            error_text = (repo_probe.stderr or repo_probe.stdout or "").strip() or probe_error
        return {
            "git_commit": None,
            "git_branch": None,
            "git_dirty": None,
            "git_repository_present": False,
            "git_root": None,
            "git_error": error_text,
        }

    git_root = Path(repo_probe.stdout.strip())
    branch = _safe_run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=git_root)
    commit = _safe_run(["git", "rev-parse", "HEAD"], cwd=git_root)
    status = _safe_run(["git", "status", "--porcelain"], cwd=git_root)
    return {
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": bool(status),
        "git_repository_present": True,
        "git_root": str(git_root),
        "git_error": None,
    }


def capture_system_snapshot() -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 3),
    }
    try:
        freq = psutil.cpu_freq()
        if freq:
            snapshot["cpu_freq_mhz"] = freq.max
    except Exception:
        pass
    return snapshot


def capture_env_snapshot() -> Dict[str, Any]:
    prefixes = ("VLLM_", "OLLAMA_", "PROM_")
    env = {}
    for k, v in os.environ.items():
        if k.startswith(prefixes):
            env[k] = v
    return env


def _safe_get_json(url: str, *, timeout: int = 5) -> Optional[Any]:
    try:
        import requests

        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def capture_backend_metadata(
    *,
    api_base: str,
    backend: str,
    model_name: str,
    use_offline: bool,
) -> Dict[str, Any]:
    api_base = api_base.rstrip("/")
    metadata: Dict[str, Any] = {
        "backend": backend,
        "api_base": api_base,
        "requested_model": model_name,
        "mode": "offline" if use_offline else "api",
        "client_library_version": None,
        "server_version": None,
        "loaded_model": None,
        "loaded_models": None,
    }

    if backend == "vllm":
        try:
            import vllm  # type: ignore

            metadata["client_library_version"] = getattr(vllm, "__version__", None)
        except Exception:
            metadata["client_library_version"] = None

        if use_offline:
            metadata["loaded_model"] = model_name
            metadata["loaded_models"] = [model_name]
            return metadata

        version_payload = _safe_get_json(f"{api_base}/version")
        if isinstance(version_payload, dict):
            metadata["server_version"] = (
                version_payload.get("version") or version_payload.get("vllm_version")
            )

        models_payload = _safe_get_json(f"{api_base}/v1/models")
        if isinstance(models_payload, dict):
            loaded_models = [
                str(item.get("id"))
                for item in models_payload.get("data", [])
                if isinstance(item, dict) and item.get("id")
            ]
            if loaded_models:
                metadata["loaded_models"] = loaded_models
                metadata["loaded_model"] = loaded_models[0]

    elif backend == "ollama":
        if shutil.which("ollama"):
            metadata["client_library_version"] = _safe_run(["ollama", "--version"])
        version_payload = _safe_get_json(f"{api_base}/api/version")
        if isinstance(version_payload, dict):
            metadata["server_version"] = version_payload.get("version")
        tags_payload = _safe_get_json(f"{api_base}/api/tags")
        if isinstance(tags_payload, dict):
            loaded_models = [
                str(item.get("name"))
                for item in tags_payload.get("models", [])
                if isinstance(item, dict) and item.get("name")
            ]
            if loaded_models:
                metadata["loaded_models"] = loaded_models
                metadata["loaded_model"] = loaded_models[0]

    return metadata


def default_experiment_label(
    baseline: str,
    *,
    sharding_policy: str,
    warmup_queries: int,
) -> str:
    if baseline == "redis":
        return "redis_retrieval_cache_cold"
    if baseline == "hybrid":
        return "hybrid_retrieval_cache_warm" if warmup_queries > 0 else "hybrid_retrieval_cache_cold"
    if baseline == "distributed":
        if sharding_policy == "replicated":
            return "distributed_router_replicated"
        return "distributed_sharded_sim"
    return baseline


def default_dataset_split(dataset_name: str) -> str:
    if dataset_name in {"humaneval", "mbpp", "hpc_code"}:
        return "test"
    return "validation"


def is_code_dataset(dataset_name: str, examples: Optional[List[CAGExample]] = None) -> bool:
    if dataset_name in {"humaneval", "mbpp", "hpc_code"}:
        return True
    for example in examples or []:
        metadata = example.metadata or {}
        dataset_type = str(metadata.get("dataset_type") or "").lower()
        if "code" in dataset_type:
            return True
    return False


def append_command_log(output_dir: Path, argv: List[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd_str = " ".join(shlex.quote(a) for a in argv)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with (output_dir / "commands.log").open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {cmd_str}\n")


def snapshot_router_stats(output_dir: Path, stats_url: str) -> None:
    try:
        import requests

        stats = {}
        for endpoint in ("stats", "health"):
            url = stats_url.rstrip("/") + f"/{endpoint}"
            try:
                resp = requests.get(url, timeout=5)
                stats[endpoint] = {
                    "status_code": resp.status_code,
                    "body": resp.json() if "application/json" in resp.headers.get("Content-Type", "") else resp.text,
                }
            except Exception as e:
                stats[endpoint] = {"error": str(e)}
        with (output_dir / "router_stats.json").open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
    except Exception:
        pass


def fetch_router_snapshot(stats_url: str) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {}
    try:
        import requests

        for endpoint in ("stats", "health"):
            url = stats_url.rstrip("/") + f"/{endpoint}"
            try:
                resp = requests.get(url, timeout=5)
                snapshot[endpoint] = {
                    "status_code": resp.status_code,
                    "body": resp.json()
                    if "application/json" in resp.headers.get("Content-Type", "")
                    else resp.text,
                }
            except Exception as e:
                snapshot[endpoint] = {"error": str(e)}
    except Exception as e:
        snapshot["error"] = str(e)

    return snapshot


def validate_distributed_artifacts(
    results: List[Dict[str, Any]],
    *,
    api_base: str,
    sharding_policy: str,
    require_distinct_replicas: bool,
) -> Dict[str, Any]:
    routed_replicas = [str(r.get("routed_replica") or "") for r in results]
    nonempty_routed_replicas = [r for r in routed_replicas if r]
    kv_transfer_raw = [str(r.get("kv_transfer_params") or "") for r in results]
    nonempty_kv_transfer_raw = [v for v in kv_transfer_raw if v]

    if not nonempty_routed_replicas:
        raise RuntimeError("Distributed run produced no routed_replica metadata.")
    if not nonempty_kv_transfer_raw:
        raise RuntimeError("Distributed run produced no kv_transfer_params metadata.")

    transfer_required_count = 0
    positive_transfer_latency_count = 0
    positive_transfer_bytes_count = 0
    for raw in nonempty_kv_transfer_raw:
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if bool(payload.get("transfer_required")):
            transfer_required_count += 1
        if float(payload.get("transfer_latency_ms") or 0.0) > 0.0:
            positive_transfer_latency_count += 1
        if int(payload.get("transfer_bytes") or 0) > 0:
            positive_transfer_bytes_count += 1

    if sharding_policy == "replicated" and (
        positive_transfer_latency_count > 0 or positive_transfer_bytes_count > 0
    ):
        raise RuntimeError(
            "Replicated distributed run unexpectedly reported simulated transfer cost."
        )

    if sharding_policy == "sharded_context" and (
        positive_transfer_latency_count == 0 or positive_transfer_bytes_count == 0
    ):
        raise RuntimeError(
            "Sharded-context distributed run did not report positive simulated transfer cost."
        )

    router_snapshot = fetch_router_snapshot(api_base)
    stats_body = (router_snapshot.get("stats") or {}).get("body")
    distinct_api_bases = None
    topology = None
    router_replicas = []
    if isinstance(stats_body, dict):
        router_replicas = stats_body.get("replicas") or []
        distinct_api_bases = stats_body.get("distinct_api_bases")
        if isinstance(distinct_api_bases, int):
            topology = (
                "isolated_replicas"
                if distinct_api_bases >= max(2, len(router_replicas))
                else "shared_backend"
            )

    if require_distinct_replicas:
        if not isinstance(stats_body, dict):
            raise RuntimeError("Distributed run could not verify router stats.")
        if not isinstance(distinct_api_bases, int) or distinct_api_bases < 2:
            raise RuntimeError("Distributed run did not use distinct replica endpoints.")
        if router_replicas and distinct_api_bases != len(router_replicas):
            raise RuntimeError(
                "Distributed run used repeated upstream endpoints instead of isolated replicas."
            )

    return {
        "sharding_policy": sharding_policy,
        "router_snapshot": router_snapshot,
        "nonempty_routed_replica": len(nonempty_routed_replicas),
        "distinct_routed_replicas": sorted(set(nonempty_routed_replicas)),
        "nonempty_kv_transfer_params": len(nonempty_kv_transfer_raw),
        "transfer_required_count": transfer_required_count,
        "positive_transfer_latency_count": positive_transfer_latency_count,
        "positive_transfer_bytes_count": positive_transfer_bytes_count,
        "topology": topology,
        "distinct_api_bases": distinct_api_bases,
    }


def chunked(items: List[Any], size: int) -> List[List[Any]]:
    if size <= 0:
        return [items]
    return [items[i : i + size] for i in range(0, len(items), size)]


def parse_top_k_values(values: str) -> List[int]:
    parts = [p.strip() for p in values.split(",") if p.strip()]
    out = []
    for p in parts:
        try:
            out.append(int(p))
        except Exception:
            continue
    return out or [1, 3, 5, 10]


def normalize_embedding_model(name: str) -> str:
    if name == "e5-large-v2":
        return "intfloat/e5-large-v2"
    return name


def normalize_reranker_model(name: Optional[str]) -> Optional[str]:
    if not name:
        return name
    if name == "bge-reranker-large":
        return "BAAI/bge-reranker-large"
    return name


def setup_inference_engine(
    model_name: str,
    baseline_config: Any,
    *,
    backend: str,
    use_offline: bool = False,
    strict: bool = True,
) -> InferenceEngine:
    """Setup inference engine based on backend.
    
    Args:
        model_name: Model to use for inference
        baseline_config: Configuration for the baseline
        backend: Backend to use (vllm, gemini, ollama)
        use_offline: Use offline/in-process engine (vllm only)
        strict: If True, fail if model doesn't match server's loaded model
    
    Returns:
        Configured inference engine
    
    Raises:
        RuntimeError: If strict=True and model validation fails
    """
    print(f"Setting up inference engine for {model_name} with backend={backend}...")

    if backend == "gemini":
        engine = GeminiAdapter(model_name=model_name)
    elif backend == "vllm":
        if use_offline:
            print("Using offline vLLM engine (in-process)")
            engine = VLLMOfflineAdapter(model_name)
        else:
            print(f"Using vLLM API at {baseline_config.api_base}")
            engine = VLLMAdapter(
                model_name=model_name,
                api_base=baseline_config.api_base,
            )
            
            # Strict validation: ensure the model on the server matches
            if strict and hasattr(engine, 'get_loaded_model'):
                loaded_model = engine.get_loaded_model()
                if loaded_model and loaded_model != model_name:
                    error_msg = (
                        f"\n{'='*70}\n"
                        f"ERROR: Model mismatch!\n"
                        f"  Requested model: {model_name}\n"
                        f"  Server has:      {loaded_model}\n"
                        f"\n"
                        f"Please restart the vLLM server with the correct model:\n"
                        f"  1. Kill the current server: pkill -f 'vllm serve'\n"
                        f"  2. Start with correct model:\n"
                        f"     vllm serve {model_name} --port 8000 --enable-prefix-caching\n"
                        f"{'='*70}"
                    )
                    print(error_msg)
                    raise RuntimeError(f"Model mismatch: expected {model_name}, server has {loaded_model}")
    elif backend == "ollama":
        print(f"Using Ollama API at {baseline_config.api_base}")
        engine = OllamaAdapter(
            model_name=model_name,
            api_base=baseline_config.api_base,
        )
    else:
        raise ValueError("Unsupported backend: choose vllm, gemini, or ollama")
    
    # Check if engine is ready
    if not engine.is_ready():
        error_msg = "Inference engine not ready."
        if backend == "vllm":
            error_msg += f"\nMake sure vLLM server is running with the correct model."
            error_msg += f"\nStart server with: vllm serve {model_name} --port 8000"
            if baseline_config.enable_prefix_caching:
                error_msg += " --enable-prefix-caching"
            error_msg += " --enable-prompt-tokens-details"
        elif backend == "ollama":
            error_msg += "\nMake sure Ollama is running and the model is pulled."
        
        if strict:
            print(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg)
        else:
            print(f"Warning: {error_msg}")
    
    return engine


def run_experiment(
    baseline: str,
    model: str,
    dataset: str,
    num_queries: int,
    max_tokens: int,
    api_base: str,
    use_offline: bool,
    output_dir: str,
    seed: int,
    *,
    backend: str,
    # IR / RAG
    top_k: int,
    embedding_model: str,
    ir_index_dir: str,
    rebuild_ir_index: bool,
    # Redis cache
    redis_host: str,
    redis_port: int,
    redis_db: int,
    redis_key_prefix: str,
    redis_ttl_seconds: Optional[int],
    flush_redis_namespace: bool,
    # Workload control
    repeat_queries: int,
    warmup_queries: int,
    workload_mode: str,
    batch_size: int,
    multi_turn_length: int,
    # Routing migration
    routing_switch_at: Optional[int],
    # Sharding simulation
    sharding_policy: str = "replicated",
    # Reranker
    reranker_model: Optional[str],
    reranker_device: str,
    # Prompt/context truncation
    truncate_prompt_tokens: Optional[int],
    max_context_chars: Optional[int],
    max_context_docs: Optional[int],
    # Speculative decoding
    speculative_model: Optional[str] = None,
    num_speculative_tokens: int = 5,
    speculative_method: str = "draft_model",
    baseline_label: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a single baseline experiment.
    
    Returns:
        Dict with experiment results and metrics
    """
    experiment_label = baseline_label or default_experiment_label(
        baseline,
        sharding_policy=sharding_policy,
        warmup_queries=warmup_queries,
    )

    print("=" * 70)
    print("CAGE Experiment")
    print(f"Baseline: {experiment_label}")
    print(f"Model: {model}")
    print(f"Dataset: {dataset}")
    print(f"Queries: {num_queries}")
    print(
        f"Workload: mode={workload_mode}, repeat_queries={repeat_queries}, warmup_queries={warmup_queries}, "
        f"batch_size={batch_size}, multi_turn_length={multi_turn_length}"
    )
    if truncate_prompt_tokens or max_context_chars or max_context_docs:
        print(
            "Truncation: "
            f"truncate_prompt_tokens={truncate_prompt_tokens}, "
            f"max_context_chars={max_context_chars}, "
            f"max_context_docs={max_context_docs}"
        )
    if baseline == "speculative" or speculative_model:
        print(
            f"Speculative: model={speculative_model}, "
            f"tokens={num_speculative_tokens}, method={speculative_method}"
        )
    print("=" * 70)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    append_command_log(output_path, sys.argv)
    stats_url = os.getenv("ROUTER_STATS_URL")
    if not stats_url and baseline == "distributed":
        stats_url = api_base
    if stats_url:
        snapshot_router_stats(output_path, stats_url)

    # Prometheus metrics (runner-level)
    prom_port = int(os.getenv("PROM_PORT", "9400"))
    label_kwargs = dict(baseline=experiment_label, backend=backend, dataset=dataset)

    metrics = get_runner_metrics()
    REQ_COUNTER = metrics["REQ_COUNTER"]
    ERR_COUNTER = metrics["ERR_COUNTER"]
    TTFT_HIST = metrics["TTFT_HIST"]
    LAT_HIST = metrics["LAT_HIST"]
    TOK_HIST = metrics["TOK_HIST"]
    PROMPT_TOK_HIST = metrics["PROMPT_TOK_HIST"]
    CACHED_PROMPT_TOK_HIST = metrics["CACHED_PROMPT_TOK_HIST"]
    CACHED_PROMPT_RATIO_HIST = metrics["CACHED_PROMPT_RATIO_HIST"]
    CACHED_PROMPT_REQ_COUNTER = metrics["CACHED_PROMPT_REQ_COUNTER"]
    CPU_GAUGE = metrics["CPU_GAUGE"]
    RSS_GAUGE = metrics["RSS_GAUGE"]

    def start_metrics_server():
        """Start the runner's Prometheus metrics server exactly once."""
        global _PROM_STARTED
        if _PROM_STARTED:
            return
        start_http_server(prom_port, registry=_PROM_REGISTRY)
        _PROM_STARTED = True
        print(f"[metrics] runner Prometheus on :{prom_port}")

    start_metrics_server()
    # Load baseline configuration
    baseline_config = get_baseline_config(
        baseline,
        api_base=api_base,
        embedding_model=embedding_model,
        top_k_retrieval=top_k,
        ir_index_dir=ir_index_dir,
        ir_rebuild=rebuild_ir_index,
        redis_host=redis_host,
        redis_port=redis_port,
        redis_db=redis_db,
        redis_key_prefix=redis_key_prefix,
        reranker_model=reranker_model,
        reranker_top_k=top_k if reranker_model else None,
        # Speculative decoding config
        speculative_model=speculative_model,
        num_speculative_tokens=num_speculative_tokens,
        speculative_method=speculative_method,
    )
    print(f"\nBaseline config: {baseline_config.description}")
    
    # Validate speculative baseline requirements
    if baseline == "speculative":
        print(
            "Warning: speculative baseline settings are recorded in the experiment config, "
            "but the current inference path does not yet forward speculative decoding "
            "parameters to the backend adapter."
        )
        if speculative_method == "draft_model" and not speculative_model:
            print("Warning: speculative baseline with method=draft_model requires --speculative-model")
            print("Hint: For Qwen3-4B, try --speculative-model Qwen/Qwen3-0.6B")
            print("      For ngram method, no draft model needed: --speculative-method ngram")
    
    # Check requirements
    requirements = check_baseline_requirements(baseline_config)
    print(f"Requirements check: {requirements}")

    if baseline_config.use_faiss and not requirements.get("faiss_available", False):
        raise RuntimeError(
            "FAISS not available but this baseline requires retrieval. "
            "Install faiss-cpu and re-run."
        )

    if baseline_config.baseline_type.value == "distributed":
        # Configure router simulation policy
        try:
            import requests
            requests.post(f"{api_base}/sharding-policy", json={"policy": sharding_policy}, timeout=5)
            print(f"[simulation] Set router sharding policy to: {sharding_policy}")
        except Exception as e:
            print(f"Warning: Failed to configure router simulation: {e}")
            
        if not requirements.get("gpu_available", False):
            print(
                "Warning: no GPU was detected. Distributed runs can still execute through "
                "multiple router-managed replicas, but sharded_context remains a simulated "
                "transfer model rather than real KV movement."
            )

    if baseline_config.baseline_type.value == "redis" and not requirements["redis_available"]:
        print("Error: Redis not available. Start Redis with:")
        print("  docker run -d --name cage-redis -p 6379:6379 redis:alpine")
        sys.exit(1)
    
    # Load dataset
    print(f"\nLoading dataset '{dataset}'...")
    loader = get_loader(dataset, split=default_dataset_split(dataset), seed=seed)
    base_examples = loader.load(max_examples=num_queries)
    code_dataset = is_code_dataset(dataset, base_examples)

    if repeat_queries < 1:
        raise ValueError("repeat_queries must be >= 1")
    if warmup_queries < 0:
        raise ValueError("warmup_queries must be >= 0")
    if warmup_queries > 0 and not base_examples:
        raise ValueError("warmup_queries > 0 requires at least one base example")

    def build_work_units(workload_examples: List[CAGExample]) -> List[List[CAGExample]]:
        if workload_mode == "single":
            return [[ex] for ex in workload_examples]
        if workload_mode == "batched":
            groups: Dict[str, List[CAGExample]] = defaultdict(list)
            order: List[str] = []
            for ex in workload_examples:
                ctx_key = hashlib.sha1("||".join(ex.context or []).encode("utf-8")).hexdigest()
                if ctx_key not in groups:
                    order.append(ctx_key)
                groups[ctx_key].append(ex)
            out: List[List[CAGExample]] = []
            for key in order:
                out.extend(chunked(groups[key], batch_size))
            return out
        if workload_mode == "multi_turn":
            return chunked(workload_examples, max(1, multi_turn_length))
        raise ValueError("workload_mode must be one of: single, batched, multi_turn")

    warmup_examples: List[CAGExample] = []
    for idx in range(warmup_queries):
        ex = base_examples[idx % len(base_examples)]
        warmup_examples.append(
            CAGExample(
                id=f"{ex.id}__warmup{idx}",
                question=ex.question,
                context=ex.context,
                answer=ex.answer,
                metadata={**(ex.metadata or {}), "warmup": True, "repeat_index": None},
            )
        )

    measured_examples: List[CAGExample] = []
    for rep in range(repeat_queries):
        for ex in base_examples:
            ex_id = ex.id if rep == 0 else f"{ex.id}__rep{rep}"
            measured_examples.append(
                CAGExample(
                    id=ex_id,
                    question=ex.question,
                    context=ex.context,
                    answer=ex.answer,
                    metadata={**(ex.metadata or {}), "warmup": False, "repeat_index": rep},
                )
            )

    print(
        f"Loaded {len(base_examples)} base examples "
        f"({len(warmup_examples)} warmup requests, {len(measured_examples)} measured requests)"
    )

    warmup_work_units = build_work_units(warmup_examples)
    work_units = build_work_units(measured_examples)

    # IR index / retriever (for RAG/Redis/Hybrid baselines)
    ir_index = None
    corpus_docs = None
    if baseline_config.use_faiss:
        print("\nBuilding/loading IR index (FAISS)...")
        base_dir = Path(baseline_config.ir_index_dir)
        index_dir = default_index_dir(
            base_dir=base_dir,
            dataset_name=dataset,
            embedding_model=baseline_config.embedding_model,
        )
        corpus_docs = build_corpus_from_contexts(base_examples, dataset_name=dataset)
        print(f"IR corpus documents: {len(corpus_docs)}")
        ir_index = ensure_ir_index(
            index_dir=index_dir,
            documents=corpus_docs,
            embedding_model=baseline_config.embedding_model,
            rebuild=bool(baseline_config.ir_rebuild),
            device="cpu",
        )
        print(f"IR index ready at: {index_dir}")

    # Optional Redis retrieval cache
    retrieval_cache = None
    if baseline_config.baseline_type.value in {"redis", "hybrid"}:
        print("\nConnecting to Redis retrieval cache...")
        rcfg = RedisConfig(
            host=baseline_config.redis_host,
            port=baseline_config.redis_port,
            db=baseline_config.redis_db,
            key_prefix=baseline_config.redis_key_prefix,
        )
        rclient = RedisClient(rcfg)
        if not rclient.ping():
            raise RuntimeError(
                f"Redis not reachable at {rcfg.host}:{rcfg.port} (db={rcfg.db}). "
                "Start Redis (Docker): docker run -d --name cage-redis -p 6379:6379 redis:alpine"
            )
        retrieval_cache = RetrievalCache(rclient)
        if flush_redis_namespace:
            deleted_keys = retrieval_cache.clear()
            print(
                f"Flushed Redis retrieval namespace '{baseline_config.redis_key_prefix}' "
                f"({deleted_keys} keys deleted)"
            )
        print("Redis cache connected")
    # Optional reranker
    reranker = None
    if baseline_config.use_faiss and reranker_model:
        try:
            reranker = CrossEncoderReranker(reranker_model, device=reranker_device)
            print(f"Reranker enabled: {reranker_model}")
        except Exception as e:
            print(f"Warning: failed to initialize reranker {reranker_model}: {e}")

    # Setup inference engine
    engine = setup_inference_engine(model, baseline_config, backend=backend, use_offline=use_offline)
    
    # Setup evaluators
    print("\nInitializing evaluators...")
    quality_evaluator = QualityEvaluator(device="cpu")
    performance_evaluator = PerformanceEvaluator(monitor_resources=True)
    cache_tracker = CacheMetricsTracker()
    code_evaluator = CodeQualityEvaluator() if code_dataset else None
    
    # Run experiment
    print(f"\nRunning experiment with {len(measured_examples)} measured queries...")
    print("-" * 70)

    results: List[Dict[str, Any]] = []
    sent_requests = 0
    measured_processed = 0

    def maybe_reshuffle_router(idx: int) -> None:
        if (
            baseline_config.baseline_type.value == "distributed"
            and routing_switch_at is not None
            and idx == routing_switch_at
        ):
            try:
                import random
                import requests

                replicas = [
                    {"replica_id": f"replica-{idx+1}", "api_base": f"http://localhost:800{idx+1}"}
                    for idx in range(3)
                ]
                random.shuffle(replicas)
                requests.post(f"{api_base}/configure", json=replicas, timeout=5)
                print(f"[routing] Reshuffled replicas at request {idx}: {replicas}")
            except Exception as e:
                print(f"[routing] Failed to reshuffle replicas: {e}")

    def prepare_example(example: CAGExample) -> Dict[str, Any]:
        question = example.question
        baseline_mode = baseline_config.baseline_type.value
        used_contexts: List[str] = list(example.context or [])

        retrieval_cached = False
        retrieval_hit = None
        retrieval_top1_score = None
        retrieved_doc_ids: List[str] = []
        retrieval_reranked = False

        if baseline_config.baseline_type.value in {"rag", "redis", "hybrid"}:
            if ir_index is None:
                raise RuntimeError("IR index not initialized for retrieval-backed baseline")

            hits_payload = None
            if retrieval_cache is not None:
                hits_payload = retrieval_cache.get(
                    dataset=dataset,
                    embedding_model=baseline_config.embedding_model,
                    top_k=baseline_config.top_k_retrieval,
                    query=question,
                )

            if hits_payload:
                retrieval_cached = True
                hits = [
                    IRHit(doc_id=h["doc_id"], score=float(h.get("score", 0.0)))
                    for h in hits_payload
                    if "doc_id" in h
                ]
            else:
                hits = ir_index.search(question, top_k=baseline_config.top_k_retrieval)
                if retrieval_cache is not None:
                    retrieval_cache.set(
                        dataset=dataset,
                        embedding_model=baseline_config.embedding_model,
                        top_k=baseline_config.top_k_retrieval,
                        query=question,
                        hits=[{"doc_id": h.doc_id, "score": h.score} for h in hits],
                        ttl_seconds=redis_ttl_seconds,
                    )

            if reranker is not None and hits:
                hits = reranker.rerank(question, hits, ir_index)
                retrieval_reranked = True

            retrieved_doc_ids = [h.doc_id for h in hits]
            retrieval_top1_score = hits[0].score if hits else 0.0

            retrieved_docs = ir_index.resolve_hits(hits)
            used_contexts = [d.text for d in retrieved_docs]

            gold_doc_ids = [stable_text_id(c) for c in (example.context or []) if c]
            retrieval_hit = retrieval_hit_rate(
                gold_doc_ids=gold_doc_ids,
                retrieved_doc_ids=retrieved_doc_ids,
            )
            if baseline_config.baseline_type.value == "redis":
                baseline_mode = (
                    "redis_retrieval_cache_hit" if retrieval_cached else "redis_retrieval_cache_miss"
                )
            elif baseline_config.baseline_type.value == "hybrid":
                baseline_mode = (
                    "hybrid_retrieval_cache_hit" if retrieval_cached else "hybrid_retrieval_cache_miss"
                )

        elif baseline_config.baseline_type.value == "distributed":
            used_contexts = list(example.context or [])
        else:
            used_contexts = list(example.context or [])

        if max_context_chars is not None and max_context_chars > 0:
            used_contexts = [c[:max_context_chars] for c in used_contexts]
        if max_context_docs is not None and max_context_docs > 0:
            used_contexts = used_contexts[:max_context_docs]

        return {
            "question": question,
            "baseline_mode": baseline_mode,
            "used_contexts": used_contexts,
            "retrieval_cached": retrieval_cached,
            "retrieval_hit": retrieval_hit,
            "retrieval_top1_score": retrieval_top1_score,
            "retrieved_doc_ids": retrieved_doc_ids,
            "retrieval_reranked": retrieval_reranked,
        }

    def record_result(
        example: CAGExample,
        meta: Dict[str, Any],
        response: Any,
        *,
        batch_id: int,
        turn_index: int,
    ) -> None:

        question = meta["question"]
        used_contexts = meta["used_contexts"]

        performance_evaluator.record_request(
            request_id=example.id,
            ttft_ms=response.ttft_ms,
            total_time_ms=response.total_time_ms,
            num_tokens=response.num_tokens,
            error=response.error,
        )

        REQ_COUNTER.labels(**label_kwargs).inc()
        if response.error:
            ERR_COUNTER.labels(**label_kwargs).inc()
        else:
            TTFT_HIST.labels(**label_kwargs).observe(response.ttft_ms / 1000.0)
            LAT_HIST.labels(**label_kwargs).observe(response.total_time_ms / 1000.0)
            TOK_HIST.labels(**label_kwargs).observe(response.num_tokens)

            if response.prompt_tokens is not None:
                PROMPT_TOK_HIST.labels(**label_kwargs).observe(response.prompt_tokens)
            if response.cached_prompt_tokens is not None:
                CACHED_PROMPT_TOK_HIST.labels(**label_kwargs).observe(response.cached_prompt_tokens)
                if response.cached_prompt_tokens > 0:
                    CACHED_PROMPT_REQ_COUNTER.labels(**label_kwargs).inc()
                if response.prompt_tokens is not None:
                    ratio = float(response.cached_prompt_tokens) / max(float(response.prompt_tokens), 1.0)
                    CACHED_PROMPT_RATIO_HIST.labels(**label_kwargs).observe(ratio)
        try:
            proc = psutil.Process()
            CPU_GAUGE.set(proc.cpu_percent() / psutil.cpu_count())
            RSS_GAUGE.set(proc.memory_info().rss / 1024 / 1024)
        except Exception:
            pass

        # Record cache metrics (distributed telemetry)
        is_distributed = baseline_config.baseline_type.value == "distributed"
        is_prefix_cache = baseline_config.enable_prefix_caching
        
        # If we went through router and it selected a replica (x-router-replica header)
        router_replica = response.router_replica
        cached_tokens = response.cached_prompt_tokens or 0
        transfer_params = response.kv_transfer_params or {}
        transfer_required = bool(transfer_params.get("transfer_required"))
        transfer_latency_ms = float(transfer_params.get("transfer_latency_ms") or 0.0)
        transfer_bytes = int(transfer_params.get("transfer_bytes") or 0)
        
        if cached_tokens > 0:
            if is_distributed and router_replica:
                if transfer_required or transfer_latency_ms > 0.0 or transfer_bytes > 0:
                    cache_tracker.record_remote_hit(
                        fetch_latency_ms=transfer_latency_ms,
                        bytes_transferred=transfer_bytes,
                    )
                else:
                    cache_tracker.record_local_hit()
            elif is_prefix_cache:
                cache_tracker.record_local_hit()
        elif response.prompt_tokens and response.prompt_tokens > 0:
            cache_tracker.record_miss()

        quality_metrics = quality_evaluator.evaluate(
            question=question,
            context=used_contexts,
            generated_text=response.generated_text,
            reference_answer=example.answer,
        )

        code_metrics = (
            code_evaluator.evaluate(response.generated_text)
            if code_evaluator is not None
            else None
        )

        kv_transfer_params_str = ""
        if response.kv_transfer_params is not None:
            try:
                kv_transfer_params_str = json.dumps(response.kv_transfer_params, sort_keys=True)
            except Exception:
                kv_transfer_params_str = ""

        cached_ratio = None
        if response.prompt_tokens is not None and response.cached_prompt_tokens is not None:
            cached_ratio = float(response.cached_prompt_tokens) / max(float(response.prompt_tokens), 1.0)

        repeat_index = None
        if isinstance(example.metadata, dict):
            repeat_index = example.metadata.get("repeat_index")

        result = {
            "example_id": example.id,
            "baseline": experiment_label,
            "baseline_family": baseline_config.baseline_type.value,
            "baseline_mode": meta["baseline_mode"],
            "workload_mode": workload_mode,
            "batch_id": batch_id,
            "turn_index": turn_index,
            "repeat_index": repeat_index,
            "question": question,
            "reference_answer": example.answer,
            "generated_answer": response.generated_text,
            "ttft_ms": response.ttft_ms,
            "latency_ms": response.total_time_ms,
            "num_tokens": response.num_tokens,
            "prompt_tokens": response.prompt_tokens,
            "cached_prompt_tokens": response.cached_prompt_tokens,
            "cached_prompt_ratio": cached_ratio,
            "finish_reason": response.finish_reason,
            "error": response.error,
            "kv_transfer_params": kv_transfer_params_str,
            "routed_replica": response.router_replica or "",
            "retrieval_top1_score": meta["retrieval_top1_score"],
            "retrieval_hit": meta["retrieval_hit"],
            "retrieval_cached": meta["retrieval_cached"],
            "retrieval_reranked": meta["retrieval_reranked"],
            "retrieved_doc_ids": ";".join(meta["retrieved_doc_ids"]) if meta["retrieved_doc_ids"] else "",
            **quality_metrics.to_dict(),
        }
        if code_metrics is not None:
            result.update(
                {
                    "code_valid_syntax": code_metrics.is_valid_syntax,
                    "code_complexity": code_metrics.complexity,
                    "code_security_issues": ";".join(code_metrics.security_issues),
                }
            )
        results.append(result)

    def execute_work_units(
        units: List[List[CAGExample]],
        *,
        collect_results: bool,
        stage_name: str,
    ) -> None:
        nonlocal sent_requests, measured_processed
        batch_id = 0
        total_stage_examples = sum(len(unit) for unit in units)
        stage_processed = 0

        for unit in units:
            batch_id += 1

            if workload_mode == "multi_turn":
                history: List[Tuple[str, str]] = []
                for turn_idx, example in enumerate(unit):
                    maybe_reshuffle_router(sent_requests)
                    meta = prepare_example(example)
                    prompt = format_multi_turn_prompt(
                        meta["question"],
                        meta["used_contexts"],
                        history=history,
                    )
                    request = InferenceRequest(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=0.7,
                        top_p=0.95,
                        request_id=example.id,
                        truncate_prompt_tokens=truncate_prompt_tokens,
                    )
                    stream_flag = backend in {"vllm", "ollama"}
                    response = engine.generate(request, stream=stream_flag)
                    if collect_results:
                        record_result(example, meta, response, batch_id=batch_id, turn_index=turn_idx)
                    elif response.error:
                        print(f"[warmup] Request {example.id} failed: {response.error}")
                    history.append((meta["question"], response.generated_text or ""))
                    sent_requests += 1
                    stage_processed += 1
                    if collect_results:
                        measured_processed += 1
            else:
                metas: List[Dict[str, Any]] = []
                requests: List[InferenceRequest] = []
                for example in unit:
                    maybe_reshuffle_router(sent_requests)
                    meta = prepare_example(example)
                    prompt = format_qa_prompt(meta["question"], meta["used_contexts"])
                    request = InferenceRequest(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=0.7,
                        top_p=0.95,
                        request_id=example.id,
                        truncate_prompt_tokens=truncate_prompt_tokens,
                    )
                    metas.append(meta)
                    requests.append(request)

                if len(requests) == 1:
                    stream_flag = backend in {"vllm", "ollama"}
                    responses = [engine.generate(requests[0], stream=stream_flag)]
                else:
                    responses = engine.batch_generate(requests)

                for example, meta, response in zip(unit, metas, responses):
                    if collect_results:
                        record_result(example, meta, response, batch_id=batch_id, turn_index=0)
                    elif response.error:
                        print(f"[warmup] Request {example.id} failed: {response.error}")
                    sent_requests += 1
                    stage_processed += 1
                    if collect_results:
                        measured_processed += 1

            if total_stage_examples and (
                stage_processed % 10 == 0 or stage_processed == total_stage_examples
            ):
                print(f"{stage_name}: processed {stage_processed}/{total_stage_examples} requests...")

    if warmup_work_units:
        print(
            f"\nRunning warmup stage with {len(warmup_examples)} requests "
            "(excluded from metrics/results)..."
        )
        execute_work_units(warmup_work_units, collect_results=False, stage_name="Warmup")
        print("Warmup complete.")
        print("-" * 70)

    performance_evaluator.start()
    execute_work_units(work_units, collect_results=True, stage_name="Measured")
    performance_evaluator.stop()
    
    print("-" * 70)
    print("Experiment complete!")
    
    # Compute aggregate metrics
    perf_metrics = performance_evaluator.compute_metrics()
    cache_metrics = cache_tracker.get_metrics()
    
    print("\n" + "=" * 70)
    print("PERFORMANCE METRICS")
    print("=" * 70)
    print(f"Throughput: {perf_metrics.queries_per_second:.2f} QPS, {perf_metrics.tokens_per_second:.2f} tokens/sec")
    print(f"Latency (avg): {perf_metrics.avg_latency_ms:.2f} ms")
    print(f"Latency (p95): {perf_metrics.p95_latency_ms:.2f} ms")
    print(f"TTFT (avg): {perf_metrics.avg_ttft_ms:.2f} ms")
    print(f"TTFT (p95): {perf_metrics.p95_ttft_ms:.2f} ms")
    print(f"CPU (avg): {perf_metrics.avg_cpu_percent:.1f}%")
    print(f"Memory (peak): {perf_metrics.peak_memory_mb:.1f} MB")
    
    print("\n" + "=" * 70)
    print("CACHE TELEMETRY")
    print("=" * 70)
    print(f"Local Hit Ratio: {cache_metrics['local_hit_ratio']:.3f}")
    print(f"Remote Hit Ratio: {cache_metrics['remote_hit_ratio']:.3f}")
    print(f"Miss Ratio: {cache_metrics['miss_ratio']:.3f}")
    print(f"Avg Remote Fetch: {cache_metrics['avg_remote_fetch_ms']:.2f} ms")
    print(f"Total Transfer: {cache_metrics['total_transfer_mb']:.2f} MB")
    
    if any(r.get("completeness_bertscore") is None for r in results):
        print("Warning: BERTScore was unavailable for part of this run; clearing completeness_bertscore for all rows.")
        for row in results:
            row["completeness_bertscore"] = None
    # Compute average quality metrics
    import numpy as np

    def mean_or_none(metric_key: str) -> Optional[float]:
        values = [r[metric_key] for r in results if r.get(metric_key) is not None]
        return float(np.mean(values)) if values else None
    # Aggregate every quality key (keys match quality_keys so cross-trial
    # aggregation downstream picks them all up; None values are excluded).
    avg_quality = {
        "grounding_score": mean_or_none("grounding_score"),
        "hallucination_detected": mean_or_none("hallucination_detected"),  # -> hallucination RATE
        "hallucinated_span_ratio": mean_or_none("hallucinated_span_ratio"),
        "faithfulness": mean_or_none("faithfulness"),
        "supported_claim_ratio": mean_or_none("supported_claim_ratio"),
        "context_relevance": mean_or_none("context_relevance"),
        "relevance": mean_or_none("relevance"),
        "completeness_bertscore": mean_or_none("completeness_bertscore"),
        "completeness_rouge_l": mean_or_none("completeness_rouge_l"),
        "f1_score": mean_or_none("f1_score"),
        "precision": mean_or_none("precision"),
        "recall": mean_or_none("recall"),
        "exact_match": mean_or_none("exact_match"),
    }

    print("\n" + "=" * 70)
    print("QUALITY METRICS (Average)")
    print("=" * 70)
    print(f"Grounding (LettuceDetect): {format_metric(avg_quality['grounding_score'])}")
    print(f"Hallucination rate: {format_metric(avg_quality['hallucination_detected'])}")
    print(f"Faithfulness (NLI claim-level): {format_metric(avg_quality['faithfulness'])}")
    print(f"Supported-claim ratio: {format_metric(avg_quality['supported_claim_ratio'])}")
    print(f"Context relevance (retriever diagnostic): {format_metric(avg_quality['context_relevance'])}")
    print(f"F1 / EM: {format_metric(avg_quality['f1_score'])} / {format_metric(avg_quality['exact_match'])}")
    print(f"Completeness (BERTScore): {format_metric(avg_quality['completeness_bertscore'])}")
    print(f"Completeness (ROUGE-L): {format_metric(avg_quality['completeness_rouge_l'])}")

    def summarize_subset(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not rows:
            return None
        latencies = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
        ttfts = [r["ttft_ms"] for r in rows if r.get("ttft_ms") is not None]
        tokens = [r["num_tokens"] for r in rows if r.get("num_tokens") is not None]
        total_time_s = sum(latencies) / 1000.0 if latencies else 0.0
        total_tokens = sum(tokens) if tokens else 0
        error_rate = float(
            sum(1 for r in rows if r.get("error")) / max(len(rows), 1)
        )
        return {
            "count": len(rows),
            "avg_latency_ms": float(np.mean(latencies)) if latencies else None,
            "avg_ttft_ms": float(np.mean(ttfts)) if ttfts else None,
            "avg_num_tokens": float(np.mean(tokens)) if tokens else None,
            "tokens_per_second": float(total_tokens / total_time_s) if total_time_s > 0 else None,
            "error_rate": error_rate,
        }
    repeat_summary: Optional[Dict[str, Any]] = None
    if repeat_queries > 1:
        repeat_summary = {}
        repeat_indices = sorted(
            {
                int(r["repeat_index"])
                for r in results
                if r.get("repeat_index") is not None
            }
        )
        for repeat_idx in repeat_indices:
            repeat_summary[f"repeat_{repeat_idx}"] = summarize_subset(
                [r for r in results if r.get("repeat_index") == repeat_idx]
            )

    # Retrieval metrics (only meaningful when retrieval is used)
    retrieval_rows = [r for r in results if r.get("retrieval_hit") is not None]
    retrieval_summary: Optional[Dict[str, Any]] = None
    if retrieval_rows:
        avg_retrieval_hit = float(np.mean([r["retrieval_hit"] for r in retrieval_rows]))
        avg_top1 = float(
            np.mean(
                [
                    r["retrieval_top1_score"]
                    for r in retrieval_rows
                    if r.get("retrieval_top1_score") is not None
                ]
            )
        )
        cache_rate = float(
            np.mean([1.0 if r.get("retrieval_cached") else 0.0 for r in retrieval_rows])
        )

        retrieval_summary = {
            "avg_hit": avg_retrieval_hit,
            "avg_top1_score": avg_top1,
            "cache_rate": cache_rate,
            "top_k": baseline_config.top_k_retrieval,
            "embedding_model": baseline_config.embedding_model,
            "reranker_model": baseline_config.reranker_model,
        }

        print("\n" + "=" * 70)
        print("RETRIEVAL METRICS (Average)")
        print("=" * 70)
        print(f"Hit@{baseline_config.top_k_retrieval}: {avg_retrieval_hit:.3f}")
        print(f"Top-1 score: {avg_top1:.3f}")
        if baseline_config.baseline_type.value in {"redis", "hybrid"}:
            print(f"Retrieval cache hit rate: {cache_rate:.3f}")

    # Prompt-cache telemetry (only meaningful when the backend returns usage)
    cache_rows = [
        r
        for r in results
        if r.get("prompt_tokens") is not None and r.get("cached_prompt_tokens") is not None
    ]
    cache_summary: Optional[Dict[str, Any]] = None
    if cache_rows:
        total_prompt_tokens = int(sum(int(r["prompt_tokens"]) for r in cache_rows))
        total_cached_prompt_tokens = int(sum(int(r["cached_prompt_tokens"]) for r in cache_rows))
        overall_cached_ratio = float(total_cached_prompt_tokens / max(total_prompt_tokens, 1))
        cached_request_rate = float(
            np.mean(
                [
                    1.0 if int(r.get("cached_prompt_tokens") or 0) > 0 else 0.0
                    for r in cache_rows
                ]
            )
        )

        cache_summary = {
            "avg_prompt_tokens": float(np.mean([r["prompt_tokens"] for r in cache_rows])),
            "avg_cached_prompt_tokens": float(
                np.mean([r["cached_prompt_tokens"] for r in cache_rows])
            ),
            "overall_cached_prompt_ratio": overall_cached_ratio,
            "cached_request_rate": cached_request_rate,
            "num_requests_with_usage": len(cache_rows),
        }

        print("\n" + "=" * 70)
        print("PROMPT-CACHE TELEMETRY (Average)")
        print("=" * 70)
        print(f"Avg prompt tokens: {cache_summary['avg_prompt_tokens']:.1f}")
        print(f"Avg cached prompt tokens: {cache_summary['avg_cached_prompt_tokens']:.1f}")
        print(f"Overall cached prompt ratio: {cache_summary['overall_cached_prompt_ratio']:.3f}")
        print(f"Cached request rate: {cache_summary['cached_request_rate']:.3f}")

    elif backend == "vllm" and results:
        print("\n" + "=" * 70)
        print("PROMPT-CACHE TELEMETRY")
        print("=" * 70)
        print(
            "Backend did not return prompt/cache usage telemetry. "
            "To enable cached token reporting, start vLLM with --enable-prompt-tokens-details."
        )

    distributed_summary: Optional[Dict[str, Any]] = None
    if baseline_config.baseline_type.value == "distributed":
        require_distinct_replicas = (
            str(os.getenv("CAGE_REQUIRE_DISTINCT_REPLICAS", "0")).lower()
            in {"1", "true", "yes", "on"}
        )
        distributed_summary = validate_distributed_artifacts(
            results,
            api_base=api_base,
            sharding_policy=sharding_policy,
            require_distinct_replicas=require_distinct_replicas,
        )

    git_metadata = capture_git_metadata()
    backend_metadata = capture_backend_metadata(
        api_base=api_base,
        backend=backend,
        model_name=model,
        use_offline=use_offline,
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_baseline = experiment_label.replace("/", "_").replace(" ", "_")
    results_file = output_path / f"{file_baseline}_{dataset}_{timestamp}_results.csv"
    metrics_file = output_path / f"{file_baseline}_{dataset}_{timestamp}_metrics.json"
    stable_results_file = output_path / "results.csv"
    stable_metrics_file = output_path / "metrics.json"
    
    # Save detailed results as CSV
    with open(results_file, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    with open(stable_results_file, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    print(f"\nResults saved to: {results_file}")
    
    # Save aggregate metrics as JSON
    experiment_summary = {
        "experiment": {
            "baseline": experiment_label,
            "baseline_family": baseline_config.baseline_type.value,
            "model": model,
            "dataset": dataset,
            "dataset_split": default_dataset_split(dataset),
            "num_queries": len(base_examples),
            "num_measured_requests": len(measured_examples),
            "num_warmup_requests": len(warmup_examples),
            "max_tokens": max_tokens,
            "timestamp": timestamp,
            "seed": seed,
            "backend": backend,
            "api_base": api_base,
        },
        "workload": {
            "repeat_queries": repeat_queries,
            "warmup_queries": warmup_queries,
            "mode": workload_mode,
            "batch_size": batch_size,
            "multi_turn_length": multi_turn_length,
        },
        "prompt_truncation": {
            "truncate_prompt_tokens": truncate_prompt_tokens,
            "max_context_chars": max_context_chars,
            "max_context_docs": max_context_docs,
        },
        "run_metadata": {
            **git_metadata,
            "backend": backend_metadata,
            "cli_args": sys.argv,
        },
        "system": {
            "snapshot": capture_system_snapshot(),
            "env": capture_env_snapshot(),
            "backend_versions": backend_metadata,
        },
        "baseline_config": baseline_config.to_dict(),
        "performance": perf_metrics.to_dict(),
        "cache_telemetry": cache_metrics,
        "distributed": distributed_summary,
        "quality": avg_quality,
        "retrieval": retrieval_summary,
        "prompt_cache": cache_summary,
        "warmup": {
            "num_queries": warmup_queries,
            "num_requests": len(warmup_examples),
            "included_in_metrics": False,
        },
        "repeat_passes": repeat_summary,
    }
    
    with open(metrics_file, "w") as f:
        json.dump(experiment_summary, f, indent=2)

    with open(stable_metrics_file, "w") as f:
        json.dump(experiment_summary, f, indent=2)
    
    print(f"Metrics saved to: {metrics_file}")
    print("=" * 70)
    
    # Cleanup
    engine.shutdown()
    
    return experiment_summary


def main():
    parser = argparse.ArgumentParser(
        description="Run CAGE baseline experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument(
        "--baseline",
        required=True,
        choices=["no_cache", "prefix_cache", "redis", "rag", "distributed", "hybrid", "speculative"],
        help="Baseline to evaluate",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name (e.g., meta-llama/Llama-3.2-1B-Instruct, HuggingFaceTB/SmolLM2-135M-Instruct)",
    )
    
    # Optional arguments
    parser.add_argument(
        "--dataset",
        default="squad_v2",
        choices=["hotpotqa", "qasper", "squad_v2", "trivia_qa", "humaneval", "mbpp", "hpc_code"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=500,
        help="Number of queries to process (keep this consistent across baselines within a phase)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate per query",
    )
    parser.add_argument(
        "--api-base",
        default="http://localhost:8000",
        help="LLM API base URL (vLLM router or direct vLLM)",
    )
    parser.add_argument(
        "--backend",
        choices=["vllm", "gemini", "ollama"],
        default="vllm",
        help="Inference backend",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use offline vLLM engine (in-process) instead of API (backend=vllm)",
    )
    parser.add_argument(
        "--output-dir",
        default="./analysis/results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--baseline-label",
        default=None,
        help="Optional label used for artifacts and summaries when multiple runs share one baseline family.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Workload controls
    parser.add_argument(
        "--repeat-queries",
        type=int,
        default=1,
        help="Repeat the same set of queries N times (useful for cache warmup/testing)",
    )
    parser.add_argument(
        "--workload-mode",
        choices=["single", "batched", "multi_turn"],
        default="single",
        help="Workload mode: single-shot, batched shared-context, or multi-turn conversational",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for batched workload mode",
    )
    parser.add_argument(
        "--multi-turn-length",
        type=int,
        default=3,
        help="Number of turns per conversation in multi-turn mode",
    )

    # Prompt / context truncation
    parser.add_argument(
        "--truncate-prompt-tokens",
        type=int,
        default=None,
        help="If set, truncate prompt tokens to this length (vLLM only).",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=None,
        help="If set, truncate each context document to N characters before prompting.",
    )
    parser.add_argument(
        "--max-context-docs",
        type=int,
        default=None,
        help="If set, limit the number of context documents included in the prompt.",
    )

    # IR / RAG options
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-k documents to retrieve for RAG/Redis/Hybrid",
    )
    parser.add_argument(
        "--top-k-sweep",
        action="store_true",
        help="If set, run a sweep over multiple top-k values for retrieval baselines",
    )
    parser.add_argument(
        "--top-k-values",
        default="1,3,5,10",
        help="Comma-separated top-k values to sweep when --top-k-sweep is set",
    )
    parser.add_argument(
        "--embedding-model",
        default="intfloat/e5-large-v2",
        help="SentenceTransformers model used for retrieval embeddings",
    )
    parser.add_argument(
        "--reranker-model",
        default="BAAI/bge-reranker-large",
        help="Optional cross-encoder reranker model (set to 'none' to disable)",
    )
    parser.add_argument(
        "--reranker-device",
        default="cpu",
        help="Device for reranker model (e.g., cpu or cuda)",
    )
    parser.add_argument(
        "--ir-index-dir",
        default="./experiments/ir_index",
        help="Base directory to store/load FAISS IR indexes",
    )
    parser.add_argument(
        "--rebuild-ir-index",
        action="store_true",
        help="Force rebuilding the FAISS IR index",
    )

    # Redis options (baseline=redis or hybrid)
    parser.add_argument(
        "--redis-host",
        default="localhost",
        help="Redis host (baseline=redis or hybrid)",
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=6379,
        help="Redis port (baseline=redis or hybrid)",
    )
    parser.add_argument(
        "--redis-db",
        type=int,
        default=0,
        help="Redis DB number (baseline=redis or hybrid)",
    )
    parser.add_argument(
        "--redis-ttl-seconds",
        type=int,
        default=None,
        help="Optional TTL for cached retrieval results in Redis",
    )
    parser.add_argument(
        "--redis-key-prefix",
        default="cage",
        help="Redis key prefix used to isolate retrieval-cache namespaces across runs",
    )
    parser.add_argument(
        "--flush-redis-namespace",
        action="store_true",
        help="Flush the selected Redis retrieval-cache namespace before the run starts",
    )

    # Routing / migration (distributed baseline)
    parser.add_argument(
        "--routing-switch-at",
        type=int,
        default=None,
        help="If set, instruct router to reshuffle replicas after N requests (baseline=distributed)",
    )

    parser.add_argument(
        "--sharding-policy",
        default="replicated",
        choices=["replicated", "sharded_context"],
        help="Policy for the distributed baseline (replicated=real router-managed replicas, sharded_context=simulated context-parallel transfer)",
    )

    # Speculative decoding options
    parser.add_argument(
        "--speculative-model",
        default=None,
        help="Draft model for speculative decoding (e.g., 'Qwen/Qwen3-0.6B' for Qwen3-4B main model)",
    )
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=5,
        help="Number of tokens to speculate per step (default: 5)",
    )
    parser.add_argument(
        "--speculative-method",
        default="draft_model",
        choices=["draft_model", "ngram", "suffix", "medusa", "eagle"],
        help="Speculative decoding method (default: draft_model)",
    )

    # Statistical rigor options
    parser.add_argument(
        "--num-trials",
        type=int,
        default=1,
        help="Number of independent trials to run (for statistical rigor, use >=3)",
    )
    parser.add_argument(
        "--warmup-queries",
        type=int,
        default=0,
        help="Number of warmup queries before measurement (excluded from metrics)",
    )

    args = parser.parse_args()
    embedding_model = normalize_embedding_model(args.embedding_model)
    reranker_model = normalize_reranker_model(args.reranker_model)
    if reranker_model and str(reranker_model).lower() in {"none", "null", "false", "0"}:
        reranker_model = None

    def _run_with_top_k(top_k_value: int) -> None:
        run_experiment(
            baseline=args.baseline,
            model=args.model,
            dataset=args.dataset,
            num_queries=args.num_queries,
            max_tokens=args.max_tokens,
            api_base=args.api_base,
            use_offline=args.offline,
            output_dir=args.output_dir,
            seed=args.seed,
            backend=args.backend,
            top_k=top_k_value,
            embedding_model=embedding_model,
            ir_index_dir=args.ir_index_dir,
            rebuild_ir_index=args.rebuild_ir_index,
            redis_host=args.redis_host,
            redis_port=args.redis_port,
            redis_db=args.redis_db,
            redis_key_prefix=args.redis_key_prefix,
            redis_ttl_seconds=args.redis_ttl_seconds,
            flush_redis_namespace=args.flush_redis_namespace,
            repeat_queries=args.repeat_queries,
            warmup_queries=args.warmup_queries,
            workload_mode=args.workload_mode,
            batch_size=args.batch_size,
            multi_turn_length=args.multi_turn_length,
            routing_switch_at=args.routing_switch_at,
            reranker_model=reranker_model,
            reranker_device=args.reranker_device,
            truncate_prompt_tokens=args.truncate_prompt_tokens,
            max_context_chars=args.max_context_chars,
            max_context_docs=args.max_context_docs,
            sharding_policy=args.sharding_policy,
            # Speculative decoding
            speculative_model=args.speculative_model,
            num_speculative_tokens=args.num_speculative_tokens,
            speculative_method=args.speculative_method,
            baseline_label=args.baseline_label,
        )

    def _run_trials(top_k_value: int) -> None:
        """Run multiple trials and aggregate results."""
        if args.num_trials == 1:
            # Single trial - run normally
            _run_with_top_k(top_k_value)
            return
        
        # Multiple trials - collect results and compute statistics
        print(f"\n{'='*60}")
        print(f"Running {args.num_trials} independent trials for statistical rigor")
        print(f"{'='*60}\n")
        
        trial_results = []
        
        for trial in range(1, args.num_trials + 1):
            print(f"\n--- Trial {trial}/{args.num_trials} (seed={args.seed + trial - 1}) ---\n")
            
            # Create trial-specific output directory
            trial_output_dir = os.path.join(args.output_dir, f"trial_{trial}")
            
            # Run with different seed for each trial
            run_experiment(
                baseline=args.baseline,
                model=args.model,
                dataset=args.dataset,
                num_queries=args.num_queries,
                max_tokens=args.max_tokens,
                api_base=args.api_base,
                use_offline=args.offline,
                output_dir=trial_output_dir,
                seed=args.seed + trial - 1,  # Different seed per trial
                backend=args.backend,
                top_k=top_k_value,
                embedding_model=embedding_model,
                ir_index_dir=args.ir_index_dir,
                rebuild_ir_index=args.rebuild_ir_index if trial == 1 else False,
                redis_host=args.redis_host,
                redis_port=args.redis_port,
                redis_db=args.redis_db,
                redis_key_prefix=args.redis_key_prefix,
                redis_ttl_seconds=args.redis_ttl_seconds,
                flush_redis_namespace=args.flush_redis_namespace,
                repeat_queries=args.repeat_queries,
                warmup_queries=args.warmup_queries,
                workload_mode=args.workload_mode,
                batch_size=args.batch_size,
                multi_turn_length=args.multi_turn_length,
                routing_switch_at=args.routing_switch_at,
                reranker_model=reranker_model,
                reranker_device=args.reranker_device,
                truncate_prompt_tokens=args.truncate_prompt_tokens,
                max_context_chars=args.max_context_chars,
                max_context_docs=args.max_context_docs,
                sharding_policy=args.sharding_policy,
                speculative_model=args.speculative_model,
                num_speculative_tokens=args.num_speculative_tokens,
                speculative_method=args.speculative_method,
                baseline_label=args.baseline_label,
            )
            
            # Load trial results
            metrics_file = os.path.join(trial_output_dir, "metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file) as f:
                    trial_results.append(json.load(f))
        
        # Aggregate results across trials
        if trial_results:
            _aggregate_trial_results(trial_results, args.output_dir, args.num_trials)
    
    def _aggregate_trial_results(trial_results: list, output_dir: str, num_trials: int) -> None:
        """Compute mean ± std across trials and save aggregated results."""
        import numpy as np
        
        print(f"\n{'='*60}")
        print(f"Aggregating results from {num_trials} trials")
        print(f"{'='*60}\n")
        
        # Collect performance metrics across trials
        perf_keys = [
            "queries_per_second", "tokens_per_second",
            "avg_ttft_ms", "p50_ttft_ms", "p95_ttft_ms", "p99_ttft_ms",
            "avg_tpot_ms", "p50_tpot_ms", "p95_tpot_ms", "p99_tpot_ms",
            "avg_latency_ms", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms",
        ]
        
        quality_keys = [
            "grounding_score", "hallucination_detected", "hallucinated_span_ratio",
            "faithfulness", "supported_claim_ratio",
            "context_relevance", "relevance",
            "completeness_bertscore", "completeness_rouge_l",
            "f1_score", "precision", "recall", "exact_match",
        ]

        def aggregate_numeric_section(section_name: str) -> Dict[str, Any]:
            numeric_values: Dict[str, List[float]] = {}
            passthrough: Dict[str, Any] = {}
            for result in trial_results:
                section = result.get(section_name) or {}
                if not isinstance(section, dict):
                    continue
                for key, value in section.items():
                    if isinstance(value, bool):
                        continue
                    if isinstance(value, (int, float)):
                        numeric_values.setdefault(key, []).append(float(value))
                    elif key not in passthrough:
                        passthrough[key] = value

            aggregated_section: Dict[str, Any] = {}
            for key, values in numeric_values.items():
                aggregated_section[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "values": values,
                }
            for key, value in passthrough.items():
                if key not in aggregated_section:
                    aggregated_section[key] = value
            return aggregated_section
        
        aggregated = {
            "num_trials": num_trials,
            "performance": {},
            "quality": {},
            "cache_telemetry": {},
            "retrieval": {},
            "prompt_cache": {},
        }
        
        # Aggregate performance metrics
        for key in perf_keys:
            values = []
            for result in trial_results:
                if "performance" in result and key in result["performance"]:
                    values.append(result["performance"][key])
            
            if values:
                aggregated["performance"][key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "values": values,
                }
        
        # Aggregate quality metrics
        for key in quality_keys:
            values = []
            for result in trial_results:
                if (
                    "quality" in result
                    and key in result["quality"]
                    and result["quality"][key] is not None
                ):
                    values.append(result["quality"][key])
            
            if values:
                aggregated["quality"][key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "values": values,
                }
        
        # Copy experiment config from first trial
        if trial_results:
            aggregated["experiment"] = trial_results[0].get("experiment", {})
            aggregated["baseline_config"] = trial_results[0].get("baseline_config", {})
            aggregated["workload"] = trial_results[0].get("workload", {})
            aggregated["warmup"] = trial_results[0].get("warmup", {})
            aggregated["distributed"] = trial_results[0].get("distributed", {})
            aggregated["repeat_passes"] = trial_results[0].get("repeat_passes", {})

        aggregated["cache_telemetry"] = aggregate_numeric_section("cache_telemetry")
        aggregated["retrieval"] = aggregate_numeric_section("retrieval")
        aggregated["prompt_cache"] = aggregate_numeric_section("prompt_cache")
        
        # Save aggregated results
        os.makedirs(output_dir, exist_ok=True)
        aggregated_file = os.path.join(output_dir, "aggregated_metrics.json")
        with open(aggregated_file, "w") as f:
            json.dump(aggregated, f, indent=2)
        
        print(f"Aggregated results saved to: {aggregated_file}")
        
        # Print summary
        print("\n--- Performance Summary (mean ± std) ---")
        for key in ["avg_ttft_ms", "avg_tpot_ms", "avg_latency_ms", "queries_per_second"]:
            if key in aggregated["performance"]:
                m = aggregated["performance"][key]
                print(f"  {key}: {m['mean']:.2f} ± {m['std']:.2f}")
        
        print("\n--- Quality Summary (mean ± std) ---")
        for key in ["f1_score", "exact_match", "faithfulness", "relevance"]:
            if key in aggregated["quality"]:
                m = aggregated["quality"][key]
                print(f"  {key}: {m['mean']:.4f} ± {m['std']:.4f}")

    # Run experiment
    try:
        if args.top_k_sweep and args.baseline in {"rag", "redis", "hybrid"}:
            for k in parse_top_k_values(args.top_k_values):
                _run_trials(k)
        else:
            _run_trials(args.top_k)
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError running experiment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
