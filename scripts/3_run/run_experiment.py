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
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
    IRDocument,
    IRHit,
    build_corpus_from_contexts,
    ensure_ir_index,
    default_index_dir,
    retrieval_hit_rate,
    retrieval_rank_of_gold,
    stable_text_id,
    CrossEncoderReranker,
)
from src.orchestration.redis_cache import RedisConfig, RedisClient, RetrievalCache
from src.evaluation.staleness import staleness_metrics, select_stale, make_stale_context
from src.utils.prompting import (
    format_qa_prompt,
    format_multi_turn_prompt,
    format_qa_messages,
    format_multi_turn_messages,
    messages_to_fallback_prompt,
    prompt_mode,
    select_distractor_texts,
)
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
    repo_root = Path(__file__).resolve().parents[2]
    return _safe_run(["git", "rev-parse", "HEAD"], cwd=repo_root)


def capture_git_metadata() -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
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
    # Context source equalization (removes the gold-vs-retrieved confound):
    #   "auto"      = current behavior (CAG arms=gold, RAG arms=retrieved)
    #   "gold"      = ALL baselines use the gold passage (isolates caching/serving)
    #   "retrieved" = ALL baselines use retrieved docs (fair to RAG)
    context_source: str = "auto",
    # Compression axis overrides (None = use the baseline's own default)
    compress_method: Optional[str] = None,
    compress_ratio: Optional[float] = None,
    kv_cache_dtype: Optional[str] = None,
    # vLLM serving telemetry via cage-stats (one-shot snapshot + dashboard)
    vllm_telemetry: bool = False,
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
    # CLI compression overrides (only when explicitly provided, to keep per-baseline defaults).
    if compress_method is not None:
        baseline_config.compress_method = None if compress_method == "none" else compress_method
    if compress_ratio is not None:
        baseline_config.compress_target_ratio = compress_ratio
    if kv_cache_dtype is not None:
        baseline_config.kv_cache_dtype = None if kv_cache_dtype == "none" else kv_cache_dtype
    # Staleness/freshness baseline: the serving path is now WIRED. The v0 (stale) evidence is
    # generated on the fly from the gold context (answer redacted), so no separate corpus
    # artifact is needed. stale_fraction is swept via the CAGE_STALE_FRACTION env var (falls
    # back to the preset default 0.0). NOTE: run a live 5-query smoke pass before a full sweep
    # (validate-before-run). See Documentation/STALENESS_BASELINE_DESIGN.md.
    if baseline_config.baseline_type.value == "staleness":
        # TTL mode is NOT IMPLEMENTED. The only wired staleness path is the deterministic
        # version sweep (stale_fraction -> v0/v1). Fail loud rather than silently ignoring a
        # ttl request and running version mode instead (a silent-mismatch validity bug).
        if baseline_config.stale_evidence_mode == "ttl" or baseline_config.cache_ttl_seconds is not None:
            raise NotImplementedError(
                "staleness: stale_evidence_mode='ttl' / cache_ttl_seconds is not implemented. "
                "The wired path is the deterministic version sweep driven by stale_fraction "
                "(CAGE_STALE_FRACTION). Use stale_evidence_mode='version'."
            )
        _sf = os.getenv("CAGE_STALE_FRACTION")
        if _sf is not None and _sf.strip() != "":
            baseline_config.stale_fraction = float(_sf)
        print(f"[staleness] serving path active: stale_fraction={baseline_config.stale_fraction} "
              f"(evidence_mode={baseline_config.stale_evidence_mode}); sweep via CAGE_STALE_FRACTION")
    print(f"\nBaseline config: {baseline_config.description}")
    
    # Validate speculative baseline requirements
    if baseline == "speculative":
        print(
            "Note: speculative decoding is a vLLM LAUNCH-time setting, not a per-request "
            "parameter. Enable it by (re)starting the server with the VLLM_SPECULATIVE_CONFIG "
            "JSON (see scripts/3_run/run_speculative_matrix.sh and scripts/2_serving/manage_vllm_server.sh). Acceptance "
            "rate is captured from /metrics via --vllm-telemetry; this baseline measures the "
            "resulting TTFT/TPOT/throughput, and since speculative decoding is output-preserving "
            "it does not change answer quality."
        )
    
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
    
    # Load the measured query set. It is IDENTICAL across all baselines (the same num_queries
    # examples under the same seed), so every baseline shares the same example_ids and the
    # per-query Wilcoxon can PAIR them against the reference. Warm baselines (warmup_queries > 0)
    # prime the caches by running this same measured set once as a discarded warmup pass. This
    # keeps the comparison paired (the Phase-2 warm hybrid fell back to unpaired Mann-Whitney
    # because it measured a disjoint, shifted slice) and reflects the realistic repeated-query
    # warm-cache scenario, where a warm cache helps precisely when the same prompts recur. The
    # warmup count is treated as a flag: when positive, the full measured set is warmed.
    print(f"\nLoading dataset '{dataset}'...")
    loader = get_loader(dataset, split=default_dataset_split(dataset), seed=seed)
    # Uniform-yardstick manifest (2026-07-15): when CAGE_QUERY_MANIFEST is set, the
    # measured query set comes from ONE auditable, pre-drawn artifact shared by every
    # cell/engine/model (scripts/1_setup/build_query_manifest.py), so per-query pairing
    # holds universally and no script can drift to its own sample. The per-run seeded
    # sampling below remains the non-manifest path.
    _manifest_path = os.getenv("CAGE_QUERY_MANIFEST", "").strip()
    _manifest = None
    if _manifest_path:
        from src.data.manifest import select_examples as _manifest_select
        _manifest = json.loads(Path(_manifest_path).read_text(encoding="utf-8"))
        if _manifest.get("dataset") and _manifest["dataset"] != dataset:
            raise ValueError(
                f"manifest is for dataset '{_manifest['dataset']}' but this run uses "
                f"'{dataset}' -- refusing to serve a mismatched yardstick"
            )
        _trial_no = int(os.getenv("CAGE_MANIFEST_TRIAL", "1") or "1")
        pool = loader.load(max_examples=None)  # full split; selection is by id
        base_examples = _manifest_select(_manifest, _trial_no, pool)
        print(f"MANIFEST workload: trial {_trial_no}, {len(base_examples)} queries "
              f"from {_manifest_path} (pool={_manifest['stats']['pool_size']}, "
              f"blocks={_manifest['stats']['n_blocks']})")
    else:
        pool = loader.load(max_examples=num_queries)
        base_examples = pool

    # cag_true corpus-as-prefix mode (2026-07-15, tasks #71/#82): pack gold paragraphs into
    # ONE shared corpus block and serve it as every query's context, so all prompts share a
    # long identical prefix -- the true-CAG layout (Chan et al., arXiv 2412.15605) that
    # vLLM's prefix cache can actually reuse. (Single-workload SQuAD shares only the
    # ~32-token system prefix across queries -> the honest -3.3% TTFT; this mode is the
    # arm that measures the CAG mechanism itself.) Examples whose gold paragraph did not
    # fit the budget are DROPPED (announced): cag_true cells answer in-corpus questions.
    _corpus_budget = int(os.getenv("CAGE_CORPUS_PREFIX_BUDGET", "0") or "0")
    if _corpus_budget > 0 and _manifest is not None:
        # Manifest mode: every query is in-corpus BY CONSTRUCTION (corpus-first
        # sampling), each using its manifest-assigned block. Queries run in block order
        # so each block's KV stays resident while its questions run (true CAG per
        # block within the L4's capacity); the ordering is identical in the paired
        # cache-off cell, so the pair stays clean.
        _blocks = _manifest["blocks"]
        _q2b = _manifest["question_to_block"]
        base_examples = [
            CAGExample(
                id=ex.id, question=ex.question,
                context=[_blocks[_q2b[ex.id]]["text"]], answer=ex.answer,
                metadata={**(ex.metadata or {}),
                          "corpus_prefix": True,
                          "corpus_block": _q2b[ex.id],
                          "corpus_tokens": _blocks[_q2b[ex.id]]["token_count"],
                          "gold_context": (ex.context or [None])[0]},
            )
            for ex in sorted(base_examples, key=lambda e: _q2b[e.id])
        ]
        print(f"CORPUS-PREFIX mode (manifest): {len(base_examples)} queries over "
              f"{len(_blocks)} blocks, block-ordered")
    elif _corpus_budget > 0:
        from src.data.corpus import build_corpus_block
        _block = build_corpus_block(base_examples, token_budget=_corpus_budget)
        _in_corpus = set(_block.example_ids)
        _n_dropped = sum(1 for ex in base_examples if ex.id not in _in_corpus)
        base_examples = [
            CAGExample(
                id=ex.id, question=ex.question, context=[_block.text], answer=ex.answer,
                metadata={**(ex.metadata or {}),
                          "corpus_prefix": True,
                          "corpus_tokens": _block.token_count,
                          "gold_context": (ex.context or [None])[0]},
            )
            for ex in base_examples if ex.id in _in_corpus
        ]
        print(f"CORPUS-PREFIX mode: block={_block.token_count} tokens, "
              f"{len(_block.paragraphs)} paragraphs; measuring {len(base_examples)} "
              f"in-corpus queries (dropped {_n_dropped} out-of-corpus)")
        if not base_examples:
            raise ValueError("corpus-prefix budget too small: no example's context fits")

    # Doc-grouped ordering (prefix_cache_grouped cell): same-paragraph questions become
    # consecutive, so their shared [system prefix][paragraph] prompt prefix is cache-hot
    # for queries 2..k of each group -- a realistic shared-document serving workload
    # (TurboRAG/CacheWeaver setting) with zero prompt-layout change. Deterministic.
    if os.getenv("CAGE_ORDER_BY_CONTEXT", "0").strip() == "1":
        import hashlib as _hashlib
        base_examples = sorted(
            base_examples,
            key=lambda ex: _hashlib.sha1("||".join(ex.context or []).encode()).hexdigest(),
        )
        print("DOC-GROUPED ordering: examples sorted by context hash (shared-prefix groups)")

    warmup_pool = list(base_examples) if warmup_queries > 0 else []
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
    for idx, ex in enumerate(warmup_pool):  # SAME queries as the measured set; primes the caches
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

    # IR index / retriever (for RAG/Redis/Hybrid baselines, or any baseline when
    # context_source == "retrieved" so gold-context arms can be fed retrieved docs).
    ir_index = None
    corpus_docs = None
    if baseline_config.use_faiss or context_source == "retrieved":
        print("\nBuilding/loading IR index (FAISS)...")
        base_dir = Path(baseline_config.ir_index_dir)
        index_dir = default_index_dir(
            base_dir=base_dir,
            dataset_name=dataset,
            embedding_model=baseline_config.embedding_model,
        )
        corpus_docs = build_corpus_from_contexts(base_examples, dataset_name=dataset)
        # Decision 3B (approved pre-run package, 2026-07-16): widen the retrieval corpus
        # with a deterministic distractor pool so retrieval is a real search problem, not
        # a near-oracle lookup over only the trial's own gold paragraphs. Manifest mode
        # only (the uniform-yardstick path): `pool` above is already the FULL split loaded
        # in stable order, so distractor selection is seed-stable by construction. The
        # first CAGE_DISTRACTOR_DOCS (default 1000) content-deduped paragraphs are added,
        # EXCLUDING the trial's gold paragraphs (both ex.context and the corpus-mode
        # metadata gold_context). 0 disables (old behavior). The IR index content-hash
        # (src/orchestration/ir.py corpus_doc_ids_sha1, checked by ensure_ir_index)
        # triggers the rebuild automatically when the corpus changes.
        _n_distractors = int(os.getenv("CAGE_DISTRACTOR_DOCS", "1000") or "0")
        if _manifest is not None and _n_distractors > 0:
            _gold_texts: List[str] = [
                c for ex in base_examples for c in (ex.context or []) if c
            ]
            _gold_texts += [
                (ex.metadata or {}).get("gold_context")
                for ex in base_examples
                if (ex.metadata or {}).get("gold_context")
            ]
            _distractor_texts = select_distractor_texts(pool, _gold_texts, _n_distractors)
            _existing_ids = {d.doc_id for d in corpus_docs}
            _added = 0
            for _dt in _distractor_texts:
                _did = stable_text_id(_dt)
                if _did in _existing_ids:
                    continue
                corpus_docs.append(
                    IRDocument(
                        doc_id=_did,
                        text=_dt,
                        metadata={"dataset": dataset, "source": "distractor"},
                    )
                )
                _existing_ids.add(_did)
                _added += 1
            print(
                f"DISTRACTOR corpus (Decision 3B): +{_added} distractor paragraphs "
                f"(requested {_n_distractors}, gold excluded, content-deduped)"
            )
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
    # Optional text compressor (compressed_rag baseline / --compress-method)
    context_compressor = None
    if baseline_config.compress_method:
        from src.orchestration.compression import ContextCompressor
        context_compressor = ContextCompressor(method=baseline_config.compress_method, device="cpu")
        print(f"Text compression enabled: {baseline_config.compress_method} "
              f"(keep ratio {baseline_config.compress_target_ratio})")

    # Optional reranker
    reranker = None
    if baseline_config.use_faiss and reranker_model:
        try:
            reranker = CrossEncoderReranker(reranker_model, device=reranker_device)
            print(f"Reranker enabled: {reranker_model}")
        except Exception as e:
            print(f"Warning: failed to initialize reranker {reranker_model}: {e}")

    # Decision 1B (approved pre-run package, 2026-07-16): serve via the model's chat
    # template by default (vLLM /v1/chat/completions with the system message carrying
    # the task+abstention instruction, Qwen3 thinking disabled via
    # chat_template_kwargs {"enable_thinking": false}); CAGE_PROMPT_MODE=raw is the
    # escape hatch that reproduces the legacy raw-completions path byte-for-byte.
    _prompt_mode = prompt_mode()
    print(f"PROMPT MODE: {_prompt_mode} "
          f"({'chat template via /v1/chat/completions' if _prompt_mode == 'chat' else 'legacy raw completions'})")

    # B5 (2026-07-16 audit): a ~35-60ms host-side TTFT constant was traced to host-side
    # scheduling/GC jitter bleeding into the timed window. A short monotonic-clock settle
    # immediately before each timed request lets that jitter drain; the ACTUAL settle
    # duration is recorded per row (settle_ms) so it is auditable, and it is never part
    # of the timed window itself. CAGE_REQUEST_SETTLE_MS=0 disables.
    _settle_ms_cfg = float(os.getenv("CAGE_REQUEST_SETTLE_MS", "150") or "0")

    def settle_before_request() -> float:
        """Sleep the configured settle window; return the measured settle in ms."""
        if _settle_ms_cfg <= 0:
            return 0.0
        _t0 = time.monotonic()
        time.sleep(_settle_ms_cfg / 1000.0)
        return (time.monotonic() - _t0) * 1000.0

    # Setup inference engine
    engine = setup_inference_engine(model, baseline_config, backend=backend, use_offline=use_offline)
    
    # Setup evaluators
    print("\nInitializing evaluators...")
    # Decoupled-scoring mode (2026-07-15): with CAGE_SKIP_QUALITY=1 (or --skip-quality),
    # the serving loop uses a MODEL-FREE evaluator -- F1/EM/abstention are still computed
    # inline (microseconds, no models), while LettuceDetect/NLI/BERTScore/embeddings are
    # skipped so the GPU is never idled by inline CPU scoring (~90% of sweep wall-clock
    # in the 2026-07-15 smoke). Model-based quality is then scored POST-serving from
    # qa_evidence.jsonl: scripts/4_analysis/rescore_quality.py --full --device cuda --apply.
    _skip_quality = os.getenv("CAGE_SKIP_QUALITY", "0").strip() == "1"
    if _skip_quality:
        print("DECOUPLED SCORING: inline model-based quality metrics OFF "
              "(score post-serving via rescore_quality.py --full --apply)")
        quality_evaluator = QualityEvaluator(
            use_nli=False, use_embeddings=False, use_bertscore=False,
            use_rouge=False, use_lettucedetect=False, device="cpu",
        )
    else:
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
        retrieval_rank = None  # 1-based rank of the gold passage among retrieved; None = miss/unused (fix #5-C, MRR)
        retrieval_top1_score = None
        retrieved_doc_ids: List[str] = []
        retrieval_reranked = False
        evidence_version = None    # staleness baseline: "v0" (stale) | "v1" (fresh)
        served_from_cache = None   # staleness baseline: warm-cache hit flag

        # Retrieve when the baseline is retrieval-backed OR when context_source forces
        # retrieved context onto every arm (confound control).
        do_retrieval = (
            baseline_config.baseline_type.value in {"rag", "redis", "hybrid"}
            or context_source == "retrieved"
        )
        if do_retrieval:
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
                # Text fallback: doc-id hashes can diverge from the corpus even when the
                # gold passage IS retrieved (whitespace/encoding), which zeroed the metric
                # in Phase 2. used_contexts here is the retrieved passage text.
                gold_texts=list(example.context or []),
                retrieved_texts=used_contexts,
            )
            # Graded companion (fix #5-C): 1-based rank of the gold passage -> MRR downstream.
            # retrieved_doc_ids / used_contexts are already in retrieval-rank (score) order here,
            # BEFORE the context caps below, so the rank reflects true retrieval position.
            retrieval_rank = retrieval_rank_of_gold(
                gold_doc_ids=gold_doc_ids,
                retrieved_doc_ids=retrieved_doc_ids,
                gold_texts=list(example.context or []),
                retrieved_texts=used_contexts,
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

        # Confound control: force the gold passage onto every arm when requested
        # (retrieval telemetry above is still recorded for the retrieval baselines).
        if context_source == "gold":
            used_contexts = list(example.context or [])

        # Staleness/freshness baseline (GATED): hold the warm served context fixed and make a
        # deterministic fraction of served hits STALE (evidence redacted so it no longer
        # supports the answer). Only entered for the staleness baseline, so it has zero effect
        # on the other nine arms. See Documentation/STALENESS_BASELINE_DESIGN.md.
        if baseline_config.baseline_type.value == "staleness":
            served_from_cache = True  # warm-cache assumption: every query is a served hit
            _fresh = list(example.context or [])
            if select_stale(example.id, baseline_config.stale_fraction, seed):
                evidence_version = "v0"
                used_contexts = make_stale_context(_fresh, example.answer)
            else:
                evidence_version = "v1"
                used_contexts = _fresh

        # Apply context caps before compression so the reported compression ratio is
        # measured against the context actually served, not a pre-truncation superset.
        if max_context_chars is not None and max_context_chars > 0:
            used_contexts = [c[:max_context_chars] for c in used_contexts]
        if max_context_docs is not None and max_context_docs > 0:
            used_contexts = used_contexts[:max_context_docs]

        # gold_position_in_prompt (pre-run package small-logging field): 0-based index of
        # the gold doc among the SERVED contexts (prompt order), -1 if absent. Computed on
        # the capped, PRE-compression list (compression rewrites text, which would break
        # the match; list positions are what the prompt layout uses). Matching is exact
        # text OR containment (corpus-prefix blocks CONTAIN the gold paragraph).
        _gold_texts = [c for c in (example.context or []) if c]
        _meta_gold = (example.metadata or {}).get("gold_context") if isinstance(example.metadata, dict) else None
        if _meta_gold:
            _gold_texts.append(_meta_gold)
        gold_position_in_prompt = -1
        for _pos, _doc in enumerate(used_contexts):
            if _doc and any(_g == _doc or (_g in _doc) for _g in _gold_texts):
                gold_position_in_prompt = _pos
                break

        # Text compression (compressed_rag): compress the already-capped context.
        # The PRE-compression served docs are preserved as original_contexts -- the
        # qa_evidence.jsonl contract field the offline scorer reads for compression arms.
        compression_stats = None
        original_contexts: Optional[List[str]] = None
        if context_compressor is not None:
            original_contexts = list(used_contexts)
            used_contexts, cstats = context_compressor.compress(
                used_contexts, question=question,
                target_ratio=baseline_config.compress_target_ratio,
            )
            compression_stats = cstats.to_dict()

        return {
            "question": question,
            "baseline_mode": baseline_mode,
            "used_contexts": used_contexts,
            "original_contexts": original_contexts,
            "gold_position_in_prompt": gold_position_in_prompt,
            "retrieval_cached": retrieval_cached,
            "retrieval_hit": retrieval_hit,
            "retrieval_rank": retrieval_rank,
            "retrieval_top1_score": retrieval_top1_score,
            "retrieved_doc_ids": retrieved_doc_ids,
            "retrieval_reranked": retrieval_reranked,
            "compression_stats": compression_stats,
            "evidence_version": evidence_version,
            "served_from_cache": served_from_cache,
        }

    def record_result(
        example: CAGExample,
        meta: Dict[str, Any],
        response: Any,
        *,
        batch_id: int,
        turn_index: int,
        settle_ms: float = 0.0,
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
            # Audit 2026-07-16 M5: official SQuAD v2 F1/EM = max over ALL gold answers;
            # loaders that provide them store the deduplicated list in metadata.
            all_answers=(example.metadata or {}).get("all_answers"),
        )
        _quality_row = quality_metrics.to_dict()

        code_metrics = (
            code_evaluator.evaluate(response.generated_text)
            if code_evaluator is not None
            else None
        )

        # Result-integrity guard: a serving error (generated_text == "") or a degenerate
        # empty answer must NOT enter the quality columns as hard 0.0s, which would depress
        # the PRIMARY grounding/F1 means and the per-query stats as if the model had produced
        # a wrong answer. Null every quality field so such rows are treated as MISSING data,
        # not as a scored zero, both here and downstream.
        _empty_gen = (not response.error) and not (response.generated_text or "").strip()
        if response.error or _empty_gen:
            _quality_row = {k: None for k in _quality_row}
            code_metrics = None

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

        # group_id (pre-run package small-logging field): the manifest corpus-block id.
        # Corpus-prefix runs stamp it into metadata; otherwise resolve via the manifest's
        # question_to_block keyed by the base example id (repeat/warmup suffixes
        # stripped). None outside manifest runs.
        _base_id = example.id.split("__rep")[0].split("__warmup")[0]
        group_id = (
            example.metadata.get("corpus_block") if isinstance(example.metadata, dict) else None
        )
        if group_id is None and _manifest is not None:
            group_id = (_manifest.get("question_to_block") or {}).get(_base_id)

        # Task 5 (logprobs): mean/sum logprob of the GENERATED tokens, attached by the
        # vLLM adapter as plain attributes (None on error rows / non-chat backends).
        # Powers the abstention risk-coverage curves downstream.
        mean_token_logprob = getattr(response, "mean_token_logprob", None)
        sum_token_logprob = getattr(response, "sum_token_logprob", None)

        result = {
            "example_id": example.id,
            "baseline": experiment_label,
            "baseline_family": baseline_config.baseline_type.value,
            "baseline_mode": meta["baseline_mode"],
            "workload_mode": workload_mode,
            "batch_id": batch_id,
            "turn_index": turn_index,
            "repeat_index": repeat_index,
            "group_id": group_id,
            "gold_position_in_prompt": meta.get("gold_position_in_prompt"),
            # B5: measured pre-request settle (monotonic clock), for auditability of the
            # 2026-07-16 ~35-60ms host-side TTFT constant finding. Never inside the
            # timed window.
            "settle_ms": settle_ms,
            "prompt_mode": _prompt_mode,
            "question": question,
            "reference_answer": example.answer,
            "generated_answer": response.generated_text,
            "ttft_ms": response.ttft_ms,
            "latency_ms": response.total_time_ms,
            # Per-query time-per-output-token (decode speed after the first token). Persisted
            # per row -- not just as a run aggregate -- so the speculative arm's discriminating
            # serving metric is testable (statistical_tests.py) and plottable. None when there
            # is <=1 output token or timing is missing.
            "tpot_ms": (
                (response.total_time_ms - response.ttft_ms) / (response.num_tokens - 1)
                if (response.ttft_ms is not None and response.total_time_ms is not None
                    and response.num_tokens and response.num_tokens > 1)
                else None
            ),
            "num_tokens": response.num_tokens,
            "prompt_tokens": response.prompt_tokens,
            "cached_prompt_tokens": response.cached_prompt_tokens,
            "cached_prompt_ratio": cached_ratio,
            "mean_token_logprob": mean_token_logprob,
            "sum_token_logprob": sum_token_logprob,
            "finish_reason": response.finish_reason,
            "error": response.error,
            # Flag degenerate empty answers (e.g. a leading newline under stop=["\n"]) so a
            # systematic empty-output regression is visible, not silently scored as a valid 0.
            "empty_generation": (not response.error) and not (response.generated_text or "").strip(),
            # Staleness baseline fields (None/False for the other arms): whether this query was
            # a served cache hit, its evidence version, and whether the answer was grounded.
            "served_from_cache": meta.get("served_from_cache"),
            "evidence_version": meta.get("evidence_version"),
            # None (not False) when grounding is N/A -- abstentions and unscored rows are
            # MISSING data for this flag, not "ungrounded"; False would mislabel a correct
            # "Don't know." as a grounding failure in the staleness curve.
            "grounded": ((_quality_row.get("grounding_score") or 0.0) >= 0.5
                         if _quality_row.get("grounding_score") is not None else None),
            "kv_transfer_params": kv_transfer_params_str,
            "routed_replica": response.router_replica or "",
            "retrieval_top1_score": meta["retrieval_top1_score"],
            "retrieval_hit": meta["retrieval_hit"],
            "retrieval_rank": meta.get("retrieval_rank"),  # graded rank for MRR (fix #5-C)
            "retrieval_cached": meta["retrieval_cached"],
            "retrieval_reranked": meta["retrieval_reranked"],
            "retrieved_doc_ids": ";".join(meta["retrieved_doc_ids"]) if meta["retrieved_doc_ids"] else "",
            "compression_ratio": (meta.get("compression_stats") or {}).get("compression_ratio"),
            "compression_applied": (meta.get("compression_stats") or {}).get("compression_applied"),
            # Audit fix B3b (2026-07-16): per-query compression CPU time -- LLMLingua-2's own
            # speedup claims INCLUDE it, so e2e comparisons must be able to add it back.
            "compression_latency_ms": (meta.get("compression_stats") or {}).get("compression_latency_ms"),
            **_quality_row,
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

        # Per-query evidence, appended INCREMENTALLY (one JSON line per query) so a mid-trial
        # OOM/SIGKILL -- the exact memory-pressure failure this work studies -- still preserves
        # completed rows, and so an analyst can reconstruct WHY an answer was (un)grounded and,
        # for the staleness arm, WHICH served text was stale. The served context text and the
        # LettuceDetect spans are otherwise computed then dropped (never in results.csv). Lands
        # under output_dir (the per-trial dir) so it is already covered by the GCS sync.
        try:
            os.makedirs(output_dir, exist_ok=True)
            _evidence = {
                "example_id": example.id,
                "baseline": experiment_label,
                "repeat_index": repeat_index,
                "turn_index": turn_index,
                "batch_id": batch_id,
                "group_id": group_id,
                "gold_position_in_prompt": meta.get("gold_position_in_prompt"),
                "prompt_mode": _prompt_mode,
                "question": question,
                "reference_answer": example.answer,
                # ALL gold answers (audit 2026-07-16 M5) so rescore_quality.py can apply
                # the official max-over-golds F1/EM; absent for datasets without the field.
                "all_answers": (example.metadata or {}).get("all_answers"),
                "generated_answer": response.generated_text,
                "used_contexts": used_contexts,
                # CONTRACT field (pre-run package task 6): for compression arms this is
                # the PRE-compression served docs (list[str]); None for the other arms.
                # The offline scorer (rescore_quality.py) consumes this exact field name.
                "original_contexts": meta.get("original_contexts"),
                "mean_token_logprob": mean_token_logprob,
                "sum_token_logprob": sum_token_logprob,
                "served_from_cache": meta.get("served_from_cache"),
                "evidence_version": meta.get("evidence_version"),
                "grounding_score": _quality_row.get("grounding_score"),
                "grounded": result.get("grounded"),
                "hallucinated_spans": getattr(quality_metrics, "hallucinated_spans", None),
                "retrieved_doc_ids": meta.get("retrieved_doc_ids") or [],
            }
            with open(os.path.join(output_dir, "qa_evidence.jsonl"), "a", encoding="utf-8") as _ef:
                _ef.write(json.dumps(_evidence, default=str) + "\n")
        except Exception as _ev_exc:
            print(f"[evidence] could not append qa_evidence.jsonl: {_ev_exc}")

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
                    # Per-query guard (B4): a single failing turn (retrieval / rerank /
                    # compression / metric-eval / OOM) must not abort the whole baseline.
                    try:
                        meta = prepare_example(example)
                        messages = None
                        if _prompt_mode == "chat":
                            # Decision 1B: chat-template serving; system carries the
                            # task+abstention instruction, history becomes proper
                            # alternating turns, current context+question last.
                            messages = format_multi_turn_messages(
                                meta["question"],
                                meta["used_contexts"],
                                history=history,
                            )
                            prompt = messages_to_fallback_prompt(messages)
                        else:
                            prompt = format_multi_turn_prompt(
                                meta["question"],
                                meta["used_contexts"],
                                history=history,
                            )
                        request = InferenceRequest(
                            prompt=prompt,
                            max_tokens=max_tokens,
                            temperature=0.0,
                            top_p=0.95,
                            request_id=example.id,
                            truncate_prompt_tokens=truncate_prompt_tokens,
                            stop=["\n"],
                        )
                        if messages is not None:
                            # Plain attribute consumed by VLLMAdapter -> /v1/chat/completions.
                            request.messages = messages
                        stream_flag = backend in {"vllm", "ollama"}
                        # B5: settle immediately before the TIMED request only.
                        settle_ms = settle_before_request() if collect_results else 0.0
                        response = engine.generate(request, stream=stream_flag)
                        if collect_results:
                            record_result(example, meta, response, batch_id=batch_id,
                                          turn_index=turn_idx, settle_ms=settle_ms)
                        elif response.error:
                            print(f"[warmup] Request {example.id} failed: {response.error}")
                        history.append((meta["question"], response.generated_text or ""))
                    except Exception as _ex:
                        print(f"[{stage_name}] {example.id} failed: {_ex}; skipping this turn")
                    sent_requests += 1
                    stage_processed += 1
                    if collect_results:
                        measured_processed += 1
            else:
                metas: List[Dict[str, Any]] = []
                requests: List[InferenceRequest] = []
                kept: List[CAGExample] = []  # examples whose prepare_example() succeeded
                for example in unit:
                    maybe_reshuffle_router(sent_requests)
                    # Per-query guard (B4): a failed prepare (retrieval / rerank /
                    # compression) skips just this example instead of aborting the baseline.
                    try:
                        meta = prepare_example(example)
                        messages = None
                        if _prompt_mode == "chat":
                            # Decision 1B: chat-template serving; context first,
                            # question last inside the user message (prefix sharing
                            # preserved), abstention instruction in the system message.
                            messages = format_qa_messages(meta["question"], meta["used_contexts"])
                            prompt = messages_to_fallback_prompt(messages)
                        else:
                            prompt = format_qa_prompt(meta["question"], meta["used_contexts"])
                        request = InferenceRequest(
                            prompt=prompt,
                            max_tokens=max_tokens,
                            temperature=0.0,
                            top_p=0.95,
                            request_id=example.id,
                            truncate_prompt_tokens=truncate_prompt_tokens,
                            stop=["\n"],
                        )
                        if messages is not None:
                            # Plain attribute consumed by VLLMAdapter -> /v1/chat/completions.
                            request.messages = messages
                    except Exception as _ex:
                        print(f"[{stage_name}] prepare failed for {example.id}: {_ex}; skipping")
                        continue
                    metas.append(meta)
                    requests.append(request)
                    kept.append(example)

                if not requests:
                    continue  # whole unit failed to prepare; nothing to send

                # B5: settle immediately before the TIMED request/batch only. For a
                # batched unit the single settle precedes the batch send; each of the
                # unit's rows records the same measured value.
                settle_ms = settle_before_request() if collect_results else 0.0
                if len(requests) == 1:
                    stream_flag = backend in {"vllm", "ollama"}
                    responses = [engine.generate(requests[0], stream=stream_flag)]
                else:
                    responses = engine.batch_generate(requests)

                for example, meta, response in zip(kept, metas, responses):
                    # Per-query guard (B4): a failed record (metric-eval / OOM) drops one
                    # row, not the whole baseline's already-collected results.
                    try:
                        if collect_results:
                            record_result(example, meta, response, batch_id=batch_id,
                                          turn_index=0, settle_ms=settle_ms)
                        elif response.error:
                            print(f"[warmup] Request {example.id} failed: {response.error}")
                    except Exception as _ex:
                        print(f"[{stage_name}] record failed for {example.id}: {_ex}; skipping row")
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

    # GPU telemetry (Phase-2+); no-op on non-NVIDIA hosts.
    from src.evaluation.performance import GPUMetricsTracker
    gpu_tracker = GPUMetricsTracker()
    gpu_monitoring = gpu_tracker.start_monitoring()

    # cage-stats serving telemetry sampled DURING the workload (not one-shot at idle),
    # so throughput / KV-usage / prefix-hit reflect the ACTIVE run.
    vllm_sampler = None
    if vllm_telemetry:
        try:
            from src.monitoring.vllm_telemetry import VllmTelemetrySampler
            # Start the sampler whenever telemetry is requested -- NOT gated on cage-stats.
            # capture_snapshot() falls back to a dependency-free /metrics scraper, so
            # speculative-decode acceptance is sampled even when cage-stats is absent
            # (Phase-2 gap: the cage-stats gate skipped the scraper -> acceptance was None).
            vllm_sampler = VllmTelemetrySampler(api_base, interval=1.0).start()
        except Exception as e:
            print(f"[telemetry] sampler not started: {e}")

    performance_evaluator.start()
    try:
        execute_work_units(work_units, collect_results=True, stage_name="Measured")
    except Exception:
        # A crash escaped the per-query guards (e.g. the server died mid-stage). Persist
        # whatever rows were already collected so the baseline is not lost, then re-raise.
        try:
            os.makedirs(output_dir, exist_ok=True)
            _pp = os.path.join(output_dir, "results.partial.csv")
            if results:
                with open(_pp, "w", newline="") as _pf:
                    # Union of keys across ALL rows (not row 0): quality.to_dict() drops
                    # hallucination_detected when None, so a narrow row-0 would otherwise
                    # make DictWriter raise mid-write and lose the partial flush too.
                    _w = csv.DictWriter(_pf, fieldnames=list(dict.fromkeys(k for r in results for k in r)))
                    _w.writeheader()
                    _w.writerows(results)
                print(f"[recovery] stage crashed; flushed {len(results)} partial rows -> {_pp}")
        except Exception as _flush_exc:
            print(f"[recovery] failed to flush partial results: {_flush_exc}")
        raise
    finally:
        performance_evaluator.stop()

    if gpu_monitoring:
        gpu_tracker.stop_monitoring()
    if vllm_sampler is not None:
        vllm_sampler.stop()

    print("-" * 70)
    print("Experiment complete!")

    # Compute aggregate metrics
    perf_metrics = performance_evaluator.compute_metrics()
    cache_metrics = cache_tracker.get_metrics()
    gpu_metrics = gpu_tracker.compute_metrics().to_dict() if gpu_monitoring else None

    # Optional vLLM serving telemetry via cage-stats — captures what CAGE's own
    # metrics don't expose (spec-decode acceptance, KV-compression ratio/dtype,
    # token-source breakdown, prefix-cache hit, multi-vendor GPU) and prints a
    # one-shot dashboard. See src/monitoring/vllm_telemetry.py.
    vllm_telemetry_snapshot = None
    if vllm_telemetry:
        try:
            from src.monitoring.vllm_telemetry import available, capture, dashboard_text, scrape_spec_decode
            # Prefer the workload-sampled aggregate (works even without cage-stats: the
            # sampler falls back to the stdlib /metrics scraper for spec-decode acceptance).
            if vllm_sampler is not None:
                vllm_telemetry_snapshot = vllm_sampler.aggregate()
                # Full per-tick time series next to vllm_telemetry.json: the aggregate
                # collapses trajectories (eviction onset, KV shrink) that the
                # memory-pressure sweep reads offline.
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    _series_path = os.path.join(output_dir, "telemetry_series.jsonl")
                    if vllm_sampler.save_series(_series_path):
                        print(f"[telemetry] series saved -> {_series_path}")
                except Exception as _series_exc:
                    print(f"[telemetry] series save failed: {_series_exc}")
            # If cage-stats is present, enrich with a one-shot snapshot + print the dashboard.
            if available():
                if vllm_telemetry_snapshot is None:
                    vllm_telemetry_snapshot, _ = capture(api_base)
                _dash = dashboard_text(api_base)
                if _dash:
                    print("\n" + "=" * 70)
                    print("vLLM TELEMETRY (cage-stats)")
                    print("=" * 70)
                    print(_dash)
            # Dependency-free backstop: ALWAYS ensure spec-decode acceptance is captured from
            # /metrics, even when cage-stats is absent or sampling missed the spec counters.
            if (
                vllm_telemetry_snapshot is None
                or vllm_telemetry_snapshot.get("spec_decode_acceptance_rate") is None
            ):
                _spec = scrape_spec_decode(api_base)
                if _spec:
                    vllm_telemetry_snapshot = vllm_telemetry_snapshot or {}
                    vllm_telemetry_snapshot.setdefault("spec_decode", _spec)
                    if _spec.get("spec_decode_acceptance_rate") is not None:
                        vllm_telemetry_snapshot["spec_decode_acceptance_rate"] = _spec["spec_decode_acceptance_rate"]
            # A speculative baseline with no acceptance number is a SILENT no-op (server
            # soft-accepted the config but speculation never engaged, or /metrics lacks the
            # vllm:spec_decode_* counters). A stdout WARNING is easy to miss across a 500x3
            # sweep, so ALSO persist a STATUS sentinel next to the results -- the same
            # convention run_speculative_matrix.sh uses -- so the cell reads as DEGRADED,
            # not silently DONE, in the stats consolidation.
            if baseline_config.baseline_type.value == "speculative":
                _acc = (vllm_telemetry_snapshot or {}).get("spec_decode_acceptance_rate")
                if _acc is None:
                    print("[telemetry] WARNING: speculative baseline but spec_decode_acceptance_rate "
                          "is None -- speculation may not have engaged, or /metrics lacks "
                          "vllm:spec_decode_* counters. Marking this cell DEGRADED.")
                    try:
                        # Write to the CELL root (parent of a trial_N subdir) so run_phase2_stats.sh,
                        # which reads STATUS at the cell top-level, actually sees the degraded flag.
                        _sent_dir = output_dir
                        if os.path.basename(os.path.normpath(output_dir)).startswith("trial_"):
                            _sent_dir = os.path.dirname(os.path.normpath(output_dir))
                        os.makedirs(_sent_dir, exist_ok=True)
                        _cell = os.path.basename(os.path.normpath(_sent_dir))
                        with open(os.path.join(_sent_dir, "STATUS"), "w") as _sf:
                            _sf.write(f"STATUS=degraded reason=no_spec_acceptance cell={_cell}\n")
                    except Exception as _se:
                        print(f"[telemetry] could not write DEGRADED sentinel: {_se}")
            # Save the snapshot as an explicit JSON artifact alongside the results.
            if vllm_telemetry_snapshot is not None:
                os.makedirs(output_dir, exist_ok=True)
                _tpath = os.path.join(output_dir, "vllm_telemetry.json")
                with open(_tpath, "w") as _tf:
                    json.dump(vllm_telemetry_snapshot, _tf, indent=2, default=str)
                print(f"[telemetry] saved -> {_tpath}")
            elif not available():
                print("[telemetry] cage-stats unavailable and no /metrics spec-decode data "
                      "(pip install -e <cage-stats> or set CAGE_STATS_HOME for full telemetry).")
        except Exception as e:
            print(f"[telemetry] skipped: {e}")

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
    
    # Only clear completeness_bertscore globally when the BERTScore MODEL was unavailable,
    # signalled by a None on a row that HAD a non-empty reference (an answerable item the
    # model should have scored). Do NOT clear merely because unanswerable rows (empty
    # reference, e.g. ~52% of SQuAD v2) legitimately return None -- that would null out the
    # valid scores on every answerable row. mean_or_none already excludes the None rows.
    if any(
        r.get("completeness_bertscore") is None and (r.get("reference_answer") or "").strip()
        and not r.get("error") and not r.get("empty_generation")
        for r in results
    ):
        print("Warning: BERTScore was unavailable for answerable rows; clearing completeness_bertscore for all rows.")
        for row in results:
            row["completeness_bertscore"] = None

    # Surface a systematic empty-generation problem loudly (degenerate answers scored as 0).
    _empty = sum(1 for r in results if r.get("empty_generation"))
    if results and _empty / len(results) > 0.05:
        print(f"[quality] WARNING: {_empty}/{len(results)} ({100.0 * _empty / len(results):.1f}%) "
              "generations were EMPTY; their quality scores are degenerate. Check the stop "
              "sequence / prompt (a leading newline under stop=['\\n'] is the usual cause).")
    # Compute average quality metrics
    import numpy as np

    def mean_or_none(metric_key: str) -> Optional[float]:
        # Exclude errored and degenerate-empty rows: their quality fields are already
        # nulled at record time, but guard here too so a stray scored 0.0 can never enter
        # a PRIMARY quality mean.
        values = [
            r[metric_key] for r in results
            if r.get(metric_key) is not None and not r.get("error") and not r.get("empty_generation")
        ]
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
        # SQuAD v2 no-answer decomposition (fix #4). answerable-only variants average over
        # answerable items only (None elsewhere); no_answer_correct averages over no-answer
        # items only (== abstention accuracy); is_answerable/predicted_no_answer are the
        # dataset/model abstention rates. mean_or_none already drops None, so each key is
        # computed over exactly the right subset with no manual filtering.
        "f1_answerable": mean_or_none("f1_answerable"),
        "exact_match_answerable": mean_or_none("exact_match_answerable"),
        "no_answer_correct": mean_or_none("no_answer_correct"),
        "is_answerable": mean_or_none("is_answerable"),
        "predicted_no_answer": mean_or_none("predicted_no_answer"),
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
    # SQuAD v2 no-answer view (fix #4): answerable-only F1/EM isolate extraction quality;
    # no_answer_correct is abstention accuracy on unanswerable items. Lines are n/a for
    # datasets without unanswerable questions (NQ/MuSiQue) -- expected, not an error.
    print(f"F1 / EM (answerable-only): {format_metric(avg_quality['f1_answerable'])} / "
          f"{format_metric(avg_quality['exact_match_answerable'])}")
    print(f"Abstention accuracy (no-answer items): {format_metric(avg_quality['no_answer_correct'])} "
          f"| answerable frac: {format_metric(avg_quality['is_answerable'])} "
          f"| model abstain rate: {format_metric(avg_quality['predicted_no_answer'])}")
    print(f"Completeness (BERTScore): {format_metric(avg_quality['completeness_bertscore'])}")
    print(f"Completeness (ROUGE-L): {format_metric(avg_quality['completeness_rouge_l'])}")

    def summarize_subset(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not rows:
            return None
        # Exclude errored rows from the throughput/latency sums so this matches the primary
        # PerformanceEvaluator (which computes serving_time over successful requests only).
        good = [r for r in rows if not r.get("error")]
        latencies = [r["latency_ms"] for r in good if r.get("latency_ms") is not None]
        ttfts = [r["ttft_ms"] for r in good if r.get("ttft_ms") is not None]
        tokens = [r["num_tokens"] for r in good if r.get("num_tokens") is not None]
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
        # Mean Reciprocal Rank (fix #5-C): graded retrieval quality that discriminates where the
        # lenient hit@k saturates at 1.0. A miss (retrieval_rank None/0) contributes reciprocal 0.
        avg_mrr = float(
            np.mean(
                [
                    (1.0 / r["retrieval_rank"]) if r.get("retrieval_rank") else 0.0
                    for r in retrieval_rows
                ]
            )
        )

        retrieval_summary = {
            "avg_hit": avg_retrieval_hit,
            "avg_mrr": avg_mrr,
            "avg_top1_score": avg_top1,
            "cache_rate": cache_rate,
            "top_k": baseline_config.top_k_retrieval,
            "embedding_model": baseline_config.embedding_model,
            "reranker_model": baseline_config.reranker_model,
        }

        print("\n" + "=" * 70)
        print("RETRIEVAL METRICS (Average)")
        print("=" * 70)
        print(f"Hit@{baseline_config.top_k_retrieval} (lenient coverage): {avg_retrieval_hit:.3f}")
        print(f"MRR (graded): {avg_mrr:.3f}")
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
    
    # Save detailed results as CSV. Header = UNION of keys across ALL rows (not row 0's):
    # quality.to_dict() omits hallucination_detected when it is None, so a narrow row-0
    # (empty/errored/retrieval-miss query) followed by a full row would otherwise make
    # DictWriter (default extrasaction='raise') throw mid-write and corrupt the whole
    # trial's results.csv -- a silent whole-trial data loss on the primary grounding column.
    _fieldnames = list(dict.fromkeys(k for r in results for k in r)) if results else []
    with open(results_file, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=_fieldnames)
            writer.writeheader()
            writer.writerows(results)

    with open(stable_results_file, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=_fieldnames)
            writer.writeheader()
            writer.writerows(results)
    
    print(f"\nResults saved to: {results_file}")
    
    # Analytical KV-cache footprint estimate (compression axis): bridge the model
    # architecture to kv_cache_bytes() so fp8 (compressed_cag) can be compared against a
    # bf16 baseline. The empirical footprint still comes from GPUMetricsTracker; this is
    # the analytical estimate the dissertation describes. Never raises.
    try:
        from src.evaluation.compression import analytical_kv_footprint
        _kv_dtype = baseline_config.kv_cache_dtype or "bf16"
        _avg_prompt_toks = (cache_summary or {}).get("avg_prompt_tokens")
        compression_analytical = (
            analytical_kv_footprint(model, int(_avg_prompt_toks), dtype=_kv_dtype)
            if _avg_prompt_toks else None
        )
    except Exception:
        compression_analytical = None

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
            "context_source": context_source,
            "backend": backend,
            "api_base": api_base,
            # Pre-run package provenance (2026-07-16): chat-template serving mode
            # (Decision 1B; "raw" = legacy escape hatch), B5 settle window, and the
            # Decision 3B distractor-corpus knob as resolved for this run.
            "prompt_mode": _prompt_mode,
            "request_settle_ms": _settle_ms_cfg,
            "distractor_docs": int(os.getenv("CAGE_DISTRACTOR_DOCS", "1000") or "0"),
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
        "gpu": gpu_metrics,
        "vllm_telemetry": vllm_telemetry_snapshot,
        "cache_telemetry": cache_metrics,
        "distributed": distributed_summary,
        "quality": avg_quality,
        "staleness": (staleness_metrics(results)
                      if baseline_config.baseline_type.value == "staleness" else None),
        "retrieval": retrieval_summary,
        "prompt_cache": cache_summary,
        "compression_analytical": compression_analytical,
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
        choices=["no_cache", "prefix_cache", "redis", "rag", "distributed", "hybrid", "speculative", "compressed_rag", "compressed_cag", "staleness"],
        help="Baseline to evaluate (staleness is a scaffold: see Documentation/STALENESS_BASELINE_DESIGN.md)",
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
        choices=["hotpotqa", "qasper", "squad_v2", "trivia_qa", "natural_questions", "musique", "crag", "sharegpt", "humaneval", "mbpp", "hpc_code"],
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
        default=256,
        help="Maximum tokens to generate per query",
    )
    parser.add_argument(
        "--skip-quality",
        action="store_true",
        help="Decoupled scoring: skip inline model-based quality metrics (grounding/NLI/"
             "BERTScore/relevance); F1/EM/abstention still computed. Score post-serving "
             "with scripts/4_analysis/rescore_quality.py --full --apply. "
             "Equivalent to CAGE_SKIP_QUALITY=1.",
    )
    parser.add_argument(
        "--corpus-prefix-budget",
        type=int,
        default=0,
        help="cag_true mode: pack gold paragraphs into one shared corpus block of at most "
             "this many tokens and serve it as every query's context (true CAG, Chan et "
             "al. 2412.15605). 0 = off. Equivalent to CAGE_CORPUS_PREFIX_BUDGET.",
    )
    parser.add_argument(
        "--order-by-context",
        action="store_true",
        help="Order queries so same-context questions are consecutive (shared-document "
             "workload; prefix_cache_grouped cell). Equivalent to CAGE_ORDER_BY_CONTEXT=1.",
    )
    parser.add_argument(
        "--query-manifest",
        default=None,
        help="Path to the uniform query manifest (build_query_manifest.py). All trials "
             "measure the manifest's pre-drawn query ids -- the fairness yardstick shared "
             "by every cell/engine/model. Equivalent to CAGE_QUERY_MANIFEST.",
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
        default="./results/_adhoc",
        help="Output directory for results (drivers pass this explicitly under the run root; "
        "this default is only for ad-hoc/debug runs and stays inside the ignored results/ tree)",
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
    parser.add_argument(
        "--context-source",
        choices=["auto", "gold", "retrieved"],
        default="auto",
        help=(
            "Context fed to ALL baselines: 'auto' (CAG=gold, RAG=retrieved), "
            "'gold' (everyone gets the gold passage — isolates caching), or "
            "'retrieved' (everyone gets retrieved docs — fair to RAG). "
            "Use gold/retrieved to remove the gold-vs-retrieved confound."
        ),
    )
    # Compression axis
    parser.add_argument(
        "--compress-method",
        choices=["none", "llmlingua2", "llmlingua"],
        default=None,
        help="Text compression of context (compressed_rag). Overrides the baseline default.",
    )
    parser.add_argument(
        "--compress-ratio",
        type=float,
        default=None,
        help="Fraction of tokens to KEEP when compressing (0.5 = 2x compression).",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        choices=["none", "fp8"],
        default=None,
        help="Server-side KV-cache compression for compressed_cag (record-only here; "
             "pass the same to vLLM via --kv-cache-dtype when launching the server).",
    )
    parser.add_argument(
        "--reset-cache-between-trials",
        action="store_true",
        help="Flush the vLLM prefix cache between trials (cold-start-per-trial). "
             "Requires the server started with VLLM_SERVER_DEV_MODE=1.",
    )
    parser.add_argument(
        "--vllm-telemetry",
        action="store_true",
        help="Capture a vLLM /metrics snapshot via cage-stats (spec-decode acceptance, "
             "KV-compression ratio, token-source, GPU) into results + print a dashboard. "
             "Needs cage-stats (pip install -e <repo> or set CAGE_STATS_HOME).",
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
        choices=["draft_model", "ngram", "suffix", "medusa", "eagle", "eagle3", "mimo_mtp", "mlp_speculator"],
        help="Speculative decoding method recorded in the manifest (must match the launched "
             "VLLM_SPECULATIVE_CONFIG method, e.g. eagle3 for Qwen3-8B, mimo_mtp for MiMo).",
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
    if args.skip_quality:
        os.environ["CAGE_SKIP_QUALITY"] = "1"
    if args.corpus_prefix_budget and args.corpus_prefix_budget > 0:
        os.environ["CAGE_CORPUS_PREFIX_BUDGET"] = str(args.corpus_prefix_budget)
    if args.order_by_context:
        os.environ["CAGE_ORDER_BY_CONTEXT"] = "1"
    if args.query_manifest:
        os.environ["CAGE_QUERY_MANIFEST"] = args.query_manifest

    def _reset_prefix_cache(api_base: str) -> None:
        """Flush the vLLM prefix cache (dev-mode endpoint) for cold-start-per-trial."""
        import urllib.request
        url = api_base.rstrip("/") + "/reset_prefix_cache"
        try:
            req = urllib.request.Request(url, method="POST")
            urllib.request.urlopen(req, timeout=10)
            print(f"[cache] reset prefix cache via {url}")
        except Exception as e:
            print(f"[cache] WARNING: could not reset prefix cache ({e}). "
                  f"Start vLLM with VLLM_SERVER_DEV_MODE=1 to enable /reset_prefix_cache.")

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
            context_source=args.context_source,
            compress_method=args.compress_method,
            compress_ratio=args.compress_ratio,
            kv_cache_dtype=args.kv_cache_dtype,
            vllm_telemetry=args.vllm_telemetry,
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
            # Manifest mode reads the trial's pre-drawn query ids by trial NUMBER (the
            # seed offset stays for generation-side reproducibility).
            os.environ["CAGE_MANIFEST_TRIAL"] = str(trial)

            # Cold-start-per-trial: flush the vLLM prefix cache between trials so each
            # trial measures from a known (empty) cache state. Requires the server to be
            # started with VLLM_SERVER_DEV_MODE=1 (enables POST /reset_prefix_cache).
            if args.reset_cache_between_trials and trial > 1:
                _reset_prefix_cache(args.api_base)

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
                context_source=args.context_source,
                compress_method=args.compress_method,
                compress_ratio=args.compress_ratio,
                kv_cache_dtype=args.kv_cache_dtype,
                vllm_telemetry=args.vllm_telemetry,
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
            aggregated["vllm_telemetry"] = trial_results[0].get("vllm_telemetry")

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
