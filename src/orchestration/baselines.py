"""
Baseline configurations for CAGE benchmarking.

Nine measured families (the ``BaselineType`` enum below is the source of truth; keep this
list in sync with it):
1.  no_cache        - full context reprocessing (worst-case control)
2.  prefix_cache    - vLLM native prefix caching (server launched with --enable-prefix-caching)
3.  redis           - Redis-backed RETRIEVAL-ARTIFACT cache (query->doc-ids), NOT a KV cache
4.  rag             - FAISS + SentenceTransformers dense retrieval
5.  distributed     - router-mediated multi-replica routing (replicated = real; sharded_context = simulated transfer)
6.  hybrid          - retrieval-artifact cache + native prefix caching (cold/warm are runtime labels)
7.  speculative     - measures a server LAUNCHED with speculative decoding; runner records TTFT/TPOT + /metrics acceptance
8.  compressed_rag  - RAG with LLMLingua-2 text compression of retrieved docs
9.  compressed_cag  - measures a server LAUNCHED with fp8 KV cache; runner records an analytical footprint

Staleness axis (serving path wired; run a live smoke pass before a full sweep; see
cloud_docs/STALENESS_BASELINE_DESIGN.md):
    staleness       - holds cache warmth fixed and sweeps a stale_fraction knob to plot the
                      serving-win vs grounding-loss curve on CAGE's joint axis.
"""

import importlib.util
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any


class BaselineType(Enum):
    """Supported baseline types."""
    
    NO_CACHE = "no_cache"
    PREFIX_CACHE = "prefix_cache"
    REDIS_CACHE = "redis"
    RAG = "rag"
    DISTRIBUTED_CACHE = "distributed"
    HYBRID = "hybrid"
    SPECULATIVE = "speculative"
    # Compression axis (Option 3 / 2x2: context source x compression)
    COMPRESSED_RAG = "compressed_rag"  # retrieved docs text-compressed (LLMLingua)
    COMPRESSED_CAG = "compressed_cag"  # cached context KV-compressed (fp8 KV / MLA)
    # Staleness/freshness axis: sweeps cache-entry AGE at fixed warmth (gold evidence redacted
    # for the stale fraction). See cloud_docs/STALENESS_BASELINE_DESIGN.md.
    STALE = "staleness"


@dataclass
class BaselineConfig:
    """Configuration for a baseline experiment."""
    
    baseline_type: BaselineType
    description: str
    
    # vLLM server configuration
    enable_prefix_caching: bool = False
    api_base: str = "http://localhost:8000"
    
    # Redis configuration (for redis baseline)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_key_prefix: str = "cage"

    # RAG / IR configuration
    use_faiss: bool = False
    embedding_model: str = "intfloat/e5-large-v2"
    top_k_retrieval: int = 3
    ir_index_dir: str = "./experiments/ir_index"  # where FAISS index + docstore are persisted
    ir_rebuild: bool = False

    # Optional reranker configuration (for retrieval baselines)
    reranker_model: Optional[str] = None
    reranker_top_k: Optional[int] = None
    
    # Hybrid configuration
    cache_threshold: float = 0.8  # Confidence threshold for cache vs RAG
    
    # Speculative decoding configuration
    speculative_model: Optional[str] = None  # Draft model for speculative decoding
    num_speculative_tokens: int = 5  # Number of tokens to speculate
    speculative_method: str = "draft_model"  # draft_model, ngram, suffix, medusa, eagle

    # Compression axis configuration
    compress_method: Optional[str] = None      # text compressor: "llmlingua2" | "llmlingua" | None
    compress_target_ratio: float = 0.5         # fraction of tokens to KEEP (0.5 = 2x compression)
    kv_cache_dtype: Optional[str] = None        # server-side KV compression: "fp8" | None (compressed_cag)

    # Staleness/freshness axis (scaffold; see cloud_docs/STALENESS_BASELINE_DESIGN.md).
    # stale_fraction = fraction of served cache hits deliberately bound to an OUTDATED
    # evidence version; warmth/hit-rate held constant so ONLY cache AGE varies.
    stale_fraction: float = 0.0                 # 0.0 = all-fresh; 1.0 = all-stale
    cache_ttl_seconds: Optional[int] = None     # TTL-mode alternative to stale_fraction
    stale_evidence_mode: str = "version"        # "version" (v0/v1 tag) | "ttl"
    evidence_version_field: str = "evidence_version"

    # Additional metadata
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "baseline_type": self.baseline_type.value,
            "description": self.description,
            "enable_prefix_caching": self.enable_prefix_caching,
            "api_base": self.api_base,
            "redis_host": self.redis_host,
            "redis_port": self.redis_port,
            "redis_db": self.redis_db,
            "redis_key_prefix": self.redis_key_prefix,
            "use_faiss": self.use_faiss,
            "embedding_model": self.embedding_model,
            "top_k_retrieval": self.top_k_retrieval,
            "ir_index_dir": self.ir_index_dir,
            "ir_rebuild": self.ir_rebuild,
            "reranker_model": self.reranker_model,
            "reranker_top_k": self.reranker_top_k,
            "cache_threshold": self.cache_threshold,
            "speculative_model": self.speculative_model,
            "num_speculative_tokens": self.num_speculative_tokens,
            "speculative_method": self.speculative_method,
            "compress_method": self.compress_method,
            "compress_target_ratio": self.compress_target_ratio,
            "kv_cache_dtype": self.kv_cache_dtype,
            "stale_fraction": self.stale_fraction,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "stale_evidence_mode": self.stale_evidence_mode,
            "evidence_version_field": self.evidence_version_field,
            "metadata": self.metadata or {},
        }


def get_baseline_config(baseline_name: str, **overrides) -> BaselineConfig:
    """
    Get predefined baseline configuration.
    
    Args:
        baseline_name: Name of baseline (no_cache, prefix_cache, redis, rag, distributed, hybrid, speculative)
        **overrides: Override specific config fields
    
    Returns:
        BaselineConfig with specified settings
    """
    configs = {
        "no_cache": BaselineConfig(
            baseline_type=BaselineType.NO_CACHE,
            description="No caching - full context reprocessing (worst-case baseline)",
            enable_prefix_caching=False,
        ),
        
        "prefix_cache": BaselineConfig(
            baseline_type=BaselineType.PREFIX_CACHE,
            description="Single-node prefix caching enabled",
            enable_prefix_caching=True,
        ),
        
        "redis": BaselineConfig(
            baseline_type=BaselineType.REDIS_CACHE,
            description="Redis-backed retrieval-artifact cache (no native prefix caching on the serving path)",
            enable_prefix_caching=False,
            redis_host="localhost",
            redis_port=6379,
            use_faiss=True,
            top_k_retrieval=3,
        ),
        
        "rag": BaselineConfig(
            baseline_type=BaselineType.RAG,
            description="Standard RAG with FAISS vector store",
            enable_prefix_caching=False,
            use_faiss=True,
            top_k_retrieval=3,
        ),
        
        "distributed": BaselineConfig(
            baseline_type=BaselineType.DISTRIBUTED_CACHE,
            description="Router-mediated multi-replica prefix-routed baseline (replicated routing is real; simulated transfer policies are experimental only)",
            enable_prefix_caching=True,
            metadata={
                "supports_real_replicated_cluster": True,
                "supports_sharded_context_simulation": True,
                "prefers_gpu": True,
            },
        ),
        
        "hybrid": BaselineConfig(
            baseline_type=BaselineType.HYBRID,
            description="Retrieval-artifact cache plus native prefix caching (no oracle gold-context shortcut)",
            enable_prefix_caching=True,
            use_faiss=True,
            cache_threshold=0.8,
        ),
        
        "speculative": BaselineConfig(
            baseline_type=BaselineType.SPECULATIVE,
            description="Speculative decoding baseline. vLLM configures speculation at LAUNCH "
                        "time (VLLM_SPECULATIVE_CONFIG='{\"method\":\"ngram\",\"num_speculative_tokens\":5}'); "
                        "the experiment measures the resulting TTFT/TPOT/latency. Acceptance-rate "
                        "telemetry requires scraping vLLM /metrics (follow-up).",
            enable_prefix_caching=True,  # Combine with prefix caching for CAG
            speculative_model=None,  # Must be set via override
            num_speculative_tokens=5,
            speculative_method="draft_model",
            metadata={
                "supported_methods": ["draft_model", "ngram", "suffix", "medusa", "eagle"],
                "requires_draft_model": True,
            },
        ),

        # --- Compression axis (Option 3) ---
        "compressed_rag": BaselineConfig(
            baseline_type=BaselineType.COMPRESSED_RAG,
            description="RAG with TEXT compression of retrieved docs (LLMLingua-2) before prompting",
            enable_prefix_caching=False,
            use_faiss=True,
            top_k_retrieval=3,
            compress_method="llmlingua2",
            compress_target_ratio=0.5,
        ),
        "compressed_cag": BaselineConfig(
            baseline_type=BaselineType.COMPRESSED_CAG,
            description="CAG with KV-cache compression (vLLM fp8 KV cache; or an MLA model). Server-side; pair with --kv-cache-dtype fp8.",
            enable_prefix_caching=True,
            kv_cache_dtype="fp8",
            metadata={"server_side_kv_compression": True, "prefers_gpu": True},
        ),

        # --- Staleness/freshness axis (SCAFFOLD) ---
        # Clones the warm hybrid serving path (identical model / decoding / retrieval) and
        # only varies the AGE/validity of served cache entries via stale_fraction. The path
        # that actually serves v0 (stale) vs v1 (fresh) evidence is NOT yet wired; the runner
        # raises a clear NotImplementedError until the evidence-version corpus and the
        # StaleServingPolicy land. See cloud_docs/STALENESS_BASELINE_DESIGN.md.
        "staleness": BaselineConfig(
            baseline_type=BaselineType.STALE,
            description="Staleness/freshness baseline: warm cache held constant, stale_fraction "
                        "sweeps cache-entry age (gold evidence redacted for the stale fraction) "
                        "to plot serving-win vs grounding-loss. See STALENESS_BASELINE_DESIGN.md.",
            enable_prefix_caching=True,
            use_faiss=False,  # serves gold-context v0/v1 directly; no retrieval index needed
            stale_fraction=0.0,
            stale_evidence_mode="version",
            metadata={"serving_path_wired": True},
        ),
    }
    
    if baseline_name not in configs:
        raise ValueError(
            f"Unknown baseline: {baseline_name}. "
            f"Supported: {list(configs.keys())}"
        )
    
    config = configs[baseline_name]
    
    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


# Baseline compatibility checks
def check_baseline_requirements(baseline: BaselineConfig) -> Dict[str, bool]:
    """
    Check if requirements for baseline are met.
    
    Returns:
        Dict with requirement checks
    """
    checks = {
        "vllm_available": importlib.util.find_spec("vllm") is not None,
        "redis_available": True,
        "faiss_available": True,
        "gpu_available": False,
    }
    
    # Check Redis
    if baseline.baseline_type == BaselineType.REDIS_CACHE:
        try:
            import redis
            r = redis.Redis(host=baseline.redis_host, port=baseline.redis_port)
            r.ping()
            checks["redis_available"] = True
        except Exception:
            checks["redis_available"] = False
    
    # Check FAISS
    if baseline.use_faiss:
        try:
            import faiss
            checks["faiss_available"] = True
        except ImportError:
            checks["faiss_available"] = False
    
    # Check GPU (for distributed baseline)
    if baseline.baseline_type == BaselineType.DISTRIBUTED_CACHE:
        try:
            import torch
            checks["gpu_available"] = torch.cuda.is_available()
        except Exception:
            checks["gpu_available"] = False
    
    return checks
