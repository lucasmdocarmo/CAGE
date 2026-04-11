"""
Baseline configurations for CAGE benchmarking.

Baselines:
1. No caching - Full context reprocessing
2. Prefix caching - Single-node prefix caching
3. Redis cache - Redis-backed retrieval-artifact cache
4. Standard RAG - FAISS vector store + retrieval
5. Distributed cache - Router-mediated multi-replica routing
6. Hybrid CAG↔RAG - Retrieval cache plus native prefix caching
7. Speculative baseline scaffold - CLI/config present, backend wiring incomplete
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
            description="Speculative baseline scaffold (configuration is captured, but backend decoding parameters are not fully wired into inference requests yet)",
            enable_prefix_caching=True,  # Combine with prefix caching for CAG
            speculative_model=None,  # Must be set via override
            num_speculative_tokens=5,
            speculative_method="draft_model",
            metadata={
                "supported_methods": ["draft_model", "ngram", "suffix", "medusa", "eagle"],
                "requires_draft_model": True,
            },
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
