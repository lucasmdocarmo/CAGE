"""
Abstract interface for Pluggable KV Cache Managers.

This defines the contract for implementing different cache distribution policies
(Sharding, Offloading, Replication) as described in the CAGE proposal.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import enum

class CachePolicy(enum.Enum):
    REPLICATED = "replicated"
    SHARDED_TENSOR = "sharded_tensor"   # Tensor Parallelism
    SHARDED_CONTEXT = "sharded_context" # Context Parallelism
    OFFLOAD_CPU = "offload_cpu"
    OFFLOAD_NVME = "offload_nvme"

@dataclass
class CacheNode:
    node_id: str
    host: str
    port: int
    vram_total: int
    vram_used: int
    cache_blocks_capacity: int
    cache_blocks_used: int

@dataclass
class CacheBlock:
    block_id: str
    token_ids: List[int]
    locations: List[str]  # List of node_ids holding this block

class KVCacheManager(ABC):
    """
    Interface for managing distributed KV caches.
    """

    @abstractmethod
    def initialize(self, nodes: List[CacheNode], policy: CachePolicy):
        """Initialize the cache cluster."""
        pass

    @abstractmethod
    def allocate_context(self, prompt_token_ids: List[int]) -> List[CacheBlock]:
        """
        Decide where to store the KV cache for a new prompt context.
        Returns a list of logical blocks and their assigned physical locations.
        """
        pass

    @abstractmethod
    def resolve_prefix(self, prefix_token_ids: List[int]) -> Dict[str, Any]:
        """
        Find best node(s) that already have this prefix cached.
        
        Returns:
            Dict containing:
            - target_node_id: The best node to route the request to.
            - cached_tokens: Number of tokens already cached on that node.
            - transfer_required: Whether data needs to be moved.
        """
        pass

    @abstractmethod
    def invalidate(self, block_ids: List[str]):
        """Remove blocks from cache."""
        pass

class SimulatedKVCacheManager(KVCacheManager):
    """
    Simulates distributed KV cache placement and transfer costs.
    """
    
    def __init__(self, nodes: List[CacheNode], policy: str = "replicated"):
        self.nodes = {n.node_id: n for n in nodes}
        self.node_list = nodes
        self.policy = policy
        # Simple logical cache: map prefix_hash -> List[node_ids]
        self.logical_cache: Dict[int, List[str]] = {}
        
        # Simulation parameters
        self.network_bandwidth_gbps = 100.0  # Default 100Gbps
        self.hidden_size = 2048 # Example for Llama-3.2-1B
        self.layers = 16
        self.bytes_per_token = self.hidden_size * self.layers * 2 * 2 # 2 (K+V) * 2 (float16)

    def initialize(self, nodes: List[CacheNode], policy: str):
        self.nodes = {n.node_id: n for n in nodes}
        self.node_list = nodes
        self.policy = policy

    def allocate_context(self, prompt_token_ids: List[int]) -> List[CacheBlock]:
        # Not used in simple simulation
        return []

    def resolve_prefix(self, prefix_token_ids: List[int]) -> Dict[str, Any]:
        """
        Determine where the prefix should be processed and cost to move data.
        """
        import hashlib
        
        # Create a stable hash of the prefix content
        prefix_str = ",".join(map(str, prefix_token_ids))
        prefix_hash = int(hashlib.md5(prefix_str.encode()).hexdigest(), 16)
        num_tokens = len(prefix_token_ids)
        
        target_node = self.node_list[prefix_hash % len(self.node_list)]
        
        transfer_bytes = 0
        transfer_latency = 0.0
        
        if self.policy == "replicated":
            # In replicated mode (CAG), we assume the router sends the request 
            # to the node responsible for this hash.
            # If the request was routed correctly (which we simulate here by picking target),
            # cost is 0. If we were forced to go elsewhere, cost would be full transfer.
            pass
            
        elif self.policy == "sharded_context":
            # Context Parallelism simulation:
            # Tokens are striped across nodes. To process the sequence on `target_node`,
            # we need to gather KV blocks from all other nodes.
            # Assumption: Data is evenly distributed.
            # We need to fetch (N-1)/N of the context from others.
            
            num_nodes = len(self.node_list)
            if num_nodes > 1:
                fraction_remote = (num_nodes - 1) / num_nodes
                tokens_to_fetch = num_tokens * fraction_remote
                transfer_bytes = tokens_to_fetch * self.bytes_per_token
                
                # Latency = Size / Bandwidth
                # bandwidth in bytes/sec = gbps * 1e9 / 8
                bw_bytes_sec = self.network_bandwidth_gbps * 1e9 / 8
                transfer_latency = transfer_bytes / bw_bytes_sec

        return {
            "target_node_id": target_node.node_id,
            "cached_tokens": num_tokens,
            "transfer_required": bool(transfer_bytes > 0),
            "transfer_bytes": int(transfer_bytes),
            "transfer_latency_ms": transfer_latency * 1000.0
        }

    def invalidate(self, block_ids: List[str]):
        pass

    def get_cluster_stats(self) -> Dict[str, Any]:
        return {"policy": self.policy, "nodes": len(self.node_list)}
