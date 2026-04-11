"""
Prefix-aware router for distributed vLLM replicas.

Routes requests based on prefix hash to maximize cache hits.

Notes:
- Supports both non-streaming JSON responses and streaming (SSE) passthrough.
- Uses a permissive request model (extra fields allowed) so new OpenAI params
  can be forwarded without changing the router.
"""

import hashlib
import json
from typing import Any, AsyncIterator, Dict, List, Optional
from dataclasses import dataclass
import aiohttp
import time
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Pydantic v2 uses ConfigDict; v1 uses an inner Config class.
try:
    from pydantic import ConfigDict as PydanticConfigDict  # type: ignore
except Exception:  # pragma: no cover
    PydanticConfigDict = None  # type: ignore


# Request/Response models
class CompletionRequest(BaseModel):
    """OpenAI-compatible completion request (subset + passthrough extras)."""

    model: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.95
    stop: Optional[List[str]] = None

    # Streaming
    stream: bool = False
    stream_options: Optional[dict[str, Any]] = None

    # Allow forwarding additional OpenAI/vLLM fields without losing them.
    if PydanticConfigDict is not None:
        model_config = PydanticConfigDict(extra="allow")
    else:  # pragma: no cover
        class Config:  # type: ignore
            extra = "allow"


class OllamaGenerateRequest(BaseModel):
    """Ollama-compatible generate request (subset + passthrough extras)."""

    model: str
    prompt: str
    stream: bool = False

    if PydanticConfigDict is not None:
        model_config = PydanticConfigDict(extra="allow")
    else:  # pragma: no cover
        class Config:  # type: ignore
            extra = "allow"


@dataclass
class ReplicaConfig:
    """Configuration for a single vLLM replica."""
    replica_id: str
    api_base: str
    weight: float = 1.0  # Load balancing weight
    
    def __hash__(self):
        """Hash by replica_id so ReplicaConfig is usable in sets/dicts."""
        return hash(self.replica_id)


from src.orchestration.cache_manager import SimulatedKVCacheManager, CacheNode
from src.utils.prompting import extract_cacheable_prefix_text
import asyncio

# ... imports ...

class PrefixAwareRouter:
    """Routes requests to replicas based on prefix hash or round-robin."""

    def __init__(self, replicas: List[ReplicaConfig], strategy: str = "hash"):
        self.replicas = replicas
        self.strategy = strategy
        self.request_count = 0
        self.replica_stats = {r.replica_id: 0 for r in replicas}
        self.tokenizer_name = os.getenv("ROUTER_TOKENIZER") or ""
        self._tokenizer: Any = None
        self.tokenization_mode = "uninitialized"
        self.tokenizer_error: Optional[str] = None
        
        # Initialize simulation
        nodes = [
            CacheNode(
                node_id=r.replica_id, 
                host=r.api_base, 
                port=0, 
                vram_total=0, 
                vram_used=0, 
                cache_blocks_capacity=0, 
                cache_blocks_used=0
            ) 
            for r in replicas
        ]
        self.cache_manager = SimulatedKVCacheManager(nodes, policy="replicated")

    def set_sharding_policy(self, policy: str):
        """Update the cache sharding policy for simulation."""
        self.cache_manager.policy = policy
        print(f"Router sharding policy set to: {policy}")
    async def initialize_tokenizer(self) -> None:
        """Best-effort tokenizer initialization for prompt-prefix routing."""
        if self._tokenizer is not None:
            return

        tokenizer_name = self.tokenizer_name
        if not tokenizer_name:
            try:
                models_payload = await self.list_models()
                data = models_payload.get("data") if isinstance(models_payload, dict) else None
                if isinstance(data, list) and data:
                    tokenizer_name = str(data[0].get("id") or "").strip()
            except Exception as e:
                self.tokenizer_error = str(e)

        if not tokenizer_name:
            self.tokenization_mode = "utf8_fallback"
            if not self.tokenizer_error:
                self.tokenizer_error = "Unable to determine tokenizer/model name from replicas."
            return

        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                trust_remote_code=True,
                use_fast=True,
            )
            self.tokenizer_name = tokenizer_name
            self.tokenization_mode = "model_tokenizer"
            self.tokenizer_error = None
        except Exception as e:
            self.tokenizer_name = tokenizer_name
            self.tokenization_mode = "utf8_fallback"
            self.tokenizer_error = str(e)
            print(
                f"Warning: failed to load tokenizer '{tokenizer_name}' for router prefix hashing: {e}"
            )

    def _prefix_tokens(self, prompt: str) -> List[int]:
        """Tokenize the cacheable prefix; fall back to UTF-8 bytes if needed."""
        prefix_text = extract_cacheable_prefix_text(prompt)
        if self._tokenizer is not None:
            try:
                return [int(t) for t in self._tokenizer.encode(prefix_text, add_special_tokens=False)]
            except Exception as e:
                self.tokenization_mode = "utf8_fallback"
                self.tokenizer_error = str(e)
        return list(prefix_text.encode("utf-8"))

    def compute_prefix_hash(self, prefix_tokens: List[int]) -> str:
        """Compute a stable hash for the actual prompt-token prefix used for routing."""
        payload = json.dumps(prefix_tokens, separators=(",", ":")).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()

    def _select_replicated_replica(self, prefix_hash: str) -> ReplicaConfig:
        if not self.replicas:
            raise HTTPException(status_code=503, detail="No replicas configured")
        if self.strategy == "round_robin":
            return self.replicas[self.request_count % len(self.replicas)]
        replica_index = int(prefix_hash, 16) % len(self.replicas)
        return self.replicas[replica_index]

    async def route_request_with_simulation(self, prompt: str) -> tuple[ReplicaConfig, Dict[str, Any]]:
        """
        Select a replica and simulate distributed cache latency.
        Returns (replica, simulation_metadata).
        """
        if self._tokenizer is None and (
            self.tokenization_mode == "uninitialized" or not self.tokenizer_name
        ):
            await self.initialize_tokenizer()
        prefix_tokens = self._prefix_tokens(prompt)
        prefix_hash = self.compute_prefix_hash(prefix_tokens)

        if self.cache_manager.policy == "replicated":
            selected_replica = self._select_replicated_replica(prefix_hash)
            target_node_id = selected_replica.replica_id
            latency_ms = 0.0
            sim_result: Dict[str, Any] = {
                "target_node_id": target_node_id,
                "cached_tokens": 0,
                "transfer_required": False,
                "transfer_bytes": 0,
                "transfer_latency_ms": 0.0,
            }
            routing_mode = "prefix_hash" if self.strategy != "round_robin" else "round_robin"
        else:
            sim_result = self.cache_manager.resolve_prefix(prefix_tokens)
            target_node_id = sim_result["target_node_id"]
            latency_ms = float(sim_result.get("transfer_latency_ms", 0.0) or 0.0)
            selected_replica = next(
                (r for r in self.replicas if r.replica_id == target_node_id),
                self.replicas[0],
            )
            routing_mode = "simulated_transfer"
        self.request_count += 1
        self.replica_stats[selected_replica.replica_id] += 1
        return selected_replica, {
            "target_node_id": target_node_id,
            "selected_replica_id": selected_replica.replica_id,
            "cached_tokens": int(sim_result.get("cached_tokens", 0) or 0),
            "transfer_required": bool(sim_result.get("transfer_required", latency_ms > 0.0)),
            "transfer_bytes": int(sim_result.get("transfer_bytes", 0) or 0),
            "transfer_latency_ms": latency_ms,
            "policy": str(self.cache_manager.policy),
            "routing_mode": routing_mode,
            "prefix_hash": prefix_hash,
            "prefix_token_count": len(prefix_tokens),
            "tokenizer_name": self.tokenizer_name or None,
            "tokenization_mode": self.tokenization_mode,
        }

    def set_strategy(self, strategy: str):
        """Update the routing strategy ('hash' or 'round_robin')."""
        if strategy not in {"hash", "round_robin"}:
            raise ValueError("strategy must be 'hash' or 'round_robin'")
        self.strategy = strategy

    async def forward_request_json(
        self,
        replica: ReplicaConfig,
        payload: Dict[str, Any],
        *,
        path: str = "/v1/completions",
    ) -> Dict[str, Any]:
        """Forward a non-streaming request and return JSON."""
        url = f"{replica.api_base}{path}"

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as upstream:
                    upstream.raise_for_status()
                    return await upstream.json()
            except Exception as e:
                raise HTTPException(
                    status_code=502,
                    detail=f"Error forwarding to replica {replica.replica_id}: {str(e)}",
                )

    async def forward_request_stream(
        self,
        replica: ReplicaConfig,
        payload: Dict[str, Any],
        *,
        start_time: float,
        path: str = "/v1/completions",
    ) -> tuple[AsyncIterator[bytes], str]:
        """Forward a streaming request and return an async byte iterator."""
        url = f"{replica.api_base}{path}"

        session = aiohttp.ClientSession()
        try:
            upstream = await session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            )
        except Exception as e:
            await session.close()
            raise HTTPException(
                status_code=502,
                detail=f"Error connecting to replica {replica.replica_id}: {str(e)}",
            )

        if upstream.status >= 400:
            try:
                body = await upstream.text()
            finally:
                upstream.release()
                await session.close()
            raise HTTPException(
                status_code=502,
                detail=f"Upstream error from {replica.replica_id}: {upstream.status} {body}",
            )

        content_type = upstream.headers.get("Content-Type", "text/event-stream")

        async def iterator() -> AsyncIterator[bytes]:
            first_byte = True
            total_bytes = 0
            try:
                async for chunk in upstream.content.iter_any():
                    if not chunk:
                        continue
                    if first_byte:
                        REQUEST_TTFT.observe(max(0.0, time.time() - start_time))
                        first_byte = False
                    total_bytes += len(chunk)
                    yield chunk
            finally:
                REQUEST_STREAM_BYTES.inc(total_bytes)
                REQUEST_STREAM_DURATION.observe(max(0.0, time.time() - start_time))
                upstream.release()
                await session.close()

        return iterator(), content_type


    async def route_request(self, prompt: str) -> ReplicaConfig:
        """Select a replica for a prompt, ignoring transfer metadata."""
        replica, _ = await self.route_request_with_simulation(prompt)
        return replica
    async def list_models(self) -> Dict[str, Any]:
        """Return a vLLM-compatible /v1/models payload from a healthy replica."""
        last_error: Optional[str] = None
        async with aiohttp.ClientSession() as session:
            for replica in self.replicas:
                try:
                    async with session.get(
                        f"{replica.api_base}/v1/models",
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as upstream:
                        upstream.raise_for_status()
                        payload = await upstream.json()
                        if isinstance(payload, dict):
                            return payload
                        last_error = f"{replica.replica_id} returned a non-JSON model payload"
                except Exception as e:
                    last_error = f"{replica.replica_id}: {str(e)}"

        raise HTTPException(
            status_code=502,
            detail=f"Unable to retrieve model metadata from replicas: {last_error or 'unknown error'}",
        )

    async def get_version(self) -> Dict[str, Any]:
        """Return backend version metadata from a healthy replica."""
        last_error: Optional[str] = None
        async with aiohttp.ClientSession() as session:
            for replica in self.replicas:
                try:
                    async with session.get(
                        f"{replica.api_base}/version",
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as upstream:
                        upstream.raise_for_status()
                        payload = await upstream.json()
                        if isinstance(payload, dict):
                            return payload
                        last_error = f"{replica.replica_id} returned a non-JSON version payload"
                except Exception as e:
                    last_error = f"{replica.replica_id}: {str(e)}"

        raise HTTPException(
            status_code=502,
            detail=f"Unable to retrieve version metadata from replicas: {last_error or 'unknown error'}",
        )
    
    def get_stats(self) -> Dict:
        """Get routing statistics."""
        return {
            "total_requests": self.request_count,
            "replica_distribution": self.replica_stats,
            "num_replicas": len(self.replicas),
            "strategy": self.strategy,
            "sharding_policy": self.cache_manager.policy,
            "tokenizer_name": self.tokenizer_name or None,
            "tokenization_mode": self.tokenization_mode,
            "tokenizer_error": self.tokenizer_error,
            "distinct_api_bases": len({r.api_base for r in self.replicas}),
            "replicas": [
                {
                    "replica_id": r.replica_id,
                    "api_base": r.api_base,
                    "weight": r.weight,
                }
                for r in self.replicas
            ],
        }


def _parse_router_replicas_env(value: str) -> List[ReplicaConfig]:
    """Parse ROUTER_REPLICAS into a list of ReplicaConfig.

    Supported formats:
    - JSON list of objects: [{"replica_id": "replica-1", "api_base": "http://...", "weight": 1.0}, ...]
    - JSON list of strings: ["http://...", "http://...", ...]
    - Comma-separated: "replica-1=http://...,replica-2=http://..." or "http://...,http://..."

    Returns:
        List of ReplicaConfig, or an empty list if parsing fails.
    """
    v = (value or "").strip()
    if not v:
        return []

    # JSON format.
    if v[:1] in {"[", "{"}:
        try:
            obj = json.loads(v)
        except Exception:
            obj = None

        if isinstance(obj, dict):
            obj = [obj]

        if isinstance(obj, list):
            if all(isinstance(x, str) for x in obj):
                return [
                    ReplicaConfig(replica_id=f"replica-{i+1}", api_base=x.rstrip("/"))
                    for i, x in enumerate(obj)
                    if x
                ]

            replicas: list[ReplicaConfig] = []
            for i, x in enumerate(obj):
                if not isinstance(x, dict):
                    continue
                replica_id = str(x.get("replica_id") or f"replica-{i+1}")
                api_base = x.get("api_base")
                if not api_base:
                    continue
                weight = x.get("weight", 1.0)
                replicas.append(
                    ReplicaConfig(
                        replica_id=replica_id,
                        api_base=str(api_base).rstrip("/"),
                        weight=float(weight) if isinstance(weight, (int, float)) else 1.0,
                    )
                )
            return replicas

    # Comma-separated.
    parts = [p.strip() for p in v.split(",") if p.strip()]
    replicas: list[ReplicaConfig] = []
    for i, part in enumerate(parts):
        if "=" in part:
            rid, base = part.split("=", 1)
            replica_id = rid.strip() or f"replica-{i+1}"
            api_base = base.strip()
        else:
            replica_id = f"replica-{i+1}"
            api_base = part

        if not api_base:
            continue
        replicas.append(ReplicaConfig(replica_id=replica_id, api_base=api_base.rstrip("/")))

    return replicas


# FastAPI app
app = FastAPI(title="CAGE Router", description="Prefix-aware router for distributed vLLM")

# Metrics
REQUEST_COUNTER = Counter("cage_router_requests_total", "Total routed requests")
REQUEST_LATENCY = Histogram(
    "cage_router_request_latency_seconds",
    "Latency for routing + forwarding (non-stream), or full stream duration (stream)",
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 30),
)
REQUEST_TTFT = Histogram(
    "cage_router_ttft_seconds",
    "Time to first bytes from upstream when proxying a stream",
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5),
)
REQUEST_STREAM_DURATION = Histogram(
    "cage_router_stream_duration_seconds",
    "Total duration of proxied streaming responses",
    buckets=(0.1, 0.2, 0.5, 1, 2, 5, 10, 30, 60, 120),
)
REQUEST_STREAM_BYTES = Counter(
    "cage_router_stream_bytes_total",
    "Total bytes proxied for streaming responses",
)

# Global router instance (will be configured on startup)
router: Optional[PrefixAwareRouter] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the global router instance on startup."""
    global router

    strategy = os.getenv("ROUTER_STRATEGY", "hash")

    # Prefer explicit configuration via env (works for Docker Compose / K8s).
    replicas_env = os.getenv("ROUTER_REPLICAS", "")
    replicas = _parse_router_replicas_env(replicas_env)

    # Fallback: local development defaults.
    if not replicas:
        replicas = [
            ReplicaConfig(replica_id="replica-1", api_base="http://localhost:8001"),
            ReplicaConfig(replica_id="replica-2", api_base="http://localhost:8002"),
            ReplicaConfig(replica_id="replica-3", api_base="http://localhost:8003"),
        ]

    router = PrefixAwareRouter(replicas, strategy=strategy)
    await router.initialize_tokenizer()
    print(f"Router initialized with {len(replicas)} replicas, strategy={strategy}")


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """OpenAI-compatible completions endpoint with prefix-aware routing."""
    if router is None:
        raise HTTPException(status_code=503, detail="Router not initialized")

    start = time.time()

    # Route request (with simulation)
    replica, kv_transfer_params = await router.route_request_with_simulation(request.prompt)
    latency_ms = float(kv_transfer_params.get("transfer_latency_ms", 0.0) or 0.0)
    if latency_ms > 0 and str(kv_transfer_params.get("policy")) != "replicated":
        # Simulate network transfer latency
        await asyncio.sleep(latency_ms / 1000.0)

    # Preserve any extra fields from the client.
    payload = request.model_dump() if hasattr(request, "model_dump") else request.dict()

    REQUEST_COUNTER.inc()
    response_headers = {
        "x-router-replica": replica.replica_id,
        "x-kv-transfer-params": json.dumps(kv_transfer_params, separators=(",", ":")),
    }

    # Streaming passthrough (SSE)
    if bool(payload.get("stream")):
        iterator, content_type = await router.forward_request_stream(
            replica, payload, start_time=start
        )
        media_type = content_type.split(";")[0].strip() if content_type else "text/event-stream"

        # Also observe the total stream duration in REQUEST_LATENCY for convenience.
        # (REQUEST_STREAM_DURATION is the authoritative stream-duration metric.)
        async def wrapped() -> AsyncIterator[bytes]:
            try:
                async for chunk in iterator:
                    yield chunk
            finally:
                REQUEST_LATENCY.observe(max(0.0, time.time() - start))

        return StreamingResponse(
            wrapped(),
            media_type=media_type,
            headers=response_headers,
        )

    # Non-streaming JSON
    response_data = await router.forward_request_json(replica, payload)
    if isinstance(response_data, dict):
        existing_kv_params = response_data.get("kv_transfer_params")
        if isinstance(existing_kv_params, dict):
            response_data["kv_transfer_params"] = {**existing_kv_params, **kv_transfer_params}
        else:
            response_data["kv_transfer_params"] = kv_transfer_params
    elapsed = time.time() - start
    REQUEST_LATENCY.observe(elapsed)
    return JSONResponse(content=response_data, headers=response_headers)


@app.post("/api/generate")
async def ollama_generate(request: OllamaGenerateRequest):
    """Ollama-compatible generate endpoint with prefix-aware routing."""
    if router is None:
        raise HTTPException(status_code=503, detail="Router not initialized")

    start = time.time()
    replica, kv_transfer_params = await router.route_request_with_simulation(request.prompt)
    payload = request.model_dump() if hasattr(request, "model_dump") else request.dict()

    REQUEST_COUNTER.inc()
    response_headers = {
        "x-router-replica": replica.replica_id,
        "x-kv-transfer-params": json.dumps(kv_transfer_params, separators=(",", ":")),
    }

    if bool(payload.get("stream")):
        iterator, content_type = await router.forward_request_stream(
            replica, payload, start_time=start, path="/api/generate"
        )
        media_type = content_type.split(";")[0].strip() if content_type else "application/json"

        async def wrapped() -> AsyncIterator[bytes]:
            try:
                async for chunk in iterator:
                    yield chunk
            finally:
                REQUEST_LATENCY.observe(max(0.0, time.time() - start))

        return StreamingResponse(
            wrapped(),
            media_type=media_type,
            headers=response_headers,
        )

    response_data = await router.forward_request_json(replica, payload, path="/api/generate")
    elapsed = time.time() - start
    REQUEST_LATENCY.observe(elapsed)
    return JSONResponse(content=response_data, headers=response_headers)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "router_initialized": router is not None}


@app.get("/stats")
async def stats():
    """Get routing statistics."""
    if router is None:
        raise HTTPException(status_code=503, detail="Router not initialized")
    
    return router.get_stats()


@app.get("/v1/models")
async def list_models():
    """Expose a vLLM-compatible models endpoint for readiness checks."""
    if router is None:
        raise HTTPException(status_code=503, detail="Router not initialized")

    return await router.list_models()


@app.get("/version")
async def version():
    """Expose backend version metadata for reproducibility capture."""
    if router is None:
        raise HTTPException(status_code=503, detail="Router not initialized")

    return await router.get_version()


@app.post("/configure")
async def configure(replicas: List[Dict]):
    """
    Reconfigure router with new replica list.
    
    Args:
        replicas: List of replica configs with keys: replica_id, api_base, weight (optional)
    """
    global router
    
    replica_configs = [
        ReplicaConfig(
            replica_id=r["replica_id"],
            api_base=r["api_base"],
            weight=r.get("weight", 1.0)
        )
        for r in replicas
    ]
    
    current_strategy = router.strategy if router else "hash"
    router = PrefixAwareRouter(replica_configs, strategy=current_strategy)
    await router.initialize_tokenizer()
    
    return {
        "status": "configured",
        "num_replicas": len(replica_configs),
        "replicas": [r.replica_id for r in replica_configs]
    }


@app.post("/sharding-policy")
async def set_sharding_policy(payload: Dict):
    """
    Set simulation sharding policy: {"policy": "replicated" | "sharded_context"}
    """
    global router
    if router is None:
        raise HTTPException(status_code=503, detail="Router not initialized")
    
    policy = payload.get("policy")
    if policy not in {"replicated", "sharded_context"}:
        raise HTTPException(status_code=400, detail="Invalid policy. Use 'replicated' or 'sharded_context'")
        
    router.set_sharding_policy(policy)
    return {"status": "ok", "policy": policy}


@app.post("/routing-strategy")
async def set_routing_strategy(payload: Dict):
    """
    Set routing strategy: {"strategy": "hash" | "round_robin"}
    """
    global router
    if router is None:
        raise HTTPException(status_code=503, detail="Router not initialized")
    strat = payload.get("strategy")
    try:
        router.set_strategy(strat)
        return {"status": "ok", "strategy": strat}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# CLI for running router
if __name__ == "__main__":
    """Run the router with uvicorn.

    Note:
        Pass the in-memory ASGI app object to avoid re-importing this module,
        which would re-register Prometheus metrics and crash.
    """

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("ROUTER_PORT", "9000")),
        log_level="info",
    )
