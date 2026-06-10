> <!-- CAGE-DOC-STATUS -->
> **⚠️ STATUS: PARTIALLY STALE (2026-06-09).** Useful background, but some commands,
> CLI flags, paths, and metric numbers here predate the June 2026 fixes. In
> particular, `run_experiment.py` has **no** `--phase`/`--all-baselines`/`--trials`/`--queries`
> flags (use `--baseline`, `--num-trials`, `--num-queries`), and pre-fix metric numbers
> (faithfulness 0.570, BERTScore 0.324) are now obsolete. For runnable commands use
> [`RUNBOOK.md`](RUNBOOK.md); for current metrics see [`KNOWLEDGE_BASE.md`](KNOWLEDGE_BASE.md).

# CAGE Technical Architecture

> **Purpose**: Deep technical reference for understanding how the codebase works, how results are generated, how vLLM is integrated, and how every component connects. Written for an AI or developer picking up the project cold.

---

## How an Experiment Runs End-to-End

When you execute `python scripts/run_experiment.py --baseline prefix_cache --trials 3 --queries 50`, this is what happens:

```
1. CLI parsing → baseline config loaded from src/orchestration/baselines.py
2. Dataset loaded → src/data/loader.py → SQuADv2 examples as CAGExample objects
3. IR index built (if retrieval baseline) → src/orchestration/ir.py → FAISS index
4. vLLM adapter created → src/inference/vllm_adapter.py → HTTP client to vLLM server
5. For each trial (1..3):
   a. For each query (1..50):
      i.   Prompt built → src/utils/prompting.py → format_qa_prompt()
      ii.  Context selected (gold passage OR retrieved via IR)
      iii. Request sent to vLLM → streaming HTTP → TTFT measured from first SSE chunk
      iv.  Response collected with telemetry (prompt_tokens, cached_prompt_tokens)
      v.   Quality scored → src/evaluation/quality.py → faithfulness, relevance, BERTScore
      vi.  Performance recorded → src/evaluation/performance.py → latency, TTFT, TPOT
   b. Trial metrics aggregated and saved to JSON
6. Cross-trial aggregation → aggregated_metrics.json written
```

### Output structure
```
analysis/phase1/results/<baseline_name>/
├── aggregated_metrics.json    # Mean ± std across all trials
├── trial_1/
│   └── <timestamp>_metrics.json   # Per-trial raw metrics
├── trial_2/
│   └── <timestamp>_metrics.json
└── trial_3/
    └── <timestamp>_metrics.json
```

---

## Source Code Modules

### `src/data/loader.py` — Dataset Loading

**Role**: Loads HuggingFace datasets and converts them to `CAGExample` objects.

**Key class**: `CAGExample`
```python
@dataclass
class CAGExample:
    id: str
    question: str
    context: List[str]    # Gold passages from the dataset
    answer: str           # Reference answer for quality evaluation
    metadata: Dict[str, Any]
```

**Supported datasets**: SQuADv2 (`SQuADv2Loader`), HotpotQA, TriviaQA, QASPER, HumanEval, MBPP.

**How it's used**: `run_experiment.py` calls `get_loader("squad_v2")` which returns a `SQuADv2Loader`. The loader pulls from HuggingFace's `datasets` library and formats each example with its gold passage as `context[0]`.

**Critical detail**: For gold-context baselines (No Cache, Prefix Cache, Distributed), the experiment uses `example.context[0]` directly. For retrieval baselines (RAG, Redis, Hybrid), the gold context is NOT used — instead, passages are retrieved via the IR module.

### `src/inference/engine.py` — Abstract Inference Interface

**Role**: Defines the `InferenceEngine` abstract class and the `InferenceRequest`/`InferenceResponse` dataclasses.

**Key dataclass**: `InferenceResponse`
```python
@dataclass
class InferenceResponse:
    request_id: Optional[str]
    generated_text: str
    ttft_ms: float              # Time to first token
    total_time_ms: float        # End-to-end latency
    num_tokens: int             # Generated token count
    model_name: str
    finish_reason: str          # "length", "stop", "error"
    prompt_tokens: Optional[int]         # From vLLM usage telemetry
    cached_prompt_tokens: Optional[int]  # From vLLM usage telemetry
    kv_transfer_params: Optional[Dict]   # For distributed baselines
```

The `prompt_tokens` and `cached_prompt_tokens` fields are populated only when vLLM returns usage telemetry (requires `--enable-prompt-tokens-details` flag on the vLLM server).

### `src/inference/vllm_adapter.py` — vLLM HTTP Client

**Role**: Sends requests to a vLLM server via the OpenAI-compatible `/v1/completions` API.

**Key class**: `VLLMAdapter`

**How TTFT is measured**: The adapter uses **streaming mode** (`stream=True`). It records `time.perf_counter()` before sending the request and again when the first SSE `data:` chunk arrives. The difference is TTFT. This is the most accurate method available without modifying vLLM internals.

**How telemetry is extracted**: When `stream_options={"include_usage": true}` is set, vLLM sends a final SSE chunk with `usage` data including:
- `prompt_tokens`: total prompt tokens processed
- `prompt_tokens_details.cached_tokens`: how many were served from prefix cache

This is how the paper's "prompt cached ratio" (68.4%, 75.6%, 89.2%) is computed.

**Connection to baselines**: For most baselines, the adapter targets `http://localhost:8000` (single vLLM instance). For the Distributed baseline, it targets `http://localhost:9000` (the router, which forwards to replicas on 8001/8002/8003).

### `src/orchestration/baselines.py` — Baseline Configuration

**Role**: Defines the 7 baseline types and their configurations.

**Key enum**: `BaselineType` — `NO_CACHE`, `PREFIX_CACHE`, `REDIS_CACHE`, `RAG`, `DISTRIBUTED_CACHE`, `HYBRID`, `SPECULATIVE`

**Key function**: `get_baseline_config(baseline_name)` — returns a `BaselineConfig` dataclass with all parameters for that baseline (API base, Redis config, IR config, etc.).

**How baselines differ at the code level**:
- `NO_CACHE`: `enable_prefix_caching=False`, uses gold context, sends to single vLLM at :8000
- `PREFIX_CACHE`: `enable_prefix_caching=True`, uses gold context, sends to single vLLM at :8000 (same server, but vLLM internally caches prefix KV blocks)
- `RAG`: `enable_prefix_caching=False`, `use_faiss=True`, retrieves context via IR module
- `REDIS_CACHE`: Like RAG but checks Redis for cached retrieval artifacts first
- `HYBRID`: Like RAG but `enable_prefix_caching=True` (retrieval + vLLM prefix cache)
- `DISTRIBUTED`: `enable_prefix_caching=True`, sends to router at :9000 instead of direct vLLM

**Important**: The vLLM server itself doesn't change between baselines (except for the `--enable-prefix-caching` flag). What changes is: (a) the context source (gold vs retrieved), (b) whether Redis is consulted, and (c) which HTTP endpoint receives the request.

### `src/orchestration/ir.py` — Information Retrieval

**Role**: Builds a FAISS vector index from dataset contexts and performs dense retrieval + optional reranking.

**How the index is built**:
1. `build_corpus_from_contexts()` extracts all unique context passages from the dataset
2. Each passage is embedded using `SentenceTransformer("intfloat/e5-large-v2")`
3. Embeddings are stored in a FAISS `IndexFlatIP` (inner product) index
4. Index is persisted to `experiments/ir_index/` for reuse across runs

**How retrieval works**:
1. Query is embedded with the same SentenceTransformer
2. FAISS returns top-k=3 nearest passages
3. If a reranker is configured (`BAAI/bge-reranker-large`), results are re-scored with a CrossEncoder
4. Final top-k passages are returned as the context for the prompt

**Retrieval hit rate**: The 98% retrieval hit rate in Phase 1 means 49 of 50 queries found at least one passage that matched the gold context document. This is computed by `retrieval_hit_rate()` which checks if any retrieved doc_id matches the gold passage's doc_id.

### `src/orchestration/router.py` — Distributed Router (FastAPI)

**Role**: A FastAPI service that receives inference requests and routes them to vLLM replicas based on prompt prefix hash.

**How routing works**:
1. Router receives a `/v1/completions` request
2. Extracts the prompt text
3. Computes `hashlib.sha256(prompt[:prefix_length])` to get a prefix hash
4. Maps the hash to a replica using modular hashing: `replica = hash % num_replicas`
5. Forwards the full request to the selected replica's API
6. Streams the response back to the client

**Why this helps**: If the same prefix is always routed to the same replica, that replica's vLLM prefix cache is more likely to have the KV blocks warm. This is why the Distributed baseline's p50 TTFT (2,876ms) is close to Prefix Cache — most requests hit a warm replica.

**Why the tail is bad**: The 7.6× p95/p50 TTFT spread happens because some requests land on a replica that hasn't seen that prefix before (cold miss). On first request, the replica must compute the full prefill, producing latency similar to No Cache.

**Deployment**: The router runs as a Docker container (see `docker/router.Dockerfile`) and is configured via `ROUTER_REPLICAS` environment variable.

### `src/orchestration/redis_cache.py` — Redis Retrieval Cache

**Role**: Caches retrieval artifacts (query → retrieved doc IDs) in Redis.

**Important distinction**: This does NOT cache vLLM KV tensors. It caches the *retrieval results* so that repeated queries don't need to re-run FAISS + reranking.

**How it works**:
- `RetrievalCache.get(query)` → checks Redis for a cached result
- `RetrievalCache.put(query, results)` → stores retrieval results in Redis
- Cache key: SHA1 hash of the query text under a namespace prefix

**Cold vs Warm**: Redis is flushed (`FLUSHDB`) before cold baselines. Warm baselines run 50 warmup queries first to populate the cache.

### `src/orchestration/cache_manager.py` — Abstract KV Cache Manager

**Role**: Defines the interface for pluggable KV cache distribution policies (replicated, sharded, offloaded). This is scaffolding for Phase 3 — not yet fully implemented.

**Policies defined**: `REPLICATED`, `SHARDED_TENSOR`, `SHARDED_CONTEXT`, `OFFLOAD_CPU`, `OFFLOAD_NVME`

### `src/evaluation/quality.py` — Quality Metrics

**Role**: Computes semantic quality metrics for each generated answer.

**Metrics computed**:
- **Faithfulness**: NLI-based entailment score (uses a SentenceTransformer cross-encoder to check if the answer is entailed by the context)
- **Relevance**: Embedding cosine similarity between answer and question
- **BERTScore**: Token-level soft F1 between generated and reference answers
- **ROUGE-L**: Longest common subsequence F1
- **F1 / Exact Match**: Token-level QA metrics (available but not used in Phase 1 paper)

**How faithfulness is computed**: The evaluator splits the generated answer into claims, then checks each claim against the provided context using an NLI model. The faithfulness score is the proportion of claims that are entailed.

### `src/evaluation/performance.py` — Performance Metrics

**Role**: Aggregates per-request timing data into summary statistics.

**Key class**: `PerformanceEvaluator`

**Metrics computed**: QPS, tokens/sec, avg/p50/p95/p99 for TTFT, TPOT, and end-to-end latency. Also tracks CPU/memory utilization.

**How TPOT is computed**: `(total_time_ms - ttft_ms) / (num_tokens - 1)` — the time per output token after the first token.

**Cache metrics**: `CacheMetricsTracker` separately tracks `prompt_tokens`, `cached_prompt_tokens`, and computes the prompt-cached ratio per request.

### `src/utils/prompting.py` — Prompt Templates

**Role**: Standardizes prompt formatting across all baselines.

**Key function**: `format_qa_prompt(question, contexts, system_prefix=...)`

**Format**:
```
You are a helpful assistant. Answer the question using ONLY the provided context.
If the context is insufficient, say you don't know.

Context 1: <passage text>

Question: <question text>
Answer:
```

**Why this matters for prefix caching**: The system prefix is identical across all requests. With prefix caching enabled, vLLM caches the KV blocks for this shared prefix. Requests that share the same context passage will also share cached KV blocks for that portion. This is the mechanism that produces the 68.4% prompt-cached ratio.

---

## vLLM Integration

### How vLLM is deployed

**CPU mode (Phase 1)**:
```bash
# Single replica (most baselines)
vllm serve Qwen/Qwen3-4B --port 8000 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --cpu-offload-gb 0
# Environment: VLLM_CPU_KVCACHE_SPACE=10

# Multi-replica (Distributed baseline) via docker-compose.yml
docker-compose up  # Starts 3 replicas on 8001/8002/8003 + router on 9000
```

**GPU mode (Phase 2)**:
```bash
# Uses docker-compose.gpu.yml
docker-compose -f docker-compose.gpu.yml up
# Environment: VLLM_GPU_MEMORY_UTILIZATION=0.9
```

### Key vLLM flags
- `--enable-prefix-caching`: Enables automatic prefix KV block reuse. Without this, every request recomputes the full prompt.
- `--enable-prompt-tokens-details`: Makes vLLM report `usage.prompt_tokens_details.cached_tokens` in the API response. This is how CAGE measures the prompt-cached ratio.
- `VLLM_CPU_KVCACHE_SPACE=10`: Allocates 10GB of RAM for KV cache blocks (CPU mode).
- `VLLM_GPU_MEMORY_UTILIZATION=0.9`: Uses 90% of GPU VRAM for KV cache + model weights (GPU mode).

### vLLM source code
The repository includes a vendored/forked copy of vLLM under `vLLM/`. This is for reference and for building custom CPU Docker images. The experiment runner does NOT import from this directory — it communicates with vLLM purely via HTTP.

### What "prefix caching" means at the vLLM level
When a request arrives, vLLM:
1. Tokenizes the prompt
2. Checks if any prefix of the token sequence has cached KV blocks (hash-based lookup)
3. If yes: skips prefill computation for the cached portion, only computes KV for new tokens
4. If no: computes full prefill

This is why Prefix Cache achieves 65.7% lower TTFT — vLLM skips recomputing KV for the shared system prefix and context passage.

---

## Infrastructure

### Docker Compose (CPU) — `docker-compose.yml`
- **Redis**: `redis:7-alpine` on port 6379
- **3 vLLM replicas**: ARM64 CPU image, ports 8001/8002/8003, 1GB KV cache each
- **Router**: Custom FastAPI container, port 9000, routes to replicas via prefix hash

### Docker Compose (GPU) — `docker-compose.gpu.yml`
- Same structure but uses `vllm/vllm-openai:latest` image with GPU reservations
- Model: `mistralai/Mistral-7B-Instruct` (default, should be updated for Phase 2)

### Kubernetes — `k8s/`
- `vllm-replica.yaml`: StatefulSet for vLLM replicas
- `router.yaml`: Deployment for the prefix-aware router
- `redis.yaml`: Redis deployment

### Terraform — `terraform/gcp/main.tf`
- Scaffolded for GCP Compute Engine with GPU instances
- Needs `terraform.tfvars` with project ID, region, and GPU type

---

## Configuration Files

### `configs/experiment/baseline.yaml`
Default experiment config: baseline type, num_queries, batch_size, evaluation flags, output directory.

### `configs/model/qwen3-4b.yaml`
Model-specific config: name, max_tokens, temperature, hardware requirements, vLLM dtype.

Available models: `qwen3-4b.yaml`, `qwen3-8b.yaml`, `qwen3-14b.yaml`, `qwen3-30b-a3b.yaml`, `qwen2.5-7b-instruct.yaml`

### `configs/dataset/squad_v2.yaml`
Dataset config: name, split, max_examples, seed, task type.

Available datasets: `squad_v2.yaml`, `hotpotqa.yaml`

---

## Plot Generation Pipeline

### `scripts/generate_publication_plots.py`
Generates 16 numbered plots from `aggregated_metrics.json` files:
- 01: Latency comparison bars
- 02: Throughput (QPS + tok/s) side-by-side
- 03: Quality metrics grouped bars
- 04: Latency breakdown (prefill vs decode stacked)
- 05: Speedup vs No Cache
- 06: Pareto frontier (latency vs faithfulness)
- 07: Radar chart
- 08: Heatmap (normalized)
- 09: Boxplots (trial variance)
- 10: Summary ranking table
- 11: Cache hit vs TTFT scatter
- 12: Quality-performance matrix
- 13: Overhead decomposition
- 14: Efficiency ranking
- 15: Context type impact
- 16: TTFT tail latency (p50 vs p95 range chart)

**Data flow**: Reads `analysis/phase1/results/*/aggregated_metrics.json` → writes PNGs to `analysis/phase1/images/`

### `scripts/generate_compact_figures.py`
Creates composite figures for the paper (AB split, CD split, grids). Uses `create_labeled_split()` to combine individual plots with panel labels.

---

## Data Files

### `analysis/phase1/results/<baseline>/aggregated_metrics.json`
The primary data file for each baseline. Contains:
```json
{
  "num_trials": 3,
  "performance": {
    "queries_per_second": {"mean": ..., "std": ..., "min": ..., "max": ..., "values": [...]},
    "avg_ttft_ms": {"mean": ..., "std": ..., ...},
    "avg_tpot_ms": {"mean": ..., "std": ..., ...},
    "avg_latency_ms": {"mean": ..., "std": ..., ...},
    "tokens_per_second": {"mean": ..., "std": ..., ...},
    "p50_ttft_ms": {"mean": ..., "std": ..., ...},
    "p95_ttft_ms": {"mean": ..., "std": ..., ...},
    ...
  },
  "quality": {
    "faithfulness": {"mean": ..., "std": ..., "values": [...]},
    "relevance": {"mean": ..., "std": ..., ...},
    "completeness_bertscore": {"mean": ..., "std": ..., ...},
    "completeness_rouge_l": {"mean": ..., "std": ..., ...}
  },
  "cache_telemetry": {
    "local_hit_ratio": {"mean": ..., ...},
    "remote_hit_ratio": {"mean": ..., ...},
    "miss_ratio": {"mean": ..., ...}
  },
  "retrieval": {
    "avg_hit": {"mean": ..., ...},
    "cache_rate": {"mean": ..., ...},
    "embedding_model": "intfloat/e5-large-v2",
    "reranker_model": "BAAI/bge-reranker-large"
  },
  "prompt_cache": {
    "overall_cached_prompt_ratio": {"mean": ..., ...},
    "num_requests_with_usage": {"mean": ..., ...}
  },
  "experiment": {
    "baseline": "prefix_cache",
    "model": "Qwen/Qwen3-4B",
    "dataset": "squad_v2",
    "num_queries": 50,
    "max_tokens": 100,
    "seed": 42
  },
  "baseline_config": { ... },
  "distributed": { ... }  // Only for distributed baseline
}
```

### `experiments/ir_index/`
Persisted FAISS index and document store. Built by `src/orchestration/ir.py` on first retrieval-baseline run.

---

## How Specific Baselines Work

### No Cache
1. vLLM server started WITHOUT `--enable-prefix-caching`
2. Gold passage from `example.context[0]` used as context
3. `format_qa_prompt(question, [gold_passage])` builds the prompt
4. Request sent to vLLM at :8000
5. Every request recomputes full prefill (no KV reuse)

### Prefix Cache
1. vLLM server started WITH `--enable-prefix-caching`
2. Same gold passage, same prompt format
3. First request computes full prefill
4. Subsequent requests with overlapping prefix tokens reuse cached KV blocks
5. The 68.4% cached ratio means ~68% of prompt tokens were served from cache on average

### RAG
1. vLLM server started WITHOUT `--enable-prefix-caching`
2. Gold passage is NOT used
3. FAISS index queried with `example.question` → returns top-3 passages
4. Reranker re-scores the 3 passages → final top-3 used as context
5. `format_qa_prompt(question, retrieved_passages)` builds the prompt
6. Request sent to vLLM — full recomputation every time
7. Retrieval + reranking adds ~12s to TTFT (explaining the 18,775ms)

### Redis Retrieval Cache Cold
1. Same as RAG but retrieval results are cached in Redis
2. Redis flushed before run → all queries miss the cache
3. After each query, results stored in Redis for potential future hits
4. In Phase 1 (cold): effectively identical to RAG (0% cache rate)

### Hybrid Cache Cold
1. vLLM server started WITH `--enable-prefix-caching`
2. Retrieval via FAISS + reranker (same as RAG)
3. Redis retrieval cache flushed (cold)
4. vLLM prefix cache is active → benefits from repeated prompt prefixes
5. Trial 1 is cold (no prefix cache warm), trials 2-3 benefit from warmed prefix cache
6. This explains the high variance: ±4,290ms TTFT std

### Hybrid Cache Warm
1. Same as Hybrid Cold BUT 50 warmup queries run first (excluded from metrics)
2. After warmup: both Redis cache (100% rate) and vLLM prefix cache are warm
3. Result: 2,791ms TTFT (close to Prefix Cache's 2,376ms)

### Distributed Router Replicated
1. 3 vLLM replicas started (ports 8001/8002/8003) WITH prefix caching
2. Router on port 9000 receives all requests
3. Router hashes each prompt prefix → routes to a specific replica
4. Gold passage used (not retrieval)
5. transfer_required_count = 0 (no KV tensors moved between replicas)
6. Replica distribution: 13/15/22 across the 3 replicas (from router snapshot)

---

## Prometheus Metrics (Optional)

`run_experiment.py` exposes Prometheus metrics if enabled:
- `cage_runner_requests_total`: Request counter by baseline
- `cage_runner_ttft_seconds`: TTFT histogram
- `cage_runner_latency_seconds`: Latency histogram
- `cage_runner_cached_prompt_ratio`: Cached prompt ratio histogram

These are optional and not used in the paper, but available for real-time monitoring during experiments.

---

## Test Suite

Located in `tests/`:
- `test_inference.py`: Tests VLLMAdapter request/response handling
- `test_ir.py`: Tests FAISS index building and retrieval
- `test_baselines.py`: Tests baseline config loading
- `test_router_integration.py`: Tests router request forwarding
- `test_vllm_integration.py`: Integration tests against a live vLLM server
- `test_data.py`: Tests dataset loading

Run with: `pytest tests/ -v`
