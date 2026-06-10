# Phase 1 Result Analysis

This document interprets the completed corrected Phase 1 results from `analysis/phase1/results` and the generated summaries in `analysis/phase1/plots`.

The focus here is not run mechanics but what the numbers actually say, which comparisons are safe, and where caveats remain necessary.

## 1. Main aggregate result table

Source: the seven baseline `aggregated_metrics.json` files and `analysis/phase1/plots/latest_metrics_summary.csv`

| Baseline | Avg latency ms | Avg TTFT ms | QPS | Faithfulness | Relevance | BERTScore | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| `no_cache` | 16006.49 ± 711.18 | 6919.05 ± 112.26 | 0.06061 ± 0.00250 | 0.5703 ± 0.0565 | 0.5051 | 0.3279 | control |
| `prefix_cache` | 10015.44 ± 883.57 | 2375.60 ± 275.01 | 0.09592 ± 0.00774 | 0.5703 ± 0.0565 | 0.5051 | 0.3279 | prompt cache active |
| `rag` | 27270.08 ± 283.39 | 18775.42 ± 144.42 | 0.03329 ± 0.00056 | 0.5044 ± 0.0228 | 0.5251 | 0.3243 | retrieval hit `0.98` |
| `redis_retrieval_cache_cold` | 26853.24 ± 39.16 | 18578.89 ± 37.29 | 0.03416 ± 0.00003 | 0.5504 ± 0.0397 | 0.5251 | 0.3243 | retrieval hit `0.98`, cache rate `0.0` |
| `hybrid_retrieval_cache_cold` | 15512.50 ± 4172.73 | 6000.61 ± 4289.88 | 0.05887 ± 0.01197 | 0.5056 ± 0.0222 | 0.5251 | 0.3243 | retrieval hit `0.98`, cache rate `0.0` |
| `hybrid_retrieval_cache_warm` | 13268.59 ± 893.24 | 2791.17 ± 65.56 | 0.06739 ± 0.00391 | 0.5123 ± 0.0267 | 0.5251 | 0.3245 | retrieval hit `0.98`, cache rate `1.0`, warmup `50` |
| `distributed_router_replicated` | 18491.77 ± 371.63 | 5278.74 ± 638.25 | 0.05248 ± 0.00110 | 0.6359 ± 0.0775 | 0.5051 | 0.3267 | 3 routed replicas, no transfer required |

## 2. Metric rankings

Aggregate ranking order:

- Latency: `prefix_cache` > `hybrid_retrieval_cache_warm` > `hybrid_retrieval_cache_cold` > `no_cache` > `distributed_router_replicated` > `redis_retrieval_cache_cold` > `rag`
- TTFT: `prefix_cache` > `hybrid_retrieval_cache_warm` > `distributed_router_replicated` > `hybrid_retrieval_cache_cold` > `no_cache` > `redis_retrieval_cache_cold` > `rag`
- QPS: `prefix_cache` > `hybrid_retrieval_cache_warm` > `no_cache` > `hybrid_retrieval_cache_cold` > `distributed_router_replicated` > `redis_retrieval_cache_cold` > `rag`
- Faithfulness: `distributed_router_replicated` > `no_cache` = `prefix_cache` > `redis_retrieval_cache_cold` > `hybrid_retrieval_cache_warm` > `hybrid_retrieval_cache_cold` > `rag`

Important caution:

- these rankings are numerically true for the aggregates
- they are not all equally safe to interpret as a single league table because some baselines still carry warm/cold or cross-trial state caveats

## 3. Strongest clean finding: `prefix_cache` vs `no_cache`

Relative to `no_cache`, `prefix_cache` changes are:

- latency: `-37.43%`
- TTFT: `-65.67%`
- QPS: `+58.26%`
- faithfulness: `0.00%`

Why this is still the strongest practical result:

- quality is unchanged
- latency and TTFT improvements are large
- the direction of the gain is unmistakable
- this is the cleanest “same context, same task, faster serving” result in the run

Important caveat:

- the aggregate includes some cross-trial prompt-cache warming because the server was not restarted between trials
- trial 1 TTFT was `2738.77 ms` versus aggregate `2375.60 ms`
- this means the exact cold-trial magnitude is slightly optimistic in the aggregate

Even with that caveat, `prefix_cache` is clearly the dominant single-node performance result in Phase 1.

## 4. Retrieval family analysis: `rag`, `redis_retrieval_cache_cold`, `hybrid_retrieval_cache_cold`, `hybrid_retrieval_cache_warm`

### 4.1 Retrieval is working; the problem is cost, not failure

All retrieval-based baselines show a consistent aggregated retrieval hit rate of `0.98`:

- `rag`
- `redis_retrieval_cache_cold`
- `hybrid_retrieval_cache_cold`
- `hybrid_retrieval_cache_warm`

That matters because it rules out a simplistic interpretation that retrieval is failing outright.

The performance problem is instead:

- retrieval overhead is large on this stack
- reranking/retrieval plus longer prompt construction are expensive
- these baselines are slower than the clean gold-context control unless the prompt-cache layer contributes significantly

### 4.2 `redis_retrieval_cache_cold` vs `rag`

Relative to `rag`, `redis_retrieval_cache_cold` changes are:

- latency: `-1.53%`
- TTFT: `-1.05%`
- QPS: `+2.62%`
- faithfulness: `+9.13%`

Interpretation:

- Redis retrieval-artifact caching helps only marginally on performance in this Phase 1 setup
- it is still drastically slower than `no_cache` and `prefix_cache`
- the semantic win here is mostly honesty and cleanliness, not dramatic acceleration

What this does support:

- the Redis flush fix worked
- the baseline is now an honest cold retrieval-cache run

What this does not support:

- any claim that the current Redis path is acting like a raw distributed KV-cache accelerator

### 4.3 `hybrid_retrieval_cache_cold` vs retrieval-only baselines

Relative to `rag`, `hybrid_retrieval_cache_cold` changes are:

- latency: `-43.12%`
- TTFT: `-68.04%`
- QPS: `+76.85%`
- faithfulness: `+0.25%`

Relative to `redis_retrieval_cache_cold`, `hybrid_retrieval_cache_cold` changes are:

- latency: `-42.23%`
- TTFT: `-67.70%`
- QPS: `+72.34%`
- faithfulness: `-8.14%`

Interpretation:

- hybrid cold is much faster than pure retrieval baselines
- the retrieval stage still happens and stays cold
- the speedup is coming from prompt-prefix reuse, not retrieval-cache warming

Critical caveat:

- the aggregate `hybrid_retrieval_cache_cold` numbers are not fully cold at the three-trial level
- trial 1 was genuinely cold, but trials 2 and 3 inherited warmed prompt-cache state
- that is why the aggregate TTFT collapses from trial 1 `12067.40 ms` to aggregate `6000.61 ms`

Therefore:

- the direction of the result is informative
- the aggregate cold number should not be treated as a strict cold baseline without qualification

### 4.4 `hybrid_retrieval_cache_warm` is the best retrieval-family speed result

Relative to `hybrid_retrieval_cache_cold`, `hybrid_retrieval_cache_warm` changes are:

- latency: `-14.47%`
- TTFT: `-53.49%`
- QPS: `+14.46%`
- faithfulness: `+1.32%`

This is the intended warm result:

- retrieval cache rate `1.0`
- 50 warmup requests excluded from metrics
- prompt cached ratio `0.8923`

Interpretation:

- once both retrieval artifacts and the prompt prefix are warm, the retrieval family becomes much more competitive
- warm hybrid is still slower than `prefix_cache`, but it is much closer to the control family than `rag` or `redis_retrieval_cache_cold`

## 5. Distributed result interpretation

### 5.1 What `distributed_router_replicated` does well

The distributed replicated baseline is significant because it validates the corrected distributed claim set:

- 3 isolated replicas actually received traffic
- the router snapshot reports `strategy = hash`
- `tokenizer_name = Qwen/Qwen3-4B`
- `tokenization_mode = model_tokenizer`
- transfer was not required

This is not a simulated transfer benchmark.

### 5.2 Performance trade-off

Relative to `no_cache`, `distributed_router_replicated` changes are:

- latency: `+15.53%`
- TTFT: `-23.71%`
- QPS: `-13.40%`
- faithfulness: `+11.52%`

Interpretation:

- it improves TTFT relative to `no_cache`
- it does not improve full-request latency or throughput
- it preserves or even slightly improves answer quality, but that should not be over-attributed to routing itself because the context source remains favorable

Relative to `prefix_cache`, it is clearly weaker on serving efficiency.

Best interpretation:

- this is a successful distributed routing/orchestration baseline
- it is not yet a throughput winner

## 6. Pareto observations from generated summaries

Source: `analysis/phase1/plots/pareto_optimal_baselines.csv`

Observed Pareto-optimal baselines:

- `prefix_cache` is Pareto-optimal for:
  - latency vs BERTScore
  - TTFT vs BERTScore
  - QPS vs faithfulness
  - latency vs relevance
- `distributed_router_replicated` is also Pareto-optimal for:
  - QPS vs faithfulness
- `hybrid_retrieval_cache_warm` is Pareto-optimal for:
  - latency vs relevance

Interpretation:

- `prefix_cache` is the dominant overall practical baseline
- `distributed_router_replicated` survives on the Pareto frontier only because it trades throughput for higher faithfulness
- `hybrid_retrieval_cache_warm` survives on the relevance/latency frontier because it keeps retrieval-family relevance while being much faster than `rag` and `redis_retrieval_cache_cold`

## 7. Which findings are safe to report

Safe statements:

- `prefix_cache` is the best overall performance baseline in corrected Phase 1
- the corrected Redis baseline is still far slower than the control family
- the retrieval family works, but its cost is high on this stack
- warm hybrid is substantially better than cold retrieval baselines
- the distributed replicated baseline is real and no longer depends on simulated transfer in the core suite

Unsafe or overstated statements:

- “all baselines are directly comparable in one strict ranking”
- “hybrid cold is fully cold across all three trials”
- “the distributed replicated baseline proves distributed KV-cache speedups”
- “the current Redis baseline demonstrates KV-block serving”

## 8. Final interpretation

The Phase 1 result story is now much cleaner than before, and it has a clear center:

- the most credible and practically useful win is `prefix_cache`
- `rag` and `redis_retrieval_cache_cold` are both functioning but too slow to compete on this setup
- `hybrid_retrieval_cache_warm` is promising within the retrieval family, but `hybrid_retrieval_cache_cold` needs a caveat because later trials are no longer fully cold
- `distributed_router_replicated` is a legitimate systems result, but not a serving-efficiency winner

If only one conclusion is carried forward from Phase 1, it should be this:

> Native prefix caching is the strongest, cleanest, and most defensible acceleration mechanism in the corrected Phase 1 benchmark.
