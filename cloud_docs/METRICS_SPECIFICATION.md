> <!-- CAGE-DOC-STATUS -->
> **⚠️ STATUS: PARTIALLY STALE (2026-06-09).** Useful background, but some commands,
> CLI flags, paths, and metric numbers here predate the June 2026 fixes. In
> particular, `run_experiment.py` has **no** `--phase`/`--all-baselines`/`--trials`/`--queries`
> flags (use `--baseline`, `--num-trials`, `--num-queries`), and pre-fix metric numbers
> (faithfulness 0.570, BERTScore 0.324) are now obsolete. For runnable commands use
> [`RUNBOOK.md`](RUNBOOK.md); for current metrics see [`KNOWLEDGE_BASE.md`](KNOWLEDGE_BASE.md).

# CAGE Metrics Specification

**Last Updated:** 2026-04-08

This document specifies all metrics captured by CAGE, how they are computed, and how to interpret them.

---

## 1. Performance Metrics

### Time to First Token (TTFT)
- **Definition:** Time from request submission to the first output token.
- **Capture:** Streaming SSE mode. `time.perf_counter()` at request start and at first `data:` chunk arrival.
- **Implementation:** `src/inference/vllm_adapter.py`
- **Reported:** avg_ttft_ms, p50_ttft_ms, p95_ttft_ms, p99_ttft_ms
- **Interpretation:** Lower is better. Directly affected by prefix caching (cached prefill is skipped).

### Time Per Output Token (TPOT)
- **Formula:** `(total_time_ms - ttft_ms) / (num_tokens - 1)`
- **Implementation:** `src/evaluation/performance.py`
- **Reported:** avg_tpot_ms, p50_tpot_ms, p95_tpot_ms, p99_tpot_ms
- **Interpretation:** Lower is better. Measures sustained decode speed. Less affected by caching.

### End-to-End Latency
- **Formula:** `request_end - request_start` (wall clock)
- **Equals:** TTFT + (TPOT × num_tokens)
- **Reported:** avg_latency_ms, p50/p95/p99_latency_ms

### Queries Per Second (QPS)
- **Formula:** `total_successful_requests / total_experiment_time_seconds`
- **Interpretation:** Higher is better. Measures system throughput capacity.

### Tokens Per Second (TPS)
- **Formula:** `total_output_tokens / total_experiment_time_seconds`
- **Interpretation:** Higher is better. More representative than QPS for varying response lengths.

---

## 2. Quality Metrics

### Faithfulness (NLI-based)
- **Definition:** Proportion of generated answer claims entailed by the context.
- **Method:** Answer split into claims → each checked via NLI model against context → mean entailment probability.
- **Implementation:** `src/evaluation/quality.py`
- **Scale:** 0.0–1.0 (higher is better)
- **Phase 1 range:** 0.504 (RAG) to 0.636 (Distributed)

### Relevance (Embedding Similarity)
- **Definition:** Cosine similarity between question and context embeddings.
- **Formula:** `max(cosine_sim(embed(question), embed(context_i)))`
- **Model:** Sentence-BERT (all-MiniLM-L6-v2 or similar)
- **Scale:** 0.0–1.0
- **Phase 1 range:** 0.505 (gold baselines) to 0.525 (retrieval baselines)

### BERTScore
- **Definition:** Token-level soft F1 between generated and reference answers using contextual embeddings.
- **Formula:** `2 × Precision × Recall / (Precision + Recall)` where Precision/Recall use max cosine similarity across token pairs.
- **Scale:** 0.0–1.0
- **Phase 1 note:** Near-constant (0.324–0.328). Not discriminative for this task.

### ROUGE-L
- **Definition:** Longest common subsequence F1 between generated and reference answers.
- **Scale:** 0.0–1.0

---

## 3. Cache Metrics

### Prompt-Cached Ratio
- **Definition:** Fraction of prompt tokens served from vLLM's prefix cache.
- **Source:** vLLM response `usage.prompt_tokens_details.cached_tokens`
- **Requires:** vLLM flag `--enable-prompt-tokens-details`
- **Formula:** `cached_prompt_tokens / total_prompt_tokens`
- **Phase 1 values:** Prefix Cache 68.4%, Hybrid Cold 75.6%, Hybrid Warm 89.2%

### Cache Hit Ratios
- **Local Hit Ratio:** `local_cache_hits / total_requests`
- **Remote Hit Ratio:** `remote_cache_hits / total_requests` (0 in Phase 1 — no real transfers)
- **Miss Ratio:** `cache_misses / total_requests`

### Retrieval Metrics (for RAG/Hybrid baselines)
- **Retrieval Hit Rate:** Fraction of queries where a relevant document appears in top-K results.
- **Phase 1 value:** 98% across all retrieval baselines.
- **Cache Rate:** Fraction of retrieval queries served from Redis cache. 0% (cold), 100% (warm).
- **Embedding Model:** intfloat/e5-large-v2
- **Reranker:** BAAI/bge-reranker-large

---

## 4. Aggregation

For multi-trial experiments, each metric is aggregated as:
- **mean:** Average across trials
- **std:** Standard deviation
- **min/max:** Trial extremes
- **values:** Raw per-trial values array

Percentiles (p50/p95/p99) use `numpy.percentile()` within each trial, then aggregated across trials.

---

## 5. Phase 1 Value Ranges

| Metric | Range | Notes |
|---|---|---|
| QPS | 0.033–0.096 | CPU-bound; GPU will be 10–100× higher |
| TTFT | 2,376–18,775 ms | Prefix Cache lowest; RAG highest |
| TPOT | 76–132 ms | Prefix Cache lowest; Distributed highest |
| Latency | 10,015–27,270 ms | Prefix Cache lowest; RAG highest |
| Faithfulness | 0.504–0.636 | RAG lowest; Distributed highest |
| BERTScore | 0.324–0.328 | Not discriminative |
| Prompt-cached ratio | 0.68–0.89 | Prefix Cache to Hybrid Warm |
