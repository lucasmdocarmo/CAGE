# CAGE Framework - Solution Description

**Last updated:** 2026-07-02 · **Status:** CURRENT (Phase 1 done; Phase 2 single-L4 clean re-run code-ready; Phase 3 deferred)
**Author:** Lucas Mariano do Carmo · **Institution:** Pontifícia Universidade Católica de Minas Gerais (PUC Minas) · **Contact:** lucas.mariano.carmo@gmail.com

> Companion deep-dive: [`TECHNICAL_ARCHITECTURE.md`](TECHNICAL_ARCHITECTURE.md). Authoritative
> references: [`KNOWLEDGE_BASE.md`](KNOWLEDGE_BASE.md), [`RUNBOOK.md`](RUNBOOK.md). Phase-2
> results: `phase2_archive/PHASE2_ANALYSIS.md`. Portuguese version: [`SOLUTION_DESCRIPTION.pt-BR.md`](SOLUTION_DESCRIPTION.pt-BR.md).

---

## 1. What is CAGE?

CAGE (Cache-Augmented Generation Evaluation) is a benchmarking framework that, for the first
time, **jointly** measures the two things every other tool measures only in isolation: LLM
**serving efficiency** (latency, throughput, KV-cache reuse) and **answer quality** (grounding,
faithfulness). It exists to answer one question rigorously: *when does reusing precomputed
key-value (KV) cache state, that is Cache-Augmented Generation (CAG), beat Retrieval-Augmented
Generation (RAG), and at what cost to answer quality?*

The gap it fills: RAG-evaluation frameworks (RAGAS, ARES) measure quality only; serving systems
(vLLM, SGLang) measure latency only. Nobody measures the trade between them per query, with
statistical significance. CAGE does, across a 9-family baseline taxonomy on a real inference
stack, and reports per-query Wilcoxon tests with Holm correction.

**Core thesis:** a distributed contextual prefix cache is superior to RAG for static or
semi-static knowledge bases, achieving lower time-to-first-token while maintaining or improving
grounding.

---

## 2. Architecture

CAGE is the orchestration and evaluation layer. **vLLM** is the inference engine. They
communicate exclusively over HTTP (the OpenAI-compatible `/v1/completions` API), so CAGE never
imports vLLM internals and can benchmark any compatible server.

```
Workload (CLI)  →  CAGE Orchestrator  →  vLLM server(s)  →  Telemetry + Quality Evaluator
 run_experiment.py   baselines.py / ir.py    HTTP API        vllm_telemetry.py + quality.py
                     compression.py          (+ router for     performance.py
                                              distributed)
```

Key components (see `TECHNICAL_ARCHITECTURE.md` for the module-by-module deep dive):
- `src/data/loader.py` - dataset loading into `CAGExample` objects. Eleven loaders: SQuAD v2, HotpotQA, Qasper, TriviaQA, Natural Questions, MuSiQue, CRAG, ShareGPT (QA/serving), plus HumanEval, MBPP, hpc_code (code). CRAG carries a gold answer alongside retrieved candidate docs (RAG-fair); ShareGPT is a serving-workload trace with no gold answer (reference-only, `no_gold_answer=True`).
- `src/inference/vllm_adapter.py` - HTTP client with streaming TTFT measurement and usage telemetry. (Also `gemini_adapter.py`, `ollama_adapter.py` for alt backends.)
- `src/orchestration/baselines.py` - baseline-family definitions and per-baseline config.
- `src/orchestration/ir.py` - FAISS dense retrieval (e5-large-v2) + BGE reranker for RAG-family baselines.
- `src/orchestration/compression.py` - LLMLingua-2 client-side prompt compression (the `compressed_rag` arm).
- `src/orchestration/redis_cache.py` - Redis retrieval-artifact cache.
- `src/orchestration/router.py` - FastAPI prefix-hash router for the distributed cluster.
- `src/orchestration/cache_manager.py` - KV-cache distribution policies (replicated now; sharded/offload for Phase 3).
- `src/evaluation/quality.py` - LettuceDetect grounding (primary), NLI faithfulness, F1/EM, ROUGE-L, context relevance.
- `src/evaluation/performance.py` - latency/throughput/percentiles + cache telemetry aggregation.
- `src/monitoring/vllm_telemetry.py` - continuous GPU/KV/serving telemetry sampler (cage-stats).
- `scripts/run_experiment.py` - main experiment runner (CLI entry point).

---

## 3. Baseline taxonomy (9 families + a 2×2 compression axis)

CAGE defines **nine baseline families** along two axes, context source (gold vs retrieved) and
reuse policy, plus orthogonal **compression** and **speculative** extensions.

**Core families:**
1. **no_cache** - gold passage, no prefix caching. Full prefill every request (worst-case control).
2. **prefix_cache** - gold passage, vLLM prefix caching on. Shared prompt prefixes reuse KV blocks.
3. **rag** - FAISS retrieval + BGE rerank, gold passage not used; no prefix caching.
4. **redis** - RAG with retrieval artifacts cached in Redis (cold/warm).
5. **hybrid** - retrieval + prefix caching (+ Redis), cold and warm variants.
6. **distributed** - N vLLM replicas behind a prefix-hash router (Phase 3 for real cross-node KV transfer).
7. **speculative** - speculative decoding (ngram, EAGLE-3, and Phase-3 MTP) on top of a context strategy.
8. **compressed_rag** - RAG + LLMLingua-2 prompt compression (client-side, ~2× fewer prompt tokens).
9. **compressed_cag** - prefix cache + FP8 KV-cache quantization (server launch lever, ~2× smaller KV).

**The 2×2 compression axis** (read it DOWN for CAG vs RAG, ACROSS for full vs compressed):

| | full | compressed |
|---|---|---|
| **CAG** (gold context) | prefix_cache / cag_full | compressed_cag (FP8 KV) |
| **RAG** (retrieved context) | rag / rag_full | compressed_rag (LLMLingua-2) |

---

## 4. Metrics

**Quality (PRIMARY: LettuceDetect grounding):**
- **Grounding (LettuceDetect, primary):** token/span-level hallucination detection via a ModernBERT model. `grounding_score = 1 − hallucinated_span_ratio`.
- **Faithfulness (secondary):** claim-level NLI entailment of the answer against the context.
- **F1 / Exact Match:** standard QA correctness.
- **ROUGE-L:** longest-common-subsequence F1.
- **Context relevance:** question/context embedding similarity (diagnostic only).
- **BERTScore: deprecated** (non-discriminative across baselines; kept only as a negative control).

**Serving (via cage-stats + pynvml):** QPS, tokens/sec, TTFT, TPOT, end-to-end latency (avg + p50/p95/p99), prefix-hit ratio, KV-cache utilization, GPU memory/power/temperature, and (when speculative is on) acceptance rate.

**Telemetry is live-only (hard guarantee).** The synthetic/mock telemetry path has been removed from the code. `capture_snapshot` resolves in this order: in-process `cage_stats.api`, then the `cage-stats --once --json` CLI, then a dependency-free stdlib `/metrics` scraper. If no vLLM server is reachable (or the scraped payload contains no `vllm:` series) the sampler returns `None`; it never fabricates zeros or synthetic values. This is an enforced invariant, not a convention.

**Statistical layer (`scripts/statistical_tests.py`):** per-query **Wilcoxon** signed-rank tests vs a reference baseline, **Holm** multiple-comparison correction, **Cliff's delta** effect size, and **bootstrap** confidence intervals.

---

## 5. Results

### Phase 1 (CPU, protocol validation, relative only)
Setup: Qwen3-4B, SQuAD v2, Apple M4 Pro CPU, 50 queries × 3 trials × 7 baselines. Absolute
CPU latencies are not generalizable; rankings are. Prefix cache wins (−37.4% latency, −65.7%
TTFT, equal quality); RAG is slowest and loses faithfulness; distributed shows a 7.6× p95/p50
tail spread; BERTScore is non-discriminative.

### Phase 2 (single NVIDIA L4 GPU, the production-relevant baseline) - clean re-run CODE-READY
Setup: **Qwen3-8B, vLLM 0.11.0, SQuAD v2, single L4 (24 GB), greedy (T=0)**, across 8 of 9
families (distributed deferred to Phase 3). Significance vs `no_cache`, Holm-corrected. Primary
metric = grounding.

> **Result status:** the original Phase-2 run was superseded (invalid baselines) and a clean
> re-run is code-ready but not yet re-executed, so the numbers below are directional from the
> earlier run and must not be treated as final/validated until the re-run lands. **Query count is
> locked at 500 × 3:** `scripts/run_phase2.sh` uses `NUM_QUERIES=500`, three trials.

| Finding | Serving | Quality | Verdict |
|---|---|---|---|
| **Prefix caching** | TTFT −3.3% (p=1.2e-11) | grounding identical (0.938) | lossless |
| **FP8 KV (compressed_cag)** | KV halved | grounding 0.936 vs 0.938 (n.s.) | lossless |
| **RAG** vs gold CAG | TTFT +87% (p=5e-17) | faithfulness −24.7% (p=8.8e-05); grounding 0.66 | costs both axes here |
| **EAGLE-3 speculative** | TPOT −41% (54→32 ms), latency −32.5% (p=5e-11) | grounding unchanged | lossless, biggest win |

Honest caveats: on SQuAD v2 the gold passage is the ideal context, so CAG dominates RAG (a
RAG-favorable dataset is a Phase-3 need); `compressed_rag` was invalid in this run because the
compression treatment never fired (now fixed in code, rerun in Phase 3); `hybrid_warm` used
unpaired statistics. Full analysis: `phase2_archive/PHASE2_ANALYSIS.md`. Cost ~$3.1; all GCP
infra has been torn down to $0 and the data is archived locally.

---

## 6. Tech stack

- **Inference:** vLLM 0.11.0 (pinned), OpenAI-compatible HTTP API; `--enforce-eager` on the L4.
- **Models:** Qwen3-4B (Phase 1 CPU), Qwen3-8B (Phase 2); Phase-3 candidates Qwen3-14B/32B and DeepSeek-V2-Lite (for MTP). Configs in `configs/model/`.
- **Datasets:** eleven loaders registered in `src/data/loader.py` - SQuAD v2, HotpotQA, Qasper, TriviaQA, Natural Questions, MuSiQue, CRAG, ShareGPT (QA/serving), plus HumanEval, MBPP, hpc_code (code). All QA loaders shuffle(seed) before select for trial-independence. CRAG (gold answer + retrieved candidate docs, RAG-fair) and ShareGPT (serving-workload trace, no gold answer) are wired end-to-end (registry + `run_experiment.py --dataset` + `scripts/download_datasets.py`).
- **Retrieval:** FAISS `IndexFlatIP` + `intfloat/e5-large-v2` embeddings + `BAAI/bge-reranker-large`.
- **Compression:** LLMLingua-2 (prompt, client-side) and FP8 KV-cache (server `--kv-cache-dtype fp8`).
- **Speculative decoding:** ngram + EAGLE-3 (`AngelSlim/Qwen3-8B_eagle3`) via `--speculative-config`.
- **Quality models:** LettuceDetect (ModernBERT) grounding; DeBERTa-mnli NLI faithfulness.
- **Caching:** Redis (retrieval artifacts) + vLLM built-in prefix cache (KV blocks).
- **Telemetry:** cage-stats (`--vllm-telemetry`) + pynvml.
- **Infrastructure:** GCP (g2-standard-8 + L4), Terraform, durable GCS bucket, plus a log-preservation + safe-teardown tool suite. Docker Compose / Kubernetes manifests exist for local and cluster use.
- **Runtime:** Python 3.12, torch, transformers, datasets, sentence-transformers, faiss-cpu, llmlingua.

---

## 7. Status and next steps

- **Phase 1 (CPU): DONE** - protocol validated; this is the state the dissertation currently reports (stochastic decoding, temperature 0.7).
- **Phase 2 (single L4 GPU): clean re-run CODE-READY** - greedy (T=0) quality + serving axes wired with statistics; the original run was superseded and the re-run is not yet re-executed, so its numbers are not final. Infra at $0, data local. Query count locked at 500 × 3.
- **Phase 3 (multi-node HPC): DEFERRED** - real cross-node KV-tensor transfer via a vLLM KV connector (LMCache/NIXL) with a sharded context policy (replacing today's analytic/simulated model), disaggregated prefilling, broader speculative decoding (MTP via DeepSeek-V2-Lite), and multi-trial confidence intervals. A RAG-favorable dataset need is now partially met: **CRAG** (gold answer + retrieved candidate docs, RAG-fair) and **ShareGPT** (serving-workload trace) are wired end-to-end and available for the Phase-3 workload. Cross-node transfer stays analytic/simulated until real RDMA lands. Plan: [`PHASE3_PLAN.md`](PHASE3_PLAN.md); models: `docs/PHASE3_MODELS.md`.

### Recent changes and fixes (this cycle)
- **Generation determinism:** temperature 0.0 + `stop=["\n"]` to stop Qwen3 chain-of-thought leakage.
- **vLLM stability:** EngineCore GPU-leak kill on restart; array-built server args; FP8 × prefix-cache gate.
- **Metric fixes:** `retrieval_hit` now uses a normalized-text fallback (was a false zero); `completeness_bertscore` returns None on empty references (was a negative sentinel).
- **Compression validity:** `llmlingua` added to requirements, and compression is now **strict by default** so the `compressed_rag` arm can never silently no-op again. If LLMLingua-2 is unavailable it raises rather than passing through. The live opt-out is `CAGE_ALLOW_NO_COMPRESSION`; `CAGE_DISABLE_COMPRESSION` disables compression entirely (pass-through).
- **Operations:** a log-preservation suite (`collect_logs.sh`, `log_sync_daemon.sh`, `gcp_shutdown_hook.sh`) and a fail-closed `teardown_vm.sh` that verifies logs reached GCS before deleting a VM.

---

## 8. Credits / prior work

CAGE builds on and positions against a body of prior work; the bib keys below (in `Main.bib`) credit that lineage.

- **Serving engine instrumented:** vLLM / PagedAttention (`kwon2023efficient`). Related cache-aware serving that CAGE positions against: SGLang/RadixAttention (`zheng2024sglang`), DistServe (`zhong2024distserve`), Mooncake (`qin2024mooncake`), LMCache (`lmcache2024`), CacheBlend (`cacheblend2025`), CacheGen (`cachegen2024`, arXiv 2310.07240).
- **Primary quality metric:** LettuceDetect span-level hallucination detection (`lettucedetect2025`).
- **RAG-evaluation lineage** CAGE co-measures against: RAGAS (`espejel2023ragas`), ARES (`ares2024`), plus the CRAG factual-QA benchmark (`yang2024crag`).
- **The "compression carries a measured quality cost" spine:** The Pitfalls of KV Cache Compression (`chen2025pitfalls`) and SCBench (`li2025scbench`, ICLR 2025, Microsoft and University of Surrey, arXiv 2412.10319), the closest cache-plus-quality prior work (no grounding metric, no per-method serving latency).
- **The CAG-vs-RAG decision CAGE operationalizes:** "Don't Do RAG" / cache-augmented generation (`yu2024dontdorag`).
- **Compression method run:** LLMLingua-2 for text-side prompt compression (`llmlingua2`).

(Bib keys are mirrored verbatim from the dissertation's `Main.bib` for `\cite` compatibility; a few keys deliberately differ from the paper's first author and are kept unchanged.)
