# CAGE — Feature Map (objective · value · delivery · comparison)

> In-depth map of **what every feature in the codebase is for**: its objective, the value it
> delivers, its delivery status, and the published work it compares to / is measured against.
> So no feature or code path is unexplained. Pairs with [`COMPARISON_MATRIX.md`](COMPARISON_MATRIX.md)
> (novelty), [`DEV_BACKLOG.md`](DEV_BACKLOG.md) (what's left), [`RELATED_WORK_COMPRESSION.md`](RELATED_WORK_COMPRESSION.md).
>
> Status legend: ✅ done & verified · ◐ partial · ⏳ cluster-validate (GPU-only) · ○ planned.
> Updated 2026-06-09 after the P0/P1/P2 implementation pass.

---

## 1. Baselines — `src/orchestration/baselines.py`

The unit of comparison. Each is a cache/retrieval policy held to identical workload + metrics.

| Baseline | Objective | Value delivered | Status | Compares to |
|---|---|---|---|---|
| `no_cache` | Worst-case control: full prefill every request | Reference point for all speedups | ✅ | the universal RAG/serving baseline |
| `prefix_cache` | Single-node KV prefix reuse (vLLM) | Isolates the gain from reusing a stable prefix | ✅ | vLLM/PagedAttention `kwon2023efficient`; CAG `yu2024dontdorag` |
| `rag` | Standard FAISS retrieval, no reuse | The retrieval baseline everyone reports against | ✅ | Lewis `lewis2020retrieval`; RAGAS/RAGBench evals |
| `redis` | Retrieval-artifact cache (cold) | Measures retrieval-cache value separately from KV reuse | ✅ | RAGCache `ragcache2024` (KV) vs artifact-cache here |
| `hybrid` (cold/warm) | Retrieval + native prefix caching | Production-realistic CAG↔RAG mix | ✅ | TurboRAG `chen2024turborag`; Self-Route `li2024selfroute` |
| `distributed` (replicated) | Router-mediated multi-replica prefix routing | Tests locality/routing across replicas | ◐ (routing real; transfer simulated) | DistServe `zhong2024distserve`; Mooncake `qin2024mooncake` |
| `speculative` | Speculative decoding effect on CAG | TPOT reduction interaction | ◐ (launch-config wired; acceptance via /metrics ○) | vLLM spec-decode |
| **`compressed_rag`** | Text-compress retrieved docs before prompting | The RAG-side compression arm of the 2×2 | ✅ (LLMLingua; graceful fallback) | RECOMP `recomp2024`, LongLLMLingua `longllmlingua2024`, CompAct |
| **`compressed_cag`** | KV-compress the cached context (fp8 / MLA) | The CAG-side compression arm; shrinks transfer bytes | ⏳ (fp8 flag + MLA config wired) | CacheGen `cachegen2024`, SnapKV `snapkv2024`, MLA/DeepSeek-V2 |

## 2. Evaluation metrics

| Metric (file) | Objective | Value | Status | Compares to |
|---|---|---|---|---|
| **Grounding / hallucination** — `evaluation/quality.py` (LettuceDetect) | Span-level detection of unsupported answer tokens | PRIMARY quality signal; catches retrieval-induced hallucination | ✅ | RAGTruth `ragtruth2024`, LettuceDetect `lettucedetect2025` |
| **NLI faithfulness (claim-level)** — `quality.py` | Claim-by-claim entailment vs context (max-over-docs) | Secondary faithfulness; cross-checks grounding | ✅ | RAGAS `espejel2023ragas`, ARES `ares2024` |
| **Context relevance** — `quality.py` | Retriever diagnostic (question↔context) | Honestly labeled retriever signal, not answer quality | ✅ | RAGChecker retrieval metrics `ragchecker2024` |
| **BERTScore (rescaled) / ROUGE-L / F1 / EM** — `quality.py` | Overlap & QA correctness | Standard QA comparability | ✅ | SQuAD/HotpotQA conventions |
| **TTFT / TPOT / latency / QPS** — `performance.py` | Serving behavior | The systems axis eval-frameworks lack | ✅ | DistServe `zhong2024distserve`, vLLM |
| **GPU telemetry** — `performance.py` (`GPUMetricsTracker`) | VRAM/util/power/PCIe during the run | Phase-2 memory-pressure evidence | ✅ wired | nvidia-smi-equivalent |
| **Cache telemetry** — `performance.py` + vLLM `cached_tokens` | Hit/miss, prompt-cached ratio | Quantifies reuse | ✅ | RAGCache hit-rate framing |
| **Compression metrics** — `evaluation/compression.py` | `compression_ratio`, `kv_cache_bytes` (MHA/GQA/MLA), `transfer_bytes` | x-axis for quality/latency-vs-compression Paretos | ✅ | RECOMP/LongLLMLingua (ratio); CacheGen (KV bytes) |

## 3. Retrieval — `src/orchestration/ir.py`

| Feature | Objective | Status | Compares to |
|---|---|---|---|
| FAISS `IndexFlatIP` (exact) + e5 `query:`/`passage:` prefixes | Correct, reproducible dense retrieval | ✅ (prefix fix + stale-index guard) | dense-retrieval norm; fixes the OOD-embedding pitfall |
| Cross-encoder reranker (bge) | Optional reranking of hits | ✅ | standard RAG rerank |
| Stale-index guard | Warns when an index predates the prefix fix | ✅ | (correctness) |

## 4. Orchestration / serving

| Feature (file) | Objective | Status | Compares to |
|---|---|---|---|
| Prefix-aware router — `router.py` | Route by prefix hash to maximize per-replica locality | ✅ (routing) | Mooncake/DistServe routing |
| Simulated KV transfer — `cache_manager.py` | Model cross-node transfer cost | ◐ simulated | to be replaced by real vLLM KV connector |
| **Real KV transfer** (vLLM `--kv-transfer-config` + LMCache/NIXL) | Measure true transfer bytes/latency | ○ planned (cluster) | CacheBlend/CacheGen/Mooncake |
| Redis retrieval cache — `redis_cache.py` | Centralized retrieval-artifact cache baseline | ✅ | (baseline) |
| vLLM HTTP adapter — `vllm_adapter.py` | Streaming TTFT, usage/`cached_tokens`, kv-transfer-params read | ✅ (non-stream TTFT now honest) | — |

## 5. Experiment protocol — `scripts/run_experiment.py`

| Control | Objective | Status | Why it matters |
|---|---|---|---|
| `--num-trials` + **seeded resampling** | Independent trials for statistics | ✅ | each trial now draws a different reproducible sample |
| **`--context-source {auto,gold,retrieved}`** | Equalize context across arms | ✅ | removes the gold-vs-retrieved confound |
| **Disjoint warmup** | Warm cache without leaking measured queries | ✅ | fixes warm-hybrid leakage |
| **`--reset-cache-between-trials`** | Cold-start-per-trial via `/reset_prefix_cache` | ✅ | controlled cache state (needs `VLLM_SERVER_DEV_MODE=1`) |
| `--compress-method/-ratio`, `--kv-cache-dtype` | Drive the compression axis from the CLI | ✅ | the 2×2 knobs |
| `statistical_tests.py` | Per-query Wilcoxon + Holm + bootstrap CIs | ✅ standalone (aggregation hook ⏳) | rigor most cited works lack |

## 6. Datasets — `src/data/loader.py`

| Dataset | Objective | Status | Compares to (who uses it) |
|---|---|---|---|
| SQuAD v2 | Single-hop reading comprehension (Phase-1 primary) | ✅ | CAG `yu2024dontdorag` |
| HotpotQA, TriviaQA | Multi-hop / multi-evidence | ✅ | RECOMP, CompAct |
| **Natural Questions, MuSiQue** | Open-domain + multi-hop for compression comparability | ✅ | LongLLMLingua (NQ), CompAct (MuSiQue) |
| QASPER, HumanEval, MBPP, hpc_code | Long-context / code (future) | ✅ loaders | — |

## 7. Infrastructure
Terraform GCP (driver install, git-clone, /health gating, GVNIC/MTU params, **durable GCS results bucket**), Docker (CPU + fixed GPU compose), K8s manifests; `cloud_run.sh`/`sync_results_to_gcs.sh` for continuous result persistence. Objective: reproducible cloud runs whose results survive teardown. Status: ✅ (single-GPU path) / ⏳ (multi-VM distributed = Path B). See [`RUNBOOK.md`](RUNBOOK.md).

---

## 8. Article-ready text (drop-in paragraphs for the new contributions)

**Compression axis (Methods).**
> We extend the CAGE baseline taxonomy with a compression dimension orthogonal to the
> context-source dimension, yielding a 2×2 design (cache vs. retrieve × full vs. compressed).
> Retrieved-context compression (`compressed_rag`) applies task-agnostic prompt compression
> (LLMLingua-2 [Pan et al., 2024]) to the retrieved passages before prompting, following the
> RAG-side compression studied by RECOMP [Xu et al., 2024] and LongLLMLingua [Jiang et al.,
> 2024]. Cached-context compression (`compressed_cag`) compresses the KV cache itself, via
> vLLM FP8 KV-cache quantization and, as an architectural variant, a Multi-head Latent
> Attention model [DeepSeek-AI, 2024] whose low-rank KV is ~7–14× smaller than MHA. We report
> a `compression_ratio` and an analytical `kv_cache_bytes` estimate, and plot quality and TTFT
> against compression — the same axes used by the prompt-compression literature, enabling a
> direct comparison.

**Protocol rigor (Methods / Threats to validity).**
> To support statistical claims, trials draw independent seeded samples; cache state is
> controlled per trial (a cold-start mode flushes the vLLM prefix cache via the
> `reset_prefix_cache` endpoint between trials); warmup queries are disjoint from the measured
> set; and a `context_source` control feeds every baseline the same context (gold or retrieved)
> to remove the confound between caching and context provenance. Significance is assessed with
> per-query Wilcoxon signed-rank tests under Holm–Bonferroni correction with bootstrap
> confidence intervals.

**Grounding metric (Methods).**
> Beyond NLI-based faithfulness, CAGE adopts span-level grounding detection (a ModernBERT
> detector trained on RAGTruth [Niu et al., 2024]) as its primary semantic-quality signal,
> localizing unsupported answer spans rather than emitting a single scalar — a finer-grained
> measure than the response-level scores of RAGAS/ARES.

**Positioning (Related Work / Intro).**
> Unlike retrieval-evaluation frameworks (RAGAS, ARES, BERGEN, RAGBench, RAGChecker), which
> score quality but not serving behavior, and KV-reuse serving systems (TurboRAG, RAGCache,
> CacheBlend, CacheGen, Mooncake), which optimize latency but not faithfulness, CAGE jointly
> evaluates serving metrics and semantic quality across a unified family of cache-aware
> baselines — now including a compression axis — under a common workload with statistical
> testing. See the comparison matrix (Table~\ref{tab:comparison}).
