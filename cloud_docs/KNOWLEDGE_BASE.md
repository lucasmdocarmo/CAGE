# CAGE — Project Knowledge Base (AI-Readable Master Reference)

> **Purpose.** Single, self-contained reference for any AI model (or human collaborator) resuming work on the CAGE research project. Consolidates project identity, architecture, source-code map, infrastructure, metrics, datasets, Phase 1 empirical results, and the actionable roadmap for Phases 2 & 3.
>
> **Scope.** Generated 2026-06-09 from a deep read of all docs, source code under `src/`, scripts, configs, Docker/Terraform/K8s infra, Phase 1 result files, and the LaTeX paper draft `paper/my-article.tex`. **Reconciled 2026-06-09** to the flattened single-level repo (the old triple-nested `CAGE/CAGE/CAGE` layout and the vendored `vllm-main/` copy were removed).
>
> **Repository root (host machine).** `/Users/lucasmariano/CAGE/` — this IS the project root (single level; git repo lives here). Docs are under `docs/`; links below are relative to the repo root.
>
> **⚠️ PHASE 2/3 OPERATIONS — READ THIS FIRST.** For *running* Phase 2 or 3 on GCP, the authoritative,
> current procedures are [`PHASE2_CHECKLIST.md`](PHASE2_CHECKLIST.md), [`PHASE3_PLAN.md`](PHASE3_PLAN.md),
> [`RUNBOOK.md`](RUNBOOK.md), and [`VLLM_COMPATIBILITY.md`](VLLM_COMPATIBILITY.md). Phase 2 runs on a
> **single L4** (not a cluster). Some inline run commands further down in this file predate the
> launch-lever wiring and reference flags that no longer exist (`--phase`, `--all-baselines`,
> `--speculative-model`, `--enable-disagg-prefill`); trust the dedicated docs and
> `python scripts/run_experiment.py --help` over those historical snippets.

---

## 1. Project Identity, Goal & Thesis

### 1.1 What CAGE is
**CAGE = Cache-Augmented Generation Evaluation.** A benchmarking framework that holistically evaluates LLM serving systems combining **KV-cache reuse** and **retrieval** strategies, jointly measuring serving performance (TTFT, TPOT, throughput, tail latency) and semantic output quality (faithfulness, relevance, completeness) across multiple baselines.

- **Author.** Lucas Mariano do Carmo (lucas.mariano.carmo@gmail.com)
- **Co-authors (paper).** Wladmir Cardoso Brandão, Henrique Cota de Freitas
- **Institution.** Pontifícia Universidade Católica de Minas Gerais (PUC Minas), Belo Horizonte, Brazil
- **Paper.** `paper/my-article.tex` (SBC template, ~12 pages)

### 1.2 Problem
LLMs need to be grounded in external knowledge. The dominant pattern, **RAG (Retrieval-Augmented Generation)**, retrieves documents on every query and adds significant prefill/embedding/reranking latency, plus residual hallucinations when retrieval is noisy. **CAG (Cache-Augmented Generation)** instead pre-computes Key/Value tensors for static/semi-static documents and reuses them, ideally eliminating retrieval latency for hot knowledge — at the cost of memory pressure and distributed coordination cost.

There is **no standard evaluation framework** that compares CAG, RAG, hybrid, and distributed cache strategies under unified semantic-quality + systems metrics. CAGE fills that gap.

### 1.3 Academic thesis (claim to prove)
> A distributed **contextual prefix cache** is strictly superior to RAG for static / semi-static knowledge bases — achieving significantly lower **Time-To-First-Token (TTFT)** while maintaining or improving **NLI-Faithfulness** — and CAGE is the first framework that can demonstrate this rigorously.

### 1.4 Status (updated 2026-06-28)
| Phase | State | Hardware | Outcome |
|---|---|---|---|
| **Phase 1** — Local CPU validation | ✅ COMPLETE | Apple M4 Pro, vLLM CPU | Thesis structurally validated. 7 baselines benchmarked (Qwen3-4B, 50q × 3 trials). CPU latencies non-generalizable. |
| **Phase 2** — Single-GPU validation | ✅ COMPLETE (2026-06-27) | GCP **single L4** (g2-standard-8) | Qwen3-8B, vLLM 0.11.0, SQuAD v2, **100q × 1 trial**, **14 baseline result sets across 8 of 9 families** (6 core + compression 2×2 + speculative 2×2; `distributed` deferred to Phase 3). Cost ~$3.1. **ALL infra torn down to $0 (VM + GCS bucket deleted); data local at `phase2_archive/`** (`PHASE2_ANALYSIS.md`, `DATA_INVENTORY.md`). Findings: prefix-cache TTFT −3.3% lossless; FP8 KV (compressed_cag) lossless on grounding; RAG faithfulness −24.7% / TTFT +87%; EAGLE-3 speculative TPOT −41% / latency −32.5% lossless. PRIMARY quality metric = **LettuceDetect grounding**. Caveats: `compressed_rag` invalid (LLMLingua never fired, fixed in code, rerun in P3); `hybrid_warm` unpaired stats; `retrieval_hit` + `completeness_bertscore` metric bugs fixed post-run. |
| **Phase 3** — Distributed HPC | 🔮 NEXT | GCP A100 (a2-highgpu-1g) + GVNIC | Real cross-node KV transfer (LMCache/NIXL; currently SIMULATED/analytic), disagg prefill, broader speculative (MTP). Rerun `compressed_rag`; add multi-trial CIs; RAG-favorable dataset. |

---

## 2. Directory Layout (flattened, single level)

> As of 2026-06-09 the repo is a single level at `/Users/lucasmariano/CAGE/`. The old
> triple-nested `CAGE/CAGE/CAGE` wrapper, the vendored `vllm-main/`, duplicate
> `Documentation/`/`Results/` trees, build logs, and `venv/`/`cage-env/` were removed.

```
/Users/lucasmariano/CAGE/                 ← repo root (git repo lives here)
├── README.md, WARP.md, LICENSE
├── requirements.txt, pytest.ini
├── src/                  ← Framework code (orchestration, evaluation, inference, data, utils, monitoring)
├── scripts/              ← run_experiment.py, statistical_tests.py, phase runners, plot scripts, *.sh
│   └── setup/            ← env bootstrap scripts (setup_ubuntu/fedora/fresh/no_sudo.sh)
├── configs/              ← Hydra YAML configs (model/, dataset/, experiment/)
├── tests/                ← pytest suite
├── docker/               ← docker-compose.yml + docker-compose.gpu.yml + router.Dockerfile (+ router.requirements.txt)
├── terraform/gcp/        ← Cloud infra IaC (main.tf, terraform.tfvars.example)
├── k8s/                  ← Kubernetes manifests
├── experiments/          ← Notebooks + persisted FAISS indices (experiments/ir_index/)
├── analysis/             ← Per-phase results & plots (phase1/ populated; phase2-4/ placeholders)
├── results/              ← Phase 1 result analyses (markdown) + plots (results/images/)
├── paper/                ← paper/my-article.tex (the academic paper draft)
├── docs/                 ← Documentation (see docs/README.md for the status index)
└── logs/                 ← cluster/ + vllm/ run logs
```

Documentation lives under `docs/` with a status index at [`docs/README.md`](README.md):
the canonical references are **this file** and [`VALIDATION_AND_SOTA_REVIEW.md`](VALIDATION_AND_SOTA_REVIEW.md);
[`RUNBOOK.md`](RUNBOOK.md) is the authoritative setup/deploy/run guide; other docs are
marked SUPERSEDED/STALE where applicable.

---

## 3. Architecture

### 3.1 High-level dataflow
```
                ┌──────────────────────────────┐
                │  scripts/run_experiment.py   │  ← workload driver (master CLI)
                └──────────────┬───────────────┘
                               │ HTTP (OpenAI-compat)
                               ▼
        ┌─────────────────────────────────────────────────┐
        │   src/orchestration/router.py  (FastAPI)        │  port 9000
        │   - prefix-hash routing                          │
        │   - SSE streaming pass-through (TTFT histogram)  │
        │   - simulated KV-transfer latency injection      │
        │   - Prometheus /metrics, /stats, /configure      │
        └────────┬──────────┬──────────┬────────────────────┘
                 │          │          │
            replica-1   replica-2   replica-3      ← vLLM HTTP servers (8001/8002/8003)
            (vLLM)      (vLLM)      (vLLM)             --enable-prefix-caching
                                                       --enable-prompt-tokens-details
                                                       (--gpu-memory-utilization 0.9 on GPU)

        ┌──────────────────────────────┐     ┌──────────────────────────────┐
        │ src/orchestration/redis_cache│◄────┤      Redis 7-alpine          │  port 6379
        │ retrieval-artifact cache     │     │  (retrieval results, NOT KV) │
        └──────────────────────────────┘     └──────────────────────────────┘

        ┌──────────────────────────────┐
        │ src/orchestration/ir.py      │  FAISS IndexFlatIP
        │  - intfloat/e5-large-v2      │  (Phase 1)
        │  - BAAI/bge-reranker-large   │  top_k=3
        └──────────────────────────────┘

        ┌──────────────────────────────┐
        │ src/evaluation/quality.py    │  NLI-Faithfulness, Relevance,
        │ src/evaluation/performance.py│  BERTScore, ROUGE-L, F1, latencies
        └──────────────────────────────┘
```

### 3.2 Module → file map

| Module | File | Responsibility |
|---|---|---|
| Master CLI / workload driver | `scripts/run_experiment.py` (~78 KB / ~2049 lines) | Argument parsing, baseline dispatch, telemetry capture, JSON/CSV output |
| Dataset loaders | `src/data/loader.py` | `CAGExample(id, question, context, answer, metadata)` for SQuAD v2, HotpotQA, TriviaQA, QASPER, HumanEval, MBPP |
| vLLM HTTP adapter | `src/inference/vllm_adapter.py` | OpenAI-compat `/v1/completions`, SSE streaming, TTFT capture, `cached_tokens` extraction |
| Quality evaluator | `src/evaluation/quality.py` (~659 lines) | Faithfulness (NLI), relevance (embedding cos sim), BERTScore, ROUGE-L, QA-F1, exact match, cache-block relevance |
| Performance evaluator | `src/evaluation/performance.py` (~842 lines) | TTFT, TPOT, e2e latency, p50/p95/p99, QPS/TPS, GPU sampling via pynvml, speculative-decoding tracker, CacheMetricsTracker |
| Code evaluator | `src/evaluation/code_evaluator.py` | AST syntax check, complexity, imports, security smells (for HumanEval/MBPP) |
| Distributed router | `src/orchestration/router.py` (~737 lines) | FastAPI; SHA1 prefix-hash routing, `_select_replicated_replica`, SSE pass-through |
| Baseline registry | `src/orchestration/baselines.py` (~230 lines) | `BaselineType` enum + `get_baseline_config()` factory |
| KV cache manager | `src/orchestration/cache_manager.py` (~153 lines) | `SimulatedKVCacheManager` — replicated vs sharded_context transfer cost simulation |
| IR module | `src/orchestration/ir.py` | `FaissIRIndex.build()` / `.search()`, deterministic doc IDs via SHA1, `build_corpus_from_contexts()` |
| Redis cache | `src/orchestration/redis_cache.py` | `RetrievalCache` keyed by (dataset, embedding_model, top_k, query_sha1) |
| Prompt formatter | `src/utils/prompting.py` (~93 lines) | `DEFAULT_SYSTEM_PREFIX`, `format_qa_prompt()`, `extract_cacheable_prefix_text()` (returns text before final `\nQuestion:`/`User:` marker) |
| Monitoring | `src/monitoring/` | Telemetry collectors (psutil, pynvml, Prometheus client) |
| Utils | `src/utils/` | Config loading, logging, helpers |

### 3.3 Prompt format (drives prefix-cache hits)
```
You are a helpful assistant. Answer the question using ONLY the provided context.
If the context is insufficient, say you don't know.

Context 1: <passage_text>

Question: <question_text>
Answer:
```
The first ~20 tokens are byte-identical across all queries; that stable prefix is exactly what vLLM's `--enable-prefix-caching` reuses, and what the distributed router hashes for replica selection.

### 3.4 Prefix-hash routing logic
```python
prefix_text   = extract_cacheable_prefix_text(prompt)       # text before final "\nQuestion:"
prefix_tokens = tokenizer(prefix_text)
prefix_hash   = sha1(json.dumps(prefix_tokens)).hexdigest()
replica_idx   = int(prefix_hash, 16) % len(replicas)
```
This guarantees identical-prefix requests land on the same replica → maximises per-replica prefix-cache warmth. UTF-8 byte fallback when the transformer tokenizer is not available.

---

## 4. Baselines (the 7-way comparison)

| # | Baseline (CLI name) | Context source | `--enable-prefix-caching` | Redis cache | Endpoint | Distinguishing trait |
|---|---|---|---|---|---|---|
| 1 | `no_cache` | gold passage | ❌ | — | vLLM :8000 | Worst-case control — full prefill every query |
| 2 | `prefix_cache` | gold passage | ✅ | — | vLLM :8000 | Native vLLM block reuse on shared prefix |
| 3 | `rag` | retrieved (FAISS top-3 + reranker) | ❌ | — | vLLM :8000 | Standard retrieval, no reuse |
| 4 | `redis_retrieval_cache_cold` | retrieved | ❌ | ✅ cold (0 % hit) | vLLM :8000 | Retrieval artifacts in Redis, cold start |
| 5 | `hybrid_retrieval_cache_cold` | retrieved | ✅ | ✅ cold | vLLM :8000 | Retrieval + prefix caching, cold |
| 6 | `hybrid_retrieval_cache_warm` | retrieved | ✅ | ✅ warm (100 % hit, pre-warmed with 50 excluded queries) | vLLM :8000 | Production-realistic warm-start |
| 7 | `distributed_router_replicated` | gold passage | ✅ | — | router :9000 → 3× vLLM | Multi-replica prefix-hash routing |

Defined in `src/orchestration/baselines.py`. Selected via `--baseline <name>` on the master runner (the suite scripts loop over them; there is no `--all-baselines` flag). Additional baselines exist in the enum but were not used in Phase 1: `speculative` (n-gram, suffix, medusa, eagle, draft-model variants), `distributed_sharded` (Phase 3 target).

---

## 5. Metrics Catalog

### 5.1 Performance metrics — `src/evaluation/performance.py`
| Metric | Definition / How computed |
|---|---|
| **TTFT (Time-To-First-Token)** | `perf_counter()` at request start → first SSE `data:` chunk. Reported avg, p50, p95, p99 (ms). Recorded in the router's Prometheus `REQUEST_TTFT` histogram with buckets `[0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]` s. |
| **TPOT (Time-Per-Output-Token)** | `(total_time_ms − ttft_ms) / (num_tokens − 1)`. avg + p50/p95/p99 (ms). |
| **End-to-end Latency** | Wall-clock; equals `TTFT + TPOT × output_tokens`. |
| **QPS (Queries / s)** | `total_successful_requests / total_experiment_time_s`. |
| **TPS (Tokens / s)** | `total_output_tokens / total_experiment_time_s`. |
| **CPU % / Memory MB** | `psutil` sampled in background thread. |
| **GPU utilisation / VRAM / power / temperature / PCIe** | `pynvml` sampled in background thread by `GPUMetricsTracker` (gracefully degrades on non-NVIDIA hosts). |
| **Speculative decoding** | `SpeculativeMetricsTracker`: acceptance rate, draft tokens / step, rollback overhead, speedup ratio. |

### 5.2 Cache metrics — `src/evaluation/performance.py` (`CacheMetricsTracker`) + vLLM telemetry
| Metric | Source | Phase 1 observation |
|---|---|---|
| **Prompt-cached ratio** | `usage.prompt_tokens_details.cached_tokens / prompt_tokens` (requires vLLM `--enable-prompt-tokens-details`) | prefix_cache 68.4 % / hybrid_cold 75.6 % / hybrid_warm **89.2 %** |
| **Local-hit / remote-hit / miss ratios** | Internal `CacheMetricsTracker` | — |
| **Retrieval hit rate** | Custom: relevant doc appears in FAISS top-k | 98 % across all retrieval baselines |
| **Retrieval-cache rate** | Custom: served from Redis | 0 % cold / 100 % warm |
| **Inter-node KV transfer (sim.)** | `SimulatedKVCacheManager` → `(num_nodes−1)/num_nodes × tokens × bytes_per_token / bandwidth` (default 100 Gbps) | 0 (replicated has no transfer) |

### 5.3 Quality metrics — `src/evaluation/quality.py` (rebuilt 2026-06-09)
> The Phase-1 numbers below (0.570 faithfulness, 0.324 BERTScore) were **artifacts of
> now-fixed bugs** and must NOT be re-quoted as results. They will be recomputed on the
> next run. Fields marked `None` when their model is unavailable (excluded from means).

| Field (output key) | Implementation | Role |
|---|---|---|
| **Grounding** (`grounding_score`, `hallucination_detected`, `hallucinated_span_ratio`) | **LettuceDetect** (`KRLabsOrg/lettucedect-base-modernbert-en-v1`), token/span-level over `(context, question, answer)`; `grounding_score = 1 − hallucinated_span_ratio`. | **PRIMARY** grounding/hallucination signal. Disable via `CAGE_DISABLE_LETTUCEDETECT=1`. |
| **Faithfulness (NLI)** (`faithfulness`, `supported_claim_ratio`) | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (fallback `facebook/bart-large-mnli`). Answer **split into claims**; each claim's entailment = **max over context docs** (proper sentence-pair input; entailment index resolved from model config); averaged over claims. | Secondary faithfulness; cross-checks LettuceDetect. |
| **Context relevance** (`context_relevance`, alias `relevance`) | `all-MiniLM-L6-v2`; `max cos_sim(question, context_i)`. | **Retriever diagnostic only** — independent of the answer; not answer quality. |
| **BERTScore** (`completeness_bertscore`) | RoBERTa F1 with **`rescale_with_baseline=True, lang="en"`**. | Secondary fluency/overlap (now discriminative). |
| **ROUGE-L** (`completeness_rouge_l`) | LCS F1 answer vs gold. | Secondary. |
| **QA F1 / Exact Match** (`f1_score`, `exact_match`) | Token-level P/R with SQuAD normalisation. | Standard QA correctness. |
| **Cache-block relevance** (`cache_relevance`) | Per-block cos-sim vs reference answer; Jaccard fallback. | Diagnostic for distributed CAG. |

### 5.4 Aggregation strategy
For each baseline: percentiles computed **within** a trial, then mean/std/min/max aggregated **across** trials. Results emitted as `aggregated_metrics.json` per baseline; raw per-trial JSON kept alongside.

---

## 6. Datasets

| Dataset | Status | Loader | FAISS index dir | Use |
|---|---|---|---|---|
| **SQuAD v2** (`rajpurkar/squad_v2`) | ✅ used in Phase 1 (validation split, 50 sampled, seed = 42) | `SquadV2Loader` | `experiments/ir_index/ir_squad_v2_intfloat_e5-large-v2/` (also a MiniLM variant) | Primary benchmark |
| **HotpotQA** | indexes pre-built, untested in P1 | `HotpotQALoader` | `experiments/ir_index/ir_trivia_qa_*` & corresponding HotpotQA dirs | Multi-hop reasoning (P2 target) |
| **TriviaQA** | indexes pre-built | loader | `ir_trivia_qa_intfloat_e5-large-v2/`, `ir_trivia_qa_all-MiniLM-L6-v2/` | Dataset diversity (P2 target) |
| **QASPER** | loader only | `QasperLoader` | — | Long-context HPC |
| **HumanEval / MBPP** | loaders + `CodeQualityEvaluator` | — | — | Code generation (future) |

Indices are persisted as `faiss.index` + `documents.jsonl` + `meta.json`. Rebuild via `scripts/download_datasets.py` or `--rebuild-ir-index` on the master runner.

---

## 7. The Master Runner — `scripts/run_experiment.py`

~2049 lines, ~78 KB. **39 CLI flags.** Single entry point for every experiment.

### 7.1 CLI surface (grouped)
**Required:** `--baseline {no_cache|prefix_cache|rag|redis|hybrid|distributed|speculative}`, `--model <HF id>`.

**Workload:** `--dataset {hotpotqa|squad_v2|qasper|trivia_qa|humaneval|mbpp|hpc_code}`, `--num-queries`, `--max-tokens`, `--num-trials`, `--repeat-queries`, `--workload-mode {single|batched|multi_turn}`, `--batch-size`, `--multi-turn-length` (no `--phase`/`--all-baselines`; phases are driven by the suite scripts).

**Inference backend:** `--api-base` (default `http://localhost:8000`), `--backend {vllm|gemini|ollama}`, `--offline` (in-process vLLM).

**Output:** `--output-dir`, `--baseline-label`.

**Retrieval / IR:** `--top-k`, `--top-k-sweep`, `--embedding-model`, `--reranker-model`, `--reranker-device`, `--ir-index-dir`, `--rebuild-ir-index`.

**Redis:** `--redis-host`, `--redis-port`, `--redis-ttl-seconds`, `--flush-redis-namespace` (forces cold start).

**Distributed:** `--sharding-policy {replicated|sharded_context}`, `--stats-url` (router `/stats`).

**Speculative:** `--speculative-model`, `--num-speculative-tokens`, `--speculative-method {draft_model|ngram|suffix|medusa|eagle}`.

### 7.2 Execution flow
1. Parse args, capture provenance (git commit, system snapshot, backend version, model id).
2. Initialise `QualityEvaluator`, `PerformanceEvaluator`, `GPUMetricsTracker` (if pynvml available), `CacheMetricsTracker`.
3. Load dataset → list of `CAGExample`.
4. If baseline ∈ retrieval set: build / load FAISS index; instantiate reranker.
5. Instantiate inference adapter (`VLLMAdapter` / `VLLMOfflineAdapter` / others).
6. For each query (× `--repeat-queries`): build prompt (gold or retrieved), send streaming request, capture TTFT + usage, evaluate quality, log cache/transfer telemetry.
7. Aggregate metrics → write outputs.

### 7.3 Output layout (per baseline)
```
analysis/phase{1,2,3}/results/<baseline_name>/
├── trial_1/
│   ├── metrics.json   ← per-trial aggregate
│   └── results.csv    ← per-query rows (or results.json)
├── trial_2/ …
├── trial_3/ …
├── aggregated_metrics.json   ← mean ± std across trials (SOURCE OF TRUTH)
├── summary.txt
└── metadata.json              ← git/system/backend provenance
```

---

## 8. Supporting Scripts (`scripts/`)

| Script | Purpose |
|---|---|
| `download_datasets.py` | Download HF datasets (SQuAD v2 today; planned: TriviaQA / Natural Questions for P2). |
| `verify_results.py` | Post-run integrity check — every JSON parsable, no truncated trials, no HTTP-timeout poisoning. |
| `statistical_tests.py` (referenced; created in P2) | Wilcoxon rank-sum p-values for latency / quality comparisons, plus bootstrap CIs. |
| `manage_vllm_server.sh` | Start/stop a single vLLM server with flags (`--enable-prefix-caching`, `--no-prefix-cache`, `--enable-prompt-tokens-details`). |
| `manage_vllm_cluster.py` (~480 lines) | Start/stop N replicas on `base_port..base_port+N-1`, healthcheck each. |
| `vllm_runtime_wrapper.py` | Thin wrapper to launch vLLM with Prometheus metrics export. |
| `deploy_cluster.sh` | Convenience wrapper for end-to-end local cluster bring-up. |
| `simulate_network.sh` | Inject artificial network latency for sensitivity studies. |
| `run_phase1.sh` … `run_phase5.sh` | One-click phase runners (start server → iterate baselines with fixed `--num-queries` → teardown). |
| `run_all_phases.sh` | Master driver of all phase scripts. |
| `generate_publication_plots.py` (~1134 lines) | Generates 16 publication figures (latency, throughput, quality, Pareto, radar, heatmap, box plots, tail latency …) → `analysis/<phase>/images/`. |
| `generate_compact_figures.py` | Compact / paper-ready composites. |
| `generate_plots.py`, `generate_additional_plots.py` | Older / supplemental plotting paths. |
| `generate_run_b_commands.py`, `run_b_batch.sh` | Batched alternative-run generator. |

### 8.1 OS bootstrap scripts (`setup_*.sh`)
| Script | OS / Scenario |
|---|---|
| `setup_ubuntu.sh` | Ubuntu 22.04/24.04 — Python 3.12, venv, Docker, NVIDIA container toolkit (the canonical Phase 2 path). |
| `setup_fedora.sh` | Fedora variant. |
| `setup_fresh.sh` | Brand-new machine, minimal. |
| `setup_no_sudo.sh` | When sudo isn't available (e.g. some sandboxes). |
| `setup_vllm.sh` / `setup_vllm_retry.sh` / `setup_vllm_curl.sh` / `setup_vllm_no_iso.sh` | Variants for compiling / installing vLLM in different network environments. |

---

## 9. Configs (Hydra)

```
configs/
├── model/        ← e.g. qwen3-4b.yaml, qwen2.5-7b-instruct.yaml, qwen3-8b.yaml, qwen3-14b.yaml
├── dataset/      ← e.g. squad_v2.yaml, hotpotqa.yaml, trivia_qa.yaml
└── experiment/   ← baseline.yaml (default experiment)
```

**Model config example:**
```yaml
name: Qwen/Qwen2.5-7B-Instruct
max_tokens: 512
temperature: 0.7
top_p: 0.95
requires_gpu: false
ram_gb: 20.0
context_length: 131072
dtype: bfloat16
enable_prefix_caching: true
tier: large
```

**Dataset config example:**
```yaml
name: hotpotqa
split: validation
max_examples: 100
seed: 42
has_context: true
task_type: multi_hop_reasoning
```

CLI flags on the master runner override Hydra config values.

---

## 10. Tests (`tests/`) and `run_tests.sh`

| File | What it covers | pytest mark |
|---|---|---|
| `conftest.py` | Fixtures: `vllm_test_api_base` (env `VLLM_TEST_API_BASE`, default `http://localhost:8000`), `vllm_test_model` (default `Qwen/Qwen2.5-Coder-0.5B-Instruct`), `vllm_available` (probes `/health`). | — |
| `test_router_integration.py` | Router `/health`, `/stats` (replica distribution), Prometheus `/metrics`, non-streaming proxy with `x-router-replica` header, SSE pass-through with `[DONE]` termination. Skipped if `ROUTER_TEST_API_BASE` unreachable. | `integration` |
| `test_vllm_integration.py` | `VLLMAdapter` streaming + non-streaming, TTFT capture, usage / `cached_tokens` extraction. | `vllm` |
| `test_inference.py` | Engine contract tests, request/response serialisation. | — |
| `test_data.py` | Dataset loaders (HotpotQA, SQuAD v2), example formatting. | — |
| `test_ir.py` | FAISS index build/search; embedding similarity sanity. | — |
| `test_baselines.py` | `BaselineConfig` instantiation, factory correctness. | — |

`pytest.ini` declares custom marks: `slow`, `integration`, `vllm`, `gpu`; excludes `venv/`, `__pycache__/` from coverage.

`run_tests.sh` runs the suite (excluding integration tests that need a live router/vLLM).

---

## 11. Infrastructure as Code

### 11.1 Docker — `docker/`
| File | Purpose | Key settings |
|---|---|---|
| `router.Dockerfile` | Build the router image (`cage-router:latest`). | Installs Python deps, ENTRYPOINT runs `python -m src.orchestration.router`. |
| `docker-compose.yml` (CPU) | Local laptop (ARM64 / x86 CPU). | 3× vLLM replicas using `public.ecr.aws/q9t5s3a7/vllm-arm64-cpu-release-repo:latest` on `Qwen/Qwen3-4B`, ports 8001/8002/8003, `VLLM_CPU_KVCACHE_SPACE=1`. `redis:7-alpine` on 6379. Router on 9000 with `ROUTER_REPLICAS=replica-1=http://vllm-replica-1:8001,replica-2=…,replica-3=…`. |
| `docker-compose.gpu.yml` (GPU) | GCP / on-prem GPU. | Same 3 replicas using `vllm/vllm-openai:v0.11.0` on `Qwen/Qwen3-8B`, NVIDIA GPU device request `capabilities: [gpu]`, env `VLLM_TENSOR_PARALLEL_SIZE=1`, `VLLM_GPU_MEMORY_UTILIZATION=0.9`, `HF_TOKEN=${HF_TOKEN}`. Identical Redis + router. |

All replicas run with `--enable-prefix-caching --enable-prompt-tokens-details`.

### 11.2 Kubernetes — `k8s/`
| Manifest | Resources |
|---|---|
| `router.yaml` | `Deployment` `cage-router` (replicas=1, image `cage-router:latest`, port 9000, env `ROUTER_REPLICAS=…`); `Service` NodePort 9000 → 30090. |
| `vllm-replica.yaml` | Three Deployment+Service pairs (`vllm-replica-1/-2/-3`), image `vllm/vllm-openai:v0.11.0`, args include model + port + prefix-caching flags. GPU `resources.limits` commented out — uncomment for cloud GPU. ClusterIP services on 8001/8002/8003. |
| `redis.yaml` | `Deployment` `cage-redis` (image `redis:7-alpine`, port 6379); ClusterIP Service named `redis`. |

### 11.3 Terraform — `terraform/gcp/main.tf`

**Resources provisioned:**
* **Network:** VPC `cage-network` (10.0.0.0/24), subnet `cage-subnet`. Firewall rules: internal (TCP/UDP 0-65535 + ICMP intra-subnet), SSH 22 (0.0.0.0/0), router external TCP 9000 + 8000 → tag `cage-router`.
* **vLLM replicas (count = `num_replicas`, default 3):**
  * Machine type `g2-standard-8` (8 vCPU, 32 GB RAM)
  * GPU `nvidia-l4` × 1 (configurable to `nvidia-tesla-a100` or `nvidia-tesla-t4`)
  * Boot disk 200 GB pd-ssd, image `deeplearning-platform-release/common-cu121-v20240128-debian-11`
  * Tag `cage-vllm`; ephemeral public IP + internal private IP
  * Scheduling: `on_host_maintenance = TERMINATE`; on-demand by default (`automatic_restart = true`), or Spot when `var.preemptible = true` (`provisioning_model = SPOT`, `automatic_restart = false`)
  * **Startup script** (metadata): installs Docker + NVIDIA Container Toolkit; pulls `vllm/vllm-openai:v0.11.0` (the pinned `var.vllm_image`); runs container on dynamic port (8002–8004) with `--model Qwen/Qwen3-8B --enable-prefix-caching --enable-prompt-tokens-details --gpu-memory-utilization 0.9`; injects `HF_TOKEN`.
* **Router (`cage-router`):** Machine `e2-standard-4` (4 vCPU, 16 GB), 50 GB pd-standard, Debian 11. **Startup script** installs Docker, runs Redis 7-alpine on 6379, queries replica IPs via `gcloud`, builds `ROUTER_REPLICAS` env var, waits for code upload to `/opt/cage`, installs `requirements.txt`, launches `python3 -m src.orchestration.router`.

**Variables (`variable {}` blocks):**
| Name | Type | Default |
|---|---|---|
| `project_id` | string | (required) |
| `region` | string | `us-central1` |
| `zone` | string | `us-central1-a` |
| `num_replicas` | number | 3 |
| `gpu_type` | string | `nvidia-l4` |
| `gpu_count` | number | 1 |
| `machine_type` | string | `g2-standard-8` |
| `model_name` | string | `Qwen/Qwen3-8B` |
| `disk_size_gb` | number | 200 |
| `hf_token` | string (sensitive) | `""` |

**Outputs:** `router_external_ip`, `replica_internal_ips`, `replica_external_ips`, `ssh_commands`, `experiment_command`.

**Phase 3 reconfiguration** (in [PHASE_EXECUTION_GUIDE.md](CAGE/PHASE_EXECUTION_GUIDE.md)):
* `machine_type → "a2-highgpu-1g"` (A100 40 GB / 80 GB)
* Add `mtu = 8896` (jumbo frames) on `google_compute_network`
* Add `nic_type = "GVNIC"` on the replica `network_interface` → unlocks 100 Gbps from the default ~15 Gbps. Without GVNIC, cross-node KV transfer would be slower than recomputing prefill from scratch.

---

## 12. Phase 1 — Validated Empirical Results

### 12.1 Experimental setup
| Parameter | Value |
|---|---|
| Run window | 2026-03-19 15:05 → 20:40 (≈ 6.5 h wall-clock, single corrected run) |
| Hardware | Apple M4 Pro, 12 cores, 24 GB unified memory, **CPU only** |
| Backend | vLLM CPU build (source), `VLLM_CPU_KVCACHE_SPACE=10`, `VLLM_CPU_OMP_THREADS_BIND=auto` |
| Model | `Qwen/Qwen3-4B` |
| Dataset | SQuAD v2 (validation), 50 sampled examples, seed = 42 |
| Trials | 3 per baseline |
| Decoding | `max_tokens = 100`, `temperature = 0.7`, `top_p = 0.95` |
| Retrieval | embedding `intfloat/e5-large-v2`, reranker `BAAI/bge-reranker-large`, `top_k = 3` |
| Distributed | 3 replicas (ports 8001/8002/8003) + router (port 9000); routing strategy = hash-based prefix routing; observed request distribution 13 / 15 / 22 |
| Total measured | 50 × 3 × 7 = **1 050 queries** |

### 12.2 Headline results

| Baseline | Avg Latency (ms) | Avg TTFT (ms) | QPS | Faithfulness | Relevance | BERTScore |
|---|---:|---:|---:|---:|---:|---:|
| **no_cache** | 16 006 ± 711 | 6 919 ± 112 | 0.0606 ± 0.0025 | 0.5703 ± 0.0565 | 0.5051 | 0.3279 |
| **prefix_cache** | **10 015** ± 884 | **2 376** ± 275 | **0.0959** ± 0.0077 | 0.5703 ± 0.0565 | 0.5051 | 0.3279 |
| **rag** | 27 270 ± 283 | 18 775 ± 144 | 0.0333 ± 0.0006 | 0.5044 ± 0.0228 | 0.5251 | 0.3243 |
| **redis_retrieval_cache_cold** | 26 853 ± 39 | 18 579 ± 37 | 0.0342 ± 0.0000 | 0.5504 ± 0.0397 | 0.5251 | 0.3243 |
| **hybrid_retrieval_cache_cold** | 15 513 ± 4 173 | 6 001 ± 4 290 | 0.0589 ± 0.0120 | 0.5056 ± 0.0222 | 0.5251 | 0.3243 |
| **hybrid_retrieval_cache_warm** | 13 269 ± 893 | 2 791 ± 66 | 0.0674 ± 0.0039 | 0.5123 ± 0.0267 | 0.5251 | 0.3245 |
| **distributed_router_replicated** | 18 492 ± 372 | 5 279 ± 638 | 0.0525 ± 0.0011 | **0.6359** ± 0.0775 | 0.5051 | 0.3267 |

### 12.3 Headline deltas vs. `no_cache`
| Baseline | Δ Latency | Δ TTFT | Δ QPS | Δ Faithfulness |
|---|---:|---:|---:|---:|
| `prefix_cache` | **−37.4 %** | **−65.7 %** | **+58.3 %** | **0.0 %** |
| `rag` | +70.4 % | +171.4 % | −45.1 % | **−11.6 %** |
| `redis_retrieval_cache_cold` | +67.7 % | +168.6 % | −43.6 % | −3.5 % |
| `hybrid_retrieval_cache_cold` | −3.1 % | −13.3 % | −2.8 % | −11.3 % |
| `hybrid_retrieval_cache_warm` | **−17.1 %** | **−59.7 %** | +11.2 % | −10.2 % |
| `distributed_router_replicated` | +15.5 % | **−23.7 %** | −13.4 % | +11.5 %* |

\* high variance ±0.0775.

### 12.4 Cache telemetry
| Baseline | Retrieval hit | Retrieval-cache rate | Prompt-cached ratio |
|---|---:|---:|---:|
| `prefix_cache` | — | — | **68.4 %** |
| `rag` / `redis_retrieval_cache_cold` | 98 % | 0 % | — |
| `hybrid_retrieval_cache_cold` | 98 % | 0 % | 75.6 % |
| `hybrid_retrieval_cache_warm` | 98 % | **100 %** | **89.2 %** |
| `distributed_router_replicated` | — | — | 68.4 % |

### 12.5 Tail latency (TTFT p95 / p50)
| Baseline | p50 (ms) | p95 (ms) | p95/p50 |
|---|---:|---:|---:|
| `no_cache` | 6 467 | 10 435 | 1.6× |
| `prefix_cache` | 2 300 | 4 716 | 2.1× |
| `rag` | 18 871 | 23 895 | 1.3× |
| `redis_retrieval_cache_cold` | 18 629 | 23 974 | 1.3× |
| `hybrid_retrieval_cache_cold` | 5 928 | 10 127 | 1.7× |
| `hybrid_retrieval_cache_warm` | 2 947 | 4 462 | 1.5× |
| `distributed_router_replicated` | 2 876 | 21 896 | **7.6× (!)** |

The distributed cold-replica effect is the headline tail-latency finding: hash routing concentrates the first request to each replica into a cold prefill burst.

### 12.6 Key insights (paper-ready conclusions)
1. **Native prefix caching is the single biggest win** — 37.4 % latency / 65.7 % TTFT with **zero quality loss**. Free lunch when a stable system prefix exists.
2. **High retrieval hit rate ≠ better performance.** RAG and redis_cold both hit 98 % yet remain the slowest baselines. Cache *reuse*, not retrieval accuracy, drives TTFT.
3. **RAG measurably degrades factual grounding** by −11.6 % on strict NLI-Faithfulness. Hybrid baselines also lose ~10–11 % faithfulness because the retrieved context is itself the bottleneck.
4. **BERTScore is non-discriminative** (Δ ≤ 0.004 across all baselines) — soft embedding metrics cannot expose subtle hallucinations. **Strict NLI-Faithfulness is the correct primary metric.**
5. **Distributed routing has a serious tail-latency problem** (7.6× p95/p50). Cold-replica warm-up must be addressed for production use.
6. **Hybrid warm** is the most production-realistic configuration — recovers 17 % latency and 60 % TTFT vs. no_cache while still using retrieval.

### 12.7 Known methodological caveats
* **Cross-trial prompt-cache carryover.** vLLM is restarted per baseline but **not** per trial, so trials 2 + 3 inherit warmed cache state. Trial-1 numbers are the truly cold readings; aggregates are mildly optimistic for cold-start claims (esp. `hybrid_retrieval_cache_cold`: trial-1 TTFT = 12 067 ms vs. aggregate 6 001 ms).
* **CPU-only** absolute latencies (10–27 s/query) are proof-of-concept, not production-comparable. Relative rankings and percentage deltas are the valid conclusions.
* **n = 3 trials** — statistically thin; Phase 2 increases to **n ≥ 10 trials, 100 queries**.
* **Single dataset.** Phase 2 adds TriviaQA / Natural Questions to test generalisation.
* **Distributed baseline did NOT measure real cross-node KV transfer** — router only forwards HTTP requests in Phase 1. Transfer-cost simulation lives in `SimulatedKVCacheManager`. Phase 3 implements real byte-level transfer.
* **Provenance gaps.** Git commit, backend version, model_id were null in some `metadata.json` files. Fix before Phase 2.

### 12.8 Result artifacts
Aggregated metrics (source of truth): `analysis/phase1/results/<baseline>/aggregated_metrics.json`.

**Plots (16 publication figures + compact variants)** in `results/images/`:
01 latency_comparison · 02 throughput_comparison · 03 quality_comparison · 04 latency_breakdown · 05 speedup_vs_nocache · 06 pareto_latency_vs_quality · 07 radar_profile · 08 heatmap · 09 boxplots_variance · 10 summary_table · 11 cache_hit_vs_ttft · 12 quality_performance_matrix · 13 overhead_decomposition · 14 efficiency_ranking · 15 context_type_impact · 16 ttft_tail_latency.

**CSV summaries** in `analysis/phase1/plots/`:
`latest_metrics_summary.csv`, `pareto_optimal_baselines.csv`.

**Analysis markdown** in `results/` (and its top-level mirror):
`completeOverviewAnalysis.md`, `detailedRunAnalysis.md`, `resultPhase1Analysis.md`, `latex_tables.md`.

**Run logs** in `logs/`:
* root: `phase1_rerun_*.log` (final corrected run: `phase1_rerun_20260319_150430.log`, 125 KB)
* `logs/cluster/`: per-replica vLLM + router logs from distributed run
* `logs/vllm/`: per-run vLLM backend logs (30+ files)

### 12.9 The academic paper — `CAGE/my-article.tex`
* **Title.** *CAGE: An Evaluation Framework for Cache-Augmented Generation Models*
* **Authors.** Lucas Mariano do Carmo, Wladmir Cardoso Brandão, Henrique Cota de Freitas (PUC Minas).
* **Format.** SBC template, 12 pt, ~12 pages.
* **Abstract claim.** Native prefix caching cuts latency 37.4 % and TTFT 65.7 % with no faithfulness loss; RAG increases latency 70.4 % and reduces faithfulness 11.6 %.
* **Section structure.** (1) Introduction; (2) Understanding KV Cache + Related Work (RAGAS / RAGBench / TurboRAG / DistServe / SelfRoute vs CAGE); (3) CAGE Framework Architecture; (4) Experimental Results (4.1 Serving Performance, 4.2 Latency Decomposition, 4.3 Quality Results, 4.4 Cache Telemetry); (5) Conclusion + 3-phase roadmap.
* **Known typos to fix:** "reduced latency *of* 37.4 %" → "*by* 37.4 %"; some `\includegraphics` options commented out.

---

## 13. Phases 2 & 3 — Roadmap

### 13.1 Phase 2 — GPU validation & scaling
**Goal.** Confirm Phase 1 rankings hold under real GPU prefill / KV-cache memory pressure, scale model sizes, and raise statistical rigour.

| Item | Detail |
|---|---|
| Hardware | GCP `g2-standard-8` × 3 + 1 CPU router (Terraform default). NVIDIA L4 (24 GB VRAM). |
| Model | `Qwen/Qwen3-8B` (primary); also sweep `Qwen3-4B` and `Qwen3-14B` for scaling curves. |
| vLLM | `--gpu-memory-utilization 0.9` |
| Workload | 10 trials × 100 queries × 7 baselines |
| Datasets | SQuAD v2 + TriviaQA (and/or Natural Questions) |
| New script | `scripts/statistical_tests.py` — Wilcoxon rank-sum p-values + bootstrap CIs |
| Cluster cost | ≈ \$4.50/hr (3× G2 ≈ \$1.20/hr + 1× CPU router) |
| Wall-clock | ≈ 3 h per full sweep |
| Total cost | ≈ \$13.50 per full sweep |

**Tunable levers:** `VLLM_GPU_MEMORY_UTILIZATION` (0.7 = safer, smaller KV space → cache evicts faster; 0.95 = more KV space but OOM risk). Phase 2 deliverables: extended paper (≥ 18 pp), scaling curves, statistical tables, multi-dataset comparison.

### 13.2 Phase 3 — Distributed HPC stress
**Goal.** Push architecture to its distributed limits — real cross-node KV transfer, disaggregated prefill, speculative decoding, dynamic retrieval fallback.

| Item | Detail |
|---|---|
| Hardware | GCP `a2-highgpu-1g` × 3 (NVIDIA A100 40 GB or 80 GB) + 1 CPU router |
| Network | VPC `mtu = 8896` (jumbo frames); replicas `nic_type = "GVNIC"` (≈ 100 Gbps). Without these, KV transfer is slower than full prefill. |
| Model | `Qwen/Qwen3-14B` primary; `Qwen/Qwen3-1B` as speculative draft. |
| Advanced features | `--enable-disagg-prefill` (separate prefill and decode workers); `--speculative-model Qwen/Qwen3-1B` with various `--speculative-method`. |
| New work | Implement real byte-level KV tensor transfer between replicas (router extension: gRPC or NCCL); new baselines `distributed_sharded`, `distributed_migrated`; eviction-policy comparison (LRU / freq / prefix-length); dynamic retrieval fallback (cache miss → retrieve → cache the synthesised result). |
| Cluster cost | ≈ \$11.50/hr |
| Wall-clock | 4–5 h per sweep |
| Total cost | ≈ \$50–60 per sweep |
| Validation signal | If tensor-transfer time > 100 ms → GVNIC mis-configured. |

---

## 14. Running the Project

### 14.1 Local CPU (laptop — repro of Phase 1)
```bash
# 1. Environment
conda create -n cage-vllm python=3.12 -y
conda activate cage-vllm
cd /Users/lucasmariano/CAGE
pip install -r requirements.txt

# 2. Build vLLM CPU from source (macOS / ARM64)
cd ~/projects && git clone https://github.com/vllm-project/vllm.git && cd vllm
pip install -r requirements/cpu.txt --index-strategy unsafe-best-match
pip install -e .

export VLLM_CPU_KVCACHE_SPACE=10
export VLLM_CPU_OMP_THREADS_BIND=auto

# 3. Data + index
cd /Users/lucasmariano/CAGE
python3 scripts/download_datasets.py

# 4. Serve vLLM
vllm serve Qwen/Qwen3-4B --port 8000 \
  --enable-prefix-caching --enable-prompt-tokens-details

# 5. Run an experiment
python3 scripts/run_experiment.py \
  --baseline prefix_cache --model Qwen/Qwen3-4B \
  --dataset squad_v2 --trials 3 --num-queries 50
```

### 14.2 Local Docker Compose (multi-replica + Redis + router)
```bash
# CPU:
docker compose -f docker/docker-compose.yml up -d
# GPU:
HF_TOKEN=hf_xxx docker compose -f docker/docker-compose.gpu.yml up -d

# Hit the router instead of a single vLLM:
python3 scripts/run_experiment.py \
  --baseline distributed --api-base http://localhost:9000 \
  --sharding-policy replicated --model Qwen/Qwen3-4B \
  --dataset squad_v2 --trials 3 --num-queries 50
```

### 14.3 Phase 2 (GCP cloud)
```bash
# Prereqs: GCP project, Compute Engine API + Cloud Resource Manager API enabled,
#          quotas: GPUs (all regions) ≥ 4, NVIDIA L4 ≥ 3 in us-central1.
gcloud auth login
gcloud config set project <PROJECT_ID>
gcloud config set compute/region us-central1
gcloud config set compute/zone   us-central1-a
export HF_TOKEN=hf_xxx

cd /Users/lucasmariano/CAGE/terraform/gcp
terraform init
terraform apply -var="project_id=$(gcloud config get-value project)" -var="hf_token=$HF_TOKEN"

# Push code:
gcloud compute scp --recurse . cage-router:/opt/cage --zone=us-central1-a
gcloud compute ssh cage-router --zone=us-central1-a
cd /opt/cage

# Phase 2 is a SINGLE L4 (not the router cluster). See cloud_docs/PHASE2_CHECKLIST.md.
bash scripts/setup/setup_gpu_cloud.sh
nohup bash scripts/cloud_run.sh Qwen/Qwen3-8B 100 10 > run.log 2>&1 &   # 7 baselines + telemetry + GCS sync
bash scripts/run_compression.sh Qwen/Qwen3-8B   # FP8 2x2 axis (gates FP8 x prefix-cache)
bash scripts/run_phase5.sh                      # speculative decoding

python3 scripts/statistical_tests.py --results-dir analysis/phase1/results/

gcloud compute scp --recurse cage-router:/opt/cage/analysis/ ./analysis_cloud_backup/ --zone=us-central1-a

# ⚠ TEARDOWN — GPU costs accrue per minute:
terraform destroy -var="project_id=$(gcloud config get-value project)" -var="hf_token=$HF_TOKEN"
```

### 14.4 Phase 3 (GCP HPC)
1. `terraform destroy` the Phase 2 cluster.
2. Edit `terraform/gcp/main.tf`:
   * `machine_type = "a2-highgpu-1g"`
   * Add `mtu = 8896` on `google_compute_network`.
   * Add `nic_type = "GVNIC"` on replica `network_interface`.
3. `terraform apply` again, push code, then:
```bash
# Phase 3 distributed runs against the router. See cloud_docs/PHASE3_PLAN.md for the
# real-KV-connector path. FP8/speculative are launch-time levers (run_compression.sh / run_phase5.sh),
# not run_experiment.py flags.
python3 scripts/run_experiment.py --baseline distributed \
    --model Qwen/Qwen3-8B --api-base http://<router>:9000 --vllm-telemetry
```

### 14.5 Tests
```bash
cd /Users/lucasmariano/CAGE
./run_tests.sh                         # unit suite
pytest -m vllm                         # vLLM adapter (live server needed)
ROUTER_TEST_API_BASE=http://localhost:9000 pytest -m integration   # router live
```

---

## 15. Critical Implementation Notes / Gotchas

1. **vLLM is consumed only via HTTP.** The framework never imports `vllm`. The vendored `vllm-main/` directory is for source-build reference only — `run_experiment.py` ignores it.
2. **`--enable-prompt-tokens-details` is mandatory** for `cached_tokens` telemetry — without it you get zero cache-ratio observability.
3. **TTFT is captured via streaming SSE** (`stream=True`, `stream_options={"include_usage": true}`) — measuring `perf_counter()` between request issue and first `data:` chunk. Any non-streaming code path silently breaks TTFT.
4. **Prefix-cache hits depend on byte-identical prompt prefixes.** Don't reshuffle `DEFAULT_SYSTEM_PREFIX` or insert per-query metadata before the question — it kills the entire cache reuse property.
5. **Gold-vs-retrieved context confound.** Baselines 1, 2, 7 use gold passages; 3, 4, 5, 6 use retrieved. Latency and quality differences confound "caching" with "context source quality". Always report this explicitly.
6. **Phase 1 Distributed baseline did NOT move any tensors** (`SimulatedKVCacheManager` returns 0 transfers in `replicated` mode). The 7.6× tail spread is purely the cold-prefill effect of routing to an unwarmed replica. Phase 3 is the moment to add real transfer.
7. **`VLLM_CPU_KVCACHE_SPACE` is in GB.** Set it to 10 in Phase 1 (laptop), 1 in the multi-replica local Docker (so three replicas fit in 24 GB RAM).
8. **GVNIC + jumbo frames are not optional for Phase 3.** Without them, `(NUM_NODES−1)/NUM_NODES × tokens × bytes_per_token` of KV blocks at ~15 Gbps is slower than full prefill ⇒ defeats the architecture's premise.
9. **CPU-only absolute numbers from Phase 1 are not production-comparable.** Use them only for relative ranking. GPU runs in Phase 2 are expected to compress all absolute latencies 10–100×, but the relative ordering should hold.
10. **n = 3 trials → overlapping CIs.** Phase 2 must move to ≥ 10 trials before claiming statistical significance with Wilcoxon.
11. **BERTScore is officially deprecated** as the primary quality lens for this project. NLI-Faithfulness with `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` is the correct metric.
12. **macOS / ARM64 was the original constraint** — vLLM's official wheels are CUDA-only. The team migrated the Phase-2 builds to Ubuntu 22.04/24.04 (`setup_ubuntu.sh`) and prefers Docker images on GCP rather than source builds.
13. **Cross-trial prompt-cache carry-over.** vLLM is restarted per-baseline only. To get truly cold trials, restart vLLM between trials — currently a known optimism source in `hybrid_retrieval_cache_cold` aggregate numbers.

---

## 16. Immediate Next Steps (action list for the next session)

1. **Capture provenance fixes** in `metadata.json` — git commit, backend version, model id were null in some Phase 1 outputs.
2. **Author `scripts/statistical_tests.py`** — Wilcoxon rank-sum + bootstrap CI generator, output formatted LaTeX tables to `analysis/phase2/tables/`.
3. **Provision Phase 2 GCP cluster** ([GCP_DEPLOYMENT_RUNBOOK.md](CAGE/GCP_DEPLOYMENT_RUNBOOK.md) §3) — apply Terraform with default L4 settings.
4. **Run full Phase 2 sweep:** `bash scripts/cloud_run.sh Qwen/Qwen3-8B 100 10` then `bash scripts/run_compression.sh Qwen/Qwen3-8B` + `bash scripts/run_phase5.sh`. See [PHASE2_CHECKLIST.md](PHASE2_CHECKLIST.md).
5. **Scaling sweep:** repeat at `--model Qwen/Qwen3-4B` and `Qwen/Qwen3-14B` to produce model-size curves.
6. **Add TriviaQA** to the workload via `--dataset trivia_qa` — index already prebuilt under `experiments/ir_index/ir_trivia_qa_intfloat_e5-large-v2/`.
7. **Restart vLLM between trials** in the experiment driver (fix the cold-start carry-over caveat).
8. **Regenerate publication plots** with Phase 2 results: `python3 scripts/generate_publication_plots.py --results-dir analysis/phase2/results --output-dir analysis/phase2/images`.
9. **Update `my-article.tex`** §4 with GPU numbers + statistical-significance table; fix the abstract typo ("of" → "by").
10. **Tear down the GCP cluster** (`terraform destroy`) immediately after a sweep — A100 burns ~\$3.67/hr each.
11. **Begin Phase 3 prep:** prototype the real cross-node KV transfer in `src/orchestration/router.py` (gRPC or NCCL), validate transfer-time < 100 ms on GVNIC.

---

## 17. Key file index (quick-jump)

> Paths are relative to the repo root `/Users/lucasmariano/CAGE/`. Docs are siblings of this file under `docs/`.

| Purpose | Path |
|---|---|
| Doc index + status | [docs/README.md](README.md) |
| Setup / deploy / run (authoritative) | [docs/RUNBOOK.md](RUNBOOK.md) |
| Limitations & SOTA review | [docs/VALIDATION_AND_SOTA_REVIEW.md](VALIDATION_AND_SOTA_REVIEW.md) |
| Dataset card | [docs/DATA_CARD.md](DATA_CARD.md) |
| KV format spec (forward-looking) | [docs/KV_FORMAT_SPEC.md](KV_FORMAT_SPEC.md) |
| Academic paper | [paper/my-article.tex](../paper/my-article.tex) |
| Master runner | [scripts/run_experiment.py](../scripts/run_experiment.py) |
| Significance tests | [scripts/statistical_tests.py](../scripts/statistical_tests.py) |
| Distributed router | [src/orchestration/router.py](../src/orchestration/router.py) |
| Baselines registry | [src/orchestration/baselines.py](../src/orchestration/baselines.py) |
| KV cache manager (simulated) | [src/orchestration/cache_manager.py](../src/orchestration/cache_manager.py) |
| IR / retrieval | [src/orchestration/ir.py](../src/orchestration/ir.py) |
| Quality eval (NLI + LettuceDetect) | [src/evaluation/quality.py](../src/evaluation/quality.py) |
| Perf eval | [src/evaluation/performance.py](../src/evaluation/performance.py) |
| vLLM adapter | [src/inference/vllm_adapter.py](../src/inference/vllm_adapter.py) |
| Prompting | [src/utils/prompting.py](../src/utils/prompting.py) |
| CPU compose | [docker/docker-compose.yml](../docker/docker-compose.yml) |
| GPU compose | [docker/docker-compose.gpu.yml](../docker/docker-compose.gpu.yml) |
| Terraform | [terraform/gcp/main.tf](../terraform/gcp/main.tf) |
| K8s manifests | [k8s/](../k8s/) |
| Phase 1 source-of-truth | [analysis/phase1/results/](../analysis/phase1/results/) |
| Phase 1 plots | [results/images/](../results/images/) |

---

*End of CAGE Knowledge Base. Anything beyond this file lives in the documents linked above and the source tree under `/Users/lucasmariano/CAGE/`.*
