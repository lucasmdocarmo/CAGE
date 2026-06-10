> <!-- CAGE-DOC-STATUS -->
> **⚠️ STATUS: SUPERSEDED / HISTORICAL (2026-06-09).** This document predates the
> June 2026 reorganization and metric fixes. It is kept for history only and
> contains stale paths, pre-fix metric numbers, and/or invalid CLI flags.
> **Authoritative docs:** [`KNOWLEDGE_BASE.md`](KNOWLEDGE_BASE.md) (project reference),
> [`RUNBOOK.md`](RUNBOOK.md) (setup/deploy/run), [`VALIDATION_AND_SOTA_REVIEW.md`](VALIDATION_AND_SOTA_REVIEW.md) (limitations).
> See [`README.md`](README.md) for the doc index.

# CAGE — Complete Project Overview

**Last Updated:** 2026-04-08
**Author:** Lucas Mariano do Carmo
**Institution:** Pontifícia Universidade Católica de Minas Gerais (PUC Minas)
**Contact:** lucas.mariano.carmo@gmail.com

---

## 1. What Is CAGE?

CAGE (Cache-Augmented Generation Evaluation) is the first comprehensive benchmarking framework for evaluating Cache-Augmented Generation (CAG) systems against traditional Retrieval-Augmented Generation (RAG) approaches. It measures the trade-offs between serving performance (throughput, latency, TTFT) and semantic output quality (faithfulness, relevance, completeness) across multiple caching strategies in distributed LLM serving environments.

The core research question is inspired by Chan et al. (2025) — "Don't Do RAG" — which argues that caching KV tensors of document context can replace retrieval when the knowledge base is stable. CAGE provides the tooling to quantify when this claim holds and when it breaks down.

---

## 2. The Research Problem

When LLMs need external knowledge, two approaches dominate:

**RAG (Retrieval-Augmented Generation):** For every query, retrieve relevant documents from a vector store, embed them in the prompt, then generate. The LLM recomputes attention over the context every time.

**CAG (Cache-Augmented Generation):** Pre-compute KV (Key-Value) cache states for documents once, then load cached attention states directly for subsequent queries. Dramatically faster for repeated or overlapping contexts.

CAG introduces new trade-offs:
- KV caches are large (hundreds of MB per document set)
- In distributed systems, cache must be shared across nodes, adding transfer cost
- Cache eviction under memory pressure degrades hit rates
- Not all cached content is relevant to every query

**No framework existed to systematically benchmark these trade-offs.** CAGE fills this gap.

---

## 3. Repository Structure

```
cag-llm-kvcache/
├── src/                          # Core framework code
│   ├── data/loader.py            # Dataset loading (CAGExample objects)
│   ├── inference/
│   │   ├── engine.py             # Abstract inference interface
│   │   └── vllm_adapter.py       # vLLM HTTP client (streaming, TTFT, telemetry)
│   ├── evaluation/
│   │   ├── quality.py            # Faithfulness, relevance, BERTScore, ROUGE-L
│   │   └── performance.py        # Latency, throughput, cache metrics aggregation
│   ├── orchestration/
│   │   ├── baselines.py          # 7 baseline type definitions + configs
│   │   ├── router.py             # FastAPI prefix-hash router for distributed
│   │   ├── ir.py                 # FAISS dense retrieval + reranking
│   │   ├── redis_cache.py        # Redis retrieval-artifact cache
│   │   └── cache_manager.py      # Abstract KV cache distribution policies
│   └── utils/prompting.py        # Standardized prompt templates
├── scripts/
│   ├── run_experiment.py         # Main experiment runner (CLI entry point)
│   ├── generate_publication_plots.py  # 16 numbered publication plots
│   ├── generate_compact_figures.py    # Composite figures for paper
│   ├── manage_vllm_cluster.py    # Multi-replica cluster management
│   ├── verify_results.py         # Data integrity verification
│   └── download_datasets.py      # Dataset setup
├── configs/
│   ├── dataset/                  # squad_v2.yaml, hotpotqa.yaml
│   ├── experiment/baseline.yaml  # Default experiment configuration
│   └── model/                    # qwen3-4b, qwen3-8b, qwen3-14b, etc.
├── analysis/
│   ├── phase1/
│   │   ├── results/              # 7 baseline directories with aggregated_metrics.json
│   │   └── images/               # Generated plots (01–16 + compact)
│   └── Articles/
│       ├── main-12page.tex       # Paper (SBC template format)
│       └── Publish/images/       # Figures referenced by LaTeX
├── documentation/                # Project documentation (this folder)
├── docker/                       # Router Dockerfile
├── docker-compose.yml            # Local CPU multi-container deployment
├── docker-compose.gpu.yml        # GPU deployment
├── terraform/gcp/main.tf         # GCP infrastructure-as-code
├── k8s/                          # Kubernetes manifests (router, replicas, redis)
├── vLLM/                         # Vendored vLLM source (reference, not imported)
├── tests/                        # Unit + integration tests
├── experiments/
│   ├── notebooks/analysis.ipynb  # Jupyter analysis
│   └── ir_index/                 # Persisted FAISS retrieval indexes
├── logs/                         # vLLM and experiment logs
├── README.md                     # Quick-start project overview
├── TECHNICAL_ARCHITECTURE.md     # Deep technical reference
├── PROJECT_ROADMAP.md            # Phase-by-phase roadmap with tasks
└── PROJECT_OVERVIEW.md           # This file
```

---

## 4. Baselines Evaluated

CAGE compares 7 serving configurations, each isolating a specific architectural choice:

**1. No Cache** — Gold passage from dataset, no KV reuse. Full prefill recomputation for every request. The worst-case control baseline.

**2. Prefix Cache** — Gold passage + vLLM's native automatic prefix caching (`--enable-prefix-caching`). Shared prompt prefixes produce cached KV blocks that subsequent requests reuse.

**3. RAG** — FAISS dense retrieval (intfloat/e5-large-v2) + BGE-reranker-large reranking. Gold passage is NOT used — context is retrieved. No prefix caching. Represents standard industry RAG.

**4. Redis Retrieval Cache Cold** — Same as RAG, but retrieval results are cached in Redis. Cold start: Redis flushed before run, so all queries miss the cache.

**5. Hybrid Cache Cold** — Retrieval via FAISS + reranker, plus vLLM prefix caching active. Redis flushed (cold). First trial suffers cold-start; subsequent trials benefit from warmed prefix cache.

**6. Hybrid Cache Warm** — Same as Hybrid Cold, but 50 warmup queries populate both Redis and vLLM prefix cache before measurement begins. Represents steady-state production behavior.

**7. Distributed Router Replicated** — 3 vLLM replicas (ports 8001/8002/8003) behind a FastAPI prefix-hash router (port 9000). Gold passage used. Prefix caching active per-replica.

---

## 5. Tasks Completed (Phase 1)

### 5.1 Phase 1 Experiments — COMPLETE

**Objective:** Validate the CAGE evaluation protocol locally by comparing all 7 baselines under identical controlled conditions on CPU.

**Setup:**
- Hardware: Apple M4 Pro, 24 GB unified memory, 12 cores
- Model: Qwen/Qwen3-4B via vLLM CPU backend
- Dataset: SQuAD v2 (validation split), 50 queries × 3 trials × 7 baselines = 1,050 measured queries
- KV cache: 10 GB allocation (`VLLM_CPU_KVCACHE_SPACE=10`)
- Generation: max_tokens=100, temperature=0.7, top_p=0.95
- Retrieval: intfloat/e5-large-v2 embeddings, BAAI/bge-reranker-large reranker, top-k=3
- Date: March 19, 2026

### 5.2 Validated Phase 1 Results

All numbers below are computed from `analysis/phase1/results/*/aggregated_metrics.json`. Every numerical claim in the paper has been verified against these files.

**Performance (mean across 3 trials):**

| Baseline | Avg Latency (ms) | Avg TTFT (ms) | QPS | Avg TPOT (ms) |
|---|---|---|---|---|
| No Cache | 16,006 | 6,919 | 0.0606 | 90.9 |
| Prefix Cache | 10,015 | 2,376 | 0.0959 | 76.4 |
| RAG | 27,270 | 18,775 | 0.0333 | 84.9 |
| Redis Cold | 26,853 | 18,579 | 0.0342 | 82.7 |
| Hybrid Cold | 15,513 | 6,001 | 0.0589 | 95.1 |
| Hybrid Warm | 13,269 | 2,791 | 0.0674 | 104.8 |
| Distributed | 18,492 | 5,279 | 0.0525 | 132.1 |

**Performance Improvements vs No Cache:**
- Prefix Cache: −37.4% latency, −65.7% TTFT, +58.3% QPS, lowest TPOT (76.4 ms)
- RAG: +70.4% latency, +171.4% TTFT, −45.1% QPS (worst overall)
- Hybrid Warm: −17.1% latency, −59.7% TTFT (best retrieval-backed baseline)
- Distributed: −23.7% TTFT but +15.5% latency (routing overhead)

**Quality:**
- Prefix Cache faithfulness is identical to No Cache: 0.570 (both use gold passage)
- RAG faithfulness: 0.504 (−11.6% vs No Cache)
- Distributed faithfulness: 0.636 (highest, but high variance from gold context)
- BERTScore: 0.324–0.328 across all baselines (not discriminative, delta = 0.004)
- Relevance: 0.505 (gold baselines) vs 0.525 (retrieval baselines) — marginal difference

**Cache Telemetry:**
- Prompt-cached ratio: Prefix Cache 68.4%, Hybrid Cold 75.6%, Hybrid Warm 89.2%
- Retrieval hit rate: 98% across all retrieval baselines
- Retrieval-cache rate: RAG/Redis/Hybrid-Cold 0%, Hybrid-Warm 100%

**Tail Latency (key finding):**
- Distributed p95/p50 TTFT spread: 7.6× (cold-replica effect on first access)
- RAG/Redis: 1.3× (stable but slow)
- Hybrid Cold: ±4,290 ms TTFT std (trial-1 cold-start drives variance)

### 5.3 Paper Draft — COMPLETE

- Location: `analysis/Articles/main-12page.tex` (SBC template format, ~12 pages)
- Template: sbc-template.sty + sbc.bst
- All 11 numerical claims verified against source data
- 16 publication plots generated in `analysis/phase1/images/`
- Compact composite figures for paper in `analysis/phase1/images/`

### 5.4 Infrastructure — COMPLETE

- Docker Compose (CPU + GPU) for local multi-replica deployment
- Kubernetes manifests for production deployment
- Terraform scaffolding for GCP GPU provisioning
- Automated experiment runner with CLI, multi-trial, and warmup support
- Prometheus metrics exposure (optional)
- Persisted FAISS indexes for reproducible retrieval

### 5.5 Test Suite — COMPLETE

Located in `tests/`:
- `test_inference.py`: VLLMAdapter request/response
- `test_ir.py`: FAISS index building and retrieval
- `test_baselines.py`: Baseline config loading
- `test_router_integration.py`: Router request forwarding
- `test_vllm_integration.py`: Integration against live vLLM
- `test_data.py`: Dataset loading

Run with: `pytest tests/ -v`

---

## 6. Tasks To Be Completed

### 6.1 Phase 2: GPU Validation and Model Scaling — NEXT

**Objective:** Confirm Phase 1 findings hold on GPU, scale to larger models, increase statistical rigor, and test dataset generalization.

**Task 2.1 — GPU Infrastructure Setup**
Deploy vLLM on a single GPU (A100 or L4) and rerun the same 7 baselines with identical configs. Use `docker-compose.gpu.yml`. The goal is to verify that relative baseline rankings (percentage improvements) are preserved even though absolute latency values will drop by 10–100×. Files: `docker-compose.gpu.yml`, `configs/experiment/*.yaml`.

**Task 2.2 — Model Scaling**
Run baselines with 2–3 model sizes (Qwen3-4B, Qwen3-8B, Qwen3-14B). Plot scaling curves: latency improvement (%) vs model size. Open questions: Does prefix caching benefit grow with model size? At what size does KV cache pressure cause evictions? Files: `configs/model/`.

**Task 2.3 — Statistical Rigor**
Increase from 3 trials to 10+, increase from 50 to 100+ queries. Add bootstrap confidence intervals or Wilcoxon rank-sum tests. Report p-values for key comparisons. New script needed: `scripts/statistical_tests.py`.

**Task 2.4 — Dataset Diversity**
Add at least one additional QA dataset (Natural Questions or TriviaQA). Build FAISS index, run all baselines, compare whether the same baselines win. Files: `src/data/`, `configs/dataset/`.

**Task 2.5 — Quality Metric Evaluation**
Assess whether BERTScore and relevance are useful (both were near-constant in Phase 1). Consider adding exact match and token-level F1. If non-discriminative, drop from main paper.

**Deliverables:** Updated paper with GPU results, scaling curves, statistical significance tables, multi-dataset comparison.

### 6.2 Phase 3: Distributed and HPC Evaluation — FUTURE

**Objective:** Evaluate real cross-node KV cache transfer, disaggregated prefilling, speculative decoding, and retrieval fallback under CAGE. Deploy on GCP.

**Task 3.1 — GCP Multi-Node Cluster**
Provision 2–4 GPU instances using `terraform/gcp/main.tf`. Push Docker images to Artifact Registry.

**Task 3.2 — Real Cross-Node KV Transfer**
Extend `src/orchestration/router.py` to support KV tensor transfer between replicas (gRPC or NCCL). Measure transfer latency (ms), bytes (MB), and hit rate. New baselines: `distributed_sharded`, `distributed_migrated`.

**Task 3.3 — Disaggregated Prefilling**
Enable vLLM's experimental disaggregated prefilling mode. Deploy prefill and decode workers on separate GPUs. Measure TTFT and TPOT independently.

**Task 3.4 — Speculative Decoding**
Configure vLLM with `--speculative-model` or n-gram speculation. New baselines: `prefix_cache_speculative`, `no_cache_speculative`. Measure whether speculation + caching compound latency benefits and whether quality is affected.

**Task 3.5 — Retrieval Fallback**
Implement a fallback policy: if cache miss → retrieve → cache result. Compare against always-retrieve (RAG) and always-cache (Prefix Cache).

**Task 3.6 — Eviction Policies**
Test KV cache eviction under constrained GPU memory (e.g., 50% utilization limit). Compare LRU, frequency-based, and prefix-length-based strategies.

**Deliverables:** Multi-node results with real transfer costs, disaggregated prefilling comparison, speculative decoding interaction, full GCP-reproducible pipeline.

---

## 7. Project Status Summary

| Component | Status | Notes |
|---|---|---|
| Phase 1 Experiments | ✅ Complete | 7 baselines, 3 trials, 50 queries each, all verified |
| Paper Draft (12-page) | ✅ Complete | SBC template, all claims validated |
| Publication Plots | ✅ Complete | 16 numbered + composite figures |
| Core Framework (src/) | ✅ Complete | Data, inference, evaluation, orchestration modules |
| Docker/K8s/Terraform | ✅ Scaffolded | CPU tested, GPU scaffolded, GCP needs tfvars |
| Tests | ✅ Complete | Unit + integration tests |
| Phase 2 (GPU) | 🔜 Next | Requires GPU access (A100/L4) |
| Phase 3 (Distributed) | 🔮 Future | Requires multi-node GCP deployment |
| Speculative Decoding | 🔮 Future | Config support exists, needs GPU to execute |

---

## 8. Tech Stack — Explained

### 8.1 Inference Engine: vLLM
vLLM is a high-throughput LLM serving engine with PagedAttention-based memory management. CAGE communicates with vLLM exclusively via the OpenAI-compatible HTTP API (`/v1/completions`). Key flags: `--enable-prefix-caching` (KV block reuse), `--enable-prompt-tokens-details` (cache telemetry in responses). The vendored `vLLM/` directory is for reference only — the experiment runner does not import from it.

### 8.2 Model: Qwen/Qwen3-4B
Chosen for Apple Silicon CPU-only inference support, reasonable size (8 GB RAM), and strong QA performance. Configs exist for Qwen3-8B, Qwen3-14B, and Qwen2.5-7B-Instruct for scaling experiments.

### 8.3 Retrieval: FAISS + SentenceTransformers
Dense retrieval uses `intfloat/e5-large-v2` embeddings stored in a FAISS `IndexFlatIP` index. Results are re-scored with `BAAI/bge-reranker-large` cross-encoder. Index is persisted under `experiments/ir_index/` for reproducibility.

### 8.4 Caching: Redis + vLLM Prefix Cache
Redis caches retrieval artifacts (query → retrieved doc IDs), NOT KV tensors. vLLM's built-in prefix caching handles KV block reuse natively. These are complementary: Redis avoids re-running FAISS+reranking; vLLM avoids recomputing attention.

### 8.5 Routing: FastAPI Prefix-Hash Router
A custom FastAPI service (`src/orchestration/router.py`) receives requests, hashes the prompt prefix (`sha256(prompt[:prefix_length]) % num_replicas`), and forwards to the appropriate vLLM replica. This maximizes per-replica prefix cache hits.

### 8.6 Quality Evaluation
- **Faithfulness:** NLI-based entailment using a cross-encoder. Splits generated answer into claims, checks each against context.
- **Relevance:** Embedding cosine similarity between question and context.
- **Completeness:** BERTScore (token-level soft F1) + ROUGE-L (longest common subsequence F1).

### 8.7 Performance Metrics
TTFT measured via streaming SSE (first `data:` chunk timestamp). TPOT computed as `(total_time - TTFT) / (num_tokens - 1)`. Prompt-cached ratio extracted from vLLM's `usage.prompt_tokens_details.cached_tokens` field.

### 8.8 Infrastructure
- **Docker Compose:** CPU (`docker-compose.yml`) and GPU (`docker-compose.gpu.yml`)
- **Kubernetes:** StatefulSet for replicas, Deployment for router, standalone Redis
- **Terraform:** GCP Compute Engine with GPU instances (scaffolded)
- **Monitoring:** Optional Prometheus histograms for TTFT, latency, cached ratio

### 8.9 Configuration
Hydra-style YAML configs under `configs/` for models, datasets, and experiments. CLI flags in `scripts/run_experiment.py` override config values.

---

## 9. How an Experiment Runs End-to-End

```
1. CLI parsing → baseline config loaded from src/orchestration/baselines.py
2. Dataset loaded → src/data/loader.py → SQuADv2 examples as CAGExample objects
3. IR index built (if retrieval baseline) → src/orchestration/ir.py → FAISS index
4. vLLM adapter created → src/inference/vllm_adapter.py → HTTP client to vLLM server
5. For each trial (1..N):
   a. For each query (1..M):
      i.   Prompt built → src/utils/prompting.py → format_qa_prompt()
      ii.  Context selected (gold passage OR retrieved via IR)
      iii. Request sent to vLLM → streaming HTTP → TTFT from first SSE chunk
      iv.  Response collected with telemetry (prompt_tokens, cached_prompt_tokens)
      v.   Quality scored → faithfulness, relevance, BERTScore
      vi.  Performance recorded → latency, TTFT, TPOT
   b. Trial metrics aggregated and saved to JSON
6. Cross-trial aggregation → aggregated_metrics.json written
```

Output structure per baseline:
```
analysis/phase1/results/<baseline_name>/
├── aggregated_metrics.json        # Mean ± std across all trials
├── trial_1/
│   └── <timestamp>_metrics.json   # Per-trial raw metrics
├── trial_2/...
└── trial_3/...
```

---

## 10. Key Decisions and Known Limitations

1. **Gold vs Retrieved Context:** No Cache, Prefix Cache, and Distributed use the gold SQuAD passage. RAG/Redis/Hybrid use retrieved passages. The comparison is not purely about caching — it also involves context source quality. The paper acknowledges this explicitly.

2. **Distributed Baseline Has No Real Transfer:** The Distributed Router Replicated baseline routes requests via prefix hash to 3 replicas, but 0 KV tensors were transferred between replicas. It is a routing/orchestration proof-of-concept. Phase 3 will add real cross-node transfers.

3. **BERTScore Is Not Discriminative:** Range 0.324–0.328 across all baselines (delta = 0.004). Faithfulness is the primary quality metric. Consider dropping BERTScore from main paper.

4. **CPU-Only Phase 1:** Absolute latency values (10–27 s) are not production-relevant. Relative rankings and percentage improvements are the valid conclusions.

5. **n=3 Trials:** Insufficient for rigorous statistical significance tests. Phase 2 will increase to 10+ trials.

6. **Single Dataset (SQuADv2):** Findings may not generalize to harder or multi-hop datasets. Phase 2 adds dataset diversity.

7. **Tail Latency Matters:** Distributed's 7.6× p95/p50 TTFT spread is a key finding. Mean metrics alone are insufficient for characterizing cache-aware serving.

---

## 11. Execution Commands Reference

### Run Experiments
```bash
# Single baseline
python scripts/run_experiment.py --baseline prefix_cache --trials 3 --queries 50

# Phase 2 (GPU, larger model)
python scripts/run_experiment.py --baseline prefix_cache --trials 10 --queries 100 --model Qwen/Qwen3-14B
```

### Generate Plots
```bash
python scripts/generate_publication_plots.py \
  --results-dir analysis/phase1/results \
  --output-dir analysis/phase1/images

python scripts/generate_compact_figures.py
```

### Validate Data
```bash
python scripts/verify_results.py
```

### Infrastructure
```bash
# Local CPU cluster
docker-compose up -d

# Local GPU cluster
docker-compose -f docker-compose.gpu.yml up -d

# GCP
cd terraform/gcp && terraform init && terraform apply

# Multi-replica management
python scripts/manage_vllm_cluster.py --replicas 3 --start
```

---

## 12. AI Handoff Context

This section is written explicitly for another AI model or engineer continuing this project.

### 12.1 What to Read First
1. This file (`PROJECT_OVERVIEW.md`) — complete project context
2. `TECHNICAL_ARCHITECTURE.md` — deep dive into how each module works
3. `PROJECT_ROADMAP.md` — phase-by-phase task breakdown with detailed instructions
4. `analysis/phase1/results/*/aggregated_metrics.json` — source of truth for all numbers

### 12.2 Current State
- **Phase 1 is complete and validated.** All 7 baselines have been run, data is in `analysis/phase1/results/`, and plots are in `analysis/phase1/images/`. The paper draft is at `analysis/Articles/main-12page.tex`.
- **Phase 2 has not started.** It requires a GPU (A100 or L4). The code is ready — `scripts/run_experiment.py` hits the vLLM HTTP API and will work unchanged with a GPU-backed vLLM server.
- **Phase 3 is future work.** It requires multi-node GCP infrastructure and real KV transfer implementation.

### 12.3 Critical Architectural Facts
- CAGE is the **orchestration/evaluation layer**. vLLM is the **inference engine**. CAGE communicates with vLLM only via HTTP — it does not import vLLM Python modules.
- The `vLLM/` directory is a vendored copy for reference. The experiment runner ignores it.
- All experiment data is self-contained in `analysis/phase1/results/`. Each baseline has an `aggregated_metrics.json` with mean/std/min/max/values for every metric.
- The paper uses SBC template format (`sbc-template.sty`). Do not override template settings.

### 12.4 What to Do Next
**If continuing the benchmark roadmap:**
1. Provision a GPU VM (A100 recommended, L4 acceptable)
2. Start vLLM with GPU: `vllm serve Qwen/Qwen3-4B --enable-prefix-caching --enable-prompt-tokens-details`
3. Run `scripts/run_experiment.py` with `--trials 10 --queries 100` for each baseline
4. Compare relative rankings (percentages) against Phase 1 — they should be preserved
5. Scale to larger models (Qwen3-8B, Qwen3-14B)

**If improving the paper:**
1. Read `analysis/Articles/main-12page.tex`
2. Known issues: "reduced latency of 37.4%" should be "by 37.4%"; cache telemetry text mentions 0%/100% retrieval-cache rate not shown in tables
3. All figures are in `analysis/Articles/Publish/images/plots/phase1/`

**If adding new baselines:**
1. Add baseline type to `BaselineType` enum in `src/orchestration/baselines.py`
2. Add config via `get_baseline_config()` function
3. Add handling logic in `scripts/run_experiment.py`

### 12.5 Environment Setup
```bash
conda create -n cage-vllm python=3.12 -y
conda activate cage-vllm
pip install -r requirements.txt
# Build vLLM for macOS CPU:
cd ~/projects/vllm && pip install -e .
export VLLM_CPU_KVCACHE_SPACE=10
export VLLM_CPU_OMP_THREADS_BIND=auto
```

### 12.6 Key File Relationships
```
run_experiment.py → loads baselines.py config
                  → loads loader.py dataset
                  → creates vllm_adapter.py client
                  → calls quality.py + performance.py evaluators
                  → writes to analysis/phase1/results/<baseline>/

generate_publication_plots.py → reads analysis/phase1/results/*/aggregated_metrics.json
                              → writes to analysis/phase1/images/

main-12page.tex → references analysis/Articles/Publish/images/plots/phase1/*.png
```

---

## 13. Citation

```bibtex
@article{carmo2025cage,
  title={CAGE: A Framework for Holistic Evaluation of Cache-Augmented Generation Models},
  author={Carmo, Lucas Mariano do},
  year={2025},
  institution={Pontifícia Universidade Católica de Minas Gerais}
}
```
