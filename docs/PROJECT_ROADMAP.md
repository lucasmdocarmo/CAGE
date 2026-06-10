> <!-- CAGE-DOC-STATUS -->
> **⚠️ STATUS: SUPERSEDED / HISTORICAL (2026-06-09).** This document predates the
> June 2026 reorganization and metric fixes. It is kept for history only and
> contains stale paths, pre-fix metric numbers, and/or invalid CLI flags.
> **Authoritative docs:** [`KNOWLEDGE_BASE.md`](KNOWLEDGE_BASE.md) (project reference),
> [`RUNBOOK.md`](RUNBOOK.md) (setup/deploy/run), [`VALIDATION_AND_SOTA_REVIEW.md`](VALIDATION_AND_SOTA_REVIEW.md) (limitations).
> See [`README.md`](README.md) for the doc index.

# CAGE Project Roadmap

> **Last updated**: 2026-03-26
> **Purpose**: Consolidated reference for continuing the CAGE project across sessions. Contains Phase 1 results, Phase 2 and Phase 3 plans, implementation tasks, and execution instructions.

---

## Project Overview

**CAGE** (Cache-Augmented Generation Evaluation) is a framework for evaluating cache-aware LLM serving systems. It combines serving metrics (latency, TTFT, TPOT, throughput), cache/retrieval telemetry, and semantic quality analysis (faithfulness, relevance, BERTScore, ROUGE-L) under a common evaluation protocol.

**Repository**: `/Users/lucasmariano/projects/cag-llm-kvcache`

**Paper**: `analysis/Articles/main-12page.tex` (SBC template format)

**Key citation**: Chan et al. (2025) — "Don't Do RAG" — argues cache-augmented generation can replace retrieval when knowledge is stable.

---

## Repository Structure

```
cag-llm-kvcache/
├── src/                    # Core framework code
│   ├── data/               # Dataset loading (SQuADv2)
│   ├── inference/          # vLLM serving client
│   ├── evaluation/         # Quality metrics (RAGAS, BERTScore, ROUGE)
│   ├── orchestration/      # Baseline orchestrator, router
│   ├── monitoring/         # Telemetry collection
│   └── utils/              # Shared utilities
├── scripts/
│   ├── run_experiment.py           # Main experiment runner
│   ├── generate_publication_plots.py  # 16 numbered plots
│   ├── generate_compact_figures.py    # Composite/split figures
│   ├── manage_vllm_cluster.py      # Multi-replica management
│   ├── verify_results.py           # Data validation
│   └── download_datasets.py        # Dataset setup
├── configs/
│   ├── dataset/            # Dataset configs
│   ├── experiment/         # Experiment configs per baseline
│   └── model/              # Model configs
├── analysis/
│   ├── phase1/
│   │   ├── results/        # 7 baseline directories with aggregated_metrics.json
│   │   └── images/         # Generated plots (01-16 + compact)
│   └── Articles/
│       ├── main-12page.tex # Paper (SBC format)
│       ├── Publish/images/ # Images referenced by LaTeX
│       ├── sbc-template.sty
│       └── sbc.bst
├── docker/                 # Router Dockerfile
├── docker-compose.yml      # Local CPU deployment
├── docker-compose.gpu.yml  # GPU deployment
├── terraform/gcp/          # GCP infrastructure (main.tf)
├── k8s/                    # Kubernetes manifests (router, replicas, redis)
├── vLLM/                   # vLLM source (forked/vendored)
├── tests/                  # Unit tests
└── experiments/
    ├── notebooks/analysis.ipynb
    └── ir_index/           # FAISS retrieval index
```

---

## Phase 1: Local Validation (COMPLETED)

### Objective
Validate the CAGE protocol locally by comparing 7 baselines under common experimental conditions on CPU.

### Setup
- **Hardware**: Apple M4 Pro, 24GB unified memory, 12 cores
- **Model**: Qwen/Qwen3-4B (vLLM CPU backend)
- **Dataset**: SQuADv2 (validation split)
- **KV cache**: 10GB allocation (`VLLM_CPU_KVCACHE_SPACE=10`)
- **Queries**: 50 per trial × 3 trials × 7 baselines = 1,050 measured queries
- **Generation**: max_tokens=100, temperature=0.7, top_p=0.95
- **Retrieval**: intfloat/e5-large-v2 embeddings, BAAI/bge-reranker-large, top-k=3

### Baselines
1. **No Cache** — gold passage, no reuse (control)
2. **Prefix Cache** — gold passage + vLLM native prefix caching
3. **RAG** — dense retrieval + reranking, no caching
4. **Redis Retrieval Cache Cold** — retrieval artifacts cached in Redis (cold start)
5. **Hybrid Cache Cold** — retrieval + prefix caching (cold start)
6. **Hybrid Cache Warm** — retrieval + prefix caching (50 warmup queries)
7. **Distributed Router Replicated** — 3 replicas, hash-based prefix routing

### Verified Results (all data in `analysis/phase1/results/*/aggregated_metrics.json`)

**Performance (vs No Cache)**:
- Prefix Cache: -37.4% latency, -65.7% TTFT, +58.3% QPS, lowest TPOT (76.4ms)
- RAG: +70.4% latency, +171.4% TTFT, -45.1% QPS
- Hybrid Warm: -17.1% latency, -59.7% TTFT (best retrieval-backed)
- Distributed: -23.7% TTFT but +15.5% latency (routing overhead)

**Quality**:
- Prefix Cache faithfulness identical to No Cache: 0.570
- RAG faithfulness: 0.504 (-11.6%)
- Distributed faithfulness: 0.636 (highest, but gold context + variance)
- BERTScore: 0.324-0.328 (near-constant, not discriminative)
- Relevance: 0.505 (gold) vs 0.525 (retrieved) — marginal

**Tail Latency**:
- Distributed: 7.6× p95/p50 TTFT spread (cold replica effect)
- RAG/Redis: 1.3× (stable but slow)
- Hybrid Cold: ±4,290ms TTFT std (trial-1 cold-start)

**Cache Telemetry**:
- All retrieval baselines: 98% retrieval hit rate (but doesn't explain performance)
- Prompt-cached ratio: Prefix Cache 68.4%, Hybrid Cold 75.6%, Hybrid Warm 89.2%
- Retrieval-cache rate: RAG/Redis/Hybrid-Cold 0%, Hybrid-Warm 100%

### How to re-run Phase 1
```bash
# Run all baselines
python scripts/run_experiment.py --phase 1 --all-baselines

# Or individual baseline
python scripts/run_experiment.py --baseline prefix_cache --trials 3 --queries 50

# Generate plots
python scripts/generate_publication_plots.py \
  --results-dir analysis/phase1/results \
  --output-dir analysis/phase1/images

# Generate compact figures
python scripts/generate_compact_figures.py

# Verify data
python scripts/verify_results.py
```

### Phase 1 Limitations (to address in Phase 2)
- CPU-only: absolute latency values (10-27s) are not production-relevant
- n=3 trials: insufficient for statistical significance tests
- Single dataset (SQuADv2): unclear if findings generalize
- Single model (Qwen3-4B): unclear if findings scale
- BERTScore/relevance showed minimal variance: may not be useful metrics
- No Cache baseline uses gold passage: comparison with RAG is not entirely fair

---

## Phase 2: GPU Validation and Model Scaling (NEXT)

### Objective
Confirm Phase 1 findings hold on GPU, scale to larger models, increase statistical rigor, and test dataset generalization.

### Task 2.1: GPU Infrastructure Setup
**What**: Deploy vLLM on GPU (single-node first) with the same 7 baselines.
**Why**: CPU latencies (10-27s) are not production-relevant. GPU will produce ms-range latencies and exercise real memory pressure.
**How**:
1. Use `docker-compose.gpu.yml` (already exists) with an NVIDIA GPU
2. Configure `VLLM_GPU_MEMORY_UTILIZATION` instead of `VLLM_CPU_KVCACHE_SPACE`
3. Start with a single A100/L4 GPU (local or GCP)
4. Run the same 7 baselines with identical configs
5. Compare relative rankings (percentages) against Phase 1

**Files to modify**:
- `docker-compose.gpu.yml` — verify GPU configuration
- `configs/experiment/*.json` — may need `api_base` updates
- `scripts/run_experiment.py` — should work unchanged (it hits the vLLM HTTP API)

**Expected outcome**: Relative rankings should hold (Prefix Cache > Hybrid Warm > No Cache > ...) even if absolute values change by 10-100×.

### Task 2.2: Model Scaling
**What**: Run baselines with larger models (Qwen3-7B, Qwen3-14B, or Llama 3.1 8B/70B).
**Why**: Phase 1 used 4B parameters. Prefix caching benefit may grow with model size as KV cache becomes proportionally more expensive.
**How**:
1. Select 2-3 model sizes (e.g., 4B, 8B, 14B)
2. Run all 7 baselines for each model size
3. Plot scaling curves: latency improvement (%) vs model size
4. Track GPU memory usage and KV cache eviction rates

**Files to modify**:
- `configs/model/` — add new model configs
- `scripts/run_experiment.py` — pass `--model` parameter

**Open questions**:
- Does prefix caching become more valuable with larger models?
- At what model size does KV cache pressure cause evictions?
- Does the Distributed baseline benefit more from larger models (more prefix to share)?

### Task 2.3: Statistical Rigor
**What**: Increase trials from 3 to 10+, add significance tests.
**Why**: Phase 1 has n=3 trials with overlapping confidence intervals. A reviewer will ask if differences are significant.
**How**:
1. Run 10 trials per baseline per model (vs 3 in Phase 1)
2. Use 100+ queries per trial (vs 50 in Phase 1)
3. Add bootstrap confidence intervals or Wilcoxon rank-sum tests to the analysis pipeline
4. Report p-values for key comparisons (Prefix Cache vs No Cache, RAG vs No Cache)

**Files to modify**:
- `scripts/run_experiment.py` — increase `--trials` and `--queries`
- `scripts/generate_publication_plots.py` — add CI/significance annotations to plots
- New script: `scripts/statistical_tests.py`

### Task 2.4: Dataset Diversity
**What**: Add at least one more QA dataset (Natural Questions, TriviaQA, or a domain-specific dataset).
**Why**: Phase 1 used only SQuADv2. Findings may be specific to its gold-passage structure.
**How**:
1. Add dataset loader for Natural Questions or TriviaQA in `src/data/`
2. Build FAISS index for the new dataset (`scripts/download_datasets.py`)
3. Run all baselines on the new dataset
4. Compare: do the same baselines win? Does RAG perform relatively better on a harder dataset?

**Files to modify**:
- `src/data/` — new dataset loader
- `configs/dataset/` — new dataset config
- `scripts/download_datasets.py` — add download logic

### Task 2.5: Quality Metric Evaluation
**What**: Assess whether BERTScore and relevance are useful, consider adding exact match / F1.
**Why**: In Phase 1, BERTScore ranged 0.324-0.328 (delta=0.004) — effectively constant. Relevance had only two values (0.505 / 0.525).
**How**:
1. Add exact-match and token-level F1 metrics to `src/evaluation/`
2. Re-evaluate Phase 1 outputs with the new metrics
3. If BERTScore/relevance remain non-discriminative, consider dropping them from the main paper and reporting only faithfulness + F1

### Phase 2 Deliverables
- Updated paper (extended to 18-page version) with GPU results
- Scaling curves (model size vs latency improvement)
- Statistical significance tables
- Multi-dataset comparison
- Updated `analysis/phase2/results/` directory structure

---

## Phase 3: Distributed and HPC Evaluation

### Objective
Evaluate real cross-node KV cache transfer, disaggregated prefilling, speculative decoding, and retrieval fallback under CAGE. Deploy on GCP.

### Task 3.1: GCP Infrastructure
**What**: Provision multi-node GPU cluster on Google Cloud Platform.
**Why**: Phase 1-2 are single-machine. Phase 3 needs real network latency between nodes.
**How**:
1. Use existing Terraform config: `terraform/gcp/main.tf`
2. Provision 2-4 Compute Engine instances with GPUs (A100 or L4)
3. Use Container-Optimized OS with Docker
4. Push experiment Docker images to Artifact Registry
5. Upload results to Cloud Storage bucket

**Files**:
- `terraform/gcp/main.tf` — already scaffolded, needs GPU instance types
- `terraform/gcp/terraform.tfvars.example` — fill in project/region/zone
- `docker/router.Dockerfile` — router container
- `k8s/` — Kubernetes manifests for multi-replica deployment (optional)

**Execution**:
```bash
cd terraform/gcp
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with GCP project, region, GPU type
terraform init
terraform plan
terraform apply
```

### Task 3.2: Real Cross-Node KV Cache Transfer
**What**: Implement and measure actual KV tensor transfer between replicas.
**Why**: Phase 1's Distributed baseline used hash routing with 0 transfers. Phase 3 needs real transfer to measure cost.
**How**:
1. Extend `src/orchestration/` router to support KV cache transfer between replicas
2. Implement transfer via gRPC or NCCL between vLLM instances
3. Measure: transfer latency (ms), transfer bytes (MB), transfer hit rate
4. New baselines: `distributed_sharded` (sharded KV), `distributed_migrated` (on-demand migration)

**Files to modify**:
- `src/orchestration/router.py` — add transfer logic
- `scripts/manage_vllm_cluster.py` — manage multi-node cluster
- New: `src/orchestration/kv_transfer.py`

### Task 3.3: Disaggregated Prefilling
**What**: Run prefill and decode on separate GPU pools (as in DistServe).
**Why**: vLLM supports experimental disaggregated prefilling. This separates TTFT from TPOT optimization.
**How**:
1. Enable vLLM's disaggregated prefilling mode (`--enable-disagg-prefill`)
2. Deploy prefill workers and decode workers on separate GPUs
3. Measure TTFT and TPOT independently
4. Compare against monolithic serving from Phase 2

**Reference**: vLLM disaggregated prefilling docs (cited as `vllm2024disagg` in paper)

### Task 3.4: Speculative Decoding
**What**: Add speculative decoding baselines (draft model or n-gram).
**Why**: Speculative decoding can reduce TPOT. Combined with prefix caching, this may yield compounding benefits.
**How**:
1. Configure vLLM with `--speculative-model` (small draft model)
2. New baselines: `prefix_cache_speculative`, `no_cache_speculative`
3. Measure: does speculation + caching compound the latency benefit?
4. Measure: does speculation affect quality (faithfulness)?

**Files to modify**:
- `configs/experiment/` — add speculative configs (`speculative_model`, `num_speculative_tokens`)
- `scripts/run_experiment.py` — already supports `speculative_model` in baseline config

### Task 3.5: Retrieval Fallback
**What**: Evaluate hybrid baselines that fall back to retrieval when cache misses.
**Why**: In production, not all queries have warm cache. The system must decide when to retrieve vs reuse.
**How**:
1. Implement a fallback policy in `src/orchestration/`: if cache miss → retrieve → cache result
2. Measure: cache miss rate over time, fallback latency, quality under fallback
3. Compare against always-retrieve (RAG) and always-cache (Prefix Cache)

### Task 3.6: Eviction Policies
**What**: Test different KV cache eviction strategies under memory pressure.
**Why**: With limited GPU memory and many concurrent requests, eviction policy determines which prefixes survive.
**How**:
1. Run experiments with constrained GPU memory (e.g., 50% utilization limit)
2. Monitor vLLM's eviction behavior (which prefixes get evicted)
3. Test: LRU, frequency-based, prefix-length-based eviction
4. Measure impact on cache hit rate and TTFT

### Phase 3 Deliverables
- Multi-node experiment results with real transfer costs
- Disaggregated prefilling comparison
- Speculative decoding interaction analysis
- Retrieval fallback evaluation
- Full GCP-reproducible pipeline
- Extended paper (journal-length or second conference paper)

---

## Paper Status

**Current**: `analysis/Articles/main-12page.tex` — SBC template format, ~12 pages
**Template**: SBC (sbc-template.sty + sbc.bst in Articles directory)
**Bibliography**: 13 references, manually formatted in `\begin{thebibliography}`

### Verified data accuracy
All 11 numerical claims in the paper have been validated against `aggregated_metrics.json`:
- Prefix Cache: 37.4% lat, 65.7% TTFT, 58.3% QPS — all correct
- RAG: 70.4% lat, 171.4% TTFT, 11.6% faith — all correct
- Distributed: 23.7% TTFT, 15.5% lat overhead, 7.6× tail — all correct
- Prefix Cache faithfulness == No Cache: True

### Known remaining issues
- Abstract: "reduced latency of 37.4%" — should be "reduced latency by 37.4%"
- Cache telemetry text mentions 0%/100% retrieval-cache rate not shown in Table
- Some commented-out `\includegraphics` options in figure blocks (cosmetic)

---

## Execution Commands Reference

### Run experiments
```bash
# Phase 1 (CPU, local)
python scripts/run_experiment.py --baseline <name> --trials 3 --queries 50

# Phase 2 (GPU, local or cloud)
python scripts/run_experiment.py --baseline <name> --trials 10 --queries 100 --model Qwen/Qwen3-14B
```

### Generate plots
```bash
# All 16 publication plots
python scripts/generate_publication_plots.py \
  --results-dir analysis/phase1/results \
  --output-dir analysis/phase1/images

# Compact/split figures for paper
python scripts/generate_compact_figures.py
```

### Copy images to paper
```bash
# Copy specific plots to the Publish directory for LaTeX
cp analysis/phase1/images/16_ttft_tail_latency.png \
   analysis/Articles/Publish/images/plots/phase1/
```

### Infrastructure
```bash
# Local Docker (CPU)
docker-compose up -d

# Local Docker (GPU)
docker-compose -f docker-compose.gpu.yml up -d

# GCP Terraform
cd terraform/gcp && terraform init && terraform apply

# Multi-replica cluster
python scripts/manage_vllm_cluster.py --replicas 3 --start
```

### Validate data
```bash
# Verify all results
python scripts/verify_results.py

# Quick data check
python -c "
import json
data = json.load(open('analysis/phase1/results/prefix_cache/aggregated_metrics.json'))
print(data['performance']['avg_latency_ms']['mean'])
"
```

---

## Key Decisions and Context for Future Sessions

1. **Gold vs Retrieved context**: No Cache and Prefix Cache use the gold SQuAD passage. RAG/Redis/Hybrid use retrieved passages. This means the comparison is not purely about caching — it also involves context source quality. The paper acknowledges this explicitly.

2. **Distributed baseline has no real transfer**: The Distributed Router Replicated baseline uses hash-based routing to 3 replicas but 0 KV transfers occurred. It is a routing/orchestration proof-of-concept, not a distributed KV benchmark. Phase 3 will add real transfers.

3. **BERTScore is not discriminative**: Range 0.324-0.328 across all baselines. Faithfulness is the primary quality metric. Consider dropping BERTScore in future papers or replacing with exact-match F1.

4. **SBC template compliance**: Paper uses sbc-template.sty with period-separated captions (`labelsep=period`), `flushbottom`, and `\bibliographystyle{sbc}`. Do not override these with custom settings.

5. **Tail latency matters**: Distributed's 7.6× p95/p50 TTFT spread is a key finding. Mean metrics alone are insufficient for characterizing cache-aware serving behavior.

6. **Chan et al. citation**: Used 4 times in the paper (intro, related work, setup, discussion) — each use is justified and not redundant.
