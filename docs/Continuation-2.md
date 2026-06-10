> <!-- CAGE-DOC-STATUS -->
> **⚠️ STATUS: SUPERSEDED / HISTORICAL (2026-06-09).** This document predates the
> June 2026 reorganization and metric fixes. It is kept for history only and
> contains stale paths, pre-fix metric numbers, and/or invalid CLI flags.
> **Authoritative docs:** [`KNOWLEDGE_BASE.md`](KNOWLEDGE_BASE.md) (project reference),
> [`RUNBOOK.md`](RUNBOOK.md) (setup/deploy/run), [`VALIDATION_AND_SOTA_REVIEW.md`](VALIDATION_AND_SOTA_REVIEW.md) (limitations).
> See [`README.md`](README.md) for the doc index.

# CAGE Project: Master Continuation & Deployment Guide

> **Last Updated:** May 2026
> **Purpose:** To serve as a standalone onboarding and deployment guide for migrating the CAGE project to a new machine (e.g., a GCP GPU instance). It contains exact setup instructions, execution commands, and the detailed roadmap for Phase 2 (GPU Scaling) and Phase 3 (Distributed HPC).

---

## 1. Machine Migration & Environment Setup

When you clone this repository onto a new machine (specifically a Linux GPU instance on GCP), follow these exact steps to restore the working state.

### A. System Requirements & Dependencies
1. **Operating System:** Ubuntu 22.04 / 24.04 (Recommended)
2. **Hardware:** NVIDIA GPU (L4, A100, or H100)
3. **Core Dependencies:** Python 3.12, Docker, NVIDIA Container Toolkit

### B. Setup Commands
Run the provided setup script to initialize the Python virtual environment and install core requirements:
```bash
# 1. Make scripts executable
chmod +x setup_ubuntu.sh

# 2. Run the environment setup (installs Python 3.12, pip, venv)
./setup_ubuntu.sh

# 3. Activate the environment
source cage-env/bin/activate

# 4. Install project requirements
pip install -r requirements.txt
```

### C. Data & Model Configuration
Before running any experiments, you must configure your HuggingFace token and download the datasets/indexes.
```bash
# 1. Export your HuggingFace token (required for vLLM to pull Qwen3 models)
export HF_TOKEN="your_huggingface_token_here"

# 2. Download the SQuADv2 dataset and build the FAISS index for the RAG baseline
python scripts/download_datasets.py
```

### D. Starting the Infrastructure (vLLM & Redis)
Instead of building vLLM from source, the recommended path for the GCP instance is utilizing the official GPU Docker images:
```bash
# Start the vLLM replicas and Redis cache via Docker Compose
docker-compose -f docker-compose.gpu.yml up -d
```

---

## 2. Executing Experiments & Analysis

Once the infrastructure is running, you can execute the CAGE evaluation pipeline.

### A. Running the Baselines
Use the main experiment runner to execute tests. The results will be automatically saved to `analysis/phase2/results/`.
```bash
# Run a specific baseline (e.g., prefix_cache)
python scripts/run_experiment.py --baseline prefix_cache --trials 10 --queries 100 --model Qwen/Qwen3-14B

# Run all baselines sequentially
python scripts/run_experiment.py --phase 2 --all-baselines --trials 10 --queries 100
```

### B. Generating Analysis & Plots
After the experiments complete, generate the publication-ready metrics and plots:
```bash
# Validate the integrity of the collected data
python scripts/verify_results.py

# Generate the 16 publication plots
python scripts/generate_publication_plots.py \
  --results-dir analysis/phase2/results \
  --output-dir analysis/phase2/images

# Generate composite figures for the LaTeX paper
python scripts/generate_compact_figures.py
```

---

## 3. Current Project State (Phase 1 Completed)

*   **Status:** Phase 1 (Local CPU Validation) is **COMPLETED**.
*   **Documentation:** `my-article.tex` has been fully updated with reviewer rebuttals.
*   **Key Findings Verified:**
    1.  **CPU Constraints:** Phase 1 structurally validated the orchestration logic but lacked true KV-cache eviction pressure.
    2.  **Faithfulness Metric:** The NLI-based Faithfulness score (ceiling 0.570 on Qwen3-4B) successfully and comparatively proved that RAG retrieves degrade factual grounding (-11.6% drop).
    3.  **BERTScore Deprecation:** BERTScore was proven to be entirely non-discriminative (0.324 flat), validating our choice of strict NLI metrics over soft-matching embedding metrics.

---

## 4. Immediate Next Tasks (Phase 2: GPU Validation)

**Objective:** Transition to cloud GPUs to induce authentic KV-cache memory pressure, scale model sizes, and increase statistical rigor.

1.  **Terraform GPU Provisioning:** 
    *   Execute `terraform apply` in `terraform/gcp/main.tf` to provision the G2 (NVIDIA L4) or A2 (NVIDIA A100) instances.
2.  **Model Scaling & Configuration Changes:** 
    *   Update `configs/model/` to transition from `Qwen3-4B` to **`Qwen3-8B`** and **`Qwen3-14B`**.
    *   Ensure `VLLM_GPU_MEMORY_UTILIZATION` is configured in `docker-compose.gpu.yml` to safely push the VRAM limits.
3.  **Statistical Rigor Update:** 
    *   Modify `scripts/run_experiment.py` (if necessary) to smoothly handle **10+ trials** and **100+ queries** per trial without HTTP timeouts.
    *   Develop `scripts/statistical_tests.py` to calculate p-values (Wilcoxon rank-sum) for the final paper.
4.  **Dataset Diversity:** 
    *   Extend `scripts/download_datasets.py` to support *Natural Questions* or *TriviaQA* to ensure Phase 1 findings generalize beyond SQuADv2.

---

## 5. Future Research & Implementation (Phase 3: HPC)

**Objective:** Evaluate the architecture under distributed high-performance computing constraints.

1.  **Cross-Node KV Tensor Transfer:** 
    *   Implement actual byte-level KV tensor transfer logic between vLLM replicas (currently, the router only forwards API requests). 
    *   **Architecture Requirement:** This requires GCP instances with **gVNIC** enabled and **NCCL Fast Socket plugins** to prevent network bottlenecks.
2.  **vLLM Disaggregated Prefilling:** 
    *   Enable vLLM's experimental `--enable-disagg-prefill` mode. Deploy prefill-only workers and decode-only workers to separate GPU instances to optimize TTFT independently of TPOT.
3.  **Speculative Decoding:** 
    *   Test `--speculative-model` configs to see if combining prefix-caching with draft-model speculation yields compounding TPOT reductions.
4.  **Dynamic Retrieval-Fallback Orchestration:** 
    *   Build advanced router logic: If a prompt prefix cache *misses* across the cluster, automatically trigger a RAG retrieval fallback, then cache the generated result for the next hit. Compare this dynamic approach against static baselines.
