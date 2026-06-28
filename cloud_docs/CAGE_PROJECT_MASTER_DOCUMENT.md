> <!-- CAGE-DOC-STATUS -->
> **⚠️ STATUS: SUPERSEDED / HISTORICAL (2026-06-09).** This document predates the
> June 2026 reorganization and metric fixes. It is kept for history only and
> contains stale paths, pre-fix metric numbers, and/or invalid CLI flags.
> **Authoritative docs:** [`KNOWLEDGE_BASE.md`](KNOWLEDGE_BASE.md) (project reference),
> [`RUNBOOK.md`](RUNBOOK.md) (setup/deploy/run), [`VALIDATION_AND_SOTA_REVIEW.md`](VALIDATION_AND_SOTA_REVIEW.md) (limitations).
> See [`README.md`](README.md) for the doc index.

# CAGE: Project Master Document & Deployment Blueprint

> **Purpose:** This document serves as the ultimate source of truth for the CAGE (Cache-Augmented Generation Environment) project. It contains the complete project history, architectural implementation details, vLLM technical documentation, past testing summaries, and the definitive step-by-step guide for deploying and executing the upcoming cloud phases.

---

## 1. Project History & Overall Objective

### The Problem (Why CAGE exists)
Modern Large Language Models (LLMs) suffer from hallucinations. The industry standard solution is **RAG (Retrieval-Augmented Generation)**, which fetches external documents to ground the LLM's answers. However, RAG introduces severe latency overhead (retrieval + reranking) and can still hallucinate if the retrieved context is imperfect. 

### The Solution & Final Objective
**CAGE** aims to prove that a distributed **Contextual Prefix Cache** is superior to RAG for static or semi-static knowledge bases. By pre-computing and caching the key-value (KV) tensors of massive documents across a multi-node GPU cluster, CAGE eliminates retrieval latency entirely. Our final academic objective is to statistically prove that CAGE not only achieves significantly lower Time-To-First-Token (TTFT) but also maintains strictly higher factual grounding (Faithfulness) than traditional retrieval pipelines.

---

## 2. Implementation Details

CAGE is a distributed microservice architecture designed to scale LLM inference:
*   **The Orchestration Router:** A Python-based routing layer that intercepts incoming prompts.
*   **Distributed Cache State (Redis):** The router queries a Redis cache to determine which GPU node currently holds the KV-cache tensors for the requested document.
*   **The Inference Engine (vLLM):** The router forwards the request to the specific GPU node (running vLLM) that already has the document cached in its VRAM, achieving a massive speedup by skipping the "prefill" computation phase.

---

## 3. vLLM Documentation & Concepts

To understand CAGE, you must understand how we leverage **vLLM**, a high-throughput memory-efficient LLM serving engine.

*   **Prefix Caching (`--enable-prefix-caching`):** When vLLM reads a prompt, it converts the text into Key-Value (KV) tensors. Prefix caching saves these tensors in GPU VRAM. If a new prompt shares the same prefix (e.g., the same long system prompt or reference document), vLLM simply reuses the cached tensors. This drops TTFT from seconds to milliseconds.
*   **Disaggregated Prefilling (`--enable-disagg-prefill`):** LLM generation has two phases: *Prefill* (reading the prompt, highly compute-bound) and *Decode* (generating the answer, highly memory-bandwidth-bound). Disaggregated prefilling separates these tasks onto entirely different GPU nodes, optimizing both simultaneously.
*   **Speculative Decoding (`--speculative-model`):** A technique where a tiny "draft" model (e.g., Qwen-1B) guesses the next 5 tokens, and the massive "target" model (e.g., Qwen-14B) verifies them all in a single step. This dramatically reduces Time-Per-Output-Token (TPOT).

---

## 4. Past Tests & Conducted Work (Phase 1)

**What we have already done:**
1.  **Local CPU Validation:** We successfully built the framework and validated the orchestration logic locally. We overcame significant architectural challenges, migrating the testing suite from ARM64/Apple Silicon to a native Ubuntu Linux environment to compile vLLM from source.
2.  **Metric Refinement:** We analyzed preliminary data on a small model (Qwen3-4B). We discovered that soft-matching metrics like **BERTScore** were entirely blind to subtle factual hallucinations. We successfully deprecated BERTScore in favor of strict **NLI-based Faithfulness**, establishing a ceiling score of 0.570 on SQuADv2, and proving that RAG retrieves degrade factual grounding by 11.6% compared to cached gold-context.

---

## 5. Cloud Setup & Next Phases Configuration

### Step 1: GCP Account Configuration
Before deploying, you must prepare your Google Cloud environment:
1.  Enable **Compute Engine API** and **Cloud Resource Manager API**.
2.  Navigate to **IAM & Quotas** and request limit increases for **NVIDIA L4** (need 3+) and **NVIDIA A100** (need 3+) GPUs in your region (`us-central1`).

### Step 2: Phase 2 Execution (GPU Validation)
**Goal:** Run strict statistical tests on Qwen3-8B under real GPU VRAM constraints.
1.  **Deploy:** `cd terraform/gcp && terraform apply` (This provisions 3x G2/L4 GPU instances and 1x CPU router).
2.  **Upload Code:** Compress your local repo to `cage.zip`. SSH into the `cage-router` instance via the GCP Web Portal, click "Upload File", and unzip the code into `/opt/cage`.
3.  **Execute Tests:**
    ```bash
    cd /opt/cage
    # Run the rigorous 10-trial baseline
    python3 scripts/run_experiment.py --phase 2 --all-baselines --trials 10 --queries 100 --model Qwen/Qwen3-8B
    ```

### Step 3: Phase 3 Execution (HPC & Distributed Stress Testing)
**Goal:** Push the architecture to its limits using massive 14B models and extreme network optimization.
1.  **Reconfigure Infrastructure:** Edit `terraform/gcp/main.tf`:
    *   Change machines to A100s (`a2-highgpu-1g`).
    *   Enable Jumbo Frames (`mtu = 8896`) for massive tensor transfers.
    *   Enable Google Virtual NIC (`nic_type = "GVNIC"`) to unlock 100 Gbps cross-node bandwidth.
2.  **Deploy & Execute Tests:**
    ```bash
    terraform apply
    
    # Test Disaggregated Prefill limits
    python3 scripts/run_experiment.py --phase 3 --baseline distributed --model Qwen/Qwen3-14B --enable-disagg-prefill
    ```

---

## 6. Testing Scripts Overview

Here is a summary of the scripts you will use to run and validate the next phases:
*   **`scripts/download_datasets.py`:** Downloads SQuADv2 and builds the FAISS index required for the RAG baseline comparisons. Needs to be expanded in Phase 2 to support *TriviaQA* or *Natural Questions*.
*   **`scripts/run_experiment.py`:** The master orchestrator. It fires queries at the CAGE router, tracks latency, checks cache hit rates, and computes the strict NLI-Faithfulness score of the generated answers.
*   **`scripts/verify_results.py`:** A data integrity script run after experiments to ensure no HTTP timeouts or corrupted JSON outputs occurred during the 10-trial run.
*   **`scripts/statistical_tests.py`:** Calculates the Wilcoxon rank-sum p-values to prove that the latency and quality differences between CAGE and RAG are statistically significant, generating the final tables for the academic paper.
