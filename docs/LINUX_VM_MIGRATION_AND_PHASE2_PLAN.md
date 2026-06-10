> <!-- CAGE-DOC-STATUS -->
> **⚠️ STATUS: SUPERSEDED / HISTORICAL (2026-06-09).** This document predates the
> June 2026 reorganization and metric fixes. It is kept for history only and
> contains stale paths, pre-fix metric numbers, and/or invalid CLI flags.
> **Authoritative docs:** [`KNOWLEDGE_BASE.md`](KNOWLEDGE_BASE.md) (project reference),
> [`RUNBOOK.md`](RUNBOOK.md) (setup/deploy/run), [`VALIDATION_AND_SOTA_REVIEW.md`](VALIDATION_AND_SOTA_REVIEW.md) (limitations).
> See [`README.md`](README.md) for the doc index.

# CAGE: Linux VM Migration & Master Project Roadmap

**Last Updated:** 2026-04-11
**Context:** CAGE (Cache-Augmented Generation Evaluation) is a framework explicitly designed to benchmark cache-based LLM serving strategies against traditional retrieval based (RAG) approaches. Phase 1 metrics on CPU are locally complete. Due to fatal PyTorch / Python 3.12 syntax parsing bugs natively blocking execution on macOS, local integration test execution is fundamentally migrating to a Linux Parallels Virtual Machine (Ubuntu or Fedora).

---

## 1. The Linux Migration Strategy (Immediate Action)

Running the real `vLLM` binary native on macOS Apple Silicon is unsupportable without dangerous hacks. We are migrating local testing to a Linux (aarch64) Parallels VM. This grants a clean, Linux-native PyTorch execution environment to run the 27 required CAGE validation tests before pushing to a Cloud GPU for Phase 2.

### Local VM Execution Workflow
1. Provision the **Ubuntu or Fedora Parallels Virtual Machine**.
2. Mount the host `CAGE` repository into the VM as a Shared Folder.
3. Execute the corresponding deployment script (`setup_ubuntu.sh` or `setup_fedora.sh`) inside the VM to rigidly lock the environment:
   - `python=3.12`
   - `vllm=0.8.3` (stable, pre-`tokenizer` crash binary)
   - `transformers=4.46.1` (stable tokenizer configuration)
4. Ensure the `tests/test_vllm_integration.py` and `tests/test_router_integration.py` files are updated to strictly query dynamic strings from `VLLM_TEST_MODEL` to prevent `404: Not Found` failures when bypassing Qwen3-4B.

---

## 2. Master Project Status & Core Objectives

### Phase 1: Local Development & CPU Validation (✅ 95% Complete)
- **Status:** 7 baselines mathematically executed over SQuADv2 locally. 
- **Validation:** 1,050 queries executed across `no_cache`, `prefix_cache`, `rag`, `redis`, `hybrid`, `distributed`.
- **Outputs:** The 12-page draft (`main-12page.tex`) natively exists highlighting −65.7% TTFT for Prefix Cache versus baseline.
- **Current Blocker:** The 27-suite Python CI testing loop (`pytest`) failed to validate native framework robustness locally due to PyTorch.
- **Goal Completion Criteria:** The Linux VM setup parses 27/27 green integration tests against the live Python cluster APIs.

### Phase 2: GPU Scaling & Significance (Next Up)
Once the Linux VM proves the orchestration pipeline works under heavy test loops, the project scales out to true Cloud environments.
- **Objective:** Provision high-compute Nodes natively (`A100` recommended). Run identical workloads 10x-100x faster than the CPU limits measured in Phase 1.
- **Tasks:**
  - Execute `docker-compose.gpu.yml` with Nvidia runtime paths.
  - Scale up parameters: run with Qwen3-8B and Qwen3-14B formats to validate prefix caching's proportional effect vs model size.
  - Automate the new experimental metrics via `scripts/statistical_tests.py` to prove mathematically significant changes.
  - Synthesize and render final Phase 2 scaling curve plots.

### Phase 3: True Multi-Node Distributed Architecture (Future Run)
- **Objective:** Implement genuine cross-node KV Cache mechanics via RPC and benchmark true Disaggregated Prefilling instead of localized prefix-cache overlaps.
- **Tasks:**
  - Spin up exactly `terraform/gcp` templates for distributed node testing.
  - Implement real KV tensor network transfers using NCCL or explicit gRPC channels via the router mapping.
  - Compare localized vs network latency on high-throughput queues.

---

## 3. Immediate Action Plan

To proceed instantly:
1. Run `setup_ubuntu.sh` (or `setup_fedora.sh` depending on your OS choice) inside your newly-provisioned Virtual Machine.
2. Confirm test executions: `python -m pytest tests/`
3. If successful, Phase 1 is canonically complete, mathematically flawless, and perfectly tested. Proceed natively to Cloud instances for Phase 2 GPU configurations.
