> <!-- CAGE-DOC-STATUS -->
> **⚠️ STATUS: PARTIALLY STALE (2026-06-09).** Useful background, but some commands,
> CLI flags, paths, and metric numbers here predate the June 2026 fixes. In
> particular, `run_experiment.py` has **no** `--phase`/`--all-baselines`/`--trials`/`--queries`
> flags (use `--baseline`, `--num-trials`, `--num-queries`), and pre-fix metric numbers
> (faithfulness 0.570, BERTScore 0.324) are now obsolete. For runnable commands use
> [`RUNBOOK.md`](RUNBOOK.md); for current metrics see [`KNOWLEDGE_BASE.md`](KNOWLEDGE_BASE.md).

# CAGE Framework: Phase Execution & Configuration Guide

> **Purpose:** This guide separates the project into explicit phases (Phase 2 and Phase 3). For each phase, it details the exact configurations, step-by-step execution, validation methods, estimated runtime, and projected Google Cloud costs. It also explains the effect of tuning specific configuration levers.

---

## Phase 2: GPU Validation & Baseline Scaling

**Objective:** Transition the codebase from local CPU validation to cloud GPUs to induce authentic KV-cache memory pressure. Scale the model to 8B parameters to analyze realistic latency overheads.

### 1. Hardware & Model Configurations
*   **Instances:** `g2-standard-8` (1x NVIDIA L4 GPU per node, 24GB VRAM).
*   **Model:** `Qwen/Qwen3-8B`
*   **vLLM Config:** `--gpu-memory-utilization 0.9` (Reserves 90% of the 24GB VRAM for the model weights and the KV cache).

### 2. Execution Steps (From Local to Cloud)
1.  **Deploy Infrastructure:** Run `terraform apply` in your `terraform/gcp` directory (defaults are set for Phase 2).
2.  **Transfer Code:** 
    *   *CLI:* `gcloud compute scp --recurse . cage-router:/opt/cage`
    *   *Web:* Upload `cage.zip` via the browser SSH window and unzip it.
3.  **Run Tests:** SSH into `cage-router` and execute:
    ```bash
    cd /opt/cage
    python3 scripts/run_experiment.py --phase 2 --all-baselines --trials 10 --queries 100 --model Qwen/Qwen3-8B
    ```
4.  **Validate:** 
    ```bash
    python3 scripts/verify_results.py
    python3 scripts/statistical_tests.py --results-dir analysis/phase2/results/
    ```

### 3. Projected Costs & Runtime
*   **Cluster Cost:** ~\$4.50 per hour (3x G2 nodes @ \$1.20/hr + 1x CPU router).
*   **Running Time:** 10 trials of 100 queries against an 8B model will take approximately **3 hours**.
*   **Total Phase 2 Cost:** ~\$13.50.

---

## Phase 3: Distributed HPC & Network Stress Testing

**Objective:** Evaluate the architecture under distributed high-performance computing constraints. This requires massive models (14B), massive GPUs (A100), and optimized networking (gVNIC) to handle cross-node KV tensor transfers.

### 1. Hardware & Model Configurations
*   **Instances:** `a2-highgpu-1g` (1x NVIDIA A100 GPU per node, 40GB/80GB VRAM).
*   **Network Config:** Jumbo Frames (`MTU=8896`) and Google Virtual NIC (`nic_type="GVNIC"`).
*   **Model:** `Qwen/Qwen3-14B`
*   **Advanced Features:** Disaggregated Prefilling & Speculative Decoding.

### 2. Execution Steps (From Phase 2 to Phase 3)
1.  **Teardown Phase 2:** `terraform destroy`
2.  **Reconfigure Terraform (`main.tf`):**
    *   Change `machine_type` to `"a2-highgpu-1g"`.
    *   Add `mtu = 8896` to the `google_compute_network`.
    *   Add `nic_type = "GVNIC"` to the `network_interface` block.
3.  **Deploy & Transfer:** `terraform apply` -> Upload code to router.
4.  **Run HPC Tests:** SSH into `cage-router` and execute:
    ```bash
    # Test 1: Disaggregated Prefilling
    python3 scripts/run_experiment.py --phase 3 --baseline distributed --model Qwen/Qwen3-14B --enable-disagg-prefill
    
    # Test 2: Speculative Decoding
    python3 scripts/run_experiment.py --phase 3 --baseline hybrid_cache_warm --model Qwen/Qwen3-14B --speculative-model Qwen/Qwen3-1B
    ```
5.  **Validate:** Run `verify_results.py` and inspect network metrics (if tensor transfer times exceed 100ms, the gVNIC configuration failed).

### 3. Projected Costs & Runtime
*   **Cluster Cost:** ~\$11.50 per hour (3x A2 nodes @ \$3.67/hr + 1x CPU router).
*   **Running Time:** Massive 14B models with complex KV routing will take approximately **4 to 5 hours** for a full 10-trial suite.
*   **Total Phase 3 Cost:** ~\$50.00 to \$60.00.

---

## Configuration Options & Their Effects

If you need to tweak the system to fix crashes or optimize speed, here is exactly what each lever does:

### 1. `VLLM_GPU_MEMORY_UTILIZATION` (Default: 0.9)
*   **What it does:** Dictates how much VRAM vLLM claims at startup.
*   **Effect of lowering (e.g., 0.7):** Reduces the chance of Out-Of-Memory (OOM) crashes, but drastically shrinks the available KV Cache space. Your cache will evict items much faster, causing your CAGE baselines to degrade to RAG performance.
*   **Effect of raising (e.g., 0.95):** Maximizes KV cache space, but risks crashing the GPU if PyTorch needs memory spikes during inference.

### 2. `MTU = 8896` (Jumbo Frames)
*   **What it does:** Increases the maximum packet size on the Google Cloud VPC.
*   **Effect:** Standard MTU (1460) forces massive KV tensor byte-arrays to be chopped into millions of tiny packets, causing severe CPU overhead and network latency. Jumbo frames allow larger chunks, significantly reducing the cross-node latency in Phase 3.

### 3. `nic_type = "GVNIC"`
*   **What it does:** Bypasses standard virtual networking to give the VM direct access to Google's physical network hardware.
*   **Effect:** Boosts bandwidth from a standard ~15 Gbps up to **100 Gbps**. If you do not enable this in Phase 3, transferring KV cache across nodes will take longer than simply recomputing the prompt from scratch, completely defeating the purpose of the architecture.

### 4. `--speculative-model` (e.g., Qwen3-1B)
*   **What it does:** Loads a tiny "draft" model alongside the main model. The draft model rapidly guesses the next 5 tokens, and the main 14B model verifies them all at once in a single forward pass.
*   **Effect:** Trades a small amount of GPU VRAM (to hold the 1B model) for massive reductions in TPOT (Time Per Output Token). This is highly recommended for Phase 3.
