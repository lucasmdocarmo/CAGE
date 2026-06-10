> <!-- CAGE-DOC-STATUS -->
> **⚠️ STATUS: PARTIALLY STALE (2026-06-09).** Useful background, but some commands,
> CLI flags, paths, and metric numbers here predate the June 2026 fixes. In
> particular, `run_experiment.py` has **no** `--phase`/`--all-baselines`/`--trials`/`--queries`
> flags (use `--baseline`, `--num-trials`, `--num-queries`), and pre-fix metric numbers
> (faithfulness 0.570, BERTScore 0.324) are now obsolete. For runnable commands use
> [`RUNBOOK.md`](RUNBOOK.md); for current metrics see [`KNOWLEDGE_BASE.md`](KNOWLEDGE_BASE.md).

# CAGE Framework: GCP Deployment & Execution Runbook

> **Purpose:** This runbook provides the exact step-by-step commands to set up your GCP account quotas, deploy the CAGE distributed architecture, push the local code, execute Phase 2 & Phase 3 GPU tests, and validate the results.

---

## 1. GCP Account Setup & Quotas (One-Time Setup)

Before you can run Terraform, you must ensure your Google Cloud account is properly configured to allow GPU access and high-performance networking.

### A. Enable APIs
1. Go to **APIs & Services > Library** in the GCP Console.
2. Search for and enable the **Compute Engine API**.
3. Search for and enable the **Cloud Resource Manager API**.

### B. Request GPU Quotas
By default, new GCP projects have a GPU quota of 0.
1. Go to **IAM & Admin > Quotas & System Limits**.
2. Filter by your target region (e.g., `us-central1`).
3. Request the following increases:
   * **`GPUs (all regions)`:** Limit of at least 4.
   * **`NVIDIA L4 GPUs`:** Limit of at least 3 (Required for Phase 2 G2 Instances).
   * **`NVIDIA A100 GPUs`:** Limit of at least 3 (Required for Phase 3 A2 Instances).

---

## 2. Prerequisites (Local Machine)

Before deploying, authenticate your environment.

```bash
# 1. Authenticate with Google Cloud
gcloud auth login
gcloud config set project [YOUR_PROJECT_ID]
gcloud config set compute/region us-central1
gcloud config set compute/zone us-central1-a

# 2. Export your HuggingFace token for gated models
export HF_TOKEN="your_hf_token_here"
```

---

## 3. Infrastructure Deployment (Phase 2)

Spin up the distributed cluster (1x CPU Router Node + 3x GPU vLLM Replica Nodes).

### Option A: Using Local Terminal
```bash
cd terraform/gcp
terraform init
# Deploy the cluster (type 'yes' when prompted)
terraform apply -var="project_id=$(gcloud config get-value project)" -var="hf_token=$HF_TOKEN"
```

### Option B: Using GCP Web Portal (Cloud Shell)
If you do not want to use a local terminal:
1. Go to [console.cloud.google.com](https://console.cloud.google.com).
2. Click the **Activate Cloud Shell** icon (`>_`).
3. Zip your local CAGE folder (`zip -r cage.zip CAGE/`).
4. In Cloud Shell, click the three-dot menu (⋮) -> **Upload** and upload `cage.zip`.
5. Unzip it: `unzip cage.zip && cd CAGE/terraform/gcp`.
6. Run the exact same Terraform commands from Option A.

---

## 4. Code Upload & Initialization

The Terraform script provisions the VMs, installs Docker, and spins up Redis on the router. It then **pauses and waits** for you to upload the codebase.

### Option A: Using Local Terminal (gcloud scp)
```bash
gcloud compute scp --recurse . cage-router:/opt/cage --zone=us-central1-a
```

### Option B: Using GCP Web Portal (Browser SSH)
1. Go to **Compute Engine > VM Instances**.
2. Click the **SSH** button next to `cage-router` to open a browser terminal.
3. Click the **Upload File** button (top-right of SSH window) and upload `cage.zip`.
4. Unzip it:
   ```bash
   sudo apt install unzip
   sudo mkdir -p /opt/cage
   sudo unzip cage.zip -d /opt/cage
   ```

**Initialization Check:** Once uploaded, the router's background script will automatically install `requirements.txt` and start the Python orchestration router.

---

## 5. Phase 2 Execution (GPU Validation)

Connect to the router to begin rigorous statistical testing on `Qwen3-8B`.

```bash
# 1. Connect to the router (skip if already using browser SSH)
gcloud compute ssh cage-router --zone=us-central1-a
cd /opt/cage

# 2. Verify router and Redis are running
ps aux | grep "src.orchestration.router"
docker ps | grep cage-redis

# 3. Execute Phase 2 baselines
python3 scripts/run_experiment.py \
    --phase 2 \
    --all-baselines \
    --trials 10 \
    --queries 100 \
    --model Qwen/Qwen3-8B \
    --api-base http://localhost:9000

# 4. Validate Results & Statistics
python3 scripts/verify_results.py
python3 scripts/statistical_tests.py --results-dir analysis/phase2/results/
```

---

## 6. Phase 3 Configuration & Execution (HPC limits)

To push the architecture to its distributed memory limits, you must scale up the hardware and networking.

### A. Infrastructure Reconfiguration (Terraform)
1. Destroy the Phase 2 cluster: `terraform destroy`
2. Open `terraform/gcp/main.tf` and make the following changes:
   * **Machine Type:** Change `g2-standard-8` to `a2-highgpu-1g` (A100 GPUs).
   * **VPC Network:** Add `mtu = 8896` to the `google_compute_network` block to enable Jumbo Frames for massive tensor transfers.
   * **Network Interface:** Add `nic_type = "GVNIC"` to the replica network interfaces to unlock 100Gbps bandwidth.
3. Re-run `terraform apply`.

### B. Phase 3 Execution
Connect to the new router and execute the Phase 3 stress tests:

```bash
# 1. Test Disaggregated Prefilling & KV Transfer limits
python3 scripts/run_experiment.py \
    --phase 3 \
    --baseline distributed \
    --model Qwen/Qwen3-14B \
    --enable-disagg-prefill

# 2. Test Speculative Decoding
python3 scripts/run_experiment.py \
    --phase 3 \
    --baseline hybrid_cache_warm \
    --model Qwen/Qwen3-14B \
    --speculative-model Qwen/Qwen3-1B
```

---

## 7. Result Retrieval

Pull the data and generated publication plots back to your local machine.

```bash
# Local Terminal Command:
gcloud compute scp --recurse cage-router:/opt/cage/analysis/ ./analysis_cloud_backup/ --zone=us-central1-a
```
*(Or use the "Download File" button in the Web Portal SSH client).*

---

## 8. Teardown (CRITICAL TO SAVE COSTS)

Cloud GPUs are extremely expensive. **Destroy the cluster** when not actively collecting data.

```bash
cd terraform/gcp
terraform destroy -var="project_id=$(gcloud config get-value project)" -var="hf_token=$HF_TOKEN"
```
