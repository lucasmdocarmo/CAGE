# CAGE — GCP Console Guide (click‑by‑click, ELI5)

> A literal, page‑by‑page walkthrough for running CAGE on Google Cloud **from the browser**.
> Written so you can follow it **by hand** or have **Claude‑in‑Chrome/Brave drive it**.
> Covers the full lifecycle per phase: **Provision → Set up & Run → Check → Validate → Gather → Tear down.**
>
> This is the gentle console version of [`RUNBOOK.md`](RUNBOOK.md) (which is the CLI reference).
> When in doubt about a command, RUNBOOK is the source of truth.
>
> **Two ways to do every step:**
> - **🖱️ Console clicks** — pure point‑and‑click in the GCP web UI.
> - **⌨️ Cloud Shell** — open the built‑in terminal and paste one command. *More reliable, and the
>   easiest path for an AI browser agent.* If a click step feels fiddly, use the Cloud Shell line instead.

---

## 0. Before you start (read once)

| Phase | What it is | Machine | Rough cost | This guide |
|---|---|---|---|---|
| **Phase 2** | Validate baselines on a single GPU | 1× `g2-standard-8` + NVIDIA **L4** | ~$1.20/hr → ~$4 for a sweep | **§2 — fully click‑by‑click** |
| **Phase 3** | Distributed HPC (A100, real KV transfer) | 3× A100 + router | ~$11/hr | **§3 — Cloud Shell + Terraform** |

You need: a Google account, a credit card on the GCP billing account, and ~30 min the first time.
**GPUs cost money the entire time the VM exists** — §2.7 (delete it) is not optional.

> 🤖 **For Claude‑in‑Chrome:** GCP console pages are trusted; you may click within them. But **you (the human) must do the Google login and any billing/payment confirmation** — the agent must not enter payment details or approve spend on its own. Creating a GPU VM and clicking **Create** *does* start billing: treat that one click as a spend action and confirm with the human first. When a step is ambiguous in the UI, prefer the **⌨️ Cloud Shell** alternative (paste a command) — it's deterministic.

---

## 1. Phase 0 — One‑time account setup

Do this once per project; you won't repeat it for later runs.

### 1.1 Pick (or create) a project
1. Open **https://console.cloud.google.com** and sign in.
2. At the **very top**, click the **project dropdown** (it shows the current project name, left of the search bar).
3. In the dialog, click **NEW PROJECT** (top‑right) → give it a **Project name** like `cage` → **CREATE**.
4. Wait ~20s, then click the project dropdown again and **select your new project**.
   - ⌨️ Cloud Shell equivalent: `gcloud config set project YOUR_PROJECT_ID`

### 1.2 Make sure billing is on
1. Hamburger menu **☰** (top‑left) → **Billing**.
2. If it says "This project has no billing account," click **LINK A BILLING ACCOUNT** → pick your account → **SET ACCOUNT**.

### 1.3 Turn on the APIs CAGE needs
For each of these three, do: **☰ → APIs & Services → Library**, type the name in the search box, click the result, then click the blue **ENABLE** button.
1. `Compute Engine API`
2. `Cloud Storage API`
3. `Cloud Resource Manager API`
   - ⌨️ Cloud Shell equivalent (does all three): `gcloud services enable compute.googleapis.com storage.googleapis.com cloudresourcemanager.googleapis.com`

### 1.4 Ask for GPU quota (new projects start at 0)
1. **☰ → IAM & Admin → Quotas & System Limits**.
2. In the **Filter** box type `NVIDIA L4 GPUs`.
3. Tick the checkbox on the row for your region (e.g. **us-central1**).
4. Click **EDIT QUOTAS** (top) → set **New limit** to `1` (Phase 2) → fill the short justification ("research benchmarking") → **SUBMIT REQUEST**.
5. Approval is usually minutes to a day. You'll get an email. **You cannot create the GPU VM until this is approved.**
   - ⌨️ Check availability in your zone: `gcloud compute accelerator-types list --filter="zone:us-central1-a AND name:nvidia-l4"`

---

## 2. Phase 2 — Single GPU VM (the main path), click‑by‑click

### 2.1 Provision — create the GPU VM

**🖱️ Console clicks:**
1. **☰ → Compute Engine → VM instances**. (First time: click **ENABLE** if prompted, wait ~1 min.)
2. Click **CREATE INSTANCE** (top).
3. **Name:** type `cage-gpu`.
4. **Region:** `us-central1`. **Zone:** `us-central1-a`.
5. **Machine configuration:** click the **GPUs** tab (or "GPU" preset).
   - **GPU type:** `NVIDIA L4`. **Number of GPUs:** `1`.
   - **Machine type:** choose `g2-standard-8` (8 vCPU, 32 GB).
6. A yellow note may appear: **"Install NVIDIA GPU driver automatically"** — if there's a checkbox for it, **tick it**. (If not, we install it in §2.3.)
7. **Boot disk:** click **CHANGE** →
   - **Operating system:** `Deep Learning on Linux` (a.k.a. Deep Learning VM). **Version:** any recent CUDA 12.x image.
   - **Size:** set to `200` GB. **Type:** `SSD persistent disk`. Click **SELECT**.
   - *(If you don't see Deep Learning images, pick `Debian 11` and we'll install the driver in §2.3.)*
8. **Firewall:** you don't need HTTP/HTTPS for Phase 2 (the dashboard is in the terminal). Leave unchecked.
9. Scroll down, click **CREATE**. ⏳ The VM boots in ~1–2 min. **(This click starts billing.)**

**⌨️ Cloud Shell alternative (one command, most reliable):** click the **`>_` (Activate Cloud Shell)** icon top‑right, then paste:
Run this **from the CAGE repo root** so `--metadata-from-file` can read the shutdown hook:
```bash
PROJECT=$(gcloud config get-value project)
gcloud compute instances create cage-gpu --zone=us-central1-a \
  --machine-type=g2-standard-8 --accelerator=type=nvidia-l4,count=1 \
  --maintenance-policy=TERMINATE --image-family=common-cu121-debian-11 \
  --image-project=deeplearning-platform-release --boot-disk-size=200GB \
  --boot-disk-type=pd-ssd --scopes=cloud-platform --metadata=install-nvidia-driver=True \
  --metadata-from-file=shutdown-script=scripts/5_observability/gcp_shutdown_hook.sh
```
> The `shutdown-script` mirrors `analysis/` + logs to GCS on ACPI soft-off (a SPOT preemption
> or `instances delete`/`stop`), so data survives even when no operator is watching. It is what
> makes the "~30s notice" tolerance below real. Already created the VM? Add it live with
> `gcloud compute instances add-metadata cage-gpu --zone=us-central1-a --metadata-from-file=shutdown-script=scripts/5_observability/gcp_shutdown_hook.sh`.

**💸 Cheaper: spot VM (~65% off — recommended for Phase 2).** Add two flags to the command above:
```bash
  --provisioning-model=SPOT --instance-termination-action=STOP
```
This rents GCP spare capacity at **~$0.30/hr** instead of ~$0.85/hr. The trade: GCP can reclaim
the VM at any time with ~30s notice. Phase 2 **absorbs that safely** — baselines run one at a time
and `cloud_run.sh` syncs each finished baseline to your bucket every 2 min, so a preemption costs
you at most the single in‑flight baseline (and even a full redo of a spot run is only ~$2–3).
> ⚠️ **Spot is for Phase 2 only — never the Phase 3 cluster (§3).** Phase 3 is 4 coordinated nodes
> that must stay up *together*: losing any one to preemption corrupts the whole distributed run, the
> per‑node preemption odds stack over a multi‑hour window, and 3× A100 spot capacity often isn't even
> available. Keep Phase 3 on‑demand (`STANDARD`).

### 2.2 Provision — create the results bucket (so results survive)
1. **☰ → Cloud Storage → Buckets** → **CREATE**.
2. **Name:** `YOUR_PROJECT_ID-cage-results` (must be globally unique; the project‑id prefix helps).
3. **Location type:** Region → `us-central1`. Click **CONTINUE** through the defaults → **CREATE**.
4. On the bucket's page, **PROTECTION** (or **Configuration**) tab → enable **Object Versioning** (keeps every result).
   - ⌨️ Cloud Shell: `gsutil mb -l us-central1 gs://$(gcloud config get-value project)-cage-results && gsutil versioning set on gs://$(gcloud config get-value project)-cage-results`
   - ⌨️ **Grant the VM's service account write access** (REQUIRED — on projects created after ~mid-2024 the default compute SA has no Editor role, so every `sync_results_to_gcs.sh` / shutdown-hook sync silently no-ops and you lose all results at teardown): `gsutil iam ch "serviceAccount:$(gcloud iam service-accounts list --filter='displayName:Compute Engine default' --format='value(email)')":roles/storage.objectAdmin gs://$(gcloud config get-value project)-cage-results`

### 2.3 Set up — open a terminal on the VM and install
1. Back to **☰ → Compute Engine → VM instances**.
2. On the `cage-gpu` row, click the **SSH** button (opens a black browser terminal in a new window). Give it ~10s.
3. In that terminal, paste these **one block at a time** (wait for each to finish):
```bash
nvidia-smi          # should print a GPU table. If "command not found", wait 1 min and retry (driver still installing).
```
```bash
sudo apt-get update -y && sudo apt-get install -y git python3-venv
git clone https://github.com/lucasmdocarmo/CAGE.git CAGE && cd CAGE
# cage-env (NOT .venv): the run scripts hard-source cage-env on the VM under set -e.
python3 -m venv cage-env && source cage-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt        # pulls cage-stats too (telemetry)
pip install vllm
```
```bash
# (optional) gated models:
export HF_TOKEN=hf_xxxxxxxx
# rebuild RAG indices once so RAG/hybrid quality is correct:
rm -rf experiments/ir_index/ir_squad_v2_*
```

### 2.4 Run — start the suite (auto‑telemetry, auto‑GCS‑sync)
Still in the same SSH terminal, inside `cage`:
```bash
nohup bash scripts/3_run/cloud_run.sh Qwen/Qwen3-8B 500 3 > run.log 2>&1 &
tail -f run.log
```
- `nohup … &` keeps it running even if the SSH window closes.
- `cloud_run.sh` runs the 6 baselines, captures **cage-stats telemetry** per baseline, and **mirrors results to your bucket every 2 min**.
- `tail -f run.log` shows live progress. Press **Ctrl‑C** to stop *watching* (the run keeps going). To re‑attach later: `tail -f ~/CAGE/run.log`.

### 2.5 Check — is it healthy and progressing?
- **In the SSH terminal:** the `run.log` should show baselines starting/finishing and a `vLLM TELEMETRY (cage-stats)` dashboard block.
- **In the console:** **☰ → Cloud Storage → Buckets → your bucket** → open `results/phase2/<run-id>/baselines/` — you should see baseline folders appearing and growing over time. That's the proof results are being saved.
- **GPU busy?** In SSH: `nvidia-smi` (run it again) → utilisation should be >0% while a baseline runs.
- **Something stuck?** `tail -n 50 run.log` for the latest lines.

### 2.6 Validate — check + statistics
When `run.log` prints "Phase 1 Complete" (or all 6 baseline folders exist), in the SSH terminal:
```bash
cd ~/CAGE && source cage-env/bin/activate
python3 scripts/4_analysis/verify_results.py                       # integrity: no truncated/errored trials
python3 scripts/4_analysis/statistical_tests.py --results-dir results/phase2/<run-id>/baselines --reference no_cache
```
- `verify_results.py` should report all baselines OK.
- `statistical_tests.py` prints a per‑metric table (Wilcoxon, Holm‑corrected). "sig=yes" means a real difference.
- Telemetry sanity: `cat results/phase2/<run-id>/baselines/prefix_cache/vllm_telemetry.json | head` — you should see `spec_acceptance`, a `kv` block, etc.

### 2.7 Gather — get the results off the cloud
Everything is already in your bucket. To download:
- **🖱️ Console:** **☰ → Cloud Storage → your bucket → `analysis/`** → tick the folder → **DOWNLOAD**. (For many files, the Cloud Shell line below is easier.)
- **⌨️ Cloud Shell / your laptop:** `gsutil -m cp -r gs://$(gcloud config get-value project)-cage-results/analysis ./analysis`
- The artifacts you care about: `aggregated_metrics.json` (per baseline), `vllm_telemetry.json` (per baseline), `stats.json` (if you saved it), and `results/phase2/<run-id>/plots/` (plots, if generated).

### 2.8 Tear down — STOP PAYING (do this!)
1. **☰ → Compute Engine → VM instances**.
2. Tick the box on the `cage-gpu` row → click **DELETE** (top, trash icon) → confirm **DELETE**.
   - ⌨️ Cloud Shell: `gcloud compute instances delete cage-gpu --zone=us-central1-a --quiet`
3. **The bucket stays** (versioned, force‑destroy off) — your results are safe. The GPU bill stops the moment the VM is deleted.

> ⚠️ A *stopped* VM still bills for its disk; a *deleted* VM with the bucket kept is the safe end state. Delete the VM, keep the bucket.

---

## 3. Phase 3 — Distributed cluster (A100), via Cloud Shell + Terraform

A 3‑GPU + router cluster is too many moving parts to build click‑by‑click safely, so Phase 3 uses
the Terraform in the repo, driven from **Cloud Shell** (still all in the browser).

### 3.1 Provision
1. Ensure **A100 quota** (`NVIDIA A100 GPUs` ≥ 3) is approved — same as §1.4 but for A100.
2. Open **Cloud Shell** (`>_` top‑right). Then:
```bash
git clone https://github.com/lucasmdocarmo/CAGE.git && cd CAGE/terraform/gcp
cp terraform.tfvars.example terraform.tfvars
# edit terraform.tfvars (Cloud Shell has a built-in editor: run `cloudshell edit terraform.tfvars`):
#   project_id, model_name="Qwen/Qwen3-14B", repo_url="https://github.com/lucasmdocarmo/CAGE.git",
#   gpu_type="nvidia-tesla-a100", machine_type="a2-highgpu-1g", nic_type="GVNIC", network_mtu=8896
terraform init
terraform apply -var="project_id=$(gcloud config get-value project)"
terraform output            # note router_external_ip and results_bucket
```

### 3.2 Check
```bash
ROUTER_IP=$(terraform output -raw router_external_ip)
curl -fsS http://$ROUTER_IP:9000/health && curl -fsS http://$ROUTER_IP:9000/stats   # expect 3 replicas
```
(If unhealthy, **☰ → Compute Engine → VM instances → vllm-replica-1 → Logs / Serial port** to read boot logs.)

### 3.3 Run / Validate / Gather
SSH into the router (console SSH button on `cage-router`, or `gcloud compute ssh cage-router`):
```bash
cd /opt/cage
# distributed baseline against the cluster, with telemetry:
CAGE_REQUIRE_DISTINCT_REPLICAS=1 python3 scripts/3_run/run_experiment.py \
  --baseline distributed --baseline-label distributed_router_replicated \
  --model Qwen/Qwen3-14B --dataset squad_v2 --num-queries 100 --num-trials 10 \
  --api-base http://localhost:9000 --sharding-policy replicated --vllm-telemetry \
  --output-dir results/phase2/<run-id>/baselines/distributed_router_replicated
bash scripts/5_observability/sync_results_to_gcs.sh results      # push to bucket
python3 scripts/4_analysis/statistical_tests.py --results-dir results/phase2/<run-id>/baselines
```

### 3.4 Tear down
```bash
cd ~/CAGE/terraform/gcp
terraform destroy -var="project_id=$(gcloud config get-value project)"   # removes GPUs/router, keeps the bucket
```

---

## 4. Driving this with Claude‑in‑Chrome (Brave)

The extension can navigate the GCP console and click buttons. To make a run go smoothly:

1. **You log in to Google first** (the agent should not handle credentials or 2FA).
2. Point the agent at this file and say e.g. *"Follow CLOUD_CONSOLE_GUIDE.md §2 to provision the Phase‑2 VM, then stop before the run so I can confirm."*
3. **Confirm spend yourself:** the click that creates the GPU VM (§2.1 **Create**) and `terraform apply` start real billing — have the agent pause for your OK there.
4. **Prefer Cloud Shell steps** (the ⌨️ lines) when handing work to the agent — pasting a known command into Cloud Shell is far more reliable than hunting for the right button across console redesigns.
5. **Never let the agent enter payment info or change billing/quota approvals** — those are human‑only.
6. After the run, ask it to do **§2.8 tear‑down** so nothing is left billing.

---

## 5. Quick troubleshooting

| Symptom | Fix |
|---|---|
| `nvidia-smi: command not found` | Driver still installing — wait 1–2 min, retry. Or `sudo /opt/deeplearning/install-driver.sh`. |
| `pip install` fails on `cage-stats` | The cage-stats repo must be reachable; it's public, so check the VM has internet. To skip telemetry: comment the `cage-stats` line in `requirements.txt`. |
| RAG/hybrid scores look low | You skipped the index rebuild — `rm -rf experiments/ir_index/ir_squad_v2_*` and re‑run (§2.3). |
| Bucket name "already taken" | Names are global — prefix with your project id (§2.2). |
| Run died when SSH closed | Use `nohup … &` (§2.4); re‑attach with `tail -f ~/CAGE/run.log`. |
| Quota error on Create | GPU quota not approved yet (§1.4) — wait for the email. |
| Still being billed | Delete the **VM** (§2.8); a stopped VM still bills disk. The bucket is cheap to keep. |
