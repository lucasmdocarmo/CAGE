# CAGE — Setup, Deploy & Run Runbook (authoritative)

> The single source of truth for **how to set up, run, deploy, and analyze** CAGE, locally
> and on GCP — including **durable result persistence to Google Cloud Storage**.
> Supersedes the command sections of `GCP_DEPLOYMENT_RUNBOOK.md`, `PHASE_EXECUTION_GUIDE.md`,
> `CAGE_PROJECT_MASTER_DOCUMENT.md`, and the older continuation guides (all of which contain
> invalid CLI flags). Verified against the code 2026-06-09.
>
> **Repo root:** `/Users/lucasmariano/CAGE/` (single level; run everything from here).
>
> ⚠️ **CLI reality check.** `run_experiment.py` runs **one baseline per call**. There is
> **no** `--phase`, `--all-baselines`, `--trials`, `--queries`, or `--enable-disagg-prefill`
> flag. The real flags are `--baseline`, `--num-trials`, `--num-queries` (full list in §6).
> Run the whole 7-baseline suite with the phase scripts (§4), which also handle the required
> vLLM prefix-cache on/off server toggling.

---

## 0. Concepts (30 seconds)
- `scripts/run_experiment.py` is the workload driver. It hits an OpenAI-compatible endpoint
  (`--api-base`) — a single vLLM server or the CAGE router — and writes results. **Where you
  run it is where results land.**
- 7 baselines (`--baseline`): `no_cache`, `prefix_cache`, `rag`, `redis`, `hybrid`,
  `distributed`, `speculative`.
- Output per baseline: `<output-dir>/aggregated_metrics.json` + `trial_N/results.csv` +
  `trial_N/metrics.json`. Schema is identical local and cloud.
- On the cloud, `scripts/cloud_run.sh` mirrors results to a durable GCS bucket continuously,
  so teardown/SSH drops never lose completed work (§9).

---

## 1. Environment setup

```bash
cd /Users/lucasmariano/CAGE

# 1a. Python env
python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt          # includes lettucedetect (primary grounding detector)

# 1b. vLLM
#  - Linux + NVIDIA GPU (cloud): official wheel
pip install vllm
#  - macOS / ARM CPU (local Phase-1 repro): source build (slow) — legacy helper:
#      bash scripts/setup/setup_fresh.sh
```

**LettuceDetect note.** It pulls a recent `transformers`. If that conflicts with the pinned
stack on a constrained box, install it in its own env or disable it for a run with
`export CAGE_DISABLE_LETTUCEDETECT=1` (faithfulness falls back to NLI-only; `grounding_score`
becomes null). First run downloads `KRLabsOrg/lettucedect-base-modernbert-en-v1`.

---

## 2. Data & retrieval indices

```bash
# Download datasets (SQuAD v2 by default; extend in scripts/download_datasets.py)
python3 scripts/download_datasets.py
```
**Rebuild the FAISS indices once.** The shipped indices under `experiments/ir_index/` predate
the e5 `query:`/`passage:` prefix fix and default to un-prefixed retrieval. Add
`--rebuild-ir-index` on the first RAG/redis/hybrid run (or delete
`experiments/ir_index/ir_squad_v2_*` and let it rebuild).

---

## 3. Start the inference server (local)

CAGE talks to vLLM over HTTP. Two modes:

```bash
# Single server (most baselines). Prefix caching ON:
./scripts/manage_vllm_server.sh restart Qwen/Qwen3-4B
# Prefix caching OFF (for no_cache / rag / redis-cold baselines):
./scripts/manage_vllm_server.sh restart Qwen/Qwen3-4B --no-prefix-cache
./scripts/manage_vllm_server.sh stop

# Multi-replica + router (distributed baseline):
python3 scripts/manage_vllm_cluster.py restart --model Qwen/Qwen3-4B \
  --replicas 3 --base-port 8001 --router-port 9000
python3 scripts/manage_vllm_cluster.py stop
```
Servers start with `--enable-prefix-caching --enable-prompt-tokens-details` (required for
cache telemetry). Redis (retrieval-cache baselines): `docker run -d -p 6379:6379 --name cage-redis redis:7-alpine`.

---

## 4. Run experiments — the easy path (phase scripts)

The phase scripts encode the **correct** server toggling, baseline labels, warmup, Redis
namespacing, and the distributed cluster bring-up.

```bash
# Full Phase-1 7-baseline suite on Qwen3-4B / SQuAD v2 (CPU-friendly defaults: 50 q, 3 trials)
bash scripts/run_phase1.sh

# Override scale via env vars (and model as $1):
NUM_QUERIES=100 NUM_TRIALS=10 bash scripts/run_phase1.sh Qwen/Qwen3-8B

# Results land in analysis/phase1/results/<baseline_label>/
```
Baseline labels produced: `no_cache`, `prefix_cache`, `rag`, `redis_retrieval_cache_cold`,
`hybrid_retrieval_cache_cold`, `hybrid_retrieval_cache_warm`, `distributed_router_replicated`.
Other runners: `scripts/run_phase2.sh` … `run_phase5.sh`, `scripts/run_all_phases.sh`.

---

## 5. Run experiments — manual single baseline (ad-hoc)

```bash
# Make sure the matching server is up (§3), then:
python3 scripts/run_experiment.py \
  --baseline prefix_cache \           # one of: no_cache prefix_cache redis rag distributed hybrid speculative
  --baseline-label prefix_cache \
  --model Qwen/Qwen3-4B \
  --dataset squad_v2 \                # one of: squad_v2 hotpotqa trivia_qa qasper humaneval mbpp hpc_code
  --num-queries 50 --num-trials 3 --seed 42 \
  --api-base http://localhost:8000 \
  --output-dir analysis/phase1/results/prefix_cache --rebuild-ir-index

# Distributed baseline goes through the router:
CAGE_REQUIRE_DISTINCT_REPLICAS=1 python3 scripts/run_experiment.py \
  --baseline distributed --baseline-label distributed_router_replicated \
  --model Qwen/Qwen3-4B --dataset squad_v2 --num-queries 50 --num-trials 3 \
  --api-base http://localhost:9000 --sharding-policy replicated \
  --output-dir analysis/phase1/results/distributed_router_replicated
```

---

## 6. `run_experiment.py` — real CLI reference

**Core:** `--baseline` (req), `--model` (req), `--dataset`, `--num-queries`, `--num-trials`,
`--seed`, `--max-tokens`, `--baseline-label`, `--output-dir`, `--api-base`
(default `http://localhost:8000`), `--backend {vllm,gemini,ollama}`, `--offline`.

**Retrieval/IR:** `--top-k`, `--top-k-values`, `--top-k-sweep`, `--embedding-model`,
`--reranker-model`, `--reranker-device`, `--ir-index-dir`, `--rebuild-ir-index`,
`--max-context-docs`, `--max-context-chars`, `--truncate-prompt-tokens`.

**Redis:** `--redis-host`, `--redis-port`, `--redis-db`, `--redis-ttl-seconds`,
`--redis-key-prefix`, `--flush-redis-namespace`.

**Distributed:** `--sharding-policy {replicated,sharded_context}`, `--routing-switch-at`.

**Workload:** `--workload-mode {single,batched,multi_turn}`, `--batch-size`,
`--multi-turn-length`, `--repeat-queries`, `--warmup-queries`.

**Protocol controls (new):** `--context-source {auto,gold,retrieved}` (confound control),
`--reset-cache-between-trials` (cold-start per trial; needs `VLLM_SERVER_DEV_MODE=1`).

**Compression axis (new):** `--compress-method {none,llmlingua2,llmlingua}`, `--compress-ratio <keep>`
(0.5 = 2× compression), `--kv-cache-dtype {none,fp8}` (record-only here; also pass to the server).

**Speculative:** vLLM configures speculation at **launch** — start the server with
`VLLM_SPECULATIVE_CONFIG='{"method":"ngram","num_speculative_tokens":5}'`, then run the
`speculative` baseline. (The old `--speculative-model` flag is deprecated.)

> No `--phase`, `--all-baselines`, `--trials`, `--queries`, or disagg-prefill flag.

### 6.1 Compression axis (the 2×2)
```bash
# RAG with text compression of retrieved docs (needs: pip install llmlingua)
python3 scripts/run_experiment.py --baseline compressed_rag --model Qwen/Qwen3-8B \
  --dataset squad_v2 --num-queries 50 --num-trials 3 --compress-ratio 0.5 --rebuild-ir-index
# CAG with KV-cache compression (GPU): launch the server with fp8 KV, then run:
VLLM_KV_CACHE_DTYPE=fp8 ./scripts/manage_vllm_server.sh restart Qwen/Qwen3-8B
python3 scripts/run_experiment.py --baseline compressed_cag --model Qwen/Qwen3-8B \
  --dataset squad_v2 --num-queries 50 --num-trials 3 --kv-cache-dtype fp8
# MLA arm (architectural KV compression): use configs/model/deepseek-v2-lite.yaml on a GPU.
```

### 6.2 Cache-state control for trials (resolves the per-trial flush question)
There are **two legitimate measurement regimes** — declare which you're using:
- **Cold-start per trial** (each trial starts from an empty cache): start the server with
  `VLLM_SERVER_DEV_MODE=1` and add `--reset-cache-between-trials`. This flushes the vLLM prefix
  cache via `POST /reset_prefix_cache` between trials — no model reload needed.
- **Warm / steady-state** (cache pre-populated, the regime `hybrid_warm` targets): just run; the
  seeded resampling now gives each trial *different* queries, so it's not a pure replay.
Either is valid; what matters is that the regime is controlled and stated. For confound-free
quality comparisons also pass `--context-source gold` (or `retrieved`).

---

## 7. Analyze results

```bash
# 7a. Integrity check (no truncated/erroring trials)
python3 scripts/verify_results.py

# 7b. Per-query significance testing (Wilcoxon signed-rank, Holm-corrected, bootstrap CIs)
python3 scripts/statistical_tests.py \
  --results-dir analysis/phase1/results --reference no_cache \
  --metrics ttft_ms latency_ms grounding_score faithfulness hallucinated_span_ratio f1_score \
  --output analysis/phase1/stats.json --latex-out analysis/phase1/stats.tex

# 7c. Publication plots
python3 scripts/generate_publication_plots.py \
  --results-dir analysis/phase1/results --output-dir analysis/phase1/images
python3 scripts/generate_compact_figures.py
```
Per-query quality fields: `grounding_score`, `hallucination_detected`,
`hallucinated_span_ratio` (LettuceDetect); `faithfulness`, `supported_claim_ratio`
(claim-level NLI); `context_relevance`; baseline-rescaled `completeness_bertscore`;
`f1_score`/`exact_match`. `None` = metric model unavailable (excluded from means).

### 7.1 vLLM serving telemetry (cage-stats)
Capture what CAGE's own metrics don't expose — **spec-decode acceptance, KV-compression
ratio/dtype, prompt-token source breakdown, prefix-cache hit rate, multi-vendor GPU** — and
print a one-shot dashboard, via the standalone
[cage-stats](https://github.com/lucasmdocarmo/cage-stats) package.

**It's a git dependency in `requirements.txt`**, so `pip install -r requirements.txt` pulls it
automatically (locally and on the cloud VM — no extra step). *Prerequisite:* the cage-stats
repo must have the restructured package committed + pushed first; if it's private, ensure git
creds on the install host. Local-dev alternative without installing: `export CAGE_STATS_HOME=/Users/lucasmariano/cage-stats`.

Then add `--vllm-telemetry` to any run; a snapshot is saved to `<output-dir>/vllm_telemetry.json`
and into `aggregated_metrics.json` under `vllm_telemetry`, and a dashboard prints to the terminal:
```bash
python3 scripts/run_experiment.py --baseline compressed_cag --model Qwen/Qwen3-8B \
  --dataset squad_v2 --num-queries 50 --num-trials 3 --vllm-telemetry --api-base http://localhost:9000
```
Standalone (no CAGE): `cage-stats --url http://localhost:9000` (live TUI) ·
`cage-stats --once` (static dashboard) · `cage-stats --once --json` (snapshot for scripting).
Gracefully skips if cage-stats isn't installed.

---

## 8. Tests

```bash
bash scripts/run_tests.sh                                          # 1-replica cluster + pytest
pytest -m vllm                                                     # adapter tests (live vLLM)
ROUTER_TEST_API_BASE=http://localhost:9000 pytest -m integration  # router tests (live router)
```

---

## 9. Cloud deploy (GCP) with durable results

> **Two topologies — pick by what you're measuring.** They are different on purpose:
>
> | | **Path A — single GPU VM** | **Path B — multi-VM cluster** |
> |---|---|---|
> | What | one GPU box runs vLLM **and** the experiment | 3 GPU replicas + 1 CPU router (Terraform) |
> | Use for | the **6 single-server baselines** (Phase 2 suite) | the **`distributed` baseline** at real scale (Phase 3) |
> | Runner | `scripts/cloud_run.sh` (wraps `run_phase1.sh`) | `run_experiment.py --baseline distributed` against the router |
> | Why split | `run_phase1.sh` starts a **local** vLLM and toggles prefix-cache on/off — that needs a GPU on the box running it | the cluster's replicas are fixed/always-on; only the distributed baseline exercises the router |
>
> ❌ Do **not** run `cloud_run.sh`/`run_phase1.sh` on the cluster's CPU **router** — it has no GPU.

### 9.0 One-time GCP setup (both paths)
```bash
gcloud auth login
gcloud config set project <PROJECT_ID>
gcloud config set compute/region us-central1
gcloud config set compute/zone   us-central1-a
gcloud services enable compute.googleapis.com cloudresourcemanager.googleapis.com storage.googleapis.com
# Request GPU quota: IAM & Admin > Quotas -> NVIDIA_L4_GPUS >= 1 (Path A) or >= 3 (Path B).
gcloud compute accelerator-types list --filter="zone:us-central1-a AND name:nvidia-l4"  # confirm
# Create the durable results bucket once (Terraform also makes one; this is for Path A):
gsutil mb -l us-central1 gs://$(gcloud config get-value project)-cage-results || true
gsutil versioning set on gs://$(gcloud config get-value project)-cage-results
```

### Path A — full suite on ONE GPU VM (recommended for Phase 2)
```bash
# 1. Create a single L4 GPU VM with NVIDIA drivers auto-installed:
PROJECT=$(gcloud config get-value project)
gcloud compute instances create cage-gpu \
  --zone=us-central1-a --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 --maintenance-policy=TERMINATE \
  --image-family=common-cu121-debian-11 --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB --boot-disk-type=pd-ssd \
  --scopes=cloud-platform --metadata=install-nvidia-driver=True

# 2. SSH in and set up:
gcloud compute ssh cage-gpu --zone=us-central1-a
nvidia-smi                                   # confirm the GPU is visible
git clone <your-repo-url> cage && cd cage    # or scp your repo
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install vllm
#   requirements.txt pulls cage-stats (git dependency) for --vllm-telemetry. If that
#   repo is private, authenticate git on the VM first (e.g. `gh auth login` / a PAT),
#   or install it explicitly:  pip install "cage-stats @ git+https://github.com/lucasmdocarmo/cage-stats.git"
#   To run WITHOUT telemetry, comment the cage-stats line in requirements.txt (or VLLM_TELEMETRY=0 below).
export HF_TOKEN=hf_xxx                        # if gated

# 3. Run the suite + continuous GCS sync (nohup survives SSH drops):
#    Telemetry is auto-captured (cloud_run.sh sets VLLM_TELEMETRY=1) ->
#    each baseline writes <output>/vllm_telemetry.json, mirrored to GCS.
nohup bash scripts/cloud_run.sh Qwen/Qwen3-8B 100 10 > run.log 2>&1 &
tail -f run.log
```
`cloud_run.sh` starts Redis, runs the 6 single-server baselines on the local GPU vLLM, and
mirrors `analysis/` to `gs://<project>-cage-results` every 2 min and at exit. It skips the
distributed baseline by default (`ENABLE_DISTRIBUTED=0`) because 3 local replicas OOM a 24GB
L4 — run that on Path B. Tunables: `SYNC_INTERVAL`, `SYNC_DIR`, `CAGE_RESULTS_BUCKET`.
When done: `gcloud compute instances delete cage-gpu --zone=us-central1-a` (results stay in GCS).

### Path B — the `distributed` baseline on the multi-VM cluster
```bash
cd terraform/gcp
cp terraform.tfvars.example terraform.tfvars   # set project_id, model_name, repo_url, hf_token
terraform init
terraform apply -var="project_id=$(gcloud config get-value project)" -var="hf_token=$HF_TOKEN"
terraform output            # router_external_ip, results_bucket

# Verify the cluster is healthy:
ROUTER_IP=$(terraform output -raw router_external_ip)
curl -fsS http://$ROUTER_IP:9000/health && curl -fsS http://$ROUTER_IP:9000/stats

# Run ONLY the distributed baseline against the router, then sync:
gcloud compute ssh cage-router --zone=us-central1-a
cd /opt/cage
CAGE_REQUIRE_DISTINCT_REPLICAS=1 python3 scripts/run_experiment.py \
  --baseline distributed --baseline-label distributed_router_replicated \
  --model Qwen/Qwen3-8B --dataset squad_v2 --num-queries 100 --num-trials 10 \
  --api-base http://localhost:9000 --sharding-policy replicated \
  --output-dir analysis/phase1/results/distributed_router_replicated
bash scripts/sync_results_to_gcs.sh analysis

# Tear down GPUs (results stay safe in the versioned bucket):
cd terraform/gcp && terraform destroy -var="project_id=$(gcloud config get-value project)" -var="hf_token=$HF_TOKEN"
```
Terraform installs the NVIDIA driver, `git clone`s the repo into `/opt/cage` (if `repo_url`
set), `/health`-gates the router, enables the APIs, and creates the versioned bucket. ⚠️ GPUs
bill continuously — destroy promptly.

### 9.5 Validate + analyze (either path)
```bash
python3 scripts/statistical_tests.py --results-dir analysis/phase1/results \
  --reference no_cache --output analysis/phase1/stats.json --latex-out analysis/phase1/stats.tex
python3 scripts/generate_publication_plots.py \
  --results-dir analysis/phase1/results --output-dir analysis/phase1/images
```

---

## 10. Retrieve & reuse results later (from anywhere)
```bash
gsutil -m cp -r gs://<project>-cage-results/analysis ./analysis
# then the same statistical_tests.py / plot commands as §7/§9.5 — identical CSV/JSON schema.
```

---

## 11. Phase 3 (HPC / A100) — switch via tfvars, no file edits
In `terraform.tfvars`:
```hcl
gpu_type     = "nvidia-tesla-a100"
machine_type = "a2-highgpu-1g"
nic_type     = "GVNIC"     # ~100 Gbps for cross-node KV transfer
network_mtu  = 8896        # jumbo frames
model_name   = "Qwen/Qwen3-14B"
```
`terraform apply`, then run the distributed baseline as in §9 Path B. (Phase 3's real
cross-node KV transfer is **not yet implemented** — the distributed baseline is still
simulated; see §13.)

---

## 12. Docker / Kubernetes (alternative to bare Terraform)
```bash
docker compose -f docker/docker-compose.yml up -d           # local CPU multi-replica (Apple Silicon)
HF_TOKEN=hf_xxx docker compose -f docker/docker-compose.gpu.yml up -d   # GPU hosts (fixed)
```
K8s manifests are in `k8s/` — GPU limits are commented out and the router image must be
pushed to a registry first (see VALIDATION_AND_SOTA_REVIEW.md I4/I7).

---

## 13. Known gaps before trusting numbers (protocol, not setup)
The metric *code* and GCS persistence are solid, but the experiment **protocol** still has
open issues (full detail in [`VALIDATION_AND_SOTA_REVIEW.md`](VALIDATION_AND_SOTA_REVIEW.md) Part C):
1. **Distributed baseline is simulated** — `cache_manager.py` models KV-transfer cost; no real
   tensors move. Wire real KV transfer (vLLM V1 + LMCache) for a legitimate "distributed" claim.
2. **Gold-vs-retrieved confound** — CAG baselines get the gold passage, RAG gets retrieved.
3. **Per-trial independence** — restart vLLM / flush caches between trials; `--seed` is a no-op for HF datasets.
4. **Warm-hybrid leakage** — warmup queries overlap the measured set.
5. **Rebuild IR indices** with `--rebuild-ir-index` so the e5 prefix fix takes effect.
6. **Install LettuceDetect** on the cluster (or `CAGE_DISABLE_LETTUCEDETECT=1`).

---

## 14. Common gotchas
- Old-doc flags fail: use `--num-queries`/`--num-trials`, **not** `--queries`/`--trials`; there is no `--phase`/`--all-baselines`/`--enable-disagg-prefill`.
- `gsutil` permission denied on the VM: the VM SA needs `roles/storage.objectAdmin` on the bucket — Terraform grants this; for a manually-created bucket, add it.
- RAG looks weak: ensure indices were rebuilt post e5-prefix fix (`--rebuild-ir-index`).
- LettuceDetect OOM/load failure: `export CAGE_DISABLE_LETTUCEDETECT=1`.

---

## 15. Quick reference card
| Goal | Command |
|---|---|
| Local full Phase 1 | `bash scripts/run_phase1.sh` |
| Bigger sweep | `NUM_QUERIES=100 NUM_TRIALS=10 bash scripts/run_phase1.sh Qwen/Qwen3-8B` |
| Single baseline | `python3 scripts/run_experiment.py --baseline rag --model … --dataset squad_v2 --num-queries 50 --num-trials 3` |
| Cloud run + persist | `nohup bash scripts/cloud_run.sh Qwen/Qwen3-8B 100 10 > run.log 2>&1 &` |
| Sync results to GCS | `bash scripts/sync_results_to_gcs.sh analysis` |
| Pull results back | `gsutil -m cp -r gs://<project>-cage-results/analysis ./analysis` |
| Significance tests | `python3 scripts/statistical_tests.py --results-dir analysis/phase1/results` |
| Provision cloud | `cd terraform/gcp && terraform apply -var=project_id=… -var=hf_token=…` |
| Tear down (keep results) | `cd terraform/gcp && terraform destroy -var=project_id=… -var=hf_token=…` |
