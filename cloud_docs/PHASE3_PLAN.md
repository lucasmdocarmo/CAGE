# Phase 3 — Multi-Node HPC Cluster: Plan & Definition of Done

> Phase 3 takes CAGE to the regime the title is about: the KV cache as a distributed object across
> several GPU nodes. It builds on Phase 2 ([`PHASE2_CHECKLIST.md`](PHASE2_CHECKLIST.md)). Be precise
> about what is **real now** versus what is the **Phase-3 implementation**, because the dissertation
> states this honestly and the defense depends on it.

## What is real now (runs on GCP today)

- A **prefix-aware router** in front of **N vLLM replicas**, each a real engine with its own local
  prefix cache. Routing hashes the tokenized prompt prefix and sends matching requests to the
  replica that already holds those KV blocks. The router forwards blocking and streamed requests
  and reports the serving replica per request.
- The `distributed` baseline runs end-to-end against this cluster (`--baseline distributed` against
  the router). Under the **replicated** policy every node holds the context, so the modeled
  cross-node transfer cost is **zero** — the arm measures real prefix-affinity routing, not transfer.
- Terraform provisions the cluster (router + N GPU replicas + Redis + durable GCS bucket), with the
  high-bandwidth interconnect (GVNIC, MTU 8896) and tensor-parallel options provisioned but not yet
  exercised for real transfer.

## What is the Phase-3 implementation (the future work the dissertation names)

**Real cross-node KV-tensor transfer.** Today `SimulatedKVCacheManager` derives a transfer size and
latency from the cache footprint and the interconnect bandwidth (analytic model; no tensors move).
Phase 3 replaces this with a real vLLM **KV connector**:

- Launch replicas with `--kv-transfer-config` and a connector such as **LMCache** or **NIXL**.
- Exercise a **sharded** context policy (each node holds 1/N of the context) so transfer cost is
  actually paid and measured, instead of zeroed by the replicated policy.
- Read the measured transfer bytes/latency from serving telemetry (cage-stats token-source
  breakdown: recomputed vs cache-hit vs externally transferred), replacing the analytic
  `transfer_bytes_for` estimate with empirical numbers.

This is tracked as `DEV_BACKLOG #6` and is the explicit object of the HPC phase.

## Steps

**0. Provision the cluster.**
- [ ] `terraform -chdir=terraform/gcp apply -var num_replicas=3` (A100 `a2-highgpu-1g` for headroom,
      or L4 for a cheaper routing-only run; keep `preemptible=false` so all replicas stay up together).
- [ ] For real transfer later: `-var nic_type=GVNIC -var network_mtu=8896`.

**1. Bootstrap the driver/router host** with `scripts/setup/setup_gpu_cloud.sh` so the orchestrator,
cage-stats, and telemetry run there too.

**2. Run the distributed baseline (routing, real replicas).**
- [ ] `python scripts/run_experiment.py --baseline distributed --api-base http://<router>:9000 ...`
      plus `sync_results_to_gcs.sh` (see `RUNBOOK.md` Path B).

**3. (Implementation) Wire the real KV connector.**
- [ ] Add `--kv-transfer-config` + LMCache/NIXL to the replica launch (terraform `vllm_extra_args`).
- [ ] Switch `cache_manager.py` from the simulated path to reading connector telemetry.
- [ ] Run the sharded policy and confirm non-zero measured transfer bytes in `vllm_telemetry.json`.

**4. Analyze + teardown** as in Phase 2.

## Definition of Done

- [ ] The `distributed` baseline runs on a real multi-node GCP cluster with prefix-affinity routing,
      and its results join the other eight baselines.
- [ ] Telemetry attributes prompt tokens to recomputed / cache-hit / transferred on the cluster.
- [ ] **Stretch (the HPC contribution):** real cross-node KV transfer is wired (LMCache/NIXL),
      the sharded policy pays a measured transfer cost, and the analytic `transfer_bytes` model is
      validated against the empirical numbers.

> Until the stretch item lands, every cross-node-transfer number in the dissertation must be labeled
> **simulated/analytic**. Do not present simulated transfer as measured.
