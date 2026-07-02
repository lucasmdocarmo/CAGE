# Phase 3 "Plan B": COMMITTED Build-Ready Plan: 2x A3 Ultra (H200) with Real GPUDirect RDMA over RoCE

> This is a COMMITTED, build-ready plan, not a feasibility study. It replaces the prior
> exploratory Plan-B study (Slurm + Parallelstore variants). The configuration below is
> FIXED. Slurm and Parallelstore are OUT and are not deliberated here. This document folds
> in the gap-filling research on vLLM NixlConnector + UCX/RoCE and the adversarial
> verification that confirmed both load-bearing claims (no-Slurm provisioning is real; vLLM
> NIXL genuinely rides GPUDirect RoCE and the TCP-fallback failure mode is detectable).
> Companion to `PHASE3_PLAN.md` (the Plan-A A100 design) and `PHASE2_PLAN_OF_RECORD.md`.
> No em-dashes. Every dollar and Gbps figure is flagged verify-live; the console is
> authoritative for the exact RDMA-capable target zone before any provisioning.

---

## 1. Scope and decision

This plan builds a single, real, cross-node KV-cache-transfer measurement over GPUDirect RDMA on RoCE, at the lowest defensible cost, with every component that does not serve that measurement removed.

### Committed configuration (FIXED)

```
+----------------------------------------------------------------------------+
| CAGE Phase 3 "Plan B" - COMMITTED CONFIG                                    |
+----------------------------------------------------------------------------+
| Compute        2x a3-ultragpu-8g (16x NVIDIA H200 total)                    |
| RDMA fabric    Real GPUDirect RDMA over RoCE v2 (8x ConnectX-7 per node)    |
| Networking     3 VPCs: 2 host gVNIC VPCs + 1 RoCE VPC (MRDMA profile)       |
|                with 8 mrdma rail subnets; MTU 8896 everywhere               |
| KV transport   vLLM NixlConnector, NIXL backend = UCX (RoCE carrier)        |
| Placement      Compact placement, collocated, same zone                     |
| Capacity       DWS Flex-start (consumes preemptible H200 quota)             |
| Storage        GCS-only (regional Standard bucket); Hyperdisk = node root   |
| Control plane  1 tiny CPU VM (e2-standard-2) running the NIXL proxy/router  |
| Run shape      ~5h single Flex-start window, then teardown to $0            |
+----------------------------------------------------------------------------+
```

### Explicitly OUT (one line each, not deliberated)

- **Slurm:** out. CAGE runs a long-lived vLLM serving cluster plus a router for a few hours with no queue or job-allocation semantics, so a batch scheduler buys nothing and only adds a persistent controller plus login VM.
- **Parallelstore:** out. CAGE's I/O is tiny (read weights once, write MB-scale CSV/JSON) and the KV transfer is GPU-to-GPU over RDMA with no shared filesystem, so a 12 TiB / multi-GBps parallel FS is pure overkill (and Parallelstore is being deprecated 2026-10-31).

### Decision

**Go is conditional** on two gates that must both clear before any spend (Section 8): the real-RDMA contribution must be load-bearing for the dissertation, and the capacity/quota gate must clear. If either fails, **abort to Plan A** (3x A100, gVNIC/TCP, ~US$50) and frame the transport result as analytic/TCP. This document is the build recipe for the case where both gates clear.

---

## 2. Why A3 Ultra (the RDMA-tier fact)

Real GPUDirect RDMA over RoCE has a hardware floor on GCP, confirmed by official GCP documentation and upheld in verification. The transport label is GCP's own:

- **A3 High (a3-highgpu-8g, H100):** GPUDirect-**TCPX** ("optimized guest TCP"). Not RDMA.
- **A3 Mega (a3-megagpu-8g, H100):** GPUDirect-**TCPXO** ("RDMA-**like**, offloaded" TCP). Not RoCE. Presenting it as RDMA is a defensibility risk.
- **A3 Ultra (a3-ultragpu-8g, H200):** **GPUDirect RDMA over RoCE v2** over 8x ConnectX-7. This is the floor and the committed choice.
- A4 (B200) and A4X (GB200/GB300) also do RoCE RDMA but at a higher, pricier tier with no benefit for moving a few-GB KV cache.

**Granularity (the cost driver, already verified):** the real-RDMA families have no sub-8-GPU SKUs. A3 Ultra is 8-GPU-only, so a cross-node measurement needs 2 whole nodes = 16 H200. That is structural and unavoidable. Aggregate per node: ~3,200 Gbps GPU RoCE over 8x ConnectX-7 plus ~400 Gbps over 2x gVNIC (~3,600 Gbps total). Reference: https://docs.cloud.google.com/compute/docs/gpus/gpu-network-bandwidth

---

## 3. Architecture

Two H200 nodes connected by a dedicated RoCE fabric, with a minimal CPU router and GCS as the only durable store. Nothing else.

### 3.1 Topology

```
                         GCS (regional Standard bucket)
                        weights in / per-query CSV+JSON out
                                     ^   ^
              copy-from-GCS on boot  |   |  results write
                                     |   |
   +----------------------------+    |   |    +----------------------------+
   | cage-node-1  a3-ultragpu-8g|    |   |    | cage-node-2  a3-ultragpu-8g|
   |  8x H200, TP=8             |    |   |    |  8x H200, TP=8             |
   |  vLLM + NixlConnector      |    |   |    |  vLLM + NixlConnector      |
   |  (prefiller role)          |    |   |    |  (decoder role)            |
   |                            |    |   |    |                            |
   |  nic0,nic1 = gVNIC --------+----+   +----+----- gVNIC = nic0,nic1     |
   |  nic2..nic9 = 8x MRDMA     |                   8x MRDMA = nic2..nic9  |
   +-----------+----------------+                   +---------+-----------+
               |                                              |
               |   ===== RoCE VPC (MRDMA profile) =====       |
               +======== 8 rail subnets, MTU 8896 ============+
                    KV-cache bulk transfer (UCX/rc_mlx5/dc_mlx5)
                    GPU buffer -> GPU buffer, GPUDirect RDMA

               +----------------------------------------+
               | router-vm  e2-standard-2 (host VPC)    |
               | vLLM NIXL toy_proxy_server / prefix     |
               | router -> prefiller/decoder; no Slurm   |
               +----------------------------------------+
```

### 3.2 Networks (the hard prerequisite that must exist before the VMs)

A3 Ultra requires exactly 3 VPCs. Without the RoCE VPC, NIXL never sees the ConnectX-7 NICs and silently uses gVNIC/TCP.

- **2 host VPCs (regular, gVNIC):** one per gVNIC, MTU 8896. Carry host traffic, the NIXL side-channel handshake (port 5600), the proxy (8192) and the vLLM HTTP ports (8000).
- **1 RoCE VPC (MRDMA network profile):** created with `--network-profile=ZONE-vpc-roce`, custom subnet mode, MTU 8896, with **8 rail subnets** (one per ConnectX-7 NIC). RoCE VPCs only accept MRDMA NICs. Reference: https://docs.cloud.google.com/vpc/docs/rdma-network-profiles
- The network profile is a beta resource and exists only in zones where A3 Ultra is offered. Confirm with `gcloud compute network-profiles list`.

### 3.3 Compute and placement

- 2x a3-ultragpu-8g, same zone, attached to a **compact placement policy** (`--collocation=collocated --max-distance=1`) for lowest RoCE latency.
- Each node: 10 NICs total (nic0/nic1 = GVNIC on the 2 host VPCs, nic2..nic9 = MRDMA on the 8 rail subnets), `--maintenance-policy=TERMINATE` (A3 Ultra cannot live-migrate), boot disk `hyperdisk-balanced`.

### 3.4 KV connector

- **vLLM NixlConnector** moves KV cache GPU-buffer to GPU-buffer over the RoCE fabric via the NIXL UCX backend. No shared filesystem. This is what makes the transfer "real."
- A small **NIXL side-channel** (metadata/handshake) rides gVNIC; the **bulk KV tensor** rides UCX/RoCE. These are distinct paths and must not be conflated (a working handshake over gVNIC with a TCP-fallback bulk transfer is still a failed result).

### 3.5 Storage

- **GCS-only** regional Standard bucket for weights in and per-query results out. Same-region egress is free.
- **Hyperdisk-balanced** is the node root disk only (size >= 200 GB to hold OS + Qwen3-8B/MiMo weights + container). CAGE keeps no durable state on it.

### 3.6 Control plane

- One **e2-standard-2** CPU VM on a host VPC runs the vLLM NIXL proxy/router (`toy_proxy_server.py` pattern). No Slurm controller, no login node. The proxy may alternatively run on one of the GPU nodes.

---

## 4. Provisioning recipe (build-ready, ordered)

For exactly 2 nodes, **plain gcloud is cleaner than the Cluster Toolkit blueprint.** Verification confirmed both paths work: Cluster Toolkit ships a no-scheduler `a3ultra-vm.yaml` (+ `a3ultra-vm-deployment.yaml`, deployed via `gcluster deploy`, Cluster Toolkit >= v1.44.1) alongside the Slurm blueprint, but its defaults are `RESERVATION_BOUND` provisioning plus a Filestore `/home` that you would have to edit out (CAGE is GCS-only, Flex-start). For 2 nodes that editing exceeds the cost of the gcloud commands below. **Use plain gcloud; reserve the blueprint only if reproducible IaC is later required.**

Placeholders: `ZONE` (an A3-Ultra zone that also exposes `ZONE-vpc-roce`), `REGION`, `RDMA_PREFIX`, `GVNIC_PREFIX`, image family/project per the AI Hypercomputer A3 Ultra guidance.

### STEP 0: Preemptible H200 quota (the unblock, slowest item)

On the Quotas page request `Preemptible NVIDIA H200 GPUs` >= 16 in `REGION` (2 nodes x 8). Flex-start consumes the **preemptible** H200 pool, not standard GPU quota.

> verify-live: the exact metric string (assumed `PREEMPTIBLE_NVIDIA_H200_GPUS` by analogy with `PREEMPTIBLE_NVIDIA_L4_GPUS`) must be confirmed on the live Quotas page. Quota approval is fast; the real blocker is capacity-wait under Flex-start.

Reference: https://docs.cloud.google.com/kubernetes-engine/docs/how-to/dws-flex-start-training

### STEP 1: Confirm the RoCE profile and capacity exist in the zone

```
gcloud compute network-profiles list --filter="name~roce"
gcloud compute network-profiles describe ZONE-vpc-roce
```

Confirm A3 Ultra capacity is offered in `ZONE`. Recreate the CAGE GCS bucket (GCS-only; no Filestore/Parallelstore).

### STEP 2: Host VPCs (2x gVNIC)

```
for N in 0 1; do
  gcloud compute networks create GVNIC_PREFIX-net-$N \
    --subnet-mode=custom --mtu=8896
  gcloud compute networks subnets create GVNIC_PREFIX-sub-$N \
    --network=GVNIC_PREFIX-net-$N --region=REGION --range=10.$N.0.0/16
done
```

### STEP 3: RoCE VPC + 8 mrdma rail subnets

```
gcloud compute networks create RDMA_PREFIX-mrdma \
  --network-profile=ZONE-vpc-roce --subnet-mode=custom --mtu=8896

for N in 0 1 2 3 4 5 6 7; do
  gcloud compute networks subnets create RDMA_PREFIX-mrdma-sub-$N \
    --network=RDMA_PREFIX-mrdma --region=REGION --range=10.$((N+2)).0.0/16
done
```

`--network-profile` is the flag that turns the VPC into a RoCE/MRDMA VPC; it is only valid with `ZONE-vpc-roce` (beta surface). IP ranges are illustrative; pick non-overlapping CIDRs. Reference: https://docs.cloud.google.com/compute/docs/gpus/create-gpu-vm-a3u-a4

### STEP 4: Firewall

Add internal allow rules on the 2 host VPCs (tcp/udp/icmp within the ranges) so the NIXL side-channel (5600), the proxy (8192) and vLLM HTTP (8000) are reachable between the two nodes and the router. RoCE bulk traffic flows on the MRDMA VPC.

### STEP 5: Compact placement policy

```
gcloud beta compute resource-policies create group-placement cage-pp \
  --collocation=collocated --max-distance=1 --region=REGION
```

> verify-live: combining `--max-distance=1` with FLEX_START may worsen capacity-grant time. If waits are long, drop to `--max-distance=2` or rely on same-zone alone (2 nodes in one zone already share the RoCE fabric).

### STEP 6: Provision both nodes via DWS Flex-start

10 `--network-interface` flags required: nic0/nic1 = GVNIC on the host VPCs, nic2..nic9 = MRDMA one per rail subnet. Repeat the MRDMA lines for `mrdma-sub-1` .. `mrdma-sub-7`.

```
gcloud compute instances create cage-node-1 \
  --machine-type=a3-ultragpu-8g --zone=ZONE \
  --image-family=IMAGE_FAMILY --image-project=IMAGE_PROJECT \
  --boot-disk-type=hyperdisk-balanced --boot-disk-size=300 \
  --scopes=cloud-platform --resource-policies=cage-pp \
  --provisioning-model=FLEX_START --instance-termination-action=DELETE \
  --max-run-duration=18000s --request-valid-for-duration=86400s \
  --maintenance-policy=TERMINATE --reservation-affinity=none \
  --network-interface=nic-type=GVNIC,network=GVNIC_PREFIX-net-0,subnet=GVNIC_PREFIX-sub-0 \
  --network-interface=nic-type=GVNIC,network=GVNIC_PREFIX-net-1,subnet=GVNIC_PREFIX-sub-1,no-address \
  --network-interface=nic-type=MRDMA,network=RDMA_PREFIX-mrdma,subnet=RDMA_PREFIX-mrdma-sub-0,no-address \
  --network-interface=nic-type=MRDMA,network=RDMA_PREFIX-mrdma,subnet=RDMA_PREFIX-mrdma-sub-1,no-address \
  --network-interface=nic-type=MRDMA,network=RDMA_PREFIX-mrdma,subnet=RDMA_PREFIX-mrdma-sub-2,no-address \
  --network-interface=nic-type=MRDMA,network=RDMA_PREFIX-mrdma,subnet=RDMA_PREFIX-mrdma-sub-3,no-address \
  --network-interface=nic-type=MRDMA,network=RDMA_PREFIX-mrdma,subnet=RDMA_PREFIX-mrdma-sub-4,no-address \
  --network-interface=nic-type=MRDMA,network=RDMA_PREFIX-mrdma,subnet=RDMA_PREFIX-mrdma-sub-5,no-address \
  --network-interface=nic-type=MRDMA,network=RDMA_PREFIX-mrdma,subnet=RDMA_PREFIX-mrdma-sub-6,no-address \
  --network-interface=nic-type=MRDMA,network=RDMA_PREFIX-mrdma,subnet=RDMA_PREFIX-mrdma-sub-7,no-address
```

Repeat verbatim for `cage-node-2`. `--max-run-duration=18000s` is the 5h lease cap (Flex-start allows up to 7 days, well above need); `--request-valid-for-duration=86400s` is how long the queued request waits for capacity before failing. Reference: https://docs.cloud.google.com/compute/docs/gpus/gpu-network-bandwidth

### STEP 7: Hyperdisk and 2 nodes up

Boot disk is `hyperdisk-balanced` (set in STEP 6; A3 Ultra does not support standard PD boot for these flows). Once both nodes are running, verify the fabric is live before any measurement (this is the standing live-infra check): `ibv_devices` / `rdma link` to confirm 8 MRDMA NICs present and RDMA up, plus a 2-node NCCL all-reduce or gIB perf test.

### STEP 8: Bootstrap

Install GPUDirect/RDMA binaries + drivers per the AI Hypercomputer A3 Ultra image guidance (or use a GPU/HPC image with the gIB NCCL plugin + CX-7 RDMA drivers preinstalled). Copy weights from GCS to each node's Hyperdisk. Stand up the `e2-standard-2` router VM on a host VPC. Proceed to Section 5 software integration.

> verify-live (carry-over): the exact public image family for A3 Ultra with RDMA drivers out-of-the-box vs needing a gIB container is unconfirmed from docs and must be checked on first boot.

---

## 5. Software integration

### 5.1 vLLM NixlConnector configuration

The KV connector is selected via `--kv-transfer-config`, a JSON blob. NixlConnector is the NIXL-backed connector; the transport backend is chosen **inside NIXL**, not by vLLM, via `kv_connector_extra_config.backends`.

```
--kv-transfer-config '{
  "kv_connector": "NixlConnector",
  "kv_role": "kv_both",
  "kv_load_failure_policy": "fail",
  "kv_connector_extra_config": {"backends": ["UCX"]}
}'
```

- `kv_role`: use `kv_both` for symmetric cross-node measurement. Verification note: `kv_both` is a **placeholder** (NixlConnector does not distinguish producer/consumer from this field; the upper-level proxy assigns prefiller/decoder via `--prefiller-hosts`/`--decoder-hosts`). It is **not** deprecated (an intermediate source mislabeled it so); `kv_producer`/`kv_consumer`/`kv_both` are all valid.
- `backends: ["UCX"]` is the correct choice and UCX is the unnamed default that carries RoCE/RDMA. **AVOID LIBFABRIC** (Section 5.4). Verification flags this as under-documented upstream: the vLLM doc's only worked backend example is the known-broken `["LIBFABRIC"]`, so confirm UCX on the live VM. Reference: https://docs.vllm.ai/en/v0.11.0/features/nixl_connector_usage.html

Side-channel env (multi-node; defaults are localhost-only and break across hosts):

```
VLLM_NIXL_SIDE_CHANNEL_HOST=<reachable IP of peer>   # default localhost
VLLM_NIXL_SIDE_CHANNEL_PORT=5600                     # unique per worker per host
VLLM_NIXL_ABORT_REQUEST_TIMEOUT=480
```

### 5.2 Forcing RoCE and preventing silent TCP fallback (UCX env, BOTH nodes)

Naming UCX does not by itself guarantee RDMA; UCX still picks a transport at runtime from `UCX_TLS`/`UCX_NET_DEVICES`. Pin it:

```
UCX_TLS=rc_mlx5,dc_mlx5,cuda_copy,cuda_ipc   # deliberately EXCLUDE tcp
UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,...,mlx5_7:1   # the 8 ConnectX-7 RoCE devices
UCX_IB_GID_INDEX=<RoCE v2 index from show_gids>
UCX_IB_TRAFFIC_CLASS=106
UCX_LOG_LEVEL=info   # smoke run only; verbose
```

Excluding `tcp` is the deliberate trick: it converts a silent TCP fallback into a loud failure during the smoke test. Honest caveats from verification:

- This RDMA-only pin is a **UCX/community hardening technique, not vLLM-official** (the upstream vLLM doc only shows the looser `UCX_TLS=all` / `UCX_NET_DEVICES=all`). It is correct hardening; keep it, but it is not vLLM-documented.
- Exact `mlx5` device names and the RoCE-v2 GID index are **instance-specific** and cannot be hardcoded ahead of provisioning. Read them live on first boot with `ibv_devinfo`, `show_gids`, and `ucx_info -d` (confirm `rc_mlx5`/`dc_mlx5` transports present). References: https://github.com/openucx/ucx/wiki/UCX-environment-parameters and https://github.com/RESMP-DEV/vllm-1/blob/main/docs/features/nixl_connector_usage.md

### 5.3 Sharded-KV vs disaggregation choice for CAGE

**Decision: use plain NixlConnector, not LMCache-on-NIXL.** For CAGE's purpose (MEASURE raw RDMA KV movement; prefix-router with each node holding 1/N of the context KV), plain NixlConnector gives the cleanest, lowest-layer RDMA transfer measurement that CAGE fully controls, and only couples vLLM<->NIXL. Honest scoping caveats:

- There is **no off-the-shelf connector for "each node holds 1/N of the KV"**; the sharded-context prefix router is custom on top of either choice. LMCache's documented cross-node story is framed around P/D disaggregation and session sharing, not a sharded-context router.
- LMCache would add a named KV-cache layer (storage/sharing mode) if the dissertation needed to DEMONSTRATE a cache layer rather than MEASURE raw movement, but it brings a fragile version matrix (Section 5.4). CAGE's contribution is the measurement, so plain NIXL wins. Reference: https://blog.lmcache.ai/en/2025/04/11/shaping-nixl-based-pd-disaggregation-in-vllm-v1/

> open item (verify-live): whether CAGE's prefix-aware router maps onto vLLM's NIXL proxy model (`--prefiller-hosts`/`--decoder-hosts`) or whether sharded-context transfer requires driving NixlConnector outside the standard P/D proxy. This determines if `toy_proxy_server.py` is reusable or needs replacement.

### 5.4 Version compatibility (pin a Phase-3-specific triple)

- **Install NIXL with `uv pip install nixl`** (NIXL >= 1.0.0 pulls the right CUDA backend; vLLM pins the required version in `requirements/kv_connectors.txt`). Confirm the wheel's CUDA major (cu12 vs cu13) matches the VM to avoid the LMCache mis-pull bug. Reference: https://github.com/vllm-project/vllm/issues/30628
- **Use the UCX backend only. Do NOT use LIBFABRIC.** Confirmed broken on matched hardware (8x H200, vLLM 0.11.0 + NIXL 0.6.1): LIBFABRIC silently moves zero data and produces garbage output with no error. Reference: https://github.com/vllm-project/vllm/issues/27055
- **Pin a Phase-3-specific (vLLM, NIXL, UCX) triple validated live on the A3 Ultra VM.** The vLLM build must include the NIXL KV-transfer metrics log (PR #25388, https://github.com/vllm-project/vllm/pull/25388), which post-dates the 0.11.0 baseline; verify it is present in the chosen build, else cherry-pick. A single vLLM version serving both Phase-2 (L4) and Phase-3 (H200/RDMA) may not exist, so Phase 3 likely needs its own pinned triple.
- If LMCache is ever used (not the plan), pin and smoke-test the exact LMCache<->vLLM pair first; the matrix is fragile (vLLM 0.11.0 + LMCache 0.3.7 throws on `torch.frombuffer`; the upstream issue closed stale with no endorsed good combo). Reference: https://github.com/LMCache/LMCache/issues/1768

### 5.5 The RoCE-not-TCP validation smoke test (the hard pre-check)

This is the gate. The silent-failure risk is real and corroborated (issue #27055 above; a field report detected TCP fallback via ~8.5x TTFT degradation, 6,785ms TCP vs 796ms RDMA). Run the layered checks in order:

1. **Hardware floor, vLLM-free (perftest GPUDirect RDMA across the two nodes).** Run this BEFORE involving vLLM so a bad vLLM number can be attributed correctly. Server: `ib_write_bw -d mlx5_0 --use_cuda=0 --use_cuda_dmabuf`; client connects to it. Expect ~400 Gbps per ConnectX-7 NIC. Discover devices with `ibv_devinfo`, `show_gids`. Reference: https://docs.oracle.com/en/learn/gpudirect-rdma-ib-write-bw/index.html
   > verify-live: the `--use_cuda` flag form is build-specific, and whether GCP's MRDMA VF abstraction exposes the perftest GPUDirect dmabuf path is unconfirmed from docs; smoke-test it. On MRDMA VFs, physical port counters may not update, so use `ethtool -S` vPort counters for byte accounting.

2. **Transport identity (the primary RDMA proof).** With `UCX_LOG_LEVEL=info`, vLLM stderr prints a per-endpoint transport map at connection setup. Look for an `ep_cfg` line, e.g. `ep_cfg[2]: tag(... rc_mlx5/mlx5_0:1)`. Presence of `rc_mlx5` or `dc_mlx5` on the data path = RDMA; presence of `tcp` on the data path = FAIL (it fell back). `cuda_copy`/`gdr_copy` refers to local GPU<->host staging, not the inter-node wire. Reference: https://openucx.readthedocs.io/en/master/faq.html

3. **NIXL telemetry (volume/time/errors, independent of transport name).** Enable `NIXL_TELEMETRY_ENABLE=1` (plus `NIXL_TELEMETRY_DIR`, `NIXL_TELEMETRY_BUFFER_SIZE=4096`, `NIXL_TELEMETRY_RUN_INTERVAL=100ms`). Read with `python3 examples/python/telemetry_reader.py --telemetry_path /tmp/<agent_name>`. Confirm non-zero `agent_tx_bytes`/`agent_rx_bytes` and zero `agent_err_*`. Pair with check 2 for transport identity. Reference: https://github.com/ai-dynamo/nixl/blob/main/docs/telemetry.md

4. **vLLM KV Transfer metrics (throughput cross-check).** The vLLM log line `KV Transfer metrics: ... Throughput (MB/s)=...` (PR #25388, also Prometheus `vllm:nixl_xfer_time_seconds` / `vllm:nixl_bytes_transferred`) should read near single-NIC RoCE line-rate (hundreds of Gbps), not gVNIC/TCP rates. Cross-check bytes against `ethtool -S` MRDMA vPort counters.

5. **Correctness gate (catches the silent-garbage mode).** Under T=0 greedy, confirm output with NIXL KV transfer is **token-for-token identical** to a single-node non-transferred run. This is the losslessness check that caught the LIBFABRIC corruption; it must pass on UCX too.

**Then:** wire NIXL telemetry fields (`agent_tx_bytes`/`agent_rx_bytes`, `agent_xfer_time`) and the vLLM KV-transfer metrics into the cage-stats token-source breakdown so a Phase-3 run reports KV-bytes-transferred-over-RDMA and transfer-time per query, distinct from locally-computed prefix tokens. Confirm telemetry is REAL (no `CAGE_TELEMETRY_MOCK`) per the standing infra gate. For the production run, drop `UCX_LOG_LEVEL` back to default (info is verbose), keep `NIXL_TELEMETRY_ENABLE=1`.

---

## 6. Cost model (simplified)

Every figure is verify-before-provisioning; the console is authoritative for the exact RDMA-capable zone. The earlier third-party node prices (~US$12/hr on-demand, ~US$3.63/hr Spot) were corrected during verification for failing GCP's own documented H200 rate. Assumptions: 2x a3-ultragpu-8g (16 H200), ~5h including setup, GCS-only, same zone, per-second billing, 15-25% buffer.

| Line item | Plan A (3x A100, gVNIC/TCP) | Plan B, DWS Flex-start | Plan B, reserved/on-demand-equivalent |
|---|---|---|---|
| GPU nodes | ~US$50 total | 2 x ~US$29.80/hr x 5h = **~US$298** (verify-live) | 2 x ~US$60-90/hr x 5h = **~US$600-900** (verify-live) |
| Control plane (e2-standard-2 router, no Slurm) | included | ~US$2-5 (verify-live) | ~US$2-5 (verify-live) |
| Storage (GCS) | negligible | ~US$1 (verify-live) | ~US$1 (verify-live) |
| Egress (same-zone) | US$0 | US$0 | US$0 |
| + 15-25% buffer | (already rough) | **~US$345-450** | **~US$700-1,130** |
| **Realistic total** | **~US$50** | **~US$350-450** (GPU-only floor ~US$298) | **~US$700-1,100** |

- **Flex-start is the committed path:** non-preemptible during the granted window (Spot can be preempted mid-run and invalidate the measurement), billed near the on-demand rate, 7-day max lease. A3 Ultra has **no plain on-demand** ("A3 Ultra and A4 instances don't support on-demand instances"), so reserved/Flex is the floor.
- **Delta over Plan A:** roughly +US$300-400 on Flex-start; +US$650-1,050 if forced onto reserved capacity. Plan B is ~7x-20x Plan A, structurally, because real RoCE is sold only in 8-GPU nodes and a cross-node measurement needs two of them.
- **Do not authorize the reserved path** unless Flex-start capacity is unobtainable.

> verify-live: aggregators disagree several-fold on the on-demand-equivalent node price; assert no single figure without the console. Confirm the live a3-ultragpu-8g rate and the ~3,600 Gbps / ~3,200 Gbps bandwidth split in the target zone.

---

## 7. Feasibility gates and pre-gate checklist

The blocker is capacity/quota, not money or technical capability. Clear both gates BEFORE any spend.

### Gate 1: Capacity / quota (the dominant risk)

- [ ] Pick a single `ZONE` that offers a3-ultragpu-8g capacity AND the `ZONE-vpc-roce` profile (verify with `gcloud compute network-profiles list --filter="name~roce"` and live A3 Ultra availability).
- [ ] Request preemptible H200 quota >= 16 in `REGION` (confirm the exact metric string on the live Quotas page).
- [ ] Confirm Flex-start grant lead time fits the timeline. Capacity-wait has no documented max; this is what most likely kills Plan B on a non-enterprise PUC project.
- [ ] Decide whether `--max-distance=1` compact placement is worth the (possibly worse) capacity-grant time, or relax to `--max-distance=2` / same-zone.

### Gate 2: RoCE-backend validation (the technical gate)

- [ ] perftest GPUDirect RDMA floor passes (~400 Gbps/NIC) before vLLM.
- [ ] `UCX_LOG_LEVEL=info` shows `rc_mlx5`/`dc_mlx5` on the data path, no `tcp`.
- [ ] NIXL telemetry shows non-zero tx/rx bytes, zero errors.
- [ ] vLLM KV Transfer metrics throughput near RoCE line-rate, not gVNIC/TCP.
- [ ] T=0 token-for-token correctness vs single-node run passes (catches silent garbage).
- [ ] cage-stats reads REAL telemetry (no `CAGE_TELEMETRY_MOCK`).

### Abort-to-Plan-A condition

**If Gate 1 does not clear within the timeline, OR Gate 2 shows TCP fallback / fails the correctness check, abort to Plan A** (3x A100, gVNIC/TCP, ~US$50) and frame the transport result as analytic/TCP, stating honestly that real RDMA was out of scope for the budget. Do not pour spend into a config that cannot substantiate the RDMA claim.

---

## 8. Go / no-go

**Conditional go**, contingent on BOTH:

1. **The real-RDMA contribution is load-bearing.** If the dissertation's central claim is the analytic/TCP transport-cost model, Plan A suffices and Plan B is wasted money (Plan A's A100 has no GPUDirect-RDMA and can only measure TCP/gVNIC). Go only if the claim is literally "measured cross-node KV transfer over RDMA," which Plan A cannot make.
2. **The capacity gate clears** (Section 7, Gate 1). This gate, not the money, most likely kills Plan B.

If both hold: provision minimally per Section 4 (2x a3-ultragpu-8g via DWS Flex-start, GCS-only, no Slurm, no Parallelstore, same zone, compact placement), validate RoCE per Section 5.5, run once cleanly in one Flex-start window, then tear down to $0 and report GPU wall-clock cost (~US$298 GPU-only, ~US$350-450 buffered). If either gate fails, no-go: abort to Plan A.

---

## 9. Publishable angle (the transport-scaling ladder, kept)

The strongest single angle is the **RDMA vs TCPX vs TCPXO vs gVNIC transport-scaling ladder**: A100/gVNIC (Plan A) -> H100/TCPX -> H100/TCPXO -> H200/RoCE-RDMA (Plan B). Measure KV-transfer latency and effective bandwidth across transports in one clean, falsifiable figure. The "how close does offloaded TCPXO get to real RoCE" point is the most interesting result. Venue: dissertation HPC chapter, a systems/serving workshop paper, or a strong technical blog post. Supporting angles: measured-vs-analytic KV-transfer cost; sharded-KV serving cost-per-query under real RDMA vs single-node; and a reproducibility/cost note on what one real-RDMA KV-transfer experiment on GCP actually takes.

**Caveat for all angles:** strength depends entirely on Gate 2. If transfers silently fall back to TCP, every angle collapses into "we measured TCP again." Validate the RoCE backend first.

---

## 10. Verify-before-provisioning carry-over list

1. **Live node pricing** for a3-ultragpu-8g in the exact RDMA-capable target zone (aggregators disagree several-fold; the console is authoritative; on-demand is unavailable for this family).
2. **Exact preemptible H200 quota metric string** (assumed `PREEMPTIBLE_NVIDIA_H200_GPUS`) and current Flex-start capacity-wait behavior in `REGION`. The gate.
3. **Which zones expose both a3-ultragpu-8g capacity AND `ZONE-vpc-roce`** in mid-2026 (run `gcloud compute network-profiles list`).
4. **Whether `--max-distance=1` + FLEX_START** materially worsens 2-node capacity-grant time (resolve empirically; relax to 2 / same-zone if needed).
5. **Exact mlx5 device names + RoCE-v2 GID index** on the live a3-ultragpu-8g image (read via `ibv_devinfo`/`show_gids`; cannot be hardcoded).
6. **The Phase-3 (vLLM, NIXL, UCX) triple** validated live: confirm the vLLM build includes KV-transfer metrics PR #25388 and that the `nixl` wheel CUDA major matches the VM.
7. **perftest `--use_cuda` form** on the provisioned image and whether GCP's MRDMA VF abstraction exposes the GPUDirect dmabuf path; use `ethtool -S` vPort counters (physical port counters may not update).
8. **The public image family** for A3 Ultra with CX-7 RDMA + gIB NCCL plugin out-of-the-box vs needing a gIB container.
9. **Whether CAGE's prefix-aware router** maps onto vLLM's NIXL `--prefiller-hosts`/`--decoder-hosts` proxy model or needs a custom driver for sharded-context transfer.
10. **bandwidth split** (~3,600 Gbps total vs ~3,200 Gbps RoCE) on the live accelerator-optimized-machines table; **same-zone topology** so no cross-zone/internet egress applies.
