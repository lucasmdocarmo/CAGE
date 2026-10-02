# vLLM Compatibility & Version Pinning (deploy‑critical)

> **Why this file exists.** Every CAGE deploy path currently pulls `vllm/...:latest`, so a
> future deploy can silently pull a vLLM release that renamed or removed a flag CAGE depends
> on. We already hit one such change (`--speculative-model` was consolidated into
> `--speculative-config`). This spec pins a known‑good version and records the exact
> features/flags CAGE relies on, so deploys are reproducible and breakages are caught at a
> gate instead of mid‑experiment.
>
> Researched & current as of 2026‑06 (vLLM ~v0.11.x, V1 engine). Verify against the pinned
> tag before each phase.

---

## 0. PIN BUMP 2026-07-26: v0.19.1 is the Phase-3 pin (v0.11.0 frozen as the Phase-2 anchor)

**Decision.** The project pin moved `v0.11.0 → v0.19.1` (latest PyPI release, 2026-04-18) for all
FUTURE runs (Phase 3). Already updated: `scripts/gcp/setup_gpu_cloud.sh` (default
`VLLM_VERSION`), `scripts/2_serving/deploy_cluster.sh` (image tag), `scripts/3_run/cloud_run.sh`
(header). **v0.11.0 stays the frozen anchor for every published Phase-2 number** — never re-run
Phase-2 cells on a newer pin and mix tables; `export VLLM_VERSION=0.11.0` reproduces that era.

**Known cross-pin comparability break:** v0.19.0 enabled the **async scheduler by default**
(scheduling/execution overlap), which shifts TTFT distributions. State the scheduler regime in the
methodology; never compare TTFT across pins.

**Migration gate (re-run §3 under v0.19.1 at the next GPU preflight — every row below was a
0.11-era fact until re-verified):**
- [ ] `--kv-transfer-config` / NixlConnector schema (Phase-3 load-bearing path)
- [ ] `--speculative-config` schema; **EAGLE-3 early-EOS re-test** (0.11 bug forced quality
      exclusion; 0.12+ EAGLE fixes may allow un-excluding — re-decide with data)
- [ ] ngram speculative: 0.18 added a GPU NGram path — record which implementation runs
- [ ] `/reset_prefix_cache` dev endpoint under `VLLM_SERVER_DEV_MODE=1` (cold-start-per-trial)
- [ ] `--enable-prompt-tokens-details` → `usage.prompt_tokens_details.cached_tokens` (the
      locality-gradient column)
- [ ] cage-stats `/metrics` name parity (Prometheus metric names drift between versions)
- [ ] dependency dances: openai>=2 reconcile, lmcache↔vLLM pairing, `transformers<5` pin —
      all three were 0.11-era workarounds; re-verify or remove
- [ ] torch/CUDA floor of the v0.19.1 wheel vs the DLVM image
- [ ] §2 flag matrix + §4 FP8×prefix-caching gate re-validated and this file's tables updated

---

## 1. The rule: pin, don't chase `latest`

vLLM's own docs recommend pinning a versioned tag for reproducible deployments. CAGE should
deploy a **single pinned version** everywhere, exposed as one knob:

| Where | Current (risk) | Pin to |
|---|---|---|
| `scripts/lib/_serving_config.sh` (the serving-uniformity source all launchers must source) | pilot pin `v0.11.0`-era | **`v0.19.1` (charter pin; see §7 matrix — SGLang/LMDeploy pins fixed at preflight)** |
| ~~`vllm_image` in the pre-2026-08 terraform~~ | — | ROW RETIRED 2026-08-02: that old module tree was replaced; the current stack (`terraform/gcp/` since the provider split) provisions bare GPU hosts (image family per `sessions/*.tfvars`) and engine pins live in the serving config, not IaC |
| `docker/docker-compose.gpu.yml` | `:latest` | `:v0.11.0` |
| `scripts/2_serving/deploy_cluster.sh` | `:latest` | `:v0.11.0` |
| `k8s/vllm-replica.yaml` (×3) | `vllm/vllm-openai:latest` | `vllm/vllm-openai:v0.11.0` ✅ pinned |
| `docker/docker-compose.yml` (CPU ARM) | `public.ecr.aws/q9t5s3a7/vllm-arm64-cpu-release-repo:latest` | a dated tag from that repo (different registry — pin separately) |

> `k8s/router.yaml` uses `cage-router:latest`, the project's **own** locally-built image (not an
> external dependency). It needs a build-versioning scheme (git SHA or release tag), not a vLLM
> pin; tracked separately.

> The CPU ARM image is a **separate** community build (AWS ECR Public), not `vllm/vllm-openai`.
> It has its own tags; pin it to a dated tag you have validated locally, not `latest`.

**Bumping versions is deliberate, not automatic:** to move to a newer vLLM, change the one
pin, re‑run the §3 gate, and record the result here.

## 2. Feature/flag matrix CAGE depends on (current vLLM API)

| CAGE feature | Flag / mechanism (current) | Status on V1 (~v0.11) | Notes |
|---|---|---|---|
| Prefix caching (`prefix_cache`, hybrid, distributed) | `--enable-prefix-caching` | ✅ valid (on by default in recent V1; flag still accepted) | the core reuse signal |
| Prompt‑cache telemetry (H1) | `--enable-prompt-tokens-details` → `usage.prompt_tokens_details.cached_tokens` | ✅ valid | exact field CAGE reads ([vllm_adapter.py:78‑80](../src/inference/vllm_adapter.py)) |
| KV compression (`compressed_cag`) | `--kv-cache-dtype fp8` (`fp8_e4m3` / `fp8_e5m2`) | ✅ valid | ⚠️ **see §4 — FP8 × prefix‑caching** |
| Speculative decoding (`speculative`) | `--speculative-config '{"method":"ngram"\|"eagle"\|...,"num_speculative_tokens":N,"model":"..."}'` | ✅ valid | **replaces the deprecated `--speculative-model`** |
| Distributed / TP | `--tensor-parallel-size N` | ✅ valid | Phase 3 |
| Per-request logging OFF (disk-fill guard on all three vLLM launch paths) | no flag by default; `--enable-log-requests` only when `VLLM_DISABLE_LOG_REQUESTS=0` | ✅ 0.19.1 has ONLY the enable form (`cli_args.py` at v0.19.1) | `--disable-log-requests` is REJECTED by 0.19.1 ("unrecognized arguments", live L40S 2026-09-27, backlog W29): the server exits at once and the launcher waits its whole readiness timeout. Never pass it |

Sources: [vLLM speculative decoding docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/n_gram/), [vLLM quantized KV cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/), [Docker usage](https://docs.vllm.ai/en/stable/deployment/docker/).

## 3. Compatibility gate (run after any pin/bump, before a phase)

On the pinned image, confirm each flag is still accepted and behaves:
1. `--enable-prefix-caching` + `--enable-prompt-tokens-details` → a repeated‑prefix request
   returns non‑zero `usage.prompt_tokens_details.cached_tokens`.
2. `--kv-cache-dtype fp8` launches **and** prefix caching still hits (§4).
3. `--speculative-config '{"method":"ngram","num_speculative_tokens":5}'` launches, and
   `/metrics` exposes `vllm:spec_decode_num_accepted_tokens_total`.
4. `--tensor-parallel-size 2` launches on a 2‑GPU node.

A failing gate = do not run that phase on that tag; fix the flag mapping here first.

## 4. ⚠️ FP8 KV cache × prefix caching — a real confound for `compressed_cag`

`compressed_cag` launches with `--kv-cache-dtype fp8`, and CAGE's CAG arms depend on prefix
caching. Historically these two were **incompatible** in vLLM (enabling FP8 KV disabled prefix
caching); recent releases make prefix caching dtype‑agnostic (hash‑based), and the state is
moving — see the vLLM blog [*The State of FP8 KV‑Cache and Attention Quantization*](https://vllm.ai/blog/2026-04-22-fp8-kvcache).

**Action:** on the pinned tag, verify that launching with `--kv-cache-dtype fp8 --enable-prefix-caching`
still produces non‑zero cached‑token hits. If FP8 silently turns prefix caching **off**, then
`compressed_cag` is not "CAG + compression" — it's "no‑reuse + compression," which **confounds
the entire compression‑axis comparison (RQ5/H4)**. Record the verified behaviour per version.

## 5. Speculative decoding — current API + metrics

- **Config (launch‑time):** `--speculative-config` JSON. Methods on V1: `ngram` (no draft
  model), `eagle`/`eagle3`/`medusa`/`mtp`/`draft_model` (need a model). EAGLE ≈ 0.8 acceptance,
  2.5–2.8× decode speedup. `manage_vllm_server.sh` already uses `VLLM_SPECULATIVE_CONFIG →
  --speculative-config` (current). `run_speculative_matrix.sh` now emits `--speculative-config` too; the old
  `--speculative-model` is deprecated and no longer used by any script.
- **Acceptance metrics (`/metrics`, Prometheus):**
  `acceptance = vllm:spec_decode_num_accepted_tokens_total / vllm:spec_decode_num_draft_tokens_total`
  (also `..._num_accepted_tokens_per_pos`, `..._num_drafts`). This is the signal CAGE must
  scrape — TTFT/TPOT alone do not characterise speculation.
- **Quality note:** speculative decoding is **output‑distribution‑preserving** (draft+verify),
  so it does **not** change faithfulness/grounding. Treat it as a serving‑throughput (Sys/TPOT)
  baseline, not part of the efficiency‑vs‑quality frontier.

Sources: [vLLM spec‑decode metrics](https://docs.vllm.ai/en/stable/api/vllm/v1/spec_decode/metrics/), [vLLM metrics design](https://docs.vllm.ai/en/latest/design/metrics/).

## 6. Status of the wiring (this change set)

- [x] Pin the vLLM image across deploy paths (§1).
- [x] `run_speculative_matrix.sh`: launch with `--speculative-config` (drop deprecated `--speculative-model`).
- [x] Scrape `/metrics` spec‑decode acceptance into telemetry (`scrape_spec_decode`).
- [x] Replace the "not wired" runner warning with the correct launch‑lever guidance.
- [x] `run_compression.sh`: 2×2 axis through the FP8 launch‑lever (`compressed_cag`) + LLMLingua (`compressed_rag`).
- [x] `check_fp8_prefix_cache.sh`: the FP8×prefix‑caching gate (§4), auto‑run by `run_compression.sh`.
- [x] `terraform vllm_extra_args` so cluster replicas can enable FP8/speculative.
- [ ] Validate on GPU (Phase 2) — speculative and FP8 are GPU‑meaningful; both are Phase‑2 runs.

---

## 7. Campaign engine × model matrix (4 × 4) — every cell VERIFY-LIVE at its session's preflight

> Charter bindings: PUBLICATION.md §7.6 (groups), P7 (LMDeploy pinning), D4 (HF-oracle
> exemption for V3), §5.2 (Qwen3-Next identity gate). **No cell below is "known good"
> until its gate has run on the provisioned node at the pinned version** — supported?/
> pin? are the plan; the gate result is the fact. Record pass/fail + actual versions
> into the run manifest and update this table per session.

Engine pins for the campaign:

| Engine | Pin | Pin status |
|---|---|---|
| vLLM | **0.19.1** (PyPI wheel / `vllm/vllm-openai:v0.19.1`) | PINNED (§0) — the §0 migration gate applies to every vLLM cell |
| SGLang | **0.5.10.post1** (PyPI, released 2026-04-08; own venv `sglang-env` created by `scripts/runpod/setup_runpod.sh` step 3c; pins torch 2.9.1 on the CUDA 12.8 runtime, the same line as the pod image and vLLM 0.19.1's torch 2.10.0; pinned 2026-09-26, pre-GO item 10; **sglang-env runs on CPython 3.12 (`SGLANG_PYTHON_VERSION`, ADR-0120, 2026-09-27): SGLang pulls `outlines_core==0.1.26`, which ships no cp313 wheel, and the image has no Rust to build it, while the cp312 wheel exists; served Qwen3-8B live on an L40S on 2026-09-27 after a source build**) | [VERIFY-LIVE at S0-3] incl. **deterministic-mode availability** (T=0 reproducible sampling) per model. Bump rule: every SGLang release since 2026-05 (0.5.11 and later) pins a CUDA 13 torch (2.11 or 2.13); a bump first needs the host driver read on the pod (`nvidia-smi`, setup step 0) to support CUDA 13, then the S0-3 gate again |
| LMDeploy | **0.17.0** (PyPI, released 2026-09-02; own venv `lmdeploy-env` with torch pinned to 2.10.0 by `LMDEPLOY_TORCH_VERSION`, since the package accepts torch 2.0 to 2.12.1 and pip would otherwise resolve a CUDA 13 build; **TurboMind backend only**; pinned 2026-09-26) | [VERIFY-LIVE at S0-3] gate: TurboMind is actually selected (not the silent PyTorch-engine fallback). **TurboMind weight formats are FP16/BF16, KV INT8, KV INT4 and W4A16 only (supported-models table read 2026-09-26; no FP8 weight path), so the S0 cross-engine rows serve the BF16 `Qwen/Qwen3-8B` on every engine and `Qwen/Qwen3-8B-FP8` runs on vLLM alone for gate (s)** |
| HF Transformers | TBD — pin `transformers` at session-A preflight (the old `transformers<5` was a 0.11-era workaround; re-verify) | [VERIFY-LIVE] |
| NIXL (vLLM NixlConnector transfer library, prefill/decode pair) | **0.9.0** (`nixl==0.9.0` + `nixl-cu12==0.9.0`, installed in the same pip call as vLLM by `scripts/runpod/setup_runpod.sh` step 2, `NIXL_VERSION`; ADR-0128, S0F-13, 2026-09-30). The newest release inside vLLM 0.19.1's own declared range (`requirements/kv_connectors.txt`: `>=0.7.1,<0.10.0`); its wheels ship no `nixl_ep`, so nothing is stripped. Not `nixl-cu13`: the pod runs CUDA 12.8 torch and the 0.9.0 dispatcher tries cu13 first when present. Each role gets its own `VLLM_NIXL_SIDE_CHANNEL_PORT` (5600 prefill, 5601 decode; `manage_vllm_pd.sh`), the handshake listener port both roles shared at S0 | [VERIFY-LIVE at Run-C-prime preflight]: 0.9.0 never ran on a pod (S0 proved nixl 1.5.0's core up to engine init; API names and agent-config keywords match). The launcher's import gate refuses a start when `nixl._api` / `nixl._bindings` do not import or a top-level `nixl_ep` is present but broken (the S0 layer-2 crash) |
| numba (vLLM 0.19.1 dependency; used only by the ngram speculative proposer, retired by the charter) | **0.67.0** (`requirements.txt`, exact; the version S0 ran on 2026-09-30). vLLM declares `numba==0.61.2` (numpy<2.3), which cannot coexist with the Tier-1 `numpy==2.5.1`; preflight gate (i) accepts the one resulting `pip check` line through `scripts/checks/pip_check_allowlist.txt` (ADR-0126, S0F-1) and fails on any other line, including the 0.68.0 drift | PINNED; no vLLM release through 0.30.0 accepts numpy 2.5.1 (all pin numba 0.65.0), so a vLLM bump does not clear the line |

The matrix (cell = supported? · pin · preflight gate):

| Engine \ Model | **Qwen3-14B** (A) | **Llama-3.3-70B** (B) | **Qwen3-Next-80B** (C) | **DeepSeek-V3-0324** (D) |
|---|---|---|---|---|
| **vLLM 0.19.1** | ✅ planned. Gate: §3 flag gate + prefix telemetry (`cached_tokens`) + thinking-mode pinned OFF | ✅ planned, TP=4. Gate: TP launch + §3 on the 4-GPU node | ⚠️ planned; hybrid-attention prefix cache is experimental. Gate: **§5.2 mandatory T=0 token-identity smoke** on every prefix-ON cell (fail → cell excluded, failure IS the result) + intra-node PD smoke | ⚠️ planned; FP8 671 GB, MLA, TP=8. Gates: **fp8-KV-on-MLA** [VERIFY-LIVE — buggy on some MLA models], MLA×TP launch, NixlConnector cross-node (§8) |
| **SGLang 0.5.10.post1** | ✅ planned. Gate: deterministic mode + radix-cache telemetry (per-request granularity [VERIFY-LIVE]) | ✅ planned, TP=4. Gate: as A + TP | ✅ planned (second engine for Group C). Gate: deterministic mode + prefix/radix reuse proven on hybrid attention; PD smoke → if it passes, SGLang joins the disaggregation rung | ✅ planned: **PURE TP for every registered V3 cell** (pin 2026-08-02). Gate: deterministic mode + MLA serving; PD smoke optional (only for the D rung if act-1 smoke passed) |
| **LMDeploy-TurboMind 0.17.0** | ✅ planned (P7). Gate: TurboMind actually selected + blocked-KV pressure counters (weakest documented telemetry: a failed cache-telemetry gate → serving-only or excluded cells) | ✅ planned, TP=4 (P7). Gate: as A + TP | ❌ **ABSENT BY POLICY (P7)**: not run, absence reported | ❌ **ABSENT BY POLICY (P7)**: not run, absence reported |
| **HF oracle** | ✅ planned. Gate: T=0 batch-1 output/logit match vs each server engine (P2/P3); sub-pressure F1 only | ✅ planned (batch-1 `device_map` across the 4-GPU box). Gate: as A | ✅ planned. Gate: as A | ❌ **EXEMPT (D4)** — 671B oracle infeasible; recorded exemption; substitute = cross-engine T=0 agreement vLLM↔SGLang |

Reading rule: ✅/⚠️/❌ is the *plan*; ⚠️ means the gate is load-bearing (a plausible
failure mode is on record). A ❌-by-policy cell is a reported ecosystem finding, never a
silent hole.

---

## 8. C/D session act-2 RDMA preflight — [RECONSTRUCTED 2026-08-02 — revalidate live at act-2 preflight]

> Reconstructed from the salvage record of the Plan-B HPC study (original doc archived
> offline; pointer: MyDocs/BACKLOG.md "C/D session prep"). Every number and knob below
> is a *recorded prior*, not a verified fact for the act-2 nodes — the whole section
> re-runs live before any RDMA-rung cell. Context: two `a3-ultragpu-8g` (8× H200)
> nodes, RoCE v2 fabric on the MRDMA VFs, vLLM NixlConnector
> (`--kv-transfer-config`) prefill/decode split.

### 8.1 The failure this section exists to catch

NIXL/UCX will **silently fall back to TCP** when the RDMA path is unusable (wrong GID,
tcp left in `UCX_TLS`, unpinned NICs, driver mismatch). The run then *works* and
produces plausible-looking numbers that measure the wrong transport.
**TCP-fallback signature: ~8.5× TTFT inflation** on transfer-bound requests vs the
RDMA envelope. Any TTFT jump of that magnitude on the RDMA rung = assume fallback,
stop, re-run the five-check smoke.

### 8.2 Five-check RoCE-not-TCP smoke (all five must pass, in order)

1. **Devices present:** `ibv_devinfo` on both nodes lists the MRDMA VFs (one per rail),
   each `PORT_ACTIVE`. Missing/DOWN devices → wrong image/driver or non-RDMA VPC.
2. **GID sanity:** `show_gids` (or `ibv_devinfo -v`) per device — identify the RoCE v2
   GID index for the fabric subnet and pin it (`UCX_IB_GID_INDEX`). Never assume index
   0/3; discover it live.
3. **Raw fabric bandwidth:** `ib_write_bw --use_cuda=<gpu>` node-to-node on each rail
   hits RoCE-class line rate (~400 Gbps/NIC ballpark on A3 Ultra [revalidate]). This
   proves GPUDirect RDMA end-to-end *below* NIXL.
4. **Counter attribution:** during check 3 (and again during the vLLM smoke),
   `ethtool -S <vf>` vPort counters on the MRDMA VFs increment by ~the transferred
   bytes while TCP socket byte counters stay flat. Bytes on the wrong counter = TCP.
5. **In-band engine smoke:** a 2-node vLLM P/D transfer with UCX debug on
   (`UCX_LOG_LEVEL=info`) — the selected transports must exclude `tcp`
   (expect `rc`/`dc` + `cuda_copy`/GDR paths), vLLM `nixl_*` metrics increment, and
   TTFT sits inside the RDMA envelope (no 8.5×-class inflation).

### 8.3 UCX hardening (set explicitly; do not trust auto-selection)

- `UCX_TLS` — explicit allowlist **excluding `tcp`** (e.g. `rc,dc,cuda_copy,cuda_ipc`
  [revalidate exact set]) so fallback becomes a loud failure instead of a silent
  downgrade.
- `UCX_NET_DEVICES` — pinned to the RDMA NICs (the MRDMA VF device:port list from
  check 1), never left to auto (which can pick the frontend gVNIC).
- `UCX_IB_GID_INDEX` — the RoCE v2 GID index discovered in check 2.
- Discovery is live, per node, every provisioning: `ibv_devinfo` + `show_gids` output
  goes into the act-2 manifest.

### 8.4 Version-triple pin rule

The transfer stack is pinned as a **triple — (vLLM, NIXL wheel, UCX)** — recorded
together in the manifest; bump any element → re-run this whole section.

Current triple (ADR-0128, 2026-09-30): vLLM 0.19.1; NIXL 0.9.0 (`nixl` + `nixl-cu12`,
section 7 row); UCX as bundled by that wheel. The UCX version is read by hand from the
role log at the Run-C-prime preflight and written into the act manifest; no launcher
field records it (review 2026-10-01). The launcher records every `UCX_*` variable the
operator sets in the per-role serving-config capture (`ucx_env`) and exports none
itself: the intra-node TCP rung runs on UCX's own transport selection, while the RDMA
rung sets the 8.3 allowlist by hand. The S0 `UCX ... IB` line was a warning printed by
the decode that reached ready, not a failure.

- **UCX backend, NOT LIBFABRIC** (vLLM issue #27055: the LIBFABRIC path is broken/
  unsupported for NixlConnector) [revalidate on 0.19.1].
- The **NIXL wheel's CUDA major must match** the node's CUDA/driver major, or transfers
  fail (sometimes silently into fallback).
- The `nixl_*` Prometheus metrics: read from the 0.19.1 connector source on 2026-10-01
  (ADR-0133, S0F-22), the connector registers `vllm:nixl_xfer_time_seconds`,
  `vllm:nixl_post_time_seconds`, `vllm:nixl_bytes_transferred` and
  `vllm:nixl_num_descriptors` (histograms), `vllm:nixl_num_failed_transfers`,
  `vllm:nixl_num_failed_notifications` and `vllm:nixl_num_kv_expired_reqs` (counters),
  labels `model_name` and `engine`, on every instance whose config names the connector;
  bytes, times and descriptors move on the DECODE (the reader) only, expiries on the
  prefill, and only when `nixl._api.nixl_agent_config` imports (no telemetry config, no
  stats; the launcher's import gate checks it). Still [VERIFY-LIVE]: `curl :8200/metrics |
  grep nixl_` on the provisioned node (RC-5, RC-13).
- The ticket (ADR-0133): the prefill writes `kv_transfer_params` on its NON-stream
  response only when the request carried `kv_transfer_params.do_remote_decode` and
  finished by its length cap; the eight keys are `do_remote_prefill`, `do_remote_decode`,
  `remote_block_ids` (nested per KV group), `remote_engine_id`, `remote_request_id`,
  `remote_host`, `remote_port`, `tp_size`, and there is no `source` key. The proxy's
  prefill leg sends vLLM's own request shape plus `ignore_eos`, drops `stream_options`
  (refused with `stream=false`), refuses a missing ticket before the decode, and relays
  the ticket verbatim in the `x-kv-transfer-params` response header; the runner's pd gate
  checks that shape on every served row (rows with `error` set are skipped). `remote_host` is `localhost` unless
  `VLLM_NIXL_SIDE_CHANNEL_HOST` is set on the prefill: the cross-node rung needs it.
- The pull (ADR-0134): the decode records one `vllm:nixl_bytes_transferred` observation
  per transfer handle (one per worker per ticketed request, `nixl_connector.py:2398-2402`),
  so for homogeneous TP `t` the `_count` rises by `t` per served request; the pulled
  prompt tokens land in `vllm:prompt_tokens_by_source_total{source="external_kv_transfer"}`
  and the decode samples the first token itself (`local_compute` and
  `vllm:prompt_tokens_recomputed_total` rise by 1 per request, `scheduler.py:2067-2074`).
  The decode's local prefix hit is capped at N-1 tokens (`kv_cache_manager.py:195-201`),
  so a ticketed request always reads at least one block: that is the basis of the
  runner's `transfer_count >= n_served_rows` clause. `metrics.json["pd_transfer"]`
  records the two-scrape deltas of these series per window (`start`, `end`, `delta`,
  `prefill_delta`, `n_served_rows`, `identities`, `verified`, `reasons`). Prometheus
  exposition spellings: histogram `_sum`/`_count`, counter `_total`; the `_created` and
  `_bucket` siblings are separate series. [VERIFY-LIVE] at RC-13: the `external_kv_transfer`
  attribution and the count-per-TP identity on the provisioned node.
- The reset (ADR-0137, S0F-27): `POST /reset_prefix_cache` takes two bool query
  parameters, `reset_running_requests` and `reset_external`, both default false
  (`entrypoints/serve/cache/api_router.py:21-26`), and answers 200 without reading the
  engine's result. With `reset_running_requests=true` the scheduler preempts the running
  requests and raises when blocks are still held (`v1/core/sched/scheduler.py:1895-1902`);
  the engine survives and the server answers 500. The reset succeeds only when no block
  besides the null block is in use, the same condition under which
  `vllm:kv_cache_usage_perc` reads exactly 0.0. A finished prefill request with a ticket
  holds its blocks outside the running queue until the decode's notification is
  processed on a prefill engine step, or for `VLLM_NIXL_ABORT_REQUEST_TIMEOUT` (480 s);
  `vllm:num_requests_running` reads 0 meanwhile. `CompletionRequest.model` is optional
  (`completion/protocol.py:45`), which the proxy's wake request relies on.
  [VERIFY-LIVE] at RC-12: the 500 on a held prefill and the release after one wake step.
- Stream endings (ADR-0135, S0F-25): the finish reasons are `stop`, `length`, `abort`,
  `error`, `repetition` (`v1/engine/__init__.py:29`; `repetition` only when the request
  sets `repetition_detection`, which CAGE never sends). `error` is turned into an in-band
  `data: {"error": {"message", "type", "param", "code"}}` event, except on a request that
  failed before its first token (a failed KV load): that stream ends with the usage chunk
  and `[DONE]` and carries no `finish_reason`. The adapter serves `stop` and `length`
  only.

### 8.5 Validation instruments (the act-2 measurement toolkit)

| Instrument | What it proves |
|---|---|
| `ib_write_bw --use_cuda` (perftest) | raw GPUDirect RDMA bandwidth per rail, below the serving stack |
| `ethtool -S` vPort counters on the MRDMA VFs | bytes actually moved over RDMA (attribution, not inference) |
| vLLM `nixl_*` metrics | transfer count/bytes/latency as the engine sees them |
| **T=0 token-identity check after transfer** | the transferred KV produced *identical tokens* to the untransferred baseline — transfer is lossless, not just fast |

Cross-layer calibration (perftest → NIXL → vLLM metrics → cage-stats) must agree
within the pre-registered tolerance; disagreement is itself a finding to chase before
running cells.
