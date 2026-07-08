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

## 1. The rule: pin, don't chase `latest`

vLLM's own docs recommend pinning a versioned tag for reproducible deployments. CAGE should
deploy a **single pinned version** everywhere, exposed as one knob:

| Where | Current (risk) | Pin to |
|---|---|---|
| `terraform/gcp/main.tf` (`vllm_image` default) | `vllm/vllm-openai:latest` | `vllm/vllm-openai:v0.11.0` |
| `terraform/gcp/terraform.tfvars.example` | `:latest` | `:v0.11.0` |
| `docker/docker-compose.gpu.yml` | `:latest` | `:v0.11.0` |
| `scripts/deploy_cluster.sh` | `:latest` | `:v0.11.0` |
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
  --speculative-config` (current). `run_phase5.sh` now emits `--speculative-config` too; the old
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
- [x] `run_phase5.sh`: launch with `--speculative-config` (drop deprecated `--speculative-model`).
- [x] Scrape `/metrics` spec‑decode acceptance into telemetry (`scrape_spec_decode`).
- [x] Replace the "not wired" runner warning with the correct launch‑lever guidance.
- [x] `run_compression.sh`: 2×2 axis through the FP8 launch‑lever (`compressed_cag`) + LLMLingua (`compressed_rag`).
- [x] `check_fp8_prefix_cache.sh`: the FP8×prefix‑caching gate (§4), auto‑run by `run_compression.sh`.
- [x] `terraform vllm_extra_args` so cluster replicas can enable FP8/speculative.
- [ ] Validate on GPU (Phase 2) — speculative and FP8 are GPU‑meaningful; both are Phase‑2 runs.
