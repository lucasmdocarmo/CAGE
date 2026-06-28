# CAGE Re-Run Experimental Design (Phase 2 + Phase 3)

**Created:** 2026-06-28 · **Status:** DESIGN (locked items + one open decision); NO run until explicit user go-ahead.

> Authoritative design for the Phase-2 re-run and Phase-3 run: queries, trials, temperature,
> models, and requisites, with the rationale and citations behind each choice. Companion:
> [`KNOWLEDGE_BASE.md`](KNOWLEDGE_BASE.md), [`PHASE3_PLAN.md`](PHASE3_PLAN.md), `docs/PHASE3_MODELS.md`.

---

## 1. Dataset and queries

- **300 queries per baseline** (locked). SQuAD v2 for Phase 2 continuity; add a RAG-favorable
  dataset (NQ / MuSiQue / a RULER or long-context slice) in Phase 3 so RAG gets a fair test
  (on SQuAD the gold passage hands the win to CAG).
- **Justification vs cited norms:** 300 sits above the metric-validation sets used in the
  literature (RAGAS 50 [arXiv:2309.15217], ARES ~150 [arXiv:2311.09476], CacheBlend 150-200
  per dataset [arXiv:2405.16444]) and just under RULER's canonical 500-per-condition
  [arXiv:2404.06654]. It is not under-powered for the per-query Wilcoxon + Holm + bootstrap layer.

## 2. Trials and determinism (the T=0 main protocol)

- **At temperature 0, generation is deterministic** for a fixed batch, so re-running the same
  query gives the same answer. Therefore split the axes:
  - **Quality: 1 pass.** Deterministic, so 1 pass == N passes. Run it **single-stream** (or with
    vLLM batch-invariant kernels) so it is genuinely reproducible.
  - **Serving: 3 trials.** Latency/throughput are genuinely noisy (scheduler jitter, KV state,
    thermals); 3 trials give serving-latency confidence intervals. Run concurrently (also the
    memory-pressure lever).
- **Determinism caveat (why "1 pass" needs care):** GPU inference at T=0 is only bitwise
  deterministic with a fixed batch. Under dynamic batching, floating-point accumulation order
  changes with batch composition, so a small fraction of outputs can differ run-to-run. Evidence:
  Thinking Machines, "Defeating Nondeterminism in LLM Inference"
  (https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/); vLLM Batch
  Invariance (https://docs.vllm.ai/en/latest/features/batch_invariance/), which fixes it at a
  ~1.6-2x throughput cost. Mitigation: measure quality single-stream; measure serving concurrent.
- **Record `temperature` (and decode params) in `metrics.json`.** Phase 1 did not, so its value
  had to be reconstructed from config + git. Fix this for provenance.

## 3. Temperature policy

- **Main protocol = T 0.0 (greedy)** for both Phase 2 and Phase 3. Rationale:
  1. Reproducibility and clean attribution: the generated text is held constant, so any metric
     delta is attributable to the serving strategy, not to sampling randomness.
  2. The losslessness claim is only checkable at T=0: speculative decoding and FP8/compression are
     output-preserving, so at greedy you can do a row-by-row identity check (spec == non-spec).
     vLLM states greedy speculative decoding matches greedy without it
     (https://docs.vllm.ai/en/latest/features/speculative_decoding/); EAGLE reports its best,
     stable speedup at T=0 [arXiv:2401.15077].
- **Phase-1 vs Phase-2/3 protocol change (state this in the dissertation):** Phase 1 ran at
  **T 0.7** (stochastic; Qwen3-4B; not logged, confirmed via config + git Initial Commit). Phases
  2-3 use **T 0.0**. This is a deliberate determinism change; do not read cross-phase deltas as a
  temperature effect (and Phase 1 also lacked the chain-of-thought-suppression fix, which partly
  explains its lower faithfulness 0.570).
- **Do NOT run a temperature sweep as "trials at different temperatures."** Trials are repeats you
  average for CIs; you cannot average across temperatures, so 1 run per temperature gives no CIs,
  leaves temperature an unreplicated factor (confounded with run noise), characterizes T>0 quality
  from a single noisy sample, and cannot demonstrate losslessness (which needs T=0).
- **Temperature sensitivity, done right (recommended, OPEN decision):** a SEPARATE, clearly
  labeled sub-study. Temperature as a factor T in {0, 0.5, 1.0}, on a SUBSET (1 model; 2-3
  representative baselines such as no_cache / prefix_cache / rag), with **k >= 3 samples per query
  at T>0** so quality at T>0 is estimated, not a single draw. Reported separately, never averaged
  into the main T=0 matrix. This answers the "production uses sampling" question without polluting
  the main results or the losslessness claim. (Reasoning-task leaderboards do use T>0 with multiple
  samples / self-consistency; that pattern is the basis for the k>=3 samples here.)

## 4. Models (citation-verified, 2026-06-28)

Every model carries a verified citation. Speculative coverage spans all three viable vLLM methods
(ngram + EAGLE-3 via the Qwen family; MTP via MiMo / GLM-Air) across dense / MoE / MLA architectures.

**Phase 2 (single L4, 24 GB):**
| Model | Fits L4 | Speculative | Citation | Note |
|---|---|---|---|---|
| Qwen3-8B | yes | ngram + EAGLE-3 | arXiv:2505.09388 | re-run anchor; no MTP |
| MiMo-7B-RL | yes | MTP (native) + ngram | arXiv:2505.07608 | unlocks MTP; LIVE-VALIDATE on stock vLLM 0.11.0 |

Optional Phase-2 adds (verified, L4-fit): Llama-3.1-8B-Instruct [arXiv:2407.21783], Mistral-7B-Instruct-v0.3 [arXiv:2310.06825], Qwen2.5-7B-Instruct [arXiv:2412.15115].

**Phase 3 (multi-A100):**
| Model | Hardware | Speculative | Citation | Note |
|---|---|---|---|---|
| Qwen3-32B | A100-80 or 2xA100-40 (TP) | EAGLE-3 + ngram | arXiv:2505.09388 | dense scale-up |
| GLM-4.5-Air (106B MoE) | 3-4x A100-80 | MTP (native) | arXiv:2508.06471 | NOT full GLM-4.5 (355B, ~710 GB, 16x H100 = avoid) |

High-value optional Phase-3 add: DeepSeek-V2-Lite [arXiv:2405.04434] (MoE + MLA) for the
compression axis: architectural MLA KV-compression vs the post-hoc FP8 KV arm. Other verified
P3 options: Llama-3.3-70B [arXiv:2407.21783], Qwen3-14B [arXiv:2505.09388], Mixtral-8x7B [arXiv:2401.04088].
Out of scope: Medusa (superseded by EAGLE-3), SAM-decoding / R-SD (not in vLLM 0.11.0).

## 5. Requisites (per run)

- **Hardware:** Phase 2 = single GCP L4 (24 GB). Phase 3 = multi-A100 per the table above.
- **vLLM 0.11.0** pinned; generation `temperature=0.0` + `stop=["\n"]` + concise system prompt
  (Qwen3 CoT suppression); `--enforce-eager` optional for fast startup.
- **Memory pressure:** add a concurrency variant and a longer-context variant (Phase-2 KV was only
  ~2% utilized single-stream, which is why caching/compression effects looked modest).
- **Compression validity:** `llmlingua` installed + run `compressed_rag` with `CAGE_REQUIRE_COMPRESSION=1`
  (strict) so it cannot silently no-op; FP8 x prefix-cache gate must pass for `compressed_cag`.
- **Telemetry/logs:** `--vllm-telemetry`; the log-preservation suite (`collect_logs.sh`,
  `log_sync_daemon.sh`, `gcp_shutdown_hook.sh`) active; teardown ONLY via the fail-closed
  `teardown_vm.sh` (verifies logs reached GCS before deleting).
- **Process discipline:** validate all infra components before every run; report cost + time
  estimate before provisioning; **never provision or start a run without explicit user go-ahead.**
- **Cost estimate (at ~$0.73/hr L4, ~$11-12/hr A100 cluster):** Phase 2 (both models, batched)
  ~$10-20; Phase 3 ~$100-250 depending on GLM-Air scope. Validate cheaply on L4 before the A100 run.

## 6. Open decision

The temperature-sensitivity design (Section 3): confirm the recommended approach (main matrix at
T=0 + a separate sub-study sweeping T in {0,0.5,1.0} on a subset with k>=3 samples per query),
versus any alternative. Everything else in this document is locked pending the run go-ahead.

## 7. Key references (shared with the dissertation)

These are the sources behind the design above, matching the entries now in the dissertation
bibliography (`my-article/CAGE___Dissertation_Mestrado_2026/Main.bib`), so this doc and the
manuscript cite the same evidence:

- **Greedy-decoding determinism (and its numerical caveat):** Yuan et al. 2025, "Understanding and
  Mitigating Numerical Sources of Nondeterminism in LLM Inference," arXiv:2506.09501
  [`yuan2025nondeterminism`]. Establishes greedy/T=0 as the deterministic mode (argmax at fixed
  batch) and shows the only thing breaking it is floating-point non-associativity under dynamic batching.
- **Temperature values {0, 0.5, 1.0} and the 0-1 sweep range:** Renze and Guven 2024, "The Effect of
  Sampling Temperature on Problem Solving in Large Language Models," arXiv:2402.05201
  [`renze2024temperature`]. Sweeps T from 0.0 to 1.0 and finds no statistically significant accuracy
  effect in that range.
- **Speculative decoding is output-preserving at greedy:** Leviathan et al. 2023, "Fast Inference from
  Transformers via Speculative Decoding," arXiv:2211.17192 [`leviathan2023fast`].
- **Sampling vs greedy / temperature effects on text:** Holtzman et al. 2020, "The Curious Case of
  Neural Text Degeneration," arXiv:1904.09751 [`holtzman2020curious`].
- **Serving baseline (PagedAttention / vLLM):** Kwon et al. 2023, SOSP [`kwon2023efficient`]; vLLM
  batch-invariance docs for the batch-determinism caveat.
- **Query/trial norms (Section 1):** RULER arXiv:2404.06654 (500 per condition); RAGAS arXiv:2309.15217
  (50); ARES arXiv:2311.09476 (~150); CacheBlend arXiv:2405.16444 (150-200 per dataset).
- **Model citations (Section 4):** Qwen3 2505.09388; MiMo 2505.07608; GLM-4.5 2508.06471; DeepSeek-V2
  2405.04434; Llama 2407.21783; Mistral 2310.06825; Qwen2.5 2412.15115; Mixtral 2401.04088.

> Note: `yuan2025nondeterminism`, `renze2024temperature`, `leviathan2023fast`, and `holtzman2020curious`
> were added to `Main.bib` this cycle; rebuild the bibliography (bibtex/biber) once so all four resolve.
