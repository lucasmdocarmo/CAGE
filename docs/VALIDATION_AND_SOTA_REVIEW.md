# CAGE — Strict Validation Report & SOTA Review

> Generated 2026-06-09. Adversarial (hostile-reviewer) audit of the actual code, infra, and methodology, plus answers to 5 research-direction questions benchmarked against current (mid-2026) state of the art. Companion to [CAGE_KNOWLEDGE_BASE.md](CAGE_KNOWLEDGE_BASE.md).
>
> **Bottom line up front:** The framework *architecture* is sound and the engineering is real, but **as currently implemented the headline results are not publishable.** Three independent failure classes each invalidate the central claims: (1) the "distributed" contribution is simulated, not real; (2) the headline quality comparison is confounded (gold vs retrieved context) and amplified by a broken retriever; (3) the two flagship quality metrics (NLI-Faithfulness, BERTScore) are mis-implemented, which fully explains the suspicious `0.570` and `0.324` constants. None of this is fatal to the *project* — every issue has a concrete fix, and Phase 2/3 is exactly where to do it.

---

## PART A — STRICT VALIDATION (failures raised)

> **🛠️ FIX STATUS (updated 2026-06-09).** This audit was written *before* the June 2026
> fix pass. Several findings are now **RESOLVED in code** — they are kept here for the
> record but are no longer "current bugs":
> - **FIXED** — Q1–Q5 (NLI `[SEP]` string → proper sentence-pair; hardcoded `LABEL_2` → entailment index resolved from model config; `top_k=1` 0/1 gate → continuous prob with claim-split + max-over-context; **claim splitting now implemented**; BERTScore `rescale_with_baseline=True`), Q7 (silent `0.5`/`0.0` sentinels → `None`), Q6 (relevance renamed `context_relevance`). LettuceDetect added as the PRIMARY grounding detector.
> - **FIXED** — O5 (e5 `query:`/`passage:` prefixes; rebuild indices with `--rebuild-ir-index`), O7 (non-stream TTFT), O6 (`statistical_tests.py` now exists).
> - **FIXED (infra)** — I1 (`install-nvidia-driver`), I2 (router `git clone` instead of SCP-wait hang), I6 (`/health` gating), I5 (`google_project_service`), I7 (Phase-3 `nic_type`/`network_mtu` params), I3/I8 (docker-compose.gpu fixes).
> - **STILL OPEN (protocol-level, for the cloud re-run)** — O1 (distributed baseline is SIMULATED — no real KV transfer), O2 (gold-vs-retrieved context confound), O3 (no per-trial cache flush / seed no-op), O4 (warm-hybrid leakage), O8 (prompt-cache ratio reflects prompt replay). These require re-running experiments under a corrected protocol; see Part C.

Severity: 🔴 CRITICAL (invalidates published numbers / thesis) · 🟠 MAJOR (a reviewer will reject on it) · 🟡 MINOR.

### A1. Evaluation metrics — `src/evaluation/quality.py`

| ID | Sev | Finding | Evidence | Consequence |
|---|---|---|---|---|
| Q1 | 🔴 | **NLI premise/hypothesis concatenated into one string** with literal `[SEP]` instead of a sentence pair. The pipeline tokenizes it as a single segment — `[SEP]` is just text. | `quality.py:276-279` `self.nli_model(f"{ctx} [SEP] {generated_text}", top_k=1)` | The DeBERTa NLI head runs fully out-of-distribution. Every faithfulness number is produced from an input format the model was never trained on. Correct call: `nli_model({"text": ctx, "text_pair": answer})`. |
| Q2 | 🔴 | **Entailment label index wrong for the primary model.** Code hard-codes `LABEL_2 = entailment`, true for the BART fallback but `LABEL_2 = contradiction` for the configured DeBERTa-mnli-fever-anli primary. | `quality.py:282-285` | If the primary model emits raw `LABEL_x`, contradiction is scored as entailment — faithfulness silently inverts. Proof the label mapping was never validated against the actual model. |
| Q3 | 🔴 | **`top_k=1` turns faithfulness into a 0/1 gate**, then averages over *all retrieved docs*. Non-entailing distractor passages each contribute `0.0`. | `quality.py:276-291`; call site `run_experiment.py:992` (top-k=3) | This is the **root cause of the 0.570 "ceiling"** — mean over k≈3 docs of a thresholded gate caps the score near 0.5-0.6 even for perfectly faithful answers. Should be `max` over docs (claim supported by *any* context) or claim-level entailment. |
| Q4 | 🔴 | **No claim splitting exists.** Docs/paper describe "split answer into claims, entail each, average" — grep for `claim`/`sent_tokenize`/`nltk` in eval path = 0 hits. Whole answer passed as one hypothesis. | `quality.py:277` | The described faithfulness metric was never implemented. A single weak clause flips the whole-answer argmax to neutral → `0.0`. |
| Q5 | 🔴 | **BERTScore computed without `rescale_with_baseline` and without `lang`.** The token "rescale" appears nowhere in the repo. | `quality.py:225-228, 351-354` | This is the **root cause of the flat 0.324-0.328**. Raw unrescaled RoBERTa-base F1 has almost no dynamic range across systems → non-discriminative by construction. Fix: `BERTScorer(lang="en", rescale_with_baseline=True)`. |
| Q6 | 🟠 | **"Relevance" measures question↔context, not answer quality.** It is independent of `generated_text` — identical context yields identical "relevance" whether the answer is perfect or empty. | `quality.py:297-323`, called with `(question, context)` | A retriever diagnostic mislabeled and presented as a serving-quality metric. |
| Q7 | 🟠 | **Silent sentinel fallbacks pollute means.** Model-load failure / exception / empty context all return `0.5` (faithfulness, relevance) or `0.0` (BERTScore), folded into the reported mean with no coverage flag, logged only via `print`. | `quality.py:268-269, 291, 294-295, 306-307, 325-327, 342-345` | Reported aggregates can be silently contaminated by undisclosed model-load failures; `0.5` is indistinguishable from a real score. |
| Q8 | 🟡 | No truncation control on NLI (`truncation`/`max_length` unset) → long contexts silently truncate the *answer* (appended last) or hit the 0.5 exception path. F1/EM scored against a *single* gold answer, not the SQuAD/TriviaQA alias set (`max` over golds). Inconsistent aggregation (faithfulness=mean, relevance=max). psutil first `cpu_percent()` returns spurious 0. | `quality.py:276-279, 400-409`; `performance.py:119,135` | Understates F1/EM; minor metric noise. |

### A2. Experimental flow & orchestration — `scripts/run_experiment.py`, `src/orchestration/*`

| ID | Sev | Finding | Evidence | Consequence |
|---|---|---|---|---|
| O1 | 🔴 | **The "distributed" contribution is simulated.** Router does HTTP forwarding + `% len(replicas)` replica pick; **zero** KV tensors ever move. "Transfer" is hardcoded arithmetic (`hidden_size=2048, layers=16, bandwidth=100Gbps`) injected as `asyncio.sleep()`. Default policy `replicated` sets transfer to literally 0. | `router.py:113,175-206,530-532`; `cache_manager.py:85-88,122-146`; default `run_experiment.py:1747` | The paper title's word "distributed" describes a round-robin HTTP proxy plus a `time.sleep()` of a back-of-envelope number. **No distributed KV system exists to evaluate.** |
| O2 | 🔴 | **Gold-vs-retrieved confound.** `no_cache`/`prefix_cache`/`distributed` are fed the GOLD SQuAD passage; `rag`/`redis`/`hybrid` are fed RETRIEVED passages. | `run_experiment.py:991-992, 1008-1011`; `loader.py:168` | The headline "prefix_cache 0% faithfulness loss vs RAG −11.6%" is dominated by *context source*, not caching. The CAG arms get the oracle answer-bearing passage; RAG has to find it. Apples-to-oranges. |
| O3 | 🔴 | **No vLLM restart / cache flush between trials, AND seed doesn't change the sample.** SQuAD loads the *first N* deterministically; `random.seed(seed)` is never consumed for HF datasets. | `run_experiment.py:1846-1890`; `loader.py:152-157` | The "3 independent trials" send the identical 50 prompts to an already-warm prefix cache. Trials 2-3 are warm replays of trial 1 → reported std is meaningless, cold-start numbers wrong, n=3 is effectively n=1. |
| O4 | 🔴 | **Warm-hybrid leakage.** The 50 "excluded" warmup queries are drawn from the *same* base set as the 50 measured queries — identical questions + context. Exclusion is cosmetic (metrics rows only); cache state is fully pre-warmed on the measured queries. | `run_experiment.py:813-838, 1251` | Textbook warm/measure leakage. Any "warm hybrid wins" result is an artifact. |
| O5 | 🔴 | **e5 retrieval is broken — required `query:`/`passage:` prefixes are missing** in both indexing and search. | `ir.py:113, 141-145` (0 prefix hits in repo) | `intfloat/e5-large-v2` runs out-of-distribution → depressed Hit@k → handicapped RAG faithfulness, which is then compared against gold-context CAG (O2). **Stacks the deck against RAG twice.** |
| O6 | 🔴 | **No statistical testing exists.** `scripts/statistical_tests.py` is referenced in docs but **absent**. No scipy/wilcoxon/bootstrap/p-value anywhere. Aggregation is mean ± `np.std` (ddof=0, understates sample std at n=3). | repo-wide grep; `run_experiment.py:1940-1995` | "65.7% TTFT reduction" has no error bar that survives review. With n=3 non-independent trials, no significance claim is supportable. |
| O7 | 🟠 | **Non-streaming / non-vLLM TTFT is fabricated** as `total_time × 0.2` (offline `× 0.1`). Gemini backend runs non-streamed → its TTFT is fiction. (Streaming vLLM path is *correct* — first content token, verified.) | `vllm_adapter.py:251, 335, 462, 514` | Any TTFT not from the streaming vLLM path is invalid. |
| O8 | 🟠 | **68.4% prompt-cached ratio is implausible for distinct documents.** Per-query context sits right after a ~25-token system preamble, so cross-query prefix reuse caps at ~25 tokens. A 68% blended ratio is only reachable if whole prompts repeat (`--repeat-queries`/warmup replay identical prompts). | `prompting.py:31-50`; `run_experiment.py:826-838, 1390-1392` | The cache ratio reflects prompt *replay*, not CAG's value on a static KB of distinct docs. Also: router hashes everything-before-"Question:", so different questions route to different replicas — undercutting the "prefix locality" story. |
| O9 | 🟡 | Reranker `reranker_top_k` is set but ignored (re-sorts, never truncates). Throughput measured under sequential single-request execution (`batch_generate` is "sequential for now") — a distributed system's concurrency benefit is untestable as run. | `ir.py:244-268`; `vllm_adapter.py:281-284` | Reranking is a reorder-only no-op for the cut; QPS numbers don't exercise load-balancing. |

**Real vs simulated (the table that matters):**

| Component | Status |
|---|---|
| HTTP routing to N replicas | ✅ Real |
| Single-node vLLM prefix caching | ✅ Real (vLLM-side) |
| FAISS retrieval (cosine/IP math) | ✅ Real but degraded (no e5 prefixes) |
| Redis retrieval-artifact cache | ✅ Real (caches doc-ids, not KV) |
| Streaming vLLM TTFT | ✅ Real |
| **Distributed KV tensor transfer** | ❌ **Simulated / nonexistent** |
| Transfer bytes / bandwidth / latency | ❌ Hardcoded arithmetic + `asyncio.sleep()` |
| Cross-trial independence | ❌ Fake (same data, warm cache) |
| Non-stream / Gemini TTFT | ❌ Fabricated (×0.2) |
| Statistical significance | ❌ Absent |

### A3. Infrastructure — Terraform / Docker / K8s / setup

| ID | Sev | Finding | Evidence | Fix |
|---|---|---|---|---|
| I1 | 🔴 | **GPU drivers never installed on replicas.** DLVM image needs `install-nvidia-driver=True` metadata; only `startup-script` is set. `docker run --gpus all` fails → vLLM never starts. | `terraform/gcp/main.tf:145,168,193-204` | Add `install-nvidia-driver = "True"` to replica metadata. |
| I2 | 🔴 | **Router blocks forever on an upload nothing performs.** Startup script `while [ ! -f /opt/cage/requirements.txt ]; do sleep 5; done` waits for an SCP that no resource/provisioner/deploy script ever does. | `main.tf:270-276`; `deploy_cluster.sh:298-304` | `git clone` in startup script, or add a `file`/`remote-exec` provisioner. **This is the "sit forever, no error" blocker.** |
| I3 | 🔴 | **CPU compose uses an ARM64 vLLM image** (`vllm-arm64-cpu-release-repo`) — `exec format error` on any x86 host. Laptop-only file presented as a path. | `docker/docker-compose.yml:9,18,27` | Use `vllm/vllm-openai` or x86 CPU image. |
| I4 | 🔴 | **K8s schedules vLLM with no GPU** (limits commented out), CUDA image on CPU node → crash; router image `cage-router:latest` + `imagePullPolicy: Never` is unpullable on multi-node GKE. No readiness probes, nodeSelector, or tolerations. | `k8s/vllm-replica.yaml:15-23`; `k8s/router.yaml:17-18` | Uncomment GPU limits, push image to Artifact Registry, add probes + accelerator nodeSelector. |
| I5 | 🔴 | **`terraform apply` fails on a fresh project:** 0 GPU quota, Compute API not enabled, L4 zone availability unchecked. Nothing requests quota or enables the API. | `main.tf:35`, no `google_project_service` | Pre-flight: `gcloud services enable compute…`, request `NVIDIA_L4_GPUS` quota, verify zone. |
| I6 | 🟠 | **No health gating anywhere in the cloud path.** Router launches as soon as a file appears; replicas are still pulling a multi-GB image + 16GB weights. Experiments fire against 503s. `depends_on` waits for start, not health; `kubectl wait … || true` swallows timeouts. | `main.tf:284`; compose `depends_on`; `deploy_cluster.sh:261` | Poll `/health` before launching router / running experiments. |
| I7 | 🟠 | **Phase-3 GVNIC/MTU/A100 not parameterized — and absent from the file entirely** (0 grep hits for `mtu`/`nic_type`/`a2-highgpu`). Requires hand-editing main.tf to even add the blocks. | `main.tf:75-85,161-166` | Parameterize `nic_type`, `mtu`, accelerator/machine pairing with a validation block. |
| I8 | 🟠 | **Pervasive inconsistency:** model is 4B in compose/k8s/tfvars but 8B in terraform default + experiment command; positional vs `--model` arg style mixed; all images `:latest` (repo claims "pinned"); `VLLM_GPU_MEMORY_UTILIZATION`/`VLLM_TENSOR_PARALLEL_SIZE` env vars are inert (vLLM reads CLI flags); GPU device-request in `gpu.yml` missing `driver`/`count`; `deploy_cluster.sh` references a non-existent `Dockerfile.router` and writes its own with the *heavy* requirements onto a 16GB router VM. | multiple | Pick ONE model + pinned image tag everywhere; complete the GPU device request; install only router requirements on the router. |
| I9 | 🟠 | **`requirements.txt` will likely not resolve in 2026.** `vllm` not listed; everything `>=` unpinned; `transformers>=4.36,<5` conflicts with setup scripts' hard `transformers==4.46.1`; `ragas` drags langchain/pydantic-v2; `numpy>=1.24` pulls numpy 2.x → breaks older faiss-cpu/bert-score ABI. setup_ubuntu.sh's `apt install python3.12` lines are commented out → dies on a fresh VM. | `requirements.txt`; `setup_ubuntu.sh:10-12,17` | Pin a known-good lockfile; uncomment Python install; use official CUDA vLLM wheel (source build unnecessary on GPU). |
| I10 | 🟡 | Replicas have public IPs + SSH open to 0.0.0.0/0 (exposed GPU boxes); HF_TOKEN embedded plaintext in instance metadata + tfstate; router SA scope is overbroad `cloud-platform`; `apt-key add` deprecated/removed on Debian 11+ and under `set -e` aborts the whole script. | `main.tf:118,163,198,182,292` | Restrict SSH source, drop replica public IPs, use keyring pattern, least-privilege SA. |

**Single coherent fresh-project→results path?** No. Biggest blockers, in order: **I2** (router hangs forever) and **I1** (no GPU driver). Fix those two or nothing downstream runs.

---

## PART B — THE 5 QUESTIONS, ANSWERED AGAINST SOTA

### Q1 — Are the evaluation metrics good enough? Should we add pytest + LettuceDetect?

**Current metrics are NOT good enough as implemented** (see Q1-Q7 above) — but the *design intent* (NLI-faithfulness + claim decomposition) is exactly right and matches what RAGAS does. Two moves:

**(a) Fix what you have** before adding anything: pair-format the NLI call, resolve the entailment index from `model.config.id2label`, use continuous probabilities, aggregate `max`-over-context (or claim-level), implement the claim splitting the paper already claims, and turn on BERTScore baseline rescaling. These fixes alone will move the numbers materially and make them defensible.

**(b) Adopt LettuceDetect — yes, strongly recommended, and it's a near-perfect fit.** It is an encoder (ModernBERT) token-classification model trained on RAGTruth (18k) that takes exactly your `(context, question, answer)` triple and flags unsupported tokens/spans. F1 ≈ 79.2% example-level (+14.8% over Luna), ~30× smaller than the best LLM judges, **30-60 examples/sec on one GPU** — i.e. cheap enough to run on every sample at Phase 2/3 scale, and handles up to 4-8k-token contexts (your DeBERTa pipeline currently truncates silently). It gives you **token/span-level hallucination localization**, a strictly stronger and more publishable signal than a single entailment scalar. MIT-licensed, pip-installable. Use it as the **primary** faithfulness detector and keep a fixed NLI score as a secondary cross-check.

**pytest as a hallucination gate — yes, but as a *regression harness*, not a metric.** Add a `tests/test_faithfulness_regression.py` with a small curated set of (context, question, known-faithful answer) and (…, known-hallucinated answer) pairs, assert LettuceDetect/NLI classifies them correctly and that aggregate faithfulness on a frozen golden run stays within a tolerance band. That catches metric regressions (like the `[SEP]` bug) automatically — which is exactly the class of bug that silently corrupted Phase 1.

**Also worth adding:** RAGAS-style answer-relevancy and context-precision/recall (these are retriever-quality metrics your current "relevance" pretends to be), and a small LLM-as-judge spot-check (a strong Claude/GPT judge on ~50 samples) to validate the cheap encoder metrics agree — the literature shows high NLI↔LLM-judge agreement on faithfulness, so this is a cheap credibility booster for the paper.

### Q2 — Do the IR quality algorithms match the purpose? FAISS / ScaNN / ANN / BERTScore?

**FAISS is the correct and sufficient choice — do NOT switch to ScaNN.** Your corpus is tiny (SQuAD/TriviaQA passage sets), so you are using `IndexFlatIP` = *exact* brute-force search. At this scale ANN libraries are irrelevant: there is no recall loss to recover and no latency problem to solve. Benchmarks (2025) put FAISS ahead of ScaNN on indexing speed, query latency, and retrieval accuracy in most configs; ScaNN only wins at billion-vector MIPS scale and is research-grade (x86/AVX-only, batch-oriented). HNSW matters only for low-latency interactive serving at scale. **None of these apply to your project** — keep FAISS Flat (exact). If Phase 3 ever indexes millions of passages, add `IndexHNSWFlat` or `IndexIVFFlat` then, not now.

**The real IR problem is not the index — it's the embedding usage (O5):** you're running e5 without its mandatory `query:`/`passage:` prefixes, which silently degrades retrieval and unfairly handicaps RAG. Fix that first; it will change your headline RAG numbers more than any index swap.

**Retrieval-quality metrics are missing entirely.** You report no Recall@k, MRR, nDCG, or Hit@k — yet the whole RAG-vs-CAG argument hinges on retrieval quality. Add them (you already have gold passages to score against). This is table-stakes for an IR-adjacent paper.

**BERTScore: usage is incorrect AND the metric is weak for your purpose.** Beyond the missing baseline-rescaling bug (Q5), BERTScore measures surface/embedding overlap, which is *blind to factual inversion* — exactly the failure mode you care about (your own docs already note this). **Demote BERTScore to a secondary "fluency/overlap" signal** (fixed with `rescale_with_baseline=True, lang="en"`), and make **LettuceDetect + claim-level NLI** your primary faithfulness axis, plus SQuAD **EM/F1 over the gold-alias set** for answer correctness. That trio (correctness, faithfulness, retrieval-quality) is what reviewers expect; "relevance" as currently defined should be renamed to a retriever diagnostic or dropped.

### Q3 — Is the GCP setup correct and complete?

**No — it will not stand up a working cluster on a fresh project today.** See A3. The two hard stops are **I2** (router waits forever for an upload that nothing performs) and **I1** (GPU driver never installed, so vLLM never starts). Add the quota/API pre-flight (I5), x86 images (I3), health-gating (I6), and resolve the 4B-vs-8B / image-tag inconsistencies (I8). The K8s manifests as-shipped (I4) are non-functional for GPU. A corrected, ordered Phase-2 checklist is in the audit; the short version:

1. `gcloud services enable compute.googleapis.com`; request `NVIDIA_L4_GPUS` quota; verify L4 in zone.
2. Pin ONE model (Qwen3-8B fits one L4 @ 0.9 util) and ONE vLLM image tag everywhere.
3. Patch terraform: `install-nvidia-driver=True`; `git clone` the repo in the router startup script; poll replica `/health` before launching.
4. `terraform apply`; verify via serial-port logs that `nvidia-smi` works and `/health` returns 200 before running experiments.
5. Run experiments from your workstation against `http://<router-ip>:9000`, writing to `analysis/phase2/`.
6. `terraform destroy` immediately after (3× L4 + 3× pd-ssd bill continuously).

Skip for cloud: the ARM CPU compose, building vLLM from source (use the official CUDA wheel), and the K8s path until I4 is fixed.

### Q4 — Is Qwen the right model family for an HPC/cloud KV-cache study? Better memory-pressure options?

**Qwen3 is a fine baseline family but it is the wrong choice for a study whose entire thesis is KV-cache memory pressure.** Qwen3 dense models use **Grouped-Query Attention (GQA)** — standard, but it means your KV-cache-per-token is "ordinary," and your central variable (cache size / eviction pressure) is not where the interesting 2026 action is.

**The single most important architectural lever for your thesis is the attention mechanism, because it dominates KV-cache size:**
- **MLA (Multi-head Latent Attention)** — DeepSeek's low-rank KV compression — shrinks the KV cache **7-14×** vs MHA while matching or beating quality. For a paper about distributed KV transfer and memory pressure, **a model family that varies the KV footprint is the most scientifically interesting axis you have.** Running the *same* experiment on a GQA model (Qwen3) and an MLA model (DeepSeek-family) would let you show how cache compression changes the CAG-vs-RAG trade-off and the cost of cross-node transfer — a far stronger contribution than "Qwen3 at three sizes."

**Concrete recommendation:**
- Keep **Qwen3-4B/8B/14B (GQA)** as the controlled scaling sweep (you already have configs).
- **Add a DeepSeek MLA model** (or another MLA/compressed-KV model) as a second attention-architecture arm. This directly feeds your Phase-3 "cross-node KV transfer" story: MLA's compressed latent KV is dramatically cheaper to move over the wire, which is *exactly* the bottleneck your GVNIC/jumbo-frames section is about.
- Note the caveat from the field: FP8 KV cache is currently buggy on some MLA models in vLLM — validate quality before trusting FP8-KV numbers on MLA.

So: Qwen is *correct but insufficient*. The memory-pressure question you're asking is best answered by **varying the attention architecture (GQA vs MLA), not just the parameter count.**

### Q5 — Does the project match current vLLM SOTA? Can the newest vLLM + your cloud setup deliver a better contribution?

**Your project is built against an old mental model of vLLM and is missing the single most important development for your thesis.** As of mid-2026 vLLM is far past the `v0.8.x` era your setup scripts pin (`vllm==0.8.3`); the project is on the **V1 engine**, CUDA 13, and — critically — has **native, first-class KV-cache transfer infrastructure** that did not exist when CAGE's "simulated transfer" was written:

- **LMCache integration + KV connectors** (`LMCacheConnector`/`LMCacheMPConnector`): a production KV-cache layer that **actually moves KV blocks across engines and nodes**, with reference-counted dedup, CPU/disk offload, and cross-query prefix reuse. **This is the real implementation of the thing CAGE currently fakes with `asyncio.sleep()` (O1).**
- **Native disaggregated prefill** (prefill workers ↔ decode workers with real KV transfer over the network) — your Phase-3 `--enable-disagg-prefill` plan is now a supported, documented feature with LMCache as the transport.
- **MLA / compressed-KV support, KV offload + HMA, FP8 KV cache** (with the MLA caveat above), and prefix caching reimplemented in the V1 KV-cache manager.

**This is an opportunity, not just a gap.** The most valuable pivot for the project:

1. **Replace the simulated `SimulatedKVCacheManager` with real vLLM KV transfer via LMCache.** Your `vllm_adapter.py` already has `kv_transfer_params` hooks. This turns the central contribution from "simulated distributed cache" into **"measured distributed KV transfer on real hardware"** — which is publishable and is the honest version of the current title.
2. **Measure, don't model, the cross-node transfer cost** (bytes, latency, hit rate) that `cache_manager.py:122-146` currently hardcodes. This is exactly what GVNIC + jumbo frames (Phase 3) are for, and LMCache gives you the real numbers.
3. **Use disaggregated prefill as a first-class baseline**, not an afterthought — TTFT-vs-TPOT decoupling is a headline systems result.
4. **Add the MLA arm (Q4)** so the transfer-cost story has a compression dimension.

Net: upgrade to a current pinned vLLM (V1 engine), adopt LMCache as the KV-transfer backbone, and your already-planned Phase 2/3 phases become a genuinely SOTA-aligned, measured study instead of a simulated one — a strict improvement on your original scope using the *same* roadmap and cloud setup.

---

## PART C — PRIORITIZED FIX LIST (what to do, in order)

**Before re-running any experiment (correctness):**
1. Fix NLI faithfulness: pair-format input, resolve entailment index per model, continuous prob, `max`/claim-level aggregation, implement claim splitting (Q1-Q4).
2. Turn on BERTScore baseline rescaling; demote it to secondary (Q5).
3. Add e5 `query:`/`passage:` prefixes (O5).
4. Remove the gold-vs-retrieved confound: feed CAG arms the *same retrieved* context as RAG, or add an explicit labeled oracle arm (O2).
5. Restart vLLM + flush caches per trial; actually randomize the sample per seed (O3).
6. Draw warmup queries from a held-out disjoint pool (O4).
7. Replace fabricated non-stream TTFT or restrict reporting to the streaming path (O7).

**Before claiming significance:**
8. Write the missing `statistical_tests.py`: paired tests at the per-query level (n=50), bootstrap CIs, effect sizes, sample std (O6).

**To make the "distributed" claim legitimate (Phase 3):**
9. Replace `SimulatedKVCacheManager` with real vLLM KV transfer via **LMCache**; measure transfer bytes/latency/hit-rate on GVNIC (O1, Q5).
10. Add disaggregated-prefill and an MLA-model arm (Q4, Q5).

**To make the cloud actually run (Phase 2):**
11. Terraform: `install-nvidia-driver=True`, `git clone` in router startup, `/health` gating, quota/API pre-flight, one model + pinned image everywhere (I1-I8).

**Metric infrastructure:**
12. Integrate **LettuceDetect** as primary faithfulness detector; add Recall@k/MRR/nDCG retrieval metrics; add a pytest faithfulness-regression gate; spot-check with an LLM judge (Q1, Q2).

---

## Sources (SOTA review)
- vLLM KV transfer / disaggregated serving: [DeepWiki](https://deepwiki.com/vllm-project/vllm/9.4-kv-cache-transfer-and-disaggregated-serving), [vLLM disagg-prefill docs](https://docs.vllm.ai/en/latest/features/disagg_prefill/), [vLLM LMCache examples](https://docs.vllm.ai/en/latest/examples/disaggregated/lmcache/), [vLLM prefix caching design](https://docs.vllm.ai/en/latest/design/prefix_caching/)
- LMCache: [arXiv 2510.09665](https://arxiv.org/pdf/2510.09665)
- vLLM releases: [GitHub releases](https://github.com/vllm-project/vllm/releases), [PyPI](https://pypi.org/project/vllm/)
- LettuceDetect: [arXiv 2502.17125](https://arxiv.org/abs/2502.17125), [HF blog](https://huggingface.co/blog/adaamko/lettucedetect)
- RAG eval / faithfulness (NLI vs LLM-judge): [CCRS arXiv 2506.20128](https://arxiv.org/pdf/2506.20128), [Deepchecks](https://deepchecks.com/rag-evaluation-metrics-answer-relevancy-faithfulness-accuracy/)
- MLA / KV memory: [PyImageSearch MLA](https://pyimagesearch.com/2025/10/13/kv-cache-optimization-via-multi-head-latent-attention/), [KV cache optimization 2026](https://www.digitalapplied.com/blog/kv-cache-optimization-techniques-2026-engineering-guide)
- FAISS vs ScaNN vs HNSW: [Zilliz](https://zilliz.com/blog/faiss-vs-scann-choosing-the-right-tool-for-vector-search), [FAISS/ScaNN study arXiv 2507.16978](https://arxiv.org/html/2507.16978v1), [Milvus](https://milvus.io/ai-quick-reference/what-is-the-role-of-faiss-hnsw-and-scann-in-ai-databases)
