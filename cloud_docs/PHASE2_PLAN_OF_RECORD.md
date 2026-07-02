# CAGE Phase-2 Re-run: Plan of Record

Status: authoritative Phase-2 re-run plan. Single on-demand NVIDIA L4. Supersedes
`PHASE2_CHECKLIST.md` for the re-run (the checklist predates this design: it still
says Spot, 100 queries x 10 trials, and "nine baselines"). Grounded against the
working tree at commit `51b68af` ("commit batch - phase2-3 changes and fixes") and a
live read-only GCP state on project `cage-framework`.

This document is the contract for the re-run. Where a script default, a sibling doc,
or an orchestrator header disagrees with this plan, this plan wins and the drift is
called out inline so it cannot bite silently.

---

## 1. Objective and Scope

### What Phase 2 proves

Phase 2 establishes the full CAGE baseline matrix under a single, reproducible decoding
protocol on real GPU serving, producing the result sets and per-query statistics that
feed the dissertation results and discussion chapters. Concretely it delivers:

- The six core single-server baselines (the no-cache reference plus RAG, Redis
  retrieval cache, prefix cache, and the hybrid cold/warm pair).
- The 2x2 compression axis (CAG vs RAG) x (full vs compressed): `cag_full`, `rag_full`,
  `compressed_rag`, `compressed_cag`.
- The 2x2 speculative axis run once per model: {ngram, native-draft} x {CAG gold,
  RAG retrieved}, for both Qwen3-8B and MiMo-7B-RL (eight cells total).
- A single per-query statistical layer (Wilcoxon signed-rank + Holm + effect sizes +
  bootstrap) against the `no_cache` reference, emitted as `phase2_stats.json` and
  `phase2_stats.tex`.

That is fourteen result sets in the headline matrix (6 core + 4 compression + 4
speculative per model), with the speculative axis doubled across the two models.

### What is in scope

- ONE on-demand GPU VM: `g2-standard-8` (8 vCPU / 32 GB) + 1x `nvidia-l4` (24 GB),
  zone `us-central1-a`, project `cage-framework`. This is Path A in the RUNBOOK.
- T=0 greedy as the main decoding protocol (Section 2).
- SQuAD v2 for Phase-2 continuity, 300 queries per baseline (locked count,
  `RERUN_DESIGN.md:13`).
- LettuceDetect-primary metric hierarchy and the full statistics stack (Section 4).

### What is explicitly out of scope (deferred to Phase 3)

- The distributed / replicated-router family. The local 3-replica distributed baseline
  OOMs a 24 GB L4, so it is gated OFF by default: `run_phase1.sh:25`
  (`ENABLE_DISTRIBUTED=${ENABLE_DISTRIBUTED:-0}`), gate at `run_phase1.sh:158`, and
  `cloud_run.sh:43` re-exports the 0 default. `distributed_router_replicated` is a
  Phase-3 baseline and must not be enabled on the L4.
- Any multi-VM cluster (Path B / Terraform). Phase 2 is a single on-demand VM only.

---

## 2. Models and Decoding Protocol

### Models

| Model | Role | Speculative support | Bib key |
|---|---|---|---|
| Qwen/Qwen3-8B | Primary / re-run anchor | ngram + EAGLE-3 (no MTP) | `qwen3report` (`Main.bib:1311`) |
| XiaomiMiMo/MiMo-7B-RL | Second model | ngram + native MTP (`mimo_mtp`), LIVE-VALIDATE on stock vLLM 0.11.0 | `mimo2025` (`Main.bib:1319`) |

Grounding: `RERUN_DESIGN.md:72-78`; the dissertation names both at
`my-article/.../TEXT/4_METHODOLOGY.tex:167`. Note the bib keys are `qwen3report` and
`mimo2025`, NOT literal `qwen3` / `mimo`.

DRIFT to fix in the manuscript (not a run blocker): the two model papers are cited only
in `7_CONCLUSIONS.tex:12`, not at the point the models are introduced in
`4_METHODOLOGY.tex:167`. Add `\cite{qwen3report}` and `\cite{mimo2025}` there.

### Decoding protocol

- T=0 greedy is the MAIN protocol for Phase 2 and Phase 3, chosen for reproducible
  attribution and the output-preservation (losslessness) check that speculative output
  must be token-for-token identical to non-speculative. `RERUN_DESIGN.md:42-49`;
  `4_METHODOLOGY.tex:138` (cites `holtzman2020curious`, `leviathan2023fast`).
- Quality measured ONCE (deterministic under greedy, single low-concurrency pass) +
  serving measured over 3 TRIALS for confidence intervals under the concurrent
  KV-pressure regime. `RERUN_DESIGN.md:24-29`; `4_METHODOLOGY.tex:142`.
- Phase-1 vs Phase-2/3 protocol change is documented and must not be read as a
  temperature effect: Phase 1 ran T=0.7 (stochastic, Qwen3-4B, unlogged); Phases 2-3
  use T=0.0. `RERUN_DESIGN.md:50-54`; `4_METHODOLOGY.tex:140`.

### Separate temperature sub-study (never pooled with the greedy main matrix)

A SEPARATE temperature-sensitivity sub-study sweeps T in {0, 0.5, 1.0} on a subset
(1 model, 2-3 representative baselines), drawing k >= 3 samples per query at T > 0,
reported separately. `RERUN_DESIGN.md:59-65`; `4_METHODOLOGY.tex:144` (cites
`renze2024temperature`).

OPEN DECISION (must be locked before the run): `RERUN_DESIGN.md:108-112` flags the
{0,0.5,1.0} sub-study as the one open decision pending confirmation, while
`4_METHODOLOGY.tex:144` already writes it as settled. Confirm and lock so the design doc
and methodology agree before launch.

### The 300 x 3 count and where it lives

- 300 queries per baseline is the LOCKED design count (`RERUN_DESIGN.md:13`,
  justified vs cited norms: RAGAS 50, ARES ~150, CacheBlend 150-200, under RULER 500).
- 3 trials for the serving portion (per Section 2 above).

CRITICAL SCALE DRIFT: no script defaults to 300 x 3. `cloud_run.sh:38-39` defaults
100 queries / 10 trials; `run_compression.sh:24-25` 100/3; `run_speculative_matrix.sh:35-36`
100/1; `run_phase1.sh:15-16` 50/3. The full design MUST be passed explicitly to every
script (`NUM_QUERIES=300 NUM_TRIALS=3`, and positional `Qwen/Qwen3-8B 300 3` for
`cloud_run.sh`) or the re-run silently reproduces the prior under-powered counts. The
`cloud_run.sh` 10-trial default is unusually high and would inflate cost if not overridden.

MANUSCRIPT DRIFT (fix before integrating results): "300 queries x 3 trials" is stated
only in `7_CONCLUSIONS.tex:14`, never in `4_METHODOLOGY.tex`; and `6_RESULTS.tex:41`
still carries a stale "n=50 queries, 3 trials" Phase-1 caption. Reconcile both to 300.

---

## 3. The Full Baseline Matrix (in run order)

The Phase-2 suite is assembled from THREE producing scripts plus one consolidator:

- (a) core suite `run_phase1.sh`, driven by `cloud_run.sh` -> `analysis/phase1/results`
- (b) compression 2x2 `run_compression.sh` -> `analysis/compression/results`
- (c) speculative 2x2 `run_speculative_matrix.sh`, run once per model ->
  `analysis/speculative_matrix`
- consolidator `run_phase2_stats.sh` symlinks all three trees into
  `analysis/all_results` and runs stats (`run_phase2_stats.sh:13`).

Context-source is implicit `auto` for every core arm (no `--context-source` flag in
`run_phase1.sh:91-101`; `run_experiment.py:1849-1850` defaults `auto`;
`run_experiment.py:644-647` resolves `auto` = gold for CAG arms, retrieved for RAG arms).

### [a] CORE SUITE - `cloud_run.sh` -> `run_phase1.sh` (output `analysis/phase1/results`)

Server-lever schedule: prefix-cache OFF for the first server block, then ON for
`prefix_cache`, then a fresh ON restart for each hybrid arm (`run_phase1.sh:132-156`;
`manage_vllm_server.sh:75-78` maps `--no-prefix-cache` to `--no-enable-prefix-caching`).

| # | Label | Family | Context | Server lever | Notes |
|---|---|---|---|---|---|
| 1 | `no_cache` | no_cache | auto -> gold | prefix-cache OFF | THE Wilcoxon reference |
| 2 | `rag` | rag | auto -> retrieved | prefix-cache OFF | |
| 3 | `redis_retrieval_cache_cold` | redis | retrieved | prefix-cache OFF | flush redis ns `phase1:squad_v2:redis_retrieval_cache_cold` |
| 4 | `prefix_cache` | prefix_cache | auto -> gold | prefix-cache ON | |
| 5 | `hybrid_retrieval_cache_cold` | hybrid cold | retrieved | prefix-cache ON, fresh restart | flush redis ns; empty caches |
| 6 | `hybrid_retrieval_cache_warm` | hybrid warm | retrieved | prefix-cache ON, fresh restart | `--warmup-queries=NUM_QUERIES` excluded from measured metrics |
| - | `distributed_router_replicated` | distributed | - | GATED OFF | Phase-3 only; OOMs 24GB L4 |

The warm/cold pairing is rows 5 and 6: the same hybrid family run cold (empty caches)
then warm (`--warmup-queries` equal to `NUM_QUERIES`, warmup excluded from metrics).
`no_cache` (row 1) is the no-cache reference and is produced ONLY here.

### [b] COMPRESSION 2x2 - `run_compression.sh` (output `analysis/compression/results`)

Eager mode, `max_model_len 4096`. Runs AFTER two pre-flight gates: the
FP8-x-prefix-cache gate (`run_compression.sh:62-69`, `check_fp8_prefix_cache.sh`) and the
llmlingua-importable gate (`run_compression.sh:73-77`; else `compressed_rag` would
silently no-op to ratio 1.0). `cag_full`/`rag_full`/`compressed_rag` share one
full-precision prefix-caching-ON server; `compressed_cag` forces a relaunch with
`VLLM_KV_CACHE_DTYPE=fp8` (`run_compression.sh:94`).

| # | Label | Cell | Context | Server lever |
|---|---|---|---|---|
| 7 | `cag_full` | CAG, full precision (baseline=prefix_cache) | auto -> gold | full precision, prefix-cache ON |
| 8 | `rag_full` | RAG, full text (baseline=rag) | auto -> retrieved | full precision, prefix-cache ON |
| 9 | `compressed_rag` | RAG + LLMLingua-2 ~2x client-side | FORCED retrieved | full precision, prefix-cache ON; `CAGE_REQUIRE_COMPRESSION=1` |
| 10 | `compressed_cag` | CAG + FP8 KV ~2x server-side | auto -> gold | RELAUNCH with `VLLM_KV_CACHE_DTYPE=fp8` |

Row 9 confound fix (B2): `compressed_rag` MUST pass `--context-source retrieved`
(`run_compression.sh:89`) or it compresses GOLD context, becoming CAG+compression and
breaking the 2x2 ACROSS read. This is now set inline, so the standalone
`rerun_compressed_rag.sh` is ONLY for repairing an old confounded tree and is not part
of a fresh re-run.

OVERLAP to flag for the paper (not a bug): `cag_full == prefix_cache` and
`rag_full == rag` are re-measured under different labels. The compression-tree copies
run eager / `max_model_len 4096`, whereas the core-suite `prefix_cache`/`rag` run with the
default (non-eager) server / `max_model_len 8192`. Serving numbers for the "same" arm are
therefore NOT directly comparable across trees; only within-tree comparisons are valid.

OPEN: confirm `compressed_cag` should stay `auto -> gold` (CAG side of the 2x2);
`run_compression.sh:95` passes no `--context-source`, so it is gold by design.

### [c] SPECULATIVE 2x2 - `run_speculative_matrix.sh`, RUN ONCE PER MODEL

Output `analysis/speculative_matrix`. Each cell is its own `--speculative-config` server
launch (eager, `max_model_len 4096`, mem-util 0.90). Cells = {ngram, native-draft} x
{cag = gold, rag = retrieved}. A failed server launch OR run writes a `STATUS=failed`
sentinel into the cell dir (`run_speculative_matrix.sh:79-94`) so a hole in the 2x2 is
loud, not silently absent. The model tag prevents Qwen and MiMo cells from colliding in
consolidation (`run_speculative_matrix.sh:44`).

Spec JSON per cell (`run_speculative_matrix.sh:45/53/59`): ngram cells use
`{"method":"ngram","num_speculative_tokens":5}`; Qwen draft cells use
`{"method":"eagle3","model":"AngelSlim/Qwen3-8B_eagle3",...}`; MiMo draft cells use
`{"method":"mimo_mtp",...}` (overridable via `MIMO_MTP_CONFIG`).

MODEL 1 - Qwen/Qwen3-8B (MTAG=qwen8b, native draft = eagle3 `AngelSlim/Qwen3-8B_eagle3`):

| # | Label | Method | Context |
|---|---|---|---|
| 11 | `spec_qwen8b_ngram_cag` | ngram | gold / CAG |
| 12 | `spec_qwen8b_ngram_rag` | ngram | retrieved / RAG |
| 13 | `spec_qwen8b_eagle3_cag` | eagle3 native draft | gold / CAG |
| 14 | `spec_qwen8b_eagle3_rag` | eagle3 native draft | retrieved / RAG |

MODEL 2 - XiaomiMiMo/MiMo-7B-RL (MTAG=mimo7b, native draft = mtp method `mimo_mtp`,
LIVE-VALIDATE on vLLM 0.11.0):

| # | Label | Method | Context |
|---|---|---|---|
| 15 | `spec_mimo7b_ngram_cag` | ngram | gold / CAG |
| 16 | `spec_mimo7b_ngram_rag` | ngram | retrieved / RAG |
| 17 | `spec_mimo7b_mtp_cag` | mimo_mtp native MTP | gold / CAG |
| 18 | `spec_mimo7b_mtp_rag` | mimo_mtp native MTP | retrieved / RAG |

DRIFT (driver selection): `cloud_run.sh:29` still advertises `run_phase5.sh` as the
speculative path, and `RUNBOOK.md:460` / `setup_gpu_cloud.sh:106` do too. That is WRONG
for Phase 2. `run_phase5.sh` is an ngram-only single arm targeting the wrong default
models (Qwen3-4B / Qwen3-0.6B) and writes `analysis/phase5/results`, which the stats
consolidator never reads. ONLY `run_speculative_matrix.sh` writes the
`analysis/speculative_matrix` tree that `run_phase2_stats.sh:13` globs. Use
`run_speculative_matrix.sh` once per model; do NOT use `run_phase5.sh` /
`run_speculative.sh`.

### [CONSOLIDATION] `run_phase2_stats.sh`

Symlinks every results subdir from `analysis/{phase1/results,compression/results,speculative_matrix}/*`
into `analysis/all_results`, HARD-FAILS if `no_cache` is missing
(`run_phase2_stats.sh:22-26`), then runs `statistical_tests.py --reference no_cache` over
`{grounding_score, faithfulness, context_relevance, ttft_ms, latency_ms, f1_score}`,
emitting `phase2_stats.json` + `phase2_stats.tex`, then syncs to GCS.

ORPHAN WARNING: `scripts/run_phase2.sh` exists with the SAME 6 core baselines but writes
`analysis/phase2/results` and is referenced by NOTHING (`run_phase2.sh:10-11`). If anyone
runs it thinking it is "the Phase 2 core", its output lands where the consolidator never
looks and `no_cache` appears missing. The canonical core path is
`cloud_run.sh` -> `run_phase1.sh` (`analysis/phase1/results`). Resolve this orphan
(delete or repoint) so it cannot mislead. Note also the namespace difference:
`run_phase1.sh` uses `phase1:*` redis keys; that is the live, accepted prefix for
Phase-2 labeling.

---

## 4. Metrics and Statistics

### Metric hierarchy (`4_METHODOLOGY.tex:104-119`)

- PRIMARY: LettuceDetect span-level grounding, `grounding = 1 - r`.
- SECONDARY: strict claim-level NLI faithfulness (cross-encoder).
- DIAGNOSTIC references: token-level F1 / exact match, ROUGE-L, completeness
  (baseline-rescaled BERTScore + ROUGE-L).
- NEGATIVE CONTROL: BERTScore, used deliberately as a control.

MANUSCRIPT DRIFT: the PRIMARY grounding metric is named but UNCITED at
`4_METHODOLOGY.tex:104-106` (no LettuceDetect citation); the secondary NLI metric is
cited to `espejel2023ragas` / `ares2024`. Add a LettuceDetect citation beside the
primary metric.

### Statistics stack (`4_METHODOLOGY.tex:127-131`; `RERUN_DESIGN.md:19`)

- Unit of analysis: the individual query (per-query pairing).
- Test: paired Wilcoxon signed-rank, with Mann-Whitney fallback when fewer than 3 shared
  queries.
- Reference baseline: `no_cache` for ALL comparisons.
- Multiplicity: Holm correction WITHIN each metric.
- Effect sizes: rank-biserial correlation + Cliff's delta.
- Confidence: bootstrap 95% CI, 10,000 seeded resamples.

`no_cache` is the single Wilcoxon reference and is produced ONLY by the core suite
(`run_phase2_stats.sh:19-21`). This is the single point of failure for the entire stats
layer: if the core suite is skipped or `no_cache` fails, `run_phase2_stats.sh:22-26` hard
-exits. Hence the hard ordering in Section 8: core suite FIRST.

The supporting reference set is internally consistent in `Main.bib`
(`holtzman2020curious`, `leviathan2023fast`, `yuan2025nondeterminism`,
`renze2024temperature`, `kwon2023efficient`, `vllmblog`, `espejel2023ragas`,
`ares2024` all resolve). The bibliography must be rebuilt once so the four newly added
keys resolve (`RERUN_DESIGN.md:139-140`).

OPEN (metric list): `run_phase2_stats.sh` omits `hallucinated_span_ratio` (the
LettuceDetect primary) and `exact_match` that RUNBOOK Section 7 includes. Confirm whether
the Phase-2 stats table should add `hallucinated_span_ratio` to match the hierarchy.

---

## 5. Infrastructure, Environment, and Grounded Cost+Time Estimate

### Live starting state (read-only, confirmed)

`gcloud` authenticated as `lucas.mariano.carmo@gmail.com`, project `cage-framework`,
ZERO compute instances, ZERO storage buckets, no `compute/zone` set. This matches the
documented "torn down to $0, bucket deleted" state. The re-run starts from a clean slate.

### VM and image

- On-demand VM: `g2-standard-8` (8 vCPU / 32 GB) + 1x `nvidia-l4` (24 GB),
  zone `us-central1-a` (`RUNBOOK.md:318-326`). On-demand means NO `--provisioning-model=SPOT`;
  the `gcp_shutdown_hook.sh` spot-preemption backstop is therefore optional (it targets
  the ~30s ACPI budget on SPOT preemption only).
- Boot image: Deep Learning VM `common-cu121-debian-11` (CUDA 12.x present),
  `deeplearning-platform-release`, 200 GB pd-ssd, `--maintenance-policy=TERMINATE`,
  `--metadata=install-nvidia-driver=True`, `--scopes=cloud-platform` (`RUNBOOK.md:323-325`).
- IMAGE PIN AGE RISK: `common-cu121-debian-11` is an older DLVM line. Verify it still
  resolves and its bundled driver satisfies vLLM 0.11.0 at run time;
  `setup_gpu_cloud.sh:34-39` hard-fails if `nvidia-smi` is not working.

### Environment (cage-env, mandatory)

`scripts/setup/setup_gpu_cloud.sh`, run ONCE on the VM, creates the venv named
`cage-env` (NOT `.venv`), installs the pinned vLLM 0.11.0 GPU wheel
(`setup_gpu_cloud.sh:59`), installs `requirements.txt` (pulls cage-stats + pynvml +
metric stack), force-upgrades `openai>=2.0` to fix a vLLM-0.11.0/lettucedetect conflict
(`setup_gpu_cloud.sh:71`; lettucedetect pins `openai==1.66.3` which crashes 0.11.0
startup), pre-stages datasets squad_v2 / natural_questions / musique
(`setup_gpu_cloud.sh:74-77`), and verifies `pynvml.nvmlInit()` + `cage_stats.api`
import (`setup_gpu_cloud.sh:79-97`). `httpx>=0.27` is pinned directly in
`requirements.txt:39-42` so the in-process cage-stats telemetry path works regardless of
how cage-stats resolves.

`cage-env` is mandatory: `run_compression.sh`, `run_speculative_matrix.sh`,
`run_phase2_stats.sh`, and `rerun_compressed_rag.sh` hard-source `cage-env` under
`set -e`; the matrix and stats scripts also hard-require the repo at `~/CAGE`
(`run_speculative_matrix.sh:26-27`). A `.venv` or a non-`~/CAGE` checkout breaks them.
`run_phase1.sh:42` probes `.venv`, then `cage-env`, then `../cage-env`.

### Durable results bucket (MUST be recreated)

The durable bucket is `gs://cage-framework-cage-results` (pattern
`gs://<project>-cage-results`), the default in `sync_results_to_gcs.sh:35`,
`cloud_run.sh`, and `teardown_vm.sh:35`. Live check shows 0 buckets, so it does NOT
exist. Recreate it FIRST (`gsutil mb -l us-central1` + `gsutil versioning set on`,
`RUNBOOK.md:312-313`). If the suite launches before the bucket exists, the background
sync fails every 120s, NO results persist, and `teardown_vm.sh` correctly aborts
fail-closed, stranding the VM (and its cost) if unnoticed.

The VM service account needs `roles/storage.objectAdmin` on the manually created bucket,
or every `gsutil` sync 403s (`RUNBOOK.md:441`). Terraform grants this automatically;
a manual bucket does NOT. Grant it before the VM runs (Section 8 step 3). Assumed SA:
`<projectNumber>-compute@developer.gserviceaccount.com`; if a custom SA is attached at
create time, target that instead.

GCS storage cost is negligible (a few cents/month for the small JSON/CSV artifacts) and
the bucket is kept after teardown, so it is not part of the run-cost estimate
(`CLOUD_CONSOLE_GUIDE.md:173-175`). The cost driver is exclusively GPU-VM wall-clock.

### Grounded cost and time estimate

| Anchor | Value | Source |
|---|---|---|
| L4 on-demand price | ~$0.85/hr (vs ~$0.30/hr spot) | `CLOUD_CONSOLE_GUIDE.md:97` |
| g2-standard-8 + L4 all-in | ~$1.20/hr | `CLOUD_CONSOLE_GUIDE.md:21` |
| Prior Phase-2 actual | ~$3.1 at 100q x 1 trial, Qwen3-8B only (~2.6 wall-clock h at $1.20/hr) | user memory / brief |

Full Phase-2 re-run projection (on-demand, conservative ~$1.20/hr all-in):
~7-12 hours wall-clock, ~$8-15 total; mid-point ~9-10 h / ~$11-12. At GPU-only
~$0.85/hr the range is ~$6-10. Scaling logic from the ~$3.1 / ~2.6 h prior: 3x for 300
queries (query count dominates wall-clock), the 3-trial serving CIs, a SECOND model
(MiMo-7B-RL) for the speculative 2x2, the FP8 2x2 + per-cell speculative server
relaunches, plus one-time model-load / index-build / setup overhead.

This is a SCALED PROJECTION, not a measured re-run figure - treat it as an
order-of-magnitude budget. L4 throughput and FP8/spec load behavior on the 24 GB L4
(possible OOM forcing a Qwen3-4B fallback, or lowering `--gpu-memory-utilization` to
0.80) can shift wall-clock materially. Re-estimate after the smoke run.

---

## 6. The Fixes That Make This Run Clean

All fixes below are in the working tree AND in commit `51b68af`. They are grouped by
what each prevents.

### Blockers (5) - the things that, unfixed, invalidate or abort the run

- B1 (prevents BERTScore being globally nulled on valid data): BERTScore is cleared
  globally ONLY when a row with a non-empty reference (answerable SQuAD-v2 item) returns
  None, not merely because the ~52% unanswerable rows legitimately return None.
  `run_experiment.py:1479-1485`.
- B2 (prevents the compression 2x2 ACROSS-read confound): `compressed_rag` passes
  `--context-source retrieved` with `CAGE_REQUIRE_COMPRESSION=1`, so it compresses
  RETRIEVED context, not gold. `run_compression.sh:84-90`; engine honours it at
  `run_experiment.py:988-991`.
- B3 (prevents stale trial dirs shadowing a corrected result): `rerun_compressed_rag.sh`
  purges stale `trial_*/` before its single-trial rerun, because `statistical_tests`
  reads `trial_*/` with precedence. `rerun_compressed_rag.sh:41-42`. (Repair-only; not
  part of a fresh re-run.)
- B4 (prevents one bad query aborting a whole baseline): per-query try/except wraps
  prepare/generate/record so one failing query skips a single row; a stage-level crash
  flushes `results.partial.csv` before re-raising. `run_experiment.py:1251-1325, 1368-1382`.
- B5 / M8 (prevents a silently missing speculative cell): a failed server launch OR run
  writes a `STATUS=failed` sentinel, and the MiMo native draft uses `mimo_mtp`
  (overridable via `MIMO_MTP_CONFIG`). `run_speculative_matrix.sh:46-61, 79-93`.

### Majors (10) - correctness/telemetry/serving robustness

- M1 (prevents spec acceptance dropping to None): acceptance schema promotion handles
  both flat cage-stats keys and the nested spec_decode dict, promoting a canonical
  `spec_decode_acceptance_rate`; `_LAST` whitelist now includes the flat keys.
  `src/monitoring/vllm_telemetry.py:112-115, 176-181`.
- M2 (prevents acceptance None when cage-stats is absent): the telemetry sampler starts
  whenever `--vllm-telemetry` is passed, not gated on cage-stats, with a dependency-free
  `/metrics` scraper backstop. `run_experiment.py:1352-1363, 1422-1433`.
- M3 (prevents a malformed HTTP-200 body crashing the run): the vLLM adapter catches
  `ValueError` alongside `RequestException` on streaming and non-streaming paths, turning
  a truncated body into a recorded error row. `src/inference/vllm_adapter.py:174, 271`.
- M4 (prevents killing co-resident GPU users / stale spec server reuse): vLLM stop kills
  the v1 EngineCore worker, scoped GPU kill matches only vLLM cmdlines, and start reuses
  a running server ONLY when no speculative/KV lever is requested.
  `manage_vllm_server.sh:99-100, 197-222`.
- M5 (prevents a degenerate empty-output regression scoring as a valid 0): every row
  carries `empty_generation`, and a >5% empty rate triggers a loud WARNING.
  `run_experiment.py:1209, 1488-1492`. (Live risk: `stop=['\n']` is hard-set on every
  request at `run_experiment.py:1265,1298`; a leading newline truncates to empty.)
- M6 (prevents falling back to system python): `run_phase1.sh` activates the venv
  env-agnostically (`.venv` or `cage-env` or `../cage-env`).
  `run_phase1.sh:42-49`; same in `run_compression.sh:39-43`.
- M7 (prevents the distributed family OOMing the L4): `run_phase1.sh` defaults
  `ENABLE_DISTRIBUTED=0`; the distributed block is gated behind it.
  `run_phase1.sh:25, 158`; `cloud_run.sh:43`.
- M8: covered with B5 above (speculative STATUS sentinel + MiMo method selection).
- M9 (prevents lost logs from the lever scripts): `run_speculative.sh` sources
  `_log_guard.sh` for the continuous GCS mirror; same source in `run_compression.sh:20`,
  `rerun_compressed_rag.sh:17`, `run_speculative_matrix.sh:29`.
- httpx pin (prevents the in-process telemetry path failing when httpx arrives only
  transitively): `requirements.txt:39-42` pins `httpx>=0.27` directly.

### Minors / hardening

- Setup verify imports the EXACT telemetry path (`cage_stats.api.snapshot_dict` + pynvml
  init) so a missing telemetry dep is caught at setup. `setup_gpu_cloud.sh:79-97`.
- Strict-compression engine path: `CAGE_REQUIRE_COMPRESSION=1` makes a missing/failed
  LLMLingua RAISE instead of falling through to ratio 1.0.
  `src/orchestration/compression.py:64-68, 88, 114-115, 139`.
- No-mock enforced by opt-in env: telemetry mock requires `CAGE_TELEMETRY_MOCK=1` (off
  by default), metric models real unless `CAGE_DISABLE_LETTUCEDETECT=1` is explicitly
  set. A clean Phase-2 run sets NEITHER. `run_experiment.py:1360, 1407`;
  `src/evaluation/quality.py:130-132, 139-170`.
- Single-L4 VRAM guards: server caps `max_model_len` (8192 default, 4096 for the
  speculative/compression matrices), raises `gpu-memory-utilization` to 0.92, eager mode.
  `manage_vllm_server.sh:145-146, 151-154`; `run_speculative_matrix.sh:39-41`;
  `run_compression.sh:33-34`.

PROCESS DRIFT (not a code fix): there is NO single runnable live-infra / smoke / preflight
script in `scripts/`. Only `verify_results.py` (a POST-run validator) and
`check_fp8_prefix_cache.sh` (the FP8 gate inside `run_compression.sh`) exist. The
user-mandated live infra check (Gate 2 below) is policy executed MANUALLY. Authoring a
single preflight script that asserts all of Gate 2 and exits non-zero on any failure is
an open deliverable.

---

## 7. Pre-run Gates (nothing runs at full scale until these pass, in order)

- GATE 0 (env + bucket): activate `cage-env` on the VM; recreate
  `gs://cage-framework-cage-results` (so the sync has a live destination); confirm Redis
  is reachable (`cloud_run.sh` auto-starts `redis:7-alpine`). Leave `CAGE_TELEMETRY_MOCK`
  and `CAGE_DISABLE_LETTUCEDETECT` UNSET so telemetry and grounding are REAL.
- GATE 1 (setup verify, once after `setup_gpu_cloud.sh`): confirm the [5/5]
  telemetry-stack check passed - `pynvml.nvmlInit()` OK and
  `from cage_stats.api import snapshot_dict` OK. If `cage_stats.api` is not importable,
  export `CAGE_STATS_HOME=<cage-stats repo>` (`cloud_run.sh:48-49` auto-resolves
  `../cage-stats`); `httpx` + `prometheus_client` must import.
- GATE 2 (user-mandated LIVE infra check, before EVERY GPU run - assert each component
  live, no codified script exists):
  - (a) vLLM serving is REAL: GET `/health` is 200 and GET `/v1/models` lists the
    requested model.
  - (b) LettuceDetect (primary grounding) AND the NLI faithfulness model LOAD: score one
    real (context, answer) pair, confirming non-None `grounding_score` and `faithfulness`.
  - (c) cage-stats importable (Gate 1) so spec-decode/KV telemetry is live.
  - (d) retrieval is REAL: the FAISS index builds and search returns hits with a non-zero
    top-1 score on a sample query.
  - (e) NO mock anywhere: `CAGE_TELEMETRY_MOCK` unset/0.
  Abort the run if any of (a)-(e) fails.
- GATE 3 (smoke run, small `num_queries` on the SAME L4): execute the suite end-to-end at
  reduced scale and inspect artifacts before committing to 300x3. This single smoke pass
  carries gates 4-8.
- GATE 4 (smoke - speculative acceptance non-null): a speculative cell's
  `vllm_telemetry.json` has a non-null `spec_decode_acceptance_rate` (M1/M2 +
  `/metrics` backstop). If acceptance is None despite the cell running,
  `run_experiment.py:1435-1440` prints the explicit WARNING - treat as a gate failure.
- GATE 5 (smoke - BERTScore populated on answerable rows): in `no_cache`/`rag`
  `results.csv`, `completeness_bertscore` is NON-null on rows with a non-empty
  `reference_answer`; it legitimately stays None on unanswerable rows. A None on a
  row WITH a reference means the B1 global-clear fired because the model was unavailable -
  fix before the full run.
- GATE 6 (smoke - compressed_rag context is RETRIEVED not gold): confirm `compressed_rag`
  ran with `--context-source retrieved` and `CAGE_REQUIRE_COMPRESSION=1`, that rows show
  `compression_applied=True` with `compression_ratio < 1.0` (~0.5 at 2x) and
  `retrieved_doc_ids` populated, and that `prompt_tokens` dropped vs `rag_full`. A no-op
  must RAISE under strict mode - verify it does.
- GATE 7 (smoke - fault injection / row persistence): inject one mid-run exception and
  confirm (a) the offending row is skipped with a `[Measured] ... failed ... skipping`
  log, (b) the rest of the baseline completes, (c) a stage-level crash flushes
  `results.partial.csv` (B4). The run must NOT abort.
- GATE 8 (smoke - MiMo `mimo_mtp` live-validates on vLLM 0.11.0 and fits the 24 GB L4):
  launch the MiMo native-draft cell with `VLLM_SPECULATIVE_CONFIG='{"method":"mimo_mtp",...}'`
  (override via `MIMO_MTP_CONFIG` if 0.11.0 expects a different name) at
  `max_model_len 4096` / mem-util 0.90 / eager. Confirm the server starts (MTP head fits
  in VRAM) and serves a request; a rejected method writes `STATUS=failed reason=server`
  (B5/M8) - that is a HARD gate failure for the MiMo speculative 2x2, resolve before the
  full run. This is the single highest-risk unknown for Phase 2.
- GATE 9 (proceed to full run only after gates 0-8 pass): run the core suite first, then
  compression, then both speculative-matrix model runs, then stats, at 300 x 3.

---

## 8. End-to-End Execution Runbook

Hard ordering dependencies, called out up front:

- The bucket (step 2) and the SA grant (step 3) MUST exist before the VM runs anything,
  or all syncs 403 and fail-closed teardown strands the VM.
- The core suite (step 10) MUST complete and produce `no_cache` BEFORE stats (step 14),
  which hard-exits without it.
- The three lever scripts each own one vLLM on port 8000; run them STRICTLY sequentially
  (10 -> 11 -> 12 -> 13). Never run two in parallel (GPU/port conflict).
- MiMo speculative (step 13) runs AFTER Qwen (step 12) so they never share the GPU.

Steps run from the WORKSTATION unless marked (VM).

0. PRE-FLIGHT (workstation, read-only): `gcloud config get-value project` (confirm
   `cage-framework`); `gcloud compute instances list` (confirm `Listed 0 items.`). Set
   `HF_TOKEN` locally if any model is gated.
1. ONE-TIME GCP CONFIG: `gcloud config set project cage-framework` ;
   `gcloud config set compute/region us-central1` ;
   `gcloud config set compute/zone us-central1-a` ;
   `gcloud services enable compute.googleapis.com cloudresourcemanager.googleapis.com storage.googleapis.com`.
2. RECREATE THE BUCKET: `gsutil mb -l us-central1 gs://cage-framework-cage-results` ;
   `gsutil versioning set on gs://cage-framework-cage-results`.
3. GRANT THE VM SA OBJECTADMIN:
   `PROJ_NUM=$(gcloud projects describe cage-framework --format='value(projectNumber)')` ;
   `gsutil iam ch serviceAccount:${PROJ_NUM}-compute@developer.gserviceaccount.com:roles/storage.objectAdmin gs://cage-framework-cage-results`.
4. PROVISION THE ON-DEMAND L4 VM (NO spot flag):
   `gcloud compute instances create cage-gpu --zone=us-central1-a --machine-type=g2-standard-8 --accelerator=type=nvidia-l4,count=1 --maintenance-policy=TERMINATE --image-family=common-cu121-debian-11 --image-project=deeplearning-platform-release --boot-disk-size=200GB --boot-disk-type=pd-ssd --scopes=cloud-platform --metadata=install-nvidia-driver=True`.
5. SSH IN + GET THE CODE (repo MUST land at `~/CAGE`):
   `gcloud compute ssh cage-gpu --zone=us-central1-a` then (VM) `nvidia-smi` (wait if the
   driver is still installing) ; `git clone <repo-url> CAGE && cd CAGE`.
6. BOOTSTRAP (VM, from `~/CAGE`): `bash scripts/setup/setup_gpu_cloud.sh` (creates
   `cage-env`, installs pinned vLLM 0.11.0, requirements + cage-stats, force `openai>=2`,
   stages squad_v2/natural_questions/musique). Then `source cage-env/bin/activate`. Set
   `export HF_TOKEN=hf_xxx` if gated.
7. VERIFY DATA STAGED (VM, confirmation): `python scripts/download_datasets.py --dataset squad_v2`
   (idempotent).
8. LIVE INFRA CHECK (VM, Gate 2 - REQUIRED): confirm pynvml + `cage_stats.api` import,
   then `bash scripts/manage_vllm_server.sh restart Qwen/Qwen3-8B` ;
   `curl -fsS localhost:8000/health` ; `cage-stats --once` ; `redis-cli ping` ; confirm
   real metric models load and no mock. Then `bash scripts/manage_vllm_server.sh stop`.
9. SMOKE RUN (VM, tiny scale, carries gates 3-8 incl. MiMo `mimo_mtp` validation):
   `NUM_QUERIES=5 NUM_TRIALS=1 nohup bash scripts/cloud_run.sh Qwen/Qwen3-8B 5 1 > smoke.log 2>&1 &`
   ; `tail -f smoke.log`. Also smoke the MiMo spec server start:
   `NUM_QUERIES=5 NUM_TRIALS=1 bash scripts/run_speculative_matrix.sh XiaomiMiMo/MiMo-7B-RL`
   and confirm the mtp cell does NOT write `STATUS=failed reason=server`. Capture the
   exact working spec JSON (or `MIMO_MTP_CONFIG` override) here.
10. CORE SUITE (VM, full scale 300q x 3, nohup survives SSH drops, own sync loop):
    `nohup bash scripts/cloud_run.sh Qwen/Qwen3-8B 300 3 > run.log 2>&1 &` ; `tail -f run.log`.
    Produces `analysis/phase1/results/{no_cache,prefix_cache,rag,redis_retrieval_cache_cold,hybrid_retrieval_cache_cold,hybrid_retrieval_cache_warm}`.
    Wait for `[cage] suite complete`. (Canonical doc example is `100 10`; override to
    `300 3` per this plan.)
11. COMPRESSION 2x2 (VM, separate lever, FP8xprefix gate then llmlingua gate first;
    aborts if either fails):
    `NUM_QUERIES=300 NUM_TRIALS=3 nohup bash scripts/run_compression.sh Qwen/Qwen3-8B > compression.log 2>&1 &`
    ; `tail -f compression.log`. Produces
    `analysis/compression/results/{cag_full,rag_full,compressed_rag,compressed_cag}`.
12. SPECULATIVE MATRIX - Qwen3-8B (VM, each cell restarts the server):
    `NUM_QUERIES=300 NUM_TRIALS=3 nohup bash scripts/run_speculative_matrix.sh Qwen/Qwen3-8B > spec_qwen.log 2>&1 &`
    ; `tail -f spec_qwen.log`. Produces
    `analysis/speculative_matrix/spec_qwen8b_{ngram_cag,ngram_rag,eagle3_cag,eagle3_rag}`.
    Wait for `SPECULATIVE_MATRIX_DONE`.
13. SPECULATIVE MATRIX - MiMo-7B-RL (VM, AFTER Qwen; full-scale `mimo_mtp` exercise):
    `NUM_QUERIES=300 NUM_TRIALS=3 nohup bash scripts/run_speculative_matrix.sh XiaomiMiMo/MiMo-7B-RL > spec_mimo.log 2>&1 &`
    ; `tail -f spec_mimo.log`. Produces
    `analysis/speculative_matrix/spec_mimo7b_{ngram_cag,ngram_rag,mtp_cag,mtp_rag}`. A
    rejected mtp method writes `STATUS=failed` (loud), not a silent hole.
14. CONSOLIDATE + STATS (VM, REQUIRES `no_cache` from step 10 or exits 1):
    `nohup bash scripts/run_phase2_stats.sh > stats.log 2>&1 &` ; `tail -f stats.log`.
    Symlinks all three trees into `analysis/all_results`, runs `statistical_tests.py` vs
    `no_cache` (Wilcoxon + Holm + Cliff/bootstrap), writes
    `analysis/all_results/phase2_stats.{json,tex}`. Wait for `STATS_DONE`.
15. FINAL FULL COLLECT (VM, belt-and-suspenders): `bash scripts/sync_results_to_gcs.sh analysis`
    ; `bash scripts/collect_logs.sh` (full mode; writes a `COLLECT_OK` sentinel).
16. FAIL-CLOSED TEARDOWN (WORKSTATION, not the VM):
    `bash scripts/teardown_vm.sh cage-gpu us-central1-a`. Final-syncs `analysis/`,
    collect_logs with a UNIQUE token, verifies `COLLECT_OK_<token>` in
    `gs://cage-framework-cage-results/vm_logs/` (ABORTS if absent unless `--force`),
    deletes the instance, prints `TEARDOWN_COMPLETE`.
17. CONFIRM $0 (workstation): `gcloud compute instances list` must print `Listed 0 items.`
    ; optionally `gcloud storage ls gs://cage-framework-cage-results/analysis/ gs://cage-framework-cage-results/vm_logs/`
    to confirm durable results + logs. The bucket is intentionally KEPT.

Decision needed before steps 10-11 for MiMo: confirm whether MiMo-7B-RL also runs the
CORE suite + compression 2x2 (it is named a second primary model), or ONLY the
speculative matrix. The speculative labels are model-tagged and never collide, but core /
compression labels WOULD collide across models in `analysis/phase1/results` and
`analysis/compression/results`. Running both models through core/compression needs a
per-model output dir or a sequential model run with a results move.

---

## 9. Teardown and Cost Discipline

- Continuous preservation runs throughout: `cloud_run.sh` mirrors results + light logs
  every `SYNC_INTERVAL=120s` and on its EXIT/INT/TERM trap; the standalone lever scripts
  each `source _log_guard.sh`, which launches `log_sync_daemon.sh` and registers an EXIT
  trap doing a final full collect. An SSH drop or VM delete cannot lose a finished
  baseline.
- Fail-closed teardown to $0: `teardown_vm.sh` does a final results sync, then
  `collect_logs.sh` with a UNIQUE per-invocation token, then VERIFIES in GCS that
  `COLLECT_OK_<token>` exists before deleting the VM. Absent sentinel ABORTS (exit 1)
  unless `--force`. `collect_logs.sh` writes the sentinel as its LAST upload in a SECOND
  sync after the content sync, so the sentinel's presence proves the content arrived
  (robust against unreliable SSH stdout on teardown). `teardown_vm.sh:37-74`;
  `collect_logs.sh:98-114`.
- On-demand has no preemption, so `gcp_shutdown_hook.sh` (the ~30s SPOT ACPI backstop) is
  optional. Attaching it via `--metadata-from-file shutdown-script=...` is harmless cheap
  insurance against a manual stop, but is not required this run.
- Cost report obligation: on every cloud action, report cost + wall-clock. After teardown,
  record actual wall-clock per phase (core / compression / spec Qwen / spec MiMo / stats)
  and the GPU bill against the ~$8-15 projection. Confirm $0 (step 17). The bucket stays
  (cents/month); only the GPU VM is deleted to stop billing.

---

## 10. Open Items and Deliverables

### Hard gates / prerequisites (block the run)

- Recreate `gs://cage-framework-cage-results` (mb + versioning) and grant the VM SA
  `roles/storage.objectAdmin` BEFORE any run; the bucket is currently absent (live 0
  buckets). Confirm `teardown_vm.sh`'s `COLLECT_OK` sentinel path points at it.
- Lock the scale at 300 queries x 3 trials and pass it explicitly to `cloud_run.sh`
  (positional `Qwen/Qwen3-8B 300 3`), `run_compression.sh`, and
  `run_speculative_matrix.sh`. No script defaults to it.
- Live-validate MiMo `mimo_mtp` on stock vLLM 0.11.0 during the smoke run (Gate 8) and
  capture the exact working spec JSON or the `MIMO_MTP_CONFIG` override before the full
  matrix. Highest-risk unknown.
- Verify `check_fp8_prefix_cache.sh` passes on vLLM 0.11.0 before the compression run; it
  exits 2 (INCONCLUSIVE) on a server/flag error and `run_compression.sh:64` treats any
  non-zero as gate failure. Do NOT blindly `SKIP_GATE=1` - that re-introduces the
  compressed_cag (RQ5/H4) confound.
- Verify cage-stats resolves on the VM (the `requirements.txt` git dep is reachable, or
  set `CAGE_STATS_HOME`) so the richer KV/token-source telemetry is captured, not just
  the `/metrics` backstop.

### Decisions to confirm

- Lock the temperature sub-study (main matrix T=0 + separate {0,0.5,1.0} subset, k>=3 at
  T>0) so `RERUN_DESIGN.md:108-112` and `4_METHODOLOGY.tex:144` agree.
- Confirm whether MiMo-7B-RL also runs core + compression (not just speculative); resolve
  the cross-model label collision if so.
- Confirm `compressed_cag` stays `auto -> gold` (CAG side of the 2x2).
- Confirm the Phase-2 stats metric list - whether to add `hallucinated_span_ratio`
  (LettuceDetect primary) and `exact_match`.

### Repo hygiene (so nothing misleads the next operator)

- Fix the `cloud_run.sh:29` header to name `run_speculative_matrix.sh` (run once per
  model), not `run_phase5.sh`; same for `RUNBOOK.md:460` and `setup_gpu_cloud.sh:106`.
- Resolve the orphaned `scripts/run_phase2.sh` (delete or repoint to
  `analysis/phase1/results`) so it cannot silently produce results the consolidator
  ignores.
- Author a single runnable live-infra preflight that asserts Gate 2 (a)-(e) together and
  exits non-zero on failure, codifying the user-mandated check.
- Update `PHASE2_CHECKLIST.md` or mark it superseded: Spot -> on-demand (line 31),
  100x10 -> 300x3 (line 45), nine baselines -> fourteen result sets (lines 16, 68).

### Manuscript deliverables (feeding the dissertation)

- The Phase-2 analysis artifacts `phase2_stats.json` + `phase2_stats.tex` (against
  `no_cache`) feed the results and discussion chapters.
- Add `\cite{qwen3report}` and `\cite{mimo2025}` at the model introduction in
  `4_METHODOLOGY.tex:167`; add a LettuceDetect citation beside the primary grounding
  metric at `4_METHODOLOGY.tex:104-106`.
- Add the explicit "300 queries x 3 trials" count to `4_METHODOLOGY.tex` (currently only
  `7_CONCLUSIONS.tex:14`) and fix the stale "n=50 queries" caption at `6_RESULTS.tex:41`.
- Rebuild the bibliography once so the four newly added keys resolve
  (`RERUN_DESIGN.md:139-140`).
