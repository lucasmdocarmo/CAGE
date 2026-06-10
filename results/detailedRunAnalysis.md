# Phase 1 Detailed Run Analysis

This document audits the completed corrected Phase 1 rerun using only the artifacts produced by the final run under `analysis/phase1/results`, the generated summaries in `analysis/phase1/plots`, and the execution log `logs/phase1_rerun_20260319_150430.log`.

It supersedes the older Phase 1 text analyses, which were written before the final semantics-hardening pass and before the new `analysis/phase1/results` layout.

## 1. Scope and evidence reviewed

Primary evidence:

- `logs/phase1_rerun_20260319_150430.log`
- `analysis/phase1/results/no_cache/aggregated_metrics.json`
- `analysis/phase1/results/rag/aggregated_metrics.json`
- `analysis/phase1/results/redis_retrieval_cache_cold/aggregated_metrics.json`
- `analysis/phase1/results/prefix_cache/aggregated_metrics.json`
- `analysis/phase1/results/hybrid_retrieval_cache_cold/aggregated_metrics.json`
- `analysis/phase1/results/hybrid_retrieval_cache_warm/aggregated_metrics.json`
- `analysis/phase1/results/distributed_router_replicated/aggregated_metrics.json`
- All `trial_*/metrics.json` and `trial_*/results.csv` files under `analysis/phase1/results/*`
- `analysis/phase1/plots/latest_metrics_summary.csv`
- `analysis/phase1/plots/pareto_optimal_baselines.csv`

Supporting code references:

- `scripts/run_phase1.sh`
- `scripts/run_experiment.py`
- `src/orchestration/router.py`
- `src/utils/prompting.py`

## 2. Completed execution structure

The corrected Phase 1 core suite now contains exactly these seven baselines:

1. `no_cache`
2. `rag`
3. `redis_retrieval_cache_cold`
4. `prefix_cache`
5. `hybrid_retrieval_cache_cold`
6. `hybrid_retrieval_cache_warm`
7. `distributed_router_replicated`

Notably absent from the final core suite:

- `distributed_sharded_sim`
- legacy unlabeled `redis`
- legacy unlabeled `hybrid`
- legacy distributed naming from the earlier provisional runs

## 3. Execution timeline from the final rerun log

Source: `logs/phase1_rerun_20260319_150430.log`

| Baseline | Started | Finished | Output directory |
|---|---|---|---|
| `no_cache` | Thu Mar 19 15:05:39 -03 2026 | Thu Mar 19 15:47:10 -03 2026 | `analysis/phase1/results/no_cache` |
| `rag` | Thu Mar 19 15:47:10 -03 2026 | Thu Mar 19 17:02:46 -03 2026 | `analysis/phase1/results/rag` |
| `redis_retrieval_cache_cold` | Thu Mar 19 17:02:46 -03 2026 | Thu Mar 19 18:16:25 -03 2026 | `analysis/phase1/results/redis_retrieval_cache_cold` |
| `prefix_cache` | Thu Mar 19 18:17:18 -03 2026 | Thu Mar 19 18:43:44 -03 2026 | `analysis/phase1/results/prefix_cache` |
| `hybrid_retrieval_cache_cold` | Thu Mar 19 18:44:37 -03 2026 | Thu Mar 19 19:29:43 -03 2026 | `analysis/phase1/results/hybrid_retrieval_cache_cold` |
| `hybrid_retrieval_cache_warm` | Thu Mar 19 19:30:38 -03 2026 | Thu Mar 19 20:50:32 -03 2026 | `analysis/phase1/results/hybrid_retrieval_cache_warm` |
| `distributed_router_replicated` | Thu Mar 19 20:52:11 -03 2026 | Thu Mar 19 21:40:06 -03 2026 | `analysis/phase1/results/distributed_router_replicated` |

The run therefore completed the full corrected suite from first baseline start to last baseline finish without interruption.

## 4. Artifact integrity verification

Observed artifact set:

- 7 baseline directories under `analysis/phase1/results`
- 21 trial directories total
- 21 `trial_*/metrics.json` files
- 21 `trial_*/results.csv` files
- 7 `aggregated_metrics.json` files at baseline root

Verification results:

- All 21 trial `metrics.json` files reported `performance.error_count = 0`
- A full CSV/JSON integrity recheck with a proper CSV parser found `0` row-count mismatches across all 21 trials
- Generated summary files were successfully produced from the completed results:
  - `analysis/phase1/plots/latest_metrics_summary.csv`
  - `analysis/phase1/plots/pareto_optimal_baselines.csv`

Conclusion:

- The run artifacts are real and internally consistent
- There is no evidence in the final Phase 1 core suite of hand-written summary substitution or fabricated aggregate files
- The benchmark outputs are genuine products of the current runner and current phase script

## 5. Semantic verification by baseline

### 5.1 `no_cache`

Observed behavior:

- 50 measured queries
- no warmup queries
- no retrieval stage
- no prompt-cache telemetry

Assessment:

- clean control baseline
- semantically straightforward
- best reference point for relative comparisons

### 5.2 `rag`

Observed behavior:

- 50 measured queries
- retrieval enabled
- aggregated retrieval hit rate `0.98`
- aggregated retrieval cache rate `0.0`

Assessment:

- real retrieval path
- no evidence of Redis or prompt-cache contamination
- retrieval is functioning, but the end-to-end latency cost is high on the current stack

### 5.3 `redis_retrieval_cache_cold`

Observed behavior:

- 50 measured queries
- retrieval enabled
- aggregated retrieval hit rate `0.98`
- aggregated retrieval cache rate `0.0`
- per-trial retrieval cache rate remained `0.0` for trials 1, 2, and 3

Assessment:

- the Redis flush fix appears to be working
- this baseline is now honestly a cold retrieval-artifact cache run, not a warm contaminated Redis run
- it should still not be described as raw KV-cache serving

### 5.4 `prefix_cache`

Observed behavior:

- 50 measured queries
- no warmup queries
- aggregated cached request rate `1.0`
- aggregated overall cached prompt ratio `0.6842`

Assessment:

- native prefix caching is active
- latency/TTFT improvements are real
- however, later trials inherited warmed prompt-cache state because the server was restarted per baseline, not per trial

### 5.5 `hybrid_retrieval_cache_cold`

Observed behavior:

- 50 measured queries
- no explicit warmup queries
- aggregated retrieval hit rate `0.98`
- aggregated retrieval cache rate `0.0`
- aggregated overall cached prompt ratio `0.7559`

Assessment:

- retrieval cache stayed cold as intended
- but the prompt-cache side did not remain fully cold across all trials
- trial 1 behaved as a truly cold retrieval + prefix-cache run, while trials 2 and 3 inherited warmed prefix state from the same server session
- this baseline is therefore only partially cold when interpreted at the three-trial aggregate level

### 5.6 `hybrid_retrieval_cache_warm`

Observed behavior:

- 50 measured queries
- 50 warmup queries excluded from metrics
- aggregated retrieval hit rate `0.98`
- aggregated retrieval cache rate `1.0`
- aggregated overall cached prompt ratio `0.8923`
- aggregated cached request rate `1.0`

Assessment:

- the measured-vs-warmup split is working as intended
- this is the cleanest current expression of the intended warm hybrid behavior
- unlike the old hybrid, it does not rely on a hidden extra measured workload; the warmup is explicit and excluded from the reported metrics

### 5.7 `distributed_router_replicated`

Observed behavior from `trial_1/metrics.json`:

- `sharding_policy = replicated`
- `topology = isolated_replicas`
- `distinct_api_bases = 3`
- `distinct_routed_replicas = ['replica-1', 'replica-2', 'replica-3']`
- `transfer_required_count = 0`
- `positive_transfer_latency_count = 0`
- `positive_transfer_bytes_count = 0`

Observed router snapshot from `trial_1/metrics.json`:

- `strategy = hash`
- `sharding_policy = replicated`
- `tokenizer_name = Qwen/Qwen3-4B`
- `tokenization_mode = model_tokenizer`
- replica distribution:
  - `replica-1 = 13`
  - `replica-2 = 15`
  - `replica-3 = 22`

Assessment:

- this baseline is a real router-mediated replicated multi-replica run
- the corrected replicated path is not injecting simulated transfer latency
- it is therefore materially cleaner than the older distributed family
- however, like `prefix_cache`, it also appears to benefit from cross-trial warmed prompt-cache state inside the same distributed session

## 6. What the final rerun verified successfully

### 6.1 Core-suite semantics are substantially cleaner than before

The rerun confirms that the implemented semantics-hardening changes are actually reflected in the final artifacts:

- baseline names match the intended corrected meanings
- the final suite is using `redis_retrieval_cache_cold` instead of ambiguous `redis`
- hybrid is split into explicit cold and warm runs
- the warm hybrid run excludes warmup requests from measured metrics
- the distributed baseline is `distributed_router_replicated`, not the simulated sharded-transfer baseline

### 6.2 Redis namespace flushing works

The strongest evidence is `redis_retrieval_cache_cold`, which shows:

- retrieval cache rate `0.0` in the aggregate
- retrieval cache rate `0.0` in all three trial metrics

That is the expected outcome for a true cold retrieval-cache baseline.

### 6.3 Warm hybrid behavior is explicit and measurable

The warm hybrid run shows:

- `experiment.num_measured_requests = 50`
- `experiment.num_warmup_requests = 50`
- `warmup.included_in_metrics = false`
- retrieval cache rate `1.0`
- prompt cached ratio `0.8923`

That is a much more defensible structure than the old mixed 150-request reporting.

### 6.4 The distributed replicated path is real and non-simulated

The distributed trial artifacts show:

- 3 actual replicas were used
- all 3 received traffic
- transfer was not required in the replicated policy
- router metadata includes a real tokenizer name and `model_tokenizer` mode

This is consistent with the intended “real replicated routing, no fake transfer” goal for the core suite.

## 7. Remaining methodological caveats discovered in the completed rerun

The rerun is much cleaner than the older Phase 1 attempts, but it is not yet perfectly publication-final.

### 7.1 Cross-trial prompt-cache carryover affects prefix-cache-enabled baselines

The phase script restarts the server or cluster per baseline, but `scripts/run_experiment.py` executes all three trials inside a single server session for that baseline. That means later trials can inherit prompt-cache state from earlier trials.

Evidence:

| Baseline | Trial 1 avg TTFT ms | Aggregate avg TTFT ms | Aggregate delta vs trial 1 | Trial 1 prompt cached ratio | Aggregate prompt cached ratio | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| `prefix_cache` | 2738.77 | 2375.60 | -13.26% | 0.6749 | 0.6842 | mild warming across trials |
| `hybrid_retrieval_cache_cold` | 12067.40 | 6000.61 | -50.27% | 0.4829 | 0.7559 | major carryover; aggregate is not fully cold |
| `distributed_router_replicated` | 6161.05 | 5278.74 | -14.32% | 0.6749 | 0.6842 | moderate warming across distributed trials |

The strongest issue is `hybrid_retrieval_cache_cold`:

- trial 1 TTFT: `12067.40 ms`
- trial 2 TTFT: `2954.87 ms`
- trial 3 TTFT: `2979.55 ms`
- retrieval cache remained cold across all three trials, but the prompt-cache ratio jumped from `0.4829` in trial 1 to `0.8923` in trials 2 and 3

Interpretation:

- the “cold” label is correct for retrieval-cache state
- it is not fully correct for aggregate prompt-cache state across all three trials

### 7.2 `prefix_cache` remains clearly better than `no_cache`, but the exact cold magnitude is somewhat overstated by aggregate averaging

Because trials 2 and 3 were not restarted from an empty prefix-cache state, the aggregate `prefix_cache` result is a “within-baseline warmed-trials average”, not a strict three-independent-cold-trials average.

This does not invalidate the direction of the finding. It does mean:

- the `prefix_cache` win is real
- the exact reported TTFT improvement should be interpreted as slightly optimistic for a strict repeated-cold-trials design

### 7.3 Provenance capture is still incomplete

Representative trial metadata still shows:

- `run_metadata.git_repository_present = false`
- `run_metadata.git_commit = null`
- `run_metadata.git_error = fatal: not a git repository (or any of the parent directories): .git`
- `run_metadata.backend.backend_version = null`
- `run_metadata.backend.backend_commit = null`
- `run_metadata.backend.model_id = null`

This is better than silent missing data because the git absence is now explicit, but it is still a reproducibility gap for publication packaging.

## 8. Trust assessment by baseline

| Baseline | Trust level | Reason |
|---|---|---|
| `no_cache` | High | clean control, no retrieval, no prompt-cache reuse |
| `rag` | High | retrieval behavior is stable and internally consistent |
| `redis_retrieval_cache_cold` | High | Redis flush worked; semantics are honest as retrieval-artifact caching |
| `prefix_cache` | Medium-high | strong real result, but later trials inherit warmed prefix state |
| `hybrid_retrieval_cache_cold` | Medium | retrieval is cold, but aggregate prompt-cache behavior is not fully cold |
| `hybrid_retrieval_cache_warm` | High | explicit warmup separation works as intended |
| `distributed_router_replicated` | Medium-high | real router/replica run with no simulated transfer, but later trials inherit warmed distributed prompt-cache state |

## 9. Bottom-line verdict on the run itself

The corrected Phase 1 rerun succeeded operationally and produced a real, coherent, and much cleaner artifact bundle than the earlier Phase 1 results.

What is now solid:

- the final baseline set is the corrected one
- the Redis cold semantics are working
- warm hybrid semantics are explicit and measurable
- the distributed replicated baseline is real and no longer relies on simulated transfer in the core suite
- all trial artifacts are internally consistent and error-free

What still blocks a fully final publication-grade interpretation:

- prompt-cache-enabled baselines are not isolated between trials
- therefore the aggregate `prefix_cache`, `hybrid_retrieval_cache_cold`, and `distributed_router_replicated` numbers include cross-trial warmed state
- reproducibility metadata is still incomplete

Practical conclusion:

- the run is valid and analyzable
- the artifact bundle is real
- the results are usable
- but the final report should explicitly caveat cross-trial prompt-cache carryover, especially for `hybrid_retrieval_cache_cold`
