# CAGE — Cache-Augmented Generation Evaluation

A mechanism-attribution harness for context reuse in LLM serving: it measures, under
one controlled protocol, what KV-cache reuse (CAG), retrieval (RAG), and their hybrids
actually buy — serving performance (TTFT, latency, throughput, KV telemetry) *jointly*
with answer quality (grounding, faithfulness, abstention) — across engines, models,
and memory-pressure regimes.

## Why this exists

LLM serving benchmarks and quality benchmarks live in different worlds. Serving
frameworks (vLLM, SGLang, LMDeploy) and the stacks built on them compete on tokens
per second, TTFT, and inter-token latency; RAG and hallucination evaluators score
faithfulness on outputs collected in isolation, with no serving system under stress.
Neither measures the situation that actually breaks production deployments: the KV
cache running out of GPU memory while requests keep arriving. When that happens,
engines evict, preempt, recompute, and retract — and every one of those policies can
silently change *what the model answers*, not just how fast.

CAGE's objective is to measure that blind spot with one number and one attribution
story:

- **Serving yield (Y)** — requests that are *both* on time (SLO-bound) and truthful
  (a registered, abstention-aware correctness predicate), per second. Reported beside
  plain goodput G, the **truth tax** (G − Y: throughput paid for wrong answers), and
  an independence null, so a coupling between speed and truth is a measured effect,
  not an anecdote.
- **Mechanism attribution** — pressure is applied as an exact byte budget (verified
  against the engine's own startup logs), occupancy is recomputed from CAGE's own
  request accounting rather than trusted from engine gauges, and a layered quality
  ladder (retrieval adequacy → correctness → grounding → degradation taxonomy) traces
  each failure to the stage that caused it. The engine under test is never the referee
  of its own experiment.

The comparison spans four engines (vLLM, SGLang, LMDeploy, and an HF Transformers
correctness oracle that is never pressured), twelve context-management baselines
(fresh vs cache-reuse vs retrieval vs compression), eight datasets chosen to isolate
one property each, and both single-node and distributed topologies — including
tensor parallelism versus prefill/decode disaggregation at matched aggregate memory
and GPU count, so the cost of distributing the cache is itself a measured contrast.
The entire analysis is pre-registered (frozen contrasts, gatekeeping, power
simulation, blinded scoring) so the boring result publishes as credibly as the
exciting one.

> **Start here**
> - [`docs/RUNBOOK.md`](docs/RUNBOOK.md) — execution authority: setup → preflight →
>   run → sync → verified pull + teardown (RunPod-first; env contract table inside)
> - [`scripts/README.md`](scripts/README.md) — the script tree by lifecycle stage +
>   the campaign analysis chain
> - [`docs/RESULTS_LAYOUT.md`](docs/RESULTS_LAYOUT.md) — results tree spec v2
>   (cells, windows, sha256 ledger seal)
> - [`docs/VLLM_COMPATIBILITY.md`](docs/VLLM_COMPATIBILITY.md) — engine pins +
>   VERIFY-LIVE matrix
>
> The design authority (groups, arms, matrices, statistics) is the publication
> charter, `MyDocs/Publication/PUBLICATION.md` — an untracked working document until the
> registration freeze embeds it.

## Status (honest)

- The **charter campaign has not run yet.** Earlier CPU/L4 sweeps are **pilots**:
  their data is read-only under `results/` and informs design only — no pilot number
  is citable as a result.
- **RunPod is the primary cloud** (owner directive 2026-08-18); GCP support is a
  retained port (`terraform/gcp/`, `scripts/gcp/setup_gpu_cloud.sh`,
  `scripts/gcp/teardown_vm.sh`).
- The **campaign engineering is complete**: the CellSpec-native driver
  (`scripts/3_run/run_campaign.py`, plan→run split with pinned window counts), the
  sealed results producer (`src/orchestration/campaign_layout.py`), the distributed
  serving stack (TP pass-through, prefill/decode pair with NIXL transfer config), and
  the registered analysis chain are built and covered by a 3,000+ test offline suite.
  Legacy pilot runners under `scripts/3_run/` remain fenced as pilot-era in their
  headers.

## What a campaign run looks like

```bash
# on the pod (see docs/RUNBOOK.md for the full contract)
bash scripts/runpod/setup_runpod.sh                          # container-shaped bootstrap
source cage-env/bin/activate
export CAGE_BACKUP_TARGET=s3://<network-volume>[/prefix]      # J4: no backup target -> run refuses
bash scripts/checks/preflight_check.sh <MODEL> <API_BASE>     # gates (a)-(s); non-zero = do NOT launch

# campaign path: mint a reviewable plan first, then execute it
python3 scripts/3_run/run_campaign.py plan --session a --floor-table <floor.json> > plan.json
python3 scripts/3_run/run_campaign.py run  --plan plan.json                       # per-window resume; seals at end

# from the workstation, when the run is drained
scripts/runpod/teardown_pod.sh <pod_id> <backup_target> <local_run_dir>
#   -> ledger-gated pull FIRST, pod delete LAST, read-only $0 listing
```

## Campaign analysis chain

Pulled campaign trees (`results/<campaign>/<session>/<run_id>/`) flow through:

```bash
python3 scripts/4_analysis/verify_results.py   <run_root>   # schema/reconciliation/ledger gate
python3 scripts/4_analysis/organize_results.py <run_root>   # layout validation -> cells index
python3 scripts/4_analysis/run_campaign_analysis.py <run_root>   # the registered stats engine
```

Quality scoring is offline and decoupled (`scripts/4_analysis/rescore_quality.py`;
Instrument B via `scripts/4_analysis/score_instrument_b.py`); scoring passes never
write into sealed raw trees. Pilot-era analysis tools are kept runnable but refuse
campaign trees — see `scripts/README.md`.

## Repository layout

```
configs/       dataset / model / experiment configs
data/          dataset manifests (query/corpus builds are pinned by sha256)
docs/          execution docs: RUNBOOK, results-layout spec, engine compatibility
scripts/       operator scripts: lifecycle-numbered stages (1_setup ... 5_observability,
               checks/, lib/, ops/) + provider-only dirs gcp/ and runpod/
src/           the framework: analysis/ (cellspec, stats), data/, evaluation/,
               inference/ (engine adapters), monitoring/, observability/, orchestration/
terraform/     provider IaC — gcp/ holds the retained GCP-port stack (apply is
               approval-gated); RunPod uses runpodctl, no terraform
tests/         pytest suite (offline; fixtures replace GPUs and clouds)
results/       run data (gitignored; pilot trees are read-only design input)
```

## Development

```bash
# local venv (canonical interpreter; see requirements.txt pins)
.venv/bin/python -m pytest                    # the suite runs offline, no GPU needed
```

Every deployable script/source file must be tracked by git — the deploy artifact is a
`git archive` tarball with `BUILD_INFO` provenance (`scripts/ops/package_repo.sh`).

## Citation

```bibtex
@misc{carmo2026cage,
  title  = {CAGE: A Mechanism-Attribution Harness for Cache-Augmented Generation},
  author = {Carmo, Lucas Mariano do},
  year   = {2026},
  note   = {Pontif\'icia Universidade Cat\'olica de Minas Gerais},
}
```

## Contact

Lucas Mariano do Carmo — lucas.mariano.carmo@gmail.com
