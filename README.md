# CAGE — Cache-Augmented Generation Evaluation

A mechanism-attribution harness for context reuse in LLM serving: it measures, under
one controlled protocol, what KV-cache reuse (CAG), retrieval (RAG), and their hybrids
actually buy — serving performance (TTFT, latency, throughput, KV telemetry) *jointly*
with answer quality (grounding, faithfulness, abstention) — across engines, models,
and memory-pressure regimes.

> **Start here**
> - [`cloud/RUNBOOK.md`](cloud/RUNBOOK.md) — execution authority: setup → preflight →
>   run → sync → verified pull + teardown (RunPod-first; env contract table inside)
> - [`scripts/README.md`](scripts/README.md) — the script tree by lifecycle stage +
>   the campaign analysis chain
> - [`cloud/RESULTS_LAYOUT.md`](cloud/RESULTS_LAYOUT.md) — results tree spec v2
>   (cells, windows, sha256 ledger seal)
> - [`cloud/VLLM_COMPATIBILITY.md`](cloud/VLLM_COMPATIBILITY.md) — engine pins +
>   VERIFY-LIVE matrix
>
> The design authority (groups, arms, matrices, statistics) is the publication
> charter, `MyDocs/PUBLICATION.md` — an untracked working document until the
> registration freeze embeds it.

## Status (honest)

- The **charter campaign has not run yet.** Earlier CPU/L4 sweeps are **pilots**:
  their data is read-only under `results/` and informs design only — no pilot number
  is citable as a result.
- **RunPod is the primary cloud** (owner directive 2026-08-18); GCP support is a
  retained port (`terraform/`, `scripts/1_setup/setup_gpu_cloud.sh`,
  `scripts/6_teardown/teardown_vm.sh`).
- The campaign results producer (`src/orchestration/campaign_layout.py`: manifest,
  cell/window tree, run-end ledger seal) is built and tested; the CellSpec-native
  campaign driver that wires it into the run loop is in progress. The runnable
  `scripts/3_run/` harness is pilot-era and fenced as such in its headers.

## What a campaign run looks like

```bash
# on the pod (see cloud/RUNBOOK.md for the full contract)
bash scripts/1_setup/setup_runpod.sh                          # container-shaped bootstrap
source cage-env/bin/activate
export CAGE_BACKUP_TARGET=s3://<network-volume>[/prefix]      # J4: no backup target -> run refuses
bash scripts/checks/preflight_check.sh <MODEL> <API_BASE>     # gates (a)-(p); non-zero = do NOT launch
nohup bash scripts/3_run/run_full_sweep.sh <MODEL> <N> <T> > sweep.log 2>&1 &

# from the workstation, when the run is drained
scripts/6_teardown/teardown_pod.sh <pod_id> <backup_target> <local_run_dir>
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
cloud/         execution docs: RUNBOOK, results-layout spec, engine compatibility
configs/       dataset / model / experiment configs
data/          dataset manifests (query/corpus builds are pinned by sha256)
scripts/       lifecycle-numbered operator scripts (1_setup ... 6_teardown, checks/, lib/, ops/)
src/           the framework: analysis/ (cellspec, stats), data/, evaluation/,
               inference/ (engine adapters), monitoring/, observability/, orchestration/
terraform/     GCP port infrastructure (retained; apply is approval-gated)
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
