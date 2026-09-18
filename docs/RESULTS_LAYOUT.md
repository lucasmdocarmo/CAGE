# CAGE — Results Layout Spec v2 (campaign)

> **Why this file exists.** The pilots' hardest analysis bugs were *layout* bugs: six
> loaders with three None policies, plots and stats disagreeing in sign, a run synced
> to a bucket that didn't exist. "No problems later in analysis" hinges on ONE tree,
> ONE key format, ONE sealing rule — written down before the first campaign cell runs.
> Design authority for cell identity: `src/analysis/cellspec.py` (charter §7.1–§7.6.1).

---

## 1. The tree

```
results/<campaign>/<session>/<run_id>/
├── manifest.json                      # run provenance (§3) — written at run START
├── ledger.json                        # sha256 seal of every artifact (§5) — written at run END
├── cells/
│   └── <cellspec_row_key>/            # ONE directory per cell tuple (§2)
│       ├── cell.json                  # the CellSpec (to_flat_dict), baseline id (B1-B12),
│       │                              #   engine launch config, drive-manifest ref,
│       │                              #   windows[] table: k -> {dataset, seed, rep,
│       │                              #   budget_r, rate_frac, t_start, t_end}
│       └── window_<k>/                # one measurement window; k = <dataset>-<ordinal>
│           ├── requests.jsonl         # per-request records (our clock at the boundary)
│           ├── qa_evidence.jsonl      # raw outputs + evidence for OFFLINE scoring
│           ├── engine_metrics.json    # engine /metrics snapshots (before/after + samples)
│           └── cage_stats.jsonl       # cage-stats telemetry stream (policy events feed)
└── scoring/
    └── <scoring_run_id>/              # offline quality-scoring pass (§6) — NEVER writes
        └── ...                        #   into cells/; mirrors cells/<row_key>/ inside itself
```

- `<campaign>`: lowercase slug minted once per campaign. Current campaign: **`camp1`**
  (the PUBLICATION.md charter campaign). Pilot data is NOT migrated into it (§7).
- `<session>`: `a` | `b` | `cd-act1` | `cd-act2` (RUNBOOK §1.1).
- `<run_id>`: lowercase, destination-name-safe (`[a-z0-9-]` only — it names the run's
  fresh backup destination on EVERY scheme: the `s3://` prefix / volume path on the
  RunPod primary, the `gs://cage-<run_id>` bucket on the GCP port):
  `YYYYMMDD-hhmm-<session>-<model-slug>`, e.g. `20260815-0230-a-qwen3-14b`.
  Minted once at run start by the run wrapper; everything downstream reads it from the
  manifest, never re-derives it.
- `window_<k>` with **`k = <dataset_id>-<ordinal>`** (e.g. `window_squad_v2-01`,
  `window_musique-03`). Dataset ids: `squad_v2 · hotpotqa · musique · qasper · ruler ·
  scbench · sharegpt` (ShareGPT = load donor; its windows carry serving streams only).
  Putting the dataset in the window name is what makes the §8 dataset-scoped globs
  possible without opening any JSON.

## 2. Cell directory names = `CellSpec.to_row_key()`

Row keys come from `src/analysis/cellspec.py::CellSpec.to_row_key()` — never
hand-built. Exact format (quoting the implementation):

```
arm|retriever|policy|topology|engine|model|family[|r<budget_r>][|lam<rate_frac>][|cb<corpus_budget_tokens>]
```

i.e. `"|".join([arm, retriever, policy, topology, engine, model, family])`, then, when
the pressure coordinates are set, `f"r{budget_r:g}"` and `f"lam{rate_frac:g}"` are
appended as extra `|`-separated parts, and, on a B12 (corpus-trunc) rung cell only,
`f"cb{corpus_budget_tokens}"` (ADR-0106: one cell per rung of the descending corpus
ladder; the full 2,800 budget is B3's own cell and never a rung). Examples:

```
gold-reuse|none|none|single|vllm|qwen3-14b|F1                       # B2 on the anchor, F1
retr-fresh|rerank|none|single|sglang|llama-3.3-70b|F2|r0.5|lam0.8   # B6 under pressure, F2
corpus-reuse|none|evict|single|lmdeploy|llama-3.3-70b|F3|r0.5|lam0.8
corpus-trunc|none|none|single|vllm|qwen3-14b|F3|r0.5|lam0.8|cb700   # B12 at the 700-token rung
gold-fresh|none|none|pd|vllm|deepseek-v3|DIST                       # transfer pair, D rung
```

Rules:
- `CellSpec.__post_init__` is the validity gate — an illegal tuple cannot mint a
  directory (fail-closed at write time; loaders re-parse names with
  `CellSpec.from_flat_dict`/`to_row_key` round-trips).
- The `|` separator is legal in POSIX filenames and in GCS/S3 object names (any
  transport.sh backend); **always quote it in shells** (`'gold-reuse|none|...'`).
  Windows checkouts of the raw tree are unsupported.
- Axis vocabularies are mutually disjoint (no model name is ever an arm/engine/policy
  value), so a single-token glob like `*'|qwen3-14b|'*` is unambiguous (§8).
- Pilot-era baseline names (`no_cache`, `prefix_cache`, `hybrid`, ...) never appear in
  this tree — `cellspec.from_legacy()` translates them for re-keyed pilot *reads* only.

## 3. `manifest.json` (written at run start, amended never — a re-run is a new run_id)

Required fields:

| Field | Content |
|---|---|
| `run_id`, `campaign`, `session` | as in §1 (session includes the act for C/D) |
| `git_sha`, `git_dirty` | repo provenance; from git, else the tarball's `BUILD_INFO` (`scripts/ops/package_repo.sh`) |
| `engine`, `engine_version` | per engine actually launched (vLLM `0.19.1` pin etc.); for act 2 also the (vLLM, NIXL, UCX) triple |
| `model` | charter slug: `qwen3-14b · llama-3.3-70b · qwen3-next-80b · deepseek-v3` |
| `seed` | the campaign seed for this run |
| `provider` | neocloud name or `gcp` (+ zone/region) |
| `hardware` | machine shape, GPU SKU × count, per-node; act 2: both nodes + fabric |
| `dataset_manifests_sha256` | one sha256 over the dataset manifest files used (pins the exact query/corpus builds) |
| `cellspec_schema_version` | so a future axis change cannot silently re-key old data |
| `created_utc` | ISO-8601 |

## 3.1 Writers — who produces this tree (task #116)

The ONE production writer is **`scripts/3_run/run_experiment.py` in campaign mode**
(`--campaign-root`, or the `CAGE_CAMPAIGN_ROOT` env the shell runners export),
writing **via `src/orchestration/campaign_layout.py`** (`CampaignRun` /
`CellWriter.add_window` — atomic tmp+`os.replace` writes, fail-closed §2/§3
validation) with `src/orchestration/campaign_session.py` as the runner↔layout
bridge. No other code writes into a campaign tree's `cells/`.

- **One runner trial = one measurement window** `window_<dataset>-<NN>`
  (NN = trial number, `%02d`). The trial's per-request rows become
  `requests.jsonl` — every row carrying the #127 join triple
  (`example_id`/`repeat_index`/`record_index`, absent components explicit
  `null`) plus the shared `ok` validity predicate and the ADR-0007
  honesty/provenance columns; `qa_evidence.jsonl` is the staged evidence
  chain re-emitted atomically; the telemetry series becomes `cage_stats.jsonl`;
  backend metadata + the telemetry aggregate become `engine_metrics.json`.
- **`metrics.json` per window** (auxiliary artifact, indexed by
  organize_results like any extra window `*.json`): the runner's experiment
  summary — it is the **completeness sentinel** the shell resume gates
  (`cell_complete`, campaign branch) parse with `metrics_json_valid` rigor;
  a missing/unparseable one makes the window incomplete → reset + re-emitted.
- **`write_time_hashes.jsonl` at the run root** (beside `manifest.json`,
  never under `cells/`): the append-only §9.10 write-time hash journal —
  every emitted artifact is sha256-hashed at write time.
  **`scripts/3_run/seal_campaign_run.py`** (run once at run end, before any
  analysis) refuses to seal when any current sealed-scope artifact was never
  journaled or hashes differently from its last journal entry, then writes
  the one §5 `ledger.json` via `campaign_layout.seal_run`.
- **`.staging/` at the run root** (dot-entry: invisible to the layout
  walkers, the §5 seal, and the H7 EXTRA sweep): the pilot-format per-trial
  staging the runner still writes (crash-preserving incremental evidence);
  the sealed tree holds the canonical atomic copies.
- Shell drivers: `run_full_sweep.sh` / `cloud_run.sh` mint the campaign root
  (`mint_campaign_run_id` — `YYYYMMDD-hhmmss-<session>-<model-slug>`, a
  seconds-granular instance of the §1 grammar closing the J3 converge
  hazard; resume re-reads the exported `CAGE_RUN_ID`, never re-mints) and
  `run_baselines.sh` / `run_compression.sh` resolve every cell dir through
  `campaign_cell_dir` → `campaign_session cell-dir` (CellSpec-minted row
  keys — never hand-built in shell). Pilot labels that §7.5-merge onto one
  tuple (redis≈rag, hybrid warm≈cold, cag_full≡prefix_cache) dedup through
  the v2 resume gate: the second label sees the tuple's windows complete and
  skips.

Everything under `scoring/` is produced by `scripts/4_analysis/rescore_quality.py`
(§6, offline, decoupled); `predicate/` by `scripts/4_analysis/build_predicate_table.py`
(§8.5); `index/` by `scripts/4_analysis/organize_results.py` — all post-seal
siblings, never inside `cells/`.

## 4. Off-box mirror — identical tree, fresh destination per run (provider-neutral)

`<CAGE_BACKUP_TARGET>/results/<campaign>/<session>/<run_id>/...` — the remote tree is
**byte-identical in structure** to the local tree for ANY scheme
(`ssh://[user@]host/path`, `s3://bucket[/prefix]` + `CAGE_S3_ENDPOINT` on the RunPod
primary, `gs://bucket` on the GCP port, `file:///path` — resolved by
`scripts/lib/transport.sh`). It is written by
`scripts/5_observability/sync_results.sh` (the one-shot mirror every daemon and run
driver routes through; the interval daemon `gcs_backup_daemon.sh` — legacy name,
provider-neutral transports — loops it for the run's duration, never with `--delete`)
and pulled back by the **fail-closed** `scripts/5_observability/pull_run.sh` (ledger
gate). Byte-identical structure is what lets `teardown_pod.sh` / `teardown_vm.sh` /
`pull_run.sh` reconstruct locally with a plain recursive copy. On the RunPod primary
the target is normally the network-volume S3 API (`s3://<volume>[/prefix]` +
`CAGE_S3_ENDPOINT`); on the GCP port it is the run's bucket `gs://cage-<run_id>`.

**Clean-room rule — one FRESH destination per run (binding operator procedure,
RUNBOOK §0.2).** Give every run its own destination, named for the run — suffix the
target with the run id (e.g. `ssh://host/cage-runs/<run_id>/`,
`s3://<volume>/cage-<run_id>/`); on the GCP port `terraform/gcp/modules/bucket`
creates the per-run bucket `gs://cage-<run_id>` at provision (labeled
`agent-run=<run_id>`). Never reuse a past run's destination; delete it at TRUE-$0
teardown only after the local pull + ledger verify. What the CODE enforces at run
start is narrower than the rule: a backup target must **resolve or the run refuses to
launch** (`require_backup_target`, the J4 gate in `transport.sh`, called by
`cloud_run.sh` before any cell), and the sync daemon's `start` probes that the target
root **exists / is reachable** (`transport_ensure` — provisioned buckets are never
auto-created; ssh/file paths are `mkdir -p`'d). **No tool checks that the destination
is EMPTY**, so per-run freshness off the GCP-terraform path rests on the operator's
provision checklist: verify the destination is new (or empty) before launch.

## 5. `ledger.json` — the seal (implementation: `src/analysis/stats/ledger.py`)

- At run end, on the node, **BEFORE any analysis touches the data**: every artifact
  under `cells/` (plus `manifest.json`) is sha256-hashed (`hash_artifacts`, keys
  relative to the run root) and sealed with `write_ledger` → `ledger.json` at the run
  root. §9.10 UPGRADE 5: the data is provably untouched after this moment.
- `write_ledger` **refuses to overwrite** an existing seal — a re-run gets a new
  run_id, never a re-seal.
- The ledger carries a hash of its own entries; a tampered ledger *raises* instead of
  verifying.
- Verification (`verify_ledger(ledger_path, base_dir)`) re-hashes the tree and returns
  mismatch lines (`MISSING <relpath>` / `HASH-MISMATCH <relpath> ...`); empty = intact.
  It runs at minimum: (a) on the node right after sealing, (b) **locally after the
  pull, before teardown** (RUNBOOK §5 end sequence [2], `teardown_pod.sh` step
  [2/5] — ledger-gated `pull_run.sh`, fail-closed), (c) at analysis load.

## 6. Scoring runs — reruns never touch raw trees

Quality scoring is offline and decoupled (the pilots' hardest-won lesson). Each pass:

- Gets its own `scoring/<scoring_run_id>/` (e.g. `s01-lettucedetect-nli`), containing a
  `scoring_manifest.json` (scorer model ids + versions, code SHA, the raw-run ledger's
  `entries_sha256` it scored against) and per-cell outputs mirroring
  `cells/<row_key>/window_<k>/` → `qa_scores.jsonl`, `quality.json`.
- **Never writes into `cells/`** — the raw tree is sealed (§5); a scoring bug is fixed
  by a NEW scoring_run_id, and old passes are kept (comparability is a feature).
- Scoring passes get their own ledger inside their own directory before being used by
  stats.

## 7. Pilot-era data — read-only historical

`results/phase2/<run-id>/{baselines,compression,speculative,...}` (the 2026-07-14
convention) stays exactly where it is, read-only, served by
`scripts/4_analysis/_results_loader.py` (ONE parser, ONE validity rule, ONE estimand).
Pilot numbers are never cited as campaign results (THE-WORK framing); when a pilot
comparison is needed, `cellspec.from_legacy()` re-keys pilot names to charter tuples at
read time. No pilot data is copied into `results/camp1/`.

## 8. Query patterns the layout guarantees (the analysis contract)

All patterns are pure path globs — no JSON opened, no directory walked twice. Quote
the `|`s.

| Question | Pattern |
|---|---|
| All cells for model X (whole campaign) | `results/camp1/*/*/cells/*'|'qwen3-14b'|'*` — one glob (axis vocabularies are disjoint, §2) |
| All cells for one run | `results/camp1/<session>/<run_id>/cells/*` |
| All windows for contrast **B6 vs B3** on dataset Y (e.g. anchor, vLLM, F1) | two globs: `.../cells/'retr-fresh|rerank|none|single|vllm|qwen3-14b|F1'/window_<Y>-*` and `.../cells/'corpus-reuse|none|none|single|vllm|qwen3-14b|F1'/window_<Y>-*` |
| One family's pressure grid for an engine | `.../cells/*'|'sglang'|'*'|F2|'r*` (the `r`-coordinate suffix only exists on pressure cells) |
| Every scored quality file from scoring pass S | `.../scoring/<S>/cells/*/window_*/qa_scores.jsonl` |
| Raw-vs-scored join | same relative path under `cells/` and `scoring/<S>/cells/` — join key is (row_key, window k), by construction |

Invariants behind the guarantees:
1. One directory per cell tuple; the tuple IS the name (no name→tuple lookup table to
   drift).
2. Dataset is in the window name; run/session/campaign are path levels — so every
   per-run / per-model / per-dataset slice is a glob, never a scan.
3. Raw trees are append-only until sealed, then immutable (§5); scoring is additive
   under `scoring/` (§6) — so no analysis rerun can invalidate another's inputs.
4. Local tree ≡ remote tree (§4) — any pattern above works with the backup target
   (e.g. `s3://<volume>/` or `gs://cage-<run_id>/`) prefixed, unchanged.

## 9. `results/ops/` — LOCAL-ONLY operator metadata (not results)

`results/ops/pod_ledger.jsonl` is the pod-ops ledger of the RunPod tooling
(`scripts/runpod/provision_pod.sh` appends `create` events, `teardown_pod.sh` the
matching `delete` events; `pod_status.sh` / `cost_report.sh` consume the pairs; path
override `CAGE_POD_LEDGER`). It sits **beside** the campaign roots
(`results/ops/`, next to `results/<campaign>/` and the pilot `results/<phase>/`) and
is **never part of any run tree**:

- **never synced off-box** — the mirror/pull tools operate on run and phase trees
  (`results/<campaign>/...`, `results/<phase>/...`), which do not contain `ops/`.
  Do not point `sync_results.sh` at bare `results/` (that would drag `ops/` along);
- **never sealed** — no `ledger.json` covers it; it is append-only local
  bookkeeping of pod cost/lifecycle, useless to reproduce and cheap to keep;
- **ignored by the analysis chain** — `verify_results.py` and
  `organize_results.py` take ONE explicit run root and walk only inside it, so
  `results/ops/` is structurally invisible to them (pointing either AT it refuses
  fail-closed on the missing `manifest.json`).
