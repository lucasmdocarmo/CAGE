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
- `<run_id>`: lowercase, bucket-name-safe (`[a-z0-9-]` only, since it names the GCS
  bucket): `YYYYMMDD-hhmm-<session>-<model-slug>`, e.g. `20260815-0230-a-qwen3-14b`.
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
arm|retriever|policy|topology|engine|model|family[|r<budget_r>][|lam<rate_frac>]
```

i.e. `"|".join([arm, retriever, policy, topology, engine, model, family])`, then, when
the pressure coordinates are set, `f"r{budget_r:g}"` and `f"lam{rate_frac:g}"` are
appended as extra `|`-separated parts. Examples:

```
gold-reuse|none|none|single|vllm|qwen3-14b|F1                       # B2 on the anchor, F1
retr-fresh|rerank|none|single|sglang|llama-3.3-70b|F2|r0.5|lam0.8   # B6 under pressure, F2
corpus-reuse|none|evict|single|lmdeploy|llama-3.3-70b|F3|r0.5|lam0.8
gold-fresh|none|none|pd|vllm|deepseek-v3|DIST                       # transfer pair, D rung
```

Rules:
- `CellSpec.__post_init__` is the validity gate — an illegal tuple cannot mint a
  directory (fail-closed at write time; loaders re-parse names with
  `CellSpec.from_flat_dict`/`to_row_key` round-trips).
- The `|` separator is legal in POSIX filenames and GCS object names; **always quote
  it in shells** (`'gold-reuse|none|...'`). Windows checkouts of the raw tree are
  unsupported.
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

## 4. Off-box mirror — identical tree, fresh destination per run (provider-neutral)

`<CAGE_BACKUP_TARGET>/results/<campaign>/<session>/<run_id>/...` — the remote tree
under the backup target (`gs://` | `s3://` | `ssh://` | `file://`, resolved by
`scripts/lib/transport.sh`) is **byte-identical in structure** to the local tree
(that is what lets `teardown_pod.sh` / `teardown_vm.sh` / `pull_run.sh` reconstruct
locally with a plain recursive copy). On the RunPod primary the target is normally the
network-volume S3 API (`s3://<volume>[/prefix]` + `CAGE_S3_ENDPOINT`); on the GCP port
it is the run's bucket `gs://cage-<run_id>`. Clean-room rule (RUNBOOK §0): one fresh
destination per run_id, created at provision (labeled `agent-run=<run_id>` where the
provider supports labels), deleted at TRUE-$0 teardown after the local pull + ledger
verify. The sync daemon (`scripts/5_observability/gcs_backup_daemon.sh`) mirrors
continuously during the run; no `--delete` is ever used.

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
