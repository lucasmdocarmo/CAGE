# P0 integration dry-run report — 2026-08-02

Integration pass over the freshly built D8/D9 analysis stack (stats engine,
proof kit, goodput/Y module, figure pipeline, quality fail-closed tests),
followed by a dry-run on the pilot archive.

**Every number below is DESIGN INPUT ONLY** (THE WORK framing, 2026-07-27):
pilot-era results inform pre-registration choices (margins, tie handling,
power inputs) and are never citable as findings.

## 1. Test suite

Full suite: `.venv/bin/python -m pytest tests/ -q` → **477 passed, 13 skipped, 0 failed**.

Per new module (all green):

| Module | Tests | Result |
|---|---|---|
| stats engine (`tests/test_stats_engine.py`) | 53 | pass |
| stats proof kit (`tests/test_stats_proof.py`) | 36 | pass |
| goodput / Y (`tests/test_goodput.py`) | 63 (incl. 1 new, see §3) | pass |
| figure pipeline (`tests/test_figure_pipeline.py`) | 31 | pass |
| quality fail-closed (`tests/test_quality_failclosed.py` + no_answer + nli_windowing) | 84 | pass |

Skips: 5 router-integration + 7 vLLM-integration (need a live server), 1 data.

## 2. Environment repairs (no code changes)

The venv was missing dependencies that `requirements.txt` already declares;
the suite could not even collect. Installed into `.venv`:

- `aiohttp`, `requests` (declared in CAGE `requirements.txt`; unblocked
  `tests/test_vllm_integration.py` collection),
- `prometheus-client` plus the sibling `cage-stats/requirements.txt` set
  (`httpx`, …) so `tests/test_vllm_telemetry.py::test_bridge_resolves_via_cage_stats_home`
  can resolve the in-process bridge via `CAGE_STATS_HOME`.

## 3. One real bug found and fixed (goodput resolution band)

`src/analysis/goodput.py::_within_band` used `hi/lo <= resolution**2` with no
float tolerance. On an **exact ×1.15 geometric grid — the §6.1 registered grid
shape — a two-step knee bracket has `hi/lo == resolution²` mathematically, but
float rounding lands a few ulp above** (1.3225000000000002 > 1.3224999999999998),
so every knee on the registered grid would have been mislabeled
`INCONCLUSIVE_AT_RESOLUTION`. Fixed with a `1e-9` relative tolerance;
regression test added:
`tests/test_goodput.py::TestFindKnee::test_exact_geometric_grid_is_conclusive`.

## 4. Dry-run on the pilot archive

Runner: scratchpad script (`p0_dryrun.py`); outputs under the session
scratchpad `p0_out/` (`forest_grounding_squad_v2.png`, `wlt_f1_squad_v2.png`,
`forest_summary_grounding_squad_v2.csv`, `p0_dryrun_report.json`).

### 4.1 Loader + legacy mapping — PASS

`_results_loader.load_results_long` on all three pilot run roots
(`2026-07-16_full_qwen3-8b_100x3_{squad_v2,musique,hotpotqa}`): 20 cells,
6600 rows, 6600 valid each. **Every one of the 20 legacy cell names resolved
through `cellspec.from_legacy`** (baselines 6, compression 4, speculative 4,
envelope 5, kv_store 1); several intentionally alias to the same charter cell
(e.g. `cag_full`/`prefix_cache_*` → gold-reuse), so figure work used a
7-cell distinct-key subset.

Pairing-unit note: the pilot's 3 trials use **disjoint** 100-example sets
(300 unique `example_id`s per cell), so `per_example` pooling has nothing to
average across trials here — n_pairs below is 300, one row per example.

### 4.2 Paired Wilcoxon + W/L/T (squad_v2 unless noted) — PASS

| Contrast (arm vs control) | Metric | n | median Δ | p (two-sided) | W/L/T |
|---|---|---|---|---|---|
| prefix_cache vs no_cache | grounding_score | 213 | +0.0000 | 0.18 | 2/0/211 |
| prefix_cache vs no_cache | f1_score | 300 | +0.0000 | 0.593 | 2/1/297 |
| cag_true_on vs no_cache | grounding_score | 191 | +0.0000 | 0.655 | 11/10/170 |
| cag_true_on vs no_cache | f1_score | 300 | +0.0000 | 0.00276 | 32/56/212 |
| rag vs no_cache | grounding_score | 204 | +0.0000 | 0.677 | 8/6/190 |
| rag vs no_cache | f1_score | 300 | +0.0000 | 0.000346 | 15/47/238 |
| cag_true_on vs no_cache (musique) | f1_score | 300 | +0.0000 | 0.0314 | 26/11/263 |
| cag_true_on vs no_cache (hotpotqa) | f1_score | 300 | +0.0000 | 0.204 | 26/32/242 |

Reads exactly as the T=0 design predicts: dominant ties (median Δ = 0
everywhere), with the Wilcoxon signal carried by the discordant minority.
This confirms the D9 choice to lead with W/L/T + conditional analyses rather
than location shifts.

### 4.3 A/A calibration — PASS

`aa_split_half` on no_cache f1 (squad_v2, 300 obs, 200 seeded splits,
test_fn = `paired_wilcoxon` p): **fp_rate = 0.060, exact binomial CI
(0.031, 0.102), `approximates_nominal=True`** at α=0.05.

### 4.4 Figures — PASS

- `plot_forest` (grounding vs gold-fresh, 6 distinct-key cells re-keyed via
  `from_legacy(...).to_row_key()`): renders; per-row W/L/T annotations and
  Holm-within-panel columns present; all Holm-adjusted p = 1.0 (as expected
  from §4.2). Minor cosmetic nit: matplotlib's `1e-12` offset text overlaps
  the x-label when all deltas are ~0 — not actionable on real-effect data.
- `plot_win_loss_tie` (F1 vs gold-fresh, 6 contrasts): renders with correct
  counts (e.g. corpus-reuse 32/56/212).

### 4.5 Goodput / Y module — SYNTHETIC ONLY (by design)

The pilot is **closed-loop** — no open-loop arrival schedule and no per-request
timeliness/veridicality predicate columns — so `evaluate_window`, `find_knee`,
`find_cliff` **cannot run on the archive**. Synthetic smoke instead:

- `evaluate_window` (400 records, 120 s window): goodput_frac 0.935,
  yield_frac 0.797, truth_tax_frac 0.138, yield 2.676 rps — Y ≤ G and all
  identities hold.
- `find_knee` on a deliberately coarse mixed grid: `INCONCLUSIVE_AT_RESOLUTION`,
  bracket (6.0, 10.0) — the fail-closed label works.
- `find_knee` on an exact ×1.15 geometric grid: `ESTIMATED`, onset ≈ 7.00
  (post-fix §3; pre-fix this was wrongly inconclusive).
- `find_cliff`: `ESTIMATED` at the first retrograde point (10.0),
  bracket (8.0, 10.0).

## 5. Blocked / out of scope for this dry-run

- **Goodput on real data**: needs the work's open-loop rate-sweep harness
  (arrival schedule + per-request SLO/veridicality records). Pilot cannot
  provide it; first real exercise comes with the D6-grid runs.
- **qasper**: no pilot archive exists (loader is a known launch blocker);
  `KNOWN_DATASETS` includes it but nothing to dry-run against.
- **Grounding coverage**: grounding_score is null on ~30% of pilot rows
  (n = 191–213 of 300 per arm) — pilot-era scorer coverage, not a loader or
  stats-engine defect; the full quality module (D8 L0–L5) supersedes it.
- **Gatekeeping / TOST / prereg assembly on real contrasts**: not exercised
  here (no registered-margin inputs from pilot data by design); covered by
  their own unit tests.
