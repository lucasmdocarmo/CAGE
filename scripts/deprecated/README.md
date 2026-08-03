# `scripts/deprecated/` — retired arms (kept for pilot forensics only)

**Do not run these against campaign data.** They are preserved so the pilot runs that used
them stay reproducible/auditable, and so their sentinel/gate patterns remain consultable.

| File | What it was | Why it is here |
|---|---|---|
| `run_speculative_matrix.sh` | Speculative-method x context-strategy 2x2 (ngram / eagle3 / mimo_mtp x CAG / RAG) for the pilot single-L4 sweeps | **Speculative arms are RETIRED** from the campaign design — charter §7.5 (Retirement list), `MyDocs/PUBLICATION.md`. The pilot EAGLE-3 quality numbers are additionally excluded (real vLLM early-EOS bug; see memory `cage-run-findings-2026-07-16`). |
| `check_mtp_spec_decode.sh` | Pre-flight gate asserting the native draft method actually speculates (`vllm:spec_decode_num_draft_tokens_total > 0`) instead of silently no-oping | Only consumer was `run_speculative_matrix.sh`; retired with it (charter §7.5). |

Authority: `MyDocs/PUBLICATION.md` (the publication charter) — §7.5 retires the speculative
arms; the pilot 9-name taxonomy these scripts drive is likewise retired (alias map:
`src/analysis/cellspec.py`).

`run_full_sweep.sh` no longer runs the speculative tree by default; pilot re-scoring can
opt back in with `CAGE_RUN_RETIRED_SPECULATIVE=1` (which calls the script at this path).

Rollback: these files were moved with plain `mv` from `scripts/3_run/` and
`scripts/checks/`; `git checkout -- <old path>` restores the pre-move copy.
