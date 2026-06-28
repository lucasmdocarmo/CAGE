# CAGE dissertation draft — validation notes

Generated June 20, 2026. Read this before using the draft. Nothing in your
existing files was modified or deleted; everything here is additive and lives in
`paper/dissertation-draft/`.

## Files delivered

- `CAGE_dissertation_draft.tex` — the dissertation draft, single document, `\chapter`/`\section`
  structure (report class), English. Chapters 1–7 follow your requested outline.
- `cage_dissertation_variations.tex` — 2–3 alternative phrasings for every
  narrative section (Intro opening, Problem, Hypotheses, Objectives, Specific
  Objectives, Contributions framing, Justification). Phrasing bank, not meant to
  compile alone.
- `cage-new-references.bib` — 6 new, peer-reviewed citations added to Related Work.
- `_article-references.clean.bib`, `_Dissertacao.clean.bib` — auto-generated
  sanitized COPIES of your two bibs (see "BibTeX issue" below). Your originals are
  untouched.
- `CAGE_dissertation_draft.pdf` — compiled output, 50 pages, builds with zero
  errors and zero undefined citations.

## Section-to-source mapping (provenance)

| Draft chapter | Built from |
|---|---|
| 1 Introduction, 1.1 Problem, Hypotheses, 1.2 Objectives, 1.3 Contributions, 1.4 Justification, 1.5 Organization | Expanded from `paper/my-cap1.tex` + `paper/updated-my-article.tex`; questions/hypotheses formalized from your central question in `my-cap1.tex` |
| 2 Background | `paper/my-cap1.tex` Sec. 1–2 (RAG/CAG, KV-cache survey) |
| 3 Related Work | `updated-my-article.tex` related work + 6 new published sources |
| 4 Methodology | Reconstructed from `docs/` (ROADMAP, PHASE_EXECUTION_GUIDE, METRICS_SPECIFICATION), `cloud_docs/` (KNOWLEDGE_BASE, VALIDATION_AND_SOTA_REVIEW), and `scripts/run_phase{2..5}.sh` |
| 5 CAGE (Present) | Adapted + improved from `updated-my-article.tex` Sec. 3 |
| 6 Results | Adapted from `updated-my-article.tex` Sec. 4, plus Threats-to-Validity and the forward phase programme |
| 7 Conclusions | Idea scaffold built on `updated-my-article.tex` Sec. 5 (you said you'll write this one) |

## CRITICAL integrity decision — quality numbers

Your `cloud_docs/KNOWLEDGE_BASE.md` (Sec. 5.3) and `VALIDATION_AND_SOTA_REVIEW.md`
state that the Phase-1 **faithfulness (~0.570)** and **BERTScore (~0.324)** values
were artefacts of metric bugs fixed in June 2026, and "must NOT be re-quoted as
results." To honour both "don't delete information" and "no false information,"
the draft:

1. Reports all **performance** results (latency, TTFT, throughput, tail latency,
   cache telemetry) as **validated Phase-1 findings** — these are robust.
2. Reports the **quality** numbers in `Table 6.4` explicitly labelled
   **"Preliminary instrumentation output"**, with a dedicated **Threats to
   Validity** section (6.6) naming each fixed bug (NLI sentence-pair input,
   hard-coded entailment label, binary claim gate, missing BERTScore rescaling),
   the gold-vs-retrieved **confound**, and the **simulated** (not measured)
   cross-node transfer.
3. Schedules the corrected-metric re-run as month 1 of the Methodology schedule.

This is defensible dissertation practice: it presents self-correction as a
strength. **If you re-run the corrected pipeline, replace Table 6.4 and update
Section 6.6.** Do not quote 0.570 / 0.324 as findings until then.

## New citations added (all peer-reviewed, 2024, web-verified June 2026)

| Key | Work | Venue |
|---|---|---|
| `gim2024promptcache` | Prompt Cache: Modular Attention Reuse | MLSys 2024 |
| `zheng2024sglang` | SGLang / RadixAttention | NeurIPS 2024 |
| `ye2024chunkattention` | ChunkAttention (prefix-aware KV) | ACL 2024 (Long) |
| `patel2024splitwise` | Splitwise (phase splitting) | ISCA 2024 |
| `yang2024crag` | CRAG benchmark | NeurIPS 2024 D&B |
| `agrawal2024sarathi` | Sarathi-Serve (chunked prefill) | OSDI 2024 |

Per your rule (2023+, no unpublished preprints in Related Work), Chapter 3 leans
on venue-published works. Page ranges were intentionally omitted from the new
`.bib` where they could not be confirmed word-for-word — add them from the venue
page if your style requires them. The full author list for CRAG is rendered as
"Yang et al." (`and others`) rather than fabricated.

## BibTeX issue found in your existing bib (act on this)

`paper/article-references.bib` contains comment lines such as
`% Formatted as @unpublished ...` and `(@Unpublished)`. **BibTeX does not treat
`%` as a comment** (only LaTeX does), so it tries to parse `@unpublished` inside
those comments and aborts, silently dropping ~30 entries. Your *article* never
hit this because `updated-my-article.tex` uses a hand-written
`\begin{thebibliography}` instead of bibtex. Your **abntex2 dissertation will hit
it.** Fixes, pick one:
- compile the dissertation with **biber** (tolerant), or
- delete/relocate the `%` comment lines that contain `@` or `(` characters, or
- use the sanitized copies provided here.

## Verification performed

- **Compile:** clean `pdflatex → bibtex → pdflatex ×2` in an isolated dir —
  0 bibtex errors, **0 undefined citations**, 50 pages.
- **Citations:** all 47 `\cite` keys resolve across the 3 bibs (checked programmatically).
- **Numbers:** every value in Tables 6.1–6.4 cross-checked against
  `analysis/phase1/results/*/aggregated_metrics.json` — exact match on latency,
  TTFT, QPS, and faithfulness for all 7 baselines.

## Open TODO for you

1. Run the corrected quality pipeline; replace the preliminary quality table.
2. Wire the real figure paths in Chapter 5–6 (placeholders/commented `\includegraphics`;
   assets exist under `analysis/companion_images/` and `paper/dissertation/en/imagem/`).
3. For final integration, move chapter bodies into your `paper/dissertation/en` tree
   and switch `\bibliographystyle` to the PUC/abntex2 style.
4. Confirm the proposed 6-month schedule (Table 4.1) against your real calendar.
5. Add page ranges to the 6 new `.bib` entries if your citation style mandates them.
