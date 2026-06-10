# CAGE Documentation Index

Status of every doc in this folder as of **2026-06-09** (post-flatten + post metric/Terraform
fixes). Start with the **canonical** docs; treat SUPERSEDED/STALE docs as history.

## ✅ Canonical — read these first

| Doc | What it is |
|---|---|
| [KNOWLEDGE_BASE.md](KNOWLEDGE_BASE.md) | Single AI-readable project reference: identity, architecture, baselines, metrics, datasets, infra, Phase-1 results, roadmap. |
| [RUNBOOK.md](RUNBOOK.md) | **Authoritative** setup / deploy / run — local + GCP, with the real CLI and durable GCS result persistence. Use this for any command. |
| [VALIDATION_AND_SOTA_REVIEW.md](VALIDATION_AND_SOTA_REVIEW.md) | Strict validation, limitations (what's fixed vs still-open), SOTA review, prioritized fix list. |
| [DATA_CARD.md](DATA_CARD.md) | Dataset sources, licenses, schema, FAISS index inventory. *(Rebuild e5 indices with `--rebuild-ir-index`.)* |
| [KV_FORMAT_SPEC.md](KV_FORMAT_SPEC.md) | Forward-looking KV-cache serialization spec (for the not-yet-built real KV transfer). |
| [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) | Deploy options overview. *(Caveat: the "distributed" baseline is simulated.)* |

## ⚠️ Partially stale — useful background, but verify against canonical

These carry a `CAGE-DOC-STATUS` banner. They predate the fixes and contain some **invalid
CLI flags** (`--phase`/`--all-baselines`/`--trials`/`--queries`), **pre-fix metric numbers**
(faithfulness 0.570, BERTScore 0.324), and/or old paths.

| Doc | Use it for | Don't trust |
|---|---|---|
| [GCP_DEPLOYMENT_RUNBOOK.md](GCP_DEPLOYMENT_RUNBOOK.md) | GCP quota/teardown narrative | its run commands → use RUNBOOK §9 |
| [PHASE_EXECUTION_GUIDE.md](PHASE_EXECUTION_GUIDE.md) | cost/runtime, MTU/GVNIC rationale | its CLI; main.tf edits are now tfvars |
| [METRICS_SPECIFICATION.md](METRICS_SPECIFICATION.md) | metric intent/history | its numbers + missing LettuceDetect → see KNOWLEDGE_BASE §5 |
| [PAPER_ARTIFACTS.md](PAPER_ARTIFACTS.md) | plot inventory | "verified" numbers predate metric fixes |
| [TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md) | module deep-dive | CLI flags + old metric description |
| [PROJECT_STATUS.md](PROJECT_STATUS.md) | short status | metric numbers + paths |
| [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) | local setup intent | vLLM source-build path → RUNBOOK §1 |
| [SOLUTION_DESCRIPTION.txt](SOLUTION_DESCRIPTION.txt) | architecture blurb | metric list (no LettuceDetect) |

## 🗄️ Superseded / historical — kept for the record only

Fully covered by KNOWLEDGE_BASE.md. Safe to delete if you want a lean `docs/`.

- [CAGE_PROJECT_MASTER_DOCUMENT.md](CAGE_PROJECT_MASTER_DOCUMENT.md) — also mis-expands the acronym ("Environment"; it's **Evaluation**).
- [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md), [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md), [PROJECT_TAKEOVER_CONTEXT.txt](PROJECT_TAKEOVER_CONTEXT.txt)
- [CONTINUATION.md](CONTINUATION.md), [CONTINUATION_GUIDE.md](CONTINUATION_GUIDE.md), [Continuation-2.md](Continuation-2.md)
- [LINUX_VM_MIGRATION_AND_PHASE2_PLAN.md](LINUX_VM_MIGRATION_AND_PHASE2_PLAN.md) — dead Parallels-VM premise.

## 🗑️ Deleted in this pass

- `ARCHITECTURE.md`, `CODE_WALKTHROUGH.md` — these described a **different project** entirely
  (a C++ high-frequency-trading engine), not CAGE. Removed as noise.

---

### Recommended lean doc set (if you prune)

Keep: `KNOWLEDGE_BASE.md`, `RUNBOOK.md`, `VALIDATION_AND_SOTA_REVIEW.md`, `DATA_CARD.md`,
`KV_FORMAT_SPEC.md`, `DEPLOYMENT_GUIDE.md`, this `README.md`. Optionally keep
`METRICS_SPECIFICATION.md` and `TECHNICAL_ARCHITECTURE.md` **after** updating them. Everything
in the "Superseded" list can be deleted.
