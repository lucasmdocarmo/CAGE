#!/usr/bin/env python3
"""Offline quality re-scorer: re-run QualityEvaluator over saved qa_evidence.jsonl.

Why this exists (2026-07-15 audit): metric fixes (abstention regex, abstention-aware
grounding) must apply RETROACTIVELY to completed runs without a GPU or a re-run. The
served context is persisted only in qa_evidence.jsonl -- results.csv does not carry it --
so this is the one artifact a grounding re-score can work from.

Modes:
  default (--fast): NO model loads. Re-scores the model-free metrics only (F1/EM,
      abstention decomposition incl. the new abstention_precision) and applies the
      abstention short-circuit (grounding/faithfulness/completeness -> None on abstained
      rows). Runs anywhere, seconds for a smoke run.
  --full: loads the full metric stack (LettuceDetect/NLI/BERTScore/embeddings) and
      re-scores everything. Needs the ML deps; use --device cuda on a GPU box.

Output layouts:
  LEGACY (default, pilot trees): one results_rescored.csv next to each
      qa_evidence.jsonl (same trial dir), with example_id/baseline/trial provenance,
      the fresh QualityMetrics columns, and old_grounding_score copied from the
      evidence for before/after comparison. Never overwrites results.csv.
  SCORING TREE (--scoring-run-id, campaign v2 trees): cloud/RESULTS_LAYOUT.md §6 —
      writes scoring/<scoring_run_id>/ under the run root with a
      scoring_manifest.json (scorer ids+versions, code SHA, the raw-run ledger's
      entries_sha256 it scored against), per-cell outputs mirroring
      cells/<row_key>/window_<k>/ as qa_scores.jsonl + quality.json, and its OWN
      content-hash ledger sealed before stats may consume it. NEVER writes into
      cells/ — the raw tree is sealed (§5); a scoring bug is fixed by a NEW
      scoring_run_id. Refuses an existing scoring_run_id, an unsealed run
      (missing ledger.json), and the --apply flag (which mutates cells/).

Usage:
  python3 scripts/4_analysis/rescore_quality.py --run-root results/phase2/<run-id>
  python3 scripts/4_analysis/rescore_quality.py --run-root <run> --full --device cuda
  python3 scripts/4_analysis/rescore_quality.py --run-root results/camp1/a/<run_id> \
      --scoring-run-id s01-fast
  python3 scripts/4_analysis/rescore_quality.py --run-root <run> --full \
      --device cuda --batch-size 32   # cross-row batched scoring (D8 §8.1)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

#: RESULTS_LAYOUT §6 scoring-run-id grammar (path-safe lowercase slug, e.g.
#: ``s01-lettucedetect-nli``) — same character class as the §1 run_id grammar.
SCORING_RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,40}$")
SCORING_DIRNAME = "scoring"
SCORING_MANIFEST_NAME = "scoring_manifest.json"
QA_SCORES_NAME = "qa_scores.jsonl"
QUALITY_JSON_NAME = "quality.json"


def _positive_int(value: str) -> int:
    """argparse type for --batch-size: fail closed on anything below 1."""
    iv = int(value)
    if iv < 1:
        raise argparse.ArgumentTypeError(f"batch size must be >= 1, got {value!r}")
    return iv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-score saved qa_evidence.jsonl offline.")
    p.add_argument("--run-root", required=True,
                   help="Run root (results/<phase>/<run-id>) or any dir containing "
                        "trial_*/qa_evidence.jsonl at any depth.")
    p.add_argument("--full", action="store_true",
                   help="Load the full metric stack (LettuceDetect/NLI/BERTScore). "
                        "Default is fast mode: model-free metrics + abstention short-circuit only.")
    p.add_argument("--device", default="cpu", help="Device for --full mode (cpu|cuda).")
    p.add_argument("--out-name", default="results_rescored.csv",
                   help="Output CSV filename written next to each qa_evidence.jsonl "
                        "(legacy layout only).")
    p.add_argument("--apply", action="store_true",
                   help="Also merge the re-scored quality columns back into each trial's "
                        "results.csv (one-time backup at results.csv.pre_rescore). Fast "
                        "mode applies only the model-free fields plus the abstention "
                        "short-circuit (model metrics -> blank on abstained rows); "
                        "--full applies every quality column. This is the post-serving "
                        "scoring step of decoupled mode (run_experiment --skip-quality). "
                        "LEGACY layout only — incompatible with --scoring-run-id.")
    p.add_argument("--scoring-run-id", default=None,
                   help="Campaign v2 mode (cloud/RESULTS_LAYOUT.md §6): write "
                        "scoring/<scoring_run_id>/ under the run root (manifest + "
                        "per-cell qa_scores.jsonl/quality.json + own ledger) instead "
                        "of beside-the-evidence CSVs. Never writes into cells/.")
    p.add_argument("--batch-size", type=_positive_int, default=None, dest="batch_size",
                   help="Cross-row batched scoring (charter D8 §8.1; review §4.6 L3): "
                        "route each evidence file through "
                        "QualityEvaluator.batch_evaluate(batched=True), forwarding "
                        "this value as the NLI pipeline batch size. Default (unset) "
                        "preserves the historical sequential row-by-row behavior "
                        "(batched=False); the two paths are output-equivalent "
                        "(proven in tests/test_rescore_wiring.py).")
    return p.parse_args()


# Model-free fields: recomputed identically in fast and full mode, always safe to apply.
# sanitized_answer (B4) is model-free: the scaffold-strip/truncation regexes run everywhere.
_MODEL_FREE_FIELDS = [
    "f1_score", "precision", "recall", "exact_match", "is_answerable",
    "predicted_no_answer", "f1_answerable", "exact_match_answerable",
    "no_answer_correct", "abstention_precision", "sanitized_answer",
]
# Model-based fields: in fast mode these are None because the models are OFF, which must
# NOT clobber real values -- applied only on abstained rows (the short-circuit fix) unless
# --full recomputed them for real. faithfulness_premise_mode (B2) and the *_source dual
# scores (B3d, vs pre-compression originals) ride with the model-based group.
_MODEL_FIELDS = [
    "grounding_score", "hallucination_detected", "hallucinated_span_ratio",
    "supported_claim_ratio", "faithfulness", "faithfulness_premise_mode",
    "context_relevance", "relevance",
    "completeness_bertscore", "completeness_rouge_l",
    "faithfulness_source", "grounding_source",
]


def _fmt_cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v)
    return str(v)


def _apply_to_results_csv(trial_dir: Path, rows_out: list, full_mode: bool) -> int:
    """Merge re-scored quality columns into trial_dir/results.csv. Returns rows updated."""
    csv_path = trial_dir / "results.csv"
    if not csv_path.is_file():
        return 0
    backup = trial_dir / "results.csv.pre_rescore"
    if not backup.exists():
        backup.write_bytes(csv_path.read_bytes())

    by_key = {(r["example_id"], str(r.get("repeat_index") or "0")): r for r in rows_out}
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        csv_rows = list(reader)

    for col in _MODEL_FREE_FIELDS + _MODEL_FIELDS:
        if col not in fieldnames:
            fieldnames.append(col)

    updated = 0
    for row in csv_rows:
        err = (row.get("error") or "").strip().lower()
        if err not in ("", "none", "false", "0"):
            continue  # errored rows stay nulled at source
        key = (row.get("example_id"), str(row.get("repeat_index") or "0").strip() or "0")
        src = by_key.get(key)
        if src is None:
            continue
        for col in _MODEL_FREE_FIELDS:
            if col in src:
                row[col] = _fmt_cell(src.get(col))
        if full_mode or src.get("abstained"):
            for col in _MODEL_FIELDS:
                if col in src:
                    row[col] = _fmt_cell(src.get(col))
            # grounded flag mirrors run_experiment.py's None-aware rule
            g = src.get("grounding_score")
            if "grounded" in row:
                row["grounded"] = "" if g is None else str(float(g) >= 0.5)
        updated += 1

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return updated


def _score_evidence_file(
    ev_path: Path, evaluator: Any, batch_size: int | None = None
) -> list[dict[str, Any]]:
    """Score every record of one qa_evidence.jsonl; shared by both layouts.

    Behavior is byte-identical to the historical inline loop: B4 sanitized
    abstention gate, M5 all-answers max-over-golds, B3d dual scoring against
    pre-compression originals when the evidence carries them.

    Scoring routes through ``QualityEvaluator.batch_evaluate`` (D8 §8.1;
    review 2026-08-04 §4.6 L3). ``batch_size=None`` (default) passes
    ``batched=False``, which by quality.py's contract IS the historical
    per-row ``evaluate()`` loop; an integer passes ``batched=True`` with the
    value forwarded as ``nli_batch_size``, enabling the cross-row batched
    model calls. Both paths yield identical rows (tests/test_rescore_wiring.py).
    """
    from src.evaluation.quality import is_no_answer_prediction, sanitize_answer

    # Phase 1: parse every record (tolerant parsing unchanged from the old loop).
    records: list[dict[str, Any]] = []
    for line in ev_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        question = rec.get("question") or ""
        contexts = rec.get("used_contexts") or []
        if isinstance(contexts, str):  # tolerate stringified lists from older runs
            try:
                contexts = json.loads(contexts)
            except json.JSONDecodeError:
                contexts = [contexts]
        generated = rec.get("generated_answer") or ""
        reference = rec.get("reference_answer") or ""
        # ALL gold answers (audit 2026-07-16 M5): newer qa_evidence rows carry the
        # deduplicated gold list so F1/EM use the official max-over-golds; older
        # evidence files lack the field -> None -> single-reference fallback.
        all_answers = rec.get("all_answers")
        if not isinstance(all_answers, list):
            all_answers = None
        records.append({
            "rec": rec,
            "question": question,
            "contexts": list(contexts),
            "generated": generated,
            "reference": reference,
            "all_answers": all_answers,
        })
    if not records:
        return []

    # Phase 2: ONE scoring call for both modes.
    metrics_list = evaluator.batch_evaluate(
        [r["question"] for r in records],
        [r["contexts"] for r in records],
        [r["generated"] for r in records],
        [r["reference"] for r in records],
        all_answers=[r["all_answers"] for r in records],
        batched=batch_size is not None,
        nli_batch_size=batch_size if batch_size is not None else 32,
    )

    rows_out: list[dict[str, Any]] = []
    for r, quality_metrics in zip(records, metrics_list):
        rec = r["rec"]
        generated = r["generated"]
        reference = r["reference"]
        question = r["question"]
        metrics = quality_metrics.to_dict()

        # B4: abstention is judged on the SANITIZED text ("A: I don't know." must
        # count), matching evaluate()'s internal gate. sanitized_answer also lands
        # in metrics via QualityMetrics.to_dict(); generated_answer stays raw.
        sanitized = sanitize_answer(generated)
        abstained = is_no_answer_prediction(sanitized)

        # B3d dual scoring: when the evidence row carries the PRE-COMPRESSION docs
        # ('original_contexts', written by the serving side for compressed arms),
        # score faithfulness/grounding against those originals ALONGSIDE the
        # served-context scores. Separates "the answer contradicts the source" from
        # "compression destroyed the evidence the answer relied on". Columns are
        # ALWAYS emitted; empty when the field is absent, the row abstained, or the
        # metric models are off (fast mode).
        faithfulness_source = None
        grounding_source = None
        original_contexts = rec.get("original_contexts")
        if isinstance(original_contexts, str):
            try:
                original_contexts = json.loads(original_contexts)
            except json.JSONDecodeError:
                original_contexts = [original_contexts]
        if (
            isinstance(original_contexts, list)
            and any(c and str(c).strip() for c in original_contexts)
            and not abstained
        ):
            src_ctx = [str(c) for c in original_contexts if c and str(c).strip()]
            faith_src = evaluator.evaluate_faithfulness(sanitized, src_ctx)
            faithfulness_source = faith_src.get("faithfulness")
            halluc_src = evaluator.evaluate_hallucination(question, src_ctx, sanitized)
            grounding_source = halluc_src.get("grounding_score")

        old_g = rec.get("grounding_score")
        old_g = None if old_g in (None, "", "None") else float(old_g)

        cell = rec.get("baseline") or ev_path.parent.parent.name

        rows_out.append({
            "example_id": rec.get("example_id"),
            "baseline": cell,
            "trial_dir": ev_path.parent.name,
            "repeat_index": str(rec.get("repeat_index") or "0"),
            "generated_answer": generated,
            "reference_answer": reference,
            "abstained": abstained,
            "old_grounding_score": old_g,
            **metrics,
            # B3d: scores vs the PRE-compression originals ("" when unavailable).
            "faithfulness_source": faithfulness_source,
            "grounding_source": grounding_source,
        })
    return rows_out


def _build_evaluator(args: argparse.Namespace) -> Any:
    # Import late so fast mode works in a lean analysis venv (quality.py's top level
    # only needs numpy; the model stacks load lazily and only in --full mode).
    from src.evaluation.quality import QualityEvaluator

    return QualityEvaluator(
        use_nli=args.full,
        use_embeddings=args.full,
        use_bertscore=args.full,
        use_rouge=args.full,
        use_lettucedetect=args.full,
        device=args.device,
    )


# ---------------------------------------------------------------------------
# Campaign v2 scoring tree (cloud/RESULTS_LAYOUT.md §6)
# ---------------------------------------------------------------------------


def _git_provenance() -> dict[str, Any]:
    """Best-effort repo SHA for the scoring manifest; a tarball checkout
    without git records an explicit null, never a fabricated SHA."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=REPO_ROOT,
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        )
        return {"code_git_sha": sha, "git_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"code_git_sha": None, "git_dirty": None,
                "git_note": "git unavailable at scoring time"}


def _instrument_models(evaluator: Any) -> dict[str, Any]:
    """Instrument model ids the evaluator is configured to consult.

    Charter D8 §8.1 ("every score row carries instrument id+version, calibration
    id") + review 2026-08-04 §4.8: the scoring manifest must record WHICH
    instruments produced a pass. Names are the resolved constructor-time ids
    (env overrides already applied), keyed like RunManifest.instrument_models.
    Only enabled instruments are recorded — a fast pass consults no model stack,
    so it honestly records an empty mapping. The claim-checker route rides with
    the NLI flag because faithfulness scoring is gated on ``use_nli``.
    """
    models: dict[str, Any] = {}
    if evaluator.use_nli:
        models["nli"] = evaluator.nli_model_name
        models["claim_checker"] = evaluator.claim_checker_name
    if evaluator.use_embeddings:
        models["embedding"] = evaluator.embedding_model_name
    if evaluator.use_bertscore:
        models["bertscore"] = evaluator.bertscore_model_name
    if evaluator.use_lettucedetect:
        models["lettucedetect"] = evaluator.lettucedetect_model_name
    return models


def _quality_aggregate(rows_out: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-window quality.json: counts + means of the numeric score columns."""
    n_abstained = sum(1 for r in rows_out if r.get("abstained"))
    sums: dict[str, list[float]] = {}
    for row in rows_out:
        for key, value in row.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            sums.setdefault(key, []).append(float(value))
    return {
        "rows": len(rows_out),
        "abstained": n_abstained,
        "means": {k: sum(v) / len(v) for k, v in sorted(sums.items()) if v},
    }


def run_scoring_tree(root: Path, scoring_run_id: str, args: argparse.Namespace) -> int:
    """RESULTS_LAYOUT §6: one offline scoring pass = one scoring/<id>/ tree.

    Fail-closed preconditions: a v2 run root (manifest.json + cells/), a
    SEALED raw tree (ledger.json — scoring must reference the seal it scored
    against), a fresh scoring_run_id (reruns never overwrite: a scoring bug is
    fixed by a NEW id), and no --apply (which would write into cells/).
    """
    from src.analysis.stats.ledger import hash_artifacts, read_ledger, write_ledger
    from src.observability.provenance import instrument_versions

    if args.apply:
        print("ERROR: --apply mutates cells/results.csv and is forbidden in "
              "scoring-tree mode (RESULTS_LAYOUT §6: scoring NEVER writes into "
              "cells/)", file=sys.stderr)
        return 2
    if not SCORING_RUN_ID_RE.match(scoring_run_id):
        print(f"ERROR: scoring run id {scoring_run_id!r} violates the §6 grammar "
              f"{SCORING_RUN_ID_RE.pattern} (e.g. 's01-lettucedetect-nli')",
              file=sys.stderr)
        return 2
    manifest_path = root / "manifest.json"
    cells_dir = root / "cells"
    ledger_path = root / "ledger.json"
    for path, why in (
        (manifest_path, "a v2 run root carries manifest.json (§3)"),
        (cells_dir, "a v2 run root carries cells/ (§1)"),
        (ledger_path, "the raw tree must be SEALED before scoring (§5/§6 — "
                      "the scoring manifest references the seal it scored against)"),
    ):
        if not path.exists():
            print(f"ERROR: {path} missing — {why}", file=sys.stderr)
            return 2

    try:
        run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"ERROR: {manifest_path} is not valid JSON: {exc}", file=sys.stderr)
        return 2
    raw_run_id = run_manifest.get("run_id")
    if not isinstance(raw_run_id, str) or not raw_run_id:
        print(f"ERROR: {manifest_path} has no non-empty 'run_id'", file=sys.stderr)
        return 2

    # read_ledger verifies the ledger's self-hash; the entries_sha256 field is
    # then trustworthy to quote in the scoring manifest.
    read_ledger(ledger_path)
    entries_sha256 = json.loads(ledger_path.read_text(encoding="utf-8"))["entries_sha256"]

    scoring_dir = root / SCORING_DIRNAME / scoring_run_id
    if scoring_dir.exists():
        print(f"ERROR: {scoring_dir} already exists — scoring passes are "
              "append-only; a scoring bug is fixed by a NEW scoring_run_id "
              "(RESULTS_LAYOUT §6)", file=sys.stderr)
        return 2

    evidence_files = sorted(cells_dir.glob("*/window_*/qa_evidence.jsonl"))
    if not evidence_files:
        print(f"ERROR: no cells/*/window_*/qa_evidence.jsonl under {root}",
              file=sys.stderr)
        return 2

    evaluator = _build_evaluator(args)
    # getattr: hand-built Namespaces predating the --batch-size flag (and older
    # callers) keep the sequential default — defaults preserved, fail-open to
    # the historical behavior, never to a new one.
    batch_size = getattr(args, "batch_size", None)
    written: list[Path] = []
    total_rows = 0
    total_abstained = 0

    scoring_dir.mkdir(parents=True)
    for ev_path in evidence_files:
        rows_out = _score_evidence_file(ev_path, evaluator, batch_size=batch_size)
        rel_window = ev_path.parent.relative_to(root)  # cells/<row_key>/window_<k>
        out_window = scoring_dir / rel_window
        out_window.mkdir(parents=True, exist_ok=True)

        scores_path = out_window / QA_SCORES_NAME
        with scores_path.open("w", encoding="utf-8") as fh:
            for row in rows_out:
                fh.write(json.dumps(row, default=str) + "\n")
        written.append(scores_path)

        quality_path = out_window / QUALITY_JSON_NAME
        quality_path.write_text(
            json.dumps(_quality_aggregate(rows_out), indent=2) + "\n",
            encoding="utf-8",
        )
        written.append(quality_path)

        total_rows += len(rows_out)
        total_abstained += sum(1 for r in rows_out if r.get("abstained"))

    scoring_manifest = {
        "scoring_run_id": scoring_run_id,
        "mode": "full" if args.full else "fast",
        "device": args.device,
        "batch_size": batch_size,  # None = sequential (historical) path
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "quality_module": "src.evaluation.quality.QualityEvaluator",
        # D8 §8.1 drift audit (review §4.8): EVERY scoring pass — fast included —
        # records the installed scoring-stack package versions via the canonical
        # provenance primitive (fail-soft: an absent package records None, honest
        # evidence in a lean analysis venv) and the instrument model ids in use.
        "instrument_versions": instrument_versions(),
        "instrument_models": _instrument_models(evaluator),
        "calibration_id": evaluator.calibration_id,
        "raw_run_id": raw_run_id,
        "raw_run_ledger_entries_sha256": entries_sha256,
        "n_evidence_files": len(evidence_files),
        "n_rows": total_rows,
        **_git_provenance(),
    }
    manifest_out = scoring_dir / SCORING_MANIFEST_NAME
    manifest_out.write_text(
        json.dumps(scoring_manifest, indent=2) + "\n", encoding="utf-8"
    )
    written.append(manifest_out)

    # §6: "Scoring passes get their own ledger inside their own directory
    # before being used by stats."
    write_ledger(
        hash_artifacts(written, base_dir=scoring_dir),
        scoring_dir / "ledger.json",
    )

    mode = "FULL" if args.full else "FAST (model-free metrics + abstention short-circuit)"
    print(f"SCORING_TREE_DONE  id={scoring_run_id}  mode={mode}  "
          f"files={len(evidence_files)}  rows={total_rows}  "
          f"abstained={total_abstained}")
    print(f"  tree   : {scoring_dir}")
    print(f"  sealed : {scoring_dir / 'ledger.json'}")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    root = Path(args.run_root)
    if not root.exists():
        print(f"ERROR: {root} does not exist", file=sys.stderr)
        return 2

    if args.scoring_run_id is not None:
        return run_scoring_tree(root, args.scoring_run_id, args)

    evidence_files = sorted(root.rglob("qa_evidence.jsonl"))
    if not evidence_files:
        print(f"ERROR: no qa_evidence.jsonl under {root}", file=sys.stderr)
        return 2

    evaluator = _build_evaluator(args)
    batch_size = getattr(args, "batch_size", None)  # see run_scoring_tree note

    total_rows = 0
    total_abstained = 0
    newly_na_grounding = 0  # abstained rows the ORIGINAL run had scored with a grounding number
    per_cell: dict[str, list[int]] = {}

    for ev_path in evidence_files:
        rows_out = _score_evidence_file(ev_path, evaluator, batch_size=batch_size)
        for row in rows_out:
            total_rows += 1
            if row["abstained"]:
                total_abstained += 1
                if row["old_grounding_score"] is not None:
                    newly_na_grounding += 1
            cell = row["baseline"]
            per_cell.setdefault(cell, [0, 0])
            per_cell[cell][0] += 1
            per_cell[cell][1] += int(bool(row["abstained"]))

        if rows_out:
            out_path = ev_path.parent / args.out_name
            # Header = union of keys across ALL rows (first-seen order), not row 0's keys:
            # QualityMetrics.to_dict() is row-dependent (hallucination_detected appears only
            # when LettuceDetect returns a verdict), so row 0 under-specifies the header.
            # Live failure 2026-07-16: "dict contains fields not in fieldnames".
            fieldnames = list(rows_out[0].keys())
            _seen = set(fieldnames)
            for _r in rows_out[1:]:
                for _k in _r.keys():
                    if _k not in _seen:
                        _seen.add(_k)
                        fieldnames.append(_k)
            with out_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
                writer.writeheader()
                writer.writerows(rows_out)
            if args.apply:
                n_upd = _apply_to_results_csv(ev_path.parent, rows_out, args.full)
                print(f"  applied -> {ev_path.parent}/results.csv ({n_upd} rows updated)")

    mode = "FULL" if args.full else "FAST (model-free metrics + abstention short-circuit)"
    print(f"RESCORE_DONE  mode={mode}  files={len(evidence_files)}  rows={total_rows}")
    print(f"  abstentions detected: {total_abstained}")
    print(f"  abstained rows the original run had scored for grounding "
          f"(now correctly N/A): {newly_na_grounding}")
    print(f"  {'cell':<32}{'rows':>6}{'abstained':>11}")
    for cell in sorted(per_cell):
        n, a = per_cell[cell]
        print(f"  {cell:<32}{n:>6}{a:>11}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
