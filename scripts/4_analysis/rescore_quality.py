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

Output: one results_rescored.csv next to each qa_evidence.jsonl (same trial dir), with
example_id/baseline/trial provenance, the fresh QualityMetrics columns, and
old_grounding_score copied from the evidence for before/after comparison. Never
overwrites results.csv.

Usage:
  python3 scripts/4_analysis/rescore_quality.py --run-root results/phase2/<run-id>
  python3 scripts/4_analysis/rescore_quality.py --run-root <run> --full --device cuda
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


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
                   help="Output CSV filename written next to each qa_evidence.jsonl.")
    p.add_argument("--apply", action="store_true",
                   help="Also merge the re-scored quality columns back into each trial's "
                        "results.csv (one-time backup at results.csv.pre_rescore). Fast "
                        "mode applies only the model-free fields plus the abstention "
                        "short-circuit (model metrics -> blank on abstained rows); "
                        "--full applies every quality column. This is the post-serving "
                        "scoring step of decoupled mode (run_experiment --skip-quality).")
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


def main() -> int:
    args = parse_args()
    root = Path(args.run_root)
    if not root.exists():
        print(f"ERROR: {root} does not exist", file=sys.stderr)
        return 2

    evidence_files = sorted(root.rglob("qa_evidence.jsonl"))
    if not evidence_files:
        print(f"ERROR: no qa_evidence.jsonl under {root}", file=sys.stderr)
        return 2

    # Import late so fast mode works in a lean analysis venv (quality.py's top level
    # only needs numpy; the model stacks load lazily and only in --full mode).
    from src.evaluation.quality import (
        QualityEvaluator,
        is_no_answer_prediction,
        sanitize_answer,
    )

    evaluator = QualityEvaluator(
        use_nli=args.full,
        use_embeddings=args.full,
        use_bertscore=args.full,
        use_rouge=args.full,
        use_lettucedetect=args.full,
        device=args.device,
    )

    total_rows = 0
    total_abstained = 0
    newly_na_grounding = 0  # abstained rows the ORIGINAL run had scored with a grounding number
    per_cell: dict[str, list[int]] = {}

    for ev_path in evidence_files:
        rows_out = []
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

            metrics = evaluator.evaluate(
                question=question,
                context=list(contexts),
                generated_text=generated,
                reference_answer=reference,
                all_answers=all_answers,
            ).to_dict()

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
            total_rows += 1
            if abstained:
                total_abstained += 1
                if old_g is not None:
                    newly_na_grounding += 1

            cell = rec.get("baseline") or ev_path.parent.parent.name
            per_cell.setdefault(cell, [0, 0])
            per_cell[cell][0] += 1
            per_cell[cell][1] += int(abstained)

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
