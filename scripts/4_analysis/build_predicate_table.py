#!/usr/bin/env python3
"""Order:     stage 4 — after a sealed run AND a sealed scoring pass; before stats consume predicates
Objective: Join the scoring sidecar onto evidence rows and compute the §8.5 per-query veridicality predicate table (seal-verified, fail-closed)
Cloud:     local

Build the §8.5 predicate table for one sealed run + one sealed scoring pass.

Task #119 (Topic-6 F1-F4): the missing producer of the registered per-query
``predicate`` metric (families.DEFAULT_METRICS) and the missing join from the
decoupled-scoring sidecar back to the serving evidence rows. This CLI:

1. verifies the RAW tree seal (ledger.json) and the scoring pass's own seal,
   and that the pass scored EXACTLY this raw seal
   (``raw_run_ledger_entries_sha256``, same check organize_results makes);
2. per scored window, joins ``scoring/<id>/cells/<row_key>/window_<k>/
   qa_scores.jsonl`` onto ``cells/<row_key>/window_<k>/qa_evidence.jsonl`` on
   the #127 identity triple (example_id, repeat_index, record_index) —
   duplicates and unmatched rows in either direction REFUSE (counts named);
3. computes the §8.5 per-dataset veridicality predicate
   (src.analysis.predicate — span-QA correctness branch, Qasper groundedness
   branch at the EXPLICIT τ, not-ok rows nulled + counted, null-fraction
   sanity bound enforced);
4. writes the joined per-query table as a NEW post-seal sibling tree,

       <run_root>/predicate/<scoring_run_id>/
           predicate_manifest.json
           cells/<row_key>/window_<k>/predicate.jsonl
           ledger.json                       # own §5-style seal

   mirroring the raw layout so run_campaign_analysis's per-query loader can
   join ``predicate.jsonl`` beside requests.jsonl/qa_evidence.jsonl per
   window. The raw tree is NEVER written (cells/ stays sealed, §5/§6); the
   predicate tree is append-only per scoring pass — a rebuild needs --force.

Windows whose dataset is outside the §8.5 predicate universe (ruler /
scbench; sharegpt has no evidence at all) are counted, labeled skips — they
never feed Y by charter.

Config doctrine (#120 executed → T6.1): --max-null-fraction is REQUIRED (no
silent default on the refusal bound). The Qasper τ is REGISTERED (τ* =
0.9955156950672646, rule max_balanced_accuracy_pooled_v1, calibrated
2026-08-19) and is CONSUMED from the frozen registration artifact — resolved
as --freeze-file, else $CAGE_FREEZE_RESOLUTIONS, else the repo-relative
MyDocs/registration/freeze_resolutions.json (present on the analysis
machine, absent on pods; scoring runs post-pull locally). Resolution is
fail-closed:

- artifact present + --qasper-tau given: the hand-passed decimal must equal
  the registered decimal AS WRITTEN (string compare) or the build REFUSES
  naming both — one typo must never produce an unregistered analysis;
- artifact present, no flag: τ comes from the artifact;
- artifact absent, flag given: accepted with a LOUD single-line
  unregistered-by-hand warning;
- neither: qasper windows refuse (src.analysis.predicate), unchanged.

The manifest records τ AND its source ("freeze-file"|"cli-flag").
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.predicate import (  # noqa: E402
    PREDICATE_DATASETS,
    PredicateConfig,
    PredicateError,
    compute_window_predicate,
    join_window_rows,
)
from src.analysis.stats.ledger import (  # noqa: E402
    LedgerError,
    hash_artifacts,
    read_ledger,
    verify_ledger,
    write_ledger,
)

PREDICATE_DIRNAME = "predicate"
PREDICATE_MANIFEST_NAME = "predicate_manifest.json"
PREDICATE_ROWS_NAME = "predicate.jsonl"
PREDICATE_SCHEMA_VERSION = 1
#: Same §6 id grammar as the scoring pass the table derives from.
QA_SCORES_NAME = "qa_scores.jsonl"
QA_EVIDENCE_NAME = "qa_evidence.jsonl"

#: T6.1 — the frozen registration artifact carrying the registered Qasper τ
#: (task #120 executed 2026-08-19, rule max_balanced_accuracy_pooled_v1).
#: Gitignored; present on the analysis machine, absent on pods.
FREEZE_ENV_VAR = "CAGE_FREEZE_RESOLUTIONS"
DEFAULT_FREEZE_FILE = REPO_ROOT / "MyDocs" / "registration" / "freeze_resolutions.json"
FREEZE_TAU_KEY = "QASPER_TAU"
#: Manifest provenance labels — WHERE the consumed τ came from.
TAU_SOURCE_FREEZE = "freeze-file"
TAU_SOURCE_CLI = "cli-flag"

_MANIFEST_REQUIRED_KEYS: tuple[str, ...] = (
    "schema_version",
    "scoring_run_id",
    "raw_run_ledger_entries_sha256",
    "scoring_ledger_entries_sha256",
    "config",
    "counts",
    "created_utc",
)


class BuildPredicateError(RuntimeError):
    """Any refusal in the predicate-table build (fail loud, message first)."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BuildPredicateError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise BuildPredicateError(
                f"{path}:{lineno}: record must be a JSON object, "
                f"got {type(obj).__name__}"
            )
        rows.append(obj)
    return rows


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _entries_sha256(ledger_path: Path) -> str:
    read_ledger(ledger_path)  # verifies the self-hash first
    return json.loads(ledger_path.read_text(encoding="utf-8"))["entries_sha256"]


def _window_dataset(window_dir_name: str) -> str:
    """Dataset id from a §1 window dir name (window_<dataset>-<NN>)."""
    stem = window_dir_name[len("window_"):]
    return stem.rsplit("-", 1)[0]


def read_frozen_tau(freeze_path: Path) -> str:
    """The registered Qasper τ as the decimal literal WRITTEN in the artifact.

    Returned verbatim (json parse with number-literals-as-str): the T6.1
    match rule string-compares a hand-passed --qasper-tau against the
    registered decimal AS WRITTEN, so a float round-trip must never get the
    chance to rewrite the digits.
    """
    try:
        text = freeze_path.read_text(encoding="utf-8")
    except OSError as exc:
        # Exists-but-unreadable (permissions, I/O) must surface as the clean
        # refusal, not a raw traceback (2026-08-30 verifier minor).
        raise BuildPredicateError(
            f"{freeze_path}: freeze artifact exists but cannot be read: {exc}"
        ) from exc
    try:
        typed = json.loads(text)
        as_written = json.loads(text, parse_float=str, parse_int=str)
    except json.JSONDecodeError as exc:
        raise BuildPredicateError(
            f"{freeze_path}: freeze artifact is not valid JSON: {exc}"
        ) from exc
    if not isinstance(typed, dict) or FREEZE_TAU_KEY not in typed:
        raise BuildPredicateError(
            f"{freeze_path}: no {FREEZE_TAU_KEY!r} key — a freeze artifact "
            "without the registered τ cannot resolve the Qasper branch; fix "
            "the artifact, do not hand-pass around it"
        )
    value = typed[FREEZE_TAU_KEY]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BuildPredicateError(
            f"{freeze_path}: {FREEZE_TAU_KEY} is {value!r} — the registered "
            "τ is a JSON number by schema; a re-typed artifact is not the "
            "registered artifact"
        )
    return as_written[FREEZE_TAU_KEY]


def resolve_qasper_tau(
    cli_tau: str | None, freeze_file: Path
) -> tuple[float | None, str | None, str | None]:
    """T6.1 τ resolution against the freeze artifact → (τ, source, warning).

    ``source`` is TAU_SOURCE_FREEZE / TAU_SOURCE_CLI / None; ``warning`` is
    the LOUD single-line unregistered-by-hand notice (caller prints it) or
    None. Rules are in the module docstring; when the artifact is present
    AND --qasper-tau matches it, the artifact is authoritative (source =
    TAU_SOURCE_FREEZE — the flag was a redundant confirmation).
    """
    if cli_tau is not None:
        try:
            cli_value = float(cli_tau)
        except ValueError as exc:
            raise BuildPredicateError(
                f"--qasper-tau {cli_tau!r} is not a decimal number"
            ) from exc
    if freeze_file.is_file():
        literal = read_frozen_tau(freeze_file)
        if cli_tau is not None and cli_tau != literal:
            raise BuildPredicateError(
                f"--qasper-tau {cli_tau!r} does not MATCH the registered τ "
                f"{literal!r} in {freeze_file} ({FREEZE_TAU_KEY}) — T6.1 "
                "string-compares the decimal as written (a numerically-equal "
                "rewrite is still a mismatch); drop the flag to consume the "
                "artifact, or fix the typo"
            )
        return float(literal), TAU_SOURCE_FREEZE, None
    if cli_tau is not None:
        return (
            cli_value,
            TAU_SOURCE_CLI,
            f"WARNING: qasper tau {cli_tau} is UNREGISTERED-BY-HAND — no "
            f"freeze artifact at {freeze_file}; the registered analysis "
            f"consumes {FREEZE_TAU_KEY} from the frozen "
            "MyDocs/registration/freeze_resolutions.json",
        )
    return None, None, None


def build_predicate_table(
    run_root: Path,
    scoring_run_id: str,
    config: PredicateConfig,
    *,
    force: bool = False,
    tau_source: str | None = None,
    freeze_file: str | None = None,
) -> Path:
    """Produce ``<run_root>/predicate/<scoring_run_id>/`` (see module doc).

    ``tau_source``/``freeze_file`` are the T6.1 provenance of
    ``config.qasper_tau`` (resolve_qasper_tau) — recorded in the manifest. A
    configured τ with no stated source refuses: provenance is part of the
    artifact, and a bare hand value here is exactly the unregistered-analysis
    hazard the freeze consumption exists to close.
    """
    if config.qasper_tau is not None and tau_source not in (
        TAU_SOURCE_FREEZE, TAU_SOURCE_CLI
    ):
        raise BuildPredicateError(
            f"qasper_tau={config.qasper_tau!r} carries no stated provenance "
            f"(tau_source={tau_source!r}, expected "
            f"{TAU_SOURCE_FREEZE!r}|{TAU_SOURCE_CLI!r}) — resolve τ via "
            "resolve_qasper_tau against the freeze artifact; never pass a "
            "bare value"
        )
    if config.qasper_tau is None and tau_source is not None:
        raise BuildPredicateError(
            f"tau_source={tau_source!r} stated but config.qasper_tau is None "
            "— a source without a value is fabricated provenance"
        )
    run_root = Path(run_root).resolve()
    raw_ledger = run_root / "ledger.json"
    cells_dir = run_root / "cells"
    scoring_dir = run_root / "scoring" / scoring_run_id
    for path, why in (
        (run_root / "manifest.json", "a v2 run root carries manifest.json (§3)"),
        (cells_dir, "a v2 run root carries cells/ (§1)"),
        (raw_ledger, "the raw tree must be SEALED (§5) before its predicate "
                     "table can be built"),
        (scoring_dir, f"no scoring pass {scoring_run_id!r} under "
                      f"{run_root / 'scoring'} — the predicate joins a "
                      "SEALED scoring pass (rescore_quality --scoring-run-id)"),
        (scoring_dir / "ledger.json", "the scoring pass must carry its own "
                                      "seal (§6) before stats may consume it"),
    ):
        if not path.exists():
            raise BuildPredicateError(f"{path} missing — {why}")

    try:
        raw_sha = _entries_sha256(raw_ledger)
        scoring_sha = _entries_sha256(scoring_dir / "ledger.json")
        mismatches = verify_ledger(scoring_dir / "ledger.json", scoring_dir)
    except LedgerError as exc:
        raise BuildPredicateError(f"ledger verification failed: {exc}") from exc
    if mismatches:
        raise BuildPredicateError(
            "scoring pass fails its own seal: " + "; ".join(mismatches)
        )
    scoring_manifest_path = scoring_dir / "scoring_manifest.json"
    if not scoring_manifest_path.is_file():
        raise BuildPredicateError(f"{scoring_manifest_path} missing (§6)")
    scoring_manifest = json.loads(scoring_manifest_path.read_text(encoding="utf-8"))
    if scoring_manifest.get("raw_run_ledger_entries_sha256") != raw_sha:
        raise BuildPredicateError(
            f"scoring pass {scoring_run_id!r} scored a DIFFERENT raw seal "
            f"({scoring_manifest.get('raw_run_ledger_entries_sha256')!r} != "
            f"{raw_sha!r}) — its verdicts do not describe this tree"
        )

    out_dir = run_root / PREDICATE_DIRNAME / scoring_run_id
    if out_dir.exists():
        if not force:
            raise BuildPredicateError(
                f"{out_dir} already exists — a predicate table is derived "
                "data; rebuild deliberately with --force"
            )
        import shutil

        shutil.rmtree(out_dir)

    score_files = sorted(scoring_dir.glob("cells/*/window_*/" + QA_SCORES_NAME))
    if not score_files:
        raise BuildPredicateError(
            f"no cells/*/window_*/{QA_SCORES_NAME} under {scoring_dir} — an "
            "empty scoring pass produces no predicate"
        )

    # Window-level reconciliation in BOTH directions: every scored window must
    # have raw evidence, every raw evidence window must be scored (the
    # sidecar covers the tree it claims to describe). Predicate-universe
    # datasets only; the rest are labeled skips below.
    scored_windows = {
        p.parent.relative_to(scoring_dir).as_posix() for p in score_files
    }
    evidence_windows = {
        p.parent.relative_to(run_root).as_posix()
        for p in cells_dir.glob("*/window_*/" + QA_EVIDENCE_NAME)
    }
    ev_only = sorted(
        w for w in evidence_windows - scored_windows
        if _window_dataset(Path(w).name) in PREDICATE_DATASETS
    )
    sc_only = sorted(w for w in scored_windows - evidence_windows)
    if ev_only or sc_only:
        raise BuildPredicateError(
            f"scoring pass {scoring_run_id!r} does not cover this tree: "
            f"{len(ev_only)} evidence window(s) never scored {ev_only[:5]} / "
            f"{len(sc_only)} scored window(s) without raw evidence "
            f"{sc_only[:5]} — re-run the scoring pass over the sealed tree"
        )

    written: list[Path] = []
    per_window: list[dict[str, Any]] = []
    skipped_windows: list[dict[str, Any]] = []
    totals = {"n_rows": 0, "n_true": 0, "n_false": 0, "n_null": 0,
              "n_not_ok_nulled": 0, "n_missing_verdict": 0}
    try:
        for score_path in score_files:
            rel_window = score_path.parent.relative_to(scoring_dir)  # cells/<rk>/window_<k>
            window_label = rel_window.as_posix()
            dataset = _window_dataset(rel_window.name)
            if dataset not in PREDICATE_DATASETS:
                skipped_windows.append(
                    {
                        "window": window_label,
                        "dataset": dataset,
                        "reason": "outside the §8.5 predicate universe "
                                  "(instrument/load-donor dataset — never feeds Y)",
                    }
                )
                continue
            evidence_path = run_root / rel_window / QA_EVIDENCE_NAME
            if not evidence_path.is_file():
                raise BuildPredicateError(
                    f"{evidence_path} missing — the scored window has no raw "
                    "evidence to join (§1 contract)"
                )
            joined = join_window_rows(
                _read_jsonl(evidence_path),
                _read_jsonl(score_path),
                window=window_label,
            )
            rows, summary = compute_window_predicate(
                joined, dataset, config, window=window_label
            )
            out_path = out_dir / rel_window / PREDICATE_ROWS_NAME
            _atomic_write_text(
                out_path, "".join(json.dumps(r) + "\n" for r in rows)
            )
            written.append(out_path)
            per_window.append({"window": window_label, **summary})
            for key in totals:
                totals[key] += int(summary.get(key, 0))
    except (PredicateError, BuildPredicateError):
        # Fail-closed: never leave a half-built table behind a passing path.
        import shutil

        shutil.rmtree(out_dir, ignore_errors=True)
        raise

    if not written:
        import shutil

        shutil.rmtree(out_dir, ignore_errors=True)
        raise BuildPredicateError(
            "no predicate-universe window was scored — nothing to build "
            f"(skipped: {[s['window'] for s in skipped_windows]})"
        )

    totals["null_fraction"] = (
        totals["n_null"] / totals["n_rows"] if totals["n_rows"] else 0.0
    )
    manifest = {
        "schema_version": PREDICATE_SCHEMA_VERSION,
        "scoring_run_id": scoring_run_id,
        "raw_run_ledger_entries_sha256": raw_sha,
        "scoring_ledger_entries_sha256": scoring_sha,
        "config": {
            "max_null_fraction": float(config.max_null_fraction),
            "qasper_tau": (
                None if config.qasper_tau is None else float(config.qasper_tau)
            ),
            # T6.1 provenance: "freeze-file" | "cli-flag"; None only when no
            # τ was configured at all (tree without qasper windows).
            "qasper_tau_source": tau_source,
            # The consumed registration artifact (None unless source is
            # freeze-file).
            "freeze_file": freeze_file,
        },
        "counts": {**totals, "n_windows": len(per_window)},
        "per_window": per_window,
        "skipped_windows": skipped_windows,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest_path = out_dir / PREDICATE_MANIFEST_NAME
    _atomic_write_text(manifest_path, json.dumps(manifest, indent=2) + "\n")
    written.append(manifest_path)
    write_ledger(hash_artifacts(written, base_dir=out_dir), out_dir / "ledger.json")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Join a sealed scoring pass onto its sealed raw tree and "
                    "produce the §8.5 per-query predicate table (task #119)."
    )
    parser.add_argument("run_root", type=Path,
                        help="v2 run root: results/<campaign>/<session>/<run_id>")
    parser.add_argument("--scoring-run-id", required=True,
                        help="the SEALED scoring pass whose verdicts feed the "
                             "predicate (scoring/<id>/)")
    parser.add_argument("--max-null-fraction", type=float, required=True,
                        help="REQUIRED refusal bound: max fraction of "
                             "predicate=None rows per window (no silent "
                             "default — state the bound)")
    parser.add_argument("--qasper-tau", type=str, default=None, metavar="TAU",
                        help="Instrument-A groundedness threshold for the "
                             "Qasper branch, as a decimal string. When the "
                             "freeze artifact is present it must MATCH the "
                             "registered decimal as written (T6.1) or the "
                             "build refuses; without the artifact it is "
                             "accepted with a LOUD unregistered-by-hand "
                             "warning")
    parser.add_argument("--freeze-file", type=Path, default=None,
                        help="frozen registration artifact carrying the "
                             f"registered τ ({FREEZE_TAU_KEY}); default "
                             f"${FREEZE_ENV_VAR}, then the repo-relative "
                             f"{DEFAULT_FREEZE_FILE}")
    parser.add_argument("--force", action="store_true",
                        help="rebuild over an existing predicate/<id>/ tree")
    args = parser.parse_args(argv)
    freeze_file = args.freeze_file
    if freeze_file is not None and not freeze_file.is_file():
        # An EXPLICIT --freeze-file that does not exist is a typo, not an
        # opt-out: silently falling through to the unregistered-by-hand path
        # is exactly the failure T6.1 exists to prevent (verifier minor).
        print(
            f"ERROR: --freeze-file {freeze_file} does not exist — fix the "
            "path, or drop the flag to use $CAGE_FREEZE_RESOLUTIONS / the "
            "repo default",
            file=sys.stderr,
        )
        return 2
    if freeze_file is None:
        env_path = os.environ.get(FREEZE_ENV_VAR, "").strip()
        freeze_file = Path(env_path) if env_path else DEFAULT_FREEZE_FILE
    try:
        tau, tau_source, warning = resolve_qasper_tau(
            args.qasper_tau, Path(freeze_file)
        )
        if warning:
            print(warning, file=sys.stderr)
        config = PredicateConfig(
            max_null_fraction=args.max_null_fraction, qasper_tau=tau
        )
        out_dir = build_predicate_table(
            args.run_root, args.scoring_run_id, config, force=args.force,
            tau_source=tau_source,
            freeze_file=(
                str(Path(freeze_file).resolve())
                if tau_source == TAU_SOURCE_FREEZE else None
            ),
        )
    except (PredicateError, BuildPredicateError, LedgerError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    tau_line = (
        "unset (no τ resolved; qasper windows would have refused)"
        if tau is None else f"{tau!r} (source: {tau_source})"
    )
    print(f"[build_predicate_table] table : {out_dir}")
    print(f"[build_predicate_table] sealed: {out_dir / 'ledger.json'}")
    print(f"[build_predicate_table] tau   : {tau_line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
