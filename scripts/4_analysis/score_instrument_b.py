#!/usr/bin/env python3
"""Instrument-B (AlignScore-large) offline scorer over saved qa_evidence.jsonl.

Charter D8 §8.5: Instrument B is the SECONDARY claim checker selected by the
2026-08-05 calibration — AlignScore-large, ``nli_sp`` (zha2023alignscore, Zha
et al. ACL 2023).  Its 2023 dependency stack (torch<2, pytorch-lightning
1.9.5, transformers 4.26.1, python 3.10) can never coexist with the modern
scoring venv, so scoring runs OUT OF PROCESS in an isolated environment
managed by ``src/evaluation/instrument_b_runner.py`` — this CLI runs in the
project venv and only exchanges JSONL with the worker (the decoupled
re-score architecture, same as rescore_quality.py).

Modes:
  flat (default): --evidence names qa_evidence.jsonl files, directories
      (searched recursively) or glob patterns; per-item rows
      ``{id, alignscore[, grounded_b]}`` go to --out, plus a sidecar
      ``<out>.provenance.json`` (env freeze, verified model shas, spec
      fingerprint, τ).
  scoring tree (--scoring-run-id, cloud/RESULTS_LAYOUT.md §6): --evidence is
      ONE sealed campaign run root; outputs land under
      ``scoring/<scoring_run_id>/cells/<row_key>/window_<k>/``
      as ``instrument_b_scores.jsonl`` with a ``scoring_manifest.json`` and
      the pass's OWN content-hash ledger.  Never writes into ``cells/``.

τ (--tau) DEFAULTS to the REGISTERED value ``instrument_b_runner.
TAU_REGISTERED`` = 0.817024 (RAGTruth-test anchor scope, owner-decided
2026-08-05 — MyDocs/registration/instrument_selection_2026-08-05/DECISION.md;
charter stamp PUBLICATION.md §8.6(c)).  An explicit --tau overrides it; the
provenance sidecar records which was applied (``tau_source`` = "registered"
vs "override").  ``grounded_b`` is added via ``instrument_b_runner.
apply_tau`` (score >= τ is grounded, mirroring select_tau).

Usage:
  python3 scripts/4_analysis/score_instrument_b.py --bootstrap-only
  python3 scripts/4_analysis/score_instrument_b.py \
      --evidence results/phase2/<run>/ --out ib_scores.jsonl --tau 0.85
  python3 scripts/4_analysis/score_instrument_b.py \
      --evidence results/camp1/a/<run_id> --scoring-run-id s02-instrument-b
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(Path(__file__).resolve().parent), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.evaluation import instrument_b_runner as ib  # noqa: E402

# Task #130 shared layer (decisions (a) + (d)): the label-stripping blinder,
# the sealed per-run-root salt and the --abandon helper live in
# rescore_quality.py (same directory — the primary owner of the offline
# scoring chain) so BOTH scripts derive identical tokens from the same salt.
import rescore_quality as rescore  # noqa: E402,F401  (module alias kept for tests)
from rescore_quality import (  # noqa: E402
    BLINDING_MAP_NAME,
    BLINDING_MODE_CONTROL,
    BLINDING_MODE_STRIPPED,
    SALT_FILE_NAME,
    LabelBlinder,
    ScoringBlindingError,
    blinding_join_checksum,
    canonical_sha256,
    load_or_create_scoring_salt,
    write_blinding_map,
)

#: RESULTS_LAYOUT §6 scoring-run-id grammar (same as rescore_quality.py).
SCORING_RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,40}$")
SCORING_DIRNAME = "scoring"
SCORING_MANIFEST_NAME = "scoring_manifest.json"
SCORES_NAME = "instrument_b_scores.jsonl"

#: Task #130 (a) cache-continuity disclosure (stamped into blinding sections):
#: token-stable ids keep the content-addressed cache shareable ACROSS passes
#: from #130 onward, but break continuity with PRE-#130 passes (whose ids
#: carried real labels).
_CACHE_CONTINUITY_NOTE = (
    "item ids are token-stable across passes from task #130 onward "
    "(deterministic per-run-root salt); content-addressed work dirs do NOT "
    "match passes scored before #130, whose ids embedded real arm labels"
)


def _positive_int(value: str) -> int:
    iv = int(value)
    if iv < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value!r}")
    return iv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score saved qa_evidence.jsonl with Instrument B (AlignScore)."
    )
    p.add_argument("--evidence", nargs="+", default=None,
                   help="qa_evidence.jsonl files, directories (recursive) or glob "
                        "patterns; with --scoring-run-id: exactly ONE sealed v2 "
                        "run root (RESULTS_LAYOUT §6).")
    p.add_argument("--out", default=None,
                   help="Output JSONL for flat mode (sidecar <out>.provenance.json "
                        "rides along). Forbidden with --scoring-run-id.")
    p.add_argument("--tau", type=float, default=None,
                   help="Grounded-verdict threshold. DEFAULT = the REGISTERED "
                        f"tau {ib.TAU_REGISTERED} (anchor scope "
                        f"'{ib.TAU_ANCHOR_SCOPE}', owner-decided 2026-08-05 -- "
                        "DECISION.md + PUBLICATION.md 8.6(c)). An explicit "
                        "--tau overrides it; the provenance sidecar records "
                        "tau_source=registered vs override. Rows carry "
                        "grounded_b = alignscore >= tau.")
    p.add_argument("--env-home", default=None,
                   help=f"Isolated env home (default ${ib.ENV_HOME_ENV_VAR} or "
                        f"{ib.DEFAULT_ENV_HOME} — always outside the repo).")
    p.add_argument("--batch", type=_positive_int, default=8,
                   help="Worker scoring batch size (default 8, the selection-run "
                        "configuration).")
    p.add_argument("--max-items", type=_positive_int, default=None,
                   help="Score at most N remaining items (resumable smoke/budget "
                        "mode; the checkpointed worker picks up where it left off).")
    p.add_argument("--device", default="cpu", help="Worker torch device (cpu|cuda).")
    p.add_argument("--bootstrap-only", action="store_true",
                   help="Only ensure the isolated environment (venv, pinned stack, "
                        "verified model downloads); score nothing.")
    p.add_argument("--scoring-run-id", default=None,
                   help="Campaign v2 mode (cloud/RESULTS_LAYOUT.md §6): write "
                        "scoring/<scoring_run_id>/ under the run root instead of "
                        "--out. Never writes into cells/.")
    p.add_argument("--fresh", action="store_true",
                   help="Task #130 decision (c): force a CLEAN content-addressed "
                        "work dir — every score is produced by THIS pass's "
                        "worker run, none served from a previous pass's "
                        "checkpoint (drift-audit passes). Either way the pass "
                        "manifest/sidecar records cache_provenance; reuse is "
                        "disclosed, never silently claimed as independence.")
    p.add_argument("--abandon", default=None, metavar="SCORING_RUN_ID",
                   help="Task #130 decision (d): rename scoring/<id>/ under the "
                        "run root (--evidence, exactly one) to "
                        "scoring/<id>.abandoned-<UTCstamp>/ with a tombstone, "
                        "freeing the id for a clean retry. Requires --reason; a "
                        "VERIFIED complete pass is refused unless "
                        "--force-abandon.")
    p.add_argument("--reason", default=None,
                   help="Reason recorded in the --abandon tombstone.")
    p.add_argument("--force-abandon", action="store_true", dest="force_abandon",
                   help="Allow --abandon on a pass whose own ledger VERIFIES "
                        "complete (normally refused: audit record).")
    p.add_argument("--no-blinding-control", action="store_true",
                   dest="no_blinding_control",
                   help="TEST-ONLY control run without the #130 label-stripping "
                        "layer (for the blinding-equivalence checksum test); "
                        f"recorded as blinding mode {BLINDING_MODE_CONTROL!r}. "
                        "Never use for a registered confirmatory pass.")
    args = p.parse_args(argv)
    # Registered-τ resolution (owner decision 2026-08-05, DECISION.md +
    # PUBLICATION.md §8.6(c)): the None sentinel distinguishes "operator gave
    # no --tau" (apply TAU_REGISTERED, tau_source=registered) from an explicit
    # --tau (tau_source=override) -- even an override that repeats the
    # registered value is recorded as an override, because it was CHOSEN.
    if args.tau is None:
        args.tau = ib.TAU_REGISTERED
        args.tau_source = "registered"
    else:
        args.tau_source = "override"
    return args


# ---------------------------------------------------------------------------
# Evidence -> scoring items
# ---------------------------------------------------------------------------


def _collect_flat_evidence(entries: list[str]) -> list[Path]:
    """Resolve --evidence entries (files, dirs, globs) to evidence files."""
    files: list[Path] = []
    for entry in entries:
        path = Path(entry)
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("qa_evidence.jsonl")))
        else:
            files.extend(Path(m) for m in sorted(glob.glob(entry)) if Path(m).is_file())
    # de-dup, stable order
    seen: set[Path] = set()
    unique: list[Path] = []
    for f in files:
        key = f.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def _build_items(
    ev_path: Path, file_key: str
) -> tuple[list[dict[str, str]], int]:
    """One evidence file -> Instrument-B items; returns (items, n_skipped_empty).

    Tolerant parsing mirrors rescore_quality.py (stringified context lists from
    older runs).  Rows with an empty served context or an empty generated
    answer are UNSCOREABLE by a premise/claim checker and are skipped —
    counted, never emitted as nulls (a null score would poison apply_tau,
    which correctly fails closed on missing scores).
    """
    items: list[dict[str, str]] = []
    skipped = 0
    for line in ev_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        contexts = rec.get("used_contexts") or []
        if isinstance(contexts, str):
            try:
                contexts = json.loads(contexts)
            except json.JSONDecodeError:
                contexts = [contexts]
        context = "\n\n".join(
            str(c).strip() for c in contexts if c and str(c).strip()
        )
        claim = str(rec.get("generated_answer") or "")
        item_id = (
            f"{file_key}::{rec.get('example_id')}::"
            f"{str(rec.get('repeat_index') or '0')}"
        )
        if not context.strip() or not claim.strip():
            skipped += 1
            continue
        items.append({"id": item_id, "context": context, "claim": claim})
    return items, skipped


# ---------------------------------------------------------------------------
# Task #130 decision (a): label-stripped item ids (charter §9.8, audit H9)
# ---------------------------------------------------------------------------


def _blind_tree_file_key(rel: Path, blinder: LabelBlinder) -> str:
    """Blind the arm-bearing row_key component of a §6 evidence path.

    ``cells/<row_key>/window_<k>/qa_evidence.jsonl`` ->
    ``cells/<token>/window_<k>/qa_evidence.jsonl``: the row_key STARTS with
    the arm (RESULTS_LAYOUT §2), so it must never reach the worker; the
    window component carries only dataset+ordinal and stays readable."""
    parts = list(rel.parts)
    if len(parts) != 4 or parts[0] != "cells":
        raise ScoringBlindingError(
            f"unexpected §6 evidence path shape {rel.as_posix()!r} — cannot "
            "locate the arm-bearing row_key component to blind"
        )
    parts[1] = blinder.token(parts[1])
    return "/".join(parts)


def _record_blind_ids(
    items: list[dict[str, str]],
    blind_key: str,
    real_key: str,
    blind_to_real: dict[str, str],
) -> None:
    """Extend the blind_id -> real_id join map for one evidence file's items.

    Item ids are ``<file_key>::<example_id>::<repeat_index>``; the real id is
    the same suffix on the real file key. Fails loud on a colliding blind id
    (the join must stay bijective — task #130 (a))."""
    for item in items:
        item_id = item["id"]
        if not item_id.startswith(blind_key):
            raise ScoringBlindingError(
                f"item id {item_id!r} does not start with its blinded file "
                f"key {blind_key!r}"
            )
        real_id = real_key + item_id[len(blind_key):]
        if item_id in blind_to_real and blind_to_real[item_id] != real_id:
            raise ScoringBlindingError(
                f"blind id {item_id!r} maps to two different input rows — "
                "refusing a non-bijective unblind join"
            )
        blind_to_real[item_id] = real_id


def _check_join_bijective(blind_to_real: dict[str, str], n_items: int) -> None:
    """The unblind join must be a bijection over the scoreable items."""
    if len(blind_to_real) != n_items:
        raise ScoringBlindingError(
            f"unblind map covers {len(blind_to_real)} ids for {n_items} "
            "scoreable items — the join is not one-to-one"
        )
    if len(set(blind_to_real.values())) != len(blind_to_real):
        raise ScoringBlindingError(
            "two blinded ids unblind to the same input row — refusing a "
            "non-injective join (silent score mis-assignment is the failure "
            "mode this check makes impossible)"
        )


def _unblind_ib_rows(
    rows: list[dict[str, Any]], blind_to_real: dict[str, str]
) -> list[dict[str, Any]]:
    """Unblind output row ids through the checked join (task #130 (a)).

    Every scored row must resolve to exactly one input row; an unknown blind
    id or a duplicate real id raises — never a silent mis-assignment."""
    out: list[dict[str, Any]] = []
    seen_real: set[str] = set()
    for row in rows:
        blind_id = row["id"]
        try:
            real_id = blind_to_real[blind_id]
        except KeyError:
            raise ScoringBlindingError(
                f"scored id {blind_id!r} has no entry in the unblind map — a "
                "scored item does not correspond to any input row"
            ) from None
        if real_id in seen_real:
            raise ScoringBlindingError(
                f"two scored items unblind to the same input row {real_id!r}"
            )
        seen_real.add(real_id)
        new = dict(row)
        new["id"] = real_id
        out.append(new)
    return out


# ---------------------------------------------------------------------------
# Provenance sidecar
# ---------------------------------------------------------------------------


def _git_provenance() -> dict[str, Any]:
    """Best-effort repo SHA (same doctrine as rescore_quality.py: a tarball
    checkout records an explicit null, never a fabricated SHA)."""
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


def _provenance_block(
    args: argparse.Namespace,
    evidence_files: list[Path],
    n_items: int,
    n_scored: int,
    n_skipped: int,
) -> dict[str, Any]:
    """The sidecar provenance block (D8 §8.1: every score names its
    instrument+version+calibration lineage).

    Env-manifest / pip-freeze lookups are provenance METADATA, not scores
    (the ``_package_version`` doctrine in quality.py): when unavailable they
    record an explicit note, never fabricate — and never fail the pass.
    """
    home = (
        Path(args.env_home).expanduser()
        if args.env_home
        else ib.default_env_home()
    )
    env_manifest: dict[str, Any] | None
    try:
        env_manifest = ib.read_env_manifest(home)
    except ib.InstrumentBError as exc:
        env_manifest = None
        env_note = str(exc)
    else:
        env_note = None
    freeze_path = home / ib.PIP_FREEZE_NAME
    env_pip_freeze = (
        freeze_path.read_text(encoding="utf-8").splitlines()
        if freeze_path.is_file()
        else None
    )
    return {
        "instrument": "alignscore_large",
        "citation": "zha2023alignscore",
        "spec_version": ib.SPEC.spec_version,
        "spec_fingerprint": ib.spec_fingerprint(ib.SPEC),
        "spec": ib.SPEC.to_dict(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tau": args.tau,
        # τ provenance (DECISION.md 2026-08-05): was the applied τ the
        # registered value by default, or an explicit operator override?
        "tau_source": getattr(args, "tau_source", None),
        "tau_registered": ib.TAU_REGISTERED,
        "tau_anchor_scope": ib.TAU_ANCHOR_SCOPE,
        "batch": args.batch,
        "device": args.device,
        "max_items": args.max_items,
        "env_home": str(home),
        "env_manifest": env_manifest,
        "env_note": env_note,
        "env_pip_freeze": env_pip_freeze,
        "evidence_files": [str(f) for f in evidence_files],
        "n_items": n_items,
        "n_scored": n_scored,
        "n_skipped_empty": n_skipped,
        **_git_provenance(),
    }


# ---------------------------------------------------------------------------
# Output rows
# ---------------------------------------------------------------------------


def _rows_for(
    items: list[dict[str, str]],
    scores: dict[str, float],
    verdicts: dict[str, bool] | None,
) -> list[dict[str, Any]]:
    """Per-item output rows in input order; unscored items (a --max-items
    partial pass) are omitted, never emitted as nulls."""
    rows: list[dict[str, Any]] = []
    for item in items:
        item_id = item["id"]
        if item_id not in scores:
            continue
        row: dict[str, Any] = {"id": item_id, "alignscore": scores[item_id]}
        if verdicts is not None:
            row["grounded_b"] = verdicts[item_id]
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Scoring-tree mode (cloud/RESULTS_LAYOUT.md §6)
# ---------------------------------------------------------------------------


def run_scoring_tree(root: Path, args: argparse.Namespace) -> int:
    """One Instrument-B pass = one scoring/<id>/ tree under a SEALED run root.

    Same fail-closed preconditions as rescore_quality.run_scoring_tree: a v2
    run root (manifest.json + cells/), a sealed raw tree (ledger.json), a
    fresh scoring_run_id, and no flat-mode --out.
    """
    from src.analysis.stats.ledger import hash_artifacts, read_ledger, write_ledger

    scoring_run_id: str = args.scoring_run_id
    if args.out is not None:
        print("ERROR: --out is forbidden with --scoring-run-id (outputs are "
              "placed by the §6 tree convention)", file=sys.stderr)
        return 2
    if not SCORING_RUN_ID_RE.match(scoring_run_id):
        print(f"ERROR: scoring run id {scoring_run_id!r} violates the §6 grammar "
              f"{SCORING_RUN_ID_RE.pattern} (e.g. 's02-instrument-b')",
              file=sys.stderr)
        return 2
    manifest_path = root / "manifest.json"
    cells_dir = root / "cells"
    ledger_path = root / "ledger.json"
    for path, why in (
        (manifest_path, "a v2 run root carries manifest.json (§3)"),
        (cells_dir, "a v2 run root carries cells/ (§1)"),
        (ledger_path, "the raw tree must be SEALED before scoring (§5/§6)"),
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
    read_ledger(ledger_path)  # verifies the seal's self-hash
    entries_sha256 = json.loads(ledger_path.read_text(encoding="utf-8"))[
        "entries_sha256"
    ]

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

    # Task #130 decision (a): item ids embed the evidence path, and the §6
    # row_key STARTS with the arm — blind that component with the per-run-root
    # sealed salt (shared with rescore_quality.py; deterministic so the
    # content-addressed cache stays shareable across passes, decision (c))
    # BEFORE anything reaches the worker.
    no_blinding = bool(getattr(args, "no_blinding_control", False))
    blinder: LabelBlinder | None = None
    scoring_salt = None
    if not no_blinding:
        scoring_salt = load_or_create_scoring_salt(
            root / SCORING_DIRNAME / SALT_FILE_NAME
        )
        blinder = LabelBlinder(scoring_salt.salt)

    per_file: dict[Path, tuple[list[dict[str, str]], int]] = {}
    blind_to_real: dict[str, str] = {}
    n_skipped = 0
    for ev_path in evidence_files:
        rel = ev_path.relative_to(root)
        real_key = rel.as_posix()
        file_key = (
            _blind_tree_file_key(rel, blinder) if blinder is not None else real_key
        )
        items, skipped = _build_items(ev_path, file_key)
        if blinder is not None:
            _record_blind_ids(items, file_key, real_key, blind_to_real)
        n_skipped += skipped
        per_file[ev_path] = (items, skipped)
    all_items = [item for items, _ in per_file.values() for item in items]
    if not all_items:
        print("ERROR: no scoreable rows (all empty-context/empty-answer)",
              file=sys.stderr)
        return 2
    if blinder is not None:
        _check_join_bijective(blind_to_real, len(all_items))

    env_home = Path(args.env_home).expanduser() if args.env_home else None
    scores, cache_provenance = ib.score_with_provenance(
        all_items,
        env_home=env_home,
        batch_size=args.batch,
        device=args.device,
        max_items=args.max_items,
        fresh=bool(getattr(args, "fresh", False)),
    )
    verdicts = ib.apply_tau(scores, args.tau) if args.tau is not None else None

    written: list[Path] = []
    n_rows = 0
    # Per-window accounting (task #127, audit H11): EVERY window gets a manifest
    # entry -- a window whose rows are all unscoreable (or all left unscored by
    # a --max-items partial pass) is an explicit n_scored=0 record with its
    # skip counts, never a bare continue (charter §9.10).
    windows: dict[str, dict[str, int]] = {}
    scoring_dir.mkdir(parents=True)
    for ev_path, (items, skipped) in per_file.items():
        rows = _rows_for(items, scores, verdicts)
        # Task #130 (a): unblind at output time — final artifacts carry REAL
        # ids; the join is bijective and checked (loud failure).
        if blinder is not None:
            rows = _unblind_ib_rows(rows, blind_to_real)
        rel_window = ev_path.parent.relative_to(root)  # cells/<row_key>/window_<k>
        windows[rel_window.as_posix()] = {
            "n_evidence_rows": len(items) + skipped,
            "n_scoreable": len(items),
            "n_skipped_unscoreable": skipped,
            "n_scored": len(rows),
        }
        if not rows:
            continue  # accounted above; no empty scores file is written
        out_window = scoring_dir / rel_window
        out_window.mkdir(parents=True, exist_ok=True)
        scores_path = out_window / SCORES_NAME
        with scores_path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        written.append(scores_path)
        n_rows += len(rows)

    # Task #130 (a): seal this pass's token->row_key map and build the
    # "blinding" section the #112 prereg text cites.
    if blinder is not None and scoring_salt is not None:
        map_path = scoring_dir / BLINDING_MAP_NAME
        map_sha256 = write_blinding_map(map_path, blinder.mapping)
        written.append(map_path)
        blinding_section: dict[str, Any] = {
            "mode": BLINDING_MODE_STRIPPED,
            "salt_file": f"{SCORING_DIRNAME}/{SALT_FILE_NAME}",  # run-root-relative
            "salt_sha256": scoring_salt.sha256,
            "map_file": BLINDING_MAP_NAME,
            "map_sha256": map_sha256,
            "join_checksum": blinding_join_checksum(blinder.mapping, n_rows),
            "n_labels": len(blinder.mapping),
            "n_rows_unblinded": n_rows,
            "cache_note": _CACHE_CONTINUITY_NOTE,
        }
    else:
        blinding_section = {
            "mode": BLINDING_MODE_CONTROL,
            "note": ("NO label stripping — test-only control for the #130 "
                     "blinding-equivalence checksum; never a registered "
                     "confirmatory pass"),
        }

    scoring_manifest = {
        "scoring_run_id": scoring_run_id,
        "raw_run_id": raw_run_id,
        "raw_run_ledger_entries_sha256": entries_sha256,
        "n_evidence_files": len(evidence_files),
        "n_rows": n_rows,
        "windows": windows,
        # Task #130 decision (a) — §9.8 label stripping (cited by #112 prereg).
        "blinding": blinding_section,
        # Task #130 decision (c) — cache disclosure (audit H12): reuse of the
        # content-addressed work dir is recorded, never silently claimed as
        # independent scoring; --fresh forces a clean dir for drift audits.
        "cache_provenance": cache_provenance.to_dict(),
        **_provenance_block(
            args, evidence_files, len(all_items), len(scores), n_skipped
        ),
    }
    manifest_out = scoring_dir / SCORING_MANIFEST_NAME
    manifest_out.write_text(
        json.dumps(scoring_manifest, indent=2) + "\n", encoding="utf-8"
    )
    written.append(manifest_out)

    # §6: the pass gets its OWN ledger before stats may consume it.
    write_ledger(
        hash_artifacts(written, base_dir=scoring_dir),
        scoring_dir / "ledger.json",
    )

    print(f"INSTRUMENT_B_TREE_DONE  id={scoring_run_id}  "
          f"files={len(evidence_files)}  rows={n_rows}  "
          f"skipped_empty={n_skipped}  tau={args.tau}")
    print(f"  tree   : {scoring_dir}")
    print(f"  sealed : {scoring_dir / 'ledger.json'}")
    return 0


# ---------------------------------------------------------------------------
# Flat mode + entry point
# ---------------------------------------------------------------------------


def run_flat(args: argparse.Namespace) -> int:
    if args.out is None:
        print("ERROR: --out is required in flat mode (or use --scoring-run-id "
              "for the §6 tree layout)", file=sys.stderr)
        return 2
    evidence_files = _collect_flat_evidence(args.evidence)
    if not evidence_files:
        print(f"ERROR: no evidence files matched {args.evidence}", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Task #130 (a): flat-mode item ids embed the FULL evidence path, which
    # for pilot trees carries arm names (baselines/no_cache/...) — blind the
    # whole path component per file. The salt is sealed BESIDE the output
    # (there is no run root here) so re-running the same --out keeps tokens —
    # and the worker's checkpointed resume — deterministic.
    no_blinding = bool(getattr(args, "no_blinding_control", False))
    blinder: LabelBlinder | None = None
    scoring_salt = None
    if not no_blinding:
        scoring_salt = load_or_create_scoring_salt(
            Path(str(out_path) + ".blinding_salt.json")
        )
        blinder = LabelBlinder(scoring_salt.salt)

    all_items: list[dict[str, str]] = []
    blind_to_real: dict[str, str] = {}
    n_skipped = 0
    for ev_path in evidence_files:
        real_key = ev_path.as_posix()
        file_key = blinder.token(real_key) if blinder is not None else real_key
        items, skipped = _build_items(ev_path, file_key)
        if blinder is not None:
            _record_blind_ids(items, file_key, real_key, blind_to_real)
        all_items.extend(items)
        n_skipped += skipped
    if not all_items:
        print("ERROR: no scoreable rows (all empty-context/empty-answer)",
              file=sys.stderr)
        return 2
    if blinder is not None:
        _check_join_bijective(blind_to_real, len(all_items))

    env_home = Path(args.env_home).expanduser() if args.env_home else None
    scores, cache_provenance = ib.score_with_provenance(
        all_items,
        env_home=env_home,
        batch_size=args.batch,
        device=args.device,
        max_items=args.max_items,
        fresh=bool(getattr(args, "fresh", False)),
    )
    verdicts = ib.apply_tau(scores, args.tau) if args.tau is not None else None
    rows = _rows_for(all_items, scores, verdicts)
    # Task #130 (a): unblind at output time (checked bijective join).
    if blinder is not None:
        rows = _unblind_ib_rows(rows, blind_to_real)

    if blinder is not None and scoring_salt is not None:
        blinding_section: dict[str, Any] = {
            "mode": BLINDING_MODE_STRIPPED,
            "salt_file": str(scoring_salt.path),
            "salt_sha256": scoring_salt.sha256,
            # No pass dir to seal a map file into — the sidecar IS the flat
            # pass's manifest, so the token->path map rides inline.
            "mapping": blinder.mapping,
            "map_sha256": canonical_sha256(blinder.mapping),
            "join_checksum": blinding_join_checksum(blinder.mapping, len(rows)),
            "n_labels": len(blinder.mapping),
            "n_rows_unblinded": len(rows),
            "cache_note": _CACHE_CONTINUITY_NOTE,
        }
    else:
        blinding_section = {
            "mode": BLINDING_MODE_CONTROL,
            "note": ("NO label stripping — test-only control for the #130 "
                     "blinding-equivalence checksum"),
        }

    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    sidecar = Path(str(out_path) + ".provenance.json")
    sidecar.write_text(
        json.dumps(
            {
                # Task #130 decisions (a) + (c): blinding + cache disclosure.
                "blinding": blinding_section,
                "cache_provenance": cache_provenance.to_dict(),
                **_provenance_block(
                    args, evidence_files, len(all_items), len(scores), n_skipped
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"INSTRUMENT_B_DONE  files={len(evidence_files)}  rows={len(rows)}  "
          f"skipped_empty={n_skipped}  tau={args.tau}")
    print(f"  scores     : {out_path}")
    print(f"  provenance : {sidecar}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.bootstrap_only:
        env_home = Path(args.env_home).expanduser() if args.env_home else None
        home = ib.ensure_env(env_home)
        print(f"BOOTSTRAP_OK  env_home={home}")
        return 0

    if getattr(args, "abandon", None) is not None:
        # Task #130 decision (d): shared helper — pass dirs are owned jointly
        # with rescore_quality.py, so the same rename+tombstone semantics
        # apply regardless of which instrument wrote the pass.
        if args.scoring_run_id is not None or args.out is not None:
            print("ERROR: --abandon is exclusive with --scoring-run-id/--out",
                  file=sys.stderr)
            return 2
        if not args.evidence or len(args.evidence) != 1:
            print("ERROR: --abandon takes exactly ONE --evidence arg: the run "
                  "root owning scoring/", file=sys.stderr)
            return 2
        root = Path(args.evidence[0])
        if not root.is_dir():
            print(f"ERROR: run root {root} is not a directory", file=sys.stderr)
            return 2
        return rescore.run_abandon(root, args)

    if not args.evidence:
        print("ERROR: --evidence is required (or --bootstrap-only)",
              file=sys.stderr)
        return 2

    if args.scoring_run_id is not None:
        if len(args.evidence) != 1:
            print("ERROR: --scoring-run-id takes exactly ONE --evidence arg: "
                  "the sealed run root", file=sys.stderr)
            return 2
        root = Path(args.evidence[0])
        if not root.is_dir():
            print(f"ERROR: run root {root} is not a directory", file=sys.stderr)
            return 2
        return run_scoring_tree(root, args)

    return run_flat(args)


if __name__ == "__main__":
    raise SystemExit(main())
