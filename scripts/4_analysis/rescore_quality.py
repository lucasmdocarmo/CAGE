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

Label stripping (task #130 decision (a), audit H9, charter §9.8): the scoring
computation NEVER sees arm identity. Arm-bearing identifiers (the evidence
"baseline" field; in tree mode the row_key directory the evidence came from)
are replaced by deterministic opaque tokens BEFORE any row reaches
QualityEvaluator; the final artifacts (qa_scores.jsonl, results_rescored.csv)
are unblinded at output time through a checked bijective join. Tokens derive
via HMAC-SHA256(salt, label): in tree mode the salt is created ONCE per run
root and stored sealed at scoring/blinding_salt.json (outside cells/ —
determinism across passes keeps instrument-B's content-addressed cache
shareable, decision (c)); legacy mode uses an ephemeral per-invocation salt
(nothing may be added to the pilot archive, decision (b)). Each tree pass
seals its token->label map (blinding_map.json) and stamps a "blinding"
section into scoring_manifest.json for the #112 prereg text.

Pilot archive is READ-ONLY (task #130 decision (b), audit H7; RESULTS_LAYOUT
§7): --apply REFUSES any run root under the pilot archive (the repo results/
tree by default; override via $CAGE_PILOT_ARCHIVE for tests) — sidecar
outputs (results_rescored.csv + accounting) are the ONLY rescore products
there. --apply keeps working for non-archive scratch trees.

Abandoning a crashed pass (task #130 decision (d), audit H10): --abandon
<scoring_run_id> renames scoring/<id>/ to scoring/<id>.abandoned-<UTCstamp>/
with a tombstone JSON inside, freeing the id for a clean retry. A pass with a
VERIFIED complete ledger is an audit record, not a failure — refused unless
--force-abandon.

Usage:
  python3 scripts/4_analysis/rescore_quality.py --run-root results/phase2/<run-id>
  python3 scripts/4_analysis/rescore_quality.py --run-root <run> --full --device cuda
  python3 scripts/4_analysis/rescore_quality.py --run-root results/camp1/a/<run_id> \
      --scoring-run-id s01-fast
  python3 scripts/4_analysis/rescore_quality.py --run-root <run> --full \
      --device cuda --batch-size 32   # cross-row batched scoring (D8 §8.1)
  python3 scripts/4_analysis/rescore_quality.py --run-root <run> \
      --abandon s01-fast --reason "worker OOM mid-pass"
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import os
import re
import secrets
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
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

#: Task #130 decision (a) — label-stripping layer (charter §9.8, audit H9).
#: The per-run-root salt lives at scoring/<SALT_FILE_NAME> (OUTSIDE cells/,
#: shared by rescore_quality AND score_instrument_b so tokens agree across
#: instruments and passes); each pass seals its token->label map as
#: <pass>/<BLINDING_MAP_NAME>.
SALT_FILE_NAME = "blinding_salt.json"
BLINDING_MAP_NAME = "blinding_map.json"
BLINDING_MODE_STRIPPED = "label-stripped"
BLINDING_MODE_CONTROL = "disabled-control-run"
_TOKEN_PREFIX = "blind-"
_TOKEN_HEX_LEN = 16

#: Task #130 decision (b) — the pilot archive root --apply must refuse.
#: Default: the repo's results/ tree (RESULTS_LAYOUT §7: pilot-era data is
#: read-only historical). Overridable for tests via $CAGE_PILOT_ARCHIVE.
PILOT_ARCHIVE_ENV = "CAGE_PILOT_ARCHIVE"

#: Task #130 decision (d) — tombstone written inside an abandoned pass dir.
ABANDONED_TOMBSTONE_NAME = "ABANDONED.json"
#: UTC stamp format appended to an abandoned pass directory name.
_ABANDON_STAMP_FMT = "%Y%m%dT%H%M%SZ"


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
    p.add_argument("--allow-duplicates", action="store_true", dest="allow_duplicates",
                   help="Duplicate (example_id, repeat_index, record_index) evidence "
                        "rows are REFUSED by default (task #127 integrity guard, "
                        "aligned with instrument_b_runner which raises). With this "
                        "flag: keep-LAST, and the dropped count is PERSISTED "
                        "(quality.json + scoring manifest in tree mode; a "
                        "<out-name>.accounting.json sidecar in legacy mode) -- "
                        "never stdout-only (charter §9.10).")
    p.add_argument("--abandon", default=None, metavar="SCORING_RUN_ID",
                   help="Task #130 decision (d): rename scoring/<id>/ under the "
                        "run root to scoring/<id>.abandoned-<UTCstamp>/ with a "
                        "tombstone JSON inside, freeing the id for a clean retry. "
                        "Requires --reason. A pass with a VERIFIED complete "
                        "ledger is an audit record and is refused unless "
                        "--force-abandon. Exclusive with --scoring-run-id/--apply.")
    p.add_argument("--reason", default=None,
                   help="Human-readable reason recorded in the --abandon "
                        "tombstone (required with --abandon).")
    p.add_argument("--force-abandon", action="store_true", dest="force_abandon",
                   help="Allow --abandon on a pass whose own ledger VERIFIES "
                        "complete (normally refused: completed passes are audit "
                        "record, not failures).")
    p.add_argument("--no-blinding-control", action="store_true",
                   dest="no_blinding_control",
                   help="TEST-ONLY control run: skip the #130 label-stripping "
                        "layer entirely. Exists so the blinding-equivalence "
                        "checksum test can compare a blinded pass against an "
                        "unblinded control bitwise. The scoring manifest records "
                        f"blinding mode {BLINDING_MODE_CONTROL!r}; never use for "
                        "a registered confirmatory pass.")
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

#: quality.json aggregation allowlist (task #127, audit H11): ONLY score columns
#: are aggregated. Provenance numerics riding in a score row (old_grounding_score,
#: record_index, faithfulness_premise_count, ...) must never fold into a mean.
#: sanitized_answer / faithfulness_premise_mode are string-valued and excluded;
#: the D8 §8.5 diagnostics that ARE rates (scored_windowed, the flag-gated
#: 3-class columns) and cache_relevance ride along explicitly.
_QUALITY_AGG_KEYS: tuple[str, ...] = tuple(
    [k for k in _MODEL_FREE_FIELDS if k != "sanitized_answer"]
    + [k for k in _MODEL_FIELDS if k != "faithfulness_premise_mode"]
    + ["scored_windowed", "cache_relevance",
       "faithfulness_contradiction", "faithfulness_neutral"]
)


# ---------------------------------------------------------------------------
# Task #130 decision (a): label stripping (charter §9.8, audit H9)
#
# Sealed-file pattern reused from src/analysis/stats/blinding.py (read-only
# authority for the §9.8 machinery): the file carries a sha256 of its own
# payload so tampering is detectable; loading verifies before trusting.
# Deliberately NOT importing that module here — its top level needs pandas,
# and fast-mode rescoring must keep working in a lean analysis venv.
# ---------------------------------------------------------------------------


class ScoringBlindingError(RuntimeError):
    """Label-stripping protocol violation in the offline scoring chain.

    Task #130 decision (a): the catastrophic failure mode is SILENT
    mis-assignment of scores to arms through a buggy unblind join — every
    integrity breach (tampered salt, unknown token, non-bijective join,
    token collision) raises this instead of degrading."""


class ScoringAbandonError(RuntimeError):
    """--abandon refusal (task #130 decision (d)): bad id, missing pass,
    missing reason, or a VERIFIED complete pass without --force-abandon."""


def canonical_sha256(obj: Any) -> str:
    """sha256 over the canonical (sorted, compact) JSON form of ``obj``."""
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ScoringSalt:
    """One run root's sealed label-stripping salt (task #130 decision (a))."""

    salt: bytes
    sha256: str  # the sealed file's self-hash (quoted by pass manifests)
    path: Path


def load_or_create_scoring_salt(salt_path: Path) -> ScoringSalt:
    """Load the sealed per-run-root salt, creating it on first use.

    Determinism is load-bearing (task #130 decisions (a)+(c)): tokens must be
    identical across passes so instrument-B's content-addressed work-dir cache
    stays shareable — a per-pass salt would silently defeat it. The salt is
    created ONCE (crypto-random, 32 bytes), sealed with a self-hash
    (blinding.py's sealed-file pattern), and every later pass re-verifies the
    seal before trusting it; a tampered or truncated salt file raises."""
    salt_path = Path(salt_path)
    if salt_path.exists():
        try:
            doc = json.loads(salt_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ScoringBlindingError(
                f"sealed salt file is not valid JSON: {salt_path}"
            ) from exc
        salt_hex = doc.get("salt_hex")
        if not isinstance(salt_hex, str) or not salt_hex:
            raise ScoringBlindingError(
                f"sealed salt file lacks 'salt_hex': {salt_path}"
            )
        if canonical_sha256(salt_hex) != doc.get("salt_sha256"):
            raise ScoringBlindingError(
                f"sealed salt self-hash mismatch — the salt file was altered "
                f"after sealing: {salt_path}"
            )
        try:
            salt = bytes.fromhex(salt_hex)
        except ValueError as exc:
            raise ScoringBlindingError(
                f"sealed salt 'salt_hex' is not hex: {salt_path}"
            ) from exc
        return ScoringSalt(salt=salt, sha256=doc["salt_sha256"], path=salt_path)

    salt = secrets.token_bytes(32)
    salt_hex = salt.hex()
    doc = {
        "seal_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": (
            "scoring label-stripping salt (charter §9.8; task #130 decision "
            "(a)) — HMAC-SHA256 key deriving opaque arm tokens; created once "
            "per run root so tokens (and instrument-B's content-addressed "
            "cache) stay deterministic across scoring passes"
        ),
        "salt_hex": salt_hex,
        "salt_sha256": canonical_sha256(salt_hex),
    }
    salt_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with salt_path.open("x", encoding="utf-8") as fh:
            fh.write(json.dumps(doc, indent=2) + "\n")
    except FileExistsError:
        # A concurrent pass won the create race — its salt is THE salt.
        return load_or_create_scoring_salt(salt_path)
    return ScoringSalt(salt=salt, sha256=doc["salt_sha256"], path=salt_path)


class LabelBlinder:
    """Deterministic arm-label -> opaque-token layer (task #130 decision (a)).

    ``token(label)`` = ``blind-`` + HMAC-SHA256(salt, label)[:16 hex]:
    deterministic for a fixed salt (cache continuity, decision (c)) and
    opaque without it. The instance accumulates the token->label map used by
    the pass's unblind join; a token collision between two DIFFERENT labels
    (astronomically unlikely, catastrophic if silent) raises."""

    def __init__(self, salt: bytes) -> None:
        if not salt:
            raise ScoringBlindingError("empty blinding salt")
        self._salt = salt
        self._token_to_label: dict[str, str] = {}

    def token(self, label: str) -> str:
        digest = hmac.new(
            self._salt, str(label).encode("utf-8"), hashlib.sha256
        ).hexdigest()
        tok = _TOKEN_PREFIX + digest[:_TOKEN_HEX_LEN]
        prior = self._token_to_label.get(tok)
        if prior is not None and prior != str(label):
            raise ScoringBlindingError(
                f"token collision: {tok!r} maps to both {prior!r} and "
                f"{label!r} — refusing a non-bijective unblind join"
            )
        self._token_to_label[tok] = str(label)
        return tok

    @property
    def mapping(self) -> dict[str, str]:
        """token -> real label, for the sealed per-pass map."""
        return dict(self._token_to_label)

    def unblind(self, token: str) -> str:
        try:
            return self._token_to_label[token]
        except KeyError:
            raise ScoringBlindingError(
                f"unknown blinding token {token!r} — a scored item does not "
                "unblind to any input row (refusing: silent mis-assignment of "
                "scores to arms is the failure mode this join check exists "
                "to make impossible)"
            ) from None


def ephemeral_blinder() -> LabelBlinder:
    """Per-invocation blinder for legacy/pilot rescoring (decision (b): the
    pilot archive gains NO new files, so the salt is never persisted there;
    determinism across passes is irrelevant without a shared cache)."""
    return LabelBlinder(secrets.token_bytes(32))


def unblind_score_rows(
    rows: list[dict[str, Any]], blinder: LabelBlinder
) -> list[dict[str, Any]]:
    """Unblind the ``baseline`` column of score rows — the checked join.

    Every row must carry a token that resolves through the (injective, checked
    at token() time) map: rows unblind 1:1 to their input rows or the join
    raises loudly. A row that somehow bypassed blinding (real label where a
    token belongs) also raises — a half-blinded pass proves nothing."""
    out: list[dict[str, Any]] = []
    for row in rows:
        tok = row.get("baseline")
        if not isinstance(tok, str) or not tok.startswith(_TOKEN_PREFIX):
            raise ScoringBlindingError(
                f"score row baseline {tok!r} is not a blinding token — the "
                "row bypassed the label-stripping layer (task #130 (a))"
            )
        new = dict(row)
        new["baseline"] = blinder.unblind(tok)
        out.append(new)
    return out


def blinding_join_checksum(mapping: dict[str, str], n_rows_unblinded: int) -> str:
    """The pass manifest's join checksum: pins WHICH token->label map was
    applied to HOW MANY rows (cited by the #112 prereg text)."""
    return canonical_sha256(
        {"mapping": mapping, "n_rows_unblinded": n_rows_unblinded}
    )


def write_blinding_map(map_path: Path, mapping: dict[str, str]) -> str:
    """Seal the pass's token->label map (self-hashed); returns map_sha256."""
    doc = {
        "seal_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": (
            "token->label unblind map for ONE scoring pass (charter §9.8; "
            "task #130 decision (a)) — the sealed record of the join that "
            "unblinded this pass's output artifacts"
        ),
        "map_sha256": canonical_sha256(mapping),
        "mapping": mapping,
    }
    map_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    return doc["map_sha256"]


# ---------------------------------------------------------------------------
# Task #130 decision (b): pilot-archive read-only guard for --apply
# ---------------------------------------------------------------------------


def _pilot_archive_root() -> Path:
    """The read-only pilot archive root (RESULTS_LAYOUT §7).

    Default: the repo's results/ tree. $CAGE_PILOT_ARCHIVE overrides —
    explicit and test-friendly, never a heuristic."""
    raw = os.environ.get(PILOT_ARCHIVE_ENV)
    root = Path(raw).expanduser() if raw else REPO_ROOT / "results"
    return root.resolve()


def _is_under_pilot_archive(path: Path) -> bool:
    try:
        Path(path).resolve().relative_to(_pilot_archive_root())
    except ValueError:
        return False
    return True


# ---------------------------------------------------------------------------
# Task #130 decision (d): --abandon (audit H10 — a crashed pass permanently
# blocked its scoring_run_id; completed passes stay untouchable audit record)
# ---------------------------------------------------------------------------


def abandon_scoring_pass(
    root: Path, scoring_run_id: str, *, reason: str, force: bool = False
) -> Path:
    """Rename scoring/<id>/ to scoring/<id>.abandoned-<UTCstamp>/ + tombstone.

    Frees ``scoring_run_id`` for a clean retry (§6 passes are append-only, so
    a crashed pass otherwise burns its id forever — audit H10). Refuses a
    pass whose own ledger VERIFIES complete unless ``force``: completed
    passes are audit record, not failures. Returns the renamed path. The
    tombstone records the reason, the timestamp, and what was present."""
    from src.analysis.stats.ledger import LedgerError, verify_ledger

    if not SCORING_RUN_ID_RE.match(scoring_run_id):
        raise ScoringAbandonError(
            f"scoring run id {scoring_run_id!r} violates the §6 grammar "
            f"{SCORING_RUN_ID_RE.pattern} — refusing (an id with separators "
            "could escape scoring/)"
        )
    if not isinstance(reason, str) or not reason.strip():
        raise ScoringAbandonError(
            "--abandon requires a non-empty --reason (the tombstone is the "
            "§9.10 record of WHY the pass was abandoned)"
        )
    pass_dir = Path(root) / SCORING_DIRNAME / scoring_run_id
    if not pass_dir.is_dir():
        raise ScoringAbandonError(f"no scoring pass to abandon: {pass_dir}")

    ledger_path = pass_dir / "ledger.json"
    if not ledger_path.is_file():
        ledger_state = "absent"
    else:
        try:
            mismatches = verify_ledger(ledger_path, pass_dir)
        except LedgerError as exc:
            ledger_state = f"corrupt ({exc})"
        else:
            ledger_state = "verified" if not mismatches else (
                f"mismatched ({len(mismatches)} line(s))"
            )
    if ledger_state == "verified" and not force:
        raise ScoringAbandonError(
            f"{pass_dir} has a VERIFIED complete ledger — completed passes "
            "are audit record, not failures (RESULTS_LAYOUT §6). Pass "
            "--force-abandon to abandon it anyway."
        )

    now = datetime.now(timezone.utc)
    target = pass_dir.with_name(
        f"{scoring_run_id}.abandoned-{now.strftime(_ABANDON_STAMP_FMT)}"
    )
    if target.exists():
        raise ScoringAbandonError(
            f"{target} already exists (two abandons of the same id within "
            "one second?) — retry"
        )

    # Census BEFORE the rename: the tombstone honestly records what was there.
    all_files = [p for p in sorted(pass_dir.rglob("*")) if p.is_file()]
    tombstone = {
        "schema_version": 1,
        "scoring_run_id": scoring_run_id,
        "abandoned_utc": now.isoformat(timespec="seconds"),
        "reason": reason.strip(),
        "forced": bool(force),
        "ledger_state": ledger_state,
        "present": {
            "manifest": (pass_dir / SCORING_MANIFEST_NAME).is_file(),
            "ledger": ledger_path.is_file(),
            "n_files": len(all_files),
            "n_cell_files": len(
                [p for p in all_files if "cells" in p.relative_to(pass_dir).parts]
            ),
        },
    }
    pass_dir.rename(target)
    (target / ABANDONED_TOMBSTONE_NAME).write_text(
        json.dumps(tombstone, indent=2) + "\n", encoding="utf-8"
    )
    return target


def run_abandon(root: Path, args: argparse.Namespace) -> int:
    """CLI wrapper for --abandon (shared with score_instrument_b.py)."""
    try:
        target = abandon_scoring_pass(
            root,
            args.abandon,
            reason=args.reason or "",
            force=bool(getattr(args, "force_abandon", False)),
        )
    except ScoringAbandonError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"ABANDONED  id={args.abandon}")
    print(f"  moved to : {target}")
    print(f"  tombstone: {target / ABANDONED_TOMBSTONE_NAME}")
    print(f"  the id {args.abandon!r} is free for a clean retry")
    return 0


def _json_default(obj: Any) -> Any:
    """qa_scores.jsonl JSON fallback (task #127, audit H3 'default=str drift'):
    numpy scalars/arrays become native numbers/lists, never quoted strings;
    everything else falls back to str exactly as the historical default=str."""
    import numpy as np

    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _evidence_row_key(rec: dict[str, Any]) -> tuple[Any, str, Any]:
    """Identity of one evidence row: (example_id, repeat_index, record_index).

    record_index is the open-loop replay disambiguator stamped by
    run_experiment.py (task #127); pre-#127 evidence lacks the field -> None,
    so genuinely replayed rows in old files collide -- which is exactly the
    ambiguity the duplicate guard must surface, not paper over."""
    return (
        rec.get("example_id"),
        str(rec.get("repeat_index") or "0"),
        rec.get("record_index"),
    )


class DuplicateEvidenceError(ValueError):
    """Duplicate (example_id, repeat_index, record_index) evidence rows.

    Task #127 (audit H3): identical triples mean the SAME logical row appears
    more than once; silently scoring both (or keeping an arbitrary one) is an
    uncounted exclusion (charter §9.10). Refuse by default -- aligned with
    instrument_b_runner, which raises on duplicate item ids -- unless the
    operator passes --allow-duplicates (keep-last, dropped count persisted)."""

    def __init__(self, ev_path: Path, dup_counts: dict[tuple[Any, str, Any], int]) -> None:
        self.ev_path = ev_path
        self.dup_counts = dup_counts
        n_extra = sum(c - 1 for c in dup_counts.values())
        shown = sorted(dup_counts, key=repr)[:10]
        more = "" if len(dup_counts) <= 10 else f" (+{len(dup_counts) - 10} more keys)"
        super().__init__(
            f"{ev_path}: {len(dup_counts)} duplicate (example_id, repeat_index, "
            f"record_index) key(s) / {n_extra} extra row(s): {shown}{more}. "
            "Pass --allow-duplicates to keep-last (dropped rows are persisted "
            "as n_duplicates_dropped, charter §9.10)."
        )


def _duplicate_key_counts(ev_path: Path) -> dict[tuple[Any, str, Any], int]:
    """Duplicate-key census of one evidence file (keys seen more than once)."""
    keys = [
        _evidence_row_key(json.loads(line))
        for line in ev_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {k: c for k, c in Counter(keys).items() if c > 1}


def _fmt_cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v)
    return str(v)


def _write_rescored_csv(out_path: Path, rows_out: list[dict[str, Any]]) -> None:
    """Write one results_rescored.csv (legacy sidecar layout).

    Header = union of keys across ALL rows (first-seen order), not row 0's
    keys: QualityMetrics.to_dict() is row-dependent (hallucination_detected
    appears only when LettuceDetect returns a verdict), so row 0
    under-specifies the header. Live failure 2026-07-16: "dict contains
    fields not in fieldnames". Extracted (task #130) so the blinding
    no-blinding-control checksum test writes its control CSV through the
    EXACT same writer."""
    fieldnames = list(rows_out[0].keys())
    seen = set(fieldnames)
    for row in rows_out[1:]:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows_out)


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
    ev_path: Path,
    evaluator: Any,
    batch_size: int | None = None,
    allow_duplicates: bool = False,
    blinder: LabelBlinder | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Score every record of one qa_evidence.jsonl; shared by both layouts.

    Returns ``(rows_out, n_duplicates_dropped)``. Duplicate
    (example_id, repeat_index, record_index) rows RAISE
    ``DuplicateEvidenceError`` unless ``allow_duplicates`` (then: keep-LAST,
    and the dropped count is returned so callers persist it -- task #127,
    charter §9.10: exclusions countable from artifacts).

    ``blinder`` (task #130 decision (a), charter §9.8): when given, the
    arm-bearing cell identity (the evidence ``baseline`` field, else the
    row_key directory name) is replaced by its opaque token in the emitted
    rows — nothing inside the scoring boundary carries arm identity; callers
    unblind at output time via :func:`unblind_score_rows` (checked join).

    Behavior is otherwise byte-identical to the historical inline loop: B4
    sanitized abstention gate, M5 all-answers max-over-golds, B3d dual scoring
    against pre-compression originals when the evidence carries them.

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
        return [], 0

    # Duplicate guard (task #127, audit H3): identical row keys are refused
    # unless --allow-duplicates, in which case the LAST occurrence wins and the
    # drop is counted for persistence (never stdout-only, §9.10).
    keys = [_evidence_row_key(r["rec"]) for r in records]
    dup_counts = {k: c for k, c in Counter(keys).items() if c > 1}
    n_duplicates_dropped = 0
    if dup_counts:
        if not allow_duplicates:
            raise DuplicateEvidenceError(ev_path, dup_counts)
        last_index: dict[tuple[Any, str, Any], int] = {}
        for i, key in enumerate(keys):
            last_index[key] = i
        keep = sorted(last_index.values())
        n_duplicates_dropped = len(records) - len(keep)
        print(f"[rescore] {ev_path}: {len(dup_counts)} duplicate (example_id, "
              f"repeat_index, record_index) key(s); kept LAST, dropped "
              f"{n_duplicates_dropped} row(s) (--allow-duplicates)")
        records = [records[i] for i in keep]

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

        # Task #130 (a): the cell identity is arm-bearing (either the evidence
        # "baseline" field or the cells/<row_key>/ directory) — with a blinder
        # the row carries its opaque token, never the real label.
        cell_source = rec.get("baseline") or ev_path.parent.parent.name
        cell = blinder.token(str(cell_source)) if blinder is not None else cell_source

        rows_out.append({
            "example_id": rec.get("example_id"),
            "baseline": cell,
            "trial_dir": ev_path.parent.name,
            "repeat_index": str(rec.get("repeat_index") or "0"),
            # Task #127: the open-loop replay disambiguator (None on closed-loop
            # rows and pre-#127 evidence) rides into the score rows so a
            # keep-last dedup pass stays reconstructible from qa_scores.jsonl.
            "record_index": rec.get("record_index"),
            "generated_answer": generated,
            "reference_answer": reference,
            "abstained": abstained,
            "old_grounding_score": old_g,
            **metrics,
            # B3d: scores vs the PRE-compression originals ("" when unavailable).
            "faithfulness_source": faithfulness_source,
            "grounding_source": grounding_source,
        })
    return rows_out, n_duplicates_dropped


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


def _quality_aggregate(
    rows_out: list[dict[str, Any]], n_duplicates_dropped: int = 0
) -> dict[str, Any]:
    """Per-window quality.json: per-metric mean WITH its denominators.

    Task #127 (audit H11): every aggregated key carries ``n`` (rows that
    actually scored) and ``n_none`` (rows missing/None for that key) alongside
    ``mean`` -- a metric scored on 3/500 rows must be distinguishable from full
    coverage. Aggregation is restricted to the ``_QUALITY_AGG_KEYS`` allowlist
    so provenance numerics (old_grounding_score, record_index, ...) never fold
    into a mean. ``mean`` is None when no row scored -- absence is not zero.
    """
    n_abstained = sum(1 for r in rows_out if r.get("abstained"))
    metrics: dict[str, dict[str, Any]] = {}
    for key in sorted(_QUALITY_AGG_KEYS):
        values = [
            float(row[key])
            for row in rows_out
            if isinstance(row.get(key), (int, float))
        ]
        metrics[key] = {
            "mean": (sum(values) / len(values)) if values else None,
            "n": len(values),
            "n_none": len(rows_out) - len(values),
        }
    return {
        "rows": len(rows_out),
        "abstained": n_abstained,
        # §9.10: keep-last dedup losses are persisted here, not stdout-only.
        "n_duplicates_dropped": n_duplicates_dropped,
        # Back-compat convenience view (scored keys only, same allowlist —
        # provenance numerics no longer fold in); "metrics" with its explicit
        # n / n_none denominators is the authoritative form.
        "means": {k: m["mean"] for k, m in metrics.items() if m["n"] > 0},
        "metrics": metrics,
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

    # getattr: hand-built Namespaces predating the newer flags (and older
    # callers) keep the historical defaults — fail-open to the historical
    # behavior, never to a new one.
    batch_size = getattr(args, "batch_size", None)
    allow_duplicates = bool(getattr(args, "allow_duplicates", False))

    # Duplicate guard pre-scan (task #127, audit H3) BEFORE the tree is
    # created: scoring ids are append-only, so a refusal mid-pass would burn
    # the id on a partial tree. Refuse here, with counts and offending keys.
    if not allow_duplicates:
        for ev_path in evidence_files:
            dup_counts = _duplicate_key_counts(ev_path)
            if dup_counts:
                print(f"ERROR: {DuplicateEvidenceError(ev_path, dup_counts)}",
                      file=sys.stderr)
                return 2

    # Task #130 decision (a): label stripping is the DEFAULT. The per-run-root
    # salt (sealed, shared with score_instrument_b.py) keeps tokens — and the
    # instrument-B cache — deterministic across passes. The control escape
    # hatch exists ONLY for the blinding-equivalence checksum test and is
    # recorded loudly in the manifest.
    no_blinding = bool(getattr(args, "no_blinding_control", False))
    blinder: LabelBlinder | None = None
    scoring_salt: ScoringSalt | None = None
    if not no_blinding:
        scoring_salt = load_or_create_scoring_salt(
            root / SCORING_DIRNAME / SALT_FILE_NAME
        )
        blinder = LabelBlinder(scoring_salt.salt)

    evaluator = _build_evaluator(args)
    written: list[Path] = []
    total_rows = 0
    total_abstained = 0
    total_duplicates_dropped = 0

    scoring_dir.mkdir(parents=True)
    for ev_path in evidence_files:
        rows_out, n_dup = _score_evidence_file(
            ev_path, evaluator, batch_size=batch_size,
            allow_duplicates=allow_duplicates, blinder=blinder,
        )
        # Task #130 (a): the unblind join — output artifacts carry REAL
        # labels; the join is bijective and checked (loud failure).
        if blinder is not None:
            rows_out = unblind_score_rows(rows_out, blinder)
        rel_window = ev_path.parent.relative_to(root)  # cells/<row_key>/window_<k>
        out_window = scoring_dir / rel_window
        out_window.mkdir(parents=True, exist_ok=True)

        scores_path = out_window / QA_SCORES_NAME
        with scores_path.open("w", encoding="utf-8") as fh:
            for row in rows_out:
                fh.write(json.dumps(row, default=_json_default) + "\n")
        written.append(scores_path)

        quality_path = out_window / QUALITY_JSON_NAME
        quality_path.write_text(
            json.dumps(
                _quality_aggregate(rows_out, n_duplicates_dropped=n_dup), indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        written.append(quality_path)

        total_rows += len(rows_out)
        total_abstained += sum(1 for r in rows_out if r.get("abstained"))
        total_duplicates_dropped += n_dup

    # Task #130 (a): seal this pass's token->label map and stamp the
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
            "join_checksum": blinding_join_checksum(blinder.mapping, total_rows),
            "n_labels": len(blinder.mapping),
            "n_rows_unblinded": total_rows,
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
        # F9/#147 revision provenance: instrument -> {model, revision} with the
        # resolved HF commit hash captured at lazy-load time (None when the
        # instrument never loaded this pass or the hash was unresolvable) --
        # a repo NAME alone lets a silent upstream update change the
        # instrument under the same provenance id.
        "instrument_revisions": evaluator.instrument_provenance(),
        "calibration_id": evaluator.calibration_id,
        "raw_run_id": raw_run_id,
        "raw_run_ledger_entries_sha256": entries_sha256,
        "n_evidence_files": len(evidence_files),
        "n_rows": total_rows,
        # Task #127 duplicate accounting (§9.10: countable from artifacts).
        "allow_duplicates": allow_duplicates,
        "n_duplicates_dropped": total_duplicates_dropped,
        # Task #130 decision (a) — §9.8 label stripping (cited by #112 prereg).
        "blinding": blinding_section,
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

    if getattr(args, "abandon", None) is not None:
        # Task #130 decision (d): --abandon is its own operation — combining
        # it with a scoring run would blur which pass the flags describe.
        if args.scoring_run_id is not None or args.apply:
            print("ERROR: --abandon is exclusive with --scoring-run-id/--apply",
                  file=sys.stderr)
            return 2
        return run_abandon(root, args)

    # Task #130 decision (b): the pilot archive is READ-ONLY (RESULTS_LAYOUT
    # §7); --apply mutates results.csv in place and is refused fail-fast for
    # any run root inside it. Sidecar outputs are the only rescore products
    # there; --apply keeps working for non-archive scratch trees.
    if args.apply and _is_under_pilot_archive(root):
        print(f"ERROR: --apply refused: {root} is inside the read-only pilot "
              f"archive ({_pilot_archive_root()}) — RESULTS_LAYOUT §7: "
              "pilot-era data stays exactly where it is, read-only; sidecar "
              "outputs (results_rescored.csv + accounting) are the ONLY "
              "rescore products there (task #130 decision (b)). Copy the tree "
              f"to a scratch dir to --apply, or unset ${PILOT_ARCHIVE_ENV} "
              "override if this is not the archive.", file=sys.stderr)
        return 2

    if args.scoring_run_id is not None:
        return run_scoring_tree(root, args.scoring_run_id, args)

    evidence_files = sorted(root.rglob("qa_evidence.jsonl"))
    if not evidence_files:
        print(f"ERROR: no qa_evidence.jsonl under {root}", file=sys.stderr)
        return 2

    evaluator = _build_evaluator(args)
    batch_size = getattr(args, "batch_size", None)  # see run_scoring_tree note
    allow_duplicates = bool(getattr(args, "allow_duplicates", False))
    # Task #130 (a): legacy passes blind too (ephemeral salt — decision (b)
    # forbids persisting anything new into the pilot archive; determinism
    # across passes is irrelevant without a shared cache).
    blinder = (
        None if getattr(args, "no_blinding_control", False) else ephemeral_blinder()
    )

    total_rows = 0
    total_abstained = 0
    total_duplicates_dropped = 0
    newly_na_grounding = 0  # abstained rows the ORIGINAL run had scored with a grounding number
    per_cell: dict[str, list[int]] = {}

    for ev_path in evidence_files:
        try:
            rows_out, n_dup = _score_evidence_file(
                ev_path, evaluator, batch_size=batch_size,
                allow_duplicates=allow_duplicates, blinder=blinder,
            )
        except DuplicateEvidenceError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        # Task #130 (a): unblind at output time (checked bijective join) —
        # results_rescored.csv and the accounting stay real-labeled.
        if blinder is not None:
            rows_out = unblind_score_rows(rows_out, blinder)
        total_duplicates_dropped += n_dup
        if n_dup:
            # §9.10: the keep-last drop must be countable from an artifact,
            # not stdout -- sidecar next to this trial's rescored CSV.
            accounting_path = ev_path.parent / f"{args.out_name}.accounting.json"
            accounting_path.write_text(
                json.dumps(
                    {
                        "evidence_file": ev_path.name,
                        "n_rows_scored": len(rows_out),
                        "n_duplicates_dropped": n_dup,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
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
            _write_rescored_csv(ev_path.parent / args.out_name, rows_out)
            if args.apply:
                # Task #130 (b) belt-and-suspenders: the fail-fast root guard
                # above covers the tree, but a symlinked trial dir could still
                # resolve into the archive — refuse per trial too.
                if _is_under_pilot_archive(ev_path.parent):
                    print(f"ERROR: --apply refused for {ev_path.parent}: "
                          "resolves inside the read-only pilot archive "
                          "(RESULTS_LAYOUT §7; task #130 decision (b))",
                          file=sys.stderr)
                    return 2
                n_upd = _apply_to_results_csv(ev_path.parent, rows_out, args.full)
                print(f"  applied -> {ev_path.parent}/results.csv ({n_upd} rows updated)")

    mode = "FULL" if args.full else "FAST (model-free metrics + abstention short-circuit)"
    print(f"RESCORE_DONE  mode={mode}  files={len(evidence_files)}  rows={total_rows}")
    if total_duplicates_dropped:
        print(f"  duplicate rows dropped (keep-last, --allow-duplicates): "
              f"{total_duplicates_dropped}")
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
