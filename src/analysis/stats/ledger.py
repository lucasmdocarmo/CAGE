"""§9.10 UPGRADE 5 — data sealing: the content-hash ledger.

Every raw measurement artifact is content-hashed at write time; the ledger is
committed immediately post-run, BEFORE any analysis — the data is provably
untouched afterward. ``verify_ledger`` re-hashes the tree and returns the list
of mismatches (empty = intact); the ledger file carries a hash of its own
entries so tampering with the LEDGER (as opposed to the data) raises rather
than verifying.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

_ALGORITHM = "sha256"
_CHUNK_BYTES = 1 << 16
_LEDGER_VERSION = 1


class LedgerError(RuntimeError):
    """Ledger-protocol violation: unreadable input, seal conflict, or tampering."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _entries_sha256(entries: Mapping[str, str]) -> str:
    canonical = json.dumps(dict(entries), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def hash_artifacts(
    paths: Iterable[Path], *, base_dir: Path | None = None
) -> dict[str, str]:
    """Hash artifact files -> ``{relpath: sha256}`` (posix keys, sorted).

    ``base_dir`` anchors the relative keys (required whenever the ledger will
    be verified from a different working directory — i.e. always for campaign
    data; defaults to keys-as-given for ad-hoc use). Missing files, directories
    and duplicate keys fail loud: a ledger with a hole proves nothing.
    """
    resolved_base = Path(base_dir).resolve() if base_dir is not None else None
    entries: dict[str, str] = {}
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise LedgerError(f"artifact does not exist: {path}")
        if not path.is_file():
            raise LedgerError(f"artifact is not a regular file: {path}")
        if resolved_base is not None:
            try:
                key = path.resolve().relative_to(resolved_base).as_posix()
            except ValueError as exc:
                raise LedgerError(
                    f"artifact {path} is outside base_dir {resolved_base}"
                ) from exc
        else:
            key = path.as_posix()
        if key in entries:
            raise LedgerError(f"duplicate artifact key {key!r}")
        entries[key] = _sha256_file(path)
    if not entries:
        raise LedgerError("no artifacts to hash — an empty ledger seals nothing")
    return dict(sorted(entries.items()))


def write_ledger(ledger: Mapping[str, str], path: Path) -> Path:
    """Seal the ledger to ``path`` as JSON; refuses to overwrite an existing seal.

    Overwriting a committed ledger is exactly the drift §9.10 exists to
    prevent — a re-run gets a NEW ledger path (or an amendment-log entry).
    """
    path = Path(path)
    if path.exists():
        raise LedgerError(f"ledger already exists (sealed): {path}")
    if not ledger:
        raise LedgerError("refusing to write an empty ledger")
    entries = dict(sorted(ledger.items()))
    document = {
        "ledger_version": _LEDGER_VERSION,
        "algorithm": _ALGORITHM,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "entries_sha256": _entries_sha256(entries),
        "entries": entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_ledger(path: Path) -> dict[str, str]:
    """Load a sealed ledger's entries, verifying the ledger's own self-hash."""
    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise LedgerError(f"ledger not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise LedgerError(f"ledger is not valid JSON: {path}") from exc
    for key in ("algorithm", "entries", "entries_sha256"):
        if key not in document:
            raise LedgerError(f"ledger missing key {key!r}: {path}")
    if document["algorithm"] != _ALGORITHM:
        raise LedgerError(f"unsupported ledger algorithm {document['algorithm']!r}")
    entries = document["entries"]
    if not isinstance(entries, dict):
        raise LedgerError(f"ledger 'entries' is not an object: {path}")
    if _entries_sha256(entries) != document["entries_sha256"]:
        raise LedgerError(
            f"ledger self-hash mismatch — the ledger itself was altered: {path}"
        )
    return dict(entries)


def verify_ledger(path: Path, base_dir: Path) -> list[str]:
    """Re-hash the tree against a sealed ledger; returns mismatches (empty = intact).

    Each mismatch is one human-readable line: ``MISSING <relpath>`` or
    ``HASH-MISMATCH <relpath> expected=<sha256> got=<sha256>``. A tampered
    LEDGER raises instead (an untrustworthy reference cannot certify anything).
    """
    entries = read_ledger(path)
    base = Path(base_dir)
    mismatches: list[str] = []
    for relpath, expected in sorted(entries.items()):
        artifact = base / relpath
        if not artifact.is_file():
            mismatches.append(f"MISSING {relpath}")
            continue
        actual = _sha256_file(artifact)
        if actual != expected:
            mismatches.append(f"HASH-MISMATCH {relpath} expected={expected} got={actual}")
    return mismatches
