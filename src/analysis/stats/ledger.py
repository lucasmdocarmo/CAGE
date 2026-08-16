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
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

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


def _require_relative_key(key: str, context: Path | str) -> None:
    """Reject non-relative ledger keys (§5: keys are relative to the run root).

    ``Path(base_dir) / key`` silently DISCARDS base_dir when ``key`` is
    absolute, so an absolute-key ledger would verify against whatever happens
    to live at that absolute path on the verifying machine — undefined
    provenance. ``..`` segments likewise escape the sealed root. Both are
    protocol violations, refused at write AND verify time.
    """
    if Path(key).is_absolute() or key.startswith("/"):
        raise LedgerError(
            f"ledger key {key!r} is absolute — §5 keys must be relative to the "
            f"run root (base_dir would be silently ignored on join): {context}"
        )
    if ".." in Path(key).parts:
        raise LedgerError(
            f"ledger key {key!r} contains '..' — a key must resolve INSIDE the "
            f"sealed root: {context}"
        )


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
    Keys must be relative posix paths without ``..`` (§5: relative to the run
    root); absolute keys are refused HERE because ``verify_ledger`` cannot
    anchor them to any base_dir (use ``hash_artifacts(..., base_dir=...)``).
    """
    path = Path(path)
    if path.exists():
        raise LedgerError(f"ledger already exists (sealed): {path}")
    if not ledger:
        raise LedgerError("refusing to write an empty ledger")
    for key in ledger:
        _require_relative_key(key, path)
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


def verify_ledger(
    path: Path, base_dir: Path, *, extra_roots: Sequence[Path] | None = None
) -> list[str]:
    """Re-hash the tree against a sealed ledger; returns mismatches (empty = intact).

    Each mismatch is one human-readable line: ``MISSING <relpath>``,
    ``HASH-MISMATCH <relpath> expected=<sha256> got=<sha256>``, or (only when
    ``extra_roots`` is given) ``EXTRA <relpath>``. A tampered LEDGER raises
    instead (an untrustworthy reference cannot certify anything), as does a
    ledger whose keys are absolute or contain ``..`` — ``base_dir / key``
    silently ignores base_dir for an absolute key, so such a seal has
    undefined provenance (§5: keys are relative to the run root).

    ``extra_roots`` (task #129, H7 seal extra-file blindness): directories —
    each inside ``base_dir`` — whose ENTIRE file contents must be covered by
    the seal. Every regular file found under them (directory symlinks
    followed; cycles guarded) whose base-relative path is not a ledger entry
    is reported as ``EXTRA`` — a file added to a sealed tree after the seal.
    Callers scope the sweep to the append-only-then-immutable subtrees (e.g.
    the run root's ``cells/``) precisely so the §6 scoring layout — legal
    POST-seal siblings like ``scoring/``, ``index/``, ``analysis/`` at the
    run root — is never falsely flagged. Existing callers that do not pass
    ``extra_roots`` keep the historical MISSING/HASH-MISMATCH-only behavior.
    """
    path = Path(path)
    entries = read_ledger(path)
    base = Path(base_dir)
    mismatches: list[str] = []
    for relpath, expected in sorted(entries.items()):
        _require_relative_key(relpath, path)
        artifact = base / relpath
        if not artifact.is_file():
            mismatches.append(f"MISSING {relpath}")
            continue
        actual = _sha256_file(artifact)
        if actual != expected:
            mismatches.append(f"HASH-MISMATCH {relpath} expected={expected} got={actual}")
    if extra_roots is not None:
        mismatches.extend(
            _find_extra_files(entries, base, extra_roots, ledger_path=path)
        )
    return mismatches


def _find_extra_files(
    entries: Mapping[str, str],
    base: Path,
    extra_roots: Sequence[Path],
    *,
    ledger_path: Path,
) -> list[str]:
    """Walk ``extra_roots`` and report files absent from ``entries`` as EXTRA.

    Uses ``os.walk(followlinks=True)`` (a plain rglob is blind to directory
    symlinks — the H7 contamination-sweep bug class) with a realpath
    visited-set so a symlink cycle terminates instead of recursing forever.
    Sweep roots must exist and lie inside ``base_dir`` (fail loud otherwise —
    a mistyped root silently sweeping nothing would prove nothing).
    """
    base_resolved = base.resolve()
    ledger_abs = ledger_path.resolve()
    known = set(entries)
    extras: set[str] = set()
    for raw_root in extra_roots:
        root = Path(raw_root)
        if not root.is_dir():
            raise LedgerError(f"extra-file sweep root is not a directory: {root}")
        try:
            root.resolve().relative_to(base_resolved)
        except ValueError:
            raise LedgerError(
                f"extra-file sweep root {root} is outside base_dir {base_resolved}"
            ) from None
        visited: set[str] = set()
        for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
            real = os.path.realpath(dirpath)
            if real in visited:
                dirnames[:] = []  # symlink cycle: prune, do not recurse forever
                continue
            visited.add(real)
            for name in filenames:
                file_path = Path(dirpath) / name
                if file_path.resolve() == ledger_abs:
                    continue  # the seal itself is not a sealed artifact
                key = Path(os.path.relpath(file_path, base)).as_posix()
                if key not in known:
                    extras.add(f"EXTRA {key}")
    return sorted(extras)
