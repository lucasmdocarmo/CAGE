"""§9.8 UPGRADE 2 — blinded analysis: arm-label scrambler + sealed map + unblind log.

The confirmatory analysis runs on scrambled arm labels until the full pipeline
output is frozen; the real↔blind mapping lives in a SEALED, sha256-keyed JSON
file (ledger-style: the file carries the hash of its own mapping so tampering
is detectable). Unblinding is a ONE-TIME dated event appended to a log; a
second unblind raises. Feasible precisely because D8 decoupled scoring from
serving — the scorer never needs to know which arm it is scoring.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_SEAL_VERSION = 1


class BlindingError(RuntimeError):
    """Base error for blinding-protocol violations."""


class AlreadyUnblindedError(BlindingError):
    """The sealed map was already unblinded — §9.8 allows exactly one event."""


class SealedMapTamperError(BlindingError):
    """The sealed map's content hash no longer matches its mapping."""


def _mapping_sha256(mapping: dict[str, str]) -> str:
    canonical = json.dumps(mapping, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def scramble_labels(
    df: pd.DataFrame,
    arm_col: str,
    seed: int,
    sealed_map_path: Path,
) -> tuple[pd.DataFrame, Path]:
    """Replace arm labels with seeded blind codes; seal the mapping to disk.

    Returns ``(blinded_df, sealed_map_path)``. The blinded frame is a copy with
    ``arm_col`` values replaced by opaque codes ``ARM-01..ARM-nn`` (assignment =
    seeded shuffle of the sorted real labels). The real→blind mapping is written
    ONLY to ``sealed_map_path`` as sha256-keyed JSON; it is never returned —
    holding the return value must not unblind anyone.

    Fails closed if ``sealed_map_path`` already exists (re-sealing over a live
    seal would orphan the first blinded frame).
    """
    if arm_col not in df.columns:
        raise BlindingError(f"arm_col {arm_col!r} not in DataFrame columns")
    sealed_map_path = Path(sealed_map_path)
    if sealed_map_path.exists():
        raise BlindingError(f"sealed map already exists: {sealed_map_path}")
    real_labels = sorted(str(v) for v in df[arm_col].dropna().unique())
    if len(real_labels) < 2:
        raise BlindingError(
            f"need >= 2 distinct arm labels to blind, got {len(real_labels)}"
        )
    if df[arm_col].isna().any():
        raise BlindingError(f"{arm_col!r} contains missing labels; blind input must be complete")
    rng = np.random.default_rng(seed)
    codes = [f"ARM-{i + 1:02d}" for i in range(len(real_labels))]
    mapping = dict(zip(real_labels, (codes[j] for j in rng.permutation(len(codes)))))
    sealed = {
        "seal_version": _SEAL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "arm_col": arm_col,
        "seed": seed,
        "map_sha256": _mapping_sha256(mapping),
        "mapping": mapping,
        "unblinded_utc": None,
    }
    sealed_map_path.parent.mkdir(parents=True, exist_ok=True)
    sealed_map_path.write_text(json.dumps(sealed, indent=2) + "\n", encoding="utf-8")
    blinded = df.copy()
    blinded[arm_col] = blinded[arm_col].astype(str).map(mapping)
    return blinded, sealed_map_path


def _load_sealed(sealed_map_path: Path) -> dict[str, object]:
    try:
        sealed = json.loads(sealed_map_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise BlindingError(f"sealed map not found: {sealed_map_path}") from None
    except json.JSONDecodeError as exc:
        raise SealedMapTamperError(f"sealed map is not valid JSON: {sealed_map_path}") from exc
    for key in ("map_sha256", "mapping", "arm_col"):
        if key not in sealed:
            raise SealedMapTamperError(f"sealed map missing key {key!r}: {sealed_map_path}")
    mapping = sealed["mapping"]
    if not isinstance(mapping, dict):
        raise SealedMapTamperError(f"sealed map 'mapping' is not an object: {sealed_map_path}")
    if _mapping_sha256(mapping) != sealed["map_sha256"]:
        raise SealedMapTamperError(
            f"sealed map hash mismatch — mapping was altered after sealing: {sealed_map_path}"
        )
    return sealed


def unblind(sealed_map_path: Path, log_path: Path) -> dict[str, str]:
    """One-time unblinding: verify the seal, log the dated event, return the map.

    Returns the blind→real mapping (inverse of the sealed real→blind map) for
    re-labeling the frozen pipeline output. Appends one dated event line to
    ``log_path`` and stamps ``unblinded_utc`` into the sealed file; calling a
    second time raises ``AlreadyUnblindedError``.
    """
    sealed_map_path = Path(sealed_map_path)
    log_path = Path(log_path)
    sealed = _load_sealed(sealed_map_path)
    if sealed.get("unblinded_utc") is not None:
        raise AlreadyUnblindedError(
            f"already unblinded at {sealed['unblinded_utc']}: {sealed_map_path}"
        )
    mapping: dict[str, str] = dict(sealed["mapping"])  # type: ignore[arg-type]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    event = {
        "event": "UNBLIND",
        "utc": now,
        "sealed_map": str(sealed_map_path),
        "map_sha256": sealed["map_sha256"],
        "arm_col": sealed["arm_col"],
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, sort_keys=True) + "\n")
    # Stamp the seal AFTER the log write: a crash between the two leaves an
    # extra log line, never a silently re-usable seal.
    sealed["unblinded_utc"] = now
    sealed_map_path.write_text(json.dumps(sealed, indent=2) + "\n", encoding="utf-8")
    return {blind: real for real, blind in mapping.items()}
