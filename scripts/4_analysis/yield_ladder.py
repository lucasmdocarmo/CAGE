#!/usr/bin/env python3
"""Order:     stage 4 helper — consumed by run_campaign_analysis.py
Objective: the S1 yield ladder + independence null made VISIBLE (T6.2)
Cloud:     local

The charter-S1 ladder renderer: per-window machine (JSON rows) and human
(markdown) tables of the full serving-yield ladder that ``goodput.py``
computes and carries but — until T6.2 — nothing printed.

Charter bindings (PUBLICATION.md S1, binding on EVERY figure/table):

- (a) ladder mandate: Y never appears without raw throughput and G beside
  it. Enforced STRUCTURALLY: this module is the only ladder-table producer
  and it refuses any row lacking ``throughput_rps``/``goodput_rps``/
  ``yield_rps`` — a caller cannot print Y alone.
- (b) independence null: the null G·E[v] (``independence_null_rps``) is
  printed beside every Y, with the covariance gap Y − G·E[v]
  (``covariance_gap_rps``) beside it — the covariance gap IS the finding.
- §6.6 never-mixed-bases guard: before the pooled multi-window table is
  rendered, EVERY row passes ``goodput.assert_single_basis`` on the table's
  single declared basis — an unlabeled or mixed-basis pool refuses with the
  §6.6 citation (lane-1a seam; this module never re-implements the audit).

Scale discipline (audit F1: one figure never mixes the ``*_frac`` and
``*_rps`` scales): the ladder renders on the RATE scale only — raw
throughput is inherently a per-window rate, so its G/Y/null/gap companions
are the ``*_rps`` fields. The dimensionless ``*_frac`` ladder is a separate
future table, never extra columns here.

Basis discipline: only ``BASIS_AGGREGATE`` (§6.6a) is renderable. The S1
ladder companions — raw throughput and the independence null — exist ONLY as
aggregate fields in ``WindowMetrics`` (lane 1a deliberately ships no
``throughput_per_gpu``/``independence_null_per_gpu``); a per-GPU ladder
request refuses rather than fabricating derived columns.

Missing-field honesty (fail-closed carve-out): a metrics record missing BOTH
rendered independence-null fields is pre-ladder pilot data — those two cells
render as the EXPLICIT string ``n/a (pre-ladder data)`` (machine table:
``null`` value + note), never a fabricated number, and the row's Y still
appears WITH its S1(a) companions. A record carrying only ONE of the two
fields is malformed (tampered/truncated, not merely old) and refuses. The
carve-out covers ONLY these columns: basis facts (``bases``, ``gpu_count``,
the rate pairs) are never defaulted — ``assert_single_basis`` refuses
unlabeled records, per lane 1a ("legacy records must be rebuilt, not
defaulted").
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.analysis.goodput import (  # noqa: E402
    BASIS_AGGREGATE,
    assert_single_basis,
)

__all__ = [
    "LADDER_JSON_NAME",
    "LADDER_MD_NAME",
    "LADDER_SCHEMA_VERSION",
    "MACHINE_COLUMNS",
    "PRE_LADDER_NA",
    "LadderError",
    "build_ladder_rows",
    "render_ladder_markdown",
    "write_yield_ladder",
]

LADDER_JSON_NAME = "yield_ladder.json"
LADDER_MD_NAME = "yield_ladder.md"
LADDER_SCHEMA_VERSION = 1

#: THE explicit absent-data cell (S1 honesty): rendered verbatim in the
#: human table and carried as the machine row's ``note`` beside null values.
PRE_LADDER_NA = "n/a (pre-ladder data)"

#: S1(a) ladder mandate fields — every row must carry ALL of them, so Y can
#: never be printed without raw throughput and G beside it. goodput_rps and
#: yield_rps are ALSO audited by assert_single_basis (they sit on the
#: aggregate basis tuple); throughput_rps is not a basis field, so this
#: module is its only auditor.
_LADDER_MANDATE_FIELDS: tuple[str, ...] = (
    "throughput_rps",
    "goodput_rps",
    "yield_rps",
)

#: S1(b) independence-null group — the two rendered companions. Present
#: together (validated + rendered) or absent together (pre-ladder n/a);
#: partial presence refuses (see the module docstring carve-out).
_NULL_GROUP_FIELDS: tuple[str, ...] = (
    "independence_null_rps",
    "covariance_gap_rps",
)

#: Machine-table column order (also the human-table order). ``note`` is None
#: on fully-laddered rows and PRE_LADDER_NA on pre-ladder rows.
MACHINE_COLUMNS: tuple[str, ...] = (
    "window",
    "throughput_rps",
    "goodput_rps",
    "yield_rps",
    "independence_null_rps",
    "covariance_gap_rps",
    "gpu_count",
    "basis",
    "note",
)

_MD_HEADER_CELLS: tuple[str, ...] = (
    "window",
    "throughput (req/s)",
    "G (req/s)",
    "Y (req/s)",
    "G·E[v] null (req/s)",
    "cov gap Y − G·E[v] (req/s)",
    "GPUs",
    "basis",
)


class LadderError(ValueError):
    """S1 ladder-contract violation (row shape, companions, stamp, basis)."""


def _atomic_write_text(path: Path, text: str) -> None:
    # G14 discipline (as in run_campaign_analysis): a crash mid-write must
    # never leave a torn artifact that a later look could half-read.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


_MISSING = object()


def _get(metrics: object, name: str) -> object:
    """Field access for WindowMetrics instances AND their dict/JSON form."""
    if isinstance(metrics, Mapping):
        return metrics.get(name, _MISSING)
    return getattr(metrics, name, _MISSING)


def _require_number(metrics: object, name: str, window: str, clause: str) -> float:
    value = _get(metrics, name)
    if value is _MISSING:
        raise LadderError(
            f"ladder row {window!r} lacks {name!r} — S1 clause ({clause}) "
            "mandates the full ladder in every table (Y never appears "
            "without raw throughput and G beside it; the independence null "
            "beside every Y); rebuild the metrics via goodput.evaluate_window"
        )
    # bool is refused before the number check (True would render as 1.0) —
    # the same policy lane 1a pins in _basis_item_number.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LadderError(
            f"ladder row {window!r}: {name}={value!r} must be a number — a "
            "non-numeric ladder cell is never rendered (fail-closed)"
        )
    value = float(value)
    if not math.isfinite(value):
        raise LadderError(
            f"ladder row {window!r}: {name}={value!r} must be finite — NaN/inf "
            "is corrupt data, not pre-ladder data; the n/a carve-out covers "
            "ABSENT fields only, never fabricates over a broken value"
        )
    return value


def _null_group(metrics: object, window: str) -> tuple[float, float] | None:
    """The S1(b) pair, or None when the row is pre-ladder (BOTH absent)."""
    present = [n for n in _NULL_GROUP_FIELDS if _get(metrics, n) is not _MISSING]
    if not present:
        return None
    if len(present) != len(_NULL_GROUP_FIELDS):
        absent = [n for n in _NULL_GROUP_FIELDS if n not in present]
        raise LadderError(
            f"ladder row {window!r} carries {present} but lacks {absent} — a "
            "partial independence-null group is a malformed record "
            "(tampered/truncated), not pre-ladder data; the n/a carve-out "
            "applies only when the WHOLE group is absent"
        )
    null_rps = _require_number(metrics, _NULL_GROUP_FIELDS[0], window, clause="b")
    gap_rps = _require_number(metrics, _NULL_GROUP_FIELDS[1], window, clause="b")
    return null_rps, gap_rps


def _normalize_items(
    items: Mapping[str, object] | Sequence[tuple[str, object]],
) -> list[tuple[str, object]]:
    pairs = list(items.items()) if isinstance(items, Mapping) else list(items)
    seen: set[str] = set()
    for window, _metrics in pairs:
        if not isinstance(window, str) or not window.strip():
            raise LadderError(
                f"ladder row id {window!r} must be a non-empty string — an "
                "unidentifiable window cannot be audited"
            )
        if window in seen:
            raise LadderError(
                f"duplicate ladder row id {window!r} — two rows claiming one "
                "window would silently shadow each other downstream; refusing"
            )
        seen.add(window)
    return pairs


def build_ladder_rows(
    items: Mapping[str, object] | Sequence[tuple[str, object]],
    *,
    basis: str = BASIS_AGGREGATE,
) -> list[dict[str, Any]]:
    """Validate and build the machine table (one dict per window, in input
    order, keys = ``MACHINE_COLUMNS``).

    ``items``: (window id → metrics) mapping or (id, metrics) pairs; metrics
    are ``WindowMetrics`` instances or their ``to_flat_dict``/JSON dict form.

    Refuses (fail-closed): a non-aggregate ``basis`` (the ladder companions
    exist only on §6.6a — see the module docstring); any row missing an S1(a)
    mandate field or holding a non-finite/non-number there (``LadderError``);
    an empty, unlabeled, or mixed-basis pool (``GoodputError`` from
    ``assert_single_basis``, carrying the §6.6 citation); a partial or broken
    S1(b) group. A row whose WHOLE S1(b) group is absent builds with
    ``None`` null cells and ``note = PRE_LADDER_NA``.
    """
    if basis != BASIS_AGGREGATE:
        raise LadderError(
            f"basis={basis!r}: the S1 ladder renders only on the "
            f"{BASIS_AGGREGATE!r} basis (§6.6a) — raw throughput and the "
            "independence null exist ONLY as aggregate WindowMetrics fields "
            "(lane 1a ships no per-GPU counterparts); deriving them by "
            "division would fabricate columns the registration never defined"
        )
    pairs = _normalize_items(items)

    # S1(a) — the ladder mandate, audited per row BEFORE any rendering.
    for window, metrics in pairs:
        for name in _LADDER_MANDATE_FIELDS:
            _require_number(metrics, name, window, clause="a")

    # §6.6 — the never-mixed-bases guard at table generation: the lane-1a
    # seam audits basis labeling, gpu_count, the aggregate-basis fields and
    # the rate-pair invariant for EVERY pooled row (it also refuses an empty
    # pool). Its GoodputError propagates unchanged — the §6.6 citation is
    # the refusal the charter mandates.
    assert_single_basis([metrics for _window, metrics in pairs], basis)

    rows: list[dict[str, Any]] = []
    for window, metrics in pairs:
        null_pair = _null_group(metrics, window)
        rows.append(
            {
                "window": window,
                "throughput_rps": _require_number(
                    metrics, "throughput_rps", window, clause="a"
                ),
                "goodput_rps": _require_number(
                    metrics, "goodput_rps", window, clause="a"
                ),
                "yield_rps": _require_number(
                    metrics, "yield_rps", window, clause="a"
                ),
                "independence_null_rps": (
                    None if null_pair is None else null_pair[0]
                ),
                "covariance_gap_rps": None if null_pair is None else null_pair[1],
                # audited by assert_single_basis above; plain int for JSON.
                "gpu_count": int(_get(metrics, "gpu_count")),  # type: ignore[call-overload]
                "basis": basis,
                "note": PRE_LADDER_NA if null_pair is None else None,
            }
        )
    return rows


def _check_stamp(stamp: object) -> str:
    # Every analysis output carries the §9.11 mode stamp; an unstamped ladder
    # could be quoted as confirmatory (the exact ambiguity the stamp exists
    # to prevent), so absence refuses instead of defaulting.
    if not isinstance(stamp, str) or not stamp.strip():
        raise LadderError(
            f"stamp={stamp!r}: the ladder artifact must carry the analysis "
            "mode stamp (DESIGN-INPUT-ONLY / CONFIRMATORY) — refusing to "
            "write an unstamped table"
        )
    return stamp


def _fmt(value: float | None) -> str:
    return PRE_LADDER_NA if value is None else format(value, ".6g")


def _markdown_from_rows(rows: Sequence[Mapping[str, Any]], stamp: str) -> str:
    lines = [
        f"# S1 yield ladder — {stamp}",
        "",
        "- S1(a) ladder mandate: Y never appears without raw throughput and "
        "G beside it (rows lacking companions REFUSE at build time).",
        "- S1(b) independence null: G·E[v] beside every Y; the covariance "
        "gap Y − G·E[v] IS the finding.",
        "- §6.6: every row audited onto the single "
        f"'{rows[0]['basis']}' basis by goodput.assert_single_basis before "
        "rendering — mixed-bases pools never reach this table.",
        f"- '{PRE_LADDER_NA}' marks pre-ladder records whose independence-"
        "null fields were never computed; nothing is back-filled.",
        "",
        "| " + " | ".join(_MD_HEADER_CELLS) + " |",
        "| " + " | ".join("---" for _ in _MD_HEADER_CELLS) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["window"]),
                    _fmt(row["throughput_rps"]),
                    _fmt(row["goodput_rps"]),
                    _fmt(row["yield_rps"]),
                    _fmt(row["independence_null_rps"]),
                    _fmt(row["covariance_gap_rps"]),
                    str(row["gpu_count"]),
                    str(row["basis"]),
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def render_ladder_markdown(
    items: Mapping[str, object] | Sequence[tuple[str, object]],
    *,
    stamp: str,
    basis: str = BASIS_AGGREGATE,
) -> str:
    """Human table. Takes raw ITEMS (never pre-built rows) so every markdown
    ladder passes the same S1(a)/(b) + §6.6 audits as the machine table —
    the structural enforcement of the ladder mandate."""
    stamp = _check_stamp(stamp)
    return _markdown_from_rows(build_ladder_rows(items, basis=basis), stamp)


def write_yield_ladder(
    out_dir: Path,
    items: Mapping[str, object] | Sequence[tuple[str, object]],
    *,
    stamp: str,
    basis: str = BASIS_AGGREGATE,
) -> dict[str, Any]:
    """Build once, write ``yield_ladder.json`` + ``yield_ladder.md`` into
    ``out_dir``, return {json, markdown, n_rows, n_pre_ladder}.

    Validation (and therefore every refusal above) happens BEFORE either
    file is opened — a refused ladder leaves no partial artifact behind.
    """
    stamp = _check_stamp(stamp)
    rows = build_ladder_rows(items, basis=basis)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    document = {
        "schema_version": LADDER_SCHEMA_VERSION,
        "mode_stamp": stamp,
        "basis": basis,
        "scale": "rate (*_rps) — audit F1: one table never mixes scales",
        "columns": list(MACHINE_COLUMNS),
        "charter": (
            "S1(a) ladder mandate + S1(b) independence null; §6.6 "
            "single-basis pool audited by goodput.assert_single_basis"
        ),
        "pre_ladder_marker": PRE_LADDER_NA,
        "rows": rows,
    }
    json_path = out_dir / LADDER_JSON_NAME
    _atomic_write_text(json_path, json.dumps(document, indent=2) + "\n")
    md_path = out_dir / LADDER_MD_NAME
    _atomic_write_text(md_path, _markdown_from_rows(rows, stamp))
    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "n_rows": len(rows),
        "n_pre_ladder": sum(1 for r in rows if r["note"] == PRE_LADDER_NA),
    }
