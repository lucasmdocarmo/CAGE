"""Campaign v2 results-tree PRODUCER — the write side of cloud/RESULTS_LAYOUT.md.

Topic-8 finding H1/H5 (MyDocs/registration/CODE_ASSERTION_2026-08.md): the read
side (scripts/4_analysis/organize_results.py) validates a §1 tree that nothing
produced. This module is that producer, built to emit EXACTLY the tree the
organizer's ``walk_run_tree`` + ``_read_cell_meta`` accept:

    results/<campaign>/<session>/<run_id>/
        manifest.json                     # §3 — written at run START, amended never
        ledger.json                       # §5 seal — written by seal_run at run END
        cells/<CellSpec.to_row_key()>/
            cell.json                     # cellspec + baseline + windows[] table
            window_<dataset>-<NN>/        # k = <dataset>-<ordinal>, zero-padded %02d
                requests.jsonl
                qa_evidence.jsonl         # sharegpt (load donor) exempt
                engine_metrics.json
                cage_stats.jsonl
                regime.json               # §6.1 regime bridge output (aux artifact)

Reader-contract bindings (each constant below cites the organize_results
symbol it must stay byte-identical to; tests/test_campaign_layout.py pins the
pairs together the way test_terraform_contract pins the run-id grammar):

- window dir names must match ``organize_results.WINDOW_DIR_RE``
  (``^window_([a-z0-9_]+)-(\\d+)$``); the reader takes ``int(group(2))`` as the
  ordinal and the VERBATIM digits into ``window_key`` — so ``window_x-1`` and
  ``window_x-01`` would collide on ordinal but diverge on window_key (Topic-8
  H12). This writer pins ONE canonical spelling, ``%02d``, and enforces a
  (row_key, dataset, ordinal-as-int) uniqueness invariant so the alias pair can
  never be emitted.
- cell directory names are ``CellSpec.to_row_key()`` VERBATIM (§2 — never
  hand-built); ``_read_cell_meta`` round-trips cell.json's ``cellspec`` mapping
  through ``CellSpec.from_flat_dict(...).to_row_key()`` against the dirname, so
  cell.json always embeds ``spec.to_flat_dict()``.
- manifest fields mirror ``organize_results.load_manifest`` (run_id grammar +
  dirname match, session vocabulary, singular model) PLUS the full §3 required
  set this writer fail-closes on.

Doctrine: fail loud (every problem listed, none skipped), atomic writes
(tmp + os.replace), absence-is-not-zero (None coordinates and refused
telemetry stay explicit, never coerced to numerics).
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, get_args

import pandas as pd

from src.analysis.cellspec import (
    BASELINES,
    CellSpec,
    Engine,
    Model,
)
from src.analysis.goodput import classify_regime
from src.analysis.regime_inputs import (
    REGIME_UNKNOWN,
    RegimeInputError,
    compute_window_regime_inputs,
)
from src.analysis.stats.ledger import hash_artifacts, verify_ledger, write_ledger
from src.observability.provenance import git_dirty, git_sha

__all__ = [
    "CampaignLayoutError",
    "CampaignRun",
    "CellWriter",
    "DATASET_IDS",
    "MANIFEST_REQUIRED_FIELDS",
    "QA_EVIDENCE_EXEMPT_DATASETS",
    "RUN_ID_RE",
    "SESSIONS",
    "WINDOW_DIR_RE",
    "WindowHandle",
    "load_telemetry_series",
    "seal_run",
    "write_manifest",
    "write_window_regime",
]

#: MUST equal organize_results.WINDOW_DIR_RE — the reader parses window dirs
#: with this exact pattern (§1: k = <dataset>-<ordinal>).
WINDOW_DIR_RE = re.compile(r"^window_([a-z0-9_]+)-(\d+)$")

#: MUST equal organize_results.RUN_ID_RE — §1 run_id grammar, bucket-name-safe
#: (it names gs://cage-<run_id> verbatim).
RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,40}$")

#: §1 <campaign> grammar: lowercase slug, same character class as run_id
#: (a path level of the GCS mirror, §4).
CAMPAIGN_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,40}$")

#: MUST equal organize_results.SESSIONS (§1 session vocabulary, RUNBOOK §1).
SESSIONS: frozenset[str] = frozenset({"a", "b", "cd-act1", "cd-act2"})

#: MUST equal organize_results.DATASET_IDS (§1 — the only legal window-name
#: datasets).
DATASET_IDS: frozenset[str] = frozenset(
    {"squad_v2", "hotpotqa", "musique", "qasper", "ruler", "scbench", "sharegpt"}
)

#: MUST equal organize_results._QA_EVIDENCE_EXEMPT_DATASETS — §1: ShareGPT is
#: the load donor; its windows carry serving streams only.
QA_EVIDENCE_EXEMPT_DATASETS: frozenset[str] = frozenset({"sharegpt"})

#: §3 required manifest fields. The task-list of Topic-8 #126 omitted `engine`,
#: but RESULTS_LAYOUT §3 (THE authority) requires `engine, engine_version` —
#: both are enforced here.
MANIFEST_REQUIRED_FIELDS: tuple[str, ...] = (
    "campaign",
    "session",
    "run_id",
    "model",
    "git_sha",
    "git_dirty",
    "engine",
    "engine_version",
    "seed",
    "provider",
    "hardware",
    "dataset_manifests_sha256",
    "cellspec_schema_version",
    "created_utc",
)

_MODELS: frozenset[str] = frozenset(get_args(Model))
_ENGINES: frozenset[str] = frozenset(get_args(Engine))
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CELL_META_NAME = "cell.json"
_TMP_SUFFIX = ".tmp"

#: Reverse of the §7.1 numbered layer, as organize_results.BASELINE_OF_CELL
#: derives it: (arm, retriever) -> baseline id; cells outside the numbered
#: layer get "".
_BASELINE_OF_CELL: dict[tuple[str, str], str] = {
    (spec.arm, spec.retriever): bid for bid, spec in BASELINES.items()
}

#: Canonical telemetry field names = the compute_window_regime_inputs defaults
#: (src/analysis/regime_inputs.py). Legacy names are what
#: VllmTelemetrySampler.save_series historically wrote (Topic-8 H1 schema
#: mismatch); the loader accepts both, canonical preferred.
_TS_FIELDS: tuple[str, ...] = ("ts_s", "ts")
_KV_FIELDS: tuple[str, ...] = ("kv_cache_usage", "kv_usage")
_PREEMPT_FIELD = "preemptions_total"

_REGIME_SCHEMA_VERSION = 1


class CampaignLayoutError(RuntimeError):
    """A write would violate the RESULTS_LAYOUT contract; carries EVERY problem."""

    def __init__(self, problems: list[str] | str) -> None:
        if isinstance(problems, str):
            problems = [problems]
        self.problems = list(problems)
        lines = "\n".join(f"  [{i + 1}] {p}" for i, p in enumerate(self.problems))
        super().__init__(
            f"refusing write — {len(self.problems)} problem(s) vs "
            f"cloud/RESULTS_LAYOUT.md:\n{lines}"
        )


# ---------------------------------------------------------------------------
# Atomic write primitives (tmp + os.replace; a crash never truncates a
# published artifact)
# ---------------------------------------------------------------------------


def _atomic_write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + _TMP_SUFFIX)
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> Path:
    try:
        text = json.dumps(document, indent=2, sort_keys=True) + "\n"
    except TypeError as exc:
        # No default=str: silent type drift is Topic-8 H3's defect, not a fix.
        raise CampaignLayoutError(f"{path.name}: not JSON-serializable: {exc}") from exc
    return _atomic_write_text(path, text)


def _atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + _TMP_SUFFIX)
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            for i, row in enumerate(rows):
                if not isinstance(row, Mapping):
                    raise CampaignLayoutError(
                        f"{path.name}: row {i} is {type(row).__name__}, not a mapping"
                    )
                try:
                    fh.write(json.dumps(dict(row)) + "\n")
                except TypeError as exc:
                    raise CampaignLayoutError(
                        f"{path.name}: row {i} is not JSON-serializable: {exc}"
                    ) from exc
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------
# §3 manifest writer
# ---------------------------------------------------------------------------


def _default_git_provenance(repo_dir: Path) -> tuple[str, bool]:
    """git_sha/git_dirty at write time (§3: from git, else BUILD_INFO) — fail loud."""
    sha = git_sha(str(repo_dir))
    dirty = git_dirty(str(repo_dir))
    problems: list[str] = []
    if not sha:
        problems.append(
            f"git_sha unresolvable for {repo_dir} (no git repo and no BUILD_INFO) — "
            "§3: a run without code provenance is not a run"
        )
    if dirty is None:
        problems.append(f"git_dirty unresolvable for {repo_dir} (§3 required field)")
    if problems:
        raise CampaignLayoutError(problems)
    assert sha is not None and dirty is not None
    return sha, dirty


def _check_int(name: str, value: object, *, minimum: int, problems: list[str]) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        problems.append(f"{name}={value!r} must be an int")
    elif value < minimum:
        problems.append(f"{name}={value!r} must be >= {minimum}")


def write_manifest(
    run_root: Path,
    *,
    campaign: str,
    session: str,
    run_id: str,
    model: str,
    engine: str,
    engine_version: str,
    seed: int,
    provider: str,
    hardware: str,
    dataset_manifests_sha256: str,
    cellspec_schema_version: int,
    repo_dir: Path | None = None,
    git_provenance: Callable[[Path], tuple[str, bool]] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write §3 manifest.json at run START — every required field enforced.

    ``git_sha``/``git_dirty`` are computed from the repo at write time via
    ``git_provenance`` (injectable for tests; default = subprocess git with
    the tarball BUILD_INFO fallback, per §3). ``created_utc`` is stamped here.
    §3: amended never — an existing manifest.json refuses (a re-run is a new
    run_id). ``extra`` may add optional keys (e.g. the organizer's optional
    ``datasets`` narrowing) but can never shadow a required field.
    """
    run_root = Path(run_root)
    manifest_path = run_root / "manifest.json"
    problems: list[str] = []
    if manifest_path.exists():
        raise CampaignLayoutError(
            f"manifest.json already exists at {manifest_path} — §3: amended never; "
            "a re-run is a new run_id"
        )

    for name, value in (
        ("campaign", campaign),
        ("session", session),
        ("run_id", run_id),
        ("model", model),
        ("engine", engine),
        ("engine_version", engine_version),
        ("provider", provider),
        ("hardware", hardware),
        ("dataset_manifests_sha256", dataset_manifests_sha256),
    ):
        if not isinstance(value, str) or not value:
            problems.append(f"{name}={value!r} must be a non-empty string (§3)")
    if isinstance(campaign, str) and campaign and not CAMPAIGN_RE.match(campaign):
        problems.append(f"campaign {campaign!r} violates the §1 slug grammar")
    if isinstance(session, str) and session not in SESSIONS:
        problems.append(f"session {session!r} is not a §1 session ({sorted(SESSIONS)})")
    if isinstance(run_id, str) and run_id and not RUN_ID_RE.match(run_id):
        problems.append(
            f"run_id {run_id!r} violates the §1 grammar {RUN_ID_RE.pattern} "
            "(it names gs://cage-<run_id> verbatim)"
        )
    if isinstance(run_id, str) and run_id and run_id != run_root.name:
        problems.append(
            f"run_id {run_id!r} != run directory name {run_root.name!r} — "
            "organize_results.load_manifest refuses this tree"
        )
    if isinstance(model, str) and model not in _MODELS:
        problems.append(f"model {model!r} is not on the D4 roster ({sorted(_MODELS)})")
    if isinstance(engine, str) and engine not in _ENGINES:
        problems.append(f"engine {engine!r} is not a §7.3 engine ({sorted(_ENGINES)})")
    if isinstance(dataset_manifests_sha256, str) and dataset_manifests_sha256 and not _SHA256_RE.match(
        dataset_manifests_sha256
    ):
        problems.append(
            f"dataset_manifests_sha256 {dataset_manifests_sha256!r} is not a "
            "lowercase hex sha256 (§3: pins the exact query/corpus builds)"
        )
    _check_int("seed", seed, minimum=0, problems=problems)
    _check_int(
        "cellspec_schema_version", cellspec_schema_version, minimum=1, problems=problems
    )
    if extra is not None:
        shadowed = sorted(set(extra) & set(MANIFEST_REQUIRED_FIELDS))
        if shadowed:
            problems.append(f"extra keys {shadowed} would shadow §3 required fields")
    if problems:
        raise CampaignLayoutError(problems)

    provenance = git_provenance if git_provenance is not None else _default_git_provenance
    resolved_repo = (
        Path(repo_dir) if repo_dir is not None else Path(__file__).resolve().parents[2]
    )
    sha, dirty = provenance(resolved_repo)
    if not isinstance(sha, str) or not sha:
        raise CampaignLayoutError(f"git provenance returned git_sha={sha!r} (§3)")
    if not isinstance(dirty, bool):
        raise CampaignLayoutError(f"git provenance returned git_dirty={dirty!r} (§3)")

    manifest: dict[str, Any] = dict(extra or {})
    manifest.update(
        {
            "campaign": campaign,
            "session": session,
            "run_id": run_id,
            "model": model,
            "git_sha": sha,
            "git_dirty": dirty,
            "engine": engine,
            "engine_version": engine_version,
            "seed": seed,
            "provider": provider,
            "hardware": hardware,
            "dataset_manifests_sha256": dataset_manifests_sha256,
            "cellspec_schema_version": cellspec_schema_version,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    missing = [k for k in MANIFEST_REQUIRED_FIELDS if manifest.get(k) in (None, "")]
    if missing:
        raise CampaignLayoutError(f"manifest missing §3 required field(s) {missing}")
    return _atomic_write_json(manifest_path, manifest)


# ---------------------------------------------------------------------------
# Cell + window writers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowHandle:
    """One emitted window: its §8 join key and on-disk location."""

    row_key: str
    dataset: str
    ordinal: int
    window_key: str  # k = <dataset>-<NN> (the reader's verbatim join key)
    window_dir: Path


class CellWriter:
    """Producer of ONE ``cells/<row_key>/`` directory (§1/§2).

    Directory name = ``CellSpec.to_row_key()`` verbatim (§2 — never
    hand-built). Maintains cell.json's windows[] table and the
    (row_key, dataset, ordinal) uniqueness invariant across BOTH the
    in-memory registry and whatever already exists on disk (ordinals are
    compared as ints, so the ``window_x-1`` / ``window_x-01`` alias pair of
    Topic-8 H12 is refused, not silently coexisting).
    """

    def __init__(self, run_root: Path, spec: CellSpec) -> None:
        self.run_root = Path(run_root)
        self.spec = spec
        self.row_key = spec.to_row_key()
        self.cell_dir = self.run_root / "cells" / self.row_key
        self._windows: dict[str, dict[str, Any]] = {}
        self._registered: set[tuple[str, int]] = set()
        self._load_existing()

    def _load_existing(self) -> None:
        """Resume support: rebuild the registry from disk, refusing aliases."""
        if not self.cell_dir.is_dir():
            return
        problems: list[str] = []
        for entry in sorted(self.cell_dir.iterdir()):
            if entry.name.startswith(".") or entry.name == _CELL_META_NAME:
                continue
            match = WINDOW_DIR_RE.match(entry.name)
            if not entry.is_dir() or match is None:
                problems.append(
                    f"cells/{self.row_key}/{entry.name}: not a "
                    "window_<dataset>-<ordinal> directory — refusing to extend a "
                    "malformed cell"
                )
                continue
            key = (match.group(1), int(match.group(2)))
            if key in self._registered:
                problems.append(
                    f"cells/{self.row_key}: windows {key[0]}-{key[1]} exist under "
                    "two spellings (ordinal alias, Topic-8 H12) — refuse"
                )
            self._registered.add(key)
        meta_path = self.cell_dir / _CELL_META_NAME
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                problems.append(f"{meta_path}: invalid JSON: {exc}")
                meta = None
            if isinstance(meta, dict) and isinstance(meta.get("windows"), dict):
                self._windows = dict(meta["windows"])
        if problems:
            raise CampaignLayoutError(problems)

    def _next_ordinal(self, dataset: str) -> int:
        used = [o for (d, o) in self._registered if d == dataset]
        return max(used, default=0) + 1

    def add_window(
        self,
        dataset: str,
        *,
        seed: int,
        rep: int,
        t_start: float,
        t_end: float,
        requests: Iterable[Mapping[str, Any]],
        cage_stats: Iterable[Mapping[str, Any]],
        engine_metrics: Mapping[str, Any],
        qa_evidence: Iterable[Mapping[str, Any]] | None = None,
        ordinal: int | None = None,
    ) -> WindowHandle:
        """Emit one measurement window + its windows[] entry (§1).

        ``t_start``/``t_end`` are seconds on the SAME clock as the telemetry
        ``ts_s`` samples (the regime bridge slices [t_start, t_end) against
        them). ``qa_evidence=None`` is legal ONLY for the sharegpt load donor
        (reader exemption ``_QA_EVIDENCE_EXEMPT_DATASETS``). ``ordinal``
        defaults to the next unused one for this dataset; an explicit ordinal
        that collides — under ANY zero-padding spelling — refuses.
        """
        problems: list[str] = []
        if dataset not in DATASET_IDS:
            problems.append(
                f"dataset {dataset!r} is not a §1 dataset id ({sorted(DATASET_IDS)})"
            )
        _check_int("seed", seed, minimum=0, problems=problems)
        _check_int("rep", rep, minimum=0, problems=problems)
        for name, value in (("t_start", t_start), ("t_end", t_end)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                problems.append(f"{name}={value!r} must be a number")
            elif not math.isfinite(value):
                problems.append(f"{name}={value!r} must be finite")
        if not problems and not float(t_end) > float(t_start):
            problems.append(f"t_end={t_end!r} must be > t_start={t_start!r}")
        if qa_evidence is None and dataset not in QA_EVIDENCE_EXEMPT_DATASETS:
            problems.append(
                f"dataset {dataset!r} requires qa_evidence.jsonl (§1 per-window "
                "contract; only sharegpt is the serving-streams-only load donor)"
            )
        if ordinal is not None:
            _check_int("ordinal", ordinal, minimum=1, problems=problems)
        if not isinstance(engine_metrics, Mapping):
            problems.append(
                f"engine_metrics must be a mapping, got {type(engine_metrics).__name__}"
            )
        if problems:
            raise CampaignLayoutError(problems)

        k_ordinal = int(ordinal) if ordinal is not None else self._next_ordinal(dataset)
        if (dataset, k_ordinal) in self._registered:
            raise CampaignLayoutError(
                f"cells/{self.row_key}: window ({dataset}, ordinal {k_ordinal}) was "
                "already emitted — the (row_key, dataset, ordinal) tuple is unique "
                "by invariant (Topic-8 H12 alias collision included)"
            )
        window_key = f"{dataset}-{k_ordinal:02d}"
        window_dir = self.cell_dir / f"window_{window_key}"
        if window_dir.exists():
            raise CampaignLayoutError(
                f"{window_dir} already exists on disk but was not registered — "
                "refusing to overwrite"
            )
        assert WINDOW_DIR_RE.match(window_dir.name), "writer/reader grammar drift"

        window_dir.mkdir(parents=True)
        _atomic_write_jsonl(window_dir / "requests.jsonl", requests)
        _atomic_write_jsonl(window_dir / "cage_stats.jsonl", cage_stats)
        _atomic_write_json(window_dir / "engine_metrics.json", engine_metrics)
        if qa_evidence is not None:
            _atomic_write_jsonl(window_dir / "qa_evidence.jsonl", qa_evidence)

        # §1 windows[] table entry: k -> {dataset, seed, rep, budget_r,
        # rate_frac, t_start, t_end}. Pressure coords come from the CELL tuple
        # (a window cannot contradict its cell); F1 cells carry null — absence
        # stays absence, never 0.
        self._windows[window_key] = {
            "dataset": dataset,
            "seed": seed,
            "rep": rep,
            "budget_r": self.spec.budget_r,
            "rate_frac": self.spec.rate_frac,
            "t_start": float(t_start),
            "t_end": float(t_end),
        }
        self._registered.add((dataset, k_ordinal))
        self._write_cell_meta()
        return WindowHandle(
            row_key=self.row_key,
            dataset=dataset,
            ordinal=k_ordinal,
            window_key=window_key,
            window_dir=window_dir,
        )

    def _write_cell_meta(self) -> Path:
        """cell.json: the reader's ``_read_cell_meta`` round-trips ``cellspec``
        through CellSpec.from_flat_dict(...).to_row_key() == dirname."""
        return _atomic_write_json(
            self.cell_dir / _CELL_META_NAME,
            {
                "cellspec": self.spec.to_flat_dict(),
                "baseline": _BASELINE_OF_CELL.get(
                    (self.spec.arm, self.spec.retriever), ""
                ),
                "windows": self._windows,
            },
        )


class CampaignRun:
    """Producer of ONE run tree: manifest at start, cells during, seal at end."""

    def __init__(self, run_root: Path) -> None:
        self.run_root = Path(run_root)
        manifest_path = self.run_root / "manifest.json"
        if not manifest_path.is_file():
            raise CampaignLayoutError(
                f"no manifest.json at {manifest_path} — use CampaignRun.create() "
                "(§3: the manifest is written at run START)"
            )
        self.manifest: dict[str, Any] = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        missing = [
            k for k in MANIFEST_REQUIRED_FIELDS if self.manifest.get(k) in (None, "")
        ]
        if missing:
            raise CampaignLayoutError(
                f"manifest.json missing §3 required field(s) {missing} — refusing "
                "to extend a non-compliant run"
            )
        self._cells: dict[str, CellWriter] = {}

    @classmethod
    def create(cls, run_root: Path, **manifest_fields: Any) -> CampaignRun:
        """Mint a new run root: mkdir + §3 manifest; keyword args = write_manifest's."""
        run_root = Path(run_root)
        run_root.mkdir(parents=True, exist_ok=True)
        write_manifest(run_root, **manifest_fields)
        return cls(run_root)

    def _ensure_unsealed(self) -> None:
        if (self.run_root / "ledger.json").exists():
            raise CampaignLayoutError(
                "ledger.json exists — the run is sealed (§5: append-only until "
                "sealed, then immutable; a re-run is a new run_id)"
            )

    def cell(self, spec: CellSpec) -> CellWriter:
        """The (memoized) writer for one cell tuple; §3: one run = one model."""
        self._ensure_unsealed()
        if spec.model != self.manifest["model"]:
            raise CampaignLayoutError(
                f"cell model {spec.model!r} != manifest model "
                f"{self.manifest['model']!r} (§3: one run = one model — "
                "organize_results refuses such trees)"
            )
        key = spec.to_row_key()
        if key not in self._cells:
            self._cells[key] = CellWriter(self.run_root, spec)
        return self._cells[key]

    def seal(self) -> Path:
        """Run-end §5 seal over the raw tree; see module-level seal_run."""
        return seal_run(self.run_root)


# ---------------------------------------------------------------------------
# §5 run-end seal
# ---------------------------------------------------------------------------


def seal_run(run_root: Path) -> Path:
    """Seal the run: sha256 every artifact under cells/ plus manifest.json.

    §5: keys relative to the run root, sealed with
    ``src.analysis.stats.ledger.write_ledger`` -> ledger.json at the run root
    (which itself refuses to overwrite an existing seal — a re-run is a new
    run_id). The fresh seal is verified in place ((a) of §5: on the node right
    after sealing) — a mismatch is impossible unless the tree changed mid-seal,
    and that must fail loud, not surface at analysis load.
    """
    run_root = Path(run_root)
    manifest_path = run_root / "manifest.json"
    cells_dir = run_root / "cells"
    problems: list[str] = []
    if not manifest_path.is_file():
        problems.append(f"manifest.json missing at {manifest_path} (§3)")
    if not cells_dir.is_dir():
        problems.append(f"cells/ missing at {cells_dir} — an empty run seals nothing")
    if problems:
        raise CampaignLayoutError(problems)

    artifacts: list[Path] = [manifest_path]
    leftovers: list[str] = []
    for path in sorted(cells_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(run_root)
        if any(part.startswith(".") for part in rel.parts):
            continue  # the reader skips dot-entries; so does the seal
        if path.name.endswith(_TMP_SUFFIX):
            leftovers.append(rel.as_posix())
            continue
        artifacts.append(path)
    if leftovers:
        raise CampaignLayoutError(
            [f"crash residue under cells/ — resolve before sealing: {p}" for p in leftovers]
        )
    if len(artifacts) == 1:
        raise CampaignLayoutError(
            "cells/ contains no artifacts — an empty run seals nothing"
        )
    ledger_path = write_ledger(
        hash_artifacts(artifacts, base_dir=run_root), run_root / "ledger.json"
    )
    mismatches = verify_ledger(ledger_path, run_root)
    if mismatches:
        raise CampaignLayoutError(
            [f"seal verification failed immediately after sealing: {m}" for m in mismatches]
        )
    return ledger_path


# ---------------------------------------------------------------------------
# §6.1 regime bridge — the first production caller of
# src.analysis.regime_inputs.compute_window_regime_inputs (Topic-8 H1: it had
# no caller)
# ---------------------------------------------------------------------------


def load_telemetry_series(path: Path) -> pd.DataFrame:
    """Load a telemetry series JSONL into the regime-inputs schema.

    Accepts BOTH field spellings per record — canonical (``ts_s``,
    ``kv_cache_usage``; what save_series now emits) preferred, legacy
    (``ts``, ``kv_usage``) fallback; ``preemptions_total`` is shared. A record
    carrying NEITHER timestamp spelling is corrupt (a sample that cannot be
    placed in time) and fails loud. Absent gauges/counters stay None (-> NaN)
    so ``compute_window_regime_inputs`` refuses them — absence is not zero.
    Returns a frame with exactly the canonical columns; an empty or
    all-blank-lines file yields an empty frame (0 samples -> refusal lane).
    """
    path = Path(path)
    if not path.is_file():
        raise CampaignLayoutError(f"telemetry series not found: {path}")
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for lineno, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"{path.name}:{lineno}: invalid JSON: {exc}")
            continue
        if not isinstance(rec, dict):
            problems.append(f"{path.name}:{lineno}: record is not an object")
            continue
        ts = next((rec[f] for f in _TS_FIELDS if f in rec), None)
        if ts is None:
            problems.append(
                f"{path.name}:{lineno}: no timestamp under any of {_TS_FIELDS} — "
                "a sample that cannot be placed in time is corrupt"
            )
            continue
        kv = next((rec[f] for f in _KV_FIELDS if f in rec), None)
        rows.append(
            {
                "ts_s": ts,
                "kv_cache_usage": kv,
                "preemptions_total": rec.get(_PREEMPT_FIELD),
            }
        )
    if problems:
        raise CampaignLayoutError(problems)
    return pd.DataFrame(rows, columns=["ts_s", "kv_cache_usage", "preemptions_total"])


def write_window_regime(
    window_dir: Path,
    *,
    t_start: float,
    t_end: float,
    attainment: float | None = None,
    telemetry_path: Path | None = None,
    min_samples: int = 2,
    min_coverage: float = 0.8,
) -> Path:
    """Compute one window's §6.1 regime inputs and write ``regime.json``.

    Reads the window's telemetry stream (default: its §1 ``cage_stats.jsonl``),
    certifies [t_start, t_end) via ``compute_window_regime_inputs``, and writes
    regime.json as an auxiliary window artifact (the organizer indexes extra
    ``*.json`` files). Label:

    - certification refused (``RegimeInputError``) -> ``label`` =
      ``UNKNOWN_TELEMETRY`` (outside the 3-label grid vocabulary), ``inputs``
      null, the refusal message in ``refusal_reason`` — absence stays absence,
      never a numeric that could read UNPRESSURED;
    - certified + ``attainment`` provided -> the §6.1 3-layer label from
      ``goodput.classify_regime`` (the ONE threshold source);
    - certified, no attainment yet -> ``label`` null (§6.1 labeling deferred
      until the caller's attainment exists — never fabricated).

    Bad window bounds are a CALLER bug, not telemetry absence: they raise
    instead of being recorded as a refusal.
    """
    window_dir = Path(window_dir)
    if not window_dir.is_dir():
        raise CampaignLayoutError(f"window directory does not exist: {window_dir}")
    for name, value in (("t_start", t_start), ("t_end", t_end)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise CampaignLayoutError(f"{name}={value!r} must be a finite number")
    if not float(t_end) > float(t_start):
        raise CampaignLayoutError(f"t_end={t_end!r} must be > t_start={t_start!r}")
    if attainment is not None and (
        isinstance(attainment, bool) or not isinstance(attainment, (int, float))
    ):
        raise CampaignLayoutError(f"attainment={attainment!r} must be a number or None")

    source = (
        Path(telemetry_path) if telemetry_path is not None else window_dir / "cage_stats.jsonl"
    )
    samples = load_telemetry_series(source)

    document: dict[str, Any] = {
        "schema_version": _REGIME_SCHEMA_VERSION,
        "t_start": float(t_start),
        "t_end": float(t_end),
        "telemetry_source": source.name,
        "attainment": None if attainment is None else float(attainment),
    }
    try:
        inputs = compute_window_regime_inputs(
            samples,
            float(t_start),
            float(t_end),
            min_samples=min_samples,
            min_coverage=min_coverage,
        )
    except RegimeInputError as exc:
        document.update(
            {
                "telemetry_ok": False,
                "inputs": None,
                "label": REGIME_UNKNOWN,
                "refusal_reason": str(exc),
            }
        )
    else:
        label: str | None = None
        if attainment is not None:
            # GoodputError (attainment outside [0,1], ...) propagates: caller bug.
            label = classify_regime(
                rho_kv=inputs.rho_kv_time_avg,
                scarcity_events=inputs.scarcity_events,
                attainment=float(attainment),
            )
        document.update(
            {
                "telemetry_ok": True,
                "inputs": inputs.to_flat_dict(),
                "label": label,
                "refusal_reason": None,
            }
        )
    return _atomic_write_json(window_dir / "regime.json", document)
