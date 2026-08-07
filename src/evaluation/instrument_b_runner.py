"""Instrument B (AlignScore-large) out-of-process runner — charter D8 §8.5.

Instrument B is the SECONDARY claim checker selected by the 2026-08-05
instrument-selection calibration: AlignScore-large in ``nli_sp`` mode
(AlignScore: zha2023alignscore — Zha et al., "AlignScore: Evaluating Factual
Consistency with a Unified Alignment Function", ACL 2023).

Why out of process (the constraint the selection run PROVED): AlignScore pins
torch<2 + pytorch-lightning 1.9.5 + transformers 4.26.1 + python 3.10 — a 2023
stack that can NEVER coexist with the modern scoring venv. So Instrument B
runs in its OWN isolated environment, exchanging JSONL with the project venv,
exactly like the decoupled re-score architecture
(``scripts/4_analysis/rescore_quality.py``): serving/scoring never blocks on
it, and a scoring bug is fixed by a new pass, never an in-place mutation.

This module is the MANAGER and runs IN the project venv. It owns:

- :class:`AlignScoreEnvSpec` — the frozen, embedded single source of truth for
  every pin (python, packages, code commit, model revisions, byte sizes,
  sha256).  The authoritative provenance artifact lives at
  ``MyDocs/registration/instrument_selection_2026-08-05/provenance.json``
  (+ ``pip_freeze_alignscore_venv.txt`` beside it), but MyDocs is gitignored,
  so the pins are EMBEDDED here and a unit test asserts the constants match
  the provenance JSON whenever that file exists
  (``tests/test_instrument_b_runner.py``).
- :func:`ensure_env` — bootstraps the isolated environment on demand into
  ``CAGE_INSTRUMENT_B_HOME`` (default ``~/.cache/cage/instrument_b`` —
  deliberately OUTSIDE the repo, so the 2026-08-04 quarantine-inside-the-repo
  mistake cannot recur; an env home inside the repo is REFUSED).
- :func:`score` — drives the checkpointed worker subprocess (JSONL in/out,
  resume by skipping already-scored ids, fsync appends, truncated-tail
  healing — the same contract the selection scorer proved on 4724 items).
- :func:`apply_tau` — the binary grounded verdict column. τ is REQUIRED as an
  explicit argument (no default at this seam — callers name the τ they apply);
  the REGISTERED value is :data:`TAU_REGISTERED` (0.817024, RAGTruth-test
  anchor scope, owner-decided 2026-08-05 — see
  ``MyDocs/registration/instrument_selection_2026-08-05/DECISION.md`` and the
  charter stamp PUBLICATION.md §8.6(c)), which
  ``scripts/4_analysis/score_instrument_b.py`` applies by default.

Engineering doctrine: fail-closed typed errors (mirroring
``InstrumentUnavailableError`` in ``src/evaluation/quality.py``) — a pin
mismatch, a missing interpreter, a sha256 mismatch or a missing score is
NEVER silently degraded.  No heavy download or model load happens in any code
path exercised by tests: bootstrap work is confined to :func:`ensure_env`'s
cold path and the worker subprocess.  Project ``.venv`` and
``requirements.txt`` are untouched — AlignScore's dependencies live ONLY in
the isolated environment this module manages.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Authoritative provenance artifact (gitignored MyDocs — may be absent on a
#: fresh clone; the embedded spec below is the in-repo copy of these facts).
PROVENANCE_PATH = (
    REPO_ROOT / "MyDocs/registration/instrument_selection_2026-08-05/provenance.json"
)

#: Env var overriding the isolated-environment home directory.
ENV_HOME_ENV_VAR = "CAGE_INSTRUMENT_B_HOME"

#: Default env home — OUTSIDE the repo by design (see module docstring).
DEFAULT_ENV_HOME = "~/.cache/cage/instrument_b"

#: The REGISTERED grounded-verdict threshold τ — OWNER-DECIDED 2026-08-05
#: (``MyDocs/registration/instrument_selection_2026-08-05/DECISION.md``;
#: charter stamp PUBLICATION.md §8.6(c)): the smallest threshold achieving
#: ≥90% precision on the RAGTruth-test anchor (n=2675; precision 0.9002,
#: recall/sensitivity 0.5468), computed from the ``alignscore_large`` scores
#: in that directory's ``scores/``. The pooled (RAGTruth+TRUE) scope was
#: REJECTED — FRANK's summarization base rate caps its top tail at
#: τ=0.989/recall 0.079, an anchor-composition artifact.
TAU_REGISTERED: float = 0.817024

#: Anchor scope the registered τ is defined on (DECISION.md §1): the
#: RAGTruth-test component ONLY — the RAG-domain anchor; CAGE measures
#: RAG-style serving.
TAU_ANCHOR_SCOPE: str = "ragtruth_test"

#: File name of the materialized worker script inside the env home.
WORKER_FILE_NAME = "alignscore_worker.py"

#: Readiness marker written at the end of a successful bootstrap.
ENV_MANIFEST_NAME = "env_manifest.json"

#: ``pip freeze`` of the isolated env, captured at bootstrap for provenance.
PIP_FREEZE_NAME = "pip_freeze.txt"


# --------------------------------------------------------------------------- #
# Typed errors (fail closed — mirror quality.InstrumentUnavailableError)
# --------------------------------------------------------------------------- #


class InstrumentBError(RuntimeError):
    """Base error for the Instrument-B runner (fail closed, D8 §8.5).

    Mirrors ``InstrumentUnavailableError`` in ``src/evaluation/quality.py``:
    an instrument that cannot be provisioned, verified or scored must raise —
    never return a degraded number under the same column name.
    """


class InstrumentBInterpreterError(InstrumentBError):
    """No suitable Python interpreter for the pinned 2023 stack was found."""


class InstrumentBVerificationError(InstrumentBError):
    """A downloaded artifact failed byte-size or sha256 verification."""


class InstrumentBWorkerError(InstrumentBError):
    """The out-of-process worker failed or returned an incomplete score set."""


class InstrumentBTauError(InstrumentBError):
    """Invalid τ or missing/invalid scores when producing grounded verdicts."""


class InstrumentBSpecMismatchError(InstrumentBError):
    """Embedded spec constants disagree with the provenance artifact."""


# --------------------------------------------------------------------------- #
# The frozen environment spec — single source of truth for every pin
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AlignScoreEnvSpec:
    """Every pin needed to reconstruct the exact selection-run environment.

    Values are verbatim from the selection-calibration provenance artifact
    ``MyDocs/registration/instrument_selection_2026-08-05/provenance.json``
    (models.alignscore_large + code.alignscore) and the exact package pins in
    ``pip_freeze_alignscore_venv.txt`` next to it.  MyDocs is gitignored, so
    this frozen dataclass IS the tracked copy; a unit test asserts it matches
    the provenance JSON whenever that file exists.
    """

    spec_version: int = 1

    # -- interpreter -------------------------------------------------------- #
    # AlignScore's 2023 stack (torch<2, pytorch-lightning<2) is incompatible
    # with py3.13+; the selection ran on a uv-managed 3.10.20.
    python_minors: tuple[str, ...] = ("3.10", "3.11")
    python_bootstrap_version: str = "3.10"  # requested from uv when no system python fits

    # -- package pins (pip_freeze_alignscore_venv.txt) ---------------------- #
    torch_pin: str = "1.13.1"
    transformers_pin: str = "4.26.1"
    pytorch_lightning_pin: str = "1.9.5"
    spacy_pin: str = "3.8.14"
    huggingface_hub_pin: str = "0.36.2"
    en_core_web_sm_url: str = (
        "https://github.com/explosion/spacy-models/releases/download/"
        "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
        "#sha256=1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85"
    )

    # -- AlignScore code (zha2023alignscore reference implementation) ------- #
    code_github_repo: str = "yuh-zha/AlignScore"
    code_commit_sha: str = "a0936d5afee642a46b22f6c02a163478447aa493"

    # -- model checkpoint --------------------------------------------------- #
    model_repo_id: str = "yzha/AlignScore"
    model_revision: str = "8509e78d25bb914939fc585c626500c9b2944249"
    ckpt_file_name: str = "AlignScore-large.ckpt"
    ckpt_byte_size: int = 4895673790
    ckpt_sha256: str = (
        "ff4336312b377edcbcdad5694a2d09d73dc4225422c0422d810aa7e78485e32d"
    )

    # -- backbone (AlignScore ctor loads roberta-large before the ckpt) ----- #
    backbone_repo_id: str = "roberta-large"
    backbone_revision: str = "722cf37b1afa9454edce342e7895e588b6ff1d59"
    backbone_weights_file: str = "pytorch_model.bin"
    backbone_weights_byte_size: int = 1425941629
    backbone_weights_sha256: str = (
        "36a10a8b694fadf9bf4f9049d14e257e88be45313ae02d882af9e60f39b8b2e8"
    )

    # -- scoring configuration (selection-run operating configuration) ------ #
    evaluation_mode: str = "nli_sp"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


#: The module-level spec instance every entry point defaults to.
SPEC = AlignScoreEnvSpec()


def spec_fingerprint(spec: AlignScoreEnvSpec = SPEC) -> str:
    """Order-invariant content hash of the spec (readiness/no-op key)."""
    canonical = json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def assert_spec_matches_provenance(
    provenance: Mapping[str, Any], spec: AlignScoreEnvSpec = SPEC
) -> None:
    """Fail closed if the embedded pins disagree with the provenance artifact.

    ``provenance`` is the parsed provenance.json.  Raises
    :class:`InstrumentBSpecMismatchError` listing EVERY divergent field —
    a silent drift between the tracked constants and the (gitignored)
    authority is exactly what this check exists to catch.
    """
    mismatches: list[str] = []

    def check(field: str, expected: Any, actual: Any) -> None:
        if expected != actual:
            mismatches.append(f"{field}: spec={expected!r} provenance={actual!r}")

    try:
        model = provenance["models"]["alignscore_large"]
        ckpt = model["files"][spec.ckpt_file_name]
        backbone = model["backbone_dependency"]
        code = provenance["code"]["alignscore"]
        versions = code["key_versions"]
    except (KeyError, TypeError) as exc:
        raise InstrumentBSpecMismatchError(
            f"provenance JSON lacks the expected structure: {exc!r}"
        ) from exc

    check("model_repo_id", spec.model_repo_id, model.get("hf_repo_id"))
    check("model_revision", spec.model_revision, model.get("revision_commit_sha"))
    check("ckpt_byte_size", spec.ckpt_byte_size, ckpt.get("byte_size"))
    check("ckpt_sha256", spec.ckpt_sha256, ckpt.get("hub_lfs_sha256"))
    if ckpt.get("local_sha256") is not None:
        check("ckpt_sha256(local)", spec.ckpt_sha256, ckpt.get("local_sha256"))

    check("backbone_repo_id", spec.backbone_repo_id, backbone.get("hf_repo_id"))
    check(
        "backbone_revision", spec.backbone_revision, backbone.get("revision_commit_sha")
    )
    check(
        "backbone_weights_file", spec.backbone_weights_file, backbone.get("file_name")
    )
    check(
        "backbone_weights_byte_size",
        spec.backbone_weights_byte_size,
        backbone.get("byte_size"),
    )
    check(
        "backbone_weights_sha256",
        spec.backbone_weights_sha256,
        backbone.get("hub_lfs_sha256"),
    )

    check("code_github_repo", spec.code_github_repo, code.get("github_repo"))
    check("code_commit_sha", spec.code_commit_sha, code.get("commit_sha_at_install"))

    check("torch_pin", spec.torch_pin, versions.get("torch"))
    check("transformers_pin", spec.transformers_pin, versions.get("transformers"))
    check(
        "pytorch_lightning_pin",
        spec.pytorch_lightning_pin,
        versions.get("pytorch_lightning"),
    )
    check("spacy_pin", spec.spacy_pin, versions.get("spacy"))

    python_version = str(versions.get("python", ""))
    if not any(
        python_version.startswith(minor + ".") or python_version == minor
        for minor in spec.python_minors
    ):
        mismatches.append(
            f"python: provenance ran {python_version!r}, spec accepts "
            f"minors {spec.python_minors}"
        )

    if mismatches:
        raise InstrumentBSpecMismatchError(
            "embedded AlignScoreEnvSpec disagrees with provenance.json:\n  "
            + "\n  ".join(mismatches)
        )


# --------------------------------------------------------------------------- #
# Env home + interpreter discovery
# --------------------------------------------------------------------------- #


def default_env_home() -> Path:
    """Resolve the env home: ``$CAGE_INSTRUMENT_B_HOME`` else the default."""
    raw = os.environ.get(ENV_HOME_ENV_VAR, DEFAULT_ENV_HOME)
    return Path(raw).expanduser()


@dataclass(frozen=True)
class PythonInterpreter:
    """How the isolated venv will be created.

    ``kind='system'``: ``command`` is a compatible ``python3.10``/``python3.11``
    executable (``python -m venv``).  ``kind='uv'``: ``command`` is the ``uv``
    binary (``uv venv --python <version>``, uv provisions the interpreter).
    """

    kind: str
    command: str


def discover_python(spec: AlignScoreEnvSpec = SPEC) -> PythonInterpreter:
    """Find an interpreter able to host the pinned 2023 stack.

    Preference order: ``python3.10``/``python3.11`` on PATH, else a uv-managed
    interpreter if ``uv`` exists.  Fail-closed otherwise, naming BOTH remedies.
    """
    for minor in spec.python_minors:
        exe = shutil.which(f"python{minor}")
        if exe:
            return PythonInterpreter(kind="system", command=exe)
    uv = shutil.which("uv")
    if uv:
        return PythonInterpreter(kind="uv", command=uv)
    raise InstrumentBInterpreterError(
        "Instrument B (AlignScore) needs a Python "
        + "/".join(spec.python_minors)
        + " interpreter (its 2023 stack — torch "
        + spec.torch_pin
        + ", pytorch-lightning "
        + spec.pytorch_lightning_pin
        + " — cannot run on the project venv's modern Python). Remedies: "
        "(a) install Python 3.10 or 3.11 so `python3.10`/`python3.11` is on "
        "PATH (e.g. `brew install python@3.10`), or (b) install uv "
        "(https://docs.astral.sh/uv/ — `curl -LsSf https://astral.sh/uv/install.sh | sh`) "
        "and the runner will provision a managed "
        + spec.python_bootstrap_version
        + " interpreter."
    )


# --------------------------------------------------------------------------- #
# Artifact verification (fail closed on any mismatch)
# --------------------------------------------------------------------------- #


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact(path: Path, expected_byte_size: int, expected_sha256: str) -> None:
    """Verify one pinned binary against the spec; raise on ANY divergence.

    A wrong-size or wrong-hash instrument binary silently scoring the campaign
    is the exact failure mode D8 §8.5 forbids — so existence, byte size and
    sha256 are all checked, and the first divergence raises
    :class:`InstrumentBVerificationError` (the file is left in place for
    forensics; it is never marked ready).
    """
    if not path.is_file():
        raise InstrumentBVerificationError(f"pinned artifact missing: {path}")
    actual_size = path.stat().st_size
    if actual_size != expected_byte_size:
        raise InstrumentBVerificationError(
            f"byte-size mismatch for {path}: expected {expected_byte_size}, "
            f"got {actual_size}"
        )
    actual_sha = _sha256_file(path)
    if actual_sha != expected_sha256:
        raise InstrumentBVerificationError(
            f"sha256 mismatch for {path}: expected {expected_sha256}, "
            f"got {actual_sha}"
        )


# --------------------------------------------------------------------------- #
# The worker script (materialized into the env home, run with the env python)
# --------------------------------------------------------------------------- #

#: The out-of-process worker. Pure stdlib + the ``alignscore`` import; the
#: manager materializes it into the env home via :func:`write_worker`.
#: Contract (proved by the 2026-08-05 selection scorer on 4724 items):
#: batch scoring, fsync'd JSONL appends, resume by skipping already-scored
#: ids, truncated-tail healing, rows ``{"id", "score", "wall_ms"}``.
WORKER_SOURCE: str = r'''#!/usr/bin/env python
"""AlignScore Instrument-B worker (charter D8 §8.5) - RUNS IN THE ISOLATED ENV.

Materialized by src/evaluation/instrument_b_runner.py (the manager, which runs
in the project venv). Pure stdlib + the `alignscore` import
(zha2023alignscore, Zha et al. ACL 2023).

Checkpointed-JSONL contract (the one the 2026-08-05 selection scorer proved):
- reads items {"id", "context", "claim"} from --input (JSONL);
- resumes by collecting ids already present in --output and skipping them;
- heals a truncated tail line in --output (a crash mid-append leaves a
  partial last line; it is truncated away before appending);
- scores in --batch-sized batches, appending {"id", "score", "wall_ms"}
  per item with flush+fsync after every batch (a kill loses at most one
  batch, never corrupts more than the tail line);
- prints PROGRESS lines to stdout for the manager to stream.

Exit codes: 0 ok; 2 usage/contract error; 3 scoring-contract violation.
"""
import argparse
import json
import os
import sys
import time


def heal_and_collect(output_path):
    """Collect already-scored ids; truncate any corrupt/partial tail.

    A line is GOOD iff it is newline-terminated, parses as JSON, and carries
    an "id" and a float "score". The file is truncated at the first non-good
    line: everything before is trusted (it was fsync'd), everything after is
    a torn write from a crash.
    """
    scored = set()
    if not os.path.exists(output_path):
        return scored
    with open(output_path, "rb") as fh:
        data = fh.read()
    good_end = 0
    for line in data.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break  # partial tail (no newline): torn write
        try:
            rec = json.loads(line.decode("utf-8"))
            rec_id = rec["id"]
            float(rec["score"])
        except (ValueError, KeyError, TypeError, UnicodeDecodeError):
            break  # corrupt line: distrust it and everything after
        scored.add(rec_id)
        good_end += len(line)
    if good_end != len(data):
        with open(output_path, "rb+") as fh:
            fh.truncate(good_end)
        print(
            "WORKER healed truncated tail: kept %d bytes, dropped %d"
            % (good_end, len(data) - good_end),
            flush=True,
        )
    return scored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="items JSONL: id/context/claim")
    parser.add_argument("--output", required=True, help="scores JSONL (append, resumable)")
    parser.add_argument("--ckpt", required=True, help="AlignScore-large.ckpt path")
    parser.add_argument("--model", default="roberta-large", help="backbone model id")
    parser.add_argument("--batch", type=int, default=8, help="scoring batch size")
    parser.add_argument("--device", default="cpu", help="torch device")
    parser.add_argument(
        "--evaluation-mode", default="nli_sp", help="AlignScore evaluation mode"
    )
    parser.add_argument(
        "--max-items", type=int, default=None, help="score at most N remaining items"
    )
    args = parser.parse_args()
    if args.batch < 1:
        print("WORKER error: --batch must be >= 1", file=sys.stderr)
        return 2

    with open(args.input, "r", encoding="utf-8") as fh:
        items = [json.loads(line) for line in fh if line.strip()]
    for item in items:
        if "id" not in item or "context" not in item or "claim" not in item:
            print(
                "WORKER error: input item lacks id/context/claim: %r" % (item,),
                file=sys.stderr,
            )
            return 2

    scored = heal_and_collect(args.output)
    todo = [item for item in items if item["id"] not in scored]
    if args.max_items is not None:
        todo = todo[: args.max_items]
    total = len(items)
    print(
        "WORKER start total=%d already_scored=%d todo=%d batch=%d"
        % (total, len(scored), len(todo), args.batch),
        flush=True,
    )
    if not todo:
        print("WORKER done (nothing to score)", flush=True)
        return 0

    from alignscore import AlignScore  # the ONLY non-stdlib import

    scorer = AlignScore(
        model=args.model,
        batch_size=args.batch,
        device=args.device,
        ckpt_path=args.ckpt,
        evaluation_mode=args.evaluation_mode,
    )

    n_done = len(scored)
    with open(args.output, "a", encoding="utf-8") as out:
        for start in range(0, len(todo), args.batch):
            batch = todo[start : start + args.batch]
            t0 = time.time()
            scores = scorer.score(
                contexts=[item["context"] for item in batch],
                claims=[item["claim"] for item in batch],
            )
            wall_ms = (time.time() - t0) * 1000.0 / len(batch)
            if len(scores) != len(batch):
                print(
                    "WORKER error: scorer returned %d scores for %d items"
                    % (len(scores), len(batch)),
                    file=sys.stderr,
                )
                return 3
            for item, score in zip(batch, scores):
                out.write(
                    json.dumps(
                        {
                            "id": item["id"],
                            "score": float(score),
                            "wall_ms": round(wall_ms, 3),
                        }
                    )
                    + "\n"
                )
            out.flush()
            os.fsync(out.fileno())
            n_done += len(batch)
            print("PROGRESS scored=%d/%d" % (n_done, total), flush=True)
    print("WORKER done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def write_worker(env_home: Path) -> Path:
    """Materialize the worker script into the env home; returns its path."""
    env_home.mkdir(parents=True, exist_ok=True)
    worker_path = env_home / WORKER_FILE_NAME
    worker_path.write_text(WORKER_SOURCE, encoding="utf-8")
    return worker_path


# --------------------------------------------------------------------------- #
# Bootstrap: ensure_env
# --------------------------------------------------------------------------- #


def _run(
    cmd: Sequence[str],
    *,
    what: str,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one bootstrap step; fail closed with the stderr tail on error."""
    try:
        proc = subprocess.run(
            list(cmd),
            capture_output=True,
            text=True,
            env=dict(env) if env is not None else None,
            cwd=str(cwd) if cwd is not None else None,
        )
    except OSError as exc:
        raise InstrumentBError(f"{what}: cannot execute {cmd[0]!r}: {exc}") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-15:]
        raise InstrumentBError(
            f"{what} failed (exit {proc.returncode}): {' '.join(cmd)}\n"
            + "\n".join(tail)
        )
    return proc


def _env_python(env_home: Path) -> Path:
    """Path of the isolated env's interpreter (POSIX layout; darwin/linux)."""
    return env_home / "venv" / "bin" / "python"


def _hf_env(env_home: Path) -> dict[str, str]:
    """Subprocess env with HF_HOME isolation into the env home.

    Keeps every model byte under ``<env_home>/hf`` — never the user's global
    ``~/.cache/huggingface`` — so teardown is a single directory delete and no
    global cache is polluted (the quarantine lesson, minus the in-repo path).
    """
    hf_home = env_home / "hf"
    env = dict(os.environ)
    env["HF_HOME"] = str(hf_home)
    env["TRANSFORMERS_CACHE"] = str(hf_home / "hub")
    env["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


#: Pinned-revision download program, run with the ISOLATED env's python
#: (the project venv has no huggingface_hub — verified — and must not gain
#: one).  argv[1] = JSON spec, argv[2] = output path for the resolved paths.
#:
#: The trailing refs/main pin is load-bearing: ``snapshot_download`` at a
#: commit SHA records NO ``refs/main``, but the worker loads
#: ``AutoModel.from_pretrained('roberta-large')`` (AlignScore's ctor passes no
#: revision, so hub resolution defaults to "main") under ``HF_HUB_OFFLINE=1``,
#: and offline cache resolution goes through ``refs/<revision>``.  Writing
#: ``refs/main`` = the verified pinned SHA makes "roberta-large" MEAN that SHA
#: inside this env — without it the first real scoring run fails offline.
DOWNLOAD_SNIPPET: str = (
    "import json, os, sys\n"
    "from huggingface_hub import hf_hub_download, snapshot_download\n"
    "spec = json.loads(sys.argv[1])\n"
    "ckpt = hf_hub_download(repo_id=spec['model_repo_id'],"
    " filename=spec['ckpt_file_name'], revision=spec['model_revision'],"
    " cache_dir=spec['cache_dir'])\n"
    "snap = snapshot_download(repo_id=spec['backbone_repo_id'],"
    " revision=spec['backbone_revision'], cache_dir=spec['cache_dir'],"
    " allow_patterns=['config.json', 'vocab.json', 'merges.txt',"
    " 'tokenizer.json', 'tokenizer_config.json', 'pytorch_model.bin'])\n"
    "refs_dir = os.path.join(os.path.dirname(os.path.dirname(snap)), 'refs')\n"
    "os.makedirs(refs_dir, exist_ok=True)\n"
    "with open(os.path.join(refs_dir, 'main'), 'w') as fh:\n"
    "    fh.write(spec['backbone_revision'])\n"
    "json.dump({'ckpt': ckpt, 'backbone_snapshot': snap},"
    " open(sys.argv[2], 'w'))\n"
)


def read_env_manifest(env_home: Path) -> dict[str, Any]:
    """Read the readiness manifest; typed error when absent or not ready."""
    manifest_path = env_home / ENV_MANIFEST_NAME
    if not manifest_path.is_file():
        raise InstrumentBError(
            f"no {ENV_MANIFEST_NAME} under {env_home} — run ensure_env() (or "
            "`scripts/4_analysis/score_instrument_b.py --bootstrap-only`) first"
        )
    try:
        manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InstrumentBError(f"{manifest_path} is not valid JSON: {exc}") from exc
    if manifest.get("ready") is not True:
        raise InstrumentBError(f"{manifest_path} does not mark a ready environment")
    return manifest


def ensure_env(
    env_home: Path | None = None, spec: AlignScoreEnvSpec = SPEC
) -> Path:
    """Bootstrap the isolated AlignScore environment on demand; idempotent.

    A ready env home (``env_manifest.json`` with a matching spec fingerprint)
    makes this a no-op.  Otherwise: discover an interpreter
    (:func:`discover_python`), create the venv, pip-install the pinned 2023
    stack, download the ckpt and backbone AT PINNED REVISIONS via
    ``huggingface_hub`` (running inside the isolated env, with HF_HOME
    isolated into the env home), verify byte sizes and sha256 against the
    spec (fail closed), materialize the worker, snapshot ``pip freeze``, and
    write the readiness manifest LAST so a crashed bootstrap is retried, not
    trusted.

    Returns the env home path.
    """
    home = (Path(env_home).expanduser() if env_home is not None else default_env_home())
    home = home.resolve() if home.exists() else Path(os.path.abspath(home))

    # The 2026-08-04 lesson: instrument quarantine INSIDE the repo polluted the
    # tree and risked accidental commits of a 14G cache. Refuse, always.
    try:
        home.relative_to(REPO_ROOT)
    except ValueError:
        pass  # outside the repo — good
    else:
        raise InstrumentBError(
            f"Instrument-B env home {home} is INSIDE the repo ({REPO_ROOT}); "
            "refusing (the 2026-08-04 quarantine-in-repo mistake). Use the "
            f"default {DEFAULT_ENV_HOME} or set {ENV_HOME_ENV_VAR} to a "
            "directory outside the repository."
        )

    manifest_path = home / ENV_MANIFEST_NAME
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
        if (
            manifest.get("ready") is True
            and manifest.get("spec_fingerprint") == spec_fingerprint(spec)
        ):
            return home  # ready — no-op

    home.mkdir(parents=True, exist_ok=True)
    interpreter = discover_python(spec)
    venv_dir = home / "venv"

    # (1) venv creation.
    if interpreter.kind == "system":
        _run(
            [interpreter.command, "-m", "venv", "--clear", str(venv_dir)],
            what="venv creation",
        )
    else:  # uv-managed interpreter
        _run(
            [
                interpreter.command,
                "venv",
                # --seed: uv venvs ship WITHOUT pip by default, but the pinned
                # installs below run `python -m pip` inside the env.
                "--seed",
                "--python",
                spec.python_bootstrap_version,
                str(venv_dir),
            ],
            what="uv venv creation",
        )
    env_python = _env_python(home)

    # (2) pinned installs — torch FIRST (provenance install order: alignscore's
    # setup resolves against the already-present torch<2).
    _run(
        [str(env_python), "-m", "pip", "install", f"torch=={spec.torch_pin}"],
        what="pip install torch",
    )
    _run(
        [
            str(env_python),
            "-m",
            "pip",
            "install",
            "alignscore @ git+https://github.com/"
            f"{spec.code_github_repo}.git@{spec.code_commit_sha}",
            f"pytorch_lightning=={spec.pytorch_lightning_pin}",
            f"transformers=={spec.transformers_pin}",
            f"spacy=={spec.spacy_pin}",
            f"huggingface_hub=={spec.huggingface_hub_pin}",
            f"en_core_web_sm @ {spec.en_core_web_sm_url}",
        ],
        what="pip install alignscore stack",
    )

    # (3) pinned downloads via huggingface_hub INSIDE the isolated env, with
    # HF_HOME isolation; resolved paths come back through a JSON file so
    # progress output cannot corrupt the channel.
    hf_env = _hf_env(home)
    cache_dir = home / "hf" / "hub"
    with tempfile.TemporaryDirectory(dir=str(home)) as tmp:
        paths_out = Path(tmp) / "paths.json"
        _run(
            [
                str(env_python),
                "-c",
                DOWNLOAD_SNIPPET,
                json.dumps(
                    {
                        "model_repo_id": spec.model_repo_id,
                        "ckpt_file_name": spec.ckpt_file_name,
                        "model_revision": spec.model_revision,
                        "backbone_repo_id": spec.backbone_repo_id,
                        "backbone_revision": spec.backbone_revision,
                        "cache_dir": str(cache_dir),
                    }
                ),
                str(paths_out),
            ],
            what="pinned model download",
            env=hf_env,
        )
        paths = json.loads(paths_out.read_text(encoding="utf-8"))
    ckpt_path = Path(paths["ckpt"])
    backbone_weights = Path(paths["backbone_snapshot"]) / spec.backbone_weights_file

    # (4) fail-closed verification against the embedded pins.
    verify_artifact(ckpt_path, spec.ckpt_byte_size, spec.ckpt_sha256)
    verify_artifact(
        backbone_weights,
        spec.backbone_weights_byte_size,
        spec.backbone_weights_sha256,
    )

    # (5) worker + provenance snapshot.
    worker_path = write_worker(home)
    freeze = _run(
        [str(env_python), "-m", "pip", "freeze"], what="pip freeze"
    ).stdout
    (home / PIP_FREEZE_NAME).write_text(freeze, encoding="utf-8")

    # (6) readiness manifest LAST — subsequent calls are no-ops.
    manifest = {
        "schema_version": 1,
        "ready": True,
        "spec_version": spec.spec_version,
        "spec_fingerprint": spec_fingerprint(spec),
        "spec": spec.to_dict(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "interpreter_kind": interpreter.kind,
        "env_python": str(env_python),
        "worker": str(worker_path),
        "pip_freeze": PIP_FREEZE_NAME,
        "verified": {
            "ckpt": {
                "path": str(ckpt_path),
                "byte_size": spec.ckpt_byte_size,
                "sha256": spec.ckpt_sha256,
            },
            "backbone_weights": {
                "path": str(backbone_weights),
                "byte_size": spec.backbone_weights_byte_size,
                "sha256": spec.backbone_weights_sha256,
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return home


# --------------------------------------------------------------------------- #
# score(): drive the checkpointed worker subprocess
# --------------------------------------------------------------------------- #


def _validate_items(items: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Fail closed on malformed scoring items; returns normalized copies."""
    if not items:
        raise InstrumentBError("no items to score (empty item sequence)")
    seen: set[str] = set()
    normalized: list[dict[str, str]] = []
    for index, item in enumerate(items):
        item_id = item.get("id")
        context = item.get("context")
        claim = item.get("claim")
        if not isinstance(item_id, str) or not item_id:
            raise InstrumentBError(f"item #{index} lacks a non-empty string 'id'")
        if item_id in seen:
            raise InstrumentBError(f"duplicate item id {item_id!r}")
        if not isinstance(context, str) or not context.strip():
            raise InstrumentBError(
                f"item {item_id!r} lacks a non-empty 'context' — an empty "
                "premise cannot ground anything; filter such rows upstream"
            )
        if not isinstance(claim, str) or not claim.strip():
            raise InstrumentBError(
                f"item {item_id!r} lacks a non-empty 'claim' — nothing to check"
            )
        seen.add(item_id)
        normalized.append({"id": item_id, "context": context, "claim": claim})
    return normalized


def read_scores(output_path: Path) -> dict[str, float]:
    """Parse a worker output JSONL into ``{id: score}`` (tolerates a torn tail).

    Mirrors the worker's own healing rule for READING: a final partial line is
    ignored (the worker will truncate it on its next run); any interior
    corruption raises — the fsync-append contract makes that impossible short
    of external tampering, which must not pass silently.
    """
    scores: dict[str, float] = {}
    if not output_path.is_file():
        return scores
    data = output_path.read_bytes()
    lines = data.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.endswith(b"\n"):
            if index == len(lines) - 1:
                break  # torn tail — healed by the worker on its next run
            raise InstrumentBWorkerError(
                f"{output_path}: interior line {index + 1} lacks a newline"
            )
        try:
            rec = json.loads(line.decode("utf-8"))
            scores[str(rec["id"])] = float(rec["score"])
        except (ValueError, KeyError, TypeError) as exc:
            if index == len(lines) - 1:
                break  # corrupt tail — healed by the worker on its next run
            raise InstrumentBWorkerError(
                f"{output_path}: corrupt interior line {index + 1}: {exc}"
            ) from exc
    return scores


def _job_dir_for(home: Path, normalized: Sequence[Mapping[str, str]]) -> Path:
    """Content-addressed default work dir: ``<home>/work/<sha256[:16]>``.

    The job key hashes the FULL normalized items (id + context + claim), not
    just ids.  This is load-bearing: worker resume skips by id alone, and §6
    tree-mode ids are root-relative (``cells/<row_key>/window_<k>/...``) —
    identical across different sealed run roots.  A shared work dir would let
    a second run silently inherit a FIRST run's scores for different
    contexts/claims (confirmed by repro).  Same items → same dir → genuine
    crash-safe resume; any content change → fresh dir → stale scores can
    never be joined.
    """
    digest = hashlib.sha256()
    for item in normalized:
        digest.update(json.dumps(item, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return home / "work" / digest.hexdigest()[:16]


def score(
    items: Sequence[Mapping[str, Any]],
    *,
    env_home: Path | None = None,
    spec: AlignScoreEnvSpec = SPEC,
    work_dir: Path | None = None,
    batch_size: int = 8,
    device: str = "cpu",
    max_items: int | None = None,
    stream: bool = True,
) -> dict[str, float]:
    """Score items out of process; returns ``{id: alignscore}``.

    Items are mappings with ``id``/``context``/``claim``.  The worker is
    checkpointed: rerunning with the same items (or the same explicit
    ``work_dir``) resumes by skipping already-scored ids (crash-safe via
    fsync appends + tail healing).  The default work dir is content-addressed
    over the full items (:func:`_job_dir_for`), so a different job — even one
    with colliding ids — can never silently reuse stale scores.  With
    ``max_items=None`` every input id must come back scored — anything less
    raises (fail closed); with ``max_items`` set, the partial score set is
    returned as-is (resumable smoke/budget mode).

    Bootstraps the environment on demand via :func:`ensure_env` (no-op when
    ready).
    """
    if batch_size < 1:
        raise InstrumentBError(f"batch_size must be >= 1, got {batch_size}")
    normalized = _validate_items(items)

    home = ensure_env(env_home, spec)
    manifest = read_env_manifest(home)
    env_python = Path(manifest["env_python"])
    worker_path = Path(manifest["worker"])
    ckpt_path = Path(manifest["verified"]["ckpt"]["path"])

    job_dir = (
        Path(work_dir) if work_dir is not None else _job_dir_for(home, normalized)
    )
    job_dir.mkdir(parents=True, exist_ok=True)
    input_path = job_dir / "input.jsonl"
    output_path = job_dir / "scores.jsonl"
    with input_path.open("w", encoding="utf-8") as fh:
        for item in normalized:
            fh.write(json.dumps(item) + "\n")

    cmd: list[str] = [
        str(env_python),
        str(worker_path),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--ckpt",
        str(ckpt_path),
        "--model",
        spec.backbone_repo_id,
        "--batch",
        str(batch_size),
        "--device",
        device,
        "--evaluation-mode",
        spec.evaluation_mode,
    ]
    if max_items is not None:
        cmd += ["--max-items", str(max_items)]

    worker_env = _hf_env(home)
    worker_env["HF_HUB_OFFLINE"] = "1"  # verified pinned cache only — never refetch

    # stderr goes to a FILE, never a PIPE: the manager reads stdout line by
    # line while the worker runs, so an undrained stderr pipe would deadlock
    # the worker once transformers/lightning warnings + tqdm output pass the
    # ~64KB pipe buffer on a multi-hour run.  The file also survives for
    # forensics; on failure its tail becomes the typed error.
    stderr_path = job_dir / "worker_stderr.log"
    with stderr_path.open("w", encoding="utf-8") as stderr_fh:
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=stderr_fh,
                text=True,
                env=worker_env,
            )
        except OSError as exc:
            raise InstrumentBWorkerError(f"cannot launch worker: {exc}") from exc
        assert proc.stdout is not None  # PIPE above
        for line in proc.stdout:
            if stream:
                print(f"[instrument-b] {line.rstrip()}", flush=True)
        proc.wait()
    if proc.returncode != 0:
        try:
            stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            stderr_text = ""
        tail = stderr_text.strip().splitlines()[-15:]
        raise InstrumentBWorkerError(
            f"worker exited {proc.returncode} (stderr: {stderr_path}):\n"
            + "\n".join(tail)
        )

    scores = read_scores(output_path)
    wanted = {item["id"] for item in normalized}
    joined = {item_id: scores[item_id] for item_id in wanted if item_id in scores}
    if max_items is None and len(joined) != len(wanted):
        missing = sorted(wanted - set(joined))
        raise InstrumentBWorkerError(
            f"worker returned {len(joined)}/{len(wanted)} scores; missing e.g. "
            f"{missing[:5]} — refusing a silent partial score set"
        )
    return joined


# --------------------------------------------------------------------------- #
# apply_tau(): the binary grounded verdict column
# --------------------------------------------------------------------------- #


def apply_tau(scores: Mapping[str, Any], tau: float) -> dict[str, bool]:
    """Binary grounded verdicts: ``grounded_b = alignscore >= tau``.

    τ is REQUIRED with no default at THIS seam — every caller names the τ it
    applies (provenance over convenience). The registered value is
    :data:`TAU_REGISTERED` (0.817024 on the :data:`TAU_ANCHOR_SCOPE` =
    'ragtruth_test' anchor, owner-decided 2026-08-05 — DECISION.md +
    PUBLICATION.md §8.6(c)); ``scripts/4_analysis/score_instrument_b.py``
    passes it by default and records whether the applied τ was registered or
    an override.
    The ``>=`` boundary mirrors ``select_tau``'s predicted-grounded rule in
    ``src/evaluation/instrument_calibration.py`` (a score exactly at τ IS
    grounded).  Missing/non-finite scores fail closed.
    """
    if isinstance(tau, bool) or not isinstance(tau, (int, float)):
        raise InstrumentBTauError(f"tau must be a real number, got {tau!r}")
    tau_value = float(tau)
    if not (tau_value == tau_value) or tau_value in (float("inf"), float("-inf")):
        raise InstrumentBTauError(f"tau must be finite, got {tau!r}")
    if not 0.0 <= tau_value <= 1.0:
        raise InstrumentBTauError(
            f"tau must be within [0, 1] (AlignScore is a probability), got {tau!r}"
        )
    if not scores:
        raise InstrumentBTauError("no scores to threshold (empty score mapping)")
    verdicts: dict[str, bool] = {}
    for item_id, value in scores.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InstrumentBTauError(
                f"score for {item_id!r} is missing or non-numeric ({value!r}) — "
                "fail closed: a row without a score gets NO verdict, not False"
            )
        value_f = float(value)
        if value_f != value_f or value_f in (float("inf"), float("-inf")):
            raise InstrumentBTauError(
                f"score for {item_id!r} is non-finite ({value!r})"
            )
        verdicts[item_id] = value_f >= tau_value
    return verdicts
