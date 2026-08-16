"""Tests for the Instrument-B (AlignScore) out-of-process runner (D8 §8.5).

All offline: no downloads, no alignscore install, no model loads. The worker
CONTRACT (batching, resume, truncated-tail healing, output schema) is proved
by running the REAL worker script under the CURRENT python with a stub
``alignscore`` module injected via PYTHONPATH — the real dependency stack is
never touched (it can only exist inside the isolated env the runner manages).
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from src.evaluation import instrument_b_runner as ib
from src.evaluation.instrument_b_runner import (
    PROVENANCE_PATH,
    SPEC,
    TAU_ANCHOR_SCOPE,
    TAU_REGISTERED,
    AlignScoreEnvSpec,
    InstrumentBError,
    InstrumentBInterpreterError,
    InstrumentBSpecMismatchError,
    InstrumentBTauError,
    InstrumentBVerificationError,
    InstrumentBWorkerError,
    apply_tau,
    assert_spec_matches_provenance,
    discover_python,
    read_scores,
    spec_fingerprint,
    verify_artifact,
    write_worker,
)

# --------------------------------------------------------------------------- #
# Stub alignscore package (injected into the worker via PYTHONPATH)
# --------------------------------------------------------------------------- #

_STUB_ALIGNSCORE = '''\
"""Stub alignscore for worker-contract tests: deterministic, dependency-free."""
import json
import os


def _log(event):
    path = os.environ.get("STUB_ALIGNSCORE_CALLS")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event) + "\\n")


class AlignScore:
    def __init__(self, model=None, batch_size=None, device=None,
                 ckpt_path=None, evaluation_mode=None):
        _log({"event": "init", "model": model, "batch_size": batch_size,
              "device": device, "ckpt_path": ckpt_path,
              "evaluation_mode": evaluation_mode})

    def score(self, contexts=None, claims=None):
        _log({"event": "score", "n": len(claims), "claims": list(claims)})
        return [stub_score(c) for c in claims]


def stub_score(claim):
    return round((len(claim) % 7) / 10.0, 3)
'''


def _stub_score(claim: str) -> float:
    """Mirror of the stub's deterministic scoring formula."""
    return round((len(claim) % 7) / 10.0, 3)


@pytest.fixture()
def worker_env(tmp_path: Path) -> dict[str, Any]:
    """Materialized worker + stub package + item files, ready to invoke."""
    stub_dir = tmp_path / "stub"
    (stub_dir / "alignscore").mkdir(parents=True)
    (stub_dir / "alignscore" / "__init__.py").write_text(
        _STUB_ALIGNSCORE, encoding="utf-8"
    )
    env_home = tmp_path / "envhome"
    worker_path = write_worker(env_home)
    items = [
        {"id": f"i{n}", "context": f"context {n}", "claim": "c" * (n + 3)}
        for n in range(5)
    ]
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        "".join(json.dumps(item) + "\n" for item in items), encoding="utf-8"
    )
    return {
        "stub_dir": stub_dir,
        "worker_path": worker_path,
        "items": items,
        "input_path": input_path,
        "output_path": tmp_path / "scores.jsonl",
        "calls_path": tmp_path / "calls.jsonl",
    }


def _run_worker(
    we: dict[str, Any], *extra: str, fresh_calls: bool = False
) -> subprocess.CompletedProcess[str]:
    if fresh_calls and we["calls_path"].exists():
        we["calls_path"].unlink()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(we["stub_dir"])
    env["STUB_ALIGNSCORE_CALLS"] = str(we["calls_path"])
    cmd = [
        sys.executable, str(we["worker_path"]),
        "--input", str(we["input_path"]),
        "--output", str(we["output_path"]),
        "--ckpt", "/fake/AlignScore-large.ckpt",
        "--batch", "2",
        *extra,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# --------------------------------------------------------------------------- #
# Worker contract: schema, batching, config plumbing
# --------------------------------------------------------------------------- #


class TestWorkerContract:
    def test_full_run_schema_and_scores(self, worker_env: dict[str, Any]) -> None:
        proc = _run_worker(worker_env)
        assert proc.returncode == 0, proc.stderr
        rows = _read_jsonl(worker_env["output_path"])
        assert [r["id"] for r in rows] == [f"i{n}" for n in range(5)]
        for row, item in zip(rows, worker_env["items"]):
            assert set(row) == {"id", "score", "wall_ms"}
            assert row["score"] == pytest.approx(_stub_score(item["claim"]))
            assert isinstance(row["wall_ms"], float)
        assert "PROGRESS scored=5/5" in proc.stdout
        assert "WORKER done" in proc.stdout

    def test_batching_splits_at_batch_size(self, worker_env: dict[str, Any]) -> None:
        proc = _run_worker(worker_env)
        assert proc.returncode == 0, proc.stderr
        calls = _read_jsonl(worker_env["calls_path"])
        init = [c for c in calls if c["event"] == "init"]
        score_calls = [c for c in calls if c["event"] == "score"]
        assert len(init) == 1
        # config plumbing: pinned defaults reach the scorer constructor
        assert init[0]["model"] == "roberta-large"
        assert init[0]["batch_size"] == 2
        assert init[0]["evaluation_mode"] == "nli_sp"
        assert init[0]["ckpt_path"] == "/fake/AlignScore-large.ckpt"
        # 5 items at --batch 2 -> batches of 2, 2, 1
        assert [c["n"] for c in score_calls] == [2, 2, 1]

    def test_missing_item_field_fails_closed(self, worker_env: dict[str, Any]) -> None:
        worker_env["input_path"].write_text(
            json.dumps({"id": "x", "context": "ctx"}) + "\n", encoding="utf-8"
        )
        proc = _run_worker(worker_env)
        assert proc.returncode == 2
        assert "lacks id/context/claim" in proc.stderr


# --------------------------------------------------------------------------- #
# Worker contract: resume + truncated-tail healing
# --------------------------------------------------------------------------- #


class TestWorkerResume:
    def test_resume_skips_scored_ids(self, worker_env: dict[str, Any]) -> None:
        first = _run_worker(worker_env, "--max-items", "3")
        assert first.returncode == 0, first.stderr
        assert len(_read_jsonl(worker_env["output_path"])) == 3

        second = _run_worker(worker_env, fresh_calls=True)
        assert second.returncode == 0, second.stderr
        assert "already_scored=3" in second.stdout
        # only the two REMAINING claims were scored on resume
        score_calls = [
            c for c in _read_jsonl(worker_env["calls_path"])
            if c["event"] == "score"
        ]
        remaining = [item["claim"] for item in worker_env["items"][3:]]
        assert [c["claims"] for c in score_calls] == [remaining]
        rows = _read_jsonl(worker_env["output_path"])
        assert [r["id"] for r in rows] == [f"i{n}" for n in range(5)]

    def test_rerun_with_all_scored_is_a_noop(self, worker_env: dict[str, Any]) -> None:
        assert _run_worker(worker_env).returncode == 0
        again = _run_worker(worker_env, fresh_calls=True)
        assert again.returncode == 0
        assert "nothing to score" in again.stdout
        # the scorer was never constructed: no import, no init event
        assert not worker_env["calls_path"].exists()

    def test_delete_half_and_rerun_rescores_only_the_deleted(
        self, worker_env: dict[str, Any]
    ) -> None:
        assert _run_worker(worker_env).returncode == 0
        rows = _read_jsonl(worker_env["output_path"])
        assert len(rows) == 5
        # simulate a lost second half: keep only the first 2 scored rows
        worker_env["output_path"].write_text(
            "".join(json.dumps(r) + "\n" for r in rows[:2]), encoding="utf-8"
        )
        proc = _run_worker(worker_env, fresh_calls=True)
        assert proc.returncode == 0, proc.stderr
        assert "already_scored=2" in proc.stdout
        # exactly the 3 deleted ids were re-scored, nothing else
        score_calls = [
            c for c in _read_jsonl(worker_env["calls_path"])
            if c["event"] == "score"
        ]
        rescored = [claim for c in score_calls for claim in c["claims"]]
        assert rescored == [item["claim"] for item in worker_env["items"][2:]]
        final = _read_jsonl(worker_env["output_path"])
        assert [r["id"] for r in final] == [f"i{n}" for n in range(5)]

    def test_heals_partial_tail_line(self, worker_env: dict[str, Any]) -> None:
        assert _run_worker(worker_env).returncode == 0
        with worker_env["output_path"].open("ab") as fh:
            fh.write(b'{"id": "i9", "sco')  # torn write: no newline
        proc = _run_worker(worker_env, fresh_calls=True)
        assert proc.returncode == 0, proc.stderr
        assert "healed truncated tail" in proc.stdout
        rows = _read_jsonl(worker_env["output_path"])
        assert [r["id"] for r in rows] == [f"i{n}" for n in range(5)]

    def test_heals_corrupt_complete_tail_line_and_rescores_it(
        self, worker_env: dict[str, Any]
    ) -> None:
        assert _run_worker(worker_env).returncode == 0
        # corrupt the LAST row into a complete-but-unparseable line
        lines = worker_env["output_path"].read_text(encoding="utf-8").splitlines()
        lines[-1] = "NOT JSON AT ALL"
        worker_env["output_path"].write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        proc = _run_worker(worker_env, fresh_calls=True)
        assert proc.returncode == 0, proc.stderr
        assert "healed truncated tail" in proc.stdout
        # the dropped id (i4) was re-scored, and ONLY it
        score_calls = [
            c for c in _read_jsonl(worker_env["calls_path"])
            if c["event"] == "score"
        ]
        assert [c["claims"] for c in score_calls] == [
            [worker_env["items"][4]["claim"]]
        ]
        rows = _read_jsonl(worker_env["output_path"])
        assert sorted(r["id"] for r in rows) == [f"i{n}" for n in range(5)]


# --------------------------------------------------------------------------- #
# Manager: score() end-to-end against a faked-ready env (stub worker run)
# --------------------------------------------------------------------------- #


def _fake_ready_env(tmp_path: Path) -> Path:
    """A fake READY env home whose python is the current interpreter."""
    env_home = tmp_path / "ib_env"
    worker_path = write_worker(env_home)
    manifest = {
        "schema_version": 1,
        "ready": True,
        "spec_version": SPEC.spec_version,
        "spec_fingerprint": spec_fingerprint(SPEC),
        "env_python": sys.executable,
        "worker": str(worker_path),
        "verified": {"ckpt": {"path": "/fake/AlignScore-large.ckpt"}},
    }
    (env_home / ib.ENV_MANIFEST_NAME).write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return env_home


class TestManagerScore:
    def test_score_joins_worker_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub_dir = tmp_path / "stub"
        (stub_dir / "alignscore").mkdir(parents=True)
        (stub_dir / "alignscore" / "__init__.py").write_text(
            _STUB_ALIGNSCORE, encoding="utf-8"
        )
        monkeypatch.setenv("PYTHONPATH", str(stub_dir))
        env_home = _fake_ready_env(tmp_path)
        items = [
            {"id": "a", "context": "ctx one", "claim": "claim aa"},
            {"id": "b", "context": "ctx two", "claim": "claim bbb"},
        ]
        scores = ib.score(
            items, env_home=env_home, batch_size=2, stream=False
        )
        assert scores == {
            "a": pytest.approx(_stub_score("claim aa")),
            "b": pytest.approx(_stub_score("claim bbb")),
        }

    def test_score_partial_allowed_only_with_max_items(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub_dir = tmp_path / "stub"
        (stub_dir / "alignscore").mkdir(parents=True)
        (stub_dir / "alignscore" / "__init__.py").write_text(
            _STUB_ALIGNSCORE, encoding="utf-8"
        )
        monkeypatch.setenv("PYTHONPATH", str(stub_dir))
        env_home = _fake_ready_env(tmp_path)
        items = [
            {"id": f"i{n}", "context": "ctx", "claim": f"claim {n}"}
            for n in range(4)
        ]
        partial = ib.score(
            items, env_home=env_home, batch_size=2, max_items=2, stream=False
        )
        assert len(partial) == 2  # partial IS the contract with max_items

    def test_same_ids_different_content_never_reuse_stale_scores(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression (confirmed by repro): §6 tree-mode ids are root-relative
        and thus IDENTICAL across different sealed run roots; a shared default
        work dir let job B silently inherit job A's scores.  The default work
        dir is now content-addressed over the full items."""
        stub_dir = tmp_path / "stub"
        (stub_dir / "alignscore").mkdir(parents=True)
        (stub_dir / "alignscore" / "__init__.py").write_text(
            _STUB_ALIGNSCORE, encoding="utf-8"
        )
        monkeypatch.setenv("PYTHONPATH", str(stub_dir))
        env_home = _fake_ready_env(tmp_path)
        shared_id = "cells/rk1/window_w1/qa_evidence.jsonl::e0::0"
        job_a = [{"id": shared_id, "context": "run A ctx", "claim": "claim aa"}]
        job_b = [
            {"id": shared_id, "context": "run B ctx", "claim": "claim aaaaa"}
        ]
        assert ib.score(job_a, env_home=env_home, stream=False) == {
            shared_id: pytest.approx(_stub_score("claim aa"))
        }
        # job B has the SAME id but different content: must be scored fresh
        assert ib.score(job_b, env_home=env_home, stream=False) == {
            shared_id: pytest.approx(_stub_score("claim aaaaa"))
        }
        assert _stub_score("claim aa") != _stub_score("claim aaaaa")

    def test_identical_items_resume_in_the_same_work_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Content-addressing must preserve the resume story: the same items
        map to the same work dir (smoke-with-max_items then full run)."""
        stub_dir = tmp_path / "stub"
        (stub_dir / "alignscore").mkdir(parents=True)
        (stub_dir / "alignscore" / "__init__.py").write_text(
            _STUB_ALIGNSCORE, encoding="utf-8"
        )
        monkeypatch.setenv("PYTHONPATH", str(stub_dir))
        env_home = _fake_ready_env(tmp_path)
        items = [
            {"id": f"i{n}", "context": "ctx", "claim": f"claim {n}"}
            for n in range(4)
        ]
        partial = ib.score(
            items, env_home=env_home, max_items=2, stream=False
        )
        assert len(partial) == 2
        full = ib.score(items, env_home=env_home, stream=False)
        assert len(full) == 4
        # the first pass's scores were resumed, not thrown away: one work dir
        work_dirs = sorted((env_home / "work").iterdir())
        assert len(work_dirs) == 1
        already = {k: v for k, v in full.items() if k in partial}
        assert already == partial

    def test_worker_failure_raises_typed_with_stderr_tail(
        self, tmp_path: Path
    ) -> None:
        env_home = _fake_ready_env(tmp_path)
        # replace the worker with one that fails loudly on stderr
        (env_home / ib.WORKER_FILE_NAME).write_text(
            "import sys\n"
            "print('boom detail on stderr', file=sys.stderr)\n"
            "sys.exit(3)\n",
            encoding="utf-8",
        )
        with pytest.raises(
            InstrumentBWorkerError, match="exited 3"
        ) as exc_info:
            ib.score(
                [{"id": "a", "context": "c", "claim": "x"}],
                env_home=env_home,
                stream=False,
            )
        assert "boom detail on stderr" in str(exc_info.value)

    def test_score_validates_items(self, tmp_path: Path) -> None:
        env_home = _fake_ready_env(tmp_path)
        with pytest.raises(InstrumentBError, match="empty item sequence"):
            ib.score([], env_home=env_home)
        with pytest.raises(InstrumentBError, match="duplicate item id"):
            ib.score(
                [
                    {"id": "a", "context": "c", "claim": "x"},
                    {"id": "a", "context": "c", "claim": "y"},
                ],
                env_home=env_home,
            )
        with pytest.raises(InstrumentBError, match="non-empty 'context'"):
            ib.score([{"id": "a", "context": " ", "claim": "x"}], env_home=env_home)
        with pytest.raises(InstrumentBError, match="non-empty 'claim'"):
            ib.score([{"id": "a", "context": "c", "claim": ""}], env_home=env_home)


class TestScoreWithProvenance:
    """Task #130 decision (c) (audit H12): the content-addressed cache may be
    reused — but its contribution is DISCLOSED, and ``fresh=True`` forces a
    clean work dir. All through the REAL worker with the stub alignscore."""

    @staticmethod
    def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        stub_dir = tmp_path / "stub"
        (stub_dir / "alignscore").mkdir(parents=True)
        (stub_dir / "alignscore" / "__init__.py").write_text(
            _STUB_ALIGNSCORE, encoding="utf-8"
        )
        monkeypatch.setenv("PYTHONPATH", str(stub_dir))
        return _fake_ready_env(tmp_path)

    @staticmethod
    def _items() -> list[dict[str, str]]:
        return [
            {"id": f"i{n}", "context": "ctx", "claim": f"claim {n}"}
            for n in range(3)
        ]

    def test_first_pass_discloses_no_reuse(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_home = self._setup(tmp_path, monkeypatch)
        scores, prov = ib.score_with_provenance(
            self._items(), env_home=env_home, stream=False
        )
        assert len(scores) == 3
        assert prov.reused is False
        assert prov.forced_fresh is False
        assert prov.n_items == 3
        assert prov.n_cached == 0
        assert prov.n_scored_fresh == 3
        # the digest is the FULL content hash; the work dir is its truncation
        assert len(prov.work_dir_digest) == 64
        assert Path(prov.work_dir).name == prov.work_dir_digest[:16]
        assert prov.to_dict()["reused"] is False  # manifest-ready form

    def test_second_pass_discloses_cache_reuse(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_home = self._setup(tmp_path, monkeypatch)
        _, first = ib.score_with_provenance(
            self._items(), env_home=env_home, stream=False
        )
        scores, second = ib.score_with_provenance(
            self._items(), env_home=env_home, stream=False
        )
        assert len(scores) == 3
        assert second.reused is True
        assert second.n_cached == 3
        assert second.n_scored_fresh == 0
        assert second.work_dir_digest == first.work_dir_digest

    def test_fresh_forces_clean_work_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_home = self._setup(tmp_path, monkeypatch)
        items = self._items()
        _, first = ib.score_with_provenance(
            items, env_home=env_home, stream=False
        )
        scores, prov = ib.score_with_provenance(
            items, env_home=env_home, stream=False, fresh=True
        )
        assert len(scores) == 3
        assert prov.forced_fresh is True
        assert prov.reused is False
        assert prov.n_cached == 0
        assert prov.n_scored_fresh == 3
        # same content-address: the DIR was cleaned, not relocated
        assert prov.work_dir_digest == first.work_dir_digest

    def test_partial_max_items_counts_cached_vs_fresh(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Smoke-then-full: the second call's provenance separates resumed
        scores (n_cached) from newly produced ones (n_scored_fresh)."""
        env_home = self._setup(tmp_path, monkeypatch)
        items = self._items()
        partial, prov1 = ib.score_with_provenance(
            items, env_home=env_home, max_items=2, stream=False
        )
        assert len(partial) == 2
        assert prov1.n_cached == 0
        assert prov1.n_scored_fresh == 2
        full, prov2 = ib.score_with_provenance(
            items, env_home=env_home, stream=False
        )
        assert len(full) == 3
        assert prov2.reused is True
        assert prov2.n_cached == 2
        assert prov2.n_scored_fresh == 1

    def test_score_wrapper_keeps_its_contract(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """score() stays the provenance-free convenience wrapper."""
        env_home = self._setup(tmp_path, monkeypatch)
        scores = ib.score(self._items(), env_home=env_home, stream=False)
        assert set(scores) == {"i0", "i1", "i2"}

    def test_items_digest_is_content_sensitive(self) -> None:
        a = [{"id": "x", "context": "c", "claim": "one"}]
        b = [{"id": "x", "context": "c", "claim": "two"}]
        assert ib._items_digest(a) != ib._items_digest(b)
        assert ib._items_digest(a) == ib._items_digest(list(a))


class TestReadScores:
    def test_ignores_torn_tail(self, tmp_path: Path) -> None:
        path = tmp_path / "scores.jsonl"
        path.write_bytes(
            json.dumps({"id": "a", "score": 0.5}).encode() + b"\n"
            + b'{"id": "b", "sc'
        )
        assert read_scores(path) == {"a": 0.5}

    def test_interior_corruption_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "scores.jsonl"
        path.write_bytes(
            b"GARBAGE\n" + json.dumps({"id": "a", "score": 0.5}).encode() + b"\n"
        )
        with pytest.raises(InstrumentBWorkerError, match="corrupt interior line"):
            read_scores(path)

    def test_missing_file_is_empty(self, tmp_path: Path) -> None:
        assert read_scores(tmp_path / "nope.jsonl") == {}


# --------------------------------------------------------------------------- #
# DOWNLOAD_SNIPPET contract (stub huggingface_hub — no network, no downloads)
# --------------------------------------------------------------------------- #

_STUB_HF_HUB = '''\
"""Stub huggingface_hub mimicking the hub cache layout (contract tests)."""
import os


def _snapshot_dir(repo_id, revision, cache_dir):
    repo_dir = os.path.join(cache_dir, "models--" + repo_id.replace("/", "--"))
    snap = os.path.join(repo_dir, "snapshots", revision)
    os.makedirs(snap, exist_ok=True)
    return snap


def hf_hub_download(repo_id=None, filename=None, revision=None, cache_dir=None):
    snap = _snapshot_dir(repo_id, revision, cache_dir)
    path = os.path.join(snap, filename)
    with open(path, "w") as fh:
        fh.write("stub-ckpt-bytes")
    return path


def snapshot_download(repo_id=None, revision=None, cache_dir=None,
                      allow_patterns=None):
    snap = _snapshot_dir(repo_id, revision, cache_dir)
    with open(os.path.join(snap, "config.json"), "w") as fh:
        fh.write("{}")
    return snap
'''


class TestDownloadSnippet:
    def test_snippet_pins_refs_main_and_reports_paths(
        self, tmp_path: Path
    ) -> None:
        """The snippet must (a) download ckpt+backbone at PINNED revisions,
        (b) write refs/main = the pinned backbone SHA — snapshot_download at
        a commit SHA records no refs, but the worker's offline
        ``from_pretrained('roberta-large')`` resolves "main" through
        refs/<revision> — and (c) emit the resolved paths as JSON."""
        stub_dir = tmp_path / "stub"
        (stub_dir / "huggingface_hub").mkdir(parents=True)
        (stub_dir / "huggingface_hub" / "__init__.py").write_text(
            _STUB_HF_HUB, encoding="utf-8"
        )
        cache_dir = tmp_path / "hf" / "hub"
        paths_out = tmp_path / "paths.json"
        env = dict(os.environ)
        env["PYTHONPATH"] = str(stub_dir)
        spec_arg = json.dumps(
            {
                "model_repo_id": SPEC.model_repo_id,
                "ckpt_file_name": SPEC.ckpt_file_name,
                "model_revision": SPEC.model_revision,
                "backbone_repo_id": SPEC.backbone_repo_id,
                "backbone_revision": SPEC.backbone_revision,
                "cache_dir": str(cache_dir),
            }
        )
        proc = subprocess.run(
            [sys.executable, "-c", ib.DOWNLOAD_SNIPPET, spec_arg, str(paths_out)],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
        paths = json.loads(paths_out.read_text(encoding="utf-8"))
        assert paths["ckpt"].endswith(
            f"snapshots/{SPEC.model_revision}/{SPEC.ckpt_file_name}"
        )
        assert paths["backbone_snapshot"].endswith(
            f"snapshots/{SPEC.backbone_revision}"
        )
        ref_main = (
            cache_dir
            / f"models--{SPEC.backbone_repo_id.replace('/', '--')}"
            / "refs"
            / "main"
        )
        assert ref_main.is_file()
        assert (
            ref_main.read_text(encoding="utf-8") == SPEC.backbone_revision
        )


# --------------------------------------------------------------------------- #
# Spec vs provenance artifact (skips cleanly on a fresh clone)
# --------------------------------------------------------------------------- #


def _provenance_from_spec(spec: AlignScoreEnvSpec) -> dict[str, Any]:
    """A minimal provenance dict CONSISTENT with the spec (test scaffolding)."""
    return {
        "models": {
            "alignscore_large": {
                "hf_repo_id": spec.model_repo_id,
                "revision_commit_sha": spec.model_revision,
                "files": {
                    spec.ckpt_file_name: {
                        "byte_size": spec.ckpt_byte_size,
                        "hub_lfs_sha256": spec.ckpt_sha256,
                        "local_sha256": spec.ckpt_sha256,
                    }
                },
                "backbone_dependency": {
                    "hf_repo_id": spec.backbone_repo_id,
                    "revision_commit_sha": spec.backbone_revision,
                    "file_name": spec.backbone_weights_file,
                    "byte_size": spec.backbone_weights_byte_size,
                    "hub_lfs_sha256": spec.backbone_weights_sha256,
                },
            }
        },
        "code": {
            "alignscore": {
                "github_repo": spec.code_github_repo,
                "commit_sha_at_install": spec.code_commit_sha,
                "key_versions": {
                    "torch": spec.torch_pin,
                    "transformers": spec.transformers_pin,
                    "pytorch_lightning": spec.pytorch_lightning_pin,
                    "spacy": spec.spacy_pin,
                    "python": "3.10.20",
                },
            }
        },
    }


class TestSpecVsProvenance:
    @pytest.mark.skipif(
        not PROVENANCE_PATH.is_file(),
        reason="provenance.json absent (fresh clone: MyDocs is gitignored)",
    )
    def test_embedded_spec_matches_real_provenance(self) -> None:
        provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
        assert_spec_matches_provenance(provenance)  # raises on any drift

    @pytest.mark.skipif(
        not (
            PROVENANCE_PATH.parent / "pip_freeze_alignscore_venv.txt"
        ).is_file(),
        reason="pip freeze artifact absent (fresh clone)",
    )
    def test_embedded_pins_match_real_pip_freeze(self) -> None:
        freeze = (
            PROVENANCE_PATH.parent / "pip_freeze_alignscore_venv.txt"
        ).read_text(encoding="utf-8")
        lines = freeze.splitlines()
        assert f"torch=={SPEC.torch_pin}" in lines
        assert f"transformers=={SPEC.transformers_pin}" in lines
        assert f"pytorch-lightning=={SPEC.pytorch_lightning_pin}" in lines
        assert f"spacy=={SPEC.spacy_pin}" in lines
        assert f"huggingface_hub=={SPEC.huggingface_hub_pin}" in lines
        assert any(
            SPEC.code_commit_sha in line and "alignscore" in line for line in lines
        )
        assert any(SPEC.en_core_web_sm_url in line for line in lines)

    def test_consistent_synthetic_provenance_passes(self) -> None:
        assert_spec_matches_provenance(_provenance_from_spec(SPEC))

    def test_divergent_byte_size_raises_naming_the_field(self) -> None:
        provenance = copy.deepcopy(_provenance_from_spec(SPEC))
        provenance["models"]["alignscore_large"]["files"][SPEC.ckpt_file_name][
            "byte_size"
        ] += 1
        with pytest.raises(InstrumentBSpecMismatchError, match="ckpt_byte_size"):
            assert_spec_matches_provenance(provenance)

    def test_divergent_commit_raises(self) -> None:
        provenance = copy.deepcopy(_provenance_from_spec(SPEC))
        provenance["code"]["alignscore"]["commit_sha_at_install"] = "deadbeef"
        with pytest.raises(InstrumentBSpecMismatchError, match="code_commit_sha"):
            assert_spec_matches_provenance(provenance)

    def test_malformed_provenance_raises(self) -> None:
        with pytest.raises(InstrumentBSpecMismatchError, match="expected structure"):
            assert_spec_matches_provenance({})


# --------------------------------------------------------------------------- #
# Interpreter discovery (typed failure naming both remedies)
# --------------------------------------------------------------------------- #


class TestDiscoverPython:
    def test_prefers_system_python(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            ib.shutil,
            "which",
            lambda name: "/opt/python3.10" if name == "python3.10" else None,
        )
        found = discover_python()
        assert found.kind == "system"
        assert found.command == "/opt/python3.10"

    def test_falls_back_to_uv(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            ib.shutil, "which", lambda name: "/opt/uv" if name == "uv" else None
        )
        found = discover_python()
        assert found.kind == "uv"
        assert found.command == "/opt/uv"

    def test_nothing_found_raises_naming_both_remedies(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ib.shutil, "which", lambda name: None)
        with pytest.raises(InstrumentBInterpreterError) as exc_info:
            discover_python()
        message = str(exc_info.value)
        assert "python3.10" in message  # remedy (a)
        assert "uv" in message  # remedy (b)


# --------------------------------------------------------------------------- #
# Artifact verification (fail closed on any mismatch)
# --------------------------------------------------------------------------- #


class TestVerifyArtifact:
    def test_correct_file_passes(self, tmp_path: Path) -> None:
        blob = b"pinned instrument bytes"
        path = tmp_path / "artifact.bin"
        path.write_bytes(blob)
        verify_artifact(path, len(blob), hashlib.sha256(blob).hexdigest())

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(InstrumentBVerificationError, match="missing"):
            verify_artifact(tmp_path / "gone.bin", 1, "00" * 32)

    def test_wrong_byte_size_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "artifact.bin"
        path.write_bytes(b"hello")
        with pytest.raises(InstrumentBVerificationError, match="byte-size mismatch"):
            verify_artifact(path, 999, hashlib.sha256(b"hello").hexdigest())

    def test_wrong_sha256_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "artifact.bin"
        path.write_bytes(b"hello")
        with pytest.raises(InstrumentBVerificationError, match="sha256 mismatch"):
            verify_artifact(path, 5, "00" * 32)


# --------------------------------------------------------------------------- #
# ensure_env: no-op fast path + the in-repo refusal
# --------------------------------------------------------------------------- #


class TestEnsureEnv:
    def test_ready_env_is_a_noop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_home = tmp_path / "ready_env"
        env_home.mkdir()
        (env_home / ib.ENV_MANIFEST_NAME).write_text(
            json.dumps(
                {"ready": True, "spec_fingerprint": spec_fingerprint(SPEC)}
            ),
            encoding="utf-8",
        )

        def _no_bootstrap(spec: AlignScoreEnvSpec = SPEC) -> None:
            raise AssertionError("bootstrap attempted on a ready env")

        monkeypatch.setattr(ib, "discover_python", _no_bootstrap)
        assert ib.ensure_env(env_home) == env_home.resolve()

    def test_stale_fingerprint_triggers_bootstrap_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_home = tmp_path / "stale_env"
        env_home.mkdir()
        (env_home / ib.ENV_MANIFEST_NAME).write_text(
            json.dumps({"ready": True, "spec_fingerprint": "not-the-fingerprint"}),
            encoding="utf-8",
        )
        # discovery raising proves the bootstrap path was entered (and stops
        # it before any install/download work).
        monkeypatch.setattr(ib.shutil, "which", lambda name: None)
        with pytest.raises(InstrumentBInterpreterError):
            ib.ensure_env(env_home)

    def test_env_home_inside_repo_is_refused(self) -> None:
        inside = ib.REPO_ROOT / "instrument_b_env_should_never_exist"
        with pytest.raises(InstrumentBError, match="INSIDE the repo"):
            ib.ensure_env(inside)
        assert not inside.exists()  # refused BEFORE any directory creation

    def test_default_env_home_respects_env_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ib.ENV_HOME_ENV_VAR, str(tmp_path / "custom"))
        assert ib.default_env_home() == tmp_path / "custom"

    def test_default_env_home_is_outside_the_repo(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(ib.ENV_HOME_ENV_VAR, raising=False)
        home = ib.default_env_home()
        with pytest.raises(ValueError):
            home.relative_to(ib.REPO_ROOT)


# --------------------------------------------------------------------------- #
# apply_tau: boundary + fail-closed cases
# --------------------------------------------------------------------------- #


class TestApplyTau:
    def test_tau_has_no_default(self) -> None:
        # apply_tau keeps its explicit-argument signature even though the
        # registered τ now exists (TAU_REGISTERED, owner-decided 2026-08-05):
        # every CALLER names the τ it applies; the CLI default lives in
        # scripts/4_analysis/score_instrument_b.py, which records
        # tau_source=registered vs override.
        parameter = inspect.signature(apply_tau).parameters["tau"]
        assert parameter.default is inspect.Parameter.empty

    def test_verdicts_and_boundary(self) -> None:
        scores = {"at": 0.5, "above": 0.51, "below": 0.49999}
        verdicts = apply_tau(scores, 0.5)
        # score EXACTLY at tau is grounded (mirrors select_tau's >= rule)
        assert verdicts == {"at": True, "above": True, "below": False}

    def test_missing_score_fails_closed(self) -> None:
        with pytest.raises(InstrumentBTauError, match="missing or non-numeric"):
            apply_tau({"a": 0.9, "b": None}, 0.5)

    def test_nan_score_fails_closed(self) -> None:
        with pytest.raises(InstrumentBTauError, match="non-finite"):
            apply_tau({"a": float("nan")}, 0.5)

    def test_bool_score_fails_closed(self) -> None:
        with pytest.raises(InstrumentBTauError, match="missing or non-numeric"):
            apply_tau({"a": True}, 0.5)

    def test_empty_scores_fail_closed(self) -> None:
        with pytest.raises(InstrumentBTauError, match="empty score mapping"):
            apply_tau({}, 0.5)

    @pytest.mark.parametrize(
        "bad_tau", [None, float("nan"), float("inf"), -0.01, 1.01, "0.5", True]
    )
    def test_invalid_tau_fails_closed(self, bad_tau: Any) -> None:
        with pytest.raises(InstrumentBTauError):
            apply_tau({"a": 0.5}, bad_tau)

    def test_tau_zero_and_one_are_legal_bounds(self) -> None:
        assert apply_tau({"a": 0.0}, 0.0) == {"a": True}
        assert apply_tau({"a": 0.5}, 1.0) == {"a": False}


# --------------------------------------------------------------------------- #
# The registered τ (OWNER-DECIDED 2026-08-05: DECISION.md + PUBLICATION.md
# §8.6(c)) — constants + re-derivation from the calibration artifacts
# --------------------------------------------------------------------------- #


_SELECTION_DIR = PROVENANCE_PATH.parent
_SELECTION_REPORT = _SELECTION_DIR / "selection_report.json"
_RAGTRUTH_ANCHOR = _SELECTION_DIR / "anchors" / "ragtruth_test.jsonl"
_ALIGNSCORE_SCORES = _SELECTION_DIR / "scores" / "alignscore_large.jsonl"


class TestRegisteredTau:
    def test_registered_constants_match_the_decision(self) -> None:
        # DECISION.md §1 verbatim: τ = 0.817024, RAGTruth-test anchor ONLY.
        assert TAU_REGISTERED == 0.817024
        assert TAU_ANCHOR_SCOPE == "ragtruth_test"
        # A registered τ must itself survive apply_tau's validation.
        assert apply_tau({"at": TAU_REGISTERED}, TAU_REGISTERED) == {"at": True}

    @pytest.mark.skipif(
        not (
            _SELECTION_REPORT.is_file()
            and _RAGTRUTH_ANCHOR.is_file()
            and _ALIGNSCORE_SCORES.is_file()
        ),
        reason="selection calibration artifacts absent "
        "(fresh clone: MyDocs is gitignored)",
    )
    def test_registered_tau_matches_selection_calibration(self) -> None:
        """TAU_REGISTERED re-derives from selection_report.json's precision
        floor + the ragtruth_test anchor/score artifacts beside it, via the
        REAL charter rule (select_tau). Skips cleanly when the gitignored
        calibration directory is absent."""
        from src.evaluation.instrument_calibration import (
            AnchorIdentity,
            select_tau,
        )

        report = json.loads(_SELECTION_REPORT.read_text(encoding="utf-8"))
        floor = report["tau_selection"]["precision_floor"]

        labels: dict[str, int] = {}
        for line in _RAGTRUTH_ANCHOR.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                labels[rec["id"]] = int(rec["label"])
        scores: dict[str, float] = {}
        for line in _ALIGNSCORE_SCORES.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec["id"] in labels:
                    scores[rec["id"]] = float(rec["score"])
        assert set(scores) == set(labels)  # every anchor item scored
        assert len(labels) == 2675  # DECISION.md: RAGTruth-test n=2675

        ids = sorted(labels)
        selection = select_tau(
            [scores[i] for i in ids],
            [labels[i] for i in ids],
            AnchorIdentity(
                dataset=TAU_ANCHOR_SCOPE,
                split="test",
                n_items=len(ids),
                fingerprint_sha256="recomputed-in-test",
            ),
            precision_floor=floor,
        )
        assert selection.tau == pytest.approx(TAU_REGISTERED, abs=1e-9)
        # DECISION.md operating point at τ (rounded there to 4 dp).
        assert selection.precision_at_tau == pytest.approx(0.9002, abs=5e-5)
        assert selection.recall_at_tau == pytest.approx(0.5468, abs=5e-5)
