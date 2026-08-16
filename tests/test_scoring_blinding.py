"""Task #130 decision (a) — true label stripping in the offline scoring chain.

Audit H9 (charter §9.8): the scoring layer previously SAW arm identity —
every evidence row embeds "baseline", the §6 evidence path embeds the
row_key (which STARTS with the arm), and instrument-B item ids carried that
path into the worker's on-disk artifacts. The owner chose the strong fix:
arm-bearing identifiers are replaced by deterministic HMAC-SHA256 tokens
BEFORE anything reaches a scoring computation, and the final artifacts are
unblinded at output time through a checked bijective join.

Proved here:

1.  The blinding primitives: deterministic sealed salt (blinding.py's
    sealed-file pattern — self-hashed, tamper-raising), token shape,
    loud-failure unblind.
2.  rescore_quality tree passes: NOTHING that reaches QualityEvaluator
    carries an arm string (spy over every evaluator entry point); outputs
    are unblinded (real labels); the pass seals its token->label map and
    stamps the "blinding" manifest section (#112 prereg citation); tokens
    are deterministic across passes (shared sealed salt).
3.  The MANDATORY equivalence checksums: a blinded pass's score artifacts
    are BITWISE identical to a no-blinding control run (tree mode and the
    legacy CSV writer) — blinding can never change a number.
4.  score_instrument_b tree passes THROUGH THE REAL WORKER (stub alignscore
    injected via PYTHONPATH, exactly the worker-contract test pattern): a
    grep over every artifact the worker world produced (input.jsonl,
    scores.jsonl, worker_stderr.log, the work dirs) finds ZERO arm strings;
    output ids are real; cache provenance discloses reuse across passes
    (decision (c)) with token-stable work-dir digests; --fresh forces clean
    scoring; blinded == control bitwise.
5.  The unblind-join integrity guards fail LOUD (unknown token, non-blinded
    row, non-injective map) — silent mis-assignment of scores to arms is the
    catastrophic failure mode these checks make impossible to ship.
6.  organize_results tolerates the two new legal scoring/ residents — the
    sealed salt file and tombstoned .abandoned-* pass dirs (decision (d)) —
    while an abandoned-named dir WITHOUT its tombstone still fails.

No GPU, no network, no model loads.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = REPO_ROOT / "scripts" / "4_analysis"
for _p in (str(_SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import organize_results as org  # noqa: E402
import rescore_quality as rq  # noqa: E402
import score_instrument_b as sib  # noqa: E402
from src.analysis.stats.ledger import (  # noqa: E402
    hash_artifacts,
    verify_ledger,
    write_ledger,
)
from src.evaluation import instrument_b_runner as ib  # noqa: E402
from src.evaluation.quality import QualityEvaluator  # noqa: E402

RUN_ID = "20260814-1200-a-qwen3-14b"

#: The arm strings the zero-hit greps assert on (pilot names + a charter arm).
ARM_STRINGS: tuple[str, ...] = ("no_cache", "prefix_cache", "rag", "gold-reuse")

#: Arm-bearing §6 row keys used by the fixtures (arm is the FIRST component).
ROW_KEY_A = "no_cache|none|none|single|vllm|qwen3-14b|F1"
ROW_KEY_B = "prefix_cache|rag|none|single|vllm|qwen3-14b|F1"

#: (row_key, evidence "baseline" field) per fixture cell.
_CELLS: tuple[tuple[str, str], ...] = (
    (ROW_KEY_A, "no_cache"),
    (ROW_KEY_B, "prefix_cache"),
)


def _evidence_rows(baseline: str) -> list[dict[str, Any]]:
    """Two scoreable QA rows; text deliberately free of arm strings."""
    return [
        {
            "example_id": f"e{i}",
            "question": "What color is the sky?",
            "used_contexts": ["The sky is blue."],
            "generated_answer": "blue",
            "reference_answer": "blue",
            "baseline": baseline,
            "repeat_index": 0,
        }
        for i in range(2)
    ]


def _write_evidence(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    return path


def _make_arm_tree(root_dir: Path) -> Path:
    """Sealed §6 run root whose row keys and baselines are ARM-BEARING."""
    run = root_dir / "runroot"
    (run / "cells").mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps({"run_id": RUN_ID}), encoding="utf-8"
    )
    for row_key, baseline in _CELLS:
        _write_evidence(
            run / "cells" / row_key / "window_squad_v2-01" / "qa_evidence.jsonl",
            _evidence_rows(baseline),
        )
    sealed = [p for p in sorted(run.rglob("*")) if p.is_file()]
    write_ledger(hash_artifacts(sealed, base_dir=run), run / "ledger.json")
    return run


def _args(**overrides: Any) -> argparse.Namespace:
    base: dict[str, Any] = dict(
        full=False, device="cpu", apply=False, batch_size=None,
        allow_duplicates=False, no_blinding_control=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _fast_evaluator() -> QualityEvaluator:
    return QualityEvaluator(
        use_nli=False, use_embeddings=False, use_bertscore=False,
        use_rouge=False, use_lettucedetect=False, device="cpu",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _assert_no_arm_strings(text: str, where: str) -> None:
    for arm in ARM_STRINGS:
        assert arm not in text, f"arm string {arm!r} leaked into {where}"


# ---------------------------------------------------------------------------
# 1. Blinding primitives
# ---------------------------------------------------------------------------


class TestBlindingPrimitives:
    def test_token_is_deterministic_and_opaque(self) -> None:
        salt = b"s" * 32
        a = rq.LabelBlinder(salt)
        b = rq.LabelBlinder(salt)
        tok = a.token("no_cache")
        assert tok == b.token("no_cache")  # same salt -> same token
        assert re.fullmatch(r"blind-[0-9a-f]{16}", tok)
        assert "no_cache" not in tok
        # a different salt yields a different token for the same label
        other = rq.LabelBlinder(b"t" * 32)
        assert other.token("no_cache") != tok
        # distinct labels get distinct tokens
        assert a.token("prefix_cache") != tok

    def test_salt_seal_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "scoring" / "blinding_salt.json"
        first = rq.load_or_create_scoring_salt(path)
        assert path.is_file()
        again = rq.load_or_create_scoring_salt(path)
        assert again.salt == first.salt  # created ONCE, reloaded thereafter
        assert again.sha256 == first.sha256
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert doc["salt_sha256"] == first.sha256

    def test_tampered_salt_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "blinding_salt.json"
        rq.load_or_create_scoring_salt(path)
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["salt_hex"] = "ab" * 32  # altered after sealing
        path.write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(rq.ScoringBlindingError, match="self-hash mismatch"):
            rq.load_or_create_scoring_salt(path)

    def test_corrupt_salt_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "blinding_salt.json"
        path.write_text("NOT JSON", encoding="utf-8")
        with pytest.raises(rq.ScoringBlindingError, match="not valid JSON"):
            rq.load_or_create_scoring_salt(path)

    def test_unknown_token_unblinds_loudly(self) -> None:
        blinder = rq.LabelBlinder(b"s" * 32)
        blinder.token("no_cache")
        with pytest.raises(rq.ScoringBlindingError, match="unknown blinding token"):
            blinder.unblind("blind-0000000000000000")

    def test_unblind_score_rows_rejects_non_blinded_row(self) -> None:
        """A row that bypassed the stripping layer must not slip through the
        join as if it were legitimately blinded."""
        blinder = rq.LabelBlinder(b"s" * 32)
        with pytest.raises(rq.ScoringBlindingError, match="not a blinding token"):
            rq.unblind_score_rows([{"baseline": "no_cache"}], blinder)

    def test_instrument_b_join_guards_fail_loud(self) -> None:
        # unknown scored id
        with pytest.raises(rq.ScoringBlindingError, match="no entry"):
            sib._unblind_ib_rows([{"id": "blind-x::e0::0"}], {})
        # two blind ids collapsing onto one input row
        with pytest.raises(rq.ScoringBlindingError, match="same input row"):
            sib._unblind_ib_rows(
                [{"id": "b1"}, {"id": "b2"}], {"b1": "real", "b2": "real"}
            )
        # non-injective map refused before scoring even starts
        with pytest.raises(rq.ScoringBlindingError, match="non-injective"):
            sib._check_join_bijective({"b1": "real", "b2": "real"}, 2)
        with pytest.raises(rq.ScoringBlindingError, match="not one-to-one"):
            sib._check_join_bijective({"b1": "r1"}, 2)


# ---------------------------------------------------------------------------
# 2. rescore_quality: the evaluator never sees an arm string
# ---------------------------------------------------------------------------


class _LeakSpyEvaluator:
    """Delegating evaluator recording EVERY argument that crosses the scoring
    boundary — the assertion surface for the zero-arm-leak guarantee."""

    def __init__(self, inner: QualityEvaluator) -> None:
        self._inner = inner
        self.seen: list[Any] = []

    def batch_evaluate(self, *args: Any, **kwargs: Any) -> Any:
        self.seen.append((args, kwargs))
        return self._inner.batch_evaluate(*args, **kwargs)

    def evaluate_faithfulness(self, *args: Any, **kwargs: Any) -> Any:
        self.seen.append((args, kwargs))
        return self._inner.evaluate_faithfulness(*args, **kwargs)

    def evaluate_hallucination(self, *args: Any, **kwargs: Any) -> Any:
        self.seen.append((args, kwargs))
        return self._inner.evaluate_hallucination(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class TestRescoreTreeBlinding:
    def test_pass_blinds_inputs_seals_map_and_unblinds_outputs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = _make_arm_tree(tmp_path)
        spy = _LeakSpyEvaluator(_fast_evaluator())
        monkeypatch.setattr(rq, "_build_evaluator", lambda args: spy)

        assert rq.run_scoring_tree(run, "s01-blind", _args()) == 0

        # (i) NOTHING that crossed the evaluator boundary carries an arm.
        assert spy.seen  # the spy actually saw the scoring calls
        _assert_no_arm_strings(repr(spy.seen), "QualityEvaluator inputs")

        # (ii) Final artifacts are UNBLINDED: real labels, one row per input.
        sdir = run / "scoring" / "s01-blind"
        baselines: set[str] = set()
        n_rows = 0
        for scores_path in sorted(sdir.glob("cells/*/window_*/qa_scores.jsonl")):
            for row in _read_jsonl(scores_path):
                baselines.add(row["baseline"])
                n_rows += 1
        assert baselines == {"no_cache", "prefix_cache"}
        assert n_rows == 4  # 2 cells x 2 rows: bijective join, nothing lost

        # (iii) The sealed per-run-root salt + the pass's sealed map exist,
        # the map covers exactly the labels seen, and the pass ledger seals it.
        assert (run / "scoring" / rq.SALT_FILE_NAME).is_file()
        map_doc = json.loads(
            (sdir / rq.BLINDING_MAP_NAME).read_text(encoding="utf-8")
        )
        assert set(map_doc["mapping"].values()) == {"no_cache", "prefix_cache"}
        assert map_doc["map_sha256"] == rq.canonical_sha256(map_doc["mapping"])
        assert verify_ledger(sdir / "ledger.json", sdir) == []

        # (iv) The manifest "blinding" section (#112 prereg citation).
        man = json.loads(
            (sdir / rq.SCORING_MANIFEST_NAME).read_text(encoding="utf-8")
        )
        blinding = man["blinding"]
        assert blinding["mode"] == rq.BLINDING_MODE_STRIPPED
        assert blinding["salt_file"] == "scoring/blinding_salt.json"
        assert blinding["map_file"] == rq.BLINDING_MAP_NAME
        assert blinding["map_sha256"] == map_doc["map_sha256"]
        assert blinding["n_labels"] == 2
        assert blinding["n_rows_unblinded"] == man["n_rows"] == 4
        assert blinding["join_checksum"] == rq.blinding_join_checksum(
            map_doc["mapping"], man["n_rows"]
        )
        salt_doc = json.loads(
            (run / "scoring" / rq.SALT_FILE_NAME).read_text(encoding="utf-8")
        )
        assert blinding["salt_sha256"] == salt_doc["salt_sha256"]

    def test_tokens_deterministic_across_passes(self, tmp_path: Path) -> None:
        """Decision (a)+(c): the per-run-root sealed salt makes tokens — and
        therefore instrument-B's content-addressed cache key — identical
        across passes; a per-pass salt would silently defeat the cache."""
        run = _make_arm_tree(tmp_path)
        assert rq.run_scoring_tree(run, "s01-a", _args()) == 0
        assert rq.run_scoring_tree(run, "s02-b", _args()) == 0
        map_a = json.loads(
            (run / "scoring" / "s01-a" / rq.BLINDING_MAP_NAME).read_text()
        )["mapping"]
        map_b = json.loads(
            (run / "scoring" / "s02-b" / rq.BLINDING_MAP_NAME).read_text()
        )["mapping"]
        assert map_a == map_b

    def test_blinded_pass_equals_control_bitwise(self, tmp_path: Path) -> None:
        """MANDATORY checksum: blinding must never change a number — the
        blinded pass's score artifacts equal a no-blinding control BITWISE."""
        run = _make_arm_tree(tmp_path)
        assert rq.run_scoring_tree(run, "s01-blind", _args()) == 0
        assert rq.run_scoring_tree(
            run, "s02-control", _args(no_blinding_control=True)
        ) == 0

        blind_dir = run / "scoring" / "s01-blind"
        ctrl_dir = run / "scoring" / "s02-control"
        compared = 0
        for name in ("qa_scores.jsonl", "quality.json"):
            for blind_path in sorted(blind_dir.glob(f"cells/*/window_*/{name}")):
                ctrl_path = ctrl_dir / blind_path.relative_to(blind_dir)
                assert ctrl_path.read_bytes() == blind_path.read_bytes()
                compared += 1
        assert compared == 4  # 2 cells x 2 artifact kinds

        # The control run is recorded LOUDLY, never passable as registered.
        ctrl_man = json.loads(
            (ctrl_dir / rq.SCORING_MANIFEST_NAME).read_text(encoding="utf-8")
        )
        assert ctrl_man["blinding"]["mode"] == rq.BLINDING_MODE_CONTROL
        assert not (ctrl_dir / rq.BLINDING_MAP_NAME).exists()

    def test_legacy_blinded_csv_equals_control_bitwise(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Legacy mode blinds with an ephemeral salt; the unblinded CSV must
        equal a control produced by the SAME writer with blinding off."""
        root = tmp_path / "pilotish"
        # arm-bearing trial path AND rows without a baseline field, so the
        # arm-bearing directory-name fallback is exercised under blinding.
        rows = _evidence_rows("no_cache")
        del rows[0]["baseline"]
        ev = _write_evidence(
            root / "no_cache" / "trial_0" / "qa_evidence.jsonl", rows
        )
        monkeypatch.setattr(
            sys, "argv", ["rescore_quality.py", "--run-root", str(root)]
        )
        assert rq.main() == 0
        blinded_csv = (ev.parent / "results_rescored.csv").read_bytes()

        control_rows, n_dup = rq._score_evidence_file(
            ev, _fast_evaluator(), blinder=None
        )
        assert n_dup == 0
        control_path = tmp_path / "control.csv"
        rq._write_rescored_csv(control_path, control_rows)
        assert blinded_csv == control_path.read_bytes()
        assert b"no_cache" in blinded_csv  # unblinded: real labels in output

    def test_tampered_salt_fails_the_next_pass(self, tmp_path: Path) -> None:
        run = _make_arm_tree(tmp_path)
        assert rq.run_scoring_tree(run, "s01-a", _args()) == 0
        salt_path = run / "scoring" / rq.SALT_FILE_NAME
        doc = json.loads(salt_path.read_text(encoding="utf-8"))
        doc["salt_hex"] = "ab" * 32
        salt_path.write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(rq.ScoringBlindingError, match="self-hash mismatch"):
            rq.run_scoring_tree(run, "s02-b", _args())


# ---------------------------------------------------------------------------
# 3. score_instrument_b through the REAL worker: zero arm strings on disk,
#    cache disclosure (decision (c)), --fresh, blinded == control
# ---------------------------------------------------------------------------

_STUB_ALIGNSCORE = '''\
"""Stub alignscore for blinding tests: deterministic, dependency-free."""


class AlignScore:
    def __init__(self, model=None, batch_size=None, device=None,
                 ckpt_path=None, evaluation_mode=None):
        pass

    def score(self, contexts=None, claims=None):
        return [round((len(c) % 7) / 10.0, 3) for c in claims]
'''


@pytest.fixture()
def ib_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A READY fake env home driving the REAL worker under the CURRENT python
    with the stub alignscore on PYTHONPATH (worker-contract test pattern)."""
    stub_dir = tmp_path / "stub"
    (stub_dir / "alignscore").mkdir(parents=True)
    (stub_dir / "alignscore" / "__init__.py").write_text(
        _STUB_ALIGNSCORE, encoding="utf-8"
    )
    monkeypatch.setenv("PYTHONPATH", str(stub_dir))
    env_home = tmp_path / "ib_env"
    worker_path = ib.write_worker(env_home)
    manifest = {
        "schema_version": 1,
        "ready": True,
        "spec_version": ib.SPEC.spec_version,
        "spec_fingerprint": ib.spec_fingerprint(ib.SPEC),
        "env_python": sys.executable,
        "worker": str(worker_path),
        "verified": {"ckpt": {"path": "/fake/AlignScore-large.ckpt"}},
    }
    (env_home / ib.ENV_MANIFEST_NAME).write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return env_home


def _ib_tree_pass(
    run: Path, env_home: Path, scoring_id: str, *extra: str
) -> dict[str, Any]:
    rc = sib.main([
        "--evidence", str(run),
        "--scoring-run-id", scoring_id,
        "--env-home", str(env_home),
        "--tau", "0.5",
        *extra,
    ])
    assert rc == 0
    return json.loads(
        (run / "scoring" / scoring_id / sib.SCORING_MANIFEST_NAME).read_text(
            encoding="utf-8"
        )
    )


class TestInstrumentBBlindingEndToEnd:
    def test_worker_world_carries_zero_arm_strings(
        self, tmp_path: Path, ib_env: Path
    ) -> None:
        """THE grep test: after a real pass, no artifact the worker produced
        or consumed (input.jsonl, scores.jsonl, stderr log, anything under
        the env home) contains an arm string."""
        run = _make_arm_tree(tmp_path)
        manifest = _ib_tree_pass(run, ib_env, "s01-ib")

        scanned = 0
        for path in sorted(ib_env.rglob("*")):
            if path.is_file():
                _assert_no_arm_strings(
                    path.read_text(encoding="utf-8", errors="replace"),
                    str(path),
                )
                scanned += 1
        assert scanned >= 3  # worker script + input.jsonl + scores.jsonl

        # Output ids are REAL §6 paths (unblinded), one per scoreable row.
        rows: list[dict[str, Any]] = []
        for scores_path in sorted(
            (run / "scoring" / "s01-ib").glob("cells/*/window_*/instrument_b_scores.jsonl")
        ):
            rows.extend(_read_jsonl(scores_path))
        assert len(rows) == 4
        ids = {row["id"] for row in rows}
        assert len(ids) == 4  # bijective join: no collapsed rows
        for row_key, _ in _CELLS:
            expected_prefix = f"cells/{row_key}/window_squad_v2-01/qa_evidence.jsonl::"
            assert sum(1 for i in ids if i.startswith(expected_prefix)) == 2
        for row in rows:
            assert "grounded_b" in row

        # Manifest: blinding section + honest first-pass cache provenance.
        blinding = manifest["blinding"]
        assert blinding["mode"] == rq.BLINDING_MODE_STRIPPED
        assert blinding["cache_note"]  # the pre-#130 continuity break is disclosed
        assert set(
            json.loads(
                (run / "scoring" / "s01-ib" / sib.BLINDING_MAP_NAME).read_text()
            )["mapping"].values()
        ) == {ROW_KEY_A, ROW_KEY_B}
        cache = manifest["cache_provenance"]
        assert cache["reused"] is False
        assert cache["forced_fresh"] is False
        assert cache["n_items"] == 4
        assert cache["n_cached"] == 0
        assert cache["n_scored_fresh"] == 4
        # The pass ledger seals the map alongside the scores.
        sdir = run / "scoring" / "s01-ib"
        assert verify_ledger(sdir / "ledger.json", sdir) == []

    def test_cache_shared_across_passes_is_disclosed(
        self, tmp_path: Path, ib_env: Path
    ) -> None:
        """Decision (c) + the (a) determinism requirement: pass 2 hits pass
        1's content-addressed work dir (token-stable ids) and the manifest
        SAYS so — reuse is never silently claimed as independence."""
        run = _make_arm_tree(tmp_path)
        man1 = _ib_tree_pass(run, ib_env, "s01-ib")
        man2 = _ib_tree_pass(run, ib_env, "s02-ib")
        assert (
            man2["cache_provenance"]["work_dir_digest"]
            == man1["cache_provenance"]["work_dir_digest"]
        )
        cache2 = man2["cache_provenance"]
        assert cache2["reused"] is True
        assert cache2["n_cached"] == 4
        assert cache2["n_scored_fresh"] == 0
        # and the reused scores joined to identical outputs
        rows1 = sorted(
            (run / "scoring" / "s01-ib").glob("cells/*/window_*/instrument_b_scores.jsonl")
        )
        for p1 in rows1:
            p2 = run / "scoring" / "s02-ib" / p1.relative_to(run / "scoring" / "s01-ib")
            assert p2.read_bytes() == p1.read_bytes()

    def test_fresh_forces_clean_work_dir(
        self, tmp_path: Path, ib_env: Path
    ) -> None:
        run = _make_arm_tree(tmp_path)
        _ib_tree_pass(run, ib_env, "s01-ib")
        man3 = _ib_tree_pass(run, ib_env, "s03-fresh", "--fresh")
        cache = man3["cache_provenance"]
        assert cache["forced_fresh"] is True
        assert cache["reused"] is False
        assert cache["n_cached"] == 0
        assert cache["n_scored_fresh"] == 4

    def test_blinded_pass_equals_control_bitwise(
        self, tmp_path: Path, ib_env: Path
    ) -> None:
        run = _make_arm_tree(tmp_path)
        _ib_tree_pass(run, ib_env, "s01-blind")
        man_ctrl = _ib_tree_pass(
            run, ib_env, "s02-control", "--no-blinding-control"
        )
        assert man_ctrl["blinding"]["mode"] == rq.BLINDING_MODE_CONTROL
        blind_dir = run / "scoring" / "s01-blind"
        ctrl_dir = run / "scoring" / "s02-control"
        compared = 0
        for bp in sorted(blind_dir.glob("cells/*/window_*/instrument_b_scores.jsonl")):
            cp = ctrl_dir / bp.relative_to(blind_dir)
            assert cp.read_bytes() == bp.read_bytes()
            compared += 1
        assert compared == 2


# ---------------------------------------------------------------------------
# 4. Flat mode: salt sealed beside --out, worker never sees the path
# ---------------------------------------------------------------------------


class TestFlatModeBlinding:
    def test_flat_pass_blinds_paths_and_unblinds_output(
        self, tmp_path: Path, ib_env: Path
    ) -> None:
        ev = _write_evidence(
            tmp_path / "no_cache" / "trial_0" / "qa_evidence.jsonl",
            _evidence_rows("no_cache"),
        )
        out = tmp_path / "outdir" / "ib_scores.jsonl"
        rc = sib.main([
            "--evidence", str(ev),
            "--out", str(out),
            "--tau", "0.5",
            "--env-home", str(ib_env),
        ])
        assert rc == 0

        # Worker world: zero arm strings (the evidence PATH embeds no_cache).
        for path in sorted(ib_env.rglob("*")):
            if path.is_file():
                _assert_no_arm_strings(
                    path.read_text(encoding="utf-8", errors="replace"),
                    str(path),
                )

        # Output rows carry the REAL ids again.
        rows = _read_jsonl(out)
        assert [r["id"] for r in rows] == [
            f"{ev.as_posix()}::e0::0",
            f"{ev.as_posix()}::e1::0",
        ]

        # Salt sealed beside the out file; sidecar discloses everything.
        salt_path = Path(str(out) + ".blinding_salt.json")
        assert salt_path.is_file()
        sidecar = json.loads(
            Path(str(out) + ".provenance.json").read_text(encoding="utf-8")
        )
        assert sidecar["blinding"]["mode"] == rq.BLINDING_MODE_STRIPPED
        assert sidecar["blinding"]["salt_file"] == str(salt_path)
        assert set(sidecar["blinding"]["mapping"].values()) == {ev.as_posix()}
        assert sidecar["cache_provenance"]["n_items"] == 2

    def test_flat_rerun_same_out_resumes_from_cache(
        self, tmp_path: Path, ib_env: Path
    ) -> None:
        """The sealed beside-the-out salt keeps flat-mode tokens — and the
        worker's checkpointed resume — deterministic across invocations."""
        ev = _write_evidence(
            tmp_path / "no_cache" / "qa_evidence.jsonl", _evidence_rows("no_cache")
        )
        out = tmp_path / "ib_scores.jsonl"
        argv = [
            "--evidence", str(ev), "--out", str(out),
            "--tau", "0.5", "--env-home", str(ib_env),
        ]
        assert sib.main(argv) == 0
        first = json.loads(
            Path(str(out) + ".provenance.json").read_text(encoding="utf-8")
        )["cache_provenance"]
        assert first["reused"] is False
        assert sib.main(argv) == 0
        second = json.loads(
            Path(str(out) + ".provenance.json").read_text(encoding="utf-8")
        )["cache_provenance"]
        assert second["reused"] is True
        assert second["n_cached"] == 2
        assert second["work_dir_digest"] == first["work_dir_digest"]


# ---------------------------------------------------------------------------
# 5. organize_results tolerance for the new legal scoring/ residents
# ---------------------------------------------------------------------------


class TestOrganizerTolerance:
    def test_salt_and_tombstoned_abandoned_pass_are_tolerated(
        self, tmp_path: Path
    ) -> None:
        run = _make_arm_tree(tmp_path)
        assert rq.run_scoring_tree(run, "s01-keep", _args()) == 0  # writes salt
        # a crashed pass (no ledger), then abandoned per decision (d)
        crashed = run / "scoring" / "s02-crash"
        (crashed / "cells").mkdir(parents=True)
        (crashed / "scoring_manifest.json").write_text("{}", encoding="utf-8")
        rq.abandon_scoring_pass(run, "s02-crash", reason="synthetic crash")

        summary = org.validate_scoring_tree(run)
        assert any("blinding salt" in line for line in summary)
        assert any("ABANDONED" in line for line in summary)
        assert any("`s01-keep`" in line for line in summary)

    def test_abandoned_dir_without_tombstone_still_fails(
        self, tmp_path: Path
    ) -> None:
        run = _make_arm_tree(tmp_path)
        assert rq.run_scoring_tree(run, "s01-keep", _args()) == 0
        fake = run / "scoring" / "s09-x.abandoned-20260101T000000Z"
        fake.mkdir()
        with pytest.raises(org.LayoutError, match="without its ABANDONED.json"):
            org.validate_scoring_tree(run)
