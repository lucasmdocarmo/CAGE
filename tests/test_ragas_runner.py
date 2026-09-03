"""W4.12 — §8.6(d) RAGAS head-to-head RUNNER (build only; no judge execution).

WHAT is pinned and WHY:

- **D11 gate refusal**: --judge-model, --judge-revision and
  --completion-max-tokens have NO defaults; absence is a NAMED refusal citing
  the D11 owner decision (PUBLICATION.md §8.6(d): "Judge model + token budget
  decided at D11"). Pinned so a future convenience default can never quietly
  un-gate the owner decision.
- **--dry-run = the owner's cost preview**: exact planned judge-call count +
  disclosed token ESTIMATE, zero judge calls (an injected judge that raises on
  any call proves it), zero files written.
- **Table math**: the three-way instrument/pair/unanimity arithmetic is
  verified on fixture scores with a deterministic test-only judge injected
  through the explicit ``judge=`` seam — the production OpenAI-compatible
  judge is NEVER constructed in tests (its env refusal is exercised, which
  fails closed BEFORE any network object exists).
- **Overwrite refusal + --force**: an existing artifact refuses BEFORE any
  judge call (a refusal must never follow a paid pass) and --force replaces
  deliberately.
- **Absence stays absence**: null instrument scores ride as None and sit out
  that instrument's pairs (counted); a claim-free judge verdict is
  ``judge-no-claims``, never a fabricated 1.0; unparseable judge replies are
  named error rows, never coerced scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = REPO_ROOT / "scripts" / "4_analysis"
for _p in (str(_SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ragas_head_to_head as rhh  # noqa: E402
from src.evaluation.instrument_b_runner import TAU_REGISTERED  # noqa: E402


# ---------------------------------------------------------------------------
# Test-only judges (the explicit injection seam; deterministic, no network)
# ---------------------------------------------------------------------------


class SequenceJudge:
    """Deterministic test-only judge: replays canned raw replies in call
    order (rows are judged in input order, so the mapping is exact)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.replies.pop(0)

    def describe(self):
        return {"kind": "test-sequence-judge (deterministic, test-only)"}


class MustNotCallJudge:
    """Injected wherever the runner must NOT spend judge budget."""

    def complete(self, prompt: str) -> str:  # pragma: no cover - the point
        raise AssertionError("judge was called — this path must spend nothing")

    def describe(self):
        return {"kind": "must-not-call (test-only)"}


# ---------------------------------------------------------------------------
# Fixture rows
# ---------------------------------------------------------------------------

_CTX = "Paris is the capital of France. The Seine crosses it."


def _row(example_id, grounding, faithfulness, *, ctx=_CTX, ans="Paris.", **extra):
    rec = {
        "example_id": example_id,
        "repeat_index": 0,
        "grounding_score": grounding,
        "faithfulness": faithfulness,
        "used_contexts": [ctx] if ctx else [],
        "generated_answer": ans,
    }
    rec.update(extra)
    return rec


def _write_rows(path: Path, rows) -> Path:
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    return path


@pytest.fixture()
def rows_file(tmp_path: Path) -> Path:
    # 5 judgeable rows in input order (q1..q4, q6) + 1 unjudgeable (q5:
    # empty served context). q4 carries a null LettuceDetect score (honest
    # absence). q6 exists to prove --limit slices deterministically.
    return _write_rows(
        tmp_path / "qa_scored.jsonl",
        [
            _row("q1", 0.9, 0.9),
            _row("q2", 0.9, 0.2),
            _row("q3", 0.2, 0.9),
            _row("q4", None, 0.9),
            _row("q5", 0.5, 0.5, ctx=None),
            _row("q6", 0.8, 0.8),
        ],
    )


def _argv(rows: Path, out_dir: Path, *extra: str) -> list[str]:
    return [
        "--rows", str(rows),
        "--out-dir", str(out_dir),
        "--judge-model", "judgezilla-9b-instruct",
        "--judge-revision", "rev-abc123",
        "--completion-max-tokens", "64",
        "--tau-a", "0.5",
        "--tau-judge", "0.5",
        *extra,
    ]


def _drop(argv: list[str], flag: str) -> list[str]:
    i = argv.index(flag)
    return argv[:i] + argv[i + 2:]


# ---------------------------------------------------------------------------
# D11 gate + threshold refusals (NAMED, exit 2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flag", ["--judge-model", "--judge-revision", "--completion-max-tokens"]
)
def test_refuses_without_d11_pin(rows_file, tmp_path, capsys, flag):
    argv = _drop(_argv(rows_file, tmp_path / "out"), flag)
    rc = rhh.main(argv, judge=MustNotCallJudge())
    err = capsys.readouterr().err
    assert rc == 2
    assert "D11" in err and flag in err
    assert not (tmp_path / "out").exists()


def test_d11_refusal_fires_in_dry_run_too(rows_file, tmp_path, capsys):
    argv = _drop(_argv(rows_file, tmp_path / "out", "--dry-run"), "--judge-model")
    rc = rhh.main(argv)
    assert rc == 2
    assert "D11" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["--tau-a", "--tau-judge"])
def test_refuses_without_explicit_threshold(rows_file, tmp_path, capsys, flag):
    argv = _drop(_argv(rows_file, tmp_path / "out"), flag)
    rc = rhh.main(argv, judge=MustNotCallJudge())
    err = capsys.readouterr().err
    assert rc == 2
    assert flag in err


def test_tau_b_defaults_to_registered_and_explicit_is_override():
    ns = argparse.Namespace(tau_a=0.5, tau_b=None, tau_judge=0.5)
    resolved = rhh._resolve_thresholds(ns)
    assert resolved["alignscore"]["value"] == TAU_REGISTERED
    assert resolved["alignscore"]["source"] == "registered"
    ns = argparse.Namespace(tau_a=0.5, tau_b=TAU_REGISTERED, tau_judge=0.5)
    # Repeating the registered value is still recorded as CHOSEN.
    assert rhh._resolve_thresholds(ns)["alignscore"]["source"] == "override"


def test_refuses_rows_missing_instrument_columns(tmp_path, capsys):
    rows = _write_rows(
        tmp_path / "unscored.jsonl",
        [{"example_id": "q1", "used_contexts": [_CTX], "generated_answer": "x",
          "grounding_score": 0.5}],  # no 'faithfulness' column at all
    )
    rc = rhh.main(_argv(rows, tmp_path / "out"), judge=MustNotCallJudge())
    err = capsys.readouterr().err
    assert rc == 2
    assert "faithfulness" in err and "not a scored run" in err


def test_refuses_duplicate_row_ids(tmp_path, capsys):
    rows = _write_rows(
        tmp_path / "dup.jsonl", [_row("q1", 0.5, 0.5), _row("q1", 0.6, 0.6)]
    )
    rc = rhh.main(_argv(rows, tmp_path / "out"), judge=MustNotCallJudge())
    assert rc == 2
    assert "duplicate row id" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# --dry-run: the owner's cost preview (validates, prints, spends nothing)
# ---------------------------------------------------------------------------


def test_dry_run_counts_and_estimate_without_calling_anything(
    rows_file, tmp_path, capsys
):
    out_dir = tmp_path / "out"
    rc = rhh.main(
        _argv(rows_file, out_dir, "--limit", "3", "--dry-run"),
        judge=MustNotCallJudge(),  # raises on ANY call — proving zero spend
    )
    out = capsys.readouterr().out
    assert rc == 0
    # Exact planned call count: 5 judgeable rows, limit 3.
    assert "planned_judge_calls=3" in out
    assert "rows_total=6" in out and "unjudgeable=1" in out
    # The estimate is reproducible from the module's own disclosed method.
    expected_prompt_est = sum(
        rhh.estimate_prompt_tokens(rhh.render_prompt(_CTX, "Paris."))
        for _ in range(3)
    )
    assert f"prompt_tokens_est={expected_prompt_est}" in out
    assert "completion_tokens_est=192" in out  # 3 calls x 64-token D11 cap
    assert f"total_tokens_est={expected_prompt_est + 192}" in out
    assert "ESTIMATE" in out
    assert f"template_sha256={rhh.prompt_template_sha256()}" in out
    # Writes nothing: the preview leaves no artifact behind.
    assert not out_dir.exists()


# ---------------------------------------------------------------------------
# Table math on fixture scores with the injected test judge
# ---------------------------------------------------------------------------


@pytest.fixture()
def head_to_head_doc(rows_file, tmp_path):
    out_dir = tmp_path / "out"
    judge = SequenceJudge([
        '{"claims_total": 4, "claims_supported": 4}',   # q1 -> 1.0
        '{"claims_total": 4, "claims_supported": 1}',   # q2 -> 0.25
        '```json\n{"claims_total": 2, "claims_supported": 2}\n```',  # q3 -> 1.0
        "not json at all",                              # q4 -> named error
    ])
    rc = rhh.main(
        _argv(rows_file, out_dir, "--tau-b", "0.5", "--limit", "4"),
        judge=judge,
    )
    assert rc == 0
    assert len(judge.replies) == 0  # exactly 4 calls, in input order
    doc = json.loads((out_dir / rhh.ARTIFACT_JSON_NAME).read_text())
    return doc, judge, out_dir


def test_judge_call_accounting_and_absence(head_to_head_doc):
    doc, judge, _ = head_to_head_doc
    assert doc["judge_calls"] == {
        "n_calls": 4, "n_scored": 3, "n_no_claims": 0, "n_errors": 1
    }
    per_row = {r["id"]: r for r in doc["per_row"]}
    assert set(per_row) == {"q1::0", "q2::0", "q3::0", "q4::0"}
    assert per_row["q1::0"]["judge_faithfulness"] == pytest.approx(1.0)
    assert per_row["q2::0"]["judge_faithfulness"] == pytest.approx(0.25)
    assert per_row["q3::0"]["judge_faithfulness"] == pytest.approx(1.0)
    # Unparseable reply -> None with a NAMED note, never a coerced score.
    assert per_row["q4::0"]["judge_faithfulness"] is None
    assert "judge-reply-unparseable" in per_row["q4::0"]["judge_note"]
    # Null LettuceDetect score rode through as honest absence.
    assert per_row["q4::0"]["grounding_score"] is None
    # The judge saw the served context and answer verbatim.
    assert _CTX in judge.prompts[0] and "Paris." in judge.prompts[0]


def test_instrument_summaries(head_to_head_doc):
    doc, _, _ = head_to_head_doc
    inst = doc["instruments"]
    assert inst["lettucedetect"]["n_present"] == 3  # q4 is null
    assert inst["lettucedetect"]["mean_score"] == pytest.approx(2.0 / 3)
    assert inst["lettucedetect"]["verdict_positive_rate"] == pytest.approx(2 / 3)
    assert inst["alignscore"]["n_present"] == 4
    assert inst["alignscore"]["mean_score"] == pytest.approx(0.725)
    assert inst["alignscore"]["verdict_positive_rate"] == pytest.approx(0.75)
    assert inst["judge"]["n_present"] == 3  # q4 judge-errored
    assert inst["judge"]["mean_score"] == pytest.approx(0.75)
    assert inst["judge"]["verdict_positive_rate"] == pytest.approx(2 / 3)


def test_pairwise_deltas_agreement_and_disagreement_examples(head_to_head_doc):
    doc, _, _ = head_to_head_doc
    ab = doc["pairs"]["lettucedetect__vs__alignscore"]
    assert ab["n_pairs"] == 3
    assert ab["mean_delta"] == pytest.approx(0.0)
    assert ab["mean_abs_delta"] == pytest.approx(1.4 / 3)
    assert ab["n_agree"] == 1 and ab["n_disagree"] == 2
    assert ab["agreement_rate"] == pytest.approx(1 / 3)
    assert ab["disagreement_example_ids"] == ["q2::0", "q3::0"]

    aj = doc["pairs"]["lettucedetect__vs__judge"]
    assert aj["n_pairs"] == 3
    assert aj["mean_delta"] == pytest.approx(-0.25 / 3)
    assert aj["agreement_rate"] == pytest.approx(1 / 3)
    assert aj["disagreement_example_ids"] == ["q2::0", "q3::0"]

    bj = doc["pairs"]["alignscore__vs__judge"]
    assert bj["n_pairs"] == 3  # q4 has alignscore but no judge score
    assert bj["mean_delta"] == pytest.approx(-0.25 / 3)
    assert bj["mean_abs_delta"] == pytest.approx(0.25 / 3)
    assert bj["n_disagree"] == 0
    assert bj["agreement_rate"] == pytest.approx(1.0)


def test_three_way_unanimity(head_to_head_doc):
    doc, _, _ = head_to_head_doc
    assert doc["three_way"] == {
        "n_all_three_present": 3,
        "n_all_three_agree": 1,  # only q1 is (T, T, T)
        "unanimous_rate": pytest.approx(1 / 3),
    }


def test_provenance_pins_recorded(head_to_head_doc):
    doc, _, out_dir = head_to_head_doc
    judge = doc["judge"]
    assert judge["model"] == "judgezilla-9b-instruct"
    assert judge["revision"] == "rev-abc123"
    assert judge["temperature"] == 0.0
    assert judge["completion_max_tokens"] == 64
    assert judge["prompt_template_sha256"] == hashlib.sha256(
        rhh.PROMPT_TEMPLATE.encode("utf-8")
    ).hexdigest()
    assert judge["provider"]["kind"].startswith("test-sequence-judge")
    assert doc["thresholds"]["alignscore"]["source"] == "override"
    assert doc["thresholds"]["lettucedetect"]["source"] == "cli"
    assert doc["mode_stamp"] == rhh.MODE_STAMP
    assert doc["inputs"]["n_selected"] == 4
    assert doc["inputs"]["n_missing_lettucedetect"] == 1
    # Human table rendered beside the machine artifact.
    md = (out_dir / rhh.ARTIFACT_MD_NAME).read_text(encoding="utf-8")
    assert "judgezilla-9b-instruct" in md and "rev-abc123" in md
    assert "lettucedetect vs alignscore" in md


# ---------------------------------------------------------------------------
# Overwrite refusal + --force (checked BEFORE any judge call)
# ---------------------------------------------------------------------------


def test_overwrite_refusal_precedes_judge_spend_and_force_replaces(
    head_to_head_doc, rows_file, capsys
):
    doc, _, out_dir = head_to_head_doc
    # Re-run onto the existing artifact: MustNotCallJudge proves the refusal
    # happens before a single judge call is made.
    rc = rhh.main(
        _argv(rows_file, out_dir, "--tau-b", "0.5", "--limit", "4"),
        judge=MustNotCallJudge(),
    )
    err = capsys.readouterr().err
    assert rc == 2
    assert rhh.ARTIFACT_JSON_NAME in err and "--force" in err
    # --force replaces deliberately (fresh deterministic judge).
    rc = rhh.main(
        _argv(rows_file, out_dir, "--tau-b", "0.5", "--limit", "1", "--force"),
        judge=SequenceJudge(['{"claims_total": 1, "claims_supported": 0}']),
    )
    assert rc == 0
    replaced = json.loads((out_dir / rhh.ARTIFACT_JSON_NAME).read_text())
    assert replaced["inputs"]["n_selected"] == 1
    assert replaced["per_row"][0]["judge_faithfulness"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Production judge seam fails closed without ever touching the network
# ---------------------------------------------------------------------------


def test_real_judge_env_refusal_is_named(rows_file, tmp_path, capsys, monkeypatch):
    monkeypatch.delenv(rhh.ENV_JUDGE_API_BASE, raising=False)
    rc = rhh.main(_argv(rows_file, tmp_path / "out"))  # no injected judge
    err = capsys.readouterr().err
    assert rc == 2
    assert rhh.ENV_JUDGE_API_BASE in err
    assert not (tmp_path / "out" / rhh.ARTIFACT_JSON_NAME).exists()


# ---------------------------------------------------------------------------
# Judge-reply parsing (strict, fail-closed)
# ---------------------------------------------------------------------------


def test_parse_judge_reply_strictness():
    assert rhh.parse_judge_reply('{"claims_total": 3, "claims_supported": 2}') == (3, 2)
    assert rhh.parse_judge_reply(
        '```json\n{"claims_total": 1, "claims_supported": 1}\n```'
    ) == (1, 1)
    with pytest.raises(rhh.JudgeReplyError):
        rhh.parse_judge_reply("[1, 2]")  # JSON but not an object
    with pytest.raises(rhh.JudgeReplyError):
        rhh.parse_judge_reply('{"claims_total": true, "claims_supported": 1}')
    with pytest.raises(rhh.JudgeReplyError):  # supported > total
        rhh.parse_judge_reply('{"claims_total": 1, "claims_supported": 2}')
    with pytest.raises(rhh.JudgeReplyError):
        rhh.parse_judge_reply("no json here")


def test_no_claims_verdict_is_absence_not_perfect_score(rows_file, tmp_path):
    out_dir = tmp_path / "out"
    rc = rhh.main(
        _argv(rows_file, out_dir, "--limit", "2"),
        judge=SequenceJudge([
            '{"claims_total": 0, "claims_supported": 0}',  # claim-free answer
            '{"claims_total": 2, "claims_supported": 1}',
        ]),
    )
    assert rc == 0
    doc = json.loads((out_dir / rhh.ARTIFACT_JSON_NAME).read_text())
    per_row = {r["id"]: r for r in doc["per_row"]}
    assert per_row["q1::0"]["judge_faithfulness"] is None
    assert per_row["q1::0"]["judge_note"] == "judge-no-claims"
    assert doc["judge_calls"]["n_no_claims"] == 1
    assert doc["judge_calls"]["n_scored"] == 1
