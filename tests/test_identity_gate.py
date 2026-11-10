"""scripts/checks/identity_gate.py (S2 block C-1, 2026-11-01, charter 5.2):
the T=0 token-identity gate on the Qwen3-Next hybrid cache.

Three subcommands, each pinned here against fakes (no GPU, no network, no
Hub): ``prompts`` renders the first N questions of one manifest trial in the
corpus-reuse shape with the pinned tokenizer and writes their token ids;
``run`` sends those ids to one server configuration for K passes (one request
in flight, token ids in and out, vLLM /v1/completions or SGLang /generate
against an in-process stub) and records every generation; ``verdict`` compares
the recorded ids (A: vLLM warm vs vLLM cache-off, B: vLLM cold vs warm decide;
C, D, E vs SGLang are recorded), asserts the telemetry that makes the test
honest, prints ONE marker line IDENTITY_GATE PASS|FAIL|ERROR and exits 0 on
PASS or FAIL (both are results) and 1 on ERROR (neither).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import CAGExample  # noqa: E402
from src.utils.prompting import format_qa_messages  # noqa: E402

SCRIPT = REPO_ROOT / "scripts" / "checks" / "identity_gate.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ig = _load("cage_identity_gate", SCRIPT)

MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
REV = "9c7f2fbe84465e40164a94cc16cd30b6999b0cc7"
REQUEST = {"temperature": 0.0, "seed": 20260902, "max_tokens": 32, "stop": None}
# the fake tokenizer renders one id per character: block zero puts q1 past two
# 528-token quanta (1,149 tokens: the warm floor applies), block one leaves q2
# under them (689 tokens: exempt from the floor and recorded, review 2026-11-01
# HIGH 1); both sit past one quantum, so the stub's warm counts clear 528
BLOCKS = ("block zero paragraph. " * 40, "block one paragraph. " * 20)
EXAMPLES = [CAGExample(id=f"q{i}", question=f"What is {i}?", context=["c"], answer=str(i), metadata={}) for i in (1, 2, 3, 4)]


class _FakeTokenizer:
    """Renders chat messages to ids deterministically: a 3-id header, one id
    per character of the joined contents, a 2-id generation prompt."""

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is True and add_generation_prompt is True and enable_thinking is False
        text = "\n".join(m["content"] for m in messages)
        return [1, 2, 3] + [ord(ch) % 251 + 10 for ch in text] + [4, 5]


def _render(question: str, block: str) -> List[int]:
    return _FakeTokenizer().apply_chat_template(
        format_qa_messages(question, [block]), tokenize=True, add_generation_prompt=True, enable_thinking=False)


def _manifest(tmp_path: Path, *, revision: str = REV, name: str = "manifest.json") -> Path:
    doc = {
        "manifest_version": 5, "dataset": "musique", "split": "validation", "seed": 42,
        "num_queries": 3, "num_trials": 1, "block_budget": 2640,
        "blocks": [{"block_id": i, "text": t, "token_count": len(t.split()), "n_paragraphs": 1} for i, t in enumerate(BLOCKS)],
        "question_to_block": {"q1": 0, "q2": 1, "q3": 0},
        "trials": {"1": ["q2", "q1", "q3"]},
        "overlap": {}, "trunc_rungs": {}, "warmup_reserve": {"ids": []},
        "tokenizer": {"id": MODEL, "revision": revision},
    }
    p = tmp_path / name
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


@pytest.fixture
def fakes(monkeypatch):
    calls: Dict[str, Any] = {}

    def load_examples(dataset, split, seed):
        calls["examples"] = (dataset, split, seed)
        return list(EXAMPLES)

    def load_tokenizer(model, revision):
        calls["tokenizer"] = (model, revision)
        return _FakeTokenizer()

    monkeypatch.setattr(ig, "load_examples", load_examples)
    monkeypatch.setattr(ig, "load_tokenizer", load_tokenizer)
    return calls


def _prompts(tmp_path: Path, *, count: int = 2, manifest: Optional[Path] = None, dataset: str = "musique",
             revision: str = REV, extra: tuple = (), name: str = "prompts.json") -> tuple:
    out = tmp_path / name
    argv = ["prompts", "--manifest", str(manifest or _manifest(tmp_path)), "--dataset", dataset, "--trial", "1",
            "--count", str(count), "--model", MODEL, "--revision", revision, "--session", "cd-act1",
            "--out", str(out), *extra]
    return ig.main(argv), out


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------


def test_prompts_follow_the_trial_order_in_the_corpus_reuse_shape(tmp_path, fakes, capsys):
    code, out = _prompts(tmp_path)
    assert code == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert fakes["examples"] == ("musique", "validation", 42) and fakes["tokenizer"] == (MODEL, REV)
    assert [p["id"] for p in doc["prompts"]] == ["q2", "q1"]
    assert [p["block_id"] for p in doc["prompts"]] == [1, 0]
    # the ids are the campaign's request: format_qa_messages(question, [block text])
    # rendered by the pinned tokenizer with the generation prompt and no thinking
    assert doc["prompts"][0]["token_ids"] == _render("What is 2?", BLOCKS[1])
    assert doc["prompts"][1]["token_ids"] == _render("What is 1?", BLOCKS[0])
    assert doc["prompts"][0]["n_tokens"] == len(doc["prompts"][0]["token_ids"])
    assert (doc["count"], doc["dataset"], doc["trial"], doc["shape"]) == (2, "musique", 1, "corpus-reuse")
    assert (doc["session"], doc["model"], doc["revision"], doc["tokenizer_revision"]) == ("cd-act1", MODEL, REV, REV)
    manifest = tmp_path / "manifest.json"
    assert doc["manifest"] == str(manifest)
    assert doc["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert doc["sha256"] == ig.prompts_sha256(doc["prompts"]) and len(doc["sha256"]) == 64
    assert doc["max_prompt_tokens"] == max(len(p["token_ids"]) for p in doc["prompts"])
    assert doc["request"] == REQUEST and doc["prompt_token_limit"] == 4096 - 32
    assert "IDENTITY_GATE_PROMPTS_OK count=2" in capsys.readouterr().out


def test_prompts_are_byte_stable_across_two_builds(tmp_path, fakes):
    _, a = _prompts(tmp_path, name="a.json")
    _, b = _prompts(tmp_path, name="b.json")
    assert a.read_bytes() == b.read_bytes()


def test_prompts_refuse_a_count_outside_the_trial_pool(tmp_path, fakes, capsys):
    for count in (0, 4):
        code, out = _prompts(tmp_path, count=count)
        assert code == 1 and not out.exists(), count
        err = capsys.readouterr().err
        assert "IDENTITY_GATE_PROMPTS_REFUSED" in err and "count" in err


def test_prompts_refuse_a_manifest_packed_with_another_tokenizer_revision(tmp_path, fakes, capsys):
    code, out = _prompts(tmp_path, manifest=_manifest(tmp_path, revision="deadbeef"))
    assert code == 1 and not out.exists()
    err = capsys.readouterr().err
    assert "tokenizer" in err and "deadbeef" in err and REV in err


def test_prompts_refuse_a_dataset_that_is_not_the_manifests(tmp_path, fakes, capsys):
    code, out = _prompts(tmp_path, dataset="qasper")
    assert code == 1 and not out.exists()
    assert "qasper" in capsys.readouterr().err


def test_prompts_refuse_a_prompt_over_the_served_limit_before_any_request(tmp_path, fakes, capsys):
    code, out = _prompts(tmp_path, extra=("--prompt-token-limit", "100"))
    assert code == 1 and not out.exists()
    err = capsys.readouterr().err
    assert "q2" in err and "100" in err


# ---------------------------------------------------------------------------
# run: an in-process stub in the vLLM and SGLang shapes
# ---------------------------------------------------------------------------


def _same(i: int, ids: List[int], k: int) -> List[int]:
    """The same generation on every engine and pass: the PASS fixture."""
    return [1000 + i, 2000, 3000]


def _warm(k: int, ids: List[int]) -> Optional[int]:
    """vLLM's cached count on a warm pass: whole 16-token blocks, never the last token."""
    return (len(ids) - 1) // 16 * 16 if k >= 2 else None


class _Stub:
    """One engine HTTP server on 127.0.0.1: GET /version or /server_info for
    the version, POST /v1/completions (vLLM) or /generate (SGLang) answered
    from ``outputs(prompt_index, ids, pass_index)`` and ``cached(pass_index, ids)``."""

    def __init__(self, shape: str, *, outputs: Callable = _same, cached: Callable = _warm, echo_ok: bool = True,
                 empty: bool = False, status: int = 200, version: str = "0.19.1") -> None:
        self.shape, self.outputs, self.cached = shape, outputs, cached
        self.echo_ok, self.empty, self.status, self.version = echo_ok, empty, status, version
        self.journal: List[Dict[str, Any]] = []
        self.seen: Dict[tuple, int] = {}
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code: int, body: Any) -> None:
                raw = json.dumps(body).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                stub.journal.append({"method": "GET", "path": self.path})
                self._send(200, {"version": stub.version, "enable_cache_report": True})

            def do_POST(self):
                n = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(n).decode("utf-8"))
                stub.journal.append({"method": "POST", "path": self.path, "payload": payload})
                if stub.status != 200:
                    self._send(stub.status, {"error": "boom " * 100})
                    return
                ids = payload["prompt"] if stub.shape == "vllm" else payload["input_ids"]
                key = tuple(ids)
                k = stub.seen.get(key, 0) + 1
                stub.seen[key] = k
                idx = list(stub.seen).index(key)
                out = [] if stub.empty else stub.outputs(idx, ids, k)
                c = stub.cached(k, ids)
                text = "ans " + " ".join(str(t) for t in out)
                if stub.shape == "vllm":
                    body = {"id": "cmpl-1", "object": "text_completion", "model": payload["model"],
                            "choices": [{"index": 0, "text": text, "finish_reason": "length", "token_ids": out,
                                         "prompt_token_ids": ids if stub.echo_ok else ids[:-1]}],
                            "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out),
                                      "total_tokens": len(ids) + len(out),
                                      "prompt_tokens_details": {"cached_tokens": c} if c else None}}
                else:
                    body = {"text": text, "output_ids": out,
                            "meta_info": {"id": "r1", "finish_reason": {"type": "length", "length": len(out)},
                                          "prompt_tokens": len(ids), "completion_tokens": len(out), "cached_tokens": c or 0}}
                self._send(200, body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def __enter__(self) -> "_Stub":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _run(tmp_path: Path, stub: _Stub, *, engine: str, phase: str, passes: int, prompts: Path) -> tuple:
    out = tmp_path / f"{phase}.json"
    argv = ["run", "--engine", engine, "--phase", phase, "--api-base", stub.url, "--model", MODEL,
            "--prompts", str(prompts), "--passes", str(passes), "--out", str(out)]
    return ig.main(argv), out


def test_run_sends_token_ids_at_t0_prompt_by_prompt_and_records_every_pass_in_the_vllm_shape(tmp_path, fakes):
    _, prompts = _prompts(tmp_path)
    pdoc = json.loads(prompts.read_text(encoding="utf-8"))
    with _Stub("vllm") as stub:
        code, out = _run(tmp_path, stub, engine="vllm", phase="vllm_on", passes=2, prompts=prompts)
    assert code == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert (doc["phase"], doc["engine"], doc["api_base"], doc["model"], doc["passes"]) == ("vllm_on", "vllm", stub.url, MODEL, 2)
    assert doc["engine_version"] == "0.19.1" and doc["prompts_sha256"] == pdoc["sha256"] and doc["request"] == REQUEST
    assert doc["started_at"] <= doc["finished_at"] and "error" not in doc
    # the version once, then every prompt's passes back to back (the warm pass
    # follows its own cold pass, whatever the pool holds)
    assert [r["path"] for r in stub.journal] == ["/version", "/v1/completions", "/v1/completions", "/v1/completions", "/v1/completions"]
    ids = [p["token_ids"] for p in pdoc["prompts"]]
    bodies = [r["payload"] for r in stub.journal[1:]]
    assert [b["prompt"] for b in bodies] == [ids[0], ids[0], ids[1], ids[1]]
    assert bodies[0] == {"model": MODEL, "prompt": ids[0], "max_tokens": 32, "temperature": 0.0, "seed": 20260902,
                         "return_token_ids": True}
    res = doc["results"]
    assert len(res) == 2 and [len(r) for r in res] == [2, 2], "results[pass][prompt]"
    r = res[1][0]
    assert (r["prompt_index"], r["prompt_id"], r["pass"]) == (0, "q2", 2)
    assert r["token_ids"] == [1000, 2000, 3000] and r["text"] == "ans 1000 2000 3000" and r["finish_reason"] == "length"
    assert r["cached_tokens"] == _warm(2, ids[0]) and r["prompt_tokens"] == len(ids[0]) and r["latency_s"] > 0
    assert res[0][0]["cached_tokens"] is None, "vLLM omits prompt_tokens_details on a cold request: recorded as None"
    assert res[0][1]["prompt_id"] == "q1" and res[0][1]["token_ids"] == [1001, 2000, 3000]


def test_run_in_the_sglang_shape_sends_input_ids_and_reads_output_ids(tmp_path, fakes):
    _, prompts = _prompts(tmp_path)
    pdoc = json.loads(prompts.read_text(encoding="utf-8"))
    with _Stub("sglang", version="0.5.10.post1", cached=lambda k, ids: 0 if k == 1 else len(ids) - 7) as stub:
        code, out = _run(tmp_path, stub, engine="sglang", phase="sglang", passes=2, prompts=prompts)
    assert code == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["engine"] == "sglang" and doc["engine_version"] == "0.5.10.post1"
    assert [r["path"] for r in stub.journal] == ["/server_info"] + ["/generate"] * 4
    ids = [p["token_ids"] for p in pdoc["prompts"]]
    assert stub.journal[1]["payload"] == {"input_ids": ids[0], "sampling_params": {"temperature": 0.0, "max_new_tokens": 32, "sampling_seed": 20260902}}
    r = doc["results"][1][1]
    assert r["token_ids"] == [1001, 2000, 3000] and r["finish_reason"] == "length" and r["cached_tokens"] == len(ids[1]) - 7
    assert doc["results"][0][0]["cached_tokens"] == 0


def test_run_refuses_an_echo_mismatch_and_writes_the_error_into_the_phase_file(tmp_path, fakes, capsys):
    _, prompts = _prompts(tmp_path)
    with _Stub("vllm", echo_ok=False) as stub:
        code, out = _run(tmp_path, stub, engine="vllm", phase="vllm_on", passes=2, prompts=prompts)
    assert code == 1
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["phase"] == "vllm_on" and "echo" in doc["error"] and "q2" in doc["error"]
    assert len([r for r in stub.journal if r["method"] == "POST"]) == 1, "no further request after the first error"
    assert "IDENTITY_GATE_RUN_ERROR" in capsys.readouterr().err


def test_run_refuses_an_empty_generation(tmp_path, fakes):
    _, prompts = _prompts(tmp_path)
    with _Stub("sglang", empty=True) as stub:
        code, out = _run(tmp_path, stub, engine="sglang", phase="sglang", passes=1, prompts=prompts)
    assert code == 1
    assert "empty" in json.loads(out.read_text(encoding="utf-8"))["error"]


def test_run_refuses_a_transport_error_without_a_retry(tmp_path, fakes):
    _, prompts = _prompts(tmp_path)
    with _Stub("vllm", status=500) as stub:
        code, out = _run(tmp_path, stub, engine="vllm", phase="vllm_off", passes=1, prompts=prompts)
    assert code == 1
    err = json.loads(out.read_text(encoding="utf-8"))["error"]
    assert "500" in err and "boom" in err and len(err) < 400, "the body's first 200 characters, not the whole body"
    assert len([r for r in stub.journal if r["method"] == "POST"]) == 1


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------


def _phase_doc(pdoc: Dict[str, Any], phase: str, engine: str, passes: int, *, outputs: Callable = _same,
               cached: Callable = _warm, version: str = "0.19.1", sha: Optional[str] = None) -> Dict[str, Any]:
    """A phase file in the shape ``run`` writes."""
    results = []
    for k in range(1, passes + 1):
        row = []
        for i, p in enumerate(pdoc["prompts"]):
            out = outputs(i, p["token_ids"], k)
            row.append({"prompt_index": i, "prompt_id": p["id"], "pass": k, "token_ids": out,
                        "text": "ans " + " ".join(str(t) for t in out), "finish_reason": "length",
                        "cached_tokens": cached(k, p["token_ids"]), "prompt_tokens": len(p["token_ids"]),
                        "latency_s": 0.01})
        results.append(row)
    return {"phase": phase, "engine": engine, "api_base": "http://stub", "model": MODEL, "passes": passes,
            "prompts_sha256": sha or pdoc["sha256"], "prompts_path": "prompts.json", "engine_version": version,
            "request": REQUEST, "started_at": f"2026-11-01T00:0{passes}:00+00:00", "finished_at": "2026-11-01T00:09:00+00:00",
            "results": results}


#: phase -> (engine, passes, the stub's keyword arguments): the three phase files every verdict test starts from
_PHASE_SPEC = {"vllm_on": ("vllm", 2, {}), "vllm_off": ("vllm", 1, {"cached": lambda k, ids: None}),
               "sglang": ("sglang", 2, {"version": "0.5.10.post1", "cached": lambda k, ids: 0 if k == 1 else len(ids) - 7})}


def _phases(tmp_path: Path, count: int = 2, **tweak: Any) -> tuple:
    """prompts.json plus the three phase files under tmp_path/phases; ``tweak``
    overrides one phase's keyword arguments (vllm_on=..., vllm_off=..., sglang=...)."""
    d = tmp_path / "phases"
    d.mkdir(exist_ok=True)
    code, prompts = _prompts(tmp_path, count=count, name="phases/prompts.json")
    assert code == 0
    pdoc = json.loads(prompts.read_text(encoding="utf-8"))
    for phase, (engine, passes, kw) in _PHASE_SPEC.items():
        if tweak.get(phase) == "absent":
            continue
        kw = {**kw, **tweak.get(phase, {})}
        (d / f"{phase}.json").write_text(json.dumps(_phase_doc(pdoc, phase, engine, passes, **kw)), encoding="utf-8")
    return d, pdoc


def _verdict(tmp_path: Path, d: Path, *extra: str) -> tuple:
    out = tmp_path / "identity_gate.json"
    code = ig.main(["verdict", "--phases", str(d), "--out", str(out), *extra])
    return code, json.loads(out.read_text(encoding="utf-8"))


def test_verdict_pass_when_both_deciding_comparisons_are_identical(tmp_path, fakes, capsys):
    d, pdoc = _phases(tmp_path)
    code, v = _verdict(tmp_path, d)
    assert code == 0 and capsys.readouterr().out.rstrip().splitlines()[-1] == "IDENTITY_GATE PASS"
    assert v["schema"] == "cage.identity_gate/1" and v["verdict"] == "PASS" and v["error"] is None
    assert (v["session"], v["model"], v["revision"], v["tokenizer_revision"]) == ("cd-act1", MODEL, REV, REV)
    assert v["manifest"] == pdoc["manifest"] and v["manifest_sha256"] == pdoc["manifest_sha256"]
    assert v["prompts"] == {"count": 2, "dataset": "musique", "trial": 1, "shape": "corpus-reuse",
                            "sha256": pdoc["sha256"], "max_prompt_tokens": pdoc["max_prompt_tokens"]}
    assert v["request"] == REQUEST
    assert v["phases"] == {"vllm_on": {"engine": "vllm", "passes": 2, "engine_version": "0.19.1"},
                           "vllm_off": {"engine": "vllm", "passes": 1, "engine_version": "0.19.1"},
                           "sglang": {"engine": "sglang", "passes": 2, "engine_version": "0.5.10.post1"},
                           # ADR-0195: the extra-buffer probe phase is recorded absent unless --sglang-xb asks for it
                           "sglang_xb": "absent"}
    n = [len(p["token_ids"]) for p in pdoc["prompts"]]
    assert n[0] < 1057 <= n[1], "the fixture renders q2 under two quanta and q1 past them"
    warm = sorted(_warm(2, [0] * k) for k in n)
    assert v["telemetry"] == {
        "vllm_on_pass1_cached_tokens": {"min": 0, "median": 0, "max": 0}, "vllm_on_pass1_hits": 0,
        "vllm_on_pass2_cached_tokens": {"min": warm[0], "median": (warm[0] + warm[1]) / 2, "max": warm[1]},
        "vllm_on_pass2_floor_exempt": [{"prompt_index": 0, "prompt_id": "q2", "prompt_tokens": n[0],
                                        "cached_tokens": _warm(2, [0] * n[0])}],
        # the block measured from the warm counts: the stub caches whole 16-token blocks
        "vllm_on_pass2_block_tokens": 16,
        "vllm_off_cached_tokens_max": 0,
        "sglang_pass1_hits": 0,
        "sglang_pass2_cached_tokens": {"min": min(n) - 7, "median": (n[0] + n[1]) / 2 - 7, "max": max(n) - 7},
        # ADR-0195: the probe phase's telemetry holds None when the phase was not started
        "sglang_xb_pass1_hits": None, "sglang_xb_pass2_cached_tokens": None,
    }
    assert v["limitations"] == [
        "the 528-token warm floor is not applied to 1 of 2 prompts under 1057 tokens (one aligned quantum may exceed "
        f"their length, review 2026-11-01 HIGH 1): q2 {n[0]} tokens, {_warm(2, [0] * n[0])} cached on pass 2"]
    pairs = {"A": ("vllm_on/2", "vllm_off/1"), "B": ("vllm_on/1", "vllm_on/2"), "C": ("vllm_off/1", "sglang/1"),
             "D": ("vllm_on/2", "sglang/1"), "E": ("sglang/1", "sglang/2")}
    for key, (left, right) in pairs.items():
        assert v["comparisons"][key] == {"decides": key in ("A", "B"), "left": left, "right": right,
                                         "compared": 2, "identical": 2, "first_divergence": None}, key
    assert v["deciding"] == ["A", "B"] and v["label_on_pass"] == "experimental-align-mode"
    assert v["started_at"] == "2026-11-01T00:01:00+00:00" and v["finished_at"] > v["started_at"]


def test_verdict_fail_records_the_first_divergence_of_a_deciding_comparison(tmp_path, fakes, capsys):
    def off(i, ids, k):
        return [1000 + i, 2000, 3000] if i == 0 else [1001, 2001, 3000]
    d, _ = _phases(tmp_path, vllm_off={"outputs": off})
    code, v = _verdict(tmp_path, d)
    assert code == 0 and capsys.readouterr().out.rstrip().splitlines()[-1] == "IDENTITY_GATE FAIL"
    assert v["verdict"] == "FAIL" and v["error"] is None and v["label_on_pass"] is None
    a = v["comparisons"]["A"]
    assert (a["compared"], a["identical"]) == (2, 1)
    assert a["first_divergence"] == {"prompt_index": 1, "prompt_id": "q1", "position": 1, "left_id": 2000, "right_id": 2001,
                                     "left_text": "ans 1001 2000 3000", "right_text": "ans 1001 2001 3000"}
    assert v["comparisons"]["B"]["identical"] == 2
    # C (vLLM off vs SGLang) diverges the same way, recorded, not deciding
    assert v["comparisons"]["C"]["identical"] == 1 and v["comparisons"]["C"]["decides"] is False


def test_verdict_fail_on_a_shorter_generation_points_at_the_missing_position(tmp_path, fakes):
    d, _ = _phases(tmp_path, vllm_on={"outputs": lambda i, ids, k: [1000 + i, 2000, 3000] if k == 1 else [1000 + i, 2000]})
    code, v = _verdict(tmp_path, d)
    assert code == 0 and v["verdict"] == "FAIL"
    b = v["comparisons"]["B"]
    assert b["first_divergence"]["position"] == 2 and b["first_divergence"]["left_id"] == 3000 and b["first_divergence"]["right_id"] is None


def test_verdict_passes_when_only_the_recorded_sglang_comparisons_diverge(tmp_path, fakes, capsys):
    d, _ = _phases(tmp_path, sglang={"outputs": lambda i, ids, k: [1000 + i, 2000, 3001]})
    code, v = _verdict(tmp_path, d)
    assert code == 0 and v["verdict"] == "PASS" and capsys.readouterr().out.rstrip().splitlines()[-1] == "IDENTITY_GATE PASS"
    assert v["comparisons"]["A"]["identical"] == 2 and v["comparisons"]["B"]["identical"] == 2
    assert v["comparisons"]["C"]["identical"] == 0 and v["comparisons"]["D"]["identical"] == 0
    assert v["comparisons"]["C"]["first_divergence"]["position"] == 2
    assert v["comparisons"]["E"]["identical"] == 2, "SGLang agrees with itself across passes"


def test_verdict_with_no_sglang_records_the_engine_absent_and_decides_on_vllm_alone(tmp_path, fakes):
    d, _ = _phases(tmp_path, sglang="absent")
    code, v = _verdict(tmp_path, d, "--no-sglang")
    assert code == 0 and v["verdict"] == "PASS"
    assert v["phases"]["sglang"] == "absent" and v["telemetry"]["sglang_pass2_cached_tokens"] is None
    assert v["telemetry"]["sglang_pass1_hits"] is None
    for key in "CDE":
        assert v["comparisons"][key] == {"decides": False, "left": None, "right": None, "compared": 0,
                                         "identical": None, "first_divergence": None, "skipped": "sglang absent"}, key


def test_verdict_exempts_prompts_under_two_quanta_from_the_warm_floor_review_high1(tmp_path, fakes):
    """The manifest's blocks run from 527 tokens and the aligned quantum at TP 4
    is unmeasured: on a prompt under two quanta a correct cache may report one
    block, or none. The floor applies past 1,057 tokens; the shorter prompts are
    recorded in the telemetry and named in the limitations."""
    short = lambda k, ids: (320 if len(ids) < 1057 else _warm(k, ids)) if k == 2 else None  # noqa: E731
    d, pdoc = _phases(tmp_path, vllm_on={"cached": short})
    n = [len(p["token_ids"]) for p in pdoc["prompts"]]
    assert n[0] < 1057 <= n[1]
    code, v = _verdict(tmp_path, d)
    assert code == 0 and v["verdict"] == "PASS" and v["error"] is None
    assert v["telemetry"]["vllm_on_pass2_floor_exempt"] == [
        {"prompt_index": 0, "prompt_id": "q2", "prompt_tokens": n[0], "cached_tokens": 320}]
    assert v["telemetry"]["vllm_on_pass2_cached_tokens"]["min"] == 320
    assert v["limitations"] == [
        "the 528-token warm floor is not applied to 1 of 2 prompts under 1057 tokens (one aligned quantum may exceed "
        f"their length, review 2026-11-01 HIGH 1): q2 {n[0]} tokens, 320 cached on pass 2"]
    # a quantum longer than the prompt caches nothing: still exempt, still recorded
    none = lambda k, ids: (0 if len(ids) < 1057 else _warm(k, ids)) if k == 2 else None  # noqa: E731
    d, _ = _phases(tmp_path, vllm_on={"cached": none})
    code, v = _verdict(tmp_path, d)
    assert code == 0 and v["verdict"] == "PASS"
    assert v["telemetry"]["vllm_on_pass2_floor_exempt"][0]["cached_tokens"] == 0
    # the prompt past two quanta keeps the floor
    long = lambda k, ids: (_warm(k, ids) if len(ids) < 1057 else 512) if k == 2 else None  # noqa: E731
    d, _ = _phases(tmp_path, vllm_on={"cached": long})
    code, v = _verdict(tmp_path, d)
    assert code == 1 and v["verdict"] == "ERROR"
    assert "cache-on path not exercised" in v["error"] and "1 of 1 floor-checked prompts" in v["error"], v["error"]


def _phases_padded(tmp_path: Path, lengths: Dict[str, int], count: int = 3, **tweak: Any) -> tuple:
    """``_phases`` with the prompts of ``lengths`` padded to those token counts
    (the fixture renders 1,149 and 689 tokens; a block multiple needs a length
    the fixture cannot reach), the sha256 recomputed so the phases match."""
    d, pdoc = _phases(tmp_path, count=count)
    for p in pdoc["prompts"]:
        if p["id"] in lengths:
            p["token_ids"] = (p["token_ids"] * 3)[: lengths[p["id"]]]
            p["n_tokens"] = len(p["token_ids"])
    pdoc["sha256"] = ig.prompts_sha256(pdoc["prompts"])
    pdoc["max_prompt_tokens"] = max(p["n_tokens"] for p in pdoc["prompts"])
    (d / "prompts.json").write_text(json.dumps(pdoc), encoding="utf-8")
    for phase, (engine, passes, kw) in _PHASE_SPEC.items():
        kw = {**kw, **tweak.get(phase, {})}
        (d / f"{phase}.json").write_text(json.dumps(_phase_doc(pdoc, phase, engine, passes, **kw)), encoding="utf-8")
    return d, pdoc


def _hybrid(k: int, ids: List[int]) -> Optional[int]:
    """The warm count the S2R 2026-11-10 record showed on every one of its 200
    prompts: whole 272-token blocks, never the last token, and 0 on a prompt
    whose length is an exact multiple of 272."""
    if k != 2:
        return None
    return 0 if len(ids) % 272 == 0 else (len(ids) - 1) // 272 * 272


def test_verdict_exempts_an_exact_block_multiple_at_zero_cached_from_the_warm_floor_s2r_2026_11_10(tmp_path, fakes):
    """On the hybrid model vLLM 0.19.1 keeps the Mamba state in the last block
    and caps the prefix lookup at prompt length minus one, so a prompt whose
    length is an exact multiple of the served block reports 0 cached tokens on
    a correct server (S2R 2026-11-10: prompts of 2,720 and 1,904 tokens at 0,
    every other warm count a multiple of 272). The gate measures the block as
    the gcd of the nonzero warm counts, checks that the block explains every
    nonzero count, and records such prompts; it does not error."""
    # q1 padded to 5 x 272 reports 0; q2 (689 tokens, under two quanta) reports 544 = 2 x 272
    # and q3 padded to 1,400 tokens reports 1360 = 5 x 272, so the measured block is 272
    d, pdoc = _phases_padded(tmp_path, {"q1": 1360, "q3": 1400}, vllm_on={"cached": _hybrid})
    n = {p["id"]: p["n_tokens"] for p in pdoc["prompts"]}
    assert n["q1"] == 1360 and n["q2"] < 1057 <= n["q3"] == 1400
    code, v = _verdict(tmp_path, d)
    assert code == 0 and v["verdict"] == "PASS" and v["error"] is None, v["error"]
    assert v["telemetry"]["vllm_on_pass2_block_tokens"] == 272
    assert v["telemetry"]["vllm_on_pass2_floor_exempt"] == [
        {"prompt_index": 0, "prompt_id": "q2", "prompt_tokens": n["q2"], "cached_tokens": 544},
        {"prompt_index": 1, "prompt_id": "q1", "prompt_tokens": 1360, "cached_tokens": 0, "block_tokens": 272}]
    assert v["limitations"][1] == (
        "the 528-token warm floor is not applied to 1 of 3 prompts whose length is an exact multiple of the 272-token "
        "block measured from the warm counts and that report 0 cached tokens (vLLM keeps the hybrid Mamba state in the "
        "last block and looks up at most prompt length minus one, S2R 2026-11-10): q1 1360 tokens")
    # a nonzero count under the floor on a multiple is still an error
    partial = lambda k, ids: (300 if len(ids) % 272 == 0 else _hybrid(k, ids)) if k == 2 else None  # noqa: E731
    d, _ = _phases_padded(tmp_path, {"q1": 1360, "q3": 1400}, vllm_on={"cached": partial})
    code, v = _verdict(tmp_path, d)
    assert code == 1 and "cache-on path not exercised" in v["error"] and "1 of 2 floor-checked" in v["error"], v["error"]
    # every count 0 leaves no block to measure: the floor errors as before
    nothing = lambda k, ids: 0 if k == 2 else None  # noqa: E731
    d, _ = _phases_padded(tmp_path, {"q1": 1360, "q3": 1400}, vllm_on={"cached": nothing})
    code, v = _verdict(tmp_path, d)
    assert code == 1 and "cache-on path not exercised" in v["error"], v["error"]
    # review 2026-11-10 MEDIUM 1: one count the block does not explain (q2 at 832, not 544)
    # lowers the gcd to 64 and would exempt a 0 at 1,600 tokens (a multiple of 64, not of 272),
    # a real miss on a 272-block server: the block must explain every nonzero count
    misaligned = lambda k, ids: (832 if len(ids) < 1057 else (0 if len(ids) == 1600 else _hybrid(k, ids))) if k == 2 else None  # noqa: E731
    d, _ = _phases_padded(tmp_path, {"q1": 1600}, vllm_on={"cached": misaligned})
    code, v = _verdict(tmp_path, d)
    assert code == 1 and "cache-on path not exercised" in v["error"] and "1 of 2 floor-checked" in v["error"], v["error"]
    # when the exemption empties the floor set the gate still refuses a vacuous floor:
    # q1 (1,632 = 6 x 272) and q3 (1,088 = 4 x 272) are multiples at 0, the short q2 alone measures the block
    d, _ = _phases_padded(tmp_path, {"q1": 1632, "q3": 1088}, vllm_on={"cached": _hybrid})
    code, v = _verdict(tmp_path, d)
    assert code == 1 and v["verdict"] == "ERROR" and "outside the 2 exact block multiples" in v["error"], v["error"]
    # the exempt list stays in prompt order (review 2026-11-10 LOW 3)
    d, _ = _phases_padded(tmp_path, {"q1": 1400, "q3": 1360}, vllm_on={"cached": _hybrid})
    code, v = _verdict(tmp_path, d)
    assert code == 0 and [e["prompt_index"] for e in v["telemetry"]["vllm_on_pass2_floor_exempt"]] == [0, 2], v["error"]


def test_verdict_errors_when_no_prompt_reaches_two_quanta_so_the_floor_is_never_vacuous(tmp_path, fakes, monkeypatch):
    monkeypatch.setattr(ig, "WARM_FLOOR_MIN_PROMPT_TOKENS", 5000, raising=False)
    d, _ = _phases(tmp_path)
    code, v = _verdict(tmp_path, d)
    assert code == 1 and v["verdict"] == "ERROR" and "no prompt past two quanta" in v["error"], v["error"]


def test_verdict_records_a_prefix_hit_on_pass_1_as_a_limitation_without_changing_the_verdict_review_m1(tmp_path, fakes):
    """36 of the 200 registered prompts share a corpus block with an earlier
    prompt of the trial, so their pass 1 is served partly from the prefix cache
    (vLLM) or the radix cache (SGLang). The verdict records the pass-1 counts
    and says so; B still compares identical ids on a correct cache."""
    on = lambda k, ids: (48 if len(ids) >= 1057 else None) if k == 1 else _warm(k, ids)  # noqa: E731
    sg = lambda k, ids: (32 if len(ids) >= 1057 else 0) if k == 1 else len(ids) - 7  # noqa: E731
    d, _ = _phases(tmp_path, vllm_on={"cached": on}, sglang={"cached": sg})
    code, v = _verdict(tmp_path, d)
    assert code == 0 and v["verdict"] == "PASS" and v["error"] is None
    assert v["telemetry"]["vllm_on_pass1_cached_tokens"] == {"min": 0, "median": 24, "max": 48}
    assert v["telemetry"]["vllm_on_pass1_hits"] == 1 and v["telemetry"]["sglang_pass1_hits"] == 1
    assert len(v["limitations"]) == 3, v["limitations"]
    assert v["limitations"][1] == ("pass 1 is not cold on 1 of 2 vLLM prompts (a corpus block shared with an earlier "
                                   "prompt of the trial is served from the prefix cache, review 2026-11-01 MEDIUM 1): "
                                   "comparison B compares a partly warm pass 1 with pass 2")
    assert v["limitations"][2] == ("pass 1 is not cold on 1 of 2 SGLang prompts (a shared corpus block is served from "
                                   "the radix cache): comparisons C and D compare against a partly warm pass 1")


@pytest.mark.parametrize("case", [
    "sglang_missing", "prompts_missing", "sha_mismatch", "warm_under_quantum", "control_cached", "phase_error",
])
def test_verdict_error_is_neither_result_and_exits_1(tmp_path, fakes, capsys, case):
    tweak: Dict[str, Any] = {}
    if case == "sglang_missing":
        tweak["sglang"] = "absent"
    elif case == "sha_mismatch":
        tweak["vllm_off"] = {"sha": "0" * 64}
    elif case == "warm_under_quantum":
        tweak["vllm_on"] = {"cached": lambda k, ids: 512 if k == 2 else None}
    elif case == "control_cached":
        tweak["vllm_off"] = {"cached": lambda k, ids: 16}
    d, _ = _phases(tmp_path, **tweak)
    if case == "prompts_missing":
        (d / "prompts.json").unlink()
    if case == "phase_error":
        (d / "vllm_on.json").write_text(json.dumps({"phase": "vllm_on", "engine": "vllm", "error": "HTTP 500 at prompt 3"}), encoding="utf-8")
    code, v = _verdict(tmp_path, d)
    assert code == 1 and v["verdict"] == "ERROR" and v["schema"] == "cage.identity_gate/1"
    assert capsys.readouterr().out.rstrip().splitlines()[-1] == "IDENTITY_GATE ERROR"
    expect = {"sglang_missing": "sglang.json", "prompts_missing": "prompts.json", "sha_mismatch": "sha256",
              "warm_under_quantum": "cache-on path not exercised", "control_cached": "control not cache-free",
              "phase_error": "HTTP 500 at prompt 3"}[case]
    assert expect in v["error"], v["error"]
    assert v["label_on_pass"] is None


def test_the_constants_and_the_schema_agree_with_the_planner():
    rc = _load("cage_run_campaign_for_identity_gate", REPO_ROOT / "scripts" / "3_run" / "run_campaign.py")
    assert ig.SCHEMA == rc.IDENTITY_GATE_SCHEMA == "cage.identity_gate/1"
    assert ig.REQUEST == REQUEST and ig.WARM_CACHED_TOKENS_MIN == 528 and ig.DEFAULT_PROMPT_TOKEN_LIMIT == 4096 - 32
    assert ig.WARM_FLOOR_MIN_PROMPT_TOKENS == 2 * 528 + 1, "the floor holds for any quantum up to 1,056 tokens [D]"
    assert ig.LABEL_ON_PASS == rc.IDENTITY_GATE_LABEL_ON_PASS == "experimental-align-mode"
