#!/usr/bin/env python3
"""Order:     stage 4 (validate), once per identity-gated landing, after the engine probes and before the start check
Objective: Prove live at T=0 that vLLM's aligned Qwen3-Next prefix cache changes no generated token (charter 5.2, S2 block C-1)
Cloud:     both

Charter 5.2 makes the T=0 token-identity check a gate: before any campaign data
on the hybrid Qwen3-Next model, the prefix-on vLLM server must generate the
same tokens as the same server with the cache off and as SGLang, at the pinned
versions. PASS lets the gated cells run labeled experimental-align-mode; FAIL
excludes them and is itself the result; ERROR is neither (the gate did not run
to a verdict) and fails the stage.

Three subcommands, composed by scripts/6_experiments/cage_experiment.sh:

  prompts  the first --count questions of one manifest trial, each rendered in
           the corpus-reuse shape (format_qa_messages(question, [block text]),
           the campaign's own request, run_experiment.py 3198-3202) by the
           pinned tokenizer with the generation prompt and enable_thinking
           False; token ids, never text, so every engine sees one input.
  run      one server configuration, --passes sweeps, one request in flight,
           prompt by prompt (pass 1 then pass 2 of the same prompt, so pass 2
           follows its own pass 1 whatever the pool holds; pass 1 is cold unless
           an earlier prompt shared its corpus block, which the telemetry
           records as a limitation). vLLM:
           POST /v1/completions with prompt = ids and return_token_ids, which
           echoes prompt_token_ids and returns token_ids [V vLLM 0.19.1
           completion/protocol.py 468-470, serving.py 528-532]; cached tokens
           from usage.prompt_tokens_details.cached_tokens, absent on a cold
           request [V serving.py 547-553]. SGLang: POST /generate with
           input_ids and sampling_params; output_ids and meta_info.cached_tokens
           [V SGLang 0.5.10.post1 tokenizer_manager.py 1579, 1618-1622];
           temperature 0 is greedy (top_k 1) [V sampling_params.py 98-101].
  verdict  the comparisons of charter 5.2 (A: vLLM warm vs cache-off and B:
           vLLM cold vs warm decide; C, D, E against SGLang are recorded; F and
           G against the SGLang extra-buffer probe phase ``sglang_xb``, run only
           when ``--sglang-xb`` says stage 4 started it, ADR-0195), the
           telemetry that makes the test honest (every warm prompt past two
           quanta cached at or above one 528-token quantum, except a prompt
           whose length is an exact multiple of the block measured from the
           warm counts and that reports 0, recorded; every cache-off prompt
           at 0; shorter prompts and pass-1 hits are recorded, not asserted),
           one
           marker line IDENTITY_GATE PASS|FAIL|ERROR, exit 0 on PASS or FAIL
           and 1 on ERROR.

Stdlib and the repo's own src only; the Hub and transformers are reached in
``prompts`` alone (the pod runs it offline on the prefetched model).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: The verdict file schema the planner reads (run_campaign.py IDENTITY_GATE_SCHEMA).
SCHEMA = "cage.identity_gate/1"
PROMPTS_SCHEMA = "cage.identity_gate.prompts/1"
#: The label the gated cells carry on PASS (charter 5.2).
LABEL_ON_PASS = "experimental-align-mode"
#: The request constants (S2 block C spec, DEC-C3): greedy, one seed, 32 tokens, no stop.
REQUEST: Dict[str, Any] = {"temperature": 0.0, "seed": 20260902, "max_tokens": 32, "stop": None}
#: Charter 5.2's quantum: 33 x 16-token blocks [D, spec section 4]; a warm prompt
#: past two quanta that reports fewer cached tokens never exercised the aligned cache.
WARM_CACHED_TOKENS_MIN = 528
#: The floor applies to prompts of at least two quanta plus one token. For any
#: aligned quantum Q <= 528 and N >= 1057 the warm count floor((N-1)/Q)*Q exceeds
#: N-1-Q >= 528; for 528 < Q <= 1056 one full block of Q >= 528 fits [D, review
#: 2026-11-01 HIGH 1]. The quantum at TP 4 is unmeasured and the manifest's blocks
#: run from 527 tokens [V], so a shorter prompt may hold one block, or none: it is
#: recorded in the telemetry and the limitations, never asserted.
WARM_FLOOR_MIN_PROMPT_TOKENS = 2 * WARM_CACHED_TOKENS_MIN + 1
#: vLLM 0.19.1 caps the prefix lookup at prompt length minus one
#: (v1/core/kv_cache_manager.py 196-204) and, on the hybrid model, keeps the
#: Mamba state in the request's last block alone (single_type_kv_cache_manager.py
#: MambaManager.allocate_new_blocks, null blocks before it; block_pool.py
#: cache_full_blocks skips null blocks), so a prompt whose length is an exact
#: multiple of the served block reports 0 cached tokens on a correct server
#: (S2R 2026-11-10: 2,720 and 1,904 tokens at 0, every other warm count a
#: multiple of 272). The verdict measures the block as the gcd of the nonzero
#: warm counts and exempts such prompts from the floor, recorded, only when
#: that block explains every nonzero count as floor((N-1)/block)*block (the
#: record's law on all 198 nonzero counts; review 2026-11-10 MEDIUM 1: one
#: misaligned count would otherwise lower the gcd and widen the exemption) and
#: is at least 16 tokens, the smallest block the gate accepts.
BLOCK_TOKENS_MIN = 16
#: The smallest served context on the gated rows minus the generation budget.
DEFAULT_PROMPT_TOKEN_LIMIT = 4096 - 32
SHAPE = "corpus-reuse"
#: Phase name -> engine; the passes the verdict needs of each. ``sglang_xb`` is
#: SGLang under the extra-buffer Mamba scheduler (ADR-0195), the probe cell's
#: serving mode, started fourth at stage 4 only on a session that plans the probe.
PHASES: Dict[str, Tuple[str, int]] = {"vllm_on": ("vllm", 2), "vllm_off": ("vllm", 1), "sglang": ("sglang", 2),
                                      "sglang_xb": ("sglang", 2)}
#: Phases the verdict reads only when asked for: (phase, the flag that asks).
OPTIONAL_PHASES: Dict[str, str] = {"sglang_xb": "sglang_xb"}
#: (key, left phase, left pass, right phase, right pass, decides)
COMPARISONS = (
    ("A", "vllm_on", 2, "vllm_off", 1, True),
    ("B", "vllm_on", 1, "vllm_on", 2, True),
    ("C", "vllm_off", 1, "sglang", 1, False),
    ("D", "vllm_on", 2, "sglang", 1, False),
    ("E", "sglang", 1, "sglang", 2, False),
    ("F", "vllm_off", 1, "sglang_xb", 1, False),
    ("G", "sglang_xb", 1, "sglang_xb", 2, False),
)
DECIDING = ["A", "B"]


class GateError(Exception):
    """A refusal or an error of the gate: the message is the reason."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def prompts_sha256(prompts: List[Dict[str, Any]]) -> str:
    """The digest of the prompt ids and token lists (what every phase compares against)."""
    raw = json.dumps([[p["id"], p["token_ids"]] for p in prompts], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------


def load_examples(dataset: str, split: str, seed: int) -> List[Any]:
    """The dataset through the runner's own loader (tests swap this in)."""
    from src.data.loader import get_loader
    return get_loader(dataset, split=split, seed=seed).load()


def load_tokenizer(model: str, revision: str) -> Any:
    """The pinned tokenizer (tests swap this in)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model, revision=revision)


def build_prompts(args: argparse.Namespace) -> Dict[str, Any]:
    from src.data.manifest import ManifestError, block_for, select_examples
    from src.utils.prompting import format_qa_messages

    path = Path(args.manifest)
    if not path.is_file():
        raise GateError(f"manifest not found: {path}")
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("dataset") != args.dataset:
        raise GateError(f"--dataset {args.dataset} but the manifest is {manifest.get('dataset')!r}")
    tok = manifest.get("tokenizer") or {}
    if tok.get("id") != args.model or tok.get("revision") != args.revision:
        raise GateError(
            f"the manifest's tokenizer record {tok} is not the pinned {args.model} at revision {args.revision}: "
            "rebuild the manifest with the pinned tokenizer (S2 block A-6)"
        )
    pool = manifest.get("trials", {}).get(str(args.trial)) or []
    if args.count < 1 or args.count > len(pool):
        raise GateError(f"--count {args.count} is outside the trial {args.trial} pool of {len(pool)} questions")
    examples = load_examples(args.dataset, manifest.get("split", "validation"), int(manifest.get("seed", 42)))
    try:
        selected = select_examples(manifest, args.trial, examples)[: args.count]
    except ManifestError as exc:
        raise GateError(str(exc)) from exc
    tokenizer = load_tokenizer(args.model, args.revision)
    prompts: List[Dict[str, Any]] = []
    for ex in selected:
        block = block_for(manifest, ex.id)
        rendered = tokenizer.apply_chat_template(
            format_qa_messages(ex.question, [block["text"]]),
            tokenize=True, add_generation_prompt=True, enable_thinking=False,
        )
        if hasattr(rendered, "get") and "input_ids" in rendered:  # a BatchEncoding, not a list
            rendered = rendered["input_ids"]
        token_ids = [int(t) for t in rendered]
        if len(token_ids) > args.prompt_token_limit:
            raise GateError(
                f"prompt {ex.id} renders to {len(token_ids)} tokens, over the {args.prompt_token_limit}-token limit "
                "(the served context minus the generation budget)"
            )
        prompts.append({"id": ex.id, "block_id": block["block_id"], "n_tokens": len(token_ids), "token_ids": token_ids})
    return {
        "schema": PROMPTS_SCHEMA,
        "session": args.session, "model": args.model, "revision": args.revision,
        "tokenizer_revision": tok["revision"],
        "manifest": str(path), "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "dataset": args.dataset, "trial": args.trial, "count": len(prompts), "shape": SHAPE,
        "prompt_token_limit": args.prompt_token_limit,
        "max_prompt_tokens": max(p["n_tokens"] for p in prompts),
        "request": dict(REQUEST),
        "sha256": prompts_sha256(prompts),
        "prompts": prompts,
    }


def cmd_prompts(args: argparse.Namespace) -> int:
    try:
        doc = build_prompts(args)
    except GateError as exc:
        print(f"IDENTITY_GATE_PROMPTS_REFUSED reason={exc}", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc), encoding="utf-8")
    print(f"IDENTITY_GATE_PROMPTS_OK count={doc['count']} max_prompt_tokens={doc['max_prompt_tokens']} sha256={doc['sha256']}")
    return 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _http_json(url: str, payload: Optional[Dict[str, Any]] = None, timeout_s: float = 300.0) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                 method="GET" if payload is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:200]
        raise GateError(f"HTTP {exc.code} from {url}: {body}") from exc
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise GateError(f"request to {url} failed: {exc}") from exc


def engine_version(engine: str, api_base: str, timeout_s: float) -> str:
    if engine == "vllm":
        return str(_http_json(f"{api_base}/version", timeout_s=timeout_s).get("version"))
    return str(_http_json(f"{api_base}/server_info", timeout_s=timeout_s).get("version"))


def generate(engine: str, api_base: str, model: str, token_ids: List[int], timeout_s: float) -> Dict[str, Any]:
    """One greedy request of token ids; the generated ids, text, finish reason,
    cached and prompt token counts, and (vLLM) the echoed prompt ids."""
    if engine == "vllm":
        body = {"model": model, "prompt": token_ids, "max_tokens": REQUEST["max_tokens"],
                "temperature": REQUEST["temperature"], "seed": REQUEST["seed"], "return_token_ids": True}
        doc = _http_json(f"{api_base}/v1/completions", body, timeout_s)
        choice = (doc.get("choices") or [{}])[0]
        usage = doc.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        return {"token_ids": choice.get("token_ids"), "text": choice.get("text", ""),
                "finish_reason": choice.get("finish_reason"), "cached_tokens": details.get("cached_tokens"),
                "prompt_tokens": usage.get("prompt_tokens"), "echoed": choice.get("prompt_token_ids")}
    body = {"input_ids": token_ids, "sampling_params": {"temperature": REQUEST["temperature"],
                                                        "max_new_tokens": REQUEST["max_tokens"],
                                                        "sampling_seed": REQUEST["seed"]}}
    doc = _http_json(f"{api_base}/generate", body, timeout_s)
    meta = doc.get("meta_info") or {}
    reason = meta.get("finish_reason")
    return {"token_ids": doc.get("output_ids"), "text": doc.get("text", ""),
            "finish_reason": reason.get("type") if isinstance(reason, dict) else reason,
            "cached_tokens": meta.get("cached_tokens"), "prompt_tokens": meta.get("prompt_tokens"), "echoed": None}


def cmd_run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc: Dict[str, Any] = {"phase": args.phase, "engine": args.engine, "api_base": args.api_base, "model": args.model,
                           "passes": args.passes, "prompts_path": args.prompts, "started_at": _now()}
    try:
        if args.engine not in ("vllm", "sglang"):
            raise GateError(f"--engine {args.engine} is not vllm or sglang")
        if args.passes < 1:
            raise GateError(f"--passes {args.passes} must be at least 1")
        ppath = Path(args.prompts)
        if not ppath.is_file():
            raise GateError(f"prompts file not found: {ppath}")
        pdoc = json.loads(ppath.read_text(encoding="utf-8"))
        doc["prompts_sha256"] = pdoc["sha256"]
        doc["request"] = dict(REQUEST)
        doc["engine_version"] = engine_version(args.engine, args.api_base, args.timeout_s)
        prompts = pdoc["prompts"]
        results: List[List[Any]] = [[None] * len(prompts) for _ in range(args.passes)]
        for i, p in enumerate(prompts):
            for k in range(1, args.passes + 1):
                where = f"{args.phase} prompt {i} ({p['id']}) pass {k}"
                t0 = time.monotonic()
                try:
                    r = generate(args.engine, args.api_base, args.model, p["token_ids"], args.timeout_s)
                except GateError as exc:
                    raise GateError(f"{where}: {exc}") from exc
                latency = time.monotonic() - t0
                ids = r["token_ids"]
                if not isinstance(ids, list) or any(not isinstance(t, int) for t in ids):
                    raise GateError(f"{where}: no token ids in the response")
                if not ids:
                    raise GateError(f"{where}: empty generation")
                if args.engine == "vllm" and r["echoed"] != p["token_ids"]:
                    raise GateError(f"{where}: the echoed prompt ids differ from the ids sent (echo check)")
                results[k - 1][i] = {"prompt_index": i, "prompt_id": p["id"], "pass": k, "token_ids": ids,
                                     "text": r["text"], "finish_reason": r["finish_reason"],
                                     "cached_tokens": r["cached_tokens"], "prompt_tokens": r["prompt_tokens"],
                                     "latency_s": round(latency, 6)}
    except GateError as exc:
        doc["error"] = str(exc)
        doc["finished_at"] = _now()
        out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        print(f"IDENTITY_GATE_RUN_ERROR phase={args.phase} reason={exc}", file=sys.stderr)
        return 1
    doc["results"] = results
    doc["finished_at"] = _now()
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"IDENTITY_GATE_RUN_OK phase={args.phase} prompts={len(prompts)} passes={args.passes} "
          f"engine_version={doc['engine_version']}")
    return 0


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------


def _load_phase(d: Path, name: str, count: int) -> Dict[str, Any]:
    p = d / f"{name}.json"
    if not p.is_file():
        raise GateError(f"phase file {name}.json not found under {d}")
    doc = json.loads(p.read_text(encoding="utf-8"))
    if doc.get("error"):
        raise GateError(f"phase {name}: {doc['error']}")
    engine, passes = PHASES[name]
    if doc.get("engine") != engine or int(doc.get("passes") or 0) < passes:
        raise GateError(f"phase {name}: engine {doc.get('engine')!r} with {doc.get('passes')} pass(es); "
                        f"the verdict needs {engine} with {passes}")
    results = doc.get("results") or []
    if len(results) < passes or any(len(row) != count or any(r is None for r in row) for row in results[:passes]):
        raise GateError(f"phase {name}: results do not cover {count} prompts on {passes} pass(es)")
    return doc


def _pass(doc: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
    return doc["results"][k - 1]


def _divergence(left: List[Dict[str, Any]], right: List[Dict[str, Any]]) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Identical prompts and the first divergence: the first differing position,
    or the shorter length when one generation is a prefix of the other."""
    identical = 0
    first: Optional[Dict[str, Any]] = None
    for i, (lt, rt) in enumerate(zip(left, right)):
        a, b = lt["token_ids"], rt["token_ids"]
        if a == b:
            identical += 1
            continue
        if first is None:
            n = min(len(a), len(b))
            pos = next((j for j in range(n) if a[j] != b[j]), n)
            first = {"prompt_index": i, "prompt_id": lt["prompt_id"], "position": pos,
                     "left_id": a[pos] if pos < len(a) else None, "right_id": b[pos] if pos < len(b) else None,
                     "left_text": lt["text"], "right_text": rt["text"]}
    return identical, first


def _stats(values: List[int]) -> Dict[str, Any]:
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def build_verdict(args: argparse.Namespace) -> Dict[str, Any]:
    d = Path(args.phases)
    v: Dict[str, Any] = {"schema": SCHEMA, "verdict": "ERROR", "error": None, "session": None, "model": None,
                         "revision": None, "tokenizer_revision": None, "manifest": None, "manifest_sha256": None,
                         "prompts": None, "request": None, "phases": None, "telemetry": None, "comparisons": None,
                         "deciding": list(DECIDING), "label_on_pass": None, "limitations": [], "started_at": None,
                         "finished_at": None}
    try:
        pp = d / "prompts.json"
        if not pp.is_file():
            raise GateError(f"prompts.json not found under {d}")
        pdoc = json.loads(pp.read_text(encoding="utf-8"))
        count = int(pdoc["count"])
        v.update({k: pdoc.get(k) for k in ("session", "model", "revision", "tokenizer_revision", "manifest", "manifest_sha256")})
        v["prompts"] = {"count": count, "dataset": pdoc.get("dataset"), "trial": pdoc.get("trial"), "shape": pdoc.get("shape"),
                        "sha256": pdoc.get("sha256"), "max_prompt_tokens": pdoc.get("max_prompt_tokens")}
        v["request"] = pdoc.get("request")
        names = [n for n in PHASES if n != "sglang" or not args.no_sglang]
        names = [n for n in names if n not in OPTIONAL_PHASES or getattr(args, OPTIONAL_PHASES[n], False)]
        phases: Dict[str, Dict[str, Any]] = {}
        for name in names:
            doc = _load_phase(d, name, count)
            if doc.get("prompts_sha256") != pdoc.get("sha256"):
                raise GateError(f"phase {name}: prompts sha256 {doc.get('prompts_sha256')} differs from prompts.json {pdoc.get('sha256')}")
            phases[name] = doc
        v["phases"] = {n: {"engine": doc["engine"], "passes": doc["passes"], "engine_version": doc.get("engine_version")}
                       for n, doc in phases.items()}
        if args.no_sglang:
            v["phases"]["sglang"] = "absent"
        for name in OPTIONAL_PHASES:
            if name not in phases:
                v["phases"][name] = "absent"
        v["started_at"] = min(doc.get("started_at") or "" for doc in phases.values()) or None
        on1 = [r["cached_tokens"] or 0 for r in _pass(phases["vllm_on"], 1)]
        warm = [r["cached_tokens"] or 0 for r in _pass(phases["vllm_on"], 2)]
        off = [r["cached_tokens"] or 0 for r in _pass(phases["vllm_off"], 1)]
        sgl1 = [r["cached_tokens"] or 0 for r in _pass(phases["sglang"], 1)] if "sglang" in phases else None
        sgl = [r["cached_tokens"] or 0 for r in _pass(phases["sglang"], 2)] if "sglang" in phases else None
        xb1 = [r["cached_tokens"] or 0 for r in _pass(phases["sglang_xb"], 1)] if "sglang_xb" in phases else None
        xb = [r["cached_tokens"] or 0 for r in _pass(phases["sglang_xb"], 2)] if "sglang_xb" in phases else None
        ntok = [int(p["n_tokens"]) for p in pdoc["prompts"]]
        exempt = [{"prompt_index": i, "prompt_id": pdoc["prompts"][i]["id"], "prompt_tokens": ntok[i], "cached_tokens": warm[i]}
                  for i in range(count) if ntok[i] < WARM_FLOOR_MIN_PROMPT_TOKENS]
        block = math.gcd(*[int(c) for c in warm if c > 0]) if any(c > 0 for c in warm) else 0
        lawful = block >= BLOCK_TOKENS_MIN and all(
            warm[j] == (ntok[j] - 1) // block * block for j in range(count) if warm[j] > 0)
        multiples = [i for i in range(count)
                     if lawful and ntok[i] >= WARM_FLOOR_MIN_PROMPT_TOKENS and warm[i] == 0 and ntok[i] % block == 0]
        exempt.extend({"prompt_index": i, "prompt_id": pdoc["prompts"][i]["id"], "prompt_tokens": ntok[i],
                       "cached_tokens": 0, "block_tokens": block} for i in multiples)
        exempt.sort(key=lambda e: e["prompt_index"])
        hits = sum(1 for c in on1 if c > 0)
        shits = sum(1 for c in sgl1 if c > 0) if sgl1 is not None else None
        xhits = sum(1 for c in xb1 if c > 0) if xb1 is not None else None
        v["telemetry"] = {"vllm_on_pass1_cached_tokens": _stats(on1), "vllm_on_pass1_hits": hits,
                          "vllm_on_pass2_cached_tokens": _stats(warm), "vllm_on_pass2_floor_exempt": exempt,
                          "vllm_on_pass2_block_tokens": block,
                          "vllm_off_cached_tokens_max": max(off), "sglang_pass1_hits": shits,
                          "sglang_pass2_cached_tokens": _stats(sgl) if sgl is not None else None,
                          # ADR-0195: the extra-buffer probe phase; None when stage 4 did not start it
                          "sglang_xb_pass1_hits": xhits,
                          "sglang_xb_pass2_cached_tokens": _stats(xb) if xb is not None else None}
        short = [e for e in exempt if "block_tokens" not in e]
        if short:
            v["limitations"].append(
                f"the {WARM_CACHED_TOKENS_MIN}-token warm floor is not applied to {len(short)} of {count} prompts under "
                f"{WARM_FLOOR_MIN_PROMPT_TOKENS} tokens (one aligned quantum may exceed their length, review 2026-11-01 HIGH 1): "
                + "; ".join(f"{e['prompt_id']} {e['prompt_tokens']} tokens, {e['cached_tokens']} cached on pass 2" for e in short))
        if multiples:
            v["limitations"].append(
                f"the {WARM_CACHED_TOKENS_MIN}-token warm floor is not applied to {len(multiples)} of {count} prompts whose "
                f"length is an exact multiple of the {block}-token block measured from the warm counts and that report 0 "
                "cached tokens (vLLM keeps the hybrid Mamba state in the last block and looks up at most prompt length "
                "minus one, S2R 2026-11-10): " + "; ".join(f"{pdoc['prompts'][i]['id']} {ntok[i]} tokens" for i in multiples))
        if hits:
            v["limitations"].append(
                f"pass 1 is not cold on {hits} of {count} vLLM prompts (a corpus block shared with an earlier prompt of the trial "
                f"is served from the prefix cache, review 2026-11-01 MEDIUM 1): comparison B compares a partly warm pass 1 with pass 2")
        if shits:
            v["limitations"].append(
                f"pass 1 is not cold on {shits} of {count} SGLang prompts (a shared corpus block is served from the radix cache): "
                f"comparisons C and D compare against a partly warm pass 1")
        if xhits:
            v["limitations"].append(
                f"pass 1 is not cold on {xhits} of {count} prompts of the SGLang extra-buffer probe (sglang_xb; a shared "
                f"corpus block is served from the Mamba radix cache, ADR-0195): comparison F compares against a partly warm pass 1")
        comps: Dict[str, Dict[str, Any]] = {}
        for key, ln, lk, rn, rk, decides in COMPARISONS:
            if ln not in phases or rn not in phases:
                missing = ln if ln not in phases else rn
                comps[key] = {"decides": False, "left": None, "right": None, "compared": 0, "identical": None,
                              "first_divergence": None, "skipped": f"{missing} absent"}
                continue
            identical, first = _divergence(_pass(phases[ln], lk), _pass(phases[rn], rk))
            comps[key] = {"decides": decides, "left": f"{ln}/{lk}", "right": f"{rn}/{rk}", "compared": count,
                          "identical": identical, "first_divergence": first}
        v["comparisons"] = comps
        checked = [i for i in range(count) if ntok[i] >= WARM_FLOOR_MIN_PROMPT_TOKENS and i not in multiples]
        if not checked:
            raise GateError(f"no prompt past two quanta ({WARM_FLOOR_MIN_PROMPT_TOKENS} tokens)"
                            + (f" outside the {len(multiples)} exact block multiples" if multiples else "")
                            + ": the warm floor cannot be checked")
        low = [(i, warm[i]) for i in checked if warm[i] < WARM_CACHED_TOKENS_MIN]
        if low:
            raise GateError(f"cache-on path not exercised: the warm pass reports cached_tokens under {WARM_CACHED_TOKENS_MIN} "
                            f"on {len(low)} of {len(checked)} floor-checked prompts (first: prompt {low[0][0]} at {low[0][1]})")
        hot = [(i, c) for i, c in enumerate(off) if c > 0]
        if hot:
            raise GateError(f"control not cache-free: the cache-off pass reports cached_tokens above 0 on {len(hot)} of "
                            f"{len(off)} prompts (first: prompt {hot[0][0]} at {hot[0][1]})")
        fail = any(comps[k]["identical"] != comps[k]["compared"] for k in DECIDING)
        v["verdict"] = "FAIL" if fail else "PASS"
        v["label_on_pass"] = None if fail else LABEL_ON_PASS
    except GateError as exc:
        v["verdict"] = "ERROR"
        v["error"] = str(exc)
    v["finished_at"] = _now()
    return v


def cmd_verdict(args: argparse.Namespace) -> int:
    v = build_verdict(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(v, indent=1), encoding="utf-8")
    if v["comparisons"]:
        print("comparisons: " + "; ".join(
            f"{k} {c['left']} vs {c['right']}: {c['identical']}/{c['compared']} identical"
            f"{' (decides)' if c['decides'] else ''}" if c.get("compared") else f"{k}: {c.get('skipped')}"
            for k, c in v["comparisons"].items()))
    if v["error"]:
        print(f"error: {v['error']}")
    print(f"IDENTITY_GATE {v['verdict']}")
    return 1 if v["verdict"] == "ERROR" else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="identity_gate", description=__doc__.split("\n\n")[1])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prompts", help="render the gate prompts (token ids) from one manifest trial")
    p.add_argument("--manifest", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--trial", type=int, default=1)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--model", required=True, help="the served model id (the manifest's tokenizer id)")
    p.add_argument("--revision", required=True, help="the pinned revision (the manifest's tokenizer revision)")
    p.add_argument("--session", required=True)
    p.add_argument("--prompt-token-limit", type=int, default=DEFAULT_PROMPT_TOKEN_LIMIT)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_prompts)

    r = sub.add_parser("run", help="send the prompts to one server configuration for K passes")
    r.add_argument("--engine", required=True, choices=("vllm", "sglang"))
    r.add_argument("--phase", required=True, help="the phase name (vllm_on, vllm_off, sglang, sglang_xb)")
    r.add_argument("--api-base", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--prompts", required=True)
    r.add_argument("--passes", type=int, required=True)
    r.add_argument("--timeout-s", type=float, default=300.0)
    r.add_argument("--out", required=True)
    r.set_defaults(func=cmd_run)

    w = sub.add_parser("verdict", help="compare the recorded phases and write the verdict")
    w.add_argument("--phases", required=True, help="the directory holding prompts.json and the phase files")
    w.add_argument("--no-sglang", action="store_true", help="SGLang is not in this landing's engines: recorded absent")
    w.add_argument("--sglang-xb", action="store_true",
                   help="stage 4 also started SGLang under the extra-buffer scheduler (ADR-0195): read phase sglang_xb")
    w.add_argument("--out", required=True)
    w.set_defaults(func=cmd_verdict)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
