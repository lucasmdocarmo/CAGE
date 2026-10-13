#!/usr/bin/env python3
"""Order:     stage 4 (validate), once, on the pod GPU before any calibration bills
Objective: Prove live that the reranker ranks on the GPU as it ranks on the CPU (S0F-88 B-6, owner ruling R8)
Cloud:     both

R8 moved the registered cross-encoder from the CPU to the GPU: on S0 attempt 4
the CPU reranker took 6.16 h of the 9.84 h run. A device change must not change
what a rerank cell serves. This probe scores 50 real squad_v2 questions (the
first 50 ids of trial 1 of the registered manifest), each against a pool of
RERANK_POOL contexts (its own and the next ones of the manifest), with the
runner's own CrossEncoderReranker built once on the CPU and once on the GPU,
and compares the served top RETRIEVAL_TOP_K query by query. Review 2026-10-13
item 7 (owner decision 4): it FAILS when the GPU serves a different SET of
texts than the CPU (an order swap inside the set is recorded as a swap and
does not fail; the pools repeat texts, so positions alone would fail on two
identical passages changing places), naming the query and both score lists;
it prints ONE line the master greps (RERANK_DEVICE_OK or RERANK_DEVICE_FAILED),
writes the evidence record to --out, and exits 0 or 1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts" / "3_run")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

N_QUERIES = 50
DEFAULT_MANIFEST = _REPO_ROOT / "data" / "manifests" / "squad_v2_50x3_seed42.json"

ScoreFn = Callable[[Sequence[Tuple[str, str]]], Sequence[float]]


def served(scores: Sequence[float], top_k: int) -> List[int]:
    """The served candidate indices, best first (ties keep the pool order,
    the stable sort CrossEncoderReranker.rerank applies)."""
    order = sorted(range(len(scores)), key=lambda i: -float(scores[i]))
    return order[:top_k]


def compare_devices(
    items: Sequence[Tuple[str, Sequence[str]]],
    cpu: ScoreFn,
    gpu: ScoreFn,
    *,
    top_k: int,
) -> Dict[str, Any]:
    """Score every (question, pool) on both devices; the record names every
    query whose served SET of texts differs, counts the queries whose served
    order differs inside the same set (``order_swaps``), and keeps the largest
    score difference."""
    differences: List[Dict[str, Any]] = []
    max_diff = 0.0
    order_swaps = 0
    for qi, (question, pool) in enumerate(items):
        pairs = [(question, text) for text in pool]
        s_cpu = [float(s) for s in cpu(pairs)]
        s_gpu = [float(s) for s in gpu(pairs)]
        max_diff = max(max_diff, max(abs(a - b) for a, b in zip(s_cpu, s_gpu)))
        cpu_top, gpu_top = served(s_cpu, top_k), served(s_gpu, top_k)
        cpu_texts, gpu_texts = [pool[i] for i in cpu_top], [pool[i] for i in gpu_top]
        if sorted(cpu_texts) == sorted(gpu_texts):
            if cpu_texts != gpu_texts:
                order_swaps += 1
            continue
        if True:
            differences.append({
                "query": qi,
                "question": question,
                "served_cpu": served(s_cpu, top_k),
                "served_gpu": served(s_gpu, top_k),
                "scores_cpu": s_cpu,
                "scores_gpu": s_gpu,
            })
    return {
        "queries": len(items),
        "top_k": top_k,
        "identical": len(items) - len(differences),
        "order_swaps": order_swaps,
        "max_abs_score_diff": max_diff,
        "differences": differences,
        "ok": not differences,
    }


def manifest_items(manifest_path: Path, pool_size: int) -> List[Tuple[str, List[str]]]:
    """(question, pool) for the first N_QUERIES ids of trial 1, loaded and
    selected the way the runner does in manifest mode; pool = the question's
    own context and the contexts of the next manifest questions."""
    from run_experiment import get_loader, select_manifest_prefix  # noqa: E402 (heavy, pod venv)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    loader = get_loader(manifest["dataset"], split=manifest["split"], seed=int(manifest["seed"]))
    examples = select_manifest_prefix(
        manifest, trial=1, examples=loader.load(max_examples=None), num_queries=N_QUERIES
    )
    contexts = [" ".join(ex.context or []) for ex in examples]
    return [
        (ex.question, [contexts[(i + j) % len(examples)] for j in range(pool_size)])
        for i, ex in enumerate(examples)
    ]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1].strip())
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=None, help="evidence JSON record")
    args = parser.parse_args(argv)
    from run_campaign import RERANK_POOL, RERANKER_MODEL, RETRIEVAL_TOP_K  # noqa: E402
    from src.orchestration.ir import CrossEncoderReranker  # noqa: E402

    try:
        items = manifest_items(args.manifest, RERANK_POOL)
        cpu = CrossEncoderReranker(RERANKER_MODEL, device="cpu")
        gpu = CrossEncoderReranker(RERANKER_MODEL, device="cuda")
        record = compare_devices(items, cpu._model.predict, gpu._model.predict, top_k=RETRIEVAL_TOP_K)
        record.update(model=RERANKER_MODEL, pool=RERANK_POOL, manifest=str(args.manifest))
    except Exception as exc:  # noqa: BLE001 -- the one line the master greps must still print
        record = {"model": RERANKER_MODEL, "ok": False, "reason": f"{type(exc).__name__}: {exc}"}
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if record["ok"]:
        print(
            f"RERANK_DEVICE_OK queries={record['queries']} identical={record['identical']} "
            f"order_swaps={record['order_swaps']} max_abs_score_diff={record['max_abs_score_diff']:.3g}"
        )
        return 0
    reason = record.get("reason") or (
        f"{len(record['differences'])} of {record['queries']} queries serve a different top "
        f"{record['top_k']} on the GPU (first: query {record['differences'][0]['query']})"
    )
    print(f"RERANK_DEVICE_FAILED reason={reason}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
