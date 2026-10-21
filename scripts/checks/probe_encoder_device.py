#!/usr/bin/env python3
"""Order:     stage 4 (validate), once, on the pod GPU before any calibration bills
Objective: Prove live that the dense encoder ranks on the GPU as it ranks on the CPU (A14, owner 2026-10-20)
Cloud:     both

A14 moved the registered dense encoder (the freeze slot's dense_retriever, at
its pinned revision) from the CPU to the GPU. A device change must not change
which passages a question is served. This probe takes the reranker probe's 50
registered questions and pools (probe_reranker_device.manifest_items), scores
each pool with the encoder on the CPU and on the GPU (cosine similarity of the
normalized query and passage vectors, with the e5 prefixes the index uses),
and applies the reranker probe's rule: it FAILS when the GPU serves a
different SET of top RETRIEVAL_TOP_K texts than the CPU (an order swap inside
the set is recorded, not failed). It prints ONE line the master greps
(ENCODER_DEVICE_OK or ENCODER_DEVICE_FAILED), writes the evidence record to
--out, and exits 0 or 1.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts" / "3_run")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _reranker_probe() -> Any:
    spec = importlib.util.spec_from_file_location("probe_reranker_device", _HERE / "probe_reranker_device.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encoder_scores(model: Any, prefixes: bool) -> Callable[[Sequence[Tuple[str, str]]], List[float]]:
    """A score function over (question, passage) pairs that share one
    question: cosine similarity of the normalized vectors, as the index ranks."""

    def score(pairs: Sequence[Tuple[str, str]]) -> List[float]:
        question = pairs[0][0]
        q = model.encode([f"query: {question}" if prefixes else question], normalize_embeddings=True)
        p = model.encode([f"passage: {t}" if prefixes else t for _q, t in pairs], normalize_embeddings=True)
        return [float(v) for v in (p @ q[0])]

    return score


def main(argv: Optional[list[str]] = None) -> int:
    probe = _reranker_probe()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1].strip())
    parser.add_argument("--manifest", type=Path, default=probe.DEFAULT_MANIFEST)
    parser.add_argument("--freeze-file", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="evidence JSON record")
    args = parser.parse_args(argv)
    from run_campaign import RERANK_POOL, RETRIEVAL_TOP_K, resolve_retrieval_pins  # noqa: E402
    from src.orchestration.ir import embedding_model_wants_e5_prefixes  # noqa: E402

    try:
        from sentence_transformers import SentenceTransformer  # noqa: E402 (pod venv)

        pins = resolve_retrieval_pins(args.freeze_file)
        items = probe.manifest_items(args.manifest, RERANK_POOL)
        prefixes = embedding_model_wants_e5_prefixes(pins.embedding_model)
        cpu = SentenceTransformer(pins.embedding_model, device="cpu", revision=pins.embedding_revision)
        gpu = SentenceTransformer(pins.embedding_model, device="cuda", revision=pins.embedding_revision)
        record = probe.compare_devices(
            items, encoder_scores(cpu, prefixes), encoder_scores(gpu, prefixes), top_k=RETRIEVAL_TOP_K
        )
        record.update(model=pins.embedding_model, revision=pins.embedding_revision, pool=RERANK_POOL,
                      manifest=str(args.manifest))
    except Exception as exc:  # noqa: BLE001 -- the one line the master greps must still print
        record = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if record["ok"]:
        print(
            f"ENCODER_DEVICE_OK queries={record['queries']} identical={record['identical']} "
            f"order_swaps={record['order_swaps']} max_abs_score_diff={record['max_abs_score_diff']:.3g}"
        )
        return 0
    reason = record.get("reason") or (
        f"{len(record['differences'])} of {record['queries']} queries serve a different top "
        f"{record['top_k']} on the GPU (first: query {record['differences'][0]['query']})"
    )
    print(f"ENCODER_DEVICE_FAILED reason={reason}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
