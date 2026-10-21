#!/usr/bin/env python3
"""Order:     stage 4 (validate), once, on the pod GPU before any calibration bills
Objective: Prove live that the GPU compressor serves S0's CPU text, fits beside the engine and is faster (A14 item 7)
Cloud:     both

A14 (ADR-0174 amendment 2026-10-20) moves LLMLingua-2 from the CPU to the GPU,
in float32. A device change must not change what a compression cell serves.
This probe compresses the 167 distinct inputs S0 compressed on the CPU
(data/fixtures/s0_compression_reference.jsonl.gz, extracted 2026-10-20 from
experiments/S0/2026-10-14_0016) with the runner's own ContextCompressor at the
registered keep ratio, while the dense encoder and the reranker are loaded
beside it as in a compressed retrieval cell. It FAILS when any output differs
from S0's text, when the helpers' peak reserved GPU memory exceeds
--headroom-gib (default 8.0, the headroom A14 item 4 leaves in unpressured
cells), or when the median time per input is not below S0's CPU median. It
prints ONE line the master greps (COMPRESS_DEVICE_OK or COMPRESS_DEVICE_FAILED),
writes the evidence record to --out, and exits 0 or 1. The run memo
(CAGE_COMPRESSION_CACHE_DIR) is removed from the probe's environment so every
input is computed; a set CAGE_COMPRESSION_RATE fails the probe, because the
registered operating point is the configured ratio.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts" / "3_run")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_REFERENCE = _REPO_ROOT / "data" / "fixtures" / "s0_compression_reference.jsonl.gz"
#: S0's CPU median per request, n = 600 (requests.jsonl compression_latency_ms;
#: MyDocs/CAGE_PAPER_DECISIONS_ANALYSIS.md A14 correction of 2026-10-20).
S0_CPU_MEDIAN_MS = 3980.0
#: A14 item 4: the GPU memory left for the helpers in unpressured cells.
DEFAULT_HEADROOM_GIB = 8.0

CompressFn = Callable[[List[str]], List[str]]


def load_reference(path: Path) -> List[Dict[str, Any]]:
    """The reference records (``key``, ``contexts``, ``compressed``), header dropped."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return [r for r in rows if "_meta" not in r]


def compare_outputs(
    items: Sequence[Dict[str, Any]],
    compress: CompressFn,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> Dict[str, Any]:
    """Compress every reference input once; name each output that differs from
    S0's and keep the time per input (median and nearest-rank 90th percentile)."""
    differences: List[Dict[str, Any]] = []
    latencies: List[float] = []
    for item in items:
        t0 = clock()
        out = compress(list(item["contexts"]))
        latencies.append((clock() - t0) * 1000.0)
        if out != item["compressed"]:
            differences.append({"key": item["key"], "s0": item["compressed"], "device": out})
    ordered = sorted(latencies)
    return {
        "inputs": len(items),
        "identical": len(items) - len(differences),
        "differences": differences,
        "median_ms": statistics.median(ordered) if ordered else None,
        "p90_ms": ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)] if ordered else None,
    }


def failure_reasons(
    record: Dict[str, Any], *, headroom_gib: float, cpu_median_ms: float = S0_CPU_MEDIAN_MS
) -> List[str]:
    """Every pass rule the record breaks (empty = pass)."""
    reasons: List[str] = []
    if record["inputs"] == 0:
        reasons.append("the reference holds no inputs")
    if record["differences"]:
        reasons.append(
            f"{len(record['differences'])} of {record['inputs']} outputs differ from S0's CPU "
            f"text (first: {record['differences'][0]['key'][:12]})"
        )
    peak = record.get("peak_reserved_gib")
    if peak is None or peak > headroom_gib:
        reasons.append(f"peak reserved GPU memory {peak} GiB is not within {headroom_gib} GiB")
    median = record.get("median_ms")
    if median is None or median >= cpu_median_ms:
        reasons.append(f"median {median} ms per input is not below S0's CPU median {cpu_median_ms} ms")
    return reasons


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1].strip())
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--freeze-file", type=Path, default=None,
                        help="the registration artifact (default: the driver's resolution chain)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--headroom-gib", type=float, default=DEFAULT_HEADROOM_GIB)
    parser.add_argument("--out", type=Path, default=None, help="evidence JSON record")
    args = parser.parse_args(argv)
    from src.orchestration.compression import COMPRESSION_CACHE_ENV  # noqa: E402

    os.environ.pop(COMPRESSION_CACHE_ENV, None)
    try:
        if (os.environ.get("CAGE_COMPRESSION_RATE") or "").strip():
            raise ValueError("CAGE_COMPRESSION_RATE is set: the probe tests the registered ratio only")
        import torch  # noqa: E402 (pod venv)
        from run_campaign import RERANKER_MODEL, resolve_retrieval_pins  # noqa: E402
        from src.orchestration.baselines import get_baseline_config  # noqa: E402
        from src.orchestration.compression import ContextCompressor  # noqa: E402
        from src.orchestration.ir import CrossEncoderReranker, FaissIRIndex, IRDocument, stable_text_id  # noqa: E402

        cfg = get_baseline_config("compressed_rag")
        items = load_reference(args.reference)
        torch.cuda.reset_peak_memory_stats()
        # the helpers a compressed retrieval cell holds at once, each exercised once
        pins = resolve_retrieval_pins(args.freeze_file)
        passages = sorted({c for it in items for c in it["contexts"]})
        encoder = FaissIRIndex(
            embedding_model=pins.embedding_model, device=args.device,
            embedding_revision=pins.embedding_revision,
        )
        encoder.build([IRDocument(doc_id=stable_text_id(t), text=t, metadata={}) for t in passages])
        reranker = CrossEncoderReranker(RERANKER_MODEL, device=args.device)
        reranker._model.predict([(passages[0][:200], passages[0])])
        compressor = ContextCompressor(method=cfg.compress_method, device=args.device)
        if compressor.compressor is None:  # loaded before the clock: no load time in the first input
            raise RuntimeError("the compressor did not load")
        record = compare_outputs(
            items, lambda docs: compressor.compress(docs, target_ratio=cfg.compress_target_ratio)[0]
        )
        record.update(
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            device=args.device,
            keep_ratio=cfg.compress_target_ratio,
            compressor_revision=compressor.revision,
            reranker_revision=reranker.revision,
            embedding_model=pins.embedding_model,
            embedding_revision=pins.embedding_revision,
            reference=str(args.reference),
        )
        reasons = failure_reasons(record, headroom_gib=args.headroom_gib)
        record.update(ok=not reasons, reason="; ".join(reasons) or None)
    except Exception as exc:  # noqa: BLE001 -- the one line the master greps must still print
        record = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if record["ok"]:
        print(
            f"COMPRESS_DEVICE_OK inputs={record['inputs']} identical={record['identical']} "
            f"median_ms={record['median_ms']:.0f} p90_ms={record['p90_ms']:.0f} "
            f"peak_reserved_gib={record['peak_reserved_gib']:.2f}"
        )
        return 0
    print(f"COMPRESS_DEVICE_FAILED reason={record['reason']}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
