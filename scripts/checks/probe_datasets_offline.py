#!/usr/bin/env python3
"""Order:     stage 4 (validate), after the load gate and before any engine starts
Objective: Prove live that every charter dataset loads through the runner's own loader under the run job's offline mode (S1 audit 2026-10-23, MEDIUM 4)
Cloud:     both

The run job exports HF_HUB_OFFLINE=1 (A10) and the first cell of each dataset
is the first time the loader opens that dataset offline; S0 ran online, and
stage 4's gate (p) checks only that the cache directories exist. This probe
loads --max-examples examples of every dataset in --datasets through
``src.data.loader.get_loader`` under the job's environment and prints ONE line
the master greps (DATASETS_OFFLINE_OK or DATASETS_OFFLINE_FAILED), writes the
evidence record to --out, and exits 0 or 1. With --require-offline it refuses
to run unless HF_HUB_OFFLINE=1 is set, so an online load can never pass for
the offline path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load(name: str, max_examples: int) -> int:
    """The number of examples the runner's loader returns for ``name``."""
    from src.data.loader import get_loader

    return len(get_loader(name).load(max_examples=max_examples))


def probe(datasets: List[str], max_examples: int, require_offline: bool) -> Dict[str, Any]:
    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    record: Dict[str, Any] = {
        "datasets": {}, "max_examples": max_examples, "hf_hub_offline": offline,
        "require_offline": require_offline, "ok": True, "reason": None,
    }
    if require_offline and not offline:
        record["ok"] = False
        record["reason"] = "HF_HUB_OFFLINE is not 1: an online load proves nothing about the run job's offline path"
        return record
    for name in datasets:
        t0 = time.monotonic()
        try:
            n = _load(name, max_examples)
        except Exception as exc:  # the loader's own refusal or the hub's offline error
            record["datasets"][name] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            record["ok"] = False
            record["reason"] = f"dataset {name} did not load: {type(exc).__name__}"
            return record
        rec = {"ok": n >= 1, "examples": n, "seconds": round(time.monotonic() - t0, 2)}
        record["datasets"][name] = rec
        if n < 1:
            record["ok"] = False
            record["reason"] = f"dataset {name} loaded zero examples"
            return record
    return record


def verdict_line(record: Dict[str, Any]) -> str:
    if record["ok"]:
        loaded = ",".join(f"{k}={v['examples']}" for k, v in record["datasets"].items())
        return f"DATASETS_OFFLINE_OK hf_hub_offline={int(record['hf_hub_offline'])} loaded={loaded}"
    return f"DATASETS_OFFLINE_FAILED reason={record['reason']}"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1].strip())
    parser.add_argument("--datasets", required=True, help="comma-separated roster keys (src.data.loader.get_loader)")
    parser.add_argument("--max-examples", type=int, default=3)
    parser.add_argument("--require-offline", action="store_true", help="refuse unless HF_HUB_OFFLINE=1")
    parser.add_argument("--out", type=Path, default=None, help="evidence JSON record")
    args = parser.parse_args(argv)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if not datasets or args.max_examples < 1:
        print("DATASETS_OFFLINE_FAILED reason=no dataset named or --max-examples < 1")
        return 1
    record = probe(datasets, args.max_examples, args.require_offline)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(verdict_line(record))
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
