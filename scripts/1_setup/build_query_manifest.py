#!/usr/bin/env python3
"""Build the uniform-yardstick query manifest for a dataset (see src/data/manifest.py).

Run ONCE per (dataset, N, T, seed); every runner then loads the SAME measured query
set via CAGE_QUERY_MANIFEST, so pairing holds across all cells/engines/models.

Usage:
  python3 scripts/1_setup/build_query_manifest.py --dataset squad_v2 \
      --num-queries 500 --num-trials 3 --seed 42
  -> data/manifests/squad_v2_500x3_seed42.json  (+ prints the stats block)

Needs the `datasets` package (loads the real split); pure CPU, no GPU/serving.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description="Build the uniform query manifest.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default=None, help="Default: the loader's default split.")
    p.add_argument("--num-queries", type=int, default=500)
    p.add_argument("--num-trials", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--block-budget", type=int, default=2800)
    p.add_argument("--pool-target", type=int, default=None,
                   help="In-corpus pool size to pack (default max(3N, N*T)).")
    p.add_argument("--max-load", type=int, default=None,
                   help="Cap on examples loaded from the split (default: all).")
    p.add_argument("--out", default=None,
                   help="Default: data/manifests/<dataset>_<N>x<T>_seed<seed>.json")
    args = p.parse_args()

    from src.data.loader import get_loader
    from src.data.manifest import build_manifest

    kwargs = {"seed": args.seed}
    if args.split:
        kwargs["split"] = args.split
    loader = get_loader(args.dataset, **kwargs)
    examples = loader.load(max_examples=args.max_load)
    split = args.split or getattr(loader, "split", "")

    manifest = build_manifest(
        examples,
        num_queries=args.num_queries,
        num_trials=args.num_trials,
        seed=args.seed,
        block_budget=args.block_budget,
        pool_target=args.pool_target,
        dataset=args.dataset,
        split=split,
    )

    out = Path(args.out) if args.out else (
        REPO_ROOT / "data" / "manifests"
        / f"{args.dataset}_{args.num_queries}x{args.num_trials}_seed{args.seed}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    s = manifest["stats"]
    print(f"MANIFEST_BUILT -> {out}")
    print(f"  dataset={args.dataset} split={split} N={args.num_queries} T={args.num_trials} "
          f"seed={args.seed} budget={args.block_budget}")
    print(f"  blocks={s['n_blocks']} pool={s['pool_size']} loaded={s['source_examples_loaded']} "
          f"excluded={s['examples_excluded']} ({s['exclusion_rate']:.1%}) "
          f"trials_disjoint={s['trials_disjoint']}")
    print("  export CAGE_QUERY_MANIFEST=" + str(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
