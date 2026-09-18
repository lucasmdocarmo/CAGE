#!/usr/bin/env python3
"""Order:     stage 1 — once per (dataset, N, T, seed), before any runner; consumed by every cell via CAGE_QUERY_MANIFEST
Objective: Build the uniform-yardstick query manifest so all cells/engines/models measure the SAME query set
Cloud:     both

Build the uniform-yardstick query manifest for a dataset (see src/data/manifest.py).

Run ONCE per (dataset, N, T, seed); every runner then loads the SAME measured query
set via CAGE_QUERY_MANIFEST, so pairing holds across all cells/engines/models.

Usage:
  python3 scripts/1_setup/build_query_manifest.py --dataset squad_v2 \
      --num-queries 500 --num-trials 3 --seed 42
  -> data/manifests/squad_v2_500x3_seed42.json  (+ prints the stats block)

Engineered-overlap store (D5 F1/F3, e.g. the HotpotQA store; charter: overlap
"ENGINEERED and reported (never assumed)"):
  python3 scripts/1_setup/build_query_manifest.py --dataset hotpotqa \
      --num-queries 500 --num-trials 3 --seed 42 --overlap-target 0.33
  -> data/manifests/hotpotqa_500x3_seed42_ov0.33.json
Every manifest (natural mode included) carries the MEASURED realized overlap in
its "overlap" field; --overlap-target additionally gates fail-closed (a target
the corpus cannot realize refuses with realized-vs-target, no silent best-effort).

B12 corpus-truncation ladder (charter §7.7(d), ADR-0106): every manifest also
carries "trunc_rungs" for --trunc-budgets (default "1400,700"; "" = none): the
same packed blocks truncated in packing order at each descending rung, every
pool query labeled in-corpus / out-of-corpus by construction. A rung >= the
block budget refuses (that point of the ladder is B3's own cell).

Needs the `datasets` package (loads the real split); pure CPU, no GPU/serving.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def parse_trunc_budgets(raw: str) -> Tuple[int, ...]:
    """'1400,700' -> (1400, 700); '' -> (). Malformed entries refuse loudly."""
    text = (raw or "").strip()
    if not text:
        return ()
    out: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            raise SystemExit(f"--trunc-budgets: {part!r} is not an integer") from None
    return tuple(out)


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
    p.add_argument("--overlap-target", type=float, default=None,
                   help="Engineer the store: target mean pairwise shared-paragraph "
                        "fraction per query group (0..1). Default: natural mode "
                        "(overlap still measured and written to the manifest).")
    p.add_argument("--overlap-tolerance", type=float, default=0.05,
                   help="Engineered mode: max |realized - target| before the build "
                        "REFUSES (fail-closed; default 0.05).")
    p.add_argument("--overlap-group-size", type=int, default=4,
                   help="Engineered mode: questions per planned query group "
                        "(>= 2; default 4).")
    p.add_argument("--trunc-budgets", default="1400,700",
                   help="B12 ladder (ADR-0106): comma list of DESCENDING corpus "
                        "rung budgets, each < --block-budget (default '1400,700'; "
                        "'' disables). The manifest then serves every corpus-trunc "
                        "rung; a rung >= the block budget REFUSES.")
    p.add_argument("--out", default=None,
                   help="Default: data/manifests/<dataset>_<N>x<T>_seed<seed>"
                        "[_ov<target>].json")
    args = p.parse_args()
    trunc_budgets = parse_trunc_budgets(args.trunc_budgets)

    from src.data.loader import get_loader, gold_only
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
        # Strip distractor paragraphs before packing (HotpotQA/MuSiQue keep gold +
        # distractors in .context; no-op for SQuAD-shaped loaders). Without this,
        # unique-per-question distractor text pollutes the shared corpus budget and
        # each block degenerates to ~1 example (see src/data/manifest.py docstring).
        context_selector=gold_only,
        overlap_target=args.overlap_target,
        overlap_tolerance=args.overlap_tolerance,
        overlap_group_size=args.overlap_group_size,
        trunc_budgets=trunc_budgets,
    )

    # An engineered store is a DIFFERENT artifact than the natural manifest for
    # the same (dataset, N, T, seed): tag the default filename so they never collide.
    ov_tag = f"_ov{args.overlap_target:g}" if args.overlap_target is not None else ""
    out = Path(args.out) if args.out else (
        REPO_ROOT / "data" / "manifests"
        / f"{args.dataset}_{args.num_queries}x{args.num_trials}_seed{args.seed}{ov_tag}.json"
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
    ov = manifest["overlap"]
    m = ov["measured"]
    frac = m["mean_pairwise_shared_fraction"]
    print(f"  overlap mode={ov['mode']} target={ov['target_shared_fraction']} "
          f"measured_mean_shared_fraction="
          f"{'None' if frac is None else f'{frac:.3f}'} "
          f"pairs={m['n_pairs']} multi_question_groups={m['n_multi_question_groups']}")
    for b_str, rung in manifest["trunc_rungs"].items():
        print(f"  trunc rung {b_str}: in_corpus={rung['n_in_corpus']} "
              f"out_of_corpus={rung['n_out_of_corpus']}")
    print("  export CAGE_QUERY_MANIFEST=" + str(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
