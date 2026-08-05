#!/usr/bin/env python3
"""
Download (stage) HuggingFace datasets for CAGE benchmarking.

Campaign datasets (`--dataset all`):
- hotpot_qa, distractor config (multi-hop reasoning; gold + distractor paragraphs)
- allenai/qasper (scientific papers; charter D5 item 4 — full-paper loader)
- squad_v2 (reading comprehension + abstention axis)
- trivia_qa (multi-evidence questions)
- nq_open (Natural Questions, open-domain)
- dgslibisey/MuSiQue (multi-hop, private-evidence pole)
- CRAG / ShareGPT (mirror paths via CAGE_CRAG_HF_PATH / CAGE_SHAREGPT_HF_PATH)
- microsoft/SCBench, scbench_kv + scbench_qa_eng configs (charter D5 item 6 —
  the scoped two-subset external-validation slice)

Instrument-calibration anchors (`--dataset calibration`, or individually —
staged here for the D8 §8.6(a) public-benchmark anchor step; consumed by the
instrument-calibration workstream, not by campaign cells):
- ragtruth: RAGTruth (Niu et al. 2024), LettuceDetect's training/eval anchor
  (mirror path via CAGE_RAGTRUTH_HF_PATH — VERIFY-LIVE before calibration)
- true: TRUE-benchmark (Honovich et al. 2022) anchor subsets. TRUE has no
  single official HF distribution, so the slice is configured as
  CAGE_TRUE_HF_SPECS="path[:config],path[:config],..." defaulting to three
  HF-hosted TRUE constituent tasks (VitaminC, PAWS, FEVER).

RULER (charter D5 item 5) is deliberately absent: it is a synthetic generated
instrument (src/data/ruler.py), never downloaded.

Every download asserts non-empty splits (fail-closed: a silently-empty split is
treated as a failed download and flips the exit code).
"""

from datasets import load_dataset
import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

# One dataset key -> ordered list of (hf_path, config) downloads. ALL specs of
# a key must succeed for the key to count as staged.
DatasetSpecs = Dict[str, List[Tuple[str, Optional[str]]]]

#: Charter campaign roster (what `--dataset all` stages).
CAMPAIGN_KEYS = [
    "hotpotqa", "qasper", "squad_v2", "trivia_qa", "natural_questions",
    "musique", "crag", "sharegpt", "scbench",
]
#: Instrument-calibration anchors (staged only on explicit request).
CALIBRATION_KEYS = ["ragtruth", "true"]


def _true_anchor_specs() -> List[Tuple[str, Optional[str]]]:
    """Parse CAGE_TRUE_HF_SPECS ("path[:config],..." — HF paths never contain
    ':') into (path, config) tuples; default = three HF-hosted TRUE-benchmark
    constituent tasks (Honovich et al. 2022): VitaminC, PAWS, FEVER."""
    raw = os.getenv(
        "CAGE_TRUE_HF_SPECS",
        "tals/vitaminc,paws:labeled_final,fever:v1.0",
    )
    specs: List[Tuple[str, Optional[str]]] = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        path, _, config = entry.partition(":")
        specs.append((path, config or None))
    return specs


def dataset_specs() -> DatasetSpecs:
    """Build the staging map (env vars resolved at call time, not import time)."""
    scbench_path = os.getenv("CAGE_SCBENCH_HF_PATH", "microsoft/SCBench")
    return {
        # Distractor config matches HotpotQALoader: 10 paragraphs/item (2 gold +
        # 8 distractors) so retrieval arms have a real selection job.
        "hotpotqa": [("hotpot_qa", "distractor")],
        "qasper": [("allenai/qasper", None)],
        "squad_v2": [("squad_v2", None)],
        "trivia_qa": [("trivia_qa", "rc")],
        "natural_questions": [("nq_open", None)],
        "musique": [("dgslibisey/MuSiQue", None)],
        # CRAG + ShareGPT HF paths vary across mirrors; override with
        # CAGE_CRAG_HF_PATH / CAGE_SHAREGPT_HF_PATH. Validate the exact schema
        # with a 5-query smoke test.
        "crag": [(os.getenv("CAGE_CRAG_HF_PATH", "crag"), None)],
        "sharegpt": [(os.getenv("CAGE_SHAREGPT_HF_PATH", "RyokoAI/ShareGPT52K"), None)],
        # Charter D5#6 two-subset slice — BOTH configs staged together so the
        # SCBenchLoader (src/data/scbench.py) can fail closed on neither.
        "scbench": [(scbench_path, "scbench_kv"), (scbench_path, "scbench_qa_eng")],
        # D8 §8.6(a) calibration anchors (staging only; consumed by the
        # instrument-calibration workstream). Mirror paths are env-overridable
        # and must be VERIFY-LIVE'd before calibration.
        "ragtruth": [(os.getenv("CAGE_RAGTRUTH_HF_PATH", "KRLabsOrg/ragtruth"), None)],
        "true": _true_anchor_specs(),
    }


def download_dataset(name: str, config: str = None, split: str = None) -> None:
    """Download a single dataset from HuggingFace."""
    print(f"Downloading {name}" + (f" ({config})" if config else "") + "...")
    try:
        if config:
            dataset = load_dataset(name, config, split=split)
        else:
            dataset = load_dataset(name, split=split)

        if split:
            print(f"✓ {name} downloaded ({len(dataset)} examples in {split} split)")
            assert len(dataset) > 0, f"{name}/{split} loaded but is empty"
        else:
            splits = list(dataset.keys()) if hasattr(dataset, 'keys') else []
            print(f"✓ {name} downloaded ({len(splits)} splits: {', '.join(splits)})")
            for split_name in splits:
                assert len(dataset[split_name]) > 0, f"{name}/{split_name} loaded but is empty"
    except Exception as e:
        print(f"✗ Failed to download {name}: {e}", file=sys.stderr)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download HuggingFace datasets for CAGE evaluation"
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(set(CAMPAIGN_KEYS + CALIBRATION_KEYS)) + ["all", "calibration"],
        default="all",
        help=(
            "Dataset to stage. 'all' = the charter campaign roster "
            f"({', '.join(CAMPAIGN_KEYS)}); 'calibration' = the D8 §8.6(a) "
            f"instrument anchors ({', '.join(CALIBRATION_KEYS)})."
        ),
    )
    args = parser.parse_args()

    specs = dataset_specs()
    if args.dataset == "all":
        selected = [(k, specs[k]) for k in CAMPAIGN_KEYS]
    elif args.dataset == "calibration":
        selected = [(k, specs[k]) for k in CALIBRATION_KEYS]
    else:
        selected = [(args.dataset, specs[args.dataset])]

    print("Starting dataset downloads...")
    print("=" * 60)

    failed = []
    for name, spec_list in selected:
        for dataset_name, config in spec_list:
            try:
                download_dataset(dataset_name, config)
            except Exception as e:
                print(f"\nWarning: Skipping {name} ({dataset_name}) due to error\n")
                failed.append((name, e))
                continue

    print("=" * 60)
    if failed:
        print(
            f"\n{len(failed)} dataset(s) failed: {[n for n, _ in failed]}",
            file=sys.stderr,
        )
        return 1

    print("\nAll requested datasets downloaded!")
    print(f"Cached in: ~/.cache/huggingface/datasets/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
