#!/usr/bin/env python3
"""
Download HuggingFace datasets for CAGE benchmarking.

Datasets downloaded:
- hotpotqa (primary, multi-hop reasoning)
- allenai/qasper (scientific papers)
- squad_v2 (reading comprehension)
- trivia_qa (multi-evidence questions)
"""

from datasets import load_dataset
import argparse
import sys


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
        else:
            splits = list(dataset.keys()) if hasattr(dataset, 'keys') else []
            print(f"✓ {name} downloaded ({len(splits)} splits: {', '.join(splits)})")
    except Exception as e:
        print(f"✗ Failed to download {name}: {e}", file=sys.stderr)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Download HuggingFace datasets for CAGE evaluation"
    )
    parser.add_argument(
        "--dataset",
        choices=["hotpotqa", "qasper", "squad_v2", "trivia_qa", "all"],
        default="all",
        help="Specific dataset to download (default: all)",
    )
    args = parser.parse_args()

    datasets_to_download = {
        "hotpotqa": ("hotpotqa", "fullwiki"),
        "qasper": ("allenai/qasper", None),
        "squad_v2": ("squad_v2", None),
        "trivia_qa": ("trivia_qa", "rc"),
    }

    if args.dataset == "all":
        selected = datasets_to_download.items()
    else:
        selected = [(args.dataset, datasets_to_download[args.dataset])]

    print("Starting dataset downloads...")
    print("=" * 60)

    for name, (dataset_name, config) in selected:
        try:
            download_dataset(dataset_name, config)
        except Exception:
            print(f"\nWarning: Skipping {name} due to error\n")
            continue

    print("=" * 60)
    print("\nAll requested datasets downloaded!")
    print(f"Cached in: ~/.cache/huggingface/datasets/")


if __name__ == "__main__":
    main()
