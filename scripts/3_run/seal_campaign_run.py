#!/usr/bin/env python3
"""
Order:     stage 3 — after the LAST tree of a campaign run, before verify_results.py / any pull
Objective: Seal ONE campaign run root (write-time-hash cross-check -> §5 ledger.json via campaign_layout.seal_run)
Cloud:     both

Campaign run sealer (task #116 seam; docs/RESULTS_LAYOUT.md §5).

Run it ONCE, on the node, after every tree that writes into the campaign root
has finished and BEFORE any analysis touches the data:

    python3 scripts/3_run/seal_campaign_run.py <run_root>

It (1) cross-checks every sealed-scope artifact (manifest.json + everything
under cells/) against the run's append-only write-time hash journal
(write_time_hashes.jsonl — S0-15: "hash ledger written at write time";
a file that changed after its write-time hash, or was written outside the
campaign writer, REFUSES the seal with every problem listed), then
(2) writes the one §5 ledger.json via campaign_layout.seal_run — which itself
refuses to overwrite an existing seal (a re-run is a new run_id) and verifies
the fresh seal in place. Exit 0 = sealed; nonzero = refused, nothing written.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.orchestration.campaign_layout import CampaignLayoutError  # noqa: E402
from src.orchestration.campaign_session import (  # noqa: E402
    CampaignSessionError,
    seal_campaign_run,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Seal a campaign run root: write-time-hash journal cross-check, "
            "then the §5 ledger.json (campaign_layout.seal_run). Run once, at "
            "run end, before any analysis."
        )
    )
    parser.add_argument(
        "run_root",
        type=Path,
        help="campaign run root: results/<campaign>/<session>/<run_id>",
    )
    args = parser.parse_args(argv)
    try:
        ledger_path = seal_campaign_run(args.run_root)
    except (CampaignSessionError, CampaignLayoutError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"[seal_campaign_run] sealed: {ledger_path}")
    print(
        "[seal_campaign_run] next: scripts/4_analysis/verify_results.py "
        f"{args.run_root}  (the v2 gate must be GREEN before the pull)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
