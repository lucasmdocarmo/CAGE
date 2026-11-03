#!/usr/bin/env python3
"""Order:     stage 14 (analyze) of cage_experiment.sh, after the yield table (S2 block F row F-7, 2026-11-03)
Objective: write the retrieval-stage sidebar of a pulled landing (gold-in-prompt rate per dataset, baseline and engine over the F1 windows of the retrieval arms, Wilson 95 percent bounds, the retrieved-but-cut count, and the paired B5 against B6 with the exact McNemar p) as retrieval_stage.md and retrieval_stage.json under --out
Cloud:     local
           (the Mac; reads the pulled landing)

Exit codes: 0 the two files written; 3 the named refusal (no F1 window of a retrieval
arm; the master runs this step with run_step "0|3"); 2 a missing or unreadable
requests.jsonl, named on stderr. The logic lives in src/analysis/retrieval_stage.py.

Usage:
    python scripts/4_analysis/retrieval_stage.py <run_root> --out <dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.analysis.retrieval_stage import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
