#!/usr/bin/env python3
"""CAGE observability sidecar (Option 1: provenance + snapshots + live sync).

Runs as a BACKGROUND process alongside a baseline sweep. It writes a run manifest, then
periodically snapshots GPU / serving telemetry / progress as durable JSON + PNG, and on exit
hashes every result file. It observes the run from OUTSIDE (reads on-disk STATUS/results.csv
for progress), so it is fully decoupled from the orchestrator and cannot perturb serving
timings.

Artifacts land under ``<run-dir>/observability/`` -- which cloud_run.sh already mirrors to
GCS every interval -- so nothing extra is needed to stream them off the VM; a laptop pulls
them with scripts/watch_run.sh.

Launch (cloud_run.sh does this automatically):
    nohup python3 scripts/observe_run.py --run-dir analysis --model Qwen/Qwen3-8B \
        --num-queries 500 --num-trials 3 --interval 30 > logs/observe.log 2>&1 &
Stop: send SIGTERM (the sidecar finalises a last snapshot + provenance hashes), or `touch
<out-dir>/OBSERVE_STOP`.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# Make ``src`` importable when run as a plain script from the repo root.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.observability import (  # noqa: E402
    SnapshotRecorder,
    TraceRecorder,
    build_manifest,
    write_manifest,
    write_provenance,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [observe] %(levelname)s %(message)s"
)
logger = logging.getLogger("cage.observe")

_STOP = threading.Event()


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in text).strip("-").lower()


def _scan_progress(run_dir: Path, expected_total: int | None) -> Dict[str, Any]:
    """Derive live progress by reading the run tree from outside (no orchestrator coupling).

    completed        = total non-header rows across every results.csv found under run_dir
    baselines_done   = number of STATUS files (one per completed baseline cell)
    active_baseline  = directory of the most recently modified results.csv
    """
    completed = 0
    active = None
    newest_mtime = -1.0
    for csv_path in run_dir.glob("**/results.csv"):
        try:
            with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
                n = sum(1 for _ in f)
            completed += max(n - 1, 0)  # minus header
            mtime = csv_path.stat().st_mtime
            if mtime > newest_mtime:
                newest_mtime = mtime
                active = csv_path.parent.name
        except OSError:
            continue
    baselines_done = sum(1 for _ in run_dir.glob("**/STATUS"))
    out: Dict[str, Any] = {
        "completed": completed,
        "baselines_done": baselines_done,
        "active_baseline": active,
    }
    if expected_total:
        out["expected_total"] = expected_total
        out["pct"] = round(100.0 * completed / expected_total, 1) if expected_total else None
    return out


def _install_signal_handlers() -> None:
    def _handler(signum, _frame):  # noqa: ANN001
        logger.info("received signal %s -> finalising", signum)
        _STOP.set()

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def main() -> int:
    ap = argparse.ArgumentParser(description="CAGE observability sidecar")
    ap.add_argument("--run-dir", default="analysis", help="Run output tree to observe/mirror.")
    ap.add_argument("--out-dir", default=None, help="Artifact dir (default <run-dir>/observability).")
    ap.add_argument("--interval", type=float, default=30.0, help="Snapshot interval (s).")
    ap.add_argument("--no-png", action="store_true", help="Disable PNG panels (JSON only).")
    ap.add_argument("--cage-dir", default=str(_REPO), help="CAGE repo dir for git SHA.")
    ap.add_argument("--cage-stats-dir", default=None, help="cage-stats repo dir for git SHA.")
    ap.add_argument("--model", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--num-queries", type=int, default=None)
    ap.add_argument("--num-trials", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--kv-cache-dtype", default=None)
    ap.add_argument("--speculative-config", default=None)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--gpu-memory-utilization", type=float, default=None)
    ap.add_argument("--expected-total", type=int, default=None,
                    help="Expected total queries for a %% bar (else derived: q*trials).")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir / "observability"
    out_dir.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now(timezone.utc).isoformat()
    run_id = f"{_slug(args.model or 'run')}_{created_at.replace(':', '').replace('-', '')[:15]}"

    # Resolve cage-stats dir for its git SHA if not given (matches cloud_run.sh's layout).
    cage_stats_dir = args.cage_stats_dir or os.environ.get("CAGE_STATS_HOME")
    if not cage_stats_dir and (_REPO.parent / "cage-stats").is_dir():
        cage_stats_dir = str(_REPO.parent / "cage-stats")

    expected_total = args.expected_total
    if expected_total is None and args.num_queries and args.num_trials:
        # Rough default: one core suite. Real sweeps have more cells; treat as a lower bound.
        expected_total = args.num_queries * args.num_trials

    # Effective serving config: the harness exports it via _serving_config.sh (VLLM_* env), which
    # IS the ground truth of how vLLM was launched; CLI flags override. Recording these -- esp.
    # gpu_memory_utilization, the swept memory-pressure variable -- ties a result set to its exact
    # serving regime and pressure point.
    def _env_int(name):
        v = os.environ.get(name)
        return int(v) if v not in (None, "") else None

    def _env_float(name):
        v = os.environ.get(name)
        return float(v) if v not in (None, "") else None

    _ee = os.environ.get("VLLM_ENFORCE_EAGER")
    enforce_eager = True if args.enforce_eager else (None if _ee is None else _ee == "1")
    max_model_len = args.max_model_len if args.max_model_len is not None else _env_int("VLLM_MAX_MODEL_LEN")
    gpu_mem_util = (args.gpu_memory_utilization if args.gpu_memory_utilization is not None
                    else _env_float("VLLM_GPU_MEMORY_UTILIZATION"))
    kv_cache_dtype = args.kv_cache_dtype or (os.environ.get("VLLM_KV_CACHE_DTYPE") or None)

    # 1) Reproducibility spine.
    manifest = build_manifest(
        run_id=run_id, created_at=created_at, cage_repo_dir=args.cage_dir,
        cage_stats_repo_dir=cage_stats_dir, model=args.model, dataset=args.dataset,
        num_queries=args.num_queries, num_trials=args.num_trials, seed=args.seed,
        kv_cache_dtype=kv_cache_dtype, speculative_config=args.speculative_config,
        enforce_eager=enforce_eager, max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem_util,
        extra={"run_dir": str(run_dir)},
    )
    write_manifest(manifest, str(out_dir / "run_manifest.json"))
    logger.info("manifest written: sha=%s vllm=%s gpu=%s zone=%s",
                (manifest.cage_git_sha or "?")[:8], manifest.vllm_version,
                (manifest.gpu or {}).get("name"), (manifest.gcp_instance or {}).get("zone"))

    # 2) Trace + periodic snapshots.
    trace = TraceRecorder(str(out_dir / "trace.jsonl"))
    trace.event("observe_start", run_id=run_id, manifest="run_manifest.json")
    recorder = SnapshotRecorder(
        str(out_dir), interval_s=args.interval, render_png=not args.no_png,
        progress_fn=lambda: _scan_progress(run_dir, expected_total),
    )
    recorder.start()

    _install_signal_handlers()
    stop_file = out_dir / "OBSERVE_STOP"
    logger.info("observing run_dir=%s -> out_dir=%s (interval=%.0fs). SIGTERM or touch %s to stop.",
                run_dir, out_dir, args.interval, stop_file)

    # 3) Idle until stopped (the recorder thread does the work).
    while not _STOP.wait(2.0):
        if stop_file.exists():
            logger.info("stop sentinel found -> finalising")
            break

    # 4) Finalise: last snapshot + hash every result file.
    trace.event("observe_stop")
    recorder.stop(final=True)
    prov = write_provenance(
        str(run_dir), str(out_dir / "provenance.json"),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    logger.info("finalised: %d result files hashed -> provenance.json", prov.get("file_count", 0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
