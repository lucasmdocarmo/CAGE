#!/usr/bin/env python3
"""Order:     stage 3 — before the D6 campaign cells it calibrates; needs a live engine ([VERIFY-LIVE at S0])
Objective: Per-cell lambda*/SLO-floor calibration (D6 §6.1): sequential floor + Poisson probe ladder -> calibration JSON for the campaign driver
Cloud:     both

Per-cell lambda*/SLO-floor calibration driver — [VERIFY-LIVE at S0].

Runs the REGISTERED calibration procedure (src/orchestration/calibration.py,
D6 §6.1) against ONE live engine cell and writes the calibration JSON
(``CellCalibration.to_manifest``) that the campaign driver consumes to build
the D6 rate grid and SLO thresholds.

Stages (both use the STREAMING adapter path, never the full-response proxy):
  1. Floor  — FLOOR_N_REQUESTS sequential ``async_stream_generate`` calls at
     concurrency 1; per-request ttft_ms plus tpot_ms derived as
     (total_time_ms - ttft_ms) / (num_tokens - 1); summarized by
     ``summarize_floor`` (medians, seconds).
  2. Probe  — geometric rate ladder (``geometric_rate_ladder``); each rate is
     driven open-loop Poisson for PROBE_WINDOW_S via the D6 dispatcher
     (``generate_arrival_schedule`` + ``OpenLoopDispatcher``), warmup-trimmed
     (``trim_to_measurement_window``); ``decide_lambda_star`` applies the
     registered decision rule.

Calibration data NEVER enters confirmatory analysis (module doctrine in
src/orchestration/calibration.py). Request replay across a probe window is
therefore acceptable here — the E4 no-replay pin binds MEASURED confirmatory
windows, not calibration probes.

Status: code-complete skeleton, NOT yet run against a live engine. Every
adapter interaction below is [VERIFY-LIVE at S0]. Fail-closed throughout:
any incomplete telemetry raises CalibrationError instead of degrading.

Usage:
  .venv/bin/python scripts/3_run/calibrate_cell.py \\
      --backend vllm --model Qwen/Qwen3-8B --api-base http://localhost:8000 \\
      --manifest results/manifests/nq_500x3.json \\
      --budget-fraction 0.5 --start-qps 0.5 \\
      --output results/calibration/vllm_qwen3-8b_b0.5.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Pure-stdlib module: safe to import at top level (keeps this script importable
# in the lean analysis venv; adapters / numpy import lazily inside functions).
from src.orchestration.calibration import (
    FLOOR_N_REQUESTS,
    PROBE_BISECT_STEPS,
    PROBE_WARMUP_S,
    PROBE_WINDOW_S,
    START_QPS_RULE,
    CalibrationError,
    CellCalibration,
    FloorMeasurement,
    LambdaStarEstimate,
    PROBE_ATTAINMENT_MIN,
    ProbeStep,
    decide_lambda_star,
    floor_start_qps,
    geometric_rate_ladder,
    summarize_floor,
)

_BACKENDS = ("vllm", "sglang", "lmdeploy")

# Fixed decode cap for every calibration request: calibration needs a stable,
# representative decode length, not task answers. Explicit constant so the
# calibration JSON's provenance is reproducible from the CLI alone.
CAL_MAX_TOKENS = 256

# Base seed for the probe windows' pre-drawn Poisson arrivals (seed + step
# index per ladder rung). Calibration-only; confirmatory windows draw their
# own registered seeds.
CAL_SEED = 20260812


def build_adapter(backend: str, model: str, api_base: str) -> Any:
    """Construct the matching src.inference streaming adapter."""
    if backend == "vllm":
        from src.inference.vllm_adapter import VLLMAdapter as Adapter
    elif backend == "sglang":
        from src.inference.sglang_adapter import SGLangAdapter as Adapter
    elif backend == "lmdeploy":
        from src.inference.lmdeploy_adapter import LMDeployAdapter as Adapter
    else:
        raise CalibrationError("backend", backend, f"must be one of {_BACKENDS}")
    return Adapter(model_name=model, api_base=api_base)


def build_requests(manifest_path: str) -> List[Any]:
    """Minimal calibration requests from the query manifest's corpus blocks.

    The manifest (src/data/manifest.py) stores block TEXTS verbatim; each
    block text becomes one raw-completion request, giving calibration the
    same prompt-length profile the cell will serve. Minimal request shape
    duplicated from ``build_open_loop_request`` in
    scripts/3_run/run_experiment.py (~line 2377) — run_experiment is a heavy
    script and is deliberately NOT imported here.
    """
    from src.inference.engine import InferenceRequest

    path = Path(manifest_path)
    if not path.is_file():
        raise CalibrationError("manifest", manifest_path, "file not found")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    blocks = manifest.get("blocks")
    if not blocks:
        raise CalibrationError(
            "manifest", manifest_path, "has no 'blocks' (need a CAGE query manifest)"
        )
    requests: List[Any] = []
    for block in blocks:
        text = block.get("text")
        if not isinstance(text, str) or not text.strip():
            raise CalibrationError(
                "manifest", block.get("block_id"), "block without verbatim 'text'"
            )
        requests.append(
            InferenceRequest(
                prompt=text,
                max_tokens=CAL_MAX_TOKENS,
                temperature=0.0,
                top_p=0.95,
                request_id=f"cal-block-{block.get('block_id', len(requests))}",
            )
        )
    return requests


async def measure_floor(adapter: Any, requests: Sequence[Any]) -> FloorMeasurement:
    """Floor stage: FLOOR_N_REQUESTS sequential streamed single-stream calls."""
    ttft_ms: List[float] = []
    tpot_ms: List[float] = []
    for i in range(FLOOR_N_REQUESTS):
        response = await adapter.async_stream_generate(requests[i % len(requests)])
        if response.error:
            raise CalibrationError(
                "floor_request",
                response.error,
                f"request {i} failed — the floor needs {FLOOR_N_REQUESTS} clean "
                f"sequential completions (drop-nothing)",
            )
        if response.num_tokens < 2:
            raise CalibrationError(
                "num_tokens",
                response.num_tokens,
                f"request {i}: TPOT needs >= 2 generated tokens",
            )
        ttft_ms.append(float(response.ttft_ms))
        tpot_ms.append(
            (float(response.total_time_ms) - float(response.ttft_ms))
            / (response.num_tokens - 1)
        )
    return summarize_floor(ttft_ms, tpot_ms)


async def probe_rate(
    adapter: Any, requests: Sequence[Any], rate_qps: float, seed: int,
    *, phase: str = "ladder",
) -> ProbeStep:
    """One probe window: open-loop Poisson dispatch, warmup-trimmed counts."""
    from src.orchestration.load_generator import (
        OpenLoopDispatcher,
        generate_arrival_schedule,
        trim_to_measurement_window,
    )

    schedule = generate_arrival_schedule(
        rate_qps, seed=seed, duration_s=PROBE_WINDOW_S
    )

    async def send(index: int, on_first_token: Any) -> Any:
        # Modulo wrap over the prepared set: replay is acceptable in
        # calibration probes (non-confirmatory by registration; E4 binds
        # measured windows only).
        return await adapter.async_stream_generate(
            requests[index % len(requests)], on_first_token=on_first_token
        )

    report = await OpenLoopDispatcher().run(schedule, send)
    kept = trim_to_measurement_window(report.records, warmup_s=PROBE_WARMUP_S)
    if not kept:
        raise CalibrationError(
            "kept_arrivals",
            0,
            f"no arrivals in the post-warmup window at {rate_qps} qps — "
            f"window too short for this rate",
        )
    n_completed = sum(
        1
        for r in kept
        if r.completed and r.result is not None and not getattr(r.result, "error", None)
    )
    return ProbeStep(
        rate_qps=float(rate_qps),
        n_scheduled=len(kept),
        n_completed=n_completed,
        throughput_rps=n_completed / (PROBE_WINDOW_S - PROBE_WARMUP_S),
        phase=phase,
    )


async def run_probe_ladder(
    adapter: Any, requests: Sequence[Any], start_qps: float
) -> LambdaStarEstimate:
    """Probe stage: climb until the first unsustainable rung, then bisect.

    ADR-0122: the climb runs the geometric ladder from ``start_qps`` until
    the FIRST unsustainable rung (attainment < PROBE_ATTAINMENT_MIN, the only
    test since ADR-0121) or the PROBE_MAX_STEPS ceiling. Once bracketed
    (a sustainable rung immediately below the failing one), PROBE_BISECT_STEPS
    midpoint windows narrow the bracket; each one moves the bracket's lower
    edge up (sustainable) or its upper edge down (unsustainable). The
    REGISTERED decision is ``decide_lambda_star``'s alone, applied to every
    probed step sorted by rate (the rule wants a ladder, not a bag; pass and
    fail are monotone in rate, so sorting changes nothing but the order).
    """
    steps: List[ProbeStep] = []
    for k, rate in enumerate(geometric_rate_ladder(start_qps)):
        print(f"[calibrate] probe {k + 1}: {rate:.4g} qps for {PROBE_WINDOW_S:.0f}s ...")
        step = await probe_rate(adapter, requests, rate, seed=CAL_SEED + k)
        steps.append(step)
        print(
            f"[calibrate]   attainment={step.attainment:.3f} "
            f"throughput={step.throughput_rps:.4g} rps"
        )
        # ADR-0121: attainment alone decides (the same test as
        # decide_lambda_star); a throughput fall between rungs is the
        # Poisson arrival draw, never a stop signal.
        if step.attainment < PROBE_ATTAINMENT_MIN:
            break  # saturation bracketed; no need to probe higher rates
    # ADR-0122: refine the bracket [last sustainable, first unsustainable].
    if len(steps) >= 2 and steps[-1].attainment < PROBE_ATTAINMENT_MIN:
        lo, hi = steps[-2].rate_qps, steps[-1].rate_qps
        for j in range(PROBE_BISECT_STEPS):
            mid = (lo + hi) / 2.0
            print(
                f"[calibrate] bisect {j + 1}/{PROBE_BISECT_STEPS}: {mid:.4g} qps "
                f"(bracket {lo:.4g} to {hi:.4g}) for {PROBE_WINDOW_S:.0f}s ..."
            )
            step = await probe_rate(
                adapter, requests, mid, seed=CAL_SEED + 1000 + j, phase="bisect"
            )
            steps.append(step)
            print(
                f"[calibrate]   attainment={step.attainment:.3f} "
                f"throughput={step.throughput_rps:.4g} rps"
            )
            if step.attainment >= PROBE_ATTAINMENT_MIN:
                lo = mid
            else:
                hi = mid
    return decide_lambda_star(sorted(steps, key=lambda s: s.rate_qps))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Registered lambda*/SLO-floor calibration for one "
        "model×engine×budget cell (D6 §6.1). [VERIFY-LIVE at S0]"
    )
    parser.add_argument("--backend", required=True, choices=_BACKENDS)
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument("--api-base", required=True, help="engine server base URL")
    parser.add_argument(
        "--manifest", required=True, help="CAGE query-manifest JSON (prompt blocks)"
    )
    parser.add_argument(
        "--output", required=True, help="calibration JSON output path"
    )
    parser.add_argument(
        "--start-qps",
        type=float,
        default=None,
        help="first probe rate (operator prior, recorded as such). Unset = "
             "ADR-0122 default: the floor's single-stream service rate, "
             "1 / (ttft + (max_tokens - 1) x tpot), derived after the floor stage",
    )
    parser.add_argument(
        "--budget-fraction",
        type=float,
        required=True,
        help="KV-budget fraction identifying the cell, in (0, 1]",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    requests = build_requests(args.manifest)
    adapter = build_adapter(args.backend, args.model, args.api_base)
    print(
        f"[calibrate] cell: backend={args.backend} model={args.model} "
        f"budget_fraction={args.budget_fraction} ({len(requests)} manifest blocks)"
    )

    print(f"[calibrate] floor: {FLOOR_N_REQUESTS} sequential streamed requests ...")
    floor = asyncio.run(measure_floor(adapter, requests))
    print(
        f"[calibrate] floor: ttft={floor.ttft_s:.4f}s tpot={floor.tpot_s:.5f}s "
        f"({floor.statistic} of {floor.n_requests})"
    )

    # ADR-0122: the first rung comes from the floor unless the operator gave one.
    if args.start_qps is None:
        start_qps = floor_start_qps(floor, max_tokens=CAL_MAX_TOKENS)
        start_source = START_QPS_RULE
    else:
        start_qps = float(args.start_qps)
        start_source = "operator"
    print(f"[calibrate] start rung: {start_qps:.4g} qps (source: {start_source})")

    estimate = asyncio.run(run_probe_ladder(adapter, requests, start_qps))
    print(
        f"[calibrate] lambda*: label={estimate.label} "
        f"lambda_star_qps={estimate.lambda_star_qps}"
    )

    calibration = CellCalibration(
        model=args.model,
        engine=getattr(adapter, "engine_id", args.backend),
        budget_fraction=args.budget_fraction,
        floor=floor,
        lambda_star=estimate,
        start_qps=start_qps,
        start_qps_source=start_source,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(calibration.to_manifest(), indent=2) + "\n", encoding="utf-8"
    )
    print(f"[calibrate] wrote {out_path}")

    if estimate.label != "ESTIMATED":
        # Fail closed: an unbracketed lambda* must never seed a rate grid.
        print(
            f"[calibrate] FAIL: lambda* not bracketed ({estimate.label}) — "
            f"extend the ladder (LADDER_EXHAUSTED) or lower --start-qps "
            f"(NONE_SUSTAINABLE); never extrapolate",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
