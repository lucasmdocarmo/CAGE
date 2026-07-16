"""vLLM serving telemetry for CAGE, via the `cage-stats` package.

Captures a full `/metrics` snapshot that CAGE's own metrics do NOT expose —
speculative-decode acceptance, KV-compression ratio + dtype, prompt-token source
breakdown (compute / cache-hit / external KV transfer), prefix-cache hit rate, and
multi-vendor GPU stats — and can print a one-shot terminal dashboard.

This closes the telemetry gaps flagged in docs/DEV_BACKLOG.md / FEATURE_MAP.md (e.g.
speculative acceptance "via /metrics", and compressed_cag KV-compression telemetry).

Resolution order (so CAGE never hard-fails if cage-stats isn't present):
  1. in-process `cage_stats.api` import (richest; install with `pip install -e <cage-stats repo>`)
  2. `cage-stats --once --json` CLI subprocess
  3. graceful skip -> returns None
Set CAGE_STATS_HOME to the cage-stats repo path to enable the in-process path without
installing it.
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from typing import Optional


def _try_import_api():
    """Return cage_stats.api if importable, else None (honouring CAGE_STATS_HOME)."""
    try:
        import cage_stats.api as api  # type: ignore
        return api
    except Exception:
        home = os.getenv("CAGE_STATS_HOME")
        if home and os.path.isdir(home) and home not in sys.path:
            sys.path.insert(0, home)
            try:
                import cage_stats.api as api  # type: ignore
                return api
            except Exception:
                return None
        return None


def capture_snapshot(
    url: str,
    *,
    metrics_path: str = "/metrics",
    api_key: Optional[str] = None,
    interval: float = 1.0,
) -> Optional[dict]:
    """Return the full vLLM telemetry snapshot as a dict, or None if unavailable.

    Reads LIVE vLLM telemetry only. There is no synthetic/mock path: CAGE must never
    record fabricated numbers, so an unavailable server yields None, never fake data.
    """
    api = _try_import_api()
    if api is not None:
        try:
            return api.snapshot_dict(
                url, metrics_path=metrics_path, api_key=api_key, interval=interval
            )
        except Exception as e:
            print(f"[telemetry] cage_stats in-process capture failed: {e}")
    exe = shutil.which("cage-stats")
    if exe:
        try:
            cmd = [exe, "--once", "--json", "--url", url]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if res.returncode == 0 and res.stdout.strip():
                return json.loads(res.stdout)
            print(f"[telemetry] cage-stats CLI returned {res.returncode}: {res.stderr.strip()[:160]}")
        except Exception as e:
            print(f"[telemetry] cage-stats subprocess failed: {e}")
    # Dependency-free fallback: at minimum capture speculative-decode acceptance from /metrics.
    spec = scrape_spec_decode(url, metrics_path=metrics_path)
    return {"spec_decode": spec} if spec else None


class VllmTelemetrySampler:
    """Threaded sampler that polls vLLM telemetry DURING a workload, then aggregates.

    A single ``capture_snapshot()`` taken after a run reads the server at idle, so the
    instantaneous rates (gen/prompt tps, running, kv_usage) come back ~0. This samples
    every ``interval`` seconds across the workload and summarizes peak + mean rates plus
    the final cumulative counters, so ``vllm_telemetry.json`` reflects the ACTIVE run.

    Usage:
        s = VllmTelemetrySampler(url).start()
        ... run workload ...
        s.stop()
        agg = s.aggregate()   # dict, or None if nothing was captured
    """

    # gauges/rates -> peak + mean; counters -> final (max, they are monotonic);
    # structural/last-value fields -> taken from the final sample.
    # session_* are NOT counters (2026-07-15 review, M2): each sampler tick builds a
    # fresh engine, so they are ~1s per-tick DELTAS -- "max" made the busiest single
    # tick masquerade as a trial total (the audit's "session_gen_tokens 12-49" was a
    # tick, not the trial). As gauges they honestly report peak/avg PER-TICK activity.
    _GAUGES = ("gen_tps", "prompt_tps", "req_rate", "running", "waiting",
               "kv_usage", "kv_used_tokens", "tokens_per_iter", "preempt_rate",
               "session_gen_tokens", "session_prompt_tokens", "session_requests")
    _COUNTERS = ("cached_tokens_total", "recomputed_tokens_total")
    # Speculative-decode acceptance reaches us under TWO schemas depending on the path:
    # the cage-stats in-process/CLI path emits FLAT keys (spec_active/spec_acceptance/
    # spec_accepted_per_draft); the dependency-free stdlib fallback emits a nested
    # "spec_decode" dict. Whitelist BOTH so acceptance is promoted to the top level of
    # vllm_telemetry.json (Phase-2 bug: only "spec_decode" was listed, so the flat
    # cage-stats acceptance was silently dropped -> "None for every speculative cell").
    _LAST = ("connected", "model_names", "engine_count", "kv_capacity_tokens",
             "kv_dtype", "kv_ratio", "kv_ratio_kind",
             "prefix_hit_lifetime", "prefix_hit_window",
             "src_compute", "src_cache_hit", "src_external",
             "spec_decode", "spec_active", "spec_acceptance", "spec_accepted_per_draft")

    def __init__(self, url: str, *, interval: float = 1.0,
                 metrics_path: str = "/metrics"):
        self.url = url
        self.interval = max(0.25, float(interval))
        self.metrics_path = metrics_path
        self._samples: list = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "VllmTelemetrySampler":
        if self._thread is not None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="vllm-telemetry-sampler", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.time()
            try:
                snap = capture_snapshot(self.url, metrics_path=self.metrics_path)
                if snap:
                    self._samples.append(snap)
            except Exception:
                pass
            dt = self.interval - (time.time() - t0)
            if dt > 0:
                self._stop.wait(dt)

    def stop(self) -> "VllmTelemetrySampler":
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=self.interval + 5)
            self._thread = None
        return self

    def aggregate(self) -> Optional[dict]:
        samples = [s for s in self._samples if isinstance(s, dict)]
        if not samples:
            return None
        agg: dict = {"sampled": True, "num_samples": len(samples)}
        for k in self._GAUGES:
            vals = [s.get(k) for s in samples if isinstance(s.get(k), (int, float))]
            if vals:
                agg[f"{k}_peak"] = round(max(vals), 4)
                agg[f"{k}_avg"] = round(statistics.fmean(vals), 4)
        for k in self._COUNTERS:
            vals = [s.get(k) for s in samples if isinstance(s.get(k), (int, float))]
            if vals:
                agg[k] = max(vals)  # monotonic counters: final value
        last = samples[-1]
        for k in self._LAST:
            if last.get(k) is not None:
                agg[k] = last[k]
        # Normalize ONE canonical top-level acceptance rate regardless of which path
        # produced the samples, so downstream readers/plots have a single stable field.
        # Counters are monotonic, so the LAST sample carries the cumulative acceptance.
        if agg.get("spec_acceptance") is not None:
            agg["spec_decode_acceptance_rate"] = agg["spec_acceptance"]
        elif isinstance(agg.get("spec_decode"), dict):
            rate = agg["spec_decode"].get("spec_decode_acceptance_rate")
            if rate is not None:
                agg["spec_decode_acceptance_rate"] = rate
        agg["final_snapshot"] = last
        return agg


def dashboard_text(
    url: str,
    *,
    metrics_path: str = "/metrics",
    api_key: Optional[str] = None,
    interval: float = 1.0,
) -> Optional[str]:
    """Return a one-shot static terminal dashboard string, or None if unavailable."""
    api = _try_import_api()
    if api is not None:
        try:
            return api.dashboard_text(
                url, metrics_path=metrics_path, api_key=api_key, interval=interval
            )
        except Exception as e:
            print(f"[telemetry] cage_stats dashboard failed: {e}")
    exe = shutil.which("cage-stats")
    if exe:
        try:
            cmd = [exe, "--once", "--url", url]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if res.returncode == 0:
                return res.stdout
        except Exception as e:
            print(f"[telemetry] cage-stats subprocess dashboard failed: {e}")
    return None


def capture(
    url: str,
    *,
    metrics_path: str = "/metrics",
    api_key: Optional[str] = None,
    interval: float = 1.0,
):
    """Return (snapshot_dict, dashboard_text). In-process this needs ONE poll for both.

    Either element may be None if telemetry is unavailable.
    """
    api = _try_import_api()
    if api is not None:
        try:
            from cage_stats.metrics.state import snapshot_to_dict
            from cage_stats.ui.text import render_dashboard

            snap = api.fetch_snapshot(
                url, metrics_path=metrics_path, api_key=api_key, interval=interval
            )
            return snapshot_to_dict(snap), render_dashboard(snap, url=url, interval=interval)
        except Exception as e:
            print(f"[telemetry] cage_stats capture failed: {e}")
    # CLI fallback: two calls (json + text).
    return (
        capture_snapshot(url, metrics_path=metrics_path, api_key=api_key, interval=interval),
        dashboard_text(url, metrics_path=metrics_path, api_key=api_key, interval=interval),
    )


def scrape_spec_decode(
    url: str, *, metrics_path: str = "/metrics", timeout: float = 10.0
) -> Optional[dict]:
    """Directly scrape vLLM's Prometheus ``/metrics`` for speculative-decode acceptance.

    Dependency-free (stdlib ``urllib``) fallback so the ``speculative`` baseline records an
    acceptance rate even when cage-stats is not installed. Sums each counter across label
    sets and returns
    ``{accepted, draft, acceptance_rate, num_drafts, mean_accept_len}`` — or ``None`` if the
    server is unreachable or speculation is not enabled (the metrics are absent).

    acceptance_rate = accepted / draft  (per the vLLM metrics design).
    """
    import urllib.request

    base = url.rstrip("/")
    endpoint = base if base.endswith(metrics_path) else base + metrics_path
    try:
        with urllib.request.urlopen(endpoint, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", "replace")
    except Exception as e:  # unreachable / no metrics endpoint
        print(f"[telemetry] /metrics scrape failed: {e}")
        return None

    def _sum(metric: str) -> Optional[float]:
        # Exact metric-name match. The Prometheus series name is everything before '{'
        # (labels) or the first space (value). A prefix match would wrongly fold sibling
        # series such as `<metric>_total`, `<metric>_bucket`, `<metric>_sum` into the sum.
        total = None
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            name = line.split("{", 1)[0].split(" ", 1)[0]
            if name != metric:
                continue
            try:
                total = (total or 0.0) + float(line.rsplit(" ", 1)[1])
            except (ValueError, IndexError):
                continue
        return total

    accepted = _sum("vllm:spec_decode_num_accepted_tokens_total")
    draft = _sum("vllm:spec_decode_num_draft_tokens_total")
    num_drafts = _sum("vllm:spec_decode_num_drafts_total")
    if accepted is None and draft is None:
        return None  # speculation not enabled, or metric not exposed by this vLLM version
    return {
        "spec_decode_accepted_tokens": accepted,
        "spec_decode_draft_tokens": draft,
        "spec_decode_acceptance_rate": (accepted / draft) if (accepted is not None and draft) else None,
        "spec_decode_num_drafts": num_drafts,
        "spec_decode_mean_accept_len": (accepted / num_drafts) if (accepted is not None and num_drafts) else None,
    }


def available() -> bool:
    """True if cage-stats telemetry can be captured (in-process or via CLI)."""
    return _try_import_api() is not None or shutil.which("cage-stats") is not None
