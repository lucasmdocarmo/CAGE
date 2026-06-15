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
import subprocess
import sys
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
    mock: bool = False,
) -> Optional[dict]:
    """Return the full vLLM telemetry snapshot as a dict, or None if unavailable.

    ``mock=True`` uses cage-stats' synthetic data (no server needed) — handy for
    dry-runs and tests.
    """
    api = _try_import_api()
    if api is not None:
        try:
            return api.snapshot_dict(
                url, metrics_path=metrics_path, api_key=api_key, interval=interval, mock=mock
            )
        except Exception as e:
            print(f"[telemetry] cage_stats in-process capture failed: {e}")
    exe = shutil.which("cage-stats")
    if exe:
        try:
            cmd = [exe, "--once", "--json", "--url", url] + (["--mock"] if mock else [])
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if res.returncode == 0 and res.stdout.strip():
                return json.loads(res.stdout)
            print(f"[telemetry] cage-stats CLI returned {res.returncode}: {res.stderr.strip()[:160]}")
        except Exception as e:
            print(f"[telemetry] cage-stats subprocess failed: {e}")
    return None


def dashboard_text(
    url: str,
    *,
    metrics_path: str = "/metrics",
    api_key: Optional[str] = None,
    interval: float = 1.0,
    mock: bool = False,
) -> Optional[str]:
    """Return a one-shot static terminal dashboard string, or None if unavailable."""
    api = _try_import_api()
    if api is not None:
        try:
            return api.dashboard_text(
                url, metrics_path=metrics_path, api_key=api_key, interval=interval, mock=mock
            )
        except Exception as e:
            print(f"[telemetry] cage_stats dashboard failed: {e}")
    exe = shutil.which("cage-stats")
    if exe:
        try:
            cmd = [exe, "--once", "--url", url] + (["--mock"] if mock else [])
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
    mock: bool = False,
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
                url, metrics_path=metrics_path, api_key=api_key, interval=interval, mock=mock
            )
            return snapshot_to_dict(snap), render_dashboard(snap, url=url, interval=interval)
        except Exception as e:
            print(f"[telemetry] cage_stats capture failed: {e}")
    # CLI fallback: two calls (json + text).
    return (
        capture_snapshot(url, metrics_path=metrics_path, api_key=api_key, interval=interval, mock=mock),
        dashboard_text(url, metrics_path=metrics_path, api_key=api_key, interval=interval, mock=mock),
    )


def available() -> bool:
    """True if cage-stats telemetry can be captured (in-process or via CLI)."""
    return _try_import_api() is not None or shutil.which("cage-stats") is not None
