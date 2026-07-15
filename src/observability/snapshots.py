"""Periodic evidence snapshots + an append-only execution trace.

``SnapshotRecorder`` runs on a background daemon thread and, every ``interval_s`` seconds,
samples GPU counters (pynvml), serving telemetry (cage-stats), and caller-supplied progress,
then writes a timestamped JSON snapshot and re-renders a PNG panel. It observes only -- it
never touches the serving request path -- so it cannot perturb the TTFT/TPOT it records.

``TraceRecorder`` is a thread-safe append-only ``trace.jsonl`` of timestamped run events
(server_start, baseline_start/done, snapshot, ...) -- a replayable execution log.

Rendering uses the matplotlib OBJECT API (Figure + Agg canvas), never pyplot, so it is safe
to call from the background thread. matplotlib and pynvml are imported lazily; if either is
absent the recorder degrades (skips PNGs / nulls GPU fields) with a single warning.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("cage.observability.snapshots")

ProgressFn = Callable[[], Dict[str, Any]]


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TraceRecorder:
    """Thread-safe append-only JSONL of timestamped run events."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def event(self, kind: str, **fields: Any) -> None:
        """Append one event line. Never raises -- tracing must not break a run."""
        record = {"ts": _utc_iso(), "epoch": time.time(), "kind": kind, **fields}
        line = json.dumps(record, sort_keys=True)
        try:
            with self._lock, self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as exc:  # pragma: no cover - disk-full etc.
            logger.warning("trace: failed to write event %s: %s", kind, exc)


class SnapshotRecorder:
    """Periodic GPU / serving / progress snapshots as durable JSON + PNG evidence."""

    def __init__(
        self,
        out_dir: str,
        *,
        interval_s: float = 30.0,
        progress_fn: Optional[ProgressFn] = None,
        serving_fn: Optional[Callable[[], Dict[str, Any]]] = None,
        render_png: bool = True,
        series_cap: int = 5000,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.snap_dir = self.out_dir / "snapshots"
        self.snap_dir.mkdir(parents=True, exist_ok=True)
        self.interval_s = float(interval_s)
        self.progress_fn = progress_fn
        # Injectable serving-telemetry source (defaults to cage-stats). Injection keeps this
        # module free of a hard cage-stats dependency and makes it unit-testable with a fake.
        self.serving_fn = serving_fn or self._default_serving_snapshot
        self.render_png = render_png
        self.series_cap = series_cap

        self._series: List[Dict[str, Any]] = []
        self._series_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._seq = 0
        self._start_epoch = time.time()
        self._nvml_handle: Any = None
        self._nvml: Any = None
        self._warned_png = False
        self._init_nvml()

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._thread is not None:
            return
        self._start_epoch = time.time()
        self._thread = threading.Thread(target=self._loop, name="cage-snapshotter", daemon=True)
        self._thread.start()
        logger.info("snapshotter: started (interval=%.0fs, dir=%s)", self.interval_s, self.snap_dir)

    def stop(self, *, final: bool = True) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 10.0)
            self._thread = None
        if final:
            self.snapshot(label="final")  # one last frame after the run ends
        self._shutdown_nvml()
        logger.info("snapshotter: stopped (%d snapshots)", self._seq)

    def _loop(self) -> None:
        # Event.wait() returns True only when stop is set -> clean, promptly-cancellable sleep.
        while not self._stop.wait(self.interval_s):
            try:
                self._tick()
            except Exception as exc:  # pragma: no cover - never let the thread die
                logger.warning("snapshotter: tick failed: %s", exc)

    # ---------------------------------------------------------------- sampling
    def snapshot(self, label: Optional[str] = None) -> Dict[str, Any]:
        """Take one explicit snapshot now (e.g. at a baseline boundary)."""
        return self._tick(label=label)

    def _tick(self, label: Optional[str] = None) -> Dict[str, Any]:
        sample = self._sample(label=label)
        with self._series_lock:
            self._series.append(sample)
            if len(self._series) > self.series_cap:
                self._series = self._series[-self.series_cap:]
        self._seq += 1
        path = self.snap_dir / f"snapshot_{self._seq:05d}.json"
        try:
            path.write_text(json.dumps(sample, indent=2, sort_keys=True), encoding="utf-8")
            # latest.json is a stable filename the watcher/dashboard can always read.
            (self.snap_dir / "latest.json").write_text(
                json.dumps(sample, indent=2, sort_keys=True), encoding="utf-8"
            )
        except OSError as exc:  # pragma: no cover
            logger.warning("snapshotter: cannot write %s: %s", path, exc)
        self._prune_snapshots()
        if self.render_png:
            self._render(self.snap_dir / "latest.png")
        return sample

    def _prune_snapshots(self) -> None:
        """Ring buffer: keep only the newest ``CAGE_SNAPSHOT_KEEP`` (default 2000)
        ``snapshot_*.json`` files so a multi-day run (7,200+ frames at 30s over 60h) cannot
        grow ``snapshots/`` without bound. ``latest.json``/``latest.png`` are untouched.
        Set ``CAGE_SNAPSHOT_KEEP=0`` (or negative) to disable pruning entirely."""
        try:
            keep = int(os.environ.get("CAGE_SNAPSHOT_KEEP", "2000"))
        except (TypeError, ValueError):
            keep = 2000
        if keep <= 0:
            return
        try:
            snaps = sorted(self.snap_dir.glob("snapshot_*.json"))
            for old in snaps[: max(0, len(snaps) - keep)]:
                old.unlink()
        except OSError as exc:  # pragma: no cover - racing deletes / dir vanished
            logger.warning("snapshotter: snapshot prune failed: %s", exc)

    def _sample(self, label: Optional[str]) -> Dict[str, Any]:
        return {
            "seq": self._seq + 1,
            "ts": _utc_iso(),
            "epoch": time.time(),
            "elapsed_s": round(time.time() - self._start_epoch, 3),
            "label": label,
            "gpu": self._gpu_sample(),
            "serving": self._safe_serving(),
            "progress": self._safe_progress(),
        }

    def _safe_progress(self) -> Dict[str, Any]:
        if self.progress_fn is None:
            return {}
        try:
            return dict(self.progress_fn())
        except Exception as exc:
            logger.warning("snapshotter: progress_fn failed: %s", exc)
            return {"error": str(exc)}

    def _safe_serving(self) -> Dict[str, Any]:
        try:
            return dict(self.serving_fn())
        except Exception as exc:
            logger.warning("snapshotter: serving_fn failed: %s", exc)
            return {"available": False, "error": str(exc)}

    @staticmethod
    def _default_serving_snapshot() -> Dict[str, Any]:
        """Serving telemetry from cage-stats, if importable; else {'available': False}."""
        try:
            from cage_stats.api import snapshot_dict  # noqa: WPS433

            snap = snapshot_dict()
            return {"available": True, **(snap if isinstance(snap, dict) else {"raw": snap})}
        except Exception:
            return {"available": False}

    # ---------------------------------------------------------------- GPU (dynamic)
    def _init_nvml(self) -> None:
        try:
            import pynvml  # noqa: WPS433

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as exc:  # pragma: no cover - GPU-dependent
            logger.warning("snapshotter: pynvml unavailable, GPU fields will be null: %s", exc)
            self._nvml = None
            self._nvml_handle = None

    def _shutdown_nvml(self) -> None:
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml = None
            self._nvml_handle = None

    def _gpu_sample(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "mem_used_mb": None, "mem_total_mb": None, "mem_used_pct": None,
            "util_pct": None, "temp_c": None, "power_w": None,
        }
        if self._nvml is None or self._nvml_handle is None:
            return out
        nv, h = self._nvml, self._nvml_handle
        try:
            mem = nv.nvmlDeviceGetMemoryInfo(h)
            out["mem_used_mb"] = int(mem.used // (1024 * 1024))
            out["mem_total_mb"] = int(mem.total // (1024 * 1024))
            out["mem_used_pct"] = round(100.0 * mem.used / mem.total, 2) if mem.total else None
        except Exception:
            pass
        try:
            out["util_pct"] = int(nv.nvmlDeviceGetUtilizationRates(h).gpu)
        except Exception:
            pass
        try:
            out["temp_c"] = int(nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU))
        except Exception:
            pass
        try:
            out["power_w"] = round(nv.nvmlDeviceGetPowerUsage(h) / 1000.0, 1)
        except Exception:
            pass
        return out

    # ---------------------------------------------------------------- rendering
    def _render(self, path: Path) -> None:
        """Render the accumulated series to a PNG using the thread-safe matplotlib OO API."""
        try:
            from matplotlib.figure import Figure  # noqa: WPS433
            from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: WPS433
        except Exception as exc:
            if not self._warned_png:
                logger.warning("snapshotter: matplotlib unavailable, PNG panels disabled: %s", exc)
                self._warned_png = True
            return

        with self._series_lock:
            series = list(self._series)
        if not series:
            return

        xs = [s["elapsed_s"] for s in series]
        mem = [(s["gpu"] or {}).get("mem_used_pct") for s in series]
        util = [(s["gpu"] or {}).get("util_pct") for s in series]
        completed = [(s.get("progress") or {}).get("completed") for s in series]

        fig = Figure(figsize=(9, 7), dpi=110)
        FigureCanvasAgg(fig)  # binds the Agg canvas to fig (thread-safe, no pyplot global state)
        ax1, ax2, ax3 = fig.subplots(3, 1, sharex=True)

        ax1.plot(xs, mem, color="#c0392b")
        ax1.set_ylabel("GPU mem %")
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"CAGE run snapshot — {series[-1]['ts']}")

        ax2.plot(xs, util, color="#2980b9")
        ax2.set_ylabel("GPU util %")
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)

        if any(c is not None for c in completed):
            ax3.plot(xs, completed, color="#27ae60", drawstyle="steps-post")
            ax3.set_ylabel("queries done")
        else:
            ax3.text(0.5, 0.5, "no progress data", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_xlabel("elapsed (s)")
        ax3.grid(True, alpha=0.3)

        fig.tight_layout()
        try:
            fig.savefig(str(path))
        except OSError as exc:  # pragma: no cover
            logger.warning("snapshotter: cannot save PNG %s: %s", path, exc)
