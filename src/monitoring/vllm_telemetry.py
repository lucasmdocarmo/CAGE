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
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from typing import Optional

#: T4.1 (Wave-3) role-token grammar for multi-instance telemetry. MUST stay
#: identical to the CAGE_TELEMETRY_ENDPOINTS role grammar in
#: scripts/3_run/run_experiment.py — a role that parses at the env boundary
#: must be constructible here, and vice versa. "single" is the reserved
#: default for the one-sampler (non-PD) path.
_ROLE_RE = re.compile(r"^[a-z0-9_-]+$")


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
    dialect: str = "vllm",
) -> Optional[dict]:
    """Return the full serving telemetry snapshot as a dict, or None if unavailable.

    Reads LIVE telemetry only. There is no synthetic/mock path: CAGE must never
    record fabricated numbers, so an unavailable server yields None, never fake data.

    ``dialect``: ``"vllm"`` (default; byte-identical to the pre-dialect path) or
    ``"sglang"`` (G-P2, 2026-08-26 — cage-stats scrapes SGLang's /metrics and
    translates its families into the engine dialect). SGLang has NO vllm-named
    fallback: the CLI and spec-decode scrapes below read vLLM series, so an
    installed cage-stats without dialect support must FAIL LOUD here rather than
    degrade into a fabricated-absence None.
    """
    api = _try_import_api()
    if dialect != "vllm":
        # Fail-closed support probe: the dialect module ships with the same
        # cage-stats commit that added the `dialect` kwarg (pin df0eab4 lacks
        # both). Probing the module is unambiguous where **kwargs are opaque.
        import importlib.util

        if api is None or importlib.util.find_spec("cage_stats.metrics.sglang_dialect") is None:
            raise RuntimeError(
                f"telemetry dialect {dialect!r} requires a cage-stats with SGLang "
                "dialect support — the installed pin predates it; bump the "
                "requirements.txt cage-stats pin (gate (q) enforces parity)"
            )
        try:
            return api.snapshot_dict(
                url, metrics_path=metrics_path, api_key=api_key,
                interval=interval, dialect=dialect,
            )
        except Exception as e:
            # vllm-named fallbacks below cannot serve sglang; absence stays absence.
            print(f"[telemetry] cage_stats {dialect} capture failed: {e}")
            return None
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
    # Phase-time SUM/COUNT pairs (vllm:request_{prefill,decode,inference,queue}_
    # time_seconds) and the raw preemption counter are MONOTONIC cumulative values:
    # "final value" aggregation keeps them diff-able, so the memory-pressure sweep can
    # compute per-trial phase-time deltas (prefill growth under prefix-cache eviction)
    # downstream from consecutive trials or from telemetry_series.jsonl.
    _COUNTERS = ("cached_tokens_total", "recomputed_tokens_total",
                 "prefill_time_sum", "prefill_time_count",
                 "decode_time_sum", "decode_time_count",
                 "inference_time_sum", "inference_time_count",
                 "queue_time_sum", "queue_time_count",
                 "preemptions_total")
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
                 metrics_path: str = "/metrics", dialect: str = "vllm",
                 role: str = "single"):
        self.url = url
        self.interval = max(0.25, float(interval))
        self.metrics_path = metrics_path
        # T4.1 multi-instance telemetry: every series record this sampler
        # emits is stamped instance=<role>, so a prefill+decode sampler pair
        # can be merged into ONE series without losing which endpoint each
        # gauge came from (an unlabeled merged series is unattributable —
        # the 2026-08-27 audit's foundation gap). Validated HERE, fail-closed:
        # a malformed role must never reach a JSONL on disk.
        if not isinstance(role, str) or not _ROLE_RE.match(role):
            raise ValueError(
                f"telemetry role {role!r} is invalid: must be a non-empty "
                "[a-z0-9_-]+ token (e.g. 'single', 'prefill', 'decode')"
            )
        self.role = role
        # Per-backend metrics dialect (T4.3), forwarded to capture_snapshot()
        # every tick. "vllm" is byte-identical to the pre-dialect sampler;
        # "sglang" makes cage-stats translate SGLang's /metrics families. A
        # sampler pointed at an SGLang server WITHOUT this would silently
        # aggregate vllm-named absence into an empty series.
        self.dialect = dialect
        self._samples: list = []
        self._sample_ts: list = []  # wall-clock capture time, parallel to _samples
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._nvml_handles: Optional[list] = None  # index-aligned; None slot = bad handle
        self._nvml_failed = False

    def _read_energy_mj(self) -> "tuple[Optional[float], Optional[list]]":
        """Cumulative GPU energy (mJ) via NVML: (sum over ALL GPUs, per-GPU list).

        Reads ``pynvml.nvmlDeviceGetTotalEnergyConsumption`` (mJ since driver
        load) for EVERY ``nvmlDeviceGetCount()`` device once per sampler tick.
        Under tensor parallelism the model spans N GPUs, so the historical
        index-0-only read under-counted energy by 1/N: the first element — the
        value the existing ``energy_mj`` snapshot field now carries — is the
        SUM across all devices (the TP-correct total), and the second is the
        per-GPU breakdown, index-aligned with NVML device indices (None for a
        device whose read failed this tick). One bad handle must not zero the
        rest: each device is guarded individually and the sum spans whichever
        devices answered (the list records which). ImportError (pynvml absent)
        or nvmlInit/count failure permanently disables the probe for this
        sampler -> (None, None), never a fabricated number.
        """
        if self._nvml_failed:
            return (None, None)
        try:
            import pynvml
        except Exception:
            self._nvml_failed = True
            return (None, None)
        if self._nvml_handles is None:
            try:
                pynvml.nvmlInit()
                count = int(pynvml.nvmlDeviceGetCount())
            except Exception:
                self._nvml_failed = True
                return (None, None)
            handles: list = []
            for i in range(count):
                try:
                    handles.append(pynvml.nvmlDeviceGetHandleByIndex(i))
                except Exception:
                    handles.append(None)
            self._nvml_handles = handles
        if not self._nvml_handles:
            return (None, None)  # zero devices: absence stays absence
        per_gpu: list = []
        for handle in self._nvml_handles:
            if handle is None:
                per_gpu.append(None)
                continue
            try:
                per_gpu.append(
                    float(pynvml.nvmlDeviceGetTotalEnergyConsumption(handle))
                )
            except Exception:
                per_gpu.append(None)
        readings = [v for v in per_gpu if v is not None]
        if not readings:
            return (None, per_gpu)
        return (sum(readings), per_gpu)

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
                snap = capture_snapshot(
                    self.url, metrics_path=self.metrics_path, dialect=self.dialect
                )
                if snap:
                    # Once per tick: cumulative NVML energy (mJ), None if
                    # unsupported. energy_mj = SUM across ALL visible GPUs
                    # (TP-correct total); the per-GPU breakdown rides alongside.
                    total_mj, per_gpu_mj = self._read_energy_mj()
                    snap["energy_mj"] = total_mj
                    snap["energy_mj_per_gpu"] = per_gpu_mj
                    self._samples.append(snap)
                    self._sample_ts.append(t0)
            except Exception as e:
                # capture_snapshot swallows flaky-network internally (returns
                # None); a raise reaching here is its fail-loud dialect-support
                # probe. Surface it ONCE and stop the thread — a silent
                # per-tick swallow would launder "unsupported dialect" into an
                # empty-but-plausible telemetry absence.
                print(f"[telemetry] sampler aborted: {e}")
                return
            dt = self.interval - (time.time() - t0)
            if dt > 0:
                self._stop.wait(dt)

    def stop(self) -> "VllmTelemetrySampler":
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=self.interval + 5)
            self._thread = None
        return self

    def save_series(self, path: str) -> Optional[str]:
        """Persist EVERY raw per-tick sample as timestamped JSONL; return path or None.

        ``aggregate()`` collapses the time dimension into peak/mean/final scalars,
        which is exactly the wrong shape for the memory-pressure sweep: eviction
        onset, KV-capacity shrink, and prefill-time growth are trajectories. One
        JSON object per line, ``{"ts": <epoch seconds>, ...full snapshot dict...}``,
        written next to vllm_telemetry.json by run_experiment as
        telemetry_series.jsonl. Returns None (writes nothing) when no samples exist,
        so an empty run never leaves a misleading empty artifact.

        Each record ALSO carries the canonical §6.1 regime-input field names —
        ``ts_s`` and ``kv_cache_usage``, the ``src.analysis.regime_inputs``
        defaults — alongside the legacy ``ts``/``kv_usage`` spellings (Topic-8
        H1: the writer/reader schema mismatch left the regime bridge with no
        producer). Legacy names are kept for existing readers of
        telemetry_series.jsonl (run_memory_sweep.sh's offline readout; possibly
        the cage-stats sibling repo). Absence stays absence: ``kv_cache_usage``
        mirrors ``kv_usage`` only when the snapshot carried the gauge at all —
        a missing gauge is never coerced into a value.

        T4.1: records additionally carry ``instance=<role>`` ("single" by
        default) — see ``_series_records`` for the stamping contract.
        """
        records = self._series_records()
        if not records:
            return None
        with open(path, "w") as fh:
            for rec in records:
                fh.write(json.dumps(rec, default=str) + "\n")
        return path

    def _series_records(self) -> "list[dict]":
        """Fully-shaped series records (oldest first), NOT yet serialized.

        The ONE source of the series record schema, shared by ``save_series``
        (single sampler) and ``save_merged_series`` (a PD sampler set):
        re-shaping records at the merge site would let the two writers drift
        on the canonical/legacy dual-field contract.

        T4.1: every record carries ``instance=<self.role>`` — "single" on the
        default path, the endpoint's role ("prefill"/"decode"/...) under
        CAGE_TELEMETRY_ENDPOINTS. The sampler polled the endpoint, so its role
        is the ground truth of which instance the gauges came from: a rogue
        same-named key in a snapshot is OVERWRITTEN, never trusted (a
        mis-tagged record would silently poison per-role regime math).
        Legacy files (written before T4.1) simply lack the field; readers
        treat absence as single-instance.
        """
        records: list = []
        for ts, snap in zip(self._sample_ts, self._samples):
            if not isinstance(snap, dict):
                continue
            rec = dict(snap)
            # The sampler's own capture clock is the ONLY timestamp the
            # regime bridge may see, and it is stamped AFTER the snapshot
            # merge so a same-named snapshot field can never replace it
            # (S0F-15, live 2026-09-30: cage-stats snapshots carry ``ts``
            # = 1.0 on every tick; the pre-fix order let it overwrite the
            # wall clock and every S0 window read UNKNOWN_TELEMETRY). Same
            # policy as ``instance`` below: the sampler is the ground truth.
            rec["ts"] = round(ts, 3)
            rec["ts_s"] = rec["ts"]
            if "kv_usage" in rec:
                rec.setdefault("kv_cache_usage", rec["kv_usage"])
            rec["instance"] = self.role
            records.append(rec)
        return records

    def aggregate(self) -> Optional[dict]:
        samples = [s for s in self._samples if isinstance(s, dict)]
        if not samples:
            return None
        agg: dict = {"sampled": True, "num_samples": len(samples),
                     "series_len": len(samples)}
        for k in self._GAUGES:
            vals = [s.get(k) for s in samples if isinstance(s.get(k), (int, float))]
            if vals:
                agg[f"{k}_peak"] = round(max(vals), 4)
                agg[f"{k}_avg"] = round(statistics.fmean(vals), 4)
        for k in self._COUNTERS:
            vals = [s.get(k) for s in samples if isinstance(s.get(k), (int, float))]
            if vals:
                agg[k] = max(vals)  # monotonic counters: final value
        # NVML energy: last-first of the cumulative mJ counter across the workload,
        # so J/token = energy_delta_mj / 1000 / tokens is computable offline. Since
        # T4.4 the per-tick energy_mj is the SUM across ALL GPUs, so this delta is
        # the TP-correct multi-GPU total (field name preserved; single-GPU hosts
        # are numerically unchanged). Absent (pynvml missing / unsupported GPU)
        # -> no key, never a fabricated zero.
        energies = [s.get("energy_mj") for s in samples
                    if isinstance(s.get("energy_mj"), (int, float))]
        if len(energies) >= 2:
            agg["energy_delta_mj"] = round(energies[-1] - energies[0], 3)
        # Per-GPU attribution: last-first per NVML index, only where BOTH endpoint
        # ticks carried a reading for that device; otherwise None for that slot
        # (a partially-failing handle yields an honest hole, not a zero).
        per_lists = [s.get("energy_mj_per_gpu") for s in samples
                     if isinstance(s.get("energy_mj_per_gpu"), list)]
        if len(per_lists) >= 2:
            first, final = per_lists[0], per_lists[-1]
            agg["energy_delta_mj_per_gpu"] = [
                round(b - a, 3)
                if isinstance(a, (int, float)) and isinstance(b, (int, float))
                else None
                for a, b in zip(first, final)
            ]
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


def save_merged_series(samplers: "list[VllmTelemetrySampler]", path: str) -> Optional[str]:
    """Merge role-tagged samplers into ONE JSONL sorted by ``ts_s``; path or None.

    T4.1 (Wave-3): the PD stack runs one sampler per role=url endpoint, but
    downstream (campaign_session.emit_window -> the window's cage_stats.jsonl
    -> the §6.1 regime bridge) reads exactly ONE telemetry_series.jsonl per
    trial. Merging happens HERE, at write time, not per-tick: each sampler's
    thread stays lock-free, and the record shaping is `_series_records` — the
    same code path as the single-sampler ``save_series``, so the two writers
    cannot drift. Records interleave by ``ts_s`` (all samplers stamp ts from
    the same ``time.time()`` clock as the measurement-window bounds); ties
    keep sampler order (Python's sort is stable).

    Fail-closed:
    - duplicate roles RAISE — two samplers claiming one role would make the
      ``instance`` column silently ambiguous, the exact unattributability
      this task exists to remove;
    - NO samples across all samplers -> returns None and writes nothing (an
      empty run never leaves a misleading empty artifact — the ``save_series``
      contract).
    """
    roles = [s.role for s in samplers]
    if len(set(roles)) != len(roles):
        raise ValueError(
            f"save_merged_series: duplicate sampler roles {roles} — every "
            "sampler in a merge must carry a distinct instance role"
        )
    records: list = []
    for sampler in samplers:
        records.extend(sampler._series_records())
    if not records:
        return None
    records.sort(key=lambda rec: rec["ts_s"])
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, default=str) + "\n")
    return path


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


#: S0F-22 Batch 2 (ADR-0134): the decode-side series whose per-window deltas
#: prove the KV pull of a prefill/decode pair. record key -> (series name,
#: label filter). Names from vLLM v0.19.1: the NIXL families
#: (nixl_connector.py:3030-3164; one histogram observation per transfer
#: handle, bytes = NIXL totalBytes) and the engine counters
#: (v1/metrics/loggers.py:600-633 by-source and recomputed, :582-588
#: preemptions). The `_sum`/`_count`/`_total` suffixes are prometheus_client's
#: exposition spellings. Labels (model_name, engine) are summed: a single
#: engine per role in CAGE, and the lifetime total is the right quantity
#: under data parallel too.
PD_TRANSFER_SERIES: "dict[str, tuple[str, dict[str, str]]]" = {
    "bytes_sum": ("vllm:nixl_bytes_transferred_sum", {}),
    "transfer_count": ("vllm:nixl_bytes_transferred_count", {}),
    "failed_transfers": ("vllm:nixl_num_failed_transfers_total", {}),
    "failed_notifications": ("vllm:nixl_num_failed_notifications_total", {}),
    "external_kv_tokens": ("vllm:prompt_tokens_by_source_total", {"source": "external_kv_transfer"}),
    "local_compute_tokens": ("vllm:prompt_tokens_by_source_total", {"source": "local_compute"}),
    "local_cache_hit_tokens": ("vllm:prompt_tokens_by_source_total", {"source": "local_cache_hit"}),
    "recomputed_tokens": ("vllm:prompt_tokens_recomputed_total", {}),
    "preemptions": ("vllm:num_preemptions_total", {}),
}
#: The prefill's only transfer family (nixl_connector.py:3128-3136): tickets
#: the decode never pulled, counted about VLLM_NIXL_ABORT_REQUEST_TIMEOUT
#: (480 s) after the ticket and only on a prefill engine step, so a
#: window-bounded delta describes requests of an earlier window. Recorded,
#: never gated.
PD_PREFILL_SERIES: "dict[str, tuple[str, dict[str, str]]]" = {
    "kv_expired_reqs": ("vllm:nixl_num_kv_expired_reqs_total", {}),
}
#: The strict clauses (ADR-0134 decision 4a), evaluated by
#: ``pd_transfer_reasons`` for the runner's refusal AND verify_results
#: check (k): one rule, two callers. The identities (decision 4b) are
#: recorded in the record and never gated: a preemption recompute changes
#: ``local_compute`` (stats.py:280-297) and pressure windows preempt by design.
PD_TRANSFER_GATE_CLAUSES: "tuple[str, ...]" = (
    "failed_transfers == 0",
    "failed_notifications == 0",
    "bytes_sum > 0",
    "transfer_count >= n_served_rows",
    "external_kv_tokens > 0",
)
_PD_GATE_KEYS = ("failed_transfers", "failed_notifications", "bytes_sum", "transfer_count", "external_kv_tokens")


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_counter_series(text: str, series: "dict[str, tuple[str, dict[str, str]]]") -> "dict[str, Optional[float]]":
    """Values of ``series`` in one Prometheus exposition: per key, the sum over
    samples whose name EQUALS the series name and whose labels contain the
    filter; None when no sample matched or the sum is not finite (absence
    stays absence, never 0). Parsed with prometheus_client's own parser, so
    label values with '}' or ',' and trailing timestamps are handled, and the
    ``_bucket``/``_created`` siblings never match a ``_sum``/``_count`` name.
    """
    from prometheus_client.parser import text_string_to_metric_families

    wanted: "dict[str, list]" = {}
    for key, (name, labels) in series.items():
        wanted.setdefault(name, []).append((key, labels))
    sums: "dict[str, float]" = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            for key, labels in wanted.get(sample.name, ()):
                if all(sample.labels.get(k) == v for k, v in labels.items()):
                    sums[key] = sums.get(key, 0.0) + float(sample.value)
    return {key: (sums[key] if key in sums and math.isfinite(sums[key]) else None) for key in series}


def scrape_counter_series(
    url: str, series: "dict[str, tuple[str, dict[str, str]]]", *,
    metrics_path: str = "/metrics", timeout: float = 10.0,
) -> "dict[str, Optional[float]]":
    """One GET of ``url`` + ``metrics_path`` read through ``read_counter_series``.
    Transport and HTTP failures RAISE (urllib's OSError family), and so does
    a body the Prometheus parser rejects (ValueError): the caller owns the
    policy (the campaign pd gate refuses, the pilot path records).
    """
    import urllib.request

    base = url.rstrip("/")
    endpoint = base if base.endswith(metrics_path) else base + metrics_path
    with urllib.request.urlopen(endpoint, timeout=timeout) as resp:
        text = resp.read().decode("utf-8", "replace")
    return read_counter_series(text, series)


def pd_transfer_reasons(delta: "dict", n_served_rows) -> "list[str]":
    """Every violated clause of PD_TRANSFER_GATE_CLAUSES for one window's
    decode deltas, as sentences; empty means verified. ``n_served_rows`` is
    the Batch 1 served-row count (rows without ``error``): every such row was
    a 2xx the proxy answered only with a ticket, so the decode pulled at least
    one block for it (local prefix hits are capped at N-1 tokens,
    kv_cache_manager.py:195-201; the connector claims N-local >= 1 external
    tokens, nixl_connector.py:784-790, and reads at least one block,
    :833-840). A None delta (series absent), a non-finite value or a negative
    delta (the decode's counters reset between the scrapes) is a reason too.
    """
    reasons: "list[str]" = []
    if isinstance(n_served_rows, bool) or not isinstance(n_served_rows, int) or n_served_rows < 0:
        reasons.append(f"n_served_rows is {n_served_rows!r}, not a non-negative integer")
        return reasons
    # n_served_rows == 0 is NOT a clause (review 2026-10-01, LOW-1): the gate
    # asks whether the pair transferred. A cliff window where every request
    # timed out after its pull is attainment data with bytes moved; a window
    # where nothing moved fails bytes_sum > 0 regardless of the row count.
    values: "dict[str, float]" = {}
    for key in _PD_GATE_KEYS:
        value = delta.get(key) if isinstance(delta, dict) else None
        if value is None:
            reasons.append(f"{key}: series absent on the decode")
        elif not _is_number(value):
            reasons.append(f"{key}: not a finite number ({value!r})")
        elif value < 0:
            reasons.append(f"{key}: negative delta {value} (the decode's counters reset between the scrapes)")
        else:
            values[key] = value
    if "failed_transfers" in values and values["failed_transfers"] != 0:
        reasons.append(f"failed_transfers == 0 violated: {values['failed_transfers']:g} failed pull(s)")
    if "failed_notifications" in values and values["failed_notifications"] != 0:
        reasons.append(
            f"failed_notifications == 0 violated: {values['failed_notifications']:g} failed notification(s)")
    if "bytes_sum" in values and values["bytes_sum"] <= 0:
        reasons.append(f"bytes_sum > 0 violated: {values['bytes_sum']:g} bytes moved")
    if "transfer_count" in values and values["transfer_count"] < n_served_rows:
        reasons.append(
            f"transfer_count >= n_served_rows violated: {values['transfer_count']:g} transfer(s) "
            f"for {n_served_rows} served row(s)")
    if "external_kv_tokens" in values and values["external_kv_tokens"] <= 0:
        reasons.append(
            f"external_kv_tokens > 0 violated: {values['external_kv_tokens']:g} prompt tokens from external KV")
    return reasons


def _native(value: Optional[float]):
    """A JSON-native number: int when integer-valued, else float; None stays."""
    if value is None:
        return None
    return int(value) if float(value).is_integer() else float(value)


def pd_transfer_record(
    *, start: "dict", end: "dict", prefill_start: "Optional[dict]", prefill_end: "Optional[dict]",
    n_served_rows: int, prompt_tokens_sum: Optional[int], decode_url: str, prefill_url: Optional[str],
    scrape_start_ts: float, scrape_end_ts: float,
) -> "dict":
    """The metrics.json ``pd_transfer`` block for one window (ADR-0134).

    ``start``/``end`` are two ``read_counter_series`` results of the decode
    (before the first measured send, after the last completion); the prefill
    pair is optional. Deltas are end minus start per key (None when either
    side is None). ``verified`` is ``not pd_transfer_reasons(delta, n)``. The
    identities (decision 4b) are recorded with a tri-state verdict each and
    never decide ``verified``. Every value is JSON-native with string keys
    and no NaN: the staging writer dumps the dict unchanged.
    """
    delta: "dict[str, Optional[float]]" = {}
    for key in PD_TRANSFER_SERIES:
        a, b = start.get(key), end.get(key)
        delta[key] = _native(b - a) if _is_number(a) and _is_number(b) else None
    prefill_delta = None
    if prefill_start is not None and prefill_end is not None:
        prefill_delta = {}
        for key in PD_PREFILL_SERIES:
            a, b = prefill_start.get(key), prefill_end.get(key)
            prefill_delta[key] = _native(b - a) if _is_number(a) and _is_number(b) else None
    reasons = pd_transfer_reasons(delta, n_served_rows)

    def _eq(x, y) -> Optional[bool]:
        return (x == y) if _is_number(x) and _is_number(y) else None

    external, cache_hit = delta["external_kv_tokens"], delta["local_cache_hit_tokens"]
    identities = {
        "local_compute_equals_served": _eq(delta["local_compute_tokens"], n_served_rows),
        "recomputed_equals_served": _eq(delta["recomputed_tokens"], n_served_rows),
        "external_plus_cache_hit_equals_prompt_tokens": (
            _eq(external + cache_hit, prompt_tokens_sum)
            if _is_number(external) and _is_number(cache_hit) and _is_number(prompt_tokens_sum) else None
        ),
        "no_preemptions": _eq(delta["preemptions"], 0),
    }
    identities_hold: Optional[bool] = (
        None if any(v is None for v in identities.values()) else all(identities.values())
    )
    count = delta["transfer_count"]
    return {
        "adr": "ADR-0134",
        "decode_url": decode_url,
        "prefill_url": prefill_url,
        "scrape_start_ts": float(scrape_start_ts),
        "scrape_end_ts": float(scrape_end_ts),
        "start": {k: _native(v) for k, v in start.items()},
        "end": {k: _native(v) for k, v in end.items()},
        "delta": delta,
        "prefill_start": None if prefill_start is None else {k: _native(v) for k, v in prefill_start.items()},
        "prefill_end": None if prefill_end is None else {k: _native(v) for k, v in prefill_end.items()},
        "prefill_delta": prefill_delta,
        "n_served_rows": int(n_served_rows),
        "prompt_tokens_sum": None if prompt_tokens_sum is None else int(prompt_tokens_sum),
        "transfers_per_served_row": (
            float(count) / n_served_rows if _is_number(count) and isinstance(n_served_rows, int)
            and not isinstance(n_served_rows, bool) and n_served_rows > 0 else None
        ),
        "identities": identities,
        "identities_hold": identities_hold,
        "gate_clauses": list(PD_TRANSFER_GATE_CLAUSES),
        "reasons": reasons,
        "verified": not reasons,
    }


def available() -> bool:
    """True if cage-stats telemetry can be captured (in-process or via CLI)."""
    return _try_import_api() is not None or shutil.which("cage-stats") is not None
