"""CAGE observability & provenance (Phase-2 evidence layer).

A SIDECAR observability subsystem: it observes a run from OUTSIDE the measurement path
(reads GPU counters, cage-stats /metrics, and on-disk progress), so it can never perturb
the serving timings it records -- the same failure class as the tps wall-clock bug, ruled
out here architecturally rather than by care.

Modules:
- provenance: the run manifest (git SHA, vLLM version, model, GPU, GCP instance, seed) and
  SHA256 result hashes -- the reproducibility spine that ties every table to exact code +
  config + hardware.
- snapshots: periodic JSON + PNG snapshots (GPU / serving telemetry / progress) written as
  durable, timestamped evidence artifacts.

All collectors FAIL SOFT (record None + warn) so a missing optional dependency never crashes
an expensive GPU run; what could be captured is always recorded.
"""

from src.observability.provenance import (  # noqa: F401
    RunManifest,
    build_manifest,
    sha256_file,
    write_manifest,
    write_provenance,
)
from src.observability.snapshots import SnapshotRecorder, TraceRecorder  # noqa: F401
