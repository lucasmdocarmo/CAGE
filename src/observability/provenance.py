"""Run provenance: the reproducibility spine for CAGE Phase-2 evidence.

A ``run_manifest.json`` records exactly WHAT produced a set of results -- code revision,
serving stack, model, hardware, cloud instance, and run parameters -- so every table in the
dissertation traces to a specific commit + config + GPU. ``provenance.json`` then records a
SHA256 of every result file, so a reported number can be tied to an exact bytes-on-disk hash.

Every collector FAILS SOFT: a missing tool / dependency / metadata endpoint records ``None``
and emits a warning, never raising -- provenance capture must not crash an expensive GPU run.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("cage.observability.provenance")

# GCP metadata server: authoritative for the instance's own name/zone/machine-type. Answers
# only from inside a GCE VM; a short timeout makes it a no-op (empty dict) off-cloud.
_GCP_METADATA_ROOT = "http://metadata.google.internal/computeMetadata/v1"
_GCP_METADATA_TIMEOUT_S = 1.0


def _run_cmd(args: List[str], cwd: Optional[str] = None, timeout: float = 5.0) -> Optional[str]:
    """Run a command and return trimmed stdout, or None on any failure (never raises)."""
    try:
        out = subprocess.run(
            args, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=True
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("provenance: command %s failed: %s", " ".join(args), exc)
        return None


def _build_info(repo_dir: str) -> Dict[str, str]:
    """Parse <repo>/BUILD_INFO (written by scripts/ops/package_repo.sh at packaging time).

    The GPU VM receives the repo as a tarball, NOT a git clone -- `git rev-parse` fails
    there ("fatal: not a git repository", observed in the 2026-07-15 smoke manifest), so
    the SHA must travel WITH the code. Format: one `key=value` per line (sha, dirty,
    packaged_at). Empty dict when absent/unreadable.
    """
    try:
        text = (Path(repo_dir) / "BUILD_INFO").read_text(encoding="utf-8")
    except OSError:
        return {}
    out: Dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def git_sha(repo_dir: str) -> Optional[str]:
    """Full commit SHA of ``repo_dir`` (BUILD_INFO fallback for tarball deploys)."""
    sha = _run_cmd(["git", "rev-parse", "HEAD"], cwd=repo_dir)
    if sha:
        return sha
    return _build_info(repo_dir).get("sha") or None


def git_dirty(repo_dir: str) -> Optional[bool]:
    """True if ``repo_dir`` has uncommitted changes -- flags a non-reproducible run."""
    out = _run_cmd(["git", "status", "--porcelain"], cwd=repo_dir)
    if out is not None:
        return bool(out.strip())
    dirty = _build_info(repo_dir).get("dirty")
    if dirty in ("0", "1"):
        return dirty == "1"
    return None


def installed_package_commit(dist_name: str) -> Optional[str]:
    """Commit id of a pip VCS-installed package (pip records it in direct_url.json).

    Fallback for code that exists on the box only as an installed package: the VM
    installs cage-stats from git (requirements.txt git dep), so there is no clone to
    rev-parse -- the 2026-07-15 smoke manifest recorded cage_stats_git_sha=null for
    exactly this reason.
    """
    try:
        from importlib import metadata

        raw = metadata.distribution(dist_name).read_text("direct_url.json")
        if not raw:
            return None
        info = json.loads(raw)
        return (info.get("vcs_info") or {}).get("commit_id")
    except Exception:  # pragma: no cover - absent package / non-VCS install
        return None


def vllm_version() -> Optional[str]:
    """Installed vLLM version (the serving stack under test), or None if not importable."""
    try:
        import vllm  # noqa: WPS433 (import inside function: optional, VM-only)

        return getattr(vllm, "__version__", None)
    except Exception as exc:  # pragma: no cover - env-dependent
        logger.warning("provenance: vllm not importable: %s", exc)
        return None


def gpu_info() -> Dict[str, Any]:
    """GPU name / memory / driver / CUDA via pynvml. Empty-ish dict if unavailable."""
    info: Dict[str, Any] = {
        "name": None, "memory_total_mb": None, "driver_version": None,
        "cuda_version": None, "device_count": None,
    }
    try:
        import pynvml  # noqa: WPS433

        pynvml.nvmlInit()
        try:
            info["device_count"] = int(pynvml.nvmlDeviceGetCount())
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            info["name"] = name.decode() if isinstance(name, bytes) else name
            info["memory_total_mb"] = int(pynvml.nvmlDeviceGetMemoryInfo(handle).total // (1024 * 1024))
            drv = pynvml.nvmlSystemGetDriverVersion()
            info["driver_version"] = drv.decode() if isinstance(drv, bytes) else drv
            try:
                info["cuda_version"] = int(pynvml.nvmlSystemGetCudaDriverVersion())
            except Exception:
                info["cuda_version"] = None
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:  # pragma: no cover - GPU-dependent
        logger.warning("provenance: pynvml GPU info unavailable: %s", exc)
    return info


def _gcp_metadata(path: str) -> Optional[str]:
    """Fetch one GCP metadata value (None off-cloud / on timeout)."""
    req = urllib.request.Request(
        f"{_GCP_METADATA_ROOT}/{path}", headers={"Metadata-Flavor": "Google"}
    )
    try:
        with urllib.request.urlopen(req, timeout=_GCP_METADATA_TIMEOUT_S) as resp:
            return resp.read().decode().strip()
    except (urllib.error.URLError, OSError, ValueError):
        return None


def gcp_instance_metadata() -> Dict[str, Any]:
    """Instance name / zone / machine-type / preemptible from the GCE metadata server."""
    zone_raw = _gcp_metadata("instance/zone")  # e.g. projects/123/zones/us-east1-b
    machine_raw = _gcp_metadata("instance/machine-type")  # e.g. projects/123/machineTypes/g2-standard-8
    return {
        "on_gce": zone_raw is not None,
        "name": _gcp_metadata("instance/name"),
        "zone": zone_raw.split("/")[-1] if zone_raw else None,
        "machine_type": machine_raw.split("/")[-1] if machine_raw else None,
        "preemptible": _gcp_metadata("instance/scheduling/preemptible"),
        "project": _gcp_metadata("project/project-id"),
    }


@dataclass
class RunManifest:
    """Everything needed to reproduce a run. Serialised to ``run_manifest.json``."""

    run_id: str
    created_at: str  # ISO-8601 UTC; passed IN (caller stamps time -- keeps this module clock-free)
    # Code provenance
    cage_git_sha: Optional[str] = None
    cage_git_dirty: Optional[bool] = None
    cage_stats_git_sha: Optional[str] = None
    # Serving stack
    vllm_version: Optional[str] = None
    python_version: str = field(default_factory=lambda: platform.python_version())
    torch_version: Optional[str] = None
    # Experiment parameters
    model: Optional[str] = None
    dataset: Optional[str] = None
    num_queries: Optional[int] = None
    num_trials: Optional[int] = None
    seed: Optional[int] = None
    # Serving levers (the confound-critical ones). enforce_eager + max_model_len are held
    # UNIFORM across trees (Option A); gpu_memory_utilization is the SWEPT memory-pressure
    # variable, so recording it is what ties a result set to its point on the pressure sweep.
    kv_cache_dtype: Optional[str] = None
    speculative_config: Optional[str] = None
    enforce_eager: Optional[bool] = None
    max_model_len: Optional[int] = None
    gpu_memory_utilization: Optional[float] = None
    # Hardware / cloud
    hostname: str = field(default_factory=platform.node)
    gpu: Dict[str, Any] = field(default_factory=dict)
    gcp_instance: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _torch_version() -> Optional[str]:
    try:
        import torch  # noqa: WPS433

        return getattr(torch, "__version__", None)
    except Exception:
        return None


def build_manifest(
    *,
    run_id: str,
    created_at: str,
    cage_repo_dir: str,
    cage_stats_repo_dir: Optional[str] = None,
    model: Optional[str] = None,
    dataset: Optional[str] = None,
    num_queries: Optional[int] = None,
    num_trials: Optional[int] = None,
    seed: Optional[int] = None,
    kv_cache_dtype: Optional[str] = None,
    speculative_config: Optional[str] = None,
    enforce_eager: Optional[bool] = None,
    max_model_len: Optional[int] = None,
    gpu_memory_utilization: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> RunManifest:
    """Collect all provenance for one run. ``created_at`` is passed in (module stays clock-free)."""
    return RunManifest(
        run_id=run_id,
        created_at=created_at,
        cage_git_sha=git_sha(cage_repo_dir),
        cage_git_dirty=git_dirty(cage_repo_dir),
        cage_stats_git_sha=(
            (git_sha(cage_stats_repo_dir) if cage_stats_repo_dir else None)
            or installed_package_commit("cage-stats")
        ),
        vllm_version=vllm_version(),
        torch_version=_torch_version(),
        model=model,
        dataset=dataset,
        num_queries=num_queries,
        num_trials=num_trials,
        seed=seed,
        kv_cache_dtype=kv_cache_dtype,
        speculative_config=speculative_config,
        enforce_eager=enforce_eager,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        gpu=gpu_info(),
        gcp_instance=gcp_instance_metadata(),
        extra=dict(extra or {}),
    )


def write_manifest(manifest: RunManifest, path: str) -> None:
    """Write the manifest as pretty JSON (atomic: temp then replace)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, p)
    # Loud, single-line reproducibility warning if the tree was dirty at run time.
    if manifest.cage_git_dirty:
        logger.warning(
            "provenance: CAGE working tree was DIRTY at run start (uncommitted changes) -- "
            "results are NOT reproducible from cage_git_sha=%s alone.", manifest.cage_git_sha,
        )


def sha256_file(path: str, chunk_size: int = 1 << 20) -> Optional[str]:
    """Streaming SHA256 of a file (None if unreadable). Chunked so large CSVs don't load fully."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError as exc:
        logger.warning("provenance: cannot hash %s: %s", path, exc)
        return None


def write_provenance(
    run_dir: str,
    out_path: str,
    *,
    created_at: str,
    patterns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Hash every result artifact under ``run_dir`` into ``provenance.json``.

    Ties each reported number to an exact file hash. ``created_at`` passed in (clock-free module).
    """
    patterns = patterns or ["**/results.csv", "**/metrics.json", "**/*_metrics.json",
                            "**/vllm_telemetry.json", "**/aggregated_metrics.json"]
    root = Path(run_dir)
    files: Dict[str, Any] = {}
    for pat in patterns:
        for fp in sorted(root.glob(pat)):
            if fp.is_file():
                rel = str(fp.relative_to(root))
                files[rel] = {"sha256": sha256_file(str(fp)), "size_bytes": fp.stat().st_size}
    payload = {"run_dir": str(root), "generated_at": created_at, "file_count": len(files),
               "files": files}
    op = Path(out_path)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("provenance: hashed %d result files -> %s", len(files), out_path)
    return payload
