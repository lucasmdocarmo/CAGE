#!/usr/bin/env python3
"""Order:     stage 4 (validate), in the vLLM job while the engine is up, when the plan has LMCache cells
Objective: Prove the pod's own memory holds the largest LMCache CPU store the plan sizes (S0F-88 U2, review H1, review 2026-10-13 item 2)
Cloud:     both

Inside a RunPod container /proc/meminfo reports the HOST's memory (S0 attempt 4:
1.4 TiB in `free` on a pod allocated 172 GB), so it cannot bound what the pod
may allocate. Review 2026-10-13 item 2: the bound is the pod's ALLOCATION as
the RunPod API reports it (``memoryInGb``, recorded by the master at
provision and passed as --pod-gib), and the usage is the sum of the resident
memory of the container's own processes (/proc/*/status VmRSS; page cache is
reclaimable and never counted). The memory cgroup's limit, when one is
readable (v2 ``memory.max``, v1 ``memory/memory.limit_in_bytes``), is a second
bound. The probe FAILS CLOSED: with no readable bound it prints
POD_MEMORY_UNKNOWN and exits 1 (the first version fell back to the host's
numbers and passed any store). It prints ONE line the master greps
(POD_MEMORY_OK, POD_MEMORY_FAILED or POD_MEMORY_UNKNOWN), writes the evidence
record to --out, and exits 0 or 1. Run while the engine is up, so the usage
counts the engine and the runner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

#: A cgroup v1 "no limit" reads as a page-aligned value near 2^63; anything at
#: or above 2^60 bytes is treated as no limit.
UNLIMITED_BYTES = 1 << 60
GIB = 1 << 30
#: RunPod reports the allocation in GB; decimal gigabytes are the smaller
#: reading of an ambiguous unit, so the bound errs toward refusing.
POD_GB = 10**9


def _read_int(path: Path) -> Optional[int]:
    """The integer in a cgroup file, None when absent, "max" or unreadable."""
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None


def meminfo_available(meminfo: Path) -> Optional[int]:
    """MemAvailable in bytes (the host's inside a container; recorded, never a bound)."""
    try:
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def cgroup_limit(root: Path) -> Tuple[Optional[int], Optional[str]]:
    """(the cgroup memory limit in bytes, source), (None, None) without a limit."""
    for limit_name, source in (
        ("memory.max", "cgroup-v2"),
        ("memory/memory.limit_in_bytes", "cgroup-v1"),
    ):
        limit = _read_int(root / limit_name)
        if limit is not None and limit < UNLIMITED_BYTES:
            return limit, source
    return None, None


def process_rss_bytes(proc_root: Path) -> Optional[int]:
    """The sum of VmRSS over every process under ``proc_root`` (the container's
    own processes: a PID namespace shows no other), None when none is readable."""
    total = 0
    seen = False
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return None
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            for line in (entry / "status").read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
                    seen = True
                    break
        except (OSError, ValueError, IndexError):
            continue
    return total if seen else None


def judge(
    need_gib: int,
    pod_gib: int,
    meminfo: Path,
    cgroup_root: Path,
    proc_root: Path = Path("/proc"),
) -> Dict[str, Any]:
    """The evidence record: the need, the allocation, the usage, each bound,
    the effective available memory and the verdict (fail closed)."""
    rss = process_rss_bytes(proc_root)
    limit, source = cgroup_limit(cgroup_root)
    bounds: Dict[str, int] = {}
    if pod_gib > 0 and rss is not None:
        bounds["pod-allocation"] = max(0, pod_gib * POD_GB - rss)
    if limit is not None and rss is not None:
        bounds[source or "cgroup"] = max(0, limit - rss)
    bound = min(bounds, key=bounds.get) if bounds else None
    available = bounds[bound] if bound else None
    ok = need_gib <= 0 or (available is not None and available >= need_gib * GIB)
    return {
        "need_gib": need_gib,
        "pod_gib": pod_gib,
        "rss_bytes": rss,
        "cgroup_limit_bytes": limit,
        "cgroup_source": source,
        "meminfo_available_bytes": meminfo_available(meminfo),  # the host's inside a container
        "bounds": bounds,
        "bound": bound,
        "available_bytes": available,
        "ok": ok,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1].strip())
    parser.add_argument("--need-gib", type=int, required=True, help="the largest LMCache store (GiB)")
    parser.add_argument("--pod-gib", type=int, default=0, help="the pod's allocated memory as the RunPod API reports it (GB; 0 = unknown)")
    parser.add_argument("--meminfo", type=Path, default=Path("/proc/meminfo"))
    parser.add_argument("--cgroup-root", type=Path, default=Path("/sys/fs/cgroup"))
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    parser.add_argument("--out", type=Path, default=None, help="evidence JSON record")
    args = parser.parse_args(argv)
    record = judge(args.need_gib, args.pod_gib, args.meminfo, args.cgroup_root, args.proc_root)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    avail = record["available_bytes"]
    rss = record["rss_bytes"]
    line = (
        f"need_gib={record['need_gib']} pod_gib={record['pod_gib']} "
        f"rss_gib={'unread' if rss is None else f'{rss / GIB:.1f}'} "
        f"available_gib={'unread' if avail is None else f'{avail / GIB:.1f}'} bound={record['bound'] or 'none'}"
    )
    if record["need_gib"] > 0 and avail is None:
        print("POD_MEMORY_UNKNOWN " + line + " (no readable bound: pass --pod-gib from the pod record, or a cgroup limit; the host's /proc/meminfo is never a bound)")
        return 1
    print(("POD_MEMORY_OK " if record["ok"] else "POD_MEMORY_FAILED ") + line)
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
