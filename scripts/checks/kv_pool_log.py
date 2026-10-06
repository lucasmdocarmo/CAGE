#!/usr/bin/env python3
"""
Order:     gate helper, imported by scripts/checks/preflight_check.sh gate (j) and called by every scripts/2_serving launcher at its ready line
Objective: Parse an engine startup log for the REALIZED KV pool (charter 6.5 iso-bytes) and record it as CURRENT.kvpool.json, which gate (j) reads before any mtime discovery
Cloud:     both

The realized KV pool of one engine start (S0F-24, ADR-0142).

Until 2026-10-06 the parser below lived inside the gate (j) heredoc of
preflight_check.sh, so only the gate could read a startup log, and it found
each engine's log by modification time. A launcher could not record what its
own start realized, a multi-start log needed the ADR-0130 tail rule, and the
suite's empty logs needed the S0F-23 filter. This module is the one source of
the rules and offers two things on top of them:

1. ``capture``: at its ready line a launcher calls
   ``kv_pool_log.py capture --engine E --log LOG --out DIR/CURRENT.kvpool.json``
   (or ``--role prefill=LOG --role decode=LOG`` for the pd pair). The log is
   parsed with the same rules the gate applies (tail rule included), with a
   bounded retry for a pool line the engine is still writing, and the record
   is written atomically. With ``--merge-into CFG`` the realized pool also
   lands in the serving-config JSON the launcher already wrote. A failure
   prints a warning and exits 3; the launcher never blocks a start on its
   recording, and the gate falls back to the log.
2. gate (j) discovery, in ``main``: an explicit pin (CAGE_ISO_BYTES_LOGS)
   wins; else the engine's current record (an exact file, no mtime); else the
   newest non-empty log as before, with byte-identical output. ``stop``
   removes the record, so a current record always means a running engine.

Every parser rule is version-bound and tagged where it was proven live
(S0 2026-09-30). A corrupt record is a gate failure naming the file, never a
silent fall-back.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

GIB = 1024 ** 3
MIB = 1024 ** 2
#: vLLM V0's default KV block size in tokens (legacy "# GPU blocks" channel
#: only; the modern channels report tokens/bytes directly). [VERIFY-LIVE at S0]
VLLM_BLOCK_TOKENS = 16

#: The record a launcher writes at ready time and the gate reads first.
CAPTURE_SCHEMA = "cage-kvpool-capture-v1"
CURRENT_NAME = "CURRENT.kvpool.json"
CURRENT_PD_NAME = "CURRENT.pd.kvpool.json"
#: Bounded wait for a pool line the engine is still writing at ready time
#: (the S0 logs print it during engine init, before the HTTP surface answers;
#: the retry covers a slower engine) [A].
DEFAULT_RETRIES = 10
DEFAULT_INTERVAL_S = 1.0


class IsoBytesError(ValueError):
    """Loud parse/parity failure -- the §6.5 gate must never guess."""


_NUM = r"(-?[0-9][0-9_,]*(?:\.[0-9]+)?)"


def _num(text: str) -> float:
    value = float(text.replace(",", "").replace("_", ""))
    if value < 0:
        raise IsoBytesError(f"negative KV-pool quantity {text!r}")
    return value


# Per engine: (anchor substring, full regex, channel, scale-to-channel-unit).
# channel 'bytes' | 'tokens' are direct; the composite channels are folded in
# parse_engine_log. An anchor hit whose full regex does NOT match raises
# (corrupted line), so silent engine log-format drift is impossible.
_RULES = {
    "vllm": [
        # V1 (>=0.8) kv_cache_utils.py: realized pool size in tokens.
        ("GPU KV cache size:",
         re.compile(rf"GPU KV cache size:\s*{_NUM}\s*tokens"), "tokens", 1.0),
        # V1 gpu_worker.py: realized pool memory.
        ("Available KV cache memory:",
         re.compile(rf"Available KV cache memory:\s*{_NUM}\s*GiB"), "bytes", float(GIB)),
        # V1 0.19.1 gpu_worker.py:361 under --kv-cache-memory-bytes: the worker
        # skips profiling, prints the budget it reserved and never the
        # "Available" line (ADR-0130, S0F-14; six S0 logs). The anchor is the
        # long phrase on purpose: the launch echo at the top of the same log
        # carries the bare words kv_cache_memory_bytes and must not trip it.
        ("memory for KV Cache as specified by kv_cache_memory_bytes config",
         re.compile(rf"reserved\s*{_NUM}\s*GiB memory for KV Cache as specified by "
                    rf"kv_cache_memory_bytes config"), "bytes", float(GIB)),
        # V0 0.6.x memory-profile summary ("...the rest of the memory reserved
        # for KV Cache is 5.33GiB").
        ("memory reserved for KV Cache is",
         re.compile(rf"memory reserved for KV Cache is\s*{_NUM}\s*GiB"), "bytes", float(GIB)),
        # Legacy block report; VLLM_BLOCK_TOKENS tokens per block.
        ("# GPU blocks:",
         re.compile(rf"# GPU blocks:\s*{_NUM}"), "tokens", float(VLLM_BLOCK_TOKENS)),
    ],
    "sglang": [
        # srt.mem_cache.memory_pool allocation report (K + V summed; SGLang
        # prints "GB" but computes 1024**3 -- binary units, like vLLM's GiB).
        ("KV Cache is allocated",
         re.compile(
             rf"KV Cache is allocated\.?\s*#tokens:\s*{_NUM},\s*"
             rf"K size:\s*{_NUM}\s*GB,\s*V size:\s*{_NUM}\s*GB"),
         "sglang-alloc", None),
        # Scheduler config echo (tokens-only fallback channel).
        ("max_total_num_tokens",
         re.compile(rf"max_total_num_tokens\s*[=:]\s*{_NUM}"), "tokens", 1.0),
    ],
    "lmdeploy": [
        # TurboMind 0.17.0 turbomind.cc:319 (ADR-0130, S0F-9): the cache buffer
        # TurboMind measured and allocated, MB = 2^20 (cache_bytes = free x
        # ratio). It is the engine's one pool-size line; the [BlockManager] pair
        # below left the source at 0.15.0. Bytes only: the token capacity needs
        # the allocator's page and slab rules (the region sits about 1.6% above
        # the usable blocks at S0, inside the 0.05 tolerance). The tqdm weight
        # bar hides this line behind carriage returns; read_text and
        # splitlines() separate it.
        ("Object cache budget:",
         re.compile(rf"Object cache budget:\s*{_NUM}\s*MB from free\s*{_NUM}\s*MB "
                    rf"and ratio\s*{_NUM}"), "bytes", float(MIB)),
        # TurboMind <= 0.14 BlockManager pair: bytes = block_size(MB=2^20) x count.
        ("[BlockManager] block_size",
         re.compile(rf"\[BlockManager\]\s*block_size\s*=\s*{_NUM}\s*MB"),
         "lmdeploy-block-mb", None),
        ("[BlockManager] max_block_count",
         re.compile(rf"\[BlockManager\]\s*max_block_count\s*=\s*{_NUM}"),
         "lmdeploy-block-count", None),
    ],
}

#: One line each engine prints exactly once per engine start, before any pool
#: line (ADR-0130 tail rule): a log that holds several starts (the cluster
#: manager appends every start of a replica to one file; S0 had 2 to 5) is
#: parsed from its LAST marker on, so a budgeted start can never read the
#: Available line of an earlier profiled start. Older shapes carry no marker
#: and are parsed whole, as before.
_START_MARKERS = {
    "vllm": "Initializing a V1 LLM engine",   # core.py:105 (v0.19.1), the EngineCore banner
    "sglang": "server_args=",                   # the ServerArgs dump at launch
    "lmdeploy": "input backend=",               # async_engine.py:130 (0.17.0)
}


def last_start_tail(engine, lines):
    """(lines from the last start marker on, number of markers seen)."""
    marker = _START_MARKERS.get(engine)
    if marker is None:
        return lines, 0
    hits = [i for i, line in enumerate(lines) if marker in line]
    if not hits:
        return lines, 0
    return lines[hits[-1]:], len(hits)


def parse_engine_log(engine, text):
    """Parse one engine startup log -> {'engine', 'bytes', 'tokens', 'evidence', 'starts'}.

    'bytes'/'tokens' are ints or None (channel not present in this log); only
    the lines after the LAST engine-start marker count (last_start_tail), and
    within them the LAST occurrence of a channel wins (a re-profiled pool must
    supersede its predecessor). 'evidence' keeps every matched line verbatim
    for the run log; 'starts' is the number of start markers in the file."""
    rules = _RULES.get(engine)
    if rules is None:
        raise IsoBytesError(
            f"engine {engine!r} has no KV-pool parser -- add its startup-log "
            f"rules before scoping it into the iso-bytes gate")
    found = {}
    evidence = []
    lines, starts = last_start_tail(engine, text.splitlines())
    for line in lines:
        for anchor, pattern, channel, scale in rules:
            if anchor in line:
                m = pattern.search(line)
                if not m:
                    raise IsoBytesError(
                        f"{engine}: corrupted KV-pool line (anchor {anchor!r} "
                        f"present but the value did not parse): {line.strip()!r}")
                if channel == "sglang-alloc":
                    found["tokens"] = _num(m.group(1))
                    found["bytes"] = (_num(m.group(2)) + _num(m.group(3))) * GIB
                else:
                    value = _num(m.group(1))
                    # scale None = composite channel folded below (lmdeploy).
                    found[channel] = value if scale is None else value * scale
                evidence.append(line.strip())
    if engine == "lmdeploy":
        mb = found.pop("lmdeploy-block-mb", None)
        count = found.pop("lmdeploy-block-count", None)
        if (mb is None) != (count is None):
            raise IsoBytesError(
                "lmdeploy: found only one of the [BlockManager] block_size / "
                "max_block_count pair -- cannot compute the realized pool from "
                "half a product")
        if mb is not None:
            found["bytes"] = mb * MIB * count
    if not found:
        raise IsoBytesError(
            f"{engine}: NO recognizable KV-pool line in the startup log -- "
            f"§6.5 realized-bytes parity cannot be certified (engine log "
            f"format drift? update the parser rules; never skip this gate)")
    return {
        "engine": engine,
        "bytes": int(found["bytes"]) if "bytes" in found else None,
        "tokens": int(found["tokens"]) if "tokens" in found else None,
        "evidence": evidence,
        "starts": starts,
    }


def note_multi_start(label, reading, path):
    """One [note] line when a log holds several engine starts (tail rule)."""
    if reading.get("starts", 0) > 1:
        print(f"  [note] {label}: {reading['starts']} engine starts in {path}; "
              f"parsed the last one (ADR-0130 tail rule)")


def relative_gap(a, b):
    top = max(float(a), float(b))
    if top <= 0:
        raise IsoBytesError(
            "KV pool of size 0 -- an engine realized NO cache; parity over "
            "zero is meaningless")
    return abs(float(a) - float(b)) / top


def compare_pair(ra, rb, tol):
    """-> (basis, gap, within). bytes when both sides report bytes; tokens as
    an explicit PROXY basis (valid only at uniform KV dtype) when bytes are
    unavailable on both sides; no common basis raises (incomparable)."""
    if ra["bytes"] is not None and rb["bytes"] is not None:
        basis, gap = "bytes", relative_gap(ra["bytes"], rb["bytes"])
    elif ra["tokens"] is not None and rb["tokens"] is not None:
        basis, gap = "tokens", relative_gap(ra["tokens"], rb["tokens"])
    else:
        raise IsoBytesError(
            f"{ra['engine']} vs {rb['engine']}: no common basis (one log "
            f"reports only bytes, the other only tokens) -- cannot certify "
            f"§6.5 parity; capture a log variant carrying the missing channel")
    return basis, gap, gap <= tol


def newest_log(root, engine):
    logs = sorted((root / engine).glob("*.log"), key=lambda p: p.stat().st_mtime)
    # S0F-23: a start whose engine never wrote leaves a 0-byte log (the test
    # suite's fake-engine starts, a launch killed before the first line); an
    # empty file can never carry a KV-pool line, so it is never "the newest log".
    empty = [p for p in logs if p.stat().st_size == 0]
    if empty:
        print(f"  [note] {engine}: skipped {len(empty)} empty log file(s) under "
              f"{root / engine}/ (a start that never wrote; S0F-23)")
    logs = [p for p in logs if p.stat().st_size > 0]
    return logs[-1] if logs else None


# ---------------------------------------------------------------------------
# The launcher-side record (S0F-24)
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fmt(reading) -> "tuple[str, str]":
    size = "n/a" if reading["bytes"] is None else f"{reading['bytes'] / GIB:.3f} GiB"
    toks = "n/a" if reading["tokens"] is None else str(reading["tokens"])
    return size, toks


def capture_record(
    engine: str,
    log_path,
    *,
    role: Optional[str] = None,
    retries: int = DEFAULT_RETRIES,
    interval_s: float = DEFAULT_INTERVAL_S,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], str] = utc_now,
) -> dict:
    """Parse one start log at ready time into the capture record.

    The parse is gate (j)'s own (``parse_engine_log``, tail rule included).
    A log with no pool line yet is re-read up to ``retries`` times, waiting
    ``interval_s`` between reads (the engine may still be writing it); a
    corrupted line raises at once, as does a missing file. The record carries
    the parse plus the log path, its mtime and the capture time.
    """
    path = Path(log_path)
    attempts = 0
    while True:
        if not path.is_file():
            raise IsoBytesError(f"{engine}: start log {path} does not exist")
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            reading = parse_engine_log(engine, text)
        except IsoBytesError as exc:
            if "NO recognizable KV-pool line" in str(exc) and attempts < retries:
                attempts += 1
                sleep(interval_s)
                continue
            raise
        return {
            "schema": CAPTURE_SCHEMA,
            "engine": engine,
            "role": role,
            "log": str(path),
            "log_mtime": path.stat().st_mtime,
            "bytes": reading["bytes"],
            "tokens": reading["tokens"],
            "evidence": reading["evidence"],
            "starts": reading["starts"],
            "captured_utc": now(),
        }


def pd_record(engine: str, role_records: Dict[str, dict], *, now: Callable[[], str] = utc_now) -> dict:
    """One record for a prefill/decode pair: both role parses under ``roles``."""
    if not role_records:
        raise IsoBytesError(f"{engine}: a pd record needs at least one role")
    return {
        "schema": CAPTURE_SCHEMA,
        "engine": engine,
        "topology": "pd",
        "roles": {role: dict(rec) for role, rec in role_records.items()},
        "captured_utc": now(),
    }


def write_record(path, record: dict) -> Path:
    """tmp + os.replace: a reader never sees a partial record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def merge_into_serving_config(cfg_path, record: dict) -> bool:
    """Add the realized pool to the launcher's serving-config JSON, in place
    (atomic). False, with a note, when the file is absent or not a JSON
    object: the record is the primary carrier, the capture JSON exists only
    under CAGE_RUN_ROOT."""
    path = Path(cfg_path)
    if not path.is_file():
        print(f"  [note] serving-config capture {path} absent; the realized pool "
              f"is in the record only")
        return False
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"  [note] serving-config capture {path} unreadable ({exc}); left untouched")
        return False
    if not isinstance(doc, dict):
        print(f"  [note] serving-config capture {path} is not a JSON object; left untouched")
        return False
    doc["kv_pool_bytes_realized"] = record["bytes"]
    doc["kv_pool_tokens_realized"] = record["tokens"]
    doc["kv_pool_evidence"] = list(record["evidence"])
    doc["kv_pool_log"] = record["log"]
    doc["kv_pool_captured_utc"] = record["captured_utc"]
    write_record(path, doc)
    return True


def current_record_paths(log_root, engine: str) -> "tuple[Path, Path]":
    d = Path(log_root) / engine
    return d / CURRENT_NAME, d / CURRENT_PD_NAME


def _load_record(path: Path, engine: str) -> dict:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise IsoBytesError(
            f"{engine}: launcher capture {path} is unreadable ({exc}); delete it "
            f"or restart the engine (S0F-24)")
    schema = doc.get("schema") if isinstance(doc, dict) else None
    if schema != CAPTURE_SCHEMA:
        raise IsoBytesError(
            f"{engine}: launcher capture {path} has schema {schema!r}, expected "
            f"{CAPTURE_SCHEMA!r}; delete it or restart the engine (S0F-24)")
    if doc.get("engine") != engine:
        raise IsoBytesError(
            f"{engine}: launcher capture {path} records engine {doc.get('engine')!r}; "
            f"delete it or restart the engine (S0F-24)")
    return doc


def read_current(log_root, engine: str) -> Optional[dict]:
    """{'path', 'record'} of the engine's newest launcher capture (the single
    or the pd record, whichever was captured later), or None when neither
    file exists. An unreadable, foreign or mis-engined record raises: the
    gate fails naming the file instead of guessing."""
    found = []
    for path in current_record_paths(log_root, engine):
        if path.is_file():
            found.append((path, _load_record(path, engine)))
    if not found:
        return None
    # captured_utc has second resolution; the file's mtime breaks a tie
    # (review 2026-10-06, LOW-6).
    path, record = max(found, key=lambda item: (
        str(item[1].get("captured_utc") or ""), item[0].stat().st_mtime_ns))
    return {"path": path, "record": record}


def _reading_from_record(engine: str, rec: dict, capture_path: Path) -> dict:
    return {
        "engine": engine,
        "bytes": rec.get("bytes"),
        "tokens": rec.get("tokens"),
        "evidence": list(rec.get("evidence") or []),
        "starts": int(rec.get("starts") or 0),
        "log": str(rec.get("log")),
        "capture": str(capture_path),
    }


# ---------------------------------------------------------------------------
# Gate (j): the CAGE-ISO-BYTES-GATE main (preflight_check.sh imports it)
# ---------------------------------------------------------------------------


def main(argv):
    raw = argv[1] if len(argv) > 1 else "vllm,sglang,lmdeploy"
    engines = []
    for token in raw.split(","):
        token = token.strip()
        token = "lmdeploy" if token == "lmdeploy-turbomind" else token
        if token and token not in engines:
            engines.append(token)
    tol_raw = os.environ.get("CAGE_ISO_BYTES_TOL", "0.05")
    try:
        tol = float(tol_raw)
    except ValueError:
        print(f"  [FAIL] CAGE_ISO_BYTES_TOL={tol_raw!r} is not a float")
        return 1
    if not (0.0 < tol < 1.0):
        print(f"  [FAIL] CAGE_ISO_BYTES_TOL={tol} outside (0, 1) -- refusing a "
              f"vacuous or impossible tolerance")
        return 1

    # pins: engine -> {role_or_None: Path}. role None = the legacy bare
    # 'engine=path' form (one whole-pool log); role tokens are free-form
    # ('engine:role=path', split on the FIRST colon). Duplicate engine[:role]
    # keys and a bare+role mix for one engine both REFUSE: either would make
    # which log counts a silent coin-flip.
    pins = {}
    seen_keys = set()
    for entry in os.environ.get("CAGE_ISO_BYTES_LOGS", "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            print(f"  [FAIL] CAGE_ISO_BYTES_LOGS entry {entry!r} is not engine[:role]=path")
            return 1
        key, _, path = entry.partition("=")
        eng, colon, role = key.strip().partition(":")
        eng = eng.strip()
        eng = "lmdeploy" if eng == "lmdeploy-turbomind" else eng
        role = role.strip() if colon else None
        if not eng or (colon and not role):
            print(f"  [FAIL] CAGE_ISO_BYTES_LOGS key {key.strip()!r} is malformed "
                  f"(want engine or engine:role)")
            return 1
        norm_key = eng if role is None else f"{eng}:{role}"
        if norm_key in seen_keys:
            print(f"  [FAIL] CAGE_ISO_BYTES_LOGS duplicate key {norm_key!r} -- each "
                  f"engine[:role] may be pinned exactly once (silent last-wins "
                  f"would hide a log from the §6.5 gate)")
            return 1
        seen_keys.add(norm_key)
        pins.setdefault(eng, {})[role] = Path(path.strip())
    for eng, eng_pins in pins.items():
        if None in eng_pins and len(eng_pins) > 1:
            print(f"  [FAIL] CAGE_ISO_BYTES_LOGS mixes bare {eng!r} with role-split "
                  f"{eng}:<role> entries -- ambiguous whether the bare log is the "
                  f"whole pool or another role; pin one form only")
            return 1

    pool_sum_raw = os.environ.get("CAGE_ISO_POOL_SUM_BYTES")
    pool_sum = None
    if pool_sum_raw is not None:
        try:
            pool_sum = int(pool_sum_raw.strip())
        except ValueError:
            print(f"  [FAIL] CAGE_ISO_POOL_SUM_BYTES={pool_sum_raw!r} is not an integer")
            return 1
        if pool_sum <= 0:
            print(f"  [FAIL] CAGE_ISO_POOL_SUM_BYTES={pool_sum} is not a positive "
                  f"byte count")
            return 1

    # The launchers write under CAGE_LOG_ROOT when it is set (S0F-23), so
    # discovery follows it; the gate-specific root still wins when both are set.
    log_root = Path(os.environ.get("CAGE_ISO_BYTES_LOG_ROOT")
                    or os.environ.get("CAGE_LOG_ROOT") or "logs")
    if pool_sum is not None:
        # A declared pool sum needs role-split pools to sum: explicit
        # engine:role pins, or (S0F-24) the pd launcher's record in scope.
        # Decided here, before any log is read, as before, so the legacy
        # stdout of the operator error is unchanged (review 2026-10-06).
        role_pinned = any(role is not None
                          for eng in engines for role in pins.get(eng, {}))
        pd_recorded = any(current_record_paths(log_root, eng)[1].is_file()
                          for eng in engines)
        if not role_pinned and not pd_recorded:
            print("  [FAIL] CAGE_ISO_POOL_SUM_BYTES is set but CAGE_ISO_BYTES_LOGS "
                  "pins no role-split (engine:role) log in scope -- a declared "
                  "pool sum with nothing to sum is operator error; drop the env "
                  "or pin the role logs")
            return 1
    readings, ok = [], True
    for engine in engines:
        engine_pins = pins.get(engine, {})
        role_pins = {r: p for r, p in engine_pins.items() if r is not None}
        # S0F-24: with no explicit pin, the launcher's own record of the
        # running engine comes before any mtime discovery.
        current = None
        if not engine_pins:
            try:
                current = read_current(log_root, engine)
            except IsoBytesError as exc:
                print(f"  [FAIL] {engine}: {exc}")
                ok = False
                continue
        pd_current = current if current is not None and current["record"].get("topology") == "pd" else None
        if not role_pins and pd_current is None:
            if current is not None:
                rec = current["record"]
                reading = _reading_from_record(engine, rec, current["path"])
                readings.append(reading)
                note_multi_start(engine, reading, reading["log"])
                size, toks = _fmt(reading)
                print(f"  [pool] {engine}: bytes={size} tokens={toks} "
                      f"capture={current['path']} captured={rec.get('captured_utc')} "
                      f"log={reading['log']}")
                continue
            path = engine_pins.get(None) or newest_log(log_root, engine)
            if path is None or not path.is_file():
                print(f"  [FAIL] {engine}: no startup log under {log_root / engine}/ "
                      f"-- launch the engine via scripts/2_serving/ first or pin "
                      f"CAGE_ISO_BYTES_LOGS; scope down CAGE_PREFLIGHT_BACKENDS "
                      f"ONLY as a recorded deviation")
                ok = False
                continue
            try:
                reading = parse_engine_log(
                    engine, path.read_text(encoding="utf-8", errors="replace"))
            except IsoBytesError as exc:
                print(f"  [FAIL] {engine}: {exc} (log: {path})")
                ok = False
                continue
            reading["log"] = str(path)
            readings.append(reading)
            note_multi_start(engine, reading, path)
            size, toks = _fmt(reading)
            # mtime printed so a STALE log (older budget point) is visible in the run log.
            print(f"  [pool] {engine}: bytes={size} tokens={toks} "
                  f"log={path} mtime={path.stat().st_mtime:.0f}")
            continue
        # Role-split (P/D disaggregation): parse each role log alone, then the
        # engine enters cross-engine parity as the SUM of its role pools
        # (charter §6.5: budget B = pools SUMMED). Role logs are explicit
        # pins, or (S0F-24) the pd launcher's own record of both roles.
        parts, engine_ok = [], True
        if pd_current is not None:
            for role, rec in (pd_current["record"].get("roles") or {}).items():
                label = f"{engine}:{role}"
                part = _reading_from_record(engine, rec, pd_current["path"])
                parts.append(part)
                note_multi_start(label, part, part["log"])
                size, toks = _fmt(part)
                print(f"  [pool] {label}: bytes={size} tokens={toks} "
                      f"capture={pd_current['path']} log={part['log']}")
            if not parts:
                print(f"  [FAIL] {engine}: launcher capture {pd_current['path']} carries "
                      f"no roles (S0F-24)")
                ok = False
                continue
        else:
            for role, path in role_pins.items():
                label = f"{engine}:{role}"
                if not path.is_file():
                    print(f"  [FAIL] {label}: pinned role log {path} does not exist -- "
                          f"every engine:role entry in CAGE_ISO_BYTES_LOGS must point "
                          f"at a real startup log")
                    engine_ok = False
                    continue
                try:
                    part = parse_engine_log(
                        engine, path.read_text(encoding="utf-8", errors="replace"))
                except IsoBytesError as exc:
                    print(f"  [FAIL] {label}: {exc} (log: {path})")
                    engine_ok = False
                    continue
                part["log"] = str(path)
                parts.append(part)
                note_multi_start(label, part, path)
                size, toks = _fmt(part)
                print(f"  [pool] {label}: bytes={size} tokens={toks} "
                      f"log={path} mtime={path.stat().st_mtime:.0f}")
        if not engine_ok:
            ok = False
            continue
        # A channel sums only when EVERY role log carries it -- summing a
        # present value with an absent one would fabricate a pool size.
        agg_bytes = (None if any(p["bytes"] is None for p in parts)
                     else sum(p["bytes"] for p in parts))
        agg_tokens = (None if any(p["tokens"] is None for p in parts)
                      else sum(p["tokens"] for p in parts))
        if agg_bytes is None and agg_tokens is None:
            print(f"  [FAIL] {engine}: role logs share no common channel (one "
                  f"reports only bytes, another only tokens) -- the §6.5 pool "
                  f"SUM cannot be formed; capture log variants carrying the "
                  f"missing channel")
            ok = False
            continue
        readings.append({
            "engine": engine,
            "bytes": agg_bytes,
            "tokens": agg_tokens,
            "evidence": [line for p in parts for line in p["evidence"]],
            "log": ",".join(p["log"] for p in parts),
            "pd_roles": len(parts),
        })
        size = "n/a" if agg_bytes is None else f"{agg_bytes / GIB:.3f} GiB"
        toks = "n/a" if agg_tokens is None else str(agg_tokens)
        print(f"  [pool] {engine}: SUM of {len(parts)} role pools -> "
              f"bytes={size} tokens={toks}")
    if pool_sum is not None:
        # Charter §6.5 budget assertion: each role-split engine's realized
        # pools must SUM to the planned budget B (cache_budget.py emits it as
        # gate_j.expected_bytes_total). Bytes-denominated by definition, so
        # token-only role logs cannot certify it and FAIL.
        for reading in readings:
            if "pd_roles" not in reading:
                continue
            engine = reading["engine"]
            if reading["bytes"] is None:
                print(f"  [FAIL] {engine}: CAGE_ISO_POOL_SUM_BYTES declared but the "
                      f"role logs carry no bytes channel -- a BYTES pool sum "
                      f"cannot be certified from token-only logs")
                ok = False
                continue
            gap = relative_gap(reading["bytes"], pool_sum)
            if gap <= tol:
                print(f"  [PASS] {engine}: pool SUM {reading['bytes']} vs declared "
                      f"CAGE_ISO_POOL_SUM_BYTES={pool_sum}: gap {gap:.4f} <= tol {tol}")
            else:
                print(f"  [FAIL] {engine}: pool SUM {reading['bytes']} vs declared "
                      f"CAGE_ISO_POOL_SUM_BYTES={pool_sum}: gap {gap:.4f} > tol {tol} "
                      f"-- realized P/D pools do NOT sum to the planned budget B "
                      f"(charter §6.5); fix the pool split before spending GPU time")
                ok = False
    if not ok:
        return 1
    if len(readings) < 2:
        print(f"  [note] single-engine scope ({raw!r}): pairwise parity is "
              f"vacuous; realized pool recorded above. The FULL §6.5 gate "
              f"needs all final-scope engines in CAGE_PREFLIGHT_BACKENDS.")
        print("  [PASS] iso-bytes gate (vacuous parity; realized pool recorded)")
        return 0
    for i in range(len(readings)):
        for j in range(i + 1, len(readings)):
            ra, rb = readings[i], readings[j]
            try:
                basis, gap, within = compare_pair(ra, rb, tol)
            except IsoBytesError as exc:
                print(f"  [FAIL] {exc}")
                ok = False
                continue
            proxy = (" (PROXY basis: tokens certify §6.5 only at uniform KV "
                     "dtype)" if basis == "tokens" else "")
            if within:
                print(f"  [PASS] {ra['engine']} vs {rb['engine']}: {basis} gap "
                      f"{gap:.4f} <= tol {tol}{proxy}")
            else:
                print(f"  [FAIL] {ra['engine']} vs {rb['engine']}: {basis} gap "
                      f"{gap:.4f} > tol {tol} -- realized KV pools are NOT "
                      f"iso-bytes; fix the dial mapping in "
                      f"scripts/lib/_serving_config.sh before spending GPU "
                      f"time{proxy}")
                ok = False
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# The launcher CLI: kv_pool_log.py capture ...
# ---------------------------------------------------------------------------


def _drop_stale_record(out) -> None:
    """A failed capture leaves NO record. A record from an earlier start that
    outlived its engine (a crash, no ``stop``) would otherwise be read by
    gate (j) as the running engine's pool (review 2026-10-06, HIGH-1); the
    launchers also remove it before they spawn."""
    path = Path(out)
    if path.is_file():
        path.unlink()
        print(f"  [note] removed the stale launcher capture {path} "
              f"(this start's pool is not recorded; gate (j) reads the log)")


def capture_main(argv: List[str]) -> int:
    """``capture --engine E (--log PATH | --role NAME=PATH ...) --out PATH
    [--merge-into [ROLE=]CFG ...] [--retries N] [--interval S]``.

    Exit 0 with the record written; 3 when no pool line parsed (a warning,
    any stale record at ``--out`` removed, the launcher continues and the
    gate falls back to the log); 2 on usage.
    """
    parser = argparse.ArgumentParser(
        prog="kv_pool_log.py capture",
        description="Record the realized KV pool of one engine start (S0F-24, ADR-0142).",
    )
    parser.add_argument("--engine", required=True, help="vllm | sglang | lmdeploy")
    parser.add_argument("--log", help="the start log of a single-instance engine")
    parser.add_argument("--role", action="append", default=[], metavar="NAME=PATH",
                        help="a role's start log (pd pair: --role prefill=... --role decode=...)")
    parser.add_argument("--out", required=True, help="the record to write (atomic)")
    parser.add_argument("--merge-into", action="append", default=[], metavar="[ROLE=]CFG",
                        help="serving-config JSON to receive the realized pool")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S)
    args = parser.parse_args(argv)
    engine = "lmdeploy" if args.engine == "lmdeploy-turbomind" else args.engine
    if bool(args.log) == bool(args.role):
        parser.error("give exactly one of --log PATH or --role NAME=PATH (repeatable)")
    if args.retries < 0 or args.interval < 0:
        parser.error("--retries and --interval must be non-negative")
    roles: Dict[str, str] = {}
    for item in args.role:
        name, sep, path = item.partition("=")
        if not sep or not name.strip() or not path.strip():
            parser.error(f"--role {item!r} is not NAME=PATH")
        if name.strip() in roles:
            parser.error(f"--role {name.strip()!r} given twice")
        roles[name.strip()] = path.strip()
    merges: Dict[Optional[str], str] = {}
    for item in args.merge_into:
        name, sep, path = item.partition("=")
        if args.log:
            # A single-instance capture takes the whole value as the CFG
            # path; a path may contain "=" (review 2026-10-06, LOW-5).
            merges[None] = item.strip()
        else:
            if not sep or name.strip() not in roles:
                parser.error(f"--merge-into {item!r} must be ROLE=CFG for one of the given roles")
            merges[name.strip()] = path.strip()

    targets: List["tuple[Optional[str], dict]"] = []
    try:
        if args.log:
            rec = capture_record(engine, args.log, retries=args.retries, interval_s=args.interval)
            record = rec
            targets.append((None, rec))
        else:
            role_recs: Dict[str, dict] = {}
            for role, path in roles.items():
                try:
                    role_recs[role] = capture_record(
                        engine, path, role=role, retries=args.retries, interval_s=args.interval)
                except IsoBytesError as exc:
                    raise IsoBytesError(f"role {role} ({path}): {exc}") from exc
            record = pd_record(engine, role_recs)
            targets.extend(role_recs.items())
    except IsoBytesError as exc:
        print(f"  [warn] KV pool capture: {exc}")
        _drop_stale_record(args.out)
        return 3
    out = write_record(args.out, record)
    for role, rec in targets:
        cfg = merges.get(role)
        if cfg:
            merge_into_serving_config(cfg, rec)
        size, toks = _fmt(rec)
        label = engine if role is None else f"{engine}:{role}"
        print(f"  KV pool captured: {label} bytes={size} tokens={toks} (log {rec['log']}) -> {out}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "capture":
        sys.exit(capture_main(sys.argv[2:]))
    print("usage: kv_pool_log.py capture --engine E (--log PATH | --role NAME=PATH ...) "
          "--out PATH [--merge-into [ROLE=]CFG ...] [--retries N] [--interval S]  "
          "(gate (j) imports main from preflight_check.sh)", file=sys.stderr)
    sys.exit(2)
