"""Open-loop load generator for the D6 pressure campaign.

Charter authority: MyDocs/PUBLICATION.md D6 §6.1 ("Load model: open-loop Poisson
arrivals, pre-registered rates, fixed pre-costed window durations (NO
CI-stopping)"), §6.3 (Graft B: "coordinated-omission-corrected timestamps
(latencies clocked from INTENDED open-loop arrival times)"), and §6.8 (pruning
rule: full 6-fraction rate grid on Group A, reduced 3-fraction grid on B/C/D).

Methodology citations
---------------------
- Open-loop vs closed-loop arrival processes: Schroeder, Wierman &
  Harchol-Balter, "Open Versus Closed: A Cautionary Tale", NSDI 2006. A
  closed-loop generator (send, wait for completion, send next) lets the system
  under test throttle its own offered load; an open-loop generator issues each
  request at a pre-drawn arrival time REGARDLESS of completions, which is the
  only arrival model under which offered rate is an independent variable.
- Poisson open-loop arrivals at pre-registered rate fractions of a predicted
  saturation rate, as used for LLM-serving evaluation: Zhong et al., "DistServe:
  Disaggregating Prefill and Decoding for Goodput-optimized Large Language
  Model Serving", OSDI 2024 (arXiv:2401.09670) [zhong2024distserve]; Agrawal et
  al., "Taming Throughput-Latency Tradeoff in LLM Inference with
  Sarathi-Serve", OSDI 2024 (arXiv:2403.02310) [agrawal2024sarathi].
- Coordinated omission: Tene, "How NOT to Measure Latency" (Strange Loop 2015).
  If latency is clocked from the ACTUAL send time while sends are delayed by a
  backed-up system (or by a client-side concurrency gate), the delay silently
  vanishes from the latency distribution. Therefore every request here records
  both its INTENDED arrival time (``scheduled_ts``) and its actual send time
  (``actual_send_ts``), and **latency for SLO purposes MUST be measured from
  ``scheduled_ts``** (``latency_from_scheduled_ms``), per D6 §6.3.
- Warmup / transient removal for the measurement window: Jain, "The Art of
  Computer Systems Performance Analysis", Wiley 1991, ch. 25 (D6 §6.3 "Jain
  warmup removal").

Fail-closed doctrine
--------------------
Invalid configuration raises the typed :class:`LoadGeneratorError` (mirroring
``InstrumentUnavailableError`` in ``src/evaluation/quality.py``) — never a
silent default. The optional max-in-flight SAFETY cap never gates silently:
every request affected by the cap is flagged (``delayed_by_cap`` /
``dropped_by_cap``) and counted in ``DispatchReport.dropped_or_delayed``,
because silent client-side gating reintroduces coordinated omission.

This module is pure orchestration: the dispatcher takes an async callable per
request (e.g. a closure over ``adapter.async_stream_generate``, the STREAMING
path — required by the runner's ``dispatch_open_loop`` so per-row TTFT is a
real first-token time, never the non-streaming full-response proxy), so unit
tests drive it with stub engines and no network.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Final, List, Literal, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "D6_RATE_FRACTIONS",
    "D6_REDUCED_RATE_FRACTIONS",
    "LoadGeneratorError",
    "ArrivalSchedule",
    "generate_arrival_schedule",
    "build_rate_grid_schedules",
    "RequestRecord",
    "DispatchReport",
    "OpenLoopDispatcher",
    "ensure_no_measured_replay",
    "trim_to_measurement_window",
]

# PUBLICATION.md D6 §6.1: offered rates are pre-registered fractions of the
# PREDICTED saturation rate lambda* per (model, engine, budget).
D6_RATE_FRACTIONS: Final[Tuple[float, ...]] = (0.5, 0.7, 0.85, 0.95, 1.05, 1.2)

# PUBLICATION.md D6 §6.8 (pruning rule, 2026-08-02): Groups B, C, D run the
# reduced rate grid {0.85, 0.95, 1.05}·lambda*.
D6_REDUCED_RATE_FRACTIONS: Final[Tuple[float, ...]] = (0.85, 0.95, 1.05)

_BIT_GENERATOR: Final[str] = "PCG64"

Distribution = Literal["poisson", "deterministic"]
CapPolicy = Literal["delay", "drop"]

# Async send callable: takes the schedule index of the request and performs
# the actual issue. It may OPTIONALLY accept a second positional argument — a
# zero-arg ``on_first_token`` callback provided by the dispatcher: a send that
# forwards it into a STREAMING engine path (e.g.
# ``adapter.async_stream_generate(req, on_first_token=cb)``) lets the
# dispatcher stamp ``first_token_ts`` on ITS OWN injectable clock at the first
# content delta, from which both ``ttft_from_send_ms`` and the D6 §6.3
# coordinated-omission-corrected ``ttft_from_scheduled_ms`` (clocked from the
# INTENDED arrival) are computed. Single-argument sends remain fully
# supported; their records simply carry no TTFT columns (absence is honest —
# a TTFT is never fabricated from a non-streaming completion time).
AsyncSendFn = Callable[..., Awaitable[Any]]


class LoadGeneratorError(RuntimeError):
    """Typed fail-closed error for invalid load-generator configuration.

    Mirrors the ``InstrumentUnavailableError`` doctrine in
    ``src/evaluation/quality.py``: a bad parameter raises immediately with the
    offending parameter named — it is never silently coerced or defaulted,
    because a silently altered arrival process voids the pre-registered D6
    load model.
    """

    def __init__(self, parameter: str, value: Any, reason: str) -> None:
        self.parameter = parameter
        self.value = value
        self.reason = reason
        super().__init__(
            f"load generator parameter '{parameter}'={value!r} invalid: {reason}"
        )


def _require_positive_finite(parameter: str, value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise LoadGeneratorError(parameter, value, "must be a real number")
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise LoadGeneratorError(parameter, value, "must be finite and > 0")


# --------------------------------------------------------------------------- #
# Arrival-schedule generation (pre-drawn, seeded, deterministic)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ArrivalSchedule:
    """A pre-drawn open-loop arrival schedule.

    ``offsets_s`` are the PLANNED arrival times in seconds relative to run
    start (t=0), strictly increasing. For the Poisson distribution they are the
    cumulative sum of seeded Exponential(1/rate) inter-arrival gaps — the
    arrival times of a homogeneous Poisson process at ``rate_qps``
    (zhong2024distserve / agrawal2024sarathi open-loop methodology, per
    D6 §6.1). The schedule is fully deterministic given ``seed``: the seed and
    bit generator are recorded here for the run manifest.
    """

    rate_qps: float
    seed: int
    distribution: Distribution
    offsets_s: Tuple[float, ...]
    duration_s: Optional[float] = None
    n_requests_requested: Optional[int] = None
    # D6 grid provenance (populated by build_rate_grid_schedules).
    rate_frac: Optional[float] = None
    lambda_star_qps: Optional[float] = None
    base_seed: Optional[int] = None
    bit_generator: str = _BIT_GENERATOR

    def __len__(self) -> int:
        return len(self.offsets_s)

    @property
    def n_arrivals(self) -> int:
        return len(self.offsets_s)

    @property
    def span_s(self) -> float:
        """Time from run start to the last planned arrival (0.0 if empty)."""
        return self.offsets_s[-1] if self.offsets_s else 0.0

    def to_manifest(self) -> Dict[str, Any]:
        """Provenance record for run_manifest.json (schedule is reconstructible
        from ``seed`` + parameters; offsets themselves are not embedded)."""
        return {
            "rate_qps": self.rate_qps,
            "seed": self.seed,
            "distribution": self.distribution,
            "duration_s": self.duration_s,
            "n_requests_requested": self.n_requests_requested,
            "n_arrivals": self.n_arrivals,
            "rate_frac": self.rate_frac,
            "lambda_star_qps": self.lambda_star_qps,
            "base_seed": self.base_seed,
            "bit_generator": self.bit_generator,
        }


def generate_arrival_schedule(
    rate_qps: float,
    *,
    seed: int,
    duration_s: Optional[float] = None,
    n_requests: Optional[int] = None,
    distribution: Distribution = "poisson",
) -> ArrivalSchedule:
    """Pre-draw an open-loop arrival schedule (D6 §6.1 load model).

    Exactly one of ``duration_s`` (fixed pre-costed window — the charter
    default: "fixed pre-costed window durations (NO CI-stopping)") or
    ``n_requests`` (exact request count) must be given.

    - ``poisson``: inter-arrival gaps ~ Exponential(mean=1/rate_qps) drawn from
      a seeded ``numpy.random.Generator(PCG64(seed))``, cumulative-summed. The
      first arrival is at the first gap (> 0), per the standard homogeneous
      Poisson process started at t=0.
    - ``deterministic``: uniform spacing, arrival k at k/rate_qps (k >= 1) — a
      degenerate-variance control, not a substitute for the registered Poisson
      load model.

    In ``duration_s`` mode only arrivals strictly inside [0, duration_s) are
    kept; a zero-arrival schedule is a valid (if unlucky) draw at low
    rate x duration.
    """
    _require_positive_finite("rate_qps", rate_qps)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise LoadGeneratorError("seed", seed, "must be an int (recorded for provenance)")
    if seed < 0:
        raise LoadGeneratorError("seed", seed, "must be >= 0")
    if distribution not in ("poisson", "deterministic"):
        raise LoadGeneratorError(
            "distribution", distribution, "must be 'poisson' or 'deterministic'"
        )
    if (duration_s is None) == (n_requests is None):
        raise LoadGeneratorError(
            "duration_s/n_requests",
            (duration_s, n_requests),
            "exactly one of duration_s or n_requests must be provided",
        )
    if duration_s is not None:
        _require_positive_finite("duration_s", duration_s)
    if n_requests is not None:
        if not isinstance(n_requests, int) or isinstance(n_requests, bool):
            raise LoadGeneratorError("n_requests", n_requests, "must be an int")
        if n_requests <= 0:
            raise LoadGeneratorError("n_requests", n_requests, "must be > 0")

    rate = float(rate_qps)
    if distribution == "deterministic":
        if n_requests is not None:
            offsets = tuple((k + 1) / rate for k in range(n_requests))
        else:
            assert duration_s is not None
            n = int(math.ceil(float(duration_s) * rate)) + 1
            offsets = tuple(
                t for t in ((k + 1) / rate for k in range(n)) if t < float(duration_s)
            )
    else:
        rng = np.random.Generator(np.random.PCG64(seed))
        scale = 1.0 / rate  # mean inter-arrival time
        if n_requests is not None:
            gaps = rng.exponential(scale, size=n_requests)
            offsets = tuple(float(t) for t in np.cumsum(gaps))
        else:
            assert duration_s is not None
            horizon = float(duration_s)
            chunk = max(16, int(math.ceil(rate * horizon)) + 16)
            arrivals: np.ndarray = np.empty(0, dtype=np.float64)
            last = 0.0
            while last < horizon:
                gaps = rng.exponential(scale, size=chunk)
                segment = last + np.cumsum(gaps)
                arrivals = np.concatenate([arrivals, segment])
                last = float(segment[-1])
            offsets = tuple(float(t) for t in arrivals[arrivals < horizon])

    return ArrivalSchedule(
        rate_qps=rate,
        seed=seed,
        distribution=distribution,
        offsets_s=offsets,
        duration_s=float(duration_s) if duration_s is not None else None,
        n_requests_requested=n_requests,
    )


def build_rate_grid_schedules(
    lambda_star_qps: float,
    *,
    seed: int,
    duration_s: Optional[float] = None,
    n_requests: Optional[int] = None,
    fractions: Sequence[float] = D6_RATE_FRACTIONS,
    distribution: Distribution = "poisson",
) -> Dict[float, ArrivalSchedule]:
    """One schedule per D6 rate-grid point (fraction of predicted lambda*).

    D6 §6.1: rates are pre-registered fractions {0.5, 0.7, 0.85, 0.95, 1.05,
    1.2} of the PREDICTED saturation rate lambda* (rates above lambda* are grid
    points, not failures). Pass ``fractions=D6_REDUCED_RATE_FRACTIONS`` for the
    §6.8 reduced grid (Groups B/C/D).

    Per-fraction child seeds are derived deterministically from the base
    ``seed`` via ``numpy.random.SeedSequence(seed).generate_state`` so grid
    points draw independent arrival processes while the whole grid remains
    reproducible from one recorded base seed.
    """
    _require_positive_finite("lambda_star_qps", lambda_star_qps)
    fracs = tuple(fractions)
    if not fracs:
        raise LoadGeneratorError("fractions", fractions, "must be non-empty")
    for f in fracs:
        _require_positive_finite("fractions[i]", f)
    if len(set(fracs)) != len(fracs):
        raise LoadGeneratorError("fractions", fractions, "must not contain duplicates")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise LoadGeneratorError("seed", seed, "must be an int >= 0")

    child_seeds = np.random.SeedSequence(seed).generate_state(len(fracs))
    grid: Dict[float, ArrivalSchedule] = {}
    for frac, child in zip(fracs, child_seeds):
        base = generate_arrival_schedule(
            float(frac) * float(lambda_star_qps),
            seed=int(child),
            duration_s=duration_s,
            n_requests=n_requests,
            distribution=distribution,
        )
        grid[float(frac)] = ArrivalSchedule(
            rate_qps=base.rate_qps,
            seed=base.seed,
            distribution=base.distribution,
            offsets_s=base.offsets_s,
            duration_s=base.duration_s,
            n_requests_requested=base.n_requests_requested,
            rate_frac=float(frac),
            lambda_star_qps=float(lambda_star_qps),
            base_seed=seed,
        )
    return grid


# --------------------------------------------------------------------------- #
# Coordinated-omission-safe per-request accounting
# --------------------------------------------------------------------------- #


@dataclass
class RequestRecord:
    """Per-request open-loop accounting row.

    Coordinated-omission safety (Tene, "How NOT to Measure Latency", 2015;
    D6 §6.3): ``scheduled_ts`` is the INTENDED arrival instant on the
    dispatcher's clock; ``actual_send_ts`` is when the request actually left;
    ``scheduler_lag_ms`` is their gap. **Latency for SLO purposes is
    ``latency_from_scheduled_ms``** (completion minus scheduled), never
    ``latency_from_send_ms``, so client-side delay can never hide server-side
    congestion.
    """

    index: int
    scheduled_offset_s: float
    scheduled_ts: float
    actual_send_ts: Optional[float] = None
    scheduler_lag_ms: Optional[float] = None
    # First-token instant on the DISPATCHER'S clock (streaming sends only):
    # stamped by the ``on_first_token`` hook at the first non-empty content
    # delta. ``ttft_from_scheduled_ms`` (first token minus INTENDED arrival)
    # is the §6.3 coordinated-omission-corrected TTFT for SLO purposes;
    # ``ttft_from_send_ms`` is the conventional client-clock TTFT.
    first_token_ts: Optional[float] = None
    ttft_from_send_ms: Optional[float] = None
    ttft_from_scheduled_ms: Optional[float] = None
    completion_ts: Optional[float] = None
    latency_from_scheduled_ms: Optional[float] = None
    latency_from_send_ms: Optional[float] = None
    in_flight_at_send: Optional[int] = None
    delayed_by_cap: bool = False
    dropped_by_cap: bool = False
    error: Optional[str] = None
    result: Any = None

    @property
    def sent(self) -> bool:
        return self.actual_send_ts is not None

    @property
    def completed(self) -> bool:
        return self.sent and self.completion_ts is not None and self.error is None

    def to_row(self) -> Dict[str, Any]:
        """Row fields for the results schema.

        ``arrival_s`` is the INTENDED arrival offset from run start — exactly
        what ``src/analysis/goodput.py`` requires ("arrival_s = INTENDED
        open-loop arrival per §6.3").
        """
        return {
            "arrival_s": self.scheduled_offset_s,
            "scheduled_ts": self.scheduled_ts,
            "actual_send_ts": self.actual_send_ts,
            "scheduler_lag_ms": self.scheduler_lag_ms,
            "first_token_ts": self.first_token_ts,
            "ttft_from_send_ms": self.ttft_from_send_ms,
            "ttft_from_scheduled_ms": self.ttft_from_scheduled_ms,
            "completion_ts": self.completion_ts,
            "latency_from_scheduled_ms": self.latency_from_scheduled_ms,
            "latency_from_send_ms": self.latency_from_send_ms,
            "in_flight_at_send": self.in_flight_at_send,
            "delayed_by_cap": self.delayed_by_cap,
            "dropped_by_cap": self.dropped_by_cap,
            "dispatch_error": self.error,
        }


@dataclass
class DispatchReport:
    """Outcome of one open-loop dispatch run."""

    schedule: ArrivalSchedule
    run_start_ts: float
    records: List[RequestRecord] = field(default_factory=list)
    dropped_or_delayed: int = 0
    max_in_flight_observed: int = 0

    @property
    def n_scheduled(self) -> int:
        return len(self.records)

    @property
    def n_sent(self) -> int:
        return sum(1 for r in self.records if r.sent)

    @property
    def n_completed(self) -> int:
        return sum(1 for r in self.records if r.completed)

    @property
    def n_errors(self) -> int:
        return sum(1 for r in self.records if r.error is not None)

    @property
    def n_dropped(self) -> int:
        return sum(1 for r in self.records if r.dropped_by_cap)

    def to_manifest(self) -> Dict[str, Any]:
        return {
            "schedule": self.schedule.to_manifest(),
            "run_start_ts": self.run_start_ts,
            "n_scheduled": self.n_scheduled,
            "n_sent": self.n_sent,
            "n_completed": self.n_completed,
            "n_errors": self.n_errors,
            "n_dropped": self.n_dropped,
            "dropped_or_delayed": self.dropped_or_delayed,
            "max_in_flight_observed": self.max_in_flight_observed,
        }


# --------------------------------------------------------------------------- #
# Open-loop asyncio dispatcher
# --------------------------------------------------------------------------- #


@dataclass
class _RunState:
    in_flight: int = 0
    max_in_flight_observed: int = 0
    dropped_or_delayed: int = 0


class OpenLoopDispatcher:
    """Issue each request AT its pre-drawn arrival time, never waiting for
    completions (Schroeder et al., NSDI 2006 open-loop arrival model; D6 §6.1).

    The coordinator loop only sleeps until the next scheduled arrival and
    spawns an ``asyncio`` task per request; in-flight requests never delay
    later arrivals. The optional ``max_in_flight`` SAFETY cap protects the
    client host from unbounded task pileup past the cliff — but it NEVER gates
    silently: a request that finds the cap saturated at its arrival instant is
    either delayed (``cap_policy='delay'``: waits for a slot, flagged
    ``delayed_by_cap``, its ``scheduler_lag_ms`` exposes the wait) or dropped
    (``cap_policy='drop'``: never sent, flagged ``dropped_by_cap``), and both
    outcomes increment ``DispatchReport.dropped_or_delayed``. Silent gating
    would reintroduce coordinated omission (Tene 2015): the client would
    throttle offered load exactly when the system is saturated, hiding the
    congestion the D6 grid exists to measure. Latency for SLO purposes is
    always measured from ``scheduled_ts`` (D6 §6.3).

    ``clock`` and ``sleep`` are injectable for pure-unit testing (fake clock,
    no network); defaults are ``time.monotonic`` and ``asyncio.sleep``.
    """

    def __init__(
        self,
        *,
        max_in_flight: Optional[int] = None,
        cap_policy: CapPolicy = "delay",
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if max_in_flight is not None:
            if not isinstance(max_in_flight, int) or isinstance(max_in_flight, bool):
                raise LoadGeneratorError("max_in_flight", max_in_flight, "must be an int or None")
            if max_in_flight <= 0:
                raise LoadGeneratorError("max_in_flight", max_in_flight, "must be > 0")
        if cap_policy not in ("delay", "drop"):
            raise LoadGeneratorError("cap_policy", cap_policy, "must be 'delay' or 'drop'")
        if not callable(clock):
            raise LoadGeneratorError("clock", clock, "must be callable")
        if not callable(sleep):
            raise LoadGeneratorError("sleep", sleep, "must be callable")
        self._max_in_flight = max_in_flight
        self._cap_policy: CapPolicy = cap_policy
        self._clock = clock
        self._sleep = sleep

    async def run(self, schedule: ArrivalSchedule, send: AsyncSendFn) -> DispatchReport:
        """Dispatch every request in ``schedule`` through ``send`` open-loop.

        ``send(i)`` performs the actual issue for schedule index ``i``. A send
        that also accepts a second positional argument receives a zero-arg
        ``on_first_token`` callback: forwarding it into a streaming engine
        path (``adapter.async_stream_generate``) stamps ``first_token_ts`` on
        the dispatcher's clock, from which ``ttft_from_send_ms`` and the §6.3
        ``ttft_from_scheduled_ms`` are computed (see ``AsyncSendFn``).
        Exceptions raised by ``send`` are captured per-record (``error``) and
        counted — a failed request is data (it counts against D6 §6.1
        attainment), not a run abort.
        """
        if not isinstance(schedule, ArrivalSchedule):
            raise LoadGeneratorError("schedule", schedule, "must be an ArrivalSchedule")
        if not callable(send):
            raise LoadGeneratorError("send", send, "must be an async callable")
        send_accepts_hook = self._send_accepts_hook(send)

        state = _RunState()
        semaphore = (
            asyncio.Semaphore(self._max_in_flight) if self._max_in_flight is not None else None
        )
        run_start = self._clock()
        report = DispatchReport(schedule=schedule, run_start_ts=run_start)
        tasks: List["asyncio.Task[None]"] = []

        for index, offset in enumerate(schedule.offsets_s):
            scheduled_ts = run_start + offset
            delay = scheduled_ts - self._clock()
            if delay > 0:
                # Open-loop: the ONLY thing the coordinator waits on is the
                # wall clock — never a completion, never the cap.
                await self._sleep(delay)
            record = RequestRecord(
                index=index,
                scheduled_offset_s=offset,
                scheduled_ts=scheduled_ts,
            )
            report.records.append(record)
            tasks.append(
                asyncio.create_task(
                    self._issue(record, send, semaphore, state, send_accepts_hook)
                )
            )

        if tasks:
            await asyncio.gather(*tasks)

        report.dropped_or_delayed = state.dropped_or_delayed
        report.max_in_flight_observed = state.max_in_flight_observed
        return report

    @staticmethod
    def _send_accepts_hook(send: AsyncSendFn) -> bool:
        """True when ``send`` can take the ``on_first_token`` callback as a
        second positional argument (see ``AsyncSendFn``). Introspection is
        done ONCE per run; an un-introspectable callable is treated as
        single-argument (the conservative, always-callable form)."""
        try:
            sig = inspect.signature(send)
        except (TypeError, ValueError):
            return False
        positional = [
            p
            for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        has_var_positional = any(
            p.kind == p.VAR_POSITIONAL for p in sig.parameters.values()
        )
        return has_var_positional or len(positional) >= 2

    async def _issue(
        self,
        record: RequestRecord,
        send: AsyncSendFn,
        semaphore: Optional[asyncio.Semaphore],
        state: _RunState,
        send_accepts_hook: bool,
    ) -> None:
        if semaphore is not None and semaphore.locked():
            # SAFETY cap saturated at this request's arrival instant. Record
            # LOUDLY (counter + per-request flag) — never gate silently, which
            # would reintroduce coordinated omission (Tene 2015).
            state.dropped_or_delayed += 1
            if self._cap_policy == "drop":
                record.dropped_by_cap = True
                return
            record.delayed_by_cap = True
        if semaphore is not None:
            await semaphore.acquire()
        try:
            state.in_flight += 1
            state.max_in_flight_observed = max(state.max_in_flight_observed, state.in_flight)
            record.in_flight_at_send = state.in_flight
            record.actual_send_ts = self._clock()
            record.scheduler_lag_ms = (record.actual_send_ts - record.scheduled_ts) * 1000.0

            def _mark_first_token() -> None:
                # Stamped on the dispatcher's OWN clock (injectable) at the
                # first non-empty content delta — the raw timestamp both TTFT
                # columns derive from (D6 §6.3).
                if record.first_token_ts is None:
                    record.first_token_ts = self._clock()

            try:
                if send_accepts_hook:
                    record.result = await send(record.index, _mark_first_token)
                else:
                    record.result = await send(record.index)
            except Exception as exc:  # noqa: BLE001 — per-request failures are data
                record.error = f"{type(exc).__name__}: {exc}"
            record.completion_ts = self._clock()
            # Coordinated-omission-safe latency: clocked from the INTENDED
            # arrival (D6 §6.3), not from the actual send.
            record.latency_from_scheduled_ms = (
                record.completion_ts - record.scheduled_ts
            ) * 1000.0
            record.latency_from_send_ms = (
                record.completion_ts - record.actual_send_ts
            ) * 1000.0
            if record.first_token_ts is not None:
                record.ttft_from_send_ms = (
                    record.first_token_ts - record.actual_send_ts
                ) * 1000.0
                # §6.3 coordinated-omission-corrected TTFT: clocked from the
                # INTENDED arrival, so client-side delay can never hide
                # server-side congestion in the first-token distribution.
                record.ttft_from_scheduled_ms = (
                    record.first_token_ts - record.scheduled_ts
                ) * 1000.0
        finally:
            state.in_flight -= 1
            if semaphore is not None:
                semaphore.release()


def ensure_no_measured_replay(
    schedule: ArrivalSchedule,
    n_unique_requests: int,
    *,
    allow_replay: bool = False,
) -> bool:
    """E4 replay pin (code-assertion walkthrough 2026-08-12): measured windows
    never replay a request unless the operator EXPLICITLY labels the run.

    Duration-mode schedules map index i to ``requests[i % n]``, so a schedule
    longer than the prepared set issues some examples MORE THAN ONCE inside
    the window. That is forbidden for confirmatory cells for two registered
    reasons: (a) duplicate example_ids meet the per-example paired joins of
    the stats layer (§9.4 pairing is per example_id), and (b) a replayed
    request re-arrives with its prefix already hot, silently shifting the
    cell's cache-locality profile mid-window. Returns False when the schedule
    fits within the unique set (no replay), True when replay WILL occur and
    ``allow_replay`` says the caller accepted it (the caller must label the
    run non-confirmatory and warn loudly); otherwise raises the typed
    fail-closed error.
    """
    if not isinstance(schedule, ArrivalSchedule):
        raise LoadGeneratorError("schedule", schedule, "must be an ArrivalSchedule")
    if not isinstance(n_unique_requests, int) or isinstance(n_unique_requests, bool):
        raise LoadGeneratorError(
            "n_unique_requests", n_unique_requests, "must be an int"
        )
    if n_unique_requests <= 0:
        raise LoadGeneratorError(
            "n_unique_requests", n_unique_requests, "must be > 0"
        )
    if schedule.n_arrivals <= n_unique_requests:
        return False
    if not allow_replay:
        raise LoadGeneratorError(
            "n_arrivals",
            schedule.n_arrivals,
            f"schedule wraps past the {n_unique_requests} unique prepared "
            f"request(s): a measured window would replay examples (duplicate "
            f"example_ids break per-example pairing; a replayed request hits "
            f"warm prefix cache and shifts the locality profile). Shorten the "
            f"window / raise the manifest size, or set CAGE_ALLOW_REPLAY=1 to "
            f"accept replay for a NON-confirmatory, labeled run",
        )
    return True


# --------------------------------------------------------------------------- #
# Warmup / measurement-window trimming (Jain 1991 ch. 25; D6 §6.3)
# --------------------------------------------------------------------------- #


def trim_to_measurement_window(
    records: Sequence[RequestRecord],
    *,
    warmup_s: float,
    measurement_s: Optional[float] = None,
) -> List[RequestRecord]:
    """Keep only requests whose INTENDED arrival falls inside the measurement
    window ``[warmup_s, warmup_s + measurement_s)`` (unbounded above when
    ``measurement_s`` is None).

    Warmup/transient removal per Jain 1991 ch. 25, required by D6 §6.3 ("Jain
    warmup removal") with fixed pre-costed window durations (D6 §6.1 — NO
    CI-stopping). Filtering keys on ``scheduled_offset_s`` (the intended
    arrival), never on send/completion times, so the trim itself cannot
    coordinate with system congestion.
    """
    if not isinstance(warmup_s, (int, float)) or isinstance(warmup_s, bool):
        raise LoadGeneratorError("warmup_s", warmup_s, "must be a real number")
    if not math.isfinite(float(warmup_s)) or float(warmup_s) < 0.0:
        raise LoadGeneratorError("warmup_s", warmup_s, "must be finite and >= 0")
    if measurement_s is not None:
        _require_positive_finite("measurement_s", measurement_s)

    lo = float(warmup_s)
    hi = lo + float(measurement_s) if measurement_s is not None else math.inf
    return [r for r in records if lo <= r.scheduled_offset_s < hi]
