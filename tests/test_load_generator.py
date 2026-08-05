"""Tests for the open-loop load generator (src/orchestration/load_generator.py).

Charter authority: MyDocs/PUBLICATION.md D6 §6.1 (open-loop Poisson load
model), §6.3 (coordinated-omission-corrected timestamps), §6.8 (reduced rate
grid). Pure-unit per the repo doctrine: no network, no GPU, no server — the
dispatcher is driven with stub async engines, short real sleeps, and a fake
clock (mirroring the mocked-transport pattern of tests/test_inference.py).
"""

from __future__ import annotations

import asyncio
import math
from typing import List

import pytest

from src.orchestration.load_generator import (
    D6_RATE_FRACTIONS,
    D6_REDUCED_RATE_FRACTIONS,
    ArrivalSchedule,
    DispatchReport,
    LoadGeneratorError,
    OpenLoopDispatcher,
    RequestRecord,
    build_rate_grid_schedules,
    generate_arrival_schedule,
    trim_to_measurement_window,
)


# --------------------------------------------------------------------------- #
# Arrival-schedule generation — happy path
# --------------------------------------------------------------------------- #


def test_poisson_schedule_deterministic_given_seed():
    a = generate_arrival_schedule(10.0, seed=42, n_requests=200)
    b = generate_arrival_schedule(10.0, seed=42, n_requests=200)
    c = generate_arrival_schedule(10.0, seed=43, n_requests=200)

    assert a.offsets_s == b.offsets_s
    assert a.offsets_s != c.offsets_s
    assert a.seed == 42
    assert a.distribution == "poisson"
    assert a.bit_generator == "PCG64"


def test_poisson_n_requests_exact_count_and_strictly_increasing():
    sched = generate_arrival_schedule(25.0, seed=7, n_requests=500)

    assert len(sched) == 500
    assert sched.n_arrivals == 500
    assert all(t > 0.0 for t in sched.offsets_s)
    assert all(b > a for a, b in zip(sched.offsets_s, sched.offsets_s[1:]))


def test_poisson_mean_interarrival_matches_rate():
    rate = 50.0
    sched = generate_arrival_schedule(rate, seed=123, n_requests=5000)
    gaps = [b - a for a, b in zip((0.0,) + sched.offsets_s, sched.offsets_s)]
    mean_gap = sum(gaps) / len(gaps)

    # Seeded draw -> deterministic; 5000 samples put the mean well within 5%.
    assert mean_gap == pytest.approx(1.0 / rate, rel=0.05)


def test_poisson_duration_mode_bounds_and_reproducibility():
    sched = generate_arrival_schedule(100.0, seed=9, duration_s=5.0)

    assert all(0.0 < t < 5.0 for t in sched.offsets_s)
    # Mean count = rate * duration = 500, sd ~ 22; [350, 650] is > 6 sigma.
    assert 350 <= len(sched) <= 650
    again = generate_arrival_schedule(100.0, seed=9, duration_s=5.0)
    assert sched.offsets_s == again.offsets_s
    assert sched.duration_s == 5.0
    assert sched.n_requests_requested is None


def test_deterministic_distribution_uniform_spacing():
    by_n = generate_arrival_schedule(4.0, seed=0, n_requests=4, distribution="deterministic")
    assert by_n.offsets_s == pytest.approx((0.25, 0.5, 0.75, 1.0))

    by_duration = generate_arrival_schedule(
        2.0, seed=0, duration_s=2.0, distribution="deterministic"
    )
    # Arrival k at k/rate, strictly inside [0, duration): 0.5, 1.0, 1.5.
    assert by_duration.offsets_s == pytest.approx((0.5, 1.0, 1.5))


def test_zero_arrival_schedule_is_valid_draw():
    # Mean inter-arrival 1000 s vs a 0.1 s window: no arrivals, and that is a
    # valid schedule, not an error.
    sched = generate_arrival_schedule(0.001, seed=5, duration_s=0.1)
    assert len(sched) == 0
    assert sched.span_s == 0.0


def test_schedule_manifest_records_seed_and_parameters():
    sched = generate_arrival_schedule(10.0, seed=99, n_requests=10)
    manifest = sched.to_manifest()

    assert manifest["seed"] == 99
    assert manifest["rate_qps"] == 10.0
    assert manifest["distribution"] == "poisson"
    assert manifest["n_arrivals"] == 10
    assert manifest["bit_generator"] == "PCG64"


# --------------------------------------------------------------------------- #
# Arrival-schedule generation — failure (fail-closed)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad_rate", [0.0, -1.0, math.inf, math.nan, "10"])
def test_invalid_rate_raises_typed_error(bad_rate):
    with pytest.raises(LoadGeneratorError):
        generate_arrival_schedule(bad_rate, seed=1, n_requests=10)


def test_neither_or_both_of_duration_and_n_requests_raises():
    with pytest.raises(LoadGeneratorError):
        generate_arrival_schedule(10.0, seed=1)
    with pytest.raises(LoadGeneratorError):
        generate_arrival_schedule(10.0, seed=1, duration_s=5.0, n_requests=10)


def test_invalid_distribution_raises():
    with pytest.raises(LoadGeneratorError):
        generate_arrival_schedule(10.0, seed=1, n_requests=10, distribution="uniform")


@pytest.mark.parametrize("bad_seed", [-1, 1.5, "42", True])
def test_invalid_seed_raises(bad_seed):
    with pytest.raises(LoadGeneratorError):
        generate_arrival_schedule(10.0, seed=bad_seed, n_requests=10)


@pytest.mark.parametrize("bad", [{"duration_s": 0.0}, {"duration_s": -2.0}, {"n_requests": 0}, {"n_requests": -3}])
def test_invalid_window_raises(bad):
    with pytest.raises(LoadGeneratorError):
        generate_arrival_schedule(10.0, seed=1, **bad)


def test_error_carries_parameter_and_reason():
    with pytest.raises(LoadGeneratorError) as excinfo:
        generate_arrival_schedule(-5.0, seed=1, n_requests=10)
    err = excinfo.value
    assert err.parameter == "rate_qps"
    assert err.value == -5.0
    assert "finite" in err.reason


# --------------------------------------------------------------------------- #
# D6 rate grid
# --------------------------------------------------------------------------- #


def test_rate_grid_maps_all_six_d6_fractions():
    lam = 8.0
    grid = build_rate_grid_schedules(lam, seed=11, n_requests=50)

    assert tuple(sorted(grid)) == tuple(sorted(D6_RATE_FRACTIONS))
    for frac, sched in grid.items():
        assert sched.rate_qps == pytest.approx(frac * lam)
        assert sched.rate_frac == frac
        assert sched.lambda_star_qps == lam
        assert sched.base_seed == 11
        assert len(sched) == 50


def test_rate_grid_reproducible_and_pointwise_independent():
    grid_a = build_rate_grid_schedules(8.0, seed=11, n_requests=50)
    grid_b = build_rate_grid_schedules(8.0, seed=11, n_requests=50)

    for frac in D6_RATE_FRACTIONS:
        assert grid_a[frac].offsets_s == grid_b[frac].offsets_s

    # Child seeds are derived per grid point: distinct seeds, distinct draws.
    child_seeds = {grid_a[frac].seed for frac in D6_RATE_FRACTIONS}
    assert len(child_seeds) == len(D6_RATE_FRACTIONS)


def test_rate_grid_reduced_fractions_for_groups_bcd():
    grid = build_rate_grid_schedules(
        8.0, seed=3, n_requests=20, fractions=D6_REDUCED_RATE_FRACTIONS
    )
    assert tuple(sorted(grid)) == tuple(sorted(D6_REDUCED_RATE_FRACTIONS))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fractions": ()},
        {"fractions": (0.5, -0.7)},
        {"fractions": (0.5, 0.5)},
    ],
)
def test_rate_grid_invalid_fractions_raise(kwargs):
    with pytest.raises(LoadGeneratorError):
        build_rate_grid_schedules(8.0, seed=1, n_requests=10, **kwargs)


def test_rate_grid_invalid_lambda_star_raises():
    with pytest.raises(LoadGeneratorError):
        build_rate_grid_schedules(0.0, seed=1, n_requests=10)


# --------------------------------------------------------------------------- #
# Open-loop dispatcher — stubs
# --------------------------------------------------------------------------- #


def _run(coro):
    return asyncio.run(coro)


async def _instant_send(index: int) -> str:
    return f"resp-{index}"


def _slow_send(delay_s: float):
    async def send(index: int) -> str:
        await asyncio.sleep(delay_s)
        return f"resp-{index}"

    return send


# --------------------------------------------------------------------------- #
# Open-loop dispatcher — happy path
# --------------------------------------------------------------------------- #


def test_open_loop_never_waits_for_completions():
    # 8 arrivals 10 ms apart against a 300 ms stub engine. A closed-loop
    # driver would serialize to ~2.4 s; open-loop finishes in ~arrival span
    # + one service time.
    sched = generate_arrival_schedule(
        100.0, seed=0, n_requests=8, distribution="deterministic"
    )
    dispatcher = OpenLoopDispatcher()

    async def scenario() -> tuple[DispatchReport, float]:
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        report = await dispatcher.run(sched, _slow_send(0.3))
        return report, loop.time() - t0

    report, elapsed = _run(scenario())

    assert report.n_scheduled == 8
    assert report.n_sent == 8
    assert report.n_completed == 8
    assert report.n_errors == 0
    assert report.dropped_or_delayed == 0
    # All 8 requests were in flight simultaneously at some point.
    assert report.max_in_flight_observed >= 4
    assert elapsed < 1.5  # far below the 2.4 s closed-loop floor
    # Arrivals tracked the schedule, not the completions.
    for record in report.records:
        assert record.scheduler_lag_ms is not None
        assert record.scheduler_lag_ms < 150.0


def test_records_are_in_schedule_order_and_carry_results():
    sched = generate_arrival_schedule(
        200.0, seed=0, n_requests=5, distribution="deterministic"
    )
    report = _run(OpenLoopDispatcher().run(sched, _instant_send))

    assert [r.index for r in report.records] == [0, 1, 2, 3, 4]
    assert [r.result for r in report.records] == [f"resp-{i}" for i in range(5)]
    assert all(r.completed for r in report.records)


def test_coordinated_omission_accounting_identity():
    # (completion - scheduled) == (completion - send) + (send - scheduled):
    # SLO latency from the INTENDED arrival always includes scheduler lag.
    sched = generate_arrival_schedule(
        100.0, seed=1, n_requests=6, distribution="deterministic"
    )
    report = _run(OpenLoopDispatcher().run(sched, _slow_send(0.02)))

    for record in report.records:
        assert record.latency_from_scheduled_ms == pytest.approx(
            record.latency_from_send_ms + record.scheduler_lag_ms, abs=1e-6
        )
        assert record.latency_from_scheduled_ms >= record.latency_from_send_ms - 1e-6


def test_to_row_exposes_arrival_s_for_goodput():
    sched = generate_arrival_schedule(
        100.0, seed=1, n_requests=3, distribution="deterministic"
    )
    report = _run(OpenLoopDispatcher().run(sched, _instant_send))

    for record, offset in zip(report.records, sched.offsets_s):
        row = record.to_row()
        # arrival_s = INTENDED open-loop arrival per D6 §6.3 -- the exact
        # column src/analysis/goodput.py requires.
        assert row["arrival_s"] == offset
        assert set(row) == {
            "arrival_s",
            "scheduled_ts",
            "actual_send_ts",
            "scheduler_lag_ms",
            "first_token_ts",
            "ttft_from_send_ms",
            "ttft_from_scheduled_ms",
            "completion_ts",
            "latency_from_scheduled_ms",
            "latency_from_send_ms",
            "in_flight_at_send",
            "delayed_by_cap",
            "dropped_by_cap",
            "dispatch_error",
        }
        # Single-argument send: no first-token hook was forwarded, so TTFT
        # columns are honestly absent (None) -- never fabricated.
        assert row["first_token_ts"] is None
        assert row["ttft_from_send_ms"] is None
        assert row["ttft_from_scheduled_ms"] is None


def test_fake_clock_gives_exact_zero_lag_accounting():
    class FakeClock:
        def __init__(self) -> None:
            self.t = 0.0

        def __call__(self) -> float:
            return self.t

        async def sleep(self, delay: float) -> None:
            # Yield first so already-created tasks run at the CURRENT virtual
            # time, then advance.
            await asyncio.sleep(0)
            self.t += max(0.0, delay)

    clock = FakeClock()
    sched = generate_arrival_schedule(
        10.0, seed=0, n_requests=4, distribution="deterministic"
    )
    dispatcher = OpenLoopDispatcher(clock=clock, sleep=clock.sleep)
    report = _run(dispatcher.run(sched, _instant_send))

    assert report.n_completed == 4
    for record, offset in zip(report.records, sched.offsets_s):
        assert record.scheduled_ts == pytest.approx(offset)
        assert record.actual_send_ts == pytest.approx(record.scheduled_ts)
        assert record.scheduler_lag_ms == pytest.approx(0.0)
        assert record.latency_from_scheduled_ms == pytest.approx(0.0)


def test_first_token_hook_stamps_ttft_columns_on_dispatcher_clock():
    """A send accepting the second positional arg gets the ``on_first_token``
    callback; invoking it stamps ``first_token_ts`` on the DISPATCHER'S
    (injectable) clock, from which both ``ttft_from_send_ms`` and the D6 §6.3
    coordinated-omission-corrected ``ttft_from_scheduled_ms`` derive."""

    class FakeClock:
        def __init__(self) -> None:
            self.t = 0.0

        def __call__(self) -> float:
            return self.t

        async def sleep(self, delay: float) -> None:
            await asyncio.sleep(0)
            self.t += max(0.0, delay)

    clock = FakeClock()

    async def streaming_send(index: int, on_first_token) -> str:
        clock.t += 0.040  # 40 ms until the first content delta
        on_first_token()
        clock.t += 0.060  # rest of the stream
        return f"resp-{index}"

    sched = generate_arrival_schedule(
        10.0, seed=0, n_requests=3, distribution="deterministic"
    )
    dispatcher = OpenLoopDispatcher(clock=clock, sleep=clock.sleep)
    report = _run(dispatcher.run(sched, streaming_send))

    assert report.n_completed == 3
    for record in report.records:
        assert record.first_token_ts == pytest.approx(record.actual_send_ts + 0.040)
        assert record.ttft_from_send_ms == pytest.approx(40.0)
        # §6.3 identity: TTFT from the INTENDED arrival always includes the
        # scheduler lag -- client-side delay can never hide congestion.
        assert record.ttft_from_scheduled_ms == pytest.approx(
            record.ttft_from_send_ms + record.scheduler_lag_ms
        )
        assert record.latency_from_send_ms == pytest.approx(100.0)
        row = record.to_row()
        assert row["ttft_from_send_ms"] == pytest.approx(40.0)
        assert row["first_token_ts"] == record.first_token_ts


def test_streaming_send_that_never_streams_leaves_ttft_none():
    """A hook-accepting send that never sees a first token (e.g. an error
    stream) leaves TTFT columns None -- absence stays honest."""

    async def no_token_send(index: int, on_first_token) -> str:
        return f"resp-{index}"  # completes without ever invoking the hook

    sched = generate_arrival_schedule(
        200.0, seed=0, n_requests=2, distribution="deterministic"
    )
    report = _run(OpenLoopDispatcher().run(sched, no_token_send))
    for record in report.records:
        assert record.completed
        assert record.first_token_ts is None
        assert record.ttft_from_send_ms is None
        assert record.ttft_from_scheduled_ms is None


def test_empty_schedule_yields_empty_report():
    sched = generate_arrival_schedule(0.001, seed=5, duration_s=0.1)
    assert len(sched) == 0

    report = _run(OpenLoopDispatcher().run(sched, _instant_send))

    assert report.n_scheduled == 0
    assert report.n_sent == 0
    assert report.dropped_or_delayed == 0
    manifest = report.to_manifest()
    assert manifest["n_scheduled"] == 0
    assert manifest["schedule"]["seed"] == 5


# --------------------------------------------------------------------------- #
# Safety cap — loud accounting, never silent gating
# --------------------------------------------------------------------------- #


def test_cap_delay_policy_flags_and_counts_loudly():
    # 4 arrivals 1 ms apart, 50 ms service, cap 1: requests 1-3 find the cap
    # saturated at their arrival instant -> flagged delayed, counted, still
    # sent, and their scheduler lag exposes the wait.
    sched = generate_arrival_schedule(
        1000.0, seed=0, n_requests=4, distribution="deterministic"
    )
    dispatcher = OpenLoopDispatcher(max_in_flight=1, cap_policy="delay")
    report = _run(dispatcher.run(sched, _slow_send(0.05)))

    assert report.n_sent == 4
    assert report.n_completed == 4
    assert report.dropped_or_delayed == 3
    assert report.max_in_flight_observed == 1
    delayed = [r for r in report.records if r.delayed_by_cap]
    assert [r.index for r in delayed] == [1, 2, 3]
    for record in delayed:
        # The wait for a slot is visible in the lag, and SLO latency (from
        # scheduled_ts) still includes it -- no coordinated omission.
        assert record.scheduler_lag_ms > 20.0
        assert record.latency_from_scheduled_ms > record.latency_from_send_ms


def test_cap_drop_policy_records_dropped_requests():
    sched = generate_arrival_schedule(
        1000.0, seed=0, n_requests=4, distribution="deterministic"
    )
    dispatcher = OpenLoopDispatcher(max_in_flight=1, cap_policy="drop")
    report = _run(dispatcher.run(sched, _slow_send(0.05)))

    assert report.n_scheduled == 4
    assert report.n_sent == 1
    assert report.n_dropped == 3
    assert report.dropped_or_delayed == 3
    dropped = [r for r in report.records if r.dropped_by_cap]
    assert [r.index for r in dropped] == [1, 2, 3]
    for record in dropped:
        assert record.actual_send_ts is None
        assert record.result is None
        assert not record.completed
        assert record.to_row()["dropped_by_cap"] is True


def test_uncapped_dispatcher_never_flags():
    sched = generate_arrival_schedule(
        1000.0, seed=0, n_requests=6, distribution="deterministic"
    )
    report = _run(OpenLoopDispatcher().run(sched, _slow_send(0.02)))

    assert report.dropped_or_delayed == 0
    assert not any(r.delayed_by_cap or r.dropped_by_cap for r in report.records)


# --------------------------------------------------------------------------- #
# Failure handling and dispatcher config validation
# --------------------------------------------------------------------------- #


def test_send_exception_is_captured_per_record_not_fatal():
    async def flaky_send(index: int) -> str:
        if index == 1:
            raise ValueError("engine exploded")
        return f"resp-{index}"

    sched = generate_arrival_schedule(
        200.0, seed=0, n_requests=3, distribution="deterministic"
    )
    report = _run(OpenLoopDispatcher().run(sched, flaky_send))

    assert report.n_errors == 1
    assert report.n_completed == 2
    errored = report.records[1]
    assert errored.error is not None and errored.error.startswith("ValueError")
    assert errored.latency_from_scheduled_ms is not None  # failure is timed data
    assert report.records[0].completed and report.records[2].completed


@pytest.mark.parametrize("bad_cap", [0, -1, 1.5, True])
def test_invalid_max_in_flight_raises(bad_cap):
    with pytest.raises(LoadGeneratorError):
        OpenLoopDispatcher(max_in_flight=bad_cap)


def test_invalid_cap_policy_raises():
    with pytest.raises(LoadGeneratorError):
        OpenLoopDispatcher(max_in_flight=4, cap_policy="block")


def test_run_rejects_non_schedule_and_non_callable():
    sched = generate_arrival_schedule(10.0, seed=0, n_requests=1)
    dispatcher = OpenLoopDispatcher()
    with pytest.raises(LoadGeneratorError):
        _run(dispatcher.run([0.1, 0.2], _instant_send))
    with pytest.raises(LoadGeneratorError):
        _run(dispatcher.run(sched, "not-a-callable"))


# --------------------------------------------------------------------------- #
# Warmup / measurement-window trimming
# --------------------------------------------------------------------------- #


def _records_at(offsets: List[float]) -> List[RequestRecord]:
    return [
        RequestRecord(index=i, scheduled_offset_s=t, scheduled_ts=t)
        for i, t in enumerate(offsets)
    ]


def test_trim_removes_warmup_by_intended_arrival():
    records = _records_at([0.5, 1.5, 2.5, 3.5])
    kept = trim_to_measurement_window(records, warmup_s=1.0)
    assert [r.scheduled_offset_s for r in kept] == [1.5, 2.5, 3.5]


def test_trim_bounds_measurement_window():
    records = _records_at([0.5, 1.5, 2.5, 3.5])
    kept = trim_to_measurement_window(records, warmup_s=1.0, measurement_s=2.0)
    # Window is [1.0, 3.0): keeps 1.5 and 2.5 only.
    assert [r.scheduled_offset_s for r in kept] == [1.5, 2.5]


def test_trim_zero_warmup_keeps_all():
    records = _records_at([0.5, 1.5])
    kept = trim_to_measurement_window(records, warmup_s=0.0)
    assert len(kept) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"warmup_s": -0.1},
        {"warmup_s": math.nan},
        {"warmup_s": 1.0, "measurement_s": 0.0},
        {"warmup_s": 1.0, "measurement_s": -2.0},
    ],
)
def test_trim_invalid_parameters_raise(kwargs):
    with pytest.raises(LoadGeneratorError):
        trim_to_measurement_window(_records_at([1.0]), **kwargs)
