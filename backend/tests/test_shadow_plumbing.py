"""Tests for the live data plumbing M8-Shadow Phase 3 depends on.

Phase 3's resolver is only as honest as the data underneath it. Two pieces of
plumbing make it work, and both are registry-driven rather than hand-wired:

* **Live M1 Bid/Ask ingestion** (GAP-13) — the intrabar series the triple-barrier
  simulator walks. Without it every live outcome would silently fall back to the
  degraded signal-timeframe path and be flagged ``ambiguous_resolution=True``.
* **Scheduler registration** — the M1 top-up and the resolver each get their own
  cron job, derived from the Timeframe registry / Settings rather than hardcoded.

No test here calls the broker: the fetch is monkeypatched so the WINDOW ARITHMETIC
and the routing are what get verified, not OANDA's uptime.

Run from backend/:  python -m pytest tests/test_shadow_plumbing.py -v
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.domain.timeframes import TIMEFRAMES, get_timeframe
from app.services import candle_service, pipeline, scheduler

_INTRABAR = "M1"
_TRADING = "H4"
_TREND = "D"


# ── 1. the M1 registry contract ─────────────────────────────────────────────
def test_m1_is_scheduled_and_refresh_only():
    tf = get_timeframe(_INTRABAR)

    assert tf.cron_hours is not None, "GAP-13: live M1 must actually be scheduled"
    assert tf.fires_signals is False, "M1 must never author a signal"


def test_m1_maintains_both_execution_sides_and_no_indicators():
    """The simulator joins Bid and Ask at a shared timestamp and skips any timestamp
    missing either side, so fetching one side would make the series unusable — and
    fetching Mid too would double the API cost for data nothing reads."""
    tf = get_timeframe(_INTRABAR)

    assert set(tf.price_types) == {"B", "A"}
    assert tf.computes_indicators is False


def test_m1_uses_a_bounded_trailing_window():
    tf = get_timeframe(_INTRABAR)

    assert tf.trailing_window_setting == "M1_LIVE_LOOKBACK_HOURS"


def test_trading_and_trend_timeframes_are_unchanged_by_the_m1_addition():
    """Adding a timeframe must be a registry addition, never a behaviour change to
    the existing ones."""
    h4, d1 = get_timeframe(_TRADING), get_timeframe(_TREND)

    assert h4.fires_signals is True and h4.price_types == ("M",) and h4.computes_indicators
    assert h4.trailing_window_setting is None
    assert d1.fires_signals is False and d1.price_types == ("M",) and d1.computes_indicators
    assert d1.trailing_window_setting is None


def test_every_trailing_window_setting_names_a_real_config_value(settings):
    """A registry entry naming a Settings attribute that does not exist would only
    fail at 03:20 on a Tuesday. Fail here instead."""
    for tf in TIMEFRAMES.values():
        if tf.trailing_window_setting is None:
            continue
        value = getattr(settings, tf.trailing_window_setting, None)
        assert value is not None, f"{tf.code}: no setting '{tf.trailing_window_setting}'"
        assert int(value) > 0


def test_trailing_lookback_exceeds_the_job_cadence(settings):
    """Consecutive runs must OVERLAP, so a missed run self-heals on the next pass
    instead of leaving a permanent hole in the resolver's evidence."""
    tf = get_timeframe(_INTRABAR)
    lookback_hours = int(getattr(settings, tf.trailing_window_setting))

    assert tf.cron_hours == "*", "the cadence assumed below is hourly"
    assert lookback_hours > 1


# ── 2. trailing-window pipeline: window arithmetic + routing ────────────────
def test_trailing_refresh_fetches_the_configured_window_for_every_side(
    settings, monkeypatch
):
    calls = []

    def _fake(*, instrument_symbol, granularity, start, end, db, settings, broker_router,
              price_type):
        calls.append((instrument_symbol, granularity, start, end, price_type))
        return 1

    monkeypatch.setattr(candle_service, "fetch_and_store_window", _fake)
    now = datetime(2026, 7, 31, 12, 0, tzinfo=timezone.utc)

    summary = pipeline.run_trailing_window_refresh(_INTRABAR, settings, now=now)

    expected_start = now - timedelta(hours=settings.M1_LIVE_LOOKBACK_HOURS)
    assert summary["lookback_hours"] == settings.M1_LIVE_LOOKBACK_HOURS
    assert summary["price_types"] == ["B", "A"]
    assert summary["errors"] == 0
    assert summary["instruments"] > 0
    assert summary["fetched"] == summary["instruments"] * 2
    assert calls, "nothing was fetched"
    for _symbol, granularity, start, end, _price_type in calls:
        assert granularity == _INTRABAR
        assert start == expected_start and end == now
    # Both sides requested for every instrument, exactly once each.
    per_symbol = {}
    for symbol, _g, _s, _e, price_type in calls:
        per_symbol.setdefault(symbol, []).append(price_type)
    assert all(sorted(v) == ["A", "B"] for v in per_symbol.values())


def test_trailing_refresh_is_instrument_agnostic(settings, monkeypatch):
    """The job discovers its universe from the DB — no symbol is named in code."""
    from app.database import SessionLocal
    from app.models.instrument import Instrument

    seen = []
    monkeypatch.setattr(
        candle_service, "fetch_and_store_window",
        lambda **kw: (seen.append(kw["instrument_symbol"]), 0)[1],
    )
    pipeline.run_trailing_window_refresh(_INTRABAR, settings, now=datetime.utcnow())

    db = SessionLocal()
    try:
        active = {i.symbol for i in db.query(Instrument).filter_by(is_active=True).all()}
    finally:
        db.close()
    assert set(seen) == active


def test_one_failing_pair_does_not_stop_the_others(settings, monkeypatch):
    def _flaky(**kw):
        if kw["price_type"] == "A":
            raise RuntimeError("broker hiccup")
        return 1

    monkeypatch.setattr(candle_service, "fetch_and_store_window", _flaky)

    summary = pipeline.run_trailing_window_refresh(
        _INTRABAR, settings, now=datetime.utcnow()
    )

    assert summary["errors"] == summary["instruments"]
    assert summary["fetched"] == summary["instruments"], "the Bid side still landed"


def test_trailing_refresh_refuses_a_timeframe_without_a_window_setting(settings):
    """Fail loud rather than silently inventing a window for H4/D1."""
    with pytest.raises(ValueError, match="trailing_window_setting"):
        pipeline.run_trailing_window_refresh(_TRADING, settings)


def test_refresh_pipeline_rejects_an_unregistered_timeframe(settings):
    with pytest.raises(ValueError, match="Unregistered timeframe"):
        pipeline.run_candle_refresh_pipeline("W", settings)


# ── 3. scheduler registration ───────────────────────────────────────────────
@pytest.fixture()
def registered_jobs(monkeypatch):
    """Build the real job table without ever starting the scheduler's threads (a
    started scheduler could fire a live ingestion mid-test)."""
    from apscheduler.schedulers.background import BackgroundScheduler

    monkeypatch.setattr(BackgroundScheduler, "start", lambda self, *a, **k: None)
    scheduler._scheduler = None
    try:
        instance = scheduler.start_scheduler()
        yield {job.id: job for job in instance.get_jobs()}
    finally:
        scheduler._scheduler = None


def test_scheduler_registers_a_job_per_scheduled_timeframe(registered_jobs):
    assert "h4_candle_close_pipeline" in registered_jobs
    assert "d_candle_refresh" in registered_jobs
    assert "m1_trailing_window_refresh" in registered_jobs


def test_scheduler_registers_the_resolver_as_its_own_job(registered_jobs):
    """Deliberately NOT bolted onto a candle-close body: resolvability is driven by
    the hold horizon and M1 coverage, not by a bar closing, and a slow resolve pass
    must never delay the signal path."""
    assert "shadow_outcome_resolver" in registered_jobs


# Jobs that read or write candle/trade data. These share one in-process threadpool
# and are the ones whose contention would actually delay a signal or a resolution.
# (The pre-existing weekly_chain / daily_circuit_breaker pair both sit at 00:00 and
# genuinely overlap once a week; both are cheap and predate this phase, so they are
# deliberately outside this check rather than silently swept into it.)
_DATA_JOBS = (
    "h4_candle_close_pipeline",
    "d_candle_refresh",
    "m1_trailing_window_refresh",
    "shadow_outcome_resolver",
    "intraday_fundamentals",
)


def test_data_jobs_do_not_collide_on_the_same_minute(registered_jobs):
    """Every cron job runs in one in-process threadpool, so two jobs firing on the
    same minute contend. Minute offsets are the cheap defence."""
    minutes: dict[int, list[str]] = {}
    for job_id in _DATA_JOBS:
        fields = {f.name: str(f) for f in registered_jobs[job_id].trigger.fields}
        minutes.setdefault(int(fields["minute"]), []).append(job_id)
    collisions = {k: v for k, v in minutes.items() if len(v) > 1}
    assert not collisions, f"data jobs share a cron minute: {collisions}"


def test_resolver_job_cadence_is_config_driven(registered_jobs, settings):
    fields = {f.name: str(f) for f in registered_jobs["shadow_outcome_resolver"].trigger.fields}

    assert fields["hour"] == f"*/{settings.SHADOW_RESOLVER_INTERVAL_HOURS}"


def test_m1_job_runs_before_the_resolver_within_the_hour(registered_jobs):
    """The resolver should see the freshest possible M1 — the top-up lands first."""
    m1 = {f.name: str(f) for f in registered_jobs["m1_trailing_window_refresh"].trigger.fields}
    res = {f.name: str(f) for f in registered_jobs["shadow_outcome_resolver"].trigger.fields}

    assert int(m1["minute"]) < int(res["minute"])
