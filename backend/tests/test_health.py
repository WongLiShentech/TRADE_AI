"""Tests for the /health readiness logic (M9 container hardening).

``/health`` is what the container HEALTHCHECK, the compose dependency graph and any
uptime monitor will believe. Its previous implementation returned a static
``{"status": "ok"}``, which would have reported healthy for a process with a dead
database, a frozen price stream and no ML artifact — i.e. for every silent failure
mode this deployment actually has.

These tests exercise ``app.main.build_health`` directly rather than through an HTTP
client, deliberately: the function is the whole decision, and calling it in-process
means the suite never has to touch (or disturb) a running instance on port 8000.

Run from backend/:  python -m pytest tests/test_health.py -v
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app import main
from app.main import (
    HEALTH_DEGRADED,
    HEALTH_OK,
    HEALTH_UNHEALTHY,
    build_health,
)


@pytest.fixture(autouse=True)
def _restore_stream_expectation():
    """``_stream_expected`` is module state set by the lifespan; isolate each test."""
    original = main._stream_expected
    yield
    main._stream_expected = original


# ── aggregate status resolution ──────────────────────────────────────────────
def test_all_components_ok_gives_200(settings):
    """A healthy process returns ok/200 — the baseline the probe compares against."""
    main._stream_expected = False  # no active instruments in a test process
    payload, code = build_health(settings)

    assert payload.database.status == HEALTH_OK, payload.database.detail
    assert payload.ml_model.status == HEALTH_OK, payload.ml_model.detail
    assert payload.status == HEALTH_OK
    assert code == 200


def test_database_failure_is_unhealthy_503(settings, monkeypatch):
    """A dead DB must fail the probe. This is the case a TCP check cannot see."""
    class _DeadSession:
        def execute(self, *_args, **_kwargs):
            raise RuntimeError("connection refused")

        def close(self):
            pass

    monkeypatch.setattr(main, "SessionLocal", _DeadSession)
    main._stream_expected = False

    payload, code = build_health(settings)

    assert payload.database.status == HEALTH_UNHEALTHY
    assert "connection refused" in (payload.database.detail or "")
    assert payload.status == HEALTH_UNHEALTHY
    assert code == 503


def test_unloadable_model_is_unhealthy_when_shadow_enabled(settings, monkeypatch):
    """Shadow mode with no usable artifact = recording zero rows = not healthy.

    This is the exact deployment defect the startup assertion also guards: an image
    built without backend/models/ would otherwise look perfectly fine forever.
    """
    monkeypatch.setattr(settings, "SHADOW_MODE_ENABLED", True)
    monkeypatch.setattr(
        main.inf, "load_model", lambda *_a, **_kw: (_ for _ in ()).throw(
            FileNotFoundError("ML artifact not found")
        )
    )
    main._stream_expected = False

    payload, code = build_health(settings)

    assert payload.ml_model.status == HEALTH_UNHEALTHY
    assert "FileNotFoundError" in (payload.ml_model.detail or "")
    assert code == 503


def test_model_check_is_skipped_when_shadow_disabled(settings, monkeypatch):
    """With SHADOW_MODE_ENABLED=false there is nothing to load and nothing to fail."""
    monkeypatch.setattr(settings, "SHADOW_MODE_ENABLED", False)
    monkeypatch.setattr(
        main.inf, "load_model", lambda *_a, **_kw: pytest.fail("must not load a model")
    )
    main._stream_expected = False

    payload, code = build_health(settings)

    assert payload.ml_model.status == HEALTH_OK
    assert code == 200


# ── price stream component ───────────────────────────────────────────────────
def test_no_active_instruments_is_not_a_fault(settings):
    """A deployment before /instruments/sync legitimately has no stream."""
    main._stream_expected = False
    payload, _ = build_health(settings)

    assert payload.price_stream.status == HEALTH_OK
    assert "intentionally not started" in (payload.price_stream.detail or "")


def test_dead_stream_task_is_unhealthy(settings, monkeypatch):
    """Retries are unbounded now, so a FINISHED task means the coroutine itself died."""
    monkeypatch.setattr(main.price_stream, "is_running", lambda: False)
    main._stream_expected = True

    payload, code = build_health(settings)

    assert payload.price_stream.status == HEALTH_UNHEALTHY
    assert code == 503


def test_stale_ticks_are_degraded_not_unhealthy(settings, monkeypatch):
    """A weekend market close makes the cache legitimately stale.

    Degraded returns 200 on purpose: an unhealthy verdict here would have the
    container restarted every Saturday for behaving exactly as designed.
    """
    budget = main._price_staleness_budget_seconds(settings)
    stale = datetime.now(timezone.utc) - timedelta(seconds=budget * 2)
    monkeypatch.setattr(main.price_stream, "is_running", lambda: True)
    monkeypatch.setattr(main.price_stream, "last_tick_at", lambda: stale)
    main._stream_expected = True

    payload, code = build_health(settings)

    assert payload.price_stream.status == HEALTH_DEGRADED
    assert payload.status == HEALTH_DEGRADED
    assert code == 200


def test_fresh_ticks_are_ok(settings, monkeypatch):
    monkeypatch.setattr(main.price_stream, "is_running", lambda: True)
    monkeypatch.setattr(
        main.price_stream, "last_tick_at", lambda: datetime.now(timezone.utc)
    )
    main._stream_expected = True

    payload, code = build_health(settings)

    assert payload.price_stream.status == HEALTH_OK
    assert code == 200


def test_started_but_no_tick_yet_is_degraded(settings, monkeypatch):
    monkeypatch.setattr(main.price_stream, "is_running", lambda: True)
    monkeypatch.setattr(main.price_stream, "last_tick_at", lambda: None)
    main._stream_expected = True

    payload, code = build_health(settings)

    assert payload.price_stream.status == HEALTH_DEGRADED
    assert code == 200


# ── staleness budget derivation ──────────────────────────────────────────────
def test_staleness_budget_is_derived_from_stream_config(settings):
    """No new env var: the budget is the stream's own escalation budget.

    heartbeat timeout × reconnect retries — the point at which price_stream itself
    escalates to ERROR and alerts. Keeping them equal means the two signals never
    contradict each other.
    """
    expected = float(settings.STREAM_HEARTBEAT_TIMEOUT_SECONDS) * float(
        settings.STREAM_MAX_RECONNECT_RETRIES
    )
    assert main._price_staleness_budget_seconds(settings) == expected
