"""Tests for the live-price-stream resilience fixes (M9 hardening).

Two silent, unattended failure modes are covered here. Neither is observable from
outside the process, which is exactly why both needed tests rather than a manual
smoke check:

H1 — the reconnect loop used to STOP after ``STREAM_MAX_RECONNECT_RETRIES``
     attempts. With the shipped values that is 50 seconds of trouble killing live
     pricing permanently, with one WARNING in the log and a process that keeps
     answering HTTP 200 forever.

H2 — ``stream_prices`` runs on an ``httpx.AsyncClient(timeout=None)``. A half-open
     TCP connection makes ``aiter_lines()`` block forever with no exception, so the
     reconnect loop never even gets the chance to fire.

Async bodies are driven with ``asyncio.run`` rather than ``pytest.mark.asyncio``:
``pytest-asyncio`` is not a project dependency, and adding one to requirements.txt
— which also builds the production image — for test plumbing alone is not a trade
worth making.

Run from backend/:  python -m pytest tests/test_stream_resilience.py -v
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import pytest

from app.brokers.base import StreamHeartbeatTimeout
from app.brokers.oanda import OandaClient
from app.services import price_stream


async def _collect(async_iterable) -> list:
    return [item async for item in async_iterable]


# ── H1: backoff schedule ─────────────────────────────────────────────────────
def test_backoff_starts_at_the_configured_delay(settings):
    """First retry waits exactly STREAM_RECONNECT_DELAY_SECONDS — no slower."""
    assert price_stream._backoff_delay(1, settings) == float(
        settings.STREAM_RECONNECT_DELAY_SECONDS
    )


def test_backoff_doubles_then_plateaus(settings):
    """Exponential, then flat — never unbounded growth, never a hardcoded second count."""
    base = float(settings.STREAM_RECONNECT_DELAY_SECONDS)
    schedule = [price_stream._backoff_delay(n, settings) for n in range(1, 12)]

    assert schedule[:4] == [base, base * 2, base * 4, base * 8]

    ceiling = base * (price_stream._BACKOFF_BASE ** price_stream._BACKOFF_MAX_STEPS)
    assert max(schedule) == ceiling
    # Plateau: once the cap is reached the delay never changes again, so a weekend
    # close costs a bounded number of connection attempts.
    assert schedule[-1] == schedule[-2] == ceiling


def test_backoff_is_expressed_relative_to_config_not_absolute_seconds(settings):
    """Doubling the configured base doubles the whole schedule (zero-hardcoding)."""

    class _Doubled:
        STREAM_RECONNECT_DELAY_SECONDS = settings.STREAM_RECONNECT_DELAY_SECONDS * 2

    for n in (1, 3, 20):
        assert price_stream._backoff_delay(n, _Doubled) == 2 * price_stream._backoff_delay(
            n, settings
        )


def test_stream_loop_retries_past_the_old_give_up_point(settings, monkeypatch):
    """The loop must still be retrying well beyond STREAM_MAX_RECONNECT_RETRIES.

    Before the fix the coroutine RETURNED at that count. The assertion is therefore
    on attempt COUNT, not on any log line: the observable defect was the loop
    exiting and the process carrying on as if nothing had happened.
    """
    attempts = 0
    target = int(settings.STREAM_MAX_RECONNECT_RETRIES) * 3

    class _AlwaysFailingClient:
        def stream_prices(self, _instruments):
            raise ConnectionError("simulated disconnect")

    class _Router:
        def for_asset_class(self, _asset_class):
            return _AlwaysFailingClient()

    async def _no_sleep(_delay):
        nonlocal attempts
        attempts += 1
        if attempts >= target:
            raise asyncio.CancelledError

    async def _noop_alert(*_args, **_kwargs):
        return None

    monkeypatch.setattr(price_stream.asyncio, "sleep", _no_sleep)
    monkeypatch.setattr(price_stream, "_dispatch_alert", _noop_alert)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(price_stream._stream_loop(["EUR_USD"], _Router(), settings))

    assert attempts >= target, "loop gave up instead of retrying indefinitely"


def test_escalation_alert_fires_at_the_configured_threshold(settings, monkeypatch):
    """STREAM_MAX_RECONNECT_RETRIES now means 'escalate here', not 'give up here'."""
    sent: list[tuple[str, str]] = []

    async def _capture(_settings, subject, body):
        sent.append((subject, body))

    monkeypatch.setattr(price_stream, "_dispatch_alert", _capture)

    threshold = int(settings.STREAM_MAX_RECONNECT_RETRIES)
    now = datetime.now(timezone.utc)

    # One below the threshold: warning only, no alert.
    asyncio.run(
        price_stream._report_failure(
            settings, ConnectionError("x"), threshold - 1, threshold, now, 5.0
        )
    )
    assert sent == []

    # At the threshold: escalate.
    asyncio.run(
        price_stream._report_failure(
            settings, ConnectionError("x"), threshold, threshold, now, 5.0
        )
    )
    assert len(sent) == 1
    assert "DOWN" in sent[0][0]


# ── H2: heartbeat watchdog ───────────────────────────────────────────────────
class _FakeResponse:
    """Stands in for an httpx streaming response.

    A float in ``lines`` is a STALL of that many seconds — the half-open-connection
    behaviour being simulated.
    """

    def __init__(self, lines):
        self._lines = lines

    def raise_for_status(self):
        return None

    async def aiter_lines(self):
        for item in self._lines:
            if isinstance(item, float):
                await asyncio.sleep(item)
            else:
                yield item

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


class _FakeAsyncClient:
    def __init__(self, lines):
        self._lines = lines

    def stream(self, *_args, **_kwargs):
        return _FakeResponse(self._lines)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


def _price_line(instrument="EUR_USD", bid="1.1000", ask="1.1002"):
    return json.dumps(
        {
            "type": "PRICE",
            "instrument": instrument,
            "tradeable": True,
            "time": "2026-08-05T10:00:00.123456789Z",
            "bids": [{"price": bid}],
            "asks": [{"price": ask}],
        }
    )


_HEARTBEAT_LINE = json.dumps({"type": "HEARTBEAT", "time": "2026-08-05T10:00:05.000000000Z"})


def _client_with(settings, monkeypatch, lines, timeout_seconds):
    client = OandaClient(settings)
    client._heartbeat_timeout = timeout_seconds
    monkeypatch.setattr(
        "app.brokers.oanda.httpx.AsyncClient", lambda *a, **kw: _FakeAsyncClient(lines)
    )
    return client


def test_silent_connection_raises_instead_of_hanging_forever(settings, monkeypatch):
    """THE fix: a stalled stream must raise, not block.

    Without the watchdog this test would hang until the suite was killed — which is
    exactly what the production process did, indefinitely and silently.
    """
    stall = 10.0  # far longer than the watchdog timeout below
    client = _client_with(settings, monkeypatch, [_price_line(), stall], timeout_seconds=0.1)

    with pytest.raises(StreamHeartbeatTimeout) as exc:
        asyncio.run(_collect(client.stream_prices(["EUR_USD"])))

    assert "heartbeat" in str(exc.value).lower()
    assert "STREAM_HEARTBEAT_TIMEOUT_SECONDS" in str(exc.value)


def test_heartbeats_reset_the_watchdog(settings, monkeypatch):
    """A quiet market sends HEARTBEATs and no prices — that must NOT trip the watchdog.

    The pre-fix code ``continue``d past every non-PRICE message, so a watchdog placed
    after that filter would have killed a perfectly healthy stream during every
    quiet spell. The watchdog wraps the ITERATOR for this reason.
    """
    timeout = 0.3
    gap = timeout / 3
    lines = [_HEARTBEAT_LINE, gap, _HEARTBEAT_LINE, gap, _HEARTBEAT_LINE, gap, _price_line()]
    client = _client_with(settings, monkeypatch, lines, timeout_seconds=timeout)

    ticks = asyncio.run(_collect(client.stream_prices(["EUR_USD"])))

    assert len(ticks) == 1, "the price after three heartbeats should have been yielded"
    assert ticks[0].instrument == "EUR_USD"


def test_normal_ticks_still_flow(settings, monkeypatch):
    """Regression guard: the watchdog must not change ordinary parsing behaviour."""
    lines = [
        _price_line(bid="1.1000", ask="1.1002"),
        _HEARTBEAT_LINE,
        "",                       # blank keep-alive line
        "not json at all",        # malformed line — skipped, not fatal
        json.dumps({"type": "PRICE", "instrument": "EUR_USD", "tradeable": False}),
        _price_line(bid="1.1010", ask="1.1012"),
    ]
    client = _client_with(settings, monkeypatch, lines, timeout_seconds=5.0)

    ticks = asyncio.run(_collect(client.stream_prices(["EUR_USD"])))

    assert [t.bid for t in ticks] == [1.1000, 1.1010]
    assert [t.ask for t in ticks] == [1.1002, 1.1012]


def test_server_closing_the_stream_is_not_a_heartbeat_timeout(settings, monkeypatch):
    """A clean end-of-stream returns; the caller's reconnect loop handles it."""
    client = _client_with(settings, monkeypatch, [_price_line()], timeout_seconds=5.0)

    ticks = asyncio.run(_collect(client.stream_prices(["EUR_USD"])))

    assert len(ticks) == 1
