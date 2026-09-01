"""
Live price stream service.

Maintains a single background task that consumes ticks from the broker stream
via BrokerRouter and stores the latest TickData per instrument in an
in-memory cache. Application code reads the cache via get_latest_price() /
get_all_prices() — never touches the stream directly.

Design notes:
- In-memory cache (dict). Tick data is ephemeral by nature; persistence would
  add latency and disk churn without providing value. The cache is rebuilt on
  every app boot once the first ticks arrive.
- Module-level state (not a class) because there is exactly one stream
  process per app instance.

Reconnect policy — UNBOUNDED, with escalation (changed 2026-08-05)
-----------------------------------------------------------------
The loop previously stopped retrying after ``STREAM_MAX_RECONNECT_RETRIES``
attempts and returned, leaving the process alive with a permanently dead stream
and a single WARNING in the log. With the shipped values (10 retries × 5s) that
is **50 seconds of connectivity trouble kills live pricing until someone notices
and restarts the app** — an unattended, silent, unrecoverable failure. Weekend
market closes, VM network blips and broker maintenance all clear that bar
routinely.

So the loop now retries FOREVER. What ``STREAM_MAX_RECONNECT_RETRIES`` means has
changed accordingly, and this is a deliberate semantic change:

    OLD: consecutive failures before the stream GIVES UP (terminal).
    NEW: consecutive failures before the stream ESCALATES — WARNING becomes
         ERROR and an alert is dispatched through services.alerts. Retrying
         never stops.

Backoff is exponential and expressed as MULTIPLES of
``STREAM_RECONNECT_DELAY_SECONDS`` so the whole schedule scales with that one
configured value and no absolute number of seconds is hardcoded:

    delay(n) = STREAM_RECONNECT_DELAY_SECONDS × 2 ** min(n - 1, _BACKOFF_MAX_STEPS)

With the shipped 5s base that is 5, 10, 20, 40, 80, 160, 320s and then a flat
320s ceiling — fast enough to reconnect within seconds of a blip, slow enough not
to hammer the broker (or the log) through a two-day weekend close.
"""
import asyncio
import logging
from datetime import datetime, timedelta, timezone

from app.brokers.base import TickData
from app.brokers.router import BrokerRouter
from app.config import Settings
from app.services.alerts.factory import get_alert_delivery

logger = logging.getLogger(__name__)

# Exponential backoff shape. Relative (a doubling count), never absolute seconds —
# the base interval is config (STREAM_RECONNECT_DELAY_SECONDS) and this only says
# how many times it may double before the delay plateaus. 6 doublings = 64× base.
_BACKOFF_BASE = 2
_BACKOFF_MAX_STEPS = 6

_price_cache: dict[str, TickData] = {}
_stream_task: asyncio.Task | None = None
# Wall-clock instant of the most recent tick accepted into the cache. Read by the
# /health readiness probe, which cannot infer liveness from the cache alone: a
# stale cache and a warm cache look identical without a timestamp.
_last_tick_at: datetime | None = None


def get_latest_price(instrument: str) -> TickData | None:
    return _price_cache.get(instrument)


def get_all_prices() -> dict[str, TickData]:
    return dict(_price_cache)


def last_tick_at() -> datetime | None:
    """UTC instant the newest tick was received, or ``None`` if none ever was.

    Exposed for the /health readiness probe. This is the RECEIPT time, not the
    broker's tick timestamp — the question being asked is "is our stream alive",
    not "how old is the market data".
    """
    return _last_tick_at


def is_running() -> bool:
    """True when the background stream task exists and has not finished."""
    return _stream_task is not None and not _stream_task.done()


async def start_stream(
    instruments: list[str],
    broker_router: BrokerRouter,
    settings: Settings,
) -> None:
    global _stream_task
    if _stream_task is not None and not _stream_task.done():
        return
    logger.info("price stream starting for %d instruments", len(instruments))
    _stream_task = asyncio.create_task(
        _stream_loop(instruments, broker_router, settings)
    )


async def stop_stream() -> None:
    global _stream_task
    if _stream_task is None:
        return
    _stream_task.cancel()
    try:
        await _stream_task
    except asyncio.CancelledError:
        pass
    _stream_task = None
    logger.info("price stream stopped")


async def _stream_loop(
    instruments: list[str],
    broker_router: BrokerRouter,
    settings: Settings,
) -> None:
    """Consume ticks forever, reconnecting with capped exponential backoff.

    Only ``asyncio.CancelledError`` ends this coroutine — that is the lifespan
    shutdown asking it to stop. Every other exception is a reconnect.
    """
    global _last_tick_at
    consecutive_failures = 0
    down_since: datetime | None = None
    escalation_threshold = max(1, int(settings.STREAM_MAX_RECONNECT_RETRIES))

    while True:
        try:
            client = broker_router.for_asset_class("forex")
            async for tick in client.stream_prices(instruments):
                _price_cache[tick.instrument] = tick
                _last_tick_at = datetime.now(timezone.utc)
                if consecutive_failures:
                    # Recovery is as newsworthy as the outage — without this line an
                    # operator reading the log cannot tell a resolved blip from an
                    # ongoing outage.
                    logger.info(
                        "price stream RECOVERED after %d consecutive failure(s)%s",
                        consecutive_failures,
                        _downtime_suffix(down_since),
                    )
                    if consecutive_failures >= escalation_threshold:
                        await _dispatch_alert(
                            settings,
                            "price stream recovered",
                            f"Live price stream reconnected after {consecutive_failures} "
                            f"consecutive failures{_downtime_suffix(down_since)}.",
                        )
                consecutive_failures = 0
                down_since = None
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — any failure is a reconnect, never an exit
            consecutive_failures += 1
            if down_since is None:
                down_since = datetime.now(timezone.utc)
            delay = _backoff_delay(consecutive_failures, settings)
            await _report_failure(
                settings, exc, consecutive_failures, escalation_threshold, down_since, delay
            )
            await asyncio.sleep(delay)


def _backoff_delay(consecutive_failures: int, settings: Settings) -> float:
    """Capped exponential backoff, in multiples of the configured base delay."""
    base = float(settings.STREAM_RECONNECT_DELAY_SECONDS)
    steps = min(max(consecutive_failures - 1, 0), _BACKOFF_MAX_STEPS)
    return base * (_BACKOFF_BASE ** steps)


async def _report_failure(
    settings: Settings,
    exc: BaseException,
    consecutive_failures: int,
    escalation_threshold: int,
    down_since: datetime,
    delay: float,
) -> None:
    """Log the disconnect, escalating to ERROR + alert past the configured threshold.

    Re-alerts every ``escalation_threshold`` further failures rather than on every
    attempt: one alert per outage is signal, one per retry is noise that trains the
    operator to ignore the channel.
    """
    if consecutive_failures < escalation_threshold:
        logger.warning(
            "price stream disconnected (failure %d/%d before escalation): %s — "
            "retrying in %.0fs",
            consecutive_failures, escalation_threshold, exc, delay,
        )
        return

    logger.error(
        "price stream DOWN%s — %d consecutive failures (threshold %d): %s — "
        "retrying in %.0fs (retries are unbounded; the stream never gives up)",
        _downtime_suffix(down_since), consecutive_failures, escalation_threshold, exc, delay,
    )
    if consecutive_failures % escalation_threshold == 0:
        await _dispatch_alert(
            settings,
            "price stream DOWN",
            f"Live price stream has failed {consecutive_failures} consecutive times"
            f"{_downtime_suffix(down_since)}. Last error: {type(exc).__name__}: {exc}. "
            f"Next retry in {delay:.0f}s. Retries continue indefinitely.",
        )


async def _dispatch_alert(settings: Settings, subject: str, body: str) -> None:
    """Send through the configured alert channel; never let alerting break the stream.

    Off-loaded to a worker thread: ``AlertDelivery.send`` is a synchronous interface
    and the email/telegram implementations do blocking network I/O. Calling it
    inline would stall the event loop — i.e. stall pricing — for the duration of an
    SMTP handshake.
    """
    try:
        await asyncio.to_thread(get_alert_delivery(settings).send, subject, body)
    except Exception as exc:  # noqa: BLE001 — a broken alert channel must not kill pricing
        logger.warning("price stream alert dispatch failed (%s: %s)", type(exc).__name__, exc)


def _downtime_suffix(down_since: datetime | None) -> str:
    if down_since is None:
        return ""
    elapsed: timedelta = datetime.now(timezone.utc) - down_since
    return f" (down {elapsed.total_seconds() / 60:.1f} min)"
