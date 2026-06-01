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
- Reconnect loop with exponential-style retry capped by
  STREAM_MAX_RECONNECT_RETRIES, sleeping STREAM_RECONNECT_DELAY_SECONDS
  between attempts. asyncio.CancelledError is propagated so the lifespan
  shutdown can terminate the task cleanly.
- Module-level state (not a class) because there is exactly one stream
  process per app instance.
"""
import asyncio
import logging

import httpx

from app.brokers.base import TickData
from app.brokers.router import BrokerRouter
from app.config import Settings

logger = logging.getLogger(__name__)

_price_cache: dict[str, TickData] = {}
_stream_task: asyncio.Task | None = None


def get_latest_price(instrument: str) -> TickData | None:
    return _price_cache.get(instrument)


def get_all_prices() -> dict[str, TickData]:
    return dict(_price_cache)


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
    retries = 0
    while retries < settings.STREAM_MAX_RECONNECT_RETRIES:
        try:
            client = broker_router.for_asset_class("forex")
            async for tick in client.stream_prices(instruments):
                _price_cache[tick.instrument] = tick
                retries = 0  # reset on successful tick
        except asyncio.CancelledError:
            raise
        except (httpx.HTTPError, Exception) as e:
            retries += 1
            logger.warning(
                "price stream disconnected (attempt %d/%d): %s",
                retries,
                settings.STREAM_MAX_RECONNECT_RETRIES,
                e,
            )
            await asyncio.sleep(settings.STREAM_RECONNECT_DELAY_SECONDS)
    logger.error("price stream: max reconnect retries exceeded — stream stopped")
