import asyncio
import json
import logging
from typing import AsyncIterator
from datetime import datetime, timezone, timedelta

import httpx

from app.brokers.base import (
    BrokerClient,
    InstrumentData,
    CandleData,
    TickData,
    OrderRequest,
    OrderResult,
    AccountInfo,
    PositionData,
    StreamHeartbeatTimeout,
)
from app.config import Settings
from app.domain.timeframes import get_timeframe

logger = logging.getLogger(__name__)

_OANDA_TYPE_TO_ASSET_CLASS: dict[str, str] = {
    "CURRENCY": "forex",
    "CFD": "cfd",
    "METAL": "metal",
}


class OandaClient(BrokerClient):
    def __init__(self, settings: Settings) -> None:
        # Arms the base class's ORDER_PLACEMENT_ENABLED guard — see brokers/base.py.
        super().__init__(settings)
        self._api_key = settings.OANDA_API_KEY
        self._account_id = settings.OANDA_ACCOUNT_ID
        self._base_url = settings.OANDA_BASE_URL.rstrip("/")
        self._stream_url = settings.OANDA_STREAM_URL
        # Per-line watchdog for stream_prices. OANDA emits a HEARTBEAT message on
        # the pricing stream every ~5s when there is no price to send, so "no line
        # at all for this long" is an unambiguous dead-connection signal.
        self._heartbeat_timeout = float(settings.STREAM_HEARTBEAT_TIMEOUT_SECONDS)

    def get_instruments(self) -> list[InstrumentData]:
        url = f"{self._base_url}/v3/accounts/{self._account_id}/instruments"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        with httpx.Client() as client:
            response = client.get(url, headers=headers, follow_redirects=True)
            response.raise_for_status()
        results: list[InstrumentData] = []
        for inst in response.json().get("instruments", []):
            results.append(
                InstrumentData(
                    symbol=inst["name"],
                    display_name=inst["displayName"],
                    pip_size=10 ** inst["pipLocation"],
                    pip_location=inst["pipLocation"],
                    asset_class=_OANDA_TYPE_TO_ASSET_CLASS.get(
                        inst["type"], inst["type"].lower()
                    ),
                    broker_id=inst["name"],
                )
            )
        return results

    def get_candles(
        self,
        instrument: str,
        granularity: str,
        start: datetime,
        end: datetime,
        price_type: str = "M",
    ) -> list[CandleData]:
        _PRICE_KEY = {"M": "mid", "B": "bid", "A": "ask"}
        candles_per_day = get_timeframe(granularity).candles_per_day
        price_key = _PRICE_KEY[price_type]
        max_per_request = 5000
        chunk_days = max_per_request // candles_per_day

        headers = {"Authorization": f"Bearer {self._api_key}"}
        url = f"{self._base_url}/v3/instruments/{instrument}/candles"
        results: list[CandleData] = []

        chunk_start = start.replace(tzinfo=timezone.utc) if start.tzinfo is None else start
        utc_end = end.replace(tzinfo=timezone.utc) if end.tzinfo is None else end

        with httpx.Client(timeout=30) as client:
            while chunk_start < utc_end:
                chunk_end = min(chunk_start + timedelta(days=chunk_days), utc_end)
                params = {
                    "granularity": granularity,
                    "from": chunk_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "to": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "price": price_type,
                }
                response = client.get(url, headers=headers, params=params)
                response.raise_for_status()

                for candle in response.json().get("candles", []):
                    if not candle.get("complete", False):
                        continue
                    ohlc = candle[price_key]
                    results.append(
                        CandleData(
                            instrument=instrument,
                            granularity=granularity,
                            timestamp=datetime.strptime(
                                candle["time"], "%Y-%m-%dT%H:%M:%S.%f000Z"
                            ).replace(tzinfo=timezone.utc),
                            open=float(ohlc["o"]),
                            high=float(ohlc["h"]),
                            low=float(ohlc["l"]),
                            close=float(ohlc["c"]),
                            volume=int(candle["volume"]),
                        )
                    )
                chunk_start = chunk_end

        return results

    async def stream_prices(self, instruments: list[str]) -> AsyncIterator[TickData]:
        """Yield ticks from OANDA's pricing stream, guarded by a heartbeat watchdog.

        Raises:
            StreamHeartbeatTimeout: no line of ANY kind (PRICE or HEARTBEAT) arrived
                within ``STREAM_HEARTBEAT_TIMEOUT_SECONDS``. Surfacing this is the
                whole point of the watchdog — see below.

        Why the watchdog exists
        -----------------------
        The client is deliberately created with ``timeout=None``: a streaming
        response is long-lived by definition, and any positive httpx read timeout
        would kill a healthy stream during a quiet market. But ``timeout=None``
        removes the ONLY mechanism that would ever have raised on a half-open TCP
        connection. When the peer disappears without an RST (VM suspend/resume, NAT
        eviction, broker-side silent drop), ``aiter_lines()`` blocks forever, no
        exception is raised, the reconnect loop in ``services.price_stream`` never
        fires, and the process keeps reporting healthy with a frozen price cache.

        OANDA solves the detection half for us: the pricing stream sends a
        ``{"type":"HEARTBEAT"}`` message every few seconds whenever there is no
        price update. So *any* line arriving proves the socket is alive. The
        watchdog therefore wraps the ITERATOR, not the tick-yielding branch —
        heartbeats and unparseable lines reset it exactly like a price does. (The
        original code ``continue``d past non-PRICE messages, so a watchdog placed
        after that filter would have fired during any quiet market.)
        """
        url = f"{self._stream_url}/v3/accounts/{self._account_id}/pricing/stream"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        params = {"instruments": ",".join(instruments), "snapshot": "true"}

        # follow_redirects: OANDA's streaming endpoint answers with a 307 to a
        # relative Location; httpx does NOT follow redirects by default, so
        # raise_for_status() would treat the 307 as fatal and the stream would
        # never connect (observed 2026-08-04).
        async with httpx.AsyncClient(timeout=None, follow_redirects=True) as client:
            async with client.stream("GET", url, headers=headers, params=params) as response:
                response.raise_for_status()
                lines = response.aiter_lines()
                while True:
                    try:
                        line = await asyncio.wait_for(
                            lines.__anext__(), timeout=self._heartbeat_timeout
                        )
                    except StopAsyncIteration:
                        # Server closed the stream cleanly. Not an error here — the
                        # caller's reconnect loop decides what to do about it.
                        logger.info("OANDA pricing stream closed by server")
                        return
                    except (asyncio.TimeoutError, TimeoutError) as exc:
                        # wait_for has already cancelled the pending read; abandoning
                        # the connection is correct — the `async with` blocks close
                        # the response and the client on the way out.
                        raise StreamHeartbeatTimeout(
                            f"no data (price or heartbeat) from the OANDA pricing stream "
                            f"for {self._heartbeat_timeout:.0f}s "
                            f"(STREAM_HEARTBEAT_TIMEOUT_SECONDS) — treating the "
                            f"connection as dead and forcing a reconnect"
                        ) from exc

                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if msg.get("type") != "PRICE" or not msg.get("tradeable"):
                        continue

                    # OANDA timestamps: "2026-05-13T10:00:00.123456789Z"
                    # Trim trailing Z + nanoseconds → keep microseconds (6 digits).
                    raw_ts = msg["time"].rstrip("Z")
                    if "." in raw_ts:
                        head, frac = raw_ts.split(".", 1)
                        frac = frac[:6]
                        raw_ts = f"{head}.{frac}"
                    parsed_dt = datetime.fromisoformat(raw_ts).replace(tzinfo=timezone.utc)

                    yield TickData(
                        instrument=msg["instrument"],
                        bid=float(msg["bids"][0]["price"]),
                        ask=float(msg["asks"][0]["price"]),
                        timestamp=parsed_dt,
                    )

    def get_pip_value(self, instrument: str, pip_size: float) -> float:
        url = f"{self._base_url}/v3/accounts/{self._account_id}/pricing"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        params = {"instruments": instrument, "includeHomeConversions": "true"}
        with httpx.Client(timeout=10) as client:
            response = client.get(url, headers=headers, params=params, follow_redirects=True)
            response.raise_for_status()
        data = response.json()
        if not data.get("prices"):
            raise ValueError(f"No pricing data returned for {instrument}")
        quote_currency = instrument.split("_")[1]
        conversions = {c["currency"]: c for c in data.get("homeConversions", [])}
        if quote_currency not in conversions:
            raise ValueError(f"No home conversion for {quote_currency} in response for {instrument}")
        position_value = float(conversions[quote_currency]["positionValue"])
        return pip_size * position_value

    def _place_order(self, order: OrderRequest) -> OrderResult:
        """Not implemented — Phase 1 is observe-only. Reached only when
        ORDER_PLACEMENT_ENABLED is true (base class enforces that)."""
        raise NotImplementedError

    def get_account(self) -> AccountInfo:
        raise NotImplementedError

    def get_open_positions(self) -> list[PositionData]:
        raise NotImplementedError
