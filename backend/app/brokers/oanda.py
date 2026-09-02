import asyncio
import json
import logging
from typing import AsyncIterator
from datetime import datetime, timezone, timedelta

import httpx

from app.brokers.base import (
    TradeState,
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
        # instrument -> price decimal places, discovered from the broker on
        # first use (see _price_precision). Fixed per instrument, so caching
        # costs nothing and saves a round-trip on every order.
        self._precision_cache: dict[str, int] = {}

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

    def _price_precision(self, instrument: str) -> int:
        """Decimal places OANDA accepts for this instrument's prices.

        Derived from the broker's own ``pipLocation`` (EUR_USD -4 → 5 dp;
        USD_JPY -2 → 3 dp), never from a hardcoded "JPY pairs are different" rule —
        the platform discovers instrument properties at runtime, and a price with
        the wrong precision is rejected outright (PRICE_PRECISION_EXCEEDED).

        Cached per process: the value is a fixed property of the instrument, and an
        extra round-trip on every order is latency for no information.
        """
        if instrument in self._precision_cache:
            return self._precision_cache[instrument]
        url = f"{self._base_url}/v3/accounts/{self._account_id}/instruments"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        with httpx.Client(timeout=10) as client:
            response = client.get(
                url, headers=headers, params={"instruments": instrument},
                follow_redirects=True,
            )
            response.raise_for_status()
        rows = response.json().get("instruments", [])
        if not rows:
            raise ValueError(f"OANDA returned no instrument details for {instrument}")
        precision = -int(rows[0]["pipLocation"]) + 1
        self._precision_cache[instrument] = precision
        return precision

    def _place_order(self, order: OrderRequest) -> OrderResult:
        """Submit a MARKET order with stop-loss and take-profit attached.

        Reached ONLY through :meth:`BrokerClient.place_order`, which has already
        verified execution mode, the master switch, and that the configured venue is
        a practice endpoint.

        Why the exits are attached to the order (``stopLossOnFill`` /
        ``takeProfitOnFill``) rather than managed by this platform
        ----------------------------------------------------------
        They then live on OANDA's servers. If this process crashes, the host is
        rebooted, the network drops, or the container is redeployed mid-trade, the
        stop is still there. A platform-managed stop is only as available as the
        platform — and an unprotected open position is precisely the failure this
        system exists to avoid. The one exit OANDA cannot enforce is the
        ``SIGNAL_MAX_HOLD_BARS`` time exit, which needs a separate closer job.

        ``FOK`` (fill-or-kill): a market order that cannot be filled in full is
        rejected rather than partially filled. A partial fill would leave a position
        whose size no longer matches the size RiskEngine sized, silently breaking the
        locked risk triangle.
        """
        units = int(order.units) if order.direction.upper() == "BUY" else -int(order.units)
        dp = self._price_precision(order.instrument)
        payload = {
            "order": {
                "type": order.order_type.upper(),
                "instrument": order.instrument,
                "units": str(units),
                "timeInForce": "FOK",
                "positionFill": "DEFAULT",
                "stopLossOnFill": {"price": f"{order.stop_loss:.{dp}f}"},
                "takeProfitOnFill": {"price": f"{order.take_profit:.{dp}f}"},
            }
        }
        url = f"{self._base_url}/v3/accounts/{self._account_id}/orders"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=20) as client:
            response = client.post(url, headers=headers, json=payload, follow_redirects=True)
        body = response.json() if response.content else {}

        # A rejection is a NORMAL outcome (market closed, insufficient margin, stop
        # too close), not an exception: the caller must record it as a rejected
        # attempt rather than lose the decision entirely.
        if response.status_code >= 400 or "orderRejectTransaction" in body:
            reason = (
                body.get("orderRejectTransaction", {}).get("rejectReason")
                or body.get("errorMessage")
                or f"HTTP {response.status_code}"
            )
            logger.warning(
                "OANDA rejected order %s %s %s: %s",
                order.direction, order.units, order.instrument, reason,
            )
            return OrderResult(
                broker_order_id=str(
                    body.get("orderRejectTransaction", {}).get("id", "")
                ),
                status=f"REJECTED:{reason}",
                filled_price=None,
                units=0,
            )

        fill = body.get("orderFillTransaction")
        if not fill:
            # Accepted but unfilled (e.g. queued while the market is closed). Report
            # honestly rather than inventing a fill price.
            create = body.get("orderCreateTransaction", {})
            return OrderResult(
                broker_order_id=str(create.get("id", "")),
                status="PENDING",
                filled_price=None,
                units=0,
            )

        return OrderResult(
            broker_order_id=str(fill.get("id", "")),
            status="FILLED",
            filled_price=float(fill["price"]),
            units=abs(int(float(fill["units"]))),
            # tradeOpened is present on a fill that opened a NEW position; a fill
            # that merely reduced an existing one carries tradeReduced instead.
            broker_trade_id=str((fill.get("tradeOpened") or {}).get("tradeID") or "") or None,
        )

    def get_trade_state(self, broker_trade_id: str) -> TradeState:
        """Ask OANDA what became of one trade. Read-only.

        This is how a stop that fired at 3am — while the process was restarting,
        or the host was down — is discovered. Nothing else records it.
        """
        url = f"{self._base_url}/v3/accounts/{self._account_id}/trades/{broker_trade_id}"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        with httpx.Client(timeout=10) as client:
            response = client.get(url, headers=headers, follow_redirects=True)
            response.raise_for_status()
        t = response.json()["trade"]
        closed = t.get("state") == "CLOSED"
        close_time = None
        if t.get("closeTime"):
            raw = t["closeTime"].rstrip("Z")
            if "." in raw:
                head, frac = raw.split(".", 1)
                raw = f"{head}.{frac[:6]}"
            close_time = datetime.fromisoformat(raw).replace(tzinfo=timezone.utc)
        return TradeState(
            broker_trade_id=str(t["id"]),
            state="CLOSED" if closed else "OPEN",
            units=abs(int(float(t["initialUnits"]))),
            open_price=float(t["price"]),
            close_price=float(t["averageClosePrice"]) if t.get("averageClosePrice") else None,
            realised_pnl=float(t["realizedPL"]) if t.get("realizedPL") is not None else None,
            close_time=close_time,
        )

    def _close_position(self, instrument: str, units: int, direction: str) -> OrderResult:
        """Close at market. Reached only through the guarded ``close_position``.

        Closes by SIDE rather than by trade id: OANDA nets positions per instrument,
        so the platform's one open trade per instrument is the whole side. Passing
        'ALL' avoids a unit-count mismatch if a partial close ever occurred.
        """
        side_key = "longUnits" if direction.upper() == "BUY" else "shortUnits"
        url = f"{self._base_url}/v3/accounts/{self._account_id}/positions/{instrument}/close"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=20) as client:
            response = client.put(
                url, headers=headers, json={side_key: "ALL"}, follow_redirects=True
            )
        body = response.json() if response.content else {}
        fill_key = "longOrderFillTransaction" if direction.upper() == "BUY" else "shortOrderFillTransaction"
        fill = body.get(fill_key)
        if response.status_code >= 400 or not fill:
            reason = body.get("errorMessage") or f"HTTP {response.status_code}"
            logger.warning("OANDA close failed for %s: %s", instrument, reason)
            return OrderResult(
                broker_order_id="", status=f"CLOSE_FAILED:{reason}",
                filled_price=None, units=0,
            )
        return OrderResult(
            broker_order_id=str(fill.get("id", "")),
            status="CLOSED",
            filled_price=float(fill["price"]),
            units=abs(int(float(fill["units"]))),
        )

    def get_account(self) -> AccountInfo:
        """Live account summary. Read-only — cannot move money, so it is unguarded.

        Position sizing MUST use this rather than ``STARTING_BALANCE``: the config
        value is a fixed number, so every size would drift from reality the moment
        the account made or lost anything.
        """
        url = f"{self._base_url}/v3/accounts/{self._account_id}/summary"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        with httpx.Client(timeout=10) as client:
            response = client.get(url, headers=headers, follow_redirects=True)
            response.raise_for_status()
        acc = response.json()["account"]
        return AccountInfo(
            account_id=str(acc["id"]),
            balance=float(acc["balance"]),
            unrealised_pnl=float(acc.get("unrealizedPL", 0.0)),
            currency=str(acc["currency"]),
        )

    def get_open_positions(self) -> list[PositionData]:
        """Positions currently open at the broker. Read-only.

        The broker is the authority on what is actually open — not this platform's
        trade table. Reconciling the two is how a fill that never reached the
        database, or a position closed by a stop the platform never saw, is caught.
        """
        url = f"{self._base_url}/v3/accounts/{self._account_id}/openPositions"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        with httpx.Client(timeout=10) as client:
            response = client.get(url, headers=headers, follow_redirects=True)
            response.raise_for_status()
        results: list[PositionData] = []
        for pos in response.json().get("positions", []):
            for side in ("long", "short"):
                leg = pos.get(side, {})
                units = int(float(leg.get("units", 0) or 0))
                if units == 0:
                    continue
                results.append(
                    PositionData(
                        instrument=pos["instrument"],
                        # OANDA signs units by side; the platform carries direction
                        # separately, so units stays a magnitude here.
                        direction="BUY" if side == "long" else "SELL",
                        units=abs(units),
                        avg_price=float(leg["averagePrice"]),
                        unrealised_pnl=float(leg.get("unrealizedPL", 0.0)),
                    )
                )
        return results
