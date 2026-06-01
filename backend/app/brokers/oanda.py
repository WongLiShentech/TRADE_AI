import json
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
)
from app.config import Settings

_OANDA_TYPE_TO_ASSET_CLASS: dict[str, str] = {
    "CURRENCY": "forex",
    "CFD": "cfd",
    "METAL": "metal",
}


class OandaClient(BrokerClient):
    def __init__(self, settings: Settings) -> None:
        self._api_key = settings.OANDA_API_KEY
        self._account_id = settings.OANDA_ACCOUNT_ID
        self._base_url = settings.OANDA_BASE_URL.rstrip("/")
        self._stream_url = settings.OANDA_STREAM_URL

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
    ) -> list[CandleData]:
        _CANDLES_PER_DAY = {"H4": 6, "D": 1}
        candles_per_day = _CANDLES_PER_DAY.get(granularity, 6)
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
                    "price": "M",
                }
                response = client.get(url, headers=headers, params=params)
                response.raise_for_status()

                for candle in response.json().get("candles", []):
                    if not candle.get("complete", False):
                        continue
                    mid = candle["mid"]
                    results.append(
                        CandleData(
                            instrument=instrument,
                            granularity=granularity,
                            timestamp=datetime.strptime(
                                candle["time"], "%Y-%m-%dT%H:%M:%S.%f000Z"
                            ).replace(tzinfo=timezone.utc),
                            open=float(mid["o"]),
                            high=float(mid["h"]),
                            low=float(mid["l"]),
                            close=float(mid["c"]),
                            volume=int(candle["volume"]),
                        )
                    )
                chunk_start = chunk_end

        return results

    async def stream_prices(self, instruments: list[str]) -> AsyncIterator[TickData]:
        url = f"{self._stream_url}/v3/accounts/{self._account_id}/pricing/stream"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        params = {"instruments": ",".join(instruments), "snapshot": "true"}

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("GET", url, headers=headers, params=params) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
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

    def place_order(self, order: OrderRequest) -> OrderResult:
        raise NotImplementedError

    def get_account(self) -> AccountInfo:
        raise NotImplementedError

    def get_open_positions(self) -> list[PositionData]:
        raise NotImplementedError
