from typing import AsyncIterator
from datetime import datetime

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


class AlpacaClient(BrokerClient):
    """Phase 2 scaffold â€” not implemented."""

    def __init__(self, settings: Settings) -> None:
        # Arms the base class's ORDER_PLACEMENT_ENABLED guard — see brokers/base.py.
        super().__init__(settings)
        self._api_key = settings.ALPACA_API_KEY
        self._secret_key = settings.ALPACA_SECRET_KEY
        self._base_url = settings.ALPACA_BASE_URL

    def get_instruments(self) -> list[InstrumentData]:
        raise NotImplementedError

    def get_candles(
        self,
        instrument: str,
        granularity: str,
        start: datetime,
        end: datetime,
    ) -> list[CandleData]:
        raise NotImplementedError

    async def stream_prices(self, instruments: list[str]) -> AsyncIterator[TickData]:
        raise NotImplementedError

    def get_pip_value(self, instrument: str) -> float:
        raise NotImplementedError

    def _place_order(self, order: OrderRequest) -> OrderResult:
        """Phase 2 scaffold. Reached only when ORDER_PLACEMENT_ENABLED is true
        (the base class enforces that before delegating here)."""
        raise NotImplementedError

    def get_account(self) -> AccountInfo:
        raise NotImplementedError

    def get_open_positions(self) -> list[PositionData]:
        raise NotImplementedError
