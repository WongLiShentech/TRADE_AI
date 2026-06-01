from abc import ABC, abstractmethod
from typing import AsyncIterator
from dataclasses import dataclass
from datetime import datetime


@dataclass
class InstrumentData:
    symbol: str
    display_name: str
    pip_size: float
    pip_location: int
    asset_class: str
    broker_id: str


@dataclass
class CandleData:
    instrument: str
    granularity: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class TickData:
    instrument: str
    bid: float
    ask: float
    timestamp: datetime


@dataclass
class OrderRequest:
    instrument: str
    direction: str
    units: int
    order_type: str
    stop_loss: float
    take_profit: float
    price: float | None = None


@dataclass
class OrderResult:
    broker_order_id: str
    status: str
    filled_price: float | None
    units: int


@dataclass
class AccountInfo:
    account_id: str
    balance: float
    unrealised_pnl: float
    currency: str


@dataclass
class PositionData:
    instrument: str
    direction: str
    units: int
    avg_price: float
    unrealised_pnl: float


class BrokerClient(ABC):
    @abstractmethod
    def get_instruments(self) -> list[InstrumentData]:
        raise NotImplementedError

    @abstractmethod
    def get_candles(
        self,
        instrument: str,
        granularity: str,
        start: datetime,
        end: datetime,
    ) -> list[CandleData]:
        raise NotImplementedError

    @abstractmethod
    async def stream_prices(self, instruments: list[str]) -> AsyncIterator[TickData]:
        raise NotImplementedError

    @abstractmethod
    def get_pip_value(self, instrument: str, pip_size: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def place_order(self, order: OrderRequest) -> OrderResult:
        raise NotImplementedError

    @abstractmethod
    def get_account(self) -> AccountInfo:
        raise NotImplementedError

    @abstractmethod
    def get_open_positions(self) -> list[PositionData]:
        raise NotImplementedError
