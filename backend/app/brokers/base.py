"""Broker abstraction — ``BrokerClient`` ABC + the transport-neutral dataclasses.

Order-placement safety
----------------------
``place_order`` is a CONCRETE template method on the ABC, not an abstract one. It
enforces ``settings.ORDER_PLACEMENT_ENABLED`` and only then delegates to the
subclass's ``_place_order``. That inversion is deliberate: the flag's documented
meaning is platform-wide ("no order ever reaches a broker"), but it was previously
only checked in the ML inference / shadow-recorder path — so any OTHER order path,
present or future, would have bypassed it entirely.

Putting the check on the base class makes the guarantee structural rather than
procedural: a new broker implements ``_place_order`` and inherits the guard whether
or not its author knew the flag existed. There is no way to add an unguarded order
path short of deliberately overriding ``place_order`` itself.
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, AsyncIterator, Optional
from dataclasses import dataclass
from datetime import datetime

if TYPE_CHECKING:  # pragma: no cover — import only for type checking
    from app.config import Settings


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


class OrderPlacementDisabledError(RuntimeError):
    """Raised when an order is attempted while ``ORDER_PLACEMENT_ENABLED`` is false.

    A ``RuntimeError`` subclass so existing ``except RuntimeError`` handlers still
    catch it, but a distinct type so a caller that legitimately wants to distinguish
    "the platform is in observe-only mode" from "the broker call failed" can.
    """


class StreamHeartbeatTimeout(TimeoutError):
    """Raised when a price stream delivers nothing — not even a heartbeat — in time.

    This is the failure a plain read timeout cannot catch. A half-open TCP
    connection (VM suspend/resume, NAT table eviction, silent broker-side drop)
    leaves the socket looking perfectly readable: no RST arrives, no exception is
    raised, and an ``async for`` over the response lines simply blocks forever. The
    process stays "healthy", the reconnect loop never fires, and live pricing is
    dead until a human notices.

    Every streaming broker is expected to enforce a per-line watchdog of
    ``STREAM_HEARTBEAT_TIMEOUT_SECONDS`` and raise this so the reconnect loop in
    ``services.price_stream`` treats it as an ordinary disconnect.

    A ``TimeoutError`` subclass (which is also ``OSError``) so generic transport
    handlers catch it, while a caller that wants to distinguish "the feed went
    silent" from "the connection was refused" still can.
    """


class BrokerClient(ABC):
    """Provider-agnostic broker interface.

    Subclasses MUST accept a ``Settings`` in ``__init__`` and call
    ``super().__init__(settings)`` — the base constructor is what arms the
    order-placement guard. A subclass that skips it gets a client whose
    :meth:`place_order` refuses every order, which is the correct fail-CLOSED
    direction for a safety flag.
    """

    def __init__(self, settings: Optional["Settings"] = None) -> None:
        self._settings: Optional["Settings"] = settings

    # ── order placement (guarded template method — do NOT override) ──────────
    def place_order(self, order: OrderRequest) -> OrderResult:
        """Place an order, if and only if the platform is permitted to trade.

        Concrete on purpose: this is the single boundary every order must cross, so
        the ``ORDER_PLACEMENT_ENABLED`` check lives here rather than being repeated
        (and eventually forgotten) in each caller. Subclasses implement
        :meth:`_place_order` and inherit the guard automatically.

        Args:
            order: the fully-specified order (instrument, direction, units, SL, TP).
                Units must already have been derived by ``RiskEngine`` — this method
                sizes nothing.

        Returns:
            The broker's :class:`OrderResult`.

        Raises:
            OrderPlacementDisabledError: ``ORDER_PLACEMENT_ENABLED`` is not ``True``,
                or no ``Settings`` was supplied to the constructor (fail-closed).
                Raised BEFORE any network call — nothing reaches the broker.
        """
        self._assert_order_placement_enabled(order)
        return self._place_order(order)

    def _assert_order_placement_enabled(self, order: OrderRequest) -> None:
        """Fail-closed gate on the platform-wide order-placement flag."""
        if self._settings is None:
            raise OrderPlacementDisabledError(
                f"{type(self).__name__} was constructed without Settings, so the "
                "platform-wide ORDER_PLACEMENT_ENABLED flag cannot be verified. "
                "Refusing to place an order. Pass settings to super().__init__()."
            )
        if self._settings.ORDER_PLACEMENT_ENABLED is not True:
            raise OrderPlacementDisabledError(
                "ORDER_PLACEMENT_ENABLED is "
                f"{self._settings.ORDER_PLACEMENT_ENABLED!r} — the platform is in "
                "observe-only mode and may not send an order to any broker. "
                f"Rejected: {order.direction} {order.units} {order.instrument}. "
                "Flipping this flag to true is the deliberate, reviewed sandbox "
                "transition; it must never be flipped while an unpromoted model is "
                "loaded."
            )

    @abstractmethod
    def _place_order(self, order: OrderRequest) -> OrderResult:
        """Broker-specific order submission. Called ONLY by :meth:`place_order`,
        which has already verified the platform is permitted to trade."""
        raise NotImplementedError

    # ── read-only operations (unguarded — they cannot move money) ────────────
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
        price_type: str = "M",
    ) -> list[CandleData]:
        raise NotImplementedError

    @abstractmethod
    async def stream_prices(self, instruments: list[str]) -> AsyncIterator[TickData]:
        raise NotImplementedError

    @abstractmethod
    def get_pip_value(self, instrument: str, pip_size: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_account(self) -> AccountInfo:
        raise NotImplementedError

    @abstractmethod
    def get_open_positions(self) -> list[PositionData]:
        raise NotImplementedError
