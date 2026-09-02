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


# ── execution modes (the venue a strategy is permitted to trade) ─────────────
# A three-state mode rather than a boolean, because "no orders", "practice orders"
# and "real money" are three genuinely different states and a boolean can only
# express two. Collapsing them is how a sandbox flag becomes a live order.
EXECUTION_MODE_OBSERVE = "observe"   # record decisions only — no order, any broker
EXECUTION_MODE_SANDBOX = "sandbox"   # real order tickets, PRACTICE account only
EXECUTION_MODE_LIVE = "live"         # real money — NOT IMPLEMENTED, blocked in base
_EXECUTION_MODES = frozenset({EXECUTION_MODE_OBSERVE, EXECUTION_MODE_SANDBOX, EXECUTION_MODE_LIVE})

# Practice and live are distinguished ONLY by the OANDA host (project rule: no
# account-type setting exists). This marker is what makes "sandbox" verifiable
# rather than merely asserted.
_PRACTICE_URL_MARKER = "fxpractice"


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
        """Fail-closed gate: TWO independent flags must agree, plus a venue check.

        Why two flags and not one
        -------------------------
        ``EXECUTION_MODE`` says WHICH venue may be traded; ``ORDER_PLACEMENT_ENABLED``
        is an independent master switch. Both must be set deliberately, so no single
        edit — or a stray environment variable in a shell, a CI job, a copied .env —
        can start sending orders. They are cheap to check and expensive to get wrong.

        Why the URL is re-checked here
        ------------------------------
        ``EXECUTION_MODE=sandbox`` means *practice money*. But the practice/live
        distinction lives entirely in ``OANDA_BASE_URL`` (project rule: no account-type
        setting exists). So "sandbox" plus a live URL is REAL MONEY with a label that
        says otherwise — the single most dangerous misconfiguration this platform can
        hold, and it is one careless copy-paste away. The mode and the venue are
        therefore verified against each other at the last possible moment, inside the
        one boundary every order must cross.
        """
        if self._settings is None:
            raise OrderPlacementDisabledError(
                f"{type(self).__name__} was constructed without Settings, so the "
                "execution-mode guard cannot be verified. Refusing to place an "
                "order. Pass settings to super().__init__()."
            )

        mode = str(getattr(self._settings, "EXECUTION_MODE", "")).strip().lower()
        rejected = f"Rejected: {order.direction} {order.units} {order.instrument}."

        # Unknown values fail CLOSED. A typo'd mode must never be permissive.
        if mode not in _EXECUTION_MODES:
            raise OrderPlacementDisabledError(
                f"EXECUTION_MODE={mode!r} is not one of {sorted(_EXECUTION_MODES)} — "
                f"refusing to place an order. {rejected}"
            )

        if mode == EXECUTION_MODE_OBSERVE:
            raise OrderPlacementDisabledError(
                "EXECUTION_MODE=observe — the platform records decisions and may not "
                f"send an order to any broker. {rejected}"
            )

        if self._settings.ORDER_PLACEMENT_ENABLED is not True:
            raise OrderPlacementDisabledError(
                f"EXECUTION_MODE={mode!r} but ORDER_PLACEMENT_ENABLED is "
                f"{self._settings.ORDER_PLACEMENT_ENABLED!r}. Both must be set "
                f"deliberately before any order is sent. {rejected}"
            )

        if mode == EXECUTION_MODE_LIVE:
            # Not a capability gap to be quietly filled in later: promoting to live
            # is an explicit, reviewed decision that must also satisfy the promotion
            # gate. Until that exists, this path stays shut.
            raise OrderPlacementDisabledError(
                "EXECUTION_MODE=live is not implemented and is deliberately blocked. "
                f"Real-money execution has not been built or reviewed. {rejected}"
            )

        # mode == sandbox: the venue must actually be a practice venue.
        base_url = str(getattr(self._settings, "OANDA_BASE_URL", "") or "")
        if _PRACTICE_URL_MARKER not in base_url:
            raise OrderPlacementDisabledError(
                f"EXECUTION_MODE=sandbox but OANDA_BASE_URL={base_url!r} is not a "
                f"practice endpoint (expected {_PRACTICE_URL_MARKER!r} in the host). "
                "Sandbox mode means practice money; this configuration would send a "
                f"real order to a live account. {rejected}"
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
