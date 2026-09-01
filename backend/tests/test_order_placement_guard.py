"""ORDER_PLACEMENT_ENABLED must block orders at the BROKER BOUNDARY — QA HIGH 4.

Before this suite the flag was only consulted inside ``ml.inference`` and the shadow
recorder, so its documented platform-wide meaning ("no order ever reaches a broker")
was procedural: any *other* order path — a manual endpoint, a future execution
service, a script — would have sailed straight past it.

These tests pin the guarantee where it actually matters:

* every ``BrokerClient`` refuses ``place_order`` while the flag is false, regardless
  of broker, and refuses BEFORE the subclass's transport method is reached;
* ``BrokerRouter`` — the mandatory access layer — refuses to even route;
* a NEW broker inherits the guard without writing a line of guard code, which is the
  property that keeps this fixed rather than merely fixed-today.

Run from backend/:  python -m pytest tests/test_order_placement_guard.py -v
"""
from __future__ import annotations

from datetime import datetime
from typing import AsyncIterator

import pytest

from app.brokers.base import (
    AccountInfo,
    BrokerClient,
    CandleData,
    InstrumentData,
    OrderPlacementDisabledError,
    OrderRequest,
    OrderResult,
    PositionData,
    TickData,
)
from app.brokers.factory import get_broker_client
from app.brokers.router import BrokerRouter

_ORDER = OrderRequest(
    instrument="TEST_PAIR",   # never a real symbol — the guard is instrument-agnostic
    direction="BUY",
    units=1000,
    order_type="MARKET",
    stop_loss=1.0,
    take_profit=2.0,
)


class _SpyClient(BrokerClient):
    """A brand-new broker that implements ONLY ``_place_order``.

    It writes no guard code of its own. If ``placed`` ever flips True while the flag
    is false, the boundary leaked — which is the whole point of the test.
    """

    def __init__(self, settings) -> None:
        super().__init__(settings)
        self.placed = False

    def _place_order(self, order: OrderRequest) -> OrderResult:
        self.placed = True
        return OrderResult(broker_order_id="spy-1", status="FILLED",
                           filled_price=1.0, units=order.units)

    def get_instruments(self) -> list[InstrumentData]:
        raise NotImplementedError

    def get_candles(self, instrument, granularity, start, end, price_type="M") -> list[CandleData]:
        raise NotImplementedError

    async def stream_prices(self, instruments: list[str]) -> AsyncIterator[TickData]:
        raise NotImplementedError

    def get_pip_value(self, instrument: str, pip_size: float) -> float:
        raise NotImplementedError

    def get_account(self) -> AccountInfo:
        raise NotImplementedError

    def get_open_positions(self) -> list[PositionData]:
        raise NotImplementedError


def _with(settings, **overrides):
    return settings.model_copy(update=overrides)


# ── 1. the live configuration is observe-only ────────────────────────────────
def test_live_settings_have_order_placement_disabled(settings):
    """A regression tripwire: M8-Shadow is only safe while this is false."""
    assert settings.ORDER_PLACEMENT_ENABLED is False


# ── 2. the boundary blocks BEFORE the transport is reached ───────────────────
def test_client_refuses_to_place_an_order_when_the_flag_is_false(settings):
    client = _SpyClient(_with(settings, ORDER_PLACEMENT_ENABLED=False))

    with pytest.raises(OrderPlacementDisabledError) as exc:
        client.place_order(_ORDER)

    assert client.placed is False, "the transport must never be reached"
    assert "ORDER_PLACEMENT_ENABLED" in str(exc.value)


def test_client_delegates_to_the_transport_when_the_flag_is_true(settings):
    """The guard must be a gate, not a wall — flipping the flag really does let the
    order through, so the test proves the block is caused by the flag alone."""
    client = _SpyClient(_with(settings, ORDER_PLACEMENT_ENABLED=True))

    result = client.place_order(_ORDER)

    assert client.placed is True
    assert result.units == _ORDER.units


def test_guard_is_inherited_by_a_broker_that_writes_no_guard_code(settings):
    """_SpyClient implements only ``_place_order``. Structural, not procedural."""
    assert "_assert_order_placement_enabled" not in _SpyClient.__dict__
    assert "place_order" not in _SpyClient.__dict__

    with pytest.raises(OrderPlacementDisabledError):
        _SpyClient(_with(settings, ORDER_PLACEMENT_ENABLED=False)).place_order(_ORDER)


def test_client_without_settings_fails_closed(settings):
    """A subclass that forgets ``super().__init__(settings)`` must refuse orders, not
    permit them — the safe direction for a safety flag."""

    class _Forgetful(_SpyClient):
        def __init__(self) -> None:  # noqa: D107 — deliberately skips super().__init__
            BrokerClient.__init__(self, None)
            self.placed = False

    with pytest.raises(OrderPlacementDisabledError) as exc:
        _Forgetful().place_order(_ORDER)

    assert "without Settings" in str(exc.value)


# ── 3. the real Phase-1 broker is covered by the same gate ───────────────────
def test_the_configured_broker_client_is_blocked_too(settings):
    """Instrument- and broker-agnostic: whichever client BROKER_FOREX names, it must
    raise the guard error rather than its own NotImplementedError."""
    client = get_broker_client(settings.BROKER_FOREX, _with(settings, ORDER_PLACEMENT_ENABLED=False))

    with pytest.raises(OrderPlacementDisabledError):
        client.place_order(_ORDER)


def test_every_broker_client_routes_place_order_through_the_base_guard():
    """No broker may override ``place_order`` — that is the only way to bypass it."""
    from app.brokers import alpaca, binance, oanda

    for module in (oanda, binance, alpaca):
        for obj in vars(module).values():
            if isinstance(obj, type) and issubclass(obj, BrokerClient) and obj is not BrokerClient:
                assert "place_order" not in obj.__dict__, (
                    f"{obj.__name__} overrides place_order and bypasses the safety guard"
                )
                assert "_place_order" in obj.__dict__


# ── 4. the mandatory access layer blocks before it even routes ───────────────
def test_router_refuses_to_route_an_order_when_the_flag_is_false(settings):
    router = BrokerRouter(_with(settings, ORDER_PLACEMENT_ENABLED=False))

    assert router.order_placement_enabled() is False
    # db=None proves the refusal happens BEFORE any instrument lookup: if the guard
    # ran late, this would raise AttributeError instead.
    with pytest.raises(OrderPlacementDisabledError):
        router.place_order(_ORDER.instrument, _ORDER, None)


def test_router_reports_enabled_when_the_flag_is_true(settings):
    assert BrokerRouter(_with(settings, ORDER_PLACEMENT_ENABLED=True)).order_placement_enabled() is True


def test_guard_error_is_a_runtime_error(settings):
    """Subclassing RuntimeError keeps existing ``except RuntimeError`` handlers working."""
    assert issubclass(OrderPlacementDisabledError, RuntimeError)
