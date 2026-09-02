"""Sandbox execution — real practice orders, recorded as ``stage='sandbox'`` rows.

What this adds over shadow
--------------------------
Shadow proves the DECISION: it records what the model chose and, later, what the
market would have done. It cannot prove the PLUMBING — whether an order fills, at
what price, whether the broker rejects it, what the spread actually costs. Those
only appear when a real ticket is sent.

So the two run together and record different things:

    model says TAKE  ->  a real practice order   ->  stage='sandbox'  (real fill)
    model says SKIP  ->  the counterfactual      ->  stage='shadow'   (simulated)

Both are needed. Sandbox alone would produce almost nothing — the model has taken
roughly one signal in twelve — and shadow alone can never tell you whether the
simulator's fill assumptions were true.

Ordering: the row is written BEFORE the order is sent
-----------------------------------------------------
Deliberately. If the process dies between the two, the outcomes are:

    row first, then order   ->  a row with no position. Harmless; reconciliation
                                sees no broker trade id and marks it unfilled.
    order first, then row   ->  a POSITION WITH NO ROW. Nothing in this platform
                                knows it exists, no time exit will ever close it,
                                and it is discoverable only by a human looking at
                                OANDA.

The second is far worse, so the row goes first and is updated with the real fill
afterwards.

Who closes what
---------------
The stop and target are attached to the order, so OANDA enforces them server-side
and they survive this process dying. The ONE exit OANDA knows nothing about is
``SIGNAL_MAX_HOLD_BARS`` — the time exit — which is why :func:`sync_open_trades`
exists and must run on a schedule.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.orm import Session

from app.brokers.base import OrderRequest, OrderPlacementDisabledError
from app.domain.execution_mode import (
    EXECUTION_MODE_SANDBOX,
    normalise as normalise_execution_mode,
)
from app.brokers.router import BrokerRouter
from app.config import Settings
from app.domain.timeframes import get_timeframe
from app.models.instrument import Instrument
from app.models.trade import Trade
from app.services.backtester.runner import label_outcome
from app.services.shadow.recorder import REASONING_SHADOW_KEY

logger = logging.getLogger(__name__)

STAGE_SANDBOX = "sandbox"
# Sub-key of the namespaced metadata holding broker-side execution facts. Kept
# beside the shadow metadata rather than in new columns: these are provenance for
# one stage, not universal trade attributes.
REASONING_SANDBOX_KEY = "sandbox"
_ORDER_TYPE_MARKET = "MARKET"


def is_sandbox(settings: Settings) -> bool:
    """True when this deployment is permitted to send practice orders."""
    return normalise_execution_mode(getattr(settings, "EXECUTION_MODE", "")) == EXECUTION_MODE_SANDBOX


def place_for_trade(
    db: Session,
    settings: Settings,
    trade: Trade,
    instrument: Instrument,
    signal,
) -> Trade:
    """Send the practice order for an ALREADY-RECORDED ``stage='sandbox'`` row.

    The row exists first by design (see module docstring), and the recorder has
    already decided this signal warrants an order — decision was 'take', risk
    approved, mode is sandbox. This function does exactly one thing: send the ticket
    and write down what the broker said.

    Never raises. A broker failure is recorded on the row and the live loop carries
    on: losing the decision would be worse than losing the order, and the unfilled
    row is resolved as a non-event by :func:`sync_open_trades`.
    """
    order = OrderRequest(
        instrument=instrument.symbol,
        direction=signal.direction,
        units=int(trade.units),
        order_type=_ORDER_TYPE_MARKET,
        stop_loss=float(signal.stop),
        take_profit=float(signal.target),
    )
    meta: dict = {"order_sent_at": datetime.now(timezone.utc).isoformat()}
    try:
        client = BrokerRouter(settings).for_instrument(instrument.symbol, db)
        result = client.place_order(order)
        meta.update(
            status=result.status,
            broker_order_id=result.broker_order_id,
            broker_trade_id=result.broker_trade_id,
            requested_units=int(trade.units),
            filled_units=result.units,
            requested_price=float(signal.entry),
            filled_price=result.filled_price,
        )
        if result.filled_price is not None:
            # Slippage against the price the signal was built on — the number shadow
            # mode structurally cannot produce, and the reason sandbox exists.
            meta["slippage"] = float(result.filled_price) - float(signal.entry)
            # The REAL fill replaces the modelled entry, so every downstream R is
            # measured from what actually happened rather than what was hoped for.
            trade.entry_price = float(result.filled_price)
            trade.units = int(result.units)
        logger.info(
            "sandbox order %s %s units=%s -> %s @ %s",
            signal.direction, instrument.symbol, trade.units,
            result.status, result.filled_price,
        )
    except OrderPlacementDisabledError as exc:
        # The guard refused. Not this module's error — record it and move on.
        meta.update(status="BLOCKED", error=str(exc))
        logger.warning("sandbox order blocked by the execution guard: %s", exc)
    except Exception as exc:                                       # noqa: BLE001
        meta.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
        logger.exception("sandbox order failed for %s", instrument.symbol)

    reasoning = dict(trade.signal_reasoning or {})
    reasoning[REASONING_SANDBOX_KEY] = meta
    trade.signal_reasoning = reasoning
    db.commit()
    return trade


def sync_open_trades(db: Session, settings: Settings, now: Optional[datetime] = None) -> dict:
    """Reconcile open sandbox trades with the broker, and apply the time exit.

    Two jobs, one pass, because both need the same broker round-trip:

    1. **Reconcile.** A stop or target may have fired at any moment — including
       while this process was restarting. The broker is the only record of that, so
       every open row is asked about rather than assumed.
    2. **Time exit.** OANDA enforces the attached stop and target but knows nothing
       about ``SIGNAL_MAX_HOLD_BARS``. Without this pass, a trade that touches
       neither barrier stays open indefinitely.

    Returns a summary dict; never raises. One bad row must not stall the queue.
    """
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    summary = {"checked": 0, "still_open": 0, "closed_by_broker": 0,
               "closed_on_time": 0, "unfilled": 0, "errors": 0}
    if not is_sandbox(settings):
        return summary

    open_rows = (
        db.query(Trade)
        .filter(Trade.stage == STAGE_SANDBOX, Trade.rr_actual.is_(None))
        .order_by(Trade.opened_at)
        .all()
    )
    if not open_rows:
        return summary

    router = BrokerRouter(settings)
    for trade in open_rows:
        summary["checked"] += 1
        try:
            meta = (trade.signal_reasoning or {}).get(REASONING_SANDBOX_KEY) or {}
            broker_trade_id = meta.get("broker_trade_id")
            if not broker_trade_id:
                # The order never filled (rejected, blocked, or the process died
                # before it was sent). Resolve it as a non-event rather than leaving
                # it open forever — it is not evidence about the market.
                trade.rr_actual = 0.0
                trade.outcome = "breakeven"
                trade.exit_reason = "not_filled"
                trade.closed_at = now
                trade.ambiguous_resolution = True
                summary["unfilled"] += 1
                db.commit()
                continue

            inst = db.get(Instrument, trade.instrument_id)
            client = router.for_instrument(inst.symbol, db)
            state = client.get_trade_state(str(broker_trade_id))

            if state.state == "CLOSED":
                _finalise(db, trade, state, exit_reason=meta.get("pending_close_reason") or "broker_exit", now=now)
                summary["closed_by_broker"] += 1
                continue

            if _bars_elapsed(trade, settings, now) >= int(settings.SIGNAL_MAX_HOLD_BARS):
                client.close_position(inst.symbol, int(trade.units), trade.direction)
                state = client.get_trade_state(str(broker_trade_id))
                _finalise(db, trade, state, exit_reason="time_exit", now=now)
                summary["closed_on_time"] += 1
                continue

            summary["still_open"] += 1
        except Exception:                                          # noqa: BLE001
            summary["errors"] += 1
            db.rollback()
            logger.exception("sandbox sync failed for trade id=%s", trade.id)

    logger.info("sandbox sync: %s", summary)
    return summary


# ── internals ────────────────────────────────────────────────────────────────
def _bars_elapsed(trade: Trade, settings: Settings, now: datetime) -> float:
    """Signal-timeframe bars since the signal, in wall-clock terms.

    Deliberately wall-clock rather than the backtester's observed-bar count: a live
    position is exposed to real time, including weekends. Counting only trading bars
    would hold a position across a closed market and call it 'within the horizon'.
    """
    meta = (trade.signal_reasoning or {}).get(REASONING_SHADOW_KEY) or {}
    granularity = meta.get("granularity") or "H4"
    period_hours = get_timeframe(granularity).period_hours
    return (now - trade.opened_at).total_seconds() / 3600.0 / period_hours


def _finalise(db: Session, trade: Trade, state, exit_reason: str, now: datetime) -> None:
    """Write the real-fill outcome onto a closed sandbox row.

    ``rr_actual`` is computed from the ACTUAL open and close prices against the
    row's own risk unit — not from the broker's P&L, which is denominated in account
    currency and so is not comparable across instruments or account sizes. R is the
    unit every other stage reports in; sandbox must match or it cannot be compared.
    """
    risk_unit = abs(float(trade.entry_price) - float(trade.stop_price))
    close_price = state.close_price
    if close_price is None or risk_unit <= 0:
        logger.warning("sandbox: trade id=%s closed without a usable price", trade.id)
        return
    sign = 1.0 if str(trade.direction).upper() == "BUY" else -1.0
    rr = sign * (float(close_price) - float(trade.entry_price)) / risk_unit

    trade.exit_price = float(close_price)
    trade.rr_actual = float(rr)
    trade.outcome = label_outcome(rr)
    trade.exit_reason = exit_reason
    trade.closed_at = (
        state.close_time.replace(tzinfo=None) if state.close_time else now
    )
    pip_size = None
    inst = db.get(Instrument, trade.instrument_id)
    if inst is not None and inst.pip_size:
        pip_size = float(inst.pip_size)
    if pip_size:
        trade.actual_pip_loss = abs(float(close_price) - float(trade.entry_price)) / pip_size

    meta = dict(trade.signal_reasoning or {})
    sandbox_meta = dict(meta.get(REASONING_SANDBOX_KEY) or {})
    sandbox_meta.update(
        close_price=float(close_price),
        realised_pnl=state.realised_pnl,
        closed_state=state.state,
    )
    meta[REASONING_SANDBOX_KEY] = sandbox_meta
    trade.signal_reasoning = meta
    db.commit()
    logger.info(
        "sandbox closed: trade id=%s %s rr=%.3f outcome=%s reason=%s",
        trade.id, trade.direction, rr, trade.outcome, exit_reason,
    )
