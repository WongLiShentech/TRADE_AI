"""
Circuit breaker — checks risk thresholds and pauses the bot when breached.

Runs:
- After every closed trade (called from M8 trade-close hook, future)
- Daily at scheduler tick (registered in services/scheduler.py)

Reads:
- settings.MIN_WIN_RATE_ALERT
- settings.MAX_DRAWDOWN_ALERT
- settings.CIRCUIT_BREAKER_LOSSES
- settings.CIRCUIT_BREAKER_WINDOW

Writes:
- bot_state.paused = True (if breached)
- alert via configured delivery channel
"""
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.config import Settings
from app.models.bot_state import BotState
from app.models.trade import Trade
from app.services.alerts.factory import get_alert_delivery


def check_circuit_breaker(db: Session, settings: Settings) -> dict:
    """
    Returns a summary dict; sets bot_state.paused if breached.
    No-op when there is insufficient data (< CIRCUIT_BREAKER_WINDOW closed trades).
    """
    state = _ensure_bot_state(db)
    if state.paused:
        return {"paused": True, "checked": False, "reason": state.paused_reason}

    closed = (
        db.query(Trade)
        .filter(
            Trade.closed_at.isnot(None),
            Trade.final_classification == "STRATEGY",
        )
        .order_by(Trade.closed_at.desc())
        .limit(settings.CIRCUIT_BREAKER_WINDOW)
        .all()
    )
    if len(closed) < settings.CIRCUIT_BREAKER_WINDOW:
        return {"paused": False, "checked": False, "reason": "insufficient_sample"}

    wins = sum(1 for t in closed if t.outcome == "win")
    win_rate = wins / len(closed)
    drawdown = _drawdown_from_trades(closed)
    consecutive_losses = _consecutive_losses(closed)

    breached_reasons: list[str] = []
    if win_rate < settings.MIN_WIN_RATE_ALERT:
        breached_reasons.append(
            f"win_rate {win_rate:.2%} < {settings.MIN_WIN_RATE_ALERT:.2%}"
        )
    if drawdown > settings.MAX_DRAWDOWN_ALERT:
        breached_reasons.append(
            f"drawdown {drawdown:.2%} > {settings.MAX_DRAWDOWN_ALERT:.2%}"
        )
    if consecutive_losses >= settings.CIRCUIT_BREAKER_LOSSES:
        breached_reasons.append(
            f"consecutive_losses {consecutive_losses} >= {settings.CIRCUIT_BREAKER_LOSSES}"
        )

    if breached_reasons:
        reason = "; ".join(breached_reasons)
        state.paused = True
        state.paused_reason = reason
        state.paused_at = datetime.now(timezone.utc)
        db.commit()
        get_alert_delivery(settings).send(
            subject="Bot paused by circuit breaker",
            body=reason,
        )
        return {"paused": True, "checked": True, "reason": reason}

    return {
        "paused": False,
        "checked": True,
        "win_rate": win_rate,
        "drawdown": drawdown,
        "consecutive_losses": consecutive_losses,
    }


def _ensure_bot_state(db: Session) -> BotState:
    state = db.query(BotState).filter_by(id=1).first()
    if state is None:
        state = BotState(id=1, paused=False)
        db.add(state)
        db.commit()
        db.refresh(state)
    return state


def _drawdown_from_trades(trades: list[Trade]) -> float:
    # Trades arrive newest-first. Walk forward by closed_at to compute peak/trough P&L.
    chrono = sorted(trades, key=lambda t: t.closed_at or datetime.min)
    peak = 0.0
    cum = 0.0
    max_dd = 0.0
    for t in chrono:
        # Use risk_amount as proxy if rr_actual missing; conservative.
        pnl = (t.rr_actual or 0.0) * (t.risk_amount or 0.0)
        cum += pnl
        peak = max(peak, cum)
        if peak > 0:
            dd = (peak - cum) / peak
            max_dd = max(max_dd, dd)
    return max_dd


def _consecutive_losses(trades: list[Trade]) -> int:
    # trades is newest-first; count contiguous losses from the most recent.
    n = 0
    for t in trades:
        if t.outcome == "loss":
            n += 1
        else:
            break
    return n
