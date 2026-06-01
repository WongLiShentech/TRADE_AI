from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class BotState(Base):
    """
    Single-row table representing the global bot pause state.

    The circuit breaker (services/circuit_breaker.py) sets paused=True when
    win rate / drawdown / consecutive-loss thresholds are breached.

    M8 (order placement) reads this before placing any new order. Existing
    positions still honour their stops while paused — only new entries are
    refused.

    There is no auto-resume. A human must call POST /api/v1/bot/resume.
    """

    __tablename__ = "bot_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    paused: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    paused_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    paused_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_resumed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
