from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(primary_key=True)
    instrument_id: Mapped[int] = mapped_column(ForeignKey("instruments.id"), nullable=False, index=True)
    direction: Mapped[str] = mapped_column(String, nullable=False)                 # BUY | SELL
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop_price: Mapped[float] = mapped_column(Float, nullable=False)
    tp_price: Mapped[float] = mapped_column(Float, nullable=False)
    units: Mapped[int] = mapped_column(Integer, nullable=False)
    risk_amount: Mapped[float] = mapped_column(Float, nullable=False)
    expected_pip_loss: Mapped[float] = mapped_column(Float, nullable=False)
    actual_pip_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    slippage: Mapped[float | None] = mapped_column(Float, nullable=True)
    rr_entry: Mapped[float] = mapped_column(Float, nullable=False)
    rr_actual: Mapped[float | None] = mapped_column(Float, nullable=True)
    signal_source: Mapped[str] = mapped_column(String, nullable=False)             # rule_based | ml
    stage: Mapped[str] = mapped_column(String, nullable=False, index=True)         # backtest | sandbox | live
    outcome: Mapped[str | None] = mapped_column(String, nullable=True)             # win | loss | breakeven
    exit_reason: Mapped[str | None] = mapped_column(String, nullable=True)         # tp_hit | sl_hit | trailing_stop | time_exit
    opened_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Part 2 — context captured at signal/entry time
    signal_reasoning: Mapped[dict | None] = mapped_column(JSON, nullable=True)     # all indicators at signal time
    confluence_score: Mapped[int | None] = mapped_column(Integer, nullable=True)   # 0–10 strict cap
    session: Mapped[str | None] = mapped_column(String, nullable=True, index=True) # asian | london | ny | overlap
    stop_method: Mapped[str | None] = mapped_column(String, nullable=True)         # atr | structure | trailing
    news_events: Mapped[list | None] = mapped_column(JSON, nullable=True)          # events found within ±NEWS_CALENDAR_WINDOW_HOURS

    # Part 4a — auto-classification with confidence
    auto_classification: Mapped[str | None] = mapped_column(String, nullable=True, index=True)  # STRATEGY | NEWS | MANIPULATION | MANUAL | UNCERTAIN
    classification_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)       # 0.0–1.0
    human_classification_override: Mapped[str | None] = mapped_column(String, nullable=True)
    final_classification: Mapped[str | None] = mapped_column(String, nullable=True, index=True)

    instrument: Mapped["Instrument"] = relationship(back_populates="trades")  # type: ignore[name-defined]
