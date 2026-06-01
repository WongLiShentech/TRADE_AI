from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(primary_key=True)
    instrument_id: Mapped[int] = mapped_column(ForeignKey("instruments.id"), nullable=False, index=True)
    granularity: Mapped[str] = mapped_column(String, nullable=False)              # H4 | D
    direction: Mapped[str] = mapped_column(String, nullable=False)                # BUY | SELL
    entry: Mapped[float] = mapped_column(Float, nullable=False)
    stop: Mapped[float] = mapped_column(Float, nullable=False)
    target: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_score: Mapped[int] = mapped_column(Integer, nullable=False)
    score_breakdown: Mapped[dict] = mapped_column(JSON, nullable=False)           # {"trend": true, "rsi": true, ...}
    status: Mapped[str] = mapped_column(String, nullable=False, default="PENDING")  # PENDING|APPROVED|REJECTED|EXPIRED|EXECUTED
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    rejection_reason: Mapped[str | None] = mapped_column(String, nullable=True)

    instrument: Mapped["Instrument"] = relationship(back_populates="signals")  # type: ignore[name-defined]

    __table_args__ = (
        Index("ix_signals_instrument_status", "instrument_id", "status"),
        Index("ix_signals_created_at", "created_at"),
    )
