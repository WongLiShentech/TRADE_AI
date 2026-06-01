from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Indicator(Base):
    __tablename__ = "indicators"
    __table_args__ = (UniqueConstraint("instrument_id", "granularity", "timestamp"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    instrument_id: Mapped[int] = mapped_column(ForeignKey("instruments.id"), nullable=False, index=True)
    granularity: Mapped[str] = mapped_column(String, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    atr14: Mapped[float | None] = mapped_column(Float, nullable=True)
    rsi14: Mapped[float | None] = mapped_column(Float, nullable=True)
    swing_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    swing_low: Mapped[float | None] = mapped_column(Float, nullable=True)

    instrument: Mapped["Instrument"] = relationship()  # type: ignore[name-defined]
