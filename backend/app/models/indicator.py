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
    # CENTRED window (bar i depends on bars up to i+k) — NOT knowable at its own
    # timestamp. Any read at a decision time must drop the newest k rows; see
    # feature_builder.causal_swing_levels. Never read raw in a signal rule.
    swing_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    swing_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    # TRAILING window including the current bar — knowable at its own timestamp, so it
    # needs no confirmation lag and is safe to read directly at a decision time.
    # Period is SIGNAL_DONCHIAN_PERIOD; deliberately not baked into the column name so
    # retuning it is a config change rather than a migration.
    donchian_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    donchian_low: Mapped[float | None] = mapped_column(Float, nullable=True)

    instrument: Mapped["Instrument"] = relationship()  # type: ignore[name-defined]
