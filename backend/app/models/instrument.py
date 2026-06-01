from sqlalchemy import Boolean, Integer, String, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Instrument(Base):
    __tablename__ = "instruments"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    pip_size: Mapped[float] = mapped_column(Float, nullable=False)
    pip_location: Mapped[int] = mapped_column(Integer, nullable=False)
    asset_class: Mapped[str] = mapped_column(String, nullable=False)
    broker_id: Mapped[str] = mapped_column(String, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    candles: Mapped[list["Candle"]] = relationship(back_populates="instrument")  # type: ignore[name-defined]
    signals: Mapped[list["Signal"]] = relationship(back_populates="instrument")  # type: ignore[name-defined]
    trades: Mapped[list["Trade"]] = relationship(back_populates="instrument")  # type: ignore[name-defined]
