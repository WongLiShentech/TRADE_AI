from datetime import datetime

from sqlalchemy import DateTime, Float, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class EquityPoint(Base):
    __tablename__ = "equity_points"

    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    balance: Mapped[float] = mapped_column(Float, nullable=False)
    unrealised_pnl: Mapped[float] = mapped_column(Float, nullable=False)
    stage: Mapped[str] = mapped_column(String, nullable=False)  # backtest | sandbox | live
