from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    instrument_id: Mapped[int] = mapped_column(ForeignKey("instruments.id"), nullable=False, index=True)
    strategy: Mapped[str] = mapped_column(String, nullable=False)
    in_sample_start: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    in_sample_end: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    oos_start: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    oos_end: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    trade_count: Mapped[int] = mapped_column(Integer, nullable=False)
    win_rate: Mapped[float] = mapped_column(Float, nullable=False)
    avg_rr: Mapped[float] = mapped_column(Float, nullable=False)
    max_drawdown: Mapped[float] = mapped_column(Float, nullable=False)
    sharpe: Mapped[float] = mapped_column(Float, nullable=False)
    expectancy: Mapped[float] = mapped_column(Float, nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    run_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    instrument: Mapped["Instrument"] = relationship()  # type: ignore[name-defined]
