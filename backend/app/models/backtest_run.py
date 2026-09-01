from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String
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

    # M7 metrics (nullable — populated by the backtest runner in Part B)
    profit_factor: Mapped[float | None] = mapped_column(Float, nullable=True)          # gross wins / gross losses
    deflated_sharpe: Mapped[float | None] = mapped_column(Float, nullable=True)        # Bailey & López de Prado (2014)
    probabilistic_sharpe: Mapped[float | None] = mapped_column(Float, nullable=True)   # P(true Sharpe > benchmark)
    trades_full_win: Mapped[int | None] = mapped_column(Integer, nullable=True)        # rr_actual >= 1.9
    trades_partial: Mapped[int | None] = mapped_column(Integer, nullable=True)         # 0.05 <= rr_actual < 1.9
    trades_breakeven: Mapped[int | None] = mapped_column(Integer, nullable=True)       # |rr_actual| < 0.05
    trades_loss: Mapped[int | None] = mapped_column(Integer, nullable=True)            # rr_actual <= -0.05
    avg_holding_hours: Mapped[float | None] = mapped_column(Float, nullable=True)      # mean trade duration
    oos_sample_size: Mapped[int | None] = mapped_column(Integer, nullable=True)        # trade count in the OOS window

    # Per-fold IS/OOS metrics + gate flags + params snapshot + universe/window
    # (the runner's structured result; the scalar columns above carry combined OOS).
    fold_breakdown: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    instrument: Mapped["Instrument"] = relationship()  # type: ignore[name-defined]
