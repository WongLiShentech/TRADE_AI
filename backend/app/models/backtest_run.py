from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    # ⚠️ For a multi-instrument run this stores insts[0].id — one of ten. The truth
    # is in fold_breakdown['instruments']. Do not join lineage on it; you would get
    # a tenth of the picture and no error.
    instrument_id: Mapped[int] = mapped_column(ForeignKey("instruments.id"), nullable=False, index=True)
    # Human-readable strategy name, denormalised so the table reads without a join.
    # `strategy_id` is the authority; the runner asserts the two agree.
    strategy: Mapped[str] = mapped_column(String, nullable=False)
    # WHICH CONFIGURATION this run executed. Added with run lineage: `strategy` alone
    # was written from a module constant and was therefore wrong the first time two
    # strategies existed — run 7 executed rule_based_v2_fixed and recorded
    # rule_based_v1.
    strategy_id: Mapped[int | None] = mapped_column(
        ForeignKey("strategies.id"), nullable=True, index=True
    )
    # WHICH CODE executed this run. NULL means "not recorded" — never a guess;
    # `git log --before <run_at>` would find the commit that existed, not the one
    # that ran. See app/services/provenance.py.
    git_commit: Mapped[str | None] = mapped_column(String, nullable=True)
    git_dirty: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
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
