from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Trade(Base):
    __tablename__ = "trades"

    # M8-Shadow Phase 2 idempotency backstop: exactly one shadow row per
    # (instrument, signal_time). PARTIAL — backtest rows are deliberately not
    # constrained (a backtest re-run legitimately repeats the pair).
    # See migration 20260801_shadow_trade_unique + services/shadow/recorder.py.
    __table_args__ = (
        Index(
            "uq_trades_shadow_natural_key",
            "instrument_id",
            "opened_at",
            unique=True,
            postgresql_where=text("stage = 'shadow'"),
        ),
    )

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
    # True when the simulator resolved the exit via SL-first tie-break (M1 bar hit both
    # barriers) or the degraded H4/mid fallback (no M1 for the window). Enables relabel.
    ambiguous_resolution: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    opened_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Part 2 — context captured at signal/entry time
    signal_reasoning: Mapped[dict | None] = mapped_column(JSON, nullable=True)     # all indicators at signal time
    confluence_score: Mapped[int | None] = mapped_column(Integer, nullable=True)   # 0–10 strict cap
    session: Mapped[str | None] = mapped_column(String, nullable=True, index=True) # asian | london | ny | overlap
    stop_method: Mapped[str | None] = mapped_column(String, nullable=True)         # atr | structure | trailing
    news_events: Mapped[list | None] = mapped_column(JSON, nullable=True)          # events found within ±NEWS_CALENDAR_WINDOW_HOURS

    # M8-Shadow — ML inference decision recorded at signal time (shadow rows use
    # stage='shadow' and NEVER place an order). Nullable: rule-only rows leave them NULL.
    ml_probability: Mapped[float | None] = mapped_column(Float, nullable=True)   # model P(win), 0.0–1.0
    ml_decision: Mapped[str | None] = mapped_column(String, nullable=True)       # take | skip
    ml_model_id: Mapped[str | None] = mapped_column(String, nullable=True)       # artifact stem that scored it (provenance)

    # ── Attribution (Phase A) ────────────────────────────────────────────────
    # WHICH configuration produced this signal. ``signal_source`` names the engine
    # ('rule_based'); this names the parameterised strategy, so two variants of one
    # engine running side by side stay distinguishable. Nullable: rows predating the
    # registry are backfilled to the strategy their run used, and a NULL here means
    # "unattributed", never "the default one".
    strategy_id: Mapped[int | None] = mapped_column(
        ForeignKey("strategies.id"), nullable=True, index=True
    )

    # ── Excursion extremes (Phase A) ─────────────────────────────────────────
    # The best and worst this trade ever looked, in R, from the intrabar walk the
    # simulator already performs. Denormalised from ``trade_paths`` on purpose:
    # these are the intrabar-EXACT extremes (a per-bar series would miss a spike
    # inside a bar), and "average MFE of losers" is a single GROUP BY rather than a
    # join over a few hundred thousand path rows.
    #
    # ⚠️  LABELS, NEVER FEATURES. Both are known only after the signal. Feeding
    # either to a model as an input is the leakage that scores ~perfectly in
    # backtest and is worthless live. See models/trade_path.py.
    mfe_r: Mapped[float | None] = mapped_column(Float, nullable=True)   # max favourable excursion
    mae_r: Mapped[float | None] = mapped_column(Float, nullable=True)   # max adverse excursion
    # The intrabar stream ran out before the intended horizon, so mfe/mae and the
    # path may understate the true range. Recorded rather than silently short: an
    # incomplete path is indistinguishable from a quiet market otherwise.
    path_truncated: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )

    # Part 4a — auto-classification with confidence
    auto_classification: Mapped[str | None] = mapped_column(String, nullable=True, index=True)  # STRATEGY | NEWS | MANIPULATION | MANUAL | UNCERTAIN
    classification_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)       # 0.0–1.0
    human_classification_override: Mapped[str | None] = mapped_column(String, nullable=True)
    final_classification: Mapped[str | None] = mapped_column(String, nullable=True, index=True)

    instrument: Mapped["Instrument"] = relationship(back_populates="trades")  # type: ignore[name-defined]
