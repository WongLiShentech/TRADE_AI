from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class ModelDecision(Base):
    """One model's verdict on one signal. MANY rows per trade — that is the point.

    Why this exists
    ---------------
    ``trades`` carries ``ml_model_id`` / ``ml_probability`` / ``ml_decision`` —
    room for exactly ONE model's opinion per signal. Score the same historical
    signal with a newly trained challenger and the champion's verdict is
    overwritten. The question champion/challenger promotion is built to answer —
    *"where the two models disagreed, which was right?"* — is unanswerable,
    because one of the two answers no longer exists.

    Splitting opinions out makes the trade an immutable FACT (what the market did)
    and the decision a revisable OPINION (what a model thought), which is what
    they actually are. A challenger can then score the entire history without
    touching a single trade row.

    The ``trades.ml_*`` columns are deliberately NOT dropped: they are the live
    read path today, and this table is backfilled from them. Migrating the read
    path is a separate, later change.

    Authority
    ---------
    When several models score one signal, exactly one of them governed what was
    recorded (and, once orders are enabled, what was actually traded).
    ``is_authoritative`` marks it. Without that flag a backfilled challenger's
    verdict is indistinguishable from the champion's, and any "what did we
    actually do?" query silently double-counts.

    Leakage note
    ------------
    Rows here are written at or after signal time and may be written LONG after,
    when a new model rescores history. They are evidence about models, never
    features about markets. Nothing in this table may ever reach
    ``feature_builder`` — a challenger's opinion of a 2023 signal is not
    information that existed in 2023.
    """

    __tablename__ = "model_decisions"

    __table_args__ = (
        # One opinion per model per trade. Rescoring updates in place rather than
        # accumulating duplicates that would skew every aggregate.
        Index("uq_model_decisions_trade_model", "trade_id", "model_id", unique=True),
        Index("ix_model_decisions_model_id", "model_id"),
        # Partial: "the decision that governed this trade" is the hot lookup, and
        # only one row per trade qualifies, so indexing the whole table would be
        # mostly dead weight.
        Index(
            "ix_model_decisions_authoritative",
            "trade_id",
            postgresql_where=text("is_authoritative"),
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    trade_id: Mapped[int] = mapped_column(
        ForeignKey("trades.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model_id: Mapped[str] = mapped_column(String, nullable=False)   # artifact filename stem
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    decision: Mapped[str] = mapped_column(String, nullable=False)   # take | skip
    # The threshold in force when this verdict was produced. Stored per row because
    # it is a property of the DECISION, not of the model: the same artifact scored
    # under a revised threshold yields a different decision from the same
    # probability, and without this column that difference is invisible.
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    # How many of the model's feature keys were NaN for this row. A verdict
    # produced on a crippled feature vector is not comparable to one produced on a
    # complete one; recording it per decision is what makes that filterable
    # instead of a footnote in the logs.
    nan_features: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_authoritative: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    scored_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
