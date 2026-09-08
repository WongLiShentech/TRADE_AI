from datetime import datetime

from sqlalchemy import JSON, DateTime, Index, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Strategy(Base):
    """A named, parameterised signal-generation configuration.

    Why this exists
    ---------------
    ``trades.signal_source`` records WHICH ENGINE fired a signal (``rule_based``),
    not WHICH CONFIGURATION of it. Two variants of the rule engine — one with
    ``MIN_RR_RATIO=2.0``, one with ``1.5`` — are genuinely different strategies
    that will produce different trades, yet both would write ``signal_source=
    'rule_based'`` and become indistinguishable the moment they run side by side.

    A strategy is therefore identified by ``params_hash``: a digest over the
    engine name plus the parameter set that affects signal generation. This
    mirrors how ML artifacts already derive ``params_hash`` (see
    ``services/ml/artifact.py``) — same idea, same guarantee: identical
    configuration ⇒ identical id, any difference ⇒ a new row.

    ``name`` is the human handle (``rule_v1``); ``params_hash`` is the identity.
    A rename does not create a new strategy; a parameter change does.

    Status lifecycle
    ----------------
    ``research``  Backtest only. Never evaluated on live bars.
    ``shadow``    Runs on live bars, records decisions, places no orders.
    ``live``      Eligible to place orders (subject to ORDER_PLACEMENT_ENABLED).
    ``retired``   Kept for history; not evaluated.

    Several strategies may be ``shadow`` at once — that is free, they only record
    opinions. Several ``live`` at once is NOT free: two strategies signalling the
    same direction on the same instrument doubles the risk taken on one view.
    Portfolio-level capital allocation is a prerequisite for that and does not
    exist yet; until it does, keep at most one strategy ``live``.
    """

    __tablename__ = "strategies"

    __table_args__ = (
        # Identity is the parameter set, not the label. Registering the same
        # configuration twice under two names is a mistake we want to fail loudly
        # rather than silently split one strategy's evidence across two rows.
        Index("uq_strategies_params_hash", "params_hash", unique=True),
        Index("ix_strategies_status", "status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    engine: Mapped[str] = mapped_column(String, nullable=False)        # SignalEngine key
    params_hash: Mapped[str] = mapped_column(String, nullable=False)   # identity
    params: Mapped[dict] = mapped_column(JSON, nullable=False)         # the frozen config
    status: Mapped[str] = mapped_column(String, nullable=False)        # see docstring
    # WHAT THIS STRATEGY IS — stable and definitional. It changes only when the
    # configuration changes, which by definition produces a new params_hash and
    # therefore a new row. Kept apart from `notes` because a definition buried
    # under accumulated commentary stops being readable at exactly the moment a
    # second strategy exists to compare it against.
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    # WHAT HAPPENED TO IT — operational, and expected to grow over time.
    notes: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
