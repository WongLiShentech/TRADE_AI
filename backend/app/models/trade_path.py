from sqlalchemy import Boolean, Float, ForeignKey, Index, Integer
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class TradePath(Base):
    """Where a trade TRAVELLED, bar by bar — not merely where it ended.

    Why this exists
    ---------------
    ``trades.rr_actual`` records one number: the result. A trade that climbed to
    +0.84R before decaying to −0.22R and one that never moved above zero both
    store ``−0.22, loss``. Those are different problems with the same row:
    the first says the profit-taking trigger is set too far away, the second says
    the setup was dead on arrival. Today they are indistinguishable, and every
    exit-policy question — *is the target too greedy? is the stop too tight?
    should we hold longer?* — is therefore unanswerable.

    The simulator already walks every M1 bar to find the barriers. It simply
    discards the extremes it passes on the way. This table is the notepad.

    Path, not just MFE/MAE
    ----------------------
    ``trades.mfe_r`` / ``mae_r`` hold the intrabar-exact extremes and answer most
    questions in one indexed read. This table holds the TRAJECTORY, which answers
    a class the scalars cannot: *"what would a 1.5R target have produced?"*,
    *"what was the R-multiple at bar 20?"*. Storing both is not duplication —
    the scalars are more precise (true intrabar extremes) and the series is more
    expressive (ordering over time).

    Beyond the exit
    ---------------
    Rows with ``beyond_exit=True`` describe bars AFTER the trade actually closed.
    They exist to answer *"should we have held longer?"* — a question the closed
    trade's own record can never answer. They must never be mixed into realised
    performance: a query that forgets to filter them is measuring a trade that
    did not happen.

    ⚠️  LEAKAGE — read this before using any column here
    ----------------------------------------------------
    Every value in this table is derived from prices AFTER the signal timestamp.
    It is therefore valid as a LABEL (train a model to PREDICT how far a trade
    will run) and INVALID as a FEATURE (telling a model how far this trade ran is
    handing it the answer). A model fed its own MFE scores near-perfectly in
    backtest and is worthless live, because at signal time the column is empty.

    The one legitimate feature-side use is strictly point-in-time: aggregates over
    trades that had already CLOSED before the signal being scored — e.g. "the last
    three closed trades on this instrument had MFE below 0.5R". That is knowable
    at signal time; this trade's own path is not.

    ``tests/test_feature_contract.py`` asserts no path field ever appears in
    ``FEATURE_KEYS_MODEL``.
    """

    __tablename__ = "trade_paths"

    __table_args__ = (
        # One row per (trade, bar). Re-running the backfill updates in place
        # rather than silently doubling every path.
        Index("uq_trade_paths_trade_bar", "trade_id", "bar", unique=True),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    trade_id: Mapped[int] = mapped_column(
        ForeignKey("trades.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Signal-timeframe bars elapsed since the signal. 0 is the bar the signal
    # fired in; ``SIGNAL_MAX_HOLD_BARS`` is the time exit; anything beyond that is
    # ``beyond_exit``.
    bar: Mapped[int] = mapped_column(Integer, nullable=False)

    # All four are R-multiples relative to the trade's OWN risk unit
    # (|entry − stop|), so they are comparable across instruments and account
    # sizes — the same convention as ``rr_actual``.
    r_close: Mapped[float] = mapped_column(Float, nullable=False)   # R at this bar's close
    r_best: Mapped[float] = mapped_column(Float, nullable=False)    # best R within this bar
    r_worst: Mapped[float] = mapped_column(Float, nullable=False)   # worst R within this bar
    mfe_r: Mapped[float] = mapped_column(Float, nullable=False)     # running max through this bar
    mae_r: Mapped[float] = mapped_column(Float, nullable=False)     # running min through this bar

    # True once the trade had already exited. See "Beyond the exit" above.
    beyond_exit: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # This bar's window had less intrabar data than the density floor requires, so
    # its extremes may understate the true range. Flagged rather than dropped: a
    # silently short path looks identical to a genuinely quiet market.
    degraded: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
