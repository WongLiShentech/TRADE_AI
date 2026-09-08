from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class TrainingDataset(Base):
    """The exact corpus a model was trained on — a declarative definition plus proof.

    Why this exists
    ---------------
    ``models.strategy_id`` says which strategy's trades a model learned from, but
    that names a LIVE QUERY: the rows behind it grow, get re-graded when new M1
    arrives, and get rebuilt when the backtest re-runs. Two models an hour apart can
    both claim the same corpus and have seen different data. "Did my change help, or
    did the ground move underneath me?" is then unanswerable — and that is the only
    question a retrain asks.

    Definition + fingerprint, not a membership table
    ------------------------------------------------
    The obvious design is a join table listing every trade in every dataset. It was
    rejected, and not on size grounds — 13,475 rows across two datasets is nothing.

    The failure it would protect against is a corpus being deleted (``--force`` on a
    backtest re-run). But a membership table full of dangling ids **recovers
    nothing** — it can only report the rows are gone, which the fingerprint also does,
    for one column. And the fingerprint additionally catches what membership cannot:
    a row whose stored features changed under an unchanged id, or a relabel that
    moved ``rr_actual``.

    So the definition is three typed columns (``stage``, ``strategy_id``, ``run_id``)
    and the proof is ``fingerprint``. Re-resolve the definition later: same
    fingerprint means provably identical data; different means the ground moved and
    any comparison across it is invalid.

    Why a table rather than five more columns on ``models``
    -------------------------------------------------------
    Exactly one property justifies it: ``fingerprint`` is UNIQUE, so a retrain on
    identical data REUSES the row, and two models pointing at one dataset is then a
    fact the schema states rather than a coincidence you verify by hand. That is
    champion-versus-challenger — the comparison this whole attribution layer exists
    to make possible. At two models it is the sole justification, and if a year from
    now no dataset has two models, this table was the wrong call.

    Diagnosing a mismatch
    ---------------------
    ``FEATURE_KEYS_MODEL`` is part of the digest, so adding one feature key changes
    EVERY historical fingerprint though no stored row moved. ``feature_keys_hash``
    and ``feature_schema_version`` are recorded so that case reports as "the feature
    contract changed" rather than "all of your data changed at once".

    ``label_threshold_r`` is recorded but deliberately NOT part of the fingerprint —
    it is a recipe knob (already inside the artifact's ``params_hash``), not a
    property of the data. See ``services/ml/fingerprint.py``.
    """

    __tablename__ = "datasets"

    __table_args__ = (
        # Identity is the CONTENT. Same digest ⇒ same rows and values ⇒ the same
        # dataset, whatever query happened to produce it.
        Index("uq_datasets_fingerprint", "fingerprint", unique=True),
        Index("ix_datasets_strategy_id", "strategy_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    fingerprint: Mapped[str] = mapped_column(String, nullable=False)

    # ── the declarative definition: what was asked for ────────────────────────
    stage: Mapped[str] = mapped_column(String, nullable=False)
    strategy_id: Mapped[int | None] = mapped_column(
        ForeignKey("strategies.id"), nullable=True
    )
    # Nullable because the pre-lineage corpus has no recoverable run. That is a
    # recorded unknown, not a defect to paper over with a placeholder.
    run_id: Mapped[int | None] = mapped_column(ForeignKey("backtest_runs.id"), nullable=True)

    # ── what it resolved to: readable without recomputing the digest ──────────
    n_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    n_positive: Mapped[int | None] = mapped_column(Integer, nullable=True)
    span_start: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    span_end: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # ── context for diagnosing a fingerprint mismatch ─────────────────────────
    label_threshold_r: Mapped[float | None] = mapped_column(Float, nullable=True)
    feature_schema_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    feature_keys_hash: Mapped[str | None] = mapped_column(String, nullable=True)

    description: Mapped[str | None] = mapped_column(String, nullable=True)
    resolved_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
