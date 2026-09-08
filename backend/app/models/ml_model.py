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


class MLModel(Base):
    """Registry of trained model artifacts — what exists, and what it learned from.

    Why this exists
    ---------------
    A model is a FILE (``backend/models/*.joblib``); the database refers to it by
    a bare string in ``trades.ml_model_id`` and ``model_decisions.model_id``.
    Nothing records what that string MEANS. Delete the artifact and the id becomes
    an unresolvable label; keep it and you still cannot answer "how many models
    have we trained, on what span, on whose trades?" without opening JSON files
    on disk one at a time.

    The critical column is ``strategy_id``
    --------------------------------------
    A model's labels are derived from ``rr_actual``, and ``rr_actual`` depends on
    the EXIT RULE. The same 5,472 entries relabelled under a pure-barrier exit
    instead of a trailing one flip 12.5% of the training set (605 losses become
    wins). A model is therefore only valid for the strategy whose outcomes taught
    it — pairing it with another is a silent mismatch that nothing would flag.
    This column is what makes that pairing checkable.

    Deliberately NOT stored here
    ----------------------------
    Hyperparameters, SHAP importances, feature lists, library versions, fold
    metrics. They live in the artifact's ``.metadata.json`` and are not things
    you eyeball in a table; ``model_id`` says which file to open for that depth.
    This table answers "what have we got?", not "how exactly was it built?".

    Status lifecycle
    ----------------
    ``candidate``  Trained, not yet evaluated on live bars.
    ``shadow``     Scoring live signals; records decisions, places no orders.
    ``champion``   The authoritative model for its strategy.
    ``retired``    Superseded; kept for history.

    ``passed_gate`` is the walk-forward promotion verdict and is NULLABLE on
    purpose: v1 predates the gate, and "unknown" is the honest value — recording
    it as ``false`` would invent a judgement that was never made.
    """

    __tablename__ = "models"

    __table_args__ = (
        Index("ix_models_strategy_id", "strategy_id"),
        Index("ix_models_status", "status"),
        # Without this, two rows can both claim "version 2 of strategy 1" and the
        # version number means nothing. Partial, because a model whose version was
        # never assigned is recorded as NULL rather than forced into the sequence.
        Index(
            "uq_models_strategy_version",
            "strategy_id",
            "version",
            unique=True,
            postgresql_where=text("version IS NOT NULL"),
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    # The FULL artifact stem, `.NOT_PROMOTED` included where present — the same string
    # `inference._model_id_for` produces and the recorder writes to
    # `trades.ml_model_id`. Stripping the marker breaks the join to every decision
    # this row describes.
    model_id: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    # Sequential model version — 1, 2, 3 — and NOT the feature schema version, which
    # the old `s1_xgb_v2_...` filenames actually encoded. They coincided by accident
    # (model 1 used schema 1, model 2 used schema 2) and would have diverged at
    # model 3, which uses schema 2. Assigned by a human, enforced unique per strategy.
    version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    strategy_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("strategies.id"), nullable=True
    )
    trained_on_stages: Mapped[str] = mapped_column(String, nullable=False)
    training_start: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    training_end: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    training_rows: Mapped[int | None] = mapped_column(Integer, nullable=True)
    decision_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    passed_gate: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False)

    # ── provenance: which data, which code ────────────────────────────────────
    # The exact corpus, by content rather than by query. `trained_on_stages` and
    # `strategy_id` say what was ASKED for; this says what was RECEIVED.
    dataset_id: Mapped[int | None] = mapped_column(
        ForeignKey("datasets.id"), nullable=True, index=True
    )
    # NULL means "not recorded", which is the honest value for v1 and v2 — both were
    # trained before provenance capture existed and it cannot be reconstructed.
    git_commit: Mapped[str | None] = mapped_column(String, nullable=True)
    git_dirty: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    # ── multiple-testing correction ───────────────────────────────────────────
    # The trial count this model's results were discounted by (deflated Sharpe).
    n_trials: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # WHERE that number came from. Mandatory context, not decoration: the counter is
    # currently derived from `backtest_runs_id_seq` GLOBALLY, so it differs by host
    # (local reads 7, the server reads 1) and counts looks at other strategies' data.
    # A bare integer would be uninterpretable six months from now.
    n_trials_source: Mapped[str | None] = mapped_column(String, nullable=True)

    description: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
