"""Which model serves which strategy — resolved from the registry, not from config.

Why this exists
---------------
A model's labels derive from ``rr_actual``, and ``rr_actual`` depends on the EXIT
RULE. The same entries relabelled under a pure-barrier exit instead of a trailing one
flip 12.5% of the training set. A model is therefore valid ONLY for the strategy whose
outcomes taught it, and ``models.strategy_id`` has recorded that pairing since the
registry was built.

Nothing read it. The live path loaded one champion from ``ML_MODEL_PATH`` and a
challenger list from ``ML_CHALLENGER_MODEL_PATHS``, both global, then scored EVERY
strategy with EVERY model. So a model trained on a trailing-exit corpus issued verdicts
on pure-barrier signals, and every champion-vs-challenger comparison was computed over
a signal set half of which neither model was valid for — which is not a small error,
it is the comparison measuring the wrong thing entirely.

Resolving here makes the pairing the database already states the one the runtime obeys.
Promotion becomes a status change on a row rather than an env edit plus a restart, and
``models.status`` stops being decoration.

Why this module and not ``inference``
-------------------------------------
``inference`` is deliberately free of database imports: it is the serving chokepoint and
has exactly one job, turning a validated artifact plus a feature dict into a probability.
Registry resolution is a database concern, so it lives on this side of the line and calls
into ``inference`` for the loading and validating it already does.

Failure policy
--------------
Per-model and never propagating, the same contract ``load_challengers`` established. A
model whose artifact is missing or whose contract no longer validates is logged and
skipped; the strategy still evaluates, still records its row, and every other model still
scores. A strategy with no champion records a rule-only row (``ml_*`` NULL) rather than
recording nothing — the signal is evidence whether or not a model had an opinion on it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.models.ml_model import MLModel
from app.services.ml import inference as inf

logger = logging.getLogger(__name__)

# Lifecycle values that put a model on the live scoring path. `candidate` (trained, not
# yet observed) and `retired` (superseded) deliberately do NOT load: becoming live is an
# explicit status change, never a side effect of existing in the table.
STATUS_CHAMPION = "champion"
STATUS_SHADOW = "shadow"

# backend/models/ — `model_id` IS the artifact filename stem (see inference._model_id_for),
# so the path is derived rather than stored. A second column holding the path could
# disagree with the id, and the id is the one every decision row already carries.
_MODELS_DIRNAME = "models"


@dataclass(frozen=True)
class RegisteredModel:
    """A loaded artifact plus the cut-off the registry says it decides at.

    The threshold travels WITH the model because it is a property of this model's
    calibration, not a global setting: v2 cuts at 0.236 and v3 at 0.343, both selected
    on their own validation tails. Judging them through one shared number would measure
    the threshold rather than the models.
    """

    loaded: inf.LoadedModel
    threshold: float
    is_champion: bool

    @property
    def model_id(self) -> str:
        return self.loaded.model_id


@dataclass(frozen=True)
class StrategyModels:
    """The models registered to score one strategy's signals."""

    champion: Optional[RegisteredModel]
    challengers: tuple[RegisteredModel, ...]

    @property
    def all(self) -> tuple[RegisteredModel, ...]:
        """Champion first, then challengers — the order verdicts are recorded in."""
        return ((self.champion,) if self.champion is not None else ()) + self.challengers


def artifact_path(model_id: str) -> Path:
    """Absolute path to an artifact, derived from its registry id."""
    return inf._BACKEND_ROOT / _MODELS_DIRNAME / f"{model_id}{inf._MODEL_SUFFIX}"


def _load(row: MLModel, settings: Settings, *, is_champion: bool) -> Optional[RegisteredModel]:
    """Load one registered model, or None if it cannot be served.

    The artifact's own ``deployment_threshold`` is the fallback when the registry row
    carries no explicit cut-off, so a row backfilled without one still decides at the
    value the model was shipped with rather than at some global default.
    """
    path = artifact_path(row.model_id)
    try:
        loaded = inf.load_artifact(path, settings)
    except Exception as exc:  # noqa: BLE001 — see module docstring: never propagate
        logger.error(
            "model %s (strategy %s, status=%s) failed to load (%s) — continuing without it",
            row.model_id, row.strategy_id, row.status, exc,
        )
        return None

    threshold = row.decision_threshold
    if threshold is None:
        threshold = loaded.metadata.get("deployment_threshold")
    if threshold is None:
        logger.error(
            "model %s has no decision_threshold in the registry and none in its artifact "
            "metadata — refusing to serve it rather than inventing a cut-off",
            row.model_id,
        )
        return None
    return RegisteredModel(loaded=loaded, threshold=float(threshold), is_champion=is_champion)


def models_for_strategy(db: Session, settings: Settings, strategy_id: int) -> StrategyModels:
    """Champion and challengers registered to score ``strategy_id``.

    Args:
        db: session (read-only here).
        settings: config — supplies the promotion/order safety guards enforced on load.
        strategy_id: the strategy whose signals are about to be scored.

    Returns:
        A :class:`StrategyModels`. ``champion`` is None when the strategy has no model
        registered as champion, which is a legitimate state: a newly registered strategy
        collects rule-only rows until its first model is trained on its own corpus.
    """
    rows = db.execute(
        select(MLModel)
        .where(MLModel.strategy_id == strategy_id)
        .where(MLModel.status.in_((STATUS_CHAMPION, STATUS_SHADOW)))
        .order_by(MLModel.version.asc().nulls_last(), MLModel.id.asc())
    ).scalars().all()

    champion_rows = [r for r in rows if r.status == STATUS_CHAMPION]
    if len(champion_rows) > 1:
        # Enforced by test; logged here because the live path must not silently pick one.
        logger.error(
            "strategy %s has %d models with status='champion' (%s) — exactly one may "
            "govern. Scoring with NONE of them until this is resolved, so no arbitrary "
            "choice is baked into the evidence.",
            strategy_id, len(champion_rows), ", ".join(r.model_id for r in champion_rows),
        )
        champion_rows = []

    champion = _load(champion_rows[0], settings, is_champion=True) if champion_rows else None
    challengers = tuple(
        m for m in (
            _load(r, settings, is_champion=False)
            for r in rows if r.status == STATUS_SHADOW
        ) if m is not None
    )

    if champion is None and not challengers:
        logger.warning(
            "strategy %s has no champion and no challengers registered — its signals will "
            "be recorded rule-only (ml_* NULL). This is expected for a strategy whose "
            "first model has not been trained yet.",
            strategy_id,
        )
    return StrategyModels(champion=champion, challengers=challengers)
