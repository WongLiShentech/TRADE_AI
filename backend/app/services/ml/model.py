"""XGBoost model construction for S1 — a single classifier for the whole universe.

One model for all majors/crosses (per-cluster splitting is deferred to S2+). Every
hyperparameter is read from ``Settings`` (zero hardcoding). Class imbalance is NOT
rebalanced with ``scale_pos_weight``: the filter policy ranks trades by predicted
P(win) and picks a probability threshold to maximise expectancy (see ``policy``),
so the decision never relies on the default 0.5 cut — rebalancing would only distort
the calibrated probabilities the threshold search depends on.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from xgboost import XGBClassifier

from app.config import Settings


def build_model(settings: Settings, *, early_stopping: bool, n_estimators: Optional[int] = None) -> XGBClassifier:
    """Construct an (unfitted) XGBoost classifier from config.

    Args:
        settings: config carrying every hyperparameter (depth, trees, lr, seed,
            early-stopping patience).
        early_stopping: when True, wire ``early_stopping_rounds`` so ``fit`` uses an
            ``eval_set`` validation tail to pick the best iteration (per-fold + the
            production best-iteration search). When False, train a fixed number of
            trees with no eval_set (the final refit on ALL rows).
        n_estimators: overrides ``settings.ML_N_ESTIMATORS`` — used to pin the final
            refit to the early-stopping-selected ``best_iteration``.

    Returns:
        An unfitted ``XGBClassifier``. Deterministic given ``settings.ML_SEED``.
    """
    params = dict(
        max_depth=int(settings.ML_MAX_DEPTH),
        n_estimators=int(n_estimators if n_estimators is not None else settings.ML_N_ESTIMATORS),
        learning_rate=float(settings.ML_LEARNING_RATE),
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=int(settings.ML_SEED),
        n_jobs=-1,
    )
    if early_stopping:
        params["early_stopping_rounds"] = int(settings.ML_EARLY_STOPPING_ROUNDS)
    return XGBClassifier(**params)


def fit_with_early_stopping(
    model: XGBClassifier,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> XGBClassifier:
    """Fit ``model`` on train, using ``(x_val, y_val)`` for early stopping.

    ``x_val`` MUST be the chronological in-sample validation tail — NEVER OOS. The
    fitted model's ``best_iteration`` is then used implicitly by ``predict_proba``.
    """
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    return model


def best_iteration(model: XGBClassifier) -> int:
    """Number of boosting rounds the early-stopped model settled on (1-indexed count).

    Falls back to the configured ``n_estimators`` if early stopping did not trigger.
    """
    bi = getattr(model, "best_iteration", None)
    if bi is None:
        return int(model.n_estimators)
    return int(bi) + 1  # best_iteration is a 0-based index → +1 to get a tree count
