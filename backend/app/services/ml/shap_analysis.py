"""SHAP global feature importance for S1 — computed on TRAINING data only.

SHAP is run over the same rows the model was trained on (never a held-out or OOS
set) — the goal here is model INTERPRETATION (which of the 20 locked features the
tree actually uses, and which contribute ~nothing), not another performance
estimate. One-hot columns of a categorical are aggregated back onto the single
parent feature so the ranking is over the 20 model features, not the expanded
encoding.

Reference: Lundberg, S. & Lee, S.-I. (2017), "A Unified Approach to Interpreting
Model Predictions", NeurIPS. TreeSHAP: Lundberg et al. (2020), Nature Machine
Intelligence 2, 56-67.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shap
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

from app.services.feature_builder import FEATURE_KEYS_MODEL
from app.services.ml.pipeline import parent_feature

# A model feature whose aggregated mean|SHAP| is below this is treated as a
# ~zero-contribution ("dead") feature for reporting.
_DEAD_EPS = 1e-6


@dataclass(frozen=True)
class ShapImportances:
    """Global mean|SHAP| per model feature, plus the dead-feature list."""

    importances: dict[str, float]      # model feature → mean|SHAP| (descending)
    top10: list[tuple[str, float]]
    dead_features: list[str]           # aggregated mean|SHAP| < _DEAD_EPS


def global_importances(
    model: XGBClassifier,
    encoder: ColumnTransformer,
    x_train_df,
) -> ShapImportances:
    """Compute global mean|SHAP| over the training rows, aggregated to model features.

    Args:
        model: the fitted XGBoost classifier.
        encoder: the FITTED encoder used to train ``model`` (same fold/production fit).
        x_train_df: the training rows as a DataFrame with the 20 model columns.

    Returns:
        A :class:`ShapImportances` ranked over the 20 model features.
    """
    x_encoded = encoder.transform(x_train_df)
    encoded_names = list(encoder.get_feature_names_out())

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_encoded)
    if isinstance(shap_values, list):  # some shap versions return per-class lists
        shap_values = shap_values[-1]
    mean_abs = np.abs(np.asarray(shap_values)).mean(axis=0)

    agg: dict[str, float] = {k: 0.0 for k in FEATURE_KEYS_MODEL}
    for name, val in zip(encoded_names, mean_abs):
        agg[parent_feature(name)] += float(val)

    ranked = dict(sorted(agg.items(), key=lambda kv: kv[1], reverse=True))
    dead = [k for k, v in ranked.items() if v < _DEAD_EPS]
    return ShapImportances(
        importances=ranked,
        top10=list(ranked.items())[:10],
        dead_features=dead,
    )
