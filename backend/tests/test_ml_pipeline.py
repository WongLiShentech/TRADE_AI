"""Unit tests for app.services.ml.pipeline — the one-hot encoder + name mapping.

Verifies LEAK-6 at the encoder boundary: fitting on a training slice that lacks a
category, then transforming a row carrying that unseen category, must NOT raise and
must encode the unseen category to all-zeros (``handle_unknown='ignore'``) — i.e.
the encoder never learns from data outside its fit slice.

Run: python -m pytest tests/test_ml_pipeline.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.ml.dataset import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from app.services.ml.pipeline import build_encoder, parent_feature


def _frame(categories: list[tuple[str, str]]) -> pd.DataFrame:
    """Build a minimal 20-column feature frame; numerics filled with a constant."""
    n = len(categories)
    data = {k: np.full(n, 1.0) for k in NUMERIC_FEATURES}
    data["instrument_category"] = [c[0] for c in categories]
    data["session"] = [c[1] for c in categories]
    return pd.DataFrame(data)


def test_encoder_fit_on_train_ignores_unseen_category_in_transform():
    train = _frame([("major_usd", "london"), ("major_usd", "ny")])
    enc = build_encoder()
    enc.fit(train)  # 'jpy_cross' + 'asian' never seen at fit time

    val = _frame([("jpy_cross", "asian")])  # both categoricals unseen
    out = enc.transform(val)  # must not raise
    names = list(enc.get_feature_names_out())

    # Every one-hot column for the categoricals is 0 for the unseen-category row.
    cat_cols = [i for i, nm in enumerate(names) if parent_feature(nm) in CATEGORICAL_FEATURES]
    assert all(out[0, i] == 0.0 for i in cat_cols)


def test_encoder_preserves_numeric_nan_for_xgboost():
    train = _frame([("major_usd", "london"), ("eur_cross", "asian")])
    train.loc[0, "vix"] = np.nan
    enc = build_encoder()
    enc.fit(train)
    out = enc.transform(train)
    names = list(enc.get_feature_names_out())
    vix_col = names.index("vix")
    assert np.isnan(out[0, vix_col])  # NaN passes through, not imputed


def test_parent_feature_maps_onehot_back_to_model_feature():
    assert parent_feature("session_london") == "session"
    assert parent_feature("instrument_category_major_usd") == "instrument_category"
    assert parent_feature("vix") == "vix"          # numeric passthrough unchanged
    assert parent_feature("rsi14") == "rsi14"
