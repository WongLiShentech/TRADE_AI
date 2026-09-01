"""sklearn preprocessing for S1 — one-hot the two categoricals, pass numerics through.

The encoder is a ``ColumnTransformer`` so all preprocessing lives inside a single
sklearn object (no standalone scalers). Numeric features are passed through
UNCHANGED — no scaling, no imputation — because XGBoost consumes NaN natively and
routes it, and the whole point of the locked contract is that a missing macro
series stays missing rather than being invented.

LEAK-6 (fit-on-train-only) is enforced by the CALLER: every ``.fit`` of the object
returned here is called on the fold's training slice only, never on OOS (see
``evaluate`` and ``artifact``). ``handle_unknown='ignore'`` means a category present
in validation/OOS but absent from the training slice encodes to all-zeros rather
than raising — so an unseen instrument/session at inference degrades gracefully.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from app.services.ml.dataset import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_encoder() -> ColumnTransformer:
    """Build the (unfitted) feature encoder.

    Returns:
        A ``ColumnTransformer`` that one-hot-encodes ``CATEGORICAL_FEATURES``
        (``handle_unknown='ignore'``) and passes ``NUMERIC_FEATURES`` through
        untouched (NaN preserved). Output feature names are un-prefixed
        (``verbose_feature_names_out=False``) so SHAP can map one-hot columns back
        to their parent categorical.
    """
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_FEATURES),
            ),
            ("num", "passthrough", list(NUMERIC_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def parent_feature(encoded_name: str) -> str:
    """Map an encoded output column name back to its parent model feature.

    One-hot columns look like ``"session_london"`` → ``"session"``; passthrough
    numeric columns keep their exact key. Used to aggregate SHAP contributions of
    a categorical's one-hot columns back onto the single model feature.
    """
    for cat in CATEGORICAL_FEATURES:
        if encoded_name == cat or encoded_name.startswith(f"{cat}_"):
            return cat
    return encoded_name
