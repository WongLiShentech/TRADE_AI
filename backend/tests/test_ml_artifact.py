"""Unit tests for app.services.ml.artifact — promotion-verdict tagging (QA fix a).

The S1-v1 QA report flagged that a candidate whose walk-forward gate verdict is
DO-NOT-PROMOTE was serialized under the SAME filename convention as a promotable
one — so nothing structurally prevented a failed model from being deployed. The
fix tags such artifacts in BOTH the filename (``.NOT_PROMOTED`` infix) and the
metadata (``promoted: false`` / ``gate_verdict: "DO_NOT_PROMOTE"``), while still
serializing them so the model card + SHAP remain available for post-mortem.

These tests build a minimal :class:`ProductionCandidate` (no training) and assert
the tagging on both branches. No DB, no model fit.

Run: python -m pytest tests/test_ml_artifact.py -v
"""
from __future__ import annotations

import json

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from app.services.ml.artifact import (
    _NOT_PROMOTED_MARKER,
    ProductionCandidate,
    save_artifact,
)
from app.services.ml.shap_analysis import ShapImportances


def _minimal_candidate() -> ProductionCandidate:
    """A serializable candidate with a metadata dict carrying every card key."""
    pipeline = Pipeline([("noop", FunctionTransformer())])  # picklable, trivial
    shap_imp = ShapImportances(
        importances={"vix": 0.5, "rsi14": 0.0},
        top10=[("vix", 0.5), ("rsi14", 0.0)],
        dead_features=["rsi14"],
    )
    metadata = {
        "params_hash": "abc123def456",
        "created_at": "2026-07-12T00:00:00+00:00",
        "feature_schema_version": 2,
        "feature_keys_model": ["vix", "rsi14"],
        "categorical_features": ["instrument_category", "session"],
        "numeric_features": ["vix", "rsi14"],
        "label_rule": "y = 1 if rr_actual >= 1.0R else 0",
        "deployment_threshold": 0.55,
        "deployment_threshold_provenance": "validation tail",
        "best_iteration": 42,
        "hyperparameters": {
            "max_depth": 4,
            "learning_rate": 0.05,
            "n_estimators_cap": 400,
            "early_stopping_rounds": 50,
            "validation_fraction": 0.2,
            "min_keep_fraction": 0.2,
            "seed": 42,
        },
        "training_span": {"start": "2021-06-01", "end": "2026-06-03", "n_rows": 7074},
        "shap_top10": [["vix", 0.5], ["rsi14", 0.0]],
        "shap_dead_features": ["rsi14"],
        "versions": {
            "python": "3.11.0",
            "xgboost": "2.0.0",
            "scikit_learn": "1.4.0",
            "shap": "0.44.0",
        },
    }
    return ProductionCandidate(pipeline=pipeline, metadata=metadata, shap=shap_imp)


def test_not_promoted_artifact_is_tagged_in_filename_and_metadata(tmp_path):
    candidate = _minimal_candidate()
    paths = save_artifact(candidate, tmp_path, promoted=False)

    assert paths["promoted"] is False
    # Filename marker on every written path.
    for key in ("model", "metadata", "model_card"):
        assert _NOT_PROMOTED_MARKER in paths[key], f"{key} missing NOT_PROMOTED marker"

    meta = json.loads((tmp_path / f"s1_xgb_v2_abc123def456.{_NOT_PROMOTED_MARKER}.metadata.json").read_text())
    assert meta["promoted"] is False
    assert meta["gate_verdict"] == "DO_NOT_PROMOTE"

    card = (tmp_path / f"s1_xgb_v2_abc123def456.{_NOT_PROMOTED_MARKER}.model_card.md").read_text()
    assert "DO-NOT-PROMOTE" in card


def test_promoted_artifact_has_clean_filename_and_flag(tmp_path):
    candidate = _minimal_candidate()
    paths = save_artifact(candidate, tmp_path, promoted=True)

    assert paths["promoted"] is True
    for key in ("model", "metadata", "model_card"):
        assert _NOT_PROMOTED_MARKER not in paths[key]

    meta = json.loads((tmp_path / "s1_xgb_v2_abc123def456.metadata.json").read_text())
    assert meta["promoted"] is True
    assert meta["gate_verdict"] == "PROMOTE"


def test_promotion_status_does_not_change_params_hash(tmp_path):
    """The reproducibility hash is identity of the recipe — promotion is orthogonal."""
    a = save_artifact(_minimal_candidate(), tmp_path / "a", promoted=True)
    b = save_artifact(_minimal_candidate(), tmp_path / "b", promoted=False)
    # Same params_hash stem despite different verdicts (only the marker differs).
    assert "abc123def456" in a["model"] and "abc123def456" in b["model"]
