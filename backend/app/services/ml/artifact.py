"""Production-candidate training + serialization for S1.

After the walk-forward EVALUATION establishes whether the ML-filtered strategy
beats the gate, the deployable candidate is trained on ALL rows with the SAME
recipe and fixed seed, then serialized as a single sklearn ``Pipeline``
(encoder + model) plus a metadata JSON and a model card.

Threshold provenance: the deployment threshold is selected on the chronological
validation TAIL of the full corpus (the same ``policy`` used per-fold), using an
early-stopping model trained on the earlier portion. That early-stopping run also
fixes ``best_iteration``; the final model is then refit on ALL rows for
``best_iteration`` trees (no early stopping). This keeps the artifact trained on
the entire corpus while pinning a tree count that was chosen out-of-training-tail.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import sklearn
import xgboost
from joblib import dump
from sklearn.pipeline import Pipeline

from app.config import Settings
from app.services.feature_builder import FEATURE_KEYS_MODEL
from app.services.ml.constants import NOT_PROMOTED_MARKER
from app.services.ml.dataset import CATEGORICAL_FEATURES, Dataset, NUMERIC_FEATURES
from app.services.ml.model import best_iteration, build_model, fit_with_early_stopping
from app.services.ml.pipeline import build_encoder
from app.services.ml.policy import select_threshold
from app.services.ml.shap_analysis import ShapImportances, global_importances


@dataclass
class ProductionCandidate:
    """The trained deployable pipeline + its metadata + SHAP interpretation."""

    pipeline: Pipeline
    metadata: dict
    shap: ShapImportances


def train_production_model(dataset: Dataset, settings: Settings) -> ProductionCandidate:
    """Train the deployable candidate on ALL corpus rows (fixed seed, single recipe)."""
    x = dataset.X
    y = dataset.y
    rr = dataset.rr
    n = len(dataset)

    n_val = int(round(n * settings.ML_VALIDATION_FRACTION))
    train_idx = list(range(0, n - n_val))
    val_idx = list(range(n - n_val, n))

    # 1) Early-stopping model on the earlier portion → best_iteration + threshold.
    enc_es = build_encoder()
    enc_es.fit(x.iloc[train_idx])
    x_tr = enc_es.transform(x.iloc[train_idx])
    x_val = enc_es.transform(x.iloc[val_idx])
    es_model = build_model(settings, early_stopping=True)
    fit_with_early_stopping(es_model, x_tr, y[train_idx], x_val, y[val_idx])
    n_trees = best_iteration(es_model)
    p_val = es_model.predict_proba(x_val)[:, 1]
    choice = select_threshold(p_val, rr[val_idx], settings.ML_MIN_KEEP_FRACTION)

    # 2) Final refit on ALL rows for the pinned tree count (no early stopping).
    enc_all = build_encoder()
    enc_all.fit(x)
    x_all = enc_all.transform(x)
    final = build_model(settings, early_stopping=False, n_estimators=n_trees)
    final.fit(x_all, y)

    pipeline = Pipeline([("encoder", enc_all), ("model", final)])
    shap_imp = global_importances(final, enc_all, x)

    metadata = _build_metadata(dataset, settings, enc_all, n_trees, choice.threshold, shap_imp)
    return ProductionCandidate(pipeline=pipeline, metadata=metadata, shap=shap_imp)


# Re-exported from the dependency-free ``ml.constants`` module. The definition
# moved there so the SERVING path (``ml/inference.py``) can read the marker without
# importing this module — which would pull `shap` (+96 MB RSS) into the always-on
# process for a single string. (QA S1-v1 fix (a); moved for the M9 RAM audit.)
_NOT_PROMOTED_MARKER = NOT_PROMOTED_MARKER


def save_artifact(
    candidate: ProductionCandidate, models_dir: Path, *, promoted: bool
) -> dict:
    """Serialize the pipeline + metadata + model card into ``models_dir``.

    Args:
        candidate: the trained production candidate.
        models_dir: destination directory (created if absent).
        promoted: the walk-forward gate verdict for the ML-filtered strategy. When
            ``False`` the artifact is tagged DO-NOT-PROMOTE in BOTH the filename
            (``.NOT_PROMOTED`` infix) and the metadata (``promoted: false``,
            ``gate_verdict: "DO_NOT_PROMOTE"``) so it can never be silently deployed.

    Returns:
        A dict of the written paths (all absolute), plus ``promoted``.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    meta = candidate.metadata
    # Stamp the verdict into the metadata (does NOT affect params_hash — promotion
    # status is not part of the reproducibility identity of the model).
    meta["promoted"] = bool(promoted)
    meta["gate_verdict"] = "PROMOTE" if promoted else "DO_NOT_PROMOTE"

    base = f"s1_xgb_v{meta['feature_schema_version']}_{meta['params_hash']}"
    stem = base if promoted else f"{base}.{_NOT_PROMOTED_MARKER}"

    model_path = models_dir / f"{stem}.joblib"
    meta_path = models_dir / f"{stem}.metadata.json"
    card_path = models_dir / f"{stem}.model_card.md"

    dump(candidate.pipeline, model_path)
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    card_path.write_text(_render_model_card(candidate), encoding="utf-8")

    return {
        "model": str(model_path.resolve()),
        "metadata": str(meta_path.resolve()),
        "model_card": str(card_path.resolve()),
        "promoted": bool(promoted),
    }


# ── helpers ───────────────────────────────────────────────────────────────────
def _build_metadata(
    dataset: Dataset,
    settings: Settings,
    encoder,
    n_trees: int,
    threshold: float,
    shap_imp: ShapImportances,
) -> dict:
    # Lazy: the ONLY thing this module wants from `shap` is its version string for
    # provenance. Kept out of module scope so importing `ml.artifact` (e.g. from a
    # training script) is the only thing that ever pays for it — see ml/constants.py.
    import shap

    hyperparams = {
        "max_depth": int(settings.ML_MAX_DEPTH),
        "n_estimators_cap": int(settings.ML_N_ESTIMATORS),
        "learning_rate": float(settings.ML_LEARNING_RATE),
        "early_stopping_rounds": int(settings.ML_EARLY_STOPPING_ROUNDS),
        "validation_fraction": float(settings.ML_VALIDATION_FRACTION),
        "min_keep_fraction": float(settings.ML_MIN_KEEP_FRACTION),
        "seed": int(settings.ML_SEED),
    }
    # Fitted one-hot categories (per categorical feature) for serving-time parity.
    ohe = encoder.named_transformers_["cat"]
    encoder_categories = {
        feat: list(cats) for feat, cats in zip(CATEGORICAL_FEATURES, ohe.categories_)
    }
    st = dataset.frame["signal_time"]
    params_for_hash = {
        "label_threshold_r": float(dataset.label_threshold_r),
        "feature_keys_model": list(FEATURE_KEYS_MODEL),
        "feature_schema_version": int(dataset.feature_schema_version),
        "hyperparameters": hyperparams,
        "best_iteration": int(n_trees),
    }
    params_hash = hashlib.sha1(
        json.dumps(params_for_hash, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]

    return {
        "model_type": "xgboost.XGBClassifier (sklearn Pipeline: encoder+model)",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_schema_version": int(dataset.feature_schema_version),
        "feature_keys_model": list(FEATURE_KEYS_MODEL),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "numeric_features": list(NUMERIC_FEATURES),
        "encoder_categories": encoder_categories,
        "label_rule": f"y = 1 if rr_actual >= {dataset.label_threshold_r}R else 0",
        "label_threshold_r": float(dataset.label_threshold_r),
        "deployment_threshold": float(threshold),
        "deployment_threshold_provenance": (
            "selected on the chronological validation tail of the full corpus "
            "(ML_VALIDATION_FRACTION) via the expectancy-maximising policy — never OOS"
        ),
        "best_iteration": int(n_trees),
        "hyperparameters": hyperparams,
        "training_span": {
            "start": st.min().isoformat() if len(st) else None,
            "end": st.max().isoformat() if len(st) else None,
            "n_rows": int(len(dataset)),
        },
        "params_hash": params_hash,
        "shap_top10": [[k, v] for k, v in shap_imp.top10],
        "shap_dead_features": shap_imp.dead_features,
        "versions": {
            "python": platform.python_version(),
            "xgboost": xgboost.__version__,
            "scikit_learn": sklearn.__version__,
            "shap": shap.__version__,
        },
    }


def _render_model_card(candidate: ProductionCandidate) -> str:
    m = candidate.metadata
    top10 = "\n".join(f"| {k} | {v:.6f} |" for k, v in m["shap_top10"])
    dead = ", ".join(m["shap_dead_features"]) or "(none)"
    hp = m["hyperparameters"]
    verdict = m.get("gate_verdict", "UNKNOWN")
    promoted = m.get("promoted")
    banner = (
        "**GATE VERDICT: DO-NOT-PROMOTE** — this artifact FAILED the walk-forward "
        "promotion gate. It is serialized for post-mortem/SHAP inspection ONLY and "
        "must not be deployed. Filename carries the `NOT_PROMOTED` marker.\n\n"
        if promoted is False
        else f"**GATE VERDICT: {verdict}** — passed the walk-forward promotion gate.\n\n"
        if promoted is True
        else ""
    )
    return f"""# Model Card — S1 XGBoost signal filter (`{m['params_hash']}`)

Generated: {m['created_at']}

{banner}## Purpose
Single XGBoost classifier that FILTERS rule-engine trade signals. It outputs
`P(win)` per signal; the deployed strategy takes a trade only when
`P(win) >= {m['deployment_threshold']:.4f}`. Signals, not sizes — position sizing
stays in RiskEngine.

## Training data
- Corpus: `trades.stage='backtest'` (M7 rule-engine simulation).
- Rows: {m['training_span']['n_rows']}
- Span: {m['training_span']['start']} → {m['training_span']['end']}
- Feature schema version: {m['feature_schema_version']}

## Label
`{m['label_rule']}` (derived at training time, not stored).

## Features ({len(m['feature_keys_model'])} locked model keys)
Categorical (one-hot, handle_unknown=ignore): {', '.join(m['categorical_features'])}
Numeric (NaN-native, no scaling/imputation): {len(m['numeric_features'])} keys.
Feature list is imported from `feature_builder.FEATURE_KEYS_MODEL` — never hardcoded.

## Hyperparameters
- max_depth: {hp['max_depth']}
- learning_rate: {hp['learning_rate']}
- n_estimators cap: {hp['n_estimators_cap']}; best_iteration (final trees): {m['best_iteration']}
- early_stopping_rounds: {hp['early_stopping_rounds']}
- validation_fraction: {hp['validation_fraction']}
- min_keep_fraction: {hp['min_keep_fraction']}
- seed: {hp['seed']}

## Deployment threshold
{m['deployment_threshold']:.6f} — {m['deployment_threshold_provenance']}.

## SHAP top-10 (mean|SHAP| on training data, aggregated to model features)
| feature | mean_abs_shap |
|---|---|
{top10}

Dead features (~zero contribution): {dead}

## Leakage controls
- Encoder + model fit on the training slice only within each fold (LEAK-6).
- Filter threshold selected on the in-sample validation tail only — never OOS.
- Walk-forward folds mirror M7 (expanding IS, embargoed OOS).
- Numeric NaNs pass through to XGBoost (no full-dataset normalization).

## Versions
python {m['versions']['python']}, xgboost {m['versions']['xgboost']},
scikit-learn {m['versions']['scikit_learn']}, shap {m['versions']['shap']}.

## Reproducibility
params_hash: `{m['params_hash']}` (sha1 of label rule + feature list + schema +
hyperparameters + best_iteration).
"""
