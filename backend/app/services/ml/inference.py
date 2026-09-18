"""Serving-side inference for the S1 signal filter — M8-Shadow Phase 1.

This module is the ONLY place the platform turns a live ``feature_builder``
feature dict into a model probability and a take/skip decision. It loads the
serialized sklearn ``Pipeline`` (encoder + XGBoost) written by
:mod:`app.services.ml.artifact` together with its metadata JSON, and refuses to
serve anything it cannot prove is contract-compatible.

Two firewalls, both fail-loud on load
-------------------------------------
1. **Training/serving skew.** The artifact's ``feature_keys_model`` must match the
   live ``feature_builder.FEATURE_KEYS_MODEL`` exactly (same set, same length) and
   its ``feature_schema_version`` must equal the live ``FEATURE_SCHEMA_VERSION``.
   A silently mismatched feature vector still produces a plausible-looking float —
   it is the single worst failure mode in a serving path, so it raises here rather
   than degrading quietly.
2. **Unpromoted-artifact guard.** An artifact tagged DO-NOT-PROMOTE (``promoted:
   false`` in metadata and/or the ``.NOT_PROMOTED`` filename marker) loads ONLY if
   ``settings.ML_ALLOW_UNPROMOTED_MODEL`` is true AND
   ``settings.ORDER_PLACEMENT_ENABLED`` is false. That pair is what makes shadow-
   observing a rejected candidate safe by construction: it can be scored and
   logged, but the process it runs in is structurally forbidden from trading.

Scoring is stateless and deterministic: XGBoost consumes NaN natively, so a
missing macro series stays missing — no imputation, no scaling, no fallback value
is ever invented at serving time (identical to training, see ``ml/pipeline.py``).

This module outputs a probability and a decision. It NEVER sizes a position and
NEVER touches a broker — position size stays in RiskEngine, order placement stays
behind ``ORDER_PLACEMENT_ENABLED``.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from joblib import load
from sklearn.pipeline import Pipeline

from app.config import Settings
from app.services.feature_builder import FEATURE_KEYS_MODEL, FEATURE_SCHEMA_VERSION
# NOT from ml.artifact: that module imports shap_analysis → shap, which costs the
# always-on serving process ~96 MB RSS for a single string constant. ml.constants is
# stdlib-only by design — keep it that way.
from app.services.ml.constants import NOT_PROMOTED_MARKER as _NOT_PROMOTED_MARKER
from app.services.ml.dataset import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    _CAT_MISSING,
    _to_number,
)

logger = logging.getLogger(__name__)

# The two decision values written to ``trades.ml_decision``. Named constants so no
# caller ever spells them as inline literals.
DECISION_TAKE = "take"
DECISION_SKIP = "skip"

# backend/ — relative ML_MODEL_PATH values resolve against this.
# inference.py → ml → services → app → backend
_BACKEND_ROOT = Path(__file__).resolve().parents[3]

_MODEL_SUFFIX = ".joblib"
_METADATA_SUFFIX = ".metadata.json"


class FeatureContractMismatch(RuntimeError):
    """The artifact is intact but describes a different feature contract than the live one.

    Distinguished from every other load failure because it means something operationally
    different. A missing file, an unreadable pickle or a tripped promotion guard is a
    FAULT — something is broken and a human should look now. A contract mismatch is the
    EXPECTED, temporary state during a deliberate feature-schema migration: the artifacts
    on disk predate the bump and the first model trained against the new contract does
    not exist yet.

    Both are correctly refused at load time. Only one of them should page anybody, so the
    health check reports this as degraded rather than unhealthy — see ``main._check_ml_model``.
    Raised rather than string-matched so the distinction cannot rot as messages are edited.
    """

# Process-level cache: the artifact is immutable on disk, so it is deserialized
# once per process. Keyed by resolved path (a config change pointing at a
# different artifact loads that artifact). Safety guards are re-checked on EVERY
# call, cache hit or not — see load_model.
_CACHE: dict[str, "LoadedModel"] = {}


@dataclass(frozen=True)
class LoadedModel:
    """A deserialized, contract-validated S1 artifact ready to score.

    Attributes:
        model_id: the artifact filename stem — the exact string persisted to
            ``trades.ml_model_id`` for provenance (e.g.
            ``"s1_xgb_v2_c430d22dd1ab.NOT_PROMOTED"``). Carries the feature schema
            version, the reproducibility ``params_hash`` and the promotion marker.
        pipeline: the sklearn ``Pipeline`` (fitted encoder + XGBoost model).
        metadata: the full metadata JSON dict written alongside the model.
        model_path: absolute path to the ``.joblib``.
        metadata_path: absolute path to the ``.metadata.json``.
        feature_order: model feature keys in the artifact's own declared order.
        numeric_features: artifact's numeric (NaN-native passthrough) keys.
        categorical_features: artifact's one-hot encoded keys.
        promoted: False when the walk-forward gate verdict was DO-NOT-PROMOTE.
    """

    model_id: str
    pipeline: Pipeline
    metadata: dict
    model_path: Path
    metadata_path: Path
    feature_order: tuple[str, ...]
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    promoted: bool

    @property
    def params_hash(self) -> str:
        """The artifact's reproducibility hash (recipe identity)."""
        return str(self.metadata.get("params_hash", ""))

    @property
    def deployment_threshold(self) -> float:
        """The threshold the artifact itself was shipped with (audit reference).

        The threshold actually APPLIED at inference is ``settings.ML_DECISION_THRESHOLD`` —
        config drives behaviour, never the file. A divergence is logged on load.
        """
        return float(self.metadata["deployment_threshold"])


@dataclass(frozen=True)
class ShadowDecision:
    """One scored signal: what the model said and which artifact said it.

    Maps 1:1 onto the ``trades`` shadow columns (``ml_probability``,
    ``ml_decision``, ``ml_model_id``).
    """

    probability: float
    decision: str
    model_id: str


# ── public API ───────────────────────────────────────────────────────────────
def load_model(settings: Settings, *, force_reload: bool = False) -> LoadedModel:
    """Load (once per process) and validate the artifact at ``settings.ML_MODEL_PATH``.

    Args:
        settings: config supplying ``ML_MODEL_PATH``, ``ML_ALLOW_UNPROMOTED_MODEL``
            and ``ORDER_PLACEMENT_ENABLED``.
        force_reload: bypass the process cache and re-read from disk.

    Returns:
        The cached :class:`LoadedModel`.

    Raises:
        FileNotFoundError: the ``.joblib`` or its ``.metadata.json`` is missing.
        RuntimeError: the artifact's feature contract or schema version disagrees
            with the live ``feature_builder``, or the artifact is unpromoted and
            the safety guards do not permit loading it.
    """
    path = resolve_model_path(settings)
    key = str(path)
    loaded = None if force_reload else _CACHE.get(key)
    if loaded is None:
        loaded = _load_from_disk(path)
        _CACHE[key] = loaded
        _log_threshold_divergence(loaded, settings)
    # Re-run on every call: the guards depend on MUTABLE config, not on the file,
    # so a cache hit must never be a way to bypass them.
    _enforce_safety_guards(loaded, settings)
    return loaded


def load_artifact(path: Path, settings: Settings) -> LoadedModel:
    """Load, validate and cache the artifact at an explicit path.

    The path-addressed counterpart to :func:`load_model`, which resolves its path from
    ``ML_MODEL_PATH``. Callers that already know WHICH artifact they want — the strategy
    registry, resolving a model bound to a strategy — need the loading and the two
    firewalls without the config indirection.

    Both guards still apply: contract/schema validation on a cache miss, and the
    unpromoted-artifact safety check on EVERY call, cache hit or not, because that one
    depends on mutable config rather than on the file.

    Raises:
        FileNotFoundError: artifact or its metadata sidecar is missing.
        RuntimeError: contract mismatch, schema mismatch, or the promotion guards refuse.
    """
    key = str(path)
    loaded = _CACHE.get(key)
    if loaded is None:
        loaded = _load_from_disk(path)
        _CACHE[key] = loaded
    _enforce_safety_guards(loaded, settings)
    return loaded


def load_challengers(settings: Settings) -> list[LoadedModel]:
    """Load every challenger named in ``ML_CHALLENGER_MODEL_PATHS``.

    A challenger is scored on the same live signals as the champion so their verdicts
    can be compared on identical evidence — the only comparison that answers "where
    did they disagree, and who was right?", which is what promotion turns on.

    Failure is per-model and NEVER propagates. A challenger that will not load is a
    research inconvenience; the champion still has to score the signal and the row
    still has to be written. Letting a broken challenger take down the live path would
    make adding one strictly more dangerous than not bothering, which would defeat the
    purpose.

    The champion is deliberately excluded if it also appears in the list, so it cannot
    be recorded twice — once authoritative, once not — which would double-count it in
    any agreement statistic.

    Returns:
        Successfully loaded challengers, possibly empty. Order follows the setting.
    """
    raw = (getattr(settings, "ML_CHALLENGER_MODEL_PATHS", "") or "").strip()
    if not raw:
        return []

    champion = str(resolve_model_path(settings))
    out: list[LoadedModel] = []
    seen: set[str] = {champion}
    for entry in (e.strip() for e in raw.split(",")):
        if not entry:
            continue
        path = Path(entry)
        if not path.is_absolute():
            path = _BACKEND_ROOT / path
        key = str(path)
        if key in seen:
            logger.warning("challenger %s is the champion or a duplicate — skipped", entry)
            continue
        seen.add(key)
        try:
            loaded = _CACHE.get(key) or _load_from_disk(path)
            _CACHE[key] = loaded
            _enforce_safety_guards(loaded, settings)
            out.append(loaded)
            logger.info("challenger loaded: %s", loaded.model_id)
        except Exception as exc:   # noqa: BLE001 — see docstring: never propagate
            logger.error("challenger %s failed to load (%s) — continuing without it", entry, exc)
    return out


def score(loaded_model: LoadedModel, features: dict) -> float:
    """Score one ``build_features`` dict → P(win).

    Args:
        loaded_model: a validated artifact from :func:`load_model`.
        features: the dict returned by ``feature_builder.build_features``. Extra
            keys (gated tier, risk payload) are ignored; only the artifact's model
            keys are selected, in the artifact's own order.

    Returns:
        The positive-class probability in [0.0, 1.0].

    Notes:
        NaN is passed through to XGBoost untouched — no imputation. Categorical
        values unseen at training time encode to all-zeros
        (``handle_unknown='ignore'``) rather than raising.
    """
    frame = _to_frame(loaded_model, features)
    proba = loaded_model.pipeline.predict_proba(frame)
    return float(proba[0][1])


def decide(prob: float, settings: Settings) -> str:
    """Turn a probability into a take/skip decision.

    Args:
        prob: P(win) from :func:`score`.
        settings: config supplying ``ML_DECISION_THRESHOLD``.

    Returns:
        :data:`DECISION_TAKE` when ``prob >= settings.ML_DECISION_THRESHOLD``
        (boundary inclusive — same convention as the training-time policy),
        otherwise :data:`DECISION_SKIP`.

    Raises:
        ValueError: ``prob`` is NaN (a NaN probability must never silently 'skip').
    """
    p = float(prob)
    if math.isnan(p):
        raise ValueError("cannot decide on a NaN probability — scoring failed upstream")
    return DECISION_TAKE if p >= float(settings.ML_DECISION_THRESHOLD) else DECISION_SKIP


def score_and_decide(
    loaded_model: LoadedModel,
    features: dict,
    settings: Settings,
    *,
    threshold: float | None = None,
) -> ShadowDecision:
    """Convenience chokepoint: score a feature dict and resolve the decision.

    This is the single call the live shadow path (Phase 2) makes per signal.

    Args:
        threshold: override the configured cut-off. Used when scoring a CHALLENGER,
            which has its own ``deployment_threshold`` selected on its own validation
            tail. Judging it at the champion's cut-off would measure the threshold
            rather than the model — v2's is 0.236 and v3's is 0.343, so the challenger
            would look far more permissive than it is and every agreement statistic
            would be about the wrong thing. ``None`` keeps the configured value, which
            is what the champion must use because that is what actually governed.
    """
    prob = score(loaded_model, features)
    cut = float(settings.ML_DECISION_THRESHOLD) if threshold is None else float(threshold)
    if prob != prob:  # NaN — same contract as decide(): never silently 'skip'
        raise ValueError(f"model {loaded_model.model_id} returned NaN probability")
    return ShadowDecision(
        probability=prob,
        decision=DECISION_TAKE if prob >= cut else DECISION_SKIP,
        model_id=loaded_model.model_id,
    )


def resolve_model_path(settings: Settings) -> Path:
    """Resolve ``settings.ML_MODEL_PATH`` to an absolute path (relative → backend/)."""
    raw = Path(str(settings.ML_MODEL_PATH))
    return (raw if raw.is_absolute() else _BACKEND_ROOT / raw).resolve()


def reset_cache() -> None:
    """Drop the process-level artifact cache (tests / hot config reload)."""
    _CACHE.clear()


# ── loading + validation ─────────────────────────────────────────────────────
def _load_from_disk(path: Path) -> LoadedModel:
    """Read metadata, validate the contract, then deserialize the pipeline."""
    meta_path = _metadata_path_for(path)
    if not path.exists():
        raise FileNotFoundError(f"ML artifact not found: {path} (ML_MODEL_PATH)")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"ML artifact metadata not found: {meta_path} — an artifact without its "
            "metadata cannot be contract-validated and will not be served"
        )

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    feature_order = _validate_feature_contract(metadata, path)
    _validate_schema_version(metadata, path)

    numeric = tuple(metadata.get("numeric_features") or NUMERIC_FEATURES)
    categorical = tuple(metadata.get("categorical_features") or CATEGORICAL_FEATURES)

    pipeline = load(path)
    model_id = _model_id_for(path)
    promoted = metadata.get("promoted") is True and _NOT_PROMOTED_MARKER not in path.name

    logger.info(
        "loaded ML artifact model_id=%s schema_v=%s promoted=%s features=%d",
        model_id,
        metadata.get("feature_schema_version"),
        promoted,
        len(feature_order),
    )
    return LoadedModel(
        model_id=model_id,
        pipeline=pipeline,
        metadata=metadata,
        model_path=path,
        metadata_path=meta_path,
        feature_order=feature_order,
        numeric_features=numeric,
        categorical_features=categorical,
        promoted=promoted,
    )


def _validate_feature_contract(metadata: dict, path: Path) -> tuple[str, ...]:
    """Assert artifact feature keys == live FEATURE_KEYS_MODEL (set + length).

    Returns the artifact's key list (its own order) on success.
    """
    keys = metadata.get("feature_keys_model") or metadata.get("feature_keys")
    if not keys:
        raise FeatureContractMismatch(
            f"ML artifact '{path.name}' metadata declares no feature key list "
            "('feature_keys_model') — cannot verify training/serving parity, refusing to load"
        )
    live = list(FEATURE_KEYS_MODEL)
    missing = sorted(set(live) - set(keys))
    unexpected = sorted(set(keys) - set(live))
    if missing or unexpected or len(keys) != len(live):
        raise FeatureContractMismatch(
            f"feature contract mismatch for ML artifact '{path.name}': "
            f"artifact declares {len(keys)} keys, live feature_builder.FEATURE_KEYS_MODEL "
            f"declares {len(live)}. "
            f"In live contract but MISSING from artifact: {missing or '(none)'}. "
            f"In artifact but NOT in live contract: {unexpected or '(none)'}. "
            "Refusing to load — scoring a mismatched feature vector would silently "
            "corrupt every prediction. Retrain the artifact against the current contract."
        )
    return tuple(keys)


def _validate_schema_version(metadata: dict, path: Path) -> None:
    """Assert artifact feature_schema_version == live FEATURE_SCHEMA_VERSION."""
    raw = metadata.get("feature_schema_version")
    if raw is None:
        raise FeatureContractMismatch(
            f"ML artifact '{path.name}' metadata declares no 'feature_schema_version' — "
            "refusing to load"
        )
    if int(raw) != int(FEATURE_SCHEMA_VERSION):
        raise FeatureContractMismatch(
            f"feature_schema_version mismatch for ML artifact '{path.name}': "
            f"artifact=v{int(raw)}, live feature_builder=v{int(FEATURE_SCHEMA_VERSION)}. "
            "The same key names can carry different semantics across schema versions — "
            "refusing to load. Retrain on the current schema."
        )


def _enforce_safety_guards(loaded: LoadedModel, settings: Settings) -> None:
    """Refuse to serve a DO-NOT-PROMOTE artifact unless shadowing is explicit and safe."""
    if loaded.promoted:
        return
    if not settings.ML_ALLOW_UNPROMOTED_MODEL:
        raise RuntimeError(
            f"ML artifact '{loaded.model_id}' FAILED its walk-forward promotion gate "
            f"(gate_verdict={loaded.metadata.get('gate_verdict')!r}). "
            "Set ML_ALLOW_UNPROMOTED_MODEL=true to shadow it deliberately, or point "
            "ML_MODEL_PATH at a promoted artifact."
        )
    if settings.ORDER_PLACEMENT_ENABLED:
        raise RuntimeError(
            f"refusing to load unpromoted ML artifact '{loaded.model_id}' while "
            "ORDER_PLACEMENT_ENABLED=true. A model that failed the promotion gate may "
            "only ever be OBSERVED (shadow mode, zero orders). Set "
            "ORDER_PLACEMENT_ENABLED=false, or deploy a promoted artifact."
        )


def _log_threshold_divergence(loaded: LoadedModel, settings: Settings) -> None:
    """Warn when the configured decision threshold differs from the artifact's own."""
    artifact_threshold = loaded.metadata.get("deployment_threshold")
    if artifact_threshold is None:
        return
    configured = float(settings.ML_DECISION_THRESHOLD)
    if not math.isclose(float(artifact_threshold), configured, rel_tol=1e-9, abs_tol=1e-12):
        logger.warning(
            "ML_DECISION_THRESHOLD=%r differs from artifact '%s' deployment_threshold=%r "
            "— config wins, but verify this divergence is intentional",
            configured,
            loaded.model_id,
            float(artifact_threshold),
        )


# ── feature vector assembly ──────────────────────────────────────────────────
def _to_frame(loaded: LoadedModel, features: dict) -> pd.DataFrame:
    """Build the one-row model matrix in the artifact's expected column order.

    Uses the SAME coercion the training extractor applies to each corpus row
    (``dataset._to_number`` / the categorical sentinel), so a value reaches the
    model identically whether it came from a stored ``signal_reasoning`` JSON at
    training time or from a live ``build_features`` dict now.
    """
    numeric = set(loaded.numeric_features)
    categorical = set(loaded.categorical_features)

    row: dict = {}
    for key in loaded.feature_order:
        value = features.get(key)
        if key in categorical:
            row[key] = value if isinstance(value, str) and value else _CAT_MISSING
        elif key in numeric:
            row[key] = _to_number(value)
        else:  # key in neither list — treat as numeric (contract already validated)
            row[key] = _to_number(value)

    frame = pd.DataFrame([row], columns=list(loaded.feature_order))
    for col in loaded.numeric_features:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    for col in loaded.categorical_features:
        frame[col] = frame[col].astype("object")
    return frame


# ── path helpers ─────────────────────────────────────────────────────────────
def _model_id_for(path: Path) -> str:
    """Artifact identifier == filename stem, e.g. ``s1_xgb_v2_<hash>.NOT_PROMOTED``."""
    name = path.name
    return name[: -len(_MODEL_SUFFIX)] if name.endswith(_MODEL_SUFFIX) else path.stem


def _metadata_path_for(path: Path) -> Path:
    """Sibling metadata JSON, per the naming convention in ``ml/artifact.save_artifact``."""
    return path.with_name(f"{_model_id_for(path)}{_METADATA_SUFFIX}")
