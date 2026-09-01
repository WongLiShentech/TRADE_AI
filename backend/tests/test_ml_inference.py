"""Integration tests for app.services.ml.inference — the serving firewall (M8-Shadow).

Loads the REAL serialized S1 artifact (``settings.ML_MODEL_PATH``) and scores a
REAL point-in-time feature dict built from the live dev Postgres, because the
whole purpose of this module is to prove that the training contract and the
serving contract are the same object — a mocked artifact would prove nothing.

Contract/guard failure cases use a COPY of the real artifact in ``tmp_path`` with
a doctored metadata JSON, so the on-disk production artifact is never mutated and
each case gets its own cache key.

Run from backend/:  python -m pytest tests/test_ml_inference.py -v
"""
from __future__ import annotations

import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import pytest

from app.services.feature_builder import (
    FEATURE_KEYS_MODEL,
    FEATURE_SCHEMA_VERSION,
    build_features,
)
from app.services.ml import inference as inf

_GRAN = "H4"
_SYMBOL = "EUR_USD"
_T = datetime(2025, 7, 1, 12, 0, 0)
_CONFLUENCE = {"trend": True, "rsi": True, "structure": False, "session": True, "spread": True}


@pytest.fixture(autouse=True)
def _clear_artifact_cache():
    """The artifact cache is process-level; isolate every test from its neighbours."""
    inf.reset_cache()
    yield
    inf.reset_cache()


@pytest.fixture()
def live_features(db, settings, instrument):
    """A real PIT feature dict for a real (instrument, signal_time)."""
    inst = instrument(_SYMBOL)
    return build_features(inst, _T, _GRAN, _CONFLUENCE, db, settings)


def _fake_artifact(tmp_path: Path, settings, mutate) -> Path:
    """Copy the real artifact into tmp_path and apply ``mutate`` to its metadata."""
    src_model = inf.resolve_model_path(settings)
    src_meta = inf._metadata_path_for(src_model)
    dst_model = tmp_path / src_model.name
    dst_meta = tmp_path / src_meta.name
    shutil.copy(src_model, dst_model)
    meta = json.loads(src_meta.read_text(encoding="utf-8"))
    mutate(meta)
    dst_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return dst_model


def _settings_with(settings, **overrides):
    """A Settings copy with overrides — never mutates the cached global settings."""
    return settings.model_copy(update=overrides)


# ── 1. happy path ────────────────────────────────────────────────────────────
def test_loads_real_artifact_and_exposes_model_id(settings):
    model = inf.load_model(settings)

    assert model.model_id, "artifact must expose a provenance id"
    assert model.model_id == inf.resolve_model_path(settings).name.removesuffix(".joblib")
    assert model.params_hash and model.params_hash in model.model_id
    assert model.model_path.exists() and model.metadata_path.exists()
    # The v2 artifact is the deliberately-shadowed rejected candidate.
    assert model.promoted is False
    assert model.metadata["gate_verdict"] == "DO_NOT_PROMOTE"


def test_load_is_cached_per_process(settings):
    assert inf.load_model(settings) is inf.load_model(settings)
    assert inf.load_model(settings, force_reload=True) is not None


def test_artifact_feature_contract_matches_live_builder(settings):
    model = inf.load_model(settings)

    assert set(model.feature_order) == set(FEATURE_KEYS_MODEL)
    assert len(model.feature_order) == len(FEATURE_KEYS_MODEL)
    assert model.metadata["feature_schema_version"] == FEATURE_SCHEMA_VERSION


# ── 2. training/serving skew firewall ────────────────────────────────────────
def test_feature_key_mismatch_missing_key_raises(tmp_path, settings):
    dropped = FEATURE_KEYS_MODEL[0]

    def mutate(meta):
        meta["feature_keys_model"] = [k for k in meta["feature_keys_model"] if k != dropped]

    path = _fake_artifact(tmp_path, settings, mutate)
    with pytest.raises(RuntimeError) as exc:
        inf.load_model(_settings_with(settings, ML_MODEL_PATH=str(path)))

    msg = str(exc.value)
    assert "feature contract mismatch" in msg
    assert dropped in msg, "the error must name the differing key"


def test_feature_key_mismatch_extra_key_raises(tmp_path, settings):
    def mutate(meta):
        meta["feature_keys_model"] = [*meta["feature_keys_model"], "totally_new_feature"]

    path = _fake_artifact(tmp_path, settings, mutate)
    with pytest.raises(RuntimeError) as exc:
        inf.load_model(_settings_with(settings, ML_MODEL_PATH=str(path)))

    assert "totally_new_feature" in str(exc.value)


def test_missing_feature_key_list_raises(tmp_path, settings):
    def mutate(meta):
        meta.pop("feature_keys_model", None)

    path = _fake_artifact(tmp_path, settings, mutate)
    with pytest.raises(RuntimeError, match="no feature key list"):
        inf.load_model(_settings_with(settings, ML_MODEL_PATH=str(path)))


def test_schema_version_mismatch_raises(tmp_path, settings):
    def mutate(meta):
        meta["feature_schema_version"] = FEATURE_SCHEMA_VERSION + 1

    path = _fake_artifact(tmp_path, settings, mutate)
    with pytest.raises(RuntimeError) as exc:
        inf.load_model(_settings_with(settings, ML_MODEL_PATH=str(path)))

    msg = str(exc.value)
    assert "feature_schema_version mismatch" in msg
    assert f"v{FEATURE_SCHEMA_VERSION + 1}" in msg and f"v{FEATURE_SCHEMA_VERSION}" in msg


def test_missing_artifact_path_raises(tmp_path, settings):
    absent = tmp_path / "does_not_exist.joblib"
    with pytest.raises(FileNotFoundError):
        inf.load_model(_settings_with(settings, ML_MODEL_PATH=str(absent)))


def test_missing_metadata_raises(tmp_path, settings):
    src = inf.resolve_model_path(settings)
    dst = tmp_path / src.name
    shutil.copy(src, dst)  # model only — no metadata sidecar
    with pytest.raises(FileNotFoundError, match="metadata"):
        inf.load_model(_settings_with(settings, ML_MODEL_PATH=str(dst)))


# ── 3. unpromoted-artifact guard ─────────────────────────────────────────────
def test_unpromoted_artifact_refused_when_not_explicitly_allowed(settings):
    with pytest.raises(RuntimeError, match="promotion gate"):
        inf.load_model(_settings_with(settings, ML_ALLOW_UNPROMOTED_MODEL=False))


def test_unpromoted_artifact_refused_when_order_placement_enabled(settings):
    with pytest.raises(RuntimeError, match="ORDER_PLACEMENT_ENABLED"):
        inf.load_model(
            _settings_with(
                settings, ML_ALLOW_UNPROMOTED_MODEL=True, ORDER_PLACEMENT_ENABLED=True
            )
        )


def test_guards_are_rechecked_on_cache_hit(settings):
    """A successful load must not become a bypass for a later unsafe config."""
    inf.load_model(settings)  # warms the cache
    with pytest.raises(RuntimeError):
        inf.load_model(_settings_with(settings, ORDER_PLACEMENT_ENABLED=True))


def test_promoted_artifact_loads_regardless_of_guards(tmp_path, settings):
    """A PROMOTED artifact is not subject to the shadow-only guards."""

    def mutate(meta):
        meta["promoted"] = True
        meta["gate_verdict"] = "PROMOTE"

    src = inf.resolve_model_path(settings)
    promoted_name = src.name.replace(".NOT_PROMOTED", "")
    path = _fake_artifact(tmp_path, settings, mutate)
    # Rename both files so the filename marker is gone too (marker ⇒ unpromoted).
    new_model = path.with_name(promoted_name)
    path.rename(new_model)
    inf._metadata_path_for(path).rename(inf._metadata_path_for(new_model))

    model = inf.load_model(
        _settings_with(
            settings,
            ML_MODEL_PATH=str(new_model),
            ML_ALLOW_UNPROMOTED_MODEL=False,
            ORDER_PLACEMENT_ENABLED=True,
        )
    )
    assert model.promoted is True


def test_filename_marker_alone_marks_artifact_unpromoted(tmp_path, settings):
    """Metadata claiming promoted:true cannot override the NOT_PROMOTED filename."""

    def mutate(meta):
        meta["promoted"] = True
        meta["gate_verdict"] = "PROMOTE"

    path = _fake_artifact(tmp_path, settings, mutate)  # keeps .NOT_PROMOTED in the name
    with pytest.raises(RuntimeError):
        inf.load_model(
            _settings_with(settings, ML_MODEL_PATH=str(path), ML_ALLOW_UNPROMOTED_MODEL=False)
        )


# ── 4. scoring on real features ──────────────────────────────────────────────
def test_score_real_feature_dict_returns_probability(settings, live_features):
    model = inf.load_model(settings)
    prob = inf.score(model, live_features)

    assert isinstance(prob, float)
    assert not math.isnan(prob)
    assert 0.0 <= prob <= 1.0


def test_score_is_deterministic(settings, live_features):
    model = inf.load_model(settings)
    assert inf.score(model, live_features) == inf.score(model, live_features)
    # ...and across a fresh deserialization of the same artifact.
    first = inf.score(model, live_features)
    inf.reset_cache()
    assert inf.score(inf.load_model(settings), live_features) == first


def test_score_ignores_non_model_keys(settings, live_features):
    """The gated tier + risk payload ride along in the dict and must not affect scoring."""
    model = inf.load_model(settings)
    trimmed = {k: live_features[k] for k in FEATURE_KEYS_MODEL}
    assert inf.score(model, trimmed) == inf.score(model, live_features)


def test_score_tolerates_all_nan_features(settings):
    """XGBoost is NaN-native: a fully-missing vector scores rather than crashing."""
    model = inf.load_model(settings)
    empty = {k: None for k in FEATURE_KEYS_MODEL}
    prob = inf.score(model, empty)
    assert 0.0 <= prob <= 1.0


def test_score_tolerates_unseen_categorical(settings, live_features):
    """handle_unknown='ignore' → an unseen instrument_category degrades, never raises."""
    model = inf.load_model(settings)
    mutated = {**live_features, "instrument_category": "brand_new_bucket"}
    prob = inf.score(model, mutated)
    assert 0.0 <= prob <= 1.0


# ── 5. decision threshold ────────────────────────────────────────────────────
def test_decide_respects_threshold_boundary(settings):
    thr = float(settings.ML_DECISION_THRESHOLD)

    assert inf.decide(thr, settings) == inf.DECISION_TAKE          # boundary inclusive
    assert inf.decide(thr + 1e-6, settings) == inf.DECISION_TAKE
    assert inf.decide(thr - 1e-6, settings) == inf.DECISION_SKIP
    assert inf.decide(0.0, settings) == inf.DECISION_SKIP
    assert inf.decide(1.0, settings) == inf.DECISION_TAKE


def test_decide_uses_config_not_artifact_threshold(settings):
    """Behaviour is config-driven: the artifact's own threshold never overrides it."""
    strict = _settings_with(settings, ML_DECISION_THRESHOLD=0.99)
    model = inf.load_model(settings)
    assert inf.decide(model.deployment_threshold, strict) == inf.DECISION_SKIP
    assert inf.decide(model.deployment_threshold, settings) == inf.DECISION_TAKE


def test_decide_rejects_nan(settings):
    with pytest.raises(ValueError):
        inf.decide(float("nan"), settings)


def test_decision_values_are_the_two_persisted_literals():
    assert (inf.DECISION_TAKE, inf.DECISION_SKIP) == ("take", "skip")


# ── 6. Phase-2 entry point ───────────────────────────────────────────────────
def test_score_and_decide_returns_full_provenance(settings, live_features):
    model = inf.load_model(settings)
    result = inf.score_and_decide(model, live_features, settings)

    assert result.model_id == model.model_id
    assert result.probability == inf.score(model, live_features)
    assert result.decision == inf.decide(result.probability, settings)
    assert result.decision in (inf.DECISION_TAKE, inf.DECISION_SKIP)


def test_relative_model_path_resolves_against_backend_root(settings):
    """ML_MODEL_PATH is relative to backend/ — never a hardcoded absolute path."""
    resolved = inf.resolve_model_path(settings)
    assert resolved.is_absolute() and resolved.exists()
    assert resolved.parent.name == "models"
