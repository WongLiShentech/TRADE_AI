"""Integration tests for app.services.shadow.recorder — M8-Shadow Phase 2.

These run against the live dev Postgres (same convention as the other integration
suites) because the properties under test are persistence properties: the shadow row
must actually land in ``trades`` with the right columns, the natural-key guard must
actually be enforced by the DB, and the feature dict must actually be the one
``feature_builder`` produced. A mocked session would prove none of that.

Every test writes only ``stage='shadow'`` rows for the test symbol and an autouse
fixture deletes them before and after each test, so the real corpus
(``stage='backtest'``) is never touched.

Run from backend/:  python -m pytest tests/test_shadow_recorder.py -v
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import func

from app.domain.timeframes import get_timeframe
from app.models.candle import Candle
from app.models.trade import Trade
from app.services.feature_builder import (
    FEATURE_KEYS_GATED,
    FEATURE_KEYS_MODEL,
    FEATURE_SCHEMA_VERSION,
    PAYLOAD_KEYS,
    build_features,
)
from app.services.ml import inference as inf
from app.services.risk_engine import ValidatedSignal
from app.services.shadow import recorder as rec
from app.services.signal_engine.base import SignalOutput

_SYMBOL = "EUR_USD"
_GRAN = "H4"
# Sentinel signal times, far from any real backtest row. Each test uses its own
# offset so a leaked row can never collide with a neighbour's natural key.
_T_BASE = datetime(2025, 7, 1, 12, 0, 1)
# SENTINEL TIME BAND. Every row this suite writes lands inside
# [_T_BASE, _T_BASE + _SENTINEL_WINDOW]; the purge fixture and the row-count helper
# are both bounded to it.
#
# Mandatory, not cosmetic: the purge runs against the real dev database via the app's
# own SessionLocal on a live-active symbol, so an unbounded
# ``instrument_id = X AND stage = 'shadow'`` DELETE would irreversibly destroy real
# live shadow observations on every pytest run. The bound makes that impossible, and
# also keeps the exact row-count assertions below independent of live traffic.
_SENTINEL_WINDOW = timedelta(days=30)
_SENTINEL_END = _T_BASE + _SENTINEL_WINDOW
_CONFLUENCE = {"trend": True, "rsi": True, "structure": False, "session": True, "spread": True}
_REJECTION = "STOP_TOO_WIDE"


# ── fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _clean_shadow_rows(db, instrument):
    """Delete every shadow row for the test symbol before and after each test."""
    inst = instrument(_SYMBOL)

    def _purge():
        db.query(Trade).filter(
            Trade.instrument_id == inst.id,
            Trade.stage == rec.STAGE_SHADOW,
            # TIME-BOUNDED — see _SENTINEL_WINDOW. Never remove this clause: without
            # it this DELETE wipes real live shadow rows on every pytest run.
            Trade.opened_at >= _T_BASE,
            Trade.opened_at <= _SENTINEL_END,
        ).delete(synchronize_session=False)
        db.commit()

    _purge()
    yield
    _purge()


@pytest.fixture(autouse=True)
def _clear_artifact_cache():
    inf.reset_cache()
    yield
    inf.reset_cache()


@pytest.fixture()
def live_row_cleanup(db, instrument):
    """Delete ONLY the live-time shadow rows a test created — by primary key.

    The two ``_observe_shadow`` wiring tests exercise the real pipeline block, which
    derives T from :func:`recorder.live_signal_time`. That lands the row in the LIVE
    present, outside this suite's sentinel band, so the time-bounded purge cannot
    (and must not) reach it.

    A broad delete would be exactly the bug this suite was fixed for, so cleanup is
    keyed on ``id > max(id) at setup``: only rows created DURING the test are removed.
    If the live pipeline had already recorded a genuine row at the same T, its id is
    lower and it survives untouched — and the test correctly observes the recorder's
    duplicate guard instead.
    """
    inst_obj = instrument(_SYMBOL)
    high_water = db.query(func.coalesce(func.max(Trade.id), 0)).scalar()
    yield
    db.query(Trade).filter(
        Trade.id > high_water,
        Trade.instrument_id == inst_obj.id,
        Trade.stage == rec.STAGE_SHADOW,
    ).delete(synchronize_session=False)
    db.commit()


@pytest.fixture()
def inst(instrument):
    return instrument(_SYMBOL)


@pytest.fixture()
def model(champion_artifact):
    """The live champion artifact — skips while no artifact matches the contract.

    See ``conftest.champion_artifact``: between a feature-schema bump and the first
    retrain, refusing to load is correct behaviour, so these tests skip rather than fail.
    """
    return champion_artifact


@pytest.fixture()
def features(db, settings, inst):
    return build_features(inst, _T_BASE, _GRAN, _CONFLUENCE, db, settings)


@pytest.fixture()
def signal(features):
    """A BUY SignalOutput sized off the real ATR at the sentinel signal time."""
    atr = features.get("atr14")
    close = features.get("h4_close")
    assert atr and not math.isnan(atr), "test fixture needs real ATR data"
    entry = float(close)
    stop = entry - 1.5 * float(atr)
    target = entry + 2.0 * (entry - stop)
    return SignalOutput(
        instrument=_SYMBOL,
        granularity=_GRAN,
        direction="BUY",
        entry=entry,
        stop=stop,
        target=target,
        confidence_score=sum(1 for v in _CONFLUENCE.values() if v),
        score_breakdown=dict(_CONFLUENCE),
    )


def _settings_with(settings, **overrides):
    """A Settings copy with overrides — never mutates the cached global settings."""
    return settings.model_copy(update=overrides)


def _approved(units: int = 1000, risk_amount: float = 1.0, pip_value: float = 0.0001):
    return rec.RiskAssessment(
        passed=True, rejection_reason=None,
        units=units, risk_amount=risk_amount, pip_value=pip_value,
    )


def _as_models(champion, settings, *, challengers=()):
    """Wrap loaded artifacts into the StrategyModels the recorder now takes.

    Models are resolved from the registry in production (a model is valid only for the
    strategy whose outcomes taught it). Tests hand the recorder the same shape directly
    so they exercise the real write path without needing registry rows.
    """
    from app.services.ml.registry import RegisteredModel, StrategyModels

    return StrategyModels(
        champion=(
            None if champion is None
            else RegisteredModel(
                loaded=champion,
                threshold=float(settings.ML_DECISION_THRESHOLD),
                is_champion=True,
            )
        ),
        challengers=tuple(
            RegisteredModel(
                loaded=c,
                threshold=float(c.metadata.get("deployment_threshold")
                                or settings.ML_DECISION_THRESHOLD),
                is_champion=False,
            )
            for c in challengers
        ),
    )


def _record(db, settings, inst, signal, features, model, *, t=_T_BASE, risk=None,
            challengers=()):
    from app.services.ml.registry import StrategyModels

    models = model if isinstance(model, StrategyModels) else _as_models(
        model, settings, challengers=challengers
    )
    return rec.record_shadow_decision(
        db, settings, inst, signal, features, models,
        signal_time=t, risk=risk if risk is not None else _approved(),
    )


def _live_shadow_rows(db, inst):
    """Shadow rows at the LIVE signal time T — the natural key the pipeline block
    writes. Scoped to that single instant, never to the whole stage."""
    return (
        db.query(Trade)
        .filter(
            Trade.instrument_id == inst.id,
            Trade.stage == rec.STAGE_SHADOW,
            Trade.opened_at == rec.live_signal_time(db, inst, _GRAN),
        )
        .all()
    )


def _shadow_rows(db, inst):
    """Shadow rows this suite wrote — scoped to the sentinel band, never all of them,
    so real live observations can neither be counted nor asserted away."""
    return (
        db.query(Trade)
        .filter(
            Trade.instrument_id == inst.id,
            Trade.stage == rec.STAGE_SHADOW,
            Trade.opened_at >= _T_BASE,
            Trade.opened_at <= _SENTINEL_END,
        )
        .all()
    )


# ── 1. the row lands with all three ML columns + the full feature dict ───────
def test_writes_shadow_row_with_all_three_ml_columns(db, settings, inst, signal, features, model):
    trade = _record(db, settings, inst, signal, features, model)

    assert trade is not None
    assert trade.id is not None
    assert trade.stage == rec.STAGE_SHADOW == "shadow"
    assert trade.signal_source == "rule_based"
    # All three ML columns populated.
    assert trade.ml_probability is not None and 0.0 <= trade.ml_probability <= 1.0
    assert trade.ml_decision in (inf.DECISION_TAKE, inf.DECISION_SKIP)
    assert trade.ml_model_id == model.model_id
    # ...and the score matches what inference would independently produce.
    assert trade.ml_probability == pytest.approx(inf.score(model, features))


def test_signal_reasoning_carries_the_full_feature_dict(db, settings, inst, signal, features, model):
    trade = _record(db, settings, inst, signal, features, model)
    reasoning = trade.signal_reasoning

    for key in FEATURE_KEYS_MODEL + FEATURE_KEYS_GATED + PAYLOAD_KEYS:
        assert key in reasoning, f"feature key '{key}' missing from signal_reasoning"
    # Flat top level (S1's extractor reads a shadow row exactly like a backtest row).
    assert reasoning["confluence_score"] == signal.confidence_score
    assert reasoning["session"] == trade.session
    # Shadow metadata is namespaced, never mixed into the feature keys.
    assert rec.REASONING_SHADOW_KEY in reasoning
    assert set(FEATURE_KEYS_MODEL).isdisjoint({rec.REASONING_SHADOW_KEY})


def test_feature_schema_version_matches_live_contract(db, settings, inst, signal, features, model):
    trade = _record(db, settings, inst, signal, features, model)

    assert trade.signal_reasoning["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert model.metadata["feature_schema_version"] == FEATURE_SCHEMA_VERSION


def test_signal_fields_are_copied_verbatim(db, settings, inst, signal, features, model):
    trade = _record(db, settings, inst, signal, features, model)

    assert trade.direction == signal.direction
    assert trade.entry_price == pytest.approx(signal.entry)
    assert trade.stop_price == pytest.approx(signal.stop)
    assert trade.tp_price == pytest.approx(signal.target)
    assert trade.confluence_score == signal.confidence_score
    assert trade.stop_method == "atr"
    assert trade.opened_at == _T_BASE


def test_outcome_columns_left_null_for_phase3(db, settings, inst, signal, features, model):
    trade = _record(db, settings, inst, signal, features, model)

    assert trade.outcome is None
    assert trade.rr_actual is None
    assert trade.exit_price is None
    assert trade.exit_reason is None
    assert trade.closed_at is None
    assert trade.actual_pip_loss is None


def test_nan_features_are_persisted_as_json_null(db, settings, inst, signal, features, model):
    """build_features emits float('nan'); Postgres JSON rejects the NaN token."""
    trade = _record(db, settings, inst, signal, features, model)
    values = [v for v in trade.signal_reasoning.values() if isinstance(v, float)]
    assert not any(math.isnan(v) for v in values)


# ── 2. BOTH decisions are recorded ───────────────────────────────────────────
def test_records_a_take_decision(db, settings, inst, signal, features, model):
    permissive = _settings_with(settings, ML_DECISION_THRESHOLD=0.0)
    trade = _record(db, permissive, inst, signal, features, model)

    assert trade.ml_decision == inf.DECISION_TAKE


def test_records_a_skip_decision(db, settings, inst, signal, features, model):
    strict = _settings_with(settings, ML_DECISION_THRESHOLD=1.0)
    trade = _record(db, strict, inst, signal, features, model)

    assert trade.ml_decision == inf.DECISION_SKIP
    # A skip is a first-class observation, not a discarded one.
    assert trade.id is not None and trade.ml_probability is not None


def test_take_and_skip_both_persist_as_rows(db, settings, inst, signal, features, model):
    _record(db, _settings_with(settings, ML_DECISION_THRESHOLD=0.0),
            inst, signal, features, model, t=_T_BASE)
    _record(db, _settings_with(settings, ML_DECISION_THRESHOLD=1.0),
            inst, signal, features, model, t=_T_BASE + timedelta(hours=4))

    decisions = {t.ml_decision for t in _shadow_rows(db, inst)}
    assert decisions == {inf.DECISION_TAKE, inf.DECISION_SKIP}


# ── 3. idempotency ───────────────────────────────────────────────────────────
def test_same_signal_recorded_twice_writes_one_row(db, settings, inst, signal, features, model):
    first = _record(db, settings, inst, signal, features, model)
    second = _record(db, settings, inst, signal, features, model)

    assert first is not None
    assert second is None, "a duplicate (instrument, signal_time) must not be re-recorded"
    assert len(_shadow_rows(db, inst)) == 1


def test_different_signal_times_write_separate_rows(db, settings, inst, signal, features, model):
    _record(db, settings, inst, signal, features, model, t=_T_BASE)
    _record(db, settings, inst, signal, features, model, t=_T_BASE + timedelta(hours=4))

    assert len(_shadow_rows(db, inst)) == 2


def _bare_shadow_row(inst, strategy_id):
    return Trade(
        instrument_id=inst.id, direction="BUY",
        entry_price=1.0, stop_price=0.9, tp_price=1.2,
        units=0, risk_amount=0.0, expected_pip_loss=0.0, rr_entry=2.0,
        signal_source="rule_based", stage=rec.STAGE_SHADOW, opened_at=_T_BASE,
        strategy_id=strategy_id,
    )


def test_db_partial_unique_index_backstops_the_natural_key(db, settings, inst, signal, features, model):
    """Bypass the query-before-insert guard: the DB index must still refuse the dup.

    The duplicate must carry the SAME strategy_id as the recorded row — the natural
    key is (instrument, signal_time, strategy), so a row with a different strategy is
    not a duplicate at all, it is the second strategy's opinion.
    """
    from sqlalchemy.exc import IntegrityError

    written = _record(db, settings, inst, signal, features, model)
    db.add(_bare_shadow_row(inst, written.strategy_id))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()
    assert len(_shadow_rows(db, inst)) == 1


def test_two_strategies_may_both_record_the_same_instrument_and_bar(
    db, settings, inst, signal, features, model
):
    """The reason the key was widened.

    Two strategies sharing an entry rule fire on the same instrument at the same H4
    close constantly — that is the normal case, not an edge case. Under the old key
    the second write raised IntegrityError, the recorder read that as "already
    recorded", and one strategy silently recorded nothing whenever it agreed with the
    other. A filter that stops recording when it agrees is worse than no filter: the
    gap is invisible and biased.
    """
    from app.models.strategy import Strategy

    written = _record(db, settings, inst, signal, features, model)
    # A REAL second strategy — strategy_id is a foreign key, so a fabricated id would
    # fail on the constraint rather than exercising the unique index.
    other = (
        db.query(Strategy)
        .filter(Strategy.id != written.strategy_id)
        .order_by(Strategy.id)
        .first()
    )
    if other is None:
        pytest.skip("only one strategy registered — nothing to collide with")

    db.add(_bare_shadow_row(inst, other.id))
    db.commit()   # must NOT raise
    rows = _shadow_rows(db, inst)
    assert len(rows) == 2, "the second strategy's observation was rejected"
    assert {r.strategy_id for r in rows} == {written.strategy_id, other.id}


def test_unattributed_rows_still_get_a_duplicate_guard(db, settings, inst, signal, features, model):
    """COALESCE(strategy_id, 0) exists so NULL rows share one bucket.

    Postgres treats NULL as distinct from NULL, so a bare nullable column in a unique
    index would remove the guard from precisely the rows that most need it — the
    unattributed ones written when strategy resolution fails.
    """
    from sqlalchemy.exc import IntegrityError

    db.add(_bare_shadow_row(inst, None))
    db.commit()
    db.add(_bare_shadow_row(inst, None))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


# ── 4. safety gates ──────────────────────────────────────────────────────────
def test_shadow_mode_disabled_writes_nothing(db, settings, inst, signal, features, model):
    off = _settings_with(settings, SHADOW_MODE_ENABLED=False)
    result = _record(db, off, inst, signal, features, model)

    assert result is None
    assert _shadow_rows(db, inst) == []


def test_order_placement_enabled_is_a_hard_failure_and_writes_nothing(
    db, settings, inst, signal, features, model
):
    unsafe = _settings_with(settings, ORDER_PLACEMENT_ENABLED=True)
    with pytest.raises(RuntimeError, match="ORDER_PLACEMENT_ENABLED"):
        _record(db, unsafe, inst, signal, features, model)

    assert _shadow_rows(db, inst) == []


def test_shadow_disabled_takes_precedence_over_the_order_guard(
    db, settings, inst, signal, features, model
):
    """Shadow off + orders on is a legitimate (future) live config — it must not raise."""
    cfg = _settings_with(settings, SHADOW_MODE_ENABLED=False, ORDER_PLACEMENT_ENABLED=True)
    assert _record(db, cfg, inst, signal, features, model) is None


def test_env_has_order_placement_disabled(settings):
    """The deployed config must be observe-only while a NOT_PROMOTED artifact is loaded."""
    assert settings.ORDER_PLACEMENT_ENABLED is False


# ── 5. NaN probability — the live loop survives, the row is still written ────
def test_nan_probability_records_row_with_null_decision(
    db, settings, inst, signal, features, model, monkeypatch
):
    """inference.decide raises on NaN by design; the recorder converts that into a
    NULL-decision row + an error marker rather than dying or dropping the signal."""
    monkeypatch.setattr(inf, "score", lambda *_a, **_k: float("nan"))

    trade = _record(db, settings, inst, signal, features, model)

    assert trade is not None, "the observation must survive a scoring failure"
    assert trade.ml_probability is None
    assert trade.ml_decision is None
    assert trade.ml_model_id == model.model_id, "the failing artifact is still identified"
    shadow = trade.signal_reasoning[rec.REASONING_SHADOW_KEY]
    assert shadow["ml_error"] and "ValueError" in shadow["ml_error"]
    # Features and the risk verdict are intact — the row remains useful to Phase 3.
    assert trade.signal_reasoning["confluence_score"] == signal.confidence_score
    assert shadow["risk_passed"] is True


def test_arbitrary_scoring_exception_also_records_row(
    db, settings, inst, signal, features, model, monkeypatch
):
    def _boom(*_a, **_k):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(inf, "score", _boom)
    trade = _record(db, settings, inst, signal, features, model)

    assert trade is not None and trade.ml_decision is None
    assert "model exploded" in trade.signal_reasoning[rec.REASONING_SHADOW_KEY]["ml_error"]


# ── 6. RiskEngine verdict is recorded and distinguishable ───────────────────
def test_risk_approved_row_is_marked_passed(db, settings, inst, signal, features, model):
    trade = _record(db, settings, inst, signal, features, model, risk=_approved(units=1234))
    shadow = trade.signal_reasoning[rec.REASONING_SHADOW_KEY]

    assert shadow["risk_passed"] is True
    assert shadow["risk_rejection_reason"] is None
    assert shadow["risk_units"] == 1234
    assert trade.units == 1234


def test_risk_rejected_signal_is_recorded_and_distinguishable(
    db, settings, inst, signal, features, model
):
    rejected = rec.RiskAssessment.rejected(_REJECTION, risk_amount=1.0)
    trade = _record(db, settings, inst, signal, features, model, risk=rejected)
    shadow = trade.signal_reasoning[rec.REASONING_SHADOW_KEY]

    assert trade is not None, "a risk-rejected signal is still an observation"
    assert shadow["risk_passed"] is False
    assert shadow["risk_rejection_reason"] == _REJECTION
    assert shadow["risk_units"] == 0 and trade.units == 0
    # The model still scored it — that is the point of shadowing.
    assert trade.ml_decision in (inf.DECISION_TAKE, inf.DECISION_SKIP)


def test_approved_and_rejected_rows_are_separable_by_query(
    db, settings, inst, signal, features, model
):
    _record(db, settings, inst, signal, features, model,
            t=_T_BASE, risk=_approved())
    _record(db, settings, inst, signal, features, model,
            t=_T_BASE + timedelta(hours=4),
            risk=rec.RiskAssessment.rejected(_REJECTION, risk_amount=1.0))

    rows = _shadow_rows(db, inst)
    verdicts = {
        r.signal_reasoning[rec.REASONING_SHADOW_KEY]["risk_passed"] for r in rows
    }
    assert verdicts == {True, False}


def test_risk_assessment_factories():
    validated = ValidatedSignal(signal=None, units=500, risk_amount=2.5, pip_value=0.0001)
    approved = rec.RiskAssessment.approved(validated)
    assert approved.passed and approved.rejection_reason is None and approved.units == 500

    rejected = rec.RiskAssessment.rejected("INSUFFICIENT_RR", risk_amount=2.5)
    assert not rejected.passed
    assert rejected.rejection_reason == "INSUFFICIENT_RR"
    assert rejected.units == 0 and rejected.pip_value is None
    assert rejected.risk_amount == 2.5


# ── 7. live signal-time convention (feature parity with the M7 runner) ──────
def test_live_signal_time_is_bar_close_plus_one_second(db, inst):
    newest = (
        db.query(Candle.timestamp)
        .filter(
            Candle.instrument_id == inst.id,
            Candle.granularity == _GRAN,
            Candle.price_type == "M",
        )
        .order_by(Candle.timestamp.desc())
        .first()
    )
    assert newest is not None
    bar_open = newest[0]
    bar_close = bar_open + timedelta(hours=get_timeframe(_GRAN).period_hours)

    t = rec.live_signal_time(db, inst, _GRAN)

    assert t == bar_close + timedelta(seconds=1)
    assert t > bar_close, "T must be STRICTLY after the decision bar's close"
    assert t.tzinfo is None, "naive UTC, matching candles.timestamp"


def test_live_signal_time_matches_the_m7_runner_convention(db, inst):
    """The runner uses `t_sig = bar_close + timedelta(seconds=1)`; live must agree."""
    t = rec.live_signal_time(db, inst, _GRAN)
    period = timedelta(hours=get_timeframe(_GRAN).period_hours)
    bar_open = t - timedelta(seconds=1) - period
    # T must exclude the bar that OPENS at bar_close (the not-yet-decided next bar)
    # while including the decision bar itself.
    assert bar_open < t
    assert (bar_open + period) < t <= (bar_open + period + timedelta(seconds=1))


def test_live_signal_time_none_when_no_mid_candles(db, inst):
    """M1 is ingested Bid/Ask only (no Mid) — no decision series, hence no T."""
    assert rec.live_signal_time(db, inst, "M1") is None


def test_live_signal_time_rejects_an_unregistered_timeframe(db, inst):
    """Fails loud rather than silently defaulting to a 4-hour period (timeframes.py)."""
    with pytest.raises(ValueError):
        rec.live_signal_time(db, inst, "H1")


# ── 8. no broker write is reachable ─────────────────────────────────────────
def test_recording_never_calls_a_broker_place_order(
    db, settings, inst, signal, features, model, monkeypatch
):
    """Trip-wire every BrokerClient implementation; recording must not touch any."""
    from app.brokers import alpaca, binance, oanda

    def _tripwire(*_a, **_k):
        raise AssertionError("shadow recording must NEVER reach a broker order method")

    for module, cls_name in (
        (oanda, "OandaClient"), (alpaca, "AlpacaClient"), (binance, "BinanceClient"),
    ):
        cls = getattr(module, cls_name, None)
        if cls is not None and hasattr(cls, "place_order"):
            monkeypatch.setattr(cls, "place_order", _tripwire)

    trade = _record(db, settings, inst, signal, features, model)
    assert trade is not None


def test_pipeline_shadow_path_has_no_broker_order_reference():
    """The live pipeline may route orders one day — today it must not, anywhere."""
    from pathlib import Path

    import app.services.pipeline as pipeline_mod

    assert "place_order" not in Path(pipeline_mod.__file__).read_text(encoding="utf-8")


def test_shadow_package_source_has_no_broker_reference():
    """Static guarantee: the shadow module cannot reach a broker even by accident."""
    from pathlib import Path

    import app.services.shadow as shadow_pkg

    for path in Path(shadow_pkg.__file__).parent.glob("*.py"):
        src = path.read_text(encoding="utf-8")
        assert "place_order" not in src, f"{path.name} references place_order"
        assert "app.brokers" not in src, f"{path.name} imports a broker module"


# ── 9. live pipeline wiring: features flow through the feature_builder chokepoint ──
def test_pipeline_observe_shadow_records_at_the_live_signal_time(
    db, settings, inst, signal, model, live_row_cleanup
):
    """End-to-end of the ADDITIVE block in run_candle_close_pipeline: derive T,
    build features via feature_builder, record. Proves the live row's features come
    from the same chokepoint (and the same T convention) the M7 corpus used."""
    from app.services import pipeline as pl

    status = pl._observe_shadow(db, settings, inst, signal, _approved(), _as_models(model, settings))

    assert status == pl.SHADOW_RECORDED
    rows = _live_shadow_rows(db, inst)
    assert len(rows) == 1
    assert rows[0].opened_at == rec.live_signal_time(db, inst, _GRAN)
    # The persisted dict is a feature_builder product, not an inline recomputation.
    for key in FEATURE_KEYS_MODEL:
        assert key in rows[0].signal_reasoning
    assert rows[0].signal_reasoning["feature_schema_version"] == FEATURE_SCHEMA_VERSION


def test_pipeline_observe_shadow_is_idempotent(db, settings, inst, signal, model, live_row_cleanup):
    from app.services import pipeline as pl

    assert pl._observe_shadow(db, settings, inst, signal, _approved(), _as_models(model, settings)) == pl.SHADOW_RECORDED
    assert pl._observe_shadow(db, settings, inst, signal, _approved(), _as_models(model, settings)) == pl.SHADOW_SKIPPED
    assert len(_live_shadow_rows(db, inst)) == 1


def test_pipeline_observe_shadow_swallows_errors(db, settings, inst, signal, model, monkeypatch):
    """A shadow failure must never propagate into the trading path."""
    from app.services import pipeline as pl

    def _boom(*_a, **_k):
        raise RuntimeError("feature build exploded")

    monkeypatch.setattr(pl, "build_features", _boom)
    assert pl._observe_shadow(db, settings, inst, signal, _approved(), _as_models(model, settings)) == pl.SHADOW_ERROR
    assert _shadow_rows(db, inst) == []


def test_pipeline_skips_shadow_when_model_is_none(db, settings, inst, signal):
    from app.services import pipeline as pl

    off = _settings_with(settings, SHADOW_MODE_ENABLED=False)
    from app.services.ml.registry import StrategyModels

    assert pl._observe_shadow(
        db, off, inst, signal, _approved(), StrategyModels(None, ())
    ) == pl.SHADOW_SKIPPED
    assert _shadow_rows(db, inst) == []


def test_load_strategy_models_returns_nothing_when_shadow_disabled(db, settings):
    from app.services import pipeline as pl

    off = _settings_with(settings, SHADOW_MODE_ENABLED=False)
    got = pl._load_strategy_models(db, off, SimpleNamespace(id=1, name="s"))
    assert got.champion is None and got.challengers == ()


def test_load_strategy_models_never_raises_when_the_registry_is_unreachable(settings):
    """A model problem disables scoring for the cycle; it never stops the pipeline.

    Passing a dead session makes the registry query itself blow up — the failure mode a
    per-model try/except would NOT catch.
    """
    from app.services import pipeline as pl

    got = pl._load_strategy_models(None, settings, SimpleNamespace(id=1, name="s"))
    assert got.champion is None and got.challengers == ()


# ── 10. D1 refresh gap ───────────────────────────────────────────────────────
def test_d1_is_scheduled_and_refresh_only():
    """Before M8-Shadow the D timeframe had no cron at all, so live D1 candles went
    stale and every D1-derived feature (dist_to_sma50_atr, the C1 trend leg) silently
    degraded on live rows."""
    from app.domain.timeframes import get_timeframe as gtf

    d1 = gtf("D")
    assert d1.cron_hours is not None, "D1 must have a scheduled refresh job"
    assert d1.fires_signals is False, "D1 must never author a signal"


def test_h4_still_fires_signals():
    from app.domain.timeframes import get_timeframe as gtf

    assert gtf("H4").fires_signals is True
    assert gtf("M1").fires_signals is False


def test_d1_refresh_runs_after_the_daily_close_and_before_the_first_h4_job():
    """OANDA's D1 bar closes at 21:00 UTC (NY DST) / 22:00 UTC (NY EST); the first H4
    candle-close job of the next day is 01:01 UTC."""
    from app.domain.timeframes import get_timeframe as gtf

    d1_hour = int(gtf("D").cron_hours)
    h4_hours = sorted(int(h) for h in gtf("H4").cron_hours.split(","))

    assert d1_hour > 22, "must run after the latest (EST) D1 close at 22:00 UTC"
    assert d1_hour > max(h4_hours), "must not precede the last H4 job of the day"
    assert d1_hour < 24, "must land on the same UTC day, before 01:01 the next day"


def test_scheduler_builds_a_refresh_job_for_non_signal_timeframes():
    from app.services import scheduler as sch

    assert callable(sch._make_candle_refresh_job)
    assert callable(sch._make_candle_close_job)
    assert callable(sch._make_candle_refresh_job("D"))


def test_refresh_pipeline_does_not_import_the_signal_engine_path():
    """run_candle_refresh_pipeline must be data-only: no engine, no risk, no shadow."""
    import inspect

    from app.services.pipeline import run_candle_refresh_pipeline

    src = inspect.getsource(run_candle_refresh_pipeline)
    for forbidden in ("get_signal_engine", "RiskEngine", "record_shadow_decision", "place_order"):
        assert forbidden not in src, f"refresh pipeline must not reference {forbidden}"


# ── 10. LIVE NaN GUARD (QA BLOCKER 1c) ──────────────────────────────────────
#
# A NaN *probability* is a scoring failure and is already handled. A NaN *feature* is
# worse and quieter: XGBoost treats NaN as a learned missing-value branch, so the model
# scores happily on a crippled vector — nothing fails, nothing warns — but the row is
# no longer comparable to the M7 corpus, where every FEATURE_KEYS_MODEL key is 100%
# populated. That is exactly how a stale FRED feed silently degraded live scoring.
#
# Rows are still WRITTEN (a hole in the corpus is worse than a flagged row); the point
# is that they are unambiguously identifiable and excludable afterwards.
def _shadow_meta(trade):
    return trade.signal_reasoning[rec.REASONING_SHADOW_KEY]


def test_clean_feature_vector_records_a_zero_nan_count(db, settings, inst, signal, features, model):
    """The healthy case, and the regression tripwire for BLOCKER 1: after the macro
    refresh the live vector has NO NaN model features, so the stored count is 0 and
    the row is directly comparable to a training row."""
    trade = _record(db, settings, inst, signal, features, model)

    meta = _shadow_meta(trade)
    assert meta[rec.NAN_MODEL_FEATURES_KEY] == 0
    assert meta[rec.NAN_MODEL_FEATURE_KEYS_KEY] == []


def test_nan_model_features_are_counted_and_named(db, settings, inst, signal, features, model):
    """A degraded vector must be self-describing: the count says HOW degraded, the key
    list says whether it was one dead series or a broad outage."""
    degraded = dict(features)
    degraded["vix"] = float("nan")
    degraded["us_2s10s"] = float("nan")

    trade = _record(db, settings, inst, signal, degraded, model)

    meta = _shadow_meta(trade)
    assert meta[rec.NAN_MODEL_FEATURES_KEY] == 2
    assert meta[rec.NAN_MODEL_FEATURE_KEYS_KEY] == ["us_2s10s", "vix"], "stable contract order"


def test_a_missing_key_counts_the_same_as_a_nan(db, settings, inst, signal, features, model):
    """An absent key and a NaN key degrade the vector identically (the encoder supplies
    NaN either way), so they must be reported identically."""
    missing = {k: v for k, v in features.items() if k != "wti"}

    trade = _record(db, settings, inst, signal, missing, model)

    assert _shadow_meta(trade)[rec.NAN_MODEL_FEATURE_KEYS_KEY] == ["wti"]


def test_a_nan_gated_feature_is_not_counted(db, settings, inst, signal, features, model):
    """Only the MODEL tier matters here — a gated feature is not a model input, so it
    cannot make a row incomparable to the training set."""
    degraded = dict(features)
    degraded["us_10y"] = float("nan")

    trade = _record(db, settings, inst, signal, degraded, model)

    assert _shadow_meta(trade)[rec.NAN_MODEL_FEATURES_KEY] == 0


def test_degraded_row_is_still_recorded_and_still_scored(db, settings, inst, signal, features, model):
    """The deliberate trade-off: a flagged row beats a hole in the corpus."""
    degraded = dict(features)
    degraded["vix"] = float("nan")

    trade = _record(db, settings, inst, signal, degraded, model)

    assert trade is not None and trade.id is not None
    assert trade.ml_decision in (inf.DECISION_TAKE, inf.DECISION_SKIP)
    assert trade.ml_probability is not None


def test_warns_when_the_nan_count_exceeds_the_configured_maximum(
    db, settings, inst, signal, features, model, caplog
):
    degraded = dict(features)
    degraded["vix"] = float("nan")

    with caplog.at_level("WARNING", logger="app.services.shadow.recorder"):
        _record(db, _settings_with(settings, SHADOW_MAX_NAN_MODEL_FEATURES=0),
                inst, signal, degraded, model)

    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("NaN model-core features" in m and "vix" in m for m in warnings)


def test_does_not_warn_when_the_count_is_within_the_configured_tolerance(
    db, settings, inst, signal, features, model, caplog
):
    """The threshold is real config, not decoration — raising it silences the warning
    while the stored count stays truthful."""
    degraded = dict(features)
    degraded["vix"] = float("nan")

    with caplog.at_level("WARNING", logger="app.services.shadow.recorder"):
        trade = _record(db, _settings_with(settings, SHADOW_MAX_NAN_MODEL_FEATURES=5),
                        inst, signal, degraded, model)

    assert not [r for r in caplog.records
                if r.levelname == "WARNING" and "NaN model-core" in r.getMessage()]
    assert _shadow_meta(trade)[rec.NAN_MODEL_FEATURES_KEY] == 1, "still recorded honestly"


def test_env_default_flags_any_nan_model_feature(settings):
    """.env ships 0: the M7 corpus has these features 100% populated, so ANY hole on a
    live row is worth an operator's attention."""
    assert settings.SHADOW_MAX_NAN_MODEL_FEATURES == 0


# ── 11. the purge fixture must not be able to delete real observations ──────
#
# QA BLOCKER 2: these fixtures ran an unbounded
# ``instrument_id = X AND stage = 'shadow'`` DELETE against the real dev database via
# the app's own SessionLocal, on live-active symbols. One pytest run would have
# irreversibly destroyed every live shadow observation ever recorded.
#
# The DECOY below is a stage='shadow' row planted OUTSIDE the sentinel band, in the
# live present, shaped exactly like a real observation. It is created and verified
# inside a single test so it can never leak, and it proves the purge is bounded:
# under the old fixture this row would be gone.
def test_purge_fixture_cannot_delete_a_shadow_row_outside_the_sentinel_band(
    db, settings, inst, signal, features, model
):
    decoy_t = datetime.utcnow().replace(microsecond=0) + timedelta(days=1)
    assert not (_T_BASE <= decoy_t <= _SENTINEL_END), "the decoy must sit outside the band"
    decoy = Trade(
        instrument_id=inst.id, direction="BUY",
        entry_price=1.1, stop_price=1.09, tp_price=1.12,
        units=1000, risk_amount=1.0, expected_pip_loss=10.0, rr_entry=2.0,
        signal_source="rule_based", stage=rec.STAGE_SHADOW, opened_at=decoy_t,
    )
    db.add(decoy)
    db.commit()
    db.refresh(decoy)
    decoy_id = decoy.id
    try:
        # Run the exact purge the autouse fixture runs, twice (setup + teardown).
        for _ in range(2):
            db.query(Trade).filter(
                Trade.instrument_id == inst.id,
                Trade.stage == rec.STAGE_SHADOW,
                Trade.opened_at >= _T_BASE,
                Trade.opened_at <= _SENTINEL_END,
            ).delete(synchronize_session=False)
            db.commit()

        survivor = db.query(Trade).filter(Trade.id == decoy_id).first()
        assert survivor is not None, (
            "the purge deleted a shadow row outside its sentinel band — this is "
            "QA BLOCKER 2 regressing; a real live observation would have been lost"
        )
        assert survivor.opened_at == decoy_t
    finally:
        # Targeted cleanup by primary key — never a stage-wide delete.
        db.query(Trade).filter(Trade.id == decoy_id).delete(synchronize_session=False)
        db.commit()


def test_sentinel_purge_still_removes_the_suites_own_rows(db, settings, inst, signal, features, model):
    """The bound must not have made the fixture useless: rows INSIDE the band still go."""
    _record(db, settings, inst, signal, features, model)
    assert len(_shadow_rows(db, inst)) == 1

    db.query(Trade).filter(
        Trade.instrument_id == inst.id,
        Trade.stage == rec.STAGE_SHADOW,
        Trade.opened_at >= _T_BASE,
        Trade.opened_at <= _SENTINEL_END,
    ).delete(synchronize_session=False)
    db.commit()

    assert _shadow_rows(db, inst) == []


# ── model_decisions: the record that makes two models comparable ─────────────
def test_a_model_decision_row_is_written_alongside_the_trade(
    db, settings, inst, signal, features, model
):
    """`trades.ml_*` holds exactly ONE opinion per signal.

    The moment a challenger scores live signals beside the champion it overwrites the
    incumbent's verdict, and "where did they disagree, and who was right?" becomes
    unanswerable — which IS the promotion decision. The columns therefore cannot be
    the record; `model_decisions` is.

    This shipped unwritten: the table existed for a week and only a one-off backfill
    ever inserted into it, so 163 live decisions produced 45 rows.
    """
    from app.models.model_decision import ModelDecision

    written = _record(db, settings, inst, signal, features, model)
    assert written is not None

    rows = db.query(ModelDecision).filter(ModelDecision.trade_id == written.id).all()
    try:
        assert len(rows) == 1, "no ModelDecision written for a scored shadow row"
        d = rows[0]
        assert d.model_id == written.ml_model_id, "decision names a different artifact"
        assert d.decision == written.ml_decision
        assert d.probability == written.ml_probability
        assert d.threshold == float(settings.ML_DECISION_THRESHOLD), (
            "the threshold must be stored with the verdict — a probability alone "
            "cannot be re-evaluated later without knowing what it was compared against"
        )
        # Exactly one verdict per trade may claim to have governed it.
        assert d.is_authoritative is True
    finally:
        for r in rows:
            db.delete(r)
        db.commit()


# ── challengers: a second model scoring the same signal ──────────────────────
def _challenger_path():
    """A promoted artifact on the same feature schema as the champion.

    v1 is schema 1 (20 features) and would be rejected by the feature-contract check,
    which is correct behaviour and the wrong thing to test here.
    """
    from pathlib import Path

    models = Path(__file__).resolve().parents[1] / "models"
    for p in sorted(models.glob("s1_model_v*_schema2_*.joblib")):
        return p
    return None


def test_challenger_records_its_own_verdict_without_governing(
    db, settings, inst, signal, features, model
):
    """A challenger must record an opinion and change nothing else.

    Running a candidate beside the incumbent on identical live signals is the only way
    to answer "where did they disagree, and who was right?" — which is what promotion
    turns on. It is worth doing precisely because it is free of consequence: the
    challenger must not touch trades.ml_*, must not alter take/skip, and must never be
    able to cause an order.
    """
    from app.models.model_decision import ModelDecision

    path = _challenger_path()
    if path is None:
        pytest.skip("no schema-2 artifact available to act as a challenger")

    challenger = inf.load_artifact(path, settings)
    written = _record(db, settings, inst, signal, features, model, challengers=(challenger,))
    assert written is not None

    rows = db.query(ModelDecision).filter(ModelDecision.trade_id == written.id).all()
    try:
        assert len(rows) == 2, f"expected champion + challenger, got {len(rows)}"
        authoritative = [r for r in rows if r.is_authoritative]
        challengers = [r for r in rows if not r.is_authoritative]
        assert len(authoritative) == 1, "exactly one verdict may claim to have governed"
        assert len(challengers) == 1

        # The row itself still reflects ONLY the champion.
        assert written.ml_model_id == authoritative[0].model_id
        assert written.ml_decision == authoritative[0].decision
        assert written.ml_probability == authoritative[0].probability

        # Each verdict records the cut-off that actually produced it. Judging a
        # challenger at the champion's threshold would measure the threshold rather
        # than the model — v2's is 0.236 and v3's 0.343.
        assert authoritative[0].threshold == float(settings.ML_DECISION_THRESHOLD)
        assert challengers[0].model_id != authoritative[0].model_id
        assert challengers[0].threshold == float(
            challenger.metadata["deployment_threshold"]
        ), "a challenger must be judged at its OWN cut-off, not the champion's"
    finally:
        for r in rows:
            db.delete(r)
        db.commit()


def test_a_broken_challenger_never_breaks_the_live_path(
    db, settings, inst, signal, features, champion_artifact
):
    """A challenger that will not load is a research inconvenience. The champion still
    has to score the signal and the row still has to be written — otherwise adding a
    challenger would be strictly more dangerous than not bothering.

    The registry is what drops it now: ``models_for_strategy`` logs the failure and
    returns the models it COULD load, so a broken row never reaches the recorder.
    """
    from app.models.model_decision import ModelDecision
    from app.services.ml import registry

    missing = registry._load(
        SimpleNamespace(
            model_id="does_not_exist", strategy_id=1, status="shadow", decision_threshold=0.3
        ),
        settings, is_champion=False,
    )
    assert missing is None, "an unloadable artifact must be dropped, not raised"

    written = _record(db, settings, inst, signal, features, champion_artifact)

    assert written is not None, "a missing challenger took down the live write path"
    assert written.ml_decision is not None, "the champion's verdict was lost"
    rows = db.query(ModelDecision).filter(ModelDecision.trade_id == written.id).all()
    try:
        assert len(rows) == 1 and rows[0].is_authoritative
    finally:
        for r in rows:
            db.delete(r)
        db.commit()


def test_a_strategy_with_no_champion_records_a_rule_only_row(db, settings, inst, signal, features):
    """A strategy whose first model is not yet trained must still collect evidence.

    Recording nothing would mean a new strategy is invisible until a model exists — but
    the signal and its resolved outcome are exactly what that model will be trained on.
    The row is written with ml_* NULL and no ModelDecision.
    """
    from app.models.model_decision import ModelDecision
    from app.services.ml.registry import StrategyModels

    written = _record(
        db, settings, inst, signal, features,
        StrategyModels(champion=None, challengers=()),
    )
    try:
        assert written is not None, "a strategy without a model recorded nothing"
        assert written.ml_model_id is None
        assert written.ml_decision is None
        assert written.ml_probability is None
        assert written.confluence_score is not None, "the rule's own signal must survive"
        assert db.query(ModelDecision).filter(
            ModelDecision.trade_id == written.id
        ).count() == 0, "no model opined, so no verdict may be recorded"
    finally:
        if written is not None:
            db.delete(written)
            db.commit()
