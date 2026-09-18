"""
Candle-close pipeline. Called by the APScheduler at H4 / D1 close times.

Two entry points, selected by the Timeframe registry's ``fires_signals`` flag:

``run_candle_close_pipeline`` (``fires_signals=True`` — H4)
  For each active instrument:
    1. Fetch latest candle from broker -> DB (extending M2 data)
    2. Compute indicator row for the new candle (M4 logic)
    3. Run SignalEngine.evaluate() -> maybe produce a SignalOutput
    4. Run RiskEngine.validate() -> persist APPROVED signal or persist REJECTED with reason
    5. M8-Shadow: build the LOCKED feature contract for the signal and record an
       observe-only ``stage='shadow'`` Trade row with the model's decision

``run_candle_refresh_pipeline`` (``fires_signals=False`` — D1)
  Steps 1-2 only. D1 authors no signals, but its candles feed the C1 trend filter
  and the D1-derived features, so they must not go stale.

``run_trailing_window_refresh`` (``fires_signals=False`` + ``trailing_window_setting``
— M1)
  Re-fetches a bounded ``[now - lookback, now]`` window of every price series the
  registry declares for the timeframe, instead of resuming incrementally from the
  newest stored bar. M1 Bid+Ask is the series the M8-Shadow outcome resolver walks
  to decide whether a live shadow trade hit its stop or its target (GAP-13).

Per-instrument errors are logged but never halt the pipeline.

M8-Shadow feature parity (Phase 2)
----------------------------------
Shadow rows MUST carry features produced by the SAME chokepoint the M7 training
corpus used, or the experiment compares two different things. So the live path calls
``feature_builder.build_features(inst, T, granularity, signal.score_breakdown, db,
settings)`` — exactly as ``backtester/runner.py`` does — with T derived by
``shadow.live_signal_time`` as ``decision_bar_close + 1 second`` (candles.timestamp
is bar-OPEN; feature_builder filters ``timestamp <= T``, so T must be STRICTLY after
the close). The engine's fire/no-fire logic is untouched: the shadow block is purely
additive and its failures are swallowed per-signal.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
from sqlalchemy.orm import Session

from app.brokers.router import BrokerRouter
from app.config import Settings
from app.database import SessionLocal
from app.models.instrument import Instrument
from app.models.signal import Signal
from app.services import candle_service, indicator_service
from app.services.feature_builder import build_features
from app.services.ml.registry import StrategyModels, models_for_strategy
from app.services.risk_engine import RiskEngine, RiskValidationError, ValidatedSignal
from app.services.sandbox import executor as sandbox
from app.services.shadow import RiskAssessment, live_signal_time, record_shadow_decision
from app.services.signal_engine.base import SignalOutput
from app.services.signal_engine.factory import get_signal_engine
from app.services.strategy_registry import (
    active_strategies,
    resolve_active_strategy,
    settings_for as strategy_settings,
)
from app.domain.timeframes import get_timeframe

logger = logging.getLogger(__name__)

# Outcomes of one shadow-observation attempt (summary counters only — never control
# flow for the trading path).
SHADOW_RECORDED = "recorded"   # a stage='shadow' row was written
SHADOW_SKIPPED = "skipped"     # shadow disabled, model unloadable, or already recorded
SHADOW_ERROR = "error"         # feature build / recording raised; trading unaffected

# Rejection reason used when RiskEngine could not COMPLETE its verdict because the
# broker pip-value call failed (HTTP error, timeout, malformed/absent pricing
# payload) rather than because the signal breached a risk rule.
#
# Why this is a distinct reason and not just an error: RiskEngine.validate fetches
# pip value from the broker (never hardcoded — locked-triangle rule), so a transient
# network blip inside validate() used to escape as an un-typed exception, hit the
# per-instrument `except Exception` below, and take the ENTIRE signal down with it —
# including the M8-Shadow observation, which is a permanently lost data point in a
# corpus that only grows a few rows a week. Converting it to a rejected
# RiskAssessment keeps the observation, and the distinct reason string keeps it
# separable from genuine risk rejections during analysis.
RISK_REASON_PIP_VALUE_UNAVAILABLE = "PIP_VALUE_UNAVAILABLE"


def run_candle_close_pipeline(granularity: str, settings: Settings) -> dict:
    db: Session = SessionLocal()
    summary = {
        "granularity": granularity,
        "fetched": 0,
        "computed": 0,
        "signals": 0,
        "rejected": 0,
        "errors": 0,
        "evaluated": 0,
        "shadow_recorded": 0,
        "shadow_errors": 0,
    }
    try:
        # 1. Expire stale PENDING signals
        expired = expire_stale_signals(db)
        summary["expired"] = expired

        # 2. Models are resolved PER STRATEGY, below — a model is valid only for the
        #    strategy whose outcomes taught it, so there is no such thing as "the"
        #    artifact for a run. Artifacts stay process-cached inside inference, so
        #    resolving per strategy costs a registry query, not a deserialization.

        # 3. Per-instrument pipeline, evaluated once PER ACTIVE STRATEGY.
        #
        # Candle fetch and indicator computation are strategy-INDEPENDENT and stay
        # outside the strategy loop: they describe the market, not a configuration,
        # and running them per strategy would re-fetch identical bars N times.
        active = db.query(Instrument).filter_by(is_active=True).all()
        router = BrokerRouter(settings)

        strategies = active_strategies(db)
        if not strategies:
            # Fall back to the running configuration rather than evaluating nothing.
            # An empty registry is a bookkeeping problem; silently trading nothing
            # because of one is a worse failure than the one it replaced.
            fallback = resolve_active_strategy(db, settings)
            strategies = [fallback] if fallback is not None else []
            logger.warning(
                "no strategy is shadow/live — falling back to the running config (%s)",
                fallback.name if fallback else "unresolvable",
            )

        # Several strategies in `shadow` is free — they only record opinions. Several
        # `live` is NOT: each sizes independently against the same balance, so a shared
        # view is taken at N times the intended risk. Capital allocation across
        # strategies does not exist, so this refuses rather than silently doubling.
        live_count = sum(1 for st in strategies if st.status == "live")
        if live_count > 1:
            raise RuntimeError(
                f"{live_count} strategies have status='live'. Each sizes independently "
                f"against the same balance, so a shared view would be taken at "
                f"{live_count}x the intended risk. Capital allocation across strategies "
                f"is not implemented — set all but one to 'shadow'."
            )

        # Each strategy carries its own engine, its own projected settings, and its own
        # models. Resolved once per run rather than per instrument: the registry does not
        # change mid-pass, and a per-instrument query would be 10x the work for the same
        # answer.
        engines = [
            (st, get_signal_engine(strategy_settings(settings, st)),
             strategy_settings(settings, st),
             _load_strategy_models(db, settings, st))
            for st in strategies
        ]
        summary["strategies"] = len(engines)
        summary["shadow_enabled"] = any(m.champion is not None for _, _, _, m in engines)

        for inst in active:
            try:
                fetched = candle_service.fetch_and_store_latest(inst.symbol, granularity, db, settings)
                summary["fetched"] += fetched

                computed = indicator_service.compute_and_store(inst.symbol, granularity, db, settings)
                summary["computed"] += computed

                for strategy, engine, s_settings, s_models in engines:
                  # One strategy failing must not cost the others their evaluation of
                  # this bar — they are independent observers of the same market.
                  try:
                    output = engine.evaluate(inst.symbol, granularity, db, s_settings)
                    summary["evaluated"] += 1
                    if output is not None:
                        risk_amount = s_settings.STARTING_BALANCE * s_settings.RISK_PCT_PER_TRADE
                        risk: RiskAssessment
                        try:
                            broker = router.for_instrument(inst.symbol, db)
                            risk_engine = RiskEngine(s_settings, broker)
                            validated = risk_engine.validate(output, s_settings.STARTING_BALANCE, db, inst)
                            persist_signal(validated, inst.id, db)
                            summary["signals"] += 1
                            risk = RiskAssessment.approved(validated)
                            logger.info(
                                "signal approved: %s %s %s score=%d units=%d",
                                output.instrument, output.granularity, output.direction,
                                output.confidence_score, validated.units,
                            )
                        except RiskValidationError as exc:
                            _persist_rejected(output, inst.id, str(exc), db)
                            summary["rejected"] = summary.get("rejected", 0) + 1
                            risk = RiskAssessment.rejected(str(exc), risk_amount)
                            logger.info("signal rejected: %s reason=%s", inst.symbol, str(exc))
                        except (httpx.HTTPError, OSError, ValueError) as exc:
                            # Broker/transport failure inside RiskEngine.validate — NOT a
                            # risk-rule breach. Degrade to a rejected assessment so the
                            # signal is still persisted and the shadow row is still
                            # recorded; letting this reach the outer handler would discard
                            # the observation entirely.
                            #   httpx.HTTPError  — connect/read/status errors from OANDA
                            #   OSError          — DNS/socket-level failures
                            #   ValueError       — OandaClient.get_pip_value raises this for
                            #                      an empty `prices` array or a missing home
                            #                      conversion, i.e. a bad broker payload
                            db.rollback()
                            summary["rejected"] = summary.get("rejected", 0) + 1
                            risk = RiskAssessment.rejected(
                                RISK_REASON_PIP_VALUE_UNAVAILABLE, risk_amount
                            )
                            _persist_rejected(
                                output, inst.id, RISK_REASON_PIP_VALUE_UNAVAILABLE, db
                            )
                            logger.error(
                                "risk validation could not complete for %s — broker pip-value "
                                "call failed (%s: %s); signal recorded as %s so the shadow "
                                "observation is not lost",
                                inst.symbol, type(exc).__name__, exc,
                                RISK_REASON_PIP_VALUE_UNAVAILABLE,
                            )

                        # M8-Shadow — ADDITIVE. Never influences the decision above and
                        # never propagates an error into the trading path.
                        shadow_status = _observe_shadow(db, s_settings, inst, output, risk, s_models)
                        if shadow_status == SHADOW_RECORDED:
                            summary["shadow_recorded"] += 1
                        elif shadow_status == SHADOW_ERROR:
                            summary["shadow_errors"] += 1
                  except Exception:
                    # Per-STRATEGY isolation. The outer handler rolls back and abandons
                    # the whole instrument — right for a broker or candle failure, wrong
                    # for one strategy's engine raising, since the others observed the
                    # same bar perfectly well and must still get to record it.
                    db.rollback()
                    summary["strategy_errors"] = summary.get("strategy_errors", 0) + 1
                    logger.exception(
                        "strategy %s (%s) failed on %s — other strategies unaffected",
                        strategy.id, strategy.name, inst.symbol,
                    )
            except Exception as exc:
                # Roll back before moving to the next instrument. The session is
                # SHARED across the whole loop, so a failed flush leaves it in a
                # broken state and EVERY subsequent instrument would then fail on an
                # unrelated PendingRollbackError — one bad pair silently taking down
                # the entire candle-close run.
                db.rollback()
                summary["errors"] += 1
                logger.warning("pipeline error for %s: %s", inst.symbol, exc, exc_info=False)
    finally:
        db.close()
    logger.info("pipeline complete: %s", summary)
    return summary


def run_candle_refresh_pipeline(granularity: str, settings: Settings) -> dict:
    """Data-refresh-only pass for a timeframe that authors no signals (D1).

    Fetches the latest candles for every price series the Timeframe registry declares
    (``Timeframe.price_types``) and recomputes indicators when the registry says this
    timeframe carries them, then stops — no signal engine, no risk engine, no shadow
    row, no broker write.

    Why this exists: D1 candles are LIVE INPUTS to the H4 decision (the C1 trend
    filter) and to the feature contract (``dist_to_sma50_atr``, ``d1_close``,
    ``d1_sma50``). Without a scheduled D1 job those candles go stale between manual
    ingests and every D1-derived feature on a live/shadow row silently degrades.

    Args:
        granularity: the timeframe code to refresh (e.g. ``"D"``).
        settings: config.

    Returns:
        A summary dict ``{granularity, fetched, computed, instruments, errors}``.

    Raises:
        ValueError: ``granularity`` is not in the Timeframe registry (fail loud).
    """
    tf = get_timeframe(granularity)
    db: Session = SessionLocal()
    summary = {
        "granularity": granularity,
        "fetched": 0,
        "computed": 0,
        "instruments": 0,
        "errors": 0,
    }
    try:
        active = db.query(Instrument).filter_by(is_active=True).all()
        summary["instruments"] = len(active)
        for inst in active:
            try:
                for price_type in tf.price_types:
                    summary["fetched"] += candle_service.fetch_and_store_latest(
                        inst.symbol, granularity, db, settings, price_type=price_type
                    )
                if tf.computes_indicators:
                    summary["computed"] += indicator_service.compute_and_store(
                        inst.symbol, granularity, db, settings
                    )
            except Exception as exc:
                # See run_candle_close_pipeline: the session is shared, so a failure
                # must be rolled back or it poisons every later instrument.
                db.rollback()
                summary["errors"] += 1
                logger.warning(
                    "%s refresh error for %s: %s", granularity, inst.symbol, exc, exc_info=False
                )
    finally:
        db.close()
    logger.info("candle refresh complete: %s", summary)
    return summary


def run_trailing_window_refresh(
    granularity: str, settings: Settings, *, now: Optional[datetime] = None
) -> dict:
    """Bounded trailing-window top-up for a high-volume timeframe (M1 Bid+Ask).

    Re-fetches ``[now - lookback, now]`` for every price series the registry declares
    on this timeframe, where ``lookback`` is read from the Settings attribute the
    registry names in ``Timeframe.trailing_window_setting`` (hours). Insertion is
    ``ON CONFLICT DO NOTHING`` (``candle_service.fetch_and_store_window``), so the
    deliberate overlap between consecutive runs is free and makes short outages
    self-healing.

    Why a fixed window instead of the incremental resume the other timeframes use:
    an incremental fetch resumes from the newest stored bar, so after any outage the
    next scheduled M1 run would attempt an unbounded backfill (months of minute bars,
    thousands of broker calls) inside a cron job. A trailing window has a constant,
    predictable per-run cost whatever the gap. A genuinely long gap is an operational
    backfill (``scripts/ingest_bid_ask.py``), not a cron job's business.

    Args:
        granularity: timeframe code to refresh (e.g. ``"M1"``).
        settings: config; supplies the lookback named by the registry entry.
        now: window right edge; defaults to ``datetime.now(timezone.utc)``. Injectable
            so a caller (or a test) can pin the window without patching the clock.

    Returns:
        ``{granularity, price_types, window_start, window_end, lookback_hours,
        fetched, instruments, errors}``.

    Raises:
        ValueError: ``granularity`` is not in the Timeframe registry, or the registry
            entry declares no ``trailing_window_setting`` (this entry point is only
            meaningful for a trailing-window timeframe — fail loud rather than
            silently defaulting to some window).
    """
    from app.brokers.router import get_broker_router

    tf = get_timeframe(granularity)
    if tf.trailing_window_setting is None:
        raise ValueError(
            f"timeframe '{granularity}' declares no trailing_window_setting — use "
            f"run_candle_refresh_pipeline (incremental) instead, or add the setting "
            f"name to its TIMEFRAMES entry."
        )
    lookback_hours = int(getattr(settings, tf.trailing_window_setting))
    end = now or datetime.now(timezone.utc)
    start = end - timedelta(hours=lookback_hours)

    db: Session = SessionLocal()
    summary = {
        "granularity": granularity,
        "price_types": list(tf.price_types),
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "lookback_hours": lookback_hours,
        "fetched": 0,
        "instruments": 0,
        "errors": 0,
    }
    try:
        router = get_broker_router()
        active = db.query(Instrument).filter_by(is_active=True).all()
        summary["instruments"] = len(active)
        for inst in active:
            for price_type in tf.price_types:
                try:
                    summary["fetched"] += candle_service.fetch_and_store_window(
                        instrument_symbol=inst.symbol,
                        granularity=granularity,
                        start=start,
                        end=end,
                        db=db,
                        settings=settings,
                        broker_router=router,
                        price_type=price_type,
                    )
                except Exception as exc:  # noqa: BLE001 — one pair must not stall the rest
                    # Shared session: roll back so the next (instrument, price_type)
                    # starts from a clean transaction rather than inheriting this
                    # one's failure.
                    db.rollback()
                    summary["errors"] += 1
                    logger.warning(
                        "%s/%s trailing refresh error for %s: %s",
                        granularity, price_type, inst.symbol, exc, exc_info=False,
                    )
    finally:
        db.close()
    logger.info("trailing window refresh complete: %s", summary)
    return summary


def expire_stale_signals(db: Session) -> int:
    now = datetime.utcnow()
    result = (
        db.query(Signal)
        .filter(Signal.status == "PENDING")
        .filter(Signal.expires_at < now)
        .update({"status": "EXPIRED"}, synchronize_session=False)
    )
    db.commit()
    return result


def persist_signal(validated: ValidatedSignal, instrument_id: int, db: Session) -> Signal:
    now = datetime.utcnow()
    expiry_hours = get_timeframe(validated.signal.granularity).expiry_hours
    signal = Signal(
        instrument_id=instrument_id,
        granularity=validated.signal.granularity,
        direction=validated.signal.direction,
        entry=validated.signal.entry,
        stop=validated.signal.stop,
        target=validated.signal.target,
        confidence_score=validated.signal.confidence_score,
        score_breakdown=validated.signal.score_breakdown,
        status="APPROVED",
        created_at=now,
        expires_at=now + timedelta(hours=expiry_hours),
    )
    db.add(signal)
    db.commit()
    db.refresh(signal)
    return signal


def _persist_rejected(output: SignalOutput, instrument_id: int, reason: str, db: Session) -> Signal:
    now = datetime.utcnow()
    expiry_hours = get_timeframe(output.granularity).expiry_hours
    signal = Signal(
        instrument_id=instrument_id,
        granularity=output.granularity,
        direction=output.direction,
        entry=output.entry,
        stop=output.stop,
        target=output.target,
        confidence_score=output.confidence_score,
        score_breakdown=output.score_breakdown,
        status="REJECTED",
        rejection_reason=reason,
        created_at=now,
        expires_at=now + timedelta(hours=expiry_hours),
    )
    db.add(signal)
    db.commit()
    db.refresh(signal)
    return signal


# ── M8-Shadow wiring (additive; isolated from the trading path) ──────────────
def _load_strategy_models(db: Session, settings: Settings, strategy) -> StrategyModels:
    """The champion and challengers registered for one strategy.

    Resolution failures never stop the rule engine: ``models_for_strategy`` already
    swallows per-model load errors, and anything it cannot handle (a registry query
    failing outright) degrades to "no models", which records rule-only rows rather than
    abandoning the strategy's observation of the bar.
    """
    if not settings.SHADOW_MODE_ENABLED:
        return StrategyModels(champion=None, challengers=())
    try:
        return models_for_strategy(db, settings, strategy.id)
    except Exception as exc:  # noqa: BLE001 — model problems never stop live trading
        logger.error(
            "could not resolve models for strategy %s (%s) (%s: %s) — recording "
            "rule-only rows this cycle",
            strategy.id, strategy.name, type(exc).__name__, exc,
        )
        return StrategyModels(champion=None, challengers=())


def _observe_shadow(
    db: Session,
    settings: Settings,
    inst: Instrument,
    output: SignalOutput,
    risk: RiskAssessment,
    models: StrategyModels,
) -> str:
    """Build live features and record one shadow row.

    Feature parity is the whole point: T comes from ``live_signal_time`` (decision-bar
    close + 1s) and the dict comes from ``feature_builder.build_features`` — the same
    call, same argument order and same timestamp convention the M7 runner used to
    generate the training corpus.

    Returns:
        :data:`SHADOW_RECORDED`, :data:`SHADOW_SKIPPED` or :data:`SHADOW_ERROR`.
        Never raises — the caller's trading decision is already persisted by now.
    """
    if not settings.SHADOW_MODE_ENABLED:
        return SHADOW_SKIPPED
    try:
        signal_time = live_signal_time(db, inst, output.granularity)
        if signal_time is None:
            logger.warning(
                "shadow: no %s Mid candle for %s — cannot derive signal_time, skipping",
                output.granularity, inst.symbol,
            )
            return SHADOW_SKIPPED
        features = build_features(
            inst, signal_time, output.granularity, output.score_breakdown, db, settings
        )
        # ONE write path, one scoring call. The recorder derives the stage from the
        # decision + risk verdict + execution mode, so the row already says truthfully
        # whether an order is about to be sent.
        trade = record_shadow_decision(
            db, settings, inst, output, features, models,
            signal_time=signal_time, risk=risk,
        )
        # Row exists BEFORE the ticket — a position with no row would be invisible to
        # this platform and would never receive its time exit. See sandbox.executor.
        if trade is not None and trade.stage == sandbox.STAGE_SANDBOX:
            sandbox.place_for_trade(db, settings, trade, inst, output)
        return SHADOW_RECORDED if trade is not None else SHADOW_SKIPPED
    except Exception as exc:  # noqa: BLE001 — shadow observation is never load-bearing
        db.rollback()
        logger.warning(
            "shadow observation failed for %s (%s: %s) — trading path unaffected",
            inst.symbol, type(exc).__name__, exc,
        )
        return SHADOW_ERROR
