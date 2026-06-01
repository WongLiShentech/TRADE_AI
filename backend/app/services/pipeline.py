"""
Candle-close pipeline. Called by the APScheduler at H4 / D1 close times.

For each active instrument:
  1. Fetch latest candle from broker -> DB (extending M2 data)
  2. Compute indicator row for the new candle (M4 logic)
  3. Run SignalEngine.evaluate() -> maybe produce a SignalOutput
  4. Run RiskEngine.validate() -> persist APPROVED signal or persist REJECTED with reason

Per-instrument errors are logged but never halt the pipeline.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from app.brokers.router import BrokerRouter
from app.config import Settings
from app.database import SessionLocal
from app.models.instrument import Instrument
from app.models.signal import Signal
from app.services import candle_service, indicator_service
from app.services.risk_engine import RiskEngine, RiskValidationError, ValidatedSignal
from app.services.signal_engine.base import SignalOutput
from app.services.signal_engine.factory import get_signal_engine

logger = logging.getLogger(__name__)

_EXPIRY_HOURS = {"H4": 4, "D": 24}


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
    }
    try:
        # 1. Expire stale PENDING signals
        expired = expire_stale_signals(db)
        summary["expired"] = expired

        # 2. Per-instrument pipeline
        active = db.query(Instrument).filter_by(is_active=True).all()
        engine = get_signal_engine(settings)
        router = BrokerRouter(settings)
        for inst in active:
            try:
                fetched = candle_service.fetch_and_store_latest(inst.symbol, granularity, db, settings)
                summary["fetched"] += fetched

                computed = indicator_service.compute_and_store(inst.symbol, granularity, db, settings)
                summary["computed"] += computed

                output = engine.evaluate(inst.symbol, granularity, db, settings)
                summary["evaluated"] += 1
                if output is not None:
                    try:
                        broker = router.for_instrument(inst.symbol, db)
                        risk_engine = RiskEngine(settings, broker)
                        validated = risk_engine.validate(output, settings.STARTING_BALANCE, db, inst)
                        persist_signal(validated, inst.id, db)
                        summary["signals"] += 1
                        logger.info(
                            "signal approved: %s %s %s score=%d units=%d",
                            output.instrument, output.granularity, output.direction,
                            output.confidence_score, validated.units,
                        )
                    except RiskValidationError as exc:
                        _persist_rejected(output, inst.id, str(exc), db)
                        summary["rejected"] = summary.get("rejected", 0) + 1
                        logger.info("signal rejected: %s reason=%s", inst.symbol, str(exc))
            except Exception as exc:
                summary["errors"] += 1
                logger.warning("pipeline error for %s: %s", inst.symbol, exc, exc_info=False)
    finally:
        db.close()
    logger.info("pipeline complete: %s", summary)
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
    expiry_hours = _EXPIRY_HOURS.get(validated.signal.granularity, 4)
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
    expiry_hours = _EXPIRY_HOURS.get(output.granularity, 4)
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
