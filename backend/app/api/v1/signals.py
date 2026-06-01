"""
Signals API.

GET  /api/v1/signals                       list signals (filter by instrument, status, granularity, since, limit)
POST /api/v1/signals/evaluate/{instrument} manually trigger pipeline for one instrument
"""
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.brokers.router import BrokerRouter
from app.config import Settings, get_settings
from app.database import get_db
from app.models.instrument import Instrument
from app.models.signal import Signal
from app.schemas.signal import EvaluateResponse, SignalRead
from app.services import candle_service, indicator_service
from app.services.pipeline import _persist_rejected, persist_signal
from app.services.risk_engine import RiskEngine, RiskValidationError
from app.services.signal_engine.factory import get_signal_engine

router = APIRouter()


@router.get("", response_model=list[SignalRead])
def list_signals(
    instrument: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    granularity: Optional[str] = Query(None),
    since: Optional[datetime] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    q = db.query(Signal)
    if instrument:
        inst = db.query(Instrument).filter_by(symbol=instrument).first()
        if inst is None:
            raise HTTPException(status_code=404, detail=f"Instrument not found: {instrument}")
        q = q.filter(Signal.instrument_id == inst.id)
    if status:
        q = q.filter(Signal.status == status)
    if granularity:
        q = q.filter(Signal.granularity == granularity)
    if since:
        q = q.filter(Signal.created_at >= since)
    return q.order_by(Signal.created_at.desc()).limit(limit).all()


@router.post("/evaluate/{instrument}", response_model=EvaluateResponse)
def evaluate_instrument(
    instrument: str,
    granularity: str = Query("H4"),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    allowed = {g.strip() for g in settings.SIGNAL_GRANULARITIES.split(",") if g.strip()}
    if granularity not in allowed:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Granularity '{granularity}' is not enabled for signal evaluation. "
                f"Allowed: {sorted(allowed)}. D1 is used as a trend filter only."
            ),
        )

    inst = db.query(Instrument).filter_by(symbol=instrument).first()
    if inst is None:
        raise HTTPException(status_code=404, detail=f"Instrument not found: {instrument}")

    # Top up candles + indicators so the engine has fresh data.
    try:
        candle_service.fetch_and_store_latest(instrument, granularity, db, settings)
        indicator_service.compute_and_store(instrument, granularity, db, settings)
        # Also top up trend timeframe (D) so the SMA gate can run.
        if granularity != settings.SIGNAL_TREND_TIMEFRAME:
            candle_service.fetch_and_store_latest(
                instrument, settings.SIGNAL_TREND_TIMEFRAME, db, settings
            )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"data fetch failed: {exc}")

    engine = get_signal_engine(settings)
    output = engine.evaluate(instrument, granularity, db, settings)

    if output is None:
        return EvaluateResponse(
            instrument=instrument,
            granularity=granularity,
            signal_fired=False,
            signal=None,
            reason="no signal — gate failed, insufficient data, or below confluence threshold",
        )

    try:
        broker = BrokerRouter(settings).for_instrument(instrument, db)
        risk_engine = RiskEngine(settings, broker)
        validated = risk_engine.validate(output, settings.STARTING_BALANCE, db, inst)
        signal = persist_signal(validated, inst.id, db)
        return EvaluateResponse(
            instrument=instrument,
            granularity=granularity,
            signal_fired=True,
            signal=SignalRead.model_validate(signal),
            reason=f"score={output.confidence_score} direction={output.direction}",
            units=validated.units,
            risk_amount=round(validated.risk_amount, 4),
        )
    except RiskValidationError as exc:
        _persist_rejected(output, inst.id, str(exc), db)
        return EvaluateResponse(
            instrument=instrument,
            granularity=granularity,
            signal_fired=False,
            signal=None,
            reason=str(exc),
        )
