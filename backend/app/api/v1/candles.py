from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.brokers.router import BrokerRouter, get_broker_router
from app.config import Settings, get_settings
from app.database import get_db
from app.models.instrument import Instrument
from app.schemas.candle import CandleFetchResponse, CandleRead
from app.services import candle_service

_SUPPORTED_GRANULARITIES = {"H4", "D"}

router = APIRouter()


@router.get("/{instrument}", response_model=list[CandleRead])
def get_candles(
    instrument: str,
    granularity: str = Query(...),
    limit: int = Query(default=500, ge=1),
    db: Session = Depends(get_db),
):
    if granularity not in _SUPPORTED_GRANULARITIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported granularity '{granularity}'. Supported: {sorted(_SUPPORTED_GRANULARITIES)}",
        )
    inst = db.query(Instrument).filter_by(symbol=instrument).first()
    if inst is None:
        raise HTTPException(
            status_code=404,
            detail=f"Instrument '{instrument}' not found. Run POST /api/v1/instruments/sync first.",
        )
    return candle_service.get_stored_candles(instrument, granularity, limit, db)


@router.post("/{instrument}/fetch", response_model=CandleFetchResponse)
def fetch_candles(
    instrument: str,
    granularity: str = Query(...),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    broker_router: BrokerRouter = Depends(get_broker_router),
):
    """Fetch and store historical candles from broker. Incremental — skips existing."""
    if granularity not in _SUPPORTED_GRANULARITIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported granularity '{granularity}'. Supported: {sorted(_SUPPORTED_GRANULARITIES)}",
        )

    inst = db.query(Instrument).filter_by(symbol=instrument).first()
    if inst is None:
        raise HTTPException(
            status_code=404,
            detail=f"Instrument '{instrument}' not found. Run POST /api/v1/instruments/sync first.",
        )

    try:
        stored = candle_service.fetch_and_store(
            instrument, granularity, db, settings, broker_router
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    total = candle_service.count_stored_candles(inst.id, granularity, db)
    return CandleFetchResponse(
        instrument=instrument,
        granularity=granularity,
        stored=stored,
        total=total,
    )
