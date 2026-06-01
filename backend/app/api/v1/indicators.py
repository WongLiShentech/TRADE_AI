from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.database import get_db
from app.models.instrument import Instrument
from app.schemas.indicator import IndicatorComputeResponse, IndicatorRead
from app.services import indicator_service

_SUPPORTED_GRANULARITIES = {"H4", "D"}

router = APIRouter()


@router.get("/{instrument}", response_model=list[IndicatorRead])
def get_indicators(
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
    return indicator_service.get_stored_indicators(instrument, granularity, limit, db)


@router.post("/{instrument}/compute", response_model=IndicatorComputeResponse)
def compute_indicators(
    instrument: str,
    granularity: str = Query(...),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
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

    try:
        computed = indicator_service.compute_and_store(instrument, granularity, db, settings)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    total = indicator_service.count_stored_indicators(inst.id, granularity, db)
    return IndicatorComputeResponse(
        instrument=instrument,
        granularity=granularity,
        computed=computed,
        total=total,
    )
