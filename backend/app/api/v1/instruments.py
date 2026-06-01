from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.instrument import InstrumentListResponse
from app.models.instrument import Instrument
from app.brokers.router import BrokerRouter, get_broker_router

router = APIRouter()


@router.get("", response_model=InstrumentListResponse)
def list_instruments(db: Session = Depends(get_db)):
    instruments = db.query(Instrument).order_by(Instrument.symbol).all()
    breakdown = (
        db.query(Instrument.asset_class, func.count(Instrument.id))
        .group_by(Instrument.asset_class)
        .all()
    )
    return InstrumentListResponse(
        total=len(instruments),
        by_asset_class={asset_class: count for asset_class, count in breakdown},
        instruments=instruments,
    )


@router.post("/sync", response_model=dict)
def sync_instruments(
    db: Session = Depends(get_db),
    broker_router: BrokerRouter = Depends(get_broker_router),
):
    """Fetch all instruments from every configured broker and upsert into DB."""
    synced: list[str] = []
    for client in broker_router.all_clients().values():
        try:
            instruments = client.get_instruments()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        for inst in instruments:
            existing = db.query(Instrument).filter_by(symbol=inst.symbol).first()
            if existing:
                existing.display_name = inst.display_name
                existing.pip_size = inst.pip_size
                existing.pip_location = inst.pip_location
                existing.asset_class = inst.asset_class
                existing.broker_id = inst.broker_id
                # is_active preserved — never overwritten on re-sync
            else:
                db.add(
                    Instrument(
                        symbol=inst.symbol,
                        display_name=inst.display_name,
                        pip_size=inst.pip_size,
                        pip_location=inst.pip_location,
                        asset_class=inst.asset_class,
                        broker_id=inst.broker_id,
                        is_active=True,
                    )
                )
            synced.append(inst.symbol)

    db.commit()
    return {"synced": len(synced)}
