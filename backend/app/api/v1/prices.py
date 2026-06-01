"""
Live price endpoints. Reads from the in-memory price stream cache populated
by the background stream task. No direct broker calls happen here.
"""
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.instrument import Instrument
from app.schemas.tick import TickRead
from app.services.price_stream import get_all_prices, get_latest_price

logger = logging.getLogger(__name__)

router = APIRouter()


def _build_tick_read(tick, pip_size: float | None) -> TickRead:
    mid = (tick.bid + tick.ask) / 2
    spread_pips = (tick.ask - tick.bid) / pip_size if pip_size else None
    return TickRead(
        instrument=tick.instrument,
        bid=tick.bid,
        ask=tick.ask,
        mid=mid,
        spread_pips=spread_pips,
        timestamp=tick.timestamp,
    )


@router.get("", response_model=dict[str, TickRead])
def list_prices(db: Session = Depends(get_db)):
    cache = get_all_prices()
    if not cache:
        return {}

    symbols = list(cache.keys())
    rows = db.query(Instrument).filter(Instrument.symbol.in_(symbols)).all()
    pip_by_symbol = {row.symbol: row.pip_size for row in rows}

    result: dict[str, TickRead] = {}
    for symbol, tick in cache.items():
        pip_size = pip_by_symbol.get(symbol)
        if pip_size is None:
            logger.warning(
                "instrument '%s' in price cache but not in DB — run POST /instruments/sync",
                symbol,
            )
        result[symbol] = _build_tick_read(tick, pip_size)
    return result


@router.get("/{instrument}", response_model=TickRead)
def get_price(instrument: str, db: Session = Depends(get_db)):
    inst = db.query(Instrument).filter_by(symbol=instrument).first()
    if inst is None:
        raise HTTPException(
            status_code=404,
            detail=f"Instrument '{instrument}' not found. Run POST /api/v1/instruments/sync first.",
        )
    tick = get_latest_price(instrument)
    if tick is None:
        raise HTTPException(
            status_code=404,
            detail="price not yet available (stream initialising)",
        )
    return _build_tick_read(tick, inst.pip_size)
