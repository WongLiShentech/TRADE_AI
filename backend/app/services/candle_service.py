from datetime import datetime, timezone, timedelta

from sqlalchemy.orm import Session
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.brokers.router import BrokerRouter
from app.config import Settings
from app.models.candle import Candle
from app.models.instrument import Instrument

_PERIOD_HOURS = {"H4": 4, "D": 24}


def fetch_and_store(
    instrument_symbol: str,
    granularity: str,
    db: Session,
    settings: Settings,
    broker_router: BrokerRouter,
) -> int:
    instrument = db.query(Instrument).filter_by(symbol=instrument_symbol).first()
    if instrument is None:
        raise ValueError(f"Instrument '{instrument_symbol}' not found — run /instruments/sync first")

    latest: Candle | None = (
        db.query(Candle)
        .filter_by(instrument_id=instrument.id, granularity=granularity)
        .order_by(Candle.timestamp.desc())
        .first()
    )

    now = datetime.now(timezone.utc)
    if latest is None:
        start = now - timedelta(days=settings.CANDLE_LOOKBACK_DAYS)
    else:
        period_hours = _PERIOD_HOURS.get(granularity, 4)
        start = latest.timestamp.replace(tzinfo=timezone.utc) + timedelta(hours=period_hours)

    if start >= now:
        return 0

    candles = broker_router.for_instrument(instrument_symbol, db).get_candles(
        instrument_symbol, granularity, start, now
    )

    if not candles:
        return 0

    rows = [
        {
            "instrument_id": instrument.id,
            "granularity": granularity,
            "timestamp": c.timestamp.replace(tzinfo=None),  # SQLite stores naive
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
        }
        for c in candles
    ]

    stmt = sqlite_insert(Candle).values(rows).on_conflict_do_nothing()
    result = db.execute(stmt)
    db.commit()
    return result.rowcount


def fetch_and_store_latest(
    instrument_symbol: str,
    granularity: str,
    db: Session,
    settings: Settings,
) -> int:
    """
    Convenience wrapper around fetch_and_store: resolves BrokerRouter lazily.

    Used by the candle-close pipeline (M5) to top up the candle table for one
    instrument at H4 / D1 close. Imports BrokerRouter inside the function to
    avoid a circular import at module load (router → factory → broker clients).
    """
    from app.brokers.router import get_broker_router

    return fetch_and_store(
        instrument_symbol=instrument_symbol,
        granularity=granularity,
        db=db,
        settings=settings,
        broker_router=get_broker_router(),
    )


def get_stored_candles(
    instrument_symbol: str,
    granularity: str,
    limit: int,
    db: Session,
) -> list[Candle]:
    instrument = db.query(Instrument).filter_by(symbol=instrument_symbol).first()
    if instrument is None:
        return []
    return (
        db.query(Candle)
        .filter_by(instrument_id=instrument.id, granularity=granularity)
        .order_by(Candle.timestamp.desc())
        .limit(limit)
        .all()
    )


def count_stored_candles(instrument_id: int, granularity: str, db: Session) -> int:
    return (
        db.query(Candle)
        .filter_by(instrument_id=instrument_id, granularity=granularity)
        .count()
    )
