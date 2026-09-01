from datetime import datetime, timezone, timedelta

from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.brokers.router import BrokerRouter
from app.config import Settings
from app.domain.timeframes import get_timeframe
from app.models.candle import Candle
from app.models.instrument import Instrument


def fetch_and_store(
    instrument_symbol: str,
    granularity: str,
    db: Session,
    settings: Settings,
    broker_router: BrokerRouter,
    price_type: str = "M",
) -> int:
    instrument = db.query(Instrument).filter_by(symbol=instrument_symbol).first()
    if instrument is None:
        raise ValueError(f"Instrument '{instrument_symbol}' not found — run /instruments/sync first")

    latest: Candle | None = (
        db.query(Candle)
        .filter_by(instrument_id=instrument.id, granularity=granularity, price_type=price_type)
        .order_by(Candle.timestamp.desc())
        .first()
    )

    now = datetime.now(timezone.utc)
    if latest is None:
        start = now - timedelta(days=settings.CANDLE_LOOKBACK_DAYS)
    else:
        period_hours = get_timeframe(granularity).period_hours
        start = latest.timestamp.replace(tzinfo=timezone.utc) + timedelta(hours=period_hours)

    if start >= now:
        return 0

    candles = broker_router.for_instrument(instrument_symbol, db).get_candles(
        instrument_symbol, granularity, start, now, price_type=price_type
    )

    if not candles:
        return 0

    rows = [
        {
            "instrument_id": instrument.id,
            "granularity": granularity,
            "price_type": price_type,
            "timestamp": c.timestamp.replace(tzinfo=None),  # store naive UTC
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
        }
        for c in candles
    ]

    stmt = pg_insert(Candle).values(rows).on_conflict_do_nothing()
    result = db.execute(stmt)
    db.commit()
    return result.rowcount


def fetch_and_store_latest(
    instrument_symbol: str,
    granularity: str,
    db: Session,
    settings: Settings,
    price_type: str = "M",
) -> int:
    """
    Convenience wrapper around fetch_and_store: resolves BrokerRouter lazily.

    Used by the candle-close pipeline (M5) to top up the candle table for one
    instrument at H4 / D1 close. Imports BrokerRouter inside the function to
    avoid a circular import at module load (router → factory → broker clients).

    Args:
        instrument_symbol: instrument to top up (never hardcoded — caller-supplied).
        granularity: timeframe code from the Timeframe registry.
        db: SQLAlchemy session.
        settings: config.
        price_type: candle series to fetch — ``"M"`` (decision/indicator series,
            the default), ``"B"`` or ``"A"`` (execution series). Registry-driven at
            the call site via ``Timeframe.price_types``.

    Returns:
        Number of candle rows inserted (duplicates are ignored).
    """
    from app.brokers.router import get_broker_router

    return fetch_and_store(
        instrument_symbol=instrument_symbol,
        granularity=granularity,
        db=db,
        settings=settings,
        broker_router=get_broker_router(),
        price_type=price_type,
    )


def fetch_and_store_window(
    instrument_symbol: str,
    granularity: str,
    start: datetime,
    end: datetime,
    db: Session,
    settings: Settings,
    broker_router: BrokerRouter,
    price_type: str,
) -> int:
    """
    Full-coverage fetch of a FIXED [start, end] window for one
    (instrument, granularity, price_type).

    Unlike fetch_and_store (which is INCREMENTAL — it resumes from the latest
    stored candle), this re-fetches the whole window so every instrument /
    timeframe / price_type can be standardized to identical A..B coverage
    (Pre-M7 Step 1.6 / GAP-11). Duplicate-safe via ON CONFLICT DO NOTHING, so
    re-running is idempotent. Inserts in batches to bound statement size for
    large M1 windows.
    """
    instrument = db.query(Instrument).filter_by(symbol=instrument_symbol).first()
    if instrument is None:
        raise ValueError(f"Instrument '{instrument_symbol}' not found — run /instruments/sync first")

    candles = broker_router.for_instrument(instrument_symbol, db).get_candles(
        instrument_symbol, granularity, start, end, price_type=price_type
    )
    if not candles:
        return 0

    inserted = 0
    batch = 2000
    for i in range(0, len(candles), batch):
        rows = [
            {
                "instrument_id": instrument.id,
                "granularity": granularity,
                "price_type": price_type,
                "timestamp": c.timestamp.replace(tzinfo=None),
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            }
            for c in candles[i : i + batch]
        ]
        stmt = pg_insert(Candle).values(rows).on_conflict_do_nothing()
        inserted += db.execute(stmt).rowcount
        db.commit()
    return inserted


def get_stored_candles(
    instrument_symbol: str,
    granularity: str,
    limit: int,
    db: Session,
    price_type: str = "M",
) -> list[Candle]:
    instrument = db.query(Instrument).filter_by(symbol=instrument_symbol).first()
    if instrument is None:
        return []
    return (
        db.query(Candle)
        .filter_by(instrument_id=instrument.id, granularity=granularity, price_type=price_type)
        .order_by(Candle.timestamp.desc())
        .limit(limit)
        .all()
    )


def count_stored_candles(
    instrument_id: int, granularity: str, db: Session, price_type: str = "M"
) -> int:
    return (
        db.query(Candle)
        .filter_by(instrument_id=instrument_id, granularity=granularity, price_type=price_type)
        .count()
    )
