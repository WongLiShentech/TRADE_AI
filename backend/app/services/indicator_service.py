from sqlalchemy.orm import Session
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.config import Settings
from app.models.candle import Candle
from app.models.indicator import Indicator
from app.models.instrument import Instrument


def compute_and_store(
    instrument_symbol: str,
    granularity: str,
    db: Session,
    settings: Settings,
) -> int:
    instrument = db.query(Instrument).filter_by(symbol=instrument_symbol).first()
    if instrument is None:
        raise ValueError(f"Instrument '{instrument_symbol}' not found — run /instruments/sync first")

    period = settings.ATR_PERIOD
    swing_lookback = settings.SWING_LOOKBACK_PERIODS

    latest_indicator = (
        db.query(Indicator)
        .filter_by(instrument_id=instrument.id, granularity=granularity)
        .order_by(Indicator.timestamp.desc())
        .first()
    )

    candle_query = (
        db.query(Candle)
        .filter_by(instrument_id=instrument.id, granularity=granularity)
        .order_by(Candle.timestamp.asc())
    )

    if latest_indicator is not None:
        # Fetch from (latest indicator - warmup buffer) so Wilder smoothing is accurate
        warmup_count = period * 2
        warmup_candles = (
            db.query(Candle)
            .filter_by(instrument_id=instrument.id, granularity=granularity)
            .filter(Candle.timestamp <= latest_indicator.timestamp)
            .order_by(Candle.timestamp.desc())
            .limit(warmup_count)
            .all()
        )
        if warmup_candles:
            boundary_ts = warmup_candles[-1].timestamp
            candle_query = candle_query.filter(Candle.timestamp >= boundary_ts)

    candles = candle_query.all()

    if len(candles) < period + 1:
        return 0

    # ── True Range ────────────────────────────────────────────────────────────
    true_ranges: list[float] = []
    for i in range(1, len(candles)):
        c = candles[i]
        prev_close = candles[i - 1].close
        tr = max(c.high - c.low, abs(c.high - prev_close), abs(c.low - prev_close))
        true_ranges.append(tr)

    # ── ATR(14) via Wilder smoothing ──────────────────────────────────────────
    atrs: list[float | None] = [None] * len(candles)
    atr_val = sum(true_ranges[:period]) / period
    atrs[period] = atr_val
    for i in range(period + 1, len(candles)):
        atr_val = (atr_val * (period - 1) + true_ranges[i - 1]) / period
        atrs[i] = atr_val

    # ── Price changes for RSI ─────────────────────────────────────────────────
    changes: list[float] = [candles[i].close - candles[i - 1].close for i in range(1, len(candles))]
    gains = [max(x, 0.0) for x in changes]
    losses = [abs(min(x, 0.0)) for x in changes]

    # ── RSI(14) via Wilder smoothing ──────────────────────────────────────────
    rsis: list[float | None] = [None] * len(candles)
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    if avg_loss == 0:
        rsis[period] = 100.0
    else:
        rsis[period] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
    for i in range(period + 1, len(candles)):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        if avg_loss == 0:
            rsis[i] = 100.0
        else:
            rsis[i] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))

    # ── Swing highs/lows ──────────────────────────────────────────────────────
    swing_highs: list[float | None] = [None] * len(candles)
    swing_lows: list[float | None] = [None] * len(candles)
    for i in range(swing_lookback, len(candles) - swing_lookback):
        window_highs = [candles[j].high for j in range(i - swing_lookback, i + swing_lookback + 1)]
        window_lows = [candles[j].low for j in range(i - swing_lookback, i + swing_lookback + 1)]
        if candles[i].high == max(window_highs):
            swing_highs[i] = candles[i].high
        if candles[i].low == min(window_lows):
            swing_lows[i] = candles[i].low

    # ── Build rows for insert ─────────────────────────────────────────────────
    existing_ts: set = set()
    if latest_indicator is not None:
        existing_ts = {
            r[0]
            for r in db.query(Indicator.timestamp)
            .filter_by(instrument_id=instrument.id, granularity=granularity)
            .all()
        }

    rows = []
    for i, candle in enumerate(candles):
        if atrs[i] is None and rsis[i] is None and swing_highs[i] is None and swing_lows[i] is None:
            continue
        naive_ts = candle.timestamp.replace(tzinfo=None) if candle.timestamp.tzinfo else candle.timestamp
        if naive_ts in existing_ts:
            continue
        rows.append({
            "instrument_id": instrument.id,
            "granularity": granularity,
            "timestamp": naive_ts,
            "atr14": atrs[i],
            "rsi14": rsis[i],
            "swing_high": swing_highs[i],
            "swing_low": swing_lows[i],
        })

    if not rows:
        return 0

    stmt = sqlite_insert(Indicator).values(rows).on_conflict_do_nothing()
    result = db.execute(stmt)
    db.commit()
    return result.rowcount


def get_stored_indicators(
    instrument_symbol: str,
    granularity: str,
    limit: int,
    db: Session,
) -> list[Indicator]:
    instrument = db.query(Instrument).filter_by(symbol=instrument_symbol).first()
    if instrument is None:
        return []
    return (
        db.query(Indicator)
        .filter_by(instrument_id=instrument.id, granularity=granularity)
        .order_by(Indicator.timestamp.desc())
        .limit(limit)
        .all()
    )


def count_stored_indicators(instrument_id: int, granularity: str, db: Session) -> int:
    return (
        db.query(Indicator)
        .filter_by(instrument_id=instrument_id, granularity=granularity)
        .count()
    )
