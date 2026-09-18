from sqlalchemy import func
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.config import Settings
from app.models.candle import Candle
from app.models.indicator import Indicator
from app.models.instrument import Instrument


def compute_and_store(
    instrument_symbol: str,
    granularity: str,
    db: Session,
    settings: Settings,
    full_recompute: bool = False,
) -> int:
    """Compute ATR14/RSI14/swings/Donchian from Mid candles and store them.

    Incremental by default: only candles from (latest indicator − warmup) onward are
    read, and the resulting rows are UPSERTed. This is what the live pipeline uses.

    Two classes of indicator, with different write semantics
    -------------------------------------------------------
    *Recursive* (``atr14``, ``rsi14``): Wilder smoothing has infinite memory, so a value
    computed from a short warmup is an approximation of the full-history one and a
    LATER pass — starting even further forward — is strictly worse. First write wins.

    *Windowed* (``swing_*``, ``donchian_*``): exact functions of a bounded window, so
    recomputation is idempotent and a later pass can only improve them. UPSERTed.

    This distinction is load-bearing, not tidiness. ``swing_*`` comes from a CENTRED
    window: the value at bar *i* depends on bars up to *i+k*, so at the moment bar *i*
    is first written its swing is unknowable and NULL. Filling that in later is the
    only way it is ever populated — and the previous code could not, because it skipped
    already-written timestamps and inserted with ``ON CONFLICT DO NOTHING``. Combined
    with a warmup too short for the window to close at all, ``c3_structure`` was False
    on 100% of live rows while being True on 20% of backtest rows (where indicators had
    been recomputed over full history, and were therefore reading the future).

    When ``full_recompute=True`` the incremental short-circuit is bypassed entirely:
    every indicator is recomputed from the EARLIEST available Mid candle for this
    instrument+granularity, and the existing indicator rows for this
    (instrument_id, granularity) are deleted first so the recompute cleanly
    OVERWRITES them (no duplicates on the natural key, no stale ragged left/right
    edge). Wilder smoothing is therefore genuinely warm from the earliest Mid bar.
    """
    instrument = db.query(Instrument).filter_by(symbol=instrument_symbol).first()
    if instrument is None:
        raise ValueError(f"Instrument '{instrument_symbol}' not found — run /instruments/sync first")

    period = settings.ATR_PERIOD
    swing_lookback = settings.SWING_LOOKBACK_PERIODS
    donchian_period = settings.SIGNAL_DONCHIAN_PERIOD

    # In full-recompute mode we recompute from the earliest Mid candle, so the
    # incremental boundary logic must be skipped (latest_indicator forced to None).
    latest_indicator = None
    if not full_recompute:
        latest_indicator = (
            db.query(Indicator)
            .filter_by(instrument_id=instrument.id, granularity=granularity)
            .order_by(Indicator.timestamp.desc())
            .first()
        )

    # Indicators are Mid-derived ONLY (Bid/Ask are execution-only). Filtering to
    # price_type="M" is REQUIRED now that candles hold M/B/A at the same timestamp,
    # otherwise TR/RSI would be computed across interleaved price types (corruption).
    candle_query = (
        db.query(Candle)
        .filter_by(instrument_id=instrument.id, granularity=granularity, price_type="M")
        .order_by(Candle.timestamp.asc())
    )

    if latest_indicator is not None:
        # Fetch from (latest indicator - warmup buffer) so Wilder smoothing is accurate
        # AND the centred swing window can actually close.
        #
        # This was `period * 2` (28 bars), which is smaller than the 2k+1 = 41 bars a
        # k=20 centred swing window needs. The swing loop below is
        # `range(swing_lookback, len(candles) - swing_lookback)` — with 29 candles that
        # is `range(20, 9)`, i.e. EMPTY, so the live path computed a swing exactly never
        # and `c3_structure` was False on 100% of live rows while being True on 20% of
        # backtest rows. Derive the warmup from every indicator's own requirement
        # instead of from ATR's alone.
        # Each indicator's OWN requirement, maxed — not ATR's alone:
        #   ATR/RSI  Wilder smoothing has infinite memory. Seeded from a finite warmup,
        #            the residual error decays as (1 - 1/period)^m. At the old m = 28
        #            that leaves ~12% of the seeding error, and this suite measured a
        #            0.4% incremental-vs-full divergence in live ATR — a train/serve
        #            skew of the same class as the swing bug, just smaller. period * 8
        #            (112 bars at period=14) puts the residual under 0.03%.
        #   swings   a CENTRED window needs 2k+1 bars to close at all.
        #   donchian a trailing window needs its own period.
        warmup_count = max(period * 8, swing_lookback * 2 + 1, donchian_period) + 1
        warmup_candles = (
            db.query(Candle)
            .filter_by(instrument_id=instrument.id, granularity=granularity, price_type="M")
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

    # ── Donchian channel ──────────────────────────────────────────────────────
    # STRICTLY BACKWARD-LOOKING over the trailing `donchian_period` bars, INCLUDING
    # the current one. That is the whole point of it: a swing pivot is confirmed by a
    # CENTRED window and therefore cannot be known at its own timestamp, which is what
    # made `c3_structure` a look-ahead in backtest and a dead condition live. A
    # Donchian extreme is knowable at the bar that closes it, so the same
    # "price is at structure" question can be asked without a confirmation lag — and
    # it is dense (every bar) rather than the ~1.6% of bars that carry a swing.
    donchian_highs: list[float | None] = [None] * len(candles)
    donchian_lows: list[float | None] = [None] * len(candles)
    for i in range(donchian_period - 1, len(candles)):
        window = range(i - donchian_period + 1, i + 1)
        donchian_highs[i] = max(candles[j].high for j in window)
        donchian_lows[i] = min(candles[j].low for j in window)

    # ── Build rows for insert ─────────────────────────────────────────────────
    # NOTE: rows that already exist are deliberately NOT filtered out any more. A bar
    # written on an earlier pass carries NULL swings (its centred window had not closed
    # yet); the ONLY chance to fill that in is a later pass, once the confirming bars
    # have arrived. The previous `if naive_ts in existing_ts: continue` skip — paired
    # with `on_conflict_do_nothing()` — meant a swing that was NULL on first write
    # stayed NULL forever. Re-emitting the warmup window and UPSERTing is what closes
    # that; see the conflict clause below for which columns may be overwritten.
    rows = []
    for i, candle in enumerate(candles):
        if (
            atrs[i] is None and rsis[i] is None
            and swing_highs[i] is None and swing_lows[i] is None
            and donchian_highs[i] is None and donchian_lows[i] is None
        ):
            continue
        naive_ts = candle.timestamp.replace(tzinfo=None) if candle.timestamp.tzinfo else candle.timestamp
        rows.append({
            "instrument_id": instrument.id,
            "granularity": granularity,
            "timestamp": naive_ts,
            "atr14": atrs[i],
            "rsi14": rsis[i],
            "swing_high": swing_highs[i],
            "swing_low": swing_lows[i],
            "donchian_high": donchian_highs[i],
            "donchian_low": donchian_lows[i],
        })

    if not rows:
        if full_recompute:
            # Nothing to write but still clear any stale rows so the table reflects
            # the (empty) recompute result rather than the old ragged history.
            db.query(Indicator).filter_by(
                instrument_id=instrument.id, granularity=granularity
            ).delete(synchronize_session=False)
            db.commit()
        return 0

    if full_recompute:
        # Clean OVERWRITE: drop the existing (possibly ragged/stale) rows for this
        # instrument+granularity, then bulk-insert the freshly recomputed full set.
        db.query(Indicator).filter_by(
            instrument_id=instrument.id, granularity=granularity
        ).delete(synchronize_session=False)
        stmt = pg_insert(Indicator).values(rows)
    else:
        # Window-function indicators are UPSERTed; recursive ones are not.
        #
        # `swing_*` must be updatable or a NULL written before its centred window
        # closed is never filled in (the live bug). `donchian_*` is updatable so a
        # backfill reaches rows written before the column existed.
        #
        # COALESCE(EXCLUDED.x, indicators.x) — never blank out a value that is already
        # there. A centred-window swing, once confirmed by k bars either side, cannot
        # stop being one, so a NULL arriving later is "not recomputed here", never
        # "no longer a swing".
        #
        # `atr14`/`rsi14` are deliberately absent: Wilder smoothing has infinite memory,
        # so a value recomputed from a SHORTER warmup is strictly worse than the one
        # already stored. First write wins for those.
        insert = pg_insert(Indicator).values(rows)
        stmt = insert.on_conflict_do_update(
            index_elements=["instrument_id", "granularity", "timestamp"],
            set_={
                "swing_high": func.coalesce(insert.excluded.swing_high, Indicator.swing_high),
                "swing_low": func.coalesce(insert.excluded.swing_low, Indicator.swing_low),
                "donchian_high": func.coalesce(insert.excluded.donchian_high, Indicator.donchian_high),
                "donchian_low": func.coalesce(insert.excluded.donchian_low, Indicator.donchian_low),
            },
        )
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
