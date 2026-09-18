"""Incremental indicator computation must agree with a full recompute.

Why this suite exists
---------------------
``c3_structure`` was True on 20.0% of backtest rows and 0.0% of live rows for six
weeks, and nothing failed. The cause was entirely inside ``compute_and_store``:

* the incremental warmup was ``ATR_PERIOD * 2`` = 28 bars, but a centred swing window
  of ``SWING_LOOKBACK_PERIODS`` = 20 needs 41, so ``range(20, len(candles) - 20)`` was
  empty and the live path computed a swing *never*;
* and an already-written timestamp was skipped, then inserted ``ON CONFLICT DO
  NOTHING`` — so a NULL swing written before its window closed could never be filled in
  once the confirming bars arrived.

Both are invisible to every other test in the suite, because every other test either
uses a fully recomputed database or asserts on the *feature* rather than on the writer.
The invariant that catches them is the one asserted here: **bars appended a few at a
time must end up with the same indicator values as bars computed all at once.** That
equality is what live/backtest parity rests on, and it is cheap to check.

Synthetic instrument and candles throughout — no dependency on what any particular
database happens to hold, and nothing real is touched.
"""
from __future__ import annotations

import math
import random
from datetime import datetime, timedelta

import pytest

from app.models.candle import Candle
from app.models.indicator import Indicator
from app.models.instrument import Instrument
from app.services.indicator_service import compute_and_store

_SYMBOL = "TEST_INDICATORS"
_GRAN = "H4"
_START = datetime(2024, 1, 1, 0, 0, 0)
_BAR = timedelta(hours=4)
# Comfortably longer than max(ATR_PERIOD*2, SWING_LOOKBACK*2+1, DONCHIAN) so every
# indicator has room to produce values in both modes.
_N_BARS = 220
# Bars appended per incremental pass. Deliberately SMALLER than the swing window, so a
# swing is guaranteed to be first written as NULL and only confirmable on a later pass —
# exactly the sequence the old code could not handle.
_CHUNK = 5


def _synthetic_candles(instrument_id: int, n: int) -> list[Candle]:
    """A deterministic random walk with sane OHLC ordering."""
    rng = random.Random(42)
    out: list[Candle] = []
    price = 1.1000
    for i in range(n):
        drift = rng.uniform(-0.0015, 0.0015)
        close = price + drift
        high = max(price, close) + abs(rng.uniform(0, 0.0008))
        low = min(price, close) - abs(rng.uniform(0, 0.0008))
        out.append(Candle(
            instrument_id=instrument_id, granularity=_GRAN, price_type="M",
            timestamp=_START + i * _BAR,
            open=price, high=high, low=low, close=close, volume=1000,
        ))
        price = close
    return out


@pytest.fixture()
def synthetic(db):
    """A throwaway instrument plus its candles; removed afterwards."""
    db.query(Instrument).filter_by(symbol=_SYMBOL).delete(synchronize_session=False)
    db.commit()
    inst = Instrument(
        symbol=_SYMBOL, display_name="synthetic", pip_size=0.0001, pip_location=-4,
        asset_class="forex", broker_id="test", is_active=False,
    )
    db.add(inst)
    db.commit()
    db.refresh(inst)
    candles = _synthetic_candles(inst.id, _N_BARS)
    try:
        yield inst, candles
    finally:
        db.query(Indicator).filter_by(instrument_id=inst.id).delete(synchronize_session=False)
        db.query(Candle).filter_by(instrument_id=inst.id).delete(synchronize_session=False)
        db.query(Instrument).filter_by(id=inst.id).delete(synchronize_session=False)
        db.commit()


def _snapshot(db, instrument_id: int) -> dict:
    rows = (
        db.query(Indicator)
        .filter_by(instrument_id=instrument_id, granularity=_GRAN)
        .order_by(Indicator.timestamp.asc())
        .all()
    )
    return {
        r.timestamp: {
            "atr14": r.atr14, "rsi14": r.rsi14,
            "swing_high": r.swing_high, "swing_low": r.swing_low,
            "donchian_high": r.donchian_high, "donchian_low": r.donchian_low,
        }
        for r in rows
    }


def test_incremental_equals_full_recompute(db, settings, synthetic):
    """Appending bars a few at a time must land where computing them all at once does.

    Windowed indicators (swings, Donchian) are exact functions of a bounded window, so
    they must match EXACTLY. Recursive ones (ATR/RSI via Wilder smoothing) have infinite
    memory, so an incremental value seeded from a finite warmup only converges towards
    the full-history one — asserted within a tolerance, which is the honest invariant.
    """
    inst, candles = synthetic

    for start in range(0, len(candles), _CHUNK):
        db.add_all(candles[start:start + _CHUNK])
        db.commit()
        compute_and_store(_SYMBOL, _GRAN, db, settings)
    incremental = _snapshot(db, inst.id)

    compute_and_store(_SYMBOL, _GRAN, db, settings, full_recompute=True)
    full = _snapshot(db, inst.id)

    assert incremental, "incremental pass wrote no indicator rows at all"
    missing = sorted(set(full) - set(incremental))
    assert not missing, f"{len(missing)} timestamps only exist after a full recompute"

    for ts, want in full.items():
        got = incremental[ts]
        for key in ("swing_high", "swing_low", "donchian_high", "donchian_low"):
            assert got[key] == want[key], (
                f"{key} at {ts}: incremental={got[key]!r} full={want[key]!r} — a windowed "
                f"indicator is an exact function of a bounded window, so these cannot differ"
            )
        for key in ("atr14", "rsi14"):
            if want[key] is None or got[key] is None:
                continue
            assert math.isclose(got[key], want[key], rel_tol=1e-3), (
                f"{key} at {ts}: incremental={got[key]!r} full={want[key]!r} — Wilder "
                f"smoothing should have converged within the warmup"
            )


def test_incremental_actually_writes_swings(db, settings, synthetic):
    """The live path must populate swings at all.

    This is the regression proper. With the old `warmup = ATR_PERIOD * 2` the centred
    swing loop was empty on every incremental pass, so `swing_high`/`swing_low` were
    NULL on every live row and `c3_structure` could not fire in production even though
    it fired on 20% of backtest rows.
    """
    inst, candles = synthetic

    for start in range(0, len(candles), _CHUNK):
        db.add_all(candles[start:start + _CHUNK])
        db.commit()
        compute_and_store(_SYMBOL, _GRAN, db, settings)

    rows = (
        db.query(Indicator)
        .filter_by(instrument_id=inst.id, granularity=_GRAN)
        .order_by(Indicator.timestamp.asc())
        .all()
    )
    assert any(r.swing_high is not None for r in rows), "no swing HIGH was ever written"
    assert any(r.swing_low is not None for r in rows), "no swing LOW was ever written"

    # A swing is confirmed k bars after the fact, so the newest confirmed one may lag by
    # up to k bars — but not by more. Unbounded lag is the bug.
    k = settings.SWING_LOOKBACK_PERIODS
    confirmed = [r for r in rows if r.swing_high is not None or r.swing_low is not None]
    newest_confirmed = max(r.timestamp for r in confirmed)
    newest_row = max(r.timestamp for r in rows)
    lag_bars = (newest_row - newest_confirmed) / _BAR
    assert lag_bars <= k * 2, (
        f"newest confirmed swing lags the newest row by {lag_bars:.0f} bars "
        f"(k={k}) — swings are not being back-filled as their windows close"
    )


def test_donchian_is_causal_and_dense(db, settings, synthetic):
    """Donchian must be knowable at its own bar, and present on (almost) every bar.

    The two properties that make it a valid replacement for swing pivots in a signal
    rule: it looks only backwards, and it is not sparse. Swings satisfy neither.
    """
    inst, candles = synthetic
    db.add_all(candles)
    db.commit()
    compute_and_store(_SYMBOL, _GRAN, db, settings, full_recompute=True)

    rows = (
        db.query(Indicator)
        .filter_by(instrument_id=inst.id, granularity=_GRAN)
        .order_by(Indicator.timestamp.asc())
        .all()
    )
    by_ts = {c.timestamp: c for c in candles}
    period = settings.SIGNAL_DONCHIAN_PERIOD
    ordered = sorted(by_ts)

    with_channel = [r for r in rows if r.donchian_high is not None]
    assert len(with_channel) > 0.8 * len(rows), (
        f"only {len(with_channel)}/{len(rows)} rows carry a Donchian channel — it must be "
        f"dense, unlike swings"
    )

    for r in with_channel:
        i = ordered.index(r.timestamp)
        window = ordered[max(0, i - period + 1): i + 1]
        assert r.donchian_high == max(by_ts[t].high for t in window), (
            f"donchian_high at {r.timestamp} is not the max over the TRAILING window — "
            f"if it reads forward, the leak this replaced has simply moved"
        )
        assert r.donchian_low == min(by_ts[t].low for t in window)
