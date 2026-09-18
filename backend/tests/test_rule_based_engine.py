"""Behaviour lock for the rule engine, so a refactor can be proven not to change it.

Why this suite exists
---------------------
``RuleBasedSignalEngine`` had no tests at all. It is the component that decides what the
platform trades, it is about to be refactored onto a condition registry, and three
separate defects have already lived inside it unnoticed (the centred-swing look-ahead,
the direction-independent votes satisfying both sides at once, and a hardcoded
``_H4_HOURS`` that ignores the timeframe registry). "The suite still passes" meant
nothing about this file.

What is locked here
-------------------
A GOLDEN VECTOR: synthetic bars engineered so each condition's truth value is known by
construction, evaluated through the real ``evaluate()``, with the resulting
``score_breakdown`` asserted key by key. Synthetic rather than a real
(instrument, timestamp) pair so the expected values cannot drift with whatever a
particular database happens to hold — the same reason ``test_indicator_service`` builds
its own candles.

This is deliberately a test of COMPOSITION, not of market wisdom: it says which
conditions fired and how the engine combined them. Whether those are good conditions is
what the backtest answers.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.models.candle import Candle
from app.models.indicator import Indicator
from app.models.instrument import Instrument
from app.services.signal_engine.base import SignalOutput
from app.services.signal_engine.rule_based import BacktestQuote, RuleBasedSignalEngine

_SYMBOL = "TEST_ENGINE"
_GRAN = "H4"
_TREND_GRAN = "D"
_PIP = 0.0001
# A Wednesday 12:00 UTC close → the following evaluation lands in the london/ny overlap,
# so C4 is satisfied and is not what any assertion below is really about.
_T0 = datetime(2025, 3, 5, 12, 0, 0)
_BAR = timedelta(hours=4)


def _mk_candle(instrument_id, granularity, ts, o, h, l, c):
    return Candle(
        instrument_id=instrument_id, granularity=granularity, price_type="M",
        timestamp=ts, open=o, high=h, low=l, close=c, volume=1000,
    )


@pytest.fixture()
def rig(db, settings):
    """A synthetic instrument whose every condition input is set deliberately.

    Built so that, at the bar under test:
      C1 trend     BUY true  — D1 close is above its SMA
      C2 rsi       BUY true  — RSI is inside the configured BUY band
      C3 structure BUY true  — close sits ON the 20-bar low, so distance is 0
      C4 session   true      — evaluation time is inside SIGNAL_SESSION_FILTER
      C5 spread    true      — quote spread is well inside SIGNAL_MAX_SPREAD_PIPS
    """
    db.query(Instrument).filter_by(symbol=_SYMBOL).delete(synchronize_session=False)
    db.commit()
    inst = Instrument(
        symbol=_SYMBOL, display_name="synthetic", pip_size=_PIP, pip_location=-4,
        asset_class="forex", broker_id="test", is_active=False,
    )
    db.add(inst)
    db.commit()
    db.refresh(inst)

    n_h4 = 120
    sma_period = settings.SIGNAL_TREND_SMA_PERIOD
    price = 1.2000

    # H4: a gentle decline into the bar under test, so its close IS the 20-bar low.
    h4 = []
    for i in range(n_h4):
        ts = _T0 - (n_h4 - 1 - i) * _BAR
        c = price - i * 0.0002
        h4.append(_mk_candle(inst.id, _GRAN, ts, c + 0.0001, c + 0.0003, c - 0.0001, c))
    # D1: rising, so the latest daily close is comfortably above its own SMA (C1 BUY).
    d1 = []
    for i in range(sma_period + 5):
        ts = (_T0 - timedelta(days=(sma_period + 4 - i))).replace(hour=21, minute=0, second=0)
        c = 1.1000 + i * 0.0030
        d1.append(_mk_candle(inst.id, _TREND_GRAN, ts, c, c + 0.0010, c - 0.0010, c))

    db.add_all(h4 + d1)
    db.commit()

    # Indicators are written directly rather than computed, so each condition's input is
    # an explicit, readable number instead of an emergent property of the synthetic walk.
    lows = [c.low for c in h4[-settings.SIGNAL_DONCHIAN_PERIOD:]]
    highs = [c.high for c in h4[-settings.SIGNAL_DONCHIAN_PERIOD:]]

    # RSI must sit in the BUY band and OUTSIDE the SELL band. The two bands overlap
    # (40-55 BUY, 45-60 SELL as configured), and an RSI in that overlap satisfies C2 for
    # BOTH directions — which, with C4/C5 already shared, is enough for SELL to reach the
    # threshold too and for the ambiguity guard to suppress the signal entirely. Picking
    # the midpoint of the two lower bounds keeps the rig unambiguous.
    rsi_buy_only = (settings.SIGNAL_RSI_OVERSOLD + settings.SIGNAL_RSI_OVERSOLD_SELL) / 2
    assert settings.SIGNAL_RSI_OVERSOLD <= rsi_buy_only <= settings.SIGNAL_RSI_OVERBOUGHT, (
        "rig RSI is not inside the BUY band — the band configuration has moved"
    )
    assert not (settings.SIGNAL_RSI_OVERSOLD_SELL <= rsi_buy_only <= settings.SIGNAL_RSI_OVERBOUGHT_SELL), (
        "rig RSI also satisfies the SELL band, so this bar is ambiguous by construction"
    )
    db.add(Indicator(
        instrument_id=inst.id, granularity=_GRAN, timestamp=_T0,
        atr14=0.0020, rsi14=rsi_buy_only,
        swing_high=None, swing_low=None,          # C3 must NOT depend on these any more
        donchian_high=max(highs), donchian_low=min(lows),
    ))
    db.commit()

    try:
        yield inst, h4[-1]
    finally:
        db.query(Indicator).filter_by(instrument_id=inst.id).delete(synchronize_session=False)
        db.query(Candle).filter_by(instrument_id=inst.id).delete(synchronize_session=False)
        db.query(Instrument).filter_by(id=inst.id).delete(synchronize_session=False)
        db.commit()


def _evaluate(db, settings, bar, *, spread_pips=1.0):
    half = spread_pips * _PIP / 2
    return RuleBasedSignalEngine().evaluate(
        _SYMBOL, _GRAN, db, settings,
        as_of=bar.timestamp + _BAR + timedelta(seconds=1),
        quote=BacktestQuote(bid=bar.close - half, ask=bar.close + half),
    )


def test_golden_vector_all_conditions_true(db, settings, rig):
    """The locked composition: five conditions, a BUY, and an ATR-derived stop/target."""
    _, bar = rig
    out = _evaluate(db, settings, bar)

    assert out is not None, "engineered all-true bar produced no signal"
    assert isinstance(out, SignalOutput)
    assert out.direction == "BUY"
    assert out.score_breakdown == {
        "trend": True, "rsi": True, "structure": True, "session": True, "spread": True,
    }, out.score_breakdown
    assert out.confidence_score == 5

    # Geometry is derived, never invented: stop is N x ATR, target is MIN_RR_RATIO x stop.
    stop_distance = settings.SIGNAL_STOP_ATR_MULTIPLIER * 0.0020
    assert out.entry == pytest.approx(bar.close + (1.0 * _PIP / 2))
    assert out.stop == pytest.approx(out.entry - stop_distance)
    assert out.target == pytest.approx(out.entry + settings.MIN_RR_RATIO * stop_distance)


def test_c3_reads_the_donchian_channel_not_swing_pivots(db, settings, rig):
    """C3 must be answerable from the channel alone.

    The rig writes ``swing_high``/``swing_low`` as NULL. Before the fix C3 read those and
    would be False here; it is now True because the close sits on the Donchian low. This
    is the regression that pins the leak fix in place: a future edit that reaches back for
    swing pivots fails here rather than six weeks later in production.
    """
    inst, bar = rig
    out = _evaluate(db, settings, bar)
    assert out is not None and out.score_breakdown["structure"] is True

    # Push the channel far below the close: same bar, same swings (still NULL), but the
    # structure condition must now be False.
    row = db.query(Indicator).filter_by(instrument_id=inst.id, granularity=_GRAN).one()
    row.donchian_low = bar.close - 50 * settings.SIGNAL_STRUCTURE_ATR_BUFFER * 0.0020
    db.commit()

    out2 = _evaluate(db, settings, bar)
    if out2 is not None:
        assert out2.score_breakdown["structure"] is False, (
            "structure stayed true after the Donchian low moved out of range — C3 is not "
            "reading the channel"
        )


def test_a_null_donchian_reads_as_false_never_as_satisfied(db, settings, rig):
    """An unevaluable condition is not a satisfied one.

    Between the migration and the backfill every row has a NULL channel. Treating that as
    True would have every signal claim structure it was never checked for.
    """
    inst, bar = rig
    row = db.query(Indicator).filter_by(instrument_id=inst.id, granularity=_GRAN).one()
    row.donchian_low = None
    row.donchian_high = None
    db.commit()

    out = _evaluate(db, settings, bar)
    if out is not None:
        assert out.score_breakdown["structure"] is False


def test_spread_gate_uses_the_quote_not_the_candle(db, settings, rig):
    """C5 is computed from the passed quote, so a wide spread must fail it."""
    _, bar = rig
    wide = settings.SIGNAL_MAX_SPREAD_PIPS * 3
    out = _evaluate(db, settings, bar, spread_pips=wide)
    if out is not None:
        assert out.score_breakdown["spread"] is False, (
            f"a {wide}-pip spread passed a {settings.SIGNAL_MAX_SPREAD_PIPS}-pip cap"
        )


def test_an_rsi_in_the_band_overlap_satisfies_both_directions(db, settings, rig):
    """The RSI bands overlap, and inside that overlap C2 is effectively direction-blind.

    Configured: BUY 40-55, SELL 45-60. An RSI in 45-55 satisfies both. Combined with C4
    and C5 — which are shared verbatim — SELL reaches 3 of 5 on session + spread + rsi
    alone, with no directional evidence whatsoever, and the ambiguity guard then
    suppresses a signal the BUY side had genuinely earned 5 points for.

    This is the same family as the session/spread defect and it is a second reason
    Stage 2 must score only directional conditions. Pinned here so the property is
    recorded rather than rediscovered.
    """
    inst, bar = rig
    lo_sell, hi_buy = settings.SIGNAL_RSI_OVERSOLD_SELL, settings.SIGNAL_RSI_OVERBOUGHT
    if lo_sell > hi_buy:
        pytest.skip("RSI bands no longer overlap — the defect this pins is gone")

    row = db.query(Indicator).filter_by(instrument_id=inst.id, granularity=_GRAN).one()
    row.rsi14 = (lo_sell + hi_buy) / 2      # inside BOTH bands
    db.commit()

    assert _evaluate(db, settings, bar) is None, (
        "a bar whose RSI satisfies both directions still produced a signal — either the "
        "ambiguity guard changed or the bands stopped overlapping"
    )


def test_gate_conditions_answer_identically_for_both_directions(db, settings, rig):
    """The property behind the ambiguity bug, asserted rather than assumed.

    Every condition the registry declares a GATE must return the same answer for BUY and
    SELL — that is what "direction-independent" means, and it is why scoring them as
    votes hands both sides the same 2 points. At a threshold of 2 that alone satisfies
    both directions, every bar is ambiguous, and the rule fires only where a gate FAILED.

    Checked behaviourally, by evaluating the real callables, rather than by matching
    source text: the property is about what the conditions DO, and it has to keep holding
    after Stage 2 moves where they are consumed.
    """
    from app.domain.conditions import Role, enabled_conditions
    from app.domain.conditions.spec import ConditionContext
    from app.models.candle import Candle as _C

    inst, bar = rig
    row = db.query(Indicator).filter_by(instrument_id=inst.id, granularity=_GRAN).one()
    bars = (
        db.query(_C).filter_by(instrument_id=inst.id, granularity=_GRAN, price_type="M")
        .order_by(_C.timestamp.asc()).all()
    )
    ctx = ConditionContext(
        instrument=inst, granularity=_GRAN,
        now_utc=bar.timestamp + _BAR + timedelta(seconds=1),
        settings=settings, bars=bars, trend_bars=[], indicators=row,
        quote_bid=bar.close - _PIP / 2, quote_ask=bar.close + _PIP / 2,
        trend_close=None, trend_sma=None,
    )

    declared_gates = [c for c in enabled_conditions(settings) if c.role is Role.GATE]
    assert declared_gates, "no gates are declared — Stage 2 has nothing to separate"
    for cond in declared_gates:
        assert cond.evaluate(ctx, "BUY") == cond.evaluate(ctx, "SELL"), (
            f"condition '{cond.key}' is declared a GATE but answers differently per "
            f"direction — it is a VOTE and the registry is wrong"
        )
