"""Unit tests for app.services.backtester.simulator — the triple-barrier engine.

Most cases here are pure unit tests: no DB reads/writes. Each injects a synthetic
``m1_source`` (a list of :class:`BidAskBar`) directly into :func:`simulate`, so
``db=None`` is safe — the simulator never falls back to the DB fallback path when
the primary source resolves the trade (every scenario below is constructed to do
so). IMPORTANT CAVEAT (QA-identified blind spot): injecting ``m1_source`` bypasses
:func:`simulate`'s own fetch-window construction entirely (the ``end`` bound /
``_db_bidask_source`` call), so these synthetic tests CANNOT catch a bug in that
window's sizing — see ``test_time_exit_weekend_straddle_real_db`` below (a real
DB-integration regression test) for that.

Convention shared by the synthetic cases (see inline comments for the exact math):
    entry = 100.0, R = 2.0 (LONG: stop=98, target=104=+2R; SHORT: stop=102, target=96=+2R)
    atr14_at_signal = 1.0, trail_mult = 1.0 -> trail_dist = 1.0 price unit
    lock_pct = 0.5 -> partial fires at +1R (entry + 0.5*(target-entry))
    granularity = "H4" (period_hours=4.0), SIGNAL_MAX_HOLD_BARS = 10 -> 40h base horizon

Run: python -m pytest tests/test_simulator.py -v
(the DB-integration test needs the running Postgres dev DB, same as test_feature_builder.py)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import pytest

from app.services.backtester.simulator import BidAskBar, simulate

_T0 = datetime(2024, 1, 1, 0, 0, 0)


@dataclass
class _FakeSettings:
    """Only the three attributes `simulate()` reads — no DB / real env needed."""

    SIGNAL_MAX_HOLD_BARS: int = 10
    BACKTEST_TRAILING_LOCK_PCT: float = 0.5
    BACKTEST_TRAILING_DISTANCE_ATR_MULT: float = 1.0


SETTINGS = _FakeSettings()

# Shared trade geometry.
ENTRY = 100.0
ATR = 1.0
LONG_STOP, LONG_TARGET = 98.0, 104.0     # R=2.0, target=+2R
SHORT_STOP, SHORT_TARGET = 102.0, 96.0   # R=2.0, target=+2R


def _bar(
    hours: float,
    bid_o: float, bid_h: float, bid_l: float, bid_c: float,
    ask_o: float, ask_h: float, ask_l: float, ask_c: float,
) -> BidAskBar:
    return BidAskBar(
        ts=_T0 + timedelta(hours=hours),
        bid_open=bid_o, bid_high=bid_h, bid_low=bid_l, bid_close=bid_c,
        ask_open=ask_o, ask_high=ask_h, ask_low=ask_l, ask_close=ask_c,
    )


def _simulate_long(bars: list[BidAskBar]):
    return simulate(
        db=None, settings=SETTINGS, instrument_id=1, direction="buy",
        entry_price=ENTRY, stop=LONG_STOP, target=LONG_TARGET,
        signal_time=_T0, atr14_at_signal=ATR, granularity="H4", m1_source=bars,
    )


def _simulate_short(bars: list[BidAskBar]):
    return simulate(
        db=None, settings=SETTINGS, instrument_id=1, direction="sell",
        entry_price=ENTRY, stop=SHORT_STOP, target=SHORT_TARGET,
        signal_time=_T0, atr14_at_signal=ATR, granularity="H4", m1_source=bars,
    )


# ── (a) clean TP-before-partial -> +2.0R ──────────────────────────────────────
def test_clean_tp_before_partial_long():
    # ask_high touches target (104) directly; bid_low stays well above stop.
    bars = [_bar(1, 100, 100, 99.0, 100, 100, 105, 100, 104.5)]
    r = _simulate_long(bars)
    assert r.exit_reason == "tp_hit"
    assert r.exit_price == pytest.approx(104.0)
    assert r.rr_actual == pytest.approx(2.0)
    assert r.ambiguous_resolution is False


# ── (b) SL-before-partial -> -1.0R ────────────────────────────────────────────
def test_sl_before_partial_long():
    # bid_low touches stop (98); ask_high stays below target/partial.
    bars = [_bar(1, 99.0, 99.0, 97.0, 98.0, 100, 101, 100, 100.5)]
    r = _simulate_long(bars)
    assert r.exit_reason == "sl_hit"
    assert r.exit_price == pytest.approx(98.0)
    assert r.rr_actual == pytest.approx(-1.0)
    assert r.ambiguous_resolution is False


# ── (c) partial then trail-out at entry -> +0.5R ──────────────────────────────
def test_partial_then_trail_out_at_entry_long():
    # Bar 1: ask_high reaches the 1R partial level (102) but not the 2R target
    # (104); bid_low doesn't touch the original stop (98). Partial fires,
    # current_sl -> entry (100), then ratchets on bid_close=101 -> stays 100
    # (candidate = 101 - trail_dist(1.0) = 100 == entry, no improvement).
    bar1 = _bar(1, 100, 103, 99.0, 101.0, 100, 103, 100, 102.5)
    # Bar 2: bid_low dips to current_sl (100, unchanged from bar1's ratchet);
    # ask_high stays well below target so this is an unambiguous trail-out.
    bar2 = _bar(2, 100.5, 100.5, 99.5, 100.0, 101, 101, 100.5, 100.8)
    r = _simulate_long([bar1, bar2])
    assert r.exit_reason == "trailing_stop"
    assert r.rr_actual == pytest.approx(0.5)
    assert r.ambiguous_resolution is False


# ── (d) partial then TP -> +1.5R ──────────────────────────────────────────────
def test_partial_then_tp_long():
    bar1 = _bar(1, 100, 103, 99.0, 101.0, 100, 103, 100, 102.5)  # fires the partial
    bar2 = _bar(2, 100.5, 100.5, 100.2, 100.4, 101, 105, 101, 104.5)  # then TP
    r = _simulate_long([bar1, bar2])
    assert r.exit_reason == "tp_hit"
    assert r.exit_price == pytest.approx(104.0)
    assert r.rr_actual == pytest.approx(1.5)
    assert r.ambiguous_resolution is False


# ── (e) ambiguous bar (both barriers touched) -> SL-first + flag ─────────────
def test_ambiguous_bar_sl_first_long():
    # Same bar touches both the stop (bid_low=97 <= 98) and the target
    # (ask_high=105 >= 104). Conservative SL-first: exits at the stop, flagged.
    bars = [_bar(1, 99.0, 99.0, 97.0, 98.5, 100, 105, 100, 103.0)]
    r = _simulate_long(bars)
    assert r.exit_reason == "sl_hit"
    assert r.exit_price == pytest.approx(98.0)
    assert r.rr_actual == pytest.approx(-1.0)
    assert r.ambiguous_resolution is True


# ── (f) time exit — counts TRADING bars, not wall-clock hours ────────────────
def test_time_exit_long():
    # bars_elapsed counts DISTINCT H4-bucket transitions observed in the M1
    # stream, not floor(elapsed_hours / period_hours). Ten bars, each exactly
    # one H4-period (4h) apart with continuous data (no gap), land in ten
    # consecutive distinct buckets -> bars_elapsed increments 1,2,...,10, exactly
    # matching SIGNAL_MAX_HOLD_BARS=10 on the 10th bar. No barrier is touched on
    # any bar (ask_high stays below the partial level 102; bid_low stays above
    # the stop 98).
    filler = [
        _bar(4 * k, 100.0, 100.3, 99.5, 100.1, 100.2, 101.0, 100.0, 100.6)
        for k in range(1, 10)
    ]
    last = _bar(40, 100.1, 100.3, 99.5, 100.2, 100.5, 101.0, 100.4, 100.6)
    r = _simulate_long(filler + [last])
    assert r.exit_reason == "time_exit"
    assert r.exit_price == pytest.approx(100.2)
    assert r.rr_actual == pytest.approx((100.2 - 100.0) / 2.0)
    assert r.ambiguous_resolution is False


def test_time_exit_weekend_straddle_pauses_clock():
    # Synthetic regression for the BAR-COUNTING logic only (see the module
    # docstring caveat: injecting m1_source bypasses simulate()'s own fetch-
    # window sizing, so this test cannot catch a fetch-bound bug — that is
    # test_time_exit_weekend_straddle_real_db's job). A Thursday-entry trade
    # whose 10-bar hold legitimately straddles a weekend must NOT be flagged
    # ambiguous, and must NOT time out mid-weekend. Eight bars land in eight
    # distinct H4 buckets covering (synthetic) Thu->Fri with continuous data ->
    # bars_elapsed 1..8. Then the M1 stream has a large gap (Sat/Sun -- zero
    # ticks, ~48 wall-clock hours with NO data at all), and trading resumes with
    # two more distinct-bucket bars. Despite the huge wall-clock jump, the gap
    # costs exactly ONE step (not several hours' worth of "missed" buckets) ->
    # bars_elapsed reaches 9 then 10, exiting on the second post-weekend bar --
    # still resolved entirely from the M1 primary stream, so NOT ambiguous.
    pre_weekend = [
        _bar(4 * k, 100.0, 100.3, 99.5, 100.1, 100.2, 101.0, 100.0, 100.6)
        for k in range(1, 9)  # hours 4..32 -> 8 distinct buckets -> bars_elapsed 1..8
    ]
    post_weekend = [
        _bar(80, 100.0, 100.3, 99.5, 100.1, 100.2, 101.0, 100.0, 100.6),  # bars_elapsed=9
        _bar(84, 100.1, 100.3, 99.5, 100.3, 100.5, 101.0, 100.4, 100.6),  # bars_elapsed=10 -> exit
    ]
    r = _simulate_long(pre_weekend + post_weekend)
    assert r.exit_reason == "time_exit"
    assert r.exit_price == pytest.approx(100.3)
    assert r.rr_actual == pytest.approx((100.3 - 100.0) / 2.0)
    assert r.ambiguous_resolution is False


# ── (g) weekend gap-open beyond stop -> rr worse than -1.0 ───────────────────
def test_gap_open_beyond_stop_long():
    # The bar's OPEN has already gapped through the stop (98); fill is honest
    # at the gap-open price (95), not the stop level.
    bars = [_bar(1, 95.0, 95.5, 94.0, 94.5, 96.0, 99.0, 96.0, 98.5)]
    r = _simulate_long(bars)
    assert r.exit_reason == "sl_hit"
    assert r.exit_price == pytest.approx(95.0)
    assert r.rr_actual == pytest.approx((95.0 - 100.0) / 2.0)
    assert r.rr_actual < -1.0
    assert r.ambiguous_resolution is False


# ── (h) LONG vs SHORT side-correctness (Bid/Ask roles swap) ──────────────────
def test_short_clean_tp_uses_bid_low():
    # SHORT TP is checked on Bid-low (mirrors LONG's Ask-high). bid_low touches
    # target (96); ask_high stays below the SL level (102).
    bars = [_bar(1, 97.0, 97.5, 95.0, 96.5, 97.0, 101.0, 97.0, 98.0)]
    r = _simulate_short(bars)
    assert r.exit_reason == "tp_hit"
    assert r.exit_price == pytest.approx(96.0)
    assert r.rr_actual == pytest.approx(2.0)


def test_short_sl_uses_ask_high():
    # SHORT SL is checked on Ask-high (mirrors LONG's Bid-low). ask_high
    # touches the stop (102); bid_low stays above the target (96).
    bars = [_bar(1, 101.0, 103.0, 99.0, 101.5, 101.0, 103.0, 100.0, 102.5)]
    r = _simulate_short(bars)
    assert r.exit_reason == "sl_hit"
    assert r.exit_price == pytest.approx(102.0)
    assert r.rr_actual == pytest.approx(-1.0)


def test_short_partial_then_trail_out_at_entry():
    # Mirrors test_partial_then_trail_out_at_entry_long: partial level for a
    # SHORT with lock_pct=0.5 is entry - 0.5*(entry-target) = 100 - 2 = 98.
    # Bar 1: bid_low hits the partial level (98); ask_high (100) stays below the
    # SL (102). Partial fires, current_sl -> entry (100), then ratchets on
    # ask_close=99.0 -> candidate = 99.0 + trail_dist(1.0) = 100 == entry (tied).
    bar1 = _bar(1, 99.5, 100.0, 98.0, 98.5, 99.5, 100.0, 98.0, 99.0)
    # Bar 2: ask_high (100) touches the ratcheted stop (100, == entry) -> trail-out.
    bar2 = _bar(2, 99.6, 100.0, 99.5, 99.8, 99.7, 100.0, 99.6, 99.9)
    r = _simulate_short([bar1, bar2])
    assert r.exit_reason == "trailing_stop"
    assert r.rr_actual == pytest.approx(0.5)


# ── input validation ──────────────────────────────────────────────────────────
def test_zero_risk_raises():
    with pytest.raises(ValueError):
        simulate(
            db=None, settings=SETTINGS, instrument_id=1, direction="buy",
            entry_price=100.0, stop=100.0, target=104.0,
            signal_time=_T0, atr14_at_signal=ATR, granularity="H4", m1_source=[],
        )


def test_unknown_direction_raises():
    with pytest.raises(ValueError):
        simulate(
            db=None, settings=SETTINGS, instrument_id=1, direction="hold",
            entry_price=100.0, stop=98.0, target=104.0,
            signal_time=_T0, atr14_at_signal=ATR, granularity="H4", m1_source=[],
        )


# ── REAL DB-integration regression (QA-required — closure-proof fetch bound) ─
def test_time_exit_weekend_straddle_real_db(db, settings, instrument):
    """QA-required regression for the closure-proof fetch-bound fix.

    Calls simulate() WITHOUT m1_source — so it exercises simulate()'s OWN window
    construction (base_horizon_hours * _CLOSURE_CAP_MULTIPLIER -> _db_bidask_source)
    against the real ingested M1 Bid/Ask table, unlike every synthetic case above
    which injects the stream directly and never touches that code path.

    Fixture: a REAL Thursday-evening H4 decision bar for EUR_USD — bar OPEN
    2024-08-08 17:00 UTC, CLOSE 2024-08-08 21:00 UTC (both real, ingested H4 Mid
    candles; the Bid/Ask execution window is 2024-06-03..2026-06-03, so this date
    is well covered). signal_time follows the feature_builder convention
    (decision_bar_close + 1s). Stop/target are ~300 pips away (entry +/- 0.03/0.06)
    — far wider than EUR/USD's actual ~60-pip move over this real week — so the
    trade resolves via TIME EXIT, not a barrier touch, isolating the fetch-bound
    behaviour under test.

    Before the fix: the fetch bound was fixed at (max_hold+1)*period_hours = 44h
    wall-clock, which for this Thursday entry ends mid-weekend (Saturday) — the
    M1 stream ran dry before bars_elapsed reached SIGNAL_MAX_HOLD_BARS, and the
    truncated-right-edge branch fired at the last real bar (Friday close),
    flagged ambiguous_resolution=True. This is the exact defect QA found live in
    the 2026-07-05 backtest run (421/2965 trades, ~96% Thu/Fri-opened).

    After the fix (empirically verified against this real fixture before writing
    these assertions): exit_reason='time_exit', exit_time=2024-08-12 08:00:00
    (a Monday), holding_hours~=83h (spans the weekend), ambiguous_resolution=False.
    """
    inst = instrument("EUR_USD")
    signal_time = datetime(2024, 8, 8, 21, 0, 1)
    entry = 1.09188
    stop = entry - 0.03
    target = entry + 0.06
    atr14 = 0.005

    result = simulate(
        db, settings, inst.id, "buy", entry, stop, target,
        signal_time, atr14, "H4",
    )

    assert result.exit_reason == "time_exit"
    assert result.ambiguous_resolution is False
    # The following Monday or Tuesday — NOT mid-weekend (Sat/Sun) and not
    # truncated at the prior Friday close.
    assert result.exit_time.date() in (date(2024, 8, 12), date(2024, 8, 13))
    assert result.holding_hours > 60.0  # comfortably spans the weekend closure
    assert result.rr_actual == pytest.approx((result.exit_price - entry) / (entry - stop))
