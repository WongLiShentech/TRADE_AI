"""Unit tests for excursion recording (Phase A) — MFE/MAE and the per-bar path.

These reuse the synthetic-``m1_source`` convention of ``test_simulator.py`` (see
that module's docstring): ``db=None``, hand-built bars, entry=100.0 with R=2.0.

The single most important property under test is NEGATIVE: attaching a recorder
must not change ANY exit. The recorder rides along with a walk that already
visits every bar; if it could alter an exit it would silently rewrite the labels
the model trains on. ``test_recorder_never_changes_an_exit`` asserts that across
every scenario in the suite by running each one twice.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pytest

from app.services.backtester.simulator import BidAskBar, simulate

_T0 = datetime(2024, 1, 1, 0, 0, 0)


@dataclass
class _FakeSettings:
    """The simulator's reads, plus the two path knobs.

    ``SHADOW_MIN_BUCKET_M1_DENSITY = 0.0`` disables the degraded-bucket flag:
    these sources are a handful of hand-built bars, not real minute streams, so a
    density check would mark every bar degraded and test nothing.
    """

    SIGNAL_MAX_HOLD_BARS: int = 10
    BACKTEST_TRAILING_LOCK_PCT: float = 0.5
    BACKTEST_TRAILING_DISTANCE_ATR_MULT: float = 1.0
    PATH_EXTENDED_BARS: int = 5
    SHADOW_MIN_BUCKET_M1_DENSITY: float = 0.0


SETTINGS = _FakeSettings()

ENTRY = 100.0
ATR = 1.0
LONG_STOP, LONG_TARGET = 98.0, 104.0     # R = 2.0
SHORT_STOP, SHORT_TARGET = 102.0, 96.0   # R = 2.0


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


def _flat(hours: float, bid: float, ask: float) -> BidAskBar:
    """A bar that neither moves nor touches anything — pure time passing."""
    return _bar(hours, bid, bid, bid, bid, ask, ask, ask, ask)


def _sim(bars, *, direction="buy", record_path=True, settings=SETTINGS):
    stop, target = (LONG_STOP, LONG_TARGET) if direction == "buy" else (SHORT_STOP, SHORT_TARGET)
    return simulate(
        db=None, settings=settings, instrument_id=1, direction=direction,
        entry_price=ENTRY, stop=stop, target=target,
        signal_time=_T0, atr14_at_signal=ATR, granularity="H4",
        m1_source=list(bars), record_path=record_path,
    )


# ── the safety property: recording must be passive ───────────────────────────
# Spreads here are REALISTIC (0.02 on a 100.0 price = 2bp, ~an FX major) rather
# than the wide synthetic spreads test_simulator.py uses to isolate barrier logic.
# That matters: the barrier checks test a target against the ENTRY side of the
# book while the excursion is measured on the EXIT side, so an implausibly wide
# spread makes the two disagree by more than the trade's whole R. With real
# spreads the gap is ~0.01R — see test_barrier_and_path_conventions_differ_by_a_spread,
# which pins that known asymmetry explicitly instead of hiding it in a tolerance.
_SCENARIOS = {
    # ask_high reaches the target (104); bid follows one spread behind.
    "clean_tp": ([_bar(1, 100.0, 104.08, 99.90, 104.00, 100.02, 104.10, 99.92, 104.02)], "buy"),
    # bid_low reaches the stop (98).
    "clean_sl": ([_bar(1, 99.90, 99.90, 97.90, 98.00, 99.92, 99.92, 97.92, 98.02)], "buy"),
    # reaches +1R (102) so the partial fires, then decays into the trail.
    "partial_then_trail": (
        [
            _bar(1, 101.98, 102.08, 101.50, 102.00, 102.00, 102.10, 101.52, 102.02),
            _bar(5, 100.90, 101.00, 100.50, 100.60, 100.92, 101.02, 100.52, 100.62),
        ],
        "buy",
    ),
    # drifts sideways until the clock runs out.
    "time_exit": ([_flat(h, 100.10, 100.12) for h in range(1, 60, 4)], "buy"),
    # short side: bid_low reaches the target (96).
    "short_tp": ([_bar(1, 95.90, 100.00, 95.88, 96.00, 95.92, 100.02, 95.90, 96.02)], "sell"),
}


@pytest.mark.parametrize("name", sorted(_SCENARIOS))
def test_recorder_never_changes_an_exit(name):
    """THE load-bearing test: identical exits with and without recording.

    If this ever fails, the recorder is influencing the walk and every label in
    the training corpus is suspect.
    """
    bars, direction = _SCENARIOS[name]
    without = _sim(bars, direction=direction, record_path=False)
    with_ = _sim(bars, direction=direction, record_path=True)

    assert with_.exit_reason == without.exit_reason
    assert with_.exit_price == pytest.approx(without.exit_price)
    assert with_.rr_actual == pytest.approx(without.rr_actual)
    assert with_.exit_time == without.exit_time
    assert with_.ambiguous_resolution == without.ambiguous_resolution


def test_no_recorder_means_no_path():
    """Not asking for a path leaves the fields empty — never zero-valued.

    An empty path and a flat path must stay distinguishable: `0.0` would read as
    "this trade never moved", which is a claim, not an absence.
    """
    bars, _ = _SCENARIOS["clean_tp"]
    r = _sim(bars, record_path=False)
    assert r.path == ()
    assert r.mfe_r is None
    assert r.mae_r is None


# ── the invariant that catches almost any excursion bug ──────────────────────
@pytest.mark.parametrize("name", sorted(_SCENARIOS))
def test_mae_le_result_le_mfe(name):
    """A trade cannot finish better than its best moment, nor worse than its worst.

    This holds for blended (post-partial) results too: the partial can only be
    banked at a level the price actually reached, so it is bounded by MFE.
    """
    bars, direction = _SCENARIOS[name]
    r = _sim(bars, direction=direction)
    assert r.mfe_r is not None and r.mae_r is not None
    # One realistic spread (0.02) over a risk unit of 2.0 is 0.01R. The margin
    # below is that, plus float slack — deliberately tight, so a genuine
    # excursion bug cannot hide inside it.
    tol = 0.02
    assert r.mae_r - tol <= r.rr_actual <= r.mfe_r + tol


def test_barrier_and_path_conventions_differ_by_at_most_a_spread():
    """Pins the KNOWN asymmetry between barrier detection and excursion measurement.

    ``_bar_signals`` tests a target against the ENTRY side of the book (a LONG's
    take-profit is checked on ``ask_high``, though the position would be closed by
    selling at the bid). The excursion record deliberately uses the EXIT side,
    which is what a close would actually realise. The two therefore disagree by up
    to one spread, and the barrier side is the optimistic one.

    That asymmetry is a real, separately-tracked issue in the barrier logic —
    fixing it changes every ``rr_actual`` and so every training label, which is
    why it is deferred to a deliberate relabel-and-retrain rather than slipped in
    here. This test exists so the gap stays BOUNDED and visible: if it ever grows
    beyond a spread, something else has drifted.
    """
    # A LONG whose ask touches the target while the bid stays one spread short.
    spread = 0.02
    bars = [_bar(1, 100.0, 103.98, 99.90, 103.96, 100.02, 104.00, 99.92, 103.98)]
    r = _sim(bars)
    assert r.exit_reason == "tp_hit"
    assert r.rr_actual == pytest.approx(2.0)          # barrier says a clean +2R
    # The exit side never quite got there — by exactly one spread over the risk unit.
    assert r.mfe_r == pytest.approx(2.0 - spread / 2.0, abs=1e-9)
    assert r.rr_actual - r.mfe_r <= spread / 2.0 + 1e-9


def test_mfe_captures_a_peak_the_result_throws_away():
    """The whole point: a losing trade that was winning first.

    Mirrors the real GBP_JPY shadow row that peaked near +0.84R and still closed
    negative — indistinguishable, today, from one that never moved.
    """
    bars = [
        _bar(1, 103.0, 103.4, 102.0, 103.0, 103.2, 103.6, 102.2, 103.2),  # ~+1.5R
        *[_flat(h, 99.6, 99.8) for h in range(5, 60, 4)],                 # decays, clock runs out
    ]
    r = _sim(bars)
    assert r.exit_reason in {"time_exit", "trailing_stop"}
    assert r.mfe_r > 1.0, "peak must be recorded"
    assert r.mae_r < 0.0, "the drawdown after the peak must be recorded too"


def test_running_extremes_are_monotonic():
    """mfe_r never falls and mae_r never rises as the path advances."""
    bars = [
        _bar(1, 101.0, 101.4, 100.8, 101.0, 101.2, 101.6, 101.0, 101.2),
        _bar(5, 99.5, 100.0, 99.0, 99.2, 99.7, 100.2, 99.2, 99.4),
        *[_flat(h, 100.0, 100.2) for h in range(9, 60, 4)],
    ]
    r = _sim(bars)
    assert len(r.path) >= 3
    for prev, cur in zip(r.path, r.path[1:]):
        assert cur.bar > prev.bar, "bars must be strictly increasing"
        assert cur.mfe_r >= prev.mfe_r
        assert cur.mae_r <= prev.mae_r
        assert prev.r_worst <= prev.r_best


def test_beyond_exit_bars_are_tagged_and_excluded_from_mfe():
    """Post-exit prices answer 'should we have held?' but must not flatter the result.

    The trade stops out early; price then rallies far past the original target.
    Those bars belong in the path, tagged — and nowhere near ``mfe_r``, which
    describes the trade that actually happened.
    """
    bars = [
        _bar(1, 99.0, 99.0, 97.0, 98.0, 100, 101, 100, 100.5),   # SL at −1R, bar 0
        *[_flat(h, 110.0, 110.2) for h in (5, 9, 13)],           # +5R territory, after the exit
    ]
    r = _sim(bars)
    assert r.exit_reason == "sl_hit"
    assert r.rr_actual == pytest.approx(-1.0)

    beyond = [p for p in r.path if p.beyond_exit]
    assert beyond, "post-exit bars must be recorded"
    assert max(p.r_close for p in beyond) > 4.0, "the rally must be visible in the path"
    # ...and must NOT have leaked into the trade's own excursion.
    assert r.mfe_r < 1.0, "post-exit prices must never inflate mfe_r"


def test_extended_bars_zero_stops_the_path_at_the_exit():
    """PATH_EXTENDED_BARS=0 disables the lookahead entirely."""
    bars = [
        _bar(1, 99.0, 99.0, 97.0, 98.0, 100, 101, 100, 100.5),
        *[_flat(h, 110.0, 110.2) for h in (5, 9, 13)],
    ]
    settings = _FakeSettings(PATH_EXTENDED_BARS=0)
    r = _sim(bars, settings=settings)
    assert r.exit_reason == "sl_hit"
    assert not any(p.beyond_exit for p in r.path)


def test_truncated_stream_is_flagged_not_silently_short():
    """A path that ran out of data must say so — silence would read as a calm market."""
    bars = [
        _bar(1, 99.0, 99.0, 97.0, 98.0, 100, 101, 100, 100.5),  # exits immediately
        _flat(5, 100.0, 100.2),                                  # only 1 of 5 extended bars
    ]
    r = _sim(bars)
    assert r.path_truncated is True


def test_short_side_uses_the_ask_for_adverse_excursion():
    """SHORT closes by BUYING at the ask, so adverse excursion is measured there.

    Using the bid would understate the loss by one spread on every short.
    """
    # Price rises against the short. Ask is 1.0 above bid; MAE must reflect ask.
    bars = [
        _bar(1, 100.5, 101.0, 100.4, 100.8, 101.5, 102.0, 101.4, 101.8),
        *[_flat(h, 100.0, 101.0) for h in range(5, 60, 4)],
    ]
    r = _sim(bars, direction="sell")
    # ask_high 102.0 → r = (100 − 102) / 2 = −1.0 exactly, on the ask.
    # Had the bid (101.0) been used it would read −0.5 — half the true adverse move.
    assert r.mae_r == pytest.approx(-1.0, abs=1e-9)
