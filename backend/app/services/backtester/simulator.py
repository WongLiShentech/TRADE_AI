"""Triple-barrier trade simulator (M7 core) — Option-B trailing-stop exits.

Given one fired signal, :func:`simulate` walks intrabar candles forward from the
signal time and returns the realised exit as an :class:`ExitResult` (exit price /
time / reason, realised R-multiple, an ambiguity flag and holding hours). It is
pure and DB-read-only — it never writes; the Part B runner persists the row.

Design (locked decisions — see the M7 plan "Component 3" and "Key constants")
----------------------------------------------------------------------------
* **Barrier checks run at the intrabar granularity (M1).** The signal timeframe
  (H4/D1, ``granularity`` arg) is used ONLY to count elapsed bars for the time
  exit, via ``app.domain.timeframes`` ``period_hours`` (never a hardcoded 4).
* **Bid + Ask, jointly.** M1 Bid and Ask are stored as *separate rows sharing a
  timestamp*; they are joined (here, in Python) into one :class:`BidAskBar`.
* **Side roles (bid vs ask swap by direction):**
    - LONG  → SL when Bid-low <= current_sl ; TP when Ask-high >= target.
    - SHORT → SL when Ask-high >= current_sl ; TP when Bid-low <= target.
* **Option B trailing stop.** Before the partial, if price reaches
  ``entry ± BACKTEST_TRAILING_LOCK_PCT × (target − entry)`` (=+1R for a 2R target
  and a 0.5 lock), 50% is banked at that lock and ``current_sl`` moves to entry.
  After the partial, every bar ``current_sl`` ratchets toward profit only, at
  ``BACKTEST_TRAILING_DISTANCE_ATR_MULT × atr14_at_signal`` from the bar close
  (``atr14_at_signal`` is frozen for the trade's life).
* **Blended R (50/50 after a partial):** SL-before-partial = −1.0R (full);
  trail-out after partial = ``0.5·partial_r + 0.5·R_at_current_sl``; TP after
  partial = ``0.5·partial_r + 0.5·2.0`` (= +1.5R for the standard config); TP
  before partial = +2.0R (full).
* **Ambiguous M1 bar** (both barriers touched in one bar): conservative SL-first,
  ``ambiguous_resolution=True``.
* **M1 gap / weekend:** a bar whose *open* has gapped through ``current_sl`` exits
  AT the gap-open (``rr_actual`` may be worse than −1.0 — recorded honestly). If
  there is NO M1 for the window at all, a degraded fallback resolves on the signal
  timeframe (Bid/Ask, or Mid if that TF has no Bid/Ask), flagged
  ``ambiguous_resolution=True``.
* **Closure-proof fetch bound (QA-verified fix).** ``bars_elapsed`` counts TRADING
  bars only (see below) — but the M1 QUERY WINDOW itself must reach far enough in
  wall-clock time to actually contain ``max_hold`` trading bars when a weekend or
  holiday closure sits inside the hold. A prior version bounded the fetch at
  exactly ``(max_hold+1) * period_hours`` wall-clock hours; for a Thu/Fri entry
  that window ended mid-weekend, so the M1 stream ran dry at Friday's close
  before ``bars_elapsed`` reached ``max_hold`` and the truncated-right-edge
  branch fired prematurely (~14% of trades, all Thu/Fri-opened or holiday-
  adjacent — objectively corrupted labels, not a flag-semantics issue). Fixed by
  widening the fetch bound to ``_CLOSURE_CAP_MULTIPLIER`` (4) times the base
  horizon — enough to absorb a long weekend AND a holiday cluster (e.g.
  Dec 23-26 + Dec 30-31) — at ZERO cost for the common (no-closure) case: the M1
  source is a lazy, chunked generator and the walk returns as soon as a barrier
  or the time-exit condition is met, however far the query's upper bound
  extends; widening it only lets the query SEE bars beyond a closure, it never
  forces the walk to consume them. Only when the stream still runs dry INSIDE
  this wider cap (a genuine data gap larger than a normal weekend+holiday, or
  the true right-edge of the dataset) does the truncated-exit +
  ``ambiguous_resolution=True`` branch fire — exactly the cases it should.
* **Time exit — TRADING bars, not wall-clock hours.** ``bars_elapsed`` counts
  DISTINCT signal-timeframe (H4) bar-windows actually observed in the M1 stream
  since the signal, not elapsed calendar time / ``period_hours``. A weekend (or
  any other stretch with zero M1 data) contains no observable H4 windows, so it
  contributes exactly ONE step to the counter when trading resumes — regardless
  of how many wall-clock H4 windows were skipped — rather than the several hours
  a naive ``floor(elapsed_hours / period_hours)`` would silently charge against
  the hold budget. A Thursday-entry trade's 10-bar horizon therefore legitimately
  extends into the following Monday. See ``_h4_bucket`` / the counting loop in
  ``_run_core``. At ``bars_elapsed >= SIGNAL_MAX_HOLD_BARS`` (barriers checked
  first), exit at that bar's close (blended if a partial already banked). This is
  independent of, and does not affect, the M1-unavailability degraded fallback
  below (a normal weekend inside the hold window is NOT ``ambiguous_resolution``;
  only a genuine missing-data window is).

Spread realism
--------------
``rr_actual`` is computed from bid/ask-aware fill prices. Barrier hits fill at the
barrier *level* (so a clean TP-before-partial is exactly +2.0R by construction),
while market exits (time / gap) fill on the exit side of the book (LONG→Bid close,
SHORT→Ask close). The runner (Part B) supplies ``entry_price`` / ``stop`` /
``target`` from H4 Bid/Ask closes (LONG buys at Ask, SHORT sells at Bid), so the
spread is inherent end-to-end rather than added as a separate deduction.

Timestamp convention
--------------------
``candles.timestamp`` is the bar OPEN instant (see ``feature_builder`` module
docstring). Consistent with the feature builder's strictly-after-close rule, the
runner passes ``signal_time = decision_bar_close + 1s``; this simulator only ever
consults bars with ``timestamp > signal_time`` (strictly after the signal).
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Iterable, Iterator, Optional

from sqlalchemy.orm import Session

from app.config import Settings
from app.domain.timeframes import get_timeframe
from app.models.candle import Candle

# ── locked constants (documented, not tunable hyperparameters) ───────────────
# Option B banks half the position at the lock. 50/50 is the definition of the
# strategy, not a free knob — hence a named constant rather than an env var.
PARTIAL_FILL_FRACTION: float = 0.5

# All path-dependent barrier checks run on this granularity. It is the finest
# stored bar (Bid+Ask) and a fixed property of the simulator, not a per-signal
# parameter. Validated against the timeframe registry at import (fail loud).
_INTRABAR_GRANULARITY: str = "M1"
get_timeframe(_INTRABAR_GRANULARITY)  # raises if the registry ever drops M1

# DB streaming page size — never loads the whole M1 table; per-trade the horizon
# is only ~(max_hold+1) signal-TF bars, but we still page defensively.
_FETCH_CHUNK: int = 10_000

# Closure-proof multiplier on the base (max_hold+1)*period_hours wall-clock
# horizon used to bound the M1 fetch window (see module docstring "Closure-proof
# fetch bound"). 4x comfortably absorbs a normal weekend AND a holiday cluster
# (e.g. Dec 23-26 + Dec 30-31) while remaining a bounded, sane ceiling rather
# than unbounded lookahead. Free for the common case: the M1 source is a lazy,
# chunked generator that the walk stops consuming as soon as it resolves,
# regardless of how far this bound extends.
_CLOSURE_CAP_MULTIPLIER: int = 4

# Fixed anchor for H4-bucket arithmetic (see _h4_bucket). The anchor's absolute
# value is irrelevant — only bucket-id EQUALITY/INEQUALITY between two timestamps
# is ever consulted — so any fixed reference works as long as it is applied
# consistently. All stored timestamps are naive UTC (project convention).
_BUCKET_EPOCH: datetime = datetime(1970, 1, 1)

_LONG_ALIASES = frozenset({"buy", "long"})
_SHORT_ALIASES = frozenset({"sell", "short"})


@dataclass(frozen=True)
class BidAskBar:
    """One intrabar candle carrying both sides of the book at a shared timestamp.

    Prices are raw quote prices (no pip scaling — instrument-agnostic). ``ts`` is
    the bar OPEN instant, matching ``candles.timestamp``.
    """

    ts: datetime
    bid_open: float
    bid_high: float
    bid_low: float
    bid_close: float
    ask_open: float
    ask_high: float
    ask_low: float
    ask_close: float


@dataclass(frozen=True)
class ExitResult:
    """The resolved outcome of one simulated trade.

    Attributes:
        exit_price: the fill price of the (final) exit.
        exit_time: the resolving bar's open instant (the canonical candle key).
        exit_reason: one of ``tp_hit`` | ``sl_hit`` | ``trailing_stop`` | ``time_exit``.
        rr_actual: realised R-multiple (blended 50/50 when a partial banked first).
        ambiguous_resolution: True if resolved via SL-first tie-break or the
            degraded (no-M1) fallback — the row can be relabelled later.
        holding_hours: ``exit_time − signal_time`` in hours.
    """

    exit_price: float
    exit_time: datetime
    exit_reason: str
    rr_actual: float
    ambiguous_resolution: bool
    holding_hours: float


@dataclass
class _State:
    """Mutable per-trade simulation state (frozen inputs + evolving stop/partial)."""

    side: int              # +1 long, -1 short
    entry: float
    target: float
    risk: float            # R = |entry - stop|, > 0
    partial_level: float
    trail_dist: float      # BACKTEST_TRAILING_DISTANCE_ATR_MULT × atr14_at_signal (price units)
    max_hold: int
    signal_time: datetime
    degraded: bool
    current_sl: float
    partial_filled: bool = False
    partial_r: float = 0.0  # R banked when the partial fires (= r_of(partial_level))


# ── public API ───────────────────────────────────────────────────────────────
def horizon_bounds(settings: Settings, granularity: str) -> tuple[float, float]:
    """The two wall-clock horizons that bound one simulated trade, in HOURS.

    Returns ``(base_hours, fetch_cap_hours)`` where:

    * ``base_hours = (SIGNAL_MAX_HOLD_BARS + 1) * period_hours`` — the horizon the
      trade would need if trading never paused (no weekend/holiday inside the hold).
      It is a LOWER bound on when an outcome can be known.
    * ``fetch_cap_hours = base_hours * _CLOSURE_CAP_MULTIPLIER`` — the closure-proof
      upper bound on the M1 query window (see the module docstring). Beyond this the
      simulator stops looking; a trade still unresolved here resolves degraded and
      flagged ``ambiguous_resolution=True``.

    Exposed (rather than recomputed by callers) so the M8-Shadow outcome resolver
    decides "is this row knowable yet?" against the EXACT horizons the simulator
    will then walk. Two independent copies of this arithmetic drifting apart would
    silently produce truncated, mislabelled outcomes.

    Args:
        settings: config; reads ``SIGNAL_MAX_HOLD_BARS``.
        granularity: the signal timeframe (e.g. ``"H4"``) — resolved through the
            Timeframe registry, never a hardcoded period.

    Returns:
        ``(base_hours, fetch_cap_hours)``.

    Raises:
        ValueError: ``granularity`` is not in the Timeframe registry.
    """
    period_hours = get_timeframe(granularity).period_hours
    base_hours = (int(settings.SIGNAL_MAX_HOLD_BARS) + 1) * period_hours
    return base_hours, base_hours * _CLOSURE_CAP_MULTIPLIER


def simulate(
    db: Optional[Session],
    settings: Settings,
    instrument_id: int,
    direction: str,
    entry_price: float,
    stop: float,
    target: float,
    signal_time: datetime,
    atr14_at_signal: float,
    granularity: str,
    *,
    m1_source: Optional[Iterable[BidAskBar]] = None,
    h4_source: Optional[Iterable[BidAskBar]] = None,
) -> ExitResult:
    """Simulate one fired signal to exit and return its :class:`ExitResult`.

    Args:
        db: SQLAlchemy session used only to stream candles. May be ``None`` when
            both ``m1_source`` and ``h4_source`` are injected (tests use fakes so
            no DB round-trip / writes occur).
        settings: config; reads ``SIGNAL_MAX_HOLD_BARS``,
            ``BACKTEST_TRAILING_LOCK_PCT`` and ``BACKTEST_TRAILING_DISTANCE_ATR_MULT``
            (zero hardcoding — every number is env-driven).
        instrument_id: candle FK; instrument-agnostic (no symbol/pip logic here).
        direction: ``"buy"``/``"long"`` or ``"sell"``/``"short"`` (case-insensitive).
        entry_price: fill price of the entry (Bid/Ask-derived by the runner).
        stop: initial protective stop; ``R = |entry_price − stop|`` must be > 0.
        target: take-profit level (the runner sizes it to satisfy ``MIN_RR_RATIO``).
        signal_time: T — strictly after the decision bar's close (runner passes
            ``close + 1s``). Only bars with ``timestamp > T`` are consulted.
        atr14_at_signal: ATR(14) on the signal timeframe at T, frozen as the trail
            distance basis for the whole trade.
        granularity: the signal timeframe (e.g. ``"H4"``) — used only to count
            elapsed bars for the time exit, via the timeframe registry.
        m1_source: optional pre-built intrabar stream (injected in tests); defaults
            to a chunked DB reader of M1 Bid/Ask over the trade horizon.
        h4_source: optional degraded-fallback stream; defaults to a DB reader of the
            signal timeframe (Bid/Ask, or Mid if that TF has none).

    Returns:
        The resolved :class:`ExitResult`.

    Raises:
        ValueError: if ``direction`` is unrecognised or ``R == 0``.
    """
    t = _naive(signal_time)
    side = _side(direction)
    risk = abs(float(entry_price) - float(stop))
    if risk == 0.0:
        raise ValueError("stop == entry_price: a zero-risk signal cannot be simulated")

    period_hours = get_timeframe(granularity).period_hours
    max_hold = int(settings.SIGNAL_MAX_HOLD_BARS)
    lock_pct = float(settings.BACKTEST_TRAILING_LOCK_PCT)
    trail_dist = float(settings.BACKTEST_TRAILING_DISTANCE_ATR_MULT) * float(atr14_at_signal)
    # partial lock level, direction-aware (+ for long, − for short).
    partial_level = float(entry_price) + side * lock_pct * abs(float(target) - float(entry_price))

    # Closure-proof fetch bound — see module docstring and :func:`horizon_bounds`
    # (the ONE definition of this arithmetic, shared with the shadow resolver).
    _base_horizon_hours, horizon_hours = horizon_bounds(settings, granularity)
    end = t + timedelta(hours=horizon_hours)

    def _make_state(degraded: bool) -> _State:
        return _State(
            side=side,
            entry=float(entry_price),
            target=float(target),
            risk=risk,
            partial_level=partial_level,
            trail_dist=trail_dist,
            max_hold=max_hold,
            signal_time=t,
            degraded=degraded,
            current_sl=float(stop),
        )

    # Primary: M1 Bid/Ask. Returns None only if the stream yielded zero usable bars.
    primary = m1_source if m1_source is not None else _db_bidask_source(
        db, instrument_id, _INTRABAR_GRANULARITY, t, end
    )
    result = _run_core(primary, _make_state(degraded=False), period_hours)
    if result is not None:
        return result

    # Degraded fallback: no M1 for the window (weekend/holiday true edge case).
    fallback = h4_source if h4_source is not None else _db_fallback_source(
        db, instrument_id, granularity, t, end
    )
    result = _run_core(fallback, _make_state(degraded=True), period_hours)
    if result is not None:
        return result

    # Nothing resolvable at all — degraded flat exit at entry, flagged for relabel.
    return ExitResult(
        exit_price=float(entry_price),
        exit_time=t,
        exit_reason="time_exit",
        rr_actual=0.0,
        ambiguous_resolution=True,
        holding_hours=0.0,
    )


# ── core loop ────────────────────────────────────────────────────────────────
def _run_core(
    source: Iterable[BidAskBar],
    st: _State,
    period_hours: float,
) -> Optional[ExitResult]:
    """Consume bars in ascending time order; return an :class:`ExitResult`, or
    ``None`` if the stream yielded no bar strictly after the signal (caller then
    tries the fallback).

    ``bars_elapsed`` counts DISTINCT H4-bucket transitions actually observed in
    the (M1) stream — not elapsed wall-clock time. ``current_bucket`` starts at
    the signal's own bucket (not yet "elapsed"); each time an incoming bar's
    bucket differs from the last one seen, the counter advances by exactly ONE,
    regardless of how many wall-clock buckets were skipped in between. A weekend
    (or any other window with zero M1 data) is therefore a single +1 step when
    trading resumes, not several hours silently charged against the hold budget.
    """
    last_bar: Optional[BidAskBar] = None
    current_bucket = _h4_bucket(st.signal_time, period_hours)
    bars_elapsed = 0
    for raw in source:
        bar = raw if raw.ts.tzinfo is None else replace(raw, ts=_naive(raw.ts))
        if bar.ts <= st.signal_time:
            continue  # strictly-after-signal only
        bucket = _h4_bucket(bar.ts, period_hours)
        if bucket != current_bucket:
            bars_elapsed += 1
            current_bucket = bucket
        result = _process_bar(bar, st, bars_elapsed)
        if result is not None:
            return result
        last_bar = bar

    if last_bar is None:
        return None

    # Bars existed but neither a barrier nor the time exit fired — the data ran out
    # before the horizon (truncated M1 right edge). Exit honestly at the last close,
    # flagged degraded.
    close_exit = last_bar.bid_close if st.side > 0 else last_bar.ask_close
    rr = _blended_rr(st, _r_of(close_exit, st))
    return ExitResult(
        exit_price=close_exit,
        exit_time=last_bar.ts,
        exit_reason="time_exit",
        rr_actual=rr,
        ambiguous_resolution=True,
        holding_hours=_hours(last_bar.ts, st.signal_time),
    )


def _process_bar(bar: BidAskBar, st: _State, bars_elapsed: int) -> Optional[ExitResult]:
    """Resolve one bar. Mutates ``st`` (partial / trailing stop); returns an
    :class:`ExitResult` on exit, else ``None``. Barriers are checked before the
    time exit so an intrabar barrier touch always wins over the bar's close."""
    sl_touch, tp_touch, partial_touch, sl_exit, close_exit = _bar_signals(bar, st)

    if not st.partial_filled:
        if sl_touch and tp_touch:
            return _exit(sl_exit, "sl_hit", st, bar, ambiguous=True)  # SL-first tie-break
        if sl_touch:
            return _exit(sl_exit, "sl_hit", st, bar)
        if tp_touch:
            return _exit(st.target, "tp_hit", st, bar)  # full +2R (news-spike case)
        if partial_touch:
            st.partial_filled = True
            st.current_sl = st.entry                       # stop → break-even
            st.partial_r = _r_of(st.partial_level, st)     # banked leg (= +1R standard)
            _ratchet(bar, st)
    else:
        if sl_touch and tp_touch:
            return _exit(sl_exit, "trailing_stop", st, bar, ambiguous=True, blended=True)
        if sl_touch:
            return _exit(sl_exit, "trailing_stop", st, bar, blended=True)
        if tp_touch:
            return _exit(st.target, "tp_hit", st, bar, blended=True)  # (1 + 2)/2 = +1.5R
        _ratchet(bar, st)

    if bars_elapsed >= st.max_hold:
        return _exit(close_exit, "time_exit", st, bar, blended=st.partial_filled)
    return None


def _bar_signals(bar: BidAskBar, st: _State):
    """Direction-aware barrier touches + fill prices for one bar.

    Returns ``(sl_touch, tp_touch, partial_touch, sl_exit_price, close_exit_price)``.
    ``sl_exit_price`` fills at the gap-open when the bar opened through the stop
    (honest weekend slippage), else at ``current_sl``.
    """
    if st.side > 0:  # LONG — SL on Bid, TP/partial on Ask
        sl_touch = bar.bid_low <= st.current_sl
        tp_touch = bar.ask_high >= st.target
        partial_touch = bar.ask_high >= st.partial_level
        sl_exit = bar.bid_open if bar.bid_open <= st.current_sl else st.current_sl
        close_exit = bar.bid_close
    else:            # SHORT — SL on Ask, TP/partial on Bid (roles swap)
        sl_touch = bar.ask_high >= st.current_sl
        tp_touch = bar.bid_low <= st.target
        partial_touch = bar.bid_low <= st.partial_level
        sl_exit = bar.ask_open if bar.ask_open >= st.current_sl else st.current_sl
        close_exit = bar.ask_close
    return sl_touch, tp_touch, partial_touch, sl_exit, close_exit


def _ratchet(bar: BidAskBar, st: _State) -> None:
    """Move the trailing stop toward profit only (never loosen). LONG references
    the Bid close, SHORT the Ask close (the exit side of the book)."""
    if st.side > 0:
        candidate = bar.bid_close - st.trail_dist
        st.current_sl = max(st.current_sl, st.entry, candidate)
    else:
        candidate = bar.ask_close + st.trail_dist
        st.current_sl = min(st.current_sl, st.entry, candidate)


def _exit(
    price: float,
    reason: str,
    st: _State,
    bar: BidAskBar,
    *,
    ambiguous: bool = False,
    blended: bool = False,
) -> ExitResult:
    rr = _r_of(price, st)
    if blended:
        rr = PARTIAL_FILL_FRACTION * st.partial_r + (1.0 - PARTIAL_FILL_FRACTION) * rr
    return ExitResult(
        exit_price=price,
        exit_time=bar.ts,
        exit_reason=reason,
        rr_actual=rr,
        ambiguous_resolution=ambiguous or st.degraded,
        holding_hours=_hours(bar.ts, st.signal_time),
    )


def _r_of(price: float, st: _State) -> float:
    """Signed R-multiple of an exit at ``price`` (direction-aware)."""
    return st.side * (price - st.entry) / st.risk


def _blended_rr(st: _State, exit_r: float) -> float:
    if st.partial_filled:
        return PARTIAL_FILL_FRACTION * st.partial_r + (1.0 - PARTIAL_FILL_FRACTION) * exit_r
    return exit_r


# ── DB candle sources (default; tests inject fakes instead) ──────────────────
def _db_bidask_source(
    db: Session,
    instrument_id: int,
    granularity: str,
    start: datetime,
    end: datetime,
    *,
    chunk: int = _FETCH_CHUNK,
) -> Iterator[BidAskBar]:
    """Stream joined Bid/Ask bars for ``(instrument, granularity)`` over
    ``(start, end]`` in ascending time order, paging by ``chunk`` so the whole M1
    table is never materialised. Bid and Ask are separate rows at the same
    timestamp; a timestamp missing either side is skipped (treated as a gap)."""
    cursor = start
    while True:
        rows = (
            db.query(
                Candle.timestamp,
                Candle.price_type,
                Candle.open,
                Candle.high,
                Candle.low,
                Candle.close,
            )
            .filter(
                Candle.instrument_id == instrument_id,
                Candle.granularity == granularity,
                Candle.price_type.in_(("B", "A")),
                Candle.timestamp > cursor,
                Candle.timestamp <= end,
            )
            .order_by(Candle.timestamp.asc(), Candle.price_type.asc())
            .limit(chunk)
            .all()
        )
        if not rows:
            return

        groups: "OrderedDict[datetime, dict]" = OrderedDict()
        for r in rows:
            groups.setdefault(r.timestamp, {})[r.price_type] = r

        timestamps = list(groups)
        full_page = len(rows) == chunk
        # On a full page the last timestamp's pair may be split across the page
        # boundary — hold it back and re-fetch from it next iteration.
        if full_page and len(timestamps) > 1:
            timestamps = timestamps[:-1]

        for ts in timestamps:
            pair = groups[ts]
            bid = pair.get("B")
            ask = pair.get("A")
            if bid is not None and ask is not None:
                yield BidAskBar(
                    ts,
                    bid.open, bid.high, bid.low, bid.close,
                    ask.open, ask.high, ask.low, ask.close,
                )

        cursor = timestamps[-1]
        if not full_page:
            return


def _db_fallback_source(
    db: Session,
    instrument_id: int,
    granularity: str,
    start: datetime,
    end: datetime,
) -> list[BidAskBar]:
    """Degraded resolution when no M1 exists for the window. Uses the signal
    timeframe's Bid/Ask; if that timeframe has no Bid/Ask (e.g. D1 is Mid-only),
    falls back to Mid (bid=ask=mid). The horizon is a handful of bars, so this is
    materialised. Any exit resolved from here is flagged ``ambiguous_resolution``."""
    bars = list(_db_bidask_source(db, instrument_id, granularity, start, end))
    if bars:
        return bars

    mid_rows = (
        db.query(Candle)
        .filter(
            Candle.instrument_id == instrument_id,
            Candle.granularity == granularity,
            Candle.price_type == "M",
            Candle.timestamp > start,
            Candle.timestamp <= end,
        )
        .order_by(Candle.timestamp.asc())
        .all()
    )
    return [
        BidAskBar(
            r.timestamp,
            r.open, r.high, r.low, r.close,
            r.open, r.high, r.low, r.close,
        )
        for r in mid_rows
    ]


# ── small helpers ────────────────────────────────────────────────────────────
def _h4_bucket(ts: datetime, period_hours: float) -> int:
    """Integer id of the ``period_hours``-wide time bucket containing ``ts``,
    anchored to a fixed epoch. Only used to detect whether two timestamps fall
    in the SAME or a DIFFERENT bucket (see ``_run_core``) — the anchor's
    absolute value never matters, only bucket-id equality/inequality does."""
    seconds = (ts - _BUCKET_EPOCH).total_seconds()
    return int(seconds // (period_hours * 3600.0))


def _side(direction: str) -> int:
    d = direction.strip().lower()
    if d in _LONG_ALIASES:
        return 1
    if d in _SHORT_ALIASES:
        return -1
    raise ValueError(f"unrecognised direction {direction!r}; expected buy/long or sell/short")


def _naive(dt: datetime) -> datetime:
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt


def _hours(a: datetime, b: datetime) -> float:
    return (a - b).total_seconds() / 3600.0
