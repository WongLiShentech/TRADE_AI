"""
Rule-based signal engine — Phase 1 implementation.

Five-condition confluence:
  C1 trend     : D1 close on the correct side of D1 SMA(SIGNAL_TREND_SMA_PERIOD)
  C2 rsi       : H4 RSI(14) inside the configured pullback band
  C3 structure : H4 latest candle within SIGNAL_STRUCTURE_ATR_BUFFER * ATR(14)
                 of a recent swing low (BUY) or swing high (SELL)
  C4 session   : current UTC session in SIGNAL_SESSION_FILTER
  C5 spread    : current spread (pips) below SIGNAL_MAX_SPREAD_PIPS

Score is sum of conditions met. Signal fires when score >= SIGNAL_MIN_CONFLUENCE_SCORE.
If both BUY and SELL clear the threshold the signal is suppressed (ambiguous).

Gating (before any scoring):
  - Cooldown: any non-EXPIRED/REJECTED signal for this instrument inside the last
    SIGNAL_COOLDOWN_BARS_AFTER_CLOSE * 4 hours blocks evaluation.
  - Pre-weekend: within SIGNAL_NO_TRADE_HOURS_BEFORE_FRIDAY_CLOSE hours of
    Friday 22:00 UTC, evaluation is skipped.

Entry/stop/target:
  BUY  : entry = tick.ask, stop = entry - (SIGNAL_STOP_ATR_MULTIPLIER * atr14)
  SELL : entry = tick.bid, stop = entry + (SIGNAL_STOP_ATR_MULTIPLIER * atr14)
  target distance = MIN_RR_RATIO * |entry - stop|
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy.orm import Query, Session

from app.config import Settings
from app.models.candle import Candle
from app.models.indicator import Indicator
from app.models.instrument import Instrument
from app.models.signal import Signal
from app.domain.conditions import ConditionContext, enabled_conditions
from app.services.price_stream import get_latest_price
from app.services.signal_engine.base import SignalEngine, SignalOutput

logger = logging.getLogger(__name__)

_H4_HOURS = 4
_BLOCKING_STATUSES = ("PENDING", "APPROVED", "EXECUTED")


@dataclass(frozen=True)
class BacktestQuote:
    """Historical quote injected by the M7 runner in place of the live tick cache.

    In backtest mode the "current" quote is the decision bar's H4 Bid/Ask CLOSE
    (``bid`` = Bid close, ``ask`` = Ask close), so a BUY fills at ``ask`` and a SELL
    at ``bid`` — identical to how the live path consumes ``TickData.ask`` / ``.bid``.
    The c5 spread is ``ask - bid`` (Ask close − Bid close at the decision bar).
    """

    bid: float
    ask: float


def _as_of_filter(query: Query, as_of: Optional[datetime]) -> Query:
    """Additively bound a candle/indicator query to ``timestamp <= as_of`` when an
    as-of instant is given; a no-op when ``as_of`` is None (live path — unchanged)."""
    if as_of is not None:
        return query.filter(Candle.timestamp <= as_of)
    return query


def _as_of_filter_ind(query: Query, as_of: Optional[datetime]) -> Query:
    if as_of is not None:
        return query.filter(Indicator.timestamp <= as_of)
    return query


class RuleBasedSignalEngine(SignalEngine):
    def evaluate(
        self,
        instrument: str,
        granularity: str,
        db: Session,
        settings: Settings,
        *,
        as_of: Optional[datetime] = None,
        quote: Optional[BacktestQuote] = None,
    ) -> Optional[SignalOutput]:
        """Evaluate one instrument at a candle close.

        Live (default): ``as_of=None`` / ``quote=None`` — reads the latest stored
        indicator/candle rows and the in-memory tick cache. Behaviour is bit-identical
        to before this parameter existed.

        Backtest: the runner passes ``as_of=T`` (strictly after the decision bar's
        close) so every indicator/candle read is bounded ``timestamp <= T``, and
        ``quote`` supplies the decision bar's H4 Bid/Ask close in place of the live
        tick. No live-path branch is altered.
        """
        inst = db.query(Instrument).filter_by(symbol=instrument).first()
        if inst is None:
            logger.debug("evaluate: instrument %s not in DB", instrument)
            return None

        backtest = as_of is not None
        now_utc = as_of if as_of is not None else datetime.utcnow()

        # ── Gate: cooldown (live only) ────────────────────────────────────────
        # The live cooldown reads the Signal table by wall-clock created_at, which
        # is meaningless against a historical as_of (live rows created "now" would
        # spuriously match a past cutoff). In backtest the runner owns cooldown
        # (SIGNAL_COOLDOWN_BARS_AFTER_CLOSE after each simulated exit, per instrument).
        if not backtest:
            cooldown_hours = settings.SIGNAL_COOLDOWN_BARS_AFTER_CLOSE * _H4_HOURS
            cooldown_cutoff = now_utc - timedelta(hours=cooldown_hours)
            recent = (
                db.query(Signal)
                .filter(Signal.instrument_id == inst.id)
                .filter(Signal.status.in_(_BLOCKING_STATUSES))
                .filter(Signal.created_at >= cooldown_cutoff)
                .first()
            )
            if recent is not None:
                logger.debug("evaluate %s: cooldown — recent signal id=%s", instrument, recent.id)
                return None

        # ── Gate: pre-weekend ────────────────────────────────────────────────
        if _within_pre_friday_window(now_utc, settings.SIGNAL_NO_TRADE_HOURS_BEFORE_FRIDAY_CLOSE):
            logger.debug("evaluate %s: pre-Friday close cutoff", instrument)
            return None

        # ── Load H4 candles + indicator ──────────────────────────────────────
        h4_candles = (
            _as_of_filter(
                db.query(Candle).filter_by(
                    instrument_id=inst.id, granularity=granularity, price_type="M"
                ),
                as_of,
            )
            .order_by(Candle.timestamp.desc())
            .limit(100)
            .all()
        )
        if len(h4_candles) < 20:
            logger.debug("evaluate %s: insufficient %s candles (%d)", instrument, granularity, len(h4_candles))
            return None
        h4_candles.reverse()  # ascending
        latest_h4 = h4_candles[-1]

        latest_ind = (
            _as_of_filter_ind(
                db.query(Indicator).filter_by(instrument_id=inst.id, granularity=granularity),
                as_of,
            )
            .order_by(Indicator.timestamp.desc())
            .first()
        )
        # `atr14` is required unconditionally — the ENGINE needs it for stop geometry,
        # whatever conditions are enabled. Everything else is required only if some
        # enabled condition declares it, so a newly added (nullable, not-yet-backfilled)
        # indicator column cannot veto every bar for conditions that never read it.
        required = {"atr14"}
        for cond in enabled_conditions(settings):
            required.update(cond.required_indicators)
        if latest_ind is None:
            logger.debug("evaluate %s: no indicator row", instrument)
            return None
        missing = sorted(k for k in required if getattr(latest_ind, k, None) is None)
        if missing:
            logger.debug("evaluate %s: indicator row lacks %s", instrument, missing)
            return None
        atr14 = latest_ind.atr14

        # ── Load D1 trend ────────────────────────────────────────────────────
        # `as_of` forward-fills: the latest CLOSED D1 row <= T is used (D1 lags H4
        # by ~2 days; never require a same-day D1 bar).
        sma_period = settings.SIGNAL_TREND_SMA_PERIOD
        d1_candles = (
            _as_of_filter(
                db.query(Candle).filter_by(
                    instrument_id=inst.id,
                    granularity=settings.SIGNAL_TREND_TIMEFRAME,
                    price_type="M",
                ),
                as_of,
            )
            .order_by(Candle.timestamp.desc())
            .limit(sma_period)
            .all()
        )
        if len(d1_candles) < sma_period:
            logger.debug(
                "evaluate %s: insufficient %s candles for SMA%d (%d)",
                instrument,
                settings.SIGNAL_TREND_TIMEFRAME,
                sma_period,
                len(d1_candles),
            )
            return None
        d1_sma = sum(c.close for c in d1_candles) / len(d1_candles)
        d1_last_close = d1_candles[0].close  # newest

        # ── Quote: live tick (live) or injected decision-bar Bid/Ask (backtest) ──
        if backtest:
            if quote is None:
                logger.debug("evaluate %s: no backtest quote supplied", instrument)
                return None
            quote_bid, quote_ask = quote.bid, quote.ask
        else:
            tick = get_latest_price(instrument)
            if tick is None:
                logger.debug("evaluate %s: no tick in cache", instrument)
                return None
            quote_bid, quote_ask = tick.bid, tick.ask

        # ── Score both directions from the condition registry ────────────────
        # The conditions themselves live in `app/domain/conditions`. This method no
        # longer knows what they ARE — only how to assemble their inputs once and how to
        # combine their answers. Adding a condition is a registry entry plus a name in
        # SIGNAL_CONDITIONS, with no edit here and no second list to keep in sync (the
        # feature builder derives its payload keys from the same registry).
        ctx = ConditionContext(
            instrument=inst,
            granularity=granularity,
            now_utc=now_utc,
            settings=settings,
            bars=h4_candles,
            trend_bars=d1_candles,
            indicators=latest_ind,
            quote_bid=quote_bid,
            quote_ask=quote_ask,
            trend_close=d1_last_close,
            trend_sma=d1_sma,
        )
        active = enabled_conditions(settings)

        # NOTE: every enabled condition is scored, GATE or VOTE. That preserves today's
        # behaviour exactly — `session` and `spread` still contribute to the score even
        # though the registry correctly declares them direction-independent. Honouring
        # `role` (gates veto, only votes score) is the next change, and it is what fixes
        # the ambiguity defect: scored as votes these two give both directions the same
        # 2 points, so at a threshold of 2 EVERY bar satisfies BUY and SELL at once.
        buy_breakdown = {c.key: bool(c.evaluate(ctx, "BUY")) for c in active}
        sell_breakdown = {c.key: bool(c.evaluate(ctx, "SELL")) for c in active}
        buy_score = sum(1 for v in buy_breakdown.values() if v)
        sell_score = sum(1 for v in sell_breakdown.values() if v)

        threshold = settings.SIGNAL_MIN_CONFLUENCE_SCORE

        buy_pass = buy_score >= threshold
        sell_pass = sell_score >= threshold

        if buy_pass and sell_pass:
            logger.debug("evaluate %s: ambiguous (buy=%d sell=%d)", instrument, buy_score, sell_score)
            return None
        if not buy_pass and not sell_pass:
            logger.debug("evaluate %s: below threshold (buy=%d sell=%d, min=%d)",
                         instrument, buy_score, sell_score, threshold)
            return None

        stop_distance = settings.SIGNAL_STOP_ATR_MULTIPLIER * atr14
        if buy_pass:
            direction = "BUY"
            entry = quote_ask
            stop = entry - stop_distance
            target = entry + settings.MIN_RR_RATIO * stop_distance
            breakdown = buy_breakdown
            score = buy_score
        else:
            direction = "SELL"
            entry = quote_bid
            stop = entry + stop_distance
            target = entry - settings.MIN_RR_RATIO * stop_distance
            breakdown = sell_breakdown
            score = sell_score

        return SignalOutput(
            instrument=instrument,
            granularity=granularity,
            direction=direction,
            entry=entry,
            stop=stop,
            target=target,
            confidence_score=score,
            score_breakdown=breakdown,
        )


def _within_pre_friday_window(now_utc: datetime, hours_before_close: int) -> bool:
    """
    True if now_utc is within `hours_before_close` of the next Friday 22:00 UTC.
    Friday close is the standard FX week close (Sun 22:00 UTC open, Fri 22:00 UTC close).
    """
    # weekday(): Mon=0 ... Sun=6 — Friday is 4.
    days_until_friday = (4 - now_utc.weekday()) % 7
    friday_close = (now_utc + timedelta(days=days_until_friday)).replace(
        hour=22, minute=0, second=0, microsecond=0
    )
    if friday_close < now_utc:
        # we're past this week's Friday close — look at next week
        friday_close += timedelta(days=7)
    delta = friday_close - now_utc
    return delta <= timedelta(hours=hours_before_close)
