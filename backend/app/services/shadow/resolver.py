"""Shadow outcome resolver — M8-Shadow Phase 3.

Phase 2 writes one ``stage='shadow'`` ``Trade`` row per live signal with the model's
decision and the full feature dict, and deliberately leaves the outcome columns NULL.
This module fills them in: it walks the pending queue, re-uses the M7 triple-barrier
:func:`app.services.backtester.simulator.simulate` on live M1 Bid/Ask, and writes
``outcome`` / ``rr_actual`` / ``exit_price`` / ``exit_reason`` / ``closed_at`` /
``ambiguous_resolution`` — the same columns, produced by the same code path and
labelled by the same :func:`runner.label_outcome` convention as the 2,809-trade M7
backtest corpus. That identity is the whole point: a shadow row and a backtest row
must be directly comparable, not merely similar.

Both decisions are resolved
---------------------------
``ml_decision='skip'`` rows are resolved exactly like ``'take'`` rows, and so are
scoring-failure rows (``ml_decision IS NULL``). The skips are the counterfactual —
the only way to answer "did the filter decline trades that would have LOST?" is to
know how they would have turned out. Resolving takes only would make the corpus
survivorship-biased and the experiment unfalsifiable.

What makes a row RESOLVABLE (never guessed)
-------------------------------------------
A pending row is resolved only when its outcome is genuinely knowable:

1. **The hold horizon has elapsed in wall-clock time** —
   ``now >= T + base_hours`` from :func:`simulator.horizon_bounds`
   (``(SIGNAL_MAX_HOLD_BARS + 1) x period_hours``). A necessary condition: before
   this instant even a no-closure trade could not have reached its time exit.
2. **The M1 Bid/Ask actually covers the hold** — the stored M1 stream after T must
   contain at least ``SIGNAL_MAX_HOLD_BARS`` DISTINCT trading-timeframe buckets that
   are each DENSELY populated (:func:`observable_bars`). This mirrors the simulator's
   own ``bars_elapsed`` counting, so satisfying it guarantees the walk can reach
   either a barrier or the time exit rather than running off the right edge of the
   data.

Condition 2 is what makes this honest rather than optimistic. Wall-clock time alone
is not enough: a Thursday-entry trade's 10-bar horizon legitimately runs into the
following Monday, and resolving it at T+44h would hit the simulator's
truncated-right-edge branch and mislabel it as a ``time_exit`` at Friday's close.

Bucket DENSITY, not mere presence
---------------------------------
Counting a bucket as observable because it holds ONE complete Bid+Ask minute would
make a bucket with 3 of 240 minutes worth exactly as much as a full one. The
simulator would then walk a stream full of holes, sail straight past the minute the
price actually touched SL or TP, and write a confident label for an exit that never
happened — the worst possible failure mode here, because it is silent and it
contaminates training data.

So a bucket only counts when at least ``SHADOW_MIN_BUCKET_M1_DENSITY`` of its
minutes are present as COMPLETE Bid+Ask bars (see :func:`min_bars_per_bucket`). The
floor is a fraction of the timeframe's own minute count, so it is timeframe-agnostic
by construction: 0.2 means 48 of 240 minutes on H4 and 288 of 1440 on D1.

**Escape valve.** Once ``now >= T + fetch_cap_hours`` (the simulator's own
closure-proof cap, ~7 days on H4) a row is resolved even if condition 2 is still
unmet — an M1 gap that large is a real data gap, not a weekend. Such a resolution is
counted as ``forced`` and is ALWAYS written with ``ambiguous_resolution=True``, even
when the simulator itself found a clean barrier: the coverage that would make that
barrier trustworthy is precisely what was missing. The row is therefore labelled
honestly and flagged for relabel rather than sitting in the queue forever.

Idempotency
-----------
The queue selector is ``stage='shadow' AND closed_at IS NULL``, and resolving a row
always sets ``closed_at`` — so a resolved row can never be selected again and a
re-run is a strict no-op. Each row is committed on its own: a failure cannot roll
back a sibling's already-written outcome.

Failure isolation
-----------------
Every row is processed in its own ``try/except``. One bad row (corrupt reasoning
JSON, missing ATR, a simulator edge case) is logged, counted and skipped — it can
never stall the queue behind it. Nothing here places an order or touches a broker.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from typing import Optional, Sequence

from sqlalchemy import distinct, func
from sqlalchemy.orm import Session

from app.config import Settings
from app.domain.timeframes import get_timeframe
from app.models.candle import Candle
from app.models.strategy import Strategy
from app.models.trade import Trade
from app.models.trade_path import TradePath
from app.services.backtester.runner import label_outcome
from app.services.backtester.simulator import horizon_bounds, simulate
from app.services.shadow.recorder import REASONING_SHADOW_KEY, STAGE_SHADOW
from app.services.strategy_registry import settings_for

logger = logging.getLogger(__name__)

# Feature key holding ATR(14) on the trading timeframe at T. The simulator freezes it
# as the trailing-stop distance basis for the trade's whole life.
_ATR_KEY = "atr14"
# Sub-key of the namespaced shadow metadata carrying the timeframe the signal fired
# on. Written by the recorder; read back here so the resolver never assumes H4.
_GRANULARITY_KEY = "granularity"
# Both M1 sides are required: the simulator joins Bid and Ask at a shared timestamp
# and skips any timestamp missing either side.
_EXECUTION_PRICE_TYPES = ("B", "A")
# The intrabar granularity the simulator resolves barriers on. Mirrors
# simulator._INTRABAR_GRANULARITY; kept as a named constant, never inline.
_INTRABAR_GRANULARITY = "M1"

_SECONDS_PER_HOUR = 3600.0
_MINUTES_PER_HOUR = 60
# Fixed anchor for bucket arithmetic — must be the SAME anchor the simulator uses
# (simulator._BUCKET_EPOCH), because a bucket id is only meaningful relative to it.
# Postgres' extract(epoch FROM timestamp) is measured from this instant too.
_BUCKET_EPOCH = datetime(1970, 1, 1)


# ── public API ───────────────────────────────────────────────────────────────
def resolve_pending(
    db: Session,
    settings: Settings,
    now: Optional[datetime] = None,
    *,
    instrument_ids: Optional[Sequence[int]] = None,
    opened_from: Optional[datetime] = None,
    opened_to: Optional[datetime] = None,
    limit: Optional[int] = None,
) -> dict:
    """Resolve every pending shadow row whose outcome is now knowable.

    Args:
        db: SQLAlchemy session. Committed once per resolved row (see "Idempotency").
        settings: config; supplies ``SIGNAL_MAX_HOLD_BARS`` and the trailing/lock
            parameters the simulator reads. Nothing is hardcoded here.
        now: the instant to evaluate resolvability against; defaults to
            ``datetime.utcnow()``. Injectable so a caller (or a test) can reason
            about the horizon without patching the clock. Naive UTC, matching every
            stored timestamp in the schema.
        instrument_ids: optional scope — resolve only these instruments' rows.
            ``None`` (default, and what the scheduled job uses) means every
            instrument. Instrument-agnostic: this is an id list, never a symbol.
        opened_from: optional inclusive lower bound on ``opened_at`` (T). ``None``
            (default, and what the scheduled job uses) means unbounded.
        opened_to: optional inclusive upper bound on ``opened_at`` (T). Together with
            ``opened_from`` this bounds a pass to a specific signal-time window —
            used for targeted re-resolution, and by the test suite so a run can only
            ever touch its own sentinel rows.
        limit: optional cap on how many pending rows to consider in this pass
            (oldest first). ``None`` (default) processes the whole queue.

    Returns:
        A summary dict::

            {
              "checked":       rows inspected,
              "resolved":      rows written (outcome + closed_at now set),
              "not_yet":       horizon not elapsed / M1 coverage insufficient,
              "forced":        SUBSET of "resolved": rows resolved past the closure
                               cap with insufficient M1 coverage. The simulator
                               resolves these degraded and flags them
                               ambiguous_resolution=True. A non-zero value here is an
                               M1 ingestion alarm, not a normal outcome.
              "no_atr":        left pending: atr14 missing/NaN/<=0 in the feature dict,
              "failed":        left pending: an exception was caught and logged,
              "still_pending": rows still awaiting resolution after this pass,
              "by_outcome":    {"win": n, "loss": n, "breakeven": n},
              "by_decision":   {"take": n, "skip": n, "unscored": n} over resolved rows,
            }

    Never raises for a per-row problem — those are counted and logged.
    """
    # A strategy's params can change between passes; a stale projection would grade
    # rows by an exit rule the strategy no longer has.
    _STRATEGY_SETTINGS_CACHE.clear()
    evaluated_at = _naive(now) if now is not None else datetime.utcnow()
    query = (
        db.query(Trade)
        .filter(Trade.stage == STAGE_SHADOW, Trade.closed_at.is_(None))
        .order_by(Trade.opened_at.asc())
    )
    if instrument_ids is not None:
        query = query.filter(Trade.instrument_id.in_(list(instrument_ids)))
    if opened_from is not None:
        query = query.filter(Trade.opened_at >= _naive(opened_from))
    if opened_to is not None:
        query = query.filter(Trade.opened_at <= _naive(opened_to))
    if limit is not None:
        query = query.limit(int(limit))
    pending = query.all()

    summary = {
        "checked": len(pending),
        "resolved": 0,
        "not_yet": 0,
        "forced": 0,
        "no_atr": 0,
        "failed": 0,
        "still_pending": 0,
        "by_outcome": {"win": 0, "loss": 0, "breakeven": 0},
        "by_decision": {"take": 0, "skip": 0, "unscored": 0},
    }

    for trade in pending:
        try:
            status = _resolve_one(db, settings, trade, evaluated_at, summary)
        except Exception as exc:  # noqa: BLE001 — one bad row must never stall the queue
            db.rollback()
            summary["failed"] += 1
            logger.warning(
                "shadow resolve failed for trade id=%s instrument_id=%s opened_at=%s "
                "(%s: %s) — left pending",
                trade.id, trade.instrument_id, trade.opened_at, type(exc).__name__, exc,
            )
            continue
        if status != "resolved":
            summary[status] += 1

    summary["still_pending"] = summary["checked"] - summary["resolved"]
    logger.info("shadow resolver: %s", summary)
    return summary


def min_bars_per_bucket(settings: Settings, granularity: str) -> int:
    """Minimum complete Bid+Ask M1 bars a trading-TF bucket needs to count as observable.

    Derived, never hardcoded: ``ceil(SHADOW_MIN_BUCKET_M1_DENSITY x minutes_in_bucket)``
    where ``minutes_in_bucket`` comes from the Timeframe registry. Expressing the floor
    as a FRACTION is what makes it timeframe-agnostic — the same config value means
    48/240 minutes on H4 and 288/1440 on D1 without a per-timeframe table.

    Args:
        settings: config supplying ``SHADOW_MIN_BUCKET_M1_DENSITY`` (0.0-1.0).
        granularity: the trading timeframe whose buckets are being measured.

    Returns:
        The per-bucket bar floor, never below 1 — a bucket with no complete Bid+Ask
        minute at all can never be observable, whatever the configured density.
    """
    minutes = int(get_timeframe(granularity).period_hours * _MINUTES_PER_HOUR)
    density = float(settings.SHADOW_MIN_BUCKET_M1_DENSITY)
    return max(1, math.ceil(density * minutes))


def observable_bars(
    db: Session,
    instrument_id: int,
    signal_time: datetime,
    granularity: str,
    upper_bound: datetime,
    settings: Settings,
) -> int:
    """How many DENSELY-COVERED trading-timeframe bars the stored M1 stream shows after T.

    This is the resolver's data-sufficiency test, and it is deliberately a mirror of
    the simulator's ``_run_core`` counter rather than an approximation of it. The
    simulator starts at ``bucket(T)`` and increments ``bars_elapsed`` once per bucket
    CHANGE observed in the M1 stream — so a weekend, which contains no M1 bars at
    all, costs exactly one step instead of the several wall-clock bars a naive
    ``elapsed_hours / period_hours`` would charge. Counting distinct buckets in the
    stored data (excluding the signal's own bucket) reproduces that number exactly.

    Two conditions make a bucket count:

    1. Only timestamps carrying BOTH M1 sides are counted, because the simulator's
       Bid/Ask join silently skips a timestamp that is missing either one.
    2. The bucket must hold at least :func:`min_bars_per_bucket` such timestamps.
       Without this floor a bucket holding 3 of 240 minutes would be worth as much as
       a full one, and the simulator could walk straight past the minute price
       actually touched SL or TP — writing a confident label for an exit that never
       happened.

    Args:
        db: SQLAlchemy session.
        instrument_id: candle FK.
        signal_time: T (naive UTC).
        granularity: the trading timeframe whose bars are being counted (e.g. ``"H4"``).
        upper_bound: right edge of the count window — pass the simulator's
            closure-proof cap so this never scans more than the walk could consume.
        settings: config supplying the per-bucket density floor.

    Returns:
        Count of distinct trading-timeframe buckets strictly after ``signal_time``'s
        own bucket that carry complete Bid+Ask M1 data at or above the density floor.
    """
    bucket_seconds = get_timeframe(granularity).period_hours * _SECONDS_PER_HOUR
    signal_bucket = math.floor(
        (_naive(signal_time) - _BUCKET_EPOCH).total_seconds() / bucket_seconds
    )
    bucket = func.floor(func.extract("epoch", Candle.timestamp) / bucket_seconds).label("b")
    complete_bars = (
        db.query(bucket)
        .filter(
            Candle.instrument_id == int(instrument_id),
            Candle.granularity == _INTRABAR_GRANULARITY,
            Candle.price_type.in_(_EXECUTION_PRICE_TYPES),
            Candle.timestamp > _naive(signal_time),
            Candle.timestamp <= _naive(upper_bound),
        )
        .group_by(Candle.timestamp)
        .having(func.count(distinct(Candle.price_type)) == len(_EXECUTION_PRICE_TYPES))
        .subquery()
    )
    # Second aggregation: minutes-per-bucket, then keep only the DENSE buckets.
    dense_buckets = (
        db.query(complete_bars.c.b)
        .filter(complete_bars.c.b != signal_bucket)
        .group_by(complete_bars.c.b)
        .having(func.count() >= min_bars_per_bucket(settings, granularity))
        .subquery()
    )
    count = db.query(func.count(distinct(dense_buckets.c.b))).scalar()
    return int(count or 0)


# ── internals ────────────────────────────────────────────────────────────────
# Cache per resolver pass: a run resolves many rows and they overwhelmingly share a
# handful of strategies. Keyed by strategy id, so a config change between passes is
# picked up on the next one.
_STRATEGY_SETTINGS_CACHE: dict[int, Settings] = {}


def _settings_for_trade(db: Session, settings: Settings, trade: Trade) -> Settings:
    """Settings as the strategy that produced ``trade`` — global config if unknown."""
    sid = getattr(trade, "strategy_id", None)
    if sid is None:
        return settings
    cached = _STRATEGY_SETTINGS_CACHE.get(sid)
    if cached is not None:
        return cached
    strategy = db.get(Strategy, sid)
    if strategy is None:
        logger.warning(
            "shadow resolve: trade id=%s names strategy %s which no longer exists — "
            "grading with the global config", trade.id, sid,
        )
        return settings
    projected = settings_for(settings, strategy)
    _STRATEGY_SETTINGS_CACHE[sid] = projected
    return projected


def _resolve_one(
    db: Session,
    settings: Settings,
    trade: Trade,
    now: datetime,
    summary: dict,
) -> str:
    """Resolve one pending row. Returns the summary key describing what happened.

    ``"resolved"`` means the outcome columns were written and committed; anything
    else leaves the row untouched and pending.
    """
    # Grade the row by the exit rule of the strategy that PRODUCED it, not by whatever
    # the process happens to be configured with.
    #
    # With one strategy those were the same thing. With two they are not, and the
    # difference is not cosmetic: relabelling one corpus under a pure-barrier exit
    # instead of a trailing one flips 12.5% of outcomes. Grading every row by the
    # global config would score a pure-barrier strategy's trades under a trailing rule
    # — and the resulting labels would train the next model on a world that never
    # existed.
    #
    # Falls back to the global settings when the row is unattributed, which is the
    # old behaviour and the only sensible default for a row whose strategy is unknown.
    settings = _settings_for_trade(db, settings, trade)

    signal_time = _naive(trade.opened_at)
    granularity = _granularity_of(trade, settings)
    base_hours, cap_hours = horizon_bounds(settings, granularity)

    # (1) necessary wall-clock condition — even a closure-free trade cannot be over.
    if now < signal_time + timedelta(hours=base_hours):
        return "not_yet"

    # (2) sufficient data condition — enough observable trading bars in stored M1 for
    #     the simulator's walk to reach a barrier or its time exit.
    cap_edge = signal_time + timedelta(hours=cap_hours)
    required_bars = int(settings.SIGNAL_MAX_HOLD_BARS)
    bars = observable_bars(db, trade.instrument_id, signal_time, granularity, cap_edge, settings)
    sufficient = bars >= required_bars
    # Escape valve: past the simulator's own closure cap, missing M1 is a real data
    # gap, not a weekend. Resolve anyway — leaving it pending forever is not honest
    # either. ``forced`` counts ONLY these genuinely-thin resolutions.
    forced = not sufficient
    if not sufficient and now < cap_edge:
        logger.debug(
            "shadow resolve: trade id=%s has %d/%d observable %s bars of M1 (density "
            "floor %d bars/bucket) — pending until coverage lands or %s (closure cap)",
            trade.id, bars, required_bars, granularity,
            min_bars_per_bucket(settings, granularity), cap_edge,
        )
        return "not_yet"

    atr14 = _atr_of(trade)
    if atr14 is None:
        logger.warning(
            "shadow resolve: trade id=%s has no usable '%s' in signal_reasoning — left "
            "pending (the trailing-stop distance basis cannot be invented)",
            trade.id, _ATR_KEY,
        )
        return "no_atr"

    result = simulate(
        db, settings, trade.instrument_id, trade.direction,
        float(trade.entry_price), float(trade.stop_price), float(trade.tp_price),
        signal_time, atr14, granularity,
        record_path=bool(settings.PATH_RECORDING_ENABLED),
    )

    trade.exit_price = float(result.exit_price)
    trade.rr_actual = float(result.rr_actual)
    trade.exit_reason = result.exit_reason
    # Excursion (Phase A). Recorded on the same walk that produced the outcome, so
    # the two can never describe different price streams. Written BEFORE the commit
    # below, in the same transaction: a row whose outcome is set but whose path is
    # missing would be indistinguishable from one that legitimately had no path,
    # and nothing would ever come back to fill it in.
    _persist_path(db, trade, result)
    # forced ⇒ ambiguous, unconditionally. The simulator flags its own degraded
    # branches, but a thin stream can also produce a CLEAN-looking barrier: the walk
    # simply never saw the minutes that would have contradicted it. The coverage that
    # would make that barrier trustworthy is exactly what was missing, so the label is
    # marked untrustworthy here rather than inheriting the simulator's optimism.
    trade.ambiguous_resolution = bool(result.ambiguous_resolution) or forced
    trade.closed_at = _naive(result.exit_time)
    trade.outcome = label_outcome(result.rr_actual)
    db.commit()

    summary["resolved"] += 1
    if forced:
        summary["forced"] += 1
    summary["by_outcome"][trade.outcome] = summary["by_outcome"].get(trade.outcome, 0) + 1
    decision_key = trade.ml_decision if trade.ml_decision else "unscored"
    summary["by_decision"][decision_key] = summary["by_decision"].get(decision_key, 0) + 1
    logger.info(
        "shadow resolved: trade id=%s %s decision=%s rr=%.3f outcome=%s reason=%s%s",
        trade.id, trade.direction, trade.ml_decision or "unscored",
        result.rr_actual, trade.outcome, result.exit_reason,
        " (FORCED past closure cap)" if forced else "",
    )
    return "resolved"


def _persist_path(db: Session, trade: Trade, result) -> None:
    """Write the excursion scalars onto the trade and its per-bar path rows.

    Idempotent: re-resolving a trade replaces its path rather than appending a
    second copy. Deleting first (rather than upserting) is deliberate — a
    re-resolution may legitimately produce FEWER bars than before, and rows left
    behind from the longer previous walk would silently extend the new path with
    stale data.

    A trade with no recorded path (recording disabled, or a walk that yielded
    nothing) leaves ``mfe_r``/``mae_r`` NULL. NULL means "not measured"; 0.0 would
    claim "never moved", which is a different and possibly false statement.
    """
    if not result.path:
        return

    db.query(TradePath).filter(TradePath.trade_id == trade.id).delete(synchronize_session=False)
    db.bulk_save_objects(
        [
            TradePath(
                trade_id=trade.id,
                bar=p.bar,
                r_close=p.r_close,
                r_best=p.r_best,
                r_worst=p.r_worst,
                mfe_r=p.mfe_r,
                mae_r=p.mae_r,
                beyond_exit=p.beyond_exit,
                degraded=p.degraded,
            )
            for p in result.path
        ]
    )
    trade.mfe_r = None if result.mfe_r is None else float(result.mfe_r)
    trade.mae_r = None if result.mae_r is None else float(result.mae_r)
    trade.path_truncated = bool(result.path_truncated)


def _granularity_of(trade: Trade, settings: Settings) -> str:
    """The timeframe this shadow signal fired on.

    Read from the recorder's namespaced ``signal_reasoning['shadow']['granularity']``.
    Falls back to the first entry of ``SIGNAL_GRANULARITIES`` — still config-driven,
    never a hardcoded ``"H4"`` — so a row written before that key existed still
    resolves on the configured trading timeframe.
    """
    reasoning = trade.signal_reasoning or {}
    shadow_meta = reasoning.get(REASONING_SHADOW_KEY) or {}
    value = shadow_meta.get(_GRANULARITY_KEY)
    if isinstance(value, str) and value:
        return value
    for candidate in settings.SIGNAL_GRANULARITIES.split(","):
        candidate = candidate.strip()
        if candidate:
            return candidate
    raise ValueError(
        "cannot determine the trading timeframe for this shadow row: neither "
        f"signal_reasoning['{REASONING_SHADOW_KEY}']['{_GRANULARITY_KEY}'] nor "
        "SIGNAL_GRANULARITIES yields one"
    )


def _atr_of(trade: Trade) -> Optional[float]:
    """ATR(14) at T off the persisted feature dict, or ``None`` if unusable.

    ``None`` (missing key, JSON null from a NaN feature, or a non-positive value)
    leaves the row PENDING rather than substituting a default: the trailing-stop
    distance is ``BACKTEST_TRAILING_DISTANCE_ATR_MULT x atr14``, so inventing an ATR
    would invent an exit — and therefore a training label.
    """
    reasoning = trade.signal_reasoning or {}
    value = reasoning.get(_ATR_KEY)
    if value is None:
        return None
    try:
        atr = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(atr) or math.isinf(atr) or atr <= 0.0:
        return None
    return atr


def _naive(dt: datetime) -> datetime:
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt
