"""M7 backtest runner + three-fold expanding walk-forward CV.

:func:`run_backtest` walks every H4 decision bar in the configured execution
window, re-uses the LIVE ``RuleBasedSignalEngine`` as-of that bar (bit-identical
gating, evaluated at a historical instant), applies the live SIGNAL-QUALITY risk
gates, computes the LOCKED feature contract through
``feature_builder.build_features`` (never inline), simulates each fired signal to
exit through the triple-barrier ``simulator.simulate``, and persists one
``Trade`` row (``stage='backtest'``) with the full feature dict as
``signal_reasoning``.

Single walk, fold slicing afterward
-----------------------------------
The window is walked ONCE. The three expanding OOS folds are derived by slicing
the persisted trades by ``signal_time`` — we never re-simulate per fold. Fold k
has ``IS = [start, Tk]`` and ``OOS = [Tk + embargo, T(k+1) | end]``; a trade that
lands in the embargo gap ``(Tk, Tk+embargo)`` belongs to the IS side only and is
in NO OOS set. The OOS slices feed :func:`metrics.evaluate_promotion_gate`.

Sizing note (labels are size-independent)
-----------------------------------------
``rr_actual`` (the ground-truth label) is a risk-normalised R-multiple and does
NOT depend on account size. Pip value is fetched ONCE per instrument at runner
start via ``BrokerRouter`` (never per-trade — that would be hundreds of live API
calls) and cached; position size is computed against ``STARTING_BALANCE`` purely
for the audit ``units`` / ``risk_amount`` columns. If ``units`` floors below
``OANDA_MIN_UNITS`` (or pip value is unavailable), the trade row is STILL recorded
— dropping it would make the training set depend on account size. Sizing shortfalls
are logged, not written into the feature dict (which stays a pure feature store).

Zero hardcoding: window, folds, embargo, hold, gate thresholds, weekly-cap policy,
trading timeframe and session filter are all env-driven (``Settings``).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Sequence

from sqlalchemy import func, text
from sqlalchemy.orm import Session

from app.brokers.router import get_broker_router
from app.config import Settings
from app.domain.timeframes import get_timeframe
from app.models.backtest_run import BacktestRun
from app.models.candle import Candle
from app.models.instrument import Instrument
from app.models.trade import Trade
from app.services.backtester import metrics as M
from app.services.backtester.simulator import simulate
from app.services.feature_builder import FEATURE_SCHEMA_VERSION, build_features, json_safe
from app.services.position_sizer import PositionSizer
from app.services.signal_engine.factory import get_signal_engine
from app.services.signal_engine.rule_based import BacktestQuote

logger = logging.getLogger(__name__)

_STRATEGY = "rule_based_v1"
_SIGNAL_SOURCE = "rule_based"
_STAGE = "backtest"
_STOP_METHOD = "atr"
_COMMIT_BATCH = 200
_PROGRESS_EVERY = 5_000

# Config execution-window bounds are day-granular (dates), so the first/last intraday
# Bid/Ask bar legitimately sits a few hours inside the window. This slack validates
# that Bid/Ask genuinely SPANS the window without inferring the window from the data.
_WINDOW_COVERAGE_SLACK = timedelta(days=4)

# The Bid/Ask price types the runner (and simulator) require over the window.
_REQUIRED_COVERAGE = (("H4", "B"), ("H4", "A"), ("M1", "B"), ("M1", "A"))


# ── result container ─────────────────────────────────────────────────────────
@dataclass
class BacktestResult:
    """Structured outcome of one :func:`run_backtest` call (returned + surfaced by
    the API). ``fold_breakdown`` mirrors what is persisted on the BacktestRun row."""

    run_id: int
    passed: bool
    instruments: list[str]
    window_start: datetime
    window_end: datetime
    n_trials: int
    runtime_seconds: float
    decision_bars_evaluated: int
    signals_fired: int
    trades_simulated: int
    rejections: dict[str, int]
    fold_breakdown: dict
    combined_oos: dict


# ── pure, testable helpers ───────────────────────────────────────────────────
def label_outcome(rr: float) -> str:
    """Trade outcome label from the realised R-multiple (matches the M7 spec):
    ``win`` if rr > +0.05, ``loss`` if rr < -0.05, else ``breakeven``."""
    if rr > 0.05:
        return "win"
    if rr < -0.05:
        return "loss"
    return "breakeven"


def iso_week_key(dt: datetime) -> str:
    """ISO-year+week key (e.g. ``2025-W07``) for the weekly-cap counter."""
    y, w, _ = dt.isocalendar()
    return f"{y:04d}-W{w:02d}"


def weekly_cap_blocks(
    week_counts: dict[str, int],
    week_key: str,
    max_per_week: int,
    respect_cap: bool,
) -> bool:
    """Whether a new trade in ``week_key`` is blocked by the weekly cap.

    ``BACKTEST_RESPECT_WEEKLY_CAP == False`` (owner decision) → never blocks: the
    weekly cap is a LIVE throttle, not a signal-quality property, so the training
    set must capture every rule-valid trade. When True, blocks once the week's
    count reaches ``MAX_TRADES_PER_WEEK``.
    """
    if not respect_cap:
        return False
    return week_counts.get(week_key, 0) >= max_per_week


@dataclass(frozen=True)
class FoldWindow:
    """One expanding walk-forward fold's IS/OOS time bounds."""

    is_start: datetime
    is_end: datetime
    oos_start: datetime
    oos_end: datetime
    is_last: bool


def fold_windows(
    window_start: datetime,
    window_end: datetime,
    fold_bounds: Sequence[float],
    embargo_delta: timedelta,
) -> list[FoldWindow]:
    """Build expanding-IS / embargoed-OOS fold windows from fractional bounds.

    ``fold_bounds`` are fractions of the window (e.g. 0.333, 0.667, 0.875) giving
    boundaries ``Tk = start + f_k * (end - start)``. Fold k: ``IS = [start, Tk]``,
    ``OOS = [Tk + embargo, T(k+1)]`` with ``T(last+1) = window_end``. OOS windows
    are disjoint (separated by the embargo gap); IS expands monotonically.
    """
    span = window_end - window_start
    boundaries = [window_start + span * float(f) for f in fold_bounds]
    out: list[FoldWindow] = []
    for i, tk in enumerate(boundaries):
        oos_end = boundaries[i + 1] if i + 1 < len(boundaries) else window_end
        out.append(
            FoldWindow(
                is_start=window_start,
                is_end=tk,
                oos_start=tk + embargo_delta,
                oos_end=oos_end,
                is_last=(i + 1 == len(boundaries)),
            )
        )
    return out


def _in_is(t: datetime, fw: FoldWindow) -> bool:
    return fw.is_start <= t <= fw.is_end


def _in_oos(t: datetime, fw: FoldWindow) -> bool:
    # Half-open [oos_start, oos_end) so a trade exactly on the next boundary falls
    # in the next fold's embargo gap (in NO OOS); the last fold includes its end.
    if fw.is_last:
        return fw.oos_start <= t <= fw.oos_end
    return fw.oos_start <= t < fw.oos_end


def slice_folds(trades: Sequence, windows: Sequence[FoldWindow]) -> list[dict]:
    """Slice trade records (each exposing ``signal_time``) into per-fold IS/OOS
    sets by ``signal_time``. Returns one ``{"is": [...], "oos": [...]}`` per fold.
    A trade inside an embargo gap appears in NO fold's OOS.
    """
    out: list[dict] = []
    for fw in windows:
        is_set = [t for t in trades if _in_is(_sig_time(t), fw)]
        oos_set = [t for t in trades if _in_oos(_sig_time(t), fw)]
        out.append({"is": is_set, "oos": oos_set})
    return out


def _sig_time(trade) -> datetime:
    if isinstance(trade, dict):
        return trade["signal_time"]
    return trade.signal_time


# ── main entry point ─────────────────────────────────────────────────────────
def run_backtest(
    db: Session,
    settings: Settings,
    instruments: Optional[list[str]] = None,
    strategy_id: Optional[int] = None,
) -> BacktestResult:
    """Run the full M7 walk-forward backtest and persist the results.

    Args:
        db: SQLAlchemy session (reads candles/indicators/macro/news; writes Trade +
            BacktestRun rows).
        settings: config — every threshold/window/fold value is read from here.
        instruments: optional subset of instrument symbols; defaults to all
            ``is_active=True`` instruments.

    Returns:
        A :class:`BacktestResult` (also persisted as one ``BacktestRun`` row).

    Raises:
        RuntimeError: if the universe is empty or Bid/Ask coverage does not span the
            execution window (fail loud — never infer the window from min/max).
    """
    t0 = time.monotonic()

    trading_tf = _first_granularity(settings)
    period_hours = get_timeframe(trading_tf).period_hours
    period_delta = timedelta(hours=period_hours)
    window_start = _naive(settings.EXECUTION_WINDOW_START)
    window_end = _naive(settings.EXECUTION_WINDOW_END)
    horizon_hours = (int(settings.SIGNAL_MAX_HOLD_BARS) + 1) * period_hours
    horizon_delta = timedelta(hours=horizon_hours)
    embargo_delta = timedelta(hours=int(settings.BACKTEST_EMBARGO_BARS) * period_hours)
    cooldown_delta = timedelta(hours=int(settings.SIGNAL_COOLDOWN_BARS_AFTER_CLOSE) * period_hours)
    balance = float(settings.STARTING_BALANCE)

    insts = _resolve_universe(db, instruments)
    logger.info(
        "backtest: universe=%d [%s] window=%s..%s tf=%s",
        len(insts), ",".join(i.symbol for i in insts), window_start, window_end, trading_tf,
    )
    _assert_coverage(db, insts, trading_tf, window_start, window_end)

    pip_values = _cache_pip_values(db, insts, settings)
    n_trials = _durable_n_trials(db)
    engine = get_signal_engine(settings)
    sizer = PositionSizer()

    records: list[dict] = []          # {signal_time, rr_actual, holding_hours} for metrics
    rejections = _new_rejections()
    decision_bars = 0
    signals_fired = 0
    trades_simulated = 0
    pending = 0
    week_counts: dict[str, int] = {}

    for inst in insts:
        symbol = inst.symbol
        pip_value = pip_values.get(inst.id)
        mid_bars = _prefetch_mid_bars(db, inst.id, trading_tf, window_start, window_end, period_delta)
        bid_close = _prefetch_close(db, inst.id, trading_tf, "B", window_start, window_end, period_delta)
        ask_close = _prefetch_close(db, inst.id, trading_tf, "A", window_start, window_end, period_delta)
        next_allowed: Optional[datetime] = None  # one-open-trade + post-exit cooldown gate

        for bar_open, _mid_close in mid_bars:
            bar_close = bar_open + period_delta
            if bar_close < window_start or bar_close > window_end:
                continue
            decision_bars += 1
            if decision_bars % _PROGRESS_EVERY == 0:
                logger.info("backtest: %d decision-bars evaluated, %d trades so far",
                            decision_bars, trades_simulated)

            t_sig = bar_close + timedelta(seconds=1)  # strictly after close (feature_builder convention)

            # Right-edge guard: need M1 coverage over [T, T+horizon] inside the window.
            if t_sig + horizon_delta > window_end:
                rejections["near_window_end"] += 1
                continue
            # One-open-trade-per-instrument + cooldown after a simulated exit.
            if next_allowed is not None and t_sig < next_allowed:
                rejections["cooldown_or_open"] += 1
                continue
            wk = iso_week_key(t_sig)
            if weekly_cap_blocks(week_counts, wk, settings.MAX_TRADES_PER_WEEK, settings.BACKTEST_RESPECT_WEEKLY_CAP):
                rejections["weekly_cap"] += 1
                continue

            bid = bid_close.get(bar_open)
            ask = ask_close.get(bar_open)
            if bid is None or ask is None:
                rejections["no_quote"] += 1  # Mid-warmup edge / missing Bid or Ask bar
                continue

            signal = engine.evaluate(
                symbol, trading_tf, db, settings,
                as_of=t_sig, quote=BacktestQuote(bid=bid, ask=ask),
            )
            if signal is None:
                continue
            signals_fired += 1

            # ── SIGNAL-QUALITY risk gates (mirror live RiskEngine, minus live-only
            #    correlation/Signal-table checks and per-trade broker calls) ──────
            risk = abs(signal.entry - signal.stop)
            reward = abs(signal.target - signal.entry)
            if risk == 0.0 or reward / risk < settings.MIN_RR_RATIO:
                rejections["insufficient_rr"] += 1
                continue

            features = build_features(inst, t_sig, trading_tf, signal.score_breakdown, db, settings)
            atr14 = features.get("atr14")
            if atr14 is None or _isnan(atr14) or atr14 <= 0:
                rejections["no_atr"] += 1
                continue
            if risk > settings.ATR_MULTIPLIER_MAX * atr14:
                rejections["stop_too_wide"] += 1
                continue

            # ── simulate to exit ──────────────────────────────────────────────
            exit_res = simulate(
                db, settings, inst.id, signal.direction,
                signal.entry, signal.stop, signal.target,
                t_sig, atr14, trading_tf,
            )

            # ── size (audit only; label is size-independent) ──────────────────
            stop_pips = risk / inst.pip_size if inst.pip_size else float("inf")
            risk_amount = balance * settings.RISK_PCT_PER_TRADE
            units = 0
            if pip_value and stop_pips not in (0.0, float("inf")):
                units = sizer.calculate(risk_amount, stop_pips, pip_value)
            if units < settings.OANDA_MIN_UNITS:
                logger.debug(
                    "backtest %s @ %s: sized units=%d < OANDA_MIN_UNITS=%d — row kept "
                    "(rr label is size-independent)",
                    symbol, t_sig, units, settings.OANDA_MIN_UNITS,
                )

            outcome = label_outcome(exit_res.rr_actual)
            db.add(Trade(
                instrument_id=inst.id,
                direction=signal.direction,
                entry_price=signal.entry,
                exit_price=exit_res.exit_price,
                stop_price=signal.stop,
                tp_price=signal.target,
                units=units,
                risk_amount=risk_amount,
                expected_pip_loss=stop_pips,
                actual_pip_loss=None,
                rr_entry=reward / risk,
                rr_actual=exit_res.rr_actual,
                signal_source=_SIGNAL_SOURCE,
                stage=_STAGE,
                # WHICH CONFIGURATION produced this row. `signal_source` names the
                # ENGINE only, so two exit variants of one engine are otherwise
                # indistinguishable — and `load_dataset` would train on both at once,
                # learning from two contradictory definitions of a win.
                strategy_id=strategy_id,
                outcome=outcome,
                exit_reason=exit_res.exit_reason,
                ambiguous_resolution=exit_res.ambiguous_resolution,
                opened_at=t_sig,
                closed_at=exit_res.exit_time,
                signal_reasoning=_json_safe(features),
                confluence_score=signal.confidence_score,
                session=features.get("session"),
                stop_method=_STOP_METHOD,
            ))
            records.append({
                "signal_time": t_sig,
                "rr_actual": exit_res.rr_actual,
                "holding_hours": exit_res.holding_hours,
            })
            trades_simulated += 1
            pending += 1
            week_counts[wk] = week_counts.get(wk, 0) + 1
            next_allowed = exit_res.exit_time + cooldown_delta

            if pending >= _COMMIT_BATCH:
                db.commit()
                pending = 0

    if pending:
        db.commit()

    # ── fold slicing + metrics + promotion gate ───────────────────────────────
    # Sort globally by signal_time so every equity-curve metric (max_drawdown is
    # order-dependent) sees a chronological PORTFOLIO curve, not an instrument-
    # grouped one (records are appended instrument-outer during the walk).
    records.sort(key=lambda r: r["signal_time"])
    windows = fold_windows(
        window_start, window_end,
        _parse_float_list(settings.BACKTEST_FOLD_BOUNDS), embargo_delta,
    )
    folds = slice_folds(records, windows)
    oos_sets = [f["oos"] for f in folds]
    passed, gate_detail = M.evaluate_promotion_gate(oos_sets, n_trials, settings)

    combined_oos = [t for oos in oos_sets for t in oos]
    combined_metrics = _metrics_block(combined_oos, n_trials, settings.RISK_PCT_PER_TRADE)
    fold_blocks = [
        {
            "fold": i + 1,
            "window": {
                "is_start": windows[i].is_start.isoformat(),
                "is_end": windows[i].is_end.isoformat(),
                "oos_start": windows[i].oos_start.isoformat(),
                "oos_end": windows[i].oos_end.isoformat(),
            },
            "is_trade_count": len(folds[i]["is"]),
            "oos_trade_count": len(folds[i]["oos"]),
            "is_metrics": _metrics_block(folds[i]["is"], n_trials, settings.RISK_PCT_PER_TRADE),
            "oos_metrics": _metrics_block(folds[i]["oos"], n_trials, settings.RISK_PCT_PER_TRADE),
            "gate": _json_safe(gate_detail["folds"][i]),
        }
        for i in range(len(windows))
    ]

    fold_breakdown = _json_safe({
        "strategy": _STRATEGY,
        "instruments": [i.symbol for i in insts],
        "trading_timeframe": trading_tf,
        "n_trials": n_trials,
        "params": _params_snapshot(settings),
        "totals": {
            "decision_bars_evaluated": decision_bars,
            "signals_fired": signals_fired,
            "trades_simulated": trades_simulated,
            "rejections": rejections,
        },
        "folds": fold_blocks,
        "combined_oos": combined_metrics,
        "gate": gate_detail,
    })

    breakdown = M.outcome_breakdown(combined_oos)
    run = BacktestRun(
        instrument_id=insts[0].id,  # schema is single-FK; run is multi-instrument (see fold_breakdown.instruments)
        strategy=_STRATEGY,
        in_sample_start=window_start,
        in_sample_end=windows[-1].is_end,
        oos_start=windows[0].oos_start,
        oos_end=windows[-1].oos_end,
        trade_count=trades_simulated,
        win_rate=_win_rate(combined_oos),
        avg_rr=_finite_or(M.expectancy(combined_oos), 0.0),
        max_drawdown=_finite_or(M.max_drawdown(combined_oos, settings.RISK_PCT_PER_TRADE), 0.0),
        sharpe=_finite_or(M.sharpe(combined_oos), 0.0),
        expectancy=_finite_or(M.expectancy(combined_oos), 0.0),
        passed=passed,
        run_at=datetime.utcnow(),
        profit_factor=_finite_or(M.profit_factor(combined_oos), None),
        deflated_sharpe=_finite_or(M.deflated_sharpe(combined_oos, n_trials), None),
        probabilistic_sharpe=_finite_or(M.probabilistic_sharpe(combined_oos), None),
        trades_full_win=breakdown["full_win"],
        trades_partial=breakdown["partial"],
        trades_breakeven=breakdown["breakeven"],
        trades_loss=breakdown["loss"],
        avg_holding_hours=_finite_or(M.avg_holding_hours(combined_oos), None),
        oos_sample_size=len(combined_oos),
        fold_breakdown=fold_breakdown,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    runtime = time.monotonic() - t0
    logger.info(
        "backtest done: run_id=%s passed=%s bars=%d fired=%d trades=%d oos=%d runtime=%.1fs",
        run.id, passed, decision_bars, signals_fired, trades_simulated, len(combined_oos), runtime,
    )
    return BacktestResult(
        run_id=run.id,
        passed=passed,
        instruments=[i.symbol for i in insts],
        window_start=window_start,
        window_end=window_end,
        n_trials=n_trials,
        runtime_seconds=runtime,
        decision_bars_evaluated=decision_bars,
        signals_fired=signals_fired,
        trades_simulated=trades_simulated,
        rejections=rejections,
        fold_breakdown=fold_breakdown,
        combined_oos=combined_metrics,
    )


# ── setup helpers ────────────────────────────────────────────────────────────
def _first_granularity(settings: Settings) -> str:
    for g in settings.SIGNAL_GRANULARITIES.split(","):
        g = g.strip()
        if g:
            return g
    raise RuntimeError("SIGNAL_GRANULARITIES is empty — cannot determine the trading timeframe")


def _durable_n_trials(db: Session) -> int:
    """Number of independent backtest attempts against this dataset so far,
    INCLUDING the one about to run — for :func:`metrics.deflated_sharpe`'s
    selection-bias correction (Bailey & Lopez de Prado 2014; see the M7 plan:
    "computed against the number of backtest runs against this dataset ... not
    against 1").

    QA-identified defect (2026-07-05): the original ``db.query(BacktestRun).count()
    + 1`` goes DISHONEST the moment ``backtest_runs`` is truncated between runs
    (which the M7 workflow does routinely while iterating) — it always reads back
    down to 1, understating the true number of attempts and making
    ``deflated_sharpe`` LESS conservative than it should be (less deflation than
    the real selection bias warrants).

    Fix: derive the count from ``backtest_runs_id_seq`` (the Postgres sequence
    backing the table's auto-increment ``id``), which survives both ``DELETE``
    and a plain ``TRUNCATE`` (Postgres only resets a sequence on
    ``TRUNCATE ... RESTART IDENTITY`` — an operational caveat for whoever
    truncates this table, not something code can enforce). ``last_value`` is the
    id of the most recently allocated row (existing or since-deleted);
    ``last_value + 1`` (when the sequence has been used at least once) is the id
    the upcoming run's row will receive, i.e. exactly "attempts so far + this
    one." This is a read-only peek — it does not itself consume a sequence value.
    """
    row = db.execute(text("SELECT last_value, is_called FROM backtest_runs_id_seq")).one()
    return int(row.last_value) + 1 if row.is_called else int(row.last_value)


def _resolve_universe(db: Session, instruments: Optional[list[str]]) -> list[Instrument]:
    q = db.query(Instrument).filter_by(is_active=True)
    if instruments:
        q = q.filter(Instrument.symbol.in_(instruments))
    insts = q.order_by(Instrument.id).all()
    if not insts:
        raise RuntimeError("no active instruments in the universe (subset filtered everything out?)")
    return insts


def _assert_coverage(
    db: Session,
    insts: list[Instrument],
    trading_tf: str,
    window_start: datetime,
    window_end: datetime,
) -> None:
    """Fail loud unless every instrument's H4+M1 Bid/Ask actually spans the window.
    The window comes from config (never inferred); this only validates the data
    reaches both ends (day-granular slack for intraday first/last-bar alignment)."""
    for inst in insts:
        for gran, ptype in _REQUIRED_COVERAGE:
            mn, mx = (
                db.query(func.min(Candle.timestamp), func.max(Candle.timestamp))
                .filter(
                    Candle.instrument_id == inst.id,
                    Candle.granularity == gran,
                    Candle.price_type == ptype,
                )
                .one()
            )
            if mn is None or mx is None:
                raise RuntimeError(
                    f"coverage: {inst.symbol} {gran}/{ptype} has NO candles — "
                    f"run scripts/ingest_bid_ask.py before backtesting"
                )
            if mn > window_start + _WINDOW_COVERAGE_SLACK:
                raise RuntimeError(
                    f"coverage: {inst.symbol} {gran}/{ptype} starts {mn} — after window "
                    f"start {window_start} (+slack); Bid/Ask does not span the window start"
                )
            if mx < window_end - _WINDOW_COVERAGE_SLACK:
                raise RuntimeError(
                    f"coverage: {inst.symbol} {gran}/{ptype} ends {mx} — before window "
                    f"end {window_end} (−slack); Bid/Ask does not span the window end"
                )


def _cache_pip_values(db: Session, insts: list[Instrument], settings: Settings) -> dict[int, Optional[float]]:
    """Fetch pip value ONCE per instrument via BrokerRouter (never per-trade). On
    failure (offline / no home-conversion) the entry is None → audit sizing falls
    back to 0 units; the label is unaffected (rr is size-independent)."""
    router = get_broker_router()
    out: dict[int, Optional[float]] = {}
    for inst in insts:
        try:
            client = router.for_instrument(inst.symbol, db)
            out[inst.id] = client.get_pip_value(inst.symbol, inst.pip_size)
        except Exception as exc:  # noqa: BLE001 — sizing is non-fatal for label generation
            logger.warning("pip value fetch failed for %s: %s — sizing audit only", inst.symbol, exc)
            out[inst.id] = None
    return out


def _prefetch_mid_bars(
    db: Session, instrument_id: int, granularity: str,
    window_start: datetime, window_end: datetime, period_delta: timedelta,
) -> list[tuple[datetime, float]]:
    """Decision bars: H4 Mid (open ts, close) whose CLOSE can fall in the window,
    ascending. Fetched from ``window_start - period`` so a bar opening just before
    the window but closing inside it is included."""
    rows = (
        db.query(Candle.timestamp, Candle.close)
        .filter(
            Candle.instrument_id == instrument_id,
            Candle.granularity == granularity,
            Candle.price_type == "M",
            Candle.timestamp >= window_start - period_delta,
            Candle.timestamp <= window_end,
        )
        .order_by(Candle.timestamp.asc())
        .all()
    )
    return [(r.timestamp, r.close) for r in rows]


def _prefetch_close(
    db: Session, instrument_id: int, granularity: str, price_type: str,
    window_start: datetime, window_end: datetime, period_delta: timedelta,
) -> dict[datetime, float]:
    rows = (
        db.query(Candle.timestamp, Candle.close)
        .filter(
            Candle.instrument_id == instrument_id,
            Candle.granularity == granularity,
            Candle.price_type == price_type,
            Candle.timestamp >= window_start - period_delta,
            Candle.timestamp <= window_end,
        )
        .all()
    )
    return {r.timestamp: r.close for r in rows}


# ── metrics / persistence helpers ────────────────────────────────────────────
def _metrics_block(trades: Sequence, n_trials: int, risk_pct: float) -> dict:
    return {
        "trade_count": len(trades),
        "profit_factor": M.profit_factor(trades),
        "expectancy": M.expectancy(trades),
        "max_drawdown": M.max_drawdown(trades, risk_pct),
        "sharpe": M.sharpe(trades),
        "deflated_sharpe": M.deflated_sharpe(trades, n_trials),
        "probabilistic_sharpe": M.probabilistic_sharpe(trades),
        "win_rate": M.win_rate(trades),  # reporting-only — NOT part of the promotion gate
        "avg_holding_hours": M.avg_holding_hours(trades),
        "outcome_breakdown": M.outcome_breakdown(trades),
    }


def _win_rate(trades: Sequence) -> float:
    rrs = [t["rr_actual"] for t in trades if t.get("rr_actual") is not None]
    if not rrs:
        return 0.0
    wins = sum(1 for r in rrs if r > 0.05)
    return wins / len(rrs)


def _params_snapshot(settings: Settings) -> dict:
    """The subset of config that defines this run's behaviour (for reproducibility)."""
    keys = [
        "SIGNAL_MIN_CONFLUENCE_SCORE", "SIGNAL_SESSION_FILTER", "SIGNAL_STOP_ATR_MULTIPLIER",
        "SIGNAL_COOLDOWN_BARS_AFTER_CLOSE", "SIGNAL_NO_TRADE_HOURS_BEFORE_FRIDAY_CLOSE",
        "MIN_RR_RATIO", "ATR_MULTIPLIER_MAX", "SIGNAL_MAX_HOLD_BARS",
        "BACKTEST_TRAILING_LOCK_PCT", "BACKTEST_TRAILING_DISTANCE_ATR_MULT",
        "BACKTEST_EMBARGO_BARS", "BACKTEST_FOLD_BOUNDS", "BACKTEST_RESPECT_WEEKLY_CAP",
        "MAX_TRADES_PER_WEEK", "STARTING_BALANCE", "RISK_PCT_PER_TRADE",
        "BACKTEST_PROMOTION_PROFIT_FACTOR_MIN", "BACKTEST_PROMOTION_EXPECTANCY_MIN",
        "BACKTEST_PROMOTION_MAX_DD_MAX", "BACKTEST_PROMOTION_MIN_OOS_TRADES",
        "EXECUTION_WINDOW_START", "EXECUTION_WINDOW_END", "feature_schema_version",
    ]
    snap: dict = {}
    for k in keys:
        if k == "feature_schema_version":
            snap[k] = FEATURE_SCHEMA_VERSION
            continue
        v = getattr(settings, k)
        snap[k] = v.isoformat() if isinstance(v, datetime) else v
    return snap


def _new_rejections() -> dict[str, int]:
    return {
        "near_window_end": 0,
        "cooldown_or_open": 0,
        "weekly_cap": 0,
        "no_quote": 0,
        "insufficient_rr": 0,
        "stop_too_wide": 0,
        "no_atr": 0,
    }


# ── misc ─────────────────────────────────────────────────────────────────────
def _finite_or(value, default):
    """Return ``value`` unless it is None/NaN/±inf, in which case ``default`` (keeps
    non-null DB columns valid and JSON serialisable)."""
    if value is None:
        return default
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return default
    return value


def _json_safe(obj):
    """Recursively replace NaN/±inf floats with None so the JSON column is valid
    Postgres JSON (which rejects the NaN/Infinity tokens Python's json emits).

    Delegates to ``feature_builder.json_safe`` — ONE implementation, so the
    backtest corpus and the M8 shadow corpus can never disagree on how a missing
    feature is persisted.
    """
    return json_safe(obj)


def _parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _isnan(x) -> bool:
    return isinstance(x, float) and x != x


def _naive(dt: datetime) -> datetime:
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt
