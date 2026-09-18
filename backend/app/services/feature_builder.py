"""
feature_builder.py — the single authoritative, point-in-time (PIT) feature
chokepoint for the whole ML program (Pre-M7 Step 4).

Both the M7 backtester and (later) the live inference engine call ``build_features``
so training and serving compute the LOCKED feature contract identically — no
training/serving skew. This module is the leakage firewall.

Point-in-time rule (applied to every external read)
----------------------------------------------------
Given a ``signal_time`` T, nothing computed here may depend on data that was not
public at T:

* macro_data (LEAK-3, bitemporal): only rows with ``release_time <= T`` are
  visible. Among visible rows we take the latest ``ref_period``, then the latest
  ``release_time`` within it (the newest vintage known at T). We NEVER gate on
  ``ref_period`` — publish lag is real, and a revision published after T must not
  leak into a pre-revision signal.
* news_calendar_events: matched by scheduled ``timestamp`` windows around T,
  currency-scoped to the pair's two legs. ``all_day`` events (e.g. BoJ) match if
  their calendar day intersects the window.
* Indicators/candles: only rows with ``timestamp <= T`` (last CLOSED bar).

TIMESTAMP CONVENTION — READ BEFORE CALLING (M7-runner-facing)
---------------------------------------------------------------
``candles.timestamp`` stores the bar's OPEN time, not its close (verified against
the ingest path: OANDA's candle ``"time"`` field is the candlestick START time —
``app/brokers/oanda.py`` parses it directly at ``candle["time"]``, and only
``complete=True`` candles are ever ingested, so every stored row is a finalized,
fully-closed bar keyed by its OPEN instant).

Every indicator/candle read in this module filters ``timestamp <= T``. Because
of that ``<=``, **``signal_time`` (T) must be STRICTLY GREATER than the decision
bar's close** — which, for contiguous bars, is numerically identical to the
following bar's OPEN timestamp. If a caller passes T exactly equal to that
boundary (T == decision bar's close == next bar's open), the next bar — whose
row already holds its full future OHLC in a backtest DB, i.e. data that would not
exist yet at real decision time T — satisfies ``timestamp <= T`` and gets treated
as the "latest" bar. That is a post-T leak.

**Rule for the M7 runner / live engine:** pass a ``signal_time`` that falls
strictly INSIDE the bar following the decision bar — never the exact boundary
instant. In practice: ``signal_time = decision_bar_close + timedelta(seconds=1)``
(or any instant up to, but not including, that following bar's own close) is
safe. Do not pass ``decision_bar_close`` (== next bar's open) unmodified.

Leakage enforcement
-------------------
* LEAK-1 (causal swings): the ``indicators`` table stored swing highs/lows using a
  CENTERED window — a pivot at bar i was confirmed using k = SWING_LOOKBACK_PERIODS
  bars AFTER i, so the most recent swings near T were confirmed with post-T bars.
  We use the **confirmation-lag** approach: only swings from indicator rows at or
  before ``T − k bars`` are consulted (the newest k indicator rows ≤ T are dropped
  because their swing values could only be confirmed by bars after T). Identical in
  backtest and live; uses the existing centered-window data without recompute.
* LEAK-2 (staleness floor): a monthly/daily series whose newest visible release is
  older than a configurable threshold returns NaN instead of a carried-forward
  constant (which would let a tree memorise the frozen-calendar regime). This
  auto-blanks the frozen CH/NZ policy legs and any future dead series.
* LEAK-3 (PIT): see above.
* NaN-safe by construction: the output dict is built from canonical key lists with
  ``float("nan")`` defaults, so a missing series/row can never KeyError. News
  booleans are a genuine value WITHIN calendar coverage (absent scheduled event =
  False, not unknown); BEFORE the earliest calendar event (the coverage floor) they
  stay NaN — a pre-coverage "False" would be a dishonest "no event" claim (v2).

NOT implemented here (research-only until GAP-13 live M1 feed): m1_spread_anomaly,
m1_late_momentum. They stay out of the serving contract to guarantee backtest==live.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import Settings
from app.domain.macro_series import (
    MACRO_SERIES_BY_NAME,
    series_name_for_currency,
    series_name_for_role,
)
from app.models.candle import Candle
from app.models.indicator import Indicator
from app.models.instrument import Instrument
from app.models.macro_data import MacroData
from app.models.news_calendar_event import NewsCalendarEvent
from app.domain.conditions import CONDITIONS, CONDITIONS_BY_KEY, LEGACY_CONFLUENCE_KEYS
from app.services.session_classifier import classify_session

FEATURE_SCHEMA_VERSION = 3

# ── LOCKED FEATURE CONTRACT ──────────────────────────────────────────────────
# v3 (2026-09-19): SAME 19 MODEL KEYS AS v2 — the bump is about MEANING, not names.
#   `c3_structure` moved from centred swing pivots to a trailing Donchian channel,
#   and the live indicator writer was fixed so `swing_*` is populated at all. Both
#   change what `c3_structure` and `swing_dist_atr` VALUE, while every key name stays
#   identical — precisely the case provenance.py warns that nothing else would catch.
#   Measured before the fix: c3_structure true on 20.0% of backtest rows vs 0.0% of
#   live rows; swing_dist_atr median 2.724 backtest vs 4.431 live (+63%). Models v1-v3
#   were trained on the leaked/skewed values, so `inference._validate_schema_version`
#   refusing to load them against a v3 contract is the intended outcome, not collateral.
# v2 (Research Cycle 2): MODEL tier = 19 keys. Dropped the four news_high_impact_*
#   booleans DOWN to the SHAP-GATE tier (they gate trades, they do not train v1);
#   promoted us_real_10y UP into the model tier; added wti + wti_change_20d (global
#   commodity — daily, non-revised; mirrors the VIX pattern).
# v1 (2026-07-01, historical): MODEL tier = 20 keys — the four news booleans WERE
#   model inputs, us_real_10y was gated, WTI absent. The 2,809 existing backtest
#   trades carry v1 feature rows (S1's mixed-version guard will correctly refuse to
#   mix them with v2 rows on the next re-run).
# MODEL CORE — the only inputs model-v1 trains on.
FEATURE_KEYS_MODEL: list[str] = [
    "confluence_score",
    "rsi14",
    "atr_pct",
    "dist_to_sma50_atr",
    "swing_dist_atr",
    "yield_differential_10y",
    "yield_differential_change_1m",
    "us_2s10s",
    "policy_rate_differential",
    "vix",
    "vix_change_5d",
    "us_real_10y",
    "wti",
    "wti_change_20d",
    "us_core_cpi_yoy",
    "us_cpi_yoy_change",
    "us_unemployment",
    "instrument_category",
    "session",
]

# SHAP-GATE tier — computed + stored on every row, but NOT a model-v1 input.
FEATURE_KEYS_GATED: list[str] = [
    "us_10y",
    "eu_cpi_yoy",
    "yield_differential_change_3m",
    "us_cpi_yoy",
    "us_unemployment_change",
    "us_retail_sales_mom",
    "news_high_impact_next_4h",
    "news_high_impact_next_8h",
    "news_high_impact_last_4h",
    "news_high_impact_last_8h",
]

# RISK PAYLOAD — stored for RiskEngine / audit, never a model input.
#
# The condition booleans are DERIVED from the condition registry rather than restated
# here. They used to be a hand-copied list, and the loop below iterated THAT list rather
# than the breakdown it was handed — so a sixth condition would have influenced the trade
# while being silently absent from the feature row, with nothing to catch the divergence.
_PRICE_PAYLOAD_KEYS: list[str] = [
    "atr14",
    "h4_close",
    "d1_close",
    "d1_sma50",
]
PAYLOAD_KEYS: list[str] = _PRICE_PAYLOAD_KEYS + [c.payload_key for c in CONDITIONS]

_ALL_KEYS: set[str] = set(FEATURE_KEYS_MODEL + FEATURE_KEYS_GATED + PAYLOAD_KEYS)

# Contract-indicator features, resolved by ROLE via the registry (consistent with
# the rate features' currency/role resolution — never a hardcoded series-name
# literal). Validated against the registry at import so a role rename/removal
# fails loud instead of silently NaN-ing every row.
_INDICATOR_ROLES = ("us_cpi", "us_core_cpi", "eu_cpi", "us_unemployment", "us_retail_sales")
_INDICATOR_SERIES = {role: series_name_for_role(role) for role in _INDICATOR_ROLES}
for _role, _s in _INDICATOR_SERIES.items():
    if _s is None or _s not in MACRO_SERIES_BY_NAME:
        raise RuntimeError(f"feature_builder: no macro series registered with role='{_role}'")

_NAN = float("nan")


# ── public API ───────────────────────────────────────────────────────────────
def build_features(
    instrument: Instrument,
    signal_time: datetime,
    granularity: str,
    confluence: dict[str, bool],
    db: Session,
    settings: Settings,
) -> dict:
    """Compute the LOCKED feature contract point-in-time for one trade signal.

    Args:
        instrument: the traded instrument (symbol split on ``_`` gives base/quote).
        signal_time: T — the signal timestamp; nothing after T may influence output.
            MUST be strictly after the decision bar's close (== the following bar's
            OPEN timestamp, since ``candles.timestamp`` is bar-open time and every
            internal read filters ``timestamp <= T``). Passing exactly the boundary
            instant risks pulling in the following (not-yet-decided) bar's future
            OHLC. See the module docstring "TIMESTAMP CONVENTION" section — pass
            e.g. ``decision_bar_close + timedelta(seconds=1)``.
        granularity: the trading timeframe of the signal (e.g. "H4"). ATR/RSI/swings
            and h4_close are read on THIS timeframe; the trend leg uses
            ``settings.SIGNAL_TREND_TIMEFRAME``.
        confluence: the signal engine's C1..C5 breakdown (keys trend/rsi/structure/
            session/spread → booleans). ``confluence_score`` is derived from it.
        db: SQLAlchemy session.
        settings: config (all thresholds; zero hardcoding).

    Returns:
        A dict containing every key in FEATURE_KEYS_MODEL + FEATURE_KEYS_GATED +
        PAYLOAD_KEYS plus ``feature_schema_version``. Unavailable numeric features
        are ``float("nan")`` (XGBoost routes NaN); missing rows never KeyError.
    """
    t = _naive(signal_time)
    base, quote = _split_pair(instrument.symbol)

    features: dict = {k: _NAN for k in _ALL_KEYS}
    features["feature_schema_version"] = FEATURE_SCHEMA_VERSION

    # ── confluence (input) ───────────────────────────────────────────────────
    # A key the registry does not know is a HARD ERROR, not a shrug. The engine and this
    # builder derive from one registry, so an unrecognised key means they have diverged —
    # and a silently dropped condition is one that steers trades while being invisible to
    # every model trained on the rows.
    unknown = sorted(set(confluence) - set(CONDITIONS_BY_KEY))
    if unknown:
        raise RuntimeError(
            f"score_breakdown carries unregistered conditions {unknown}. Register them in "
            f"app/domain/conditions so the engine and the feature contract stay in step."
        )
    for cond in CONDITIONS:
        features[cond.payload_key] = bool(confluence.get(cond.key, False))
    # Pinned to the LEGACY five, forever: `confluence_score` is a model-tier feature, so
    # changing what it counts changes the meaning of a name every existing corpus uses.
    # New conditions are recorded in the payload tier and promoted deliberately.
    features["confluence_score"] = sum(
        1 for k in LEGACY_CONFLUENCE_KEYS if bool(confluence.get(k, False))
    )

    # ── price / indicator core (payload + derived) ───────────────────────────
    _fill_price_features(features, instrument, granularity, t, db, settings)

    # ── causal swings (LEAK-1) ───────────────────────────────────────────────
    _fill_swing_feature(features, instrument, granularity, t, db, settings)

    # ── rates: yields / policy / curve / vix ─────────────────────────────────
    _fill_rate_features(features, base, quote, t, db, settings)

    # ── macro indicators: CPI / unemployment / retail ────────────────────────
    _fill_macro_indicator_features(features, t, db, settings)

    # ── news windows ─────────────────────────────────────────────────────────
    _fill_news_features(features, base, quote, t, db)

    # ── categoricals ─────────────────────────────────────────────────────────
    features["instrument_category"] = _instrument_category(base, quote)
    features["session"] = classify_session(t)

    # NOTE: m1_spread_anomaly / m1_late_momentum intentionally NOT computed —
    # research-only until GAP-13 wires a live M1 feed (backtest==live guarantee).

    return features


# ── price / indicator core ───────────────────────────────────────────────────
def _fill_price_features(
    features: dict,
    instrument: Instrument,
    granularity: str,
    t: datetime,
    db: Session,
    settings: Settings,
) -> None:
    ind = (
        db.query(Indicator)
        .filter(
            Indicator.instrument_id == instrument.id,
            Indicator.granularity == granularity,
            Indicator.timestamp <= t,
        )
        .order_by(Indicator.timestamp.desc())
        .first()
    )
    atr14 = ind.atr14 if ind is not None else None
    rsi14 = ind.rsi14 if ind is not None else None

    h4_close = _latest_close(db, instrument.id, granularity, t)
    trend_tf = settings.SIGNAL_TREND_TIMEFRAME
    d1_close = _latest_close(db, instrument.id, trend_tf, t)
    d1_sma50 = _sma_close(db, instrument.id, trend_tf, settings.SIGNAL_TREND_SMA_PERIOD, t)

    features["atr14"] = atr14 if atr14 is not None else _NAN
    features["h4_close"] = h4_close if h4_close is not None else _NAN
    features["d1_close"] = d1_close if d1_close is not None else _NAN
    features["d1_sma50"] = d1_sma50 if d1_sma50 is not None else _NAN
    features["rsi14"] = rsi14 if rsi14 is not None else _NAN

    if atr14 and h4_close:  # both truthy → non-zero, safe to divide
        features["atr_pct"] = atr14 / h4_close
        if d1_sma50 is not None:
            features["dist_to_sma50_atr"] = (h4_close - d1_sma50) / atr14


def _fill_swing_feature(
    features: dict,
    instrument: Instrument,
    granularity: str,
    t: datetime,
    db: Session,
    settings: Settings,
) -> None:
    """swing_dist_atr = distance from h4_close to the nearest CAUSALLY-CONFIRMED
    swing level, normalised by ATR14 (LEAK-1)."""
    h4_close = features.get("h4_close")
    atr14 = features.get("atr14")
    if h4_close is None or atr14 is None or _isnan(h4_close) or _isnan(atr14) or atr14 == 0:
        return
    levels = causal_swing_levels(
        db, instrument.id, granularity, t, settings.SWING_LOOKBACK_PERIODS
    )
    candidates = [lvl for lvl in (levels["swing_high"], levels["swing_low"]) if lvl is not None]
    if not candidates:
        return
    nearest = min(abs(h4_close - lvl) for lvl in candidates)
    features["swing_dist_atr"] = nearest / atr14


def causal_swing_levels(
    db: Session,
    instrument_id: int,
    granularity: str,
    signal_time: datetime,
    swing_lookback: int,
) -> dict:
    """Return the nearest causally-confirmed swing high/low visible at ``signal_time``.

    The stored swings use a centered ±k window (k = ``swing_lookback``): a pivot at
    bar i is confirmed only once bar i+k closes. So among indicator rows with
    ``timestamp <= signal_time`` ordered newest-first, the newest ``k`` rows are
    dropped (their swing could only be confirmed by post-T bars). This function is
    the assertable LEAK-1 boundary: ``cutoff_timestamp`` is the newest indicator
    timestamp that MAY be consulted; nothing newer is read.

    Bounded by construction
    -----------------------
    This used to materialise EVERY indicator row at or before T (``.all()``) purely
    to read the first non-NULL swing on each side — ~8.7k ORM objects per instrument,
    per signal, per candle-close cycle, of which two were used. That is invisible on
    a dev workstation and a real per-cycle cost on a small always-on host with slow
    storage (Raspberry Pi microSD / USB SSD).

    The rewrite below is EXACTLY equivalent, not an approximation — no "look back
    N rows and hope" heuristic, which could silently return a nearer swing than the
    old code and change a live feature value. It reproduces the same three answers
    with three indexed lookups:

      1. ``cutoff_ts`` — the (k+1)-th newest row's timestamp, via ``OFFSET k LIMIT 1``.
         Identical to ``rows[k].timestamp``.
      2/3. the newest non-NULL ``swing_high`` / ``swing_low`` at or before that
         cutoff, via ``IS NOT NULL ... ORDER BY timestamp DESC LIMIT 1``.

    ``timestamp <= cutoff_ts`` selects precisely the old ``confirmed`` slice because
    ``(instrument_id, granularity, timestamp)`` is UNIQUE — no ties can straddle the
    boundary. That same unique constraint's composite btree serves every query here,
    so each is an index scan with an early stop rather than a table sweep.
    """
    cutoff_t = _naive(signal_time)
    base = db.query(Indicator).filter(
        Indicator.instrument_id == instrument_id,
        Indicator.granularity == granularity,
    )

    cutoff_ts = (
        base.filter(Indicator.timestamp <= cutoff_t)
        .with_entities(Indicator.timestamp)
        .order_by(Indicator.timestamp.desc())
        .offset(swing_lookback)
        .limit(1)
        .scalar()
    )
    if cutoff_ts is None:
        # Fewer than k+1 rows exist at or before T — nothing is causally confirmed.
        return {
            "swing_high": None,
            "swing_low": None,
            "cutoff_timestamp": None,
            "n_confirmed_rows": 0,
        }

    confirmed = base.filter(Indicator.timestamp <= cutoff_ts)

    def _newest(column):
        return (
            confirmed.filter(column.isnot(None))
            .with_entities(column)
            .order_by(Indicator.timestamp.desc())
            .limit(1)
            .scalar()
        )

    return {
        "swing_high": _newest(Indicator.swing_high),
        "swing_low": _newest(Indicator.swing_low),
        "cutoff_timestamp": cutoff_ts,
        # Diagnostic only (asserted by the LEAK-1 test). An index-only COUNT, not a
        # row materialisation.
        "n_confirmed_rows": confirmed.with_entities(func.count(Indicator.id)).scalar() or 0,
    }


# ── rates: yields / policy / curve / vix ─────────────────────────────────────
def _fill_rate_features(
    features: dict,
    base: str,
    quote: str,
    t: datetime,
    db: Session,
    settings: Settings,
) -> None:
    # 10Y yield differential (base − quote), OECD monthly, same methodology.
    base_10y = series_name_for_currency(base, "yield")
    quote_10y = series_name_for_currency(quote, "yield")
    features["yield_differential_10y"] = _differential(db, base_10y, quote_10y, t, settings)

    for months in _parse_int_list(settings.YIELD_DIFFERENTIAL_CHANGE_MONTHS):
        key = f"yield_differential_change_{months}m"
        if key not in _ALL_KEYS:
            continue  # ignore configured months outside the locked contract
        now_diff = _differential(db, base_10y, quote_10y, t, settings)
        past_diff = _differential(db, base_10y, quote_10y, _minus_months(t, months), settings)
        features[key] = _sub(now_diff, past_diff)

    # Policy-rate differential (front-end carry). Staleness auto-NaNs frozen CH/NZ.
    base_pol = series_name_for_currency(base, "policy")
    quote_pol = series_name_for_currency(quote, "policy")
    features["policy_rate_differential"] = _differential(db, base_pol, quote_pol, t, settings)

    # US curve + global level (pair-independent, daily).
    us_10y_series = series_name_for_role("us_10y_daily")
    us_2y_series = series_name_for_role("us_2y_daily")
    us_10y = _pit_level(db, us_10y_series, t, settings)
    us_2y = _pit_level(db, us_2y_series, t, settings)
    features["us_10y"] = us_10y
    features["us_2s10s"] = _sub(us_10y, us_2y)

    # VIX level + N-day change.
    vix_series = series_name_for_role("vix")
    vix = _pit_level(db, vix_series, t, settings)
    features["vix"] = vix
    vix_past = _pit_level(db, vix_series, t - timedelta(days=settings.VIX_CHANGE_DAYS), settings)
    features["vix_change_5d"] = _sub(vix, vix_past)

    # WTI crude level + N-day change (global commodity, daily cadence). Resolved by
    # ROLE — never a hardcoded series name; PIT + staleness identical to VIX.
    wti_series = series_name_for_role("wti")
    wti = _pit_level(db, wti_series, t, settings)
    features["wti"] = wti
    wti_past = _pit_level(db, wti_series, t - timedelta(days=settings.WTI_CHANGE_DAYS), settings)
    features["wti_change_20d"] = _sub(wti, wti_past)


# ── macro indicators: CPI / unemployment / retail ────────────────────────────
def _fill_macro_indicator_features(
    features: dict,
    t: datetime,
    db: Session,
    settings: Settings,
) -> None:
    # YoY on CPI index levels (US headline/core, EU). PIT current index / index of
    # the ref_period 12 months earlier, both as-known at T.
    features["us_cpi_yoy"] = _yoy(db, _INDICATOR_SERIES["us_cpi"], t, settings)
    features["us_core_cpi_yoy"] = _yoy(db, _INDICATOR_SERIES["us_core_cpi"], t, settings)
    features["eu_cpi_yoy"] = _yoy(db, _INDICATOR_SERIES["eu_cpi"], t, settings)

    # us_cpi_yoy_change = current YoY − YoY computed one release (ref-month) earlier.
    us_cpi = _INDICATOR_SERIES["us_cpi"]
    cur = _pit_observation(db, us_cpi, t, settings)
    if cur is not None:
        prev_ref = _shift_ref(cur["ref_period"], -1)
        yoy_now = features["us_cpi_yoy"]
        yoy_prev = _yoy_for_ref(db, us_cpi, prev_ref, t)
        features["us_cpi_yoy_change"] = _sub(yoy_now, yoy_prev)

    # Unemployment: passthrough rate + release-over-release change.
    unemp = _INDICATOR_SERIES["us_unemployment"]
    features["us_unemployment"] = _pit_level(db, unemp, t, settings)
    obs = _pit_observation(db, unemp, t, settings)
    if obs is not None:
        prev = _value_for_ref(db, unemp, _shift_ref(obs["ref_period"], -1), t)
        features["us_unemployment_change"] = _sub(obs["value"], prev)

    # Retail sales MoM (index/level ratio).
    retail = _INDICATOR_SERIES["us_retail_sales"]
    r_obs = _pit_observation(db, retail, t, settings)
    if r_obs is not None:
        r_prev = _value_for_ref(db, retail, _shift_ref(r_obs["ref_period"], -1), t)
        if r_prev is not None and not _isnan(r_prev) and r_prev != 0:
            # staleness still applies via the current-obs level check
            if not _isnan(_pit_level(db, retail, t, settings)):
                features["us_retail_sales_mom"] = r_obs["value"] / r_prev - 1.0

    # Real 10Y = nominal US 10Y (percent, e.g. 4.24) minus US CPI YoY (percent).
    # us_cpi_yoy is stored as a FRACTION (cur/prior - 1.0, e.g. 0.0238) — do not
    # change its stored scale (locked contract); convert to percent HERE only,
    # for this derived feature alone.
    us_cpi_yoy_pct = (
        features["us_cpi_yoy"] * 100.0 if not _isnan(features["us_cpi_yoy"]) else _NAN
    )
    features["us_real_10y"] = _sub(features["us_10y"], us_cpi_yoy_pct)


# ── news windows ─────────────────────────────────────────────────────────────
# Coverage floor cache: the earliest news_calendar_events.timestamp is the instant
# before which we hold ZERO calendar data. It is queried once and memoised module-
# wide (the corpus's coverage start is fixed within a process). A ``_LOADED`` flag
# distinguishes "not yet queried" from a legitimately empty table (None).
_NEWS_COVERAGE_START: datetime | None = None
_NEWS_COVERAGE_LOADED: bool = False


def _news_coverage_start(db: Session) -> datetime | None:
    """Earliest news_calendar_events timestamp (the coverage floor), memoised.

    Returns None if the calendar table is empty. Signals whose ``signal_time``
    predates this floor get NaN news booleans (unknown), never False.
    """
    global _NEWS_COVERAGE_START, _NEWS_COVERAGE_LOADED
    if not _NEWS_COVERAGE_LOADED:
        _NEWS_COVERAGE_START = db.query(func.min(NewsCalendarEvent.timestamp)).scalar()
        _NEWS_COVERAGE_LOADED = True
    return _NEWS_COVERAGE_START


def _fill_news_features(
    features: dict,
    base: str,
    quote: str,
    t: datetime,
    db: Session,
) -> None:
    # Coverage gate: before the earliest calendar event we have NO news data, so the
    # four booleans stay at their NaN default (unknown) rather than a dishonest False
    # ("no scheduled event"). Within coverage, absent event = a genuine False.
    coverage_start = _news_coverage_start(db)
    if coverage_start is None or t < coverage_start:
        return

    currencies = {base, quote}
    events = (
        db.query(NewsCalendarEvent)
        .filter(
            NewsCalendarEvent.currency.in_(currencies),
            NewsCalendarEvent.impact == "high",
            NewsCalendarEvent.timestamp >= t - timedelta(hours=8),
            NewsCalendarEvent.timestamp <= t + timedelta(hours=8),
        )
        .all()
    )
    # Also pull all_day events whose day may straddle the window even if the stored
    # midnight timestamp sits just outside the ±8h band.
    day_events = (
        db.query(NewsCalendarEvent)
        .filter(
            NewsCalendarEvent.currency.in_(currencies),
            NewsCalendarEvent.impact == "high",
            NewsCalendarEvent.all_day.is_(True),
            NewsCalendarEvent.timestamp >= t - timedelta(days=1),
            NewsCalendarEvent.timestamp <= t + timedelta(days=1),
        )
        .all()
    )
    pool = {e.id: e for e in events + day_events}.values()

    features["news_high_impact_next_4h"] = _any_event(pool, t, t + timedelta(hours=4), incl_end=True)
    features["news_high_impact_next_8h"] = _any_event(pool, t, t + timedelta(hours=8), incl_end=True)
    features["news_high_impact_last_4h"] = _any_event(pool, t - timedelta(hours=4), t, incl_end=False)
    features["news_high_impact_last_8h"] = _any_event(pool, t - timedelta(hours=8), t, incl_end=False)


def _any_event(events, window_start: datetime, window_end: datetime, incl_end: bool) -> bool:
    """True if any event falls in [window_start, window_end]. Point events use the
    stored timestamp; all_day events match if their calendar day intersects the
    window. Absent event = False (a genuine value, not unknown)."""
    for e in events:
        if e.all_day:
            day_start = datetime(e.timestamp.year, e.timestamp.month, e.timestamp.day)
            day_end = day_start + timedelta(days=1)
            if day_start < window_end and day_end > window_start:
                return True
        else:
            ts = e.timestamp
            if incl_end:
                if window_start <= ts <= window_end:
                    return True
            else:
                if window_start <= ts < window_end:
                    return True
    return False


# ── macro PIT primitives ─────────────────────────────────────────────────────
def _visible_rows(db: Session, series: str | None, t: datetime) -> list[MacroData]:
    if series is None:
        return []
    return (
        db.query(MacroData)
        .filter(MacroData.series == series, MacroData.release_time <= t)
        .all()
    )


def _pit_observation(db: Session, series: str | None, t: datetime, settings: Settings) -> dict | None:
    """Latest ref_period visible at T, then newest vintage within it. Applies the
    LEAK-2 staleness floor (returns None if the series' newest release is too old).
    """
    rows = _visible_rows(db, series, t)
    if not rows:
        return None
    newest_release = max(r.release_time for r in rows)
    if _is_stale(series, newest_release, t, settings):
        return None
    latest_ref = max(r.ref_period for r in rows)
    best = max((r for r in rows if r.ref_period == latest_ref), key=lambda r: r.release_time)
    return {"ref_period": latest_ref, "value": best.value, "release_time": best.release_time}


def _pit_level(db: Session, series: str | None, t: datetime, settings: Settings) -> float:
    obs = _pit_observation(db, series, t, settings)
    return obs["value"] if obs is not None else _NAN


def _value_for_ref(db: Session, series: str | None, ref_period: str, t: datetime) -> float:
    """Newest vintage of a SPECIFIC ref_period known at T. No staleness floor —
    this is an intentionally historical leg (e.g. the 12-months-earlier CPI)."""
    rows = [r for r in _visible_rows(db, series, t) if r.ref_period == ref_period]
    if not rows:
        return _NAN
    return max(rows, key=lambda r: r.release_time).value


def _is_stale(series: str | None, newest_release: datetime, t: datetime, settings: Settings) -> bool:
    meta = MACRO_SERIES_BY_NAME.get(series) if series else None
    if meta is None:
        return True
    threshold = (
        settings.FEATURE_STALENESS_DAILY_DAYS
        if meta.cadence == "daily"
        else settings.FEATURE_STALENESS_MONTHLY_DAYS
    )
    return (t - newest_release).days > threshold


def _differential(db: Session, base_series: str | None, quote_series: str | None,
                  t: datetime, settings: Settings) -> float:
    return _sub(_pit_level(db, base_series, t, settings), _pit_level(db, quote_series, t, settings))


def _yoy(db: Session, series: str, t: datetime, settings: Settings) -> float:
    """Year-over-year growth from a CPI INDEX-LEVEL series, PIT at T."""
    obs = _pit_observation(db, series, t, settings)
    if obs is None:
        return _NAN
    return _yoy_for_ref(db, series, obs["ref_period"], t)


def _yoy_for_ref(db: Session, series: str, ref_period: str, t: datetime) -> float:
    """YoY for a specific ref_period: value(ref) / value(ref − 12 months), both
    as-known at T. Note: staleness is not re-checked here — the caller gates on the
    current observation via _pit_observation before requesting a YoY."""
    cur = _value_for_ref(db, series, ref_period, t)
    prior = _value_for_ref(db, series, _shift_ref(ref_period, -12), t)
    if _isnan(cur) or _isnan(prior) or prior == 0:
        return _NAN
    return cur / prior - 1.0


# ── small helpers ────────────────────────────────────────────────────────────
def _instrument_category(base: str, quote: str) -> str:
    """≤4 buckets for the 10-pair universe. JPY takes priority, then USD, then EUR."""
    legs = {base, quote}
    if "JPY" in legs:
        return "jpy_cross"
    if "USD" in legs:
        return "major_usd"
    if "EUR" in legs:
        return "eur_cross"
    return "other"


def _split_pair(symbol: str) -> tuple[str, str]:
    base, _, quote = symbol.partition("_")
    return base, quote


def _latest_close(db: Session, instrument_id: int, granularity: str, t: datetime) -> float | None:
    c = (
        db.query(Candle)
        .filter(
            Candle.instrument_id == instrument_id,
            Candle.granularity == granularity,
            Candle.price_type == "M",
            Candle.timestamp <= t,
        )
        .order_by(Candle.timestamp.desc())
        .first()
    )
    return c.close if c is not None else None


def _sma_close(db: Session, instrument_id: int, granularity: str, period: int, t: datetime) -> float | None:
    closes = [
        c.close
        for c in db.query(Candle)
        .filter(
            Candle.instrument_id == instrument_id,
            Candle.granularity == granularity,
            Candle.price_type == "M",
            Candle.timestamp <= t,
        )
        .order_by(Candle.timestamp.desc())
        .limit(period)
        .all()
    ]
    if len(closes) < period:
        return None
    return sum(closes) / period


def _naive(dt: datetime) -> datetime:
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt


def _minus_months(dt: datetime, months: int) -> datetime:
    """Subtract calendar months, clamping the day (e.g. Mar 31 − 1mo → Feb 28)."""
    month_index = (dt.year * 12 + (dt.month - 1)) - months
    year, month0 = divmod(month_index, 12)
    month = month0 + 1
    day = min(dt.day, _days_in_month(year, month))
    return dt.replace(year=year, month=month, day=day)


def _shift_ref(ref_period: str, months: int) -> str:
    """Shift a 'YYYY-MM-DD' ref_period by ``months`` (usually negative), keeping day
    01 as ref periods are month-anchored."""
    y, m, _d = (int(x) for x in ref_period.split("-"))
    month_index = (y * 12 + (m - 1)) + months
    ny, nm0 = divmod(month_index, 12)
    return f"{ny:04d}-{nm0 + 1:02d}-01"


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        nxt = datetime(year + 1, 1, 1)
    else:
        nxt = datetime(year, month + 1, 1)
    return (nxt - datetime(year, month, 1)).days


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _sub(a: float, b: float) -> float:
    if _isnan(a) or _isnan(b):
        return _NAN
    return a - b


def _isnan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)


def json_safe(obj):
    """Recursively replace NaN/±inf floats with ``None`` so a feature dict is valid
    Postgres JSON.

    ``build_features`` deliberately emits ``float('nan')`` for every unavailable
    numeric feature (XGBoost routes NaN natively — see the module docstring), but
    Python's ``json`` serialises those as the bare tokens ``NaN`` / ``Infinity``,
    which Postgres' JSON parser rejects. Every writer that persists a feature dict
    to a JSON column (the M7 backtest runner, the M8 shadow recorder) funnels
    through this one function so the NaN→NULL convention can never drift between
    the training corpus and the shadow corpus.

    Args:
        obj: any JSON-shaped value (dict / list / tuple / scalar).

    Returns:
        The same structure with every non-finite float replaced by ``None``.
    """
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj
