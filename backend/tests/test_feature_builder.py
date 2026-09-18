"""
Integration tests for app.services.feature_builder — the leakage firewall.

Run from backend/:  python -m pytest tests/test_feature_builder.py -v
Reads the live dev Postgres (docker) via app.database.SessionLocal.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

from sqlalchemy import and_, func

from app.models.indicator import Indicator
from app.models.macro_data import MacroData
from app.models.news_calendar_event import NewsCalendarEvent
from app.services import feature_builder as fb
from app.services.feature_builder import (
    FEATURE_KEYS_GATED,
    FEATURE_KEYS_MODEL,
    FEATURE_SCHEMA_VERSION,
    PAYLOAD_KEYS,
    build_features,
    causal_swing_levels,
)

_NEWS_KEYS = [
    "news_high_impact_next_4h",
    "news_high_impact_next_8h",
    "news_high_impact_last_4h",
    "news_high_impact_last_8h",
]

_GRAN = "H4"
_T_MID_2025 = datetime(2025, 7, 1, 12, 0, 0)
_CONFLUENCE = {"trend": True, "rsi": True, "structure": False, "session": True, "spread": True}


def _isnan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)


# ── Test 1 — contract ────────────────────────────────────────────────────────
def test_contract_keys_exact(db, settings, instrument):
    inst = instrument("EUR_USD")
    feats = build_features(inst, _T_MID_2025, _GRAN, _CONFLUENCE, db, settings)

    expected = set(FEATURE_KEYS_MODEL + FEATURE_KEYS_GATED + PAYLOAD_KEYS) | {"feature_schema_version"}
    assert set(feats.keys()) == expected, set(feats.keys()) ^ expected
    assert feats["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    # No KeyError even when macro rows are absent: every declared key exists.
    for k in FEATURE_KEYS_MODEL:
        assert k in feats


def test_contract_v3_exact_tiers():
    """v3 tier membership + counts are locked: 19 model / 10 gated / 9 payload.

    v3 carries the SAME 19 model keys as v2 — the bump was about MEANING, not names.
    ``c3_structure`` moved from centred swing pivots (unknowable at their own timestamp,
    so a look-ahead in backtest and never populated live) to a trailing Donchian channel,
    and the live indicator writer was fixed so ``swing_*`` is written at all. Both change
    what ``c3_structure`` and ``swing_dist_atr`` VALUE while every key name stays put —
    which is exactly the case a schema version exists to catch, since nothing downstream
    compares values.

    Research Cycle 2 (v2) promoted us_real_10y + wti + wti_change_20d INTO the model tier
    and dropped the four news_high_impact_* booleans DOWN to the gated tier.
    """
    assert FEATURE_SCHEMA_VERSION == 3
    assert len(FEATURE_KEYS_MODEL) == 19, FEATURE_KEYS_MODEL
    assert len(FEATURE_KEYS_GATED) == 10, FEATURE_KEYS_GATED
    assert len(PAYLOAD_KEYS) == 9, PAYLOAD_KEYS

    assert set(FEATURE_KEYS_MODEL) == {
        "confluence_score", "rsi14", "atr_pct", "dist_to_sma50_atr", "swing_dist_atr",
        "yield_differential_10y", "yield_differential_change_1m", "us_2s10s",
        "policy_rate_differential", "vix", "vix_change_5d", "us_real_10y", "wti",
        "wti_change_20d", "us_core_cpi_yoy", "us_cpi_yoy_change", "us_unemployment",
        "instrument_category", "session",
    }
    assert set(FEATURE_KEYS_GATED) == {
        "us_10y", "eu_cpi_yoy", "yield_differential_change_3m", "us_cpi_yoy",
        "us_unemployment_change", "us_retail_sales_mom",
        *_NEWS_KEYS,
    }
    # Tiers are disjoint and the moved keys landed on the correct side.
    assert set(FEATURE_KEYS_MODEL).isdisjoint(FEATURE_KEYS_GATED)
    for k in ("us_real_10y", "wti", "wti_change_20d"):
        assert k in FEATURE_KEYS_MODEL and k not in FEATURE_KEYS_GATED
    for k in _NEWS_KEYS:
        assert k in FEATURE_KEYS_GATED and k not in FEATURE_KEYS_MODEL


def test_contract_no_keyerror_far_past(db, settings, instrument):
    # A T before macro/news history exists must still return the full dict (NaNs).
    inst = instrument("EUR_USD")
    feats = build_features(inst, datetime(2020, 1, 2, 0, 0), _GRAN, _CONFLUENCE, db, settings)
    assert "yield_differential_10y" in feats
    assert _isnan(feats["vix"])  # no VIX visible that far back


# ── Test 2 — PIT vintage selection (revisions) ───────────────────────────────
def test_pit_vintage_us_cpi_2022_01(db, settings):
    ref = "2022-01-01"
    first = fb._value_for_ref(db, "US_CPI", ref, datetime(2022, 6, 1))
    revised = fb._value_for_ref(db, "US_CPI", ref, datetime(2024, 6, 1))
    assert abs(first - 281.933) < 1e-6, first
    assert abs(revised - 282.39) < 1e-6, revised
    assert first != revised


# ── Test 3 — staleness floor ─────────────────────────────────────────────────
def test_staleness_policy_frozen_chf_nan(db, settings, instrument):
    t = datetime(2025, 7, 1, 12, 0)
    chf = build_features(instrument("USD_CHF"), t, _GRAN, _CONFLUENCE, db, settings)
    eur = build_features(instrument("EUR_USD"), t, _GRAN, _CONFLUENCE, db, settings)
    assert _isnan(chf["policy_rate_differential"]), chf["policy_rate_differential"]
    assert not _isnan(eur["policy_rate_differential"]), eur["policy_rate_differential"]


# ── Test 4 — LEAK-1 causal swings ────────────────────────────────────────────
def test_leak1_causal_swing_cutoff(db, settings, instrument):
    inst = instrument("EUR_USD")
    t = _T_MID_2025
    k = settings.SWING_LOOKBACK_PERIODS

    rows_desc = (
        db.query(Indicator)
        .filter(and_(Indicator.instrument_id == inst.id, Indicator.granularity == _GRAN,
                     Indicator.timestamp <= t))
        .order_by(Indicator.timestamp.desc())
        .all()
    )
    assert len(rows_desc) > k

    levels = causal_swing_levels(db, inst.id, _GRAN, t, k)
    cutoff = levels["cutoff_timestamp"]
    # The cutoff is exactly the (k+1)-th newest row — the newest k rows are dropped
    # because their centered-window swing could only be confirmed by post-T bars.
    assert cutoff == rows_desc[k].timestamp
    # Nothing newer than the cutoff may be consulted.
    newer = [r for r in rows_desc if r.timestamp > cutoff]
    assert len(newer) == k
    assert cutoff <= t


# ── Test 5 — news windows ────────────────────────────────────────────────────
def _isolated_usd_event(db) -> datetime:
    """Find a USD high-impact, non-all_day event with no other USD high-impact
    event within +/-8h, so the window assertions are unambiguous. It must also sit
    at least 8h AFTER the calendar coverage floor — otherwise the test's -5h/-8h
    look-back signal_times predate coverage and the v2 gate NaNs the booleans."""
    coverage_start = db.query(func.min(NewsCalendarEvent.timestamp)).scalar()
    usd = (
        db.query(NewsCalendarEvent)
        .filter(NewsCalendarEvent.currency == "USD",
                NewsCalendarEvent.impact == "high",
                NewsCalendarEvent.all_day.is_(False))
        .order_by(NewsCalendarEvent.timestamp)
        .all()
    )
    times = [e.timestamp for e in usd]
    for i, ts in enumerate(times):
        if coverage_start is not None and (ts - coverage_start) < timedelta(hours=8):
            continue  # too close to the coverage floor for a clean -8h look-back
        near = [o for o in times if o != ts and abs((o - ts).total_seconds()) <= 8 * 3600]
        if not near:
            return ts
    raise AssertionError("no isolated USD high-impact event found")


def test_news_windows(db, settings, instrument):
    e = _isolated_usd_event(db)
    inst = instrument("EUR_USD")  # USD leg present

    two_before = build_features(inst, e - timedelta(hours=2), _GRAN, _CONFLUENCE, db, settings)
    assert two_before["news_high_impact_next_4h"] is True
    assert two_before["news_high_impact_last_4h"] is False

    five_before = build_features(inst, e - timedelta(hours=5), _GRAN, _CONFLUENCE, db, settings)
    assert five_before["news_high_impact_next_4h"] is False
    assert five_before["news_high_impact_next_8h"] is True

    two_after = build_features(inst, e + timedelta(hours=2), _GRAN, _CONFLUENCE, db, settings)
    assert two_after["news_high_impact_last_4h"] is True


# ── Test — news coverage floor (v2) ─────────────────────────────────────────
def test_news_features_nan_before_coverage(db, settings, instrument):
    """Before the earliest calendar event, the four news booleans must be NaN
    (unknown), NOT False — a pre-coverage False would be a dishonest 'no event'.
    Within coverage they are genuine booleans."""
    coverage_start = db.query(func.min(NewsCalendarEvent.timestamp)).scalar()
    assert coverage_start is not None, "news calendar empty — cannot exercise coverage gate"
    inst = instrument("EUR_USD")

    before = build_features(
        inst, coverage_start - timedelta(days=30), _GRAN, _CONFLUENCE, db, settings
    )
    for k in _NEWS_KEYS:
        assert _isnan(before[k]), (k, before[k])

    after = build_features(inst, _T_MID_2025, _GRAN, _CONFLUENCE, db, settings)
    assert _T_MID_2025 > coverage_start
    for k in _NEWS_KEYS:
        assert isinstance(after[k], bool), (k, after[k])


# ── Test — WTI PIT level + change (v2) ──────────────────────────────────────
def _pit_macro_value(db, series: str, t: datetime) -> float:
    """Replicate the PIT selection directly from macro_data (independent of the
    feature_builder helper): visible rows (release_time <= T) → latest ref_period →
    newest vintage within it. WTI is daily/non-revised so this is unambiguous."""
    rows = (
        db.query(MacroData)
        .filter(MacroData.series == series, MacroData.release_time <= t)
        .all()
    )
    assert rows, f"no visible {series} rows at {t}"
    latest_ref = max(r.ref_period for r in rows)
    return max((r for r in rows if r.ref_period == latest_ref), key=lambda r: r.release_time).value


def test_wti_pit_level_and_change(db, settings, instrument):
    inst = instrument("EUR_USD")
    feats = build_features(inst, _T_MID_2025, _GRAN, _CONFLUENCE, db, settings)

    exp_now = _pit_macro_value(db, "WTI", _T_MID_2025)
    exp_past = _pit_macro_value(
        db, "WTI", _T_MID_2025 - timedelta(days=settings.WTI_CHANGE_DAYS)
    )
    assert feats["wti"] == exp_now, (feats["wti"], exp_now)
    assert abs(feats["wti_change_20d"] - (exp_now - exp_past)) < 1e-9, (
        feats["wti_change_20d"], exp_now - exp_past,
    )
    assert feats["wti"] > 0.0  # crude price is positive


# ── Test — us_real_10y unit fix (QA FIX 1) ──────────────────────────────────
def test_us_real_10y_unit_consistent(db, settings, instrument):
    """us_10y is PERCENT (e.g. 4.24); us_cpi_yoy is a FRACTION (e.g. 0.0238).
    us_real_10y must subtract on the SAME scale (us_10y - us_cpi_yoy*100), not the
    raw fraction — else the result is inflated by ~us_cpi_yoy*99 (~4.2 vs true ~1.9)."""
    inst = instrument("EUR_USD")
    feats = build_features(inst, _T_MID_2025, _GRAN, _CONFLUENCE, db, settings)

    assert not _isnan(feats["us_real_10y"])
    assert -5.0 <= feats["us_real_10y"] <= 10.0, feats["us_real_10y"]
    expected = feats["us_10y"] - feats["us_cpi_yoy"] * 100.0
    assert abs(feats["us_real_10y"] - expected) < 1e-9, (feats["us_real_10y"], expected)
    # Regression guard: the old (buggy) fraction-scale subtraction must NOT match.
    buggy = feats["us_10y"] - feats["us_cpi_yoy"]
    assert abs(feats["us_real_10y"] - buggy) > 1.0


# ── Test 6 — cross-pair sanity ───────────────────────────────────────────────
def test_cross_pair_yield_and_category(db, settings, instrument):
    t = _T_MID_2025
    eur_usd = build_features(instrument("EUR_USD"), t, _GRAN, _CONFLUENCE, db, settings)
    usd_jpy = build_features(instrument("USD_JPY"), t, _GRAN, _CONFLUENCE, db, settings)
    eur_gbp = build_features(instrument("EUR_GBP"), t, _GRAN, _CONFLUENCE, db, settings)

    assert eur_usd["yield_differential_10y"] != usd_jpy["yield_differential_10y"]
    assert eur_usd["instrument_category"] == "major_usd"
    assert usd_jpy["instrument_category"] == "jpy_cross"
    assert eur_gbp["instrument_category"] == "eur_cross"


# ── Smoke — full dict, sane values ───────────────────────────────────────────
def test_smoke_full_dict(db, settings, instrument):
    feats = build_features(instrument("EUR_USD"), _T_MID_2025, _GRAN, _CONFLUENCE, db, settings)
    import json

    print("\n=== EUR_USD @", _T_MID_2025, "===")
    print(json.dumps(feats, indent=2, default=str))

    assert 0.0 <= feats["rsi14"] <= 100.0
    assert 0.0 < feats["atr_pct"] < 0.05  # FX ATR% is small
    assert abs(feats["yield_differential_10y"]) < 15.0
    assert feats["vix"] > 0.0


# ── Phase A leakage firewall: excursion data must never become a feature ─────
def test_excursion_fields_are_never_model_features():
    """MFE/MAE and path columns are LABELS, not features — enforced, not trusted.

    Every value in ``trade_paths`` and in ``trades.mfe_r``/``mae_r`` is derived
    from prices AFTER the signal timestamp. Feeding any of them to the model as an
    input is the textbook leakage failure: it scores near-perfectly in backtest
    and is worthless live, because at signal time the column does not exist yet.

    They ARE legitimate as prediction targets (train a model to predict how far a
    trade will run) and as strictly point-in-time aggregates over trades that had
    already CLOSED before the signal being scored. Neither of those routes goes
    through FEATURE_KEYS_MODEL, so this assertion costs nothing it should not.

    Guards the whole namespace rather than a fixed list, so a future column named
    e.g. ``mfe_at_bar_5`` is caught the day it is added.
    """
    from app.models.trade_path import TradePath

    banned_exact = {"mfe_r", "mae_r", "path_truncated", "beyond_exit", "degraded"}
    banned_exact |= {c.name for c in TradePath.__table__.columns}
    banned_prefixes = ("mfe", "mae", "path_", "excursion")

    for key in FEATURE_KEYS_MODEL:
        assert key not in banned_exact, (
            f"{key!r} is post-signal excursion data and must never be a model feature"
        )
        assert not key.startswith(banned_prefixes), (
            f"{key!r} looks like excursion data (post-signal) and must not be a model feature"
        )
