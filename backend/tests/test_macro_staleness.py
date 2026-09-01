"""Macro staleness watchdog — QA BLOCKER 1(b).

``refresh_macro`` returns 0 rows for a series both when it is already current AND
when its upstream feed has been dead for months. That ambiguity is what let the daily
FRED series (US 2s10s, VIX, WTI) go three weeks stale in silence while
``build_features`` NaN-ed six of the nineteen model-core features on every live
signal — the model kept scoring, on a crippled vector, with nothing failing.

So the watchdog checks the stored DATA, not the insert count, using the EXACT
thresholds ``feature_builder._is_stale`` applies. These tests pin that equivalence:
anything the watchdog calls fresh must be something the feature builder will read,
and vice versa. A watchdog that disagreed with the consumer would be worse than none.

Run from backend/:  python -m pytest tests/test_macro_staleness.py -v
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.domain.macro_series import MACRO_SERIES, MACRO_SERIES_BY_NAME
from app.services import feature_builder as fb
from app.services.fundamental.refresh import (
    SeriesStaleness,
    macro_staleness,
    stale_series,
)


def _with(settings, **overrides):
    return settings.model_copy(update=overrides)


# ── 1. shape: every registered series gets a verdict ─────────────────────────
def test_reports_one_verdict_per_registered_series(db, settings):
    report = macro_staleness(db, settings)

    assert len(report) == len(MACRO_SERIES)
    assert [r.series for r in report] == [s.name for s in MACRO_SERIES]
    assert all(isinstance(r, SeriesStaleness) for r in report)


def test_threshold_matches_the_series_cadence(db, settings):
    """Daily and monthly series must be judged by their own ceiling — a 45-day-old
    monthly print is normal, a 45-day-old VIX is a dead feed."""
    for row in macro_staleness(db, settings):
        expected = (
            settings.FEATURE_STALENESS_DAILY_DAYS
            if MACRO_SERIES_BY_NAME[row.series].cadence == "daily"
            else settings.FEATURE_STALENESS_MONTHLY_DAYS
        )
        assert row.threshold_days == expected


# ── 2. the property that matters: the watchdog agrees with feature_builder ───
def test_watchdog_verdict_matches_feature_builder_staleness_rule(db, settings):
    """The alarm and the consumer must never disagree.

    ``feature_builder._is_stale`` is the function that decides whether a macro value
    becomes a real number or a NaN. If the watchdog used its own thresholds, an
    operator could see 'all fresh' while every derived feature was NaN — exactly the
    silent failure this whole fix exists to end.
    """
    now = datetime.utcnow()
    for row in macro_staleness(db, settings, now=now):
        if row.newest_release is None:
            assert row.stale, "a series with no data at all is unusable by definition"
            continue
        assert row.stale == fb._is_stale(row.series, row.newest_release, now, settings)


# ── 3. sensitivity: the verdict is driven by the threshold, not baked in ─────
def test_an_impossible_threshold_marks_everything_stale(db, settings):
    zero = _with(settings, FEATURE_STALENESS_DAILY_DAYS=0, FEATURE_STALENESS_MONTHLY_DAYS=0)

    report = macro_staleness(db, zero, now=datetime.utcnow() + timedelta(days=1))

    assert all(r.stale for r in report)
    assert len(stale_series(db, zero, now=datetime.utcnow() + timedelta(days=1))) == len(MACRO_SERIES)


def test_a_generous_threshold_clears_every_series_that_has_data(db, settings):
    """Separates "stale" from "absent": with a century-wide window nothing is stale
    unless the series genuinely holds no rows."""
    forever = _with(
        settings, FEATURE_STALENESS_DAILY_DAYS=36500, FEATURE_STALENESS_MONTHLY_DAYS=36500
    )

    still_stale = stale_series(db, forever)

    assert all(r.newest_release is None for r in still_stale)


def test_stale_series_is_the_stale_subset(db, settings):
    report = macro_staleness(db, settings)

    assert stale_series(db, settings) == [r for r in report if r.stale]


# ── 4. the age arithmetic is anchored on the injected clock, not utcnow ──────
def test_age_is_measured_against_the_injected_instant(db, settings):
    baseline = macro_staleness(db, settings, now=datetime.utcnow())
    later = macro_staleness(db, settings, now=datetime.utcnow() + timedelta(days=10))

    for before, after in zip(baseline, later):
        if before.age_days is None:
            assert after.age_days is None
            continue
        assert after.age_days == before.age_days + 10


# ── 5. the regression this was written for ──────────────────────────────────
def test_the_daily_series_behind_the_six_nan_model_features_are_fresh(db, settings):
    """US_10Y_DAILY / US_2Y_DAILY / VIX / WTI feed us_2s10s, vix, vix_change_5d,
    us_real_10y, wti and wti_change_20d. All six were NaN live because these four
    series had aged past FEATURE_STALENESS_DAILY_DAYS. This test fails the moment
    that recurs."""
    stale_now = {r.series for r in stale_series(db, settings)}

    offenders = stale_now & {"US_10Y_DAILY", "US_2Y_DAILY", "VIX", "WTI"}
    assert not offenders, (
        f"daily FRED series are stale again: {sorted(offenders)} — run the "
        "fundamentals refresh; do NOT relax FEATURE_STALENESS_DAILY_DAYS"
    )
