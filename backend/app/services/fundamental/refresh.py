"""
Live macro refresh — shared by the one-shot backfill (`scripts/ingest_macro.py`)
and the scheduled Cron A/B fundamentals jobs.

Idempotent: same `ON CONFLICT (series, ref_period, release_time) DO NOTHING` as the
backfill, so a future revision lands as a NEW release_time row (bitemporal),
never an in-place overwrite — that is the point-in-time leakage firewall. Per-series
isolation: one bad series never aborts the run (returns -1 in the summary so the
caller's staleness check can alert on it).

Two release_time paths, branched per series on `MacroSeries.revised`:
  - revised=False (yields/policy/VIX): SYNTHETIC release_time = ref_period +
    publish_lag_days (these series are not revised; output_type=1 latest value).
  - revised=True (CPI/unemployment/retail-sales): FULL BITEMPORAL — EVERY vintage of
    each observation from FRED output_type=1 over the full realtime window, each row's
    TRUE release_time = realtime_start (first release + one row per later revision).
Both paths share the same on_conflict_do_nothing idempotency and per-series
try/except isolation (a bad series returns -1, never aborts the run).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.config import Settings
from app.domain.macro_series import MACRO_SERIES, MacroSeries
from app.models.macro_data import MacroData
from app.services.fundamental.factory import get_fundamental_data

# Cadence labels emitted by ``MacroSeries.cadence`` → the Settings attribute holding
# that cadence's staleness ceiling. Named mapping so the watchdog uses EXACTLY the
# thresholds feature_builder._is_stale uses — a series the watchdog calls fresh can
# never be one the feature builder silently NaNs out.
_CADENCE_THRESHOLD_SETTING: dict[str, str] = {
    "daily": "FEATURE_STALENESS_DAILY_DAYS",
    "monthly": "FEATURE_STALENESS_MONTHLY_DAYS",
}


def _synthetic_lag_rows(provider, s: MacroSeries, obs_start: str) -> list[dict]:
    """
    NON-revised path (yields/policy/VIX). One row per (series, ref_period) with a
    SYNTHETIC release_time = ref_period + publish_lag_days. Future-dated synthetic
    rows are skipped (the publish estimate hasn't arrived); release_time stays
    deterministic so a re-run is idempotent and the row lands once today passes it.
    """
    obs = provider.get_series(s.fred_id, observation_start=obs_start)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    rows = [
        {
            "series": s.name,
            "ref_period": o.ref_period,
            "release_time": datetime.strptime(o.ref_period, "%Y-%m-%d")
            + timedelta(days=s.publish_lag_days),
            "value": o.value,
        }
        for o in obs
    ]
    return [r for r in rows if r["release_time"] <= now]


def _all_vintage_rows(provider, s: MacroSeries, obs_start: str) -> list[dict]:
    """
    REVISED path (CPI/unemployment/retail-sales/HICP), FULL BITEMPORAL. MULTIPLE rows
    per (series, ref_period) — one per vintage — using the TRUE publication date (FRED
    output_type=1 realtime_start) as release_time and that vintage's value. The
    earliest release_time per ref_period is the first release; each later row is a
    revision. No synthetic lag, so no future-date filter is needed: a realtime_start
    is by definition a date the value actually went public. UNIQUE(series, ref_period,
    release_time) + on_conflict_do_nothing makes storing all vintages idempotent.
    """
    vintages = provider.get_series_all_vintages(s.fred_id, observation_start=obs_start)
    return [
        {
            "series": s.name,
            "ref_period": v.ref_period,
            "release_time": v.release_time,
            "value": v.value,
        }
        for v in vintages
    ]


def refresh_macro(db: Session, settings: Settings, lookback_days: int) -> dict[str, int]:
    """
    Upsert macro_data for every registered series over [now - lookback_days, now].
    Branches per series on `MacroSeries.revised`:
      - False → synthetic-lag release_time (market data, output_type=1)
      - True  → FULL BITEMPORAL vintages (revised indicators, output_type=1 over the
        full realtime window — every vintage, TRUE release_time per row)
    Returns {series_name: rows_inserted | 0 | -1(error)}.
    """
    provider = get_fundamental_data(settings)
    default_start = (
        datetime.now(timezone.utc) - timedelta(days=lookback_days)
    ).strftime("%Y-%m-%d")
    summary: dict[str, int] = {}

    for s in MACRO_SERIES:
        obs_start = s.obs_start_override or default_start
        try:
            rows = (
                _all_vintage_rows(provider, s, obs_start)
                if s.revised
                else _synthetic_lag_rows(provider, s, obs_start)
            )
        except Exception:  # noqa: BLE001 — isolate one series' failure; flag via -1
            db.rollback()
            summary[s.name] = -1
            continue

        if not rows:
            summary[s.name] = 0
            continue

        stmt = pg_insert(MacroData).values(rows).on_conflict_do_nothing()
        summary[s.name] = db.execute(stmt).rowcount
        db.commit()

    return summary


# ── staleness watchdog ───────────────────────────────────────────────────────
@dataclass(frozen=True)
class SeriesStaleness:
    """Freshness verdict for ONE registered macro series.

    Attributes:
        series: the internal ``macro_data.series`` name.
        cadence: ``"daily"`` or ``"monthly"``, derived from the registry entry.
        newest_release: newest ``release_time`` stored for the series, or ``None``
            when the series has no rows at all (never ingested / provider failure).
        age_days: whole days between ``newest_release`` and the evaluation instant,
            or ``None`` when the series is empty.
        threshold_days: the cadence's staleness ceiling from Settings.
        stale: True when the series is empty or ``age_days > threshold_days`` — i.e.
            exactly the condition under which ``feature_builder`` returns NaN for
            every feature derived from it.
    """

    series: str
    cadence: str
    newest_release: datetime | None
    age_days: int | None
    threshold_days: int
    stale: bool


def macro_staleness(
    db: Session, settings: Settings, now: datetime | None = None
) -> list[SeriesStaleness]:
    """Freshness of every registered macro series, one verdict per series.

    ``refresh_macro`` returning 0 for a series is ambiguous — it means "nothing NEW
    landed", which is the correct outcome both for a series that is already current
    and for one whose upstream feed died months ago. This function removes that
    ambiguity by asking the only question that matters downstream: *is the newest
    stored release recent enough that ``feature_builder`` will still use it?*

    The thresholds are read from the SAME Settings fields
    (``FEATURE_STALENESS_DAILY_DAYS`` / ``FEATURE_STALENESS_MONTHLY_DAYS``) that
    ``feature_builder._is_stale`` applies, so a series reported fresh here cannot be
    one the feature builder is quietly NaN-ing out.

    Args:
        db: SQLAlchemy session.
        settings: config supplying the per-cadence staleness ceilings.
        now: evaluation instant (naive UTC); defaults to ``datetime.utcnow()``.
            Injectable so a caller or test can reason about staleness without
            patching the clock.

    Returns:
        One :class:`SeriesStaleness` per entry in the ``MACRO_SERIES`` registry, in
        registry order. Series with no stored rows are included and marked stale.
    """
    evaluated_at = now or datetime.now(timezone.utc).replace(tzinfo=None)
    if evaluated_at.tzinfo is not None:
        evaluated_at = evaluated_at.replace(tzinfo=None)
    newest_by_series = dict(
        db.query(MacroData.series, func.max(MacroData.release_time))
        .group_by(MacroData.series)
        .all()
    )
    report: list[SeriesStaleness] = []
    for s in MACRO_SERIES:
        threshold = int(getattr(settings, _CADENCE_THRESHOLD_SETTING[s.cadence]))
        newest = newest_by_series.get(s.name)
        if newest is not None and newest.tzinfo is not None:
            newest = newest.replace(tzinfo=None)
        age = None if newest is None else (evaluated_at - newest).days
        report.append(
            SeriesStaleness(
                series=s.name,
                cadence=s.cadence,
                newest_release=newest,
                age_days=age,
                threshold_days=threshold,
                stale=(age is None or age > threshold),
            )
        )
    return report


def stale_series(
    db: Session, settings: Settings, now: datetime | None = None
) -> list[SeriesStaleness]:
    """The subset of :func:`macro_staleness` that is actually stale (the alarm set)."""
    return [r for r in macro_staleness(db, settings, now) if r.stale]
